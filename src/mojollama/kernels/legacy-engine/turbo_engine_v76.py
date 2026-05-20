#!/usr/bin/env python3
"""TurboEngine v7.6 — Fused OMP batch_matmul for ALL per-layer projections.

Fuses Q+K+V+Gate+Up (5 projections sharing x_norm input) into a SINGLE
batch_matmul OMP call — eliminates 4 OMP fork/join overheads per layer.

Total OMP regions per layer: 3 (fused5 + Oproj + Down)
vs v7: 4 (QKV + Oproj + GateUp + Down)
vs v55: 7 (indiv Q,K,V,O,gate,up,down)

Expected: ~15% overall = 78 → 90 tok/s
"""

import numpy as np, ctypes, os, gguf
from gguf.constants import GGMLQuantizationType as QT

GGML_F32=0; GGML_F16=1; GGML_Q4_0=2; GGML_Q4_1=3; GGML_Q8_0=8
GGML_Q4_K=12; GGML_Q5_K=13; GGML_Q6_K=14
BLOCK_SIZES = {2:18,3:20,12:144,13:176,14:210,8:34}
C_KERNEL_TYPES = {GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0}
AVX2_TYPES = {GGML_Q4_0, GGML_Q4_1, GGML_Q8_0}

MAX_PROJ = 16  # room for all projections

class TurboEngineV76:
    """v7.6 — Fused batch_matmul for all x_norm projections (Q,K,V,Gate,Up)."""
    
    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        self.n_threads = n_threads
        self.reader = gguf.GGUFReader(model_path)
        self._parse_metadata()
        self._load_weights()
        self._load_kernels()
        self._preload_layer_weights()
        self._init_kv_cache()
        self.reset()
        moe_str = f'/MoE-{self.n_experts}x{self.n_experts_per_tok}' if self.is_moe else ''
        print(f"TurboEngine v7.6: {self.n_layers}L/{self.n_embd}D/"
              f"{self.n_ff}FF/{self.n_head}H/{self.n_kv_head}KV"
              f"{moe_str} | t={self.n_threads} | vocab={self.vocab_size}")

    def _parse_metadata(self):
        fields = self.reader.fields
        def _get(name):
            for key, val in fields.items():
                if key == name:
                    parts = val.parts if hasattr(val, 'parts') else []
                    if len(parts) >= 1:
                        data = parts[-1]
                        if hasattr(data, '__iter__') and len(data) == 1: return int(data[0])
                        return data
        arch = 'llama'
        for p in ['qwen3moe','qwen2moe','llama','mistral']:
            for k in fields:
                if f'{p}.block_count' in k: arch = p; break
        self.n_layers = int(_get(f'{arch}.block_count') or 16)
        self.n_embd = int(_get(f'{arch}.embedding_length') or 2048)
        self.n_ff = int(_get(f'{arch}.feed_forward_length') or self.n_embd * 4)
        self.n_head = int(_get(f'{arch}.attention.head_count') or 32)
        self.n_kv_head = int(_get(f'{arch}.attention.head_count_kv') or self.n_head)
        self.head_dim = int(_get(f'{arch}.attention.key_length') or (self.n_embd // self.n_head))
        self.rope_freq_base = float(_get(f'{arch}.rope.freq_base') or 10000.0)
        self.eps = float(_get(f'{arch}.attention.layer_norm_rms_epsilon') or 1e-6)
        self.vocab_size = int(_get(f'{arch}.vocab_size') or 0)
        self.n_experts = _get(f'{arch}.expert_count')
        self.n_experts_per_tok = _get(f'{arch}.expert_used_count')
        self.is_moe = self.n_experts is not None
        if not self.is_moe: self.n_experts = 1; self.n_experts_per_tok = 1
        else: self.n_experts = int(self.n_experts); self.n_experts_per_tok = int(self.n_experts_per_tok)
        self.arch_prefix = arch

    def _load_weights(self):
        self.weights = {}; self.raw_weights = {}; self.weight_qtypes = {}; self.weight_info = {}
        for t in self.reader.tensors:
            name = t.name; qtype = QT(t.tensor_type).value
            self.weight_qtypes[name] = qtype
            if len(t.shape) == 2:
                in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(out_dim, in_dim)
                self.weights[name] = np.ascontiguousarray(f32)
                if qtype in C_KERNEL_TYPES:
                    raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8).copy()
                    self.raw_weights[name] = raw
                    self.weight_info[name] = (out_dim, in_dim, BLOCK_SIZES[qtype], qtype)
            elif len(t.shape) == 3:
                in_dim,out_dim,n_exp = int(t.shape[0]),int(t.shape[1]),int(t.shape[2])
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(n_exp,out_dim,in_dim)
                self.weights[name] = np.ascontiguousarray(f32)
            elif len(t.shape) == 1:
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
                self.weights[name] = np.ascontiguousarray(f32)
        self.out_w_name = 'output.weight' if 'output.weight' in self.weights else 'token_embd.weight'
        emb = self.weights['token_embd.weight']
        if emb.ndim == 2 and emb.shape[1] != self.n_embd:
            emb = np.ascontiguousarray(emb.T)
            self.weights['token_embd.weight'] = emb
        if self.vocab_size == 0: self.vocab_size = emb.shape[0]
        self.emb = emb

    def _load_kernels(self):
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        cf = ctypes.POINTER(ctypes.c_float); cu = ctypes.POINTER(ctypes.c_uint8); ci = ctypes.c_int

        # OMP kernels
        omp_path = os.path.join(kernel_dir, 'quant_kernels_omp.so')
        omp = ctypes.CDLL(omp_path)
        omp.quant_matmul_omp.argtypes = [cu, cf, cf, ci, ci, ci]; omp.quant_matmul_omp.restype = None
        omp.batch_matmul.argtypes = [cu*MAX_PROJ, cf, cf*MAX_PROJ, ci*MAX_PROJ, ci*MAX_PROJ, ci*MAX_PROJ, ci]
        omp.batch_matmul.restype = None
        omp.set_num_threads.argtypes = [ci]; omp.set_num_threads.restype = None
        omp.set_num_threads(self.n_threads)
        self._kern = omp

        # SIMD ops
        simd_path = os.path.join(kernel_dir, 'simd_ops.so')
        simd = ctypes.CDLL(simd_path)
        simd.rms_norm.argtypes = [cf, cf, cf, ci, ctypes.c_float]; simd.rms_norm.restype = None
        simd.silu.argtypes = [cf, cf, ci]; simd.silu.restype = None
        self._simd = simd

    def _preload_layer_weights(self):
        self._layers = []
        for i in range(self.n_layers):
            pfx = f'blk.{i}'
            proj_names = ['attn_q','attn_k','attn_v','attn_output','ffn_gate','ffn_up','ffn_down']
            lw = {}
            for attr in proj_names:
                name = f'{pfx}.{attr}.weight'
                info = self.weight_info.get(name)
                if info is not None:
                    raw = self.raw_weights[name]
                    lw[attr] = {
                        'raw': raw.ctypes.data_as(cu),
                        'nr': ci(info[0]), 'nc': ci(info[1]),
                        'qt': ci(info[3]), 'bs': ci(BLOCK_SIZES[info[3]]),
                    }
                else:
                    lw[attr] = {'f32': self.weights[name], 'use_f32': True}
            lw['attn_norm_w'] = self.weights[f'{pfx}.attn_norm.weight']
            lw['ffn_norm_w'] = self.weights[f'{pfx}.ffn_norm.weight']
            self._layers.append(lw)

        # Output projection — optimize K-quants to Q8_0
        out_info = self.weight_info.get(self.out_w_name)
        if out_info is not None:
            self._out_nr = ci(out_info[0]); self._out_nc = ci(out_info[1])
            if out_info[3] in (6,14,12,10):
                self._out_raw = self._requantize_output(self.raw_weights[self.out_w_name], out_info)
                self._out_qt = ci(8)
            else:
                self._out_raw = self.raw_weights[self.out_w_name].ctypes.data_as(cu)
                self._out_qt = ci(out_info[3])
        else:
            self._out_f32 = self.weights[self.out_w_name]
            self._out_use_c = False
        self._out_norm_w = self.weights['output_norm.weight']

    def _requantize_output(self, raw_weights, out_info):
        n_rows, n_cols = out_info[0], out_info[1]
        n_blocks = n_cols // 32; orig_qt = out_info[3]
        f32 = np.zeros((n_rows, n_cols), dtype=np.float32)
        if orig_qt == 14:
            n_blk = n_cols // 256
            self._kern.q6_k_dequantize_row.argtypes = [cu, cf, ci]
            self._kern.q6_k_dequantize_row.restype = None
            raw_ptr = raw_weights.ctypes.data_as(cu)
            base = ctypes.addressof(raw_ptr.contents)
            for r in range(n_rows):
                ptr = ctypes.cast(base + r * n_blk * 210, cu)
                self._kern.q6_k_dequantize_row(ptr, f32[r].ctypes.data_as(cf), ci(n_cols))
        blocks = f32.reshape(n_rows, n_blocks, 32)
        amax = np.max(np.abs(blocks), axis=2, keepdims=True)
        amax = np.clip(amax, 1e-10, None); d = amax / 127.0
        qs = np.clip(np.round(blocks / d), -127, 127).astype(np.int8)
        d_f16 = d[:,:,0].astype(np.float16); d_u16 = d_f16.view(np.uint16)
        q8 = np.zeros((n_rows, n_blocks, 34), dtype=np.uint8)
        q8[:,:,0] = d_u16 & 0xFF; q8[:,:,1] = (d_u16>>8)&0xFF; q8[:,:,2:] = qs.view(np.uint8)
        q8_flat = q8.reshape(n_rows*n_blocks*34)
        self._out_q8_buf = np.ascontiguousarray(q8_flat, dtype=np.uint8)
        print(f"  Output: {n_rows}×{n_cols} → Q8_0 ({len(self._out_q8_buf)/1024/1024:.0f} MB)")
        return self._out_q8_buf.ctypes.data_as(cu)

    def _init_kv_cache(self):
        N=self.n_embd; NKH=self.n_kv_head; HD=self.head_dim; P=4096
        self.kv_k=np.zeros((self.n_layers,P,NKH*HD),dtype=np.float32)
        self.kv_v=np.zeros((self.n_layers,P,NKH*HD),dtype=np.float32)
        self.kv_len=np.zeros(self.n_layers,dtype=np.int32)
        NK=self.n_head*HD; FF=self.n_ff
        self._x=np.zeros(N,dtype=np.float32)
        self._xn=np.zeros(N,dtype=np.float32)
        self._res=np.zeros(N,dtype=np.float32)
        self._q=np.zeros(NK,dtype=np.float32)
        self._k=np.zeros(NKH*HD,dtype=np.float32)
        self._v=np.zeros(NKH*HD,dtype=np.float32)
        self._att=np.zeros(NK,dtype=np.float32)
        self._gate=np.zeros(FF,dtype=np.float32)
        self._up=np.zeros(FF,dtype=np.float32)
        self._silu=np.zeros(FF,dtype=np.float32)
        self._oproj=np.zeros(N,dtype=np.float32)
        self._ffn=np.zeros(N,dtype=np.float32)
        self._logits=np.zeros(self.vocab_size,dtype=np.float32)
        self._px=self._x.ctypes.data_as(cf); self._pxn=self._xn.ctypes.data_as(cf)
        self._pr=self._res.ctypes.data_as(cf); self._pq=self._q.ctypes.data_as(cf)
        self._pk=self._k.ctypes.data_as(cf); self._pv=self._v.ctypes.data_as(cf)
        self._patt=self._att.ctypes.data_as(cf); self._pg=self._gate.ctypes.data_as(cf)
        self._pu=self._up.ctypes.data_as(cf); self._ps=self._silu.ctypes.data_as(cf)
        self._po=self._oproj.ctypes.data_as(cf); self._pf=self._ffn.ctypes.data_as(cf)
        self._pl=self._logits.ctypes.data_as(cf)
        self._eps_f=ctypes.c_float(self.eps); self._gqa_rep=self.n_head//self.n_kv_head
        HD=self.head_dim
        freq=self.rope_freq_base**(np.arange(0,HD,2,dtype=np.float32)/HD)
        self._rc={}; self._rs={}
        
    def reset(self): self.kv_len[:]=0; self.pos=0

    def _rope(self, x, pos, nh):
        hd=self.head_dim; h=hd//2
        if pos not in self._rc:
            freq=self.rope_freq_base**(np.arange(0,hd,2,dtype=np.float32)/hd)
            a=pos/freq; self._rc[pos]=np.cos(a).astype(np.float32); self._rs[pos]=np.sin(a).astype(np.float32)
        c=self._rc[pos]; s=self._rs[pos]
        x2=x.reshape(nh,hd); o=x2.copy()
        o[:,:h]=x2[:,:h]*c-x2[:,h:]*s; o[:,h:]=x2[:,h:]*c+x2[:,:h]*s
        return o.reshape(-1)

    def forward(self, token_id):
        b_x=self._x; b_xn=self._xn; b_r=self._res; b_q=self._q; b_k=self._k; b_v=self._v
        b_a=self._att; b_g=self._gate; b_u=self._up; b_s=self._silu
        b_o=self._oproj; b_f=self._ffn
        N=self.n_embd; NH=self.n_head; NKH=self.n_kv_head; HD=self.head_dim
        FF=self.n_ff; L=self.n_layers
        p_x=self._px; p_xn=self._pxn; p_r=self._pr
        p_q=self._pq; p_k=self._pk; p_v=self._pv
        p_a=self._patt; p_g=self._pg; p_u=self._pu
        p_s=self._ps; p_o=self._po; p_f=self._pf

        np.copyto(b_x, self.emb[token_id])

        for i in range(L):
            lw = self._layers[i]
            cu_arr = cu*MAX_PROJ; cf_arr = cf*MAX_PROJ; ci_arr = ci*MAX_PROJ

            # Residual
            np.copyto(b_r, b_x)

            # RMS norm 1
            self._simd.rms_norm(p_xn, p_x,
                lw['attn_norm_w'].ctypes.data_as(cf), N, self._eps_f)

            # ── FUSED: Q + K + V + Gate + Up in ONE OMP region ──
            projs_5 = ['attn_q','attn_k','attn_v','ffn_gate','ffn_up']
            w_ptrs = (cu_arr)(*[lw[p]['raw'] for p in projs_5])
            out_ptrs_5 = (cf_arr)(p_q, p_k, p_v, p_g, p_u)
            nrows_5 = (ci_arr)(*[lw[p]['nr'] for p in projs_5])
            ncols_5 = (ci_arr)(*[lw[p]['nc'] for p in projs_5])
            qts_5 = (ci_arr)(*[lw[p]['qt'] for p in projs_5])
            self._kern.batch_matmul(w_ptrs, p_xn, out_ptrs_5, nrows_5, ncols_5, qts_5, 5)

            # RoPE
            b_q[:] = self._rope(b_q, self.pos, NH)
            b_k[:] = self._rope(b_k, self.pos, NKH)

            # KV cache store
            self.kv_k[i,self.kv_len[i],:NKH*HD] = b_k[:NKH*HD]
            self.kv_v[i,self.kv_len[i],:NKH*HD] = b_v[:NKH*HD]

            # Attention (numpy einsum)
            sl = self.kv_len[i]+1
            kc = self.kv_k[i,:sl].reshape(sl,NKH,HD)
            vc = self.kv_v[i,:sl].reshape(sl,NKH,HD)
            q2 = b_q.reshape(NH,HD)
            if self._gqa_rep>1:
                ke = np.repeat(kc,self._gqa_rep,axis=1); ve = np.repeat(vc,self._gqa_rep,axis=1)
            else:
                ke=kc; ve=vc
            sc = np.einsum('hd,shd->hs',q2,ke.reshape(-1,NH,HD))/np.sqrt(float(HD))
            sc-=np.max(sc,axis=1,keepdims=True); np.exp(sc,out=sc); sc/=np.sum(sc,axis=1,keepdims=True)
            att = np.einsum('hs,shd->hd',sc,ve.reshape(-1,NH,HD)).astype(np.float32)
            b_a[:] = att.reshape(-1)
            self.kv_len[i] += 1

            # O proj (individual)
            self._kern.quant_matmul_omp(lw['attn_output']['raw'], p_a, p_o,
                                        lw['attn_output']['nr'], lw['attn_output']['nc'], lw['attn_output']['qt'])
            b_x[:N] = b_r[:N] + b_o[:N]

            # FFN
            np.copyto(b_r, b_x)
            self._simd.rms_norm(p_xn, p_x, lw['ffn_norm_w'].ctypes.data_as(cf), N, self._eps_f)

            # SiLU(gate) * up
            self._simd.silu(p_s, p_g, ctypes.c_int(FF))
            b_s[:FF] = b_s[:FF] * b_u[:FF]

            # Down proj (individual)
            self._kern.quant_matmul_omp(lw['ffn_down']['raw'], p_s, p_f,
                                        lw['ffn_down']['nr'], lw['ffn_down']['nc'], lw['ffn_down']['qt'])
            b_x[:N] = b_r[:N] + b_f[:N]

        # Final norm + output
        self._simd.rms_norm(p_xn, p_x, self._out_norm_w.ctypes.data_as(cf), N, self._eps_f)
        self._kern.quant_matmul_omp(self._out_raw, p_xn, self._pl,
                                    self._out_nr, self._out_nc, self._out_qt)
        self.pos += 1
        return self._logits

cf = ctypes.POINTER(ctypes.c_float)
cu = ctypes.POINTER(ctypes.c_uint8)
ci = ctypes.c_int
