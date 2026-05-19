#!/usr/bin/env python3
"""TurboEngine v7.7 — v7 + C GQA attention + AVX2 v2 kernels + RoPE precompute.

Changes from v7.5:
  1. C GQA attention kernel (replaces numpy einsum + GQA repeat)
  2. AVX2 v2 for individual matmuls (O proj, Down)
  3. Pre-computed RoPE tables for all positions up to max_seq_len
  4. Keep batch_qkv_omp and batch_gate_up_omp for fused OMP matmuls

Combined expected: 78 → ~86 tok/s = 92% of llama.cpp
"""
import numpy as np, ctypes, os, gguf
from gguf.constants import GGMLQuantizationType as QT
import math

GGML_F32=0; GGML_F16=1; GGML_Q4_0=2; GGML_Q4_1=3; GGML_Q8_0=8
GGML_Q4_K=12; GGML_Q5_K=13; GGML_Q6_K=14
BLOCK_SIZES={2:18,3:20,12:144,13:176,14:210,8:34}
C_KERNEL_TYPES={GGML_Q4_0,GGML_Q4_1,GGML_Q4_K,GGML_Q5_K,GGML_Q6_K,GGML_Q8_0}
AVX2_TYPES={GGML_Q4_0,GGML_Q4_1,GGML_Q8_0}

class TurboEngineV77:
    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS']=str(n_threads)
        self.n_threads=n_threads
        self.reader=gguf.GGUFReader(model_path)
        self._parse_metadata()
        self._load_weights()
        self._load_kernels()
        self._preload_layer_weights()
        self._init_kv_cache()
        self._precompute_rope()
        self._eps_f=ctypes.c_float(self.eps)
        self.reset()
        print(f"TurboEngine v7.7: {self.n_layers}L/{self.n_embd}D/"
              f"{self.n_ff}FF/{self.n_head}H/{self.n_kv_head}KV | t={n_threads}")

    def _parse_metadata(self):
        fields=self.reader.fields
        def _get(n):
            for k,v in fields.items():
                if k==n:
                    p=v.parts if hasattr(v,'parts') else []
                    if len(p)>=1:
                        d=p[-1]
                        if hasattr(d,'__iter__') and len(d)==1: return int(d[0])
                        return d
        arch='llama'
        for p in ['qwen3moe','qwen2moe','llama','mistral']:
            for k in fields:
                if f'{p}.block_count' in k: arch=p; break
        self.n_layers=int(_get(f'{arch}.block_count') or 16)
        self.n_embd=int(_get(f'{arch}.embedding_length') or 2048)
        self.n_ff=int(_get(f'{arch}.feed_forward_length') or self.n_embd*4)
        self.n_head=int(_get(f'{arch}.attention.head_count') or 32)
        self.n_kv_head=int(_get(f'{arch}.attention.head_count_kv') or self.n_head)
        self.head_dim=int(_get(f'{arch}.attention.key_length') or (self.n_embd//self.n_head))
        self.rope_freq_base=float(_get(f'{arch}.rope.freq_base') or 10000.0)
        self.eps=float(_get(f'{arch}.attention.layer_norm_rms_epsilon') or 1e-6)
        self.vocab_size=int(_get(f'{arch}.vocab_size') or 0)
        self.n_experts=_get(f'{arch}.expert_count')
        self.n_experts_per_tok=_get(f'{arch}.expert_used_count')
        self.is_moe=self.n_experts is not None
        if not self.is_moe: self.n_experts=1; self.n_experts_per_tok=1
        else: self.n_experts=int(self.n_experts); self.n_experts_per_tok=int(self.n_experts_per_tok)
        self.arch_prefix=arch

    def _load_weights(self):
        self.weights={}; self.raw_weights={}; self.weight_info={}
        for t in self.reader.tensors:
            name=t.name; qt=QT(t.tensor_type).value
            if len(t.shape)==2:
                in_dim,out_dim=int(t.shape[0]),int(t.shape[1])
                f32=gguf.dequantize(t.data,t.tensor_type).astype(np.float32).reshape(out_dim,in_dim)
                self.weights[name]=np.ascontiguousarray(f32)
                if qt in C_KERNEL_TYPES:
                    raw=np.ascontiguousarray(t.data.reshape(-1),dtype=np.uint8).copy()
                    self.raw_weights[name]=raw
                    self.weight_info[name]=(out_dim,in_dim,BLOCK_SIZES[qt],qt)
            elif len(t.shape)==3:
                in_dim,out_dim,ne=int(t.shape[0]),int(t.shape[1]),int(t.shape[2])
                f32=gguf.dequantize(t.data,t.tensor_type).astype(np.float32).reshape(ne,out_dim,in_dim)
                self.weights[name]=np.ascontiguousarray(f32)
            elif len(t.shape)==1:
                f32=gguf.dequantize(t.data,t.tensor_type).astype(np.float32).reshape(-1)
                self.weights[name]=np.ascontiguousarray(f32)
        self.out_w_name='output.weight' if 'output.weight' in self.weights else 'token_embd.weight'
        emb=self.weights['token_embd.weight']
        if emb.ndim==2 and emb.shape[1]!=self.n_embd: emb=np.ascontiguousarray(emb.T); self.weights['token_embd.weight']=emb
        if self.vocab_size==0: self.vocab_size=emb.shape[0]
        self.emb=emb

    def _load_kernels(self):
        kd=os.path.dirname(os.path.abspath(__file__))
        cf=ctypes.POINTER(ctypes.c_float); cu=ctypes.POINTER(ctypes.c_uint8); ci=ctypes.c_int

        # OMP kernels
        o=ctypes.CDLL(os.path.join(kd,'quant_kernels_omp.so'))
        o.quant_matmul_omp.argtypes=[cu,cf,cf,ci,ci,ci]; o.quant_matmul_omp.restype=None
        o.batch_qkv_omp.argtypes=[cu,cu,cu,cf,cf,cf,cf,ci,ci,ci,ci,ci,ci]; o.batch_qkv_omp.restype=None
        o.batch_gate_up_omp.argtypes=[cu,cu,cf,cf,cf,ci,ci,ci,ci]; o.batch_gate_up_omp.restype=None
        o.set_num_threads.argtypes=[ci]; o.set_num_threads.restype=None; o.set_num_threads(self.n_threads)
        self._kern=o

        # AVX2 v2 kernels
        v2p=os.path.join(kd,'turbo_kernels_v2.so')
        if os.path.exists(v2p):
            v2=ctypes.CDLL(v2p)
            v2.quant_matmul_v2.argtypes=[cu,cf,cf,ci,ci,ci]; v2.quant_matmul_v2.restype=None
            self._v2=v2
        else: self._v2=None

        # SIMD ops
        s=ctypes.CDLL(os.path.join(kd,'simd_ops.so'))
        s.rms_norm.argtypes=[cf,cf,cf,ci,ctypes.c_float]; s.rms_norm.restype=None
        s.silu.argtypes=[cf,cf,ci]; s.silu.restype=None
        self._simd=s

        # GQA attention
        a=ctypes.CDLL(os.path.join(kd,'gqa_attention.so'))
        a.gqa_attention_decode.argtypes=[cf,cf,cf,cf,ci,ci,ci,ci]; a.gqa_attention_decode.restype=None
        self._attn=a

    def _matmul(self, rp, xp, op, nr, nc, qt, v2):
        if v2 and self._v2: self._v2.quant_matmul_v2(rp,xp,op,nr,nc,qt)
        else: self._kern.quant_matmul_omp(rp,xp,op,nr,nc,qt)

    def _preload_layer_weights(self):
        self._layers=[]
        for i in range(self.n_layers):
            pfx=f'blk.{i}'
            pn=['attn_q','attn_k','attn_v','attn_output','ffn_gate','ffn_up','ffn_down']
            lw={}
            for a in pn:
                n=f'{pfx}.{a}.weight'
                info=self.weight_info.get(n)
                if info:
                    r=self.raw_weights[n]
                    lw[a]={'raw':r.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                           'nr':ctypes.c_int(info[0]),'nc':ctypes.c_int(info[1]),
                           'qt':ctypes.c_int(info[3]),'v2':info[3] in AVX2_TYPES and self._v2}
                else: lw[a]={'f32':self.weights[n],'v2':False}
            lw['ann_w']=self.weights[f'{pfx}.attn_norm.weight']
            lw['fnn_w']=self.weights[f'{pfx}.ffn_norm.weight']
            self._layers.append(lw)
        # Output proj
        oi=self.weight_info.get(self.out_w_name)
        if oi:
            self._onr=ctypes.c_int(oi[0]); self._onc=ctypes.c_int(oi[1])
            if oi[3] in (6,14,12,10):
                self._or=self._requant(oi)
                self._oqt=ctypes.c_int(8); self._ov2=True
            else:
                self._or=self.raw_weights[self.out_w_name].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                self._oqt=ctypes.c_int(oi[3]); self._ov2=oi[3] in AVX2_TYPES
        else: self._or=None
        self._onw=self.weights['output_norm.weight']

    def _requant(self, oi):
        nr,nc=oi[0],oi[1]; nb=nc//32
        f32=np.zeros((nr,nc),dtype=np.float32)
        if oi[3]==14:
            nblk=nc//256
            self._kern.q6_k_dequantize_row.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_float),ctypes.c_int]
            self._kern.q6_k_dequantize_row.restype=None
            rp=self.raw_weights[self.out_w_name].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            ba=ctypes.addressof(rp.contents)
            for r in range(nr):
                ptr=ctypes.cast(ba+r*nblk*210,ctypes.POINTER(ctypes.c_uint8))
                self._kern.q6_k_dequantize_row(ptr,f32[r].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),ctypes.c_int(nc))
        blks=f32.reshape(nr,nb,32); amax=np.max(np.abs(blks),axis=2,keepdims=True)
        amax=np.clip(amax,1e-10,None); d=amax/127.0
        qs=np.clip(np.round(blks/d),-127,127).astype(np.int8)
        dh=d[:,:,0].astype(np.float16).view(np.uint16)
        q8=np.zeros((nr,nb,34),dtype=np.uint8)
        q8[:,:,0]=dh&0xFF; q8[:,:,1]=(dh>>8)&0xFF; q8[:,:,2:]=qs.view(np.uint8)
        self._oq8=np.ascontiguousarray(q8.reshape(-1),dtype=np.uint8)
        print(f"  Output: {nr}×{nc} → Q8_0 ({len(self._oq8)/1024/1024:.0f} MB)")
        return self._oq8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    def _init_kv_cache(self):
        N=self.n_embd; NKH=self.n_kv_head; HD=self.head_dim; MP=4096
        self.kvk=np.zeros((self.n_layers,MP,NKH*HD),dtype=np.float32)
        self.kvv=np.zeros((self.n_layers,MP,NKH*HD),dtype=np.float32)
        self.kvl=np.zeros(self.n_layers,dtype=np.int32)
        NK=self.n_head*HD; FF=self.n_ff
        self._x=np.zeros(N,dtype=np.float32); self._xn=np.zeros(N,dtype=np.float32)
        self._res=np.zeros(N,dtype=np.float32)
        self._q=np.zeros(NK,dtype=np.float32); self._k=np.zeros(NKH*HD,dtype=np.float32)
        self._v=np.zeros(NKH*HD,dtype=np.float32); self._att=np.zeros(NK,dtype=np.float32)
        self._g=np.zeros(FF,dtype=np.float32); self._u=np.zeros(FF,dtype=np.float32)
        self._s=np.zeros(FF,dtype=np.float32); self._o=np.zeros(N,dtype=np.float32)
        self._f=np.zeros(N,dtype=np.float32); self._lg=np.zeros(self.vocab_size,dtype=np.float32)

    def _precompute_rope(self):
        """Pre-compute all RoPE cos/sin tables for positions 0-4095."""
        hd=self.head_dim; half=hd//2
        freq=self.rope_freq_base**(np.arange(0,hd,2,dtype=np.float32)/hd)
        self._rc=np.zeros((4096,half),dtype=np.float32)
        self._rs=np.zeros((4096,half),dtype=np.float32)
        for p in range(4096):
            ang=p/freq
            self._rc[p]=np.cos(ang).astype(np.float32)
            self._rs[p]=np.sin(ang).astype(np.float32)

    def _rope(self, x, pos, nh):
        hd=self.head_dim; h=hd//2
        c=self._rc[pos]; s=self._rs[pos]
        x2=x.reshape(nh,hd); o=x2.copy()
        o[:,:h]=x2[:,:h]*c-x2[:,h:]*s; o[:,h:]=x2[:,h:]*c+x2[:,:h]*s
        return o.reshape(-1)

    def reset(self): self.kvl[:]=0; self.pos=0

    def forward(self, token_id):
        bx=self._x; bxn=self._xn; br=self._res; bq=self._q; bk=self._k; bv=self._v
        ba=self._att; bg=self._g; bu=self._u; bs=self._s; bo=self._o; bf=self._f
        N=self.n_embd; NH=self.n_head; NKH=self.n_kv_head; HD=self.head_dim
        FF=self.n_ff; L=self.n_layers
        k=self._kern; d=self._simd; a=self._attn; e=self._eps_f
        px=self._x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pxn=self._xn.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pr=self._res.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pq=self._q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pk=self._k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pv=self._v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pa=self._att.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pg=self._g.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pu=self._u.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ps=self._s.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        po=self._o.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pf=self._f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pl=self._lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ci=ctypes.c_int

        np.copyto(bx, self.emb[token_id])

        for i in range(L):
            lw=self._layers[i]
            np.copyto(br, bx)
            d.rms_norm(pxn, px, lw['ann_w'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, e)

            # QKV batch
            if 'attn_q' in lw and 'raw' in lw['attn_q']:
                k.batch_qkv_omp(lw['attn_q']['raw'],lw['attn_k']['raw'],lw['attn_v']['raw'],
                                pxn,pq,pk,pv,lw['attn_q']['nr'],lw['attn_k']['nr'],lw['attn_v']['nr'],
                                lw['attn_q']['nc'],lw['attn_q']['qt'],lw['attn_k']['qt'],lw['attn_v']['qt'])

            # RoPE
            bq[:]=self._rope(bq, self.pos, NH)
            bk[:]=self._rope(bk, self.pos, NKH)

            # KV store
            self.kvk[i,self.kvl[i],:NKH*HD]=bk[:NKH*HD]
            self.kvv[i,self.kvl[i],:NKH*HD]=bv[:NKH*HD]

            # ── C GQA attention ──
            sl=self.kvl[i]+1
            kc=self.kvk[i,:sl].reshape(-1); vc=self.kvv[i,:sl].reshape(-1)
            a.gqa_attention_decode(bq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   kc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   vc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   ba.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   ci(sl),ci(NH),ci(NKH),ci(HD))
            self.kvl[i]+=1

            # O proj + residual
            self._matmul(lw['attn_output']['raw'], pa, po,
                         lw['attn_output']['nr'], lw['attn_output']['nc'], lw['attn_output']['qt'],
                         lw['attn_output'].get('v2',False))
            bx[:N]=br[:N]+bo[:N]

            # FFN
            np.copyto(br, bx)
            d.rms_norm(pxn, px, lw['fnn_w'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, e)

            # Gate+Up batch
            k.batch_gate_up_omp(lw['ffn_gate']['raw'],lw['ffn_up']['raw'],pxn,pg,pu,
                                lw['ffn_gate']['nr'],lw['ffn_up']['nr'],
                                lw['ffn_gate']['nc'],lw['ffn_gate']['qt'],lw['ffn_up']['qt'])

            # SiLU * up
            d.silu(ps, pg, ci(FF))
            bs[:FF]=bs[:FF]*bu[:FF]

            # Down + residual
            self._matmul(lw['ffn_down']['raw'], ps, pf,
                         lw['ffn_down']['nr'], lw['ffn_down']['nc'], lw['ffn_down']['qt'],
                         lw['ffn_down'].get('v2',False))
            bx[:N]=br[:N]+bf[:N]

        # Final
        d.rms_norm(pxn, px, self._onw.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, e)
        self._matmul(self._or, pxn, pl, self._onr, self._onc, self._oqt, self._ov2)
        self.pos+=1
        return self._lg
