#!/usr/bin/env python3
"""ZAYA-8B forward pass — optimized. Interleaved ATTN / MoE+MoD."""
import numpy as np, ctypes, os, gguf
from gguf.constants import GGMLQuantizationType as QT
from forward.base import ArchitectureForwardPass

GGML_Q4_K=12; GGML_Q6_K=14
C_KERNEL_TYPES = {GGML_Q4_K, GGML_Q6_K}
BLOCK_SIZES = {12:144, 14:210, 2:18, 3:20, 8:34, 13:176, 39:17}
BLOCK_VALS  = {12:256, 14:256, 2:32, 3:32, 8:32, 13:256, 39:32}

class ForwardZaya(ArchitectureForwardPass):
    def __init__(self, engine):
        super().__init__(engine)
        e = engine
        # Use engine's thread count (MXFP4 benefits from multi-threading)
        nt = max(1, getattr(e, 'n_threads', 1))
        os.environ['OMP_NUM_THREADS'] = str(nt)
        e.vocab_size = 262272
        e.head_dim = 128
        e.rope_dim = 64
        self._cos_table = {}
        self._sin_table = {}
        self._fused_gate_up = np.zeros(4096, dtype=np.float32)

    def init_weights(self, weights, raw_weights, weight_info, weight_qtypes):
        e = self.engine
        for t in e.reader.tensors:
            if t.name in ('token_embd.weight', 'model.embed_tokens.weight'):
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
                arr = np.ascontiguousarray(f32.reshape(out_dim, in_dim))
                # Ensure correct orientation: [vocab_size, n_embd]
                if arr.shape[1] != e.n_embd:
                    arr = np.ascontiguousarray(arr.T)
                e.emb = arr
                break
        if e.emb is not None and e._kern is not None:
            pass  # Output via np.dot

    def init_pointers(self, layer_idx, pfx, lw):
        """Set per-layer weight pointers."""
        e = self.engine
        weights = e.weights
        raw = e.raw_weights
        wi = e.weight_info
        cu_p = ctypes.POINTER(ctypes.c_uint8)

        # Norm + scaling (always present)
        lw.attn_norm_w = weights.get(f'{pfx}.attn_norm.weight')
        lw.res_scale_hs_w = weights.get(f'{pfx}.res_scale_hs.weight',
                                          np.ones(e.n_embd, dtype=np.float32)).ravel()
        lw.res_scale_hs_b = weights.get(f'{pfx}.res_scale_hs.bias',
                                          np.zeros(e.n_embd, dtype=np.float32)).ravel()
        rw = weights.get(f'{pfx}.res_scale_res.weight')
        lw.res_scale_res_w = rw.ravel() if rw is not None else np.ones(e.n_embd, dtype=np.float32)
        rb = weights.get(f'{pfx}.res_scale_res.bias')
        lw.res_scale_res_b = rb.ravel() if rb is not None else np.zeros(e.n_embd, dtype=np.float32)

        if layer_idx % 2 == 1:  # MoE layers
            lw.ffn_gate_inp_w = weights.get(f'{pfx}.ffn_gate_inp.weight')
            lw.ffn_gate_inp_b = weights.get(f'{pfx}.ffn_gate_inp.bias')
            if lw.ffn_gate_inp_b is not None:
                lw.ffn_gate_inp_b = lw.ffn_gate_inp_b.ravel()
            lw.ffn_gate_w = weights.get(f'{pfx}.ffn_gate.weight')
            lw.ffn_gate_b = weights.get(f'{pfx}.ffn_gate.bias')
            if lw.ffn_gate_b is not None:
                lw.ffn_gate_b = lw.ffn_gate_b.ravel()
            lw.ffn_norm_w = weights.get(f'{pfx}.ffn_norm.weight')
            lw.zaya_router_mlp2_w = weights.get(f'{pfx}.zaya_router_mlp2.weight')
            lw.zaya_router_mlp2_b = weights.get(f'{pfx}.zaya_router_mlp2.bias')
            if lw.zaya_router_mlp2_b is not None:
                lw.zaya_router_mlp2_b = lw.zaya_router_mlp2_b.ravel()
            lw.zaya_router_mlp4_w = weights.get(f'{pfx}.zaya_router_mlp4.weight')
            lw.zaya_router_biases_w = weights.get(f'{pfx}.zaya_router_biases.weight')
            if lw.zaya_router_biases_w is not None:
                lw.zaya_router_biases_w = lw.zaya_router_biases_w.ravel()

            # Load expert raw pointers — try per-expert first, then packed 3D
            cu_p = ctypes.POINTER(ctypes.c_uint8)
            n_exp = e.n_experts or 16
            for attr, wname in [('gate_up', 'ffn_gate_up_exps'), ('down', 'ffn_down_exps')]:
                ptrs, info = [], None
                # Try per-expert format: blk.N.wname.E.weight
                per_exp = any(e.raw_weights.get(f'{pfx}.{wname}.{ex}.weight') is not None for ex in range(1))
                if per_exp:
                    for exp in range(n_exp):
                        eqname = f'{pfx}.{wname}.{exp}.weight'
                        raw = e.raw_weights.get(eqname)
                        winfo = e.weight_info.get(eqname)
                        if raw is not None and winfo is not None:
                            ptr = raw.ctypes.data_as(cu_p)
                            ptrs.append(ptr)
                            if info is None:
                                info = (winfo[0], winfo[1], winfo[3])
                        else:
                            ptrs.append(ctypes.cast(0, cu_p))
                else:
                    # Packed 3D format: blk.N.wname.weight  [out_dim, in_dim, n_exp]
                    pname = f'{pfx}.{wname}.weight'
                    raw = e.raw_weights.get(pname)
                    winfo = e.weight_info.get(pname)
                    if raw is not None and winfo is not None:
                        out_dim, in_dim, bs, qt, n_exp2 = winfo
                        bv = {8: 32, 12: 256, 14: 256, 39: 32}.get(qt, 32)
                        rb = ((in_dim + bv - 1) // bv) * bs
                        eb = out_dim * rb
                        raw_ptr = raw.ctypes.data_as(cu_p)
                        info = (out_dim, in_dim, qt)
                        for exp in range(n_exp2):
                            addr = ctypes.addressof(raw_ptr.contents) + exp * eb
                            ptrs.append(ctypes.cast(addr, cu_p))
                if ptrs:
                    arr_type = cu_p * n_exp
                    setattr(lw, f'{attr}_ptrs', arr_type(*ptrs))
                    if info:
                        setattr(lw, f'{attr}_info', info)

    def _rope(self, x, pos, n_heads, hd):
        half = self.engine.rope_dim // 2
        theta = 5000000.0
        key = (theta, pos, half)
        if key not in self._cos_table:
            f = theta ** (np.arange(0, self.engine.rope_dim, 2, dtype=np.float32) / self.engine.rope_dim)
            self._cos_table[key] = np.cos(pos / f).astype(np.float32)
            self._sin_table[key] = np.sin(pos / f).astype(np.float32)
        c, s = self._cos_table[key], self._sin_table[key]
        x2 = x.reshape(n_heads, hd)
        o = x2.copy()
        o[:, :half] = x2[:, :half] * c - x2[:, half:self.engine.rope_dim] * s
        o[:, half:self.engine.rope_dim] = x2[:, half:self.engine.rope_dim] * c + x2[:, :half] * s
        return o.reshape(-1)

    def forward(self, token_id):
        e = self.engine
        N = e.n_embd; NH = e.n_head; NKH = e.n_kv_head; HD = e.head_dim; L = e.n_layers
        NQ = NH * HD; NK = NKH * HD

        b_x = e._x; b_r = e._residual; b_xn = e._x_norm
        b_q = e._q; b_k = e._k[:NK]; b_v = e._k[:NK]
        b_att = e._att_out; b_up = e._up; b_silu = e._silu_gate
        b_ffn = e._ffn_out; b_oproj = e._o_proj; b_log = e._logits

        kern = e._kern; simd = e._simd; eps = e._eps_f
        cf = ctypes.POINTER(ctypes.c_float); ci = ctypes.c_int; cu = ctypes.POINTER(ctypes.c_uint8)
        
        # Use cengine's mxfp4 matmul for MXFP4 model (type 39)
        ce = getattr(e, '_cengine', None)
        if ce is not None and not hasattr(ce, '_mxfp4_ready'):
            ce.mxfp4_batch_matmul.argtypes = [cu, cf, cf, ci, ci, ci]
            ce.mxfp4_batch_matmul.restype = None
            ce._mxfp4_ready = True
        # Use ce's mxfp4 matmul when qt == 39 (MXFP4)
        _mxm_ce_count = [0]
        def mxm(w, x, out, nr, nc, qt):
            qtv = qt.value if isinstance(qt, ctypes.c_int) else qt
            if qtv == 39:
                _mxm_ce_count[0] += 1
                ce.mxfp4_batch_matmul(w, x, out, nr, nc, ci(1))
            else:
                kern.quant_matmul_omp(w, x, out, nr, nc, qt)
        import forward.zaya as _zmod_mxm
        _zmod_mxm._mxm_ce_count = _mxm_ce_count

        np.copyto(b_x, e.emb[token_id])

        for i in range(L):
            lw = e._layers[i]
            np.clip(b_x, -1000.0, 1000.0, out=b_x)
            np.copyto(b_r, b_x)
            simd.rms_norm(e._p_x_norm, e._p_x, lw.attn_norm_w.ctypes.data_as(cf), N, eps)

            if i % 2 == 0:  # ATTN layer
                mxm(lw.attn_q_raw, e._p_x_norm, e._p_q, ci(NQ), ci(N), lw.attn_q_qt)
                mxm(lw.attn_k_raw, e._p_x_norm, e._p_k, ci(NK), ci(N), lw.attn_k_qt)

                b_q[:] = self._rope(b_q, e.pos, NH, HD)
                b_k[:NK] = self._rope(b_k[:NK], e.pos, NKH, HD)

                e.kv_k[i, e.kv_len[i], :NK] = b_k[:NK]
                b_v[:] = b_k[:NK]
                sl = e.kv_len[i] + 1
                kc = e.kv_k[i, :sl, :NK].reshape(sl, NKH, HD)
                q2 = b_q.reshape(NH, HD)
                gr = NH // NKH
                qg = q2.reshape(NKH, gr, HD)
                kt = kc.transpose(1, 0, 2)
                sc = np.einsum('khd,ksd->khs', qg, kt) / np.sqrt(float(HD))
                s2 = sc.reshape(NH, sl)
                s2 -= np.max(s2, axis=1, keepdims=True)
                np.exp(s2, out=s2); s2 /= np.sum(s2, axis=1, keepdims=True)
                at = np.einsum('khs,ksd->khd', s2.reshape(NKH, gr, sl), kc.transpose(1, 0, 2))
                b_att[:NQ] = at.reshape(-1)
                e.kv_len[i] += 1

                mxm(lw.attn_out_raw, e._p_att_out, e._p_o_proj,
                    lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt)
                b_x[:N] = b_r[:N] * lw.res_scale_res_w + lw.res_scale_res_b + \
                          b_oproj[:N] * lw.res_scale_hs_w + lw.res_scale_hs_b
            else:  # MoE layer
                h = lw.ffn_gate_inp_w @ b_xn
                if lw.ffn_gate_inp_b is not None: h += lw.ffn_gate_inp_b
                h = lw.ffn_gate_w @ h
                if lw.ffn_gate_b is not None: h += lw.ffn_gate_b
                h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (h + 0.044715 * h * h * h)))
                h = lw.zaya_router_mlp2_w @ h
                if lw.zaya_router_mlp2_b is not None: h += lw.zaya_router_mlp2_b
                ms = lw.zaya_router_mlp4_w @ h
                if lw.zaya_router_biases_w is not None: ms += lw.zaya_router_biases_w
                ms -= np.max(ms); np.exp(ms, out=ms); ms /= np.sum(ms)

                ec = int(np.argmax(ms)); ew = float(ms[ec])
                if ec < 16 and ew > 0.01:
                    ew /= ms[:16].sum()
                    od, id_, qt = lw.gate_up_info
                    mxm(lw.gate_up_ptrs[ec], e._p_x_norm,
                        self._fused_gate_up.ctypes.data_as(cf),
                        ci(od), ci(id_), ci(qt))
                    F2 = od // 2  # 2048
                    np.copyto(b_silu[:F2], self._fused_gate_up[:F2])
                    np.copyto(b_up[:F2], self._fused_gate_up[F2:])
                    b_silu[:F2] = b_silu[:F2] / (1.0 + np.exp(-b_silu[:F2]))
                    b_silu[:F2] *= b_up[:F2]

                    od2, id2, qt2 = lw.down_info
                    mxm(lw.down_ptrs[ec], e._p_silu_gate, e._p_ffn,
                        ci(od2), ci(id2), ci(qt2))
                    b_ffn[:N] *= ew
                else:
                    b_ffn[:N] = 0.0

                b_x[:N] = b_r[:N] * lw.res_scale_res_w + lw.res_scale_res_b + \
                          b_ffn[:N] * lw.res_scale_hs_w + lw.res_scale_hs_b

        np.clip(b_x, -1000.0, 1000.0, out=b_x)
        simd.rms_norm(e._p_x_norm, e._p_x, e._out_norm_w.ctypes.data_as(cf), N, eps)
        np.dot(e.emb, b_xn, out=b_log[:e.emb.shape[0]])
        e.pos += 1
        return b_log
