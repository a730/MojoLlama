"""Gemma4 26B forward — shared dense FFN + per-expert MXFP4 MoE.
Architecture: llama.cpp CANNOT run this (no MXFP4 support). MojoLlama CAN.

Gemma4 26B has alternating head dimensions:
  - Most layers: Q_dim=4096 (NH=16, HD=256), K_dim=2048 (NKH=2)
  - Every 6th layer (5, 11, 17, 23, 29): Q_dim=8192 (HD=512), K_dim=1024 (HD=512)
Detected per-layer from q_norm_w.shape.
"""
import numpy as np, ctypes
from forward.base import ArchitectureForwardPass

class ForwardGemma4_26B(ArchitectureForwardPass):
    def __init__(self, engine):
        super().__init__(engine)
        self._cos_table, self._sin_table = {}, {}
        self._per_layer_hd = None
        self._max_hd = 256

    def init_weights(self, *a): pass
    def init_pointers(self, *a): pass
    def init_buffers(self):
        """Detect per-layer HD and resize buffers for max (512)."""
        e = self.engine
        self._per_layer_hd = []
        self._max_hd = 256
        for i in range(e.n_layers):
            lw = e._layers[i]
            hd = lw.q_norm_w.shape[0] if lw.has_q_norm else 256
            self._per_layer_hd.append(hd)
            if hd > self._max_hd:
                self._max_hd = hd
        max_nk = e.n_kv_head * self._max_hd
        # Q/K buffers for max HD
        q_need = e.n_head * self._max_hd
        k_need = e.n_kv_head * self._max_hd
        if len(e._q) < q_need:
            e._q = np.zeros(q_need, dtype=np.float32)
            e._p_q = e._q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if len(e._k) < k_need:
            e._k = np.zeros(k_need, dtype=np.float32)
            e._p_k = e._k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if len(e._att_out) < q_need:
            e._att_out = np.zeros(q_need, dtype=np.float32)
            e._p_att_out = e._att_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # KV cache for max HD
        if e.kv_k.shape[2] < max_nk:
            old = e.kv_k
            e.kv_k = np.zeros((e.n_layers, 4096, max_nk), dtype=np.float32)
            e.kv_k[:, :old.shape[1], :old.shape[2]] = old[:, :, :]
            old = e.kv_v
            e.kv_v = np.zeros((e.n_layers, 4096, max_nk), dtype=np.float32)
            e.kv_v[:, :old.shape[1], :old.shape[2]] = old[:, :, :]

    def reset_state(self): pass

    def _rope(self, x, pos, n_heads, hd, rope_dim):
        half = rope_dim // 2
        theta = 1000000.0
        key = (theta, pos, half)
        if key not in self._cos_table:
            freqs = theta ** (-np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim)
            self._cos_table[key] = np.cos(pos * freqs).astype(np.float32)
            self._sin_table[key] = np.sin(pos * freqs).astype(np.float32)
        c, s = self._cos_table[key], self._sin_table[key]
        x2 = x.reshape(n_heads, hd)
        out = x2.copy()
        out[:, :half] = x2[:, :half] * c - x2[:, half:rope_dim] * s
        out[:, half:rope_dim] = x2[:, half:rope_dim] * c + x2[:, :half] * s
        return out.reshape(-1)

    def _gate_up_ptrs(self, me, l):
        import ctypes as ct
        if not hasattr(self, '_gate_up_ptr_cache'):
            self._gate_up_ptr_cache = {}
        if l in self._gate_up_ptr_cache:
            return self._gate_up_ptr_cache[l]
        e = self.engine
        name = f'blk.{l}.ffn_gate_up_exps.weight'
        if name not in e.raw_weights:
            self._gate_up_ptr_cache[l] = None
            return None
        raw = e.raw_weights[name]
        base_addr = raw.ctypes.data
        per_exp = 2106368
        ptrs = []
        cu = ct.POINTER(ct.c_uint8)
        for exp in range(128):
            addr = base_addr + exp * per_exp
            ptrs.append(ct.cast(addr, cu))
        arr_type = cu * 128
        self._gate_up_ptr_cache[l] = arr_type(*ptrs)
        return self._gate_up_ptr_cache[l]

    def forward(self, token_id):
        e = self.engine; N = e.n_embd; NH = e.n_head; NKH = e.n_kv_head
        L = e.n_layers; FF = e.n_ff; V = e.vocab_size
        b_x = e._x; b_r = e._residual; b_xn = e._x_norm
        b_oproj = e._o_proj
        b_gate = e._gate; b_up = e._up; b_silu = e._silu_gate; b_ffn = e._ffn_out
        b_log = e._logits
        kern = e._kern; simd = e._simd; eps = e._eps_f
        cf = ctypes.POINTER(ctypes.c_float); ci = ctypes.c_int; cu = ctypes.POINTER(ctypes.c_uint8)

        np.copyto(b_x, e.emb[token_id])

        for i in range(L):
            lw = e._layers[i]; me = e._moe_layers[i]
            hd = self._per_layer_hd[i]
            nq = NH * hd; nk = NKH * hd
            b_q = e._q[:nq]; b_k = e._k[:nk]; b_v = e._k[:nk]; b_att = e._att_out[:nq]

            # Attention
            np.clip(b_x, -1000.0, 1000.0, out=b_x)
            np.copyto(b_r, b_x)
            simd.rms_norm(b_xn.ctypes.data_as(cf), b_x.ctypes.data_as(cf),
                          lw.attn_norm_w.ctypes.data_as(cf), N, eps)
            kern.quant_matmul_omp(lw.attn_q_raw, b_xn.ctypes.data_as(cf), b_q.ctypes.data_as(cf),
                                   lw.attn_q_nr, lw.attn_q_nc, lw.attn_q_qt)
            kern.quant_matmul_omp(lw.attn_k_raw, b_xn.ctypes.data_as(cf), b_k.ctypes.data_as(cf),
                                   lw.attn_k_nr, lw.attn_k_nc, lw.attn_k_qt)
            b_v[:nk] = b_k[:nk]
            if lw.has_q_norm:
                q2d = b_q.reshape(NH, hd)
                qrms = np.sqrt(np.mean(q2d*q2d, axis=1, keepdims=True) + 1e-6)
                q2d[:] = q2d / qrms * lw.q_norm_w.reshape(1, hd)
            if lw.has_k_norm:
                k2d = b_k[:nk].reshape(NKH, hd)
                krms = np.sqrt(np.mean(k2d*k2d, axis=1, keepdims=True) + 1e-6)
                k2d[:] = k2d / krms * lw.k_norm_w.reshape(1, hd)
            rd = min(hd, 256)
            b_q[:] = self._rope(b_q, e.pos, NH, hd, rd)
            b_k[:nk] = self._rope(b_k[:nk], e.pos, NKH, hd, rd)
            e.kv_k[i, e.kv_len[i], :nk] = b_k[:nk]
            e.kv_v[i, e.kv_len[i], :nk] = b_v[:nk]
            sl = e.kv_len[i] + 1; e.kv_len[i] += 1
            # GQA attention
            gr = NH // NKH
            kc = e.kv_k[i, :sl, :nk].reshape(sl, NKH, hd)
            q2d = b_q.reshape(NH, hd); qg = q2d.reshape(NKH, gr, hd)
            kt = kc.transpose(1, 0, 2)
            sc = np.einsum('khd,ksd->khs', qg, kt) / np.sqrt(float(hd))
            s2 = sc.reshape(NH, sl)
            s2 -= np.max(s2, axis=1, keepdims=True)
            np.exp(s2, out=s2); s2 /= np.sum(s2, axis=1, keepdims=True)
            at = np.einsum('khs,ksd->khd', s2.reshape(NKH, gr, sl), kc.transpose(1, 0, 2))
            b_att[:] = at.reshape(-1)
            # O proj + residual
            kern.quant_matmul_omp(lw.attn_out_raw, b_att.ctypes.data_as(cf), b_oproj.ctypes.data_as(cf),
                                   lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt)
            b_x[:N] = b_r[:N] * getattr(lw, 'res_scale_res_w', 1.0)
            if hasattr(lw, 'res_scale_res_b'): b_x[:N] += lw.res_scale_res_b
            b_x[:N] += b_oproj[:N] * getattr(lw, 'res_scale_hs_w', 1.0)
            if hasattr(lw, 'res_scale_hs_b'): b_x[:N] += lw.res_scale_hs_b

            # FFN (shared dense + MoE)
            np.clip(b_x, -1000.0, 1000.0, out=b_x)
            np.copyto(b_r, b_x)
            simd.rms_norm(b_xn.ctypes.data_as(cf), b_x.ctypes.data_as(cf),
                          lw.ffn_norm_w.ctypes.data_as(cf), N, eps)
            # Shared dense
            kern.quant_matmul_omp(lw.ffn_gate_raw, b_xn.ctypes.data_as(cf), b_gate.ctypes.data_as(cf),
                                   lw.ffn_gate_nr, lw.ffn_gate_nc, lw.ffn_gate_qt)
            kern.quant_matmul_omp(lw.ffn_up_raw, b_xn.ctypes.data_as(cf), b_up.ctypes.data_as(cf),
                                   lw.ffn_up_nr, lw.ffn_up_nc, lw.ffn_up_qt)
            b_silu[:FF] = b_gate[:FF] / (1.0 + np.exp(-np.clip(b_gate[:FF], -80, 80)))
            b_silu[:FF] *= b_up[:FF]
            kern.quant_matmul_omp(lw.ffn_down_raw, b_silu.ctypes.data_as(cf), b_ffn.ctypes.data_as(cf),
                                   lw.ffn_down_nr, lw.ffn_down_nc, lw.ffn_down_qt)
            # MoE per-expert
            moe_ffn = np.zeros(N, dtype=np.float32)
            if me.router_f32 is not None:
                scores = me.router_f32 @ b_xn
                scores -= np.max(scores)
                np.exp(scores, out=scores); scores /= np.sum(scores)
                top_k = min(e.n_experts_per_tok, 8)
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_wt = scores[top_idx] / np.sum(scores[top_idx])
                gu_ptrs = self._gate_up_ptrs(me, i)
                if gu_ptrs is not None and me.down_ptrs_arr is not None:
                    for exp, wt in zip(top_idx, top_wt):
                        if wt < 0.01: continue
                        gu_out = np.zeros(1408, dtype=np.float32)
                        if hasattr(e._cengine, 'mxfp4_batch_matmul'):
                            e._cengine.mxfp4_batch_matmul(
                                gu_ptrs[exp], b_xn.ctypes.data_as(cf),
                                gu_out.ctypes.data_as(cf), ci(1408), ci(2816), ci(1))
                        else:
                            kern.quant_matmul_omp(gu_ptrs[exp], b_xn.ctypes.data_as(cf),
                                                   gu_out.ctypes.data_as(cf),
                                                   ci(1408), ci(2816), ci(39))
                        g = gu_out[:704]; u = gu_out[704:]
                        # Clip to prevent overflow in g*u (Gemma4 26B experts produce large values)
                        g = np.clip(g, -1e10, 1e10)
                        u = np.clip(u, -1e10, 1e10)
                        g = g / (1.0 + np.exp(-np.clip(g, -80, 80)))
                        g_u = g * u
                        exp_out = np.zeros(N, dtype=np.float32)
                        if hasattr(e._cengine, 'mxfp4_batch_matmul'):
                            e._cengine.mxfp4_batch_matmul(
                                me.down_ptrs_arr[exp], g_u.ctypes.data_as(cf),
                                exp_out.ctypes.data_as(cf), ci(N), ci(704), ci(1))
                        else:
                            kern.quant_matmul_omp(me.down_ptrs_arr[exp], g_u.ctypes.data_as(cf),
                                                   exp_out.ctypes.data_as(cf),
                                                   me.down_nr, me.down_nc, me.down_qt)
                        moe_ffn += wt * exp_out
            b_ffn[:N] += moe_ffn
            b_x[:N] = b_r[:N] * getattr(lw, 'res_scale_res_w', 1.0)
            if hasattr(lw, 'res_scale_res_b'): b_x[:N] += lw.res_scale_res_b
            b_x[:N] += b_ffn[:N] * getattr(lw, 'res_scale_hs_w', 1.0)
            if hasattr(lw, 'res_scale_hs_b'): b_x[:N] += lw.res_scale_hs_b

        # Final norm + output
        np.clip(b_x, -1000.0, 1000.0, out=b_x)
        simd.rms_norm(b_xn.ctypes.data_as(cf), b_x.ctypes.data_as(cf),
                      e._out_norm_w.ctypes.data_as(cf), N, eps)
        # Output: b_log = emb @ x_norm
        np.dot(e.emb, b_xn, out=b_log[:V])
        e.pos += 1
        return b_log
