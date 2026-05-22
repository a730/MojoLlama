#!/usr/bin/env python3
"""Gemma 4 forward pass via C function — single call from Python, no layer loop overhead."""
import numpy as np, ctypes
from forward.base import ArchitectureForwardPass

class ForwardGemma4(ArchitectureForwardPass):
    """Gemma4 forward pass — calls gemma4_forward_c for all 35 layers in one C call."""

    def __init__(self, engine):
        super().__init__(engine)
        e = engine; a = e.arch_name

        def _get(name):
            fields = e.reader.fields
            for key, val in fields.items():
                if key == name:
                    parts = val.parts if hasattr(val, 'parts') else []
                    if len(parts) >= 1:
                        data = parts[-1]
                        if hasattr(data, '__iter__') and len(data) == 1: return int(data[0])
                        return data
            return None

        self.per_layer_dim = int(_get(f'{a}.embedding_length_per_layer_input') or 256)
        self.sliding_window = int(_get(f'{a}.attention.sliding_window') or 512)
        self.shared_kv = int(_get(f'{a}.attention.shared_kv_layers') or 20)
        self.logit_cap = float(_get(f'{a}.final_logit_softcapping') or 30.0)
        self.rope_freq_base_swa = float(_get(f'{a}.rope.freq_base_swa') or 10000.0)
        self.rope_freq_base = float(_get(f'{a}.rope.freq_base') or 1000000.0)
        self.rope_dim_swa = int(_get(f'{a}.rope.dimension_count_swa') or 256)
        self.rope_dim = int(_get(f'{a}.rope.dimension_count') or 512)
        self._c_built = False

    def _build_c(self):
        """Build per-layer ctype arrays once."""
        if self._c_built: return
        self._c_built = True
        e = self.engine; L = e.n_layers
        cf = ctypes.POINTER(ctypes.c_float); cu = ctypes.POINTER(ctypes.c_uint8); ci = ctypes.c_int

        def fp(arr):
            return arr.ctypes.data_as(cf) if arr is not None and hasattr(arr, 'ctypes') else cf()
        def up(arr):
            return arr.ctypes.data_as(cu) if arr is not None and hasattr(arr, 'ctypes') else cu()

        # F32 norm/projection weight arrays
        self._ca_attn = (cf*L)(*[fp(lw.attn_norm_w) for lw in e._layers])
        self._ca_ffn = (cf*L)(*[fp(lw.ffn_norm_w) for lw in e._layers])
        self._ca_pan = (cf*L)(*[fp(getattr(lw,'post_attn_norm_w',None)) for lw in e._layers])
        self._ca_pfw = (cf*L)(*[fp(getattr(lw,'post_ffw_norm_w',None)) for lw in e._layers])
        self._ca_pn = (cf*L)(*[fp(getattr(lw,'post_norm_w',None)) for lw in e._layers])
        self._ca_ls = (cf*L)(*[fp(getattr(lw,'layer_scale',None)) for lw in e._layers])
        self._ca_qn = (cf*L)(*[fp(getattr(lw,'q_norm_w',None)) for lw in e._layers])
        self._ca_kn = (cf*L)(*[fp(getattr(lw,'k_norm_w',None)) for lw in e._layers])
        self._ca_pr = (cf*L)(*[fp(getattr(lw,'per_layer_proj',None)) for lw in e._layers])
        self._ca_ig = (cf*L)(*[fp(getattr(lw,'per_layer_inp_gate',None)) for lw in e._layers])

        # Quantized weight pointer arrays
        self._ca_wq = (cu*L)(*[up(getattr(lw,'attn_q_raw',None)) for lw in e._layers])
        self._ca_wk = (cu*L)(*[up(getattr(lw,'attn_k_raw',None)) for lw in e._layers])
        self._ca_wv = (cu*L)(*[up(getattr(lw,'attn_v_raw',None)) for lw in e._layers])
        self._ca_wo = (cu*L)(*[up(getattr(lw,'attn_out_raw',None)) for lw in e._layers])
        self._ca_wg = (cu*L)(*[up(getattr(lw,'ffn_gate_raw',None)) for lw in e._layers])
        self._ca_wu = (cu*L)(*[up(getattr(lw,'ffn_up_raw',None)) for lw in e._layers])
        self._ca_wd = (cu*L)(*[up(getattr(lw,'ffn_down_raw',None)) for lw in e._layers])

        def gv(lw, attr):
            v = getattr(lw, attr, None)
            if v is None: return 0
            return v.value if hasattr(v,'value') else int(v)

        # Dimension and quant type arrays
        self._ca_nq = (ci*L)(*[gv(lw,'attn_q_nr') for lw in e._layers])
        self._ca_nk = (ci*L)(*[gv(lw,'attn_k_nr') for lw in e._layers])
        self._ca_nv = (ci*L)(*[gv(lw,'attn_v_nr') for lw in e._layers])
        self._ca_no = (ci*L)(*[gv(lw,'attn_out_nr') for lw in e._layers])
        self._ca_ng = (ci*L)(*[gv(lw,'ffn_gate_nr') for lw in e._layers])
        self._ca_nu = (ci*L)(*[gv(lw,'ffn_up_nr') for lw in e._layers])
        self._ca_nd = (ci*L)(*[gv(lw,'ffn_down_nr') for lw in e._layers])
        self._ca_qq = (ci*L)(*[gv(lw,'attn_q_qt') for lw in e._layers])
        self._ca_kq = (ci*L)(*[gv(lw,'attn_k_qt') for lw in e._layers])
        self._ca_vq = (ci*L)(*[gv(lw,'attn_v_qt') for lw in e._layers])
        self._ca_oq = (ci*L)(*[gv(lw,'attn_out_qt') for lw in e._layers])
        self._ca_gq = (ci*L)(*[gv(lw,'ffn_gate_qt') for lw in e._layers])
        self._ca_uq = (ci*L)(*[gv(lw,'ffn_up_qt') for lw in e._layers])
        self._ca_dq = (ci*L)(*[gv(lw,'ffn_down_qt') for lw in e._layers])

        # Per-layer metadata
        hd = []; sw = []; kvi = []; rd = []; fb = []
        for lw in e._layers:
            hd.append(getattr(lw,'gemma4_head_dim',e.head_dim))
            sw.append(1 if getattr(lw,'gemma4_is_swa',False) else 0)
            kvi.append(getattr(lw,'gemma4_kv_idx',0))
            rd.append(getattr(lw,'gemma4_rope_dim',e.head_dim))
            fb.append(getattr(lw,'gemma4_freq_base',1000000.0))
        self._ca_hd = (ci*L)(*hd); self._ca_sw = (ci*L)(*sw)
        self._ca_kv = (ci*L)(*kvi); self._ca_rd = (ci*L)(*rd)
        self._ca_fb = (ctypes.c_float*L)(*fb)

    def init_weights(self, *a): pass  # handled by engine
    def init_pointers(self, *a): pass
    def init_buffers(self): pass
    def reset_state(self): pass

    def forward(self, token_id):
        e = self.engine; self._build_c()
        L=e.n_layers; N=e.n_embd; V=e.vocab_size; NH=e.n_head; NKH=e.n_kv_head
        S = int(max(int(e.n_embd), int(e.n_head)*512, int(e.n_ff)//2, int(self.per_layer_dim), 8192))
        ws = np.zeros(14*S, dtype=np.float32)
        if isinstance(token_id, (list, np.ndarray)):
            tid = np.array([[int(token_id[0])]], dtype=np.int32)
        else:
            tid = np.array([[int(token_id)]], dtype=np.int32)
        ci = ctypes.c_int; cf = ctypes.POINTER(ctypes.c_float); cu = ctypes.POINTER(ctypes.c_uint8)
        cpi = ctypes.POINTER(ci); cpf = ctypes.POINTER(cf); cpu = ctypes.POINTER(cu)
        ce = e._cengine
        fn = ce.gemma4_forward_c

        # Set argtypes once
        if not hasattr(fn, '_at_set'):
            # Build argtypes carefully matching C function signature
            at = []
            # tokens, B
            at += [cpi, ci]
            # L, N, NH, NKH, V, PL, SW, FF_half (8 ints)
            at += [ci]*8
            # eps, logit_cap (2 floats)
            at += [ctypes.c_float]*2
            # emb
            at += [cf]
            # w_out, qt_out, nr_out, nc_out
            at += [cu, ci, ci, ci]
            # out_norm_w
            at += [cf]
            # 10 F32 pointer arrays (norm, proj, etc.)
            at += [cpf]*10
            # 7 uint8 pointer arrays + 7 nr int arrays + 7 qt int arrays (interleaved per weight)
            # wq, q_nr, q_qt, wk, k_nr, k_qt, wv, v_nr, v_qt, wo, o_nr, o_qt,
            # wg, g_nr, g_qt, wu, u_nr, u_qt, wd, d_nr, d_qt
            at += [cpu, cpi, cpi] * 7
            # head_dims, is_swa, kv_idx, rope_dim (4 int arrays)
            at += [cpi]*4
            # freq_base (float array) — single float*, not float**
            at += [cf]
            # kv_k, kv_v, kv_lens, max_ctx
            at += [cf, cf, cpi, ci]
            # cos_rope, sin_rope
            at += [cf, cf]
            # logits, ws
            at += [cf, cf]
            fn.argtypes = at
            fn.restype = None
            fn._at_set = True
        # RoPE table
        mpos = int(max(8192, int(self.sliding_window)*2))
        cos_t = np.zeros(mpos*512, dtype=np.float32)
        sin_t = np.zeros(mpos*512, dtype=np.float32)
        rope_base = float(self.rope_freq_base)
        for p in range(mpos):
            for j in range(min(256, int(e.head_dim))):
                hp = j
                t = float(p) / (rope_base ** (2.0*hp/512.0))
                cos_t[p*512+hp] = np.cos(t); sin_t[p*512+hp] = np.sin(t)

        # Handle both quantized (C path) and F32 output weights
        if hasattr(e, '_out_raw') and e._out_raw is not None:
            out_w_arg = e._out_raw         # uint8_t* for quantized weights
        elif hasattr(e, '_out_f32'):
            # Cast F32 float* to uint8_t* for C function compatibility
            out_w_arg = e._out_f32.ctypes.data_as(cu)
        else:
            out_w_arg = cu()
        out_qt_arg = e._out_qt if hasattr(e, '_out_qt') else ci(0)
        out_nr_arg = e._out_nr if hasattr(e, '_out_nr') else ci(N)
        out_nc_arg = e._out_nc if hasattr(e, '_out_nc') else ci(V)
        
        fn(tid.ctypes.data_as(ctypes.POINTER(ci)), ci(1),
           ci(L), ci(N), ci(NH), ci(NKH), ci(V), ci(self.per_layer_dim),
           ci(self.sliding_window), ci(e.n_ff//2),
           e._eps_f, ctypes.c_float(self.logit_cap),
           e.emb.ctypes.data_as(cf),
           out_w_arg, out_qt_arg, out_nr_arg, out_nc_arg,
           e._out_norm_w.ctypes.data_as(cf),
           self._ca_attn, self._ca_ffn, self._ca_pan, self._ca_pfw,
           self._ca_pn, self._ca_ls, self._ca_qn, self._ca_kn,
           self._ca_pr, self._ca_ig,
           self._ca_wq, self._ca_nq, self._ca_qq,
           self._ca_wk, self._ca_nk, self._ca_kq,
           self._ca_wv, self._ca_nv, self._ca_vq,
           self._ca_wo, self._ca_no, self._ca_oq,
           self._ca_wg, self._ca_ng, self._ca_gq,
           self._ca_wu, self._ca_nu, self._ca_uq,
           self._ca_wd, self._ca_nd, self._ca_dq,
           self._ca_hd, self._ca_sw, self._ca_kv,
           self._ca_rd, self._ca_fb,
           e.kv_k.ctypes.data_as(cf), e.kv_v.ctypes.data_as(cf),
           e.kv_len.ctypes.data_as(ctypes.POINTER(ci)), ci(e.kv_k.shape[1]),
           cos_t.ctypes.data_as(cf), sin_t.ctypes.data_as(cf),
           e._p_logits, ws.ctypes.data_as(cf))
        e.pos += 1
        return e._logits
