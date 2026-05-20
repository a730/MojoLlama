#!/usr/bin/env python3
"""
Gemma 4 forward pass — per-layer projections, GeGLU, sliding window attention,
Q/K norms, shared KV cache groups, logit softcapping.

Integrates with TurboEngineV7MoE via the ArchitectureForwardPass ABC.

Architecture per-layer flow:
1. RMS norm on input x (attn_norm.weight)
2. Per-layer signal from normed x: proj(x) → [256], inp_gate @ [256] → [1536]
3. Q, K, V projections from normed x
4. Q/K norm (attn_q_norm.weight, attn_k_norm.weight)
5. RoPE on Q and K
6. Sliding window / full attention
7. O projection
8. Combine: x = x + rms_norm(O_proj, post_attention_norm) + per_layer_gated
9. RMS norm before FFN (ffn_norm.weight — already set by engine)
10. GeGLU FFN: down(GELU(gate(x_norm)) * up(x_norm))
11. Combine: x = x + rms_norm(ffn_out, post_ffw_norm)
12. Final layer RMS norm (post_norm.weight)
13. Layer output scale multiplication
"""

import numpy as np
import ctypes
from forward.base import ArchitectureForwardPass


class ForwardGemma4(ArchitectureForwardPass):
    """Gemma 4 forward pass implementation."""

    def __init__(self, engine):
        super().__init__(engine)
        e = engine
        a = e.arch_name

        # Architecture-specific metadata (may not exist in engine._parse_metadata)
        def _get(name):
            fields = e.reader.fields
            for key, val in fields.items():
                if key == name:
                    parts = val.parts if hasattr(val, 'parts') else []
                    if len(parts) >= 1:
                        data = parts[-1]
                        if hasattr(data, '__iter__') and len(data) == 1:
                            return int(data[0])
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

        # RoPE lookup tables (lazy)
        self._cos_table = {}
        self._sin_table = {}

        # Set vocab_size from embedding (will be refined in init_weights)
        if e.vocab_size == 0 or e.vocab_size < 10000:
            e.vocab_size = 262144

    def _apply_rope_gemma(self, x, pos, n_heads, hd, freq_base):
        """Apply RoPE with caching for speed."""
        half = hd // 2
        key = (freq_base, pos, half)
        if key not in self._cos_table:
            freqs = freq_base ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
            cos_a = np.cos(pos / freqs).astype(np.float32)
            sin_a = np.sin(pos / freqs).astype(np.float32)
            self._cos_table[key] = cos_a
            self._sin_table[key] = sin_a
        cos_a = self._cos_table[key]
        sin_a = self._sin_table[key]
        x2d = x.reshape(n_heads, hd)
        out = x2d.copy()
        out[:, :half] = x2d[:, :half] * cos_a - x2d[:, half:hd] * sin_a
        out[:, half:hd] = x2d[:, half:hd] * cos_a + x2d[:, :half] * sin_a
        return out.reshape(-1)

    def init_weights(self, weights, raw_weights, weight_info, weight_qtypes):
        """Dequant token_embd.weight for embedding lookup and set vocab size."""
        e = self.engine

        emb_name = 'token_embd.weight'
        if emb_name in raw_weights or emb_name in weights:
            import gguf
            # Find the tensor in reader and dequantize
            for t in e.reader.tensors:
                if t.name == emb_name:
                    f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                    if len(t.shape) == 2:
                        in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
                        f32 = f32.reshape(out_dim, in_dim)
                        if f32.shape[1] != e.n_embd:
                            f32 = np.ascontiguousarray(f32.T)
                    e.emb = np.ascontiguousarray(f32)
                    e.vocab_size = f32.shape[0]
                    print(f"  Gemma4: embedding={e.emb.shape}, vocab={e.vocab_size}", flush=True)
                    break

        # Update vocab_size in the engine's metadata
        if hasattr(e, '_logits') and e.vocab_size != len(e._logits):
            print(f"  Gemma4: resizing logits buffer from {len(e._logits)} to {e.vocab_size}", flush=True)
            e._logits = np.zeros(e.vocab_size, dtype=np.float32)
            e._p_logits = e._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def init_pointers(self, layer_idx, pfx, lw):
        """Set Gemma 4-specific weight pointers on the layer object.

        Called BEFORE the engine's standard pointer setup in _preload_pointers,
        so standard attributes (attn_q_raw, etc.) are set after this.
        """
        e = self.engine
        weights = e.weights

        # F32 norm and projection weights (stored in e.weights since they're small)
        lw.post_attn_norm_w = weights.get(f'{pfx}.post_attention_norm.weight')
        lw.post_ffw_norm_w = weights.get(f'{pfx}.post_ffw_norm.weight')
        lw.post_norm_w = weights.get(f'{pfx}.post_norm.weight')
        proj_w = weights.get(f'{pfx}.proj.weight')
        inp_gate_w = weights.get(f'{pfx}.inp_gate.weight')
        # GGUF stores proj as [256, 1536] (in_dim=256, out_dim=1536)
        # _load_weights reshapes to [out_dim, in_dim] = [1536, 256]
        # For matmul: x (1536,) @ proj (1536, 256) → (256,) ✓
        # For inp_gate: stored as [256, 1536] after reshape
        # Need: per (256,) @ inp_gate (256, 1536) → (1536,) ✓
        if proj_w is not None:
            lw.per_layer_proj = proj_w  # [1536, 256] — keep as-is for x @ proj
        else:
            lw.per_layer_proj = None
        lw.per_layer_inp_gate = inp_gate_w  # [256, 1536] — keep as-is for per @ inp_gate
        lw.layer_scale = weights.get(f'{pfx}.layer_output_scale.weight')  # [1]

        # Determine per-layer head dimensions from weight shapes
        q_name = f'{pfx}.attn_q.weight'
        q_info = e.weight_info.get(q_name)
        if q_info is not None:
            q_nr = q_info[0]  # output dim: 2048 or 4096
            lw.gemma4_nq = q_nr
            lw.gemma4_head_dim = q_nr // e.n_head  # 256 or 512
        else:
            lw.gemma4_nq = e.n_head * e.head_dim
            lw.gemma4_head_dim = e.head_dim

        k_name = f'{pfx}.attn_k.weight'
        k_info = e.weight_info.get(k_name)
        if k_info is not None:
            lw.gemma4_nk = k_info[0]  # total K output dim
        else:
            lw.gemma4_nk = e.n_kv_head * e.head_dim

        v_name = f'{pfx}.attn_v.weight'
        v_info = e.weight_info.get(v_name)
        if v_info is not None:
            lw.gemma4_nv = v_info[0]  # total V output dim
        else:
            lw.gemma4_nv = e.n_kv_head * e.head_dim

        # Determine if this is a SWA or full attention layer
        hd = lw.gemma4_head_dim
        if hd >= 512:
            lw.gemma4_is_swa = False
            lw.gemma4_rope_dim = self.rope_dim      # 512
            lw.gemma4_freq_base = self.rope_freq_base  # 1e6
        else:
            lw.gemma4_is_swa = True
            lw.gemma4_rope_dim = self.rope_dim_swa        # 256
            lw.gemma4_freq_base = self.rope_freq_base_swa  # 1e4

        # Shared KV group index
        lw.gemma4_kv_idx = (layer_idx // self.shared_kv) * self.shared_kv

    def init_buffers(self):
        """Allocate per-layer intermediate buffer."""
        e = self.engine
        self._per_layer = np.zeros(self.per_layer_dim, dtype=np.float32)
        self._per_gated = np.zeros(e.n_embd, dtype=np.float32)

    def reset_state(self):
        """No state to reset beyond what the engine does."""
        pass

    def forward(self, token_id):
        """Gemma 4 single-token forward pass."""
        e = self.engine
        b_x = e._x
        b_r = e._residual
        b_xn = e._x_norm
        b_q = e._q
        b_k = e._k
        b_v = e._v
        b_att = e._att_out
        b_gate = e._gate
        b_up = e._up
        b_silu = e._silu_gate
        b_ffn = e._ffn_out
        b_oproj = e._o_proj
        b_logits = e._logits
        N = e.n_embd
        NH = e.n_head
        NKH = e.n_kv_head
        L = e.n_layers
        PL = self.per_layer_dim
        SW = self.sliding_window
        FF_half = e.n_ff // 2  # 6144 (gate/up intermediate dim)

        kern = e._kern
        simd = e._simd
        eps_f = e._eps_f
        cf = ctypes.POINTER(ctypes.c_float)
        cu = ctypes.POINTER(ctypes.c_uint8)
        ci = ctypes.c_int

        p_x = e._p_x
        p_xn = e._p_x_norm
        p_r = e._p_residual
        p_q = e._p_q
        p_k = e._p_k
        p_v = e._p_v
        p_att = e._p_att_out
        p_gate = e._p_gate
        p_up = e._p_up
        p_silu = e._p_silu_gate
        p_oproj = e._p_o_proj
        p_ffn = e._p_ffn
        p_logits = e._p_logits

        per = self._per_layer
        per_gated = self._per_gated

        # Embedding lookup
        np.copyto(b_x, e.emb[token_id])

        pos = e.pos

        for i in range(L):
            lw = e._layers[i]
            hd = lw.gemma4_head_dim
            nq = lw.gemma4_nq
            nk = lw.gemma4_nk
            nv = lw.gemma4_nv
            rope_hd = lw.gemma4_rope_dim
            freq_base = lw.gemma4_freq_base
            is_swa = lw.gemma4_is_swa
            kv_idx = lw.gemma4_kv_idx

            # NaN clamp
            np.clip(b_x, -1000.0, 1000.0, out=b_x)

            # Residual copy
            np.copyto(b_r, b_x)

            # ── Step 1: Pre-attention RMS norm ──
            simd.rms_norm(p_xn, p_x,
                          lw.attn_norm_w.ctypes.data_as(cf),
                          N, eps_f)

            # ── Step 2: Per-layer signal from normed x ──
            np.copyto(per, b_xn @ lw.per_layer_proj)
            np.copyto(per_gated, per @ lw.per_layer_inp_gate)

            # ── Steps 3-7: QKV projections, Q/K norm, RoPE, attention ──
            kv_lw = e._layers[kv_idx]  # KV-shared layer (cache only)

            # Q projection
            if lw.attn_q_use_c:
                kern.quant_matmul_omp(
                    lw.attn_q_raw, p_xn, p_q,
                    ci(nq), ci(N), lw.attn_q_qt)
            else:
                b_q[:nq] = lw.attn_q_f32 @ b_xn

            # K projection (current layer's weights, shared KV cache)
            if lw.attn_k_use_c:
                kern.quant_matmul_omp(
                    lw.attn_k_raw, p_xn, p_k,
                    ci(nk), ci(N), lw.attn_k_qt)
            else:
                b_k[:nk] = lw.attn_k_f32 @ b_xn

            # V projection (current layer's weights, shared KV cache)
            if lw.attn_v_use_c:
                kern.quant_matmul_omp(
                    lw.attn_v_raw, p_xn, p_v,
                    ci(nv), ci(N), lw.attn_v_qt)
            else:
                b_v[:nv] = lw.attn_v_f32 @ b_xn

            # Q/K RMS norm
            if lw.has_q_norm:
                q_2d = b_q[:nq].reshape(NH, hd)
                q_rms = np.sqrt(np.mean(q_2d * q_2d, axis=1, keepdims=True) + e.eps)
                q_2d[:] = q_2d / q_rms * lw.q_norm_w.reshape(1, hd)
            if lw.has_k_norm:
                k_2d = b_k[:nk].reshape(NKH, hd)
                k_rms = np.sqrt(np.mean(k_2d * k_2d, axis=1, keepdims=True) + e.eps)
                k_2d[:] = k_2d / k_rms * lw.k_norm_w.reshape(1, hd)

            # RoPE
            b_q[:nq] = self._apply_rope_gemma(b_q[:nq], pos, NH, rope_hd, freq_base)
            b_k[:nk] = self._apply_rope_gemma(b_k[:nk], pos, NKH, rope_hd, freq_base)

            # KV cache (shared KV groups)
            if is_swa:
                k_pos = e.kv_len[kv_idx] % SW
                e.kv_k[kv_idx, k_pos, :nk] = b_k[:nk]
                e.kv_v[kv_idx, k_pos, :nv] = b_v[:nv]
            else:
                k_pos = e.kv_len[kv_idx]
                if k_pos < e.kv_k.shape[1]:
                    e.kv_k[kv_idx, k_pos, :nk] = b_k[:nk]
                    e.kv_v[kv_idx, k_pos, :nv] = b_v[:nv]

            # Python attention fallback (GQA)
            seq_len = e.kv_len[kv_idx] + 1
            gqa = NH // NKH
            q_2d = b_q[:nq].reshape(NH, hd)

            if is_swa:
                if seq_len <= SW:
                    k_cache = e.kv_k[kv_idx, :seq_len, :nk].reshape(seq_len, NKH, hd)
                    v_cache = e.kv_v[kv_idx, :seq_len, :nv].reshape(seq_len, NKH, hd)
                else:
                    start = (pos + 1) % SW
                    idx = np.arange(SW) if start == 0 else np.concatenate([np.arange(start, SW), np.arange(start)])
                    k_cache = e.kv_k[kv_idx, idx[:SW], :nk].reshape(SW, NKH, hd)
                    v_cache = e.kv_v[kv_idx, idx[:SW], :nv].reshape(SW, NKH, hd)
            else:
                seq_max = min(seq_len, e.kv_k.shape[1])
                k_cache = e.kv_k[kv_idx, :seq_max, :nk].reshape(seq_max, NKH, hd)
                v_cache = e.kv_v[kv_idx, :seq_max, :nv].reshape(seq_max, NKH, hd)

            nk_heads = NKH
            q_g = q_2d.reshape(nk_heads, gqa, hd)
            k_T = k_cache.transpose(1, 0, 2)
            v_T = v_cache.transpose(1, 0, 2)
            scores = np.einsum('khd,ksd->khs', q_g, k_T) / np.sqrt(float(hd))
            scores = scores.reshape(NH, scores.shape[-1])
            scores -= np.max(scores, axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= np.sum(scores, axis=1, keepdims=True)
            att = np.einsum('khs,ksd->khd',
                            scores.reshape(nk_heads, gqa, scores.shape[-1]),
                            v_T)
            b_att[:nq] = att.reshape(-1)

            e.kv_len[kv_idx] += 1

            # ── Step 7 (cont): O projection ──
            if lw.attn_out_use_c:
                kern.quant_matmul_omp(
                    lw.attn_out_raw, p_att, p_oproj,
                    lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt)
            else:
                b_oproj[:N] = lw.attn_out_f32 @ b_att[:nq]

            # ── Step 8: Combine with post-attention norm on O_proj ──
            if lw.post_attn_norm_w is not None:
                simd.rms_norm(p_xn, p_oproj,
                              lw.post_attn_norm_w.ctypes.data_as(cf),
                              N, eps_f)
                b_x[:N] = b_r[:N] + b_xn[:N] + per_gated[:N]
            else:
                b_x[:N] = b_r[:N] + b_oproj[:N] + per_gated[:N]

            # ── Step 9: Pre-FFN RMS norm ──
            np.clip(b_x, -1000.0, 1000.0, out=b_x)
            np.copyto(b_r, b_x)
            simd.rms_norm(p_xn, p_x,
                          lw.ffn_norm_w.ctypes.data_as(cf),
                          N, eps_f)

            # ── Step 10: GeGLU FFN ──
            if lw.ffn_gate_use_c and lw.ffn_up_use_c:
                kern.batch_gate_up_omp(
                    lw.ffn_gate_raw, lw.ffn_up_raw, p_xn, p_gate, p_up,
                    lw.ffn_gate_nr, lw.ffn_up_nr,
                    lw.ffn_gate_nc, lw.ffn_gate_qt, lw.ffn_up_qt)
            else:
                if lw.ffn_gate_use_c:
                    kern.quant_matmul_omp(lw.ffn_gate_raw, p_xn, p_gate,
                                           lw.ffn_gate_nr, lw.ffn_gate_nc, lw.ffn_gate_qt)
                if lw.ffn_up_use_c:
                    kern.quant_matmul_omp(lw.ffn_up_raw, p_xn, p_up,
                                           lw.ffn_up_nr, lw.ffn_up_nc, lw.ffn_up_qt)

            # GeGLU: GELU(gate) * up — tanh approximation
            gate = b_gate[:FF_half]
            sqrt_2pi = np.sqrt(2.0 / np.pi)
            gelu_gate = 0.5 * gate * (1.0 + np.tanh(
                sqrt_2pi * (gate + 0.044715 * gate * gate * gate)))
            np.multiply(gelu_gate, b_up[:FF_half], out=b_silu[:FF_half])

            # Down projection
            if lw.ffn_down_use_c:
                kern.quant_matmul_omp(lw.ffn_down_raw, p_silu, p_ffn,
                                       lw.ffn_down_nr, lw.ffn_down_nc, lw.ffn_down_qt)
            else:
                b_ffn[:N] = lw.ffn_down_f32 @ b_silu[:FF_half]

            # ── Step 11: Add with post-FFW norm on FFN output ──
            if lw.post_ffw_norm_w is not None:
                simd.rms_norm(p_xn, p_ffn,
                              lw.post_ffw_norm_w.ctypes.data_as(cf),
                              N, eps_f)
                b_x[:N] = b_r[:N] + b_xn[:N]
            else:
                b_x[:N] = b_r[:N] + b_ffn[:N]

            # ── Step 12: Final layer RMS norm ──
            if lw.post_norm_w is not None:
                simd.rms_norm(p_xn, p_x,
                              lw.post_norm_w.ctypes.data_as(cf),
                              N, eps_f)
                np.copyto(b_x, b_xn)

            # ── Step 13: Layer output scale ──
            if lw.layer_scale is not None:
                b_x[:N] *= lw.layer_scale[0]

        # ═══════════ Final RMS norm ═══════════
        np.clip(b_x, -1000.0, 1000.0, out=b_x)
        simd.rms_norm(p_xn, p_x,
                      e._out_norm_w.ctypes.data_as(cf),
                      N, eps_f)

        # ═══════════ Output projection ═══════════
        if e._out_use_c:
            kern.quant_matmul_omp(e._out_raw, p_xn, p_logits,
                                   e._out_nr, e._out_nc, e._out_qt)
        else:
            b_logits[:] = e._out_f32 @ b_xn

        # ═══════════ Logit softcapping ═══════════
        np.clip(b_logits, -self.logit_cap, self.logit_cap, out=b_logits)
        np.tanh(b_logits / self.logit_cap, out=b_logits)
        b_logits *= self.logit_cap

        e.pos = pos + 1
        return b_logits
