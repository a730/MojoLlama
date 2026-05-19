#!/usr/bin/env python3
"""
Hot-path profiler for MojoLlama MoE engine.

Instruments the TurboEngineV7MoE.forward() method with perf_counter()
timing around each component within the per-layer loop. Reports per-token
and per-layer breakdown.
"""
import sys, os, time, ctypes
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels'))
from turbo_engine_v7_moe import TurboEngineV7MoE

MODEL = '/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'

# ---------- Timing accumulator ----------
TIMERS = {}  # component_name -> list of elapsed seconds per call

def tic(comp):
    """Start a timer for a component. Returns a callback that accumulates."""
    t0 = time.perf_counter()
    def stop():
        dt = time.perf_counter() - t0
        TIMERS.setdefault(comp, []).append(dt)
    return stop


class ProfiledTurboEngineV7MoE(TurboEngineV7MoE):
    """Subclass that adds Python-level perf_counter() timing around each component."""

    def forward(self, token_id):
        b_x = self._x
        b_r = self._residual
        b_qn = self._x_norm
        b_q = self._q
        b_k = self._k
        b_v = self._v
        b_att = self._att_out
        b_gate = self._gate
        b_up = self._up
        b_silu = self._silu_gate
        b_ffn = self._ffn_out
        b_oproj = self._o_proj
        N = self.n_embd
        NH = self.n_head
        NKH = self.n_kv_head * self.head_dim
        HD = self.head_dim
        FF = self.n_ff
        L = self.n_layers
        kern = self._kern
        simd = self._simd
        eps_f = self._eps_f
        p_x = self._p_x
        p_xn = self._p_x_norm
        p_r = self._p_residual
        p_q = self._p_q
        p_k = self._p_k
        p_v = self._p_v
        p_att = self._p_att_out
        p_gate = self._p_gate
        p_up = self._p_up
        p_silu = self._p_silu_gate
        p_oproj = self._p_o_proj
        p_ffn = self._p_ffn
        p_logits = self._p_logits

        # --- Embedding lookup ---
        tok = tic('emb_lookup')
        np.copyto(b_x, self.emb[token_id])
        tok()

        for i in range(L):
            lw = self._layers[i]

            # Residual copy
            np.copyto(b_r, b_x)

            # --- RMS norm 1 (pre-attention) ---
            t_rms1 = tic('rms_norm_pre_attn')
            simd.rms_norm(
                p_xn, p_x,
                lw.attn_norm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                N, eps_f,
            )
            t_rms1()

            # --- QKV matmuls ---
            t_qkv = tic('attn_qkv')
            if lw.attn_q_use_c and lw.attn_k_use_c and lw.attn_v_use_c:
                kern.batch_qkv_omp(
                    lw.attn_q_raw, lw.attn_k_raw, lw.attn_v_raw,
                    p_xn, p_q, p_k, p_v,
                    lw.attn_q_nr, lw.attn_k_nr, lw.attn_v_nr,
                    lw.attn_q_nc,
                    lw.attn_q_qt, lw.attn_k_qt, lw.attn_v_qt,
                )
            else:
                if lw.attn_q_use_c:
                    kern.quant_matmul_omp(
                        lw.attn_q_raw, p_xn, p_q,
                        lw.attn_q_nr, lw.attn_q_nc, lw.attn_q_qt,
                    )
                if lw.attn_k_use_c:
                    kern.quant_matmul_omp(
                        lw.attn_k_raw, p_xn, p_k,
                        lw.attn_k_nr, lw.attn_k_nc, lw.attn_k_qt,
                    )
                if lw.attn_v_use_c:
                    kern.quant_matmul_omp(
                        lw.attn_v_raw, p_xn, p_v,
                        lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt,
                    )
            t_qkv()

            # --- Q/K norm ---
            t_qk_norm = tic('attn_qk_norm')
            if lw.has_q_norm:
                q_2d = b_q.reshape(NH, HD)
                q_rms = np.sqrt(np.mean(q_2d * q_2d, axis=1, keepdims=True) + self.eps)
                q_2d[:] = q_2d / q_rms * lw.q_norm_w.reshape(1, HD)
            if lw.has_k_norm:
                k_2d = b_k[:NKH].reshape(self.n_kv_head, HD)
                k_rms = np.sqrt(np.mean(k_2d * k_2d, axis=1, keepdims=True) + self.eps)
                k_2d[:] = k_2d / k_rms * lw.k_norm_w.reshape(1, HD)
            t_qk_norm()

            # --- RoPE ---
            t_rope = tic('rope')
            b_q[:] = self._apply_rope_fast(b_q, self.pos, NH)
            b_k[:] = self._apply_rope_fast(b_k, self.pos, self.n_kv_head)
            t_rope()

            # --- KV cache store ---
            t_kv_store = tic('kv_cache_store')
            self.kv_k[i, self.kv_len[i], :NKH] = b_k[:NKH]
            self.kv_v[i, self.kv_len[i], :NKH] = b_v[:NKH]
            t_kv_store()

            # --- Attention (GQA) ---
            t_attn = tic('attention')
            seq_len = self.kv_len[i] + 1
            if self._gqa_attn is not None and seq_len <= 4096:
                self._gqa_attn.gqa_attention_decode(
                    b_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    self.kv_k[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    self.kv_v[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    b_att.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int(seq_len),
                    ctypes.c_int(NH),
                    ctypes.c_int(self.n_kv_head),
                    ctypes.c_int(HD),
                )
            else:
                k_cache = self.kv_k[i, :seq_len].reshape(seq_len, self.n_kv_head, HD)
                v_cache = self.kv_v[i, :seq_len].reshape(seq_len, self.n_kv_head, HD)
                q_2d = b_q.reshape(NH, HD)
                nk = self.n_kv_head
                gqa = self._gqa_rep
                q_g = q_2d.reshape(nk, gqa, HD)
                k_T = k_cache.transpose(1, 0, 2)
                scores = np.einsum('khd,ksd->khs', q_g, k_T) / np.sqrt(float(HD))
                scores = scores.reshape(NH, seq_len)
                scores -= np.max(scores, axis=1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= np.sum(scores, axis=1, keepdims=True)
                v_T = v_cache.transpose(1, 0, 2)
                att = np.einsum('khs,ksd->khd', scores.reshape(nk, gqa, seq_len), v_T)
                b_att[:] = att.reshape(-1)
            self.kv_len[i] += 1
            t_attn()

            # --- O projection ---
            t_o = tic('attn_out_proj')
            if lw.attn_out_use_c:
                kern.quant_matmul_omp(
                    lw.attn_out_raw, p_att, p_oproj,
                    lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt,
                )
                b_x[:N] = b_r[:N] + b_oproj[:N]
            else:
                b_x[:N] = b_r[:N] + (lw.attn_out_f32 @ b_att)[:N]
            t_o()

            # --- FFN ---
            np.copyto(b_r, b_x)

            # --- RMS norm 2 (pre-FFN) ---
            t_rms2 = tic('rms_norm_pre_ffn')
            simd.rms_norm(
                p_xn, p_x,
                lw.ffn_norm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                N, eps_f,
            )
            t_rms2()

            # --- MoE FFN ---
            if self.is_moe and self._moe_layers:
                self._forward_moe_profiled(i, p_xn, b_ffn, b_r, N)
            else:
                # Dense FFN (not used for MoE models, but keep for completeness)
                t_dense = tic('ffn_dense')
                if lw.ffn_gate_use_c and lw.ffn_up_use_c:
                    kern.batch_gate_up_omp(
                        lw.ffn_gate_raw, lw.ffn_up_raw, p_xn, p_gate, p_up,
                        lw.ffn_gate_nr, lw.ffn_up_nr,
                        lw.ffn_gate_nc, lw.ffn_gate_qt, lw.ffn_up_qt,
                    )
                else:
                    if lw.ffn_gate_use_c:
                        kern.quant_matmul_omp(
                            lw.ffn_gate_raw, p_xn, p_gate,
                            lw.ffn_gate_nr, lw.ffn_gate_nc, lw.ffn_gate_qt,
                        )
                    if lw.ffn_up_use_c:
                        kern.quant_matmul_omp(
                            lw.ffn_up_raw, p_xn, p_up,
                            lw.ffn_up_nr, lw.ffn_up_nc, lw.ffn_up_qt,
                        )
                simd.silu(p_silu, p_gate, ctypes.c_int(FF))
                b_silu[:FF] *= b_up[:FF]
                if lw.ffn_down_use_c:
                    kern.quant_matmul_omp(
                        lw.ffn_down_raw, p_silu, p_ffn,
                        lw.ffn_down_nr, lw.ffn_down_nc, lw.ffn_down_qt,
                    )
                    b_x[:N] = b_r[:N] + b_ffn[:N]
                else:
                    b_x[:N] = b_r[:N] + (lw.ffn_down_f32 @ b_silu[:lw.ffn_down_nc.value])[:N]
                t_dense()

        # --- Final norm ---
        t_fn = tic('final_norm')
        simd.rms_norm(
            p_xn, p_x,
            self._out_norm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, eps_f,
        )
        t_fn()

        # --- Output projection ---
        t_out = tic('output_proj')
        if self._out_use_c:
            kern.quant_matmul_omp(
                self._out_raw, p_xn, p_logits,
                self._out_nr, self._out_nc, self._out_qt,
            )
        else:
            self._logits[:] = self._out_f32 @ b_qn
        t_out()

        self.pos += 1
        return self._logits

    def _forward_moe_profiled(self, layer, p_xn, b_ffn_out, b_residual, N):
        """Profiled MoE FFN dispatch with separate router/softmax/expert timing."""
        me = self._moe_layers[layer]
        n_exp = me.n_experts
        top_k = self.n_experts_per_tok
        FF = me.n_ff_expert
        kern = self._kern

        # --- Router ---
        t_router = tic('moe_router')
        if me.router_raw is not None:
            kern.quant_matmul_omp(
                me.router_raw, p_xn, self._p_moe_router,
                me.router_nr, me.router_nc, me.router_qt,
            )
        else:
            np.copyto(
                self._moe_router_scores[:n_exp],
                (me.router_f32 @ self._x_norm)[:n_exp],
            )
        t_router()

        # --- Softmax + Top-K ---
        t_topk = tic('moe_softmax_topk')
        scores = self._moe_router_scores[:n_exp]
        scores -= np.max(scores)
        np.exp(scores, out=scores)
        scores /= np.sum(scores)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_weights = scores[top_indices]
        top_weights /= np.sum(top_weights)
        t_topk()

        # --- Fused MoE dispatch (gate/up/down matmuls + silu + combine) ---
        t_moe = tic('moe_fused_experts')
        gate_ptrs = me.gate_ptrs_arr
        up_ptrs = me.up_ptrs_arr
        down_ptrs = me.down_ptrs_arr

        top_idx_arr = np.ascontiguousarray(top_indices.astype(np.int32))
        top_wt_arr = np.ascontiguousarray(top_weights.astype(np.float32))

        self._moe_combined[:] = 0.0
        kern.moe_forward_omp(
            gate_ptrs, up_ptrs, down_ptrs,
            p_xn,
            ctypes.c_int(FF), ctypes.c_int(N),
            me.gate_qt, me.up_qt, me.down_qt,
            top_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            top_wt_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(top_k),
            self._p_moe_combined,
            self._moe_prealloc_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._moe_prealloc_q8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )

        b_ffn_out[:N] = self._moe_combined[:N]
        t_moe()


def main():
    print("Loading model...", flush=True)
    t0 = time.perf_counter()
    e = ProfiledTurboEngineV7MoE(MODEL, 32)
    print(f"Load time: {time.perf_counter()-t0:.1f}s", flush=True)

    L = e.n_layers
    N = e.n_embd
    NH = e.n_head
    NKH = e.n_kv_head
    HD = e.head_dim
    NE = e.n_experts
    NK = e.n_experts_per_tok
    moe_int = e.n_ff_expert if hasattr(e, 'n_ff_expert') else e.n_ff
    print(f"Model: {L}L/{N}D/{NH}H/{NKH}KV | MoE {NE}x{NK} | int={moe_int} | V={e.vocab_size}", flush=True)

    # Warmup: 5 tokens
    print("\nWarmup (5 tokens)...", flush=True)
    e.reset()
    logits = e.forward(1)  # BOS token
    tok = int(np.argmax(np.nan_to_num(logits, nan=-1e10)))
    for _ in range(4):
        logits = e.forward(tok)
        tok = int(np.argmax(np.nan_to_num(logits, nan=-1e10)))

    # Clear accumulators
    TIMERS.clear()
    e.reset()

    # Profiled run: 20 tokens
    N_TOKENS = 20
    print(f"Profiling {N_TOKENS} tokens...", flush=True)
    logits = e.forward(1)  # BOS
    tok = int(np.argmax(np.nan_to_num(logits, nan=-1e10)))
    for i in range(N_TOKENS - 1):
        logits = e.forward(tok)
        tok = int(np.argmax(np.nan_to_num(logits, nan=-1e10)))

    # Aggregate
    n_layers = L
    total_calls_per_token = {}
    total_ms_per_token = {}
    grand_total_ms = 0.0
    for comp, times in TIMERS.items():
        total_s = sum(times)
        total_ms = total_s * 1000.0
        grand_total_ms += total_ms
        total_calls_per_token[comp] = len(times)
        total_ms_per_token[comp] = total_ms / N_TOKENS

    grand_total_per_token = grand_total_ms / N_TOKENS

    # Group into categories for the final report
    cat_map = {
        'emb_lookup':       'Emb lookup',
        'rms_norm_pre_attn': 'RMS norms',
        'rms_norm_pre_ffn':  'RMS norms',
        'final_norm':       'RMS norms',
        'attn_qkv':         'Attention QKV',
        'attn_qk_norm':     'Attention QKV',
        'rope':             'Attention QKV',
        'kv_cache_store':   'Attention QKV',
        'attention':        'Attention QKV',
        'attn_out_proj':    'Attention QKV',
        'moe_router':       'Router',
        'moe_softmax_topk': 'Router',
        'moe_fused_experts':'MoE FFN gate/up/down',
        'output_proj':      'Output proj',
    }
    cat_times = {}
    for comp, t in total_ms_per_token.items():
        cat = cat_map.get(comp, 'Other')
        cat_times[cat] = cat_times.get(cat, 0.0) + t

    # Print detailed per-component
    print(f"\n{'='*72}")
    print(f"  HOT-PATH PROFILE: {N_TOKENS} tokens x {n_layers} layers = {N_TOKENS*n_layers} layer-tokens")
    print(f"{'='*72}")
    print(f"  {'COMPONENT':30s} {'ms/tok':>10s} {'%':>8s} {'calls/tok':>10s}")
    print(f"  {'─'*60}")
    for comp in sorted(total_ms_per_token.keys(), key=lambda c: total_ms_per_token[c], reverse=True):
        ms = total_ms_per_token[comp]
        pct = ms / grand_total_per_token * 100 if grand_total_per_token > 0 else 0
        ct = total_calls_per_token[comp] / N_TOKENS
        label = comp.replace('_', ' ').title()
        print(f"  {comp:30s} {ms:9.2f}ms {pct:7.1f}% {ct:9.1f}")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':30s} {grand_total_per_token:9.2f}ms {'100.0%':>8s}")

    # Print category breakdown
    print(f"\n  {'─'*60}")
    print(f"  {'CATEGORY':30s} {'ms/tok':>10s} {'%':>8s}")
    print(f"  {'─'*60}")
    cat_order = ['Attention QKV', 'MoE FFN gate/up/down', 'Router', 'Output proj', 'RMS norms', 'Emb lookup', 'Other']
    for cat in cat_order:
        if cat in cat_times:
            ms = cat_times[cat]
            pct = ms / grand_total_per_token * 100
            print(f"  {cat:30s} {ms:9.2f}ms {pct:7.1f}%")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':30s} {grand_total_per_token:9.2f}ms {1000.0/grand_total_per_token:7.1f} tok/s")
    print(f"{'='*72}")

    # Per-layer detail (top components)
    print(f"\nPer-layer detail (ms/layer, avg over {N_TOKENS} tokens):")
    comps_by_layer = {}
    for comp, times in TIMERS.items():
        # Group into batches of n_layers * N_TOKENS
        calls_per_token_per_layer = len(times) // (n_layers * N_TOKENS)
        if calls_per_token_per_layer == 0:
            continue
        layers_per_token_tuples = []
        idx = 0
        for _ in range(N_TOKENS):
            for layer_i in range(n_layers):
                for _ in range(calls_per_token_per_layer):
                    layers_per_token_tuples.append((layer_i, times[idx]))
                    idx += 1
        # Average per layer across tokens
        layer_sums = {}
        layer_counts = {}
        for layer_i, dt in layers_per_token_tuples:
            layer_sums[layer_i] = layer_sums.get(layer_i, 0.0) + dt
            layer_counts[layer_i] = layer_counts.get(layer_i, 0) + 1
        avg_per_layer = {}
        for li in range(n_layers):
            if li in layer_sums:
                avg_per_layer[li] = layer_sums[li] / N_TOKENS * 1000.0  # ms/layer/token
        comps_by_layer[comp] = avg_per_layer

    # Show layer-variance for top 3 components
    top_comps = sorted(total_ms_per_token.keys(), key=lambda c: total_ms_per_token[c], reverse=True)[:4]
    for comp in top_comps:
        if comp in comps_by_layer and comps_by_layer[comp]:
            vals = comps_by_layer[comp]
            layer_ms = [vals.get(li, 0) for li in range(n_layers)]
            min_v = min(layer_ms)
            max_v = max(layer_ms)
            avg_v = sum(layer_ms) / len(layer_ms)
            print(f"  {comp:25s}: min={min_v:.3f}ms  max={max_v:.3f}ms  avg={avg_v:.3f}ms  "
                  f"range/avg={max_v/min_v:.1f}x" if min_v > 0 else
                  f"  {comp:25s}: min={min_v:.3f}ms  max={max_v:.3f}ms  avg={avg_v:.3f}ms")

    print(f"\nDone. (grand total = {grand_total_ms:.1f}ms for {N_TOKENS} tokens)")


if __name__ == '__main__':
    main()
