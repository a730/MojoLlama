#!/usr/bin/env python3
"""Layer-by-layer hidden state comparison between step 1 and step 2."""
import sys, os, numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from turbo_engine_v7_moe import TurboEngineV7MoE

print("Loading model...", flush=True)
e = TurboEngineV7MoE('/tmp/models/gpt-oss-20b-Q4_K_M.gguf', n_threads=1)
print("Model loaded.", flush=True)

# Save original forward and monkey-patch for instrumentation
original_forward = TurboEngineV7MoE.forward

def instrumented_forward(self, token_id, _call_id=0):
    """Instrumented forward that saves hidden state at each layer."""
    import functools
    
    # Create unique call tracking
    if not hasattr(self, '_inst_call_count'):
        self._inst_call_count = 0
    self._inst_call_count += 1
    call_id = self._inst_call_count
    
    # Store hidden_state snapshots per layer
    if not hasattr(self, '_inst_hidden_states'):
        self._inst_hidden_states = {}
    self._inst_hidden_states[call_id] = []
    
    b_x = self._x; b_r = self._residual; b_qn = self._x_norm
    b_q = self._q; b_k = self._k; b_v = self._v
    b_att = self._att_out; b_gate = self._gate; b_up = self._up
    b_silu = self._silu_gate; b_ffn = self._ffn_out
    b_oproj = self._o_proj
    N = self.n_embd; NH = self.n_head; NKH = self.n_kv_head * self.head_dim; HD = self.head_dim
    FF = self.n_ff; L = self.n_layers
    kern = self._kern; simd = self._simd
    eps_f = self._eps_f
    p_x = self._p_x; p_xn = self._p_x_norm; p_r = self._p_residual
    p_q = self._p_q; p_k = self._p_k; p_v = self._p_v
    p_qk = self._p_qk
    p_att = self._p_att_out; p_gate = self._p_gate; p_up = self._p_up
    p_silu = self._p_silu_gate; p_oproj = self._p_o_proj
    p_ffn = self._p_ffn; p_logits = self._p_logits

    # Embedding lookup
    np.copyto(b_x, self.emb[token_id])
    
    # Save embedding
    if call_id <= 2:
        print(f"  [Call {call_id}] Embedding _x[:5] = {self._x[:5].copy()}", flush=True)

    cf = ctypes.POINTER(ctypes.c_float)
    cu = ctypes.POINTER(ctypes.c_uint8)
    ci = ctypes.c_int

    for i in range(L):
        lw = self._layers[i]
        is_ssm_layer = (self.layer_types is not None and self.layer_types[i] == 1)

        b_r[:] = b_x
        simd.rms_norm(p_xn, p_x,
                      lw.attn_norm_w.ctypes.data_as(cf),
                      N, eps_f)

        if is_ssm_layer and self._cengine is not None and lw.ssm_conv1d_ptr is not None:
            # SSM path (not used for gpt-oss)
            pass
        else:
            # Attention path
            if hasattr(lw, 'attn_qkv_use_c') and lw.attn_qkv_use_c:
                pass  # fused QKV
            elif hasattr(lw, 'attn_qk_use_c') and lw.attn_qk_use_c:
                kern.quant_matmul_omp(lw.attn_qk_raw, p_xn, p_qk,
                                       lw.attn_qk_nr, lw.attn_qk_nc, lw.attn_qk_qt)
                kern.quant_matmul_omp(lw.attn_v_raw, p_xn, p_v,
                                       lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)
            elif lw.attn_q_use_c and lw.attn_k_use_c and lw.attn_v_use_c:
                kern.batch_qkv_omp(
                    lw.attn_q_raw, lw.attn_k_raw, lw.attn_v_raw,
                    p_xn, p_q, p_k, p_v,
                    lw.attn_q_nr, lw.attn_k_nr, lw.attn_v_nr,
                    lw.attn_q_nc,
                    lw.attn_q_qt, lw.attn_k_qt, lw.attn_v_qt)
            else:
                if lw.attn_q_use_c:
                    kern.quant_matmul_omp(lw.attn_q_raw, p_xn, p_q, lw.attn_q_nr, lw.attn_q_nc, lw.attn_q_qt)
                if lw.attn_k_use_c:
                    kern.quant_matmul_omp(lw.attn_k_raw, p_xn, p_k, lw.attn_k_nr, lw.attn_k_nc, lw.attn_k_qt)
                if lw.attn_v_use_c:
                    kern.quant_matmul_omp(lw.attn_v_raw, p_xn, p_v, lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)

            # Q/K norm
            if lw.has_q_norm:
                q_2d = b_q.reshape(NH, HD)
                q_rms = np.sqrt(np.mean(q_2d * q_2d, axis=1, keepdims=True) + self.eps)
                q_2d[:] = q_2d / q_rms * lw.q_norm_w.reshape(1, HD)
            if lw.has_k_norm:
                k_2d = b_k[:NKH].reshape(self.n_kv_head, HD)
                k_rms = np.sqrt(np.mean(k_2d * k_2d, axis=1, keepdims=True) + self.eps)
                k_2d[:] = k_2d / k_rms * lw.k_norm_w.reshape(1, HD)

            # RoPE
            rope_d = self.rope_dim if hasattr(self, 'rope_dim') and self.rope_dim > 0 else HD
            b_q[:] = self._apply_rope_fast(b_q, self.pos, NH, rope_dim=rope_d)
            b_k[:] = self._apply_rope_fast(b_k, self.pos, self.n_kv_head, rope_dim=rope_d)

            # KV cache
            self.kv_k[i, self.kv_len[i], :NKH] = b_k[:NKH]
            self.kv_v[i, self.kv_len[i], :NKH] = b_v[:NKH]

            # GQA Attention
            seq_len = self.kv_len[i] + 1
            if self._gqa_attn is not None and seq_len <= 4096:
                self._gqa_attn.gqa_attention_decode(
                    b_q.ctypes.data_as(cf),
                    self.kv_k[i].ctypes.data_as(cf),
                    self.kv_v[i].ctypes.data_as(cf),
                    b_att.ctypes.data_as(cf),
                    ci(seq_len), ci(NH), ci(self.n_kv_head), ci(HD),
                    self._p_gqa_ws)
            else:
                k_cache = self.kv_k[i, :seq_len].reshape(seq_len, self.n_kv_head, HD)
                v_cache = self.kv_v[i, :seq_len].reshape(seq_len, self.n_kv_head, HD)
                q_2d = b_q.reshape(NH, HD)
                nk = self.n_kv_head; gqa = self._gqa_rep
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

            # Output projection
            if hasattr(lw, 'attn_gate_use_c') and lw.attn_gate_use_c:
                pass  # Qwen3.6 specific
            else:
                if lw.attn_out_use_c:
                    kern.quant_matmul_omp(lw.attn_out_raw, p_att, p_oproj,
                                           lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt)
                    b_x[:N] = b_r[:N] + b_oproj[:N]
                else:
                    b_x[:N] = b_r[:N] + (lw.attn_out_f32 @ b_att)[:N]

        # FFN
        b_r[:] = b_x
        simd.rms_norm(p_xn, p_x,
                      lw.ffn_norm_w.ctypes.data_as(cf),
                      N, eps_f)

        if self.is_moe and self._moe_layers:
            self._forward_moe(i, p_xn, b_ffn, b_r, N)
        else:
            if lw.ffn_gate_use_c and lw.ffn_up_use_c:
                kern.batch_gate_up_omp(
                    lw.ffn_gate_raw, lw.ffn_up_raw, p_xn, p_gate, p_up,
                    lw.ffn_gate_nr, lw.ffn_up_nr,
                    lw.ffn_gate_nc, lw.ffn_gate_qt, lw.ffn_up_qt)
            else:
                pass
            simd.silu(p_silu, p_gate, ci(FF))
            b_silu[:FF] *= b_up[:FF]
            if lw.ffn_down_use_c:
                kern.quant_matmul_omp(lw.ffn_down_raw, p_silu, p_ffn,
                                       lw.ffn_down_nr, lw.ffn_down_nc, lw.ffn_down_qt)
                b_x[:N] = b_r[:N] + b_ffn[:N]
            else:
                b_x[:N] = b_r[:N] + (lw.ffn_down_f32 @ b_silu[:lw.ffn_down_nc.value])[:N]

        # Save hidden state after this layer
        if call_id <= 2:
            self._inst_hidden_states[call_id].append(self._x[:10].copy())

    # Final norm
    simd.rms_norm(p_xn, p_x,
                  self._out_norm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  N, eps_f)

    # Output projection
    if self._out_use_c:
        kern.quant_matmul_omp(self._out_raw, p_xn, p_logits,
                               self._out_nr, self._out_nc, self._out_qt)
    else:
        self._logits[:] = self._out_f32 @ b_qn

    self.pos += 1
    return self._logits

# Patch
TurboEngineV7MoE.forward = instrumented_forward

import ctypes

e.reset()
print("\n=== Call 1: forward(42) ===", flush=True)
l1 = e.forward(42).copy()

print("\n=== Call 2: forward(42) ===", flush=True)
l2 = e.forward(42).copy()

print(f"\nMax diff l1 vs l2: {np.max(np.abs(l1 - l2)):.6f}", flush=True)

# Compare hidden states layer by layer
print("\n=== Layer-by-layer hidden state comparison ===", flush=True)
hs1 = e._inst_hidden_states[1]
hs2 = e._inst_hidden_states[2]

for i in range(len(hs1)):
    diff = np.max(np.abs(hs1[i] - hs2[i]))
    print(f"  Layer {i}: max diff = {diff:.8f} | hs1[:5] = {hs1[i][:5]} | hs2[:5] = {hs2[i][:5]}", flush=True)
    if diff > 0.001:
        print(f"    *** DIVERGENCE at layer {i} ***", flush=True)
        break
