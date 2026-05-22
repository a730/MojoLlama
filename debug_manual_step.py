#!/usr/bin/env python3
"""Debug the actual Q, K, attention values."""
import sys, os, numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from turbo_engine_v7_moe import TurboEngineV7MoE

print("Loading model...", flush=True)
e = TurboEngineV7MoE('/tmp/models/gpt-oss-20b-Q4_K_M.gguf', n_threads=1)
print("Model loaded.", flush=True)

# Let me check the actual forward with careful instrumentation
# First, let's understand the model architecture
print(f"\nArchitecture: {e.arch_prefix}", flush=True)
print(f"n_layers: {e.n_layers}", flush=True)
print(f"n_embd: {e.n_embd}, n_head: {e.n_head}, n_kv_head: {e.n_kv_head}, head_dim: {e.head_dim}", flush=True)
print(f"n_ff: {e.n_ff}", flush=True)
print(f"rope_freq_base: {e.rope_freq_base}", flush=True)
print(f"rope_dim: {e.rope_dim if hasattr(e, 'rope_dim') else 'N/A'}", flush=True)
print(f"is_moe: {e.is_moe}", flush=True)
print(f"has _gqa_attn: {e._gqa_attn is not None}", flush=True)

# Check first layer
lw = e._layers[0]
print(f"\nLayer 0:", flush=True)
print(f"  attn_qk_use_c: {hasattr(lw, 'attn_qk_use_c') and lw.attn_qk_use_c}", flush=True)
print(f"  attn_qkv_use_c: {hasattr(lw, 'attn_qkv_use_c') and lw.attn_qkv_use_c}", flush=True)
print(f"  attn_q_use_c: {lw.attn_q_use_c if hasattr(lw, 'attn_q_use_c') else 'N/A'}", flush=True)
print(f"  attn_out_use_c: {lw.attn_out_use_c if hasattr(lw, 'attn_out_use_c') else 'N/A'}", flush=True)
print(f"  has_q_norm: {lw.has_q_norm if hasattr(lw, 'has_q_norm') else 'N/A'}", flush=True)
print(f"  has_k_norm: {lw.has_k_norm if hasattr(lw, 'has_k_norm') else 'N/A'}", flush=True)

# Check the first layer's kv cache
print(f"\nInitial kv_len[0] = {e.kv_len[0]}", flush=True)

e.reset()

# Manually trace through part of the first layer
b_x = e._x
b_x_norm = e._x_norm
b_q = e._q  # view of _qk
b_k = e._k  # view of _qk[NK:]
b_v = e._v
b_att = e._att_out
b_oproj = e._o_proj

N = e.n_embd
NH = e.n_head
NKH = e.n_kv_head * e.head_dim
HD = e.head_dim

print(f"\nN={N}, NKH={NKH}, HD={HD}", flush=True)

# Embedding
np.copyto(b_x, e.emb[42])
print(f"After embedding, b_x[:5] = {b_x[:5]}", flush=True)
print(f"After embedding, self._x[:5] = {e._x[:5]}", flush=True)
print(f"b_x is self._x: {b_x is e._x}", flush=True)

import ctypes
cf = ctypes.POINTER(ctypes.c_float)
ci = ctypes.c_int

####### CALL 1 #######
print("\n\n========== CALL 1 (token 42, pos=0) ==========", flush=True)

# Reset state
e.reset()
print(f"After reset: pos={e.pos}, kv_len={e.kv_len[0]}", flush=True)

# Embedding
np.copyto(b_x, e.emb[42])
print(f"Embedding _x[:5] = {b_x[:5]}", flush=True)

# Layer 0
for layer_idx in [0]:  # Just layer 0
    lw = e._layers[layer_idx]
    
    # Save residual
    b_r = e._residual
    b_r[:] = b_x
    
    # RMS norm
    e._simd.rms_norm(e._p_x_norm, e._p_x,
                     lw.attn_norm_w.ctypes.data_as(cf),
                     N, e._eps_f)
    print(f"After norm, x_norm[:5] = {e._x_norm[:5]}", flush=True)
    
    # Fused QK
    e._kern.quant_matmul_omp(lw.attn_qk_raw, e._p_x_norm, e._p_qk,
                              lw.attn_qk_nr, lw.attn_qk_nc, lw.attn_qk_qt)
    e._kern.quant_matmul_omp(lw.attn_v_raw, e._p_x_norm, e._p_v,
                              lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)
    print(f"After QK matmul, _q[:5] = {e._q[:5]}", flush=True)
    
    # RoPE
    rope_d = HD if not (hasattr(e, 'rope_dim') and e.rope_dim > 0) else e.rope_dim
    print(f"rope_d = {rope_d}, pos = {e.pos}", flush=True)
    
    # Save Q and K before RoPE
    q_before = e._q[:10].copy()
    k_before = e._k[:10].copy()
    
    e._q[:] = e._apply_rope_fast(e._q, e.pos, NH, rope_dim=rope_d)
    e._k[:] = e._apply_rope_fast(e._k, e.pos, e.n_kv_head, rope_dim=rope_d)
    
    q_after = e._q[:10].copy()
    k_after = e._k[:10].copy()
    
    print(f"Q before RoPE[:5] = {q_before[:5]}", flush=True)
    print(f"Q after  RoPE[:5] = {q_after[:5]}", flush=True)
    print(f"Q diff = {np.max(np.abs(q_before - q_after)):.6f}", flush=True)
    
    # KV cache
    e.kv_k[layer_idx, e.kv_len[layer_idx], :NKH] = e._k[:NKH]
    e.kv_v[layer_idx, e.kv_len[layer_idx], :NKH] = e._v[:NKH]
    print(f"KV cache pos {e.kv_len[layer_idx]}: kv_k[0,:3] = {e.kv_k[layer_idx, e.kv_len[layer_idx], :3]}", flush=True)
    
    # GQA Attention
    seq_len = e.kv_len[layer_idx] + 1
    print(f"GQA with seq_len={seq_len}", flush=True)
    
    # Save att_out before
    att_before = e._att_out[:10].copy()
    
    e._gqa_attn.gqa_attention_decode(
        e._q.ctypes.data_as(cf),
        e.kv_k[layer_idx].ctypes.data_as(cf),
        e.kv_v[layer_idx].ctypes.data_as(cf),
        e._att_out.ctypes.data_as(cf),
        ci(seq_len), ci(NH), ci(e.n_kv_head), ci(HD),
        e._p_gqa_ws)
    
    att_after = e._att_out[:10].copy()
    print(f"att_out[:5] = {att_after[:5]}", flush=True)
    
    e.kv_len[layer_idx] += 1
    
    # Output projection
    e._kern.quant_matmul_omp(lw.attn_out_raw, e._p_att_out, e._p_o_proj,
                              lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt)
    e._x[:N] = b_r[:N] + e._o_proj[:N]
    print(f"After attn out proj, _x[:5] = {e._x[:5]}", flush=True)

print(f"\npos after call 1: {e.pos}", flush=True)  # Not incremented yet since we didn't call forward()

# Check the KV cache state
print(f"\nKV cache after call 1:", flush=True)
print(f"  kv_len[0] = {e.kv_len[0]}", flush=True)
print(f"  kv_k[0,0,:3] = {e.kv_k[0,0,:3]}", flush=True)

# Manually increment pos for call 2
e.pos += 1

# Also save hidden state
hidden_call1 = e._x[:10].copy()

####### CALL 2 (simulated, with same token 42) #######
print("\n\n========== CALL 2 (token 42, pos=1) ==========", flush=True)

# Embedding - same token
np.copyto(b_x, e.emb[42])
print(f"Embedding _x[:5] = {b_x[:5]}", flush=True)

# Layer 0
for layer_idx in [0]:
    lw = e._layers[layer_idx]
    
    # Save residual
    b_r = e._residual
    b_r[:] = b_x
    
    # RMS norm
    e._simd.rms_norm(e._p_x_norm, e._p_x,
                     lw.attn_norm_w.ctypes.data_as(cf),
                     N, e._eps_f)
    print(f"After norm, x_norm[:5] = {e._x_norm[:5]}", flush=True)
    
    # Fused QK
    e._kern.quant_matmul_omp(lw.attn_qk_raw, e._p_x_norm, e._p_qk,
                              lw.attn_qk_nr, lw.attn_qk_nc, lw.attn_qk_qt)
    e._kern.quant_matmul_omp(lw.attn_v_raw, e._p_x_norm, e._p_v,
                              lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)
    
    # RoPE - NOW WITH pos=1
    rope_d = HD
    print(f"rope_d = {rope_d}, pos = {e.pos}", flush=True)
    
    # Save Q and K before RoPE
    q_before = e._q[:10].copy()
    k_before = e._k[:10].copy()
    
    e._q[:] = e._apply_rope_fast(e._q, e.pos, NH, rope_dim=rope_d)
    e._k[:] = e._apply_rope_fast(e._k, e.pos, e.n_kv_head, rope_dim=rope_d)
    
    q_after = e._q[:10].copy()
    k_after = e._k[:10].copy()
    
    print(f"Q before RoPE[:5] = {q_before[:5]}", flush=True)
    print(f"Q after  RoPE[:5] = {q_after[:5]}", flush=True)
    print(f"Q diff = {np.max(np.abs(q_before - q_after)):.6f}", flush=True)
    
    # KV cache - store at position 1
    print(f"kv_len before store: {e.kv_len[layer_idx]}", flush=True)
    e.kv_k[layer_idx, e.kv_len[layer_idx], :NKH] = e._k[:NKH]
    e.kv_v[layer_idx, e.kv_len[layer_idx], :NKH] = e._v[:NKH]
    print(f"KV cache pos {e.kv_len[layer_idx]}: kv_k[0,:3] = {e.kv_k[layer_idx, e.kv_len[layer_idx], :3]}", flush=True)
    print(f"KV cache pos 0: kv_k[0,:3] = {e.kv_k[layer_idx, 0, :3]}", flush=True)
    
    # GQA Attention
    seq_len = e.kv_len[layer_idx] + 1
    print(f"GQA with seq_len={seq_len}", flush=True)
    print(f"kv_len after store but before GQA: {e.kv_len[layer_idx]}", flush=True)
    
    # Save att_out before
    att_before = e._att_out[:10].copy()
    
    e._gqa_attn.gqa_attention_decode(
        e._q.ctypes.data_as(cf),
        e.kv_k[layer_idx].ctypes.data_as(cf),
        e.kv_v[layer_idx].ctypes.data_as(cf),
        e._att_out.ctypes.data_as(cf),
        ci(seq_len), ci(NH), ci(e.n_kv_head), ci(HD),
        e._p_gqa_ws)
    
    att_after = e._att_out[:10].copy()
    print(f"att_out[:5] = {att_after[:5]}", flush=True)
    
    e.kv_len[layer_idx] += 1
    
    # Output projection
    e._kern.quant_matmul_omp(lw.attn_out_raw, e._p_att_out, e._p_o_proj,
                              lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt)
    e._x[:N] = b_r[:N] + e._o_proj[:N]
    print(f"After attn out proj, _x[:5] = {e._x[:5]}", flush=True)

hidden_call2 = e._x[:10].copy()

print(f"\n\n=== COMPARISON ===", flush=True)
print(f"Hidden call1[:10] = {hidden_call1}", flush=True)
print(f"Hidden call2[:10] = {hidden_call2}", flush=True)
print(f"Max diff: {np.max(np.abs(hidden_call1 - hidden_call2)):.8f}", flush=True)
