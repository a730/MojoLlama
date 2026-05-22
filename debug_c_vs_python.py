#!/usr/bin/env python3
"""Compare C GQA attention vs Python attention with identical inputs."""
import sys, os, numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from turbo_engine_v7_moe import TurboEngineV7MoE

print("Loading model...", flush=True)
e = TurboEngineV7MoE('/tmp/models/gpt-oss-20b-Q4_K_M.gguf', n_threads=1)
print("Model loaded.", flush=True)

import ctypes
cf = ctypes.POINTER(ctypes.c_float)
ci = ctypes.c_int

N = e.n_embd; NH = e.n_head; NKH = e.n_kv_head * e.head_dim; HD = e.head_dim
NK = NH * HD
print(f"N={N}, NH={NH}, NKH={NKH}, HD={HD}")

e.reset()

# Run layer 0 manually, but save Q/K/V BEFORE attention
def run_layer_and_capture_qkv(layer_idx):
    """Run layer up to GQA attention and return (q_rope, k_rope, v, kv_k, kv_v, kv_len, b_r)."""
    lw = e._layers[layer_idx]
    b_x = e._x; b_r = e._residual; b_x_norm = e._x_norm
    b_q = e._q; b_k = e._k; b_v = e._v
    
    np.copyto(b_x, e.emb[42])  # token 42
    
    b_r[:] = b_x
    e._simd.rms_norm(e._p_x_norm, e._p_x,
                     lw.attn_norm_w.ctypes.data_as(cf),
                     N, e._eps_f)
    
    # Fused QK
    e._kern.quant_matmul_omp(lw.attn_qk_raw, e._p_x_norm, e._p_qk,
                              lw.attn_qk_nr, lw.attn_qk_nc, lw.attn_qk_qt)
    e._kern.quant_matmul_omp(lw.attn_v_raw, e._p_x_norm, e._p_v,
                              lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)
    
    # RoPE
    e._q[:] = e._apply_rope_fast(e._q, e.pos, NH, rope_dim=HD)
    e._k[:] = e._apply_rope_fast(e._k, e.pos, e.n_kv_head, rope_dim=HD)
    
    # Return copies
    return {
        'q': e._q.copy(),
        'k': e._k.copy(),
        'v': e._v.copy(),
        'b_r': b_r.copy(),
    }

# Step 1: Run and capture Q, K, V
e.reset()
e.pos = 0
print("\n=== Step 1 (pos=0) ===")
qkv1 = run_layer_and_capture_qkv(0)

# Store K, V in cache at position 0
e.kv_k[0, 0, :NKH] = qkv1['k'][:NKH]
e.kv_v[0, 0, :NKH] = qkv1['v'][:NKH]
e.kv_len[0] = 1

# Step 2: Run with different pos but DO NOT store in cache
e.pos = 1
print("\n=== Step 2 (pos=1) - fresh Q, K, V but NOT stored ===")
qkv2 = run_layer_and_capture_qkv(0)

# Now we have:
# qkv1: Q at pos=0, K at pos=0, V at pos=0
# qkv2: Q at pos=1, K at pos=1, V at pos=1 (but not in cache)
# kv_cache has position 0 only (from qkv1)

# Test 1: C attention with seq_len=1, call 1's Q
print("\n\n=== TEST 1: C attention seq_len=1 with qkv1's Q ===")
att_c_1 = np.zeros(NK, dtype=np.float32)
e._gqa_attn.gqa_attention_decode(
    qkv1['q'].ctypes.data_as(cf),
    e.kv_k[0].ctypes.data_as(cf),
    e.kv_v[0].ctypes.data_as(cf),
    att_c_1.ctypes.data_as(cf),
    ci(1), ci(NH), ci(e.n_kv_head), ci(HD),
    e._p_gqa_ws)
print(f"C att[:5] = {att_c_1[:5]}")

# Test 2: Python attention with seq_len=1
print("\n=== TEST 2: Python attention seq_len=1 with qkv1's Q ===")
k_cache = e.kv_k[0, :1].reshape(1, e.n_kv_head, HD)
v_cache = e.kv_v[0, :1].reshape(1, e.n_kv_head, HD)
q_2d = qkv1['q'].reshape(NH, HD)
nk = e.n_kv_head; gqa = e._gqa_rep
q_g = q_2d.reshape(nk, gqa, HD)
k_T = k_cache.transpose(1, 0, 2)
scores = np.einsum('khd,ksd->khs', q_g, k_T) / np.sqrt(float(HD))
scores = scores.reshape(NH, 1)
scores -= np.max(scores, axis=1, keepdims=True)
np.exp(scores, out=scores)
scores /= np.sum(scores, axis=1, keepdims=True)
v_T = v_cache.transpose(1, 0, 2)
att_py_1 = np.einsum('khs,ksd->khd', scores.reshape(nk, gqa, 1), v_T).reshape(-1)
print(f"Py att[:5] = {att_py_1[:5]}")
print(f"C vs Py diff: {np.max(np.abs(att_c_1 - att_py_1)):.8f}")

# Test 3: C attention with seq_len=2, qkv2's Q (pos=1 Q), but ONLY position 0 in cache
print("\n\n=== TEST 3: C attention seq_len=2 with qkv2's Q (pos=1), only pos 0 in cache ===")
att_c_2 = np.zeros(NK, dtype=np.float32)
e._gqa_attn.gqa_attention_decode(
    qkv2['q'].ctypes.data_as(cf),
    e.kv_k[0].ctypes.data_as(cf),
    e.kv_v[0].ctypes.data_as(cf),
    att_c_2.ctypes.data_as(cf),
    ci(2), ci(NH), ci(e.n_kv_head), ci(HD),
    e._p_gqa_ws)
print(f"C att[:5] = {att_c_2[:5]}")
print(f"Diff from test 1: {np.max(np.abs(att_c_2 - att_c_1)):.8f}")

# Now add position 1 to cache (simulating normal operation)
e.kv_k[0, 1, :NKH] = qkv2['k'][:NKH]
e.kv_v[0, 1, :NKH] = qkv2['v'][:NKH]
e.kv_len[0] = 2

# Test 4: C attention with seq_len=2, BOTH positions in cache
print("\n\n=== TEST 4: C attention seq_len=2 with qkv2's Q, BOTH positions in cache ===")
att_c_3 = np.zeros(NK, dtype=np.float32)
e._gqa_attn.gqa_attention_decode(
    qkv2['q'].ctypes.data_as(cf),
    e.kv_k[0].ctypes.data_as(cf),
    e.kv_v[0].ctypes.data_as(cf),
    att_c_3.ctypes.data_as(cf),
    ci(2), ci(NH), ci(e.n_kv_head), ci(HD),
    e._p_gqa_ws)
print(f"C att[:5] = {att_c_3[:5]}")
print(f"Diff from test 1: {np.max(np.abs(att_c_3 - att_c_1)):.8f}")
print(f"Diff from test 3: {np.max(np.abs(att_c_3 - att_c_2)):.8f}")

# Test 5: Python attention with seq_len=2, BOTH positions
print("\n\n=== TEST 5: Python attention seq_len=2 with qkv2's Q, both positions ===")
k_cache2 = e.kv_k[0, :2].reshape(2, e.n_kv_head, HD)
v_cache2 = e.kv_v[0, :2].reshape(2, e.n_kv_head, HD)
q_2d2 = qkv2['q'].reshape(NH, HD)
q_g2 = q_2d2.reshape(nk, gqa, HD)
k_T2 = k_cache2.transpose(1, 0, 2)
scores2 = np.einsum('khd,ksd->khs', q_g2, k_T2) / np.sqrt(float(HD))
scores2 = scores2.reshape(NH, 2)
scores2 -= np.max(scores2, axis=1, keepdims=True)
np.exp(scores2, out=scores2)
scores2 /= np.sum(scores2, axis=1, keepdims=True)
v_T2 = v_cache2.transpose(1, 0, 2)
att_py_2 = np.einsum('khs,ksd->khd', scores2.reshape(nk, gqa, 2), v_T2).reshape(-1)
print(f"Py att[:5] = {att_py_2[:5]}")
print(f"C vs Py diff (test 4 vs test 5): {np.max(np.abs(att_c_3 - att_py_2)):.8f}")

# Test 6: Use same Q (qkv1's Q) but seq_len=2
print("\n\n=== TEST 6: C attention seq_len=2 with qkv1's Q (pos=0), both positions in cache ===")
att_c_4 = np.zeros(NK, dtype=np.float32)
e._gqa_attn.gqa_attention_decode(
    qkv1['q'].ctypes.data_as(cf),
    e.kv_k[0].ctypes.data_as(cf),
    e.kv_v[0].ctypes.data_as(cf),
    att_c_4.ctypes.data_as(cf),
    ci(2), ci(NH), ci(e.n_kv_head), ci(HD),
    e._p_gqa_ws)
print(f"C att[:5] = {att_c_4[:5]}")
print(f"Diff from test 1 (same Q, seq_len=1): {np.max(np.abs(att_c_4 - att_c_1)):.8f}")
