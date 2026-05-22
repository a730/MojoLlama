#!/usr/bin/env python3
"""Critical test: compare attention with different-token inputs."""
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
nk = e.n_kv_head
gqa = e._gqa_rep

def compute_qkv(engine, token_id, pos):
    """Compute Q, K, V for one layer."""
    e._x[:] = e.emb[token_id]
    lw = e._layers[0]
    
    b_r = e._residual
    b_r[:] = e._x
    
    e._simd.rms_norm(e._p_x_norm, e._p_x,
                     lw.attn_norm_w.ctypes.data_as(cf),
                     N, e._eps_f)
    
    # Fused QK
    e._kern.quant_matmul_omp(lw.attn_qk_raw, e._p_x_norm, e._p_qk,
                              lw.attn_qk_nr, lw.attn_qk_nc, lw.attn_qk_qt)
    e._kern.quant_matmul_omp(lw.attn_v_raw, e._p_x_norm, e._p_v,
                              lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)
    
    # RoPE
    orig_pos = e.pos
    e.pos = pos
    e._q[:] = e._apply_rope_fast(e._q, pos, NH, rope_dim=HD)
    e._k[:] = e._apply_rope_fast(e._k, pos, nk, rope_dim=HD)
    e.pos = orig_pos
    
    return {
        'q': e._q.copy(),
        'k': e._k.copy(),
        'v': e._v.copy(),
    }

# Compute QKV for token 42 at pos 0
qkv42_0 = compute_qkv(e, 42, 0)
print(f"Token 42 V[:5] = {qkv42_0['v'][:5]}")

# Compute QKV for token 99 at pos 1
qkv99_1 = compute_qkv(e, 99, 1)
print(f"Token 99 V[:5] = {qkv99_1['v'][:5]}")

# Compute QKV for token 99 at pos 0 (for baseline comparison)
qkv99_0 = compute_qkv(e, 99, 0)
print(f"Token 99 (pos0) V[:5] = {qkv99_0['v'][:5]}")

# Are V values different for different tokens?
print(f"\nV42 vs V99 max diff: {np.max(np.abs(qkv42_0['v'] - qkv99_1['v'])):.6f}")
print(f"V99 pos0 vs pos1 max diff: {np.max(np.abs(qkv99_0['v'] - qkv99_1['v'])):.6f}")

# Now test C attention with DIFFERENT tokens at different positions
cache_k = np.zeros((4096, NKH), dtype=np.float32)
cache_v = np.zeros((4096, NKH), dtype=np.float32)

# Fill position 0 with token 42 K,V
cache_k[0, :NKH] = qkv42_0['k'][:NKH]
cache_v[0, :NKH] = qkv42_0['v'][:NKH]

# Fill position 1 with token 99 K,V  
cache_k[1, :NKH] = qkv99_1['k'][:NKH]
cache_v[1, :NKH] = qkv99_1['v'][:NKH]

# Python attention with seq_len=2, Q from token 99 at pos 1
def python_attention(q, k_cache, v_cache, seq_len):
    k_3d = k_cache[:seq_len].reshape(seq_len, nk, HD)
    v_3d = v_cache[:seq_len].reshape(seq_len, nk, HD)
    q_2d = q.reshape(NH, HD)
    q_g = q_2d.reshape(nk, gqa, HD)
    k_T = k_3d.transpose(1, 0, 2)
    scores = np.einsum('khd,ksd->khs', q_g, k_T) / np.sqrt(float(HD))
    scores = scores.reshape(NH, seq_len)
    scores -= np.max(scores, axis=1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= np.sum(scores, axis=1, keepdims=True)
    v_T = v_3d.transpose(1, 0, 2)
    att = np.einsum('khs,ksd->khd', scores.reshape(nk, gqa, seq_len), v_T)
    return att.reshape(-1)

# C attention
def c_attention(q, k_cache, v_cache, seq_len):
    att = np.zeros(NK, dtype=np.float32)
    e._gqa_attn.gqa_attention_decode(
        q.ctypes.data_as(cf),
        k_cache.ctypes.data_as(cf),
        v_cache.ctypes.data_as(cf),
        att.ctypes.data_as(cf),
        ci(seq_len), ci(NH), ci(nk), ci(HD),
        e._p_gqa_ws)
    return att

# Test: Q from token 99 at pos 1, cache has token 42 at pos 0, token 99 at pos 1
print("\n\n=== Test: seq_len=2, different tokens at different positions ===")
py_att = python_attention(qkv99_1['q'], cache_k, cache_v, 2)
c_att = c_attention(qkv99_1['q'], cache_k, cache_v, 2)
print(f"Py att[:5] = {py_att[:5]}")
print(f"C  att[:5] = {c_att[:5]}")
print(f"Py vs C diff: {np.max(np.abs(py_att - c_att)):.8f}")

# Compare with attending to only pos 0 (token 42) or only pos 1 (token 99)
py_att_pos0 = python_attention(qkv99_1['q'], cache_k[:1], cache_v[:1], 1)
py_att_pos1 = python_attention(qkv99_1['q'], cache_k[1:2], cache_v[1:2], 1)
print(f"\nPy att only pos0 (token 42)[:5] = {py_att_pos0[:5]}")
print(f"Py att only pos1 (token 99)[:5] = {py_att_pos1[:5]}")
print(f"Are they different? {np.max(np.abs(py_att_pos0 - py_att_pos1)):.6f}")
print(f"Seq=2 vs only pos1 diff: {np.max(np.abs(py_att - py_att_pos1)):.8f}")
print(f"Seq=2 vs only pos0 diff: {np.max(np.abs(py_att - py_att_pos0)):.8f}")

# NOW: Full model test with different tokens
print("\n\n=== Full model test: token 42 then token 99 ===")
e.reset()
l1 = e.forward(42).copy()
l2 = e.forward(99).copy()
print(f"Step 1 (token 42) logits[:5] = {l1[:5]}")
print(f"Step 2 (token 99) logits[:5] = {l2[:5]}")
print(f"Diff: {np.max(np.abs(l1 - l2)):.6f}")

# Reset and do token 99 alone
print("\n=== Baseline: token 99 alone (no context) ===")
e.reset()
l99_alone = e.forward(99).copy()
print(f"Token 99 alone logits[:5] = {l99_alone[:5]}")

# Compare: step 2 with token 99 vs token 99 alone
print(f"\nToken 99 step 2 vs token 99 alone diff: {np.max(np.abs(l2 - l99_alone)):.6f}")

# This is the key test: does step 2 output with different token depend on KV cache?
# If diff > 0, the attention is correctly using the cache
print("\n=== Is the KV cache being used? ===")
print(f"Token 99 step 2 (with 42 in cache) != token 99 alone: {np.max(np.abs(l2 - l99_alone)) > 0.001}")
