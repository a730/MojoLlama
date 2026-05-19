#!/usr/bin/env python3
"""Profile the OMP engine layer-by-layer to find bottlenecks."""
import time, sys, os, ctypes, numpy as np

# Add project path
sys.path.insert(0, '/onedev-workspace/work/src')
os.chdir('/onedev-workspace/work')

from mojollama.kernels.omp_engine import Model, mm, rms, Q4_TS
import gguf

# Load model
mdl = Model('Llama-3.2-1B-Instruct-Q4_0.gguf')
n = mdl.n
hd = n["head_dim"]
nh = n["n_head"]
nkh = n["n_kv_head"]
print(f"Model: {n}")

# Setup KV cache
kvc = [{"k": np.zeros((0, n["n_kv"]), dtype=np.float32),
         "v": np.zeros((0, n["n_kv"]), dtype=np.float32)}
        for _ in range(n["n_layers"])]

# Get first token embedding
h = mdl.get("token_embd.weight")[0][128000].astype(np.float32).copy()
print(f"Embedding shape: {h.shape}, dtype: {h.dtype}")

# Warmup C library
dummy_out = np.zeros(1, dtype=np.float32)
w_test = mdl.get("blk.0.attn_q.weight")[0]
x_test = h.copy()
mm(mdl.get("blk.0.attn_q.weight"), h, n["n_embd"])

print("\n=== Per-Operation Profiling (Layer 0) ===\n")

layer = 0
iterations = 50

# 1. RMS Norm
times_rms = []
for _ in range(iterations):
    t0 = time.perf_counter()
    h_normed = rms(h, mdl.get(f"blk.{layer}.attn_norm.weight")[0])
    t1 = time.perf_counter()
    times_rms.append((t1-t0)*1000)
print(f"RMS norm: {np.median(times_rms):.3f} ms (median of {iterations})")

# 2. Q projection (2048 → 2048)
times_q = []
for _ in range(iterations):
    t0 = time.perf_counter()
    q = mm(mdl.get(f"blk.{layer}.attn_q.weight"), h_normed.copy(), n["n_embd"])
    t1 = time.perf_counter()
    times_q.append((t1-t0)*1000)
print(f"Q proj (2048→2048): {np.median(times_q):.3f} ms")

# 3. K projection (2048 → 512)
times_k = []
for _ in range(iterations):
    t0 = time.perf_counter()
    k = mm(mdl.get(f"blk.{layer}.attn_k.weight"), h_normed.copy(), n["n_embd"])
    t1 = time.perf_counter()
    times_k.append((t1-t0)*1000)
print(f"K proj (2048→512): {np.median(times_k):.3f} ms")

# 4. V projection (2048 → 512)
times_v = []
for _ in range(iterations):
    t0 = time.perf_counter()
    v = mm(mdl.get(f"blk.{layer}.attn_v.weight"), h_normed.copy(), n["n_embd"])
    t1 = time.perf_counter()
    times_v.append((t1-t0)*1000)
print(f"V proj (2048→512): {np.median(times_v):.3f} ms")

# 5. RoPE
q2 = q.reshape(nh, hd)
k2 = k.reshape(nkh, hd)
times_rope = []
for _ in range(iterations):
    q_copy = q2.copy()
    k_copy = k2.copy()
    t0 = time.perf_counter()
    freqs = 1.0 / (500000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    cos_f = np.cos(0 * freqs)
    sin_f = np.sin(0 * freqs)
    for xx, nx in [(q_copy, nh), (k_copy, nkh)]:
        xr = xx.reshape(nx, hd//2, 2)
        xr_rot = np.stack([-xr[...,1], xr[...,0]], axis=-1)
        xx[:] = (xr*cos_f.reshape(1,hd//2,1) + xr_rot*sin_f.reshape(1,hd//2,1)).reshape(nx, hd)
    t1 = time.perf_counter()
    times_rope.append((t1-t0)*1000)
print(f"RoPE: {np.median(times_rope):.3f} ms")

# 6. Attention
times_attn = []
for _ in range(iterations):
    t0 = time.perf_counter()
    ng = nh // nkh
    att = np.zeros(nh*hd, dtype=np.float32)
    for hh in range(nh):
        kvh = hh // ng
        ks = kvc[layer]["k"][0:1, kvh*hd:(kvh+1)*hd]
        vs = kvc[layer]["v"][0:1, kvh*hd:(kvh+1)*hd]
        sc = q2[hh] @ ks.T
        sc = sc - np.max(sc)
        att[hh*hd:(hh+1)*hd] = (np.exp(sc)/np.sum(np.exp(sc))) @ vs
    t1 = time.perf_counter()
    times_attn.append((t1-t0)*1000)
print(f"Attention (1 pos, 32 heads): {np.median(times_attn):.3f} ms")

# 7. O projection + residual
times_o = []
for _ in range(iterations):
    t0 = time.perf_counter()
    h = rms(h, mdl.get(f"blk.{layer}.attn_norm.weight")[0])
    r = h.copy()
    att_out = mm(mdl.get(f"blk.{layer}.attn_output.weight"), att, n["n_embd"])
    h = r + att_out
    t1 = time.perf_counter()
    times_o.append((t1-t0)*1000)
print(f"O proj + residual (2048→2048): {np.median(times_o):.3f} ms")

# 8. FFN
times_ffn = []
for _ in range(iterations):
    h_ffn = h.copy()
    t0 = time.perf_counter()
    h_normed = rms(h_ffn, mdl.get(f"blk.{layer}.ffn_norm.weight")[0])
    gate = mm(mdl.get(f"blk.{layer}.ffn_gate.weight"), h_normed, n["n_embd"])
    gate = gate / (1 + np.exp(-gate))
    up = mm(mdl.get(f"blk.{layer}.ffn_up.weight"), h_normed, n["n_embd"])
    down = mm(mdl.get(f"blk.{layer}.ffn_down.weight"), gate*up, n["n_ff"])
    h_ffn = h_ffn + down
    t1 = time.perf_counter()
    times_ffn.append((t1-t0)*1000)
print(f"FFN (gate+up+down): {np.median(times_ffn):.3f} ms")

# 9. Just the overhead (no computation)
times_overhead = []
for _ in range(iterations):
    t0 = time.perf_counter()
    _ = mdl.get(f"blk.{layer}.attn_q.weight")  # weight lookup
    _ = h.copy()  # numpy copy
    t1 = time.perf_counter()
    times_overhead.append((t1-t0)*1000)
print(f"Python overhead (lookup+copy): {np.median(times_overhead):.3f} ms")

# Summary
total_matmul = np.median(times_q) + np.median(times_k) + np.median(times_v) + np.median(times_o) + np.median(times_ffn)
other = 57.0 - total_matmul  # from observed layer time
print(f"\n=== Summary ===")
print(f"Q+K+V+O+FFN matmul time: {total_matmul:.3f} ms")
print(f"RMS norm: {np.median(times_rms):.3f} ms")
print(f"RoPE: {np.median(times_rope):.3f} ms")
print(f"Attention: {np.median(times_attn):.3f} ms")
print(f"Python overhead: {np.median(times_overhead):.3f} ms")
print(f"Estimated layer total: {total_matmul + np.median(times_rms)*2 + np.median(times_rope) + np.median(times_attn):.3f} ms")
print(f"Target: <3ms per layer (for >80 tok/s × 16 layers = ~12ms total)")