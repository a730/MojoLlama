#!/usr/bin/env python3
"""Detailed forward pass profiling for FastEngine."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mojollama.kernels.fast_engine import FastEngine

engine = FastEngine('Llama-3.2-1B-Instruct-Q4_0.gguf', n_threads=32)
engine.reset()

# Warmup
engine.forward([128000])
engine.forward([9906])

print("\n=== Detailed Layer Profile ===\n")

# Profile each component of layer 0
h = engine._emb_cache[128000].astype(np.float32).copy()
n_embd = engine.n_embd

# Time each op 100 times
def bench(label, fn, iters=100):
    # Warmup
    for _ in range(5): fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1-t0)*1e6)
    times.sort()
    med = times[iters//2]
    print(f"  {label:40s}: {med:8.1f} µs")
    return med

# RMS norm
t_rms = bench("RMS norm (C kernel)", lambda: engine.rms_norm(h, 'blk.0.attn_norm.weight'))

# Q projection
t_q = bench("Q proj (C OMP)", lambda: engine.matmul('blk.0.attn_q.weight', h, n_embd))

# K projection
t_k = bench("K proj (C OMP)", lambda: engine.matmul('blk.0.attn_k.weight', h, n_embd))

# V projection  
t_v = bench("V proj (C OMP)", lambda: engine.matmul('blk.0.attn_v.weight', h, n_embd))

# RoPE (both Q and K)
q = engine.matmul('blk.0.attn_q.weight', h, n_embd)
t_rope = bench("RoPE Q+K", lambda: (engine.apply_rope(q, 0, engine.n_head),
                                       engine.apply_rope(q, 0, engine.n_kv_head)))

# Attention (single position, Python loop)
k_new = engine.matmul('blk.0.attn_k.weight', h, n_embd)
v_new = engine.matmul('blk.0.attn_v.weight', h, n_embd)

# Reset KV for attention test
engine.kv_k[0] = np.zeros((0, engine.n_kv), dtype=np.float32)
engine.kv_v[0] = np.zeros((0, engine.n_kv), dtype=np.float32)
t_attn = bench("Attention (1 pos, Python+einsum)", lambda: engine.attention(q, k_new, v_new, 0, 0))

# O projection
att = np.random.randn(engine.n_embd).astype(np.float32)
t_o = bench("O proj (C OMP)", lambda: engine.matmul('blk.0.attn_output.weight', att, n_embd))

# FFN norm
t_ffn_norm = bench("RMS norm FFN (C kernel)", lambda: engine.rms_norm(h, 'blk.0.ffn_norm.weight'))

# FFN gate
t_gate = bench("FFN gate (C OMP)", lambda: engine.matmul('blk.0.ffn_gate.weight', h, n_embd))

# SiLU
gate_raw = np.random.randn(engine.n_ff).astype(np.float32)
t_silu = bench("SiLU (C kernel)", lambda: engine.silu(gate_raw.copy()))

# FFN up
t_up = bench("FFN up (C OMP)", lambda: engine.matmul('blk.0.ffn_up.weight', h, n_embd))

# FFN down
gate_up = np.random.randn(engine.n_ff).astype(np.float32)
t_down = bench("FFN down (C OMP)", lambda: engine.matmul('blk.0.ffn_down.weight', gate_up, engine.n_ff))

# Residual add
t_resid = bench("Residual add (numpy)", lambda: h + h)

# KV cache vstack
kv_k = np.random.randn(1, engine.n_kv).astype(np.float32)
kv_v = np.random.randn(1, engine.n_kv).astype(np.float32)
t_kv_update = bench("KV cache update (vstack)", 
    lambda: np.vstack([np.zeros((1, engine.n_kv), dtype=np.float32), kv_k]))

# Total estimate
total_per_layer = (t_rms*2 + t_q + t_k + t_v + t_rope + t_attn + t_o + 
                   t_ffn_norm + t_gate + t_silu + t_up + t_down + t_resid*2 + t_kv_update*2)
total_16layers = total_per_layer * engine.n_layers
tok_per_sec = 1e6 / total_16layers

print(f"\n=== Summary ===")
print(f"Per layer: {total_per_layer/1000:.2f} ms")
print(f"Total (16 layers): {total_16layers/1000:.2f} ms")
print(f"Estimated decode: {tok_per_sec:.1f} tok/s")
print(f"Target: 85 tok/s (llama.cpp)")
print(f"Gap: {(85 - tok_per_sec)/85*100:.1f}%")

# Breakdown by category
matmul_total = t_q + t_k + t_v + t_o + t_gate + t_up + t_down
other_total = total_per_layer - matmul_total
print(f"\nMatmul total: {matmul_total/1000:.2f} ms ({matmul_total/total_per_layer*100:.1f}%)")
print(f"Other (Python): {other_total/1000:.2f} ms ({other_total/total_per_layer*100:.1f}%)")

# Now measure actual forward pass
engine.reset()
for tok in [128000, 9906]:
    engine.forward([tok])

times = []
next_tok = 3000
for i in range(20):
    t0 = time.perf_counter()
    logits = engine.forward([next_tok])
    t1 = time.perf_counter()
    times.append((t1-t0)*1000)
    next_tok = int(np.argmax(logits))

times.sort()
print(f"\nActual decode (median): {times[10]:.2f} ms/tok = {1000/times[10]:.1f} tok/s")