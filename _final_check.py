"""Final verification: check correctness and benchmark."""
import sys, os, time
import numpy as np
sys.path.insert(0, 'src')

from mojollama.kernels.turbo_engine_v7_moe import TurboEngineV7MoE

print("=== Loading model ===")
engine = TurboEngineV7MoE('/onedev-workspace/work/Qwen3-30B-A3B-Q4_K_M.gguf', n_threads=32)

# Verify basic properties
print(f"\n=== Configuration ===")
print(f"  head_dim: {engine.head_dim} (expect 128)")
print(f"  NKH = {engine.n_kv_head} * {engine.head_dim} = {engine.n_kv_head * engine.head_dim} (expect 512)")
print(f"  NK = {engine.n_head} * {engine.head_dim} = {engine.n_head * engine.head_dim} (expect 4096)")
print(f"  Q norm: {engine._layers[0].has_q_norm}")
print(f"  K norm: {engine._layers[0].has_k_norm}")

# Test with deterministic output
engine.reset()
out0 = engine.forward(0)
top5 = np.argsort(out0)[-5:][::-1]
print(f"\n  Token 0 top-5 IDs: {top5.tolist()}")
print(f"  Token 0 top-5 logits: {out0[top5].tolist()}")
print(f"  No NaN: {not np.any(np.isnan(out0))}")

# Multi-token generation
engine.reset()
tokens = []
tok = 0
for i in range(20):
    logits = engine.forward(tok)
    tok = int(np.argmax(logits))
    tokens.append(tok)
print(f"  Generated 20 tokens (IDs): {tokens}")
print(f"  No NaN: {all(not np.any(np.isnan(engine.forward(t))) for t in tokens[:3])}")

# Full benchmark
print(f"\n=== Performance benchmark ===")
times = []
engine.reset()
for _ in range(10):
    t0 = time.perf_counter()
    engine.forward(0)
    times.append(time.perf_counter() - t0)
    
t_prompt = np.median(sorted(times))
print(f"  Prompt (first token): {t_prompt*1000:.1f} ms (small KV)")
  
# Generation benchmark (single-token, warm + timed)
engine.reset()
for i in range(10):
    engine.forward(i)

times_gen = []
tok = 1
for _ in range(10):
    t0 = time.perf_counter()
    logits = engine.forward(tok)
    times_gen.append(time.perf_counter() - t0)
    tok = int(np.argmax(logits))

times_gen.sort()
med_gen = np.median(times_gen)
print(f"  Generation (median of 10): {med_gen*1000:.1f} ms/tok = {1/med_gen:.1f} tok/s")
print(f"  Min: {min(times_gen)*1000:.1f} Max: {max(times_gen)*1000:.1f}")
print(f"\n  vs llama.cpp baseline: 26.9 tok/s (Q4_K_M, t=32)")
print(f"  Speed ratio: {(1/med_gen) / 26.9 * 100:.1f}%")
