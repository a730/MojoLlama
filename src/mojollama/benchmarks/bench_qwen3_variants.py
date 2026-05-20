#!/usr/bin/env python3
"""Compare Q4KXL vs MXFP4 on Qwen3.6-35B-A3B."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))

models = {
    "Q4_K_XL": "/tmp/models/qwen3.6-q4kxl.gguf",
    "MXFP4":   "/tmp/models/qwen3.6-mxfp4.gguf",
}

results = []
for name, path in models.items():
    print(f"\n{'='*60}")
    print(f"  Loading {name}...")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    
    from turbo_engine_v7_moe import TurboEngineV7MoE
    e = TurboEngineV7MoE(path, 32)
    load_time = time.perf_counter() - t0
    print(f"  Load: {load_time:.1f}s | {e.n_layers}L/{e.n_embd}D MoE {e.n_experts}x{e.n_experts_per_tok}", flush=True)
    
    # Verify correctness
    tokens = np.array([[1]], dtype=np.int32)
    logits = e.forward(tokens)
    nan_c = int(np.isnan(logits).sum())
    maxv, minv = float(logits.max()), float(logits.min())
    print(f"  Verify: nan={nan_c} max={maxv:.2f} min={minv:.2f}", flush=True)
    
    if nan_c > 0:
        print(f"  FAIL: NaN detected!")
        results.append((name, 0, 0, 0, 0, load_time))
        continue
    
    # Warmup + benchmark
    for _ in range(5):
        token = int(np.argmax(logits))
        tokens = np.array([[token]], dtype=np.int32)
        logits = e.forward(tokens)
    
    start = time.perf_counter()
    N = 30
    for _ in range(N):
        token = int(np.argmax(logits))
        tokens = np.array([[token]], dtype=np.int32)
        logits = e.forward(tokens)
    elapsed = time.perf_counter() - start
    ms = elapsed / N * 1000
    tps = N / elapsed
    
    # Model size
    size_gb = os.path.getsize(path) / 1024**3
    
    print(f"  Size: {size_gb:.1f} GB | {ms:.1f} ms/tok | {tps:.1f} tok/s", flush=True)
    results.append((name, size_gb, ms, tps, load_time, nan_c))

print(f"\n{'='*60}")
print(f"  COMPARISON: Qwen3.6-35B-A3B")
print(f"{'='*60}")
print(f"  {'Variant':>12} {'Size':>8} {'ms/tok':>10} {'tok/s':>10} {'Load':>8} {'NaN':>6}")
print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*6}")
for name, sz, ms, tps, lt, nc in results:
    print(f"  {name:>12} {sz:>7.1f}G {ms:>9.1f}ms {tps:>9.1f} {lt:>7.1f}s {nc:>5}")
if len(results) == 2:
    r = results[1][3] / results[0][3] if results[0][3] > 0 else 0
    print(f"\n  MXFP4 vs Q4_K_XL: {r:.2f}x tok/s")
    if r > 1:
        print(f"  ✅ MXFP4 is faster on MojoLlama")
    else:
        print(f"  ✅ Q4_K_XL is faster on MojoLlama")
print(f"{'='*60}")
