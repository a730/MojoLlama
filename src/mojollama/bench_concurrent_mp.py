#!/usr/bin/env python3
"""Qwen3.6 MXFP4 concurrent server via multiprocessing with shared mmap weights."""
import sys, os, time, multiprocessing as mp
import numpy as np

MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"
N_USERS = 10
GEN_TOKENS = 50
N_WARMUP = 10

# ── Load model once in parent ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
print(f"Loading Qwen3.6 MXFP4 (shared via mmap)...", flush=True)

# Use fork context for shared memory
mp.set_start_method('fork', force=True)

# Pre-load the GGUF data into a shared dict
import gguf
reader = gguf.GGUFReader(MODEL)

# ── Worker function ──
def worker_fn(worker_id, result_queue):
    """Each worker loads a fresh engine from the shared GGUF reader."""
    os.environ['OMP_NUM_THREADS'] = '32'
    sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
    
    from turbo_engine_v7_moe import TurboEngineV7MoE
    
    # Each worker loads independently — weights are mmap'd, physical pages shared
    e = TurboEngineV7MoE(MODEL, 32)
    
    # Warmup
    for _ in range(N_WARMUP):
        e.forward(np.array([[1]], dtype=np.int32))
    e.reset()
    
    # Benchmark: generate GEN_TOKENS tokens
    tok = np.array([[1]], dtype=np.int32)
    times = []
    for _ in range(GEN_TOKENS):
        t0 = time.perf_counter()
        logits = e.forward(tok)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        tok = np.array([[int(np.argmax(logits))]], dtype=np.int32)
    
    avg_ms = np.mean(times[5:]) * 1000
    tps = 1000 / avg_ms
    result_queue.put((worker_id, avg_ms, tps))

# ── Spawn workers ──
print(f"Spawning {N_USERS} workers (fork + shared mmap)...", flush=True)
result_queue = mp.Queue()
workers = []

t_start = time.perf_counter()
for i in range(N_USERS):
    p = mp.Process(target=worker_fn, args=(i, result_queue))
    workers.append(p)
    p.start()

# Collect results
results = {}
for _ in range(N_USERS):
    wid, avg_ms, tps = result_queue.get()
    results[wid] = (avg_ms, tps)

# Wait for all
for p in workers:
    p.join()

total_time = time.perf_counter() - t_start

# ── Report ──
print(f"\n{'='*60}")
print(f"  Qwen3.6 MXFP4 — Concurrent ({N_USERS} users, mp.fork)")
print(f"{'='*60}")
avg_tps_all = sum(t for _, t in results.values()) / len(results)
min_tps = min(t for _, t in results.values())
max_tps = max(t for _, t in results.values())
best_ms = min(m for m, _ in results.values())
worst_ms = max(m for m, _ in results.values())

for wid, (avg_ms, tps) in sorted(results.items()):
    print(f"  Worker {wid:2d}: {avg_ms:5.1f}ms → {tps:5.1f} tok/s")

print(f"{'─'*60}")
print(f"  Average:  {avg_tps_all:.1f} tok/s per user")
print(f"  Best:     {max_tps:.1f} tok/s ({best_ms:.1f}ms)")
print(f"  Worst:    {min_tps:.1f} tok/s ({worst_ms:.1f}ms)")
print(f"  Aggregate: {avg_tps_all * N_USERS:.0f} tok/s ({N_USERS} users)")
print(f"  Wall time: {total_time:.1f}s")
print(f"  Speedup vs single: {avg_tps_all * N_USERS / avg_tps_all:.0f}x")
