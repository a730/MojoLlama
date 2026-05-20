#!/usr/bin/env python3
"""Test TinyLlama via TurboEngineV77 (dense engine)."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
from turbo_engine_v77 import TurboEngineV77

MODEL_PATH = "/tmp/tl-Q4_0.gguf"
N_THREADS = 32

print("=" * 60)
print("ITEM 4: TinyLlama dense path via TurboEngineV77")
print("=" * 60)

os.environ['OMP_NUM_THREADS'] = str(N_THREADS)

# Load
t0 = time.perf_counter()
engine = TurboEngineV77(MODEL_PATH, N_THREADS)
load_time = time.perf_counter() - t0
print(f"\nLoad time: {load_time:.2f}s")
print(f"Model: {engine.n_layers}L/{engine.n_embd}D/{engine.n_ff}FF")
print(f"  heads: {engine.n_head}H/{engine.n_kv_head}KV, head_dim={engine.head_dim}")
print(f"  vocab: {engine.vocab_size}")
print(f"  is_moe: {engine.is_moe}")

# Try a forward pass with a few tokens
print("\n--- Forward test ---")
engine.reset()
for i in range(5):
    tid = 1 if i == 0 else int(np.argmax(engine._lg)) if i > 1 else 42
    logits = engine.forward(tid)
    has_nan = np.any(np.isnan(logits))
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"  Step {i}: token_id={tid}, top5={top5.tolist()}, NaN={has_nan}")

# Benchmark
print("\n--- Benchmark ---")
engine.reset()
N = 50
start = time.perf_counter()
for i in range(N):
    if i == 0:
        logits = engine.forward(1)
    else:
        token = int(np.argmax(engine._lg))
        logits = engine.forward(token)
elapsed = time.perf_counter() - start

ms_per_tok = elapsed / N * 1000
toks_per_sec = N / elapsed

print(f"\n--- Results ---")
print(f"Dense forward: {ms_per_tok:.1f} ms/tok ({toks_per_sec:.1f} tok/s)")
print(f"TurboEngineV77 successfully loaded and ran TinyLlama Q4_0")
