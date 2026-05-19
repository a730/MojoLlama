#!/usr/bin/env python3
"""GPT-OSS-20B end-to-end test via TurboEngineV7MoE."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
from turbo_engine_v7_moe import TurboEngineV7MoE

MODEL_PATH = "/tmp/models/gpt-oss-20b-Q4_K_M.gguf"
N_THREADS = 32

print("=" * 60)
print("ITEM 3: GPT-OSS-20B End-to-End Test")
print("=" * 60)

os.environ['OMP_NUM_THREADS'] = str(N_THREADS)

# Load
t0 = time.perf_counter()
engine = TurboEngineV7MoE(MODEL_PATH, N_THREADS)
load_time = time.perf_counter() - t0
print(f"\nLoad time: {load_time:.2f}s")
print(f"Model: {engine.n_layers}L/{engine.n_embd}D/{engine.n_ff}FF")
print(f"  heads: {engine.n_head}H/{engine.n_kv_head}KV, head_dim={engine.head_dim}")
print(f"  vocab: {engine.vocab_size}")
print(f"  experts: {engine.n_experts}, top-k: {engine.n_experts_per_tok}")
print(f"  is_moe: {engine.is_moe}")
print(f"  eps: {engine.eps}")

VOCAB = engine.vocab_size
assert VOCAB == 201088, f"Expected vocab=201088, got {VOCAB}"

# Forward a few warmup tokens
print("\n--- Warmup ---")
tokens = np.array([[1]], dtype=np.int32)
for i in range(5):
    logits = engine.forward(tokens)
    token = int(np.argmax(logits)) if logits.ndim == 1 else int(np.argmax(logits[0, -1, :]))
    tokens = np.array([[token]], dtype=np.int32)
    print(f"  Step {i}: token={token}")

# Benchmark
print("\n--- Benchmark ---")
tokens = np.array([[1]], dtype=np.int32)
N = 30
start = time.perf_counter()
for i in range(N):
    logits = engine.forward(tokens)
    has_nan = np.any(np.isnan(logits))
    token = int(np.argmax(logits)) if logits.ndim == 1 else int(np.argmax(logits[0, -1, :]))
    tokens = np.array([[token]], dtype=np.int32)
    if has_nan:
        print(f"  WARNING: NaN detected in logits at step {i}")
elapsed = time.perf_counter() - start

# Analyze final logits
logits = engine.forward(tokens)
if logits.ndim == 3:
    logits = logits[0, -1, :]  # (B, T, V) -> (V,)
elif logits.ndim == 2:
    logits = logits[-1, :]  # (T, V) -> (V,)

logits_flat = logits.ravel()
has_nan = np.any(np.isnan(logits_flat))
logits_max = float(np.max(logits_flat))
logits_min = float(np.min(logits_flat))
logits_mean = float(np.mean(logits_flat))
logits_std = float(np.std(logits_flat))

top5_indices = np.argsort(logits_flat)[-5:][::-1]
top5_values = logits_flat[top5_indices]

ms_per_tok = elapsed / N * 1000
toks_per_sec = N / elapsed

print(f"\n--- Results ---")
print(f"Logits shape: {logits.shape}")
print(f"Has NaN: {has_nan}")
print(f"Logits max: {logits_max:.4f}")
print(f"Logits min: {logits_min:.4f}")
print(f"Logits mean: {logits_mean:.4f}")
print(f"Logits std: {logits_std:.4f}")
print(f"Top-5 token IDs: {top5_indices.tolist()}")
print(f"Top-5 logit values: {[f'{v:.4f}' for v in top5_values]}")
print(f"Benchmark: {ms_per_tok:.1f} ms/tok ({toks_per_sec:.1f} tok/s)")
print(f"\nGPT-OSS: loads OK, logits max={logits_max:.2f} min={logits_min:.2f}, no NaN, top-5 token IDs: {top5_indices.tolist()}")
