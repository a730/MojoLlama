#!/usr/bin/env python3
"""Test GPT-OSS-20B end-to-end."""
import sys, time, numpy as np
sys.path.insert(0, 'mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE

print('=== GPT-OSS-20B ===')
e = TurboEngineV7MoE('/tmp/models/gpt-oss-20b-Q4_K_M.gguf', n_threads=32)
print(f'Loaded: {e.n_layers}L/{e.n_embd}D/{e.n_head}H MoE {e.n_experts}x{e.n_experts_per_tok}')
tokens = np.array([[1]], dtype=np.int32)
logits = e.forward(tokens)
print(f'Logits: shape={logits.shape}, dtype={logits.dtype}')
nan_c = int(np.isnan(logits).sum())
inf_c = int(np.isinf(logits).sum())
maxv, minv, meanv = float(logits.max()), float(logits.min()), float(logits.mean())
top5 = list(np.argsort(logits)[-5:][::-1])
print(f'nan={nan_c} inf={inf_c} max={maxv:.2f} min={minv:.2f} mean={meanv:.6f}')
print(f'Top-5: {top5}')
assert nan_c == 0, 'NaN detected!'
assert inf_c == 0, 'Inf detected!'
print('GPT-OSS: PASS (no NaN/Inf)')

# Quick benchmark
for _ in range(5):
    token = int(np.argmax(logits))
    tokens = np.array([[token]], dtype=np.int32)
    logits = e.forward(tokens)
start = time.perf_counter()
N = 20
for _ in range(N):
    token = int(np.argmax(logits))
    tokens = np.array([[token]], dtype=np.int32)
    logits = e.forward(tokens)
elapsed = time.perf_counter() - start
print(f'Bench: {elapsed/N*1000:.1f} ms/tok ({N/elapsed:.1f} tok/s)')
print('DONE')
