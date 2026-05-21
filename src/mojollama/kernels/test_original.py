#!/usr/bin/env python3
"""Minimal reproducer for the segfault."""
import os, sys, time, numpy as np, ctypes
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
sys.path.insert(0, '/onedev-workspace/work/src/mojollama')
from turbo_engine_v7_moe import TurboEngineV7MoE

kernel_dir = '/onedev-workspace/work/src/mojollama/kernels'

print("Loading model...", flush=True)
eng = TurboEngineV7MoE('/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf', n_threads=1)
print(f"Loaded.", flush=True)

# Test regular forward first
print("\n1. Testing regular forward(0)...", flush=True)
t0 = time.time()
try:
    logits = eng.forward(0)
    print(f"   OK in {time.time()-t0:.2f}s, max={logits.max():.2f}, NaN={np.isnan(logits).sum()}", flush=True)
except Exception as e:
    print(f"   ERROR: {e}", flush=True)

# Test forward_c_batch directly (this should crash)
print("\n2. Testing forward_c_batch([0])...", flush=True)
t0 = time.time()
try:
    logits = eng.forward_c_batch([0])
    print(f"   OK in {time.time()-t0:.2f}s, max={logits.max():.2f}, NaN={np.isnan(logits).sum()}", flush=True)
except Exception as e:
    print(f"   ERROR: {e}", flush=True)
except:
    print("   SEGFAULT (process terminated)", flush=True)
