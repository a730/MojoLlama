#!/usr/bin/env python3
"""Debug script for forward_c_batch segfault - test original .so."""
import os, sys, time, numpy as np, ctypes
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
sys.path.insert(0, '/onedev-workspace/work/src/mojollama')
from turbo_engine_v7_moe import TurboEngineV7MoE

kernel_dir = '/onedev-workspace/work/src/mojollama/kernels'

print("Loading model...", flush=True)
t0 = time.time()
eng = TurboEngineV7MoE('/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf', n_threads=1)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

# Test with original .so
print(f"_c_batch_ready={eng._c_batch_ready}", flush=True)

print("\nCalling forward_c_batch([0]) with original .so...", flush=True)
t0 = time.time()
try:
    logits = eng.forward_c_batch([0])
    elapsed = time.time() - t0
    print(f"OK: forward_c_batch returned in {elapsed:.2f}s", flush=True)
    print(f"logits shape={logits.shape}, max={logits.max():.2f}, min={logits.min():.2f}", flush=True)
    nan_count = np.isnan(logits).sum()
    print(f"NaN count: {nan_count}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Now compare with regular forward
print("\nCalling forward([0])...", flush=True)
t0 = time.time()
try:
    logits2 = eng.forward(0)
    elapsed = time.time() - t0
    print(f"OK: forward returned in {elapsed:.2f}s", flush=True)
    print(f"logits shape={logits2.shape}, max={logits2.max():.2f}, min={logits2.min():.2f}", flush=True)
    nan_count = np.isnan(logits2).sum()
    print(f"NaN count: {nan_count}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
