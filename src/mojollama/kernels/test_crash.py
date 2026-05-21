#!/usr/bin/env python3
"""Debug script for forward_c_batch segfault."""
import os, sys, time, numpy as np, ctypes
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
sys.path.insert(0, '/onedev-workspace/work/src/mojollama')
from turbo_engine_v7_moe import TurboEngineV7MoE

# Load the debug version of the .so
import importlib, types
# Monkey-patch to use debug .so
kernel_dir = '/onedev-workspace/work/src/mojollama/kernels'

# Load engine
print("Loading model...", flush=True)
t0 = time.time()
eng = TurboEngineV7MoE('/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf', n_threads=1)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

# Swap the .so to our debug version
debug_so = ctypes.CDLL(os.path.join(kernel_dir, 'combined_engine_debug.so'))
eng._kern = debug_so
# Re-init caches with new .so
eng._init_c_batch_caches()

print(f"_c_batch_ready={eng._c_batch_ready}", flush=True)
print(f"n_layers={eng.n_layers} n_embd={eng.n_embd} n_head={eng.n_head}", flush=True)
print(f"head_dim={eng.head_dim} n_ff={eng.n_ff}", flush=True)
print(f"vocab_size={eng.vocab_size}", flush=True)
print(f"_c_ws_S={eng._c_ws_S}", flush=True)

# Test forward_c_batch
print("\nCalling forward_c_batch([0])...", flush=True)
t0 = time.time()
try:
    logits = eng.forward_c_batch([0])
    elapsed = time.time() - t0
    print(f"forward_c_batch returned in {elapsed:.2f}s", flush=True)
    print(f"logits shape={logits.shape}, max={logits.max():.2f}, min={logits.min():.2f}", flush=True)
    nan_count = np.isnan(logits).sum()
    print(f"NaN count: {nan_count}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
