#!/usr/bin/env python3
"""Test that properly swaps to debug .so WITH argtypes."""
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

# Record original .so path
original_so_path = os.path.join(kernel_dir, 'combined_engine.so')

# Copy argtypes from original .so to debug .so
import copy
debug_so = ctypes.CDLL(os.path.join(kernel_dir, 'combined_engine_debug.so'))

# Set up argtypes on debug .so (same as original)
cf = ctypes.POINTER(ctypes.c_float)
cu = ctypes.POINTER(ctypes.c_uint8)
ci = ctypes.c_int

debug_so.quant_matmul_omp.argtypes = [cu, cf, cf, ci, ci, ci]
debug_so.quant_matmul_omp.restype = None
debug_so.f32_matmul_omp.argtypes = [cf, cf, cf, ci, ci]
debug_so.f32_matmul_omp.restype = None
debug_so.set_num_threads.argtypes = [ci]
debug_so.set_num_threads.restype = None
debug_so.set_num_threads(1)
debug_so.ssm_layer_fused.argtypes = [
    cf, cf, cf,
    cf, cf, cf,
    cf, cf, cf,
    cf,
    cu, ci, ci, ci,
    cu, ci, ci, ci,
    ci, ci, ci, ci, ci, ci, ci]
debug_so.ssm_layer_fused.restype = None
debug_so.qwen36_batch_forward.argtypes = [
    ci, ci, ci, ci, ci, ci, ci,
    ci, ci, ci, cf,
    ci, ci, ci, ci, ci, ci, ci,
    cf, ci,
    ctypes.POINTER(ci),
    ctypes.POINTER(ci),
    cf,
    ctypes.POINTER(cf), ctypes.POINTER(cf),
    ctypes.POINTER(cu), ctypes.POINTER(ci), ctypes.POINTER(ci), ctypes.POINTER(ci),
    ctypes.POINTER(cu), ctypes.POINTER(ci), ctypes.POINTER(ci), ctypes.POINTER(ci),
    ctypes.POINTER(cu), ctypes.POINTER(ci), ctypes.POINTER(ci), ctypes.POINTER(ci),
    ctypes.POINTER(cu), ctypes.POINTER(ci), ctypes.POINTER(ci), ctypes.POINTER(ci),
    ctypes.POINTER(cu), ctypes.POINTER(ci),
    ctypes.POINTER(cu), ctypes.POINTER(ci),
    ctypes.POINTER(cu), ctypes.POINTER(ci),
    ctypes.POINTER(cf), ctypes.POINTER(cf), ctypes.POINTER(cf),
    ctypes.POINTER(cf),
    ctypes.POINTER(ci),
    ctypes.POINTER(cu), ctypes.POINTER(ci),
    ctypes.POINTER(ci), ctypes.POINTER(ci), ctypes.POINTER(ci),
    cf, cu, ci, ci, ci,
    ctypes.POINTER(cf), ctypes.POINTER(cf), ctypes.POINTER(cf),
    ctypes.POINTER(cf), ctypes.POINTER(cf), ctypes.POINTER(cf),
    cf,
    cf, cf, ctypes.POINTER(ci),
    cf, cf,
    cf, cu,
]
debug_so.qwen36_batch_forward.restype = None
debug_so.moe_forward_omp.argtypes = [
    ctypes.POINTER(cu),
    ctypes.POINTER(cu),
    ctypes.POINTER(cu),
    cf,
    ci, ci,
    ci, ci, ci,
    ctypes.POINTER(ctypes.c_int),
    cf,
    ci,
    cf,
    cf,
    cu,
]
debug_so.moe_forward_omp.restype = None
debug_so.q6_k_dequantize_row.argtypes = [cu, cf, ci]
debug_so.q6_k_dequantize_row.restype = None
debug_so.q4_k_dequantize_row.argtypes = [cu, cf, ci]
debug_so.q4_k_dequantize_row.restype = None

# Swap to debug .so
eng._kern = debug_so
eng._init_c_batch_caches()

print(f"_c_batch_ready={eng._c_batch_ready}", flush=True)

# Test forward_c_batch with debug .so
print("\nCalling forward_c_batch([0]) with debug .so + argtypes...", flush=True)
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

# Compare with regular forward
print("\nCalling forward(0)...", flush=True)
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
