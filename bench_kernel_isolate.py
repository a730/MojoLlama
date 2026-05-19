#!/usr/bin/env python3
"""Minimal C kernel benchmark — isolate the Python overhead."""
import ctypes, time, numpy as np, os, sys

sys.path.insert(0, '/onedev-workspace/work/src')
os.chdir('/onedev-workspace/work')

# Load C kernel
SO = os.path.join(os.path.dirname('/onedev-workspace/work/src/mojollama/kernels/'), "q4_kernel_omp.so")
lib = ctypes.CDLL(SO)
lib.q4_matmul_omp.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.q4_matmul_omp.restype = None

# Also load AVX2 single-thread kernel
SO2 = os.path.join(os.path.dirname('/onedev-workspace/work/src/mojollama/kernels/'), "q4_kernel_avx2.so")
lib2 = ctypes.CDLL(SO2)
lib2.q4_matmul_avx2.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib2.q4_matmul_avx2.restype = None

# Also test the-weight-store approach from omp_engine
from mojollama.kernels.omp_engine import Model, mm, Q4_TS
mdl = Model('Llama-3.2-1B-Instruct-Q4_0.gguf')

# Get Q weight
w_q, tt_q = mdl.get("blk.0.attn_q.weight")
print(f"Q weight: type={tt_q}, shape={w_q.shape}, dtype={w_q.dtype}, size={len(w_q)} bytes")
n_embd = mdl.n["n_embd"]
bpr = n_embd // 32
block_sz = bpr * 18
nr_q = len(w_q) // block_sz
print(f"Q weight: nr={nr_q}, nc={n_embd}, blocks_per_row={bpr}")

# Prepare input
x = np.random.randn(n_embd).astype(np.float32)

# Test 1: Direct C kernel call (OMP)
print("\n=== Direct C Kernel (OMP, 32 threads) ===")
out = np.zeros(nr_q, dtype=np.float32)
for _ in range(3):
    lib.q4_matmul_omp(
        w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nr_q, n_embd, 18)

t0 = time.perf_counter()
for _ in range(50):
    lib.q4_matmul_omp(
        w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nr_q, n_embd, 18)
t1 = time.perf_counter()
print(f"Q4_0 matmul 2048x2048 (OMP t=32): {(t1-t0)/50*1000:.3f} ms per call")

# Test 2: Direct C kernel call (AVX2, single thread)  
print("\n=== Direct C Kernel (AVX2, single thread, full matrix) ===")
out2 = np.zeros(nr_q, dtype=np.float32)
for _ in range(3):
    lib2.q4_matmul_avx2(
        w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nr_q, n_embd, 0, nr_q)

t0 = time.perf_counter()
for _ in range(50):
    lib2.q4_matmul_avx2(
        w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nr_q, n_embd, 0, nr_q)
t1 = time.perf_counter()
print(f"Q4_0 matmul 2048x2048 (AVX2 single): {(t1-t0)/50*1000:.3f} ms per call")

# Test 3: mm() wrapper overhead
print("\n=== mm() Wrapper Overhead ===")
for _ in range(3):
    mm(mdl.get("blk.0.attn_q.weight"), x, n_embd)

t0 = time.perf_counter()
for _ in range(50):
    q = mm(mdl.get("blk.0.attn_q.weight"), x, n_embd)
t1 = time.perf_counter()
print(f"mm() wrapper Q proj: {(t1-t0)/50*1000:.3f} ms per call")

# Test 4: Weight lookup overhead
print("\n=== Weight Lookup Overhead ===")
t0 = time.perf_counter()
for _ in range(50):
    w, tt = mdl.get("blk.0.attn_q.weight")
t1 = time.perf_counter()
print(f"Weight lookup: {(t1-t0)/50*1000:.3f} ms")

# Test 5: Output array allocation
t0 = time.perf_counter()
for _ in range(50):
    o = np.zeros(nr_q, dtype=np.float32)
t1 = time.perf_counter()
print(f"np.zeros({nr_q}): {(t1-t0)/50*1000:.3f} ms")

# Test 6: ctypes conversion overhead
w_ptr = w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
t0 = time.perf_counter()
for _ in range(50):
    w_p = w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    x_p = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
t1 = time.perf_counter()
print(f"ctypes conversion: {(t1-t0)/50*1000:.3f} ms")