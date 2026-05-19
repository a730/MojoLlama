#!/usr/bin/env python3
"""Vectorized Q4_0 quantizer — numpy-fast, runs once at load time."""

import numpy as np

def quantize_q4_0_vectorized(data_f32: np.ndarray, n_cols: int) -> np.ndarray:
    """Quantize F32 row-major matrix to Q4_0 format matching our C kernel.
    
    Q4_0 block layout (18 bytes per 32 values):
      bytes[0:2]: f16 scale (little-endian)
      bytes[2:18]: 16 bytes, each packing two 4-bit values
        byte[j] = (val[2j] + 8) & 0xF | (((val[2j+1] + 8) & 0xF) << 4)
    
    Returns: uint8 array of shape (n_blocks * 18,)
    """
    flat = data_f32.astype(np.float32).ravel()
    n = len(flat)
    assert n % 32 == 0, f"n={n} not divisible by 32"
    n_blocks = n // 32
    
    # Reshape to (n_blocks, 32)
    blocks = flat.reshape(n_blocks, 32)
    
    # Scale per block: max(|values|) / 8
    amax = np.max(np.abs(blocks), axis=1)  # (n_blocks,)
    scales = np.where(amax > 0, amax / 8.0, 0.0).astype(np.float16)
    
    # Quantize: q = round(val / scale) + 8, clipped to [0, 15]
    inv_scales = np.where(scales.astype(np.float32) > 0, 
                          1.0 / scales.astype(np.float32), 0.0).astype(np.float32)
    # (n_blocks, 32) — quantized values in 0..15
    q = np.clip(np.round(blocks * inv_scales[:, None] + 8.0), 0, 15).astype(np.uint8)
    
    # Pack pairs: byte[j] = (q[2j] & 0xF) | ((q[2j+1] & 0xF) << 4)
    q_pairs = q.reshape(n_blocks, 16, 2)
    packed = (q_pairs[:, :, 0] & 0x0F) | ((q_pairs[:, :, 1] & 0x0F) << 4)  # (n_blocks, 16)
    
    # Assemble output: scale (2 bytes LE) + packed nibbles (16 bytes)
    result = np.zeros((n_blocks, 18), dtype=np.uint8)
    # Write f16 scale as little-endian
    scale_view = scales.view(np.uint16)  # (n_blocks,)
    result[:, 0] = (scale_view & 0xFF).astype(np.uint8)
    result[:, 1] = ((scale_view >> 8) & 0xFF).astype(np.uint8)
    result[:, 2:18] = packed
    
    return result.ravel()


if __name__ == '__main__':
    import time
    
    # Correctness test
    _so = __import__('ctypes').CDLL('/onedev-workspace/work/src/mojollama/kernels/q4_kernel_omp.so')
    _so.q4_matmul_omp.argtypes = [
        __import__('ctypes').POINTER(__import__('ctypes').c_uint8),
        __import__('ctypes').POINTER(__import__('ctypes').c_float),
        __import__('ctypes').POINTER(__import__('ctypes').c_float),
        __import__('ctypes').c_int, __import__('ctypes').c_int, __import__('ctypes').c_int,
    ]
    _so.q4_set_num_threads(32)
    
    # Small correctness test
    np.random.seed(42)
    W = np.random.randn(4, 32).astype(np.float32) * 0.5
    x = np.random.randn(32).astype(np.float32)
    expected = W @ x
    
    W_q4 = quantize_q4_0_vectorized(W, 32)
    w_ptr = W_q4.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_uint8))
    x_ptr = x.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float))
    out = np.zeros(4, dtype=np.float32)
    out_ptr = out.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float))
    _so.q4_matmul_omp(w_ptr, x_ptr, out_ptr, 4, 32, 18)
    
    print(f"Expected: {expected}")
    print(f"Got:      {out}")
    print(f"Max error: {np.max(np.abs(expected - out)):.4f}")
    print(f"Mean error: {np.mean(np.abs(expected - out)):.4f}")
    
    # Large benchmark
    W_big = np.random.randn(128256, 2048).astype(np.float32)
    t0 = time.perf_counter()
    W_q4 = quantize_q4_0_vectorized(W_big, 2048)
    t1 = time.perf_counter()
    print(f"\nQuantize 128K×2048: {(t1-t0):.3f}s (one-time cost)")
    print(f"Q4_0 size: {len(W_q4)/1e6:.1f} MB vs F32: {W_big.nbytes/1e6:.1f} MB")
    
    # Benchmark Q4_0 matmul for output projection
    x_big = np.random.randn(2048).astype(np.float32)
    logits = np.zeros(128256, dtype=np.float32)
    _so.q4_matmul_omp(
        W_q4.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_uint8)),
        x_big.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float)),
        logits.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float)),
        128256, 2048, 18)
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _so.q4_matmul_omp(
            W_q4.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_uint8)),
            x_big.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float)),
            logits.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float)),
            128256, 2048, 18)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    times.sort()
    print(f"Q4_0 output proj (128K×2048): {times[10]:.2f} ms median")
    
    # Compare with F32
    logits_f32 = np.zeros(128256, dtype=np.float32)
    _so.f32_matmul_omp.argtypes = [
        __import__('ctypes').POINTER(__import__('ctypes').c_float),
        __import__('ctypes').POINTER(__import__('ctypes').c_float),
        __import__('ctypes').POINTER(__import__('ctypes').c_float),
        __import__('ctypes').c_int, __import__('ctypes').c_int,
    ]
    _so.f32_matmul_omp.restype = None
    times_f32 = []
    for _ in range(10):
        t0 = time.perf_counter()
        _so.f32_matmul_omp(
            W_big.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float)),
            x_big.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float)),
            logits_f32.ctypes.data_as(__import__('ctypes').POINTER(__import__('ctypes').c_float)),
            128256, 2048)
        t1 = time.perf_counter()
        times_f32.append((t1-t0)*1000)
    times_f32.sort()
    print(f"F32 output proj (128K×2048):  {times_f32[5]:.2f} ms median")
    print(f"Speedup: {times_f32[5]/times[10]:.1f}x")