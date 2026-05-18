#!/usr/bin/env python3
"""Benchmark C AVX2+FMA+OMP kernel performance on Threadripper 3970X."""
import ctypes
import numpy as np
import time
import os

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_omp_lib():
    lib = ctypes.CDLL(os.path.join(KERNEL_DIR, "q4_kernel_omp.so"))
    lib.q4_matmul_omp.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.q4_matmul_omp.restype = None
    lib.q4_matmul_omp_blocked.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.q4_matmul_omp_blocked.restype = None
    lib.q4_rms_norm.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.q4_rms_norm.restype = None
    lib.q4_silu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.q4_silu.restype = None
    lib.q4_get_max_threads.argtypes = []
    lib.q4_get_max_threads.restype = ctypes.c_int
    lib.q4_set_num_threads.argtypes = [ctypes.c_int]
    lib.q4_set_num_threads.restype = None
    return lib


def make_q4_weights(n_rows, n_cols):
    """Create random Q4_0 weight data (18 bytes per 32-element block)."""
    bpr = n_cols // 32
    total_bytes = n_rows * bpr * 18
    return (ctypes.c_uint8 * total_bytes).from_buffer_copy(
        np.random.randint(0, 256, total_bytes, dtype=np.uint8).tobytes()
    )


def bench(label, fn, n_iters=50):
    # Warmup
    for _ in range(3):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"  {label}: {avg:.2f} ± {std:.2f} ms")
    return avg


def main():
    lib = load_omp_lib()
    
    # Detect physical cores
    phys_cores = 0
    try:
        import glob
        cores = set()
        for p in glob.glob("/sys/devices/system/cpu/cpu*/topology/core_id"):
            with open(p) as f:
                cores.add(int(f.read().strip()))
        phys_cores = len(cores) if cores else os.cpu_count() // 2
    except:
        phys_cores = os.cpu_count() // 2
    
    print(f"═══ C AVX2+FMA+OMP Kernel Benchmark ═══")
    print(f"Physical cores: {phys_cores}")
    print(f"OMP max threads: {lib.q4_get_max_threads()}")
    print()
    
    # Test configs
    configs = [
        (2048, 2048, "1B QKV"),
        (8192, 2048, "1B FFN"),
        (4096, 2048, "1B O_proj"),
    ]
    
    for n_rows, n_cols, label in configs:
        w = make_q4_weights(n_rows, n_cols)
        x = np.random.randn(n_cols).astype(np.float32)
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        print(f"── {label}: {n_rows}x{n_cols} ──")
        
        # Test with different thread counts
        for n_threads in [1, phys_cores]:
            lib.q4_set_num_threads(n_threads)
            out = np.zeros(n_rows, dtype=np.float32)
            out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Row-outer (standard)
            t1 = bench(f"OMP t={n_threads:2d} row_outer",
                       lambda: lib.q4_matmul_omp(w, x_ptr, out_ptr, n_rows, n_cols, 18))
            
            # 4-row blocked
            t2 = bench(f"OMP t={n_threads:2d} blocked ",
                       lambda: lib.q4_matmul_omp_blocked(w, x_ptr, out_ptr, n_rows, n_cols, 18))
            
            if t1 > 0 and t2 > 0:
                print(f"    blocked/row_outer ratio: {t2/t1:.2f}x")
        print()
    
    # Thread sweep
    print("── Thread sweep: 2048x2048 ──")
    w = make_q4_weights(2048, 2048)
    x = np.random.randn(2048).astype(np.float32)
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for t in [1, 2, 4, 8, 16, 24, 32, 48, 64]:
        if t > os.cpu_count():
            continue
        lib.q4_set_num_threads(t)
        out = np.zeros(2048, dtype=np.float32)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = bench(f"  t={t:2d}", lambda t=t: lib.q4_matmul_omp(w, x_ptr, out_ptr, 2048, 2048, 18))
    
    print()
    
    # RMS norm benchmark
    print("── Vectorized ops ──")
    n = 2048
    x_data = np.random.randn(n).astype(np.float32)
    w_data = np.ones(n, dtype=np.float32) / np.sqrt(n)
    x_ptr = x_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    w_ptr = w_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_data = np.zeros(n, dtype=np.float32)
    out_ptr = out_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    def bench_rms():
        lib.q4_rms_norm(x_ptr, w_ptr, out_ptr, n)
    
    bench("RMS norm 2048", bench_rms, 500)
    
    def bench_silu():
        x_copy = x_data.copy()
        lib.q4_silu(x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
    
    bench("SiLU 2048", bench_silu, 500)

    # Throughput estimate for Llama 3.2 1B Q4_0
    print()
    print("═══ Estimated throughput (1B Q4_0, 32 threads) ═══")
    lib.q4_set_num_threads(phys_cores)
    
    # Measure matmul components
    w_qkv = make_q4_weights(3 * 2048, 2048)  # QKV combined
    w_ffn = make_q4_weights(2 * 8192, 2048)  # gate+up
    w_down = make_q4_weights(2048, 8192)     # down
    
    x_in = np.random.randn(2048).astype(np.float32)
    x_ptr = x_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # QKV
    out_qkv = np.zeros(3 * 2048, dtype=np.float32)
    t_qkv = bench("QKV (6144x2048)", lambda: lib.q4_matmul_omp(w_qkv, x_ptr, out_qkv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 3 * 2048, 2048, 18), 20)
    
    # FFN gate+up
    out_ffn = np.zeros(2 * 8192, dtype=np.float32)
    t_ffn = bench("FFN gate+up (16384x2048)", lambda: lib.q4_matmul_omp(w_ffn, x_ptr, out_ffn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 2 * 8192, 2048, 18), 10)
    
    # FFN down
    x_down = np.random.randn(8192).astype(np.float32)
    out_down = np.zeros(2048, dtype=np.float32)
    t_down = bench("FFN down (2048x8192)", lambda: lib.q4_matmul_omp(w_down, x_down.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_down.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 2048, 8192, 18), 20)
    
    total_layer = (t_qkv + t_ffn + t_down + 1.0) * 16 / 1000  # 16 layers + overhead
    print(f"\n  Total layer: {total_layer*1000:.1f} ms")
    print(f"  Estimated throughput: {16/(total_layer*1000):.1f} tok/s (decoder)")
    print(f"  llama.cpp benchmark: ~84 tok/s (for reference)")


if __name__ == "__main__":
    main()