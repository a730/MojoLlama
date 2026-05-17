# ===----------------------------------------------------------------------=== #
# MojoLlama — Ops benchmark (v3)
# ===----------------------------------------------------------------------=== #

from std.python import Python
from std.math import rsqrt, exp
from std.time import perf_counter


fn silu_scalar(x: Float32) -> Float32:
    return x / (1 + exp(-x))


fn rmsnorm_element(x: Float32, weight: Float32, eps: Float32) -> Float32:
    return x / rsqrt(x * x + eps) * weight


fn main() raises:
    var np = Python.evaluate("__import__('numpy')")
    
    print("MojoLlama — Ops Benchmark")
    print("=" * 50)
    
    var n = 2048
    var x_py = np.random.randn(n).astype(np.float32)
    var out_py = np.zeros(n, np.float32)
    
    # Benchmark SiLU - Mojo
    print("\n1. SiLU (2048 x 1000)")
    var t0 = perf_counter()
    for _ in range(1000):
        for i in range(n):
            var val = Float32(py=x_py.__getitem__(i))
            out_py.__setitem__(i, value=silu_scalar(val))
    var t1 = perf_counter()
    var mojo_ms = (t1 - t0) * 1000.0
    print("Mojo:  " + String(mojo_ms) + "ms")
    
    # Benchmark SiLU - NumPy
    t0 = perf_counter()
    var np_silu = Python.evaluate("lambda x: x / (1 + __import__('numpy').exp(-x))")
    for _ in range(1000):
        _ = np_silu(x_py)
    t1 = perf_counter()
    var np_ms = (t1 - t0) * 1000.0
    print("NumPy: " + String(np_ms) + "ms")
    if np_ms > 0.0 and mojo_ms > 0.0:
        print("Speedup: " + String(np_ms / mojo_ms) + "x")
    
    # Benchmark RMSNorm
    print("\n2. RMSNorm (1x2048 x 500)")
    var n2 = Python.evaluate("2048")
    var b1 = Python.evaluate("1")
    var x2_py = np.random.randn(b1, n2).astype(np.float32)
    var w_py = np.ones(n2, np.float32)
    var eps_f = Float32(1.0e-6)
    var out2_py = np.zeros(b1, n2).astype(np.float32)
    
    # Mojo
    t0 = perf_counter()
    for _ in range(500):
        var sum_sq = Float32(0.0)
        for j in range(2048):
            var val = Float32(py=x2_py.__getitem__(0).__getitem__(j))
            sum_sq += val * val
        var inv_std = rsqrt(sum_sq / 2048.0 + eps_f)
        for j in range(2048):
            var xv = Float32(py=x2_py.__getitem__(0).__getitem__(j))
            var wv = Float32(py=w_py.__getitem__(j))
            out2_py.__getitem__(0).__setitem__(j, value=xv * inv_std * wv)
    t1 = perf_counter()
    var rn_mojo_ms = (t1 - t0) * 1000.0
    
    # NumPy
    t0 = perf_counter()
    for _ in range(500):
        var v = np.mean((x2_py.astype(np.float64)) ** 2, Python.evaluate("-1"), Python.evaluate("True"))
        _ = x2_py / (v + 1.0e-6).__call__("sqrt") * w_py
    t1 = perf_counter()
    var rn_np_ms = (t1 - t0) * 1000.0
    
    print("Mojo:  " + String(rn_mojo_ms) + "ms")
    print("NumPy: " + String(rn_np_ms) + "ms")
    if rn_np_ms > 0.0 and rn_mojo_ms > 0.0:
        print("Speedup: " + String(rn_np_ms / rn_mojo_ms) + "x")
    
    print("\nDone.")
