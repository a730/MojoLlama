# ===----------------------------------------------------------------------=== #
# MojoLlama — Mojo inference ops (Phase 1)
# Pure Mojo implementations of core neural network operations.
# ===----------------------------------------------------------------------=== #

from std.python import Python, PythonObject
from std.math import rsqrt, exp


fn silu_scalar(x: Float32) -> Float32:
    """SiLU activation: x / (1 + exp(-x))"""
    return x / (1 + exp(-x))


fn rmsnorm_scalar(x: Float32, weight: Float32, eps: Float32) -> Float32:
    """RMSNorm element: x / sqrt(variance + eps) * weight"""
    return x / rsqrt(x * x + eps) * weight


fn silu(x: PythonObject) raises -> PythonObject:
    """Apply SiLU activation to a numpy array. Uses Mojo scalar ops per element."""
    var np = Python.evaluate("__import__('numpy')")
    var arr = np.asarray(x, np.float32)
    var out = np.zeros_like(arr)
    for i in range(Int(py=arr.size())):
        out.flat.__setitem__(i, value=silu_scalar(Float32(py=arr.flat.__getitem__(i))))
    return out


fn rmsnorm(x: PythonObject, weight: PythonObject, eps: PythonObject) raises -> PythonObject:
    """Apply RMSNorm to numpy arrays. Uses Mojo scalar ops per element."""
    var np = Python.evaluate("__import__('numpy')")
    var arr = np.asarray(x, np.float32)
    var w = np.asarray(weight, np.float32)
    var eps_f = Float32(py=eps)
    var out = np.zeros_like(arr)
    var dim = Int(py=arr.shape[-1].__index__())
    var n_rows = Int(py=arr.size()) // dim
    
    for row in range(n_rows):
        var offset = row * dim
        # Compute sum of squares
        var sum_sq: Float32 = 0.0
        for j in range(dim):
            var val = Float32(py=arr.flat.__getitem__(offset + j))
            sum_sq += val * val
        var inv_std = rsqrt(sum_sq / Float32(dim) + eps_f)
        # Normalize and scale
        for j in range(dim):
            var xv = Float32(py=arr.flat.__getitem__(offset + j))
            var wv = Float32(py=w.flat.__getitem__(j))
            out.flat.__setitem__(offset + j, value=xv * inv_std * wv)
    return out
