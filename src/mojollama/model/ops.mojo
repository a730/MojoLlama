# ===----------------------------------------------------------------------=== #
# MojoLlama — Mojo inference ops (Phase 1)
# Pure Mojo implementations of core neural network operations.
# With Intel Arc GPU support via Python/SYCL interop bridge.
# ===----------------------------------------------------------------------=== #

from std.python import Python, PythonObject
from std.math import rsqrt, exp


# ─── Scalar ops (always available, CPU-only) ────────────────────────────────

fn silu_scalar(x: Float32) -> Float32:
    """SiLU activation: x / (1 + exp(-x))"""
    return x / (1 + exp(-x))


fn rmsnorm_scalar(x: Float32, weight: Float32, eps: Float32) -> Float32:
    """RMSNorm element: x / sqrt(variance + eps) * weight"""
    return x / rsqrt(x * x + eps) * weight


# ─── Python interop ops (CPU via numpy) ─────────────────────────────────────

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


# ─── Intel Arc GPU dispatch (Phase 2) ──────────────────────────────────────
#
# These functions dispatch to Intel Arc GPU via Python SYCL bridge.
# When an Intel Arc GPU is detected (via dpctl), operations are
# offloaded to the SYCL device. Falls back to CPU numpy otherwise.
#
# Usage from Mojo:
#   from ops import intel_arc_silu, intel_arc_set_device
#   intel_arc_set_device()           # auto-detect Intel Arc
#   result = intel_arc_silu(x_array) # runs on Arc GPU
#
# Prerequisites (at Python level):
#   pip install dpctl dpnp
#   Intel GPU driver (i915 + Level Zero)

alias _INTEL_ARC_MODULE = "mojollama.model.backends.intel_arc"


fn _ensure_intel_arc_module() raises -> PythonObject:
    """Import the Intel Arc backend module (cached)."""
    var sys = Python.evaluate("__import__('sys')")
    var module_name = _INTEL_ARC_MODULE
    if sys.modules.__contains__(module_name):
        return sys.modules[module_name]
    return Python.evaluate("__import__")(module_name)


fn intel_arc_is_available() raises -> bool:
    """Check if an Intel Arc GPU is available for compute."""
    try:
        var arc_mod = _ensure_intel_arc_module()
        var backend = arc_mod.create_intel_arc_backend()
        return Python.evaluate("lambda b: b.is_available")(backend)
    except:
        return False


fn intel_arc_device_info() raises -> PythonObject:
    """Get Intel Arc GPU device info, or None if unavailable."""
    try:
        var arc_mod = _ensure_intel_arc_module()
        return arc_mod.query_intel_arc_devices()
    except:
        return Python.evaluate("[]")


fn intel_arc_silu(x: PythonObject) raises -> PythonObject:
    """Apply SiLU on Intel Arc GPU via SYCL/dpnp. Falls back to CPU."""
    try:
        var arc_mod = _ensure_intel_arc_module()
        var backend = arc_mod.create_intel_arc_backend()
        if Python.evaluate("lambda b: b.is_available")(backend):
            var dev_x = backend.to_device(x)
            var result = backend.silu(dev_x)
            return backend.to_cpu(result)
    except:
        pass
    # Fallback to CPU numpy
    return silu(x)


fn intel_arc_rmsnorm(x: PythonObject, weight: PythonObject, eps: PythonObject) raises -> PythonObject:
    """Apply RMSNorm on Intel Arc GPU via SYCL/dpnp. Falls back to CPU."""
    try:
        var arc_mod = _ensure_intel_arc_module()
        var backend = arc_mod.create_intel_arc_backend()
        if Python.evaluate("lambda b: b.is_available")(backend):
            var dev_x = backend.to_device(x)
            var dev_w = backend.to_device(weight)
            var result = backend.rms_norm(dev_x, dev_w, Float32(py=eps))
            return backend.to_cpu(result)
    except:
        pass
    # Fallback to CPU
    return rmsnorm(x, weight, eps)


fn intel_arc_matmul(a: PythonObject, b: PythonObject) raises -> PythonObject:
    """Matrix multiply on Intel Arc GPU. Falls back to CPU numpy."""
    try:
        var arc_mod = _ensure_intel_arc_module()
        var backend = arc_mod.create_intel_arc_backend()
        if Python.evaluate("lambda b: b.is_available")(backend):
            var dev_a = backend.to_device(a)
            var dev_b = backend.to_device(b)
            var result = backend.matmul(dev_a, dev_b)
            return backend.to_cpu(result)
    except:
        pass
    # Fallback to CPU matmul
    var np = Python.evaluate("__import__('numpy')")
    return np.asarray(a) @ np.asarray(b)


# ─── Device auto-select ────────────────────────────────────────────────────
#
# intel_arc_prefer: When True (default), tries Intel Arc GPU first.
# Set to False to force CPU path.
#
# intel_arc_active: True when Intel Arc GPU was detected and initialized.

var intel_arc_prefer: Bool = True
var intel_arc_active: Bool = False


fn intel_arc_set_device(prefer: Bool = True) raises:
    """Try to detect and initialize Intel Arc GPU.
    
    Args:
        prefer: When True, ops will use Intel Arc GPU if available.
                When False, forces CPU-only path.
    
    Returns:
        True if Intel Arc GPU is now active.
    """
    intel_arc_prefer = prefer
    if not prefer:
        intel_arc_active = False
        print("[MojoLlama] Intel Arc GPU: disabled (CPU mode)")
        return
    
    var available = intel_arc_is_available()
    if available:
        intel_arc_active = True
        var info = intel_arc_device_info()
        print("[MojoLlama] Intel Arc GPU: detected")
        for i in range(Int(py=len(info))):
            var dev = info[i]
            print("  Device " + String(i) + ": " + String(dev["name"]))
            var vram = Float64(py=dev.get("vram_gb", 0))
            print("  VRAM: " + String(vram) + " GB")
    else:
        intel_arc_active = False
        print("[MojoLlama] Intel Arc GPU: not available (using CPU)")
