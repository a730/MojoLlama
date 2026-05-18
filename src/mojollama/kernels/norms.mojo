"""MojoLlama Normalization Kernels v2 — stride-8 SIMD loops.

RMSNorm: x / sqrt(mean(x^2) + eps) * weight
RoPE: rotary position embeddings via complex rotation
SiLU: x * sigmoid(x) = x / (1 + exp(-x))

All use stride-8 F32x8 loops for full-dimension vectorization.
Memory layout: pointer-based for interoperability with C kernels.
"""
from std.memory.unsafe_pointer import alloc
from std.math import exp, sqrt, cos, sin, pow

alias F32x8 = SIMD[DType.float32, 8]

# ─── RMSNorm (full-dimension stride-8 SIMD) ────────────────────────────

def rms_norm(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    out: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    """Vectorized RMS norm over full dimension n.
    Uses F32x8 SIMD for sum-of-squares and output.
    Matches C kernel's hadd-based reduction pattern.
    """
    var ss: Float32 = 0.0
    var i = 0
    # SIMD sum of squares
    while i + 8 <= n:
        var v = x.load[width=8](i)
        ss += (v * v).reduce_add()
        i += 8
    # Scalar tail
    while i < n:
        ss += x.load(i) * x.load(i)
        i += 1
    var inv_rms = 1.0 / sqrt(ss / Float32(n) + 1e-6)
    var inv_v = F32x8(inv_rms)
    # SIMD output: out[i] = x[i] * inv_rms * weight[i]
    i = 0
    while i + 8 <= n:
        var v = x.load[width=8](i)
        var w = weight.load[width=8](i)
        out.store[width=8](i, v * inv_v * w)
        i += 8
    while i < n:
        out.store(i, x.load(i) * inv_rms * weight.load(i))
        i += 1

# ─── SiLU (full-dimension stride-8, scalar exp) ───────────────────────

def silu(
    x: UnsafePointer[Float32, MutAnyOrigin],
    out: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    """SiLU activation: out[i] = x[i] / (1 + exp(-x[i]))
    Uses F32x8 load/store, scalar exp per lane (AVX2 has no vector exp).
    Matches C kernel's approach.
    """
    var one = F32x8(1.0)
    var i = 0
    while i + 8 <= n:
        var v = x.load[width=8](i)
        var result = F32x8()
        for j in range(8):
            result[j] = v[j] / (1.0 + exp(-v[j]))
        out.store[width=8](i, result)
        i += 8
    while i < n:
        var val = x.load(i)
        out.store(i, val / (1.0 + exp(-val)))
        i += 1

# ─── RoPE (rotary position embeddings) ────────────────────────────────

def rope(
    x: UnsafePointer[Float32, MutAnyOrigin],
    out: UnsafePointer[Float32, MutAnyOrigin],
    pos: Int, head_dim: Int,
):
    """Apply rotary position embeddings.
    For each pair (x[i], x[i+half]):
      out[i]      = x[i] * cos(pos * freq_i) - x[i+half] * sin(pos * freq_i)
      out[i+half] = x[i] * sin(pos * freq_i) + x[i+half] * cos(pos * freq_i)
    where freq_i = 1 / 10000^(2i/d)
    """
    var half = head_dim // 2
    var i = 0
    while i + 8 <= half:
        for j in range(8):
            var idx = Float32(i + j)
            var freq = 1.0 / pow(10000.0, 2.0 * idx / Float32(head_dim))
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var x0 = x.load(i + j)
            var x1 = x.load(i + j + half)
            out.store(i + j, x0 * cos_a - x1 * sin_a)
            out.store(i + j + half, x0 * sin_a + x1 * cos_a)
        i += 8
    while i < half:
        var idx = Float32(i)
        var freq = 1.0 / pow(10000.0, 2.0 * idx / Float32(head_dim))
        var angle = Float32(pos) * freq
        var cos_a = cos(angle)
        var sin_a = sin(angle)
        var x0 = x.load(i)
        var x1 = x.load(i + half)
        out.store(i, x0 * cos_a - x1 * sin_a)
        out.store(i + half, x0 * sin_a + x1 * cos_a)
        i += 1
    # Copy remaining dimensions unchanged
    while i < head_dim:
        out.store(i, x.load(i))
        i += 1

# ─── Softmax (stride-8 max/sum, scalar exp) ─────────────────────────────

def softmax(
    x: UnsafePointer[Float32, MutAnyOrigin],
    out: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    """Vectorized softmax: find max (SIMD), exp(x-max) (scalar), normalize (SIMD).
    Matches C kernel's AVX2 hadd-based max reduction pattern.
    """
    # Find max for numerical stability
    var max_val: Float32 = -1e30
    var i = 0
    while i + 8 <= n:
        var v = x.load[width=8](i)
        # SIMD max reduction
        var m = v.reduce_max()
        if m > max_val: max_val = m
        i += 8
    while i < n:
        if x.load(i) > max_val: max_val = x.load(i)
        i += 1

    # exp(x - max) and sum
    var sum: Float32 = 0.0
    i = 0
    while i + 8 <= n:
        var v = x.load[width=8](i)
        var result = F32x8()
        for j in range(8):
            result[j] = exp(v[j] - max_val)
        out.store[width=8](i, result)
        sum += result.reduce_add()
        i += 8
    while i < n:
        var val = exp(x.load(i) - max_val)
        out.store(i, val)
        sum += val
        i += 1

    # Normalize
    var inv_sum = 1.0 / sum
    var inv_v = F32x8(inv_sum)
    i = 0
    while i + 8 <= n:
        var v = out.load[width=8](i)
        out.store[width=8](i, v * inv_v)
        i += 8
    while i < n:
        out.store(i, out.load(i) * inv_sum)
        i += 1