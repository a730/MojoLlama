"""MojoLlama Normalization Kernels — RMSNorm and RoPE.

RMSNorm: x / sqrt(mean(x^2) + eps) * weight
RoPE: rotary position embeddings via complex rotation

Both use SIMD vector types. Works on stack data in Mojo 0.26.2.
Becomes pointer-based when heap APIs land.
"""

from std.sys.info import CompilationTarget
from std.math import exp, sqrt

alias F32x8 = SIMD[DType.float32, 8]


# ─── RMSNorm (single vector) ────────────────────────────────────────────

@always_inline
fn rms_norm_vec(x: F32x8, weight: F32x8, eps: Float32) -> F32x8:
    """RMSNorm for one vector of 8 float32 values.
    
    out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
    
    Reference: MAX's rms_norm_ops.mojo
    """
    # mean(x^2)
    var ss: Float32 = 0.0
    for i in range(8):
        ss += x[i] * x[i]
    var mean_sq = ss / 8.0
    
    # 1 / sqrt(mean_sq + eps)
    var inv_rms = 1.0 / sqrt(mean_sq + eps)
    
    # out = x * inv_rms * weight
    var out = F32x8()
    for i in range(8):
        out[i] = x[i] * inv_rms * weight[i]
    return out


# ─── RMSNorm (pointer-based, future) ────────────────────────────────────
# When heap APIs land: takes pointer to N floats, returns pointer to output.
# For now: use rms_norm_vec for fixed-size SIMD vectors.


# ─── RoPE (single vector pair) ──────────────────────────────────────────

@always_inline
fn rope_pair(x0: Float32, x1: Float32, cos_val: Float32, sin_val: Float32) -> F32x8:
    """Apply RoPE rotation to one pair of dimensions.
    
    RoPE rotates a 2D vector (x0, x1) by angle theta:
        out0 = x0 * cos(theta) - x1 * sin(theta)
        out1 = x0 * sin(theta) + x1 * cos(theta)
    
    Reference: MAX's rotary_embedding.mojo
    """
    var out = F32x8()
    out[0] = x0 * cos_val - x1 * sin_val
    out[1] = x0 * sin_val + x1 * cos_val
    return out


@always_inline
fn rope_vec(x: F32x8, cos_vals: F32x8, sin_vals: F32x8) -> F32x8:
    """Apply RoPE to an 8-dim vector (4 pairs of dims).
    
    x has shape [d0, d1, d2, d3, d4, d5, d6, d7]
    RoPE rotates pairs (d0,d1), (d2,d3), (d4,d5), (d6,d7)
    cos_vals has cos(theta/2) for each pair
    sin_vals has sin(theta/2) for each pair
    
    Using complex rotation:
        out[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
        out[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]
    """
    var out = F32x8()
    for i in range(4):
        var c = cos_vals[i]
        var s = sin_vals[i]
        out[i * 2]     = x[i * 2] * c - x[i * 2 + 1] * s
        out[i * 2 + 1] = x[i * 2] * s + x[i * 2 + 1] * c
    return out


# ─── SiLU activation ────────────────────────────────────────────────────

@always_inline
fn silu_vec(x: F32x8) -> F32x8:
    """SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))"""
    var out = F32x8()
    for i in range(8):
        out[i] = x[i] / (1.0 + exp(-x[i]))
    return out


# ─── Tests ──────────────────────────────────────────────────────────────

fn approx_eq(a: Float32, b: Float32, tol: Float32 = 0.01) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    return d < tol

fn test_rms_norm():
    """RMSNorm with known values.
    
    x = [3, 1, 0, 0, 0, 0, 0, 0], weight = [1,1,1,...], eps = 1e-6
    mean_sq = (9 + 1 + 0 + ...) / 8 = 10/8 = 1.25
    inv_rms = 1/sqrt(1.25 + 1e-6) = 0.8944
    out[0] = 3 * 0.8944 * 1 = 2.683
    out[1] = 1 * 0.8944 * 1 = 0.894
    """
    var x = F32x8(3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var w = F32x8(1.0)
    var out = rms_norm_vec(x, w, 1e-6)
    print("rms_norm[0]:", out[0], "(expect 2.683)")
    print("rms_norm[1]:", out[1], "(expect 0.894)")
    if approx_eq(out[0], 2.683) and approx_eq(out[1], 0.894):
        print("  PASS")
    else:
        print("  FAIL")

fn test_rope():
    """RoPE with known values.
    
    x = [1, 0, 0, 0, 0, 0, 0, 0]  (unit vector along dim 0)
    theta = 0 (cos=1, sin=0) → no rotation
    Expected: output = input = [1, 0, 0, ...]
    
    theta = pi/2 (cos=0, sin=1):
    out[0] = 1*0 - 0*1 = 0
    out[1] = 1*1 + 0*0 = 1
    Expected: [0, 1, 0, ...]
    """
    var x = F32x8(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # No rotation
    var cos0 = F32x8(1.0)
    var sin0 = F32x8(0.0)
    var out0 = rope_vec(x, cos0, sin0)
    print("rope(theta=0)[0]:", out0[0], "(expect 1.0)")
    print("rope(theta=0)[1]:", out0[1], "(expect 0.0)")
    
    # 90 degree rotation (pi/2)
    var cos1 = F32x8(0.0)
    var sin1 = F32x8(1.0)
    var out1 = rope_vec(x, cos1, sin1)
    print("rope(theta=pi/2)[0]:", out1[0], "(expect 0.0)")
    print("rope(theta=pi/2)[1]:", out1[1], "(expect 1.0)")
    
    if approx_eq(out0[0], 1.0) and approx_eq(out1[1], 1.0):
        print("  PASS")
    else:
        print("  FAIL")

fn test_silu():
    """SiLU with known values.
    
    silu(0) = 0 / (1 + exp(0)) = 0 / 2 = 0
    silu(1) = 1 / (1 + exp(-1)) = 1 / (1 + 0.368) = 0.731
    silu(-1) = -1 / (1 + exp(1)) = -1 / (1 + 2.718) = -0.269
    """
    var x = F32x8(0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var out = silu_vec(x)
    print("silu(0):", out[0], "(expect 0.0)")
    print("silu(1):", out[1], "(expect 0.731)")
    print("silu(-1):", out[2], "(expect -0.269)")
    if approx_eq(out[0], 0.0) and approx_eq(out[1], 0.731) and approx_eq(out[2], -0.269):
        print("  PASS")
    else:
        print("  FAIL")


fn main():
    print("=== MojoLlama Normalization Kernels ===")
    print()
    
    test_rms_norm()
    print()
    test_rope()
    print()
    test_silu()
    print()
    print("Done.")
