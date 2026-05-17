"""MojoLlama Attention Kernel — CPU MHA with SIMD softmax.

Multi-head attention for Llama3: softmax(Q @ K.T / sqrt(d)) @ V.
When heap APIs land, integrates with MAX's mha.mojo from:
  /tmp/modular/max/kernels/src/nn/attention/cpu/mha.mojo

For now: stack-allocated SIMD vectors and scalar fallbacks for verification.
"""

from std.sys.info import CompilationTarget
from std.math import exp

alias F32x8 = SIMD[DType.float32, 8]


# ─── Softmax (single vector) ────────────────────────────────────────────

@always_inline
fn softmax_max(x: F32x8) -> Float32:
    """Horizontal max of 8 float32 values."""
    var m = x[0]
    for i in range(1, 8):
        if x[i] > m:
            m = x[i]
    return m

@always_inline
fn softmax_sum(x: F32x8) -> Float32:
    """Horizontal sum of 8 float32 values."""
    var s: Float32 = 0.0
    for i in range(8):
        s += x[i]
    return s

@always_inline
fn softmax(x: F32x8) -> F32x8:
    """In-place softmax for one SIMD vector of 8 values.
    
    Numerically stable: subtracts max before exp.
    Reference: MAX's softmax.mojo
    """
    var m = softmax_max(x)
    var e = F32x8()
    for i in range(8):
        e[i] = exp(x[i] - m)
    var s = softmax_sum(e)
    var inv_s = 1.0 / s
    for i in range(8):
        e[i] = e[i] * inv_s
    return e


# ─── Attention Score (single head, small kv) ────────────────────────────

@always_inline
fn attn_score(q: F32x8, k: F32x8) -> Float32:
    """Dot product of query and key: q @ k for one head.
    
    One step of Q @ K.T for attention scores.
    Both are 8-dimensional float32 vectors.
    """
    var s: Float32 = 0.0
    for i in range(8):
        s += q[i] * k[i]
    return s


@always_inline
fn attn_scores_head(q: F32x8, keys: AnyType, n_keys: Int) -> F32x8:
    """Compute attention scores for one head against N keys.
    
    Returns a SIMD vector of scores. For n_keys <= 8, returns F32x8.
    For larger n_keys, this would be a tiled approach.
    
    NOTE: AnyType not subscriptable in 0.26.2 — placeholder for pointer-based version.
    """
    # In 0.26.2: can't index AnyType. This becomes pointer-based when APIs land.
    # For now, the tests use fixed-size hardcoded values.
    return F32x8()


# ─── Full Attention Step (simplified for verification) ───────────────────

@always_inline
fn attention_step_single(q: F32x8, k0: F32x8, k1: F32x8, 
                          v0: F32x8, v1: F32x8, mask: F32x8) -> F32x8:
    """Single-head attention: softmax(Q @ K.T / sqrt(d)) @ V
    
    Simplified for 2 key vectors (n_keys=2), 8-dim hidden.
    Verifies correctness of the softmax + weighted sum pattern.
    
    Args:
        q: query vector (8-dim)
        k0, k1: key vectors
        v0, v1: value vectors
        mask: attention mask (-inf for masked positions)
    
    Returns:
        output vector (8-dim) = weighted sum of values
    """
    # Scores: s_i = q @ k_i / sqrt(8)
    var s0 = attn_score(q, k0) / 2.828 + mask[0]
    var s1 = attn_score(q, k1) / 2.828 + mask[1]
    
    # Softmax
    var m = s0 if s0 > s1 else s1
    var e0 = exp(s0 - m)
    var e1 = exp(s1 - m)
    var inv_sum = 1.0 / (e0 + e1)
    var a0 = e0 * inv_sum
    var a1 = e1 * inv_sum
    
    # Apply: out = Σ a_i * v_i
    var out = F32x8()
    for i in range(8):
        out[i] = a0 * v0[i] + a1 * v1[i]
    
    return out


# ─── Tests ──────────────────────────────────────────────────────────────

fn test_softmax():
    """Test softmax with known values.
    
    Input: [2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    Max = 3.0
    exp([-1, -2, -3, -4, -3, -2, -1, 0]) = [0.368, 0.135, 0.050, 0.018, 0.050, 0.135, 0.368, 1.0]
    Sum = 2.124
    """
    var x = F32x8(2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    var s = softmax(x)
    print("softmax([2,1,0,-1,0,1,2,3]):")
    var total: Float32 = 0.0
    for i in range(8):
        print("  [", i, "]:", s[i])
        total += s[i]
    print("  sum:", total, "(expect ~1.0)")
    if abs(total - 1.0) < 0.01:
        print("  PASS")
    else:
        print("  FAIL")


fn test_attention():
    """Test single-head attention with known values.
    
    q = [1, 0, 0, 0, 0, 0, 0, 0]  (query looking at first dim)
    k0 = [1, 0, 0, 0, 0, 0, 0, 0]  (key matching q)
    k1 = [0, 1, 0, 0, 0, 0, 0, 0]  (key not matching q)
    v0 = [2, 0, 0, 0, 0, 0, 0, 0]  (value for first key)
    v1 = [0, 0, 0, 0, 0, 0, 0, 0]  (value for second key)
    mask = [0, 0]  (no masking)
    
    Expected: s0 = 1/2.828 = 0.354, s1 = 0/2.828 = 0
    Softmax: ~0.587, ~0.413
    Output: ~0.587 * v0 + 0.413 * v1 = [1.174, 0, ...]
    """
    var q = F32x8(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var k0 = F32x8(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var k1 = F32x8(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var v0 = F32x8(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var v1 = F32x8(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var mask = F32x8(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    var out = attention_step_single(q, k0, k1, v0, v1, mask)
    print("attention output[0]:", out[0], "(expect ~1.174)")
    if abs(out[0] - 1.174) < 0.01:
        print("  PASS")
    else:
        print("  FAIL")


fn main():
    print("=== MojoLlama Attention Kernel ===")
    print()
    
    test_softmax()
    print()
    test_attention()
    print()
    print("Done.")
