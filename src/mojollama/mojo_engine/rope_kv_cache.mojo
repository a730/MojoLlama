# MojoLlama RoPE + KV Cache — Pure Mojo rotary embeddings
# WHAT:  RoPE (rotary positional embedding) for Q/K + KV cache + element-wise ops.
# WHY:   Replaces simd_ops.c. Every modern LLM (Llama, Mistral, Gemma) uses RoPE.
# WHEN:  May 2026 — pure Mojo port.
from std import math
from std.algorithm.backend.cpu.parallelize import parallelize

def apply_rope_inplace(q: UnsafePointer[Float32, MutExternalOrigin],
                        k: UnsafePointer[Float32, MutExternalOrigin],
                        n_heads: Int, n_kv_heads: Int, head_dim: Int, 
                        pos: Int, rope_theta: Float32):
    """Apply RoPE to Q and K in-place. For each pair (d,d+1), rotate by pos*freq."""
    var inv_theta = 1.0 / rope_theta
    for h in range(n_heads):
        for d in range(0, head_dim, 2):
            var half_d = Float32(d // 2) / Float32(head_dim)
            var freq = math.pow(inv_theta, half_d)
            var angle = Float32(pos) * freq
            var cos_a = math.cos(angle); var sin_a = math.sin(angle)
            var idx = h * head_dim + d
            var q0 = q.load(idx); var q1 = q.load(idx + 1)
            q.store(idx, q0 * cos_a - q1 * sin_a)
            q.store(idx + 1, q0 * sin_a + q1 * cos_a)
    for h in range(n_kv_heads):
        for d in range(0, head_dim, 2):
            var half_d = Float32(d // 2) / Float32(head_dim)
            var freq = math.pow(inv_theta, half_d)
            var angle = Float32(pos) * freq
            var cos_a = math.cos(angle); var sin_a = math.sin(angle)
            var idx = h * head_dim + d
            var k0 = k.load(idx); var k1 = k.load(idx + 1)
            k.store(idx, k0 * cos_a - k1 * sin_a)
            k.store(idx + 1, k0 * sin_a + k1 * cos_a)

def kv_cache_update(k_cache: UnsafePointer[Float32, MutExternalOrigin],
                    v_cache: UnsafePointer[Float32, MutExternalOrigin],
                    k_new: UnsafePointer[Float32, MutExternalOrigin],
                    v_new: UnsafePointer[Float32, MutExternalOrigin],
                    n_past: Int, n_kv_heads: Int, head_dim: Int):
    """Copy new K, V into cache at position n_past."""
    var offset = n_past * n_kv_heads * head_dim
    var n = n_kv_heads * head_dim
    for i in range(n):
        k_cache.store(offset + i, k_new.load(i))
        v_cache.store(offset + i, v_new.load(i))

def residual_add(a: UnsafePointer[Float32, MutExternalOrigin],
                 b: UnsafePointer[Float32, MutExternalOrigin],
                 o: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    """o = a + b (element-wise)"""
    for i in range(n):
        o.store(i, a.load(i) + b.load(i))

def rms_norm(x: UnsafePointer[Float32, MutExternalOrigin],
             w: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin],
             n: Int, eps: Float32 = 1e-6):
    """o[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)"""
    var ss: Float64 = 0.0
    for i in range(n): ss += Float64(x.load(i) * x.load(i))
    var rms = math.sqrt(Float32(ss / Float64(n)) + eps)
    var inv_rms = 1.0 / rms
    for i in range(n):
        o.store(i, x.load(i) * w.load(i) * inv_rms)
