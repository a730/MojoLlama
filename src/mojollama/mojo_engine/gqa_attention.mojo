# MojoLlama GQA Attention — Pure Mojo Grouped Query Attention
# WHAT:  GQA attention decode for single token. Replaces gqa_attention.c (135L)
#        which depended on cblas_sgemm (OpenBLAS). Pure Mojo SIMD + parallelize.
# WHY:   Remove OpenBLAS dependency. GQA is used by Llama 3, Gemma, Mistral, Qwen2.
# WHEN:  May 2026 — pure Mojo port. Uses scalar softmax + parallel KV heads.
from std import math
from std.algorithm.backend.cpu.parallelize import parallelize

comptime W: Int = 8

fn clamp_f32(x: Float32, lo: Float32, hi: Float32) -> Float32:
    if x < lo: return lo
    if x > hi: return hi
    return x

fn softmax(v: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    """In-place softmax on v[0:n]. Numerically stable: subtracts max first."""
    # Find max
    var m: Float32 = -1e30
    for i in range(n):
        if v.load(i) > m: m = v.load(i)
    # Compute exp(x - max) and sum
    var sum_exp: Float32 = 0.0
    for i in range(n):
        var e = math.exp(clamp_f32(v.load(i) - m, -80.0, 80.0))
        v.store(i, e)
        sum_exp += e
    # Normalize
    var inv_sum = 1.0 / (sum_exp + 1e-10)
    for i in range(n): v.store(i, v.load(i) * inv_sum)

def gqa_attention_decode(
    q: UnsafePointer[Float32, MutExternalOrigin],
    k_cache: UnsafePointer[Float32, MutExternalOrigin],
    v_cache: UnsafePointer[Float32, MutExternalOrigin],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    seq_len: Int, n_head: Int, n_kv_head: Int, head_dim: Int):
    """GQA attention for single-token decode. Pure Mojo, no cblas_sgemm."""
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    var gqa_rep = n_head // n_kv_head
    var scale = 1.0 / math.sqrt(Float32(head_dim))
    var kv_stride = n_kv_head * head_dim
    var scores = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_head * seq_len * 4))))
    
    def head_worker(kv_h: Int) capturing -> None:
        var qh = q + kv_h * gqa_rep * head_dim
        var kh = k_cache + kv_h * head_dim
        var vh = v_cache + kv_h * head_dim
        for qi in range(gqa_rep):
            var q_off = qi * head_dim; var h_idx = kv_h * gqa_rep + qi
            # scores[seq_len] = Q[HD] @ K[seq_len, HD]^T * scale
            for t in range(seq_len):
                var acc: Float32 = 0.0
                for d in range(head_dim):
                    acc += qh.load(q_off + d) * k_cache.load(kv_h * head_dim + t * kv_stride + d)
                scores.store(h_idx * seq_len + t, acc * scale)
            # Softmax
            softmax(scores + h_idx * seq_len, seq_len)
            # out[HD] = scores[seq_len] @ V[seq_len, HD]
            for d in range(head_dim):
                var acc: Float32 = 0.0
                for t in range(seq_len):
                    acc += (scores + h_idx * seq_len).load(t) * v_cache.load(kv_h * head_dim + t * kv_stride + d)
                out_ptr.store(h_idx * head_dim + d, acc)
    parallelize[func=head_worker](num_work_items=n_kv_head)
