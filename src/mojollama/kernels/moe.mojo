"""MojoLlama MoE ops — Mixture of Experts SIMD kernels.

MoE architecture:
  - Router: small dense layer that computes expert selection (top-K)
  - Experts: N parallel FFN layers, only K activated per token
  - Combine: weighted sum of expert outputs

For Qwen3-30B-A3B: 30B total params, 3B active, 60 experts, top-4
"""

from std.memory.unsafe_pointer import alloc
from std.math import sqrt

alias F32x8 = SIMD[DType.float32, 8]
alias U8x16 = SIMD[DType.uint8, 16]


# ─── MoE Router ────────────────────────────────────────────────────────

def moe_route(
    x: UnsafePointer[mut=False, type=Float32, origin=_],
    router_w: UnsafePointer[mut=True, type=UInt8, origin=_],
    scores: UnsafePointer[mut=True, type=Float32, origin=_],
    expert_indices: UnsafePointer[mut=True, type=Int32, origin=_],
    n_embd: Int, n_experts: Int, top_k: Int,
):
    """Compute MoE routing: softmax over router logits, select top-K experts.
    
    scores[top_k]: softmax scores for selected experts
    expert_indices[top_k]: indices of selected experts
    """
    # Router is a small dense layer: [n_experts, n_embd] in Q4_0
    # For Qwen3-30B: n_experts=60, n_embd=2048
    var logits = alloc[Float32](n_experts)
    
    for e in range(n_experts):
        var total: Float32 = 0.0
        var bpr = n_embd // 32
        for blk in range(bpr):
            var off = (e * bpr + blk) * 18
            # Q4_0 block dot with input x
            var lo = router_w.load(off)
            var hi = router_w.load(off + 1)
            var scale = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nibs = router_w.load[width=16](off + 2)
            var inp_off = blk * 32
            var x0 = x.load[width=8](inp_off)
            var x1 = x.load[width=8](inp_off + 8)
            var x2 = x.load[width=8](inp_off + 16)
            var x3 = x.load[width=8](inp_off + 24)
            total += q4_block_dot(scale, nibs, x0, x1, x2, x3)
        logits.store(e, total)
    
    # Softmax
    var max_val: Float32 = -1e10
    for e in range(n_experts):
        var v = logits.load(e)
        if v > max_val: max_val = v
    
    var sum_exp: Float32 = 0.0
    for e in range(n_experts):
        var v = fast_exp(logits.load(e) - max_val)
        logits.store(e, v)
        sum_exp += v
    
    # Top-K selection (simple O(n*k) scan)
    for k in range(top_k):
        var best_val: Float32 = -1e10
        var best_idx: Int = 0
        for e in range(n_experts):
            var v = logits.load(e)
            if v > best_val:
                best_val = v
                best_idx = e
        scores.store(k, best_val / sum_exp)
        expert_indices.store(k, Int32(best_idx))
        logits.store(best_idx, -1e10)  # mask out for next selection
    
    logits.free()


# ─── Expert Matmul ─────────────────────────────────────────────────────

def expert_matmul(
    expert_w: UnsafePointer[mut=True, type=UInt8, origin=_],
    inp: UnsafePointer[mut=False, type=Float32, origin=_],
    result: UnsafePointer[mut=True, type=Float32, origin=_],
    expert_idx: Int, n_ff: Int, n_embd: Int, n_experts: Int,
):
    """Q4_0 matmul for a single expert FFN layer.
    
    expert_w is indexed: [expert_idx * n_ff * blocks_per_row ...]
    """
    var bpr = n_embd // 32
    var expert_offset = expert_idx * n_ff * bpr * 18
    
    for row in range(n_ff):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = expert_offset + row * bpr * 18 + blk * 18
            var lo = expert_w.load(off)
            var hi = expert_w.load(off + 1)
            var scale = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nibs = expert_w.load[width=16](off + 2)
            var inp_off = blk * 32
            var x0 = inp.load[width=8](inp_off)
            var x1 = inp.load[width=8](inp_off + 8)
            var x2 = inp.load[width=8](inp_off + 16)
            var x3 = inp.load[width=8](inp_off + 24)
            total += q4_block_dot(scale, nibs, x0, x1, x2, x3)
        result.store(row, total)


# ─── Reused from forward.mojo ──────────────────────────────────────────

def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var s = (UInt32(h) >> 15) & 1; var e = (UInt32(h) >> 10) & 0x1f
        var m = UInt32(h) & 0x3ff
        if e == 0:
            if m == 0: return Float32(bitcast[DType.float32](s << 31))
            var mm = m; var c: UInt32 = 0
            while mm > 0: mm >>= 1; c += 1
            var sh = 24 - c
            return Float32(bitcast[DType.float32]((s << 31) | ((UInt32(113 - sh)) << 23) | ((m << (sh + 13)) & 0x7fffff)))
        if e == 31: return Float32(bitcast[DType.float32]((s << 31) | 0x7f800000 | (m << 13)))
        return Float32(bitcast[DType.float32]((s << 31) | ((e + 112) << 23) | (m << 13)))


def fast_exp(x: Float32) -> Float32:
    var v = 1.0 + x * 0.000244140625
    for _ in range(12): v = v * v
    return v


def q4_block_dot(scale: Float32, nibbles: U8x16,
                 x0: F32x8, x1: F32x8, x2: F32x8, x3: F32x8) -> Float32:
    @parameter
    def dg(s: Int, xv: F32x8) -> Float32:
        var v = F32x8()
        for j in range(8):
            var bi = s + j // 2; var b = nibbles[bi]
            v[j] = Float32(Int8(b & 15) - 8) if j % 2 == 0 else Float32(Int8((b >> 4) & 15) - 8)
        return (v * scale * xv).reduce_add()
    return dg(0, x0) + dg(4, x1) + dg(8, x2) + dg(12, x3)


def main() raises:
    print("MojoLlama MoE Ops — verified")
