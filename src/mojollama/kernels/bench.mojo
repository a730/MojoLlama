"""Real Mojo SIMD benchmark — measures actual compute, not optimized-away loops."""
from std.memory.unsafe_pointer import alloc
from std.math import sqrt

alias F32x8 = SIMD[DType.float32, 8]
alias U8x16 = SIMD[DType.uint8, 16]

# ─── F16→F32 ──────────────────────────────────────────────────────────

def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var s = (UInt32(h) >> 15) & 1
        var e = (UInt32(h) >> 10) & 0x1f
        var m = UInt32(h) & 0x3ff
        if e == 0:
            if m == 0: return Float32(bitcast[DType.float32](s << 31))
            var mm = m; var c: UInt32 = 0
            while mm > 0: mm >>= 1; c += 1
            var sh = 24 - c
            return Float32(bitcast[DType.float32]((s << 31) | ((UInt32(113 - sh)) << 23) | ((m << (sh + 13)) & 0x7fffff)))
        if e == 31: return Float32(bitcast[DType.float32]((s << 31) | 0x7f800000 | (m << 13)))
        return Float32(bitcast[DType.float32]((s << 31) | ((e + 112) << 23) | (m << 13)))


# ─── Q4_0 Matmul ──────────────────────────────────────────────────────

def q4_block_dot(scale: Float32, nibbles: U8x16,
                 x0: F32x8, x1: F32x8, x2: F32x8, x3: F32x8) -> Float32:
    @parameter
    def dg(s: Int, xv: F32x8) -> Float32:
        var v = F32x8()
        for j in range(8):
            var bi = s + j // 2
            var b = nibbles[bi]
            v[j] = Float32(Int8(b & 15) - 8) if j % 2 == 0 else Float32(Int8((b >> 4) & 15) - 8)
        return (v * scale * xv).reduce_add()
    return dg(0, x0) + dg(4, x1) + dg(8, x2) + dg(12, x3)


def q4_matmul(
    w: UnsafePointer[mut=True, type=UInt8, origin=_],
    inp: UnsafePointer[mut=False, type=Float32, origin=_],
    result: UnsafePointer[mut=True, type=Float32, origin=_],
    n_rows: Int, n_cols: Int, checksum: UnsafePointer[mut=True, type=Float32, origin=_],
):
    var bpr = n_cols // 32
    var total_chk: Float32 = 0.0
    for row in range(n_rows):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = w.load(off)
            var hi = w.load(off + 1)
            var scale = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nibs = w.load[width=16](off + 2)
            var inp_off = blk * 32
            var x0 = inp.load[width=8](inp_off)
            var x1 = inp.load[width=8](inp_off + 8)
            var x2 = inp.load[width=8](inp_off + 16)
            var x3 = inp.load[width=8](inp_off + 24)
            total += q4_block_dot(scale, nibs, x0, x1, x2, x3)
        result.store(row, total)
        total_chk += total
    checksum.store(0, total_chk)


# ─── RMSNorm ───────────────────────────────────────────────────────────

def rms_norm(
    x: UnsafePointer[mut=False, type=Float32, origin=_],
    w: UnsafePointer[mut=False, type=Float32, origin=_],
    result: UnsafePointer[mut=True, type=Float32, origin=_],
    n: Int, eps: Float32, checksum: UnsafePointer[mut=True, type=Float32, origin=_],
):
    var ss: Float32 = 0.0
    for i in range(n):
        var v = x.load(i)
        ss += v * v
    var inv_rms = Float32(1.0 / sqrt(Float64(ss) / Float64(n) + Float64(eps)))
    var ck: Float32 = 0.0
    for i in range(n):
        var v = x.load(i) * w.load(i) * inv_rms
        result.store(i, v)
        ck += v
    checksum.store(0, ck)


# ─── SiLU ──────────────────────────────────────────────────────────────

def fast_exp(x: Float32) -> Float32:
    var v = 1.0 + x * 0.000244140625
    for _ in range(12): v = v * v
    return v

def silu_act(
    x: UnsafePointer[mut=False, type=Float32, origin=_],
    result: UnsafePointer[mut=True, type=Float32, origin=_],
    n: Int, checksum: UnsafePointer[mut=True, type=Float32, origin=_],
):
    var ck: Float32 = 0.0
    for i in range(n):
        var v = x.load(i) / (1.0 + fast_exp(-x.load(i)))
        result.store(i, v)
        ck += v
    checksum.store(0, ck)


# ─── Benchmark ─────────────────────────────────────────────────────────

def main() raises:
    from python import Python
    var tim = Python.import_module("time")
    
    var n_embd: Int = 2048
    var n_head: Int = 32
    var n_kv_head: Int = 8
    var n_ff: Int = 8192
    var head_dim: Int = 64
    var n_cols: Int = n_embd
    var n_rows_q: Int = n_head * head_dim  # 2048
    var n_rows_ff: Int = n_ff  # 8192
    
    print("MojoLlama SIMD — Real Benchmarks")
    print("Mojo 1.0.0b1, AVX2:", end="")
    from std.sys.info import CompilationTarget
    print("YES" if CompilationTarget.has_avx2() else "NO")
    print()
    
    # Allocate test data
    var inp = alloc[Float32](n_embd)
    var result = alloc[Float32](2048)
    var norm_w = alloc[Float32](n_embd)
    var chk = alloc[Float32](1)
    
    for i in range(n_embd):
        inp.store(i, Float32(0.5))
        norm_w.store(i, Float32(1.0))
    
    # Allocate Q4_0 weight buffer (simulated: all zeros → dequant to -8)
    var n_blocks = n_rows_q * (n_cols // 32)
    var w = alloc[UInt8](n_blocks * 18)
    for i in range(n_blocks * 18):
        w.store(i, UInt8(0))  # scale=0 (f16→0), nibbles=0 (dequant→-8)
    
    # Real benchmark RMSNorm
    print("=== RMSNorm (n=2048) ===")
    var t0 = tim.time()
    for iter in range(1000):
        rms_norm(inp, norm_w, result, n_embd, 1e-5, chk)
    var t1 = tim.time()
    var rms_elapsed = t1 - t0
    print("  1000 iterations:", rms_elapsed, "s")
    print("  per iteration:", rms_elapsed / 1000 * 1e6, "us")
    print("  checksum:", chk.load(0))
    
    # Real benchmark SiLU
    print("\n=== SiLU (n=2048) ===")
    t0 = tim.time()
    for iter in range(1000):
        silu_act(inp, result, n_embd, chk)
    t1 = tim.time()
    var silu_elapsed = t1 - t0
    print("  1000 iterations:", silu_elapsed, "s")
    print("  per iteration:", silu_elapsed / 1000 * 1e6, "us")
    print("  checksum:", chk.load(0))
    
    # Real benchmark Q4_0 matmul (2048×2048)
    print("\n=== Q4_0 Matmul (2048×2048) ===")
    t0 = tim.time()
    for iter in range(10):
        q4_matmul(w, inp, result, n_rows_q, n_cols, chk)
    t1 = tim.time()
    var elapsed_qkv = t1 - t0
    print("  10 iterations:", elapsed_qkv, "s")
    print("  per iteration:", elapsed_qkv / 10 * 1e3, "ms")
    print("  checksum:", chk.load(0))
    
    # Benchmark Q4_0 matmul (8192×2048 - FFN size)
    print("\n=== Q4_0 Matmul (8192×2048 - FFN) ===")
    var w2 = alloc[UInt8](n_rows_ff * (n_cols // 32) * 18)
    var result2 = alloc[Float32](n_rows_ff)
    t0 = tim.time()
    for iter in range(5):
        q4_matmul(w2, inp, result2, n_rows_ff, n_cols, chk)
    t1 = tim.time()
    var ffn_elapsed = t1 - t0
    print("  5 iterations:", ffn_elapsed, "s")
    print("  per iteration:", ffn_elapsed / 5 * 1e3, "ms")
    print("  checksum:", chk.load(0))
    var ffn_matmul_time = ffn_elapsed / 5
    var qkv_matmul_time = elapsed_qkv / 10
    result2.free()
    w2.free()
    
    # Full forward estimate
    print("\n=== Estimated Full Forward Pass ===")
    # Per layer: 6 matmuls + 2 rmsnorm + 1 silu + attention  
    var layer_time = ffn_matmul_time + 5 * qkv_matmul_time + 2 * (rms_elapsed / 1000) + (silu_elapsed / 1000)
    var fwd_time = layer_time * 16
    print("  Estimated time per forward pass (16 layers):", fwd_time * 1e3, "ms")
    print("  Estimated throughput:", 1.0 / fwd_time, "tok/s")
    
    inp.free()
    result.free()
    norm_w.free()
    w.free()
    chk.free()
    print("\nDone.")
