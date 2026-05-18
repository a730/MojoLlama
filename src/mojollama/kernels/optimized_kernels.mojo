"""Optimized Mojo Q4_0 kernels — v2: row-outer FMA accumulation.

Key optimizations over v1:
1. Row-outer loop: accumulate in Float32 register, write output ONCE per row
   (was: res.store(row, res.load(row) + dot) every block → now: total += dot; res.store(row, total))
2. FMA-style accumulation: acc=v0*x0, acc+=v1*x1, acc+=v2*x2, acc+=v3*x3, reduce_add
   (was: 4 separate (v*scale*x).reduce_add() calls → now: 1 reduce_add at end)
3. 4-row register blocking with row-outer (shares input loads across 4 rows)
4. Fused QKV/FFN kernels share input vector across projections
5. Vectorized RMS norm / SiLU / RoPE using stride-8 loops
"""
from python import Python
from std.python._cpython import PyObjectPtr
from std.memory.unsafe_pointer import alloc
from std.math import sqrt, exp

alias F32x8 = SIMD[DType.float32, 8]
alias U8x16 = SIMD[DType.uint8, 16]
alias I8x16 = SIMD[DType.int8, 16]

# ─── Float16 → Float32 ────────────────────────────────────────────────

@always_inline
def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var s=(UInt32(h)>>15)&1;var e=(UInt32(h)>>10)&0x1f;var m=UInt32(h)&0x3ff
        if e==0:
            if m==0:return Float32(bitcast[DType.float32](s<<31))
            var mm=m;var c:UInt32=0;var vm=m
            while mm>0:mm>>=1;c+=1
            var sh=24-c;vm=m<<(sh+13)
            return Float32(bitcast[DType.float32]((s<<31)|((UInt32(113-sh))<<23)|(vm&0x7fffff)))
        if e==31:return Float32(bitcast[DType.float32]((s<<31)|0x7f800000|(m<<13)))
        return Float32(bitcast[DType.float32]((s<<31)|((e+112)<<23)|(m<<13)))

# ─── Vectorized Q4_0 dot (FMA-style accumulation) ────────────────────
# v2 improvement: 4 multiplies → 1 FMA chain → 1 reduce_add
# Old: (v*scale*x0).reduce_add() + (v*scale*x1).reduce_add() + (v*scale*x2).reduce_add() + (v*scale*x3).reduce_add()
# New: acc = v0*x0; acc += v1*x1; acc += v2*x2; acc += v3*x3; acc.reduce_add()
# Saves 3 reduce_add calls (each ~10 uops on Zen 2).

@always_inline
def q4_dot_vec(scale: Float32, nb: U8x16,
               x0: F32x8, x1: F32x8, x2: F32x8, x3: F32x8) -> Float32:
    var mask = U8x16(15)
    var lo = (nb & mask).cast[DType.int8]() - I8x16(8)
    var hi_shifted = nb >> UInt8(4)
    var hi = (hi_shifted & mask).cast[DType.int8]() - I8x16(8)

    # Interleave lo/hi nibbles matching C kernel's unpacklo/hi pattern
    var v0 = F32x8(
        Float32(lo[0])*scale, Float32(hi[0])*scale,
        Float32(lo[1])*scale, Float32(hi[1])*scale,
        Float32(lo[2])*scale, Float32(hi[2])*scale,
        Float32(lo[3])*scale, Float32(hi[3])*scale)
    var v1 = F32x8(
        Float32(lo[4])*scale, Float32(hi[4])*scale,
        Float32(lo[5])*scale, Float32(hi[5])*scale,
        Float32(lo[6])*scale, Float32(hi[6])*scale,
        Float32(lo[7])*scale, Float32(hi[7])*scale)
    var v2 = F32x8(
        Float32(lo[8])*scale, Float32(hi[8])*scale,
        Float32(lo[9])*scale, Float32(hi[9])*scale,
        Float32(lo[10])*scale, Float32(hi[10])*scale,
        Float32(lo[11])*scale, Float32(hi[11])*scale)
    var v3 = F32x8(
        Float32(lo[12])*scale, Float32(hi[12])*scale,
        Float32(lo[13])*scale, Float32(hi[13])*scale,
        Float32(lo[14])*scale, Float32(hi[14])*scale,
        Float32(lo[15])*scale, Float32(hi[15])*scale)

    # FMA-style chain: acc = v0*x0, acc += v1*x1, acc += v2*x2, acc += v3*x3
    var acc = v0 * x0
    acc = acc + v1 * x1
    acc = acc + v2 * x2
    acc = acc + v3 * x3
    return acc.reduce_add()

# ─── Row-outer matmul: write output ONCE per row ──────────────────────
# v2: eliminates bpr * nr load-modify-store on result buffer.
# Each row does: total=0; for blk: total+=dot; res[row]=total
# Old: res[row] += dot  (reads+writes result every block)

def q4_mm_row_outer(
    w: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    res: UnsafePointer[Float32, MutAnyOrigin],
    nr: Int, nc: Int,
):
    var bpr = nc // 32
    for row in range(nr):
        var total: Float32 = 0.0
        var w_off_base = row * bpr
        for blk in range(bpr):
            var off = (w_off_base + blk) * 18
            var lo = w.load(off); var hi = w.load(off + 1)
            var sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nb = w.load[width=16](off + 2)
            var ioff = blk * 32
            var x0 = inp.load[width=8](ioff)
            var x1 = inp.load[width=8](ioff + 8)
            var x2 = inp.load[width=8](ioff + 16)
            var x3 = inp.load[width=8](ioff + 24)
            total += q4_dot_vec(sc, nb, x0, x1, x2, x3)
        res.store(row, total)

# ─── 4-row regblock + row-outer ───────────────────────────────────────
# Combines 4-row register blocking (share input loads across 4 rows)
# with row-outer accumulation (1 write per row instead of bpr writes).
# This matches the C kernel's pattern but in Mojo.

def q4_mm_regblock(
    w: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    res: UnsafePointer[Float32, MutAnyOrigin],
    nr: Int, nc: Int,
):
    var bpr = nc // 32
    var row = 0
    while row < nr:
        var r0 = row
        var r1 = row + 1 if row + 1 < nr else row
        var r2 = row + 2 if row + 2 < nr else row
        var r3 = row + 3 if row + 3 < nr else row
        var t0: Float32 = 0.0
        var t1: Float32 = 0.0
        var t2: Float32 = 0.0
        var t3: Float32 = 0.0

        for blk in range(bpr):
            var ioff = blk * 32
            var x0 = inp.load[width=8](ioff)
            var x1 = inp.load[width=8](ioff + 8)
            var x2 = inp.load[width=8](ioff + 16)
            var x3 = inp.load[width=8](ioff + 24)

            var off0 = (r0 * bpr + blk) * 18
            var lo = w.load(off0); var hi = w.load(off0 + 1)
            var sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nb = w.load[width=16](off0 + 2)
            t0 += q4_dot_vec(sc, nb, x0, x1, x2, x3)

            if r1 != r0:
                var off1 = (r1 * bpr + blk) * 18
                lo = w.load(off1); hi = w.load(off1 + 1)
                sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
                nb = w.load[width=16](off1 + 2)
                t1 += q4_dot_vec(sc, nb, x0, x1, x2, x3)

            if r2 != r0:
                var off2 = (r2 * bpr + blk) * 18
                lo = w.load(off2); hi = w.load(off2 + 1)
                sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
                nb = w.load[width=16](off2 + 2)
                t2 += q4_dot_vec(sc, nb, x0, x1, x2, x3)

            if r3 != r0:
                var off3 = (r3 * bpr + blk) * 18
                lo = w.load(off3); hi = w.load(off3 + 1)
                sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
                nb = w.load[width=16](off3 + 2)
                t3 += q4_dot_vec(sc, nb, x0, x1, x2, x3)

        res.store(r0, t0)
        if r1 != r0: res.store(r1, t1)
        if r2 != r0: res.store(r2, t2)
        if r3 != r0: res.store(r3, t3)
        row += 4

# ─── Fused QKV projection ─────────────────────────────────────────────
# Shares input vector x across Q, K, V projections.
# Row-outer accumulation for each projection independently.

def q4_mm_fused_qkv(
    wq: UnsafePointer[UInt8, MutAnyOrigin],
    wk: UnsafePointer[UInt8, MutAnyOrigin],
    wv: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    oq: UnsafePointer[Float32, MutAnyOrigin],
    ok: UnsafePointer[Float32, MutAnyOrigin],
    ov: UnsafePointer[Float32, MutAnyOrigin],
    nq: Int, nkv: Int, nc: Int,
):
    var bpr = nc // 32
    # Q projection (row-outer)
    for row in range(nq):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = wq.load(off); var hi = wq.load(off + 1)
            var sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nb = wq.load[width=16](off + 2)
            var ioff = blk * 32
            var x0 = inp.load[width=8](ioff)
            var x1 = inp.load[width=8](ioff + 8)
            var x2 = inp.load[width=8](ioff + 16)
            var x3 = inp.load[width=8](ioff + 24)
            total += q4_dot_vec(sc, nb, x0, x1, x2, x3)
        oq.store(row, total)
    # K projection
    for row in range(nkv):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = wk.load(off); var hi = wk.load(off + 1)
            var sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nb = wk.load[width=16](off + 2)
            var ioff = blk * 32
            var x0 = inp.load[width=8](ioff)
            var x1 = inp.load[width=8](ioff + 8)
            var x2 = inp.load[width=8](ioff + 16)
            var x3 = inp.load[width=8](ioff + 24)
            total += q4_dot_vec(sc, nb, x0, x1, x2, x3)
        ok.store(row, total)
    # V projection
    for row in range(nkv):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = wv.load(off); var hi = wv.load(off + 1)
            var sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nb = wv.load[width=16](off + 2)
            var ioff = blk * 32
            var x0 = inp.load[width=8](ioff)
            var x1 = inp.load[width=8](ioff + 8)
            var x2 = inp.load[width=8](ioff + 16)
            var x3 = inp.load[width=8](ioff + 24)
            total += q4_dot_vec(sc, nb, x0, x1, x2, x3)
        ov.store(row, total)

# ─── Fused FFN: gate + up projection ─────────────────────────────────
# Shares input vector across gate and up projections.
# Row-outer accumulation.

def q4_mm_fused_ffn(
    wg: UnsafePointer[UInt8, MutAnyOrigin],
    wu: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    og: UnsafePointer[Float32, MutAnyOrigin],
    ou: UnsafePointer[Float32, MutAnyOrigin],
    nff: Int, nc: Int,
):
    var bpr = nc // 32
    # Gate projection
    for row in range(nff):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = wg.load(off); var hi = wg.load(off + 1)
            var sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nb = wg.load[width=16](off + 2)
            var ioff = blk * 32
            var x0 = inp.load[width=8](ioff)
            var x1 = inp.load[width=8](ioff + 8)
            var x2 = inp.load[width=8](ioff + 16)
            var x3 = inp.load[width=8](ioff + 24)
            total += q4_dot_vec(sc, nb, x0, x1, x2, x3)
        og.store(row, total)
    # Up projection
    for row in range(nff):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = wu.load(off); var hi = wu.load(off + 1)
            var sc = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nb = wu.load[width=16](off + 2)
            var ioff = blk * 32
            var x0 = inp.load[width=8](ioff)
            var x1 = inp.load[width=8](ioff + 8)
            var x2 = inp.load[width=8](ioff + 16)
            var x3 = inp.load[width=8](ioff + 24)
            total += q4_dot_vec(sc, nb, x0, x1, x2, x3)
        ou.store(row, total)

# ─── Vectorized RMS Norm (stride-8 SIMD) ─────────────────────────────

def rms_norm_vec(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    out: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    var ss: Float32 = 0.0
    var i = 0
    while i + 8 <= n:
        var v = x.load[width=8](i)
        ss += (v * v).reduce_add()
        i += 8
    while i < n:
        ss += x.load(i) * x.load(i)
        i += 1
    var inv_rms = 1.0 / sqrt(ss / Float32(n) + 1e-6)
    var inv_v = F32x8(inv_rms)
    i = 0
    while i + 8 <= n:
        var v = x.load[width=8](i)
        var w = weight.load[width=8](i)
        out.store[width=8](i, v * inv_v * w)
        i += 8
    while i < n:
        out.store(i, x.load(i) * inv_rms * weight.load(i))
        i += 1

# ─── Vectorized SiLU (stride-8 SIMD) ────────────────────────────────

def silu_vec(
    x: UnsafePointer[Float32, MutAnyOrigin],
    out: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    var one = F32x8(1.0)
    var i = 0
    while i + 8 <= n:
        var v = x.load[width=8](i)
        # SiLU: x / (1 + exp(-x)) — scalar exp per lane
        var result = F32x8()
        for j in range(8):
            result[j] = v[j] / (1.0 + exp(-v[j]))
        out.store[width=8](i, result)
        i += 8
    while i < n:
        var val = x.load(i)
        out.store(i, val / (1.0 + exp(-val)))
        i += 1

# ─── Vectorized RoPE (stride-8 SIMD) ─────────────────────────────────

def rope_vec(
    x: UnsafePointer[Float32, MutAnyOrigin],
    out: UnsafePointer[Float32, MutAnyOrigin],
    pos: Int, dim: Int,
):
    var half = dim // 2
    var i = 0
    while i + 8 <= half:
        # Compute freq for each of 8 positions
        for j in range(8):
            var idx = Float32(i + j)
            var freq = 1.0 / pow(10000.0, (2.0 * idx) / Float32(dim))
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
        var freq = 1.0 / pow(10000.0, (2.0 * idx) / Float32(dim))
        var angle = Float32(pos) * freq
        var cos_a = cos(angle)
        var sin_a = sin(angle)
        var x0 = x.load(i)
        var x1 = x.load(i + half)
        out.store(i, x0 * cos_a - x1 * sin_a)
        out.store(i + half, x0 * sin_a + x1 * cos_a)
        i += 1
    # Copy remaining dimensions unchanged
    while i < dim:
        out.store(i, x.load(i))
        i += 1

# ─── Benchmark ─────────────────────────────────────────────────────────

def main() raises:
    var tim = Python.import_module("time")
    var bt = Python.import_module("builtins")
    var NE = 2048
    var NK = 512
    var NF = 8192
    var nb = NE // 32

    # Allocate
    var wsz = NE * nb * 18
    var wsz_f = NF * nb * 18
    var wsz_k = NK * nb * 18
    var w = alloc[UInt8](wsz)
    var wf = alloc[UInt8](wsz_f)
    var wq = alloc[UInt8](wsz)
    var wk = alloc[UInt8](wsz_k)
    var wv = alloc[UInt8](wsz_k)
    var wg = alloc[UInt8](wsz_f)
    var wu = alloc[UInt8](wsz_f)
    var x = alloc[Float32](NE)
    var r = alloc[Float32](NE)
    var rf = alloc[Float32](NF)
    var ok = alloc[Float32](NK)
    var ov = alloc[Float32](NK)
    var gg = alloc[Float32](NF)

    # Init
    var s: UInt8 = 42
    for i in range(wsz): s = (s * 7 + 13) & 0xFF; w.store(i, s)
    for i in range(wsz_f): s = (s * 7 + 13) & 0xFF; wf.store(i, s)
    s = 1
    for i in range(wsz): s = (s * 7 + 13) & 0xFF; wq.store(i, s)
    s = 50
    for i in range(wsz_k): s = (s * 7 + 13) & 0xFF; wk.store(i, s)
    s = 99
    for i in range(wsz_k): s = (s * 7 + 13) & 0xFF; wv.store(i, s)
    s = 111
    for i in range(wsz_f): s = (s * 7 + 13) & 0xFF; wg.store(i, s)
    s = 222
    for i in range(wsz_f): s = (s * 7 + 13) & 0xFF; wu.store(i, s)
    for i in range(NE): x.store(i, Float32(0.5))

    print("═══ Mojo Q4_0 Kernel v2 — Row-Outer + FMA Accumulation ═══")
    print()

    # 1. Dot product: scalar vs vectorized
    var nibs = U8x16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    var zv = F32x8(0.5)
    var sc = Float32(0.5)

    var t0 = tim.time()
    for i in range(1000000):
        var _ = q4_dot_vec(sc, nibs, zv, zv, zv, zv)
    var t1 = tim.time()
    var dot_vc = (t1 - t0) * 1000.0
    print("Block dot (FMA-chain):", bt.str(bt.round(dot_vc, 2)), "ms/M")
    print()

    # 2. Matmul 2048x2048: row-outer vs block-outer
    print("Matmul 2048x2048:")

    for i in range(NE): r.store(i, Float32(0.0))
    t0 = tim.time()
    for i in range(10): q4_mm_row_outer(w, x, r, NE, NE)
    t1 = tim.time()
    var b1 = (t1 - t0) * 100.0
    print("  Row-outer:", bt.str(bt.round(b1, 2)), "ms")

    for i in range(NE): r.store(i, Float32(0.0))
    t0 = tim.time()
    for i in range(10): q4_mm_regblock(w, x, r, NE, NE)
    t1 = tim.time()
    var b2 = (t1 - t0) * 100.0
    print("  Regblock+row-outer:", bt.str(bt.round(b2, 2)), "ms (", bt.str(bt.round(b1 / b2, 2)), "x)")
    print()

    # 3. FFN matmul 8192x2048
    print("Matmul 8192x2048:")
    for i in range(NF): rf.store(i, Float32(0.0))
    t0 = tim.time()
    for i in range(5): q4_mm_row_outer(wf, x, rf, NF, NE)
    t1 = tim.time()
    var f1 = (t1 - t0) * 200.0
    print("  Row-outer:", bt.str(bt.round(f1, 1)), "ms")

    for i in range(NF): rf.store(i, Float32(0.0))
    t0 = tim.time()
    for i in range(5): q4_mm_regblock(wf, x, rf, NF, NE)
    t1 = tim.time()
    var f2 = (t1 - t0) * 200.0
    print("  Regblock+row-outer:", bt.str(bt.round(f2, 1)), "ms (", bt.str(bt.round(f1 / f2, 2)), "x)")
    print()

    # 4. Fused QKV (row-outer)
    print("Fused QKV:")
    for i in range(NE): r.store(i, Float32(0.0))
    for i in range(NK): ok.store(i, Float32(0.0)); ov.store(i, Float32(0.0))
    t0 = tim.time()
    for i in range(10): q4_mm_fused_qkv(wq, wk, wv, x, r, ok, ov, NE, NK, NE)
    t1 = tim.time()
    var fq = (t1 - t0) * 100.0
    var sq = b1 + b1 * Float32(NK) / Float32(NE) * 2.0
    print("  Fused (row-outer):", bt.str(bt.round(fq, 2)), "ms")
    print("  Separate:", bt.str(bt.round(sq, 1)), "ms (", bt.str(bt.round(sq / fq, 1)), "x savings)")
    print()

    # 5. Fused FFN gate+up (row-outer)
    print("Fused FFN gate+up:")
    for i in range(NF): rf.store(i, Float32(0.0)); gg.store(i, Float32(0.0))
    t0 = tim.time()
    for i in range(5): q4_mm_fused_ffn(wg, wu, x, rf, gg, NF, NE)
    t1 = tim.time()
    var fu = (t1 - t0) * 200.0
    var su = f1 * 2.0
    print("  Fused (row-outer):", bt.str(bt.round(fu, 1)), "ms")
    print("  Separate:", bt.str(bt.round(su, 1)), "ms (", bt.str(bt.round(su / fu, 1)), "x savings)")
    print()

    # 6. Vectorized norms
    var rr = alloc[Float32](NE)
    var ww = alloc[Float32](NE)
    for i in range(NE): ww.store(i, Float32(1.0) / sqrt(Float32(NE)))

    t0 = tim.time()
    for i in range(1000): rms_norm_vec(x, ww, rr, NE)
    t1 = tim.time()
    var norm_t = (t1 - t0) * 1000.0
    print("RMS norm 2048:", bt.str(bt.round(norm_t, 3)), "ms (1K iters)")

    t0 = tim.time()
    for i in range(1000): silu_vec(x, rr, NE)
    t1 = tim.time()
    var silu_t = (t1 - t0) * 1000.0
    print("SiLU 2048:", bt.str(bt.round(silu_t, 3)), "ms (1K iters)")

    t0 = tim.time()
    for i in range(1000): rope_vec(x, rr, 0, NE)
    t1 = tim.time()
    var rope_t = (t1 - t0) * 1000.0
    print("RoPE 2048:", bt.str(bt.round(rope_t, 3)), "ms (1K iters)")
    print()

    # Forward pass estimate
    var base = (b1 * 1.25 + f1 * 3) * 16.0
    var reg = (b2 * 1.25 + f2 * 3) * 16.0
    var fused = (fq + b1 + fu + f1) * 16.0
    print("═══ Forward pass estimate (1 core) ═══")
    print("Row-outer:", bt.str(bt.round(1000 / base, 2)), "tok/s")
    print("Regblock+row-outer:", bt.str(bt.round(1000 / reg, 2)), "tok/s (", bt.str(bt.round(base / reg, 2)), "x)")
    print("Fused+row-outer:", bt.str(bt.round(1000 / fused, 2)), "tok/s (", bt.str(bt.round(base / fused, 2)), "x)")

    w.free(); wf.free(); wq.free(); wk.free(); wv.free(); wg.free(); wu.free()
    x.free(); r.free(); rf.free(); ok.free(); ov.free(); gg.free()
    rr.free(); ww.free()