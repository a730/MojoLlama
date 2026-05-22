# MojoLlama SIMD Kernels — pure Mojo + parallelize threading
# WHY:  Port C AVX2+OMP engine to pure Mojo. parallelize gives CPU threading.
# WHEN: May 2026 — added parallelize after stdlib threading discovery.
from std import time
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize

comptime B5: Int = 22; comptime B8: Int = 34; comptime B4: Int = 17
comptime W: Int = 8

def f16_to_f32(h: UInt16) -> Float32:
    var s = UInt32(h >> 15); var e = UInt32((h >> 10) & 0x1F)
    var m = UInt32(h & 0x3FF)
    if e == 0:
        if m == 0: return 0.0
        return Float32(Float64(m) * 5.960464477539063e-8)
    if e == 31: return 0.0
    var r = Float32(m | 0x400)
    var ei = Int(e) - 25
    if ei >= 0:
        for _ in range(ei): r *= 2.0
    else:
        for _ in range(-ei): r *= 0.5
    return -r if s != 0 else r

def main():
    print("MojoLlama MT SIMD Kernels — Benchmark")
    print("======================================")
    var nc = 2880; var nr = 256; var bpr = nc // 32

    var w_b5 = alloc[UInt8](nr * bpr * B5)
    for i in range(nr * bpr * B5): w_b5.store(i, UInt8((i * 7 + 13) & 0xFF))
    var w_b8 = alloc[UInt8](nr * bpr * B8)
    for i in range(nr * bpr * B8): w_b8.store(i, UInt8(i & 0xFF))
    var w_b4 = alloc[UInt8](nr * bpr * B4)
    for i in range(nr * bpr * B4): w_b4.store(i, UInt8(i & 0xFF))
    var w_f32 = alloc[Float32](nr * nc)
    for i in range(nr * nc):
        w_f32.store(i, Float32(Float64((i * 7 + 13) % 100 - 50) / 50.0))
    var x = alloc[Float32](nc)
    for i in range(nc): x.store(i, Float32(Float64((i * 3) % 100 - 50) / 50.0))
    var o_st = alloc[Float32](nr)
    var o_mt = alloc[Float32](nr)

    # ── MT closures ──
    def f32_mt_row(r: Int) capturing -> None:
        var acc = SIMD[DType.float32, W](0.0)
        var ro = r * nc
        for bc in range(0, nc, W):
            var vals = SIMD[DType.float32, W](0.0)
            for k in range(W): vals[k] = w_f32.load(ro + bc + k)
            var xv = SIMD[DType.float32, W](0.0)
            for k in range(W): xv[k] = x.load(bc + k)
            acc = acc + vals * xv
        o_mt.store(r, acc.reduce_add())

    def q5_mt_row(r: Int) capturing -> None:
        var acc = SIMD[DType.float32, W](0.0)
        var ro = r * bpr * B5
        for blk in range(bpr):
            var bo = ro + blk * B5
            var lo = UInt16(w_b5.load(bo))
            var hi = UInt16(w_b5.load(bo + 1))
            var d = f16_to_f32(lo | (hi << 8))
            for ch in range(4):
                var vals = SIMD[DType.float32, W](0.0)
                for k in range(W):
                    var idx = ch * W + k
                    var p = w_b5.load(bo + 6 + idx // 2)
                    var qh = w_b5.load(bo + 2 + idx // 4)
                    var hs = 2 * (idx % 4)
                    var hb_u8 = (UInt8(qh) >> UInt8(hs)) & UInt8(1)
                    var hb = Int32(hb_u8)
                    var nib = Int32(p >> 4) if idx % 2 == 1 else Int32(p & 0x0F)
                    if nib > 7: nib -= 16
                    var v = nib + hb * 16
                    if v > 15: v -= 32
                    vals[k] = Float32(v)
                var xb = blk * 32 + ch * W
                var xv = SIMD[DType.float32, W](0.0)
                for k in range(W): xv[k] = x.load(xb + k)
                acc = acc + vals * xv * d
        o_mt.store(r, acc.reduce_add())

    def q8_mt_row(r: Int) capturing -> None:
        var acc = SIMD[DType.float32, W](0.0)
        var ro = r * bpr * B8
        for blk in range(bpr):
            var bo = ro + blk * B8
            var lo = UInt16(w_b8.load(bo))
            var hi = UInt16(w_b8.load(bo + 1))
            var d = f16_to_f32(lo | (hi << 8))
            for ch in range(4):
                var vals = SIMD[DType.float32, W](0.0)
                for k in range(W):
                    var idx = ch * W + k
                    var qb = w_b8.load(bo + 2 + idx)
                    var q = Int32(qb)
                    if q > 127: q -= 256
                    vals[k] = Float32(q)
                var xb = blk * 32 + ch * W
                var xv = SIMD[DType.float32, W](0.0)
                for k in range(W): xv[k] = x.load(xb + k)
                acc = acc + vals * xv * d
        o_mt.store(r, acc.reduce_add())

    def mxfp4_mt_row(r: Int) capturing -> None:
        var acc: Float32 = 0.0
        var ro = r * bpr * B4
        for blk in range(bpr):
            var bo = ro + blk * B4
            var eb = w_b4.load(bo + 16)
            var sf: Float32 = 0.0
            if eb != 0:
                if eb < 255:
                    var ei = Int(eb) - 127
                    sf = 1.0
                    if ei >= 0:
                        for _ in range(ei): sf *= 2.0
                    else:
                        for _ in range(-ei): sf *= 0.5
                    if sf > 1e20: sf = 1e20
                else: sf = 1e20
            var ai: Int32 = 0
            for j in range(16):
                var p = w_b4.load(bo + j)
                var lo = Int32(p & 0x0F)
                if lo > 7: lo -= 16
                var hi = Int32(p >> 4)
                if hi > 7: hi -= 16
                ai += lo * Int32(x.load(blk * 32 + j * 2)) + \
                      hi * Int32(x.load(blk * 32 + j * 2 + 1))
            acc += Float32(ai) * sf
        o_mt.store(r, acc)

    # Warmup
    parallelize[func=f32_mt_row](num_work_items=4)
    parallelize[func=q5_mt_row](num_work_items=4)
    parallelize[func=q8_mt_row](num_work_items=4)
    parallelize[func=mxfp4_mt_row](num_work_items=4)

    var bn = 200; var t0: Float64; var t1: Float64; var ms: Float64; var rows_s: Float64
    var ok: Int; var match_str: String

    # ════ F32 ════
    t0 = time.perf_counter()
    for _ in range(bn):
        for r in range(nr):
            var acc = SIMD[DType.float32, W](0.0)
            var ro = r * nc
            for bc in range(0, nc, W):
                var vals = SIMD[DType.float32, W](0.0)
                for k in range(W): vals[k] = w_f32.load(ro + bc + k)
                var xv = SIMD[DType.float32, W](0.0)
                for k in range(W): xv[k] = x.load(bc + k)
                acc = acc + vals * xv
            o_st.store(r, acc.reduce_add())
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    rows_s = Float64(nr) / (ms / 1000.0)
    print("F32 ST:     ", Int(ms * 1000), "us => ", Int(rows_s), "rows/s")

    t0 = time.perf_counter()
    for _ in range(bn): parallelize[func=f32_mt_row](num_work_items=nr)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    rows_s = Float64(nr) / (ms / 1000.0)
    ok = 1
    for i in range(nr):
        if o_st.load(i) != o_mt.load(i):
            ok = 0
            break
    match_str = "MISMATCH"
    if ok: match_str = "MATCH"
    print("F32 MT:     ", Int(ms * 1000), "us => ", Int(rows_s), "rows/s  (", match_str, ")")

    # ════ Q5_0 ════
    bn = 200
    t0 = time.perf_counter()
    for _ in range(bn):
        for r in range(nr):
            var acc = SIMD[DType.float32, W](0.0)
            var ro = r * bpr * B5
            for blk in range(bpr):
                var bo = ro + blk * B5
                var lo = UInt16(w_b5.load(bo))
                var hi = UInt16(w_b5.load(bo + 1))
                var d = f16_to_f32(lo | (hi << 8))
                for ch in range(4):
                    var vals = SIMD[DType.float32, W](0.0)
                    for k in range(W):
                        var idx = ch * W + k
                        var p = w_b5.load(bo + 6 + idx // 2)
                        var qh = w_b5.load(bo + 2 + idx // 4)
                        var hs = 2 * (idx % 4)
                        var hb_u8 = (UInt8(qh) >> UInt8(hs)) & UInt8(1)
                        var hb = Int32(hb_u8)
                        var nib = Int32(p >> 4) if idx % 2 == 1 else Int32(p & 0x0F)
                        if nib > 7: nib -= 16
                        var v = nib + hb * 16
                        if v > 15: v -= 32
                        vals[k] = Float32(v)
                    var xb = blk * 32 + ch * W
                    var xv = SIMD[DType.float32, W](0.0)
                    for k in range(W): xv[k] = x.load(xb + k)
                    acc = acc + vals * xv * d
            o_st.store(r, acc.reduce_add())
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    rows_s = Float64(nr) / (ms / 1000.0)
    print("Q5_0 ST:    ", Int(ms * 1000), "us => ", Int(rows_s), "rows/s")

    t0 = time.perf_counter()
    for _ in range(bn): parallelize[func=q5_mt_row](num_work_items=nr)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    rows_s = Float64(nr) / (ms / 1000.0)
    ok = 1
    for i in range(nr):
        if o_st.load(i) != o_mt.load(i):
            ok = 0
            break
    match_str = "MISMATCH"
    if ok: match_str = "MATCH"
    print("Q5_0 MT:    ", Int(ms * 1000), "us => ", Int(rows_s), "rows/s  (", match_str, ")")

    # ════ Q8_0 ════
    t0 = time.perf_counter()
    for _ in range(bn):
        for r in range(nr):
            var acc = SIMD[DType.float32, W](0.0)
            var ro = r * bpr * B8
            for blk in range(bpr):
                var bo = ro + blk * B8
                var lo = UInt16(w_b8.load(bo))
                var hi = UInt16(w_b8.load(bo + 1))
                var d = f16_to_f32(lo | (hi << 8))
                for ch in range(4):
                    var vals = SIMD[DType.float32, W](0.0)
                    for k in range(W):
                        var idx = ch * W + k
                        var qb = w_b8.load(bo + 2 + idx)
                        var q = Int32(qb)
                        if q > 127: q -= 256
                        vals[k] = Float32(q)
                    var xb = blk * 32 + ch * W
                    var xv = SIMD[DType.float32, W](0.0)
                    for k in range(W): xv[k] = x.load(xb + k)
                    acc = acc + vals * xv * d
            o_st.store(r, acc.reduce_add())
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    rows_s = Float64(nr) / (ms / 1000.0)
    print("Q8_0 ST:    ", Int(ms * 1000), "us => ", Int(rows_s), "rows/s")

    t0 = time.perf_counter()
    for _ in range(bn): parallelize[func=q8_mt_row](num_work_items=nr)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    rows_s = Float64(nr) / (ms / 1000.0)
    ok = 1
    for i in range(nr):
        if o_st.load(i) != o_mt.load(i):
            ok = 0
            break
    match_str = "MISMATCH"
    if ok: match_str = "MATCH"
    print("Q8_0 MT:    ", Int(ms * 1000), "us => ", Int(rows_s), "rows/s  (", match_str, ")")

    # ════ MXFP4 ════
    bn = 50
    t0 = time.perf_counter()
    for _ in range(bn):
        for r in range(nr):
            var acc: Float32 = 0.0
            var ro = r * bpr * B4
            for blk in range(bpr):
                var bo = ro + blk * B4
                var eb = w_b4.load(bo + 16)
                var sf: Float32 = 0.0
                if eb != 0:
                    if eb < 255:
                        var ei = Int(eb) - 127
                        sf = 1.0
                        if ei >= 0:
                            for _ in range(ei): sf *= 2.0
                        else:
                            for _ in range(-ei): sf *= 0.5
                        if sf > 1e20: sf = 1e20
                    else: sf = 1e20
                var ai: Int32 = 0
                for j in range(16):
                    var p = w_b4.load(bo + j)
                    var lo = Int32(p & 0x0F)
                    if lo > 7: lo -= 16
                    var hi = Int32(p >> 4)
                    if hi > 7: hi -= 16
                    ai += lo * Int32(x.load(blk * 32 + j * 2)) + \
                          hi * Int32(x.load(blk * 32 + j * 2 + 1))
                acc += Float32(ai) * sf
            o_st.store(r, acc)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    rows_s = Float64(nr) / (ms / 1000.0)
    print("MXFP4 ST:   ", Int(ms * 1000), "us => ", Int(rows_s), "rows/s")

    t0 = time.perf_counter()
    for _ in range(bn): parallelize[func=mxfp4_mt_row](num_work_items=nr)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    rows_s = Float64(nr) / (ms / 1000.0)
    ok = 1
    for i in range(nr):
        if o_st.load(i) != o_mt.load(i):
            ok = 0
            break
    match_str = "MISMATCH"
    if ok: match_str = "MATCH"
    print("MXFP4 MT:   ", Int(ms * 1000), "us => ", Int(rows_s), "rows/s  (", match_str, ")")

    # Element-wise benchmarks (sequential)
    var w_nrm = alloc[Float32](nc)
    for i in range(nc): w_nrm.store(i, 1.0)
    var oe = alloc[Float32](nc)

    t0 = time.perf_counter()
    for _ in range(1000):
        var s: Float32 = 0.0
        for i in range(nc): s += x.load(i) * x.load(i)
        var r = sqrt(Float64(s / Float32(nc) + 1e-6))
        var inv = Float32(1.0 / r)
        for i in range(nc): oe.store(i, w_nrm.load(i) * x.load(i) * inv)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / 1000.0
    print("RmsNorm:   ", Int(ms * 1000), "us for", nc, "elems")

    t0 = time.perf_counter()
    for _ in range(1000):
        for i in range(nc):
            var xi = x.load(i)
            var sig = Float32(1.0 / (1.0 + exp(Float64(-xi))))
            oe.store(i, xi * sig)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / 1000.0
    print("SiLU:      ", Int(ms * 1000), "us for", nc, "elems")

    t0 = time.perf_counter()
    for _ in range(1000):
        var mx = x.load(0)
        for i in range(1, nc):
            var xv = x.load(i)
            if xv > mx: mx = xv
        var total: Float32 = 0.0
        for i in range(nc): total += Float32(exp(Float64(x.load(i) - mx)))
        for i in range(nc):
            oe.store(i, Float32(exp(Float64(x.load(i) - mx))) / total)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / 1000.0
    print("Softmax:   ", Int(ms * 1000), "us for", nc, "elems")

    # Cleanup
    w_b5.free(); w_b8.free(); w_b4.free(); w_f32.free()
    x.free(); o_st.free(); o_mt.free(); w_nrm.free(); oe.free()
    print("\nDone!")
