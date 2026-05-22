# correctness_mini.mojo — Correctness test: old mm16 vs new universal_mm
# Fixed: correct UnsafePointer types per dtype (Float16 for f16, not Int16)

from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime RPW: Int = 32; comptime W: Int = 8

@extern("malloc")
def _alc(sz: Int) abi("C") -> Int: ...

@extern("free")
def _c_free(p: Int) abi("C") -> None: ...

# OLD mm16 (reference)
def mm16_old(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    if wa == 0: return
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var bpr = nc // 32; var nb = (nr + RPW - 1) // RPW
    def wk(b: Int) capturing:
        var rs = b * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc; var acc = SIMD[DType.float32, W](0.0)
            for blk in range(bpr):
                comptime for grp in range(4):
                    var w16 = w.load[width=W](ro + blk * 32 + grp * 8)
                    var wf32 = w16.cast[DType.float32]()
                    var xv = x.load[width=W](blk * 32 + grp * 8)
                    acc = wf32.fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=32)

# NEW f16 matmul (correct pointer type: Float16)
@always_inline("nodebug")
def mm16_new_f16(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
                 o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + 7) // 8
    def wk(b: Int) capturing:
        var rs = b * 8; var re = rs + 8
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc; var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=32)

# NEW int16 matmul (correct pointer type: Int16)
@always_inline("nodebug")
def mm16_new_int16(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
                   o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Int16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + 7) // 8
    def wk(b: Int) capturing:
        var rs = b * 8; var re = rs + 8
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc; var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=32)

def main():
    print("Correctness: old mm16 vs new universal_mm")
    print("=" * 50)
    
    var nr = 640; var nc = 640
    
    var w_buf = _alc(nr * nc * 2)
    var x_buf = _alc(nr * 4)
    var o_old = _alc(nr * 4)
    var o_new = _alc(nr * 4)
    
    var w_float = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=w_buf)
    var w_int = UnsafePointer[Int16, MutExternalOrigin](unsafe_from_address=w_buf)
    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=x_buf)
    var oo = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=o_old)
    var on = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=o_new)
    
    # Fill with f16 values (range -5.0 to 4.95)
    for i in range(nr * nc):
        w_float.store(i, Float16(Float32(i % 200 - 100) * 0.05))
    for i in range(nc):
        x_ptr.store(i, Float32(i % 50 - 25) * 0.1)
    
    # Test 1: f16 weights (should match exactly)
    print("Test 1: f16 weights (old f16 vs new f16 matmul)")
    mm16_old(w_buf, x_ptr, oo, nr, nc)
    mm16_new_f16(w_buf, x_ptr, on, nr, nc)
    
    var max_d: Float32 = 0.0; var avg_d: Float64 = 0.0; var bad = 0
    for i in range(nr):
        var d = oo.load(i) - on.load(i)
        if d < 0: d = -d
        if d > max_d: max_d = d
        avg_d += Float64(d)
        if d > 0.001: bad += 1
    avg_d /= Float64(nr)
    print("  Max diff:", max_d, "Avg diff:", avg_d, "Bad:", bad)
    print("  ", "✓ PASS" if max_d < 0.01 else "✗ FAIL")
    
    # Test 2: same data interpreted as int16 (should be different from f16 matmul)
    print()
    print("Test 2: Int16 weights (old f16 vs new int16 matmul)")
    for i in range(nr * nc):
        w_int.store(i, Int16(i % 200 - 100))
    mm16_new_f16(w_buf, x_ptr, oo, nr, nc)  # old still uses f16 interpretation
    mm16_new_int16(w_buf, x_ptr, on, nr, nc)  # new uses int16 interpretation
    
    max_d = 0.0; avg_d = 0.0; bad = 0
    for i in range(nr):
        var d = oo.load(i) - on.load(i)
        if d < 0: d = -d
        if d > max_d: max_d = d
        avg_d += Float64(d)
        if d > 0.001: bad += 1
    avg_d /= Float64(nr)
    print("  Max diff:", max_d, "Avg diff:", avg_d, "Bad:", bad)
    print("  ", "✓ Int16 and f16 produce DIFFERENT results (expected)" if max_d > 0.1 else "✗ Unexpected")
    
    _c_free(w_buf); _c_free(x_buf); _c_free(o_old); _c_free(o_new)
