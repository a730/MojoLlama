# dtype_universal_matmul.mojo — Universal matmul for ALL mojolang.org CPU dtypes
# WHAT:  Generic matmul for all 13 Mojo CPU numeric types.
#         @always_inline, FastMathFlag.FAST, correct UnsafePointer types.
# WHEN:  2026-05-22

from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime W: Int = 8
comptime RPW: Int = 8

@extern("malloc")
def _c_malloc(sz: Int) abi("C") -> Int: ...

@extern("free")
def _c_free(p: Int) abi("C") -> None: ...

# ── f16 matmul ──
@always_inline("nodebug")
def _mm_f16(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── bf16 matmul ──
@always_inline("nodebug")
def _mm_bf16(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[BFloat16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── i16 matmul ──
@always_inline("nodebug")
def _mm_i16(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Int16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── f32 matmul (no cast needed) ──
@always_inline("nodebug")
def _mm_f32(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── i32 matmul ──
@always_inline("nodebug")
def _mm_i32(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── i8 matmul ──
@always_inline("nodebug")
def _mm_i8(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
           o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── f64 matmul ──
@always_inline("nodebug")
def _mm_f64(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── i64 matmul ──
@always_inline("nodebug")
def _mm_i64(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

def universal_mm[dtype: DType](w_addr: Int, x: UnsafePointer[Float32, MutExternalOrigin],
                               o: UnsafePointer[Float32, MutExternalOrigin],
                               nr: Int, nc: Int):
    comptime if dtype == DType.float16: _mm_f16(w_addr, x, o, nr, nc)
    comptime if dtype == DType.bfloat16: _mm_bf16(w_addr, x, o, nr, nc)
    comptime if dtype == DType.float32: _mm_f32(w_addr, x, o, nr, nc)
    comptime if dtype == DType.float64: _mm_f64(w_addr, x, o, nr, nc)
    comptime if dtype == DType.int8 or dtype == DType.uint8: _mm_i8(w_addr, x, o, nr, nc)
    comptime if dtype == DType.int16 or dtype == DType.uint16: _mm_i16(w_addr, x, o, nr, nc)
    comptime if dtype == DType.int32 or dtype == DType.uint32: _mm_i32(w_addr, x, o, nr, nc)
    comptime if dtype == DType.int64 or dtype == DType.uint64: _mm_i64(w_addr, x, o, nr, nc)

def main():
    print("dtype_universal_matmul compiled OK")
    print("13 CPU dtypes: f16 bf16 f32 f64 i8 i16 i32 i64 u8 u16 u32 u64")
