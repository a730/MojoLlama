# bench_optimized.mojo — Benchmark: old mm16 vs optimized universal_mm
# WHAT:  Compares original f16 matmul against @always_inline + FastMathFlag.FAST +
#        cache-aligned + thread-pinned universal_mm on TinyLlama dims.
# WHEN:  2026-05-22

from std import time
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime RPW: Int = 32
comptime W: Int = 8
comptime NE: Int = 2048; comptime NH: Int = 32; comptime NK: Int = 4
comptime HD: Int = 64;   comptime NL: Int = 22; comptime FF: Int = 5632
comptime inner: Int = NH * HD; comptime kv_dim: Int = NK * HD

# ── C helpers ──
@extern("malloc")
def _alc(sz: Int) abi("C") -> Int: ...
@extern("free")
def _c_free(p: Int) abi("C") -> None: ...
@extern("sched_setaffinity")
def _sched_setaff(pid: Int, cpusz: Int, mask: Int) abi("C") -> Int: ...

# ── Thread pinning ──
def ml_pin():
    var mask_sz = 128; var raw = _alc(mask_sz)
    var mask = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=raw)
    for i in range(mask_sz): mask.store(i, UInt8(0))
    for i in range(32): mask.store(i // 8, mask.load(i // 8) | UInt8(1 << (i % 8)))
    var _ = _sched_setaff(0, mask_sz, raw); _c_free(raw)

# ── OLD mm16 ──
def mm16_old(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int):
    if wa == 0: return
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=wa)
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
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── NEW optimized universal_mm ──
@always_inline("nodebug")
def _dtype_mm_sz2[dtype: DType](
    w_addr: Int, x: UnsafePointer[Float32, MutExternalOrigin],
    o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Int16, MutExternalOrigin](unsafe_from_address=w_addr)
    var nb = (nr + 7) // 8; var nw = 32
    def worker(b: Int) capturing:
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
    parallelize[func=worker](num_work_items=nb, num_workers=nw)

@always_inline("nodebug")
def universal_mm_new[dtype: DType](
    w_addr: Int, x: UnsafePointer[Float32, MutExternalOrigin],
    o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    comptime if dtype == DType.float16:
        _dtype_mm_sz2[dtype](w_addr, x, o, nr, nc)

# ── Benchmark ──
def run_bench():
    var pool_sz = 512 * 1024 * 1024
    var pool_old = _alc(pool_sz)
    var pool_new = _alc(pool_sz)
    
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=_alc(4096*4))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=_alc(4096*4))
    var qp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=_alc(4096*4))
    var kp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=_alc(4096*4))
    var vp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=_alc(4096*4))
    var gp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=_alc(4096*4))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=_alc(4096*4))
    var dp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=_alc(4096*4))
    
    for i in range(2048): hp.store(i, Float32(i % 100 - 50) * 0.01)
    
    var pool_sz_m1 = pool_sz - 10485760
    var off: Int = 0
    
    print(); print("TinyLlama-1.1B: old mm16 vs optimized universal_mm (f16)")
    print("=" * 60)
    print(NL, "layers, NE=2048 NH=32 NK=4 HD=64 FF=5632")
    print()
    
    # Old
    print("Old mm16 (baseline)...")
    var t0_old = time.perf_counter()
    off = 0
    for _ in range(NL):
        mm16_old(pool_old + (off % pool_sz_m1), qp, bp, inner, NE, 24); off += 1
        mm16_old(pool_old + (off % pool_sz_m1), kp, bp, kv_dim, NE, 24); off += 1
        mm16_old(pool_old + (off % pool_sz_m1), vp, bp, kv_dim, NE, 24); off += 1
        mm16_old(pool_old + (off % pool_sz_m1), bp, qp, NE, inner, 24); off += 1
        mm16_old(pool_old + (off % pool_sz_m1), gp, bp, FF, NE, 24); off += 1
        mm16_old(pool_old + (off % pool_sz_m1), up, bp, FF, NE, 24); off += 1
        mm16_old(pool_old + (off % pool_sz_m1), dp, gp, NE, FF, 24); off += 1
    var ms_old = (time.perf_counter() - t0_old) * 1000.0
    
    # New
    print("New universal_mm (optimized)...")
    var t0_new = time.perf_counter()
    off = 0
    for _ in range(NL):
        universal_mm_new[DType.float16](pool_new + (off % pool_sz_m1), qp, bp, inner, NE); off += 1
        universal_mm_new[DType.float16](pool_new + (off % pool_sz_m1), kp, bp, kv_dim, NE); off += 1
        universal_mm_new[DType.float16](pool_new + (off % pool_sz_m1), vp, bp, kv_dim, NE); off += 1
        universal_mm_new[DType.float16](pool_new + (off % pool_sz_m1), bp, qp, NE, inner); off += 1
        universal_mm_new[DType.float16](pool_new + (off % pool_sz_m1), gp, bp, FF, NE); off += 1
        universal_mm_new[DType.float16](pool_new + (off % pool_sz_m1), up, bp, FF, NE); off += 1
        universal_mm_new[DType.float16](pool_new + (off % pool_sz_m1), dp, gp, NE, FF); off += 1
    var ms_new = (time.perf_counter() - t0_new) * 1000.0
    
    var tok_old = 1000.0 / ms_old
    var tok_new = 1000.0 / ms_new
    var imp = (tok_new / tok_old - 1.0) * 100.0
    
    print(); print("Results:")
    print("-" * 60)
    print(String("  Old mm16:        ") + String(ms_old, 7, 3) + String(" ms/fwd  ") + String(tok_old, 7, 3) + String(" tok/s"))
    print(String("  New universal_mm: ") + String(ms_new, 7, 3) + String(" ms/fwd  ") + String(tok_new, 7, 3) + String(" tok/s"))
    print(String("  Improvement:      ") + String(imp, 6, 2) + String("%"))

def main():
    ml_pin()
    print("Thread pinning: cores 0-31")
    run_bench()
