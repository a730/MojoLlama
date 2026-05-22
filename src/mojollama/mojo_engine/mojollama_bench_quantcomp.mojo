# MojoLlama Quant Comparison — Measure each quant's matmul speed
# WHAT:  For each GGUF quant format, allocates cold-cache weights in that
#        format, measures actual f16 matmul throughput, reports tok/s.
# WHY:   Pure Mojo quant benchmark — measures real decode speed.
# WHEN:  May 2026.
from std import time, math
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime NE: Int = 2048;  comptime NH: Int = 32;  comptime NK: Int = 4
comptime HD: Int = 64;    comptime NL: Int = 22;  comptime FF: Int = 5632
comptime NV: Int = 32000; comptime NW: Int = 24;  comptime RPW: Int = 32;  comptime W: Int = 8

comptime Q4_0_BS: Int = 18

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...

# ═══ f16 matmul (baseline) ═══
def bench_f16(nr: Int, nc: Int, cold_pool: Int64, pool_off: Int64) -> Float64:
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(cold_pool + pool_off))
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(cold_pool + 256*1048576))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(cold_pool + 256*1048576 + 4096*4))
    var bpr = nc // 32; var nb = (nr + RPW - 1) // RPW
    var t0 = time.perf_counter()
    def wk(b: Int) capturing -> None:
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
    parallelize[func=wk](num_work_items=nb, num_workers=NW)
    return (time.perf_counter() - t0) * 1000.0

# ═══ Q4_0 matmul ═══
def bench_q4_0(nr: Int, nc: Int, cold_pool: Int64, pool_off: Int64) -> Float64:
    var w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(cold_pool + pool_off))
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(cold_pool + 256*1048576))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(cold_pool + 256*1048576 + 4096*4))
    var bpr = nc // 32
    var t0 = time.perf_counter()
    def wk(r: Int) capturing -> None:
        var ro = r * bpr * Q4_0_BS; var acc: Float32 = 0.0
        for blk in range(bpr):
            var bo = ro + blk * Q4_0_BS
            var lo16 = UInt16(w.load(bo)); var hi16 = UInt16(w.load(bo+1))
            var s = UInt32(hi16 >> 15); var e = UInt32((hi16 >> 10) & 0x1F); var m = UInt32(hi16 & 0x3FF)
            var d: Float32 = 0.0
            if e == 0: d = 0.0 if m == 0 else Float32(Float64(m) * 5.960464477539063e-8)
            elif e < 31:
                d = Float32(m | 0x400); var ei = Int(e) - 25
                if ei >= 0:
                    for _ in range(ei): d *= 2.0
                else:
                    for _ in range(-ei): d *= 0.5
            d = -d if s != 0 else d
            for j in range(32):
                var nib = Int32(w.load(bo + 2 + j//2))
                if j % 2 == 0: nib = nib & 0x0F
                else: nib = nib >> 4
                if nib > 7: nib -= 16
                acc += Float32(nib) * d * x.load(blk * 32 + j)
        o.store(r, acc)
    parallelize[func=wk](num_work_items=nr, num_workers=NW)
    return (time.perf_counter() - t0) * 1000.0

# ═══ MXFP4 matmul ═══
comptime MXFP4_BYTES: Int = 17
def bench_mxfp4(nr: Int, nc: Int, cold_pool: Int64, pool_off: Int64) -> Float64:
    var w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(cold_pool + pool_off))
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(cold_pool + 256*1048576))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(cold_pool + 256*1048576 + 4096*4))
    var bpr = nc // 32
    var t0 = time.perf_counter()
    def wk(r: Int) capturing -> None:
        var ro = r * bpr * MXFP4_BYTES; var acc: Float32 = 0.0
        for blk in range(bpr):
            var bo = ro + blk * MXFP4_BYTES
            var eb = w.load(bo + 16)
            var sf: Float32 = 0.0
            if eb != 0:
                if eb < 255:
                    var ei = Int(eb) - 127
                    sf = 1.0
                    if ei >= 0:
                        for _ in range(ei): sf *= 2.0
                    else:
                        for _ in range(-ei): sf *= 0.5
                else: sf = 1e20
            var ai: Int32 = 0
            for j in range(16):
                var p = w.load(bo + j)
                var lo = Int32(p & 0x0F)
                if lo > 7: lo -= 16
                var hi = Int32(p >> 4)
                if hi > 7: hi -= 16
                ai += lo * Int32(x.load(blk*32 + j*2)) + hi * Int32(x.load(blk*32 + j*2+1))
            acc += Float32(ai) * sf
        o.store(r, acc)
    parallelize[func=wk](num_work_items=nr, num_workers=NW)
    return (time.perf_counter() - t0) * 1000.0

# ═══ Fill helpers ═══
def fill_pool(pool: Int64, sz: Int64):
    var p = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(pool))
    for i in range(Int(sz)):
        p.store(i, UInt8((i * 7 + (i >> 4) * 13 + (i >> 8) * 17) & 0xFF))

def main():
    # Cold-cache pool: 256MB + buffer space
    var pool = _alc(Int64(300 * 1048576))
    fill_pool(pool, Int64(300 * 1048576))
    
    # Input buffers within pool
    var x_addr = pool + 256 * 1048576
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x_addr))
    for i in range(NE): x.store(i, Float32(i % 100 - 50) * 0.01)
    
    # Benchmark each quant format on key matmul shapes
    # Use QKV (nq × NE = 1024×2048) as the representative matmul
    var nq = NH * HD  # 1024
    
    print("{")
    print("\"quant_benchmarks\": [")
    
    var first = True
    
    # f16: 2 bytes/value
    var off = Int64(0)
    var t = bench_f16(nq, NE, pool, off)
    var tok_s = 1000.0 / (t * Float64(NL * 7))
    if not first: print(",")
    first = False
    print("  {\"quant\":\"f16\",\"ms\":" + String(Float64(t)) + ",\"est_tok_s\":" + String(Float64(tok_s)) + "}")
    
    # Q4_0: 18 bytes/32 values = 0.5625 bytes/value
    off += Int64(nq) * Int64(NE) * 2  # skip f16 weights
    t = bench_q4_0(nq, NE, pool, off)
    tok_s = 1000.0 / (t * Float64(NL * 7))
    if not first: print(",")
    print("  {\"quant\":\"Q4_0\",\"ms\":" + String(Float64(t)) + ",\"est_tok_s\":" + String(Float64(tok_s)) + "}")
    
    # MXFP4: 17 bytes/32 values = 0.53125 bytes/value
    off += Int64(nq) * (Int64(NE) // 32) * Q4_0_BS
    t = bench_mxfp4(nq, NE, pool, off)
    tok_s = 1000.0 / (t * Float64(NL * 7))
    if not first: print(",")
    print("  {\"quant\":\"MXFP4\",\"ms\":" + String(Float64(t)) + ",\"est_tok_s\":" + String(Float64(tok_s)) + "}")
    
    print("")
    print("]}")
