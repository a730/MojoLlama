# MojoLlama — SIMD-optimized MXFP4 matmul
from std import time

def mxfp4_simd(o: Span[mut=True, Float32, _], w: Span[UInt8, _], x: Span[Float32, _],
               nr: Int, nc: Int):
    var bpr = nc // 32
    for r in range(nr):
        var acc = SIMD[DType.float32, 8](0.0)
        var ro = r * bpr * 17
        for blk in range(bpr):
            var bo = ro + blk * 17
            var eb = w[bo + 16]
            if eb == 0: continue
            var ei = Int(eb) - 127
            var sf: Float32 = 1.0
            if ei >= 0:
                for _ in range(ei): sf *= 2.0
            else:
                for _ in range(-ei): sf *= 0.5
            if sf > 1e20: sf = 1e20
            var sv = SIMD[DType.float32, 8](sf)
            var xb = blk * 32
            
            # Bytes 0-7: 16 values (lo/even-x, hi/odd-x)
            var b0 = SIMD[DType.uint8, 8](w[bo], w[bo+1], w[bo+2], w[bo+3],
                                          w[bo+4], w[bo+5], w[bo+6], w[bo+7])
            var b0h = b0 >> UInt8(4)
            b0 = b0 & UInt8(0x0F)
            var lo0 = ((b0 ^ UInt8(8)).cast[DType.int32]() - Int32(8)).cast[DType.float32]()
            var hi0 = ((b0h ^ UInt8(8)).cast[DType.int32]() - Int32(8)).cast[DType.float32]()
            var xl0 = SIMD[DType.float32, 8](x[xb], x[xb+2], x[xb+4], x[xb+6],
                                              x[xb+8], x[xb+10], x[xb+12], x[xb+14])
            var xh0 = SIMD[DType.float32, 8](x[xb+1], x[xb+3], x[xb+5], x[xb+7],
                                              x[xb+9], x[xb+11], x[xb+13], x[xb+15])
            acc = acc + lo0 * xl0 * sv + hi0 * xh0 * sv
            
            # Bytes 8-15: values 16-31
            var b1 = SIMD[DType.uint8, 8](w[bo+8], w[bo+9], w[bo+10], w[bo+11],
                                          w[bo+12], w[bo+13], w[bo+14], w[bo+15])
            var b1h = b1 >> UInt8(4)
            b1 = b1 & UInt8(0x0F)
            var lo1 = ((b1 ^ UInt8(8)).cast[DType.int32]() - Int32(8)).cast[DType.float32]()
            var hi1 = ((b1h ^ UInt8(8)).cast[DType.int32]() - Int32(8)).cast[DType.float32]()
            var xl1 = SIMD[DType.float32, 8](x[xb+16], x[xb+18], x[xb+20], x[xb+22],
                                              x[xb+24], x[xb+26], x[xb+28], x[xb+30])
            var xh1 = SIMD[DType.float32, 8](x[xb+17], x[xb+19], x[xb+21], x[xb+23],
                                              x[xb+25], x[xb+27], x[xb+29], x[xb+31])
            acc = acc + lo1 * xl1 * sv + hi1 * xh1 * sv
        o[r] = acc.reduce_add()

def mxfp4_scalar(o: Span[mut=True, Float32, _], w: Span[UInt8, _], x: Span[Float32, _],
                 nr: Int, nc: Int):
    var bpr = nc // 32
    for r in range(nr):
        var acc: Float32 = 0.0
        var ro = r * bpr * 17
        for blk in range(bpr):
            var bo = ro + blk * 17
            var eb = w[bo + 16]
            if eb == 0: continue
            var ei = Int(eb) - 127; var sf: Float32 = 1.0
            if ei >= 0:
                for _ in range(ei): sf *= 2.0
            else:
                for _ in range(-ei): sf *= 0.5
            if sf > 1e20: sf = 1e20
            var ai: Int32 = 0
            for j in range(16):
                var p = w[bo + j]
                var lo = Int32(p & 0x0F)
                if lo > 7: lo -= 16
                var hi = Int32(p >> 4)
                if hi > 7: hi -= 16
                ai += lo * Int32(x[blk * 32 + j * 2]) + hi * Int32(x[blk * 32 + j * 2 + 1])
            acc += Float32(ai) * sf
        o[r] = acc

def main():
    print("Mojo SIMD MXFP4 Benchmark")
    print("========================")
    var nc = 2880; var nr = 256; var bpr = nc // 32
    
    var w = alloc[UInt8](nr * bpr * 17)
    for i in range(nr * bpr * 17): w.store(i, UInt8((i * 7 + 13) & 0xFF))
    var ws = Span[UInt8, _](ptr=w, length=nr * bpr * 17)
    var x = alloc[Float32](nc)
    for i in range(nc): x.store(i, Float32(Float64((i * 3) % 100 - 50) / 50.0))
    var xs = Span[Float32, _](ptr=x, length=nc)
    var os = alloc[Float32](nr); var ov = alloc[Float32](nr)
    var oss = Span[mut=True, Float32, _](ptr=os, length=nr)
    var osv = Span[mut=True, Float32, _](ptr=ov, length=nr)
    
    mxfp4_scalar(oss, ws, xs, 4, nc)
    mxfp4_simd(osv, ws, xs, 4, nc)
    mxfp4_scalar(oss, ws, xs, nr, nc)
    mxfp4_simd(osv, ws, xs, nr, nc)
    
    var max_err: Float32 = 0.0
    for i in range(nr):
        var e = oss[i] - osv[i]
        if e < 0.0: e = -e
        if e > max_err: max_err = e
    print("Max error:", max_err)
    
    var bn = 200
    var t0 = time.perf_counter()
    for _ in range(bn): mxfp4_scalar(oss, ws, xs, nr, nc)
    var t1 = time.perf_counter()
    var ms = (t1 - t0) * 1000.0 / Float64(bn)
    print("Scalar:", ms, "ms =>", Int(Float64(nr) / (ms / 1000.0)), "rows/s")
    
    t0 = time.perf_counter()
    for _ in range(bn): mxfp4_simd(osv, ws, xs, nr, nc)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / Float64(bn)
    var rps = Float64(nr) / (ms / 1000.0)
    print("SIMD:  ", ms, "ms =>", Int(rps), "rows/s")
    
    print("\nExpert estimate (SIMD):")
    var exp_ms = Float64(2880) / rps * 1000.0
    print(" 1 expert x3 matmuls:", exp_ms * 3.0, "ms")
    print(" 4 experts x3:", exp_ms * 12.0, "ms")
    print(" Per layer:", exp_ms * 12.0 + 3.0, "ms")
    print(" 24 layers:", (exp_ms * 12.0 + 3.0) * 24.0, "ms")
    print(" Tok/s:", Float64(1000.0) / ((exp_ms * 12.0 + 3.0) * 24.0 + 1.0))
    
    w.free(); x.free(); os.free(); ov.free()
