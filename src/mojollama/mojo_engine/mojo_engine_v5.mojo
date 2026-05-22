from std.prelude import *
from std import time

comptime MXFP4_BS: Int = 17
comptime Q8_0_BS: Int = 34

fn decode_e8m0(ebyte: UInt8) -> Float32:
    if ebyte == 0: return 0.0
    if ebyte >= 255: return 1e20
    var e = Int(ebyte) - 127
    var result: Float32 = 1.0
    if e >= 0:
        for _ in range(e): result *= 2.0
    else:
        for _ in range(-e): result *= 0.5
    return result

fn f16_to_f32(h: UInt16) -> Float32:
    var sign = UInt32(h >> 15)
    var exp = UInt32((h >> 10) & 0x1F)
    var mant = UInt32(h & 0x3FF)
    if exp == 0:
        if mant == 0: return 0.0
        return Float32(Float64(mant) * 5.960464477539063e-8)
    elif exp == 31: return 0.0
    var m = Float32(mant | 0x400)
    var e = Int(exp) - 25
    var result = m
    if e >= 0:
        for _ in range(e): result *= 2.0
    else:
        for _ in range(-e): result *= 0.5
    return -result if sign != 0 else result

fn sext4(v: UInt8) -> Int32:
    if v >= 8: return Int32(v) - 16
    return Int32(v)

fn mxfp4_row_dot_f32(
    W: List[UInt8], x: List[Float32], n_cols: Int, row: Int
) -> Float32:
    var bpr = n_cols / 32
    var row_off = row * bpr * MXFP4_BS
    var acc: Float32 = 0.0
    for blk in range(bpr):
        var acc_i: Int32 = 0
        var w_off = row_off + blk * MXFP4_BS
        var sf = decode_e8m0(W[w_off + 16])
        if sf > 1e20: sf = 1e20
        for j in range(16):
            var p = W[w_off + j]
            var lo = sext4(p & 0x0F)
            var hi = sext4(p >> 4)
            var x_lo = Int32(x[blk * 32 + j * 2])
            var x_hi = Int32(x[blk * 32 + j * 2 + 1])
            acc_i += lo * x_lo + hi * x_hi
        acc += Float32(acc_i) * sf
    return acc

fn main():
    var N = 2880
    var n_rows = 256
    var bpr = N / 32
    var total_w = n_rows * bpr * MXFP4_BS
    
    var W = List[UInt8](capacity=total_w)
    for i in range(total_w):
        W.append(UInt8((i * 7 + 13) & 0xFF))
    
    var x = List[Float32](capacity=N)
    for i in range(N):
        x.append(Float32((i * 3) % 100 - 50) / 50.0)
    
    var result: Float32 = 0.0
    for row in range(4):
        result += mxfp4_row_dot_f32(W, x, N, row)
    
    var t0 = time.perf_counter()
    for row in range(n_rows):
        result += mxfp4_row_dot_f32(W, x, N, row)
    var t1 = time.perf_counter()
    
    print("Mojo Engine v5 — MXFP4×f32 kernel")
    print("  Dims: ", N, " rows: ", n_rows)
    print("  Rows/s: ", Int(Float64(n_rows) / (t1 - t0)))
    print("  Check: ", result)
