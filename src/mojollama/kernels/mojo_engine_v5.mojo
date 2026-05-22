# MojoLlama Mojo Engine — Pure Mojo Matmul Kernel
# No Python, no ctypes — pure Mojo SIMD for MXFP4 × Q8_0
from std.prelude import *
from std import time

# MXFP4 block: 16 bytes quants + 1 byte exponent = 17 bytes
comptime MXFP4_BLOCK: Int = 17
# Q8_0 block: 2 bytes f16 + 32 bytes i8 = 34 bytes
comptime Q8_0_BLOCK: Int = 34

# ── Helpers ────────────────────────────────────────────────
fn sext4(v: UInt8) -> Int32:
    if v >= 8:
        return Int32(v) - 16
    return Int32(v)

fn f16_to_f32(h: UInt16) -> Float32:
    var sign = UInt32(h >> 15)
    var exp = UInt32((h >> 10) & 0x1F)
    var mant = UInt32(h & 0x3FF)
    if exp == 0:
        if mant == 0:
            return 0.0
        var m = Float64(mant) * 5.960464477539063e-8
        return Float32(m)
    elif exp == 31:
        return 0.0
    var m = Float32(mant | 0x400)
    var e = Int(exp) - 25
    var result = m
    if e >= 0:
        for _ in range(e):
            result *= 2.0
    else:
        for _ in range(-e):
            result *= 0.5
    return -result if sign != 0 else result

# ── Row-dot using List (no raw pointer arithmetic) ────────
fn dot_row_gptoss(w: List[UInt8], x: List[UInt8], bpr: Int, row: Int) -> Float32:
    var acc: Float32 = 0.0
    var row_off = row * bpr * MXFP4_BLOCK
    for blk in range(bpr):
        var acc_blk: Int32 = 0
        var w_off = row_off + blk * MXFP4_BLOCK
        var x_off = blk * Q8_0_BLOCK
        var ebyte = w[w_off + 16]
        var d_lo = x[x_off]
        var d_hi = x[x_off + 1]
        var q8d = f16_to_f32(UInt16(d_lo) | (UInt16(d_hi) << 8))
        # MXFP4 E8M0 exponent decode: value = 2^(ebyte - 127)
        # Special: ebyte=0 → 0, ebyte=255 → limit to 1e20
        var sf_val: Float32 = 0.0
        if ebyte != 0 and ebyte < 255:
            var sf_exp = Int(ebyte) - 127
            sf_val = 1.0
            if sf_exp >= 0:
                for _ in range(sf_exp):
                    sf_val *= 2.0
            else:
                for _ in range(-sf_exp):
                    sf_val *= 0.5
        elif ebyte >= 255:
            sf_val = 1e20
        var sf = sf_val * q8d
        if sf > 1e20: sf = 1e20
        if sf < -1e20: sf = -1e20
        for j in range(16):
            var p = w[w_off + j]
            var lo = sext4(p & 0x0F)
            var hi = sext4(p >> 4)
            var ql = Int32(Int8(x[x_off + 2 + j * 2]))
            var qh = Int32(Int8(x[x_off + 2 + j * 2 + 1]))
            acc_blk += lo * ql + hi * qh
        acc += Float32(acc_blk) * sf
    return acc

# ── Benchmark ─────────────────────────────────────────────
fn main():
    var N = 2880  # GPT-OSS n_embd
    var n_rows = 256
    var bpr = N / 32
    
    # Fill weight buffer
    var total_w = n_rows * bpr * MXFP4_BLOCK
    var w = List[UInt8](capacity=total_w)
    for i in range(total_w):
        w.append(UInt8((i * 7 + 13) & 0xFF))
    
    # Fill Q8 input
    var q8_total = bpr * Q8_0_BLOCK
    var q = List[UInt8](capacity=q8_total)
    for i in range(q8_total):
        q.append(UInt8((i * 3 + 7) & 0xFF))
    
    # Warmup (use Python for sanity check — no, don't use Python)
    # Pure Mojo warmup + benchmark
    var result: Float32 = 0.0
    for row in range(4):
        result += dot_row_gptoss(w, q, bpr, row)
    
    var t0 = time.perf_counter()
    for row in range(n_rows):
        result += dot_row_gptoss(w, q, bpr, row)
    var t1 = time.perf_counter()
    
    print("Pure Mojo Matmul")
    print("  Dims: ", N, " rows: ", n_rows)
    print("  Result: ", result)
    print("  Time: ", (t1 - t0) * 1000000, " us")
    print("  Rows/s: ", Float64(n_rows) / (t1 - t0))
