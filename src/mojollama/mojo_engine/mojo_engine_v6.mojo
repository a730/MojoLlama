# Mojo v6 — use fgetc for byte-by-byte file reading
from std.prelude import *
from std import time

@extern("fopen")
fn c_fopen(path: String, mode: String) -> Int64: ...

@extern("fclose")
fn c_fclose(fp: Int64) -> Int32: ...

@extern("fgetc")
fn c_fgetc(fp: Int64) -> Int32: ...

fn load_file(path: String) -> List[UInt8]:
    var fp = c_fopen(path, "rb")
    if fp == 0: return List[UInt8]()
    var result = List[UInt8]()
    while True:
        var c = c_fgetc(fp)
        if c < 0: break
        result.append(UInt8(c))
    c_fclose(fp)
    return result^

fn main() raises:
    print("MojoLlama v6")
    print("============")
    
    var meta = load_file("/tmp/mojo_weights/gpt-oss/meta.bin")
    print("meta: ", len(meta), " bytes")
    if len(meta) > 0:
        for i in range(min(60, len(meta))):
            var c = meta[i]
            print(chr(Int(c)) if c >= 32 and c <= 126 else ".", end="")
        print()
    
    # Pure Mojo matmul benchmark
    var N = 2880
    var n_rows = 256
    var bpr = N / 32
    comptime MXFP4_BS: Int = 17
    var total_w = n_rows * bpr * MXFP4_BS
    
    var W = List[UInt8](capacity=total_w)
    for i in range(total_w):
        W.append(UInt8((i * 7 + 13) & 0xFF))
    
    var x = List[Float32](capacity=N)
    for i in range(N):
        x.append(Float32(Float64((i * 3) % 100 - 50) / 50.0))
    
    var result: Float32 = 0.0
    for row in range(4):
        var off = row * bpr * MXFP4_BS
        for blk in range(bpr):
            var wb = off + blk * MXFP4_BS
            var ebyte = W[wb + 16]
            var sf: Float32 = 0.0
            if ebyte != 0 and ebyte < 255:
                var e = Int(ebyte) - 127
                sf = 1.0
                if e >= 0:
                    for _ in range(e): sf *= 2.0
                else:
                    for _ in range(-e): sf *= 0.5
            elif ebyte >= 255: sf = 1e20
            for j in range(16):
                var p = W[wb + j]
                var lo = Int32(p & 0x0F)
                if lo > 7: lo -= 16
                var hi = Int32(p >> 4)
                if hi > 7: hi -= 16
                result += Float32(lo * Int32(x[blk * 32 + j * 2]) + hi * Int32(x[blk * 32 + j * 2 + 1])) * sf
    
    var t0 = time.perf_counter()
    for row in range(n_rows):
        var off = row * bpr * MXFP4_BS
        for blk in range(bpr):
            var wb = off + blk * MXFP4_BS
            var ebyte = W[wb + 16]
            var sf: Float32 = 0.0
            if ebyte != 0 and ebyte < 255:
                var e = Int(ebyte) - 127
                sf = 1.0
                if e >= 0:
                    for _ in range(e): sf *= 2.0
                else:
                    for _ in range(-e): sf *= 0.5
            elif ebyte >= 255: sf = 1e20
            for j in range(16):
                var p = W[wb + j]
                var lo = Int32(p & 0x0F)
                if lo > 7: lo -= 16
                var hi = Int32(p >> 4)
                if hi > 7: hi -= 16
                result += Float32(lo * Int32(x[blk * 32 + j * 2]) + hi * Int32(x[blk * 32 + j * 2 + 1])) * sf
    var t1 = time.perf_counter()
    
    print("Result: ", result)
    print("Rows/s: ", Int(Float64(n_rows) / (t1 - t0)))
