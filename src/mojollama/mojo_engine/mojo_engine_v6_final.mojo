# Mojo v6 Final — links with read_helpers.o
from std.prelude import *
from std import time

@extern("read_u8")
fn c_read_u8(addr: Int64) -> UInt8: ...

@extern("read_f32") 
fn c_read_f32(addr: Int64) -> Float32: ...

@extern("load_file_size")
fn c_load_file_size(path: String) -> Int64: ...

@extern("load_file_data")
fn c_load_file_data(path: String) -> Int64: ...

@extern("free_buf")
fn c_free_buf(ptr: Int64): ...

fn main():
    print("MojoLlama v6 Final")
    print("==================")
    
    # Load meta file  
    var sz = c_load_file_size("/tmp/mojo_weights/gpt-oss/meta.bin")
    var meta = c_load_file_data("/tmp/mojo_weights/gpt-oss/meta.bin")
    if meta != 0 and sz > 0:
        print("meta: ", sz, " bytes")
        print("First 60: ", end="")
        for i in range(min(60, Int(sz))):
            var c = c_read_u8(meta + Int64(i))
            print(chr(Int(c)) if c >= 32 and c <= 126 else ".", end="")
        print()
        c_free_buf(meta)
    
    # Load embedding
    var esz = c_load_file_size("/tmp/mojo_weights/gpt-oss/emb.bin")
    var emb = c_load_file_data("/tmp/mojo_weights/gpt-oss/emb.bin")
    if emb != 0 and esz > 0:
        var n_floats = Int(esz / 4)
        print("emb: ", esz, " bytes = ", n_floats, " floats")
        print("emb[0:5]: ", end="")
        for i in range(min(5, n_floats)):
            print(c_read_f32(emb + Int64(i * 4)), " ", end="")
        print()
        c_free_buf(emb)
    
    # MXFP4 matmul benchmark
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
            var ebyte = W[off + blk * MXFP4_BS + 16]
            var sf: Float32 = 0.0
            if ebyte != 0 and ebyte < 255:
                var e = Int(ebyte) - 127; sf = 1.0
                if e >= 0:
                    for _ in range(e): sf *= 2.0
                else:
                    for _ in range(-e): sf *= 0.5
            elif ebyte >= 255: sf = 1e20
            for j in range(16):
                var p = W[off + blk * MXFP4_BS + j]
                var lo = Int32(p & 0x0F)
                if lo > 7: lo -= 16
                var hi = Int32(p >> 4)
                if hi > 7: hi -= 16
                result += Float32(lo * Int32(x[blk * 32 + j * 2]) + hi * Int32(x[blk * 32 + j * 2 + 1])) * sf
    
    var t0 = time.perf_counter()
    for row in range(n_rows):
        var off = row * bpr * MXFP4_BS
        for blk in range(bpr):
            var ebyte = W[off + blk * MXFP4_BS + 16]
            var sf: Float32 = 0.0
            if ebyte != 0 and ebyte < 255:
                var e = Int(ebyte) - 127; sf = 1.0
                if e >= 0:
                    for _ in range(e): sf *= 2.0
                else:
                    for _ in range(-e): sf *= 0.5
            elif ebyte >= 255: sf = 1e20
            for j in range(16):
                var p = W[off + blk * MXFP4_BS + j]
                var lo = Int32(p & 0x0F)
                if lo > 7: lo -= 16
                var hi = Int32(p >> 4)
                if hi > 7: hi -= 16
                result += Float32(lo * Int32(x[blk * 32 + j * 2]) + hi * Int32(x[blk * 32 + j * 2 + 1])) * sf
    var t1 = time.perf_counter()
    
    print("Result: ", result)
    print("Rows/s: ", Int(Float64(n_rows) / (t1 - t0)))
