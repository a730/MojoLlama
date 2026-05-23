# benchmark_all.mojo — Next-Gen Mojo Benchmark
# All architectures, 8 matmul shapes, thread sweep, CSV output
#
# Build: mojo build benchmark_all.mojo && cp benchmark_all /tmp/
# Run: OMP_PLACES=cores OMP_PROC_BIND=close ./benchmark_all 32

from std import time
from std.sys import argv
from std.algorithm.backend.cpu.parallelize import parallelize

comptime W: Int = 8
comptime RPW: Int = 8

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...
@extern("free")
def _c_free(p: Int64) abi("C") -> None: ...
@extern("open")
def _open(p: UnsafePointer[UInt8, MutExternalOrigin], f: Int) abi("C") -> Int: ...
@extern("close")
def _close(fd: Int) abi("C") -> Int: ...
@extern("write")
def _write(fd: Int, b: UnsafePointer[UInt8, MutExternalOrigin], c: Int64) abi("C") -> Int64: ...

def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1); var e = Int((h >> 10) & 0x1F); var m = Int(h & 0x3FF)
    if e == 0: var r = Float32(m) * 5.960464477539063e-8; return -r if s != 0 else r
    if e == 31: return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    var tmp = alloc[UInt8](4)
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp)).store(0, bits)
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp)).load(0)

def fill_pool16(p: UnsafePointer[UInt16, MutExternalOrigin], n: Int):
    for i in range(n): p.store(i, UInt16(0x3800))  # f16(0.5)

def matmul_f16(wa: Int64, x: UnsafePointer[Float32, MutExternalOrigin],
               o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int):
    var w = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nrb = (nr + RPW - 1) // RPW
    def wk(wi: Int) capturing:
        var rs = wi * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var acc = SIMD[DType.float32, W](0.0); var c = 0
            while c + W <= nc:
                acc = acc + h2f(w.load(r * nc + c)) * x.load[width=W](c); c += W
            var s = acc.reduce_add()
            while c < nc: s += h2f(w.load(r * nc + c)) * x.load(c); c += 1
            o.store(r, s)
    parallelize[func=wk](num_work_items=nrb, num_workers=nw)

def make_path(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var buf = alloc[UInt8](s.byte_length() + 1)
    var sp = s.unsafe_ptr()
    for i in range(s.byte_length()): buf.store(i, sp.load(i))
    buf.store(s.byte_length(), UInt8(0))
    return buf

def csv(fd: Int, s: String): _ = _write(fd, make_path(s), Int64(s.byte_length()))

# ─── Benchmark one model ───
def bench_model(mname: String, NE: Int, NH: Int, NK: Int, HD: Int,
                NL: Int, FF: Int, NV: Int, nw_max: Int, pool_base: Int64,
                csv_fd: Int):
    var QI = NH * HD
    var shapes_nr = [QI, NK*HD, NK*HD, NE, FF, FF, NE, NV]
    var shapes_nc = [NE, NE, NE, QI, NE, NE, FF, NE]
    var shapes_desc = ["Q_proj", "K_proj", "V_proj", "O_proj",
                      "FFN_gate", "FFN_up", "FFN_down", "LM_head"]
    var shapes_wt = [3.0, 1.0, 1.0, 3.0, 10.0, 10.0, 10.0, 6.0]
    var thr = [1, 2, 4, 8, 16, 24, 32]
    var weighted_ms = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var total_wt = 44.0  # sum of shapes_wt
    
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NV * 4))))
    for i in range(NE): x.store(i, 1.0)
    
    print("───", mname, "───")
    print("  ", NE, "x", FF, "x", NV, " ", NL, "layers")
    print()
    print("  Matmul      |", end="")
    for t in range(7): print(" ", thr[t], "t |", end="")
    print()
    print("  ", "─" * 50, sep="")
    
    for si in range(8):
        var nr = shapes_nr[si]; var nc = shapes_nc[si]
        var desc = shapes_desc[si]; var wt = shapes_wt[si]
        var wb = nr * nc * 2
        # Offset into cold-cache pool
        var pool_off = Int64((si * 1000000) % (256 * 1024 * 1024 - wb))
        var wa = pool_base + pool_off
        
        var pd = desc + "         "
        print("  ", pd, " |", end="")
        
        for ti in range(7):
            var tnw = thr[ti]
            var nw_u = tnw if tnw <= nw_max else nw_max
            var iters = 3 if tnw >= 16 else 5
            
            var t0 = time.perf_counter()
            for it in range(iters): matmul_f16(Int(Int64(wa)), x, o, nr, nc, nw_u)
            var t1 = time.perf_counter()
            var ms = (t1 - t0) * 1000.0 / Float64(iters)
            var gb_s = (Float64(wb) / 1e9) / (ms / 1000.0)
            
            weighted_ms[ti] += ms * wt
            print(" ", String(Int(ms)), " |", end="")
            
            csv(csv_fd, mname + "," + String(1) + "," + String(tnw) + "," +
                desc + "," + String(nr) + "," + String(nc) + "," +
                String(ms) + "," + String(gb_s) + "\n")
        print()
    
    # Weighted tok/s
    print("  est.tok/s  |", end="")
    for ti in range(7):
        var avg_ms = weighted_ms[ti] / total_wt
        var total = avg_ms * Float64(NL) + avg_ms
        var ts = 1000.0 / total if total > 0.0 else 0.0
        print(" ", Int(ts), " |", end="")
    print()
    print()
    
    # Note: buffers allocated with _alc, freed on process exit

def main() raises:
    print("MojoLlama Next-Gen Benchmark")
    print("=============================")
    
    var args = argv(); var nw = 32
    if len(args) > 1: nw = Int(String(args[1]))
    
    var csv_fd = _open(make_path(String("/tmp/bench_results.csv")), 0x41)
    csv(csv_fd, "model,batch,threads,matmul,nr,nc,ms,gb_s\n")
    
    # Cold-cache pool (256MB, all f16 0.5)
    var pool = _alc(Int64(268435456))
    fill_pool16(UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(pool)), 134217728)
    
    # Benchmark each model (pass csv_fd for CSV output)
    bench_model("TinyLlama",  2048, 32,  4,  64, 22, 5632,  32000,  nw, pool, csv_fd)
    bench_model("ZAYA1-8B",   2048, 8,   2,  128, 80, 4096,  262147, nw, pool, csv_fd)
    bench_model("GPT-OSS-20B", 2880, 64, 8,  64, 24, 2880,  201088, nw, pool, csv_fd)
    bench_model("Llama3.2-1B", 2048, 32, 8,  64, 16, 8192,  128256, nw, pool, csv_fd)
    
    _ = _close(csv_fd)
    print("Results: /tmp/bench_results.csv")
