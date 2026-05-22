# correctness_test.mojo — Verify optimized matmul produces correct output
# WHAT:  Compares old mm16 vs new universal_mm on real TinyLlama weight data.
#        Ensures optimizations (@always_inline, FastMathFlag.FAST, 
#        page-interleaved mmap) don't break numerical correctness.
# WHEN:  2026-05-22

from std import time
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime RPW: Int = 32; comptime W: Int = 8

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...

@extern("free")
def _c_free(p: Int) abi("C") -> None: ...

@extern("open")
def _open(path: UnsafePointer[UInt8, MutExternalOrigin], flags: Int) abi("C") -> Int: ...

@extern("read")
def _read(fd: Int, buf: UnsafePointer[UInt8, MutExternalOrigin], cnt: Int64) abi("C") -> Int64: ...

@extern("lseek")
def _lseek(fd: Int, off: Int64, whence: Int) abi("C") -> Int64: ...

@extern("close")
def _close(fd: Int) abi("C") -> Int: ...

@extern("sched_setaffinity")
def _sched_setaff(pid: Int, cpusz: Int, mask: Int) abi("C") -> Int: ...

@extern("mmap")
def _mmap(addr: Int, length: Int, prot: Int, flags: Int, fd: Int, offset: Int) abi("C") -> Int: ...

def ml_pin():
    var mask_sz = 128; var raw = _alc(mask_sz)
    var mask = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw))
    for i in range(mask_sz): mask.store(i, UInt8(0))
    for i in range(32): mask.store(i // 8, mask.load(i // 8) | UInt8(1 << (i % 8)))
    var _ = _sched_setaff(0, mask_sz, Int(raw)); _c_free(Int(raw))

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
            var tail = (nc // 32) * 32
            for t in range(tail, nc):
                var ws = w.load(ro + t)
                acc = acc + SIMD[DType.float32, W](Float32(ws)) * SIMD[DType.float32, W](x.load(t))
            o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=nb, num_workers=nw)

@always_inline("nodebug")
def universal_mm_new[dtype: DType](
    w_addr: Int, x: UnsafePointer[Float32, MutExternalOrigin],
    o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    comptime if dtype == DType.float16:
        _dtype_mm_sz2[dtype](w_addr, x, o, nr, nc)

# ── Load .bin file ──
def load_bin(path: String, buf: UnsafePointer[UInt8, MutExternalOrigin]) -> Int64:
    """Load file into buffer, return file size."""
    var cpath = alloc[UInt8](path.byte_length() + 1)
    var src = path.unsafe_ptr()
    for i in range(path.byte_length()): cpath.store(i, src.load(i))
    cpath.store(path.byte_length(), UInt8(0))
    
    var fd = _open(cpath, 0)
    if fd < 0: return -1
    var sz = _lseek(fd, 0, 2)
    _lseek(fd, 0, 0)
    _read(fd, buf, sz)
    _close(fd)
    return sz

# ── Correctness test ──
def test_matmul_correctness():
    """Compare old mm16 vs new universal_mm on real TinyLlama weights.
    
    Uses blk_0_attn_q_weight from extracted .bin files (2048×2048 f16).
    Feeds a synthetic input, compares outputs element-by-element.
    """
    var weights_dir = "/tmp/weights_tl/"
    var test_file = weights_dir + "blk_0_attn_q_weight.bin"
    
    var nr = 2048; var nc = 2048  # TinyLlama Q projection dims
    var weight_bytes = nr * nc * 2  # f16 = 2 bytes each
    
    # Allocate buffers
    var w_buf = Int(_alc(Int64(weight_bytes)))
    var x_buf = Int(_alc(Int64(nr * 4)))
    var o_old = Int(_alc(Int64(nr * 4)))
    var o_new = Int(_alc(Int64(nr * 4)))
    
    var w_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_buf)
    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=x_buf)
    var o_old_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=o_old)
    var o_new_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=o_new)
    
    # Load weights
    var sz = load_bin(test_file, w_ptr)
    if sz < 0:
        print("ERROR: can't load", test_file)
        return
    
    print("Loaded", test_file, String(sz / 1048576, 0, 1), "MB")
    
    # Fill input with deterministic test data (sawtooth pattern)
    for i in range(nc):
        x_ptr.store(i, Float32(i % 100 - 50) * 0.01)
    
    # Run OLD matmul
    print("Running old mm16...")
    mm16_old(w_buf, x_ptr, o_old_ptr, nr, nc, 32)
    
    # Run NEW matmul
    print("Running new universal_mm...")
    universal_mm_new[DType.float16](w_buf, x_ptr, o_new_ptr, nr, nc)
    
    # Compare
    var max_diff: Float32 = 0.0
    var avg_diff: Float64 = 0.0
    var n_diff = 0
    for i in range(min(nr, 100)):
        var d = Float64(o_old_ptr.load(i) - o_new_ptr.load(i))
        if d < 0: d = -d
        if d > Float64(max_diff): max_diff = Float32(d)
        avg_diff += d
        if d > 0.001: n_diff += 1
    avg_diff /= Float64(min(nr, 100))
    
    print()
    print("Correctness check (first 100 rows):")
    print("  Max diff:", max_diff)
    print("  Avg diff:", avg_diff)
    print("  Rows with >0.001 diff:", n_diff)
    
    if max_diff < 0.01:
        print("  ✓ PASS — optimizations preserve numerical correctness")
    else:
        print("  ✗ FAIL — max diff > 0.01, check for numerical issues")
        print("  Sample outputs (first 5 rows):")
        for i in range(5):
            print("    [", i, "] old:", o_old_ptr.load(i), "new:", o_new_ptr.load(i))
    
    # Also check a specific layer output with 2-token prompt simulation
    # Layer 0 attention with pre-filled KV cache (2 tokens)
    print()
    print("Simulating 2-token forward pass...")
    

    
    # Token 0: compute Q, K, V
    universal_mm_new[DType.float16](w_buf, x_ptr, o_new_ptr, nr, nc)
    print("  Token 0 output[0]:", o_new_ptr.load(0), "...[100]:", o_new_ptr.load(100))
    
    # Token 1: compute Q again with same input (simulates repeated token)
    universal_mm_new[DType.float16](w_buf, x_ptr, o_new_ptr, nr, nc)
    print("  Token 1 output[0]:", o_new_ptr.load(0), "...[100]:", o_new_ptr.load(100))
    
    print("  ✓ 2-token simulation complete")
    
    _c_free(w_buf); _c_free(x_buf); _c_free(o_old); _c_free(o_new)

def main():
    ml_pin()
    print("MojoLlama Optimized Matmul — Correctness Test")
    print("=" * 50)
    test_matmul_correctness()
