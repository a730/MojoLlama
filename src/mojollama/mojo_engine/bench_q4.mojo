# bench_q4.mojo — Q4_0 vs f16 matmul throughput benchmark
# Tests whether SIMD-vectorized Q4_0 decode can beat f16 bandwidth savings.

from std import time
from std.math import sqrt
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime W: Int = 8
comptime RPW: Int = 8
comptime NW: Int = 32
comptime QK: Int = 32  # Q4_0 block size

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...

@extern("free")
def _c_free(p: Int64) abi("C") -> None: ...

@extern("open")
def _open(p: UnsafePointer[UInt8, MutExternalOrigin], f: Int) abi("C") -> Int: ...

@extern("read")
def _read(fd: Int, b: UnsafePointer[UInt8, MutExternalOrigin], c: Int64) abi("C") -> Int64: ...

@extern("lseek")
def _lseek(fd: Int, o: Int64, w: Int) abi("C") -> Int64: ...

@extern("close")
def _close(fd: Int) abi("C") -> Int: ...

# ── f16 to f32 ──
def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1)
    var e = Int((h >> 10) & 0x1F)
    var m = Int(h & 0x3FF)
    if e == 0:
        var r = Float32(m) * 5.960464477539063e-8
        if s == 1: return -r
        return r
    if e == 31: return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    return Float32(bits)

# ── f16 matmul ──
@always_inline("nodebug")
def _mm_f16(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    def wk(wi: Int) capturing:
        var rs = wi * RPW
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
    parallelize[func=wk](num_work_items=nb, num_workers=NW)

# ── Q4_0 matmul (SIMD-vectorized decode) ──
@always_inline("nodebug")
def _mm_q4_0(q4_ptr: Int,
             x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin],
             nr: Int, nc: Int):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(q4_ptr))
    var nb = (nr + RPW - 1) // RPW
    def wk(wi: Int) capturing:
        var rs = wi * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * (nc // QK * 18)
            var acc = SIMD[DType.float32, W](0.0)
            var col = 0
            while col < nc:
                # Load scale
                var lo_byte = Int(q.load(ro + (col // QK) * 18 + 0))
                var hi_byte = Int(q.load(ro + (col // QK) * 18 + 1))
                var scale_raw = UInt16(lo_byte | (hi_byte << 8))
                var scale = h2f(scale_raw)
                var scale_v = SIMD[DType.float32, W](scale)

                # Decode 32 quantized weights in 4 groups of 8
                var qoff = ro + (col // QK) * 18 + 2
                comptime for chunk in range(4):
                    var b0 = Int(q.load(qoff + chunk * 4 + 0))
                    var b1 = Int(q.load(qoff + chunk * 4 + 1))
                    var b2 = Int(q.load(qoff + chunk * 4 + 2))
                    var b3 = Int(q.load(qoff + chunk * 4 + 3))

                    # Extract 8 nibbles from 4 bytes, sign-extend
                    var n0 = (b0 & 0x0F) - 16 * ((b0 & 0x0F) >> 3)
                    var n1 = ((b0 >> 4) & 0x0F) - 16 * (((b0 >> 4) & 0x0F) >> 3)
                    var n2 = (b1 & 0x0F) - 16 * ((b1 & 0x0F) >> 3)
                    var n3 = ((b1 >> 4) & 0x0F) - 16 * (((b1 >> 4) & 0x0F) >> 3)
                    var n4 = (b2 & 0x0F) - 16 * ((b2 & 0x0F) >> 3)
                    var n5 = ((b2 >> 4) & 0x0F) - 16 * (((b2 >> 4) & 0x0F) >> 3)
                    var n6 = (b3 & 0x0F) - 16 * ((b3 & 0x0F) >> 3)
                    var n7 = ((b3 >> 4) & 0x0F) - 16 * (((b3 >> 4) & 0x0F) >> 3)

                    # Build float32 vector and FMA
                    var wv = SIMD[DType.float32, W](
                        Float32(n0), Float32(n1), Float32(n2), Float32(n3),
                        Float32(n4), Float32(n5), Float32(n6), Float32(n7)
                    )
                    wv = wv * scale_v
                    var xv = x.load[width=W](col + chunk * 8)
                    acc = wv.fma[FastMathFlag.FAST](xv, acc)
                col += 32
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=NW)

def load_file(path: String) -> Int64:
    var p = alloc[UInt8](path.byte_length() + 1)
    var sp = path.unsafe_ptr()
    for i in range(path.byte_length()):
        p.store(i, sp.load(i))
    p.store(path.byte_length(), UInt8(0))
    var fd = _open(p, 0)
    if fd < 0: return -1
    var sz = _lseek(fd, 0, 2)
    _ = _lseek(fd, 0, 0)
    var buf = _alc(sz)
    _ = _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
    _ = _close(fd)
    return buf

# ── Main ──
def main():
    var nr = 2048
    var nc = 2048
    print("Benchmarking matmul ", nr, "x", nc)

    # Allocate f16 weight matrix
    var w_f16 = _alc(Int64(nr * nc * 2))
    var w_ptr = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(w_f16))
    # Fill with random-ish f16 values
    for i in range(nr * nc):
        var v = Float32(i % 100) / 100.0 - 0.5  # roughly uniform [-0.5, 0.5]
        # Float32 → f16 (simple truncation for now)
        # Actually, let's just store some values
        w_ptr.store(i, UInt16(i & 0xFFFF))

    # Allocate Q4_0 weight buffer
    var nblk = nr * (nc // QK) * 18
    var w_q4 = _alc(Int64(nblk))

    # Quick convert: set each block's max to 7.0, quantize weights to nearest int
    # This is a rough conversion for benchmarking only
    var f16w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_f16))
    var q4w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_q4))
    for r in range(nr):
        for blk in range(nc // QK):
            # Find max abs value in this block
            var maxv = Float32(0.0)
            for i in range(QK):
                var lo = Int(f16w.load(r * nc * 2 + blk * QK * 2 + i * 2 + 0))
                var hi = Int(f16w.load(r * nc * 2 + blk * QK * 2 + i * 2 + 1))
                var raw = UInt16(lo | (hi << 8))
                var f = h2f(raw)
                var fa = f
                if fa < 0: fa = -fa
                if fa > maxv: maxv = fa

            var scale = maxv / 7.0
            if scale < 1e-10: scale = 1.0

            # Store scale as f16 (approximation)
            var scale_bits = UInt16(0)
            if maxv > 1e-10:
                # Very crude f16 encoding
                var s = 0
                var e = 0
                var m = 0
                var f32_bits = UInt32(0)
                # Skip proper encoding for benchmark; use a fixed scale
                scale_bits = UInt16(14336 + Int(maxv * 512.0 + 0.5))  # rough

            var qoff = r * (nc // QK) * 18 + blk * 18
            q4w.store(qoff + 0, UInt8(Int(scale_bits) & 0xFF))
            q4w.store(qoff + 1, UInt8((Int(scale_bits) >> 8) & 0xFF))

            # Quantize weights
            var d = scale
            for i in range(QK):
                var lo = Int(f16w.load(r * nc * 2 + blk * QK * 2 + i * 2 + 0))
                var hi = Int(f16w.load(r * nc * 2 + blk * QK * 2 + i * 2 + 1))
                var raw = UInt16(lo | (hi << 8))
                var f = h2f(raw)
                var qv = Int(f / d + 0.5)
                if qv > 7: qv = 7
                if qv < -8: qv = -8
                if qv < 0: qv += 16  # store as unsigned 4-bit

                var byte_idx = qoff + 2 + i // 2
                var existing = Int(q4w.load(byte_idx))
                if (i & 1) == 0:
                    q4w.store(byte_idx, UInt8((existing & 0xF0) | (qv & 0x0F)))
                else:
                    q4w.store(byte_idx, UInt8((existing & 0x0F) | ((qv & 0x0F) << 4)))

    # Allocate input/output
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nc * 4))))
    var o_f16 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nr * 4))))
    var o_q4 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nr * 4))))

    for i in range(nc):
        x.store(i, Float32(i % 100) / 100.0)

    # Warmup
    _mm_f16(Int(w_f16), x, o_f16, nr, nc)
    _mm_q4_0(Int(w_q4), x, o_q4, nr, nc)

    # Benchmark f16
    var t0 = time.perf_counter()
    var iters = 100
    for _ in range(iters):
        _mm_f16(Int(w_f16), x, o_f16, nr, nc)
    var t1 = time.perf_counter()
    var f16_ms = (t1 - t0) * 1000.0 / Float64(iters)
    print("f16:  ", Int(f16_ms * 1000), " µs per call")

    # Benchmark Q4_0
    t0 = time.perf_counter()
    for _ in range(iters):
        _mm_q4_0(Int(w_q4), x, o_q4, nr, nc)
    t1 = time.perf_counter()
    var q4_ms = (t1 - t0) * 1000.0 / Float64(iters)
    print("Q4_0: ", Int(q4_ms * 1000), " µs per call")

    # Compare speedup
    print("Speedup: ", f16_ms / q4_ms, "×")

    # Verify outputs are roughly similar
    var max_diff = Float32(0.0)
    for i in range(nr):
        var d = o_f16.load(i) - o_q4.load(i)
        if d < 0: d = -d
        if d > max_diff: max_diff = d
    print("Max diff (should be small for valid Q4_0): ", max_diff)
