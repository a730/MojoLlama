# MojoLlama Quant Kernels — Bench + Correctness Test
# WHAT:  Verify all quant matmul types produce non-zero output.
# WHY:   C library had type 1 (f16) as no-op → all zeros. Fix in Mojo kernels.
# WHEN:  May 2026 — benchmark + correctness verification.
from std import time
from std.algorithm.backend.cpu.parallelize import parallelize

# ── C bridge for pointer conversion ──
@extern("addr_to_f32")
def _af32(a: Int64) abi("C") -> UnsafePointer[Float32, MutExternalOrigin]: ...
@extern("addr_to_u8")
def _au8(a: Int64) abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...
@extern("mojo_alloc")
def _alc(sz: Int64) abi("C") -> Int64: ...
@extern("mojo_free")
def _free(p: Int64) abi("C") -> None: ...
@extern("load_weight")
def _lw(i: Int64) abi("C") -> Int64: ...
@extern("free_weight")
def _fw(p: Int64) abi("C") -> None: ...
@extern("mojo_set_threads")
def _st(n: Int) abi("C") -> None: ...

# ── Quant constants ──
comptime Q5_0_BS: Int = 22
comptime Q8_0_BS: Int = 34
comptime Q4_K_BS: Int = 144
comptime MXFP4_BS: Int = 17
comptime W: Int = 8

def f16_to_f32(h: UInt16) -> Float32:
    var s = UInt32(h >> 15); var e = UInt32((h >> 10) & 0x1F)
    var m = UInt32(h & 0x3FF)
    if e == 0:
        if m == 0: return 0.0
        return Float32(Float64(m) * 5.960464477539063e-8)
    if e == 31: return 0.0
    var r = Float32(m | 0x400)
    var ei = Int(e) - 25
    if ei >= 0:
        for _ in range(ei): r *= 2.0
    else:
        for _ in range(-ei): r *= 0.5
    return -r if s != 0 else r

# ── Q5_0 kernel ──
def q5_0_matmul(w: UnsafePointer[UInt8, MutExternalOrigin],
                 x: UnsafePointer[Float32, MutExternalOrigin],
                 o: UnsafePointer[Float32, MutExternalOrigin],
                 nr: Int, nc: Int):
    var bpr = nc // 32
    def worker(r: Int) capturing -> None:
        var ro = r * bpr * Q5_0_BS
        var acc = SIMD[DType.float32, W](0.0)
        for blk in range(bpr):
            var bo = ro + blk * Q5_0_BS
            var lo = UInt16(w.load(bo)); var hi = UInt16(w.load(bo + 1))
            var d = f16_to_f32(lo | (hi << 8))
            for ch in range(4):
                var vals = SIMD[DType.float32, W](0.0)
                for k in range(W):
                    var idx = ch * W + k
                    var p = w.load(bo + 6 + idx // 2)
                    var qh = w.load(bo + 2 + idx // 4)
                    var hs = 2 * (idx % 4)
                    var hb_u8 = (UInt8(qh) >> UInt8(hs)) & UInt8(1)
                    var nib = Int32(p >> 4) if idx % 2 == 1 else Int32(p & 0x0F)
                    if nib > 7: nib -= 16
                    var v = nib + Int32(hb_u8) * 16
                    if v > 15: v -= 32
                    vals[k] = Float32(v)
                var xb = blk * 32 + ch * W
                var xv = x.load[width=W](xb)
                acc = acc + vals * xv * d
        o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=nr)

# ── f16 kernel (the critical fix) ──
def f16_matmul(w: UnsafePointer[UInt8, MutExternalOrigin],
                x: UnsafePointer[Float32, MutExternalOrigin],
                o: UnsafePointer[Float32, MutExternalOrigin],
                nr: Int, nc: Int):
    def worker(r: Int) capturing -> None:
        var acc = SIMD[DType.float32, W](0.0)
        for bc in range(0, nc, W):
            var vals = SIMD[DType.float32, W](0.0)
            for k in range(W):
                var wi = r * nc + bc + k
                var lo = UInt16(w.load(wi * 2)); var hi = UInt16(w.load(wi * 2 + 1))
                vals[k] = f16_to_f32(lo | (hi << 8))
            var xv = x.load[width=W](bc)
            acc = acc + vals * xv
        o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=nr)

def main():
    _st(64)
    print("Mojo Quant Kernels — Bench + Correctness")
    print("=========================================")
    var ni = 2880; var nq = 4096  # Q dims
    var t0 = time.perf_counter()

    # Load Q weight (layer 0, Q5_0)
    var qw = _lw(Int64(4 + 12))  # layer_0 q_w at index 4+12
    var kw = _lw(Int64(4 + 7))   # layer_0 k_w
    var vw = _lw(Int64(4 + 17))  # layer_0 v_w
    var ow = _lw(Int64(4 + 10))  # layer_0 o_w
    var emb = _lw(Int64(0))      # embedding (f16 LM head)
    var xb = _alc(Int64(ni * 4))
    var qb = _alc(Int64(nq * 4))
    var ok = _alc(Int64(ni * 4))
    var lg = _alc(Int64(201088 * 4))

    var xp = _af32(xb); var qp = _af32(qb); var op = _af32(ok); var lp = _af32(lg)
    var wp = _au8(qw); var kp8 = _au8(kw); var vp8 = _au8(vw)
    var op8 = _au8(ow); var ep8 = _au8(emb)

    # Fill input with known values
    for i in range(ni): xp.store(i, Float32(Float64((i * 3) % 100 - 50) / 50.0))

    # Test Q5_0 matmul (Q projection)
    t0 = time.perf_counter()
    q5_0_matmul(wp, xp, qp, nq, ni)
    var t1 = time.perf_counter()
    var ms_q5 = (t1 - t0) * 1000.0
    var non_zero = 1
    for i in range(min(nq, 64)):
        if qp.load(i) != 0.0: non_zero = 0; break
    # Actually check for non-zero: count how many are non-zero
    var nz_count = 0
    for i in range(min(nq, 64)):
        if qp.load(i) != 0.0: nz_count += 1
    print("Q5_0 matmul (", nq, "x", ni, "):", Int(ms_q5 * 1000), "us, non-zero:", nz_count, "/64")
    print("  Q[0]:", qp.load(0), "Q[1]:", qp.load(1))
    if nz_count > 0: print("  ✓ Q5_0 produces non-zero output")
    else: print("  ✗ Q5_0 all zeros!")

    # Test f16 matmul with real embedding weight (LM head)
    print("Embedding LM head size: 201088 x", ni)
    t0 = time.perf_counter()
    f16_matmul(ep8, xp, lp, 201088, ni)
    t1 = time.perf_counter()
    ms_q5 = (t1 - t0) * 1000.0
    print("f16 LM head matmul: ", Int(ms_q5), "ms")
    print("  Logit[0]:", lp.load(0), "Logit[1]:", lp.load(1))
    var lz = 1
    for i in range(min(100, 201088)):
        if lp.load(i) != 0.0: lz = 0; break
    if lz == 0: print("  ✓ f16 LM head produces non-zero output!")
    else: print("  ✗ f16 LM head all zeros")

    # Cleanup
    _fw(qw); _fw(kw); _fw(vw); _fw(ow); _fw(emb)
    _free(xb); _free(qb); _free(ok); _free(lg)
