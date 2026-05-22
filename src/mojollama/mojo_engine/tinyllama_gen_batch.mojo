# tinyllama_gen_batch.mojo — Batched TinyLlama with generic B
# Weights loaded once per block, shared across all B items.

from std import time
from std.math import sqrt, exp, cos, sin, pow
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime NE: Int = 2048
comptime NH: Int = 32
comptime NK: Int = 4
comptime HD: Int = 64
comptime NL: Int = 22
comptime NF: Int = 5632
comptime NV: Int = 32000
comptime MAX_SEQ: Int = 128
comptime W: Int = 8
comptime RPW: Int = 8
comptime B: Int = 4
comptime CS: Int = NL * NK * MAX_SEQ * HD

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

# ── Batched f16 matmul (explicit registers, no InlineArray spill) ──
@always_inline("nodebug")
def _mm_f16_batch(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
                  o: UnsafePointer[Float32, MutExternalOrigin],
                  nr: Int, nc: Int):
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(wi: Int) capturing:
        var rs = wi * RPW
        var re = rs + RPW
        if re > nr:
            re = nr
        for r in range(rs, re):
            var ro = r * nc
            comptime if B == 2:
                var acc0 = SIMD[DType.float32, W](0.0)
                var acc1 = SIMD[DType.float32, W](0.0)
                for blk in range(0, nc, 32):
                    comptime for grp in range(4):
                        var wv = w.load[width=W](ro + blk + grp * 8)
                        var wf = wv.cast[DType.float32]()
                        acc0 = wf.fma[FastMathFlag.FAST](x.load[width=W](0*nc + blk + grp*8), acc0)
                        acc1 = wf.fma[FastMathFlag.FAST](x.load[width=W](1*nc + blk + grp*8), acc1)
                o.store(0*nr + r, acc0.reduce_add())
                o.store(1*nr + r, acc1.reduce_add())
            elif B == 4:
                var acc0 = SIMD[DType.float32, W](0.0)
                var acc1 = SIMD[DType.float32, W](0.0)
                var acc2 = SIMD[DType.float32, W](0.0)
                var acc3 = SIMD[DType.float32, W](0.0)
                for blk in range(0, nc, 32):
                    comptime for grp in range(4):
                        var wv = w.load[width=W](ro + blk + grp * 8)
                        var wf = wv.cast[DType.float32]()
                        acc0 = wf.fma[FastMathFlag.FAST](x.load[width=W](0*nc + blk + grp*8), acc0)
                        acc1 = wf.fma[FastMathFlag.FAST](x.load[width=W](1*nc + blk + grp*8), acc1)
                        acc2 = wf.fma[FastMathFlag.FAST](x.load[width=W](2*nc + blk + grp*8), acc2)
                        acc3 = wf.fma[FastMathFlag.FAST](x.load[width=W](3*nc + blk + grp*8), acc3)
                o.store(0*nr + r, acc0.reduce_add())
                o.store(1*nr + r, acc1.reduce_add())
                o.store(2*nr + r, acc2.reduce_add())
                o.store(3*nr + r, acc3.reduce_add())
            elif B == 8:
                var acc0 = SIMD[DType.float32, W](0.0)
                var acc1 = SIMD[DType.float32, W](0.0)
                var acc2 = SIMD[DType.float32, W](0.0)
                var acc3 = SIMD[DType.float32, W](0.0)
                var acc4 = SIMD[DType.float32, W](0.0)
                var acc5 = SIMD[DType.float32, W](0.0)
                var acc6 = SIMD[DType.float32, W](0.0)
                var acc7 = SIMD[DType.float32, W](0.0)
                for blk in range(0, nc, 32):
                    comptime for grp in range(4):
                        var wv = w.load[width=W](ro + blk + grp * 8)
                        var wf = wv.cast[DType.float32]()
                        acc0 = wf.fma[FastMathFlag.FAST](x.load[width=W](0*nc + blk + grp*8), acc0)
                        acc1 = wf.fma[FastMathFlag.FAST](x.load[width=W](1*nc + blk + grp*8), acc1)
                        acc2 = wf.fma[FastMathFlag.FAST](x.load[width=W](2*nc + blk + grp*8), acc2)
                        acc3 = wf.fma[FastMathFlag.FAST](x.load[width=W](3*nc + blk + grp*8), acc3)
                        acc4 = wf.fma[FastMathFlag.FAST](x.load[width=W](4*nc + blk + grp*8), acc4)
                        acc5 = wf.fma[FastMathFlag.FAST](x.load[width=W](5*nc + blk + grp*8), acc5)
                        acc6 = wf.fma[FastMathFlag.FAST](x.load[width=W](6*nc + blk + grp*8), acc6)
                        acc7 = wf.fma[FastMathFlag.FAST](x.load[width=W](7*nc + blk + grp*8), acc7)
                o.store(0*nr + r, acc0.reduce_add())
                o.store(1*nr + r, acc1.reduce_add())
                o.store(2*nr + r, acc2.reduce_add())
                o.store(3*nr + r, acc3.reduce_add())
                o.store(4*nr + r, acc4.reduce_add())
                o.store(5*nr + r, acc5.reduce_add())
                o.store(6*nr + r, acc6.reduce_add())
                o.store(7*nr + r, acc7.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

@always_inline("nodebug")
def _mm_f16_2out_batch(wa: Int, wb: Int,
                       x: UnsafePointer[Float32, MutExternalOrigin],
                       oa: UnsafePointer[Float32, MutExternalOrigin],
                       ob: UnsafePointer[Float32, MutExternalOrigin],
                       nr: Int, nc: Int):
    var wa_ptr = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var wb_ptr = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wb))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(wi: Int) capturing:
        var rs = wi * RPW
        var re = rs + RPW
        if re > nr:
            re = nr
        for r in range(rs, re):
            var ro = r * nc
            comptime if B == 2:
                var a0 = SIMD[DType.float32, W](0.0)
                var a1 = SIMD[DType.float32, W](0.0)
                var b0 = SIMD[DType.float32, W](0.0)
                var b1 = SIMD[DType.float32, W](0.0)
                for blk in range(0, nc, 32):
                    comptime for grp in range(4):
                        var off = blk + grp * 8
                        var wfa = wa_ptr.load[width=W](ro + off).cast[DType.float32]()
                        var wfb = wb_ptr.load[width=W](ro + off).cast[DType.float32]()
                        a0 = wfa.fma[FastMathFlag.FAST](x.load[width=W](0*nc + off), a0)
                        a1 = wfa.fma[FastMathFlag.FAST](x.load[width=W](1*nc + off), a1)
                        b0 = wfb.fma[FastMathFlag.FAST](x.load[width=W](0*nc + off), b0)
                        b1 = wfb.fma[FastMathFlag.FAST](x.load[width=W](1*nc + off), b1)
                oa.store(0*nr + r, a0.reduce_add())
                oa.store(1*nr + r, a1.reduce_add())
                ob.store(0*nr + r, b0.reduce_add())
                ob.store(1*nr + r, b1.reduce_add())
            elif B == 4:
                var a0 = SIMD[DType.float32, W](0.0)
                var a1 = SIMD[DType.float32, W](0.0)
                var a2 = SIMD[DType.float32, W](0.0)
                var a3 = SIMD[DType.float32, W](0.0)
                var b0 = SIMD[DType.float32, W](0.0)
                var b1 = SIMD[DType.float32, W](0.0)
                var b2 = SIMD[DType.float32, W](0.0)
                var b3 = SIMD[DType.float32, W](0.0)
                for blk in range(0, nc, 32):
                    comptime for grp in range(4):
                        var off = blk + grp * 8
                        var wfa = wa_ptr.load[width=W](ro + off).cast[DType.float32]()
                        var wfb = wb_ptr.load[width=W](ro + off).cast[DType.float32]()
                        a0 = wfa.fma[FastMathFlag.FAST](x.load[width=W](0*nc + off), a0)
                        a1 = wfa.fma[FastMathFlag.FAST](x.load[width=W](1*nc + off), a1)
                        a2 = wfa.fma[FastMathFlag.FAST](x.load[width=W](2*nc + off), a2)
                        a3 = wfa.fma[FastMathFlag.FAST](x.load[width=W](3*nc + off), a3)
                        b0 = wfb.fma[FastMathFlag.FAST](x.load[width=W](0*nc + off), b0)
                        b1 = wfb.fma[FastMathFlag.FAST](x.load[width=W](1*nc + off), b1)
                        b2 = wfb.fma[FastMathFlag.FAST](x.load[width=W](2*nc + off), b2)
                        b3 = wfb.fma[FastMathFlag.FAST](x.load[width=W](3*nc + off), b3)
                oa.store(0*nr + r, a0.reduce_add())
                oa.store(1*nr + r, a1.reduce_add())
                oa.store(2*nr + r, a2.reduce_add())
                oa.store(3*nr + r, a3.reduce_add())
                ob.store(0*nr + r, b0.reduce_add())
                ob.store(1*nr + r, b1.reduce_add())
                ob.store(2*nr + r, b2.reduce_add())
                ob.store(3*nr + r, b3.reduce_add())
            elif B == 8:
                var a0 = SIMD[DType.float32, W](0.0)
                var a1 = SIMD[DType.float32, W](0.0)
                var a2 = SIMD[DType.float32, W](0.0)
                var a3 = SIMD[DType.float32, W](0.0)
                var a4 = SIMD[DType.float32, W](0.0)
                var a5 = SIMD[DType.float32, W](0.0)
                var a6 = SIMD[DType.float32, W](0.0)
                var a7 = SIMD[DType.float32, W](0.0)
                var b0 = SIMD[DType.float32, W](0.0)
                var b1 = SIMD[DType.float32, W](0.0)
                var b2 = SIMD[DType.float32, W](0.0)
                var b3 = SIMD[DType.float32, W](0.0)
                var b4 = SIMD[DType.float32, W](0.0)
                var b5 = SIMD[DType.float32, W](0.0)
                var b6 = SIMD[DType.float32, W](0.0)
                var b7 = SIMD[DType.float32, W](0.0)
                for blk in range(0, nc, 32):
                    comptime for grp in range(4):
                        var off = blk + grp * 8
                        var wfa = wa_ptr.load[width=W](ro + off).cast[DType.float32]()
                        var wfb = wb_ptr.load[width=W](ro + off).cast[DType.float32]()
                        a0 = wfa.fma[FastMathFlag.FAST](x.load[width=W](0*nc + off), a0)
                        a1 = wfa.fma[FastMathFlag.FAST](x.load[width=W](1*nc + off), a1)
                        a2 = wfa.fma[FastMathFlag.FAST](x.load[width=W](2*nc + off), a2)
                        a3 = wfa.fma[FastMathFlag.FAST](x.load[width=W](3*nc + off), a3)
                        a4 = wfa.fma[FastMathFlag.FAST](x.load[width=W](4*nc + off), a4)
                        a5 = wfa.fma[FastMathFlag.FAST](x.load[width=W](5*nc + off), a5)
                        a6 = wfa.fma[FastMathFlag.FAST](x.load[width=W](6*nc + off), a6)
                        a7 = wfa.fma[FastMathFlag.FAST](x.load[width=W](7*nc + off), a7)
                        b0 = wfb.fma[FastMathFlag.FAST](x.load[width=W](0*nc + off), b0)
                        b1 = wfb.fma[FastMathFlag.FAST](x.load[width=W](1*nc + off), b1)
                        b2 = wfb.fma[FastMathFlag.FAST](x.load[width=W](2*nc + off), b2)
                        b3 = wfb.fma[FastMathFlag.FAST](x.load[width=W](3*nc + off), b3)
                        b4 = wfb.fma[FastMathFlag.FAST](x.load[width=W](4*nc + off), b4)
                        b5 = wfb.fma[FastMathFlag.FAST](x.load[width=W](5*nc + off), b5)
                        b6 = wfb.fma[FastMathFlag.FAST](x.load[width=W](6*nc + off), b6)
                        b7 = wfb.fma[FastMathFlag.FAST](x.load[width=W](7*nc + off), b7)
                oa.store(0*nr + r, a0.reduce_add())
                oa.store(1*nr + r, a1.reduce_add())
                oa.store(2*nr + r, a2.reduce_add())
                oa.store(3*nr + r, a3.reduce_add())
                oa.store(4*nr + r, a4.reduce_add())
                oa.store(5*nr + r, a5.reduce_add())
                oa.store(6*nr + r, a6.reduce_add())
                oa.store(7*nr + r, a7.reduce_add())
                ob.store(0*nr + r, b0.reduce_add())
                ob.store(1*nr + r, b1.reduce_add())
                ob.store(2*nr + r, b2.reduce_add())
                ob.store(3*nr + r, b3.reduce_add())
                ob.store(4*nr + r, b4.reduce_add())
                ob.store(5*nr + r, b5.reduce_add())
                ob.store(6*nr + r, b6.reduce_add())
                ob.store(7*nr + r, b7.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── RMS Norm helper ──
@always_inline("nodebug")
def apply_rms_norm(hp: UnsafePointer[Float32, MutExternalOrigin],
                    bp: UnsafePointer[Float32, MutExternalOrigin],
                    wp: UnsafePointer[Float32, MutExternalOrigin],
                    n: Int):
    var ss = Float32(0.0)
    var i = 0
    while i + 8 <= n:
        var v = hp.load[width=8](i)
        ss += (v * v).reduce_add()
        i += 8
    while i < n:
        ss += hp.load(i) * hp.load(i)
        i += 1
    var inv = 1.0 / sqrt(ss / Float32(n) + 1e-6)
    var inv_v = SIMD[DType.float32, 8](inv)
    i = 0
    while i + 8 <= n:
        var v = hp.load[width=8](i)
        var w = wp.load[width=8](i)
        bp.store[width=8](i, v * inv_v * w)
        i += 8
    while i < n:
        bp.store(i, hp.load(i) * wp.load(i) * inv)
        i += 1

# ── f16 to f32 ──
def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1)
    var e = Int((h >> 10) & 0x1F)
    var m = Int(h & 0x3FF)
    if e == 0:
        var r = Float32(m) * 5.960464477539063e-8
        if s == 1:
            return -r
        return r
    if e == 31:
        return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    var tmp = alloc[UInt8](4)
    var uptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp))
    uptr.store(0, bits)
    var fptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp))
    return fptr.load(0)

# ── Load file ──
def load_file(dir_cptr: UnsafePointer[UInt8, MutExternalOrigin],
              name_cptr: UnsafePointer[UInt8, MutExternalOrigin]) -> Int64:
    var dlen = 0
    while dir_cptr.load(dlen) != 0:
        dlen += 1
    var nlen = 0
    while name_cptr.load(nlen) != 0:
        nlen += 1
    var path = alloc[UInt8](dlen + nlen + 1)
    for i in range(dlen):
        path.store(i, dir_cptr.load(i))
    for i in range(nlen):
        path.store(dlen + i, name_cptr.load(i))
    path.store(dlen + nlen, UInt8(0))
    var fd = _open(path, 0)
    if fd < 0:
        return -1
    var sz = _lseek(fd, 0, 2)
    _ = _lseek(fd, 0, 0)
    var buf = _alc(sz)
    if buf == 0:
        _ = _close(fd)
        return -1
    _ = _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
    _ = _close(fd)
    return buf

# ── Decode token ──
def decode_token(voc: UnsafePointer[UInt8, MutExternalOrigin],
                 voc_lens: UnsafePointer[Int32, MutExternalOrigin],
                 nv: Int, tok: Int) -> String:
    for i in range(nv):
        var vid = Int(voc_lens.load(nv + i))
        if vid == tok:
            var off = Int(voc_lens.load(2 * nv + i))
            var l = Int(voc_lens.load(i))
            var p = voc + off
            if l >= 3 and p.load(0) == 0xE2 and p.load(1) == 0x96 and p.load(2) == 0x81:
                p += 3
                l -= 3
            var result = String("")
            for j in range(l):
                result = result + chr(Int(p.load(j)))
            return result
    return String("")

# ── Main ──
def main():
    var t0 = time.perf_counter()

    var vocab_file = String("/tmp/vocab.bin")
    var vf = alloc[UInt8](vocab_file.byte_length() + 1)
    var vfp = vocab_file.unsafe_ptr()
    for i in range(vocab_file.byte_length()):
        vf.store(i, vfp.load(i))
    vf.store(vocab_file.byte_length(), UInt8(0))

    var vocab_fd = _open(vf, 0)
    var vocab_sz = _lseek(vocab_fd, 0, 2)
    _ = _lseek(vocab_fd, 0, 0)
    var vocab_buf = _alc(vocab_sz)
    _ = _read(vocab_fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(vocab_buf)), vocab_sz)
    _ = _close(vocab_fd)

    var vb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(vocab_buf))
    var nv = Int(vb.load(0)) | (Int(vb.load(1)) << 8) | (Int(vb.load(2)) << 16) | (Int(vb.load(3)) << 24)

    var voc_data = alloc[UInt8](Int(vocab_sz) - 4)
    var voc_meta = alloc[Int32](3 * nv)
    var off = 4
    var text_off = 0
    for i in range(nv):
        var plen = Int(vb.load(off)) | (Int(vb.load(off + 1)) << 8) | (Int(vb.load(off + 2)) << 16) | (Int(vb.load(off + 3)) << 24)
        off += 4
        for j in range(plen):
            voc_data.store(text_off + j, vb.load(off + j))
        off += plen
        var tid = Int(vb.load(off)) | (Int(vb.load(off + 1)) << 8) | (Int(vb.load(off + 2)) << 16) | (Int(vb.load(off + 3)) << 24)
        off += 4
        voc_meta.store(i, Int32(plen))
        voc_meta.store(nv + i, Int32(tid))
        voc_meta.store(2 * nv + i, Int32(text_off))
        text_off += plen

    var wdir = String("/tmp/weights_tl/")
    var dcp_len = wdir.byte_length()
    var dcp = alloc[UInt8](dcp_len + 1)
    var dsp = wdir.unsafe_ptr()
    for i in range(dcp_len):
        dcp.store(i, dsp.load(i))
    dcp.store(dcp_len, UInt8(0))

    var fn1 = String("token_embd_weight.bin")
    var fn1p = alloc[UInt8](fn1.byte_length() + 1)
    var f1s = fn1.unsafe_ptr()
    for i in range(fn1.byte_length()):
        fn1p.store(i, f1s.load(i))
    fn1p.store(fn1.byte_length(), UInt8(0))
    var w_emb = load_file(dcp, fn1p)

    var fn2 = String("output_norm_weight.bin")
    var fn2p = alloc[UInt8](fn2.byte_length() + 1)
    var f2s = fn2.unsafe_ptr()
    for i in range(fn2.byte_length()):
        fn2p.store(i, f2s.load(i))
    fn2p.store(fn2.byte_length(), UInt8(0))
    var w_on = load_file(dcp, fn2p)

    var fn3 = String("output_weight.bin")
    var fn3p = alloc[UInt8](fn3.byte_length() + 1)
    var f3s = fn3.unsafe_ptr()
    for i in range(fn3.byte_length()):
        fn3p.store(i, f3s.load(i))
    fn3p.store(fn3.byte_length(), UInt8(0))
    var w_lm = load_file(dcp, fn3p)

    var wl = alloc[Int64](NL * 9)
    var s0 = String("_attn_norm_weight.bin")
    var s1 = String("_ffn_norm_weight.bin")
    var s2 = String("_attn_q_weight.bin")
    var s3 = String("_attn_k_weight.bin")
    var s4 = String("_attn_v_weight.bin")
    var s5 = String("_attn_output_weight.bin")
    var s6 = String("_ffn_gate_weight.bin")
    var s7 = String("_ffn_up_weight.bin")
    var s8 = String("_ffn_down_weight.bin")
    var suffixes = [s0, s1, s2, s3, s4, s5, s6, s7, s8]

    for l in range(NL):
        var prefix = String("blk_") + String(l)
        var plen = prefix.byte_length()
        var pp = prefix.unsafe_ptr()
        for f in range(9):
            var sf = suffixes[f]
            var flen = plen + sf.byte_length()
            var fcp = alloc[UInt8](flen + 1)
            for j in range(plen):
                fcp.store(j, pp.load(j))
            var sfp = sf.unsafe_ptr()
            for j in range(sf.byte_length()):
                fcp.store(plen + j, sfp.load(j))
            fcp.store(flen, UInt8(0))
            var addr = load_file(dcp, fcp)
            wl.store(l * 9 + f, addr)

    var t_load = time.perf_counter()
    print("Load: ", Int((t_load - t0) * 1000), " ms")
    var prof_matmul = Float64(0.0)
    var prof_rope_attn = Float64(0.0)
    var prof_other = Float64(0.0)

    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var qp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NH * HD * 4))))
    var kp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NK * HD * 4))))
    var vbuf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NK * HD * 4))))
    var gp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NF * 4))))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NF * 4))))
    var dp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NV * 4))))

    var kc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * CS * 4))))
    var vc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * CS * 4))))
    var sc_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_SEQ * 4))))

    var batch_toks = alloc[Int32](B * MAX_SEQ)
    var prompt_toks = [1, 29871, 29906, 29974, 29906, 29922]
    var np = 6
    var max_gen = 50
    for bi in range(B):
        for i in range(np):
            batch_toks.store(bi * MAX_SEQ + i, Int32(prompt_toks[i]))

    var nt = alloc[Int32](B)
    for bi in range(B):
        nt.store(bi, Int32(np))

    print("B=", B, " max_gen=", max_gen)
    var t_gen = time.perf_counter()

    for pos in range(max_gen):
        var any_room = False
        for bi in range(B):
            if Int(nt.load(bi)) < MAX_SEQ:
                any_room = True
        if not any_room:
            break

        var emb = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(w_emb))
        for bi in range(B):
            var tok = Int(batch_toks.load(bi * MAX_SEQ + pos))
            for i in range(NE):
                hp.store(bi * NE + i, h2f(emb.load(tok * NE + i)))

        for l in range(NL):
            var lw = l * 9
            var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw)))
            var fnp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw + 1)))

            for bi in range(B):
                apply_rms_norm(hp + bi * NE, bp + bi * NE, anp, NE)

            _mm_f16_batch(Int(wl.load(lw + 2)), bp, qp, NH * HD, NE)
            _mm_f16_batch(Int(wl.load(lw + 3)), bp, kp, NK * HD, NE)
            _mm_f16_batch(Int(wl.load(lw + 4)), bp, vbuf, NK * HD, NE)

            # Precompute RoPE cos/sin for this position (avoids pow per head)
            var rope_cv = InlineArray[Float32, HD // 2](uninitialized=True)
            var rope_sv = InlineArray[Float32, HD // 2](uninitialized=True)
            for d2 in range(0, HD, 2):
                var freq = Float32(Float64(pos) / pow(10000.0, Float64(d2) / Float64(HD)))
                rope_cv[d2 // 2] = cos(freq)
                rope_sv[d2 // 2] = sin(freq)

            for bi in range(B):
                var qp_bi = qp + bi * NH * HD
                var kp_bi = kp + bi * NK * HD
                var vbuf_bi = vbuf + bi * NK * HD

                # RoPE Q (precomputed cos/sin)
                for h in range(NH):
                    for d2 in range(0, HD, 2):
                        var cv = rope_cv[d2 // 2]
                        var sv = rope_sv[d2 // 2]
                        var x0 = qp_bi.load(h * HD + d2)
                        var x1 = qp_bi.load(h * HD + d2 + 1)
                        qp_bi.store(h * HD + d2, x0 * cv - x1 * sv)
                        qp_bi.store(h * HD + d2 + 1, x0 * sv + x1 * cv)

                # RoPE K
                for h in range(NK):
                    for d2 in range(0, HD, 2):
                        var cv = rope_cv[d2 // 2]
                        var sv = rope_sv[d2 // 2]
                        var x0 = kp_bi.load(h * HD + d2)
                        var x1 = kp_bi.load(h * HD + d2 + 1)
                        kp_bi.store(h * HD + d2, x0 * cv - x1 * sv)
                        kp_bi.store(h * HD + d2 + 1, x0 * sv + x1 * cv)

                var cache_base = bi * CS + l * NK * MAX_SEQ * HD
                for h in range(NK):
                    for d in range(HD):
                        kc.store(cache_base + h * MAX_SEQ * HD + pos * HD + d, kp_bi.load(h * HD + d))
                        vc.store(cache_base + h * MAX_SEQ * HD + pos * HD + d, vbuf_bi.load(h * HD + d))

                # GQA attention with SIMD dot product + SIMD weighted sum
                var kr = NH // NK
                for hq in range(NH):
                    var hk = hq // kr
                    var lm = hk * MAX_SEQ * HD
                    var base = cache_base + lm
                    var sc = sc_buf
                    var smax = Float32(-1e9)
                    var qbase = hq * HD
                    for p in range(pos + 1):
                        var sv = SIMD[DType.float32, W](0.0)
                        var dd = 0
                        while dd + W <= HD:
                            var qv = qp_bi.load[width=W](qbase + dd)
                            var kv = kc.load[width=W](base + p * HD + dd)
                            sv = sv + qv * kv
                            dd += W
                        var s = sv.reduce_add() / sqrt(Float32(HD))
                        sc.store(p, s)
                        if s > smax:
                            smax = s
                    var ssum = Float32(0.0)
                    for p in range(pos + 1):
                        var es = exp(sc.load(p) - smax)
                        sc.store(p, es)
                        ssum += es
                    var dd = 0
                    while dd + W <= HD:
                        var ov = SIMD[DType.float32, W](0.0)
                        for p in range(pos + 1):
                            var vv = vc.load[width=W](base + p * HD + dd)
                            var w = sc.load(p) / ssum
                            ov = ov + vv * w
                        qp_bi.store[width=W](qbase + dd, ov)
                        dd += W
                    while dd < HD:
                        var o = Float32(0.0)
                        for p in range(pos + 1):
                            o += vc.load(base + p * HD + dd) * (sc.load(p) / ssum)
                        qp_bi.store(qbase + dd, o)
                        dd += 1

            _mm_f16_batch(Int(wl.load(lw + 5)), qp, bp, NE, NH * HD)
            for bi in range(B):
                var hp_bi = hp + bi * NE
                var bp_bi = bp + bi * NE
                for i in range(NE):
                    hp_bi.store(i, hp_bi.load(i) + bp_bi.load(i))

            for bi in range(B):
                apply_rms_norm(hp + bi * NE, bp + bi * NE, fnp, NE)

            _mm_f16_2out_batch(Int(wl.load(lw + 6)), Int(wl.load(lw + 7)), bp, gp, up, NF, NE)

            for bi in range(B):
                var gp_bi = gp + bi * NF
                var up_bi = up + bi * NF
                for i in range(NF):
                    var gv = gp_bi.load(i)
                    if gv < -80.0:
                        gv = -80.0
                    if gv > 80.0:
                        gv = 80.0
                    gp_bi.store(i, (gv / (1.0 + exp(-gv))) * up_bi.load(i))

            _mm_f16_batch(Int(wl.load(lw + 8)), gp, dp, NE, NF)
            for bi in range(B):
                var hp_bi = hp + bi * NE
                var dp_bi = dp + bi * NE
                for i in range(NE):
                    hp_bi.store(i, hp_bi.load(i) + dp_bi.load(i))

        for bi in range(B):
            var hp_bi = hp + bi * NE
            var bp_bi = bp + bi * NE
            var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w_on))
            apply_rms_norm(hp_bi, bp_bi, onp, NE)

        _mm_f16_batch(Int(w_lm), bp, lp, NV, NE)

        for bi in range(B):
            var lp_bi = lp + bi * NV
            var best = 0
            var bv = lp_bi.load(0)
            for i in range(1, NV):
                var v = lp_bi.load(i)
                if v > bv:
                    bv = v
                    best = i
            var nti = Int(nt.load(bi))
            if nti < MAX_SEQ:
                batch_toks.store(bi * MAX_SEQ + nti, Int32(best))
                nt.store(bi, Int32(nti + 1))
            if best != 2:
                var out_text = decode_token(voc_data, voc_meta, nv, best)
                if bi == 0:
                    print(out_text, end="")

    print()
    var t_end = time.perf_counter()
    var gen_ms = (t_end - t_gen) * 1000.0
    var total_gen = 0
    for bi in range(B):
        total_gen += Int(nt.load(bi)) - np
    print("B=", B, " cum_toks=", total_gen, " time=", Int(gen_ms), " ms (", Float64(total_gen) / (gen_ms / 1000.0), " tok/s)")

    print("=== Correctness check ===")
    for bi in range(B):
        var nti = Int(nt.load(bi))
        var line = String("item ") + String(bi) + String(": ")
        for pi in range(np, nti):
            line = line + String(Int(batch_toks.load(bi * MAX_SEQ + pi))) + String(" ")
        print(line)
    # Check all items match item 0
    var n0 = Int(nt.load(0))
    var all_match = True
    for bi in range(1, B):
        var nbi = Int(nt.load(bi))
        if nbi != n0:
            all_match = False
        else:
            for pi in range(np, n0):
                if batch_toks.load(bi * MAX_SEQ + pi) != batch_toks.load(0 * MAX_SEQ + pi):
                    all_match = False
    if all_match:
        print("All B items match (output is valid)")
    else:
        print("ERROR: output divergence detected!")
