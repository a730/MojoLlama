# tinyllama_gen_q8.mojo — TinyLlama with Q8_0 quantized weights + batch B=4
# Q8_0: 34 bytes per 32 weights (1.88× compression over f16)
# No nibble extraction — just load int8, cast float32, scale, FMA.

from std import time
from std.sys import argv
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
comptime QK: Int = 32  # Q8_0 block size
comptime QB: Int = 34  # Q8_0 bytes per block (2 scale + 32 quants)

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

# ── f16 → Q8_0 converter ──
def convert_to_q8_0(f16_addr: Int, nr: Int, nc: Int):
    var src = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(f16_addr))
    var dst = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(f16_addr))
    var tmp = alloc[UInt16](nc)
    for r in range(nr):
        for i in range(nc): tmp.store(i, src.load(r * nc + i))
        for blk in range(nc // QK):
            var maxv = Float32(0.0)
            for i in range(QK):
                var f = h2f(tmp.load(blk * QK + i))
                var fa = f
                if fa < 0: fa = -fa
                if fa > maxv: maxv = fa
            var scale = maxv / 127.0
            if scale < 1e-10:
                scale = 1.0
            var scale_u16 = f32_to_f16_bits(scale)
            var qoff = r * (nc // QK) * QB + blk * QB
            dst.store(qoff + 0, UInt8(Int(scale_u16) & 0xFF))
            dst.store(qoff + 1, UInt8((Int(scale_u16) >> 8) & 0xFF))
            for i in range(QK):
                var f = h2f(tmp.load(blk * QK + i))
                var qv = Int(f / scale + 0.5)
                if qv > 127: qv = 127
                if qv < -128: qv = -128
                dst.store(qoff + 2 + i, UInt8(qv + 128))

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
    var tmp = alloc[UInt8](4)
    var uptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp))
    uptr.store(0, bits)
    var fptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp))
    var result = fptr.load(0)
    return result

def f32_to_f16_bits(f: Float32) -> UInt16:
    var tmp = alloc[UInt8](4)
    var fptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp))
    fptr.store(0, f)
    var uptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp))
    var bits = uptr.load(0)
    var s = Int((bits >> 31) & 1)
    var e = Int((bits >> 23) & 0xFF)
    var m = Int(bits & 0x7FFFFF)
    if e == 0: return UInt16(s << 15)
    if e >= 0x8F: return UInt16((s << 15) | (31 << 10))
    var f16_exp = e - 127 + 15
    if f16_exp <= 0:
        if f16_exp <= -10: return UInt16(s << 15)
        var sub_m = (m | 0x800000) >> (1 - f16_exp)
        return UInt16((s << 15) | (sub_m >> 13))
    if f16_exp >= 31: return UInt16((s << 15) | (31 << 10))
    return UInt16((s << 15) | (f16_exp << 10) | (m >> 13))

# ── Batched Q8_0 matmul ──
@always_inline("nodebug")
def _mm_q8_batch(q8addr: Int, x: UnsafePointer[Float32, MutExternalOrigin],
                  o: UnsafePointer[Float32, MutExternalOrigin],
                  nr: Int, nc: Int, nw: Int = 32):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(q8addr))
    var nb = (nr + RPW - 1) // RPW
    def wk(wi: Int) capturing:
        var rs = wi * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * (nc // QK) * QB
            comptime if B == 2:
                var acc0 = SIMD[DType.float32, W](0.0)
                var acc1 = SIMD[DType.float32, W](0.0)
                var col = 0
                while col < nc:
                    var b_off = ro + (col // QK) * QB
                    var lo = Int(q.load(b_off + 0))
                    var hi = Int(q.load(b_off + 1))
                    var scale = h2f(UInt16(lo | (hi << 8)))
                    var sv = SIMD[DType.float32, W](scale)
                    comptime for grp in range(4):
                        var wo = q.load[width=8](b_off + 2 + grp * 8)
                        var wf = wo.cast[DType.int8]().cast[DType.float32]() * sv
                        acc0 = wf.fma[FastMathFlag.FAST](x.load[width=W](0*nc + col + grp*8), acc0)
                        acc1 = wf.fma[FastMathFlag.FAST](x.load[width=W](1*nc + col + grp*8), acc1)
                    col += QK
                o.store(0*nr + r, acc0.reduce_add())
                o.store(1*nr + r, acc1.reduce_add())
            elif B == 4:
                var acc0 = SIMD[DType.float32, W](0.0)
                var acc1 = SIMD[DType.float32, W](0.0)
                var acc2 = SIMD[DType.float32, W](0.0)
                var acc3 = SIMD[DType.float32, W](0.0)
                var col = 0
                while col < nc:
                    var b_off = ro + (col // QK) * QB
                    var lo = Int(q.load(b_off + 0))
                    var hi = Int(q.load(b_off + 1))
                    var scale = h2f(UInt16(lo | (hi << 8)))
                    var sv = SIMD[DType.float32, W](scale)
                    comptime for grp in range(4):
                        var wo = q.load[width=8](b_off + 2 + grp * 8)
                        var wf = wo.cast[DType.int8]().cast[DType.float32]() * sv
                        acc0 = wf.fma[FastMathFlag.FAST](x.load[width=W](0*nc + col + grp*8), acc0)
                        acc1 = wf.fma[FastMathFlag.FAST](x.load[width=W](1*nc + col + grp*8), acc1)
                        acc2 = wf.fma[FastMathFlag.FAST](x.load[width=W](2*nc + col + grp*8), acc2)
                        acc3 = wf.fma[FastMathFlag.FAST](x.load[width=W](3*nc + col + grp*8), acc3)
                    col += QK
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
                var col = 0
                while col < nc:
                    var b_off = ro + (col // QK) * QB
                    var lo = Int(q.load(b_off + 0))
                    var hi = Int(q.load(b_off + 1))
                    var scale = h2f(UInt16(lo | (hi << 8)))
                    var sv = SIMD[DType.float32, W](scale)
                    comptime for grp in range(4):
                        var wo = q.load[width=8](b_off + 2 + grp * 8)
                        var wf = wo.cast[DType.int8]().cast[DType.float32]() * sv
                        acc0 = wf.fma[FastMathFlag.FAST](x.load[width=W](0*nc + col + grp*8), acc0)
                        acc1 = wf.fma[FastMathFlag.FAST](x.load[width=W](1*nc + col + grp*8), acc1)
                        acc2 = wf.fma[FastMathFlag.FAST](x.load[width=W](2*nc + col + grp*8), acc2)
                        acc3 = wf.fma[FastMathFlag.FAST](x.load[width=W](3*nc + col + grp*8), acc3)
                        acc4 = wf.fma[FastMathFlag.FAST](x.load[width=W](4*nc + col + grp*8), acc4)
                        acc5 = wf.fma[FastMathFlag.FAST](x.load[width=W](5*nc + col + grp*8), acc5)
                        acc6 = wf.fma[FastMathFlag.FAST](x.load[width=W](6*nc + col + grp*8), acc6)
                        acc7 = wf.fma[FastMathFlag.FAST](x.load[width=W](7*nc + col + grp*8), acc7)
                    col += QK
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
def _mm_q8_2out_batch(q8a: Int, q8b: Int,
                       x: UnsafePointer[Float32, MutExternalOrigin],
                       oa: UnsafePointer[Float32, MutExternalOrigin],
                       ob: UnsafePointer[Float32, MutExternalOrigin],
                       nr: Int, nc: Int, nw: Int = 32):
    var qa = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(q8a))
    var qb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(q8b))
    var nb = (nr + RPW - 1) // RPW
    def wk(wi: Int) capturing:
        var rs = wi * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ra = r * (nc // QK) * QB
            var rb = r * (nc // QK) * QB
            comptime if B == 2:
                var a0 = SIMD[DType.float32, W](0.0)
                var a1 = SIMD[DType.float32, W](0.0)
                var b0 = SIMD[DType.float32, W](0.0)
                var b1 = SIMD[DType.float32, W](0.0)
                var col = 0
                while col < nc:
                    var bo = (col // QK) * QB
                    var lo_a = Int(qa.load(ra + bo + 0)); var hi_a = Int(qa.load(ra + bo + 1))
                    var lo_b = Int(qb.load(rb + bo + 0)); var hi_b = Int(qb.load(rb + bo + 1))
                    var sa = SIMD[DType.float32, W](h2f(UInt16(lo_a | (hi_a << 8))))
                    var sb = SIMD[DType.float32, W](h2f(UInt16(lo_b | (hi_b << 8))))
                    comptime for grp in range(4):
                        var wa = qa.load[width=8](ra+bo+2+grp*8).cast[DType.int8]().cast[DType.float32]() * sa
                        var wb = qb.load[width=8](rb+bo+2+grp*8).cast[DType.int8]().cast[DType.float32]() * sb
                        var xv = x.load[width=W](0*nc + col + grp*8)
                        a0 = wa.fma[FastMathFlag.FAST](xv, a0)
                        b0 = wb.fma[FastMathFlag.FAST](xv, b0)
                        xv = x.load[width=W](1*nc + col + grp*8)
                        a1 = wa.fma[FastMathFlag.FAST](xv, a1)
                        b1 = wb.fma[FastMathFlag.FAST](xv, b1)
                    col += QK
                oa.store(0*nr + r, a0.reduce_add()); oa.store(1*nr + r, a1.reduce_add())
                ob.store(0*nr + r, b0.reduce_add()); ob.store(1*nr + r, b1.reduce_add())
            elif B == 4:
                var a0 = SIMD[DType.float32, W](0.0)
                var a1 = SIMD[DType.float32, W](0.0)
                var a2 = SIMD[DType.float32, W](0.0)
                var a3 = SIMD[DType.float32, W](0.0)
                var b0 = SIMD[DType.float32, W](0.0)
                var b1 = SIMD[DType.float32, W](0.0)
                var b2 = SIMD[DType.float32, W](0.0)
                var b3 = SIMD[DType.float32, W](0.0)
                var col = 0
                while col < nc:
                    var bo = (col // QK) * QB
                    var lo_a = Int(qa.load(ra + bo + 0)); var hi_a = Int(qa.load(ra + bo + 1))
                    var lo_b = Int(qb.load(rb + bo + 0)); var hi_b = Int(qb.load(rb + bo + 1))
                    var sa = SIMD[DType.float32, W](h2f(UInt16(lo_a | (hi_a << 8))))
                    var sb = SIMD[DType.float32, W](h2f(UInt16(lo_b | (hi_b << 8))))
                    comptime for grp in range(4):
                        var wa = qa.load[width=8](ra+bo+2+grp*8).cast[DType.int8]().cast[DType.float32]() * sa
                        var wb = qb.load[width=8](rb+bo+2+grp*8).cast[DType.int8]().cast[DType.float32]() * sb
                        var xv = x.load[width=W](0*nc + col + grp*8); a0 = wa.fma[FastMathFlag.FAST](xv, a0); b0 = wb.fma[FastMathFlag.FAST](xv, b0)
                        xv = x.load[width=W](1*nc + col + grp*8); a1 = wa.fma[FastMathFlag.FAST](xv, a1); b1 = wb.fma[FastMathFlag.FAST](xv, b1)
                        xv = x.load[width=W](2*nc + col + grp*8); a2 = wa.fma[FastMathFlag.FAST](xv, a2); b2 = wb.fma[FastMathFlag.FAST](xv, b2)
                        xv = x.load[width=W](3*nc + col + grp*8); a3 = wa.fma[FastMathFlag.FAST](xv, a3); b3 = wb.fma[FastMathFlag.FAST](xv, b3)
                    col += QK
                oa.store(0*nr + r, a0.reduce_add()); oa.store(1*nr + r, a1.reduce_add())
                oa.store(2*nr + r, a2.reduce_add()); oa.store(3*nr + r, a3.reduce_add())
                ob.store(0*nr + r, b0.reduce_add()); ob.store(1*nr + r, b1.reduce_add())
                ob.store(2*nr + r, b2.reduce_add()); ob.store(3*nr + r, b3.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── RMS Norm helper ──
@always_inline("nodebug")
def apply_rms_norm(hp: UnsafePointer[Float32, MutExternalOrigin],
                    bp: UnsafePointer[Float32, MutExternalOrigin],
                    wp: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    var ss = Float32(0.0); var i = 0
    while i + 8 <= n:
        var v = hp.load[width=8](i); ss += (v * v).reduce_add(); i += 8
    while i < n: ss += hp.load(i) * hp.load(i); i += 1
    var inv = 1.0 / sqrt(ss / Float32(n) + 1e-6)
    var inv_v = SIMD[DType.float32, 8](inv); i = 0
    while i + 8 <= n:
        var v = hp.load[width=8](i); var w = wp.load[width=8](i)
        bp.store[width=8](i, v * inv_v * w); i += 8
    while i < n: bp.store(i, hp.load(i) * wp.load(i) * inv); i += 1

# ── Load file ──
def load_file(dir_cptr: UnsafePointer[UInt8, MutExternalOrigin],
              name_cptr: UnsafePointer[UInt8, MutExternalOrigin]) -> Int64:
    var dlen = 0; var nlen = 0
    while dir_cptr.load(dlen) != 0: dlen += 1
    while name_cptr.load(nlen) != 0: nlen += 1
    var path = alloc[UInt8](dlen + nlen + 1)
    for i in range(dlen): path.store(i, dir_cptr.load(i))
    for i in range(nlen): path.store(dlen + i, name_cptr.load(i))
    path.store(dlen + nlen, UInt8(0))
    var fd = _open(path, 0)
    if fd < 0: return -1
    var sz = _lseek(fd, 0, 2)
    _ = _lseek(fd, 0, 0)
    var buf = _alc(sz)
    if buf == 0: _ = _close(fd); return -1
    _ = _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
    _ = _close(fd); return buf

def decode_token(voc: UnsafePointer[UInt8, MutExternalOrigin],
                 voc_lens: UnsafePointer[Int32, MutExternalOrigin],
                 nv: Int, tok: Int) -> String:
    for i in range(nv):
        var vid = Int(voc_lens.load(nv + i))
        if vid == tok:
            var off = Int(voc_lens.load(2 * nv + i)); var l = Int(voc_lens.load(i))
            var p = voc + off
            if l >= 3 and p.load(0) == 0xE2 and p.load(1) == 0x96 and p.load(2) == 0x81: p += 3; l -= 3
            var result = String("")
            for j in range(l): result = result + chr(Int(p.load(j)))
            return result
    return String("")

def main() raises:
    var t0 = time.perf_counter()
    var args = argv()
    var nw = 32
    if len(args) > 1:
        nw = Int(String(args[1]))
    var vocab_file = String("/tmp/vocab.bin")
    var vf = alloc[UInt8](vocab_file.byte_length() + 1)
    var vfp = vocab_file.unsafe_ptr()
    for i in range(vocab_file.byte_length()): vf.store(i, vfp.load(i))
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
        var plen = Int(vb.load(off)) | (Int(vb.load(off+1))<<8) | (Int(vb.load(off+2))<<16) | (Int(vb.load(off+3))<<24)
        off += 4
        for j in range(plen):
            voc_data.store(text_off+j, vb.load(off+j))
        off += plen
        var tid = Int(vb.load(off)) | (Int(vb.load(off+1))<<8) | (Int(vb.load(off+2))<<16) | (Int(vb.load(off+3))<<24)
        off += 4
        voc_meta.store(i, Int32(plen)); voc_meta.store(nv+i, Int32(tid)); voc_meta.store(2*nv+i, Int32(text_off))
        text_off += plen

    var wdir = String("/tmp/weights_tl/")
    var dcp = alloc[UInt8](wdir.byte_length() + 1)
    var dsp = wdir.unsafe_ptr()
    for i in range(wdir.byte_length()): dcp.store(i, dsp.load(i))
    dcp.store(wdir.byte_length(), UInt8(0))

    var fn1 = String("token_embd_weight.bin")
    var fn1p = alloc[UInt8](fn1.byte_length()+1)
    var f1s = fn1.unsafe_ptr()
    for i in range(fn1.byte_length()): fn1p.store(i, f1s.load(i))
    fn1p.store(fn1.byte_length(), UInt8(0))
    var w_emb = load_file(dcp, fn1p)

    var fn2 = String("output_norm_weight.bin")
    var fn2p = alloc[UInt8](fn2.byte_length()+1)
    var f2s = fn2.unsafe_ptr()
    for i in range(fn2.byte_length()): fn2p.store(i, f2s.load(i))
    fn2p.store(fn2.byte_length(), UInt8(0))
    var w_on = load_file(dcp, fn2p)

    var fn3 = String("output_weight.bin")
    var fn3p = alloc[UInt8](fn3.byte_length()+1)
    var f3s = fn3.unsafe_ptr()
    for i in range(fn3.byte_length()): fn3p.store(i, f3s.load(i))
    fn3p.store(fn3.byte_length(), UInt8(0))
    var w_lm = load_file(dcp, fn3p)

    var wl = alloc[Int64](NL * 9)
    var s0 = String("_attn_norm_weight.bin"); var s1 = String("_ffn_norm_weight.bin")
    var s2 = String("_attn_q_weight.bin"); var s3 = String("_attn_k_weight.bin")
    var s4 = String("_attn_v_weight.bin"); var s5 = String("_attn_output_weight.bin")
    var s6 = String("_ffn_gate_weight.bin"); var s7 = String("_ffn_up_weight.bin")
    var s8 = String("_ffn_down_weight.bin")
    var suffixes = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
    for l in range(NL):
        var prefix = String("blk_") + String(l)
        var pp = prefix.unsafe_ptr()
        for f in range(9):
            var sf = suffixes[f]; var flen = prefix.byte_length() + sf.byte_length()
            var fcp = alloc[UInt8](flen + 1)
            for j in range(prefix.byte_length()): fcp.store(j, pp.load(j))
            var sfp = sf.unsafe_ptr()
            for j in range(sf.byte_length()): fcp.store(prefix.byte_length()+j, sfp.load(j))
            fcp.store(flen, UInt8(0))
            var addr = load_file(dcp, fcp)
            wl.store(l * 9 + f, addr)

    var t_load = time.perf_counter()
    print("Load: ", Int((t_load - t0) * 1000), " ms")

    # Convert matmul weights from f16 to Q8_0 (in-place)
    var t_conv = time.perf_counter()
    convert_to_q8_0(Int(w_lm), NV, NE)                    # LM head
    for l in range(NL):
        convert_to_q8_0(Int(wl.load(l*9+2)), NH*HD, NE)  # Q
        convert_to_q8_0(Int(wl.load(l*9+3)), NK*HD, NE)  # K
        convert_to_q8_0(Int(wl.load(l*9+4)), NK*HD, NE)  # V
        convert_to_q8_0(Int(wl.load(l*9+5)), NE, NH*HD)   # O
        convert_to_q8_0(Int(wl.load(l*9+6)), NF, NE)      # gate
        convert_to_q8_0(Int(wl.load(l*9+7)), NF, NE)      # up
        convert_to_q8_0(Int(wl.load(l*9+8)), NE, NF)      # down
    var t_conv_end = time.perf_counter()
    print("Q8_0 convert: ", Int((t_conv_end - t_conv) * 1000), " ms")

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
    var np = 6; var max_gen = 128
    for bi in range(B):
        for i in range(np): batch_toks.store(bi * MAX_SEQ + i, Int32(prompt_toks[i]))
    var nt = alloc[Int32](B)
    for bi in range(B): nt.store(bi, Int32(np))
    print("B=", B, " max_gen=", max_gen, " (Q8_0 weights)")

    var t_gen = time.perf_counter()
    for pos in range(max_gen):
        var any_room = False
        for bi in range(B):
            if Int(nt.load(bi)) < MAX_SEQ: any_room = True
        if not any_room: break
        var emb = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(w_emb))
        for bi in range(B):
            var tok = Int(batch_toks.load(bi * MAX_SEQ + pos))
            for i in range(NE): hp.store(bi * NE + i, h2f(emb.load(tok * NE + i)))
        for l in range(NL):
            var lw = l * 9
            var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw)))
            var fnp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw + 1)))
            for bi in range(B): apply_rms_norm(hp + bi * NE, bp + bi * NE, anp, NE)

            _mm_q8_batch(Int(wl.load(lw+2)), bp, qp, NH*HD, NE, nw)
            _mm_q8_batch(Int(wl.load(lw+3)), bp, kp, NK*HD, NE, nw)
            _mm_q8_batch(Int(wl.load(lw+4)), bp, vbuf, NK*HD, NE, nw)

            for bi in range(B):
                var qp_bi = qp + bi * NH * HD
                var kp_bi = kp + bi * NK * HD
                var vbuf_bi = vbuf + bi * NK * HD
                for d2 in range(0, HD, 2):
                    var freq = Float32(Float64(pos) / pow(10000.0, Float64(d2) / Float64(HD)))
                    var cv = cos(freq); var sv = sin(freq)
                    for h in range(NH):
                        var x0 = qp_bi.load(h * HD + d2); var x1 = qp_bi.load(h * HD + d2 + 1)
                        qp_bi.store(h*HD+d2, x0*cv - x1*sv); qp_bi.store(h*HD+d2+1, x0*sv + x1*cv)
                    for h in range(NK):
                        var x0 = kp_bi.load(h * HD + d2); var x1 = kp_bi.load(h * HD + d2 + 1)
                        kp_bi.store(h*HD+d2, x0*cv - x1*sv); kp_bi.store(h*HD+d2+1, x0*sv + x1*cv)
                var cache_base = bi * CS + l * NK * MAX_SEQ * HD
                for h in range(NK):
                    for d in range(HD):
                        kc.store(cache_base + h*MAX_SEQ*HD + pos*HD + d, kp_bi.load(h*HD + d))
                        vc.store(cache_base + h*MAX_SEQ*HD + pos*HD + d, vbuf_bi.load(h*HD + d))
                var kr = NH // NK
                for hq in range(NH):
                    var hk = hq // kr; var lm = hk * MAX_SEQ * HD; var base = cache_base + lm
                    var sc = sc_buf; var smax = Float32(-1e9); var qbase = hq * HD
                    for p in range(pos + 1):
                        var sv = SIMD[DType.float32, W](0.0); var dd = 0
                        while dd + W <= HD:
                            var qv = qp_bi.load[width=W](qbase + dd)
                            var kv = kc.load[width=W](base + p * HD + dd)
                            sv = sv + qv * kv; dd += W
                        var s = sv.reduce_add() / sqrt(Float32(HD))
                        sc.store(p, s)
                        if s > smax: smax = s
                    var ssum = Float32(0.0)
                    for p in range(pos + 1):
                        var es = exp(sc.load(p) - smax); sc.store(p, es); ssum += es
                    var dd = 0
                    while dd + W <= HD:
                        var ov = SIMD[DType.float32, W](0.0)
                        for p in range(pos + 1):
                            var vv = vc.load[width=W](base + p * HD + dd)
                            ov = ov + vv * (sc.load(p) / ssum)
                        qp_bi.store[width=W](qbase + dd, ov); dd += W
                    while dd < HD:
                        var o = Float32(0.0)
                        for p in range(pos + 1): o += vc.load(base + p*HD + dd) * (sc.load(p) / ssum)
                        qp_bi.store(qbase + dd, o); dd += 1

            _mm_q8_batch(Int(wl.load(lw+5)), qp, bp, NE, NH*HD, nw)
            for bi in range(B):
                var hp_bi = hp + bi * NE; var bp_bi = bp + bi * NE
                for i in range(NE): hp_bi.store(i, hp_bi.load(i) + bp_bi.load(i))

            for bi in range(B): apply_rms_norm(hp + bi * NE, bp + bi * NE, fnp, NE)

            _mm_q8_2out_batch(Int(wl.load(lw+6)), Int(wl.load(lw+7)), bp, gp, up, NF, NE, nw)

            for bi in range(B):
                var gp_bi = gp + bi * NF; var up_bi = up + bi * NF
                for i in range(NF):
                    var gv = gp_bi.load(i)
                    if gv < -80.0: gv = -80.0
                    if gv > 80.0: gv = 80.0
                    gp_bi.store(i, (gv / (1.0 + exp(-gv))) * up_bi.load(i))

            _mm_q8_batch(Int(wl.load(lw+8)), gp, dp, NE, NF, nw)
            for bi in range(B):
                var hp_bi = hp + bi * NE; var dp_bi = dp + bi * NE
                for i in range(NE): hp_bi.store(i, hp_bi.load(i) + dp_bi.load(i))

        for bi in range(B):
            var hp_bi = hp + bi * NE; var bp_bi = bp + bi * NE
            var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w_on))
            apply_rms_norm(hp_bi, bp_bi, onp, NE)
        _mm_q8_batch(Int(w_lm), bp, lp, NV, NE, nw)



        for bi in range(B):
            var lp_bi = lp + bi * NV; var best = 0; var bv = lp_bi.load(0)
            for i in range(1, NV):
                var v = lp_bi.load(i)
                if v > bv: bv = v; best = i
            var nti = Int(nt.load(bi))
            if nti < MAX_SEQ: batch_toks.store(bi*MAX_SEQ+nti, Int32(best)); nt.store(bi, Int32(nti+1))
            if best != 2:
                var out_text = decode_token(voc_data, voc_meta, nv, best)
                if bi == 0: print(out_text, end="")

    print()
    var t_end = time.perf_counter()
    var gen_ms = (t_end - t_gen) * 1000.0
    var total_gen = 0
    for bi in range(B): total_gen += Int(nt.load(bi)) - np
    print("B=", B, " nw=", nw, " Q8_0 toks=", total_gen, " time=", Int(gen_ms), " ms (", Float64(total_gen) / (gen_ms / 1000.0), " tok/s)")
