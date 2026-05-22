# tinyllama_gen.mojo — Full TinyLlama inference: f16 weights + vocab decode + generation
# WHAT:  Loads f16 .bin weights, runs autoregressive generation with KV cache.
#        Decodes tokens via vocab.bin lookup. Pure Mojo, no Python after load.
# WHY:   Real benchmark numbers vs synthetic matmul throughput.
# WHEN:  2026-05-22

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

# ── f16 matmul ──
@always_inline("nodebug")
def _mm_f16(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr:
            re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

@always_inline("nodebug")
def _mm_f16_2out(wa: Int, wb: Int,
                  x: UnsafePointer[Float32, MutExternalOrigin],
                  oa: UnsafePointer[Float32, MutExternalOrigin],
                  ob: UnsafePointer[Float32, MutExternalOrigin],
                  nr: Int, nc: Int):
    var wa_ptr = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var wb_ptr = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wb))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr:
            re = nr
        for r in range(rs, re):
            var ro = r * nc
            var acc_a = SIMD[DType.float32, W](0.0)
            var acc_b = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var xv = x.load[width=W](blk + grp * 8)
                    var wva = wa_ptr.load[width=W](ro + blk + grp * 8)
                    var wvb = wb_ptr.load[width=W](ro + blk + grp * 8)
                    acc_a = wva.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc_a)
                    acc_b = wvb.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc_b)
            oa.store(r, acc_a.reduce_add())
            ob.store(r, acc_b.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

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
    # Reinterpret bits as Float32
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

# ── Decode token to String ──
def decode_token(voc: UnsafePointer[UInt8, MutExternalOrigin],
                 voc_lens: UnsafePointer[Int32, MutExternalOrigin],
                 nv: Int, tok: Int) -> String:
    for i in range(nv):
        var vid = Int(voc_lens.load(nv + i))
        if vid == tok:
            var off = Int(voc_lens.load(2 * nv + i))
            var l = Int(voc_lens.load(i))
            var p = voc + off
            # Strip leading space marker (0xE2 0x96 0x81)
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

    # Vocab
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

    # Parse vocab
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

    # Load base weights
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

    # Buffers
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var qp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NH * HD * 4))))
    var kp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NK * HD * 4))))
    var vbuf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NK * HD * 4))))
    var gp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NF * 4))))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NF * 4))))
    var dp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NV * 4))))

    var kc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NL * NK * MAX_SEQ * HD * 4))))
    var vc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NL * NK * MAX_SEQ * HD * 4))))

    # Prompt "2+2=" tokens
    var prompt_toks = [1, 29871, 29906, 29974, 29906, 29922]
    var np = 6
    var max_gen = 10
    var toks = alloc[Int32](MAX_SEQ)
    for i in range(np):
        toks.store(i, Int32(prompt_toks[i]))
    var nt = np

    var prompt_text = String("")
    for i in range(np):
        prompt_text = prompt_text + decode_token(voc_data, voc_meta, nv, Int(toks.load(i)))
    print("Prompt: ", prompt_text)
    print("Output: ", end="")

    var t_gen = time.perf_counter()

    for pos in range(max_gen):
        if nt >= MAX_SEQ:
            break
        var tok = Int(toks.load(pos))

        # Embed
        var emb = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(w_emb))
        for i in range(NE):
            hp.store(i, h2f(emb.load(tok * NE + i)))

        for l in range(NL):
            var lw = l * 9

            # RMS Norm pre-attention (stride-8 SIMD)
            var ss = Float32(0.0)
            var i = 0
            while i + 8 <= NE:
                var v = hp.load[width=8](i)
                ss += (v * v).reduce_add()
                i += 8
            while i < NE:
                ss += hp.load(i) * hp.load(i)
                i += 1
            var inv = 1.0 / sqrt(ss / Float32(NE) + 1e-6)
            var inv_v = SIMD[DType.float32, 8](inv)
            var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw)))
            i = 0
            while i + 8 <= NE:
                var v = hp.load[width=8](i)
                var w = anp.load[width=8](i)
                bp.store[width=8](i, v * inv_v * w)
                i += 8
            while i < NE:
                bp.store(i, hp.load(i) * anp.load(i) * inv)
                i += 1

            # QKV
            _mm_f16(Int(wl.load(lw + 2)), bp, qp, NH * HD, NE)
            _mm_f16(Int(wl.load(lw + 3)), bp, kp, NK * HD, NE)
            _mm_f16(Int(wl.load(lw + 4)), bp, vbuf, NK * HD, NE)

            # RoPE
            for h in range(NH):
                for d2 in range(0, HD, 2):
                    var freq = Float32(Float64(pos) / pow(10000.0, Float64(d2) / Float64(HD)))
                    var cv = cos(freq)
                    var sv = sin(freq)
                    var x0 = qp.load(h * HD + d2)
                    var x1 = qp.load(h * HD + d2 + 1)
                    qp.store(h * HD + d2, x0 * cv - x1 * sv)
                    qp.store(h * HD + d2 + 1, x0 * sv + x1 * cv)
            for h in range(NK):
                for d2 in range(0, HD, 2):
                    var freq = Float32(Float64(pos) / pow(10000.0, Float64(d2) / Float64(HD)))
                    var cv = cos(freq)
                    var sv = sin(freq)
                    var x0 = kp.load(h * HD + d2)
                    var x1 = kp.load(h * HD + d2 + 1)
                    kp.store(h * HD + d2, x0 * cv - x1 * sv)
                    kp.store(h * HD + d2 + 1, x0 * sv + x1 * cv)

            # Store KV cache
            var lo = l * NK * MAX_SEQ * HD
            for h in range(NK):
                for d in range(HD):
                    kc.store(lo + h * MAX_SEQ * HD + pos * HD + d, kp.load(h * HD + d))
                    vc.store(lo + h * MAX_SEQ * HD + pos * HD + d, vbuf.load(h * HD + d))

            # GQA attention with KV cache
            var kr = NH // NK
            for hq in range(NH):
                var hk = hq // kr
                var sc = alloc[Float32](pos + 1)
                var smax = Float32(-1e9)
                for p in range(pos + 1):
                    var s = Float32(0.0)
                    for d in range(HD):
                        s += qp.load(hq * HD + d) * kc.load(lo + hk * MAX_SEQ * HD + p * HD + d)
                    s = s / sqrt(Float32(HD))
                    sc.store(p, s)
                    if s > smax:
                        smax = s
                var ssum = Float32(0.0)
                for p in range(pos + 1):
                    var es = exp(sc.load(p) - smax)
                    sc.store(p, es)
                    ssum += es
                for d in range(HD):
                    var o = Float32(0.0)
                    for p in range(pos + 1):
                        o += vc.load(lo + hk * MAX_SEQ * HD + p * HD + d) * (sc.load(p) / ssum)
                    qp.store(hq * HD + d, o)

            # O projection
            _mm_f16(Int(wl.load(lw + 5)), qp, bp, NE, NH * HD)
            for i in range(NE):
                hp.store(i, hp.load(i) + bp.load(i))

            # RMS Norm pre-FFN (stride-8 SIMD)
            ss = Float32(0.0)
            i = 0
            while i + 8 <= NE:
                var v = hp.load[width=8](i)
                ss += (v * v).reduce_add()
                i += 8
            while i < NE:
                ss += hp.load(i) * hp.load(i)
                i += 1
            inv = 1.0 / sqrt(ss / Float32(NE) + 1e-6)
            inv_v = SIMD[DType.float32, 8](inv)
            var fnp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw + 1)))
            i = 0
            while i + 8 <= NE:
                var v = hp.load[width=8](i)
                var w = fnp.load[width=8](i)
                bp.store[width=8](i, v * inv_v * w)
                i += 8
            while i < NE:
                bp.store(i, hp.load(i) * fnp.load(i) * inv)
                i += 1

            # FFN gate + up (fused 2-output matmul)
            _mm_f16_2out(Int(wl.load(lw + 6)), Int(wl.load(lw + 7)), bp, gp, up, NF, NE)

            # SiLU
            for i in range(NF):
                var gv = gp.load(i)
                if gv < -80.0:
                    gv = -80.0
                if gv > 80.0:
                    gv = 80.0
                gp.store(i, (gv / (1.0 + exp(-gv))) * up.load(i))

            # Down projection
            _mm_f16(Int(wl.load(lw + 8)), gp, dp, NE, NF)
            for i in range(NE):
                hp.store(i, hp.load(i) + dp.load(i))

        # Final norm (stride-8 SIMD) + LM head
        ss = Float32(0.0)
        i = 0
        while i + 8 <= NE:
            var v = hp.load[width=8](i)
            ss += (v * v).reduce_add()
            i += 8
        while i < NE:
            ss += hp.load(i) * hp.load(i)
            i += 1
        inv = 1.0 / Float32(sqrt(Float64(ss / Float32(NE) + 1e-6)))
        inv_v = SIMD[DType.float32, 8](inv)
        var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w_on))
        i = 0
        while i + 8 <= NE:
            var v = hp.load[width=8](i)
            var w = onp.load[width=8](i)
            bp.store[width=8](i, v * inv_v * w)
            i += 8
        while i < NE:
            bp.store(i, hp.load(i) * onp.load(i) * inv)
            i += 1
        _mm_f16(Int(w_lm), bp, lp, NV, NE)

        # Argmax
        var best = 0
        var bv = lp.load(0)
        for i in range(1, NV):
            var v = lp.load(i)
            if v > bv:
                bv = v
                best = i

        toks.store(nt, Int32(best))
        nt += 1
        if best == 2:
            break
        var out_text = decode_token(voc_data, voc_meta, nv, best)
        print(out_text, end="")

    print()
    var t_end = time.perf_counter()
    var gen_ms = (t_end - t_gen) * 1000.0
    var n_gen = nt - np
    print("Generated ", n_gen, " tokens in ", Int(gen_ms), " ms (", Float64(n_gen) / (gen_ms / 1000.0), " tok/s)")
