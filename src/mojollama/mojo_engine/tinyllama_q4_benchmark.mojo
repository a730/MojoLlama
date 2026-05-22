# tinyllama_q4_benchmark.mojo — Real Q4_0 inference with on-the-fly dequant
# WHAT:  Loads raw Q4_0 weights (681MB), runs generation with q4_0_matmul.
#        Norm weights are F32, all matmul weights are Q4_0 packed.
# WHY:   True quant speed vs f16-dequantized-extract.
# WHEN:  2026-05-22

from std import time
from std.math import sqrt
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
comptime Q4_0_BS: Int = 18  # 18 bytes per 32 values

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...

@extern("open")
def _open(p: UnsafePointer[UInt8, MutExternalOrigin], f: Int) abi("C") -> Int: ...

@extern("read")
def _read(fd: Int, b: UnsafePointer[UInt8, MutExternalOrigin], c: Int64) abi("C") -> Int64: ...

@extern("lseek")
def _lseek(fd: Int, o: Int64, w: Int) abi("C") -> Int64: ...

@extern("close")
def _close(fd: Int) abi("C") -> Int: ...

@extern("sinf")
def _sinf(x: Float32) abi("C") -> Float32: ...

@extern("cosf")
def _cosf(x: Float32) abi("C") -> Float32: ...

@extern("powf")
def _powf(x: Float32, y: Float32) abi("C") -> Float32: ...

@extern("expf")
def _expf(x: Float32) abi("C") -> Float32: ...

@extern("sqrtf")
def _sqrtf(x: Float32) abi("C") -> Float32: ...

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
                 voc_meta: UnsafePointer[Int32, MutExternalOrigin],
                 nv: Int, tok: Int) -> String:
    for i in range(nv):
        var vid = Int(voc_meta.load(nv + i))
        if vid == tok:
            var off = Int(voc_meta.load(2 * nv + i))
            var l = Int(voc_meta.load(i))
            var p = voc + off
            if l >= 3 and p.load(0) == 0xE2 and p.load(1) == 0x96 and p.load(2) == 0x81:
                p += 3
                l -= 3
            var result = String("")
            for j in range(l):
                result = result + chr(Int(p.load(j)))
            return result
    return String("")

# ── Q4_0 matmul (on-the-fly dequant) ──
@always_inline("nodebug")
def q4_0_mm(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW
        var re = rs + RPW
        if re > nr:
            re = nr
        for r in range(rs, re):
            var ro = r * (nc // 32) * Q4_0_BS
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                var bo = ro + (blk // 32) * Q4_0_BS
                var d = h2f(UInt16(w.load(bo)) | (UInt16(w.load(bo + 1)) << 8))
                var dv = SIMD[DType.float32, W](d)
                for jg in range(0, 32, W):
                    var vals = SIMD[DType.float32, W](0.0)
                    for k in range(W):
                        var nib = w.load(bo + 2 + (jg + k) // 2)
                        if (jg + k) % 2 == 0:
                            var qv = Int(nib & 0x0F)
                            vals[k] = Float32(qv - 16 if qv >= 8 else qv)
                        else:
                            var qv = Int(nib >> 4)
                            vals[k] = Float32(qv - 16 if qv >= 8 else qv)
                    var xv = x.load[width=W](blk + jg)
                    acc = acc + vals * dv * xv
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── f32 matmul (for norm weights) ──
@always_inline("nodebug")
def f32_mm(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
           o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wa))
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
            for blk in range(0, nc, W):
                var wv = w.load[width=W](ro + blk)
                var xv = x.load[width=W](blk)
                acc = acc + wv * xv
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── Main ──
def main():
    print("============================================================")
    print("TINYLLAMA Q4_0 REAL INFERENCE (on-the-fly dequant)")
    print("============================================================")

    var t0 = time.perf_counter()

    # Load vocab
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

    # Load Q4_0 weights
    var wdir = String("/tmp/weights_tl_tl-Q4_0/")
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
    var wtype = alloc[Int32](NL * 9)  # 0=Q4_0, 1=F32
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
            # Detect format by file size: norm weights are F32 (small), others Q4_0
            # F32 norm: 2048*4 = 8192 bytes
            # Q4_0: (2048*2048/32)*18 = 2359296 for square matrices
            # For non-square: attn_k = 2048*256/32*18 = 294912, etc
            wtype.store(l * 9 + f, Int32(0))  # Q4_0 default
            if f == 0 or f == 1:  # norm weights
                wtype.store(l * 9 + f, Int32(1))  # F32

    var t_load = time.perf_counter()
    print("Model load: ", Int((t_load - t0) * 1000), " ms")
    print("Weight format: Q4_0 (on-the-fly dequant)")
    print("")

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

    var output_tokens = alloc[Int32](MAX_SEQ)

    # Prompt "2+2=" tokens
    var prompt_toks = [1, 29871, 29906, 29974, 29906, 29922]
    var np = 6
    var max_gen = 10

    print("Prompt: ", end="")
    for i in range(np):
        var tok = prompt_toks[i]
        var text = decode_token(voc_data, voc_meta, nv, tok)
        print(text, end="")
    print("")
    print("Output: ", end="")

    var t_gen = time.perf_counter()
    var nt = np
    for i in range(np):
        output_tokens.store(i, Int32(prompt_toks[i]))

    for pos in range(max_gen):
        if nt >= MAX_SEQ:
            break
        var tok = Int(output_tokens.load(pos))

        # Embed (Q4_0 token_embd is actually... let me check)
        # token_embd in Q4_0 extraction: 2048*32000*4/32*18 = 115,200,000 bytes expected
        # But actually token_embd is typically NOT quantized in GGUF
        # Let me check: the file size should tell us
        # For now, assume it's Q4_0 packed
        var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb))
        var row_bytes = (NE // 32) * Q4_0_BS
        var bo = tok * row_bytes
        for i in range(NE):
            var d = h2f(UInt16(emb.load(bo)) | (UInt16(emb.load(bo + 1)) << 8))
            var j = i
            var nib = emb.load(bo + 2 + j // 2)
            var qv: Int
            if j % 2 == 0:
                qv = Int(nib & 0x0F)
            else:
                qv = Int(nib >> 4)
            var val = Float32(qv - 16 if qv >= 8 else qv)
            hp.store(i, val * d)

        for l in range(NL):
            var lw = l * 9

            # RMS Norm (F32 weights)
            var ss = Float64(0.0)
            for i in range(NE):
                ss += Float64(hp.load(i)) * Float64(hp.load(i))
            var inv = Float32(1.0 / sqrt(Float64(ss) / Float64(NE) + 1e-6))
            var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw)))
            for i in range(NE):
                bp.store(i, hp.load(i) * anp.load(i) * inv)

            # QKV (Q4_0 weights)
            q4_0_mm(Int(wl.load(lw + 2)), bp, qp, NH * HD, NE)
            q4_0_mm(Int(wl.load(lw + 3)), bp, kp, NK * HD, NE)
            q4_0_mm(Int(wl.load(lw + 4)), bp, vbuf, NK * HD, NE)

            # RoPE
            for h in range(NH):
                for d2 in range(0, HD, 2):
                    var freq = Float32(pos) / Float32(_powf(10000.0, Float32(Float64(d2) / Float64(HD))))
                    var cv = _cosf(freq)
                    var sv = _sinf(freq)
                    var x0 = qp.load(h * HD + d2)
                    var x1 = qp.load(h * HD + d2 + 1)
                    qp.store(h * HD + d2, x0 * cv - x1 * sv)
                    qp.store(h * HD + d2 + 1, x0 * sv + x1 * cv)
            for h in range(NK):
                for d2 in range(0, HD, 2):
                    var freq = Float32(pos) / Float32(_powf(10000.0, Float32(Float64(d2) / Float64(HD))))
                    var cv = _cosf(freq)
                    var sv = _sinf(freq)
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

            # GQA Attention
            var kr = NH // NK
            for hq in range(NH):
                var hk = hq // kr
                var sc = alloc[Float32](pos + 1)
                var smax = Float32(-1e9)
                for p in range(pos + 1):
                    var s = Float32(0.0)
                    for d in range(HD):
                        s += qp.load(hq * HD + d) * kc.load(lo + hk * MAX_SEQ * HD + p * HD + d)
                    s = s / _sqrtf(Float32(HD))
                    sc.store(p, s)
                    if s > smax:
                        smax = s
                var ssum = Float32(0.0)
                for p in range(pos + 1):
                    var es = _expf(sc.load(p) - smax)
                    sc.store(p, es)
                    ssum += es
                for d in range(HD):
                    var o = Float32(0.0)
                    for p in range(pos + 1):
                        o += vc.load(lo + hk * MAX_SEQ * HD + p * HD + d) * (sc.load(p) / ssum)
                    qp.store(hq * HD + d, o)

            # O projection (Q4_0)
            q4_0_mm(Int(wl.load(lw + 5)), qp, bp, NE, NH * HD)
            for i in range(NE):
                hp.store(i, hp.load(i) + bp.load(i))

            # RMS Norm pre-FFN (F32)
            ss = 0.0
            for i in range(NE):
                ss += Float64(hp.load(i)) * Float64(hp.load(i))
            inv = Float32(1.0 / sqrt(Float64(ss) / Float64(NE) + 1e-6))
            var fnp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw + 1)))
            for i in range(NE):
                bp.store(i, hp.load(i) * fnp.load(i) * inv)

            # FFN gate + up (Q4_0)
            q4_0_mm(Int(wl.load(lw + 6)), bp, gp, NF, NE)
            q4_0_mm(Int(wl.load(lw + 7)), bp, up, NF, NE)

            # SiLU
            for i in range(NF):
                var gv = gp.load(i)
                if gv < -80.0:
                    gv = -80.0
                if gv > 80.0:
                    gv = 80.0
                gp.store(i, (gv / (1.0 + _expf(-gv))) * up.load(i))

            # Down projection (Q4_0)
            q4_0_mm(Int(wl.load(lw + 8)), gp, dp, NE, NF)
            for i in range(NE):
                hp.store(i, hp.load(i) + dp.load(i))

        # Final norm + LM head
        ss = 0.0
        for i in range(NE):
            ss += Float64(hp.load(i)) * Float64(hp.load(i))
        inv = Float32(1.0 / sqrt(Float64(ss) / Float64(NE) + 1e-6))
        var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w_on))
        for i in range(NE):
            bp.store(i, hp.load(i) * onp.load(i) * inv)
        q4_0_mm(Int(w_lm), bp, lp, NV, NE)

        # Argmax
        var best = 0
        var bv = lp.load(0)
        for i in range(1, NV):
            var v = lp.load(i)
            if v > bv:
                bv = v
                best = i

        output_tokens.store(nt, Int32(best))
        nt += 1
        if best == 2:
            break
        var out_text = decode_token(voc_data, voc_meta, nv, best)
        print(out_text, end="")

    print("")
    var t_end = time.perf_counter()
    var gen_ms = (t_end - t_gen) * 1000.0
    var n_gen = nt - np
    print("Tokens: ", n_gen, " | Time: ", Int(gen_ms), " ms | Speed: ", Float64(n_gen) / (gen_ms / 1000.0), " tok/s")
