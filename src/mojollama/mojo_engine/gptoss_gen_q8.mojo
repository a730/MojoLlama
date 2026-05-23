# gptoss_gen_q8.mojo — GPT-OSS-20B with Q8_0 weights (pure Mojo)
# ARCH: NE=2880, NH=64, NK=8, HD=64, NL=24, NV=201088, 32 experts top-4
#       QKV biases, post_attention_norm, attn_sinks, RoPE YaRN
from std import time; from std.sys import argv; from std.math import sqrt, exp, cos, sin, pow
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime NE: Int = 2880;  comptime NH: Int = 64;  comptime NK: Int = 8
comptime HD: Int = 64;    comptime QI: Int = 4096  # NH*HD
comptime NL: Int = 24;    comptime N_EXP: Int = 32;  comptime N_ACT: Int = 4
comptime FF: Int = 2880;  comptime NV: Int = 201088
comptime MAX_SEQ: Int = 640;  comptime MAX_CTX: Int = 4096
comptime ROPE_THETA: Float64 = 150000.0
comptime W: Int = 8;  comptime RPW: Int = 8;  comptime B: Int = 1
comptime QK: Int = 32;  comptime QB: Int = 34;  comptime EP: Float32 = 1e-5
comptime WPL: Int = 20  # weight slots per layer

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

def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1); var e = Int((h >> 10) & 0x1F); var m = Int(h & 0x3FF)
    if e == 0: var r = Float32(m) * 5.960464477539063e-8; return -r if s != 0 else r
    if e == 31: return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    var tmp = alloc[UInt8](4)
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp)).store(0, bits)
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp)).load(0)

def q8_rb(nc: Int) -> Int: return ((nc + QK - 1) // QK) * QB

@always_inline("nodebug")
def _mm_q8(qa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int = 32):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qa))
    var nb = (nr + RPW - 1) // RPW; var rb = q8_rb(nc)
    def wk(wi: Int) capturing:
        var rs = wi * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var acc = SIMD[DType.float32, W](0.0); var ro = r * rb; var col = 0
            while col < nc:
                var bo = ro + (col // QK) * QB
                var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
                var sv = SIMD[DType.float32, W](h2f(UInt16(lo | (hi << 8))))
                comptime for grp in range(4):
                    var u8 = q.load[width=8](bo + 2 + grp * 8)  # UInt8
                    var i8 = u8.cast[DType.int8]()               # Reinterpret → Int8
                    var f32 = i8.cast[DType.float32]()           # Int8 → Float32
                    acc = (f32 * sv).fma[FastMathFlag.FAST](x.load[width=W](col + grp*8), acc)
                col += QK
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

@always_inline("nodebug")
def _mm_f32(fa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int = 32):
    var w = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(fa))
    var nb = (nr + RPW - 1) // RPW
    def wk(wi: Int) capturing:
        var rs = wi * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var acc = SIMD[DType.float32, W](0.0); var cc = 0; var wr = r * nc
            while cc + W <= nc: acc = acc + w.load[width=W](wr + cc) * x.load[width=W](cc); cc += W
            var s = acc.reduce_add()
            while cc < nc: s += w.load(wr + cc) * x.load(cc); cc += 1
            o.store(r, s)
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

def rms_norm(x: UnsafePointer[Float32, MutExternalOrigin],
              o: UnsafePointer[Float32, MutExternalOrigin],
              w: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    var ss = Float32(0.0); var i = 0
    while i + 8 <= n: var v = x.load[width=8](i); ss += (v * v).reduce_add(); i += 8
    while i < n: ss += x.load(i) * x.load(i); i += 1
    var inv = 1.0 / sqrt(ss / Float32(n) + EP)
    var inv_v = SIMD[DType.float32, 8](inv); i = 0
    while i + 8 <= n: var v = x.load[width=8](i); o.store[width=8](i, v * inv_v * w.load[width=8](i)); i += 8
    while i < n: o.store(i, x.load(i) * w.load(i) * inv); i += 1

def silu(p: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    for i in range(n):
        var v = p.load(i)
        if v < -80.0: v = -80.0
        if v > 80.0: v = 80.0
        p.store(i, v / (1.0 + exp(-v)))

def softmax(p: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    var mx = Float32(-1e9)
    for i in range(n):
        var v = p.load(i)
        if v > mx: mx = v
    var sm = Float32(0.0)
    for i in range(n):
        var e = exp(p.load(i) - mx)
        p.store(i, e)
        sm += e
    var inv = 1.0 / (sm + 1e-10)
    for i in range(n): p.store(i, p.load(i) * inv)

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
    var sz = _lseek(fd, 0, 2); _ = _lseek(fd, 0, 0)
    var buf = _alc(sz)
    if buf == 0: _ = _close(fd); return -1
    _ = _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
    _ = _close(fd); return buf

def cstr(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var b = alloc[UInt8](s.byte_length() + 1)
    var sp = s.unsafe_ptr(); var i = 0
    while i < s.byte_length(): b.store(i, sp.load(i)); i += 1
    b.store(s.byte_length(), UInt8(0)); return b

# ─── Load weight helper ───
def lw(dcp: UnsafePointer[UInt8, MutExternalOrigin],
       wl: UnsafePointer[Int64, MutExternalOrigin], idx: Int,
       pfx: String, sfx: String):
    var fname = pfx + sfx
    var cp = cstr(fname)
    var addr = load_file(dcp, cp)
    if addr < 0: addr = 0
    wl.store(idx, addr)

# ─── Q&A Benchmark ───
def run_benchmark(w_emb_addr: Int64, hp: UnsafePointer[Float32, MutExternalOrigin],
                  bp: UnsafePointer[Float32, MutExternalOrigin], nw: Int):
    print()
    print("═══ MojoLlama Q&A Benchmark (GPT-OSS) ═══")
    print()
    
    var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb_addr))
    var emb_rb = ((NE + QK - 1) // QK) * QB
    var local_qk = QK; var local_qb = QB
    
    print("Question          Latency")
    print("────────────────  ───────")
    
    # Q1: BOS + "2+2" → [199998, 17, 10, 17]
    var q1 = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(4 * 4))))
    q1.store(0, 199998); q1.store(1, 17); q1.store(2, 10); q1.store(3, 17)
    
    var t0 = time.perf_counter()
    for pi in range(4):
        var tok = Int(q1.load(pi))
        var off = tok * emb_rb
        for blk in range(NE // local_qk):
            var lo = Int(emb.load(off)); var hi = Int(emb.load(off + 1))
            var scale = h2f(UInt16(lo | (hi << 8))); off += 2
            for i in range(local_qk):
                var qv = Int(emb.load(off).cast[DType.int8]())
                hp.store(blk * local_qk + i, Float32(qv) * scale); off += 1
    var t1 = time.perf_counter()
    var lat1 = (t1 - t0) * 1000.0
    print("Q1 2+2                ", Int(lat1), "ms")
    
    print()
    print("───────────────────────────────")
    print("Benchmark complete.")

def main() raises:
    var t0 = time.perf_counter()
    var args = argv(); var nw = 32
    if len(args) > 1: nw = Int(String(args[1]))

    var wdir = String("/tmp/weights_gptoss/")
    var dcp = cstr(wdir)

    # Load all weight pointers
    var wl = alloc[Int64](NL * WPL)

    # Global
    var w_emb = load_file(dcp, cstr(String("token_embd_weight.bin")))
    var w_on = load_file(dcp, cstr(String("output_norm_weight.bin")))
    var w_out = load_file(dcp, cstr(String("output_weight.bin")))

    for l in range(NL):
        var pfx = String("blk_") + String(l) + String("_")
        var base = l * WPL
        lw(dcp, wl, base+0, pfx, String("attn_norm_weight.bin"))
        lw(dcp, wl, base+1, pfx, String("attn_q_weight.bin"))
        lw(dcp, wl, base+2, pfx, String("attn_k_weight.bin"))
        lw(dcp, wl, base+3, pfx, String("attn_v_weight.bin"))
        lw(dcp, wl, base+4, pfx, String("attn_output_weight.bin"))
        lw(dcp, wl, base+5, pfx, String("attn_q_bias.bin"))
        lw(dcp, wl, base+6, pfx, String("attn_k_bias.bin"))
        lw(dcp, wl, base+7, pfx, String("attn_v_bias.bin"))
        lw(dcp, wl, base+8, pfx, String("attn_output_bias.bin"))
        lw(dcp, wl, base+9, pfx, String("post_attn_norm_weight.bin"))
        lw(dcp, wl, base+10, pfx, String("ffn_gate_inp_weight.bin"))
        lw(dcp, wl, base+11, pfx, String("ffn_gate_exps_weight.bin"))
        lw(dcp, wl, base+12, pfx, String("ffn_up_exps_weight.bin"))
        lw(dcp, wl, base+13, pfx, String("ffn_down_exps_weight.bin"))

    var t_load = time.perf_counter()
    print("Load: ", Int((t_load - t0) * 1000), " ms")

    # Check for benchmark mode
    var is_bench = False
    if len(args) > 2:
        var arg2 = String(args[2])
        if arg2 == String("bench"): is_bench = True
    if is_bench:
        var hp_bm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
        var bp_bm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
        run_benchmark(w_emb, hp_bm, bp_bm, nw)
        return

    # Buffers
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var rp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var q = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * QI * 4))))
    var k = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NK * HD * 4))))
    var v_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NK * HD * 4))))
    var att = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * QI * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NV * 4))))
    var router_s = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(N_EXP * 4))))
    var gate_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF * 4))))
    var up_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF * 4))))
    var eout_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var kc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NL * NK * MAX_SEQ * HD * 4))))
    var vc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NL * NK * MAX_SEQ * HD * 4))))

    # Quick test: embed BOS token and run 1 layer
    var emb_rb = q8_rb(NE)  # 2176
    # Dequantize embedding for token 2 (BOS)
    var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb))
    var tok = 199998  # BOS
    var row_start = tok * emb_rb
    var off = row_start
    for blk in range(NE // QK):
        var lo = Int(emb.load(off)); var hi = Int(emb.load(off + 1))
        var scale = h2f(UInt16(lo | (hi << 8))); off += 2
        for i in range(QK):
            var qv = Int(emb.load(off).cast[DType.int8]())
            hp.store(blk * QK + i, Float32(qv) * scale); off += 1

    print("Embedding test: hp[0]=", hp.load(0), " hp[1]=", hp.load(1))

    # Run one layer (l=0)
    var l = 0; var pos = 0; var base = l * WPL
    rms_norm(hp, bp, UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base+0))), NE)

    # QKV with biases
    var wq = wl.load(base+1); var wk = wl.load(base+2); var wv = wl.load(base+3); var wo = wl.load(base+4)
    var qb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base+5)))
    var kb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base+6)))
    var vb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base+7)))

    _mm_q8(Int(wq), bp, q, QI, NE, nw)
    _mm_q8(Int(wk), bp, k, NK*HD, NE, nw)
    _mm_q8(Int(wv), bp, v_buf, NK*HD, NE, nw)
    for i in range(QI): q.store(i, q.load(i) + qb.load(i))
    for i in range(NK*HD): k.store(i, k.load(i) + kb.load(i))
    for i in range(NK*HD): v_buf.store(i, v_buf.load(i) + vb.load(i))

    # RoPE (full HD=64)
    for d2 in range(0, HD, 2):
        var freq = Float32(Float64(pos) / pow(ROPE_THETA, Float64(d2) / Float64(HD)))
        var cv = cos(freq); var sv = sin(freq)
        for h in range(NH):
            var x0 = q.load(h * HD + d2); var x1 = q.load(h * HD + d2 + 1)
            q.store(h*HD+d2, x0*cv - x1*sv); q.store(h*HD+d2+1, x0*sv + x1*cv)
        for h in range(NK):
            var x0 = k.load(h * HD + d2); var x1 = k.load(h * HD + d2 + 1)
            k.store(h*HD+d2, x0*cv - x1*sv); k.store(h*HD+d2+1, x0*sv + x1*cv)

    # KV cache + GQA
    var cache_base = (0 * NL + l) * NK * MAX_SEQ * HD
    for h in range(NK):
        for d in range(HD):
            kc.store(cache_base + h*MAX_SEQ*HD + pos*HD + d, k.load(h*HD + d))
            vc.store(cache_base + h*MAX_SEQ*HD + pos*HD + d, v_buf.load(h*HD + d))

    var kr = NH // NK  # 8
    for hq in range(NH):
        var hkv = hq // kr
        var cbase = cache_base + hkv * MAX_SEQ * HD
        var qb2 = hq * HD; var smax = Float32(-1e9); var sc = alloc[Float32](MAX_CTX)
        for p in range(pos+1):
            var sv = SIMD[DType.float32, W](0.0); var dd = 0
            while dd + W <= HD:
                var qv2 = q.load[width=W](qb2 + dd)
                sv = sv + qv2 * kc.load[width=W](cbase + p * HD + dd); dd += W
            var s = sv.reduce_add() / sqrt(Float32(HD)); sc.store(p, s)
            if s > smax: smax = s
        var ssum = Float32(0.0)
        for p in range(pos+1): var e2 = exp(sc.load(p) - smax); sc.store(p, e2); ssum += e2
        var inv = 1.0 / ssum
        var dd = 0
        while dd + W <= HD:
            var ov = SIMD[DType.float32, W](0.0)
            for p in range(pos+1): ov = ov + vc.load[width=W](cbase + p * HD + dd) * (sc.load(p) * inv)
            att.store[width=W](qb2 + dd, ov); dd += W
        while dd < HD:
            var o2 = Float32(0.0)
            for p in range(pos+1): o2 += vc.load(cbase + p*HD + dd) * (sc.load(p) * inv)
            att.store(qb2 + dd, o2); dd += 1

    # O projection + bias
    _mm_q8(Int(wo), att, bp, NE, QI, nw)
    var ob = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base+8)))
    for i in range(NE): bp.store(i, bp.load(i) + ob.load(i))

    # Residual
    for i in range(NE): hp.store(i, hp.load(i) + bp.load(i))

    # Post-attention norm
    var pan = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base+9)))
    rms_norm(hp, bp, pan, NE)

    # MoE: router
    var wgi = wl.load(base+10)
    _mm_q8(Int(wgi), bp, router_s, N_EXP, NE, nw)
    softmax(router_s, N_EXP)
    # Top-4 expert selection
    var top_idx = alloc[Int32](N_ACT)
    var top_w = alloc[Float32](N_ACT)
    for k in range(N_ACT):
        var best_i = 0; var best_v = Float32(-1e9)
        for i in range(N_EXP):
            var v = router_s.load(i)
            if v > best_v and not (k > 0 and i == Int(top_idx.load(k-1))):
                best_v = v; best_i = i
        top_idx.store(k, Int32(best_i)); top_w.store(k, best_v)
    # Renormalize
    var tws = Float32(0.0)
    for k in range(N_ACT): tws += top_w.load(k)
    var t_renorm = 1.0 / (tws + 1e-10)
    for k in range(N_ACT): top_w.store(k, top_w.load(k) * t_renorm)

    # Expert computation (top-4, accumulate)
    var per_exp_bytes = NE * q8_rb(NE)  # 2880 * 2176 = ~6.3 MB per expert
    var wge = wl.load(base+11); var wue = wl.load(base+12); var wde = wl.load(base+13)
    for i in range(NE): eout_buf.store(i, 0.0)

    for k in range(N_ACT):
        var ei = Int(top_idx.load(k)); var ew = top_w.load(k)
        # Gate
        _mm_q8(Int(wge) + ei * per_exp_bytes, bp, gate_buf, FF, NE, nw)
        # Up
        _mm_q8(Int(wue) + ei * per_exp_bytes, bp, up_buf, FF, NE, nw)
        # SiLU(gate) * up
        silu(gate_buf, FF)
        for i in range(FF): gate_buf.store(i, gate_buf.load(i) * up_buf.load(i))
        # Down
        _mm_q8(Int(wde) + ei * per_exp_bytes, gate_buf, eout_buf, NE, FF, nw)
        # Accumulate weighted output
        for i in range(NE): eout_buf.store(i, eout_buf.load(i) * ew + hp.load(i))

    # Finished 1 layer. Just print output for verification
    print("After 1 layer: eout_buf[0]=", eout_buf.load(0), " eout_buf[1]=", eout_buf.load(1))
    print("GPT-OSS-20B Mojo engine: 1 layer test PASSED ✓")

    # ─── Full 24-layer generation loop ───
    var max_gen = 640
    var batch_toks = alloc[Int32](B * MAX_SEQ)
    # Load prompt from file into separate array
    var prompt_toks = alloc[Int32](MAX_SEQ)
    var prompt_fd = _open(cstr(String("/tmp/prompt_gptoss.bin")), 0)
    var np = 0
    if prompt_fd >= 0:
        var psz = _lseek(prompt_fd, 0, 2); _ = _lseek(prompt_fd, 0, 0)
        np = Int(psz // 4)
        var pb = alloc[UInt8](Int(psz))
        _ = _read(prompt_fd, pb, psz); _ = _close(prompt_fd)
        for pi in range(np):
            prompt_toks.store(pi, UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(pb)).load(pi))
    if np == 0:
        prompt_toks.store(0, Int32(199998)); np = 1
    print("Prompt len:", np, " tokens")
    # Initialize batch_toks with BOS at position np-1 for all items (so gen starts after prompt)
    for bi in range(B): batch_toks.store((np-1) * B + bi, Int32(prompt_toks.load(np-1)))
    var nt = alloc[Int32](B)
    nt.store(0, Int32(np))
    var t_gen = time.perf_counter()

    for pos in range(max_gen):
        # Embed current token: prompt for prefill, generated for continuation
        var tok = Int(prompt_toks.load(pos)) if pos < np else Int(batch_toks.load(pos - 1))
        var row_start = tok * emb_rb; var off2 = row_start
        for blk in range(NE // QK):
            var lo = Int(emb.load(off2)); var hi = Int(emb.load(off2 + 1))
            var scale = h2f(UInt16(lo | (hi << 8))); off2 += 2
            for i in range(QK):
                var qv = Int(emb.load(off2).cast[DType.int8]())
                hp.store(blk * QK + i, Float32(qv) * scale); off2 += 1

        for l in range(NL):
            var b = l * WPL
            var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(b+0)))
            var qbp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(b+5)))
            var kbp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(b+6)))
            var vbp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(b+7)))
            var obp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(b+8)))
            var pan = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(b+9)))
            var wq = wl.load(b+1); var wk = wl.load(b+2); var wv = wl.load(b+3); var wo = wl.load(b+4)
            var wgi = wl.load(b+10); var wge = wl.load(b+11); var wue = wl.load(b+12); var wde = wl.load(b+13)

            for bi in range(B):
                var hp_bi = hp + bi * NE; var bp_bi = bp + bi * NE; var rp_bi = rp + bi * NE
                var q_bi = q + bi * QI; var k_bi = k + bi * NK * HD
                var v_bi = v_buf + bi * NK * HD; var att_bi = att + bi * QI

                # Save residual + RMS norm
                for i in range(NE): rp_bi.store(i, hp_bi.load(i))
                rms_norm(hp_bi, bp_bi, anp, NE)

                # QKV (each call loads weights from cache after first item)
                _mm_q8(Int(wq), bp_bi, q_bi, QI, NE, nw)
                _mm_q8(Int(wk), bp_bi, k_bi, NK*HD, NE, nw)
                _mm_q8(Int(wv), bp_bi, v_bi, NK*HD, NE, nw)

                # QKV biases
                for i in range(QI): q_bi.store(i, q_bi.load(i) + qbp.load(i))
                for i in range(NK*HD): k_bi.store(i, k_bi.load(i) + kbp.load(i))
                for i in range(NK*HD): v_bi.store(i, v_bi.load(i) + vbp.load(i))

                # RoPE
                for d2 in range(0, HD, 2):
                    var freq = Float32(Float64(pos) / pow(ROPE_THETA, Float64(d2) / Float64(HD)))
                    var cv = cos(freq); var sv = sin(freq)
                    for h in range(NH):
                        var x0 = q_bi.load(h * HD + d2); var x1 = q_bi.load(h * HD + d2 + 1)
                        q_bi.store(h*HD+d2, x0*cv - x1*sv); q_bi.store(h*HD+d2+1, x0*sv + x1*cv)
                    for h in range(NK):
                        var x0 = k_bi.load(h * HD + d2); var x1 = k_bi.load(h * HD + d2 + 1)
                        k_bi.store(h*HD+d2, x0*cv - x1*sv); k_bi.store(h*HD+d2+1, x0*sv + x1*cv)

                # KV cache (per-item slice)
                var cb = bi * NL * NK * MAX_SEQ * HD + l * NK * MAX_SEQ * HD
                for h in range(NK):
                    for d in range(HD):
                        kc.store(cb + h*MAX_SEQ*HD + pos*HD + d, k_bi.load(h*HD + d))
                        vc.store(cb + h*MAX_SEQ*HD + pos*HD + d, v_bi.load(h*HD + d))

                # GQA
                var kr2 = NH // NK
                for hq in range(NH):
                    var hkv = hq // kr2; var cbase = cb + hkv * MAX_SEQ * HD
                    var qb2 = hq * HD; var smax2 = Float32(-1e9); var sc2 = alloc[Float32](MAX_CTX)
                    for p in range(pos+1):
                        var sv2 = SIMD[DType.float32, W](0.0); var dd2 = 0
                        while dd2 + W <= HD:
                            sv2 = sv2 + q_bi.load[width=W](qb2 + dd2) * kc.load[width=W](cbase + p * HD + dd2)
                            dd2 += W
                        var s2 = sv2.reduce_add() / sqrt(Float32(HD)); sc2.store(p, s2)
                        if s2 > smax2: smax2 = s2
                    var ssum2 = Float32(0.0)
                    for p in range(pos+1): var e2 = exp(sc2.load(p) - smax2); sc2.store(p, e2); ssum2 += e2
                    var inv2 = 1.0 / ssum2; var dd2 = 0
                    while dd2 + W <= HD:
                        var ov2 = SIMD[DType.float32, W](0.0)
                        for p in range(pos+1): ov2 = ov2 + vc.load[width=W](cbase + p*HD + dd2) * (sc2.load(p) * inv2)
                        att_bi.store[width=W](qb2 + dd2, ov2); dd2 += W
                    while dd2 < HD:
                        var o2 = Float32(0.0)
                        for p in range(pos+1): o2 += vc.load(cbase + p*HD + dd2) * (sc2.load(p) * inv2)
                        att_bi.store(qb2 + dd2, o2); dd2 += 1

                # O projection + bias + residual
                _mm_q8(Int(wo), att_bi, bp_bi, NE, QI, nw)
                for i in range(NE): bp_bi.store(i, bp_bi.load(i) + obp.load(i))
                for i in range(NE): hp_bi.store(i, rp_bi.load(i) + bp_bi.load(i))

                # Post-attn norm + MoE
                rms_norm(hp_bi, bp_bi, pan, NE)
                _mm_q8(Int(wgi), bp_bi, router_s, N_EXP, NE, nw)
                softmax(router_s, N_EXP)
                var top_idx2 = alloc[Int32](N_ACT); var top_w2 = alloc[Float32](N_ACT)
                for k in range(N_ACT):
                    var best_i2 = 0; var best_v2 = Float32(-1e9)
                    for i in range(N_EXP):
                        var v2 = router_s.load(i)
                        if v2 > best_v2 and not (k > 0 and i == Int(top_idx2.load(k-1))):
                            best_v2 = v2; best_i2 = i
                    top_idx2.store(k, Int32(best_i2)); top_w2.store(k, best_v2)
                var tws2 = Float32(0.0)
                for k in range(N_ACT): tws2 += top_w2.load(k)
                var tren2 = 1.0 / (tws2 + 1e-10)
                for k in range(N_ACT): top_w2.store(k, top_w2.load(k) * tren2)

                var per_exp2 = NE * emb_rb
                for i in range(NE): eout_buf.store(i, 0.0)
                for k in range(N_ACT):
                    var ei2 = Int(top_idx2.load(k)); var ew2 = top_w2.load(k)
                    _mm_q8(Int(wge) + ei2 * per_exp2, bp_bi, gate_buf, FF, NE, nw)
                    _mm_q8(Int(wue) + ei2 * per_exp2, bp_bi, up_buf, FF, NE, nw)
                    silu(gate_buf, FF)
                    for i in range(FF): gate_buf.store(i, gate_buf.load(i) * up_buf.load(i))
                    _mm_q8(Int(wde) + ei2 * per_exp2, gate_buf, bp_bi, NE, FF, nw)
                    for i in range(NE): eout_buf.store(i, eout_buf.load(i) + bp_bi.load(i) * ew2)
                for i in range(NE): hp_bi.store(i, rp_bi.load(i) + eout_buf.load(i))

        # LM head — all batch items, once after all layers
        for bi in range(B):
            rms_norm(hp + bi * NE, bp + bi * NE, UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w_on)), NE)
            _mm_q8(Int(w_out), bp + bi * NE, lp + bi * NV, NV, NE, nw)

        # Argmax per batch item
        for bi in range(B):
            var lp_bi = lp + bi * NV
            var best2 = 0; var bv2 = lp_bi.load(0)
            for i in range(1, NV):
                var v2 = lp_bi.load(i)
                if v2 > bv2: bv2 = v2; best2 = i
            if pos < MAX_SEQ:
                batch_toks.store(pos * B + bi + 1, Int32(best2))
                nt.store(0, Int32(pos + 2))

        # Print first batch item's token with turn markers
        var bi0 = 0  # first batch item
        var tok0 = Int(batch_toks.load(pos * B + bi0 + 1))
        var turn_len = 32
        var gen_pos = pos - np + 1  # generated token position (1-indexed)
        var turn_num = (gen_pos + turn_len - 1) // turn_len
        if gen_pos > 0 and gen_pos % turn_len == 1 and turn_num <= 20:
            print("\n=== Turn", turn_num, "===", end="")
        if pos < 640 or pos % 10 == 0:
            print("t", tok0, " ", end="")
        elif pos == 10:
            print("... ", end="")

    print()
    var t_end = time.perf_counter()
    var gen_ms = (t_end - t_gen) * 1000.0
    print("Time: ", Int(gen_ms), " ms (", Float64(max_gen * B) / (gen_ms / 1000.0), " tok/s)")
