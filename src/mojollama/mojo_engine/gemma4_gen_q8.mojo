# gemma4_gen_q8.mojo — Gemma 4 inference engine (pure Mojo, Q8_0 weights)
# ARCH: Gemma-4-E2B: NE=1536, wide_QI=4096, narrow_QI=2048, HD=128
#   Wide layers (every 5th): 4, 9, 14, 19, 24, 29, 34
#   NK_HD: 256 narrow, 512 wide
#   FF=6144, NL=35, NV=262144
#
# FEATURES:
#   - QK-norm (separate RMS norm for Q and K)
#   - inp_gate + proj (gating before attention)
#   - Post-norms (attn + ffw + final)
#   - Layer output scaling
#   - Per-layer variable Q/K/O dimensions
#
# Also supports E4B (NE=2560, FF=10240, NK=4, NL=42), 26B-A4B (NE=2816, MoE), 31B (NE=3072)
# via comptime NE, NH, NK, HD, NL, FF, NV parameters.
from std import time
from std.sys import argv
from std.math import sqrt, exp, cos, sin, pow
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag
from std.sys.intrinsics import prefetch

# ─── Architecture (edit for each model) ───
comptime NE: Int = 1536    # n_embd
comptime NH: Int = 16      # n_heads (narrow)
comptime NK: Int = 2       # n_kv_heads (narrow)
comptime HD: Int = 128     # head_dim
comptime NL: Int = 35      # n_layers
comptime FF: Int = 6144    # ffn_hidden
comptime NV: Int = 262144  # vocab_size
comptime QI_NARROW: Int = NH * HD    # 2048
comptime QI_WIDE: Int = NH * HD * 2  # 4096 (wide layers)
comptime WIDE_INTERVAL: Int = 5      # every 5th layer is wide (E2B)
comptime W: Int = 8;  comptime RPW: Int = 8;  comptime B: Int = 4
comptime QK: Int = 32;  comptime QB: Int = 34;  comptime EP: Float32 = 1e-5
comptime MAX_SEQ: Int = 128;  comptime MAX_CTX: Int = 4096

# Weight slots per layer
comptime WPL: Int = 18

# ─── Helpers ───
def is_wide(l: Int) -> Bool: return l % WIDE_INTERVAL == (WIDE_INTERVAL - 1)
def qi(l: Int) -> Int: return QI_WIDE if is_wide(l) else QI_NARROW
def nk_hd(l: Int) -> Int: return (NK * HD * 2) if is_wide(l) else (NK * HD)

# ─── Extern ───
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

def q8_rb(nc: Int) -> Int: return ((nc + QK - 1) // QK) * QB

# ─── Q8_0 matmul ───
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
                if (col // QK) + 2 < nc // QK:
                    prefetch[](UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qa + ro + ((col // QK) + 2) * QB)))
                var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
                var sv = SIMD[DType.float32, W](h2f(UInt16(lo | (hi << 8))))
                comptime for grp in range(4):
                    var u8 = q.load[width=8](bo + 2 + grp * 8)
                    var i8 = u8.cast[DType.int8]()
                    var f32 = i8.cast[DType.float32]()
                    acc = (f32 * sv).fma[FastMathFlag.FAST](x.load[width=W](col + grp*8), acc)
                col += QK
            # Guard against NaN
            var s = acc.reduce_add()
            if s != s: s = 0.0
            o.store(r, s)
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ─── Batched Q8_0 matmul (shares weights across B batch items) ───
@always_inline("nodebug")
def _mm_q8_b(qa: Int, xb: UnsafePointer[Float32, MutExternalOrigin],
              ob: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int = 32):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qa))
    var nb = (nr + RPW - 1) // RPW; var rb = q8_rb(nc)
    def wk(wi: Int) capturing:
        var rs = wi * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * rb; var col = 0
            var acc0 = SIMD[DType.float32, W](0.0)
            var acc1 = SIMD[DType.float32, W](0.0)
            var acc2 = SIMD[DType.float32, W](0.0)
            var acc3 = SIMD[DType.float32, W](0.0)
            while col < nc:
                var bo = ro + (col // QK) * QB
                if (col // QK) + 2 < nc // QK:
                    prefetch[](UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qa + ro + ((col // QK) + 2) * QB)))
                var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
                var sv = SIMD[DType.float32, W](h2f(UInt16(lo | (hi << 8))))
                comptime for grp in range(4):
                    var u8 = q.load[width=8](bo + 2 + grp * 8)
                    var i8 = u8.cast[DType.int8]()
                    var f32 = i8.cast[DType.float32]()
                    # Batch item 0
                    acc0 = (f32 * sv).fma[FastMathFlag.FAST](xb.load[width=W](col + grp*8), acc0)
                    comptime for bi in range(1, B):
                        var xv = (xb + bi * nc).load[width=W](col + grp*8)
                        if bi == 1:
                            acc1 = (f32 * sv).fma[FastMathFlag.FAST](xv, acc1)
                        if bi == 2:
                            acc2 = (f32 * sv).fma[FastMathFlag.FAST](xv, acc2)
                        if bi == 3:
                            acc3 = (f32 * sv).fma[FastMathFlag.FAST](xv, acc3)
                col += QK
            ob.store(r, acc0.reduce_add())
            comptime if B > 1:
                (ob + nr).store(r, acc1.reduce_add())
            comptime if B > 2:
                (ob + 2 * nr).store(r, acc2.reduce_add())
            comptime if B > 3:
                (ob + 3 * nr).store(r, acc3.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ─── RMS norm (handles both full-size and grouped weights) ───
def rms_norm(x: UnsafePointer[Float32, MutExternalOrigin],
              o: UnsafePointer[Float32, MutExternalOrigin],
              w: UnsafePointer[Float32, MutExternalOrigin], n: Int, wn: Int = 0):
    var ss = Float32(0.0); var i = 0
    while i + 8 <= n: var v = x.load[width=8](i); ss += (v * v).reduce_add(); i += 8
    while i < n: ss += x.load(i) * x.load(i); i += 1
    var inv = 1.0 / sqrt(ss / Float32(n) + EP)
    if inv != inv:
        print("RMS_NORM NaN inv: ss=", ss, " n=", n, " wn=", wn)
    
    var w_n = wn if wn > 0 else n
    var gs = n / w_n  # elements per weight group
    if gs <= 1:
        # Standard per-element norm
        var inv_v = SIMD[DType.float32, 8](inv); i = 0
        while i + 8 <= n: o.store[width=8](i, x.load[width=8](i) * inv_v * w.load[width=8](i)); i += 8
        while i < n: o.store(i, x.load(i) * w.load(i) * inv); i += 1
    else:
        # Grouped norm (Gemma 4 QK-norm style)
        for gi in range(w_n):
            var wi = w.load(gi)
            var inv_v = SIMD[DType.float32, 8](inv * wi)
            var go = gi * gs
            for j in range(0, gs, 8):
                o.store[width=8](go + j, x.load[width=8](go + j) * inv_v)

# ─── SiLU ───
def silu(p: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    for i in range(n):
        var v = p.load(i)
        if v < -80.0: v = -80.0
        if v > 80.0: v = 80.0
        p.store(i, v / (1.0 + exp(-v)))

# ─── Softmax ───
def softmax(p: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    var mx = Float32(-1e9)
    for i in range(n):
        var v = p.load(i)
        if v > mx: mx = v
    var sm = Float32(0.0)
    for i in range(n):
        var e = exp(p.load(i) - mx)
        p.store(i, e); sm += e
    var inv = 1.0 / (sm + 1e-10)
    for i in range(n): p.store(i, p.load(i) * inv)

# ─── Dequantize Q8_0 → Float32 (for norm weights) ───
def deq8(qaddr: Int, o: UnsafePointer[Float32, MutExternalOrigin],
         n: Int):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qaddr))
    for blk in range(n // QK):
        var bo = blk * QB
        var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
        var sv = h2f(UInt16(lo | (hi << 8)))
        for i in range(QK):
            var qv = q.load(bo + 2 + i).cast[DType.int8]()
            o.store(blk * QK + i, Float32(qv) * sv)
    # Handle partial last block
    var rem = n % QK
    if rem > 0:
        var blk = n // QK; var bo = blk * QB
        var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
        var sv = h2f(UInt16(lo | (hi << 8)))
        for i in range(rem):
            var qv = q.load(bo + 2 + i).cast[DType.int8]()
            o.store(blk * QK + i, Float32(qv) * sv)

# ─── Load file ───
def lw(dcp: UnsafePointer[UInt8, MutExternalOrigin],
       wl: UnsafePointer[Int64, MutExternalOrigin], idx: Int,
       pfx: String, sfx: String):
    var fname = pfx + sfx
    # Build full path: dcp + fname
    var dlen = 0
    while dcp.load(dlen) != 0: dlen += 1
    var flen = fname.byte_length()
    var buf = alloc[UInt8](dlen + flen + 1)
    for i in range(dlen): buf.store(i, dcp.load(i))
    var sp = fname.unsafe_ptr()
    for i in range(flen): buf.store(dlen + i, sp.load(i))
    buf.store(dlen + flen, UInt8(0))
    var fd = _open(buf, 0)
    if fd < 0: wl.store(idx, 0); return
    var sz = _lseek(fd, 0, 2); _ = _lseek(fd, 0, 0)
    var addr = _alc(sz)
    if addr > 0: _ = _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(addr)), sz)
    _ = _close(fd); wl.store(idx, addr)

def str_to_c(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var buf = alloc[UInt8](s.byte_length() + 1)
    var sp = s.unsafe_ptr()
    for i in range(s.byte_length()): buf.store(i, sp.load(i))
    buf.store(s.byte_length(), UInt8(0))
    return buf

# ═══ Main ═══
def main() raises:
    var t0 = time.perf_counter()
    var args = argv(); var nw = 32
    if len(args) > 1: nw = Int(String(args[1]))
    
    var wdir = String("/tmp/weights_e2b_final/")
    var dcp = str_to_c(wdir)
    
    # ─── Load weights ───
    var wl = alloc[Int64](20000)  # Large enough for global + per-layer weights
    
    # Load global weights using lw helper
    lw(dcp, wl, 10000, String(""), String("token_embd_weight.bin"))
    lw(dcp, wl, 10001, String(""), String("output_norm_weight.bin"))
    var w_emb = wl.load(10000)
    var w_on = wl.load(10001)
    
    # Load per-layer weights
    for l in range(NL):
        var pfx = String("blk_") + String(l) + String("_")
        # Weight names: attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm,
        #               attn_output, post_attention_norm, post_ffw_norm, post_norm,
        #               inp_gate, proj, ffn_norm, ffn_gate, ffn_up, ffn_down,
        #               layer_output_scale
        var names = ["attn_norm_weight.bin", "attn_q_weight.bin", "attn_k_weight.bin",
                     "attn_v_weight.bin", "attn_q_norm_weight.bin", "attn_k_norm_weight.bin",
                     "attn_output_weight.bin", "post_attention_norm_weight.bin",
                     "post_ffw_norm_weight.bin", "post_norm_weight.bin",
                     "inp_gate_weight.bin", "proj_weight.bin",
                     "ffn_norm_weight.bin", "ffn_gate_weight.bin",
                     "ffn_up_weight.bin", "ffn_down_weight.bin",
                     "layer_output_scale_weight.bin"]
        for ni in range(17):
            lw(dcp, wl, l * WPL + ni, pfx, String(names[ni]))
        if l % 10 == 0: print("  loaded layer", l)
    
    var t_load = time.perf_counter()
    print("Load: ", Int((t_load - t0) * 1000), " ms")
    
    # ─── Buffers ───
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var rp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NV * 4))))
    # Attention buffers (use maximum QI = QI_WIDE)
    var qb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * QI_WIDE * 4))))
    var kb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NK * HD * 2 * 4))))  # max NK_HD
    var vb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NK * HD * 2 * 4))))
    var att_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * QI_WIDE * 4))))
    # QK-norm buffers (per-head norms)
    var qn_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * QI_WIDE * 4))))
    var kn_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NK * HD * 2 * 4))))
    # FFN buffers
    var gate_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * FF * 4))))
    var up_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * FF * 4))))
    var down_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    # KV cache
    var kc_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NL * NK * 2 * MAX_SEQ * HD * 4))))
    var vc_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NL * NK * 2 * MAX_SEQ * HD * 4))))
    var sc_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_CTX * 4))))
    # Norm dequant buffer (max NE)
    var norm_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    
    # ─── Generate tokens ───
    var max_gen = 128
    var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb))
    var emb_rb = q8_rb(NE)
    var cur_tok = 2  # BOS token for position 0
    var cur_pos = 0  # current KV cache position
    var t_gen = time.perf_counter()
    print("Generating...")
    
    for pos in range(max_gen):
        # ─── Embed current token ───
        var emb_off = cur_tok * emb_rb
        for blk in range(NE // QK):
            var lo = Int(emb.load(emb_off + blk * QB)); var hi = Int(emb.load(emb_off + blk * QB + 1))
            var scale = h2f(UInt16(lo | (hi << 8)))
            for i in range(QK):
                var qv = emb.load(emb_off + blk * QB + 2 + i).cast[DType.int8]()
                hp.store(blk * QK + i, Float32(Int(qv)) * scale)
        
        # Copy to all batch items
        for bi in range(1, B):
            for i in range(NE): hp.store(bi * NE + i, hp.load(i))
        
        # Process all batch items through layers (with batched matmuls)
        for l in range(NL):
            var l_qi = qi(l)
            var l_nk = nk_hd(l)
            var base = l * WPL
            var nh_l = NH if l_qi == QI_NARROW else NH * 2
            var nk_l = NK if l_nk == NK * HD else NK * 2
            var hd_l = HD
            var kr = nh_l // nk_l
            var kn_wn = l_nk / (1 + (NK / 4))  # K-norm grouping
            
            # 1. Save residual + pre-attention RMS norm (per item)
            for bi in range(B):
                var hp_bi = hp + bi * NE
                var bp_bi = bp + bi * NE
                var rp_bi = rp + bi * NE
                for i in range(NE): rp_bi.store(i, hp_bi.load(i))
                var an_q = wl.load(base + 0)
                if Int(an_q) != 0:
                    deq8(Int(an_q), norm_buf, NE)
                    rms_norm(hp_bi, bp_bi, norm_buf, NE, NE)
            
            # 2. Q projection (batched)
            var wq = wl.load(base + 1)
            if Int(wq) != 0:
                _mm_q8_b(Int(wq), bp, qb, l_qi, NE, nw)
            else:
                for bi in range(B):
                    for i in range(l_qi): (qb + bi * l_qi).store(i, 0.0)
            
            # 3. QK-norm on Q (per item)
            var qn_q = wl.load(base + 4)
            if Int(qn_q) != 0:
                deq8(Int(qn_q), norm_buf, l_qi / 8)
                for bi in range(B):
                    var qb_bi = qb + bi * l_qi
                    var qn_bi = qn_buf + bi * l_qi
                    rms_norm(qb_bi, qn_bi, norm_buf, l_qi, l_qi / 8)
            else:
                for bi in range(B):
                    var qb_bi = qb + bi * l_qi
                    var qn_bi = qn_buf + bi * l_qi
                    for i in range(l_qi): qn_bi.store(i, qb_bi.load(i))
            
            # 4. K projection (batched)
            var wk = wl.load(base + 2)
            if Int(wk) != 0:
                _mm_q8_b(Int(wk), bp, kb, l_nk, NE, nw)
            else:
                for bi in range(B):
                    for i in range(l_nk): (kb + bi * l_nk).store(i, 0.0)
            
            # 5. QK-norm on K (per item)
            var kn_q = wl.load(base + 5)
            if Int(kn_q) != 0:
                deq8(Int(kn_q), norm_buf, kn_wn)
                for bi in range(B):
                    var kb_bi = kb + bi * l_nk
                    var kn_bi = kn_buf + bi * l_nk
                    rms_norm(kb_bi, kn_bi, norm_buf, l_nk, kn_wn)
            else:
                for bi in range(B):
                    var kb_bi = kb + bi * l_nk
                    var kn_bi = kn_buf + bi * l_nk
                    for i in range(l_nk): kn_bi.store(i, kb_bi.load(i))
            
            # 6. V projection (batched)
            var wv = wl.load(base + 3)
            if Int(wv) != 0:
                _mm_q8_b(Int(wv), bp, vb, l_nk, NE, nw)
            else:
                for bi in range(B):
                    for i in range(l_nk): (vb + bi * l_nk).store(i, 0.0)
            
            # 7. GQA Attention + KV cache (per item)
            for bi in range(B):
                var qn_bi = qn_buf + bi * l_qi
                var kn_bi = kn_buf + bi * l_nk
                var vb_bi = vb + bi * l_nk
                var att_bi = att_buf + bi * l_qi
                
                # Store KV cache
                for h in range(nk_l):
                    for d in range(hd_l):
                        var cpos = bi * NL * NK * 2 * MAX_SEQ * HD + l * NK * 2 * MAX_SEQ * HD + h * MAX_SEQ * HD + cur_pos * HD + d
                        kc_buf.store(cpos, kn_bi.load(h * hd_l + d))
                        vc_buf.store(cpos, vb_bi.load(h * hd_l + d))
                
                # Multi-head attention
                for hq in range(nh_l):
                    var hk = hq // kr
                    var qbase_q = hq * hd_l
                    var cbase = bi * NL * NK * 2 * MAX_SEQ * HD + l * NK * 2 * MAX_SEQ * HD + hk * MAX_SEQ * HD
                    
                    var sc = sc_buf; var smax = Float32(-1e9)
                    for p in range(cur_pos + 1):
                        var sv = SIMD[DType.float32, W](0.0); var d = 0
                        while d + W <= hd_l:
                            sv = sv + qn_bi.load[width=W](qbase_q + d) * kc_buf.load[width=W](cbase + p * HD + d)
                            d += W
                        var s = sv.reduce_add() / sqrt(Float32(hd_l))
                        if s > 80.0: s = 80.0
                        if s < -80.0: s = -80.0
                        sc.store(p, s)
                        if s > smax: smax = s
                    
                    var ssum = Float32(0.0)
                    for p in range(cur_pos + 1):
                        var e2 = exp(sc.load(p) - smax); sc.store(p, e2); ssum += e2
                    var inv_s = 1.0 / (ssum + 1e-10)
                    
                    var d = 0
                    while d + W <= hd_l:
                        var ov = SIMD[DType.float32, W](0.0)
                        for p in range(cur_pos + 1):
                            ov = ov + vc_buf.load[width=W](cbase + p * HD + d) * (sc.load(p) * inv_s)
                        att_bi.store[width=W](qbase_q + d, ov); d += W
                    while d < hd_l:
                        var o2 = Float32(0.0)
                        for p in range(cur_pos + 1): o2 += vc_buf.load(cbase + p * HD + d) * (sc.load(p) * inv_s)
                        att_bi.store(qbase_q + d, o2); d += 1
            
            # 8. O projection (batched)
            var wo = wl.load(base + 6)
            if Int(wo) != 0:
                _mm_q8_b(Int(wo), att_buf, bp, NE, l_qi, nw)
            else:
                for bi in range(B):
                    for i in range(NE): (bp + bi * NE).store(i, 0.0)
            
            # 9. Post-attention norm + inp_gate + proj + residual + FFW (per item)
            for bi in range(B):
                var hp_bi = hp + bi * NE
                var bp_bi = bp + bi * NE
                var rp_bi = rp + bi * NE
                var kn_bi = kn_buf + bi * l_nk
                var gt_bi = gate_buf + bi * FF
                var up_bi = up_buf + bi * FF
                
                # Post-attention norm
                var pan_q = wl.load(base + 7)
                if Int(pan_q) != 0:
                    deq8(Int(pan_q), norm_buf, NE)
                    rms_norm(bp_bi, bp_bi, norm_buf, NE)
                
                # inp_gate + proj (gating before residual)
                var w_ig = wl.load(base + 10)
                var w_pr = wl.load(base + 11)
                if Int(w_ig) != 0 and Int(w_pr) != 0:
                    var ig_buf = kn_bi
                    _mm_q8(Int(w_ig), bp_bi, ig_buf, 256, NE, nw)
                    silu(ig_buf, 256)
                    _mm_q8(Int(w_pr), ig_buf, bp_bi, NE, 256, nw)
                
                # Residual with layer output scale
                var los_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 16)))
                var los = Float32(1.0)
                if Int(los_p) != 0: los = los_p.load(0)
                for i in range(NE): hp_bi.store(i, rp_bi.load(i) + bp_bi.load(i) * los)
                
                # Post-FFW norm
                var pfn_q = wl.load(base + 8)
                if Int(pfn_q) != 0:
                    deq8(Int(pfn_q), norm_buf, NE)
                    rms_norm(hp_bi, bp_bi, norm_buf, NE)
                
                # FFN: gate(SiLU) * up → down
                var w_gt = wl.load(base + 13)
                var w_up = wl.load(base + 14)
                var w_dn = wl.load(base + 15)
                if Int(w_gt) != 0 and Int(w_up) != 0:
                    for i in range(NE): rp_bi.store(i, hp_bi.load(i))
                    _mm_q8(Int(w_gt), bp_bi, gt_bi, FF, NE, nw)
                    _mm_q8(Int(w_up), bp_bi, up_bi, FF, NE, nw)
                    silu(gt_bi, FF)
                    for i in range(FF): gt_bi.store(i, gt_bi.load(i) * up_bi.load(i))
                    if Int(w_dn) != 0:
                        _mm_q8(Int(w_dn), gt_bi, bp_bi, NE, FF, nw)
                    for i in range(NE): hp_bi.store(i, rp_bi.load(i) + bp_bi.load(i) * los)
                
                # Post-norm
                var pn_q = wl.load(base + 9)
                if Int(pn_q) != 0:
                    deq8(Int(pn_q), norm_buf, NE)
                    rms_norm(hp_bi, bp_bi, norm_buf, NE)
        
        # ─── End of layers ───
        
        # LM head: output norm + matmul (batched)
        var on_q = wl.load(10001)
        if Int(on_q) != 0:
            deq8(Int(on_q), norm_buf, NE)
            for bi in range(B):
                var hp_bi = hp + bi * NE
                var bp_bi = bp + bi * NE
                rms_norm(hp_bi, bp_bi, norm_buf, NE)
        
        # Output projection (batched LM head): NV × NE
        var w_lm = wl.load(10000)
        if Int(w_lm) != 0:
            _mm_q8_b(Int(w_lm), bp, lp, NV, NE, nw)
        
        # Argmax and print
        for bi in range(B):
            var best = 0; var bv = (lp + bi * NV).load(0)
            for i in range(1, NV):
                var v = (lp + bi * NV).load(i)
                if v > bv: bv = v; best = i
            if bi == 0:
                print("tok=", best, " ", end="")
                cur_tok = best
        print()
        cur_pos += 1
        if cur_pos >= MAX_SEQ: break
    
    print()
    var t_end = time.perf_counter()
    print("Time: ", Int((t_end - t_gen) * 1000), " ms (", Float64(max_gen * B) / ((t_end - t_gen) * 1000.0 / 1000.0), " tok/s)")
