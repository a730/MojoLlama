# gemma4_gen_q8.mojo — Gemma 4 inference engine (pure Mojo, Q8_0 weights)
# ARCH: Gemma-4-E4B: NE=2560, wide_QI=4096, narrow_QI=2048, HD=128
#   Wide layers (every 6th): 5, 11, 17, 23, 29, 35, 41
#   NK_HD: 512 narrow, 1024 wide
#   FF=10240, NL=42, NV=262144
#
# FEATURES:
#   - QK-norm (separate RMS norm for Q and K)
#   - inp_gate + proj (gating before attention)
#   - Post-norms (attn + ffw + final)
#   - Layer output scaling
#   - Per-layer variable Q/K/O dimensions
#
# Also supports E2B (NE=2048, FF=16384), 26B-A4B (NE=2816, MoE), 31B (NE=3072)
# via comptime NE, NH, NK, HD, NL, FF, NV parameters.
from std import time
from std.sys import argv
from std.math import sqrt, exp, cos, sin, pow
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag
from std.sys.intrinsics import prefetch

# ─── Architecture (edit for each model) ───
comptime NE: Int = 2560    # n_embd
comptime NH: Int = 16      # n_heads (narrow)
comptime NK: Int = 4       # n_kv_heads (narrow)
comptime HD: Int = 128     # head_dim
comptime NL: Int = 42      # n_layers
comptime FF: Int = 10240   # ffn_hidden
comptime NV: Int = 262144  # vocab_size
comptime QI_NARROW: Int = NH * HD    # 2048
comptime QI_WIDE: Int = NH * HD * 2  # 4096 (wide layers)
comptime WIDE_INTERVAL: Int = 6      # every 6th layer is wide
comptime W: Int = 8;  comptime RPW: Int = 8;  comptime B: Int = 4
comptime QK: Int = 32;  comptime QB: Int = 34;  comptime EP: Float32 = 1e-5
comptime MAX_SEQ: Int = 128;  comptime MAX_CTX: Int = 4096

# Weight slots per layer
comptime WPL: Int = 18

# ─── Helpers ───
def is_wide(l: Int) -> Bool: return l % WIDE_INTERVAL == 5
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
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ─── RMS norm ───
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
    
    var wdir = String("/tmp/weights_e4b_q8_v2/")
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
    
    # ─── Embed BOS token 2 ───
    var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb))
    var emb_rb = q8_rb(NE)
    var bos_tok = 2  # Gemma 4 BOS token
    var emb_off = bos_tok * emb_rb
    for blk in range(NE // QK):
        var lo = Int(emb.load(emb_off + blk * QB)); var hi = Int(emb.load(emb_off + blk * QB + 1))
        var scale = h2f(UInt16(lo | (hi << 8)))
        for i in range(QK):
            var qv = Int(emb.load(emb_off + blk * QB + 2 + i).cast[DType.int8]())
            hp.store(blk * QK + i, Float32(qv) * scale)
    
    # Copy to all batch items
    for bi in range(1, B):
        for i in range(NE): hp.store(bi * NE + i, hp.load(i))
    
    var cur_pos = 0  # current KV cache position
    var t_gen = time.perf_counter()
    print("Generating...")
    
    # ─── Generate tokens ───
    var max_gen = 4
    for pos in range(max_gen):
        # Process each batch item through all layers
        for bi in range(B):
            var hp_bi = hp + bi * NE
            var bp_bi = bp + bi * NE
            var rp_bi = rp + bi * NE
            var qb_bi = qb + bi * qi(0)  # use max QI
            var kb_bi = kb + bi * nk_hd(0)
            var vb_bi = vb + bi * nk_hd(0)
            var att_bi = att_buf + bi * qi(0)
            var qn_bi = qn_buf + bi * qi(0)
            var kn_bi = kn_buf + bi * nk_hd(0)
            var gt_bi = gate_buf + bi * FF
            var up_bi = up_buf + bi * FF
            var dn_bi = down_buf + bi * NE
            
            for l in range(NL):
                var l_qi = qi(l)
                var l_nk = nk_hd(l)
                var base = l * WPL
                
                # Save residual
                for i in range(NE): rp_bi.store(i, hp_bi.load(i))
                
                if l == 0 and pos == 0 and bi == 0:
                    print("hp[0]=", hp_bi.load(0), " hp[10]=", hp_bi.load(10))
                
                # 1. Pre-attention RMS norm
                var attn_norm_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 0)))
                if Int(attn_norm_p) != 0: rms_norm(hp_bi, bp_bi, attn_norm_p, NE)
                
                # 2. Q projection + QK-norm
                var wq = wl.load(base + 1)
                if Int(wq) != 0:
                    _mm_q8(Int(wq), bp_bi, qb_bi, l_qi, NE, nw)
                    var qn_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 4)))
                    if Int(qn_p) != 0: rms_norm(qb_bi, qn_bi, qn_p, l_qi)
                    else:
                        for i in range(l_qi): qn_bi.store(i, qb_bi.load(i))
                else:
                    for i in range(l_qi): qn_bi.store(i, 0.0)
                
                # 3. K projection + QK-norm
                var wk = wl.load(base + 2)
                if Int(wk) != 0:
                    _mm_q8(Int(wk), bp_bi, kb_bi, l_nk, NE, nw)
                    var kn_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 5)))
                    if Int(kn_p) != 0: rms_norm(kb_bi, kn_bi, kn_p, l_nk)
                    else:
                        for i in range(l_nk): kn_bi.store(i, kb_bi.load(i))
                else:
                    for i in range(l_nk): kn_bi.store(i, 0.0)
                
                # 4. V projection
                var wv = wl.load(base + 3)
                if Int(wv) != 0: _mm_q8(Int(wv), bp_bi, vb_bi, l_nk, NE, nw)
                else:
                    for i in range(l_nk): vb_bi.store(i, 0.0)
                
                # 5. GQA Attention (simplified: per-head dot product)
                var nh_l = NH if l_qi == QI_NARROW else NH * 2
                var nk_l = NK if l_nk == NK * HD else NK * 2
                var hd_l = HD
                var kr = nh_l // nk_l
                
                for hq in range(nh_l):
                    var hk = hq // kr
                    var qbase = hq * hd_l
                    var kbase = hk * hd_l
                    var cbase = bi * NL * NK * 2 * MAX_SEQ * HD + l * NK * 2 * MAX_SEQ * HD + hk * MAX_SEQ * HD
                    
                    # Score = Q·K / sqrt(HD)
                    var sc = sc_buf
                    var smax = Float32(-1e9)
                    for p in range(cur_pos + 1):
                        var sv = SIMD[DType.float32, W](0.0); var d = 0
                        while d + W <= hd_l:
                            sv = sv + qn_bi.load[width=W](qbase + d) * kc_buf.load[width=W](cbase + p * HD + d)
                            d += W
                        var s = sv.reduce_add() / sqrt(Float32(hd_l))
                        # Clamp score to prevent softmax overflow
                        if s > 80.0: s = 80.0
                        if s < -80.0: s = -80.0
                        sc.store(p, s)
                        if s > smax: smax = s
                    
                    # Softmax
                    var ssum = Float32(0.0)
                    for p in range(cur_pos + 1):
                        var e2 = exp(sc.load(p) - smax); sc.store(p, e2); ssum += e2
                    var inv = 1.0 / (ssum + 1e-10)
                    
                    # Weighted sum of V
                    var d = 0
                    while d + W <= hd_l:
                        var ov = SIMD[DType.float32, W](0.0)
                        for p in range(cur_pos + 1):
                            ov = ov + vc_buf.load[width=W](cbase + p * HD + d) * (sc.load(p) * inv)
                        att_bi.store[width=W](qbase + d, ov); d += W
                    while d < hd_l:
                        var o2 = Float32(0.0)
                        for p in range(cur_pos + 1): o2 += vc_buf.load(cbase + p * HD + d) * (sc.load(p) * inv)
                        att_bi.store(qbase + d, o2); d += 1
                
                # 6. Store KV cache
                for h in range(nk_l):
                    for d in range(hd_l):
                        var cpos = bi * NL * NK * 2 * MAX_SEQ * HD + l * NK * 2 * MAX_SEQ * HD + h * MAX_SEQ * HD + cur_pos * HD + d
                        kc_buf.store(cpos, kn_bi.load(h * hd_l + d))
                        vc_buf.store(cpos, vb_bi.load(h * hd_l + d))
                
                # 7. O projection
                var wo = wl.load(base + 6)
                if Int(wo) != 0: _mm_q8(Int(wo), att_bi, bp_bi, NE, l_qi, nw)
                else:
                    for i in range(NE): bp_bi.store(i, 0.0)
                
                # 8. Post-attention norm
                var pan_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 7)))
                if Int(pan_p) != 0: rms_norm(bp_bi, bp_bi, pan_p, NE)
                
                # 9. inp_gate + proj (gating before residual)
                var w_ig = wl.load(base + 10)  # inp_gate: NE → 256
                var w_pr = wl.load(base + 11)  # proj: 256 → NE
                if Int(w_ig) != 0 and Int(w_pr) != 0:
                    var ig_buf = kn_bi  # reuse K buffer as temp (256)
                    _mm_q8(Int(w_ig), bp_bi, ig_buf, 256, NE, nw)
                    silu(ig_buf, 256)
                    _mm_q8(Int(w_pr), ig_buf, bp_bi, NE, 256, nw)
                
                # 10. Residual with layer output scale
                var los_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 16)))
                var los = Float32(1.0)
                if Int(los_p) != 0: los = los_p.load(0)
                for i in range(NE): hp_bi.store(i, rp_bi.load(i) + bp_bi.load(i) * los)
                
                # 11. Post-FFW norm
                var pfn_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 8)))
                if Int(pfn_p) != 0: rms_norm(hp_bi, bp_bi, pfn_p, NE)
                
                # 12. FFN: gate(SiLU) * up → down
                var w_gt = wl.load(base + 13)
                var w_up = wl.load(base + 14)
                var w_dn = wl.load(base + 15)
                if Int(w_gt) != 0 and Int(w_up) != 0:
                    for i in range(NE): rp_bi.store(i, hp_bi.load(i))  # save residual
                    _mm_q8(Int(w_gt), bp_bi, gt_bi, FF, NE, nw)
                    _mm_q8(Int(w_up), bp_bi, up_bi, FF, NE, nw)
                    silu(gt_bi, FF)
                    for i in range(FF): gt_bi.store(i, gt_bi.load(i) * up_bi.load(i))
                    if Int(w_dn) != 0:
                        _mm_q8(Int(w_dn), gt_bi, bp_bi, NE, FF, nw)
                    # FFN residual with output scale
                    for i in range(NE): hp_bi.store(i, rp_bi.load(i) + bp_bi.load(i) * los)
                
                # 13. Post-norm
                var pn_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 9)))
                if Int(pn_p) != 0: rms_norm(hp_bi, bp_bi, pn_p, NE)
            
            # End of layers
            
            # LM head: output norm + matmul
            var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w_on))
            if Int(onp) != 0: rms_norm(hp_bi, bp_bi, onp, NE)
            
            # Output projection (LM head): NV × NE
            var w_lm = wl.load(10000)  # token_embd.weight (tied embeddings)
            if Int(w_lm) != 0:
                _mm_q8(Int(w_lm), bp_bi, lp + bi * NV, NV, NE, nw)
            
            # Argmax
            var best = 0; var bv = (lp + bi * NV).load(0)
            for i in range(1, NV):
                var v = (lp + bi * NV).load(i)
                if v > bv: bv = v; best = i
            
            if bi == 0 and pos == 0:
                print("logits[0]=", (lp + bi * NV).load(0), " logits[10]=", (lp + bi * NV).load(10), " logits[100]=", (lp + bi * NV).load(100))
            
            if bi == 0:
                print("tok=", best, " ", end="")
        
        cur_pos += 1
        if cur_pos >= MAX_SEQ: break
    
    print()
    var t_end = time.perf_counter()
    print("Time: ", Int((t_end - t_gen) * 1000), " ms (", Float64(max_gen * B) / ((t_end - t_gen) * 1000.0 / 1000.0), " tok/s)")
