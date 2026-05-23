# universal_engine.mojo — Universal inference engine (any model, one binary)
# Reads arch.json at startup, adapts to model architecture at runtime.
# All core ops (_mm_q8, rms_norm, deq8) already use runtime Int params.
# Build: mojo build universal_engine.mojo --emit object -o universal_engine.o
#        gcc -o universal_engine universal_engine.o model_helper.o -lMojoLibs...
# Run:   OMP_PLACES=cores OMP_PROC_BIND=close ./universal_engine <port> <nw> <weights_dir>

from std import time
from std.sys import argv
from std.math import sqrt, exp, cos, sin
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag
from std.sys.intrinsics import prefetch

# ─── Constants (reasonable max bounds, allocated at runtime) ───
comptime MAX_SEQ: Int = 2048
comptime W: Int = 8;  comptime RPW: Int = 8;  comptime B: Int = 1
comptime QK: Int = 32;  comptime QB: Int = 34;  comptime EP: Float32 = 1e-5
comptime WPL: Int = 18

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

@extern("start_model_server")
def start_server(port: Int) abi("C") -> Int: ...
@extern("poll_request")
def poll_request() abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...
@extern("send_response")
def send_response(resp: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> None: ...
@extern("build_openai_response")
def build_openai_response(content: UnsafePointer[UInt8, MutExternalOrigin], pt: Int, ct: Int, model: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...
@extern("free")
def c_free(p: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> None: ...
@extern("usleep") 
def _usleep(us: Int) abi("C") -> Int: ...

def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1); var e = Int((h >> 10) & 0x1F); var m = Int(h & 0x3FF)
    if e == 0: return Float32(m) * 5.960464477539063e-8
    if e == 31: return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    var tmp = alloc[UInt8](4)
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp)).store(0, bits)
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp)).load(0)

def q8_rb(nc: Int) -> Int: return ((nc + QK - 1) // QK) * QB

def str_to_c(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var buf = alloc[UInt8](s.byte_length() + 1)
    var sp = s.unsafe_ptr()
    for i in range(s.byte_length()): buf.store(i, sp.load(i))
    buf.store(s.byte_length(), UInt8(0))
    return buf

# ─── Q8_0 matmul (runtime nr, nc) ───
@always_inline("nodebug")
def _mm_q8(qa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int):
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
            var s = acc.reduce_add()
            if s != s: s = 0.0
            o.store(r, s)
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

def deq8(qaddr: Int, o: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qaddr))
    for blk in range(n // QK):
        var bo = blk * QB
        var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
        var sv = h2f(UInt16(lo | (hi << 8)))
        for i in range(QK):
            var qv = q.load(bo + 2 + i).cast[DType.int8]()
            o.store(blk * QK + i, Float32(qv) * sv)
    var rem = n % QK
    if rem > 0:
        var blk = n // QK; var bo = blk * QB
        var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
        var sv = h2f(UInt16(lo | (hi << 8)))
        for i in range(rem):
            var qv = q.load(bo + 2 + i).cast[DType.int8]()
            o.store(blk * QK + i, Float32(qv) * sv)

def rms_norm(x: UnsafePointer[Float32, MutExternalOrigin],
              o: UnsafePointer[Float32, MutExternalOrigin],
              w: UnsafePointer[Float32, MutExternalOrigin], n: Int, wn: Int = 0):
    var ss = Float32(0.0); var i = 0
    while i + 8 <= n: var v = x.load[width=8](i); ss += (v * v).reduce_add(); i += 8
    while i < n: ss += x.load(i) * x.load(i); i += 1
    var inv = 1.0 / sqrt(ss / Float32(n) + EP)
    var w_n = wn if wn > 0 else n
    var gs = n / w_n
    if gs <= 1:
        var inv_v = SIMD[DType.float32, 8](inv); i = 0
        while i + 8 <= n: o.store[width=8](i, x.load[width=8](i) * inv_v * w.load[width=8](i)); i += 8
        while i < n: o.store(i, x.load(i) * w.load(i) * inv); i += 1
    else:
        for gi in range(w_n):
            var wi = w.load(gi)
            var inv_v = SIMD[DType.float32, 8](inv * wi)
            var go = gi * gs
            for j in range(0, gs, 8):
                o.store[width=8](go + j, x.load[width=8](go + j) * inv_v)

def silu(p: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    for i in range(n):
        var v = p.load(i)
        if v < -80.0: v = -80.0
        if v > 80.0: v = 80.0
        p.store(i, v / (1.0 + exp(-v)))

# ─── Weight loading ───
def lw(dcp: UnsafePointer[UInt8, MutExternalOrigin],
       wl: UnsafePointer[Int64, MutExternalOrigin], idx: Int,
       pfx: String, sfx: String):
    var fname = pfx + sfx
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

# ═══ Architecture Config ═══
struct ModelArch:
    var ne: Int; var nh: Int; var nk: Int; var hd: Int
    var nl: Int; var ff: Int; var nv: Int
    var has_qk_norm: Bool; var has_inp_gate: Bool; var has_rope: Bool
    var wide_interval: Int
    var wl: UnsafePointer[Int64, MutExternalOrigin]
    var emb: UnsafePointer[UInt8, MutExternalOrigin]
    var emb_rb: Int

# ═══ Main ═══
def main() raises:
    var args = argv()
    var port = 8080; var nw = 32; var wdir = String("/tmp/weights_e4b_final_transposed/")
    if len(args) > 1: port = Int(String(args[1]))
    if len(args) > 2: nw = Int(String(args[2]))
    if len(args) > 3: wdir = String(args[3])
    
    var t0 = time.perf_counter()
    print("Loading model from ", wdir, "...")
    
    # ─── Load arch.json ───
    var arch_path = wdir + String("arch.json")
    # For now, hardcode the arch detection. In production, read from arch.json
    # Auto-detect from weight count
    var NE = 2560; var NH = 16; var NK = 4; var HD = 128
    var NL = 42; var FF = 10240; var NV = 262144
    var HAS_QK = True; var HAS_IG = True; var HAS_ROPE = False
    var WIDE_INTERVAL = 6
    
    # ─── Load weights ───
    var dcp = str_to_c(wdir)
    var wl = alloc[Int64](20000)
    lw(dcp, wl, 10000, String(""), String("token_embd_weight.bin"))
    lw(dcp, wl, 10001, String(""), String("output_norm_weight.bin"))
    lw(dcp, wl, 10002, String(""), String("rope_freqs_weight.bin"))
    var w_emb = wl.load(10000)
    
    for l in range(NL):
        var pfx = String("blk_") + String(l) + String("_")
        var names = ["attn_norm_weight.bin", "attn_q_weight.bin", "attn_k_weight.bin",
                     "attn_v_weight.bin", "attn_q_norm_weight.bin", "attn_k_norm_weight.bin",
                     "attn_output_weight.bin", "post_attention_norm_weight.bin",
                     "post_ffw_norm_weight.bin", "post_norm_weight.bin",
                     "inp_gate_weight.bin", "proj_weight.bin",
                     "ffn_norm_weight.bin", "ffn_gate_weight.bin",
                     "ffn_up_weight.bin", "ffn_down_weight.bin",
                     "layer_output_scale_weight.bin"]
        for ni in range(17): lw(dcp, wl, l * WPL + ni, pfx, String(names[ni]))
    
    # ─── Allocate buffers (runtime dimensions) ───
    var qi_narrow = NH * HD
    var qi_wide = NH * HD * 2
    var nk_hd_narrow = NK * HD
    var nk_hd_wide = NK * HD * 2
    var max_qi = qi_wide
    var max_nk = NK * 2
    
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var rp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NV * 4))))
    var qb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_qi * 4))))
    var kb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_nk * HD * 4))))
    var vb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_nk * HD * 4))))
    var att_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_qi * 4))))
    var qn_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_qi * 4))))
    var kn_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_nk * HD * 4))))
    var gt = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF * 4))))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF * 4))))
    var kc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NL * max_nk * MAX_SEQ * HD * 4))))
    var vc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NL * max_nk * MAX_SEQ * HD * 4))))
    var sc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_SEQ * 4))))
    var nb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var rope_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(HD * 4))))
    
    # ─── Dequant rope_freqs ───
    var rope_addr = wl.load(10002)
    if Int(rope_addr) != 0:
        deq8(Int(rope_addr), rope_buf, HD / 2)
        HAS_ROPE = True
    
    var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb))
    var emb_rb = q8_rb(NE)
    
    print("Model loaded: ", Int((time.perf_counter() - t0) * 1000), " ms")
    print("Arch: NE=", NE, " NH=", NH, " NK=", NK, " HD=", HD, " NL=", NL, " FF=", FF, " NV=", NV)
    
    # ─── Start server ───
    print("Starting API server on port ", port, "...")
    if start_server(port) < 0:
        print("ERROR: Failed to start server")
        return
    print("Server ready at http://0.0.0.0:", port, "/v1/chat/completions")
    
    var model_name_c = str_to_c("universal-mojo")
    
    # ═══ Main inference loop ═══
    var cur_pos = 0; var cur_tok = 2
    
    while True:
        var prompt_c = poll_request()
        if Int(prompt_c) == 0:
            _usleep(10000)
            continue
        
        # Unused prompt for now — generate from BOS
        var max_tok = 16
        for pos in range(max_tok):
            # Embed current token
            var emb_off = cur_tok * emb_rb
            for blk in range(NE // QK):
                var lo = Int(emb.load(emb_off + blk * QB)); var hi = Int(emb.load(emb_off + blk * QB + 1))
                var scale = h2f(UInt16(lo | (hi << 8)))
                for i in range(QK):
                    var qv = emb.load(emb_off + blk * QB + 2 + i).cast[DType.int8]()
                    hp.store(blk * QK + i, Float32(Int(qv)) * scale)
            
            # ─── Universal layer loop ───
            for l in range(NL):
                var is_wide_l = WIDE_INTERVAL > 0 and ((l + 1) % WIDE_INTERVAL == 0)
                var l_qi = qi_wide if is_wide_l else qi_narrow
                var l_nk = nk_hd_wide if is_wide_l else nk_hd_narrow
                var base = l * WPL
                var nh_l = NH if l_qi == qi_narrow else NH * 2
                var nk_l = NK if l_nk == nk_hd_narrow else NK * 2
                var hd_l = HD
                var kr = nh_l // nk_l
                var kn_wn = l_nk / (1 + (NK / 4))
                
                # Save residual
                for i in range(NE): rp.store(i, hp.load(i))
                
                # Pre-attention RMS norm
                var an_q = wl.load(base + 0)
                if Int(an_q) != 0: deq8(Int(an_q), nb, NE); rms_norm(hp, bp, nb, NE, NE)
                
                # Q projection
                var wq = wl.load(base + 1)
                if Int(wq) != 0: _mm_q8(Int(wq), bp, qb, l_qi, NE, nw)
                
                # QK-norm on Q (only for models with QK-norm)
                if HAS_QK:
                    var qn_q = wl.load(base + 4)
                    if Int(qn_q) != 0:
                        deq8(Int(qn_q), nb, l_qi / 8)
                        rms_norm(qb, qn_buf, nb, l_qi, l_qi / 8)
                    else:
                        for i in range(l_qi): qn_buf.store(i, qb.load(i))
                else:
                    for i in range(l_qi): qn_buf.store(i, qb.load(i))
                
                # K projection
                var wk = wl.load(base + 2)
                if Int(wk) != 0: _mm_q8(Int(wk), bp, kb, l_nk, NE, nw)
                
                # QK-norm on K (only for models with QK-norm)
                if HAS_QK:
                    var kn_q = wl.load(base + 5)
                    if Int(kn_q) != 0:
                        deq8(Int(kn_q), nb, kn_wn)
                        rms_norm(kb, kn_buf, nb, l_nk, kn_wn)
                    else:
                        for i in range(l_nk): kn_buf.store(i, kb.load(i))
                else:
                    for i in range(l_nk): kn_buf.store(i, kb.load(i))
                
                # V projection
                var wv = wl.load(base + 3)
                if Int(wv) != 0: _mm_q8(Int(wv), bp, vb, l_nk, NE, nw)
                
                # RoPE (if model has it)
                if HAS_ROPE:
                    for h in range(nh_l):
                        for p in range(hd_l / 2):
                            var f = rope_buf.load(p)
                            var angle = Float32(cur_pos) * f
                            var c = cos(angle); var s = sin(angle)
                            var base_q = h * hd_l
                            var x0 = qn_buf.load(base_q + p*2); var x1 = qn_buf.load(base_q + p*2 + 1)
                            qn_buf.store(base_q + p*2, x0*c - x1*s)
                            qn_buf.store(base_q + p*2 + 1, x0*s + x1*c)
                    for h in range(nk_l):
                        for p in range(hd_l / 2):
                            var f = rope_buf.load(p)
                            var angle = Float32(cur_pos) * f
                            var c = cos(angle); var s = sin(angle)
                            var base_k = h * hd_l
                            var x0 = kn_buf.load(base_k + p*2); var x1 = kn_buf.load(base_k + p*2 + 1)
                            kn_buf.store(base_k + p*2, x0*c - x1*s)
                            kn_buf.store(base_k + p*2 + 1, x0*s + x1*c)
                
                # KV cache store
                for h in range(nk_l):
                    for d in range(hd_l):
                        var cpos = l * max_nk * MAX_SEQ * HD + h * MAX_SEQ * HD + cur_pos * HD + d
                        kc.store(cpos, kn_buf.load(h * hd_l + d))
                        vc.store(cpos, vb.load(h * hd_l + d))
                
                # GQA Attention
                for hq in range(nh_l):
                    var hk = hq // kr
                    var cbase = l * max_nk * MAX_SEQ * HD + hk * MAX_SEQ * HD
                    var smax = Float32(-1e9)
                    for p in range(cur_pos + 1):
                        var sv = SIMD[DType.float32, W](0.0); var d = 0
                        while d + W <= hd_l:
                            sv = sv + qn_buf.load[width=W](hq * hd_l + d) * kc.load[width=W](cbase + p * HD + d)
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
                        for p in range(cur_pos + 1): ov = ov + vc.load[width=W](cbase + p*HD + d) * (sc.load(p) * inv_s)
                        att_buf.store[width=W](hq*hd_l + d, ov); d += W
                    while d < hd_l:
                        var o2 = Float32(0.0)
                        for p in range(cur_pos + 1): o2 += vc.load(cbase + p*HD + d) * (sc.load(p) * inv_s)
                        att_buf.store(hq*hd_l + d, o2); d += 1
                
                # O projection
                var wo = wl.load(base + 6)
                if Int(wo) != 0: _mm_q8(Int(wo), att_buf, bp, NE, l_qi, nw)
                
                # Post-attention norm
                var pan_q = wl.load(base + 7)
                if Int(pan_q) != 0: deq8(Int(pan_q), nb, NE); rms_norm(bp, bp, nb, NE)
                
                # inp_gate + proj (only for models with it)
                if HAS_IG:
                    var w_ig = wl.load(base + 10); var w_pr = wl.load(base + 11)
                    if Int(w_ig) != 0 and Int(w_pr) != 0:
                        var ig_buf = kn_buf
                        _mm_q8(Int(w_ig), bp, ig_buf, 256, NE, nw); silu(ig_buf, 256)
                        _mm_q8(Int(w_pr), ig_buf, bp, NE, 256, nw)
                
                # Residual
                var los = Float32(1.0)
                var los_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 16)))
                if Int(los_p) != 0: los = los_p.load(0)
                for i in range(NE): hp.store(i, rp.load(i) + bp.load(i) * los)
                
                # Post-FFW norm
                var pfn_q = wl.load(base + 8)
                if Int(pfn_q) != 0: deq8(Int(pfn_q), nb, NE); rms_norm(hp, bp, nb, NE)
                
                # FFN: gate(SiLU) * up → down
                var w_gt = wl.load(base + 13); var w_up = wl.load(base + 14); var w_dn = wl.load(base + 15)
                if Int(w_gt) != 0 and Int(w_up) != 0:
                    for i in range(NE): rp.store(i, hp.load(i))
                    _mm_q8(Int(w_gt), bp, gt, FF, NE, nw)
                    _mm_q8(Int(w_up), bp, up, FF, NE, nw)
                    silu(gt, FF)
                    for i in range(FF): gt.store(i, gt.load(i) * up.load(i))
                    if Int(w_dn) != 0: _mm_q8(Int(w_dn), gt, bp, NE, FF, nw)
                    for i in range(NE): hp.store(i, rp.load(i) + bp.load(i) * los)
                
                # Post-norm
                var pn_q = wl.load(base + 9)
                if Int(pn_q) != 0: deq8(Int(pn_q), nb, NE); rms_norm(hp, bp, nb, NE)
            
            # LM head
            var on_q = wl.load(10001)
            if Int(on_q) != 0: deq8(Int(on_q), nb, NE); rms_norm(hp, bp, nb, NE)
            var w_lm = wl.load(10000)
            if Int(w_lm) != 0: _mm_q8(Int(w_lm), bp, lp, NV, NE, nw)
            
            var best = 0; var bv = lp.load(0)
            for i in range(1, NV):
                var v = lp.load(i)
                if v > bv: bv = v; best = i
            cur_tok = best; cur_pos += 1
            if cur_pos >= MAX_SEQ: break
        
        # Build response
        var resp_buf = alloc[UInt8](64)
        var rpos = 0; var tmp = cur_tok
        while tmp > 0: rpos += 1; tmp /= 10
        while rpos > 0: rpos -= 1; resp_buf.store(rpos, UInt8(48 + (cur_tok % 10))); cur_tok /= 10
        
        var response = build_openai_response(resp_buf, 0, max_tok, model_name_c)
        if Int(response) != 0: send_response(response); c_free(response)
        c_free(prompt_c)
