# universal_engine.mojo — Universal Mojo inference engine with prompt prefill
# Clean version with proper Mojo syntax (no inline if/for, no nested captures)

from std import time
from std.sys import argv
from std.math import sqrt, exp, cos, sin
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag
from std.sys.intrinsics import prefetch

comptime MAX_SEQ: Int = 2048
comptime W: Int = 8; comptime RPW: Int = 8; comptime B: Int = 1
comptime QK: Int = 32; comptime QB: Int = 34; comptime EP: Float32 = 1e-5
comptime WPL: Int = 18

# ─── Extern ───
@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...
@extern("free")
def c_free(p: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> None: ...
@extern("open")
def _open(p: UnsafePointer[UInt8, MutExternalOrigin], f: Int) abi("C") -> Int: ...
@extern("read")
def _read(fd: Int, b: UnsafePointer[UInt8, MutExternalOrigin], c: Int64) abi("C") -> Int64: ...
@extern("lseek")
def _lseek(fd: Int, o: Int64, w: Int) abi("C") -> Int64: ...
@extern("close")
def _close(fd: Int) abi("C") -> Int: ...
@extern("mojo_write_tokens")
def _sys_write(fd: Int, b: UnsafePointer[UInt8, MutExternalOrigin], c: Int) abi("C") -> Int: ...
@extern("getenv")
def _getenv(n: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...
@extern("read_arch_int")
def read_arch_int(d: UnsafePointer[UInt8, MutExternalOrigin], k: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> Int: ...

def str_to_c(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var b = alloc[UInt8](s.byte_length() + 1)
    var sp = s.unsafe_ptr()
    for i in range(s.byte_length()):
        b.store(i, sp.load(i))
    b.store(s.byte_length(), UInt8(0))
    return b

def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1); var e = Int((h >> 10) & 0x1F); var m = Int(h & 0x3FF)
    if e == 0: return Float32(m) * 5.960464477539063e-8
    if e == 31: return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    var tmp = alloc[UInt8](4)
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp)).store(0, bits)
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp)).load(0)

def q8_rb(nc: Int) -> Int: return ((nc + QK - 1) // QK) * QB

def _mm_q8(qa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
           o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qa))
    var nb = (nr + RPW - 1) // RPW; var rb = q8_rb(nc)
    def wk(wi: Int) capturing:
        var rs = wi * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var acc = SIMD[DType.float32, W](0.0)
            var ro = r * rb; var col = 0
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

def rms_norm(x: UnsafePointer[Float32, MutExternalOrigin],
              o: UnsafePointer[Float32, MutExternalOrigin],
              w: UnsafePointer[Float32, MutExternalOrigin], n: Int, wn: Int = 0):
    var ss = Float32(0.0); var i = 0
    while i + 8 <= n:
        var v = x.load[width=8](i)
        ss += (v * v).reduce_add()
        i += 8
    while i < n:
        ss += x.load(i) * x.load(i)
        i += 1
    var inv = 1.0 / sqrt(ss / Float32(n) + EP)
    var w_n = wn if wn > 0 else n
    var gs = n / w_n
    if gs <= 1:
        var inv_v = SIMD[DType.float32, 8](inv)
        i = 0
        while i + 8 <= n:
            o.store[width=8](i, x.load[width=8](i) * inv_v * w.load[width=8](i))
            i += 8
        while i < n:
            o.store(i, x.load(i) * w.load(i) * inv)
            i += 1
    else:
        for gi in range(w_n):
            var wi = w.load(gi)
            var inv_v = SIMD[DType.float32, 8](inv * wi)
            var go = gi * gs
            for j in range(0, gs, 8):
                o.store[width=8](go + j, x.load[width=8](go + j) * inv_v)

def main() raises:
    var args = argv()
    var port = 8080
    var nw = 32
    var wdir = String("/tmp/weights_e4b_final_transposed/")
    if len(args) > 1: port = Int(String(args[1]))
    if len(args) > 2: nw = Int(String(args[2]))
    if len(args) > 3: wdir = String(args[3])
    
    # ─── Arch config ───
    var NE = 2560; var NH = 16; var NK = 4; var HD = 128
    var NL = 42; var FF = 10240; var NV = 262144
    var HAS_QK = True; var HAS_IG = True; var HAS_ROPE = False
    var WIDE_INTERVAL = 6
    
    var dc = str_to_c(wdir)
    var ne_v = read_arch_int(dc, str_to_c(String("ne")))
    if ne_v > 0: NE = ne_v
    var nh_v = read_arch_int(dc, str_to_c(String("nh")))
    if nh_v > 0: NH = nh_v
    var nk_v = read_arch_int(dc, str_to_c(String("nk")))
    if nk_v > 0: NK = nk_v
    var hd_v = read_arch_int(dc, str_to_c(String("hd")))
    if hd_v > 0: HD = hd_v
    var nl_v = read_arch_int(dc, str_to_c(String("nl")))
    if nl_v > 0: NL = nl_v
    var ff_v = read_arch_int(dc, str_to_c(String("ff")))
    if ff_v > 0: FF = ff_v
    var nv_v = read_arch_int(dc, str_to_c(String("nv")))
    if nv_v > 0: NV = nv_v
    var wi_v = read_arch_int(dc, str_to_c(String("wide_interval")))
    if wi_v > 0: WIDE_INTERVAL = wi_v
    
    var qi_n = NH * HD; var qi_w = NH * HD * 2
    var nk_n = NK * HD; var nk_w = NK * HD * 2
    var max_nk = NK * 2
    
    # ─── Load weights ───
    var wl = alloc[Int64](20000)
    lw_proc(dc, wl, 10000, "token_embd_weight.bin")
    lw_proc(dc, wl, 10001, "output_norm_weight.bin")
    lw_proc(dc, wl, 10002, "rope_freqs_weight.bin")
    var w_emb = wl.load(10000)
    
    for l in range(NL):
        var pfx = "blk_" + String(l) + "_"
        lw_proc(dc, wl, l * WPL + 0, pfx + "attn_norm_weight.bin")
        lw_proc(dc, wl, l * WPL + 1, pfx + "attn_q_weight.bin")
        lw_proc(dc, wl, l * WPL + 2, pfx + "attn_k_weight.bin")
        lw_proc(dc, wl, l * WPL + 3, pfx + "attn_v_weight.bin")
        lw_proc(dc, wl, l * WPL + 4, pfx + "attn_q_norm_weight.bin")
        lw_proc(dc, wl, l * WPL + 5, pfx + "attn_k_norm_weight.bin")
        lw_proc(dc, wl, l * WPL + 6, pfx + "attn_output_weight.bin")
        lw_proc(dc, wl, l * WPL + 7, pfx + "post_attention_norm_weight.bin")
        lw_proc(dc, wl, l * WPL + 8, pfx + "post_ffw_norm_weight.bin")
        lw_proc(dc, wl, l * WPL + 9, pfx + "post_norm_weight.bin")
        lw_proc(dc, wl, l * WPL + 10, pfx + "inp_gate_weight.bin")
        lw_proc(dc, wl, l * WPL + 11, pfx + "proj_weight.bin")
        lw_proc(dc, wl, l * WPL + 12, pfx + "ffn_norm_weight.bin")
        lw_proc(dc, wl, l * WPL + 13, pfx + "ffn_gate_weight.bin")
        lw_proc(dc, wl, l * WPL + 14, pfx + "ffn_up_weight.bin")
        lw_proc(dc, wl, l * WPL + 15, pfx + "ffn_down_weight.bin")
        lw_proc(dc, wl, l * WPL + 16, pfx + "layer_output_scale_weight.bin")
    
    # ─── Allocate buffers ───
    var hp = _alc_buf(NE)
    var bp = _alc_buf(NE)
    var rp = _alc_buf(NE)
    var lp = _alc_buf(NV)
    var qb = _alc_buf(qi_w)
    var kb = _alc_buf(max_nk * HD)
    var vb = _alc_buf(max_nk * HD)
    var ab = _alc_buf(qi_w)
    var qn = _alc_buf(qi_w)
    var kn = _alc_buf(max_nk * HD)
    var gt = _alc_buf(FF)
    var up_b = _alc_buf(FF)
    var kc = _alc_buf(NL * max_nk * MAX_SEQ * HD)
    var vc = _alc_buf(NL * max_nk * MAX_SEQ * HD)
    var sc = _alc_buf(MAX_SEQ)
    var nb = _alc_buf(NE)
    var rope_b = _alc_buf(HD)
    
    # Dequant rope
    var ra = wl.load(10002)
    if Int(ra) != 0:
        deq8_proc(Int(ra), rope_b, HD / 2)
    
    var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb))
    var emb_rb = q8_rb(NE)
    
    # ─── Read prompt tokens ───
    var prompt_path = _getenv(str_to_c(String("MOJO_PROMPT_FILE")))
    var output_path = _getenv(str_to_c(String("MOJO_OUTPUT_FILE")))
    var max_tok_str = _getenv(str_to_c(String("MOJO_MAX_TOKENS")))
    
    var max_tok = 16
    if Int(max_tok_str) != 0:
        max_tok = 0
        var mti = 0
        while max_tok_str.load(mti) >= UInt8(48) and max_tok_str.load(mti) <= UInt8(57):
            max_tok = max_tok * 10 + Int(max_tok_str.load(mti) - UInt8(48))
            mti += 1
    
    var prompt_tokens = alloc[Int64](MAX_SEQ)
    var prompt_len = 0
    
    if Int(prompt_path) != 0:
        var pf = _open(prompt_path, 0)
        if pf >= 0:
            var sz = _lseek(pf, 0, 2)
            _ = _lseek(pf, 0, 0)
            prompt_len = Int(sz) / 4
            if prompt_len > MAX_SEQ: prompt_len = MAX_SEQ
            # Read Int32 tokens directly into Int64 array
            _ = _read(pf, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(prompt_tokens)), Int64(prompt_len * 4))
            _ = _close(pf)
            # Expand: Int32 values are packed into the buffer; convert to Int64
            var i = prompt_len
            while i > 0:
                i -= 1
                var val32 = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(prompt_tokens)).load(i)
                prompt_tokens.store(i, Int64(val32))
    
    var cur_tok = 2
    var cur_pos = 0
    
    # ─── Prefill ───
    var prefill_end = prompt_len if prompt_len > 0 else 1
    for pos in range(prefill_end):
        if prompt_len > 0:
            cur_tok = Int(prompt_tokens.load(pos))
        else:
            if pos == 0: cur_tok = 2
        
        embed_token(cur_tok, emb, emb_rb, hp, NE)
        run_layers(NL, NE, NH, NK, HD, FF, NV, WIDE_INTERVAL,
                  qi_n, qi_w, nk_n, nk_w, max_nk, max_nk,
                  HAS_QK, HAS_IG, HAS_ROPE,
                  wl, hp, bp, rp, qb, kb, vb, ab, qn, kn, gt, up_b,
                  kc, vc, sc, nb, rope_b, cur_pos, nw, False)
        cur_pos += 1
    
    # ─── Generate ───
    for pos in range(max_tok):
        embed_token(cur_tok, emb, emb_rb, hp, NE)
        run_layers(NL, NE, NH, NK, HD, FF, NV, WIDE_INTERVAL,
                  qi_n, qi_w, nk_n, nk_w, max_nk, max_nk,
                  HAS_QK, HAS_IG, HAS_ROPE,
                  wl, hp, bp, rp, qb, kb, vb, ab, qn, kn, gt, up_b,
                  kc, vc, sc, nb, rope_b, cur_pos, nw, True)
        
        # LM head
        var on_q = wl.load(10001)
        if Int(on_q) != 0:
            deq8_proc(Int(on_q), nb, NE)
            rms_norm(hp, bp, nb, NE)
        var w_lm = wl.load(10000)
        if Int(w_lm) != 0:
            _mm_q8(Int(w_lm), bp, lp, NV, NE, nw)
        
        var best = 0
        var bv = lp.load(0)
        for i in range(1, NV):
            var v = lp.load(i)
            if v > bv: bv = v; best = i
        
        prompt_tokens.store(prompt_len + pos, Int64(best))
        cur_tok = best
        cur_pos += 1
        if cur_pos >= MAX_SEQ: break
    
    # Write binary token data to stdout (4 tokens = 32 bytes)
    _sys_write(1, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(prompt_tokens + prompt_len)), max_tok * 8)

# Need these helper functions at module level
def _alc_buf(n: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n * 4))))

def lw_proc(dcp: UnsafePointer[UInt8, MutExternalOrigin],
            wl: UnsafePointer[Int64, MutExternalOrigin], idx: Int, fname: String):
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
    if addr > 0:
        _ = _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(addr)), sz)
    _ = _close(fd)
    wl.store(idx, addr)

def deq8_proc(qaddr: Int, o: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qaddr))
    for blk in range(n // QK):
        var bo = blk * QB
        var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
        var sv = h2f(UInt16(lo | (hi << 8)))
        for i in range(QK):
            var qv = q.load(bo + 2 + i).cast[DType.int8]()
            o.store(blk * QK + i, Float32(Int(qv)) * sv)
    var rem = n % QK
    if rem > 0:
        var blk = n // QK; var bo = blk * QB
        var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
        var sv = h2f(UInt16(lo | (hi << 8)))
        for i in range(rem):
            var qv = q.load(bo + 2 + i).cast[DType.int8]()
            o.store(blk * QK + i, Float32(Int(qv)) * sv)

def embed_token(tok: Int, emb: UnsafePointer[UInt8, MutExternalOrigin],
                emb_rb: Int, hp: UnsafePointer[Float32, MutExternalOrigin], NE: Int):
    var emb_off = tok * emb_rb
    for blk in range(NE // QK):
        var lo = Int(emb.load(emb_off + blk * QB))
        var hi = Int(emb.load(emb_off + blk * QB + 1))
        var scale = h2f(UInt16(lo | (hi << 8)))
        for i in range(QK):
            var qv = emb.load(emb_off + blk * QB + 2 + i).cast[DType.int8]()
            hp.store(blk * QK + i, Float32(Int(qv)) * scale)

def run_layers(NL: Int, NE: Int, NH: Int, NK: Int, HD: Int, FF: Int, NV: Int,
               WIDE_INTERVAL: Int, qi_n: Int, qi_w: Int, nk_n: Int, nk_w: Int,
               max_nk: Int, max_nk_hd: Int,
               HAS_QK: Bool, HAS_IG: Bool, HAS_ROPE: Bool,
               wl: UnsafePointer[Int64, MutExternalOrigin],
               hp: UnsafePointer[Float32, MutExternalOrigin],
               bp: UnsafePointer[Float32, MutExternalOrigin],
               rp: UnsafePointer[Float32, MutExternalOrigin],
               qb: UnsafePointer[Float32, MutExternalOrigin],
               kb: UnsafePointer[Float32, MutExternalOrigin],
               vb: UnsafePointer[Float32, MutExternalOrigin],
               ab: UnsafePointer[Float32, MutExternalOrigin],
               qn: UnsafePointer[Float32, MutExternalOrigin],
               kn: UnsafePointer[Float32, MutExternalOrigin],
               gt: UnsafePointer[Float32, MutExternalOrigin],
               up: UnsafePointer[Float32, MutExternalOrigin],
               kc: UnsafePointer[Float32, MutExternalOrigin],
               vc: UnsafePointer[Float32, MutExternalOrigin],
               sc: UnsafePointer[Float32, MutExternalOrigin],
               nb: UnsafePointer[Float32, MutExternalOrigin],
               rope_b: UnsafePointer[Float32, MutExternalOrigin],
               cur_pos: Int, nw: Int, do_lm_head: Bool):
    for l in range(NL):
        var is_wide = WIDE_INTERVAL > 0 and ((l + 1) % WIDE_INTERVAL == 0)
        var l_qi = qi_w if is_wide else qi_n
        var l_nk = nk_w if is_wide else nk_n
        var base = l * WPL
        var nh_l = NH if l_qi == qi_n else NH * 2
        var nk_l = NK if l_nk == nk_n else NK * 2
        var hd_l = HD
        var kr = nh_l // nk_l
        var kn_wn = l_nk / (1 + (NK / 4))
        
        # Save residual
        for i in range(NE): rp.store(i, hp.load(i))
        
        # Pre-attention RMS norm
        var an_q = wl.load(base + 0)
        if Int(an_q) != 0:
            deq8_proc(Int(an_q), nb, NE)
            rms_norm(hp, bp, nb, NE, NE)
        
        # Q projection
        var wq = wl.load(base + 1)
        if Int(wq) != 0:
            _mm_q8(Int(wq), bp, qb, l_qi, NE, nw)
        
        # QK-norm on Q
        if HAS_QK:
            var qn_q = wl.load(base + 4)
            if Int(qn_q) != 0:
                deq8_proc(Int(qn_q), nb, l_qi / 8)
                rms_norm(qb, qn, nb, l_qi, l_qi / 8)
            else:
                for i in range(l_qi): qn.store(i, qb.load(i))
        else:
            for i in range(l_qi): qn.store(i, qb.load(i))
        
        # K projection
        var wk = wl.load(base + 2)
        if Int(wk) != 0:
            _mm_q8(Int(wk), bp, kb, l_nk, NE, nw)
        
        # QK-norm on K
        if HAS_QK:
            var kn_q = wl.load(base + 5)
            if Int(kn_q) != 0:
                deq8_proc(Int(kn_q), nb, kn_wn)
                rms_norm(kb, kn, nb, l_nk, kn_wn)
            else:
                for i in range(l_nk): kn.store(i, kb.load(i))
        else:
            for i in range(l_nk): kn.store(i, kb.load(i))
        
        # V projection
        var wv = wl.load(base + 3)
        if Int(wv) != 0:
            _mm_q8(Int(wv), bp, vb, l_nk, NE, nw)
        
        # KV cache store
        for h in range(nk_l):
            for d in range(hd_l):
                var cpos = l * max_nk * HD * MAX_SEQ + h * HD * MAX_SEQ + cur_pos * HD + d
                kc.store(cpos, kn.load(h * hd_l + d))
                vc.store(cpos, vb.load(h * hd_l + d))
        
        # GQA Attention
        for hq in range(nh_l):
            var hk = hq // kr
            var cbase = l * max_nk * HD * MAX_SEQ + hk * HD * MAX_SEQ
            var smax = Float32(-1e9)
            for p in range(cur_pos + 1):
                var sv = SIMD[DType.float32, W](0.0)
                var d = 0
                while d + W <= hd_l:
                    sv = sv + qn.load[width=W](hq * hd_l + d) * kc.load[width=W](cbase + p * HD + d)
                    d += W
                var s = sv.reduce_add() / sqrt(Float32(hd_l))
                if s > 80.0: s = 80.0
                if s < -80.0: s = -80.0
                sc.store(p, s)
                if s > smax: smax = s
            
            # Softmax
            var ssum = Float32(0.0)
            for p in range(cur_pos + 1):
                var e2 = exp(sc.load(p) - smax)
                sc.store(p, e2)
                ssum += e2
            var inv_s = 1.0 / (ssum + 1e-10)
            
            # Weighted sum
            var d = 0
            while d + W <= hd_l:
                var ov = SIMD[DType.float32, W](0.0)
                for p in range(cur_pos + 1):
                    ov = ov + vc.load[width=W](cbase + p * HD + d) * (sc.load(p) * inv_s)
                ab.store[width=W](hq * hd_l + d, ov)
                d += W
            while d < hd_l:
                var o2 = Float32(0.0)
                for p in range(cur_pos + 1):
                    o2 += vc.load(cbase + p * HD + d) * (sc.load(p) * inv_s)
                ab.store(hq * hd_l + d, o2)
                d += 1
        
        # O projection
        var wo = wl.load(base + 6)
        if Int(wo) != 0:
            _mm_q8(Int(wo), ab, bp, NE, l_qi, nw)
        
        # Post-attention norm
        var pan_q = wl.load(base + 7)
        if Int(pan_q) != 0:
            deq8_proc(Int(pan_q), nb, NE)
            rms_norm(bp, bp, nb, NE)
        
        # inp_gate + proj
        if HAS_IG:
            var w_ig = wl.load(base + 10)
            var w_pr = wl.load(base + 11)
            if Int(w_ig) != 0 and Int(w_pr) != 0:
                _mm_q8(Int(w_ig), bp, kn, 256, NE, nw)
                silu_proc(kn, 256)
                _mm_q8(Int(w_pr), kn, bp, NE, 256, nw)
        
        # Residual
        var los = Float32(1.0)
        var los_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 16)))
        if Int(los_p) != 0: los = los_p.load(0)
        for i in range(NE): hp.store(i, rp.load(i) + bp.load(i) * los)
        
        # Post-FFW norm
        var pfn_q = wl.load(base + 8)
        if Int(pfn_q) != 0:
            deq8_proc(Int(pfn_q), nb, NE)
            rms_norm(hp, bp, nb, NE)
        
        # FFN
        var w_gt = wl.load(base + 13)
        var w_up = wl.load(base + 14)
        var w_dn = wl.load(base + 15)
        if Int(w_gt) != 0 and Int(w_up) != 0:
            for i in range(NE): rp.store(i, hp.load(i))
            _mm_q8(Int(w_gt), bp, gt, FF, NE, nw)
            _mm_q8(Int(w_up), bp, up, FF, NE, nw)
            silu_proc(gt, FF)
            for i in range(FF): gt.store(i, gt.load(i) * up.load(i))
            if Int(w_dn) != 0:
                _mm_q8(Int(w_dn), gt, bp, NE, FF, nw)
            for i in range(NE): hp.store(i, rp.load(i) + bp.load(i) * los)
        
        # Post-norm
        var pn_q = wl.load(base + 9)
        if Int(pn_q) != 0:
            deq8_proc(Int(pn_q), nb, NE)
            rms_norm(hp, bp, nb, NE)

def silu_proc(p: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    for i in range(n):
        var v = p.load(i)
        if v < -80.0: v = -80.0
        if v > 80.0: v = 80.0
        p.store(i, v / (1.0 + exp(-v)))
