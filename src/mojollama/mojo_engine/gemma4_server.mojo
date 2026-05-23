# gemma4_server.mojo — Pure Mojo OpenAI API server
# Build: gcc -c server_helper.c -o server_helper.o -lm
#        mojo build gemma4_server.mojo
#        gcc -o gemma4_server gemma4_server.o server_helper.o -lpthread -lm
# Run:   OMP_PLACES=cores OMP_PROC_BIND=close ./gemma4_server [port] [nw]

from std import time
from std.sys import argv
from std.math import sqrt, exp, cos, sin
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag
from std.sys.intrinsics import prefetch

# ─── Architecture ───
comptime NE: Int = 2560; comptime NH: Int = 16; comptime NK: Int = 4
comptime HD: Int = 128; comptime NL: Int = 42; comptime FF: Int = 10240
comptime NV: Int = 262144
comptime QI_NARROW: Int = NH * HD; comptime QI_WIDE: Int = NH * HD * 2
comptime WIDE_INTERVAL: Int = 6
comptime W: Int = 8; comptime RPW: Int = 8; comptime B: Int = 1
comptime QK: Int = 32; comptime QB: Int = 34; comptime EP: Float32 = 1e-5
comptime MAX_SEQ: Int = 4096; comptime WPL: Int = 18

def is_wide(l: Int) -> Bool: return l % WIDE_INTERVAL == (WIDE_INTERVAL - 1)
def qi(l: Int) -> Int: return QI_WIDE if is_wide(l) else QI_NARROW
def nk_hd(l: Int) -> Int: return (NK * HD * 2) if is_wide(l) else (NK * HD)

# ─── C FFI: memory + IO ───
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

# ─── C FFI: server functions ───
@extern("start_server")
def start_server(port: Int) abi("C") -> Int: ...
@extern("poll_request")
def poll_request() abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...
@extern("send_response")
def send_response(resp: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> None: ...
@extern("build_openai_response")
def build_openai_response(content: UnsafePointer[UInt8, MutExternalOrigin], pt: Int, ct: Int, model: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...
@extern("free")
def c_free_str(p: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> None: ...

# ─── Helpers ───
def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1); var e = Int((h >> 10) & 0x1F); var m = Int(h & 0x3FF)
    if e == 0: var r = Float32(m) * 5.960464477539063e-8; return -r if s != 0 else r
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

# ═══ Main ═══
def main() raises:
    var args = argv()
    var port = 8080
    var nw = 32
    if len(args) > 1: port = Int(String(args[1]))
    if len(args) > 2: nw = Int(String(args[2]))
    
    var t0 = time.perf_counter()
    print("Loading Gemma 4 model...")
    var wdir = String("/tmp/weights_e4b_final_transposed/")
    var dcp = str_to_c(wdir)
    var wl = alloc[Int64](20000)
    lw(dcp, wl, 10000, String(""), String("token_embd_weight.bin"))
    lw(dcp, wl, 10001, String(""), String("output_norm_weight.bin"))
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
    
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var rp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NV * 4))))
    var qb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(QI_WIDE * 4))))
    var kb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NK * HD * 2 * 4))))
    var vb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NK * HD * 2 * 4))))
    var att = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(QI_WIDE * 4))))
    var qn = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(QI_WIDE * 4))))
    var kn = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NK * HD * 2 * 4))))
    var gt = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF * 4))))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF * 4))))
    var kc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NL * NK * 2 * MAX_SEQ * HD * 4))))
    var vc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NL * NK * 2 * MAX_SEQ * HD * 4))))
    var sc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_SEQ * 4))))
    var nb = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb))
    var emb_rb = q8_rb(NE)
    
    print("Model loaded: ", Int((time.perf_counter() - t0) * 1000), " ms")
    print("Starting server on port ", port, "...")
    
    if start_server(port) < 0:
        print("ERROR: Failed to start server on port ", port)
        return
    
    print("Server ready at http://0.0.0.0:", port, "/v1/chat/completions")
    var model_name = str_to_c("gemma-4-e4b")
    
    while True:
        var prompt_c = poll_request()
        if Int(prompt_c) == 0:
            var us = 10000
            @extern("usleep")
            def _usleep(u: Int) abi("C") -> Int: ...
            _usleep(10000)
            continue
        
        # Run inference inline
        var cur_pos = 0
        var cur_tok = 2
        var max_tok = 128
        
        for pos in range(max_tok):
            var emb_off = cur_tok * emb_rb
            for blk in range(NE // QK):
                var lo = Int(emb.load(emb_off + blk * QB))
                var hi = Int(emb.load(emb_off + blk * QB + 1))
                var scale = h2f(UInt16(lo | (hi << 8)))
                for i in range(QK):
                    var qv = emb.load(emb_off + blk * QB + 2 + i).cast[DType.int8]()
                    hp.store(blk * QK + i, Float32(Int(qv)) * scale)
            
            for l in range(NL):
                var l_qi = qi(l); var l_nk = nk_hd(l); var base = l * WPL
                var nh_l = NH if l_qi == QI_NARROW else NH * 2
                var nk_l = NK if l_nk == NK * HD else NK * 2
                var hd_l = HD; var kr = nh_l // nk_l
                var kn_wn = l_nk / (1 + (NK / 4))
                
                for i in range(NE): rp.store(i, hp.load(i))
                var an_q = wl.load(base + 0)
                if Int(an_q) != 0: deq8(Int(an_q), nb, NE); rms_norm(hp, bp, nb, NE, NE)
                
                var wq = wl.load(base + 1); var wk = wl.load(base + 2)
                var wv = wl.load(base + 3); var wo = wl.load(base + 6)
                
                if Int(wq) != 0: _mm_q8(Int(wq), bp, qb, l_qi, NE, nw)
                var qn_q = wl.load(base + 4)
                if Int(qn_q) != 0: deq8(Int(qn_q), nb, l_qi / 8); rms_norm(qb, qn, nb, l_qi, l_qi / 8)
                
                if Int(wk) != 0: _mm_q8(Int(wk), bp, kb, l_nk, NE, nw)
                var kn_q = wl.load(base + 5)
                if Int(kn_q) != 0: deq8(Int(kn_q), nb, kn_wn); rms_norm(kb, kn, nb, l_nk, kn_wn)
                
                if Int(wv) != 0: _mm_q8(Int(wv), bp, vb, l_nk, NE, nw)
                
                for h in range(nk_l):
                    for d in range(hd_l):
                        var cpos = l * NK * 2 * MAX_SEQ * HD + h * MAX_SEQ * HD + cur_pos * HD + d
                        kc.store(cpos, kn.load(h * hd_l + d))
                        vc.store(cpos, vb.load(h * hd_l + d))
                
                for hq in range(nh_l):
                    var hk = hq // kr; var qbase_q = hq * hd_l
                    var cbase = l * NK * 2 * MAX_SEQ * HD + hk * MAX_SEQ * HD
                    var smax = Float32(-1e9)
                    for p in range(cur_pos + 1):
                        var sv = SIMD[DType.float32, W](0.0); var d = 0
                        while d + W <= hd_l:
                            sv = sv + qn.load[width=W](qbase_q + d) * kc.load[width=W](cbase + p * HD + d)
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
                            ov = ov + vc.load[width=W](cbase + p * HD + d) * (sc.load(p) * inv_s)
                        att.store[width=W](qbase_q + d, ov); d += W
                    while d < hd_l:
                        var o2 = Float32(0.0)
                        for p in range(cur_pos + 1): o2 += vc.load(cbase + p * HD + d) * (sc.load(p) * inv_s)
                        att.store(qbase_q + d, o2); d += 1
                
                if Int(wo) != 0: _mm_q8(Int(wo), att, bp, NE, l_qi, nw)
                
                var pan_q = wl.load(base + 7)
                if Int(pan_q) != 0: deq8(Int(pan_q), nb, NE); rms_norm(bp, bp, nb, NE)
                
                var w_ig = wl.load(base + 10); var w_pr = wl.load(base + 11)
                if Int(w_ig) != 0 and Int(w_pr) != 0:
                    _mm_q8(Int(w_ig), bp, kn, 256, NE, nw); silu(kn, 256)
                    _mm_q8(Int(w_pr), kn, bp, NE, 256, nw)
                
                var los = Float32(1.0)
                var los_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(base + 16)))
                if Int(los_p) != 0: los = los_p.load(0)
                for i in range(NE): hp.store(i, rp.load(i) + bp.load(i) * los)
                
                var pfn_q = wl.load(base + 8)
                if Int(pfn_q) != 0: deq8(Int(pfn_q), nb, NE); rms_norm(hp, bp, nb, NE)
                
                var w_gt = wl.load(base + 13); var w_up = wl.load(base + 14); var w_dn = wl.load(base + 15)
                if Int(w_gt) != 0 and Int(w_up) != 0:
                    for i in range(NE): rp.store(i, hp.load(i))
                    _mm_q8(Int(w_gt), bp, gt, FF, NE, nw)
                    _mm_q8(Int(w_up), bp, up, FF, NE, nw)
                    silu(gt, FF)
                    for i in range(FF): gt.store(i, gt.load(i) * up.load(i))
                    if Int(w_dn) != 0: _mm_q8(Int(w_dn), gt, bp, NE, FF, nw)
                    for i in range(NE): hp.store(i, rp.load(i) + bp.load(i) * los)
                
                var pn_q = wl.load(base + 9)
                if Int(pn_q) != 0: deq8(Int(pn_q), nb, NE); rms_norm(hp, bp, nb, NE)
            
            var on_q = wl.load(10001)
            if Int(on_q) != 0: deq8(Int(on_q), nb, NE); rms_norm(hp, bp, nb, NE)
            var w_lm = wl.load(10000)
            if Int(w_lm) != 0: _mm_q8(Int(w_lm), bp, lp, NV, NE, nw)
            
            var best = 0; var bv = lp.load(0)
            for i in range(1, NV):
                var v = lp.load(i)
                if v > bv: bv = v; best = i
            cur_tok = best
            cur_pos += 1
            if cur_pos >= MAX_SEQ: break
        
        # Build response text (token IDs)
        var resp_buf = alloc[UInt8](64)
        var rpos = 0
        var tmp = cur_tok
        if tmp == 0: rpos += 1
        while tmp > 0: rpos += 1; tmp /= 10
        var rp2 = rpos
        while rpos > 0:
            rpos -= 1
            resp_buf.store(rpos, UInt8(48 + (cur_tok % 10)))
            cur_tok /= 10
        
        var response = build_openai_response(resp_buf, 0, max_tok, model_name)
        if Int(response) != 0:
            send_response(response)
            c_free_str(response)
        
        c_free_str(prompt_c)
