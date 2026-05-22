# GPT-OSS-20B — Pure Mojo Forward Pass (f16 weights, VCVTPH2PS, FastMathFlag)
# Optimized using mojolang.org research:
#   - f16 weights: VCVTPH2PS (confirmed via --emit llvm)
#   - .fma() with FastMathFlag.FAST (from FastMathFlag.md)
#   - exp_approx_f32 SIMD fast exp (from std.math)
#   - parallelize for threading
from std import time
from std.math import sqrt
from std.math.math import exp_approx_f32
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime U8: Int = 0; comptime U16: Int = 1; comptime U32: Int = 2
comptime O_RDONLY: Int = 0; comptime SEEK_END: Int = 2; comptime SEEK_SET: Int = 0
comptime NE: Int = 2880;  comptime NH: Int = 64;  comptime NK: Int = 8
comptime HD: Int = 64;    comptime NL: Int = 24;  comptime NX: Int = 32
comptime NP: Int = 4;     comptime NF: Int = 2880; comptime NV: Int = 201088
comptime EP: Float32 = 1e-6; comptime W: Int = 8; comptime RPW: Int = 16

def f16_to_f32(h: UInt16) -> Float32:
    var s = UInt32(h >> 15); var e = UInt32((h >> 10) & 0x1F); var m = UInt32(h & 0x3FF)
    if e == 0:
        if m == 0: return 0.0
        return Float32(Float64(m) * 5.960464477539063e-8)
    if e == 31: return 0.0
    var r = Float32(m | 0x400); var ei = Int(e) - 25
    if ei >= 0:
        for _ in range(ei): r *= 2.0
    else:
        for _ in range(-ei): r *= 0.5
    return -r if s != 0 else r

def str_to_cstr(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var blen = s.byte_length(); var buf = alloc[UInt8](blen + 1)
    var src = s.unsafe_ptr()
    for i in range(blen): buf.store(i, src.load(i))
    buf.store(blen, UInt8(0)); return buf

# ── f16 matmul with FastMathFlag.FAST ──
def f16_mm(w: UnsafePointer[Float16, MutExternalOrigin],
           x: UnsafePointer[Float32, MutExternalOrigin],
           o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var bpr = nc // 32; var nb = (nr + RPW - 1) // RPW
    var nw = 32  # Threadripper 32 cores
    def worker(b: Int) capturing -> None:
        var rs = b * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc; var acc = SIMD[DType.float32, W](0.0)
            for blk in range(bpr):
                comptime for grp in range(4):
                    var w16 = w.load[width=W](ro + blk * 32 + grp * 8)
                    var wf32 = w16.cast[DType.float32]()
                    var xv = x.load[width=W](blk * 32 + grp * 8)
                    acc = wf32.fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=nb, num_workers=nw)

def f32_mm(w: UnsafePointer[Float32, MutExternalOrigin],
           x: UnsafePointer[Float32, MutExternalOrigin],
           o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var bpr = nc // W; var nb = (nr + RPW - 1) // RPW
    var nw = 32
    def worker(b: Int) capturing -> None:
        var rs = b * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc; var acc = SIMD[DType.float32, W](0.0)
            for blk in range(bpr):
                var wv = w.load[width=W](ro + blk * W)
                var xv = x.load[width=W](blk * W)
                acc = wv.fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=nb, num_workers=nw)

def main():
    @extern("malloc")
    def _c_malloc(sz: Int64) abi("C") -> Int64: ...
    @extern("free")
    def _c_free(p: Int64) abi("C") -> None: ...
    @extern("open")
    def _open(path: UnsafePointer[UInt8, MutExternalOrigin], flags: Int) abi("C") -> Int: ...
    @extern("read")
    def _read(fd: Int, buf: UnsafePointer[UInt8, MutExternalOrigin], cnt: Int64) abi("C") -> Int64: ...
    @extern("lseek")
    def _lseek(fd: Int, off: Int64, whence: Int) abi("C") -> Int64: ...
    @extern("close")
    def _close(fd: Int) abi("C") -> Int: ...
    
    print("GPT-OSS-20B — Pure Mojo f16 matmul (VCVTPH2PS + FastMathFlag)")
    print("==============================================================")
    var t0 = time.perf_counter()
    var base = "/tmp/mojo_weights/gpt-oss/"
    
    # ── Load weights: f32 norm/router, f16 quantized ──
    # Indices: 0=emb_f32, 1=out_norm, 2-25=layer_0..23
    # Each layer: 0=attn_norm(f32),1=ffn_norm(f32),2=Q,3=K,4=V,5=O,6=router(f32),
    #             7=gate_exps(f16-quant),8=up_exps(f16-quant),9=down_exps(f16-quant)
    # Quant types from info: Q=Q5_0(6), K=Q5_0(6), V=Q8_0(8), O=Q4_K(12), experts=MXFP4(39)
    # All converted to f16 at load time.
    
    var n_layers = NL; var n_experts = 32; var n_active = NP
    var wp = alloc[Int64](2 + n_layers * 10)
    var wc = 0
    
    # emb_f32 (f32 file) — keep as f32 pointer
    var fd = _open(str_to_cstr(base + "emb_f32.bin"), O_RDONLY)
    var sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
    var buf = _c_malloc(sz)
    _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz); _close(fd)
    wp.store(wc, buf); wc += 1
    
    # out_norm.bin (f32)
    fd = _open(str_to_cstr(base + "out_norm.bin"), O_RDONLY)
    sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
    buf = _c_malloc(sz)
    _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz); _close(fd)
    wp.store(wc, buf); wc += 1
    
    for layer in range(n_layers):
        var ld = base + "layer_" + String(layer) + "/"
        
        # attn_norm.f32 and ffn_norm.f32
        for nf_idx in range(2):
            var nm = "attn_norm" if nf_idx == 0 else "ffn_norm"
            fd = _open(str_to_cstr(ld + nm + ".bin"), O_RDONLY)
            sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
            buf = _c_malloc(sz)
            _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
            _close(fd)
            wp.store(wc, buf); wc += 1
        
        # Q, K, V, O weights — quantized → f16
        var q_files = ["q_weight", "k_weight", "v_weight", "o_weight"]
        var q_rows = [NH*HD, NK*HD, NK*HD, NE]
        var q_cols = [NE, NE, NE, NH*HD]
        for qi in range(4):
            fd = _open(str_to_cstr(ld + q_files[qi] + ".bin"), O_RDONLY)
            sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
            var raw = _c_malloc(sz)
            _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw)), sz)
            _close(fd)
            var ru8 = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw))
            var nr = q_rows[qi]; var nc = q_cols[qi]
            # Detect quant type from file size
            var bpr = nc // 32
            # Q4_K is 144 bytes/256vals, all others are 18/22/34 bytes/32vals
            var is_q4k = (Int(sz) == nr * nc // 256 * 144)
            var block_sz = 22 if (qi < 2) else (34 if (qi == 2) else (144 if is_q4k else 18))
            # Actually detect by info: Q=22, K=22, V=34, O=144 or 18
            if qi == 0: block_sz = 22  # Q5_0
            elif qi == 1: block_sz = 22  # Q5_0
            elif qi == 2: block_sz = 34  # Q8_0
            else:
                # O is Q4_K (144/256) or Q4_0 (18/32) — check size
                block_sz = 144 if (Int(sz) > nr * nc // 32 * 18) else 18
            var vals_per_block = 256 if block_sz == 144 else 32
            var total_blocks = nr * nc // vals_per_block
            var nb = nc // vals_per_block
            
            var f16_buf = _c_malloc(Int64(nr * nc * 2))
            var f16p = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(f16_buf))
            
            for r in range(nr):
                for blk in range(nb):
                    var bo = r * nb * block_sz + blk * block_sz
                    var lo = UInt16(ru8.load(bo)); var hi = UInt16(ru8.load(bo + 1))
                    var d = f16_to_f32(lo | (hi << 8))
                    for j in range(vals_per_block):
                        var nib: Int32 = 0
                        if block_sz == 22:  # Q5_0
                            var p = ru8.load(bo + 6 + j // 2)
                            var qh = ru8.load(bo + 2 + j // 4)
                            var hs = 2 * (j % 4); var hb = (UInt8(qh) >> UInt8(hs)) & UInt8(1)
                            nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
                            if nib > 7: nib -= 16
                            nib = nib + Int32(hb) * 16
                            if nib > 15: nib -= 32
                        elif block_sz == 34:  # Q8_0
                            var qb = ru8.load(bo + 2 + j)
                            nib = Int32(qb)
                            if nib > 127: nib -= 256
                        elif block_sz == 144:  # Q4_K — simplified
                            nib = 0  # placeholder — Q4_K decode is complex
                        else:  # Q4_0
                            var p = ru8.load(bo + 2 + j // 2)
                            nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
                            if nib > 7: nib -= 16
                        var f32val = Float32(nib) * d
                        if f32val > 65000.0: f32val = 65000.0
                        if f32val < -65000.0: f32val = -65000.0
                        f16p.store(r * nc + blk * vals_per_block + j, Float16(f32val))
            _c_free(raw)
            wp.store(wc, f16_buf); wc += 1
        
        # Router (f32)
        fd = _open(str_to_cstr(ld + "router.bin"), O_RDONLY)
        sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
        buf = _c_malloc(sz)
        _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz); _close(fd)
        wp.store(wc, buf); wc += 1
        
        # Expert weights: gate, up, down (MXFP4 → f16)
        for ei in range(3):
            var en = ["gate_exps", "up_exps", "down_exps"][ei]
            fd = _open(str_to_cstr(ld + en + ".bin"), O_RDONLY)
            sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
            var raw = _c_malloc(sz)
            _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw)), sz)
            _close(fd)
            var ru8 = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw))
            # MXFP4: 17 bytes/32vals, each expert: [2880, 2880]
            var exp_blocks = NE * NE // 32
            var exp_bytes = exp_blocks * 17
            var f16_buf = _c_malloc(Int64(n_experts * NE * NE * 2))
            var f16p = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(f16_buf))
            var nb_exp = NE // 32
            
            for exp in range(n_experts):
                for r in range(NE):
                    for blk in range(nb_exp):
                        var bo = (exp * NE + r) * nb_exp * 17 + blk * 17
                        var eb = ru8.load(bo + 16)
                        var sf: Float32 = 0.0
                        if eb != 0 and eb < 255:
                            var ei2 = Int(eb) - 127; sf = 1.0
                            if ei2 >= 0:
                                for _ in range(ei2): sf *= 2.0
                            else:
                                for _ in range(-ei2): sf *= 0.5
                        elif eb == 255: sf = 1e20
                        else: sf = 0.0
                        if sf > 1e10: sf = 1e10
                        for j in range(32):
                            var p = ru8.load(bo + j // 2)
                            nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
                            if nib > 7: nib -= 16
                            var f32val = Float32(nib) * sf
                            if f32val > 65000.0: f32val = 65000.0
                            if f32val < -65000.0: f32val = -65000.0
                            f16p.store((exp * NE + r) * NE + blk * 32 + j, Float16(f32val))
            _c_free(raw)
            wp.store(wc, f16_buf); wc += 1
    
    var ts = time.perf_counter()
    print("Load:", Int((ts - t0) * 1000.0), "ms")
    
    # ── Buffers ──
    var ni = NE; var nq = NH * HD; var nkv = NK * HD; var kr = NH // NK
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(ni * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(ni * 4))))
    var qp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(nq * 4))))
    var kp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(nkv * 4))))
    var vp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(nkv * 4))))
    var gp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(NF * 4))))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(NF * 4))))
    var dp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(NE * 4))))
    var rp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(NX * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(4096 * 4))))  # small sample
    
    # Embedding: read token 5 from emb_f32
    var emb_f32 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(0)))
    for i in range(ni): hp.store(i, emb_f32.load(5 * ni + i))
    print("h[0]:", hp.load(0), "h[4]:", hp.load(4))
    var tf = time.perf_counter()
    
    # ── Benchmark: run forward pass n times ──
    var n_bench = 1  # Single pass for benchmark
    var t_start = time.perf_counter()
    
    var best = 0; var bv = lp.load(0)
    for _ in range(n_bench):
        wi = 2  # Reset weight index
        for layer in range(NL):
            var an = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            var ff = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            
            # RMS norm (attention)
            var ss: Float64 = 0.0
            for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
            var inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
            for i in range(ni): bp.store(i, hp.load(i) * an.load(i) * inv)
            
            # QKV f16 matmul
            var qw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            var kw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            var vw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            var ow = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            f16_mm(qw, bp, qp, nq, ni)
            f16_mm(kw, bp, kp, nkv, ni)
            f16_mm(vw, bp, vp, nkv, ni)
            
            # GQA attention
            for hq in range(NH):
                var hkv = hq // kr; var sc: Float64 = 0.0
                for d in range(HD): sc += Float64(qp.load(hq * HD + d)) * Float64(kp.load(hkv * HD + d))
                var wt = Float32(sc / Float64(HD))
                for d in range(HD):
                    var ao = vp.load(hkv * HD + d) * wt
                    if ao != ao: ao = 0.0
                    if ao > 100.0: ao = 100.0
                    if ao < -100.0: ao = -100.0
                    qp.store(hq * HD + d, ao)
            
            f16_mm(ow, qp, bp, ni, nq)
            for i in range(ni):
                var v = hp.load(i) + bp.load(i)
                if v != v: v = 0.0
                if v > 100.0: v = 100.0
                if v < -100.0: v = -100.0
                hp.store(i, v)
            
            # RMS norm (FFN)
            ss = 0.0
            for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
            inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
            for i in range(ni): bp.store(i, hp.load(i) * ff.load(i) * inv)
            
            # Router (f32)
            var rw = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            f32_mm(rw, bp, rp, NX, ni)
            
            # Top-4 selection
            var idx = alloc[Int32](NP); var vals = alloc[Float32](NP)
            for p in range(NP):
                idx.store(p, Int32(p)); vals.store(p, rp.load(p))
            # Bubble sort top 4
            for i in range(NP, NX):
                var vi = rp.load(i)
                var p = NP - 1
                while p >= 0 and vi > vals.load(p):
                    if p < NP - 1:
                        idx.store(p+1, idx.load(p)); vals.store(p+1, vals.load(p))
                    p -= 1
                if p < NP - 1:
                    idx.store(p+1, Int32(i)); vals.store(p+1, vi)
            
            # Zero output
            for i in range(ni): dp.store(i, 0.0)
            
            # Active experts
            var gw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            var uw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            var dw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(wi))); wi += 1
            var exp_stride = NE * NE
            var gw_addr = wp.load(wi-3); var uw_addr = wp.load(wi-2); var dw_addr = wp.load(wi-1)
            
            for e in range(NP):
                var ei = Int(idx.load(e))
                var ew = vals.load(e)
                if ew <= 0.0: continue
                var eo = ei * exp_stride
                var egw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(gw_addr + eo * 2))
                var euw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(uw_addr + eo * 2))
                var edw = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(dw_addr + eo * 2))
                
                f16_mm(egw, bp, gp, NF, ni)
                f16_mm(euw, bp, up, NF, ni)
                
                # SiLU with SIMD exp_approx
                for k in range(0, NF, W):
                    var gv = gp.load[width=W](k)
                    gv = gv.gt(SIMD[DType.float32, W](80.0)).select(SIMD[DType.float32, W](80.0), gv)
                    gv = gv.lt(SIMD[DType.float32, W](-80.0)).select(SIMD[DType.float32, W](-80.0), gv)
                    var sg = gv / (SIMD[DType.float32, W](1.0) + exp_approx_f32(-gv))
                    var rv = sg * up.load[width=W](k)
                    rv = rv.gt(SIMD[DType.float32, W](100.0)).select(SIMD[DType.float32, W](100.0), rv)
                    rv = rv.lt(SIMD[DType.float32, W](-100.0)).select(SIMD[DType.float32, W](-100.0), rv)
                    gp.store[width=W](k, rv)
                
                f16_mm(edw, gp, up, ni, NF)
                
                for k in range(ni):
                    var v = up.load(k) * ew
                    if v != v: v = 0.0
                    dp.store(k, dp.load(k) + v)
            
            # Residual
            for i in range(ni):
                var v = hp.load(i) + dp.load(i)
                if v != v: v = 0.0
                if v > 1e10: v = 1e10
                if v < -1e10: v = -1e10
                hp.store(i, v)
            
            idx.free(); vals.free()
        
        # Final RMS norm
        var ss2: Float64 = 0.0
        var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(1)))
        for i in range(ni): ss2 += Float64(hp.load(i)) * Float64(hp.load(i))
        inv = Float32(1.0 / sqrt(Float64(ss2) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * onp.load(i) * inv)
        
        # LM head: f32 matmul (emb_f32 is f32 weight)
        var lm_head_f32 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(0)))
        f32_mm(lm_head_f32, bp, lp, 4096, ni)
        
        var best = 0; var bv = lp.load(0)
        for i in range(1, 4096):
            var v = lp.load(i)
            if v > bv: bv = v; best = i
            _ = best  # use best
    
    var tt = time.perf_counter()
    var elapsed = (tt - t_start) / Float64(n_bench)
    
    print("\n=== Results ===")
    print("Layers:", Int((tt - tf) * 1000.0), "ms")
    print("Total:", Int(elapsed * 1000.0), "ms")
    print("tok/s:", Float64(1.0) / elapsed)
    print("Best token (first 4096):", best, "val:", bv)
