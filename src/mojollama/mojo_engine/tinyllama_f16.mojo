# TinyLlama — Pure Mojo with f16 weights (VCVTPH2PS matmul)
from std import time
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime O_RDONLY: Int = 0; comptime SEEK_END: Int = 2; comptime SEEK_SET: Int = 0
comptime NE: Int = 2048;  comptime NH: Int = 32;  comptime NK: Int = 4
comptime HD: Int = 64;    comptime NL: Int = 22;  comptime NF: Int = 5632
comptime NV: Int = 32000; comptime EP: Float32 = 1e-6
comptime W: Int = 8;  comptime RPW: Int = 32; comptime Q4_0_BS: Int = 18
comptime N_BASE: Int = 3;  comptime N_LF: Int = 9
comptime W_EMB: Int = 0;  comptime W_ON: Int = 1;  comptime W_LM: Int = 2
comptime LA: Int = 0;  comptime LF: Int = 1;  comptime LQ: Int = 2
comptime LK: Int = 3;  comptime LV: Int = 4;  comptime LO: Int = 5
comptime LG: Int = 6;  comptime LU: Int = 7;  comptime LD: Int = 8

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

def get_filename(idx: Int) -> String:
    if idx == 0: return "token_embd_weight.q4"
    if idx == 1: return "output_norm_weight.f32"
    if idx == 2: return "output_weight.q4"
    var l = (idx - N_BASE) // N_LF; var f = (idx - N_BASE) % N_LF
    var fname: String
    if f == 0: fname = "attn_norm_weight.f32"
    elif f == 1: fname = "ffn_norm_weight.f32"
    elif f == 2: fname = "attn_q_weight.q4"
    elif f == 3: fname = "attn_k_weight.q4"
    elif f == 4: fname = "attn_v_weight.q4"
    elif f == 5: fname = "attn_output_weight.q4"
    elif f == 6: fname = "ffn_gate_weight.q4"
    elif f == 7: fname = "ffn_up_weight.q4"
    elif f == 8: fname = "ffn_down_weight.q4"
    else: return ""
    return "blk_" + String(l) + "_" + fname

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
    
    print("TinyLlama — Pure Mojo f16 matmul")
    print("=================================")
    var t0 = time.perf_counter()
    
    # ── Load + convert weights ──
    var nf = N_BASE + NL * N_LF
    var wp = alloc[Int64](nf)
    var wtype = alloc[Int64](nf)
    for i in range(nf):
        wp.store(i, 0)
        wtype.store(i, 0)
    
    for idx in range(nf):
        var path = str_to_cstr("/tmp/tinyllama/" + get_filename(idx))
        var fd = _open(path, O_RDONLY)
        path.free()
        if fd < 0: print("Error: open", idx); return
        var fname = get_filename(idx)
        
        if fname.endswith(".q4"):
            var sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
            var raw = _c_malloc(sz)
            _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw)), sz)
            _close(fd)
            var ru8 = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw))
            var bpr = Int(sz // Int64(Q4_0_BS))
            var expected_vals = bpr * 32
            var f16_buf = _c_malloc(Int64(expected_vals * 2))
            var f16p = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(f16_buf))
            for blk in range(bpr):
                var bo = blk * Q4_0_BS
                var lo = UInt16(ru8.load(bo)); var hi = UInt16(ru8.load(bo + 1))
                var d = f16_to_f32(lo | (hi << 8))
                for j in range(32):
                    var nib = Int32(ru8.load(bo + 2 + j // 2))
                    if j % 2 == 0:
                        nib = nib & 0x0F
                    else:
                        nib = nib >> 4
                    if nib > 7: nib -= 16
                    var f32val = Float32(nib) * d
                    if f32val > 65000.0: f32val = 65000.0
                    if f32val < -65000.0: f32val = -65000.0
                    f16p.store(blk * 32 + j, Float16(f32val))
            _c_free(raw)
            wp.store(idx, f16_buf)
            wtype.store(idx, 0)
        else:
            var sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
            var _n = Int(sz // 4); var buf = _c_malloc(sz)
            _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
            _close(fd)
            wp.store(idx, buf)
            wtype.store(idx, 1)
    
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
    var dp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(ni * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_c_malloc(Int64(NV * 4))))
    
    # Embedding — token 5
    var emb_path = str_to_cstr("/tmp/tinyllama/token_embd_weight.q4")
    var emb_fd = _open(emb_path, O_RDONLY); emb_path.free()
    var emb_sz = _lseek(emb_fd, 0, SEEK_END); _lseek(emb_fd, 0, SEEK_SET)
    var emb_raw = _c_malloc(emb_sz)
    _read(emb_fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(emb_raw)), emb_sz)
    _close(emb_fd)
    var emb_u8 = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(emb_raw))
    var tok = 5; var bpr_emb = ni // 32; var emb_row_off = tok * bpr_emb * Q4_0_BS
    for i in range(ni):
        var blk = i // 32; var off = i % 32
        var bo = emb_row_off + blk * Q4_0_BS
        var lo = UInt16(emb_u8.load(bo)); var hi = UInt16(emb_u8.load(bo + 1))
        var d = f16_to_f32(lo | (hi << 8))
        var nib = Int32(emb_u8.load(bo + 2 + off // 2))
        if off % 2 == 0:
            nib = nib & 0x0F
        else:
            nib = nib >> 4
        if nib > 7: nib -= 16
        hp.store(i, Float32(nib) * d)
    _c_free(emb_raw)
    print("h[0]:", hp.load(0), "h[100]:", hp.load(100))
    var tf = time.perf_counter()
    
    # ── f16 matmul function (inside main to avoid @extern+cx crash) ──
    def f16_mm(w: UnsafePointer[Float16, MutExternalOrigin],
               x: UnsafePointer[Float32, MutExternalOrigin],
               o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
        var bpr = nc // 32; var nb = (nr + RPW - 1) // RPW
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
        parallelize[func=worker](num_work_items=nb)
    
    # ── Layer loop ──
    for layer in range(NL):
        var lw = N_BASE + layer * N_LF
        
        # RMS norm before attention
        var ss: Float64 = 0.0
        var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LA)))
        for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
        var inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * anp.load(i) * inv)
        
        # Q, K, V f16 matmul
        f16_mm(UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LQ))), bp, qp, nq, ni)
        f16_mm(UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LK))), bp, kp, nkv, ni)
        f16_mm(UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LV))), bp, vp, nkv, ni)
        
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
        
        # O f16 matmul
        f16_mm(UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LO))), qp, bp, ni, nq)
        
        # Residual + attention
        for i in range(ni):
            var v = hp.load(i) + bp.load(i)
            if v != v: v = 0.0
            if v > 100.0: v = 100.0
            if v < -100.0: v = -100.0
            hp.store(i, v)
        
        # RMS norm before FFN
        ss = 0.0
        var fnp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LF)))
        for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
        inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * fnp.load(i) * inv)
        
        # Gate + Up f16 matmul
        f16_mm(UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LG))), bp, gp, NF, ni)
        f16_mm(UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LU))), bp, up, NF, ni)
        
        # SiLU
        for k in range(NF):
            var gv = gp.load(k)
            if gv > 80.0: gv = 80.0
            if gv < -80.0: gv = -80.0
            var rv = (gv / Float32(1.0 + exp(Float64(-gv)))) * up.load(k)
            if rv != rv: rv = 0.0
            if rv > 100.0: rv = 100.0
            if rv < -100.0: rv = -100.0
            gp.store(k, rv)
        
        # Down f16 matmul
        f16_mm(UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LD))), gp, dp, ni, NF)
        
        # Residual + FFN
        for i in range(ni):
            var v = hp.load(i) + dp.load(i)
            if v != v: v = 0.0
            if v > 100.0: v = 100.0
            if v < -100.0: v = -100.0
            hp.store(i, v)
    
    var tg = time.perf_counter()
    
    # Final RMS norm
    ss = 0.0
    var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(W_ON)))
    for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
    inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
    for i in range(ni): bp.store(i, hp.load(i) * onp.load(i) * inv)
    
    # LM head
    f16_mm(UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(W_LM))), bp, lp, NV, ni)
    
    var best = 0; var bv = lp.load(0)
    for i in range(1, NV):
        var v = lp.load(i)
        if v > bv: bv = v; best = i
    print("First 5 logits:", lp.load(0), lp.load(1), lp.load(2), lp.load(3), lp.load(4))
    
    var tt = time.perf_counter()
    print("\n=== Results ===")
    print("Layers:", Int((tg - tf) * 1000.0), "ms (", Int((tg - tf) * 1000.0 / Float64(NL)), "ms/layer)")
    print("LM head:", Int((tt - tg) * 1000.0), "ms")
    print("Total:", Int((tt - tf) * 1000.0), "ms")
    print("tok/s:", Float64(1.0) / (tt - tf))
    print("Best token:", best, "val:", bv)
