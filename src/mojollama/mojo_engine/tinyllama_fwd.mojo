# TinyLlama — 100% Pure Mojo Forward Pass (ZERO C files)
# All compute, weight loading, path construction in Mojo.
from std import time
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize

@extern("malloc")
def _c_malloc(sz: Int64) abi("C") -> Int64: ...
@extern("open")
def _open(path: UnsafePointer[UInt8, MutExternalOrigin], flags: Int) abi("C") -> Int: ...
@extern("read")
def _read(fd: Int, buf: UnsafePointer[UInt8, MutExternalOrigin], cnt: Int64) abi("C") -> Int64: ...
@extern("lseek")
def _lseek(fd: Int, off: Int64, whence: Int) abi("C") -> Int64: ...
@extern("close")
def _close(fd: Int) abi("C") -> Int: ...

comptime O_RDONLY: Int = 0; comptime SEEK_END: Int = 2; comptime SEEK_SET: Int = 0
comptime NE: Int = 2048;  comptime NH: Int = 32;  comptime NK: Int = 4
comptime HD: Int = 64;    comptime NL: Int = 22;  comptime NF: Int = 5632
comptime NV: Int = 32000; comptime EP: Float32 = 1e-6
comptime Q4_0_BS: Int = 18;  comptime W: Int = 8;  comptime RPW: Int = 32
comptime N_BASE: Int = 3;  comptime N_LF: Int = 9
comptime W_EMB: Int = 0;  comptime W_ON: Int = 1
comptime LA: Int = 0;  comptime LF: Int = 1;  comptime LQ: Int = 2
comptime LK: Int = 3;  comptime LV: Int = 4;  comptime LO: Int = 5
comptime LG: Int = 6;  comptime LU: Int = 7;  comptime LD: Int = 8

def f16_to_f32(h: UInt16) -> Float32:
    var s = UInt32(h >> 15); var e = UInt32((h >> 10) & 0x1F)
    var m = UInt32(h & 0x3FF)
    if e == 0:
        if m == 0: return 0.0
        return Float32(Float64(m) * 5.960464477539063e-8)
    if e == 31: return 0.0
    var r = Float32(m | 0x400)
    var ei = Int(e) - 25
    if ei >= 0:
        for _ in range(ei): r *= 2.0
    else:
        for _ in range(-ei): r *= 0.5
    return -r if s != 0 else r

def str_to_cstr(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var blen = s.byte_length()
    var buf = alloc[UInt8](blen + 1)
    var src = s.unsafe_ptr()
    for i in range(blen): buf.store(i, src.load(i))
    buf.store(blen, UInt8(0))
    return buf

@always_inline
def se4(x: Int32) -> Int32:
    """Sign-extend a 4-bit value (0-15 → -8 to 7)."""
    if x > 7: return x - 16
    return x

# ── Q4_0 matmul (optimized: unrolled nibble extraction, 2× fewer iterations) ──
def q4_0_matmul(w: UnsafePointer[UInt8, MutExternalOrigin],
                 x: UnsafePointer[Float32, MutExternalOrigin],
                 o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var bpr = nc // 32; var nb = (nr + RPW - 1) // RPW
    def worker(b: Int) capturing -> None:
        var rs = b * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * bpr * Q4_0_BS; var acc = SIMD[DType.float32, W](0.0)
            for blk in range(bpr):
                var bo = ro + blk * Q4_0_BS
                var lo = UInt16(w.load(bo)); var hi = UInt16(w.load(bo + 1))
                var d = f16_to_f32(lo | (hi << 8))
                # Compile-time unrolled: 2 iterations handling 8 bytes → 16 nibbles each
                comptime for ch in range(2):
                    var b0 = w.load(bo + 2 + ch * 8)
                    var b1 = w.load(bo + 2 + ch * 8 + 1)
                    var b2 = w.load(bo + 2 + ch * 8 + 2)
                    var b3 = w.load(bo + 2 + ch * 8 + 3)
                    var b4 = w.load(bo + 2 + ch * 8 + 4)
                    var b5 = w.load(bo + 2 + ch * 8 + 5)
                    var b6 = w.load(bo + 2 + ch * 8 + 6)
                    var b7 = w.load(bo + 2 + ch * 8 + 7)
                    # 8 nibbles from b0-b3
                    var v0 = Int32(b0); var lo0 = se4(v0 & 0x0F); var hi0 = se4(v0 >> 4)
                    var v1 = Int32(b1); var lo1 = se4(v1 & 0x0F); var hi1 = se4(v1 >> 4)
                    var v2 = Int32(b2); var lo2 = se4(v2 & 0x0F); var hi2 = se4(v2 >> 4)
                    var v3 = Int32(b3); var lo3 = se4(v3 & 0x0F); var hi3 = se4(v3 >> 4)
                    var vs1 = SIMD[DType.float32, W](Float32(lo0), Float32(hi0), Float32(lo1), Float32(hi1),
                                                      Float32(lo2), Float32(hi2), Float32(lo3), Float32(hi3))
                    acc = acc + vs1 * x.load[width=W](blk * 32 + ch * 16) * d
                    # 8 nibbles from b4-b7
                    var v4 = Int32(b4); var lo4 = se4(v4 & 0x0F); var hi4 = se4(v4 >> 4)
                    var v5 = Int32(b5); var lo5 = se4(v5 & 0x0F); var hi5 = se4(v5 >> 4)
                    var v6 = Int32(b6); var lo6 = se4(v6 & 0x0F); var hi6 = se4(v6 >> 4)
                    var v7 = Int32(b7); var lo7 = se4(v7 & 0x0F); var hi7 = se4(v7 >> 4)
                    var vs2 = SIMD[DType.float32, W](Float32(lo4), Float32(hi4), Float32(lo5), Float32(hi5),
                                                      Float32(lo6), Float32(hi6), Float32(lo7), Float32(hi7))
                    acc = acc + vs2 * x.load[width=W](blk * 32 + ch * 16 + 8) * d
            o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=nb)

# ── File path construction (pure Mojo, 0 C) ──
def get_filename(idx: Int) -> String:
    """Get the filename component for a weight index (no directory prefix)."""
    if idx == 0: return "token_embd_weight.q4"
    if idx == 1: return "output_norm_weight.f32"
    if idx == 2: return "output_weight.q4"
    var l = (idx - N_BASE) // N_LF
    var f = (idx - N_BASE) % N_LF
    # Layer file name by index
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

def open_weight_file(idx: Int) -> Int:
    var path: String = "/tmp/tinyllama/" + get_filename(idx)
    var cstr = str_to_cstr(path)
    var fd = _open(cstr, O_RDONLY)
    cstr.free()
    return fd

# ── Main ──
def main():
    print("TinyLlama — 100% Pure Mojo (0 C files)")
    print("========================================")
    var t0 = time.perf_counter()
    
    var nf = N_BASE + NL * N_LF
    var wp = alloc[Int64](nf)
    for i in range(nf): wp.store(i, 0)
    
    for idx in range(nf):
        var fd = open_weight_file(idx)
        if fd < 0: print("Error: open", idx); return
        var sz = _lseek(fd, 0, SEEK_END)
        if sz <= 0: _close(fd); continue
        _lseek(fd, 0, SEEK_SET)
        var buf = _c_malloc(sz)
        if buf == 0: _close(fd); print("Error: OOM", idx); return
        _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
        _close(fd)
        wp.store(idx, buf)
    
    var ts = time.perf_counter()
    print("Load:", Int((ts - t0) * 1000.0), "ms")
    
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
    
    # Embedding
    var emb_u8 = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wp.load(W_EMB)))
    var npr = ni // 32
    for i in range(ni):
        var blk = i // 32; var off = i % 32
        var bo = 5 * npr * Q4_0_BS + blk * Q4_0_BS
        var lo = UInt16(emb_u8.load(bo)); var hi = UInt16(emb_u8.load(bo + 1))
        var d = f16_to_f32(lo | (hi << 8))
        var nib = Int32(emb_u8.load(bo + 2 + off // 2))
        if off % 2 == 0: nib = nib & 0x0F
        else: nib = nib >> 4
        if nib > 7: nib -= 16
        hp.store(i, Float32(nib) * d)
    print("h[0]:", hp.load(0), "h[100]:", hp.load(100))
    var tf = time.perf_counter()
    
    # ── Layer loop ──
    for layer in range(NL):
        var lw = N_BASE + layer * N_LF
        var ss: Float64 = 0.0
        var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LA)))
        for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
        var inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * anp.load(i) * inv)
        
        q4_0_matmul(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LQ))), bp, qp, nq, ni)
        q4_0_matmul(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LK))), bp, kp, nkv, ni)
        q4_0_matmul(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LV))), bp, vp, nkv, ni)
        
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
        
        q4_0_matmul(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LO))), qp, bp, ni, nq)
        
        for i in range(ni):
            var v = hp.load(i) + bp.load(i)
            if v != v: v = 0.0
            if v > 100.0: v = 100.0
            if v < -100.0: v = -100.0
            hp.store(i, v)
        
        ss = 0.0
        var fnp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LF)))
        for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
        inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * fnp.load(i) * inv)
        
        q4_0_matmul(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LG))), bp, gp, NF, ni)
        q4_0_matmul(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LU))), bp, up, NF, ni)
        
        for k in range(NF):
            var gv = gp.load(k)
            if gv > 80.0: gv = 80.0
            if gv < -80.0: gv = -80.0
            var sg = gv / Float32(1.0 + exp(Float64(-gv)))
            var rv = sg * up.load(k)
            if rv != rv: rv = 0.0
            if rv > 100.0: rv = 100.0
            if rv < -100.0: rv = -100.0
            gp.store(k, rv)
        
        q4_0_matmul(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + LD))), gp, dp, ni, NF)
        
        for i in range(ni):
            var v = hp.load(i) + dp.load(i)
            if v != v: v = 0.0
            if v > 100.0: v = 100.0
            if v < -100.0: v = -100.0
            hp.store(i, v)
    
    var tg = time.perf_counter()
    
    ss = 0.0
    var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(W_ON)))
    for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
    inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
    for i in range(ni): bp.store(i, hp.load(i) * onp.load(i) * inv)
    q4_0_matmul(emb_u8, bp, lp, NV, ni)
    
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
