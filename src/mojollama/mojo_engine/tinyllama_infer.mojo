# tinyllama_infer.mojo — Correct TinyLlama inference using f16 .bin weights
# WHAT:  Loads raw f16 .bin files (not Q4_0), uses optimized _mm_f16 matmul.
#         Full forward pass with RMS norm, RoPE, SiLU, GQA.
#         Generates one token at a time. Python-wrapper-ready.
# WHY:   Real inference benchmark. Fixes 1e6 logit overflow from Q4_0 misread.
# WHEN:  2026-05-22

from std import time
from std.math import sqrt, exp, cos, sin
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime NE: Int = 2048;  comptime NH: Int = 32;  comptime NK: Int = 4
comptime HD: Int = 64;    comptime NL: Int = 22;  comptime NF: Int = 5632
comptime NV: Int = 32000; comptime EP: Float32 = 1e-6
comptime W: Int = 8;  comptime RPW: Int = 8

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

# ─── Optimized f16 matmul ───
@always_inline("nodebug")
def _mm_f16(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW; var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc; var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.cast[DType.float32]().fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ─── F32 matmul (for norm weights that are F32) ───
@always_inline("nodebug")
def _mm_f32(wa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
            o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int):
    var w = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wa))
    var nb = (nr + RPW - 1) // RPW; var nw = 32
    def wk(b: Int) capturing:
        var rs = b * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var ro = r * nc; var acc = SIMD[DType.float32, W](0.0)
            for blk in range(0, nc, 32):
                comptime for grp in range(4):
                    var wv = w.load[width=W](ro + blk + grp * 8)
                    var xv = x.load[width=W](blk + grp * 8)
                    acc = wv.fma[FastMathFlag.FAST](xv, acc)
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ─── File loading helper ───
def load_file(dir_cptr: UnsafePointer[UInt8, MutExternalOrigin],
              name_cptr: UnsafePointer[UInt8, MutExternalOrigin]) -> Int64:
    """Load file = dir + name into malloc'd buffer. Returns address or -1."""
    # Construct path: dir + name
    var dlen = 0
    while dir_cptr.load(dlen) != 0:
        dlen += 1
    var nlen = 0
    while name_cptr.load(nlen) != 0:
        nlen += 1
    var path = alloc[UInt8](dlen + nlen + 1)
    for i in range(dlen): path.store(i, dir_cptr.load(i))
    for i in range(nlen): path.store(dlen + i, name_cptr.load(i))
    path.store(dlen + nlen, UInt8(0))
    
    var fd = _open(path, 0)
    if fd < 0: return -1
    var sz = _lseek(fd, 0, 2)
    _lseek(fd, 0, 0)
    var buf = _alc(sz)
    if buf == 0: _close(fd); return -1
    _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
    _close(fd)
    return buf

# ─── Main ───
def main():
    var t0 = time.perf_counter()
    
    # Number of weight files: 3 base + 22*9 = 201
    var nf = 3 + NL * 9
    var wp = alloc[Int64](nf)  # weight pointers
    
    var weights_dir = String("/tmp/weights_tl/")
    var dir_cptr = alloc[UInt8](weights_dir.byte_length() + 1)
    var src = weights_dir.unsafe_ptr()
    for i in range(weights_dir.byte_length()): dir_cptr.store(i, src.load(i))
    dir_cptr.store(weights_dir.byte_length(), UInt8(0))
    
    # Base weights
    var base_names = ["token_embd_weight.bin", "output_norm_weight.bin", "output_weight.bin"]
    for i in range(3):
        var nc = alloc[UInt8](base_names[i].byte_length() + 1)
        for j in range(base_names[i].byte_length()): nc.store(j, UInt8(base_names[i][j]))
        nc.store(base_names[i].byte_length(), UInt8(0))
        var addr = load_file(dir_cptr, nc)
        if addr < 0: print("FAIL: can't load", base_names[i]); return
        wp.store(i, addr)
    
    # Layer weights
    var layer_names = ["_attn_norm_weight.bin", "_ffn_norm_weight.bin",
                       "_attn_q_weight.bin", "_attn_k_weight.bin", "_attn_v_weight.bin",
                       "_attn_output_weight.bin",
                       "_ffn_gate_weight.bin", "_ffn_up_weight.bin", "_ffn_down_weight.bin"]
    for l in range(NL):
        var prefix = String("blk_") + String(l)
        for f in range(9):
            var fname = String("") + prefix + layer_names[f]
            var idx = 3 + l * 9 + f
            var nc = alloc[UInt8](fname.byte_length() + 1)
            for j in range(fname.byte_length()): nc.store(j, UInt8(fname[j]))
            nc.store(fname.byte_length(), UInt8(0))
            var addr = load_file(dir_cptr, nc)
            if addr < 0: print("FAIL: can't load", fname); return
            wp.store(idx, addr)
    
    var t_load = time.perf_counter()
    print("Load:", Int((t_load - t0) * 1000.0), "ms")
    
    # Buffers
    var ni = NE; var nq = NH * HD; var nkv = NK * HD
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni * 4))))
    var qp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nq * 4))))
    var kp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nkv * 4))))
    var vp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nkv * 4))))
    var gp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NF * 4))))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NF * 4))))
    var dp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NV * 4))))
    
    # Embedding: load row for token 1 (BOS)
    var emb_f16 = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wp.load(0)))
    var tok = 1  # BOS token
    for i in range(ni):
        hp.store(i, emb_f16.load(tok * ni + i).cast[DType.float32]())
    
    var t_emb = time.perf_counter()
    print("Embed:", Int((t_emb - t_load) * 1000.0), "ms")
    
    var ss: Float64 = 0.0
    var inv: Float32 = 0.0
    
    var tf = time.perf_counter()
    
    for l in range(NL):
        var lw = 3 + l * 9
        
        # RMS Norm (pre-attention) — norm weight is F32
        var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + 0)))
        ss = 0.0
        for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
        inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * anp.load(i) * inv)
        
        # Q, K, V projections (all f16 weights)
        _mm_f16(Int(wp.load(lw + 2)), bp, qp, nq, ni)
        _mm_f16(Int(wp.load(lw + 3)), bp, kp, nkv, ni)
        _mm_f16(Int(wp.load(lw + 4)), bp, vp, nkv, ni)
        
        # RoPE
        for h in range(nq):
            for d2 in range(0, HD, 2):
                var freq = Float32(tok) / Float32(10000.0 ** (Float64(d2) / Float64(HD)))
                var cv = cos(freq); var sv = sin(freq)
                var x0 = qp.load(h * HD + d2); var x1 = qp.load(h * HD + d2 + 1)
                qp.store(h * HD + d2, x0 * cv - x1 * sv)
                qp.store(h * HD + d2 + 1, x0 * sv + x1 * cv)
        for h in range(nkv):
            for d2 in range(0, HD, 2):
                var freq = Float32(tok) / Float32(10000.0 ** (Float64(d2) / Float64(HD)))
                var cv = cos(freq); var sv = sin(freq)
                var x0 = kp.load(h * HD + d2); var x1 = kp.load(h * HD + d2 + 1)
                kp.store(h * HD + d2, x0 * cv - x1 * sv)
                kp.store(h * HD + d2 + 1, x0 * sv + x1 * cv)
        
        # GQA attention (no KV cache for single token)
        var khr = nq // nkv  # 8
        for hq in range(nq):
            var hkv = hq // khr
            var scores = alloc[Float32](1)  # single position
            var score_sum: Float32 = 0.0
            for d in range(HD):
                scores[0] += qp.load(hq * HD + d) * kp.load(hkv * HD + d)
            scores[0] /= sqrt(Float32(HD))
            score_sum = exp(scores[0])
            # Weighted sum of V
            var wt = exp(scores[0]) / score_sum
            for d in range(HD):
                var attn_out = vp.load(hkv * HD + d) * wt
                qp.store(hq * HD + d, attn_out)
        
        # O projection
        _mm_f16(Int(wp.load(lw + 5)), qp, bp, ni, nq)
        
        # Residual
        for i in range(ni): hp.store(i, hp.load(i) + bp.load(i))
        
        # RMS Norm (pre-FFN)
        var fnp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(lw + 1)))
        ss = 0.0
        for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
        inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * fnp.load(i) * inv)
        
        # FFN gate + up
        _mm_f16(Int(wp.load(lw + 6)), bp, gp, NF, ni)
        _mm_f16(Int(wp.load(lw + 7)), bp, up, NF, ni)
        
        # SiLU gate
        for i in range(NF):
            var gv = gp.load(i)
            if gv < -80.0:
                gv = -80.0
            if gv > 80.0:
                gv = 80.0
            gp.store(i, (gv / (1.0 + exp(-gv))) * up.load(i))
        
        # Down projection
        _mm_f16(Int(wp.load(lw + 8)), gp, dp, ni, NF)
        
        # Residual
        for i in range(ni): hp.store(i, hp.load(i) + dp.load(i))
    
    var tg = time.perf_counter()
    
    # Final RMS Norm
    var onp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp.load(1)))
    ss = 0.0
    for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
    inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
    for i in range(ni): bp.store(i, hp.load(i) * onp.load(i) * inv)
    
    # LM head
    _mm_f16(Int(wp.load(2)), bp, lp, NV, ni)
    
    var tt = time.perf_counter()
    
    # Find best token
    var best = 0; var bv = lp.load(0)
    for i in range(1, NV):
        var v = lp.load(i)
        if v > bv: bv = v; best = i
    
    print("First 5 logits:", lp.load(0), lp.load(1), lp.load(2), lp.load(3), lp.load(4))
    print("\n=== Results ===")
    print("Layers:", Int((tg - tf) * 1000.0), "ms")
    print("LM head:", Int((tt - tg) * 1000.0), "ms")
    print("Total:", Int((tt - tf) * 1000.0), "ms")
    print("tok/s:", Float64(1.0) / (tt - tf))
    print("Best token:", best, "val:", bv)
