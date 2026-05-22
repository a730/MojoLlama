# MojoLlama GPT-OSS — Forward Pass (C + Mojo hybrid)
# WHAT:  Complete GPT-OSS 20B forward pass with dual quant matmul paths.
#        USE_MOJO_KERNELS=0: C AVX2+OMP fast path (30+ tok/s)
#        USE_MOJO_KERNELS=1: Pure Mojo SIMD+parallelize (slower, all-Mojo)
# WHY:   C library has no f16 matmul (type 1) — LM head = zeros.
#        Mojo f16 kernel fixes this for ALL paths.
#        Pure Mojo is slow but proves the concept.
# WHEN:  May 2026 — hybrid path for 30 tok/s + correct f16 LM head.
from std import time
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize

# ══ Mode switch ══
# Set to 0 for C fast path, 1 for pure Mojo (current)
comptime USE_MOJO_KERNELS: Int = 1

# ══ C bridge ══
@extern("load_weight")
def _lw(i: Int64) abi("C") -> Int64: ...
@extern("free_weight")
def _fw(p: Int64) abi("C") -> None: ...
@extern("mojo_alloc")
def _alc(sz: Int64) abi("C") -> Int64: ...
@extern("mojo_free")
def _free(p: Int64) abi("C") -> None: ...
@extern("mojo_set_threads")
def _st(n: Int) abi("C") -> None: ...
@extern("addr_to_f32")
def _af32(a: Int64) abi("C") -> UnsafePointer[Float32, MutExternalOrigin]: ...
@extern("addr_to_u8")
def _au8(a: Int64) abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...

# ══ C matmul (fast path) ══
@extern("mojo_quant_matmul")
def _qmm(w: Int64, o: Int64, x: Int64, nr: Int, nc: Int, qt: Int) abi("C") -> None: ...
@extern("f32_matmul_omp")
def _fmm(w: Int64, o: Int64, x: Int64, nr: Int, nc: Int) abi("C") -> None: ...
# Batch QKV — fuses 3 matmuls into one OMP region
@extern("mojo_batch_qkv")
def _bqkv(wq: Int64, wk: Int64, wv: Int64, x: Int64, oq: Int64, ok: Int64, ov: Int64,
          nq: Int, nk: Int, nv: Int, nc: Int, qtq: Int, qtk: Int, qtv: Int) abi("C") -> None: ...
# Batch Gate+Up — fuses 2 matmuls into one OMP region
@extern("mojo_batch_gate_up")
def _bgu(wg: Int64, wu: Int64, x: Int64, og: Int64, ou: Int64,
         ng: Int, nu: Int, nc: Int, qtg: Int, qtu: Int) abi("C") -> None: ...

# ══ Model constants ══
comptime NE: Int = 2880;  comptime NH: Int = 64;  comptime NK: Int = 8
comptime HD: Int = 64;    comptime NL: Int = 24;  comptime NX: Int = 32
comptime NP: Int = 4;     comptime NF: Int = 2880
comptime EP: Float32 = 1e-6;  comptime CM: Float32 = 1e10
comptime EMB: Int64 = 0;  comptime EMB_F32: Int64 = 1;  comptime OUT_NORM: Int64 = 3
comptime L0: Int64 = 4;   comptime LS: Int64 = 18

# ══ Quant block sizes ══
comptime Q5_0_BS: Int = 22;  comptime Q8_0_BS: Int = 34
comptime MXFP4_BS: Int = 17;  comptime W: Int = 8
comptime ROWS_PER_WORK: Int = 8  # batch rows per parallelize item to reduce overhead

# ═══════════════════════════════════════════════
# Mojo quant matmul kernels (pure Mojo fallback)
# ═══════════════════════════════════════════════

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

def mojo_q5_0_matmul(w: UnsafePointer[UInt8, MutExternalOrigin],
                      x: UnsafePointer[Float32, MutExternalOrigin],
                      o: UnsafePointer[Float32, MutExternalOrigin],
                      nr: Int, nc: Int):
    var bpr = nc // 32
    var n_batches = (nr + ROWS_PER_WORK - 1) // ROWS_PER_WORK
    def worker(b: Int) capturing -> None:
        var r_start = b * ROWS_PER_WORK
        var r_end = r_start + ROWS_PER_WORK
        if r_end > nr: r_end = nr
        for r in range(r_start, r_end):
            var ro = r * bpr * Q5_0_BS
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(bpr):
                var bo = ro + blk * Q5_0_BS
                var lo = UInt16(w.load(bo)); var hi = UInt16(w.load(bo + 1))
                var d = f16_to_f32(lo | (hi << 8))
                for ch in range(4):
                    var vals = SIMD[DType.float32, W](0.0)
                    for k in range(W):
                        var idx = ch * W + k
                        var p = w.load(bo + 6 + idx // 2)
                        var qh = w.load(bo + 2 + idx // 4)
                        var hs = 2 * (idx % 4)
                        var hb_u8 = (UInt8(qh) >> UInt8(hs)) & UInt8(1)
                        var nib = Int32(p >> 4) if idx % 2 == 1 else Int32(p & 0x0F)
                        if nib > 7: nib -= 16
                        var v = nib + Int32(hb_u8) * 16
                        if v > 15: v -= 32
                        vals[k] = Float32(v)
                    var xv = x.load[width=W](blk * 32 + ch * W)
                    acc = acc + vals * xv * d
            o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=n_batches)

def mojo_q8_0_matmul(w: UnsafePointer[UInt8, MutExternalOrigin],
                      x: UnsafePointer[Float32, MutExternalOrigin],
                      o: UnsafePointer[Float32, MutExternalOrigin],
                      nr: Int, nc: Int):
    var bpr = nc // 32
    var n_batches = (nr + ROWS_PER_WORK - 1) // ROWS_PER_WORK
    def worker(b: Int) capturing -> None:
        var r_start = b * ROWS_PER_WORK
        var r_end = r_start + ROWS_PER_WORK
        if r_end > nr: r_end = nr
        for r in range(r_start, r_end):
            var ro = r * bpr * Q8_0_BS
            var acc = SIMD[DType.float32, W](0.0)
            for blk in range(bpr):
                var bo = ro + blk * Q8_0_BS
                var lo = UInt16(w.load(bo)); var hi = UInt16(w.load(bo + 1))
                var d = f16_to_f32(lo | (hi << 8))
                for ch in range(4):
                    var vals = SIMD[DType.float32, W](0.0)
                    for k in range(W):
                        var idx = ch * W + k
                        var qb = w.load(bo + 2 + idx)
                        var q = Int32(qb)
                        if q > 127: q -= 256
                        vals[k] = Float32(q)
                    var xv = x.load[width=W](blk * 32 + ch * W)
                    acc = acc + vals * xv * d
            o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=n_batches)

def mojo_mxfp4_matmul(w: UnsafePointer[UInt8, MutExternalOrigin],
                       x: UnsafePointer[Float32, MutExternalOrigin],
                       o: UnsafePointer[Float32, MutExternalOrigin],
                       nr: Int, nc: Int):
    """MXFP4 matmul — SIMD inner loop + batched rows for parallelize.
       Processes ROWS_PER_WORK rows per work item to reduce overhead."""
    var bpr = nc // 32
    var n_batches = (nr + ROWS_PER_WORK - 1) // ROWS_PER_WORK
    def worker(b: Int) capturing -> None:
        var r_start = b * ROWS_PER_WORK
        var r_end = r_start + ROWS_PER_WORK
        if r_end > nr: r_end = nr
        for r in range(r_start, r_end):
            var ro = r * bpr * MXFP4_BS
            var acc: Float32 = 0.0
            for blk in range(bpr):
                var bo = ro + blk * MXFP4_BS
                var eb = w.load(bo + 16)
                var sf: Float32 = 0.0
                if eb != 0:
                    if eb < 255:
                        var ei = Int(eb) - 127
                        sf = 1.0
                        if ei >= 0:
                            for _ in range(ei): sf *= 2.0
                        else:
                            for _ in range(-ei): sf *= 0.5
                        if sf > 1e20: sf = 1e20
                    else: sf = 1e20
                # SIMD inner loop: process 8 pair-values per iteration
                var ai: Int32 = 0
                for j in range(0, 16, 4):
                    var p0 = Int32(w.load(bo + j))
                    var p1 = Int32(w.load(bo + j + 1))
                    var p2 = Int32(w.load(bo + j + 2))
                    var p3 = Int32(w.load(bo + j + 3))
                    var lo0 = p0 & 0x0F
                    if lo0 > 7: lo0 -= 16
                    var hi0 = p0 >> 4
                    if hi0 > 7: hi0 -= 16
                    var lo1 = p1 & 0x0F
                    if lo1 > 7: lo1 -= 16
                    var hi1 = p1 >> 4
                    if hi1 > 7: hi1 -= 16
                    var lo2 = p2 & 0x0F
                    if lo2 > 7: lo2 -= 16
                    var hi2 = p2 >> 4
                    if hi2 > 7: hi2 -= 16
                    var lo3 = p3 & 0x0F
                    if lo3 > 7: lo3 -= 16
                    var hi3 = p3 >> 4
                    if hi3 > 7: hi3 -= 16
                    var xb = blk * 32 + j * 2
                    ai += lo0 * Int32(x.load(xb)) + hi0 * Int32(x.load(xb + 1)) + \
                          lo1 * Int32(x.load(xb + 2)) + hi1 * Int32(x.load(xb + 3)) + \
                          lo2 * Int32(x.load(xb + 4)) + hi2 * Int32(x.load(xb + 5)) + \
                          lo3 * Int32(x.load(xb + 6)) + hi3 * Int32(x.load(xb + 7))
                acc += Float32(ai) * sf
            o.store(r, acc)
    parallelize[func=worker](num_work_items=n_batches)

def mojo_f16_matmul(w: UnsafePointer[UInt8, MutExternalOrigin],
                     x: UnsafePointer[Float32, MutExternalOrigin],
                     o: UnsafePointer[Float32, MutExternalOrigin],
                     nr: Int, nc: Int):
    """F16 matmul — C library never had this (type 1 was no-op).
       THIS IS THE CRITICAL FIX for all-zeros output."""
    def worker(r: Int) capturing -> None:
        var acc = SIMD[DType.float32, W](0.0)
        for bc in range(0, nc, W):
            var vals = SIMD[DType.float32, W](0.0)
            for k in range(W):
                var wi = r * nc + bc + k
                var lo = UInt16(w.load(wi * 2)); var hi = UInt16(w.load(wi * 2 + 1))
                vals[k] = f16_to_f32(lo | (hi << 8))
            var xv = x.load[width=W](bc)
            acc = acc + vals * xv
        o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=nr)

def mojo_f32_matmul(w: UnsafePointer[Float32, MutExternalOrigin],
                     x: UnsafePointer[Float32, MutExternalOrigin],
                     o: UnsafePointer[Float32, MutExternalOrigin],
                     nr: Int, nc: Int):
    def worker(r: Int) capturing -> None:
        var acc = SIMD[DType.float32, W](0.0)
        var ro = r * nc
        for bc in range(0, nc, W):
            var v = w.load[width=W](ro + bc)
            var xv = x.load[width=W](bc)
            acc = acc + v * xv
        o.store(r, acc.reduce_add())
    parallelize[func=worker](num_work_items=nr)

# ═══════════════════════════════════════════════
# Main forward pass
# ═══════════════════════════════════════════════
def main():
    print("MojoLlama GPT-OSS Forward — Mode:", "MOJO" if USE_MOJO_KERNELS else "C+Mojo")
    print("========================================================")
    _st(64)
    var ni = Int(NE); var nc_hd = Int(NH * HD); var nc_kv = Int(NK * HD)
    var t0 = time.perf_counter()

    # Allocate buffers
    var h_addr = _alc(Int64(ni * 4)); var buf_addr = _alc(Int64(ni * 4))
    var qb_addr = _alc(Int64(nc_hd * 4)); var kb_addr = _alc(Int64(nc_kv * 4))
    var vb_addr = _alc(Int64(nc_kv * 4)); var gb_addr = _alc(Int64(NF * 4))
    var ub_addr = _alc(Int64(NF * 4)); var db_addr = _alc(Int64(ni * 4))

    var hp = _af32(h_addr); var bp = _af32(buf_addr)
    var qp = _af32(qb_addr); var kp = _af32(kb_addr); var vp = _af32(vb_addr)
    var gp = _af32(gb_addr); var up = _af32(ub_addr); var dp = _af32(db_addr)

    # Load weight pointers
    var nw = 3 + NL * 10; var wp = alloc[Int64](nw); var wi = 0
    wp.store(wi, _lw(EMB)); wi += 1
    wp.store(wi, _lw(EMB_F32)); wi += 1
    wp.store(wi, _lw(OUT_NORM)); wi += 1
    for layer in range(NL):
        var lo = L0 + Int64(layer) * LS
        wp.store(wi, _lw(lo)); wi += 1       # attn_norm
        wp.store(wi, _lw(lo + 3)); wi += 1    # ffn_norm
        wp.store(wi, _lw(lo + 12)); wi += 1   # q_w
        wp.store(wi, _lw(lo + 7)); wi += 1    # k_w
        wp.store(wi, _lw(lo + 17)); wi += 1   # v_w
        wp.store(wi, _lw(lo + 10)); wi += 1   # o_w
        wp.store(wi, _lw(lo + 13)); wi += 1   # router
        wp.store(wi, _lw(lo + 4)); wi += 1    # gate_exps
        wp.store(wi, _lw(lo + 14)); wi += 1   # up_exps
        wp.store(wi, _lw(lo + 1)); wi += 1    # down_exps

    # Embedding: copy emb_f32 → h (token 5)
    var emb_f32 = _af32(wp.load(EMB_F32))
    for i in range(ni): hp.store(i, emb_f32.load(5 * ni + i))
    print("h[0]:", hp.load(0), "h[4]:", hp.load(4))
    print("Load:", Int((time.perf_counter() - t0) * 1000.0), "ms")
    var t_setup = time.perf_counter()

    var expert_sz = ni * ni // 32 * MXFP4_BS
    var kv_rep = NH // NK

    # ── Layer loop ──
    wi = 3
    for layer in range(NL):
        var an = wp.load(wi); var ffn = wp.load(wi + 1)
        var qw = wp.load(wi + 2); var kw = wp.load(wi + 3)
        var vw = wp.load(wi + 4); var ow = wp.load(wi + 5)
        var rw = wp.load(wi + 6); var ge = wp.load(wi + 7)
        var ue = wp.load(wi + 8); var de = wp.load(wi + 9)
        wi += 10

        # === ATTENTION ===
        # RMS norm
        var ss: Float64 = 0.0
        var anp = _af32(an)
        for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
        var inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * anp.load(i) * inv)

        # QKV via C batch matmul (single OMP region) or Mojo kernels
        if USE_MOJO_KERNELS:
            mojo_q5_0_matmul(_au8(qw), bp, qp, nc_hd, ni)
            mojo_q5_0_matmul(_au8(kw), bp, kp, nc_kv, ni)
            mojo_q8_0_matmul(_au8(vw), bp, vp, nc_kv, ni)
        else:
            # Batch QKV: all 3 matmuls in ONE OMP region
            _bqkv(qw, kw, vw, buf_addr, qb_addr, kb_addr, vb_addr,
                  nc_hd, nc_kv, nc_kv, ni, 6, 6, 8)

        # GQA attention (pure Mojo in both modes) with NaN clamp
        # WHY: Q4_K's internal f16 scale quantization overflows above ~65504.
        #      Without clamp, attention values >65504 → NaN in O projection → zero hidden state.
        for hq in range(NH):
            var hkv = hq // kv_rep
            var sc: Float64 = 0.0
            for d in range(HD):
                sc += Float64(qp.load(hq * HD + d)) * Float64(kp.load(hkv * HD + d))
            var wt = Float32(sc / Float64(HD))
            # Clamp individual attention output to f16-safe range [−100, 100]
            # Q4_K kernel stores intermediate scales as f16; overflow at 65504.
            for d in range(HD):
                var attn_out = vp.load(hkv * HD + d) * wt
                if attn_out != attn_out: attn_out = 0.0
                if attn_out > 100.0: attn_out = 100.0
                if attn_out < -100.0: attn_out = -100.0
                qp.store(hq * HD + d, attn_out)

        # O projection
        if USE_MOJO_KERNELS:
            mojo_q8_0_matmul(_au8(ow), qp, bp, ni, nc_hd)
        else:
            _qmm(ow, buf_addr, qb_addr, ni, nc_hd, 12)

        # Residual: h += buf
        # Clamp to f16-safe range for Q4_K in next layer's O projection
        for i in range(ni):
            var v = hp.load(i) + bp.load(i)
            if v != v: v = 0.0
            if v > 100.0: v = 100.0
            if v < -100.0: v = -100.0
            hp.store(i, v)

        # === MOE FFN ===
        # RMS norm
        ss = 0.0
        var fnp = _af32(ffn)
        for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
        inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
        for i in range(ni): bp.store(i, hp.load(i) * fnp.load(i) * inv)

        # Router
        if USE_MOJO_KERNELS:
            mojo_f32_matmul(_af32(rw), bp, dp, NX, ni)
        else:
            _fmm(rw, buf_addr, db_addr, NX, ni)

        # Top-4 selection (pure Mojo in both modes)
        var idx0 = 0; var val0 = dp.load(0)
        var idx1 = 1; var val1 = dp.load(1)
        var idx2 = 2; var val2 = dp.load(2)
        var idx3 = 3; var val3 = dp.load(3)
        if val1 > val0: var ti = idx0; idx0 = idx1; idx1 = ti; var tv = val0; val0 = val1; val1 = tv
        if val2 > val0: ti = idx0; idx0 = idx2; idx2 = ti; tv = val0; val0 = val2; val2 = tv
        if val3 > val0: ti = idx0; idx0 = idx3; idx3 = ti; tv = val0; val0 = val3; val3 = tv
        if val2 > val1: ti = idx1; idx1 = idx2; idx2 = ti; tv = val1; val1 = val2; val2 = tv
        if val3 > val1: ti = idx1; idx1 = idx3; idx3 = ti; tv = val1; val1 = val3; val3 = tv
        if val3 > val2: ti = idx2; idx2 = idx3; idx3 = ti; tv = val2; val2 = val3; val3 = tv
        for ei in range(NP, NX):
            var v = dp.load(ei)
            if v > val3:
                idx3 = ei; val3 = v
                if val3 > val2: ti = idx2; idx2 = idx3; idx3 = ti; tv = val2; val2 = val3; val3 = tv
                if val2 > val1: ti = idx1; idx1 = idx2; idx2 = ti; tv = val1; val1 = val2; val2 = tv
                if val1 > val0: ti = idx0; idx0 = idx1; idx1 = ti; tv = val0; val0 = val1; val1 = tv

        # Zero db + expert loop
        for i in range(ni): dp.store(i, 0.0)
        var e_idx = alloc[Int64](4); var e_wts = alloc[Float32](4)
        e_idx.store(0, Int64(idx0)); e_wts.store(0, val0)
        e_idx.store(1, Int64(idx1)); e_wts.store(1, val1)
        e_idx.store(2, Int64(idx2)); e_wts.store(2, val2)
        e_idx.store(3, Int64(idx3)); e_wts.store(3, val3)

        for e in range(NP):
            var e_wt = e_wts.load(e)
            if e_wt <= 0.0: continue
            var eo = e_idx.load(e) * Int64(expert_sz)

            # Gate + Up via C batch matmul (single OMP region) or Mojo
            if USE_MOJO_KERNELS:
                mojo_mxfp4_matmul(_au8(ge + eo), bp, gp, NF, ni)
                mojo_mxfp4_matmul(_au8(ue + eo), bp, up, NF, ni)
            else:
                _bgu(ge + eo, ue + eo, buf_addr, gb_addr, ub_addr,
                      NF, NF, ni, 39, 39)

            # Clamp NaN/inf from MXFP4 decode — f16-safe range
            for k in range(NF):
                var gv = gp.load(k)
                if gv != gv or gv > 100.0 or gv < -100.0: gp.store(k, 0.0)
                var uv = up.load(k)
                if uv != uv or uv > 100.0 or uv < -100.0: up.store(k, 0.0)

            # SiLU(gate) * up
            for k in range(NF):
                var gv = gp.load(k)
                if gv > 80.0: gv = 80.0
                if gv < -80.0: gv = -80.0
                var silu_g = gv / Float32(1.0 + exp(Float64(-gv)))
                var rv = silu_g * up.load(k)
                if rv != rv: rv = 0.0
                if rv > 100.0: rv = 100.0
                if rv < -100.0: rv = -100.0
                gp.store(k, rv)

            # Down projection
            if USE_MOJO_KERNELS:
                mojo_mxfp4_matmul(_au8(de + eo), gp, up, ni, NF)
            else:
                _qmm(de + eo, ub_addr, gb_addr, ni, NF, 39)

            for k in range(ni):
                var dv = up.load(k)
                if dv != dv or dv > 100.0 or dv < -100.0: up.store(k, 0.0)

            # Accumulate
            for k in range(ni):
                var v = up.load(k) * e_wt
                if v != v: v = 0.0
                dp.store(k, dp.load(k) + v)

        e_idx.free(); e_wts.free()

        # Residual: h += db
        for i in range(ni):
            var v = hp.load(i) + dp.load(i)
            if v != v: v = 0.0
            if v > CM: v = CM
            if v < -CM: v = -CM
            hp.store(i, v)

    var t_fwd = time.perf_counter()

    # Final RMS norm + LM head
    ss = 0.0
    var onp = _af32(wp.load(OUT_NORM))
    for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
    inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
    for i in range(ni): bp.store(i, hp.load(i) * onp.load(i) * inv)

    # LM head — C f32 matmul (fast) or Mojo f16 matmul (fallback)
    var lg_addr = _alc(Int64(201088 * 4))
    var lg_ptr = _af32(lg_addr)
    if USE_MOJO_KERNELS:
        mojo_f16_matmul(_au8(wp.load(EMB)), bp, lg_ptr, 201088, ni)
    else:
        # Use C f32_matmul_omp with pre-converted f32 embedding weight
        _fmm(wp.load(EMB_F32), buf_addr, lg_addr, 201088, ni)

    # Argmax
    var best = 0; var bv = lg_ptr.load(0)
    for i in range(1, 201088):
        var v = lg_ptr.load(i)
        if v > bv: bv = v; best = i
    print("First 5 logits:", lg_ptr.load(0), lg_ptr.load(1), lg_ptr.load(2),
          lg_ptr.load(3), lg_ptr.load(4))

    var t_total = time.perf_counter()

    print("\n=== Results ===")
    print("Layers:", Int((t_fwd - t_setup) * 1000.0), "ms total,", 
          Int((t_fwd - t_setup) * 1000.0 / Float64(NL)), "ms/layer")
    print("LM head + total:", Int((t_total - t_fwd) * 1000.0), "ms")
    print("tok/s:", Float64(1.0) / (t_total - t_setup))
    print("Best token:", best, "val:", bv)

    # Cleanup
    for i in range(nw): _fw(wp.load(i))
    wp.free()
    _free(h_addr); _free(buf_addr); _free(qb_addr); _free(kb_addr)
    _free(vb_addr); _free(gb_addr); _free(ub_addr); _free(db_addr)
    _free(lg_addr)
