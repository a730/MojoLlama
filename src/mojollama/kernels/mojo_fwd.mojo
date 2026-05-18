"""MojoLlama SIMD Forward Pass.
Pure Mojo, synthetic weights, single-core decode.
Quando Mojo 1.5+ aggiungerà unsafe_from_address e file I/O,
questo codice caricherà pesi reali con modifiche minime.
"""

from std.memory.unsafe_pointer import alloc
from std.math import sqrt, exp
from std.python._cpython import PyObjectPtr
from python import Python

alias F32x8 = SIMD[DType.float32, 8]
alias U8x16 = SIMD[DType.uint8, 16]

# Model: Llama 3.2 1B
alias NE = 2048
alias NH = 32
alias NKH = 8
alias NF = 8192
alias NL = 16
alias HD = 64
alias NK = 512  # NKH * HD


# ─── F16 decode ─────────────────────────────────────────────────────

def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var s = (UInt32(h) >> 15) & 1
        var e = (UInt32(h) >> 10) & 0x1f
        var m = UInt32(h) & 0x3ff
        if e == 0:
            if m == 0: return Float32(bitcast[DType.float32](s << 31))
            var mm = m; var c: UInt32 = 0
            while mm > 0: mm >>= 1; c += 1
            var sh = 24 - c
            var vm = m << (sh + 13)
            return Float32(bitcast[DType.float32]((s << 31) | ((UInt32(113 - sh)) << 23) | (vm & 0x7fffff)))
        if e == 31:
            return Float32(bitcast[DType.float32]((s << 31) | 0x7f800000 | (m << 13)))
        return Float32(bitcast[DType.float32]((s << 31) | ((e + 112) << 23) | (m << 13)))


# ─── Q4_0 block dot (vectorized SIMD) ──────────────────────────────

def q4_dot_vec(scale:Float32,nb:U8x16,x0:F32x8,x1:F32x8,x2:F32x8,x3:F32x8)->Float32:
    """Vectorized: extract all 32 nibbles via SIMD shift+mask, interleave."""
    var mask=U8x16(15)
    var lo=(nb & mask).cast[DType.int8]()-8
    var hi=((nb>>UInt8(4))&mask).cast[DType.int8]()-8
    var v0=F32x8(Float32(lo[0]),Float32(hi[0]),Float32(lo[1]),Float32(hi[1]),
                 Float32(lo[2]),Float32(hi[2]),Float32(lo[3]),Float32(hi[3]))
    var v1=F32x8(Float32(lo[4]),Float32(hi[4]),Float32(lo[5]),Float32(hi[5]),
                 Float32(lo[6]),Float32(hi[6]),Float32(lo[7]),Float32(hi[7]))
    var v2=F32x8(Float32(lo[8]),Float32(hi[8]),Float32(lo[9]),Float32(hi[9]),
                 Float32(lo[10]),Float32(hi[10]),Float32(lo[11]),Float32(hi[11]))
    var v3=F32x8(Float32(lo[12]),Float32(hi[12]),Float32(lo[13]),Float32(hi[13]),
                 Float32(lo[14]),Float32(hi[14]),Float32(lo[15]),Float32(hi[15]))
    return(v0*scale*x0).reduce_add()+(v1*scale*x1).reduce_add()+\
           (v2*scale*x2).reduce_add()+(v3*scale*x3).reduce_add()

# ─── Old q4_dot renamed (for reference) ────────────────────────────

def q4_dot_old(scale:Float32,nb:U8x16,x0:F32x8,x1:F32x8,x2:F32x8,x3:F32x8)->Float32:
    @parameter
    fn dg(s:Int,xv:F32x8)->Float32:
        var v=F32x8()
        for j in range(4):
            var b=nb[s+j]
            v[j*2]=Float32(Int8(b&15)-8)
            v[j*2+1]=Float32(Int8((b>>4)&15)-8)
        return(v*scale*xv).reduce_add()
    return dg(0,x0)+dg(4,x1)+dg(8,x2)+dg(12,x3)


# ─── Q4_0 Matmul ────────────────────────────────────────────────────

def q4_mm(
    w:UnsafePointer[UInt8,MutAnyOrigin],
    inp:UnsafePointer[Float32,MutAnyOrigin],
    res:UnsafePointer[Float32,MutAnyOrigin],
    nr:Int,nc:Int,
):
    """Q4_0 matmul with 4-row register blocking + vectorized SIMD."""
    var bpr=nc//32
    for blk in range(bpr):
        var ioff=blk*32
        var x0=inp.load[width=8](ioff);var x1=inp.load[width=8](ioff+8)
        var x2=inp.load[width=8](ioff+16);var x3=inp.load[width=8](ioff+24)
        var row=0
        while row<nr:
            var r0=row;var r1=row+1 if row+1<nr else row
            var r2=row+2 if row+2<nr else row;var r3=row+3 if row+3<nr else row
            var o0=(r0*bpr+blk)*18;var o1=(r1*bpr+blk)*18
            var o2=(r2*bpr+blk)*18;var o3=(r3*bpr+blk)*18
            var lo=w.load(o0);var hi=w.load(o0+1)
            var sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
            var nb=w.load[width=16](o0+2);var t0=q4_dot_vec(sc,nb,x0,x1,x2,x3)
            lo=w.load(o1);hi=w.load(o1+1)
            sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
            nb=w.load[width=16](o1+2);var t1=q4_dot_vec(sc,nb,x0,x1,x2,x3)
            lo=w.load(o2);hi=w.load(o2+1)
            sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
            nb=w.load[width=16](o2+2);var t2=q4_dot_vec(sc,nb,x0,x1,x2,x3)
            lo=w.load(o3);hi=w.load(o3+1)
            sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
            nb=w.load[width=16](o3+2);var t3=q4_dot_vec(sc,nb,x0,x1,x2,x3)
            res.store(r0,res.load(r0)+t0)
            if r1!=r0:res.store(r1,res.load(r1)+t1)
            if r2!=r0:res.store(r2,res.load(r2)+t2)
            if r3!=r0:res.store(r3,res.load(r3)+t3)
            row+=4


# ─── Norms ──────────────────────────────────────────────────────────

def rms_norm(
    x: UnsafePointer[mut=False, type=Float32, origin=_],
    wt: UnsafePointer[mut=False, type=Float32, origin=_],
    res: UnsafePointer[mut=True, type=Float32, origin=_],
    n: Int,
):
    var ss: Float32 = 0.0
    for i in range(n):
        var v = x.load(i)
        ss += v * v
    var r = 1.0 / sqrt(ss / Float32(n) + Float32(1e-5))
    for i in range(n):
        res.store(i, x.load(i) * r * wt.load(i))


def silu_act(
    x: UnsafePointer[mut=False, type=Float32, origin=_],
    res: UnsafePointer[mut=True, type=Float32, origin=_],
    n: Int,
):
    for i in range(n):
        var v = x.load(i)
        res.store(i, v / (1.0 + exp(-v)))


def rope(
    q: UnsafePointer[mut=True, type=Float32, origin=_],
    k: UnsafePointer[mut=True, type=Float32, origin=_],
    cos_val: Float32, sin_val: Float32,
):
    for h in range(NH):
        for d2 in range(HD // 2):
            var off = h * HD + d2 * 2
            var x0 = q.load(off)
            var x1 = q.load(off + 1)
            q.store(off, x0 * cos_val - x1 * sin_val)
            q.store(off + 1, x0 * sin_val + x1 * cos_val)
    for h in range(NKH):
        for d2 in range(HD // 2):
            var off = h * HD + d2 * 2
            var x0 = k.load(off)
            var x1 = k.load(off + 1)
            k.store(off, x0 * cos_val - x1 * sin_val)
            k.store(off + 1, x0 * sin_val + x1 * cos_val)


# ─── Causal Attention (decode) ──────────────────────────────────────

def attn_decode(
    q: UnsafePointer[mut=False, type=Float32, origin=_],
    k_cache: UnsafePointer[mut=False, type=Float32, origin=_],
    v_cache: UnsafePointer[mut=False, type=Float32, origin=_],
    res: UnsafePointer[mut=True, type=Float32, origin=_],
    pos: Int,
):
    var ng = NH // NKH
    var sl = pos + 1
    for h in range(NH):
        var k_h = h // ng
        var mx: Float32 = -1e10
        var sc = alloc[Float32](sl)
        for t in range(sl):
            var s: Float32 = 0.0
            for d in range(HD):
                s += q.load(h * HD + d) * k_cache.load(t * NK + k_h * HD + d)
            sc.store(t, s)
            if s > mx: mx = s
        var se: Float32 = 0.0
        for t in range(sl):
            var e = exp(sc.load(t) - mx)
            sc.store(t, e)
            se += e
        for d in range(HD):
            var tot: Float32 = 0.0
            for t in range(sl):
                tot += sc.load(t) / se * v_cache.load(t * NK + k_h * HD + d)
            res.store(h * HD + d, tot)
        sc.free()


# ─── Weight init ────────────────────────────────────────────────────

def fill_weight(
    buf: UnsafePointer[mut=True, type=UInt8, origin=_],
    n: Int, seed: UInt8,
):
    var s = seed
    for i in range(n):
        s = (s * 7 + 13) & 0xFF
        buf.store(i, s)


def zerofill_f32(
    buf: UnsafePointer[mut=True, type=Float32, origin=_],
    n: Int,
):
    for i in range(n):
        buf.store(i, Float32(0.0))


# ─── Main benchmark ─────────────────────────────────────────────────

def main() raises:
    var tim = Python.import_module("time")
    var builtins = Python.import_module("builtins")
    
    # Weight buffer sizes
    var eb = NE // 32  # blocks per row for n_embd matmuls
    var fb = NF // 32  # blocks per row for FFN matmuls
    
    var wq_sz = NL * NE * eb * 18
    var wk_sz = NL * NK * eb * 18
    var wv_sz = NL * NK * eb * 18
    var wo_sz = NL * NE * eb * 18
    var wg_sz = NL * NF * eb * 18
    var wu_sz = NL * NF * eb * 18
    var wd_sz = NL * NE * fb * 18
    
    var norm_sz = NL * NE
    
    # Allocate weights (Q4_0 uint8 on Mojo heap)
    var w_q = alloc[UInt8](wq_sz)
    var w_k = alloc[UInt8](wk_sz)
    var w_v = alloc[UInt8](wv_sz)
    var w_o = alloc[UInt8](wo_sz)
    var w_g = alloc[UInt8](wg_sz)
    var w_u = alloc[UInt8](wu_sz)
    var w_d = alloc[UInt8](wd_sz)
    var w_an = alloc[Float32](norm_sz)
    var w_fn = alloc[Float32](norm_sz)
    
    # Compute buffers
    var h = alloc[Float32](NE)    # hidden state
    var s1 = alloc[Float32](NE)   # scratch 1 (rms output)
    var qb = alloc[Float32](NE)   # Q / attention output
    var kc = alloc[Float32](NK)   # K buffer (per layer, reused)
    var vc = alloc[Float32](NK)   # V buffer (per layer, reused)
    var gg = alloc[Float32](NF)   # gate * up intermediate (NF sized)
    var ff = alloc[Float32](NF)   # FFN intermediate (NF sized)
    
    # Init weights with varied data
    fill_weight(w_q, wq_sz, UInt8(1))
    fill_weight(w_k, wk_sz, UInt8(2))
    fill_weight(w_v, wv_sz, UInt8(3))
    fill_weight(w_o, wo_sz, UInt8(4))
    fill_weight(w_g, wg_sz, UInt8(5))
    fill_weight(w_u, wu_sz, UInt8(6))
    fill_weight(w_d, wd_sz, UInt8(7))
    for i in range(norm_sz):
        w_an.store(i, Float32(1.0))
        w_fn.store(i, Float32(1.0))
    
    # Init hidden state with varied values
    for i in range(NE):
        h.store(i, Float32(Float32(i % 100 + 1) / 100.0))
    
    # Print info
    print("Mojo SIMD Forward Pass")
    print("  Layers:", NL, "Dim:", NE, "FF:", NF)
    var total_mb = (wq_sz + wk_sz + wv_sz + wo_sz + wg_sz + wu_sz + wd_sz) // (1024 * 1024)
    print("  Q4_0 weights:", total_mb, "MB")
    print()
    
    # ── Layer offsets ──
    var wq_l = wq_sz // NL
    var wk_l = wk_sz // NL
    var wv_l = wv_sz // NL
    var wo_l = wo_sz // NL
    var wg_l = wg_sz // NL
    var wu_l = wu_sz // NL
    var wd_l = wd_sz // NL
    
    # ── Forward Pass ──
    var t0 = tim.time()
    
    for l in range(NL):
        var ln = l * NE
        
        # ── Attention ──
        rms_norm(h, w_an + ln, s1, NE)
        zerofill_f32(qb, NE)
        q4_mm(w_q + l * wq_l, s1, qb, NE, NE)  # Q projection
        zerofill_f32(kc, NK)
        q4_mm(w_k + l * wk_l, s1, kc, NK, NE)  # K
        zerofill_f32(vc, NK)
        q4_mm(w_v + l * wv_l, s1, vc, NK, NE)  # V
        rope(qb, kc, Float32(1.0), Float32(0.0))
        attn_decode(qb, kc, vc, s1, 0)
        zerofill_f32(qb, NE)
        q4_mm(w_o + l * wo_l, s1, qb, NE, NE)  # O projection
        for i in range(NE):
            h.store(i, h.load(i) + qb.load(i))
        
        # ── FFN ──
        rms_norm(h, w_fn + ln, s1, NE)
        zerofill_f32(ff, NF)
        q4_mm(w_g + l * wg_l, s1, ff, NF, NE)  # gate
        silu_act(ff, ff, NF)
        zerofill_f32(gg, NF)
        q4_mm(w_u + l * wu_l, s1, gg, NF, NE)  # up
        for i in range(NF):
            ff.store(i, ff.load(i) * gg.load(i))
        zerofill_f32(gg, NE)
        q4_mm(w_d + l * wd_l, ff, gg, NE, NF)  # down
        for i in range(NE):
            h.store(i, h.load(i) + gg.load(i))
    
    var t1 = tim.time()
    
    var ms = (t1 - t0) * 1000.0
    var fps = 1.0 / (t1 - t0)
    print()
    print("Time:", builtins.round(ms, 1), "ms")
    print("Throughput:", builtins.round(fps, 2), "tok/s (1 core, Q4_0)")
    print()
    print("Done.")
    
    # Cleanup
    w_q.free()
    w_k.free()
    w_v.free()
    w_o.free()
    w_g.free()
    w_u.free()
    w_d.free()
    w_an.free()
    w_fn.free()
    kc.free()
    vc.free()
    h.free()
    s1.free()
    qb.free()
    gg.free()
    ff.free()
