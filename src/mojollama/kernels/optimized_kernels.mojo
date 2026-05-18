"""Optimized Mojo Q4_0 kernels — fused + multi-row + vectorized."""
from python import Python
from std.python._cpython import PyObjectPtr
from std.memory.unsafe_pointer import alloc
from std.math import sqrt, exp

alias F32x8 = SIMD[DType.float32, 8]
alias U8x16 = SIMD[DType.uint8, 16]

def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var s=(UInt32(h)>>15)&1;var e=(UInt32(h)>>10)&0x1f;var m=UInt32(h)&0x3ff
        if e==0:
            if m==0:return Float32(bitcast[DType.float32](s<<31))
            var mm=m;var c:UInt32=0;var vm=m
            while mm>0:mm>>=1;c+=1
            var sh=24-c;vm=m<<(sh+13)
            return Float32(bitcast[DType.float32]((s<<31)|((UInt32(113-sh))<<23)|(vm&0x7fffff)))
        if e==31:return Float32(bitcast[DType.float32]((s<<31)|0x7f800000|(m<<13)))
        return Float32(bitcast[DType.float32]((s<<31)|((e+112)<<23)|(m<<13)))

# ─── Vectorized Q4_0 dot ──────────────────────────────────────────

def q4_dot_vec(scale:Float32,nb:U8x16,x0:F32x8,x1:F32x8,x2:F32x8,x3:F32x8)->Float32:
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

# ─── Scalar Q4_0 dot (baseline) ───────────────────────────────────

def q4_dot_scalar(scale:Float32,nb:U8x16,x0:F32x8,x1:F32x8,x2:F32x8,x3:F32x8)->Float32:
    @parameter
    fn dg(s:Int,xv:F32x8)->Float32:
        var v=F32x8()
        for j in range(4):
            var b=nb[s+j]
            v[j*2]=Float32(Int8(b&15)-8)
            v[j*2+1]=Float32(Int8((b>>4)&15)-8)
        return(v*scale*xv).reduce_add()
    return dg(0,x0)+dg(4,x1)+dg(8,x2)+dg(12,x3)

# ─── Baseline matmul (reference) ──────────────────────────────────

def q4_mm_baseline(
    w:UnsafePointer[UInt8,MutAnyOrigin],
    inp:UnsafePointer[Float32,MutAnyOrigin],
    res:UnsafePointer[Float32,MutAnyOrigin],
    nr:Int,nc:Int,
):
    var bpr=nc//32
    for blk in range(bpr):
        var ioff=blk*32
        var x0=inp.load[width=8](ioff);var x1=inp.load[width=8](ioff+8)
        var x2=inp.load[width=8](ioff+16);var x3=inp.load[width=8](ioff+24)
        for row in range(nr):
            var off=(row*bpr+blk)*18
            var lo=w.load(off);var hi=w.load(off+1)
            var sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
            var nb=w.load[width=16](off+2)
            res.store(row,res.load(row)+q4_dot_scalar(sc,nb,x0,x1,x2,x3))

# ─── Optimization 1: Vectorized dot product ───────────────────────

def q4_mm_vec(
    w:UnsafePointer[UInt8,MutAnyOrigin],
    inp:UnsafePointer[Float32,MutAnyOrigin],
    res:UnsafePointer[Float32,MutAnyOrigin],
    nr:Int,nc:Int,
):
    var bpr=nc//32
    for blk in range(bpr):
        var ioff=blk*32
        var x0=inp.load[width=8](ioff);var x1=inp.load[width=8](ioff+8)
        var x2=inp.load[width=8](ioff+16);var x3=inp.load[width=8](ioff+24)
        for row in range(nr):
            var off=(row*bpr+blk)*18
            var lo=w.load(off);var hi=w.load(off+1)
            var sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
            var nb=w.load[width=16](off+2)
            res.store(row,res.load(row)+q4_dot_vec(sc,nb,x0,x1,x2,x3))

# ─── Optimization 2: 4-row register blocking ──────────────────────

def q4_mm_regblock(
    w:UnsafePointer[UInt8,MutAnyOrigin],
    inp:UnsafePointer[Float32,MutAnyOrigin],
    res:UnsafePointer[Float32,MutAnyOrigin],
    nr:Int,nc:Int,
):
    var bpr=nc//32
    for blk in range(bpr):
        var ioff=blk*32
        var x0=inp.load[width=8](ioff);var x1=inp.load[width=8](ioff+8)
        var x2=inp.load[width=8](ioff+16);var x3=inp.load[width=8](ioff+24)
        var row=0
        while row<nr:
            var r0=row
            var r1=row+1 if row+1<nr else row
            var r2=row+2 if row+2<nr else row
            var r3=row+3 if row+3<nr else row
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

# ─── Optimization 3: Fused QKV ────────────────────────────────────

def q4_mm_fused_qkv(
    wq:UnsafePointer[UInt8,MutAnyOrigin],
    wk:UnsafePointer[UInt8,MutAnyOrigin],
    wv:UnsafePointer[UInt8,MutAnyOrigin],
    inp:UnsafePointer[Float32,MutAnyOrigin],
    oq:UnsafePointer[Float32,MutAnyOrigin],
    ok:UnsafePointer[Float32,MutAnyOrigin],
    ov:UnsafePointer[Float32,MutAnyOrigin],
    nq:Int,nkv:Int,nc:Int,
):
    var bpr=nc//32
    var mr=nq if nq>nkv else nkv
    for blk in range(bpr):
        var ioff=blk*32
        var x0=inp.load[width=8](ioff);var x1=inp.load[width=8](ioff+8)
        var x2=inp.load[width=8](ioff+16);var x3=inp.load[width=8](ioff+24)
        for row in range(mr):
            if row<nq:
                var off=(row*bpr+blk)*18
                var lo=wq.load(off);var hi=wq.load(off+1)
                var sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
                var nb=wq.load[width=16](off+2)
                oq.store(row,oq.load(row)+q4_dot_vec(sc,nb,x0,x1,x2,x3))
            if row<nkv:
                var off=(row*bpr+blk)*18
                var lo=wk.load(off);var hi=wk.load(off+1)
                var sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
                var nb=wk.load[width=16](off+2)
                ok.store(row,ok.load(row)+q4_dot_vec(sc,nb,x0,x1,x2,x3))
                lo=wv.load(off);hi=wv.load(off+1)
                sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
                nb=wv.load[width=16](off+2)
                ov.store(row,ov.load(row)+q4_dot_vec(sc,nb,x0,x1,x2,x3))

# ─── Optimization 4: Fused FFN gate+up ────────────────────────────

def q4_mm_fused_ffn(
    wg:UnsafePointer[UInt8,MutAnyOrigin],
    wu:UnsafePointer[UInt8,MutAnyOrigin],
    inp:UnsafePointer[Float32,MutAnyOrigin],
    og:UnsafePointer[Float32,MutAnyOrigin],
    ou:UnsafePointer[Float32,MutAnyOrigin],
    nff:Int,nc:Int,
):
    var bpr=nc//32
    for blk in range(bpr):
        var ioff=blk*32
        var x0=inp.load[width=8](ioff);var x1=inp.load[width=8](ioff+8)
        var x2=inp.load[width=8](ioff+16);var x3=inp.load[width=8](ioff+24)
        for row in range(nff):
            var off=(row*bpr+blk)*18
            var lo=wg.load(off);var hi=wg.load(off+1)
            var sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
            var nb=wg.load[width=16](off+2)
            og.store(row,og.load(row)+q4_dot_vec(sc,nb,x0,x1,x2,x3))
            lo=wu.load(off);hi=wu.load(off+1)
            sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo))
            nb=wu.load[width=16](off+2)
            ou.store(row,ou.load(row)+q4_dot_vec(sc,nb,x0,x1,x2,x3))

# ─── Benchmark ─────────────────────────────────────────────────────

def main() raises:
    var tim=Python.import_module("time");var bt=Python.import_module("builtins")
    var NE=2048;var NK=512;var NF=8192;var nb=NE//32
    
    # Allocate
    var wsz=NE*nb*18;var wsz_f=NF*nb*18;var wsz_k=NK*nb*18
    var w=alloc[UInt8](wsz);var wf=alloc[UInt8](wsz_f)
    var wq=alloc[UInt8](wsz);var wk=alloc[UInt8](wsz_k)
    var wv=alloc[UInt8](wsz_k);var wg=alloc[UInt8](wsz_f);var wu=alloc[UInt8](wsz_f)
    var x=alloc[Float32](NE);var r=alloc[Float32](NE);var rf=alloc[Float32](NF)
    var ok=alloc[Float32](NK);var ov=alloc[Float32](NK);var gg=alloc[Float32](NF)
    
    # Init with varied data
    var s:UInt8=42
    for i in range(wsz):s=(s*7+13)&0xFF;w.store(i,s)
    for i in range(wsz_f):s=(s*7+13)&0xFF;wf.store(i,s)
    s=1
    for i in range(wsz):s=(s*7+13)&0xFF;wq.store(i,s)
    s=50
    for i in range(wsz_k):s=(s*7+13)&0xFF;wk.store(i,s)
    s=99
    for i in range(wsz_k):s=(s*7+13)&0xFF;wv.store(i,s)
    s=111
    for i in range(wsz_f):s=(s*7+13)&0xFF;wg.store(i,s)
    s=222
    for i in range(wsz_f):s=(s*7+13)&0xFF;wu.store(i,s)
    for i in range(NE):x.store(i,Float32(0.5))
    
    print("═══ Mojo Q4_0 Kernel Optimizations ═══")
    print()
    
    # 1. Dot product: scalar vs vectorized
    var nibs=U8x16(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
    var zv=F32x8(0.5);var sc=Float32(0.5)
    
    var t0=tim.time()
    for i in range(1000000):var _=q4_dot_scalar(sc,nibs,zv,zv,zv,zv)
    var t1=tim.time()
    var dot_sc=(t1-t0)*1000.0
    print("Block dot scalar:",bt.str(bt.round(dot_sc,2)),"ms/M")
    
    t0=tim.time()
    for i in range(1000000):var _=q4_dot_vec(sc,nibs,zv,zv,zv,zv)
    t1=tim.time()
    var dot_vc=(t1-t0)*1000.0
    print("Block dot vectorized:",bt.str(bt.round(dot_vc,2)),"ms/M (",bt.str(bt.round(dot_sc/dot_vc,2)),"x)")
    print()
    
    # 2. Matmul 2048x2048
    print("Matmul 2048x2048:")
    for i in range(NE):r.store(i,Float32(0.0))
    t0=tim.time()
    for i in range(10):q4_mm_baseline(w,x,r,NE,NE)
    t1=tim.time()
    var b1=(t1-t0)*100.0
    print("  Baseline:",bt.str(bt.round(b1,2)),"ms")
    
    for i in range(NE):r.store(i,Float32(0.0))
    t0=tim.time()
    for i in range(10):q4_mm_vec(w,x,r,NE,NE)
    t1=tim.time()
    var b2=(t1-t0)*100.0
    print("  Vec only:",bt.str(bt.round(b2,2)),"ms (",bt.str(bt.round(b1/b2,2)),"x)")
    
    for i in range(NE):r.store(i,Float32(0.0))
    t0=tim.time()
    for i in range(10):q4_mm_regblock(w,x,r,NE,NE)
    t1=tim.time()
    var b3=(t1-t0)*100.0
    print("  Regblock:",bt.str(bt.round(b3,2)),"ms (",bt.str(bt.round(b1/b3,2)),"x)")
    print()
    
    # 3. FFN matmul 8192x2048
    print("Matmul 8192x2048:")
    for i in range(NF):rf.store(i,Float32(0.0))
    t0=tim.time()
    for i in range(5):q4_mm_baseline(wf,x,rf,NF,NE)
    t1=tim.time()
    var f1=(t1-t0)*200.0
    print("  Baseline:",bt.str(bt.round(f1,1)),"ms")
    
    for i in range(NF):rf.store(i,Float32(0.0))
    t0=tim.time()
    for i in range(5):q4_mm_regblock(wf,x,rf,NF,NE)
    t1=tim.time()
    var f2=(t1-t0)*200.0
    print("  Regblock:",bt.str(bt.round(f2,1)),"ms (",bt.str(bt.round(f1/f2,2)),"x)")
    print()
    
    # 4. Fused QKV
    print("Fused QKV:")
    for i in range(NE):r.store(i,Float32(0.0))
    for i in range(NK):ok.store(i,Float32(0.0));ov.store(i,Float32(0.0))
    t0=tim.time()
    for i in range(10):q4_mm_fused_qkv(wq,wk,wv,x,r,ok,ov,NE,NK,NE)
    t1=tim.time()
    var fq=(t1-t0)*100.0
    var sq=b1+b1*Float32(NK)/Float32(NE)*2.0
    print("  Fused:",bt.str(bt.round(fq,2)),"ms")
    print("  Separate:",bt.str(bt.round(sq,1)),"ms (",bt.str(bt.round(sq/fq,1)),"x savings)")
    print()
    
    # 5. Fused FFN
    print("Fused FFN gate+up:")
    for i in range(NF):rf.store(i,Float32(0.0));gg.store(i,Float32(0.0))
    t0=tim.time()
    for i in range(5):q4_mm_fused_ffn(wg,wu,x,rf,gg,NF,NE)
    t1=tim.time()
    var fu=(t1-t0)*200.0
    print("  Fused:",bt.str(bt.round(fu,1)),"ms")
    var su=f1*2.0
    print("  Separate:",bt.str(bt.round(su,1)),"ms (",bt.str(bt.round(su/fu,1)),"x savings)")
    print()
    
    # Full forward estimate
    var base=(b1*1.25+ f1*3)*16.0
    var reg=(b3*1.25+ f2*3)*16.0
    var fused=(fq+b1+ fu+f1)*16.0
    print("═══ Forward pass estimate (1 core) ═══")
    print("Baseline:",bt.str(bt.round(1000/base,2)),"tok/s")
    print("Regblock:",bt.str(bt.round(1000/reg,2)),"tok/s (",bt.str(bt.round(base/reg,2)),"x)")
    print("Fused:",bt.str(bt.round(1000/fused,2)),"tok/s (",bt.str(bt.round(base/fused,2)),"x)")
    print("llama.cpp 1-core:",bt.str(bt.round(16.9,1)),"tok/s")
    print()
    print("Regblock vs llama.cpp:",bt.str(bt.round(16.9/(1000/reg),1)),"x gap")
    print("Fused vs llama.cpp:",bt.str(bt.round(16.9/(1000/fused),1)),"x gap")
    
    w.free();wf.free();wq.free();wk.free();wv.free();wg.free();wu.free()
    x.free();r.free();rf.free();ok.free();ov.free();gg.free()
