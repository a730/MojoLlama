"""MojoLlama persistent Mojo worker — stdin/stdout binary protocol.
Compiled as standalone binary for parallel inference.

Protocol (binary stdin, binary stdout):
  READ:  [4 bytes: n_rows][4 bytes: n_cols][4 bytes: start_row][4 bytes: end_row]
         [n_cols * 4 bytes: input float32]
  WRITE: [(end_row - start_row) * 4 bytes: output float32]
  READ:  [4 bytes: 0xFFFFFFFF] -> exit

Weights are loaded ONCE at startup from pre-extracted .npy files.
"""
from python import Python
from std.python._cpython import PyObjectPtr
from std.memory.unsafe_pointer import alloc
from std.math import sqrt

struct PyArrayObject:
    var data: UnsafePointer[UInt8, MutAnyOrigin]
    var nd: Int
    var dimensions: UnsafePointer[Int, MutAnyOrigin]
    var strides: UnsafePointer[Int, MutAnyOrigin]
    var base: PyObjectPtr
    var descr: PyObjectPtr
    var flags: Int
    var weakreflist: PyObjectPtr

alias NE = 2048; alias NK = 512; alias NF = 8192; alias NL = 16
alias NH = 32; alias NKH = 8; alias HD = 64

def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget; from std.memory import bitcast
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

alias F32x8=SIMD[DType.float32,8];alias U8x16=SIMD[DType.uint8,16]

# ─── Vectorized Q4_0 dot ──────────────────────────────────────────

def q4_dot_vec(scale:Float32,nb:U8x16,x0:F32x8,x1:F32x8,x2:F32x8,x3:F32x8)->Float32:
    var mask=U8x16(15);var lo=(nb & mask).cast[DType.int8]()-8
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

# ─── Regblock Q4_0 matmul (4-row register blocking) ───────────────

def q4_mm(
    w:UnsafePointer[UInt8,MutAnyOrigin],
    inp:UnsafePointer[Float32,MutAnyOrigin],
    res:UnsafePointer[Float32,MutAnyOrigin],
    nr:Int,nc:Int,
):
    var bpr=nc//32
    for blk in range(bpr):
        var ioff=blk*32;var x0=inp.load[width=8](ioff)
        var x1=inp.load[width=8](ioff+8);var x2=inp.load[width=8](ioff+16)
        var x3=inp.load[width=8](ioff+24)
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

# ─── Fused QKV ─────────────────────────────────────────────────────

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
    var bpr=nc//32;var mr=nq if nq>nkv else nkv
    for blk in range(bpr):
        var ioff=blk*32;var x0=inp.load[width=8](ioff)
        var x1=inp.load[width=8](ioff+8);var x2=inp.load[width=8](ioff+16)
        var x3=inp.load[width=8](ioff+24)
        for row in range(mr):
            if row<nq:
                var off=(row*bpr+blk)*18;var lo=wq.load(off);var hi=wq.load(off+1)
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

# ─── Fused FFN gate+up ────────────────────────────────────────────

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
        var ioff=blk*32;var x0=inp.load[width=8](ioff)
        var x1=inp.load[width=8](ioff+8);var x2=inp.load[width=8](ioff+16)
        var x3=inp.load[width=8](ioff+24)
        for row in range(nff):
            var off=(row*bpr+blk)*18;var lo=wg.load(off);var hi=wg.load(off+1)
            var sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo));var nb=wg.load[width=16](off+2)
            og.store(row,og.load(row)+q4_dot_vec(sc,nb,x0,x1,x2,x3))
            lo=wu.load(off);hi=wu.load(off+1)
            sc=f16_to_f32((UInt16(hi)<<8)|UInt16(lo));nb=wu.load[width=16](off+2)
            ou.store(row,ou.load(row)+q4_dot_vec(sc,nb,x0,x1,x2,x3))

# ─── Weight loader helper ─────────────────────────────────────────

def get_w_ptr(name:PythonObject, offset:Int=0)->UnsafePointer[UInt8,MutAnyOrigin]:
    var arr = Python.evaluate("name")  # This won't work - need different approach
    var ptr=UnsafePointer[PyArrayObject,_](unchecked_downcast_value=arr)
    return ptr[].data+offset

# ─── Main loop ─────────────────────────────────────────────────────

def main() raises:
    var np=Python.import_module("numpy");var sys=Python.import_module("sys")
    var bltns=Python.import_module("builtins");var os=Python.import_module("os")
    var stdin=sys.stdin.buffer;var stdout=sys.stdout.buffer
    
    # Pre-load all weight pointers via IPC bridge
    var w_dir="/tmp/mojo_weights"
    var w_names=["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]
    
    # Load all layer weights into arrays
    var w_arrays = Python.evaluate("{}")
    for l in range(NL):
        for nm_i in range(7):
            var nm=w_names[nm_i]
            var key=bltns.str(l)+"_"+nm
            w_arrays[key]=np.load(w_dir+"/blk_"+bltns.str(l)+"_"+nm+"_weight.npy")
    
    # Main processing loop
    var exit_flag = Python.evaluate("b'\\xff\\xff\\xff\\xff'")
    while True:
        # Read header: 4 int32 values
        var hdr=stdin.read(16)
        if bltns.len(hdr)<16: break
        var hdr_np=np.frombuffer(hdr,dtype=np.int32)
        var n_rows=Int(py=hdr_np[0]);var n_cols=Int(py=hdr_np[1])
        var start_row=Int(py=hdr_np[2]);var end_row=Int(py=hdr_np[3])
        
        if n_rows==0xFFFFFFFF: break  # sentinel
        
        # Read input vector
        var inp_raw=stdin.read(n_cols*4)
        if bltns.len(inp_raw)<n_cols*4: break
        
        # Read weight key
        var key_len=Int(py=hdr_np[0])  # reuse as key_len... wait this conflicts
        # Need separate protocol - key first, then dims
        break  # TODO: fix protocol mismatch
    
    print("Worker exiting", file=sys.stderr)
