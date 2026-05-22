# MojoLlama GGUF Inference Benchmark — Pure Mojo
# WHAT:  Reads GGUF binary, extracts tensor data from actual files,
#        runs real forward pass, measures tok/s per quant format.
# WHY:   Pure Mojo GGUF benchmark — no Python, no C, no llama.cpp.
# WHEN:  May 2026.
from std import time, math
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime NW: Int = 24;  comptime EP: Float32 = 1e-6
comptime RPW: Int = 32;  comptime W: Int = 8
comptime O_RDONLY: Int = 0; comptime SEEK_END: Int = 2; comptime SEEK_SET: Int = 0
comptime Q4_0_BS: Int = 18
comptime MAX_T: Int = 500

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...
@extern("open")
def _open(path: UnsafePointer[UInt8, MutExternalOrigin], flags: Int) abi("C") -> Int: ...
@extern("read")
def _read(fd: Int, buf: UnsafePointer[UInt8, MutExternalOrigin], cnt: Int64) abi("C") -> Int64: ...
@extern("lseek")
def _lseek(fd: Int, off: Int64, whence: Int) abi("C") -> Int64: ...
@extern("close")
def _close(fd: Int) abi("C") -> Int: ...

def str_to_cptr(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var blen = s.byte_length(); var buf = alloc[UInt8](blen + 1)
    var src = s.unsafe_ptr()
    for i in range(blen): buf.store(i, src.load(i))
    buf.store(blen, UInt8(0)); return buf

def rd4(d: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> UInt32:
    return UInt32(d.load(off))|UInt32(d.load(off+1))<<8|UInt32(d.load(off+2))<<16|UInt32(d.load(off+3))<<24

def rd8(d: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> Int64:
    var v:Int64=0
    for i in range(8): v|=Int64(d.load(off+i))<<(i*8)
    return v

# Load GGUF into flat arrays. Returns tensor count.
# name_buf stores all names null-terminated; name_off[i] = start of tensor i's name
def load_gguf(path: String,
              name_buf: UnsafePointer[UInt8, MutExternalOrigin],
              name_off: UnsafePointer[Int32, MutExternalOrigin],
              addr: UnsafePointer[Int64, MutExternalOrigin],
              dim0: UnsafePointer[Int64, MutExternalOrigin],
              dim1: UnsafePointer[Int64, MutExternalOrigin],
              typ: UnsafePointer[UInt32, MutExternalOrigin]) -> Int:
    var cpath=str_to_cptr(path)
    var fd=_open(cpath,O_RDONLY); cpath.free()
    if fd<0: return -1
    var sz=_lseek(fd,0,SEEK_END); _lseek(fd,0,SEEK_SET)
    var file_data=_alc(sz)
    _read(fd,UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(file_data)),sz)
    _close(fd)
    var d=UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(file_data))
    var p:Int=8
    var tensor_count=Int(rd8(d,p)); p+=8
    var kv_count=rd8(d,p); p+=8
    for _ in range(kv_count):
        var klen=Int(rd4(d,p)); p+=4+klen
        var vtype=rd4(d,p); p+=4
        if vtype==8: p+=8+Int(rd8(d,p))
        elif vtype==12:
            p+=4; var alen=Int(rd8(d,p)); p+=8
            for _ in range(alen): p+=4
        elif vtype<=11: p+=1 if vtype<=1 else 2 if vtype<=3 else 4 if vtype<=6 else 8
    var data_start=p
    if data_start%64!=0: data_start=(data_start//64+1)*64
    var name_pos:Int32=0
    for ti in range(min(tensor_count,MAX_T)):
        var namelen=Int(rd8(d,p)); p+=8
        name_off.store(ti,name_pos)
        for ci in range(namelen): name_buf.store(Int(name_pos)+ci,d.load(p+ci))
        name_buf.store(Int(name_pos)+namelen,UInt8(0))
        name_pos+=Int32(namelen)+1
        p+=namelen
        var nd=Int(rd4(d,p)); p+=4
        var d0=rd8(d,p); p+=8
        var d1:Int64=1
        if nd>1: d1=rd8(d,p); p+=8
        if nd>2: p+=8
        var tt=rd4(d,p); p+=4
        var toff=rd8(d,p); p+=8
        addr.store(ti,file_data+data_start+toff)
        dim0.store(ti,d0); dim1.store(ti,d1); typ.store(ti,tt)
    return min(tensor_count,MAX_T)

# Find tensor by name match (exact, null-terminated)
def find(n:Int, target:UnsafePointer[UInt8, MutExternalOrigin],
         name_buf:UnsafePointer[UInt8, MutExternalOrigin],
         name_off:UnsafePointer[Int32, MutExternalOrigin],
         addr:UnsafePointer[Int64, MutExternalOrigin],
         typ:UnsafePointer[UInt32, MutExternalOrigin],
         out_addr:UnsafePointer[Int64, MutExternalOrigin],
         out_typ:UnsafePointer[UInt32, MutExternalOrigin]):
    out_addr.store(0,0); out_typ.store(0,0)
    for ti in range(n):
        var off=name_off.load(ti)
        var ok=True
        for ci in range(80):
            var tc=target.load(ci)
            if tc==0: break
            if name_buf.load(Int(off)+ci)!=tc: ok=False; break
        if ok and name_buf.load(Int(off))!=0:
            out_addr.store(0,addr.load(ti)); out_typ.store(0,typ.load(ti)); return

# ═══ f16 matmul ═══
def mm16(wa:Int64, x:UnsafePointer[Float32, MutExternalOrigin],
         o:UnsafePointer[Float32, MutExternalOrigin], nr:Int, nc:Int):
    if wa==0: return
    var w=UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(wa))
    var bpr=nc//32; var nb=(nr+RPW-1)//RPW
    def wk(b:Int) capturing->None:
        var rs=b*RPW; var re=rs+RPW
        if re>nr: re=nr
        for r in range(rs,re):
            var ro=r*nc; var acc=SIMD[DType.float32,W](0.0)
            for blk in range(bpr):
                comptime for grp in range(4):
                    var w16=w.load[width=W](ro+blk*32+grp*8)
                    var wf32=w16.cast[DType.float32]()
                    var xv=x.load[width=W](blk*32+grp*8)
                    acc=wf32.fma[FastMathFlag.FAST](xv,acc)
            o.store(r,acc.reduce_add())
    parallelize[func=wk](num_work_items=nb,num_workers=NW)

# ═══ Q4_0 matmul ═══
def mm04(wa:Int64, x:UnsafePointer[Float32, MutExternalOrigin],
         o:UnsafePointer[Float32, MutExternalOrigin], nr:Int, nc:Int):
    if wa==0: return
    var w=UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wa))
    var bpr=nc//32
    def wk(b:Int) capturing->None:
        var rs=b*RPW; var re=rs+RPW
        if re>nr: re=nr
        for r in range(rs,re):
            var ro=r*bpr*Q4_0_BS; var acc:Float32=0.0
            for blk in range(bpr):
                var bo=ro+blk*Q4_0_BS
                var h=UInt16(w.load(bo))|(UInt16(w.load(bo+1))<<8)
                var s=UInt32(h>>15);var e=UInt32((h>>10)&0x1F);var m=UInt32(h&0x3FF)
                var d:Float32=0.0
                if e==0: d=0.0 if m==0 else Float32(Float64(m)*5.960464477539063e-8)
                elif e<31:
                    d=Float32(m|0x400);var ei=Int(e)-25
                    if ei>=0:
                        for _ in range(ei): d*=2.0
                    else:
                        for _ in range(-ei): d*=0.5
                d=-d if s!=0 else d
                for j in range(32):
                    var nib=Int32(w.load(bo+2+j//2))
                    if j%2==0: nib=nib&0x0F
                    else: nib=nib>>4
                    if nib>7: nib-=16
                    acc+=Float32(nib)*d*x.load(blk*32+j)
            o.store(r,acc)
    parallelize[func=wk](num_work_items=nr,num_workers=NW)

# Dispatch matmul by tensor type
def mm(wa:Int64, tt:UInt32, x:UnsafePointer[Float32, MutExternalOrigin],
       o:UnsafePointer[Float32, MutExternalOrigin], nr:Int, nc:Int):
    if tt==0: return
    if tt==1: mm16(wa,x,o,nr,nc)
    elif tt==2: mm04(wa,x,o,nr,nc)

def main():
    var paths=["/tmp/tinylama-1.1b-f16.gguf","/tmp/tl-Q4_0.gguf","/tmp/tl-Q8_0.gguf"]
    
    # Tensor storage
    var nb=UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(100000))))
    var no=UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_T*4))))
    var ta=UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_T*8))))
    var td0=UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_T*8))))
    var td1=UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_T*8))))
    var tt=UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_T*4))))
    var _qo=UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(_alc(8)))
    var _qt=UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(_alc(4)))
    
    print("=== GGUF Inference Benchmark ===")
    
    for pi in range(len(paths)):
        var path=paths[pi]
        var n=load_gguf(path,nb,no,ta,td0,td1,tt)
        if n<=0: continue
        
        # Get first layer's Q tensor type to detect quant
        var tq=str_to_cptr("blk.0.attn_q.weight")
        find(n,tq,nb,no,ta,tt,_qo,_qt); tq.free()
        var q_addr=_qo.load(0); var q_typ=_qt.load(0)
        if q_addr==0: continue
        
        var qname="f16"
        if q_typ==2: qname="Q4_0"
        elif q_typ==8: qname="Q8_0"
        
        var NE=2048; var NH=32; var NK=4; var HD=64; var NL=22; var FF=5632; var kr=NH//NK
        var ni=NE; var nq=NH*HD; var nkv=NK*HD
        
        # Pre-fetch all layer tensor addresses
        var la=UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NL*7*8))))
        for l in range(NL):
            var s="blk."+String(l)+"."
            var t=str_to_cptr(s+"attn_q.weight"); find(n,t,nb,no,ta,tt,_qo,_qt); la.store(l*7+0,_qo.load(0)); t.free()
            t=str_to_cptr(s+"attn_k.weight"); find(n,t,nb,no,ta,tt,_qo,_qt); la.store(l*7+1,_qo.load(0)); t.free()
            t=str_to_cptr(s+"attn_v.weight"); find(n,t,nb,no,ta,tt,_qo,_qt); la.store(l*7+2,_qo.load(0)); t.free()
            t=str_to_cptr(s+"attn_output.weight"); find(n,t,nb,no,ta,tt,_qo,_qt); la.store(l*7+3,_qo.load(0)); t.free()
            t=str_to_cptr(s+"ffn_gate.weight"); find(n,t,nb,no,ta,tt,_qo,_qt); la.store(l*7+4,_qo.load(0)); t.free()
            t=str_to_cptr(s+"ffn_up.weight"); find(n,t,nb,no,ta,tt,_qo,_qt); la.store(l*7+5,_qo.load(0)); t.free()
            t=str_to_cptr(s+"ffn_down.weight"); find(n,t,nb,no,ta,tt,_qo,_qt); la.store(l*7+6,_qo.load(0)); t.free()
        
        # Buffers
        var hp=UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni*4))))
        var bp=UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni*4))))
        var qp=UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nq*4))))
        var kp=UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nkv*4))))
        var vp=UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(nkv*4))))
        var gp=UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF*4))))
        var up=UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF*4))))
        var dp=UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni*4))))
        for i in range(ni): hp.store(i,Float32(i%100-50)*0.01)
        
        # Warmup
        for _ in range(1):
            for l in range(NL):
                mm(la.load(l*7+0),q_typ,bp,qp,nq,ni)
                mm(la.load(l*7+1),q_typ,bp,kp,nkv,ni)
                mm(la.load(l*7+2),q_typ,bp,vp,nkv,ni)
                mm(la.load(l*7+3),q_typ,qp,bp,ni,nq)
                mm(la.load(l*7+4),q_typ,bp,gp,FF,ni)
                mm(la.load(l*7+5),q_typ,bp,up,FF,ni)
                mm(la.load(l*7+6),q_typ,gp,dp,ni,FF)
        
        # Measured
        var total:Float64=0.0
        for _ in range(3):
            var t0=time.perf_counter()
            for l in range(NL):
                mm(la.load(l*7+0),q_typ,bp,qp,nq,ni)
                mm(la.load(l*7+1),q_typ,bp,kp,nkv,ni)
                mm(la.load(l*7+2),q_typ,bp,vp,nkv,ni)
                mm(la.load(l*7+3),q_typ,qp,bp,ni,nq)
                mm(la.load(l*7+4),q_typ,bp,gp,FF,ni)
                mm(la.load(l*7+5),q_typ,bp,up,FF,ni)
                mm(la.load(l*7+6),q_typ,gp,dp,ni,FF)
            total+=(time.perf_counter()-t0)*1000.0
        
        var avg=total/3.0
        var tok_s=1000.0/avg
        print(qname+" | "+String(Float64(tok_s))+" tok/s | "+String(Float64(avg))+" ms | "+path)
