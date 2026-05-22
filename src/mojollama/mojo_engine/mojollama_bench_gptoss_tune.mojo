# mojollama_bench_gptoss_tune.mojo — Optimize GPT-OSS-20B throughput
# WHAT:  Thread-count sweep for GPT-OSS-20B (NE=6144, NH=32, NK=8, NL=50, FF=24576).
# WHY:   Find optimal thread count to increase 1.4 tok/s baseline. Measure memory BW bottleneck.
# WHEN:  May 2026.
from std import time
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

comptime RPW:Int=32; comptime W:Int=8
@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...

def pool_cold(sz:Int64)->Int64:
    var p=_alc(sz)
    var b=UnsafePointer[UInt8,MutExternalOrigin](unsafe_from_address=Int(p))
    for i in range(Int(min(sz,Int64(536870912)))):
        b.store(i,UInt8((i*7+(i>>4)*13+(i>>8)*17)&0xFF))
    return p

def mm16(wa:Int64, x:UnsafePointer[Float32,MutExternalOrigin],
         o:UnsafePointer[Float32,MutExternalOrigin], nr:Int, nc:Int, nw:Int):
    if wa==0: return
    var w=UnsafePointer[Float16,MutExternalOrigin](unsafe_from_address=Int(wa))
    var bpr=nc//32; var nb=(nr+RPW-1)//RPW
    def wk(b:Int)capturing->None:
        var rs=b*RPW; var re=rs+RPW
        if re>nr: re=nr
        for r in range(rs,re):
            var ro=r*nc; var acc=SIMD[DType.float32,W](0.0)
            for blk in range(bpr):
                for grp in range(4):
                    var w16=w.load[width=W](ro+blk*32+grp*8)
                    acc=w16.cast[DType.float32]().fma[FastMathFlag.FAST](
                        x.load[width=W](blk*32+grp*8), acc)
            o.store(r,acc.reduce_add())
    parallelize[func=wk](num_work_items=nb,num_workers=nw)

def run_pass(pool:Int64, pool_sz:Int64, NE:Int, NH:Int, NK:Int, HD:Int,
             NL:Int, FF:Int, NW:Int,
             hp:UnsafePointer[Float32,MutExternalOrigin],
             bp:UnsafePointer[Float32,MutExternalOrigin],
             qp:UnsafePointer[Float32,MutExternalOrigin],
             kp:UnsafePointer[Float32,MutExternalOrigin],
             vp:UnsafePointer[Float32,MutExternalOrigin],
             gp:UnsafePointer[Float32,MutExternalOrigin],
             up:UnsafePointer[Float32,MutExternalOrigin],
             dp:UnsafePointer[Float32,MutExternalOrigin])->Float64:
    var inner=NH*HD; var kv_dim=NK*HD; var ni=NE
    var off:Int64=0; var szm=pool_sz-10485760; var t0=time.perf_counter()
    for _ in range(NL):
        mm16(pool+(off%szm),qp,bp,inner,ni,NW);off+=1
        mm16(pool+(off%szm),kp,bp,kv_dim,ni,NW);off+=1
        mm16(pool+(off%szm),vp,bp,kv_dim,ni,NW);off+=1
        mm16(pool+(off%szm),bp,qp,ni,inner,NW);off+=1
        mm16(pool+(off%szm),gp,bp,FF,ni,NW);off+=1
        mm16(pool+(off%szm),up,bp,FF,ni,NW);off+=1
        mm16(pool+(off%szm),dp,gp,ni,FF,NW);off+=1
    return (time.perf_counter()-t0)*1000.0

def main():
    var NE=6144; var NH=32; var NK=8; var HD=128; var NL=50; var FF=24576
    var inner=NH*HD; var kv_dim=NK*HD
    
    var pool_sz=Int64(1073741824)
    var pool=pool_cold(pool_sz)
    var max_buf=Int64(131072)
    var hp=UnsafePointer[Float32,MutExternalOrigin](unsafe_from_address=Int(_alc(max_buf)))
    var bp=UnsafePointer[Float32,MutExternalOrigin](unsafe_from_address=Int(_alc(max_buf)))
    var qp=UnsafePointer[Float32,MutExternalOrigin](unsafe_from_address=Int(_alc(max_buf)))
    var kp=UnsafePointer[Float32,MutExternalOrigin](unsafe_from_address=Int(_alc(max_buf)))
    var vp=UnsafePointer[Float32,MutExternalOrigin](unsafe_from_address=Int(_alc(max_buf)))
    var gp=UnsafePointer[Float32,MutExternalOrigin](unsafe_from_address=Int(_alc(max_buf)))
    var up=UnsafePointer[Float32,MutExternalOrigin](unsafe_from_address=Int(_alc(max_buf)))
    var dp=UnsafePointer[Float32,MutExternalOrigin](unsafe_from_address=Int(_alc(max_buf)))
    
    print("")
    print("GPT-OSS-20B (NE=6144, NH=32, NL=50, FF=24576) — Thread Sweep (f16)")
    print("Thrds | ms/fwd | tok/s | vs 1-thr")
    print("------|--------|-------|---------")
    var base:Float64=0.0
    for nw in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64]:
        for _ in range(2):
            run_pass(pool,pool_sz,NE,NH,NK,HD,NL,FF,nw,hp,bp,qp,kp,vp,gp,up,dp)
        var total:Float64=0.0
        for _ in range(3):
            total+=run_pass(pool,pool_sz,NE,NH,NK,HD,NL,FF,nw,hp,bp,qp,kp,vp,gp,up,dp)
        var avg=total/3.0; var tok_s=1000.0/avg
        var nw_s=String(nw)
        while nw_s.byte_length()<5: nw_s+=" "
        if nw==1: base=avg
        var speedup=base/avg
        print(nw_s+"| "+String(Float64(avg),7,1)+" | "+String(Float64(tok_s),7,2)+" | "+String(Float64(speedup),4,2)+"×")
    
    # Memory bandwidth analysis
    print("")
    var ff_w=Float64(FF*NE*2+NE*FF*2)/1048576.0
    var q_w=Float64(inner*NE*2)/1048576.0
    var kv_w=Float64(kv_dim*NE*2*2)/1048576.0
    var o_w=Float64(NE*inner*2)/1048576.0
    var per_layer=ff_w+q_w+kv_w+o_w
    print("Memory traffic per forward pass:")
    print("  FFN (3 matmuls) : "+String(Float64(ff_w*Float64(NL)),7,0)+" MB")
    print("  Attn (4 matmuls): "+String(Float64((q_w+kv_w+o_w)*Float64(NL)),7,0)+" MB")
    print("  Total           : "+String(Float64(per_layer*Float64(NL)),7,0)+" MB")
    print("")
    print("At 12 GB/s (DDR4-3200 8ch):  "+String(Float64(per_layer*Float64(NL)/12000.0*1000.0),7,1)+" ms minimum")
    print("At 20 GB/s (compute-bound):  "+String(Float64(per_layer*Float64(NL)/20000.0*1000.0),7,1)+" ms minimum")
    print("At 4× Q4_0 BW (48 GB/s):    "+String(Float64(per_layer*Float64(NL)/48000.0*1000.0),7,1)+" ms → "+String(Float64(1000.0/(per_layer*Float64(NL)/48000.0*1000.0)),7,2)+" tok/s est.")
