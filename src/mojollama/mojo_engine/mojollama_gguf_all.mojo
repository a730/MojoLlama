# MojoLlama Universal Engine Bench — all GGUF models benchmark
# WHAT:  One file to benchmark ANY model. Set comptime constants for
#        each model's architecture, then build and run.
# WHY:   Single Mojo file for all GGUF models found on the system.
# WHEN:  May 2026.
from std import time, math
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

# ═══ MODEL CONFIG — change for each GGUF ═══
comptime NE: Int = 2048       # n_embd
comptime NH: Int = 32         # n_heads
comptime NK: Int = 4          # n_kv_heads
comptime HD: Int = 64         # head_dim (0 = auto = NE/NH)
comptime NL: Int = 22         # n_layers
comptime FF: Int = 5632       # ff_hidden
comptime NV: Int = 32000      # vocab_size
# MoE
comptime IS_MOE: Bool = False
comptime N_EXP: Int = 1
comptime N_ACT: Int = 1
comptime FF_EXP: Int = 512
# 0=alternating, 1=Qwen hybrid (attn every 4th)
comptime LAYER_PATTERN: Int = 0
# Run config
comptime NW: Int = 24
comptime RPW: Int = 32; comptime W: Int = 8; comptime EP: Float32 = 1e-6
comptime QUANT_TYPE: Int = 1  # 1=f16, 39=MXFP4

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...

# ═══ f16 matmul ═══
def mm16(wa:Int64, x:UnsafePointer[Float32, MutExternalOrigin],
         o:UnsafePointer[Float32, MutExternalOrigin], nr:Int, nc:Int, nw:Int):
    if wa==0: return
    var w=UnsafePointer[Float16, MutExternalOrigin>(unsafe_from_address=Int(wa))
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
    parallelize[func=wk](num_work_items=nb,num_workers=nw)

# ═══ MXFP4 matmul ═══
def mm_mx(wa:Int64, x:UnsafePointer[Float32, MutExternalOrigin],
          o:UnsafePointer[Float32, MutExternalOrigin], nr:Int, nc:Int, nw:Int):
    if wa==0: return
    var w=UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(wa))
    var bpr=nc//32; var bs=17
    def wk(b:Int) capturing->None:
        var rs=b*RPW; var re=rs+RPW
        if re>nr: re=nr
        for r in range(rs,re):
            var ro=r*bpr*bs; var acc:Float32=0.0
            for blk in range(bpr):
                var bo=ro+blk*bs
                var eb=w.load(bo+16)
                var sf:Float32=0.0
                if eb!=0:
                    if eb<255:
                        var ei=Int(eb)-127; sf=1.0
                        if ei>=0:
                            for _ in range(ei): sf*=2.0
                        else:
                            for _ in range(-ei): sf*=0.5
                    else: sf=1e20
                var ai:Int32=0
                for j in range(16):
                    var p=w.load(bo+j)
                    var lo=Int32(p&0x0F)
                    if lo>7: lo-=16
                    var hi=Int32(p>>4)
                    if hi>7: hi-=16
                    ai+=lo*Int32(x.load(blk*32+j*2))+hi*Int32(x.load(blk*32+j*2+1))
                acc+=Float32(ai)*sf
            o.store(r,acc)
    parallelize[func=wk](num_work_items=nr,num_workers=nw)

# Dispatch
def mm(wa:Int64, x:UnsafePointer[Float32, MutExternalOrigin],
       o:UnsafePointer[Float32, MutExternalOrigin], nr:Int, nc:Int, nw:Int):
    @parameter
    if QUANT_TYPE==1: mm16(wa,x,o,nr,nc,nw)
    @parameter
    if QUANT_TYPE==39: mm_mx(wa,x,o,nr,nc,nw)

# ═══ Forward pass ═══
def run_fwd(pool:Int64, pool_sz:Int64, inner:Int, kv_dim:Int, ni:Int, eff:Int) -> Float64:
    var off:Int64=0; var szm=pool_sz-10485760; var nq=inner; var nkv=kv_dim
    var t0=time.perf_counter()
    @parameter
    if not IS_MOE:
        for _ in range(NL):
            mm(pool+(off%szm), qp, bp, nq, ni, NW); off+=1
            mm(pool+(off%szm), kp, bp, nkv, ni, NW); off+=1
            mm(pool+(off%szm), vp, bp, nkv, ni, NW); off+=1
            mm(pool+(off%szm), bp, qp, ni, nq, NW); off+=1
            mm(pool+(off%szm), gp, bp, FF, ni, NW); off+=1
            mm(pool+(off%szm), up, bp, FF, ni, NW); off+=1
            mm(pool+(off%szm), dp, gp, ni, FF, NW); off+=1
    @parameter
    if IS_MOE:
        for layer in range(NL):
            @parameter
            if LAYER_PATTERN==0:
                if layer%2==0:
                    mm(pool+(off%szm), qp, bp, nq, ni, NW); off+=1
                    mm(pool+(off%szm), kp, bp, nkv, ni, NW); off+=1
                    mm(pool+(off%szm), vp, bp, nkv, ni, NW); off+=1
                    mm(pool+(off%szm), bp, qp, ni, nq, NW); off+=1
            @parameter
            if LAYER_PATTERN==1:
                if layer%4==3:
                    mm(pool+(off%szm), qp, bp, nq, ni, NW); off+=1
                    mm(pool+(off%szm), kp, bp, nkv, ni, NW); off+=1
                    mm(pool+(off%szm), vp, bp, nkv, ni, NW); off+=1
                    mm(pool+(off%szm), bp, qp, ni, nq, NW); off+=1
            for _ in range(N_ACT):
                mm(pool+(off%szm), gp, bp, eff, ni, NW); off+=1
                mm(pool+(off%szm), up, bp, eff, ni, NW); off+=1
                mm(pool+(off%szm), dp, gp, ni, eff, NW); off+=1
    return (time.perf_counter()-t0)*1000.0

# Global buffers (allocated in main, used by run_fwd)
var g_hp: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin]()
var g_bp = g_hp; var g_qp = g_hp; var g_kp = g_hp; var g_vp = g_hp
var g_gp = g_hp; var g_up = g_hp; var g_dp = g_hp

# These need to be accessible inside run_fwd's inner closures
# Mojo 1.0.0b1 limitation: closures can't access main()'s locals
# So we define them at module level (but Mojo doesn't support globals either!)
# Alternative: pass everything as params to a helper

# Let's use a different approach: inline the forward pass in main()
def main():
    # Use local buffer pointers and inline the loop
    print("=== MojoLlama Universal Engine Bench ===")
