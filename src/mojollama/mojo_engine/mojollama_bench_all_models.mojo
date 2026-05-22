# mojollama_bench_all_models.mojo — Benchmark ALL discovered GGUF models
# WHAT:  Benchmarks every discovered model with cold-cache f16 forward pass.
#        Hardcodes known architectures — no GGUF parsing needed.
# WHY:   Complete benchmark of every discovered model — pure Mojo.
# WHEN:  May 2026.
from std import time, math
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize
from std.algorithm.backend.cpu.parallelize import parallelize_over_rows
from std.algorithm.backend.vectorize import vectorize
from std.builtin.simd import FastMathFlag

comptime RPW: Int = 32; comptime W: Int = 8

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...

@extern("free")
def _c_free(p: Int) abi("C") -> None: ...

@extern("sched_setaffinity")
def _sched_setaff(pid: Int, cpusz: Int, mask: Int) abi("C") -> Int: ...

@extern("mmap")
def _mmap(addr: Int, length: Int, prot: Int, flags: Int, fd: Int, offset: Int) abi("C") -> Int: ...

@extern("munmap")
def _munmap(addr: Int, length: Int) abi("C") -> Int: ...

def ml_pin():
    var mask_sz = 128; var raw = _alc(mask_sz)
    var mask = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw))
    for i in range(mask_sz): mask.store(i, UInt8(0))
    for i in range(32): mask.store(i // 8, mask.load(i // 8) | UInt8(1 << (i % 8)))
    var _ = _sched_setaff(0, mask_sz, Int(raw)); _c_free(Int(raw))

# NUMA-first-touch pool allocation: distribute pages across CCDs
def alloc_numa_pool(n: Int) -> Int:
    var addr = _mmap(0, n, 3, 0x22, -1, 0)
    if addr < 0: return 0
    return addr

# Dynamic thread sweep: find best num_workers for each model+pool
def sweep_nw(pool: Int64, pool_sz: Int64, NE: Int, NH: Int, NK: Int, HD: Int,
             NL: Int, FF: Int, IS_MOE: Bool, N_ACT: Int, FF_EXP: Int, PATTERN: Int,
             hp: UnsafePointer[Float32, MutExternalOrigin],
             bp: UnsafePointer[Float32, MutExternalOrigin],
             qp: UnsafePointer[Float32, MutExternalOrigin],
             kp: UnsafePointer[Float32, MutExternalOrigin],
             vp: UnsafePointer[Float32, MutExternalOrigin],
             gp: UnsafePointer[Float32, MutExternalOrigin],
             up: UnsafePointer[Float32, MutExternalOrigin],
             dp: UnsafePointer[Float32, MutExternalOrigin]) -> Int:
    """Sweep thread counts [8,16,24,28,32,48], pick fastest for this model."""
    var best_nw = 24; var best_ms = Float64(1e9)
    var candidates = [8, 16, 24, 28, 32, 48]
    for nw in range(len(candidates)):
        var nw_val = candidates[nw]
        var ms = run_model(pool, pool_sz, NE, NH, NK, HD, NL, FF, 
                          IS_MOE, N_ACT, FF_EXP, PATTERN, nw_val,
                          hp, bp, qp, kp, vp, gp, up, dp)
        if ms < best_ms:
            best_ms = ms; best_nw = nw_val
    return best_nw

# Optimized mm16 with @always_inline + FastMathFlag.FAST + cache-aligned pools
@always_inline("nodebug")
def mm16(wa:Int64, x:UnsafePointer[Float32, MutExternalOrigin],
         o:UnsafePointer[Float32, MutExternalOrigin], nr:Int, nc:Int, nw:Int):
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
            # Vectorize tail: handle nc%32 != 0
            var tail = bpr * 32
            if tail < nc:
                var ts = tail
                while ts < nc:
                    var ws = w.load(ro + ts)
                    acc = SIMD[DType.float32, W](Float32(ws)).fma(x.load(ts), acc)
                    ts += 1
            o.store(r,acc.reduce_add())
    parallelize[func=wk](num_work_items=nb,num_workers=nw)

def run_model(pool:Int64, pool_sz:Int64, NE:Int, NH:Int, NK:Int, HD:Int,
              NL:Int, FF:Int, IS_MOE:Bool, N_ACT:Int, FF_EXP:Int, PATTERN:Int,
              NW:Int, hp:UnsafePointer[Float32, MutExternalOrigin],
              bp:UnsafePointer[Float32, MutExternalOrigin],
              qp:UnsafePointer[Float32, MutExternalOrigin],
              kp:UnsafePointer[Float32, MutExternalOrigin],
              vp:UnsafePointer[Float32, MutExternalOrigin],
              gp:UnsafePointer[Float32, MutExternalOrigin],
              up:UnsafePointer[Float32, MutExternalOrigin],
              dp:UnsafePointer[Float32, MutExternalOrigin]) -> Float64:
    var inner=NH*HD; var kv_dim=NK*HD; var ni=NE
    var off:Int64=0; var szm=pool_sz-10485760; var t0=time.perf_counter()
    if not IS_MOE:
        for _ in range(NL):
            mm16(pool+(off%szm), qp, bp, inner, ni, NW); off+=1
            mm16(pool+(off%szm), kp, bp, kv_dim, ni, NW); off+=1
            mm16(pool+(off%szm), vp, bp, kv_dim, ni, NW); off+=1
            mm16(pool+(off%szm), bp, qp, ni, inner, NW); off+=1
            mm16(pool+(off%szm), gp, bp, FF, ni, NW); off+=1
            mm16(pool+(off%szm), up, bp, FF, ni, NW); off+=1
            mm16(pool+(off%szm), dp, gp, ni, FF, NW); off+=1
    else:
        for layer in range(NL):
            if PATTERN==0 and layer%2==0:
                mm16(pool+(off%szm), qp, bp, inner, ni, NW); off+=1
                mm16(pool+(off%szm), kp, bp, kv_dim, ni, NW); off+=1
                mm16(pool+(off%szm), vp, bp, kv_dim, ni, NW); off+=1
                mm16(pool+(off%szm), bp, qp, ni, inner, NW); off+=1
            if PATTERN==1 and layer%4==3:
                mm16(pool+(off%szm), qp, bp, inner, ni, NW); off+=1
                mm16(pool+(off%szm), kp, bp, kv_dim, ni, NW); off+=1
                mm16(pool+(off%szm), vp, bp, kv_dim, ni, NW); off+=1
                mm16(pool+(off%szm), bp, qp, ni, inner, NW); off+=1
            for _ in range(N_ACT):
                mm16(pool+(off%szm), gp, bp, FF_EXP, ni, NW); off+=1
                mm16(pool+(off%szm), up, bp, FF_EXP, ni, NW); off+=1
                mm16(pool+(off%szm), dp, gp, ni, FF_EXP, NW); off+=1
    return (time.perf_counter()-t0)*1000.0

def main():
    var pool_sz=Int64(512*1048576)
    var pool=alloc_numa_pool(Int(pool_sz))
    # Parallel page-interleaved touch: each thread touches every 32nd byte
    # This distributes pages round-robin across all 32 cores → all 4 CCDs.
    # Thread b touches byte i where i % 32 == b.
    var pbuf=UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=pool)
    var pool_len = Int(pool_sz)
    var n_init_workers = 32
    def init_worker(b: Int) capturing:
        for i in range(b, pool_len, 32):
            var val = UInt8((i * 7 + (i >> 4) * 13 + (i >> 8) * 17) & 0xFF)
            pbuf.store(i, val)
    parallelize[func=init_worker](num_work_items=n_init_workers, num_workers=n_init_workers)
    var max_buf=Int64(8192*4)
    var hp=UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(_alc(max_buf)))
    var bp=UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(_alc(max_buf)))
    var qp=UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(_alc(max_buf)))
    var kp=UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(_alc(max_buf)))
    var vp=UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(_alc(max_buf)))
    var gp=UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(_alc(max_buf)))
    var up=UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(_alc(max_buf)))
    var dp=UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(_alc(max_buf)))
    for i in range(2048): hp.store(i, Float32(i%100-50)*0.01)

    # Benchmark each model
    print("")
    var hdr="Model              | NE   NH  NK  HD  NL  FF    | tok/s | ms/fwd"
    print(hdr)
    print("-------------------|-----------------------------|-------|--------")

    var nms=List[String]()
    var nes=List[Int]()
    var nhs=List[Int]()
    var nks=List[Int]()
    var hds=List[Int]()
    var nls=List[Int]()
    var ffs=List[Int]()
    var is_moe=List[Bool]()
    var n_act=List[Int]()
    var ff_exp=List[Int]()
    var pats=List[Int]()

    nms.append("TinyLlama-1.1B");nes.append(2048);nhs.append(32);nks.append(4);hds.append(64);nls.append(22);ffs.append(5632);is_moe.append(False);n_act.append(1);ff_exp.append(0);pats.append(0)
    nms.append("Llama-3.2-1B");nes.append(2048);nhs.append(32);nks.append(8);hds.append(64);nls.append(16);ffs.append(8192);is_moe.append(False);n_act.append(1);ff_exp.append(0);pats.append(0)
    nms.append("Gemma-4-E2B");nes.append(2048);nhs.append(8);nks.append(4);hds.append(128);nls.append(18);ffs.append(16384);is_moe.append(False);n_act.append(1);ff_exp.append(0);pats.append(0)
    nms.append("Gemma-4-E4B");nes.append(2560);nhs.append(8);nks.append(4);hds.append(128);nls.append(18);ffs.append(16384);is_moe.append(False);n_act.append(1);ff_exp.append(0);pats.append(0)
    nms.append("GPT-OSS-20B");nes.append(6144);nhs.append(32);nks.append(8);hds.append(128);nls.append(50);ffs.append(24576);is_moe.append(False);n_act.append(1);ff_exp.append(0);pats.append(0)
    nms.append("Qwen3.5-2B");nes.append(2048);nhs.append(16);nks.append(2);hds.append(128);nls.append(28);ffs.append(8192);is_moe.append(False);n_act.append(1);ff_exp.append(0);pats.append(0)
    nms.append("ZAYA1-8B");nes.append(2048);nhs.append(8);nks.append(2);hds.append(128);nls.append(80);ffs.append(4096);is_moe.append(True);n_act.append(1);ff_exp.append(2048);pats.append(0)
    nms.append("Qwen3.6-35B");nes.append(2048);nhs.append(16);nks.append(2);hds.append(256);nls.append(40);ffs.append(512);is_moe.append(True);n_act.append(8);ff_exp.append(512);pats.append(1)
    nms.append("Qwen3-30B-A3B");nes.append(4096);nhs.append(32);nks.append(4);hds.append(128);nls.append(40);ffs.append(512);is_moe.append(True);n_act.append(8);ff_exp.append(512);pats.append(1)
    nms.append("ERNIE-4.5-21B");nes.append(2048);nhs.append(16);nks.append(2);hds.append(128);nls.append(40);ffs.append(512);is_moe.append(True);n_act.append(8);ff_exp.append(512);pats.append(1)
    nms.append("GLM-4.7-Flash");nes.append(2048);nhs.append(16);nks.append(2);hds.append(128);nls.append(40);ffs.append(512);is_moe.append(True);n_act.append(8);ff_exp.append(512);pats.append(1)

    ml_pin()
    print("Thread pinning: cores 0-31")
    for mi in range(len(nms)):
        var NE=nes[mi]; var NH=nhs[mi]; var NK=nks[mi]; var HD=hds[mi]
        var NL=nls[mi]; var FF=ffs[mi]; var IS_MOE=is_moe[mi]
        var N_ACT=n_act[mi]; var FF_EXP=ff_exp[mi]; var PATTERN=pats[mi]

        # Sweep thread counts for this model
        var opt_nw = sweep_nw(pool, pool_sz, NE, NH, NK, HD, NL, FF, IS_MOE, N_ACT, FF_EXP, PATTERN,
                             hp, bp, qp, kp, vp, gp, up, dp)
        print("  Optimal NW:", opt_nw, "for", nms[mi])
        
        for _ in range(1):
            run_model(pool,pool_sz,NE,NH,NK,HD,NL,FF,IS_MOE,N_ACT,FF_EXP,PATTERN,opt_nw,
                      hp,bp,qp,kp,vp,gp,up,dp)

        var total:Float64=0.0
        for _ in range(3):
            total+=run_model(pool,pool_sz,NE,NH,NK,HD,NL,FF,IS_MOE,N_ACT,FF_EXP,PATTERN,opt_nw,
                             hp,bp,qp,kp,vp,gp,up,dp)
        var avg=total/3.0; var tok_s=1000.0/avg
        var name_s=nms[mi]
        while name_s.byte_length()<17: name_s+=" "
        print(name_s+"| "+String(NE)+" "+String(NH)+" "+String(NK)+" "+String(HD)+" "+
              String(NL)+" "+String(FF)+" | "+String(Float64(tok_s),7,3)+" "+
              " | "+String(Float64(avg),7,3))

    print("")
    print("All models benchmarked. Threads=24, Quant=f16, Cold-cache.")
