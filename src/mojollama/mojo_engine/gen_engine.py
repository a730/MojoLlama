#!/usr/bin/env python3
"""Generate Mojo engine source with hardcoded literal paths for all layers.
Workaround: Mojo 0.26.2 String concat + C bridge crashes, so we generate
explicit literal paths for each layer."""
import os, sys

MODEL = 'gpt-oss'
OUT = '/tmp/mojo_weights/gpt-oss'
L = 24
N = 2880
NH = 32
NKH = 8
HD = 96
FF = 2880

template = '''# MojoLlama Engine — Full Forward (%d layers)
# AUTO-GENERATED with hardcoded literal paths.
# Workaround for Mojo 0.26.2: String concat + C bridge crashes.
from std.prelude import *
from std import time

@extern("read_u8") fn _u8(addr: Int64) -> UInt8: ...
@extern("read_f32") fn _f32(addr: Int64) -> Float32: ...
@extern("read_i32") fn _i32(addr: Int64) -> Int32: ...
@extern("load_file_size") fn _fsz(path: String) -> Int64: ...
@extern("load_file_data") fn _ld(path: String) -> Int64: ...
@extern("free_buf") fn _fr(ptr: Int64): ...

comptime BS5 = 22; comptime BS8 = 34; comptime BS4 = 17
fn f16(h: UInt16) -> Float32:
    fn _h(h: UInt16) -> Float32:
        var s = UInt32(h>>15); var e=UInt32((h>>10)&0x1F); var m=UInt32(h&0x3FF)
        if e==0: return 0.0 if m==0 else Float32(Float64(m)*5.960464477539063e-8)
        if e==31: return 0.0
        var r=Float32(m|0x400); var ei=Int(e)-25
        if ei>=0: for _ in range(ei): r*=2.0
        else: for _ in range(-ei): r*=0.5
        return -r if s!=0 else r
    return _h(h)
fn e8m0(e: UInt8)->Float32:
    if e==0: return 0.0; if e>=255: return 1e20
    var ei=Int(e)-127; var r:Float32=1.0
    if ei>=0: for _ in range(ei): r*=2.0; else: for _ in range(-ei): r*=0.5
    return 1e20 if r>1e20 else r

# Kernels
fn q5(addr:Int64, x:List[Float32], nr:Int, nc:Int, dst:List[Float32]):
    var bp=nc//32; var s=BS5
    for r in range(nr):
        var a:Float32=0.0; var o=r*bp*s
        for blk in range(bp):
            var wo=o+blk*s; var lo=UInt16(_u8(addr+Int64(wo))); var hi=UInt16(_u8(addr+Int64(wo+1)))
            var d=f16(lo|(hi<<8))
            for j in range(16):
                var p=_u8(addr+Int64(wo+6+j)); var qh=_u8(addr+Int64(wo+2+(j//4)))
                var hs=2*UInt8(j%4); var h0=Int32((qh>>hs)&UInt8(1)); var h1=Int32((qh>>(hs+1))&UInt8(1))
                var nl=Int32(p&UInt8(0x0F)); var nh=Int32(p>>4)
                var v0=nl+h0*16; var v1=nh+h1*16
                if v0>15:v0-=32; if v1>15:v1-=32
                a+=Float32(v0)*x[blk*32+j*2]*d; a+=Float32(v1)*x[blk*32+j*2+1]*d
        dst.append(a)

fn q8(addr:Int64, x:List[Float32], nr:Int, nc:Int, dst:List[Float32]):
    var bp=nc//32; var s=BS8
    for r in range(nr):
        var a:Float32=0.0; var o=r*bp*s
        for blk in range(bp):
            var wo=o+blk*s; var lo=UInt16(_u8(addr+Int64(wo))); var hi=UInt16(_u8(addr+Int64(wo+1)))
            var d=f16(lo|(hi<<8))
            for j in range(32):
                var q=Int32(_u8(addr+Int64(wo+2+j))); if q>127:q-=256
                a+=Float32(q)*x[blk*32+j]*d
        dst.append(a)

fn mxfp4(addr:Int64, x:List[Float32], nr:Int, nc:Int, dst:List[Float32]):
    var bp=nc//32; var s=BS4
    for r in range(nr):
        var a:Float32=0.0; var o=r*bp*s
        for blk in range(bp):
            var wo=o+blk*s; var sf=e8m0(_u8(addr+Int64(wo+16)))
            var ai:Int32=0
            for j in range(16):
                var p=_u8(addr+Int64(wo+j)); var lo=Int32(p&UInt8(0x0F)); var hi=Int32(p>>4)
                if lo>7:lo-=16; if hi>7:hi-=16
                ai+=lo*Int32(x[blk*32+j*2])+hi*Int32(x[blk*32+j*2+1])
            a+=Float32(ai)*sf
        dst.append(a)

fn add(a:List[Float32], b:List[Float32], n:Int)->List[Float32]:
    var o=List[Float32](); for i in range(n): o.append(a[i]+b[i]); return o^
fn mul(a:List[Float32], b:List[Float32], n:Int)->List[Float32]:
    var o=List[Float32](); for i in range(n): o.append(a[i]*b[i]); return o^
fn scale(x:List[Float32], s:Float32, n:Int)->List[Float32]:
    var o=List[Float32](); for i in range(n): o.append(x[i]*s); return o^
fn load_list(ptr:Int64, n:Int)->List[Float32]:
    var o=List[Float32](); for i in range(n): o.append(_f32(ptr+Int64(i*4))); return o^
fn rms(x:List[Float32], w:List[Float32], n:Int, e:Float32)->List[Float32]:
    var ss:Float32=0.0; for i in range(n): ss+=x[i]*x[i]
    var r=Float32(Float64(Float64(ss)/Float64(n)+Float64(e))**0.5)
    var o=List[Float32](); for i in range(n): o.append(x[i]/r*w[i]); return o^
fn softmax(x:List[Float32])->List[Float32]:
    var n=len(x); var mx=x[0]
    for i in range(1,n): if x[i]>mx:mx=x[i]
    var s:Float32=0.0; var e=List[Float32]()
    for i in range(n): var ev=Float32(Float64(2.718281828459045)**Float64(x[i]-mx)); e.append(ev); s+=ev
    var iv=1.0/s; var o=List[Float32]()
    for i in range(n): o.append(e[i]*iv); return o^

fn main():
    print("MojoLlama Full Engine —", %d, "layers")
    var N=%d; var NH=%d; var NKH=%d; var HD=%d
    var eps:Float32=1e-6; var S=N
    if NH*HD>S:S=NH*HD; if NKH*HD>S:S=NKH*HD
    if S<8192:S=8192

    # Load embedding
    var eb=_ld("/tmp/mojo_weights/gpt-oss/emb.bin")
    if eb==0: print("FAIL"); return
    var x=List[Float32]()
    for i in range(N): x.append(_f32(eb+Int64(i*4)))
    print("emb:", x[0], x[1], x[2])

    var on=_ld("/tmp/mojo_weights/gpt-oss/out_norm.bin")

''' % (L, L, N, NH, NKH, HD)

# Generate per-layer weight loading code
layer_code = ""
for layer in range(L):
    base = "/tmp/mojo_weights/gpt-oss/layer_%d" % layer
    # Use Python to hardcode all paths as literals
    layer_code += '''
    # Layer %(l)d weights from %(base)s
    var an_%(l)d = _ld("%(base)s/attn_norm.bin")
    var fn_%(l)d = _ld("%(base)s/ffn_norm.bin")
    var qi_%(l)d = _ld("%(base)s/q_info.bin")
    var qw_%(l)d = _ld("%(base)s/q_weight.bin")
    var ki_%(l)d = _ld("%(base)s/k_info.bin")
    var kw_%(l)d = _ld("%(base)s/k_weight.bin")
    var vi_%(l)d = _ld("%(base)s/v_info.bin")
    var vw_%(l)d = _ld("%(base)s/v_weight.bin")
    var oi_%(l)d = _ld("%(base)s/o_info.bin")
    var ow_%(l)d = _ld("%(base)s/o_weight.bin")
    var rt_%(l)d = _ld("%(base)s/router.bin")
    var gw_%(l)d = _ld("%(base)s/gate_exps.bin")
    var uw_%(l)d = _ld("%(base)s/up_exps.bin")
    var dw_%(l)d = _ld("%(base)s/down_exps.bin")
    var mi_%(l)d = _ld("%(base)s/moe_info.bin")
''' % {'l': layer, 'base': base}

# Forward loop
fwd_loop = '''
    # ═══════ FORWARD PASS ═══════
    for layer in range(%d):
        # Load xn for this layer's attention norm
        var xn = List[Float32]()
''' % L

# This approach is getting unwieldy. Let me instead generate individual
# functions per layer, so the compiler only needs to compile small units.

template += layer_code
template += fwd_loop
template += '''
    print("Done!")
'''

with open(os.path.join(os.path.dirname(__file__), 'mojollama_engine.mojo'), 'w') as f:
    f.write(template)
print(f"Generated {len(template)} bytes")
