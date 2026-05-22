# Test: load weights via C bridge
from std import time

@extern("load_weight")
def _load_w(idx: Int64) abi("C") -> Int64: ...
@extern("free_weight")
def _free_w(p: Int64) abi("C") -> None: ...
@extern("weight_size")
def _w_size(idx: Int64) abi("C") -> Int64: ...
@extern("read_u8")
def _u8(a: Int64) abi("C") -> UInt8: ...
@extern("read_f32")
def _f32(a: Int64) abi("C") -> Float32: ...

def main():
    print("Testing C weight loader...")
    
    var emb = _load_w(Int64(0))
    var emb_sz = _w_size(Int64(0))
    if emb == 0: print("FAIL"); return
    print("emb: ptr=", emb, "size=", emb_sz, "bytes")
    
    var meta = _load_w(Int64(1))
    var meta_sz = _w_size(Int64(1))
    print("meta: ptr=", meta, "size=", meta_sz, "bytes")
    
    print("meta bytes:")
    for i in range(Int64(meta_sz)):
        var c = _u8(meta + Int64(i))
        print(" ", c, end="")
    print()
    _free_w(meta)
    
    # Layer 0 attn_norm: index 3
    var norm = _load_w(Int64(3))
    var norm_sz = _w_size(Int64(3))
    print("\nattn_norm: size=", norm_sz, "first 4 f32:")
    for i in range(4):
        print("  [", i, "]:", _f32(norm + Int64(i * 4)))
    _free_w(norm)
    
    # q_weight (layer_0 files start at index 3)
    # 0=attn_norm, 1=down_e, 2=down_info, 3=ffn_norm, 4=gate_e, 5=gate_info
    # 6=k_info, 7=k_w, 8=o_info, 9=o_w, 10=q_info, 11=q_w
    var qw = _load_w(Int64(3 + 11))
    var qw_sz = _w_size(Int64(3 + 11))
    print("\nq_weight: size=", qw_sz, "expected=", 4096 * 2880 // 32 * 22)
    _free_w(qw)
    
    # gate_exps (index 3 + 4 = 7)
    var gw = _load_w(Int64(3 + 4))
    var gw_sz = _w_size(Int64(3 + 4))
    var per_exp = 2880 * 2880 // 32 * 17
    print("gate_exps: size=", gw_sz, "per_exp=", per_exp, "experts=", gw_sz / per_exp)
    _free_w(gw)
    
    # router (index 3 + ... let me find it)
    # Let's just loop and find all files
    print("\nAll layer_0 files:")
    for i in range(17):
        var p = _load_w(Int64(3 + i))
        var sz = _w_size(Int64(3 + i))
        print("  [", i, "]: ptr=", p, "size=", sz)
        _free_w(p)
    
    _free_w(emb)
    print("\nDone!")
