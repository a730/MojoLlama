# TinyLlama Weight Loader — Pure Mojo Logic
# Uses minimal C I/O wrappers (Mojo String→C is broken).
# All indexing, memory management in Mojo.
from std.algorithm.backend.cpu.parallelize import parallelize

@extern("tl_fsize")
def _fsz(idx: Int) abi("C") -> Int64: ...
@extern("tl_fread")
def _fread(idx: Int, buf: UnsafePointer[UInt8, MutExternalOrigin], sz: Int64) abi("C") -> Int64: ...
@extern("malloc")
def _c_malloc(sz: Int64) abi("C") -> Int64: ...
@extern("free")
def _c_free(p: Int64) abi("C") -> None: ...
@extern("addr_to_f32")
def _af32(a: Int64) abi("C") -> UnsafePointer[Float32, MutExternalOrigin]: ...
@extern("addr_to_u8")
def _au8(a: Int64) abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...

comptime NL: Int = 22;  comptime N_BASE: Int = 3
comptime N_LAYER_FILES: Int = 9;  comptime N_FILES: Int = N_BASE + NL * N_LAYER_FILES

comptime W_EMB: Int = 0;  comptime W_OUT_NORM: Int = 1;  comptime W_OUT: Int = 2
comptime L_ATTN_NORM: Int = 0;  comptime L_FFN_NORM: Int = 1
comptime L_Q: Int = 2;  comptime L_K: Int = 3;  comptime L_V: Int = 4
comptime L_O: Int = 5;  comptime L_GATE: Int = 6;  comptime L_UP: Int = 7
comptime L_DOWN: Int = 8;  comptime L_STRIDE: Int = 9

var _wp: UnsafePointer[Int64, MutExternalOrigin] = UnsafePointer[Int64, MutExternalOrigin]()
var _n_loaded: Int = 0

def tl_load():
    _wp = alloc[Int64](N_FILES)
    for i in range(N_FILES): _wp.store(i, 0)
    for idx in range(N_FILES):
        var fsz = _fsz(idx)
        if fsz < 0: print("Error: can't stat file", idx); return
        if fsz == 0: continue
        var buf = _c_malloc(fsz)
        if buf == 0: print("Error: OOM for file", idx, "size", fsz); return
        var nread = _fread(idx, _au8(buf), fsz)
        if nread != fsz: print("Error: read mismatch", idx); _c_free(buf); return
        _wp.store(idx, buf)
    _n_loaded = N_FILES
    print("Loaded", _n_loaded, "files (", N_BASE, "+", NL, "×", N_LAYER_FILES, ")")

def tl_get(idx: Int) -> Int64:
    if _n_loaded == 0 or idx < 0 or idx >= _n_loaded: return 0
    return _wp.load(idx)

def tl_free():
    for idx in range(_n_loaded):
        var p = _wp.load(idx)
        if p != 0: _c_free(p)
    _wp.free(); _n_loaded = 0
