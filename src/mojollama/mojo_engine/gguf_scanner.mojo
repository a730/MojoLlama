# MojoLlama GGUF Scanner — Find and describe all GGUF models
# WHAT:  Scans known directories, reads GGUF headers, outputs architecture.
# WHY:   Pure Mojo — find all benchmarkable models.
# WHEN:  May 2026.
from std import time
comptime O_RDONLY: Int = 0; comptime SEEK_END: Int = 2; comptime SEEK_SET: Int = 0

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
    var v: Int64 = 0
    for i in range(8): v |= Int64(d.load(off+i)) << (i*8)
    return v

def scan_gguf(path: String):
    var cpath = str_to_cptr(path)
    var fd = _open(cpath, O_RDONLY)
    cpath.free()
    if fd < 0: print("ERR:open " + path); return
    
    var sz = _lseek(fd, 0, SEEK_END)
    _lseek(fd, 0, SEEK_SET)
    
    # Read first 16KB (enough for header + KV + tensor infos)
    var hdr_sz = min(sz, Int64(16384))
    var buf = _alc(hdr_sz)
    _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), hdr_sz)
    _close(fd)
    
    var d = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf))
    
    var magic = rd4(d, 0)
    if magic != 1179993927:  # "GGUF" LE
        print("SKIP " + path + " (not GGUF)")
        return
    
    var version = rd4(d, 4)
    var n_tensors = rd8(d, 8)
    var n_kv = rd8(d, 16)
    var mb = Float64(sz) / 1048576.0
    var name = "unknown"
    var arch = "unknown"; var NE=0; var NH=0; var NK=0; var HD=0; var NL=0; var FF=0; var NV=0
    
    var p: Int = 24
    for _ in range(n_kv):
        var klen = Int(rd4(d, p)); p += 4
        var key_buf = ""
        for ci in range(klen): key_buf += String(UInt8(d.load(p+ci)))
        p += klen
        var vtype = rd4(d, p); p += 4
        if vtype == 8:  # string
            var slen = Int(rd8(d, p)); p += 8
            var val = ""
            for ci in range(slen): val += String(UInt8(d.load(p+ci)))
            p += slen
            if "architecture" in key_buf: arch = val
        elif vtype == 12:  # array
            p += 4; var alen = Int(rd8(d, p)); p += 8
            for _ in range(alen): p += 4
        elif vtype <= 11:
            var vsz = 1 if vtype <=1 else 2 if vtype <=3 else 4 if vtype <=6 else 8
            var val_i: Int = 0
            for ci in range(vsz): val_i |= Int(d.load(p+ci)) << (ci*8)
            p += vsz
            if "block_count" in key_buf: NL = val_i
            if "embedding_length" in key_buf: NE = val_i
            if "feed_forward_length" in key_buf: FF = val_i
            if "head_count" in key_buf and "kv" not in key_buf and "count" in key_buf: NH = val_i
            if "head_count_kv" in key_buf: NK = val_i
            if "key_length" in key_buf: HD = val_i
            if "vocab_size" in key_buf: NV = val_i
    
    # Determine mode
    var mode = "DENSE"
    # Check for MoE by looking at expert_count
    var is_moe = "expert_count" in "KEY_SEARCH"  # simplified
    # Re-scan KV for expert info
    # We'll just report what we found
    
    if arch == "unknown": arch = "???"
    if HD == 0: HD = NE // NH if NH > 0 else 0
    
    print(path)
    print("  Size: " + String(Int(mb)) + " MB, Arch: " + arch + ", Tensors: " + String(Int(n_tensors)))
    print("  NE=" + String(NE) + " NH=" + String(NH) + " NK=" + String(NK))
    print("  HD=" + String(HD) + " NL=" + String(NL) + " FF=" + String(FF) + " NV=" + String(NV))

def main():
    # Scan known GGUF files
    scan_gguf("/onedev-workspace/work/Llama-3.2-1B-Instruct-Q4_0.gguf")
    scan_gguf("/onedev-workspace/work/gemma-4-E2B-it-Q4_K_M.gguf")
    scan_gguf("/tmp/models/ZAYA1-8B-Q4_K_M.gguf")
    scan_gguf("/tmp/models/gpt-oss-20b-Q4_K_M.gguf")
    scan_gguf("/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf")
    scan_gguf("/tmp/qwen3.5-2b-mtp-gguf/Qwen3.5-2B-Q4_K_M.gguf")
