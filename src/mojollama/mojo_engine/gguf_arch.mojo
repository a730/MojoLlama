# MojoLlama GGUF Arch Scanner — Extract model architecture from GGUF headers
# WHAT:  Reads GGUF file headers (<2GB), extracts integer metadata
#        (block_count, embedding_length, head_count, etc.)
# WHY:   Get comptime constants for mojollama_engine_bench.mojo
# WHEN:  May 2026.
from std import time
comptime O_RDONLY:Int=0; comptime SEEK_END:Int=2; comptime SEEK_SET:Int=0

@extern("malloc")
def _alc(sz:Int64)abi("C")->Int64:...
@extern("open")
def _open(p:UnsafePointer[UInt8, MutExternalOrigin],f:Int)abi("C")->Int:...
@extern("read")
def _read(fd:Int,b:UnsafePointer[UInt8, MutExternalOrigin],c:Int64)abi("C")->Int64:...
@extern("lseek")
def _lseek(fd:Int,o:Int64,w:Int)abi("C")->Int64:...
@extern("close")
def _close(fd:Int)abi("C")->Int:...

def str_to_cptr(s:String)->UnsafePointer[UInt8, MutExternalOrigin]:
    var blen=s.byte_length();var buf=alloc[UInt8](blen+1)
    var src=s.unsafe_ptr()
    for i in range(blen):buf.store(i,src.load(i))
    buf.store(blen,UInt8(0));return buf

def rd4(d:UnsafePointer[UInt8, MutExternalOrigin],off:Int)->UInt32:
    return UInt32(d.load(off))|UInt32(d.load(off+1))<<8|UInt32(d.load(off+2))<<16|UInt32(d.load(off+3))<<24

def rd8(d:UnsafePointer[UInt8, MutExternalOrigin],off:Int)->Int64:
    var v:Int64=0
    for i in range(8):v|=Int64(d.load(off+i))<<(i*8)
    return v

def scan(path:String):
    var cpath=str_to_cptr(path)
    var fd=_open(cpath,O_RDONLY);cpath.free()
    if fd<0:print("ERR");return
    var sz=_lseek(fd,0,SEEK_END);_lseek(fd,0,SEEK_SET)
    var hdr_sz=min(sz,Int64(524288))  # 64KB header
    var buf=_alc(hdr_sz)
    _read(fd,UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)),hdr_sz)
    _close(fd)
    var d=UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf))
    var magic=rd4(d,0)
    if magic!=1179993927:print("NOT GGUF: "+path);return
    var n_tensors=Int(rd8(d,8));var n_kv=Int(rd8(d,16))
    var mb=Float64(sz)/1048576.0
    var NE=0;var NH=0;var NK=0;var HD=0;var NL=0;var FF=0;var NV=0
    var p:Int=24;var arch=""
    
    # Read each KV pair
    for _ in range(n_kv):
        var klen=Int(rd4(d,p));p+=4
        # Skip key (we can't match strings in Mojo 1.0.0b1)
        p+=klen
        var vtype=rd4(d,p);p+=4
        if vtype==8:p+=8+Int(rd8(d,p))  # string - skip
        elif vtype==12:
            p+=4;var alen=Int(rd8(d,p));p+=8
            for _ in range(alen):p+=4
        elif vtype<=11:
            var vsz=1 if vtype<=1 else 2 if vtype<=3 else 4 if vtype<=6 else 8
            # We can read the value but can't match the key
            # Instead, count each integer KV and infer arch from position
            p+=vsz
    
    # We can't match keys, but we know the GGUF spec order for common models
    # Print file info for manual interpretation
    print(path)
    print("  "+String(Int(mb))+" MB, "+String(n_tensors)+" tensors, "+String(n_kv)+" KV pairs")
    
    # Scan again but this time track known key byte sequences
    # We match on the first few bytes of each key name
    p=24;var kv_idx=0
    p=24
    for ki in range(n_kv):
        if p>=524000: break  # prevent buffer overflow
        var klen=Int(rd4(d,p));p+=4
        var k0=0;var k1=0;var k2=0;var k3=0
        if klen>0:k0=Int(d.load(p))
        if klen>1:k1=Int(d.load(p+1))
        if klen>2:k2=Int(d.load(p+2))
        if klen>3:k3=Int(d.load(p+3))
        p+=klen
        var vtype=rd4(d,p);p+=4
        if vtype<=11:
            var vsz=1 if vtype<=1 else 2 if vtype<=3 else 4 if vtype<=6 else 8
            var val:Int=0
            for ci in range(vsz):val|=Int(d.load(p+ci))<<(ci*8)
            p+=vsz
            kv_idx+=1
            # Known GGUF key start bytes:
            # block_count: "bloc" (98,108,111,99)
            # embedding_length: "embe" (101,109,98,101)
            # feed_forward_length: "feed" (102,101,101,100)
            # head_count: "head" (104,101,97,100)
            # head_count_kv: "head" (104,101,97,100) — appears after head_count
            # key_length: "key_" (107,101,121,95)
            # vocab_size: "voca" (118,111,99,97)
            if k0==98 and k1==108 and k2==111 and k3==99:NL=val    # block
            if k0==101 and k1==109 and k2==98 and k3==101:NE=val   # embe
            if k0==102 and k1==101 and k2==101 and k3==100:FF=val  # feed
            if k0==104 and k1==101 and k2==97 and k3==100:
                if NH==0:NH=val  # head_count (first "head" match)
                else:NK=val       # head_count_kv (second match)
            if k0==107 and k1==101 and k2==121 and k3==95:HD=val   # key_
            if k0==118 and k1==111 and k2==99 and k3==97:NV=val    # voca
    
    if HD==0:HD=NE//NH if NH>0 else 0
    print("  NE="+String(NE)+" NH="+String(NH)+" NK="+String(NK))
    print("  HD="+String(HD)+" NL="+String(NL)+" FF="+String(FF)+" NV="+String(NV))

def main():
    scan("/tmp/models/gpt-oss-20b-Q4_K_M.gguf")
    scan("/tmp/models/ZAYA1-8B-Q4_K_M.gguf")
    scan("/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf")
