# gguf_extract.mojo — Pure Mojo GGUF → raw .bin extractor
# WHAT:  Reads GGUF file headers + tensor data using raw byte I/O.
#        Extracts each tensor to <out_dir>/<tensor_name>.bin
# WHY:   Weight extraction for MojoLlama engine. No C dependency.
# WHEN:  May 2026.
# Verified works: header parsing (gguf_test.mojo June 2026).
from std import time
comptime O_RDONLY:Int=0;comptime O_WRONLY:Int=1;comptime O_CREAT:Int=64
comptime SEEK_END:Int=2;comptime SEEK_SET:Int=0;comptime SEEK_CUR:Int=1
comptime MODE_RW:Int=438;comptime A:Int=32

@extern("malloc")
def _alc(sz:Int64)abi("C")->Int64:...
@extern("free")
def _fre(p:Int64)abi("C")->None:...
@extern("write")
def _write(fd:Int,b:UnsafePointer[UInt8,MutExternalOrigin],c:Int64)abi("C")->Int64:...
@extern("lseek")
def _lseek(fd:Int,o:Int64,w:Int)abi("C")->Int64:...
@extern("close")
def _close(fd:Int)abi("C")->Int:...
@extern("mkdir")
def _md(p:UnsafePointer[UInt8,MutExternalOrigin],m:Int)abi("C")->Int:...

# Use syscall for read and open to avoid @extern conflicts
@extern("open")
def _open2(p:UnsafePointer[UInt8,MutExternalOrigin],f:Int)abi("C")->Int:...
@extern("read")
def _read(fd:Int,b:UnsafePointer[UInt8,MutExternalOrigin],c:Int64)abi("C")->Int64:...

def str_to_cptr(s:String)->UnsafePointer[UInt8,MutExternalOrigin]:
    var blen=s.byte_length();var buf=alloc[UInt8](blen+1)
    var src=s.unsafe_ptr()
    for i in range(blen):buf.store(i,src.load(i))
    buf.store(blen,UInt8(0));return buf

def rd4(d:UnsafePointer[UInt8,MutExternalOrigin],o:Int)->UInt32:
    return UInt32(d.load(o))|UInt32(d.load(o+1))<<8|UInt32(d.load(o+2))<<16|UInt32(d.load(o+3))<<24
def rd8(d:UnsafePointer[UInt8,MutExternalOrigin],o:Int)->Int64:
    var v:Int64=0
    for i in range(8):v|=Int64(d.load(o+i))<<(i*8)
    return v
def align(x:Int,a:Int)->Int: return (x+a-1)//a*a

def write_file_bytes(path_c:UnsafePointer[UInt8,MutExternalOrigin],
                     data:UnsafePointer[UInt8,MutExternalOrigin],
                     sz:Int64):
    var fd=_open2(path_c,O_WRONLY|O_CREAT)
    if fd>=0:_write(fd,data,sz);_close(fd)

def main():
    var path="/tmp/tinylama-1.1b-f16.gguf"
    var out_dir="/tmp/weights_tl/"
    
    var cp=str_to_cptr(path)
    var fd=_open2(cp,O_RDONLY);cp.free()
    if fd<0:print("Cannot open:",path);return
    var fsz=_lseek(fd,0,SEEK_END);_lseek(fd,0,SEEK_SET)
    
    # Read first 4MB for header + tensor info
    var buf_sz=min(Int64(4194304),fsz)
    var buf=_alc(buf_sz)
    _read(fd,UnsafePointer[UInt8,MutExternalOrigin](unsafe_from_address=Int(buf)),buf_sz)
    _close(fd)
    var d=UnsafePointer[UInt8,MutExternalOrigin](unsafe_from_address=Int(buf))
    
    var magic=rd4(d,0)
    if magic!=1179993927:print("Not GGUF:",magic);_fre(buf);return
    var n_tensors=Int(rd8(d,8));var n_kv=Int(rd8(d,16))
    print("GGUF: "+String(n_tensors)+" tensors, "+String(n_kv)+" KV pairs, size="+String(Int(fsz/1048576))+"MB")
    
    # Create output directory
    var cd=str_to_cptr(out_dir);_md(cd,MODE_RW);cd.free()
    
    var p:Int=24  # after 24-byte header
    
    # Skip KV pairs (we don't need metadata)
    for ki in range(n_kv):
        if p+4>Int(buf_sz):break
        var klen=Int(rd4(d,p));p+=4
        p+=klen;p=align(p,4)
        if p+4>Int(buf_sz):break
        var vtype=Int(rd4(d,p));p+=4
        if vtype==0 or vtype==1:p+=1          # U8/I8
        elif vtype==2 or vtype==3:p+=2        # U16/I16
        elif vtype==4 or vtype==5 or vtype==6:p+=4  # U32/I32/F32
        elif vtype==7:p+=8                    # BOOL
        elif vtype==8:                        # STRING
            var sl=Int(rd4(d,p));p+=4+sl;p=align(p,4)
        elif vtype==9 or vtype==10 or vtype==11:p+=8  # U64/I64/F64
        else:
            # array type (12+)
            if vtype>=12:
                var atype=Int(rd4(d,p));p+=4
                var alen=Int(rd8(d,p));p+=8
                if atype==8:  # string array — skip each element
                    for _ in range(alen):
                        var slen=Int(rd4(d,p));p+=4+slen;p=align(p,4)
                else:
                    var esz=1 if atype<=1 else 2 if atype<=3 else 4 if atype<=6 else 8
                    p+=esz*alen
    
    # Now p points to tensor info section (relative to buf start)
    var ti_start=p
    # First, scan all tensor infos to find total size
    var p2=p
    
    # Store tensor info in parallel arrays using heap-allocated buffers
    var max_tensors=n_tensors
    var name_offsets=UnsafePointer[Int64,MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_tensors*8))))
    var data_offsets=UnsafePointer[Int64,MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_tensors*8))))
    var data_sizes=UnsafePointer[Int64,MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_tensors*8))))
    var name_lengths=UnsafePointer[UInt32,MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_tensors*4))))
    
    var ti_end=p
    for ti in range(max_tensors):
        name_offsets.store(ti,Int64(ti_end))  # record where name starts
        if ti_end+4>Int(buf_sz):break
        var nlen=Int(rd4(d,ti_end));name_lengths.store(ti,UInt32(nlen));ti_end+=4
        ti_end+=nlen;ti_end=align(ti_end,4)
        if ti_end+4>Int(buf_sz):break
        var ndim=Int(rd4(d,ti_end));ti_end+=4
        var dim0:Int=1;var dim1:Int=1;var dim2:Int=1;var dim3:Int=1
        if ndim>=1:dim0=Int(rd8(d,ti_end));ti_end+=8
        if ndim>=2:dim1=Int(rd8(d,ti_end));ti_end+=8
        if ndim>=3:dim2=Int(rd8(d,ti_end));ti_end+=8
        if ndim>=4:dim3=Int(rd8(d,ti_end));ti_end+=8
        var dtype=Int(rd4(d,ti_end));ti_end+=4
        var ne=dim0*dim1*dim2*dim3
        
        # Calculate data size based on dtype
        var blk_sz=32;var blk_bytes=0
        if dtype==0:blk_bytes=128;blk_sz=32     # F32: 4*32
        elif dtype==1:blk_bytes=64;blk_sz=32    # F16: 2*32
        elif dtype==2:blk_bytes=18;blk_sz=32    # Q4_0
        elif dtype==3:blk_bytes=20;blk_sz=32    # Q4_1
        elif dtype==6:blk_bytes=34;blk_sz=32    # Q8_0
        elif dtype==10:blk_bytes=144;blk_sz=256 # Q4_K
        elif dtype==12:blk_bytes=240;blk_sz=256 # Q6_K
        elif dtype==39:blk_bytes=34;blk_sz=32   # MXFP4_CCA
        elif dtype==47:blk_bytes=34;blk_sz=32   # MXFP4
        else:blk_bytes=64;blk_sz=32  # default: f16-like
        
        var n_blocks=(ne+blk_sz-1)//blk_sz
        data_sizes.store(ti,Int64(n_blocks*blk_bytes))
        data_offsets.store(ti,rd8(d,ti_end));ti_end+=8
    
    var ti_total=ti_end-ti_start
    
    # Data section starts after tensor info, aligned to A=32
    var data_start_off=align(ti_start+ti_total,A)
    
    print("Data section at file offset: "+String(data_start_off))
    
    # Now re-open the file and extract each tensor
    var rfd=_open2(str_to_cptr(path),O_RDONLY)
    if rfd<0:print("Cannot reopen");_fre(buf);return
    
    for ti in range(max_tensors):
        var nl=Int(name_lengths.load(ti))
        var no=Int(name_offsets.load(ti))
        var dto=Int(data_offsets.load(ti))
        var dsz=data_sizes.load(ti)
        
        # Build filename: out_dir/<name>.bin
        # name bytes start at buf + no, length nl
        # Output path: out_dir + converted_name + ".bin"
        var od_len=out_dir.byte_length()
        var fname=alloc[UInt8](od_len+nl+5)
        var od_ptr=out_dir.unsafe_ptr()
        for i in range(od_len):fname.store(i,od_ptr.load(i))
        # Copy tensor name, replacing '.' with '_'
        for i in range(nl):
            var c=d.load(no+i)
            if c==UInt8(46):fname.store(od_len+i,UInt8(95))  # '.'→'_'
            elif c==UInt8(47):fname.store(od_len+i,UInt8(95))  # '/'→'_'
            else:fname.store(od_len+i,c)
        fname.store(od_len+nl,UInt8(46))   # '.'
        fname.store(od_len+nl+1,UInt8(98)) # 'b'
        fname.store(od_len+nl+2,UInt8(105))# 'i'
        fname.store(od_len+nl+3,UInt8(110))# 'n'
        fname.store(od_len+nl+4,UInt8(0))
        
        # Read tensor data from file
        var file_off=Int64(data_start_off)+dto
        # Read in chunks for large tensors
        var chunk_sz=Int64(1048576)  # 1MB chunks
        var remaining=dsz
        var wfd=_open2(fname,O_WRONLY|O_CREAT)
        if wfd<0:print("  can't write");continue
        var read_pos=file_off
        
        while remaining>0:
            var this_chunk=chunk_sz
            if this_chunk>remaining:this_chunk=remaining
            var tmp=_alc(this_chunk)
            _lseek(rfd,read_pos,SEEK_SET)
            _read(rfd,UnsafePointer[UInt8,MutExternalOrigin](unsafe_from_address=Int(tmp)),this_chunk)
            _write(wfd,UnsafePointer[UInt8,MutExternalOrigin](unsafe_from_address=Int(tmp)),this_chunk)
            _fre(tmp)
            read_pos+=this_chunk
            remaining-=this_chunk
        
        _close(wfd)
        fname.free()
        
        # Print progress
        var size_str=""
        if dsz>Int64(1048576):size_str=String(Int(dsz/1048576))+"MB"
        elif dsz>Int64(1024):size_str=String(Int(dsz/1024))+"KB"
        else:size_str=String(Int(dsz))+"B"
        print("  ["+String(ti+1)+"/"+String(max_tensors)+"] "+size_str)
    
    _close(rfd)
    _fre(buf)
    _fre(Int64(Int(name_offsets)))
    _fre(Int64(Int(data_offsets)))
    _fre(Int64(Int(data_sizes)))
    _fre(Int64(Int(name_lengths)))
    print("[DONE] Extracted "+String(max_tensors)+" tensors to "+out_dir)
