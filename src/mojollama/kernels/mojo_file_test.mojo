from std.prelude import *
from std import time

@extern("fopen")
fn c_fopen(path: String, mode: String) -> Int64: ...

@extern("fread")
fn c_fread(buf: Int64, size: Int64, count: Int64, fp: Int64) -> Int64: ...

@extern("fclose")
fn c_fclose(fp: Int64) -> Int32: ...

@extern("fseek")
fn c_fseek(fp: Int64, offset: Int64, whence: Int32) -> Int32: ...

@extern("ftell")
fn c_ftell(fp: Int64) -> Int64: ...

@extern("malloc")
fn c_malloc(size: Int64) -> Int64: ...

@extern("free")
fn c_free(ptr: Int64): ...

@extern("memcpy")
fn c_memcpy(dst: Int64, src: Int64, n: Int64): ...

fn main():
    var path = "/tmp/mojo_weights/gpt-oss/meta.bin"
    var fp = c_fopen(path, "rb")
    if fp == 0:
        print("Failed to open: ", path)
        return
    
    c_fseek(fp, 0, 2)
    var file_size = c_ftell(fp)
    c_fseek(fp, 0, 0)
    
    var buf = c_malloc(file_size)
    var nread = c_fread(buf, 1, file_size, fp)
    c_fclose(fp)
    
    print("Read ", nread, " bytes")
    
    # Read first 100 bytes and print as string
    # Use a local buffer via List
    var content = List[UInt8]()
    for i in range(Int(min(100, nread))):
        content.append(UInt8((buf + i).load[UInt8]()))  # Still doesn't work
    
    print("Content length: ", len(content))
    c_free(buf)
    
    print("File I/O test complete")
