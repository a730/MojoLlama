from std.prelude import *

@extern("dlopen")
fn dlopen(path: String, flags: Int32) -> Int64: ...

@extern("dlerror")
fn dlerror() -> String: ...

fn main():
    var handle = dlopen("/onedev-workspace/work/src/mojollama/mojo_engine/libbridge.so", 2)
    print("dlopen handle: ", handle)
    if handle == 0:
        var err = dlerror()
        print("Error: ", err)
