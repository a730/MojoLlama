from std.prelude import *

# Try linking with custom library
@extern("bridge_load", "libbridge.so")
fn bridge_load(path: String) -> Int64: ...

fn main():
    print("test")
