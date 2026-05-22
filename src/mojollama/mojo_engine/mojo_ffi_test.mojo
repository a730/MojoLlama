from std.prelude import *

@extern("printf")
fn c_printf(str: String) -> Int32:
    ...

fn main():
    c_printf("Hello from Mojo C FFI!\n")
