from std.prelude import *

fn main():
    print("Mojo works")
    var s: Float32 = 0.0
    for i in range(10):
        s += Float32(i)
    print("Sum: ", s)
