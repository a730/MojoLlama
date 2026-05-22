from std.prelude import *

fn main():
    # Try tensor types
    var t = Tensor[Float32](10)
    for i in range(10):
        t[i] = Float32(i * 2)
    print("Tensor[0]: ", t[0])
    print("Tensor[5]: ", t[5])
    print("Tensor shape: ", t.shape())
    
    # Check if SIMD pointer exists
    var v = SIMD[DType.float32, 4](3.14)
    print("SIMD: ", v[0])
    
    # Try C-ABI compatible pointer (needed for @extern)
    print("Done")
