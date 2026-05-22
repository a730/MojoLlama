from std.prelude import *

fn main() raises:
    # SIMD with DType enum
    var v = SIMD[DType.float32, 4](1.0)
    v = v + SIMD[DType.float32, 4](2.0)
    print("SIMD[0]: ", v[0])
    print("SIMD[3]: ", v[3])
    
    # Int to float
    var i: Int = 42
    var f = Float32(i)
    print("Int->Float: ", f)
