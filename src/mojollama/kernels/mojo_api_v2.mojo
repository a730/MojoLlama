# Mojo API exploration
from std.prelude import *

fn main():
    # Type conversions
    var ui16: UInt16 = 65000
    print("UInt16: ", ui16)
    
    # Convert UInt16 to UInt32
    var ui32: UInt32 = ui16  # implicit?
    # Explicit
    var ui32b: UInt32 = UInt32(ui16)
    print("UInt32: ", ui32b)
    
    # SIMD
    var v = SIMD[DType.uint8, 32](0)
    print("SIMD len: ", v.size)
    
    # Pointer
    var arr = InlineArray[UInt8, 10]()
    for i in range(10):
        arr[i] = UInt8(i * 2)
    print("InlineArray[0]: ", arr[0])
    print("InlineArray[9]: ", arr[9])
    
    # Try DTypePointer
    var dp = DTypePointer[DType.uint8].alloc(10)
    for i in range(10):
        dp.store(i, UInt8(i * 3))
    for i in range(10):
        print("dp[", i, "] = ", dp.load(i), end=" ")
    print()
    dp.free()
    
    # Benchmark timing
    var t0 = time.perf_counter()
    var s: Float64 = 0.0
    for i in range(10000000):
        s += 0.00000001
    var t1 = time.perf_counter()
    print("Time: ", (t1 - t0) * 1000, "ms")
