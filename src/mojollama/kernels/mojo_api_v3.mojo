from std.prelude import *
from std import time

fn main():
    # Test time
    var t0 = time.perf_counter()
    var s: Float64 = 0.0
    for i in range(10000000):
        s += 0.00000001
    var t1 = time.perf_counter()
    print("Time: ", (t1 - t0) * 1000, "ms")
    
    # Test Pointer with origin
    var arr = InlineArray[UInt8, 100]()
    for i in range(100):
        arr[i] = UInt8(i & 0xFF)
    print("arr[42]: ", arr[42])
    
    # Get pointer from InlineArray
    var ptr = arr.unsafe_ptr()
    print("ptr[42]: ", ptr.load(42))
