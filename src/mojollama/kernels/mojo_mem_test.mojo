# Explore Mojo memory APIs
from std.prelude import *
from std.memory import OpaquePointer, address_of

fn main():
    # Check OpaquePointer
    var x: Int = 42
    var addr = address_of(x)
    print("Address: ", addr.address)
    
    # Try UnsafePointer from List
    var lst = List[Int]()
    for i in range(10):
        lst.append(i)
    var p = lst.unsafe_ptr()
    print("List[0]: ", p.load(0))
    print("List[5]: ", p.load(5))
    
    # Try pointer arithmetic via offset
    var p5 = p.offset(5)
    print("List via offset[5]: ", p5.load(0))
