from std.prelude import *
from std import time

def load_list(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(Float32(i))
    return out^

def main():
    var x = load_list(10)
    print("test:", x[0], x[1])
    print("OK!")
