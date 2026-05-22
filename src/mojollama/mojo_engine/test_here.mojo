from std import time

def test_fn(out: Span[Float32, _], w: Span[UInt8, _], n: Int):
    for i in range(n): out[i] = Float32(w[i])

def main():
    var a = alloc[Float32](4)
    var b = alloc[UInt8](4)
    for i in range(4): b.store(i, UInt8(i * 3))
    var sa = Span[Float32, _](ptr=a, length=4)
    var sb = Span[UInt8, _](ptr=b, length=4)
    test_fn(sa, sb, 4)
    for i in range(4): print(i, sa[i])
    a.free(); b.free()
