from std.prelude import *
from std.io import *

fn main() raises:
    var f = open("/tmp/mojo_weights/gpt-oss/meta.bin", "r")
    var content = f.read()
    print("Len: ", len(content))
    for i in range(min(100, len(content))):
        print(chr(content[i]), end="")
    print()
    f.close()
