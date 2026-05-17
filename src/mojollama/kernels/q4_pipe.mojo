"""Pure Mojo Q4_0 matmul — stdin/stdout binary protocol.
No Python interop needed for the compute path.

Protocol (stdin, all raw binary):
  1. int32: num_rows  
  2. int32: num_cols
  3. Q4_0 weight data (num_rows * (num_cols/32) * 18 bytes)
  4. float32 input vector (num_cols * 4 bytes)

Protocol (stdout):
  1. float32 output vector (num_rows * 4 bytes)
"""

from std.memory.unsafe_pointer import alloc

alias U8x16 = SIMD[DType.uint8, 16]
alias F32x8 = SIMD[DType.float32, 8]


def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var sign = (UInt32(h) >> 15) & 1
        var exp = (UInt32(h) >> 10) & 0x1f
        var mant = UInt32(h) & 0x3ff
        if exp == 0:
            if mant == 0: return Float32(bitcast[DType.float32](sign << 31))
            var m = mant; var c: UInt32 = 0
            while m > 0: m >>= 1; c += 1
            var sh = 24 - c
            return Float32(bitcast[DType.float32]((sign << 31) | ((UInt32(113 - sh)) << 23) | ((mant << (sh + 13)) & 0x7fffff)))
        elif exp == 31:
            return Float32(bitcast[DType.float32]((sign << 31) | 0x7f800000 | (mant << 13)))
        else:
            return Float32(bitcast[DType.float32]((sign << 31) | ((exp + 112) << 23) | (mant << 13))))


def q4_block_dot(scale: Float32, nibbles: U8x16,
                 x0: F32x8, x1: F32x8, x2: F32x8, x3: F32x8) -> Float32:
    @parameter
    def dg(s: Int, xv: F32x8) -> Float32:
        var v = F32x8()
        for j in range(8):
            var bi = s + j // 2
            var b = nibbles[bi]
            v[j] = Float32(Int8(b & 15) - 8) if j % 2 == 0 else Float32(Int8((b >> 4) & 15) - 8)
        return (v * scale * xv).reduce_add()
    return dg(0, x0) + dg(4, x1) + dg(8, x2) + dg(12, x3)


def main():
    from python import Python
    var sys = Python.import_module("sys")
    var stdin = sys.stdin.buffer
    var stdout = sys.stdout.buffer
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    
    # Read header: 8 bytes (2 x int32)
    var hdr = stdin.read(8)
    if builtins.len(hdr) < 8:
        return
    
    var hdr_arr = np.frombuffer(hdr, dtype=np.int32)
    var n_rows = Int(hdr_arr[0])
    var n_cols = Int(hdr_arr[1])
    var bpr = n_cols // 32
    var n_blocks = n_rows * bpr
    
    # Read weights
    var w_bytes = n_blocks * 18
    var raw_w = stdin.read(w_bytes)
    var raw_inp = stdin.read(n_cols * 4)
    
    if builtins.len(raw_w) < w_bytes or builtins.len(raw_inp) < n_cols * 4:
        return
    
    # Copy to Mojo heap via helper that avoids per-element Python interop
    # Use numpy to convert bytes to the right type, then extract as Python
    # We do iterative byte copy which is OK for now (bounded by I/O time)
    var w = alloc[UInt8](w_bytes)
    for i in range(w_bytes):
        w.store(i, UInt8(raw_w[i]))
    
    var inp_arr = np.frombuffer(raw_inp, dtype=np.float32)
    var inp = alloc[Float32](n_cols)
    for i in range(n_cols):
        inp.store(i, Float32(inp_arr[i]))
    
    var out = alloc[Float32](n_rows)
    
    # Run matmul using pure Mojo SIMD
    for row in range(n_rows):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = w.load(off)
            var hi = w.load(off + 1)
            var scale = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nibs = w.load[width=16](off + 2)
            var inp_off = blk * 32
            var x0 = inp.load[width=8](inp_off)
            var x1 = inp.load[width=8](inp_off + 8)
            var x2 = inp.load[width=8](inp_off + 16)
            var x3 = inp.load[width=8](inp_off + 24)
            total += q4_block_dot(scale, nibs, x0, x1, x2, x3)
        out.store(row, total)
    
    # Write output — construct numpy array in Mojo
    var out_list = Python.evaluate("[]")
    for i in range(n_rows):
        var v = out.load(i)
        Python.evaluate("out_list.append", v)
    
    var out_arr = np.array(out_list, dtype=np.float32)
    stdout.write(out_arr.tobytes())
    stdout.flush()
    
    w.free()
    inp.free()
    out.free()
