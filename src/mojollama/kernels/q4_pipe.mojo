"""MojoLlama Q4_0 matmul worker — reads weights via mmap, input via stdin.
Part of the parallel inference pipeline.

Protocol:
  CLI args: <weight_path> <n_rows> <n_cols> <start_row> <end_row>
  stdin (raw binary): input vector (n_cols × float32 = n_cols * 4 bytes)
  stdout (raw binary): output vector ((end_row-start_row) × float32)

Weight file format: raw Q4_0 blocks (n_rows * (n_cols/32) * 18 bytes uint8)
Each block: 2 bytes f16 scale, 16 bytes nibbles
Nibble order (GGUF v2/v3): positions 0-15 = low nibbles, 16-31 = high nibbles
"""

from std.memory.unsafe_pointer import alloc, free
from python import Python

alias F32x8 = SIMD[DType.float32, 8]
alias U8x16 = SIMD[DType.uint8, 16]


def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var s = (UInt32(h) >> 15) & 1
        var e = (UInt32(h) >> 10) & 0x1f
        var m = UInt32(h) & 0x3ff
        if e == 0:
            if m == 0: return Float32(bitcast[DType.float32](s << 31))
            var mm = m; var c: UInt32 = 0
            while mm > 0: mm >>= 1; c += 1
            var sh = 24 - c
            return Float32(bitcast[DType.float32]((s << 31) | ((UInt32(113 - sh)) << 23) | ((m << (sh + 13)) & 0x7fffff)))
        if e == 31: return Float32(bitcast[DType.float32]((s << 31) | 0x7f800000 | (m << 13)))
        return Float32(bitcast[DType.float32]((s << 31) | ((e + 112) << 23) | (m << 13))))


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


def q4_matmul_chunk(
    w_buf: UnsafePointer[mut=True, type=UInt8, origin=_],
    inp_buf: UnsafePointer[mut=False, type=Float32, origin=_],
    out_buf: UnsafePointer[mut=True, type=Float32, origin=_],
    n_cols: Int, start_row: Int, end_row: Int,
):
    """Compute Q4_0 matmul for rows [start_row, end_row).
    Uses correct GGUF v2/v3 nibble order: all lows first, then all highs.
    """
    var bpr = n_cols // 32
    for row in range(start_row, end_row):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = w_buf.load(off)
            var hi = w_buf.load(off + 1)
            var scale = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nibs = w_buf.load[width=16](off + 2)
            var inp_off = blk * 32
            # Load 32 input values
            var x0 = inp_buf.load[width=8](inp_off)
            var x1 = inp_buf.load[width=8](inp_off + 8)
            var x2 = inp_buf.load[width=8](inp_off + 16)
            var x3 = inp_buf.load[width=8](inp_off + 24)
            total += q4_block_dot(scale, nibs, x0, x1, x2, x3)
        out_buf.store(row - start_row, total)


def main() raises:
    var sys = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    var np = Python.import_module("numpy")
    var mmap_mod = Python.import_module("mmap")
    var bltns = Python.import_module("builtins")
    var argv = sys.argv
    var argc = Int(bltns.len(argv))

    if argc < 6:
        print("Usage: mojollama_q4_pipe <weight_path> <n_rows> <n_cols> <start_row> <end_row>", file=sys.stderr)
        sys.exit(1)

    var weight_path = String(argv[1])
    var n_rows = Int(String(argv[2]))
    var n_cols = Int(String(argv[3]))
    var start_row = Int(String(argv[4]))
    var end_row = Int(String(argv[5]))
    var n_chunk_rows = end_row - start_row

    var bpr = n_cols // 32
    var w_bytes = n_rows * bpr * 18

    # Open weight file and mmap
    var fd = os_mod.open(weight_path, 0)  # O_RDONLY
    var w_mmap = mmap_mod.mmap(fd, w_bytes, prot=1, flags=1, offset=0)  # PROT_READ, MAP_SHARED

    # Read input from stdin (n_cols float32 values)
    var inp_raw = sys.stdin.buffer.read(n_cols * 4)

    # Allocate Mojo buffers
    var w_buf = alloc[UInt8](w_bytes)
    var inp_buf = alloc[Float32](n_cols)
    var out_buf = alloc[Float32](n_chunk_rows)

    # Copy from mmap to Mojo heap (byte by byte — slow but correct)
    # TODO: use memoryview for bulk copy when Mojo supports it
    for i in range(w_bytes):
        w_buf.store(i, UInt8(w_mmap[i]))

    # Copy input from numpy to Mojo
    var inp_np = np.frombuffer(inp_raw, dtype=np.float32)
    for i in range(n_cols):
        inp_buf.store(i, Float32(inp_np[i]))

    # Compute
    q4_matmul_chunk(w_buf, inp_buf, out_buf, n_cols, start_row, end_row)

    # Write output to stdout
    var stdout = sys.stdout.buffer
    # Build numpy array from Mojo result
    var out_list = Python.evaluate("[]")
    for i in range(n_chunk_rows):
        Python.evaluate("out_list.append").__call__(out_buf.load(i))
    var out_np = np.array(out_list, dtype=np.float32)
    stdout.write(out_np.tobytes())
    stdout.flush()

    # Cleanup
    w_buf.free()
    inp_buf.free()
    out_buf.free()
    w_mmap.close()
    os_mod.close(fd)
