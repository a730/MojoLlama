"""MojoLlama — parallel Q4_0 matmul via file-based IPC.

Compile: mojo build parallel_matmul.mojo -o mojollama_q4

Each invocation reads weights/input from binary files, computes
a chunk of rows, and writes results to an output file. Python
multiprocessing fans out across CPU cores — one process per chunk.

File formats (raw binary, no numpy dependency):
  weights: [uint8] — n_blocks * 18 bytes  (Q4_0 blocks)
  input:   [float32] — n_cols values
  output:  [float32] — n_rows values (pre-allocated, each worker writes subset)

Usage:
  mojollama_q4 <weights.bin> <input.bin> <output.bin> <n_rows> <n_cols> <start_row> <end_row>
"""

from std.memory.unsafe_pointer import alloc, free, pointer_to_int
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
            if m == 0:
                return Float32(bitcast[DType.float32](s << 31))
            var mm = m
            var c: UInt32 = 0
            while mm > 0:
                mm >>= 1
                c += 1
            var sh = 24 - c
            return Float32(
                bitcast[DType.float32](
                    (s << 31) | ((UInt32(113 - sh)) << 23) | ((m << (sh + 13)) & 0x7fffff)
                )
            )
        if e == 31:
            return Float32(
                bitcast[DType.float32]((s << 31) | 0x7f800000 | (m << 13))
            )
        return Float32(
            bitcast[DType.float32]((s << 31) | ((e + 112) << 23) | (m << 13))
        )


def q4_block_dot(
    scale: Float32, nibbles: U8x16, x0: F32x8, x1: F32x8, x2: F32x8, x3: F32x8
) -> Float32:
    @parameter
    def dg(s: Int, xv: F32x8) -> Float32:
        var v = F32x8()
        for j in range(8):
            var bi = s + j // 2
            var b = nibbles[bi]
            v[j] = (
                Float32(Int8(b & 15) - 8)
                if j % 2 == 0
                else Float32(Int8((b >> 4) & 15) - 8)
            )
        return (v * scale * xv).reduce_add()
    return dg(0, x0) + dg(4, x1) + dg(8, x2) + dg(12, x3)


def q4_matmul_file(
    w_path: String,
    inp_path: String,
    out_path: String,
    n_rows: Int,
    n_cols: Int,
    start_row: Int,
    end_row: Int,
) raises:
    """Read weights/input from binary files, compute chunk, write results."""
    var posix = Python.import_module("os")

    # Calculate sizes
    var bpr = n_cols // 32                           # blocks per row
    var bsize = 18                                    # bytes per block (2 scale + 16 nibbles)

    var w_size = n_rows * bpr * bsize
    var inp_size = n_cols * 4                         # float32
    var out_size = n_rows * 4

    # Open files
    var w_fd = posix.open(w_path, 0)                  # O_RDONLY
    var inp_fd = posix.open(inp_path, 0)
    var out_fd = posix.open(out_path, 2)              # O_RDWR

    # mmap the files (shared memory across processes)
    var mmap_mod = Python.import_module("mmap")
    var prot_read = 1                                  # PROT_READ
    var prot_write = 2                                 # PROT_WRITE
    var mmap_shared = 1                                # MAP_SHARED

    var w_mmap = mmap_mod.mmap(w_fd, w_size, prot=prot_read, flags=mmap_shared, offset=0)
    var inp_mmap = mmap_mod.mmap(inp_fd, inp_size, prot=prot_read, flags=mmap_shared, offset=0)
    var out_mmap = mmap_mod.mmap(out_fd, out_size, prot=prot_read | prot_write, flags=mmap_shared, offset=0)

    # Get buffer pointers from mmap
    var buf_mod = Python.import_module("builtins")
    var w_view = buf_mod.memoryview(w_mmap).cast("B")
    var inp_view = buf_mod.memoryview(inp_mmap).cast("f")
    var out_view = buf_mod.memoryview(out_mmap).cast("f")

    # Read data into Mojo-managed buffers
    var w_buf = alloc[UInt8](w_size)
    var inp_buf = alloc[Float32](n_cols)
    var out_buf = alloc[Float32](n_rows)

    # Copy from Python memory views to Mojo buffers
    for i in range(n_cols):
        inp_buf.store(i, Float32(inp_view[i]))

    # For weights, copy byte-by-byte via Python
    for i in range(w_size):
        w_buf.store(i, UInt8(w_view[i]))

    # Compute
    for row in range(start_row, end_row):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * bsize
            var lo = w_buf.load(off)
            var hi = w_buf.load(off + 1)
            var scale = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nibs = w_buf.load[width=16](off + 2)
            var inp_off = blk * 32
            var x0 = inp_buf.load[width=8](inp_off)
            var x1 = inp_buf.load[width=8](inp_off + 8)
            var x2 = inp_buf.load[width=8](inp_off + 16)
            var x3 = inp_buf.load[width=8](inp_off + 24)
            total += q4_block_dot(scale, nibs, x0, x1, x2, x3)
        out_buf.store(row, total)

    # Write results back to mmap
    for row in range(start_row, end_row):
        out_view[row] = out_buf.load(row)

    # Cleanup
    w_buf.free()
    inp_buf.free()
    out_buf.free()
    w_mmap.close()
    inp_mmap.close()
    out_mmap.close()
    posix.close(w_fd)
    posix.close(inp_fd)
    posix.close(out_fd)


def main() raises:
    var sys = Python.import_module("sys")
    var bltns = Python.import_module("builtins")
    var argv = sys.argv
    var argc = Int(bltns.len(argv))

    if argc < 8:
        print("Usage: mojollama_q4 <weights.bin> <input.bin> <output.bin> <n_rows> <n_cols> <start_row> <end_row>")
        sys.exit(1)

    var w_path = String(argv[1])
    var inp_path = String(argv[2])
    var out_path = String(argv[3])
    var n_rows = Int(String(argv[4]))
    var n_cols = Int(String(argv[5]))
    var start_row = Int(String(argv[6]))
    var end_row = Int(String(argv[7]))

    q4_matmul_file(w_path, inp_path, out_path, n_rows, n_cols, start_row, end_row)
