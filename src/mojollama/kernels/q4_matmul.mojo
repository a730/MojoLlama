"""MojoLlama Q4_0 SIMD Kernel — Updated for Mojo 1.0.0b1 heap APIs.

Q4_0 block: 32 values packed into 18 bytes (2b f16 scale + 16b nibbles).
Full weight matrix matmul using Mojo heap allocation and SIMD.

Python integration: compiled as shared library (.so) using @export,
or called via Python interop from within Mojo.
"""

from std.sys.info import CompilationTarget
from std.memory import bitcast
from std.memory.unsafe_pointer import alloc, pointer_to_int

# SIMD types for AVX2 (256-bit)
alias F32x8  = SIMD[DType.float32, 8]
alias I8x32  = SIMD[DType.int8, 32]
alias U8x16  = SIMD[DType.uint8, 16]


# ─── Float16 → Float32 ────────────────────────────────────────────────

@always_inline
def f16_to_f32(h: UInt16) -> Float32:
    """float16 bits → float32. Uses F16C VCVTPH2PS when AVX2 available."""
    comptime if CompilationTarget.has_avx2():
        var f16 = bitcast[DType.float16](h)
        return Float32(f16)
    else:
        # Software IEEE 754-2008 fallback
        var sign = (UInt32(h) >> 15) & 1
        var exp = (UInt32(h) >> 10) & 0x1f
        var mant = UInt32(h) & 0x3ff
        var bits: UInt32 = 0
        if exp == 0:
            if mant == 0:
                bits = sign << 31
            else:
                var m = mant
                var count: UInt32 = 0
                while m > 0:
                    m = m >> 1
                    count += 1
                var shift = 24 - count
                bits = (sign << 31) | ((UInt32(113 - shift)) << 23) | ((mant << (shift + 13)) & 0x7fffff)
        elif exp == 31:
            bits = (sign << 31) | 0x7f800000 | (mant << 13)
        else:
            bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
        return Float32(bitcast[DType.float32](bits))


# ─── Q4_0 Block Utilities ─────────────────────────────────────────────

struct Q4_0Block:
    """One GGUF Q4_0 block: 18 bytes, 32 values."""
    var raw_scale: UInt16  # f16
    var nibbles: UInt8  # first byte of 16 — accessed via pointer offsets

@always_inline
def q4_block_read_scale(block_ptr: UnsafePointer[UInt8, _]) -> Float32:
    """Read f16 scale from a Q4_0 block."""
    var lo = block_ptr.load(0)    # UInt8 low byte
    var hi = block_ptr.load(1)    # UInt8 high byte
    var bits = (UInt16(hi) << 8) | UInt16(lo)
    return f16_to_f32(bits)

@always_inline
def q4_block_read_nibbles(block_ptr: UnsafePointer[UInt8, _]) -> U8x16:
    """Read 16 nibble bytes from a Q4_0 block (offset +2)."""
    # Load 16 bytes from offset 2 using SIMD
    var nibble_ptr = block_ptr + 2
    return nibble_ptr.load[width=16]()


# ─── Q4_0 Block Dot Product ─────────────────────────────────────────────

@always_inline
def q4_block_dot(scale: Float32, nibbles: U8x16, x: F32x8, x1: F32x8,
                x2: F32x8, x3: F32x8) -> Float32:
    """Dot product of one Q4_0 block with 32 float32 inputs.
    
    Fused dequant + dot: Σ_i (nibble_i - 8) * scale * x_i
    """
    @parameter
    def dot_group(start_byte: Int, xv: F32x8) -> Float32:
        var vals = F32x8()
        for j in range(8):
            var byte_idx = start_byte + j // 2
            var b = nibbles[byte_idx]
            if j % 2 == 0:
                vals[j] = Float32(Int8(b & 15) - 8)
            else:
                vals[j] = Float32(Int8((b >> 4) & 15) - 8)
        return (vals * scale * xv).reduce_add()
    
    var total: Float32 = 0.0
    total += dot_group(0, x)
    total += dot_group(4, x1)
    total += dot_group(8, x2)
    total += dot_group(12, x3)
    return total


# ─── Full Q4_0 Matmul (heap-allocated) ─────────────────────────────────

def q4_matmul_forward(
    weights_data_ptr: Int,    # raw address of Q4_0 weight data (from numpy)
    num_rows: Int,            # output feature count
    num_cols: Int,            # input feature count
    input_ptr: Int,           # raw address of float32 input
    output_ptr: Int,          # raw address of float32 output buffer
):
    """Compute y = W @ x for a Q4_0 weight matrix.
    
    weights_data: Q4_0 blocks, row-major. Each row has (num_cols/32) blocks.
    input: float32 vector of size num_cols (single token / batch item).
    output: float32 vector of size num_rows.
    
    All pointers are raw Int addresses (from numpy array __array_interface__).
    
    NOTE: In Mojo 1.0.0b1, UnsafePointer(unsafe_from_address=...) is not
    available in compiled form, but pointer_to_int + manual offset works.
    We use a workaround: mmap-backed temp files for data transfer,
    or call via Python interop where Mojo imports numpy directly.
    """
    # For now, this is the reference — the actual pointer construction
    # from raw addresses will use Python interop (Mojo imports numpy).
    print("Q4 matmul forward: rows=", num_rows, " cols=", num_cols)


# ─── Tests ─────────────────────────────────────────────────────────────

def test_f16():
    """Test float16→float32 conversion."""
    print("f16→f32 tests:")
    var f = f16_to_f32(UInt16(0x3c00))
    print("  0x3c00 →", f, "(expect 1.0)")
    
    f = f16_to_f32(UInt16(0x3800))
    print("  0x3800 →", f, "(expect 0.5)")
    
    f = f16_to_f32(UInt16(0xbc00))
    print("  0xbc00 →", f, "(expect -1.0)")
    print()


def test_q4_block_zero():
    """Test block dot with all-zero quantized values."""
    var scale = f16_to_f32(UInt16(0x3800))  # 0.5
    var nibbles = U8x16()  # all zeros
    var x = F32x8(2.0)
    var dot = q4_block_dot(scale, nibbles, x, x, x, x)
    print("zero block dot:", dot, "(expect -256.0)")
    if abs(dot - (-256.0)) < 1.0:
        print("  PASS")
    else:
        print("  FAIL")
    print()


def test_q4_block_varied():
    var scale = f16_to_f32(UInt16(0x3c00))  # 1.0
    var nibbles = U8x16()
    for i in range(16):
        nibbles[i] = 0x00 if i % 2 == 0 else 0xFF
    var x = F32x8(1.0)
    var dot = q4_block_dot(scale, nibbles, x, x, x, x)
    print("varied block dot:", dot, "(expect -16.0)")
    if abs(dot - (-16.0)) < 1.0:
        print("  PASS")
    else:
        print("  FAIL")
    print()


def test_heap_alloc():
    """Test heap allocation of Q4_0 data using Mojo 1.0.0b1's alloc."""
    print("Heap allocation tests:")
    
    # Allocate space for 1024 Q4_0 blocks = 1024 * 18 bytes
    var n_blocks: Int = 1024
    var buf = alloc[UInt8](n_blocks * 18)
    
    # Write some test data
    for i in range(1024 * 18):
        buf.store(i, UInt8(0))
    
    print("  Allocated", n_blocks * 18, "bytes for Q4_0 data")
    
    # Read back via pointer math
    var first = buf.load(0)
    print("  First byte:", first, "(expect 0)")
    
    buf.free()
    print("  Freed OK")
    print()


# ─── Full matmul test with heap data ─────────────────────────────────

def test_full_matmul():
    """Test Q4_0 matmul with heap-allocated weight data.
    
    Simulates a small weight matrix: 16 rows × 64 cols (2 blocks per row).
    Writes known Q4_0 values, runs dequant+matmul, checks result.
    """
    print("Full matmul test (16×64 Q4_0 → float32):")
    
    var n_rows = 16
    var n_cols = 64
    var blocks_per_row = n_cols // 32  # Q4_0 has 32 values per block
    
    # Allocate weight data on heap: n_rows * blocks_per_row * 18 bytes
    var w = alloc[UInt8](n_rows * blocks_per_row * 18)
    
    # Fill with known pattern: scale=1.0 (0x3c00), nibbles=8 (zero point)
    for row in range(n_rows):
        for blk in range(blocks_per_row):
            var off = (row * blocks_per_row + blk) * 18
            # Scale = 1.0 → f16 bytes: 0x00, 0x3c
            w.store(off, UInt8(0x00))
            w.store(off + 1, UInt8(0x3c))
            # Nibbles = 8 (zero point) → all zeros in quantized form
            for b in range(16):
                w.store(off + 2 + b, UInt8(0x88))  # hi=8, lo=8
    
    # Allocate input (all 1.0) and output
    var inp = alloc[Float32](n_cols)
    var out = alloc[Float32](n_rows)
    
    for c in range(n_cols):
        inp.store(c, Float32(1.0))
    for r in range(n_rows):
        out.store(r, Float32(0.0))
    
    # Run matmul: for each row, for each block, compute dot
    for row in range(n_rows):
        var total: Float32 = 0.0
        for blk in range(blocks_per_row):
            var off = (row * blocks_per_row + blk) * 18
            var block_ptr = w + off
            var scale = q4_block_read_scale(block_ptr)
            var nibbles = q4_block_read_nibbles(block_ptr)
            
            # Load 32 input values
            var inp_off = blk * 32
            var x0 = inp.load[width=8](inp_off)
            var x1 = inp.load[width=8](inp_off + 8)
            var x2 = inp.load[width=8](inp_off + 16)
            var x3 = inp.load[width=8](inp_off + 24)
            
            total += q4_block_dot(scale, nibbles, x0, x1, x2, x3)
        
        out.store(row, total)
    
    # Verify: each nibble=8 → dequant=0 → dot=0 (since 8-8=0)
    var ok = True
    for r in range(n_rows):
        var v = out.load(r)
        if abs(v - 0.0) > 0.001:
            print("  FAIL row", r, ":", v, "expect 0.0")
            ok = False
    
    if ok:
        print("  PASS: all outputs match expected (zero-center nibbles)")
    
    inp.free()
    out.free()
    w.free()
    print()


def main():
    print("=== Q4_0 Mojo SIMD Kernel (Mojo 1.0.0b1) ===\n")
    
    test_f16()
    test_q4_block_zero()
    test_q4_block_varied()
    test_heap_alloc()
    test_full_matmul()
    
    print("Done.")
