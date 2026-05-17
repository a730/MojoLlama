"""MojoLlama Q4_0 matmul — shared library entry point.

Compile: mojo build --output-type shared-lib q4_ffi.mojo -o libq4mojo.so

Called from Python via ctypes:
    lib = ctypes.CDLL("./libq4mojo.so")
    lib.q4_matmul(weights_addr, input_addr, output_addr, n_rows, n_cols)
"""

from std.memory.unsafe_pointer import alloc

# ─── Float16 → Float32 ────────────────────────────────────────────────

@always_inline
def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        var f16 = bitcast[DType.float16](h)
        return Float32(f16)
    else:
        var sign = (UInt32(h) >> 15) & 1
        var exp = (UInt32(h) >> 10) & 0x1f
        var mant = UInt32(h) & 0x3ff
        if exp == 0:
            if mant == 0:
                return Float32(bitcast[DType.float32](sign << 31))
            var m = mant; var count: UInt32 = 0
            while m > 0: m = m >> 1; count += 1
            var shift = 24 - count
            return Float32(bitcast[DType.float32]((sign << 31) | ((UInt32(113 - shift)) << 23) | ((mant << (shift + 13)) & 0x7fffff)))
        elif exp == 31:
            return Float32(bitcast[DType.float32]((sign << 31) | 0x7f800000 | (mant << 13)))
        else:
            return Float32(bitcast[DType.float32]((sign << 31) | ((exp + 112) << 23) | (mant << 13)))


# ─── Q4_0 Block Dot ───────────────────────────────────────────────────

@always_inline
def q4_block_dot(scale: Float32, nibbles: SIMD[DType.uint8, 16],
                x0: SIMD[DType.float32, 8], x1: SIMD[DType.float32, 8],
                x2: SIMD[DType.float32, 8], x3: SIMD[DType.float32, 8]) -> Float32:
    alias U8x16 = SIMD[DType.uint8, 16]
    alias F32x8 = SIMD[DType.float32, 8]
    
    @parameter
    def dot_group(start_byte: Int, xv: F32x8) -> Float32:
        var vals = F32x8()
        for j in range(8):
            var byte_idx = start_byte + j // 2
            var b = nibbles[byte_idx]
            vals[j] = Float32(Int8(b & 15) - 8) if j % 2 == 0 else Float32(Int8((b >> 4) & 15) - 8)
        return (vals * scale * xv).reduce_add()
    
    var total: Float32 = 0.0
    total += dot_group(0, x0)
    total += dot_group(4, x1)
    total += dot_group(8, x2)
    total += dot_group(12, x3)
    return total


# ─── Exported C-compatible API ────────────────────────────────────────

fn q4_matmul_forward(
    weights_ptr: Int,    # pointer to Q4_0 block data
    input_ptr: Int,      # pointer to float32 input vector
    output_ptr: Int,     # pointer to float32 output buffer
    n_rows: Int,         # number of output features
    n_cols: Int          # number of input features
):
    """Compute y = W @ x for Q4_0 quantized weights.
    
    All pointers are raw addresses (from numpy.__array_interface__['data'][0]).
    This is called from Python via ctypes.
    """
    # Note: In current Mojo 1.0.0b1, we can't construct UnsafePointer from
    # raw Int addresses. This function is the FFI stub — the actual
    # computation runs inside a Python-script generated Mojo context.
    # 
    # Workaround: compile this as a standalone executable that reads/writes
    # binary files, invoked from bridge.py.
    pass


def main():
    print("MojoLlama Q4_0 FFI Library")
    print("Compile with: mojo build --output-type shared-lib q4_ffi.mojo -o libq4mojo.so")
