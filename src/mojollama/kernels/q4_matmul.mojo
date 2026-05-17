"""MojoLlama Q4_0 SIMD Kernel — pure SIMD operations on Q4_0 blocks.

Target: AVX2 (Threadripper 3970X). No heap allocation needed.
When Mojo's heap APIs mature, wraps pointer-based matrix ops around these.

Q4_0 block: 32 values packed into 18 bytes (2b f16 scale + 16b nibbles).
Reference: MAX qmatmul.mojo — AVX2 pmaddw path.
"""

from std.sys.info import CompilationTarget
from std.memory import bitcast, UnsafePointer

# SIMD types for AVX2 (256-bit)
alias F32x8  = SIMD[DType.float32, 8]
alias I8x32  = SIMD[DType.int8, 32]
alias U8x16  = SIMD[DType.uint8, 16]


# ─── Float16 → Float32 ────────────────────────────────────────────────

@always_inline
fn f16_to_f32(h: UInt16) -> Float32:
    """float16 bits → float32. Uses F16C via hardware conversion when AVX2 available."""
    comptime if CompilationTarget.has_avx2():
        # Bitcast UInt16 to Float16, then extend to Float32.
        # This compiles to VCVTPH2PS (F16C instruction) on AVX2.
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
                # Count leading zeros to find the highest set bit
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


# ─── Q4_0 Block Dot Product ─────────────────────────────────────────────

@always_inline
fn q4_block_dot(scale: Float32, nibbles: U8x16, x: F32x8, x1: F32x8,
                x2: F32x8, x3: F32x8) -> Float32:
    """Dot product of one Q4_0 block with 32 float32 inputs.
    
    Fused dequant + dot: Σ_i (nibble_i - 8) * scale * x_i
    
    nibbles: 16 bytes = 32 × 4-bit quantized values
    x, x1, x2, x3: 4 SIMD vectors covering 32 float32 inputs
    scale: float32 dequant scale
    
    Returns: scalar dot product (reduced from SIMDFMA).
    """
    var total: Float32 = 0.0
    
    # Process 4 groups of 8 values (1 SIMD width each)
    @parameter
    fn dot_group(start_byte: Int, xv: F32x8) -> Float32:
        """Dot product of 8 values from nibbles[start_byte:start_byte+4] with xv."""
        var vals = F32x8()
        for j in range(8):
            var byte_idx = start_byte + j // 2
            var b = nibbles[byte_idx]
            if j % 2 == 0:
                # Low nibble: value[2i]
                vals[j] = Float32(Int8(b & 15) - 8)
            else:
                # High nibble: value[2i+1]
                vals[j] = Float32(Int8((b >> 4) & 15) - 8)
        return (vals * scale * xv).reduce_add()
    
    total += dot_group(0, x)
    total += dot_group(4, x1)
    total += dot_group(8, x2)
    total += dot_group(12, x3)
    
    return total


# ─── Full Q4_0 Matmul (pointer-based, future) ───────────────────────────
# When heap APIs (unsafe_from_address, alloc) land in Mojo, this module
# gets a `q4_matmul_forward` that takes raw pointer addresses as Int and
# processes full weight matrices using the `q4_block_dot` primitive above.
# 
# Reference implementation: see C kernel at model/q4_matmul_c.c


# ─── Tests ─────────────────────────────────────────────────────────────

fn test_f16():
    """Test float16→float32 conversion."""
    print("f16→f32 tests:")
    # 0x3c00 = +1.0
    var f = f16_to_f32(UInt16(0x3c00))
    print("  0x3c00 →", f, "(expect 1.0)")
    
    # 0x3800 = 0.5
    f = f16_to_f32(UInt16(0x3800))
    print("  0x3800 →", f, "(expect 0.5)")
    
    # 0xbc00 = -1.0
    f = f16_to_f32(UInt16(0xbc00))
    print("  0xbc00 →", f, "(expect -1.0)")
    print()


fn test_q4_block_zero():
    """Test block dot with all-zero quantized values.
    
    scale=0.5 (0x3800), all nibbles=0 → dequant=-4, input=2.0
    Each term: -4 * 0.5 * 2.0 = -4. 32 terms → -128
    """
    var scale = f16_to_f32(UInt16(0x3800))
    var nibbles = U8x16()  # all zeros
    var x = F32x8(2.0)
    var dot = q4_block_dot(scale, nibbles, x, x, x, x)
    print("zero block dot:", dot, "(expect -256.0)")
    if abs(dot - (-256.0)) < 1.0:
        print("  PASS")
    else:
        print("  FAIL")
    print()


fn test_q4_block_varied():
    """Test with specific nibble pattern.
    
    scale=1.0, nibbles alternate 0x00 (lo=0, hi=0) and 0xFF (lo=-1, hi=-1)
    Input all 1.0
    byte 0: lo=-8, hi=-8 → both -8
    byte 1: lo=-7, hi=-7 → both -7
    Expected: 16* -8 + 16* -7 = -240
    """
    var scale = f16_to_f32(UInt16(0x3c00))  # 1.0
    var nibbles = U8x16()
    for i in range(16):
        nibbles[i] = 0x00 if i % 2 == 0 else 0xFF
    var x = F32x8(1.0)
    var dot = q4_block_dot(scale, nibbles, x, x, x, x)
    # Byte 0: both -8. Byte 1: both +7 (0xFF: lo=15-8=7, hi=15-8=7)
    # 8 values of -8 + 8 values of 7 = -8 for 2 bytes
    # 16 bytes / 2 = 8 pairs, each -1 → total -8
    # Wait, let me recalculate
    # nibbles[0]=0x00 → lo=-8, hi=-8 → values[0:2] = -8, -8
    # nibbles[1]=0xFF → lo=7, hi=7 → values[2:4] = 7, 7
    # Pattern repeats: -8, -8, 7, 7, -8, -8, 7, 7, ...
    # 8 of each = 8*(-8) + 8*(7) + 8*(-8) + 8*(7) = -64+56-64+56 = -16
    print("varied block dot:", dot, "(expect -16.0)")
    if abs(dot - (-16.0)) < 1.0:
        print("  PASS")
    else:
        print("  FAIL")
    print()


fn main():
    print("=== Q4_0 Mojo SIMD Kernel ===")
    print("Target: AVX2=", CompilationTarget.has_avx2(), 
          "FMA=", CompilationTarget.has_fma())
    print()
    
    test_f16()
    test_q4_block_zero()
    test_q4_block_varied()
    
    print("Done.")
