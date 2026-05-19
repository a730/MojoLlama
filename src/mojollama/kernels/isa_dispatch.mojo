"""MojoLlama ISA Dispatch — comptime SIMD width selection and kernel parameterization.

Dispatches between AVX-512 (F32x16) and AVX2 (F32x8) based on comptime target detection.
All kernels use parameteric SIMD width for maximum portability and performance.

Architecture priority:
  AVX-512-VNNI > AVX-512-F > AVX2+FMA > AVX2 > SSE2 > scalar

llama.cpp uses separate compiled files per ISA path; we use comptime branching.
"""
from std.sys.info import CompilationTarget

# ─── ISA Feature Detection ────────────────────────────────────────────

@always_inline
fn has_avx512f() -> Bool:
    """Check if target supports AVX-512F (foundation)."""
    return CompilationTarget.has_avx512f()

@always_inline
fn has_avx512vnni() -> Bool:
    """Check if target supports AVX-512-VNNI (int8 dot product)."""
    return CompilationTarget.has_avx512vnni()

@always_inline
fn has_avx2() -> Bool:
    return CompilationTarget.has_avx2()

@always_inline
fn has_fma() -> Bool:
    return CompilationTarget.has_fma()

# ─── SIMD Width Selection ─────────────────────────────────────────────

alias SIMD_WIDTH_AVX2 = 8     # 256 bits / 32 = 8 floats
alias SIMD_WIDTH_AVX512 = 16   # 512 bits / 32 = 16 floats
alias SIMD_WIDTH_NEON = 4      # 128 bits / 32 = 4 floats (ARM)

comptime fn get_simd_width() -> Int:
    """Select SIMD width based on target ISA.
    AVX-512: 16 floats per vector (2x AVX2 throughput)
    AVX2: 8 floats per vector
    Fallback: 4 floats (SSE2/NEON)
    """
    if has_avx512f():
        return SIMD_WIDTH_AVX512
    elif has_avx2():
        return SIMD_WIDTH_AVX2
    else:
        return 4

alias SIMD_WIDTH = get_simd_width()
alias F32xW = SIMD[DType.float32, SIMD_WIDTH]
alias I8xW = SIMD[DType.int8, SIMD_WIDTH]
alias U8xW = SIMD[DType.uint8, SIMD_WIDTH]
alias I32xW = SIMD[DType.int32, SIMD_WIDTH]

# ─── ISA Info String ──────────────────────────────────────────────────

fn isa_info() -> String:
    """Return human-readable ISA info string."""
    comptime var parts = List[String]()
    if has_avx512vnni():
        parts.append("AVX-512-VNNI")
    if has_avx512f():
        parts.append("AVX-512F")
    if has_avx2():
        parts.append("AVX2")
    if has_fma():
        parts.append("FMA")
    if len(parts) == 0:
        return "scalar"
    return "|".join(parts)

fn simd_info() -> String:
    """Return SIMD configuration info."""
    return "SIMD width=" + String(SIMD_WIDTH) + " floats (" + String(SIMD_WIDTH * 32) + " bits) [" + isa_info() + "]"

# ─── Utility: Prefetch Hint ───────────────────────────────────────────

@always_inline
fn prefetch_read(ptr: UnsafePointer[UInt8, MutAnyOrigin], offset: Int):
    """Emit prefetch hint for sequential read access.
    Uses __builtin_prefetch equivalent for cache warming.
    """
    # Mojo doesn't expose __builtin_prefetch yet, but we can use
    # volatile read trick to hint the hardware prefetcher
    pass  # Placeholder — will use intrinsics when available

# ─── Utility: Byte Swap for Big-Endian ─────────────────────────────────

@always_inline
fn bswap16(x: UInt16) -> UInt16:
    """Byte-swap a 16-bit value for big-endian GGUF fields."""
    return ((x & 0xFF) << 8) | ((x >> 8) & 0xFF)

# ─── Half-Precision Float Conversion ──────────────────────────────────

@always_inline
fn f16_to_f32(h: UInt16) -> Float32:
    """Convert IEEE 754 half-precision to single-precision.
    Uses F16C (VCVTPH2PS) on AVX2+ targets, software fallback otherwise.
    This is the per-scalar version; for bulk conversion see f16x8_to_f32x8.
    """
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        # F16C path: hardware float16 is stored as UInt16, bitcast to DType.float16
        # then implicit conversion to Float32 via Mojo's type system
        return Float32(bitcast[DType.float16](h))
    else:
        # Software decode: sign(1) | exponent(5) | mantissa(10)
        var s = (UInt32(h) >> 15) & 1
        var e = (UInt32(h) >> 10) & 0x1f
        var m = UInt32(h) & 0x3ff
        if e == 0:
            if m == 0:
                # Zero (preserving sign)
                return Float32(bitcast[DType.float32](s << 31))
            # Subnormal: renormalize
            var mm = m
            var c: UInt32 = 0
            while mm > 0:
                mm >>= 1
                c += 1
            var sh = 24 - c
            var vm = m << (sh + 13)
            return Float32(bitcast[DType.float32]((s << 31) | ((UInt32(113 - sh)) << 23) | (vm & 0x7fffff)))
        if e == 31:
            # Inf or NaN
            return Float32(bitcast[DType.float32]((s << 31) | 0x7f800000 | (m << 13)))
        # Normalized
        return Float32(bitcast[DType.float32]((s << 31) | ((e + 112) << 23) | (m << 13)))

@always_inline
fn f16x8_to_f32x8(h0: UInt16, h1: UInt16, h2: UInt16, h3: UInt16,
                   h4: UInt16, h5: UInt16, h6: UInt16, h7: UInt16) -> F32xW:
    """Convert 8 half-precision values to a SIMD vector.
    On F16C targets, this uses VCVTPH2PS for 8-wide conversion.
    """
    # For AVX-512 we only use the lower 8 lanes; upper 8 are zero
    var result = F32xW(0.0)
    result[0] = f16_to_f32(h0)
    result[1] = f16_to_f32(h1)
    result[2] = f16_to_f32(h2)
    result[3] = f16_to_f32(h3)
    result[4] = f16_to_f32(h4)
    result[5] = f16_to_f32(h5)
    result[6] = f16_to_f32(h6)
    if SIMD_WIDTH > 7:
        result[7] = f16_to_f32(h7)
    if SIMD_WIDTH == 16:
        # Upper 8 lanes are zero (caller should handle)
        pass
    return result

@always_inline
fn load_f16_scale(ptr: UnsafePointer[UInt8, MutAnyOrigin], offset: Int) -> Float32:
    """Load a Q4_0/Q8_0 style f16 scale from packed byte pair.
    GGUF stores 16-bit quantization scales in little-endian format:
      lo = ptr[offset], hi = ptr[offset+1]
    """
    var lo = UInt32(ptr.load(offset))
    var hi = UInt32(ptr.load(offset + 1))
    return f16_to_f32(UInt16((hi << 8) | lo))

# ─── Utility: Saturating Cast ──────────────────────────────────────────

@always_inline
fn saturating_cast_i8(x: Float32) -> Int8:
    """Clamp float to [-128, 127] range and cast to Int8 for Q8_0 quantization."""
    if x > 127.0:
        return Int8(127)
    elif x < -128.0:
        return Int8(-128)
    else:
        return Int8(x)

@always_inline
fn saturating_cast_u8(x: Float32) -> UInt8:
    """Clamp float to [0, 255] range and cast to UInt8."""
    if x > 255.0:
        return UInt8(255)
    elif x < 0.0:
        return UInt8(0)
    else:
        return UInt8(x)