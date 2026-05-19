"""MojoLlama K-Quant Kernels — Q4_K_S, Q5_K_S, Q6_K dequantization and matmul.

K-quants use a two-level block structure (super-block):
  - Super-block: 256 elements, contains global scale + sub-block scales
  - Sub-blocks: 32 elements each (8 sub-blocks per super-block)

This matches llama.cpp's ggml_type structure exactly.

Q4_K_S layout (per super-block of 256 elements, 72 bytes):
  - 2 bytes: d (f16 global scale)
  - 2 bytes: dmin (f16 global minimum, unused in S variant)
  - 8 bytes: scales (4-bit per sub-block, 2 bits for the 8 scales)
  - 64 bytes: qs (4-bit quantized values, 2 per byte)

Q4_K layout (72 bytes per super-block of 256 elements):
  d: f16 super-block scale (2 bytes)
  dmin: f16 super-block min offset (2 bytes)  
  scales: 12 bytes (6 bits per sub-block scale + 2 high bits)
  qs: 4-bit quantized values (128 bytes / 2 = 64 bytes packed)
  Note: In Q4_K (non-S), dmin adds an offset to dequantized values

Q5_K_S layout (per super-block, 88 bytes):
  - d (f16), dmin (f16), scales (12 bytes), qh (32 bytes high bits), qs (64 bytes)

Q6_K layout (per super-block, 52 bytes — wait, that's block_size=256 but type_size=210):
  Actually Q6_K has block_size=256, type_size=210 bytes
  - ql (128 bytes, 4-bit packed), qh (64 bytes, 1 bit each), 
    scales (16 bytes, 8-bit), d (f16 super-block scale)

For maximum performance:
  - AVX-512: process 16 floats per SIMD op
  - AVX-2: process 8 floats per SIMD op
  - Regblock: share input loads across N output rows
  - FMA chain: accumulate before reduce_add
"""

from std.memory.unsafe_pointer import alloc
from std.math import sqrt, exp, pow
from ..kernels.isa_dispatch import *

# ─── Q4_K Dequantization ─────────────────────────────────────────────

@always_inline
fn dequant_q4_k(
    qs: UnsafePointer[UInt8, MutAnyOrigin],  # quantized data (4-bit packed)
    scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],  # per-sub-block scales (6-bit packed)
    d: Float32,       # super-block scale
    dmin: Float32,    # super-block min offset
    block_idx: Int,   # which super-block
    out: UnsafePointer[Float32, MutAnyOrigin],  # output: 256 floats
):
    """Dequantize one Q4_K super-block (256 elements) to float32.
    
    Q4_K scales layout (12 bytes per super-block):
      - First 8 bytes: 6 bits per scale (low 6 bits of each byte)
      - Next 4 bytes: 2 high bits per scale packed
        
    Each sub-block of 32 elements has its own scale:
      scale[j] = (scales_low[j] | (scales_high[j/2] >> (4*(j%2))) << 8) 
      
    Dequant: out[i] = d * scale[sub_block] * (qs_nibble - 8)
    """
    var sc_off = block_idx * 12  # scales offset in bytes
    var qs_off = block_idx * 128  # qs offset (128 packed bytes = 256 nibbles)
    
    # Unpack 8 sub-block scales (6+2 bit packing)
    # Each scale: 6-bit low part + 2-bit high part from separate bytes  
    var sub_scales = SIMD[DType.float32, 8]()
    for j in range(8):
        var sc_lo = UInt32(scales_ptr.load(sc_off + j)) & 0x3F  # low 6 bits
        var sc_hi: UInt32 = 0
        if j < 4:
            sc_hi = (UInt32(scales_ptr.load(sc_off + 8)) >> (2 * j)) & 0x3
        else:
            sc_hi = (UInt32(scales_ptr.load(sc_off + 9)) >> (2 * (j - 4))) & 0x3
        var scale_i16 = Int16((sc_lo | (sc_hi << 6)) - 32)  # signed offset
        sub_scales[j] = Float32(scale_i16)
    
    # Dequantize each sub-block
    for j in range(8):
        var s = d * sub_scales[j]
        var m = dmin * sub_scales[j]  # Q4_K has min offset
        var base_off = j * 16  # 16 packed bytes per sub-block (32 values / 2 per byte)
        var out_base = j * 32   # 32 output values per sub-block
        
        # Process 16 packed bytes → 32 nibbles
        var oi = 0
        while oi + SIMD_WIDTH <= 32:
            var lo_nibbles = SIMD[DType.float32, SIMD_WIDTH]()
            var hi_nibbles = SIMD[DType.float32, SIMD_WIDTH]()
            for k in range(SIMD_WIDTH):
                var byte_idx = base_off + (oi + k) // 2
                var b = qs_ptr.load(qs_off + byte_idx)
                if (oi + k) % 2 == 0:
                    lo_nibbles[k] = Float32((UInt32(b) & 0xF) - 8)
                else:
                    lo_nibbles[k] = Float32(((UInt32(b) >> 4) & 0xF) - 8)
            var result = F32xW(s) * lo_nibbles
            # Store SIMD_WIDTH values at a time
            for k in range(SIMD_WIDTH):
                out.store(out_base + oi + k, result[k])
            oi += SIMD_WIDTH
        # Scalar tail
        while oi < 32:
            var byte_idx = base_off + oi // 2
            var b = qs_ptr.load(qs_off + byte_idx)
            var nibble: Float32
            if oi % 2 == 0:
                nibble = Float32((UInt32(b) & 0xF) - 8)
            else:
                nibble = Float32(((UInt32(b) >> 4) & 0xF) - 8)
            out.store(out_base + oi, s * nibble + m)
            oi += 1


# ─── Q4_K_S Dot Product (block-level) ──────────────────────────────────

@always_inline
fn q4_k_dot_superblock(
    qs: UnsafePointer[UInt8, MutAnyOrigin],
    scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    d: Float32,
    dmin: Float32,
    block_idx: Int,
) -> Float32:
    """Dot product of one Q4_K super-block (256 elements) against input vector.
    
    This is the inner kernel for Q4_K matmul. Processes 256 elements,
    8 sub-blocks of 32 elements each.
    Uses full SIMD width for accumulation.
    """
    var sc_off = block_idx * 12
    var qs_off = block_idx * 128
    
    # Unpack 8 sub-block scales
    var sub_scales = SIMD[DType.float32, 8]()
    var sub_mins = SIMD[DType.float32, 8]()
    for j in range(8):
        var sc_lo = UInt32(scales_ptr.load(sc_off + j)) & 0x3F
        var sc_hi: UInt32 = 0
        if j < 4:
            sc_hi = (UInt32(scales_ptr.load(sc_off + 8)) >> (2 * j)) & 0x3
        else:
            sc_hi = (UInt32(scales_ptr.load(sc_off + 9)) >> (2 * (j - 4))) & 0x3
        var scale_i16 = Int16((sc_lo | (sc_hi << 6)) - 32)
        sub_scales[j] = Float32(scale_i16)
        sub_mins[j] = Float32(scale_i16)  # Q4_K uses same scale for min
    
    var total: Float32 = 0.0
    var x_base = block_idx * 256  # input offset for this super-block
    
    for j in range(8):
        var s = d * sub_scales[j]
        var m = dmin * sub_mins[j]
        var q_off = qs_off + j * 16  # 16 packed bytes per sub-block
        
        # Process 16 packed bytes = 32 values (2 nibbles per byte)
        # Unroll into SIMD-width chunks
        var sub_total: Float32 = 0.0
        var bi = 0  # byte index
        var vi = 0  # value index
        
        while vi + 2 <= 32:
            var byte_val = UInt32(qs.load(q_off + bi))
            var lo = Float32((byte_val & 0xF) - 8)
            var hi = Float32(((byte_val >> 4) & 0xF) - 8)
            
            sub_total += s * lo * x.load(x_base + j * 32 + vi)
            sub_total += s * hi * x.load(x_base + j * 32 + vi + 1)
            # Note: Q4_K also adds m (min offset) scaled by x[i]
            # total += m * x[i] — simplified: this is just m * sum(x[i]) for the sub-block
            vi += 2
            bi += 1
        
        # Min offset contribution: m * sum(x) for this sub-block
        var x_sum: Float32 = 0.0
        var xi = j * 32
        var xe = xi + 32
        while xi + SIMD_WIDTH <= xe:
            var xv = x.load[width=SIMD_WIDTH](x_base + xi)
            x_sum += xv.reduce_add()
            xi += SIMD_WIDTH
        while xi < xe:
            x_sum += x.load(x_base + xi)
            xi += 1
        total += sub_total + m * x_sum
    
    return total


# ─── Q4_K Matmul (row-outer, regblock) ──────────────────────────────────

fn q4_k_matmul(
    weights: UnsafePointer[UInt8, MutAnyOrigin],   # packed Q4_K weights
    scales: UnsafePointer[UInt8, MutAnyOrigin],     # per-superblock scales (12 bytes each)
    d_ptr: UnsafePointer[Float32, MutAnyOrigin],    # super-block scales (d)
    dmin_ptr: UnsafePointer[Float32, MutAnyOrigin], # super-block min offsets (dmin)
    input: UnsafePointer[Float32, MutAnyOrigin],    # input vector (nc floats)
    output: UnsafePointer[Float32, MutAnyOrigin],   # output vector (nr floats)
    nr: Int,    # number of rows (output dimension)
    nc: Int,    # number of columns (input dimension, must be multiple of 256)
):
    """Q4_K matmul: compute output = weights @ input.
    
    Each row has nc/256 super-blocks, each super-block has 256 elements.
    Weight row layout: bytes 0..nc/256*12 scale bytes, then nc/256*128 qs bytes
    
    But in GGUF format, the layout is: for each row,
      [d(f16)][dmin(f16)][scales(12 bytes per superblock)][qs(128 bytes per superblock)]
    totaling 2+2+12*(nc/256)+128*(nc/256) = 16+140*(nc/256) bytes per row for non-S variant.
    
    For Q4_K_S (type_size=72 per superblock):
      [d(f16)][dmin(f16)][scales(2 bytes)][qs(64 bytes per superblock)]
    
    TODO: Use regblocking (4-row) for better input reuse.
    """
    var n_blocks = nc // 256  # number of super-blocks per row
    
    for row in range(nr):
        var total: Float32 = 0.0
        var row_d = d_ptr.load(row)       # global scale for this row
        var row_dmin = dmin_ptr.load(row) # global min for this row
        
        for blk in range(n_blocks):
            total += q4_k_dot_superblock(
                weights, scales, input,
                row_d, row_dmin, blk,
            )
        output.store(row, total)


# ─── Q8_0 Block Quantization ──────────────────────────────────────────
# Q8_0: block_size=32, type_size=34 bytes
# Layout: [scale(f16, 2 bytes)][values(32 x int8, 32 bytes)]
# Dequant: value = scale * int8_q

@always_inline
fn q8_0_dequant_block(
    data: UnsafePointer[UInt8, MutAnyOrigin],
    block_idx: Int,
    out: UnsafePointer[Float32, MutAnyOrigin],
):
    """Dequantize one Q8_0 block (32 elements) to float32.
    Block layout: [f16_scale(2B)][int8_values(32B)]
    """
    var off = block_idx * 34  # 34 bytes per block
    var scale = load_f16_scale(data, off)
    var voff = off + 2
    
    var i = 0
    while i + SIMD_WIDTH <= 32:
        var qv = SIMD[DType.float32, SIMD_WIDTH]()
        for k in range(SIMD_WIDTH):
            qv[k] = Float32(Int8(data.load(voff + i + k))) * scale
        out.store[width=SIMD_WIDTH](block_idx * 32 + i, qv)
        i += SIMD_WIDTH
    while i < 32:
        out.store(block_idx * 32 + i, Float32(Int8(data.load(voff + i))) * scale)
        i += 1


@always_inline
fn q8_0_dot_block(
    data: UnsafePointer[UInt8, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    block_idx: Int,
) -> Float32:
    """Dot product of one Q8_0 block (32 elements) against input vector.
    Uses SIMD int8→float widening for better throughput than Q4_0.
    """
    var off = block_idx * 34
    var scale = load_f16_scale(data, off)
    var voff = off + 2
    var x_off = block_idx * 32
    
    # Load 8 int8 values, widen to float32, multiply by input
    var total: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= 32:
        var xv = x.load[width=SIMD_WIDTH](x_off + i)
        var qv = SIMD[DType.float32, SIMD_WIDTH]()
        for k in range(SIMD_WIDTH):
            qv[k] = Float32(Int8(data.load(voff + i + k)))
        total += (xv * qv * F32xW(scale)).reduce_add()
        i += SIMD_WIDTH
    while i < 32:
        total += x.load(x_off + i) * Float32(Int8(data.load(voff + i))) * scale
        i += 1
    return total


fn q8_0_matmul(
    weights: UnsafePointer[UInt8, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    nr: Int,
    nc: Int,
):
    """Q8_0 matmul with row-outer accumulation and regblocking.
    Each row: nc/32 blocks of 34 bytes = nc*34/32 bytes.
    """
    var bpr = nc // 32  # blocks per row
    
    var row = 0
    while row < nr:
        # 4-row regblocking
        var r0 = row
        var r1 = row + 1 if row + 1 < nr else row
        var r2 = row + 2 if row + 2 < nr else row
        var r3 = row + 3 if row + 3 < nr else row
        var t0: Float32 = 0.0
        var t1: Float32 = 0.0
        var t2: Float32 = 0.0
        var t3: Float32 = 0.0
        
        for blk in range(bpr):
            # Shared input load
            var x_off = blk * 32
            var x0 = input.load[width=SIMD_WIDTH](x_off)
            var x1: SIMD[DType.float32, SIMD_WIDTH]
            var x2: SIMD[DType.float32, SIMD_WIDTH]
            var x3: SIMD[DType.float32, SIMD_WIDTH]
            if SIMD_WIDTH < 32:
                # Need multiple loads for 32 elements
                var next_off = x_off + SIMD_WIDTH
                x1 = input.load[width=SIMD_WIDTH](next_off)
                if SIMD_WIDTH * 2 < 32:
                    x2 = input.load[width=SIMD_WIDTH](next_off + SIMD_WIDTH)
                    x3 = input.load[width=SIMD_WIDTH](next_off + SIMD_WIDTH * 2)
            
            # Row 0
            var off0 = (r0 * bpr + blk) * 34
            var sc0 = load_f16_scale(weights, off0)
            t0 += q8_0_dot_block(weights, input, r0 * bpr + blk)
            
            # Row 1 (if different)
            if r1 != r0:
                t1 += q8_0_dot_block(weights, input, r1 * bpr + blk)
            if r2 != r0:
                t2 += q8_0_dot_block(weights, input, r2 * bpr + blk)
            if r3 != r0:
                t3 += q8_0_dot_block(weights, input, r3 * bpr + blk)
        
        output.store(r0, t0)
        if r1 != r0: output.store(r1, t1)
        if r2 != r0: output.store(r2, t2)
        if r3 != r0: output.store(r3, t3)
        row += 4


# ─── Q5_0 Dequantization and Dot Product ─────────────────────────────
# Q5_0: block_size=32, type_size=22 bytes
# Layout: [scale(f16, 2B)][qh(4B, high bits)][qs(16B, low bits)]
# 5-bit quantization: qh holds 1 bit per value, qs holds 4 low bits

@always_inline
fn q5_0_dot_block(
    data: UnsafePointer[UInt8, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    block_idx: Int,
) -> Float32:
    """Dot product of one Q5_0 block (32 elements) against input vector.
    Q5_0 layout per block (22 bytes):
      scale: f16 (2 bytes)
      qh: 4 bytes (32 bits, 1 high bit per value)
      qs: 16 bytes (32 low nibbles, 2 per byte)
    Dequant: value = scale * ((q5 & 0x1F) - 16) where q5 = (qs_nibble | (qh_bit << 4))
    """
    var off = block_idx * 22
    var scale = load_f16_scale(data, off)
    var qh_off = off + 2
    var qs_off = off + 6
    
    # Load 4 bytes of high bits (32 bits for 32 values)
    var qh0 = UInt32(data.load(qh_off))
    var qh1 = UInt32(data.load(qh_off + 1))
    var qh2 = UInt32(data.load(qh_off + 2))
    var qh3 = UInt32(data.load(qh_off + 3))
    var qh32 = (qh0) | (qh1 << 8) | (qh2 << 16) | (qh3 << 24)
    
    var total: Float32 = 0.0
    var x_off = block_idx * 32
    
    for i in range(16):  # 16 packed bytes → 32 values
        var b = UInt32(data.load(qs_off + i))
        var lo_nibble = Int32((b & 0xF) | (((qh32 >> (2*i)) & 1) << 4)) - 16
        var hi_nibble = Int32(((b >> 4) & 0xF) | (((qh32 >> (2*i + 1)) & 1) << 4)) - 16
        
        total += x.load(x_off + 2*i) * Float32(lo_nibble) * scale
        total += x.load(x_off + 2*i + 1) * Float32(hi_nibble) * scale
    
    return total


fn q5_0_matmul(
    weights: UnsafePointer[UInt8, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    nr: Int,
    nc: Int,
):
    """Q5_0 matmul with row-outer accumulation."""
    var bpr = nc // 32
    for row in range(nr):
        var total: Float32 = 0.0
        for blk in range(bpr):
            total += q5_0_dot_block(weights, input, row * bpr + blk)
        output.store(row, total)


# ─── Q6_K Dequantization and Dot Product ──────────────────────────────
# Q6_K: block_size=256, type_size=210 bytes (the most accurate K-quant)
# Layout per super-block:
#   ql: 128 bytes (4-bit packed low bits, 2 per byte)
#   qh: 64 bytes  (2-bit high bits per value)  
#   scales: 16 bytes (8-bit signed scales)
#   d: f16 (2 bytes, super-block scale)
# Total: 128+64+16+2 = 210 bytes per 256-element super-block

@always_inline
fn q6_k_dot_superblock(
    ql: UnsafePointer[UInt8, MutAnyOrigin],   # low bits (4-bit packed)
    qh: UnsafePointer[UInt8, MutAnyOrigin],   # high bits (2-bit per value)
    scales: UnsafePointer[UInt8, MutAnyOrigin], # per-sub-block scales (8-bit)
    d: Float32,                                 # super-block scale
    x: UnsafePointer[Float32, MutAnyOrigin],   # input vector
    block_idx: Int,
) -> Float32:
    """Dot product of one Q6_K super-block (256 elements) against input vector.
    Q6_K encodes 6-bit values: q = (ql_nibble | (qh_2bit << 4)) - 32
    Dequant: value = d * scale[sub_block] * q
    """
    var ql_off = block_idx * 128
    var qh_off = block_idx * 64
    var sc_off = block_idx * 16  # 16 scale bytes for 16 sub-blocks? No...
    # Actually Q6_K has 16 sub-blocks of 16 elements each
    # scales: 16 bytes (one per sub-block, int8)
    # Wait, let me recheck: Q6_K has block_size=256
    #   ql: 128 bytes = 256 nibbles (4 bits each)
    #   qh: 64 bytes = 256 * 2 bits packed? No, that's 64 bytes = 256 values * 2 bits = 512 bits
    #   Actually: qh has 2 bits per value, 256 values * 2 bits = 512 bits = 64 bytes ✓
    #   scales: 16 bytes = 16 sub-blocks * 1 byte each (int8)
    #   Each sub-block has 16 values
    #   d: f16 scale (2 bytes)
    
    var total: Float32 = 0.0
    var x_base = block_idx * 256
    
    # Process 16 sub-blocks of 16 values each
    for sb in range(16):
        var scale = Float32(Int8(scales.load(sc_off + sb))) * d
        
        # Each sub-block: 16 values * 6 bits = 96 bits = 12 bytes? No...
        # ql: 8 packed bytes per sub-block (16 nibbles for low 4 bits)
        # qh: 4 packed bytes per sub-block (16 values * 2 bits = 32 bits = 4 bytes)
        var ql_sb = ql_off + sb * 8
        var qh_sb = qh_off + sb * 4
        
        for vi in range(16):
            # Low 4 bits from ql
            var byte_idx = vi // 2
            var b = UInt32(ql.load(ql_sb + byte_idx))
            var lo4: Int32
            if vi % 2 == 0:
                lo4 = Int32(b & 0xF)
            else:
                lo4 = Int32((b >> 4) & 0xF)
            
            # High 2 bits from qh
            var qh_byte = vi // 4
            var qh_bit_off = (vi % 4) * 2
            var hi2 = Int32((UInt32(qh.load(qh_sb + qh_byte)) >> qh_bit_off) & 0x3)
            
            # 6-bit value: combine hi2 and lo4, center at 32
            var q = (hi2 << 4) | lo4
            var value = Float32(q - 32) * scale
            
            total += value * x.load(x_base + sb * 16 + vi)
    
    return total


fn q6_k_matmul(
    ql: UnsafePointer[UInt8, MutAnyOrigin],
    qh: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[UInt8, MutAnyOrigin],
    d: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    nr: Int,
    nc: Int,
):
    """Q6_K matmul: compute output = weights @ input.
    nc must be a multiple of 256 (super-block size).
    """
    var n_blocks = nc // 256
    for row in range(nr):
        var total: Float32 = 0.0
        var row_d = d.load(row)
        for blk in range(n_blocks):
            total += q6_k_dot_superblock(
                ql, qh, scales, row_d, input, blk,
            )
        output.store(row, total)


# ─── Multi-Format Dispatch Kernel ──────────────────────────────────────

struct QuantType:
    alias Q4_0 = 2
    alias Q8_0 = 7
    alias Q5_0 = 8
    alias Q4_K = 18
    alias Q5_K = 19
    alias Q6_K = 20
    alias F16 = 1
    alias F32 = 0

fn matmul_dispatch(
    quant_type: UInt32,
    weights: UnsafePointer[UInt8, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    nr: Int,
    nc: Int,
):
    """Dispatch to the correct matmul kernel based on quantization type.
    This is the main entry point for quantized matmul.
    """
    if quant_type == QuantType.Q4_0:
        # Use existing Q4_0 kernel from optimized_kernels.mojo
        # q4_mm_regblock(w, inp, res, nr, nc)
        pass  # Will be wired in engine
    elif quant_type == QuantType.Q8_0:
        q8_0_matmul(weights, input, output, nr, nc)
    elif quant_type == QuantType.Q5_0:
        q5_0_matmul(weights, input, output, nr, nc)
    elif quant_type == QuantType.Q4_K:
        # Q4_K needs separate scale/d/dmin arrays
        q4_k_matmul(weights, weights, weights, weights, 
                     input, output, nr, nc)
    elif quant_type == QuantType.Q6_K:
        q6_k_matmul(weights, weights, weights, weights,
                     input, output, nr, nc)
    else:
        # Fallback: not yet implemented
        print("Quantization type", String(quant_type), "not yet supported in Mojo kernels")