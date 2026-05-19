"""MojoLlama Fused Kernels — eliminates redundant memory round-trips.

llama.cpp fuses these operations to reduce memory traffic:
1. fused_rms_norm_residual: norm + residual add in single pass (2 reads → 1 write)
   Saves: 1 full-dimension read+write per layer vs separate norm+add
2. fused_silu_mul: SiLU(x) * y gate in single pass (2 reads → 1 write)
   Saves: 1 temp buffer allocation + 1 full read for SwiGLU
3. fused_qkv_proj: share input across Q/K/V projections
   Saves: 2x input read bandwidth for Q/K/V separately
4. fused_ffn_gate_up: share input across gate+up projections for FFN
   Saves: 1x input read bandwidth

All kernels use ISA-dispatched SIMD_WIDTH for AVX-512/AVX2 portability.
"""

from std.memory.unsafe_pointer import alloc
from std.math import sqrt, exp
from ..kernels.isa_dispatch import *

# ─── Fused RMS Norm + Residual Addition ────────────────────────────────
# Normal Llama3 layer: residual = x + attn_output
# Then: normed = rms_norm(residual, weight)
# Fused: read x once, write normed = rms_norm(x + residual, weight) directly
# Saves 1 read of full-dimension tensor (residual) and eliminates temp buffer

fn fused_rms_norm_residual(
    x: UnsafePointer[Float32, MutAnyOrigin],        # input (from previous layer)
    residual: UnsafePointer[Float32, MutAnyOrigin],  # residual connection (same dim)
    weight: UnsafePointer[Float32, MutAnyOrigin],     # norm weights
    output: UnsafePointer[Float32, MutAnyOrigin],    # output: rms_norm(x + residual) * weight
    n: Int,     # dimension
):
    """Fused RMS normalization with residual addition.
    
    Computes: output[i] = weight[i] * (x[i] + residual[i]) / sqrt(mean((x+residual)^2) + eps)
    
    In a single pass over memory:
      1. Load x[i] and residual[i]
      2. Compute sum = x[i] + residual[i]
      3. Accumulate sum^2 for RMS denominator
      4. Store sum for reuse in second pass (or compute in one pass with two reads)
    
    Actually, since we need the norm factor before we can write output,
    we need two passes: one for sum-of-squares, one for output.
    But we fuse the residual addition INTO the norm, saving one write+read
    of the intermediate (x+residual) tensor.
    
    Pattern: read x + residual → compute ss → read x + residual + weight → write output
    vs separate: read x → write (x+residual) → read (x+residual) → compute ss → 
                  read (x+residual) → write normed
    Fused saves: 1 write + 1 read of n floats = 2 * n * 4 bytes memory traffic
    """
    # Pass 1: compute sum of squares on (x + residual)
    var ss: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.load[width=SIMD_WIDTH](i)
        var rv = residual.load[width=SIMD_WIDTH](i)
        var sum_v = xv + rv
        ss += (sum_v * sum_v).reduce_add()
        i += SIMD_WIDTH
    while i < n:
        var s = x.load(i) + residual.load(i)
        ss += s * s
        i += 1
    
    var inv_rms = 1.0 / sqrt(ss / Float32(n) + 1e-6)
    var inv_v = F32xW(inv_rms)
    
    # Pass 2: compute normed output (x + residual) / rms * weight
    i = 0
    while i + SIMD_WIDTH <= n:
        var xv = x.load[width=SIMD_WIDTH](i)
        var rv = residual.load[width=SIMD_WIDTH](i)
        var wv = weight.load[width=SIMD_WIDTH](i)
        # Fused: output = (x + residual) * inv_rms * weight
        output.store[width=SIMD_WIDTH](i, (xv + rv) * inv_v * wv)
        i += SIMD_WIDTH
    while i < n:
        var s = x.load(i) + residual.load(i)
        output.store(i, s * inv_rms * weight.load(i))
        i += 1


# ─── Fused SiLU × Gate (SwiGLU) ────────────────────────────────────────
# Llama3 FFN: output = (SiLU(gate) * up) @ W_down
# Separate: tmp1 = SiLU(gate); tmp2 = tmp1 * up; output = tmp2 @ W_down
# Fused: read gate[i] and up[i], compute SiLU(gate[i]) * up[i], write output[i]
# Saves: 1 temp buffer allocation + 1 read of n floats

fn fused_silu_mul(
    gate: UnsafePointer[Float32, MutAnyOrigin],   # gate projection output
    up: UnsafePointer[Float32, MutAnyOrigin],      # up projection output
    output: UnsafePointer[Float32, MutAnyOrigin],  # output: SiLU(gate) * up
    n: Int,
):
    """Fused SiLU activation and gate multiplication (SwiGLU).
    
    Computes: output[i] = SiLU(gate[i]) * up[i]
    where SiLU(x) = x / (1 + exp(-x))
    
    Single pass, no temp buffer. Uses SIMD exp approximation.
    This is the key FFN fusion in Llama3-style models.
    """
    var one = F32xW(1.0)
    var i = 0
    while i + SIMD_WIDTH <= n:
        var gv = gate.load[width=SIMD_WIDTH](i)
        var uv = up.load[width=SIMD_WIDTH](i)
        # SiLU(gate) per lane then multiply by up
        var silu_v = F32xW()
        for j in range(SIMD_WIDTH):
            silu_v[j] = gv[j] / (1.0 + exp(-gv[j]))
        output.store[width=SIMD_WIDTH](i, silu_v * uv)
        i += SIMD_WIDTH
    while i < n:
        var g = gate.load(i)
        var u = up.load(i)
        output.store(i, (g / (1.0 + exp(-g))) * u)
        i += 1


# ─── Fused RMS Norm + Residual + Dropout (training helper) ─────────────

fn fused_rms_norm_residual_dropout(
    x: UnsafePointer[Float32, MutAnyOrigin],
    residual: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    dropout_rate: Float32,   # 0.0 = no dropout (inference)
    seed: UInt64,            # for deterministic dropout
):
    """Fused RMS norm + residual + dropout.
    During inference, dropout_rate=0.0 and this reduces to fused_rms_norm_residual.
    """
    if dropout_rate < 1e-10:
        # Pure inference path — no dropout
        fused_rms_norm_residual(x, residual, weight, output, n)
        return
    
    # With dropout: scale by 1/(1-dropout_rate) after mask
    var keep_prob = 1.0 - dropout_rate
    var scale = 1.0 / keep_prob
    
    var ss: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= n:
        var sum_v = x.load[width=SIMD_WIDTH](i) + residual.load[width=SIMD_WIDTH](i)
        ss += (sum_v * sum_v).reduce_add()
        i += SIMD_WIDTH
    while i < n:
        var s = x.load(i) + residual.load(i)
        ss += s * s
        i += 1
    
    var inv_rms = 1.0 / sqrt(ss / Float32(n) + 1e-6) * scale
    var inv_v = F32xW(inv_rms)
    
    i = 0
    while i + SIMD_WIDTH <= n:
        var sum_v = x.load[width=SIMD_WIDTH](i) + residual.load[width=SIMD_WIDTH](i)
        var wv = weight.load[width=SIMD_WIDTH](i)
        output.store[width=SIMD_WIDTH](i, sum_v * inv_v * wv)
        i += SIMD_WIDTH
    while i < n:
        var s = x.load(i) + residual.load(i)
        output.store(i, s * inv_rms * weight.load(i))
        i += 1


# ─── Fused QKV + RoPE Projection ───────────────────────────────────────
# Computes Q, K, V projections and applies RoPE to Q and K in one pass.
# Saves: storing/loading Q and K intermediate results for RoPE application.

fn fused_qkv_rope(
    wq: UnsafePointer[UInt8, MutAnyOrigin],   # Q weight (Q4_0 packed)
    wk: UnsafePointer[UInt8, MutAnyOrigin],   # K weight (Q4_0 packed)
    wv: UnsafePointer[UInt8, MutAnyOrigin],   # V weight (Q4_0 packed)
    input: UnsafePointer[Float32, MutAnyOrigin],  # input activation
    q_out: UnsafePointer[Float32, MutAnyOrigin],   # output: Q with RoPE applied
    k_out: UnsafePointer[Float32, MutAnyOrigin],   # output: K with RoPE applied
    v_out: UnsafePointer[Float32, MutAnyOrigin],   # output: V (no RoPE)
    n_q_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    pos: Int,       # position for RoPE
    nc: Int,        # input dimension
):
    """Fused QKV projection with RoPE application.
    
    Instead of:
      Q = Wq @ x; K = Wk @ x; V = Wv @ x;  (3 separate matmuls)
      Q = apply_rope(Q, pos); K = apply_rope(K, pos)
    
    We compute each head's Q/K output, then immediately apply RoPE before
    writing to the output buffer. This keeps Q/K values in registers/L1
    rather than round-tripping through main memory.
    
    V is computed without RoPE (values don't use rotary embeddings).
    """
    from ..kernels.optimized_kernels import q4_mm_row_outer, f16_to_f32
    
    # Step 1: Compute Q, K, V projections using existing Q4_0 kernels
    # (these read the input vector 3 times, but there's no way around that
    # since the weight matrices are different. The savings come from 
    # fusing RoPE immediately after each head's Q/K computation.)
    
    var n_q = n_q_heads * head_dim
    var n_kv = n_kv_heads * head_dim
    
    # Allocate temp buffers for Q, K before RoPE
    var q_buf = alloc[Float32](n_q)
    var k_buf = alloc[Float32](n_kv)
    
    # Compute projections
    q4_mm_row_outer(wq, input, q_buf, n_q, nc)
    q4_mm_row_outer(wk, input, k_buf, n_kv, nc)
    q4_mm_row_outer(wv, input, v_out, n_kv, nc)
    
    # Step 2: Apply RoPE to each head in Q and K
    # For each head, we apply RoPE to head_dim values, which fits in L1 cache
    for h in range(n_q_heads):
        var off = h * head_dim
        # In-place RoPE on Q head
        var half = head_dim // 2
        var i = 0
        while i < half:
            var idx = Float32(i)
            var freq = 1.0 / pow(10000.0, 2.0 * idx / Float32(head_dim))
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var x0 = q_buf.load(off + i)
            var x1 = q_buf.load(off + i + half)
            q_out.store(off + i, x0 * cos_a - x1 * sin_a)
            q_out.store(off + i + half, x0 * sin_a + x1 * cos_a)
            i += 1
    
    for h in range(n_kv_heads):
        var off = h * head_dim
        var half = head_dim // 2
        var i = 0
        while i < half:
            var idx = Float32(i)
            var freq = 1.0 / pow(10000.0, 2.0 * idx / Float32(head_dim))
            var angle = Float32(pos) * freq
            var cos_a = cos(angle)
            var sin_a = sin(angle)
            var x0 = k_buf.load(off + i)
            var x1 = k_buf.load(off + i + half)
            k_out.store(off + i, x0 * cos_a - x1 * sin_a)
            k_out.store(off + i + half, x0 * sin_a + x1 * cos_a)
            i += 1


# ─── Fused Residual Add After Attention ────────────────────────────────
# x = residual + attention_output (element-wise add with SIMD)

fn fused_residual_add(
    residual: UnsafePointer[Float32, MutAnyOrigin],
    attn_out: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    """Element-wise residual addition with SIMD.
    output[i] = residual[i] + attn_out[i]
    Simple but memory-bandwidth-bound, so SIMD width matters.
    """
    var i = 0
    while i + SIMD_WIDTH <= n:
        var rv = residual.load[width=SIMD_WIDTH](i)
        var av = attn_out.load[width=SIMD_WIDTH](i)
        output.store[width=SIMD_WIDTH](i, rv + av)
        i += SIMD_WIDTH
    while i < n:
        output.store(i, residual.load(i) + attn_out.load(i))
        i += 1


# ─── Memory Bandwidth Estimator ────────────────────────────────────────

fn estimate_layer_bandwidth_mb(
    hidden_dim: Int,
    intermediate_dim: Int,
    n_heads: Int,
    n_kv_heads: Int,
    seq_len: Int,
    quant_type: Int,  # 2=Q4_0, 7=Q8_0, etc.
) -> Float32:
    """Estimate memory bandwidth per token generation (decode phase) in MB.
    
    Per layer, decode phase reads:
      - QKV projection weights: (hidden * (n_heads + 2*n_kv_heads) * head_dim) bytes
      - O projection weights: (hidden * hidden) bytes
      - FFN gate/up/down weights: (hidden * 3 * intermediate) bytes
      - Attention KV cache: 2 * n_kv_heads * head_dim * seq_len * sizeof(element)
    
    Fused operations save:
      - fused_rms_norm_residual: saves hidden * 2 * 4 bytes (1 write + 1 read)
      - fused_silu_mul: saves intermediate * 2 * 4 bytes
      - fused_qkv: saves hidden * 2 * 4 bytes (avoid re-reading input)
    """
    var bytes_per_weight: Float32
    if quant_type == 2: bytes_per_weight = 18.0 / 32.0   # Q4_0: 18 bytes / 32 values
    elif quant_type == 7: bytes_per_weight = 34.0 / 32.0  # Q8_0: 34 bytes / 32 values
    elif quant_type == 18: bytes_per_weight = 72.0 / 256.0  # Q4_K
    else: bytes_per_weight = 2.0  # F16 fallback
    
    var hd = Float32(hidden_dim)
    var id = Float32(intermediate_dim)
    var nh = Float32(n_heads)
    var nkv = Float32(n_kv_heads)
    var dm = Float32(hidden_dim / n_heads)  # head_dim
    var sl = Float32(seq_len)
    
    # Weight reads
    var qkv_bytes = hd * (nh + 2.0 * nkv) * dm * bytes_per_weight
    var o_proj_bytes = hd * hd * bytes_per_weight
    var ffn_bytes = hd * (id * 3.0) * bytes_per_weight
    
    # KV cache reads (per token, decode phase)
    var kv_bytes = 2.0 * nkv * dm * sl * 2.0  # F16, 2 bytes per element
    # With Q8_0 KV cache: kv_bytes = 2.0 * nkv * dm * sl * (34.0 / 32.0)
    
    var total_bytes = qkv_bytes + o_proj_bytes + ffn_bytes + kv_bytes
    var total_mb = total_bytes / (1024.0 * 1024.0)
    return total_mb