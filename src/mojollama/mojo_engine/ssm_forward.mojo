# MojoLlama SSM Forward — Pure Mojo Mamba-style SSM layer
# WHAT:  Single-token SSM forward pass for hybrid SSM+Attention models (Qwen3.5-MoE).
# WHY:   Enable Mamba/SSM-based model support in pure Mojo. Replaces ssm_forward.c (389L).
#        All matmuls dispatch through mojo_quant_kernels.mojo quant decoders.
# WHEN:  May 2026 — initial pure Mojo port from C implementation.
from std import math
from std.algorithm.backend.cpu.parallelize import parallelize

# ─── Helper functions ───
def silu(x: Float32) -> Float32:
    """SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))"""
    return x / (1.0 + math.exp(-x))

def softplus(x: Float32) -> Float32:
    """Softplus: log(1 + exp(x)), numerically stable."""
    if x > 20.0: return x
    if x < -20.0: return 0.0
    return math.log(1.0 + math.exp(x))

def rms_norm(x: UnsafePointer[Float32, MutExternalOrigin],
             w: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin],
             n: Int):
    """RMS LayerNorm: o[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)"""
    var ss: Float64 = 0.0
    for i in range(n): ss += Float64(x.load(i) * x.load(i))
    var rms = math.sqrt(Float32(ss / Float64(n)) + 1e-6)
    var inv = 1.0 / rms
    for i in range(n): o.store(i, x.load(i) * w.load(i) * inv)

# ─── Row-dot dispatch: single row of quantized weight × f32 vector ───
# Uses the matmul helpers from mojo_quant_kernels (must be linked or inline)
# For now, implement Q8_0 row-dot directly (most common SSM quant type)

def q8_0_row_dot(w: UnsafePointer[UInt8, MutExternalOrigin],
                 x: UnsafePointer[Float32, MutExternalOrigin],
                 n_cols: Int, row: Int) -> Float32:
    """Q8_0 row × f32 vector dot product for a single output row."""
    var bpr = n_cols // 32
    var total: Float32 = 0.0
    for blk in range(bpr):
        var bo = (row * bpr + blk) * 34
        var d = f16_to_f32(UInt16(w.load(bo)) | (UInt16(w.load(bo+1)) << 8))
        var xo = blk * 32
        comptime W: Int = 8
        var acc = SIMD[DType.float32, W](0.0)
        for j in range(0, 32, W):
            var vals = SIMD[DType.float32, W](0.0)
            for k in range(W):
                var qs = Int8(w.load(bo + 2 + xo + j + k))
                vals[k] = Float32(qs) * d
            var xv = x.load[width=W](xo + j)
            acc = acc + vals * xv
        total += acc.reduce_add()
    return total

def f16_to_f32(h: UInt16) -> Float32:
    var s = UInt32(h >> 15); var e = UInt32((h >> 10) & 0x1F)
    var m = UInt32(h & 0x3FF)
    if e == 0: return 0.0 if m == 0 else Float32(Float64(m) * 5.960464477539063e-8)
    if e == 31: return 0.0
    var r = Float32(m | 0x400); var ei = Int(e) - 25
    if ei >= 0:
        for _ in range(ei): r *= 2.0
    else:
        for _ in range(-ei): r *= 0.5
    return -r if s != 0 else r

# ═══════════════════════════════════════════════════════════════
# SSM Forward — Single-token SSM layer
# ═══════════════════════════════════════════════════════════════
# For each SSM layer:
#   1. QKV projection: qkv = W_qkv @ x_norm  (Q8_0)
#   2. Depthwise conv1d: conv_out = qkv * conv_w
#   3. Split: x_ssm = conv_out[0:inner]
#   4. Gate: gate = W_gate @ x_norm (Q8_0)
#   5. SSM step: discretize A, update state, apply C
#   6. y_ssm = y_ssm * silu(gate)
#   7. Output: out = W_down @ y_ssm (Q8_0)
# ═══════════════════════════════════════════════════════════════

def ssm_forward(
    x_norm: UnsafePointer[Float32, MutExternalOrigin],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    w_qkv: UnsafePointer[UInt8, MutExternalOrigin],
    w_gate: UnsafePointer[UInt8, MutExternalOrigin],
    w_down: UnsafePointer[UInt8, MutExternalOrigin],
    conv_w: UnsafePointer[Float32, MutExternalOrigin],
    ssm_state: UnsafePointer[Float32, MutExternalOrigin],
    ssm_a: UnsafePointer[Float32, MutExternalOrigin],
    ssm_alpha: UnsafePointer[Float32, MutExternalOrigin],
    ssm_beta: UnsafePointer[Float32, MutExternalOrigin],
    ssm_dt_bias: UnsafePointer[Float32, MutExternalOrigin],
    n_embd: Int, inner: Int, qkv_dim: Int,
    groups: Int, state_size: Int, conv_kernel: Int, dt_rank: Int):
    """Pure Mojo SSM forward pass for one layer, batch=1.
    
    All matmuls use Q8_0 quantized weights. SSM state is persistent.
    """
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    
    # Allocate temporary buffers
    var qkv = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(qkv_dim * 4))))
    var gate = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(inner * 4))))
    var x_ssm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(inner * 4))))
    var gate_silu_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(inner * 4))))
    var y_mid = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(inner * 4))))
    
    # ── Step 1: QKV projection ──
    # Parallel row-dot: qkv[r] = q8_0_row_dot(w_qkv, x_norm, n_embd, r)
    def qkv_worker(r: Int) capturing -> None:
        qkv.store(r, q8_0_row_dot(w_qkv, x_norm, n_embd, r))
    parallelize[func=qkv_worker](num_work_items=qkv_dim)
    
    # ── Step 2: Depthwise conv1d (single token: only first kernel element) ──
    for i in range(qkv_dim):
        qkv.store(i, qkv.load(i) * conv_w.load(i))
    
    # ── Step 3: Split — first `inner` dims are SSM input ──
    for i in range(inner):
        x_ssm.store(i, qkv.load(i))
    
    # ── Step 4: Gate projection ──
    def gate_worker(r: Int) capturing -> None:
        gate.store(r, q8_0_row_dot(w_gate, x_norm, n_embd, r))
    parallelize[func=gate_worker](num_work_items=inner)
    
    # ── Step 4.5: SiLU gate ──
    for i in range(inner):
        gate_silu_buf.store(i, silu(gate.load(i)))
    
    # ── Step 5: SSM step ──
    # Project x_norm to dt_rank using ssm_alpha
    # ssm_alpha shape: [groups * state_size, dt_rank]
    var x_dt = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(dt_rank * 4))))
    for d in range(dt_rank):
        var s: Float32 = 0.0
        for j in range(n_embd):
            var g = j % groups
            var s_pos = (j // 16) % state_size
            s += ssm_alpha.load((g * state_size + s_pos) * dt_rank + d) * x_norm.load(j)
        x_dt.store(d, s)
    
    # Compute dt = softplus(dt_bias + x_dt)
    var dt = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(dt_rank * 4))))
    for d in range(dt_rank):
        dt.store(d, softplus(ssm_dt_bias.load(d) + x_dt.load(d)))
    
    # Per-group SSM update
    var per_group_state = state_size
    var per_group_inner = inner // groups
    
    def ssm_worker(g: Int) capturing -> None:
        var state_g = ssm_state + g * per_group_state
        var x_g = x_ssm + g * per_group_inner
        var y_g = y_mid + g * per_group_inner
        
        # Compute B and C for this group
        var B = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(per_group_state * 4))))
        var C = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(per_group_state * 4))))
        var A_disc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(per_group_state * 4))))
        
        for s in range(per_group_state):
            var b_s: Float32 = 0.0
            var c_s: Float32 = 0.0
            for d in range(dt_rank):
                var idx = (g * per_group_state + s) * dt_rank + d
                b_s += ssm_alpha.load(idx) * x_dt.load(d)
                c_s += ssm_beta.load(idx) * x_dt.load(d)
            B.store(s, b_s); C.store(s, c_s)
            var a_idx = s % dt_rank
            A_disc.store(s, math.exp(ssm_a.load(a_idx) * dt.load(a_idx)))
        
        # Compute Bx (input contribution)
        var x_g_sum: Float32 = 0.0
        for j in range(per_group_inner): x_g_sum += x_g.load(j)
        x_g_sum /= Float32(per_group_inner) if per_group_inner > 0 else 1.0
        
        # State update + output
        var y_scalar: Float32 = 0.0
        for s in range(per_group_state):
            var bx = B.load(s) * x_g_sum * dt.load(s % dt_rank)
            var new_state = A_disc.load(s) * state_g.load(s) + bx
            state_g.store(s, new_state)
            y_scalar += C.load(s) * new_state
        
        # Expand to per-group output dims
        for j in range(per_group_inner):
            y_g.store(j, y_scalar * x_g.load(j))
    
    parallelize[func=ssm_worker](num_work_items=groups)
    
    # ── Step 6: Output gating ──
    for i in range(inner):
        y_mid.store(i, y_mid.load(i) * gate_silu_buf.load(i))
    
    # ── Step 7: Output projection ──
    def down_worker(r: Int) capturing -> None:
        out_ptr.store(r, q8_0_row_dot(w_down, y_mid, inner, r))
    parallelize[func=down_worker](num_work_items=n_embd)
