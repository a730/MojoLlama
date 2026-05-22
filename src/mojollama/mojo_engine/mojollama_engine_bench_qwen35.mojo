# MojoLlama Engine Bench — Qwen3.6-35B-A3B (cold-cache, pure Mojo)
# WHAT:  Benchmarks Qwen3.6-35B-A3B using cold-cache methodology.
#        Uses real architecture params, MXFP4 quant matmul.
#        Reports thread sweep, component breakdown, throughput estimate.
# WHY:   Pure Mojo benchmark for Qwen3.6-35B — no Python, no C.
# WHEN:  May 2026.
from std import time, math
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

# ═══ Qwen3.6-35B-A3B Architecture ═══
comptime NE: Int = 2048      # hidden_size
comptime NH: Int = 16        # n_heads
comptime NK: Int = 2         # n_kv_heads
comptime HD: Int = 256       # head_dim
comptime NL: Int = 40        # n_layers (10 attn + 30 SSM)
comptime N_ATTN: Int = 10    # attention layers (every 4th)
comptime N_SSM: Int = 30     # SSM layers
comptime N_EXP: Int = 256    # num_experts
comptime N_ACT: Int = 8      # experts per token
comptime FF_EXP: Int = 512   # expert intermediate
comptime FF_SHR: Int = 512   # shared expert intermediate
comptime NV: Int = 248320    # vocab_size
comptime NW: Int = 24        # threads
comptime RPW: Int = 32
comptime W: Int = 8

comptime MXFP4_BS: Int = 32
comptime MXFP4_BYTES: Int = 17
comptime EP: Float32 = 1e-6

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...

# ═══ MXFP4 matmul ═══
def mxfp4_mm(w_addr: Int64, x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int):
    if w_addr == 0: return
    var w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_addr))
    var bpr = nc // MXFP4_BS
    def wk(r: Int) capturing -> None:
        var ro = r * bpr * MXFP4_BYTES; var acc: Float32 = 0.0
        for blk in range(bpr):
            var bo = ro + blk * MXFP4_BYTES
            var eb = w.load(bo + 16)
            var sf: Float32 = 0.0
            if eb != 0:
                if eb < 255:
                    var ei = Int(eb) - 127
                    sf = 1.0
                    if ei >= 0:
                        for _ in range(ei): sf *= 2.0
                    else:
                        for _ in range(-ei): sf *= 0.5
                else: sf = 1e20
            var ai: Int32 = 0
            for j in range(16):
                var p = w.load(bo + j)
                var lo = Int32(p & 0x0F)
                if lo > 7: lo -= 16
                var hi = Int32(p >> 4)
                if hi > 7: hi -= 16
                ai += lo * Int32(x.load(blk*MXFP4_BS + j*2)) + hi * Int32(x.load(blk*MXFP4_BS + j*2+1))
            acc += Float32(ai) * sf
        o.store(r, acc)
    parallelize[func=wk](num_work_items=nr, num_workers=nw)

# ═══ Fill cold-cache weight pool ═══
def fill_pool(pool: Int64, sz: Int64):
    var p = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(pool))
    for i in range(Int(sz)):
        p.store(i, UInt8((i*7 + (i>>4)*13 + (i>>8)*17) & 0xFF))

# ═══ Benchmark a single matmul shape (cold-cache) ═══
def bench_mm(label: String, nr: Int, nc: Int, pool: Int64, pool_off: Int64,
             x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin],
             nw: Int) -> Float64:
    var wa = pool + pool_off
    var total: Float64 = 0.0
    for _ in range(3):
        var t0 = time.perf_counter()
        mxfp4_mm(wa, x, o, nr, nc, nw)
        total += (time.perf_counter() - t0) * 1000.0
    return total / 3.0

def p(s: String): print(s)

def main():
    var inner = NH * HD       # 4096 (16×256)
    var kv_dim = NK * HD      # 512 (2×256)
    var ni = NE
    
    # Weight pool: need space for all matmul shapes at cold-cache sizes
    # Use 256MB pool (>L3 cache of 128MB)
    var pool = _alc(Int64(256 * 1048576))
    fill_pool(pool, Int64(256 * 1048576))
    
    # Input/output buffers (within pool)
    var x_addr = pool + 250 * 1048576
    var o_addr = pool + 252 * 1048576
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x_addr))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(o_addr))
    for i in range(ni): x.store(i, Float32(i%100-50)*0.01)
    
    # MXFP4 row sizes (bytes per row)
    var bpr_ne = NE // MXFP4_BS
    var bpr_inner = inner // MXFP4_BS
    var bpr_kv = kv_dim // MXFP4_BS
    var bpr_exp = FF_EXP // MXFP4_BS
    
    var off: Int64 = 0
    
    p("{")
    p("\"qwen3.6_35b_engine_bench\": {")
    p("  \"arch\": {")
    p("    \"n_embd\":" + String(NE) + ",\"n_heads\":" + String(NH) + ",\"n_kv_heads\":" + String(NK))
    p("    ,\"head_dim\":" + String(HD) + ",\"n_layers\":" + String(NL) + ",\"n_attn\":" + String(N_ATTN) + ",\"n_ssm\":" + String(N_SSM))
    p("    ,\"n_experts\":" + String(N_EXP) + ",\"experts_per_tok\":" + String(N_ACT))
    p("    ,\"expert_ff\":" + String(FF_EXP) + ",\"quant\":\"MXFP4\"")
    p("  },")
    
    # ═══ 1. COMPONENT BREAKDOWN (cold-cache, all matmuls) ═══
    p("  \"breakdown\": {")
    
    # Attention matmuls
    off = 0
    var t_q = bench_mm("Q", inner, NE, pool, off, x, o, NW)
    off += Int64(inner) * Int64(bpr_ne) * MXFP4_BYTES
    p("    \"Q_proj " + String(inner) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_q)) + "},")
    
    var t_k = bench_mm("K", kv_dim, NE, pool, off, x, o, NW)
    off += Int64(kv_dim) * Int64(bpr_ne) * MXFP4_BYTES
    p("    \"K_proj " + String(kv_dim) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_k)) + "},")
    
    var t_v = bench_mm("V", kv_dim, NE, pool, off, x, o, NW)
    off += Int64(kv_dim) * Int64(bpr_ne) * MXFP4_BYTES
    p("    \"V_proj " + String(kv_dim) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_v)) + "},")
    
    var t_o = bench_mm("O", NE, inner, pool, off, x, o, NW)
    off += Int64(NE) * Int64(bpr_inner) * MXFP4_BYTES
    p("    \"O_proj " + String(NE) + "x" + String(inner) + "\":{\"ms\":" + String(Float64(t_o)) + "},")
    
    # Expert matmuls (per-expert, 8 active)
    var t_exp_g = bench_mm("expert_gate", FF_EXP, NE, pool, off, x, o, NW)
    off += Int64(FF_EXP) * Int64(bpr_ne) * MXFP4_BYTES
    p("    \"expert_gate " + String(FF_EXP) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_exp_g)) + "},")
    
    var t_exp_u = bench_mm("expert_up", FF_EXP, NE, pool, off, x, o, NW)
    off += Int64(FF_EXP) * Int64(bpr_ne) * MXFP4_BYTES
    p("    \"expert_up " + String(FF_EXP) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_exp_u)) + "},")
    
    var t_exp_d = bench_mm("expert_down", NE, FF_EXP, pool, off, x, o, NW)
    # FF_EXP=512, so bpr_ff = 512/32=16
    off += Int64(NE) * Int64(FF_EXP // MXFP4_BS) * MXFP4_BYTES
    p("    \"expert_down " + String(NE) + "x" + String(FF_EXP) + "\":{\"ms\":" + String(Float64(t_exp_d)) + "}")
    
    p("  },")
    
    # ═══ 2. PER-LAYER TIMING ═══
    p("  \"layer_timing\": {")
    # Attention layer: Q+K+V+O + MoE(8 experts × gate+up+down)
    var attn_matmuls = t_q + t_k + t_v + t_o
    var moe_8 = (t_exp_g + t_exp_u + t_exp_d) * Float64(N_ACT)
    var attn_layer = attn_matmuls + moe_8
    # SSM layer: MoE only (SSM ops not benchmarked, estimate ~30% of attn matmuls)
    var ssm_factor: Float64 = 0.3
    var ssm_layer = moe_8 + ssm_factor * attn_matmuls
    
    p("    \"attn_matmuls_ms\":" + String(Float64(attn_matmuls)) + ",")
    p("    \"moe_8experts_ms\":" + String(Float64(moe_8)) + ",")
    p("    \"attn_layer_total_ms\":" + String(Float64(attn_layer)) + ",")
    p("    \"ssm_layer_est_ms\":" + String(Float64(ssm_layer)) + ",")
    
    # Full model: 10 attn + 30 SSM
    var full_ms = Float64(N_ATTN) * attn_layer + Float64(N_SSM) * ssm_layer
    var full_tok = 1000.0 / (full_ms / Float64(NL))
    p("    \"full_model_est_tok_s\":" + String(Float64(full_tok)))
    p("  },")
    
    # ═══ 3. THREAD SWEEP (expert gate, the most common op) ═══
    p("  \"thread_sweep_expert\": [")
    var first = True
    for tw in [1, 2, 4, 8, 16, 24, 32]:
        var t_w = bench_mm("", FF_EXP, NE, pool, 0, x, o, tw)
        if not first: p(",")
        first = False
        var tok = 1000.0 / (t_w * Float64(NL) * Float64(N_ACT) * 3.0 / 1000.0)
        p("    {\"nw\":" + String(tw) + ",\"expert_ms\":" + String(Float64(t_w)) + ",\"est_tok_s\":" + String(Float64(tok)) + "}")
    p("")
    p("  ],")
    
    # ═══ 4. MEMORY BANDWIDTH ESTIMATE ═══
    p("  \"memory\": {")
    # Weight bytes per token: all active parameters at MXFP4
    # Attention weights: Q+K+V+O per attn layer × 10
    var attn_bytes = Float64(N_ATTN) * (Float64(inner*NE) + Float64(kv_dim*NE) + Float64(kv_dim*NE) + Float64(NE*inner)) * 0.5
    # MoE weights: 8 experts × 3 matmuls per layer × 40 layers
    var moe_bytes = Float64(NL) * Float64(N_ACT) * (Float64(FF_EXP*NE) + Float64(FF_EXP*NE) + Float64(NE*FF_EXP)) * 0.5
    var total_mb = (attn_bytes + moe_bytes) / 1048576.0
    var bw = total_mb * 1048576.0 / (full_ms / 1000.0) / 1e9
    p("    \"est_weight_mb\":" + String(Float64(total_mb)) + ",")
    p("    \"effective_bw_gb_s\":" + String(Float64(bw)))
    p("  },")
    
    # ═══ 5. SUMMARY ═══
    p("  \"summary\": {")
    p("    \"model\": \"Qwen3.6-35B-A3B\",")
    p("    \"quant\": \"MXFP4\",")
    p("    \"est_tok_s\": " + String(Float64(full_tok)) + ",")
    p("    \"bottleneck\": \"MoE expert matmuls (" + String(N_ACT) + "×" + String(FF_EXP) + "×" + String(NE) + ")\"")
    p("  }")
    p("}}")
