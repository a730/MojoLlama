# MojoLlama Bench — Qwen3.6-35B-A3B MXFP4 Benchmark
# WHAT:  Benchmarks Qwen3.6-35B-A3B (MoE) with MXFP4 quantized matmul.
#        Architecture: 256 experts×8 active/token, 40 hybrid layers,
#        head_dim=256, shared expert, attention every 4th layer.
# WHY:   Measure MXFP4 throughput for MoE models on Threadripper.
# WHEN:  May 2026 — Qwen3.6-35B-A3B target.
from std import time, math
from std.algorithm.backend.cpu.parallelize import parallelize

comptime MXFP4_BS: Int = 32
comptime MXFP4_BYTES: Int = 17  # 16 nibbles + 1 e8m0 scale

# ═══ MXFP4 matmul kernel ═══
def mxfp4_row(r: Int, w: UnsafePointer[UInt8, MutExternalOrigin],
              x: UnsafePointer[Float32, MutExternalOrigin],
              o: UnsafePointer[Float32, MutExternalOrigin],
              nr: Int, nc: Int) capturing -> None:
    var bpr = nc // MXFP4_BS
    var ro = r * bpr * MXFP4_BYTES
    var acc: Float32 = 0.0
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
                if sf > 1e20: sf = 1e20
            else: sf = 1e20
        var ai: Int32 = 0
        for j in range(16):
            var p = w.load(bo + j)
            var lo = Int32(p & 0x0F)
            if lo > 7: lo -= 16
            var hi = Int32(p >> 4)
            if hi > 7: hi -= 16
            ai += lo * Int32(x.load(blk * MXFP4_BS + j * 2)) + \
                  hi * Int32(x.load(blk * MXFP4_BS + j * 2 + 1))
        acc += Float32(ai) * sf
    o.store(r, acc)

def mxfp4_mm(w: UnsafePointer[UInt8, MutExternalOrigin],
             x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin],
             nr: Int, nc: Int, nw: Int) -> Float64:
    var t0 = time.perf_counter()
    def wk(r: Int) capturing -> None:
        mxfp4_row(r, w, x, o, nr, nc)
    parallelize[func=wk](num_work_items=nr, num_workers=nw)
    return (time.perf_counter() - t0) * 1000.0

# ═══ Stats ═══
def bench_mean(w: UnsafePointer[UInt8, MutExternalOrigin],
               x: UnsafePointer[Float32, MutExternalOrigin],
               o: UnsafePointer[Float32, MutExternalOrigin],
               nr: Int, nc: Int, nw: Int,
               buf: UnsafePointer[Float64, MutExternalOrigin],
               n_warm: Int, n_iter: Int) -> Float64:
    for _ in range(n_warm):
        var _ = mxfp4_mm(w, x, o, nr, nc, nw)
    var total: Float64 = 0.0
    for mi in range(n_iter):
        var t = mxfp4_mm(w, x, o, nr, nc, nw)
        buf.store(mi, t)
        total += t
    return total / Float64(n_iter)

def fill_mxfp4(w: UnsafePointer[UInt8, MutExternalOrigin], nr: Int, nc: Int):
    var bpr = nc // MXFP4_BS
    var rb = bpr * MXFP4_BYTES
    for r in range(nr):
        var ro = r * rb
        for blk in range(bpr):
            var bo = ro + blk * MXFP4_BYTES
            for j in range(16):
                var v = UInt8(((r*7 + j*3 + blk*11) & 0xF) | (((r*13 + j*5 + blk*17) & 0xF) << 4))
                w.store(bo + j, v)
            w.store(bo + 16, UInt8(120 + (r ^ blk) % 20))

def fill_f32(buf: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    for i in range(n): buf.store(i, Float32(Float64(i % 100 - 50) * 0.01))

# Print a line. Helper to reduce boilerplate.
def p(s: String): print(s)

def main():
    # ═══ Qwen3.6-35B-A3B Architecture ═══
    var NE: Int = 2048      # hidden_size
    var NH: Int = 16        # num_attention_heads
    var NK: Int = 2         # num_key_value_heads
    var HD: Int = 256       # head_dim (NOT 128!)
    var NL: Int = 40        # num_hidden_layers
    var N_EXP: Int = 256    # num_experts
    var N_ACT: Int = 8      # experts per token
    var FF_EXP: Int = 512   # moe_intermediate_size per expert
    var FF_SHR: Int = 512   # shared_expert_intermediate_size
    var NV: Int = 248320    # vocab_size
    var NW: Int = 32        # threads
    var n_warm: Int = 2
    var n_iter: Int = 10
    
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    
    var inner = NH * HD        # 4096 (16×256)
    var kv_dim = NK * HD       # 512 (2×256)
    
    # For attention layers: Q=NE×inner, K=kv_dim×NE, V=kv_dim×NE, O=inner×NE
    # For MoE experts: gate=FF_EXP×NE, up=FF_EXP×NE, down=NE×FF_EXP
    # For shared expert: same shapes
    # Expert matmuls: 8 active experts per layer
    
    var max_wr = max(NE, max(inner, max(kv_dim, FF_EXP)))
    var max_wc = max(max(NE, inner), max(kv_dim, FF_EXP))
    
    var dense_bpr = max_wc // MXFP4_BS
    var dense_weight_bytes = max_wr * dense_bpr * MXFP4_BYTES
    var expert_weight_bytes = FF_EXP * (max_wc // MXFP4_BS) * MXFP4_BYTES  # for one expert
    
    # Allocate buffers
    var buf = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_iter * 8))))
    var b = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(dense_weight_bytes))))
    var b_exp_gate = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(expert_weight_bytes))))
    var b_exp_down = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(FF_EXP * (FF_EXP//32) * 17))))  # NE×FF_EXP
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_wc * 4))))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max(max_wr, FF_EXP) * 4))))
    
    fill_mxfp4(b, max_wr, NE)
    fill_mxfp4(b_exp_gate, FF_EXP, NE)
    fill_mxfp4(b_exp_down, NE, FF_EXP)
    fill_f32(x, max_wc)
    
    p("{")
    p("\"qwen3.6_35b_mxfp4\": {")
    p("  \"arch\": {")
    p("    \"hidden_size\": " + String(NE) + ",\"n_heads\": " + String(NH) + ",\"n_kv_heads\": " + String(NK) + ",")
    p("    \"head_dim\": " + String(HD) + ",\"n_layers\": " + String(NL) + ",")
    p("    \"num_experts\": " + String(N_EXP) + ",\"experts_per_tok\": " + String(N_ACT) + ",")
    p("    \"expert_intermediate\": " + String(FF_EXP) + ",\"shared_intermediate\": " + String(FF_SHR) + ",")
    p("    \"vocab_size\": " + String(NV) + ",\"threads\": " + String(NW) + ",\"quant\": \"MXFP4\"")
    p("  },")
    
    # ═══ 1. COMPONENT BREAKDOWN (MXFP4) ═══
    p("  \"breakdown\": {")
    
    # Attention matmuls (dense)
    var t = bench_mean(b, x, o, inner, NE, NW, buf, n_warm, n_iter)
    var ops = Int64(inner) * Int64(NE) * 2
    p("    \"Q_proj " + String(inner) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t)) + ",\"gflops\":" + String(Float64(Float64(ops)/t/1e6)) + "},")
    
    var tk = bench_mean(b, x, o, kv_dim, NE, NW, buf, n_warm, n_iter)
    ops = Int64(kv_dim) * Int64(NE) * 2
    p("    \"K_proj " + String(kv_dim) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(tk)) + ",\"gflops\":" + String(Float64(Float64(ops)/tk/1e6)) + "},")
    
    var tv = bench_mean(b, x, o, kv_dim, NE, NW, buf, n_warm, n_iter)
    p("    \"V_proj " + String(kv_dim) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(tv)) + ",\"gflops\":" + String(Float64(Float64(ops)/tv/1e6)) + "},")
    
    var to = bench_mean(b, x, o, NE, inner, NW, buf, n_warm, n_iter)
    ops = Int64(NE) * Int64(inner) * 2
    p("    \"O_proj " + String(NE) + "x" + String(inner) + "\":{\"ms\":" + String(Float64(to)) + ",\"gflops\":" + String(Float64(Float64(ops)/to/1e6)) + "},")
    
    # Per-expert matmul (gate/up: FF_EXP × NE)
    var t_exp_g = bench_mean(b_exp_gate, x, o, FF_EXP, NE, NW, buf, n_warm, n_iter)
    ops = Int64(FF_EXP) * Int64(NE) * 2
    p("    \"expert_gate " + String(FF_EXP) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_exp_g)) + ",\"gflops\":" + String(Float64(Float64(ops)/t_exp_g/1e6)) + "},")
    
    var t_exp_u = bench_mean(b_exp_gate, x, o, FF_EXP, NE, NW, buf, n_warm, n_iter)
    p("    \"expert_up " + String(FF_EXP) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_exp_u)) + ",\"gflops\":" + String(Float64(Float64(ops)/t_exp_u/1e6)) + "},")
    
    var t_exp_d = bench_mean(b_exp_down, x, o, NE, FF_EXP, NW, buf, n_warm, n_iter)
    ops = Int64(NE) * Int64(FF_EXP) * 2
    p("    \"expert_down " + String(NE) + "x" + String(FF_EXP) + "\":{\"ms\":" + String(Float64(t_exp_d)) + ",\"gflops\":" + String(Float64(Float64(ops)/t_exp_d/1e6)) + "}")
    
    p("  },")
    
    # ═══ 2. MOE EXPERT DECODE (8 active experts) ═══
    p("  \"moe_decode\": {")
    # 8 experts × (gate + up + down) per layer
    var moe_experts = bench_mean(b_exp_gate, x, o, FF_EXP, NE, NW, buf, n_warm, n_iter)
    var moe_time = moe_experts * 3.0 * Float64(N_ACT)  # gate+up+down for 8 experts
    p("    \"per_expert_ms\":" + String(Float64(moe_experts)) + ",")
    p("    \"expert_count_active\":" + String(N_ACT) + ",")
    p("    \"total_moe_ms_per_layer\":" + String(Float64(moe_time)) + ",")
    p("    \"moe_gflops\":" + String(Float64(Float64(N_ACT * FF_EXP * NE * 6) / moe_time / 1e6)))
    p("  },")
    
    # ═══ 3. ATTENTION LAYER DECODE ═══
    p("  \"attention_decode\": {")
    var attn_time = t + tk + tv + to  # Q+K+V+O matmuls only (not softmax, scores×V)
    p("    \"attn_matmuls_ms\":" + String(Float64(attn_time)) + ",")
    p("    \"full_attention_layers\": 10")  # every 4th of 40
    p("  },")
    
    # ═══ 4. FULL LAYER TIME (weighted average) ═══
    # 40 layers: 10 attention layers, 30 SSM layers
    # Each layer has MoE FFN (8 experts × 3 matmuls)
    # Attention layers also have QKV+O matmuls
    # SSM layers have Mamba operations (not benchmarked here)
    p("  \"layer_estimate\": {")
    var moe_full = moe_experts * 3.0 * Float64(N_ACT)  # 8 experts × 3 matmuls
    var attn_layer = attn_time  # attention matmuls only
    var ssm_layer = 0.0  # SSM layers not benchmarked (different ops)
    
    # Weighted: 10 attn layers + 30 ssm layers, each with MoE
    var total_ms_est = Float64(NL) * moe_full + Float64(10) * attn_layer
    var tok_s_est = 1000.0 / (moe_full + attn_layer)  # worst case: attn layer
    p("    \"moe_per_layer_ms\":" + String(Float64(moe_full)) + ",")
    p("    \"attn_matmul_per_layer_ms\":" + String(Float64(attn_layer)) + ",")
    p("    \"attn_layer_decode_tok_s\":" + String(Float64(tok_s_est)) + ",")
    p("    \"note\": \"SSM/Mamba layers not included (need conv1d+SSM kernels)\"")
    p("  },")
    
    # ═══ 5. THREAD SWEEP (per expert) ═══
    p("  \"thread_sweep_expert\": [")
    var tw_first = True
    for tw in [1, 2, 4, 8, 16, 24, 32]:
        var t_w = bench_mean(b_exp_gate, x, o, FF_EXP, NE, tw, buf, n_warm, n_iter // 2)
        if not tw_first: p(",")
        tw_first = False
        var tok = 1000.0 / t_w
        p("    {\"threads\":" + String(tw) + ",\"expert_ms\":" + String(Float64(t_w)) + ",\"tok_s\":" + String(Float64(tok)) + "}")
    p("")
    p("  ],")
    
    # ═══ 6. FULL MODEL THROUGHPUT ESTIMATE ═══
    p("  \"throughput_estimate\": {")
    # Simplified model: 40 layers, each with MoE FFN (8 experts)
    # 10 attention layers + 30 SSM layers
    # Per attention layer: QKV+O matmuls + MoE = attn_time + moe_full
    # Per SSM layer: MoE only (SSM ≈ 2-3× faster than attention matmuls)
    var ssm_factor: Float64 = 0.3  # guess: SSM layer is 30% of attention matmul time
    var ssm_layer_ms = ssm_factor * attn_layer
    var total_full_ms = Float64(10) * (attn_layer + moe_full) + Float64(30) * (ssm_layer_ms + moe_full)
    var full_tok_s = 1000.0 / (total_full_ms / Float64(NL))
    p("    \"full_model_est_tok_s\":" + String(Float64(full_tok_s)) + ",")
    p("    \"attn_only_decode_tok_s\":" + String(Float64(1000.0 / (attn_layer + moe_full))) + ",")
    p("    \"moe_only_decode_tok_s\":" + String(Float64(1000.0 / moe_full)) + ",")
    p("    \"bottleneck\": \"MoE expert matmuls (" + String(N_ACT) + "×" + String(FF_EXP) + "×" + String(NE) + ")\"")
    p("  },")
    
    # ═══ 7. SUMMARY ═══
    p("  \"summary\": {")
    p("    \"model\": \"Qwen3.6-35B-A3B (MXFP4)\",")
    p("    \"active_params\": \"~3B\",")
    p("    \"total_params\": \"~35B\",")
    p("    \"expert_matmul_gflops\": " + String(Float64(Float64(FF_EXP * NE * 2) / t_exp_g / 1e6)) + ",")
    p("    \"attn_matmul_gflops\": " + String(Float64(Float64(NE * NE * 2) / t / 1e6)) + ",")
    p("    \"key_insight\": \"MoE expert matmuls dominate. 8×512×2048 per layer.\"")
    p("  }")
    p("}}")
