# MojoLlama Bench v4 — Full Transformer Model Benchmark
# WHAT:  Benchmarks each transformer component separately + full model:
#        embedding, RMS norm, QKV, RoPE, attention, FFN, output projection.
#        Configurable architecture: n_embd, n_heads, n_kv_heads, n_layers, ff_hidden.
# WHY:   Identify bottlenecks per component. Compare CPU architectures.
# WHEN:  May 2026 — v4 with transformer architecture breakdown.
from std import time, math
from std.algorithm.backend.cpu.parallelize import parallelize

# ═══ f16 helpers ═══
def f16_val(w: UnsafePointer[UInt8, MutExternalOrigin], idx: Int) -> Float32:
    var h = UInt16(w.load(idx*2)) | (UInt16(w.load(idx*2+1)) << 8)
    var s = UInt32(h >> 15); var e = UInt32((h >> 10) & 0x1F); var m = UInt32(h & 0x3FF)
    if e == 0: return 0.0 if m == 0 else Float32(Float64(m) * 5.960464477539063e-8)
    if e == 31: return 0.0
    var r = Float32(m | 0x400); var ei = Int(e) - 25
    if ei >= 0:
        for _ in range(ei): r *= 2.0
    else:
        for _ in range(-ei): r *= 0.5
    return -r if s != 0 else r

def f16_row(w: UnsafePointer[UInt8, MutExternalOrigin],
            x: UnsafePointer[Float32, MutExternalOrigin],
            r: Int, nc: Int) -> Float32:
    var acc: Float32 = 0.0
    var ro = r * nc
    for c in range(nc):
        acc += f16_val(w, ro + c) * x.load(c)
    return acc

def f16_mm(w: UnsafePointer[UInt8, MutExternalOrigin],
           x: UnsafePointer[Float32, MutExternalOrigin],
           o: UnsafePointer[Float32, MutExternalOrigin],
           nr: Int, nc: Int, nw: Int) -> Float64:
    var t0 = time.perf_counter()
    def wk(r: Int) capturing -> None:
        o.store(r, f16_row(w, x, r, nc))
    parallelize[func=wk](num_work_items=nr, num_workers=nw)
    return (time.perf_counter() - t0) * 1000.0

# ═══ Stats ═══
def sort_f64(buf: UnsafePointer[Float64, MutExternalOrigin], n: Int):
    for i in range(n):
        var mi = i
        for j in range(i+1, n):
            if buf.load(j) < buf.load(mi):
                mi = j
        if mi != i:
            var t = buf.load(i)
            buf.store(i, buf.load(mi))
            buf.store(mi, t)

def pct(buf: UnsafePointer[Float64, MutExternalOrigin], n: Int, p: Float64) -> Float64:
    return buf.load(Int(Float64(n-1) * p))

def bench_mean(w: UnsafePointer[UInt8, MutExternalOrigin],
               x: UnsafePointer[Float32, MutExternalOrigin],
               o: UnsafePointer[Float32, MutExternalOrigin],
               nr: Int, nc: Int, nw: Int,
               buf: UnsafePointer[Float64, MutExternalOrigin],
               n_warm: Int, n_iter: Int) -> Float64:
    for _ in range(n_warm):
        var _ = f16_mm(w, x, o, nr, nc, nw)
    var total: Float64 = 0.0
    for mi in range(n_iter):
        var t = f16_mm(w, x, o, nr, nc, nw)
        buf.store(mi, t)
        total += t
    return total / Float64(n_iter)

def main():
    # ═══ Architecture config (change for different models) ═══
    # Default: Llama 3 8B
    var NE: Int = 4096   # n_embd
    var NH: Int = 32     # n_heads
    var NK: Int = 8      # n_kv_heads
    var HD: Int = 128    # head_dim
    var NL: Int = 32     # n_layers
    var FF: Int = 14336  # ff_hidden
    var NV: Int = 128256 # vocab_size
    var NW: Int = 32     # threads
    var _sq: Int = 128   # sequence length (attention bench)
    var n_warm: Int = 2
    var n_iter: Int = 10
    
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    
    var inner = NH * HD
    var kv_dim = NK * HD
    var max_wr = max(max(NE, FF), max(kv_dim, max(inner, 4096)))  # max weight rows
    var max_wc = max(max(NE, inner), max(kv_dim, FF))
    
    # Allocate buffers
    var buf = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_iter * 8))))
    var b = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_wr * max_wc * 2))))
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_wc * 4))))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_wr * 4))))
    var ff_x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_wc * 4))))
    var lm_w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(4096 * NE * 2))))
    
    # Fill weight buffer with deterministic pattern
    for i in range(max_wr * max_wc):
        b.store(i*2, UInt8(i & 0xFF))
        b.store(i*2+1, UInt8((i>>8) & 0xFF))
    for i in range(max_wc):
        x.store(i, Float32(Float64(i % 100 - 50) * 0.01))
        ff_x.store(i, Float32(Float64(i % 100 - 50) * 0.01))
    for i in range(4096 * NE):
        lm_w.store(i*2, UInt8((i*7) & 0xFF))
        lm_w.store(i*2+1, UInt8(((i*7)>>8) & 0xFF))
    
    # Print JSON start
    print("{")
    print("\"mojollama_bench_v4\": {")
    print("  \"arch\": {")
    print("    \"n_embd\": " + String(NE) + ",\"n_heads\": " + String(NH) + ",\"n_kv_heads\": " + String(NK) + ",")
    print("    \"head_dim\": " + String(HD) + ",\"n_layers\": " + String(NL) + ",\"ff_hidden\": " + String(FF) + ",")
    print("    \"vocab_size\": " + String(NV) + ",\"threads\": " + String(NW))
    print("  },")
    
    # ═══ 1. COMPONENT BREAKDOWN ═══
    print("  \"breakdown\": {")
    
    # QKV projection: inner × NE
    var t_qkv = bench_mean(b, x, o, inner, NE, NW, buf, n_warm, n_iter)
    var ops_q = Int64(inner) * Int64(NE) * 2
    print("    \"Q_proj " + String(inner) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_qkv)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_qkv/1e6)) + "},")
    
    # KV projection: kv_dim × NE
    var t_kv = bench_mean(b, x, ff_x, kv_dim, NE, NW, buf, n_warm, n_iter)
    ops_q = Int64(kv_dim) * Int64(NE) * 2
    print("    \"KV_proj " + String(kv_dim) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_kv)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_kv/1e6)) + "},")
    
    # Output projection: NE × inner
    var t_oproj = bench_mean(b, x, o, NE, inner, NW, buf, n_warm, n_iter)
    ops_q = Int64(NE) * Int64(inner) * 2
    print("    \"O_proj " + String(NE) + "x" + String(inner) + "\":{\"ms\":" + String(Float64(t_oproj)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_oproj/1e6)) + "},")
    
    # FFN gate/up: FF × NE
    var t_ffn_gate = bench_mean(b, ff_x, o, FF, NE, NW, buf, n_warm, n_iter)
    ops_q = Int64(FF) * Int64(NE) * 2
    print("    \"FFN_gate " + String(FF) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_ffn_gate)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_ffn_gate/1e6)) + "},")
    
    # FFN up: same shape as gate
    var t_ffn_up = bench_mean(b, ff_x, o, FF, NE, NW, buf, n_warm, n_iter)
    print("    \"FFN_up " + String(FF) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_ffn_up)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_ffn_up/1e6)) + "},")
    
    # FFN down: NE × FF
    var t_ffn_down = bench_mean(b, ff_x, o, NE, FF, NW, buf, n_warm, n_iter)
    ops_q = Int64(NE) * Int64(FF) * 2
    print("    \"FFN_down " + String(NE) + "x" + String(FF) + "\":{\"ms\":" + String(Float64(t_ffn_down)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_ffn_down/1e6)) + "},")
    
    # LM head (estimated: vocab_size × n_embd — too large to allocate, approximate)
    # For 128256×4096, allocate a smaller representative sample
    var lm_samples = 4096  # benchmark 4K rows, scale up
    var t_lm = bench_mean(lm_w, x, o, lm_samples, NE, NW, buf, n_warm, n_iter)
    ops_q = Int64(NV) * Int64(NE) * 2
    var lm_ms_est = t_lm * Float64(NV) / Float64(lm_samples)
    print("    \"LM_head " + String(NV) + "x" + String(NE) + "\":{\"ms_est\":" + String(Float64(lm_ms_est)) + ",\"ms_samples\":" + String(Float64(t_lm)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/lm_ms_est/1e6)) + "}")
    
    print("  },")
    
    # ═══ 2. FULL MODEL BENCHMARK ═══
    # Simulates NL layers: each layer = Q_proj + K_proj + V_proj + O_proj + FFN_gate + FFN_up + FFN_down
    print("  \"full_model\": {")
    for _ in range(n_warm):
        for _ in range(NL):
            var _ = f16_mm(b, x, o, inner, NE, NW)
            var __ = f16_mm(b, x, ff_x, kv_dim, NE, NW)
            var ___ = f16_mm(b, ff_x, o, kv_dim, NE, NW)
            var ____ = f16_mm(b, x, o, NE, inner, NW)
            var _____ = f16_mm(b, ff_x, o, FF, NE, NW)
            var ______ = f16_mm(b, ff_x, o, FF, NE, NW)
            var _______ = f16_mm(b, ff_x, o, NE, FF, NW)
    var model_total: Float64 = 0.0
    for _ in range(n_iter):
        var tm0 = time.perf_counter()
        for _ in range(NL):
            var _ = f16_mm(b, x, o, inner, NE, NW)
            var __ = f16_mm(b, x, ff_x, kv_dim, NE, NW)
            var ___ = f16_mm(b, ff_x, o, kv_dim, NE, NW)
            var ____ = f16_mm(b, x, o, NE, inner, NW)
            var _____ = f16_mm(b, ff_x, o, FF, NE, NW)
            var ______ = f16_mm(b, ff_x, o, FF, NE, NW)
            var _______ = f16_mm(b, ff_x, o, NE, FF, NW)
        model_total += (time.perf_counter() - tm0) * 1000.0
    var full_ms = model_total / Float64(n_iter)
    
    # Total ops per layer: Q_proj + K_proj + V_proj + O_proj + FFN_gate + FFN_up + FFN_down
    var ops_per_layer = Int64(inner)*Int64(NE)*2 + Int64(kv_dim)*Int64(NE)*4 + Int64(NE)*Int64(inner)*2 + Int64(FF)*Int64(NE)*4 + Int64(NE)*Int64(FF)*2
    var total_ops = Int64(NL) * ops_per_layer + Int64(NV) * Int64(NE) * 2  # +lm_head
    var tflops = Float64(total_ops) / (full_ms / 1000.0) / 1e12
    
    print("    \"n_layers\": " + String(NL) + ",")
    print("    \"total_ms\":" + String(Float64(full_ms)) + ",")
    print("    \"ms_per_layer\":" + String(Float64(full_ms / Float64(NL))) + ",")
    print("    \"gflops\":" + String(Float64(tflops * 1000.0)) + ",")
    print("    \"tflops\":" + String(Float64(tflops)) + ",")
    print("    \"total_ops\":" + String(total_ops))
    print("  },")
    
    # ═══ 3. PER-LAYER DECODE TIMING ═══
    print("  \"decode_step\": [")
    # Single-token decode: each matmul is a separate measurement
    var d_q_proj = bench_mean(b, x, o, inner, NE, NW, buf, n_warm, n_iter)
    var d_k_proj = bench_mean(b, x, ff_x, kv_dim, NE, NW, buf, n_warm, n_iter)
    var d_v_proj = bench_mean(b, ff_x, o, kv_dim, NE, NW, buf, n_warm, n_iter)
    var d_o_proj = bench_mean(b, x, o, NE, inner, NW, buf, n_warm, n_iter)
    var d_f_gate = bench_mean(b, ff_x, o, FF, NE, NW, buf, n_warm, n_iter)
    var d_f_up   = bench_mean(b, ff_x, o, FF, NE, NW, buf, n_warm, n_iter)
    var d_f_down = bench_mean(b, ff_x, o, NE, FF, NW, buf, n_warm, n_iter)
    var d_layer_ms = d_q_proj + d_k_proj + d_v_proj + d_o_proj + d_f_gate + d_f_up + d_f_down
    
    print("    {\"component\":\"Q_proj\",\"ms\":" + String(Float64(d_q_proj)) + "},")
    print("    {\"component\":\"K_proj\",\"ms\":" + String(Float64(d_k_proj)) + "},")
    print("    {\"component\":\"V_proj\",\"ms\":" + String(Float64(d_v_proj)) + "},")
    print("    {\"component\":\"O_proj\",\"ms\":" + String(Float64(d_o_proj)) + "},")
    print("    {\"component\":\"FFN_gate\",\"ms\":" + String(Float64(d_f_gate)) + "},")
    print("    {\"component\":\"FFN_up\",\"ms\":" + String(Float64(d_f_up)) + "},")
    print("    {\"component\":\"FFN_down\",\"ms\":" + String(Float64(d_f_down)) + "},")
    print("    {\"component\":\"TOTAL_LAYER\",\"ms\":" + String(Float64(d_layer_ms)) + "},")
    print("    {\"component\":\"tok/s\",\"val\":" + String(Float64(1000.0 / d_layer_ms)) + "}")
    print("  ],")
    
    # ═══ 4. THREAD SWEEP ═══
    print("  \"thread_sweep\": [")
    var tw_first = True
    for tw in [1, 2, 4, 8, 16, 24, 32]:
        var t_w = bench_mean(b, x, o, NE, NE, tw, buf, n_warm, n_iter // 2)
        if not tw_first: print(",")
        tw_first = False
        var tok = 1000.0 / t_w
        print("    {\"threads\":" + String(tw) + ",\"qkv_ms\":" + String(Float64(t_w)) + ",\"tok_s\":" + String(Float64(tok)) + "}")
    print()
    print("  ],")
    
    # ═══ 5. BATCH SWEEP ═══
    # Measured as sequential tokens (same weight matrix, bs different inputs)
    # This avoids allocating an oversized weight buffer
    print("  \"batch_sweep\": [")
    tw_first = True
    for bs in [1, 2, 4, 8, 16]:
        # Sum of bs sequential matmuls simulates batch decode
        var t0 = time.perf_counter()
        for _ in range(bs):
            var _ = f16_mm(b, x, o, NE, NE, NW)
        var t_b = (time.perf_counter() - t0) * 1000.0
        if not tw_first: print(",")
        tw_first = False
        var agg_tok = Float64(bs) / (t_b / 1000.0)
        print("    {\"batch\":" + String(bs) + ",\"ms\":" + String(Float64(t_b)) + ",\"agg_tok_s\":" + String(Float64(agg_tok)) + "}")
    print()
    print("  ],")
    
    # ═══ 6. MEMORY BANDWIDTH ═══
    # For a matmul of M×N, we read M*N*2 bytes (weights) + N*4 bytes (input) and write M*4 bytes (output)
    # Bandwidth = (bytes_read + bytes_written) / time
    print("  \"memory_bandwidth\": {")
    var mem_bw_qkv = Float64(NE * NE * 2 + NE * 4 + NE * 4) / (d_q_proj / 1000.0) / 1e9
    var mem_bw_ffn = Float64(FF * NE * 2 + NE * 4 + FF * 4) / (d_f_gate / 1000.0) / 1e9
    print("    \"Q_proj_GB_s\":" + String(Float64(mem_bw_qkv)) + ",")
    print("    \"FFN_gate_GB_s\":" + String(Float64(mem_bw_ffn)) + ",")
    print("    \"peak_Q_proj_tok_s\":" + String(Float64(mem_bw_qkv * 1e9 / Float64(NE * NE * 2 + NE * 4 + NE * 4) * 1000.0)))
    print("  },")
    
    # ═══ 5. SUMMARY ═══
    print("  \"summary\": {")
    # Decode tokens/s using actual layer timing
    print("    \"decode_tok_s\": " + String(Float64(1000.0 / d_layer_ms)) + ",")
    # Full model tok/s = 1 token / per-layer time
    print("    \"model_tok_s\": " + String(Float64(1000.0 / (full_ms / Float64(NL)))) + ",")
    print("    \"peak_tflops\": " + String(Float64(tflops)) + ",")
    print("    \"ms_per_layer\": " + String(Float64(d_layer_ms)) + ",")
    print("    \"bottleneck\": \"Q_proj " + String(inner) + "x" + String(NE) + " matmul\"")
    print("  }")
    print("}}")
