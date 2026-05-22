# MojoLlama Latency Benchmark — vLLM-style latency profiling in pure Mojo
# WHAT:  Measures time-to-first-token, per-token decode latency, and
#        end-to-end inference latency for synthetic LLM workloads.
# WHY:   Replace Python benchmark_latency.py. Critical for real-time apps.
# WHEN:  May 2026 — pure Mojo, no Python FFI, no external deps.
from std import time, math
from std.algorithm.backend.cpu.parallelize import parallelize

# ─── f16 → f32 decoder ───
def f16_to_f32(h: UInt16) -> Float32:
    var s = UInt32(h >> 15); var e = UInt32((h >> 10) & 0x1F); var m = UInt32(h & 0x3FF)
    if e == 0: return 0.0 if m == 0 else Float32(Float64(m) * 5.960464477539063e-8)
    if e == 31: return 0.0
    var r = Float32(m | 0x400); var ei = Int(e) - 25
    if ei >= 0:
        for _ in range(ei): r *= 2.0
    else:
        for _ in range(-ei): r *= 0.5
    return -r if s != 0 else r

# ─── Matmul kernels ───
def f16_row(w: UnsafePointer[UInt8, MutExternalOrigin],
            x: UnsafePointer[Float32, MutExternalOrigin],
            r: Int, nc: Int) -> Float32:
    var acc: Float32 = 0.0; var ro = r * nc
    for c in range(nc):
        var h = UInt16(w.load((ro + c) * 2)) | (UInt16(w.load((ro + c) * 2 + 1)) << 8)
        acc += f16_to_f32(h) * x.load(c)
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

# ─── Sorting for percentiles ───
def sort_f64(buf: UnsafePointer[Float64, MutExternalOrigin], n: Int):
    for i in range(n):
        var mi = i
        for j in range(i+1, n):
            if buf.load(j) < buf.load(mi): mi = j
        if mi != i:
            var t = buf.load(i); buf.store(i, buf.load(mi)); buf.store(mi, t)

def percentile(buf: UnsafePointer[Float64, MutExternalOrigin], n: Int, p: Float64) -> Float64:
    return buf.load(Int(Float64(n-1) * p))

def main():
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    
    # ─── Model config ───
    var n_embd = 4096         # model dimension (Llama 3 8B)
    var n_heads = 32
    var n_kv_heads = 8
    var head_dim = 128
    var n_layers = 32
    var ff_hidden = 14336
    var vocab_size = 128256
    
    # ─── Benchmark params ───
    var n_warmup = 3
    var n_iter = 15
    var prefill_tokens = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
    var decode_steps = [1, 8, 16, 32, 64, 128]
    var nw = 32  # threads
    
    # Allocate working buffers
    var qk_dim = n_embd  # QKV projection size
    var w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_embd * qk_dim * 2))))
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_embd * 4))))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_embd * 4))))
    var buf = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(64 * 8))))  # temp for stats
    
    # Init data
    for i in range(n_embd * qk_dim):
        w.store(i*2, UInt8(i & 0xFF)); w.store(i*2+1, UInt8((i>>8) & 0xFF))
    for i in range(n_embd): x.store(i, Float32(Float64(i % 100 - 50) * 0.01))
    
    print("{")
    print("\"benchmark_latency\": {")
    print("  \"model\": \"synthetic_llama3_8b\",")
    print("  \"config\": {")
    print("    \"n_embd\": " + String(n_embd) + ",")
    print("    \"n_heads\": " + String(n_heads) + ",")
    print("    \"n_kv_heads\": " + String(n_kv_heads) + ",")
    print("    \"head_dim\": " + String(head_dim) + ",")
    print("    \"n_layers\": " + String(n_layers) + ",")
    print("    \"ff_hidden\": " + String(ff_hidden) + ",")
    print("    \"vocab_size\": " + String(vocab_size))
    print("  },")
    
    # ═══════════════════════════════════════════════
    # 1. PREFILL LATENCY (Time to First Token)
    # ═══════════════════════════════════════════════
    # Measures how long it takes to process the input prompt.
    # Simulated: QKV matmul with prompt_len rows (each token = 1 row).
    print("  \"prefill_latency\": [")
    var first = True
    for pi in range(len(prefill_tokens)):
        var pt = prefill_tokens[pi]
        var nr = pt  # number of tokens = rows in QKV matmul
        if nr < 1: nr = 1
        for _ in range(n_warmup): f16_mm(w, x, o, nr, qk_dim, nw)
        for mi in range(n_iter): buf.store(mi, f16_mm(w, x, o, nr, qk_dim, nw))
        sort_f64(buf, n_iter)
        var mean: Float64 = 0.0
        for mi in range(n_iter): mean += buf.load(mi)
        mean /= Float64(n_iter)
        var med = percentile(buf, n_iter, 0.5)
        var p95 = percentile(buf, n_iter, 0.95)
        var p99 = percentile(buf, n_iter, 0.99)
        var tpms = mean / Float64(nr)  # ms per token
        if not first: print(",")
        first = False
        print("    {\"prompt_tokens\": " + String(pt) + ",")
        print("      \"total_ms\": " + String(Float64(mean)) + ",")
        print("      \"ms_per_token\": " + String(Float64(tpms)) + ",")
        print("      \"median_ms\": " + String(Float64(med)) + ",")
        print("      \"p95_ms\": " + String(Float64(p95)) + ",")
        print("      \"p99_ms\": " + String(Float64(p99)) + ",")
        print("      \"tok_s\": " + String(Float64(1000.0 / tpms)) + "}")
    print("  ],")
    
    # ═══════════════════════════════════════════════
    # 2. DECODE LATENCY (Per-Token Generation)
    # ═══════════════════════════════════════════════
    # Measures per-token decode latency at batch=1.
    # Simulated: 1-row QKV matmul (single token generation step).
    print("  \"decode_latency\": [")
    first = True
    for di in range(len(decode_steps)):
        var ds = decode_steps[di]
        for _ in range(n_warmup): f16_mm(w, x, o, 1, qk_dim, nw)
        for mi in range(n_iter): buf.store(mi, f16_mm(w, x, o, 1, qk_dim, nw))
        sort_f64(buf, n_iter)
        var mean: Float64 = 0.0
        for mi in range(n_iter): mean += buf.load(mi)
        mean /= Float64(n_iter)
        var med = percentile(buf, n_iter, 0.5)
        var p95 = percentile(buf, n_iter, 0.95)
        var p99 = percentile(buf, n_iter, 0.99)
        if not first: print(",")
        first = False
        print("    {\"decode_tokens\": " + String(ds) + ",")
        print("      \"per_token_ms\": " + String(Float64(mean)) + ",")
        print("      \"median_ms\": " + String(Float64(med)) + ",")
        print("      \"p95_ms\": " + String(Float64(p95)) + ",")
        print("      \"p99_ms\": " + String(Float64(p99)) + ",")
        print("      \"tok_s\": " + String(Float64(1000.0 / mean)) + "}")
    print("  ],")
    
    # ═══════════════════════════════════════════════
    # 3. END-TO-END LATENCY
    # ═══════════════════════════════════════════════
    # Measures total time for prefill + N decode steps.
    # Common patterns: 128+128, 512+128, 2048+128, 4096+256
    print("  \"end_to_end\": [")
    var e2e_patterns = [(128, 128), (512, 128), (1024, 128), (2048, 128), (4096, 256)]
    first = True
    for ei in range(len(e2e_patterns)):
        var pp = e2e_patterns[ei][0]
        var gg = e2e_patterns[ei][1]
        for _ in range(n_warmup):
            f16_mm(w, x, o, pp, qk_dim, nw)  # prefill
            for _ in range(gg): f16_mm(w, x, o, 1, qk_dim, nw)  # decode
        for mi in range(n_iter):
            var t0 = time.perf_counter()
            f16_mm(w, x, o, pp, qk_dim, nw)  # prefill
            for _ in range(gg): f16_mm(w, x, o, 1, qk_dim, nw)  # decode
            buf.store(mi, (time.perf_counter() - t0) * 1000.0)
        sort_f64(buf, n_iter)
        var mean: Float64 = 0.0
        for mi in range(n_iter): mean += buf.load(mi)
        mean /= Float64(n_iter)
        var med = percentile(buf, n_iter, 0.5)
        var p95 = percentile(buf, n_iter, 0.95)
        var p99 = percentile(buf, n_iter, 0.99)
        var total_tokens = pp + gg
        var tps = 1000.0 * Float64(total_tokens) / mean
        if not first: print(",")
        first = False
        print("    {\"prompt\": " + String(pp) + ",")
        print("      \"gen\": " + String(gg) + ",")
        print("      \"total_tokens\": " + String(total_tokens) + ",")
        print("      \"total_ms\": " + String(Float64(mean)) + ",")
        print("      \"median_ms\": " + String(Float64(med)) + ",")
        print("      \"p95_ms\": " + String(Float64(p95)) + ",")
        print("      \"p99_ms\": " + String(Float64(p99)) + ",")
        print("      \"tok_s\": " + String(Float64(tps)) + ",")
        print("      \"ttft_ms\": " + String(Float64(mean / Float64(pp + 1) * Float64(pp))) + "}")
    print("  ],")
    
    # ═══════════════════════════════════════════════
    # 4. COLD vs WARM CACHE
    # ═══════════════════════════════════════════════
    print("  \"cache_effects\": {")
    # Cold: fresh alloc (simulate by allocating new buffers)
    var cold_total: Float64 = 0.0
    for mi in range(3):
        var cw = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_embd * qk_dim * 2))))
        var cx = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_embd * 4))))
        var co = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_embd * 4))))
        for i in range(n_embd * qk_dim):
            cw.store(i*2, UInt8(i & 0xFF)); cw.store(i*2+1, UInt8((i>>8) & 0xFF))
        for i in range(n_embd): cx.store(i, Float32(Float64(i % 100 - 50) * 0.01))
        var t0 = time.perf_counter()
        def wk_c(r: Int) capturing -> None:
            co.store(r, f16_row(cw, cx, r, qk_dim))
        parallelize[func=wk_c](num_work_items=128, num_workers=nw)
        cold_total += (time.perf_counter() - t0) * 1000.0
    cold_total /= 3.0
    # Warm: use already-cached buffers
    var warm_total: Float64 = 0.0
    for mi in range(3):
        warm_total += f16_mm(w, x, o, 128, qk_dim, nw)
    warm_total /= 3.0
    print("    \"cold_128tok_ms\": " + String(Float64(cold_total)) + ",")
    print("    \"warm_128tok_ms\": " + String(Float64(warm_total)) + ",")
    print("    \"cold_warm_ratio\": " + String(Float64(cold_total / warm_total)))
    print("  },")
    
    # ═══════════════════════════════════════════════
    # 5. SUMMARY / SCORECARD
    # ═══════════════════════════════════════════════
    print("  \"summary\": {")
    # Best prefill tok/s (at 4096 tokens)
    print("    \"prefill_tok_s_4096\": " + String(Float64(1000.0 / (percentile(buf, 3, 0.5) / 4096.0))) + ",")
    # Decode tok/s (single token)
    for _ in range(n_warmup): f16_mm(w, x, o, 1, qk_dim, nw)
    for mi in range(3): buf.store(mi, f16_mm(w, x, o, 1, qk_dim, nw))
    var decode_ms = percentile(buf, 3, 0.5)
    print("    \"decode_tok_s\": " + String(Float64(1000.0 / decode_ms)) + ",")
    print("    \"ttft_128tok_ms\": " + String(Float64(128.0 * decode_ms)) + ",")
    print("    \"ttft_2048tok_ms\": " + String(Float64(2048.0 * decode_ms)))
    print("  }")
    print("}}")
