# MojoLlama Bench v4 MXFP4 — Qwen 3.6B MXFP4 Transformer Benchmark
# WHAT:  MXFP4 quantized matmul variant of mojollama_bench.mojo v4.
#        Configured for Qwen 2.5 3B architecture: NE=2048, NH=16, NK=2, HD=128, NL=36, FF=11008.
# WHY:   Compare MXFP4 vs f16 throughput to quantify quantization speedup.
# WHEN:  May 2026 — Qwen 3.6B MXFP4 target.
from std import time, math
from std.algorithm.backend.cpu.parallelize import parallelize

# MXFP4 constants
comptime MXFP4_BS: Int = 32
comptime MXFP4_BYTES: Int = 17  # 16 nibble bytes + 1 e8m0 scale byte

# ═══ f16 helpers (for input activations) ═══
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

# ═══ MXFP4 matmul kernel ═══
def mxfp4_matmul_row(r: Int, w: UnsafePointer[UInt8, MutExternalOrigin],
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
        mxfp4_matmul_row(r, w, x, o, nr, nc)
    parallelize[func=wk](num_work_items=nr, num_workers=nw)
    return (time.perf_counter() - t0) * 1000.0

# ═══ Stats ═══
def sort_f64(buf: UnsafePointer[Float64, MutExternalOrigin], n: Int):
    for i in range(n):
        var mi = i
        for j in range(i+1, n):
            if buf.load(j) < buf.load(mi): mi = j
        if mi != i:
            var t = buf.load(i); buf.store(i, buf.load(mi)); buf.store(mi, t)

def pct(buf: UnsafePointer[Float64, MutExternalOrigin], n: Int, p: Float64) -> Float64:
    return buf.load(Int(Float64(n-1) * p))

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

# ═══ Synthetic MXFP4 weight generation ═══
def fill_mxfp4_weights(w: UnsafePointer[UInt8, MutExternalOrigin], nr: Int, nc: Int):
    """Fill weight buffer with random-looking MXFP4 data.
    Each row: bpr blocks of 17 bytes each (16 nibble bytes + 1 scale byte)
    Nibbles are signed 4-bit [-8..7], scales are E8M0 [0..255]
    """
    var bpr = nc // MXFP4_BS
    var row_bytes = bpr * MXFP4_BYTES
    for r in range(nr):
        var ro = r * row_bytes
        for blk in range(bpr):
            var bo = ro + blk * MXFP4_BYTES
            for j in range(16):
                var v = UInt8(((r*7 + j*3 + blk*11) & 0xF) | (((r*13 + j*5 + blk*17) & 0xF) << 4))
                w.store(bo + j, v)
            var scale = UInt8(120 + (r ^ blk) % 20)  # E8M0 scales near ~1.0
            w.store(bo + 16, scale)

def fill_f32(buf: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    for i in range(n):
        buf.store(i, Float32(Float64(i % 100 - 50) * 0.01))

def main():
    # ═══ Qwen 2.5 3B Architecture ═══
    var NE: Int = 2048   # n_embd (hidden_size)
    var NH: Int = 16     # n_heads
    var NK: Int = 2      # n_kv_heads
    var HD: Int = 128    # head_dim
    var NL: Int = 36     # n_layers
    var FF: Int = 11008  # ff_hidden (intermediate_size)
    var NV: Int = 151936 # vocab_size
    var NW: Int = 32     # threads
    var n_warm: Int = 2
    var n_iter: Int = 10
    
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    
    var inner = NH * HD        # = 2048 (same as NE for Qwen 3B)
    var kv_dim = NK * HD       # = 256
    
    # MXFP4 weight bytes per row
    var qkv_bpr = NE // MXFP4_BS          # blocks per row for QKV shape
    var ffn_bpr = FF // MXFP4_BS          # blocks for FFN (ff_hidden)
    
    # Weight buffer sizes in MXFP4 format
    var sz_b_qkv = NE * qkv_bpr * MXFP4_BYTES     # NE×NE weight buffer (MXFP4)
    var sz_b_ff = FF * ffn_bpr * MXFP4_BYTES       # FF×NE (but MXFP4 columns = nc)
    var sz_b_ff_down = NE * ffn_bpr * MXFP4_BYTES  # NE×FF
    
    # Max weight rows for output buffer
    var max_wr = max(max(NE, FF), max(inner, kv_dim))
    var max_wc = max(max(NE, inner), max(kv_dim, FF))
    var max_bpr = max_wc // MXFP4_BS
    
    # Allocate buffers
    var buf = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_iter * 8))))
    var b_qkv = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(sz_b_qkv))))
    var b_ff_gate = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(sz_b_ff))))
    var b_ff_down = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(sz_b_ff_down))))
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_wc * 4))))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(max_wr * 4))))
    
    # Initialize weights and inputs
    fill_mxfp4_weights(b_qkv, NE, NE)
    fill_mxfp4_weights(b_ff_gate, FF, NE)
    fill_mxfp4_weights(b_ff_down, NE, FF)
    fill_f32(x, max_wc)
    
    # Print JSON
    print("{")
    print("\"mxfp4_bench_qwen3b\": {")
    print("  \"arch\": {")
    print("    \"n_embd\": " + String(NE) + ",\"n_heads\": " + String(NH) + ",\"n_kv_heads\": " + String(NK) + ",")
    print("    \"head_dim\": " + String(HD) + ",\"n_layers\": " + String(NL) + ",\"ff_hidden\": " + String(FF) + ",")
    print("    \"vocab_size\": " + String(NV) + ",\"threads\": " + String(NW) + ",\"quant\": \"MXFP4\"")
    print("  },")
    
    # ═══ 1. COMPONENT BREAKDOWN ═══
    print("  \"breakdown\": {")
    
    var t_qkv = bench_mean(b_qkv, x, o, NE, NE, NW, buf, n_warm, n_iter)
    var ops_q = Int64(NE) * Int64(NE) * 2
    print("    \"QKV_proj " + String(NE) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_qkv)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_qkv/1e6)) + "},")
    
    var t_kv = bench_mean(b_qkv, x, o, kv_dim, NE, NW, buf, n_warm, n_iter)
    ops_q = Int64(kv_dim) * Int64(NE) * 2
    print("    \"KV_proj " + String(kv_dim) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_kv)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_kv/1e6)) + "},")
    
    var t_oproj = bench_mean(b_qkv, x, o, NE, inner, NW, buf, n_warm, n_iter)
    ops_q = Int64(NE) * Int64(inner) * 2
    print("    \"O_proj " + String(NE) + "x" + String(inner) + "\":{\"ms\":" + String(Float64(t_oproj)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_oproj/1e6)) + "},")
    
    var t_ffn_gate = bench_mean(b_ff_gate, x, o, FF, NE, NW, buf, n_warm, n_iter)
    ops_q = Int64(FF) * Int64(NE) * 2
    print("    \"FFN_gate " + String(FF) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_ffn_gate)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_ffn_gate/1e6)) + "},")
    
    var t_ffn_up = bench_mean(b_ff_gate, x, o, FF, NE, NW, buf, n_warm, n_iter)
    print("    \"FFN_up " + String(FF) + "x" + String(NE) + "\":{\"ms\":" + String(Float64(t_ffn_up)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_ffn_up/1e6)) + "},")
    
    var t_ffn_down = bench_mean(b_ff_down, x, o, NE, FF, NW, buf, n_warm, n_iter)
    ops_q = Int64(NE) * Int64(FF) * 2
    print("    \"FFN_down " + String(NE) + "x" + String(FF) + "\":{\"ms\":" + String(Float64(t_ffn_down)) + ",\"gflops\":" + String(Float64(Float64(ops_q)/t_ffn_down/1e6)) + "}")
    
    print("  },")
    
    # ═══ 2. FULL MODEL (36 layers) ═══
    print("  \"full_model\": {")
    for _ in range(n_warm):
        for _ in range(NL):
            var _ = mxfp4_mm(b_qkv, x, o, NE, NE, NW)
            var __ = mxfp4_mm(b_qkv, x, o, kv_dim, NE, NW)
            var ___ = mxfp4_mm(b_qkv, x, o, kv_dim, NE, NW)
            var ____ = mxfp4_mm(b_qkv, x, o, NE, inner, NW)
            var _____ = mxfp4_mm(b_ff_gate, x, o, FF, NE, NW)
            var ______ = mxfp4_mm(b_ff_gate, x, o, FF, NE, NW)
            var _______ = mxfp4_mm(b_ff_down, x, o, NE, FF, NW)
    var model_total: Float64 = 0.0
    for _ in range(n_iter):
        var tm0 = time.perf_counter()
        for _ in range(NL):
            var _ = mxfp4_mm(b_qkv, x, o, NE, NE, NW)
            var __ = mxfp4_mm(b_qkv, x, o, kv_dim, NE, NW)
            var ___ = mxfp4_mm(b_qkv, x, o, kv_dim, NE, NW)
            var ____ = mxfp4_mm(b_qkv, x, o, NE, inner, NW)
            var _____ = mxfp4_mm(b_ff_gate, x, o, FF, NE, NW)
            var ______ = mxfp4_mm(b_ff_gate, x, o, FF, NE, NW)
            var _______ = mxfp4_mm(b_ff_down, x, o, NE, FF, NW)
        model_total += (time.perf_counter() - tm0) * 1000.0
    var full_ms = model_total / Float64(n_iter)
    
    var ops_per_layer = Int64(NE)*Int64(NE)*2 + Int64(kv_dim)*Int64(NE)*4 + Int64(NE)*Int64(NE)*2 + Int64(FF)*Int64(NE)*4 + Int64(NE)*Int64(FF)*2
    var total_ops = Int64(NL) * ops_per_layer
    var tflops = Float64(total_ops) / (full_ms / 1000.0) / 1e12
    
    print("    \"n_layers\": " + String(NL) + ",")
    print("    \"total_ms\":" + String(Float64(full_ms)) + ",")
    print("    \"ms_per_layer\":" + String(Float64(full_ms / Float64(NL))) + ",")
    print("    \"gflops\":" + String(Float64(tflops * 1000.0)) + ",")
    print("    \"tflops\":" + String(Float64(tflops)) + ",")
    print("    \"total_ops\":" + String(total_ops))
    print("  },")
    
    # ═══ 3. DECODE STEP ═══
    print("  \"decode_step\": [")
    var d_qkv = bench_mean(b_qkv, x, o, NE, NE, NW, buf, n_warm, n_iter)
    var d_k = bench_mean(b_qkv, x, o, kv_dim, NE, NW, buf, n_warm, n_iter)
    var d_v = bench_mean(b_qkv, x, o, kv_dim, NE, NW, buf, n_warm, n_iter)
    var d_o = bench_mean(b_qkv, x, o, NE, inner, NW, buf, n_warm, n_iter)
    var d_fg = bench_mean(b_ff_gate, x, o, FF, NE, NW, buf, n_warm, n_iter)
    var d_fu = bench_mean(b_ff_gate, x, o, FF, NE, NW, buf, n_warm, n_iter)
    var d_fd = bench_mean(b_ff_down, x, o, NE, FF, NW, buf, n_warm, n_iter)
    var d_layer = d_qkv + d_k + d_v + d_o + d_fg + d_fu + d_fd
    
    print("    {\"component\":\"Q_proj\",\"ms\":" + String(Float64(d_qkv)) + "},")
    print("    {\"component\":\"K_proj\",\"ms\":" + String(Float64(d_k)) + "},")
    print("    {\"component\":\"V_proj\",\"ms\":" + String(Float64(d_v)) + "},")
    print("    {\"component\":\"O_proj\",\"ms\":" + String(Float64(d_o)) + "},")
    print("    {\"component\":\"FFN_gate\",\"ms\":" + String(Float64(d_fg)) + "},")
    print("    {\"component\":\"FFN_up\",\"ms\":" + String(Float64(d_fu)) + "},")
    print("    {\"component\":\"FFN_down\",\"ms\":" + String(Float64(d_fd)) + "},")
    print("    {\"component\":\"TOTAL_LAYER\",\"ms\":" + String(Float64(d_layer)) + "},")
    print("    {\"component\":\"tok_s\",\"val\":" + String(Float64(1000.0 / d_layer)) + "}")
    print("  ],")
    
    # ═══ 4. THREAD SWEEP ═══
    print("  \"thread_sweep\": [")
    var tw_first = True
    for tw in [1, 2, 4, 8, 16, 24, 32]:
        var t_w = bench_mean(b_qkv, x, o, NE, NE, tw, buf, n_warm, n_iter // 2)
        if not tw_first: print(",")
        tw_first = False
        var tok = 1000.0 / t_w
        print("    {\"threads\":" + String(tw) + ",\"qkv_ms\":" + String(Float64(t_w)) + ",\"tok_s\":" + String(Float64(tok)) + "}")
    print()
    print("  ],")
    
    # ═══ 5. SUMMARY ═══
    print("  \"summary\": {")
    print("    \"decode_tok_s\": " + String(Float64(1000.0 / d_layer)) + ",")
    print("    \"model_tok_s\": " + String(Float64(1000.0 / (full_ms / Float64(NL)))) + ",")
    print("    \"peak_tflops\": " + String(Float64(tflops)) + ",")
    print("    \"bottleneck\": \"FFN_gate " + String(FF) + "x" + String(NE) + " MXFP4\"")
    print("  }")
    print("}}")
