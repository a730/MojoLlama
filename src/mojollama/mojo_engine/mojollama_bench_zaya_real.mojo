# MojoLlama Bench — ZAYA1-8B Realistic MXFP4 Benchmark
# WHAT:  Benchmarks ZAYA1-8B with realistic memory bandwidth constraints.
#        Uses a large weight buffer (>L3 cache) to force cold-cache behavior.
# WHY:   Synthetic benchmarks overestimate by keeping weights hot in cache.
# WHEN:  May 2026 — corrected ZAYA1-8B throughput.
from std import time, math
from std.algorithm.backend.cpu.parallelize import parallelize

comptime MXFP4_BS: Int = 32
comptime MXFP4_BYTES: Int = 17

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

def mxfp4_mm(w_addr: Int64,
             x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin],
             nr: Int, nc: Int, nw: Int) -> Float64:
    var w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_addr))
    var t0 = time.perf_counter()
    def wk(r: Int) capturing -> None:
        mxfp4_row(r, w, x, o, nr, nc)
    parallelize[func=wk](num_work_items=nr, num_workers=nw)
    return (time.perf_counter() - t0) * 1000.0

def fill_mxfp4(addr: Int64, sz: Int):
    var w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(addr))
    for i in range(sz):
        w.store(i, UInt8((i * 7 + (i >> 4) * 13 + (i >> 8) * 17) & 0xFF))

def p(s: String): print(s)

def main():
    var NE: Int = 2048       # hidden_size
    var NH: Int = 8
    var NK: Int = 2
    var HD: Int = 128
    var N_ATTN: Int = 40
    var N_MOE: Int = 40
    var NW: Int = 32
    var n_iter: Int = 3
    
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    
    var q_dim = NH * HD       # 1024
    var kv_dim = NK * HD      # 256
    var v1_dim = HD           # 128
    var ff_gate = 4096        # fc1
    var ff_hid = 2048         # fc2
    
    # Weight sizes per layer (MXFP4)
    var bpr_ne = NE // MXFP4_BS
    var bpr_q = q_dim // MXFP4_BS
    var bpr_h = ff_hid // MXFP4_BS
    
    var w_attn = q_dim * bpr_ne * MXFP4_BYTES \
               + kv_dim * bpr_ne * MXFP4_BYTES \
               + v1_dim * bpr_ne * MXFP4_BYTES \
               + v1_dim * bpr_ne * MXFP4_BYTES \
               + NE * bpr_q * MXFP4_BYTES
    
    var w_moe = ff_gate * bpr_ne * MXFP4_BYTES \
              + ff_hid * bpr_h * MXFP4_BYTES
    
    var total_bytes = Int64(N_ATTN) * Int64(w_attn) + Int64(N_MOE) * Int64(w_moe)
    
    p("// Weights: " + String(total_bytes / 1048576) + " MB total")
    
    # Allocate one giant weight pool
    var pool_addr = _alc(total_bytes)
    fill_mxfp4(pool_addr, Int(total_bytes))
    
    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var o = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(4096 * 4))))
    for i in range(NE): x.store(i, Float32(Float64(i % 100 - 50) * 0.031))
    
    var bpr_ne64 = Int64(bpr_ne)
    var bpr_q64 = Int64(bpr_q)
    var bpr_h64 = Int64(bpr_h)
    var mb = Int64(MXFP4_BYTES)
    
    var w_attn64 = Int64(w_attn)
    var w_moe64 = Int64(w_moe)
    
    p("{")
    p("\"zaya1_8b_realistic\": {")
    p("  \"arch\": {")
    p("    \"hidden\": " + String(NE) + ",\"weight_mb\": " + String(total_bytes / 1048576))
    p("  },")
    
    # ═══ COLD-CACHE FULL MODEL ═══
    p("  \"runs\": [")
    var total_ms: Float64 = 0.0
    
    for ri in range(n_iter):
        var t0 = time.perf_counter()
        var wp = pool_addr
        
        # 40 attention layers
        for li in range(N_ATTN):
            var _ = mxfp4_mm(wp, x, o, q_dim, NE, NW); wp += Int64(q_dim) * bpr_ne64 * mb
            var __ = mxfp4_mm(wp, x, o, kv_dim, NE, NW); wp += Int64(kv_dim) * bpr_ne64 * mb
            var ___ = mxfp4_mm(wp, x, o, v1_dim, NE, NW); wp += Int64(v1_dim) * bpr_ne64 * mb
            var ____ = mxfp4_mm(wp, x, o, v1_dim, NE, NW); wp += Int64(v1_dim) * bpr_ne64 * mb
            var _____ = mxfp4_mm(wp, x, o, NE, q_dim, NW); wp += Int64(NE) * bpr_q64 * mb
        
        # 40 MoE layers
        for li in range(N_MOE):
            var _ = mxfp4_mm(wp, x, o, ff_gate, NE, NW); wp += Int64(ff_gate) * bpr_ne64 * mb
            var __ = mxfp4_mm(wp, x, o, ff_hid, ff_hid, NW); wp += Int64(ff_hid) * bpr_h64 * mb
        
        var ms = (time.perf_counter() - t0) * 1000.0
        total_ms += ms
        if ri > 0: p(",")
        p("    {\"run\":" + String(ri) + ",\"ms\":" + String(Float64(ms)) + "}")
    
    p("")
    p("  ],")
    
    var avg_ms = total_ms / Float64(n_iter)
    var tok_s = 1000.0 / avg_ms
    var bw = Float64(total_bytes) / (avg_ms / 1000.0) / 1e9
    
    p("  \"result\": {")
    p("    \"avg_80_layers_ms\":" + String(Float64(avg_ms)) + ",")
    p("    \"tokens_per_second\":" + String(Float64(tok_s)) + ",")
    p("    \"effective_bw_gb_s\":" + String(Float64(bw)) + ",")
    p("    \"total_weights_mb\": " + String(total_bytes / 1048576))
    p("  },")
    
    # ═══ CACHE-HOT COMPARISON ═══
    p("  \"cache_hot\": {")
    # Single attn layer, same weights reused
    var t_hot: Float64 = 0.0
    for _ in range(n_iter):
        var t0 = time.perf_counter()
        # Run 40 attn layers but with SAME weights (cache hot)
        var w0 = pool_addr
        for li in range(N_ATTN):
            var _ = mxfp4_mm(w0, x, o, q_dim, NE, NW)
            var __ = mxfp4_mm(w0, x, o, kv_dim, NE, NW)
            var ___ = mxfp4_mm(w0, x, o, v1_dim, NE, NW)
            var ____ = mxfp4_mm(w0, x, o, v1_dim, NE, NW)
            var _____ = mxfp4_mm(w0, x, o, NE, q_dim, NW)
        t_hot += (time.perf_counter() - t0) * 1000.0
    var hot_attn = t_hot / Float64(n_iter)
    
    var t_moe_hot: Float64 = 0.0
    var w_moe_start = pool_addr + Int64(N_ATTN) * w_attn64
    for _ in range(n_iter):
        var t0 = time.perf_counter()
        var w0 = w_moe_start
        for li in range(N_MOE):
            var _ = mxfp4_mm(w0, x, o, ff_gate, NE, NW)
            var __ = mxfp4_mm(w0, x, o, ff_hid, ff_hid, NW)
        t_moe_hot += (time.perf_counter() - t0) * 1000.0
    var hot_moe = t_moe_hot / Float64(n_iter)
    
    p("    \"hot_40_attn_ms\":" + String(Float64(hot_attn)) + ",")
    p("    \"hot_40_moe_ms\":" + String(Float64(hot_moe)) + ",")
    p("    \"hot_tok_s\":" + String(Float64(1000.0 / ((hot_attn + hot_moe) / 80.0))))
    p("  }")
    
    p("}}")
