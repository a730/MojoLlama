# benchmark_qa.mojo — Pure Mojo Q&A Benchmark
# Uses std.benchmark.Bencher for timing questions through TinyLlama Q8_0
# ARCH: TinyLlama 1.1B, NE=2048, NH=32, NK=4, HD=64, NL=22
#
# HOW IT WORKS:
# 1. Loads Q8_0 weights once
# 2. Defines questions as inline token arrays + expected substrings
# 3. Runs each through the engine, wrapped in Bencher for timing
# 4. Checks output for expected answer
# 5. Reports per-question latency + accuracy table
#
# USAGE:
#   mojo build benchmark_qa.mojo --emit object -o benchmark_qa.o \
#     && gcc -o benchmark_qa benchmark_qa.o ... (same link flags as build_q8.sh)
#   LD_LIBRARY_PATH=... OMP_PLACES=cores OMP_PROC_BIND=close ./benchmark_qa 32

from std import time
from std.benchmark import Bencher, Bench, BenchConfig, BenchId, Format
from std.sys import argv
from std.math import sqrt, exp

# ─── Architecture (TinyLlama 1.1B) ───
comptime NE: Int = 2048
comptime NH: Int = 32
comptime NK: Int = 4
comptime HD: Int = 64
comptime NL: Int = 22
comptime NF: Int = 5632
comptime NV: Int = 32000
comptime MAX_SEQ: Int = 256  # enough for question + response
comptime W: Int = 8
comptime RPW: Int = 8
comptime B: Int = 1  # single-item benchmark
comptime QK: Int = 32
comptime QB: Int = 34
comptime EP: Float32 = 1e-5
comptime CS: Int = MAX_SEQ * NK * HD  # KV cache size per layer

# ─── Questions ───
# Each question: [BOS, token1, token2, ..., tokenN]
# expected answer substring (lowercase for matching)
# format: [num_tokens, token_0, token_1, ..., expected_answer_len, expected_answer_bytes...]

# Question 1: "2+2" → [1, 29871, 29906, 29974, 29906]
var Q1 = [1, 29871, 29906, 29974, 29906]
var Q1_NP = 5
var Q1_ANS = [1, 52]  # "4" as bytes

# Question 2: "3+5" → [1, 29871, 29906, 29974, 29906] (actually 3+5)
var Q2 = [1, 29871, 29907, 29974, 29907]
var Q2_NP = 5
var Q2_ANS = [1, 56]  # "8" as bytes

# Question 3: "Capital of France?" (approximate tokens)
var Q3 = [1, 29871, 4870, 338, 29951, 29966, 29947, 4870, 338, 16742, 29973]
var Q3_NP = 11
var Q3_ANS = [5, 80, 97, 114, 105, 115]  # "Paris"

# ─── External functions (same as tinyllama_gen_q8) ───
@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...
@extern("free")  
def _c_free(p: Int64) abi("C") -> None: ...
@extern("open")
def _open(p: UnsafePointer[UInt8, MutExternalOrigin], f: Int) abi("C") -> Int: ...
@extern("read")
def _read(fd: Int, b: UnsafePointer[UInt8, MutExternalOrigin], c: Int64) abi("C") -> Int64: ...
@extern("lseek")
def _lseek(fd: Int, o: Int64, w: Int) abi("C") -> Int64: ...
@extern("close")
def _close(fd: Int) abi("C") -> Int: ...

# ─── f16→f32 conversion ───
def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1)
    var e = Int((h >> 10) & 0x1F)
    var m = Int(h & 0x3FF)
    if e == 0: var r = Float32(m) * 5.960464477539063e-8; return -r if s != 0 else r
    if e == 31: return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    var tmp = alloc[UInt8](4)
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp)).store(0, bits)
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp)).load(0)

# ─── Q8_0 row bytes ───
def q8_rb(nc: Int) -> Int:
    return ((nc + QK - 1) // QK) * QB

# ─── Q8_0 matmul (single row, no parallelize for benchmark) ───
def mm_q8(qa: Int, x: UnsafePointer[Float32, MutExternalOrigin],
          o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int = 4):
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(qa))
    var rb = q8_rb(nc); var nb = (nr + RPW - 1) // RPW
    def wk(wi: Int) capturing:
        var rs = wi * RPW; var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            var acc = SIMD[DType.float32, W](0.0); var ro = r * rb; var col = 0
            while col < nc:
                var bo = ro + (col // QK) * QB
                var lo = Int(q.load(bo)); var hi = Int(q.load(bo + 1))
                var sv = SIMD[DType.float32, W](h2f(UInt16(lo | (hi << 8))))
                comptime for grp in range(4):
                    var u8 = q.load[width=8](bo + 2 + grp * 8)
                    var wf = (u8.cast[DType.float32]() - SIMD[DType.float32, 8](128.0)) * sv
                    acc = wf.fma[FastMathFlag.FAST](x.load[width=W](col + grp*8), acc)
                col += QK
            o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ─── RMS norm ───
def rms_norm(x, o, w, n):
    var ss = Float32(0.0); var i = 0
    while i + 8 <= n: var v = x.load[width=8](i); ss += (v * v).reduce_add(); i += 8
    while i < n: ss += x.load(i) * x.load(i); i += 1
    var inv = 1.0 / sqrt(ss / Float32(n) + EP)
    var inv_v = SIMD[DType.float32, 8](inv); i = 0
    while i + 8 <= n: var v = x.load[width=8](i); o.store[width=8](i, v * inv_v * w.load[width=8](i)); i += 8
    while i < n: o.store(i, x.load(i) * w.load(i) * inv); i += 1

# ─── SiLU ───
def silu(p, n):
    for i in range(n):
        var v = p.load(i)
        if v < -80.0: v = -80.0
        if v > 80.0: v = 80.0
        p.store(i, v / (1.0 + exp(-v)))

# ─── Softmax ───
def softmax(p, n):
    var mx = Float32(-1e9)
    for i in range(n):
        var v = p.load(i)
        if v > mx: mx = v
    var sm = Float32(0.0)
    for i in range(n):
        var e = exp(p.load(i) - mx)
        p.store(i, e); sm += e
    var inv = 1.0 / (sm + 1e-10)
    for i in range(n): p.store(i, p.load(i) * inv)

# ─── Load file helper ───
def load_file(path_str: String) -> Int64:
    var buf = alloc[UInt8](path_str.byte_length() + 1)
    var sp = path_str.unsafe_ptr()
    for i in range(path_str.byte_length()): buf.store(i, sp.load(i))
    buf.store(path_str.byte_length(), UInt8(0))
    var fd = _open(buf, 0)
    if fd < 0: return -1
    var sz = _lseek(fd, 0, 2); _ = _lseek(fd, 0, 0)
    var ptr = _alc(sz)
    if ptr == 0: _ = _close(fd); return -1
    _ = _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(ptr)), sz)
    _ = _close(fd); return ptr

# ─── Vocab decode ───
def decode_token(voc: UnsafePointer[UInt8, MutExternalOrigin],
                 voc_lens: UnsafePointer[Int32, MutExternalOrigin],
                 nv: Int, tok: Int) -> String:
    for i in range(nv):
        var vid = Int(voc_lens.load(nv + i))
        if vid == tok:
            var off = Int(voc_lens.load(2 * nv + i)); var l = Int(voc_lens.load(i))
            var p = voc + off
            var result = String("")
            for j in range(l): result = result + chr(Int(p.load(j)))
            return result
    return String("")

# ─── Check if answer appears in decoded text ───
def contains_answer(decoded: String, ans_ptr: UnsafePointer[UInt8, MutExternalOrigin],
                    ans_len: Int) -> Bool:
    var dl = decoded.byte_length()
    for i in range(dl - ans_len + 1):
        var match = True
        for j in range(ans_len):
            var dc = Int(decoded.unsafe_ptr().load(i + j))
            var ac = Int(ans_ptr.load(j))
            # lowercase comparison
            if dc >= 65 and dc <= 90: dc += 32
            if ac >= 65 and ac <= 90: ac += 32
            if dc != ac: match = False; break
        if match: return True
    return False

# ─── Initialize benchmark questions from data ───
struct QAItem:
    var prompt: UnsafePointer[Int32, MutExternalOrigin]
    var np: Int
    var ans_ptr: UnsafePointer[UInt8, MutExternalOrigin]
    var ans_len: Int

# ─── Main benchmark ───
def main() raises:
    print("MojoLlama Q&A Benchmark (Pure Mojo)")
    print("===================================")
    
    var t0 = time.perf_counter()
    var args = argv(); var nw = 32
    if len(args) > 1: nw = Int(String(args[1]))
    
    # ─── Load weights ───
    var wdir = String("/tmp/weights/")
    var wl = alloc[Int64](NL * 8)  # 8 weight slots per layer
    var w_emb = load_file(String("/tmp/weights/token_embd_weight.q4"))
    var w_on = load_file(String("/tmp/weights/output_norm_weight.f32"))
    var w_out = load_file(String("/tmp/weights/output_weight.q4"))
    
    # Load per-layer weights (simplified — just the key ones)
    # (In a full implementation, load all weights like tinyllama_gen_q8)
    
    var t_load = time.perf_counter()
    print("Load: ", Int((t_load - t0) * 1000), " ms")
    
    # ─── Allocate buffers ───
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var rp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NE * 4))))
    var q = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NH * HD * 4))))
    var k = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NK * HD * 4))))
    var v_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NK * HD * 4))))
    var att = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NH * HD * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(NV * 4))))
    var kc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(CS * 4))))
    var vc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(CS * 4))))
    
    # ─── Build question list ───
    var num_q = 3
    var items = alloc[QAItem](num_q)
    
    # Q1: "2+2"
    var q1_toks = alloc[Int32](Q1_NP)
    for i in range(Q1_NP): q1_toks.store(i, Int32(Q1[i]))
    var q1_ans = alloc[UInt8](1)
    q1_ans.store(0, UInt8(52))  # "4"
    items.store(0, QAItem(q1_toks, Q1_NP, q1_ans, 1))
    
    # Q2: "3+5"
    var q2_toks = alloc[Int32](Q2_NP)
    for i in range(Q2_NP): q2_toks.store(i, Int32(Q2[i]))
    var q2_ans = alloc[UInt8](1)
    q2_ans.store(0, UInt8(56))  # "8"
    items.store(1, QAItem(q2_toks, Q2_NP, q2_ans, 1))
    
    # Q3: "Capital of France?"
    var q3_toks = alloc[Int32](Q3_NP)
    for i in range(Q3_NP): q3_toks.store(i, Int32(Q3[i]))
    var q3_ans = alloc[UInt8](5)
    q3_ans = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=UnsafePointer[UInt8, MutExternalOrigin](q3_ans).address)
    q3_ans.store(0, 80); q3_ans.store(1, 97)
    q3_ans.store(2, 114); q3_ans.store(3, 105); q3_ans.store(4, 115)  # "Paris"
    items.store(2, QAItem(q3_toks, Q3_NP, q3_ans, 5))
    
    # ─── Bencher configuration ───
    var config = BenchConfig(
        min_runtime_secs=0.1,
        max_runtime_secs=5.0,
        num_warmup_iters=0,  # no warmup — first call is the real one
        max_iters=3,
        num_repetitions=1,
        format=Format.table,
    )
    
    # Q: each question runs Bencher internally (not through Bench) 
    # because we need the weight loading and model state
    
    var correct_count = 0
    var total_latency = 0.0
    
    print()
    print(f"{'Question':<25s} {'Result':>6s} {'Latency':>8s}")
    print(f"{'─'*25} {'─'*6} {'─'*8}")
    
    for qi in range(num_q):
        var item = items.load(qi)
        var np = item.np
        var prompt = item.prompt
        
        # Time a single forward pass (embed + generate 16 tokens)
        var bencher = Bencher(num_iters=1)
        bencher.iter[def() capturing -> None:
            # Reset KV cache (zero it out)
            # (In real impl, KV cache persists or resets per question)
            
            # Embed first token
            var tok = Int(prompt.load(0))
            var emb = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(w_emb))
            for i in range(NE): hp.store(i, h2f(emb.load(tok * NE + i)))
            
            # Generate 16 tokens
            for pos in range(np, np + 16):
                # Single-layer forward pass
                # (In real impl, runs all NL layers)
                
                # Argmax
                var best = 0; var bv = lp.load(0)
                for i in range(1, NV):
                    if lp.load(i) > bv: bv = lp.load(i); best = i
                
                # Store next token
                if best != 2:
                    # Embed next token
                    var emb2 = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=Int(w_emb))
                    for i in range(NE): hp.store(i, h2f(emb2.load(best * NE + i)))
        ]()
        
        var latency_ms = Float64(bencher.elapsed) / 1e6  # ns to ms
        total_latency += latency_ms
        
        # Check answer (placeholder — real impl checks output)
        var is_correct = False
        
        var status = "✅" if is_correct else "❌"
        print(f"{'Q'+String(qi+1):<25s} {status:>6s} {latency_ms:7.1f}ms")
        if is_correct: correct_count += 1
    
    # ─── Summary ───
    var avg_lat = total_latency / Float64(num_q)
    print()
    print(f"{'─'*41}")
    print(f"  Accuracy:  {correct_count}/{num_q}")
    print(f"  Avg latency: {avg_lat:.0f} ms per query")
    print(f"  Questions:  {num_q}")
    print()
    print("  NOTE: This is a template. Full implementation needs:")
    print("  1. Complete weight loading (like tinyllama_gen_q8)")
    print("  2. Full 22-layer forward pass")
    print("  3. Proper KV cache management")
    print("  4. Vocab loading for output decoding")
    print("  5. Bencher per-question timing")
    print()
    print("  See mojolang.org std.benchmark docs for:")
    print("  - Bench, BenchConfig for multi-benchmark orchestration")
    print("  - Bencher.iter() for function timing")
    print("  - BenchmarkInfo for statistical aggregation")
    print("  - BenchMetric for throughput measurement")
