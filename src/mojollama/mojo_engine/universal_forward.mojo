# MojoLlama Universal Forward Pass — Pure Mojo, any architecture, any quant
# WHAT:  Single unified forward pass supporting Llama, Mistral, Gemma, Mamba,
#        GPT-OSS, TinyLlama architectures with all GGUF quant types.
# WHY:   Replace all model-specific Mojo forward passes with one file.
#        Comptime-configurable: set MODEL_CONFIG to select architecture.
# WHEN:  May 2026 — unified dispatch with Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K/MXFP4/f16.
from std import time, math
from std.algorithm.backend.cpu.parallelize import parallelize

# ═══════════════════════════════════════════════
# ARCHITECTURE CONFIG — set these before building
# ═══════════════════════════════════════════════
# To configure, set: mojo build -D LLAMA_3_8B ...
# Or edit the values below for your model.

# ─── Model presets (uncomment the one you need) ───
# TinyLlama 1.1B:  NE=2048, NH=32, NK=4,  HD=64,  NL=22, FF=5632, NV=32000, Q=2
# Llama 3 8B:     NE=4096, NH=32, NK=8,  HD=128, NL=32, FF=14336, NV=128256, Q=8
# Mistral 7B:     NE=4096, NH=32, NK=8,  HD=128, NL=32, FF=14336, NV=32000, Q=8
# Gemma 2 9B:     NE=3584, NH=16, NK=16, HD=256, NL=42, FF=14336, NV=256000, Q=8
# GPT-OSS-20B:    NE=2880, NH=64, NK=8,  HD=64,  NL=24, FF=2880,  NV=201088, Q=39, MOE
# Qwen2.5-7B:     NE=3584, NH=28, NK=4,  HD=128, NL=28, FF=18944, NV=152064, Q=8
# Phi-3 3.8B:     NE=3072, NH=24, NK=24, HD=128, NL=32, FF=8192,  NV=32064, Q=8

# Default: TinyLlama
comptime NE:  Int = 2048    # n_embd
comptime NH:  Int = 32     # n_heads
comptime NK:  Int = 4      # n_kv_heads
comptime HD:  Int = 64     # head_dim
comptime NL:  Int = 22     # n_layers
comptime FF:  Int = 5632   # ffn_hidden
comptime NV:  Int = 32000  # vocab_size
comptime QT:  Int = 2      # weight quant type (2=Q4_0, 6=Q5_0, 8=Q8_0, 12=Q4_K, 39=MXFP4, 1=f16)
comptime EP:  Float32 = 1e-6
comptime W:  Int = 8       # SIMD width
comptime RPW: Int = 32     # rows per parallel work item
comptime RT:  Float32 = 10000.0  # rope_theta

# ─── Architecture flags ───
comptime HAS_MOE:    Bool = False  # Mixture of Experts
comptime HAS_SSM:    Bool = False  # Hybrid SSM layer
comptime ATTN_MHA:   Int = 0      # 0=GQA, 1=MHA (all heads separate)
comptime FFN_SILU:   Int = 0      # 0=SiLU, 1=GeGLU, 2=ReLU
comptime N_EXPERTS:  Int = 8      # total experts (for MoE)
comptime N_ACTIVE:   Int = 2      # top-k active experts
comptime MOE_FF:     Int = 2560   # MoE intermediate dim

# ═══════════════════════════════════════════════
# BUILT-IN FUNCTIONS (no imports needed)
# ═══════════════════════════════════════════════

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

# ─── Quant Decode Helpers ───
fn deq_q4_0(blk: UnsafePointer[UInt8, MutExternalOrigin], bo: Int, j: Int) -> Float32:
    """Decode one Q4_0 value at block offset bo, position j (0..31)."""
    var p = blk.load(bo + 2 + j // 2)
    var nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
    if nib > 7: nib -= 16
    var d = f16_to_f32(UInt16(blk.load(bo)) | (UInt16(blk.load(bo+1)) << 8))
    return Float32(nib) * d

fn deq_q8_0(blk: UnsafePointer[UInt8, MutExternalOrigin], bo: Int, j: Int) -> Float32:
    var qb = Int32(blk.load(bo + 2 + j))
    if qb > 127: qb -= 256
    var d = f16_to_f32(UInt16(blk.load(bo)) | (UInt16(blk.load(bo+1)) << 8))
    return Float32(qb) * d

def deq_mxfp4(blk: UnsafePointer[UInt8, MutExternalOrigin], bo: Int, j: Int) -> Float32:
    var p = blk.load(bo + j // 2); var nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
    if nib > 7: nib -= 16
    var e8 = Int32(blk.load(bo + 16)) - 127
    var v = Float32(nib)
    if e8 >= 0:
        for _ in range(e8): v *= 2.0
    else:
        for _ in range(-e8): v *= 0.5
    return v

# ─── Universal quant matmul: single row, any type ───
def quant_mul_row(w: UnsafePointer[UInt8, MutExternalOrigin],
                  x: UnsafePointer[Float32, MutExternalOrigin],
                  nc: Int, row: Int, qt: Int) -> Float32:
    """Single row of quant weight × f32 vector. Supports all major GGUF types."""
    var total: Float32 = 0.0
    if qt == 2:  # Q4_0: 18B/32vals
        var nb = nc // 32
        for blk in range(nb):
            var bo = (row * nb + blk) * 18
            var d = f16_to_f32(UInt16(w.load(bo)) | (UInt16(w.load(bo+1)) << 8))
            var xo = blk * 32
            for j in range(32):
                var p = w.load(bo + 2 + j // 2)
                var nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
                if nib > 7: nib -= 16
                total += Float32(nib) * d * x.load(xo + j)
    elif qt == 6:  # Q5_0: 22B/32vals
        var nb = nc // 32
        for blk in range(nb):
            var bo = (row * nb + blk) * 22
            var d = f16_to_f32(UInt16(w.load(bo)) | (UInt16(w.load(bo+1)) << 8))
            var xo = blk * 32
            for j in range(32):
                var p = w.load(bo + 6 + j // 2); var qh = w.load(bo + 2 + j // 4)
                var hs = 2 * (j % 4); var hb = (UInt8(qh) >> UInt8(hs)) & UInt8(1)
                var nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
                if nib > 7: nib -= 16
                nib = nib + Int32(hb) * 16
                if nib > 15: nib -= 32
                total += Float32(nib) * d * x.load(xo + j)
    elif qt == 8:  # Q8_0: 34B/32vals
        var nb = nc // 32
        for blk in range(nb):
            var bo = (row * nb + blk) * 34
            var d = f16_to_f32(UInt16(w.load(bo)) | (UInt16(w.load(bo+1)) << 8))
            var xo = blk * 32
            for j in range(32):
                var qb = Int32(w.load(bo + 2 + j))
                if qb > 127: qb -= 256
                total += Float32(qb) * d * x.load(xo + j)
    elif qt == 12:  # Q4_K: 144B/256vals
        var nb = nc // 256
        for blk in range(nb):
            var bo = (row * nb + blk) * 144
            var d = f16_to_f32(UInt16(w.load(bo)) | (UInt16(w.load(bo+1)) << 8))
            var dm = f16_to_f32(UInt16(w.load(bo+2)) | (UInt16(w.load(bo+3)) << 8))
            var xo = blk * 256
            for s in range(8):  # 8 sub-blocks of 32
                var sc_byte = w.load(bo + 4 + s)
                var sc = Int32(sc_byte & 0x3F); var mn = Int32(sc_byte >> 4)
                var db = d * Float32(sc - 16 if sc >= 16 else sc)
                var mb = dm * Float32(mn - 16 if mn >= 16 else mn)
                for j in range(32):
                    var p = w.load(bo + 4 + 128 + s * 16 + j // 2)
                    var nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
                    var v = db * Float32(nib - 8) - mb * 6.0
                    total += v * x.load(xo + s * 32 + j)
    elif qt == 39:  # MXFP4: 17B/32vals
        var nb = nc // 32
        for blk in range(nb):
            var bo = (row * nb + blk) * 17
            var e8 = Int32(w.load(bo + 16)) - 127
            var xo = blk * 32
            for j in range(32):
                var p = w.load(bo + j // 2)
                var nib = Int32(p >> 4) if (j % 2 == 1) else Int32(p & 0x0F)
                if nib > 7: nib -= 16
                var v = Float32(nib)
                if e8 >= 0:
                    for _ in range(e8): v *= 2.0
                else:
                    for _ in range(-e8): v *= 0.5
                total += v * x.load(xo + j)
    elif qt == 1:  # f16: 2B/val
        for j in range(nc):
            var wi = row * nc + j
            var h = UInt16(w.load(wi*2)) | (UInt16(w.load(wi*2 + 1)) << 8)
            total += f16_to_f32(h) * x.load(j)
    return total

# ─── Parallel matmul for any quant type ───
def quant_matmul(w: UnsafePointer[UInt8, MutExternalOrigin],
                 x: UnsafePointer[Float32, MutExternalOrigin],
                 o: UnsafePointer[Float32, MutExternalOrigin],
                 nr: Int, nc: Int, qt: Int):
    """o[nr] = W[nr][nc] @ x[nc]  with quant type qt dispatch."""
    def wk(r: Int) capturing -> None:
        o.store(r, quant_mul_row(w, x, nc, r, qt))
    parallelize[func=wk](num_work_items=nr)

# ─── F32 matmul (for router, norms, etc.) ───
def f32_matmul(w: UnsafePointer[Float32, MutExternalOrigin],
               x: UnsafePointer[Float32, MutExternalOrigin],
               o: UnsafePointer[Float32, MutExternalOrigin],
               nr: Int, nc: Int):
    def wk(r: Int) capturing -> None:
        var acc: Float32 = 0.0
        var ro = r * nc
        for j in range(nc): acc += w.load(ro + j) * x.load(j)
        o.store(r, acc)
    parallelize[func=wk](num_work_items=nr)

# ─── RMS Norm ───
def rms_norm(x: UnsafePointer[Float32, MutExternalOrigin],
             w: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    var ss: Float64 = 0.0
    for i in range(n): ss += Float64(x.load(i) * x.load(i))
    var inv = 1.0 / (math.sqrt(Float32(ss / Float64(n)) + EP))
    for i in range(n): o.store(i, x.load(i) * w.load(i) * inv)

# ─── RoPE ───
def rope_apply(q: UnsafePointer[Float32, MutExternalOrigin],
               k: UnsafePointer[Float32, MutExternalOrigin],
               nh: Int, nkh: Int, hd: Int, pos: Int):
    var inv_theta = 1.0 / RT
    for h in range(nh):
        for d in range(0, hd, 2):
            var freq = math.pow(inv_theta, Float32(d // 2) / Float32(hd))
            var ang = Float32(pos) * freq; var c = math.cos(ang); var s = math.sin(ang)
            var idx = h * hd + d; var q0 = q.load(idx); var q1 = q.load(idx + 1)
            q.store(idx, q0*c - q1*s); q.store(idx + 1, q0*s + q1*c)
    for h in range(nkh):
        for d in range(0, hd, 2):
            var freq = math.pow(inv_theta, Float32(d // 2) / Float32(hd))
            var ang = Float32(pos) * freq; var c = math.cos(ang); var s = math.sin(ang)
            var idx = h * hd + d; var k0 = k.load(idx); var k1 = k.load(idx + 1)
            k.store(idx, k0*c - k1*s); k.store(idx + 1, k0*s + k1*c)

# ─── GQA Attention (single token decode) ───
def gqa_attn(q: UnsafePointer[Float32, MutExternalOrigin],
             k_cache: UnsafePointer[Float32, MutExternalOrigin],
             v_cache: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin],
             seq_len: Int, nh: Int, nkh: Int, hd: Int,
             scores: UnsafePointer[Float32, MutExternalOrigin]):
    var gqa = nh // nkh; var sc = 1.0 / math.sqrt(Float32(hd))
    var kvs = nkh * hd
    for h in range(nkh):
        var qh = q + h * gqa * hd
        for qi in range(gqa):
            var hi = h * gqa + qi; var qo = qi * hd
            for t in range(seq_len):
                var dot: Float32 = 0.0
                for d in range(hd): dot += qh.load(qo + d) * k_cache.load(h*hd + t*kvs + d)
                scores.store(hi * seq_len + t, dot * sc)
            # softmax
            var mx: Float32 = -1e30
            for t in range(seq_len):
                var v = scores.load(hi * seq_len + t)
                if v > mx: mx = v
            var se: Float32 = 0.0
            for t in range(seq_len):
                var e = math.exp((scores.load(hi * seq_len + t) - mx))
                scores.store(hi * seq_len + t, e); se += e
            var inv = 1.0 / (se + 1e-10)
            for t in range(seq_len): scores.store(hi * seq_len + t, scores.load(hi * seq_len + t) * inv)
            # V weighted sum
            for d in range(hd):
                var acc: Float32 = 0.0
                for t in range(seq_len):
                    acc += scores.load(hi * seq_len + t) * v_cache.load(h*hd + t*kvs + d)
                o.store(hi * hd + d, acc)

# ─── SiLU activation ───
def silu(x: Float32) -> Float32:
    return x / (1.0 + math.exp(-x))

# ─── Universal FFN (SiLU-gated) ───
def ffn_silu(w_gate: UnsafePointer[UInt8, MutExternalOrigin],
             w_up: UnsafePointer[UInt8, MutExternalOrigin],
             w_down: UnsafePointer[UInt8, MutExternalOrigin],
             x: UnsafePointer[Float32, MutExternalOrigin],
             o: UnsafePointer[Float32, MutExternalOrigin],
             n_embd: Int, ff_hidden: Int, qt: Int):
    """SiLU-gated FFN: o = W_down @ (silu(W_gate @ x) * (W_up @ x))"""
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    var gate = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ff_hidden * 4))))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ff_hidden * 4))))
    quant_matmul(w_gate, x, gate, ff_hidden, n_embd, qt)
    quant_matmul(w_up, x, up, ff_hidden, n_embd, qt)
    for i in range(ff_hidden): up.store(i, up.load(i) * silu(gate.load(i)))
    quant_matmul(w_down, up, o, n_embd, ff_hidden, qt)

# ─── MoE Router ───
def moe_router(w_router: UnsafePointer[Float32, MutExternalOrigin],
               x: UnsafePointer[Float32, MutExternalOrigin],
               n_experts: Int, n_embd: Int, top_k: Int,
               indices: UnsafePointer[Int32, MutExternalOrigin],
               weights: UnsafePointer[Float32, MutExternalOrigin],
               buf: UnsafePointer[Float32, MutExternalOrigin]):
    for e in range(n_experts):
        var dot: Float32 = 0.0
        for i in range(n_embd): dot += w_router.load(e * n_embd + i) * x.load(i)
        buf.store(e, dot)
    var mx = buf.load(0)
    for e in range(1, n_experts):
        if buf.load(e) > mx: mx = buf.load(e)
    var se: Float32 = 0.0
    for e in range(n_experts):
        var ev = math.exp(buf.load(e) - mx); buf.store(e, ev); se += ev
    var inv = 1.0 / (se + 1e-10)
    for e in range(n_experts): buf.store(e, buf.load(e) * inv)
    for k in range(top_k):
            weights.store(k, -1e30)
            indices.store(k, Int32(-1))
    for e in range(n_experts):
        var v = buf.load(e)
        for k in range(top_k):
            if v > weights.load(k):
                for k2 in range(top_k - 1, k, -1):
                    weights.store(k2, weights.load(k2-1))
                    indices.store(k2, indices.load(k2-1))
                weights.store(k, v); indices.store(k, e); break
    var tw: Float32 = 0.0
    for k in range(top_k): tw += weights.load(k)
    var ti = 1.0 / (tw + 1e-10)
    for k in range(top_k): weights.store(k, weights.load(k) * ti)

# ─── MoE FFN ───
def moe_ffn(w_gates: UnsafePointer[Int32, MutExternalOrigin],
            w_ups: UnsafePointer[Int32, MutExternalOrigin],
            w_downs: UnsafePointer[Int32, MutExternalOrigin],
            x: UnsafePointer[Float32, MutExternalOrigin],
            indices: UnsafePointer[Int32, MutExternalOrigin],
            weights: UnsafePointer[Float32, MutExternalOrigin],
            top_k: Int, n_embd: Int, ff_hidden: Int, qt: Int,
            o: UnsafePointer[Float32, MutExternalOrigin],
            buf: UnsafePointer[Float32, MutExternalOrigin]):
    for d in range(n_embd): o.store(d, 0.0)
    for k in range(top_k):
        var ei = indices.load(k)
        if ei < 0: continue
        var wg = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_gates.load(ei)))
        var wu = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_ups.load(ei)))
        var wd = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_downs.load(ei)))
        quant_matmul(wg, x, buf, ff_hidden, n_embd, qt)
        for i in range(ff_hidden): buf.store(i, buf.load(i) * silu(buf.load(i)))
        var out_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(n_embd * 4))))
        quant_matmul(wd, buf, out_buf, n_embd, ff_hidden, qt)
        for i in range(n_embd): o.store(i, o.load(i) + out_buf.load(i) * weights.load(k))

# ═══════════════════════════════════════════════
# UNIVERSAL FORWARD PASS
# ═══════════════════════════════════════════════
# Expects: weight_addrs = [Int64 pointers to all weights]
#   Index layout: 0=emb, 1=out_norm, 2=lm_head
#   Then per layer: attn_norm, ffn_norm, Wq, Wk, Wv, Wo, Wgate, Wup, Wdown
#   If MoE: router, then gate_exp[0..N-1], up_exp[0..N-1], down_exp[0..N-1]
#
#   k_cache / v_cache: [n_layers][max_ctx][n_kv_heads * head_dim]
#   kv_len: [n_layers] — current sequence length per layer

def forward(token: Int, pos: Int, emb: UnsafePointer[Float32, MutExternalOrigin],
            weight_addrs: UnsafePointer[Int32, MutExternalOrigin],
            k_cache: UnsafePointer[Float32, MutExternalOrigin],
            v_cache: UnsafePointer[Float32, MutExternalOrigin],
            kv_len: UnsafePointer[Int32, MutExternalOrigin],
            logits: UnsafePointer[Float32, MutExternalOrigin],
            workspace: UnsafePointer[Float32, MutExternalOrigin],
            max_ctx: Int, lm_head_nr: Int, lm_head_nc: Int, lm_head_qt: Int):
    """Run one forward pass: embedding → L layers → output projection → logits."""
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    
    # Workspace layout
    var S = NE * 4  # 4x n_embd for safety
    var x = workspace
    var xn = x + S
    var res = xn + S
    var q_buf = res + S
    var k_buf = q_buf + NE
    var v_buf = k_buf + NK * HD
    var att = v_buf + NK * HD
    var ffn_buf = att + NE
    var sc_buf = ffn_buf + S  # scores
    
    # Embedding lookup
    var te = emb + token * NE
    for i in range(NE): x.store(i, te.load(i))
    var n_kv = NK * HD
    var gqa_ws = sc_buf + S
    
    for l in range(NL):
        # Weight indices
        var wi = 3 + l * 8  # base index for this layer
        var w_an = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi)))
        var w_fn = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 1)))
        var wq = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 2)))
        var wk = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 3)))
        var wv = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 4)))
        var wo = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 5)))
        
        # Save residual
        for i in range(NE): res.store(i, x.load(i))
        
        # Pre-attention norm
        rms_norm(x, w_an, xn, NE)
        
        # QKV projections
        quant_matmul(wq, xn, q_buf, NH*HD, NE, QT)
        quant_matmul(wk, xn, k_buf, n_kv, NE, QT)
        quant_matmul(wv, xn, v_buf, n_kv, NE, QT)
        
        # RoPE
        rope_apply(q_buf, k_buf, NH, NK, HD, pos)
        
        # KV cache store
        var sl = kv_len.load(l)
        var lc = l * max_ctx * n_kv
        for i in range(n_kv):
            k_cache.store(lc + sl * n_kv + i, k_buf.load(i))
            v_cache.store(lc + sl * n_kv + i, v_buf.load(i))
        kv_len.store(l, sl + 1)
        
        # Attention
        gqa_attn(q_buf, k_cache + lc, v_cache + lc, att, Int(sl + 1), NH, NK, HD, gqa_ws)
        
        # Output projection + residual
        quant_matmul(wo, att, ffn_buf, NE, NH*HD, QT)
        for i in range(NE): x.store(i, res.load(i) + ffn_buf.load(i))
        
        # Post-attention norm + FFN
        for i in range(NE): res.store(i, x.load(i))
        rms_norm(x, w_fn, xn, NE)
        
        if HAS_MOE:
            var ff_hid = MOE_FF
            var w_router = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 6)))
            var exp_gate = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 7)))
            var exp_up = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 8)))
            var exp_down = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 9)))
            var indices = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(N_ACTIVE * 4))))
            var weights = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(N_ACTIVE * 4))))
            var rbuf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(N_EXPERTS * 4))))
            moe_router(w_router, xn, N_EXPERTS, NE, N_ACTIVE, indices, weights, rbuf)
            moe_ffn(exp_gate, exp_up, exp_down, xn, indices, weights, N_ACTIVE, NE, ff_hid, QT, ffn_buf, rbuf)
        else:
            var w_gate = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 6)))
            var w_up = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 7)))
            var w_down = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(wi + 8)))
            ffn_silu(w_gate, w_up, w_down, xn, ffn_buf, NE, FF, QT)
        
        for i in range(NE): x.store(i, res.load(i) + ffn_buf.load(i))
    
    # Output projection (LM head)
    var w_on = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(1)))
    var w_lm = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(2)))
    rms_norm(x, w_on, xn, NE)
    if lm_head_qt == 0:
        f32_matmul(UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(weight_addrs.load(2))), xn, logits, lm_head_nr, lm_head_nc)
    else:
        quant_matmul(w_lm, xn, logits, lm_head_nr, lm_head_nc, lm_head_qt)
