# MojoLlama Engine Bench — Universal (one file, all models)
# WHAT:  One Mojo benchmark for ALL model types: DENSE, MoE, HYBRID.
#        Supports REAL WEIGHTS mode (TinyLlama from /tmp/tinyllama/)
#        and COLD-CACHE mode (any architecture, synthetic weights).
# WHY:   Single file — no Python, no C, no llama.cpp.
# WHEN:  May 2026 — universal engine bench v3.
from std import time, math
from std.math import sqrt, exp
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag

# ═══ MODEL CONFIG — change comptime for any model ═══
comptime NE: Int = 2048       # n_embd
comptime NH: Int = 32         # n_heads
comptime NK: Int = 4          # n_kv_heads
comptime HD: Int = 64         # head_dim
comptime NL: Int = 22         # n_layers
comptime FF: Int = 5632       # ff_hidden
comptime NV: Int = 32000      # vocab_size

# MoE (set True for MoE models)
comptime IS_MOE: Bool = False
comptime N_EXP: Int = 1       # num_experts
comptime N_ACT: Int = 1       # experts per token
comptime FF_EXP: Int = 512    # per-expert intermediate
# Layer pattern: 0=alternating(ZAYA), 1=Qwen hybrid(attn every 4th)
comptime LAYER_PATTERN: Int = 0

# Run config
comptime NW: Int = 24
comptime RPW: Int = 32; comptime W: Int = 8; comptime EP: Float32 = 1e-6
comptime QUANT_TYPE: Int = 1  # 1=f16, 39=MXFP4
# Real weights: set True to load from /tmp/tinyllama/ (TinyLlama only)
comptime REAL_WEIGHTS: Bool = False

# I/O (used only when REAL_WEIGHTS=True)
comptime O_RDONLY: Int = 0; comptime SEEK_END: Int = 2; comptime SEEK_SET: Int = 0
comptime Q4_0_BS: Int = 18; comptime N_BASE: Int = 3; comptime N_LF: Int = 9

@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...
@extern("free")
def _c_free(p: Int64) abi("C") -> None: ...
@extern("open")
def _open(path: UnsafePointer[UInt8, MutExternalOrigin], flags: Int) abi("C") -> Int: ...
@extern("read")
def _read(fd: Int, buf: UnsafePointer[UInt8, MutExternalOrigin], cnt: Int64) abi("C") -> Int64: ...
@extern("lseek")
def _lseek(fd: Int, off: Int64, whence: Int) abi("C") -> Int64: ...
@extern("close")
def _close(fd: Int) abi("C") -> Int: ...

def str_to_cptr(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var blen = s.byte_length(); var buf = alloc[UInt8](blen + 1)
    var src = s.unsafe_ptr()
    for i in range(blen): buf.store(i, src.load(i))
    buf.store(blen, UInt8(0)); return buf

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

# ═══ Matmul dispatch ═══
def matmul(w_addr: Int64, x: UnsafePointer[Float32, MutExternalOrigin],
           o: UnsafePointer[Float32, MutExternalOrigin], nr: Int, nc: Int, nw: Int):
    if w_addr == 0: return
    @parameter
    if QUANT_TYPE == 1:
        var w = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(w_addr))
        var bpr = nc // 32; var nb = (nr + RPW - 1) // RPW
        def wk_f16(b: Int) capturing -> None:
            var rs = b * RPW; var re = rs + RPW
            if re > nr: re = nr
            for r in range(rs, re):
                var ro = r * nc; var acc = SIMD[DType.float32, W](0.0)
                for blk in range(bpr):
                    comptime for grp in range(4):
                        var w16 = w.load[width=W](ro + blk * 32 + grp * 8)
                        var wf32 = w16.cast[DType.float32]()
                        var xv = x.load[width=W](blk * 32 + grp * 8)
                        acc = wf32.fma[FastMathFlag.FAST](xv, acc)
                o.store(r, acc.reduce_add())
        parallelize[func=wk_f16](num_work_items=nb, num_workers=nw)
    @parameter
    if QUANT_TYPE == 39:
        var w = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_addr))
        var bpr = nc // 32; var bs = 17
        def wk(b: Int) capturing -> None:
            var rs = b * RPW; var re = rs + RPW
            if re > nr: re = nr
            for r in range(rs, re):
                var ro = r * bpr * bs; var acc: Float32 = 0.0
                for blk in range(bpr):
                    var bo = ro + blk * bs
                    var eb = w.load(bo + 16)
                    var sf: Float32 = 0.0
                    if eb != 0:
                        if eb < 255:
                            var ei = Int(eb) - 127; sf = 1.0
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
                        ai += lo * Int32(x.load(blk*32 + j*2)) + hi * Int32(x.load(blk*32 + j*2+1))
                    acc += Float32(ai) * sf
                o.store(r, acc)
        parallelize[func=wk](num_work_items=nr, num_workers=nw)

@always_inline
fn bpr(nc: Int) -> Int:
    @parameter
    if QUANT_TYPE == 1: return nc // 32
    if QUANT_TYPE == 39: return nc // 32
    return nc // 32

@always_inline
fn bytes_per_row(nr: Int, nc: Int) -> Int:
    @parameter
    if QUANT_TYPE == 1: return nr * nc * 2
    if QUANT_TYPE == 39: return nr * (nc // 32) * 17
    return nr * nc * 2

# ═══ Fill pool (cold-cache mode) ═══
def fill_pool(pool: Int64, sz: Int64):
    var p = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(pool))
    for i in range(Int(sz)):
        p.store(i, UInt8((i*7 + (i>>4)*13 + (i>>8)*17) & 0xFF))

# ═══ Forward pass (runs once, returns ms) ═══
def run_fwd(wp_addr: Int64, pool_sz: Int64, inner: Int, kv_dim: Int, ni: Int, kr: Int,
            eff: Int, hp: UnsafePointer[Float32, MutExternalOrigin],
            bp: UnsafePointer[Float32, MutExternalOrigin],
            qp: UnsafePointer[Float32, MutExternalOrigin],
            kp: UnsafePointer[Float32, MutExternalOrigin],
            vp: UnsafePointer[Float32, MutExternalOrigin],
            gp: UnsafePointer[Float32, MutExternalOrigin],
            up: UnsafePointer[Float32, MutExternalOrigin],
            dp: UnsafePointer[Float32, MutExternalOrigin],
            use_real: Bool) -> Float64:
    
    var nq = inner; var nkv = kv_dim
    var t0 = time.perf_counter()
    
    @parameter
    if not IS_MOE:
        # Dense forward (all layers = attn + FFN)
        if not use_real:
            # Cold-cache synthetic
            var off: Int64 = 0; var szm = pool_sz - 10485760
            for layer in range(NL):
                matmul(wp_addr+(off%szm), qp, bp, nq, ni, NW); off += 1
                matmul(wp_addr+(off%szm), kp, bp, nkv, ni, NW); off += 1
                matmul(wp_addr+(off%szm), vp, bp, nkv, ni, NW); off += 1
                matmul(wp_addr+(off%szm), bp, qp, ni, nq, NW); off += 1
                matmul(wp_addr+(off%szm), gp, bp, FF, ni, NW); off += 1
                matmul(wp_addr+(off%szm), up, bp, FF, ni, NW); off += 1
                matmul(wp_addr+(off%szm), dp, gp, ni, FF, NW); off += 1
        if use_real:
            # Real weights from wp_addr array
            var wp_arr = UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(wp_addr))
            for layer in range(NL):
                var lw = N_BASE + layer * N_LF
                var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp_arr.load(lw)))
                var ss: Float64 = 0.0
                for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
                var inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
                for i in range(ni): bp.store(i, hp.load(i) * anp.load(i) * inv)
                matmul(wp_arr.load(lw+2), bp, qp, nq, ni, NW)
                matmul(wp_arr.load(lw+3), bp, kp, nkv, ni, NW)
                matmul(wp_arr.load(lw+4), bp, vp, nkv, ni, NW)
                for hq in range(NH):
                    var hkv = hq // kr; var sc: Float64 = 0.0
                    for d in range(HD): sc += Float64(qp.load(hq*HD+d)) * Float64(kp.load(hkv*HD+d))
                    var wt = Float32(sc / Float64(HD))
                    for d in range(HD): qp.store(hq*HD+d, vp.load(hkv*HD+d) * wt)
                matmul(wp_arr.load(lw+5), qp, bp, ni, nq, NW)
                for i in range(ni): hp.store(i, hp.load(i) + bp.load(i))
                var fnp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wp_arr.load(lw+1)))
                ss = 0.0
                for i in range(ni): ss += Float64(hp.load(i)) * Float64(hp.load(i))
                inv = Float32(1.0 / sqrt(Float64(ss) / Float64(ni) + Float64(EP)))
                for i in range(ni): bp.store(i, hp.load(i) * fnp.load(i) * inv)
                matmul(wp_arr.load(lw+6), bp, gp, FF, ni, NW)
                matmul(wp_arr.load(lw+7), bp, up, FF, ni, NW)
                for k in range(FF):
                    var gv = gp.load(k)
                    if gv > 80.0: gv = 80.0
                    if gv < -80.0: gv = -80.0
                    gp.store(k, (gv / Float32(1.0 + exp(Float64(-gv)))) * up.load(k))
                matmul(wp_arr.load(lw+8), gp, dp, ni, FF, NW)
                for i in range(ni): hp.store(i, hp.load(i) + dp.load(i))
    
    @parameter
    if IS_MOE:
        # MoE forward (cold-cache synthetic only)
        var off: Int64 = 0; var szm = pool_sz - 10485760
        for layer in range(NL):
            @parameter
            if LAYER_PATTERN == 0:  # alternating
                if layer % 2 == 0:
                    matmul(wp_addr+(off%szm), qp, bp, nq, ni, NW); off += 1
                    matmul(wp_addr+(off%szm), kp, bp, nkv, ni, NW); off += 1
                    matmul(wp_addr+(off%szm), vp, bp, nkv, ni, NW); off += 1
                    matmul(wp_addr+(off%szm), bp, qp, ni, nq, NW); off += 1
            @parameter
            if LAYER_PATTERN == 1:  # Qwen hybrid
                if layer % 4 == 3:
                    matmul(wp_addr+(off%szm), qp, bp, nq, ni, NW); off += 1
                    matmul(wp_addr+(off%szm), kp, bp, nkv, ni, NW); off += 1
                    matmul(wp_addr+(off%szm), vp, bp, nkv, ni, NW); off += 1
                    matmul(wp_addr+(off%szm), bp, qp, ni, nq, NW); off += 1
            for _ in range(N_ACT):
                matmul(wp_addr+(off%szm), gp, bp, eff, ni, NW); off += 1
                matmul(wp_addr+(off%szm), up, bp, eff, ni, NW); off += 1
                matmul(wp_addr+(off%szm), dp, gp, ni, eff, NW); off += 1
    
    return (time.perf_counter() - t0) * 1000.0

def p(s: String): print(s)

def main():
    var inner = NH * HD; var kv_dim = NK * HD; var ni = NE; var kr = NH // NK
    var eff = FF_EXP if IS_MOE else FF
    
    # ═══ REAL WEIGHTS MODE (load from /tmp/tinyllama/) ═══
    var wp_addr: Int64 = 0
    var use_real = False
    
    @parameter
    if REAL_WEIGHTS:
        use_real = True
        var nf = N_BASE + NL * N_LF
        wp_addr = _alc(Int64(nf * 8))
        var wp = UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(wp_addr))
        for i in range(nf): wp.store(i, 0)
        
        def wpath(idx: Int) -> String:
            var base = "/tmp/tinyllama/"
            if idx == 0: return base + "token_embd_weight.q4"
            if idx == 1: return base + "output_norm_weight.f32"
            if idx == 2: return base + "output_weight.q4"
            var li = (idx - N_BASE) // N_LF
            var ti = (idx - N_BASE) % N_LF
            var names = ["attn_norm_weight.f32","ffn_norm_weight.f32",
                         "attn_q_weight.q4","attn_k_weight.q4","attn_v_weight.q4",
                         "attn_output_weight.q4","ffn_gate_weight.q4",
                         "ffn_up_weight.q4","ffn_down_weight.q4"]
            return base + "blk_" + String(li) + "_" + names[ti]
        
        for idx in range(nf):
            var ps = wpath(idx)
            var path = str_to_cptr(ps)
            var fd = _open(path, O_RDONLY)
            path.free()
            if fd < 0: p("ERR:open "+String(idx)); return
            var is_q4 = ".q4" in ps
            var sz = _lseek(fd, 0, SEEK_END); _lseek(fd, 0, SEEK_SET)
            if is_q4:
                var raw = _alc(sz)
                _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw)), sz)
                _close(fd)
                var ru8 = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw))
                var total_vals = Int(sz // Int64(Q4_0_BS)) * 32
                var nr_files = total_vals // NE
                var f16_buf = _alc(Int64(nr_files) * Int64(NE) * 2)
                var f16p = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=Int(f16_buf))
                var bpr_total = Int(sz // Int64(Q4_0_BS))
                var bpr_row = bpr_total // nr_files
                for r in range(nr_files):
                    for blk in range(bpr_row):
                        var bo = (r * bpr_row + blk) * Q4_0_BS
                        var lo16 = UInt16(ru8.load(bo)); var hi16 = UInt16(ru8.load(bo+1))
                        var d = f16_to_f32(lo16 | (hi16 << 8))
                        for j in range(32):
                            var nib = Int32(ru8.load(bo+2+j//2))
                            if j % 2 == 0:
                                nib = nib & 0x0F
                            else:
                                nib = nib >> 4
                            if nib > 7: nib -= 16
                            var f32v = Float32(nib) * d
                            if f32v > 65000.0: f32v = 65000.0
                            if f32v < -65000.0: f32v = -65000.0
                            f16p.store(r*NE + blk*32 + j, Float16(f32v))
                _c_free(raw)
                wp.store(idx, f16_buf)
            else:
                var buf = _alc(sz)
                _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
                _close(fd)
                wp.store(idx, buf)
    
    # ═══ COLD-CACHE MODE (synthetic weights) ═══
    var pool: Int64 = 0
    var pool_sz = Int64(256 * 1048576)
    if not use_real:
        pool = _alc(pool_sz)
        fill_pool(pool, pool_sz)
    
    # Buffers
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni*4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni*4))))
    var qp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(inner*4))))
    var kp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(kv_dim*4))))
    var vp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(kv_dim*4))))
    var gp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(eff*4))))
    var up = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(eff*4))))
    var dp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(ni*4))))
    for i in range(ni): hp.store(i, Float32(i%100-50)*0.01)
    
    var model_type = "Dense"
    @parameter
    if IS_MOE:
        model_type = "MoE(" + String(LAYER_PATTERN) + ")"
    if use_real: model_type = model_type + "+real"
    
    p("{")
    p("\"engine_bench\":{")
    p("  \"model\":\"" + model_type + "\",")
    p("  \"arch\":{\"NE\":" + String(NE) + ",\"NH\":" + String(NH) + ",\"NK\":" + String(NK))
    p("  ,\"HD\":" + String(HD) + ",\"NL\":" + String(NL) + ",\"FF\":" + String(FF) + ",\"NV\":" + String(NV))
    @parameter
    if IS_MOE:
        p("  ,\"EXP\":" + String(N_EXP) + ",\"ACT\":" + String(N_ACT) + ",\"FF_EXP\":" + String(FF_EXP))
        p("  ,\"PATTERN\":" + String(LAYER_PATTERN))
    p("  },")
    
    # ═══ COMPONENT BREAKDOWN ═══
    p("  \"breakdown\":{")
    var src = pool if not use_real else wp_addr
    var off: Int64 = 0; var szm = pool_sz - 10485760
    
    # Run forward pass for measurement
    # Warmup
    for _ in range(1):
        run_fwd(src, pool_sz, inner, kv_dim, ni, kr, eff, hp, bp, qp, kp, vp, gp, up, dp, use_real)
    
    # Measured
    var total: Float64 = 0.0
    for _ in range(3):
        total += run_fwd(src, pool_sz, inner, kv_dim, ni, kr, eff, hp, bp, qp, kp, vp, gp, up, dp, use_real)
    var avg = total / 3.0
    var tok_s = 1000.0 / avg
    
    p("  },")
    p("  \"forward\":{\"ms\":" + String(Float64(avg)) + ",\"tok_s\":" + String(Float64(tok_s)) + "}")
    p("}}")
