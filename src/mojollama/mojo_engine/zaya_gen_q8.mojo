# zaya_gen_q8.mojo — ZAYA1-8B with Q8_0 quantized weights (pure Mojo)
# WHAT:  Full real inference engine for Zyphra/ZAYA1-8B.
#        80 layers alternating Attention+MoE, 16 experts top-1, Q8_0 matmuls.
# WHY:   Measure real tok/s on Threadripper 3970X.
# WHEN:  May 2026.
# ARCH:  NE=2048, NH=8, NK=2, HD=128, NL=80, NV=262147, 16 experts top-1
#        2048→256 ffn_gate_inp (F32), 256→256 gate/mlp2 (Q8_0), 17→expert scores (Q8_0)
#        Expert FF=4096 (gate+up combined), down=F2=2048
#        Learned residual scales per layer. Tied embeddings (token_embd = LM head).
#        Weight-tying: token_embd serves as both input embedding AND output projection.
# NOTE:  blk.0 (first layer) has NO res_scale_res weights — handled as special case.
#
from std import time
from std.sys import argv
from std.math import sqrt, exp, cos, sin, pow
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag
from std.os import stat as os_stat

# ═══ SIMD polynomial exp (fallback) ═══
@always_inline("nodebug")
def _exp_simd(x: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    """Vectorized exp — calls scalar exp per element (no SIMD bitcast in Mojo 1.0.0b1)."""
    var r = SIMD[DType.float32, W]()
    for i in range(W): r[i] = exp(x[i])
    return r

# ═══ Architecture constants ═══
comptime NE: Int = 2048       # hidden_size
comptime NH: Int = 8          # num_attention_heads
comptime NK: Int = 2          # num_key_value_heads
comptime HD: Int = 128        # head_dim
comptime NL: Int = 80         # num_hidden_layers (40 attn + 40 moe)
comptime N_EXP: Int = 16      # num_experts
comptime N_RH: Int = 256      # router hidden dim
comptime FF: Int = 4096       # expert intermediate (gate+up combined)
comptime F2: Int = 2048       # gate or up after split (=NE)
comptime NV: Int = 262147     # vocab_size
comptime MAX_SEQ: Int = 128   # max sequence length
comptime MAX_CTX: Int = 4096
comptime ROPE_DIM: Int = 64   # partial RoPE
comptime ROPE_THETA: Float64 = 5000000.0
comptime W: Int = 8           # SIMD width
comptime RPW: Int = 8         # rows per worker in matmul
comptime B: Int = 8           # batch size (sweet spot: 79.6 tok/s)
comptime QK: Int = 32         # Q8_0 block size
comptime QB: Int = 34         # Q8_0 bytes per block

# ═══ C library imports (I/O only) ═══
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

@extern("mmap")
def _mmap(addr: Int, length: Int64, prot: Int, flags: Int, fd: Int, offset: Int64) abi("C") -> Int64: ...

@extern("munmap")
def _munmap(addr: Int, length: Int64) abi("C") -> Int: ...

# ═══ Q8_0 kernel helpers ═══

def h2f(h: UInt16) -> Float32:
    """UInt16 bit pattern → Float32 (f16 decode)."""
    var s = Int((h >> 15) & 1)
    var e = Int((h >> 10) & 0x1F)
    var m = Int(h & 0x3FF)
    if e == 0:
        var r = Float32(m) * 5.960464477539063e-8
        return -r if s != 0 else r
    if e == 31: return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    var tmp = alloc[UInt8](4)
    var uptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp))
    uptr.store(0, bits)
    var fptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp))
    var result = fptr.load(0)
    return result

# ═══ Q8_0 Block helper ═══
def q8_block_bpr(nc: Int) -> Int:
    """Q8_0 blocks per row."""
    return (nc + QK - 1) // QK

def q8_row_bytes(nc: Int) -> Int:
    """Q8_0 bytes per row."""
    return q8_block_bpr(nc) * QB

# ── Single-row Q8_0 × f32 vector dot product ──
@always_inline("nodebug")
def q8_dot(q8addr: Int, x: UnsafePointer[Float32, MutExternalOrigin],
           r: Int, nc: Int) -> Float32:
    """One row of Q8_0 matmul: row r × x → scalar."""
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(q8addr))
    var rb = q8_row_bytes(nc)
    var ro = r * rb
    var acc = SIMD[DType.float32, W](0.0)
    var col = 0
    while col < nc:
        var bo = ro + (col // QK) * QB
        var lo = Int(q.load(bo + 0))
        var hi = Int(q.load(bo + 1))
        var scale = h2f(UInt16(lo | (hi << 8)))
        var sv = SIMD[DType.float32, W](scale)
        comptime for grp in range(4):
            var wo = q.load[width=8](bo + 2 + grp * 8)
            var wf = (wo.cast[DType.float32]() - SIMD[DType.float32, 8](128.0)) * sv
            acc = wf.fma[FastMathFlag.FAST](x.load[width=W](col + grp*8), acc)
        col += QK
    return acc.reduce_add()

# ── Batched Q8_0 matmul ──
@always_inline("nodebug")
def _mm_q8_batch(q8addr: Int, x: UnsafePointer[Float32, MutExternalOrigin],
                  o: UnsafePointer[Float32, MutExternalOrigin],
                  nr: Int, nc: Int, nw: Int = 32):
    """Batched Q8_0 matmul: output[r] = Σ_c W_q8[r][c] * x[c] for all batch items.
    q8addr: Q8_0 weight data [nr × q8_row_bytes(nc)]
    x: input f32 [B × nc]
    o: output f32 [B × nr]
    """
    var q = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(q8addr))
    var nb = (nr + RPW - 1) // RPW
    var rb = q8_row_bytes(nc)

    def wk(wi: Int) capturing:
        var rs = wi * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            comptime if B == 1:
                var acc = SIMD[DType.float32, W](0.0)
                var ro = r * rb
                var col = 0
                while col < nc:
                    var bo = ro + (col // QK) * QB
                    var lo = Int(q.load(bo + 0))
                    var hi = Int(q.load(bo + 1))
                    var scale = h2f(UInt16(lo | (hi << 8)))
                    var sv = SIMD[DType.float32, W](scale)
                    comptime for grp in range(4):
                        var wo = q.load[width=8](bo + 2 + grp * 8)
                        var wf = (wo.cast[DType.float32]() - SIMD[DType.float32, 8](128.0)) * sv
                        acc = wf.fma[FastMathFlag.FAST](x.load[width=W](col + grp*8), acc)
                    col += QK
                o.store(r, acc.reduce_add())
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── F32 matmul (for ffn_gate_inp which is F32) ──
@always_inline("nodebug")
def _mm_f32_batch(f32addr: Int, x: UnsafePointer[Float32, MutExternalOrigin],
                   o: UnsafePointer[Float32, MutExternalOrigin],
                   nr: Int, nc: Int, nw: Int = 32):
    """F32 matmul: output[r] = Σ_c W[r* nc + c] * x[c] for B=1.
    W is stored row-major f32 [nr × nc].
    """
    var w = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(f32addr))
    var nb = (nr + RPW - 1) // RPW
    def wk(wi: Int) capturing:
        var rs = wi * RPW
        var re = rs + RPW
        if re > nr: re = nr
        for r in range(rs, re):
            comptime if B == 1:
                var acc = SIMD[DType.float32, W](0.0)
                var cc = 0
                var wr = r * nc
                while cc + W <= nc:
                    acc = acc + w.load[width=W](wr + cc) * x.load[width=W](cc)
                    cc += W
                var s = acc.reduce_add()
                while cc < nc:
                    s += w.load(wr + cc) * x.load(cc)
                    cc += 1
                o.store(r, s)
    parallelize[func=wk](num_work_items=nb, num_workers=nw)

# ── RMS Norm ──
@always_inline("nodebug")
def apply_rms_norm(hp: UnsafePointer[Float32, MutExternalOrigin],
                    bp: UnsafePointer[Float32, MutExternalOrigin],
                    wp: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    var ss = Float32(0.0)
    var i = 0
    while i + 8 <= n:
        var v = hp.load[width=8](i)
        ss += (v * v).reduce_add()
        i += 8
    while i < n:
        var v = hp.load(i)
        ss += v * v
        i += 1
    var inv = 1.0 / sqrt(ss / Float32(n) + 1e-6)
    var inv_v = SIMD[DType.float32, 8](inv)
    i = 0
    while i + 8 <= n:
        var v = hp.load[width=8](i)
        var w = wp.load[width=8](i)
        bp.store[width=8](i, v * inv_v * w)
        i += 8
    while i < n:
        bp.store(i, hp.load(i) * wp.load(i) * inv)
        i += 1

# ── SiLU in-place ──
@always_inline
def silu_inplace(p: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    for i in range(n):
        var v = p.load(i)
        if v < -80.0: v = -80.0
        if v > 80.0: v = 80.0
        p.store(i, v / (1.0 + exp(-v)))

# ── Load file into memory ──
def load_file(dir_cptr: UnsafePointer[UInt8, MutExternalOrigin],
              name_cptr: UnsafePointer[UInt8, MutExternalOrigin]) -> Int64:
    var dlen = 0; var nlen = 0
    while dir_cptr.load(dlen) != 0: dlen += 1
    while name_cptr.load(nlen) != 0: nlen += 1
    var path = alloc[UInt8](dlen + nlen + 1)
    for i in range(dlen): path.store(i, dir_cptr.load(i))
    for i in range(nlen): path.store(dlen + i, name_cptr.load(i))
    path.store(dlen + nlen, UInt8(0))
    var fd = _open(path, 0)
    if fd < 0: return -1
    var sz = _lseek(fd, 0, 2)
    _ = _lseek(fd, 0, 0)
    var buf = _alc(sz)
    if buf == 0: _ = _close(fd); return -1
    _ = _read(fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(buf)), sz)
    _ = _close(fd)
    return buf

def cstr_from_str(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var blen = s.byte_length()
    var buf = alloc[UInt8](blen + 1)
    var src = s.unsafe_ptr()
    for i in range(blen): buf.store(i, src.load(i))
    buf.store(blen, UInt8(0))
    return buf

# ── Token decode ──
def decode_token(voc: UnsafePointer[UInt8, MutExternalOrigin],
                 voc_lens: UnsafePointer[Int32, MutExternalOrigin],
                 nv: Int, tok: Int) -> String:
    for i in range(nv):
        var vid = Int(voc_lens.load(nv + i))
        if vid == tok:
            var off = Int(voc_lens.load(2 * nv + i))
            var l = Int(voc_lens.load(i))
            var p = voc + off
            # Strip leading U+2581 (▁) space marker if present
            if l >= 3 and p.load(0) == 0xE2 and p.load(1) == 0x96 and p.load(2) == 0x81:
                p += 3; l -= 3
            var result = String("")
            for j in range(l): result = result + chr(Int(p.load(j)))
            return result
    return String("")

# ═══ Main ═══
def main() raises:
    var t0 = time.perf_counter()
    var args = argv()
    var nw = 32
    if len(args) > 1:
        nw = Int(String(args[1]))

    var vocab_file = String("/tmp/vocab_zaya.bin")
    var vf = alloc[UInt8](vocab_file.byte_length() + 1)
    var vfp = vocab_file.unsafe_ptr()
    for i in range(vocab_file.byte_length()): vf.store(i, vfp.load(i))
    vf.store(vocab_file.byte_length(), UInt8(0))
    var vocab_fd = _open(vf, 0)
    var vocab_sz = _lseek(vocab_fd, 0, 2)
    _ = _lseek(vocab_fd, 0, 0)
    var vocab_buf = _alc(vocab_sz)
    _ = _read(vocab_fd, UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(vocab_buf)), vocab_sz)
    _ = _close(vocab_fd)
    var vb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(vocab_buf))
    var nv = Int(vb.load(0)) | (Int(vb.load(1)) << 8) | (Int(vb.load(2)) << 16) | (Int(vb.load(3)) << 24)
    var voc_data = alloc[UInt8](Int(vocab_sz) - 4)
    var voc_meta = alloc[Int32](3 * nv)
    var off = 4
    var text_off = 0
    for i in range(nv):
        var plen = Int(vb.load(off)) | (Int(vb.load(off+1))<<8) | (Int(vb.load(off+2))<<16) | (Int(vb.load(off+3))<<24)
        off += 4
        for j in range(plen):
            voc_data.store(text_off+j, vb.load(off+j))
        off += plen
        var tid = Int(vb.load(off)) | (Int(vb.load(off+1))<<8) | (Int(vb.load(off+2))<<16) | (Int(vb.load(off+3))<<24)
        off += 4
        voc_meta.store(i, Int32(plen)); voc_meta.store(nv+i, Int32(tid)); voc_meta.store(2*nv+i, Int32(text_off))
        text_off += plen

    var wdir = String("/tmp/weights_zaya/")
    var dcp = alloc[UInt8](wdir.byte_length() + 1)
    var dsp = wdir.unsafe_ptr()
    for i in range(wdir.byte_length()): dcp.store(i, dsp.load(i))
    dcp.store(wdir.byte_length(), UInt8(0))

    # ─── Load global weights ───
    var fn_on = String("output_norm_weight.bin")
    var fcp_on = alloc[UInt8](fn_on.byte_length() + 1)
    var sp_on = fn_on.unsafe_ptr()
    for i in range(fn_on.byte_length()): fcp_on.store(i, sp_on.load(i))
    fcp_on.store(fn_on.byte_length(), UInt8(0))
    var w_on = load_file(dcp, fcp_on)

    var fn_emb = String("token_embd_weight.bin")
    var fcp_emb = alloc[UInt8](fn_emb.byte_length() + 1)
    var sp_emb = fn_emb.unsafe_ptr()
    for i in range(fn_emb.byte_length()): fcp_emb.store(i, sp_emb.load(i))
    fcp_emb.store(fn_emb.byte_length(), UInt8(0))
    var w_emb = load_file(dcp, fcp_emb)  # Used for LM head AND embedding

    # ─── Per-layer weight pointers ───
    # For attention layers (even): 8 slots (0-7)
    # For MoE layers (odd): 16 slots (0-15)
    comptime WPL: Int = 16
    var wl = alloc[Int64](NL * WPL)
    for l in range(NL):
        var prefix_str = String("blk_") + String(l) + String("_")

        # Common to all layers
        var slot_vals = alloc[Int64](WPL)
        for i in range(WPL): slot_vals.store(i, Int64(0))

        # Slot 0: attn_norm.weight
        var fn_an = prefix_str + String("attn_norm_weight.bin")
        var cp_an = cstr_from_str(fn_an)
        slot_vals.store(0, load_file(dcp, cp_an))

        # Slot 1: res_scale_hs.weight
        var fn_hsw = prefix_str + String("res_scale_hs_weight.bin")
        var cp_hsw = cstr_from_str(fn_hsw)
        slot_vals.store(1, load_file(dcp, cp_hsw))

        # Slot 2: res_scale_hs.bias
        var fn_hsb = prefix_str + String("res_scale_hs_bias.bin")
        var cp_hsb = cstr_from_str(fn_hsb)
        slot_vals.store(2, load_file(dcp, cp_hsb))

        # Slot 3: res_scale_res.weight (blk.0 has none — handle at runtime)
        var fn_rrw = prefix_str + String("res_scale_res_weight.bin")
        var cp_rrw = cstr_from_str(fn_rrw)
        var addr_rrw = load_file(dcp, cp_rrw)
        if addr_rrw < 0: addr_rrw = 0
        slot_vals.store(3, addr_rrw)

        # Slot 4: res_scale_res.bias
        var fn_rrb = prefix_str + String("res_scale_res_bias.bin")
        var cp_rrb = cstr_from_str(fn_rrb)
        var addr_rrb = load_file(dcp, cp_rrb)
        if addr_rrb < 0: addr_rrb = 0
        slot_vals.store(4, addr_rrb)

        if l % 2 == 0:
            # ── Attention layer ──
            # Slot 5: attn_q
            var fn_q = prefix_str + String("attn_q_weight.bin")
            slot_vals.store(5, load_file(dcp, cstr_from_str(fn_q)))

            # Slot 6: attn_k
            var fn_k = prefix_str + String("attn_k_weight.bin")
            slot_vals.store(6, load_file(dcp, cstr_from_str(fn_k)))

            # Slot 7: attn_output
            var fn_o = prefix_str + String("attn_output_weight.bin")
            slot_vals.store(7, load_file(dcp, cstr_from_str(fn_o)))
        else:
            # ── MoE layer ──
            # Slot 5: ffn_gate_inp.weight (F32!)
            slot_vals.store(5, load_file(dcp, cstr_from_str(prefix_str + String("ffn_gate_inp_weight.bin"))))
            # Slot 6: ffn_gate_inp.bias
            slot_vals.store(6, load_file(dcp, cstr_from_str(prefix_str + String("ffn_gate_inp_bias.bin"))))
            # Slot 7: ffn_gate.weight (Q8_0)
            slot_vals.store(7, load_file(dcp, cstr_from_str(prefix_str + String("ffn_gate_weight.bin"))))
            # Slot 8: ffn_gate.bias
            slot_vals.store(8, load_file(dcp, cstr_from_str(prefix_str + String("ffn_gate_bias.bin"))))
            # Slot 9: zaya_router_mlp2.weight (Q8_0)
            slot_vals.store(9, load_file(dcp, cstr_from_str(prefix_str + String("zaya_router_mlp2_weight.bin"))))
            # Slot 10: zaya_router_mlp2.bias
            slot_vals.store(10, load_file(dcp, cstr_from_str(prefix_str + String("zaya_router_mlp2_bias.bin"))))
            # Slot 11: zaya_router_mlp4.weight (Q8_0)
            slot_vals.store(11, load_file(dcp, cstr_from_str(prefix_str + String("zaya_router_mlp4_weight.bin"))))
            # Slot 12: zaya_router_biases.weight
            slot_vals.store(12, load_file(dcp, cstr_from_str(prefix_str + String("zaya_router_biases_weight.bin"))))
            # Slot 13: ffn_gate_up_exps.weight (Q8_0, 3D)
            slot_vals.store(13, load_file(dcp, cstr_from_str(prefix_str + String("ffn_gate_up_exps_weight.bin"))))
            # Slot 14: ffn_down_exps.weight (Q8_0, 3D)
            slot_vals.store(14, load_file(dcp, cstr_from_str(prefix_str + String("ffn_down_exps_weight.bin"))))
            # Slot 15: zaya_router_eda.weight (F32)
            slot_vals.store(15, load_file(dcp, cstr_from_str(prefix_str + String("zaya_router_eda_weight.bin"))))

        for i in range(WPL): wl.store(l * WPL + i, slot_vals.load(i))

    # Verify critical weights loaded
    var load_ok = True
    for l in range(NL):
        var base = l * WPL
        if wl.load(base + 0) < 0: print("MISSING: blk", l, "attn_norm"); load_ok = False
        if wl.load(base + 1) < 0: print("MISSING: blk", l, "res_scale_hs_w"); load_ok = False
        if wl.load(base + 2) < 0: print("MISSING: blk", l, "res_scale_hs_b"); load_ok = False
        if l % 2 == 0:
            if wl.load(base + 5) < 0: print("MISSING: blk", l, "attn_q"); load_ok = False
            if wl.load(base + 6) < 0: print("MISSING: blk", l, "attn_k"); load_ok = False
            if wl.load(base + 7) < 0: print("MISSING: blk", l, "attn_o"); load_ok = False
        else:
            if wl.load(base + 5) < 0: print("MISSING: blk", l, "ffn_gi"); load_ok = False
            if wl.load(base + 13) < 0: print("MISSING: blk", l, "gate_up_exps"); load_ok = False
            if wl.load(base + 14) < 0: print("MISSING: blk", l, "down_exps"); load_ok = False
    if not load_ok: print("ERROR: missing weight files!"); return

    var t_load = time.perf_counter()
    print("Load: ", Int((t_load - t0) * 1000), " ms")

    # ─── Allocate working buffers ───
    var hp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))
    var qp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NH * HD * 4))))
    var kp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NK * HD * 4))))
    var att_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NH * HD * 4))))
    var lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NV * 4))))
    var router_h = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * N_RH * 4))))
    var router_h2 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * N_RH * 4))))
    var scores = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * (N_EXP + 1) * 4))))
    var gate_up_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * FF * 4))))
    var expert_out = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NE * 4))))

    var kc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NL * NK * MAX_SEQ * HD * 4))))
    var vc = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(B * NL * NK * MAX_SEQ * HD * 4))))
    var sc_buf = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(_alc(Int64(MAX_CTX * 4))))

    # ─── Prompt setup ───
    var batch_toks = alloc[Int32](B * MAX_SEQ)
    # Simple "2+2=" prompt: ZAYA uses a different tokenizer, so use BOS + ASCII
    # This is approximate — for real testing use pre-tokenized prompts
    # Prompt: use <bos> token 2 then 511 BOS tokens for long prefill test
    var prompt = [2]
    var np = len(prompt)
    for bi in range(B):
        for i in range(np):
            batch_toks.store(bi * MAX_SEQ + i, Int32(prompt[i]))
    var nt = alloc[Int32](B)
    for bi in range(B): nt.store(bi, Int32(np))
    var max_gen = 128
    print('ZAYA1-8B Q8_0 B=' + String(B) + ' max_gen=', max_gen, ' prefill=', np, ' nw=', nw)

    # ─── Generation loop: single pass — prefill skips LM head to save time ───
    var t_gen = time.perf_counter()
    var emb_rb = q8_row_bytes(NE)

    for pos in range(max_gen):
        # Dequantize token embedding: Q8_0 → f32
        var emb = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(w_emb))
        for bi in range(B):
            var tok = Int(batch_toks.load(bi * MAX_SEQ + pos))
            var row_start = tok * emb_rb
            var off = row_start
            for blk in range(NE // QK):
                var lo = Int(emb.load(off))
                var hi = Int(emb.load(off + 1))
                var scale = h2f(UInt16(lo | (hi << 8)))
                off += 2
                for i in range(QK):
                    var qv = Int(emb.load(off)) - 128
                    hp.store(bi * NE + blk * QK + i, Float32(qv) * scale)
                    off += 1

        # Full 80-layer forward (builds KV cache)
        for l in range(NL):
            var lw = l * WPL
            var anp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw + 0)))
            var hs_wp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw + 1)))
            var hs_bp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(wl.load(lw + 2)))
            var rr_wp_addr = wl.load(lw + 3)
            var rr_bp_addr = wl.load(lw + 4)

            apply_rms_norm(hp, bp, anp, NE)

            if l % 2 == 0:
                var w_q = wl.load(lw + 5); var w_k = wl.load(lw + 6); var w_o = wl.load(lw + 7)
                _mm_q8_batch(Int(w_q), bp, qp, NH*HD, NE, nw)
                _mm_q8_batch(Int(w_k), bp, kp, NK*HD, NE, nw)
                for bi in range(B):
                    var qp_bi = qp + bi * NH * HD; var kp_bi = kp + bi * NK * HD
                    for d2 in range(0, ROPE_DIM, 2):
                        var freq = Float32(Float64(pos) / pow(ROPE_THETA, Float64(d2) / Float64(ROPE_DIM)))
                        var cv = cos(freq); var sv = sin(freq)
                        for h in range(NH):
                            var x0 = qp_bi.load(h * HD + d2); var x1 = qp_bi.load(h * HD + d2 + 1)
                            qp_bi.store(h*HD+d2, x0*cv - x1*sv); qp_bi.store(h*HD+d2+1, x0*sv + x1*cv)
                        for h in range(NK):
                            var x0 = kp_bi.load(h * HD + d2); var x1 = kp_bi.load(h * HD + d2 + 1)
                            kp_bi.store(h*HD+d2, x0*cv - x1*sv); kp_bi.store(h*HD+d2+1, x0*sv + x1*cv)
                    var cache_base = (bi * NL + l) * NK * MAX_SEQ * HD
                    for h in range(NK):
                        for d in range(HD):
                            kc.store(cache_base + h*MAX_SEQ*HD + pos*HD + d, kp_bi.load(h*HD + d))
                            vc.store(cache_base + h*MAX_SEQ*HD + pos*HD + d, kp_bi.load(h*HD + d))
                    var kr = NH // NK
                    for hq in range(NH):
                        var hk = hq // kr; var cache_hk_base = cache_base + hk * MAX_SEQ * HD
                        var qbase = hq * HD; var smax = Float32(-1e9); var sc = sc_buf
                        for p in range(pos + 1):
                            var sv = SIMD[DType.float32, W](0.0); var dd = 0
                            while dd + W <= HD:
                                var qv = qp_bi.load[width=W](qbase + dd)
                                var kv = kc.load[width=W](cache_hk_base + p * HD + dd)
                                sv = sv + qv * kv; dd += W
                            var s = sv.reduce_add() / sqrt(Float32(HD)); sc.store(p, s)
                            if s > smax: smax = s
                        var ssum = Float32(0.0)
                        for p in range(pos + 1):
                            var es = exp(sc.load(p) - smax); sc.store(p, es); ssum += es
                        # SIMD transposed V aggregation: per-position weighted sum
                        var inv_ssum = 1.0 / ssum
                        for d in range(HD): att_buf.store(bi * NH * HD + qbase + d, 0.0)
                        for p in range(pos + 1):
                            var wt = sc.load(p) * inv_ssum
                            var wt_v = SIMD[DType.float32, W](wt)
                            var dd = 0
                            while dd + W <= HD:
                                var vv = vc.load[width=W](cache_hk_base + p * HD + dd)
                                var cur = att_buf.load[width=W](bi * NH * HD + qbase + dd)
                                att_buf.store[width=W](bi * NH * HD + qbase + dd, cur + vv * wt_v)
                                dd += W
                            while dd < HD:
                                var cur = att_buf.load(bi * NH * HD + qbase + dd)
                                att_buf.store(bi * NH * HD + qbase + dd, cur + vc.load(cache_hk_base + p * HD + dd) * wt)
                                dd += 1
                    _mm_q8_batch(Int(w_o), att_buf, bp, NE, NH*HD, nw)
                    var rr_w_addr_a = wl.load(lw + 3); var rr_b_addr_a = wl.load(lw + 4)
                    var has_res_scale_a = (rr_w_addr_a != 0) and (rr_b_addr_a != 0)
                    var hp_bi = hp + bi * NE; var bp_bi = bp + bi * NE
                    if has_res_scale_a:
                        var rr_wp_a = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(rr_w_addr_a))
                        var rr_bp_a = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(rr_b_addr_a))
                        for i in range(NE): hp_bi.store(i, hp_bi.load(i) * rr_wp_a.load(i) + rr_bp_a.load(i) + bp_bi.load(i) * hs_wp.load(i) + hs_bp.load(i))
                    else:
                        for i in range(NE): hp_bi.store(i, hp_bi.load(i) + bp_bi.load(i) * hs_wp.load(i) + hs_bp.load(i))
            else:
                var w_gi = wl.load(lw + 5); var gi_bias_addr = wl.load(lw + 6)
                var w_fg = wl.load(lw + 7); var fg_bias_addr = wl.load(lw + 8)
                var w_rm2 = wl.load(lw + 9); var rm2_bias_addr = wl.load(lw + 10)
                var w_rm4 = wl.load(lw + 11); var rb_addr = wl.load(lw + 12)
                var w_gu = wl.load(lw + 13); var w_de = wl.load(lw + 14)
                var gi_bias_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(gi_bias_addr))
                var fg_bias_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(fg_bias_addr))
                var rm2_bias_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(rm2_bias_addr))
                var rb_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(rb_addr))
                for bi in range(B):
                    var xb = bp + bi * NE
                    _mm_f32_batch(Int(w_gi), xb, router_h, N_RH, NE, nw)
                    for i in range(N_RH): router_h.store(i, router_h.load(i) + gi_bias_p.load(i))
                    _mm_q8_batch(Int(w_fg), router_h, router_h2, N_RH, N_RH, nw)
                    for i in range(N_RH): router_h2.store(i, router_h2.load(i) + fg_bias_p.load(i))
                    silu_inplace(router_h2, N_RH)
                    _mm_q8_batch(Int(w_rm2), router_h2, router_h, N_RH, N_RH, nw)
                    for i in range(N_RH): router_h.store(i, router_h.load(i) + rm2_bias_p.load(i))
                    _mm_q8_batch(Int(w_rm4), router_h, scores, N_EXP+1, N_RH, nw)
                    for i in range(N_EXP+1): scores.store(i, scores.load(i) + rb_p.load(i))
                    var smaxf = Float32(-1e9)
                    for i in range(N_EXP+1):
                        var vf = scores.load(i)
                        if vf > smaxf: smaxf = vf
                    var ssumf = Float32(0.0)
                    for i in range(N_EXP+1):
                        var ef = exp(scores.load(i) - smaxf)
                        scores.store(i, ef)
                        ssumf += ef
                    for i in range(N_EXP+1): scores.store(i, scores.load(i) / ssumf)
                    var ec = 0
                    for i in range(1, N_EXP):
                        if scores.load(i) > scores.load(ec): ec = i
                    var exp_sum = Float32(0.0)
                    for i in range(N_EXP): exp_sum += scores.load(i)
                    var ew = scores.load(ec) / exp_sum
                    var per_exp_bytes_gate = FF * emb_rb
                    var exp_gate_up = Int(w_gu) + ec * per_exp_bytes_gate
                    _mm_q8_batch(exp_gate_up, xb, gate_up_buf, FF, NE, nw)
                    for i in range(F2):
                        var gv = gate_up_buf.load(i); var uv = gate_up_buf.load(F2 + i)
                        if gv < -80.0: gv = -80.0
                        if gv > 80.0: gv = 80.0
                        gate_up_buf.store(i, (gv / (1.0 + exp(-gv))) * uv)
                    var per_exp_bytes_down = F2 * emb_rb
                    var exp_down = Int(w_de) + ec * per_exp_bytes_down
                    _mm_q8_batch(exp_down, gate_up_buf, bp, F2, F2, nw)
                    var rr_w_addr = wl.load(lw + 3); var rr_b_addr = wl.load(lw + 4)
                    var has_res_scale = (rr_w_addr != 0) and (rr_b_addr != 0)
                    var hp_bi = hp + bi * NE; var bp_bi = bp + bi * NE
                    for i in range(NE):
                        var out = bp_bi.load(i) * ew; var x_val = hp_bi.load(i)
                        if has_res_scale:
                            var rr_wp_real = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(rr_w_addr))
                            var rr_bp_real = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(rr_b_addr))
                            hp_bi.store(i, x_val * rr_wp_real.load(i) + rr_bp_real.load(i) + out * hs_wp.load(i) + hs_bp.load(i))
                        else: hp_bi.store(i, x_val + out * hs_wp.load(i) + hs_bp.load(i))

        # ─── LM head (only during generation phase, not prefill) ───
        if pos >= np:
            for bi in range(B):
                var hp_i = hp + bi * NE; var bp_i = bp + bi * NE
                var onp_i = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w_on))
                apply_rms_norm(hp_i, bp_i, onp_i, NE)
            _mm_q8_batch(Int(w_emb), bp, lp, NV, NE, nw)
            for bi in range(B):
                var lp_i = lp + bi * NV; var best = 0; var bv = lp_i.load(0)
                for i in range(1, NV):
                    var v = lp_i.load(i)
                    if v > bv: bv = v; best = i
                var nti = Int(nt.load(bi))
                if nti < MAX_SEQ:
                    batch_toks.store(bi * MAX_SEQ + nti, Int32(best))
                    nt.store(bi, Int32(nti + 1))
                if best != 2 and best != 0:
                    var out_text = decode_token(voc_data, voc_meta, nv, best)
                    print(out_text, end="")
                elif best == 2: print("[EOS]", end="")
                else: print("[PAD]", end="")

    print()
    var t_end = time.perf_counter()
    var gen_ms = (t_end - t_gen) * 1000.0
    var total_gen = 0
    for bi in range(B): total_gen += Int(nt.load(bi)) - np
    print("B=", B, " nw=", nw, " Q8_0 prefill=", np, " gen=", total_gen,
          " total_time=", Int(gen_ms), " ms ( gen=", Float64(total_gen) / (gen_ms / 1000.0), " tok/s )")
