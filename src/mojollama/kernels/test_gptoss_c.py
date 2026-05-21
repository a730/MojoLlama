#!/usr/bin/env python3
"""Test gptoss_batch_forward C engine."""
import sys, os, ctypes, time, numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'src/mojollama/kernels')
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['OMP_PROC_BIND'] = 'close'
os.environ['OMP_PLACES'] = 'cores'

from turbo_engine_v7_moe import TurboEngineV7MoE

# Load engine
eng = TurboEngineV7MoE('models/gpt-oss-20b-MXFP4.gguf', n_threads=32)
eng.reset()

# Load C engine
ceng_path = os.path.join(os.path.dirname(__file__), 'gptoss_batch_forward.so')
ceng = ctypes.CDLL(ceng_path)

cf = ctypes.POINTER(ctypes.c_float)
cu = ctypes.POINTER(ctypes.c_uint8)
ci = ctypes.c_int

ceng.gptoss_forward.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # token
    ci,                            # pos
    cf,                            # emb
    ci,                            # V
    ctypes.POINTER(cu),            # wQ (array of L pointers)
    ctypes.POINTER(cu),            # wK
    ctypes.POINTER(cu),            # wV
    ctypes.POINTER(cu),            # wO
    ctypes.POINTER(cf),            # wAN
    ctypes.POINTER(cf),            # wFN
    ctypes.POINTER(ci),            # nQ
    ctypes.POINTER(ci),            # nK
    ctypes.POINTER(ci),            # nV
    ctypes.POINTER(ci),            # nO
    ctypes.POINTER(ci),            # q_quant
    ctypes.POINTER(ci),            # k_quant
    ctypes.POINTER(ci),            # v_quant
    ctypes.POINTER(ci),            # o_quant
    cu,                            # wOut
    ci,                            # outNR
    ci,                            # outNC
    ci,                            # outQuant
    cf,                            # k_cache
    cf,                            # v_cache
    ctypes.POINTER(ci),            # kv_len
    cf,                            # cos_table
    cf,                            # sin_table
    ci,                            # max_ctx
    ci,                            # rope_dim
    cf,                            # workspace
    ci,                            # ws_size
    ci,                            # L
    ci,                            # N
    ci,                            # NH
    ci,                            # NKH
    ci,                            # HD
    ci,                            # FF
    ctypes.c_float,                # eps
    ci,                            # n_experts
    ci,                            # n_experts_per_tok
    ci,                            # moe_intermediate
    ctypes.POINTER(cf),            # w_gate_inp (array of L pointers)
    ctypes.POINTER(ctypes.POINTER(cu)),  # w_gate_exps (array of L arrays of expert pointers)
    ctypes.POINTER(ctypes.POINTER(cu)),  # w_up_exps
    ctypes.POINTER(ctypes.POINTER(cu)),  # w_down_exps
    ci,                            # gate_exp_quant
    ci,                            # up_exp_quant
    ci,                            # down_exp_quant
    cf,                            # logits
    cf,                            # moe_prealloc_buf
    cu,                            # moe_prealloc_q8
    cf,                            # gqa_ws
]
ceng.gptoss_forward.restype = None

print("C engine loaded successfully")

# Now we need to set up all the pointers
L = eng.n_layers
N = eng.n_embd
NH = eng.n_head
NKV = eng.n_kv_head
HD = eng.head_dim
NKH = NKV * HD
FF = eng.n_ff
V = eng.vocab_size
eps = eng.eps
n_experts = eng.n_experts_per_layer if hasattr(eng, 'n_experts_per_layer') else 32
n_experts_per_tok = eng.n_experts_per_tok
moe_intermediate = eng.n_ff_expert if hasattr(eng, 'n_ff_expert') else FF
rope_dim = eng.rope_dim if hasattr(eng, 'rope_dim') and eng.rope_dim > 0 else HD
max_ctx = 4096

print(f"Model: L={L}, N={N}, NH={NH}, NKH={NKH}, HD={HD}, FF={FF}, V={V}")
print(f"MoE: n_experts={n_experts}, top_k={n_experts_per_tok}, moe_int={moe_intermediate}")
print(f"rope_dim={rope_dim}")

# Build pointer arrays
wQ_ptrs = (cu * L)()
wK_ptrs = (cu * L)()
wV_ptrs = (cu * L)()
wO_ptrs = (cu * L)()
wAN_ptrs = (cf * L)()
wFN_ptrs = (cf * L)()
nQ_arr = (ci * L)()
nK_arr = (ci * L)()
nV_arr = (ci * L)()
nO_arr = (ci * L)()
q_quant_arr = (ci * L)()
k_quant_arr = (ci * L)()
v_quant_arr = (ci * L)()
o_quant_arr = (ci * L)()

for l in range(L):
    lw = eng._layers[l]
    wQ_ptrs[l] = lw.attn_q_raw
    wK_ptrs[l] = lw.attn_k_raw
    wV_ptrs[l] = lw.attn_v_raw
    wO_ptrs[l] = lw.attn_out_raw
    wAN_ptrs[l] = lw.attn_norm_w.ctypes.data_as(cf)
    wFN_ptrs[l] = lw.ffn_norm_w.ctypes.data_as(cf)
    nQ_arr[l] = lw.attn_q_nr.value
    nK_arr[l] = lw.attn_k_nr.value
    nV_arr[l] = lw.attn_v_nr.value
    nO_arr[l] = lw.attn_out_nr.value
    q_quant_arr[l] = lw.attn_q_qt
    k_quant_arr[l] = lw.attn_k_qt
    v_quant_arr[l] = lw.attn_v_qt
    o_quant_arr[l] = lw.attn_out_qt

# MoE expert pointers (array of L pointers to arrays of n_experts pointers)
w_gate_inp_ptrs = (cf * L)()
w_gate_exps_ptrs = (ctypes.POINTER(cu) * L)()
w_up_exps_ptrs = (ctypes.POINTER(cu) * L)()
w_down_exps_ptrs = (ctypes.POINTER(cu) * L)()

for l in range(L):
    me = eng._moe_layers[l]
    # Router is F32, not raw quantized
    if me.router_f32 is not None:
        w_gate_inp_ptrs[l] = me.router_f32.ctypes.data_as(cf)
    else:
        w_gate_inp_ptrs[l] = me.router_raw
    w_gate_exps_ptrs[l] = ctypes.cast(me.gate_ptrs_arr, ctypes.POINTER(cu))
    w_up_exps_ptrs[l] = ctypes.cast(me.up_ptrs_arr, ctypes.POINTER(cu))
    w_down_exps_ptrs[l] = ctypes.cast(me.down_ptrs_arr, ctypes.POINTER(cu))

# Output weight
wOut = eng._out_raw
outNR = eng._out_nr.value
outNC = eng._out_nc.value
outQuant = eng._out_qt

# KV cache
k_cache = eng.kv_k.ctypes.data_as(cf)
v_cache = eng.kv_v.ctypes.data_as(cf)
kv_len = (ci * L)(*eng.kv_len)

# RoPE tables
# Need to precompute
half_dim = rope_dim // 2
cos_table = np.zeros((max_ctx, half_dim), dtype=np.float32)
sin_table = np.zeros((max_ctx, half_dim), dtype=np.float32)
freq = eng.rope_freq_base ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim)
print(f"freq shape: {freq.shape}, rope_freq_base: {eng.rope_freq_base}")
for pos in range(max_ctx):
    angle = pos / freq
    cos_table[pos] = np.cos(angle).astype(np.float32)
    sin_table[pos] = np.sin(angle).astype(np.float32)

cos_table_ptr = cos_table.ctypes.data_as(cf)
sin_table_ptr = sin_table.ctypes.data_as(cf)

# Workspace
S = max(N, NH * HD, FF, NKH * HD, 8192)
ws_size = 13 * S + N + n_experts  # x, xn, res, q, k, v, att, gate, up, silu, oproj, ffn, router
workspace = np.zeros(ws_size, dtype=np.float32)
ws_ptr = workspace.ctypes.data_as(cf)

# MoE prealloc
moe_prealloc_buf = eng._moe_prealloc_buf.ctypes.data_as(cf)
moe_prealloc_q8 = eng._moe_prealloc_q8.ctypes.data_as(cu)

# GQA workspace
gqa_ws = eng._gqa_workspace.ctypes.data_as(cf)

# Logits
logits = np.zeros(V, dtype=np.float32)
logits_ptr = logits.ctypes.data_as(cf)

# Embedding
emb = eng.emb.ctypes.data_as(cf)

# Token
token = ctypes.c_int(1)
pos = 0

print("Running C forward pass...")
t0 = time.perf_counter()
ceng.gptoss_forward(
    ctypes.byref(token), pos,
    emb, V,
    wQ_ptrs, wK_ptrs, wV_ptrs, wO_ptrs,
    wAN_ptrs, wFN_ptrs,
    nQ_arr, nK_arr, nV_arr, nO_arr,
    q_quant_arr, k_quant_arr, v_quant_arr, o_quant_arr,
    wOut, outNR, outNC, outQuant,
    k_cache, v_cache, kv_len,
    cos_table_ptr, sin_table_ptr, max_ctx, rope_dim,
    ws_ptr, ws_size,
    L, N, NH, NKH, HD, FF, eps,
    n_experts, n_experts_per_tok, moe_intermediate,
    w_gate_inp_ptrs, w_gate_exps_ptrs, w_up_exps_ptrs, w_down_exps_ptrs,
    eng._moe_layers[0].gate_qt, eng._moe_layers[0].up_qt, eng._moe_layers[0].down_qt,
    logits_ptr,
    moe_prealloc_buf, moe_prealloc_q8,
    gqa_ws
)
t1 = time.perf_counter()
print(f"C forward pass: {(t1-t0)*1000:.1f} ms")
print(f"Logits top 5: {np.argsort(logits)[-5:][::-1]}")
print(f"Logits max: {np.max(logits):.2f}")
