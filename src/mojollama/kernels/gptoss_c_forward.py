#!/usr/bin/env python3
"""GPT-OSS C Forward — call gptoss_forward_c from combined_engine.so.
Replaces 530 Python ctypes calls with 1 C call.

Usage: python3 gptoss_c_forward.py /path/to/model.gguf [n_tokens=30]
"""
import os, sys, time, numpy as np, ctypes

os.environ['OMP_PROC_BIND'] = 'close'
os.environ['OMP_PLACES'] = 'cores'
os.environ['OMP_NUM_THREADS'] = '32'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from turbo_engine_v7_moe import TurboEngineV7MoE

model = sys.argv[1] if len(sys.argv) > 1 else '/tmp/models/gpt-oss-20b-Q4_K_M.gguf'
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 30

eng = TurboEngineV7MoE(model, n_threads=32)
V = int(eng.vocab_size); L = eng.n_layers; N = eng.n_embd
NH = eng.n_head; NKH = eng.n_kv_head; HD = eng.head_dim
FF = eng.n_ff; top_k = eng.n_experts_per_tok
n_ff_expert = eng.n_ff_expert; n_experts = eng.n_experts
eps = eng.eps; rope_dim = eng.rope_dim; max_ctx = 4096

kernel_dir = os.path.dirname(__file__)
so_path = os.path.join(kernel_dir, 'combined_engine.so')
so = ctypes.CDLL(so_path)

# ctypes types
cf = ctypes.POINTER(ctypes.c_float)
cu = ctypes.POINTER(ctypes.c_uint8)
ci = ctypes.c_int

# Build per-layer arrays of pointers
def ptr_arr_float(items):
    arr = (cf * len(items))()
    for i, it in enumerate(items):
        arr[i] = it.ctypes.data_as(cf) if hasattr(it, 'ctypes') else it
    return arr
def ptr_arr_uint8(items):
    arr = (cu * len(items))()
    for i, it in enumerate(items):
        arr[i] = it
    return arr
def int_arr(items):
    arr = (ci * len(items))()
    for i, it in enumerate(items):
        arr[i] = it.value if hasattr(it, 'value') else int(it)
    return arr

attn_norm_arr = ptr_arr_float([lw.attn_norm_w for lw in eng._layers])
ffn_norm_arr = ptr_arr_float([lw.ffn_norm_w for lw in eng._layers])

wQ_arr = ptr_arr_uint8([lw.attn_q_raw for lw in eng._layers])
q_nr = int_arr([lw.attn_q_nr.value for lw in eng._layers])
q_nc = int_arr([lw.attn_q_nc.value for lw in eng._layers])
q_qt = int_arr([lw.attn_q_qt for lw in eng._layers])

wK_arr = ptr_arr_uint8([lw.attn_k_raw for lw in eng._layers])
k_nr = int_arr([lw.attn_k_nr.value for lw in eng._layers])
k_nc = int_arr([lw.attn_k_nc.value for lw in eng._layers])
k_qt = int_arr([lw.attn_k_qt for lw in eng._layers])

wV_arr = ptr_arr_uint8([lw.attn_v_raw for lw in eng._layers])
v_nr = int_arr([lw.attn_v_nr.value for lw in eng._layers])
v_nc = int_arr([lw.attn_v_nc.value for lw in eng._layers])
v_qt = int_arr([lw.attn_v_qt for lw in eng._layers])

wO_arr = ptr_arr_uint8([lw.attn_out_raw for lw in eng._layers])
o_nr = int_arr([lw.attn_out_nr.value for lw in eng._layers])
o_nc = int_arr([lw.attn_out_nc.value for lw in eng._layers])
o_qt = int_arr([lw.attn_out_qt for lw in eng._layers])

# MoE
me0 = eng._moe_layers[0]
gqt = me0.gate_qt.value if hasattr(me0.gate_qt, 'value') else int(me0.gate_qt)
uqt = me0.up_qt.value if hasattr(me0.up_qt, 'value') else int(me0.up_qt)
dqt = me0.down_qt.value if hasattr(me0.down_qt, 'value') else int(me0.down_qt)

w_gate_inp_arr = ptr_arr_float([me.router_f32 for me in eng._moe_layers])
w_gate_exps_arr = ptr_arr_uint8([ctypes.cast(me.gate_ptrs_arr, cu) for me in eng._moe_layers])
w_up_exps_arr = ptr_arr_uint8([ctypes.cast(me.up_ptrs_arr, cu) for me in eng._moe_layers])
w_down_exps_arr = ptr_arr_uint8([ctypes.cast(me.down_ptrs_arr, cu) for me in eng._moe_layers])

# Shared expert (NULL)
null_pf = (cf * L)()
null_pu = (cu * L)()
for i in range(L): null_pf[i] = None; null_pu[i] = None

# KV cache
kv_k = eng.kv_k.ctypes.data_as(cf)
kv_v = eng.kv_v.ctypes.data_as(cf)

# Output
out_norm_w = eng._out_norm_w.ctypes.data_as(cf)
w_out = eng._out_raw
o_nr_val = eng._out_nr.value; o_nc_val = eng._out_nc.value
o_qt_val = eng._out_qt

# Workspace
S = max(N, NH*HD, FF, NKH*HD, 8192)
ws = np.zeros(14 * S + max_ctx * NKH * HD * 2, dtype=np.float32)
q8_ws = np.zeros((N+31)//32 * 34, dtype=np.uint8)
logits = np.zeros(V, dtype=np.float32)

# RoPE tables
hd2 = rope_dim // 2
cos_t = np.zeros((max_ctx, hd2), dtype=np.float32)
sin_t = np.zeros((max_ctx, hd2), dtype=np.float32)
for pos in range(max_ctx):
    angle = pos / (10000.0 ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
    cos_t[pos] = np.cos(angle); sin_t[pos] = np.sin(angle)

# Set argtypes
so.gptoss_forward_c.argtypes = [
    ctypes.c_void_p, ci,  # emb, token
    ci, ci, ci, ci, ci, ci, ci,  # L, N, NH, NKH, HD, FF, V
    ci, ci, ci, ctypes.c_float, ci, ci,  # n_experts, top_k, n_ff_expert, eps, rope_dim, max_ctx
    ctypes.c_void_p, ctypes.c_void_p,  # attn_norm_w, ffn_norm_w
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # wQ, q_nr, q_nc, q_qt
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # wK, k_nr, k_nc, k_qt
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # wV, v_nr, v_nc, v_qt
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # wO, o_nr, o_nc, o_qt
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ci, ci, ci,  # w_gate_inp, w_gate_exps/w_up/w_down, gate/up/down_qt
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ci, ci, ci, ci,  # w_shexp_router/g/u/d, shexp_g/u/d/qt, shexp_int
    ctypes.c_void_p, ctypes.c_void_p,  # cos_table, sin_table
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # kv_k, kv_v, kv_lens
    ctypes.c_void_p, ctypes.c_void_p, ci, ci, ci, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p  # out_norm_w, w_out, out_nr/out_nc/out_qt, logits, ws, q8_ws
]

def run_one(token_id):
    kv_len = (ci * L)(*[0 for _ in range(L)])
    so.gptoss_forward_c(
        eng.emb.ctypes.data_as(cf), ci(token_id),
        ci(L), ci(N), ci(NH), ci(eng.n_kv_head), ci(HD), ci(FF), ci(V),
        ci(n_experts), ci(top_k), ci(n_ff_expert), ctypes.c_float(eps), ci(rope_dim), ci(max_ctx),
        attn_norm_arr, ffn_norm_arr,
        wQ_arr, q_nr, q_nc, q_qt,
        wK_arr, k_nr, k_nc, k_qt,
        wV_arr, v_nr, v_nc, v_qt,
        wO_arr, o_nr, o_nc, o_qt,
        w_gate_inp_arr, w_gate_exps_arr, w_up_exps_arr, w_down_exps_arr,
        ci(gqt), ci(uqt), ci(dqt),
        null_pf, null_pu, null_pu, null_pu,
        ci(0), ci(0), ci(0), ci(0),
        cos_t.ctypes.data_as(cf), sin_t.ctypes.data_as(cf),
        kv_k, kv_v, kv_len,
        out_norm_w, w_out, ci(o_nr_val), ci(o_nc_val), ci(o_qt_val),
        logits.ctypes.data_as(cf), ws.ctypes.data_as(cf), q8_ws.ctypes.data_as(cu))
    return logits.copy()

# Warmup
for i in range(3):
    l = run_one(i % V)
    print(f"  Warmup {i}: max_logit={np.max(l):.2f}")

# Benchmark
print(f"\nBenchmarking {n_tok} tokens via C forward...")
tc = []
for tok in range(n_tok):
    t0 = time.perf_counter()
    l = run_one(tok % V)
    tc.append((time.perf_counter() - t0) * 1000)

tc_a = np.array(tc)
print(f"\n{'='*50}")
print(f"  GPT-OSS C Forward Benchmark")
print(f"{'='*50}")
print(f"  Median: {np.median(tc_a):.2f} ms")
print(f"  Mean:   {np.mean(tc_a):.2f} ms")
print(f"  P95:    {np.percentile(tc_a, 95):.2f} ms")
print(f"  Tok/s:  {1000/np.mean(tc_a):.1f}")
print(f"  llama.cpp ref: 27.27 tok/s")
print(f"  Ratio:  {1000/np.mean(tc_a)/27.27*100:.0f}%")
print(f"{'='*50}")
