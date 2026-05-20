#!/usr/bin/env python3
"""Instrumented test with expert quant types set correctly."""
import signal, faulthandler, sys, os, time, ctypes, json
faulthandler.enable()
def debug_segfault(sig, frame):
    print(f"\n!!! SIGNAL {sig} at frame: {frame}", flush=True)
    import traceback
    traceback.print_stack(frame)
    sys.exit(1)
signal.signal(signal.SIGSEGV, debug_segfault)
import numpy as np

sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE

MODEL = sys.argv[1] if len(sys.argv) > 1 else '/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
TOKENIZER_PATH = '/tmp/qwen3-tokenizer/'

t0 = time.perf_counter()
e = TurboEngineV7MoE(MODEL, 32)
L = e.n_layers; N = e.n_embd; NH = e.n_head; NKH = e.n_kv_head
HD = e.head_dim; FF = e.n_ff; V = e.vocab_size; eps = e.eps
NE = e.n_experts; NK = e.n_experts_per_tok
moe_int = e.n_ff_expert if hasattr(e, 'n_ff_expert') else FF
S = max(N, NH*HD, FF, NKH*HD, moe_int)
print(f"Loaded: {L}L/{N}D/{FF}FF/{NH}H/{NKH}KV | MoE {NE}x{NK} | V={V} | S={S}", flush=True)
print(f"Load time: {time.perf_counter()-t0:.1f}s", flush=True)

# Use instrumented library
lib = ctypes.CDLL('/onedev-workspace/work/src/mojollama/kernels/cengine_batch_instr.so')
cv = ctypes.c_void_p; ci = ctypes.c_int; cf = ctypes.c_float

class BC(ctypes.Structure):
    _fields_ = [
        ("L",ci),("N",ci),("NH",ci),("NKH",ci),("HD",ci),("FF",ci),("V",ci),("eps",cf),
        ("wQ",cv),("wK",cv),("wV",cv),("wO",cv),("wG",cv),("wU",cv),("wD",cv),
        ("wAN",cv),("wFN",cv),("nQ",cv),("nK",cv),("nV",cv),("nO",cv),("nG",cv),("nU",cv),("nD",cv),
        ("nc",ci),("emb",cv),("onw",cv),("wOut",cv),("outNR",ci),("outNC",ci),("outQuant",ci),
        ("kv_array",cv),("logits",cv),
        ("n_experts",ci),("n_experts_per_tok",ci),("moe_intermediate",ci),
        ("w_gate_inp",cv),("w_gate_exps",cv),("w_up_exps",cv),("w_down_exps",cv),
        ("gate_exp_quant",ci),("up_exp_quant",ci),("down_exp_quant",ci),
        ("q_quant",cv),("k_quant",cv),("v_quant",cv),("o_quant",cv),
        ("g_quant",cv),("u_quant",cv),("d_quant",cv),("emb_quant",ci),
        ("cos_table",cv),("sin_table",cv),("max_ctx",ci),
        ("workspace",cv),("ws_size",ci),
    ]

class KVBlock(ctypes.Structure):
    _fields_ = [("k",cv),("v",cv),("n_blocks",ci),("seq_len",ci*64),("block_map",(ci*1024)*64)]

lib.batch_forward.argtypes = [cv, cv, ci, cv]; lib.batch_forward.restype = None
lib.kv_init.argtypes = [cv, ci, ci, ci]; lib.kv_init.restype = None

def wa(arr):
    return (cv*L)(*[ctypes.cast(a, cv) for a in arr])
def ia(arr):
    return (ci*L)(*[int(a) for a in arr])

bc = BC()

# Read actual quant types from the MoE layer info
moe_layer = e._moe_layers[0] if e.is_moe else None
if moe_layer:
    gate_qt = moe_layer.gate_qt.value if hasattr(moe_layer, 'gate_qt') else 2
    up_qt = moe_layer.up_qt.value if hasattr(moe_layer, 'up_qt') else 2
    down_qt = moe_layer.down_qt.value if hasattr(moe_layer, 'down_qt') else 2
else:
    gate_qt = 2; up_qt = 2; down_qt = 2

print(f"Expert quant types from engine: gate={gate_qt} up={up_qt} down={down_qt}", flush=True)

for attr, val in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',eps),
                  ('nc',N),('outNR',V),('outNC',N),('outQuant',8),
                  ('n_experts',NE),('n_experts_per_tok',NK),('moe_intermediate',moe_int),
                  ('gate_exp_quant',gate_qt),('up_exp_quant',up_qt),('down_exp_quant',down_qt),
                  ('emb_quant',0)]:
    setattr(bc, attr, val)

wQ_arr = [0]*L; wK_arr = [0]*L; wV_arr = [0]*L; wO_arr = [0]*L
wG_arr = [0]*L; wU_arr = [0]*L; wD_arr = [0]*L
wAN_arr = [0]*L; wFN_arr = [0]*L
nQ_arr = [0]*L; nK_arr = [0]*L; nV_arr = [0]*L; nO_arr = [0]*L
nG_arr = [0]*L; nU_arr = [0]*L; nD_arr = [0]*L
qQ_arr = [0]*L; qK_arr = [0]*L; qV_arr = [0]*L; qO_arr = [0]*L
qG_arr = [0]*L; qU_arr = [0]*L; qD_arr = [0]*L

for i, lw in enumerate(e._layers):
    for attr, nm in [('attn_q','Q'),('attn_k','K'),('attn_v','V'),('attn_out','O'),
                     ('ffn_gate','G'),('ffn_up','U'),('ffn_down','D')]:
        raw_attr = f'{attr}_raw'
        if hasattr(lw, raw_attr):
            raw = getattr(lw, raw_attr)
            nr = getattr(lw, f'{attr}_nr').value
            qt = getattr(lw, f'{attr}_qt').value
        else:
            f32 = getattr(lw, f'{attr}_f32')
            raw = f32.ctypes.data_as(cv) if f32 is not None else cv(0)
            nr = f32.shape[0] if f32 is not None else 0
            qt = 0
        target = nm
        if target == 'Q': wQ_arr[i] = raw; nQ_arr[i] = nr; qQ_arr[i] = qt
        elif target == 'K': wK_arr[i] = raw; nK_arr[i] = nr; qK_arr[i] = qt
        elif target == 'V': wV_arr[i] = raw; nV_arr[i] = nr; qV_arr[i] = qt
        elif target == 'O': wO_arr[i] = raw; nO_arr[i] = nr; qO_arr[i] = qt
        elif target == 'G': wG_arr[i] = raw; nG_arr[i] = nr; qG_arr[i] = qt
        elif target == 'U': wU_arr[i] = raw; nU_arr[i] = nr; qU_arr[i] = qt
        elif target == 'D': wD_arr[i] = raw; nD_arr[i] = nr; qD_arr[i] = qt
    wAN_arr[i] = lw.attn_norm_w.ctypes.data_as(cv)
    wFN_arr[i] = lw.ffn_norm_w.ctypes.data_as(cv)

# MoE pointers
w_gate_inp_arr = [0]*L; w_gate_exps_arr = [0]*L; w_up_exps_arr = [0]*L; w_down_exps_arr = [0]*L
for i, lw in enumerate(e._layers):
    if e.is_moe:
        me = e._moe_layers[i]
        if me.router_raw is not None:
            w_gate_inp_arr[i] = me.router_raw.ctypes.data_as(cv)
        elif me.router_f32 is not None:
            w_gate_inp_arr[i] = me.router_f32.ctypes.data_as(cv)
        else:
            w_gate_inp_arr[i] = cv(0)
        gate_raw_t = me.gate_raw
        up_raw_t = me.up_raw
        down_raw_t = me.down_raw
        if gate_raw_t and len(gate_raw_t) > 0:
            w_gate_exps_arr[i] = gate_raw_t[0].ctypes.data_as(cv)
        else:
            w_gate_exps_arr[i] = cv(0)
        if up_raw_t and len(up_raw_t) > 0:
            w_up_exps_arr[i] = up_raw_t[0].ctypes.data_as(cv)
        else:
            w_up_exps_arr[i] = cv(0)
        if down_raw_t and len(down_raw_t) > 0:
            w_down_exps_arr[i] = down_raw_t[0].ctypes.data_as(cv)
        else:
            w_down_exps_arr[i] = cv(0)

bc.wQ = ctypes.cast(wa(wQ_arr), cv); bc.wK = ctypes.cast(wa(wK_arr), cv)
bc.wV = ctypes.cast(wa(wV_arr), cv); bc.wO = ctypes.cast(wa(wO_arr), cv)
bc.wG = ctypes.cast(wa(wG_arr), cv); bc.wU = ctypes.cast(wa(wU_arr), cv)
bc.wD = ctypes.cast(wa(wD_arr), cv)
bc.wAN = ctypes.cast(wa(wAN_arr), cv); bc.wFN = ctypes.cast(wa(wFN_arr), cv)
bc.nQ = ctypes.cast(ia(nQ_arr), cv); bc.nK = ctypes.cast(ia(nK_arr), cv)
bc.nV = ctypes.cast(ia(nV_arr), cv); bc.nO = ctypes.cast(ia(nO_arr), cv)
bc.nG = ctypes.cast(ia(nG_arr), cv); bc.nU = ctypes.cast(ia(nU_arr), cv)
bc.nD = ctypes.cast(ia(nD_arr), cv)
bc.q_quant = ctypes.cast(ia(qQ_arr), cv); bc.k_quant = ctypes.cast(ia(qK_arr), cv)
bc.v_quant = ctypes.cast(ia(qV_arr), cv); bc.o_quant = ctypes.cast(ia(qO_arr), cv)
bc.g_quant = ctypes.cast(ia(qG_arr), cv); bc.u_quant = ctypes.cast(ia(qU_arr), cv)
bc.d_quant = ctypes.cast(ia(qD_arr), cv)
bc.w_gate_inp = ctypes.cast(wa(w_gate_inp_arr), cv)
bc.w_gate_exps = ctypes.cast(wa(w_gate_exps_arr), cv)
bc.w_up_exps = ctypes.cast(wa(w_up_exps_arr), cv)
bc.w_down_exps = ctypes.cast(wa(w_down_exps_arr), cv)
bc.emb = e.emb.ctypes.data_as(cv) if hasattr(e, 'emb') else cv(0)
bc.onw = e._onw.ctypes.data_as(cv) if hasattr(e, '_onw') else cv(0)
bc.wOut = ctypes.cast(e._or, cv) if hasattr(e, '_or') else cv(0)

hd2 = HD // 2
max_ctx = 4096
freq = e.rope_freq_base ** (np.arange(0, HD, 2, dtype=np.float32) / HD)
ang = np.arange(max_ctx, dtype=np.float32).reshape(-1, 1) / freq.reshape(1, -1)
cos_all = np.cos(ang).astype(np.float32).reshape(-1)
sin_all = np.sin(ang).astype(np.float32).reshape(-1)
cos_t = cos_all.ctypes.data_as(cv)
sin_t = sin_all.ctypes.data_as(cv)
bc.cos_table = cos_t
bc.sin_table = sin_t
bc.max_ctx = max_ctx
print(f"RoPE table: {max_ctx}x{hd2}", flush=True)

ws = np.zeros(12 * S, dtype=np.float32)
logits_buf = np.zeros(V, dtype=np.float32)

kv = KVBlock()
lib.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
kva = (cv*1)(ctypes.cast(ctypes.pointer(kv), cv))
bc.kv_array = ctypes.cast(kva, cv)
bc.logits = logits_buf.ctypes.data_as(cv)

token = 785
print(f"\n--- Running batch_forward(token={token}) ---", flush=True)
sys.stdout.flush()
sys.stderr.flush()
t0 = time.perf_counter()
lib.batch_forward(ctypes.byref(bc), (ci*1)(token), ci(1), ws.ctypes.data_as(cv))
elapsed = time.perf_counter() - t0

logits = logits_buf[:V].copy()
nan_count = np.isnan(logits).sum()
inf_count = np.isinf(logits).sum()
max_val = float(np.max(logits))
min_val = float(np.min(logits))
mean_val = float(np.mean(logits))
top5 = np.argsort(logits)[-5:][::-1]

print(f"  Time: {elapsed*1000:.1f}ms ({1/elapsed:.1f} tok/s)", flush=True)
print(f"  Logits: nan={nan_count}/{V}, inf={inf_count}, max={max_val:.2f}, min={min_val:.2f}, mean={mean_val:.4f}", flush=True)
print(f"  Top-5 tokens: {top5.tolist()}", flush=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print(f"  Top-5 decoded: {[tok.decode([t], skip_special_tokens=True) for t in top5]}", flush=True)

print(f"\n--- Generating 5 more tokens ---", flush=True)
t0 = time.perf_counter()
for _ in range(5):
    logits_buf[:] = 0
    safe = np.nan_to_num(logits, nan=-1e10)
    token = int(np.argmax(safe))
    lib.batch_forward(ctypes.byref(bc), (ci*1)(token), ci(1), ws.ctypes.data_as(cv))
    logits = logits_buf[:V].copy()
elapsed = time.perf_counter() - t0
print(f"  5 tokens: {elapsed*1000:.1f}ms ({5/elapsed:.1f} tok/s)", flush=True)
