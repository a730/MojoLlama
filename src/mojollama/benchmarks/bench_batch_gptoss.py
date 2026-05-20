#!/usr/bin/env python3
"""GPT-OSS-20B batch benchmark — optimized Q8_0 conversion + C engine batch_forward."""
import sys, os, time, ctypes, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
MODEL = "/tmp/models/gpt-oss-20b-Q4_K_M.gguf"
N_USERS = 10
PROMPT_LEN = 512
GEN_TOKENS = 20

# ── Fast Q8_0 converter (vectorized) ──
def f32_to_q8_vectorized(arr):
    """Convert F32 (rows, cols) to Q8_0. Each 32-element block: float16 scale (2B) + 32×int8 (32B) = 34B."""
    assert arr.dtype == np.float32
    n_rows, nc = arr.shape
    bpr = nc // 32
    blocks = arr.reshape(n_rows, bpr, 32)
    abs_max = np.max(np.abs(blocks), axis=-1, keepdims=True)
    d = np.where(abs_max > 0, abs_max / 127.0, 1.0)
    q = np.clip(np.round(blocks / d), -128, 127).astype(np.int8)
    d_f16 = d.reshape(-1).astype(np.float16).view(np.uint16)
    q_flat = q.reshape(-1, 32)
    n_blocks = q_flat.shape[0]
    out = np.empty(n_blocks * 34, dtype=np.uint8).reshape(n_blocks, 34)
    out[:, 0:2] = d_f16.reshape(-1, 1).view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:34] = q_flat.view(np.uint8).reshape(n_blocks, 32)
    return out.reshape(-1)

print("Loading engine...", flush=True)
from turbo_engine_v7_moe import TurboEngineV7MoE
e = TurboEngineV7MoE(MODEL, 32)
N = e.n_embd; NH = e.n_head; NKH = e.n_kv_head; HD = e.head_dim
FF = e.n_ff; L = e.n_layers; V = e.vocab_size
print(f"  {L}L/{N}D/{NH}H/{NKH}KV/{HD}hd MoE {e.n_experts}x{e.n_experts_per_tok}", flush=True)

libfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels', 'cengine_batch_instr.so')
lib = ctypes.CDLL(libfile)
cv = ctypes.c_void_p; ci = ctypes.c_int; cf = ctypes.c_float

def wa(arr):
    return (cv * L)(*[ctypes.cast(a, cv) for a in arr])
def ia(arr):
    return (ci * L)(*[int(a) for a in arr])

wQ = [0]*L; nQ = [0]*L; qQ = [0]*L
wK = [0]*L; nK = [0]*L; qK = [0]*L
wV = [0]*L; nV = [0]*L; qV = [0]*L
wO = [0]*L; nO = [0]*L; qO = [0]*L
wG = [0]*L; nG = [0]*L; qG = [0]*L
wU = [0]*L; nU = [0]*L; qU = [0]*L
wD = [0]*L; nD = [0]*L; qD = [0]*L
wAN = [0]*L; wFN = [0]*L
_q8_bufs = []

print("Converting F32 attention weights → Q8_0 (vectorized)...", flush=True)
t0 = time.perf_counter()
for i, lw in enumerate(e._layers):
    if hasattr(lw, 'attn_q_f32') and lw.attn_q_f32 is not None:
        q8 = f32_to_q8_vectorized(lw.attn_q_f32); _q8_bufs.append(q8)
        wQ[i] = q8.ctypes.data_as(cv); nQ[i] = NH * HD; qQ[i] = 8
    elif hasattr(lw, 'attn_q_raw') and lw.attn_q_raw is not None:
        wQ[i] = lw.attn_q_raw; nQ[i] = lw.attn_q_nr.value; qQ[i] = lw.attn_q_qt.value
    if hasattr(lw, 'attn_k_f32') and lw.attn_k_f32 is not None:
        q8 = f32_to_q8_vectorized(lw.attn_k_f32); _q8_bufs.append(q8)
        wK[i] = q8.ctypes.data_as(cv); nK[i] = NKH * HD; qK[i] = 8
    elif hasattr(lw, 'attn_k_raw') and lw.attn_k_raw is not None:
        wK[i] = lw.attn_k_raw; nK[i] = lw.attn_k_nr.value; qK[i] = lw.attn_k_qt.value
    if hasattr(lw, 'attn_v_raw') and lw.attn_v_raw is not None:
        wV[i] = lw.attn_v_raw; nV[i] = lw.attn_v_nr.value; qV[i] = lw.attn_v_qt.value
    if hasattr(lw, 'attn_out_raw') and lw.attn_out_raw is not None:
        wO[i] = lw.attn_out_raw; nO[i] = lw.attn_out_nr.value; qO[i] = lw.attn_out_qt.value
    if hasattr(lw, 'attn_norm_w') and lw.attn_norm_w is not None:
        wAN[i] = lw.attn_norm_w.ctypes.data_as(cv)
    if hasattr(lw, 'ffn_norm_w') and lw.ffn_norm_w is not None:
        wFN[i] = lw.ffn_norm_w.ctypes.data_as(cv)
conv_s = time.perf_counter() - t0
print(f"  Conversion: {conv_s:.1f}s", flush=True)

print("Loading MoE expert pointers...", flush=True)
wgi_arr = [0]*L; wge_arr = [0]*L; wue_arr = [0]*L; wde_arr = [0]*L
for i, me in enumerate(e._moe_layers):
    wgi_arr[i] = me.router_f32.ctypes.data_as(cv) if me.router_f32 is not None else cv(0)
    if me.gate_raw and len(me.gate_raw) > 0:
        wge_arr[i] = me.gate_raw[0].ctypes.data_as(cv)
        wG[i] = me.gate_raw[0].ctypes.data_as(cv); nG[i] = me.gate_nr.value; qG[i] = me.gate_qt.value
    if me.up_raw and len(me.up_raw) > 0:
        wue_arr[i] = me.up_raw[0].ctypes.data_as(cv)
        wU[i] = me.up_raw[0].ctypes.data_as(cv); nU[i] = me.up_nr.value; qU[i] = me.up_qt.value
    if me.down_raw and len(me.down_raw) > 0:
        wde_arr[i] = me.down_raw[0].ctypes.data_as(cv)
        wD[i] = me.down_raw[0].ctypes.data_as(cv); nD[i] = me.down_nr.value; qD[i] = me.down_qt.value

gate_qt = qG[0] if qG[0] else 39; up_qt = qU[0] if qU[0] else 39; down_qt = qD[0] if qD[0] else 39

# ── BC struct ──
class PageTable(ctypes.Structure):
    _fields_ = [('page_size', ci), ('n_pages', ci), ('pages', cv), ('table', cv),
                ('free_pages', cv), ('free_count', ci), ('_nkh', ci), ('_hd', ci), ('_n_layers', ci)]
class KVBlock(ctypes.Structure):
    _fields_ = [('k', cv), ('v', cv), ('n_blocks', ci), ('seq_len', ci*64),
                ('block_map', (ci*1024)*64), ('pt', PageTable)]
class BC(ctypes.Structure):
    _fields_ = [
        ('L', ci), ('N', ci), ('NH', ci), ('NKH', ci), ('HD', ci), ('FF', ci), ('V', ci), ('eps', cf),
        ('wQ', cv), ('wK', cv), ('wV', cv), ('wO', cv), ('wG', cv), ('wU', cv), ('wD', cv),
        ('wAN', cv), ('wFN', cv),
        ('nQ', cv), ('nK', cv), ('nV', cv), ('nO', cv), ('nG', cv), ('nU', cv), ('nD', cv),
        ('nc', ci), ('emb', cv), ('onw', cv), ('wOut', cv), ('outNR', ci), ('outNC', ci), ('outQuant', ci),
        ('kv_array', cv), ('logits', cv),
        ('n_experts', ci), ('n_experts_per_tok', ci), ('moe_intermediate', ci),
        ('w_gate_inp', cv), ('w_gate_exps', cv), ('w_up_exps', cv), ('w_down_exps', cv),
        ('gate_exp_quant', ci), ('up_exp_quant', ci), ('down_exp_quant', ci),
        ('q_quant', cv), ('k_quant', cv), ('v_quant', cv), ('o_quant', cv),
        ('g_quant', cv), ('u_quant', cv), ('d_quant', cv), ('emb_quant', ci),
        ('wQK', cv), ('qk_quant', cv), ('cos_table', cv), ('sin_table', cv), ('max_ctx', ci),
        ('workspace', cv), ('ws_size', ci),
        ('rope_dim', ci), ('full_attn_interval', ci), ('wQKV', cv), ('qkv_quant', cv),
        ('wAttnG', cv), ('attnG_quant', cv),
        ('ssm_conv1d', cv), ('ssm_a', cv), ('ssm_dt_bias', cv), ('ssm_alpha', cv),
        ('ssm_beta', cv), ('ssm_norm', cv), ('wSsmOut', cv), ('ssm_out_quant', cv), ('ssm_state', cv),
        ('wSHexpG', cv), ('wSHexpU', cv), ('wSHexpD', cv),
        ('shexp_g_quant', cv), ('shexp_u_quant', cv), ('shexp_d_quant', cv), ('wShexpRouter', cv),
        ('layer_types', cv), ('n_layers_actual', ci)
    ]

MAX_CTX = 4096
bc = BC()
for a, v in [('L', L), ('N', N), ('NH', NH), ('NKH', NKH), ('HD', HD), ('FF', FF), ('V', V), ('eps', e.eps),
             ('nc', N), ('outNR', V), ('outNC', N), ('outQuant', 8),
             ('n_experts', e.n_experts), ('n_experts_per_tok', e.n_experts_per_tok),
             ('moe_intermediate', e.n_ff_expert),
             ('gate_exp_quant', gate_qt), ('up_exp_quant', up_qt), ('down_exp_quant', down_qt),
             ('emb_quant', 0), ('max_ctx', MAX_CTX),
             ('rope_dim', HD), ('full_attn_interval', 0), ('n_layers_actual', L)]:
    setattr(bc, a, v)

bc.wQ = ctypes.cast(wa(wQ), cv); bc.wK = ctypes.cast(wa(wK), cv)
bc.wV = ctypes.cast(wa(wV), cv); bc.wO = ctypes.cast(wa(wO), cv)
bc.wG = ctypes.cast(wa(wG), cv); bc.wU = ctypes.cast(wa(wU), cv); bc.wD = ctypes.cast(wa(wD), cv)
bc.wAN = ctypes.cast(wa(wAN), cv); bc.wFN = ctypes.cast(wa(wFN), cv)
bc.nQ = ctypes.cast(ia(nQ), cv); bc.nK = ctypes.cast(ia(nK), cv)
bc.nV = ctypes.cast(ia(nV), cv); bc.nO = ctypes.cast(ia(nO), cv)
bc.nG = ctypes.cast(ia(nG), cv); bc.nU = ctypes.cast(ia(nU), cv); bc.nD = ctypes.cast(ia(nD), cv)
bc.q_quant = ctypes.cast(ia(qQ), cv); bc.k_quant = ctypes.cast(ia(qK), cv)
bc.v_quant = ctypes.cast(ia(qV), cv); bc.o_quant = ctypes.cast(ia(qO), cv)
bc.g_quant = ctypes.cast(ia(qG), cv); bc.u_quant = ctypes.cast(ia(qU), cv); bc.d_quant = ctypes.cast(ia(qD), cv)
bc.w_gate_inp = ctypes.cast(wa(wgi_arr), cv)
bc.w_gate_exps = ctypes.cast(wa(wge_arr), cv)
bc.w_up_exps = ctypes.cast(wa(wue_arr), cv)
bc.w_down_exps = ctypes.cast(wa(wde_arr), cv)
bc.emb = e.emb.ctypes.data_as(cv) if hasattr(e, 'emb') and e.emb is not None else cv(0)
bc.onw = e._out_norm_w.ctypes.data_as(cv) if hasattr(e, '_out_norm_w') else cv(0)
if hasattr(e, '_out_raw') and e._out_raw is not None:
    bc.wOut = ctypes.cast(e._out_raw, cv)
elif hasattr(e, 'weights'):
    for k in ('output.weight', 'lm_head.weight'):
        if k in e.weights and e.weights[k] is not None:
            ow = np.ascontiguousarray(e.weights[k].astype(np.float32)); _q8_bufs.append(ow)
            bc.wOut = ow.ctypes.data_as(cv); break
for pf in ['wQKV','wAttnG','ssm_conv1d','ssm_a','ssm_dt_bias','ssm_alpha','ssm_beta',
           'ssm_norm','wSsmOut','ssm_state','wSHexpG','wSHexpU','wSHexpD','wShexpRouter',
           'layer_types','qkv_quant','attnG_quant','ssm_out_quant','shexp_g_quant',
           'shexp_u_quant','shexp_d_quant']:
    setattr(bc, pf, cv(0))
bc.wQK = cv(0); bc.qk_quant = cv(0)

cos_t = np.zeros(MAX_CTX * HD // 2, dtype=np.float32)
sin_t = np.zeros(MAX_CTX * HD // 2, dtype=np.float32)
for p in range(MAX_CTX):
    for j in range(HD // 2):
        ang = p / (10000.0 ** (2.0 * j / HD))
        cos_t[p * HD // 2 + j] = np.cos(ang); sin_t[p * HD // 2 + j] = np.sin(ang)
bc.cos_table = cos_t.ctypes.data_as(cv); bc.sin_table = sin_t.ctypes.data_as(cv)

S_actual = max(N, NH*HD, FF, NKH*HD, NH*HD+NKH*HD, 8192)
KV_scratch = 2 * MAX_CTX * NKH * HD
min_ws = 12 * N_USERS * S_actual + KV_scratch + 100000
ws = np.zeros(min_ws, dtype=np.float32)
bc.workspace = ws.ctypes.data_as(cv)
bc.ws_size = min_ws
print(f"  Workspace: {min_ws:,} floats ({min_ws*4/1024/1024:.0f} MB)", flush=True)

lib.kv_init.argtypes = [cv, ci, ci, ci]; lib.kv_init.restype = None
lib.batch_forward.argtypes = [cv, cv, ci, cv]; lib.batch_forward.restype = None
lib.rope_init.argtypes = [cv, cv, ci, ci, ci]; lib.rope_init.restype = None
lib.rope_init(bc.cos_table, bc.sin_table, ci(MAX_CTX), ci(HD), ci(HD))

# Helper: call batch_forward with correct ws pointer
ws_ptr = ws.ctypes.data_as(cv)
def bf(tokens, B):
    lib.batch_forward(
        ctypes.byref(bc),
        (ci * B)(*tokens) if isinstance(tokens, (list, tuple)) else tokens,
        ci(B), ws_ptr
    )

# ── Step 1: B=1 test ──
print("\nStep 1: batch_forward B=1...", flush=True)
kv1 = KVBlock()
lib.kv_init(ctypes.byref(kv1), ci(L), ci(NKH), ci(HD))
kv_arr1 = (cv*1)(ctypes.cast(ctypes.pointer(kv1), cv))
bc.kv_array = ctypes.cast(kv_arr1, cv)
lbuf1 = np.zeros(V, dtype=np.float32)
bc.logits = lbuf1.ctypes.data_as(cv)

try:
    t0 = time.perf_counter()
    bf([1], 1)
    elapsed = time.perf_counter() - t0
    maxv = float(lbuf1.max()); nan_c = int(np.isnan(lbuf1).sum())
    print(f"  B=1: max={maxv:.1f}, nan={nan_c}, {elapsed*1000:.0f}ms", flush=True)
    if nan_c == 0 and maxv > 0: print("  ✅ B=1 PASS", flush=True)
    else: print("  ❌ B=1 FAIL", flush=True); sys.exit(1)
except Exception as ex:
    print(f"  ❌ B=1 CRASH: {ex}", flush=True); import traceback; traceback.print_exc(); sys.exit(1)

# ── Step 2: B=10 same token ──
print(f"\nStep 2: batch_forward B={N_USERS}...", flush=True)
kvs = [KVBlock() for _ in range(N_USERS)]
for kv in kvs: lib.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
bc.kv_array = ctypes.cast((cv*N_USERS)(*[ctypes.cast(ctypes.pointer(k), cv) for k in kvs]), cv)
lbuf = np.zeros(N_USERS * V, dtype=np.float32)
bc.logits = lbuf.ctypes.data_as(cv)

try:
    t0 = time.perf_counter()
    bf([1]*N_USERS, N_USERS)
    elapsed = time.perf_counter() - t0
    ok = all(np.isnan(lbuf[u*V:(u+1)*V]).sum() == 0 for u in range(N_USERS))
    if ok: print(f"  B=10: max={lbuf.max():.1f}, {elapsed*1000:.0f}ms", flush=True); print("  ✅ B=10 PASS", flush=True)
    else: print("  ❌ B=10 NaN", flush=True); sys.exit(1)
except Exception as ex:
    print(f"  ❌ B=10 CRASH: {ex}", flush=True); sys.exit(1)

# ── Step 3: 512-tok prefill + 20-tok gen ──
print(f"\nStep 3: {PROMPT_LEN}-tok prefill + {GEN_TOKENS}-tok gen", flush=True)
for kv in kvs: kv.seq_len[0] = 0
print("  Warmup (3 steps)...", flush=True)
for _ in range(3): bf([100]*N_USERS, N_USERS)
for kv in kvs: kv.seq_len[0] = 0

print(f"  Prefill {PROMPT_LEN}x{N_USERS} = {PROMPT_LEN*N_USERS} tokens...", flush=True)
t0 = time.perf_counter()
for pos in range(PROMPT_LEN):
    bf([100+(pos%50000)]*N_USERS, N_USERS)
    if pos % 100 == 99: print(f"    pos {pos+1}/{PROMPT_LEN}  ({time.perf_counter()-t0:.1f}s)", flush=True)
prefill_s = time.perf_counter() - t0
print(f"    Prefill done in {prefill_s:.1f}s", flush=True)

print(f"  Generate {GEN_TOKENS} tokens...", flush=True)
gen_t0 = time.perf_counter()
for step in range(GEN_TOKENS):
    if step == 0: bf([1]*N_USERS, N_USERS)
    else: bf([int(np.argmax(lbuf[u*V:(u+1)*V])) for u in range(N_USERS)], N_USERS)
gen_s = time.perf_counter() - gen_t0
total_s = prefill_s + gen_s
tps = N_USERS * GEN_TOKENS / gen_s
lat_ms = gen_s / GEN_TOKENS * 1000

print(f"\n  ╔══════════════════════════════════════╗")
print(f"  ║  GPT-OSS-20B batch B={N_USERS} Results     ║")
print(f"  ╠══════════════════════════════════════╣")
print(f"  ║  Prefill: {prefill_s:.1f}s ({PROMPT_LEN} steps)       ║")
print(f"  ║  Generate: {gen_s*1000:.0f}ms ({GEN_TOKENS} steps)    ║")
print(f"  ║  Throughput: {tps:.0f} tok/s ({tps/N_USERS:.0f}/user) ║")
print(f"  ║  Avg step: {lat_ms:.0f}ms                  ║")
print(f"  ╟──────────────────────────────────────╢")
ref = 9.1
ratio = tps / ref
if ratio > 1:
    print(f"  ║  VS llama.cpp 512-prefill: {ref} tok/s            ║")
    print(f"  ║  ✅ MojoLlama {ratio:.1f}x FASTER!                          ║")
else:
    print(f"  ║  VS llama.cpp: {ref} tok/s → {ratio:.1f}x        ║")
print(f"  ╚══════════════════════════════════════╝")
