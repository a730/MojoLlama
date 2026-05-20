#!/usr/bin/env python3
"""Qwen3.6 MXFP4 multi-user concurrency benchmark (C engine batch_forward)."""
import sys, os, time, ctypes, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"
N_USERS = 10
PROMPT_LEN = 128  # shorter prefill for faster benchmark
GEN_TOKENS = 20

# ── Load engine ──
print(f"Loading Qwen3.6 MXFP4 ({N_USERS} users, {PROMPT_LEN}-tok prefill + {GEN_TOKENS}-tok gen)...", flush=True)
from turbo_engine_v7_moe import TurboEngineV7MoE
e = TurboEngineV7MoE(MODEL, 32)
N=e.n_embd; NH=e.n_head; NKH=e.n_kv_head; HD=e.head_dim
FF=e.n_ff; L=e.n_layers; V=e.vocab_size
EXP=e.n_experts; ET=e.n_experts_per_tok; EFF=e.n_ff_expert
print(f"  {L}L/{N}D/{NH}H/{NKH}KV/{HD}hd MoE {EXP}x{ET} ef={EFF}", flush=True)

libfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels', 'cengine_batch_instr.so')
ce = ctypes.CDLL(libfile)
cv=ctypes.c_void_p; ci=ctypes.c_int; cf=ctypes.c_float
def wa(arr): return (cv*L)(*[ctypes.cast(a,cv) for a in arr])
def ia(arr): return (ci*L)(*[int(a) for a in arr])

# ── Build BC struct ──
LT = e.layer_types if hasattr(e, 'layer_types') else [0]*L

wQKV = [cv(0)]*L; qkv_qt = [0]*L
wQ_sep = [cv(0)]*L; nQ_sep = [0]*L; qQ_sep = [0]*L
wK_sep = [cv(0)]*L; nK_sep = [0]*L; qK_sep = [0]*L
wV_sep = [cv(0)]*L; nV_sep = [0]*L; qV_sep = [0]*L
wAG = [cv(0)]*L; ag_qt = [0]*L
wO_attn = [cv(0)]*L; nO_attn = [0]*L; qO_attn = [0]*L
wSO = [cv(0)]*L; so_qt = [0]*L
ssm_c1d = [cv(0)]*L; ssm_a = [cv(0)]*L; ssm_db = [cv(0)]*L
ssm_al = [cv(0)]*L; ssm_be = [cv(0)]*L; ssm_nm = [cv(0)]*L
wSHG = [cv(0)]*L; wSHU = [cv(0)]*L; wSHD = [cv(0)]*L; wSHR = [cv(0)]*L
shg_qt = [0]*L; shu_qt = [0]*L; shd_qt = [0]*L
wAN = [0]*L; wFN = [0]*L
wGI = [cv(0)]*L; wGE = [cv(0)]*L; wUE = [cv(0)]*L; wDE = [cv(0)]*L
ns_G = [0]*L; ns_U = [0]*L; ns_D = [0]*L
qt_G = [0]*L; qt_U = [0]*L; qt_D = [0]*L

print("Building weight arrays...", flush=True)
for i in range(L):
    lw = e._layers[i]; me = e._moe_layers[i]
    if hasattr(lw,'attn_qkv_raw') and lw.attn_qkv_raw is not None:
        wQKV[i]=lw.attn_qkv_raw; qkv_qt[i]=lw.attn_qkv_qt.value
    if hasattr(lw,'attn_q_raw') and lw.attn_q_raw is not None:
        wQ_sep[i]=lw.attn_q_raw; nQ_sep[i]=lw.attn_q_nr.value; qQ_sep[i]=lw.attn_q_qt.value
    if hasattr(lw,'attn_k_raw') and lw.attn_k_raw is not None:
        wK_sep[i]=lw.attn_k_raw; nK_sep[i]=lw.attn_k_nr.value; qK_sep[i]=lw.attn_k_qt.value
    if hasattr(lw,'attn_v_raw') and lw.attn_v_raw is not None:
        wV_sep[i]=lw.attn_v_raw; nV_sep[i]=lw.attn_v_nr.value; qV_sep[i]=lw.attn_v_qt.value
    if hasattr(lw,'attn_gate_raw') and lw.attn_gate_raw is not None:
        wAG[i]=lw.attn_gate_raw; ag_qt[i]=lw.attn_gate_qt.value
    if hasattr(lw,'attn_out_raw') and lw.attn_out_raw is not None:
        wO_attn[i]=lw.attn_out_raw; nO_attn[i]=lw.attn_out_nr.value; qO_attn[i]=lw.attn_out_qt.value
    if hasattr(lw,'ssm_conv1d_ptr') and lw.ssm_conv1d_ptr is not None:
        ssm_c1d[i]=lw.ssm_conv1d_ptr; ssm_a[i]=lw.ssm_a_ptr; ssm_db[i]=lw.ssm_dt_bias_ptr
        ssm_al[i]=lw.ssm_alpha_ptr; ssm_be[i]=lw.ssm_beta_ptr; ssm_nm[i]=lw.ssm_norm_ptr
    if hasattr(lw,'ssm_out_raw') and lw.ssm_out_raw is not None:
        wSO[i]=lw.ssm_out_raw; so_qt[i]=lw.ssm_out_qt.value
    if hasattr(lw,'shexp_gate_raw') and lw.shexp_gate_raw is not None:
        wSHG[i]=lw.shexp_gate_raw; shg_qt[i]=lw.shexp_gate_qt.value
    if hasattr(lw,'shexp_up_raw') and lw.shexp_up_raw is not None:
        wSHU[i]=lw.shexp_up_raw; shu_qt[i]=lw.shexp_up_qt.value
    if hasattr(lw,'shexp_down_raw') and lw.shexp_down_raw is not None:
        wSHD[i]=lw.shexp_down_raw; shd_qt[i]=lw.shexp_down_qt.value
    if hasattr(lw,'shexp_router_ptr') and lw.shexp_router_ptr is not None:
        wSHR[i]=lw.shexp_router_ptr
    if hasattr(me,'router_f32') and me.router_f32 is not None:
        wGI[i]=me.router_f32.ctypes.data_as(cv)
    if me.gate_raw and len(me.gate_raw)>0:
        wGE[i]=me.gate_raw[0].ctypes.data_as(cv); ns_G[i]=me.gate_nr.value; qt_G[i]=me.gate_qt.value
    if me.up_raw and len(me.up_raw)>0:
        wUE[i]=me.up_raw[0].ctypes.data_as(cv); ns_U[i]=me.up_nr.value; qt_U[i]=me.up_qt.value
    if me.down_raw and len(me.down_raw)>0:
        wDE[i]=me.down_raw[0].ctypes.data_as(cv); ns_D[i]=me.down_nr.value; qt_D[i]=me.down_qt.value
    if hasattr(lw,'attn_norm_w') and lw.attn_norm_w is not None: wAN[i]=lw.attn_norm_w.ctypes.data_as(cv)
    if hasattr(lw,'ffn_norm_w') and lw.ffn_norm_w is not None: wFN[i]=lw.ffn_norm_w.ctypes.data_as(cv)

me0 = e._moe_layers[0]
gate_qt_all = me0.gate_qt.value; up_qt_all = me0.up_qt.value; down_qt_all = me0.down_qt.value

class PageTable(ctypes.Structure):
    _fields_=[('page_size',ci),('n_pages',ci),('pages',cv),('table',cv),('free_pages',cv),('free_count',ci),('_nkh',ci),('_hd',ci),('_n_layers',ci)]
class KVBlock(ctypes.Structure):
    _fields_=[('k',cv),('v',cv),('n_blocks',ci),('seq_len',ci*64),('block_map',(ci*1024)*64),('pt',PageTable)]
class BC(ctypes.Structure):
    _fields_=[
        ('L',ci),('N',ci),('NH',ci),('NKH',ci),('HD',ci),('FF',ci),('V',ci),('eps',cf),
        ('wQ',cv),('wK',cv),('wV',cv),('wO',cv),('wG',cv),('wU',cv),('wD',cv),
        ('wAN',cv),('wFN',cv),('nQ',cv),('nK',cv),('nV',cv),('nO',cv),('nG',cv),('nU',cv),('nD',cv),
        ('nc',ci),('emb',cv),('onw',cv),('wOut',cv),('outNR',ci),('outNC',ci),('outQuant',ci),
        ('kv_array',cv),('logits',cv),
        ('n_experts',ci),('n_experts_per_tok',ci),('moe_intermediate',ci),
        ('w_gate_inp',cv),('w_gate_exps',cv),('w_up_exps',cv),('w_down_exps',cv),
        ('gate_exp_quant',ci),('up_exp_quant',ci),('down_exp_quant',ci),
        ('q_quant',cv),('k_quant',cv),('v_quant',cv),('o_quant',cv),
        ('g_quant',cv),('u_quant',cv),('d_quant',cv),('emb_quant',ci),
        ('wQK',cv),('qk_quant',cv),('cos_table',cv),('sin_table',cv),('max_ctx',ci),
        ('workspace',cv),('ws_size',ci),
        ('rope_dim',ci),('full_attn_interval',ci),
        ('wQKV',cv),('qkv_quant',cv),('wAttnG',cv),('attnG_quant',cv),
        ('ssm_conv1d',cv),('ssm_a',cv),('ssm_dt_bias',cv),('ssm_alpha',cv),
        ('ssm_beta',cv),('ssm_norm',cv),('wSsmOut',cv),('ssm_out_quant',cv),('ssm_state',cv),
        ('wSHexpG',cv),('wSHexpU',cv),('wSHexpD',cv),
        ('shexp_g_quant',cv),('shexp_u_quant',cv),('shexp_d_quant',cv),('wShexpRouter',cv),
        ('layer_types',cv),('n_layers_actual',ci)
    ]

MAX_CTX=4096
bc=BC()
for a,v in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',e.eps),
            ('nc',N),('outNR',V),('outNC',N),('outQuant',8),
            ('n_experts',EXP),('n_experts_per_tok',ET),('moe_intermediate',EFF),
            ('gate_exp_quant',gate_qt_all),('up_exp_quant',up_qt_all),('down_exp_quant',down_qt_all),
            ('emb_quant',0),('max_ctx',MAX_CTX),
            ('rope_dim',HD),('full_attn_interval',4),('n_layers_actual',L)]:
    setattr(bc,a,v)

bc.wQKV = ctypes.cast(wa(wQKV),cv); bc.qkv_quant = ctypes.cast(ia(qkv_qt),cv)
bc.wQ = ctypes.cast(wa(wQ_sep),cv); bc.wK = ctypes.cast(wa(wK_sep),cv); bc.wV = ctypes.cast(wa(wV_sep),cv)
bc.nQ = ctypes.cast(ia(nQ_sep),cv); bc.nK = ctypes.cast(ia(nK_sep),cv); bc.nV = ctypes.cast(ia(nV_sep),cv)
bc.q_quant = ctypes.cast(ia(qQ_sep),cv); bc.k_quant = ctypes.cast(ia(qK_sep),cv); bc.v_quant = ctypes.cast(ia(qV_sep),cv)
bc.wAttnG = ctypes.cast(wa(wAG),cv); bc.attnG_quant = ctypes.cast(ia(ag_qt),cv)
bc.wO = ctypes.cast(wa(wO_attn),cv); bc.nO = ctypes.cast(ia(nO_attn),cv); bc.o_quant = ctypes.cast(ia(qO_attn),cv)
bc.wSsmOut = ctypes.cast(wa(wSO),cv); bc.ssm_out_quant = ctypes.cast(ia(so_qt),cv)
bc.ssm_conv1d = ctypes.cast(wa(ssm_c1d),cv); bc.ssm_a = ctypes.cast(wa(ssm_a),cv)
bc.ssm_dt_bias = ctypes.cast(wa(ssm_db),cv); bc.ssm_alpha = ctypes.cast(wa(ssm_al),cv)
bc.ssm_beta = ctypes.cast(wa(ssm_be),cv); bc.ssm_norm = ctypes.cast(wa(ssm_nm),cv)
bc.wSHexpG = ctypes.cast(wa(wSHG),cv); bc.wSHexpU = ctypes.cast(wa(wSHU),cv)
bc.wSHexpD = ctypes.cast(wa(wSHD),cv); bc.wShexpRouter = ctypes.cast(wa(wSHR),cv)
bc.shexp_g_quant = ctypes.cast(ia(shg_qt),cv); bc.shexp_u_quant = ctypes.cast(ia(shu_qt),cv)
bc.shexp_d_quant = ctypes.cast(ia(shd_qt),cv)
bc.wAN = ctypes.cast(wa(wAN),cv); bc.wFN = ctypes.cast(wa(wFN),cv)
bc.w_gate_inp = ctypes.cast(wa(wGI),cv)
bc.w_gate_exps = ctypes.cast(wa(wGE),cv); bc.w_up_exps = ctypes.cast(wa(wUE),cv); bc.w_down_exps = ctypes.cast(wa(wDE),cv)
bc.nG = ctypes.cast(ia(ns_G),cv); bc.nU = ctypes.cast(ia(ns_U),cv); bc.nD = ctypes.cast(ia(ns_D),cv)
bc.g_quant = ctypes.cast(ia(qt_G),cv); bc.u_quant = ctypes.cast(ia(qt_U),cv); bc.d_quant = ctypes.cast(ia(qt_D),cv)
for pf in ['wG','wU','wD','nG','nU','nD','g_quant','u_quant','d_quant','wQK','qk_quant']:
    setattr(bc,pf,cv(0))
bc.emb = e.emb.ctypes.data_as(cv) if hasattr(e,'emb') and e.emb is not None else cv(0)
bc.onw = e._out_norm_w.ctypes.data_as(cv) if hasattr(e,'_out_norm_w') else cv(0)
bc.wOut = ctypes.cast(e._out_raw, cv)
ssm_state = np.zeros(L * e.ssm_groups * e.ssm_state_size, dtype=np.float32)
bc.ssm_state = ssm_state.ctypes.data_as(cv)

lt_arr = (ci * L)(*LT); bc.layer_types = ctypes.cast(lt_arr, cv)
cos_t = np.zeros(MAX_CTX*HD//2,np.float32); sin_t = np.zeros(MAX_CTX*HD//2,np.float32)
for p in range(MAX_CTX):
    for j in range(HD//2):
        ang = p/(10000.0**(2.0*j/HD))
        cos_t[p*HD//2+j]=np.cos(ang); sin_t[p*HD//2+j]=np.sin(ang)
bc.cos_table=cos_t.ctypes.data_as(cv); bc.sin_table=sin_t.ctypes.data_as(cv)

S_act = max(N, NH*HD, FF, NKH, NH*HD+NKH, 8192)
KVscr = 2 * MAX_CTX * NKH * HD
ws = np.zeros(12 * N_USERS * S_act + KVscr + 100000, dtype=np.float32)
bc.workspace = ws.ctypes.data_as(cv); bc.ws_size = len(ws)
print(f"  Workspace: {len(ws):,} floats ({len(ws)*4/1024/1024:.0f} MB)", flush=True)

ce.kv_init.argtypes=[cv,ci,ci,ci]; ce.kv_init.restype=None
ce.batch_forward.argtypes=[cv,cv,ci,cv]; ce.batch_forward.restype=None
ce.rope_init.argtypes=[cv,cv,ci,ci,ci]; ce.rope_init.restype=None
ce.rope_init(bc.cos_table, bc.sin_table, ci(MAX_CTX), ci(HD), ci(HD))

ws_ptr = ws.ctypes.data_as(cv)
def bf(tokens, B):
    ce.batch_forward(ctypes.byref(bc), (ci*B)(*tokens), ci(B), ws_ptr)

# ═══════════════════════════════════════
#  BENCHMARK: B=1 baseline
# ═══════════════════════════════════════
print(f"\nStep 1: B=1 baseline...", flush=True)
kv_single = KVBlock(); ce.kv_init(ctypes.byref(kv_single), ci(L), ci(NKH), ci(HD))
bc.kv_array = ctypes.cast((cv*1)(ctypes.cast(ctypes.pointer(kv_single), cv)), cv)
lbuf = np.zeros(V, dtype=np.float32)
bc.logits = lbuf.ctypes.data_as(cv)

for _ in range(3): bf([1], 1); kv_single.seq_len[0] = 0
t1 = []
for _ in range(30):
    t0 = time.perf_counter(); bf([1], 1); t1.append(time.perf_counter()-t0)
    kv_single.seq_len[0] = 0
t1 = t1[5:]
avg1 = np.mean(t1)
print(f"  B=1: {avg1*1000:.1f}ms → {1/avg1:.1f} tok/s, max={lbuf.max():.1f}", flush=True)

# ═══════════════════════════════════════
#  BENCHMARK: B=10 single step
# ═══════════════════════════════════════
print(f"\nStep 2: B={N_USERS} single step...", flush=True)
kvs = [KVBlock() for _ in range(N_USERS)]
for kv in kvs: ce.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
bc.kv_array = ctypes.cast((cv*N_USERS)(*[ctypes.cast(ctypes.pointer(k), cv) for k in kvs]), cv)
lbuf10 = np.zeros(N_USERS * V, dtype=np.float32)
bc.logits = lbuf10.ctypes.data_as(cv)

t10 = []
for _ in range(10):
    t0 = time.perf_counter(); bf([1]*N_USERS, N_USERS)
    t10.append(time.perf_counter()-t0)
    for kv in kvs: kv.seq_len[0] = 0
t10 = t10[3:]
avg10 = np.mean(t10)
ok = all(np.isnan(lbuf10[u*V:(u+1)*V]).sum() == 0 for u in range(N_USERS))
print(f"  B={N_USERS}: {avg10*1000:.1f}ms → {N_USERS/avg10:.1f} tok/s aggregate", flush=True)
print(f"  Per-user: {N_USERS/avg10/N_USERS:.1f} tok/s/user (no, aggregate/{N_USERS})", flush=True)
print(f"  Correct: {'✅' if ok else '❌'} NaN={int(np.isnan(lbuf10).sum())}", flush=True)

# ═══════════════════════════════════════
#  BENCHMARK: B=10 prefill + gen
# ═══════════════════════════════════════
print(f"\nStep 3: {PROMPT_LEN}-tok prefill + {GEN_TOKENS}-tok gen, B={N_USERS}...", flush=True)
for kv in kvs: kv.seq_len[0] = 0; ce.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))

print(f"  Prefill {PROMPT_LEN} steps...", flush=True)
pt0 = time.perf_counter()
for pos in range(PROMPT_LEN):
    bf([100+(pos%50000)]*N_USERS, N_USERS)
    if pos % 32 == 31: print(f"    pos {pos+1}/{PROMPT_LEN}  ({time.perf_counter()-pt0:.1f}s)", flush=True)
prefill_s = time.perf_counter() - pt0
print(f"    Prefill: {prefill_s:.1f}s", flush=True)

print(f"  Generate {GEN_TOKENS} tokens...", flush=True)
gt0 = time.perf_counter()
for step in range(GEN_TOKENS):
    if step == 0: bf([1]*N_USERS, N_USERS)
    else: bf([int(np.argmax(lbuf10[u*V:(u+1)*V])) for u in range(N_USERS)], N_USERS)
gen_s = time.perf_counter() - gt0
tps = N_USERS * GEN_TOKENS / gen_s

print(f"\n  ╔══════════════════════════════════════════════╗")
print(f"  ║  Qwen3.6 MXFP4 batch B={N_USERS} Results             ║")
print(f"  ╠══════════════════════════════════════════════╣")
print(f"  ║  C engine B=1:         {1000*avg1:6.0f}ms  ({1/avg1:5.1f} tok/s)   ║")
print(f"  ║  C engine B={N_USERS}:          {1000*avg10:6.0f}ms  ({N_USERS/avg10:5.1f} tok/s aggr) ║")
print(f"  ║  Prefill ({PROMPT_LEN} steps):     {prefill_s:6.1f}s                ║")
print(f"  ║  Generate ({GEN_TOKENS} steps):     {gen_s*1000:6.0f}ms               ║")
print(f"  ║  Throughput:            {tps:6.0f} tok/s ({tps/N_USERS:.0f}/user)   ║")
print(f"  ╟──────────────────────────────────────────────╢")
print(f"  ║  VS Python single:      39.2ms → 25.5 tok/s               ║")
print(f"  ║  VS llama.cpp:             ~16 tok/s single                 ║")
print(f"  ║  Efficiency (B=10/B=1): {N_USERS/avg10/(1/avg1):.2f}x                      ║")
print(f"  ╚══════════════════════════════════════════════╝")
