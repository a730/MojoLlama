#!/usr/bin/env python3
"""Qwen3.6-35B MXFP4 in C engine batch_forward B=1 benchmark."""
import sys, os, time, ctypes, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"

print("Loading engine...", flush=True)
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

# ── Collect all weight arrays ──
LT = e.layer_types if hasattr(e, 'layer_types') else [0]*L

# Attention paths
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
# MoE
wGI = [cv(0)]*L; wGE = [cv(0)]*L; wUE = [cv(0)]*L; wDE = [cv(0)]*L
ns_G = [0]*L; ns_U = [0]*L; ns_D = [0]*L
qt_G = [0]*L; qt_U = [0]*L; qt_D = [0]*L

print("Building weight arrays...", flush=True)
for i in range(L):
    lw = e._layers[i]; me = e._moe_layers[i]
    # QKV fused (attn layers)
    if hasattr(lw,'attn_qkv_raw') and lw.attn_qkv_raw is not None:
        wQKV[i]=lw.attn_qkv_raw; qkv_qt[i]=lw.attn_qkv_qt.value
    # Separate Q/K/V (SSM-sub layers)
    if hasattr(lw,'attn_q_raw') and lw.attn_q_raw is not None:
        wQ_sep[i]=lw.attn_q_raw; nQ_sep[i]=lw.attn_q_nr.value; qQ_sep[i]=lw.attn_q_qt.value
    if hasattr(lw,'attn_k_raw') and lw.attn_k_raw is not None:
        wK_sep[i]=lw.attn_k_raw; nK_sep[i]=lw.attn_k_nr.value; qK_sep[i]=lw.attn_k_qt.value
    if hasattr(lw,'attn_v_raw') and lw.attn_v_raw is not None:
        wV_sep[i]=lw.attn_v_raw; nV_sep[i]=lw.attn_v_nr.value; qV_sep[i]=lw.attn_v_qt.value
    # Attention gate (all layers)
    if hasattr(lw,'attn_gate_raw') and lw.attn_gate_raw is not None:
        wAG[i]=lw.attn_gate_raw; ag_qt[i]=lw.attn_gate_qt.value
    # Attention output (for layers without attn_gate)
    if hasattr(lw,'attn_out_raw') and lw.attn_out_raw is not None:
        wO_attn[i]=lw.attn_out_raw
        nO_attn[i]=lw.attn_out_nr.value if hasattr(lw,'attn_out_nr') else N
        qO_attn[i]=lw.attn_out_qt.value if hasattr(lw,'attn_out_qt') else 8
    # SSM specific
    if hasattr(lw,'ssm_conv1d_ptr') and lw.ssm_conv1d_ptr is not None:
        ssm_c1d[i]=lw.ssm_conv1d_ptr; ssm_a[i]=lw.ssm_a_ptr; ssm_db[i]=lw.ssm_dt_bias_ptr
        ssm_al[i]=lw.ssm_alpha_ptr; ssm_be[i]=lw.ssm_beta_ptr; ssm_nm[i]=lw.ssm_norm_ptr
    if hasattr(lw,'ssm_out_raw') and lw.ssm_out_raw is not None:
        wSO[i]=lw.ssm_out_raw; so_qt[i]=lw.ssm_out_qt.value
    # Shared expert
    if hasattr(lw,'shexp_gate_raw') and lw.shexp_gate_raw is not None:
        wSHG[i]=lw.shexp_gate_raw; shg_qt[i]=lw.shexp_gate_qt.value
    if hasattr(lw,'shexp_up_raw') and lw.shexp_up_raw is not None:
        wSHU[i]=lw.shexp_up_raw; shu_qt[i]=lw.shexp_up_qt.value
    if hasattr(lw,'shexp_down_raw') and lw.shexp_down_raw is not None:
        wSHD[i]=lw.shexp_down_raw; shd_qt[i]=lw.shexp_down_qt.value
    if hasattr(lw,'shexp_router_ptr') and lw.shexp_router_ptr is not None:
        wSHR[i]=lw.shexp_router_ptr
    # MoE router
    if hasattr(me,'router_f32') and me.router_f32 is not None:
        wGI[i]=me.router_f32.ctypes.data_as(cv)
    # MoE experts (first expert's pointer — stride handles rest)
    if me.gate_raw and len(me.gate_raw)>0:
        wGE[i]=me.gate_raw[0].ctypes.data_as(cv)
        ns_G[i]=me.gate_nr.value; qt_G[i]=me.gate_qt.value
    if me.up_raw and len(me.up_raw)>0:
        wUE[i]=me.up_raw[0].ctypes.data_as(cv)
        ns_U[i]=me.up_nr.value; qt_U[i]=me.up_qt.value
    if me.down_raw and len(me.down_raw)>0:
        wDE[i]=me.down_raw[0].ctypes.data_as(cv)
        ns_D[i]=me.down_nr.value; qt_D[i]=me.down_qt.value
    # Norm weights
    if hasattr(lw,'attn_norm_w') and lw.attn_norm_w is not None:
        wAN[i]=lw.attn_norm_w.ctypes.data_as(cv)
    if hasattr(lw,'ffn_norm_w') and lw.ffn_norm_w is not None:
        wFN[i]=lw.ffn_norm_w.ctypes.data_as(cv)

me0 = e._moe_layers[0]
gate_qt_all = me0.gate_qt.value; up_qt_all = me0.up_qt.value; down_qt_all = me0.down_qt.value

# ── Build BC ──
class PageTable(ctypes.Structure):
    _fields_=[('page_size',ci),('n_pages',ci),('pages',cv),('table',cv),
              ('free_pages',cv),('free_count',ci),('_nkh',ci),('_hd',ci),('_n_layers',ci)]
class KVBlock(ctypes.Structure):
    _fields_=[('k',cv),('v',cv),('n_blocks',ci),('seq_len',ci*64),
              ('block_map',(ci*1024)*64),('pt',PageTable)]
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

MAX_CTX = 4096
bc = BC()
for a,v in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',e.eps),
            ('nc',N),('outNR',V),('outNC',N),('outQuant',8),
            ('n_experts',EXP),('n_experts_per_tok',ET),('moe_intermediate',EFF),
            ('gate_exp_quant',gate_qt_all),('up_exp_quant',up_qt_all),('down_exp_quant',down_qt_all),
            ('emb_quant',0),('max_ctx',MAX_CTX),
            ('rope_dim',HD),('full_attn_interval',4),('n_layers_actual',L)]:
    setattr(bc,a,v)

# Wire all attention paths
bc.wQKV = ctypes.cast(wa(wQKV), cv); bc.qkv_quant = ctypes.cast(ia(qkv_qt), cv)
bc.wQ = ctypes.cast(wa(wQ_sep), cv); bc.wK = ctypes.cast(wa(wK_sep), cv); bc.wV = ctypes.cast(wa(wV_sep), cv)
bc.nQ = ctypes.cast(ia(nQ_sep), cv); bc.nK = ctypes.cast(ia(nK_sep), cv); bc.nV = ctypes.cast(ia(nV_sep), cv)
bc.q_quant = ctypes.cast(ia(qQ_sep), cv); bc.k_quant = ctypes.cast(ia(qK_sep), cv); bc.v_quant = ctypes.cast(ia(qV_sep), cv)
bc.wAttnG = ctypes.cast(wa(wAG), cv); bc.attnG_quant = ctypes.cast(ia(ag_qt), cv)
bc.wO = ctypes.cast(wa(wO_attn), cv); bc.nO = ctypes.cast(ia(nO_attn), cv); bc.o_quant = ctypes.cast(ia(qO_attn), cv)
bc.wSsmOut = ctypes.cast(wa(wSO), cv); bc.ssm_out_quant = ctypes.cast(ia(so_qt), cv)
bc.ssm_conv1d = ctypes.cast(wa(ssm_c1d), cv); bc.ssm_a = ctypes.cast(wa(ssm_a), cv)
bc.ssm_dt_bias = ctypes.cast(wa(ssm_db), cv); bc.ssm_alpha = ctypes.cast(wa(ssm_al), cv)
bc.ssm_beta = ctypes.cast(wa(ssm_be), cv); bc.ssm_norm = ctypes.cast(wa(ssm_nm), cv)
bc.wSHexpG = ctypes.cast(wa(wSHG), cv); bc.wSHexpU = ctypes.cast(wa(wSHU), cv)
bc.wSHexpD = ctypes.cast(wa(wSHD), cv); bc.wShexpRouter = ctypes.cast(wa(wSHR), cv)
bc.shexp_g_quant = ctypes.cast(ia(shg_qt), cv); bc.shexp_u_quant = ctypes.cast(ia(shu_qt), cv)
bc.shexp_d_quant = ctypes.cast(ia(shd_qt), cv)
bc.wAN = ctypes.cast(wa(wAN), cv); bc.wFN = ctypes.cast(wa(wFN), cv)

# MoE
bc.w_gate_inp = ctypes.cast(wa(wGI), cv)
bc.w_gate_exps = ctypes.cast(wa(wGE), cv)
bc.w_up_exps = ctypes.cast(wa(wUE), cv)
bc.w_down_exps = ctypes.cast(wa(wDE), cv)
bc.nG = ctypes.cast(ia(ns_G), cv); bc.nU = ctypes.cast(ia(ns_U), cv); bc.nD = ctypes.cast(ia(ns_D), cv)
bc.g_quant = ctypes.cast(ia(qt_G), cv); bc.u_quant = ctypes.cast(ia(qt_U), cv); bc.d_quant = ctypes.cast(ia(qt_D), cv)

# NULL unused fields (Qwen3.6 doesn't use dense FFN weight arrays)
for pf in ['wG','wU','wD','nG','nU','nD',
           'g_quant','u_quant','d_quant',
           'wQK','qk_quant']:
    setattr(bc, pf, cv(0))

bc.emb = e.emb.ctypes.data_as(cv) if hasattr(e,'emb') and e.emb is not None else cv(0)
bc.onw = e._out_norm_w.ctypes.data_as(cv) if hasattr(e,'_out_norm_w') else cv(0)
bc.wOut = ctypes.cast(e._out_raw, cv)

# SSM state
ssm_state = np.zeros(L * e.ssm_groups * e.ssm_state_size, dtype=np.float32)
bc.ssm_state = ssm_state.ctypes.data_as(cv)

# Layer types
lt_arr = (ci * L)(*LT)
bc.layer_types = ctypes.cast(lt_arr, cv)

# RoPE
cos_t = np.zeros(MAX_CTX * HD // 2, dtype=np.float32)
sin_t = np.zeros(MAX_CTX * HD // 2, dtype=np.float32)
for p in range(MAX_CTX):
    for j in range(HD // 2):
        ang = p / (10000.0 ** (2.0 * j / HD))
        cos_t[p*HD//2+j]=np.cos(ang); sin_t[p*HD//2+j]=np.sin(ang)
bc.cos_table=cos_t.ctypes.data_as(cv); bc.sin_table=sin_t.ctypes.data_as(cv)

S_act = max(N, NH*HD, FF, NKH, NH*HD+NKH, 8192)
min_ws = 12 * 1 * S_act + 2 * MAX_CTX * NKH * HD + 100000
ws = np.zeros(min_ws, dtype=np.float32)
bc.workspace = ws.ctypes.data_as(cv); bc.ws_size = min_ws
print(f"  Workspace: {min_ws:,} floats", flush=True)

ce.kv_init.argtypes=[cv,ci,ci,ci]; ce.kv_init.restype=None
ce.batch_forward.argtypes=[cv,cv,ci,cv]; ce.batch_forward.restype=None
ce.rope_init.argtypes=[cv,cv,ci,ci,ci]; ce.rope_init.restype=None
ce.rope_init(bc.cos_table, bc.sin_table, ci(MAX_CTX), ci(HD), ci(HD))

ws_ptr = ws.ctypes.data_as(cv)
def bf(tok, B=1):
    ce.batch_forward(ctypes.byref(bc), (ci*B)(*([tok]*B)), ci(B), ws_ptr)

# ── Benchmark ──
print("\nB=1 test...", flush=True)
kv = KVBlock(); ce.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
bc.kv_array = ctypes.cast((cv*1)(ctypes.cast(ctypes.pointer(kv), cv)), cv)
lbuf = np.zeros(V, dtype=np.float32)
bc.logits = lbuf.ctypes.data_as(cv)

for _ in range(3): bf(1); kv.seq_len[0] = 0

times = []
for _ in range(30):
    t0 = time.perf_counter(); bf(1); t1 = time.perf_counter()
    times.append(t1 - t0); kv.seq_len[0] = 0

times = times[5:]
avg = np.mean(times); tps = 1.0/avg
mx = float(lbuf.max()); nc = int(np.isnan(lbuf).sum())
print(f"  C engine B=1: {avg*1000:.1f}ms → {tps:.1f} tok/s", flush=True)
print(f"  max={mx:.1f}, nan={nc}", flush=True)
print(f"  VS Python: 40.1ms → 24.9 tok/s", flush=True)
print(f"  Ratio: {tps/24.9:.2f}x", flush=True)
