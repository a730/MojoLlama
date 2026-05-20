#!/usr/bin/env python3
"""GPT-OSS-20B: 10-user batch inference with F32→Q8_0 conversion."""
import sys, os, time, ctypes, numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))

MODEL = "/tmp/models/gpt-oss-20b-Q4_K_M.gguf"
N_USERS = 10
PROMPT_LEN = 512
GEN_TOKENS = 20
B_MAX = N_USERS

# ── Load engine ──
print("Loading GPT-OSS engine...", flush=True)
from turbo_engine_v7_moe import TurboEngineV7MoE
e = TurboEngineV7MoE(MODEL, 32)

N=e.n_embd; NH=e.n_head; NKH=e.n_kv_head; HD=e.head_dim
FF=e.n_ff; L=e.n_layers; V=e.vocab_size
print(f"  {L}L/{N}D/{NH}H/{NKH}KV/{HD}hd MoE {e.n_experts}x{e.n_experts_per_tok}", flush=True)

# ── Load C library ──
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels', 'cengine_batch_instr.so'))
cv=ctypes.c_void_p; ci=ctypes.c_int; cf=ctypes.c_float

# ── Q8_0 quantize helper ──
def f32_to_q8_0(arr):
    """Convert F32 numpy array to Q8_0 raw bytes."""
    arr = arr.astype(np.float32)
    flat = arr.reshape(-1)
    n = len(flat)
    n_blocks = (n + 31) // 32
    out = bytearray()
    for b in range(n_blocks):
        block = flat[b*32:(b+1)*32]
        if len(block) < 32:
            block = np.pad(block, (0, 32 - len(block)))
        d = float(np.abs(block).max() / 127.0)
        if d == 0: d = 1.0
        quants = np.clip(np.round(block / d), -128, 127).astype(np.int8)
        out += struct.pack('<f', d)
        out += quants.tobytes()
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()

# ── Build BC struct with F32→Q8_0 converted attention weights ──
def wa(arr): return (cv*L)(*[ctypes.cast(a,cv) for a in arr])
def ia(arr): return (ci*L)(*[int(a) for a in arr])

import struct

wQ_raw_arr=[0]*L; wK_raw_arr=[0]*L; wV_raw_arr=[0]*L; wO_raw_arr=[0]*L
nQ_arr=[0]*L; nK_arr=[0]*L; nV_arr=[0]*L; nO_arr=[0]*L
qQ_arr=[0]*L; qK_arr=[0]*L; qV_arr=[0]*L; qO_arr=[0]*L
wG_arr=[0]*L; wU_arr=[0]*L; wD_arr=[0]*L
nG_arr=[0]*L; nU_arr=[0]*L; nD_arr=[0]*L
qG_arr=[0]*L; qU_arr=[0]*L; qD_arr=[0]*L
wAN_arr=[0]*L; wFN_arr=[0]*L

# Keep Q8_0 buffers alive (prevent GC)
_q8_buffers = []
print("  Converting F32 weights → Q8_0 for C engine...", flush=True)
for i, lw in enumerate(e._layers):
    # attn_q: F32 → Q8_0
    if hasattr(lw, 'attn_q_f32') and lw.attn_q_f32 is not None:
        q8 = f32_to_q8_0(lw.attn_q_f32)
        _q8_buffers.append(q8)
        wQ_raw_arr[i] = q8.ctypes.data_as(cv)
        nQ_arr[i] = lw.attn_q_f32.shape[0]  # output rows = NH*HD
        qQ_arr[i] = 8  # Q8_0
    elif hasattr(lw, 'attn_q_raw') and lw.attn_q_raw is not None:
        wQ_raw_arr[i] = lw.attn_q_raw
        nQ_arr[i] = lw.attn_q_nr.value
        qQ_arr[i] = lw.attn_q_qt.value
    
    if hasattr(lw, 'attn_k_f32') and lw.attn_k_f32 is not None:
        q8 = f32_to_q8_0(lw.attn_k_f32)
        _q8_buffers.append(q8)
        wK_raw_arr[i] = q8.ctypes.data_as(cv)
        nK_arr[i] = lw.attn_k_f32.shape[0]  # output rows = NKH*HD
        qK_arr[i] = 8
    elif hasattr(lw, 'attn_k_raw') and lw.attn_k_raw is not None:
        wK_raw_arr[i] = lw.attn_k_raw
        nK_arr[i] = lw.attn_k_nr.value
        qK_arr[i] = lw.attn_k_qt.value
    
    # attn_v: already Q8_0 raw
    if hasattr(lw, 'attn_v_raw') and lw.attn_v_raw is not None:
        wV_raw_arr[i] = lw.attn_v_raw
        nV_arr[i] = lw.attn_v_nr.value
        qV_arr[i] = lw.attn_v_qt.value
    
    # attn_out: already Q4_K raw
    if hasattr(lw, 'attn_out_raw') and lw.attn_out_raw is not None:
        wO_raw_arr[i] = lw.attn_out_raw
        nO_arr[i] = lw.attn_out_nr.value
        qO_arr[i] = lw.attn_out_qt.value
    
    wAN_arr[i] = lw.attn_norm_w.ctypes.data_as(cv)
    wFN_arr[i] = lw.ffn_norm_w.ctypes.data_as(cv)

# MoE expert weights
for i, me in enumerate(e._moe_layers):
    wG_arr[i] = me.gate_raw[0].ctypes.data_as(cv) if me.gate_raw else cv(0)
    wU_arr[i] = me.up_raw[0].ctypes.data_as(cv) if me.up_raw else cv(0)
    wD_arr[i] = me.down_raw[0].ctypes.data_as(cv) if me.down_raw else cv(0)
    nG_arr[i] = me.gate_nr.value; nU_arr[i] = me.up_nr.value; nD_arr[i] = me.down_nr.value
    qG_arr[i] = me.gate_qt.value; qU_arr[i] = me.up_qt.value; qD_arr[i] = me.down_qt.value

# Router + expert weight arrays
w_gate_inp_arr=[0]*L; w_gate_exps_arr=[0]*L; w_up_exps_arr=[0]*L; w_down_exps_arr=[0]*L
for i,lw in enumerate(e._layers):
    me = e._moe_layers[i]
    w_gate_inp_arr[i] = (me.router_raw.ctypes.data_as(cv) if me.router_raw is not None
                        else me.router_f32.ctypes.data_as(cv) if me.router_f32 is not None
                        else cv(0))
    gate_raw_t = me.gate_raw
    w_gate_exps_arr[i] = gate_raw_t[0].ctypes.data_as(cv) if gate_raw_t and len(gate_raw_t)>0 else cv(0)
    w_up_exps_arr[i] = me.up_raw[0].ctypes.data_as(cv) if me.up_raw and len(me.up_raw)>0 else cv(0)
    w_down_exps_arr[i] = me.down_raw[0].ctypes.data_as(cv) if me.down_raw and len(me.down_raw)>0 else cv(0)

gate_qt = e._moe_layers[0].gate_qt.value if len(e._moe_layers)>0 else 2
up_qt = e._moe_layers[0].up_qt.value if len(e._moe_layers)>0 else 2
down_qt = e._moe_layers[0].down_qt.value if len(e._moe_layers)>0 else 2
print(f"  MoE quant: gate={gate_qt} up={up_qt} down={down_qt}", flush=True)

# ── BC struct ──
class PageTable(ctypes.Structure):
    _fields_ = [('page_size',ci),('n_pages',ci),('pages',cv),('table',cv),
                ('free_pages',cv),('free_count',ci),('_nkh',ci),('_hd',ci),('_n_layers',ci)]

class KVBlock(ctypes.Structure):
    _fields_ = [('k',cv),('v',cv),('n_blocks',ci),
                ('seq_len',ci*64),('block_map',(ci*1024)*64),('pt',PageTable)]

class BC(ctypes.Structure):
    _fields_ = [(n,t) for n,t in [
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
        ("wQK",cv),("qk_quant",cv),
        ("cos_table",cv),("sin_table",cv),("max_ctx",ci),
        ("workspace",cv),("ws_size",ci),
    ]]

lib.kv_init.argtypes=[cv,ci,ci,ci]; lib.kv_init.restype=None
lib.batch_forward.argtypes=[cv,cv,ci,cv]; lib.batch_forward.restype=None

bc=BC()
for attr,val in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',e.eps),
                 ('nc',N),('outNR',V),('outNC',N),('outQuant',8),
                 ('n_experts',e.n_experts),('n_experts_per_tok',e.n_experts_per_tok),
                 ('moe_intermediate',e.n_ff_expert),
                 ('gate_exp_quant',gate_qt),('up_exp_quant',up_qt),('down_exp_quant',down_qt),
                 ('emb_quant',0)]:
    setattr(bc,attr,val)
bc.wQ=ctypes.cast(wa(wQ_raw_arr),cv); bc.wK=ctypes.cast(wa(wK_raw_arr),cv)
bc.wV=ctypes.cast(wa(wV_raw_arr),cv); bc.wO=ctypes.cast(wa(wO_raw_arr),cv)
bc.wG=ctypes.cast(wa(wG_arr),cv); bc.wU=ctypes.cast(wa(wU_arr),cv); bc.wD=ctypes.cast(wa(wD_arr),cv)
bc.wAN=ctypes.cast(wa(wAN_arr),cv); bc.wFN=ctypes.cast(wa(wFN_arr),cv)
bc.wQK=cv(0); bc.qk_quant=cv(0)
bc.nQ=ctypes.cast(ia(nQ_arr),cv); bc.nK=ctypes.cast(ia(nK_arr),cv)
bc.nV=ctypes.cast(ia(nV_arr),cv); bc.nO=ctypes.cast(ia(nO_arr),cv)
bc.nG=ctypes.cast(ia(nG_arr),cv); bc.nU=ctypes.cast(ia(nU_arr),cv); bc.nD=ctypes.cast(ia(nD_arr),cv)
bc.q_quant=ctypes.cast(ia(qQ_arr),cv); bc.k_quant=ctypes.cast(ia(qK_arr),cv)
bc.v_quant=ctypes.cast(ia(qV_arr),cv); bc.o_quant=ctypes.cast(ia(qO_arr),cv)
bc.g_quant=ctypes.cast(ia(qG_arr),cv); bc.u_quant=ctypes.cast(ia(qU_arr),cv); bc.d_quant=ctypes.cast(ia(qD_arr),cv)
bc.w_gate_inp=ctypes.cast(wa(w_gate_inp_arr),cv)
bc.w_gate_exps=ctypes.cast(wa(w_gate_exps_arr),cv)
bc.w_up_exps=ctypes.cast(wa(w_up_exps_arr),cv)
bc.w_down_exps=ctypes.cast(wa(w_down_exps_arr),cv)
bc.emb=e.emb.ctypes.data_as(cv) if hasattr(e,'emb') else cv(0)
bc.onw=e._out_norm_w.ctypes.data_as(cv) if hasattr(e,'_out_norm_w') else cv(0)
bc.wOut=ctypes.cast(e._out_raw,cv) if hasattr(e,'_out_raw') else cv(0)

# RoPE table (need HD/2 since GPT-OSS has head_dim=64)
MAX_CTX=4096
cos_t=np.zeros(MAX_CTX*HD//2,dtype=np.float32)
sin_t=np.zeros(MAX_CTX*HD//2,dtype=np.float32)
for p in range(MAX_CTX):
    for j in range(HD//2):
        ang=p/(10000.0**(2.0*j/HD))
        cos_t[p*HD//2+j]=np.cos(ang)
        sin_t[p*HD//2+j]=np.sin(ang)
bc.cos_table=cos_t.ctypes.data_as(cv)
bc.sin_table=sin_t.ctypes.data_as(cv)
bc.max_ctx=MAX_CTX

# Workspace for large prefill
ws=np.zeros(B_MAX*MAX_CTX*N*4,dtype=np.float32)
bc.workspace=ws.ctypes.data_as(cv)
bc.ws_size=len(ws)

# Logits buffer
logits_buf=np.zeros(B_MAX*V,dtype=np.float32)
bc.logits=logits_buf.ctypes.data_as(cv)

# ── KV caches: one per user ──
print(f"  Creating {N_USERS} KV caches...", flush=True)
kv_blocks=[]
for _ in range(N_USERS):
    kv=KVBlock()
    lib.kv_init(ctypes.byref(kv),ci(L),ci(NKH),ci(HD))
    kv_blocks.append(kv)

# ── Benchmark: prefill + decode ──
print(f"\n{'='*60}", flush=True)
print(f"  Benchmark: {N_USERS} users, {PROMPT_LEN}-tok prompt, {GEN_TOKENS} gen", flush=True)
print(f"{'='*60}", flush=True)

# Phase 1: Prefill using batch_forward (token-by-token but batched B=N_USERS)
print(f"  Phase 1: Prefill {PROMPT_LEN} tokens for {N_USERS} users (batched)...", flush=True)
t0=time.perf_counter()

# Prefill: feed each position, all 10 users in one batch_forward call
import math
for pos in range(PROMPT_LEN):
    batch_tokens=(ci*N_USERS)(*[100+(pos % 50000) for _ in range(N_USERS)])
    kv_ptrs=(cv*N_USERS)(*[ctypes.cast(ctypes.pointer(k), cv) for k in kv_blocks])
    bc.kv_array=ctypes.cast(kv_ptrs,cv)
    lib.batch_forward(ctypes.byref(bc),batch_tokens,ci(N_USERS),cv(0))
    if pos%100==0: print(f"    prefill pos {pos}/{PROMPT_LEN}", end='\r', flush=True)

prefill_time=time.perf_counter()-t0
print(f"  Prefill: {prefill_time:.1f}s", flush=True)

# Phase 2: Generate tokens (batched B=N_USERS)
print(f"  Phase 2: Generate {GEN_TOKENS} tokens for {N_USERS} users (batched)...", flush=True)
gen_start=time.perf_counter()
total_gen=0

for step in range(GEN_TOKENS):
    # Each user's current token is derived from the logits
    # For first step, use BOS; for subsequent, use argmax from last step
    batch_tok_vals=[]
    for u in range(N_USERS):
        if step==0:
            batch_tok_vals.append(1)  # BOS for decode
        else:
            logits_slice=logits_buf[u*V:(u+1)*V]
            batch_tok_vals.append(int(np.argmax(logits_slice)))
    
    batch_tokens=(ci*N_USERS)(*batch_tok_vals)
    kv_ptrs=(cv*N_USERS)(*[ctypes.cast(ctypes.pointer(k), cv) for k in kv_blocks])
    bc.kv_array=ctypes.cast(kv_ptrs,cv)
    lib.batch_forward(ctypes.byref(bc),batch_tokens,ci(N_USERS),cv(0))
    total_gen+=N_USERS

gen_time=time.perf_counter()-gen_start
total_time=prefill_time+gen_time
tps=total_gen/total_time

print(f"  Generation: {total_gen} tok in {gen_time*1000:.0f}ms → {total_gen/gen_time:.1f} tok/s", flush=True)
print(f"  Total:      {total_time*1000:.0f}ms wall", flush=True)
print(f"  Throughput: {tps:.1f} tok/s ({tps/N_USERS:.1f} per user)", flush=True)
print(f"", flush=True)
print(f"{'='*60}", flush=True)
print(f"  VS llama.cpp (same workload): 22s wall, 9.1 tok/s", flush=True)
if tps > 9.1:
    print(f"  ✅ MojoLlama is {tps/9.1:.1f}x faster than llama.cpp!", flush=True)
else:
    print(f"  ⚠️  MojoLlama at {tps:.1f} tok/s vs llama.cpp 9.1 tok/s", flush=True)
print(f"{'='*60}", flush=True)
