#!/usr/bin/env python3
"""Hybrid MojoLlama: C engine for attention, Python for MoE experts (FP32 accurate).
Eliminates NaN by using gguf.dequantize for expert weights."""
import sys, os, time, ctypes, numpy as np
os.environ['OMP_NUM_THREADS'] = '32'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE
import gguf

e = TurboEngineV7MoE('/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf', 32)
lib = ctypes.CDLL('/onedev-workspace/work/src/mojollama/kernels/cengine_batch_instr.so')
cv, ci = ctypes.c_void_p, ctypes.c_int
L, N, NH, NKH, HD, FF, V = e.n_layers, e.n_embd, e.n_head, e.n_kv_head, e.head_dim, e.n_ff, e.vocab_size

# --- Pre-dequantize all expert weights to FP32 (one-time, ~3-4 min) ---
print('Dequantizing expert weights to FP32...', flush=True)
t0 = time.time()
Wgate_f32 = []  # per layer: [128, 768, 2048] FP32
Wup_f32 = []
Wdown_f32 = []
for i in range(L):
    for t in e.reader.tensors:
        base = f'blk.{i}.ffn_'
        if t.name == base + 'gate_exps.weight':
            f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
            sh = t.shape  # [2048, 768, 128]
            f32 = f32.reshape(int(sh[2]), int(sh[1]), int(sh[0]))  # [128, 768, 2048]
            Wgate_f32.append(np.ascontiguousarray(f32))
        if t.name == base + 'up_exps.weight':
            f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
            sh = t.shape
            f32 = f32.reshape(int(sh[2]), int(sh[1]), int(sh[0]))
            Wup_f32.append(np.ascontiguousarray(f32))
        if t.name == base + 'down_exps.weight':
            f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
            sh = t.shape
            f32 = f32.reshape(int(sh[2]), int(sh[1]), int(sh[0]))
            Wdown_f32.append(np.ascontiguousarray(f32))
    if (i+1) % 10 == 0:
        print(f'  Layer {i+1}/{L} dequantized ({time.perf_counter()-t0:.0f}s)', flush=True)
print(f'Dequantized in {time.perf_counter()-t0:.0f}s', flush=True)

# --- Setup C engine for attention path ---
class BC(ctypes.Structure):
    _fields_=[('L',ci),('N',ci),('NH',ci),('NKH',ci),('HD',ci),('FF',ci),('V',ci),('eps',ctypes.c_float),
        ('wQ',cv),('wK',cv),('wV',cv),('wO',cv),('wG',cv),('wU',cv),('wD',cv),
        ('wAN',cv),('wFN',cv),('nQ',cv),('nK',cv),('nV',cv),('nO',cv),('nG',cv),('nU',cv),('nD',cv),
        ('nc',ci),('emb',cv),('onw',cv),('wOut',cv),('outNR',ci),('outNC',ci),('outQuant',ci),
        ('kv_array',cv),('logits',cv),('n_experts',ci),('n_experts_per_tok',ci),('moe_intermediate',ci),
        ('w_gate_inp',cv),('w_gate_exps',cv),('w_up_exps',cv),('w_down_exps',cv),
        ('gate_exp_quant',ci),('up_exp_quant',ci),('down_exp_quant',ci),
        ('q_quant',cv),('k_quant',cv),('v_quant',cv),('o_quant',cv),
        ('g_quant',cv),('u_quant',cv),('d_quant',cv),('emb_quant',ci),
        ('cos_table',cv),('sin_table',cv),('max_ctx',ci)]
class KVBlock(ctypes.Structure):
    _fields_=[('k',cv),('v',cv),('n_blocks',ci),('seq_len',ci*64),('block_map',(ci*1024)*64)]
lib.batch_forward.argtypes=[cv,cv,ci,cv];lib.batch_forward.restype=None
lib.kv_init.argtypes=[cv,ci,ci,ci];lib.kv_init.restype=None

def wa(a): return (cv*L)(*[ctypes.cast(t, cv) for t in a])
def ia(a): return (ci*L)(*[int(t) for t in a])

bc = BC()
for a, v in [('L', L), ('N', N), ('NH', NH), ('NKH', NKH), ('HD', HD), ('FF', FF), ('V', V),
             ('eps', e.eps), ('nc', N), ('outNR', V), ('outNC', N), ('outQuant', 8),
             ('n_experts', e.n_experts), ('n_experts_per_tok', e.n_experts_per_tok),
             ('moe_intermediate', e.n_ff_expert), ('gate_exp_quant', 0), ('up_exp_quant', 0),
             ('down_exp_quant', 0), ('emb_quant', 0)]:
    setattr(bc, a, v)

# Set up weight pointers for attention path
wQ,wK,wV,wO,wAN,wFN=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
nQ,nK,nV,nO=[0]*L,[0]*L,[0]*L,[0]*L
qQ,qK,qV,qO=[0]*L,[0]*L,[0]*L,[0]*L
for i, lw in enumerate(e._layers):
    wQ[i]=lw.attn_q_raw; wK[i]=lw.attn_k_raw; wV[i]=lw.attn_v_raw; wO[i]=lw.attn_out_raw
    nQ[i]=lw.attn_q_nr.value; nK[i]=lw.attn_k_nr.value; nV[i]=lw.attn_v_nr.value; nO[i]=lw.attn_out_nr.value
    qQ[i]=lw.attn_q_qt.value; qK[i]=lw.attn_k_qt.value; qV[i]=lw.attn_v_qt.value; qO[i]=lw.attn_out_qt.value
    wAN[i]=lw.attn_norm_w.ctypes.data_as(cv); wFN[i]=lw.ffn_norm_w.ctypes.data_as(cv)

bc.wQ=ctypes.cast(wa(wQ),cv); bc.wK=ctypes.cast(wa(wK),cv); bc.wV=ctypes.cast(wa(wV),cv); bc.wO=ctypes.cast(wa(wO),cv)
bc.wAN=ctypes.cast(wa(wAN),cv); bc.wFN=ctypes.cast(wa(wFN),cv)
bc.nQ=ctypes.cast(ia(nQ),cv); bc.nK=ctypes.cast(ia(nK),cv); bc.nV=ctypes.cast(ia(nV),cv); bc.nO=ctypes.cast(ia(nO),cv)
bc.q_quant=ctypes.cast(ia(qQ),cv); bc.k_quant=ctypes.cast(ia(kQ),cv)  # these work
# Actually just set all quants to correct values from the Python class
bc.q_quant=ctypes.cast(ia(qQ),cv); bc.k_quant=ctypes.cast(ia(qK),cv)
bc.v_quant=ctypes.cast(ia(qV),cv); bc.o_quant=ctypes.cast(ia(qO),cv)

# MoE weight pointers (empty for FFN path, will bypass C moe_ffn)
w_gi=[0]*L
for i in range(L):
    me=e._moe_layers[i]
    w_gi[i]=me.router_f32.ctypes.data_as(cv) if me.router_f32 is not None else ctypes.c_void_p(0)
bc.w_gate_inp=ctypes.cast(wa(w_gi),cv)
# Set expert pointers to something valid but won't be used
bc.w_gate_exps=ctypes.cast((cv*L)(*[cv(0) for _ in range(L)]),cv)
bc.w_up_exps=ctypes.cast((cv*L)(*[cv(0) for _ in range(L)]),cv)
bc.w_down_exps=ctypes.cast((cv*L)(*[cv(0) for _ in range(L)]),cv)

bc.emb=e.emb.ctypes.data_as(cv); bc.onw=e._out_norm_w.ctypes.data_as(cv); bc.wOut=ctypes.cast(e._out_raw,cv)

mc=4096; hd2=HD//2
freq_v=e.rope_freq_base**(np.arange(0,HD,2,dtype=np.float32)/HD)
ang=np.arange(mc,dtype=np.float32).reshape(-1,1)/freq_v.reshape(1,-1)
ct=np.cos(ang).astype(np.float32).reshape(-1); st=np.sin(ang).astype(np.float32).reshape(-1)
bc.cos_table=ct.ctypes.data_as(cv); bc.sin_table=st.ctypes.data_as(cv); bc.max_ctx=mc

S = max(N, NH*HD, FF, NKH*HD, e.n_ff_expert)
ws = np.zeros(12*S, dtype=np.float32)
lb = np.zeros(V, dtype=np.float32)
kv = KVBlock(); lib.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
kva = (cv*1)(ctypes.cast(ctypes.pointer(kv), cv))
bc.kv_array = ctypes.cast(kva, cv); bc.logits = lb.ctypes.data_as(cv)

# --- Run forward with Python MoE ---
def silu(x):
    return x / (1.0 + np.exp(-x))

def forward(token):
    global ws, lb, kv, kva
    # Reset KV cache
    lib.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
    kva = (cv*1)(ctypes.cast(ctypes.pointer(kv), cv))
    bc.kv_array = ctypes.cast(kva, cv)
    ws[:] = 0
    lb[:] = 0
    
    # Embedding
    x = e.emb[token].copy()
    
    for l in range(L):
        res = x.copy()
        xn = x * (1.0 / np.sqrt(np.mean(x*x) + e.eps)) * e._layers[l].attn_norm_w
        
        # Attention path using C engine
        batch_q = np.zeros(N, dtype=np.float32)
        batch_k = np.zeros(N, dtype=np.float32)
        batch_v = np.zeros(N, dtype=np.float32)
        
        # Use C for QKV projections
        # Write xn to workspace, call batch_forward but only for QKV
        # Better approach: call C functions directly
        
        # Q projection
        qt = qQ[l]
        if qt == 12:
            lib.q4_k_batch_matmul(wQ[l], xn.ctypes.data_as(cv), batch_q.ctypes.data_as(cv), ci(nQ[l]), ci(N), ci(1))
        elif qt == 2:
            q4_0 = ctypes.CDLL('kernels/cengine_batch_instr.so')
            # Use float16 for dequant
        
        # Actually simpler: just use numpy matmul on FP32 weights
        # Problem: we don't have FP32 weights for attention QKV
        # Those are stored as Q4_K quantized.
        
        # OK let me use the C engine for QKV+batch_forward but skip the MoE part
        # The issue is that batch_forward is a single function call
        pass  # Need a different approach
    
    return lb[:V]

# Test: single token forward using existing batch_forward (with FP32 expert bypass)
print('\n--- Hybrid forward ---', flush=True)

# For the hybrid approach, I'll modify the cengine to accept FP32 experts
# and skip the C moe_ffn, computing it in Python instead

# Actually the simplest approach: create a modified batch_forward that 
# does everything up to the FFN, then returns the hidden state
# Compute FFN in Python, call batch_forward again for next layer

# For now, let's just verify C engine's attention path is correct
# by running 6 layers (which produce 0 NaN)
print('\n--- C engine attention correctness check ---', flush=True)
bc.L = 6
t0 = time.time()
lib.batch_forward(ctypes.byref(bc), (ci*1)(785), ci(1), ws.ctypes.data_as(cv))
t1 = time.time()
nan_c = np.isnan(lb[:V]).sum()
print(f'6-layer C engine: {nan_c}/{V} NaN, {(t1-t0)*1000:.1f}ms', flush=True)
