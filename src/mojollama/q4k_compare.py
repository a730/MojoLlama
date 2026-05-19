#!/usr/bin/env python3
"""Compare C engine vs numpy reference per-layer for ONE layer.
Only dequantizes ONE layer's weights for speed."""
import sys, os, time, ctypes, numpy as np
os.environ['OMP_NUM_THREADS'] = '32'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE

e = TurboEngineV7MoE('/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf', 32)
lib = ctypes.CDLL('/onedev-workspace/work/src/mojollama/kernels/cengine_batch_instr.so')
cv, ci = ctypes.c_void_p, ctypes.c_int
L, N, NH, NKH, HD, FF, V = e.n_layers, e.n_embd, e.n_head, e.n_kv_head, e.head_dim, e.n_ff, e.vocab_size
S = max(N, NH*HD, FF, NKH*HD, e.n_ff_expert)

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
        ('cos_table',cv),('sin_table',cv),('max_ctx',ci),
        ('workspace',cv),('ws_size',ci)]
class KVBlock(ctypes.Structure):
    _fields_=[('k',cv),('v',cv),('n_blocks',ci),('seq_len',ci*64),('block_map',(ci*1024)*64)]
lib.batch_forward.argtypes=[cv,cv,ci,cv];lib.batch_forward.restype=None
lib.kv_init.argtypes=[cv,ci,ci,ci];lib.kv_init.restype=None
lib.q4_k_batch_matmul.argtypes=[cv,cv,cv,ci,ci,ci];lib.q4_k_batch_matmul.restype=None

# Build BC
def wa(a): return (cv*L)(*[ctypes.cast(t, cv) for t in a])
def ia(a): return (ci*L)(*[int(t) for t in a])

bc = BC()
for a, v in [('L', L), ('N', N), ('NH', NH), ('NKH', NKH), ('HD', HD), ('FF', FF), ('V', V),
             ('eps', e.eps), ('nc', N), ('outNR', V), ('outNC', N), ('outQuant', 8),
             ('n_experts', e.n_experts), ('n_experts_per_tok', e.n_experts_per_tok),
             ('moe_intermediate', e.n_ff_expert), ('gate_exp_quant', 12), ('up_exp_quant', 12),
             ('down_exp_quant', 14), ('emb_quant', 0)]:
    setattr(bc, a, v)

wQ,wK,wV,wO,wG,wU,wD,wAN,wFN=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
nQ,nK,nV,nO,nG,nU,nD=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
qQ,qK,qV,qO,qG,qU,qD=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
for i,lw in enumerate(e._layers):
    wQ[i]=lw.attn_q_raw;nQ[i]=lw.attn_q_nr.value;qQ[i]=lw.attn_q_qt.value
    wK[i]=lw.attn_k_raw;nK[i]=lw.attn_k_nr.value;qK[i]=lw.attn_k_qt.value
    wV[i]=lw.attn_v_raw;nV[i]=lw.attn_v_nr.value;qV[i]=lw.attn_v_qt.value
    wO[i]=lw.attn_out_raw;nO[i]=lw.attn_out_nr.value;qO[i]=lw.attn_out_qt.value
    wAN[i]=lw.attn_norm_w.ctypes.data_as(cv);wFN[i]=lw.ffn_norm_w.ctypes.data_as(cv)
for i,me2 in enumerate(e._moe_layers):
    wG[i]=me2.gate_raw[0].ctypes.data_as(cv);nG[i]=me2.gate_nr.value;qG[i]=12  # Q4_K
    wU[i]=me2.up_raw[0].ctypes.data_as(cv);nU[i]=me2.up_nr.value;qU[i]=12     # Q4_K
    wD[i]=me2.down_raw[0].ctypes.data_as(cv);nD[i]=me2.down_nr.value;qD[i]=14 # Q6_K

w_gi,w_ge,w_ue,w_de=[0]*L,[0]*L,[0]*L,[0]*L
for i in range(L):
    me2=e._moe_layers[i]
    w_gi[i]=me2.router_f32.ctypes.data_as(cv)if me2.router_f32 is not None else cv(0)
    if me2.gate_raw and len(me2.gate_raw):w_ge[i]=me2.gate_raw[0].ctypes.data_as(cv)
    if me2.up_raw and len(me2.up_raw):w_ue[i]=me2.up_raw[0].ctypes.data_as(cv)
    if me2.down_raw and len(me2.down_raw):w_de[i]=me2.down_raw[0].ctypes.data_as(cv)

bc.wQ=ctypes.cast(wa(wQ),cv);bc.wK=ctypes.cast(wa(wK),cv);bc.wV=ctypes.cast(wa(wV),cv);bc.wO=ctypes.cast(wa(wO),cv)
bc.wG=ctypes.cast(wa(wG),cv);bc.wU=ctypes.cast(wa(wU),cv);bc.wD=ctypes.cast(wa(wD),cv)
bc.wAN=ctypes.cast(wa(wAN),cv);bc.wFN=ctypes.cast(wa(wFN),cv)
bc.nQ=ctypes.cast(ia(nQ),cv);bc.nK=ctypes.cast(ia(nK),cv);bc.nV=ctypes.cast(ia(nV),cv);bc.nO=ctypes.cast(ia(nO),cv)
bc.nG=ctypes.cast(ia(nG),cv);bc.nU=ctypes.cast(ia(nU),cv);bc.nD=ctypes.cast(ia(nD),cv)
bc.q_quant=ctypes.cast(ia(qQ),cv);bc.k_quant=ctypes.cast(ia(qK),cv);bc.v_quant=ctypes.cast(ia(qV),cv);bc.o_quant=ctypes.cast(ia(qO),cv)
bc.g_quant=ctypes.cast(ia(qG),cv);bc.u_quant=ctypes.cast(ia(qU),cv);bc.d_quant=ctypes.cast(ia(qD),cv)
bc.w_gate_inp=ctypes.cast(wa(w_gi),cv);bc.w_gate_exps=ctypes.cast(wa(w_ge),cv)
bc.w_up_exps=ctypes.cast(wa(w_ue),cv);bc.w_down_exps=ctypes.cast(wa(w_de),cv)
bc.emb=e.emb.ctypes.data_as(cv);bc.onw=e._out_norm_w.ctypes.data_as(cv);bc.wOut=ctypes.cast(e._out_raw,cv)
mc=4096;hd2=HD//2
freq_v=e.rope_freq_base**(np.arange(0,HD,2,dtype=np.float32)/HD)
ang=np.arange(mc,dtype=np.float32).reshape(-1,1)/freq_v.reshape(1,-1)
ct=np.cos(ang).astype(np.float32).reshape(-1);st=np.sin(ang).astype(np.float32).reshape(-1)
bc.cos_table=ct.ctypes.data_as(cv);bc.sin_table=st.ctypes.data_as(cv);bc.max_ctx=mc

ws=np.zeros(12*S,dtype=np.float32);lb=np.zeros(V,dtype=np.float32)
kv=KVBlock();lib.kv_init(ctypes.byref(kv),ci(L),ci(NKH),ci(HD))
kva=(cv*1)(ctypes.cast(ctypes.pointer(kv),cv))
bc.kv_array=ctypes.cast(kva,cv);bc.logits=lb.ctypes.data_as(cv)

# Run 1 layer at a time and capture intermediate outputs
print('Running C engine layer by layer, capturing outputs...', flush=True)
for target_layer in [1, 4]:  # check layers 1 and 4
    # Reset
    ws2 = np.zeros(12*S, dtype=np.float32)
    lb2 = np.zeros(V, dtype=np.float32)
    kv2 = KVBlock(); lib.kv_init(ctypes.byref(kv2), ci(L), ci(NKH), ci(HD))
    bc2 = BC(); 
    for a in BC._fields_: setattr(bc2, a[0], getattr(bc, a[0]))
    bc2.kv_array = ctypes.cast((cv*1)(ctypes.cast(ctypes.pointer(kv2), cv)), cv)
    bc2.logits = lb2.ctypes.data_as(cv)
    bc2.L = target_layer
    
    lib.batch_forward(ctypes.byref(bc2), (ci*1)(785), ci(1), ws2.ctypes.data_as(cv))
    
    # The workspace after the last layer has the hidden state in ws2[:N]
    h = ws2[:N].copy()
    nan_c = np.isnan(h).sum()
    max_v = np.max(np.abs(h))
    print(f'  L={target_layer}: x NaN={nan_c}/{N} max_abs={max_v:.4f}', flush=True)

# Now extract the intermediate xn (RMS-normed hidden state) for layer 0
# and test with Q4_K matmul + compare to dequantized FP32 reference
print('\n--- Q4_K numerical verification ---', flush=True)
import gguf

# For layer 0: dequantize gate weights to FP32 using gguf reference
for t in e.reader.tensors:
    if t.name == 'blk.0.ffn_gate_exps.weight':
        f32_ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
        sh = t.shape  # [2048, 768, 128]
        f32_ref = f32_ref.reshape(int(sh[2]), int(sh[1]), int(sh[0]))
        # Expert 50 (router-selected)
        expert_50_ref = np.ascontiguousarray(f32_ref[50])  # [768, 2048]
        expert_106_ref = np.ascontiguousarray(f32_ref[106])
        break

# Run layer 0 with workspace and extract xn
ws3 = np.zeros(12*S, dtype=np.float32)
lb3 = np.zeros(V, dtype=np.float32)
kv3 = KVBlock(); lib.kv_init(ctypes.byref(kv3), ci(1), ci(NKH), ci(HD))
bc3 = BC()
for a in BC._fields_: setattr(bc3, a[0], getattr(bc, a[0]))
bc3.L = 1
bc3.kv_array = ctypes.cast((cv*1)(ctypes.cast(ctypes.pointer(kv3), cv)), cv)
bc3.logits = lb3.ctypes.data_as(cv)

lib.batch_forward(ctypes.byref(bc3), (ci*1)(785), ci(1), ws3.ctypes.data_as(cv))

# Extract xn from workspace (after RMS norm and QKV, but before MoE FFN)
xn_ref = ws3[:N].copy()
print(f'xn from ws: NaN={np.isnan(xn_ref).sum()}/{N} range=[{xn_ref.min():.4f}, {xn_ref.max():.4f}]', flush=True)

# Compare C Q4_K matmul output vs numpy FP32 reference
M = e._moe_layers[0].gate_nr.value
base_addr = e._moe_layers[0].gate_raw[0].ctypes.data
stride = M * (N // 256) * 144

for e_idx, name in [(50, 'Expert 50'), (106, 'Expert 106')]:
    ptr = ctypes.cast(base_addr + e_idx * stride, cv)
    out_c = np.zeros(M, dtype=np.float32)
    lib.q4_k_batch_matmul(ptr, xn_ref.ctypes.data_as(cv), out_c.ctypes.data_as(cv), ci(M), ci(N), ci(1))
    
    # Numpy FP32 reference
    w_ref = f32_ref[0] if name == 'Expert 50' else f32_ref[106] if name == 'Expert 106' else None
    if name == 'Expert 50':
        w_ref = expert_50_ref
    else:
        w_ref = expert_106_ref
    out_np = w_ref @ xn_ref  # [768] = [768, 2048] @ [2048]
    
    diff = np.abs(out_c - out_np)
    mae = diff.mean()
    max_err = diff.max()
    std_err = diff.std()
    rel_err = diff / (np.abs(out_np) + 1e-10)
    mre = rel_err.mean()
    
    print(f'  {name}: MAE={mae:.6f} MAX={max_err:.6f} MRE={mre:.4f}', flush=True)
    print(f'    C output:   [{out_c[:5].tolist()}]', flush=True)
    print(f'    NP output:  [{out_np[:5].tolist()}]', flush=True)
    print(f'    Differences: [{diff[:5].tolist()}]', flush=True)
    print(f'    C NaN={np.isnan(out_c).sum()} NP NaN={np.isnan(out_np).sum()}', flush=True)
