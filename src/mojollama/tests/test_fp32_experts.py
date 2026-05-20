#!/usr/bin/env python3
"""Compare C engine output vs numpy for layer 0 only. Fast."""
import sys, os, time, ctypes, numpy as np
os.environ['OMP_NUM_THREADS'] = '32'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE

e = TurboEngineV7MoE('/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf', 32)
lib = ctypes.CDLL('/onedev-workspace/work/src/mojollama/kernels/cengine_batch_instr.so')
cv,ci = ctypes.c_void_p, ctypes.c_int

L,N,NH,NKH,HD,FF,V = e.n_layers,e.n_embd,e.n_head,e.n_kv_head,e.head_dim,e.n_ff,e.vocab_size
S = max(N, NH*HD, FF, NKH*HD, e.n_ff_expert)
lw = e._layers[0]; me = e._moe_layers[0]

# Build BC struct with FP32 expert weights
class BC(ctypes.Structure):
    _fields_ = [('L',ci),('N',ci),('NH',ci),('NKH',ci),('HD',ci),('FF',ci),('V',ci),('eps',ctypes.c_float),
                ('wQ',cv),('wK',cv),('wV',cv),('wO',cv),('wG',cv),('wU',cv),('wD',cv),
                ('wAN',cv),('wFN',cv),('nQ',cv),('nK',cv),('nV',cv),('nO',cv),('nG',cv),('nU',cv),('nD',cv),
                ('nc',ci),('emb',cv),('onw',cv),('wOut',cv),('outNR',ci),('outNC',ci),('outQuant',ci),
                ('kv_array',cv),('logits',cv),
                ('n_experts',ci),('n_experts_per_tok',ci),('moe_intermediate',ci),
                ('w_gate_inp',cv),('w_gate_exps',cv),('w_up_exps',cv),('w_down_exps',cv),
                ('gate_exp_quant',ci),('up_exp_quant',ci),('down_exp_quant',ci),
                ('q_quant',cv),('k_quant',cv),('v_quant',cv),('o_quant',cv),
                ('g_quant',cv),('u_quant',cv),('d_quant',cv),('emb_quant',ci),
                ('cos_table',cv),('sin_table',cv),('max_ctx',ci),
                ('workspace',cv),('ws_size',ci)]
class KVBlock(ctypes.Structure):
    _fields_ = [('k',cv),('v',cv),('n_blocks',ci),('seq_len',ci*64),('block_map',(ci*1024)*64)]
lib.batch_forward.argtypes=[cv,cv,ci,cv]; lib.batch_forward.restype=None
lib.kv_init.argtypes=[cv,ci,ci,ci]; lib.kv_init.restype=None

def wa(a): return (cv*L)(*[ctypes.cast(t,cv) for t in a])
def ia(a): return (ci*L)(*[int(t) for t in a])

bc=BC()
for a,v in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',e.eps),('nc',N),('outNR',V),('outNC',N),('outQuant',8),('n_experts',e.n_experts),('n_experts_per_tok',e.n_experts_per_tok),('moe_intermediate',e.n_ff_expert),('gate_exp_quant',0),('up_exp_quant',0),('down_exp_quant',0),('emb_quant',0)]:
    setattr(bc,a,v)

wQ,wK,wV,wO,wG,wU,wD,wAN,wFN=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
nQ,nK,nV,nO,nG,nU,nD=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
qQ,qK,qV,qO,qG,qU,qD=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
for i,lw in enumerate(e._layers):
    for atr,nm in [('attn_q','Q'),('attn_k','K'),('attn_v','V'),('attn_out','O'),('ffn_gate','G'),('ffn_up','U'),('ffn_down','D')]:
        if hasattr(lw,f'{atr}_raw'):rw=getattr(lw,f'{atr}_raw');nr=getattr(lw,f'{atr}_nr').value;qt=getattr(lw,f'{atr}_qt').value
        else:f32=getattr(lw,f'{atr}_f32');rw=f32.ctypes.data_as(cv)if f32 is not None else cv(0);nr=f32.shape[0]if f32 is not None else 0;qt=0
        locals()[f'w{nm}'][i]=rw;locals()[f'n{nm}'][i]=nr;locals()[f'q{nm}'][i]=qt
    wAN[i]=lw.attn_norm_w.ctypes.data_as(cv);wFN[i]=lw.ffn_norm_w.ctypes.data_as(cv)

# FP32 expert weights - dequantize in Python, pass as FP32 pointers
import gguf
Wgate_f32 = []; Wup_f32 = []; Wdown_f32 = []
for i in range(L):
    for t in e.reader.tensors:
        if t.name == f'blk.{i}.ffn_gate_exps.weight':
            f32 = gguf.dequantize(t.data,t.tensor_type).astype(np.float32)
            in_d,out_d,n_exp = int(t.shape[0]),int(t.shape[1]),int(t.shape[2])
            f32 = f32.reshape(n_exp,out_d,in_d)
            gate_f32 = np.ascontiguousarray(f32[0])  # just first expert for test
            Wgate_f32.append(gate_f32.ctypes.data_as(cv))
        if t.name == f'blk.{i}.ffn_down_exps.weight':
            f32 = gguf.dequantize(t.data,t.tensor_type).astype(np.float32)
            in_d,out_d,n_exp = int(t.shape[0]),int(t.shape[1]),int(t.shape[2])
            f32 = f32.reshape(n_exp,out_d,in_d)
            down_f32 = np.ascontiguousarray(f32[0])
            Wdown_f32.append(down_f32.ctypes.data_as(cv))

# Set MoE FP32 weight pointers
w_ge=[0]*L; w_ue=[0]*L; w_de=[0]*L
w_gi=[0]*L
for i in range(L):
    me=e._moe_layers[i]
    w_gi[i]=me.router_f32.ctypes.data_as(cv) if me.router_f32 is not None else cv(0)
    if i < len(Wgate_f32): w_ge[i]=Wgate_f32[i]
    if i < len(Wdown_f32): w_de[i]=Wdown_f32[i]

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

hd2=HD//2;mc=4096
freq_v=e.rope_freq_base**(np.arange(0,HD,2,dtype=np.float32)/HD)
ang=np.arange(mc,dtype=np.float32).reshape(-1,1)/freq_v.reshape(1,-1)
ct=np.cos(ang).astype(np.float32).reshape(-1);st=np.sin(ang).astype(np.float32).reshape(-1)
bc.cos_table=ct.ctypes.data_as(cv);bc.sin_table=st.ctypes.data_as(cv);bc.max_ctx=mc

ws=np.zeros(12*S,dtype=np.float32);lb=np.zeros(V,dtype=np.float32)
kv=KVBlock();lib.kv_init(ctypes.byref(kv),ci(L),ci(NKH),ci(HD))
kva=(cv*1)(ctypes.cast(ctypes.pointer(kv),cv))
bc.kv_array=ctypes.cast(kva,cv);bc.logits=lb.ctypes.data_as(cv)

bc.L=1
print('Running C engine with FP32 expert weights...', flush=True)
t0=time.time()
lib.batch_forward(ctypes.byref(bc),(ci*1)(785),ci(1),ws.ctypes.data_as(cv))
t1=time.time()
print(f'Time: {(t1-t0)*1000:.1f}ms', flush=True)
nan_count=np.isnan(lb[:V]).sum()
print(f'Logits NaN: {nan_count}/{V}', flush=True)
if nan_count > 0:
    print('  FP32 experts still producing NaN!', flush=True)
else:
    top5=np.argsort(lb[:V])[-5:][::-1]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/tmp/qwen3-tokenizer/')
    print(f'  Top-5: {[tok.decode([t]) for t in top5]}', flush=True)
