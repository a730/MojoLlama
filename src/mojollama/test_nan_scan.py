#!/usr/bin/env python3
"""Scan each layer for NaN introduction in Qwen3-30B-A3B inference."""
import sys, os, time, ctypes, numpy as np
os.environ['OMP_NUM_THREADS']='32'
sys.path.insert(0,'/onedev-workspace/work/src/mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE

e=TurboEngineV7MoE('/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf',32)
lib=ctypes.CDLL('/onedev-workspace/work/src/mojollama/kernels/cengine_batch_instr.so')
cv=ctypes.c_void_p;ci=ctypes.c_int

L,N,NH,NKH,HD,FF,V=e.n_layers,e.n_embd,e.n_head,e.n_kv_head,e.head_dim,e.n_ff,e.vocab_size
S=max(N,NH*HD,FF,NKH*HD,e.n_ff_expert)

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

def build_bc(weights_setup=True):
    bc=BC()
    for a,v in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',e.eps),('nc',N),('outNR',V),('outNC',N),('outQuant',8),('n_experts',e.n_experts),('n_experts_per_tok',e.n_experts_per_tok),('moe_intermediate',e.n_ff_expert),('gate_exp_quant',12),('up_exp_quant',12),('down_exp_quant',14),('emb_quant',0)]:
        setattr(bc,a,v)
    
    if weights_setup:
        wQ,wK,wV,wO,wG,wU,wD,wAN,wFN=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
        nQ,nK,nV,nO,nG,nU,nD=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
        qQ,qK,qV,qO,qG,qU,qD=[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L,[0]*L
        for i,lw in enumerate(e._layers):
            for atr,nm in [('attn_q','Q'),('attn_k','K'),('attn_v','V'),('attn_out','O'),('ffn_gate','G'),('ffn_up','U'),('ffn_down','D')]:
                if hasattr(lw,f'{atr}_raw'):rw=getattr(lw,f'{atr}_raw');nr=getattr(lw,f'{atr}_nr').value;qt=getattr(lw,f'{atr}_qt').value
                else:f32=getattr(lw,f'{atr}_f32');rw=f32.ctypes.data_as(cv)if f32 is not None else cv(0);nr=f32.shape[0]if f32 is not None else 0;qt=0
                t=nm
                if t=='Q':wQ[i]=rw;nQ[i]=nr;qQ[i]=qt
                elif t=='K':wK[i]=rw;nK[i]=nr;qK[i]=qt
                elif t=='V':wV[i]=rw;nV[i]=nr;qV[i]=qt
                elif t=='O':wO[i]=rw;nO[i]=nr;qO[i]=qt
                elif t=='G':wG[i]=rw;nG[i]=nr;qG[i]=qt
                elif t=='U':wU[i]=rw;nU[i]=nr;qU[i]=qt
                elif t=='D':wD[i]=rw;nD[i]=nr;qD[i]=qt
            wAN[i]=lw.attn_norm_w.ctypes.data_as(cv);wFN[i]=lw.ffn_norm_w.ctypes.data_as(cv)
        w_gi,w_ge,w_ue,w_de=[0]*L,[0]*L,[0]*L,[0]*L
        for i in range(L):
            if e.is_moe:
                me2=e._moe_layers[i]
                w_gi[i]=me2.router_f32.ctypes.data_as(cv)if me2.router_f32 is not None else cv(0)
                if me2.gate_raw and len(me2.gate_raw)>0:w_ge[i]=me2.gate_raw[0].ctypes.data_as(cv)
                if me2.up_raw and len(me2.up_raw)>0:w_ue[i]=me2.up_raw[0].ctypes.data_as(cv)
                if me2.down_raw and len(me2.down_raw)>0:w_de[i]=me2.down_raw[0].ctypes.data_as(cv)
        bc.wQ=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wQ]),cv)
        bc.wK=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wK]),cv)
        bc.wV=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wV]),cv)
        bc.wO=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wO]),cv)
        bc.wG=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wG]),cv)
        bc.wU=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wU]),cv)
        bc.wD=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wD]),cv)
        bc.wAN=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wAN]),cv)
        bc.wFN=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in wFN]),cv)
        bc.nQ=ctypes.cast((ci*L)(*[int(t)for t in nQ]),cv)
        bc.nK=ctypes.cast((ci*L)(*[int(t)for t in nK]),cv)
        bc.nV=ctypes.cast((ci*L)(*[int(t)for t in nV]),cv)
        bc.nO=ctypes.cast((ci*L)(*[int(t)for t in nO]),cv)
        bc.nG=ctypes.cast((ci*L)(*[int(t)for t in nG]),cv)
        bc.nU=ctypes.cast((ci*L)(*[int(t)for t in nU]),cv)
        bc.nD=ctypes.cast((ci*L)(*[int(t)for t in nD]),cv)
        bc.q_quant=ctypes.cast((ci*L)(*[int(t)for t in qQ]),cv)
        bc.k_quant=ctypes.cast((ci*L)(*[int(t)for t in qK]),cv)
        bc.v_quant=ctypes.cast((ci*L)(*[int(t)for t in qV]),cv)
        bc.o_quant=ctypes.cast((ci*L)(*[int(t)for t in qO]),cv)
        bc.g_quant=ctypes.cast((ci*L)(*[int(t)for t in qG]),cv)
        bc.u_quant=ctypes.cast((ci*L)(*[int(t)for t in qU]),cv)
        bc.d_quant=ctypes.cast((ci*L)(*[int(t)for t in qD]),cv)
        bc.w_gate_inp=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in w_gi]),cv)
        bc.w_gate_exps=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in w_ge]),cv)
        bc.w_up_exps=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in w_ue]),cv)
        bc.w_down_exps=ctypes.cast((cv*L)(*[ctypes.cast(t,cv)for t in w_de]),cv)
        bc.emb=e.emb.ctypes.data_as(cv)
        bc.onw=e._out_norm_w.ctypes.data_as(cv)
        bc.wOut=ctypes.cast(e._out_raw,cv)
        
        hd2=HD//2;mc=4096
        freq_v=e.rope_freq_base**(np.arange(0,HD,2,dtype=np.float32)/HD)
        ang=np.arange(mc,dtype=np.float32).reshape(-1,1)/freq_v.reshape(1,-1)
        ct=np.cos(ang).astype(np.float32).reshape(-1);st=np.sin(ang).astype(np.float32).reshape(-1)
        bc.cos_table=ct.ctypes.data_as(cv);bc.sin_table=st.ctypes.data_as(cv);bc.max_ctx=mc
    
    return bc

bc=build_bc()
s=np.zeros(12*S,dtype=np.float32)
lb=np.zeros(V,dtype=np.float32)
kv=KVBlock();lib.kv_init(ctypes.byref(kv),ci(L),ci(NKH),ci(HD))
kva=(cv*1)(ctypes.cast(ctypes.pointer(kv),cv))
bc.kv_array=ctypes.cast(kva,cv);bc.logits=lb.ctypes.data_as(cv)

# Run all layers first
bc.L=L
lib.batch_forward(ctypes.byref(bc),(ci*1)(785),ci(1),s.ctypes.data_as(cv))
nan_all=np.isnan(lb[:V]).sum()
print(f'Full {L}-layer: NaN={nan_all}/{V}', flush=True)

# If all NaN, find where NaN first appears by running 1 layer at a time
print('\nBinary search for NaN introduction...', flush=True)
lo, hi = 1, L
while lo < hi:
    mid = (lo + hi) // 2
    bc_i=build_bc()
    si=np.zeros(12*S,dtype=np.float32)
    lbi=np.zeros(V,dtype=np.float32)
    kvi=KVBlock();lib.kv_init(ctypes.byref(kvi),ci(L),ci(NKH),ci(HD))
    kvai=(cv*1)(ctypes.cast(ctypes.pointer(kvi),cv))
    bc_i.kv_array=ctypes.cast(kvai,cv);bc_i.logits=lbi.ctypes.data_as(cv)
    bc_i.L=mid
    lib.batch_forward(ctypes.byref(bc_i),(ci*1)(785),ci(1),si.ctypes.data_as(cv))
    nan_c=np.isnan(lbi[:V]).sum()
    print(f'  L=0..{mid-1}: NaN={nan_c}/{V}', flush=True)
    if nan_c > 0:
        hi = mid
    else:
        lo = mid + 1

print(f'\nNaN first appears at layer {lo}', flush=True)
if lo > 0:
    print(f'Re-running up to layer {lo-1} with NaN checks...', flush=True)
