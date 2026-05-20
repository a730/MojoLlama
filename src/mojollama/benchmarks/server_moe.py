#!/usr/bin/env python3
"""MoE K-quant server using cengine_batch.so with TurboEngineV7MoE."""
import sys,os,json,time,threading,queue,ctypes
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from socketserver import ThreadingMixIn

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else '/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

# Auto-load tuned config for thread count & concurrency
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_moe_threads = 32
try:
    from mojollama.backends import load_tuned_config
    _cfg = load_tuned_config()
    if _cfg and 'mojollama_engine' in _cfg:
        _me = _cfg['mojollama_engine']
        _moe_threads = _me.get('threads', 32)
        os.environ['OMP_NUM_THREADS'] = str(_moe_threads)
        print(f"[Config] Loaded auto-tuned settings: {_moe_threads} threads, "
              f"optimal concurrency {_me.get('optimal_concurrency', '?')}", flush=True)
except Exception:
    pass

sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'kernels'))
from turbo_engine_v7_moe import TurboEngineV7MoE
e = TurboEngineV7MoE(MODEL_PATH, _moe_threads)
L=e.n_layers; N=e.n_embd; NH=e.n_head; NKH=e.n_kv_head; HD=e.head_dim; FF=e.n_ff; V=e.vocab_size
NE=e.n_experts; NK=e.n_experts_per_tok; moe_int=e.n_ff_expert
S = max(N, NH*HD, FF, NKH*HD, moe_int); BOS=1; EOS=2
print(f"Engine: {L}L/{N}D/{FF}FF/{NH}H/{NKH}KV | MoE {NE}×{NK} | int={moe_int} | V={V}", flush=True)

cv=ctypes.c_void_p; ci=ctypes.c_int; cf=ctypes.c_float
lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'kernels','cengine_batch_instr.so'))

class BC(ctypes.Structure):
    _fields_ = [("L",ci),("N",ci),("NH",ci),("NKH",ci),("HD",ci),("FF",ci),("V",ci),("eps",cf),
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
        ("workspace",cv),("ws_size",ci)]

class KVBlock(ctypes.Structure):
    _fields_ = [("k",cv),("v",cv),("n_blocks",ci),("seq_len",ci*64),("block_map",(ci*1024)*64)]

lib.kv_init.argtypes=[cv,ci,ci,ci]; lib.kv_init.restype=None
lib.batch_forward.argtypes=[cv,cv,ci,cv]; lib.batch_forward.restype=None

# Build per-layer pointer arrays
def wa(arr): return (cv*L)(*[ctypes.cast(a,cv) for a in arr])
def ia(arr): return (ci*L)(*[int(a) for a in arr])

wQ_arr=[0]*L; wK_arr=[0]*L; wV_arr=[0]*L; wO_arr=[0]*L
wG_arr=[0]*L; wU_arr=[0]*L; wD_arr=[0]*L
wAN_arr=[0]*L; wFN_arr=[0]*L
wQK_arr=[0]*L; qkQ_arr=[0]*L  # fused Q+K weight (NULL if types don't match)
nQ_arr=[0]*L; nK_arr=[0]*L; nV_arr=[0]*L; nO_arr=[0]*L; nG_arr=[0]*L; nU_arr=[0]*L; nD_arr=[0]*L
qQ_arr=[0]*L; qK_arr=[0]*L; qV_arr=[0]*L; qO_arr=[0]*L; qG_arr=[0]*L; qU_arr=[0]*L; qD_arr=[0]*L

for i,lw in enumerate(e._layers):
    wQ_arr[i]=lw.attn_q_raw; nQ_arr[i]=lw.attn_q_nr.value; qQ_arr[i]=lw.attn_q_qt.value
    wK_arr[i]=lw.attn_k_raw; nK_arr[i]=lw.attn_k_nr.value; qK_arr[i]=lw.attn_k_qt.value
    wV_arr[i]=lw.attn_v_raw; nV_arr[i]=lw.attn_v_nr.value; qV_arr[i]=lw.attn_v_qt.value
    wO_arr[i]=lw.attn_out_raw; nO_arr[i]=lw.attn_out_nr.value; qO_arr[i]=lw.attn_out_qt.value
    wAN_arr[i]=lw.attn_norm_w.ctypes.data_as(cv); wFN_arr[i]=lw.ffn_norm_w.ctypes.data_as(cv)
    # Fused Q+K weight (if available)
    if hasattr(lw,'attn_qk_use_c') and lw.attn_qk_use_c:
        wQK_arr[i]=lw.attn_qk_raw; qkQ_arr[i]=lw.attn_qk_qt.value
    else:
        wQK_arr[i]=cv(0); qkQ_arr[i]=0
for i,me in enumerate(e._moe_layers):
    wG_arr[i]=me.gate_raw[0].ctypes.data_as(cv) if me.gate_raw else cv(0)
    wU_arr[i]=me.up_raw[0].ctypes.data_as(cv) if me.up_raw else cv(0)
    wD_arr[i]=me.down_raw[0].ctypes.data_as(cv) if me.down_raw else cv(0)
    nG_arr[i]=me.gate_nr.value; nU_arr[i]=me.up_nr.value; nD_arr[i]=me.down_nr.value
    qG_arr[i]=me.gate_qt.value; qU_arr[i]=me.up_qt.value; qD_arr[i]=me.down_qt.value

# MoE pointers
w_gate_inp_arr=[0]*L; w_gate_exps_arr=[0]*L; w_up_exps_arr=[0]*L; w_down_exps_arr=[0]*L
for i,lw in enumerate(e._layers):
    if e.is_moe:
        me = e._moe_layers[i]
        if me.router_raw is not None:
            w_gate_inp_arr[i] = me.router_raw.ctypes.data_as(cv)
        elif me.router_f32 is not None:
            w_gate_inp_arr[i] = me.router_f32.ctypes.data_as(cv)
        else:
            w_gate_inp_arr[i] = cv(0)
        # Expert weight pointers — pass raw blob pointer, C code offsets by stride
        # Use the underlying contiguous raw weight data address
        gate_raw_t = me.gate_raw  # list of numpy expert views (contiguous in memory)
        up_raw_t = me.up_raw
        down_raw_t = me.down_raw
        if gate_raw_t and len(gate_raw_t) > 0:
            # All experts are contiguous; first expert's pointer = base for all
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

# Read actual expert quant types from engine (like test_fixed.py)
if e.is_moe and len(e._moe_layers) > 0:
    me = e._moe_layers[0]
    gate_qt = me.gate_qt.value; up_qt = me.up_qt.value; down_qt = me.down_qt.value
else:
    gate_qt = 2; up_qt = 2; down_qt = 2
print(f"Expert quant types: gate={gate_qt} up={up_qt} down={down_qt}", flush=True)

bc=BC()
for attr,val in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',e.eps),
                 ('nc',N),('outNR',V),('outNC',N),('outQuant',8),
                 ('n_experts',NE),('n_experts_per_tok',NK),('moe_intermediate',moe_int),
                 ('gate_exp_quant',gate_qt),('up_exp_quant',up_qt),('down_exp_quant',down_qt),
                 ('emb_quant',0)]:
    setattr(bc,attr,val)
bc.wQ=ctypes.cast(wa(wQ_arr),cv); bc.wK=ctypes.cast(wa(wK_arr),cv); bc.wV=ctypes.cast(wa(wV_arr),cv); bc.wO=ctypes.cast(wa(wO_arr),cv)
bc.wG=ctypes.cast(wa(wG_arr),cv); bc.wU=ctypes.cast(wa(wU_arr),cv); bc.wD=ctypes.cast(wa(wD_arr),cv)
bc.wAN=ctypes.cast(wa(wAN_arr),cv); bc.wFN=ctypes.cast(wa(wFN_arr),cv)
bc.wQK=ctypes.cast(wa(wQK_arr),cv); bc.qk_quant=ctypes.cast(ia(qkQ_arr),cv)
bc.nQ=ctypes.cast(ia(nQ_arr),cv); bc.nK=ctypes.cast(ia(nK_arr),cv); bc.nV=ctypes.cast(ia(nV_arr),cv); bc.nO=ctypes.cast(ia(nO_arr),cv)
bc.nG=ctypes.cast(ia(nG_arr),cv); bc.nU=ctypes.cast(ia(nU_arr),cv); bc.nD=ctypes.cast(ia(nD_arr),cv)
bc.q_quant=ctypes.cast(ia(qQ_arr),cv); bc.k_quant=ctypes.cast(ia(qK_arr),cv); bc.v_quant=ctypes.cast(ia(qV_arr),cv); bc.o_quant=ctypes.cast(ia(qO_arr),cv)
bc.g_quant=ctypes.cast(ia(qG_arr),cv); bc.u_quant=ctypes.cast(ia(qU_arr),cv); bc.d_quant=ctypes.cast(ia(qD_arr),cv)
bc.w_gate_inp=ctypes.cast(wa(w_gate_inp_arr),cv); bc.w_gate_exps=ctypes.cast(wa(w_gate_exps_arr),cv)
bc.w_up_exps=ctypes.cast(wa(w_up_exps_arr),cv); bc.w_down_exps=ctypes.cast(wa(w_down_exps_arr),cv)
bc.emb=e.emb.ctypes.data_as(cv) if hasattr(e,'emb') else cv(0)
bc.onw=e._out_norm_w.ctypes.data_as(cv) if hasattr(e,'_out_norm_w') else cv(0)
# FIX: use _out_raw instead of _or (like test_fixed.py)
bc.wOut=ctypes.cast(e._out_raw,cv) if hasattr(e,'_out_raw') else cv(0)

# RoPE precompute — Python approach from test_fixed.py
max_ctx=4096; hd2=HD//2
freq = e.rope_freq_base ** (np.arange(0, HD, 2, dtype=np.float32) / HD)
ang = np.arange(max_ctx, dtype=np.float32).reshape(-1, 1) / freq.reshape(1, -1)
cos_all = np.cos(ang).astype(np.float32).reshape(-1)
sin_all = np.sin(ang).astype(np.float32).reshape(-1)
cos_t = cos_all.ctypes.data_as(cv)
sin_t = sin_all.ctypes.data_as(cv)
bc.cos_table=cos_t; bc.sin_table=sin_t; bc.max_ctx=max_ctx
print(f"RoPE table: {max_ctx}x{hd2} = {len(cos_all)*4*2//1024}KB", flush=True)

# BC workspace: replaces __builtin_alloca for kct/vct in batch_forward
# Size = 2 * max_ctx * NKH * HD (kct + vct) + safety margin
ws_bc_size = 2 * max_ctx * NKH * HD + 4096
ws_bc = np.zeros(ws_bc_size, dtype=np.float32)
bc.workspace = ws_bc.ctypes.data_as(cv)
bc.ws_size = ws_bc_size
print(f"BC workspace: {ws_bc_size} floats = {ws_bc_size*4//1024}KB", flush=True)

MAX_SEQ=32; ws=np.zeros(MAX_SEQ*12*S,dtype=np.float32); logits_buf=np.zeros(MAX_SEQ*V,dtype=np.float32)

def sample_token(logits,temperature=0.0):
    safe=np.nan_to_num(logits,nan=-1e10,posinf=1e10,neginf=-1e10)
    if temperature<=0: return int(np.argmax(safe))
    safe-=safe.max(); p=np.exp(np.clip(safe/temperature,-50,50)); p[np.isnan(p)]=0
    s=p.sum()
    return int(np.random.choice(len(p),p=p/s)) if s>0 else 0

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/tmp/qwen3-tokenizer/')

class Seq:
    __slots__ = ('kv','tokens','gen_tokens','done','result','event','max_tokens','temperature','token_queue')
    def __init__(s): s.kv=KVBlock(); lib.kv_init(ctypes.byref(s.kv),ci(L),ci(NKH),ci(HD)); s.tokens=[]; s.gen_tokens=[]; s.done=False; s.result=None; s.event=threading.Event(); s.max_tokens=100; s.temperature=0.0; s.token_queue=None

pending=queue.Queue(); active=[None]*MAX_SEQ
forward_lock = threading.Lock()

def scheduler():
    while True:
        while not pending.empty():
            try: s=pending.get_nowait()
            except: break
            for i in range(MAX_SEQ):
                if active[i] is None: active[i]=s; break
        batch=[(i,s) for i,s in enumerate(active) if s and not s.done]
        if not batch: time.sleep(0.002); continue
        B=len(batch); tok_arr=(ci*B)(); kv_arr=(cv*B)()
        for j,(i,s) in enumerate(batch): tok_arr[j]=s.tokens[-1] if s.tokens else 1; kv_arr[j]=ctypes.cast(ctypes.pointer(s.kv),cv)
        with forward_lock:
            bc.kv_array=ctypes.cast(kv_arr,cv); bc.logits=logits_buf.ctypes.data_as(cv)
            lib.batch_forward(ctypes.byref(bc),tok_arr,ci(B),ws.ctypes.data_as(cv))
        for j,(i,s) in enumerate(batch):
            tok=sample_token(logits_buf[j*V:(j+1)*V].copy(),s.temperature)
            s.tokens.append(tok); s.gen_tokens.append(tok)
            if s.token_queue: s.token_queue.put(tok)
            if tok==2 or len(s.gen_tokens)>=s.max_tokens:
                s.done=True; s.result={'tokens':s.gen_tokens.copy(),'finish_reason':'stop'if tok==2 else'length'}
                if s.token_queue: s.token_queue.put(None)
                s.event.set(); active[i]=None
threading.Thread(target=scheduler,daemon=True).start()

class ThreadedHTTPServer(ThreadingMixIn,HTTPServer):
    allow_reuse_address=True; daemon_threads=True

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if urlparse(self.path).path!='/v1/completions': self.send_error(404); return
        n=int(self.headers.get('Content-Length',0))
        body=json.loads(self.rfile.read(n)) if n else {}
        prompt=body.get('prompt',''); max_tokens=body.get('max_tokens',100)
        temperature=body.get('temperature',0.0); stream=body.get('stream',False)
        t0=time.perf_counter()
        try:
            ids=tokenizer.encode(prompt) if isinstance(prompt,str) else prompt[:512]
            print(f"  Encode: {len(ids)} ids", flush=True)
            # Filter out-of-range tokens
            ids=[t for t in ids if 0 <= t < V][:512]
            if not ids: ids=[1]
            s=Seq(); s.max_tokens=max_tokens; s.temperature=temperature
            print(f"  Prefill {len(ids)} tokens...", flush=True)
            if ids:
                kva1=(cv*1)(ctypes.cast(ctypes.pointer(s.kv),cv))
                with forward_lock:
                    bc.kv_array=ctypes.cast(kva1,cv); bc.logits=logits_buf[0:V].ctypes.data_as(cv)
                    logits_buf[:V]=0
                    for pt in ids:
                        lib.batch_forward(ctypes.byref(bc),(ci*1)(pt),ci(1),ws.ctypes.data_as(cv))
            print(f"  First sample...", flush=True)
            ft=sample_token(logits_buf[:V].copy(),temperature)
            print(f"  First token: {ft}", flush=True)
            if stream: s.token_queue=queue.Queue(); s.token_queue.put(ft)
            s.tokens.append(ft)
            pending.put(s)
            if stream:
                self.send_response(200); self.send_header('Content-Type','text/event-stream')
                self.send_header('Cache-Control','no-cache'); self.end_headers()
                while True:
                    t=s.token_queue.get()
                    if t is None: break
                    self.wfile.write(f'data: {json.dumps({"choices":[{"text":tokenizer.decode([t],skip_special_tokens=True),"index":0}]})}\n\n'.encode()); self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')
            else:
                s.event.wait(timeout=600)
                gt=s.gen_tokens if s.result is None else s.result['tokens']
                text=tokenizer.decode(gt,skip_special_tokens=True)
                self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
                self.wfile.write(json.dumps({'id':f'cmpl-{int(time.time())}','object':'text_completion','model':'mojollama','choices':[{'text':text,'index':0,'finish_reason':'length'if len(gt)>=max_tokens else'stop'}],'usage':{'prompt_tokens':len(ids),'completion_tokens':len(gt),'total_tokens':len(ids)+len(gt)}}).encode())
            print(f"  [{len(ids)}p+{len(s.gen_tokens)}g {time.perf_counter()-t0:.1f}s] '{tokenizer.decode(s.gen_tokens,skip_special_tokens=True)[:60]}'",flush=True)
        except Exception as ex:
            self.send_response(500); self.send_header('Content-Type','application/json'); self.end_headers()
            self.wfile.write(json.dumps({'error':str(ex)}).encode())
            import traceback; traceback.print_exc()
    def log_message(self,*a): pass

if __name__=='__main__':
    print(f"Serving on 0.0.0.0:{PORT} | MoE K-quant engine", flush=True)
    HTTPServer(('0.0.0.0',PORT),Handler).serve_forever()
