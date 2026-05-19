#!/usr/bin/env python3
"""Unified server: dense models → our batch engine, MoE models → llama.cpp backend."""
import sys,os,json,time,threading,queue,ctypes,subprocess,atexit,urllib.request
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from socketserver import ThreadingMixIn

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else '/tmp/tl-Q4_0.gguf'
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
LLAMA_CPP = '/tmp/llama.cpp/build/bin/llama-server'

# ── Fast MoE detection ──
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'kernels'))
import gguf
r = gguf.GGUFReader(MODEL_PATH)
arch = str(r.fields.get('general.architecture').parts[0])
has_expert = any('expert' in k.lower() for k in r.fields)
is_moe = has_expert or 'moe' in arch.lower()
del r  # free

if is_moe:
    print(f"MoE model ({arch}) — using llama.cpp backend", flush=True)
    llama_port = PORT + 100
    subprocess.run(['fuser','-k',f'{llama_port}/tcp'], capture_output=True)
    proc = subprocess.Popen([LLAMA_CPP,'-m',MODEL_PATH,'-c','4096','-t','32','-tb','16',
        '-b','1024','-ub','256','-np','8','--port',str(llama_port),'--host','127.0.0.1',
        '--no-webui','--mlock','--cont-batching','-fa','1','-C','0x00000000FFFFFFFF'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(lambda: proc and proc.kill())
    for i in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{llama_port}/health',timeout=2)
            print(f"llama.cpp ready on port {llama_port}", flush=True); break
        except:
            if i%10==9: print(f"  waiting for llama.cpp... ({i*2}s)", flush=True)
    else:
        print("ERROR: llama.cpp failed to start", flush=True); sys.exit(1)
else:
    print(f"Dense model ({arch}) — using custom batch engine", flush=True)

# ── Tokenizer ──
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/tmp/tinyllama-tokenizer/')

# ── For dense: custom engine setup ──
cv=ctypes.c_void_p; ci=ctypes.c_int; cf=ctypes.c_float
if not is_moe:
    from turbo_engine_v77 import TurboEngineV77
    e = TurboEngineV77(MODEL_PATH,32)
    L=e.n_layers; N=e.n_embd; NH=e.n_head; NKH=e.n_kv_head; HD=e.head_dim; FF=e.n_ff; V=e.vocab_size
    S = max(N, NH*HD, FF, NKH*HD); BOS=1; EOS=2
    print(f"Engine: {L}L/{N}D/{FF}FF/{NH}H/{NKH}KV | {V}vocab", flush=True)
    
    lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'kernels','cengine_batch.so'))
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
            ("cos_table",cv),("sin_table",cv),("max_ctx",ci)]
    class KVBlock(ctypes.Structure):
        _fields_ = [("k",cv),("v",cv),("n_blocks",ci),("seq_len",ci*64),("block_map",(ci*1024)*64)]
    lib.batch_forward.argtypes=[cv,cv,ci,cv]; lib.batch_forward.restype=None
    lib.prefill_forward.argtypes=[cv,cv,ci,cv]; lib.prefill_forward.restype=None
    lib.kv_init.argtypes=[cv,ci,ci,ci]; lib.kv_init.restype=None
    forward_lock = threading.Lock()
    def wa(a): return (cv*L)(*[ctypes.cast(a[i],cv) for i in range(L)])
    def ia(a): return (ci*L)(*[a[i].value if hasattr(a[i],'value') else int(a[i]) for i in range(L)])
    def mk_kv(): kv=KVBlock(); lib.kv_init(ctypes.byref(kv),ci(L),ci(NKH),ci(HD)); return kv
    
    bc=BC()
    for attr,val in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',e.eps)]: setattr(bc,attr,val)
    for w,src in [('wQ','attn_q'),('wK','attn_k'),('wV','attn_v'),('wO','attn_output'),('wG','ffn_gate'),('wU','ffn_up'),('wD','ffn_down')]:
        setattr(bc,w,ctypes.cast(wa([e._layers[i][src]['raw'] for i in range(L)]),cv))
    for w,src in [('wAN','ann_w'),('wFN','fnn_w')]:
        setattr(bc,w,ctypes.cast(wa([e._layers[i][src].ctypes.data_as(cv) for i in range(L)]),cv))
    for w,src in [('nQ','attn_q'),('nK','attn_k'),('nV','attn_v'),('nO','attn_output'),('nG','ffn_gate'),('nU','ffn_up'),('nD','ffn_down')]:
        setattr(bc,w,ctypes.cast(ia([e._layers[i][src]['nr'] for i in range(L)]),cv))
    bc.nc=N; bc.emb=e.emb.ctypes.data_as(cv); bc.onw=e._onw.ctypes.data_as(cv)
    bc.wOut=ctypes.cast(e._or,cv); bc.outNR=V; bc.outNC=N; bc.outQuant=1
    
    # Quant type fields (all Q4_0=2 for this dense model, type dispatch unused)
    bc._qtypes=(ci*L)(*[2]*L)
    for w in 'q_quant k_quant v_quant o_quant g_quant u_quant d_quant'.split():
        setattr(bc,w,ctypes.cast(bc._qtypes,cv))
    bc.emb_quant=0; bc.n_experts=0; bc.n_experts_per_tok=0; bc.moe_intermediate=0
    bc.w_gate_inp=cv(0); bc.w_gate_exps=cv(0); bc.w_up_exps=cv(0); bc.w_down_exps=cv(0)
    bc.gate_exp_quant=0; bc.up_exp_quant=0; bc.down_exp_quant=0
    
    # RoPE precompute (keep refs alive — Python GC would free the arrays)
    max_ctx=4096; hd2=HD//2
    bc._cos_t=(cf*(max_ctx*hd2))(); bc._sin_t=(cf*(max_ctx*hd2))()
    lib.rope_init(ctypes.cast(bc._cos_t,cv),ctypes.cast(bc._sin_t,cv),ci(max_ctx),ci(HD))
    bc.cos_table=ctypes.cast(bc._cos_t,cv); bc.sin_table=ctypes.cast(bc._sin_t,cv); bc.max_ctx=max_ctx
    print(f"RoPE table: {max_ctx}x{hd2} = {max_ctx*hd2*4*2//1024}KB", flush=True)
    
    MAX_SEQ=32; ws=np.zeros(MAX_SEQ*12*S,dtype=np.float32); logits_buf=np.zeros(MAX_SEQ*V,dtype=np.float32)
    def sample_token(logits,temperature=0.0):
        safe=np.nan_to_num(logits,nan=-1e10,posinf=1e10,neginf=-1e10)
        if temperature<=0: return int(np.argmax(safe))
        safe-=safe.max(); p=np.exp(np.clip(safe/temperature,-50,50)); p[np.isnan(p)]=0
        s=p.sum()
        return int(np.random.choice(len(p),p=p/s)) if s>0 else 0
    
    class Seq:
        __slots__ = ('kv','tokens','gen_tokens','done','result','event','max_tokens','temperature','token_queue')
        def __init__(s): s.kv=mk_kv(); s.tokens=[]; s.gen_tokens=[]; s.done=False; s.result=None; s.event=threading.Event(); s.max_tokens=100; s.temperature=0.0; s.token_queue=None
    
    pending=queue.Queue(); active=[None]*MAX_SEQ
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

# ── HTTP handler ──
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
        
        if is_moe:
            data=json.dumps({"prompt":prompt,"n_predict":max_tokens,"temperature":temperature,"stream":False}).encode()
            req=urllib.request.Request(f'http://127.0.0.1:{llama_port}/v1/completions',data=data,headers={'Content-Type':'application/json'})
            try:
                resp=urllib.request.urlopen(req,timeout=600)
                body=json.loads(resp.read())
                text=body['choices'][0]['text'] if 'choices' in body else ''
                self.send_response(200)
                self.send_header('Content-Type','application/json'); self.end_headers()
                self.wfile.write(json.dumps({'id':f'cmpl-{int(time.time())}','object':'text_completion','model':os.path.basename(MODEL_PATH),'choices':[{'text':text,'index':0,'finish_reason':'stop','logprobs':None}]}).encode())
                print(f"  [{time.perf_counter()-t0:.1f}s] '{text[:60]}'", flush=True)
            except Exception as ex:
                self.send_response(500)
                self.send_header('Content-Type','application/json'); self.end_headers()
                self.wfile.write(json.dumps({'error':str(ex)}).encode())
        else:
            try:
                ids=tokenizer.encode(prompt) if isinstance(prompt,str) else prompt[:512]
                ids=[t for t in ids if t!=1]
                s=Seq(); s.max_tokens=max_tokens; s.temperature=temperature
                if ids:
                    kva1=(cv*1)(ctypes.cast(ctypes.pointer(s.kv),cv))
                    p_ws=np.zeros(len(ids)*12*S,dtype=np.float32)
                    with forward_lock:
                        bc.kv_array=ctypes.cast(kva1,cv); bc.logits=logits_buf[0:V].ctypes.data_as(cv)
                        logits_buf[:V]=0
                        lib.prefill_forward(ctypes.byref(bc),(ci*len(ids))(*ids),ci(len(ids)),p_ws.ctypes.data_as(cv))
                ft=sample_token(logits_buf[:V].copy(),temperature)
                s.tokens=[ft]; s.gen_tokens=[ft]
                if stream: s.token_queue=queue.Queue(); s.token_queue.put(ft)
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
                    self.wfile.write(json.dumps({'id':f'cmpl-{int(time.time())}','object':'text_completion','model':'mojollama-batch','choices':[{'text':text,'index':0,'finish_reason':'length'if len(gt)>=max_tokens else'stop'}],'usage':{'prompt_tokens':len(ids),'completion_tokens':len(gt),'total_tokens':len(ids)+len(gt)}}).encode())
                print(f"  [{len(ids)}p+{len(s.gen_tokens)}g {time.perf_counter()-t0:.1f}s] '{tokenizer.decode(s.gen_tokens,skip_special_tokens=True)[:60]}'",flush=True)
            except Exception as ex:
                self.send_response(500); self.send_header('Content-Type','application/json'); self.end_headers()
                self.wfile.write(json.dumps({'error':str(ex)}).encode())
                import traceback; traceback.print_exc()
    def log_message(self,*a): pass

if __name__=='__main__':
    print(f"Serving on 0.0.0.0:{PORT} | backend={'llama.cpp (MoE)' if is_moe else 'custom engine (dense)'}", flush=True)
    HTTPServer(('0.0.0.0',PORT),Handler).serve_forever()
