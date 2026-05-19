#!/usr/bin/env python3
"""vLLM-style concurrent LLM serving with per-sequence KV batch_forward."""
import sys,os,json,time,threading,queue,ctypes,traceback
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'kernels'))
from turbo_engine_v77 import TurboEngineV77

lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'kernels','cengine_batch.so'))
cv=ctypes.c_void_p; ci=ctypes.c_int; cf=ctypes.c_float

class BC(ctypes.Structure):
    _fields_ = [("L",ci),("N",ci),("NH",ci),("NKH",ci),("HD",ci),("FF",ci),("V",ci),("eps",cf),
        ("wQ",cv),("wK",cv),("wV",cv),("wO",cv),("wG",cv),("wU",cv),("wD",cv),
        ("wAN",cv),("wFN",cv),
        ("nQ",cv),("nK",cv),("nV",cv),("nO",cv),("nG",cv),("nU",cv),("nD",cv),
        ("nc",ci),("emb",cv),("onw",cv),("wOut",cv),("outNR",ci),("outNC",ci),("outQuant",ci),
        ("kv_array",cv),("logits",cv)]

class KVBlock(ctypes.Structure):
    _fields_ = [("k",cv),("v",cv),("n_blocks",ci),
        ("seq_len",ci*64),("block_map",(ci*1024)*64)]

lib.batch_forward.argtypes=[cv,cv,ci,cv]; lib.batch_forward.restype=None
lib.kv_init.argtypes=[cv,ci,ci,ci]; lib.kv_init.restype=None

forward_lock = threading.Lock()

# ── Model ──
MODEL_PATH = '/tmp/tl-Q4_0.gguf'
print(f"Loading {MODEL_PATH}...", flush=True)
e = TurboEngineV77(MODEL_PATH, 32)
L=e.n_layers; N=e.n_embd; NH=e.n_head; NKH=e.n_kv_head; HD=e.head_dim; FF=e.n_ff; V=e.vocab_size
S = max(N, NH*HD, FF, NKH*HD)
BOS=1; EOS=2
print(f"Model: {L}L/{N}D/{FF}FF/{NH}H/{NKH}KV | {V}vocab | S={S}", flush=True)

def wa(a): return (cv*L)(*[ctypes.cast(a[i],cv) for i in range(L)])
def ia(a): return (ci*L)(*[a[i].value if hasattr(a[i],'value') else int(a[i]) for i in range(L)])

def mk_kv():
    kv = KVBlock()
    lib.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
    return kv

bc=BC()
bc.L=L; bc.N=N; bc.NH=NH; bc.NKH=NKH; bc.HD=HD; bc.FF=FF; bc.V=V; bc.eps=e.eps
bc.wQ=ctypes.cast(wa([e._layers[i]['attn_q']['raw'] for i in range(L)]),cv)
bc.wK=ctypes.cast(wa([e._layers[i]['attn_k']['raw'] for i in range(L)]),cv)
bc.wV=ctypes.cast(wa([e._layers[i]['attn_v']['raw'] for i in range(L)]),cv)
bc.wO=ctypes.cast(wa([e._layers[i]['attn_output']['raw'] for i in range(L)]),cv)
bc.wG=ctypes.cast(wa([e._layers[i]['ffn_gate']['raw'] for i in range(L)]),cv)
bc.wU=ctypes.cast(wa([e._layers[i]['ffn_up']['raw'] for i in range(L)]),cv)
bc.wD=ctypes.cast(wa([e._layers[i]['ffn_down']['raw'] for i in range(L)]),cv)
bc.wAN=ctypes.cast(wa([e._layers[i]['ann_w'].ctypes.data_as(cv) for i in range(L)]),cv)
bc.wFN=ctypes.cast(wa([e._layers[i]['fnn_w'].ctypes.data_as(cv) for i in range(L)]),cv)
bc.nQ=ctypes.cast(ia([e._layers[i]['attn_q']['nr'] for i in range(L)]),cv)
bc.nK=ctypes.cast(ia([e._layers[i]['attn_k']['nr'] for i in range(L)]),cv)
bc.nV=ctypes.cast(ia([e._layers[i]['attn_v']['nr'] for i in range(L)]),cv)
bc.nO=ctypes.cast(ia([e._layers[i]['attn_output']['nr'] for i in range(L)]),cv)
bc.nG=ctypes.cast(ia([e._layers[i]['ffn_gate']['nr'] for i in range(L)]),cv)
bc.nU=ctypes.cast(ia([e._layers[i]['ffn_up']['nr'] for i in range(L)]),cv)
bc.nD=ctypes.cast(ia([e._layers[i]['ffn_down']['nr'] for i in range(L)]),cv)
bc.nc=N; bc.emb=e.emb.ctypes.data_as(cv); bc.onw=e._onw.ctypes.data_as(cv)
bc.wOut=ctypes.cast(e._or,cv); bc.outNR=32000; bc.outNC=2048; bc.outQuant=1
print("BC struct ready", flush=True)

# ── Shared buffers ──
MAX_SEQ = 32
ws = np.zeros(MAX_SEQ * 12 * S, dtype=np.float32)
logits_buf = np.zeros(MAX_SEQ * V, dtype=np.float32)

def sample_token(logits, temperature=0.0):
    if temperature <= 0:
        return int(logits.argmax())
    p = np.exp(np.clip(logits / temperature, -50, 50))
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

class Sequence:
    __slots__ = ('kv','tokens','gen_tokens','done','result','event','max_tokens','temperature')
    def __init__(self):
        self.kv = mk_kv()
        self.tokens = []
        self.gen_tokens = []
        self.done = False
        self.result = None
        self.event = threading.Event()
        self.max_tokens = 100
        self.temperature = 0.0

pending = queue.Queue()
active_slots = [None] * MAX_SEQ

def run_scheduler():
    global bc, ws, logits_buf, active_slots
    while True:
        # Fill empty slots
        while not pending.empty():
            try:
                seq = pending.get_nowait()
            except queue.Empty:
                break
            for i in range(MAX_SEQ):
                if active_slots[i] is None:
                    active_slots[i] = seq
                    break
        
        # Collect batch
        batch = [(i, s) for i, s in enumerate(active_slots) if s is not None and not s.done]
        if not batch:
            time.sleep(0.002)
            continue
        
        B = len(batch)
        tokens_arr = (ci*B)()
        kv_ptrs = (cv*B)()
        for j, (slot, seq) in enumerate(batch):
            tokens_arr[j] = seq.tokens[-1] if seq.tokens else BOS
            kv_ptrs[j] = ctypes.cast(ctypes.pointer(seq.kv), cv)
        
        try:
            with forward_lock:
                bc.kv_array = ctypes.cast(kv_ptrs, cv)
                bc.logits = logits_buf.ctypes.data_as(cv)
                lib.batch_forward(ctypes.byref(bc), tokens_arr, ci(B), ws.ctypes.data_as(cv))
        except Exception as ex:
            print(f"SCHEDULER forward error: {ex}", flush=True)
            continue
        
        for j, (slot, seq) in enumerate(batch):
            logits = logits_buf[j*V:(j+1)*V].copy()
            next_tok = sample_token(logits, seq.temperature)
            seq.tokens.append(next_tok)
            seq.gen_tokens.append(next_tok)
            
            if next_tok == EOS or len(seq.gen_tokens) >= seq.max_tokens:
                seq.done = True
                seq.result = {
                    'tokens': seq.gen_tokens.copy(),
                    'finish_reason': 'stop' if next_tok == EOS else 'length',
                }
                seq.event.set()
                active_slots[slot] = None

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
print("Scheduler started", flush=True)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            print("  [handler] do_POST called", flush=True)
            path = urlparse(self.path).path
            print(f"  [handler] path={path}", flush=True)
            if path != '/v1/completions':
                self.send_error(404)
                return
            
            n = int(self.headers.get('Content-Length', 0))
            print(f"  [handler] content-length={n}", flush=True)
            body = json.loads(self.rfile.read(n)) if n else {}
            prompt = body.get('prompt', '')
            max_tokens = body.get('max_tokens', 100)
            temperature = body.get('temperature', 0.0)
            stream = body.get('stream', False)
            print(f"  [handler] prompt='{prompt}' max_tokens={max_tokens}", flush=True)
            
            prompt_ids = e.encode(prompt) if isinstance(prompt, str) else prompt[:512]
            print(f"  [handler] prompt_ids={len(prompt_ids)} tokens", flush=True)
            t0 = time.perf_counter()
            
            seq = Sequence()
            seq.max_tokens = max_tokens
            seq.temperature = temperature
            
            # Prefill: one token at a time
            print(f"  [handler] starting prefill ({len(prompt_ids)} tokens)", flush=True)
            for tid in prompt_ids:
                kva1 = (cv*1)(ctypes.cast(ctypes.pointer(seq.kv), cv))
                with forward_lock:
                    bc.kv_array = ctypes.cast(kva1, cv)
                    bc.logits = logits_buf[0:V].ctypes.data_as(cv)
                    logits_buf[:V] = 0
                    lib.batch_forward(ctypes.byref(bc), (ci*1)(tid), ci(1), ws.ctypes.data_as(cv))
            print(f"  [handler] prefill done", flush=True)
            
            # Set up for generation: last prompt token is the "current" token
            seq.tokens = [prompt_ids[-1]] if prompt_ids else [BOS]
            
            # Enqueue for generation
            pending.put(seq)
            print(f"  [handler] seq enqueued, waiting...", flush=True)
            
            # Wait for generation to complete
            if not seq.event.wait(timeout=600):
                gen_tokens = seq.gen_tokens[:20]
                finish = 'timeout'
            elif seq.result is None:
                gen_tokens = seq.gen_tokens
                finish = 'length' if len(gen_tokens) >= max_tokens else 'stop'
            else:
                gen_tokens = seq.result['tokens']
                finish = seq.result['finish_reason']
            
            print(f"  [handler] got result: {len(gen_tokens)} tokens, finish={finish}", flush=True)
            t_total = time.perf_counter() - t0
            generated_text = e.decode(gen_tokens)
            
            res = {
                'id': f'cmpl-{int(time.time())}',
                'object': 'text_completion',
                'model': 'mojollama-batch',
                'choices': [{
                    'text': generated_text,
                    'index': 0,
                    'finish_reason': finish,
                    'logprobs': None,
                }],
                'usage': {
                    'prompt_tokens': len(prompt_ids),
                    'completion_tokens': len(gen_tokens),
                    'total_tokens': len(prompt_ids) + len(gen_tokens),
                },
            }
            
            if stream:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for tok in gen_tokens:
                    chunk = json.dumps({'choices':[{'text':e.decode([tok]),'index':0}]})
                    self.wfile.write(f'data: {chunk}\n\n'.encode())
                self.wfile.write('data: [DONE]\n\n'.encode())
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode())
            
            tok_s = len(gen_tokens) / t_total if t_total > 0 else 0
            print(f"  [{len(prompt_ids)}p+{len(gen_tokens)}g {t_total:.1f}s {tok_s:.0f}t/s] '{generated_text[:60]}'", flush=True)
        except Exception as ex:
            print(f"  [handler] EXCEPTION: {ex}", flush=True)
            traceback.print_exc()
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(ex)}).encode())
            except:
                pass
    
    def log_message(self, *a): pass

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv)>1 else 9000
    print(f"Listening on 0.0.0.0:{port}/v1/completions", flush=True)
    
    from socketserver import ThreadingMixIn
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True
    
    server = ThreadedHTTPServer(('0.0.0.0', port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown", flush=True)
