#!/usr/bin/env python3
"""
Qwen3.6 MXFP4 continuous batching server — B_MAX=10 concurrent users.
Uses C engine batch_forward with full SSM+attention hybrid support.
"""
import sys, os, json, time, threading, queue, ctypes
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from socketserver import ThreadingMixIn

MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
B_MAX = 10  # concurrent users

# ── Engine ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels'))
from turbo_engine_v7_moe import TurboEngineV7MoE
e = TurboEngineV7MoE(MODEL, 32)
L=e.n_layers; N=e.n_embd; NH=e.n_head; NKH=e.n_kv_head; HD=e.head_dim; FF=e.n_ff; V=e.vocab_size
NE=e.n_experts; NK=e.n_experts_per_tok; moe_int=e.n_ff_expert
LT = e.layer_types if hasattr(e, 'layer_types') else None
S = max(N, NH*HD, FF, NKH*HD, moe_int, 8192); BOS=1; EOS=2
print(f"Engine: {L}L/{N}D/{NH}H/{NKH}KV/{HD}hd | MoE {NE}x{NK} | V={V} | B_MAX={B_MAX}", flush=True)

# ── C library ──
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels', 'cengine_batch_instr.so'))
cv=ctypes.c_void_p; ci=ctypes.c_int; cf=ctypes.c_float

# ── Structs ──
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

lib.kv_init.argtypes=[cv,ci,ci,ci]; lib.kv_init.restype=None
lib.batch_forward.argtypes=[cv,cv,ci,cv]; lib.batch_forward.restype=None
lib.rope_init.argtypes=[cv,cv,ci,ci,ci]; lib.rope_init.restype=None

def wa(arr): return (cv*L)(*[ctypes.cast(a,cv) for a in arr])
def ia(arr): return (ci*L)(*[int(a) for a in arr])

# ── Build Qwen3.6 weight arrays ──
print("Building Qwen3.6 weight arrays...", flush=True)

wQKV=[cv(0)]*L; qkv_qt=[0]*L
wQ_sep=[cv(0)]*L; nQ_sep=[0]*L; qQ_sep=[0]*L
wK_sep=[cv(0)]*L; nK_sep=[0]*L; qK_sep=[0]*L
wV_sep=[cv(0)]*L; nV_sep=[0]*L; qV_sep=[0]*L
wAG=[cv(0)]*L; ag_qt=[0]*L
wO_attn=[cv(0)]*L; nO_attn=[0]*L; qO_attn=[0]*L
wSO=[cv(0)]*L; so_qt=[0]*L
ssm_c1d=[cv(0)]*L; ssm_a=[cv(0)]*L; ssm_db=[cv(0)]*L
ssm_al=[cv(0)]*L; ssm_be=[cv(0)]*L; ssm_nm=[cv(0)]*L
wSHG=[cv(0)]*L; wSHU=[cv(0)]*L; wSHD=[cv(0)]*L; wSHR=[cv(0)]*L
shg_qt=[0]*L; shu_qt=[0]*L; shd_qt=[0]*L
wAN=[0]*L; wFN=[0]*L
wGI=[cv(0)]*L; wGE=[cv(0)]*L; wUE=[cv(0)]*L; wDE=[cv(0)]*L
ns_G=[0]*L; ns_U=[0]*L; ns_D=[0]*L; qt_G=[0]*L; qt_U=[0]*L; qt_D=[0]*L

for i in range(L):
    lw=e._layers[i]; me=e._moe_layers[i]
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
        wGE[i]=me.gate_raw[0].ctypes.data_as(cv)
        ns_G[i]=me.gate_nr.value; qt_G[i]=me.gate_qt.value
    if me.up_raw and len(me.up_raw)>0:
        wUE[i]=me.up_raw[0].ctypes.data_as(cv)
        ns_U[i]=me.up_nr.value; qt_U[i]=me.up_qt.value
    if me.down_raw and len(me.down_raw)>0:
        wDE[i]=me.down_raw[0].ctypes.data_as(cv)
        ns_D[i]=me.down_nr.value; qt_D[i]=me.down_qt.value
    if hasattr(lw,'attn_norm_w') and lw.attn_norm_w is not None:
        wAN[i]=lw.attn_norm_w.ctypes.data_as(cv)
    if hasattr(lw,'ffn_norm_w') and lw.ffn_norm_w is not None:
        wFN[i]=lw.ffn_norm_w.ctypes.data_as(cv)

me0=e._moe_layers[0]
gate_qt_all=me0.gate_qt.value; up_qt_all=me0.up_qt.value; down_qt_all=me0.down_qt.value

# ── BC struct ──
bc=BC()
for a,v in [('L',L),('N',N),('NH',NH),('NKH',NKH),('HD',HD),('FF',FF),('V',V),('eps',e.eps),
            ('nc',N),('outNR',V),('outNC',N),('outQuant',8),
            ('n_experts',NE),('n_experts_per_tok',NK),('moe_intermediate',moe_int),
            ('gate_exp_quant',gate_qt_all),('up_exp_quant',up_qt_all),('down_exp_quant',down_qt_all),
            ('emb_quant',0),('max_ctx',4096),('rope_dim',HD),('full_attn_interval',4),('n_layers_actual',L)]:
    setattr(bc,a,v)

bc.wQKV=ctypes.cast(wa(wQKV),cv); bc.qkv_quant=ctypes.cast(ia(qkv_qt),cv)
bc.wQ=ctypes.cast(wa(wQ_sep),cv); bc.wK=ctypes.cast(wa(wK_sep),cv); bc.wV=ctypes.cast(wa(wV_sep),cv)
bc.nQ=ctypes.cast(ia(nQ_sep),cv); bc.nK=ctypes.cast(ia(nK_sep),cv); bc.nV=ctypes.cast(ia(nV_sep),cv)
bc.q_quant=ctypes.cast(ia(qQ_sep),cv); bc.k_quant=ctypes.cast(ia(qK_sep),cv); bc.v_quant=ctypes.cast(ia(qV_sep),cv)
bc.wAttnG=ctypes.cast(wa(wAG),cv); bc.attnG_quant=ctypes.cast(ia(ag_qt),cv)
bc.wO=ctypes.cast(wa(wO_attn),cv); bc.nO=ctypes.cast(ia(nO_attn),cv); bc.o_quant=ctypes.cast(ia(qO_attn),cv)
bc.wSsmOut=ctypes.cast(wa(wSO),cv); bc.ssm_out_quant=ctypes.cast(ia(so_qt),cv)
bc.ssm_conv1d=ctypes.cast(wa(ssm_c1d),cv); bc.ssm_a=ctypes.cast(wa(ssm_a),cv)
bc.ssm_dt_bias=ctypes.cast(wa(ssm_db),cv); bc.ssm_alpha=ctypes.cast(wa(ssm_al),cv)
bc.ssm_beta=ctypes.cast(wa(ssm_be),cv); bc.ssm_norm=ctypes.cast(wa(ssm_nm),cv)
bc.wSHexpG=ctypes.cast(wa(wSHG),cv); bc.wSHexpU=ctypes.cast(wa(wSHU),cv)
bc.wSHexpD=ctypes.cast(wa(wSHD),cv); bc.wShexpRouter=ctypes.cast(wa(wSHR),cv)
bc.shexp_g_quant=ctypes.cast(ia(shg_qt),cv); bc.shexp_u_quant=ctypes.cast(ia(shu_qt),cv)
bc.shexp_d_quant=ctypes.cast(ia(shd_qt),cv)
bc.wAN=ctypes.cast(wa(wAN),cv); bc.wFN=ctypes.cast(wa(wFN),cv)
bc.w_gate_inp=ctypes.cast(wa(wGI),cv)
bc.w_gate_exps=ctypes.cast(wa(wGE),cv); bc.w_up_exps=ctypes.cast(wa(wUE),cv)
bc.w_down_exps=ctypes.cast(wa(wDE),cv)
bc.nG=ctypes.cast(ia(ns_G),cv); bc.nU=ctypes.cast(ia(ns_U),cv); bc.nD=ctypes.cast(ia(ns_D),cv)
bc.g_quant=ctypes.cast(ia(qt_G),cv); bc.u_quant=ctypes.cast(ia(qt_U),cv); bc.d_quant=ctypes.cast(ia(qt_D),cv)
for pf in ['wG','wU','wD','nG','nU','nD','g_quant','u_quant','d_quant','wQK','qk_quant']:
    setattr(bc,pf,cv(0))
bc.emb=e.emb.ctypes.data_as(cv) if hasattr(e,'emb') and e.emb is not None else cv(0)
bc.onw=e._out_norm_w.ctypes.data_as(cv) if hasattr(e,'_out_norm_w') else cv(0)
bc.wOut=ctypes.cast(e._out_raw,cv)

# SSM state
ssm_state = np.zeros(L * e.ssm_groups * e.ssm_state_size, dtype=np.float32)
bc.ssm_state = ssm_state.ctypes.data_as(cv)

# Layer types
if LT:
    lt_arr=(ci*L)(*LT)
    bc.layer_types=ctypes.cast(lt_arr,cv)

# RoPE
max_ctx=4096; hd2=HD//2
freq = e.rope_freq_base ** (np.arange(0, HD, 2, dtype=np.float32) / HD)
ang = np.arange(max_ctx, dtype=np.float32).reshape(-1, 1) / freq.reshape(1, -1)
cos_all = np.cos(ang).astype(np.float32).reshape(-1)
sin_all = np.sin(ang).astype(np.float32).reshape(-1)
bc.cos_table=cos_all.ctypes.data_as(cv)
bc.sin_table=sin_all.ctypes.data_as(cv)
bc.max_ctx=max_ctx
lib.rope_init(bc.cos_table, bc.sin_table, ci(max_ctx), ci(HD), ci(HD))

# Workspace for KV temp in batch_forward
ws_bc_size = 2 * max_ctx * NKH * HD + 4096
ws_bc = np.zeros(ws_bc_size, dtype=np.float32)
bc.workspace = ws_bc.ctypes.data_as(cv)
bc.ws_size = ws_bc_size
print(f"BC workspace: {ws_bc_size} floats", flush=True)

# Shared buffers for B_MAX concurrent
ws = np.zeros(B_MAX * 12 * S, dtype=np.float32)
logits_buf = np.zeros(B_MAX * V, dtype=np.float32)

def mk_kv():
    kv = KVBlock()
    lib.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
    return kv

def sample_token(logits, temperature=0.0):
    safe = np.nan_to_num(logits, nan=-1e10, posinf=1e10, neginf=-1e10)
    if temperature <= 0: return int(np.argmax(safe))
    safe -= safe.max()
    p = np.exp(np.clip(safe / temperature, -50, 50))
    p[np.isnan(p)] = 0; s = p.sum()
    return int(np.random.choice(len(p), p=p/s)) if s > 0 else 0

# Tokenizer
print("Loading tokenizer...", flush=True)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
tokenizer.bos_token_id = BOS; tokenizer.eos_token_id = EOS
print(f"Tokenizer loaded, vocab={tokenizer.vocab_size}", flush=True)

# ── Scheduler ──
class Sequence:
    __slots__ = ('kv','tokens','gen_tokens','done','result','event','max_tokens','temperature','token_queue','stream')
    def __init__(self):
        self.kv = mk_kv()
        self.tokens = []; self.gen_tokens = []
        self.done = False; self.result = None
        self.event = threading.Event()
        self.max_tokens = 100; self.temperature = 0.0
        self.token_queue = None; self.stream = False

pending = queue.Queue()
active_slots = [None] * B_MAX
forward_lock = threading.Lock()

def run_scheduler():
    global bc, ws, logits_buf
    while True:
        # Fill slots
        while not pending.empty():
            try: seq = pending.get_nowait()
            except queue.Empty: break
            for i in range(B_MAX):
                if active_slots[i] is None:
                    active_slots[i] = seq; break
        
        batch = [(i, s) for i, s in enumerate(active_slots) if s is not None and not s.done]
        if not batch:
            time.sleep(0.002); continue
        
        B = len(batch)
        tokens_arr = (ci * B)()
        kv_ptrs = (cv * B)()
        for j, (slot, seq) in enumerate(batch):
            tokens_arr[j] = seq.tokens[-1] if seq.tokens else BOS
            kv_ptrs[j] = ctypes.cast(ctypes.pointer(seq.kv), cv)
        
        try:
            with forward_lock:
                bc.kv_array = ctypes.cast(kv_ptrs, cv)
                bc.logits = logits_buf.ctypes.data_as(cv)
                lib.batch_forward(ctypes.byref(bc), tokens_arr, ci(B), ws.ctypes.data_as(cv))
        except Exception as ex:
            print(f"SCHEDULER error: {ex}", flush=True); continue
        
        for j, (slot, seq) in enumerate(batch):
            logits = logits_buf[j*V:(j+1)*V].copy()
            next_tok = sample_token(logits, seq.temperature)
            seq.tokens.append(next_tok); seq.gen_tokens.append(next_tok)
            if seq.token_queue is not None:
                seq.token_queue.put(next_tok)
            if next_tok == EOS or len(seq.gen_tokens) >= seq.max_tokens:
                seq.done = True
                seq.result = {'tokens': seq.gen_tokens.copy(), 'finish_reason': 'stop' if next_tok == EOS else 'length'}
                if seq.token_queue: seq.token_queue.put(None)
                seq.event.set(); active_slots[slot] = None

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
print(f"Scheduler started (B_MAX={B_MAX})", flush=True)

# ── HTTP ──
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True; daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/health':
            active = sum(1 for s in active_slots if s is not None and not s.done)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps({'status':'ok','model':'qwen3.6-mxfp4','active':active,'max':B_MAX}).encode())
        else: self.send_error(404)
    
    def do_POST(self):
        if urlparse(self.path).path != '/v1/completions': self.send_error(404); return
        try:
            n = int(self.headers.get('Content-Length',0))
            body = json.loads(self.rfile.read(n)) if n else {}
            prompt = body.get('prompt',''); max_tokens = body.get('max_tokens',100)
            temperature = body.get('temperature',0.0); stream = body.get('stream',False)
            
            prompt_ids = tokenizer.encode(prompt) if isinstance(prompt,str) else prompt[:512]
            prompt_ids = [t for t in prompt_ids if t != BOS][:512]
            if not prompt_ids: prompt_ids = [BOS]
            
            t0 = time.perf_counter()
            seq = Sequence(); seq.max_tokens=max_tokens; seq.temperature=temperature; seq.stream=stream
            
            # Prefill token-by-token
            if prompt_ids:
                with forward_lock:
                    kva1=(cv*1)(ctypes.cast(ctypes.pointer(seq.kv),cv))
                    bc.kv_array=ctypes.cast(kva1,cv)
                    bc.logits=logits_buf[:V].ctypes.data_as(cv)
                    logits_buf[:V]=0
                    for pt in prompt_ids:
                        lib.batch_forward(ctypes.byref(bc),(ci*1)(pt),ci(1),ws.ctypes.data_as(cv))
            
            first_tok = sample_token(logits_buf[:V].copy(), temperature)
            seq.tokens=[first_tok]; seq.gen_tokens=[first_tok]
            
            if stream:
                seq.token_queue=queue.Queue()
                seq.token_queue.put(first_tok)
            
            pending.put(seq)
            
            if stream:
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream'); self.send_header('Cache-Control','no-cache')
                self.end_headers()
                while True:
                    tok=seq.token_queue.get()
                    if tok is None: break
                    self.wfile.write(f'data: {json.dumps({"choices":[{"text":tokenizer.decode([tok],skip_special_tokens=True),"index":0}]})}\n\n'.encode())
                    self.wfile.flush()
                self.wfile.write('data: [DONE]\n\n'.encode())
            else:
                if not seq.event.wait(timeout=600):
                    gen_tokens=seq.gen_tokens[:20]; finish='timeout'
                elif seq.result is None:
                    gen_tokens=seq.gen_tokens; finish='length' if len(gen_tokens)>=max_tokens else 'stop'
                else:
                    gen_tokens=seq.result['tokens']; finish=seq.result['finish_reason']
                t_total=time.perf_counter()-t0
                text=tokenizer.decode(gen_tokens,skip_special_tokens=True)
                self.send_response(200)
                self.send_header('Content-Type','application/json'); self.end_headers()
                self.wfile.write(json.dumps({
                    'choices':[{'text':text,'index':0,'finish_reason':finish}],
                    'usage':{'prompt_tokens':len(prompt_ids),'completion_tokens':len(gen_tokens),'total_tokens':len(prompt_ids)+len(gen_tokens)}
                }).encode())
                print(f"  [{len(prompt_ids)}p+{len(gen_tokens)}g {t_total:.1f}s {len(gen_tokens)/t_total:.0f}t/s]", flush=True)
        except Exception as ex:
            print(f"  ERROR: {ex}",flush=True)
            try: self.send_response(500); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(json.dumps({'error':str(ex)}).encode())
            except: pass
    
    def log_message(self,*a): pass

if __name__ == '__main__':
    print(f"Serving on 0.0.0.0:{PORT}", flush=True)
    print(f"  POST /v1/completions (B_MAX={B_MAX}, ~{B_MAX * 2.5:.0f} tok/s at saturation)", flush=True)
    print(f"  GET  /health", flush=True)
    server = ThreadedHTTPServer(('0.0.0.0', PORT), Handler)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nShutdown", flush=True)
