#!/usr/bin/env python3
"""Continuous batching server for MoE models using PagedAttention KV cache.
Adapted from server_batch.py (dense) and server_moe.py (single-request MoE).
Uses TurboEngineV7MoE + cengine_batch_instr.so with batch_forward(B≥1).
"""
import sys, os, json, time, threading, queue, ctypes
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from socketserver import ThreadingMixIn
from transformers import AutoTokenizer

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else '/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
B_MAX = 4  # maximum concurrent sequences in a batch

# ── Engine ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kernels'))
from turbo_engine_v7_moe import TurboEngineV7MoE
e = TurboEngineV7MoE(MODEL_PATH, 32)
L=e.n_layers; N=e.n_embd; NH=e.n_head; NKH=e.n_kv_head; HD=e.head_dim; FF=e.n_ff; V=e.vocab_size
NE=e.n_experts; NK=e.n_experts_per_tok; moe_int=e.n_ff_expert
S = max(N, NH*HD, FF, NKH*HD, moe_int); BOS=1; EOS=2
print(f"Engine: {L}L/{N}D/{FF}FF/{NH}H/{NKH}KV | MoE {NE}×{NK} | V={V} | B_MAX={B_MAX}", flush=True)

# ── C library ──
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kernels', 'cengine_batch_instr.so'))
cv=ctypes.c_void_p; ci=ctypes.c_int; cf=ctypes.c_float

# ── PagedAttention-backed KVBlock (Python side matches C struct) ──
class PageTable(ctypes.Structure):
    _fields_ = [
        ('page_size', ci), ('n_pages', ci), ('pages', cv), ('table', cv),
        ('free_pages', cv), ('free_count', ci), ('_nkh', ci), ('_hd', ci), ('_n_layers', ci),
    ]

class KVBlock(ctypes.Structure):
    _fields_ = [
        ('k', cv), ('v', cv), ('n_blocks', ci),
        ('seq_len', ci * 64), ('block_map', (ci * 1024) * 64),
        ('pt', PageTable),
    ]

class BC(ctypes.Structure):
    _fields_ = [
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
    ]

lib.kv_init.argtypes=[cv,ci,ci,ci]; lib.kv_init.restype=None
lib.batch_forward.argtypes=[cv,cv,ci,cv]; lib.batch_forward.restype=None

# ── Build per-layer pointer arrays (from server_moe.py) ──
def wa(arr): return (cv*L)(*[ctypes.cast(a,cv) for a in arr])
def ia(arr): return (ci*L)(*[int(a) for a in arr])

wQ_arr=[0]*L; wK_arr=[0]*L; wV_arr=[0]*L; wO_arr=[0]*L
wG_arr=[0]*L; wU_arr=[0]*L; wD_arr=[0]*L
wAN_arr=[0]*L; wFN_arr=[0]*L
wQK_arr=[0]*L; qkQ_arr=[0]*L
nQ_arr=[0]*L; nK_arr=[0]*L; nV_arr=[0]*L; nO_arr=[0]*L; nG_arr=[0]*L; nU_arr=[0]*L; nD_arr=[0]*L
qQ_arr=[0]*L; qK_arr=[0]*L; qV_arr=[0]*L; qO_arr=[0]*L; qG_arr=[0]*L; qU_arr=[0]*L; qD_arr=[0]*L

for i,lw in enumerate(e._layers):
    wQ_arr[i]=lw.attn_q_raw; nQ_arr[i]=lw.attn_q_nr.value; qQ_arr[i]=lw.attn_q_qt.value
    wK_arr[i]=lw.attn_k_raw; nK_arr[i]=lw.attn_k_nr.value; qK_arr[i]=lw.attn_k_qt.value
    wV_arr[i]=lw.attn_v_raw; nV_arr[i]=lw.attn_v_nr.value; qV_arr[i]=lw.attn_v_qt.value
    wO_arr[i]=lw.attn_out_raw; nO_arr[i]=lw.attn_out_nr.value; qO_arr[i]=lw.attn_out_qt.value
    wAN_arr[i]=lw.attn_norm_w.ctypes.data_as(cv); wFN_arr[i]=lw.ffn_norm_w.ctypes.data_as(cv)
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
        gate_raw_t = me.gate_raw
        up_raw_t = me.up_raw
        down_raw_t = me.down_raw
        if gate_raw_t and len(gate_raw_t) > 0:
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

if e.is_moe and len(e._moe_layers) > 0:
    me = e._moe_layers[0]
    gate_qt = me.gate_qt.value; up_qt = me.up_qt.value; down_qt = me.down_qt.value
else:
    gate_qt = 2; up_qt = 2; down_qt = 2
print(f"Expert quant types: gate={gate_qt} up={up_qt} down={down_qt}", flush=True)

# ── Build BC struct ──
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
bc.wOut=ctypes.cast(e._out_raw,cv) if hasattr(e,'_out_raw') else cv(0)

# RoPE precompute
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

# ── Shared buffers ──
# Allocate enough workspace for B_MAX concurrent sequences
ws = np.zeros(B_MAX * 12 * S, dtype=np.float32)
logits_buf = np.zeros(B_MAX * V, dtype=np.float32)

def mk_kv():
    kv = KVBlock()
    lib.kv_init(ctypes.byref(kv), ci(L), ci(NKH), ci(HD))
    return kv

def sample_token(logits, temperature=0.0):
    safe = np.nan_to_num(logits, nan=-1e10, posinf=1e10, neginf=-1e10)
    if temperature <= 0:
        return int(np.argmax(safe))
    safe -= safe.max()
    p = np.exp(np.clip(safe / temperature, -50, 50))
    p[np.isnan(p)] = 0
    s = p.sum()
    return int(np.random.choice(len(p), p=p/s)) if s > 0 else 0

# ── Tokenizer ──
print(f"Loading tokenizer from /tmp/qwen3-tokenizer/...", flush=True)
tokenizer = AutoTokenizer.from_pretrained('/tmp/qwen3-tokenizer/')
tokenizer.bos_token_id = BOS
tokenizer.eos_token_id = EOS
print(f"Tokenizer loaded, vocab={tokenizer.vocab_size}", flush=True)

# ── Sequence state ──
class Sequence:
    __slots__ = ('kv','tokens','gen_tokens','done','result','event','max_tokens','temperature','token_queue','stream')
    def __init__(self):
        self.kv = mk_kv()
        self.tokens = []
        self.gen_tokens = []
        self.done = False
        self.result = None
        self.event = threading.Event()
        self.max_tokens = 100
        self.temperature = 0.0
        self.token_queue = None
        self.stream = False

# ── Scheduler ──
pending = queue.Queue()
active_slots = [None] * B_MAX
forward_lock = threading.Lock()

def run_scheduler():
    global bc, ws, logits_buf, active_slots
    while True:
        # Fill empty slots from pending queue
        while not pending.empty():
            try:
                seq = pending.get_nowait()
            except queue.Empty:
                break
            for i in range(B_MAX):
                if active_slots[i] is None:
                    active_slots[i] = seq
                    break

        # Collect active non-done sequences
        batch = [(i, s) for i, s in enumerate(active_slots) if s is not None and not s.done]
        if not batch:
            time.sleep(0.002)
            continue

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
            print(f"SCHEDULER forward error: {ex}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        for j, (slot, seq) in enumerate(batch):
            logits = logits_buf[j*V:(j+1)*V].copy()
            next_tok = sample_token(logits, seq.temperature)
            seq.tokens.append(next_tok)
            seq.gen_tokens.append(next_tok)

            # Push token to streaming queue if active
            if seq.token_queue is not None:
                seq.token_queue.put(next_tok)

            if next_tok == EOS or len(seq.gen_tokens) >= seq.max_tokens:
                seq.done = True
                seq.result = {
                    'tokens': seq.gen_tokens.copy(),
                    'finish_reason': 'stop' if next_tok == EOS else 'length',
                }
                if seq.token_queue is not None:
                    seq.token_queue.put(None)  # signal done
                seq.event.set()
                active_slots[slot] = None

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
print(f"Scheduler started (B_MAX={B_MAX})", flush=True)

# ── HTTP Server ──
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/health':
            # Count active sequences
            active_count = sum(1 for s in active_slots if s is not None and not s.done)
            pending_count = pending.qsize()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'ok',
                'model': 'mojollama-moe',
                'active_sequences': active_count,
                'pending_requests': pending_count,
                'max_concurrent': B_MAX,
            }).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != '/v1/completions':
            self.send_error(404)
            return

        try:
            n = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(n)) if n else {}
            prompt = body.get('prompt', '')
            max_tokens = body.get('max_tokens', 100)
            temperature = body.get('temperature', 0.0)
            stream = body.get('stream', False)

            # Tokenize
            prompt_ids = tokenizer.encode(prompt) if isinstance(prompt, str) else prompt[:512]
            prompt_ids = [t for t in prompt_ids if t != BOS]  # strip BOS
            # Filter out-of-range tokens
            prompt_ids = [t for t in prompt_ids if 0 <= t < V][:512]
            if not prompt_ids:
                prompt_ids = [BOS]

            t0 = time.perf_counter()

            # Create sequence
            seq = Sequence()
            seq.max_tokens = max_tokens
            seq.temperature = temperature
            seq.stream = stream

            # Prefill: token-by-token (MoE engine doesn't have prefill_forward)
            if prompt_ids:
                kva1 = (cv * 1)(ctypes.cast(ctypes.pointer(seq.kv), cv))
                with forward_lock:
                    bc.kv_array = ctypes.cast(kva1, cv)
                    bc.logits = logits_buf[0:V].ctypes.data_as(cv)
                    logits_buf[:V] = 0
                    for pt in prompt_ids:
                        lib.batch_forward(ctypes.byref(bc), (ci * 1)(pt), ci(1), ws.ctypes.data_as(cv))

            # Sample first generated token
            first_tok = sample_token(logits_buf[:V].copy(), temperature)
            seq.tokens = [first_tok]
            seq.gen_tokens = [first_tok]

            # Streaming mode: send token-by-token via SSE
            if stream:
                seq.token_queue = queue.Queue()
                seq.token_queue.put(first_tok)

            # Enqueue for generation
            pending.put(seq)

            if stream:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()

                while True:
                    tok = seq.token_queue.get()
                    if tok is None:
                        break
                    chunk_data = json.dumps({
                        'choices': [{
                            'text': tokenizer.decode([tok], skip_special_tokens=True),
                            'index': 0,
                        }]
                    })
                    self.wfile.write(f'data: {chunk_data}\n\n'.encode())
                    self.wfile.flush()

                self.wfile.write('data: [DONE]\n\n'.encode())
                t_total = time.perf_counter() - t0
                tok_s = len(seq.gen_tokens) / t_total if t_total > 0 else 0
                generated_text = tokenizer.decode(seq.gen_tokens, skip_special_tokens=True)
                print(f"  [{len(prompt_ids)}p+{len(seq.gen_tokens)}g {t_total:.1f}s {tok_s:.0f}t/s] '{generated_text[:60]}'", flush=True)
            else:
                # Non-streaming: wait for completion
                if not seq.event.wait(timeout=600):
                    gen_tokens = seq.gen_tokens[:20]
                    finish = 'timeout'
                elif seq.result is None:
                    gen_tokens = seq.gen_tokens
                    finish = 'length' if len(gen_tokens) >= max_tokens else 'stop'
                else:
                    gen_tokens = seq.result['tokens']
                    finish = seq.result['finish_reason']

                t_total = time.perf_counter() - t0
                generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                res = {
                    'id': f'cmpl-{int(time.time())}',
                    'object': 'text_completion',
                    'model': 'mojollama-moe-batch',
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

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode())

                tok_s = len(gen_tokens) / t_total if t_total > 0 else 0
                print(f"  [{len(prompt_ids)}p+{len(gen_tokens)}g {t_total:.1f}s {tok_s:.0f}t/s] '{generated_text[:60]}'", flush=True)

        except Exception as ex:
            print(f"  [handler] EXCEPTION: {ex}", flush=True)
            import traceback
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
    print(f"Serving on 0.0.0.0:{PORT}", flush=True)
    print(f"  POST /v1/completions (streaming + non-streaming)", flush=True)
    print(f"  GET  /health", flush=True)
    server = ThreadedHTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown", flush=True)
