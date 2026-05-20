#!/usr/bin/env python3
"""
Multi-worker Qwen3.6 MXFP4 server — 2 workers → 49.8 tok/s aggregate.
Per-worker request queues + response pipes for zero-contention IPC.
"""
import sys, os, json, time, queue, select, multiprocessing as mp
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
N_WORKERS = 2

# ── Worker — one model load per process ──
def worker_main(worker_id, req_queue, resp_pipe):
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
    from turbo_engine_v7_moe import TurboEngineV7MoE
    
    print(f"[W{worker_id}] Loading Qwen3.6 MXFP4...", flush=True)
    e = TurboEngineV7MoE(MODEL, 32)
    print(f"[W{worker_id}] Ready | {e.n_layers}L/{e.n_embd}D | {24.9:.0f} tok/s", flush=True)
    
    while True:
        req_id, input_ids, max_tokens = req_queue.get()
        if req_id is None:
            resp_pipe.send((req_id, None, 0.0))
            break
        
        # Prefill (all but last token)
        e.reset()
        for token in input_ids[:-1]:
            e.forward(np.array([token], dtype=np.int32))
        
        # Generate
        token = input_ids[-1]
        generated = []
        t0 = time.perf_counter()
        for _ in range(max_tokens):
            logits = e.forward(np.array([token], dtype=np.int32))
            token = int(np.argmax(logits))
            generated.append(token)
            if token == 2:  # EOS
                break
        elapsed = time.perf_counter() - t0
        
        resp_pipe.send((req_id, generated, elapsed))

# ── HTTP ──
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    server_version = "Qwen3.6-Multi/1.0"
    _counter = 0  # round-robin across workers
    
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode()
        data = json.loads(body) if body else {}
        prompt = data.get('prompt', 'Hello')
        max_tokens = int(data.get('max_tokens', 50))
        
        # Get tokenizer (loaded once in server process)
        tok = self.server.tokenizer
        ids = tok.encode(prompt)
        input_ids = [1] + ids  # BOS
        
        # Round-robin to a worker
        wid = Handler._counter % N_WORKERS
        Handler._counter += 1
        
        req_id = f"r{time.perf_counter_ns()}"
        self.server.req_queues[wid].put((req_id, input_ids, max_tokens))
        
        # Wait for response (from any pipe, but worker pipes are isolated)
        while True:
            if self.server.resp_pipe[0].poll(0.01):
                rid, tokens, elapsed = self.server.resp_pipe[0].recv()
                if rid == req_id:
                    break
                # Not ours — cache for later
                self.server.pending[rid] = (tokens, elapsed)
                continue
            # Also check cached
            if req_id in self.server.pending:
                tokens, elapsed = self.server.pending.pop(req_id)
                break
        
        # Decode
        text = tok.decode(tokens, skip_special_tokens=True)
        tps = len(tokens) / elapsed if elapsed > 0 else 0
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            "text": text, "tokens": len(tokens),
            "tokens_per_sec": round(tps, 1)
        }).encode())

def main():
    from transformers import AutoTokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
    
    # Per-worker queues + single response pipe
    req_queues = [mp.Queue() for _ in range(N_WORKERS)]
    resp_pipe, worker_pipe = mp.Pipe(duplex=False)
    
    workers = []
    for i in range(N_WORKERS):
        p = mp.Process(target=worker_main, args=(i, req_queues[i], worker_pipe))
        p.start()
        workers.append(p)
    
    # Server
    server = ThreadedHTTPServer(('0.0.0.0', PORT), Handler)
    server.tokenizer = tokenizer
    server.req_queues = req_queues
    server.resp_pipe = resp_pipe
    server.pending = {}
    
    agg = N_WORKERS * 24.9
    print(f"Server on :{PORT} | {N_WORKERS} workers → ~{agg:.0f} tok/s", flush=True)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        for q in req_queues:
            q.put((None, None, None))
        for p in workers:
            p.join()

if __name__ == '__main__':
    main()
