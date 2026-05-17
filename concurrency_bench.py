"""Mojo Hybrid Serve — Multi-worker concurrent server + batch benchmark.
Tests: multi-process, batched forward, combined.
"""
import argparse, json, os, sys, time, gc, glob
import numpy as np
import multiprocessing as mp
import threading
from pathlib import Path
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.mojollama.model.inference import LLMInference


# ─── Multi-Worker Benchmark ──────────────────────────────────────────────────

def worker_process(model_path: str, request_queue, result_queue, worker_id: int):
    """Worker process that handles requests sequentially with KV cache."""
    os.environ['MOJOLLAMA_DEVICE'] = 'cpu'
    model = LLMInference(model_path)
    
    while True:
        try:
            req_id, prompt, max_tokens = request_queue.get(timeout=2)
        except Empty:
            break
        
        t0 = time.time()
        try:
            result = model.generate(prompt, max_tokens=max_tokens)
            elapsed = time.time() - t0
            result_queue.put((req_id, result, elapsed))
        except Exception as e:
            result_queue.put((req_id, f"[ERROR: {e}]", time.time() - t0))
        
        # Reset KV cache between requests
        model._kv_cache = None
        model._cached_len = 0
        gc.collect()


def bench_multiworker(model_path: str, num_workers: list[int] = None):
    """Benchmark multi-worker throughput."""
    if num_workers is None:
        num_workers = [1, 2, 4, 8, 16, 32]
    
    prompt = "The capital of France is"
    n_requests = 32  # total requests to distribute
    gen_tokens = 10
    
    print("=" * 70)
    print("MojoLlama — Multi-Worker Benchmark")
    print(f"Model: {model_path}")
    print(f"Requests: {n_requests} × {gen_tokens} tokens each")
    print(f"Prompt: '{prompt}'")
    print(f"CPU cores: {os.cpu_count()}")
    print("=" * 70)
    
    for nw in num_workers:
        if nw > n_requests:
            continue
        
        request_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Enqueue requests
        for i in range(n_requests):
            request_queue.put((i, prompt, gen_tokens))
        
        # Start workers
        t0 = time.time()
        workers = []
        for w in range(nw):
            p = mp.Process(target=worker_process, 
                           args=(model_path, request_queue, result_queue, w))
            p.start()
            workers.append(p)
        
        # Collect results
        results = []
        completed = 0
        timeouts = 0
        while completed < n_requests and timeouts < 5:
            try:
                rid, text, elapsed = result_queue.get(timeout=10)
                results.append((rid, elapsed))
                completed += 1
                timeouts = 0
            except Empty:
                timeouts += 1
        
        total_time = time.time() - t0
        
        # Wait for workers
        for p in workers:
            p.terminate()
            p.join(timeout=2)
        
        if results:
            times = [r[1] for r in results]
            avg_time = sum(times) / len(times)
            total_tokens = len(results) * gen_tokens
            throughput = total_tokens / total_time if total_time > 0 else 0
            req_s = len(results) / total_time if total_time > 0 else 0
            
            print(f"  workers={nw:2d} | total={total_time:.1f}s | "
                  f"avg_req={avg_time:.1f}s | throughput={throughput:.1f} tok/s | "
                  f"{req_s:.1f} req/s")
        else:
            print(f"  workers={nw:2d} | FAILED (no results)")
    
    # llama.cpp comparison
    print(f"\n  llama.cpp 10-concurrent: 114 tok/s, 14.3 req/s")
    print(f"  Target: match or exceed at higher worker counts")


# ─── Continuous Batching Server ──────────────────────────────────────────────

@dataclass
class Request:
    prompt: str
    max_tokens: int = 50
    result: list = field(default_factory=list)
    event: threading.Event = field(default_factory=threading.Event)


class BatchServer:
    """Multi-worker server with request queue."""
    
    def __init__(self, model_path: str, num_workers: int = 4):
        self.model_path = model_path
        self.num_workers = num_workers
        self.mp_requests = mp.Queue()
        self.mp_results = mp.Queue()
        self.workers = []
        self.running = True
        self._start_workers()
    
    def _start_workers(self):
        for w in range(self.num_workers):
            p = mp.Process(target=worker_process,
                           args=(self.model_path, self.mp_requests, 
                                 self.mp_results, w))
            p.start()
            self.workers.append(p)
        print(f"[BatchServer] {self.num_workers} workers started")
    
    def submit(self, prompt: str, max_tokens: int = 50) -> str:
        req_id = id(prompt) + int(time.time() * 1000)
        self.mp_requests.put((req_id, prompt, max_tokens))
        # Wait for result with timeout
        deadline = time.time() + max_tokens * 60 + 30  # generous timeout
        while time.time() < deadline:
            try:
                rid, text, elapsed = self.mp_results.get(timeout=1)
                if rid == req_id:
                    return text
                # Wrong result, put back
                self.mp_results.put((rid, text, elapsed))
            except Empty:
                continue
        return "[timeout]"
    
    def submit_batch(self, prompts: list[str], max_tokens: int = 50) -> list[str]:
        """Submit multiple prompts and wait for all results."""
        ids = [(id(p) + int(time.time() * 1000) + i) for i, p in enumerate(prompts)]
        for i, p in enumerate(prompts):
            self.mp_requests.put((ids[i], p, max_tokens))
        
        results = {}
        deadline = time.time() + max_tokens * 60 + 60
        while len(results) < len(prompts) and time.time() < deadline:
            try:
                rid, text, elapsed = self.mp_results.get(timeout=1)
                if rid in ids:
                    results[rid] = text
                else:
                    self.mp_results.put((rid, text, elapsed))
            except Empty:
                continue
        
        return [results.get(rid, "[timeout]") for rid in ids]
    
    def shutdown(self):
        self.running = False
        for p in self.workers:
            p.terminate()
            p.join(timeout=3)


# ─── Mixed Benchmark ─────────────────────────────────────────────────────────

def bench_continuous(model_path: str):
    """Benchmark the continuous batching server with concurrent users."""
    import threading as th
    
    print("=" * 70)
    print("MojoLlama — Concurrent User Benchmark")
    print("=" * 70)
    
    for n_workers in [2, 4, 8]:
        server = BatchServer(model_path, num_workers=n_workers)
        time.sleep(5)  # Wait for workers to start
        
        n_users = min(n_workers * 2, 16)
        prompts = ["What is the capital of France?"] * n_users
        
        t0 = time.time()
        threads = []
        results = []
        lock = th.Lock()
        
        def user_task(prompt):
            r = server.submit(prompt, max_tokens=10)
            with lock:
                results.append(r)
        
        for p in prompts:
            t = th.Thread(target=user_task, args=(p,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        total = time.time() - t0
        server.shutdown()
        
        tok_count = sum(len(r.split()) for r in results if r and not r.startswith('['))
        print(f"  workers={n_workers}, users={n_users}: {total:.1f}s, "
              f"{tok_count/total:.1f} tok/s, {n_users/total:.1f} req/s")
        time.sleep(2)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.environ.get('MODEL_PATH', ''))
    parser.add_argument('--bench', choices=['multi', 'batch', 'all'], default='all')
    args = parser.parse_args()
    
    model_path = args.model
    if not model_path:
        files = glob.glob('*.gguf')
        model_path = files[0] if files else 'Llama-3.2-1B-Instruct-Q4_0.gguf'
    
    if args.bench in ('multi', 'all'):
        bench_multiworker(model_path)
    if args.bench in ('all',):
        bench_continuous(model_path)
