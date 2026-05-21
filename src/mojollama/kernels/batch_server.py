#!/usr/bin/env python3
"""MojoLlama Multi-Worker Batch Server.

Spawns N worker processes with K OMP threads each.
Routes concurrent inference requests across workers round-robin.
Achieves aggregate throughput > single-llama.cpp by utilizing all CPU cores
with independent model copies (each reading weights from shared mmap'd files).

Optimal config for Threadripper 3970X (32C/64T): 8-10 workers × 3 threads.
"""
import os, sys, time, json, socket, multiprocessing, queue
import numpy as np
from multiprocessing import Process, Queue, Value, Lock
from ctypes import c_bool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORKER_OMP_THREADS = 3
NUM_WORKERS = 10
MODEL_PATH = None

def worker_main(worker_id, model_path, request_queue, response_queue, stop_flag):
    """Worker process: loads model copy and serves inference requests."""
    os.environ["OMP_NUM_THREADS"] = str(WORKER_OMP_THREADS)
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    
    # Delay import to avoid forking issues with C extensions
    from turbo_engine_v7_moe import TurboEngineV7MoE
    
    try:
        eng = TurboEngineV7MoE(model_path, n_threads=WORKER_OMP_THREADS)
    except Exception as e:
        response_queue.put({
            "worker_id": worker_id,
            "error": f"Engine load failed: {e}",
            "done": True
        })
        return
    
    V = int(eng.vocab_size)
    response_queue.put({"worker_id": worker_id, "ready": True})
    
    while not stop_flag.value:
        try:
            req = request_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if req is None:
            break
        
        req_id = req.get("id", 0)
        tokens = req.get("tokens", [0])
        n_gen = req.get("n_gen", 1)
        
        eng.reset()
        times = []
        for i, tok in enumerate(tokens):
            t0 = time.perf_counter()
            _, logits = eng.forward([tok])
            t1 = time.perf_counter()
            if i > 0:  # skip prompt processing (first token)
                times.append((t1 - t0) * 1000)
        
        result = {
            "worker_id": worker_id,
            "req_id": req_id,
            "n_tokens": len(tokens),
            "ms_per_token": np.mean(times) if times else 0,
            "tok_s": 1000 / np.mean(times) if times else 0,
            "max_logit": float(np.max(logits)),
        }
        response_queue.put(result)
    
    response_queue.put({"worker_id": worker_id, "done": True})


def run_benchmark(model_path, num_workers=10, threads_per_worker=3, 
                  tokens_per_request=10, num_requests=50):
    """Run batch benchmark with multiple workers."""
    global WORKER_OMP_THREADS, NUM_WORKERS
    WORKER_OMP_THREADS = threads_per_worker
    NUM_WORKERS = num_workers
    
    request_queue = Queue()
    response_queue = Queue()
    stop_flag = Value(c_bool, False)
    
    workers = []
    for i in range(num_workers):
        p = Process(target=worker_main, args=(
            i, model_path, request_queue, response_queue, stop_flag))
        p.start()
        workers.append(p)
    
    # Wait for all workers to be ready
    ready_count = 0
    while ready_count < num_workers:
        resp = response_queue.get()
        if resp.get("ready"):
            ready_count += 1
            print(f"  Worker {resp['worker_id']} ready ({ready_count}/{num_workers})", flush=True)
    
    # Generate tokens for each request (same tokens for all)
    test_tokens = list(range(tokens_per_request))
    
    # Benchmark: send requests round-robin
    print(f"\n  Running {num_requests} requests × {tokens_per_request} tokens across {num_workers} workers...", flush=True)
    t_start = time.perf_counter()
    
    for i in range(num_requests):
        request_queue.put({
            "id": i,
            "tokens": test_tokens,
            "n_gen": tokens_per_request,
        })
    
    # Collect results
    results = []
    for i in range(num_requests):
        resp = response_queue.get()
        results.append(resp)
    
    t_elapsed = time.perf_counter() - t_start
    
    # Signal workers to stop
    stop_flag.value = True
    for _ in range(num_workers):
        request_queue.put(None)
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    # Analyze results
    tok_rates = [r["tok_s"] for r in results if r.get("tok_s", 0) > 0]
    total_tokens = sum(r["n_tokens"] for r in results if "n_tokens" in r)
    
    agg_tok_s = total_tokens / t_elapsed if t_elapsed > 0 else 0
    avg_per_worker = np.mean(tok_rates) if tok_rates else 0
    
    print(f"  Elapsed: {t_elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Aggregate throughput: {agg_tok_s:.1f} tok/s")
    print(f"  Avg per worker: {avg_per_worker:.1f} tok/s")
    print(f"  Best worker: {max(tok_rates):.1f} tok/s" if tok_rates else "")
    
    return {
        "model": os.path.basename(model_path),
        "workers": num_workers,
        "threads_per_worker": threads_per_worker,
        "total_threads": num_workers * threads_per_worker,
        "total_tokens": total_tokens,
        "elapsed_s": round(t_elapsed, 2),
        "aggregate_tok_s": round(agg_tok_s, 1),
        "avg_per_worker_tok_s": round(avg_per_worker, 1),
        "per_request": tok_rates,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-worker batch server")
    parser.add_argument("-m", "--model", required=True, help="GGUF model path")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of workers")
    parser.add_argument("-t", "--threads", type=int, default=3, help="OMP threads per worker")
    parser.add_argument("-n", "--n-tokens", type=int, default=10, help="Tokens per request")
    parser.add_argument("-r", "--n-requests", type=int, default=50, help="Number of requests")
    args = parser.parse_args()
    
    print(f"MojoLlama Batch Server")
    print(f"  Model: {args.model}")
    print(f"  Workers: {args.workers} × {args.threads}t = {args.workers*args.threads} total threads")
    print(f"  Tokens/request: {args.n_tokens}")
    print(f"  Requests: {args.n_requests}")
    
    result = run_benchmark(
        args.model, args.workers, args.threads,
        args.n_tokens, args.n_requests)
    
    print(f"\n{'='*55}")
    print(f"  Result: {result['aggregate_tok_s']} tok/s aggregate")
    print(f"  Per worker avg: {result['avg_per_worker_tok_s']} tok/s")
    print(f"{'='*55}")
    
    # Save result
    result_path = f"/tmp/batch_result_{os.path.basename(args.model)}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {result_path}")
