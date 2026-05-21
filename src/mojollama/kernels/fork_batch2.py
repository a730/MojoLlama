#!/usr/bin/env python3
"""Fork-based batch — load model once, fork workers, measure agg throughput."""
import os, sys, time, json, multiprocessing, numpy as np
from multiprocessing import Process, Queue

os.environ['OMP_PROC_BIND'] = 'close'
os.environ['OMP_PLACES'] = 'cores'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

MODEL_PATH = None
ENGINE = None

def load_model(path):
    global MODEL_PATH, ENGINE
    from turbo_engine_v7_moe import TurboEngineV7MoE
    MODEL_PATH = path
    ENGINE = TurboEngineV7MoE(path, n_threads=32)
    return ENGINE

def worker_run(worker_id, n_tokens, result_queue, threads=16):
    """Forked worker: shares ENGINE via parent's mmap (COW)."""
    os.environ['OMP_NUM_THREADS'] = str(threads)
    from turbo_engine_v7_moe import TurboEngineV7MoE
    # ENGINE is inherited from parent via fork COW
    # But the C .so handles need to be re-acquired
    eng = ENGINE  # shared via fork!
    V = int(eng.vocab_size)
    
    # Warmup
    eng.reset()
    for i in range(3):
        eng.forward([i % V])
    
    # Benchmark 
    eng.reset()
    times = []
    for i in range(n_tokens):
        t0 = time.perf_counter()
        logits = eng.forward([i % V])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    ms_arr = np.array(times)
    result_queue.put({
        'worker_id': worker_id,
        'median_ms': float(np.median(ms_arr)),
        'mean_ms': float(np.mean(ms_arr)),
        'tok_s': 1000.0 / float(np.mean(ms_arr)),
    })

def run_bench(model_path, n_workers=2, threads_per_worker=16, n_tokens=10):
    # First, load model in parent
    print(f"Loading model ({threads_per_worker * n_workers} effective threads)...", flush=True)
    eng = load_model(model_path)
    V = int(eng.vocab_size)
    # Warmup
    eng.reset()
    for i in range(3):
        eng.forward([i % V])
    
    print(f"Forking {n_workers} workers...", flush=True)
    result_queue = Queue()
    workers = []
    t_start = time.perf_counter()
    
    for i in range(n_workers):
        p = Process(target=worker_run, args=(i, n_tokens, result_queue, threads_per_worker))
        p.start()
        workers.append(p)
    
    results = []
    for _ in workers:
        r = result_queue.get(timeout=120)
        results.append(r)
        print(f"  Worker {r['worker_id']}: {r['tok_s']:.1f} tok/s  median={r['median_ms']:.1f}ms", flush=True)
    
    t_elapsed = time.perf_counter() - t_start
    for p in workers: p.join(timeout=5)
    
    tok_rates = [r['tok_s'] for r in results]
    total_tokens = sum([n_tokens for _ in results])
    agg = total_tokens / t_elapsed if t_elapsed > 0 else 0
    
    print(f"\n{'='*55}")
    print(f"  Aggregate: {agg:.1f} tok/s ({n_workers}w×{threads_per_worker}t)")
    print(f"  Per-worker avg: {np.mean(tok_rates):.1f} tok/s")
    print(f"  llama.cpp single-seq: ~27 tok/s")
    print(f"  Ratio: {agg/27:.2f}x" if agg > 0 else "")
    print(f"{'='*55}")
    return agg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-w', '--workers', type=int, default=2)
    parser.add_argument('-t', '--threads', type=int, default=16)
    parser.add_argument('-n', '--n-tokens', type=int, default=10)
    args = parser.parse_args()
    print(f"\nMojoLlama Fork Batch Benchmark")
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Workers: {args.workers} × {args.threads}t = {args.workers*args.threads} total")
    run_bench(args.model, args.workers, args.threads, args.n_tokens)
