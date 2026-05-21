#!/usr/bin/env python3
"""Fork-based batch server — load model once, fork workers, measure agg throughput."""
import os, sys, time, json, multiprocessing, numpy as np
from ctypes import c_int, c_float, c_uint8, POINTER, CDLL

os.environ['OMP_PROC_BIND'] = 'close'
os.environ['OMP_PLACES'] = 'cores'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

def worker_run(worker_id, model_path, n_tokens, result_queue):
    """Worker process: load engine, generate tokens, report tok/s."""
    os.environ['OMP_NUM_THREADS'] = os.environ.get('MOJO_WORKER_THREADS', '3')
    from turbo_engine_v7_moe import TurboEngineV7MoE
    
    eng = TurboEngineV7MoE(model_path, n_threads=int(os.environ.get('MOJO_WORKER_THREADS', '3')))
    V = int(eng.vocab_size)
    eng.reset()
    
    # Warmup
    for i in range(5):
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
        'n_tokens': n_tokens,
        'median_ms': float(np.median(ms_arr)),
        'mean_ms': float(np.mean(ms_arr)),
        'tok_s': 1000.0 / float(np.mean(ms_arr)),
        'p95_ms': float(np.percentile(ms_arr, 95)),
    })

def run_bench(model_path, n_workers=4, threads_per_worker=3, n_tokens_per_worker=10):
    workers = []
    result_queue = multiprocessing.Queue()
    
    print(f"Spawning {n_workers} workers × {threads_per_worker}t each...", flush=True)
    os.environ['MOJO_WORKER_THREADS'] = str(threads_per_worker)
    
    t_start = time.perf_counter()
    for i in range(n_workers):
        p = multiprocessing.Process(target=worker_run, 
            args=(i, model_path, n_tokens_per_worker, result_queue))
        p.start()
        workers.append(p)
    
    results = []
    for _ in workers:
        r = result_queue.get()
        results.append(r)
        print(f"  Worker {r['worker_id']}: {r['tok_s']:.1f} tok/s  median={r['median_ms']:.1f}ms", flush=True)
    
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    t_elapsed = time.perf_counter() - t_start
    
    tok_rates = [r['tok_s'] for r in results]
    total_tokens = sum(r['n_tokens'] for r in results)
    
    agg = total_tokens / t_elapsed if t_elapsed > 0 else 0
    avg = np.mean(tok_rates) if tok_rates else 0
    best = max(tok_rates) if tok_rates else 0
    
    print(f"\n{'='*55}")
    print(f"  Aggregate: {agg:.1f} tok/s  ({n_workers} workers × {threads_per_worker}t)")
    print(f"  Avg/worker: {avg:.1f} tok/s  Best: {best:.1f} tok/s")
    print(f"  Total tokens: {total_tokens} in {t_elapsed:.2f}s")
    print(f"{'='*55}")
    
    return {
        'model': os.path.basename(model_path),
        'n_workers': n_workers,
        'threads_per_worker': threads_per_worker,
        'total_threads': n_workers * threads_per_worker,
        'aggregate_tok_s': round(agg, 1),
        'avg_per_worker_tok_s': round(avg, 1),
        'elapsed_s': round(t_elapsed, 2),
        'total_tokens': total_tokens,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-t', '--threads', type=int, default=3)
    parser.add_argument('-n', '--n-tokens', type=int, default=10)
    args = parser.parse_args()
    
    print(f"MojoLlama Fork Batch Server")
    print(f"  Model: {args.model}")
    r = run_bench(args.model, args.workers, args.threads, args.n_tokens)
    print(json.dumps(r, indent=2))
