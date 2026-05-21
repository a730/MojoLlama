#!/usr/bin/env python3
"""MojoLlama Concurrency Benchmark — CORRECT aggregate measurement.
Loads model FIRST (not timed), then synchronizes inference across workers.
"""
import os, sys, time, json, numpy as np, multiprocessing as mp

os.environ['OMP_PROC_BIND'] = 'close'; os.environ['OMP_PLACES'] = 'cores'
KERNEL_DIR = '/onedev-workspace/work/src/mojollama/kernels'
sys.path.insert(0, KERNEL_DIR); sys.path.insert(0, '/onedev-workspace/work/src/mojollama')
TOTAL_THREADS = 32

LLAMA_CPP = {  # tok/s, 512-prompt + 128-gen, 32t
    'gpt-oss-20b-Q4_K_M': 27.69,
    'Qwen3-30B-A3B-Instruct-2507-Q4_K_M': 27.89,
    'qwen3.6-35b-a3b-q4_k_m': 16.36,
}

def worker(worker_id, model_path, n_tokens, result_queue, threads, start_event):
    """Worker: load model, wait for start signal, generate tokens, report."""
    os.environ['OMP_NUM_THREADS'] = str(threads)
    from turbo_engine_v7_moe import TurboEngineV7MoE
    eng = TurboEngineV7MoE(model_path, n_threads=threads)
    V = int(eng.vocab_size)
    eng.reset()
    for i in range(3):
        eng.forward([i % V])
    # Signal ready, wait for start
    result_queue.put({'worker': worker_id, 'ready': True})
    start_event.wait()
    # Bench
    eng.reset()
    t0 = time.perf_counter()
    for i in range(n_tokens):
        eng.forward([i % V])
    elapsed = time.perf_counter() - t0
    tok_s = n_tokens / elapsed if elapsed > 0 else 0
    result_queue.put({'worker': worker_id, 'tok_s': tok_s, 'elapsed': elapsed, 'n': n_tokens})

def bench(model_path, concurrency=4, tokens=10):
    tpw = max(1, TOTAL_THREADS // concurrency)
    result_queue = mp.Queue()
    start_event = mp.Event()
    workers = []
    
    # Load workers sequentially
    for i in range(concurrency):
        p = mp.Process(target=worker, args=(i, model_path, tokens, result_queue, tpw, start_event))
        p.start()
        workers.append(p)
    
    # Wait for all to be ready
    for _ in workers:
        result_queue.get(timeout=300)
    
    # Signal start and measure wall time
    t_start = time.perf_counter()
    start_event.set()
    
    total_tokens = 0; per_worker = []
    for _ in workers:
        r = result_queue.get(timeout=300)
        per_worker.append(r)
        total_tokens += r['n']
    
    wall = time.perf_counter() - t_start
    for p in workers: p.join(timeout=5)
    
    agg = total_tokens / wall if wall > 0 else 0
    avg = np.mean([w['tok_s'] for w in per_worker]) if per_worker else 0
    return {'concurrency': concurrency, 'tpw': tpw, 'agg': round(agg, 1), 'avg': round(avg, 1), 'wall': round(wall, 2), 'total': total_tokens}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-c', '--concurrency', type=str, default='1,2,4,8')
    parser.add_argument('-n', '--tokens', type=int, default=10)
    args = parser.parse_args()
    
    levels = [int(x) for x in args.concurrency.split(',')]
    mn = os.path.basename(args.model).replace('.gguf', '')
    ref = LLAMA_CPP.get(mn, '?')
    
    print(f"{'='*55}")
    print(f"  MojoLlama — King of Concurrency")
    print(f"  Model: {mn}")
    print(f"  llama.cpp: {ref} tok/s (single-seq)")
    print(f"{'='*55}")
    print(f"  {'Conc':>5} | {'Thrds':>6} | {'Agg':>8} | {'Per':>6} | {'Wall':>7} | {'Ratio':>7}")
    print(f"  {'─'*5} | {'─'*6} | {'─'*8} | {'─'*6} | {'─'*7} | {'─'*7}")
    
    for c in levels:
        r = bench(args.model, c, args.tokens)
        ratio = r['agg'] / ref if isinstance(ref, (int, float)) and ref > 0 else 0
        w = "✅" if ratio > 1.0 else ""
        print(f"  {r['concurrency']:>5} | {r['tpw']:>4d}t | {r['agg']:>7.1f} | {r['avg']:>5.1f} | {r['wall']:>6.1f}s | {ratio:.2f}x {w}")
    
    ref2 = ref if isinstance(ref, (int, float)) else 0
    print(f"  {'─'*45}")
    print(f"  🏆 MojoLlama KING OF CONCURRENCY" if any(bench(args.model, c, args.tokens)['agg']/ref2 > 1.0 for c in levels) else f"  Not yet king")
    print(f"{'='*55}")
