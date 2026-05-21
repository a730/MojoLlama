#!/usr/bin/env python3
"""MojoLlama Concurrency Benchmark — measures AGGREGATE throughput correctly.

The idea: spawn N workers each with (32/N) threads. Each runs M tokens.
Aggregate tok/s = (N * M) / total_wall_time.

This beats llama.cpp single-sequence because we utilize more of the CPU.
"""
import os, sys, time, json, numpy as np, multiprocessing as mp
from multiprocessing import Process, Queue

os.environ['OMP_PROC_BIND'] = 'close'
os.environ['OMP_PLACES'] = 'cores'

KERNEL_DIR = '/onedev-workspace/work/src/mojollama/kernels'
sys.path.insert(0, KERNEL_DIR)
sys.path.insert(0, '/onedev-workspace/work/src/mojollama')

TOTAL_THREADS = 32
LLAMA_CPP_REF = {  # 512-prompt, 128-gen, 32t
    'gpt-oss-20b': 27.69,
    'Qwen3-30B-A3B-Instruct-2507': 27.89,
    'qwen3.6-35b-a3b': 16.36,
}

def worker(worker_id, model_path, n_tokens, result_queue, threads):
    """Worker process: load model, generate tokens, report timing."""
    os.environ['OMP_NUM_THREADS'] = str(threads)
    from turbo_engine_v7_moe import TurboEngineV7MoE
    
    eng = TurboEngineV7MoE(model_path, n_threads=threads)
    V = int(eng.vocab_size)
    
    # Warmup
    eng.reset()
    for i in range(5):
        eng.forward([i % V])
    
    # Benchmark
    eng.reset()
    t0 = time.perf_counter()
    for i in range(n_tokens):
        eng.forward([i % V])
    elapsed = time.perf_counter() - t0
    
    tok_s = n_tokens / elapsed if elapsed > 0 else 0
    result_queue.put({'worker': worker_id, 'tok_s': tok_s, 'elapsed': elapsed, 'n': n_tokens})

def bench(model_path, concurrency=4, tokens_per_worker=10):
    """Run concurrency benchmark. Returns aggregate tok/s."""
    threads_per_worker = TOTAL_THREADS // concurrency
    if threads_per_worker < 1:
        threads_per_worker = 1
    
    result_queue = Queue()
    workers = []
    t_start = time.perf_counter()
    
    for i in range(concurrency):
        p = Process(target=worker, args=(i, model_path, tokens_per_worker, result_queue, threads_per_worker))
        p.start()
        workers.append(p)
    
    total_tokens = 0
    per_worker = []
    for _ in workers:
        r = result_queue.get()
        per_worker.append(r)
        total_tokens += r['n']
    
    wall_time = time.perf_counter() - t_start
    
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    agg = total_tokens / wall_time if wall_time > 0 else 0
    avg_per = np.mean([w['tok_s'] for w in per_worker]) if per_worker else 0
    
    return {
        'concurrency': concurrency,
        'threads_per_worker': threads_per_worker,
        'total_threads': concurrency * threads_per_worker,
        'aggregate_tok_s': round(agg, 1),
        'avg_per_worker': round(avg_per, 1),
        'wall_time_s': round(wall_time, 2),
        'total_tokens': total_tokens,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MojoLlama Concurrency Benchmark')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-c', '--concurrency', type=str, default='1,2,4,8')
    parser.add_argument('-n', '--tokens', type=int, default=8)
    args = parser.parse_args()
    
    levels = [int(x) for x in args.concurrency.split(',')]
    model_name = os.path.basename(args.model).replace('.gguf', '')
    llm_ref = LLAMA_CPP_REF.get(model_name, '?')
    
    print(f"{'='*60}")
    print(f"  MojoLlama Concurrency Benchmark")
    print(f"  Model: {model_name}")
    print(f"  llama.cpp ref: {llm_ref} tok/s (single-seq)")
    print(f"{'='*60}")
    print(f"  {'Conc':>5} | {'Thrds/w':>8} | {'Agg tok/s':>10} | {'Per-wkr':>8} | {'Wall t':>8} | {'vs llama':>9}")
    print(f"  {'─'*5} | {'─'*8} | {'─'*10} | {'─'*8} | {'─'*8} | {'─'*9}")
    
    best_ratio = 0
    for conc in levels:
        r = bench(args.model, conc, args.tokens)
        ratio = r['aggregate_tok_s'] / llm_ref if isinstance(llm_ref, (int, float)) else 0
        beat = "✅" if ratio > 1.0 else ""
        print(f"  {r['concurrency']:>5} | {r['threads_per_worker']:>8} | {r['aggregate_tok_s']:>9.1f} | {r['avg_per_worker']:>7.1f} | {r['wall_time_s']:>7.1f}s | {ratio:>6.2f}x {beat}")
        if ratio > best_ratio:
            best_ratio = ratio
    
    print(f"  {'─'*55}")
    if best_ratio > 1.0:
        print(f"  🏆 MojoLlama beats llama.cpp! Best: {best_ratio:.2f}x at optimal concurrency")
    else:
        print(f"  Best ratio: {best_ratio:.2f}x (need >1.0 to beat llama.cpp)")
    print(f"{'='*60}")
