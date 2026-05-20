#!/usr/bin/env python3
"""Qwen3.6 MXFP4 concurrent server via multiprocessing.spawn.
Patched engine: no .copy() on quantized tensors → mmap views shared via OS page cache."""
import sys, os, time, multiprocessing as mp, numpy as np

MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"
N_WORKERS = 9
GEN_TOKENS = 50
OMP_THR = 3  # 9 × 3 = 27 threads (roomier)

os.environ['OMP_NUM_THREADS'] = str(OMP_THR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
from turbo_engine_v7_moe import TurboEngineV7MoE

def worker(tid, queue):
    os.environ['OMP_NUM_THREADS'] = str(OMP_THR)
    sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
    from turbo_engine_v7_moe import TurboEngineV7MoE
    
    try:
        e = TurboEngineV7MoE(MODEL, OMP_THR)
        V = e.vocab_size
        
        for _ in range(5):
            e.forward(np.array([[1]], dtype=np.int32))
        e.reset()
        
        tok = np.array([[1 + tid]], dtype=np.int32)
        t_list = []
        for step in range(GEN_TOKENS):
            t0 = time.perf_counter()
            logits = e.forward(tok)
            elapsed = time.perf_counter() - t0
            t_list.append(elapsed)
            tok = np.array([[int(np.argmax(logits))]], dtype=np.int32)
        
        t_list = t_list[5:]
        avg_ms = float(np.mean(t_list) * 1000)
        tps = float(1000 / avg_ms)
        queue.put((tid, avg_ms, tps))
    except Exception as ex:
        queue.put((tid, -1, str(ex)[:60]))

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print(f"Spawning {N_WORKERS} workers ({OMP_THR} thr/worker, patched engine)...", flush=True)
    queue = mp.Queue()
    procs = []
    
    t_start = time.perf_counter()
    for i in range(N_WORKERS):
        p = mp.Process(target=worker, args=(i, queue))
        procs.append(p)
        p.start()
    
    results = {}
    for _ in range(N_WORKERS):
        try:
            tid, ms, tps = queue.get(timeout=300)
            results[tid] = (ms, tps)
            print(f"  Worker {tid:2d}: {ms:5.1f}ms → {tps:5.1f} tok/s ({len(results)}/{N_WORKERS})", flush=True)
        except Exception as ex:
            print(f"  Timeout: {ex}", flush=True)
    
    for p in procs:
        p.join(timeout=5)
    
    wall = time.perf_counter() - t_start
    
    print(f"\n{'='*60}")
    print(f"  Qwen3.6 MXFP4 — Concurrent Server (spawn, {N_WORKERS} workers)")
    print(f"{'='*60}")
    tps_list = []
    for tid in sorted(results):
        ms, tps = results[tid]
        print(f"  Worker {tid:2d}: {ms:5.1f}ms → {tps:5.1f} tok/s")
        tps_list.append(tps)
    
    if tps_list:
        avg_tps = float(np.mean(tps_list))
        agg = avg_tps * N_WORKERS
        print(f"{'─'*60}")
        print(f"  Per-user:  {avg_tps:.1f} tok/s avg")
        print(f"  Aggregate: {agg:.0f} tok/s")
        print(f"  Speedup vs single (30.5): {agg/30.5:.1f}x")
        print(f"  200% target (51 tok/s):  {'PASS' if agg>51 else 'FAIL'} {agg/51:.1f}x")
        print(f"  750% target (191 tok/s): {'PASS' if agg>191 else 'FAIL'} {agg/191:.1f}x")
