#!/usr/bin/env python3
"""Benchmark GPT-OSS-20B Q4_K_M — find path to 65 tok/s."""
import sys, gc, time, numpy as np, os
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
gc.disable()
os.environ['OMP_NUM_THREADS'] = '32'
from turbo_engine_v7_moe import TurboEngineV7MoE

engine = TurboEngineV7MoE('/tmp/models/gpt-oss-20b-Q4_K_M.gguf', n_threads=32)

for i in range(10): engine.forward(i)

ts = []
for i in range(50):
    t0 = time.perf_counter()
    y = engine.forward(i + 30)
    ts.append(time.perf_counter() - t0)

avg = np.mean(ts[5:])
best = min(ts[5:])
s = sorted(ts[5:])
p50 = s[len(s)//2]
p95 = s[int(len(s)*0.95)]

print(f'Avg: {1/avg:.1f} tok/s ({avg*1000:.2f}ms)')
print(f'Best: {1/best:.1f} tok/s')
print(f'p50: {1/p50:.1f}  p95: {1/p95:.1f}')
print(f'Std: {np.std(ts[5:])*1000:.2f}ms')

# Show individual times
for i, t in enumerate(ts):
    if i >= 5:
        print(f'  tok {i+30}: {t*1000:.2f}ms')
gc.enable()
