#!/usr/bin/env python3
"""Minimal profiler: wrap key C functions with timers."""
import sys, os, time, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turbo_engine_v7_moe import TurboEngineV7MoE

MODEL = '/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf'
e = TurboEngineV7MoE(MODEL, n_threads=32)

# Wrap key C functions with timing
timings = {}  # name -> list of durations

def wrap_so(so, fn_name, label):
    original = getattr(so, fn_name)
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000  # ms
        timings.setdefault(label, []).append(dt)
        return result
    setattr(so, fn_name, wrapped)
    return wrapped

# Wrap all key functions
wrap_so(e._kern, 'quant_matmul_omp', 'quant_matmul')
wrap_so(e._kern, 'rms_norm', 'rms_norm')
wrap_so(e._kern, 'silu', 'silu')
wrap_so(e._kern, 'moe_forward_omp', 'moe_forward')
wrap_so(e._kern, 'batch_qkv_omp', 'batch_qkv')
wrap_so(e._kern, 'batch_gate_up_omp', 'batch_gate_up')

if e._cengine is not None:
    wrap_so(e._cengine, 'ssm_decode_step', 'ssm_decode')
if e._gqa_attn is not None:
    wrap_so(e._gqa_attn, 'gqa_attention_decode', 'gqa_attention')
if e._simd is not None and hasattr(e._simd, 'rms_norm'):
    wrap_so(e._simd, 'rms_norm', 'simd_rms_norm')

# Warmup
e.reset()
for i in range(5):
    e.forward([i % e.vocab_size])

# Clear warmup timings
for k in timings: timings[k].clear()

# Profile 20 tokens
e.reset()
t0 = time.perf_counter()
for i in range(20):
    e.forward([i % e.vocab_size])
wall = (time.perf_counter() - t0) * 1000

tok_s = 20000 / wall

print(f'\n===== Qwen3.6-35B Profile ({20} tokens) =====')
print(f'Wall: {wall:.0f}ms = {tok_s:.1f} tok/s\n')

# Aggregate timings
print(f'{"Category":25s} {"Calls":>6s} {"Total ms":>10s} {"Avg ms":>8s} {"% of Wall":>10s}')
print('-' * 62)
total_categorized = 0
for label, vals in sorted(timings.items(), key=lambda x: -sum(x[1])):
    total = sum(vals)
    avg = total / len(vals)
    pct = total / wall * 100
    total_categorized += total
    print(f'{label:25s} {len(vals):>6d} {total:>9.1f}ms {avg:>7.2f}ms {pct:>8.1f}%')

unaccounted = wall - total_categorized
if unaccounted > 0:
    print(f'{"Python overhead":25s} {"—":>6s} {unaccounted:>9.1f}ms {"—":>8s} {unaccounted/wall*100:>8.1f}%')
print(f'{"TOTAL":25s} {"":>6s} {wall:>9.0f}ms {"":>8s} {100:>8.0f}%')

# Target analysis
target_tok_s = 30.0
target_ms = 1000 / target_tok_s
need_save = wall / 20 - target_ms
print(f'\nTarget: {target_tok_s} tok/s ({target_ms:.1f}ms/tok)')
print(f'Need to save: {need_save:.1f}ms/tok ({need_save/(wall/20)*100:.0f}% reduction)')

# Biggest opportunities
print(f'\nBiggest opportunities (by total time):')
for label, vals in sorted(timings.items(), key=lambda x: -sum(x[1]))[:5]:
    total = sum(vals)
    possible_save = total * 0.3  # assume 30% possible via C/optimization
    print(f'  {label}: {total/20:.1f}ms/tok — 30% save = {possible_save/20:.1f}ms/tok')
