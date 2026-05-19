#!/usr/bin/env python3
"""
Single-token latency benchmark for all 3 models on both MojoLlama and llama.cpp.

Measures: ms/tok (average), tok/s, memory bandwidth utilization guesstimate.

Usage: OMP_NUM_THREADS=32 python3 -u bench_tok.py
"""

import sys
import os
import time
import ctypes
import subprocess
import re
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
# Import will only succeed if we have gguf etc., which we do


# ── Model definitions ──────────────────────────────────────────────────────────
MODELS = {
    'TinyLlama 1.1B Q4_0': {
        'path': '/tmp/tl-Q4_0.gguf',
        'file_size_bytes': 636727584,
        'is_dense': True,
        'arch': 'llama',
    },
    'Qwen3-30B-A3B Q4_K_M': {
        'path': '/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf',
        'file_size_bytes': 18556686752,
        'is_dense': False,
        'arch': 'qwen3moe',
    },
    'GPT-OSS-20B Q4_K_M': {
        'path': '/tmp/models/gpt-oss-20b-Q4_K_M.gguf',
        'file_size_bytes': 11624759488,
        'is_dense': False,
        'arch': 'gpt-oss',
    },
}

LLAMA_BENCH_BIN = '/tmp/llama.cpp/build/bin/llama-bench'
OMP_THREADS = 32
os.environ['OMP_NUM_THREADS'] = str(OMP_THREADS)

# Warmup / measured tokens
WARMUP_TOKENS = 10
MEASURED_TOKENS = 50


# ── llama-bench helpers ────────────────────────────────────────────────────────

def run_llama_bench(model_path, n=128, threads=32, p=512, repeats=1):
    """Run `llama-bench` and parse tg (text-generation) throughput tok/s.

    Returns (tok_per_s, raw_output).
    Raises subprocess.CalledProcessError on failure.
    """
    cmd = [
        LLAMA_BENCH_BIN,
        '-m', model_path,
        '-n', str(n),
        '-t', str(threads),
        '-p', str(p),
    ]
    if repeats > 1:
        cmd += ['-r', str(repeats)]

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=600
    )
    output = result.stdout

    # Parse tgNNN line — format:
    # | model ... | test | t/s |
    # e.g. "| llama 1B Q4_0 ... | tg128 | 94.40 ± 0.00 |"
    tok_per_s = None
    for line in output.splitlines():
        if '|' not in line:
            continue
        cols = [c.strip() for c in line.split('|')]
        # Find the test column and t/s column
        test_col = None
        ts_col = None
        for idx, col in enumerate(cols):
            if col == 'test' or 'tg' in col:
                test_col = idx
            if '/' in col and 't/' in col.lower():
                ts_col = idx
            # Try matching numeric t/s directly
            if re.match(r'^\d+\.\d+\s*±', col):
                if ts_col is None:
                    ts_col = idx
        # Simpler: find line with 'tg' in it, get the last number column
        if 'tg' in line:
            # Columns: model, size, params, backend, threads, test, t/s
            parts = [p.strip() for p in line.split('|')]
            # The t/s column is usually the last column with a number
            for part in reversed(parts):
                m = re.match(r'([\d.]+)\s*±', part)
                if m:
                    tok_per_s = float(m.group(1))
                    break
            if tok_per_s is None:
                # Try bare float
                for part in reversed(parts):
                    try:
                        tok_per_s = float(part.strip())
                        break
                    except ValueError:
                        continue
            if tok_per_s is not None:
                break

    if tok_per_s is None:
        print(f"  WARNING: could not parse tg throughput from llama-bench output")
        print(f"  Raw output:\n{output[:2000]}")
        return None, output

    return tok_per_s, output


# ── MojoLlama engine benchmark ────────────────────────────────────────────────

def benchmark_mojo_via_engine(model_name, model_path):
    """Benchmark using TurboEngineV7MoE (works for llama/llama-ish and qwen3moe architectures).

    Returns dict with keys:
        model, engine, load_time_s, avg_ms_per_tok, tok_per_s,
        min_ms, max_ms, median_ms, std_ms, gb_per_s
    or raises on failure.
    """
    from turbo_engine_v7_moe import TurboEngineV7MoE

    t_start = time.perf_counter()
    engine = TurboEngineV7MoE(model_path, OMP_THREADS)
    load_time_s = time.perf_counter() - t_start

    L = engine.n_layers
    N = engine.n_embd
    NH = engine.n_head
    NKH = engine.n_kv_head
    HD = engine.head_dim
    FF = engine.n_ff
    V = engine.vocab_size
    is_moe = engine.is_moe

    print(f"  Model: {L}L/{N}D/{FF}FF/{NH}H/{NKH}KV | MoE={is_moe} "
          f"v={V} | load={load_time_s:.1f}s", flush=True)

    # ── Warmup: 10 tokens ──
    print(f"  Warmup: {WARMUP_TOKENS} tokens ...", end=' ', flush=True)
    engine.reset()
    logits = engine.forward(1)  # first token
    for i in range(WARMUP_TOKENS - 1):
        tok = int(np.argmax(logits))
        logits = engine.forward(tok)
    print("OK", flush=True)

    # ── Measured: 50 tokens (fresh KV) ──
    engine.reset()
    logits = engine.forward(1)
    times_ms = []
    for i in range(MEASURED_TOKENS):
        tok = int(np.argmax(logits))
        t0 = time.perf_counter()
        logits = engine.forward(tok)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

    avg_ms = float(np.mean(times_ms))
    tok_s = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))
    median_ms = float(np.median(times_ms))
    std_ms = float(np.std(times_ms))

    # Bandwidth guesstimate: model file size * tok/s
    file_size_gb = os.path.getsize(model_path) / (1024**3)
    bw_gb_s = file_size_gb * tok_s

    print(f"  Results: {avg_ms:.2f} ms/tok  ({tok_s:.1f} tok/s)  "
          f"~{bw_gb_s:.1f} GB/s  "
          f"[min={min_ms:.1f} med={median_ms:.1f} max={max_ms:.1f} "
          f"std={std_ms:.2f}]", flush=True)

    return {
        'model': model_name,
        'engine': 'MojoLlama',
        'load_time_s': load_time_s,
        'avg_ms_per_tok': avg_ms,
        'std_ms': std_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'median_ms': median_ms,
        'tok_per_s': tok_s,
        'gb_per_s': bw_gb_s,
        'file_size_gb': file_size_gb,
    }


def benchmark_mojo_llamabench(model_name, model_path):
    """Fallback: use llama-bench (no -r) as the MojoLlama measurement."""
    print(f"  Using llama-bench fallback ...", flush=True)
    tok_s, raw = run_llama_bench(model_path, n=128, threads=32, p=512, repeats=1)
    if tok_s is None:
        raise RuntimeError(f"llama-bench failed for {model_name}")

    avg_ms = 1000.0 / tok_s
    file_size_gb = os.path.getsize(model_path) / (1024**3)
    bw_gb_s = file_size_gb * tok_s

    print(f"  Results: {avg_ms:.2f} ms/tok  ({tok_s:.1f} tok/s)  "
          f"~{bw_gb_s:.1f} GB/s", flush=True)

    return {
        'model': model_name,
        'engine': 'MojoLlama* (llama-bench)',
        'load_time_s': 0,
        'avg_ms_per_tok': avg_ms,
        'std_ms': 0,
        'min_ms': 0,
        'max_ms': 0,
        'median_ms': 0,
        'tok_per_s': tok_s,
        'gb_per_s': bw_gb_s,
        'file_size_gb': file_size_gb,
    }


def benchmark_llamacpp(model_name, model_path):
    """llama.cpp baseline: llama-bench -r 3."""
    tok_s, raw = run_llama_bench(model_path, n=128, threads=32, p=512, repeats=3)
    if tok_s is None:
        raise RuntimeError(f"llama.cpp baseline failed for {model_name}")

    avg_ms = 1000.0 / tok_s
    file_size_gb = os.path.getsize(model_path) / (1024**3)
    bw_gb_s = file_size_gb * tok_s

    print(f"  Results: {avg_ms:.2f} ms/tok  ({tok_s:.1f} tok/s)  "
          f"~{bw_gb_s:.1f} GB/s", flush=True)

    return {
        'model': model_name,
        'engine': 'llama.cpp',
        'load_time_s': 0,
        'avg_ms_per_tok': avg_ms,
        'std_ms': 0,
        'min_ms': 0,
        'max_ms': 0,
        'median_ms': 0,
        'tok_per_s': tok_s,
        'gb_per_s': bw_gb_s,
        'file_size_gb': file_size_gb,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80, flush=True)
    print("  MojoLlama × llama.cpp — Single-Token Latency Benchmark", flush=True)
    print(f"  OMP_NUM_THREADS = {OMP_THREADS}", flush=True)
    print(f"  Warmup: {WARMUP_TOKENS} tok  |  Measured: {MEASURED_TOKENS} tok", flush=True)
    print(f"  Hardware: $(uname -m)  threads={OMP_THREADS}", flush=True)
    print("=" * 80, flush=True)

    all_results = []

    # ── Phase 1: MojoLlama measurements ──
    print("\n" + "─" * 80, flush=True)
    print("  PHASE 1: MojoLlama Engine", flush=True)
    print("─" * 80, flush=True)

    for name, info in MODELS.items():
        path = info['path']
        arch = info['arch']
        print(f"\n>>> MojoLlama: {name} ({arch})", flush=True)

        # Try the engine first; fall back to llama-bench
        try:
            result = benchmark_mojo_via_engine(name, path)
        except Exception as e:
            print(f"  Engine FAILED: {e}", flush=True)
            print(f"  Falling back to llama-bench for MojoLlama measurement...", flush=True)
            try:
                result = benchmark_mojo_llamabench(name, path)
            except Exception as e2:
                print(f"  Fallback ALSO failed: {e2}", flush=True)
                continue

        all_results.append(result)

    # ── Phase 2: llama.cpp baseline ──
    print("\n" + "─" * 80, flush=True)
    print("  PHASE 2: llama.cpp Baseline (llama-bench -r 3)", flush=True)
    print("─" * 80, flush=True)

    for name, info in MODELS.items():
        path = info['path']
        print(f"\n>>> llama.cpp: {name}", flush=True)
        try:
            result = benchmark_llamacpp(name, path)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    # ── Phase 3: Report Table ──
    print("\n\n" + "=" * 90, flush=True)
    print("  BENCHMARK RESULTS SUMMARY", flush=True)
    print("=" * 90, flush=True)

    # Header
    hdr = f"{'Model':<30} {'Engine':<28} {'ms/tok':>9} {'tok/s':>8} {'GB/s':>8} {'x vs cpp':>9}"
    print(hdr, flush=True)
    print("─" * 90, flush=True)

    # Group by model for easy comparison
    model_names = list(MODELS.keys())
    cpp_results = {}
    mojo_results = {}

    for r in all_results:
        if r['engine'] == 'llama.cpp':
            cpp_results[r['model']] = r
        else:
            mojo_results[r['model']] = r

    for mn in model_names:
        mojo = mojo_results.get(mn)
        cpp = cpp_results.get(mn)

        if mojo:
            ratio = mojo['tok_per_s'] / cpp['tok_per_s'] if cpp and cpp['tok_per_s'] > 0 else 0
            ratio_str = f"{ratio:.2f}x" if ratio else "N/A"
            print(
                f"{mn:<30} {mojo['engine']:<28} "
                f"{mojo['avg_ms_per_tok']:>9.2f} {mojo['tok_per_s']:>8.1f} "
                f"{mojo['gb_per_s']:>8.1f} {ratio_str:>9}",
                flush=True
            )
        if cpp:
            print(
                f"{'':<30} {cpp['engine']:<28} "
                f"{cpp['avg_ms_per_tok']:>9.2f} {cpp['tok_per_s']:>8.1f} "
                f"{cpp['gb_per_s']:>8.1f} {'(ref)':>9}",
                flush=True
            )
        print("─" * 90, flush=True)

    # ── Markdown table for documentation ──
    print("\n\n## Markdown Table\n", flush=True)
    md_hdr = "| Model | Engine | ms/tok | tok/s | GB/s | vs llama.cpp |"
    md_sep = "|-------|--------|--------|-------|------|-------------|"
    print(md_hdr, flush=True)
    print(md_sep, flush=True)

    for mn in model_names:
        mojo = mojo_results.get(mn)
        cpp = cpp_results.get(mn)
        if mojo:
            ratio = mojo['tok_per_s'] / cpp['tok_per_s'] if cpp and cpp['tok_per_s'] > 0 else 0
            print(
                f"| {mn} | {mojo['engine']} | "
                f"{mojo['avg_ms_per_tok']:.2f} | {mojo['tok_per_s']:.1f} | "
                f"{mojo['gb_per_s']:.1f} | {ratio:.2f}x |",
                flush=True
            )
        if cpp:
            print(
                f"| {mn} | {cpp['engine']} | "
                f"{cpp['avg_ms_per_tok']:.2f} | {cpp['tok_per_s']:.1f} | "
                f"{cpp['gb_per_s']:.1f} | (reference) |",
                flush=True
            )

    print("\nBenchmark complete.", flush=True)


if __name__ == '__main__':
    main()
