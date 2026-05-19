#!/usr/bin/env python3
"""Single-token latency benchmark for MojoLlama.

Benchmarks all available models and reports ms/tok, tok/s.
Usage:
  OMP_NUM_THREADS=32 python3 -u bench_tok.py
"""

import os
import sys
import time
import subprocess
import numpy as np

MODELS = {
    "TinyLlama-1.1B-Q4_0": "/tmp/tl-Q4_0.gguf",
    "Qwen3-30B-A3B-Q4_K_M": "/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
    "GPT-OSS-20B-Q4_K_M": "/tmp/models/gpt-oss-20b-Q4_K_M.gguf",
}

N_WARMUP = 10
N_MEASURED = 50
OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", "32")


def bench_mojollama_qwen3(model_path):
    """Benchmark Qwen3-30B-A3B using TurboEngineV7MoE + C engine."""
    import gguf
    from mojollama.kernels.turbo_engine_v7_moe import TurboEngineV7MoE

    engine = TurboEngineV7MoE(model_path, n_threads=int(OMP_NUM_THREADS))
    B = 1
    tokens = np.array([[1]], dtype=np.int32)  # BOS

    # Warmup
    for _ in range(N_WARMUP):
        logits = engine.forward(tokens)
        if logits.ndim == 1:
            token = int(np.argmax(logits))
        elif logits.ndim == 3:
            token = int(np.argmax(logits[0, -1, :]))
        else:
            token = int(np.argmax(logits[-1]))
        tokens = np.array([[token]], dtype=np.int32)

    # Measured
    start = time.perf_counter()
    for _ in range(N_MEASURED):
        logits = engine.forward(tokens)
        if logits.ndim == 1:
            token = int(np.argmax(logits))
        elif logits.ndim == 3:
            token = int(np.argmax(logits[0, -1, :]))
        else:
            token = int(np.argmax(logits[-1]))
        tokens = np.array([[token]], dtype=np.int32)
    elapsed = time.perf_counter() - start

    ms_per_tok = elapsed / N_MEASURED * 1000
    tok_per_s = N_MEASURED / elapsed
    return ms_per_tok, tok_per_s


def bench_mojollama_dense(model_path, model_name):
    """Benchmark dense models using the batch C engine."""
    import numpy as np
    from mojollama.llama_backend import LlamaCppBackend

    backend = LlamaCppBackend(model_path, n_threads=int(OMP_NUM_THREADS))
    tokens = np.array([[1]], dtype=np.int32)
    
    for _ in range(N_WARMUP):
        logits = backend.forward(tokens)
        token = int(np.argmax(logits[0, -1]))
        tokens = np.array([[token]], dtype=np.int32)

    start = time.perf_counter()
    for _ in range(N_MEASURED):
        logits = backend.forward(tokens)
        token = int(np.argmax(logits[0, -1]))
        tokens = np.array([[token]], dtype=np.int32)
    elapsed = time.perf_counter() - start

    ms_per_tok = elapsed / N_MEASURED * 1000
    tok_per_s = N_MEASURED / elapsed
    return ms_per_tok, tok_per_s


def bench_llama_cpp(model_path):
    """Run llama-bench and parse result."""
    result = subprocess.run(
        ["/tmp/llama.cpp/build/bin/llama-bench",
         "-m", model_path,
         "-n", "128",
         "-t", OMP_NUM_THREADS,
         "-p", "512",
         "-r", "2"],
        capture_output=True, text=True, timeout=300
    )
    output = result.stdout + result.stderr
    # Parse the last line of the table
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    for line in reversed(lines):
        parts = line.split()
        if len(parts) >= 5 and "|" not in line:
            continue
        if "|" in line and "model" not in line.lower() and "size" not in line.lower():
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 7:
                try:
                    tok_per_s = float(cols[-1])
                    ms_per_tok = 1000.0 / tok_per_s
                    return ms_per_tok, tok_per_s
                except ValueError:
                    pass
    return None, None


def main():
    print("=" * 72)
    print("  MojoLlama Phase 1: Single-Token Latency Benchmark")
    print("=" * 72)
    print(f"  OMP_NUM_THREADS={OMP_NUM_THREADS}")
    print(f"  Warmup: {N_WARMUP} tok, Measured: {N_MEASURED} tok")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print()

    results = []

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"  [SKIP] {model_name}: model not found at {model_path}")
            results.append((model_name, 0, 0, 0, 0))
            continue

        print(f"  ─── {model_name} ───")

        # MojoLlama
        ml_ms = ml_tok = None
        try:
            if "Qwen3" in model_name:
                ml_ms, ml_tok = bench_mojollama_qwen3(model_path)
            elif "TinyLlama" in model_name:
                ml_ms, ml_tok = bench_mojollama_dense(model_path, model_name)
            elif "GPT-OSS" in model_name:
                # GPT-OSS not yet wired into engine, skip for now
                print(f"    MojoLlama: SKIP (GPT-OSS engine not wired)")
        except Exception as e:
            print(f"    MojoLlama: ERROR {e}")
            import traceback; traceback.print_exc()

        if ml_ms is not None:
            print(f"    MojoLlama: {ml_ms:.1f} ms/tok  ({ml_tok:.1f} tok/s)")

        # llama.cpp baseline
        cp_ms, cp_tok = bench_llama_cpp(model_path)
        if cp_ms is not None:
            print(f"    llama.cpp: {cp_ms:.1f} ms/tok  ({cp_tok:.1f} tok/s)")
            if ml_ms is not None:
                ratio = (ml_tok / cp_tok) * 100 if cp_tok > 0 else 0
                print(f"    Ratio:     {ratio:.0f}% of llama.cpp throughput")

        results.append((model_name, ml_ms, ml_tok, cp_ms, cp_tok))
        print()

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Model':<28} {'MojoLlama tok/s':<18} {'llama.cpp tok/s':<18} {'Ratio':<10}")
    print(f"  {'─'*27} {'─'*17} {'─'*17} {'─'*9}")
    for name, ml_ms, ml_tok, cp_ms, cp_tok in results:
        ml_str = f"{ml_tok:.1f}" if ml_tok else "N/A"
        cp_str = f"{cp_tok:.1f}" if cp_tok else "N/A"
        if ml_tok and cp_tok:
            ratio = f"{(ml_tok/cp_tok)*100:.0f}%"
        else:
            ratio = "N/A"
        print(f"  {name:<28} {ml_str:<18} {cp_str:<18} {ratio:<10}")
    print("=" * 72)


if __name__ == "__main__":
    main()
