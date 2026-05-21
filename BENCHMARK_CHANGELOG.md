# MojoLlama Performance Optimization Changelog

> Systematic, benchmarked improvements to surpass llama.cpp performance.
> Every change is measured — no guessing.

## System

- **CPU:** AMD Ryzen Threadripper 3970X 32C/64t (AVX2+FMA, no AVX-512)
- **RAM:** 251.6 GB
- **Compiler:** GCC with `-O3 -mavx2 -mfma -mf16c -fopenmp`

## Baseline (2026-05-20)

| Engine | Model | Threads | PP tok/s | TG tok/s | TG median ms | TG p95 ms |
|--------|-------|---------|----------|----------|--------------|-----------|
| v77 | Llama-3.2-1B Q4_0 | 32 | 53.2 | 47.4 | 21.15 | 21.71 |
| v7_moe | Qwen3-30B-A3B Q4_K_M | 32 | — | 20.96* | 47.71* | 192.64* |
| v7_moe | Qwen3-30B-A3B Q4_K_M | 32 | 17.7 | 16.3 | 61.08 | 64.16 |

*First measurement was with 20 tokens, not representative. 30-token measurement is the real baseline.

## Optimization Log

### Step 1: P0.1 — Fix MoE atomic bottleneck in `moe_forward_omp`
- **Date:** 2026-05-20
- **File:** `src/mojollama/kernels/quant_kernels_omp.c`
- **Change:** Tried per-thread buffers (calloc overhead worse) and loop restructure (4x OMP fork/join overhead worse)
- **Result:** REGRESSION — reverted to original. Atomic overhead is negligible vs matmul cost.

### Step 2: P0.2 — SIMD exp approximation for SiLU
- **Date:** 2026-05-20
- **File:** `src/mojollama/kernels/quant_kernels_omp.c`
- **Change:** Added `silu_fast()` using Pade tanh approximation, replaced `expf` in MoE SiLU loop
- **Result:** 16.3 tok/s (baseline). SiLU is too small fraction to matter — matmul dominates.

### Step 3: Remove np.clip + np.copyto overhead
- **Date:** 2026-05-20
- **File:** `src/mojollama/kernels/turbo_engine_v7_moe.py`
- **Change:** Removed 3× `np.clip()` calls, replaced `np.copyto()` with slice assignment
- **Result:** 16.4 tok/s (+0.6%). Small win — clip was ~1% of total.

### Step 4: Replace numpy residual with C `residual_add`
- **Date:** 2026-05-20
- **File:** `src/mojollama/kernels/turbo_engine_v7_moe.py`
- **Change:** Replaced `b_x[:N] = b_r[:N] + b_oproj[:N]` with `simd.residual_add()`. Also tried C RoPE (regression, reverted).
- **Result:** 16.1 tok/s (-1.2%). ctypes call overhead > numpy slice add for 2048 floats.
- **Action:** Reverted. Keep numpy for small vector ops.

### Step 5: P1.1 — GQA Attention SIMD softmax + workspace buffer
- **Date:** 2026-05-20
- **Files:** `src/mojollama/kernels/gqa_attention.c`, `src/mojollama/kernels/turbo_engine_v7_moe.py`
- **Change:** 
  - Added SIMD softmax (vectorized exp via exp2 decomposition, AVX2 max/exp/accumulate)
  - Replaced stack-allocated `scores[4096]` with pre-allocated workspace buffer
  - Added `workspace` float pointer parameter to `gqa_attention_decode`
- **Result:** No measurable change at seq_len=512 (attention <1% of total time). Enables longer contexts.

### Step 6: P1.2 — gpt-oss-20b support + MXFP4 kernel tuning
- **Date:** 2026-05-20
- **Model:** `gpt-oss-20b-MXFP4` (24L/2880D/MoE-32x4, MXFP4 quant, 12GB GGUF)
- **Baseline:** 21.2 tok/s (47.14ms median)
- **Changes tried:**
  - 4-block unrolling of `mxfp4_row_dot_f32` — no benefit (GCC already unrolls)
  - Sequential per-expert gate+up/down — no benefit (OMP fork-join overhead ~7%)
  - Q8 quantization of SiLU before down matmul — regression (quant cost > savings)
- **Winning change:** `-march=native` + `OMP_PROC_BIND=close`
- **Result:** 21.6 tok/s (+1.9%)
- **Build:** `gcc -O3 -march=native -fopenmp -shared -fPIC -o quant_kernels_omp.so quant_kernels_omp.c -lm`
- **Benchmark:** `OMP_PROC_BIND=close OMP_PLACES=cores python3 -m src.mojollama.benchmarks.bench_tok -m gpt-oss-20b-MXFP4.gguf`

## Latest Results

| Engine | Model | Threads | PP tok/s | TG tok/s | TG median ms | TG p95 ms |
|--------|-------|---------|----------|----------|--------------|-----------|
| v77 | Llama-3.2-1B Q4_0 | 32 | 54.4 | 48.9 | 20.43 | 20.99 |
| v7_moe | Qwen3-30B-A3B Q4_K_M | 32 | 17.7 | 16.5 | 60.54 | 61.12 |
| v7_moe | gpt-oss-20b-MXFP4 | 32 | 22.2 | 21.6 | 46.14 | 48.66 | 

## Benchmark Commands

```bash
# Single model baseline
python3 src/mojollama/benchmarks/bench_tok.py -m model.gguf -t 32 --measured 30

# Thread sweep
python3 src/mojollama/benchmarks/bench_tok.py -m model.gguf -t 1,2,4,8,16,24,32 --save results.json

# Compare with previous
python3 src/mojollama/benchmarks/bench_tok.py --load benchmarks/baseline.json --load benchmarks/step1.json

# Concurrency test
python3 src/mojollama/benchmarks/bench_tok.py -m model.gguf --concurrent "1,2,4,8,16"

# Kernel-level benchmark
python3 src/mojollama/kernels/benchmarks/bench_c_avx2.py
```

## Rebuild Commands

```bash
# Rebuild all C kernels
cd src/mojollama/kernels && gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o quant_kernels_omp.so quant_kernels_omp.c -lm
cd src/mojollama/kernels && gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC -o gqa_attention.so gqa_attention.c -lm
cd src/mojollama/kernels && gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o simd_ops.so simd_ops.c -lm
cd src/mojollama/kernels && gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o cengine_batch_instr.so cengine_batch_instr.c -lm
```
