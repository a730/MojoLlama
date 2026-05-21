# MojoLlama Optimizations — May 21

## Changes Made

### 1. Q4_K MoE Down Path Q8_0 Activation (quant_kernels_omp.c)
- **Before**: MoE down matmul for Q4_K weights used `q4_k_row_dot_avx2` (dequant weights to F32 → FMA with F32 activation)
- **After**: Pre-quantizes the siLU*up buffer to Q8_0, then uses `q4_k_row_dot_q8` (VPMADDUBSW integer dot product on packed weights)
- **Impact**: Keeps Q4_K weights packed (144 bytes/256 values vs 1024 bytes dequantized), reduces cache pressure, saves dequantization FMA cost
- **Triggered**: For all MoE models with Q4_K expert down weights (Qwen3-30B, Qwen3.6, GPT-OSS MoE variant)

### 2. Static OMP Scheduling (quant_kernels_omp.c)
- **Before**: Both gate+up and down phases used `schedule(dynamic, 64)` — dynamic work-stealing with 64-iteration chunks
- **After**: `schedule(static)` — static iteration distribution
- **Impact**: Eliminates thread-safe work-stealing overhead. Uniform work per iteration (same quant type, same dimensions) makes static distribution optimal. Chunks are perfectly divisible by 32 threads (5760÷32=180, 5760÷32=180).

### 3. prealloc_q8 Buffer Size (turbo_engine_v7_moe.py)
- **Before**: Allocated only `n_blocks_x * 34` bytes (single Q8_0 block set for the norm input)
- **After**: Allocates `top_k * FF_expert * 34 // 32` bytes (enough for all top_k experts' Q8_0 activation buffers)
- **Impact**: Enables the Q4_K Q8_0 down path (change #1) which needs per-expert Q8_0 workspace

## Expected Impact
- **MoE models (Qwen3-30B, Qwen3.6)**: ~5-10% improvement on MoE FFN portion
- **Overall speedup**: ~3-5% on total throughput for MoE models
- **Dense models (GPT-OSS)**: Minimal impact (non-MoE path was already optimized)

## Current Status vs llama.cpp

| Model | MojoLlama | llama.cpp | Ratio | Status |
|-------|:---------:|:---------:|:-----:|--------|
| Qwen3.6-35B | 19.5 tok/s | 16.36 | 119% | ✅ BEATS |
| GPT-OSS-20B | 27.3 agg | 27.69 | 99% | 🔥 Near parity |
| Qwen3-30B | ~23.3 agg | 27.89 | ~83% | ⚠️ Needs work |

## Next Frontier
The remaining gap (especially Qwen3-30B) is bandwidth utilization:
- **MojoLlama**: ~46 GB/s DDR4 bandwidth
- **Llama.cpp**: ~70 GB/s (52% higher)
- **Root cause**: Less efficient memory pipeline (no prefetch, OMP barrier overhead between regions)
