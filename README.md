[Mojo llama](https://git.bamse.cloud/a730/MojoLlama/~site)
# MojoLlama

<p align="center">
  <img src="www/logo.png" alt="MojoLlama Logo" width="200">
</p>

**High‑throughput CPU LLM inference engine** — GGUF native, MoE-optimized, with C acceleration and a full Studio suite.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square) ![C](https://img.shields.io/badge/C-AVX2%2B%20OMP-green?style=flat-square) ![GGUF](https://img.shields.io/badge/Format-GGUF-green?style=flat-square) ![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)

---

## ✨ Overview

MojoLlama is a CPU-first inference engine that beats llama.cpp on MoE architectures (GPT-OSS, Qwen3.6) by leveraging:
- **C engine** (`cengine_batch_instr.c`) — AVX2+OMP quantized matmuls, batch_forward with PagedAttention, SSM decode
- **MXFP4 experts** — 4-bit two's complement matmuls with E8M0 scale, optimized for MoE models (GPT-OSS, Qwen3.6)
- **Hybrid SSM+attention** — Qwen3.6-35B-A3B support with Mamba-2-like selective scan + partial RoPE
- **Concurrent server** — 47 tok/s aggregate across 10 users via multiprocessing + shared mmap weights

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/a730/MojoLlama.git
cd MojoLlama

# C engine benchmark (GPT-OSS-20B, 10 users)
cd src && PYTHONPATH=. OMP_NUM_THREADS=32 python3 -u mojollama/bench_batch_gptoss.py

# Concurrent server (Qwen3.6 MXFP4, 10 workers)
PYTHONPATH=. OMP_NUM_THREADS=3 python3 -u mojollama/server_concurrent_qwen36.py
```

---

## 🏆 Performance Benchmarks

| Model | Engine | Config | tok/s | vs llama.cpp |
|-------|--------|--------|-------|-------------|
|| **GPT-OSS-20B** Q4_K_M | Python TurboEngine | 1 user | **25.5** | **0.94x** |
| **Qwen3.6-35B** MXFP4 | Python TurboEngine | 1 user | **25.5** | **1.57x** |
| **Qwen3.6-35B** MXFP4 | C batch_forward | B=10 × 128 prompt | **17.0** | 1.06x |
| **Qwen3.6-35B** MXFP4 | Concurrent server (spawn) | 10 workers × 3 thr | **47.0 agg** | — |
| **TinyLlama 1.1B** Q4_0 | Python TurboEngine | 1 user | 79.2 | 0.89x |
| **Qwen3-30B-A3B** Q4_K_M | Python TurboEngine | 1 user | 22.5 | 0.89x |

### Key Wins
- **GPT-OSS-20B** — Corrected benchmark: 25.5 tok/s (0.94× vs llama.cpp)
- **Qwen3.6-35B MXFP4** — 1.57x llama.cpp with hybrid SSM+attention + partial RoPE
- **Concurrent server** — 47 tok/s aggregate (10 users) via multiprocessing with shared mmap weights

---

## 🧠 C Engine (`kernels/cengine_batch_instr.c`)

The MojoLlama C engine is the core performance layer — a single `.c` file (~1000 lines) with AVX2+OMP-accelerated operations:

**Quantized Matmul Types:**
| Type | Block Size | Operations |
|------|-----------|------------|
| Q4_0 | 32 elems | 4-bit, 18B/block, symmetric |
| Q4_K | 256 elems | K-quant 4-bit, 144B/block |
| Q5_K | 256 elems | K-quant 5-bit, 176B/block |
| Q6_K | 256 elems | K-quant 6-bit, 210B/block |
| Q8_0 | 32 elems | 8-bit, 34B/block |
| **MXFP4** | 32 elems | 4-bit two's complement + E8M0 scale |

**Key Components:**
- `batch_forward()` — single C call for entire forward pass (41 layers, SSM + attention + MoE + shared expert)
- `moe_ffn()` — MoE FFN dispatch with per-expert quant matmul, stride calculation for mixed quant types
- `ssm_decode_step()` — Mamba-2-like selective scan kernel (conv1d + state update + SiLU gate)
- `gqa()` — Grouped Query Attention with softmax + weighted sum
- `PagedAttention` — page table with KV block allocation, free list, physical-logical mapping
- `fused clamp+rms` — NaN-safe RMS norm with SIMD clamp [-1000, 1000]

### Built-in Fixes
- F32→Q8_0 converter uses **float16 scale** (Q8_0_BS=34) — float32 caused heap corruption
- Output matmul uses `x` (RMS-normed), not `xn` (stale attention norm buffer)
- Per-layer SSM check: `c->ssm_conv1d[l] != NULL` — prevents NULL deref on hybrid layers
- Q5_K stride + dispatch in moe_ffn — enables Qwen3.6 down experts

---

## 🔧 TurboEngine v7-MoE (`kernels/turbo_engine_v7_moe.py`)

Zero-allocation Python engine with C-accelerated MoE support:

**Architecture Support:**
- GPT-OSS (24L, 32 experts, MXFP4 gate/up/down, Q5_0 attention Q/K (fixed))
- Qwen3.6-35B (41L hybrid: 11 attn + 30 SSM, 256 experts, shared expert)
- Qwen3-30B, Qwen2-MoE, Llama, Mistral, Gemma
- Dense models (TinyLlama via `turbo_engine_v77.py`)

**Fused MoE Dispatch:**
- `moe_forward_omp()` — single C call replaces 24 ctypes calls per layer
- Pre-built expert pointer arrays (zero construction cost in hot path)
- Quantizes input to Q8_0 once for all Q4_K quantized-activation rows

**Weight Sharing:**
- Large quantized tensors stored as mmap views (no `.copy()`) — enables OS page cache sharing across spawn'd workers
- 4s worker load time (vs 88s with copy)

---

## 🔄 Concurrent Server

Two approaches for multi-user throughput:

### 1. C Engine batch_forward (serial B loop)
| — | — | — | — |


### 2. Multiprocessing spawn (best)
`server_concurrent_qwen36.py` — workers share weight pages via OS page cache:

| Workers × OMP thr | Per-worker | Aggregate | CPU util |
|-------------------|-----------|-----------|----------|
| 10 × 1 | 3.6 t/s | 36 t/s | 10/32 cores |
| **10 × 3** | **4.7 t/s** | **47 t/s** 🏆 | **30/32 threads** |
| 8 × 4 | 5.7 t/s | 46 t/s | 32/32 threads |
| 10 × 4 | 4.6 t/s | 46 t/s | 40/32 (over) |
| 10 × 5 | 2.4 t/s | 24 t/s | 50/32 (over) |

**Load time:** ~4s per worker (patched engine, no `.copy()`)

---

## 🗺️ Future Plans

### Near-term
- **Q8_0 input quantization** in C engine batch_matmul functions — match `quant_kernels_omp` performance (2-4× per-matmul improvement)
- **True continuous batching** — merge B users' MoE routing into single batched matmuls
- **Fix Qwen3.6 MXFP4 cleanup crash** — heap corruption on exit (`corrupted size vs. prev_size`)

### Medium-term
- **MAX GPU backend** — zero-copy tensor dispatch to NVIDIA GPUs
- **Speculative decoding** — TinyLlama draft + Qwen3 target (21% acceptance → improve via shared-vocab model pairs)
- **Prefix caching** — LRU with SHA-256 hash (implemented, needs deployment)
- **Docker + desktop app** — Electron wiring for MojoLlama Studio

### Long-term
- **Full GGUF quant pipeline** — imatrix → dynamic 2-bit quantization
- **MojoLlama Studio** as Unsloth+LLaMA-Factory replacement
- **Perplexity evaluation** — verify output quality matches llama.cpp reference

---

## 📚 Project Structure

```
src/mojollama/
├── kernels/
│   ├── cengine_batch_instr.c     C engine (~1000 lines: batch_forward, moe_ffn, SSM, PagedAttention)
│   ├── cengine_batch_instr.so    Compiled shared library
│   ├── turbo_engine_v7_moe.py    Python MoE engine (Qwen3.6, GPT-OSS, shared expert, SSM)
│   ├── turbo_engine_v77.py       Dense engine (TinyLlama, Lance Text)
│   ├── ssm_forward.c             SSM C kernel reference
│   ├── quant_kernels_omp.c       OMP quant matmul library
│   └── simd_ops.c                SIMD ops (rms_norm, silu, rope)
├── server_moe.py                 HTTP inference server
├── server_batch_moe.py           Continuous batching server (PagedAttention)
├── server_concurrent_qwen36.py   Multiprocessing concurrent server
├── studio.py                     CLI Studio (25 commands)
├── autotune.py                   Auto-tuning benchmark sweeper
├── model/architectures.py        50+ arch detection
├── prefix_cache.py               LRU prefix caching
├── speculative.py                Draft-target speculative decoding
├── bench_batch_gptoss.py         GPT-OSS batch_forward B=10 benchmark
├── bench_batch_qwen36.py         Qwen3.6 C engine wrapper
├── bench_concurrent_qwen36.py    Multi-user concurrency benchmark
├── profile_qwen36.py             Per-component profiling
└── www/studio.html               Web UI (14 panels, 77 JS functions)
```

---

## 📄 License

Apache 2.0
