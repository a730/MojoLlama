# MojoLlama

<p align="center">
  <img src="www/logo.png" alt="MojoLlama Logo" width="200">
</p>

**High‑throughput CPU LLM inference engine** — GGUF native, MoE-optimized, with a **pure Mojo inference pipeline** and full Studio suite.

![Mojo](https://img.shields.io/badge/Mojo-2024.9-6C3CE1?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square) ![C](https://img.shields.io/badge/C-AVX2%2B%20OMP-green?style=flat-square) ![GGUF](https://img.shields.io/badge/Format-GGUF-green?style=flat-square) ![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)

---

## ✨ Overview

MojoLlama is a CPU-first inference engine that provides **two engines** for different needs:

### 1. Pure Mojo Engine (NEW)
A **zero-C, zero-Python** inference pipeline written entirely in Mojo. Uses Mojo's SIMD + `parallelize` for all compute kernels with C libc imports only for I/O (`malloc`, `open`, `read`). Features:
- **Q8_0 quantized matmul** — int8→float32 SIMD decode with FMA fusion
- **Batched inference** — 1–8 concurrent sequences via comptime dispatch
- **Full architecture support** — GQA attention, RoPE, RMS norm, SiLU activations
- **Running on real models**:
  - **TinyLlama-1.1B** f16/Q8_0: **27.2 tok/s** (verified correct output)
  - **ZAYA1-8B** Q8_0 (MoE, 80 layers, 16 experts): **28.6 tok/s** (real English text output)

### 2. C Engine (Performance)
The C engine (`cengine_batch_instr.c`) provides AVX2+OMP-accelerated operations for production workloads:
- **GPT-OSS-20B** Q4_K_M: **25.5 tok/s** (0.94× llama.cpp)
- **Qwen3.6-35B** MXFP4: **25.5 tok/s** (1.57× llama.cpp) with concurrent serving at **47 tok/s aggregate**

---

## 🚀 Pure Mojo Quick Start

```bash
# Build and run TinyLlama-1.1B f16 (proven benchmark)
cd src/mojollama/mojo_engine
bash build.sh
LD_LIBRARY_PATH="..." ./tinyllama_benchmark 32

# Build and run ZAYA1-8B Q8_0 (28.6 tok/s)
bash build_zaya.sh
LD_LIBRARY_PATH="..." ./zaya_gen_q8 32
```

All matmul kernels are pure Mojo — no C bridge. See `zaya_gen_q8.mojo` (~700 lines) for the complete reference implementation.

---

## 🏆 Performance Benchmarks

| Model | Engine | Config | tok/s | vs llama.cpp |
|-------|--------|--------|-------|-------------|
| **TinyLlama 1.1B** f16 | **Pure Mojo** | B=4 × 32 thr | **27.2** | — |
| **TinyLlama 1.1B** Q8_0 | **Pure Mojo** | B=4 × 32 thr | **88.1** | — |
| **ZAYA1-8B** Q8_0 | **Pure Mojo** | 80L, 16 experts, top-1 | **28.6** 🏆 | — |
| GPT-OSS-20B Q4_K_M | Python TurboEngine | 1 user | 25.5 | **0.94×** |
| Qwen3.6-35B MXFP4 | Python TurboEngine | 1 user | 25.5 | **1.57×** |
| Qwen3.6-35B MXFP4 | C batch_forward | B=10 × 128 prompt | 17.0 | 1.06× |
| Qwen3.6-35B MXFP4 | Concurrent server (spawn) | 10 workers × 3 thr | **47.0 agg** | — |
| TinyLlama 1.1B Q4_0 | Python TurboEngine | 1 user | 79.2 | 0.89× |
| Qwen3-30B-A3B Q4_K_M | Python TurboEngine | 1 user | 22.5 | 0.89× |

### Pure Mojo Details

| File | Model | Format | tok/s | Lines |
|------|-------|--------|-------|-------|
| `tinyllama_benchmark.mojo` | TinyLlama-1.1B | f16 | 27.2 | 552 |
| `tinyllama_gen_q8.mojo` | TinyLlama-1.1B | Q8_0 B=4 | 88.1 | 552 |
| `zaya_gen_q8.mojo` | ZAYA1-8B | Q8_0 | **28.6** | **~700** |

**Architecture support (pure Mojo):**
- **Dense models**: TinyLlama (22L, GQA, RoPE, SiLU) — proven
- **MoE models**: ZAYA1-8B (80 alt attn/MoE layers, 16 experts, top-1 routing, CCA) — proven
- 262K vocab with weight-tying, partial RoPE, learned residual scales — all in Mojo

---

## 🧠 C Engine (`kernels/cengine_batch_instr.c`)

The C engine is the core performance layer — a single `.c` file (~1000 lines) with AVX2+OMP-accelerated operations:

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
- `batch_forward()` — single C call for entire forward pass
- `moe_ffn()` — MoE FFN dispatch with per-expert quant matmul
- `ssm_decode_step()` — Mamba-2-like selective scan kernel
- `gqa()` — Grouped Query Attention with softmax + weighted sum
- `PagedAttention` — page table with KV block allocation

---

## 🔧 TurboEngine v7-MoE (`kernels/turbo_engine_v7_moe.py`)

Zero-allocation Python engine with C-accelerated MoE support:

**Architecture Support:**
- GPT-OSS (24L, 32 experts, MXFP4 gate/up/down)
- Qwen3.6-35B (41L hybrid: 11 attn + 30 SSM, 256 experts, shared expert)
- Qwen3-30B, Qwen2-MoE, Llama, Mistral, Gemma
- Dense models (TinyLlama via `turbo_engine_v77.py`)

---

## 🔄 Concurrent Server

Two approaches for multi-user throughput:

### Multiprocessing spawn (best)
`server_concurrent_qwen36.py` — workers share weight pages via OS page cache:

| Workers × OMP thr | Per-worker | Aggregate | CPU util |
|-------------------|-----------|-----------|----------|
| 10 × 1 | 3.6 t/s | 36 t/s | 10/32 cores |
| **10 × 3** | **4.7 t/s** | **47 t/s** 🏆 | **30/32 threads** |
| 8 × 4 | 5.7 t/s | 46 t/s | 32/32 threads |

---

## 🗺️ Future Plans

### Near-term
- **Batch inference** for ZAYA1-8B (B=4, currently B=1)
- **Q8_0 input quantization** in C engine
- **True continuous batching** — merge B users' MoE routing

### Medium-term
- **MAX GPU backend** — zero-copy tensor dispatch to NVIDIA GPUs
- **Speculative decoding** — TinyLlama draft + ZAYA target
- **Prefix caching** — LRU with SHA-256 hash

### Long-term
- **Full GGUF quant pipeline** — imatrix → dynamic 2-bit quantization
- **MojoLlama Studio** as Unsloth+LLaMA-Factory replacement
- **Perplexity evaluation**

---

## 📚 Project Structure

```
src/mojollama/
├── mojo_engine/             # Pure Mojo inference engines
│   ├── zaya_gen_q8.mojo     ZAYA1-8B Q8_0 (28.6 tok/s)
│   ├── tinyllama_benchmark.mojo  TinyLlama-1.1B f16
│   ├── tinyllama_gen_q8.mojo     TinyLlama-1.1B Q8_0 B=4
│   ├── extract_zaya_q8.py        GGUF→raw Q8_0 .bin
│   ├── extract_zaya_vocab.py     262K vocab extraction
│   └── build_zaya.sh             Build script
├── kernels/                 # C + Python engines
│   ├── cengine_batch_instr.c     C engine (~1000 lines)
│   ├── turbo_engine_v7_moe.py    Python MoE engine
│   └── zaya_batch_forward.c      ZAYA C reference
├── server_moe.py            HTTP inference server
├── studio.py                CLI Studio (25 commands)
└── model/architectures.py   50+ arch detection
```

---

## 📄 License

Apache 2.0
