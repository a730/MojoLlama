# 🚀 Mojo Hybrid Serve

**High‑concurrency LLM serving engine with native GGUF support and dynamic CPU/GPU offloading – built in Mojo on the MAX platform.**

![Mojo](https://img.shields.io/badge/Mojo-🔥-orange?style=flat-square)  ![MAX](https://img.shields.io/badge/MAX-Serve-blue?style=flat-square)  ![GGUF](https://img.shields.io/badge/Format-GGUF-green?style=flat-square)  ![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)

---

## ✨ Overview

Mojo Hybrid Serve is a production‑ready inference server that combines the ease of `llama.cpp` with the throughput of `vLLM`.  
It loads **GGUF** models directly, runs them on **CPU**, **GPU**, or a **hybrid mix** (layer offloading), and handles massive concurrency thanks to MAX’s built‑in continuous batching and PagedAttention.

No more rewrites. No more glue scripts. Just a single binary that speaks OpenAI’s API.

---

## 🎯 Key Features

- ✅ **Native GGUF support** – load any quantized model without conversion.
- ⚡ **High concurrency** – powered by MAX Serve’s continuous batching, RadixAttention, and PagedAttention.
- 🧠 **Dynamic hybrid execution** – offload layers to GPU when available, fall back to CPU automatically.
- 🔌 **OpenAI‑compatible API** – drop‑in replacement for any client using `/v1/chat/completions`.
- 🦀 **Blazing fast** – compiled to native code via Mojo; outperforms vLLM on dense models by **12‑70%** (source: Modular benchmarks).
- 📦 **Single binary** – no Python, no Docker. Distribute a statically linked executable.

|---

## 🖥️ GPU Support

MojoLlama supports multiple GPU backends, including **Intel Arc (SYCL)**, **NVIDIA CUDA**, and **CPU** fallback.

### Intel Arc GPU

Intel Arc (Alchemist, Battlemage, and future Xe architectures) is supported via Intel SYCL through the `dpctl` and `dpnp` Python libraries.

**Requirements:**
- Intel Arc GPU (A310, A580, A750, A770, or newer)
- Intel GPU kernel driver (`i915` — included in Linux kernel 6.2+)
- Intel Level Zero runtime: `libze-intel-gpu1`, `level-zero-gpu`
- Python packages: `pip install dpctl dpnp`

**Usage:**
```bash
# Auto-detect best available device (Intel Arc > NVIDIA > CPU)
python server.py --model model.gguf

# Force Intel Arc GPU
python server.py --model model.gguf --device intel_arc

# Force CPU
python server.py --model model.gguf --device cpu

# List available devices
python server.py --list-devices

# Or set environment variable
MOJOLLAMA_DEVICE=intel_arc python server.py --model model.gguf
```

**How it works:**
The device abstraction layer (`mojollama.model.device`) auto-detects available hardware:
1. Checks for Intel Arc GPU via `dpctl` (Intel SYCL runtime)
2. Falls back to NVIDIA CUDA via `cupy`
3. Falls back to CPU via `numpy`

When an Intel Arc GPU is detected, tensors are loaded into GPU device memory and operations are accelerated using Intel oneDNN through `dpnp`. The KV cache is retained on the GPU for fast incremental decoding.

**Intel Arc optimizations:**
- **XMX acceleration**: Matrix operations use Intel Xe Matrix eXtensions when available
- **USM memory**: Unified Shared Memory for efficient CPU-GPU transfers
- **SYCL queues**: Asynchronous compute streams with explicit synchronization
- **Block quantization**: INT8/INT4 quantization support via XMX DP4A instructions

### Mojo-native GPU ops (Phase 2)

The Mojo-native rewrite (`ops.mojo`) includes an Intel Arc GPU dispatch layer that can call through to Python's SYCL bridge via Mojo's Python interop. This enables:
- `intel_arc_silu()` — SiLU activation on Intel GPU
- `intel_arc_rmsnorm()` — RMS normalization on Intel GPU  
- `intel_arc_matmul()` — Matrix multiply on Intel GPU
- `intel_arc_is_available()` — Check for Intel Arc hardware
- `intel_arc_set_device()` — Select Intel Arc GPU

---

## 📐 Architecture
