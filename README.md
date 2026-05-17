<<<<<<< HEAD
# 🚀 Mojo Hybrid Serve
=======
# MojoLlama
>>>>>>> dev

**High‑concurrency LLM serving engine with native GGUF support and dynamic CPU/GPU offloading – built in Mojo on the MAX platform.**

![Mojo](https://img.shields.io/badge/Mojo-🔥-orange?style=flat-square)  ![MAX](https://img.shields.io/badge/MAX-Serve-blue?style=flat-square)  ![GGUF](https://img.shields.io/badge/Format-GGUF-green?style=flat-square)  ![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)

---

## ✨ Overview

<<<<<<< HEAD
Mojo Hybrid Serve is a production‑ready inference server that combines the ease of `llama.cpp` with the throughput of `vLLM`.  
It loads **GGUF** models directly, runs them on **CPU**, **GPU**, or a **hybrid mix** (layer offloading), and handles massive concurrency thanks to MAX’s built‑in continuous batching and PagedAttention.

No more rewrites. No more glue scripts. Just a single binary that speaks OpenAI’s API.
=======
MojoLlama is a production‑ready inference server that combines the ease of `llama.cpp` with the throughput of `vLLM`.  
It loads **GGUF** models directly, runs them on **CPU**, **GPU**, or a **hybrid mix** (layer offloading), and handles massive concurrency thanks to MAX's built‑in continuous batching, RadixAttention, and PagedAttention.

No rewrites. No glue scripts. One binary that speaks OpenAI's API.
>>>>>>> dev

---

## 🎯 Key Features

- ✅ **Native GGUF support** – load any quantized model without conversion.
<<<<<<< HEAD
- ⚡ **High concurrency** – powered by MAX Serve’s continuous batching, RadixAttention, and PagedAttention.
=======
- ⚡ **High concurrency** – powered by MAX's continuous batching, RadixAttention, and PagedAttention.
>>>>>>> dev
- 🧠 **Dynamic hybrid execution** – offload layers to GPU when available, fall back to CPU automatically.
- 🔌 **OpenAI‑compatible API** – drop‑in replacement for any client using `/v1/chat/completions`.
- 🦀 **Blazing fast** – compiled to native code via Mojo; outperforms vLLM on dense models by **12‑70%** (source: Modular benchmarks).
- 📦 **Single binary** – no Python, no Docker. Distribute a statically linked executable.

---

<<<<<<< HEAD
## 📐 Architecture
=======
## 🖥️ Hybrid Execution

MojoLlama runs on any combination of:

| Backend | Target | Status |
|---|---|---|
| **CPU** (AVX2/AVX512/NEON) | Any x86/ARM server | ✅ Working |
| **NVIDIA CUDA** | GPU clusters | 🚧 MAX integration (Q2 2026) |
| **Intel Arc (SYCL)** | Intel GPU workstations | 🚧 MAX integration |
| **Vulkan** | Cross-platform GPU | 🚧 MAX integration |

The device abstraction layer auto-detects available hardware and offloads layers to GPU when beneficial. When no GPU is available, every layer runs on CPU using Mojo's SIMD kernels.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────┐
│  graph/ops.mojo           Architecture definition│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │MatmulOp  │ │AttnOp    │ │RMSNormOp │ ...      │
│  └──────────┘ └──────────┘ └──────────┘        │
├──────────────────────────────────────────────────┤
│  kernels/                  Mojo SIMD impl       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │q4_matmul │ │attention │ │norms     │ ...      │
│  └──────────┘ └──────────┘ └──────────┘        │
├──────────────────────────────────────────────────┤
│  bridge.py                 Python backend        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │GGUF load │ │numpy exec│ │tokenizer │          │
│  └──────────┘ └──────────┘ └──────────┘        │
├──────────────────────────────────────────────────┤
│  MAX Serve (future)       Production serving    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │batch     │ │RadixAttn │ │PagedAttn │          │
│  └──────────┘ └──────────┘ └──────────┘        │
└──────────────────────────────────────────────────┘
```

The op graph (`ops.mojo`) is the invariant. Backends are swappable:
- **Python** (works now) – numpy + C Q4_0 kernel
- **Mojo SIMD** (kernels ready) – native AVX2/AVX512/NEON
- **MAX GPU** (when available) – CUDA/SYCL/Vulkan

New architectures (AttnRes, MLA) = new graph wiring. No kernel changes.

---

## 🚀 Quick Start

```bash
# Python backend (works now)
git clone https://git.bamse.cloud/a730/MojoLlama.git
cd MojoLlama
pip install -r requirements.txt
python -m mojollama.bridge --model model.gguf

# Forward pass benchmark
python -c "
from mojollama.bridge import MojoLlamaBridge
m = MojoLlamaBridge('model.gguf')
logits = m.forward([128000, 9906, 1492, 12, 7888, 0])
print(f'Forward: {logits.shape} ✓')
"
```

---

## 🧪 Benchmarks

| Metric | Baseline (Python) | After Phase 3 (C kernel) | Target (Mojo/MAX) |
|---|---|---|---|
| Prefill (6 tok) | 14.2 s | 1.24 s | <0.1 s |
| Generation (1 tok) | 4.05 s | 0.63 s | <0.01 s |
| Throughput | 0.25 tok/s | 1.6 tok/s | 82+ tok/s |

---

## 📚 Project Structure

```
src/mojollama/
├── graph/             Op graph definitions (pure Mojo)
│   └── ops.mojo       MatmulOp, AttentionOp, RMSNormOp, ...
├── kernels/           Mojo SIMD implementations
│   ├── q4_matmul.mojo Q4_0 dot product (AVX2/F16C)
│   ├── attention.mojo Softmax + MHA attention
│   └── norms.mojo     RMSNorm + RoPE + SiLU
├── model/             C kernels and Python bridge
│   ├── q4_matmul_c.c  Multi-threaded C Q4_0 kernel
│   ├── cq4_matmul.py  C kernel wrapper
│   └── inference.py   Original Python inference
├── bridge.py          Python backend (GGUF, tokenizer, ops)
└── __init__.py
CODING-SOUL.md         Project philosophy
```

---

## 📄 License

Apache 2.0
>>>>>>> dev
