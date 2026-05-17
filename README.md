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

---

## 📐 Architecture
