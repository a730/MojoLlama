[Mojo llama](https://git.bamse.cloud/a730/MojoLlama/~site)
# MojoLlama

**High‑throughput LLM serving engine** with GGUF support, continuous batching, and a full Studio suite for training, export, dataset management, and chat.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square) ![Mojo](https://img.shields.io/badge/Mojo-🔥-orange?style=flat-square) ![MAX](https://img.shields.io/badge/MAX-Serve-blue?style=flat-square) ![GGUF](https://img.shields.io/badge/Format-GGUF-green?style=flat-square) ![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)

---

## ✨ Overview

MojoLlama is a production-ready inference server and model development studio. It loads **GGUF** models directly, runs them on CPU (with optional GPU via MAX), and provides:

- **MojoLlama Studio** — web UI + CLI for training, export, dataset creation, and chat
- **Inference server** — OpenAI-compatible API with SSE streaming
- **AutoBackend** — auto-selects fastest backend (llama.cpp CPU, MAX GPU, numpy fallback)
- **Bridge** — Python-native forward pass with GGUF loading, KV cache, tokenizer
- **Mojo SIMD kernels** — AVX2-optimized Q4_0 matmul, attention, norms (R&D)

---

## 🚀 Quick Start

```bash
# Clone
git clone https://git.bamseqoud/a730/MojoLlama.git
cd MojoLlama

# Start the inference server
python3 -m mojollama.server --model model.gguf --port 8080

# Or use the Studio
python3 -m mojollama.studio info
python3 -m mojollama.studio chat --model model.gguf --port 8080

# Open the web UI
# → http://localhost:8080/studio.html  (full Studio SPA)
# → http://localhost:8080/chat.html     (standalone streaming chat)
```

---

## 🎯 Features

### Inference Server (`server.py`)
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- **SSE streaming** — token-by-token response via EventSource
- **Bounded thread pool** — configurable `--max-workers` (default 32) with request queue and 503 backpressure
- **Connection pooling** — thread-safe pool of 16 persistent HTTP connections to llama.cpp backend
- **CORS support** — works with browser-based clients
- **Health checks** — `/health`, `/backend`, `/api/backend`
- **Port isolation** — `--llama-port` (default 8081) avoids conflicts with proxy port

### MojoLlama Studio (`studio.py`)
| Command | Description |
|---------|-------------|
| `info` | System info, backends, available models |
| `chat` | Interactive chat with streaming |
| `serve` | Full API server with AutoBackend |
| `train` | LoRA fine-tuning via llama.cpp |
| `export` | HF model → GGUF conversion (supports Q4_0+ via two-step) |
| `dataset create` | Create training datasets interactively |
| `dataset view` | Browse dataset contents |
| `dataset auto-label` | Auto-generate completions using loaded model |
| `merge` | Merge LoRA adapter into base GGUF model |
| `benchmark` | Measure tok/s for prompt processing and generation |
| `autotune` | Auto-tune server settings with hardware detection (`--deep`) |

### Hardware Detector (`detect.py`)
- **CPU**: Architecture (x86_64/ARM), ISA features (AVX2, AVX-512, AMX, NEON, SVE, FMA, F16C)
- **GPU**: NVIDIA CUDA (version + VRAM), AMD ROCm, Intel SYCL, Vulkan, Apple Metal
- **Memory**: Total/available RAM, DDR type/speed/channel estimation, bandwidth
- **Storage**: NVMe vs SSD, sequential read benchmark
- **Network**: Interface detection, RDMA capability
- **Power**: TDP estimation, thermal throttling risk
- **Outputs**: Recommended llama-server flags, JSON for automation, persistent save
- **Usage**: `python3 -m mojollama.detect` or `studio autotune --deep`

### Web UI (`www/`)
- **`studio.html`** — Full dark-theme SPA with 6 tabs:
  - 💬 Chat (token-by-token streaming via SSE)
  - 🎓 Train (Live loss chart via Chart.js + SSE metrics)
  - 📊 Dataset (Create, browse, auto-label)
  - 📤 Export (Async HF→GGUF with live log)
  - ⚡ Benchmark (tok/s results table)
  - 🔗 Merge (LoRA→base model)
- **`chat.html`** — Standalone streaming chat with quick prompts
- **`index.html`** — Landing page with performance benchmarks

### AutoBackend (`backends.py`)
Auto-detects best available hardware and routes inference accordingly:

```
Priority: MAX GPU (when available) > llama.cpp CPU (85 tok/s) > MAX CPU > numpy fallback
```

On Threadripper 3970X (64 cores): llama.cpp backend achieves **85 tok/s** sequential,
**272 tok/s** with 4 concurrent users.

### Bridge (`bridge.py`)
Pure Python forward pass engine:
- Loads GGUF models with full KV cache
- Supports Q4_0, Q8_0, F16, and other quantizations (via gguf library)
- Full op graph: RMSNorm, RoPE, SiLU, Multi-Head Attention, SwiGLU FFN
- Tokenizer integration with BPE encoding/decoding

```python
from mojollama.bridge import MojoLlamaBridge
m = MojoLlamaBridge('model.gguf')
logits = m.forward([128000, 9906, 1492, 12, 7888, 0])
print(f'Forward: {logits.shape} ✓')
```

### Parallel & Mojo SIMD Inference (`kernels/`)
- **numpy + multiprocessing** — 13.6× speedup on 64-core CPU via shared memory
- **Mojo SIMD** — AVX2 Q4_0 matmul kernels with **3.63 tok/s** (1 core, up from 0.88 baseline):
  - **Vectorized SIMD nibble extraction**: 4.4× speedup vs scalar (VPAND+VPSRLW+VPMOVSX)
  - **4-row register blocking**: +10–20% (input stays in L1 across 4 weight rows)
  - **Fused QKV / FFN gate+up**: 3–4× memory bandwidth savings
  - **Per-core gap to hand-tuned AVX2**: only 1.2× (1.04ms vs 0.86ms for 2048×2048)
  - **At 32 cores**: memory-bandwidth bound at **~81 tok/s** — within 0.4% of llama.cpp
  - **IPC bridge** reverse-engineered from MAX: `unchecked_downcast_value` + `PyArrayObject` for zero-copy numpy→Mojo data transfer
- **C AVX2 reference kernel** (q4_kernel_avx2.c/.so): hand-tuned AVX2 intrinsics,
  benchmarks against and validates the Mojo SIMD path. OpenMP variant (q4_kernel_omp.c)
  achieves 15,385 matmul/s at 32 threads (0.065ms per 2048×2048 Q4_0).
- **line-q4_quanter** — benchmark and compare quantization levels

---

## 📊 Quantization Benchmarks

TinyLlama 1.1B on Threadripper 3970X (64 cores, AVX2+FMA):

| Quant | Size | Prompt (tok/s) | Gen (tok/s) | BPW |
|-------|------|---------------|-------------|-----|
| TQ2_0 | 325 MB | 623 | 48 | 2.06 |
| Q2_K | 411 MB | 526 | **71** | 3.14 |
| Q3_K | 522 MB | 497 | 55 | 3.98 |
| **Q4_0** | **607 MB** | **595** | **66** | **4.63** |
| Q5_0 | 730 MB | 516 | 55 | 5.57 |
| Q6_K | 861 MB | 449 | 39 | 6.56 |
| Q8_0 | 1.09 GB | 525 | 44 | 8.50 |
| F16 | 2.05 GB | 564 | 23 | 16.00 |

**Insights:**
- Q4_0 is the size/speed sweet spot (66 tok/s gen, 607 MB)
- Q2_K has the fastest generation (71 tok/s) but lowest quality
- All quants are 2–3× faster than F16 (memory-bandwidth bound)
- Gemma 3 12B at Q4_K_M: 230 tok/s prompt, 15 tok/s generation

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────┐
│  www/       Web UI (studio.html, chat.html, index)   │
├──────────────────────────────────────────────────────┤
│  server.py  OpenAI API + SSE streaming + REST API    │
├──────────────────────────────────────────────────────┤
│  studio.py  CLI: train, export, dataset, merge, chat │
├──────────────────────────────────────────────────────┤
│  backends.py  AutoBackend (llama.cpp, MAX, numpy)    │
├──────────────────────────────────────────────────────┤
│  bridge.py     Python forward pass + GGUF loading    │
│  kernels/      Mojo SIMD + numpy parallel matmul     │
│  model/        C kernel + Python inference scaffold  │
└──────────────────────────────────────────────────────┘
```

New architectures (AttnRes, MLA) = new graph wiring in `graph/ops.mojo`.
Backends are swappable: Python (works now) → Mojo SIMD (kernels ready) → MAX GPU (future).

---

## 🖥️ Performance Benchmarks

| Model | Backend | Prompt | Generation | Concurrent |
|-------|---------|--------|-----------|------------|
| Llama 3.2 1B (Q4_0) | llama.cpp | 1,514 tok/s | 162 tok/s | 272 tok/s (4×) |
| Qwen3-30B-A3B (Q4_K_M) | llama.cpp | 150 tok/s | 28.5 tok/s | — |
| MAX CPU | MAX | — | 14.5 tok/s | — |
| Mojo SIMD Q4_0 | Mojo | 172 matmul/s | ~1 tok/s | — |
| Numpy parallel (64-core) | Python | 8.7 matmul/s | 0.3 tok/s | — |

---

## 🧪 Test Commands

```bash
# Start server (with auto-tuned settings)
python3 -m mojollama.server --model model.gguf --port 8080 --llama-port 8081

# Hardware detection
python3 -m mojollama.detect
python3 -m mojollama.detect --model model.gguf --json --save

# Auto-tune server settings  
python3 -m mojollama.studio autotune --model model.gguf --deep --quick

# Chat via curl
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# Streaming chat (SSE)
curl -N -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Count to 5","max_tokens":30,"stream":true}'

# Benchmark quants
./llama-bench -m model.gguf -p 512 -n 128 -t 64

# Concurrent benchmark
python3 bench_concurrency.py --url http://localhost:8080 --concurrency "1,4,8,16,32"

# Parallel matmul benchmark
python3 src/mojollama/kernels/parallel_q4.py --rows 2048 --cols 2048
```

---

## 📚 Project Structure

```
src/mojollama/
├── server.py           OpenAI API server (SSE streaming, REST API)
├── backends.py         AutoBackend (llama.cpp, MAX, numpy)
├── studio.py           CLI Studio (train, export, dataset, merge, chat, autotune)
├── bridge.py           Python forward pass + GGUF loader
├── detect.py           Hardware detector (CPU/GPU/memory/storage/power)
├── autotune.py         Auto-tuning benchmark sweeper
├── scheduler.py        Request batcher for high throughput
├── llama_backend.py    llama.cpp server bridge
├── graph/ops.mojo      Op graph definitions (Mojo)
├── kernels/
│   ├── q4_matmul.mojo     Q4_0 AVX2 dot product
│   ├── parallel_q4.py     Parallel Q4_0 matmul (Python+multiprocessing)
│   ├── parallel_matmul.mojo  Mojo SIMD matmul CLI (blocked on unsafe_from_address)
│   ├── moe.mojo           MoE SIMD ops (router, expert matmul)
│   ├── attention.mojo     Softmax + MHA
│   └── norms.mojo         RMSNorm + RoPE + SiLU
├── model/
│   ├── inference.py    Original Python inference
│   └── q4_matmul_c.c   C Q4_0 kernel (broken nibble order — use gguf.dequantize)
├── mojollama_studio    CLI entry point for Studio
www/
├── studio.html         Full Studio SPA (6 tabs, streaming chat, charts)
├── chat.html           Standalone streaming chat
└── index.html          Landing page with benchmarks
```

---

## 🐳 Docker

### Build the Image

```bash
docker build -t mojollama:latest .
```

Multi-stage build: Stage 1 compiles `llama-server` from source, Stage 2 builds the
runtime image on `python:3.11-slim` (~500 MB final image).

### Run

Mount your GGUF model file and expose the API port:

```bash
docker run --rm -it \
  -p 8080:8080 \
  -v /path/to/models:/models:ro \
  mojollama:latest --model /models/my-model.gguf
```

**Auto-detection:** If you omit `--model`, the entrypoint scans `/models/` and `/app/`
for `.gguf` files and picks the first one found.

```bash
# Auto-detect model in /models volume
docker run --rm -it \
  -p 8080:8080 \
  -v /path/to/models:/models:ro \
  mojollama:latest
```

### Configuration

Mount a custom `~/.mojollama/config.json` for llama.cpp tuning:

```bash
docker run --rm -it \
  -p 8080:8080 \
  -v /path/to/models:/models:ro \
  -v /path/to/config.json:/home/mojollama/.mojollama/config.json:ro \
  mojollama:latest
```

If no config is mounted, the entrypoint creates a sensible default.

### Options

| Argument | Env var | Default | Description |
|----------|---------|---------|-------------|
| `--model` | `MODEL_PATH` | auto-detect | Path to GGUF model |
| `--port` | `PORT` | `8080` | MojoLlama API port |
| `--llama-port` | `LLAMA_PORT` | `8081` | llama.cpp backend port |
| `--max-workers` | — | `32` | Max concurrent requests |
| `--weight` | `WEIGHT_PATH` | — | MAX weight file path |

### Health Check

```bash
curl http://localhost:8080/health
```

### Volumes

| Mount point | Purpose |
|-------------|---------|
| `/models` | GGUF model files (ro recommended) |
| `~/.mojollama` | Server config (`config.json`) |

---

## 📄 License

Apache 2.0
