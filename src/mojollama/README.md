# MojoLlama

**High-throughput CPU LLM inference engine** — GGUF native, MoE-optimized, multi-architecture.

```bash
mojollama chat -m model.gguf        # Interactive chat
mojollama serve -m model.gguf -p 8080  # API server
mojollama bench -m model.gguf -n 128   # Benchmark
```

## Features

- **Multi-architecture** — Llama, Mistral, Qwen2/3, GPT-OSS, Gemma 4, ZAYA1, DeepSeek V2/V3, DBRX, Mixtral, Falcon, Phi, and 50+ more
- **GGUF native** — all quant types: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, MXFP4
- **MoE optimized** — C-accelerated fused MoE kernel
- **Hybrid SSM+Attention** — Qwen3.6-35B (Mamba-2 + Attention hybrid)
- **Per-layer projections** — Gemma 4 support with GeGLU, sliding window, logit softcapping
- **Mixture of Depth** — ZAYA1-8B with interleaved ATTN/MoE+MoD, CCA, SSM conv1d
- **C engine** — batched inference, PagedAttention, concurrent serving
- **Auto-tune** — hardware-specific optimization for OMP threads, batch sizes, concurrency
- **Studio** — web UI for chat, training, export, dataset management
- **llama.cpp fallback** — built-in llama-server for compatibility
- **MAX backend** — GPU acceleration via Modular MAX engine
- **Docker** — ready-to-use image on GHCR

## Quick Start

### Install

```bash
# From pip (coming soon)
pip install mojollama

# Or from source
git clone https://github.com/a730/MojoLlama.git
cd MojoLlama/src
pip install -e .
```

### CLI Usage

```bash
# Chat with a model
mojollama chat -m /path/to/model.gguf

# Start API server
mojollama serve -m /path/to/model.gguf -p 8080

# Benchmark
mojollama bench -m /path/to/model.gguf -n 256 -t 32

# Quantize a model
mojollama quantize model.gguf -t Q4_K_M

# Auto-tune for your hardware
mojollama autotune -m /path/to/model.gguf --mojollama

# Import from HuggingFace
mojollama convert --hf meta-llama/Llama-3.2-1B --outtype q4_0

# System info
mojollama info
```

### Docker

```bash
# Pull pre-built image
docker pull ghcr.io/a730/mojollama:latest

# Run API server
docker run --rm -v /models:/models:ro ghcr.io/a730/mojollama:latest \
    serve -m /models/model.gguf -p 8080

# Or use docker-compose
docker compose up -d
```

Auto-download models from HuggingFace:
```bash
docker run --rm -e HF_REPO=bartowski/TinyLlama-1.1B-GGUF \
    -e HF_FILE=tinyllama-1.1b.Q4_K_M.gguf \
    -e MODEL_PATH=/models/tinyllama-1.1b.Q4_K_M.gguf \
    ghcr.io/a730/mojollama:latest serve
```

## Architecture Support

| Architecture | Status | Features |
|-------------|--------|----------|
| Llama 2/3/4, Mistral | ✅ Full | Standard dense, GQA, SwiGLU, RoPE |
| Qwen2, Qwen3 | ✅ Full | TNT embedding, partial rotary |
| GPT-OSS | ✅ Full | MoE, packed experts, bias attn |
| Qwen3 MoE | ✅ Full | 128 experts top-8, shared expert |
| Qwen3.6-35B | ✅ Full | Hybrid SSM+Attention, MXFP4, MoE |
| Gemma 4 | ✅ Full | Per-layer proj, GeGLU, SWA, softcap |
| ZAYA1-8B | ✅ Full | ATTN/CCA/SSM + MoE+MoD interleaved |
| DeepSeek V2/V3 | ✅ | MLA, DeepSeekMoE |
| Mixtral | ✅ | Standard MoE |
| DBRX | ✅ | MoE with gated FFN |
| Falcon | 🔧 Planned | Parallel attention+FFN |
| Phi-3/4 | 🔧 Planned | Fused QKV, long RoPE |
| Command-R | ✅ | Tanh-gated RoPE |

## Performance

| Model | Engine | tok/s | vs llama.cpp |
|-------|--------|-------|-------------|
| TinyLlama 1.1B Q4_0 | Dense | 79.2 | 0.89x |
| GPT-OSS-20B Q4_K_M | MoE (Python) | **25.5** | **0.94x** |
| Qwen3.6-35B MXFP4 | MoE (Python) | 25.5 | 1.57x 🏆 |
| Qwen3.6-35B MXFP4 | Concurrent (10×3) | **47 agg** | — |
| Gemma 4 4.6B Q4_K_M | Gemma4 | 18.8 | — |
| ZAYA1-8B Q4_K_M | ZAYA (1 thread) | 0.9 | — |

Benchmarked on AMD Threadripper 3970X (32C/64T, 251GB DDR4), 32 threads unless noted.

## Project Structure

```
src/mojollama/
├── __main__.py            # Unified CLI entry point
├── studio.py              # Web UI and training commands
├── server.py              # API server
├── backends.py            # Backend selector (llama.cpp, MAX, numpy)
├── autotune.py            # Hardware auto-tuner
├── quantizer.py           # GGUF quantization pipeline
├── kernels/               # Inference engine
│   ├── turbo_engine_v7_moe.py  # Main MoE engine
│   ├── turbo_engine_v77.py     # Dense fallback engine
│   ├── cengine_batch_instr.c/.so  # C batched engine
│   ├── quant_kernels_omp.c/.so    # Quantized matmul kernels
│   ├── simd_ops.c/.so             # SIMD ops (RMS norm, SiLU, RoPE)
│   ├── gqa_attention.c/.so        # GQA attention kernel
│   ├── forward/                   # Architecture forward passes
│   │   ├── base.py                # ArchitectureForwardPass ABC
│   │   ├── gemma4.py              # Gemma 4 forward pass
│   │   └── zaya.py                # ZAYA1 forward pass
│   ├── dsv4/                      # DeepSeek V4 engine
│   ├── legacy-engine/             # Archived Python engines
│   └── legacy-c/                  # Archived C kernels
├── benchmarks/            # Benchmark and profiling scripts
├── tests/                 # Test scripts
├── Dockerfile             # Docker image build
├── docker-compose.yml     # Docker compose (all services)
├── docker-entrypoint.sh   # Docker entrypoint with auto-tune
└── .github/workflows/     # GitHub Actions CI/CD
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | auto | OpenMP threads for CPU parallelism |
| `MODEL_PATH` | — | GGUF model path |
| `PORT` | 8080 | API server port |
| `MOJOLLAMA_DEVICE` | auto | Device override (cpu/cuda) |
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `HF_REPO` | — | HuggingFace repo for auto-download |
| `HF_FILE` | — | HuggingFace filename for auto-download |
| `AUTO_TUNE` | true | Run auto-tune on first model load |

## Docker Services

| Service | Profile | Description |
|---------|---------|-------------|
| `mojollama` | default | API server |
| `chat` | chat | Web UI chat |
| `bench` | bench | One-shot benchmark |
| `quantize` | quantize | Model quantization |
| `autotune` | autotune | Hardware tuning |
| `llama-cpp` | fallback | llama.cpp fallback server |
| `max` | max | MAX GPU engine |

## Development

### Build C Kernels

```bash
cd src/mojollama/kernels

# Quant matmul OMP kernel
gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o quant_kernels_omp.so quant_kernels_omp.c -lm

# C batched engine
gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o cengine_batch_instr.so cengine_batch_instr.c -lm

# SIMD ops
gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o simd_ops.so simd_ops.c -lm

# GQA attention
gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o gqa_attention.so gqa_attention.c -lm
```

### Architecture Forward Passes

To add a new architecture:
1. Create `kernels/forward/<arch>.py` with a class extending `ArchitectureForwardPass`
2. Implement `forward()`, `init_weights()`, `init_pointers()`
3. Add dispatch in `turbo_engine_v7_moe.py` `__init__`
4. Add needed slots to `LayerWeights.__slots__`

## License

MIT
