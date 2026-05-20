# MojoLlama Roadmap

> **Vision:** One-stop CPU-first LLM platform — quantize, train, evaluate, benchmark, and serve any model on HuggingFace. The Unsloth+LLaMA-Factory alternative built on Modular MAX, with a full Studio experience.

**Current:** v0.5.0 — CPU inference engine, 50+ arch detection, 2 custom forward passes, C batched engine, 14-panel Studio dashboard, Docker, 2.26× vs llama.cpp (GPT-OSS).

**Codebase:** ~39K Python, ~12K C, ~5.6K HTML across 60+ Python files, 7 C kernels, 1 comprehensive web UI.

---

## 🎯 Strategic Pillars

```
┌──────────────────────────────────────────────────────────────────┐
│                       MojoLlama Platform                          │
├─────────────┬──────────────┬─────────────┬───────────────┬────────┤
│  INFERENCE  │   STUDIO     │   ENGINE    │   GPU         │ DEPLOY │
│  Engine     │   One-stop   │   Full HF   │   MAX +       │ Docker │
│  + Serving  │   ML App     │   Coverage  │   Tuning      │ + App  │
├─────────────┼──────────────┼─────────────┼───────────────┼────────┤
│ C engine    │ Quantization │ 60+ arch    │ CUDA/ROCm    │ pip    │
│ Batch inf   │ Training     │ forward     │ Metal backend │ Desktop│
│ PagedAttn   │ Dataset      │ passes      │ GPU tuning    │ CLI    │
│ Concurrent  │ Benchmark    │ Auto-tune   │ Multi-GPU     │ CI/CD  │
└─────────────┴──────────────┴─────────────┴───────────────┴────────┘
```

---

## Phase 0: Foundation (current — shipped)

| Area | What exists | Lines |
|------|------------|-------|
| **C engine** | `cengine_batch_instr.c` — AVX2+OMP quant matmul, batch_forward, SSM decode, PagedAttention | ~1000 |
| **Python engine** | `turbo_engine_v7_moe.py` — zero-allocation MoE, C-accelerated quant matmul | 1177 |
| **Custom forward passes** | `forward/gemma4.py` (445), `forward/zaya.py` (187) | 632 |
| **Studio web UI** | 14 panels: Chat, Train, Dataset, Export, Benchmark, Auto-Tune, Evaluate, Quantize, Merge, Checkpoints, Templates, HF Hub, Experiments, Model Cards | 4121 |
| **Studio CLI** | `studio.py` — 25+ commands: train, export, merge, dataset, chat, serve, benchmark, evaluate, quantize, imatrix, dynamic_quant, hub, checkpoint | 1929 |
| **Training** | `trainer.py` — 9 methods (LoRA, QLoRA, DoRA, GaLore, DPO, ORPO, KTO, SimPO, GRPO), all subprocess wrappers | 1505 |
| **Dataset** | `dataset.py` — 5 format readers, streaming, auto-labeling, splits, stats | 1221 |
| **Quantizer** | `quantizer.py` — 35 quant types, HF→GGUF converter, benchmark, validate | 1389 |
| **Server** | `server.py` — OpenAI-compatible, SSE streaming, tool calling, 12 API endpoints | 965 |
| **Auto-tune** | `autotune.py` — 7-phase hardware sweep (threads, batch, flash attn, parallel, mlock, engine, concurrency) | 867 |
| **Backends** | `backends.py` — Auto-backend selector (llama.cpp > MAX(stub) > numpy) | 631 |
| **Arch detection** | `model/architectures.py` — 60+ archs mapped to 9 forward types | 918 |
| **Docker** | Dockerfile, docker-compose.yml (6 profiles), entrypoint with auto-tune + HF download | — |
| **Web site** | `www/` — index.html, studio.html, docs/, studio.html | 5624 |

---

## Phase 1: Engine — Full HF Model Coverage (ongoing — 4 weeks)

The biggest gap: only 2 of 60+ detected architectures have custom forward passes. Everything else uses generic fallback paths that may produce incorrect output.

| # | Task | Est. | Impact |
|---|------|------|--------|
| 1.1 | **Profile generic fallback vs custom** — measure correctness (perplexity diff) on 10 archs using generic path | 3d | Foundation |
| 1.2 | **Falcon forward pass** — parallel attention+FFN, fused QKV, inherent norm fusion | 4d | Unlocks Falcon-7B/40B/180B |
| 1.3 | **Phi-3/4 forward pass** — fused QKV, long RoPE (128K), dense MoE variants (Phi-3 MoE) | 4d | Unlocks Phi-3/4 family |
| 1.4 | **DeepSeek V2/V3 forward pass** — Multi-head Latent Attention, DeepSeekMoE, KV compression | 7d | Unlocks DeepSeek family |
| 1.5 | **Qwen2-VL forward pass** — vision encoder + language decoder interop | 5d | Unlocks vision models |
| 1.6 | **Command-R / Cohere2 forward pass** — Tanh-gated RoPE, logit softcapping | 3d | Unlocks Cohere family |
| 1.7 | **BTLM / GLM forward pass** — bidirectional + autoregressive hybrid | 4d | Unlocks GLM family |
| 1.8 | **StableLM / OLMo forward pass** — grouped-query attention variants, ALiBi | 3d | Unlocks OLMo family |
| 1.9 | **Automated forward pass generator** — template-based from GGUF metadata (infer n_head, n_kv_head, n_ff, act_fn, norm_type) | 5d | Rapid-fire remaining 40 archs |
| 1.10 | **Regression benchmark** — compare output tokens vs llama.cpp reference per arch | 3d | Quality gate |

**Success metric:** All 60+ architectures produce correct output (perplexity within 1% of llama.cpp reference).

---

## Phase 2: Studio — One-Stop ML App (4-8 weeks)

The Studio has 14 panels but many delegate to external tools. The vision: everything in one app, no external binaries required.

### 2A. Quantization Pipeline (2 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 2A.1 | **Pure-Python quantizer** — full Q4_K, Q5_K, Q6_K, Q8_0 in Python (no llama.cpp binary) | 5d | Remove external dependency |
| 2A.2 | **MXFP4 quantizer** — fast C-based conversion from Q8_0 (replace 600s Python timeout) | 3d | Enable MXFP4 without Python loop |
| 2A.3 | **Imatrix computation in Python** — importance matrix from calibration data | 4d | No llama-imatrix binary needed |
| 2A.4 | **Dynamic 2-bit quantization** — per-tensor quant assignment (Q6_K→Q2_K) guided by imatrix | 5d | Max compression with min quality loss |
| 2A.5 | **NF4 quantizer** — 4-bit NormalFloat for QLoRA training | 3d | In-house NF4, no external tool |
| 2A.6 | **Studio quant wizard** — pick model → pick quants → pick calibration data → preview size/speed → run | 4d | 7.7 from previous roadmap |

### 2B. Training Pipeline (4 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 2B.1 | **LoRA in Python** — actual gradient computation in Python using MojoLlama engine (no subprocess) | 7d | First in-house training |
| 2B.2 | **QLoRA in Python** — NF4 base + LoRA adapters, fused forward/backward | 5d | Train 70B on consumer hardware |
| 2B.3 | **Training loop UI** — live loss charts, step counter, ETA, learning rate schedule | 4d | Studio panel enhancement |
| 2B.4 | **Dataset import from HF** — load datasets from HuggingFace Hub directly | 3d | No local file needed |
| 2B.5 | **DPO/ORPO training** — preference optimization with Python gradient | 5d | Alignment training |
| 2B.6 | **Checkpoint management UI** — save/load/resume, compare checkpoints, export LoRA | 3d | Studio panel enhancement |
| 2B.7 | **Multi-LoRA serving** — load multiple adapters, switch at runtime, merge on-the-fly | 4d | Production LoRA |

### 2C. Benchmarking (2 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 2C.1 | **Capability benchmarks** — MMLU, GSM8K, HumanEval, HellaSwag, ARC, BBH in-studio (no external server) | 5d | Measure quality, not just speed |
| 2C.2 | **Benchmark comparison view** — side-by-side tok/s + accuracy across engines/quant | 3d | Studio panel enhancement |
| 2C.3 | **Perplexity evaluation** — cross-entropy loss on held-out text, compare to llama.cpp reference | 2d | Regression detection |
| 2C.4 | **Benchmark history** — track changes over time, plot regression charts | 2d | Track progress |
| 2C.5 | **Auto-generated benchmark report** — export as HTML/Markdown for sharing | 2d | Share results |

### 2D. Dataset Management (1 week)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 2D.1 | **HF dataset integration** — search, browse, download datasets from HuggingFace Hub | 3d | No local files |
| 2D.2 | **Dataset preview UI** — view samples, filter, search within dataset | 3d | Studio panel enhancement |
| 2D.3 | **Quality filters** — perplexity-based filtering, near-dedup, length filters | 3d | Clean training data |
| 2D.4 | **Preference pair generator** — create DPO/RLHF pairs from completions | 3d | Alignment data |

---

## Phase 3: Inference Engine — Performance + GPU (4-8 weeks)

### 3A. CPU Engine Maturation (3 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 3A.1 | **Q8_0 input C quant** — quantize activations to Q8_0 before matmul in C engine (2-4× improvement) | 3d | Critical bottleneck fix |
| 3A.2 | **MXFP4 C kernel** — native C MXFP4 matmul (replace Python fallback, 10× speedup expected) | 4d | Unlocks Qwen3.6 C engine |
| 3A.3 | **Fused Q+K projection** — single matmul for Q+K instead of 3 calls | 2d | 2-5% end-to-end |
| 3A.4 | **True continuous batching** — merge multiple users' MoE routing into single batched matmuls | 5d | Linear scaling with users |
| 3A.5 | **PagedAttention production** — page table with KV block reclamation, defrag, swap | 4d | Long-context serving |
| 3A.6 | **Fix Qwen3.6 MXFP4 cleanup crash** — heap corruption on exit | 1d | Stability |

### 3B. GPU via MAX — Full GPU Pipeline (6 weeks)

The MAX backend is currently a stub. This phase builds end-to-end GPU support: loading models, dispatching layers, tuning performance, and handling multi-GPU.

#### 3B.1 GPU Integration (2 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 3B.1.1 | **MAX GGUF reader** — load GGUF tensors into MAX-compatible buffers, map quantized types to MAX types | 5d | Foundation for all GPU work |
| 3B.1.2 | **MAX CPU inference** — run full LLM forward pass via MAX engine on CPU as baseline | 3d | Validate MAX engine works |
| 3B.1.3 | **MAX GPU pipeline** — model load → transfer to GPU → generate tokens → return logits | 5d | First GPU inference |
| 3B.1.4 | **Layer-by-layer GPU dispatch** — per-layer offloading decision (attn→GPU, FFN→CPU, or all→GPU) | 4d | Hybrid inference control |

#### 3B.2 GPU Acceleration (2 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 3B.2.1 | **GPU attention** — offload Q, K, V projections + softmax attention to GPU | 4d | Biggest speedup per layer |
| 3B.2.2 | **GPU MoE dispatch** — expert routing + top-k selection + expert matmuls on GPU | 5d | MoE models on GPU |
| 3B.2.3 | **GPU KV cache** — KV cache blocks on GPU memory, PagedAttention with GPU page table | 4d | Long-context on GPU |
| 3B.2.4 | **CUDA graphs** — capture repeated forward pass structure as CUDA graph for <10μs kernel launch | 3d | Minimize GPU overhead |
| 3B.2.5 | **Flash Attention** — use MAX's FlashAttention-2 integration for O(1) memory attention | 3d | 32K+ context on GPU |

#### 3B.3 Multi-GPU (2 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 3B.3.1 | **Tensor parallelism** — split weight matrices across GPUs, all-reduce during forward | 5d | Model sharding |
| 3B.3.2 | **Pipeline parallelism** — assign layer groups to different GPUs | 4d | Deeper models on multi-GPU |
| 3B.3.3 | **GPU memory management** — dynamic offloading, swap KV cache between GPU↔CPU, memory budget tracking | 4d | Fit larger models |
| 3B.3.4 | **Multi-GPU benchmark suite** — speedup curves (1/2/4/8 GPUs), memory scaling, batch scaling | 2d | Know your hardware |

#### 3B.4 GPU Formats & Quantization (2 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 3B.4.1 | **GPU-native quant types** — FP16, FP8 (E4M3/E5M2), INT8, INT4 on GPU natively (no dequant) | 5d | GPU memory efficiency |
| 3B.4.2 | **On-the-fly quant conversion** — convert GGUF Q4_K/Q8_0 to GPU-native FP8/INT8 at load time | 3d | No re-quantizing needed |
| 3B.4.3 | **GPU imatrix** — importance matrix computed via GPU (10× faster than CPU) | 3d | Fast calibration |
| 3B.4.4 | **Mixed-precision GPU inference** — FP16 attn + INT8 FFN + FP32 embeddings automatically | 3d | Max throughput per layer type |

#### 3B.5 GPU Auto-Tune (2 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 3B.5.1 | **GPU layer sweep** — sweep n_gpu_layers 0→all, measure tok/s, find knee | 3d | Optimal GPU/CPU split |
| 3B.5.2 | **GPU batch size sweep** — test batch sizes 1→64 with GPU, find throughput plateau | 2d | Max concurrency |
| 3B.5.3 | **GPU memory tuning** — measure peak memory per model/size/quant, auto-pick batch to fit | 3d | No OOM crashes |
| 3B.5.4 | **GPU concurrent tuning** — sweep concurrent users 1→N, find latency knee (p50/p95/p99) | 3d | Production config |
| 3B.5.5 | **Multi-GPU tuning** — sweep GPU count (1/2/4/8), parallelism strategy, interconnect type | 4d | Multi-GPU optimization |
| 3B.5.6 | **Metal/ROCm tuning** — Apple Silicon GPU and AMD GPU specific parameter sweeps | 3d | Cross-platform GPU |

#### 3B.6 GPU Support Matrix

| Backend | Hardware | Status | Target tok/s (7B Q4_K_M) |
|---------|----------|--------|--------------------------|
| **MAX CUDA** | NVIDIA Tesla/RTX/A100/H100/B300 | ⬜ Not started | 100+ (A100) |
| **MAX ROCm** | AMD MI250/MI300X/RX 7000 | ⬜ Not started | 80+ (MI250) |
| **MAX Metal** | Apple M1/M2/M3/M4 (Pro/Max/Ultra) | ⬜ Not started | 40+ (M2 Ultra) |
| **MAX CPU** | Any x86_64/ARM with AVX2/SVE | ⬜ Not started | 20+ (Threadripper) |
| **MojoLlama CPU** | x86_64 AVX2+OMP | ✅ Working | 13-63 (Threadripper) |

#### 3B.7 GPU Production (1 week)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 3B.7.1 | **GPU server** — serve models from GPU via /v1/chat/completions, SSE streaming | 3d | Production GPU serving |
| 3B.7.2 | **GPU health monitoring** — GPU utilization, memory, temperature, power in Studio metrics | 2d | Ops visibility |
| 3B.7.3 | **Automatic fallback** — GPU OOM → auto-offload to CPU → continue with degraded throughput | 2d | Reliability |
| 3B.7.4 | **GPU model hot-swap** — unload model from GPU, load new model, without restart | 2d | Multi-model serving |

### 3C. Auto-Tune Improvement (2 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 3C.1 | **Self-contained auto-tune** — no external llama-bench binary required | 4d | Pure MojoLlama sweeps |
| 3C.2 | **Per-model auto-tune cache** — store per-model-hash configs, share with community | 2d | Instant optimal config |
| 3C.3 | **GPU auto-tune** — sweep GPU layers, batch size, concurrency with MAX backend | 3d | GPU optimization |
| 3C.4 | **Auto-tune recommendation engine** — suggest optimal config from hardware fingerprint | 3d | Zero-click tuning |
| 3C.5 | **Auto-tune Studio UI** — guided wizard with progress, charts, one-click apply | 3d | Studio panel |

---

## Phase 4: Platform — Ship It (4-6 weeks)

| # | Task | Est. | Description |
|---|------|------|-------------|
| 4.1 | **pip package** — `pip install mojollama` with pre-built wheels (manylinux x86_64, aarch64) | 5d | Zero-install |
| 4.2 | **Desktop app** — Tauri/Electron wrapper for Studio, bundled Python runtime + model downloader | 7d | Desktop experience |
| 4.3 | **CI/CD pipeline** — GitHub Actions: test every PR, benchmark regression checks, build wheels | 3d | Quality gates |
| 4.4 | **Comprehensive test suite** — unit tests for each module, integration tests for forward passes | 5d | Regression prevention |
| 4.5 | **API hardening** — API keys, rate limiting, model hot-swap, token usage tracking | 4d | Production-ready server |
| 4.6 | **Model registry** — local + HF Hub model browser, one-click download, version management | 4d | Model management |
| 4.7 | **Release automation** — semantic versioning, changelog, automated release notes | 2d | Clean releases |

---

## Timeline

```
Month 1                Month 2                Month 3                Month 4                Month 5+        
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────►
│ Phase 1: HF Coverage                       │ Phase 2: Studio Pipeline                    │ Phase 4
│ 1.1-1.5 Custom forward passes              │ 2A. Quant Pipeline                          │ Ship
│ 1.6-1.10 Broader arch + auto-gen           │ 2B. Training                                │ pip
│                                             │ 2C. Benchmarking                            │ Desktop
│ Phase 3A: CPU Engine                       │ 2D. Dataset                                 │ CI/CD
│ 3A.1-3.3 C kernel improvements             │                                             │
│ 3A.4-3.6 Batching + stability              │ Phase 3B: GPU via MAX (Full Pipeline)       │
│                                             ├─────────────────────────────────────────────┤
│ Phase 3C: Auto-Tune (CPU)                  │ 3B.1 GPU Integration                         │
│ 3C.1-3.3 Self-contained                    │ 3B.2 GPU Acceleration                        │
│ 3C.4-3.5 Studio wizard                     │ 3B.3 Multi-GPU                               │
│                                             │ 3B.4 GPU Formats & Quant                    │
│                                             │ 3B.5 GPU Auto-Tune                          │
│                                             │ 3B.6 Support Matrix                         │
│                                             │ 3B.7 GPU Production                         │
└──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┴──────────────►
```

## Architecture Coverage Roadmap

```
Current (2 custom passes):      Target (60+ passes):
┌────────────────────┐         ┌──────────────────────────┐
│ ✅ Gemma 4         │         │ ✅ Llama 2/3/4           │
│ ✅ ZAYA1           │         │ ✅ Mistral               │
│ ⬜ Falcon          │         │ ✅ Qwen2/3               │
│ ⬜ Phi-3/4         │         │ ✅ Gemma 2/3/4           │
│ ⬜ DeepSeek V2/V3  │         │ ✅ GPT-OSS               │
│ ⬜ 50+ more...     │         │ ✅ ZAYA1                 │
└────────────────────┘         │ ✅ Falcon                │
                               │ ✅ Phi-3/4               │
Auto-generated template        │ ✅ DeepSeek V2/V3        │
for remaining archs:           │ ✅ Command-R / Cohere    │
  GGUF metadata → forward      │ ✅ DBRX / Mixtral        │
  pass skeleton → manual        │ ✅ 50+ more (generated)  │
  tuning                       └──────────────────────────┘
```

## Risk Factors

| Risk | Impact | Mitigation |
|------|--------|------------|
| MAX engine GGUF compatibility gaps | Phase 3B.1 blocked | Maintain llama.cpp fallback; upstream patches to MAX |
| GPU memory insufficient for model size | Phase 3B.4/3B.7 fails | Layer-by-layer CPU offload; quantize to FP8/INT4 for GPU |
| Custom forward passes diverge from upstream | Perplexity regressions | Automated perplexity CI per arch |
| Training on CPU is slow | Phase 2B perceived as useless | Focus on QLoRA (small adapters); leverage MAX GPU |
| Multi-GPU scaling sublinear (<1.5× per GPU) | Phase 3B.3 underwhelming | Optimize all-reduce; use tensor parallelism with async comms |
| CUDA graphs fragile to model changes | Phase 3B.2.4 breaks | Regenerate graphs on model load; fallback to eager mode |
| Apple Metal GPU limited memory (unified) | Phase 3B.6 Apple perf low | Optimize for unified memory; use small models (7B max) |
| Dataset format fragmentation | Phase 2D scope creep | Support top 5 HF formats only |
| MXFP4 in C engine complex | Phase 3A.2 delayed | Ship Q8_0 input quant first (biggest win) |
