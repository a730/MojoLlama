# MojoLlama Roadmap

> **Vision:** Production-grade CPU-first LLM platform — inference, fine-tuning, quantization, and serving. Beats llama.cpp on MoE architectures today; becomes a full Unsloth+LLaMA-Factory alternative with MAX GPU acceleration.

**Current:** v0.5.0 — CPU inference engine, 50+ arch support, C batched engine, Studio dashboard, Docker, 2.26× vs llama.cpp on GPT-OSS.

---

## Phase 1: Near-term (now — 2 weeks)

| # | Task | Est. | Priority |
|---|------|------|----------|
| 1.1 | **Q8_0 input quantization in C engine** — match `quant_kernels_omp` perf (2-4× per-matmul improvement) | 3d | Critical |
| 1.2 | **Fix Qwen3.6 MXFP4 cleanup crash** — heap corruption on exit (`corrupted size vs. prev_size`) | 1d | High |
| 1.3 | **True continuous batching** — merge multiple users' MoE routing into single batched matmuls | 5d | High |
| 1.4 | **Falcon forward pass** — parallel attention+FFN, fused QKV | 3d | Medium |
| 1.5 | **Phi-3/4 forward pass** — fused QKV, long RoPE | 3d | Medium |
| 1.6 | **C engine MXFP4 matmul** — native C kernel (replace Python fallback) | 4d | Medium |

**Milestone:** C engine matches/exceeds Python engine throughput. No crashes. Falcon + Phi supported.

---

## Phase 2: Medium-term (2-6 weeks)

| # | Task | Est. | Priority |
|---|------|------|----------|
| 2.1 | **MAX GPU backend** — zero-copy tensor dispatch to NVIDIA GPUs via Modular MAX | 5d | High |
| 2.2 | **Speculative decoding** — TinyLlama draft + Qwen3 target (21% acceptance → improve via shared-vocab pairs) | 5d | Medium |
| 2.3 | **Prefix caching** — LRU with SHA-256 hash (implemented, needs deployment/production wiring) | 2d | Medium |
| 2.4 | **Docker + desktop app** — Electron/Tauri wrapper for MojoLlama Studio | 5d | Medium |
| 2.5 | **PagedAttention in Python engine** — long-context serving for 32K+ sequences | 4d | Medium |
| 2.6 | **DeepSeek V2/V3 forward pass** — Multi-head Latent Attention, DeepSeekMoE | 5d | Low |
| 2.7 | **Concurrent server productionization** — health checks, streaming, metrics endpoint | 3d | Medium |

**Milestone:** GPU acceleration via MAX. Desktop app ships. Production-grade serving.

---

## Phase 3: Long-term (6-12 weeks)

| # | Task | Est. | Priority |
|---|------|------|----------|
| 3.1 | **Full GGUF quant pipeline** — imatrix computation → dynamic 2-bit quantization → GGUF export | 4w | High |
| 3.2 | **MojoLlama Studio as Unsloth+LLaMA-Factory replacement** — LoRA/QLoRA training, dataset mgmt, evaluation, experiment tracking | 5w | High |
| 3.3 | **Perplexity evaluation** — verify output quality matches llama.cpp reference across architectures | 1w | Medium |
| 3.4 | **pip package** — `pip install mojollama` on PyPI with pre-built wheels (manylinux x86_64, aarch64) | 1w | High |
| 3.5 | **Multi-GPU support** — distribute layers across GPUs via MAX engine | 2w | Low |
| 3.6 | **Comprehensive benchmark suite** — automated regression across all architectures vs llama.cpp | 1w | Medium |

**Milestone:** Full ML platform — train, quantize, serve, monitor. pip installable. GPU multi-GPU.

---

## Dependencies

```
Phase 1 (Stability + Arch) ───────────────► Phase 2 (GPU + Production)
                                                  │
Phase 3a (Quant Pipeline) ◄──────────────────────┘
       │
       ▼
Phase 3b (Training Studio) ◄────────────── Phase 3a (GGUF models needed for training)
       │
       ▼
Phase 3c (pip + Polish) ◄───────────────── All phases
```

## Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| GPT-OSS throughput | 62.6 tok/s | 80+ tok/s (C engine) |
| Qwen3.6 throughput | 25.5 tok/s | 40+ tok/s (C engine) |
| Concurrent throughput | 47 tok/s (10 users) | 100+ tok/s (20 users) |
| Models with "Full" status | 7/12 | 12/12 |
| GPU inference | None | MAX backend functional |
| Startup time | `git clone + pip install` | `pip install mojollama` |
| Studio capabilities | Dashboard | Full training + evaluation |
| Docker maturity | Single image | Desktop app + multi-profile |
