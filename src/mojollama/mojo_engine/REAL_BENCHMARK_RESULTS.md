# MojoLlama Real Inference Benchmark Results
## Date: 2026-05-22

### What Was Built

**Pure Mojo inference pipeline:**
1. **Tokenizer** — vocab.bin loader + token ID → text decoder (strips `▁` space markers)
2. **Weight loader** — reads f16 .bin files via libc syscalls (`open`, `read`, `lseek`)
3. **Forward pass** — RMS Norm, RoPE, GQA attention with KV cache, FFN (SiLU gate + up + down)
4. **Generation loop** — autoregressive, feeds output token back as next input
5. **Benchmark harness** — multiple prompts, measures tok/s per prompt + aggregate

**Files:**
- `tinyllama_benchmark.mojo` — Universal benchmark harness
- `tinyllama_gen.mojo` — Single-prompt generation
- `tl_gen.c` — C reference implementation (for comparison)

---

### Real Benchmark Results (TinyLlama-1.1B f16)

**Hardware:** Threadripper 3970X (32 cores, 4 CCDs)

| Prompt | Tokens | Time (ms) | tok/s |
|--------|--------|-----------|-------|
| 2+2= | 10 | 383 | **26.1** |
| 3*4= | 10 | 344 | **29.0** |
| Hello, how are you? | 10 | 378 | **26.4** |
| def fib(n): | 10 | 511 | **19.6** |
| The capital of France is | 10 | 343 | **29.1** |
| What is 5+7? | 10 | 343 | **29.2** |
| 1. First / 2. Second | 10 | 345 | **28.9** |
| **Aggregate** | **70** | **2650** | **26.4** |

**Comparison: Real vs Synthetic**

| Benchmark Type | tok/s | Notes |
|----------------|-------|-------|
| **Real generation (Mojo)** | **26.4** | Full forward pass + KV cache + decode |
| Real generation (C + OpenMP) | 25.0 | Same algorithm in C |
| Synthetic matmul only | 103 | Random pool, no forward pass logic |
| Old `tinyllama_f16.mojo` | 12.2 | Q4_0 weights (buggy, reading packed nibbles as f16) |

---

### Architecture Support Status

| Model | Real Weights | Architecture | Status |
|-------|-------------|--------------|--------|
| **TinyLlama-1.1B** | ✅ f16 .bin | Llama (GQA, RoPE, RMS) | **Benchmarked** |
| Llama-3.2-1B | ❌ | Llama | Needs extraction |
| Gemma-4-E2B/E4B | ❌ | Gemma (diff FFN) | Needs extraction |
| GPT-OSS-20B | ❌ | GPT-OSS (MoE) | Needs extraction |
| Qwen3.5-2B | ❌ | Qwen (SSM+attention) | Needs extraction |
| ZAYA1-8B | ❌ | Custom | Needs extraction |
| **ZAYA1-8B** (May 22) | ✅ | ZAYA (MoE+Attn, CCA, top-1 expert) | **Benchmarked: 28.6 tok/s Q8_0** |
| Qwen3.6-35B | ❌ | Qwen (SSM) | Needs extraction |
| Qwen3-30B-A3B | ❌ | Qwen (MoE) | Needs extraction |
| ERNIE-4.5-21B | ❌ | ERNIE | Needs extraction |
| GLM-4.7-Flash | ❌ | GLM | Needs extraction |
| TinyStories-33M | ✅ pytorch.bin | GPT-2 (learned pos emb, LayerNorm) | Different arch |

---

### What "Real Numbers" Means

**Real generation includes:**
1. Token embedding lookup (f16→f32 decode)
2. 22 layers of forward pass:
   - RMS Norm (2× per layer)
   - Q/K/V projection (3× f16 matmul)
   - RoPE position encoding
   - GQA attention with KV cache (stores K/V per position)
   - Softmax over attention scores
   - O projection
   - FFN gate + up + SiLU + down
   - Residual connections
3. Final RMS Norm + LM head (f16 matmul, 32000×2048)
4. Argmax sampling
5. Token decode (vocab lookup + UTF-8 output)

**Synthetic benchmark was:**
- Random f16 data in a 512MB pool
- 7 matmuls per layer, no actual forward pass logic
- No KV cache, no softmax, no decode
- Measured only matmul throughput

---

### Next Steps for Other Models

To benchmark other models with real numbers:

1. **Extract weights** from GGUF → .bin using `gguf_extract`
2. **Adapt forward pass** for each architecture:
   - Llama models: same as TinyLlama (different dims)
   - Gemma: uses GEGLU instead of SiLU gate
   - GPT-OSS/Qwen MoE: add router + expert selection
   - Qwen SSM: add Mamba state space layer
3. **Add encode** — implement BPE tokenizer encode in Mojo (currently pre-computed)

---

### Key Files

| File | Purpose |
|------|---------|
| `tinyllama_benchmark.mojo` | Multi-prompt benchmark harness |
| `tinyllama_gen.mojo` | Single-prompt generation |
| `zaya_gen_q8.mojo` | **ZAYA1-8B Q8_0 benchmark (NEW!)** |
| `tl_gen.c` | C reference (verification) |
| `/tmp/vocab.bin` | 32000-entry decode vocab |
| `/tmp/vocab_zaya.bin` | 262147-entry ZAYA decode vocab |
