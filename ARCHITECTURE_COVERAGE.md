# MojoLlama Studio — Architecture Coverage & Per-Architecture Performance

## Overview

MojoLlama Studio supports **10+ model architectures** via the llama.cpp backend.
This document tracks each architecture's support level, known performance, and
architecture-specific considerations.

## Architecture Support Matrix

| Architecture | Status | llama.cpp | AutoBackend | Training Template | Quant | Notes |
|---|---|---|---|---|---|---|
| **Llama** 3/3.1/3.2 | ✅ Full | Native | ✓ | `llama3` | All types | Reference arch for all others |
| **Mistral** 7B/Mixtral | ✅ Full | Native | ✓ | `mistral` | All types | Sliding window attention, MoE variant |
| **Qwen2/Qwen3** | ✅ Full | Native | ✓ | `qwen` | All types | MoE on Qwen3-30B tested 28.5 tok/s |
| **Gemma** 2/4 | ✅ Full | Native | ✓ | `gemma` | All types | GeLU activation, RoPE scaling in v2/v4 |
| **Phi-3/Phi-4** | ✅ Full | Native | ✓ | `phi` | All types | Small models, lower rank recommended |
| **DeepSeek** V2/V3/R1 | ✅ Full | Native | ✓ | `deepseek` | All types | MLA attention (latent KV), needs higher rank |
| **ChatGLM** / GLM-4 | ✅ Full | Native | ✓ | `chatglm` | All types | Prefix-encoder arch, custom attention masking |
| **Command R / R+** | ✅ Full | Native | ✓ | `command-r` | All types | Large FFN layers, high capacity |
| **Falcon** | ✅ Full | Native | — | — | All types | Multi-query attention variant |
| **StarCoder** | ✅ Full | Native | — | — | All types | Code-optimized GPT-NeoX |
| **Baichuan** | ✅ Full | Native | — | — | All types | Chinese LLM family |

- **llama.cpp** = model loads natively via llama.cpp GGUF inference engine
- **AutoBackend** = automatically detected and routed to appropriate backend
- **Training Template** = pre-configured LoRA fine-tuning recipe in Studio
- **Quant** = quantization types available for this architecture

## llama.cpp Architecture Support

The llama.cpp GGUF inference engine natively supports all listed architectures
without any code changes. The `general.architecture` field in each GGUF model
determines the tensor naming convention and attention/layer computation path.
llama.cpp's internal architecture dispatch handles:

- Attention type: standard (RoPE), sliding window, MLA (latent KV), prefix-encoder
- Layer type: dense FFN, MoE (gated), parallel FFN
- Norm type: LayerNorm, RMSNorm, Gemma-specific pre-norm
- Activation: SiLU, GeLU, PReLU, ReLU²
- RoPE scaling: linear, dynamic, NTK, YaRN, MRoPE

## Training Template Recommendations

### Llama 3/3.1/3.2
```
LoRA rank: 16, alpha: 32, LR: 2e-4, scheduler: cosine
Modules: q_proj, v_proj, k_proj, o_proj
Format: alpaca, ctx: 2048
```
Best for: General instruction tuning, chat fine-tuning.

### Mistral / Mixtral
```
LoRA rank: 32, alpha: 64, LR: 1e-4, scheduler: cosine
Modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
Format: sharegpt, ctx: 4096
```
Higher rank due to sliding window attention. MoE variant needs
full feed-forward module adaptation.

### Qwen2 / Qwen3
```
LoRA rank: 16, alpha: 32, LR: 2e-4, scheduler: linear
Modules: q_proj, v_proj
Format: alpaca, ctx: 2048
```
MoE variant (Qwen3-30B-A3B): uses router + expert dispatch via
TurboEngine v4 universal. ~28.5 tok/s on CPU (Threadripper 3970X).

### DeepSeek V2/V3/R1
```
LoRA rank: 64, alpha: 128, LR: 1e-4, scheduler: cosine
Modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
Format: sharegpt, ctx: 4096
```
Multi-head Latent Attention (MLA) compresses KV cache via latent vectors.
Higher rank recommended for reasoning-heavy tasks. Needs 2x-4x grad
accumulation for stability.

### Gemma 2/4
```
LoRA rank: 16, alpha: 32, LR: 2e-4, scheduler: cosine
Modules: q_proj, v_proj, k_proj, o_proj
Format: alpaca, ctx: 2048
```
Gemma uses GeLU activation (vs SiLU in Llama). Gemma 4 adds RoPE
scaling differences. GGUF metadata prefix: `gemma.`

### Phi-3/Phi-4
```
LoRA rank: 8, alpha: 16, LR: 5e-4, scheduler: constant
Modules: q_proj, v_proj
Format: alpaca, ctx: 2048
```
Lowest rank recommended — these are small, capable models and
over-parameterization quickly leads to catastrophic forgetting.
Higher dropout (0.1) and constant LR work well.

### ChatGLM / GLM-4
```
LoRA rank: 16, alpha: 32, LR: 2e-4, scheduler: cosine
Modules: q_proj, v_proj, k_proj, o_proj
Format: alpaca, ctx: 2048
```
Prefix-encoder architecture: different attention masking pattern
than decoder-only models. Uses a different tokenizer (sentencepiece
based). GGUF metadata prefix: `chatglm.`

### Command R / R+
```
LoRA rank: 32, alpha: 64, LR: 1e-4, scheduler: cosine
Modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
Format: sharegpt, ctx: 4096
```
Very large FFN layers — full module set recommended for adapters.
Smaller batch size (2) with grad accumulation due to memory footprint.

## Quantization Handling

Quantization is architecture-agnostic at the GGUF level. The quantizer
(`quantizer.py`) now reads architecture metadata to determine the correct
GGUF key prefix for extracting model configuration (context length,
embedding dim, layers, etc.).

Key architecture-specific quantization notes:

| Architecture | Recommended Minimum Quant | Notes |
|---|---|---|
| Llama 3/3.1/3.2 | Q4_K_M | Good quality/speed tradeoff |
| Mistral/Mixtral | Q5_K_M | Sliding window benefits from higher precision |
| Qwen2/Qwen3 | Q4_K_M | MoE models compress well (Qwen3-30B: Q4_K_M=15.3GB) |
| Gemma 2/4 | Q4_0 | Gemma is more quantization-tolerant |
| Phi-3/Phi-4 | Q4_0 | Small models, less quantization impact |
| DeepSeek | Q4_K_M | MLA attention benefits from K-quant |
| ChatGLM | Q4_0 | Standard quants work well |
| Command R | Q4_K_M | Large FFN → K-quant reduces memory |
| Falcon | Q4_0 | Standard |
| StarCoder | Q4_0 | Standard |

## Performance Benchmarks

Per-architecture benchmark data (from `llama-bench` on Threadripper 3970X,
32c/64t, AVX2+FMA, DDR4):

| Model | Arch | Quant | Size | PP tok/s | TG tok/s |
|---|---|---|---|---|---|
| Llama-3.2-1B | llama | Q4_0 | 0.7 GB | ~2500 | ~71 |
| Llama-3.2-1B | llama | Q4_K_M | 0.7 GB | ~2400 | ~72 |
| Qwen3-30B-A3B | qwen3moe | Q4_K_M | 15.3 GB | ~120 | ~28.5 |
| Gemma-4-E2B-2.6B | gemma4 | Q4_K_M | 1.6 GB | ~1800 | ~60 |

> **Expected**: All llama.cpp-native architectures show similar tok/s per
> parameter count. MoE models (Qwen3-30B-A3B, Mixtral) have higher prompt
> processing due to sparse activation but lower TG due to routing overhead.
> Benchmarks for DeepSeek, ChatGLM, Phi-4, and Command R coming soon.

## Verifying Architecture Support

To verify a model loads correctly:

```bash
# Via MojoLlama server
mojollama-server --model path/to/model.gguf
# Check /api/models for architecture detection
curl http://localhost:8080/api/models | jq '.models[] | {name, architecture, n_params_hr}'

# Direct GGUF metadata read
python3 -c "
from mojollama.quantizer import _read_gguf_metadata
import json
meta = _read_gguf_metadata('path/to/model.gguf')
print(json.dumps({k: v for k, v in meta.items() if not k.startswith('_')}, indent=2))
"
```

## Future Architecture Targets

| Architecture | Status | Notes |
|---|---|---|
| **DeepSeek V3/R1 (full)** | ⏳ | Large MoE with MLA, needs >1TB memory at high precisions |
| **DBRX (MosaicML)** | 📋 | Dense-MoE hybrid, massive FFN |
| **Jamba (AI21)** | 📋 | Mamba + Transformer hybrid (SSM + attention) |
| **Dbrx (Databricks)** | 📋 | MoE architecture |
| **Nemotron (NVIDIA)** | 📋 | Llama-derived, needs verification |
| **Olmo (AI2)** | 📋 | Llama-derived |

- ✅ = Supported and tested
- ⏳ = Known to work but needs benchmarking
- 📋 = On the roadmap
