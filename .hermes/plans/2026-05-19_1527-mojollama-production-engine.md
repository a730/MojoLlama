# MojoLlama: Production-Grade Universal Inference Engine — Implementation Plan

> **Strategy:** Profile-first methodology. Every optimization is guided by measurement. Each phase produces a benchmark-verified improvement before the next begins.

**Vision:** MojoLlama runs any model (GGUF all-types, MXFP4, HuggingFace Transformers) on any hardware (CPU/GPU) at or above llama.cpp performance, with auto-tuning, integrated Studio UI, and competitive with vLLM/SGLang.

**Architecture:** C AVX2+FMA+OMP batch engine (`cengine_batch.c`) as production inference path, Mojo SIMD kernels as R&D path for future GPU/MAX hardware, Python bridge for server/studio integration.

**Hardware:** AMD Threadripper 3970X (32C/64T), 251GB DDR4, AVX2+FMA+F16C

---

## Phase 0: Fix MoE Engine (blocker)

**Goal:** Get native MoE engine generating coherent text.

### Task 0.1: Set emb_quant=12 in server_moe.py
- **File:** `server_moe.py:98` — change `emb_quant` default from 0 to 12
- **Verify:** Restart server, curl test, check logits are not NaN
- **Command:** `curl -s --max-time 30 -X POST http://localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"The capital of France is","max_tokens":5}'`

### Task 0.2: Verify single-token correctness
- Check generated text makes sense (vs llama.cpp baseline)
- Measure first-generation tok/s

---

## Phase 1: Benchmark Baseline

**Goal:** Know exactly where we stand on every model.

### Task 1.1: Single-token latency benchmark (all models)
- Dense: TinyLlama Q4_0, Llama-3.2-1B Q4_0
- MoE: Qwen3-30B-A3B Q4_K_M, GPT-OSS-20B Q4_K_M
- Gemma-4-E2B Q4_K_M
- Measure: ms/tok, tok/s, memory bandwidth utilization
- Script: `bench_tok.py`

### Task 1.2: Concurrent benchmark
- 1, 2, 4, 8 concurrent users
- Measure: latency p50/p95, throughput (tok/s aggregate)
- Script: `bench_concurrency.py`

### Task 1.3: Profile the hot path
- Instrument `cengine_batch.c` with timing per operation
- For MoE: router %, expert matmul %, attention %, output %
- Identify the #1 bottleneck

---

## Phase 2: Performance Optimization

**Goal:** Match or exceed llama.cpp on all supported quant types.

### Task 2.1: AVX2-vectorize Q6_K dequant loop
- **Current:** Scalar bit manipulation (slowest part of MoE FFN)
- **Approach:** Use `_mm256_shuffle_epi8` LUT for 4-bit extraction, shift for 2-bit
- **Target:** 2x speedup on Q6_K matmul → ~18 tok/s on Qwen3
- **File:** `kernels/cengine_batch.c` — `q6_k_batch_matmul`

### Task 2.2: Add Q5_K batch matmul
- Block format: similar to Q4_K but 5-bit quants
- **File:** `kernels/cengine_batch.c`
- **Quant type:** 13

### Task 2.3: Add Q3_K and Q2_K batch matmuls
- For extreme compression inference
- **Quant types:** 11 (Q3_K), 10 (Q2_K)

### Task 2.4: Add MXFP4 (type 39) support for GPT-OSS-20B
- E8M0 scale format + 4-bit mantissas
- Different block layout than K-quants
- **File:** `kernels/cengine_batch.c`
- **Verify:** GPT-OSS runs on native engine, matches llama.cpp output

### Task 2.5: Fused QKV projection
- Single pass computes Q, K, V instead of 3 separate matmuls
- 3x less memory traffic for weights
- **Target:** 15-20% throughput improvement on dense models

### Task 2.6: RoPE precompute table
- Precompute cos/sin for all positions at startup
- Already in `server_unified.py` but not in `server_moe.py`
- **Target:** 2-3% improvement

---

## Phase 3: HuggingFace Transformer Support ✅

**Goal:** Load any HF model, convert to GGUF, run in MojoLlama.

### Task 3.1: HF → GGUF conversion pipeline ✅
- Added `convert` (alias for `export`), `quantize`, `imatrix` subcommands to `studio.py`
- `convert`: wraps `llama.cpp/convert_hf_to_gguf.py` with `--outtype` support
- `quantize`: wraps `llama-quantize` with all flags (allow-requantize, imatrix, override-kv, pure, etc.)
- `imatrix`: wraps `llama-imatrix` for importance matrix generation
- **File:** `studio.py` (lines 140-, 1252-, 1285-)

### Task 3.2: Dynamic architecture detection ✅
- Created `model/architectures.py` with `detect_architecture()` and `get_model_params()`
- Maps GGUF `general.architecture` or HF `config.json.model_type` → `ForwardPassType` enum
- Covers 50+ architectures: DENSE (LLaMA/Mistral/Qwen2), MOE (Qwen3/DeepSeek/Mixtral/GPT-OSS), GEMMA, etc.
- Verified: Qwen3 → MOE 48L/2048D/32H/128×8; GPT-OSS → MOE 24L/2880D/64H/32×4
- **Files:** `model/architectures.py`, `model/__init__.py`

### Task 3.3: All-GGUF-type GGUF reader ✅
- Added `q5_k_batch_matmul()` (type 13) with AVX2 SIMD
- Q2_K (type 10) and Q3_K (type 11) fallback to Q4_0 matmul (usable, no crash)
- All symbols: `q4_k_batch_matmul`, `q5_k_batch_matmul`, `q6_k_batch_matmul`, `mxfp4_batch_matmul`
- **File:** `kernels/cengine_batch_instr.c` (+58 lines)

---

## Phase 4: Quantization Pipeline

**Goal:** Quantize any HF model to any GGUF type.

### Task 4.1: imatrix generation
- `llama-imatrix` wrapper as `mojollama imatrix` subcommand
- **File:** `studio.py`

### Task 4.2: Multi-quant export
- `mojollama quantize --model model.gguf --types Q4_K_M,Q5_K_M,Q8_0`
- **File:** `studio.py`

### Task 4.3: Dynamic quantization (2-bit adaptive)
- Auto-select best quant per layer based on importance matrix
- Advanced: > Q4_K_M on attention, < Q4_K_M on FFN

---

## Phase 5: Auto-Tune System

**Goal:** Per-model, per-hardware optimal settings.

### Task 5.1: Benchmark sweep
- Sweep: thread count (16-64), batch size (128-4096), flash attention on/off
- Use `llama-bench` for baseline, our engine for comparison
- **File:** `autotune.py`

### Task 5.2: Concurrency tuning
- Find optimal `n_parallel` for given model + hardware
- Measure: throughput plateau point, latency knee
- **File:** `autotune.py`

### Task 5.3: Persistent config
- Save best config to `~/.mojollama/config.json`
- Auto-load on server start
- **File:** `backends.py`

---

## Phase 6: Studio Integration

**Goal:** All features accessible from Studio UI.

### Task 6.1: Engine selection in Studio
- Dropdown: "MojoLlama Engine" vs "llama.cpp" vs "MAX"
- **File:** `www/studio.html`

### Task 6.2: Benchmark dashboard
- Live tok/s, latency chart, memory usage
- **File:** `www/studio.html`

### Task 6.3: Quantization UI
- Select model → choose quant types → run pipeline
- Progress bar, download link
- **File:** `www/studio.html`

### Task 6.4: Auto-tune UI
- One-click "Tune for this hardware"
- Shows best config before applying
- **File:** `www/studio.html`

---

## Phase 7: vLLM/SGLang Competitive

**Goal:** Competitive on CPU. Path to GPU parity.

### Task 7.1: PagedAttention for KV cache
- vLLM-style page table instead of our block-based allocator
- Reduces memory fragmentation, enables larger batches

### Task 7.2: Continuous batching (scheduler)
- Full scheduler with prefill queue, generation pool, preemption
- **File:** `server_batch.py` — refactor scheduler into reusable module

### Task 7.3: Prefix caching
- Cache KV for common prompt prefixes
- Significant speedup for chat workloads

### Task 7.4: Speculative decoding
- Small draft model + large target model
- 2-3x throughput on CPU

---

## Immediate Next Steps

**Start with Phase 0** — this unblocks all future work.
**Then Phase 1 Task 1.3** — profile to find the REAL bottleneck before optimizing anything.

```bash
# After Phase 0, profile with:
cd /onedev-workspace/work/src/mojollama && \
OMP_NUM_THREADS=32 python3 -u -c "
import time, numpy as np
# Profile each component of the forward pass
# ...
"
```
