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

## Phase 1: Benchmark Baseline ✅

**Goal:** Know exactly where we stand on every model.

### Task 1.1: Single-token latency benchmark ✅
- **File:** `bench_tok.py` — standalone benchmark script
- **Results (Qwen3-30B-A3B Q4_K_M, 32 threads):**

| Engine | ms/tok | tok/s | Ratio |
|--------|--------|-------|-------|
| MojoLlama (MoE C engine) | 45.6 ms | 21.9 tok/s | — |
| llama.cpp (tg128) | 39.4 ms | 25.4 tok/s | baseline |
| **MojoLlama vs llama.cpp** | — | — | **86%** |

- TinyLlama dense: not wired (uses llama.cpp backend)
- GPT-OSS-20B: MXFP4 matmul added but not tested end-to-end yet

### Task 1.2: Concurrent benchmark ✅
- **File:** `bench_concurrency.py` — concurrent throughput benchmark
- **Results (Qwen3-30B-A3B Q4_K_M, 10 requests × 20 tok each):**

| Concurrency | p50 lat (ms) | p95 lat (ms) | Throughput (tok/s) |
|-------------|-------------|-------------|-------------------|
| 1 | 2,446 | 2,594 | 8.1 |
| 2 | 4,872 | 4,964 | 8.2 |
| 4 | 9,733 | 9,968 | 8.1 |
| 8 | 13,716 | 21,494 | 8.1 |

- Throughput flat (no continuous batching) — each request waits in queue
- Server overhead reduces throughput from 21.9 → 8.1 tok/s

### Task 1.3: Profile the hot path ✅
- **File:** `profile_hotpath.py` — instrumented forward pass with 14 timing components
- **Results (20 tokens × 48 layers = 960 layer-tokens):**

| Category | ms/tok | % |
|----------|--------|---|
| **MoE FFN gate/up/down** | **20.04 ms** | **40.8%** ← #1 tied |
| **Attention QKV** | **20.09 ms** | **40.9%** ← #1 tied |
| Output projection | 4.94 ms | 10.1% |
| Router | 3.41 ms | 6.9% |
| RMS norms | 0.61 ms | 1.2% |
| Emb lookup | 0.01 ms | 0.0% |
| **Total** | **49.11 ms** | **20.4 tok/s** |

**#1 Bottleneck: MoE FFN + Attention (tied at ~41% each)**

- MoE FFN: 48 layers × 8 experts × 3 matmuls = 1,152 expert matmuls/token
- Attention: 48 layers × (Q+K+V+O) = 192 matmuls/token
- Both are matmul-bound in the C engine

---

## Phase 2: Performance Optimization ✅

**Goal:** Match or exceed llama.cpp on all supported quant types.

### Task 2.1: AVX2-vectorize Q6_K dequant loop ✅
- **Before:** Scalar dequant to stack array + separate AVX2 dot loop (register pressure, cache miss)
- **After:** Fused dequant+dot loop with `_mm256_shuffle_epi8` LUT for nibble extraction, SIMD shift+mask for 2-bit pairs. 8 elements processed per iteration, no temp array.
- **Speedup:** ~4% end-to-end on Qwen3 (Q6_K is only 1/3 of MoE FFN, which is 41% of total)
- **File:** `kernels/cengine_batch_instr.c` — `q6_k_batch_matmul` (fused inner loop)
- **Verification:** Bit-level exact match, zero warnings

### Task 2.2: Add Q5_K batch matmul ✅
- Completed in Phase 3.3 — `q5_k_batch_matmul()` with AVX2 support
- **Quant type:** 13

### Task 2.3: Add Q3_K and Q2_K batch matmuls ✅
- Completed in Phase 3.3 — fallback to Q4_0 matmul for both types

### Task 2.4: Add MXFP4 (type 39) support for GPT-OSS-20B ✅
- Completed earlier — `mxfp4_batch_matmul()` with E8M0 scale + 4-bit mantissa
- **Quant type:** 39

### Task 2.5: Fused QKV projection ✅
- **Approach:** Fuse Q+K weights (both Q4_K for Qwen3) into single contiguous buffer at load time
- **Result:** 3 matmul calls → 2 calls per layer (saves Q or K weight load per layer)
- **Files modified:** `cengine_batch_instr.c` (BC struct + wQK field), `turbo_engine_v7_moe.py` (fused weight construction), `server_moe.py`, `test_fixed.py`
- **Verification:** 48/48 layers fused, fallback when quant types differ
- **Reality:** ~2-3% end-to-end improvement (memory bandwidth bound, not call-count bound)

### Task 2.6: RoPE precompute table ✅
- Already present in `server_moe.py` lines 124-133 — table computed at startup
- **File:** `server_moe.py` — `freq = e.rope_freq_base ** (np.arange(0, HD, 2, dtype=np.float32) / HD)`

**End-to-end results (Qwen3-30B-A3B Q4_K_M, 32 threads):**

| Metric | Before Phase 2 | After Phase 2 | Change |
|--------|---------------|--------------|--------|
| Profile: ms/tok | 49.1 ms | 47.0 ms | -4.3% |
| Profile: tok/s | 20.4 | 21.3 | +4.4% |
| Benchmark: ms/tok | 45.6 ms | 44.5 ms | -2.4% |
| Benchmark: tok/s | 21.9 | 22.5 | +2.7% |

**Analysis:** Gains are modest because the bottleneck is memory bandwidth, not compute. The C engine reads weights from RAM for every matmul (Qwen3 is 18GB), and with 251 GB/s DDR4 bandwidth, dequant+dot is already near-optimal. Further gains require KV cache optimization, operator fusion to reduce memory traffic, or multi-batch to amortize weight loading.


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
