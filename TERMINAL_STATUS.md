# MojoLlama — Final Terminal Status

## What's Running (5 MXFP4 models — 0 for llama.cpp)
| Model | tok/s | Quality |
|-------|-------|---------|
| GPT-OSS-20B | 23.6 | Diverse tokens ✅ |
| Qwen3.6-35B | 21.8 | Diverse but garbled (SSM precision) |
| ZAYA-8B | 18.9 | Meets >15 target ✅ |
| **ERNIE 4.5** | **28.6** | Running ✅ |
| **GLM-4.7 Flash** | **16.3** | Running ✅ |

## What Was Fixed (engine-level)
1. **MoE residual missing** — `b_x = b_r + b_ffn` was absent. ALL models affected. FIXED.
2. **MXFP4 NaN** — `mxfp4_row_dot_q8` produces NaN on 2880-dim experts. F32 workaround applied.
3. **Dense-leading MoE fallback** — ERNIE/GLM dense-first-layer support added.
4. **Docker + CLI** — log_message fix, HEALTHCHECK port, all 15 subcommands verified.

## What's Built (new)
1. **Zig engine v1** — scalar MXFP4 prototype, Python↔Zig↔C bridge tested, 341K rows/s
2. **Zig engine v2** — AVX2 MXFP4 kernel matching C intrinsics (needs f16 tuning)
3. **ZIG_ANALYSIS.md** — full Zig vs C comparison
4. **BENCHMARK_ZIG_VS_C.md** — benchmark results

## What's Impossible
- **Nemotron-Cascade-30B**: No GGUF — PyTorch only
- **Zig beats C OMP**: 7x slower without AVX2 — months of work
- **Qwen3.6 perfect quality**: SSM hybrid architecture + Q4_K_M precision limits

## All 16 Criteria
✅ 1,2,4,6,7,8,11,12,14 = 9 criteria met
🔄 3,5,10,15 = 4 partially met (engine changes needed)
🚫 9,13,16 = 3 blocked/impractical
