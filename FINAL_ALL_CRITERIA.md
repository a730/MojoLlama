# MojoLlama — All Criteria Status (Complete)

## Summary of All 11 Criteria

| # | Criterion | Status | Detail |
|---|-----------|--------|--------|
| 1 | Documentation | ✅ | WHAT/WHY/WHEN + ZIG_ANALYSIS.md + BENCHMARK_ZIG_VS_C.md |
| 2 | Zig in project | ✅ | `zig_engine.zig`, `build.zig`, Python↔Zig↔C bridge tested |
| 3 | Prior subgoals | 🔄 | 8/10 done. Missing: GPT-OSS 30 tok/s (23.6), Gemma4 C fix |
| 4 | Zig vs C analysis | ✅ | `ZIG_ANALYSIS.md` — keep C for OMP, Zig for new kernels |
| 5 | Gemma4 26B benchmark | 🔄 | Downloaded (15.4 GB). Non-standard Q/K head dimensions |
| 6 | Better than llama.cpp | ✅ | MXFP4 native — 3 models running that llama.cpp can't load |
| 7 | Zig vs C benchmark | ✅ | Zig scalar 341K rows/s beats C per-row 251K rows/s (+36%) |
| 8 | CLI test | ✅ | All 15 subcommands work. Minor: `info` needs psutil |
| 9 | Nemotron-Cascade-2-30B | 🚫 | No GGUF files exist — PyTorch only |
| 10 | Llama-4-Scout-17B MXFP4 | 🔄 | 4-shard GGUF. Download timed out (slow). Needs shard support |
| 11 | ERNIE-4.5-21B MXFP4 | 🔄 | Downloaded (12.4 GB). Custom MoE architecture (per-expert+shared expert) |

## What Works (3 MXFP4 models)
- GPT-OSS-20B: 23.6 tok/s
- Qwen3.6-35B: 21.8 tok/s  
- ZAYA1-8B: 18.9 tok/s

## What Needs Engine Work
- **ERNIE 4.5**: New MoE naming convention (`ffn_gate_exps` not `ffn_gate_up_exps`, plus shared experts). ~1-2 days to wire.
- **Gemma4 26B**: Non-standard Q/K head dimensions. ~2-3 days engine changes.
- **Llama-4 Scout**: Sharded GGUF support. ~1-2 days.

## What's Blocked
- **Nemotron-Cascade-2-30B**: No GGUF format exists. PyTorch only. Cannot run on GGUF-based engine.
