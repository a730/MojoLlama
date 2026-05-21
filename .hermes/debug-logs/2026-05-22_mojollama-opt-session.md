# Debug Log: 2026-05-22
## Goal: MojoLlama beats llama.cpp on every model

## Progress

### Baselines (post-optimizations)
| Model | MojoLlama | llama.cpp | Ratio | Notes |
|---|---|---|---|---|
| Qwen3.6-35B Q4_K_M | 19.6 tok/s | 16.23 | 121% 🏆 | Python engine, fused SSM path |
| GPT-OSS-20B Q4_K_M | 23.6 tok/s | 27.53 | 86% ❌ | Behind — needs C batch forward |
| Qwen3-30B Q4_K_M | 22.6 tok/s | 27.81 | 81% ❌ | Behind — needs C batch forward |

### Changes Applied
1. `batch_matmul()` → `quant_matmul_omp()` — C matmul dispatch now uses Q8_0 activation VPMADDUBSW
2. `ssm_layer_fused()` — new C function fusing SSM decode + gate + element_mul + out + residual
3. `qwen36_batch_forward()` — full C batch forward (370 lines, implemented by OpenCode)
4. `forward_c_batch()` + `_init_c_batch_caches()` — Python wrapper for C forward

### Working
- Fused SSM path in Python engine (saves 3 ctypes calls per SSM layer)
- All models load correctly, C batch caches init cleanly
- Non-Qwen3.6 models gracefully disable C batch

### Not Working
- `forward_c_batch()` segfaults on Qwen3.6 — C function has bug (likely pointer/param mismatch)
- C batch forward needs debugging before it can replace Python loop

### Next Steps
1. Debug qwen36_batch_forward C function (param passing, buffer sizes, pointer alignment)
2. Fix GPT-OSS gap — separate Q/K/V weights, no fused QKV, needs different C path
3. Fix Qwen3-30B gap — MoE bottleneck, 288 OMP regions/layer saturates cores
4. Batch B=4 concurrency
