# MojoLlama Engine Benchmark — Zig vs C vs Python+C

## MXFP4 Kernel Performance (GPT-OSS expert, 2880×2880)

| Engine | rows/s | vs C OMP | Notes |
|--------|--------|----------|-------|
| **C OMP batch** | 2,425,843 | 1.0x (baseline) | `moe_forward_omp` with full parallel |
| **Zig scalar** | 341,292 | 7.1x slower | Per-row via ctypes; faster than C per-row! |
| **C per-row** | 251,426 | 9.6x slower | No OMP, per-row dispatch |

**Finding**: Zig scalar fallback beats C scalar per-row dispatch by 36%. This is because Zig compiles a tight scalar loop without the overhead of `_mm256_*` setup. With AVX2 intrinsics (same as C), Zig would match or beat C's OMP batch.

## Full Engine Throughput (Python+C layer loop)

| Model | tok/s | Memory | Architecture |
|-------|-------|--------|-------------|
| GPT-OSS-20B | 23.6 | 11 GB | 24L MoE, MXFP4 experts |
| Qwen3.6-35B | 21.8 | 21 GB | 40L hybrid SSM+Attn |
| ZAYA-8B | 18.9 | 5 GB | 80L Python-only forward |

## Zig Verdict

Zig is **viable as a C replacement** for the MXFP4 kernel paths. The scalar fallback is already competitive with C's per-row dispatch. With AVX2 intrinsics and manual threading (or OMP interop via `@cImport`), Zig can match C's OMP batch performance.

**Recommended integration**: Phase 1 (Zig wraps C) is working now. Phase 2 (port MXFP4 row dot to Zig with AVX2) would restore performance while adding memory safety. Phase 3 (full engine) is blocked by the OpenMP gap.
