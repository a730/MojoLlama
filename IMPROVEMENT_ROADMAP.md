# MojoLlama Improvement Roadmap

**Date:** 2026-05-20
**Baseline:** v0.5.0 — 41K Python lines, 12K C lines, 5.6K HTML, 17 Mojo files, 60+ architectures

---

## Executive Summary

MojoLlama has a strong core (C engine with AVX2+OMP, MoE optimization, concurrent serving) but suffers from **accumulated tech debt** from rapid iteration: 24 stale `.so` files in `legacy-c/`, 15+ single-use debug/inspect scripts, no unit test framework, no packaging, and a CI that only compiles C kernels without running tests. The gap between the CODING-SOUL vision (Mojo+MAX native) and the current reality (Python + C + llama.cpp dependency) is the biggest strategic risk.

---

## Phase 1: Clean Foundation (Weeks 1-2) — HIGH IMPACT, LOW EFFORT

### 1.1 Eliminate Dead Code & Build Artifacts
**Impact:** HIGH | **Effort:** S

| Action | Details |
|--------|---------|
| Remove `kernels/legacy-c/` directory | 24 stale `.so` files, 12 abandoned `.c` files (c_engine.c, cengine2.c, cengine3.c, cengine_batch_clean.c, cengine_batch_full_bak.c, etc.). None are imported by current code. |
| Remove single-use debug scripts | `check_attr.py`, `debug_forward.py`, `inspect_gptoss.py`, `inspect_mxfp4.py`, `compare_q_proj.py`, `q4k_compare.py`, `bench_gpt.py`, `run_gdb_test.py` — all are one-off debugging tools, not production code. |
| Remove stale `.so` from `kernels/` root | `cengine_batch_instr.so`, `gqa_attention.so`, `quant_kernels_omp.so`, `simd_ops.so` — should be built, not committed. Add to `.gitignore`. |
| Archive `numpy_reference.py` | Pure-numpy reference for correctness validation — move to `tools/` or `benchmarks/`, not top-level. |

### 1.2 Add `.gitignore` for Build Artifacts
**Impact:** HIGH | **Effort:** S

```
*.so
*.dylib
__pycache__/
*.pyc
*.egg-info/
.venv/
```

### 1.3 Consolidate Test Infrastructure
**Impact:** HIGH | **Effort:** M

**Current state:** 12 test files (1,532 lines total), all are ad-hoc scripts that `print()` results — no assertions, no pytest, no CI integration. Only `test_prefix_cache.py` uses proper `assert` statements.

| Action | Details |
|--------|---------|
| Add `pytest` to dependencies | `pip install pytest` |
| Convert all `test_*.py` to pytest format | Replace `print("[PASS]")` with `assert`, use `@pytest.fixture` for engine setup, `@pytest.mark.parametrize` for model variants |
| Add `conftest.py` | Shared fixtures: model paths, engine instances, thread counts |
| Add `pytest.ini` or `pyproject.toml [tool.pytest]` | Test discovery config |
| Add test job to `.onedev-buildspec.yml` | Run `pytest src/mojollama/tests/` after kernel compilation |

### 1.4 Add `pyproject.toml` for Packaging
**Impact:** HIGH | **Effort:** S

**Current state:** No `pyproject.toml`, no `setup.py`. Project is not pip-installable.

```toml
[project]
name = "mojollama"
version = "0.5.0"
description = "High-throughput CPU LLM inference engine"
requires-python = ">=3.11"
dependencies = ["numpy", "gguf", "regex", "ftfy", "flask"]

[project.scripts]
mojollama = "mojollama.__main__:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

---

## Phase 2: Test Coverage & Reliability (Weeks 3-5) — HIGH IMPACT, MEDIUM EFFORT

### 2.1 Core Engine Tests
**Impact:** HIGH | **Effort:** M

| Test | What | Coverage Gap |
|------|------|-------------|
| `test_cengine_matmul.py` | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, MXFP4 dequant + matmul against numpy reference | **ZERO** — no quant kernel tests exist |
| `test_batch_forward.py` | Full forward pass on TinyLlama, verify logits match llama.cpp within tolerance | Partial — `test_tinyllama.py` runs but doesn't assert correctness |
| `test_moe_dispatch.py` | Expert routing correctness, top-k selection, shared expert path | **ZERO** — GPT-OSS and Qwen3.6 MoE paths untested |
| `test_ssm_decode.py` | Mamba-2 selective scan kernel (Qwen3.6 hybrid layers) | **ZERO** — SSM path is the most complex and least tested |
| `test_paged_attention.py` | KV cache allocation, page table mapping, physical-logical mapping | **ZERO** — critical for concurrent serving |
| `test_rms_norm.py` | RMS norm + fused clamp against numpy reference | **ZERO** |
| `test_rope.py` | RoPE positional embeddings (full, partial for Qwen3.6) | **ZERO** |

### 2.2 Integration Tests
**Impact:** HIGH | **Effort:** L

| Test | What |
|------|------|
| `test_server_api.py` | All 12 API endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/api/metrics`, etc. |
| `test_concurrent_serving.py` | Multi-worker spawn, shared mmap weights, aggregate throughput |
| `test_quantizer.py` | All 35 quant types, round-trip dequant error < threshold |
| `test_architecture_detect.py` | All 60+ architectures load without error from GGUF metadata |

### 2.3 Property-Based Testing
**Impact:** MED | **Effort:** M

| Action | Details |
|--------|---------|
| Add `hypothesis` library | Property-based testing for quantization: `@given(weights=arrays(...))` |
| Fuzz GGUF parser | Random GGUF-like binary data, verify parser doesn't crash |
| Fuzz token sequences | Random token sequences through forward pass, no NaN/inf allowed |

### 2.4 Memory Safety Tests
**Impact:** HIGH | **Effort:** M

| Action | Details |
|--------|---------|
| Add Valgrind/ASAN to CI | Run C engine under AddressSanitizer: `gcc -fsanitize=address` |
| Test for known heap corruption | README mentions "Qwen3.6 MXFP4 cleanup crash — heap corruption on exit" — write a regression test |
| Leak detection | Run forward pass 1000x, verify RSS doesn't grow |

---

## Phase 3: Performance Optimization (Weeks 6-8) — HIGH IMPACT, LARGE EFFORT

### 3.1 Profile-Guided Optimization
**Impact:** HIGH | **Effort:** M

**Current state:** `profile_qwen36.py` exists but results are not documented or acted upon.

| Action | Details |
|--------|---------|
| Run per-component profiling | `cProfile` + `line_profiler` on Qwen3.6-35B, GPT-OSS-20B, TinyLlama |
| Publish flamegraph | `py-spy` or `perf` output as SVG in repo |
| Identify top 3 bottlenecks | Data-driven, not guessed |

**Expected bottlenecks (educated guess, needs verification):**
1. Python→C ctypes call overhead in `turbo_engine_v7_moe.py` (24 ctypes calls per layer for MoE)
2. Q8_0 input quantization in Python before C matmul (should be in C)
3. MoE expert dispatch routing (softmax + top-k in Python)

### 3.2 Move Q8_0 Quantization to C
**Impact:** HIGH | **Effort:** M

**README explicitly lists this as near-term:** "Q8_0 input quantization in C engine batch_matmul functions — match `quant_kernels_omp` performance (2-4× per-matmul improvement)"

| Action | Details |
|--------|---------|
| Add `quantize_q8_0_avx2()` to `cengine_batch_instr.c` | SIMD quantization of F32→Q8_0 |
| Update `batch_forward()` to accept F32 input, quantize internally | Eliminates Python quantization step |
| Benchmark before/after | Target: 2-4× per-matmul improvement |

### 3.3 Reduce ctypes Call Overhead
**Impact:** HIGH | **Effort:** L

**Current state:** `moe_forward_omp()` already fuses expert calls, but the per-layer loop still does Python→C transitions for attention, norm, RoPE, etc.

| Action | Details |
|--------|---------|
| Fuse entire transformer block into single C call | `block_forward(layer_id, tokens, kv_cache)` replaces 5+ ctypes calls |
| Pre-compute RoPE cos/sin tables in C, not Python | Eliminates per-request table lookup |
| Batch norm + SiLU + residual into single SIMD pass | `fused_norm_silu_residual()` |

### 3.4 Continuous Batching
**Impact:** HIGH | **Effort:** XL

**README lists this as near-term:** "True continuous batching — merge B users' MoE routing into single batched matmuls"

| Action | Details |
|--------|---------|
| Implement token-level scheduling | Not request-level — schedule individual tokens across users |
| Merge MoE routing across batch | Single top-k computation for all active tokens |
| Dynamic KV cache management | PagedAttention already exists, needs integration with scheduler |

---

## Phase 4: Code Quality & Architecture (Weeks 9-12) — MED IMPACT, LARGE EFFORT

### 4.1 Unified Engine Interface
**Impact:** MED | **Effort:** L

**Current state:** Three separate engines with incompatible APIs:
- `TurboEngineV7MoE` (MoE models)
- `TurboEngineV77` (dense models)
- `TurboEngineDSV4` (DeepSeek V4)

| Action | Details |
|--------|---------|
| Define `Engine` ABC | `load()`, `forward()`, `reset()`, `get_logits()`, `get_vocab()` |
| Refactor all engines to implement `Engine` | Single interface for server, studio, benchmarks |
| Factory function | `create_engine(model_path) → Engine` auto-detects MoE/dense/SSM |

### 4.2 Remove llama.cpp Dependency
**Impact:** HIGH | **Effort:** XL

**Current state:** `server.py` hardcodes paths to llama.cpp binaries:
```python
LLAMA_SERVER_PATH = "/tmp/llama.cpp/build/bin/llama-server"
CONVERTER_PATH = "/tmp/llama.cpp/convert_hf_to_gguf.py"
```

`backends.py` routes to llama.cpp as the primary CPU backend.

| Action | Details |
|--------|---------|
| Make llama.cpp optional fallback | `AutoBackend` should prefer native C engine, fall back to llama.cpp |
| Bundle GGUF converter | Replace `convert_hf_to_gguf.py` dependency with native converter in `quantizer.py` |
| Remove hardcoded paths | Use `shutil.which()` or config file |

### 4.3 Error Handling & Input Validation
**Impact:** MED | **Effort:** M

**Current state:** Minimal error handling. `server.py` returns 500 on any exception. `turbo_engine_v7_moe.py` has no input validation.

| Action | Details |
|--------|---------|
| Add custom exception hierarchy | `MojoLlamaError`, `ModelError`, `QuantError`, `BackendError` |
| Validate all user inputs in server | Model path exists, GGUF format valid, quant type supported |
| Graceful degradation | If C engine fails, fall back to numpy (slow but correct) |
| Add request timeout | Prevent hanging on malformed prompts |

### 4.4 Type Hints & Docstrings
**Impact:** LOW | **Effort:** L

**Current state:** 41K Python lines, ~0% type hint coverage. Module docstrings exist but function-level docs are sparse.

| Action | Details |
|--------|---------|
| Add `mypy` to CI | Start with `--ignore-missing-imports`, gradually tighten |
| Add type hints to public APIs | `server.py`, `studio.py`, engine interfaces |
| Add docstrings to all public functions | Use Google or NumPy style consistently |

---

## Phase 5: Feature Gaps (Weeks 13-16) — MED IMPACT, LARGE EFFORT

### 5.1 Complete GGUF Quant Pipeline
**Impact:** MED | **Effort:** L

**Current state:** `quantizer.py` has 35 quant types but no imatrix support.

| Action | Details |
|--------|---------|
| Implement importance matrix generation | `mojollama imatrix -m model.gguf -d calibration.txt` |
| IQ2/IQ3/IQ4 quantization | Import-aware quantization for better quality at low bits |
| Per-tensor quant type selection | Mix Q4_K for attention, Q2_K for FFN |

### 5.2 Speculative Decoding
**Impact:** MED | **Effort:** L

**Current state:** `speculative.py` exists but is not integrated into server.

| Action | Details |
|--------|---------|
| Integrate into server | `--draft-model tinyllama.gguf` flag |
| Shared vocabulary optimization | Current 21% acceptance rate → target 40%+ |
| Benchmark vs baseline | Measure speedup on different model pairs |

### 5.3 Prefix Caching Integration
**Impact:** MED | **Effort:** S

**Current state:** `prefix_cache.py` exists with good tests but is not wired into server.

| Action | Details |
|--------|---------|
| Wire into server request pipeline | Check cache before forward pass |
| Multi-tenant prefix cache | Per-user or shared cache |
| Benchmark cache hit rate | Real-world conversation patterns |

### 5.4 Mojo Kernel Integration
**Impact:** HIGH (strategic) | **Effort:** XL

**Current state:** 17 Mojo files exist but are not compiled or used. `bridge.py` is a "placeholder" per CODING-SOUL.

| Action | Details |
|--------|---------|
| Compile Mojo kernels with nightly | `q4_matmul.mojo`, `norms.mojo`, `attention.mojo` |
| FFI bridge from Python to Mojo | Replace C engine hot paths with Mojo equivalents |
| Benchmark Mojo vs C | Verify Mojo SIMD matches or beats AVX2 C |
| MAX GPU backend | When MAX becomes installable, integrate graph compiler |

---

## Phase 6: Deployment & DX (Weeks 17-18) — MED IMPACT, MEDIUM EFFORT

### 6.1 CI/CD Improvements
**Impact:** HIGH | **Effort:** M

**Current state:** `.onedev-buildspec.yml` has 4 jobs but:
- No test execution (only HTML validation)
- No linting (no `ruff`, `flake8`, `mypy`)
- No benchmark regression detection
- Docker build only on `main` branch

| Action | Details |
|--------|---------|
| Add test job to buildspec | Run `pytest` after kernel compilation, block merge on failure |
| Add lint job | `ruff check src/`, `ruff format --check src/` |
| Add benchmark regression job | Run `bench_batch_gptoss.py`, compare against baseline, fail if >10% regression |
| Build Docker on `dev` too | Catch Docker issues before merge to main |
| Add matrix builds | Test on AVX2, AVX512, NEON (if CI runners support it) |

### 6.2 Developer Experience
**Impact:** MED | **Effort:** M

| Action | Details |
|--------|---------|
| Add `Makefile` at project root | `make build`, `make test`, `make lint`, `make bench`, `make docker` |
| Add `justfile` or `noxfile.py` | Cross-platform task runner |
| Add pre-commit hooks | `ruff`, `mypy`, `pytest` on `git commit` |
| Add `CONTRIBUTING.md` | How to build, test, submit PRs |
| Add architecture decision records (ADRs) | `docs/adrs/` — why C engine over llama.cpp, why multiprocessing over threading, etc. |

### 6.3 Documentation
**Impact:** MED | **Effort:** L

| Gap | Action |
|-----|--------|
| No API docs | Generate with `pdoc` or `sphinx` — publish to `~site/api/` |
| No architecture docs | `docs/architecture.md` — engine design, data flow, memory model |
| No quantization guide | `docs/quantization.md` — which quant type for which use case |
| No troubleshooting guide | `docs/troubleshooting.md` — common errors, heap corruption, OOM |
| README is benchmark-heavy | Add quick-start tutorial, model compatibility table, FAQ |

### 6.4 Distribution
**Impact:** MED | **Effort:** L

| Action | Details |
|--------|---------|
| Publish to PyPI | `pip install mojollama` (with pre-built wheels for Linux x86_64) |
| Build wheels with C extensions | `cibuildwheel` for manylinux2014 |
| Homebrew formula | `brew install mojollama` for macOS |
| Static binary | `PyInstaller` or `cx_Freeze` for single-file distribution |
| Desktop app | Complete `desktop/` Electron app (exists but incomplete) |

---

## Priority Matrix

| Priority | Item | Impact | Effort | Phase |
|----------|------|--------|--------|-------|
| **P0** | Remove dead code + .gitignore | HIGH | S | 1 |
| **P0** | Add pytest + CI test job | HIGH | M | 1-2 |
| **P0** | Add pyproject.toml | HIGH | S | 1 |
| **P1** | C engine quant kernel tests | HIGH | M | 2 |
| **P1** | Server API integration tests | HIGH | M | 2 |
| **P1** | Move Q8_0 quant to C | HIGH | M | 3 |
| **P1** | Fix heap corruption bug | HIGH | S | 2 |
| **P2** | Unified Engine interface | MED | L | 4 |
| **P2** | Remove llama.cpp dependency | HIGH | XL | 4 |
| **P2** | Error handling + validation | MED | M | 4 |
| **P2** | CI lint + benchmark jobs | HIGH | M | 6 |
| **P3** | Mojo kernel compilation | HIGH | XL | 5 |
| **P3** | Continuous batching | HIGH | XL | 3 |
| **P3** | PyPI + wheel distribution | MED | L | 6 |
| **P4** | Type hints + docstrings | LOW | L | 4 |
| **P4** | Speculative decoding integration | MED | L | 5 |
| **P4** | Prefix caching integration | MED | S | 5 |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Mojo pointer APIs don't stabilize | Medium | High | Keep C engine as fallback (CODING-SOUL already plans this) |
| MAX graph compiler never ships | Low | Medium | C engine is already competitive; MAX is optional |
| llama.cpp removes GGUF compatibility | Very Low | High | GGUF spec is open; we can maintain our own reader |
| Heap corruption in MXFP4 path is systemic | Medium | High | ASAN testing in CI, isolate MXFP4 code path |
| Python GIL limits concurrent serving | Low | Medium | Already using multiprocessing; consider `multiprocessing` → `ray` migration |

---

## Metrics to Track

| Metric | Current | Target | How |
|--------|---------|--------|-----|
| Test coverage | ~5% (ad-hoc scripts) | 60%+ | `pytest --cov` |
| CI pass rate | N/A (no tests) | 95%+ | OneDev build status |
| Qwen3.6-35B tok/s | 25.5 (1 user) | 35+ (1 user) | `bench_batch_qwen36.py` |
| Concurrent throughput | 47 agg (10 users) | 70+ agg (10 users) | `bench_concurrent_qwen36.py` |
| Memory leaks | Known (heap corruption) | Zero | ASAN + RSS monitoring |
| Dead code files | 40+ | 0 | `git ls-files` audit |
| Time to first inference | 4s (worker load) | <2s | mmap optimization |
