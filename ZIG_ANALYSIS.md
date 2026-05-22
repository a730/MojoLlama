# Zig vs C for MojoLlama Engine — Analysis

## Current Architecture

The MojoLlama inference engine is a C AVX2+FMA+OMP pipeline:
- `quant_kernels_omp.c` — quantized matmuls (Q4_K, Q6_K, Q8_0, MXFP4), MoE dispatch
- `cengine_batch_instr.c` — fused batch forward (qwen36, zaya, gemma4)
- `simd_ops.c` — RMS norm, SiLU, softmax
- Combined into `combined_engine.so`, called from Python via ctypes

## Zig's Advantages for This Codebase

| Dimension | C (current) | Zig (proposed) |
|-----------|-------------|----------------|
| **Memory safety** | Manual — buffer overflows common | Built-in bounds checking (release: safety off) |
| **Comptime** | Preprocessor macros only | `comptime` metaprogramming — generate AVX2/AVX512/fallback at compile time |
| **Cross-compilation** | Requires per-target toolchain | Built-in — `zig build -Dtarget=x86_64-linux-gnu` |
| **C ABI** | Natural — C is the ABI | First-class — `callconv(.C)` exports, `@cImport` for C headers |
| **SIMD** | AVX2 intrinsics (`_mm256_*`) | `@cImport` for intrinsics or native `std.simd` (unstable) |
| **Build system** | Makefile/cmake | `build.zig` — single file, hermetic, cross-compiling |
| **Error handling** | Return codes, errno | `!T` error unions, `try`/`catch` |
| **Testing** | No built-in framework | `test` blocks, `zig test` runner |
| **Package management** | None | `zon` packages (immature) |
| **OMP parallelism** | `#pragma omp parallel for` | Not built-in — needs `@cImport` to OMP or manual threading |
| **Ecosystem** | Mature, every library available | Small, growing |
| **Team familiarity** | Universal | Low — small community |

## Key Analysis

### Memory Safety Is the Biggest Win

The MojoLlama C engine has had numerous memory bugs:
- Buffer overflows in workspace allocation (qwen36_batch_forward `S=8192` too small for GPT-OSS)
- Pointer corruption from ctypes type-size mismatch (`cf` vs `c_float`)
- Use-after-free in expert weight pointer arrays
- NaN from uninitialized/out-of-bounds reads

Zig eliminates this class of bugs at compile time with bounds-checked slices, optional pointers, and defined initialization.

### OpenMP Gap Is the Biggest Loss

The C engine gets parallelization for free with `#pragma omp parallel for`. Zig has no built-in equivalent. Options:
1. `@cImport` calling into OMP C code — works but ugly
2. `std.Thread.Pool` — manual work decomposition
3. Simple thread pool — lightweight but needs to be written

For the MoE dispatch (the hottest path), OMP parallelism is critical. Replacing it would be the hardest part.

### Comptime SIMD Dispatch

Zig's `comptime` can auto-generate AVX2/AVX512/fallback paths:

```zig
fn matmul(comptime T: type, w: []const u8, x: []const T, out: []T) void {
    if (comptime @import("builtin").target.cpu.features.isEnabled(.avx2)) {
        avx2_matmul(w, x, out);
    } else {
        scalar_matmul(w, x, out);
    }
}
```

This eliminates runtime dispatch and `#ifdef` chains.

### Feasibility: Phased Adoption

Given OpenMP dependency and codebase size (~3000 lines C), a full rewrite is not justified. **Recommended: Phased Zig incorporation replacing isolated hot spots.**

**Phase 1 (NOW)**: Zig shared library wrapping C (Zig calls C quant_matmul_omp). Zero performance change, adds safety layer for new code, demonstrates the bridge.

**Phase 2**: Port MXFP4 decoder and row dot product to Zig. Currently these are in C with AVX2 intrinsics. A Zig-native version with comptime SIMD dispatch provides the same performance with better safety.

**Phase 3 (future)**: Port individual quantized matmul kernels (Q4_K first — it's simplest). This requires implementing block-level dequant+dot in Zig, matching or beating C AVX2.

**Phase 4 (stretch)**: Port the fused batch forward function. This requires solving the OpenMP gap — either OMP interop or manual thread pool.

## Recommendation: Keep C for Now, Use Zig for New Kernel Development

### Keep C for:
1. The existing quantized matmul kernels (proven, optimized, OMP-parallel)
2. The fused batch forward functions (complex, OMP-dependent)
3. OpenMP-dependent MoE dispatch

### Use Zig for:
1. **New quantization formats** (MXFP8, IQ4_NL, etc.) — Zig's safety catches bugs early
2. **Kernel validation** — Zig test blocks are better than pytest for C kernel correctness tests
3. **The autotune engine** — Zig's comptime enables compile-time kernel selection
4. **CLI tools** — `mojollama quantize`, `mojollama convert` as Zig native binaries (faster startup)

### Concrete Next Step

Add `zig_engine.zig` and `build.zig` to the repo. Include the Zig bridge in CI. Port the MXFP4 dot product to Zig as a proof of concept. Add a `zig build test` step that runs alongside the existing C tests. This incorporates Zig into the project without disrupting the working C engine.
