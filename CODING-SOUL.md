# MojoLlama CODING-SOUL

## Identity

MojoLlama is a hybrid LLM inference engine built with Mojo + MAX.
It runs on CPU (AVX2/AVX512/NEON), GPU (CUDA/SYCL/Vulkan), and hybrid
setups — competing with vLLM/SGLang while also running on machines without
a discrete GPU.

No Python in the hot path. No hand-rewriting kernels for every model.
The architecture IS the advantage.

## Principles

### 1. Mojo + MAX, not C + llama.cpp

We use Mojo and the MAX ecosystem. When Mojo's heap/pointer APIs
mature in nightly, we adopt them. When MAX becomes installable, we
integrate its graph compiler and GPU kernels.

Until then, the Python backend (bridge.py) is a placeholder — not a
compromise. Every line of Python is temporary, clearly marked, and
designed to be swapped for Mojo without changing the op graph.

### 2. Ops, not models

An `AttentionOp` is an `AttentionOp`. It doesn't know or care whether
it's part of Llama, AttnRes, MLA, or something unreleased.

NEW ARCHITECTURE = new graph wiring in `graph/ops.mojo`.
No new kernels. No new backends. No new anything except the DAG.

This is why we beat the vLLM/llama.cpp approach: they write kernels
per architecture. We write ops once, compose them differently.

### 3. The graph is the source of truth

The computation is defined in `graph/ops.mojo`:
```
MatmulOp, AttentionOp, RMSNormOp, RoPEOp, SiLUOp
AddOp, EmbedOp, TransformerBlock
```

These compile in Mojo 0.26.2 with zero errors. They are the invariant.
Backends are swappable: Python → Mojo SIMD → MAX GPU → whatever comes next.

No runtime type dispatch. No `if model_type == "llama"`. The graph
structure IS the model.

### 4. Every line of SIMD is deliberate

The Q4_0 kernel (`kernels/q4_matmul.mojo`) is hand-written for AVX2.
Not because we couldn't use BLAS — because we want the ASM to be
exactly what we specify.

Every `SIMD[DType.float32, 8]` is an explicit 256-bit vector register.
Every `reduce_add()` is a horizontal sum. Every nibble extraction is
a sequence of `& 15`, `>> 4`, `- 8` that maps directly to AVX2
instructions. No surprises, no compiler guessing.

When VNNI/AVX512/CUDA are available, `comptime if` dispatches.
The code changes at compile time, not runtime.

### 5. Future architectures are graph changes, not kernel changes

Attention Residuals: one new op (`DepthAttentionOp`), same backends.
MLA: one new op (`LatentAttentionOp`), same backends.
Whatever comes next: same pattern.

The backends (CPU SIMD, CUDA, Vulkan, SYCL) are stable. The graph evolves.
This is how we stay ahead of llama.cpp and vLLM.

## Non-Negotiable

- Mojo stdlib imports only. No Python in the forward pass.
- The `ops.mojo` graph compiles in current Mojo stable.
- Every kernel has a test with deterministic expected values.
- No hand-writing CUDA kernels for new architectures — the graph compiler handles it.

## What We Ship When

"Works on my machine" is not the standard. The standard is:
- AVX2 today (our dev box)
- AVX512 tomorrow (server-class)
- CUDA/SYCL/Vulkan when MAX is installable
- Any graph architecture (Llama, AttnRes, MLA, ...) = same binary, different input

## The Bet

We're betting that Mojo's pointer APIs stabilize before MAX becomes
installable, and that MAX's graph compiler beats hand-written kernels.
History suggests this is right: specialized kernels don't scale to
50+ architectures. A graph compiler does.

If we're wrong, we rewrite the backends in C. The ops stay.
