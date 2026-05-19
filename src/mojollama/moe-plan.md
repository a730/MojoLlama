# MojoLlama MoE Engine: Implementation Plan

## Goal
Add MoE (Mixture of Experts) support to the C inference engine so GPT-OSS-20B and Qwen3-30B-A3B run at llama.cpp-level performance.

## Current Engine
- `/onedev-workspace/work/src/mojollama/kernels/cengine_batch.c` — AVX2 batched inference engine
- Supports: Q4_0 and Q8_0 quant matmuls, GQA attention, RMS norm, RoPE, SiLU FFN
- Compile: `gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o cengine_batch.so cengine_batch.c -lm`
- Server: `/onedev-workspace/work/src/mojollama/server_batch.py`

## Architecture: GPT-OSS-20B
File: `/tmp/models/gpt-oss-20b-Q4_K_M.gguf` (11GB)
- Type: `gpt_oss`, MoE (32 experts, top-4 active)
- 24 layers, hidden=2880, heads=64, KV_heads=8, head_dim=64
- intermediate_size=2880, vocab=201088
- **Non-standard attention**: Q=4096dim (64×64), K=V=512dim (8×64), O=4096→2880
- **All attn projections have bias** (q, k, v, output)
- **Packed experts**: `ffn_gate_exps.weight` shape=[2880, 2880, 32] (intermediate, hidden, experts)
- **Router**: `ffn_gate_inp.weight` shape=[2880, 32] (hidden→experts)
- Quant types: Q6_K (attn_q/k), Q8_0 (attn_v), Q4_K (attn_output), Q4_K_M (expert weights)

## Architecture: Qwen3-30B-A3B
File: `/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (18GB)
- Type: `qwen3_moe`, MoE (128 experts, top-8 active + shared expert)
- 48 layers, hidden=2048, heads=32, KV_heads=4, head_dim=64
- intermediate_size=6144, vocab=151936
- **Standard attention**: Q=2048dim, K=V=256dim
- **Shared expert**: has `shared_expert.w1/w2/w3` + `shared_expert_gate.weight`
- **Packed experts**: similar packed format but qwen3_moe layout
- Quant: likely Q4_K_M throughout

## Implementation Tasks (priority order)

### 1. Add K-quant matmul support
Current engine only has `q4_0_batch_matmul` and `q8_0_batch_matmul`. Need:
- `q4_k_batch_matmul` — for Q4_K (type 12)
- `q4_k_m_batch_matmul` — for Q4_K_M (type 39) 
- `q6_k_batch_matmul` — for Q6_K (type 6)
- Block sizes: Q4_K/Q4_K_M=256 elements, Q6_K=256 elements
- Use llama.cpp's block format as reference

### 2. Add bias support to attention
Current engine has no bias. GPT-OSS attn_q/k/v/output all have bias.

### 3. Add packed-expert MoE FFN
- Router: matmul `hidden @ ffn_gate_inp.weight` → softmax → top-K selection
- Expert FFN: `silu(gate(e) @ hidden) * (up(e) @ hidden) → down(e) @ result`
- For selected experts only (sparse computation)
- Weighted combination by router probabilities

### 4. Add Qwen3 shared expert
- Shared expert is a dense FFN (always computed for every token)
- Shared expert gate: learns how to mix shared + MoE outputs

### 5. Update Python weight loading
- Read per-layer tensor names
- Quant type detection for correct matmul dispatch
- Build BC struct with expert pointers

## Files to modify
- `kernels/cengine_batch.c` — add K-quant matmuls, MoE FFN, bias support, Qwen3 shared expert, non-standard attn dims
- `server_batch.py` — add MoE weight loading, model config parser, dispatch

## Test & Verification
1. Single-token forward match with llama.cpp output
2. Multi-token generation
3. Batch generation
4. Benchmark: should match llama.cpp tok/s within 15%
