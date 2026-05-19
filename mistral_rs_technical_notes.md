# mistral.rs Technical Research Notes for MojoLlama

## 1. GGUF File Loading and Parsing

### Directory: `mistralrs-core/src/gguf/`

**Architecture enum** (`mod.rs`):
- `GGUFArchitecture` maps string names (from `general.architecture` metadata key) to Rust enum variants: Llama, Mpt, Gptneox, Falcon, Qwen2, Phi3, etc.
- Uses `strum::EnumString` with case-insensitive matching via `from_value()`.
- Model loading pipeline checks this to determine which model struct to construct.

**Content struct** (`content.rs`):
- `Content<'a, R>` is the central GGUF file abstraction. It wraps:
  - `contents: Vec<gguf_file::Content>` — parsed GGUF metadata + tensor info from Candle's `gguf_file`
  - `readers: &'a mut [&'a mut R]` — paired file readers (supports sharding)
  - `arch: GGUFArchitecture` — the detected architecture
  - `all_metadata: HashMap<String, Value>` — merged metadata across all shards
- **Construction**: `Content::from_readers(readers)` calls `gguf_file::Content::read(reader)` on each reader, collects TensorInfos, validates shard consistency via `split.count` metadata.
- **Tensor access**: 
  - `tensor_info(name)` — searches across shards (no I/O)
  - `tensor(name, device)` — reads tensor data from the correct shard via `tensor_info.read(reader, data_offset, device)`, returns a `QTensor` (Candle's quantized tensor type)
  - `has_tensor(name)` — existence check across shards
- **Key insight for Mojo**: The GGUF format is fundamentally a key-value metadata store + a flat array of tensor data with offsets. A Mojo equivalent would memmap the file, parse the header once, and use pointer arithmetic + dtype-specific strides to read values. No need for a full parser — just a header reader.

**Known DTypes** (from Candle's `GgmlDType`):
F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K.

**Tokenizer conversion** (`gguf_tokenizer.rs`): Converts GGUF tokenizer data → HuggingFace tokenizer JSON format. Not directly relevant to Mojo.

**Loading pipeline** (`pipeline/gguf.rs`):
- The GGUF pipeline loader reads `Content` metadata to populate model config (hidden_size, n_layers, vocab_size, etc.)
- Maps `GGUFArchitecture` -> specific `ModelParams` struct
- Builds individual layer tensors by name pattern: `model.layers.{i}.attention.wq.weight`
- Uses Candle's `VarBuilder` path system (`.pp("sub")`) to match PyTorch naming

### Mojo Translation Patterns:
1. GGUF header parsing can be a simple struct-deserialize from bytes (magic + version + metadata_kv_count + tensor_count)
2. Tensor metadata is (name, shape, dtype, offset) — just read the header and build a lookup table
3. Sharding means multiple files — maintain a list of `(file_handle, tensor_infos)` pairs
4. The `general.architecture` string drives model dispatch — Mojo can use a match or dictionary

---

## 2. Candle Tensor Operations (ops.rs)

### File: `mistralrs-core/src/ops.rs` (857 lines)

**Design pattern**: Extend `Tensor` with trait-based custom ops.

**Kernel dispatch architecture**:
1. **Trait pattern**: Define a trait (e.g., `TopKLastDimOp`), implement on `Tensor`, with `#[cfg(feature = "cuda")]` gates for CUDA fast paths.
2. **Fallback chain**: CUDA → CPU fallback (full sort, then narrow).
3. **CustomOp1 / CustomOp2 interface**: Candle's mechanism for user-defined operations that get JIT-compiled on CUDA.

**Example — TopK dispatch**:
```rust
impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        #[cfg(feature = "cuda")]
        if self.device().is_cuda() {
            return cuda_topk(self, topk);  // Custom CUDA kernel
        }
        // Fallback: full sort then narrow
        let (values, sorted_indices) = self.sort_last_dim(false)?;
        // ...
    }
}
```

**CUDA kernel integration pattern** (from `cuda_topk`):
```rust
fn cuda_topk(input: &Tensor, k: usize) -> Result<TopKOutput> {
    let input = input.contiguous()?;
    let (storage, _layout) = input.storage_and_layout();
    let storage = match &*storage { Storage::Cuda(s) => s, _ => bail!() };
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(inp.stream()),
        // ...
    };
    // Allocate output on device
    let values_dst = unsafe { dev.alloc::<f32>(out_elem_count) }?;
    // Call the C function
    unsafe { ffi::topk_f32(src_ptr, values_ptr, indices_ptr, nrows, ncols, k, stream); }
    // Wrap device pointers back into Tensor
    let storage = CudaStorage { slice: CudaStorageSlice::F32(values_dst), device: dev.clone() };
    Tensor::from((Storage::Cuda(storage), Shape::from_dims(&out_dims)))
}
```

**Key patterns**:
- `device_ptr()` gets the raw CUDA pointer from a `CudaSlice`
- `ffi::` module contains `extern "C"` declarations for CUDA kernel launchers
- Stream management via `dev.cuda_stream().cu_stream()`
- Dtype dispatch via match on `DType` enum (BF16, F16, F32, etc.)
- Two-phase: extract slice pointers → call kernel → wrap results back into tensors

**Fused Operations**:
- `cuda_topk_softmax` — single kernel that does topk + softmax in one pass (eliminates intermediate allocation)
- `mul_and_act` — fused gated activation for FFN (SiLU, GELU, ReLU) with dispatch to `mistralrs_quant::fused_glu`

**CustomOp1 for `ArgSort`**:
```rust
impl candle_core::CustomOp1 for ArgSort {
    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<...> { panic!("not impl") }
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<...> {
        // Extract pointers from CudaStorage, call ffi::asort_asc_f32 etc.
    }
}
// Used via: tensor.apply_op1_no_bwd(&ArgSort { ... })
```

### Mojo Translation:
1. Use `trait` pattern for ops extensions — exactly like Candle/Rust
2. In MAX/accelerator kernels, the dispatch is implicit (no `#[cfg]` needed — `@parameter` if/elif handles it)
3. The "get device pointers → call kernel → wrap result" pattern maps to MAX's `Buffer` API
4. Fused ops (topk+softmax, mul+act) are natural in Mojo with `fn` parameterization

---

## 3. Quantized MatMul Dispatch

### File: `mistralrs-quant/src/gguf/mod.rs` (GGUF matmul wrapper)
### File: `mistralrs-quant/src/gguf/fast_mmvq.rs` (small-batch CUDA path)
### File: `mistralrs-quant/src/gguf/fast_mmq.rs` (large-batch CUDA path)
### File: `mistralrs-quant/src/gguf/cuda.rs` (indexed MoE)
### File: `mistralrs-quant/src/gguf/cpu.rs` (CPU fallback)

**Architecture**: `QuantMethod` trait is the central abstraction:

```rust
pub trait QuantMethod: Send + Sync + Debug + QuantizedSerde {
    fn forward_raw(&self, a: &Tensor) -> Result<Tensor>;   // matmul
    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor>; // MoE
    fn quantized_act_type(&self) -> Option<DType>;         // activation casting hint
    fn get_qtensor(&self) -> Option<&QTensor>;             // for direct kernel access
    // ...
}
```

**GgufMatMul** (`mistralrs-quant/src/gguf/mod.rs`):
- Wraps `QMatMul` (Candle's quantized weight container) + optional bias
- `forward_raw` implements the **dispatch decision tree**:
  1. Try fast path: `try_fast_forward()` (CUDA only)
  2. Fallback: Cast to F32 → `self.w.forward()` (Candle's generic quant matmul) → cast back

**Fast path selection** (batch-size dependent):
```rust
fn try_fast_forward(&self, a: &Tensor) -> Result<Option<Tensor>> {
    // Batch 1-8: use MMVQ (decode kernel - optimized for single tokens)
    if (1..=MMVQ_MAX_BATCH).contains(&flat_batch) {
        return fast_mmvq::plain(q, a);
    }
    // Batch > 8: use MMQ (prompt kernel - tiled matmul)
    if flat_batch > MMVQ_MAX_BATCH {
        return fast_mmq::plain(q, a);
    }
}
```

**CUDA Kernel Naming Convention** (from FFI):
- `launch_mmvq_gguf_{Q4_0|Q4_1|Q5_0|Q5_1|Q8_0|Q2K|Q3K|Q4K|Q5K|Q6K}_{bf16|f16|f32}_plain`
- `launch_mmvq_gguf_quantize_q8_1_{bf16|f16|f32}` (quantize activation to Q8_1)
- `launch_mmq_gguf_{quant_type}` with fixup/SM params for tiled kernel
- `launch_indexed_moe_forward_{qtype}_q8_1` for MoE

### How the fast MMVQ works (decode path):

1. **Two-phase approach**:
   - Phase 1: Quantize activation to Q8_1 (using `launch_mmvq_gguf_quantize_q8_1_*`)
   - Phase 2: Compute matmul with quantized weights using Q8_1 launchers

2. **Scratch buffer management**: Uses `OnceLock<Mutex<HashMap<DeviceId, WorkspaceSlot>>>` for caching device-side buffers across calls.

3. **Output dtype matches input dtype** (BF16→BF16, F16→F16, F32→F32).

### How the fast MMQ works (prefill path):

1. **More complex**: Needs device properties (SM count, compute capability, shared memory per block, warp size)
2. Also two-phase: quantize activations to `block_q8_1_mmq` format, then tiled matmul
3. Different `DsLayout` per quant type: D4, DS4, D2S6 (affects how intermediate sums are structured)
4. Uses "Stream-K" decomposition for load balancing across SMs

### CPU Indexed MoE (`mistralrs-quant/src/gguf/cpu.rs`):

Simple fallback: dequantize all weights → create `UnquantLinear` → use its `gather_forward`.

### CUDA Indexed MoE (`mistralrs-quant/src/gguf/cuda.rs`):

1. Quantize input to Q8_1
2. Dispatch to dtype-specific CUDA kernel (`launch_indexed_moe_forward_*`)
3. Each kernel handles the dequant-interleaved matmul on-the-fly
4. Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K

### Mojo Translation:

1. The `QuantMethod` trait maps directly to a Mojo trait (or struct with function pointer fields for dynamic dispatch)
2. The batch-size-aware dispatch (MMVQ vs MMQ) is critical — decode vs prefill need different kernels
3. The Q8_1 quantize-then-matmul pattern works well with Mojo's SIMD:
   - Activations continuously quantized to Q8_1
   - Weights stored in quantized block format
   - Dot product dequantizes on the fly
4. Mojo's `@parameter` if/elif replaces Rust's `#[cfg(feature = "cuda")]` gates
5. The workspace allocation cache pattern (`OnceLock<Mutex<HashMap<>>>`) is a must for performance

---

## 4. K-Quant Dequantization Kernels

### The GGUF K-Quant Family: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K

These are "importance-tuned" block quantization formats from llama.cpp:

**Common structure (256-element blocks)**:
- `Q2_K`: 2-bit + scales (super-block + sub-block)
- `Q3_K`: 3-bit + scales
- `Q4_K`: 4-bit + scales
- `Q5_K`: 5-bit + scales
- `Q6_K`: 6-bit + scales (no sub-block scales)

**Key characteristics**:
- Block size = 256 (vs Q4_0/Q8_0 which are 32)
- Two-level hierarchy: super-block scale (f16 or f32) + sub-block scales (6-bit or 8-bit)
- The "K" prefix indicates importance-based bit allocation
- CUDA kernels for these have `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K` in their names

**FFI pattern** (from `ffi.rs`):
- Each K-quant type has its own kernel launcher (e.g., `launch_mmq_gguf_q2_k`)
- The kernels are in `*.cu` files compiled with nvcc, linked as `extern "C"` functions
- Signature varies:
  - MMVQ: `(weight_ptr, scratch_ptr, out_ptr, k, nrows, stride_col_y, stride_col_dst, b_size, stream)`
  - MMQ: `(fixup_ptr, weight_ptr, scratch_ptr, out_ptr, ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, cc, nsm, smpbo, warp_size, stream)`

### What the CUDA Kernel Actually Does (dequant + compute):

Conceptually (from llama.cpp's kernel patterns):
```cuda
// For each block of 256 weight elements:
float block_scale = super_block_scale[block_idx];
// For each sub-block of 16 elements:
float sub_scale = sub_block_scales[sub_block_idx];
// Dequantize elements: x[i] = (quant_value - midpoint) * sub_scale * block_scale
// Compute dot product with Q8_1 quantized activation
```

The key insight for Mojo: **Dequantization is interleaved with computation** — you never fully dequantize, you just load quant values, scales, and compute on the fly. This maps perfectly to Mojo's SIMD vectors.

### Mojo Translation for K-Quant:

1. **Block layout**: Each K-quant type is a SIMD-friendly data structure
   - Q8_1 activation: (16 int8 values + half-scale) × 2
   - Q4_0 weight: (32 int4 values + half-scale)
   - Q4_K weight: 256 values in 4-bit + super-scale + 8 sub-block scales (6-bit each)

2. **Dequantize-and-dot pattern**:
   ```mojo
   fn dot_product_k_quant[type: DType, simd_width: Int](weight_ptr, act_ptr) -> F32:
       # Load block of quantized weights
       # Load block of Q8_1 activations
       # Dequantize weights on the fly using SIMD
       # Compute fused multiply-add, accumulate in F32
       # Return sum
   ```

3. **Block-size dimension**: 64-256 elements per block for K-quant (must handle in SIMD loops)

---

## 5. Paged Attention

### Directory: `mistralrs-paged-attn/src/cuda/`

**Architecture**:
- Uses vLLM's attention template heavily adapted
- Supports V1 (single-iteration) and V2 (split across two kernel launches for softmax) paged attention
- Separate .cu files per dtype: `pagedattention_v1_{bf16,f16,f32}.cu`, same for v2
- FlashInfer-based MLA decode kernels for Multi-Latent Attention (deepseek-style)

**Kernel dispatch** (from `paged_attention.rs` backend):
```rust
fn cuda_fwd_t<T>(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
    // Check cache dtype (F16=0, BF16=1, F32=2, FP8=3)
    // Extract: key_cache, value_cache, block_tables, context_lens
    // Dispatch to: paged_attention_v1_{f32|f16|bf16}(...)
    //          or: paged_attention_v2_{f32|f16|bf16}(...)
}
```

**Key structure** (from the `.cuh` template):
```cpp
template<typename T, int BLOCK_SIZE, int NUM_WARPS>
__global__ void paged_attention_v1_kernel(...) {
    // Warp-level: each warp handles one head
    // Q tile in registers
    // K/V tiles from block tables (page table indirection)
    // Online softmax (safe softmax with max tracking)
    // Final output: each warp writes its head's results
}
```

**Online softmax** (vital pattern for Mojo):
- Keeps running max `m` and sum `s`
- For each new element: `m_new = max(m_old, s_val)`, renormalize previous, add new
- `m * s * v` update pattern
- This is the exact same pattern used in CPU flash attention in `cpu.rs`

**CPU Flash Attention** (`mistralrs-core/src/attention/backends/cpu.rs`):
- Pure Rust implementation with Rayon parallelism
- Uses `vec_dot_f32` — manually unrolled 4-element dot product
- Ping-pong prefetch via `_mm_prefetch` (x86) and `prfm` (ARM)
- Online softmax (same v1.0/v2.0 scalar kernel formulation)
- Tile-based with `TILE_KV = 16`
- Thread QoS hint on macOS via `pthread_set_qos_class_self_np`

### Mojo Translation:

1. **Page table indirection**: KV cache access uses block table (logical → physical block mapping). In Mojo/MAX, this is a simple index indirection: `block_tables[batch_pos, head_pos]` gives the physical block number.

2. **Warp-level decomposition**: Each warp handles one attention head. In Mojo/Max, this maps to `@parameter` groups executing in SIMT fashion.

3. **Online softmax**: The scalar formulation (m/s/v) is the same. In Mojo, this composes cleanly with SIMD reductions.

4. **F32 accumulation**: All CUDA attention variants accumulate in F32 regardless of input dtype — matches Mojo's `simd[DType.f32]` pattern.

---

## 6. Relationship Between Candle's Kernel Approach and Mojo SIMD

### Candle's Architecture:

1. **Trait-based tensor ops**: CustomOp1/CustomOp2 for extendable operations
2. **Device-specific implementations**: CPU (Rayon), CUDA (cudarc), Metal (objc2-metal)
3. **Staged dispatch**:
   - Feature gates (`#[cfg(feature = "cuda")]`)
   - Runtime checks (`if device.is_cuda()`)
   - Fallback: trait method with default implementation
4. **CUDA integration**: `extern "C"` FFI → compiled .cu files → launches

### Mojo Equivalent (from MojoLlama perspective):

1. **`@parameter` if/elif**: Replaces both compile-time feature gates AND runtime device checks — the compiler specializes for each path
2. **`simd[DType, width]`**: Replaces vector types and half precision handling
3. **`Buffer`/`NDBuffer`**: Replaces Candle's `Storage` + `Layout` + `Tensor` separation
4. **`@always_inline`**: Replaces C++ template inlining in CUDA kernels
5. **MAX accelerator**: Backend-agnostic — same Mojo code compiles to CPU SIMD, GPU kernels, or metal without separate .cu files

### Key Learnings for MojoLlama:

| Candle/Rust Pattern | Mojo Equivalent |
|---|---|
| `CustomOp1` trait + `cpu_fwd`/`cuda_fwd` | `fn` with `@parameter` if/elif for backend dispatch |
| `Storage::Cuda(s)`, `Storage::Cpu(s)` | `Buffer` abstractions in MAX |
| `Tensor` + `Layout` + `Shape` | `NDBuffer[DType, shape]` |
| `ffi::launch_*` CUDA calls | MAX compiler handles accelerator kernel generation |
| `Q8_1` quantize for activation | Same pattern, but using `simd` ops |
| Workspace cache (OnceLock+HashMap) | `@staticmethod` with per-device cache |
| Batch-size dispatch (MMVQ vs MMQ) | `@parameter` branch on batch dimension at compile time |
| Online softmax (m/s/v) | Same scalar formulation, vectorized |
| GgmlDType enum for 20+ quant types | `DType` parameter with per-type block struct |
| `extern "C"` FFI for CUDA | No FFI — MAX compiles Mojo directly to PTX |

### The Two-Phase Compute Pattern (Critically Important for Mojo):

The entire quantized matmul infrastructure relies on a **two-phase approach** that Mojo should replicate:

1. **Phase 1 (Quantize)**: Convert activation (F32/BF16/F16) → Q8_1 block format
   - Each block of 32 floats → 32 int8 values + 2 half-precision scales
   - This is trivially SIMD-friendly: find max, scale to [-127, 127]
   
2. **Phase 2 (Matmul)**: Block-level dot product between Q8_1 activation and quantized weight (any format)
   - Weight types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (block_size=32) or Q2K..Q6K (block_size=256)
   - Each block's scale(s) dequantized and multiplied with Q8_1 scales
   - Dot product accumulates in F32

This means MojoLlama only needs:
- One Q8_1 quantize kernel (can be generic over input dtype)
- One dequantize-dot kernel per weight type (all Q8_1 input → various weight formats)

---

## Summary of Files Analyzed

| File | Lines | Key Content |
|------|-------|-------------|
| `mistralrs-core/src/gguf/mod.rs` | 44 | GGUFArchitecture enum + from_value |
| `mistralrs-core/src/gguf/content.rs` | 257 | Content struct, multi-file reader, tensor access |
| `mistralrs-core/src/ops.rs` | 857 | TopK, topk_softmax, ArgSort, SplitOp, mul_and_act |
| `mistralrs-core/src/layers.rs` | 3077 | LayerNorm, RmsNorm, QLinear, RotaryEmbedding, Mlp |
| `mistralrs-quant/src/lib.rs` | 1319 | QuantMethod trait, QuantMethodConfig, MatMul wrapper |
| `mistralrs-quant/src/gguf/mod.rs` | 556 | GgufMatMul, forward dispatch, serde, ISQ pipeline |
| `mistralrs-quant/src/gguf/fast_mmvq.rs` | 333 | MMVQ decode path: quantize + matmul launcher |
| `mistralrs-quant/src/gguf/fast_mmq.rs` | 385 | MMQ prefill path: tiled quantized matmul |
| `mistralrs-quant/src/gguf/cuda.rs` | 810 | Indexed MoE CUDA forward |
| `mistralrs-quant/src/gguf/cpu.rs` | 60 | CPU fallback (dequantize + matmul) |
| `mistralrs-quant/src/gguf/ffi.rs` | 1292 | All extern "C" CUDA kernel declarations |
| `mistralrs-core/src/cuda/ffi.rs` | — | CUDA topk, sort FFI declarations |
| `mistralrs-core/src/attention/mod.rs` | — | Sdpa, AttentionMask, dispatch, chunked_attention |
| `mistralrs-core/src/attention/backends/naive.rs` | — | Naive SDPA (softmax(QK^T)V) |
| `mistralrs-core/src/attention/backends/cpu.rs` | — | CPU flash attention (online softmax, Rayon, TILE_KV=16) |
| `mistralrs-paged-attn/src/cuda/backend/paged_attention.rs` | — | Paged attention CUDA backend dispatch |
| `mistralrs-paged-attn/src/cuda/attention/attention_generic.cuh` | — | vLLM-derived attention template, Vec<> types, dot product |
| `mistralrs-paged-attn/src/cuda/pagedattention.cuh` | — | Block sum, fast_tanh, warp reduction helpers |
