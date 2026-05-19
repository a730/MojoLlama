# MAX API Surface: Candle/CUDA Replacement Mapping for MojoLlama

## Overview

This document catalogs the MAX framework API surface found in the Modular repository at `/tmp/modular/max/`. It maps Candle/PyTorch/CUDA patterns to their MAX equivalents across the inference engine, graph runtime, kernel compilation, tensor operations, device management, and quantization.

## 1. Inference Engine (`max.engine`)

**File:** `max/python/max/engine/`

| MAX API | Candle/PyTorch Equivalent | Description |
|---------|---------------------------|-------------|
| `InferenceSession(devices=[...])` | — | Session manager, analogous to `candle::Device` ownership |
| `session.load(model_path)` | `torch.jit.load()` | Load + compile a model for inference |
| `session.compile(model)` | — | Separate compilation from weight binding |
| `session.init(compiled, weights_registry=...)` | — | Bind weights to compiled model |
| `Model.execute(*args)` | `model.forward()` | Synchronous inference; returns `list[Buffer]` |
| `Model.__call__(*args, **kwargs)` | `model(x)` | Python-callable wrapper |
| `Model.capture(graph_keys, *inputs)` | `torch.cuda.CUDAGraph.capture_begin/end()` | Capture execution as device graph (CUDA Graph) |
| `Model.replay(graph_keys, *inputs)` | `CUDAGraph.replay()` | Replay captured device graph |
| `session.gpu_profiling("detailed")` | `nsys profile` | Nsight Systems/Nsight Compute profiling |
| `TensorSpec(shape, dtype, name)` | — | Describes tensor metadata in graph |

**Candle/CUDA Pattern → MAX:**

```python
# Candle
let xs: Tensor = ...
let ws = candle::nn::Linear::new(...)

# MAX
session = InferenceSession(devices=[Accelerator()])
model = session.load("model.mof")
outputs = model.execute(input_buffer)
```

## 2. Graph Runtime (`max.graph`)

**File:** `max/python/max/graph/`

### 2.1 Graph Construction

| MAX API | Candle/PyTorch Equivalent |
|---------|---------------------------|
| `Graph(name, input_types=..., forward=...)` | `nn.Module.__init__` + `forward()` |
| `Graph.__enter__()` / `__exit__()` | Context manager for graph building |
| `graph.inputs[i].tensor` | Model input tensor |
| `graph.output(...)` | Sets graph output |
| `graph.add_weight(Weight(...))` | `nn.Parameter` |
| `graph.add_subgraph(name, ...)` | Subgraph / function-like block |
| `ops.custom(name, device, values, out_types)` | `torch.ops.custom_op` |
| `ops.call(subgraph, ...)` | Call a subgraph |
| `Module` | Multi-graph container for compilation |

### 2.2 Graph Operations (`max.graph.ops`)

#### Linear Algebra

| MAX Graph Op | Candle Op | PyTorch Op |
|---|---|---|
| `ops.matmul` | `candle::Tensor::matmul` | `torch.matmul` / `@` |
| `ops.add`, `ops.sub`, `ops.mul`, `ops.div` | Element-wise arithmetic | `+`, `-`, `*`, `/` |
| `ops.transpose` | `.t()` | `torch.transpose` |
| `ops.permute` | `.permute()` | `torch.permute` |
| `ops.reshape` | `.reshape()` | `torch.reshape` |
| `ops.broadcast_to` | `.broadcast()` | `torch.broadcast_to` |
| `ops.cast` | `.to_dtype()` | `.to(dtype)` |
| `ops.concat` | `candle::Tensor::cat` | `torch.cat` |
| `ops.stack` | — | `torch.stack` |
| `ops.split` | `.split()` | `torch.split` |
| `ops.squeeze`/`ops.unsqueeze` | `.squeeze()`/`.unsqueeze()` | `torch.squeeze/unsqueeze` |

#### Neural Network

| MAX Graph Op | Candle/PyTorch |
|---|---|
| `ops.softmax` | `F.softmax` |
| `ops.gelu` | `F.gelu` |
| `ops.silu` | `F.silu` (SwiGLU) |
| `ops.sigmoid` | `F.sigmoid` |
| `ops.tanh` | `F.tanh` |
| `ops.relu` (via `maximum(x,0)`) | `F.relu` |
| `ops.layer_norm` | `F.layer_norm` |
| `ops.rms_norm` | Custom RMSNorm |
| `ops.group_norm` | `F.group_norm` |
| `ops.sum`, `ops.mean`, `ops.prod` | reductions |
| `ops.argmax`, `ops.argmin` | `argmax`/`argmin` |
| `ops.top_k` | `torch.topk` |
| `ops.where` | `torch.where` |
| `ops.gather` | `torch.gather` |
| `ops.scatter` | `torch.scatter` |
| `ops.pad` | `F.pad` |
| `ops.concat` | `torch.cat` |
| `ops.conv2d`, `ops.conv3d` | `F.conv2d/3d` |
| `ops.avg_pool2d`, `ops.max_pool2d` | `F.avg/max_pool2d` |
| `ops.repeat_interleave` | `torch.repeat_interleave` |

#### Quantized Operations

| MAX Graph Op | Description |
|---|---|
| `ops.qmatmul(encoding, config, lhs, *rhs)` | Quantized matrix multiply (Q4_0, Q4_K, Q6_K, GPTQ) |
| `ops.dequantize(encoding, quantized)` | Dequantize GGUF-style weights |
| `repack_gguf_quantized_weights(weight, encoding)` | Repack GGUF weights for optimized matmul |

## 3. Neural Network Layers (`max.nn`)

**File:** `max/python/max/nn/`

| MAX Layer | Candle/PyTorch Equivalent |
|-----------|---------------------------|
| `nn.Linear(in_dim, out_dim, ...)` | `candle::nn::Linear` / `nn.Linear` |
| `nn.Embedding(vocab_size, hidden_dim)` | `candle::nn::Embedding` / `nn.Embedding` |
| `nn.RMSNorm(dim, eps, ...)` | Llama RMSNorm |
| `nn.LayerNorm(dims, eps, ...)` | `nn.LayerNorm` |
| `nn.GroupNorm` | `nn.GroupNorm` |
| `nn.RotaryEmbedding` | RoPE |
| `nn.AttentionWithRope(...)` | MHA + RoPE (Llama-style) |
| `nn.MultiheadAttention` | `nn.MultiheadAttention` |
| `nn.GPTQAttentionWithRope` | GPTQ-quantized attention |
| `nn.KVCacheParams` | KV cache config |
| `nn.VocabParallelEmbedding` | Distributed embedding |
| `nn.ColumnParallelLinear` | Tensor parallelism linear |
| `nn.MoE`, `nn.MoEGate` | Mixture-of-Experts |
| `nn.MLP` | MLP block |
| `Module` (base class) | `nn.Module` |
| `LayerList` / `Sequential` | `nn.Sequential` |

### KV Cache

| MAX Component | Description |
|---|---|
| `KVCacheParams` | Configuration (dtype, nheads, cache_size) |
| `KVCacheInputs` | Input types for cache operations |
| `PagedCacheValues` | Paged cache implementation |
| `KVCacheMetrics` | Cache hit/miss statistics |

## 4. Device Management (`max.driver`)

**File:** `max/python/max/driver/`

| MAX API | Candle/PyTorch | Description |
|---------|-----------------|-------------|
| `CPU()` | `Device::Cpu` | Host device |
| `Accelerator(id=0)` | `Device::Cuda(0)` | GPU accelerator device |
| `DeviceStream(device)` | `cudaStream_t` | Device execution stream |
| `Buffer(dtype, shape, device)` | `candle::Tensor` storage | Device buffer / tensor storage |
| `Buffer.to(device)` | `.to_device()` | Device transfer |
| `Buffer.from_numpy(arr)` | `Tensor::from_slice` | Import numpy data |
| `Buffer.to_numpy()` | `.to_vec2()` | Export to numpy |
| `Buffer.from_dlpack(obj)` | DLPack interop | Import via DLPack |
| `Buffer.mmap(path, ...)` | — | Memory-map file as buffer |
| `DeviceSpec(id, device_type)` | — | Device specification tuple |
| `accelerator_count()` | — | Count available GPUs |
| `scan_available_devices()` | — | Enumerate devices |

**Candle → MAX Device Pattern:**

```python
# Candle
let device = Device::Cuda(0);

# MAX
from max.driver import Accelerator, Buffer, CPU
device = Accelerator(0)
cpu = CPU()
buffer_on_gpu = Buffer(dtype, shape, device)
buffer_on_cpu = buffer_on_gpu.to(cpu)
```

## 5. Mojo Kernels (`max/kernels/src/`)

### 5.1 Linear Algebra Kernels

**Directory:** `max/kernels/src/linalg/`

| Kernel File | Functionality |
|-------------|---------------|
| `matmul/` | High-performance GEMM (CPU: vnni/neon/apple, GPU: sm80/sm90/sm100, AMD) |
| `bmm.mojo` | Batched matrix multiply |
| `gemv.mojo` | Matrix-vector multiply |
| `grouped_matmul.mojo` | Grouped batched matmul (MoE dispatch) |
| `transpose.mojo` | Matrix transpose |
| `fp8_quantization.mojo` | FP8 quantization/dequantization |
| `fp4_quantization.mojo` | FP4 quantization (MXFP4) |
| `accumulate.mojo` | Accumulation operations |
| `packing.mojo` | Weight packing utilities |
| `lora.mojo` | LoRA matrix operations |

### 5.2 Neural Network Kernels

**Directory:** `max/kernels/src/nn/`

| Kernel File | Candle/CUDA Equivalent |
|-------------|------------------------|
| `softmax.mojo` | `candle::ops::softmax` |
| `normalization.mojo` | Layer/RMS norm |
| `activations.mojo` | GELU, SiLU, etc. |
| `rope.mojo` | RoPE implementation |
| `topk.mojo` | Top-K sampling |
| `sampling.mojo` | Token sampling (top-p, min-p) |
| `moe.mojo` | Mixture-of-Experts |
| `kv_cache.mojo` | KV cache operations |
| `kv_cache_ragged.mojo` | Ragged (variable-length) KV cache |
| `attention/` | Flash attention kernels (MHA, MLA) |
| `conv/` | Convolution (2D, 3D, transposed) |
| `broadcast.mojo` | Tensor broadcasting |
| `concat.mojo` | Concatenation |
| `slice.mojo` | Slicing |
| `split.mojo` | Splitting |
| `reshape.mojo` | Reshaping |
| `pad.mojo` | Padding |
| `pool.mojo` | Pooling |

### 5.3 Quantization Kernels

**Directory:** `max/kernels/src/quantization/`

| Kernel | Description |
|--------|-------------|
| `per_channel_grouped_4bit.mojo` | Per-channel grouped 4-bit quant matmul |
| `qmatmul.mojo` | Q4_0 quantized matmul (CPU) |
| `qmatmul_gpu.mojo` | Q4_0 quantized matmul (GPU) |
| `qmatmul_k.mojo` | Q4_K/Q6_K quantized matmul |

### 5.4 Attention Kernels

**Directory:** `max/kernels/src/nn/attention/gpu/`

| Kernel | Description |
|--------|-------------|
| `nvidia/sm90/mha.mojo` | Multi-head attention SM90 (H100) |
| `nvidia/sm100/attention.mojo` | Flash attention SM100 (B200) |
| `nvidia/sm100/mla_decode.mojo` | Multi-head Latent Attention decode (DeepSeek) |
| `nvidia/sm100/mla_prefill.mojo` | MLA prefill |
| `nvidia/sm100/mha_depth512/` | Depth-512 flash attention |
| `amd_rdna/mha_decode.mojo` | AMD attention decode |
| `amd_structured/mha_decode.mojo` | AMD structured attention |

### 5.5 Vendor Library Bindings

| Directory | Vendor Library |
|-----------|---------------|
| `_cublas/` | NVIDIA cuBLAS + cuBLASLt |
| `_cudnn/` | NVIDIA cuDNN |
| `_cufft/` | NVIDIA cuFFT |
| `_curand/` | NVIDIA cuRAND |
| `_rocblas/` | AMD rocBLAS + hipBLASLt |
| `_miopen/` | AMD MIOpen |

### 5.6 Communication Kernels

**Directory:** `max/kernels/src/comm/`

| Kernel | Description |
|--------|-------------|
| `allreduce.mojo` | All-reduce |
| `allgather.mojo` | All-gather |
| `reducescatter.mojo` | Reduce-scatter |
| `broadcast.mojo` | Broadcast |
| `scatter.mojo` | Scatter |
| `sync.mojo` | Synchronization |
| `device_query.mojo` | Device property query |

### 5.7 State Space Model Kernels

**Directory:** `max/kernels/src/state_space/`

| Kernel | Description |
|--------|-------------|
| `selective_scan.mojo` | Mamba selective scan |
| `causal_conv1d.mojo` | Causal 1D convolution (Mamba) |
| `rms_norm_fused_residual.mojo` | Fused RMSNorm + residual |

## 6. Custom Ops / Kernel Compilation

### 6.1 Python: `max.graph.ops.custom`

Register and call custom Mojo kernels from Python graphs:

```python
result = ops.custom(
    name="my_kernel",           # matches @compiler.register("my_kernel")
    device=Accelerator(0),
    values=[input_tensor],
    out_types=[TensorType(DType.float32, [M, N], device=Accelerator(0))],
    parameters={"param": 42},
)
```

### 6.2 Mojo: `@compiler.register` (Kernel Authoring)

Custom kernels are written in Mojo and registered with the compiler:

```mojo
from compiler import register

@register("my_kernel")
fn my_kernel[
    dtype: DType,
    rank: Int,
](input: ManagedTensorSlice[dtype, rank], output: ManagedTensorSlice[dtype, rank]):
    # Kernel body
    ...
```

Located in: `max/examples/custom_ops/kernels/`

### 6.3 Mojo Tensor Extensibility API

**Directory:** `max/kernels/src/extensibility/tensor/`

| API | Description |
|-----|-------------|
| `RuntimeTensorSpec[dtype, rank]` | Compile-time tensor spec |
| `ManagedTensorSlice[dtype, rank]` | Managed tensor view for custom ops |
| `InputTensor`, `OutputTensor` | Input/output tensor declarations |
| `ElementwiseUnaryOp`, `ElementwiseBinaryOp` | Operation traits for auto-generated kernels |
| `IOSpec`, `Input`, `Output`, `FusedInput`, `FusedOutput` | I/O specification for custom ops |
| `StaticTensorSpec`, `StaticTensorSpecList` | Static tensor specs |

### 6.4 Kernel Library Loading

```python
from max.graph import KernelLibrary

kernels = KernelLibrary([Path("my_kernels.mojoc")])
# Or from source
kernels.load_paths([Path("kernels/")])
```

## 7. DType Support (`max.dtype`)

| `DType` Enum | NumPy Equivalent | Description |
|---|---|---|
| `DType.bool` | `np.bool_` | Boolean |
| `DType.int8` .. `DType.int64` | `np.int*` | Signed integers |
| `DType.uint8` .. `DType.uint64` | `np.uint*` | Unsigned integers |
| `DType.float8_e4m3fn` | — | FP8 (E4M3) |
| `DType.float8_e5m2` | — | FP8 (E5M2) |
| `DType.float16` | `np.float16` | FP16 |
| `DType.bfloat16` | — | BF16 |
| `DType.float32` | `np.float32` | FP32 (default) |
| `DType.float64` | `np.float64` | FP64 |

## 8. C API (`max/include/max/c/`)

For C/C++ integration (e.g., from Rust via FFI):

| Header | Key Functions |
|--------|---------------|
| `tensor.h` | `M_newTensorSpec()`, `M_borrowTensorInto()`, `M_getTensorByNameFrom()`, `M_copyTensorToDevice()` |
| `device.h` | `M_newDevice()`, `M_getAcceleratorCount()`, `M_synchronizeDevice()` |
| `context.h` | `M_newRuntimeContext()`, `M_newRuntimeConfig()` |
| `model.h` | `M_compileModel()`, `M_executeModelSync()`, `M_captureModelSync()`, `M_replayModelSync()` |
| `weights.h` | `M_newWeightsRegistry()` |

## 9. Pipeline Architectures (`max/pipelines/architectures/`)

Pre-built model architectures (analogous to HuggingFace `transformers`):

| Architecture | Models |
|---|---|
| `llama3/` | Llama 3, Llama 3.1, Llama 3.2 |
| `qwen3/`, `qwen2/` | Qwen 3, Qwen 2.5 |
| `deepseekV2/`, `deepseekV3/` | DeepSeek V2, V3, R1 |
| `mistral/`, `mistral3/` | Mistral v0.1, v0.2, v0.3, Mistral 3 |
| `gemma3/`, `gemma4/` | Gemma 3, Gemma 4 |
| `phi3/` | Phi-3 |
| `olmo/`, `olmo2/`, `olmo3/` | OLMo family |
| `bert/` | BERT |
| `clip/` | CLIP |
| `whisper/` | Whisper |
| `mamba/` | Mamba SSM |
| `flux2/` | Flux image generation |
| `wan/` | Video generation |

## 10. Key Replacements Summary

| Candle/CUDA Pattern | MAX Replacement | Where |
|---------------------|----------------|-------|
| `candle::Tensor::from_slice` | `Buffer.from_numpy()` | `max.driver` |
| `tensor.to_device()` | `Buffer.to()` | `max.driver` |
| `candle::nn::Linear` | `nn.Linear` | `max.nn` |
| `candle::nn::Embedding` | `nn.Embedding` | `max.nn` |
| `tensor.matmul()` | `ops.matmul` | `max.graph.ops` |
| `candle::quantized::Q4Matmul` | `ops.qmatmul` | `max.graph.ops` |
| `candle::attention::Attention` | `nn.AttentionWithRope` | `max.nn` |
| `CUDA kernels` | Mojo `@compiler.register` | `max/kernels/src/` |
| `nvcc / CUDA C++` | Mojo `gpu` module | `max/kernels/src/` |
| `cudaStream_t` | `DeviceStream` | `max.driver` |
| `torch.cuda.CUDAGraph` | `Model.capture`/`Model.replay` | `max.engine` |
| `ggml` quant types (Q4_0..Q6_K) | `QuantizationEncoding` | `max.graph.quantization` |
| `cudaMemcpy` | `Buffer.inplace_copy_from` | `max.driver` |
| `torch.jit.script` | `Graph` + compile | `max.graph` + `max.engine` |
| NCCL all-reduce | `ops.allreduce` | `max.graph.ops` |

**File created:** `/onedev-workspace/work/max-api-surface.md`
