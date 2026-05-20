# Hybrid MXFP4→FP8 + Hot/Cold MoE Tiering — Implementation Plan

> **For Hermes:** Execute using subagent-driven-development, targeting GTX 1660 for all non-FP8 work. Ada 2000 needed only for Phase 5.

**Goal:** Build a hybrid GPU/CPU inference backend for MojoLlama that (a) converts MXFP4-quantized weights to FP8 E4M3 for tensor core matmuls on Ada GPUs, (b) profiles expert routing to identify hot/cold experts, and (c) dispatches hot experts to GPU (FP8 tensor cores) and cold experts to CPU (existing C engine).

**Architecture:** Runtime GPU capability detection (`sm_89+` = FP8 path, `sm_70+` = FP16 fallback, else FP32 fallback). MXFP4 weights are converted to FP8 at load time. Expert routing profiler runs once per model, tags experts as hot/cold per layer. Hybrid dispatch sorts the top-K selected experts into GPU list and CPU list, executes both paths in parallel, and combines results weighted by router scores.

**Tech Stack:** C (existing C engine), CUDA C (cuBLAS FP8 matmul wrapper), Python (orchestrator, profiler, conversion), Mojo (optional — MAXBackend for full-GPU fallback).

**Dev strategy:** GTX 1660 (sm_75, no FP8) for Phases 1-4 using FP32 fallback. Ada 2000 for Phase 5 (FP8 activation + benchmark). Everything except the FP8 cuBLAS call compiles and runs on sm_75.

**Dependency:** Existing C engine (`cengine_batch_instr.c`), existing MojoLlama server (`server_moe.py`), existing hardware detection (`detect.py`), existing MXFP4 format (`block_mxfp4`, type 39).

---

## Phase 0: Infrastructure — GPU Capability Detection

### Task 0.1: Add runtime GPU feature detection

**Files:**
- Modify: `src/mojollama/detect.py` (add `gpu_caps()` function)

Add a function that probes CUDA device properties and returns available features:

```python
def gpu_caps(dev_id=0) -> dict:
    """Probe GPU at runtime. Returns format support and VRAM info."""
    import ctypes
    # cudaGetDeviceProperties via ctypes
    props = ...  # struct with major, minor, totalGlobalMem, name
    cc = props.major * 10 + props.minor
    return {
        "name": props.name,
        "cc": f"{props.major}.{props.minor}",
        "vram_bytes": props.totalGlobalMem,
        "vram_gb": props.totalGlobalMem / 1e9,
        "has_fp8_tensor_cores": cc >= 89,
        "has_fp16_tensor_cores": cc >= 70,
        "has_tensor_cores": cc >= 70,
        "best_matmul_format": "fp8" if cc >= 89 else "fp16" if cc >= 70 else "fp32",
    }
```

Test: on GTX 1660 returns `has_fp8_tensor_cores=False, best_matmul_format="fp32"`. On Ada 2000 returns `has_fp8_tensor_cores=True, best_matmul_format="fp8"`.

### Task 0.2: Create GPU matmul dispatch wrapper

**Files:**
- Create: `src/mojollama/backends/gpu_matmul.py`

```python
class GpuMatmulDispatcher:
    """Dispatch matmul to best available GPU path based on capability."""
    
    def __init__(self, dev_id=0):
        self.caps = gpu_caps(dev_id)
        self.lib = ctypes.CDLL("libgpu_backend.so")  # compiled CUDA wrapper
        
    def matmul(self, weights_gpu, input_gpu, output_gpu, M, N, K, fmt=None):
        """Dispatch to best available format."""
        fmt = fmt or self.caps["best_matmul_format"]
        if fmt == "fp8":
            self.lib.fp8_matmul(weights_gpu, input_gpu, output_gpu, M, N, K)
        elif fmt == "fp16":
            self.lib.fp16_matmul(weights_gpu, input_gpu, output_gpu, M, N, K)
        else:
            self.lib.fp32_matmul(weights_gpu, input_gpu, output_gpu, M, N, K)
```

### Task 0.3: Create CUDA stub lib that compiles on sm_75 and sm_89

**Files:**
- Create: `src/mojollama/kernels/gpu_backend.cu`
- Create: `src/mojollama/kernels/Makefile.gpu`

```
gpu_backend.cu contains:
- fp32_matmul()   — cublasSgemm, works on ALL CUDA GPUs
- fp16_matmul()   — cublasGemmEx with CUDA_R_16F, works sm_70+
- fp8_matmul()    — cublasLtMatmul with CUBLAS_COMPUTE_32F_FAST_FP8, sm_89+
                   → compiled with --generate-code=arch=compute_89,code=sm_89
                   → on sm_75: stub returns error code at runtime
```

Makefile compiles two fatbin versions (sm_75 + sm_89). At runtime, CUDA driver loads the matching cubin.

---

## Phase 1: MXFP4 → FP8 Converter

### Task 1.1: Write CPU-side MXFP4 → FP8 E4M3 bit converter

**Files:**
- Create: `src/mojollama/backends/fp8_convert.py`

```python
import numpy as np

# MXFP4 constants
MXFP4_BLOCK_SIZE = 32   # elements per block
MXFP4_BLOCK_BYTES = 17  # 16 bytes q + 1 byte e

def mxfp4_block_to_fp8_e4m3(q_bytes: np.ndarray, e_byte: int) -> np.ndarray:
    """Convert one MXFP4 block (32×4bit + E8M0) → 32×FP8 E4M3 bytes.
    
    This is a pure-Python reference. The GPU kernel will do the same.
    """
    # 1. Unpack 16 packed bytes → 32 × 4-bit values
    nibbles = np.unpackbits(q_bytes, bitorder='little').reshape(-1, 4)
    # Hmm, actually unpackbits gives bits. We have packed nibbles.
    # Better: each byte has two 4-bit values: low_nibble, high_nibble
    low = q_bytes & 0x0F
    high = (q_bytes >> 4) & 0x0F
    vals_4bit = np.empty(32, dtype=np.int8)
    vals_4bit[0::2] = low
    vals_4bit[1::2] = high
    
    # 2. Twos-complement decode: nibble > 7 → neg (nibble - 16)
    vals_int = np.where(vals_4bit > 7, vals_4bit.astype(np.int16) - 16, 
                        vals_4bit.astype(np.int16))
    
    # 3. Apply shared E8M0 scale
    scale = 2.0 ** (int(e_byte) - 127)
    vals_f32 = vals_int.astype(np.float32) * scale
    
    # 4. Clamp to FP8 E4M3 range [-240, 240]
    vals_f32 = np.clip(vals_f32, -240.0, 240.0)
    
    # 5. Pack to FP8 E4M3 bit representation
    # E4M3: sign(1) | exponent(4) | mantissa(3)
    # Bias = 7
    fp8_bytes = f32_to_fp8_e4m3(vals_f32)
    return fp8_bytes

def f32_to_fp8_e4m3(f32_array: np.ndarray) -> np.ndarray:
    """Convert float32 array to FP8 E4M3 byte array."""
    # Implementation: handle sign, handle zero/subnorm, normal encode
    # ...
    pass
```

Verification: round-trip test — convert known MXFP4 patterns, verify bit-exact FP8 output matches a CUDA-calculated reference.

### Task 1.2: Write GPU-side MXFP4 → FP8 bulk converter kernel

**Files:**
- Create: `src/mojollama/kernels/mxfp4_to_fp8.cu`

```cuda
// Launch config: 256 threads, process N blocks of MXFP4
// Each thread handles multiple blocks
__global__ void convert_mxfp4_to_fp8(
    const uint8_t* src,   // MXFP4 blocks, 17 bytes each
    uint8_t* dst,          // FP8 E4M3 bytes, 32 bytes per original block
    int num_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    
    const uint8_t* block = src + (size_t)tid * 17;
    uint8_t e_byte = block[16];
    float scale = __exp2f((float)(int)e_byte - 127);  // 2^(e-127)
    
    // Unpack 16 bytes → 32 nibbles, sign-extend, scale, pack FP8
    for (int i = 0; i < 16; i++) {
        uint8_t low = block[i] & 0x0F;
        uint8_t high = (block[i] >> 4) & 0x0F;
        // ... twos-complement, scale, clamp, FP8 pack ...
        dst[(size_t)tid * 32 + i * 2] = fp8_low;
        dst[(size_t)tid * 32 + i * 2 + 1] = fp8_high;
    }
}
```

Verification: CPU reference and GPU kernel produce same FP8 bytes for random MXFP4 data. Test passes on both GTX 1660 (convert on CPU, use FP32 fallback) and Ada 2000.

### Task 1.3: Write MXFP4 tensor → FP8 tensor converter (model load time)

**Files:**
- Modify: `src/mojollama/backends/gpu_matmul.py` (add weight conversion)

At model load time: iterate all MXFP4 (type 39) weight tensors from the GGUF, convert each block to FP8 on GPU, store the FP8 buffer. Original MXFP4 weights remain in CPU memory for cold-expert fallback.

For GPT-OSS-20B (72× MXFP4 experts at ~6MB each = ~430MB to convert): conversion takes ~50ms on GPU. Done once at load, not per-token.

---

## Phase 2: Expert Routing Profiler

### Task 2.1: Add routing statistics collector to MoE engine

**Files:**
- Modify: `src/mojollama/kernels/turbo_engine_v7_moe.py` (add `profiler.py` hook)

```python
class RoutingProfiler:
    """Collect expert routing statistics per layer."""
    
    def __init__(self, n_layers, n_experts, n_top_k):
        self.counts = np.zeros((n_layers, n_experts), dtype=np.int64)
        self.total = 0
        self.n_top_k = n_top_k
    
    def record(self, layer, top_indices):
        """Record which experts were selected for one token."""
        self.counts[layer, top_indices] += 1
        self.total += 1
    
    def summary(self, coverage=0.80):
        """Return list of hot expert IDs per layer covering `coverage` fraction."""
        hot_per_layer = []
        for l in range(self.counts.shape[0]):
            counts = self.counts[l]
            order = np.argsort(-counts)  # descending
            cumsum = np.cumsum(counts[order]) / max(1, counts.sum())
            n_hot = int((cumsum < coverage).sum()) + 1
            hot_per_layer.append(set(order[:n_hot].tolist()))
        return hot_per_layer
```

### Task 2.2: Create hybrid-profile CLI command

**Files:**
- Modify: `src/mojollama/studio/__init__.py` (add `hybrid-profile` subcommand)

```bash
mojollama hybrid-profile model.gguf --calibration-file wiki.txt --output routing.json

# Output:
{
  "model": "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
  "n_layers": 48,
  "n_experts": 256,
  "n_top_k": 8,
  "tokens_profiled": 8192,
  "routing": {
    "layer_0": {"total_hits": 65536, "hot_80pct": [12, 45, 67, ...], "distribution_zipf_param": 1.2},
    ...
  }
}
```

Verification: run on Qwen3-30B, confirm hot experts are ~20-40 per layer and cover ~80% of routing.

### Task 2.3: Add VRAM budget calculator

**Files:**
- Create: `src/mojollama/backends/vram_budget.py`

Given GPU VRAM (16GB for Ada 2000, 6GB for GTX 1660) and model metadata, calculate how many hot experts per layer fit in VRAM after MXFP4→FP8 conversion (2× size increase).

```python
def calc_vram_budget(model_size_gb, gpu_vram_gb, n_layers, n_experts, 
                     expert_params_per_layer_mb, conversion_expansion=2.0):
    """Calculate max hot experts per layer given VRAM constraints."""
    model_on_gpu = model_size_gb * conversion_expansion  # if converting all
    available = gpu_vram_gb - 2.0  # 2GB reserve for KV cache + buffers
    expert_mb = n_layers * n_experts * expert_params_per_layer_mb
    hot_fraction = available / (expert_mb / 1024 * conversion_expansion)
    return max(1, int(n_experts * hot_fraction))
```

---

## Phase 3: Hybrid GGUF Format

### Task 3.1: Define hybrid metadata schema

**Files:**
- Create: `src/mojollama/backends/hybrid_schema.py`

```python
HYBRID_METADATA_KEYS = {
    "hybrid.enabled": "bool",
    "hybrid.hot_experts": "json",     # per-layer list of hot expert IDs
    "hybrid.profile": "json",          # routing_profile.json embedded
    "hybrid.conversion": "json",       # {hot_format, cold_format, vram_usage}
    "hybrid.gpu_caps_min": "string",   # minimal CC e.g. "8.9" for FP8
}
```

### Task 3.2: Write hybrid-convert CLI command

**Files:**
- Create: `src/mojollama/studio/cmd_hybrid_convert.py`
- Modify: `src/mojollama/studio/__init__.py` (register subcommand)

```bash
mojollama hybrid-convert model.gguf \
    --profile routing.json \
    --hot-experts 32 \
    --dry-run
# Output: "Would tag 32 experts/layer as GPU-hot. VRAM estimate: 14.2 GB. Suitable for RTX 2000 Ada."

mojollama hybrid-convert model.gguf \
    --profile routing.json \
    --hot-experts 40 \
    --output model.hybrid.gguf
# Writes same GGUF with added hybrid metadata keys
# Does NOT rewrite weights — just adds metadata
```

---

## Phase 4: Hybrid Inference Orchestrator

### Task 4.1: Build GPU expert weight manager

**Files:**
- Create: `src/mojollama/backends/gpu_weights.py`

```python
class GpuExpertManager:
    """Manages hot expert weights on GPU."""
    
    def __init__(self, caps, dispatcher):
        self.caps = caps
        self.dispatcher = dispatcher
        self.hot_ptrs = {}  # (layer, expert_id) → GPU pointer
        self.gpu_mem_pool = []
    
    def load_hot_experts(self, model, hot_map):
        """Convert hot expert MXFP4→FP8 and upload to GPU."""
        total_bytes = 0
        for layer in range(model.n_layers):
            for eid in hot_map[layer]:
                src = model.get_expert_weights(layer, eid)  # MXFP4 bytes
                fp8_bytes = convert_mxfp4_to_fp8(src)        # Task 1.1
                gpu_ptr = cuda_malloc(fp8_bytes.nbytes)
                cuda_memcpy_htod(gpu_ptr, fp8_bytes)
                self.hot_ptrs[(layer, eid)] = gpu_ptr
                total_bytes += fp8_bytes.nbytes
        return total_bytes
```

### Task 4.2: Build hybrid MoE FFN dispatcher

**Files:**
- Create: `src/mojollama/backends/hybrid_ffn.py`

```python
class HybridMoEFFN:
    """Route selected experts to GPU (hot) or CPU (cold), combine results."""
    
    def __init__(self, engine, gpu_mgr, hot_map):
        self.engine = engine         # CPU C engine
        self.gpu = gpu_mgr           # GPU weight manager
        self.hot_map = hot_map       # per-layer set of hot expert IDs
    
    def forward(self, layer, x_hidden_state, top_indices, top_weights):
        hot_list = [e for e in top_indices if e in self.hot_map[layer]]
        cold_list = [e for e in top_indices if e not in self.hot_map[layer]]
        
        result = np.zeros_like(x_hidden_state)
        
        # GPU path: batch all hot experts into one GPU matmul call
        if hot_list:
            gpu_out = self.gpu.batch_expert_ffn(layer, hot_list, x_hidden_state)
            result += self._weighted_sum(gpu_out, hot_list, top_weights)
        
        # CPU path: existing C engine for cold experts
        if cold_list:
            cpu_out = self.engine.cpu_expert_ffn(layer, cold_list, x_hidden_state)
            result += self._weighted_sum(cpu_out, cold_list, top_weights)
        
        return result
```

### Task 4.3: Integrate into server_moe.py

**Files:**
- Modify: `src/mojollama/server_moe.py` (add `--hybrid` flag)

```python
if args.hybrid:
    caps = gpu_caps()
    if caps["vram_gb"] < model_size_gb * 0.5:
        print(f"[Hybrid] GPU VRAM ({caps['vram_gb']:.0f}GB) < model, enabling hot/cold tiering")
        hot_map = load_hot_map(args.hybrid)  # from GGUF metadata
        gpu_mgr = GpuExpertManager(caps, dispatcher)
        gpu_mgr.load_hot_experts(engine.model, hot_map)
        engine.moe_ffn = HybridMoEFFN(engine, gpu_mgr, hot_map)
```

---

## Phase 5: FP8 Tensor Core Activation (Ada 2000 Only)

### Task 5.1: Write cuBLAS FP8 matmul wrapper

**Files:**
- Modify: `src/mojollama/kernels/gpu_backend.cu` (add fp8_matmul)

```cuda
// sm_89+ only — guard with __CUDA_ARCH__ >= 890
cudaError_t fp8_matmul(const void* A, const void* B, void* C, 
                       int M, int N, int K) {
#if __CUDA_ARCH__ >= 890
    cublasLtMatmulDesc_t op_desc;
    cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F_FAST_FP8);
    
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M);
    
    cublasLtMatmul(handle, op_desc, 
                   &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, Cdesc,
                   NULL, NULL, 0);
    return CUBLAS_STATUS_SUCCESS;
#else
    return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
}
```

### Task 5.2: Add FP8 weight pre-packing to GpuExpertManager

On Ada 2000, the MXFP4→FP8 conversion (Task 1.1) runs on GPU as a pre-processing kernel (Task 1.2). This happens once at model load and takes <100ms for all hot expert weights.

### Task 5.3: End-to-end benchmark

**Files:**
- Modify: `src/mojollama/bench_tok.py` (add `--hybrid` flag)

```bash
# Qwen3-30B on Ada 2000, hybrid mode
OMP_NUM_THREADS=32 python3 -u bench_tok.py \
    --model /tmp/models/qwen3.hybrid.gguf \
    --hybrid

# Baseline: CPU-only
OMP_NUM_THREADS=32 python3 -u bench_tok.py \
    --model /tmp/models/Qwen3-30B.gguf

# Baseline: llama.cpp on CPU
/tmp/llama.cpp/build/bin/llama-bench -m /tmp/models/Qwen3-30B.gguf -n 128 -t 32 -p 512 -r 3
```

Track: tok/s, ms/token, CPU→GPU transfer bytes/token, hot/cold hit rate.

---

## File Inventory

### New files (~1,800 lines total):
- `src/mojollama/backends/gpu_matmul.py` — GPU matmul dispatch (150 lines)
- `src/mojollama/backends/fp8_convert.py` — MXFP4↔FP8 converter (200 lines)
- `src/mojollama/backends/vram_budget.py` — VRAM budget calculator (80 lines)
- `src/mojollama/backends/hybrid_schema.py` — hybrid GGUF metadata schema (50 lines)
- `src/mojollama/backends/gpu_weights.py` — GPU expert weight manager (200 lines)
- `src/mojollama/backends/hybrid_ffn.py` — Hybrid MoE FFN dispatcher (300 lines)
- `src/mojollama/kernels/gpu_backend.cu` — CUDA matmul wrappers (300 lines)
- `src/mojollama/kernels/mxfp4_to_fp8.cu` — GPU MXFP4→FP8 converter (200 lines)
- `src/mojollama/kernels/Makefile.gpu` — GPU kernel build (40 lines)
- `src/mojollama/studio/cmd_hybrid_convert.py` — CLI command (150 lines)
- `tests/test_hybrid_*.py` — test suite (~5 files, 400 lines total)

### Modified files:
- `src/mojollama/detect.py` — add `gpu_caps()` (+40 lines)
- `src/mojollama/server_moe.py` — add `--hybrid` flag (+80 lines)
- `src/mojollama/studio/__init__.py` — register hybrid-* subcommands (+20 lines)
- `src/mojollama/kernels/turbo_engine_v7_moe.py` — add profiler hook (+60 lines)

---

## Verification Milestones

1. `gpu_caps()` returns correct capabilities on GTX 1660 (fp32) and Ada 2000 (fp8)
2. MXFP4→FP8 round-trip: same float32 values (±1ULP) for random test data
3. Routing profiler on Qwen3-30B produces Zipfian distribution, hot-80% = ~30 experts/layer
4. Hybrid GGUF written and re-read with correct metadata
5. `--hybrid` server runs on GTX 1660 with FP32 fallback — correct output, all tests pass
6. `--hybrid` server on Ada 2000 with FP8 — correct output, 1.5-2× tok/s vs CPU-only
7. Benchmark: hot coverage ≥80%, cold expert overhead <10% of total time
