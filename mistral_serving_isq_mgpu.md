# mistral.rs Deep Research: Serving Layer, ISQ, Multi-GPU, & Auto-Tuning

> Research conducted: May 18, 2026
> Source: github.com/EricLBuehler/mistral.rs @ 0ed6f6c

---

## Area 1: HTTP Serving Layer (OpenAI API)

### Architecture Overview

The serving layer lives in `mistralrs-server-core/` and is built on **axum** (Tokio-based async HTTP framework). It exposes OpenAI-compatible endpoints for chat completions, completions, embeddings, image generation, speech, files, and responses.

The architecture follows a **channel-based request/response** pattern:

```
Client HTTP Request
  -> axum Router (MistralRsServerRouterBuilder)
    -> Endpoint Handler
      -> create_response_channel() -> (Sender, Receiver)
        -> parse_request() converts OpenAI JSON -> Request
          -> send_request() dispatches via MPSC to engine
            -> Engine processes, sends Response via Receiver
              -> Handler converts Response to HTTP response (JSON or SSE)
```

### Router Construction (mistralrs_server_router_builder.rs)

Core pattern: Builder pattern that wraps axum `Router::new()`.

Routes registered:
- POST /v1/chat/completions
- POST /v1/completions
- POST /v1/embeddings
- GET /v1/models
- POST /v1/models/unload, /v1/models/reload, /v1/models/status, /v1/models/tune
- GET /v1/system/info
- POST /v1/system/doctor
- GET /health
- POST /re_isq
- POST /v1/images/generations
- GET /v1/files, GET/DELETE /v1/files/{id}, GET /v1/files/{id}/content
- POST /v1/audio/speech
- POST /v1/responses, GET/DELETE /v1/responses/{response_id}
- POST /v1/responses/{response_id}/cancel
- GET/PUT/DELETE /v1/sessions/{session_id}

Middleware chain:
1. CORS layer (configurable origins, methods GET/POST/PUT/DELETE, headers Content-Type/Authorization)
2. DefaultBodyLimit (50 MB, configurable via builder)
3. Extension injector for AgenticDefaults (max_tool_rounds, tool_dispatch_url)
4. Shared state via Arc<MistralRs>

Multi-model routing: model field selects sender via state.get_sender(model_id). If model == "default", primary model is used.

### Request/Response Channel Pattern (handler_core.rs)

Core dispatch creates buffered channel (default 10_000), parses OpenAI JSON to internal Request, sends via MPSC, processes response as SSE or JSON.

Error handling via ErrorToResponse trait (Serializes to JSON with HTTP status code).

BaseCompletionResponder<R, S> enum:
- Sse(Sse<S>) - streaming SSE
- Json(R) - complete JSON
- ModelError(String, R) - partial error with data
- InternalError(Box<dyn Error>) - server error
- ValidationError(Box<dyn Error>) - 422

### Streaming Implementation (streaming.rs, chat_completion.rs)

SSE via axum Sse::new(stream) with KeepAlive. BaseStreamer<R, C, D> implements futures::Stream.

State machine: Running -> SendingDone (emits [DONE]) -> Done (fires on_done).

Callback hooks: on_chunk (Fn(R) -> R), on_done (Fn(&[R])).
Keep-alive: KEEP_ALIVE_INTERVAL env var (default 10s).

### Applicability to MojoLlama (Mojo/MAX Terms)

1. Channel-based request/response for async serving
2. Builder pattern for router construction and middleware composition
3. SSE streaming with KeepAlive as Mojo Stream iterator
4. Error-as-response-variant enum for composable error handling
5. Callback hooks for middleware/injection (logging, metrics, content filtering)
6. Multi-model dispatch via string-keyed pipeline lookup

---

## Area 2: ISQ (In-Situ Quantization) Pipeline

### Architecture Overview

ISQ converts weights from FP16/BF16/F32 to quantized formats after loading. Operates at layer level through QuantMethod trait.

Flow:
- Model weights loaded
- IsqModel::get_layers() returns all quantizable layers
- For each layer: layer.apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
- UQFF file written for fast reload

### Core Types

IsqType enum: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K-Q8K, HQQ4, HQQ8, F8E4M3, AFQ2-AFQ8, F8Q8, MXFP4.

IsqBits: numeric shorthand (2-8). Platform-adaptive resolution:
- Metal: Two->AFQ2, Three->AFQ3, Four->AFQ4, Six->AFQ6, Eight->AFQ8
- CUDA/CPU: Two->Q2K, Three->Q3K, Four->Q4K, Six->Q6K, Eight->Q8_0

QuantMethod trait: apply_isq(), forward(), dequantize_w(), quantized_act_type(), add_delta_w(), gather_forward(), is_distributed().

### ISQ Pipeline Flow (mistralrs-core/src/pipeline/isq.rs)

IsqModel::quantize(): load imatrix, get_layers(), apply_isq() per layer, parallel via rayon (GPU serialized by QuantizeOntoGuard), write UQFF artifacts.

IsqOrganization: Default (all layers) or MoeExpertsOnly (MoQE).

### Immediate ISQ (mistralrs-quant/src/utils/isq.rs)

ISQ during loading via thread-local ENGINE_IMMEDIATE_ISQ RefCell. Supports:
- Parallel path (spawn on pool, return PendingIsqLayer wrapper)
- Sync path (Metal/integrated GPU)
- Backpressure system (prevents OOM on MoE models)
- Per-layer overrides (regex-based ty/device)

### UQFF Binary Format

[UQFF_VERSION: u32 LE] [QuantizedSerdeType: u8] [data_length: u32 LE] [has_bias: u8] [dtype: u32 LE] [n_dims: u32 LE] [dims...: u32 LE] [tensor_data: u8[]] [optional bias tensor]

10GB shard limit; files named {name}-{N}.uqff.

### Applicability to MojoLlama

1. Trait-based quantization dispatch via QuantMethod analogue
2. Platform-adaptive quantization (detect backend, select optimal format)
3. Immediate ISQ during loading with thread pool + backpressure
4. Importance matrix (imatrix) support for quality preservation
5. UQFF-style format with residuals for fast reload
6. Backpressure for memory-constrained MoE environments

---

## Area 3: Multi-GPU and Device Mapping

### Architecture Overview

Three strategies:
1. Layer-pipeline parallelism (per-layer device)
2. NCCL tensor parallelism (per-weight sharding + all-reduce)
3. Ring-based distributed (TCP multi-node)

### Device Map System (device_map.rs)

DeviceMapSetting: Map (manual/auto layers), Auto (compute optimal), DummyNccl, Nccl (real NCCL).

DeviceMapper trait:
- map(input, layer) - runtime tensor move
- set_device(layer, vb, isq) - load-time varbuilder assignment
- device_for(layer, isq) - query device
- get_unique_devices() - all distinct devices
- cast_nm_device(x, isq) - non-mapped layers (embed, lm_head)
- get_comm_for(layer_idx) - communicator for TP
- num_device_mapping_layers()

LayerDeviceMapper: Vec<Device>, one per layer. Forward: .to_device(&mappings[layer]).

Auto device map: layer_sizes_in_bytes(), non_mapped_size_in_bytes(), MemoryUsage::query(). Round-robin until memory fills, then CPU.

### Tensor Parallelism (distributed/layers.rs)

ColumnParallelLayer: shards weight along output dim. No all-reduce.

RowParallelLayer: shards weight along input dim. All-reduce sum.

KV head replication: compute_kv_shard() when world_size > num_kv_heads.

### Distributed Setup (distributed.rs)

prepare_distributed_mapper(): world sizes, NCCL communicator via IPC or ring, spawn workers, load shards, multi-node via env vars.

IPC: Unix socket, daemon flag, NCCL unique ID broadcast.

### MoE Expert Sharding

PackedExperts/FusedExperts: stacked (single tensor) and per-expert formats. Only UnquantLinear and BlockwiseFP8Linear support gather_forward().

### Applicability to MojoLlama

1. Layer-pipeline parallelism with per-layer device assignment
2. Tensor parallelism via row/column sharding + all-reduce
3. KV head replication for GQA models
4. Comm abstraction wrapping MAX collectives
5. Auto device mapping algorithm
6. Pre-computed device-mapped masks

---

## Area 4: Hardware Auto-Tuning

### Architecture Overview

AutoTuneRequest -> load config -> detect backend -> find GPUs -> for each ISQ candidate (estimate size via pack_factor, compute device layers, determine fit, calculate max context from remaining VRAM) -> pick best -> emit AutoTuneResult.

### Profiles

TuneProfile: Quality (Q8_0..Q2K or AFQ equivalents on Metal), Balanced (Q6K..Q3K), Fast (Q4K..Q2K).

### Candidate Evaluation

For each IsqType: pack_factor, layer_sizes_in_bytes(), get_device_layers_for_loader(), max context from remaining VRAM (kv_cache_elements_per_token * dtype_size * num_layers).

Fit: Fits (all GPU), Hybrid (GPU+CPU), TooLarge.

CLI: mistralrs tune outputs table, JSON (--json), config (--emit-config).

### Applicability to MojoLlama

1. Size estimation via pack_factor
2. Layer-by-layer memory budgeting
3. KV cache sizing via ModelConfigLike
4. Profile-guided recommendation (Quality/Balanced/Fast)
5. Hybrid CPU-GPU offloading

---

## Cross-Cutting Patterns

1. Trait-Based Abstraction Layers: QuantMethod, DeviceMapper, IsqModel, QuantizedSerde
2. Thread-Local + Thread Pool Async (immediate ISQ config propagation)
3. Channel-Based Decoupling (serving, engine, distributed workers)
4. Builder Pattern for all major components

## File References

Area 1 (Serving Layer):
- mistralrs-server-core/src/lib.rs
- mistralrs-server-core/src/mistralrs_server_router_builder.rs
- mistralrs-server-core/src/handler_core.rs
- mistralrs-server-core/src/streaming.rs
- mistralrs-server-core/src/chat_completion.rs
- mistralrs-server-core/src/completions.rs
- mistralrs-server-core/src/completion_core.rs
- mistralrs-server-core/src/types.rs
- mistralrs-server-core/src/openai.rs

Area 2 (ISQ):
- mistralrs-core/src/pipeline/isq.rs
- mistralrs-quant/src/lib.rs
- mistralrs-quant/src/utils/isq.rs
- mistralrs-quant/src/gguf/mod.rs
- mistralrs-quant/src/gguf/cpu.rs
- mistralrs-quant/src/unquantized/mod.rs

Area 3 (Multi-GPU):
- mistralrs-core/src/device_map.rs
- mistralrs-core/src/distributed.rs
- mistralrs-quant/src/distributed/layers.rs
- mistralrs-quant/src/distributed/mod.rs
- mistralrs-quant/src/distributed/socket.rs

Area 4 (Tuning):
- mistralrs-core/src/tuning.rs
- mistralrs-cli/src/main.rs
