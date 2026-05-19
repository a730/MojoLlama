# mistral.rs Concurrency & Throughput Architecture — MojoLlama Applicable Patterns

## 1. Server Concurrency Model

### axum + tokio HTTP Layer
- Each HTTP request spawns a tokio task.
- Per-request `mpsc::channel(10_000)` created. `Sender` attached to `NormalRequest.response`.
- Axum router built by `MistralRsServerRouterBuilder`.

### Multi-Model Management
- `MistralRs` struct owns `HashMap<String, EngineInstance>` — one per model.
- Each model gets a **dedicated OS thread** with its own `tokio::runtime::Runtime`.
- Each runtime runs a single `Engine::run()` async task.
- `get_sender(model_id)` resolves model aliases, auto-reloads unloaded models.

### Model Loading Flow
```rust
// Load a model: spawns a thread, creates engine, channels everything
fn load_model_from_config(config: ModelConfig) -> Result<Arc<MistralRs>> {
    // 1. Build pipeline (model architecture)
    // 2. Create scheduler + KV cache manager
    // 3. Spawn OS thread: `std::thread::spawn(move || { runtime.block_on(engine.run()) })`
    // 4. Return Arc<MistralRs> with sender channel stored
}
```

**Threading model:**
```
Client 1 ──HTTP──┐
Client 2 ──HTTP──┤
Client 3 ──HTTP──┤  axum router (tokio tasks)
                 │
                 ▼
           mpsc::channel (per-request Sender)
                 │
                 ▼
      ┌──────────────────┐
      │  Engine::run()   │  ← Single OS thread per model
      │  (event loop)    │
      └──────────────────┘
              │
              ▼
      Scheduler → Pipeline::step() → forward → sample → respond
```

### MojoLlama Pattern:
```mojo
# Per-model event loop in its own thread
struct EngineInstance:
    var model_id: String
    var request_rx: Channel[Request]
    var scheduler: Scheduler
    var pipeline: Pipeline
    
    fn run(out self):
        while self.running:
            # Drain channel
            while var req = self.request_rx.try_recv():
                if req.type == REQUEST_TERMINATE:
                    self.running = False
                    break
                self.scheduler.add_request(req)
            
            # Schedule and step
            if var batch = self.scheduler.schedule():
                self.pipeline.step(batch)
            
            # Respond to completed sequences
            self.dispatch_responses()
            
            # Idle wait
            if nothing_to_do:
                self.request_rx.wait()
```

**Key insight:** One OS thread per model with async channels is simpler and more predictable than a thread-per-request model. Mojo's `@parameter` + threading can replicate this cleanly.

---

## 2. The Engine Event Loop

From `mistralrs-core/src/engine/mod.rs`:

```rust
pub async fn run(&mut self) {
    loop {
        // 1. Drain channel: collect all pending requests
        while let Ok(normal_or_terminate) = self.rx.try_recv() {
            match normal_or_terminate {
                Request::Normal(req) => {
                    if let Err(e) = self.add_request(req).await { ... }
                }
                Request::Terminate => { ... break; }
            }
        }

        // 2. Schedule: form a batch from waiting + running sequences
        if self.scheduler.schedule()? {
            let batch = self.scheduler.get_batch()?;
            // 3. Step: forward pass + sampling
            let result = self.pipeline.step(self.scheduler, batch).await;
            // 4. Handle output: stream to channels, mark done
            self.process_step_output(result);
        }

        // 5. Cleanup: remove finished sequences
        self.scheduler.remove_done_seqs();

        // 6. Wait if idle
        if is_idle {
            select! {
                _ = self.rx.recv() => {},
                _ = self.pending_notify.notified() => {},
            }
        }
    }
}
```

**Key design decisions:**
- **Single-threaded event loop** — no contention, no locks in hot path
- **Drain-then-process** — all pending requests collected before scheduling
- **Synchronous step** — `Pipeline::step()` is blocking (GPU kernel launch), no async in hot path
- **`pending_notify`** — scheduler wakes the engine when a sequence becomes ready (e.g., after waiting for preempted sequence)

### MojoLlama Pattern:
```mojo
fn engine_loop(out self):
    while self.active:
        # Step 1: Drain incoming requests
        for request in self.request_queue.drain_all():
            self.handle_request(request)
        
        # Step 2: Try to schedule a batch
        if var batch = self.scheduler.schedule():
            # Step 3: Run inference (blocking, SIMD/MAX kernel)
            var outputs = self.pipeline.step(batch)
            
            # Step 4: Dispatch per-sequence responses
            for output in outputs:
                if output.completed:
                    self.respond(output)  # Send via mpsc channel
                else:
                    self.scheduler.requeue(output.sequence)
        
        # Step 5: Cleanup
        self.scheduler.purge_completed()
        
        # Step 6: Backpressure — yield if nothing to do
        if self.request_queue.empty() and self.scheduler.idle():
            self.request_queue.wait_for_request()
```

---

## 3. DefaultScheduler (Fixed-Slot, Normal KV Cache)

From `mistralrs-core/src/scheduler/default_scheduler.rs`:

### Data Structures
```rust
pub struct DefaultScheduler {
    waiting: VecDeque<Sequence>,         // Not yet admitted
    running: Vec<Sequence>,              // Currently active (decode loop)
    method: DefaultSchedulerMethod,      // Fixed(N) or MaxThroughput
    n_grace: f32,                        // Grace period for preempted sequences
}
```

### Admission Policy
- `DefaultSchedulerMethod::Fixed(N)`: At most `N` sequences in `running` at any time
- A waiting sequence is admitted when `running.len() < max_seqs`
- No KV memory admission check (assumes fits)

### Scheduling Algorithm (called every engine loop iteration)

```rust
fn schedule(&mut self) -> bool {
    // 1. Remove completed sequences from running
    self.running.retain(|s| !s.is_done());

    // 2. Try to admit waiting sequences
    while self.running.len() < self.max_seqs && !self.waiting.is_empty() {
        let seq = self.waiting.pop_front().unwrap();
        self.running.push(seq);
    }
    
    // 3. If no running sequences, nothing to do
    if self.running.is_empty() {
        return Ok(false);
    }

    // 4. Length-based bucketing: group by (seq_len, has_images, token_offset)
    let buckets = self.bucket_sequences(&self.running);
    
    // 5. Pick the bucket with the shortest sequence length
    let chosen = self.select_bucket(&buckets);
    
    // 6. Preempt all sequences NOT in the chosen bucket
    //    (with urgency boost to prevent starvation)
    for seq in &mut self.running {
        if !chosen.contains(seq.id()) {
            seq.preempt_with_urgency_boost();
            self.waiting.push_front(seq);
        }
    }
    
    // 7. Build batch from chosen bucket
    self.current_batch = Some(Batch::from(chosen));
    Ok(true)
}
```

### Bucket Selection
```rust
fn bucket_sequences(&self, seqs: &[Sequence]) -> Vec<Bucket> {
    // Group by (padded_length, has_images && is_prompt, token_offset)
    // padded_length = ceil(log2(seq_len))  (power-of-2 padding)
    let mut buckets: HashMap<(usize, bool, usize), Vec<Sequence>> = HashMap::new();
    for seq in seqs {
        let key = (
            seq.padded_length(),
            seq.has_images() && seq.is_prompt(),
            seq.token_offset(),
        );
        buckets.entry(key).or_default().push(seq.clone());
    }
    buckets.into_values().map(Bucket).collect()
}

fn select_bucket(&self, buckets: &[Bucket]) -> &Bucket {
    // Pick the bucket with the minimum sequence length
    // (shortest sequences run first — maximizes throughput)
    buckets.iter().min_by_key(|b| b.min_length()).unwrap()
}
```

**Why length-based bucketing?** Padding all sequences to the same length wastes computation. By grouping same-length sequences together, the padding overhead is eliminated. The shortest bucket runs first (fastest to finish, frees slots).

### Preemption with Urgency Boost
```rust
fn preempt(&mut self, seq: &mut Sequence) {
    seq.waiting_count += 1;  // Starvation counter
    self.waiting.push_front(seq.clone());
}
```

Each time a sequence is preempted, its `waiting_count` increases. The selection algorithm biases toward high-count sequences. After `MAX_PREEMPTIONS` (64), a running sequence is forcibly evicted.

### MojoLlama Pattern:
```mojo
struct DefaultScheduler:
    var waiting: List[Sequence]
    var running: List[Sequence]
    var max_seqs: Int
    var current_batch: Batch
    
    fn schedule(out self) -> Bool:
        # 1. Remove done sequences
        self.running = self.running.filter(not done)
        
        # 2. Admit waiting up to capacity
        while self.running.size() < self.max_seqs:
            if var seq = self.waiting.pop_front():
                self.running.append(seq)
            else: break
        
        # 3. Bucket by sequence length (power-of-2 groups)
        var buckets = self.bucket_sequences(self.running)
        if buckets.size() == 0: return False
        
        # 4. Pick shortest bucket (maximize throughput)
        var chosen = self.select_shortest_bucket(buckets)
        
        # 5. Preempt others with starvation awareness
        for seq in self.running:
            if not chosen.contains(seq):
                seq.starvation_counter += 1
                self.waiting.push_front(seq)
        
        self.running = chosen.to_list()
        self.current_batch = Batch(self.running)
        return True
    
    fn bucket_sequences(self, seqs: List[Sequence]) -> List[Bucket]:
        # Group by (log2_padded_length, is_prefill)
        var buckets = Dict[(Int, Bool), Bucket]()
        for seq in seqs:
            var key = (seq.padded_length_log2(), seq.is_prefill)
            buckets[key].append(seq)
        return buckets.values()
```

---

## 4. PagedAttentionScheduler (Block-Level KV Cache)

From `mistralrs-core/src/paged_attention/scheduler.rs`:

### Block-Level Admission Control
```rust
fn schedule_block_level(...) -> bool {
    // 1. Compute which KV cache blocks each running sequence needs
    // 2. Check if blocks are available in the cache pool
    
    // For each sequence in running, ensure it has enough allocated blocks
    // If not enough blocks, preempt the lowest-urgency sequence
    
    // For each sequence in waiting, check if we can allocate its first block
    // Blocks are allocated on-demand, not upfront
}
```

### KV Cache Allocation
```rust
fn allocate_slots(&mut self, seq: &mut Sequence, n_tokens: usize) -> bool {
    let blocks_needed = ceil_div(seq.seq_len + n_tokens, BLOCK_SIZE) - seq.num_blocks;
    // Check pool; if insufficient, try LRU eviction from inactive sequences
    // If still insufficient, return false (sequence not admitted)
}
```

### Prefix Caching
```rust
// Each block has a content hash
// On admission, check if any existing block has the same hash
// If so, reuse it (increment reference count)
fn get_computed_blocks(&self, seq: &Sequence) -> Vec<BlockId> {
    let mut blocks = Vec::new();
    for token in seq.tokens.iter_chunks(BLOCK_SIZE) {
        let hash = compute_hash(token);
        if let Some(block) = self.block_hashtable.get(&hash) {
            blocks.push(block.id);
        } else {
            break;  // Prefix match only
        }
    }
    blocks
}
```

### Starvation Prevention
```rust
// After 64 scheduling passes without being picked for inference,
// the waiting sequence forces a running sequence to be preempted
const MAX_STARVATION_PASSES: u32 = 64;

fn check_starvation(&mut self) {
    for seq in &self.waiting {
        seq.passes_without_inference += 1;
        if seq.passes_without_inference > MAX_STARVATION_PASSES {
            // Force-preempt the lowest-urgency running sequence
            let victim = self.running.iter().min_by_key(|s| s.urgency()).unwrap();
            self.preempt(victim);
            break;
        }
    }
}
```

### MojoLlama Pattern:
```mojo
struct PagedAttentionScheduler:
    var block_pool: BlockPool         # Physical KV cache blocks
    var block_table: Dict[Int, List[Int]]  # Logical → physical mapping per sequence
    var waiting: List[Sequence]
    var running: List[Sequence]
    
    fn allocate_kv_blocks(inout self, seq: Sequence, n_tokens: Int) -> Bool:
        var needed = (seq.seq_len + n_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        var current = self.block_table[seq.id].size()
        var allocate = needed - current
        
        for _ in range(allocate):
            if var block = self.block_pool.allocate():
                self.block_table[seq.id].append(block)
            else:
                # Try evicting LRU from completed sequences
                self.evict_lru()
                if var block = self.block_pool.allocate():
                    self.block_table[seq.id].append(block)
                else:
                    return False  # Out of memory
        return True
    
    fn schedule(out self) -> Bool:
        # 1. Check starvation
        self.check_starvation()
        
        # 2. Try admitting waiting sequences
        for seq in self.waiting:
            if self.allocate_kv_blocks(seq, 0):
                self.running.append(seq)
            # else: stays in waiting, retried next iteration
        
        # 3. Build batch from all running sequences
        #    (PagedAttention can handle variable-length sequences)
        if self.running.size() > 0:
            self.current_batch = Batch(self.running)
            return True
        return False
```

---

## 5. Pipeline::step() — The Core Inference Step

From `mistralrs-core/src/pipeline/mod.rs` and `mistralrs-core/src/pipeline/normal.rs`:

```rust
async fn step(&mut self, scheduler: &mut Box<dyn Scheduler>, batch: Batch) -> Result<()> {
    // ---- Phase 1: Cache Pre-Op ----
    // For each sequence in batch, ensure KV cache is contiguous
    // (clone cache tensors or reset for new sequences)
    for seq in batch.iter() {
        self.cache_manager.clone_in_cache(seq)?;
    }
    
    // ---- Phase 2: Input Processing ----
    // Tokenize + multimodal processing
    let input = self.process_inputs(&batch).await?;
    // input.input_ids: Tensor [batch_size, seq_len]
    // input.position_ids: Tensor [batch_size, seq_len]
    
    // ---- Phase 3: Forward Pass ----
    // Model forward (blocking GPU call)
    let logits = self.model.forward(
        input.input_ids,
        input.position_ids,
        input.cache_input,
    )?;
    
    // ---- Phase 4: Output Splitting ----
    // Split the [batch, seq_len, vocab] logits per sequence
    let per_seq_logits = self.split_logits(logits, &batch);
    
    // ---- Phase 5: Cache Post-Op ----
    // Copy KV cache from scratch space to sequence storage
    for seq in batch.iter() {
        self.cache_manager.clone_out_cache(seq)?;
    }
    
    // ---- Phase 6: Sampling ----
    // For each sequence: sample token, check stop, stream response
    for (seq, logits) in batch.iter().zip(per_seq_logits) {
        let token = self.sample(logits, seq.sampling_params)?;
        let done = seq.append_token(token);
        if done {
            self.send_response(seq, Response::Done(last_token));
        } else {
            self.send_response(seq, Response::Chunk(token));
        }
    }
    
    Ok(())
}
```

### Prefill-then-Decode Ordering

From the scheduler's `schedule()` method, completions have priority:
1. **Completions first** (decode): sequences already in `running` that are generating tokens
2. **Prompts second** (prefill): newly admitted sequences that need their first forward pass

This ensures prefilling a long prompt doesn't block ongoing generation.

### MojoLlama Pattern:
```mojo
struct Pipeline:
    var model: LlamaModel
    var cache_manager: CacheManager
    var sampler: Sampler
    
    fn step(inout self, batch: Batch) -> List[StepOutput]:
        # Phase 1: Prepare KV cache
        for seq in batch:
            self.cache_manager.prepare(seq)
        
        # Phase 2: Process inputs
        var input_ids = self.prepare_inputs(batch)
        
        # Phase 3: Forward
        var logits = self.model.forward(
            input_ids,
            self.cache_manager.get_kv_cache(batch)
        )
        
        # Phase 4: Split per-sequence logits
        var per_seq = self.split_logits(logits, batch)
        
        # Phase 5: Commit KV cache
        for seq in batch:
            self.cache_manager.commit(seq)
        
        # Phase 6: Sample and stream
        var outputs = List[StepOutput]()
        for i in range(batch.size()):
            var token = self.sampler.sample(per_seq[i], batch[i].params)
            var done = batch[i].append(token)
            outputs.append(StepOutput(batch[i].id, token, done))
        
        return outputs
```

---

## 6. Streaming Architecture

From `mistralrs-server-core/src/streaming.rs` and `chat_completion.rs`:

### Per-Request Channel
```rust
// When a request arrives:
let (tx, rx) = mpsc::channel::<Response>(10_000);
let request = NormalRequest {
    response: RequestResponse(tx),  // Sender embedded in request
    ...
};

// Engine sends responses through this channel:
let _ = request.response.0.send(Response::Chunk(token));

// Server reads from the channel and formats as SSE:
pub struct BaseStreamer {
    receiver: tokio::sync::mpsc::Receiver<Response>,
    state: DoneState,
}
```

### SSE Format
```
data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"choices":[{"delta":{"content":" world"},"index":0}]}

data: [DONE]
```

### Keep-Alive
- Every 10 seconds of no data, a keep-alive comment is sent.
- Implemented with `tokio::time::interval` in the streamer.

### SequenceGroup for Multi-Choice
- A `SequenceGroup` contains `n_choices` sequences.
- Chunk is sent to the client only when ALL choices have output ready.
- When all choices are done, `Done` is sent.

### MojoLlama Pattern:
```mojo
# Per-request response channel
struct Request:
    var prompt: String
    var params: SamplingParams
    var response_tx: MPSCSender[StepOutput]  # Channel back to HTTP handler
    var id: Int

# HTTP handler — SSE streaming
fn handle_chat_completion(request: ChatRequest) -> Response:
    var (tx, rx) = MPSCChannel[StepOutput](10_000)
    engine.send_request(Request(prompt, params, tx))
    
    # Stream response as SSE
    response_stream = rx.to_stream().map(|output| 
        SSE(data=output.to_json())
    )
    
    return Response(
        content_type="text/event-stream",
        body=response_stream.chain(SSE("[DONE]"))
    )
```

---

## 7. KV Cache Management

### Normal Cache (DefaultScheduler)
- Per-layer `SingleCache`: grows in 512-token increments
- Batching via `Tensor::cat` (concat) and `Tensor::chunk` (split)
- Simple but wastes padding space

### PagedAttention Cache (PagedAttentionScheduler)
- Block pool with configurable GPU memory limit (`--cache-free-memory`)
- Blocks allocated on demand, reference-counted
- LRU eviction to CPU when GPU memory runs low
- Prefix caching via content hashing

### Hybrid Cache
- For MoE/recurrent models: attention layers use KV cache, recurrent layers use state pool
- State pool: index-based gather/scatter

### MojoLlama Pattern:
```mojo
# Block-based KV cache (PagedAttention-style)
struct BlockPool:
    var blocks: NDBuffer   # [num_blocks, block_size, num_heads, head_dim]
    var free_list: List[Int]
    var lru_queue: List[Int]
    
    fn allocate(out self) -> Option[Int]:
        if self.free_list.size() > 0:
            return self.free_list.pop()
        # Evict LRU
        if self.lru_queue.size() > 0:
            var victim = self.lru_queue.pop_front()
            self.evict_to_cpu(victim)
            return victim
        return None
    
    fn evict_to_cpu(self, block_id: Int):
        # Copy block from GPU to CPU swap
        self.cpu_swap.store(self.blocks[block_id, :, :, :])

# Cache manager with prefix caching
struct CacheManager:
    var pool: BlockPool
    var block_tables: Dict[Int, List[Int]]  # seq_id → [physical_block_ids]
    var content_hashes: Dict[Int, Int]       # hash → physical_block_id
    
    fn compute_prefix_blocks(self, tokens: List[Int]) -> List[Int]:
        var result = List[Int]()
        for chunk in tokens.chunks(BLOCK_SIZE):
            var h = hash(chunk)
            if self.content_hashes.contains(h):
                result.append(self.content_hashes[h])
            else: break
        return result
```

---

## 8. Request Flow Summary

```
HTTP Request
     │
     ▼
axum route handler
     │
     ├─ NormalRequest { messages, sampling_params, response_tx }
     │
     ▼
mpsc::channel[10000]
     │
     ▼
┌────────────────────────────────────────┐
│  Engine::run()  [single OS thread]      │
│                                         │
│  Loop:                                  │
│    drain channel                         │
│    scheduler.schedule() → batch         │
│    pipeline.step(batch) → outputs       │
│      ├─ model.forward()                 │
│      │     ├─ input_ids, cache          │
│      │     └─ logits                    │
│      └─ sampler.sample() → tokens       │
│            ├─ Response::Chunk(tx) → SSE │
│            └─ Response::Done(tx) → [DONE]│
│    scheduler.remove_done()              │
└────────────────────────────────────────┘
     │
     ▼
SSE stream to HTTP client
```

---

## 9. Benchmarking Concurrency

```bash
# Single request throughput
mistralrs bench -m Qwen/Qwen3-4B --isq Q4_0

# Concurrent requests (server mode)
mistralrs serve -m Qwen/Qwen3-4B --isq Q4_0

# Then bench with something like:
# oha -c 4 -n 100 http://localhost:1234/v1/chat/completions -m POST ...

# Key metrics to measure on MojoLlama:
# - tok/s at batch_size=1 (decode latency)
# - tok/s at batch_size=4,8,16 (continuous batching throughput)
# - p50/p95 TTFT (time to first token)
# - Max concurrent sequences before OOM
# - Token throughput decay as context grows
```

## 10. 10 Design Patterns MojoLlama Should Adopt (Ranked)

| Priority | Pattern | Why | Mojo Impl |
|----------|---------|-----|-----------|
| P0 | **Single-threaded engine per model** | No locking in hot path, deterministic scheduling | `@parameter` + dedicated OS thread per model |
| P0 | **Continuous batching** | Max GPU utilization during decode | Batch formed every `step()`, sequences exit when done |
| P0 | **mpsc channel dispatching** | Decoupled HTTP and inference, backpressure | `MPSCChannel[Request]` between server and engine |
| P0 | **Prefill-then-decode ordering** | Decode never blocked by long prompts | Sort batch: decodes first, prefills second |
| P0 | **PagedAttention block KV cache** | Memory proportional to active tokens, not max context | Block pool + page table + LRU eviction |
| P1 | **Length-based bucketing** | Minimizes padding waste | Group sequences by `ceil(log2(seq_len))` |
| P1 | **SSE streaming via per-request channel** | Low-latency token delivery | `MPSCChannel[StepOutput]` per request |
| P1 | **Prefix caching with content hashing** | Reuse KV blocks across requests | Hash-based block dedup in page table |
| P2 | **Starvation-aware scheduling** | Fairness under load | Urgency counter + forced preemption after N passes |
| P2 | **SequenceGroup for multi-choice** | Coordinate parallel sampled responses | Group `n` sequences, dispatch only when all ready |
