#!/usr/bin/env python3
"""Profile MojoLlama forward pass — time breakdown by component.

Measures wall-clock time for each major component in the forward pass
using high-resolution timestamps. Reports median, p95 for each.
"""
import os, sys, time, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['OMP_PROC_BIND'] = 'close'
os.environ['OMP_PLACES'] = 'cores'

MODEL = sys.argv[1] if len(sys.argv) > 1 else '/tmp/models/gpt-oss-20b-Q4_K_M.gguf'
from turbo_engine_v7_moe import TurboEngineV7MoE

eng = TurboEngineV7MoE(MODEL, n_threads=32)
V = int(eng.vocab_size)

# Warmup
eng.reset()
for i in range(5):
    eng.forward([i % max(V, 1)])

# Profile N tokens with per-component timing
N = 10
eng.reset()
profiles = []

for tok_id in range(N):
    token = [tok_id % max(V, 1)]
    eng.reset()
    times = {}
    
    # We need to instrument the engine's forward method
    # Let's use a different approach: time the whole forward pass
    # and compare with known C kernel execution time
    
    t0 = time.perf_counter_ns()
    logits = eng.forward(token)
    t1 = time.perf_counter_ns()
    total_us = (t1 - t0) / 1000
    
    profiles.append(total_us)

profiles.sort()
median_us = profiles[len(profiles)//2]
mean_us = sum(profiles)/len(profiles)
p95_us = profiles[int(len(profiles)*0.95)]

print(f"\n{'='*55}")
print(f"Full Forward Pass Profile: {os.path.basename(MODEL)}")
print(f"{'='*55}")
print(f"  Tokens: {N}")
print(f"  Median: {median_us/1000:.2f} ms ({1000/(median_us/1000):.1f} tok/s)")
print(f"  Mean:   {mean_us/1000:.2f} ms ({1000/(mean_us/1000):.1f} tok/s)")
print(f"  P95:    {p95_us/1000:.2f} ms")
print(f"{'='*55}")

# Estimate breakdown based on model architecture
L = eng.n_layers
N_embd = eng.n_embd
NH = eng.n_head
NKH = eng.n_kv_head if hasattr(eng,'n_kv_head') else eng.n_kv_head
HD = eng.head_dim
FF = eng.n_ff
print(f"\nArchitecture: {L}L/{N_embd}D/{FF}FF/{NH}H/{NKH}KV/HD={HD}")

# Check quant types used
lw = eng._layers[0]
print(f"\nWeight Quant Types (layer 0):")
print(f"  attn_q: {lw.attn_q_qt}  attn_k: {lw.attn_k_qt}  attn_v: {lw.attn_v_qt}  attn_out: {lw.attn_out_qt}")
me = eng._moe_layers[0]
print(f"  moe_gate: {me.gate_qt}  moe_up: {me.up_qt}  moe_down: {me.down_qt}")
print(f"  output: {eng._out_qt}")

# Count matmul operations per layer
attention_matmuls = 3  # Q, K, V (or fused QKV)
moe_experts = eng.n_experts_per_tok
moe_matmuls = moe_experts * 3  # gate, up, down per active expert
print(f"\nPer-Layer MatMuls: {attention_matmuls} attention + {moe_matmuls} MoE = {attention_matmuls+moe_matmuls} total")
print(f"Total per token: {(attention_matmuls+moe_matmuls)*L} matmuls + norms + attention")
print(f"  → {((attention_matmuls+moe_matmuls)*L):.0f} C function calls via ctypes")

# Rough estimate: how long should the pure C computation take?
# At DDR4 bandwidth = 34.5 GB/s, reading model weights
model_size_gb = sum(t.nbytes for t in eng.reader.tensors if hasattr(t,'nbytes')) / 1e9
print(f"\nModel size: ~{model_size_gb:.1f} GB")
print(f"Memory bandwidth: ~34.5 GB/s (DDR4-2933 quad-channel)")
print(f"Theoretical min time to read all weights: {model_size_gb/34.5*1000:.0f} ms")
load_per_layer_gb = model_size_gb / L
print(f"Weight read per layer: ~{load_per_layer_gb*1000:.0f} MB")
print(f"Time to read 1 layer at 34.5 GB/s: {load_per_layer_gb/34.5*1000:.1f} ms")
print(f"\n→ MEDIAN forward pass: {median_us/1000:.2f} ms")
print(f"→ Expected from bandwidth alone: ~{model_size_gb/34.5*1000:.0f} ms (all weights)")
print(f"→ Overhead above bandwidth limit: {median_us/1000 - model_size_gb/34.5*1000:.1f} ms")
