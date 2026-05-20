#!/usr/bin/env python3
"""Test loading DeepSeek V4 Flash - just metadata and a few tensors."""
import sys, os
sys.path.insert(0, '/onedev-workspace/work/src')

from mojollama.kernels.turbo_engine_dsv4 import SafetensorsLoader, DeepSeekV4Config

cfg = DeepSeekV4Config(n_layers=1)
print(f"Config: {cfg}")

print("Loading metadata...")
loader = SafetensorsLoader(cfg.model_id)
print(f"Loaded {len(loader.shard_of)} tensor entries across {len(loader._shard_cache)} shards")

# Check if we can load the embed weight
print("\nLoading embed.weight...")
try:
    embed = loader.get_tensor('embed.weight')
    print(f"  embed.weight: {embed.shape}, {embed.dtype}, {embed.nbytes/1e6:.1f} MB")
except Exception as e:
    print(f"  Error: {e}")

# Check a few layer 0 tensors
print("\nLoading layer 0 weights...")
for name in ['attn_norm.weight', 'attn.wq_a.qweight', 'attn.wq_a.qzeros', 'attn.wq_a.scales',
             'attn.wkv.qweight']:
    try:
        t = loader.get_tensor(f"layers.0.{name}")
        print(f"  layers.0.{name}: {t.shape}, {t.dtype}")
    except KeyError as e:
        print(f"  layers.0.{name}: NOT FOUND")

# Try dequantizing one weight
print("\nTesting W4A16 dequantize...")
from mojollama.kernels.turbo_engine_dsv4 import dequantize_w4a16
try:
    qw = loader.get_tensor('layers.0.attn.wq_a.qweight')
    qz = loader.get_tensor('layers.0.attn.wq_a.qzeros')
    sc = loader.get_tensor('layers.0.attn.wq_a.scales')
    print(f"  qweight: {qw.shape}, qzeros: {qz.shape}, scales: {sc.shape}")
    deq = dequantize_w4a16(qw, qz, sc)
    print(f"  dequantized: {deq.shape}, mean={deq.mean():.4f}, std={deq.std():.4f}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback; traceback.print_exc()

# Check a few expert weights
print("\nLoading expert weights for layer 0...")
for expert_id in [0, 1, 2]:
    for wname in ['w1', 'w2', 'w3']:
        try:
            t = loader.get_tensor(f"layers.0.ffn.experts.{expert_id}.{wname}.qweight")
            print(f"  expert {expert_id} {wname}: {t.shape}")
        except KeyError:
            print(f"  expert {expert_id} {wname}: NOT FOUND")

loader.close()
print("\nDone!")
