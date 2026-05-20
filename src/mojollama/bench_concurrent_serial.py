#!/usr/bin/env python3
"""Qwen3.6 MXFP4 — Python engine serial loop concurrency benchmark."""
import sys, os, time, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"
N_USERS = 10
GEN_TOKENS = 50
WARMUP = 10

print(f"Loading Qwen3.6 MXFP4...", flush=True)
from turbo_engine_v7_moe import TurboEngineV7MoE
e = TurboEngineV7MoE(MODEL, 32)
V = e.vocab_size

# Create per-user state
class UserState:
    __slots__ = ['pos', 'logits', 'token']
    def __init__(self):
        self.pos = 0
        self.logits = None
        self.token = np.array([[1]], dtype=np.int32)

users = [UserState() for _ in range(N_USERS)]
for u in range(N_USERS):
    users[u].token = np.array([[1 + u]], dtype=np.int32)

# Warmup
print(f"Warming up ({WARMUP} tokens)...", flush=True)
e.reset()
tok = np.array([[1]], dtype=np.int32)
for _ in range(WARMUP):
    e.forward(tok)
    tok = np.array([[int(np.argmax(e._logits))]], dtype=np.int32)

# Single-user baseline
print(f"\nSingle-user baseline...", flush=True)
e.reset()
tok = np.array([[1]], dtype=np.int32)
times_1 = []
for _ in range(GEN_TOKENS):
    t0 = time.perf_counter()
    logits = e.forward(tok)
    times_1.append(time.perf_counter() - t0)
    tok = np.array([[int(np.argmax(logits))]], dtype=np.int32)
times_1 = times_1[5:]
avg_1 = np.mean(times_1)
print(f"  {avg_1*1000:.1f}ms → {1/avg_1:.1f} tok/s", flush=True)

# Multi-user: round-robin through users
print(f"\n{N_USERS}-user round-robin ({GEN_TOKENS} tokens each)...", flush=True)
e.reset()
# Reset all users
for u in range(N_USERS):
    users[u].pos = 0
    users[u].token = np.array([[1 + u]], dtype=np.int32)

t0 = time.perf_counter()
total_tokens = 0

for step in range(GEN_TOKENS):
    for u in range(N_USERS):
        logits = e.forward(users[u].token)
        users[u].logits = logits
        users[u].token = np.array([[int(np.argmax(logits))]], dtype=np.int32)
        total_tokens += 1

total_s = time.perf_counter() - t0
tps = total_tokens / total_s
per_user_tps = tps / N_USERS

print(f"  Total: {total_tokens} tokens in {total_s:.1f}s", flush=True)
print(f"  Aggregate: {tps:.1f} tok/s ({per_user_tps:.1f}/user)", flush=True)
print(f"  VS single: {1/avg_1*N_USERS:.1f} tok/s ideal", flush=True)
print(f"  Efficiency: {tps/(1/avg_1*N_USERS)*100:.0f}%", flush=True)

# Project to target
target_tps = 1/avg_1 * N_USERS * 2  # 200%
print(f"\n  Target (200%): {target_tps:.0f} tok/s", flush=True)
print(f"  Gap: {target_tps - tps:.0f} tok/s (need {target_tps/tps:.1f}x)", flush=True)
