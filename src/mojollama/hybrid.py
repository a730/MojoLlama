#!/usr/bin/env python3
"""MojoLlama Hybrid Inference Engine — Mojo SIMD + Python multiprocessing.

Architecture:
  - Weights pre-loaded into shared memory (numpy memmap via /dev/shm/)
  - Forward pass orchestrated in Python (layer loop, norms, attention)
  - Q4_0 matmuls dispatched to Mojo SIMD worker processes
  - Workers run in parallel across all physical CPU cores
  - Workers communicate via pipes (stdin/stdout binary protocol)

This avoids all Mojo 1.0.0b1 compiler limitations:
  ✓ No unsafe_from_address needed (workers read weights from files)
  ✓ No PythonObject→Int conversion needed (CLI args as strings)
  ✓ No free() import needed (process exits cleanly after work)
  ✓ Multi-threading via multiprocessing (not Mojo threads)

Usage:
  python3 mojollama_hybrid.py --model model.gguf --prompt "Hello"
"""

import os
import sys
import json
import time
import math
import struct
import subprocess
import multiprocessing as mp
import numpy as np
from pathlib import Path

# ─── Q4_0 Constants ──────────────────────────────────────────────────

Q4_BLOCK_SIZE = 32  # values per block
Q4_TYPE_SIZE = 18    # bytes per block (2 scale + 16 nibbles)

# Path to compiled Mojo worker binary
MOJO_WORKER = os.path.join(
    os.path.dirname(__file__), "mojollama_q4_pipe"
)


# ─── Weight Manager ───────────────────────────────────────────────────

class WeightStore:
    """Manages Q4_0 weights in shared memory for parallel access."""

    def __init__(self, gguf_path: str):
        import gguf
        from gguf.constants import GGMLQuantizationType
        
        self.path = gguf_path
        self.reader = gguf.GGUFReader(gguf_path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self._load_config()
        
        # Load and extract Q4_0 weights into raw block format
        self.raw_weights = {}  # name -> (n_rows, n_cols, raw_bytes)
        self.f32_weights = {}  # name -> np.ndarray (for norms, embed)
        
        for name, t in self.tensors.items():
            tt = t.tensor_type if hasattr(t, 'tensor_type') else None
            if tt == GGMLQuantizationType.Q4_0:
                raw = np.asarray(t.data)
                n_rows, row_bytes = raw.shape
                n_cols = (row_bytes // Q4_TYPE_SIZE) * Q4_BLOCK_SIZE
                self.raw_weights[name] = (n_rows, n_cols, raw.tobytes())
            else:
                arr = gguf.dequantize(t.data, tt) if tt else np.array(t.data, dtype=np.float32)
                self.f32_weights[name] = arr
    
    def _load_config(self):
        f = self.reader.get_field
        arch = f('general.architecture')
        self.arch = str(arch.parts[-1]) if arch else 'unknown'
        prefix = f'{self.arch}.'
        self.n_layers = self._int(f(f'{prefix}block_count'))
        self.n_embd = self._int(f(f'{prefix}embedding_length'))
        self.n_head = self._int(f(f'{prefix}attention.head_count'))
        self.n_kv_head = self._int(f(f'{prefix}attention.head_count_kv'))
        self.n_ff = self._int(f(f'{prefix}feed_forward_length'))
        self.norm_eps = self._float(f(f'{prefix}attention.layer_norm_rms_epsilon'))
    
    def _int(self, f): return int(f.parts[-1].item()) if f and hasattr(f.parts[-1], 'item') else (int(f.parts[-1]) if f else 0)
    def _float(self, f): return float(f.parts[-1].item()) if f and hasattr(f.parts[-1], 'item') else (float(f.parts[-1]) if f else 0.0)
    
    def get_q4(self, name):
        """Get Q4_0 weight info for a tensor."""
        return self.raw_weights.get(name)
    
    def get_f32(self, name):
        """Get float32 weight for norms, embed."""
        return self.f32_weights.get(name)


# ─── Mojo Worker Pool ────────────────────────────────────────────────

class MojoWorkerPool:
    """Pool of Mojo subprocess workers for parallel Q4_0 matmul."""

    def __init__(self, weight_store: WeightStore, n_workers=None):
        self.ws = weight_store
        self.n_workers = n_workers or mp.cpu_count()
        self._worker_bin = MOJO_WORKER
        
        if not os.path.exists(self._worker_bin):
            print(f"⚠️ Mojo worker not found at {self._worker_bin}")
            print("   Fallback: using numpy dequant + BLAS")
            self._fallback = True
        else:
            self._fallback = False
    
    def matmul(self, weight_name, x):
        """Compute x @ W.T for a Q4_0 weight using parallel Mojo workers."""
        info = self.ws.get_q4(weight_name)
        if info is None or self._fallback:
            # Fallback to numpy
            w = self.ws.get_f32(weight_name)
            if w is None:
                raise ValueError(f"Weight not found: {weight_name}")
            # w might be gguf shape [cols, rows] → we need x @ w.T
            if w.ndim == 2:
                return x @ w.T
            return x @ w
        
        n_rows, n_cols, raw_bytes = info
        batch = x.shape[0] if x.ndim > 1 else 1
        
        # For small batches, use single worker
        if batch == 1 and n_rows <= 2048:
            return self._single_worker(raw_bytes, n_rows, n_cols, x)
        
        # Parallel: split rows across workers
        return self._parallel_matmul(raw_bytes, n_rows, n_cols, x)
    
    def _single_worker(self, raw_bytes, n_rows, n_cols, x):
        """Single Mojo worker (no parallel overhead for small jobs)."""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            f.write(raw_bytes)
            wpath = f.name
        
        inp = x.flatten().astype(np.float32).tobytes()
        
        try:
            proc = subprocess.run(
                [self._worker_bin, wpath, str(n_rows), str(n_cols), "0", str(n_rows)],
                input=inp, capture_output=True, timeout=60
            )
            if proc.returncode == 0:
                result = np.frombuffer(proc.stdout, dtype=np.float32)
                return result.reshape(x.shape[:-1] + (n_rows,))
            else:
                raise RuntimeError(f"Worker failed: {proc.stderr.decode()[:200]}")
        finally:
            os.unlink(wpath)
    
    def _parallel_matmul(self, raw_bytes, n_rows, n_cols, x):
        """Parallel Mojo workers via file-based weight sharing."""
        # Write weights to shared memory once
        wpath = "/dev/shm/mojo_weights.bin"
        if not os.path.exists(wpath):
            with open(wpath, 'wb') as f:
                f.write(raw_bytes)
        
        batch = x.shape[0] if x.ndim > 1 else 1
        x_2d = x.reshape(-1, n_cols) if x.ndim == 1 else x
        
        # Split rows across workers
        chunk_size = max(1, n_rows // self.n_workers)
        chunks = []
        start = 0
        while start < n_rows:
            end = min(start + chunk_size, n_rows)
            chunks.append((start, end))
            start = end
        
        # Launch workers
        with mp.Pool(min(self.n_workers, len(chunks))) as pool:
            results = pool.starmap(
                self._run_worker,
                [(wpath, n_rows, n_cols, s, e, x_2d[b % batch].tobytes())
                 for b in range(batch) for i, (s, e) in enumerate(chunks)]
            )
        
        # Combine results
        out = np.zeros((batch, n_rows), dtype=np.float32)
        idx = 0
        for b in range(batch):
            for s, e in chunks:
                n_chunk = e - s
                out[b, s:e] = np.frombuffer(results[idx], dtype=np.float32)[:n_chunk]
                idx += 1
        
        return out.reshape(x.shape[:-1] + (n_rows,))
    
    def _run_worker(self, wpath, n_rows, n_cols, start, end, inp_bytes):
        """Run a single Mojo worker subprocess."""
        proc = subprocess.run(
            [self._worker_bin, wpath, str(n_rows), str(n_cols),
             str(start), str(end)],
            input=inp_bytes, capture_output=True, timeout=30
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Worker [{start},{end}): {proc.stderr.decode()[:100]}")
        return proc.stdout


# ─── Hybrid Forward Engine ───────────────────────────────────────────

class HybridEngine:
    """Mojo + Python hybrid forward pass engine."""

    def __init__(self, gguf_path: str):
        self.ws = WeightStore(gguf_path)
        self.pool = MojoWorkerPool(self.ws)
        self._kv_cache = None
        self._cached_len = 0
        self._timing = {}
    
    def embed(self, input_ids):
        w = self.ws.get_f32('token_embd.weight')
        if w is None:
            raise ValueError("token_embd.weight not found")
        if w.ndim == 2 and w.shape[0] == self.ws.n_embd:
            w = w.T
        return w[input_ids].astype(np.float32)
    
    def rms_norm(self, x, weight_name):
        w = self.ws.get_f32(weight_name)
        ss = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(ss + self.ws.norm_eps) * w
    
    def rope(self, x, cos, sin):
        n, h, d = x.shape
        x2 = x.reshape(n, h, d // 2, 2)
        xr = np.stack([-x2[..., 1], x2[..., 0]], axis=-1)
        c = cos[:n, np.newaxis, :d // 2, np.newaxis]
        s = sin[:n, np.newaxis, :d // 2, np.newaxis]
        return (x2 * c + xr * s).reshape(n, h, d)
    
    def silu(self, x):
        return x / (1 + np.exp(-x))
    
    def attention(self, q, k, v, mask):
        d = q.shape[-1]
        scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(d) + mask
        att = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        att = att / np.sum(att, axis=-1, keepdims=True)
        return att @ v
    
    def _precompute_freqs(self, dim, end, theta=500000.0):
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(end).astype(np.float32)
        f = np.outer(t, freqs)
        return np.cos(f), np.sin(f)
    
    def forward(self, input_ids):
        """Full forward pass."""
        seq_len = len(input_ids)
        head_dim = self.ws.n_embd // self.ws.n_head
        n_rep = self.ws.n_head // self.ws.n_kv_head
        cos, sin = self._precompute_freqs(head_dim, seq_len)
        
        # KV cache mode
        if self._kv_cache is not None:
            cached_len = self._cached_len
            new_ids = np.array(input_ids[cached_len:])
            if len(new_ids) == 0:
                return None
            h = self.embed(new_ids)
            return self._forward_step(h, cos, sin, head_dim, n_rep, cached_len)
        
        # Full prefill
        h = self.embed(np.array(input_ids))
        mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), 1)
        return self._forward_full(h, cos, sin, mask, head_dim, n_rep)
    
    def _forward_full(self, h, cos, sin, mask, head_dim, n_rep):
        seq_len = h.shape[0]
        self._kv_cache = []
        
        for i in range(self.ws.n_layers):
            t0 = time.time()
            r = h
            h = self.rms_norm(h, f'blk.{i}.attn_norm.weight')
            t1 = time.time()
            
            # Q4_0 matmuls (dispatched to Mojo workers)
            q = self.pool.matmul(f'blk.{i}.attn_q.weight', h)
            k = self.pool.matmul(f'blk.{i}.attn_k.weight', h)
            v = self.pool.matmul(f'blk.{i}.attn_v.weight', h)
            t2 = time.time()
            
            q = q.reshape(seq_len, self.ws.n_head, head_dim)
            k = k.reshape(seq_len, self.ws.n_kv_head, head_dim)
            v = v.reshape(seq_len, self.ws.n_kv_head, head_dim)
            q = self.rope(q, cos, sin)
            k = self.rope(k, cos, sin)
            self._kv_cache.append((k.copy(), v.copy()))
            
            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=1)
                v = np.repeat(v, n_rep, axis=1)
            
            q_t = q.transpose(1, 0, 2)
            k_t = k.transpose(1, 0, 2)
            v_t = v.transpose(1, 0, 2)
            att = self.attention(q_t, k_t, v_t, mask)
            att = att.transpose(1, 0, 2).reshape(seq_len, self.ws.n_embd)
            h = r + self.pool.matmul(f'blk.{i}.attn_output.weight', att)
            t3 = time.time()
            
            # FFN
            r = h
            h = self.rms_norm(h, f'blk.{i}.ffn_norm.weight')
            gate = self.silu(self.pool.matmul(f'blk.{i}.ffn_gate.weight', h))
            up = self.pool.matmul(f'blk.{i}.ffn_up.weight', h)
            h = r + self.pool.matmul(f'blk.{i}.ffn_down.weight', gate * up)
            t4 = time.time()
            
            self._timing[i] = {
                "norm": t1 - t0,
                "matmul_qkv": t2 - t1,
                "attn_proj": t3 - t2,
                "ffn": t4 - t3,
                "total": t4 - t0,
            }
        
        h = self.rms_norm(h, 'output_norm.weight')
        logits = self.pool.matmul('output.weight', h) if 'output.weight' in self.ws.raw_weights else h @ self.ws.get_f32('token_embd.weight')[1].T
        self._cached_len = seq_len
        return logits
    
    def generate(self, prompt_ids, max_tokens=50):
        """Generate tokens."""
        ids = list(prompt_ids)
        out = []
        self._kv_cache = None
        self._cached_len = 0
        
        for step in range(max_tokens):
            t0 = time.time()
            logits = self.forward(ids)
            if logits is None:
                break
            nid = int(np.argmax(logits[-1]))
            elapsed = time.time() - t0
            out.append(nid)
            ids.append(nid)
            if step == 0:
                print(f"  Step 0: {nid} ({elapsed*1000:.0f}ms)")
            if step % 5 == 0 and step > 0:
                tok_s = 5 / (time.time() - t0 + 0.001)
                print(f"  ... {step} tokens ({tok_s:.1f} tok/s)")
        
        return out


# ─── Quick Benchmark ─────────────────────────────────────────────────

def benchmark():
    """Quick end-to-end benchmark."""
    path = sys.argv[1] if len(sys.argv) > 1 else "Llama-3.2-1B-Instruct-Q4_0.gguf"
    
    # Check Mojo worker
    if not os.path.exists(MOJO_WORKER):
        print(f"❌ Mojo worker not compiled at {MOJO_WORKER}")
        print("   Run: mojo build q4_pipe.mojo -o mojollama_q4_pipe")
        sys.exit(1)
    
    print(f"Loading model: {path}")
    t0 = time.time()
    engine = HybridEngine(path)
    print(f"  Loaded: {time.time()-t0:.1f}s")
    print(f"  Architecture: {engine.ws.arch}, {engine.ws.n_layers} layers")
    print(f"  Mojo workers: {engine.pool.n_workers}")
    print()
    
    # Quick forward pass
    ids = [128000, 9906, 1492, 12, 7888, 0]
    print(f"Forward pass ({len(ids)} tokens)...")
    t0 = time.time()
    logits = engine.forward(ids)
    elapsed = time.time() - t0
    print(f"  {elapsed*1000:.0f}ms ({len(ids)/elapsed:.1f} tok/s)")
    print(f"  Logits: {logits.shape}, argmax={np.argmax(logits[-1])}")
    
    # Timing breakdown
    print()
    print("Per-layer timing (layer 0):")
    if 0 in engine._timing:
        t = engine._timing[0]
        print(f"  RMSNorm:   {t['norm']*1000:.1f}ms")
        print(f"  QKV matmul: {t['matmul_qkv']*1000:.1f}ms")
        print(f"  Attn proj: {t['attn_proj']*1000:.1f}ms")
        print(f"  FFN:       {t['ffn']*1000:.1f}ms")
        print(f"  Total:     {t['total']*1000:.1f}ms")
    
    # Generation
    print()
    print(f"Generating {10} tokens...")
    ids = [128000, 9906, 1492, 12, 7888, 0]
    engine._kv_cache = None
    engine._cached_len = 0
    t0 = time.time()
    tokens = engine.generate(ids, max_tokens=10)
    total = time.time() - t0
    print(f"  {len(tokens)} tokens in {total:.2f}s ({len(tokens)/total:.1f} tok/s)")


if __name__ == "__main__":
    benchmark()
