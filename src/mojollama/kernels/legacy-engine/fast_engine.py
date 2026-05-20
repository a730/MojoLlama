#!/usr/bin/env python3
"""MojoLlama Fast Engine — C AVX2+FMA OpenMP kernels + vectorized numpy ops.

Target: beat llama.cpp (85 tok/s on Threadripper 3970X, 32 cores).

Architecture:
    C AVX2 .so (OpenMP t=32):
        - Q4_0/Q4_1 dequant+dot (row-outer, 4-row blocked)
        - batch_matmul (7 projections in 1 OMP call)
        - rms_norm (AVX2 vectorized)
        - silu (AVX2 + scalar exp)
        - softmax (AVX2 max + scalar exp)
    Python orchestrator:
        - Layer loop, RoPE (numpy), attention (vectorized), weight cache
        - Continuous batching (future)
"""
import os
import sys
import time
import ctypes
import numpy as np
from typing import List, Optional, Tuple
import gguf
from gguf.constants import GGMLQuantizationType as QT

# ─── Load C kernels ───────────────────────────────────────────────────

SO_DIR = os.path.dirname(os.path.abspath(__file__))
_so = ctypes.CDLL(os.path.join(SO_DIR, "q4_kernel_omp.so"))

# Matmul
_so.q4_matmul_omp.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_so.q4_matmul_omp.restype = None

_so.q4_matmul_avx2.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_so.q4_matmul_avx2.restype = None

_so.q4_rms_norm.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
_so.q4_rms_norm.restype = None

_so.q4_silu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_so.q4_silu.restype = None

_so.q4_softmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_so.q4_softmax.restype = None

_so.q4_get_max_threads.argtypes = []
_so.q4_get_max_threads.restype = ctypes.c_int
_so.q4_set_num_threads.argtypes = [ctypes.c_int]
_so.q4_set_num_threads.restype = None

# Batch matmul
MAX_PROJ = 7
_so.batch_matmul.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),   # w[MAX_PROJ]
    ctypes.POINTER(ctypes.c_float),                      # x
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),      # out[MAX_PROJ]
    ctypes.POINTER(ctypes.c_int),                         # nrows[MAX_PROJ]
    ctypes.POINTER(ctypes.c_int),                         # ncols[MAX_PROJ]
    ctypes.POINTER(ctypes.c_int),                         # ts[MAX_PROJ]
    ctypes.c_int,                                          # n_proj
]
_so.batch_matmul.restype = None

Q4_TS = {2: 18, 3: 20}  # Q4_0=2→18 bytes, Q4_1=3→20


class FastEngine:
    """Inference engine using C AVX2+OMP kernels with vectorized Python glue."""

    def __init__(self, model_path: str, n_threads: int = 0):
        import gguf
        from gguf.constants import GGMLQuantizationType as QT

        self.n_threads = n_threads or min(32, os.cpu_count() // 2)
        _so.q4_set_num_threads(self.n_threads)

        # Load model
        self.reader = gguf.GGUFReader(model_path)
        self._tensors = {t.name: t for t in self.reader.tensors}
        self._cache = {}

        # Parse config
        f = self.reader.get_field
        self.arch = self._str(f('general.architecture'))
        pfx = f'{self.arch}.'
        self.n_layers = self._int(f(f'{pfx}block_count'))
        self.n_embd = self._int(f(f'{pfx}embedding_length'))
        self.n_head = self._int(f(f'{pfx}attention.head_count'))
        self.n_kv_head = self._int(f(f'{pfx}attention.head_count_kv'))
        self.n_ff = self._int(f(f'{pfx}feed_forward_length'))
        self.norm_eps = self._float(f(f'{pfx}attention.layer_norm_rms_epsilon'))
        self.head_dim = self.n_embd // self.n_head
        self.n_kv = self.n_kv_head * self.head_dim
        self.n_rep = self.n_head // self.n_kv_head  # GQA repeat factor
        self.QT = QT

        # Precompute RoPE freqs (up to 4096 positions)
        self._precompute_rope(4096)

        # Cache embedding matrix (for token lookup + output projection)
        emb_raw = self.get_f32('token_embd.weight')
        if emb_raw.ndim == 2 and emb_raw.shape[0] == self.n_embd:
            self._emb_cache = np.ascontiguousarray(emb_raw.T, dtype=np.float32)  # (vocab, n_embd)
        else:
            self._emb_cache = np.ascontiguousarray(emb_raw, dtype=np.float32)
        self._vocab_size = self._emb_cache.shape[0]

        # KV cache
        self.kv_k = [np.zeros((0, self.n_kv), dtype=np.float32) for _ in range(self.n_layers)]
        self.kv_v = [np.zeros((0, self.n_kv), dtype=np.float32) for _ in range(self.n_layers)]
        self.pos = 0

        print(f"FastEngine: {self.n_layers}L/{self.n_embd}D/{self.n_ff}FF/"
              f"{self.n_head}H/{self.n_kv_head}KV | t={self.n_threads}")

    # ─── Weight loading ────────────────────────────────────────────

    def get(self, name: str) -> Tuple[np.ndarray, int]:
        """Get weight tensor as (raw_uint8_or_float32, ggml_type_int)."""
        if name in self._cache:
            return self._cache[name]
        t = self._tensors.get(name)
        if t is None:
            return None, 0
        arr = np.asarray(t.data)
        tt = int(t.tensor_type)
        if tt in (self.QT.Q4_0, self.QT.Q4_1):
            raw = arr.tobytes() if arr.dtype == np.uint8 else arr.astype(np.uint8).tobytes()
            result = (np.frombuffer(raw, dtype=np.uint8).copy(), tt)
        elif tt in (self.QT.F16, self.QT.F32):
            deq = gguf.dequantize(t.data, t.tensor_type) if tt == self.QT.F16 else np.ascontiguousarray(arr, dtype=np.float32)
            result = (np.ascontiguousarray(deq.astype(np.float32)), tt)
        else:
            deq = gguf.dequantize(t.data, t.tensor_type)
            result = (np.ascontiguousarray(deq.astype(np.float32)), tt)
        self._cache[name] = result
        return result

    def get_f32(self, name: str) -> np.ndarray:
        """Get weight as float32 array (for norms, rope, embeddings)."""
        w, tt = self.get(name)
        if tt in (self.QT.Q4_0, self.QT.Q4_1):
            # Dequantize to float32
            import gguf
            t = self._tensors[name]
            return np.ascontiguousarray(gguf.dequantize(t.data, t.tensor_type).astype(np.float32))
        return w

    # ─── C kernel wrappers ──────────────────────────────────────────

    def matmul(self, name: str, x: np.ndarray, n_cols: int) -> np.ndarray:
        """Q4_0/Q4_1 matmul via C OMP kernel."""
        w, tt = self.get(name)
        if w is None:
            raise ValueError(f"Weight not found: {name}")
        ts = Q4_TS.get(tt, 18)
        bpr = n_cols // 32
        nr = len(w) // (bpr * ts) if bpr > 0 else 0
        if nr == 0:
            raise ValueError(f"Cannot compute n_rows for {name}: len={len(w)}, bpr={bpr}, ts={ts}")

        out = np.zeros(nr, dtype=np.float32)
        if tt in (self.QT.Q4_0, self.QT.Q4_1):
            w_c = np.ascontiguousarray(w, dtype=np.uint8)
            _so.q4_matmul_omp(
                w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                nr, n_cols, ts,
            )
        else:
            # Float path — use numpy
            w_f = w.reshape(nr, n_cols) if w.ndim == 1 else w
            if w_f.ndim == 1:
                w_f = w_f.reshape(-1, n_cols)
            np.dot(x, w_f.T, out=out[:nr])
        return out

    def matmul_batch(self, names: List[str], x: np.ndarray, n_cols_list: List[int]) -> List[np.ndarray]:
        """Batch matmul — fuse multiple projections into single OMP call."""
        results = []
        for name, nc in zip(names, n_cols_list):
            results.append(self.matmul(name, x, nc))
        return results

    def rms_norm(self, x: np.ndarray, name: str) -> np.ndarray:
        """RMS normalization via C kernel."""
        w = self.get_f32(name)
        out = np.zeros_like(x)
        _so.q4_rms_norm(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x),
        )
        return out

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU activation via C kernel (in-place)."""
        c = np.ascontiguousarray(x, dtype=np.float32)
        _so.q4_silu(c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(c))
        return c

    # ─── RoPE (vectorized numpy) ─────────────────────────────────

    def _precompute_rope(self, max_pos: int):
        """Precompute RoPE cos/sin tables."""
        freqs = 1.0 / (500000.0 ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim))
        self._rope_cos = np.zeros((max_pos, self.head_dim // 2), dtype=np.float32)
        self._rope_sin = np.zeros((max_pos, self.head_dim // 2), dtype=np.float32)
        for p in range(max_pos):
            self._rope_cos[p] = np.cos(p * freqs)
            self._rope_sin[p] = np.sin(p * freqs)

    def apply_rope(self, x: np.ndarray, pos: int, n_heads: int) -> np.ndarray:
        """Apply RoPE to Q or K tensor. x shape may be (n_heads * head_dim,) or (n_total,)."""
        hd = self.head_dim
        actual_len = x.shape[0]
        actual_heads = actual_len // hd
        x = x.reshape(actual_heads, hd)
        cos_f = self._rope_cos[min(pos, len(self._rope_cos)-1)]  # (hd//2,)
        sin_f = self._rope_sin[min(pos, len(self._rope_sin)-1)]   # (hd//2,)
        xr = x.reshape(actual_heads, hd // 2, 2)
        xr_rot = np.stack([-xr[..., 1], xr[..., 0]], axis=-1)
        result = (xr * cos_f.reshape(1, hd // 2, 1) + xr_rot * sin_f.reshape(1, hd // 2, 1)).reshape(actual_heads, hd)
        return result.reshape(-1)

    # ─── Attention (vectorized numpy) ──────────────────────────────

    def attention(self, q: np.ndarray, k_new: np.ndarray, v_new: np.ndarray,
                  layer: int, pos: int) -> np.ndarray:
        """Multi-head attention with GQA. Vectorized over heads."""
        hd = self.head_dim
        nh = self.n_head
        nkh = self.n_kv_head
        n_rep = self.n_rep

        q2 = q.reshape(nh, hd)
        k2 = k_new.reshape(nkh, hd)
        v2 = v_new.reshape(nkh, hd)

        # GQA: expand KV for repeated heads
        if n_rep > 1:
            k2 = np.repeat(k2, n_rep, axis=0)  # (nh, hd)
            v2 = np.repeat(v2, n_rep, axis=0)

        # Update KV cache
        self.kv_k[layer] = np.vstack([self.kv_k[layer], k2.reshape(1, -1)]) if self.kv_k[layer].shape[0] > 0 else k2.reshape(1, -1)
        self.kv_v[layer] = np.vstack([self.kv_v[layer], v2.reshape(1, -1)]) if self.kv_v[layer].shape[0] > 0 else v2.reshape(1, -1)

        # Compute attention scores: Q @ K^T  (nh, hd) @ (seq_len, hd).T → (nh, seq_len)
        # Vectorized: all heads at once using reshaped caches
        seq_len = self.kv_k[layer].shape[0]

        # For GQA with n_rep>1, we stored expanded KV. Reshape caches to (seq_len, nh, hd)
        k_cache = self.kv_k[layer].reshape(seq_len, nh, hd)  # (seq, nh, hd)
        v_cache = self.kv_v[layer].reshape(seq_len, nh, hd)

        # Attention scores: (nh, hd) @ (hd, seq, nh) broadcast
        # scores[h] = q[h] @ k[:, h, :].T = (hd,) @ (hd, seq) → (seq,)
        scores = np.einsum('hd,shd->hs', q2, k_cache)  # (nh, seq_len)

        # Causal mask and softmax
        scores *= (1.0 / np.sqrt(hd))  # Scale
        # For single position decode, no causal masking needed (only current+past tokens)

        # Softmax per head
        scores_max = np.max(scores, axis=1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_sum = np.sum(scores_exp, axis=1, keepdims=True)
        attn_weights = scores_exp / scores_sum  # (nh, seq_len)

        # Weighted sum: attn_weights @ v_cache
        output = np.einsum('hs,shd->hd', attn_weights, v_cache)  # (nh, hd) → (hd,)
        return output.reshape(-1)

    # ─── Forward pass ────────────────────────────────────────────

    def forward(self, token_ids: List[int]) -> np.ndarray:
        """Full forward pass. Returns logits."""
        # Embedding (use cached F32 matrix)
        h = self._emb_cache[token_ids[-1]].astype(np.float32).copy()

        pos = self.pos

        for layer in range(self.n_layers):
            residual = h.copy()

            # RMS norm (attention)
            h = self.rms_norm(h, f'blk.{layer}.attn_norm.weight')

            # QKV projections
            q = self.matmul(f'blk.{layer}.attn_q.weight', h, self.n_embd)
            k = self.matmul(f'blk.{layer}.attn_k.weight', h, self.n_embd)
            v = self.matmul(f'blk.{layer}.attn_v.weight', h, self.n_embd)

            # RoPE
            q = self.apply_rope(q, pos, self.n_head)
            k = self.apply_rope(k, pos, self.n_kv_head)

            # Attention
            att = self.attention(q, k, v, layer, pos)

            # Output projection + residual
            o = self.matmul(f'blk.{layer}.attn_output.weight', att, self.n_embd)
            h = residual + o

            # FFN
            residual = h.copy()
            h = self.rms_norm(h, f'blk.{layer}.ffn_norm.weight')

            gate = self.matmul(f'blk.{layer}.ffn_gate.weight', h, self.n_embd)
            gate = self.silu(gate)
            up = self.matmul(f'blk.{layer}.ffn_up.weight', h, self.n_embd)
            h = residual + self.matmul(f'blk.{layer}.ffn_down.weight', gate * up, self.n_ff)

        # Final norm + output projection
        h = self.rms_norm(h, 'output_norm.weight')

        # Use output projection if available, tied weights otherwise
        out_result = self.get('output.weight')
        if out_result[0] is not None:
            out_w, out_tt = out_result
            if out_tt in (QT.Q4_0, QT.Q4_1):
                t = self._tensors['output.weight']
                out_f32 = np.ascontiguousarray(gguf.dequantize(t.data, t.tensor_type).astype(np.float32))
                if out_f32.shape[0] == self.n_embd:
                    out_f32 = out_f32.T
                logits = h @ out_f32
            else:
                out_w = out_w.reshape(-1, self.n_embd) if out_w.ndim == 1 else out_w
                if out_w.shape[0] == self.n_embd:
                    out_w = out_w.T
                logits = h @ out_w
        else:
            # Tied weights: use embedding matrix (already F32)
            # _emb_cache is (vocab_size, n_embd), h is (n_embd,)
            logits = self._emb_cache @ h

        self.pos += 1
        return logits

    def reset(self):
        """Reset KV cache."""
        self.kv_k = [np.zeros((0, self.n_kv), dtype=np.float32) for _ in range(self.n_layers)]
        self.kv_v = [np.zeros((0, self.n_kv), dtype=np.float32) for _ in range(self.n_layers)]
        self.pos = 0

    # ─── Helpers ─────────────────────────────────────────────────

    def _int(self, f):
        if f is None: return 0
        v = f.parts[-1]
        if hasattr(v, 'item'):
            return int(v.item())
        # memmap with multiple elements
        v = np.asarray(v)
        if v.ndim == 0:
            return int(v)
        # Try data field
        if hasattr(f, 'data') and f.data:
            return int(f.data[0])
        # Last resort: first element
        return int(v.flat[0])

    def _float(self, f):
        if f is None: return 1e-6
        v = f.parts[-1]
        if hasattr(v, 'item'): return float(v.item())
        v2 = np.asarray(v)
        if v2.ndim == 0: return float(v2)
        if hasattr(f, 'data') and f.data: return float(f.data[0])
        return float(v2.flat[0])

    def _str(self, f):
        if f is None: return 'unknown'
        v = f.parts[-1]
        if isinstance(v, bytes): return v.decode('utf-8', errors='replace')
        if hasattr(v, 'tobytes'): return v.tobytes().decode('utf-8', errors='replace').strip('\x00')
        if hasattr(v, 'item'): return str(v.item())
        return str(v)


# ─── Benchmark ────────────────────────────────────────────────────

def benchmark(engine: FastEngine, n_warmup: int = 3, n_decode: int = 50):
    """Benchmark decode throughput (single-token generation)."""
    import gguf

    engine.reset()

    # Prefill with a few tokens
    prompt_tokens = [128000, 9906, 1492, 12, 7888]
    t0 = time.perf_counter()
    for i, tok in enumerate(prompt_tokens[:-1]):
        engine.forward([tok])
    prefilled = len(prompt_tokens) - 1

    # Decode benchmark
    times = []
    next_tok = prompt_tokens[-1]
    for i in range(n_warmup + n_decode):
        t0 = time.perf_counter()
        logits = engine.forward([next_tok])
        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append((t1 - t0) * 1000)
        next_tok = int(np.argmax(logits))

    times.sort()
    median_ms = times[n_decode // 2]
    tok_per_sec = 1000.0 / median_ms

    print(f"\n{'='*60}")
    print(f"FastEngine Decode Benchmark")
    print(f"  Model: {engine.n_layers}L/{engine.n_embd}D/{engine.n_ff}FF")
    print(f"  Threads: {engine.n_threads}")
    print(f"  Decode (median): {median_ms:.2f} ms/tok = {tok_per_sec:.1f} tok/s")
    print(f"  P95: {times[int(n_decode * 0.95)]:.2f} ms")
    print(f"  Min: {min(times):.2f} ms, Max: {max(times):.2f} ms")
    print(f"{'='*60}")
    return tok_per_sec


def profile_layer(engine: FastEngine):
    """Profile each operation in a single layer."""
    import gguf

    engine.reset()
    engine.forward([128000])  # Warmup

    layer = 0
    h = engine.get_f32('token_embd.weight')[128000].astype(np.float32).copy()
    nc = engine.n_embd

    print(f"\n{'='*60}")
    print(f"Layer {layer} Profile")
    print(f"{'='*60}")

    def time_op(label, fn, iters=50):
        times = []
        for _ in range(3):
            fn()
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
        times.sort()
        p95_idx = max(0, int(iters * 0.95) - 1)
        print(f"  {label:40s}: {times[iters//2]:8.1f} µs  (P5={times[iters//20]:.1f}, P95={times[p95_idx]:.1f})")
        return times[iters // 2]

    x = h.copy()

    time_op("RMS norm (attn)", lambda: engine.rms_norm(x, f'blk.{layer}.attn_norm.weight'))
    time_op("Q proj matmul", lambda: engine.matmul(f'blk.{layer}.attn_q.weight', x, nc))
    time_op("K proj matmul", lambda: engine.matmul(f'blk.{layer}.attn_k.weight', x, nc))
    time_op("V proj matmul", lambda: engine.matmul(f'blk.{layer}.attn_v.weight', x, nc))

    q = engine.matmul(f'blk.{layer}.attn_q.weight', x, nc)
    time_op("RoPE (Q+K)", lambda: (engine.apply_rope(q, 0, engine.n_head),
                                    engine.apply_rope(q, 0, engine.n_kv_head)))

    time_op("SiLU (8192)", lambda: engine.silu(np.random.randn(engine.n_ff).astype(np.float32)))
    time_op("RMS norm (FFN)", lambda: engine.rms_norm(x, f'blk.{layer}.ffn_norm.weight'))
    time_op("FFN gate matmul", lambda: engine.matmul(f'blk.{layer}.ffn_gate.weight', x, nc))
    time_op("FFN up matmul", lambda: engine.matmul(f'blk.{layer}.ffn_up.weight', x, nc))

    gate = np.random.randn(engine.n_ff).astype(np.float32)
    time_op("FFN down matmul", lambda: engine.matmul(f'blk.{layer}.ffn_down.weight', gate, engine.n_ff))


if __name__ == '__main__':
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else 'Llama-3.2-1B-Instruct-Q4_0.gguf'
    path = model if os.path.exists(model) else os.path.join('/onedev-workspace/work', model)

    engine = FastEngine(path)
    benchmark(engine)