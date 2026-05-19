#!/usr/bin/env python3
"""MojoLlama Turbo Engine v4 — Batched matmuls, F32 output proj, Q6_K support.

Key optimizations over v3:
1. Fused Q+K+V batch matmul (1 OMP region vs 3)
2. Fused Gate+Up batch matmul (1 OMP region vs 2)
3. F32 output projection via C kernel (omp parallel, 2-chain ILP)
4. Q6_K weight type support (dequantize to F32, C matmul)
5. Reduced Python overhead per layer (pre-resolved ctypes pointers)

Target: >80 tok/s on Threadripper 3970X (32c, AVX2+FMA).
"""

import os, sys, time, ctypes, numpy as np

SO_DIR = os.path.dirname(os.path.abspath(__file__))
_so = ctypes.CDLL(os.path.join(SO_DIR, "q4_kernel_omp.so"))

# Q4 matmul
_so.q4_matmul_omp.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_so.q4_matmul_omp.restype = None

# Batch Q4 matmul: fused projections in single OMP region
MAX_PROJ = 7
_so.batch_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_uint8) * MAX_PROJ,  # w[MAX_PROJ]
    ctypes.POINTER(ctypes.c_float),              # x
    ctypes.POINTER(ctypes.c_float) * MAX_PROJ,  # out[MAX_PROJ]
    ctypes.c_int * MAX_PROJ,                     # nrows[MAX_PROJ]
    ctypes.c_int * MAX_PROJ,                     # ncols[MAX_PROJ]
    ctypes.c_int * MAX_PROJ,                     # ts[MAX_PROJ]
    ctypes.c_int,                                 # n_proj
]
_so.batch_matmul.restype = None

# F32 matmul (OMP parallel) — for output projection
_so.f32_matmul_omp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
]
_so.f32_matmul_omp.restype = None

# RMS norm
_so.q4_rms_norm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_so.q4_rms_norm.restype = None

# SiLU (OMP parallel)
_so.q4_silu_omp.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_so.q4_silu_omp.restype = None

# Thread control
_so.q4_get_max_threads.argtypes = []
_so.q4_get_max_threads.restype = ctypes.c_int
_so.q4_set_num_threads.argtypes = [ctypes.c_int]

Q4_TS = {2: 18, 3: 20}  # Q4_0 = 18 bytes/block, Q4_1 = 20 bytes/block


class TurboEngine:
    """Zero-alloc inference engine. C AVX2+FMA+OMP kernels for all hot-path ops.
    v4: Batched matmuls to eliminate OMP synchronization overhead."""

    def __init__(self, path: str, n_threads: int = 32):
        import gguf
        from gguf.constants import GGMLQuantizationType as QT
        self.QT = QT

        _so.q4_set_num_threads(n_threads)
        self.n_threads = n_threads

        r = gguf.GGUFReader(path)
        self._tensors = {t.name: t for t in r.tensors}
        f = r.get_field

        def _s(fi):
            if fi is None: return 'unknown'
            v = fi.parts[-1]
            if isinstance(v, bytes): return v.decode()
            if hasattr(v, 'tobytes'): return v.tobytes().decode().strip('\x00')
            return str(v)

        def _i(fi):
            if fi is None: return 0
            v = fi.parts[-1]
            return int(v.item()) if hasattr(v, 'item') else int(np.asarray(v).flat[0])

        def _f(fi):
            if fi is None: return 1e-6
            v = fi.parts[-1]
            return float(v.item()) if hasattr(v, 'item') else float(np.asarray(v).flat[0])

        arch = _s(f('general.architecture'))
        pfx = f'{arch}.'
        self.n_layers = _i(f(f'{pfx}block_count'))
        self.n_embd = _i(f(f'{pfx}embedding_length'))
        self.n_head = _i(f(f'{pfx}attention.head_count'))
        self.n_kv_head = _i(f(f'{pfx}attention.head_count_kv'))
        self.n_ff = _i(f(f'{pfx}feed_forward_length'))
        self.norm_eps = _f(f(f'{pfx}attention.layer_norm_rms_epsilon'))
        self.head_dim = self.n_embd // self.n_head
        self.n_kv = self.n_kv_head * self.head_dim

        # Pre-load ALL weight tensors
        self._w_cache = {}
        self._load_all_weights()

        # Pre-cache norm weight F32 arrays
        self._norm_weights = {}
        self._norm_ptrs = {}
        for L in range(self.n_layers):
            for suffix in ['attn_norm', 'ffn_norm']:
                w = np.ascontiguousarray(self._dequant_f32(f'blk.{L}.{suffix}.weight'), dtype=np.float32)
                self._norm_weights[f'blk.{L}.{suffix}'] = w
                self._norm_ptrs[f'blk.{L}.{suffix}'] = w.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        onw = np.ascontiguousarray(self._dequant_f32('output_norm.weight'), dtype=np.float32)
        self._norm_weights['output_norm'] = onw
        self._norm_ptrs['output_norm'] = onw.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Pre-allocate work buffers
        N = self.n_embd
        FF = self.n_ff
        self._buf = {
            'h': np.zeros(N, dtype=np.float32),
            'residual': np.zeros(N, dtype=np.float32),
            'q': np.zeros(N, dtype=np.float32),
            'k': np.zeros(self.n_kv, dtype=np.float32),
            'v': np.zeros(self.n_kv, dtype=np.float32),
            'att': np.zeros(N, dtype=np.float32),
            'gate': np.zeros(FF, dtype=np.float32),
            'up': np.zeros(FF, dtype=np.float32),
            'gate_up': np.zeros(FF, dtype=np.float32),
            'ffn_down': np.zeros(N, dtype=np.float32),
        }

        # KV cache
        MAX_POS = 4096
        self.kv_k = np.zeros((self.n_layers, MAX_POS, self.n_kv), dtype=np.float32)
        self.kv_v = np.zeros((self.n_layers, MAX_POS, self.n_kv), dtype=np.float32)
        self.kv_len = np.zeros(self.n_layers, dtype=np.int32)
        self.pos = 0

        # Embedding matrix (F32, for token lookup)
        emb_f32 = self._dequant_f32('token_embd.weight')
        if emb_f32 is not None:
            if emb_f32.shape[0] == self.n_embd:
                emb_f32 = emb_f32.T.copy()
            self.emb = np.ascontiguousarray(emb_f32, dtype=np.float32)
        else:
            raise ValueError("No token_embd.weight found in model")
        self._vocab_size = self.emb.shape[0]

        # Output projection: F32 (Q4_0 re-quantization destroys logits for 128K+ vocab)
        out_w_f32 = self._dequant_f32('output.weight') if 'output.weight' in self._tensors else None
        if out_w_f32 is not None:
            if out_w_f32.shape[0] == self.n_embd:
                out_w_f32 = out_w_f32.T.copy()
        else:
            out_w_f32 = self.emb.copy()

        self._out_w_f32 = np.ascontiguousarray(out_w_f32, dtype=np.float32)
        self._out_w_ptr = self._out_w_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Pre-compute RoPE freqs
        freqs = 1.0 / (500000.0 ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim))
        self.rope_cos = np.cos(np.arange(MAX_POS)[:, None] * freqs[None, :]).astype(np.float32)
        self.rope_sin = np.sin(np.arange(MAX_POS)[:, None] * freqs[None, :]).astype(np.float32)

        # Pre-resolve Python ctypes objects for hot-path buffers
        self._ptrs = {k: v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                      for k, v in self._buf.items()}

        print(f"TurboEngine v4: {self.n_layers}L/{self.n_embd}D/{self.n_ff}FF/"
              f"{self.n_head}H/{self.n_kv_head}KV | t={self.n_threads} | "
              f"vocab={self._vocab_size}")

    def _dequant_f32(self, name: str) -> np.ndarray:
        import gguf
        t = self._tensors.get(name)
        if t is None: return None
        arr = np.ascontiguousarray(gguf.dequantize(t.data, t.tensor_type).astype(np.float32))
        return arr

    def _load_all_weights(self):
        preload_projs = ['attn_q', 'attn_k', 'attn_v', 'attn_output',
                          'ffn_gate', 'ffn_up', 'ffn_down']
        for L in range(self.n_layers):
            for suffix in preload_projs:
                name = f'blk.{L}.{suffix}.weight'
                t = self._tensors.get(name)
                if t is None: continue
                tt = int(t.tensor_type)
                nc = self.n_ff if 'down' in suffix else self.n_embd
                if tt in (self.QT.Q4_0, self.QT.Q4_1):
                    raw = np.ascontiguousarray(np.asarray(t.data).reshape(-1), dtype=np.uint8)
                    ts = Q4_TS.get(tt, 18)
                    bpr = nc // 32
                    nrows = len(raw) // (bpr * ts)
                    ptr = raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                    self._w_cache[name] = (ptr, nrows, nc, ts)
                elif tt == self.QT.Q6_K:
                    # Q6_K: dequantize to F32, use C f32_matmul_omp
                    f32 = self._dequant_f32(name)
                    if f32.shape[0] == self.n_embd and 'down' not in suffix:
                        # Transpose: GGUF stores (in_dim, out_dim), we need (out_dim, in_dim)
                        f32 = f32.T.copy()
                    elif f32.shape[1] == self.n_embd and 'down' in suffix:
                        f32 = f32.T.copy()
                    f32 = np.ascontiguousarray(f32, dtype=np.float32)
                    ptr = f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    self._w_cache[name] = ('f32', ptr, f32.shape[0], f32.shape[1])
                elif tt == self.QT.F32:
                    raw_f = np.ascontiguousarray(gguf.dequantize(t.data, t.tensor_type).astype(np.float32))
                    if raw_f.shape[0] == self.n_embd:
                        raw_f = raw_f.T.copy()
                    raw_f = np.ascontiguousarray(raw_f, dtype=np.float32)
                    ptr = raw_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    self._w_cache[name] = ('f32', ptr, raw_f.shape[0], raw_f.shape[1])
                else:
                    # Fallback: dequantize to F32
                    f32 = self._dequant_f32(name)
                    if f32.shape[0] == self.n_embd:
                        f32 = f32.T.copy()
                    f32 = np.ascontiguousarray(f32, dtype=np.float32)
                    ptr = f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    self._w_cache[name] = ('f32', ptr, f32.shape[0], f32.shape[1])

    def _matmul(self, w_cache, x_ptr, out_ptr):
        """Dispatch matmul to Q4 or F32 kernel based on weight type."""
        if w_cache[0] == 'f32':
            _, w_ptr, nrows, ncols = w_cache
            _so.f32_matmul_omp(w_ptr, x_ptr, out_ptr, nrows, ncols)
        else:
            w_ptr, nrows, ncols, ts = w_cache
            _so.q4_matmul_omp(w_ptr, x_ptr, out_ptr, nrows, ncols, ts)

    def _batch_matmul3(self, wc1, wc2, wc3, x_ptr, out_ptrs):
        """Batch 3 projections (e.g., Q+K+V) in single OMP region if all Q4."""
        if all(w[0] != 'f32' for w in [wc1, wc2, wc3]):
            # All Q4_0/Q4_1 — use batch_matmul with shared input
            w_arr = (ctypes.POINTER(ctypes.c_uint8) * MAX_PROJ)(wc1[0], wc2[0], wc3[0],
                     *[ctypes.POINTER(ctypes.c_uint8)()] * (MAX_PROJ - 3))
            out_arr = (ctypes.POINTER(ctypes.c_float) * MAX_PROJ)(out_ptrs[0], out_ptrs[1], out_ptrs[2],
                       *[ctypes.POINTER(ctypes.c_float)()] * (MAX_PROJ - 3))
            nr_arr = (ctypes.c_int * MAX_PROJ)(wc1[1], wc2[1], wc3[1], *([0] * (MAX_PROJ - 3)))
            nc_arr = (ctypes.c_int * MAX_PROJ)(wc1[2], wc2[2], wc3[2], *([0] * (MAX_PROJ - 3)))
            ts_arr = (ctypes.c_int * MAX_PROJ)(wc1[3], wc2[3], wc3[3], *([18] * (MAX_PROJ - 3)))
            _so.batch_matmul(w_arr, x_ptr, out_arr, nr_arr, nc_arr, ts_arr, 3)
        elif all(w[0] == 'f32' for w in [wc1, wc2, wc3]):
            # All F32 — separate calls (batch F32 not yet implemented)
            for wc, op in zip([wc1, wc2, wc3], out_ptrs):
                self._matmul(wc, x_ptr, op)
        else:
            # Mixed — fall back to separate calls
            self._matmul(wc1, x_ptr, out_ptrs[0])
            self._matmul(wc2, x_ptr, out_ptrs[1])
            self._matmul(wc3, x_ptr, out_ptrs[2])

    def forward(self, token_id: int) -> np.ndarray:
        """Single-token decode. Returns logits."""
        b = self._buf
        p = self._ptrs  # pre-resolved ctypes pointers
        N = self.n_embd
        FF = self.n_ff
        hd = self.head_dim
        nh = self.n_head
        nkh = self.n_kv_head
        pos = self.pos

        # Embedding lookup
        np.copyto(b['h'], self.emb[token_id])

        for L in range(self.n_layers):
            # Save residual
            np.copyto(b['residual'], b['h'])

            # RMS norm (attn)
            _so.q4_rms_norm(p['h'], self._norm_ptrs[f'blk.{L}.attn_norm'], p['h'], N)

            # QKV projections — batched in single OMP region
            wq = self._w_cache[f'blk.{L}.attn_q.weight']
            wk = self._w_cache[f'blk.{L}.attn_k.weight']
            wv = self._w_cache[f'blk.{L}.attn_v.weight']
            self._batch_matmul3(wq, wk, wv, p['h'], [p['q'], p['k'], p['v']])

            # RoPE
            cos_p = self.rope_cos[pos]
            sin_p = self.rope_sin[pos]
            # Q RoPE
            q2 = b['q'].reshape(nh, hd)
            qxr = q2.reshape(nh, hd // 2, 2)
            qxrot = np.stack([-qxr[..., 1], qxr[..., 0]], axis=-1)
            b['q'][:] = (qxr * cos_p.reshape(1, hd // 2, 1) +
                         qxrot * sin_p.reshape(1, hd // 2, 1)).reshape(-1)
            # K RoPE
            k2 = b['k'].reshape(nkh, hd)
            kxr = k2.reshape(nkh, hd // 2, 2)
            kxrot = np.stack([-kxr[..., 1], kxr[..., 0]], axis=-1)
            b['k'][:] = (kxr * cos_p.reshape(1, hd // 2, 1) +
                         kxrot * sin_p.reshape(1, hd // 2, 1)).reshape(-1)

            # Attention (GQA)
            n_rep = nh // nkh
            self.kv_k[L, self.kv_len[L], :] = b['k'][:self.n_kv]
            self.kv_v[L, self.kv_len[L], :] = b['v'][:self.n_kv]
            self.kv_len[L] += 1

            k_cache = self.kv_k[L, :self.kv_len[L]].reshape(self.kv_len[L], nkh, hd)
            v_cache = self.kv_v[L, :self.kv_len[L]].reshape(self.kv_len[L], nkh, hd)

            if n_rep > 1:
                k_cache_exp = np.repeat(k_cache, n_rep, axis=1).reshape(self.kv_len[L], nh, hd)
                v_cache_exp = np.repeat(v_cache, n_rep, axis=1).reshape(self.kv_len[L], nh, hd)
            else:
                k_cache_exp = k_cache
                v_cache_exp = v_cache

            q_2d = b['q'].reshape(nh, hd)
            scores = np.einsum('hd,shd->hs', q_2d, k_cache_exp) / np.sqrt(hd)
            scores -= np.max(scores, axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= np.sum(scores, axis=1, keepdims=True)
            att = np.einsum('hs,shd->hd', scores, v_cache_exp).astype(np.float32).reshape(-1)

            # Output projection
            np.copyto(b['att'], att)
            wo = self._w_cache[f'blk.{L}.attn_output.weight']
            self._matmul(wo, p['att'], p['h'])

            b['h'][:] = b['residual'] + b['h']

            # FFN
            np.copyto(b['residual'], b['h'])
            _so.q4_rms_norm(p['h'], self._norm_ptrs[f'blk.{L}.ffn_norm'], p['h'], N)

            # Gate + Up — batched in single OMP region
            wg = self._w_cache[f'blk.{L}.ffn_gate.weight']
            wu = self._w_cache[f'blk.{L}.ffn_up.weight']
            if wg[0] != 'f32' and wu[0] != 'f32':
                # Both Q4 — batch them
                w_arr = (ctypes.POINTER(ctypes.c_uint8) * MAX_PROJ)(
                    wg[0], wu[0], *[ctypes.POINTER(ctypes.c_uint8)()] * (MAX_PROJ - 2))
                out_arr = (ctypes.POINTER(ctypes.c_float) * MAX_PROJ)(
                    p['gate'], p['up'], *[ctypes.POINTER(ctypes.c_float)()] * (MAX_PROJ - 2))
                nr_arr = (ctypes.c_int * MAX_PROJ)(wg[1], wu[1], *([0] * (MAX_PROJ - 2)))
                nc_arr = (ctypes.c_int * MAX_PROJ)(wg[2], wu[2], *([0] * (MAX_PROJ - 2)))
                ts_arr = (ctypes.c_int * MAX_PROJ)(wg[3], wu[3], *([18] * (MAX_PROJ - 2)))
                _so.batch_matmul(w_arr, p['h'], out_arr, nr_arr, nc_arr, ts_arr, 2)
            else:
                self._matmul(wg, p['h'], p['gate'])
                self._matmul(wu, p['h'], p['up'])

            # SiLU(gate) * up
            _so.q4_silu_omp(p['gate'], FF)
            np.multiply(b['gate'], b['up'], out=b['gate_up'])

            wd = self._w_cache[f'blk.{L}.ffn_down.weight']
            self._matmul(wd, p['gate_up'], p['ffn_down'])

            b['h'][:] = b['residual'] + b['ffn_down']

        # Final norm
        _so.q4_rms_norm(p['h'], self._norm_ptrs['output_norm'], p['h'], N)

        # Output projection: F32 via C kernel
        logits = np.zeros(self._vocab_size, dtype=np.float32)
        _so.f32_matmul_omp(self._out_w_ptr, p['h'],
                           logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           self._vocab_size, self.n_embd)

        self.pos += 1
        return logits

    def prefill(self, token_ids: list) -> np.ndarray:
        self.reset()
        for i, tid in enumerate(token_ids[:-1]):
            self.forward(tid)
        return self.forward(token_ids[-1])

    def reset(self):
        self.kv_len[:] = 0
        self.pos = 0


def benchmark(path: str = 'Llama-3.2-1B-Instruct-Q4_0.gguf', n_warmup: int = 5, n_decode: int = 50):
    engine = TurboEngine(path)

    # Warmup
    engine.reset()
    engine.forward(128000)
    for _ in range(n_warmup):
        engine.forward(9906)

    # Decode benchmark
    times = []
    next_tok = 9906
    for i in range(n_decode):
        t0 = time.perf_counter()
        logits = engine.forward(next_tok)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        next_tok = int(np.argmax(logits))

    times.sort()
    median_ms = times[n_decode // 2]
    tok_per_sec = 1000.0 / median_ms

    print(f"\n{'='*60}")
    print(f"MojoLlama TurboEngine v4 — Decode Benchmark")
    print(f"  Model: {engine.n_layers}L/{engine.n_embd}D/{engine.n_ff}FF")
    print(f"  Threads: {engine.n_threads}")
    print(f"  Decode (median): {median_ms:.2f} ms/tok = {tok_per_sec:.1f} tok/s")
    print(f"  P50: {times[n_decode//2]:.2f} ms, P95: {times[int(n_decode*0.95)]:.2f} ms")
    print(f"  Min: {min(times):.2f} ms, Max: {max(times):.2f} ms")
    print(f"{'='*60}")
    return tok_per_sec


if __name__ == '__main__':
    benchmark(sys.argv[1] if len(sys.argv) > 1 else 'Llama-3.2-1B-Instruct-Q4_0.gguf')