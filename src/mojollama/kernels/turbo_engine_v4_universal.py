#!/usr/bin/env python3
"""Universal TurboEngine v4 — Supports dense (Llama) and MoE (Qwen3) models.
Quant types: Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32.
Architecture: dense (Llama-style) and MoE with expert routing (Qwen3-style).

KEY INSIGHT: GGUF quantized data is stored OUTPUT-NEURON-MAJOR, not input-major.
The declared shape (in_dim, out_dim) is logical; the physical data layout packs
all input weights for each output neuron contiguously (grouped by output neuron).
For all quant types, blocks per row = n_cols / block_size, nrows = out_dim.
Matmul computes: W_T @ x = x @ W, where W_T has shape (out_dim, in_dim).

C kernels: quant_kernels_omp.so provides Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0/F32.
All use the same calling convention: (W_raw, x, out, n_rows, n_cols) or with quant_type.
"""

import numpy as np
import ctypes
import os
import time
import gguf
from gguf.constants import GGMLQuantizationType as QT

# GGML quant type constants (matching ggml/GGUF convention)
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14

QTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    12: "Q4_K", 8: "Q8_0", 13: "Q5_K", 14: "Q6_K",
}

# Block sizes in bytes per block (super-block for K-types)
# Q4_0/Q4_1/Q8_0: block_size=32 values per block
# Q4_K/Q5_K/Q6_K: block_size=256 values per super-block
BLOCK_SIZES = {2: 18, 3: 20, 12: 144, 13: 176, 14: 210, 8: 34}

# Values per block (for computing n_blocks from n_cols)
BLOCK_VALS = {2: 32, 3: 32, 12: 256, 13: 256, 14: 256, 8: 32}

# Quant types that can use the C kernel directly
C_KERNEL_TYPES = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K,
                 GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}


class TurboEngine:
    """Universal inference engine for dense and MoE models."""

    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        self.n_threads = n_threads

        # Load GGUF model
        self.reader = gguf.GGUFReader(model_path)
        self._parse_metadata()
        self._load_weights()
        self._load_kernels()
        self._init_kv_cache()
        self.reset()

        moe_str = f'/MoE-{self.n_experts}x{self.n_experts_per_tok}' if self.is_moe else ''
        print(f"TurboEngine v4: {self.n_layers}L/{self.n_embd}D/"
              f"{self.n_ff}FF/{self.n_head}H/{self.n_kv_head}KV"
              f"{moe_str} | t={self.n_threads} | vocab={self.vocab_size}")

    def _parse_metadata(self):
        """Extract architecture info from GGUF metadata."""
        fields = self.reader.fields

        def _get(name):
            for key, val in fields.items():
                if key == name:
                    parts = val.parts if hasattr(val, 'parts') else []
                    if len(parts) >= 1:
                        data = parts[-1]
                        if hasattr(data, '__iter__') and len(data) == 1:
                            return int(data[0])
                        return data

        # Detect architecture prefix
        arch = 'llama'
        for prefix in ['qwen3moe', 'qwen2moe', 'llama', 'mistral']:
            for key in fields:
                if f'{prefix}.block_count' in key:
                    arch = prefix
                    break

        self.n_layers = int(_get(f'{arch}.block_count') or 16)
        self.n_embd = int(_get(f'{arch}.embedding_length') or 2048)
        self.n_ff = int(_get(f'{arch}.feed_forward_length') or self.n_embd * 4)
        self.n_head = int(_get(f'{arch}.attention.head_count') or 32)
        self.n_kv_head = int(_get(f'{arch}.attention.head_count_kv') or self.n_head)
        self.head_dim = int(_get(f'{arch}.attention.key_length') or (self.n_embd // self.n_head))
        self.rope_freq_base = float(_get(f'{arch}.rope.freq_base') or 10000.0)
        self.eps = float(_get(f'{arch}.attention.layer_norm_rms_epsilon') or 1e-6)
        # Derive vocab_size from output/token_embd weight size if not in metadata
        vocab_size_meta = _get(f'{arch}.vocab_size')
        if vocab_size_meta is not None:
            self.vocab_size = int(vocab_size_meta)
        else:
            # Will be set in _load_weights after we see the embedding shape
            self.vocab_size = 0

        # MoE parameters
        self.n_experts = _get(f'{arch}.expert_count')
        self.n_experts_per_tok = _get(f'{arch}.expert_used_count')
        self.n_ff_expert = int(_get(f'{arch}.expert_feed_forward_length') or self.n_ff)
        self.is_moe = self.n_experts is not None

        if not self.is_moe:
            self.n_experts = 1
            self.n_experts_per_tok = 1
        else:
            self.n_experts = int(self.n_experts)
            self.n_experts_per_tok = int(self.n_experts_per_tok)

        self.arch_prefix = arch

    def _load_weights(self):
        """Load weights from GGUF. Store F32 as (out_dim, in_dim) for x @ W convention.
        
        GGUF quantized data is stored output-neuron-major. gguf.dequantize returns
        (in_dim, out_dim) shape, but reshaping to (out_dim, in_dim) gives the correct
        weight matrix. The C kernels read raw quantized data directly in output-major order.
        """
        self.weights = {}       # F32 weights as (out_dim, in_dim)
        self.raw_weights = {}   # Flat uint8 bytes for C kernel dispatch
        self.weight_qtypes = {}
        self.weight_info = {}   # (n_rows, n_cols, type_size_or_0, quant_type) per weight

        for t in self.reader.tensors:
            name = t.name
            qtype = QT(t.tensor_type).value
            self.weight_qtypes[name] = qtype

            if len(t.shape) == 2:
                in_dim = int(t.shape[0])   # GGUF: first dim is input
                out_dim = int(t.shape[1])  # GGUF: second dim is output

                # Dequantize and reshape as (out_dim, in_dim)
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                f32 = f32.reshape(out_dim, in_dim)
                self.weights[name] = np.ascontiguousarray(f32)

                # Store raw quantized bytes and C kernel parameters
                if qtype in C_KERNEL_TYPES:
                    raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8).copy()
                    self.raw_weights[name] = raw
                    ts = BLOCK_SIZES[qtype]
                    self.weight_info[name] = (out_dim, in_dim, ts, qtype)

            elif len(t.shape) == 3:
                # MoE expert weights: GGUF shape (in_dim, out_dim, n_experts)
                # gguf.dequantize returns (n_experts, out_dim, in_dim)
                in_dim = int(t.shape[0])
                out_dim = int(t.shape[1])
                n_experts = int(t.shape[2])

                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                # Dequantize returns (n_experts, out_dim, in_dim) — already correct
                assert f32.shape == (n_experts, out_dim, in_dim), \
                    f"Unexpected shape {f32.shape} for {name}, expected ({n_experts}, {out_dim}, {in_dim})"
                self.weights[name] = np.ascontiguousarray(f32)

            elif len(t.shape) == 1:
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
                self.weights[name] = f32

        # 3D expert weights (MoE)
        for t in self.reader.tensors:
            name = t.name
            qtype = QT(t.tensor_type).value
            if len(t.shape) == 3:
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                in_dim = int(t.shape[0])
                out_dim = int(t.shape[1])
                n_experts = int(t.shape[2])
                f32 = f32.reshape(out_dim, in_dim, n_experts)
                # Expert weights: (out_dim, in_dim, n_experts) for X @ W per-expert
                self.weights[name] = np.ascontiguousarray(f32)
                self.weight_qtypes[name] = qtype
                # For MoE, store each expert's raw data separately
                if qtype in C_KERNEL_TYPES:
                    raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8).copy()
                    ts = BLOCK_SIZES[qtype]
                    self.weight_info[name] = (out_dim, in_dim, ts, qtype)
                    # We'll handle expert dispatch in _matmul

# Identify output weight and derive vocab_size
        if 'output.weight' in self.weights:
            self.out_w_name = 'output.weight'
        else:
            self.out_w_name = 'token_embd.weight'

        # Ensure embedding matrix is (vocab, n_embd) for lookup
        emb = self.weights['token_embd.weight']
        if emb.ndim == 2:
            # emb was reshaped as (out_dim, in_dim) from GGUF (n_embd, vocab_size)
            # For lookup we need (vocab_size, n_embd)
            if emb.shape[1] == self.n_embd:
                # Shape is (vocab_size, n_embd) — correct for lookup
                pass
            else:
                # Shape is (n_embd, vocab_size) — needs transpose
                emb = np.ascontiguousarray(emb.T)
            self.weights['token_embd.weight'] = emb

        # Derive vocab_size from embedding shape
        if self.vocab_size == 0:
            self.vocab_size = emb.shape[0]

        self.emb = emb

    def _load_kernels(self):
        """Load the unified quant kernel library."""
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        cf = ctypes.POINTER(ctypes.c_float)
        cu = ctypes.POINTER(ctypes.c_uint8)

        self._quant_so = None
        so_path = os.path.join(kernel_dir, 'quant_kernels_omp.so')
        if os.path.exists(so_path):
            so = ctypes.CDLL(so_path)
            so.quant_matmul_omp.argtypes = [cu, cf, cf, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            so.quant_matmul_omp.restype = None
            so.f32_matmul_omp.argtypes = [cf, cf, cf, ctypes.c_int, ctypes.c_int]
            so.f32_matmul_omp.restype = None
            self._quant_so = so

    def _init_kv_cache(self):
        """Pre-allocate KV cache and computation buffers."""
        N = self.n_embd
        FF = self.n_ff if not self.is_moe else self.n_ff_expert
        NH = self.n_head
        NKH = self.n_kv_head
        HD = self.head_dim
        MAX_POS = 4096

        self.kv_k = np.zeros((self.n_layers, MAX_POS, NKH * HD), dtype=np.float32)
        self.kv_v = np.zeros((self.n_layers, MAX_POS, NKH * HD), dtype=np.float32)
        self.kv_len = np.zeros(self.n_layers, dtype=np.int32)

        self._buf = {
            'h': np.zeros(N, dtype=np.float32),
            'residual': np.zeros(N, dtype=np.float32),
            'q': np.zeros(NH * HD, dtype=np.float32),
            'k': np.zeros(NKH * HD, dtype=np.float32),
            'v': np.zeros(NKH * HD, dtype=np.float32),
            'att_out': np.zeros(NH * HD, dtype=np.float32),
            'gate': np.zeros(FF, dtype=np.float32),
            'up': np.zeros(FF, dtype=np.float32),
            'silu_gate': np.zeros(FF, dtype=np.float32),
            'ffn_out': np.zeros(N, dtype=np.float32),
        }

    def reset(self):
        """Reset KV cache and position."""
        self.kv_len[:] = 0
        self.pos = 0

    def _matmul(self, name, x):
        """Dispatch matmul. Weight stored as (out_dim, in_dim).
        
        Computes: W_T @ x where W_T has shape (out_dim, in_dim).
        For C kernels: raw data is output-neuron-major, pass (nrows=out_dim, ncols=in_dim).
        For F32/numpy: W_T @ x is equivalent to x @ W (correct LLM projection).
        """
        w = self.weights[name]
        if w.ndim == 1:
            return w * x

        n_rows = w.shape[0]  # out_dim
        n_cols = w.shape[1]  # in_dim
        qtype = self.weight_qtypes.get(name, GGML_TYPE_F32)

        # Try C kernel for supported quant types
        info = self.weight_info.get(name)
        if info is not None and self._quant_so is not None:
            nr, nc, ts, qt = info
            raw = self.raw_weights[name]
            out = np.zeros(nr, dtype=np.float32)
            self._quant_so.quant_matmul_omp(
                raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                nr, nc, qt  # quant_type=GGUF type code
            )
            return out

        # Fallback: F32 numpy BLAS — W_T @ x where W_T is (out_dim, in_dim)
        return w @ x

    def _rms_norm(self, x, name, eps=None):
        """RMS normalization."""
        if eps is None:
            eps = self.eps
        w = self.weights[name]
        var = np.mean(x * x)
        return x / np.sqrt(var + eps) * w

    def _silu(self, x):
        """SiLU activation."""
        return x / (1.0 + np.exp(-x))

    def _apply_rope(self, x, pos, n_heads):
        """Apply Rotary Position Embedding."""
        hd = self.head_dim
        half = hd // 2
        freq = self.rope_freq_base ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
        angle = pos / freq
        cos_a = np.cos(angle).astype(np.float32)
        sin_a = np.sin(angle).astype(np.float32)

        x2d = x.reshape(n_heads, hd)
        out = np.empty_like(x2d)
        out[:, :half] = x2d[:, :half] * cos_a - x2d[:, half:] * sin_a
        out[:, half:] = x2d[:, half:] * cos_a + x2d[:, :half] * sin_a
        return out.reshape(-1)

    def forward(self, token_id):
        """Run one forward pass. Returns logits."""
        b = self._buf
        N = self.n_embd
        NH = self.n_head
        NKH = self.n_kv_head
        HD = self.head_dim
        L = self.n_layers

        # Embedding lookup
        np.copyto(b['h'], self.emb[token_id])

        for i in range(L):
            pfx = f'blk.{i}'

            # Attention
            np.copyto(b['residual'], b['h'])
            b['h'][:] = self._rms_norm(b['h'], f'{pfx}.attn_norm.weight')

            # Q/K/V projections
            b['q'][:] = self._matmul(f'{pfx}.attn_q.weight', b['h'])
            b['k'][:] = self._matmul(f'{pfx}.attn_k.weight', b['h'])[:NKH * HD]
            b['v'][:] = self._matmul(f'{pfx}.attn_v.weight', b['h'])[:NKH * HD]

            # Q/K norm (Qwen3-specific: per-head RMS norm)
            q_norm_name = f'{pfx}.attn_q_norm.weight'
            k_norm_name = f'{pfx}.attn_k_norm.weight'
            if q_norm_name in self.weights:
                q_w = self.weights[q_norm_name]
                if q_w.shape[0] == HD:
                    # Per-head norm: reshape Q from (NH*HD,) to (NH, HD), norm each head
                    q_2d = b['q'].reshape(NH, HD)
                    for h in range(NH):
                        q_2d[h] = q_2d[h] / np.sqrt(np.mean(q_2d[h] * q_2d[h]) + self.eps) * q_w
                else:
                    b['q'][:] = self._rms_norm(b['q'], q_norm_name)
            if k_norm_name in self.weights:
                k_w = self.weights[k_norm_name]
                if k_w.shape[0] == HD:
                    k_2d = b['k'][:NKH * HD].reshape(NKH, HD)
                    for h in range(NKH):
                        k_2d[h] = k_2d[h] / np.sqrt(np.mean(k_2d[h] * k_2d[h]) + self.eps) * k_w

            # RoPE
            b['q'][:] = self._apply_rope(b['q'], self.pos, NH)
            b['k'][:] = self._apply_rope(b['k'], self.pos, NKH)

            # KV cache
            self.kv_k[i, self.kv_len[i], :NKH * HD] = b['k'][:NKH * HD]
            self.kv_v[i, self.kv_len[i], :NKH * HD] = b['v'][:NKH * HD]

            # Attention scores
            seq_len = self.kv_len[i] + 1
            k_cache = self.kv_k[i, :seq_len].reshape(seq_len, NKH, HD)
            v_cache = self.kv_v[i, :seq_len].reshape(seq_len, NKH, HD)

            # GQA
            if NH != NKH:
                n_rep = NH // NKH
                k_exp = np.repeat(k_cache, n_rep, axis=1)
                v_exp = np.repeat(v_cache, n_rep, axis=1)
            else:
                k_exp = k_cache
                v_exp = v_cache

            q_2d = b['q'].reshape(NH, HD)
            scores = np.einsum('hd,shd->hs', q_2d, k_exp.reshape(-1, NH, HD)) / np.sqrt(float(HD))
            scores -= np.max(scores, axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= np.sum(scores, axis=1, keepdims=True)
            att = np.einsum('hs,shd->hd', scores, v_exp.reshape(-1, NH, HD)).astype(np.float32)
            b['att_out'][:] = att.reshape(-1)

            self.kv_len[i] += 1

            # Output projection
            o_out = self._matmul(f'{pfx}.attn_output.weight', b['att_out'])
            b['h'][:] = b['residual'] + o_out[:N]

            # FFN
            np.copyto(b['residual'], b['h'])
            b['h'][:] = self._rms_norm(b['h'], f'{pfx}.ffn_norm.weight')

            if self.is_moe:
                self._forward_moe(i, b)
            else:
                self._forward_ffn(i, b)

            b['h'][:N] = b['residual'][:N] + b['ffn_out'][:N]

        # Final norm + output projection
        b['h'][:] = self._rms_norm(b['h'], 'output_norm.weight')
        logits = self._matmul(self.out_w_name, b['h'])
        self.pos += 1
        return logits

    def _forward_ffn(self, layer, b):
        """Dense FFN."""
        pfx = f'blk.{layer}'
        FF = self.n_ff
        N = self.n_embd

        gate = self._matmul(f'{pfx}.ffn_gate.weight', b['h'])[:FF]
        up = self._matmul(f'{pfx}.ffn_up.weight', b['h'])[:FF]
        b['silu_gate'][:FF] = self._silu(gate) * up
        b['ffn_out'][:N] = self._matmul(f'{pfx}.ffn_down.weight', b['silu_gate'])[:N]

    def _forward_moe(self, layer, b):
        """MoE FFN with expert routing."""
        pfx = f'blk.{layer}'
        N = self.n_embd
        FF = self.n_ff_expert
        n_exp = self.n_experts
        top_k = self.n_experts_per_tok

        # Expert routing scores
        gate_scores = self._matmul(f'{pfx}.ffn_gate_inp.weight', b['h'])[:n_exp]

        # Softmax
        gate_scores -= np.max(gate_scores)
        np.exp(gate_scores, out=gate_scores)
        gate_scores /= np.sum(gate_scores)

        # Top-K experts
        top_indices = np.argsort(gate_scores)[-top_k:][::-1]
        top_weights = gate_scores[top_indices]
        top_weights /= np.sum(top_weights)

        result = np.zeros(N, dtype=np.float32)
        for expert_id, weight in zip(top_indices, top_weights):
            # Expert weight matrices: (out_dim, in_dim, n_experts)
            gate_exp = self.weights[f'{pfx}.ffn_gate_exps.weight'][:, :, expert_id]
            up_exp = self.weights[f'{pfx}.ffn_up_exps.weight'][:, :, expert_id]
            down_exp = self.weights[f'{pfx}.ffn_down_exps.weight'][:, :, expert_id]

            gate_out = (gate_exp @ b['h'])[:FF]
            up_out = (up_exp @ b['h'])[:FF]
            silu_out = self._silu(gate_out) * up_out
            expert_out = (down_exp @ silu_out)[:N]
            result += weight * expert_out

        b['ffn_out'][:N] = result


if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'Llama-3.2-1B-Instruct-Q4_0.gguf'
    n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    engine = TurboEngine(model_path, n_threads=n_threads)

    # Warmup
    engine.reset()
    logits = engine.forward(128000)
    tok = int(np.argmax(logits))
    for _ in range(5):
        logits = engine.forward(tok)
        tok = int(np.argmax(logits))

    # Benchmark
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        logits = engine.forward(tok)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        tok = int(np.argmax(logits))

    times.sort()
    median = times[len(times) // 2]
    print(f"\nResult: {median:.2f} ms/tok = {1000.0/median:.1f} tok/s")
    print(f"Top-5 tokens: {np.argsort(logits)[-5:][::-1]}")