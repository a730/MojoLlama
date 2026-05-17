"""
MojoLlama Bridge — Python backend for the Mojo op graph.

Loads GGUF models, executes ops via numpy, handles tokenizer/sampling.
When Mojo's native APIs mature, ops are swapped — the graph stays the same.

Architecture:
    bridge.py (Python)          ← GGUF loading, tokenizer, numpy backend
        ↕ (calls via Mojo interop or stdin/stdout)
    ops.mojo (Mojo)             ← Op graph definitions
    kernels/ (Mojo, future)     ← Native Mojo SIMD implementations
    max/ (MAX, future)          ← MAX GPU/CUDA/SYCL implementations

Current: bridge.py executes the full forward pass via numpy.
Future: bridge.py only handles I/O, ops execute in Mojo SIMD or MAX GPU.
"""
import numpy as np
import gguf
from gguf.constants import GGMLQuantizationType
import struct
import time
import sys
import os


class MojoLlamaBridge:
    """Python bridge for MojoLlama. Manages model loading and op execution."""

    def __init__(self, path: str):
        self.path = path
        self.reader = gguf.GGUFReader(path)
        self._tensors_by_name = {t.name: t for t in self.reader.tensors}
        self._load_config()
        self._load_weights()
        self._kv_cache = None
        self._cached_len = 0

    def _load_config(self):
        f = self.reader.get_field
        self.arch = self._get_str(f('general.architecture'))
        prefix = f'{self.arch}.'
        self.n_layers = self._get_scalar(f(f'{prefix}block_count'))
        self.n_embd = self._get_scalar(f(f'{prefix}embedding_length'))
        self.n_head = self._get_scalar(f(f'{prefix}attention.head_count'))
        self.n_kv_head = self._get_scalar(f(f'{prefix}attention.head_count_kv'))
        self.n_ff = self._get_scalar(f(f'{prefix}feed_forward_length'))
        self.norm_eps = self._get_scalar(f(f'{prefix}attention.layer_norm_rms_epsilon'))

    def _get_scalar(self, f):
        if f is None:
            return 0
        v = f.parts[-1]
        if hasattr(v, 'item'):
            return v.item()
        return int(v)

    def _get_str(self, f):
        if f is None:
            return 'unknown'
        v = f.parts[-1]
        if isinstance(v, bytes):
            return v.decode('utf-8', errors='replace')
        if hasattr(v, 'tobytes'):
            return v.tobytes().decode('utf-8', errors='replace').strip('\x00')
        return str(v)

    def _get_int(self, f):
        if f is None:
            return 0
        return int(f.parts[-1])

    def _get_float(self, f):
        if f is None:
            return 1e-6
        return float(f.parts[-1])

    def _load_weights(self):
        """Load all tensors, dequantizing Q4_0 to float32."""
        self.weights = {}
        for name, t in self._tensors_by_name.items():
            if hasattr(t, 'tensor_type') and t.tensor_type is not None:
                if t.tensor_type == GGMLQuantizationType.Q4_0:
                    # Dequantize Q4_0 to float32
                    arr = gguf.dequantize(t.data, t.tensor_type)
                    self.weights[name] = ('q4_0', arr)
                else:
                    arr = gguf.dequantize(t.data, t.tensor_type)
                    self.weights[name] = ('f32', arr)
            else:
                self.weights[name] = ('f32', np.array(t.data, dtype=np.float32))

    def embed(self, input_ids):
        """EmbedOp: token_embd.weight[input_ids]"""
        typ, embed = self.weights.get('token_embd.weight', (None, None))
        if embed is None:
            raise ValueError("token_embd.weight not found")
        # Handle reverse shape: GGUF stores [in_cols, out_rows]
        if embed.ndim == 2 and embed.shape[0] == self.n_embd:
            embed = embed.T
        return embed[input_ids].astype(np.float32)

    def matmul(self, name, x):
        """MatmulOp: x @ W.T for a named weight tensor."""
        typ, w = self.weights.get(name, (None, None))
        if w is None:
            raise ValueError(f"Weight {name} not found")

        if typ == 'q4_0':
            # Q4_0: w is (out_rows, packed_cols) — dequantized via gguf
            # gguf stores weights as 2D: shape=[cols, rows] but data is (rows, packed)
            return x @ w.T
        else:
            # Float32 — always x @ W.T (GGUF stores transposed)
            return x @ w.T

    def _q4_matmul(self, raw, x):
        """Compute x @ W.T for Q4_0 weight stored in GGUF raw format."""
        import ctypes
        # Load C kernel
        lib_path = os.path.join(os.path.dirname(__file__), 'model', 'libq4matmul.so')
        if not hasattr(self, '_q4_lib'):
            self._q4_lib = ctypes.CDLL(lib_path)
            self._q4_lib.q4_matmul_forward_t.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self._q4_lib.q4_matmul_forward_t.restype = None

        # Determine dimensions from raw GGUF format
        # raw shape: (out_rows, N) where N = (in_cols / 32) * 18 for Q4_0
        n_rows, row_bytes = raw.shape
        in_cols = (row_bytes // 18) * 32

        x_2d = x.reshape(-1, in_cols) if x.ndim == 1 else x
        batch = x_2d.shape[0]

        out = np.zeros((batch, n_rows), dtype=np.float32)
        raw_buf = (ctypes.c_uint8 * raw.nbytes).from_buffer_copy(raw.tobytes())

        self._q4_lib.q4_matmul_forward_t(
            raw_buf,
            x_2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_rows, in_cols, batch,
        )
        return out.reshape(x.shape[:-1] + (n_rows,))

    def rms_norm(self, x, weight_name):
        """RMSNormOp."""
        _, w = self.weights[weight_name]
        ss = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(ss + self.norm_eps) * w

    def rope(self, x, cos, sin):
        """RoPEOp."""
        n, h, d = x.shape
        x2 = x.reshape(n, h, d // 2, 2)
        xr = np.stack([-x2[..., 1], x2[..., 0]], axis=-1)
        c = cos[:n, np.newaxis, :d // 2, np.newaxis]
        s = sin[:n, np.newaxis, :d // 2, np.newaxis]
        return (x2 * c + xr * s).reshape(n, h, d)

    def silu(self, x):
        """SiLUOp."""
        return x / (1 + np.exp(-x))

    def attention(self, q, k, v, mask):
        """AttentionOp: softmax(Q @ K.T / sqrt(d)) @ V"""
        d = q.shape[-1]
        scores = (q @ k.swapaxes(-1, -2)) / (d ** 0.5) + mask
        att = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        att = att / np.sum(att, axis=-1, keepdims=True)
        return att @ v

    def forward(self, input_ids):
        """Full forward pass — executes the Llama3 op graph via numpy."""
        # Precompute RoPE
        head_dim = self.n_embd // self.n_head
        cos, sin = self._precompute_freqs(head_dim, len(input_ids))

        # Embedding
        h = self.embed(input_ids)

        # KV cache
        if self._kv_cache is not None:
            cached_len = self._cached_len
            new_ids = input_ids[cached_len:]
            if len(new_ids) == 0:
                return None
            h = self.embed(new_ids)
            return self._forward_step(h, cos, sin, cached_len)

        # Full forward (prefill)
        mask = np.triu(np.full((len(input_ids), len(input_ids)), -np.inf), 1)
        return self._forward_full(h, cos, sin, mask)

    def _forward_full(self, h, cos, sin, mask):
        """Full forward pass through all layers."""
        seq_len = h.shape[0]
        head_dim = self.n_embd // self.n_head
        n_rep = self.n_head // self.n_kv_head

        self._kv_cache = []
        self._cached_len = seq_len

        for i in range(self.n_layers):
            # Attention block
            r = h
            h = self.rms_norm(h, f'blk.{i}.attn_norm.weight')
            q = self.matmul(f'blk.{i}.attn_q.weight', h)
            k = self.matmul(f'blk.{i}.attn_k.weight', h)
            v = self.matmul(f'blk.{i}.attn_v.weight', h)

            q = q.reshape(seq_len, self.n_head, head_dim)
            k = k.reshape(seq_len, self.n_kv_head, head_dim)
            v = v.reshape(seq_len, self.n_kv_head, head_dim)
            q = self.rope(q, cos, sin)
            k = self.rope(k, cos, sin)

            self._kv_cache.append((k.copy(), v.copy()))

            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=1)
                v = np.repeat(v, n_rep, axis=1)

            # Transpose to (head, seq, dim) for attention
            q_t = q.transpose(1, 0, 2)
            k_t = k.transpose(1, 0, 2)
            v_t = v.transpose(1, 0, 2)

            att = self.attention(q_t, k_t, v_t, mask)
            att = att.transpose(1, 0, 2).reshape(seq_len, self.n_embd)
            h = r + self.matmul(f'blk.{i}.attn_output.weight', att)

            # FFN block
            r = h
            h = self.rms_norm(h, f'blk.{i}.ffn_norm.weight')
            gate = self.silu(self.matmul(f'blk.{i}.ffn_gate.weight', h))
            up = self.matmul(f'blk.{i}.ffn_up.weight', h)
            h = r + self.matmul(f'blk.{i}.ffn_down.weight', gate * up)

        # Output
        h = self.rms_norm(h, 'output_norm.weight')
        logits = self.matmul('output.weight', h) if 'output.weight' in self.weights else h @ self.weights.get('token_embd.weight', (None, None))[1].T
        return logits

    def _forward_step(self, h, cos, sin, cached_len):
        """Single-token forward with KV cache."""
        new_len = h.shape[0]
        new_total = cached_len + new_len
        self._cached_len = new_total
        head_dim = self.n_embd // self.n_head
        n_rep = self.n_head // self.n_kv_head

        for i in range(self.n_layers):
            r = h
            h = self.rms_norm(h, f'blk.{i}.attn_norm.weight')
            q = self.matmul(f'blk.{i}.attn_q.weight', h)
            k_new = self.matmul(f'blk.{i}.attn_k.weight', h)
            v_new = self.matmul(f'blk.{i}.attn_v.weight', h)

            q = q.reshape(new_len, self.n_head, head_dim)
            k_new = k_new.reshape(new_len, self.n_kv_head, head_dim)
            v_new = v_new.reshape(new_len, self.n_kv_head, head_dim)
            q = self.rope(q, cos[cached_len:new_total], sin[cached_len:new_total])
            k_new = self.rope(k_new, cos[cached_len:new_total], sin[cached_len:new_total])

            k_cached, v_cached = self._kv_cache[i]
            k = np.concatenate([k_cached, k_new], axis=0)
            v = np.concatenate([v_cached, v_new], axis=0)
            self._kv_cache[i] = (k, v)

            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=1)
                v = np.repeat(v, n_rep, axis=1)

            mask = np.full((new_len, new_total), -np.inf, dtype=np.float32)
            for j in range(new_len):
                mask[j, :cached_len + j + 1] = 0.0

            q_t = q.transpose(1, 0, 2)
            k_t = k.transpose(1, 0, 2)
            v_t = v.transpose(1, 0, 2)
            att = self.attention(q_t, k_t, v_t, mask)
            att = att.transpose(1, 0, 2).reshape(new_len, self.n_embd)
            h = r + self.matmul(f'blk.{i}.attn_output.weight', att)

            r = h
            h = self.rms_norm(h, f'blk.{i}.ffn_norm.weight')
            gate = self.silu(self.matmul(f'blk.{i}.ffn_gate.weight', h))
            up = self.matmul(f'blk.{i}.ffn_up.weight', h)
            h = r + self.matmul(f'blk.{i}.ffn_down.weight', gate * up)

        h = self.rms_norm(h, 'output_norm.weight')
        logits = self.matmul('output.weight', h) if 'output.weight' in self.weights else h @ self.weights['token_embd.weight'][1].T
        return logits

    def _precompute_freqs(self, dim, end):
        theta = 500000.0
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(end).astype(np.float32)
        freqs = np.outer(t, freqs)
        return np.cos(freqs), np.sin(freqs)

    def generate(self, prompt: str, max_tokens: int = 50):
        """Generate text from prompt."""
        ids = self.tokenize(prompt)
        out = []
        self._kv_cache = None
        self._cached_len = 0

        for step in range(max_tokens):
            logits = self.forward(ids)
            if logits is None:
                break
            nid = int(np.argmax(logits[-1]))
            if nid == self._eos_id():
                break
            out.append(nid)
            ids.append(nid)

        return self.detokenize(out)

    def tokenize(self, text: str):
        """Simple BPE tokenizer."""
        from mojollama.model.inference import LLMInference
        return LLMInference(self.path).encode(text)

    def detokenize(self, ids):
        from mojollama.model.inference import LLMInference
        return LLMInference(self.path).decode(ids)

    def _eos_id(self):
        f = self.reader.get_field('tokenizer.ggml.eos_token_id')
        if f is None:
            return 2
        return int(f.parts[-1].item()) if hasattr(f.parts[-1], 'item') else int(f.parts[-1])
