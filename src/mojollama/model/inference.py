"""
Python inference module for MojoLlama — memory-efficient per-layer processing.
Supports Qwen2 and Llama architectures from GGUF files.
With GPU acceleration: Intel Arc (SYCL), NVIDIA CUDA, or CPU.
"""
import gc
import logging
import os
import numpy as np
import gguf
# pyrefly: ignore [untyped-import]
import regex as re

from mojollama.model.device import get_device, DeviceBackend, DeviceType

logger = logging.getLogger(__name__)

_gpt2_pat = re.compile(r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# Llama 3 regex pattern (tiktoken-style pre-tokenization)
_llama3_pat = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

# GPT-2 byte-to-unicode encoder/decoder
_byte_encoder = {}
for i in range(ord('!'), ord('~') + 1):
    _byte_encoder[i] = chr(i)
for i in range(ord('\xa1'), ord('\xac') + 1):
    _byte_encoder[i] = chr(i)
for i in range(ord('\xae'), ord('\xff') + 1):
    _byte_encoder[i] = chr(i)
bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('\xa1'), ord('\xac')+1)) + list(range(ord('\xae'), ord('\xff')+1))
n = 0
for b in range(256):
    if b not in bs:
        _byte_encoder[b] = chr(256 + n)
        n += 1
_byte_decoder = {v: k for k, v in _byte_encoder.items()}

def _bytes_to_unicode(s: bytes) -> str:
    return ''.join(_byte_encoder[b] for b in s)

def _unicode_to_bytes(s: str) -> bytes:
    return bytes(_byte_decoder[c] for c in s)


def _get_np(val):
    """Convert gguf scalar to Python value."""
    if val is None:
        return None
    if hasattr(val, 'parts'):
        p = val.parts[-1]
        if hasattr(p, 'item'):
            return p.item() if p.size == 1 else p
        return p
    if isinstance(val, np.ndarray):
        return val.item() if val.size == 1 else val
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return val


def to_cpu_if_needed(arr, device_backend=None):
    """Convert device array to CPU numpy if needed."""
    if device_backend is not None and hasattr(arr, '__class__'):
        import numpy as _np
        if type(arr).__module__ != 'numpy':
            try:
                return _np.asarray(arr)
            except Exception:
                pass
    return arr


class LLMInference:
    """General LLM inference engine supporting multiple architectures from GGUF files."""

    ARCH_PREFIXES = {
        'qwen2': ['qwen2.', 'qwen2moe.'],
        'llama': ['llama.', 'llama3.', 'llama2.', 'codellama.'],
    }

    def __init__(self, path: str, device: str = 'auto'):
        """
        Initialize inference engine.

        Args:
            path: Path to GGUF model file
            device: Compute device ('auto', 'cpu', 'intel_arc', 'nvidia')
                    Defaults to 'auto' which picks best available.
                    Override via MOJOLLAMA_DEVICE env var.
        """
        # Initialize device backend
        env_device = os.environ.get('MOJOLLAMA_DEVICE', '').lower()
        self.device = get_device(env_device or device)
        if not self.device.is_available:
            print(f"[MojoLlama] Device '{self.device.device_type.value}' unavailable, falling back to CPU")
            self.device = get_device('cpu')
        print(f"[MojoLlama] Using device: {self.device.capability}")

        self.reader = gguf.GGUFReader(path)
        self._tensors_by_name = {t.name: t for t in self.reader.tensors}
        self._load_config()
        self._build_tokenizer()
        self._device_tensors = {}  # GPU-side tensor cache

    def _try_config(self, key: str, default=None):
        f = self.reader.get_field
        for prefix in self._config_prefixes:
            v = f(f'{prefix}{key}')
            if v is not None:
                return _get_np(v)
        v = f(key)
        if v is not None:
            return _get_np(v)
        return default

    def _load_config(self):
        f = self.reader.get_field
        arch_field = f('general.architecture')
        arch_val = _get_np(arch_field) if arch_field else None
        if isinstance(arch_val, np.ndarray) and arch_val.dtype.kind == 'u':
            arch_str = arch_val.tobytes().decode('utf-8', errors='replace')
        elif isinstance(arch_val, bytes):
            arch_str = arch_val.decode('utf-8', errors='replace')
        elif arch_val is not None:
            arch_str = str(arch_val)
        else:
            arch_str = 'unknown'
        self.arch = arch_str

        if arch_str in self.ARCH_PREFIXES:
            self._config_prefixes = self.ARCH_PREFIXES[arch_str]
        else:
            all_prefixes = set()
            for prefixes in self.ARCH_PREFIXES.values():
                all_prefixes.update(prefixes)
            self._config_prefixes = [f'{arch_str}.'] + sorted(all_prefixes)

        self.n_layers = int(self._try_config('block_count', 0))
        self.n_embd = int(self._try_config('embedding_length', 0))
        self.n_head = int(self._try_config('attention.head_count', 0))
        self.n_kv_head = int(self._try_config('attention.head_count_kv', 0))
        self.n_ff = int(self._try_config('feed_forward_length', 0))
        self.max_seq_len = int(self._try_config('context_length',
                                                self._try_config('max_position_embeddings', 32768)))
        self.norm_eps = float(self._try_config('attention.layer_norm_rms_epsilon',
                                                self._try_config('attn_layer_norm_rms_epsilon', 1e-6)))
        base = self._try_config('rope.freq_base', None)
        self.rope_theta = float(base) if base is not None else 1000000.0
        self.rope_type = self._try_config('rope.type', 'default')
        self.has_bias = False
        t0 = self._tensors_by_name.get('blk.0.attn_q.bias')
        if t0 is not None:
            self.has_bias = True

        print(f"[MojoLlama] Architecture: {arch_str}")
        print(f"[MojoLlama] Layers: {self.n_layers}, dim: {self.n_embd}, heads: {self.n_head}, kv_heads: {self.n_kv_head}")
        print(f"[MojoLlama] FFN dim: {self.n_ff}, max_seq_len: {self.max_seq_len}")
        print(f"[MojoLlama] norm_eps: {self.norm_eps}, rope_theta: {self.rope_theta}")
        print(f"[MojoLlama] rope_type: {self.rope_type}, has_bias: {self.has_bias}")

    def _tensor(self, gguf_name: str):
        """Load a tensor from GGUF and move to device if accelerator is active."""
        # Check device cache first
        if gguf_name in self._device_tensors:
            return self._device_tensors[gguf_name]

        t = self._tensors_by_name.get(gguf_name)
        if t is None:
            return None

        # Dequantize / load to numpy
        if hasattr(t, 'tensor_type') and t.tensor_type is not None:
            tt = t.tensor_type
            if tt == gguf.GGMLQuantizationType.F32:
                arr = np.array(t.data, dtype=np.float32)
            elif tt == gguf.GGMLQuantizationType.F16:
                arr = np.array(t.data, dtype=np.float16).astype(np.float32)
            else:
                try:
                    arr = gguf.dequantize(t.data, tt)
                except Exception:
                    arr = np.array(t.data, dtype=np.float32)
        else:
            arr = np.array(t.data, dtype=np.float32)

        # Move to device if we have a GPU backend
        if self.device.device_type != DeviceType.CPU:
            device_arr = self.device.to_device(arr)
            self._device_tensors[gguf_name] = device_arr
            return device_arr

        return arr

    def _get_persistent(self, name: str):
        if not hasattr(self, '_persistent_cache'):
            self._persistent_cache = {}
        if name not in self._persistent_cache:
            self._persistent_cache[name] = self._tensor(name)
        return self._persistent_cache[name]

    def _build_tokenizer(self):
        fields = self.reader.fields
        token_field = fields.get('tokenizer.ggml.tokens')
        merge_field = fields.get('tokenizer.ggml.merges')

        # Build vocab: raw_bytes -> token_id (for Llama 3), utf8_str -> token_id (for Qwen2)
        self._token_bytes = {}  # bytes -> id
        self.vocab = {}  # str -> id (for GPT-2 style)
        header_size = 5
        if token_field:
            parts = list(token_field.parts)
            n_tokens = int(parts[4].item()) if len(parts) > 4 else 0
            for i in range(n_tokens):
                data_idx = header_size + i * 2 + 1
                if data_idx >= len(parts):
                    break
                token_data = parts[data_idx].tobytes() if hasattr(parts[data_idx], 'tobytes') else b''
                self._token_bytes[token_data] = i
                try:
                    s = token_data.decode('utf-8', errors='replace')
                    self.vocab[s] = i
                except:
                    self.vocab[token_data.hex()] = i

        # Build merge_ranks: (str, str) -> rank in GPT-2 encoded form
        self.merge_ranks = {}
        if merge_field:
            parts = list(merge_field.parts)
            n_merges = int(parts[4].item()) if len(parts) > 4 else 0
            for i in range(n_merges):
                data_idx = header_size + i * 2 + 1
                if data_idx >= len(parts):
                    break
                merge_data = parts[data_idx].tobytes() if hasattr(parts[data_idx], 'tobytes') else b''
                try:
                    s = merge_data.decode('utf-8', errors='replace')
                    sp = s.split(' ')
                    if len(sp) == 2:
                        self.merge_ranks[tuple(sp)] = i
                except:
                    pass

        bos = fields.get('tokenizer.ggml.bos_token_id')
        self.bos_id = int(_get_np(bos)) if bos else 1
        eos = fields.get('tokenizer.ggml.eos_token_id')
        self.eos_id = int(_get_np(eos)) if eos else 2
        add = fields.get('tokenizer.ggml.add_bos_token')
        self.add_bos = bool(int(_get_np(add))) if add else True

        # Detect if this is a tiktoken-style vocab (Llama 3+)
        self._is_tiktok = len(self.vocab) >= 128000 and len(self.merge_ranks) > 250000

        print(f"[MojoLlama] Vocab size: {len(self.vocab)}, merges: {len(self.merge_ranks)}")
        print(f"[MojoLlama] BOS: {self.bos_id}, EOS: {self.eos_id}, add_bos: {self.add_bos}")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs using BPE."""
        ids = self._encode_bpe(text)
        if self.add_bos:
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        # Build reverse lookup if not cached
        if not hasattr(self, '_id_to_bytes'):
            self._id_to_bytes = {v: k for k, v in self._token_bytes.items()}
        raw_chunks = []
        for tid in ids:
            tb = self._id_to_bytes.get(tid)
            if tb is not None:
                gpt2_str = tb.decode('utf-8', errors='replace')
                raw_chunks.append(gpt2_str)
        full_gpt2 = ''.join(raw_chunks)
        try:
            return _unicode_to_bytes(full_gpt2).decode('utf-8', errors='replace')
        except:
            return full_gpt2

    def _encode_bpe(self, text: str) -> list[int]:
        """BPE encoding. For tiktoken vocabs: uses GPT-2 byte encoding for merges,
        then encodes as UTF-8 for vocab lookup."""
        import ftfy
        text = ftfy.fix_text(text)

        if self._is_tiktok:
            pat = _llama3_pat
        else:
            pat = _gpt2_pat

        tokens = []
        for m in re.findall(pat, text):
            # Convert word to GPT-2 byte-encoded form
            gpt2_str = _bytes_to_unicode(m.encode('utf-8'))

            # BPE merge loop on GPT-2 encoded string
            pairs = list(gpt2_str)
            while True:
                best = None
                rank = None
                for i in range(len(pairs) - 1):
                    key = (pairs[i], pairs[i+1])
                    r = self.merge_ranks.get(key)
                    if r is not None and (rank is None or r < rank):
                        best = i
                        rank = r
                if best is None:
                    break
                pairs = pairs[:best] + [pairs[best] + pairs[best+1]] + pairs[best+2:]

            # Look up each merged GPT-2 token: encode as UTF-8 and find in token_bytes
            for token_gpt2 in pairs:
                token_utf8 = token_gpt2.encode('utf-8')
                tid = self._token_bytes.get(token_utf8, 0)
                tokens.append(tid)

        return tokens

    # ─── Device-aware ops ──────────────────────────────────────────────────

    def rms_norm(self, x, weight, eps=1e-6):
        return self.device.rms_norm(x, weight, eps)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float):
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(end).astype(np.float32)
        freqs = np.outer(t, freqs)
        return np.cos(freqs), np.sin(freqs)

    def apply_rope(self, x, cos, sin):
        return self.device.rope(x, cos, sin)

    def silu(self, x):
        return self.device.silu(x)

    def softmax(self, x, axis=-1):
        return self.device.softmax(x, axis)

    # ─── Attention helper (device-aware) ───────────────────────────────────

    def _attention_scores(self, q, k, head_dim: float):
        """Compute attention scores using device-aware batch matmul.
        
        Args:
            q: (seq_len, n_head, head_dim)
            k: (seq_len, n_kv_head or n_head, head_dim)
            
        Returns:
            att: (n_head, seq_len, seq_len) OR (n_kv_head, seq_len, seq_len)
        """
        # Transpose to (n_head, seq_len, head_dim) for batch matmul
        q_t = q.transpose(1, 0, 2) if hasattr(q, 'transpose') else q
        k_t = k.transpose(1, 0, 2) if hasattr(k, 'transpose') else k
        # (n_head, seq_len, seq_len) = (n_head, seq_len, hd) @ (n_head, hd, seq_len)
        return self.device.matmul(q_t, k_t.swapaxes(-1, -2)) / head_dim

    def _attention_apply(self, att, v):
        """Apply attention weights to values using device-aware batch matmul.
        
        Args:
            att: (n_head, seq_len, seq_len) 
            v: (seq_len, n_kv_head or n_head, head_dim)
            
        Returns:
            out: (seq_len, n_head, head_dim)
        """
        v_t = v.transpose(1, 0, 2) if hasattr(v, 'transpose') else v
        # (n_head, seq_len, head_dim) = (n_head, seq_len, seq_len) @ (n_head, seq_len, hd)
        out = self.device.matmul(att, v_t)
        return out.transpose(1, 0, 2) if hasattr(out, 'transpose') else out

    # ─── Forward passes ────────────────────────────────────────────────────

    def forward(self, input_ids: list[int], use_cache: bool = True):
        seq_len = len(input_ids)
        h_ids = np.array(input_ids, dtype=np.int64)
        embed = self._get_persistent('token_embd.weight')

        head_dim = self.n_embd // self.n_head
        n_rep = self.n_head // self.n_kv_head

        # KV cache management
        if use_cache and hasattr(self, '_kv_cache') and self._kv_cache is not None:
            cached_len = self._cached_len
            new_ids = h_ids[cached_len:]
            x = embed[new_ids].astype(np.float32)
            cos, sin = self.precompute_freqs_cis(head_dim, seq_len, self.rope_theta)
            return self._forward_step(x, cos, sin, head_dim, n_rep, cached_len)
        else:
            # Full forward pass (prefill)
            x = embed[h_ids].astype(np.float32)
            cos, sin = self.precompute_freqs_cis(head_dim, seq_len, self.rope_theta)
            mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), 1)
            result = self._forward_full(x, cos, sin, mask, head_dim, n_rep, use_cache)
            return result

    def _forward_full(self, x, cos, sin, mask, head_dim, n_rep, use_cache):
        """Full forward pass (prefill)."""
        seq_len = x.shape[0]
        use_gpu = self.device.device_type != DeviceType.CPU

        if use_cache:
            self._kv_cache = []
            self._cached_len = seq_len

        # Move inputs to device if using GPU
        if use_gpu:
            x = self.device.to_device(x)
            cos = self.device.to_device(cos)
            sin = self.device.to_device(sin)
            mask = self.device.to_device(mask)

        for i in range(self.n_layers):
            ln1 = self._tensor(f'blk.{i}.attn_norm.weight')
            q_w = self._tensor(f'blk.{i}.attn_q.weight')
            k_w = self._tensor(f'blk.{i}.attn_k.weight')
            v_w = self._tensor(f'blk.{i}.attn_v.weight')
            o_w = self._tensor(f'blk.{i}.attn_output.weight')

            r = x
            x = self.rms_norm(x, ln1, self.norm_eps)
            q = self.device.matmul(x, q_w.T) if use_gpu else x @ q_w.T
            k = self.device.matmul(x, k_w.T) if use_gpu else x @ k_w.T
            v = self.device.matmul(x, v_w.T) if use_gpu else x @ v_w.T

            if self.has_bias:
                for qkv_b_name in ['attn_q', 'attn_k', 'attn_v']:
                    b = self._tensor(f'blk.{i}.{qkv_b_name}.bias')
                    if b is not None:
                        if qkv_b_name == 'attn_q':
                            q = q + b
                        elif qkv_b_name == 'attn_k':
                            k = k + b
                        else:
                            v = v + b

            q = q.reshape(seq_len, self.n_head, head_dim)
            k = k.reshape(seq_len, self.n_kv_head, head_dim)
            v = v.reshape(seq_len, self.n_kv_head, head_dim)
            q = self.apply_rope(q, cos, sin)
            k = self.apply_rope(k, cos, sin)

            if use_cache:
                # KV cache: store copies (on device if GPU)
                self._kv_cache.append((k.copy() if hasattr(k, 'copy') else k[:], v.copy() if hasattr(v, 'copy') else v[:]))

            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=1)
                v = np.repeat(v, n_rep, axis=1)

            att = self._attention_scores(q, k, np.sqrt(head_dim))
            att = att + mask  # broadcast: (n_head, seq, seq) + (seq, seq)
            att = self.softmax(att, axis=-1)
            out = self._attention_apply(att, v)
            out = out.reshape(seq_len, self.n_embd)
            out = self.device.matmul(out, o_w.T) if use_gpu else out @ o_w.T
            x = r + out

            ln2 = self._tensor(f'blk.{i}.ffn_norm.weight')
            gate_w = self._tensor(f'blk.{i}.ffn_gate.weight')
            up_w = self._tensor(f'blk.{i}.ffn_up.weight')
            down_w = self._tensor(f'blk.{i}.ffn_down.weight')

            r = x
            x = self.rms_norm(x, ln2, self.norm_eps)
            gate = self.device.matmul(x, gate_w.T) if use_gpu else x @ gate_w.T
            up = self.device.matmul(x, up_w.T) if use_gpu else x @ up_w.T
            x = (self.silu(gate) * up)
            x = self.device.matmul(x, down_w.T) if use_gpu else x @ down_w.T
            x = r + x

            del ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w

        gc.collect()

        norm_w = self._tensor('output_norm.weight')
        x = self.rms_norm(x, norm_w, self.norm_eps)
        lm_w = self._tensor('output.weight')
        if lm_w is None:
            embed = self._get_persistent('token_embd.weight')
            lm_w = embed
        logits = self.device.matmul(x, lm_w.T) if use_gpu else x @ lm_w.T

        # Ensure logits are on CPU for Python consumption
        if use_gpu:
            logits = to_cpu_if_needed(logits)
        return logits

    def _forward_step(self, x, cos, sin, head_dim, n_rep, cached_len):
        """Single-token forward step using KV cache."""
        new_len = x.shape[0]
        new_total = cached_len + new_len
        self._cached_len = new_total
        use_gpu = self.device.device_type != DeviceType.CPU

        # Move inputs to device if using GPU
        if use_gpu:
            x = self.device.to_device(x)

        for i in range(self.n_layers):
            ln1 = self._tensor(f'blk.{i}.attn_norm.weight')
            q_w = self._tensor(f'blk.{i}.attn_q.weight')
            k_w = self._tensor(f'blk.{i}.attn_k.weight')
            v_w = self._tensor(f'blk.{i}.attn_v.weight')
            o_w = self._tensor(f'blk.{i}.attn_output.weight')

            r = x
            x = self.rms_norm(x, ln1, self.norm_eps)
            q = self.device.matmul(x, q_w.T) if use_gpu else x @ q_w.T
            k_new = self.device.matmul(x, k_w.T) if use_gpu else x @ k_w.T
            v_new = self.device.matmul(x, v_w.T) if use_gpu else x @ v_w.T

            if self.has_bias:
                for qkv_b_name in ['attn_q', 'attn_k', 'attn_v']:
                    b = self._tensor(f'blk.{i}.{qkv_b_name}.bias')
                    if b is not None:
                        if qkv_b_name == 'attn_q':
                            q = q + b
                        elif qkv_b_name == 'attn_k':
                            k_new = k_new + b
                        else:
                            v_new = v_new + b

            q = q.reshape(new_len, self.n_head, head_dim)
            k_new = k_new.reshape(new_len, self.n_kv_head, head_dim)
            v_new = v_new.reshape(new_len, self.n_kv_head, head_dim)

            # Apply RoPE to new token positions
            q = self.apply_rope(q, cos[cached_len:new_total], sin[cached_len:new_total])
            k_new = self.apply_rope(k_new, cos[cached_len:new_total], sin[cached_len:new_total])

            # Update KV cache
            k_cached, v_cached = self._kv_cache[i]
            k = np.concatenate([k_cached, k_new], axis=0)
            v = np.concatenate([v_cached, v_new], axis=0)
            self._kv_cache[i] = (k, v)

            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=1)
                v = np.repeat(v, n_rep, axis=1)

            # Attention
            att = self._attention_scores(q, k, np.sqrt(head_dim))
            # Build causal mask for new tokens
            mask = np.full((new_len, new_total), -np.inf, dtype=np.float32)
            for j in range(new_len):
                pos = cached_len + j
                mask[j, :pos+1] = 0.0
            if use_gpu:
                mask = self.device.to_device(mask)
            att = att + mask
            att = self.softmax(att, axis=-1)
            out = self._attention_apply(att, v)
            out = out.reshape(new_len, self.n_embd)
            out = self.device.matmul(out, o_w.T) if use_gpu else out @ o_w.T
            x = r + out

            ln2 = self._tensor(f'blk.{i}.ffn_norm.weight')
            gate_w = self._tensor(f'blk.{i}.ffn_gate.weight')
            up_w = self._tensor(f'blk.{i}.ffn_up.weight')
            down_w = self._tensor(f'blk.{i}.ffn_down.weight')

            r = x
            x = self.rms_norm(x, ln2, self.norm_eps)
            gate = self.device.matmul(x, gate_w.T) if use_gpu else x @ gate_w.T
            up = self.device.matmul(x, up_w.T) if use_gpu else x @ up_w.T
            x = (self.silu(gate) * up)
            x = self.device.matmul(x, down_w.T) if use_gpu else x @ down_w.T
            x = r + x

            del ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w

        gc.collect()

        norm_w = self._tensor('output_norm.weight')
        x = self.rms_norm(x, norm_w, self.norm_eps)
        lm_w = self._tensor('output.weight')
        if lm_w is None:
            embed = self._get_persistent('token_embd.weight')
            lm_w = embed
        logits = self.device.matmul(x, lm_w.T) if use_gpu else x @ lm_w.T

        if use_gpu:
            logits = to_cpu_if_needed(logits)
        return logits

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        ids = self.encode(prompt)
        out = []
        # Reset KV cache
        self._kv_cache = None
        self._cached_len = 0

        for step in range(max_tokens):
            logits = self.forward(ids, use_cache=True)
            logits_np = to_cpu_if_needed(logits)
            nid = int(np.argmax(logits_np[-1]))
            if nid == self.eos_id:
                break
            out.append(nid)
            ids.append(nid)
            if (step + 1) % 10 == 0:
                print(f"  [step {step+1}/{max_tokens}] token={nid}")
                gc.collect()
        return self.decode(out)


def load_model(path: str, device: str = 'auto'):
    """Load a GGUF model with device auto-detection.

    Args:
        path: Path to GGUF file.
        device: Device to use. 'auto' (default) picks best available.
                Options: 'auto', 'cpu', 'intel_arc', 'nvidia'.
                Override via MOJOLLAMA_DEVICE env var.
    """
    return LLMInference(path, device=device)


def list_compute_devices():
    """List all available compute devices."""
    from mojollama.model.device import list_devices
    return list_devices()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        device_arg = sys.argv[4] if len(sys.argv) > 4 else 'auto'
        print(f"Loading model from: {path}")
        print(f"Device: {device_arg}")
        model = load_model(path, device=device_arg)
        prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
        tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        print(f"Testing generate with prompt: '{prompt}' ({tokens} tokens)")
        result = model.generate(prompt, max_tokens=tokens)
        print(f"Generated: {result}")
    else:
        print("Usage: python inference.py <model.gguf> [prompt] [max_tokens] [device]")
        print("  device: auto (default), cpu, intel_arc, nvidia")
        print("  Or set MOJOLLAMA_DEVICE env var")
