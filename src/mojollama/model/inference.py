"""Python inference module for MojoLlama — memory-efficient per-layer processing.
Supports Qwen2 and Llama architectures from GGUF files.
"""
import gc
import numpy as np
import gguf
# pyrefly: ignore [untyped-import]
import regex as re

_gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# Llama 3 regex pattern (tiktoken-style pre-tokenization)
_llama3_pat = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

# GPT-2 byte-to-unicode encoder/decoder
_byte_encoder = {}
for i in range(ord('!'), ord('~') + 1):
    _byte_encoder[i] = chr(i)
for i in range(ord('¡'), ord('¬') + 1):
    _byte_encoder[i] = chr(i)
for i in range(ord('®'), ord('ÿ') + 1):
    _byte_encoder[i] = chr(i)
bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
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


class LLMInference:
    """General LLM inference engine supporting multiple architectures from GGUF files."""

    ARCH_PREFIXES = {
        'qwen2': ['qwen2.', 'qwen2moe.'],
        'llama': ['llama.', 'llama3.', 'llama2.', 'codellama.'],
    }

    def __init__(self, path: str):
        self.reader = gguf.GGUFReader(path)
        self._tensors_by_name = {t.name: t for t in self.reader.tensors}
        self._load_config()
        self._build_tokenizer()

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
        t = self._tensors_by_name.get(gguf_name)
        if t is None:
            return None
        if hasattr(t, 'tensor_type') and t.tensor_type is not None:
            tt = t.tensor_type
            if tt == gguf.GGMLQuantizationType.F32:
                return np.array(t.data, dtype=np.float32)
            elif tt == gguf.GGMLQuantizationType.F16:
                return np.array(t.data, dtype=np.float16).astype(np.float32)
            try:
                return gguf.dequantize(t.data, tt)
            except Exception:
                return np.array(t.data, dtype=np.float32)
        return np.array(t.data, dtype=np.float32)

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

    @staticmethod
    def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(variance + eps) * weight.astype(np.float32)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float):
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(end).astype(np.float32)
        freqs = np.outer(t, freqs)
        return np.cos(freqs), np.sin(freqs)

    @staticmethod
    def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        n, h, d = x.shape
        x2 = x.astype(np.float32).reshape(n, h, d // 2, 2)
        xr = np.stack([-x2[..., 1], x2[..., 0]], axis=-1)
        c = cos[:n, np.newaxis, :d//2, np.newaxis]
        s = sin[:n, np.newaxis, :d//2, np.newaxis]
        return (x2 * c + xr * s).reshape(n, h, d)

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1 + np.exp(-x))

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
        if use_cache:
            self._kv_cache = []
            self._cached_len = seq_len

        for i in range(self.n_layers):
            ln1 = self._tensor(f'blk.{i}.attn_norm.weight')
            q_w = self._tensor(f'blk.{i}.attn_q.weight')
            k_w = self._tensor(f'blk.{i}.attn_k.weight')
            v_w = self._tensor(f'blk.{i}.attn_v.weight')
            o_w = self._tensor(f'blk.{i}.attn_output.weight')

            r = x
            x = self.rms_norm(x, ln1, self.norm_eps)
            q = x @ q_w.T
            k = x @ k_w.T
            v = x @ v_w.T

            if self.has_bias:
                for qkv_b in ['attn_q', 'attn_k', 'attn_v']:
                    b = self._tensor(f'blk.{i}.{qkv_b}.bias')
                    if b is not None:
                        if qkv_b == 'attn_q': q += b
                        elif qkv_b == 'attn_k': k += b
                        else: v += b

            q = q.reshape(seq_len, self.n_head, head_dim)
            k = k.reshape(seq_len, self.n_kv_head, head_dim)
            v = v.reshape(seq_len, self.n_kv_head, head_dim)
            q = self.apply_rope(q, cos, sin)
            k = self.apply_rope(k, cos, sin)

            if use_cache:
                self._kv_cache.append((k.copy(), v.copy()))

            if n_rep > 1:
                k = np.repeat(k, n_rep, axis=1)
                v = np.repeat(v, n_rep, axis=1)

            att = np.einsum('ihd,jhd->hij', q, k) / np.sqrt(head_dim)
            att = att + mask[np.newaxis, :, :]
            am = np.max(att, axis=-1, keepdims=True)
            att = np.exp(att - am)
            att = att / np.sum(att, axis=-1, keepdims=True)
            out = np.einsum('hij,jhd->ihd', att, v).reshape(seq_len, self.n_embd)
            out = out @ o_w.T
            x = r + out

            ln2 = self._tensor(f'blk.{i}.ffn_norm.weight')
            gate_w = self._tensor(f'blk.{i}.ffn_gate.weight')
            up_w = self._tensor(f'blk.{i}.ffn_up.weight')
            down_w = self._tensor(f'blk.{i}.ffn_down.weight')

            r = x
            x = self.rms_norm(x, ln2, self.norm_eps)
            gate = x @ gate_w.T
            up = x @ up_w.T
            x = (self.silu(gate) * up) @ down_w.T
            x = r + x

            del ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w

        gc.collect()

        norm_w = self._tensor('output_norm.weight')
        x = self.rms_norm(x, norm_w, self.norm_eps)
        lm_w = self._tensor('output.weight')
        if lm_w is None:
            embed = self._get_persistent('token_embd.weight')
            lm_w = embed
        logits = x @ lm_w.T
        return logits

    def _forward_step(self, x, cos, sin, head_dim, n_rep, cached_len):
        """Single-token forward step using KV cache."""
        new_len = x.shape[0]
        new_total = cached_len + new_len
        # Update cached length after we process these new tokens
        self._cached_len = new_total

        for i in range(self.n_layers):
            ln1 = self._tensor(f'blk.{i}.attn_norm.weight')
            q_w = self._tensor(f'blk.{i}.attn_q.weight')
            k_w = self._tensor(f'blk.{i}.attn_k.weight')
            v_w = self._tensor(f'blk.{i}.attn_v.weight')
            o_w = self._tensor(f'blk.{i}.attn_output.weight')

            r = x
            x = self.rms_norm(x, ln1, self.norm_eps)
            q = x @ q_w.T
            k_new = x @ k_w.T
            v_new = x @ v_w.T

            if self.has_bias:
                for qkv_b in ['attn_q', 'attn_k', 'attn_v']:
                    b = self._tensor(f'blk.{i}.{qkv_b}.bias')
                    if b is not None:
                        if qkv_b == 'attn_q': q += b
                        elif qkv_b == 'attn_k': k_new += b
                        else: v_new += b

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

            # Causal mask for new tokens only
            att = np.einsum('ihd,jhd->hij', q, k) / np.sqrt(head_dim)
            # Apply causal mask: for each new query position, only attend to cached + itself
            mask = np.full((new_len, new_total), -np.inf, dtype=np.float32)
            for j in range(new_len):
                pos = cached_len + j
                mask[j, :pos+1] = 0.0
            att = att + mask[np.newaxis, :, :]
            am = np.max(att, axis=-1, keepdims=True)
            att = np.exp(att - am)
            att = att / np.sum(att, axis=-1, keepdims=True)
            out = np.einsum('hij,jhd->ihd', att, v).reshape(new_len, self.n_embd)
            out = out @ o_w.T
            x = r + out

            ln2 = self._tensor(f'blk.{i}.ffn_norm.weight')
            gate_w = self._tensor(f'blk.{i}.ffn_gate.weight')
            up_w = self._tensor(f'blk.{i}.ffn_up.weight')
            down_w = self._tensor(f'blk.{i}.ffn_down.weight')

            r = x
            x = self.rms_norm(x, ln2, self.norm_eps)
            gate = x @ gate_w.T
            up = x @ up_w.T
            x = (self.silu(gate) * up) @ down_w.T
            x = r + x

            del ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w

        gc.collect()

        norm_w = self._tensor('output_norm.weight')
        x = self.rms_norm(x, norm_w, self.norm_eps)
        lm_w = self._tensor('output.weight')
        if lm_w is None:
            embed = self._get_persistent('token_embd.weight')
            lm_w = embed
        logits = x @ lm_w.T
        return logits

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        ids = self.encode(prompt)
        out = []
        # Reset KV cache
        self._kv_cache = None
        self._cached_len = 0

        for step in range(max_tokens):
            logits = self.forward(ids, use_cache=True)
            nid = int(np.argmax(logits[-1]))
            if nid == self.eos_id:
                break
            out.append(nid)
            ids.append(nid)
            if (step + 1) % 10 == 0:
                print(f"  [step {step+1}/{max_tokens}] token={nid}")
                gc.collect()
        return self.decode(out)


def load_model(path: str):
    return LLMInference(path)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Loading model from: {path}")
        model = load_model(path)
        prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
        tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        print(f"Testing generate with prompt: '{prompt}' ({tokens} tokens)")
        result = model.generate(prompt, max_tokens=tokens)
        print(f"Generated: {result}")
    else:
        print("Usage: python inference.py <model.gguf> [prompt] [max_tokens]")
