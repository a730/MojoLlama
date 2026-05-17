"""Python inference module for MojoLlama — memory-efficient per-layer processing."""
import gc
import numpy as np
import gguf
# pyrefly: ignore [untyped-import]
import regex as re

_gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
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

GGUF_TO_MODEL_MAP = {
    "token_embd": "embed_tokens",
    "blk": "layers",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "attn_k": "self_attn.k_proj",
    "attn_output": "self_attn.o_proj",
    "output.weight": "lm_head.weight",
    "output_norm": "norm",
}

def _map_name(gguf_name: str) -> str:
    result = gguf_name
    for before, after in GGUF_TO_MODEL_MAP.items():
        result = result.replace(before, after)
    return result

def _get_np(val):
    """Convert gguf scalar to Python value."""
    if hasattr(val, 'parts'):
        p = val.parts[-1]
        return p.item() if p.size == 1 else p
    return val


class Qwen2ForCausalLM:
    def __init__(self, path: str):
        self.reader = gguf.GGUFReader(path)
        self._tensors_by_name = {t.name: t for t in self.reader.tensors}
        self._load_config()
        self._build_tokenizer()

    def _load_config(self):
        f = self.reader.get_field
        def g(key, default=None):
            v = f(key)
            return _get_np(v) if v is not None else default
        # pyrefly: ignore [bad-argument-type]
        self.n_layers = int(g('qwen2.block_count', 0))
        # pyrefly: ignore [bad-argument-type]
        self.n_embd = int(g('qwen2.embedding_length', 0))
        # pyrefly: ignore [bad-argument-type]
        self.n_head = int(g('qwen2.attention.head_count', 0))
        # pyrefly: ignore [bad-argument-type]
        self.n_kv_head = int(g('qwen2.attention.head_count_kv', 0))
        # pyrefly: ignore [bad-argument-type]
        self.n_ff = int(g('qwen2.feed_forward_length', 0))
        # pyrefly: ignore [bad-argument-type]
        self.max_seq_len = int(g('qwen2.context_length', 32768))
        # pyrefly: ignore [bad-argument-type]
        self.norm_eps = float(g('qwen2.attention.layer_norm_rms_epsilon',
                                g('qwen2.attn_layer_norm_rms_epsilon', 1e-6)))
        base = g('qwen2.rope.freq_base', None)
        self.rope_theta = float(base) if base is not None else 1000000.0

    def _get_tensor(self, gguf_name: str) -> np.ndarray:
        t = self._tensors_by_name.get(gguf_name)
        if t is None:
            # pyrefly: ignore [bad-return]
            return None
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            return np.array(t.data, dtype=np.float32)
        return gguf.dequantize(t.data, t.tensor_type)

    def _build_tokenizer(self):
        fields = self.reader.fields
        token_field = fields.get('tokenizer.ggml.tokens')
        merge_field = fields.get('tokenizer.ggml.merges')
        self.vocab = {}
        if token_field:
            parts = token_field.parts
            str_parts = [p for p in parts
                         if isinstance(p, np.ndarray) and p.dtype == np.uint8 and p.size > 1]
            for i, p in enumerate(str_parts):
                s = p.tobytes().decode('utf-8', errors='replace')
                self.vocab[s] = i
        self.merges = []
        if merge_field:
            parts = merge_field.parts
            str_parts = [p for p in parts
                         if isinstance(p, np.ndarray) and p.dtype == np.uint8 and p.size > 1]
            for p in str_parts:
                s = p.tobytes().decode('utf-8', errors='replace')
                sp = s.split(' ')
                if len(sp) == 2:
                    self.merges.append(tuple(sp))
        self.merge_ranks = {p: i for i, p in enumerate(self.merges)}
        bos = fields.get('tokenizer.ggml.bos_token_id')
        self.bos_id = int(_get_np(bos)) if bos else 151643
        eos = fields.get('tokenizer.ggml.eos_token_id')
        self.eos_id = int(_get_np(eos)) if eos else 151645
        add = fields.get('tokenizer.ggml.add_bos_token')
        self.add_bos = bool(int(_get_np(add))) if add else False

    def _tensor(self, gguf_name: str) -> np.ndarray:
        """Get weight tensor by GGUF name, dequantizing if needed."""
        t = self._tensors_by_name.get(gguf_name)
        if t is None:
            # pyrefly: ignore [bad-return]
            return None
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            return np.array(t.data, dtype=np.float32)
        return gguf.dequantize(t.data, t.tensor_type)

    def _get_persistent(self, name: str) -> np.ndarray:
        """Get a weight that persists across forward calls (embed, lm_head)."""
        if not hasattr(self, '_persistent_cache'):
            self._persistent_cache = {}
        if name not in self._persistent_cache:
            self._persistent_cache[name] = self._tensor(name)
        return self._persistent_cache[name]

    def encode(self, text: str) -> list[int]:
        ids = self._encode_bpe(text)
        if self.add_bos:
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        rev = {v: k for k, v in self.vocab.items()}
        chunks = []
        for tid in ids:
            s = rev.get(tid, '')
            if isinstance(s, str):
                chunks.append(s)
        raw = ''.join(chunks)
        try:
            return _unicode_to_bytes(raw).decode('utf-8', errors='replace')
        except:
            return raw

    def _encode_bpe(self, text: str) -> list[int]:
        import ftfy
        text = ftfy.fix_text(text)
        tokens = []
        for m in re.findall(_gpt2_pat, text):
            word = _bytes_to_unicode(m.encode('utf-8'))
            pairs = list(word)
            while True:
                best = None
                rank = None
                for i in range(len(pairs) - 1):
                    r = self.merge_ranks.get((pairs[i], pairs[i+1]))
                    if r is not None and (rank is None or r < rank):
                        best = i
                        rank = r
                if best is None:
                    break
                pairs = pairs[:best] + [pairs[best] + pairs[best+1]] + pairs[best+2:]
            for token in pairs:
                tokens.append(self.vocab.get(token, 0))
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

    def forward(self, input_ids: list[int]) -> np.ndarray:
        seq_len = len(input_ids)
        h_ids = np.array(input_ids, dtype=np.int64)

        embed = self._get_persistent('token_embd.weight')
        x = embed[h_ids].astype(np.float32)

        head_dim = self.n_embd // self.n_head
        cos, sin = self.precompute_freqs_cis(head_dim, seq_len, self.rope_theta)
        n_rep = self.n_head // self.n_kv_head
        mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), 1)

        for i in range(self.n_layers):
            ln1 = self._tensor(f'blk.{i}.attn_norm.weight')
            q_w = self._tensor(f'blk.{i}.attn_q.weight')
            k_w = self._tensor(f'blk.{i}.attn_k.weight')
            v_w = self._tensor(f'blk.{i}.attn_v.weight')
            o_w = self._tensor(f'blk.{i}.attn_output.weight')
            q_b = self._tensor(f'blk.{i}.attn_q.bias')
            k_b = self._tensor(f'blk.{i}.attn_k.bias')
            v_b = self._tensor(f'blk.{i}.attn_v.bias')
            ln2 = self._tensor(f'blk.{i}.ffn_norm.weight')
            gate_w = self._tensor(f'blk.{i}.ffn_gate.weight')
            up_w = self._tensor(f'blk.{i}.ffn_up.weight')
            down_w = self._tensor(f'blk.{i}.ffn_down.weight')

            r = x
            x = self.rms_norm(x, ln1, self.norm_eps)
            q = x @ q_w.T
            k = x @ k_w.T
            v = x @ v_w.T
            if q_b is not None: q += q_b
            if k_b is not None: k += k_b
            if v_b is not None: v += v_b
            q = q.reshape(seq_len, self.n_head, head_dim)
            k = k.reshape(seq_len, self.n_kv_head, head_dim)
            v = v.reshape(seq_len, self.n_kv_head, head_dim)
            q = self.apply_rope(q, cos, sin)
            k = self.apply_rope(k, cos, sin)
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

            r = x
            x = self.rms_norm(x, ln2, self.norm_eps)
            gate = x @ gate_w.T
            up = x @ up_w.T
            x = (self.silu(gate) * up) @ down_w.T
            x = r + x

            del ln1, q_w, k_w, v_w, o_w, q_b, k_b, v_b, ln2, gate_w, up_w, down_w

        norm_w = self._tensor('output_norm.weight')
        x = self.rms_norm(x, norm_w, self.norm_eps)

        lm_w = self._tensor('output.weight')
        if lm_w is None:
            lm_w = embed
        logits = x @ lm_w.T
        return logits

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        ids = self.encode(prompt)
        out = []
        for _ in range(max_tokens):
            logits = self.forward(ids)
            nid = int(np.argmax(logits[-1]))
            if nid == self.eos_id:
                break
            out.append(nid)
            ids.append(nid)
        return self.decode(out)


def load_model(path: str):
    return Qwen2ForCausalLM(path)
