#!/usr/bin/env python3
"""Pure numpy inference on Qwen3-30B-A3B for correctness validation.
Uses gguf.dequantize for all weights. Slow but mathematically exact."""
import sys, os, time, numpy as np
os.environ['OMP_NUM_THREADS'] = '32'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
import gguf
from gguf.constants import GGMLQuantizationType as QT
from transformers import AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else '/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
TOKENIZER_PATH = '/tmp/qwen3-tokenizer/'

t0 = time.perf_counter()
reader = gguf.GGUFReader(MODEL)
r = reader
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Architecture
arch = 'qwen3moe'
def get_field(name):
    for k, v in r.fields.items():
        if k == name:
            p = v.parts if hasattr(v, 'parts') else []
            if len(p) >= 1:
                d = p[-1]
                if hasattr(d, '__iter__'):
                    return int(d) if len(d) == 1 else float(d[0]) if hasattr(d[0], '__float__') else d
                return int(d) if hasattr(d, '__int__') else float(d)
    return None

L = int(get_field(f'{arch}.block_count'))
N = int(get_field(f'{arch}.embedding_length'))
FF = int(get_field(f'{arch}.feed_forward_length'))
NH = int(get_field(f'{arch}.attention.head_count'))
NKH = int(get_field(f'{arch}.attention.head_count_kv'))
HD = int(get_field(f'{arch}.attention.key_length') or (N // NH))
NE = int(get_field(f'{arch}.expert_count'))
NK = int(get_field(f'{arch}.expert_used_count'))
M = int(get_field(f'{arch}.expert_feed_forward_length'))
eps = float(get_field(f'{arch}.attention.layer_norm_rms_epsilon'))
freq = float(get_field(f'{arch}.rope.freq_base'))
V = int(get_field('general.vocab_size') or 0)
print(f'{L}L/{N}D/{FF}FF/{NH}H/{NKH}KV HD={HD} MoE {NE}x{NK} M={M} V={V} eps={eps}', flush=True)

def get_t(name):
    for t in r.tensors:
        if t.name == name: return t
    return None

def load_f32(name, shape2d=None):
    """Load and dequantize a 2D tensor to FP32."""
    t = get_t(name)
    if t is None: return None
    f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
    if len(t.shape) == 2:
        in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
        f32 = f32.reshape(out_dim, in_dim)
    return np.ascontiguousarray(f32)

def load_f32_3d(name):
    """Load and dequantize a 3D tensor (MoE experts)."""
    t = get_t(name)
    if t is None: return None
    f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
    if len(t.shape) == 3:
        in_dim, out_dim, n_exp = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
        f32 = f32.reshape(n_exp, out_dim, in_dim)
    return np.ascontiguousarray(f32)

# Load embedding
emb_t = get_t('token_embd.weight')
emb = gguf.dequantize(emb_t.data, emb_t.tensor_type).astype(np.float32)
emb = emb.reshape(int(emb_t.shape[1]), int(emb_t.shape[0]))
print(f'Embedding: {emb.shape}', flush=True)

# Load weights per layer
print('Loading weights...', flush=True)
Wq = [load_f32(f'blk.{i}.attn_q.weight') for i in range(L)]
Wk = [load_f32(f'blk.{i}.attn_k.weight') for i in range(L)]
Wv = [load_f32(f'blk.{i}.attn_v.weight') for i in range(L)]
Wo = [load_f32(f'blk.{i}.attn_output.weight') for i in range(L)]
Wrouter = [load_f32(f'blk.{i}.ffn_gate_inp.weight') for i in range(L)]
Wgate = [load_f32_3d(f'blk.{i}.ffn_gate_exps.weight') for i in range(L)]
Wup = [load_f32_3d(f'blk.{i}.ffn_up_exps.weight') for i in range(L)]
Wdown = [load_f32_3d(f'blk.{i}.ffn_down_exps.weight') for i in range(L)]
AN = [load_f32(f'blk.{i}.attn_norm.weight') for i in range(L)]
FN = [load_f32(f'blk.{i}.ffn_norm.weight') for i in range(L)]
onw = load_f32('output_norm.weight')
outw = load_f32('output.weight')
print(f'Loaded in {time.perf_counter()-t0:.1f}s', flush=True)

def rms_norm(x, w, e=eps):
    return x * (1.0 / np.sqrt(np.mean(x*x) + e)) * w

def rope(x, pos, hd=HD):
    """Apply RoPE to Q/K vectors (x has shape (n_heads*hd,))."""
    h2 = hd // 2
    freqs = freq ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
    cos = np.cos(pos / freqs).astype(np.float32)
    sin = np.sin(pos / freqs).astype(np.float32)
    xr = x.reshape(-1, hd)
    out = xr.copy()
    out[:, :h2] = xr[:, :h2] * cos - xr[:, h2:] * sin
    out[:, h2:] = xr[:, h2:] * cos + xr[:, :h2] * sin
    return out.reshape(-1)

def gqa(q, k, v, nh=NH, nkh=NKH, hd=HD):
    """Grouped-query attention."""
    gr = nh // nkh  # heads per KV group
    ks = nkh * hd
    q = q.reshape(nh, hd)
    k = k.reshape(nkh, hd)
    v = v.reshape(nkh, hd)
    out = np.zeros(nh * hd, dtype=np.float32)
    for h in range(nh):
        kv = h // gr
        scores = np.dot(q[h], k[kv]) / np.sqrt(hd)
        w = np.exp(scores - np.max(scores))
        w = w / np.sum(w + 1e-10)
        out[h*hd:(h+1)*hd] = w * v[kv]
    return out

def silu(x):
    return x / (1.0 + np.exp(-x))

def forward(token_id):
    """Pure numpy forward pass for one token."""
    x = emb[token_id].copy()
    kv_cache_k = np.zeros((L, 4096, NKH * HD), dtype=np.float32)  # dummy
    kv_cache_v = np.zeros((L, 4096, NKH * HD), dtype=np.float32)
    
    for l in range(L):
        # Attention
        res = x.copy()
        xn = rms_norm(x, AN[l])
        q = Wq[l] @ xn
        k = Wk[l] @ xn
        v = Wv[l] @ xn
        
        pos = l  # simplified: position 0 for first token then l
        q = rope(q, 0, HD)
        k = rope(k, 0, HD)
        
        # Store KV (simplified: always at position 0)
        kv_cache_k[l, 0] = k
        kv_cache_v[l, 0] = v
        
        # Attention
        sl = 1
        kct = kv_cache_k[l, :sl].reshape(sl * NKH * HD)
        vct = kv_cache_v[l, :sl].reshape(sl * NKH * HD)
        att = gqa(q, kct, vct)
        
        x = res + Wo[l] @ att
        
        # MoE FFN
        res2 = x.copy()
        xn2 = rms_norm(x, FN[l])
        
        router = Wrouter[l] @ xn2  # [NE]
        scores = np.exp(router - np.max(router))
        scores = scores / np.sum(scores + 1e-10)
        topk = np.argsort(scores)[-NK:][::-1]
        
        ffn_out = np.zeros(N, dtype=np.float32)
        for e_idx in topk:
            w = scores[e_idx]
            gate = Wgate[l][e_idx] @ xn2  # [M]
            up = Wup[l][e_idx] @ xn2     # [M]
            act = silu(gate) * up         # [M]
            down = Wdown[l][e_idx] @ act  # [N]
            ffn_out += w * down
        
        x = res2 + ffn_out
    
    # Final norm + output
    xn = rms_norm(x, onw)
    logits = outw @ xn
    
    # Check for NaN
    nan_count = np.isnan(logits).sum()
    return logits, nan_count

# Test
print('\n--- Pure numpy forward ---', flush=True)
t0 = time.perf_counter()
logits, nan_count = forward(785)
t1 = time.perf_counter()
print(f'Time: {(t1-t0)*1000:.1f}ms', flush=True)
print(f'NaN: {nan_count}/{V}', flush=True)
if nan_count == 0:
    top5 = np.argsort(logits)[-5:][::-1]
    print(f'Top-5 tokens: {[tokenizer.decode([t]) for t in top5]}', flush=True)
    print(f'Logits: max={logits.max():.2f} min={logits.min():.2f}', flush=True)
