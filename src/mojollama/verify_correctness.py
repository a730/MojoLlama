#!/usr/bin/env python3
"""Ultimate fix: Fully numpy reference for Qwen3-30B-A3B correctness.
Dequantizes ALL weights to FP32 once, then runs pure numpy inference."""
import sys, os, time, numpy as np
os.environ['OMP_NUM_THREADS'] = '32'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
import gguf
from gguf.constants import GGMLQuantizationType as QT
from transformers import AutoTokenizer

MODEL = '/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
TOKENIZER_PATH = '/tmp/qwen3-tokenizer/'

t0 = time.perf_counter()
reader = gguf.GGUFReader(MODEL)
r = reader

def get_field(name):
    for k, v in r.fields.items():
        if k == name:
            p = v.parts
            if len(p) >= 1:
                d = p[-1]
                if hasattr(d, '__len__') and len(d) == 1:
                    return float(d[0])
                return float(d)
    return None

arch = 'qwen3moe'
L = int(get_field(f'{arch}.block_count'))
N = int(get_field(f'{arch}.embedding_length'))
NH = int(get_field(f'{arch}.attention.head_count'))
NKH = int(get_field(f'{arch}.attention.head_count_kv'))
HD = int(get_field(f'{arch}.attention.key_length') or (N // NH))
NE = int(get_field(f'{arch}.expert_count'))
NK = int(get_field(f'{arch}.expert_used_count'))
M = int(get_field(f'{arch}.expert_feed_forward_length'))
eps = float(get_field(f'{arch}.attention.layer_norm_rms_epsilon'))
freq = float(get_field(f'{arch}.rope.freq_base'))
V = int(get_field('general.vocab_size'))

print(f'{L}L/{N}D/{NH}H/{NKH}KV HD={HD} MoE {NE}x{NK} M={M} V={V}', flush=True)

# Find all tensors
tensor_map = {t.name: t for t in r.tensors}
print(f'Total tensors: {len(tensor_map)}', flush=True)

def load_f32(name):
    if name not in tensor_map: return None
    t = tensor_map[name]
    qt = t.tensor_type
    if qt == QT.F32:
        f32 = t.data.reshape(t.shape).astype(np.float32)
    else:
        f32 = gguf.dequantize(t.data, qt).astype(np.float32)
    # GGUF stores as (width, height) → convert to (height, width) for matmul
    if len(t.shape) == 2:
        f32 = f32.reshape(int(t.shape[1]), int(t.shape[0]))
    return np.ascontiguousarray(f32)

def load_f32_3d(name):
    """3D tensor: (in_dim, out_dim, n_experts) → (n_experts, out_dim, in_dim)"""
    if name not in tensor_map: return None
    t = tensor_map[name]
    qt = t.tensor_type
    if qt == QT.F32:
        f32 = t.data.reshape(t.shape).astype(np.float32)
    else:
        f32 = gguf.dequantize(t.data, qt).astype(np.float32)
    s = t.shape  # [in_dim, out_dim, n_exp] GGUF
    f32 = f32.reshape(int(s[2]), int(s[1]), int(s[0]))
    return np.ascontiguousarray(f32)

# Load everything
print('Loading weights...', flush=True)
emb = load_f32('token_embd.weight')
print(f'  emb: {emb.shape}', flush=True)

AN = [load_f32(f'blk.{i}.attn_norm.weight') for i in range(L)]
FN = [load_f32(f'blk.{i}.ffn_norm.weight') for i in range(L)]
Wq = [load_f32(f'blk.{i}.attn_q.weight') for i in range(L)]
Wk = [load_f32(f'blk.{i}.attn_k.weight') for i in range(L)]
Wv = [load_f32(f'blk.{i}.attn_v.weight') for i in range(L)]
Wo = [load_f32(f'blk.{i}.attn_output.weight') for i in range(L)]
print(f'  Attention weights loaded', flush=True)

Wrouter = [load_f32(f'blk.{i}.ffn_gate_inp.weight') for i in range(L)]
print(f'  Router weights loaded', flush=True)

# Dequantize expert weights (biggest load)
t1 = time.perf_counter()
Wgate = [load_f32_3d(f'blk.{i}.ffn_gate_exps.weight') for i in range(L)]
Wup = [load_f32_3d(f'blk.{i}.ffn_up_exps.weight') for i in range(L)]
Wdown = [load_f32_3d(f'blk.{i}.ffn_down_exps.weight') for i in range(L)]
print(f'  Expert weights loaded ({time.perf_counter()-t1:.0f}s)', flush=True)

onw = load_f32('output_norm.weight')
outw = load_f32('output.weight')
print(f'Total load time: {time.perf_counter()-t0:.0f}s', flush=True)

# Pre-compute RoPE tables
hd2 = HD // 2
freqs = freq ** (np.arange(0, HD, 2, dtype=np.float64) / HD)
cos_table = {pos: np.cos(pos / freqs).astype(np.float32) for pos in range(4096)}
sin_table = {pos: np.sin(pos / freqs).astype(np.float32) for pos in range(4096)}

def rope(x, pos):
    x = x.reshape(-1, HD)
    c = cos_table[pos]; s = sin_table[pos]
    out = np.empty_like(x)
    out[:, :hd2] = x[:, :hd2] * c - x[:, hd2:] * s
    out[:, hd2:] = x[:, hd2:] * c + x[:, :hd2] * s
    return out.reshape(-1)

def rms_norm(x, w):
    return x * (1.0 / np.sqrt(np.mean(x*x) + eps)) * w

def gqa(q, k_ctx, v_ctx, sl=1):
    """GQA: q shape (NH*HD,), k_ctx (sl*NKH*HD,), v_ctx (sl*NKH*HD,)"""
    gr = NH // NKH
    q = q.reshape(NH, HD, 1)  # [NH, HD, 1]
    k = k_ctx.reshape(sl, NKH, HD)  # [sl, NKH, HD]
    v = v_ctx.reshape(sl, NKH, HD)  # [sl, NKH, HD]
    
    out = np.zeros(NH * HD, dtype=np.float32)
    for h in range(NH):
        kv = h // gr
        scores = np.sum(q[h] * k[:, kv], axis=1) / np.sqrt(HD)  # [sl]
        soft = np.exp(scores - scores.max())
        soft /= soft.sum() + 1e-10
        out[h*HD:(h+1)*HD] = np.sum(soft[:, None] * v[:, kv], axis=0)
    return out

def forward_token(token):
    """Single token forward pass."""
    x = emb[token].copy()
    kv_k = np.zeros((L, NKH * HD), dtype=np.float32)
    kv_v = np.zeros((L, NKH * HD), dtype=np.float32)
    
    for l in range(L):
        # Attention
        res = x.copy()
        xn = rms_norm(x, AN[l])
        
        q = Wq[l] @ xn
        k = Wk[l] @ xn
        v = Wv[l] @ xn
        
        q = rope(q, 0)
        k = rope(k, 0)
        
        kv_k[l] = k
        kv_v[l] = v
        
        # GQA with all previous KV
        sl = l + 1
        kc = kv_k[:sl].reshape(sl * NKH * HD)
        vc = kv_v[:sl].reshape(sl * NKH * HD)
        att = gqa(q, kc, vc, sl)
        
        x = res + Wo[l] @ att
        
        # MoE FFN
        res2 = x.copy()
        xn2 = rms_norm(x, FN[l])
        
        router_scores = Wrouter[l] @ xn2
        router_soft = np.exp(router_scores - router_scores.max())
        router_soft /= router_soft.sum() + 1e-10
        
        topk = np.argsort(router_soft)[-NK:][::-1]
        topk_scores = router_soft[topk]
        
        ffn_out = np.zeros(N, dtype=np.float32)
        for idx, e_idx in enumerate(topk):
            w = topk_scores[idx]
            gate = Wgate[l][e_idx] @ xn2
            up = Wup[l][e_idx] @ xn2
            act = (gate / (1.0 + np.exp(-gate))) * up
            down = Wdown[l][e_idx] @ act
            ffn_out += w * down
        
        x = res2 + ffn_out
        
        # NaN check
        nan_c = np.isnan(x).sum()
        if nan_c > 0:
            print(f'  NaN at layer {l}: {nan_c}/{N}', flush=True)
            # Replace NaN with 0 to continue
            x = np.nan_to_num(x, nan=0.0)
    
    xn = rms_norm(x, onw)
    logits = outw @ xn
    return logits

# Test
print('\n--- Pure numpy inference ---', flush=True)
for token_id in [785, 12345, 67890]:
    t0 = time.perf_counter()
    logits = forward_token(token_id)
    t1 = time.perf_counter()
    nan_c = np.isnan(logits).sum()
    top5 = np.argsort(logits)[-5:][::-1]
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    top5_str = [tok.decode([t]) for t in top5]
    print(f'Token {token_id}: {nan_c}/{V} NaN, {(t1-t0)*1000:.1f}ms', flush=True)
    print(f'  Top-5: {top5_str}', flush=True)
    print(f'  Logits range: [{logits.min():.2f}, {logits.max():.2f}]', flush=True)
