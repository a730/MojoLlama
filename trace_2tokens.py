"""Trace forward pass with 2 tokens."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from mojollama.model.inference import load_model

model = load_model('test_model.gguf')

# Use 2 tokens
ids = [model.bos_id, model.bos_id]
seq_len = len(ids)
h_ids = np.array(ids, dtype=np.int64)

embed = model._get_persistent('token_embd.weight')
x = embed[h_ids].astype(np.float32)
print(f'embed: {x.shape}')

head_dim = model.n_embd // model.n_head
cos, sin = model.precompute_freqs_cis(head_dim, seq_len, model.rope_theta)
n_rep = model.n_head // model.n_kv_head
mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), 1)

t_start = time.time()
for i in range(model.n_layers):
    t0 = time.time()
    ln1 = model._tensor(f'blk.{i}.attn_norm.weight')
    q_w = model._tensor(f'blk.{i}.attn_q.weight')
    k_w = model._tensor(f'blk.{i}.attn_k.weight')
    v_w = model._tensor(f'blk.{i}.attn_v.weight')
    o_w = model._tensor(f'blk.{i}.attn_output.weight')
    q_b = model._tensor(f'blk.{i}.attn_q.bias')
    k_b = model._tensor(f'blk.{i}.attn_k.bias')
    v_b = model._tensor(f'blk.{i}.attn_v.bias')
    ln2 = model._tensor(f'blk.{i}.ffn_norm.weight')
    gate_w = model._tensor(f'blk.{i}.ffn_gate.weight')
    up_w = model._tensor(f'blk.{i}.ffn_up.weight')
    down_w = model._tensor(f'blk.{i}.ffn_down.weight')
    load_time = time.time() - t0

    r = x
    x = model.rms_norm(x, ln1, model.norm_eps)
    q = x @ q_w.T; k = x @ k_w.T; v = x @ v_w.T
    if q_b is not None: q += q_b
    if k_b is not None: k += k_b
    if v_b is not None: v += v_b
    q = q.reshape(seq_len, model.n_head, head_dim)
    k = k.reshape(seq_len, model.n_kv_head, head_dim)
    v = v.reshape(seq_len, model.n_kv_head, head_dim)
    q = model.apply_rope(q, cos, sin)
    k = model.apply_rope(k, cos, sin)
    if n_rep > 1: k = np.repeat(k, n_rep, axis=1); v = np.repeat(v, n_rep, axis=1)
    att = np.einsum('ihd,jhd->hij', q, k) / np.sqrt(head_dim)
    att = att + mask[np.newaxis, :, :]
    am = np.max(att, axis=-1, keepdims=True)
    att = np.exp(att - am)
    att = att / np.sum(att, axis=-1, keepdims=True)
    out = np.einsum('hij,jhd->ihd', att, v).reshape(seq_len, model.n_embd)
    out = out @ o_w.T; x = r + out
    r = x
    x = model.rms_norm(x, ln2, model.norm_eps)
    gate = x @ gate_w.T; up = x @ up_w.T
    x = (model.silu(gate) * up) @ down_w.T; x = r + x
    del ln1, q_w, k_w, v_w, o_w, q_b, k_b, v_b, ln2, gate_w, up_w, down_w
    
    elapsed = time.time() - t_start
    if i < 3 or i % 5 == 0:
        print(f'layer {i}: {elapsed:.3f}s (load={load_time:.3f}s)')

norm_w = model._tensor('output_norm.weight')
x = model.rms_norm(x, norm_w, model.norm_eps)
lm_w = model._get_persistent('output.weight')
logits = x @ lm_w.T
print(f'Done! {time.time()-t_start:.1f}s logits={logits.shape}')
