"""Trace each layer to find the slow one."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from mojollama.model.inference import load_model

logf = open('/tmp/opencode/trace2.log', 'w')
def log(msg):
    print(msg, flush=True)
    logf.write(msg + '\n')
    logf.flush()

model = load_model('test_model.gguf')
ids = model.encode('Hello')
h_ids = np.array(ids, dtype=np.int64)

embed = model._get_persistent('token_embd.weight')
x = embed[h_ids].astype(np.float32)

head_dim = model.n_embd // model.n_head
cos, sin = model.precompute_freqs_cis(head_dim, 1, model.rope_theta)
mask = np.triu(np.full((1, 1), -np.inf, dtype=np.float32), 1)
n_rep = model.n_head // model.n_kv_head

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
    q = x @ q_w.T
    k = x @ k_w.T
    v = x @ v_w.T
    if q_b is not None: q += q_b
    if k_b is not None: k += k_b
    if v_b is not None: v += v_b
    q = q.reshape(1, model.n_head, head_dim)
    k = k.reshape(1, model.n_kv_head, head_dim)
    v = v.reshape(1, model.n_kv_head, head_dim)
    q = model.apply_rope(q, cos, sin)
    k = model.apply_rope(k, cos, sin)
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=1)
        v = np.repeat(v, n_rep, axis=1)
    att = np.einsum('ihd,jhd->hij', q, k) / np.sqrt(head_dim)
    att = att + mask[np.newaxis, :, :]
    am = np.max(att, axis=-1, keepdims=True)
    att = np.exp(att - am)
    att = att / np.sum(att, axis=-1, keepdims=True)
    out = np.einsum('hij,jhd->ihd', att, v).reshape(1, model.n_embd)
    out = out @ o_w.T
    x = r + out
    r = x
    x = model.rms_norm(x, ln2, model.norm_eps)
    gate = x @ gate_w.T
    up = x @ up_w.T
    x = (model.silu(gate) * up) @ down_w.T
    x = r + x

    del ln1, q_w, k_w, v_w, o_w, q_b, k_b, v_b, ln2, gate_w, up_w, down_w
    elapsed = time.time() - t_start
    log(f'layer {i}: {elapsed:.3f}s (load={load_time:.3f}s)')

norm_w = model._tensor('output_norm.weight')
t0 = time.time()
x = model.rms_norm(x, norm_w, model.norm_eps)
log(f'norm: {time.time()-t0:.3f}s')

t0 = time.time()
lm_w = model._get_persistent('output.weight')
log(f'lm_head load: {time.time()-t0:.3f}s')

t0 = time.time()
logits = x @ lm_w.T
log(f'lm_head matmul: {time.time()-t0:.3f}s')
log(f'Done! total={time.time()-t_start:.1f}s')

logf.close()
