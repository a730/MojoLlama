#!/usr/bin/env python3
"""Convert ZAYA-8B MXFP4 (HF format) → llama.cpp naming for MojoLlama engine."""
import sys, os, gguf, numpy as np
from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType as QT

SRC = '/onedev-workspace/work/models/ZAYA1-8B-MXFP4.gguf'
DST = '/onedev-workspace/work/models/ZAYA1-8B-MXFP4-converted.gguf'

reader = GGUFReader(SRC)
fields = reader.fields
def gm(k, d=None):
    v = fields.get(k)
    if v is None: return d
    p = getattr(v, 'parts', None)
    if p:
        d = p[-1]
        if hasattr(d, '__iter__'):
            vals = list(d)
            return vals[0] if len(vals) == 1 else vals
        return d
    return d

n_layer, n_exp = int(gm('zaya.block_count',80)), int(gm('zaya.expert_count',16))
n_embd = int(gm('zaya.embedding_length',2048))
n_head, n_kv_head = int(gm('zaya.attention.head_count',8)), int(gm('zaya.attention.head_count_kv',2))
vocab = int(gm('zaya.vocab_size',262272))
print(f'{n_layer}L {n_exp}E {n_embd}D {n_head}H')

tensors = {t.name: t for t in reader.tensors}
has = lambda n: n in tensors
t = lambda n: tensors[n]

writer = GGUFWriter(DST, 'zaya')
writer.add_block_count(n_layer); writer.add_embedding_length(n_embd)
writer.add_feed_forward_length(4096); writer.add_context_length(131072)
writer.add_head_count(n_head); writer.add_head_count_kv(n_kv_head)
writer.add_layer_norm_rms_eps(1e-5); writer.add_expert_count(n_exp)
writer.add_expert_used_count(1); writer.add_file_type(39)
writer.add_name('ZAYA1-8B-MXFP4-converted')
writer.add_vocab_size(vocab); writer.add_rope_dimension_count(64)
writer.add_rope_freq_base(5000000.0); writer.add_quantization_version(1)

def _raw_shape(tsrc):
    """Compute byte-level shape for a tensor."""
    log = [int(s) for s in tsrc.shape]
    qt = tsrc.tensor_type
    if qt == 0:  # F32
        return log
    elif qt == 1:  # F16
        return log[:-1] + [log[-1] * 2]
    else:  # Quantized — quantize the second-to-last dim (columns)
        blk_sz = 32
        blk_bytes = {39: 17}.get(qt, 34)
        # For 2D: [rows, quantized_cols]
        # For 3D (packed experts): [rows, quantized_cols, n_experts]
        n_blk = (log[-2] + blk_sz - 1) // blk_sz if len(log) >= 2 else (log[-1] + blk_sz - 1) // blk_sz
        if len(log) >= 3:
            return log[:-2] + [n_blk * blk_bytes] + log[-1:]
        else:
            return log[:-1] + [n_blk * blk_bytes]

def add(name, tsrc):
    log_shape = [int(s) for s in tsrc.shape]
    qt = QT(tsrc.tensor_type)
    raw = tsrc.data.tobytes() if hasattr(tsrc.data,'tobytes') else bytes(tsrc.data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    
    # Handle edge case: quantized tensors with last dim < block size
    if qt not in (QT.F32, QT.F16) and len(log_shape) >= 2 and log_shape[-1] < 32:
        f32 = gguf.dequantize(tsrc.data, tsrc.tensor_type).astype(np.float32)
        if len(log_shape) == 2:
            f32 = f32.reshape(log_shape)
        add_f32(name, f32)
        return
    
    writer.add_tensor(name, arr, raw_shape=_raw_shape(tsrc), raw_dtype=qt)

def add_f32(name, d32):
    # Preserve 2D shape when adding
    if len(d32.shape) >= 2:
        writer.add_tensor(name, d32.astype(np.float32), raw_dtype=QT(0))
    else:
        writer.add_tensor(name, d32.astype(np.float32), raw_dtype=QT(0))

# Global
for s,d in [('model.embed_tokens.weight','token_embd.weight'),('model.final_norm.weight','output_norm.weight')]:
    if has(s): add(d, t(s)); print(f'  {s} → {d}')
for sn in ['hidden_states_scale','hidden_states_bias','residual_scale','residual_bias']:
    k = f'model.res_scale.{sn}'
    if has(k):
        d32 = gguf.dequantize(t(k).data, t(k).tensor_type).astype(np.float32).ravel()
        pr = 'res_scale_hs' if 'hidden' in sn else 'res_scale_res'
        sf = 'weight' if 'scale' in sn else 'bias'
        add_f32(f'{pr}.{sf}', d32)

for l in range(n_layer):
    ps, pd = f'model.layers.{l}', f'blk.{l}'
    even = l % 2 == 0
    if has(f'{ps}.input_norm.weight'):
        add(f'{pd}.attn_norm.weight', t(f'{ps}.input_norm.weight'))
    
    if even:
        for src_s, dst_s in [('self_attn.qkv.linear_q','attn_q'),('self_attn.qkv.linear_k','attn_k'),
                              ('self_attn.o_proj','attn_output')]:
            if has(f'{ps}.{src_s}.weight'): add(f'{pd}.{dst_s}.weight', t(f'{ps}.{src_s}.weight'))
        for i in [1,2]:
            if has(f'{ps}.self_attn.qkv.val_proj{i}.weight'):
                add(f'{pd}.cca_val_proj{i}.weight', t(f'{ps}.self_attn.qkv.val_proj{i}.weight'))
        for sf in ['weight','bias']:
            if has(f'{ps}.self_attn.qkv.conv_qk.0.{sf}'):
                add(f'{pd}.ssm_conv1d.{sf}', t(f'{ps}.self_attn.qkv.conv_qk.0.{sf}'))
        for sn in ['hidden_states_scale','hidden_states_bias']:
            k = f'{ps}.res_scale.{sn}'
            if has(k):
                d32 = gguf.dequantize(t(k).data, t(k).tensor_type).astype(np.float32).ravel()
                add_f32(f'{pd}.res_scale_hs.{"weight" if "scale" in sn else "bias"}', d32)
    else:
        router_map = [
            ('zaya_block.router.rmsnorm_eda.weight', 'ffn_norm.weight'),
            ('zaya_block.router.down_proj.weight', 'ffn_gate_inp.weight'),
            ('zaya_block.router.down_proj.bias', 'ffn_gate_inp.bias'),
            ('zaya_block.router.router_mlp.0.weight', 'ffn_gate.weight'),
            ('zaya_block.router.router_mlp.0.bias', 'ffn_gate.bias'),
            ('zaya_block.router.router_mlp.2.weight', 'zaya_router_mlp2.weight'),
            ('zaya_block.router.router_mlp.2.bias', 'zaya_router_mlp2.bias'),
            ('zaya_block.router.router_mlp.4.weight', 'zaya_router_mlp4.weight'),
        ]
        for src_s, dst_s in router_map:
            k = f'{ps}.{src_s}'
            if has(k): add(f'{pd}.{dst_s}', t(k))
        
        k = f'{ps}.zaya_block.router.balancing_biases'
        if has(k):
            d32 = gguf.dequantize(t(k).data, t(k).tensor_type).astype(np.float32).ravel()[:17]
            add_f32(f'{pd}.zaya_router_biases.weight', d32)
        
        for sn in ['hidden_states_scale','hidden_states_bias','residual_scale','residual_bias']:
            k = f'{ps}.res_scale.{sn}'
            if has(k):
                d32 = gguf.dequantize(t(k).data, t(k).tensor_type).astype(np.float32).ravel()
                pr = 'res_scale_hs' if 'hidden' in sn else 'res_scale_res'
                sf = 'weight' if 'scale' in sn else 'bias'
                add_f32(f'{pd}.{pr}.{sf}', d32)
        
        # Per-expert weights (store individually since GGUF can't pack MXFP4 into 3D)
        exp_names = {}
        for e in range(n_exp):
            for src_w, dst_w in [('linear_fc1','ffn_gate_up_exps'),('linear_fc2','ffn_down_exps')]:
                k = f'{ps}.zaya_block.experts.local_experts.{e}.{src_w}.weight'
                if has(k):
                    dst = f'{pd}.{dst_w}.{e}.weight'
                    add(dst, t(k))
                    exp_names.setdefault(dst_w, []).append(dst)
    if l % 10 == 0: print(f'  L{l}/{n_layer}', flush=True)

print('Writing GGUF...', flush=True)
writer.write_header_to_file(); writer.write_kv_data_to_file()
writer.write_tensors_to_file(); writer.close()
sz = os.path.getsize(DST) / 1e9
print(f'Done! {sz:.2f} GB → {DST}')
