#!/usr/bin/env python3
"""
Extract ZAYA1-8B weights from Q8_0 GGUF to .bin files for Mojo engine.

Output: /tmp/weights_zaya/*.bin — one file per weight matrix.
- Q8_0 matmul weights: raw Q8_0 bytes (34 bytes per 32-element block)
- F32 norm/scale/bias weights: raw f32 bytes
- Expert weights: packed 3D Q8_0 tensors, one file per expert group
"""
import sys, os, json, struct
import numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
import gguf

GGUF_PATH = '/tmp/models/ZAYA1-8B-Q8_0.gguf'
OUT_DIR = '/tmp/weights_zaya'

# Architecture constants
NE = 2048       # hidden_size
NH = 8          # num_attention_heads
NK = 2          # num_key_value_heads
HD = 128        # head_dim
NL = 80         # num_hidden_layers
N_EXP = 16      # num_experts
N_EXP_USED = 1  # moe_router_topk
FF = 4096       # expert intermediate (gate+up combined)
NV = 262147     # vocab_size
ROPE_DIM = 64
RPE_THETA = 5000000.0

# Q8_0 constants
QK = 32         # block size
QB = 34         # bytes per block (2 scale + 32 quants)

def dump_raw_bin(tensor_data: bytes, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(tensor_data)
    print(f"  {os.path.basename(path)}: {len(tensor_data)} bytes")

def make_tensor_map(reader):
    return {t.name: t for t in reader.tensors}

def extract_tensor_raw(tensor_map, tensor_name: str) -> bytes:
    t = tensor_map.get(tensor_name)
    if t is None:
        return None
    data = t.data
    # numpy memmap or ndarray
    if hasattr(data, 'tobytes'):
        return data.tobytes()
    return bytes(data)

def extract_f32_as_f16(tensor_map, tensor_name: str) -> bytes:
    t = tensor_map.get(tensor_name)
    if t is None:
        return None
    arr = np.array(t.data, dtype=np.float32)
    f16_arr = arr.astype(np.float16)
    return f16_arr.tobytes()

def extract_f32_raw(tensor_map, tensor_name: str) -> bytes:
    t = tensor_map.get(tensor_name)
    if t is None:
        return None
    arr = np.array(t.data, dtype=np.float32)
    return arr.tobytes()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    reader = gguf.GGUFReader(GGUF_PATH)
    tensor_map = make_tensor_map(reader)
    
    print(f"=== Extracting ZAYA1-8B weights from GGUF ===")
    print(f"Total tensors: {len(tensor_map)}")
    
    # ─── Global weights ───
    print("\n--- Global ---")
    
    # token_embd.weight: [2048, 262147] Q8_0 → token_embd_weight.bin (raw Q8_0)
    raw = extract_tensor_raw(tensor_map, 'token_embd.weight')
    if raw:
        dump_raw_bin(raw, f'{OUT_DIR}/token_embd_weight.bin')
    
    # output_norm.weight: [2048] F32 → output_norm_weight.bin
    raw = extract_f32_raw(tensor_map, 'output_norm.weight')
    if raw:
        dump_raw_bin(raw, f'{OUT_DIR}/output_norm_weight.bin')
    
    # ─── Per-layer weights ───
    for l in range(NL):
        prefix = f'blk.{l}'
        
        # Every layer has these:
        tensors_every = [
            ('attn_norm.weight', f'attn_norm_weight.bin', 'f32'),
            ('res_scale_hs.weight', f'res_scale_hs_weight.bin', 'f32'),
            ('res_scale_hs.bias', f'res_scale_hs_bias.bin', 'f32'),
            ('res_scale_res.weight', f'res_scale_res_weight.bin', 'f32'),
            ('res_scale_res.bias', f'res_scale_res_bias.bin', 'f32'),
        ]
        
        if l % 2 == 0:  # Attention layer
            tensors_specific = [
                ('attn_q.weight', f'attn_q_weight.bin', 'q8'),
                ('attn_k.weight', f'attn_k_weight.bin', 'q8'),
                ('attn_output.weight', f'attn_output_weight.bin', 'q8'),
                # CCA weights (extracted but not used in simple GQA path)
                ('cca_val_proj1.weight', f'cca_val_proj1_weight.bin', 'q8'),
                ('cca_val_proj2.weight', f'cca_val_proj2_weight.bin', 'q8'),
                ('cca_conv_grp.weight', f'cca_conv_grp_weight.bin', 'raw'),
                ('cca_conv_grp.bias', f'cca_conv_grp_bias.bin', 'f32'),
                ('cca_k_scale.weight', f'cca_k_scale_weight.bin', 'f32'),
                ('ssm_conv1d.weight', f'ssm_conv1d_weight.bin', 'raw'),
                ('ssm_conv1d.bias', f'ssm_conv1d_bias.bin', 'f32'),
            ]
        else:  # MoE layer
            tensors_specific = [
                ('ffn_norm.weight', f'ffn_norm_weight.bin', 'f32'),
                ('ffn_gate_inp.weight', f'ffn_gate_inp_weight.bin', 'f32'),
                ('ffn_gate_inp.bias', f'ffn_gate_inp_bias.bin', 'f32'),
                ('ffn_gate.weight', f'ffn_gate_weight.bin', 'q8'),
                ('ffn_gate.bias', f'ffn_gate_bias.bin', 'f32'),
                ('zaya_router_mlp2.weight', f'zaya_router_mlp2_weight.bin', 'q8'),
                ('zaya_router_mlp2.bias', f'zaya_router_mlp2_bias.bin', 'f32'),
                ('zaya_router_mlp4.weight', f'zaya_router_mlp4_weight.bin', 'q8'),
                ('zaya_router_biases.weight', f'zaya_router_biases_weight.bin', 'f32'),
                ('zaya_router_eda.weight', f'zaya_router_eda_weight.bin', 'f32'),
                ('ffn_gate_up_exps.weight', f'ffn_gate_up_exps_weight.bin', 'q8'),
                ('ffn_down_exps.weight', f'ffn_down_exps_weight.bin', 'q8'),
            ]
        
        for tensor_suffix, out_name, dtype in tensors_every + tensors_specific:
            tensor_name = f'{prefix}.{tensor_suffix}'
            fn = f'{OUT_DIR}/blk_{l}_{out_name}'
            
            if dtype == 'q8':
                raw = extract_tensor_raw(tensor_map, tensor_name)
            elif dtype == 'f32':
                raw = extract_f32_raw(tensor_map, tensor_name)
            else:
                raw = extract_tensor_raw(tensor_map, tensor_name)
            
            if raw:
                dump_raw_bin(raw, fn)
            else:
                print(f"  MISSING: {tensor_name}")
    
    # ─── Write metadata ───
    meta = {
        'hidden_size': NE,
        'num_attention_heads': NH,
        'num_key_value_heads': NK,
        'head_dim': HD,
        'num_hidden_layers': NL,
        'num_experts': N_EXP,
        'expert_used_count': N_EXP_USED,
        'expert_intermediate_size': FF,
        'vocab_size': NV,
        'rope_dim': ROPE_DIM,
        'rope_theta': RPE_THETA,
        'max_seq_len': 128,
        'quant': 'Q8_0',
    }
    with open(f'{OUT_DIR}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata written to {OUT_DIR}/meta.json")
    print("Done!")

if __name__ == '__main__':
    main()
