#!/usr/bin/env python3
"""Extract GPT-OSS-20B weights from GGUF → Q8_0 .bin for Mojo engine.
Uses gguf.dequantize() to handle all quant types.
"""
import sys, os, json, struct
import numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
import gguf
from gguf import GGMLQuantizationType as QType, dequantize, quantize

GGUF_PATH = '/tmp/models/gpt-oss-20b-Q4_K_M.gguf'
OUT_DIR = '/tmp/weights_gptoss'

NE, NH, NK, HD = 2880, 64, 8, 64
QI = NH * HD  # 4096
NL, N_EXP, N_ACT, FF = 24, 32, 4, 2880
NV = 201088
QK, QB = 32, 34

os.makedirs(OUT_DIR, exist_ok=True)

def dump(path, data):
    with open(path, 'wb') as f: f.write(data)
    mb = len(data) / 1024 / 1024
    print(f"  {os.path.basename(path)}: {mb:.0f} MB")

def quant_q8(f32_2d):
    """f32 [rows,cols] → Q8_0 bytes using built-in quantize."""
    if len(f32_2d.shape) == 1:
        return f32_2d.tobytes()  # 1D = f32
    q8 = quantize(f32_2d, QType.Q8_0)
    return q8.tobytes()

def dequant_gguf(tensor):
    """Dequantize a GGUF tensor to f32 using gguf.dequantize()."""
    data = tensor.data
    qtype = QType(tensor.tensor_type)
    # gguf.dequantize expects the raw packed numpy array
    return dequantize(data, qtype)

def save_tensor(tensor, out_name, reshape_2d=None):
    """Dequantize → reshape → Q8_0 → save."""
    try:
        f32 = dequant_gguf(tensor)
    except Exception as e:
        print(f"  SKIP {out_name}: dequant failed: {e}")
        return False
    
    if reshape_2d:
        f32 = f32.reshape(reshape_2d)
    
    q8 = quant_q8(f32)
    dump(f'{OUT_DIR}/{out_name}', q8)
    return True

def save_raw(tensor, out_name):
    """Save raw bytes directly."""
    dump(f'{OUT_DIR}/{out_name}', tensor.data.tobytes())

# ─── Main ───
reader = gguf.GGUFReader(GGUF_PATH)
tmap = {t.name: t for t in reader.tensors}
print(f"Extracting GPT-OSS-20B → {OUT_DIR}/")

print("\n--- Global ---")
save_raw(tmap['output_norm.weight'], 'output_norm_weight.bin')
save_tensor(tmap['output.weight'], 'output_weight.bin', (NV, NE))
save_tensor(tmap['token_embd.weight'], 'token_embd_weight.bin', (NV, NE))

print("\n--- Per-layer ---")
for l in range(NL):
    p = f'blk.{l}'
    if l % 2 == 0:
        print(f"\n  Layer {l}:")
    
    # Attention weights (types: 6=Q6_K, 8=Q8_0, 12=Q4_0)
    save_tensor(tmap[f'{p}.attn_norm.weight'], f'blk_{l}_attn_norm_weight.bin')
    save_tensor(tmap[f'{p}.attn_q.weight'], f'blk_{l}_attn_q_weight.bin', (NE, QI))
    save_tensor(tmap[f'{p}.attn_k.weight'], f'blk_{l}_attn_k_weight.bin', (NE, NK*HD))
    save_tensor(tmap[f'{p}.attn_v.weight'], f'blk_{l}_attn_v_weight.bin', (NE, NK*HD))
    save_tensor(tmap[f'{p}.attn_output.weight'], f'blk_{l}_attn_output_weight.bin', (QI, NE))
    # Biases (F32)
    save_raw(tmap[f'{p}.attn_q.bias'], f'blk_{l}_attn_q_bias.bin')
    save_raw(tmap[f'{p}.attn_k.bias'], f'blk_{l}_attn_k_bias.bin')
    save_raw(tmap[f'{p}.attn_v.bias'], f'blk_{l}_attn_v_bias.bin')
    save_raw(tmap[f'{p}.attn_output.bias'], f'blk_{l}_attn_output_bias.bin')
    # Sinks
    save_raw(tmap[f'{p}.attn_sinks.weight'], f'blk_{l}_attn_sinks_weight.bin')
    # FFN norm
    save_tensor(tmap[f'{p}.post_attention_norm.weight'], f'blk_{l}_post_attn_norm_weight.bin')
    
    # MoE weights
    save_tensor(tmap[f'{p}.ffn_gate_inp.weight'], f'blk_{l}_ffn_gate_inp_weight.bin', (32, NE))
    
    # Expert weights (MXFP4 type 39) — dequant full 3D to f32 → reshape [N_EXP*rows, cols] → Q8_0
    # Shapes from GGUF: gate/up/down all [NE, FF, N_EXP] = [2880, 2880, 32]
    for exp_name, exp_out in [
        (f'{p}.ffn_gate_exps.weight', f'blk_{l}_ffn_gate_exps_weight.bin'),
        (f'{p}.ffn_up_exps.weight', f'blk_{l}_ffn_up_exps_weight.bin'),
        (f'{p}.ffn_down_exps.weight', f'blk_{l}_ffn_down_exps_weight.bin'),
    ]:
        if exp_name in tmap:
            t = tmap[exp_name]
            n_rows, n_cols, n_exp = [int(x) for x in t.shape]  # [2880, 2880, 32]
            f32_3d = dequant_gguf(t)  # f32 [n_exp, n_rows, n_cols] or [n_rows, n_cols, n_exp]
            # The gguf might return shape as (n_exp, n_rows, n_cols) or (n_rows, n_cols, n_exp)
            # Check actual shape
            if f32_3d.shape[0] == n_exp:
                # Shape is (n_exp, n_rows, n_cols) — already expert-major
                pass
            else:
                # Shape is (n_rows, n_cols, n_exp) — need to reorder
                f32_3d = np.moveaxis(f32_3d, -1, 0)  # → (n_exp, n_rows, n_cols)
            # Stack experts: [n_exp * n_rows, n_cols]
            f32_2d = f32_3d.reshape(n_exp * n_rows, n_cols)
            q8 = quant_q8(f32_2d)
            dump(f'{OUT_DIR}/{exp_out}', q8)

# Metadata
meta = {'hidden_size': NE, 'num_attention_heads': NH, 'num_key_value_heads': NK,
        'head_dim': HD, 'num_hidden_layers': NL, 'num_experts': N_EXP,
        'experts_per_tok': N_ACT, 'expert_intermediate_size': FF,
        'vocab_size': NV, 'quant': 'Q8_0'}
with open(f'{OUT_DIR}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\nDone! Weights in {OUT_DIR}/")
