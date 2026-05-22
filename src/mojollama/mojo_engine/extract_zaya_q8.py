#!/usr/bin/env python3
"""
Extract ZAYA1-8B Q8_0 weights from GGUF to raw .bin files for Mojo engine.
Output: /tmp/weights_zaya/*.bin — one file per weight matrix.

Key findings from GGUF:
- token_embd.weight shape [2048, 262147] Q8_0, data shape (262147, 2176)
  → Physical layout is [NV, NE] Q8_0 — ready for row-wise matmul as LM head
  → Also used as embedding lookup (row = token id)
- ffn_gate_inp.weight is F32 (type 0), not Q8_0
- ffn_norm.weight is [256] (router hidden dim), separate from attn_norm.weight [2048]
- No separate output weight — tied embeddings
"""

import sys, os, json
import numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
import gguf

GGUF_PATH = '/tmp/models/ZAYA1-8B-Q8_0.gguf'
OUT_DIR = '/tmp/weights_zaya'

# Architecture constants (verified from GGUF scanner)
NE = 2048       # hidden_size
NH = 8          # num_attention_heads
NK = 2          # num_key_value_heads
HD = 128        # head_dim
NL = 80         # num_hidden_layers
N_EXP = 16      # num_experts
N_RH = 256      # router hidden dim
FF = 4096       # expert intermediate (gate+up combined)
F2 = 2048       # half after split
NV = 262147     # vocab_size
QK = 32         # Q8_0 block size
QB = 34         # Q8_0 bytes per block


def dump_raw_bin(data: bytes, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    print(f"  {os.path.basename(path)}: {len(data)} bytes ({len(data)/1024/1024:.1f} MB)")


def get_bytes(tensor_map, name: str) -> bytes | None:
    t = tensor_map.get(name)
    if t is None:
        return None
    data = t.data
    if hasattr(data, 'tobytes'):
        return data.tobytes()
    return bytes(data)


def main():
    if not os.path.exists(GGUF_PATH):
        print(f"ERROR: GGUF not found: {GGUF_PATH}")
        sys.exit(1)

    reader = gguf.GGUFReader(GGUF_PATH)
    tensor_map = {t.name: t for t in reader.tensors}
    print(f"Reading: {GGUF_PATH}")
    print(f"Total tensors: {len(reader.tensors)}")

    # ─── Global weights ───
    print("\n--- Global ---")

    # output_norm.weight: [2048] F32
    raw = get_bytes(tensor_map, 'output_norm.weight')
    if raw: dump_raw_bin(raw, f'{OUT_DIR}/output_norm_weight.bin')

    # token_embd.weight: [2048, 262147] Q8_0, physical (262147, 2176)
    # Serves as both embedding table AND LM head (tied weights)
    raw = get_bytes(tensor_map, 'token_embd.weight')
    if raw: dump_raw_bin(raw, f'{OUT_DIR}/token_embd_weight.bin')

    # ─── Per-layer weights ───
    extracted = 0
    for l in range(NL):
        prefix = f'blk.{l}'

        # Common: res_scales on EVERY layer
        common = [
            ('res_scale_hs.weight', f'blk_{l}_res_scale_hs_weight.bin', 'f32'),
            ('res_scale_hs.bias',   f'blk_{l}_res_scale_hs_bias.bin', 'f32'),
            ('res_scale_res.weight', f'blk_{l}_res_scale_res_weight.bin', 'f32'),
            ('res_scale_res.bias',  f'blk_{l}_res_scale_res_bias.bin', 'f32'),
        ]

        if l % 2 == 0:
            # ── Attention layer ──
            tensors = common + [
                ('attn_norm.weight',     f'blk_{l}_attn_norm_weight.bin', 'f32'),
                ('attn_q.weight',        f'blk_{l}_attn_q_weight.bin', 'q8'),     # [2048, 1024]
                ('attn_k.weight',        f'blk_{l}_attn_k_weight.bin', 'q8'),     # [2048, 256]
                ('attn_output.weight',   f'blk_{l}_attn_output_weight.bin', 'q8'), # [1024, 2048]
            ]
        else:
            # ── MoE layer ──
            tensors = common + [
                ('attn_norm.weight',            f'blk_{l}_attn_norm_weight.bin', 'f32'),
                ('ffn_gate_inp.weight',         f'blk_{l}_ffn_gate_inp_weight.bin', 'f32'),  # F32!
                ('ffn_gate_inp.bias',           f'blk_{l}_ffn_gate_inp_bias.bin', 'f32'),
                ('ffn_gate.weight',             f'blk_{l}_ffn_gate_weight.bin', 'q8'),
                ('ffn_gate.bias',               f'blk_{l}_ffn_gate_bias.bin', 'f32'),
                ('zaya_router_mlp2.weight',     f'blk_{l}_zaya_router_mlp2_weight.bin', 'q8'),
                ('zaya_router_mlp2.bias',       f'blk_{l}_zaya_router_mlp2_bias.bin', 'f32'),
                ('zaya_router_mlp4.weight',     f'blk_{l}_zaya_router_mlp4_weight.bin', 'q8'),
                ('zaya_router_biases.weight',   f'blk_{l}_zaya_router_biases_weight.bin', 'f32'),
                ('zaya_router_eda.weight',      f'blk_{l}_zaya_router_eda_weight.bin', 'f32'),
                ('ffn_gate_up_exps.weight',     f'blk_{l}_ffn_gate_up_exps_weight.bin', 'q8'),
                ('ffn_down_exps.weight',        f'blk_{l}_ffn_down_exps_weight.bin', 'q8'),
            ]

        for suffix, out_name, dtype in tensors:
            tname = f'{prefix}.{suffix}'
            raw = get_bytes(tensor_map, tname)
            if raw:
                dump_raw_bin(raw, f'{OUT_DIR}/{out_name}')
                extracted += 1
            else:
                print(f"  MISSING: {tname}")

    # ─── Metadata ───
    meta = {
        'hidden_size': NE, 'num_attention_heads': NH,
        'num_key_value_heads': NK, 'head_dim': HD,
        'num_hidden_layers': NL, 'num_experts': N_EXP,
        'router_hidden_size': N_RH, 'expert_intermediate_size': FF,
        'vocab_size': NV,
        'quant': 'Q8_0', 'weight_tying': True,
    }
    with open(f'{OUT_DIR}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtracted {extracted} tensors to {OUT_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
