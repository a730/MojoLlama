#!/usr/bin/env python3
"""
Convert Zyphra/ZAYA1-8B HuggingFace safetensors → MXFP4 GGUF.

MXFP4 format (GGML type 39):
  - 32-element blocks
  - 16 bytes packed 4-bit two's complement mantissas
  - 1 byte E8M0 scale per block (exponent, bias 127)
  - Total: 17 bytes per 32 elements

Usage:
    python3 convert_zaya_to_mxfp4.py [output_dir]
"""

import os, sys, json, time, math, gc
import numpy as np
from pathlib import Path

MXFP4_BLOCK_SIZE = 32
MXFP4_BLOCK_BYTES = 17


# ─── MXFP4 quantization ──────────────────────────────────────────────────────

def quantize_mxfp4(arr_f32: np.ndarray) -> bytes:
    """Vectorized MXFP4 quantization of a flat float32 array."""
    n = arr_f32.shape[0]
    n_blocks = (n + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    padded_n = n_blocks * MXFP4_BLOCK_SIZE

    if n < padded_n:
        arr_f32 = np.pad(arr_f32, (0, padded_n - n))

    blocks = arr_f32.reshape(n_blocks, MXFP4_BLOCK_SIZE)

    amax = np.max(np.abs(blocks), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        exp = np.floor(np.log2(amax)).astype(np.int32) + 127
    exp = np.clip(exp, 0, 255)
    exp[amax == 0.0] = 0

    scales = np.ldexp(np.ones(n_blocks, dtype=np.float32), exp.astype(np.int32) - 127)

    q = np.round(blocks / scales[:, np.newaxis]).astype(np.int32)
    q = np.clip(q, -8, 7)
    q[exp == 0] = 0

    nibbles = np.clip(q + 8, 0, 15).astype(np.uint8)
    packed = np.bitwise_or(nibbles[:, 0::2], nibbles[:, 1::2].astype(np.uint16) << 4).astype(np.uint8)

    result = np.zeros(n_blocks * MXFP4_BLOCK_BYTES, dtype=np.uint8)
    result.reshape(n_blocks, MXFP4_BLOCK_BYTES)[:, :16] = packed
    result.reshape(n_blocks, MXFP4_BLOCK_BYTES)[:, 16] = exp.astype(np.uint8)

    return bytes(result)


def quantize_mxfp4_rowwise(arr_f32: np.ndarray) -> np.ndarray:
    """Quantize float32 array to MXFP4 byte array, row by row.
    
    For 2D [rows, cols]: quantize each row → byte shape [rows, n_blocks * 17]
    For 1D [n]: → byte shape [1, n_blocks * 17]
    For 3D+ [d0, d1, d2, ...]: → flatten to 2D [d0, d1*d2*...], quantize, restore shape
    """
    orig_ndim = arr_f32.ndim
    if orig_ndim == 1:
        arr_f32 = arr_f32.reshape(1, -1)

    # Flatten 3D+ to 2D [rows, cols]
    if arr_f32.ndim > 2:
        arr_f32 = arr_f32.reshape(arr_f32.shape[0], -1)

    rows, cols = arr_f32.shape
    n_blocks_per_row = (cols + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    n_bytes_per_row = n_blocks_per_row * MXFP4_BLOCK_BYTES

    result = np.zeros((rows, n_bytes_per_row), dtype=np.uint8)
    for r in range(rows):
        qbytes = quantize_mxfp4(arr_f32[r])
        result[r] = np.frombuffer(qbytes, dtype=np.uint8)

    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def should_quantize(name: str) -> bool:
    """Only quantize weight matrices (keep norms/biases/embeddings as F16)."""
    skip_suffixes = [
        'norm.weight', 'norm.bias',
        'attn_norm.weight', 'attn_norm.bias',
        'ffn_norm.weight', 'ffn_norm.bias',
        'res_scale_hs.weight', 'res_scale_hs.bias',
        'res_scale_res.weight', 'res_scale_res.bias',
        'zaya_router_mlp2.bias', 'zaya_router_mlp4.bias',
        'zaya_router_biases.bias',
        'ffn_gate_inp.bias', 'ffn_gate.bias',
        'token_embd.weight',
        'embed_tokens.weight',
        'output_norm.weight', 'output_norm.bias',
        'output.weight',
        'lm_head.weight',
    ]
    for suffix in skip_suffixes:
        if name.endswith(suffix):
            return False
    return True


def download_shards(repo_id: str, index_file: str, cache_dir: str = None):
    """Download all safetensor shards."""
    from huggingface_hub import hf_hub_download
    import json as j

    index_path = hf_hub_download(repo_id, index_file, cache_dir=cache_dir)
    with open(index_path) as f:
        index = j.load(f)

    weight_map = index.get('weight_map', {})

    shard_files = set(weight_map.values())
    shard_paths = {}
    for shard in shard_files:
        shard_paths[shard] = hf_hub_download(repo_id, shard, cache_dir=cache_dir)

    return weight_map, shard_paths


# ─── Main converter ───────────────────────────────────────────────────────────

def convert_zaya_to_mxfp4(output_dir: str = "models", hf_cache: str = None):
    """Convert Zyphra/ZAYA1-8B safetensors to MXFP4 GGUF."""
    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType as QT

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ZAYA1-8B-MXFP4.gguf")
    repo_id = "Zyphra/ZAYA1-8B"

    print("=" * 60)
    print("  Zyphra/ZAYA1-8B → MXFP4 GGUF Converter")
    print("=" * 60)

    # ─── Load config ───
    print("\n[1/4] Loading config...")
    config_path = hf_hub_download(repo_id, "config.json", cache_dir=hf_cache)
    with open(config_path) as f:
        config = json.load(f)

    hidden_size = config['hidden_size']
    n_layers = config['num_hidden_layers']
    n_heads = config['num_attention_heads']
    n_kv_heads = config.get('num_key_value_heads', config.get('num_query_groups', n_heads))
    n_experts = config.get('num_experts', 16)
    n_experts_per_tok = config.get('moe_router_topk', 1)
    vocab_size = config['vocab_size']
    intermediate_size = config.get('ffn_hidden_size', config.get('intermediate_size'))
    head_dim = config.get('head_dim', hidden_size // n_heads)
    rope_theta = config.get('rope_theta', 10000.0)
    norm_eps = config.get('norm_epsilon', 1e-5)
    partial_rotary = config.get('partial_rotary_factor', 0.5)
    max_seq_len = config.get('max_position_embeddings', 131072)

    print(f"  Layers: {n_layers}, Heads: {n_heads}, KV: {n_kv_heads}")
    print(f"  Hidden: {hidden_size}, Intermediate: {intermediate_size}")
    print(f"  Vocab: {vocab_size}, Experts: {n_experts}")

    # ─── Download ───
    print("\n[2/4] Downloading safetensor shards...")
    weight_map, shard_paths = download_shards(repo_id, "model.safetensors.index.json", hf_cache)
    print(f"  {len(shard_paths)} shard(s), {len(weight_map)} tensor(s)")

    total_params = len(weight_map)
    total_quantized = 0
    total_skipped = 0

    shard_handles = {name: safe_open(path, framework="pt") for name, path in shard_paths.items()}

    # ─── Quantize ───
    print("\n[3/4] Quantizing tensors to MXFP4...")

    quantized_tensors = []  # (name, data_bytes, ggml_type, shape)
    t0 = time.time()

    for idx, (name, shard_name) in enumerate(weight_map.items()):
        handle = shard_handles[shard_name]
        tensor_pt = handle.get_tensor(name)
        tensor = tensor_pt.to(torch.float32).numpy()
        is_quant = should_quantize(name)

        if is_quant:
            qdata = quantize_mxfp4_rowwise(tensor)
            ggml_type = QT.MXFP4
            total_quantized += 1
        else:
            qdata = tensor.astype(np.float16).view(np.uint8)
            ggml_type = QT.F16
            total_skipped += 1

        quantized_tensors.append((name, qdata, ggml_type))

        if (idx + 1) % 20 == 0 or (idx + 1) == total_params:
            elapsed = time.time() - t0
            print(f"    [{idx+1}/{total_params}] {name:60s} → "
                  f"{'MXFP4' if is_quant else 'F16':5s}  ({elapsed:.1f}s)")

    for h in shard_handles.values():
        del h

    # ─── Write GGUF ───
    print("\n[4/4] Writing GGUF using gguf.GGUFWriter...")
    t1 = time.time()

    rope_dim = int(head_dim * partial_rotary)
    w = GGUFWriter(output_path, "zaya")

    # Metadata
    w.add_string("general.architecture", "zaya")
    w.add_string("general.name", "ZAYA1-8B-MXFP4")
    w.add_string("general.description", "Zyphra ZAYA1-8B quantized to MXFP4")
    w.add_int32("general.file_type", 39)
    w.add_int32("general.quantization_version", 1)
    w.add_string("general.source.huggingface", "Zyphra/ZAYA1-8B")
    w.add_int32("zaya.block_count", n_layers)
    w.add_int32("zaya.context_length", max_seq_len)
    w.add_int32("zaya.embedding_length", hidden_size)
    w.add_int32("zaya.feed_forward_length", intermediate_size)
    w.add_int32("zaya.attention.head_count", n_heads)
    w.add_int32("zaya.attention.head_count_kv", n_kv_heads)
    w.add_int32("zaya.attention.head_dim", head_dim)
    w.add_float32("zaya.attention.layer_norm_rms_epsilon", norm_eps)
    w.add_int32("zaya.rope.dimension_count", rope_dim)
    w.add_float32("zaya.rope.freq_base", rope_theta)
    w.add_int32("zaya.expert_count", n_experts)
    w.add_int32("zaya.experts_used_count", n_experts_per_tok)
    w.add_int32("zaya.vocab_size", vocab_size)

    # Tensors
    for name, data, ggml_type in quantized_tensors:
        if ggml_type == QT.MXFP4:
            w.add_tensor(name, data, raw_dtype=QT.MXFP4)
        else:
            w.add_tensor(name, data, raw_dtype=QT.F16)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    elapsed_total = time.time() - t0
    file_size = os.path.getsize(output_path)
    gc.collect()

    print(f"\n  {'='*50}")
    print(f"  Conversion Complete")
    print(f"  {'='*50}")
    print(f"  Output:          {output_path}")
    print(f"  Total tensors:   {total_params}")
    print(f"  MXFP4:           {total_quantized}")
    print(f"  F16:             {total_skipped}")
    print(f"  File size:       {file_size / 1024**3:.2f} GB")
    print(f"  Total time:      {elapsed_total:.1f}s")
    print(f"  {'='*50}")

    return output_path


if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    output = convert_zaya_to_mxfp4(out_dir)
    print(f"\nDone: {output}")
