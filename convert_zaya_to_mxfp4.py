#!/usr/bin/env python3
"""
Convert Zyphra/ZAYA1-8B HuggingFace safetensors → MXFP4 GGUF.

MXFP4 format (GGML type 39):
  - 32-element blocks
  - 16 bytes packed 4-bit two's complement mantissas
  - 1 byte E8M0 scale per block (power-of-2 exponent, bias 127)
  - Total: 17 bytes per 32 elements

Usage:
    python3 convert_zaya_to_mxfp4.py [output_dir]
"""

import os, sys, json, time, math, struct, gc
import numpy as np
from pathlib import Path

# ─── GGUF constants ──────────────────────────────────────────────────────────
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGUF_TYPE_UINT8  = 0; GGUF_TYPE_INT8    = 1; GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16  = 3; GGUF_TYPE_UINT32  = 4; GGUF_TYPE_INT32  = 5
GGUF_TYPE_FLOAT32 = 6; GGUF_TYPE_BOOL    = 7; GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY  = 9; GGUF_TYPE_UINT64  = 10; GGUF_TYPE_INT64  = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_MXFP4 = 39

MXFP4_BLOCK_SIZE = 32
MXFP4_BLOCK_BYTES = 17  # 16 bytes mantissas + 1 byte scale


# ─── MXFP4 quantization ──────────────────────────────────────────────────────

def quantize_mxfp4(arr_f32: np.ndarray) -> bytes:
    """Quantize float32 array to MXFP4 blocks.

    Each block: 32 elements → 16 bytes (packed 4-bit nibbles) + 1 byte E8M0 scale.
    Mantissas are signed 4-bit two's complement (-8..7), stored with bias 8.
    """
    n = arr_f32.shape[0]
    n_blocks = (n + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    result = bytearray()

    for b in range(n_blocks):
        start = b * MXFP4_BLOCK_SIZE
        end = min(start + MXFP4_BLOCK_SIZE, n)
        chunk = arr_f32[start:end]

        if len(chunk) < MXFP4_BLOCK_SIZE:
            chunk = np.pad(chunk, (0, MXFP4_BLOCK_SIZE - len(chunk)))

        # E8M0 scale: power-of-2 exponent (bias 127)
        amax = np.max(np.abs(chunk))
        if amax == 0.0:
            exponent = 0
        else:
            exponent = min(255, max(0, int(math.floor(math.log2(amax)) + 127)))
        scale = math.ldexp(1.0, exponent - 127) if exponent > 0 else 0.0

        # Quantize to 4-bit two's complement
        if scale > 0:
            q = np.round(chunk / scale).astype(np.int32)
            q = np.clip(q, -8, 7)
        else:
            q = np.zeros(MXFP4_BLOCK_SIZE, dtype=np.int32)

        # Bias by 8 for unsigned storage
        nibbles = np.clip(q + 8, 0, 15).astype(np.uint8)

        # Pack 2 nibbles per byte (lo, hi)
        packed = np.bitwise_or(nibbles[0::2], np.left_shift(nibbles[1::2], 4)).astype(np.uint8)

        result += packed.tobytes()
        result += struct.pack('<B', exponent & 0xFF)

    return bytes(result)


def get_mxfp4_byte_size(n_elements: int) -> int:
    blocks = (n_elements + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    return blocks * MXFP4_BLOCK_BYTES


# ─── GGUF helpers ─────────────────────────────────────────────────────────────

def encode_gguf_value(value):
    """Encode a Python value as GGUF metadata bytes (little-endian)."""
    if isinstance(value, bool):
        return struct.pack('<B', GGUF_TYPE_BOOL) + struct.pack('<B', 1 if value else 0)
    elif isinstance(value, int):
        return struct.pack('<B', GGUF_TYPE_INT32) + struct.pack('<i', value)
    elif isinstance(value, float):
        return struct.pack('<B', GGUF_TYPE_FLOAT32) + struct.pack('<f', value)
    elif isinstance(value, str):
        encoded = value.encode('utf-8')
        return struct.pack('<B', GGUF_TYPE_STRING) + struct.pack('<Q', len(encoded)) + encoded
    elif isinstance(value, bytes):
        return struct.pack('<B', GGUF_TYPE_STRING) + struct.pack('<Q', len(value)) + value
    elif isinstance(value, list):
        if not value:
            return struct.pack('<B', GGUF_TYPE_ARRAY) + struct.pack('<B', GGUF_TYPE_STRING) + struct.pack('<Q', 0)
        elem_type = GGUF_TYPE_STRING if isinstance(value[0], str) else \
                    GGUF_TYPE_INT32 if isinstance(value[0], int) else \
                    GGUF_TYPE_FLOAT32
        buf = struct.pack('<B', GGUF_TYPE_ARRAY) + struct.pack('<B', elem_type) + struct.pack('<Q', len(value))
        for v in value:
            if elem_type == GGUF_TYPE_STRING:
                v_enc = str(v).encode('utf-8')
                buf += struct.pack('<Q', len(v_enc)) + v_enc
            elif elem_type == GGUF_TYPE_INT32:
                buf += struct.pack('<i', v)
            elif elem_type == GGUF_TYPE_FLOAT32:
                buf += struct.pack('<f', v)
        return buf
    else:
        return struct.pack('<B', GGUF_TYPE_STRING) + struct.pack('<Q', 0)


def should_quantize(name: str) -> bool:
    """Only quantize weight matrices (not norms, biases, embeddings)."""
    skip_suffixes = [
        'norm.weight', 'norm.bias',
        'attn_norm.weight', 'attn_norm.bias',
        'ffn_norm.weight', 'ffn_norm.bias',
        'res_scale_hs.weight', 'res_scale_hs.bias',
        'res_scale_res.weight', 'res_scale_res.bias',
        'zaya_router_mlp2.weight', 'zaya_router_mlp2.bias',
        'zaya_router_mlp4.weight', 'zaya_router_mlp4.bias',
        'zaya_router_biases.weight', 'zaya_router_biases.bias',
        'ffn_gate_inp.bias', 'ffn_gate.bias',
        'token_embd.weight',
        'output_norm.weight', 'output_norm.bias',
        'output.weight',
        'lm_head.weight',
    ]
    for suffix in skip_suffixes:
        if name.endswith(suffix):
            return False
    return True


def get_tensor_dtype(dtype_str: str) -> int:
    if dtype_str == 'F32' or dtype_str == 'f32':
        return GGML_TYPE_F32
    elif dtype_str == 'F16' or dtype_str == 'f16':
        return GGML_TYPE_F16
    elif dtype_str == 'MXFP4' or dtype_str == 'mxfp4':
        return GGML_TYPE_MXFP4
    else:
        return GGML_TYPE_F16


# ─── Main converter ───────────────────────────────────────────────────────────

def download_shards(repo_id: str, index_file: str, cache_dir: str = None):
    """Download all safetensor shards and return {tensor_name: (filepath, offset, size)}."""
    from huggingface_hub import hf_hub_download
    import json as j

    # Download and parse index
    if cache_dir:
        index_path = hf_hub_download(repo_id, index_file, cache_dir=cache_dir)
    else:
        index_path = hf_hub_download(repo_id, index_file)
    
    with open(index_path) as f:
        index = j.load(f)
    
    weight_map = index.get('weight_map', {})
    
    # Download all unique shard files
    shard_files = set(weight_map.values())
    shard_paths = {}
    for shard in shard_files:
        if cache_dir:
            shard_paths[shard] = hf_hub_download(repo_id, shard, cache_dir=cache_dir)
        else:
            shard_paths[shard] = hf_hub_download(repo_id, shard)
    
    return weight_map, shard_paths


def convert_zaya_to_mxfp4(output_dir: str = "models", hf_cache: str = None):
    """Convert Zyphra/ZAYA1-8B safetensors to MXFP4 GGUF."""
    from huggingface_hub import hf_hub_download

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

    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {n_layers}")
    print(f"  Heads: {n_heads}, KV heads: {n_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Intermediate: {intermediate_size}")
    print(f"  Vocab: {vocab_size}")
    print(f"  Experts: {n_experts}, top-{n_experts_per_tok}")
    print(f"  Max seq len: {max_seq_len}")

    # ─── Download ───
    print("\n[2/4] Downloading safetensor shards...")
    from safetensors import safe_open
    weight_map, shard_paths = download_shards(repo_id, "model.safetensors.index.json", hf_cache)
    print(f"  {len(shard_paths)} shard(s), {len(weight_map)} tensor(s)")

    total_params = len(weight_map)
    total_quantized = 0
    total_skipped = 0

    # Open all shards with PyTorch (supports bfloat16 natively)
    import torch
    shard_handles = {name: safe_open(path, framework="pt") for name, path in shard_paths.items()}

    # ─── Quantize ───
    print("\n[3/4] Quantizing tensors to MXFP4...")
    tensor_infos = []
    quantized_chunks = {}  # name -> bytes
    skipped_chunks = {}    # name -> bytes (F16)

    t0 = time.time()
    for idx, (name, shard_name) in enumerate(weight_map.items()):
        handle = shard_handles[shard_name]
        tensor_pt = handle.get_tensor(name)
        # Convert to numpy float32 (handles bfloat16, float16, float32)
        tensor = tensor_pt.to(torch.float32).numpy()
        is_quant = should_quantize(name)

        if is_quant:
            arr = tensor.astype(np.float32)
            qdata = quantize_mxfp4(arr.ravel())
            quantized_chunks[name] = qdata
            byte_size = len(qdata)
            ggml_type = GGML_TYPE_MXFP4
            total_quantized += 1
        else:
            # Keep as F16
            f16 = tensor.astype(np.float16)
            skipped_chunks[name] = f16.tobytes()
            byte_size = f16.nbytes
            ggml_type = GGML_TYPE_F16
            total_skipped += 1

        shape = list(tensor.shape)
        tensor_infos.append({
            'name': name,
            'n_dims': len(shape),
            'shape': shape,
            'ggml_type': ggml_type,
            'byte_size': byte_size,
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == total_params:
            elapsed = time.time() - t0
            print(f"    [{idx+1}/{total_params}] {name:60s} → {'MXFP4' if is_quant else 'F16'}  "
                  f"({elapsed:.1f}s)")

    # Close shard handles
    for h in shard_handles.values():
        h.close()
    shard_handles.clear()

    # ─── Write GGUF ───
    print("\n[4/4] Writing GGUF file...")
    t1 = time.time()

    # Build metadata
    rope_dim = int(head_dim * partial_rotary)
    metadata_kv = [
        ('general.architecture',          encode_gguf_value('zaya')),
        ('general.name',                  encode_gguf_value('ZAYA1-8B-MXFP4')),
        ('general.description',           encode_gguf_value('Zyphra ZAYA1-8B quantized to MXFP4')),
        ('general.file_type',             encode_gguf_value(39)),
        ('general.quantization_version',  encode_gguf_value(1)),
        ('general.source.huggingface',    encode_gguf_value('Zyphra/ZAYA1-8B')),
        ('zaya.block_count',             encode_gguf_value(n_layers)),
        ('zaya.context_length',           encode_gguf_value(max_seq_len)),
        ('zaya.embedding_length',         encode_gguf_value(hidden_size)),
        ('zaya.feed_forward_length',      encode_gguf_value(intermediate_size)),
        ('zaya.attention.head_count',     encode_gguf_value(n_heads)),
        ('zaya.attention.head_count_kv',  encode_gguf_value(n_kv_heads)),
        ('zaya.attention.head_dim',       encode_gguf_value(head_dim)),
        ('zaya.attention.layer_norm_rms_epsilon', encode_gguf_value(norm_eps)),
        ('zaya.rope.dimension_count',     encode_gguf_value(rope_dim)),
        ('zaya.rope.freq_base',           encode_gguf_value(rope_theta)),
        ('zaya.expert_count',             encode_gguf_value(n_experts)),
        ('zaya.experts_used_count',       encode_gguf_value(n_experts_per_tok)),
        ('zaya.vocab_size',               encode_gguf_value(vocab_size)),
    ]

    with open(output_path, 'wb') as f:
        # ── Header ──
        n_tensors = len(tensor_infos)
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', n_tensors))
        metadata_offset_pos = f.tell()
        f.write(struct.pack('<Q', 0))  # placeholder

        # ── Tensor info block ──
        tensor_info_start = f.tell()
        for ti in tensor_infos:
            name_bytes = ti['name'].encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', ti['n_dims']))
            for s in ti['shape']:
                f.write(struct.pack('<Q', s))
            f.write(struct.pack('<I', ti['ggml_type']))
            f.write(struct.pack('<Q', ti['byte_size']))
            f.write(struct.pack('<Q', 0))  # data offset placeholder

        # ── Tensor data ──
        tensor_data_start = f.tell()
        for ti in tensor_infos:
            name = ti['name']
            data = quantized_chunks.get(name) or skipped_chunks.get(name)
            if data is None:
                raise RuntimeError(f"Missing data for tensor: {name}")
            f.write(data)
            # 32-byte alignment
            pad = (32 - (f.tell() % 32)) % 32
            if pad:
                f.write(b'\x00' * pad)

        # ── Metadata ──
        metadata_start = f.tell()
        f.write(struct.pack('<Q', len(metadata_kv)))
        for key, encoded_value in metadata_kv:
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            f.write(encoded_value)

        # ── Patching ──
        # Patch metadata offset in header
        f.seek(metadata_offset_pos)
        f.write(struct.pack('<Q', metadata_start))

        # Patch tensor data offsets
        f.seek(tensor_info_start)
        offset = tensor_data_start
        for ti in tensor_infos:
            name_bytes = ti['name'].encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', ti['n_dims']))
            for s in ti['shape']:
                f.write(struct.pack('<Q', s))
            f.write(struct.pack('<I', ti['ggml_type']))
            f.write(struct.pack('<Q', ti['byte_size']))
            f.write(struct.pack('<Q', offset))
            offset += ti['byte_size']
            pad = (32 - (offset % 32)) % 32
            offset += pad

    elapsed_total = time.time() - t0
    file_size = os.path.getsize(output_path)

    # ─── Summary ───
    print(f"\n  {'='*50}")
    print(f"  Conversion Complete")
    print(f"  {'='*50}")
    print(f"  Output:          {output_path}")
    print(f"  Total tensors:   {n_tensors}")
    print(f"  MXFP4 tensors:   {total_quantized}")
    print(f"  F16 tensors:     {total_skipped}")
    print(f"  File size:       {file_size / 1024**3:.2f} GB")
    print(f"  Total time:      {elapsed_total:.1f}s")
    print(f"  {'='*50}")

    return output_path


if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    output = convert_zaya_to_mxfp4(out_dir)
    print(f"\nDone: {output}")
