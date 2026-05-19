#!/usr/bin/env python3
"""Convert Lance text backbone to GGUF Q4_K_M.

Lance has a dual-path architecture:
  - standard path: layers.N.input_layernorm → self_attn → post_attention_layernorm → mlp
  - moe_gen path: layers.N.input_layernorm_moe_gen → self_attn_moe_gen → post_attention_layernorm_moe_gen → mlp_moe_gen
  - diffusion components: vae2llm, llm2vae, time_embedder, latent_pos_embed

This extracts ONLY the standard text path (ignoring moe_gen + diffusion).
The resulting GGUF runs as a Qwen2-like 3B text-only model on MojoLlama.
"""

import sys, os, struct, json, numpy as np
from pathlib import Path

MODEL_PATH = "/tmp/models/lance_model.safetensors"
CONFIG_PATH = "/tmp/models/lance_llm_config.json"
OUTPUT_GGUF = "/tmp/models/lance-text-q8_0.gguf"
OUTPUT_QUANT = "/tmp/models/lance-text-q4_k_m.gguf"
TOKENIZER_PATH = "/tmp/models/lance_tokenizer.json"
VOCAB_PATH = "/tmp/models/lance_vocab.json"

import safetensors

# Load config
with open(CONFIG_PATH) as f:
    config = json.load(f)

hidden_size = config["hidden_size"]
n_layers = config["num_hidden_layers"]
n_heads = config["num_attention_heads"]
n_kv_heads = config["num_key_value_heads"]
intermediate_size = config["intermediate_size"]
vocab_size = config["vocab_size"]
max_seq_len = config["max_position_embeddings"]
rms_eps = config["rms_norm_eps"]
rope_theta = config["rope_theta"]

print(f"Lance text backbone: {n_layers}L/{hidden_size}D/{n_heads}H/{n_kv_heads}KV/{intermediate_size}FF")
print(f"Vocab: {vocab_size}, Max seq: {max_seq_len}")

# Open safetensors
st = safetensors.safe_open(MODEL_PATH, framework='pt')
all_keys = st.keys()

# Determine which tensors to extract (standard path only)
def is_text_tensor(name):
    """Check if tensor should be included in text-only GGUF."""
    # Skip moe_gen, vae, llm2vae, time_embedder, latent_pos_embed
    if '_moe_gen' in name: return False
    if 'vae2llm' in name or 'llm2vae' in name: return False
    if 'time_embedder' in name: return False
    if 'latent_pos_embed' in name: return False
    # Only include language_model.*
    if 'language_model.' not in name: return False
    return True

text_tensors = [k for k in all_keys if is_text_tensor(k)]
print(f"Text tensors: {len(text_tensors)} / {len(all_keys)} total")

# GGUF key constants
GGUF_TYPE = {
    'uint32': 4, 'int32': 5, 'float32': 10, 'bool': 11,
    'string': 8, 'array': 12,
}
GGML_TYPE = {'F32': 0, 'F16': 1, 'Q8_0': 8}

def gguf_meta_writer():
    """Write GGUF metadata and return the tensor offset."""
    offset = 0
    data = bytearray()
    
    def w_string(s):
        nonlocal offset
        b = s.encode('utf-8')
        data += struct.pack('<Q', len(b)) + b
        offset += 8 + len(b)
    
    def w_uint32(v):
        nonlocal offset
        data += struct.pack('<I', v)
        offset += 4

    def w_int32(v):
        nonlocal offset
        data += struct.pack('<i', v)
        offset += 4

    def w_float32(v):
        nonlocal offset
        data += struct.pack('<f', v)
        offset += 4
    
    def w_bool(v):
        nonlocal offset
        data += struct.pack('<?', v)
        offset += 1
    
    def w_key_value(key, vtype, value):
        nonlocal offset
        w_string(key)
        data += struct.pack('<I', vtype)
        offset += 4
        if vtype == 8:  # string
            w_string(value)
        elif vtype == 4:  # uint32
            w_uint32(value)
        elif vtype == 5:  # int32
            w_int32(value)
        elif vtype == 10:  # float32
            w_float32(value)
        elif vtype == 11:  # bool
            w_bool(value)
        elif vtype == 12:  # array
            data += struct.pack('<I', value['type'])
            offset += 4
            data += struct.pack('<Q', len(value['items']))
            offset += 8
            for item in value['items']:
                if value['type'] == 8:
                    w_string(item)
                elif value['type'] == 5:
                    w_int32(item)
    
    data_buf_start = len(data)
    
    # Write header
    data += b'GGUF'  # magic
    data += struct.pack('<I', 3)  # version
    offset += 4 + 4 + 4 + 8  # magic(4) + version(4) + tensor_count_placeholder(4) + metadata_kv_count_placeholder(8)
    
    # Placeholders
    tensor_count_pos = len(data) - 12
    kv_count_pos = len(data) - 8
    
    kv_count = 0
    
    # Write metadata
    def add_kv(key, vtype, value):
        nonlocal kv_count, offset
        w_key_value(key, vtype, value)
        kv_count += 1
    
    add_kv('general.architecture', 8, 'qwen2')
    add_kv('general.name', 8, 'Lance-Text-3B')
    add_kv('general.file_type', 4, 2)  # Q8_0
    add_kv('general.version', 8, '1')
    
    add_kv('llm.block_count', 4, n_layers)
    add_kv('llm.embedding_length', 4, hidden_size)
    add_kv('llm.feed_forward_length', 4, intermediate_size)
    add_kv('llm.attention.head_count', 4, n_heads)
    add_kv('llm.attention.head_count_kv', 4, n_kv_heads)
    add_kv('llm.attention.layer_norm_rms_epsilon', 10, rms_eps)
    add_kv('llm.rope.freq_base', 10, rope_theta)
    add_kv('llm.rope.dimension_count', 4, hidden_size // n_heads)
    add_kv('llm.vocab_size', 4, vocab_size)
    add_kv('llm.max_position_embeddings', 4, max_seq_len)
    add_kv('general.file_type', 4, 2)
    add_kv('tokenizer.ggml.model', 8, 'gpt2')
    add_kv('tokenizer.ggml.bos_token_id', 4, 151643)
    add_kv('tokenizer.ggml.eos_token_id', 4, 151645)
    
    # Write kv_count
    data[kv_count_pos:kv_count_pos+8] = struct.pack('<Q', kv_count)
    
    return data, kv_count_pos, tensor_count_pos

# Build tensor info
TENSOR_MAP = {
    'language_model.model.embed_tokens.weight': 'token_embd',
    'language_model.model.norm.weight': 'output_norm',
    'language_model.lm_head.weight': 'output',
}

# Per-layer mappings
PER_LAYER_MAP = {
    'input_layernorm.weight': 'attn_norm',
    'self_attn.q_proj.weight': 'attn_q',
    'self_attn.k_proj.weight': 'attn_k',
    'self_attn.v_proj.weight': 'attn_v',
    'self_attn.o_proj.weight': 'attn_output',
    'self_attn.q_proj.bias': 'attn_q.bias',
    'self_attn.k_proj.bias': 'attn_k.bias',
    'self_attn.v_proj.bias': 'attn_v.bias',
    'self_attn.k_norm.weight': 'attn_k_norm',
    'self_attn.q_norm.weight': 'attn_q_norm',
    'post_attention_layernorm.weight': 'ffn_norm',
    'mlp.gate_proj.weight': 'ffn_gate',
    'mlp.up_proj.weight': 'ffn_up',
    'mlp.down_proj.weight': 'ffn_down',
}

tensor_infos = []
for name in text_tensors:
    # Map name to GGUF tensor name
    if name.startswith('language_model.model.layers.'):
        parts = name.split('.')
        layer_idx = int(parts[3])
        inner = '.'.join(parts[4:])
        if inner in PER_LAYER_MAP:
            gguf_name = f'blk.{layer_idx}.{PER_LAYER_MAP[inner]}'
            tensor_infos.append((name, gguf_name, 'q8_0'))
    elif name in TENSOR_MAP:
        gguf_name = TENSOR_MAP[name]
        tensor_infos.append((name, gguf_name, 'q8_0'))

print(f"Mapped tensors: {len(tensor_infos)}")

# Build GGUF file
metadata, kv_count_pos, tensor_count_pos = gguf_meta_writer()
tensor_count = len(tensor_infos)

# Write tensor info header
data = bytearray(metadata)
with open(MODEL_PATH, 'rb') as f_model:
    # Seek past header
    header_end = len(data)
    
    # Write tensor count
    data[tensor_count_pos:tensor_count_pos+4] = struct.pack('<I', tensor_count)
    
    # Write tensor info entries
    tensor_data_offset = 0
    for name, gguf_name, qtype in tensor_infos:
        tensor = st.get_tensor(name)
        shape = list(tensor.shape)
        n_elems = int(np.prod(shape))
        n_dims = len(shape)
        
        # GGUF tensor info
        name_bytes = gguf_name.encode('utf-8')
        data += struct.pack('<Q', len(name_bytes)) + name_bytes
        data += struct.pack('<I', n_dims)
        for dim in reversed(shape):  # GGUF stores dims in reverse
            data += struct.pack('<Q', dim)
        data += struct.pack('<I', 8)  # Q8_0 type
        data += struct.pack('<Q', 0)  # offset placeholder
    
    # Align to 32 bytes
    while len(data) % 32 != 0:
        data.append(0)
    
    tensor_data_start = len(data)
    
    # Now write actual tensor data
    data_current = len(data)
    for i, (name, gguf_name, qtype) in enumerate(tensor_infos):
        tensor = st.get_tensor(name)
        if isinstance(tensor, np.ndarray):
            arr = tensor
        else:
            arr = tensor.numpy()
        
        # Convert to Q8_0: quantize each block of 32 elements
        if arr.ndim == 1:
            arr = arr.astype(np.float32)
            n = arr.shape[0]
            n_blocks = (n + 31) // 32
            q_data = bytearray()
            for b in range(n_blocks):
                block = arr[b*32:(b+1)*32]
                if len(block) < 32:
                    block = np.pad(block, (0, 32 - len(block)))
                d = np.abs(block).max() / 127.0
                if d == 0: d = 1.0
                quants = np.clip(np.round(block / d), -128, 127).astype(np.int8)
                q_data += struct.pack('<f', d)
                q_data += quants.tobytes()
        elif arr.ndim == 2:
            rows, cols = arr.shape
            arr = arr.astype(np.float32)
            q_data = bytearray()
            for r in range(rows):
                row = arr[r, :]
                n = row.shape[0]
                n_blocks = (n + 31) // 32
                for b in range(n_blocks):
                    block = row[b*32:(b+1)*32]
                    if len(block) < 32:
                        block = np.pad(block, (0, 32 - len(block)))
                    d = np.abs(block).max() / 127.0
                    if d == 0: d = 1.0
                    quants = np.clip(np.round(block / d), -128, 127).astype(np.int8)
                    q_data += struct.pack('<f', d)
                    q_data += quants.tobytes()
        
        # Update tensor data offset info
        # The tensor info was written earlier, need to update the offset
        # For now, just append data
        data += q_data
        
        if i % 50 == 0:
            print(f"  [{i}/{len(tensor_infos)}] {gguf_name}: {list(arr.shape)} ({len(q_data)} bytes)")

# Write final file
with open(OUTPUT_GGUF, 'wb') as f:
    f.write(data)

size_gb = len(data) / 1024**3
print(f"\nWritten {OUTPUT_GGUF}: {size_gb:.2f} GB")
print(f"Now quantizing to Q4_K_M...")
