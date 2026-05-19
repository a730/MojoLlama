#!/usr/bin/env python3
"""Extract Lance text backbone, rename to standard Qwen2 naming, convert to GGUF."""
import sys, os, json, subprocess, numpy as np
import safetensors

SRC = "/tmp/models/lance_model.safetensors"
OUT_DIR = "/tmp/models/lance_text_backbone"
os.makedirs(OUT_DIR, exist_ok=True)

# Load and extract text-only tensors
st = safetensors.safe_open(SRC, framework='np')
all_keys = st.keys()

tensors = {}
n_total = len(all_keys)
n_skipped = 0

for name in all_keys:
    # Skip non-text tensors
    if '_moe_gen' in name: n_skipped += 1; continue
    if 'vae2llm' in name or 'llm2vae' in name: n_skipped += 1; continue
    if 'time_embedder' in name: n_skipped += 1; continue
    if 'latent_pos_embed' in name: n_skipped += 1; continue
    if 'language_model.' not in name: n_skipped += 1; continue
    
    # Skip Q/K norms (Qwen2 VL only, not standard Qwen2)
    if '_norm' in name and ('k_norm' in name or 'q_norm' in name):
        n_skipped += 1; continue
    
    # Skip biases (standard Qwen2/LLaMA has no QKV biases)
    if '.bias' in name: n_skipped += 1; continue
    
    # Rename: language_model.model.layers.N.XXX -> model.layers.N.XXX
    new_name = name.replace('language_model.', '', 1)
    tensors[new_name] = st.get_tensor(name)

print(f"Extracted {len(tensors)} text tensors (skipped {n_skipped})")

from safetensors.numpy import save_file
# Save as safetensors
out_path = f"{OUT_DIR}/model.safetensors"
save_file(tensors, out_path)
import os
size_gb = os.path.getsize(out_path) / 1024**3
print(f"Saved to {out_path} ({size_gb:.2f} GB)")

# Create config.json for converter compatibility
config = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 2048,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "intermediate_size": 11008,
    "vocab_size": 151936,
    "max_position_embeddings": 128000,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
    "torch_dtype": "bfloat16",
    "use_cache": True,
}

with open(f"{OUT_DIR}/config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Copy tokenizer files
import shutil
for fn in ['tokenizer.json', 'tokenizer_config.json']:
    src_fn = f"/tmp/models/lance_{fn}" if fn != 'tokenizer_config.json' else f"/tmp/models/lance_tokenizer.json"
    # Create simple tokenizer_config
    if fn == 'tokenizer_config.json':
        tc = {"tokenizer_class": "Qwen2Tokenizer", "bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"}
        with open(f"{OUT_DIR}/{fn}", 'w') as f: json.dump(tc, f)
    elif os.path.exists(src_fn):
        shutil.copy2(src_fn, f"{OUT_DIR}/{fn}")

# Also check for specific HF-required files
for fn in ['vocab.json', 'merges.txt', 'added_tokens.json', 'special_tokens_map.json']:
    src = f"/tmp/models/lance_{fn}"
    dst = f"{OUT_DIR}/{fn}"
    if os.path.exists(src):
        shutil.copy2(src, dst)

print(f"Config and tokenizer copied to {OUT_DIR}")
print(f"\nNow run conversion:\n"
      f"  python3 /tmp/llama.cpp/convert_hf_to_gguf.py {OUT_DIR} \\\n"
      f"    --outtype q8_0 --outfile /tmp/models/lance-text-q8_0.gguf")
