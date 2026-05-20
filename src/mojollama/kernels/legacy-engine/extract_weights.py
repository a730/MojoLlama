#!/usr/bin/env python3
"""MojoLlama Weight Extractor — dumps GGUF weights to binary/.npy format.

Usage:
  python3 extract_weights.py <model.gguf> [--dir /dev/shm/mojo_weights/]

Output: /dev/shm/mojo_weights/<layer>_<name>.npy

Format: numpy .npy files, uint8 for Q4_0 weights, float32 for norms/embed
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path


def extract(gguf_path, out_dir="/dev/shm/mojo_weights"):
    import gguf
    from gguf.constants import GGMLQuantizationType
    
    reader = gguf.GGUFReader(gguf_path)
    os.makedirs(out_dir, exist_ok=True)
    
    f = reader.get_field
    arch_field = f('general.architecture')
    if arch_field:
        arch = bytes(arch_field.parts[-1]).decode('utf-8')
    else:
        # Fallback: detect from field names
        arch = None
        for k in reader.fields.keys():
            if k.endswith('.block_count'):
                arch = k.replace('.block_count', '')
                break
        if arch is None:
            print("Could not determine architecture!")
            return
    
    prefix = f'{arch}.'
    
    n_layers = int(f(f'{prefix}block_count').parts[-1].item())
    n_embd = int(f(f'{prefix}embedding_length').parts[-1].item())
    n_head = int(f(f'{prefix}attention.head_count').parts[-1].item())
    n_kv_head = int(f(f'{prefix}attention.head_count_kv').parts[-1].item())
    n_ff = int(f(f'{prefix}feed_forward_length').parts[-1].item())
    head_dim = n_embd // n_head
    
    info = {
        "arch": arch,
        "n_layers": n_layers,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_ff": n_ff,
        "head_dim": head_dim,
        "n_kv": n_kv_head * head_dim,
    }
    
    with open(os.path.join(out_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Extracting {gguf_path} → {out_dir}/")
    print(f"  {n_layers} layers, {n_embd} dim, {n_ff} FF")
    
    weight_names = [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
    ]
    
    for i in range(n_layers):
        weight_names.extend([
            f"blk.{i}.attn_norm.weight",
            f"blk.{i}.ffn_norm.weight",
            f"blk.{i}.attn_q.weight",
            f"blk.{i}.attn_k.weight",
            f"blk.{i}.attn_v.weight",
            f"blk.{i}.attn_output.weight",
            f"blk.{i}.ffn_gate.weight",
            f"blk.{i}.ffn_up.weight",
            f"blk.{i}.ffn_down.weight",
        ])
    
    t0 = time.time()
    for name in weight_names:
        tensor = None
        for t in reader.tensors:
            if t.name == name:
                tensor = t
                break
        
        if tensor is None:
            print(f"  ⚠ missing: {name}")
            continue
        
        tt = tensor.tensor_type
        data = np.asarray(tensor.data)
        
        if tt == GGMLQuantizationType.Q4_0:
            # Save raw uint8 blocks
            raw = data.tobytes() if data.dtype == np.uint8 else data.astype(np.uint8).tobytes()
            out_path = os.path.join(out_dir, name.replace('.', '_') + ".q4")
            with open(out_path, "wb") as f:
                f.write(raw)
            # Also save as .npy for easy loading
            arr = np.frombuffer(raw, dtype=np.uint8)
            np.save(os.path.join(out_dir, name.replace('.', '_') + ".npy"), arr)
        else:
            # F32
            f32 = data.astype(np.float32) if data.dtype != np.float32 else data
            f32 = np.ascontiguousarray(f32)
            np.save(os.path.join(out_dir, name.replace('.', '_') + ".npy"), f32)
        
        if i % 4 == 0 and 'blk.' in name:
            print(f"  {name}: {data.shape} → saved")
    
    elapsed = time.time() - t0
    print(f"\nDone. {elapsed:.1f}s, {len(weight_names)} tensors extracted.")
    print(f"Weight info saved to {out_dir}/model_info.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_weights.py <model.gguf> [--dir <out_dir>]")
        sys.exit(1)
    
    out_dir = "/dev/shm/mojo_weights"
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--dir" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--dir") + 1]
    
    extract(paths[0], out_dir)
