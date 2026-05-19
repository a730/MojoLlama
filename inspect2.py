#!/usr/bin/env python3
import gguf
import numpy as np

reader = gguf.GGUFReader('/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf')

# Find ALL ssm-related tensors 
count = 0
for t in reader.tensors:
    name = t.name
    if 'ssm' in name.lower() or 'attn_' in name.lower():
        print(f'{name}: shape={list(t.shape)} type={t.tensor_type}')
        count += 1
        if count > 50:
            break
