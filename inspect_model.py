#!/usr/bin/env python3
import gguf
import numpy as np
import sys

reader = gguf.GGUFReader('/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf')

# List all tensor names for layer 0 only (fast iteration with break)
count = 0
for t in reader.tensors:
    name = t.name
    if name.startswith('blk.0.'):
        print(f'{name}: shape={list(t.shape)} type={t.tensor_type}')
        count += 1
    if count >= 20:
        break

print()

# Check ssm_a values
for t in reader.tensors:
    if t.name == 'blk.0.ssm_a':
        data = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
        print(f'SSM_A: {data}')
        print(f'  shape: {data.shape}')
        print(f'  exp(-exp(A)): {np.exp(-np.exp(data))}')
        break

# Check ssm_dt.bias
for t in reader.tensors:
    if t.name == 'blk.0.ssm_dt.bias':
        data = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
        print(f'ssm_dt.bias: {data}')
        print(f'  shape: {data.shape}')
        break

# Check ssm_norm.weight
for t in reader.tensors:
    if t.name == 'blk.0.ssm_norm.weight':
        data = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
        print(f'ssm_norm.weight: {data}')
        print(f'  shape: {data.shape}')
        break

# Check layer 3 (attention layer) tensors
print()
count = 0
for t in reader.tensors:
    name = t.name
    if name.startswith('blk.3.'):
        print(f'L3 {name}: shape={list(t.shape)} type={t.tensor_type}')
        count += 1
    if count >= 20:
        break
