#!/usr/bin/env python3
"""Check GPT-OSS MXFP4 tensor format."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
import gguf
from gguf.constants import GGMLQuantizationType as QT
import numpy as np

r = gguf.GGUFReader("/tmp/models/gpt-oss-20b-Q4_K_M.gguf")

# Find an MXFP4 expert tensor
for t in r.tensors:
    if 'ffn_gate_exps' in t.name and 'blk.0' in t.name:
        qtype = QT(t.tensor_type).value
        print(f"Tensor: {t.name}")
        print(f"  Shape: {list(t.shape)}")
        print(f"  Type: {qtype} ({t.tensor_type})")
        data = np.array(t.data)
        print(f"  Raw data shape: {data.shape}")
        print(f"  Raw data dtype: {data.dtype}")
        print(f"  Raw data min/max: {data.min()}/{data.max()}")
        print(f"  First 32 bytes: {data[:32].tolist()}")
        
        # Try dequantizing
        f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
        print(f"  Dequantized shape: {f32.shape}")
        print(f"  Dequantized min/max: {f32.min():.4f}/{f32.max():.4f}")
        print(f"  Dequantized first 5 values: {f32.ravel()[:5].tolist()}")
        break

# Also check output weight
for t in r.tensors:
    if t.name == 'output.weight':
        qtype = QT(t.tensor_type).value
        print(f"\nOutput: shape={list(t.shape)}, type={qtype}")
        break

# Check attn_q weight
for t in r.tensors:
    if 'attn_q.weight' in t.name and 'blk.0' in t.name:
        qtype = QT(t.tensor_type).value
        print(f"\nattn_q: shape={list(t.shape)}, type={qtype}")
        f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
        print(f"  Dequantized min/max: {f32.min():.4f}/{f32.max():.4f}")
        break
