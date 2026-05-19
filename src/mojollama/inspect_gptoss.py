#!/usr/bin/env python3
"""Inspect GPT-OSS GGUF tensor names."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
import gguf

r = gguf.GGUFReader("/tmp/models/gpt-oss-20b-Q4_K_M.gguf")
print("Fields:")
for k, v in r.fields.items():
    parts = v.parts if hasattr(v, 'parts') else v
    print(f"  {k}: {parts}")

print("\n\nTensor names (first 100):")
for i, t in enumerate(r.tensors):
    if i >= 100:
        print(f"  ... ({len(r.tensors)} total)")
        break
    print(f"  {t.name}  shape={list(t.shape)}  type={t.tensor_type}")
