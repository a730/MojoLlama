#!/usr/bin/env python3
"""Test _read_gguf_metadata on available GGUF models."""
import sys
sys.path.insert(0, '/onedev-workspace/work/src')
from mojollama.quantizer import _read_gguf_metadata
import os

models_dir = '/onedev-workspace/work'
found = 0
for f in sorted(os.listdir(models_dir)):
    if f.endswith('.gguf'):
        path = os.path.join(models_dir, f)
        try:
            meta = _read_gguf_metadata(path)
            arch = meta.get('architecture', '?')
            params = meta.get('n_params_hr', '?')
            ctx = meta.get('context_length', '?')
            blk = meta.get('block_count', '?')
            quant = meta.get('quant_type', '?')
            prefixes_used = meta.get('_prefixes_used', [])
            print(f'  {f}:')
            print(f'    arch={arch}, params={params}, ctx={ctx}, layers={blk}, quant={quant}')
            print(f'    prefixes={prefixes_used}')
            found += 1
        except Exception as e:
            print(f'  {f}: ERROR - {e}')

if found == 0:
    print('  No .gguf models found')
else:
    print(f'\n  Read {found} models successfully')
