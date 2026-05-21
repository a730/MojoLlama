#!/usr/bin/env python3
"""Fix relative paths in benchmark scripts to point to kernels/."""
import os, sys

scripts = ['bench_batch_qwen36.py', 'bench_concurrent_qwen36.py']
for s in scripts:
    f = os.path.join(os.path.dirname(__file__), s)
    with open(f) as fh:
        content = fh.read()
    old = "'kernels'"
    new = "'..', '..', 'kernels'"
    if old in content:
        content = content.replace(old, new)
        with open(f, 'w') as fh:
            fh.write(content)
        print(f"Fixed {s}")
    else:
        print(f"No change in {s}")
