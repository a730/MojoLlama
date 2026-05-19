#!/usr/bin/env python3
"""Verify all modified modules load correctly."""
import sys
sys.path.insert(0, '/onedev-workspace/work/src')

errors = []

try:
    from mojollama.quantizer import _read_gguf_metadata
    print("✓ quantizer.py - _read_gguf_metadata")
except Exception as e:
    errors.append(f"quantizer.py: {e}")

try:
    from mojollama.model.inference import LLMInference
    print("✓ inference.py - LLMInference")
except Exception as e:
    errors.append(f"inference.py: {e}")

try:
    from mojollama.server import MojoLlamaHandler
    print("✓ server.py - MojoLlamaHandler")
except Exception as e:
    errors.append(f"server.py: {e}")

if errors:
    print(f"\n❌ {len(errors)} errors:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("\n✅ All modules load successfully")
