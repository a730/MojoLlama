#!/usr/bin/env python3
"""Quick check: engine attributes for output weight."""
import sys
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE

e = TurboEngineV7MoE('/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf', 1)

print("Engine attributes for output weight:")
for attr in ['_or', '_out_raw', '_out_f32', 'out_w_name', '_out_qt', '_out_nr', '_out_nc']:
    exists = hasattr(e, attr)
    val = getattr(e, attr, 'N/A')
    print(f"  {attr}: exists={exists}, value={val}")
    
# Check what _out_raw looks like
if hasattr(e, '_out_raw'):
    print(f"  _out_raw type: {type(e._out_raw)}")
    print(f"  _out_raw value: {e._out_raw}")
