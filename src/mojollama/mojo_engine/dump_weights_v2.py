#!/usr/bin/env python3
"""MojoLlama Weight Dumper v2 — dumps quantized weight arrays from raw_weights dict.
Usage: python3 dump_weights.py [model.gguf] [out_dir]"""
import sys, os, numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
sys.path.insert(0, '/onedev-workspace/work/src/mojollama')
from turbo_engine_v7_moe import TurboEngineV7MoE

model = sys.argv[1] if len(sys.argv) > 1 else '/tmp/models/gpt-oss-20b-Q4_K_M.gguf'
out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/mojo_weights/gpt-oss'
e = TurboEngineV7MoE(model, n_threads=8)
rw = e.raw_weights
V = int(e.vocab_size)
L = int(e.n_layers)
N = int(e.n_embd)

# Output norm
e._out_norm_w.astype(np.float32).tofile(f'{out}/out_norm.bin')
print(f"out_norm: {e._out_norm_w.shape}")

# Per-layer
for i in range(L):
    ld = f'{out}/layer_{i}'
    os.makedirs(ld, exist_ok=True)
    
    # Norms
    e._layers[i].attn_norm_w.astype(np.float32).tofile(f'{ld}/attn_norm.bin')
    e._layers[i].ffn_norm_w.astype(np.float32).tofile(f'{ld}/ffn_norm.bin')
    
    # Q — from raw_weights dict
    q_key = f'blk.{i}.attn_q.weight'
    if q_key in rw:
        data = rw[q_key]
        data.tofile(f'{ld}/q_weight.bin')
        lw = e._layers[i]
        nr = int(getattr(lw.attn_q_nr, 'value', lw.attn_q_nr))
        nc = int(getattr(lw.attn_q_nc, 'value', lw.attn_q_nc))
        qt_val = lw.attn_q_qt
        if isinstance(qt_val, bytes): 
            qt_val = int.from_bytes(qt_val, 'little')
        else:
            qt_val = int(getattr(qt_val, 'value', qt_val))
        np.array([nr, nc, qt_val], dtype=np.int32).tofile(f'{ld}/q_info.bin')
        print(f"  L{i} Q: {data.nbytes}B ({nr}x{nc} qt={qt_val})")
    
    # K
    k_key = f'blk.{i}.attn_k.weight'
    if k_key in rw:
        data = rw[k_key]
        data.tofile(f'{ld}/k_weight.bin')
        lw = e._layers[i]
        nr = int(getattr(lw.attn_k_nr, 'value', lw.attn_k_nr))
        nc = int(getattr(lw.attn_k_nc, 'value', lw.attn_k_nc))
        qt_val = lw.attn_k_qt
        if isinstance(qt_val, bytes): 
            qt_val = int.from_bytes(qt_val, 'little')
        else:
            qt_val = int(getattr(qt_val, 'value', qt_val))
        np.array([nr, nc, qt_val], dtype=np.int32).tofile(f'{ld}/k_info.bin')
        print(f"  L{i} K: {data.nbytes}B ({nr}x{nc} qt={qt_val})")
    
    # V
    v_key = f'blk.{i}.attn_v.weight'
    if v_key in rw:
        data = rw[v_key]
        data.tofile(f'{ld}/v_weight.bin')
        lw = e._layers[i]
        nr = int(getattr(lw.attn_v_nr, 'value', lw.attn_v_nr))
        nc = int(getattr(lw.attn_v_nc, 'value', lw.attn_v_nc))
        qt_val = lw.attn_v_qt
        if isinstance(qt_val, bytes): 
            qt_val = int.from_bytes(qt_val, 'little')
        else:
            qt_val = int(getattr(qt_val, 'value', qt_val))
        np.array([nr, nc, qt_val], dtype=np.int32).tofile(f'{ld}/v_info.bin')
        print(f"  L{i} V: {data.nbytes}B ({nr}x{nc} qt={qt_val})")
    
    # O
    o_key = f'blk.{i}.attn_output.weight'
    if o_key in rw:
        data = rw[o_key]
        data.tofile(f'{ld}/o_weight.bin')
        lw = e._layers[i]
        nr = int(getattr(lw.attn_out_nr, 'value', lw.attn_out_nr))
        nc = int(getattr(lw.attn_out_nc, 'value', lw.attn_out_nc))
        qt_val = lw.attn_out_qt
        if isinstance(qt_val, bytes): 
            qt_val = int.from_bytes(qt_val, 'little')
        else:
            qt_val = int(getattr(qt_val, 'value', qt_val))
        np.array([nr, nc, qt_val], dtype=np.int32).tofile(f'{ld}/o_info.bin')
        print(f"  L{i} O: {data.nbytes}B ({nr}x{nc} qt={qt_val})")
    
    # MoE router
    me = e._moe_layers[i]
    if me.router_f32 is not None:
        me.router_f32.astype(np.float32).tofile(f'{ld}/router.bin')
    
    # MoE experts
    for name, attr_key in [('gate', 'ffn_gate_exps'), ('up', 'ffn_up_exps'), ('down', 'ffn_down_exps')]:
        ek = f'blk.{i}.{attr_key}.weight'
        if ek in rw:
            data = rw[ek]
            data.tofile(f'{ld}/{name}_exps.bin')
            nr = getattr(me, f'{name}_nr', 0)
            nc = getattr(me, f'{name}_nc', 0)
            qt = getattr(me, f'{name}_qt', 0)
            if hasattr(nr, 'value'): nr = nr.value
            if hasattr(nc, 'value'): nc = nc.value
            if hasattr(qt, 'value'): qt = qt.value
            ne = int(data.nbytes / (nr * nc)) if nr > 0 and nc > 0 else 0
            np.array([ne, nr, nc, qt], dtype=np.int32).tofile(f'{ld}/{name}_info.bin')
            print(f"  L{i} {name}: {data.nbytes}B ({ne}exp x {nr}x{nc} qt={qt})")

print(f"\nDone! Weights dumped to {out}")
