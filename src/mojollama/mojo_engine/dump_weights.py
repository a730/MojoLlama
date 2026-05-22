# MojoLlama Weight Dumper — exports GGUF weights as raw binary
# One-time use per model. Output: directory of .bin files that
# the Mojo engine loads directly (no Python dependency at runtime).
import sys, os, struct, numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE

def dump_model(model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    e = TurboEngineV7MoE(model_path, n_threads=8)
    
    # Dump metadata
    meta = {
        'n_embd': e.n_embd, 'n_layers': e.n_layers,
        'n_head': e.n_head, 'n_kv_head': e.n_kv_head,
        'head_dim': e.head_dim, 'vocab_size': e.vocab_size,
        'n_ff': e.n_ff, 'n_experts': e.n_experts,
        'n_experts_per_tok': e.n_experts_per_tok,
        'n_ff_expert': e.n_ff_expert,
        'is_moe': e.is_moe, 'eps': e.eps,
    }
    with open(f'{out_dir}/meta.bin', 'wb') as f:
        for k, v in meta.items():
            f.write(f'{k}={v}\n'.encode())
    
    # Dump embedding
    e.emb.astype(np.float32).tofile(f'{out_dir}/emb.bin')
    
    # Dump output norm
    e._out_norm_w.astype(np.float32).tofile(f'{out_dir}/out_norm.bin')
    
    # Dump per-layer weights
    for i in range(e.n_layers):
        lw = e._layers[i]
        ld = f'{out_dir}/layer_{i}'
        os.makedirs(ld, exist_ok=True)
        
        # Attention norm
        lw.attn_norm_w.astype(np.float32).tofile(f'{ld}/attn_norm.bin')
        
        # Q projection (quantized)
        if hasattr(lw, 'attn_q_raw') and lw.attn_q_raw is not None:
            np.array([lw.attn_q_nr.value, lw.attn_q_nc.value, lw.attn_q_qt.value],
                     dtype=np.int32).tofile(f'{ld}/q_info.bin')
            raw = lw.attn_q_raw
            if hasattr(raw, 'tobytes'):
                raw.tofile(f'{ld}/q_weight.bin')
        
        # K projection
        if hasattr(lw, 'attn_k_raw') and lw.attn_k_raw is not None:
            np.array([lw.attn_k_nr.value, lw.attn_k_nc.value, lw.attn_k_qt.value],
                     dtype=np.int32).tofile(f'{ld}/k_info.bin')
            raw = lw.attn_k_raw
            if hasattr(raw, 'tobytes'):
                raw.tofile(f'{ld}/k_weight.bin')
        
        # Output projection
        if hasattr(lw, 'attn_out_raw') and lw.attn_out_raw is not None:
            np.array([lw.attn_out_nr.value, lw.attn_out_nc.value, lw.attn_out_qt.value],
                     dtype=np.int32).tofile(f'{ld}/o_info.bin')
            raw = lw.attn_out_raw
            if hasattr(raw, 'tobytes'):
                raw.tofile(f'{ld}/o_weight.bin')
        
        # Q norm (Gemma4)
        if lw.has_q_norm:
            lw.q_norm_w.astype(np.float32).tofile(f'{ld}/q_norm.bin')
        if lw.has_k_norm:
            lw.k_norm_w.astype(np.float32).tofile(f'{ld}/k_norm.bin')
        
        # FFN norms
        if lw.ffn_norm_w is not None:
            lw.ffn_norm_w.astype(np.float32).tofile(f'{ld}/ffn_norm.bin')
        
        # MoE weights
        me = e._moe_layers[i]
        if me.router_f32 is not None:
            me.router_f32.astype(np.float32).tofile(f'{ld}/router.bin')
        
        # Expert weights (MXFP4)
        for name, attr in [('gate', 'gate_raw'), ('up', 'up_raw'), ('down', 'down_raw')]:
            raw_list = getattr(me, attr, None)
            if raw_list and len(raw_list) > 0:
                # First expert's raw data
                raw0 = raw_list[0]
                if hasattr(raw0, 'tobytes') or hasattr(raw0, 'tobytes'):
                    data = np.frombuffer(raw0, dtype=np.uint8) if hasattr(raw0, 'tobytes') else raw0
                    # Get per-expert stride
                    nr = getattr(me, f'{name}_nr', 0)
                    nc = getattr(me, f'{name}_nc', 0)
                    qt = getattr(me, f'{name}_qt', 0)
                    if hasattr(nr, 'value'): nr = nr.value
                    if hasattr(nc, 'value'): nc = nc.value
                    if hasattr(qt, 'value'): qt = qt.value
                    np.array([len(raw_list), nr, nc, qt], dtype=np.int32).tofile(f'{ld}/{name}_info.bin')
                    # Dump all experts concatenated
                    all_exp = np.concatenate([np.frombuffer(r, dtype=np.uint8) for r in raw_list])
                    all_exp.tofile(f'{ld}/{name}_exps.bin')
    
    print(f'Dumped to {out_dir}')
    print(f'  Layers: {e.n_layers}')
    print(f'  Embedding: {e.emb.shape}')
    print(f'  Output norm: {e._out_norm_w.shape}')

if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else '/tmp/models/gpt-oss-20b-Q4_K_M.gguf'
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/mojo_weights/gpt-oss'
    dump_model(model, out)
