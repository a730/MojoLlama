#!/usr/bin/env python3
"""Patch the engine to load original MXFP4 GGUF with full name aliasing."""
import re, numpy as np, ctypes, gguf
from gguf.constants import GGMLQuantizationType as QT

def patch_zaya_engine(engine):
    """Call after engine init to alias MXFP4 tensor names to llama.cpp format."""
    if engine.arch_name != 'zaya':
        return
    if 'model.layers.0.input_norm.weight' not in engine.weights:
        return  # Not MXFP4 format
    
    print("  MXFP4 naming adapter: creating full alias map...", flush=True)
    n_layer = engine.n_layers
    n_exp = engine.n_experts
    
    # 1) Global tensors
    gmap = {
        'model.embed_tokens.weight': 'token_embd.weight',
        'model.final_norm.weight': 'output_norm.weight',
    }
    for src, dst in gmap.items():
        if src in engine.weights and dst not in engine.weights:
            engine.weights[dst] = engine.weights[src]
        if src in engine.raw_weights and dst not in engine.raw_weights:
            engine.raw_weights[dst] = engine.raw_weights[src]
            if src in engine.weight_info:
                engine.weight_info[dst] = engine.weight_info[src]
    
    # 2) Global res_scale
    for sn in ['hidden_states_scale','hidden_states_bias','residual_scale','residual_bias']:
        src = f'model.res_scale.{sn}'
        if src in engine.weights:
            d32 = engine.weights[src].ravel()
            pref = 'res_scale_hs' if 'hidden' in sn else 'res_scale_res'
            suff = 'weight' if 'scale' in sn else 'bias'
            engine.weights[f'{pref}.{suff}'] = d32
    
    # 3) Per-layer tensors
    for l in range(n_layer):
        ps = f'model.layers.{l}'
        pd = f'blk.{l}'
        even = l % 2 == 0
        
        # input_norm → attn_norm
        src = f'{ps}.input_norm.weight'
        dst = f'{pd}.attn_norm.weight'
        if src in engine.weights and dst not in engine.weights:
            engine.weights[dst] = engine.weights[src]
        
        if even:
            # Q/K/O projections
            for s, d in [('self_attn.qkv.linear_q','attn_q'),
                          ('self_attn.qkv.linear_k','attn_k'),
                          ('self_attn.o_proj','attn_output')]:
                _alias(engine, f'{ps}.{s}.weight', f'{pd}.{d}.weight')
            
            # CCA value projections
            for i in [1,2]:
                _alias(engine, f'{ps}.self_attn.qkv.val_proj{i}.weight', f'{pd}.cca_val_proj{i}.weight')
            
            # SSM conv1d
            for sf in ['weight','bias']:
                _alias(engine, f'{ps}.self_attn.qkv.conv_qk.0.{sf}', f'{pd}.ssm_conv1d.{sf}')
            
            # Res scale (even layers only have hs)
            for sn in ['hidden_states_scale','hidden_states_bias']:
                _copy_scale(engine, f'{ps}.res_scale.{sn}', f'{pd}.res_scale_hs', sn)
        else:
            # Router weights (different MXFP4 naming)
            rmap = [
                ('zaya_block.router.rmsnorm_eda.weight', 'ffn_norm.weight'),
                ('zaya_block.router.down_proj.weight', 'ffn_gate_inp.weight'),
                ('zaya_block.router.down_proj.bias', 'ffn_gate_inp.bias'),
                ('zaya_block.router.router_mlp.0.weight', 'ffn_gate.weight'),
                ('zaya_block.router.router_mlp.0.bias', 'ffn_gate.bias'),
                ('zaya_block.router.router_mlp.2.weight', 'zaya_router_mlp2.weight'),
                ('zaya_block.router.router_mlp.2.bias', 'zaya_router_mlp2.bias'),
                ('zaya_block.router.router_mlp.4.weight', 'zaya_router_mlp4.weight'),
            ]
            for s, d in rmap:
                _alias(engine, f'{ps}.{s}', f'{pd}.{d}')
            
            # balancing_biases → zaya_router_biases (first 17 of 32)
            src = f'{ps}.zaya_block.router.balancing_biases'
            dst = f'{pd}.zaya_router_biases.weight'
            if src in engine.weights and dst not in engine.weights:
                w = engine.weights[src]
                engine.weights[dst] = w.ravel()[:17] if w.ndim > 1 else w[:17]
            
            # Res scales
            for sn in ['hidden_states_scale','hidden_states_bias','residual_scale','residual_bias']:
                pref = 'res_scale_hs' if 'hidden' in sn else 'res_scale_res'
                _copy_scale(engine, f'{ps}.res_scale.{sn}', f'{pd}.{pref}', sn)
            
            # Per-expert weights
            # For each expert, create aliases
            for e in range(n_exp):
                for src_w, dst_w in [('linear_fc1', 'ffn_gate_up_exps'), ('linear_fc2', 'ffn_down_exps')]:
                    src = f'{ps}.zaya_block.experts.local_experts.{e}.{src_w}.weight'
                    dst = f'{pd}.{dst_w}.{e}.weight'
                    _alias_raw(engine, src, dst)
    
    print(f"  Alias mapping complete for {n_layer} layers x {n_exp} experts", flush=True)

def _alias(engine, src, dst):
    """Create an alias from src to dst in weights and raw_weights."""
    if src in engine.weights and dst not in engine.weights:
        engine.weights[dst] = engine.weights[src]
    if src in engine.raw_weights and dst not in engine.raw_weights:
        engine.raw_weights[dst] = engine.raw_weights[src]
        if src in engine.weight_info:
            engine.weight_info[dst] = engine.weight_info[src]

def _alias_raw(engine, src, dst):
    """Create alias specifically for raw_weights with proper weight_info."""
    if src in engine.raw_weights and dst not in engine.raw_weights:
        engine.raw_weights[dst] = engine.raw_weights[src]
        if src in engine.weight_info:
            engine.weight_info[dst] = engine.weight_info[src]

def _copy_scale(engine, src, dst_base, sn):
    """Copy a res_scale tensor."""
    if src in engine.weights:
        d32 = engine.weights[src].ravel()
        suff = 'weight' if 'scale' in sn else 'bias'
        dst = f'{dst_base}.{suff}'
        if dst not in engine.weights:
            engine.weights[dst] = d32
