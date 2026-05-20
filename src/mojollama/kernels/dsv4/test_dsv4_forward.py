#!/usr/bin/env python3
"""Minimal DeepSeek V4 forward pass test - 1 layer, 1 token."""
import sys, os, time, json, argparse
sys.path.insert(0, '/onedev-workspace/work/src')
import numpy as np

from mojollama.kernels.turbo_engine_dsv4 import (
    SafetensorsLoader, DeepSeekV4Config, dequantize_w4a16,
    rms_norm, linear, apply_rotary_emb, precompute_freqs,
    hc_split_sinkhorn
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Skip weight downloads if cached")
    args = parser.parse_args()

    cfg = DeepSeekV4Config(n_layers=args.layers)
    print(f"DeepSeek V4 Flash Test: {cfg.n_layers} layer(s)")
    print(f"  dim={cfg.dim}, n_heads={cfg.n_heads}, head_dim={cfg.head_dim}")
    print(f"  rope_dim={cfg.rope_head_dim}, q_lora={cfg.q_lora_rank}")
    print(f"  experts={cfg.n_routed_experts}, top-k={cfg.n_activated_experts}")
    print(f"  hc_mult={cfg.hc_mult}")

    loader = SafetensorsLoader(cfg.model_id)

    # Load global weights
    print("\nLoading global weights...")
    embed_w = loader.get_tensor('embed.weight').astype(np.float32)
    head_w = loader.get_tensor('head.weight').astype(np.float32)
    norm_w = loader.get_tensor('norm.weight').astype(np.float32)
    hc_head_fn = loader.get_tensor('hc_head_fn')
    hc_head_base = loader.get_tensor('hc_head_base')
    hc_head_scale = loader.get_tensor('hc_head_scale')
    print(f"  embed: {embed_w.shape}, head: {head_w.shape}, norm: {norm_w.shape}")

    # Load per-layer weights (only what we need)
    layer_weights = {}
    for l in range(args.layers):
        print(f"\nLoading layer {l}...")
        lw = {}

        # Norms
        lw['attn_norm'] = loader.get_tensor(f'layers.{l}.attn_norm.weight').astype(np.float32)
        lw['ffn_norm'] = loader.get_tensor(f'layers.{l}.ffn_norm.weight').astype(np.float32)

        # Attention (W4A16 -> dequantized)
        for wname in ['wq_a', 'wkv', 'wo_a', 'wo_b']:
            key = f'layers.{l}.attn.{wname}'
            qw = loader.get_tensor(key + '.qweight')
            qz = loader.get_tensor(key + '.qzeros')
            sc = loader.get_tensor(key + '.scales')
            lw[f'attn_{wname}'] = dequantize_w4a16(qw, qz, sc)
            print(f"    attn.{wname}: {lw[f'attn_{wname}'].shape}")

        # wq_b: [128, 32768] qweight
        key = f'layers.{l}.attn.wq_b'
        qw = loader.get_tensor(key + '.qweight')
        qz = loader.get_tensor(key + '.qzeros')
        sc = loader.get_tensor(key + '.scales')
        lw['attn_wq_b'] = dequantize_w4a16(qw, qz, sc)
        print(f"    attn.wq_b: {lw['attn_wq_b'].shape}")

        # q_norm, kv_norm
        lw['q_norm'] = loader.get_tensor(f'layers.{l}.attn.q_norm.weight').astype(np.float32)
        lw['kv_norm'] = loader.get_tensor(f'layers.{l}.attn.kv_norm.weight').astype(np.float32)
        lw['attn_sink'] = loader.get_tensor(f'layers.{l}.attn.attn_sink').astype(np.float32)
        print(f"    attn_sink: {lw['attn_sink'].shape}, q_norm: {lw['q_norm'].shape}")

        # Gate (W4A16 or F32)
        try:
            key = f'layers.{l}.ffn.gate'
            qw = loader.get_tensor(key + '.qweight')
            qz = loader.get_tensor(key + '.qzeros')
            sc = loader.get_tensor(key + '.scales')
            lw['gate_w'] = dequantize_w4a16(qw, qz, sc)
        except:
            lw['gate_w'] = loader.get_tensor(f'layers.{l}.ffn.gate.weight').astype(np.float32)
        try:
            lw['gate_bias'] = loader.get_tensor(f'layers.{l}.ffn.gate.bias')
        except:
            lw['gate_bias'] = None
        try:
            lw['tid2eid'] = loader.get_tensor(f'layers.{l}.ffn.gate.tid2eid')
        except:
            lw['tid2eid'] = None
        print(f"    gate_w: {lw['gate_w'].shape}, bias={'yes' if lw['gate_bias'] is not None else 'no'}, hash={'yes' if lw['tid2eid'] is not None else 'no'}")

        # Shared expert
        for wname in ['w1', 'w2', 'w3']:
            key = f'layers.{l}.ffn.shared_experts.{wname}'
            qw = loader.get_tensor(key + '.qweight')
            qz = loader.get_tensor(key + '.qzeros')
            sc = loader.get_tensor(key + '.scales')
            lw[f'shared_{wname}'] = dequantize_w4a16(qw, qz, sc)
            print(f"    shared.{wname}: {lw[f'shared_{wname}'].shape}")

        # HC params
        for kind in ['attn', 'ffn']:
            prefix = f'layers.{l}.hc_{kind}_'
            lw[f'hc_{kind}_fn'] = loader.get_tensor(prefix + 'fn')
            lw[f'hc_{kind}_base'] = loader.get_tensor(prefix + 'base')
            lw[f'hc_{kind}_scale'] = loader.get_tensor(prefix + 'scale')
            print(f"    hc_{kind}_fn: {lw[f'hc_{kind}_fn'].shape}")

        layer_weights[l] = lw

    print("\n" + "=" * 60)
    print("Running forward pass...")
    print("=" * 60)

    D = cfg.dim
    HC = cfg.hc_mult
    NH = cfg.n_heads
    HD = cfg.head_dim
    RHD = cfg.rope_head_dim

    # Init hidden state
    token = 1  # arbitrary token
    bx = embed_w[token].copy()
    br = np.zeros((HC, D), dtype=np.float32)
    br[:] = bx[np.newaxis, :] / HC

    # KV cache
    kv_cache = np.zeros((cfg.window_size, HD), dtype=np.float32)

    t_start = time.perf_counter()

    for l in range(args.layers):
        lw = layer_weights[l]

        # ── HC Pre (Attention) ──
        # mixes = hc_fn @ flatten(br) * rsqrt
        x_flat = br.flatten()
        rsqrt = 1.0 / np.sqrt(np.mean(x_flat * x_flat) + 1e-6)
        mixes = np.dot(lw['hc_attn_fn'], x_flat * rsqrt)
        pre, post, comb = hc_split_sinkhorn(
            mixes[np.newaxis, np.newaxis, :],
            lw['hc_attn_scale'], lw['hc_attn_base'], HC, 1, 1e-6)
        pre = pre[0, 0]; post = post[0, 0]; comb = comb[0, 0]
        # Weighted sum
        x = np.sum(pre[:, None] * br, axis=0)

        # ── Attention Norm ──
        x = rms_norm(x, lw['attn_norm'], cfg.eps)

        # ── MLA Attention ──
        # Q: wq_a -> q_norm -> wq_b -> reshape
        latent = np.dot(lw['attn_wq_a'], x)
        latent = rms_norm(latent, lw['q_norm'], cfg.eps)
        q = np.dot(lw['attn_wq_b'], latent)
        q = q.reshape(NH, HD)

        # QK norm
        q = q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + cfg.eps)

        # KV: wkv -> kv_norm
        kv = np.dot(lw['attn_wkv'], x)
        kv = rms_norm(kv, lw['kv_norm'], cfg.eps)

        # RoPE
        rope_dim = RHD
        cos, sin = precompute_freqs(rope_dim, 2, 0, cfg.rope_theta, 1.0, 32, 1)
        apply_rotary_emb(q[:, :rope_dim], cos, sin, 0, rope_dim, NH)
        apply_rotary_emb(kv[np.newaxis, :rope_dim], cos, sin, 0, rope_dim, 1)

        # Sliding window attention (pos=0, no compression)
        kv_cache[0] = kv
        # For pos=0, attention is over just this token
        scale = HD ** -0.5
        scores = np.dot(q, kv) * scale  # [NH]
        scores += lw['attn_sink']
        # Softmax
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        weights = exp_scores / (np.sum(exp_scores) + 1e-10)
        o = weights[:, np.newaxis] * kv[np.newaxis, :]  # [NH, HD]

        # Output projection
        # wo_a: [groups*o_lora_rank, NHG*HD] = [8192, 4096] block-diagonal: 8 blocks of [1024, 4096]
        # wo_b: [D, groups*o_lora_rank] = [4096, 8192]
        groups, o_lora = cfg.o_groups, cfg.o_lora_rank
        NHG = NH // groups  # heads per group = 8
        o_r = o.reshape(groups, NHG * HD)  # [8, 4096]
        o_proj = np.zeros(groups * o_lora, dtype=np.float32)
        for g in range(groups):
            w_g = lw['attn_wo_a'][g * o_lora:(g + 1) * o_lora, :]  # [1024, 4096]
            o_proj[g * o_lora:(g + 1) * o_lora] = w_g @ o_r[g]  # [1024]
        o_out = np.dot(lw['attn_wo_b'], o_proj)  # [D]

        # ── HC Post (Attention) ──
        br_new = post[:, None] * o_out[np.newaxis, :] + np.dot(comb, br)
        br = br_new.astype(np.float32)

        # ── HC Pre (FFN) ──
        x_flat = br.flatten()
        rsqrt = 1.0 / np.sqrt(np.mean(x_flat * x_flat) + 1e-6)
        mixes = np.dot(lw['hc_ffn_fn'], x_flat * rsqrt)
        pre, post, comb = hc_split_sinkhorn(
            mixes[np.newaxis, np.newaxis, :],
            lw['hc_ffn_scale'], lw['hc_ffn_base'], HC, 1, 1e-6)
        pre = pre[0, 0]; post = post[0, 0]; comb = comb[0, 0]
        x = np.sum(pre[:, None] * br, axis=0)

        # ── FFN Norm ──
        x = rms_norm(x, lw['ffn_norm'], cfg.eps)

        # ── MoE FFN (just shared expert for first test) ──
        # Gate: compute scores (top-1 for simplicity)
        scores = np.dot(lw['gate_w'], x)
        if lw['gate_bias'] is not None:
            scores += lw['gate_bias']
        # sqrtsoftplus
        scores = np.sqrt(np.log1p(np.exp(scores)))
        topk = min(cfg.n_activated_experts, cfg.n_routed_experts)
        indices = np.argpartition(-scores, topk)[:topk]
        weights = scores[indices]
        weights = weights / (np.sum(weights) + 1e-10)
        weights *= cfg.route_scale

        moe_out = np.zeros(D, dtype=np.float32)
        # Only load and compute activated experts
        for k, (eid, w) in enumerate(zip(indices, weights)):
            if w <= 0:
                continue
            # Load expert weights on-demand
            key_w1 = f'layers.{l}.ffn.experts.{int(eid)}.w1'
            key_w2 = f'layers.{l}.ffn.experts.{int(eid)}.w2'
            key_w3 = f'layers.{l}.ffn.experts.{int(eid)}.w3'
            qw1 = loader.get_tensor(key_w1 + '.qweight')
            qz1 = loader.get_tensor(key_w1 + '.qzeros')
            sc1 = loader.get_tensor(key_w1 + '.scales')
            w1 = dequantize_w4a16(qw1, qz1, sc1)
            qw2 = loader.get_tensor(key_w2 + '.qweight')
            qz2 = loader.get_tensor(key_w2 + '.qzeros')
            sc2 = loader.get_tensor(key_w2 + '.scales')
            w2 = dequantize_w4a16(qw2, qz2, sc2)
            qw3 = loader.get_tensor(key_w3 + '.qweight')
            qz3 = loader.get_tensor(key_w3 + '.qzeros')
            sc3 = loader.get_tensor(key_w3 + '.scales')
            w3 = dequantize_w4a16(qw3, qz3, sc3)

            gate_act = np.dot(w1, x)
            up_act = np.dot(w3, x)
            if cfg.swiglu_limit > 0:
                up_act = np.clip(up_act, -cfg.swiglu_limit, cfg.swiglu_limit)
                gate_act = np.clip(gate_act, None, cfg.swiglu_limit)
            sig = 1.0 / (1.0 + np.exp(-gate_act))
            activated = sig * gate_act * up_act
            if w != 1.0:
                activated *= w
            moe_out += np.dot(w2, activated)

        # Shared expert
        gate_s = np.dot(lw['shared_w1'], x)
        up_s = np.dot(lw['shared_w3'], x)
        if cfg.swiglu_limit > 0:
            up_s = np.clip(up_s, -cfg.swiglu_limit, cfg.swiglu_limit)
            gate_s = np.clip(gate_s, None, cfg.swiglu_limit)
        sig_s = 1.0 / (1.0 + np.exp(-gate_s))
        activated_s = sig_s * gate_s * up_s
        moe_out += np.dot(lw['shared_w2'], activated_s)

        # ── HC Post (FFN) ──
        br_new = post[:, None] * moe_out[np.newaxis, :] + np.dot(comb, br)
        br = br_new.astype(np.float32)

        elapsed = time.perf_counter() - t_start
        print(f"  Layer {l}: {elapsed*1000:.0f}ms, bx_norm={np.linalg.norm(x):.4f}, "
              f"moe_out_norm={np.linalg.norm(moe_out):.4f}, "
              f"br_norm={np.linalg.norm(br):.4f}", flush=True)

    # ── HC Head + Norm + LM Head ──
    x_flat = br.flatten()
    rsqrt = 1.0 / np.sqrt(np.mean(x_flat * x_flat) + cfg.eps)
    mixes = np.dot(hc_head_fn, x_flat * rsqrt)
    pre = 1.0 / (1.0 + np.exp(-(mixes * hc_head_scale[0] + hc_head_base))) + cfg.hc_eps
    x = np.sum(pre[:, None] * br, axis=0)
    x = rms_norm(x, norm_w, cfg.eps)
    logits = np.dot(head_w, x)

    elapsed = time.perf_counter() - t_start
    print(f"\nLogits: shape={logits.shape}, range=[{logits.min():.4f}, {logits.max():.4f}]")
    top5 = np.argsort(-logits)[:5]
    print(f"Top-5 tokens: {top5.tolist()}")
    print(f"Top-5 logits: {logits[top5].tolist()}")
    print(f"Total forward time: {elapsed*1000:.0f}ms")

    loader.close()
    print("\nSUCCESS!")

if __name__ == "__main__":
    main()
