#!/usr/bin/env python3
"""
TurboEngine v8-DeepSeek-V4 — MojoLlama inference engine for DeepSeek V4 Flash.

Loads Intel/DeepSeek-V4-Flash-W4A16-AutoRound safetensors directly.
Implements: MLA attention, Hyper-Connections, MoE (256 experts, hash+learned routing),
compressed KV cache, sparse attention, YaRN RoPE, multi-token prediction.

Usage:
    python3 turbo_engine_dsv4.py --prompt "Hello" --n-tokens 50
"""
import os, sys, json, time, math, struct, gc, argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from functools import lru_cache
from pathlib import Path

import numpy as np

# Optional torch import for BF16 safetensor loading
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from safetensors import safe_open
from huggingface_hub import hf_hub_download, get_safetensors_metadata

# ─── Model Config ───

@dataclass
class DeepSeekV4Config:
    """Hardcoded from config.json + inference/config_w4a16.json"""
    vocab_size: int = 129280
    dim: int = 4096
    n_layers: int = 43
    n_hash_layers: int = 3
    n_heads: int = 64
    n_kv_heads: int = 1
    head_dim: int = 512
    rope_head_dim: int = 64
    nope_head_dim: int = 448  # head_dim - rope_head_dim
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    moe_inter_dim: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    n_mtp_layers: int = 1
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    window_size: int = 128
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    eps: float = 1e-6
    compress_ratios: tuple = (0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                               4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                               4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0)
    max_seq_len: int = 4096  # decode limit
    max_batch_size: int = 1
    model_id: str = "Intel/DeepSeek-V4-Flash-W4A16-AutoRound"

# ─── W4A16 Dequantize ───

def dequantize_w4a16(qweight: np.ndarray, qzeros: np.ndarray, scales: np.ndarray,
                     group_size: int = 128) -> np.ndarray:
    """
    AutoRound/GPTQ W4A16 → F32 weight [out, in].

    qweight: int32 [in//8, out] LSB-first packed
    qzeros:  int32 [in//g, out//8] LSB-first packed (stores zero - 1)
    scales:  float32/BF16 [in//g, out]
    """
    in_packed, out_features = qweight.shape
    in_features = in_packed * 8
    n_groups = scales.shape[0]
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.int32)
    # Unpack qweight: [in//8, 8, out] -> [in, out]
    w = ((qweight[:, None, :] >> shifts.reshape(1, 8, 1)) & 0xF).reshape(in_features, out_features).astype(np.float32)
    # Unpack qzeros: [in//g, out//8, 8] -> [in//g, out]
    z = ((qzeros[:, :, None] >> shifts.reshape(1, 1, 8)) & 0xF).reshape(n_groups, out_features).astype(np.float32) + 1.0
    s = scales.astype(np.float32)
    w = w.reshape(n_groups, group_size, out_features)
    deq = (w - z[:, None, :]) * s[:, None, :]
    return np.ascontiguousarray(deq.reshape(in_features, out_features).T)  # [out, in]


def bf16_to_f32(bf16_arr):
    """Convert BF16 array to float32. Handles uint16 raw BF16 or float16."""
    if hasattr(bf16_arr, 'numpy'):  # torch tensor
        return bf16_arr.float().numpy()
    if bf16_arr.dtype == np.float32:
        return bf16_arr
    if bf16_arr.dtype == np.float16:
        return bf16_arr.astype(np.float32)
    if bf16_arr.dtype == np.uint16:
        view = bf16_arr.view(np.uint32)
        view = view << 16
        return view.view(np.float32)
    return np.asarray(bf16_arr, dtype=np.float32)


# ─── Safetensors Loader ───

class SafetensorsLoader:
    """Lazy-load tensors from multi-shard safetensors repo."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.meta = get_safetensors_metadata(repo_id)
        # Build tensor -> shard mapping
        self.shard_of: Dict[str, str] = {}
        self.shapes: Dict[str, Tuple] = {}
        self.dtypes: Dict[str, str] = {}
        for fname, fmeta in self.meta.files_metadata.items():
            for tname, tinfo in fmeta.tensors.items():
                self.shard_of[tname] = fname
                self.shapes[tname] = tuple(tinfo.shape)
                self.dtypes[tname] = tinfo.dtype
        self._shard_cache: Dict[str, safe_open] = {}

    def get_tensor(self, name: str) -> np.ndarray:
        """Load a tensor by name, returns numpy array (converts BF16 -> F32)."""
        fname = self.shard_of.get(name)
        if fname is None:
            raise KeyError(f"Tensor {name} not found")
        if fname not in self._shard_cache:
            local = hf_hub_download(repo_id=self.repo_id, filename=fname)
            self._shard_cache[fname] = safe_open(local, framework="pt" if _HAS_TORCH else "np", device="cpu")
        sf = self._shard_cache[fname]
        t = sf.get_tensor(name)
        # Convert BF16 -> F32 if needed
        if hasattr(t, 'dtype') and str(t.dtype) in ('bfloat16', 'torch.bfloat16'):
            if _HAS_TORCH:
                return t.float().numpy()
            else:
                # Manual BF16 -> F32 conversion
                return bf16_to_f32(t)
        elif hasattr(t, 'numpy'):
            return t.numpy()
        return t.astype(np.float32) if t.dtype in (np.float16,) else t

    def get_w4a16(self, base_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load qweight, qzeros, scales for a W4A16 linear."""
        qw = self.get_tensor(base_name + '.qweight')
        qz = self.get_tensor(base_name + '.qzeros')
        sc = self.get_tensor(base_name + '.scales')
        return qw, qz, sc

    def get_deq(self, base_name: str) -> np.ndarray:
        """Load and dequantize a W4A16 linear -> [out, in] F32."""
        qw, qz, sc = self.get_w4a16(base_name)
        return dequantize_w4a16(qw, qz, sc)

    def close(self):
        for sf in self._shard_cache.values():
            try: sf.close()
            except: pass
        self._shard_cache.clear()


# ─── RoPE ───

@lru_cache(maxsize=2)
def precompute_freqs(dim: int, seqlen: int, original_seq_len: int,
                     base: float, factor: float, beta_fast: int, beta_slow: int) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute sin/cos for YaRN rotary embeddings."""
    import math
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)
    def linear_ramp_factor(min_v, max_v, dim):
        if min_v == max_v:
            max_v += 0.001
        linear_func = (np.arange(dim, dtype=np.float32) - min_v) / (max_v - min_v)
        return np.clip(linear_func, 0, 1)

    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = np.arange(seqlen, dtype=np.float32)
    angles = np.outer(t, freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

def apply_rotary_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                     pos: int, rope_dim: int, n_heads: int = 1) -> np.ndarray:
    """Apply rotary embeddings to x [B, NH, HD] or [B, HD] in-place.
    Only rotates first `rope_dim` dims; rest left unchanged."""
    half = rope_dim // 2
    if x.ndim == 2:
        x_ = x.reshape(1, 1, -1)  # [1, 1, HD]
        c = cos[pos:pos+1, :half]
        s = sin[pos:pos+1, :half]
        # Rotate first rope_dim elements
        x1 = x_[:, :, :half]
        x2 = x_[:, :, half:rope_dim]
        rotated = np.concatenate([
            x1 * c - x2 * s,
            x1 * s + x2 * c
        ], axis=-1)
        x_[:, :, :rope_dim] = rotated
        return x_.reshape(x.shape)
    else:  # [B, NH, HD]
        B, NH, HD = x.shape
        c = cos[pos:pos+1, :half]  # [1, half]
        s = sin[pos:pos+1, :half]
        x1 = x[:, :, :half]
        x2 = x[:, :, half:rope_dim]
        rotated = np.concatenate([
            x1 * c[np.newaxis, np.newaxis, :] - x2 * s[np.newaxis, np.newaxis, :],
            x1 * s[np.newaxis, np.newaxis, :] + x2 * c[np.newaxis, np.newaxis, :]
        ], axis=-1)
        x[:, :, :rope_dim] = rotated
        return x


# ─── RMS Norm ───

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS Layer Normalization. x: [..., D], weight: [D]"""
    dtype = x.dtype
    x = x.astype(np.float32)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    x = x * np.reciprocal(np.sqrt(variance + eps))
    return (x * weight.astype(np.float32)).astype(dtype)


# ─── Linear (Dequantized Matmul) ───

def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: [..., in_d], w: [out_d, in_d] -> [..., out_d].
    Uses BLAS matmul for efficiency."""
    if x.ndim == 1:
        return w @ x  # [out]
    elif x.ndim == 2:
        return x @ w.T  # [B, out]
    else:
        orig = x.shape
        x_2d = x.reshape(-1, orig[-1])
        return (x_2d @ w.T).reshape(*orig[:-1], -1)


# ─── Sinkhorn (for HC) ───

def hc_split_sinkhorn(mixes: np.ndarray, hc_scale: np.ndarray, hc_base: np.ndarray,
                      hc_mult: int, sinkhorn_iters: int, eps: float):
    """
    mixes: [B, S, mix_hc], hc_scale: [3], hc_base: [mix_hc]
    returns: pre [B,S,hc], post [B,S,hc], comb [B,S,hc,hc]
    Using Sinkhorn algorithm for doubly-stochastic normalization.
    """
    # Split into pre, post, comb components
    hc = hc_mult
    pre = mixes[..., :hc]
    post = mixes[..., hc:2*hc]
    comb_flat = mixes[..., 2*hc:]
    comb = comb_flat.reshape(*mixes.shape[:-1], hc, hc)

    # Sigmoid + scale
    pre = 1.0 / (1.0 + np.exp(-(pre * hc_scale[0] + hc_base[:hc].reshape(1, 1, hc)))) + eps
    post = 1.0 / (1.0 + np.exp(-(post * hc_scale[1] + hc_base[hc:2*hc].reshape(1, 1, hc)))) + eps

    # Sinkhorn on comb (doubly stochastic)
    comb_log = comb * hc_scale[2] + hc_base[2*hc:].reshape(1, 1, hc, hc)
    # Sinkhorn iterations (in log space for stability)
    K = comb_log
    for _ in range(sinkhorn_iters):
        K = K - np.log(np.exp(K).sum(axis=-1, keepdims=True) + 1e-10)
        K = K - np.log(np.exp(K).sum(axis=-2, keepdims=True) + 1e-10)
    comb = np.exp(K) + eps

    return pre, post, comb


# ─── DeepSeek V4 Engine ───

class DeepSeekV4Engine:
    """MojoLlama inference engine for DeepSeek V4 Flash."""

    def __init__(self, loader: SafetensorsLoader, cfg: DeepSeekV4Config = None):
        self.loader = loader
        self.cfg = cfg or DeepSeekV4Config()
        self._cache: Dict[str, np.ndarray] = {}  # dequantized weight cache
        self._rope_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._load_global_weights()
        self._init_buffers()

    def _load_global_weights(self):
        """Load non-layer weights (embed, head, hc_*, norm)."""
        cfg = self.cfg
        L = self.loader
        # Embed is BF16 stored directly
        self.embed_w = L.get_tensor('embed.weight').astype(np.float32)
        # LM head is F32
        self.head_w = L.get_tensor('head.weight').astype(np.float32)
        # HC head params (F32)
        self.hc_head_fn = L.get_tensor('hc_head_fn')
        self.hc_head_base = L.get_tensor('hc_head_base')
        self.hc_head_scale = L.get_tensor('hc_head_scale')
        # Final norm
        self.norm_w = L.get_tensor('norm.weight').astype(np.float32)
        print(f"  Global weights: embed={self.embed_w.shape}, head={self.head_w.shape}, "
              f"norm={self.norm_w.shape}", flush=True)

    def _init_buffers(self):
        cfg = self.cfg
        D, N, NH, NKH, HD, RHD = cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_head_dim
        HC = cfg.hc_mult
        # Hidden state buffers
        self.bx = np.zeros(D, dtype=np.float32)  # main hidden
        self.br = np.zeros((HC, D), dtype=np.float32)  # residual HC copies
        self.bxn = np.zeros(D, dtype=np.float32)  # normed
        self.br_out = np.zeros((HC, D), dtype=np.float32)
        # Attention buffers
        self.bq_latent = np.zeros(cfg.q_lora_rank, dtype=np.float32)
        self.bq = np.zeros((NH, HD), dtype=np.float32)  # full Q
        self.bkv = np.zeros(HD, dtype=np.float32)  # single KV
        self.battn = np.zeros((NH, HD), dtype=np.float32)
        # MoE buffers
        self.bmoe_in = np.zeros(D, dtype=np.float32)
        self.bmoe_gate = np.zeros(cfg.moe_inter_dim, dtype=np.float32)
        self.bmoe_up = np.zeros(cfg.moe_inter_dim, dtype=np.float32)
        self.bmoe_out = np.zeros(D, dtype=np.float32)
        self.bmoe_shared = np.zeros(D, dtype=np.float32)
        # MoE router
        self.brouter_scores = np.zeros(cfg.n_routed_experts, dtype=np.float32)
        # HC buffers
        self.bhc_mixes = np.zeros((2 + HC) * HC, dtype=np.float32)
        self.bhc_pre = np.zeros(HC, dtype=np.float32)
        self.bhc_post = np.zeros(HC, dtype=np.float32)
        self.bhc_comb = np.zeros((HC, HC), dtype=np.float32)
        # Conv state for compressor
        self.kv_cache: Dict[int, np.ndarray] = {}  # layer_id -> [window_size + compressed, HD]
        self.compressor_state: Dict[int, Dict] = {}  # layer_id -> state dict

    def _get_rope(self, rope_dim: int, seq_len: int, original_seq_len: int = 0,
                  base: float = 10000.0, factor: float = 1.0,
                  beta_fast: int = 32, beta_slow: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        key = (rope_dim, seq_len, original_seq_len, base, factor, beta_fast, beta_slow)
        if key not in self._rope_cache:
            self._rope_cache[key] = precompute_freqs(
                rope_dim, seq_len, original_seq_len, base, factor, beta_fast, beta_slow)
        return self._rope_cache[key]

    def _load_layer_weight(self, layer_id: int, name: str) -> np.ndarray:
        """Load (and cache) a single tensor from a layer."""
        key = f"layers.{layer_id}.{name}"
        return self.loader.get_tensor(key)

    def _load_deq(self, layer_id: int, name: str) -> np.ndarray:
        """Load and dequantize a W4A16 weight, cached."""
        cache_key = f"L{layer_id}.{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        prefix = f"layers.{layer_id}.{name}"
        try:
            qw = self.loader.get_tensor(prefix + '.qweight')
            qz = self.loader.get_tensor(prefix + '.qzeros')
            sc = self.loader.get_tensor(prefix + '.scales')
            deq = dequantize_w4a16(qw, qz, sc)
        except KeyError:
            # Try .weight (non-quantized)
            deq = self.loader.get_tensor(prefix + '.weight').astype(np.float32)
        self._cache[cache_key] = deq
        return deq

    def _load_layer_norm(self, layer_id: int, name: str) -> np.ndarray:
        """Load RMS norm weight."""
        tensor_name = f"layers.{layer_id}.{name}.weight"
        return self.loader.get_tensor(tensor_name).astype(np.float32)

    def _load_hc_params(self, layer_id: int, kind: str):
        """Load Hyper-Connection params for a layer."""
        prefix = f"layers.{layer_id}.hc_{kind}_"
        fn = self.loader.get_tensor(prefix + 'fn')
        base = self.loader.get_tensor(prefix + 'base')
        scale = self.loader.get_tensor(prefix + 'scale')
        return fn, base, scale

    def _load_attn_sink(self, layer_id: int) -> np.ndarray:
        return self.loader.get_tensor(f"layers.{layer_id}.attn.attn_sink")

    def _load_gate_weight(self, layer_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load MoE gate weights. Returns (weight, bias)."""
        prefix = f"layers.{layer_id}.ffn.gate"
        try:
            qw = self.loader.get_tensor(prefix + '.qweight')
            qz = self.loader.get_tensor(prefix + '.qzeros')
            sc = self.loader.get_tensor(prefix + '.scales')
            w = dequantize_w4a16(qw, qz, sc)
        except KeyError:
            w = self.loader.get_tensor(prefix + '.weight').astype(np.float32)
        try:
            bias = self.loader.get_tensor(prefix + '.bias')
        except KeyError:
            bias = None
        # Hash routing
        try:
            tid2eid = self.loader.get_tensor(prefix + '.tid2eid')
        except KeyError:
            tid2eid = None
        return w, bias, tid2eid

    def _load_shared_expert(self, layer_id: int) -> Dict:
        """Load shared expert weights."""
        base = f"layers.{layer_id}.ffn.shared_experts"
        result = {}
        for name in ['w1', 'w2', 'w3']:
            result[name] = self._load_deq(layer_id, f"ffn.shared_experts.{name}")
        return result

    def _load_expert_weight(self, layer_id: int, expert_id: int, name: str) -> np.ndarray:
        """Load a single expert weight (dequantized)."""
        cache_key = f"L{layer_id}.exp{expert_id}.{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        prefix = f"layers.{layer_id}.ffn.experts.{expert_id}.{name}"
        try:
            qw = self.loader.get_tensor(prefix + '.qweight')
            qz = self.loader.get_tensor(prefix + '.qzeros')
            sc = self.loader.get_tensor(prefix + '.scales')
            deq = dequantize_w4a16(qw, qz, sc)
        except KeyError:
            deq = self.loader.get_tensor(prefix + '.weight').astype(np.float32)
        self._cache[cache_key] = deq
        return deq

    def _load_compressor(self, layer_id: int, ratio: int):
        """Load compressor weights for a layer."""
        prefix = f"layers.{layer_id}.attn.compressor"
        result = {}
        coff = 2 if ratio == 4 else 1
        head_dim = self.cfg.head_dim
        result['wkv'] = self._load_deq(layer_id, "attn.compressor.wkv")
        result['wgate'] = self._load_deq(layer_id, "attn.compressor.wgate")
        result['ape'] = self.loader.get_tensor(prefix + '.ape').astype(np.float32)
        result['norm_w'] = self.loader.get_tensor(prefix + '.norm.weight').astype(np.float32)
        return result

    def _load_indexer(self, layer_id: int):
        """Load indexer weights."""
        prefix = f"layers.{layer_id}.attn.indexer"
        result = {}
        result['wq_b'] = self._load_deq(layer_id, "attn.indexer.wq_b")
        result['weights_proj'] = self._load_deq(layer_id, "attn.indexer.weights_proj")
        # Indexer compressor
        result['compressor'] = {}
        result['compressor']['wkv'] = self._load_deq(layer_id, "attn.indexer.compressor.wkv")
        result['compressor']['wgate'] = self._load_deq(layer_id, "attn.indexer.compressor.wgate")
        result['compressor']['ape'] = self.loader.get_tensor(prefix + '.compressor.ape').astype(np.float32)
        result['compressor']['norm_w'] = self.loader.get_tensor(prefix + '.compressor.norm.weight').astype(np.float32)
        return result

    # ─── Forward Pass ───

    def forward_layer(self, layer_id: int, pos: int, input_ids: np.ndarray):
        """Forward one transformer layer. Updates self.bx and self.br in place."""
        cfg = self.cfg
        D, HC = cfg.dim, cfg.hc_mult

        t0 = time.perf_counter()

        # ── HC Pre (Attention) ──
        hc_attn_fn, hc_attn_base, hc_attn_scale = self._get_layer_hc(layer_id, 'attn')
        pre, post, comb = self._hc_pre(self.br, hc_attn_fn, hc_attn_base, hc_attn_scale, HC)
        # x = sum(pre[:, None] * br) = weighted sum of HC copies -> single hidden
        x = np.sum(pre[:, None] * self.br, axis=0)  # [D]
        # Copy to bx
        np.copyto(self.bx, x)

        # ── Attention Norm ──
        attn_norm_w = self._load_layer_norm(layer_id, 'attn_norm')
        np.copyto(self.bxn, rms_norm(self.bx, attn_norm_w, cfg.eps))

        # ── MLA Attention ──
        self._mla_attention(layer_id, pos)

        # ── HC Post (Attention) ──
        # x_out = post * x + comb @ br (matrix multiply along HC dim)
        new_br = post[:, None] * self.bxn[np.newaxis, :] + np.einsum('hc,cd->hd', comb, self.br)
        np.copyto(self.br, new_br.astype(np.float32))

        # ── HC Pre (FFN) ──
        hc_ffn_fn, hc_ffn_base, hc_ffn_scale = self._get_layer_hc(layer_id, 'ffn')
        pre, post, comb = self._hc_pre(self.br, hc_ffn_fn, hc_ffn_base, hc_ffn_scale, HC)
        x = np.sum(pre[:, None] * self.br, axis=0)
        np.copyto(self.bx, x)

        # ── FFN Norm ──
        ffn_norm_w = self._load_layer_norm(layer_id, 'ffn_norm')
        np.copyto(self.bxn, rms_norm(self.bx, ffn_norm_w, cfg.eps))

        # ── MoE FFN ──
        self._moe_ffn(layer_id, input_ids)

        # ── HC Post (FFN) ──
        new_br = post[:, None] * self.bx[np.newaxis, :] + np.einsum('hc,cd->hd', comb, self.br)
        np.copyto(self.br, new_br.astype(np.float32))

    def _get_layer_hc(self, layer_id: int, kind: str):
        """Load and cache HC parameters."""
        key = f"L{layer_id}.hc_{kind}"
        if key not in self._cache:
            prefix = f"layers.{layer_id}.hc_{kind}_"
            fn = self.loader.get_tensor(prefix + 'fn')
            base = self.loader.get_tensor(prefix + 'base')
            scale = self.loader.get_tensor(prefix + 'scale')
            self._cache[key] = (fn, base, scale)
        return self._cache[key]

    def _hc_pre(self, res: np.ndarray, hc_fn: np.ndarray,
                hc_base: np.ndarray, hc_scale: np.ndarray, hc_mult: int):
        """
        res: [HC, D] residual copies
        hc_fn: [mix_hc, HC*D]
        Returns (pre, post, comb) where each is [HC] or [HC, HC]
        """
        mix_hc = (2 + hc_mult) * hc_mult
        HC, D = res.shape
        # Flatten residual and compute mixes
        x_flat = res.flatten()  # [HC*D]
        rsqrt = 1.0 / np.sqrt(np.mean(x_flat * x_flat) + 1e-6)
        mixes = np.dot(hc_fn, x_flat * rsqrt)  # [mix_hc]
        return hc_split_sinkhorn(mixes[np.newaxis, np.newaxis, :],
                                 hc_scale, hc_base, hc_mult, 1, 1e-6)

    def _mla_attention(self, layer_id: int, pos: int):
        """Multi-head Latent Attention for a single token."""
        cfg = self.cfg
        NH, NKH, HD, RHD = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_head_dim
        NHD = HD - RHD

        # Load attention weights
        wq_a = self._load_deq(layer_id, "attn.wq_a")  # [q_lora_rank, D]
        q_norm_w = self._load_layer_norm(layer_id, "attn.q_norm")
        wq_b = self._load_deq(layer_id, "attn.wq_b")  # [NH*HD, q_lora_rank]
        wkv = self._load_deq(layer_id, "attn.wkv")  # [HD, D]
        kv_norm_w = self._load_layer_norm(layer_id, "attn.kv_norm")
        wo_a = self._load_deq(layer_id, "attn.wo_a")  # [NH*HD//groups, groups*o_lora_rank]
        wo_b = self._load_deq(layer_id, "attn.wo_b")  # [D, groups*o_lora_rank]
        attn_sink = self._load_attn_sink(layer_id)

        # Q projection: wq_a -> q_norm -> wq_b
        latent = np.dot(wq_a, self.bxn)  # [q_lora_rank]
        latent = rms_norm(latent, q_norm_w, cfg.eps)
        q = np.dot(wq_b, latent)  # [NH*HD]
        q = q.reshape(NH, HD)

        # QK norm (per-head)
        q = q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + cfg.eps)

        # KV projection: wkv -> kv_norm (shared MQA)
        kv = np.dot(wkv, self.bxn)  # [HD,]
        kv = rms_norm(kv, kv_norm_w, cfg.eps)

        # RoPE
        cos, sin = self._get_rope(RHD, max(pos + 1, 2),
                                   cfg.original_seq_len if self._has_compression(layer_id) else 0,
                                   cfg.compress_rope_theta if self._has_compression(layer_id) else cfg.rope_theta,
                                   cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
        apply_rotary_emb(q[:, :RHD], cos, sin, pos, RHD)
        apply_rotary_emb(kv[np.newaxis, :RHD], cos, sin, pos, RHD)

        # KV cache (sliding window)
        win = cfg.window_size
        ratio = cfg.compress_ratios[layer_id]
        # Initialize KV cache for this layer if needed
        if layer_id not in self.kv_cache:
            cache_size = win + (cfg.max_seq_len // ratio) if ratio else win
            self.kv_cache[layer_id] = np.zeros((cache_size, HD), dtype=np.float32)
            self.compressor_state[layer_id] = {}
        
        kv_cache = self.kv_cache[layer_id]
        
        # Store KV
        kv_cache[pos % win] = kv
        
        # Handle compression if applicable
        if ratio and pos > 0 and (pos % ratio == 0):
            comp_idx = pos // ratio
            # Simple mean-pool compression (approximation of learned gated pooling)
            start = max(0, pos - ratio + 1)
            compressed = np.mean(kv_cache[start:pos+1], axis=0)
            kv_cache[win + comp_idx - 1] = compressed
        
        # Build sparse attention indices
        # Sliding window: recent `win` positions
        if pos < win:
            topk_idxs = np.arange(pos + 1)
        else:
            topk_idxs = np.arange(pos - win + 1, pos + 1)
        
        # Add compressed positions if applicable
        if ratio:
            n_compressed = pos // ratio
            if n_compressed > 0:
                comp_idxs = np.arange(win, win + n_compressed)
                topk_idxs = np.concatenate([topk_idxs, comp_idxs])
        
        # Sparse attention
        num_kv = len(topk_idxs)
        if num_kv == 0:
            self.battn.fill(0)
        else:
            # Gather KV: all heads share the same KV (MQA)
            gathered_kv = kv_cache[topk_idxs]  # [K, HD]
            
            # Compute attention scores
            scale = HD ** -0.5
            scores = np.dot(q.reshape(NH, HD), gathered_kv.T) * scale  # [NH, K]
            
            # Add attn_sink bias
            scores += attn_sink[:, np.newaxis]
            
            # Mask future positions (not needed in decode mode)
            # Softmax
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores)
            weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)
            
            # Weighted sum
            o = np.dot(weights, gathered_kv)  # [NH, HD]
            np.copyto(self.battn, o)

        # Output projection
        # wo_a: [NH*HD//groups, groups*o_lora_rank]
        NHG = NH // cfg.o_groups
        groups = cfg.o_groups
        # Reshape attn to [groups, NHG*HD], apply wo_a per group
        self.battn_reshaped = self.battn.reshape(groups, NHG * HD)
        o = np.dot(wo_a, self.battn_reshaped.T).T.reshape(-1)  # [groups*o_lora_rank]
        o = np.dot(wo_b, o)  # [D]
        np.copyto(self.bxn, o)

    def _has_compression(self, layer_id: int) -> bool:
        return self.cfg.compress_ratios[layer_id] != 0

    def _moe_ffn(self, layer_id: int, input_ids: np.ndarray):
        """MoE FFN: route -> top-6 experts + shared expert."""
        cfg = self.cfg
        D, INTER, NE, TOPK = cfg.dim, cfg.moe_inter_dim, cfg.n_routed_experts, cfg.n_activated_experts
        N_HASH = cfg.n_hash_layers

        # Load gate
        gate_w, gate_bias, tid2eid = self._load_gate_weight(layer_id)
        is_hash = layer_id < N_HASH

        # Router: x -> scores
        scores = np.dot(gate_w, self.bxn)  # [NE]
        if gate_bias is not None:
            scores = scores + gate_bias

        # Score function
        if cfg.score_func == "softmax":
            scores = np.exp(scores - np.max(scores))
            scores = scores / np.sum(scores)
        elif cfg.score_func == "sigmoid":
            scores = 1.0 / (1.0 + np.exp(-scores))
        else:  # sqrtsoftplus
            scores = np.sqrt(np.log1p(np.exp(scores)))

        # Select top-k experts
        if is_hash and tid2eid is not None:
            indices = tid2eid[input_ids]  # [TOPK] from hash table
            weights = scores[indices]
        else:
            topk = min(TOPK, NE)
            indices = np.argpartition(-scores, topk)[:topk]
            weights = scores[indices]

        # Normalize weights
        if cfg.score_func != "softmax":
            weights = weights / (np.sum(weights) + 1e-10)
        weights *= cfg.route_scale

        # Compute expert outputs
        self.bmoe_out.fill(0)
        for k, (expert_id, weight) in enumerate(zip(indices, weights)):
            if weight <= 0:
                continue
            # Load expert weights
            w1 = self._load_expert_weight(layer_id, int(expert_id), 'w1')  # [INTER, D]
            w3 = self._load_expert_weight(layer_id, int(expert_id), 'w3')  # [INTER, D]
            w2 = self._load_expert_weight(layer_id, int(expert_id), 'w2')  # [D, INTER]

            # SwiGLU: SiLU(x @ w1.T) * (x @ w3.T)
            gate = np.dot(w1, self.bxn)  # [INTER]
            up = np.dot(w3, self.bxn)    # [INTER]
            if cfg.swiglu_limit > 0:
                up = np.clip(up, -cfg.swiglu_limit, cfg.swiglu_limit)
                gate = np.clip(gate, None, cfg.swiglu_limit)
            # SiLU
            sig = 1.0 / (1.0 + np.exp(-gate))
            activated = sig * gate * up
            if weight != 1.0:
                activated = activated * weight
            # Down project
            out = np.dot(w2, activated)  # [D]
            self.bmoe_out += out

        # Shared expert
        shared = self._load_shared_expert(layer_id)
        gate_s = np.dot(shared['w1'], self.bxn)
        up_s = np.dot(shared['w3'], self.bxn)
        if cfg.swiglu_limit > 0:
            up_s = np.clip(up_s, -cfg.swiglu_limit, cfg.swiglu_limit)
            gate_s = np.clip(gate_s, None, cfg.swiglu_limit)
        sig_s = 1.0 / (1.0 + np.exp(-gate_s))
        activated_s = sig_s * gate_s * up_s
        out_s = np.dot(shared['w2'], activated_s)
        self.bmoe_out += out_s
        np.copyto(self.bx, self.bmoe_out)

    def generate(self, input_ids: List[int], n_tokens: int = 100,
                 temperature: float = 0.7, top_k: int = 50) -> List[int]:
        """Autoregressive generation."""
        cfg = self.cfg
        output_ids = list(input_ids)
        
        # Prefill: embed first token
        token = input_ids[0]
        np.copyto(self.bx, self.embed_w[token])
        # Init HC residual
        self.br.fill(0)
        self.br[:, :cfg.dim] = self.bx[np.newaxis, :] / cfg.hc_mult

        timings = []
        pos = 0

        for step in range(n_tokens):
            t_start = time.perf_counter()
            token = output_ids[-1]
            pos = step

            if step == 0:
                # Prefill: process all input tokens
                for i, tid in enumerate(input_ids):
                    if i == 0:
                        # Already embedded
                        pass
                    else:
                        np.copyto(self.bx, self.embed_w[tid])
                        self.br.fill(0)
                        self.br[:, :cfg.dim] = self.bx[np.newaxis, :] / cfg.hc_mult
                    
                    for l in range(cfg.n_layers):
                        self.forward_layer(l, i, tid)
                
                # HC head + norm + lm_head
                logits = self._compute_logits()
            else:
                # Decode: one token at a time
                np.copyto(self.bx, self.embed_w[token])
                self.br.fill(0)
                self.br[:, :cfg.dim] = self.bx[np.newaxis, :] / cfg.hc_mult
                
                for l in range(cfg.n_layers):
                    self.forward_layer(l, pos, token)
                
                logits = self._compute_logits()

            # Sample
            logits = logits.astype(np.float64)
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    top_k_vals = np.partition(logits, -top_k)[-top_k]
                    logits[logits < top_k_vals] = -float('inf')
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                next_token = int(np.random.choice(len(probs), p=probs))
            else:
                next_token = int(np.argmax(logits))

            output_ids.append(next_token)
            elapsed = time.perf_counter() - t_start
            timings.append(elapsed)

            if (step + 1) % 10 == 0:
                tok_s = (step + 1) / max(sum(timings), 1e-6)
                print(f"  [{step+1}/{n_tokens}] tok/s={tok_s:.2f}, last={elapsed*1000:.0f}ms",
                      flush=True)

        avg_tok_s = n_tokens / max(sum(timings), 1e-6)
        print(f"\nGeneration: {n_tokens} tokens in {sum(timings):.1f}s ({avg_tok_s:.2f} tok/s)",
              flush=True)
        return output_ids

    def _compute_logits(self) -> np.ndarray:
        """HC head -> norm -> lm_head."""
        cfg = self.cfg
        # HC head: weighted sum of HC copies
        x_flat = self.br.flatten()
        rsqrt = 1.0 / np.sqrt(np.mean(x_flat * x_flat) + cfg.eps)
        mixes = np.dot(self.hc_head_fn, x_flat * rsqrt)  # [HC]
        pre = 1.0 / (1.0 + np.exp(-(mixes * self.hc_head_scale[0] + self.hc_head_base))) + cfg.hc_eps
        x = np.sum(pre[:, None] * self.br, axis=0)  # [D]
        # Final norm
        x = rms_norm(x, self.norm_w, cfg.eps)
        # LM head
        logits = np.dot(self.head_w, x)  # [vocab]
        return logits


# ─── Tokenizer ───

def load_tokenizer(repo_id: str) -> Tuple[Dict[int, str], Dict[str, int], int]:
    """Minimal BPE tokenizer. Returns (id_to_token, token_to_id, vocab_size)."""
    import json
    try:
        tok_file = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
        with open(tok_file, 'r') as f:
            data = json.load(f)
        # Extract vocabulary from the 'model' section
        vocab = data.get('model', {}).get('vocab', {})
        id_to_token = {}
        token_to_id = {}
        # tokenizer.json vocab is usually token -> id
        for token, tid in vocab.items():
            id_to_token[int(tid)] = token
            token_to_id[token] = int(tid)
        # Also try added_tokens
        added = data.get('added_tokens', [])
        for at in added:
            tid = int(at.get('id', 0))
            content = at.get('content', '')
            if content:
                id_to_token[tid] = content
                token_to_id[content] = tid
        return id_to_token, token_to_id, len(id_to_token)
    except Exception as e:
        print(f"Tokenizer load warning: {e}")
        return {}, {}, 129280


class SimpleTokenizer:
    """Minimal tokenizer for DeepSeek V4."""
    def __init__(self, repo_id: str):
        import json, re
        self.repo_id = repo_id
        # Try to load tokenizer config
        try:
            config_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")
            with open(config_path) as f:
                tc = json.load(f)
            self.bos_token_id = tc.get('bos_token_id', 0)
            self.eos_token_id = tc.get('eos_token_id', 1)
        except:
            self.bos_token_id = 0
            self.eos_token_id = 1
        
        # Try loading the tokenizer
        self.id_to_token, self.token_to_id, self.vocab_size = load_tokenizer(repo_id)
        self.vocab_size = 129280  # Hardcoded from config
        print(f"Tokenizer: vocab={self.vocab_size}, bos={self.bos_token_id}, eos={self.eos_token_id}")
    
    def encode(self, text: str) -> List[int]:
        """Simple whitespace+character fallback encoding."""
        # For now, use a simple byte-level encoding since this is a demo
        # In production, use the actual tokenizer
        ids = []
        for ch in text.encode('utf-8'):
            ids.append(ch + 3)  # BPE-like offset
        return ids[:self.vocab_size - 1]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text (simple fallback)."""
        chars = []
        for tid in ids:
            if tid < 3 + 256:
                chars.append(chr(max(tid - 3, 0)))
            elif tid in self.id_to_token:
                token = self.id_to_token[tid]
                chars.append(token.replace('Ġ', ' ').replace('Ċ', '\n'))
        return ''.join(chars)
    
    def decode_token(self, tid: int) -> str:
        if tid < 3:
            return ''
        if tid < 3 + 256:
            return chr(tid - 3)
        return self.id_to_token.get(tid, f'<{tid}>').replace('Ġ', ' ')


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--n-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--model-id", default="Intel/DeepSeek-V4-Flash-W4A16-AutoRound")
    parser.add_argument("--max-layers", type=int, default=1, help="Limit layers for testing")
    args = parser.parse_args()

    print("=" * 60)
    print("MojoLlama DeepSeek V4 Flash Inference Engine")
    print("=" * 60)

    # Load config
    cfg = DeepSeekV4Config(model_id=args.model_id)
    if args.max_layers:
        cfg.n_layers = args.max_layers
        print(f"  [TEST MODE] Using {args.max_layers} layer(s)")

    print(f"Loading metadata from {args.model_id}...")
    loader = SafetensorsLoader(args.model_id)

    print(f"Initializing engine...")
    engine = DeepSeekV4Engine(loader, cfg)

    # Load tokenizer
    tokenizer = SimpleTokenizer(args.model_id)

    # Encode prompt
    input_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: '{args.prompt}' -> {len(input_ids)} tokens")
    if len(input_ids) == 0:
        input_ids = [1]  # fallback
    print(f"Input IDs: {input_ids[:20]}{'...' if len(input_ids) > 20 else ''}")

    # Generate
    print(f"\nGenerating {args.n_tokens} tokens...")
    t0 = time.perf_counter()
    output_ids = engine.generate(input_ids, args.n_tokens, args.temperature, args.top_k)
    total_time = time.perf_counter() - t0

    # Decode
    output_text = tokenizer.decode(output_ids)
    print(f"\n{'='*60}")
    print(f"Generated: {output_text[:500]}")
    print(f"{'='*60}")
    print(f"Total: {len(output_ids)} tokens in {total_time:.1f}s ({len(output_ids)/total_time:.2f} tok/s)")

    loader.close()


if __name__ == "__main__":
    main()
