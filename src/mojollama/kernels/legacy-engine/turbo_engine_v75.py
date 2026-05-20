#!/usr/bin/env python3
"""TurboEngine v7.5 — v7 + AVX2 v2 kernels for all Q4_0/Q4_1 matmuls.

Key change: uses quant_matmul_v2 (AVX2 dequant+FMA) for Q4_0/Q4_1 matmuls
instead of quant_matmul_omp (scalar OMP). Falls back to OMP for K-quants.

Expected: 1.37x speedup on Q4_0 matmuls → ~15-20% overall = 84-88 tok/s
"""

import numpy as np
import ctypes
import os
import time
import gguf
from gguf.constants import GGMLQuantizationType as QT

GGML_F32  = 0; GGML_F16 = 1; GGML_Q4_0 = 2; GGML_Q4_1 = 3
GGML_Q8_0 = 8; GGML_Q4_K = 12; GGML_Q5_K = 13; GGML_Q6_K = 14

QTYPE_NAMES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",12:"Q4_K",8:"Q8_0",13:"Q5_K",14:"Q6_K"}
BLOCK_SIZES = {2:18, 3:20, 12:144, 13:176, 14:210, 8:34}
BLOCK_VALS  = {2:32, 3:32, 12:256, 13:256, 14:256, 8:32}
C_KERNEL_TYPES = {GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0}
# Types that have AVX2 v2 kernels
AVX2_TYPES = {GGML_Q4_0, GGML_Q4_1, GGML_Q8_0}


class LayerWeights:
    __slots__ = ['attn_q_raw', 'attn_k_raw', 'attn_v_raw', 'attn_out_raw',
                 'ffn_gate_raw', 'ffn_up_raw', 'ffn_down_raw',
                 'attn_q_nr', 'attn_q_nc', 'attn_q_qt',
                 'attn_k_nr', 'attn_k_nc', 'attn_k_qt',
                 'attn_v_nr', 'attn_v_nc', 'attn_v_qt',
                 'attn_out_nr', 'attn_out_nc', 'attn_out_qt',
                 'ffn_gate_nr', 'ffn_gate_nc', 'ffn_gate_qt',
                 'ffn_up_nr', 'ffn_up_nc', 'ffn_up_qt',
                 'ffn_down_nr', 'ffn_down_nc', 'ffn_down_qt',
                 'attn_norm_w', 'ffn_norm_w',
                 'attn_q_f32', 'attn_k_f32', 'attn_v_f32', 'attn_out_f32',
                 'ffn_gate_f32', 'ffn_up_f32', 'ffn_down_f32',
                 'attn_q_use_c', 'attn_k_use_c', 'attn_v_use_c', 'attn_out_use_c',
                 'ffn_gate_use_c', 'ffn_up_use_c', 'ffn_down_use_c',
                 'q_norm_w', 'k_norm_w',
                 'has_q_norm', 'has_k_norm',
                 'attn_q_use_v2', 'attn_k_use_v2', 'attn_v_use_v2',
                 'attn_out_use_v2', 'ffn_gate_use_v2', 'ffn_up_use_v2', 'ffn_down_use_v2']


class TurboEngineV75:
    """v7.5 — Zero-allocation + AVX2 v2 kernels for Q4_0/Q4_1 matmuls."""

    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        self.n_threads = n_threads
        self.reader = gguf.GGUFReader(model_path)
        self._parse_metadata()
        self._load_weights()
        self._load_kernels()
        self._preload_layer_weights()
        self._init_kv_cache()
        self.reset()
        moe_str = f'/MoE-{self.n_experts}x{self.n_experts_per_tok}' if self.is_moe else ''
        print(f"TurboEngine v7.5: {self.n_layers}L/{self.n_embd}D/"
              f"{self.n_ff}FF/{self.n_head}H/{self.n_kv_head}KV"
              f"{moe_str} | t={self.n_threads} | vocab={self.vocab_size}")

    def _parse_metadata(self):
        fields = self.reader.fields
        def _get(name):
            for key, val in fields.items():
                if key == name:
                    parts = val.parts if hasattr(val, 'parts') else []
                    if len(parts) >= 1:
                        data = parts[-1]
                        if hasattr(data, '__iter__') and len(data) == 1:
                            return int(data[0])
                        return data
        arch = 'llama'
        for prefix in ['qwen3moe', 'qwen2moe', 'llama', 'mistral']:
            for key in fields:
                if f'{prefix}.block_count' in key:
                    arch = prefix; break
        self.n_layers = int(_get(f'{arch}.block_count') or 16)
        self.n_embd = int(_get(f'{arch}.embedding_length') or 2048)
        self.n_ff = int(_get(f'{arch}.feed_forward_length') or self.n_embd * 4)
        self.n_head = int(_get(f'{arch}.attention.head_count') or 32)
        self.n_kv_head = int(_get(f'{arch}.attention.head_count_kv') or self.n_head)
        self.head_dim = int(_get(f'{arch}.attention.key_length') or (self.n_embd // self.n_head))
        self.rope_freq_base = float(_get(f'{arch}.rope.freq_base') or 10000.0)
        self.eps = float(_get(f'{arch}.attention.layer_norm_rms_epsilon') or 1e-6)
        self.vocab_size = int(_get(f'{arch}.vocab_size') or 0)
        self.n_experts = _get(f'{arch}.expert_count')
        self.n_experts_per_tok = _get(f'{arch}.expert_used_count')
        self.is_moe = self.n_experts is not None
        if not self.is_moe:
            self.n_experts = 1; self.n_experts_per_tok = 1
        else:
            self.n_experts = int(self.n_experts); self.n_experts_per_tok = int(self.n_experts_per_tok)
        self.arch_prefix = arch

    def _load_weights(self):
        self.weights = {}; self.raw_weights = {}; self.weight_qtypes = {}; self.weight_info = {}
        for t in self.reader.tensors:
            name = t.name; qtype = QT(t.tensor_type).value
            self.weight_qtypes[name] = qtype
            if len(t.shape) == 2:
                in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(out_dim, in_dim)
                self.weights[name] = np.ascontiguousarray(f32)
                if qtype in C_KERNEL_TYPES:
                    raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8).copy()
                    self.raw_weights[name] = raw
                    self.weight_info[name] = (out_dim, in_dim, BLOCK_SIZES[qtype], qtype)
            elif len(t.shape) == 3:
                in_dim, out_dim, n_exp = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(n_exp, out_dim, in_dim)
                self.weights[name] = np.ascontiguousarray(f32)
            elif len(t.shape) == 1:
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
                self.weights[name] = np.ascontiguousarray(f32)
        self.out_w_name = 'output.weight' if 'output.weight' in self.weights else 'token_embd.weight'
        emb = self.weights['token_embd.weight']
        if emb.ndim == 2 and emb.shape[1] != self.n_embd:
            emb = np.ascontiguousarray(emb.T)
            self.weights['token_embd.weight'] = emb
        if self.vocab_size == 0: self.vocab_size = emb.shape[0]
        self.emb = emb

    def _load_kernels(self):
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        cf = ctypes.POINTER(ctypes.c_float); cu = ctypes.POINTER(ctypes.c_uint8); ci = ctypes.c_int

        # AVX2 v2 kernels (fast for Q4_0/Q4_1/Q8_0)
        v2_path = os.path.join(kernel_dir, 'turbo_kernels_v2.so')
        if os.path.exists(v2_path):
            v2 = ctypes.CDLL(v2_path)
            v2.quant_matmul_v2.argtypes = [cu, cf, cf, ci, ci, ci]
            v2.quant_matmul_v2.restype = None
            self._v2 = v2
        else:
            self._v2 = None

        # OMP kernels (fallback for all types including K-quants)
        omp_path = os.path.join(kernel_dir, 'quant_kernels_omp.so')
        if os.path.exists(omp_path):
            omp = ctypes.CDLL(omp_path)
            omp.quant_matmul_omp.argtypes = [cu, cf, cf, ci, ci, ci]
            omp.quant_matmul_omp.restype = None
            omp.f32_matmul_omp.argtypes = [cf, cf, cf, ci, ci]
            omp.f32_matmul_omp.restype = None
            omp.batch_qkv_omp.argtypes = [cu, cu, cu, cf, cf, cf, cf, ci, ci, ci, ci, ci, ci]
            omp.batch_qkv_omp.restype = None
            omp.batch_gate_up_omp.argtypes = [cu, cu, cf, cf, cf, ci, ci, ci, ci]
            omp.batch_gate_up_omp.restype = None
            omp.set_num_threads.argtypes = [ci]; omp.set_num_threads.restype = None
            omp.set_num_threads(self.n_threads)
            self._kern = omp
        else:
            self._kern = None

        # SIMD ops
        simd_path = os.path.join(kernel_dir, 'simd_ops.so')
        if os.path.exists(simd_path):
            simd = ctypes.CDLL(simd_path)
            simd.rms_norm.argtypes = [cf, cf, cf, ci, ctypes.c_float]
            simd.rms_norm.restype = None
            simd.silu.argtypes = [cf, cf, ci]; simd.silu.restype = None
            simd.residual_add.argtypes = [cf, cf, cf, ci]; simd.residual_add.restype = None
            self._simd = simd
        else:
            self._simd = None

    def _matmul_any(self, raw_ptr, x_ptr, out_ptr, nr, nc, qt, use_v2):
        """Dispatch to fastest kernel: AVX2 v2 for Q4_0/Q4_1/Q8_0, OMP for rest."""
        if use_v2 and self._v2 is not None:
            self._v2.quant_matmul_v2(raw_ptr, x_ptr, out_ptr, nr, nc, qt)
        else:
            self._kern.quant_matmul_omp(raw_ptr, x_ptr, out_ptr, nr, nc, qt)

    def _preload_layer_weights(self):
        self._layers = []
        for i in range(self.n_layers):
            lw = LayerWeights()
            pfx = f'blk.{i}'
            proj_names = {
                'attn_q': 'attn_q', 'attn_k': 'attn_k', 'attn_v': 'attn_v',
                'attn_out': 'attn_output',
                'ffn_gate': 'ffn_gate', 'ffn_up': 'ffn_up', 'ffn_down': 'ffn_down',
            }
            for attr, wname_suffix in proj_names.items():
                name = f'{pfx}.{wname_suffix}.weight'
                info = self.weight_info.get(name)
                if info is not None:
                    raw = self.raw_weights[name]
                    setattr(lw, f'{attr}_raw', raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
                    setattr(lw, f'{attr}_nr', ctypes.c_int(info[0]))
                    setattr(lw, f'{attr}_nc', ctypes.c_int(info[1]))
                    qt = info[3]
                    setattr(lw, f'{attr}_qt', ctypes.c_int(qt))
                    setattr(lw, f'{attr}_use_c', True)
                    # Use AVX2 v2 for Q4_0/Q4_1/Q8_0
                    setattr(lw, f'{attr}_use_v2', qt in AVX2_TYPES)
                else:
                    setattr(lw, f'{attr}_f32', self.weights[name])
                    setattr(lw, f'{attr}_use_c', False)
                    setattr(lw, f'{attr}_use_v2', False)
            lw.attn_norm_w = self.weights[f'{pfx}.attn_norm.weight']
            lw.ffn_norm_w = self.weights[f'{pfx}.ffn_norm.weight']
            lw.has_q_norm = f'{pfx}.attn_q_norm.weight' in self.weights
            lw.has_k_norm = f'{pfx}.attn_k_norm.weight' in self.weights
            if lw.has_q_norm: lw.q_norm_w = self.weights[f'{pfx}.attn_q_norm.weight']
            if lw.has_k_norm: lw.k_norm_w = self.weights[f'{pfx}.attn_k_norm.weight']
            self._layers.append(lw)

        # Output projection — optimize K-quants to Q8_0
        out_info = self.weight_info.get(self.out_w_name)
        if out_info is not None:
            self._out_nr = ctypes.c_int(out_info[0])
            self._out_nc = ctypes.c_int(out_info[1])
            self._out_qt_orig = ctypes.c_int(out_info[3])
            if out_info[3] in (6, 14, 12, 10):
                print(f"  Optimizing output projection: type {out_info[3]} → Q8_0")
                self._out_raw = self._requantize_output(self.raw_weights[self.out_w_name], out_info)
                self._out_qt = ctypes.c_int(8)
                self._out_use_v2 = True
                self._out_use_c = True
            else:
                self._out_raw = self.raw_weights[self.out_w_name].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                self._out_qt = self._out_qt_orig
                self._out_use_v2 = int(out_info[3]) in AVX2_TYPES
                self._out_use_c = True
        else:
            self._out_f32 = self.weights[self.out_w_name]
            self._out_use_c = False
            self._out_use_v2 = False
        self._out_norm_w = self.weights['output_norm.weight']

    def _requantize_output(self, raw_weights, out_info):
        n_rows, n_cols = out_info[0], out_info[1]
        n_blocks = n_cols // 32; orig_qt = out_info[3]
        f32 = np.zeros((n_rows, n_cols), dtype=np.float32)
        t0 = __import__('time').perf_counter()
        if orig_qt == 14:
            n_blk = n_cols // 256
            self._kern.q6_k_dequantize_row.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            self._kern.q6_k_dequantize_row.restype = None
            raw_ptr = raw_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            base_addr = ctypes.addressof(raw_ptr.contents)
            for r in range(n_rows):
                ptr = ctypes.cast(base_addr + r * n_blk * 210, ctypes.POINTER(ctypes.c_uint8))
                self._kern.q6_k_dequantize_row(
                    ptr, f32[r].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(n_cols))
        else:
            raise NotImplementedError(f"Requantize from type {orig_qt}")
        t_deq = __import__('time').perf_counter() - t0
        t0 = __import__('time').perf_counter()
        blocks = f32.reshape(n_rows, n_blocks, 32)
        amax = np.max(np.abs(blocks), axis=2, keepdims=True)
        amax = np.clip(amax, 1e-10, None)
        d = amax / 127.0
        qs = np.clip(np.round(blocks / d), -127, 127).astype(np.int8)
        d_f16 = d[:, :, 0].astype(np.float16); d_u16 = d_f16.view(np.uint16)
        q8 = np.zeros((n_rows, n_blocks, 34), dtype=np.uint8)
        q8[:, :, 0] = d_u16 & 0xFF; q8[:, :, 1] = (d_u16 >> 8) & 0xFF
        q8[:, :, 2:] = qs.view(np.uint8)
        q8_flat = q8.reshape(n_rows * n_blocks * 34)
        q8_contiguous = np.ascontiguousarray(q8_flat, dtype=np.uint8)
        t_req = __import__('time').perf_counter() - t0
        q8_mb = len(q8_contiguous) / 1024 / 1024
        q6_mb = n_rows * (n_cols // 256) * 210 / 1024 / 1024
        print(f"  Dequantize Q6_K→F32: {t_deq:.2f}s, Requantize F32→Q8_0: {t_req:.2f}s")
        print(f"  Q8_0 output: {n_rows}×{n_cols} = {q8_mb:.0f} MB")
        self._out_q8_buf = q8_contiguous
        return q8_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    def _init_kv_cache(self):
        N = self.n_embd; NKH = self.n_kv_head; HD = self.head_dim; MAX_POS = 4096
        self.kv_k = np.zeros((self.n_layers, MAX_POS, NKH*HD), dtype=np.float32)
        self.kv_v = np.zeros((self.n_layers, MAX_POS, NKH*HD), dtype=np.float32)
        self.kv_len = np.zeros(self.n_layers, dtype=np.int32)
        NK = self.n_head * HD
        self._x = np.zeros(N, dtype=np.float32)
        self._x_norm = np.zeros(N, dtype=np.float32)
        self._residual = np.zeros(N, dtype=np.float32)
        self._q = np.zeros(NK, dtype=np.float32)
        self._k = np.zeros(NKH*HD, dtype=np.float32)
        self._v = np.zeros(NKH*HD, dtype=np.float32)
        self._att_out = np.zeros(NK, dtype=np.float32)
        self._gate = np.zeros(self.n_ff, dtype=np.float32)
        self._up = np.zeros(self.n_ff, dtype=np.float32)
        self._silu_gate = np.zeros(self.n_ff, dtype=np.float32)
        self._o_proj = np.zeros(N, dtype=np.float32)
        self._ffn_out = np.zeros(N, dtype=np.float32)
        self._logits = np.zeros(self.vocab_size, dtype=np.float32)
        self._p_x = self._x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_x_norm = self._x_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_residual = self._residual.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_q = self._q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_k = self._k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_v = self._v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_att_out = self._att_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_gate = self._gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_up = self._up.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_silu_gate = self._silu_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_o_proj = self._o_proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_ffn_out = self._ffn_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._p_logits = self._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        HD = self.head_dim; half = HD // 2
        freq = self.rope_freq_base ** (np.arange(0, HD, 2, dtype=np.float32) / HD)
        self._rope_cos_table = {}
        self._rope_sin_table = {}
        self._eps_f = ctypes.c_float(self.eps)
        if self.n_head != self.n_kv_head:
            self._gqa_rep = self.n_head // self.n_kv_head
        else:
            self._gqa_rep = 1

    def reset(self):
        self.kv_len[:] = 0; self.pos = 0

    def _c_rms_norm(self, out, x, w, n):
        self._simd.rms_norm(out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, self._eps_f)

    def _c_silu(self, out, x, n):
        self._simd.silu(out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)

    def _apply_rope_fast(self, x, pos, n_heads):
        hd = self.head_dim; half = hd // 2
        if pos not in self._rope_cos_table:
            freq = self.rope_freq_base ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
            angle = pos / freq
            self._rope_cos_table[pos] = np.cos(angle).astype(np.float32)
            self._rope_sin_table[pos] = np.sin(angle).astype(np.float32)
        cos_a = self._rope_cos_table[pos]; sin_a = self._rope_sin_table[pos]
        x2d = x.reshape(n_heads, hd)
        out = x2d.copy()
        out[:, :half] = x2d[:, :half] * cos_a - x2d[:, half:] * sin_a
        out[:, half:] = x2d[:, half:] * cos_a + x2d[:, :half] * sin_a
        return out.reshape(-1)

    def forward(self, token_id):
        b_x = self._x; b_r = self._residual; b_qn = self._x_norm
        b_q = self._q; b_k = self._k; b_v = self._v
        b_att = self._att_out; b_gate = self._gate; b_up = self._up
        b_silu = self._silu_gate; b_ffn = self._ffn_out; b_oproj = self._o_proj
        N = self.n_embd; NH = self.n_head; NKH = self.n_kv_head; HD = self.head_dim
        nk = NH * HD; FF = self.n_ff; L = self.n_layers
        kern = self._kern; simd = self._simd; eps_f = self._eps_f
        p_x = self._p_x; p_xn = self._p_x_norm; p_r = self._p_residual
        p_q = self._p_q; p_k = self._p_k; p_v = self._p_v
        p_att = self._p_att_out; p_gate = self._p_gate; p_up = self._p_up
        p_silu = self._p_silu_gate; p_oproj = self._p_o_proj
        p_ffn = self._p_ffn_out; p_logits = self._p_logits

        # Embed
        np.copyto(b_x, self.emb[token_id])

        for i in range(L):
            lw = self._layers[i]

            # Residual
            np.copyto(b_r, b_x)

            # RMS norm 1
            simd.rms_norm(p_xn, p_x,
                          lw.attn_norm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          N, eps_f)

            # QKV — always uses OMP batch (handles mixed types)
            if lw.attn_q_use_c and lw.attn_k_use_c and lw.attn_v_use_c:
                kern.batch_qkv_omp(
                    lw.attn_q_raw, lw.attn_k_raw, lw.attn_v_raw,
                    p_xn, p_q, p_k, p_v,
                    lw.attn_q_nr, lw.attn_k_nr, lw.attn_v_nr,
                    lw.attn_q_nc,
                    lw.attn_q_qt, lw.attn_k_qt, lw.attn_v_qt
                )
            else:
                if lw.attn_q_use_c:
                    self._matmul_any(lw.attn_q_raw, p_xn, p_q, lw.attn_q_nr, lw.attn_q_nc, lw.attn_q_qt, lw.attn_q_use_v2)
                else:
                    b_q[:] = lw.attn_q_f32 @ b_qn[:lw.attn_q_nc]
                if lw.attn_k_use_c:
                    self._matmul_any(lw.attn_k_raw, p_xn, p_k, lw.attn_k_nr, lw.attn_k_nc, lw.attn_k_qt, lw.attn_k_use_v2)
                else:
                    b_k[:] = lw.attn_k_f32 @ b_qn[:lw.attn_k_nc]
                if lw.attn_v_use_c:
                    self._matmul_any(lw.attn_v_raw, p_xn, p_v, lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt, lw.attn_v_use_v2)
                else:
                    b_v[:] = lw.attn_v_f32 @ b_qn[:lw.attn_v_nc]

            # Q/K norm (Qwen3)
            if lw.has_q_norm:
                q_2d = b_q.reshape(NH, HD)
                for h in range(NH):
                    q_2d[h] = q_2d[h] / np.sqrt(np.mean(q_2d[h]*q_2d[h]) + self.eps) * lw.q_norm_w
            if lw.has_k_norm:
                k_2d = b_k[:NKH*HD].reshape(NKH, HD)
                for h in range(NKH):
                    k_2d[h] = k_2d[h] / np.sqrt(np.mean(k_2d[h]*k_2d[h]) + self.eps) * lw.k_norm_w

            # RoPE
            b_q[:] = self._apply_rope_fast(b_q, self.pos, NH)
            b_k[:] = self._apply_rope_fast(b_k, self.pos, NKH)

            # KV cache
            self.kv_k[i, self.kv_len[i], :NKH*HD] = b_k[:NKH*HD]
            self.kv_v[i, self.kv_len[i], :NKH*HD] = b_v[:NKH*HD]

            # Attention
            seq_len = self.kv_len[i] + 1
            k_cache = self.kv_k[i, :seq_len].reshape(seq_len, NKH, HD)
            v_cache = self.kv_v[i, :seq_len].reshape(seq_len, NKH, HD)
            q_2d = b_q.reshape(NH, HD)
            if self._gqa_rep > 1:
                k_exp = np.repeat(k_cache, self._gqa_rep, axis=1)
                v_exp = np.repeat(v_cache, self._gqa_rep, axis=1)
            else:
                k_exp = k_cache; v_exp = v_cache
            scores = np.einsum('hd,shd->hs', q_2d, k_exp.reshape(-1, NH, HD)) / np.sqrt(float(HD))
            scores -= np.max(scores, axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= np.sum(scores, axis=1, keepdims=True)
            att = np.einsum('hs,shd->hd', scores, v_exp.reshape(-1, NH, HD)).astype(np.float32)
            b_att[:] = att.reshape(-1)
            self.kv_len[i] += 1

            # O proj + residual — USE AVX2 for Q4_0
            self._matmul_any(lw.attn_out_raw, p_att, p_oproj,
                             lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt,
                             lw.attn_out_use_v2)
            b_x[:N] = b_r[:N] + b_oproj[:N]

            # FFN
            np.copyto(b_r, b_x)
            simd.rms_norm(p_xn, p_x,
                          lw.ffn_norm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          N, eps_f)

            # Gate+Up — use individual AVX2 when faster than batch OMP
            if lw.ffn_gate_use_v2 and lw.ffn_up_use_v2 and self._v2 is not None:
                self._v2.quant_matmul_v2(lw.ffn_gate_raw, p_xn, p_gate,
                                         lw.ffn_gate_nr, lw.ffn_gate_nc, lw.ffn_gate_qt)
                self._v2.quant_matmul_v2(lw.ffn_up_raw, p_xn, p_up,
                                         lw.ffn_up_nr, lw.ffn_up_nc, lw.ffn_up_qt)
            elif lw.ffn_gate_use_c and lw.ffn_up_use_c:
                kern.batch_gate_up_omp(
                    lw.ffn_gate_raw, lw.ffn_up_raw,
                    p_xn, p_gate, p_up,
                    lw.ffn_gate_nr, lw.ffn_up_nr,
                    lw.ffn_gate_nc, lw.ffn_gate_qt, lw.ffn_up_qt
                )
            else:
                if lw.ffn_gate_use_c:
                    self._matmul_any(lw.ffn_gate_raw, p_xn, p_gate, lw.ffn_gate_nr, lw.ffn_gate_nc, lw.ffn_gate_qt, lw.ffn_gate_use_v2)
                else:
                    b_gate[:FF] = lw.ffn_gate_f32 @ b_qn[:FF]
                if lw.ffn_up_use_c:
                    self._matmul_any(lw.ffn_up_raw, p_xn, p_up, lw.ffn_up_nr, lw.ffn_up_nc, lw.ffn_up_qt, lw.ffn_up_use_v2)
                else:
                    b_up[:FF] = lw.ffn_up_f32 @ b_qn[:FF]
            # SiLU * up
            simd.silu(p_silu, p_gate, ctypes.c_int(FF))
            b_silu[:FF] = b_silu[:FF] * b_up[:FF]

            # Down proj — USE AVX2 for Q4_0
            self._matmul_any(lw.ffn_down_raw, p_silu, p_ffn,
                             lw.ffn_down_nr, lw.ffn_down_nc, lw.ffn_down_qt,
                             lw.ffn_down_use_v2)
            b_x[:N] = b_r[:N] + b_ffn[:N]

        # Final norm + output proj
        simd.rms_norm(p_xn, p_x,
                      self._out_norm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      N, eps_f)
        if self._out_use_c:
            self._matmul_any(self._out_raw, p_xn, p_logits,
                             self._out_nr, self._out_nc, self._out_qt,
                             self._out_use_v2)
        else:
            self._logits[:] = self._out_f32 @ self._x_norm

        self.pos += 1
        return self._logits
