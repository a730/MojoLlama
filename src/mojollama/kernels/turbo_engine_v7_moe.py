#!/usr/bin/env python3
"""
TurboEngine v7-MoE — Zero-allocation engine with C-accelerated MoE support.

Optimized load: keeps large 2D tensors as raw quantized (no F32 dequant).
Uses C quant_matmul_omp for all projections including expert matmuls.
"""

import numpy as np
import ctypes
import os
import time
import gguf
from gguf.constants import GGMLQuantizationType as QT

from forward.base import ArchitectureForwardPass

GGML_F32  = 0; GGML_F16 = 1; GGML_Q4_0 = 2; GGML_Q4_1 = 3
GGML_Q8_0 = 8; GGML_Q4_K = 12; GGML_Q5_K = 13; GGML_Q6_K = 14; GGML_MXFP4 = 39

QTYPE_NAMES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",12:"Q4_K",8:"Q8_0",13:"Q5_K",14:"Q6_K"}
BLOCK_SIZES = {2:18, 3:20, 12:144, 13:176, 14:210, 8:34, 39:17}
BLOCK_VALS  = {2:32, 3:32, 12:256, 13:256, 14:256, 8:32, 39:32}
C_KERNEL_TYPES = {GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGML_MXFP4}


class MoEExpertPointers:
    """Pre-computed raw quantized pointers for one MoE layer's experts."""
    __slots__ = ['gate_raw', 'up_raw', 'down_raw',
                 'gate_ptrs', 'up_ptrs', 'down_ptrs',
                 'gate_ptrs_arr', 'up_ptrs_arr', 'down_ptrs_arr',
                 'gate_nr', 'gate_nc', 'gate_qt',
                 'up_nr', 'up_nc', 'up_qt',
                 'down_nr', 'down_nc', 'down_qt',
                 'gate_use_c', 'up_use_c', 'down_use_c',
                 'gate_f32', 'up_f32', 'down_f32',
                 'router_raw', 'router_nr', 'router_nc', 'router_qt',
                 'router_f32',
                 'n_experts', 'n_ff_expert', 'n_embd']
    
    def __init__(self):
        self.gate_raw = []
        self.up_raw = []
        self.down_raw = []
        self.gate_ptrs = []
        self.up_ptrs = []
        self.down_ptrs = []
        self.gate_ptrs_arr = None
        self.up_ptrs_arr = None
        self.down_ptrs_arr = None
        self.router_raw = None
        self.router_f32 = None
        self.gate_use_c = False
        self.up_use_c = False
        self.down_use_c = False


class LayerWeights:
    """Dense layer weights — pre-computed ctypes pointers."""
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
                 'moe',
                 'attn_qk_raw', 'attn_qk_nr', 'attn_qk_nc', 'attn_qk_qt',
                 'attn_qk_use_c',
                 '_attn_qk_raw_buf',
                 # Qwen3.6 hybrid SSM+attention fields
                 'attn_qkv_raw', 'attn_qkv_nr', 'attn_qkv_nc', 'attn_qkv_qt', 'attn_qkv_use_c',
                 'attn_gate_raw', 'attn_gate_nr', 'attn_gate_nc', 'attn_gate_qt', 'attn_gate_use_c',
                 'ssm_conv1d_ptr', 'ssm_a_ptr', 'ssm_dt_bias_ptr',
                 'ssm_alpha_ptr', 'ssm_beta_ptr', 'ssm_norm_ptr',
                 'ssm_out_raw', 'ssm_out_nr', 'ssm_out_nc', 'ssm_out_qt', 'ssm_out_use_c',
                 'shexp_gate_raw', 'shexp_gate_nr', 'shexp_gate_nc', 'shexp_gate_qt', 'shexp_gate_use_c',
                 'shexp_up_raw', 'shexp_up_nr', 'shexp_up_nc', 'shexp_up_qt', 'shexp_up_use_c',
                 'shexp_down_raw', 'shexp_down_nr', 'shexp_down_nc', 'shexp_down_qt', 'shexp_down_use_c',
                 'shexp_router_ptr', 'shexp_router_nr',
                 # Gemma4 specific fields
                 'post_attn_norm_w', 'post_ffw_norm_w', 'post_norm_w',
                 'per_layer_proj', 'per_layer_inp_gate', 'layer_scale',
                 'gemma4_nq', 'gemma4_nk', 'gemma4_nv', 'gemma4_head_dim',
                'gemma4_is_swa', 'gemma4_rope_dim', 'gemma4_freq_base', 'gemma4_kv_idx',
                 # ZAYA specific fields
                 'res_scale_hs_w', 'res_scale_hs_b', 'res_scale_res_w', 'res_scale_res_b',
                 'ffn_gate_inp_w', 'ffn_gate_inp_b', 'ffn_gate_w', 'ffn_gate_b', 'ffn_norm_w',
                 'zaya_router_mlp2_w', 'zaya_router_mlp2_b', 'zaya_router_mlp4_w',
                 'zaya_router_biases_w', 'zaya_router_eda_w',
                 'cca_val_proj1_w', 'cca_val_proj2_w',
                 'cca_vp1_raw', 'cca_vp2_raw', 'cca_vp1_nr', 'cca_vp2_nr',
                 'cca_vp1_nc', 'cca_vp2_nc', 'cca_vp1_qt', 'cca_vp2_qt',
                 'ssm_conv1d_w', 'ssm_conv1d_b',
                 'gate_up_exps_raw_arr', 'down_exps_raw_arr',
                 'gate_up_exps_nr', 'down_exps_nr',
                 'gate_up_exps_nc', 'down_exps_nc',
                 'gate_up_exps_qt', 'down_exps_qt',
                 'gate_up_exps_n_exp', 'down_exps_n_exp',
                 'gate_up_ptrs', 'down_ptrs',
                 'gate_up_info', 'down_info']


class TurboEngineV7MoE:
    """Zero-allocation engine with MoE support via C quant matmul."""

    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        self.n_threads = n_threads
        self.model_path = model_path

        t_start = time.perf_counter()
        self.reader = gguf.GGUFReader(model_path)
        self._parse_metadata()
        print(f"  GGUFReader: {time.perf_counter()-t_start:.1f}s", flush=True)

        # Architecture forward dispatch
        self._arch_forward = None
        if self.arch_name == 'gemma4':
            from forward.gemma4 import ForwardGemma4
            self._arch_forward = ForwardGemma4(self)
        elif self.arch_name == 'zaya':
            from forward.zaya import ForwardZaya
            self._arch_forward = ForwardZaya(self)
        
        self._load_kernels()
        t0 = time.perf_counter()
        self._load_weights()
        print(f"  Weights: {time.perf_counter()-t0:.1f}s", flush=True)
        
        t0 = time.perf_counter()
        self._preload_pointers()
        print(f"  Preload pointers: {time.perf_counter()-t0:.1f}s", flush=True)
        
        t0 = time.perf_counter()
        self._init_buffers()
        print(f"  Buffers: {time.perf_counter()-t0:.1f}s", flush=True)
        
        self.reset()
        moe_str = f'/MoE-{self.n_experts}x{self.n_experts_per_tok}' if self.is_moe else ''
        print(f"TurboEngine v7-MoE: {self.n_layers}L/{self.n_embd}D/"
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
        
        # Read architecture from GGUF metadata (general.architecture)
        arch_raw = _get('general.architecture')
        if arch_raw is not None:
            if isinstance(arch_raw, bytes):
                arch = arch_raw.decode('utf-8')
            elif isinstance(arch_raw, str):
                arch = arch_raw
            elif hasattr(arch_raw, 'tobytes'):
                # numpy array of uint8 (GGUF string encoding)
                arch = bytes(arch_raw.tolist()).decode('utf-8')
            else:
                arch = str(arch_raw)
        else:
            # Fallback: detect from known prefixes
            arch = 'llama'
            for prefix in ['gpt-oss', 'qwen35moe', 'qwen3moe', 'qwen2moe', 'qwen2', 'llama', 'mistral', 'falcon', 'gemma', 'starcoder2', 'phi3', 'deepseek2', 'mixtral']:
                for key in fields:
                    if f'{prefix}.block_count' in key:
                        arch = prefix; break
                if arch != 'llama':
                    break
        
        self.arch_name = arch
        self.arch_prefix = arch
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
        self.n_experts_per_tok = _get(f'{arch}.expert_used_count') or _get(f'{arch}.experts_used_count')
        self.n_ff_expert = int(_get(f'{arch}.expert_feed_forward_length') or self.n_ff)
        self.is_moe = self.n_experts is not None
        if not self.is_moe:
            self.n_experts = 1; self.n_experts_per_tok = 1
        else:
            self.n_experts = int(self.n_experts); self.n_experts_per_tok = int(self.n_experts_per_tok)
        self.arch_prefix = arch
        # Qwen3.6 hybrid SSM+attention fields
        self.rope_dim = int(_get(f'{arch}.rope.dimension_count') or 0)
        self.full_attn_interval = int(_get(f'{arch}.full_attention_interval') or 4)
        self.n_layers_actual = self.n_layers
        # SSM parameters
        self.ssm_groups = 32
        self.ssm_state_size = 128
        self.ssm_dt_rank = 32
        self.ssm_conv_kernel = 4
        self.ssm_inner = self.n_head * self.head_dim  # 4096
        # Layer types: 0=full attention, 1=SSM (Qwen3.6 hybrid)
        if self.arch_prefix == 'qwen35moe':
            self.layer_types = [0 if i % self.full_attn_interval == 0 else 1 for i in range(self.n_layers)]
            print(f"  Qwen3.6 hybrid: {sum(1 for t in self.layer_types if t==0)} attn + {sum(1 for t in self.layer_types if t==1)} ssm layers, rope_dim={self.rope_dim}", flush=True)
        else:
            self.layer_types = None

    def _load_kernels(self):
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        cf = ctypes.POINTER(ctypes.c_float)
        cu = ctypes.POINTER(ctypes.c_uint8)
        ci = ctypes.c_int

        so_path = os.path.join(kernel_dir, 'quant_kernels_omp.so')
        if os.path.exists(so_path):
            so = ctypes.CDLL(so_path)
            so.quant_matmul_omp.argtypes = [cu, cf, cf, ci, ci, ci]
            so.quant_matmul_omp.restype = None
            so.f32_matmul_omp.argtypes = [cf, cf, cf, ci, ci]
            so.f32_matmul_omp.restype = None
            so.batch_qkv_omp.argtypes = [cu, cu, cu, cf, cf, cf, cf, ci, ci, ci, ci, ci, ci]
            so.batch_qkv_omp.restype = None
            so.batch_gate_up_omp.argtypes = [cu, cu, cf, cf, cf, ci, ci, ci, ci]
            so.batch_gate_up_omp.restype = None
            so.set_num_threads.argtypes = [ci]
            so.set_num_threads.restype = None
            so.set_num_threads(self.n_threads)
            # MoE fused kernel — eliminates 1,152 ctypes calls/token
            so.moe_forward_omp.argtypes = [
                ctypes.POINTER(cu),  # gate_raw[] — array of per-expert uint8_t*
                ctypes.POINTER(cu),  # up_raw[]
                ctypes.POINTER(cu),  # down_raw[]
                cf,                   # x_norm
                ci, ci,               # n_ff_expert, n_embd
                ci, ci, ci,           # qt_gate, qt_up, qt_down
                ctypes.POINTER(ctypes.c_int),  # top_indices[top_k]
                cf,                   # top_weights[top_k]
                ci,                   # top_k
                cf,                   # combined[n_embd]
                cf,                   # prealloc_buf[3*top_k*n_ff_expert]
                cu,                   # prealloc_q8[n_blocks_x*34]
            ]
            so.moe_forward_omp.restype = None
            self._kern = so
            # GQA attention C kernel
            gqa_path = os.path.join(kernel_dir, 'gqa_attention.so')
            if os.path.exists(gqa_path):
                self._gqa_attn = ctypes.CDLL(gqa_path)
                self._gqa_attn.gqa_attention_decode.argtypes = [
                    cf, cf, cf, cf, ci, ci, ci, ci]
                self._gqa_attn.gqa_attention_decode.restype = None
            else:
                self._gqa_attn = None
            # Dequant functions for requantize
            so.q6_k_dequantize_row.argtypes = [cu, cf, ci]
            so.q6_k_dequantize_row.restype = None
            so.q4_k_dequantize_row.argtypes = [cu, cf, ci]
            so.q4_k_dequantize_row.restype = None
        else:
            self._kern = None

        # Load cengine_batch_instr.so for batch_forward and ssm_decode_step
        cengine_path = os.path.join(kernel_dir, 'cengine_batch_instr.so')
        self._cengine = None
        if os.path.exists(cengine_path):
            ce = ctypes.CDLL(cengine_path)
            # ssm_decode_step: void ssm_decode_step(x, ssm_intermediate, conv1d_w, a_param, dt_bias, alpha, beta, ssm_norm_w, state, B, N, inner, groups, state_size, conv_kernel, dt_rank)
            ce.ssm_decode_step.argtypes = [
                cf, cf, cf, cf, cf, cf, cf, cf, cf,
                ci, ci, ci, ci, ci, ci, ci]
            ce.ssm_decode_step.restype = None
            # batch_forward: void batch_forward(BC*, tokens, B, ws)
            # We'll define BC struct later
            self._cengine = ce

        simd_path = os.path.join(kernel_dir, 'simd_ops.so')
        if os.path.exists(simd_path):
            simd = ctypes.CDLL(simd_path)
            simd.rms_norm.argtypes = [cf, cf, cf, ci, ctypes.c_float]
            simd.rms_norm.restype = None
            simd.silu.argtypes = [cf, cf, ci]
            simd.silu.restype = None
            simd.residual_add.argtypes = [cf, cf, cf, ci]
            simd.residual_add.restype = None
            simd.apply_rope.argtypes = [cf, cf, ci, ci, ci, ci, ctypes.c_float]
            simd.apply_rope.restype = None
            self._simd = simd
        else:
            self._simd = None

    def _load_weights(self):
        """Load weights: keep quantized for C kernel dispatch, F32 only when needed."""
        self.weights = {}          # F32 weights (1D norms, small 2D, fallback)
        self.raw_weights = {}      # Raw quantized bytes for C kernel
        self.weight_qtypes = {}
        self.weight_info = {}      # For 2D: (n_rows, n_cols, block_sz, qt)
                                   # For 3D: (out_dim, in_dim, block_sz, qt, n_experts)

        # Pre-scan to identify large tensors that should stay quantized only
        skip_f32 = set()  # tensor names to NOT dequantize to F32
        if self._kern is not None:
            for t in self.reader.tensors:
                name = t.name
                if len(t.shape) == 2:
                    in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
                    qtype = QT(t.tensor_type).value
                    if qtype in C_KERNEL_TYPES:
                        f32_mb = out_dim * in_dim * 4 / 1024 / 1024
                        if f32_mb > 10:
                            skip_f32.add(name)

        total_tensors = len(list(self.reader.tensors))
        processed = 0
        for t in self.reader.tensors:
            name = t.name
            qtype = QT(t.tensor_type).value
            self.weight_qtypes[name] = qtype
            processed += 1
            if processed % 100 == 0:
                print(f"  Loading tensors: {processed}/{total_tensors}...", flush=True)

            if len(t.shape) == 2:
                in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
                
                if name in skip_f32 and qtype in C_KERNEL_TYPES:
                    # Raw quantized only — use C kernel (no copy — mmap view for multiprocessing sharing)
                    raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8)
                    self.raw_weights[name] = raw
                    self.weight_info[name] = (out_dim, in_dim, BLOCK_SIZES.get(qtype, 0), qtype)
                else:
                    # Dequantize to F32
                    f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                    f32 = f32.reshape(out_dim, in_dim)
                    self.weights[name] = np.ascontiguousarray(f32)
                    if qtype in C_KERNEL_TYPES:
                        raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8)
                        self.raw_weights[name] = raw
                        self.weight_info[name] = (out_dim, in_dim, BLOCK_SIZES.get(qtype, 0), qtype)

            elif len(t.shape) == 3:
                # MoE 3D: store raw quantized only (no F32 dequant)
                in_dim, out_dim, n_exp = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
                if qtype in C_KERNEL_TYPES:
                    raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8)
                    self.raw_weights[name] = raw
                    self.weight_info[name] = (out_dim, in_dim, BLOCK_SIZES.get(qtype, 0), qtype, n_exp)
                else:
                    # Unsupported quant — dequant to F32
                    f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                    f32 = f32.reshape(n_exp, out_dim, in_dim)
                    self.weights[name] = np.ascontiguousarray(f32)

            elif len(t.shape) == 1:
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
                self.weights[name] = np.ascontiguousarray(f32)

        # Token embedding: must dequantize for lookup
        print(f"  Loading token embedding...", flush=True)
        for t in self.reader.tensors:
            if t.name in ('token_embd.weight', 'model.embed_tokens.weight'):
                qtype = QT(t.tensor_type).value
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                if len(t.shape) == 2:
                    in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
                    f32 = f32.reshape(out_dim, in_dim) if f32.shape[0] == in_dim else f32
                    if f32.shape[1] != self.n_embd:
                        f32 = np.ascontiguousarray(f32.T)
                self.weights['token_embd.weight'] = np.ascontiguousarray(f32)
                self.vocab_size = f32.shape[0]
                self.emb = self.weights['token_embd.weight']
                break

        # Output weight
        self.out_w_name = 'output.weight' if 'output.weight' in self.raw_weights or 'output.weight' in self.weights else 'token_embd.weight' if 'token_embd.weight' in self.raw_weights or 'token_embd.weight' in self.weights else 'model.embed_tokens.weight'
        out_info = self.weight_info.get(self.out_w_name)
        if out_info is not None and self._kern is not None and self.arch_name != 'zaya':
            # Skip requantization for arch-specific forward passes that use np.dot
            self._out_nr = ctypes.c_int(out_info[0])
            self._out_nc = ctypes.c_int(out_info[1])
            self._out_qt_orig = ctypes.c_int(out_info[3])
            if out_info[3] in (12, 14, 13):  # K-quants → requantize to Q8_0
                print(f"  Converting output: {QTYPE_NAMES.get(out_info[3], out_info[3])} → Q8_0...", flush=True)
                self._out_raw = self._requantize_output(self.raw_weights[self.out_w_name], out_info)
                self._out_qt = ctypes.c_int(8)
            elif out_info[3] == 1:  # F16 → requantize to Q8_0
                print(f"  Converting output: F16 → Q8_0 for C matmul...", flush=True)
                self._out_raw = self._requantize_output(self.raw_weights[self.out_w_name], out_info)
                self._out_qt = ctypes.c_int(8)
            else:
                self._out_raw = self.raw_weights[self.out_w_name].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                self._out_qt = out_info[3]
            self._out_use_c = True
        else:
            self._out_f32 = self.weights[self.out_w_name]
            self._out_use_c = False

        self._out_norm_w = self.weights.get('output_norm.weight')
        if self._out_norm_w is None and 'model.final_norm.weight' in self.weights:
            self._out_norm_w = self.weights['model.final_norm.weight']
            self.weights['output_norm.weight'] = self._out_norm_w

        # ZAYA MXFP4 naming adapter: map model.layers.N.zaya_block. → blk.N.
        if self.arch_name == 'zaya' and 'model.layers.0.zaya_block.attn_k.weight' in self.raw_weights:
            print("  Detected HF naming convention — adding blk.N aliases...", flush=True)
            import re as _re
            for old_name in list(self.weights.keys()):
                m = _re.match(r'model\.layers\.(\d+)\.zaya_block\.(.+)', old_name)
                if m:
                    new_name = f'blk.{m.group(1)}.{m.group(2)}'
                    self.weights[new_name] = self.weights[old_name]
            for old_name in list(self.raw_weights.keys()):
                m = _re.match(r'model\.layers\.(\d+)\.zaya_block\.(.+)', old_name)
                if m:
                    new_name = f'blk.{m.group(1)}.{m.group(2)}'
                    self.raw_weights[new_name] = self.raw_weights[old_name]
                    info = self.weight_info.get(old_name)
                    if info:
                        self.weight_info[new_name] = info

        if self._arch_forward is not None:
            self._arch_forward.init_weights(self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)

    def _requantize_output(self, raw_weights, out_info):
        """Requantize output from K-quant to Q8_0 for faster logits projection."""
        n_rows, n_cols = out_info[0], out_info[1]
        orig_qt = out_info[3]
        
        f32 = np.zeros((n_rows, n_cols), dtype=np.float32)
        
        if orig_qt == 14:  # Q6_K
            n_blk = n_cols // 256
            raw_ptr = raw_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            for r in range(n_rows):
                off = r * n_blk * 210
                ptr = ctypes.cast(ctypes.addressof(raw_ptr.contents) + off, ctypes.POINTER(ctypes.c_uint8))
                self._kern.q6_k_dequantize_row(
                    ptr, f32[r].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(n_cols))
        elif orig_qt == 12:  # Q4_K
            n_blk = n_cols // 256
            raw_ptr = raw_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            for r in range(n_rows):
                off = r * n_blk * 144
                ptr = ctypes.cast(ctypes.addressof(raw_ptr.contents) + off, ctypes.POINTER(ctypes.c_uint8))
                self._kern.q4_k_dequantize_row(
                    ptr, f32[r].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(n_cols))
        elif orig_qt == 1:  # F16 — view as float16 then convert to float32
            raw_u16 = np.frombuffer(bytes(raw_weights[:n_rows*n_cols*2]), dtype=np.uint16).reshape(n_rows, n_cols)
            f32 = raw_u16.astype(np.float32).view(np.float32)
        else:
            raise NotImplementedError(f"Requantize from type {orig_qt}")
        
        # Quantize to Q8_0 (34 bytes per 32 values)
        n_blocks = n_cols // 32
        blocks = f32.reshape(n_rows, n_blocks, 32)
        amax = np.max(np.abs(blocks), axis=2, keepdims=True)
        amax = np.clip(amax, 1e-10, None)
        d = amax / 127.0
        qs = np.clip(np.round(blocks / d), -127, 127).astype(np.int8)
        d_f16 = d[:, :, 0].astype(np.float16)
        d_u16 = d_f16.view(np.uint16)
        
        q8 = np.zeros((n_rows, n_blocks, 34), dtype=np.uint8)
        q8[:, :, 0] = d_u16 & 0xFF
        q8[:, :, 1] = (d_u16 >> 8) & 0xFF
        q8[:, :, 2:] = qs.view(np.uint8)
        
        q8_flat = q8.reshape(n_rows * n_blocks * 34)
        self._out_q8_buf = np.ascontiguousarray(q8_flat, dtype=np.uint8)
        return self._out_q8_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    def _compute_expert_raw_offsets(self, raw_bytes, n_experts, out_dim, in_dim, block_sz, block_vals):
        """Compute byte offsets into raw weight array for each expert."""
        n_blocks_per_row = (in_dim + block_vals - 1) // block_vals
        row_bytes = n_blocks_per_row * block_sz
        expert_bytes = out_dim * row_bytes
        raw_ptr_type = ctypes.POINTER(ctypes.c_uint8)
        
        if isinstance(raw_bytes, np.ndarray):
            base_ptr = raw_bytes.ctypes.data_as(raw_ptr_type)
        else:
            base_ptr = raw_bytes
        
        offsets = []
        for e in range(n_experts):
            off = e * expert_bytes
            ptr = ctypes.cast(ctypes.addressof(base_ptr.contents) + off, raw_ptr_type)
            offsets.append(ptr)
        return offsets

    def _preload_pointers(self):
        """Pre-compute ALL weight pointers — zero dict lookups in forward pass."""
        cf = ctypes.POINTER(ctypes.c_float)
        cu = ctypes.POINTER(ctypes.c_uint8)

        self._layers = []
        self._moe_layers = [] if self.is_moe else None

        for i in range(self.n_layers):
            lw = LayerWeights()
            pfx = f'blk.{i}'

            # Architecture-specific pointer setup
            if self._arch_forward is not None:
                self._arch_forward.init_pointers(i, pfx, lw)
            
            # Dense projection weights
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
                    setattr(lw, f'{attr}_raw', raw.ctypes.data_as(cu))
                    setattr(lw, f'{attr}_nr', ctypes.c_int(info[0]))
                    setattr(lw, f'{attr}_nc', ctypes.c_int(info[1]))
                    setattr(lw, f'{attr}_qt', ctypes.c_int(info[3]))
                    setattr(lw, f'{attr}_use_c', True)
                else:
                    if name not in self.weights:
                        for t in self.reader.tensors:
                            if t.name == name:
                                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                                if len(t.shape) == 2:
                                    f32 = f32.reshape(int(t.shape[1]), int(t.shape[0]))
                                self.weights[name] = np.ascontiguousarray(f32)
                                break
                    setattr(lw, f'{attr}_f32', self.weights.get(name))
                    setattr(lw, f'{attr}_use_c', False)

            lw.attn_norm_w = self.weights[f'{pfx}.attn_norm.weight']
            # GPT-OSS and some other architectures call this post_attention_norm
            if f'{pfx}.ffn_norm.weight' in self.weights:
                lw.ffn_norm_w = self.weights[f'{pfx}.ffn_norm.weight']
            elif f'{pfx}.post_attention_norm.weight' in self.weights:
                lw.ffn_norm_w = self.weights[f'{pfx}.post_attention_norm.weight']
            elif self._arch_forward is not None:
                lw.ffn_norm_w = None
            else:
                raise KeyError(f'No FFN norm weight found for layer {i}')

            # Build fused Q+K weight if Q and K have same quant type
            if (lw.attn_q_use_c and lw.attn_k_use_c and
                lw.attn_q_qt.value == lw.attn_k_qt.value):
                q_name = f'{pfx}.attn_q.weight'
                k_name = f'{pfx}.attn_k.weight'
                q_raw_arr = self.raw_weights.get(q_name)
                k_raw_arr = self.raw_weights.get(k_name)
                if q_raw_arr is not None and k_raw_arr is not None:
                    qk_raw_arr = np.concatenate([q_raw_arr, k_raw_arr])
                    qk_raw_arr = np.ascontiguousarray(qk_raw_arr, dtype=np.uint8)
                    # Keep the fused buffer alive on the layer object
                    lw._attn_qk_raw_buf = qk_raw_arr
                    lw.attn_qk_raw = qk_raw_arr.ctypes.data_as(cu)
                    lw.attn_qk_nr = ctypes.c_int(lw.attn_q_nr.value + lw.attn_k_nr.value)
                    lw.attn_qk_nc = lw.attn_q_nc  # same n_cols as Q
                    lw.attn_qk_qt = lw.attn_q_qt  # same quant type as Q
                    lw.attn_qk_use_c = True
            lw.has_q_norm = f'{pfx}.attn_q_norm.weight' in self.weights
            lw.has_k_norm = f'{pfx}.attn_k_norm.weight' in self.weights
            if lw.has_q_norm:
                lw.q_norm_w = self.weights[f'{pfx}.attn_q_norm.weight']
            if lw.has_k_norm:
                lw.k_norm_w = self.weights[f'{pfx}.attn_k_norm.weight']
            self._layers.append(lw)

            # ── Qwen3.6 hybrid SSM+attention: load additional tensors ──
            if self.arch_prefix == 'qwen35moe':
                # attn_qkv.weight [2048, 8192] Q8_0
                qkv_name = f'{pfx}.attn_qkv.weight'
                qkv_info = self.weight_info.get(qkv_name)
                if qkv_info is not None:
                    lw.attn_qkv_raw = self.raw_weights[qkv_name].ctypes.data_as(cu)
                    lw.attn_qkv_nr = ctypes.c_int(qkv_info[0])  # 8192
                    lw.attn_qkv_nc = ctypes.c_int(qkv_info[1])  # 2048
                    lw.attn_qkv_qt = ctypes.c_int(qkv_info[3])
                    lw.attn_qkv_use_c = True
                else:
                    lw.attn_qkv_use_c = False
                    lw.attn_qkv_raw = None

                # attn_gate.weight [2048, 4096] Q8_0 — N→inner projection
                gate_name = f'{pfx}.attn_gate.weight'
                gate_info = self.weight_info.get(gate_name)
                if gate_info is not None:
                    lw.attn_gate_raw = self.raw_weights[gate_name].ctypes.data_as(cu)
                    lw.attn_gate_nr = ctypes.c_int(gate_info[0])  # 4096 = inner
                    lw.attn_gate_nc = ctypes.c_int(gate_info[1])  # 2048 = N
                    lw.attn_gate_qt = ctypes.c_int(gate_info[3])
                    lw.attn_gate_use_c = True
                else:
                    lw.attn_gate_use_c = False

                # SSM F32 tensors (store pointers to numpy arrays)
                for ssm_attr, ssm_wname in [
                    ('ssm_conv1d_ptr', 'ssm_conv1d.weight'),
                    ('ssm_a_ptr', 'ssm_a'),
                    ('ssm_dt_bias_ptr', 'ssm_dt.bias'),
                    ('ssm_alpha_ptr', 'ssm_alpha.weight'),
                    ('ssm_beta_ptr', 'ssm_beta.weight'),
                    ('ssm_norm_ptr', 'ssm_norm.weight'),
                ]:
                    name = f'{pfx}.{ssm_wname}'
                    w = self.weights.get(name)
                    if w is not None:
                        setattr(lw, ssm_attr, w.ctypes.data_as(cf))
                    else:
                        setattr(lw, ssm_attr, None)

                # ssm_out.weight [4096, 2048] Q8_0 — inner→N projection
                ssm_out_name = f'{pfx}.ssm_out.weight'
                ssm_out_info = self.weight_info.get(ssm_out_name)
                if ssm_out_info is not None:
                    lw.ssm_out_raw = self.raw_weights[ssm_out_name].ctypes.data_as(cu)
                    lw.ssm_out_nr = ctypes.c_int(ssm_out_info[0])  # 2048 = N
                    lw.ssm_out_nc = ctypes.c_int(ssm_out_info[1])  # 4096 = inner
                    lw.ssm_out_qt = ctypes.c_int(ssm_out_info[3])
                    lw.ssm_out_use_c = True
                else:
                    lw.ssm_out_use_c = False

                # Shared expert weights
                shexp_names = [
                    ('shexp_gate', 'ffn_gate_shexp.weight', 512, 2048),
                    ('shexp_up', 'ffn_up_shexp.weight', 512, 2048),
                    ('shexp_down', 'ffn_down_shexp.weight', 2048, 512),
                ]
                for attr, wname, nr, nc_val in shexp_names:
                    name = f'{pfx}.{wname}'
                    info = self.weight_info.get(name)
                    if info is not None:
                        setattr(lw, f'{attr}_raw', self.raw_weights[name].ctypes.data_as(cu))
                        setattr(lw, f'{attr}_nr', ctypes.c_int(info[0]))
                        setattr(lw, f'{attr}_nc', ctypes.c_int(info[1]))
                        setattr(lw, f'{attr}_qt', ctypes.c_int(info[3]))
                        setattr(lw, f'{attr}_use_c', True)
                    else:
                        setattr(lw, f'{attr}_use_c', False)
                        setattr(lw, f'{attr}_raw', None)

                # Shared expert router (1D F32 vector [2048])
                router_name = f'{pfx}.ffn_gate_inp_shexp.weight'
                router_w = self.weights.get(router_name)
                if router_w is not None:
                    lw.shexp_router_ptr = router_w.ctypes.data_as(cf)
                    lw.shexp_router_nr = ctypes.c_int(router_w.shape[0])  # 2048
                else:
                    lw.shexp_router_ptr = None

            # ── MoE pointers ──
            if self.is_moe:
                me = MoEExpertPointers()
                me.n_experts = self.n_experts
                me.n_ff_expert = self.n_ff_expert
                me.n_embd = self.n_embd
                
                # Router (2D)
                router_name = f'{pfx}.ffn_gate_inp.weight'
                router_info = self.weight_info.get(router_name)
                if router_info is not None:
                    me.router_raw = self.raw_weights[router_name].ctypes.data_as(cu)
                    me.router_nr = ctypes.c_int(router_info[0])
                    me.router_nc = ctypes.c_int(router_info[1])
                    me.router_qt = ctypes.c_int(router_info[3])
                else:
                    me.router_f32 = self.weights.get(router_name)

                # Expert gate/up/down - 3D tensors, store per-expert numpy slices
                for attr, wname in [('gate', 'ffn_gate_exps'), ('up', 'ffn_up_exps'), ('down', 'ffn_down_exps')]:
                    name = f'{pfx}.{wname}.weight'
                    info = self.weight_info.get(name)
                    raw = self.raw_weights.get(name)
                    if info is not None and raw is not None:
                        out_dim, in_dim, bs, qt, n_exp = info
                        bv = BLOCK_VALS[qt]
                        n_blocks_per_row = (in_dim + bv - 1) // bv
                        row_bytes = n_blocks_per_row * bs
                        expert_bytes = out_dim * row_bytes
                        
                        # Store numpy views for each expert (zero copy slicing)
                        expert_views = []
                        for e in range(n_exp):
                            start = e * expert_bytes
                            end = (e + 1) * expert_bytes
                            ev = raw[start:end]
                            expert_views.append(ev)  # numpy ndarray view
                        
                        # Pre-compute ctypes pointers for fused MoE kernel
                        raw_ptr = raw.ctypes.data_as(cu)
                        expert_ptrs = [
                            ctypes.cast(ctypes.addressof(raw_ptr.contents) + e * expert_bytes, cu)
                            for e in range(n_exp)
                        ]
                        
                        # Pre-build ctypes array for zero-cost dispatch
                        cu_arr_full = ctypes.POINTER(ctypes.c_uint8) * n_exp
                        expert_ptrs_arr = cu_arr_full(*expert_ptrs)
                        
                        setattr(me, f'{attr}_raw', expert_views)
                        setattr(me, f'{attr}_ptrs', expert_ptrs)
                        setattr(me, f'{attr}_ptrs_arr', expert_ptrs_arr)
                        setattr(me, f'{attr}_nr', ctypes.c_int(out_dim))
                        setattr(me, f'{attr}_nc', ctypes.c_int(in_dim))
                        setattr(me, f'{attr}_qt', ctypes.c_int(qt))
                        setattr(me, f'{attr}_use_c', True)
                    else:
                        setattr(me, f'{attr}_use_c', False)
                        setattr(me, f'{attr}_nr', ctypes.c_int(0))
                        setattr(me, f'{attr}_nc', ctypes.c_int(0))
                        setattr(me, f'{attr}_qt', ctypes.c_int(0))
                
                self._moe_layers.append(me)

    def _init_buffers(self):
        """Pre-allocate ALL buffers — zero allocation in forward pass."""
        N = self.n_embd
        NK = self.n_head * self.head_dim
        NKH = self.n_kv_head * self.head_dim
        MAX_POS = 4096
        FF_expert = self.n_ff_expert if self.is_moe else self.n_ff

        # KV cache
        self.kv_k = np.zeros((self.n_layers, MAX_POS, NKH), dtype=np.float32)
        self.kv_v = np.zeros((self.n_layers, MAX_POS, NKH), dtype=np.float32)
        self.kv_len = np.zeros(self.n_layers, dtype=np.int32)

        self._x = np.zeros(N, dtype=np.float32)
        self._x_norm = np.zeros(N, dtype=np.float32)
        self._residual = np.zeros(N, dtype=np.float32)
        # Fused Q+K buffer: _q = _qk[:NK], _k = _qk[NK:] for fused QK matmul
        # For Qwen3.6 with attn_qkv (8192 dims), allocate extra space
        qkv_sz = max(NK + NKH, 8192) if hasattr(self, 'arch_prefix') and self.arch_prefix == 'qwen35moe' else NK + NKH
        self._qk = np.zeros(qkv_sz, dtype=np.float32)
        self._q = self._qk[:NK]
        self._k = self._qk[NK:NK+NKH]
        self._v = np.zeros(NKH, dtype=np.float32)
        self._att_out = np.zeros(NK, dtype=np.float32)
        self._o_proj = np.zeros(N, dtype=np.float32)
        self._gate = np.zeros(self.n_ff, dtype=np.float32)
        self._up = np.zeros(self.n_ff, dtype=np.float32)
        self._silu_gate = np.zeros(self.n_ff, dtype=np.float32)
        self._ffn_out = np.zeros(N, dtype=np.float32)
        
        # MoE buffers
        self._moe_router_scores = np.zeros(max(128, self.n_experts), dtype=np.float32)
        self._moe_gate_out = np.zeros(FF_expert, dtype=np.float32)
        self._moe_up_out = np.zeros(FF_expert, dtype=np.float32)
        self._moe_silu_out = np.zeros(FF_expert, dtype=np.float32)
        self._moe_expert_out = np.zeros(N, dtype=np.float32)
        self._moe_combined = np.zeros(N, dtype=np.float32)

        # Pre-allocated buffers for C moe_forward_omp (no malloc/calloc in hot path)
        max_top_k = max(8, self.n_experts_per_tok)
        buf_sz = 3 * max_top_k * FF_expert
        self._moe_prealloc_buf = np.zeros(buf_sz, dtype=np.float32)
        n_blocks_x = N // 32
        self._moe_prealloc_q8 = np.zeros(n_blocks_x * 34, dtype=np.uint8)

        self._logits = np.zeros(self.vocab_size, dtype=np.float32)

        # SSM state buffer for Qwen3.6 hybrid model
        if self.arch_prefix == 'qwen35moe':
            n_layers = self.n_layers
            groups = self.ssm_groups
            state_size = self.ssm_state_size
            self._ssm_state = np.zeros((n_layers, groups, state_size), dtype=np.float32)
            self._ssm_intermediate = np.zeros(self.ssm_inner, dtype=np.float32)
            self._gate_4096 = np.zeros(self.ssm_inner, dtype=np.float32)
            print(f"  SSM state: {n_layers}x{groups}x{state_size} = {n_layers*groups*state_size*4/1024/1024:.1f} MB", flush=True)
        else:
            self._ssm_state = None
            self._ssm_intermediate = None
            self._gate_4096 = None

        # ctypes pointers
        cf = ctypes.POINTER(ctypes.c_float)
        self._p_x = self._x.ctypes.data_as(cf)
        self._p_x_norm = self._x_norm.ctypes.data_as(cf)
        self._p_residual = self._residual.ctypes.data_as(cf)
        self._p_q = self._q.ctypes.data_as(cf)
        self._p_k = self._k.ctypes.data_as(cf)
        self._p_v = self._v.ctypes.data_as(cf)
        self._p_qk = self._qk.ctypes.data_as(cf)  # fused QK output
        self._p_att_out = self._att_out.ctypes.data_as(cf)
        self._p_gate = self._gate.ctypes.data_as(cf)
        self._p_up = self._up.ctypes.data_as(cf)
        self._p_silu_gate = self._silu_gate.ctypes.data_as(cf)
        self._p_o_proj = self._o_proj.ctypes.data_as(cf)
        self._p_ffn = self._ffn_out.ctypes.data_as(cf)
        self._p_logits = self._logits.ctypes.data_as(cf)
        self._p_moe_router = self._moe_router_scores.ctypes.data_as(cf)
        self._p_moe_gate = self._moe_gate_out.ctypes.data_as(cf)
        self._p_moe_up = self._moe_up_out.ctypes.data_as(cf)
        self._p_moe_silu = self._moe_silu_out.ctypes.data_as(cf)
        self._p_moe_expert = self._moe_expert_out.ctypes.data_as(cf)
        self._p_moe_combined = self._moe_combined.ctypes.data_as(cf)

        # RoPE tables
        HD = self.head_dim
        self._rope_cos_table = {}
        self._rope_sin_table = {}
        self._eps_f = ctypes.c_float(self.eps)
        self._gqa_rep = self.n_head // self.n_kv_head if self.n_head != self.n_kv_head else 1

        if self._arch_forward is not None:
            self._arch_forward.init_buffers()

    def reset(self):
        self.kv_len[:] = 0
        self.pos = 0
        self.reset_state()

    def reset_state(self):
        """Reset SSM state to zeros (call between sequences)."""
        if self._ssm_state is not None:
            self._ssm_state.fill(0.0)
        if self._arch_forward is not None:
            self._arch_forward.reset_state()

    def _apply_rope_fast(self, x, pos, n_heads, rope_dim=None):
        hd = self.head_dim
        if rope_dim is None or rope_dim <= 0:
            rope_dim = hd
        half = rope_dim // 2
        key = (pos, rope_dim)
        if key not in self._rope_cos_table:
            freq = self.rope_freq_base ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim)
            angle = pos / freq
            self._rope_cos_table[key] = np.cos(angle).astype(np.float32)
            self._rope_sin_table[key] = np.sin(angle).astype(np.float32)
        cos_a = self._rope_cos_table[key]
        sin_a = self._rope_sin_table[key]
        x2d = x.reshape(n_heads, hd)
        out = x2d.copy()
        # Only rotate first rope_dim dimensions
        out[:, :half] = x2d[:, :half] * cos_a - x2d[:, half:rope_dim] * sin_a
        out[:, half:rope_dim] = x2d[:, half:rope_dim] * cos_a + x2d[:, :half] * sin_a
        # Dimensions beyond rope_dim are unchanged
        return out.reshape(-1)

    def forward(self, token_id):
        """Single-token forward pass."""
        # Architecture-specific dispatch
        if self._arch_forward is not None:
            return self._arch_forward.forward(token_id)
        b_x = self._x; b_r = self._residual; b_qn = self._x_norm
        b_q = self._q; b_k = self._k; b_v = self._v
        b_att = self._att_out; b_gate = self._gate; b_up = self._up
        b_silu = self._silu_gate; b_ffn = self._ffn_out
        b_oproj = self._o_proj
        N = self.n_embd; NH = self.n_head; NKH = self.n_kv_head * self.head_dim; HD = self.head_dim
        FF = self.n_ff; L = self.n_layers
        kern = self._kern; simd = self._simd
        eps_f = self._eps_f
        p_x = self._p_x; p_xn = self._p_x_norm; p_r = self._p_residual
        p_q = self._p_q; p_k = self._p_k; p_v = self._p_v
        p_qk = self._p_qk  # fused QK output buffer
        p_att = self._p_att_out; p_gate = self._p_gate; p_up = self._p_up
        p_silu = self._p_silu_gate; p_oproj = self._p_o_proj
        p_ffn = self._p_ffn; p_logits = self._p_logits

        # Embedding lookup
        np.copyto(b_x, self.emb[token_id])

        cf = ctypes.POINTER(ctypes.c_float)
        cu = ctypes.POINTER(ctypes.c_uint8)
        ci = ctypes.c_int
        
        # Qwen3.6 constants
        ssm_inner = self.ssm_inner if hasattr(self, 'ssm_inner') else (NH * HD)

        for i in range(L):
            lw = self._layers[i]
            is_ssm_layer = (self.layer_types is not None and self.layer_types[i] == 1)

            # NaN clamp: prevent FP32 overflow (critical for Lance, Qwen3.6)
            np.clip(b_x, -1000.0, 1000.0, out=b_x)
            
            # Residual copy
            np.copyto(b_r, b_x)

            # RMS norm 1
            simd.rms_norm(p_xn, p_x,
                          lw.attn_norm_w.ctypes.data_as(cf),
                          N, eps_f)

            if is_ssm_layer and self._cengine is not None and lw.ssm_conv1d_ptr is not None:
                # ═══ SSM path ═══
                ce = self._cengine
                # Call ssm_decode_step (C function)
                state_ptr = self._ssm_state[i].ctypes.data_as(cf)
                interm_ptr = self._ssm_intermediate.ctypes.data_as(cf)
                ce.ssm_decode_step(
                    p_xn, interm_ptr,
                    lw.ssm_conv1d_ptr, lw.ssm_a_ptr, lw.ssm_dt_bias_ptr,
                    lw.ssm_alpha_ptr, lw.ssm_beta_ptr, lw.ssm_norm_ptr,
                    state_ptr,
                    ci(1), ci(N), ci(ssm_inner),
                    ci(self.ssm_groups), ci(self.ssm_state_size),
                    ci(self.ssm_conv_kernel), ci(self.ssm_dt_rank))
                
                # attn_gate: xn → gate_4096 (N→inner via quant_matmul)
                kern.quant_matmul_omp(
                    lw.attn_gate_raw, p_xn, self._gate_4096.ctypes.data_as(cf),
                    lw.attn_gate_nr, lw.attn_gate_nc, lw.attn_gate_qt)
                
                # Element-wise gating: combined = gate * intermediate
                np.multiply(self._gate_4096, self._ssm_intermediate, out=self._gate_4096)
                
                # ssm_out projection (inner→N)
                kern.quant_matmul_omp(
                    lw.ssm_out_raw, self._gate_4096.ctypes.data_as(cf), p_oproj,
                    lw.ssm_out_nr, lw.ssm_out_nc, lw.ssm_out_qt)
                
                # Residual
                b_x[:N] = b_r[:N] + b_oproj[:N]

            else:
                # ═══ Full Attention path ═══
                
                # QKV using attn_qkv (fused) if available, else separate Q/K/V
                if hasattr(lw, 'attn_qkv_use_c') and lw.attn_qkv_use_c:
                    # attn_qkv: N→8192 (Q:4096, K:2048, V:2048)
                    qkv_buf = self._qk  # reuse _qk buffer for full QKV output (needs 8192 floats)
                    # Ensure buffer is large enough — _qk was allocated as NK + NKH = 4096+512=4608
                    # For 8192 we need larger buffer, so use a temp allocation
                    if len(self._qk) < 8192:
                        self._qk = np.zeros(8192, dtype=np.float32)
                        self._p_qk = self._qk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    p_qkv = self._p_qk
                    kern.quant_matmul_omp(lw.attn_qkv_raw, p_xn, p_qkv, 
                                           lw.attn_qkv_nr, lw.attn_qkv_nc, lw.attn_qkv_qt)
                    # Split: first NH*HD = Q, next N = K, next N = V
                    nq_vals = NH * HD  # 4096
                    b_q[:] = self._qk[:nq_vals]
                    b_k[:NKH] = self._qk[nq_vals:nq_vals + NKH]  # first NKH*HD from K section
                    b_v[:NKH] = self._qk[nq_vals + N:nq_vals + N + NKH]  # first NKH*HD from V section
                elif hasattr(lw, 'attn_qk_use_c') and lw.attn_qk_use_c:
                    # Fused Q+K matmul into _qk buffer
                    kern.quant_matmul_omp(
                        lw.attn_qk_raw, p_xn, p_qk,
                        lw.attn_qk_nr, lw.attn_qk_nc, lw.attn_qk_qt)
                    # Separate V matmul
                    kern.quant_matmul_omp(
                        lw.attn_v_raw, p_xn, p_v,
                        lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)
                elif lw.attn_q_use_c and lw.attn_k_use_c and lw.attn_v_use_c:
                    kern.batch_qkv_omp(
                        lw.attn_q_raw, lw.attn_k_raw, lw.attn_v_raw,
                        p_xn, p_q, p_k, p_v,
                        lw.attn_q_nr, lw.attn_k_nr, lw.attn_v_nr,
                        lw.attn_q_nc,
                        lw.attn_q_qt, lw.attn_k_qt, lw.attn_v_qt)
                else:
                    if lw.attn_q_use_c:
                        kern.quant_matmul_omp(lw.attn_q_raw, p_xn, p_q, lw.attn_q_nr, lw.attn_q_nc, lw.attn_q_qt)
                    if lw.attn_k_use_c:
                        kern.quant_matmul_omp(lw.attn_k_raw, p_xn, p_k, lw.attn_k_nr, lw.attn_k_nc, lw.attn_k_qt)
                    if lw.attn_v_use_c:
                        kern.quant_matmul_omp(lw.attn_v_raw, p_xn, p_v, lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)

                # Q/K norm
                if lw.has_q_norm:
                    q_2d = b_q.reshape(NH, HD)
                    q_rms = np.sqrt(np.mean(q_2d * q_2d, axis=1, keepdims=True) + self.eps)
                    q_2d[:] = q_2d / q_rms * lw.q_norm_w.reshape(1, HD)
                if lw.has_k_norm:
                    k_2d = b_k[:NKH].reshape(self.n_kv_head, HD)
                    k_rms = np.sqrt(np.mean(k_2d * k_2d, axis=1, keepdims=True) + self.eps)
                    k_2d[:] = k_2d / k_rms * lw.k_norm_w.reshape(1, HD)

                # RoPE with partial rotary (rope_dim)
                rope_d = self.rope_dim if hasattr(self, 'rope_dim') and self.rope_dim > 0 else HD
                b_q[:] = self._apply_rope_fast(b_q, self.pos, NH, rope_dim=rope_d)
                b_k[:] = self._apply_rope_fast(b_k, self.pos, self.n_kv_head, rope_dim=rope_d)

                # KV cache
                self.kv_k[i, self.kv_len[i], :NKH] = b_k[:NKH]
                self.kv_v[i, self.kv_len[i], :NKH] = b_v[:NKH]

                # GQA Attention
                seq_len = self.kv_len[i] + 1
                if self._gqa_attn is not None and seq_len <= 4096:
                    self._gqa_attn.gqa_attention_decode(
                        b_q.ctypes.data_as(cf),
                        self.kv_k[i].ctypes.data_as(cf),
                        self.kv_v[i].ctypes.data_as(cf),
                        b_att.ctypes.data_as(cf),
                        ci(seq_len), ci(NH), ci(self.n_kv_head), ci(HD))
                else:
                    k_cache = self.kv_k[i, :seq_len].reshape(seq_len, self.n_kv_head, HD)
                    v_cache = self.kv_v[i, :seq_len].reshape(seq_len, self.n_kv_head, HD)
                    q_2d = b_q.reshape(NH, HD)
                    nk = self.n_kv_head; gqa = self._gqa_rep
                    q_g = q_2d.reshape(nk, gqa, HD)
                    k_T = k_cache.transpose(1, 0, 2)
                    scores = np.einsum('khd,ksd->khs', q_g, k_T) / np.sqrt(float(HD))
                    scores = scores.reshape(NH, seq_len)
                    scores -= np.max(scores, axis=1, keepdims=True)
                    np.exp(scores, out=scores)
                    scores /= np.sum(scores, axis=1, keepdims=True)
                    v_T = v_cache.transpose(1, 0, 2)
                    att = np.einsum('khs,ksd->khd', scores.reshape(nk, gqa, seq_len), v_T)
                    b_att[:] = att.reshape(-1)
                self.kv_len[i] += 1

                # Output projection: attn_gate (if available) else standard O proj
                if hasattr(lw, 'attn_gate_use_c') and lw.attn_gate_use_c:
                    # Qwen3.6: attn_gate maps inner→N (produced by attn_qkv GQA)
                    kern.quant_matmul_omp(
                        lw.attn_gate_raw, p_att, p_oproj,
                        lw.attn_gate_nr, lw.attn_gate_nc, lw.attn_gate_qt)
                    b_x[:N] = b_r[:N] + b_oproj[:N]
                else:
                    if lw.attn_out_use_c:
                        kern.quant_matmul_omp(lw.attn_out_raw, p_att, p_oproj,
                                               lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt)
                        b_x[:N] = b_r[:N] + b_oproj[:N]
                    else:
                        b_x[:N] = b_r[:N] + (lw.attn_out_f32 @ b_att)[:N]

            # ── FFN ──
            np.copyto(b_r, b_x)
            np.clip(b_x, -1000.0, 1000.0, out=b_x)
            simd.rms_norm(p_xn, p_x,
                          lw.ffn_norm_w.ctypes.data_as(cf),
                          N, eps_f)

            if self.is_moe and self._moe_layers:
                self._forward_moe(i, p_xn, b_ffn, b_r, N)
                # b_x now = b_r + MoE_FFN_out
                # Shared expert (Qwen3.6) — add on top
                if (hasattr(lw, 'shexp_router_ptr') and lw.shexp_router_ptr is not None
                    and hasattr(lw, 'shexp_gate_use_c') and lw.shexp_gate_use_c):
                    # Router score: dot product with 1D vector
                    shexp_score = float(np.dot(self._x_norm,
                        np.ctypeslib.as_array(lw.shexp_router_ptr, shape=(N,))))
                    if shexp_score > 0:
                        shexp_int = lw.shexp_gate_nr.value  # 512
                        kern.quant_matmul_omp(lw.shexp_gate_raw, p_xn, p_gate,
                                               lw.shexp_gate_nr, lw.shexp_gate_nc, lw.shexp_gate_qt)
                        kern.quant_matmul_omp(lw.shexp_up_raw, p_xn, p_up,
                                               lw.shexp_up_nr, lw.shexp_up_nc, lw.shexp_up_qt)
                        simd.silu(p_silu, p_gate, ci(shexp_int))
                        b_silu[:shexp_int] *= b_up[:shexp_int]
                        kern.quant_matmul_omp(lw.shexp_down_raw, p_silu, p_ffn,
                                               lw.shexp_down_nr, lw.shexp_down_nc, lw.shexp_down_qt)
                        b_x[:N] += shexp_score * b_ffn[:N]
            else:
                if lw.ffn_gate_use_c and lw.ffn_up_use_c:
                    kern.batch_gate_up_omp(
                        lw.ffn_gate_raw, lw.ffn_up_raw, p_xn, p_gate, p_up,
                        lw.ffn_gate_nr, lw.ffn_up_nr,
                        lw.ffn_gate_nc, lw.ffn_gate_qt, lw.ffn_up_qt)
                else:
                    if lw.ffn_gate_use_c:
                        kern.quant_matmul_omp(lw.ffn_gate_raw, p_xn, p_gate,
                                               lw.ffn_gate_nr, lw.ffn_gate_nc, lw.ffn_gate_qt)
                    if lw.ffn_up_use_c:
                        kern.quant_matmul_omp(lw.ffn_up_raw, p_xn, p_up,
                                               lw.ffn_up_nr, lw.ffn_up_nc, lw.ffn_up_qt)
                simd.silu(p_silu, p_gate, ci(FF))
                b_silu[:FF] *= b_up[:FF]
                if lw.ffn_down_use_c:
                    kern.quant_matmul_omp(lw.ffn_down_raw, p_silu, p_ffn,
                                           lw.ffn_down_nr, lw.ffn_down_nc, lw.ffn_down_qt)
                    b_x[:N] = b_r[:N] + b_ffn[:N]
                else:
                    b_x[:N] = b_r[:N] + (lw.ffn_down_f32 @ b_silu[:lw.ffn_down_nc.value])[:N]

        # Final norm
        np.clip(b_x, -1000.0, 1000.0, out=b_x)
        simd.rms_norm(p_xn, p_x,
                      self._out_norm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      N, eps_f)

        # Output projection
        if self._out_use_c:
            kern.quant_matmul_omp(self._out_raw, p_xn, p_logits,
                                   self._out_nr, self._out_nc, self._out_qt)
        else:
            self._logits[:] = self._out_f32 @ b_qn

        self.pos += 1
        return self._logits

    def _forward_moe(self, layer, p_xn, b_ffn_out, b_residual, N):
        """C-accelerated MoE FFN — quant matmul for expert dispatch."""
        me = self._moe_layers[layer]
        n_exp = me.n_experts
        top_k = self.n_experts_per_tok
        FF = me.n_ff_expert
        kern = self._kern

        # Router
        if me.router_raw is not None:
            kern.quant_matmul_omp(
                me.router_raw, p_xn, self._p_moe_router,
                me.router_nr, me.router_nc, me.router_qt)
        else:
            np.copyto(self._moe_router_scores[:n_exp],
                      (me.router_f32 @ self._x_norm)[:n_exp])

        # Softmax + Top-K
        scores = self._moe_router_scores[:n_exp]
        scores -= np.max(scores)
        np.exp(scores, out=scores)
        scores /= np.sum(scores)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_weights = scores[top_indices]
        top_weights /= np.sum(top_weights)

        # Fused MoE dispatch — single C call replaces 24 ctypes calls
        # Use pre-built full-pointer arrays (zero construction cost)
        gate_ptrs = me.gate_ptrs_arr
        up_ptrs   = me.up_ptrs_arr
        down_ptrs = me.down_ptrs_arr

        top_idx_arr = np.ascontiguousarray(top_indices.astype(np.int32))
        top_wt_arr  = np.ascontiguousarray(top_weights.astype(np.float32))

        self._moe_combined[:] = 0.0
        kern.moe_forward_omp(
            gate_ptrs, up_ptrs, down_ptrs,
            p_xn,
            ctypes.c_int(FF), ctypes.c_int(N),
            me.gate_qt, me.up_qt, me.down_qt,
            top_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            top_wt_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(top_k),
            self._p_moe_combined,
            self._moe_prealloc_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._moe_prealloc_q8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

        b_ffn_out[:N] = self._moe_combined[:N]

    def _silu(self, x):
        return x / (1.0 + np.exp(-x))


if __name__ == '__main__':
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else '/onedev-workspace/work/Qwen3-30B-A3B-Q4_K_M.gguf'
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    engine = TurboEngineV7MoE(model, n_threads=t)
    
    # Warmup
    engine.reset()
    logits = engine.forward(0)
    tok = int(np.argmax(logits))
    for _ in range(5):
        logits = engine.forward(tok)
        tok = int(np.argmax(logits))
    
    # Benchmark
    times = []
    for _ in range(n_gen):
        t0 = time.perf_counter()
        logits = engine.forward(tok)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        tok = int(np.argmax(logits))
    
    times.sort()
    median = times[len(times)//2]
    avg = sum(times) / len(times)
    print(f"\n{'='*60}")
    print(f"TurboEngine v7-MoE Benchmark")
    print(f"{'='*60}")
    print(f"Model:       {model}")
    print(f"Threads:     {t}")
    print(f"Tokens gen:  {n_gen}")
    print(f"{'─'*60}")
    print(f"Median:      {median:.1f} ms/tok ({1000.0/median:.1f} tok/s)")
    print(f"Average:     {avg:.1f} ms/tok ({1000.0/avg:.1f} tok/s)")
    print(f"Min:         {min(times):.1f} ms/tok ({1000.0/min(times):.1f} tok/s)")
    print(f"Max:         {max(times):.1f} ms/tok ({1000.0/max(times):.1f} tok/s)")
    print(f"{'='*60}")
    
    # Compare with llama.cpp baseline
    print(f"\nllama.cpp reference (Qwen3-30B-A3B Q4_K_M, 32t): ~28.5 tok/s")
    ratio = 1000.0/median / 28.5
    print(f"MojoLlama custom engine: {ratio*100:.0f}% of llama.cpp speed")
