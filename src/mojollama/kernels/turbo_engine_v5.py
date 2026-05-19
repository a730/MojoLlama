#!/usr/bin/env python3
"""TurboEngine v5 — Full C forward pass, zero Python per-layer overhead.

Uses turbo_forward.so which runs the ENTIRE transformer in a single C call.
Python only does: load model → build C structs → call turbo_forward() per token.
"""

import numpy as np
import ctypes
import os
import time
import gguf
from gguf.constants import GGMLQuantizationType as QT

# GGML quant type codes
GGML_F32   = 0
GGML_F16   = 1
GGML_Q4_0  = 2
GGML_Q4_1  = 3
GGML_Q8_0  = 8
GGML_Q4_K  = 12
GGML_Q5_K  = 13
GGML_Q6_K  = 14

QTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    12: "Q4_K", 8: "Q8_0", 13: "Q5_K", 14: "Q6_K",
}

BLOCK_SIZES = {2: 18, 3: 20, 12: 144, 13: 176, 14: 210, 8: 34}
BLOCK_VALS  = {2: 32, 3: 32, 12: 256, 13: 256, 14: 256, 8: 32}
C_KERNEL_TYPES = {GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0}


class WeightDesc(ctypes.Structure):
    _fields_ = [
        ('data',              ctypes.c_void_p),
        ('nrows',             ctypes.c_int),
        ('ncols',             ctypes.c_int),
        ('qtype',             ctypes.c_int),
        ('block_size',        ctypes.c_int),
        ('vals_per_block',    ctypes.c_int),
    ]


class LayerDesc(ctypes.Structure):
    _fields_ = [
        ('attn_norm',         WeightDesc),
        ('attn_q',            WeightDesc),
        ('attn_k',            WeightDesc),
        ('attn_v',            WeightDesc),
        ('attn_out',          WeightDesc),
        ('attn_q_norm',       WeightDesc),
        ('attn_k_norm',       WeightDesc),
        ('ffn_norm',          WeightDesc),
        ('ffn_gate',          WeightDesc),
        ('ffn_up',            WeightDesc),
        ('ffn_down',          WeightDesc),
        ('is_moe',            ctypes.c_int),
        ('moe_gate_inp',      WeightDesc),
        ('n_experts',         ctypes.c_int),
        ('n_experts_per_tok', ctypes.c_int),
        ('n_ff_expert',       ctypes.c_int),
        ('moe_gate_exps',     ctypes.c_void_p),  # pointer to array of WeightDesc
        ('moe_up_exps',       ctypes.c_void_p),
        ('moe_down_exps',     ctypes.c_void_p),
        ('has_shared_expert', ctypes.c_int),
        ('shared_gate',       WeightDesc),
        ('shared_up',         WeightDesc),
        ('shared_down',       WeightDesc),
    ]


class ModelDesc(ctypes.Structure):
    _fields_ = [
        ('n_layers',        ctypes.c_int),
        ('n_embd',          ctypes.c_int),
        ('n_head',           ctypes.c_int),
        ('n_kv_head',        ctypes.c_int),
        ('head_dim',         ctypes.c_int),
        ('n_ff',             ctypes.c_int),
        ('rope_freq_base',   ctypes.c_float),
        ('eps',              ctypes.c_float),
        ('vocab_size',       ctypes.c_int),
        ('is_moe',           ctypes.c_int),
        ('token_embd',       WeightDesc),
        ('output_norm',      WeightDesc),
        ('output_weight',    WeightDesc),
        ('layers',          ctypes.POINTER(LayerDesc)),
        ('rope_cos',        ctypes.c_void_p),
        ('rope_sin',        ctypes.c_void_p),
        ('max_pos',         ctypes.c_int),
    ]


class ForwardState(ctypes.Structure):
    _fields_ = [
        ('kv_k',           ctypes.c_void_p),
        ('kv_v',           ctypes.c_void_p),
        ('kv_len',         ctypes.c_void_p),
        ('pos',            ctypes.c_int),
        ('h',              ctypes.c_void_p),
        ('residual',       ctypes.c_void_p),
        ('q',              ctypes.c_void_p),
        ('k',              ctypes.c_void_p),
        ('v',              ctypes.c_void_p),
        ('att_out',        ctypes.c_void_p),
        ('gate',           ctypes.c_void_p),
        ('up',             ctypes.c_void_p),
        ('silu_gate',      ctypes.c_void_p),
        ('ffn_out',        ctypes.c_void_p),
        ('logits',         ctypes.c_void_p),
        ('scores',         ctypes.c_void_p),
        ('expert_result',  ctypes.c_void_p),
        ('max_pos',        ctypes.c_int),
        ('n_embd',         ctypes.c_int),
        ('n_head',         ctypes.c_int),
        ('n_kv_head',      ctypes.c_int),
        ('head_dim',       ctypes.c_int),
        ('n_ff',           ctypes.c_int),
        ('n_layers',       ctypes.c_int),
    ]


def _make_weight_desc(name, weights, raw_weights, weight_info, weight_qtypes):
    """Create a WeightDesc ctypes struct from a weight name."""
    wd = WeightDesc()
    if name not in weights:
        wd.data = 0
        wd.nrows = 0
        wd.ncols = 0
        wd.qtype = 0
        wd.block_size = 0
        wd.vals_per_block = 0
        return wd

    w = weights[name]
    if w.ndim == 1:
        wd.nrows = w.shape[0]
        wd.ncols = 1
    else:
        wd.nrows = w.shape[0]
        wd.ncols = w.shape[1]

    # Special case: token embedding and output norm MUST be F32
    # (embedding is read directly, not via matmul kernel)
    is_f32_only = (name == 'token_embd.weight' or name == 'output_norm.weight' or
                   '.attn_norm.weight' in name or '.ffn_norm.weight' in name or
                   '.attn_q_norm.weight' in name or '.attn_k_norm.weight' in name or
                   'norm.weight' in name)

    qtype = weight_qtypes.get(name, GGML_F32)

    if name in raw_weights and qtype in C_KERNEL_TYPES and not is_f32_only:
        raw = raw_weights[name]
        wd.data = raw.ctypes.data_as(ctypes.c_void_p)
        wd.qtype = qtype
        info = weight_info[name]
        wd.nrows = info[0]
        wd.ncols = info[1]
        wd.block_size = info[2]
        wd.vals_per_block = {2: 32, 3: 32, 8: 32, 12: 256, 13: 256, 14: 256}.get(qtype, 0)
    else:
        # F32 or F16 — use dequantized data
        if w.dtype != np.float32:
            w = w.astype(np.float32)
            weights[name] = w  # update in place
        # Ensure contiguous
        w = np.ascontiguousarray(w)
        weights[name] = w
        wd.data = w.ctypes.data_as(ctypes.c_void_p)
        wd.qtype = GGML_F32  # always F32 after dequantization
        wd.block_size = 0
        wd.vals_per_block = 0

    return wd


class TurboEngineV5:
    """Full C forward pass engine — single C call per token."""

    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        self.n_threads = n_threads

        # Load GGUF model
        self.reader = gguf.GGUFReader(model_path)
        self._parse_metadata()
        self._load_weights()

        moe_str = f'/MoE-{self.n_experts}x{self.n_experts_per_tok}' if self.is_moe else ''
        print(f"TurboEngine v5: {self.n_layers}L/{self.n_embd}D/"
              f"{self.n_ff}FF/{self.n_head}H/{self.n_kv_head}KV"
              f"{moe_str} | t={self.n_threads} | vocab={self.vocab_size}")

        # Load C library
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(kernel_dir, 'turbo_forward.so')
        self.lib = ctypes.CDLL(so_path)

        # Set up function signatures
        self.lib.turbo_forward.argtypes = [
            ctypes.POINTER(ForwardState),
            ctypes.POINTER(ModelDesc),
            ctypes.c_int,     # token_id
            ctypes.POINTER(ctypes.c_float),  # logits out
        ]
        self.lib.turbo_forward.restype = None

        self.lib.set_num_threads.argtypes = [ctypes.c_int]
        self.lib.set_num_threads.restype = None
        self.lib.set_num_threads(n_threads)

        # Build C structs and initialize
        self._build_c_model()
        self._alloc_state()
        self.reset()

    def _parse_metadata(self):
        """Extract architecture info from GGUF metadata."""
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
                    arch = prefix
                    break

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
        self.n_ff_expert = int(_get(f'{arch}.expert_feed_forward_length') or self.n_ff)
        self.is_moe = self.n_experts is not None

        if not self.is_moe:
            self.n_experts = 1
            self.n_experts_per_tok = 1
        else:
            self.n_experts = int(self.n_experts)
            self.n_experts_per_tok = int(self.n_experts_per_tok)

        self.arch_prefix = arch

    def _load_weights(self):
        """Load and dequantize all weights."""
        self.weights = {}
        self.raw_weights = {}
        self.weight_qtypes = {}
        self.weight_info = {}

        for t in self.reader.tensors:
            name = t.name
            qtype = QT(t.tensor_type).value
            self.weight_qtypes[name] = qtype

            if len(t.shape) == 2:
                in_dim = int(t.shape[0])
                out_dim = int(t.shape[1])

                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                f32 = f32.reshape(out_dim, in_dim)
                self.weights[name] = np.ascontiguousarray(f32)

                if qtype in C_KERNEL_TYPES:
                    raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8).copy()
                    self.raw_weights[name] = raw
                    ts = BLOCK_SIZES[qtype]
                    self.weight_info[name] = (out_dim, in_dim, ts, qtype)

            elif len(t.shape) == 3:
                # MoE expert weights
                in_dim = int(t.shape[0])
                out_dim = int(t.shape[1])
                n_exp = int(t.shape[2])

                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                f32 = f32.reshape(out_dim, in_dim, n_exp)
                self.weights[name] = np.ascontiguousarray(f32)

                if qtype in C_KERNEL_TYPES:
                    raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8).copy()
                    ts = BLOCK_SIZES[qtype]
                    self.weight_info[name] = (out_dim, in_dim, ts, qtype)

            elif len(t.shape) == 1:
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
                self.weights[name] = np.ascontiguousarray(f32)

        # Output weight
        if 'output.weight' in self.weights:
            self.out_w_name = 'output.weight'
        else:
            self.out_w_name = 'token_embd.weight'

        # Fix embedding matrix for lookup
        emb = self.weights['token_embd.weight']
        if emb.ndim == 2 and emb.shape[1] == self.n_embd:
            pass  # (vocab, n_embd) — correct for lookup
        elif emb.ndim == 2:
            emb = np.ascontiguousarray(emb.T)
            self.weights['token_embd.weight'] = emb

        if self.vocab_size == 0:
            self.vocab_size = emb.shape[0]

    def _build_c_model(self):
        """Build the C ModelDesc struct from loaded weights."""
        N = self.n_embd
        NH = self.n_head
        NKH = self.n_kv_head
        HD = self.head_dim

        # Build layer descriptors
        self._layer_descs = []
        self._expert_arrays = []  # Keep refs alive

        for i in range(self.n_layers):
            pfx = f'blk.{i}'
            ld = LayerDesc()
            ld.attn_norm = _make_weight_desc(f'{pfx}.attn_norm.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
            ld.attn_q = _make_weight_desc(f'{pfx}.attn_q.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
            ld.attn_k = _make_weight_desc(f'{pfx}.attn_k.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
            ld.attn_v = _make_weight_desc(f'{pfx}.attn_v.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
            ld.attn_out = _make_weight_desc(f'{pfx}.attn_output.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
            ld.attn_q_norm = _make_weight_desc(f'{pfx}.attn_q_norm.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
            ld.attn_k_norm = _make_weight_desc(f'{pfx}.attn_k_norm.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
            ld.ffn_norm = _make_weight_desc(f'{pfx}.ffn_norm.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)

            if self.is_moe:
                ld.is_moe = 1
                ld.moe_gate_inp = _make_weight_desc(f'{pfx}.ffn_gate_inp.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
                ld.n_experts = self.n_experts
                ld.n_experts_per_tok = self.n_experts_per_tok
                ld.n_ff_expert = self.n_ff_expert

                # Expert weight arrays
                gate_exps = (WeightDesc * self.n_experts)()
                up_exps = (WeightDesc * self.n_experts)()
                down_exps = (WeightDesc * self.n_experts)()

                for e in range(self.n_experts):
                    gname = f'{pfx}.ffn_gate_exps.weight'
                    uname = f'{pfx}.ffn_up_exps.weight'
                    dname = f'{pfx}.ffn_down_exps.weight'

                    # Expert weights are 3D: (out_dim, in_dim, n_experts)
                    # For C, store each expert's slice separately
                    gw = self.weights[gname]
                    uw = self.weights[uname]
                    dw = self.weights[dname]

                    # Slice out this expert: (out_dim, in_dim)
                    g_slice = np.ascontiguousarray(gw[:, :, e])
                    u_slice = np.ascontiguousarray(uw[:, :, e])
                    d_slice = np.ascontiguousarray(dw[:, :, e])

                    self._expert_arrays.extend([g_slice, u_slice, d_slice])

                    gate_exps[e] = _make_weight_desc_from_array(g_slice)
                    up_exps[e] = _make_weight_desc_from_array(u_slice)
                    down_exps[e] = _make_weight_desc_from_array(d_slice)

                ld.moe_gate_exps = ctypes.cast(gate_exps, ctypes.c_void_p)
                ld.moe_up_exps = ctypes.cast(up_exps, ctypes.c_void_p)
                ld.moe_down_exps = ctypes.cast(down_exps, ctypes.c_void_p)

                # Store arrays to prevent GC
                self._expert_arrays.extend([gate_exps, up_exps, down_exps])

                # Check for shared expert (Qwen3 MoE)
                sgate = f'{pfx}.ffn_shared_gate.weight'
                if sgate in self.weights:
                    ld.has_shared_expert = 1
                    ld.shared_gate = _make_weight_desc(sgate, self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
                    ld.shared_up = _make_weight_desc(f'{pfx}.ffn_shared_up.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
                    ld.shared_down = _make_weight_desc(f'{pfx}.ffn_shared_down.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
                else:
                    ld.has_shared_expert = 0
            else:
                ld.is_moe = 0
                ld.ffn_gate = _make_weight_desc(f'{pfx}.ffn_gate.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
                ld.ffn_up = _make_weight_desc(f'{pfx}.ffn_up.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
                ld.ffn_down = _make_weight_desc(f'{pfx}.ffn_down.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)

            self._layer_descs.append(ld)

        # Build model descriptor
        self._model_desc = ModelDesc()
        self._model_desc.n_layers = self.n_layers
        self._model_desc.n_embd = self.n_embd
        self._model_desc.n_head = NH
        self._model_desc.n_kv_head = NKH
        self._model_desc.head_dim = HD
        self._model_desc.n_ff = self.n_ff
        self._model_desc.rope_freq_base = self.rope_freq_base
        self._model_desc.eps = self.eps
        self._model_desc.vocab_size = self.vocab_size
        self._model_desc.is_moe = 1 if self.is_moe else 0

        self._model_desc.token_embd = _make_weight_desc('token_embd.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
        self._model_desc.output_norm = _make_weight_desc('output_norm.weight', self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)
        self._model_desc.output_weight = _make_weight_desc(self.out_w_name, self.weights, self.raw_weights, self.weight_info, self.weight_qtypes)

        # Layer array
        self._layer_array = (LayerDesc * self.n_layers)(*self._layer_descs)
        self._model_desc.layers = self._layer_array

        # Max position for KV cache
        self.max_pos = 4096
        self._model_desc.max_pos = self.max_pos

        # Rope tables (not used yet — apply_rope is computed on the fly in C)
        self._model_desc.rope_cos = ctypes.c_void_p(0)
        self._model_desc.rope_sin = ctypes.c_void_p(0)

    def _alloc_state(self):
        """Allocate C-compatible forward pass state buffers."""
        N   = self.n_embd
        NH  = self.n_head
        NKH = self.n_kv_head
        HD  = self.head_dim
        FF  = self.n_ff if not self.is_moe else self.n_ff_expert
        L   = self.n_layers
        MP  = self.max_pos
        VS  = self.vocab_size
        kv_dim = NKH * HD

        self._state = ForwardState()
        self._kv_k_buf = np.zeros((L, MP, kv_dim), dtype=np.float32)
        self._kv_v_buf = np.zeros((L, MP, kv_dim), dtype=np.float32)
        self._kv_len_buf = np.zeros(L, dtype=np.int32)

        buf_size = max(N, FF, NH * HD, NKH * HD, VS)
        self._h_buf = np.zeros(buf_size, dtype=np.float32)
        self._res_buf = np.zeros(buf_size, dtype=np.float32)
        self._q_buf = np.zeros(NH * HD, dtype=np.float32)
        self._k_buf = np.zeros(NKH * HD, dtype=np.float32)
        self._v_buf = np.zeros(NKH * HD, dtype=np.float32)
        self._att_buf = np.zeros(NH * HD, dtype=np.float32)
        self._gate_buf = np.zeros(FF * 2, dtype=np.float32)
        self._up_buf = np.zeros(FF * 2, dtype=np.float32)
        self._silu_buf = np.zeros(FF * 2, dtype=np.float32)
        self._ffn_buf = np.zeros(buf_size, dtype=np.float32)
        self._logits_buf = np.zeros(VS, dtype=np.float32)
        self._scores_buf = np.zeros(NH * MP, dtype=np.float32)
        self._expert_buf = np.zeros(buf_size, dtype=np.float32)

        self._state.kv_k = self._kv_k_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.kv_v = self._kv_v_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.kv_len = self._kv_len_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.pos = 0
        self._state.h = self._h_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.residual = self._res_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.q = self._q_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.k = self._k_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.v = self._v_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.att_out = self._att_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.gate = self._gate_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.up = self._up_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.silu_gate = self._silu_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.ffn_out = self._ffn_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.logits = self._logits_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.scores = self._scores_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.expert_result = self._expert_buf.ctypes.data_as(ctypes.c_void_p)
        self._state.max_pos = MP
        self._state.n_embd = N
        self._state.n_head = NH
        self._state.n_kv_head = NKH
        self._state.head_dim = HD
        self._state.n_ff = FF
        self._state.n_layers = L

    def reset(self):
        """Reset KV cache and position."""
        self._kv_len_buf[:] = 0
        self._state.pos = 0

    def forward(self, token_id):
        """Run one forward pass — single C call."""
        self.lib.turbo_forward(
            ctypes.byref(self._state),
            ctypes.byref(self._model_desc),
            ctypes.c_int(token_id),
            self._logits_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return self._logits_buf


def _make_weight_desc_from_array(arr):
    """Create a WeightDesc from a numpy array (for expert slices)."""
    wd = WeightDesc()
    arr = np.ascontiguousarray(arr.astype(np.float32))
    wd.data = arr.ctypes.data_as(ctypes.c_void_p)
    wd.qtype = GGML_F32
    if arr.ndim == 1:
        wd.nrows = arr.shape[0]
        wd.ncols = 1
    else:
        wd.nrows = arr.shape[0]
        wd.ncols = arr.shape[1]
    wd.block_size = 0
    wd.vals_per_block = 0
    return wd


if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'Llama-3.2-1B-Instruct-Q4_0.gguf'
    n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    engine = TurboEngineV5(model_path, n_threads=n_threads)

    # Warmup
    engine.reset()
    logits = engine.forward(128000)
    tok = int(np.argmax(logits))
    for _ in range(5):
        logits = engine.forward(tok)
        tok = int(np.argmax(logits))

    # Benchmark
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        logits = engine.forward(tok)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        tok = int(np.argmax(logits))

    times.sort()
    median = times[len(times) // 2]
    print(f"\nResult: {median:.2f} ms/tok = {1000.0/median:.1f} tok/s")
    print(f"Top-5 tokens: {np.argsort(logits)[-5:][::-1]}")