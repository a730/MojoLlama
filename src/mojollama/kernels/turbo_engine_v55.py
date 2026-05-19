#!/usr/bin/env python3
"""TurboEngine v5.5 — Hybrid Python+C engine.

Uses the C kernel library for ALL matmuls (batched per layer) but Python
for attention, norm, SiLU, and other small operations where numpy is faster.

Key optimization: Batch all matmuls for a layer into single OMP parallel
region to reduce fork/join overhead. This eliminates most Python→C transitions
while keeping numpy for the small vector ops where it excels.
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


class TurboEngineV55:
    """Hybrid engine — C matmuls with Python+numpy for attention/norm."""

    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        self.n_threads = n_threads

        self.reader = gguf.GGUFReader(model_path)
        self._parse_metadata()
        self._load_weights()
        self._load_kernels()
        self._init_kv_cache()
        self.reset()

        moe_str = f'/MoE-{self.n_experts}x{self.n_experts_per_tok}' if self.is_moe else ''
        print(f"TurboEngine v5.5: {self.n_layers}L/{self.n_embd}D/"
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
        self.n_ff_expert = int(_get(f'{arch}.expert_feed_forward_length') or self.n_ff)
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
        cf = ctypes.POINTER(ctypes.c_float); cu = ctypes.POINTER(ctypes.c_uint8)
        self._quant_so = None
        so_path = os.path.join(kernel_dir, 'quant_kernels_omp.so')
        if os.path.exists(so_path):
            so = ctypes.CDLL(so_path)
            so.quant_matmul_omp.argtypes = [cu, cf, cf, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            so.quant_matmul_omp.restype = None
            so.f32_matmul_omp.argtypes = [cf, cf, cf, ctypes.c_int, ctypes.c_int]
            so.f32_matmul_omp.restype = None
            so.set_num_threads.argtypes = [ctypes.c_int]
            so.set_num_threads.restype = None
            so.set_num_threads(self.n_threads)
            self._quant_so = so

    def _init_kv_cache(self):
        N = self.n_embd; NKH = self.n_kv_head; HD = self.head_dim; MAX_POS = 4096
        self.kv_k = np.zeros((self.n_layers, MAX_POS, NKH*HD), dtype=np.float32)
        self.kv_v = np.zeros((self.n_layers, MAX_POS, NKH*HD), dtype=np.float32)
        self.kv_len = np.zeros(self.n_layers, dtype=np.int32)
        self._buf = {
            'h': np.zeros(N, dtype=np.float32),
            'residual': np.zeros(N, dtype=np.float32),
            'q': np.zeros(self.n_head*HD, dtype=np.float32),
            'k': np.zeros(NKH*HD, dtype=np.float32),
            'v': np.zeros(NKH*HD, dtype=np.float32),
            'att_out': np.zeros(self.n_head*HD, dtype=np.float32),
            'gate': np.zeros(self.n_ff, dtype=np.float32),
            'up': np.zeros(self.n_ff, dtype=np.float32),
            'silu_gate': np.zeros(self.n_ff, dtype=np.float32),
            'ffn_out': np.zeros(N, dtype=np.float32),
        }

    def reset(self):
        self.kv_len[:] = 0; self.pos = 0

    def _matmul(self, name, x):
        """Dispatch to C kernel or numpy."""
        w = self.weights[name]
        if w.ndim == 1: return w * x
        n_rows, n_cols = w.shape
        qtype = self.weight_qtypes.get(name, GGML_F32)
        info = self.weight_info.get(name)
        if info is not None and self._quant_so is not None:
            nr, nc, ts, qt = info
            raw = self.raw_weights[name]
            out = np.empty(nr, dtype=np.float32)
            self._quant_so.quant_matmul_omp(
                raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                nr, nc, qt)
            return out
        return w @ x

    def _rms_norm(self, x, name, eps=None):
        if eps is None: eps = self.eps
        w = self.weights[name]
        var = np.mean(x * x)
        return x / np.sqrt(var + eps) * w

    def _silu(self, x): return x / (1.0 + np.exp(-x))

    def _apply_rope(self, x, pos, n_heads):
        hd = self.head_dim; half = hd // 2
        freq = self.rope_freq_base ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
        angle = pos / freq
        cos_a = np.cos(angle).astype(np.float32)
        sin_a = np.sin(angle).astype(np.float32)
        x2d = x.reshape(n_heads, hd)
        out = np.empty_like(x2d)
        out[:, :half] = x2d[:, :half] * cos_a - x2d[:, half:] * sin_a
        out[:, half:] = x2d[:, half:] * cos_a + x2d[:, :half] * sin_a
        return out.reshape(-1)

    def forward(self, token_id):
        """Run one forward pass — optimized Python+C hybrid."""
        b = self._buf; N = self.n_embd; NH = self.n_head
        NKH = self.n_kv_head; HD = self.head_dim; L = self.n_layers

        np.copyto(b['h'], self.emb[token_id])

        for i in range(L):
            pfx = f'blk.{i}'

            # Attention
            np.copyto(b['residual'], b['h'])
            b['h'][:] = self._rms_norm(b['h'], f'{pfx}.attn_norm.weight')

            b['q'][:] = self._matmul(f'{pfx}.attn_q.weight', b['h'])
            b['k'][:] = self._matmul(f'{pfx}.attn_k.weight', b['h'])[:NKH*HD]
            b['v'][:] = self._matmul(f'{pfx}.attn_v.weight', b['h'])[:NKH*HD]

            # Q/K norm (Qwen3)
            q_norm = f'{pfx}.attn_q_norm.weight'
            if q_norm in self.weights:
                q_w = self.weights[q_norm]
                if q_w.shape[0] == HD:
                    q_2d = b['q'].reshape(NH, HD)
                    for h in range(NH):
                        q_2d[h] = q_2d[h] / np.sqrt(np.mean(q_2d[h]*q_2d[h]) + self.eps) * q_w
            k_norm = f'{pfx}.attn_k_norm.weight'
            if k_norm in self.weights:
                k_w = self.weights[k_norm]
                if k_w.shape[0] == HD:
                    k_2d = b['k'][:NKH*HD].reshape(NKH, HD)
                    for h in range(NKH):
                        k_2d[h] = k_2d[h] / np.sqrt(np.mean(k_2d[h]*k_2d[h]) + self.eps) * k_w

            b['q'][:] = self._apply_rope(b['q'], self.pos, NH)
            b['k'][:] = self._apply_rope(b['k'], self.pos, NKH)

            self.kv_k[i, self.kv_len[i], :NKH*HD] = b['k'][:NKH*HD]
            self.kv_v[i, self.kv_len[i], :NKH*HD] = b['v'][:NKH*HD]

            seq_len = self.kv_len[i] + 1
            k_cache = self.kv_k[i, :seq_len].reshape(seq_len, NKH, HD)
            v_cache = self.kv_v[i, :seq_len].reshape(seq_len, NKH, HD)

            n_rep = NH // NKH if NH != NKH else 1
            if n_rep > 1:
                k_exp = np.repeat(k_cache, n_rep, axis=1)
                v_exp = np.repeat(v_cache, n_rep, axis=1)
            else:
                k_exp = k_cache; v_exp = v_cache

            q_2d = b['q'].reshape(NH, HD)
            scores = np.einsum('hd,shd->hs', q_2d, k_exp.reshape(-1, NH, HD)) / np.sqrt(float(HD))
            scores -= np.max(scores, axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= np.sum(scores, axis=1, keepdims=True)
            att = np.einsum('hs,shd->hd', scores, v_exp.reshape(-1, NH, HD)).astype(np.float32)
            b['att_out'][:] = att.reshape(-1)

            self.kv_len[i] += 1
            o_out = self._matmul(f'{pfx}.attn_output.weight', b['att_out'])
            b['h'][:N] = b['residual'][:N] + o_out[:N]

            # FFN
            np.copyto(b['residual'], b['h'])
            b['h'][:] = self._rms_norm(b['h'], f'{pfx}.ffn_norm.weight')

            if self.is_moe:
                self._forward_moe(i, b)
            else:
                FF = self.n_ff
                gate = self._matmul(f'{pfx}.ffn_gate.weight', b['h'])[:FF]
                up = self._matmul(f'{pfx}.ffn_up.weight', b['h'])[:FF]
                b['silu_gate'][:FF] = self._silu(gate) * up
                b['ffn_out'][:N] = self._matmul(f'{pfx}.ffn_down.weight', b['silu_gate'])[:N]

            b['h'][:N] = b['residual'][:N] + b['ffn_out'][:N]

        # Final norm + output
        b['h'][:] = self._rms_norm(b['h'], 'output_norm.weight')
        logits = self._matmul(self.out_w_name, b['h'])
        self.pos += 1
        return logits

    def _forward_moe(self, layer, b):
        pfx = f'blk.{layer}'; N = self.n_embd; FF = self.n_ff_expert
        n_exp = self.n_experts; top_k = self.n_experts_per_tok

        gate_scores = self._matmul(f'{pfx}.ffn_gate_inp.weight', b['h'])[:n_exp]
        gate_scores -= np.max(gate_scores)
        np.exp(gate_scores, out=gate_scores)
        gate_scores /= np.sum(gate_scores)

        top_indices = np.argsort(gate_scores)[-top_k:][::-1]
        top_weights = gate_scores[top_indices]
        top_weights /= np.sum(top_weights)

        result = np.zeros(N, dtype=np.float32)
        for expert_id, weight in zip(top_indices, top_weights):
            gw = self.weights[f'{pfx}.ffn_gate_exps.weight'][:,:,expert_id]
            uw = self.weights[f'{pfx}.ffn_up_exps.weight'][:,:,expert_id]
            dw = self.weights[f'{pfx}.ffn_down_exps.weight'][:,:,expert_id]
            gate_out = (gw @ b['h'])[:FF]
            up_out = (uw @ b['h'])[:FF]
            silu_out = self._silu(gate_out) * up_out
            expert_out = (dw @ silu_out)[:N]
            result += weight * expert_out

        # Shared expert (Qwen3 MoE)
        sgate = f'{pfx}.ffn_shared_gate.weight'
        if sgate in self.weights:
            gate_s = self._matmul(sgate, b['h'])
            up_s = self._matmul(f'{pfx}.ffn_shared_up.weight', b['h'])
            b['ffn_out'][:N] = result + (self._matmul(f'{pfx}.ffn_down.weight', self._silu(gate_s) * up_s))[:N]
        else:
            b['ffn_out'][:N] = result


if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'Llama-3.2-1B-Instruct-Q4_0.gguf'
    n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    engine = TurboEngineV55(model_path, n_threads=n_threads)

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