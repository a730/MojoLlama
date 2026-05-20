#!/usr/bin/env python3
"""Profile Qwen3.6 MXFP4 by instrumenting the actual forward pass."""
import sys, os, time, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"

# Monkey-patch timing into actual engine
import turbo_engine_v7_moe as te_mod
orig_forward = te_mod.TurboEngineV7MoE.forward
orig_fwd_moe = te_mod.TurboEngineV7MoE._forward_moe
orig_apply_rope = te_mod.TurboEngineV7MoE._apply_rope_fast

def timed_forward(self, token_id):
    """Instrumented forward with per-layer timing."""
    import time as _t
    N = self.n_embd; NH = self.n_head; NKH = self.n_kv_head * self.head_dim
    HD = self.head_dim; L = self.n_layers
    
    # Counters
    self._timing = getattr(self, '_timing', {
        'ssm_decode': 0.0, 'ssm_gate': 0.0, 'ssm_gate_mul': 0.0, 'ssm_out': 0.0,
        'attn_qkv': 0.0, 'attn_split': 0.0, 'rope': 0.0, 'kv_cache': 0.0, 'gqa': 0.0,
        'attn_proj': 0.0, 'ffn_rms': 0.0, 'ffn_moe': 0.0, 'ffn_shexp': 0.0,
        'rms1': 0.0, 'rms2': 0.0, 'residual': 0.0, 'final': 0.0,
        'count': 0, 'attn_layers': 0, 'ssm_layers': 0
    })
    tim = self._timing
    
    import ctypes
    b_x = self._x; b_r = self._residual
    N = self.n_embd; NH = self.n_head; NKH = self.n_kv_head * self.head_dim
    HD = self.head_dim; L = self.n_layers
    kern = self._kern; simd = self._simd
    cf = ctypes.POINTER(ctypes.c_float); ci = ctypes.c_int
    
    tim['count'] += 1
    t_all = _t.perf_counter()
    
    # Embedding
    np.copyto(b_x, self.emb[token_id])
    
    for i in range(L):
        lw = self._layers[i]
        is_ssm = (self.layer_types is not None and self.layer_types[i] == 1)
        
        # Residual copy
        t0 = _t.perf_counter()
        np.copyto(b_r, b_x)
        tim['residual'] += _t.perf_counter() - t0
        
        # RMS norm 1
        t0 = _t.perf_counter()
        simd.rms_norm(self._p_x_norm, self._p_x,
                      lw.attn_norm_w.ctypes.data_as(cf), N, self._eps_f)
        tim['rms1'] += _t.perf_counter() - t0
        
        if is_ssm and self._cengine is not None and lw.ssm_conv1d_ptr is not None:
            # ═══ SSM path ═══
            tim['ssm_layers'] += 1
            ce = self._cengine
            t0 = _t.perf_counter()
            ce.ssm_decode_step(
                self._p_x_norm, self._ssm_intermediate.ctypes.data_as(cf),
                lw.ssm_conv1d_ptr, lw.ssm_a_ptr, lw.ssm_dt_bias_ptr,
                lw.ssm_alpha_ptr, lw.ssm_beta_ptr, lw.ssm_norm_ptr,
                self._ssm_state[i].ctypes.data_as(cf),
                ci(1), ci(N), ci(self.ssm_inner),
                ci(self.ssm_groups), ci(self.ssm_state_size),
                ci(self.ssm_conv_kernel), ci(self.ssm_dt_rank))
            tim['ssm_decode'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            kern.quant_matmul_omp(lw.attn_gate_raw, self._p_x_norm,
                self._gate_4096.ctypes.data_as(cf),
                lw.attn_gate_nr, lw.attn_gate_nc, lw.attn_gate_qt)
            tim['ssm_gate'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            np.multiply(self._gate_4096, self._ssm_intermediate, out=self._gate_4096)
            tim['ssm_gate_mul'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            kern.quant_matmul_omp(lw.ssm_out_raw,
                self._gate_4096.ctypes.data_as(cf), self._p_o_proj,
                lw.ssm_out_nr, lw.ssm_out_nc, lw.ssm_out_qt)
            tim['ssm_out'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            self._x[:N] = self._residual[:N] + self._o_proj[:N]
            tim['residual'] += _t.perf_counter() - t0
            
        else:
            # ═══ Attention path ═══
            tim['attn_layers'] += 1
            t0 = _t.perf_counter()
            qkv_buf = self._qk
            if len(self._qk) < 8192:
                self._qk = np.zeros(8192, dtype=np.float32)
                self._p_qk = self._qk.ctypes.data_as(cf)
            p_qkv = self._p_qk
            kern.quant_matmul_omp(lw.attn_qkv_raw, self._p_x_norm, p_qkv,
                lw.attn_qkv_nr, lw.attn_qkv_nc, lw.attn_qkv_qt)
            tim['attn_qkv'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            nq = NH * HD; nk = NKH
            self._q[:] = self._qk[:nq]
            self._k[:nk] = self._qk[nq:nq+nk]
            self._v[:nk] = self._qk[nq+N:nq+N+nk]
            tim['attn_split'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            rope_d = self.rope_dim if hasattr(self, 'rope_dim') and self.rope_dim > 0 else HD
            self._q[:] = self._apply_rope_fast(self._q, self.pos, NH, rope_dim=rope_d)
            self._k[:] = self._apply_rope_fast(self._k, self.pos, self.n_kv_head, rope_dim=rope_d)
            tim['rope'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            self.kv_k[i, self.kv_len[i], :NKH] = self._k[:NKH]
            self.kv_v[i, self.kv_len[i], :NKH] = self._v[:NKH]
            tim['kv_cache'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            seq_len = self.kv_len[i] + 1
            if self._gqa_attn is not None and seq_len <= 4096:
                self._gqa_attn.gqa_attention_decode(
                    self._q.ctypes.data_as(cf),
                    self.kv_k[i].ctypes.data_as(cf),
                    self.kv_v[i].ctypes.data_as(cf),
                    self._att_out.ctypes.data_as(cf),
                    ci(seq_len), ci(NH), ci(self.n_kv_head), ci(HD))
            self.kv_len[i] += 1
            tim['gqa'] += _t.perf_counter() - t0
            
            t0 = _t.perf_counter()
            kern.quant_matmul_omp(lw.attn_gate_raw, self._p_att_out, self._p_o_proj,
                lw.attn_gate_nr, lw.attn_gate_nc, lw.attn_gate_qt)
            self._x[:N] = self._residual[:N] + self._o_proj[:N]
            tim['attn_proj'] += _t.perf_counter() - t0
        
        # ── FFN ──
        t0 = _t.perf_counter()
        np.copyto(self._residual, self._x)
        simd.rms_norm(self._p_x_norm, self._p_x,
                      lw.ffn_norm_w.ctypes.data_as(cf), N, self._eps_f)
        tim['ffn_rms'] += _t.perf_counter() - t0
        
        t0 = _t.perf_counter()
        self._forward_moe(i, self._p_x_norm, self._ffn_out, self._residual, N)
        tim['ffn_moe'] += _t.perf_counter() - t0
        
        # Shared expert
        if hasattr(lw, 'shexp_router_ptr') and lw.shexp_router_ptr is not None:
            t0 = _t.perf_counter()
            shexp_score = float(np.dot(self._x_norm,
                np.ctypeslib.as_array(lw.shexp_router_ptr, shape=(N,))))
            if shexp_score > 0:
                shexp_int = lw.shexp_gate_nr.value
                kern.quant_matmul_omp(lw.shexp_gate_raw, self._p_x_norm, self._p_gate,
                    lw.shexp_gate_nr, lw.shexp_gate_nc, lw.shexp_gate_qt)
                kern.quant_matmul_omp(lw.shexp_up_raw, self._p_x_norm, self._p_up,
                    lw.shexp_up_nr, lw.shexp_up_nc, lw.shexp_up_qt)
                simd.silu(self._p_silu_gate, self._p_gate, ci(shexp_int))
                self._silu_gate[:shexp_int] *= self._up[:shexp_int]
                kern.quant_matmul_omp(lw.shexp_down_raw, self._p_silu_gate, self._p_ffn_out,
                    lw.shexp_down_nr, lw.shexp_down_nc, lw.shexp_down_qt)
                self._x[:N] += shexp_score * self._ffn_out[:N]
            tim['ffn_shexp'] += _t.perf_counter() - t0
    
    # Final norm + output
    t0 = _t.perf_counter()
    simd.rms_norm(self._p_x_norm, self._p_x,
                  self._out_norm_w.ctypes.data_as(cf), N, self._eps_f)
    kern.quant_matmul_omp(self._out_raw, self._p_x_norm, self._p_logits,
                          self._out_nr, self._out_nc, self._out_qt)
    self.pos += 1
    tim['final'] += _t.perf_counter() - t0
    
    return self._logits

te_mod.TurboEngineV7MoE.forward = timed_forward

print(f"Loading Qwen3.6 MXFP4...", flush=True)
e = te_mod.TurboEngineV7MoE(MODEL, 32)
N=e.n_embd; L=e.n_layers; NH=e.n_head; NKH=e.n_kv_head; HD=e.head_dim

print("\nWarmup...", flush=True)
for _ in range(5):
    e.forward(np.array([1], dtype=np.int32))
e._timing = {k: 0.0 for k in e._timing}

# Benchmark 30 tokens
print("Benchmarking 30 tokens...", flush=True)
for tok in range(30):
    e.forward(np.array([tok], dtype=np.int32))

tim = e._timing
n = tim['count']
total = sum(v for k,v in tim.items() if k not in ('count','attn_layers','ssm_layers'))

print(f"\n{'='*60}")
print(f"  Qwen3.6 MXFP4 — Per-Component Breakdown (30 tokens, {n} samples)")
print(f"{'='*60}")
print(f"  {'Component':25s} {'ms/tok':>10s} {'%':>6s}")
print(f"  {'-'*43}")

components = [
    ('RMS Norm (pre)', 'rms1'),
    ('Residual copy', 'residual'),
    ('SSM decode (C)', 'ssm_decode'),
    ('SSM attn_gate matmul', 'ssm_gate'),
    ('SSM gate multiply', 'ssm_gate_mul'),
    ('SSM out matmul', 'ssm_out'),
    ('Attention QKV matmul', 'attn_qkv'),
    ('Attention QKV split', 'attn_split'),
    ('RoPE', 'rope'),
    ('KV cache store', 'kv_cache'),
    ('GQA attention', 'gqa'),
    ('Attn output proj', 'attn_proj'),
    ('FFN RMS+residual', 'ffn_rms'),
    ('MoE FFN (fused)', 'ffn_moe'),
    ('Shared expert', 'ffn_shexp'),
    ('Final norm+output', 'final'),
]

for name, key in components:
    if key in tim:
        ms = tim[key] / n * 1000
        print(f"  {name:25s} {ms:10.1f} {ms/total*100:5.0f}%")

print(f"  {'-'*43}")
print(f"  {'TOTAL':25s} {total/n*1000:10.1f} {'100':5s}%")
print(f"  {'Throughput':25s} {n/(total):10.1f} tok/s")
print(f"\n  {tim['attn_layers']/n:.1f} attn layers/tok, {tim['ssm_layers']/n:.1f} SSM layers/tok")
