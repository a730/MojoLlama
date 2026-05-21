#!/usr/bin/env python3
"""Profile Qwen3.6 forward pass to find bottlenecks."""
import sys, os, time, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turbo_engine_v7_moe import TurboEngineV7MoE

MODEL = '/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf'
OMP_NUM_THREADS = os.environ.get('OMP_NUM_THREADS', '32')

e = TurboEngineV7MoE(MODEL, n_threads=int(OMP_NUM_THREADS))
cf = ctypes.POINTER(ctypes.c_float)
cu = ctypes.POINTER(ctypes.c_uint8)
ci = ctypes.c_int

N = e.n_embd; NH = e.n_head; NKH = e.n_kv_head * e.head_dim
HD = e.head_dim; FF = e.n_ff; L = e.n_layers
kern = e._kern; simd = e._simd
eps_f = e._eps_f

# Pre-alloc
b_x = e._x; b_r = e._residual; b_xn = e._x_norm
p_x = e._p_x; p_xn = e._p_x_norm; p_r = e._p_residual
p_oproj = e._p_o_proj
b_att = e._att_out; p_att = e._p_att_out
b_ffn = e._ffn_out
ssm_inner = e.ssm_inner

e.reset()
token_id = 42
np.copyto(b_x, e.emb[token_id])

profile = {}

for i in range(L):
    lw = e._layers[i]
    is_ssm = (e.layer_types is not None and e.layer_types[i] == 1)
    t_lyr0 = time.perf_counter()
    
    # Residual copy
    b_r[:] = b_x
    
    # RMS norm attn
    t0 = time.perf_counter()
    simd.rms_norm(p_xn, p_x, lw.attn_norm_w.ctypes.data_as(cf), N, eps_f)
    t_rms1 = time.perf_counter() - t0
    
    if is_ssm and e._cengine is not None and lw.ssm_conv1d_ptr is not None:
        # SSM path
        t0 = time.perf_counter()
        state_ptr = e._ssm_state[i].ctypes.data_as(cf)
        interm_ptr = e._ssm_intermediate.ctypes.data_as(cf)
        e._cengine.ssm_decode_step(
            p_xn, interm_ptr, lw.ssm_conv1d_ptr, lw.ssm_a_ptr, lw.ssm_dt_bias_ptr,
            lw.ssm_alpha_ptr, lw.ssm_beta_ptr, lw.ssm_norm_ptr, state_ptr,
            ci(1), ci(N), ci(ssm_inner), ci(e.ssm_groups), ci(e.ssm_state_size),
            ci(e.ssm_conv_kernel), ci(e.ssm_dt_rank))
        t_ssm = time.perf_counter() - t0
        
        # attn_gate matmul
        t0 = time.perf_counter()
        kern.quant_matmul_omp(lw.attn_gate_raw, p_xn, e._gate_4096.ctypes.data_as(cf),
                              lw.attn_gate_nr, lw.attn_gate_nc, lw.attn_gate_qt)
        t_gate = time.perf_counter() - t0
        
        # element-wise multiply
        t0 = time.perf_counter()
        np.multiply(e._gate_4096, e._ssm_intermediate, out=e._gate_4096)
        t_mul = time.perf_counter() - t0
        
        # ssm_out matmul
        t0 = time.perf_counter()
        kern.quant_matmul_omp(lw.ssm_out_raw, e._gate_4096.ctypes.data_as(cf), p_oproj,
                              lw.ssm_out_nr, lw.ssm_out_nc, lw.ssm_out_qt)
        t_out = time.perf_counter() - t0
        
        # residual add
        t0 = time.perf_counter()
        b_x[:N] = b_r[:N] + e._o_proj[:N]
        t_res = time.perf_counter() - t0
        
        # FFN section
        t0 = time.perf_counter()
        b_r[:] = b_x
        simd.rms_norm(p_xn, p_x, lw.ffn_norm_w.ctypes.data_as(cf), N, eps_f)
        t_rms2 = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        e._forward_moe(i, p_xn, b_ffn, b_r, N)
        t_moe = time.perf_counter() - t0
        
        # Shared expert
        t0 = time.perf_counter()
        if (hasattr(lw, 'shexp_router_ptr') and lw.shexp_router_ptr is not None):
            shexp_score = float(np.dot(e._x_norm,
                np.ctypeslib.as_array(lw.shexp_router_ptr, shape=(N,))))
            if shexp_score > 0:
                shexp_int = lw.shexp_gate_nr.value
                kern.quant_matmul_omp(lw.shexp_gate_raw, p_xn, e._p_gate,
                                      lw.shexp_gate_nr, lw.shexp_gate_nc, lw.shexp_gate_qt)
                kern.quant_matmul_omp(lw.shexp_up_raw, p_xn, e._p_up,
                                      lw.shexp_up_nr, lw.shexp_up_nc, lw.shexp_up_qt)
                simd.silu(e._p_silu_gate, e._p_gate, ci(shexp_int))
                e._silu_gate[:shexp_int] *= e._up[:shexp_int]
                kern.quant_matmul_omp(lw.shexp_down_raw, e._p_silu_gate, e._p_ffn_out,
                                      lw.shexp_down_nr, lw.shexp_down_nc, lw.shexp_down_qt)
                b_x[:N] += shexp_score * e._ffn_out[:N]
        t_shexp = time.perf_counter() - t0
        
        phase = 'ssm'
        phases = {
            'rms_norm_attn': t_rms1, 'ssm_decode': t_ssm, 'gate_matmul': t_gate,
            'elem_mul': t_mul, 'out_matmul': t_out, 'residual': t_res,
            'rms_norm_ffn': t_rms2, 'moe': t_moe,
        }
    else:
        # Attention path
        # QKV matmuls
        t0 = time.perf_counter()
        if hasattr(lw, 'attn_qkv_use_c') and lw.attn_qkv_use_c:
            qkv_buf = e._qk if len(e._qk) >= 8192 else np.zeros(8192, dtype=np.float32)
            p_qkv = qkv_buf.ctypes.data_as(cf)
            kern.quant_matmul_omp(lw.attn_qkv_raw, p_xn, p_qkv,
                                  lw.attn_qkv_nr, lw.attn_qkv_nc, lw.attn_qkv_qt)
            nq = NH * HD; nkh = e.n_kv_head * HD
            e._q[:nq] = qkv_buf[:nq]
            e._k[:nkh] = qkv_buf[nq:nq + nkh]
            e._v[:nkh] = qkv_buf[nq + N:nq + N + nkh]
        elif hasattr(lw, 'attn_qk_use_c') and lw.attn_qk_use_c:
            kern.quant_matmul_omp(lw.attn_qk_raw, p_xn, e._p_qk,
                                  lw.attn_qk_nr, lw.attn_qk_nc, lw.attn_qk_qt)
            kern.quant_matmul_omp(lw.attn_v_raw, p_xn, e._p_v,
                                  lw.attn_v_nr, lw.attn_v_nc, lw.attn_v_qt)
        else:
            kern.batch_qkv_omp(lw.attn_q_raw, lw.attn_k_raw, lw.attn_v_raw,
                               p_xn, e._p_q, e._p_k, e._p_v,
                               lw.attn_q_nr, lw.attn_k_nr, lw.attn_v_nr,
                               lw.attn_q_nc, lw.attn_q_qt, lw.attn_k_qt, lw.attn_v_qt)
        t_qkv = time.perf_counter() - t0
        
        # RoPE
        t0 = time.perf_counter()
        rope_d = e.rope_dim if hasattr(e, 'rope_dim') and e.rope_dim > 0 else HD
        e._q[:] = e._apply_rope_fast(e._q, e.pos, NH, rope_dim=rope_d)
        e._k[:] = e._apply_rope_fast(e._k, e.pos, e.n_kv_head, rope_dim=rope_d)
        t_rope = time.perf_counter() - t0
        
        # KV cache
        t0 = time.perf_counter()
        e.kv_k[i, e.kv_len[i], :NKH] = e._k[:NKH]
        e.kv_v[i, e.kv_len[i], :NKH] = e._v[:NKH]
        seq_len = e.kv_len[i] + 1
        t_kvcache = time.perf_counter() - t0
        
        # GQA attention
        t0 = time.perf_counter()
        if e._gqa_attn is not None and seq_len <= 4096:
            e._gqa_attn.gqa_attention_decode(
                e._p_q, e.kv_k[i].ctypes.data_as(cf), e.kv_v[i].ctypes.data_as(cf),
                p_att, ci(seq_len), ci(NH), ci(e.n_kv_head), ci(HD), e._p_gqa_ws)
        else:
            k_cache = e.kv_k[i, :seq_len].reshape(seq_len, e.n_kv_head, HD)
            v_cache = e.kv_v[i, :seq_len].reshape(seq_len, e.n_kv_head, HD)
            q_2d = e._q.reshape(NH, HD)
            nk = e.n_kv_head; gqa = e._gqa_rep
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
        e.kv_len[i] += 1
        t_gqa = time.perf_counter() - t0
        
        # Output projection
        t0 = time.perf_counter()
        if hasattr(lw, 'attn_gate_use_c') and lw.attn_gate_use_c:
            kern.quant_matmul_omp(lw.attn_gate_raw, p_att, p_oproj,
                                  lw.attn_gate_nr, lw.attn_gate_nc, lw.attn_gate_qt)
            b_x[:N] = b_r[:N] + e._o_proj[:N]
        else:
            kern.quant_matmul_omp(lw.attn_out_raw, p_att, p_oproj,
                                  lw.attn_out_nr, lw.attn_out_nc, lw.attn_out_qt)
            b_x[:N] = b_r[:N] + e._o_proj[:N]
        t_out = time.perf_counter() - t0
        
        # FFN
        t0 = time.perf_counter()
        b_r[:] = b_x
        simd.rms_norm(p_xn, p_x, lw.ffn_norm_w.ctypes.data_as(cf), N, eps_f)
        t_rms2 = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        e._forward_moe(i, p_xn, b_ffn, b_r, N)
        t_moe = time.perf_counter() - t0
        
        # Shared expert
        t0 = time.perf_counter()
        if (hasattr(lw, 'shexp_router_ptr') and lw.shexp_router_ptr is not None):
            shexp_score = float(np.dot(e._x_norm,
                np.ctypeslib.as_array(lw.shexp_router_ptr, shape=(N,))))
            if shexp_score > 0:
                shexp_int = lw.shexp_gate_nr.value
                kern.quant_matmul_omp(lw.shexp_gate_raw, p_xn, e._p_gate,
                                      lw.shexp_gate_nr, lw.shexp_gate_nc, lw.shexp_gate_qt)
                kern.quant_matmul_omp(lw.shexp_up_raw, p_xn, e._p_up,
                                      lw.shexp_up_nr, lw.shexp_up_nc, lw.shexp_up_qt)
                simd.silu(e._p_silu_gate, e._p_gate, ci(shexp_int))
                e._silu_gate[:shexp_int] *= e._up[:shexp_int]
                kern.quant_matmul_omp(lw.shexp_down_raw, e._p_silu_gate, e._p_ffn_out,
                                      lw.shexp_down_nr, lw.shexp_down_nc, lw.shexp_down_qt)
                b_x[:N] += shexp_score * e._ffn_out[:N]
        t_shexp = time.perf_counter() - t0
        
        phase = 'attn'
        phases = {
            'rms_norm_attn': t_rms1, 'qkv': t_qkv, 'rope': t_rope,
            'kv_cache': t_kvcache, 'gqa_attention': t_gqa, 'out_proj': t_out,
            'rms_norm_ffn': t_rms2, 'moe': t_moe,
        }
    
    t_lyr1 = time.perf_counter()
    total = t_lyr1 - t_lyr0
    phases['python_overhead'] = total - sum(phases.values())
    profile[i] = {'phase': phase, 'total_ms': total * 1000, **{k: v*1000 for k, v in phases.items()}}

# Aggregate
agg = {}
for i in profile:
    for k, v in profile[i].items():
        if k not in agg:
            agg[k] = {'sum': 0.0, 'count': 0}
        if isinstance(v, (int, float)):
            agg[k]['sum'] += v
            agg[k]['count'] += 1

total_ms = agg['total_ms']['sum']
print(f'\n{"Section":20s} {"Avg (ms/lyr)":>14s} {"Total (ms)":>14s} {"%":>7s}')
print('-' * 60)

# Group by major category
categories = {
    'RMS Norms': ['rms_norm_attn', 'rms_norm_ffn'],
    'Matmuls (QKV/Gate/Out)': ['qkv', 'out_proj'],
    'SSM Path': ['ssm_decode', 'gate_matmul', 'elem_mul', 'out_matmul'],
    'Attention': ['rope', 'kv_cache', 'gqa_attention'],
    'MoE FFN': ['moe'],
}

for cat, keys in categories.items():
    cat_sum = sum(agg[k]['sum'] for k in keys if k in agg)
    cat_avg = cat_sum / L
    print(f'{cat:20s} {cat_avg:9.3f}ms     {cat_sum:9.1f}ms    {cat_sum/total_ms*100:5.1f}%')

# Remaining phases
seen = set()
for keys in categories.values():
    seen.update(keys)
for k, v in sorted(agg.items()):
    if k not in ('phase', 'total_ms') and k not in seen and isinstance(v['sum'], (int, float)):
        avg = v['sum'] / v['count']
        print(f'{k:20s} {avg:9.3f}ms     {v["sum"]:9.1f}ms    {v["sum"]/total_ms*100:5.1f}%')

print(f'{"Total":20s} {"":>14s} {total_ms:9.1f}ms    {100:5.1f}%')
print(f'  Effective: {1000/(total_ms/1000):.1f} tok/s')
print(f'  Lyr type breakdown: attn={sum(1 for i in profile if profile[i]["phase"]=="attn")}, ssm={sum(1 for i in profile if profile[i]["phase"]=="ssm")}')
