#!/usr/bin/env python3
"""
Qwen3.6 MXFP4 threaded server — weight-efficient.
Master engine loaded once. Child threads share weight refs, get fresh buffers.
"""
import sys, os, time, threading, ctypes, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"
N_WORKERS = 1
GEN_TOKENS = 50
OMP_PER = 1  # each thread runs single-threaded (avoids OMP pool contention)

os.environ['OMP_NUM_THREADS'] = str(OMP_PER)

from turbo_engine_v7_moe import TurboEngineV7MoE

def create_child(master):
    """Create a lightweight child engine sharing master's weights."""
    child = TurboEngineV7MoE.__new__(TurboEngineV7MoE)
    # Share read-only data (references — no copy)
    for attr in ['n_embd','n_head','n_kv_head','head_dim','n_ff','n_layers',
                 'vocab_size','eps','n_experts','n_experts_per_tok','n_ff_expert',
                 'is_moe','arch_prefix','rope_dim','rope_freq_base',
                 'full_attn_interval','n_layers_actual',
                 'ssm_groups','ssm_state_size','ssm_dt_rank','ssm_conv_kernel','ssm_inner',
                 'layer_types','n_threads','model_path',
                 'weights','raw_weights','weight_qtypes','weight_info',
                 'emb','reader',
                 '_kern','_simd','_cengine','_gqa_attn',
                 '_layers','_moe_layers',
                 '_out_raw','_out_nr','_out_nc','_out_qt','_out_use_c',
                 '_out_norm_w','_out_f32',
                 'out_w_name']:
        if hasattr(master, attr):
            setattr(child, attr, getattr(master, attr))
    
    # Create fresh per-thread buffers (same as _init_buffers)
    N = master.n_embd
    NK = master.n_head * master.head_dim
    NKH = master.n_kv_head * master.head_dim
    MAX_POS = 4096
    FF_expert = master.n_ff_expert if master.is_moe else master.n_ff
    FF = master.n_ff
    
    child.kv_k = np.zeros((master.n_layers, MAX_POS, NKH), dtype=np.float32)
    child.kv_v = np.zeros((master.n_layers, MAX_POS, NKH), dtype=np.float32)
    child.kv_len = np.zeros(master.n_layers, dtype=np.int32)
    
    child._x = np.zeros(N, dtype=np.float32)
    child._x_norm = np.zeros(N, dtype=np.float32)
    child._residual = np.zeros(N, dtype=np.float32)
    qkv_sz = max(NK + NKH, 8192) if getattr(master, 'arch_prefix', '') == 'qwen35moe' else NK + NKH
    child._qk = np.zeros(qkv_sz, dtype=np.float32)
    child._q = child._qk[:NK]
    child._k = child._qk[NK:NK+NKH]
    child._v = np.zeros(NKH, dtype=np.float32)
    child._att_out = np.zeros(NK, dtype=np.float32)
    child._o_proj = np.zeros(N, dtype=np.float32)
    child._gate = np.zeros(FF, dtype=np.float32)
    child._up = np.zeros(FF, dtype=np.float32)
    child._silu_gate = np.zeros(FF, dtype=np.float32)
    child._ffn_out = np.zeros(N, dtype=np.float32)
    
    n_exp = max(128, master.n_experts) if master.is_moe else 128
    child._moe_router_scores = np.zeros(n_exp, dtype=np.float32)
    child._moe_gate_out = np.zeros(FF_expert, dtype=np.float32)
    child._moe_up_out = np.zeros(FF_expert, dtype=np.float32)
    child._moe_silu_out = np.zeros(FF_expert, dtype=np.float32)
    child._moe_expert_out = np.zeros(N, dtype=np.float32)
    child._moe_combined = np.zeros(N, dtype=np.float32)
    max_top_k = max(8, master.n_experts_per_tok)
    child._moe_prealloc_buf = np.zeros(3 * max_top_k * FF_expert, dtype=np.float32)
    child._moe_prealloc_q8 = np.zeros((N // 32) * 34, dtype=np.uint8)
    child._logits = np.zeros(master.vocab_size, dtype=np.float32)
    
    if master.arch_prefix == 'qwen35moe':
        child._ssm_state = np.zeros((master.n_layers, master.ssm_groups, master.ssm_state_size), dtype=np.float32)
        child._ssm_intermediate = np.zeros(master.ssm_inner, dtype=np.float32)
        child._gate_4096 = np.zeros(master.ssm_inner, dtype=np.float32)
    else:
        child._ssm_state = None; child._ssm_intermediate = None; child._gate_4096 = None
    
    # ctypes pointers
    cf = ctypes.POINTER(ctypes.c_float)
    child._p_x = child._x.ctypes.data_as(cf)
    child._p_x_norm = child._x_norm.ctypes.data_as(cf)
    child._p_residual = child._residual.ctypes.data_as(cf)
    child._p_q = child._q.ctypes.data_as(cf)
    child._p_k = child._k.ctypes.data_as(cf)
    child._p_v = child._v.ctypes.data_as(cf)
    child._p_qk = child._qk.ctypes.data_as(cf)
    child._p_att_out = child._att_out.ctypes.data_as(cf)
    child._p_gate = child._gate.ctypes.data_as(cf)
    child._p_up = child._up.ctypes.data_as(cf)
    child._p_silu_gate = child._silu_gate.ctypes.data_as(cf)
    child._p_o_proj = child._o_proj.ctypes.data_as(cf)
    child._p_ffn = child._ffn_out.ctypes.data_as(cf)
    child._p_logits = child._logits.ctypes.data_as(cf)
    child._p_moe_router = child._moe_router_scores.ctypes.data_as(cf)
    child._p_moe_gate = child._moe_gate_out.ctypes.data_as(cf)
    child._p_moe_up = child._moe_up_out.ctypes.data_as(cf)
    child._p_moe_silu = child._moe_silu_out.ctypes.data_as(cf)
    child._p_moe_expert = child._moe_expert_out.ctypes.data_as(cf)
    child._p_moe_combined = child._moe_combined.ctypes.data_as(cf)
    
    child._rope_cos_table = {}
    child._rope_sin_table = {}
    child._eps_f = ctypes.c_float(master.eps)
    child._gqa_rep = master.n_head // master.n_kv_head if master.n_head != master.n_kv_head else 1
    child.pos = 0
    
    return child

# ── Server ──
print(f"Loading master engine...", flush=True)
master = TurboEngineV7MoE(MODEL, 32)
print(f"Creating {N_WORKERS} children...", flush=True)

results = [None] * N_WORKERS

def worker(tid):
    os.environ['OMP_NUM_THREADS'] = str(OMP_PER)
    try:
        e = create_child(master)
        # Warmup
        for _ in range(3):
            e.forward(np.array([[1]], dtype=np.int32))
        e.kv_len[:] = 0; e.pos = 0
        if e._ssm_state is not None: e._ssm_state.fill(0)
        
        # Benchmark
        tok = np.array([[1 + tid]], dtype=np.int32)
        t_list = []
        for step in range(GEN_TOKENS):
            t0 = time.perf_counter()
            logits = e.forward(tok)
            t_list.append(time.perf_counter() - t0)
            tok = np.array([[int(np.argmax(logits))]], dtype=np.int32)
        t_list = t_list[5:]
        avg_ms = float(np.mean(t_list) * 1000)
        tps = float(1000 / avg_ms)
        results[tid] = (avg_ms, tps)
    except Exception as ex:
        import traceback; traceback.print_exc()
        results[tid] = ('err', str(ex)[:100])

threads = []
t_start = time.perf_counter()
for i in range(N_WORKERS):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

wall = time.perf_counter() - t_start

print(f"\n{'='*60}")
print(f"  Qwen3.6 MXFP4 — Threaded Server (children share weight refs)")
print(f"{'='*60}")
tps_list = []
for tid in range(N_WORKERS):
    r = results[tid]
    if r and r[0] != 'err':
        ms, tps = r
        print(f"  Worker {tid:2d}: {ms:5.1f}ms → {tps:5.1f} tok/s")
        tps_list.append(tps)
    elif r and r[0] == 'err':
        print(f"  Worker {tid:2d}: ERROR {r[1]}")

if tps_list:
    avg_tps = float(np.mean(tps_list))
    agg = avg_tps * N_WORKERS
    print(f"{'─'*60}")
    print(f"  Per-user:  {avg_tps:.1f} tok/s avg")
    print(f"  Aggregate: {agg:.0f} tok/s")
    print(f"  Wall: {wall:.1f}s")
    print(f"  200% target (51 tok/s):  {agg/51:.1f}x {'PASS' if agg>51 else 'FAIL'}")
    print(f"  750% target (191 tok/s): {agg/191:.1f}x {'PASS' if agg>191 else 'FAIL'}")
