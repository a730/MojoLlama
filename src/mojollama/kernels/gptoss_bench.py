#!/usr/bin/env python3
"""Benchmark gptoss_forward_c — properly cast all arrays to void_p.
Run: OMP_NUM_THREADS=32 python3 gptoss_bench.py
"""
import os, sys, time, numpy as np, ctypes
os.environ['OMP_PROC_BIND'] = 'close'
os.environ['OMP_PLACES'] = 'cores'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from turbo_engine_v7_moe import TurboEngineV7MoE

MODEL = '/tmp/models/gpt-oss-20b-Q4_K_M.gguf'
eng = TurboEngineV7MoE(MODEL, n_threads=32)
V = int(eng.vocab_size)
L = eng.n_layers; N = eng.n_embd; NH = eng.n_head
KV = eng.n_kv_head; HD = eng.head_dim; FF = eng.n_ff
E = eng.eps; RD = eng.rope_dim; MC = 4096

so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_engine.so')
so = ctypes.CDLL(so_path)
cf = ctypes.POINTER(ctypes.c_float)
cu = ctypes.POINTER(ctypes.c_uint8)
ci = ctypes.c_int
vp = ctypes.c_void_p

# === Build argtypes: 61 params ===
so.gptoss_forward_c.argtypes = (
    [vp, ci] +                     # 1-2: emb, token
    [ci]*7 +                       # 3-9: L, N, NH, KV, HD, FF, V
    [ci]*3 + [ctypes.c_float] + [ci]*2 +  # 10-15: n_exp, top_k, n_ff, eps, RD, MC
    [vp]*2 +                       # 16-17: attn_norm, ffn_norm
    [vp]*4 + [vp]*4 + [vp]*4 + [vp]*4 +  # 18-33: Q/K/V/O weights + dims
    [vp]*4 + [ci]*3 +              # 34-40: gi, ge, ue, de, gqt, uqt, dqt
    [vp]*4 + [ci]*4 +              # 41-48: shexp arrays + shexp_quants
    [vp]*2 +                       # 49-50: cos_table, sin_table
    [vp]*3 +                       # 51-53: kv_k, kv_v, kv_lens
    [vp]*2 + [ci]*3 +              # 54-58: out_norm, w_out, out_nr, out_nc, out_qt
    [vp]*3                         # 59-61: logits, ws, q8_ws
)
assert len(so.gptoss_forward_c.argtypes) == 61, f"argtypes count: {len(so.gptoss_forward_c.argtypes)}"

# === Build pointer arrays (cast ALL to void_p) ===
def arr_t(t, items):
    a = (t * len(items))()
    for i, item in enumerate(items):
        a[i] = item
    return ctypes.cast(a, vp)

def arr_i(items):
    a = (ci * len(items))()
    for i, item in enumerate(items):
        a[i] = item.value if hasattr(item, 'value') else int(item)
    return ctypes.cast(a, vp)

def ptr(x):
    return ctypes.cast(x, vp)

Lw = eng._layers
Me = eng._moe_layers

# Attention weight arrays
an = arr_t(cf, [l.attn_norm_w.ctypes.data_as(cf) for l in Lw])
fn = arr_t(cf, [l.ffn_norm_w.ctypes.data_as(cf) for l in Lw])
wQ = arr_t(cu, [l.attn_q_raw for l in Lw])
qN = arr_i([l.attn_q_nr.value for l in Lw])
qC = arr_i([l.attn_q_nc.value for l in Lw])
qQ = arr_i([l.attn_q_qt for l in Lw])
wK = arr_t(cu, [l.attn_k_raw for l in Lw])
kN = arr_i([l.attn_k_nr.value for l in Lw])
kC = arr_i([l.attn_k_nc.value for l in Lw])
kQ = arr_i([l.attn_k_qt for l in Lw])
wV = arr_t(cu, [l.attn_v_raw for l in Lw])
vN = arr_i([l.attn_v_nr.value for l in Lw])
vC = arr_i([l.attn_v_nc.value for l in Lw])
vQ = arr_i([l.attn_v_qt for l in Lw])
wO = arr_t(cu, [l.attn_out_raw for l in Lw])
oN = arr_i([l.attn_out_nr.value for l in Lw])
oC = arr_i([l.attn_out_nc.value for l in Lw])
oQ = arr_i([l.attn_out_qt for l in Lw])

# MoE arrays
gi = arr_t(cf, [m.router_f32.ctypes.data_as(cf) for m in Me])

# FLAT raw expert weights (prevent GC with local vars)
_gate_raws = [eng.raw_weights[f'blk.{l}.ffn_gate_exps.weight'] for l in range(L)]
_up_raws = [eng.raw_weights[f'blk.{l}.ffn_up_exps.weight'] for l in range(L)]
_down_raws = [eng.raw_weights[f'blk.{l}.ffn_down_exps.weight'] for l in range(L)]
ge = arr_t(cu, [r.ctypes.data_as(cu) for r in _gate_raws])
ue = arr_t(cu, [r.ctypes.data_as(cu) for r in _up_raws])
de = arr_t(cu, [r.ctypes.data_as(cu) for r in _down_raws])

m0 = Me[0]; gqt = m0.gate_qt.value if hasattr(m0.gate_qt, 'value') else int(bytes(m0.gate_qt).strip(b"'\\x00")) 
uqt = m0.up_qt.value if hasattr(m0.up_qt, 'value') else int(bytes(m0.up_qt).strip(b"'\\x00"))
dqt = m0.down_qt.value if hasattr(m0.down_qt, 'value') else int(bytes(m0.down_qt).strip(b"'\\x00"))

# Shared expert arrays (NULL)
nu_f = ctypes.cast((cf * L)(), vp)
nu_u = ctypes.cast((cu * L)(), vp)

# KV cache, output, workspace
kv_k = ptr(eng.kv_k.ctypes.data_as(cf))
kv_v = ptr(eng.kv_v.ctypes.data_as(cf))
onw = ptr(eng._out_norm_w.ctypes.data_as(cf))
wo = ptr(eng._out_raw)
onv = eng._out_nr.value; ocv = eng._out_nc.value; oqv = eng._out_qt

S = max(N, NH*HD, FF, KV*HD, 8192)
ws = np.zeros(14*S + MC*KV*HD*2, dtype=np.float32)
q8 = np.zeros((N+31)//32*34, dtype=np.uint8)
lg = np.zeros(V, dtype=np.float32)
h2 = RD//2
ct = np.zeros((MC, h2), dtype=np.float32)
st = np.zeros((MC, h2), dtype=np.float32)
for p in range(MC):
    a = p / (10000.0 ** (np.arange(0, RD, 2, dtype=np.float32) / RD))
    ct[p] = np.cos(a); st[p] = np.sin(a)
emb_p = ptr(eng.emb.ctypes.data_as(cf))

# === Helper to call C forward ===
def c_forward(token):
    kl = (ci * L)(*[0]*L)
    so.gptoss_forward_c(
        emb_p, ci(token),
        ci(L), ci(N), ci(NH), ci(KV), ci(HD), ci(FF), ci(V),
        ci(32), ci(4), ci(2880), ctypes.c_float(E), ci(RD), ci(MC),
        an, fn, wQ, qN, qC, qQ, wK, kN, kC, kQ,
        wV, vN, vC, vQ, wO, oN, oC, oQ,
        gi, ge, ue, de, ci(gqt), ci(uqt), ci(dqt),
        nu_f, nu_u, nu_u, nu_u, ci(0), ci(0), ci(0), ci(0),
        ptr(ct.ctypes.data_as(cf)), ptr(st.ctypes.data_as(cf)),
        kv_k, kv_v, ptr(kl),
        onw, wo, ci(onv), ci(ocv), ci(oqv),
        ptr(lg.ctypes.data_as(cf)), ptr(ws.ctypes.data_as(cf)), ptr(q8.ctypes.data_as(cu)))
    return lg.copy()

# === Warmup ===
for i in range(5):
    l = c_forward(i % V)
    print(f"Warmup {i}: max_logit={np.max(l):.2f}", flush=True)

# === Benchmark C forward ===
tc = []
for i in range(30):
    t0 = time.perf_counter()
    c_forward(i % V)
    tc.append((time.perf_counter() - t0) * 1000)

# === Python baseline ===
tp = []
for i in range(30):
    eng.reset()
    t0 = time.perf_counter()
    eng.forward([i % V])
    tp.append((time.perf_counter() - t0) * 1000)

tc_a = np.array(tc); tp_a = np.array(tp)

print(f"\n{'='*55}")
print(f"  GPT-OSS: C forward({np.mean(tc_a):.1f}ms) vs Python({np.mean(tp_a):.1f}ms)")
print(f"{'='*55}")
print(f"  {'':>20} {'C forward':>14} {'Python':>14}")
print(f"  {'─'*50}")
print(f"  {'Median ms':>20} {np.median(tc_a):>10.2f} {np.median(tp_a):>10.2f}")
print(f"  {'Mean ms':>20} {np.mean(tc_a):>10.2f} {np.mean(tp_a):>10.2f}")
print(f"  {'Tok/s':>20} {1000/np.mean(tc_a):>10.1f} {1000/np.mean(tp_a):>10.1f}")
print(f"  {'Speedup':>20} {'':>4} {np.mean(tp_a)/np.mean(tc_a):>5.2f}x")
print(f"  {'llama.cpp':>20} {'':>10} 27.69 tok/s")
print(f"  {'vs llama.cpp (C)':>20} {'':>4} {1000/np.mean(tc_a)/27.69*100:.0f}%")
print(f"{'='*55}")

# Compare with benchmark-quality numbers (with prompt)
print(f"\n{'='*55}")
print(f"  CONCURRENCY: Muli-seq through shared engine")
print(f"{'='*55}")
import concurrent.futures
for conc in [1, 2, 4, 8]:
    N_SEQ = 8; N_GEN = 20; TOTAL = N_SEQ * N_GEN
    def run_seq(tid):
        t = abs(hash(str(tid))) % V
        eng.reset()
        for _ in range(N_GEN):
            l = eng.forward([t]); t = int(np.argmax(np.asarray(l).ravel())) % V
        return True
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
        list(pool.map(run_seq, range(N_SEQ)))
    wall = time.perf_counter() - t0
    agg = TOTAL / wall
    ratio = agg / 27.69
    beat = "🏆 BEATS!" if ratio > 1.0 else ""
    print(f"  conc={conc:2d} | {TOTAL:3d} tok | {wall:.2f}s | {agg:>5.1f} agg | {ratio:.2f}x vs llama.cpp {beat}")
