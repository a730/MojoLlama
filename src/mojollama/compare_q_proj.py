#!/usr/bin/env python3
"""Compare C engine hidden state vs numpy reference for layer 0.
This isolates whether the issue is in the C matmuls or elsewhere."""
import sys, os, time, ctypes, numpy as np, gguf
os.environ['OMP_NUM_THREADS'] = '32'
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
from turbo_engine_v7_moe import TurboEngineV7MoE, GGML_Q4_K, GGML_Q6_K

lib = ctypes.CDLL('/onedev-workspace/work/src/mojollama/kernels/cengine_batch.so')
cv,ci = ctypes.c_void_p, ctypes.c_int

e = TurboEngineV7MoE('/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf', 32)
N,NH,NKH,HD,FF,V = e.n_embd,e.n_head,e.n_kv_head,e.head_dim,e.n_ff,e.vocab_size
S = max(N, NH*HD, FF, NKH*HD, e.n_ff_expert)

lw = e._layers[0]
me = e._moe_layers[0]

def q4_k_dequant_row(raw_ptr, nr, nc):
    """Dequantize Q4_K weight matrix to float32 using numpy"""
    bpr = nc // 256; bs = 144
    f32 = np.zeros((nr, nc), dtype=np.float32)
    for r in range(nr):
        for bi in range(bpr):
            blk = raw_ptr[r * bpr + bi]
            d = np.frombuffer(blk[0:2], dtype=np.float16)[0]
            dmin = np.frombuffer(blk[2:4], dtype=np.float16)[0]
            scales12 = np.frombuffer(blk[4:16], dtype=np.uint8)
            qs = np.frombuffer(blk[16:144], dtype=np.uint8)
            off = bi * 256
            for g in range(8):
                if g < 4:
                    sc = scales12[g] & 0x3F
                    mn = scales12[g+4] & 0x3F
                else:
                    sc = (scales12[g+4] & 0xF) | ((scales12[g-4] >> 6) << 4)
                    mn = (scales12[g+4] >> 4) | ((scales12[g] >> 6) << 4)
                for j in range(32):
                    idx = g * 32 + j
                    s = (qs[idx//2] >> ((idx & 1) * 4)) & 0xF
                    f32[r, off + idx] = d * sc * s - dmin * mn
    return f32

def q6_k_dequant_row(raw_ptr, nr, nc):
    """Dequantize Q6_K weight matrix to float32"""
    bpr = nc // 256; bs = 210
    f32 = np.zeros((nr, nc), dtype=np.float32)
    for r in range(nr):
        for bi in range(bpr):
            blk = raw_ptr[r * bpr + bi]
            d = np.frombuffer(blk[0:2], dtype=np.float16)[0]
            ql = np.frombuffer(blk[2:130], dtype=np.uint8)
            qh = np.frombuffer(blk[130:194], dtype=np.uint8)
            sc = np.frombuffer(blk[194:210], dtype=np.int8)
            off = bi * 256
            for j in range(256):
                s = (ql[j//2] >> ((j & 1) * 4)) & 0xF
                sh = (qh[j//4] >> ((j & 3) * 2)) & 3
                s = s | (sh << 4)
                s -= 32
                f32[r, off + j] = d * sc[j // 16] * s
    return f32

# Get raw weight data as numpy arrays
def get_raw(t):
    return np.frombuffer(t.reshape(-1), dtype=np.uint8)

# Token embedding
x = e.emb[785].copy()  # float32, shape (N,)
print(f"Input embedding nan={np.isnan(x).sum()}/{N}", flush=True)

# ── Numpy reference: layer 0 ──
# RMS norm
ss = np.sum(x * x)
ir = 1.0 / np.sqrt(ss / N + e.eps)
xn = x * ir * lw.attn_norm_w
print(f"RMS norm output: nan={np.isnan(xn).sum()}/{N}", flush=True)

# Q projection (numpy)
Wq_f32 = gguf.dequantize(
    e.reader.get_tensor('blk.0.attn_q.weight').data, 
    e.reader.get_tensor('blk.0.attn_q.weight').tensor_type
).astype(np.float32)
Wq = Wq_f32.reshape(int(Wq_f32.shape[1]), int(Wq_f32.shape[0]))
q_ref = xn @ Wq.T  # or Wq @ xn
print(f"Q proj ref: nan={np.isnan(q_ref).sum()}/{len(q_ref)} max={q_ref.max():.2f}", flush=True)

# ── C engine: Q projection ──
q_c = np.zeros(lw.attn_q_nr.value, dtype=np.float32)
lib.q4_k_batch_matmul(lw.attn_q_raw, x.ctypes.data_as(cv), q_c.ctypes.data_as(cv),
    ci(lw.attn_q_nr.value), ci(lw.attn_q_nc.value), ci(1))
print(f"Q proj C: nan={np.isnan(q_c).sum()}/{len(q_c)} max={q_c.max():.2f}", flush=True)
# Compare
diff = np.abs(q_ref.reshape(-1)[:100] - q_c[:100])
print(f"Q diff (first 100): mean={diff.mean():.4f} max={diff.max():.4f}", flush=True)
match = np.corrcoef(q_ref.reshape(-1)[:min(1000, len(q_c))], q_c[:min(1000, len(q_c))])[0,1]
print(f"Q correlation: {match:.4f}", flush=True)
