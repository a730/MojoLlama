#!/usr/bin/env python3
"""
Qwen3.6 MXFP4 concurrent server via os.fork() with fork-safe allocator.
Key insight: after fork, set PYTHONMALLOC=malloc and call malloc_trim(0).
"""
import sys, os, time, ctypes, struct, numpy as np

# Must be set BEFORE any Python memory allocations
os.environ['PYTHONMALLOC'] = 'malloc'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
MODEL = "/tmp/models/qwen3.6-mxfp4.gguf"
N_CHILDREN = 10
GEN_TOKENS = 50
OMP_PER_CHILD = 4

print(f"Loading Qwen3.6 MXFP4 (PYTHONMALLOC=malloc)...", flush=True)
from turbo_engine_v7_moe import TurboEngineV7MoE
t0 = time.perf_counter()
e = TurboEngineV7MoE(MODEL, 32)
V = e.vocab_size
print(f"  Load: {time.perf_counter()-t0:.1f}s | Forking {N_CHILDREN} children...", flush=True)

# Import libc for malloc_trim
libc = ctypes.CDLL('libc.so.6')
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int

children = []
for i in range(N_CHILDREN):
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        # Child: fix memory allocator after fork
        libc.malloc_trim(0)
        os.environ['OMP_NUM_THREADS'] = str(OMP_PER_CHILD)
        
        # Minimal reset: only metadata (KV data inherited zeros)
        self = e
        self.kv_len.fill(0)
        self.pos = 0
        self._x.fill(0)
        self._residual.fill(0)
        self._x_norm.fill(0)
        self._ssm_state.fill(0)
        
        # Warmup (1 step — creates OMP thread pool for this child)
        self.forward(np.array([[1]], dtype=np.int32))
        self.kv_len.fill(0); self.pos = 0
        
        # Benchmark
        tok = np.array([[1 + i]], dtype=np.int32)
        t_list = []
        for step in range(GEN_TOKENS):
            t0 = time.perf_counter()
            logits = self.forward(tok)
            elapsed = time.perf_counter() - t0
            t_list.append(elapsed)
            tok = np.array([[int(np.argmax(logits))]], dtype=np.int32)
        
        t_list = t_list[5:]
        avg_ms = float(np.mean(t_list) * 1000)
        tps = float(1000 / avg_ms)
        os.write(w_fd, struct.pack('dd', avg_ms, tps))
        os.close(w_fd)
        sys.exit(0)
    else:
        os.close(w_fd)
        children.append((pid, r_fd, i))

# Collect
results = {}
for pid, r_fd, wid in children:
    data = b''
    while True:
        chunk = os.read(r_fd, 16)
        if not chunk: break
        data += chunk
    os.close(r_fd)
    if len(data) >= 16:
        ms, tps = struct.unpack('dd', data[:16])
        results[wid] = (ms, tps)

for pid, _, _ in children:
    os.waitpid(pid, 0)

# Report
print(f"\n{'='*60}")
print(f"  Qwen3.6 MXFP4 — Fork ({N_CHILDREN} workers, {OMP_PER_CHILD} thr/worker)")
print(f"{'='*60}")
tps_list = []
for wid in sorted(results):
    ms, tps = results[wid]
    print(f"  Worker {wid:2d}: {ms:5.1f}ms → {tps:5.1f} tok/s")
    tps_list.append(tps)

if tps_list:
    avg_tps = float(np.mean(tps_list))
    agg = avg_tps * N_CHILDREN
    print(f"{'─'*60}")
    print(f"  Per-user:  {avg_tps:.1f} tok/s avg")
    print(f"  Aggregate: {agg:.0f} tok/s")
    print(f"  200% target (51 tok/s):  {'✅' if agg > 51 else '❌'} {agg/51:.1f}x")
    print(f"  750% target (191 tok/s): {'✅' if agg > 191 else '❌'} {agg/191:.1f}x")
