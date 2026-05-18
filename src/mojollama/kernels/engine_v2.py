#!/usr/bin/env python3
"""MojoLlama Hybrid Engine v2 — C AVX2 kernel + batch-row parallelism.
Architecture: 1 pool call per layer (all 7 matmuls), 32 workers.
Target: >77 tok/s (within 5% of llama.cpp 81 tok/s).
"""
import os, sys, time, json, math, ctypes, multiprocessing as mp
import numpy as np
from functools import partial

SO = os.path.join(os.path.dirname(__file__), 'q4_kernel_avx2.so')
_lib = ctypes.CDLL(SO)
for fn in ['q4_0_matmul', 'q4_1_matmul']:
    f = getattr(_lib, fn)
    f.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int]
    f.restype = None

Q4_0_TS = 18  # type size for Q4_0
Q4_1_TS = 20

class Model:
    def __init__(self, path):
        import gguf
        self.r = gguf.GGUFReader(path)
        self.t = {x.name: x for x in self.r.tensors}
        f = self.r.get_field
        arch = bytes(f('general.architecture').parts[-1]).decode()
        pfx = arch + '.'
        def gi(n): return int(f(pfx + n).parts[-1].item())
        self.n = {
            'n_layers': gi('block_count'), 'n_embd': gi('embedding_length'),
            'n_head': gi('attention.head_count'), 'n_kv_head': gi('attention.head_count_kv'),
            'n_ff': gi('feed_forward_length'),
        }
        self.n['head_dim'] = self.n['n_embd'] // self.n['n_head']
        self.n['n_kv'] = self.n['n_kv_head'] * self.n['head_dim']
        self._cache = {}
    
    def get(self, name):
        if name in self._cache: return self._cache[name]
        t = self.t.get(name)
        if t is None: return None
        import gguf as _g
        from gguf.constants import GGMLQuantizationType as QT
        arr = np.asarray(t.data)
        if t.tensor_type in (QT.Q4_0, QT.Q4_1):
            raw = arr.tobytes() if arr.dtype == np.uint8 else arr.astype(np.uint8).tobytes()
            result = (np.frombuffer(raw, dtype=np.uint8), t.tensor_type)
        elif t.tensor_type in (QT.F32, QT.F16):
            result = (np.ascontiguousarray(arr.astype(np.float32)), t.tensor_type)
        else:
            deq = _g.dequantize(arr, t.tensor_type)
            result = (np.ascontiguousarray(deq.astype(np.float32)), t.tensor_type)
        self._cache[name] = result
        return result

def worker_fn(chunks, model_path):
    """Worker: compute all matmuls for assigned row chunks across all layers."""
    import gguf
    mdl = Model(model_path)
    results = []
    for (weight_name, x_bytes, x_shape, start, end) in chunks:
        x = np.frombuffer(x_bytes, dtype=np.float32).reshape(x_shape)
        w_info = mdl.get(weight_name)
        if w_info is None:
            results.append(np.zeros(end - start, dtype=np.float32))
            continue
        w, tt = w_info
        nc = mdl.n['n_embd']
        if 'down' in weight_name: nc = mdl.n['n_ff']
        from gguf.constants import GGMLQuantizationType as QT
        if tt == QT.Q4_1:
            # Q4_1 already dequantized to f32 in old code... but now we store raw
            # Actually we store raw now. Let's dequantize
            import gguf as _g
            w_f32 = _g.dequantize(w, tt)
            nr = len(w_f32) // nc
            w_f32 = w_f32.reshape(nr, nc)
            out = x @ w_f32.T
        else:
            fn = 'q4_1_matmul' if tt == QT.Q4_1 else 'q4_0_matmul'
            n_rows = len(w) // ((nc // 32) * (20 if tt == QT.Q4_1 else 18))
            out = np.zeros(end - start, dtype=np.float32)
            getattr(_lib, fn)(
                w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_rows, nc, start, end
            )
        results.append(out)
    return results

class Engine:
    def __init__(self, path, nw=32):
        self.mdl = Model(path)
        self.n = self.mdl.n
        self.nw = min(nw, mp.cpu_count())
        self._pool = None
    
    def get_pool(self):
        if self._pool is None:
            self._pool = mp.Pool(self.nw)
        return self._pool
    
    def forward(self, token_id, kvc, pos):
        n = self.n
        h = self.mdl.get('token_embd.weight')[0][token_id].astype(np.float32)
        
        for layer in range(n['n_layers']):
            t0 = time.time()
            r = h.copy()
            h = h / np.sqrt(np.mean(h**2, keepdims=True) + 1e-5) * self.mdl.get(f'blk.{layer}.attn_norm.weight')[0]
            
            # Build chunk list for this layer
            wnames = [f'blk.{layer}.attn_q.weight',   f'blk.{layer}.attn_k.weight',
                      f'blk.{layer}.attn_v.weight',   f'blk.{layer}.attn_output.weight',
                      f'blk.{layer}.ffn_gate.weight', f'blk.{layer}.ffn_up.weight',
                      f'blk.{layer}.ffn_down.weight']
            
            # Split rows
            chunk = []
            for wn in wnames:
                wi = self.mdl.get(wn)
                if wi is None: continue
                w, tt = wi
                nc = n['n_ff'] if 'down' in wn else n['n_embd']
                ts = Q4_1_TS if tt == 3 else Q4_0_TS
                nr = len(w) // ((nc // 32) * ts)
                cs = max(1, nr // self.nw)
                s = 0
                while s < nr:
                    e = min(s + cs, nr)
                    chunk.append((wn, h.tobytes(), h.shape, s, e))
                    s = e
            
            pool = self.get_pool()
            res_all = pool.apply(worker_fn, (chunk,))  # single worker as test
            
            # This is wrong - need proper fan-out. Let's simplify to single-core first
            import traceback; traceback.print_exc()
            return h

e = Engine('/onedev-workspace/work/Llama-3.2-1B-Instruct-Q4_0.gguf')
print("Engine loaded")
