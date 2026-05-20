#!/usr/bin/env python3
"""MojoLlama Threaded Engine — C AVX2 via threading releases GIL.
Threads share memory (no serialization). Each thread calls C .so
which releases the Python GIL during AVX2 compute.
Target: 81 tok/s.
"""
import os, sys, time, ctypes, concurrent.futures
import numpy as np

SO = os.path.join(os.path.dirname(__file__), "q4_kernel_avx2.so")
_lib = ctypes.CDLL(SO)
for fn in ["q4_0_matmul", "q4_1_matmul"]:
    f = getattr(_lib, fn)
    f.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int]
    f.restype = None
Q4_TS = {2: 18, 3: 20}

class Model:
    def __init__(self, path):
        import gguf
        self.r = gguf.GGUFReader(path)
        self.t = {x.name: x for x in self.r.tensors}
        f = self.r.get_field
        arch = bytes(f("general.architecture").parts[-1]).decode()
        pfx = arch + "."
        def gi(n): return int(f(pfx + n).parts[-1].item())
        self.n = {"n_layers": gi("block_count"), "n_embd": gi("embedding_length"),
                   "n_head": gi("attention.head_count"),
                   "n_kv_head": gi("attention.head_count_kv"),
                   "n_ff": gi("feed_forward_length")}
        self.n["head_dim"] = self.n["n_embd"] // self.n["n_head"]
        self.n["n_kv"] = self.n["n_kv_head"] * self.n["head_dim"]
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

def q4_chunk(w, tt, x, start, end, nr, nc):
    out = np.zeros(end - start, dtype=np.float32)
    fn = "q4_1_matmul" if tt == 3 else "q4_0_matmul"
    getattr(_lib, fn)(
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nr, nc, start, end)
    return out

def threaded_matmul(wi, x, pool, nw, nc):
    w, tt = wi; nr = len(w) // ((nc // 32) * Q4_TS.get(tt, 18))
    cs = max(1, nr // nw)
    fut_map = {}; s = 0
    while s < nr:
        e = min(s + cs, nr)
        fut_map[(s, e)] = pool.submit(q4_chunk, w, tt, x, s, e, nr, nc)
        s = e
    out = np.zeros(nr, dtype=np.float32)
    for (s, e), f in fut_map.items():
        out[s:e] = f.result()
    return out

class Engine:
    def __init__(self, path, nw=None):
        self.mdl = Model(path); self.n = self.mdl.n
        self.nw = nw or (os.cpu_count() or 32) // 2
        self._pool = concurrent.futures.ThreadPoolExecutor(self.nw)

    def forward(self, token_id, kvc, pos):
        n = self.n
        h = self.mdl.get("token_embd.weight")[0][token_id].astype(np.float32)
        for l in range(n["n_layers"]):
            t0 = time.time()
            r = h.copy()
            an = self.mdl.get(f"blk.{l}.attn_norm.weight")[0]
            h = h / np.sqrt(np.mean(h**2) + 1e-5) * an

            q = threaded_matmul(self.mdl.get(f"blk.{l}.attn_q.weight"), h, self._pool, self.nw, n["n_embd"])
            k = threaded_matmul(self.mdl.get(f"blk.{l}.attn_k.weight"), h, self._pool, self.nw, n["n_embd"])
            v = threaded_matmul(self.mdl.get(f"blk.{l}.attn_v.weight"), h, self._pool, self.nw, n["n_embd"])

            hd, nh, nkh = n["head_dim"], n["n_head"], n["n_kv_head"]
            freqs = 1.0 / (500000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
            cos = np.cos(pos * freqs); sin = np.sin(pos * freqs)
            for xx, nx in [(q.reshape(nh, hd), nh), (k.reshape(nkh, hd), nkh)]:
                xr = xx.reshape(nx, hd//2, 2)
                xr_rot = np.stack([-xr[..., 1], xr[..., 0]], axis=-1)
                xx[:] = (xr*cos.reshape(1, hd//2, 1) + xr_rot*sin.reshape(1, hd//2, 1)).reshape(nx, hd)

            if kvc[l]["k"].shape[0] <= pos:
                kvc[l]["k"] = np.vstack([kvc[l]["k"], k.reshape(1, -1)])
                kvc[l]["v"] = np.vstack([kvc[l]["v"], v.reshape(1, -1)])
            else:
                kvc[l]["k"][pos] = k; kvc[l]["v"][pos] = v

            ng = nh // nkh; att = np.zeros(nh*hd, dtype=np.float32); q2 = q.reshape(nh, hd)
            for hh in range(nh):
                kvh = hh // ng; ks = kvc[l]["k"][:pos+1, kvh*hd:(kvh+1)*hd]
                vs = kvc[l]["v"][:pos+1, kvh*hd:(kvh+1)*hd]
                sc = q2[hh] @ ks.T - np.max(q2[hh] @ ks.T)
                att[hh*hd:(hh+1)*hd] = (np.exp(sc)/np.sum(np.exp(sc))) @ vs

            h = r + threaded_matmul(self.mdl.get(f"blk.{l}.attn_output.weight"), att, self._pool, self.nw, n["n_embd"])

            r = h.copy()
            fn_ = self.mdl.get(f"blk.{l}.ffn_norm.weight")[0]
            h = h / np.sqrt(np.mean(h**2) + 1e-5) * fn_
            gate = threaded_matmul(self.mdl.get(f"blk.{l}.ffn_gate.weight"), h, self._pool, self.nw, n["n_embd"])
            gate = gate / (1 + np.exp(-gate))
            up = threaded_matmul(self.mdl.get(f"blk.{l}.ffn_up.weight"), h, self._pool, self.nw, n["n_embd"])
            h = r + threaded_matmul(self.mdl.get(f"blk.{l}.ffn_down.weight"), gate*up, self._pool, self.nw, n["n_ff"])

            if pos == 0:
                t1 = time.time()
                print(f"  L{l}: {(t1-t0)*1000:.0f}ms", flush=True)

        on = self.mdl.get("output_norm.weight")[0]
        h = h / np.sqrt(np.mean(h**2) + 1e-5) * on
        wo = self.mdl.get("output.weight") or self.mdl.get("token_embd.weight")
        return h @ wo[0].T if wo[0].ndim > 1 else h @ wo[0]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/onedev-workspace/work/Llama-3.2-1B-Instruct-Q4_0.gguf"
    e = Engine(path)
    n = e.n; print(f"Threaded Engine ({e.nw} workers)")
    print(f"Model: {n['n_layers']}L/{n['n_embd']}D/{n['n_ff']}FF/{n['n_head']}H")
    kvc = [{"k": np.zeros((0, n["n_kv"]), dtype=np.float32),"v": np.zeros((0, n["n_kv"]), dtype=np.float32)} for _ in range(n["n_layers"])]
    t0 = time.time(); logits = e.forward(128000, kvc, 0)
    t = time.time()-t0
    print(f"First: {t*1000:.0f}ms ({1/t:.1f} tok/s), next={np.argmax(logits)}")
