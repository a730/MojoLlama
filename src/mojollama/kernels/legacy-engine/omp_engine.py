#!/usr/bin/env python3
"""MojoLlama OpenMP Engine — C kernel with internal OpenMP parallelism.
Single Python thread orchestrates, C does all heavy lifting with OpenMP.
No Python threading overhead.
Target: >77 tok/s (within 5% of llama.cpp).
"""
import os, sys, time, ctypes, numpy as np

SO = os.path.join(os.path.dirname(__file__), "q4_kernel_omp.so")
_lib = ctypes.CDLL(SO)
_lib.q4_matmul_omp.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.q4_matmul_omp.restype = None
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
            self._cache[name] = (np.frombuffer(raw, dtype=np.uint8), t.tensor_type)
        elif t.tensor_type in (QT.F32, QT.F16):
            self._cache[name] = (np.ascontiguousarray(arr.astype(np.float32)), t.tensor_type)
        else:
            deq = _g.dequantize(arr, t.tensor_type)
            self._cache[name] = (np.ascontiguousarray(deq.astype(np.float32)), t.tensor_type)
        return self._cache[name]

def mm(wi, x, nc):
    """Single-call matmul — OpenMP internally parallelizes."""
    w, tt = wi
    nr = len(w) // ((nc // 32) * Q4_TS.get(tt, 18))
    out = np.zeros(nr, dtype=np.float32)
    _lib.q4_matmul_omp(
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nr, nc, Q4_TS.get(tt, 18))
    return out

def rms(x, w):
    return x / np.sqrt(np.mean(x**2) + 1e-5) * w

def engine(path):
    mdl = Model(path); n = mdl.n
    print(f"OpenMP Engine (32 threads internal)")
    print(f"Model: {n['n_layers']}L/{n['n_embd']}D/{n['n_ff']}FF/{n['n_head']}H")
    print()

    kvc = [{"k": np.zeros((0, n["n_kv"]), dtype=np.float32),
            "v": np.zeros((0, n["n_kv"]), dtype=np.float32)}
           for _ in range(n["n_layers"])]

    token_id = 128000
    h = mdl.get("token_embd.weight")[0][token_id].astype(np.float32)
    hd, nh, nkh = n["head_dim"], n["n_head"], n["n_kv_head"]

    t0 = time.time()
    for l in range(n["n_layers"]):
        lt = time.time()
        r = h.copy()
        h = rms(h, mdl.get(f"blk.{l}.attn_norm.weight")[0])

        q = mm(mdl.get(f"blk.{l}.attn_q.weight"), h, n["n_embd"])
        k = mm(mdl.get(f"blk.{l}.attn_k.weight"), h, n["n_embd"])
        v = mm(mdl.get(f"blk.{l}.attn_v.weight"), h, n["n_embd"])

        # RoPE
        freqs = 1.0 / (500000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        cos = np.cos(0 * freqs); sin = np.sin(0 * freqs)
        q2 = q.reshape(nh, hd); k2 = k.reshape(nkh, hd)
        for xx, nx in [(q2, nh), (k2, nkh)]:
            xr = xx.reshape(nx, hd//2, 2)
            xr_rot = np.stack([-xr[...,1], xr[...,0]], axis=-1)
            xx[:] = (xr*cos.reshape(1,hd//2,1) + xr_rot*sin.reshape(1,hd//2,1)).reshape(nx, hd)

        kvc[l]["k"] = k.reshape(1, -1)
        kvc[l]["v"] = v.reshape(1, -1)

        # Attention (Python loop — slow but <1ms for pos=0)
        ng = nh // nkh; att = np.zeros(nh*hd, dtype=np.float32)
        for hh in range(nh):
            kvh = hh // ng
            ks = kvc[l]["k"][0:1, kvh*hd:(kvh+1)*hd]
            vs = kvc[l]["v"][0:1, kvh*hd:(kvh+1)*hd]
            sc = q2[hh] @ ks.T - np.max(q2[hh] @ ks.T)
            att[hh*hd:(hh+1)*hd] = (np.exp(sc)/np.sum(np.exp(sc))) @ vs

        h = r + mm(mdl.get(f"blk.{l}.attn_output.weight"), att, n["n_embd"])

        r = h.copy()
        h = rms(h, mdl.get(f"blk.{l}.ffn_norm.weight")[0])
        gate = mm(mdl.get(f"blk.{l}.ffn_gate.weight"), h, n["n_embd"])
        gate = gate / (1 + np.exp(-gate))
        up = mm(mdl.get(f"blk.{l}.ffn_up.weight"), h, n["n_embd"])
        h = r + mm(mdl.get(f"blk.{l}.ffn_down.weight"), gate*up, n["n_ff"])

        print(f"  L{l}: {(time.time()-lt)*1000:.1f}ms", flush=True)

    h = rms(h, mdl.get("output_norm.weight")[0])
    wo = mdl.get("output.weight") or mdl.get("token_embd.weight")
    logits = h @ wo[0].T if wo[0].ndim > 1 else h @ wo[0]
    t = time.time() - t0

    print()
    print(f"First token: {t*1000:.0f}ms ({1/t:.1f} tok/s)")
    print(f"Next: {np.argmax(logits)}")
    print()
    print(f"llama.cpp: 81 tok/s")
    print(f"Gap: {(81 - 1/t)/81*100:.1f}%")

if __name__ == "__main__":
    engine(sys.argv[1] if len(sys.argv) > 1 else "/onedev-workspace/work/Llama-3.2-1B-Instruct-Q4_0.gguf")
