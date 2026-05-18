#!/usr/bin/env python3
"""MojoLlama Parallel Engine v2 — 1 pool call per layer, 32 workers.
Each worker does ALL 7 matmuls for its row chunk in one batch.
Target: 81 tok/s (matches llama.cpp on Threadripper 3970X).
"""

import os, sys, time, math, json, ctypes, struct, multiprocessing as mp
import numpy as np

SO = os.path.join(os.path.dirname(__file__), "q4_kernel_avx2.so")
Q4_TS = {2: 18, 3: 20}


class Model:
    def __init__(self, path):
        import gguf
        self.r = gguf.GGUFReader(path)
        self.t = {x.name: x for x in self.r.tensors}
        f = self.r.get_field
        arch = bytes(f("general.architecture").parts[-1]).decode()
        p = arch + "."
        def gi(n): return int(f(p + n).parts[-1].item())
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


# ─── Worker: batch-layer dispatch ───────────────────────────────────

def _worker_init(model_path):
    global _mdl, _lib
    _mdl = Model(model_path)
    _lib = ctypes.CDLL(SO)
    for fn in ["q4_0_matmul", "q4_1_matmul"]:
        f = getattr(_lib, fn)
        f.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int]
        f.restype = None


def _layer_chunk(layer, h_bytes, h_shape, start_row, n_rows_q, n_rows_kv, n_rows_ff):
    """Compute ALL 7 matmuls for one layer on one row chunk."""
    mdl = _mdl; lib = _lib
    h = np.frombuffer(h_bytes, dtype=np.float32).reshape(h_shape)
    n = mdl.n; nc = n["n_embd"]

    wnames = [f"blk.{layer}.attn_q.weight", f"blk.{layer}.attn_k.weight",
              f"blk.{layer}.attn_v.weight", f"blk.{layer}.attn_output.weight",
              f"blk.{layer}.ffn_gate.weight", f"blk.{layer}.ffn_up.weight",
              f"blk.{layer}.ffn_down.weight"]
    shapes = [(n_rows_q, nc), (n_rows_kv, nc), (n_rows_kv, nc),
              (n_rows_q, nc), (n_rows_ff, nc), (n_rows_ff, nc),
              (n_rows_q, n["n_ff"])]

    results = []
    for i, wn in enumerate(wnames):
        nr, nc_i = shapes[i]
        wi = mdl.get(wn)
        if wi is None:
            results.append(np.zeros(nr, dtype=np.float32))
            continue
        w, tt = wi
        ts = Q4_TS.get(tt, 18)
        total_rows = len(w) // ((nc_i // 32) * ts)

        # Compute chunk range for this worker
        cs = max(1, total_rows // 32)  # nw from outer scope... use parameter
        # Actually, just compute the full range since we pass start_row
        end_row = min(start_row + nr, total_rows)
        actual_nr = end_row - start_row
        if actual_nr <= 0:
            results.append(np.zeros(0, dtype=np.float32))
            continue

        out = np.zeros(actual_nr, dtype=np.float32)
        fn = "q4_1_matmul" if tt == 3 else "q4_0_matmul"
        getattr(lib, fn)(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            total_rows, nc_i, start_row, end_row)
        results.append(out)

    return results  # [q, k, v, o, gate, up, down]


# ─── Engine ─────────────────────────────────────────────────────────

class Engine:
    def __init__(self, path, nw=None):
        self.mdl = Model(path)
        self.n = self.mdl.n
        self.nw = nw or mp.cpu_count() // 2
        self._pool = mp.Pool(self.nw, initializer=_worker_init, initargs=(path,))

    def forward(self, token_id, kvc, pos):
        n = self.n
        h = self.mdl.get("token_embd.weight")[0][token_id].astype(np.float32)

        for l in range(n["n_layers"]):
            t0 = time.time()
            r = h.copy()
            h = h / np.sqrt(np.mean(h**2, keepdims=True) + 1e-5) * \
                self.mdl.get(f"blk.{l}.attn_norm.weight")[0]

            # Batch-dispatch all 7 matmuls to workers
            chunks = []
            nr_q = len(self.mdl.get(f"blk.{l}.attn_q.weight")[0]) // ((n["n_embd"]//32)*18)
            nr_kv = len(self.mdl.get(f"blk.{l}.attn_k.weight")[0]) // ((n["n_embd"]//32)*18)
            nr_ff = len(self.mdl.get(f"blk.{l}.ffn_gate.weight")[0]) // ((n["n_embd"]//32)*18)
            cs = max(1, max(nr_q, nr_kv, nr_ff) // self.nw)

            chunk_starts = []
            s = 0
            while s < max(nr_q, nr_kv, nr_ff):
                e = min(s + cs, max(nr_q, nr_kv, nr_ff))
                nq = min(nr_q - s, cs) if s < nr_q else 0
                nkv = min(nr_kv - s, cs) if s < nr_kv else 0
                nff = min(nr_ff - s, cs) if s < nr_ff else 0
                chunks.append((l, h.tobytes(), h.shape, s, nq, nkv, nff))
                chunk_starts.append(s)
                s = e

            all_results = self._pool.starmap(_layer_chunk, chunks)

            # Combine results
            q = np.zeros(nr_q, dtype=np.float32)
            k = np.zeros(nr_kv, dtype=np.float32)
            v = np.zeros(nr_kv, dtype=np.float32)
            o_att = np.zeros(nr_q, dtype=np.float32)
            gate = np.zeros(nr_ff, dtype=np.float32)
            up = np.zeros(nr_ff, dtype=np.float32)
            down = np.zeros(nr_q, dtype=np.float32)

            for i, s in enumerate(chunk_starts):
                res = all_results[i]
                if len(res) >= 7:
                    nr_chunk = len(res[0])
                    if nr_chunk > 0:
                        q[s:s+nr_chunk] = res[0][:nr_chunk]
                        k[s:min(s+nr_chunk, nr_kv)] = res[1][:min(nr_chunk, nr_kv-s)]
                        v[s:min(s+nr_chunk, nr_kv)] = res[2][:min(nr_chunk, nr_kv-s)]
                        o_att[s:s+nr_chunk] = res[3][:nr_chunk]
                        gate[s:min(s+nr_chunk, nr_ff)] = res[4][:min(nr_chunk, nr_ff-s)]
                        up[s:min(s+nr_chunk, nr_ff)] = res[5][:min(nr_chunk, nr_ff-s)]
                        down[s:s+nr_chunk] = res[6][:nr_chunk]

            hd, nh, nkh = n["head_dim"], n["n_head"], n["n_kv_head"]

            # RoPE
            freqs = 1.0 / (500000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
            cos = np.cos(pos * freqs); sin = np.sin(pos * freqs)
            for xx, nh_x in [(q.reshape(nh, hd), nh), (k.reshape(nkh, hd), nkh)]:
                xr = xx.reshape(nh_x, hd//2, 2)
                xr_rot = np.stack([-xr[..., 1], xr[..., 0]], axis=-1)
                xx[:] = (xr * cos.reshape(1, hd//2, 1) + xr_rot * sin.reshape(1, hd//2, 1)).reshape(nh_x, hd)

            if kvc[l]["k"].shape[0] <= pos:
                kvc[l]["k"] = np.vstack([kvc[l]["k"], k.reshape(1, -1)])
                kvc[l]["v"] = np.vstack([kvc[l]["v"], v.reshape(1, -1)])
            else:
                kvc[l]["k"][pos] = k; kvc[l]["v"][pos] = v

            # Attention
            ng = nh // nkh; att = np.zeros(nh * hd, dtype=np.float32)
            q2 = q.reshape(nh, hd)
            for hh in range(nh):
                kvh = hh // ng
                ks = kvc[l]["k"][:pos+1, kvh*hd:(kvh+1)*hd]
                vs = kvc[l]["v"][:pos+1, kvh*hd:(kvh+1)*hd]
                sc = q2[hh] @ ks.T - np.max(q2[hh] @ ks.T)
                att[hh*hd:(hh+1)*hd] = (np.exp(sc) / np.sum(np.exp(sc))) @ vs

            h = r + o_att

            # FFN
            r = h.copy()
            h = h / np.sqrt(np.mean(h**2, keepdims=True) + 1e-5) * \
                self.mdl.get(f"blk.{l}.ffn_norm.weight")[0]
            gate = gate / (1 + np.exp(-gate))
            h = r + down

            t1 = time.time()
            if pos == 0:
                print(f"  Layer {l}: {(t1-t0)*1000:.0f}ms", flush=True)

        h = h / np.sqrt(np.mean(h**2, keepdims=True) + 1e-5) * \
            self.mdl.get("output_norm.weight")[0]
        wo = self.mdl.get("output.weight") or self.mdl.get("token_embd.weight")
        return h @ wo[0].T if wo[0].ndim > 1 else h @ wo[0]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/onedev-workspace/work/Llama-3.2-1B-Instruct-Q4_0.gguf"
    e = Engine(path)
    n = e.n
    print(f"MojoLlama v2 ({e.nw} workers)")
    print(f"Model: {n['n_layers']}L/{n['n_embd']}D/{n['n_ff']}FF/{n['n_head']}H")

    kvc = [{"k": np.zeros((0, n["n_kv"]), dtype=np.float32),
            "v": np.zeros((0, n["n_kv"]), dtype=np.float32)}
           for _ in range(n["n_layers"])]

    t0 = time.time()
    logits = e.forward(128000, kvc, 0)
    print(f"First: {(time.time()-t0)*1000:.0f}ms, next={np.argmax(logits)}")
