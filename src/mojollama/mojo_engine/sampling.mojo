# MojoLlama Sampling — Pure Mojo top-k extraction + token sampling
# WHAT:  quantize_row_q8_0 + top-k logits + sampling functions.
# WHY:   Replaces logits_topk.c + logits_avx2.c + simd_ops.c.
#        Final step in text generation — selects which token to emit.
# WHEN:  May 2026 — pure Mojo port.
from std import math
from std.algorithm.backend.cpu.parallelize import parallelize

def encode_f16(f: Float32) -> UInt16:
    if f == 0.0: return UInt16(0)
    var sign: UInt16 = 0; var fd = Float64(f)
    if fd < 0: sign = UInt16(1); fd = -fd
    var e: Int = 0
    if fd >= 1.0:
        while fd >= 2.0: fd *= 0.5; e += 1
    else:
        while fd < 1.0: fd *= 2.0; e -= 1
    var he = e + 15
    if he < 0: he = 0
    if he > 31: he = 31
    var m = Int(fd * 2048.0)
    if m > 1023: m = 1023
    return UInt16(Int(sign) << 15 | he << 10 | m)

def quantize_row_q8_0(x: UnsafePointer[Float32, MutExternalOrigin],
                       q8: UnsafePointer[UInt8, MutExternalOrigin], n: Int):
    """Quantize f32 vector to Q8_0 format (34 bytes per 32 values)."""
    var nb = n // 32
    for b in range(nb):
        var xo = b * 32; var bo = b * 34
        var amax: Float32 = 0.0
        for j in range(32):
            var v = x.load(xo + j)
            if v < 0: v = -v
            if v > amax: amax = v
        if amax == 0.0: amax = 1.0
        var d = amax / 127.0; var inv_d = 127.0 / amax
        var su = encode_f16(d)
        q8.store(bo, UInt8(su & 0xFF)); q8.store(bo+1, UInt8((su >> 8) & 0xFF))
        for j in range(32):
            var qv = Int32(x.load(xo + j) * inv_d)
            if qv > 127: qv = 127
            if qv < -128: qv = -128
            if qv < 0: qv += 256
            q8.store(bo + 2 + j, UInt8(qv & 0xFF))

def topk_extract(logits: UnsafePointer[Float32, MutExternalOrigin],
                 n_vocab: Int, k: Int,
                 tk_val: UnsafePointer[Float32, MutExternalOrigin],
                 tk_idx: UnsafePointer[Int32, MutExternalOrigin]):
    """Extract top-k logits using min-heap (parallel arrays for compat)."""
    var heap_size = 0
    for i in range(n_vocab):
        var v = logits.load(i)
        if heap_size < k:
            tk_val.store(heap_size, v); tk_idx.store(heap_size, i)
            heap_size += 1
            # Sift up
            var j = heap_size - 1
            while j > 0:
                var p = (j - 1) // 2
                if tk_val.load(j) < tk_val.load(p):
                    var tv = tk_val.load(j); var ti = tk_idx.load(j)
                    tk_val.store(j, tk_val.load(p)); tk_idx.store(j, tk_idx.load(p))
                    tk_val.store(p, tv); tk_idx.store(p, ti)
                    j = p
                else: break
        elif v > tk_val.load(0):
            tk_val.store(0, v); tk_idx.store(0, i)
            # Sift down
            var j = 0
            while 2 * j + 1 < heap_size:
                var c = 2 * j + 1
                if c + 1 < heap_size and tk_val.load(c+1) < tk_val.load(c): c += 1
                if tk_val.load(j) <= tk_val.load(c): break
                var tv = tk_val.load(j); var ti = tk_idx.load(j)
                tk_val.store(j, tk_val.load(c)); tk_idx.store(j, tk_idx.load(c))
                tk_val.store(c, tv); tk_idx.store(c, ti)
                j = c
    # Sort descending
    for i in range(heap_size - 1):
        var max_idx = i
        for j in range(i + 1, heap_size):
            if tk_val.load(j) > tk_val.load(max_idx): max_idx = j
        if max_idx != i:
            var tv = tk_val.load(i); var ti = tk_idx.load(i)
            tk_val.store(i, tk_val.load(max_idx)); tk_idx.store(i, tk_idx.load(max_idx))
            tk_val.store(max_idx, tv); tk_idx.store(max_idx, ti)

def sample_argmax(tk_val: UnsafePointer[Float32, MutExternalOrigin],
                  tk_idx: UnsafePointer[Int32, MutExternalOrigin],
                  k: Int) -> Int:
    """Return index of highest logit (greedy)."""
    return Int(tk_idx.load(0))

def softmax_with_temp(logits: UnsafePointer[Float32, MutExternalOrigin],
                      probs: UnsafePointer[Float32, MutExternalOrigin],
                      n: Int, temperature: Float32):
    """Compute softmax with temperature."""
    var max_val: Float32 = -1e30
    for i in range(n):
        if logits.load(i) > max_val: max_val = logits.load(i)
    var inv_temp = 1.0 / temperature if temperature > 0 else 1e10
    var sum_exp: Float32 = 0.0
    for i in range(n):
        var e = math.exp((logits.load(i) - max_val) * inv_temp)
        if e > 1e30: e = 1e30
        probs.store(i, e); sum_exp += e
    var inv_sum = 1.0 / (sum_exp + 1e-10)
    for i in range(n): probs.store(i, probs.load(i) * inv_sum)

def sample_topk(probs: UnsafePointer[Float32, MutExternalOrigin],
                tk_idx: UnsafePointer[Int32, MutExternalOrigin],
                k: Int, rng_val: Float32) -> Int:
    """Sample from top-k distribution."""
    var cumsum: Float32 = 0.0
    for i in range(k):
        cumsum += probs.load(tk_idx.load(i))
        if cumsum > rng_val: return Int(tk_idx.load(i))
    return Int(tk_idx.load(k - 1))
