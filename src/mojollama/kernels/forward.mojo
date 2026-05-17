"""MojoLlama forward pass — reads model.bin, runs inference using SIMD kernels.

Compile: mojo build forward.mojo -o mojollama_forward
Run: echo "token_id" | ./mojollama_forward model.bin
"""

from std.memory.unsafe_pointer import alloc

alias F32x8 = SIMD[DType.float32, 8]

# ─── Float16 → Float32 ────────────────────────────────────────────────

def f16_to_f32(h: UInt16) -> Float32:
    from std.sys.info import CompilationTarget
    from std.memory import bitcast
    comptime if CompilationTarget.has_avx2():
        return Float32(bitcast[DType.float16](h))
    else:
        var sign = (UInt32(h) >> 15) & 1
        var exp = (UInt32(h) >> 10) & 0x1f
        var mant = UInt32(h) & 0x3ff
        if exp == 0:
            if mant == 0: return Float32(bitcast[DType.float32](sign << 31))
            var m = mant; var c: UInt32 = 0
            while m > 0: m >>= 1; c += 1
            var sh = 24 - c
            return Float32(bitcast[DType.float32]((sign << 31) | ((UInt32(113 - sh)) << 23) | ((mant << (sh + 13)) & 0x7fffff)))
        elif exp == 31:
            return Float32(bitcast[DType.float32]((sign << 31) | 0x7f800000 | (mant << 13)))
        else:
            return Float32(bitcast[DType.float32]((sign << 31) | ((exp + 112) << 23) | (mant << 13))))


# ─── Q4_0 Block Dot ───────────────────────────────────────────────────

alias U8x16 = SIMD[DType.uint8, 16]

def q4_block_dot(scale: Float32, nibbles: U8x16,
                 x0: F32x8, x1: F32x8, x2: F32x8, x3: F32x8) -> Float32:
    @parameter
    def dg(s: Int, xv: F32x8) -> Float32:
        var v = F32x8()
        for j in range(8):
            var bi = s + j // 2
            var b = nibbles[bi]
            v[j] = Float32(Int8(b & 15) - 8) if j % 2 == 0 else Float32(Int8((b >> 4) & 15) - 8)
        return (v * scale * xv).reduce_add()
    return dg(0, x0) + dg(4, x1) + dg(8, x2) + dg(12, x3)


# ─── Q4_0 Matmul ──────────────────────────────────────────────────────

def q4_matmul(
    w: UnsafePointer[UInt8],
    inp: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    n_rows: Int, n_cols: Int,
):
    var bpr = n_cols // 32
    for row in range(n_rows):
        var total: Float32 = 0.0
        for blk in range(bpr):
            var off = (row * bpr + blk) * 18
            var lo = w.load(off)
            var hi = w.load(off + 1)
            var scale = f16_to_f32((UInt16(hi) << 8) | UInt16(lo))
            var nibs = w.load[width=16](off + 2)
            var inp_off = blk * 32
            var x0 = inp.load[width=8](inp_off)
            var x1 = inp.load[width=8](inp_off + 8)
            var x2 = inp.load[width=8](inp_off + 16)
            var x3 = inp.load[width=8](inp_off + 24)
            total += q4_block_dot(scale, nibs, x0, x1, x2, x3)
        out.store(row, total)


# ─── RMSNorm ───────────────────────────────────────────────────────────

def rms_norm(
    x: UnsafePointer[Float32],
    w: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    n: Int, eps: Float32,
):
    var ss: Float32 = 0.0
    for i in range(n):
        var v = x.load(i)
        ss += v * v
    var rms = Float32(1.0 / Float64(Float32(n * n) * ss + eps).sqrt())
    for i in range(n):
        out.store(i, x.load(i) * w.load(i) * rms)


# ─── SiLU ──────────────────────────────────────────────────────────────

def silu(x: Float32) -> Float32:
    return x / (1.0 + Float32(Float64(-x).exp()))


# ─── RoPE ──────────────────────────────────────────────────────────────

def rope(
    q: UnsafePointer[Float32],
    k: UnsafePointer[Float32],
    pos: Int, head_dim: Int, n_head: Int, n_kv_head: Int,
    sin_ptr: UnsafePointer[Float32],
    cos_ptr: UnsafePointer[Float32],
):
    for h in range(n_head):
        for d2 in range(head_dim // 2):
            var off = h * head_dim + d2 * 2
            var x0 = q.load(off)
            var x1 = q.load(off + 1)
            var c = cos_ptr.load(pos * head_dim + d2 * 2)
            var s = sin_ptr.load(pos * head_dim + d2 * 2 + 1)
            q.store(off, x0 * c - x1 * s)
            q.store(off + 1, x0 * s + x1 * c)
    for h in range(n_kv_head):
        for d2 in range(head_dim // 2):
            var off = h * head_dim + d2 * 2
            var x0 = k.load(off)
            var x1 = k.load(off + 1)
            var c = cos_ptr.load(pos * head_dim + d2 * 2)
            var s = sin_ptr.load(pos * head_dim + d2 * 2 + 1)
            k.store(off, x0 * c - x1 * s)
            k.store(off + 1, x0 * s + x1 * c)


# ─── Attention (single token) ─────────────────────────────────────────

def attention(
    q: UnsafePointer[Float32],
    k: UnsafePointer[Float32],
    v: UnsafePointer[Float32],
    k_cache: UnsafePointer[Float32],
    v_cache: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    pos: Int, n_head: Int, n_kv_head: Int, head_dim: Int,
):
    var n_kv_groups = n_head // n_kv_head
    # Store K,V into cache
    for h in range(n_kv_head):
        for d in range(head_dim):
            k_cache.store(pos * n_kv_head * head_dim + h * head_dim + d, k.load(h * head_dim + d))
            v_cache.store(pos * n_kv_head * head_dim + h * head_dim + d, v.load(h * head_dim + d))
    
    # Compute attention for each query head
    for h in range(n_head):
        var kv_h = h // n_kv_groups
        var max_score: Float32 = -1e30
        var scores = alloc[Float32](pos + 1)
        for t in range(pos + 1):
            var s: Float32 = 0.0
            for d in range(head_dim):
                s += q.load(h * head_dim + d) * k_cache.load(t * n_kv_head * head_dim + kv_h * head_dim + d)
            scores.store(t, s)
            if s > max_score: max_score = s
        
        var sum_exp: Float32 = 0.0
        for t in range(pos + 1):
            var sv = scores.load(t) - max_score
            var e = Float32(Float64(sv).exp())
            scores.store(t, e)
            sum_exp += e
        
        var o: Float32 = 0.0
        for d in range(head_dim):
            var total: Float32 = 0.0
            for t in range(pos + 1):
                total += scores.load(t) / sum_exp * v_cache.load(t * n_kv_head * head_dim + kv_h * head_dim + d)
            out.store(h * head_dim + d, total)
        
        scores.free()


# ─── Model Weights ─────────────────────────────────────────────────────

struct ModelWeights:
    var token_embd: UnsafePointer[Float32]
    var output_norm: UnsafePointer[Float32]
    var n_layers: Int
    var n_embd: Int
    var n_head: Int
    var n_kv_head: Int
    var n_ff: Int
    var n_vocab: Int
    var head_dim: Int
    
    # Per-layer weights — stored as flat arrays of pointers
    var attn_norm: UnsafePointer[UnsafePointer[Float32]]
    var ffn_norm: UnsafePointer[UnsafePointer[Float32]]
    var wq: UnsafePointer[UnsafePointer[UInt8]]
    var wk: UnsafePointer[UnsafePointer[UInt8]]
    var wv: UnsafePointer[UnsafePointer[UInt8]]
    var wo: UnsafePointer[UnsafePointer[UInt8]]
    var wgate: UnsafePointer[UnsafePointer[UInt8]]
    var wup: UnsafePointer[UnsafePointer[UInt8]]
    var wdown: UnsafePointer[UnsafePointer[UInt8]]


def main() raises:
    from python import Python
    var sys = Python.import_module("sys")
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    
    var argv = sys.argv
    var argc = builtins.len(argv)
    if argc < 2:
        print("Usage: mojollama_forward model.bin")
        return
    
    var model_path = String(argv[1])
    
    # Load model.bin via Python interop
    print("Loading model..." + model_path)
    var t0 = Python.evaluate("__import__('time').time()")
    
    var f = builtins.open(model_path, "rb")
    var header = f.read(64)
    
    # Parse header
    var hdr_arr = np.frombuffer(header, dtype=np.uint8)
    var magic = bytes(hdr_arr[:4].tobytes()).decode()
    var n_layers = int(np.frombuffer(header[8:12], dtype=np.int32)[0])
    var n_embd = int(np.frombuffer(header[12:16], dtype=np.int32)[0])
    var n_head = int(np.frombuffer(header[16:20], dtype=np.int32)[0])
    var n_kv_head = int(np.frombuffer(header[20:24], dtype=np.int32)[0])
    var n_ff = int(np.frombuffer(header[24:28], dtype=np.int32)[0])
    var n_vocab = int(np.frombuffer(header[28:32], dtype=np.int32)[0])
    var head_dim = int(np.frombuffer(header[32:36], dtype=np.int32)[0])
    
    print("Model:", n_layers, "layers,", n_embd, "dim,", n_head, "heads,", n_vocab, "vocab")
    
    # Read all remaining data
    var data = f.read()
    f.close()
    
    print("Loaded", builtins.len(data) / 1024**3, "GB")
    
    # NOTE: Full weight loading and forward pass implementation
    # would follow here. This is the scaffold — the moment we 
    # have a working pipeline for pointer construction from Python
    # numpy arrays, we can load weights and run inference.
    
    var t1 = Python.evaluate("__import__('time').time()")
    print("Setup in", t1 - t0, "s")
    print("MojoLlama forward pass scaffold ready.")
    print("Next: Complete weight loading and layer loop with SIMD kernels.")
