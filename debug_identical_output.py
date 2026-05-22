#!/usr/bin/env python3
"""Debug MojoLlama identical output bug."""
import sys, os, numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from turbo_engine_v7_moe import TurboEngineV7MoE

print("Loading model...", flush=True)
e = TurboEngineV7MoE('/tmp/models/gpt-oss-20b-Q4_K_M.gguf', n_threads=1)
print("Model loaded.", flush=True)

# Store initial state
print(f"Initial pos={e.pos}, kv_len[0]={e.kv_len[0]}", flush=True)
print(f"Initial _x[:5] = {e._x[:5]}", flush=True)
print(f"n_embd={e.n_embd}, n_head={e.n_head}, n_kv_head={e.n_kv_head}, head_dim={e.head_dim}", flush=True)

N = e.n_embd
NH = e.n_head
NKH = e.n_kv_head * e.head_dim
HD = e.head_dim
print(f"N={N}, NH={NH}, NKH={NKH}, HD={HD}", flush=True)

# First forward call with token 42
print("\n=== Forward call 1 (token 42) ===", flush=True)
l1 = e.forward(42).copy()
print(f"pos after call 1: {e.pos}", flush=True)
print(f"kv_len[0] after call 1: {e.kv_len[0]}", flush=True)
print(f"kv_k[0,0,:3] after call 1: {e.kv_k[0,0,:3]}", flush=True)
print(f"logits[:5] = {l1[:5]}", flush=True)

# Second forward call with token 42 (same token)
print("\n=== Forward call 2 (token 42) ===", flush=True)
l2 = e.forward(42).copy()
print(f"pos after call 2: {e.pos}", flush=True)
print(f"kv_len[0] after call 2: {e.kv_len[0]}", flush=True)
print(f"kv_k[0,0,:3] = {e.kv_k[0,0,:3]}", flush=True)
print(f"kv_k[0,1,:3] = {e.kv_k[0,1,:3]}", flush=True)
print(f"logits[:5] = {l2[:5]}", flush=True)

max_diff = np.max(np.abs(l1 - l2))
print(f"\nMax diff between call 1 and call 2: {max_diff:.6f}", flush=True)

# Now let's trace what happens layer by layer
# Reset and instrument
print("\n=== Layer-by-layer diagnostic ===", flush=True)
e.reset()
print(f"After reset: pos={e.pos}, kv_len[0]={e.kv_len[0]}", flush=True)

# We'll monkey-patch to trace hidden state after each layer
original_forward = e.forward.__func__ if hasattr(e.forward, '__func__') else None

# Instead, let's create a diagnostic using available hooks
# Check if _arch_forward is being used
print(f"_arch_forward is {e._arch_forward}", flush=True)
print(f"arch_prefix = {e.arch_prefix if hasattr(e, 'arch_prefix') else 'N/A'}", flush=True)

# Check buffer aliasing
print(f"\nBuffer addresses:", flush=True)
print(f"  _x     addr: {e._x.ctypes.data}", flush=True)
print(f"  _x_norm addr: {e._x_norm.ctypes.data}", flush=True)
print(f"  _residual addr: {e._residual.ctypes.data}", flush=True)
print(f"  _qk    addr: {e._qk.ctypes.data}", flush=True)
print(f"  _v     addr: {e._v.ctypes.data}", flush=True)
print(f"  _att_out addr: {e._att_out.ctypes.data}", flush=True)
print(f"  _o_proj addr: {e._o_proj.ctypes.data}", flush=True)

# Check for buffer overlaps
bufs = {
    '_x': (e._x.ctypes.data, e._x.nbytes),
    '_x_norm': (e._x_norm.ctypes.data, e._x_norm.nbytes),
    '_residual': (e._residual.ctypes.data, e._residual.nbytes),
    '_qk': (e._qk.ctypes.data, e._qk.nbytes),
    '_v': (e._v.ctypes.data, e._v.nbytes),
    '_att_out': (e._att_out.ctypes.data, e._att_out.nbytes),
    '_o_proj': (e._o_proj.ctypes.data, e._o_proj.nbytes),
    '_gate': (e._gate.ctypes.data, e._gate.nbytes),
    '_up': (e._up.ctypes.data, e._up.nbytes),
    '_silu_gate': (e._silu_gate.ctypes.data, e._silu_gate.nbytes),
    '_ffn_out': (e._ffn_out.ctypes.data, e._ffn_out.nbytes),
    '_logits': (e._logits.ctypes.data, e._logits.nbytes),
    'kv_k[0]': (e.kv_k[0].ctypes.data, e.kv_k[0].nbytes),
}

for name1, (addr1, sz1) in bufs.items():
    for name2, (addr2, sz2) in bufs.items():
        if name1 < name2:
            # Check if they overlap
            start1, end1 = addr1, addr1 + sz1
            start2, end2 = addr2, addr2 + sz2
            if start1 < end2 and start2 < end1:
                print(f"  *** OVERLAP: {name1} and {name2} overlap!")
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                print(f"      Overlap of {overlap_end - overlap_start} bytes")

# Test: call forward with intermediate value tracing
# First, save the original forward method
print("\n=== Testing with monkey-patched forward ===", flush=True)

# Let me check if RoPE is working
print(f"\nRope freq_base = {e.rope_freq_base}", flush=True)
print(f"Rope dim = {e.rope_dim if hasattr(e, 'rope_dim') else 'N/A'}", flush=True)

# Let me manually test with a simple approach
# After the first forward, let's check _x
e.reset()
print(f"\nBefore any forward: _x[:5] = {e._x[:5]}", flush=True)
print(f"pos = {e.pos}", flush=True)

l1 = e.forward(42).copy()
print(f"After call 1: _x[:5] = {e._x[:5]}", flush=True)
print(f"pos = {e.pos}", flush=True)

l2 = e.forward(43).copy()
print(f"After call 2 (token 43): _x[:5] = {e._x[:5]}", flush=True)
print(f"pos = {e.pos}", flush=True)

print(f"\nMax diff btwn token 42 and 43 (call 1 vs call 2): {np.max(np.abs(l1 - l2)):.6f}", flush=True)

# Now check: what if we call with diff token on step 2?
e.reset()
l1a = e.forward(42).copy()
l2a = e.forward(99).copy()
print(f"\nToken 42 step 1, token 99 step 2:", flush=True)
print(f"  Max diff: {np.max(np.abs(l1a - l2a)):.6f}", flush=True)

e.reset()
l1b = e.forward(42).copy()
l2b = e.forward(42).copy()
print(f"Token 42 step 1, token 42 step 2:", flush=True)
print(f"  Max diff: {np.max(np.abs(l1b - l2b)):.6f}", flush=True)

# So the question is: does call 2 always give the same output regardless of input?
print(f"\nl2a[:10] = {l2a[:10]}", flush=True)
print(f"l2b[:10] = {l2b[:10]}", flush=True)
print(f"Max diff l2a vs l2b: {np.max(np.abs(l2a - l2b)):.6f}", flush=True)
