#!/usr/bin/env python3
"""Check workspace buffer overflow and other potential issues."""
import sys, os, numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from turbo_engine_v7_moe import TurboEngineV7MoE

print("Loading model...", flush=True)
e = TurboEngineV7MoE('/tmp/models/gpt-oss-20b-Q4_K_M.gguf', n_threads=1)
print("Model loaded.", flush=True)

N = e.n_embd; NH = e.n_head; NKH = e.n_kv_head * e.head_dim; HD = e.head_dim
gqa_rep = NH // e.n_kv_head

print(f"\nModel dimensions:", flush=True)
print(f"  N={N}, NH={NH}, NKH={NKH}, HD={HD}, gqa_rep={gqa_rep}", flush=True)

# Check workspace size
ws_size = e._gqa_ws_size
print(f"\nWorkspace allocated: {ws_size} floats ({ws_size * 4 / 1024:.0f} KB)", flush=True)
print(f"  For seq_len=4096: needs NH*seq_len + gqa_rep*seq_len = {NH*4096 + gqa_rep*4096}", flush=True)
if ws_size < NH * 4096 + gqa_rep * 4096:
    print(f"  *** WORKSPACE OVERFLOW for seq_len=4096! Missing {(NH + gqa_rep)*4096 - ws_size} floats", flush=True)
else:
    print(f"  Workspace OK for seq_len=4096", flush=True)

# Check: does _p_gqa_ws point to the right place?
print(f"\n_gqa_workspace addr: {e._gqa_workspace.ctypes.data}", flush=True)
print(f"_p_gqa_ws addr: {e._p_gqa_ws}", flush=True)

# Check batch_forward path — is there a bug there?
print(f"\nChecking batch forward availability:", flush=True)
print(f"  _c_batch_ready: {getattr(e, '_c_batch_ready', 'N/A')}", flush=True)
print(f"  Has batch forward: {hasattr(e, 'forward_c_batch')}", flush=True)

# Check if there's a forward_c_batch for gpt-oss
if hasattr(e, '_kern') and e._kern is not None:
    for attr in dir(e._kern):
        if 'batch' in attr.lower() or 'forward' in attr.lower():
            print(f"  C function: {attr}", flush=True)

# Check if pos is correctly handled
print(f"\npos = {e.pos}", flush=True)

# CRITICAL TEST: monkey-patch to check the actual seq_len passed to GQA
import ctypes
cf = ctypes.POINTER(ctypes.c_float)
ci = ctypes.c_int

original_gqa = e._gqa_attn.gqa_attention_decode

call_count = [0]
def traced_gqa(q, k, v, out, seq_len, n_head, n_kv_head, head_dim, ws):
    call_count[0] += 1
    print(f"  GQA call {call_count[0]}: seq_len={seq_len}, n_head={n_head}, n_kv_head={n_kv_head}, head_dim={head_dim}", flush=True)
    return original_gqa(q, k, v, out, seq_len, n_head, n_kv_head, head_dim, ws)

e._gqa_attn.gqa_attention_decode = traced_gqa

# Now do 2 forward calls
e.reset()
print(f"\n=== Forward call 1 (token 42) ===", flush=True)
l1 = e.forward(42).copy()
print(f"pos after call 1: {e.pos}", flush=True)

print(f"\n=== Forward call 2 (token 99) ===", flush=True)
l2 = e.forward(99).copy()
print(f"pos after call 2: {e.pos}", flush=True)

print(f"\n=== Forward call 3 (token 42) ===", flush=True)
l3 = e.forward(42).copy()
print(f"pos after call 3: {e.pos}", flush=True)

# Check: is l2 (token 99 with context) different from just token 99 with no context?
e.reset()
l99_alone = e.forward(99).copy()
print(f"\nToken 99 alone vs with context (token 42 then 99): {np.max(np.abs(l2 - l99_alone)):.6f}")

# Check: is l3 (token 42 with 42 and 99 in cache) different from l1 (token 42 alone)?
print(f"Token 42 alone vs with 2 tokens context: {np.max(np.abs(l3 - l1)):.6f}")

# The KEY question: is the output on step 3 different from step 1?
# If V values for token 42 are the same at all positions (which they are for the same layer input),
# then the attention output would be the same, and the hidden state propagation would make
# everything identical.
# 
# BUT! After step 2 with token 99, the hidden state at step 3 should be DIFFERENT from
# the hidden state at step 1 (because step 2's output was different, affecting subsequent layers).
# 
# So: step 3 output should be different from step 1!

print(f"\nCRITICAL: Is step 3 (token 42 after token 99) different from step 1 (token 42 alone)?")
print(f"  Step 1 logits[:5] = {l1[:5]}")
print(f"  Step 3 logits[:5] = {l3[:5]}")
print(f"  Diff = {np.max(np.abs(l3 - l1)):.6f}")
print(f"  Different? {np.max(np.abs(l3 - l1)) > 0.001}")
