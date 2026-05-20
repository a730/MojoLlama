#!/usr/bin/env python3
import sys, numpy as np, os, ctypes
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
os.environ['OMP_NUM_THREADS'] = '8'
from turbo_engine_v7_moe import TurboEngineV7MoE

engine = TurboEngineV7MoE('/onedev-workspace/work/models/ZAYA1-8B-MXFP4.gguf', n_threads=8)
e = engine; l0 = e._layers[0]

print(f'attn_q_qt={l0.attn_q_qt}, attn_q_raw={l0.attn_q_raw}')

# Check cengine
ce = getattr(e, '_cengine', None)
print(f'cengine: {ce}')
if ce:
    cu = ctypes.POINTER(ctypes.c_uint8); cf = ctypes.POINTER(ctypes.c_float); ci = ctypes.c_int
    ce.mxfp4_batch_matmul.argtypes = [cu, cf, cf, ci, ci, ci]
    ce.mxfp4_batch_matmul.restype = None
    out = np.zeros(e.n_head * e.head_dim, dtype=np.float32)
    ce.mxfp4_batch_matmul(l0.attn_q_raw, e._p_x_norm,
                           out.ctypes.data_as(cf),
                           ci(e.n_head * e.head_dim), ci(e.n_embd), ci(1))
    print(f'Direct mxfp4_batch_matmul: non-zero={np.any(out!=0)} sum={out.sum():.2f}')

# Monkey-patch mxm to debug
from forward import zaya
orig_forward = zaya.ForwardZaya.forward
def debug_forward(self, token_id):
    import builtins
    # Override mxm for debugging
    e2 = self.engine
    ce2 = getattr(e2, '_cengine', None)
    kern = e2._kern
    cf2 = ctypes.POINTER(ctypes.c_float); ci2 = ctypes.c_int; cu2 = ctypes.POINTER(ctypes.c_uint8)
    
    def mxm_debug(w, x, out, nr, nc, qt):
        qt_val = qt.value if isinstance(qt, ctypes.c_int) else qt
        use_mxfp4 = ce2 is not None and qt_val == 39
        if use_mxfp4:
            ce2.mxfp4_batch_matmul(w, x, out, nr, nc, ci2(1))
        else:
            kern.quant_matmul_omp(w, x, out, nr, nc, qt)
    
    # Patch into the instance for this call
    self.mxm = mxm_debug
    return orig_forward(self, token_id)

zaya.ForwardZaya.forward = debug_forward
# Also need to make the forward method use self.mxm instead of mxm
# This is getting complex. Let me just directly patch the forward to bypass the mxm check

y0 = engine._arch_forward.forward(0)
y1 = engine._arch_forward.forward(1)
print(f'\nFwd 0: NaN={np.isnan(y0).sum()}, range=[{y0.min():.1f}, {y0.max():.1f}]')
print(f'Fwd 1: NaN={np.isnan(y1).sum()}, range=[{y1.min():.1f}, {y1.max():.1f}]')
print(f'Diff: {np.abs(y1-y0).max():.1f}')
