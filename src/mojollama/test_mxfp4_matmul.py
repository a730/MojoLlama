#!/usr/bin/env python3
"""Test MXFP4 matmul in cengine_batch_instr.so"""
import ctypes, numpy as np, os, sys
os.environ['OMP_NUM_THREADS'] = '8'

ce = ctypes.CDLL('/onedev-workspace/work/src/mojollama/kernels/cengine_batch_instr.so')
cu = ctypes.POINTER(ctypes.c_uint8); cf = ctypes.POINTER(ctypes.c_float); ci = ctypes.c_int
ce.mxfp4_batch_matmul.argtypes = [cu, cf, cf, ci, ci, ci]
ce.mxfp4_batch_matmul.restype = None

n_rows, n_cols = 2, 32
w_f32 = np.array([[1.0, 2.0, 3.0, 4.0] + [0]*28, [5.0, 6.0, 7.0, 8.0] + [0]*28], dtype=np.float32)
x = np.array([0.5, 1.0, 1.5, 2.0] + [0]*28, dtype=np.float32)
y_expected = w_f32 @ x
print(f'Expected: {y_expected}')

# MXFP4 quantize: 4-bit per value, E8M0 exponent per block
n_blk = 1
w_mxfp4 = np.zeros((n_rows, n_blk, 17), dtype=np.uint8)
for r in range(n_rows):
    # Find scale as power of 2
    amax = np.max(np.abs(w_f32[r]))
    exp = int(round(np.log2(amax / 7.0))) + 127
    exp = max(0, min(255, exp))
    scale = 2.0 ** (exp - 127)
    w_mxfp4[r, 0, 16] = exp
    for i in range(32):
        nibble = max(-8, min(7, int(round(w_f32[r, i] / scale))))
        byte_idx = i // 2
        nibble_pos = i % 2
        if nibble_pos == 0:  # lower nibble
            w_mxfp4[r, 0, byte_idx] = (w_mxfp4[r, 0, byte_idx] & 0xF0) | (nibble & 0x0F)
        else:  # upper nibble
            w_mxfp4[r, 0, byte_idx] = (w_mxfp4[r, 0, byte_idx] & 0x0F) | ((nibble & 0x0F) << 4)

w_mxfp4 = np.ascontiguousarray(w_mxfp4)
y = np.zeros(n_rows, dtype=np.float32)

ce.mxfp4_batch_matmul(w_mxfp4.ctypes.data_as(cu), x.ctypes.data_as(cf),
                      y.ctypes.data_as(cf), ci(n_rows), ci(n_cols), ci(1))
print(f'MXFP4 result: {y}')
diff = np.abs(y - y_expected).max()
print(f'Max diff: {diff:.3f}')
