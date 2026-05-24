#!/usr/bin/env python3
"""Transpose 3D expert MoE weights: [d0, d1, n_exp] → [d1, d0, n_exp] per expert.
   Each expert's 2D slice is dequantized, transposed, and requantized individually."""
import os, struct, numpy as np, sys

QK, QB = 32, 34

def q8_rb(n): return ((n + QK - 1) // QK) * QB

def deq8_blk(blk):
    """Dequantize a Q8_0 block (34 bytes → 32 float32)."""
    s = struct.unpack('<e', blk[:2])[0]
    q = np.frombuffer(blk[2:], dtype=np.int8, count=QK)
    return q.astype(np.float32) * s

def q8_quant(arr):
    """Quantize float32 array to Q8_0 bytes."""
    n = len(arr)
    n_blk = (n + QK - 1) // QK
    buf = bytearray()
    for b in range(n_blk):
        s = b * QK; e = min(s + QK, n)
        blk = arr[s:e]
        max_abs = float(np.max(np.abs(blk))) if len(blk) > 0 else 0.0
        scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
        qvals = np.clip(np.round(blk / scale), -128, 127).astype(np.int8)
        buf += struct.pack('<e', np.float16(scale))
        buf += qvals.tobytes()
        buf += b'\x00' * (QK - len(qvals))
    return bytes(buf)

def transpose_expert_file(in_path, out_path, d0, d1, n_exp):
    """Transpose 3D expert weights: each expert [d0, d1] → [d1, d0]."""
    with open(in_path, 'rb') as f:
        raw = f.read()
    
    # Each expert: d0 rows × d1 cols in Q8_0
    exp_stride = q8_rb(d1)  # bytes per row for input
    exp_size = d0 * exp_stride  # bytes per expert (input)
    
    print(f"  {os.path.basename(in_path)}: {d0}×{d1}×{n_exp} → {d1}×{d0}×{n_exp}")
    
    buf = bytearray()
    for e in range(n_exp):
        # Dequantize expert e
        f32 = np.zeros(d0 * d1, dtype=np.float32)
        for r in range(d0):
            off = e * d0 * exp_stride + r * exp_stride
            for b in range((d1 + QK - 1) // QK):
                blk_off = off + b * QB
                if blk_off + QB > len(raw): break
                blk = raw[blk_off:blk_off + QB]
                vals = deq8_blk(blk)
                start = r * d1 + b * QK
                end = min(start + QK, r * d1 + d1)
                f32[start:end] = vals[:end - start]
        
        # Reshape and transpose
        mat = f32.reshape(d0, d1).T  # Now [d1, d0]
        
        # Requantize
        for r in range(d1):
            buf += q8_quant(mat[r])
    
    with open(out_path, 'wb') as f:
        f.write(buf)
    return True

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    
    # Expert gate_up: [NE, 1408, 128] → per expert [2816, 1408] → [1408, 2816]
    os.makedirs(out_dir, exist_ok=True)
    
    for layer in range(30):
        for suffix, d0, d1, n_exp in [
            (f'blk_{layer}_ffn_gate_up_exps_weight.bin', 2816, 1408, 128),
            (f'blk_{layer}_ffn_down_exps_weight.bin', 704, 2816, 128),
        ]:
            in_path = os.path.join(in_dir, suffix)
            out_path = os.path.join(out_dir, suffix)
            if os.path.exists(in_path) and not os.path.exists(out_path):
                transpose_expert_file(in_path, out_path, d0, d1, n_exp)
    
    print("Done!")
