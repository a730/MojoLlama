#!/usr/bin/env python3
"""Convert C-extracted Gemma 4 weights (Q4_K/Q6_K/F32) to Q8_0 for Mojo engine."""
import os, struct, sys, numpy as np, gguf, time
from gguf import GGMLQuantizationType as QType

QK, QB = 32, 34
ARCH = {'NE': 2560, 'FF': 10240, 'NV': 262144, 'NL': 42}

def q8_rb(nc): return ((nc + QK - 1) // QK) * QB

def f32_to_q8_bytes(f32):
    """Convert 2D f32 array to Q8_0 bytes."""
    if len(f32.shape) == 1: f32 = f32.reshape(1, -1)
    nr, nc = f32.shape
    buf = bytearray()
    for r in range(nr):
        for blk in range((nc + QK - 1) // QK):
            s = blk * QK; e = min(s + QK, nc)
            block = f32[r, s:e]
            max_abs = float(np.max(np.abs(block))) if len(block) > 0 else 0.0
            scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
            buf += struct.pack('<e', np.float16(scale))
            for i in range(QK):
                if i < len(block):
                    qv = int(round(block[i] / scale))
                    if qv > 127: qv = 127
                    if qv < -128: qv = -128
                    buf += bytes([qv & 0xFF])
                else:
                    buf += bytes([0])
    return bytes(buf)

def convert(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    total_in = 0; total_out = 0; converted = 0
    
    files = sorted(os.listdir(in_dir))
    for fname in files:
        if not fname.endswith('.bin'): continue
        in_path = os.path.join(in_dir, fname)
        in_size = os.path.getsize(in_path)
        total_in += in_size
        
        # Skip: determine format and convert
        f32 = None
        data = np.frombuffer(open(in_path, 'rb').read(), dtype=np.uint8)
        
        # Determine tensor type from size using shape info
        # Try gguf.dequantize with all known types
        for qtype in [QType.F32, QType.BF16, QType.Q4_K, QType.Q6_K]:
            try:
                result = gguf.dequantize(data, qtype)
                if len(result.shape) == 1:
                    result = result.reshape(1, -1)
                f32 = np.array(result, dtype=np.float32)
                break
            except:
                continue
        
        if f32 is None:
            # Try direct read as F32
            try:
                f32 = np.frombuffer(data, dtype=np.float32).copy()
            except:
                pass
        
        if f32 is None:
            # Try reading as BF16 (bitshift conversion)
            try:
                bits = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
                f32 = (bits << 16).view(np.float32)
            except:
                pass
        
        if f32 is None:
            print(f"  SKIP {fname}: unknown format ({in_size}B)")
            continue
        
        # Fix NaN/Inf values
        f32 = np.nan_to_num(f32, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(f32.shape) == 1:
            f32 = f32.reshape(1, -1)
        
        # Convert to Q8_0
        q8_bytes = f32_to_q8_bytes(f32)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, 'wb') as f:
            f.write(q8_bytes)
        
        total_out += len(q8_bytes)
        converted += 1
        
        if converted % 50 == 0:
            print(f"  {converted}/{len(files)} files ({time.time()-t0:.0f}s)")
    
    print(f"Converted {converted}/{len(files)} files: {total_in>>20}MB -> {total_out>>20}MB in {time.time()-t0:.0f}s")

if __name__ == '__main__':
    convert(sys.argv[1] if len(sys.argv) > 1 else '/tmp/weights_e4b/',
            sys.argv[2] if len(sys.argv) > 2 else '/tmp/weights_e4b_q8/')
