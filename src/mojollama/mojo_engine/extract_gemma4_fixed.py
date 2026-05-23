#!/usr/bin/env python3
"""
Gemma 4 weight extractor: GGUF → correctly-oriented Q8_0.
Fixes two bugs in the C pipeline:
  1. Q4_K dequant layout (C converter used wrong block structure)
  2. Weight shape transposition (GGUF stores [NE, QI], engine needs [QI, NE])

Usage: python3 extract_gemma4_fixed.py <arch> <gguf_path> <out_dir>
  arch: e2b, e4b, 26b, 31b
"""
import gguf, numpy as np, os, struct, sys, time

QK, QB = 32, 34

ARCH = {
    'e2b':  {'NE': 2048, 'FF': 16384, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 4,  'HD': 128},
    'e4b':  {'NE': 2560, 'FF': 10240, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 4,  'HD': 128},
    '26b':  {'NE': 2816, 'FF': 2112,  'NV': 262144, 'NL': 30, 'NH': 16, 'NK': 8,  'HD': 128, 'n_exp': 128, 'n_act': 4},
    '31b':  {'NE': 3072, 'FF': 24576, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 8,  'HD': 128},
}

def q8_quantize(f32: np.ndarray) -> bytes:
    """Quantize float32 1D array to Q8_0 bytes."""
    n = len(f32)
    n_blk = (n + QK - 1) // QK
    buf = bytearray()
    for b in range(n_blk):
        s = b * QK
        e = min(s + QK, n)
        block = f32[s:e]
        max_abs = float(np.max(np.abs(block))) if len(block) > 0 else 0.0
        scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
        qvals = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
        buf += struct.pack('<e', np.float16(scale))
        buf += qvals.tobytes()
        buf += b'\x00' * (QK - len(qvals))
    return bytes(buf)

def dequant_tensor(t):
    """Dequantize a GGUF tensor to float32."""
    raw = np.frombuffer(t.data, dtype=np.uint8)
    if t.tensor_type in [0, 30]:  # F32, BF16
        if t.tensor_type == 0:
            return np.frombuffer(raw, dtype=np.float32).copy()
        else:  # BF16
            bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            return (bits << 16).view(np.float32)
    else:  # Q4_K(12), Q6_K(14)
        blk_sz = {12: 144, 14: 210}[t.tensor_type]
        n_blocks = len(raw) // blk_sz
        arr_2d = raw[:n_blocks * blk_sz].reshape(n_blocks, blk_sz)
        return gguf.dequantize(arr_2d, t.tensor_type).ravel()

def should_transpose(name: str, shape) -> bool:
    """Check if weight needs transposition for the engine.
    
    Engine stores weights as [output_dim, input_dim] for W @ x compute.
    GGUF stores as [NE, dim] (first dim is input/hidden dimension).
    
    All 2D weights need transposition except:
    - 1D tensors (norms, scales)
    - Weights that happen to be square (dim == NE)
    """
    if len(shape) < 2:
        return False
    d0, d1 = int(shape[0]), int(shape[1])
    if d0 == d1:
        return False  # Square, no transposition needed
    # All Gemma 4 2D weights: GGUF stores as [NE, dim] 
    # Engine needs [dim, NE], so always transpose
    return True

def extract(gguf_path, out_dir, arch):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    
    r = gguf.GGUFReader(gguf_path)
    print(f"Loaded GGUF: {time.time()-t0:.1f}s", flush=True)
    
    tensors = list(r.tensors)
    print(f"  {len(tensors)} tensors", flush=True)
    
    for idx, t in enumerate(tensors):
        # Build filename from tensor name
        fname = t.name.replace('.', '_').replace('/', '_')
        if not fname.endswith('_weight.bin'):
            fname = fname + '_weight.bin'
        else:
            fname = fname.replace('.bin', '') + '.bin'
        # Handle ".weight" already in name
        if '_weight_weight.bin' in fname:
            fname = fname.replace('_weight_weight.bin', '_weight.bin')
        if 'weight_weight' in fname:
            fname = fname.replace('weight_weight', 'weight')
        
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            continue
        
        # Dequant to f32
        f32 = dequant_tensor(t)
        f32 = np.nan_to_num(np.array(f32, dtype=np.float32), nan=0.0, 
                            posinf=0.0, neginf=0.0)
        
        ndim = len(t.shape)
        
        if ndim >= 2 and should_transpose(t.name, t.shape):
            # Reshape and transpose from [NE, dim] to [dim, NE]
            d0, d1 = int(t.shape[0]), int(t.shape[1])
            f32_2d = f32.reshape(d0, d1)
            f32_2d = f32_2d.T  # Now [d1, d0]
            nr, nc = f32_2d.shape
            
            # Q8_0 quantize the transposed matrix
            buf = bytearray()
            for r_idx in range(nr):
                row = f32_2d[r_idx]
                buf += q8_quantize(row)
            q8_data = bytes(buf)
            print(f"  [{idx}/{len(tensors)}] {t.name}: {d0}×{d1} → {d1}×{d0} "
                  f"({len(q8_data)/1e6:.1f}MB)", flush=True)
        else:
            # 1D weight or square 2D: no transpose
            if ndim >= 2:
                nr, nc = int(t.shape[0]), int(t.shape[1])
                f32_2d = f32.reshape(nr, nc)
                buf = bytearray()
                for r_idx in range(nr):
                    buf += q8_quantize(f32_2d[r_idx])
                q8_data = bytes(buf)
                print(f"  [{idx}/{len(tensors)}] {t.name}: {nr}×{nc} "
                      f"({len(q8_data)/1e6:.1f}MB)", flush=True)
            else:
                # 1D tensor: norm weight
                q8_data = q8_quantize(f32)
                print(f"  [{idx}/{len(tensors)}] {t.name}: {len(f32)} "
                      f"({len(q8_data)/1e6:.1f}MB)", flush=True)
        
        with open(fpath, 'wb') as f:
            f.write(q8_data)
        
        if idx % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(tensors)}] ... ({elapsed:.0f}s)", flush=True)
    
    # Save arch metadata
    with open(os.path.join(out_dir, 'arch.json'), 'w') as f:
        import json
        json.dump(arch, f)
    
    print(f"\nDone: {len(tensors)} tensors in {time.time()-t0:.1f}s")
    print(f"Output: {out_dir}")

if __name__ == '__main__':
    model = sys.argv[1]
    extract(sys.argv[2], sys.argv[3], ARCH[model])
