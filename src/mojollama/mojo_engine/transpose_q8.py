#!/usr/bin/env python3
"""
Transpose Q8_0 weight files from GGUF orientation [d0, d1] to engine [d1, d0].
Usage: python3 transpose_q8.py <in_dir> <out_dir> <tensor_shapes.json>
  Where tensor_shapes.json maps filename -> [d0, d1]
"""
import os, sys, struct, json, numpy as np, time
from concurrent.futures import ProcessPoolExecutor, as_completed

QK, QB = 32, 34

def q8_read(fpath):
    """Read Q8_0 file, return (data, n_elements)."""
    with open(fpath, 'rb') as f:
        raw = f.read()
    n_blocks = len(raw) // QB
    n_elems = n_blocks * QK
    # Trim any trailing partial block (shouldn't happen for full files)
    return raw, n_elems

def q8_dequant_block(blk_bytes):
    """Dequantize a single Q8_0 block (QB=34 bytes) -> 32 float32 values."""
    scale = struct.unpack('<e', blk_bytes[:2])[0]
    quants = np.frombuffer(blk_bytes[2:], dtype=np.int8, count=QK)
    return quants.astype(np.float32) * scale

def dequant_q8(raw, n):
    """Dequantize entire Q8_0 file to float32."""
    n_blocks = len(raw) // QB
    # Process in chunks for efficiency
    f32 = np.zeros(n_blocks * QK, dtype=np.float32)
    for b in range(n_blocks):
        off = b * QB
        scale = struct.unpack('<e', raw[off:off+2])[0]
        quants = np.frombuffer(raw[off+2:off+QB], dtype=np.int8)
        f32[b*QK:(b+1)*QK] = quants.astype(np.float32) * scale
    return f32[:n]

def quant_q8_row(row):
    """Quantize a 1D float32 array to Q8_0 bytes."""
    n = len(row)
    n_blk = (n + QK - 1) // QK
    buf = bytearray()
    for b in range(n_blk):
        s = b * QK
        e = min(s + QK, n)
        block = row[s:e]
        max_abs = float(np.max(np.abs(block))) if len(block) > 0 else 0.0
        scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
        qvals = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
        buf += struct.pack('<e', np.float16(scale))
        buf += qvals.tobytes()
        buf += b'\x00' * (QK - len(qvals))
    return bytes(buf)

def transpose_file(in_path, out_path, d0, d1):
    """Transpose a Q8_0 weight matrix from [d0, d1] to [d1, d0]."""
    raw, n = q8_read(in_path)
    n_actual = d0 * d1
    # Dequantize
    f32 = dequant_q8(raw, n)
    if len(f32) != n_actual:
        print(f"    WARNING: {os.path.basename(in_path)}: expected {n_actual} elements, got {len(f32)}")
        # Trim or pad
        f32 = f32[:n_actual] if len(f32) > n_actual else np.pad(f32, (0, n_actual - len(f32)))
    # Reshape and transpose
    mat = f32.reshape(d0, d1).T  # Now [d1, d0]
    # Requantize to Q8_0
    buf = bytearray()
    for r in range(d1):
        buf += quant_q8_row(mat[r])
    with open(out_path, 'wb') as f:
        f.write(buf)
    return True

def main():
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    shape_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load shape mapping
    with open(shape_file) as f:
        shapes = json.load(f)  # dict: filename -> [d0, d1, need_transpose]
    
    t0 = time.time()
    n_total = len([v for v in shapes.values() if len(v) >= 3 and v[2]])
    n_done = 0
    
    for fname, info in sorted(shapes.items()):
        if len(info) < 3 or not info[2]:
            # No transpose needed, just copy
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)
            if os.path.exists(in_path) and not os.path.exists(out_path):
                with open(in_path, 'rb') as f:
                    data = f.read()
                with open(out_path, 'wb') as f:
                    f.write(data)
            continue
        
        d0, d1 = info[0], info[1]
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        
        if os.path.exists(out_path):
            continue  # Already done
        
        transpose_file(in_path, out_path, d0, d1)
        n_done += 1
        elapsed = time.time() - t0
        if n_done % 20 == 0:
            print(f"  [{n_done}/{n_total}] {fname} ({elapsed:.0f}s)", flush=True)
    
    # Copy remaining non-transposed files
    for fname in os.listdir(in_dir):
        if fname not in shapes:
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)
            if os.path.exists(in_path) and not os.path.exists(out_path):
                with open(in_path, 'rb') as f:
                    with open(out_path, 'wb') as f2:
                        f2.write(f.read())
    
    print(f"\nDone: {n_done} transposed in {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    main()
