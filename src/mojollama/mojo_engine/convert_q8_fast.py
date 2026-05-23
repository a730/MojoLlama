#!/usr/bin/env python3
"""Fast converter: C-extracted weights → Q8_0. Handles Q4_K/Q6_K/F32/BF16 directly."""
import os, struct, numpy as np, sys, time

QK, QB = 32, 34

def q8_rb(nc): return ((nc + QK - 1) // QK) * QB

def to_q8(f32):
    """f32 array → Q8_0 bytes using vectorized ops."""
    if len(f32.shape) == 1: f32 = f32.reshape(1, -1)
    nr, nc = f32.shape
    nblk = (nc + QK - 1) // QK
    buf = bytearray()
    for r in range(nr):
        row = f32[r]
        for b in range(nblk):
            s = b * QK; e = min(s + QK, nc)
            blk = row[s:e]
            max_abs = float(np.max(np.abs(blk))) if len(blk) > 0 else 0.0
            scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
            qvals = np.clip(np.round(blk / scale), -128, 127).astype(np.int8)
            # Ensure we always write QK values
            padded = np.zeros(QK, dtype=np.int8)
            padded[:len(qvals)] = qvals
            buf += struct.pack('<e', np.float16(scale))
            buf += padded.tobytes()
    return bytes(buf)

def dequant_q4k(raw, nr, nc):
    """Dequant Q4_K → f32. Block: 144 bytes → 256 values."""
    nblk = (nc + 255) // 256
    result = np.zeros((nr, nc), dtype=np.float32)
    for r in range(nr):
        for b in range(nblk):
            off = (r * nblk + b) * 144
            if off + 144 > len(raw): break
            block = raw[off:off+144]
            d_hi = struct.unpack('<e', block[0:2])[0]
            d_lo = struct.unpack('<e', block[2:4])[0]
            m_hi = struct.unpack('<e', block[4:6])[0]
            m_lo = struct.unpack('<e', block[6:8])[0]
            q = np.frombuffer(block[8:136], dtype=np.uint8)
            lo = (q & 0x0F).astype(np.float32) - 8.0
            hi = ((q >> 4) & 0x0F).astype(np.float32) - 8.0
            c = b * 256
            n = min(128, nc - c)
            result[r, c:c+n] = lo[:n] * d_lo + m_lo
            result[r, c+128:c+128+n] = hi[:n] * d_hi + m_hi
    return result

def dequant_q6k(raw, nr, nc):
    """Dequant Q6_K → f32. Block: 240 bytes → 256 values."""
    nblk = (nc + 255) // 256
    result = np.zeros((nr, nc), dtype=np.float32)
    for r in range(nr):
        for b in range(nblk):
            off = (r * nblk + b) * 240
            if off + 240 > len(raw): break
            block = raw[off:off+240]
            d_hi = struct.unpack('<e', block[0:2])[0]
            d_lo = struct.unpack('<e', block[2:4])[0]
            # Q6_K stores 256 values in 192 nibbles (6-bit each) = 192 bytes
            # blocks[4:8] = d_hi/d_lo (f16), blocks[8:200] = 192 bytes = 256*6bit values
            n = min(128, nc - b * 256)
            # Simplified: read as 6-bit packed
            r0 = block[8:200]
            q = np.unpackbits(np.frombuffer(r0, dtype=np.uint8))
            # 6-bit values: take first 768 bits (128*6) for lo, next 768 for hi
            lo_bits = q[:n*6].reshape(n, 6)
            hi_bits = q[768:768+n*6].reshape(n, 6)
            lo_val = np.zeros(n, dtype=np.float32)
            hi_val = np.zeros(n, dtype=np.float32)
            for i in range(6):
                lo_val += lo_bits[:, i].astype(np.float32) * (1 << i)
                hi_val += hi_bits[:, i].astype(np.float32) * (1 << i)
            lo_val -= 32.0; hi_val -= 32.0
            result[r, b*256:b*256+n] = lo_val * d_lo
            result[r, b*256+128:b*256+128+n] = hi_val * d_hi
    return result

def convert_single(in_path, out_path):
    raw = np.frombuffer(open(in_path, 'rb').read(), dtype=np.uint8)
    size = len(raw)
    
    # Determine format from size patterns
    f32 = None
    # Try each type until we get a valid shape
    for qtype, blk_sz, vals_per_blk, fn in [
        ('f32', 0, 0, lambda r: np.frombuffer(r, dtype=np.float32).copy()),
        ('bf16', 0, 0, lambda r: (np.frombuffer(r, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)),
    ]:
        try:
            f32 = fn(raw)
            if np.all(np.isfinite(f32)):
                break
        except: pass
    
    if f32 is None:
        # Try Q4_K: size should be multiple of 144
        if size % 144 == 0:
            nblk = size // 144
            nvals = nblk * 256
            # Estimate nr, nc from nvals
            nr = 1
            while nr * nr < nvals: nr *= 2
            nc = nvals // nr
            f32 = dequant_q4k(raw, nr, nc)
    
    if f32 is None and size % 240 == 0:
        nblk = size // 240
        nvals = nblk * 256
        nr = int(np.sqrt(nvals))
        nc = nvals // nr
        f32 = dequant_q6k(raw, nr, nc)
    
    if f32 is None:
        print(f"  SKIP {os.path.basename(in_path)}: unknown format ({size}B)")
        return False
    
    f32 = np.nan_to_num(f32, nan=0.0, posinf=0.0, neginf=0.0)
    q8 = to_q8(f32)
    with open(out_path, 'wb') as f: f.write(q8)
    return True

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    
    t0 = time.time()
    files = sorted([f for f in os.listdir(in_dir) if f.endswith('.bin') and f.count('\n') == 0 and all(c.isascii() for c in f)])
    done = 0
    for fname in files:
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path): continue
        if convert_single(in_path, out_path):
            done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(files)} ({time.time()-t0:.0f}s)")
    
    print(f"Done: {done}/{len(files)} files in {time.time()-t0:.0f}s")
