#!/usr/bin/env python3
"""One-pass Gemma 4 weight extractor: GGUF → Q8_0 directly."""
import gguf, numpy as np, os, struct, sys, time, json

QK, QB = 32, 34

ARCH = {
    'e2b':  {'NE': 2048, 'FF': 16384, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 4,  'HD': 128},
    'e4b':  {'NE': 2560, 'FF': 10240, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 4,  'HD': 128},
    '26b':  {'NE': 2816, 'FF': 2112,  'NV': 262144, 'NL': 30, 'NH': 16, 'NK': 8,  'HD': 128, 'n_exp': 128, 'n_act': 4},
    '31b':  {'NE': 3072, 'FF': 24576, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 8,  'HD': 128},
}

def extract(gguf_path, out_dir, arch):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    
    r = gguf.GGUFReader(gguf_path)
    print(f"Loaded GGUF: {time.time()-t0:.1f}s")
    
    # Build tensor list
    tensors = list(r.tensors)
    print(f"  {len(tensors)} tensors")
    
    def dequant_to_q8(t):
        """Dequantize tensor and convert to Q8_0 bytes."""
        raw = np.frombuffer(t.data, dtype=np.uint8)
        if t.tensor_type in [0, 30]:  # F32, BF16
            if t.tensor_type == 0:
                f32 = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                f32 = (bits << 16).view(np.float32)
        else:  # Q4_K(12), Q6_K(14)
            # Need to reshape for dequantize
            blk_sz = {12: 144, 14: 210}[t.tensor_type]
            n_blocks = len(raw) // blk_sz
            arr_2d = raw[:n_blocks*blk_sz].reshape(n_blocks, blk_sz)
            f32 = gguf.dequantize(arr_2d, t.tensor_type).ravel()
        
        f32 = np.nan_to_num(np.array(f32, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Q8_0 quantize
        nr, nc = t.shape[0], np.prod(t.shape[1:]) if len(t.shape) > 1 else 1
        if len(f32) != nr * nc:
            nc = len(f32) // nr
        f32 = f32[:nr*nc].reshape(nr, nc)
        
        buf = bytearray()
        for r_idx in range(nr):
            row = f32[r_idx]
            for blk in range((nc + QK - 1) // QK):
                s = blk * QK; e = min(s + QK, nc)
                block = row[s:e]
                max_abs = float(np.max(np.abs(block))) if len(block) > 0 else 0.0
                scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
                qvals = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
                buf += struct.pack('<e', np.float16(scale))
                buf += qvals.tobytes()
                # Pad to QK
                buf += b'\x00' * (QK - len(qvals))
        return bytes(buf)
    
    for idx, t in enumerate(tensors):
        # Build filename
        fname = t.name.replace('.', '_').replace('/', '_') + '_weight.bin'
        # Handle Gemma 4 naming: some tensors are already "xxx.weight"
        if t.name.endswith('.weight'):
            fname = t.name.replace('.', '_').replace('/', '_') + '.bin'
        
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath): continue
        
        q8 = dequant_to_q8(t)
        with open(fpath, 'wb') as f: f.write(q8)
        
        if idx % 50 == 0:
            sz_mb = len(q8) / 1e6
            print(f"  [{idx}/{len(tensors)}] {fname} ({sz_mb:.0f}MB, {time.time()-t0:.0f}s)")
    
    with open(os.path.join(out_dir, 'arch.json'), 'w') as f:
        json.dump(arch, f)
    
    print(f"\nDone: {len(tensors)} tensors in {time.time()-t0:.0f}s")

if __name__ == '__main__':
    model = sys.argv[1]
    extract(sys.argv[2], sys.argv[3], ARCH[model])
