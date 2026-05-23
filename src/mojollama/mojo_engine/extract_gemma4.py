#!/usr/bin/env python3
"""Extract Gemma 4 weights from GGUF to Q8_0 .bin files for Mojo engine."""
import struct, sys, os, json
import gguf
import numpy as np

MODELS = {
    'gemma-4-E2B':  {'NE': 2048, 'NH': 8,  'NK': 4,  'HD': 128, 'NL': 18, 'FF': 16384, 'NV': 262144},
    'gemma-4-E4B':  {'NE': 2560, 'NH': 8,  'NK': 4,  'HD': 128, 'NL': 18, 'FF': 10240, 'NV': 262144},
    'gemma-4-26B':  {'NE': 2816, 'NH': 16, 'NK': 8,  'HD': 128, 'NL': 18, 'FF': 2112,  'NV': 262144, 'n_exp': 128, 'n_act': 4},
    'gemma-4-31B':  {'NE': 3072, 'NH': 16, 'NK': 8,  'HD': 128, 'NL': 18, 'FF': 24576, 'NV': 262144},
}

def extract_gguf(gguf_path, out_dir, arch):
    """Extract all weights as Q8_0 .bin files."""
    os.makedirs(out_dir, exist_ok=True)
    r = gguf.GGUFReader(gguf_path)
    
    NE = arch['NE']; NH = arch['NH']; NK = arch['NK']
    HD = arch['HD']; NL = arch['NL']; FF = arch['FF']
    QK = 32; QB = 34  # Q8_0 block
    
    def q8_rb(nc): return ((nc + QK - 1) // QK) * QB
    
    def save_q8(name, f32_data):
        """Convert f32 array to Q8_0 .bin file."""
        nr, nc = f32_data.shape
        rb = q8_rb(nc)
        buf = bytearray()
        for r in range(nr):
            for blk in range((nc + QK - 1) // QK):
                start = blk * QK
                end = min(start + QK, nc)
                block = f32_data[r, start:end]
                max_abs = max(abs(block)) if len(block) > 0 else 0.0
                scale = max_abs / 127.0 if max_abs > 1e-10 else 1.0
                # f16 scale
                import struct as st
                buf += st.pack('<e', scale)
                for i in range(QK):
                    if i < len(block):
                        qv = int(round(block[i] / scale))
                        if qv > 127: qv = 127
                        if qv < -128: qv = -128
                        buf += bytes([qv & 0xFF])
                    else:
                        buf += bytes([0])
        with open(f'{out_dir}/{name}.bin', 'wb') as f:
            f.write(buf)
        return len(buf)
    
    print(f"Extracting {arch['name']} to {out_dir}/")
    
    # Global weights
    for t in r.tensors:
        name = t.name.replace('.', '_')
        data = t.data
        # Dequantize to f32
        if data.dtype == np.float16:
            f32 = data.astype(np.float32)
        elif data.dtype in [np.float32, np.float64]:
            f32 = np.array(data, dtype=np.float32)
        else:
            # Quantized — use gguf.dequantize
            try:
                f32 = gguf.dequantize(data, t.data.dtype)
            except:
                continue
        
        # Determine shape
        if len(f32.shape) == 1:
            f32 = f32.reshape(1, -1)
        
        print(f"  {t.name}: {f32.shape} -> {out_dir}/{name}.bin")
        save_q8(name, f32)
    
    # Save arch config
    with open(f'{out_dir}/arch.json', 'w') as f:
        json.dump(arch, f)
    
    total_mb = sum(os.path.getsize(f'{out_dir}/{f}') for f in os.listdir(out_dir) if f.endswith('.bin')) / 1e6
    print(f"  Total: {total_mb:.0f} MB")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(MODELS.keys()))
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--out', default='/tmp/weights_gemma')
    args = parser.parse_args()
    
    arch = MODELS[args.model]
    arch['name'] = args.model
    extract_gguf(args.gguf, args.out, arch)
