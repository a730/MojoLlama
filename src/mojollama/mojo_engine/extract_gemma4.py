#!/usr/bin/env python3
"""Fast Gemma 4 weight extractor. Dequantizes Q4_K/Q6_K/F32 to Q8_0 .bin files."""
import gguf, numpy as np, os, struct, json, sys, time

ARCH = {
    'e2b':  {'NE': 2048, 'FF': 16384, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 4,  'HD': 128},
    'e4b':  {'NE': 2560, 'FF': 10240, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 4,  'HD': 128},
    '26b':  {'NE': 2816, 'FF': 2112,  'NV': 262144, 'NL': 30, 'NH': 16, 'NK': 8,  'HD': 128, 'n_exp': 128, 'n_act': 4},
    '31b':  {'NE': 3072, 'FF': 24576, 'NV': 262144, 'NL': 42, 'NH': 16, 'NK': 8,  'HD': 128},
}
QK, QB = 32, 34

def q8_rb(nc): return ((nc + QK - 1) // QK) * QB

def f32_to_q8(f32):
    """Convert f32 array to Q8_0 bytes."""
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

def extract(gguf_path, out_dir, arch, max_layers=None):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    r = gguf.GGUFReader(gguf_path)
    print(f"GGUF loaded: {time.time()-t0:.1f}s")
    
    # Build tensor lookup by name
    tensor_map = {}
    for t in r.tensors:
        tensor_map[t.name] = t
    
    def get_f32(name):
        t = tensor_map.get(name)
        if t is None: return None
        raw = np.frombuffer(t.data, dtype=np.uint8)
        try:
            result = gguf.dequantize(raw, int(t.tensor_type))
        except:
            result = np.frombuffer(t.data, dtype=np.float32)
        if len(result.shape) == 1 and result.size > 0:
            result = result.reshape(t.shape)
        return np.array(result, dtype=np.float32)
    
    total_mb = 0
    NE = arch['NE']; FF = arch['FF']; NV = arch['NV']; NL = arch['NL']
    
    def save(name, f32):
        nonlocal total_mb
        q8_bytes = f32_to_q8(f32)
        path = f'{out_dir}/{name}'
        with open(path, 'wb') as f: f.write(q8_bytes)
        mb = len(q8_bytes) / 1e6
        total_mb += mb
        print(f"  {name}: {f32.shape} -> {mb:.0f}MB")
    
    n_layers = min(NL, max_layers) if max_layers else NL
    
    # Global weights
    for gname in ['token_embd', 'output_norm']:
        f32 = get_f32(f'{gname}.weight')
        if f32 is not None: save(f'{gname}_weight.bin', f32)
    
    # Per-layer weights
    for l in range(n_layers):
        wi = 4096 if (l % 6 == 5) else (NE if NE == 2816 else 2048)
        wk = 1024 if (l % 6 == 5) else (NK * HD)
        wv = wk
        
        tnames = [
            ('attn_norm', [NE]),
            ('attn_q', [NE, wi]),
            ('attn_k', [NE, wk]),
            ('attn_v', [NE, wv]),
            ('attn_q_norm', [256]),
            ('attn_k_norm', [256]),
            ('attn_output', [wi, NE]),
            ('post_attention_norm', [NE]),
            ('post_ffw_norm', [NE]),
            ('post_norm', [NE]),
            ('inp_gate', [NE, 256]),
            ('proj', [256, NE]),
            ('ffn_norm', [NE]),
            ('ffn_gate', [NE, FF]),
            ('ffn_up', [NE, FF]),
            ('ffn_down', [FF, NE]),
            ('layer_output_scale', [1]),
        ]
        for tn, _ in tnames:
            f32 = get_f32(f'blk.{l}.{tn}.weight')
            if f32 is not None:
                save(f'blk_{l}_{tn}_weight.bin', f32)
        
        if l % 10 == 0 or l == n_layers - 1:
            print(f"  layer {l+1}/{n_layers} ({time.time()-t0:.1f}s)")
    
    with open(f'{out_dir}/arch.json', 'w') as f:
        json.dump(arch, f)
    print(f"Total: {total_mb:.0f}MB in {time.time()-t0:.1f}s")

if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'e4b'
    gguf_path = sys.argv[2] if len(sys.argv) > 2 else f'/tmp/models/gemma-4-E4B-it-Q4_K_M.gguf'
    out_dir = sys.argv[3] if len(sys.argv) > 3 else f'/tmp/weights_gemma4_{model}/'
    max_layers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    extract(gguf_path, out_dir, ARCH[model], max_layers)
