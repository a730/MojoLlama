#!/usr/bin/env python3
"""Extract vocabulary from GGUF and create vocab.bin for the tokenizer."""
import gguf, numpy as np, struct, os, sys

def extract_vocab(gguf_path, out_path):
    r = gguf.GGUFReader(gguf_path)
    
    # Get tokenizer data from metadata
    tokens_field = r.fields['tokenizer.ggml.tokens']
    scores_field = r.fields['tokenizer.ggml.scores']
    
    # Parse token strings from the string table
    parts = tokens_field.parts
    data = tokens_field.data
    
    # Reconstruct token strings using offsets
    tokens = []
    for i in range(len(parts)):
        offset = int(parts[i][0])
        if i + 1 < len(parts):
            next_offset = int(parts[i+1][0])
        else:
            next_offset = len(data)
        token_bytes = data[offset:next_offset].tobytes()
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
        except:
            token_str = str(token_bytes)
        tokens.append(token_str)
    
    # Get scores
    scores = np.frombuffer(scores_field.data, dtype=np.float32).copy()
    
    # Get special token IDs
    bos = int(r.fields['tokenizer.ggml.bos_token_id'].parts[0][0])
    eos = int(r.fields['tokenizer.ggml.eos_token_id'].parts[0][0])
    
    print(f"Vocabulary: {len(tokens)} tokens")
    print(f"BOS={bos}, EOS={eos}")
    print(f"Sample tokens: {tokens[:5]}, ... {tokens[-5:]}")
    
    # Write binary format: [num_tokens][len][bytes][score_f32]...
    buf = bytearray()
    buf += struct.pack('<i', len(tokens))
    for i in range(min(len(tokens), len(scores))):
        tok = tokens[i].encode('utf-8') if isinstance(tokens[i], str) else tokens[i].encode('utf-8', errors='replace')
        buf += struct.pack('<i', len(tok))
        buf += tok
        buf += struct.pack('<f', float(scores[i]))
    
    with open(out_path, 'wb') as f:
        f.write(buf)
    
    print(f"Written: {out_path} ({len(buf)} bytes, {len(tokens)} tokens)")
    
    # Also write special token IDs
    special_path = os.path.join(os.path.dirname(out_path), 'special_tokens.txt')
    with open(special_path, 'w') as f:
        f.write(f"{bos} {eos}\n")
    print(f"Special tokens: {special_path}")
    
    return len(tokens)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: extract_vocab.py <gguf_path> <out_dir>")
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'vocab.bin')
    
    extract_vocab(gguf_path, out_path)
