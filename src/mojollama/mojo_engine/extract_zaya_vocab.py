#!/usr/bin/env python3
"""
Extract ZAYA1-8B vocabulary from GGUF to flat .bin format.
Output: /tmp/vocab_zaya.bin

Uses low-level GGUF parsing to get tokens.
"""

import sys, os, struct
import numpy as np
sys.path.insert(0, '/onedev-workspace/work/src/mojollama/kernels')
import gguf

GGUF_PATH = '/tmp/models/ZAYA1-8B-Q8_0.gguf'
OUT_PATH = '/tmp/vocab_zaya.bin'


def main():
    if not os.path.exists(GGUF_PATH):
        print(f"ERROR: GGUF not found: {GGUF_PATH}")
        sys.exit(1)

    reader = gguf.GGUFReader(GGUF_PATH)

    # Find tokenizer.ggml.tokens field
    field = None
    for name, f in reader.fields.items():
        if name == 'tokenizer.ggml.tokens':
            field = f
            break

    if field is None:
        print("ERROR: tokenizer.ggml.tokens not found")
        sys.exit(1)

    # field.parts contains: [type_meta, key_name_bytes, ...tokens_data]
    # The token data is interleaved as: [uint64_count][bytes...][uint64_count][bytes...]...
    # Each token's data is preceded by a uint64 length indicator
    tokens = []

    # Walk through parts looking for string data
    i = 0
    parts = field.parts
    while i < len(parts):
        p = parts[i]
        # Check if this is a size indicator (uint64 with small value like string length)
        if hasattr(p, 'dtype') and p.dtype == np.uint64 and p.shape == (1,):
            length = int(p[0])
            if length > 0 and length < 1000 and i + 1 < len(parts):
                next_p = parts[i + 1]
                if hasattr(next_p, 'dtype') and next_p.dtype == np.uint8 and next_p.shape == (length,):
                    token_bytes = bytes(next_p)
                    tokens.append(token_bytes)
                    i += 2
                    continue
        i += 1

    # If no tokens found via parsing, try to read the raw tensor data
    if len(tokens) == 0:
        print("Trying direct token extraction from field...")
        # The field data might be a list of integers representing byte data
        if hasattr(field, 'data') and isinstance(field.data, list):
            raw_data = bytes(field.data)
            # Try to parse as null-terminated strings
            offset = 0
            while offset < len(raw_data):
                # Skip any non-printable or length-like prefix
                end = raw_data.find(b'\x00', offset)
                if end < 0:
                    break
                token = raw_data[offset:end]
                if len(token) > 0 and len(token) < 500:
                    tokens.append(token)
                offset = end + 1

    if len(tokens) == 0:
        # Last resort: check tokenizer.ggml.scores field — it should have same count
        print("Trying alternative extraction...")
        # Check all fields with 'token' in name
        for name, f in reader.fields.items():
            if 'token' in name.lower():
                print(f"  {name}: parts={len(f.parts)}, types={f.types}")

        # Try to extract from raw GGUF file using gguf parser internals
        try:
            # The tokens might be at a specific offset
            token_list = reader.get_field("tokenizer.ggml.tokens")
            if token_list:
                print(f"token_list type: {type(token_list)}")
        except:
            pass

        sys.exit(1)

    print(f"Found {len(tokens)} tokens")
    if len(tokens) > 0:
        print(f"First token ({len(tokens[0])} bytes): {tokens[0][:40]}")
        if len(tokens) > 1:
            print(f"Second token ({len(tokens[1])} bytes): {tokens[1][:40]}")
        print(f"Last token ({len(tokens[-1])} bytes): {tokens[-1][:40]}")

    # Skip the first entry if it's the metadata key name
    # token_embd.weight has NV=262147 rows, so skip 1 metadata entry
    if tokens and tokens[0].startswith(b'tokenizer'):
        print(f"Skipping metadata entry: {tokens[0]}")
        tokens = tokens[1:]

    print(f"Clean vocab: {len(tokens)} tokens")

    # Build flat binary format
    nv = len(tokens)
    buf = bytearray()
    buf += struct.pack('<I', nv)

    for i, token_bytes in enumerate(tokens):
        buf += struct.pack('<I', len(token_bytes))
        buf += token_bytes
        buf += struct.pack('<I', i)

    with open(OUT_PATH, 'wb') as f:
        f.write(bytes(buf))

    print(f"Wrote {OUT_PATH}: {len(buf)} bytes ({nv} tokens)")
    print("Done!")


if __name__ == '__main__':
    main()
