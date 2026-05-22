#!/usr/bin/env python3
"""Scan ZAYA vocab for special tokens and test encoding."""
import struct

with open('/tmp/vocab_zaya.bin', 'rb') as f:
    data = f.read()

nv = struct.unpack('<I', data[0:4])[0]
print(f'Vocab size: {nv}')

# Check first 15 tokens
print('\nFirst 15 tokens:')
off = 4
for i in range(min(15, nv)):
    plen = struct.unpack('<I', data[off:off+4])[0]
    off += 4
    tok_data = data[off:off+plen]
    off += plen
    tid = struct.unpack('<I', data[off:off+4])[0]
    off += 4
    txt = tok_data.decode('utf-8', errors='replace')
    print(f'  {tid}: {repr(txt)}')

# Search special tokens
print('\nSpecial tokens:')
keywords = ['<s>', '</s>', '<pad>', '<unk>', '<|im_end', '<|start', '<|user', 
            '<|assistant', '<|system', '<|begin', '[INST', '[/INST]']
off = 4
found = {}
for i in range(nv):
    plen = struct.unpack('<I', data[off:off+4])[0]
    off += 4
    tok_data = data[off:off+plen]
    off += plen
    tid = struct.unpack('<I', data[off:off+4])[0]
    off += 4
    txt = tok_data.decode('utf-8', errors='replace')
    for kw in keywords:
        if kw in txt:
            if kw not in found:
                found[kw] = []
            found[kw].append((tid, txt))

for kw, entries in sorted(found.items()):
    for tid, txt in entries[:3]:
        print(f'  tok {tid}: {repr(txt)}')

# Check last 5 tokens
print('\nLast 5 tokens:')
off = 4
for i in range(nv - 5, nv):
    # Quick seek by re-scanning
    pass

# Better: sequential scan from end
offsets = []
off = 4
for i in range(nv):
    plen = struct.unpack('<I', data[off:off+4])[0]
    offsets.append((off, plen))
    off += 4 + plen + 4

for i in range(max(0, nv-5), nv):
    off, plen = offsets[i]
    tok_data = data[off+4:off+4+plen]
    tid = struct.unpack('<I', data[off+4+plen:off+4+plen+4])[0]
    txt = tok_data.decode('utf-8', errors='replace')
    print(f'  {tid}: {repr(txt)}')
