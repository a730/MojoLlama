"""
weights_to_bin.py — Convert GGUF model to MojoLlama binary weight file.

Usage: python weights_to_bin.py model.gguf model.bin

Binary format:
  Header (64 bytes):
    [0-3]   magic: "MLMF" (0x4d4c4d46)
    [4-7]   num_tensors: int32
    [8-11]  n_layers: int32
    [12-15] n_embd: int32
    [16-19] n_head: int32
    [20-23] n_kv_head: int32
    [24-27] n_ff: int32
    [28-31] n_vocab: int32
    [32-35] head_dim: int32
    [36-63] reserved (28 bytes)

  Tensor entries (sequential):
    [0-3]   name_len: int32
    [4..]   name: name_len bytes
    [after name, padded to 4B] dtype: uint8 (0=f32, 1=q4_0)
    [after dtype] ndim: int32
    [after ndim] shape: ndim * int32
    [after shape] data_size: int32
    [after data_size] data: data_size bytes

  Q4_0 convention: stored as raw GGUF block bytes.
  Shape in file = logical [n_rows, n_cols]; bytes = n_rows * (n_cols/32) * 18.
  Mojo computes blocks-per-row = n_cols // 32.
"""

import struct
import sys
import numpy as np
import gguf
from gguf.constants import GGMLQuantizationType


def write_weights(gguf_path: str, output_path: str):
    reader = gguf.GGUFReader(gguf_path)
    f = reader.get_field
    builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    print_fn = builtins.get('print', print)

    # Detect architecture prefix
    arch_field = f('general.architecture')
    if arch_field is None:
        print_fn("WARNING: no general.architecture, trying 'llama' prefix")
        prefix = "llama."
        arch_str = "llama"
    else:
        arch_bytes = arch_field.parts[-1]
        if isinstance(arch_bytes, bytes):
            arch_str = arch_bytes.decode('utf-8', errors='replace').strip('\x00')
        elif hasattr(arch_bytes, 'tobytes'):
            arch_str = arch_bytes.tobytes().decode('utf-8', errors='replace').strip('\x00')
        else:
            arch_str = str(arch_bytes)
        prefix = arch_str + '.'
        print_fn(f"Architecture: {arch_str}")

    def get_scalar(field):
        if field is None:
            return 0
        v = field.parts[-1]
        if hasattr(v, 'item'):
            return int(v.item())
        return int(v)
    
    # Retry with fallback if all zeros
    def get_scalar_with_fallback(key, fallback_key=None):
        val = get_scalar(f(f'{prefix}{key}'))
        if val == 0 and fallback_key:
            val = get_scalar(f(fallback_key))
        if val == 0:
            # Try without prefix
            val = get_scalar(f(key))
        return val

    n_layers = get_scalar(f(f'{prefix}block_count'))
    n_embd = get_scalar(f(f'{prefix}embedding_length'))
    n_head = get_scalar(f(f'{prefix}attention.head_count'))
    n_kv_head = get_scalar(f(f'{prefix}attention.head_count_kv'))
    n_ff = get_scalar(f(f'{prefix}feed_forward_length'))
    head_dim = n_embd // n_head

    # Get vocab size from tokenizer
    token_field = f('tokenizer.ggml.tokens')
    if token_field is not None:
        parts = list(token_field.parts)
        n_vocab = int(parts[4].item()) if len(parts) > 4 else 0
    else:
        n_vocab = get_scalar(f(f'{prefix}vocab_size'))

    print_fn(f"Architecture: {arch_str}")
    print_fn(f"  Layers: {n_layers}, Embed: {n_embd}, Heads: {n_head}, KV: {n_kv_head}")
    print_fn(f"  FFN: {n_ff}, Vocab: {n_vocab}, Head_dim: {head_dim}")

    # Collect tensors with their raw data
    # We want: per-layer attn_norm, ffn_norm, wq, wk, wv, wo, wgate, wup, wdown
    # Also: token_embd, output_norm, output_weight
    target_names = set()
    target_names.add('token_embd.weight')
    target_names.add('output_norm.weight')
    target_names.add('output.weight')
    for i in range(n_layers):
        target_names.add(f'blk.{i}.attn_norm.weight')
        target_names.add(f'blk.{i}.ffn_norm.weight')
        target_names.add(f'blk.{i}.attn_q.weight')
        target_names.add(f'blk.{i}.attn_k.weight')
        target_names.add(f'blk.{i}.attn_v.weight')
        target_names.add(f'blk.{i}.attn_output.weight')
        target_names.add(f'blk.{i}.ffn_gate.weight')
        target_names.add(f'blk.{i}.ffn_up.weight')
        target_names.add(f'blk.{i}.ffn_down.weight')

    with open(output_path, 'wb') as fout:
        # Reserve header (64 bytes)
        header_start = fout.tell()
        fout.write(b'\x00' * 64)

        total_tensors = 0
        for t in reader.tensors:
            name = t.name
            if name not in target_names:
                continue
            total_tensors += 1

            raw_data = np.array(t.data)
            tensor_type = t.tensor_type if hasattr(t, 'tensor_type') and t.tensor_type is not None else GGMLQuantizationType.F32

            if tensor_type == GGMLQuantizationType.Q4_0:
                dtype_code = 1  # q4_0
                # Raw shape: [n_rows, (n_cols/32)*18]
                raw_shape = raw_data.shape
                n_rows = int(raw_shape[0])
                raw_bytes_per_row = int(raw_shape[1]) if len(raw_shape) > 1 else 0
                n_cols = (raw_bytes_per_row // 18) * 32
                logical_shape = [n_rows, n_cols]
                data_bytes = raw_data.tobytes()
            elif tensor_type == GGMLQuantizationType.F32:
                dtype_code = 0
                logical_shape = list(raw_data.shape) if raw_data.ndim > 0 else [1]
                data_bytes = raw_data.tobytes()
            elif tensor_type == GGMLQuantizationType.F16:
                dtype_code = 0
                f32_data = raw_data.astype(np.float32)
                logical_shape = list(f32_data.shape)
                data_bytes = f32_data.tobytes()
            else:
                # Try dequantizing
                try:
                    f32_data = gguf.dequantize(raw_data, tensor_type)
                except Exception:
                    f32_data = np.array(raw_data, dtype=np.float32)
                dtype_code = 0
                logical_shape = list(f32_data.shape)
                data_bytes = f32_data.tobytes()

            name_bytes = name.encode('utf-8')
            name_padded = (len(name_bytes) + 3) & ~3
            padding = name_padded - len(name_bytes)
            write_buf = bytearray()
            write_buf += struct.pack('<I', len(name_bytes))
            write_buf += name_bytes
            write_buf += b'\x00' * padding
            write_buf += struct.pack('<B', dtype_code)
            write_buf += b'\x00\x00\x00'  # padding after dtype
            write_buf += struct.pack('<I', len(logical_shape))
            for d in logical_shape:
                write_buf += struct.pack('<I', d)
            write_buf += struct.pack('<I', len(data_bytes))
            fout.write(write_buf)
            fout.write(data_bytes)

            print_fn(f"  [{name}] dtype={'q4_0' if dtype_code else 'f32'} shape={logical_shape} bytes={len(data_bytes)}")

        # Go back and write header
        file_size = fout.tell()
        fout.seek(header_start)
        fout.write(struct.pack('<4s', b'MLMF'))
        fout.write(struct.pack('<I', total_tensors))
        fout.write(struct.pack('<I', n_layers))
        fout.write(struct.pack('<I', n_embd))
        fout.write(struct.pack('<I', n_head))
        fout.write(struct.pack('<I', n_kv_head))
        fout.write(struct.pack('<I', n_ff))
        fout.write(struct.pack('<I', n_vocab))
        fout.write(struct.pack('<I', head_dim))
        remaining = 64 - 4 - 4 - 7 * 4
        fout.write(b'\x00' * remaining)

        assert fout.tell() == header_start + 64, f"Header size mismatch: {fout.tell() - header_start}"

    print_fn(f"\nWrote {total_tensors} tensors to {output_path}")
    print_fn(f"  File size: {file_size / 1024**3:.2f} GB")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python weights_to_bin.py model.gguf output.bin")
        sys.exit(1)
    write_weights(sys.argv[1], sys.argv[2])
