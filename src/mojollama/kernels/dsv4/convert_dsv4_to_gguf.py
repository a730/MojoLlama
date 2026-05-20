#!/usr/bin/env python3
"""
Convert Intel/DeepSeek-V4-Flash-W4A16-AutoRound to MojoLlama GGUF.

Streaming converter: processes one tensor at a time, downloads shards progressively.
Dequantizes W4A16 (AutoRound/GPTQ) → FP32 → re-quantizes to Q8_0 in GGUF.

Usage:
    python3 convert_dsv4_to_gguf.py /tmp/models/deepseek-v4-flash-w4a16.gguf
"""
import os, sys, json, time, math, struct, gc
import numpy as np
from huggingface_hub import hf_hub_download, get_safetensors_metadata
from safetensors import safe_open

# ─── GGUF constants ───
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF key types
GGUF_TYPE_UINT8 = 0; GGUF_TYPE_INT8 = 1; GGUF_TYPE_UINT16 = 2; GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4; GGUF_TYPE_INT32 = 5; GGUF_TYPE_FLOAT32 = 6; GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8; GGUF_TYPE_ARRAY = 9; GGUF_TYPE_UINT64 = 10; GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Quant types
GGML_TYPE_F32 = 0; GGML_TYPE_I32 = 26; GGML_TYPE_Q8_0 = 8

QK8_0 = 32

def pack_q8_0(arr_f32):
    """Convert F32 block to Q8_0: scale (f16) + 32 int8 values."""
    n = arr_f32.shape[0]
    assert n == QK8_0
    amax = np.max(np.abs(arr_f32))
    if amax == 0:
        scale = 0.0
        d_i32 = np.zeros(n, dtype=np.int8)
    else:
        scale = amax / 127.0
        d_i32 = np.clip(np.round(arr_f32 / scale), -128, 127).astype(np.int8)
    # f16 scale
    scale_f16 = np.float32(scale).view(np.uint16)
    return scale_f16, d_i32

def quantize_q8_0(data):
    """Quantize [N] F32 → Q8_0 bytes. N must be multiple of 32."""
    N = data.shape[0]
    if N % QK8_0 != 0:
        pad = QK8_0 - (N % QK8_0)
        data = np.pad(data, (0, pad))
    blocks = data.reshape(-1, QK8_0)
    result = bytearray()
    for block in blocks:
        f16_scale, i8_vals = pack_q8_0(block)
        result += struct.pack('<e', f16_scale)  # f16
        result += i8_vals.tobytes()
    return bytes(result), data.shape[0]  # (bytes, original_N)

class DequantizeW4A16:
    """Dequantize AutoRound/GPTQ W4A16 packed weights → FP32."""
    @staticmethod
    def dequantize(qweight, qzeros, scales, group_size=128):
        """
        qweight: [in//8, out] int32, LSB-first packed
        qzeros:  [in//g, out//8] int32, LSB-first packed
        scales:  [in//g, out] bf16
        returns: [out, in] float32
        """
        in_packed, out_features = qweight.shape
        in_features = in_packed * 8
        n_groups = scales.shape[0]
        # Unpack qweight
        shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.int32)
        w = ((qweight[:, None, :] >> shifts) & 0xF)  # [in//8, 8, out]
        w = w.reshape(in_features, out_features).astype(np.float32)
        # Unpack qzeros
        z = ((qzeros[:, :, None] >> shifts) & 0xF)  # [in//g, out//8, 8]
        z = z.reshape(n_groups, out_features).astype(np.float32) + 1.0
        # BF16 -> F32
        s = scales.astype(np.float32)
        # Dequantize
        w = w.reshape(n_groups, group_size, out_features)
        deq = (w - z[:, None, :]) * s[:, None, :]
        return deq.reshape(in_features, out_features).T.copy()  # [out, in]

class GGUFWriter:
    """Minimal GGUF writer."""
    def __init__(self, path):
        self.file = open(path, 'wb')
        self.tensor_start = None
        self._write_header()
    
    def _write_header(self):
        self.file.write(struct.pack('<I', GGUF_MAGIC))
        self.file.write(struct.pack('<I', GGUF_VERSION))
        # Tensor count placeholder
        self.n_tensors = 0
        self.file.write(struct.pack('<Q', 0))  # n_tensors placeholder
        self.metadata_offset = self.file.tell()
        self.file.write(struct.pack('<Q', 0))  # metadata_size placeholder
    
    def write_metadata_kv(self, key, value, vtype=None):
        """Write a single metadata key-value pair."""
        if vtype is None:
            if isinstance(value, int):
                vtype = GGUF_TYPE_INT32 if -2**31 <= value < 2**31 else GGUF_TYPE_INT64
            elif isinstance(value, float):
                vtype = GGUF_TYPE_FLOAT32
            elif isinstance(value, str):
                vtype = GGUF_TYPE_STRING
            elif isinstance(value, bool):
                vtype = GGUF_TYPE_BOOL
            else:
                raise ValueError(f"Unknown type for {key}: {type(value)}")
        # Write key
        key_bytes = key.encode('utf-8')
        self.file.write(struct.pack('<Q', len(key_bytes)))
        self.file.write(key_bytes)
        # Write value
        if vtype == GGUF_TYPE_UINT8:
            self.file.write(struct.pack('<B', value))
        elif vtype == GGUF_TYPE_INT8:
            self.file.write(struct.pack('<b', value))
        elif vtype == GGUF_TYPE_UINT32:
            self.file.write(struct.pack('<I', value))
        elif vtype == GGUF_TYPE_INT32:
            self.file.write(struct.pack('<i', value))
        elif vtype == GGUF_TYPE_FLOAT32:
            self.file.write(struct.pack('<f', value))
        elif vtype == GGUF_TYPE_BOOL:
            self.file.write(struct.pack('<?', value))
        elif vtype == GGUF_TYPE_UINT64:
            self.file.write(struct.pack('<Q', value))
        elif vtype == GGUF_TYPE_INT64:
            self.file.write(struct.pack('<q', value))
        elif vtype == GGUF_TYPE_FLOAT64:
            self.file.write(struct.pack('<d', value))
        elif vtype == GGUF_TYPE_STRING:
            val_bytes = value.encode('utf-8')
            self.file.write(struct.pack('<Q', len(val_bytes)))
            self.file.write(val_bytes)
        elif vtype == GGUF_TYPE_ARRAY:
            arr = value
            if not arr:
                self.file.write(struct.pack('<I', GGUF_TYPE_BOOL))
                self.file.write(struct.pack('<Q', 0))
            else:
                elem_type = self._infer_type(arr[0])
                self.file.write(struct.pack('<I', elem_type))
                self.file.write(struct.pack('<Q', len(arr)))
                for v in arr:
                    self.write_metadata_kv('', v, elem_type)
    
    def _infer_type(self, v):
        if isinstance(v, int): return GGUF_TYPE_INT32 if -2**31 <= v < 2**31 else GGUF_TYPE_INT64
        if isinstance(v, float): return GGUF_TYPE_FLOAT32
        if isinstance(v, str): return GGUF_TYPE_STRING
        if isinstance(v, bool): return GGUF_TYPE_BOOL
        return GGUF_TYPE_INT32
    
    def finalize_metadata(self):
        self.metadata_end = self.file.tell()
    
    def start_tensors(self):
        self.tensor_start_actual = self.file.tell()
        self.tensor_data_start = self.file.tell() + 8 * self.n_tensors
    
    def compute_tensor_info_size(self, n_elements, qt):
        if qt == GGML_TYPE_F32:
            return n_elements * 4
        elif qt == GGML_TYPE_Q8_0:
            n_blocks = (n_elements + QK8_0 - 1) // QK8_0
            return n_blocks * (2 + 32)  # f16 scale + 32 int8
        elif qt == GGML_TYPE_I32:
            return n_elements * 4
        raise ValueError(f"Unknown quant type {qt}")
    
    def write_tensor_info(self, name, dims, qt, offset, n_elements):
        """Write tensor info entry."""
        name_bytes = name.encode('utf-8')
        n_dims = len(dims)
        self.file.write(struct.pack('<Q', len(name_bytes)))
        self.file.write(name_bytes)
        self.file.write(struct.pack('<I', n_dims))
        for d in dims:
            self.file.write(struct.pack('<Q', d))
        self.file.write(struct.pack('<I', qt))
        self.file.write(struct.pack('<Q', offset))
    
    def write_tensor_data(self, data_bytes):
        """Write tensor data at the current position."""
        pos = self.file.tell()
        self.file.write(data_bytes)
        return pos
    
    def close(self):
        self.file.close()

def write_gguf(model_path, save_path, max_layers=None):
    """Convert DSV4 W4A16 model to GGUF Q8_0."""
    import requests
    from huggingface_hub import hf_hub_url
    
    REPO = "Intel/DeepSeek-V4-Flash-W4A16-AutoRound"
    
    print(f"Loading metadata for {REPO}...")
    meta = get_safetensors_metadata(REPO)
    
    # Build tensor name map
    shard_map = {}  # tensor_name -> (shard_file, dtype, shape)
    for fname, fmeta in meta.files_metadata.items():
        for tname, tinfo in fmeta.tensors.items():
            shard_map[tname] = (fname, tinfo.dtype, tinfo.shape)
    
    print(f"Total tensors: {len(shard_map)}")
    print(f"Total parameters: {meta.parameter_count}")
    
    # Model config
    config = {
        "general.architecture": "deepseek_v4",
        "general.name": "DeepSeek-V4-Flash-W4A16-AutoRound",
        "general.file_type": GGML_TYPE_Q8_0,
        "general.quantization_version": 2,
        "deepseek_v4.block_count": 43,
        "deepseek_v4.context_length": 1048576,
        "deepseek_v4.embedding_length": 4096,
        "deepseek_v4.feed_forward_length": 2048,
        "deepseek_v4.attention.head_count": 64,
        "deepseek_v4.attention.head_count_kv": 1,
        "deepseek_v4.attention.key_length": 512,
        "deepseek_v4.attention.layer_norm_rms_epsilon": 1e-6,
        "deepseek_v4.rope.freq_base": 10000.0,
        "deepseek_v4.rope.dimension_count": 64,
        "deepseek_v4.rope.scaling.type": "yarn",
        "deepseek_v4.rope.scaling.factor": 16,
        "deepseek_v4.rope.scaling.original_max_position_embeddings": 65536,
        "deepseek_v4.rope.scaling.beta_fast": 32,
        "deepseek_v4.rope.scaling.beta_slow": 1,
        "deepseek_v4.vocab_size": 129280,
        "deepseek_v4.expert_count": 256,
        "deepseek_v4.expert_used_count": 6,
        "deepseek_v4.expert_feed_forward_length": 2048,
        "deepseek_v4.n_shared_experts": 1,
        "deepseek_v4.n_hash_layers": 3,
        "deepseek_v4.q_lora_rank": 1024,
        "deepseek_v4.o_lora_rank": 1024,
        "deepseek_v4.o_groups": 8,
        "deepseek_v4.hc_mult": 4,
        "deepseek_v4.head_dim": 512,
        "deepseek_v4.rope_head_dim": 64,
        "deepseek_v4.sliding_window": 128,
        "deepseek_v4.n_mtp_layers": 1,
        "deepseek_v4.swiglu_limit": 10.0,
        "deepseek_v4.score_func": "sqrtsoftplus",
        "deepseek_v4.route_scale": 1.5,
        "deepseek_v4.compress_rope_theta": 160000,
        "deepseek_v4.index_n_heads": 64,
        "deepseek_v4.index_head_dim": 128,
        "deepseek_v4.index_topk": 512,
        "deepseek_v4.hc_sinkhorn_iters": 20,
        "deepseek_v4.hc_eps": 1e-6,
    }
    
    # Compress ratios
    compress_ratios = [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                       4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                       4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
    config["deepseek_v4.compress_ratios"] = compress_ratios
    
    print(f"Creating GGUF: {save_path}")
    writer = GGUFWriter(save_path)
    
    # Write metadata keys
    for key, value in config.items():
        if isinstance(value, list):
            writer.write_metadata_kv(key, value, GGUF_TYPE_ARRAY)
        else:
            writer.write_metadata_kv(key, value)
    
    writer.finalize_metadata()
    
    # Count tensors first to set n_tensors
    tensor_list = []
    for tname in sorted(shard_map.keys()):
        fname, dtype, shape = shard_map[tname]
        # Skip if not a weight (only .qweight, .qzeros, .scales, .weight, .bias, .ape, etc.)
        if max_layers is not None:
            if tname.startswith("layers."):
                layer_id = int(tname.split(".")[1])
                if layer_id >= max_layers:
                    continue
        tensor_list.append((tname, fname, dtype, shape))
    
    # We'll convert each W4A16 weight to Q8_0
    # Non-quantized weights (BF16 embed, F32 head, etc.) stay F32
    
    # Pre-compute tensor sizes
    tensor_offsets = {}
    current_offset = 0
    
    # Actually, let's do this more efficiently - process and write sequentially
    print(f"Processing {len(tensor_list)} tensors...")
    
    writer.n_tensors = len(tensor_list)
    # Seek back and write n_tensors
    writer.file.seek(8)  # After magic + version
    writer.file.write(struct.pack('<Q', writer.n_tensors))
    writer.file.seek(0, 2)  # Seek to end
    
    # Write metadata size
    writer.file.seek(16)
    metadata_size = writer.metadata_end - 24  # After magic(4) + version(4) + n_tensors(8) + metadata_size(8)
    writer.file.write(struct.pack('<Q', writer.metadata_end - 24))
    writer.file.seek(0, 2)
    
    # Start tensor info section
    tensor_info_start = writer.file.tell()
    print(f"Tensor info section at offset {tensor_info_start}")
    
    # First pass: write tensor info with placeholder offsets
    temp_offsets = []
    for tname, fname, dtype, shape in tensor_list:
        # Determine quant type and size
        is_w4a16 = any(tname.endswith(s) for s in ('.qweight', '.qzeros', '.scales'))
        is_qweight = tname.endswith('.qweight')
        is_qzeros = tname.endswith('.qzeros')
        is_scales = tname.endswith('.scales')
        
        if is_qweight:
            # Will convert to Q8_0
            out_features = shape[1]
            in_features = shape[0] * 8  # int32 packing
            n_elements = out_features * in_features
            qt = GGML_TYPE_Q8_0
            dims = [out_features, in_features]
        elif is_qzeros or is_scales:
            # Skip - these are metadata for W4A16, not stored separately
            continue
        elif dtype == 'BF16':
            n_elements = 1
            for d in shape: n_elements *= d
            qt = GGML_TYPE_F32
            dims = list(shape)
        elif dtype == 'F32':
            n_elements = 1
            for d in shape: n_elements *= d
            qt = GGML_TYPE_F32
            dims = list(shape)
        elif dtype == 'I32':
            n_elements = 1
            for d in shape: n_elements *= d
            qt = GGML_TYPE_I32
            dims = list(shape)
        else:
            continue
        
        temp_offsets.append((tname, qt, dims, n_elements, fname, dtype, shape, is_qweight, is_qzeros, is_scales))
    
    # Write tensor info headers with placeholder offsets
    tensor_info_entries = []
    for tname, qt, dims, n_elements, fname, dtype, shape, is_qweight, is_qzeros, is_scales in temp_offsets:
        # Compute size
        if qt == GGML_TYPE_F32:
            data_size = n_elements * 4
        elif qt == GGML_TYPE_Q8_0:
            n_blocks = (n_elements + QK8_0 - 1) // QK8_0
            data_size = n_blocks * 34
        elif qt == GGML_TYPE_I32:
            data_size = n_elements * 4
        else:
            continue
        
        tensor_info_entries.append((tname, dims, qt, data_size, fname, dtype, shape, is_qweight, n_elements))
    
    # Write tensor info (with placeholder offset=0)
    placeholder_offset = 0
    for tname, dims, qt, data_size, fname, dtype, shape, is_qweight, n_elements in tensor_info_entries:
        writer.write_tensor_info(tname, dims, qt, placeholder_offset, n_elements)
    
    # Now write tensor data
    data_start = writer.file.tell()
    print(f"Tensor data starts at offset {data_start}")
    
    current_data_offset = data_start
    actual_offsets = []
    
    # Cache downloaded shards to avoid re-downloading
    shard_cache = {}
    
    download_count = 0
    for idx, (tname, dims, qt, data_size, fname, dtype, shape, is_qweight, n_elements) in enumerate(tensor_info_entries):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(tensor_info_entries)}] Processing...", flush=True)
        
        # Load tensor from safetensors
        if fname not in shard_cache:
            # Download and open
            local_path = hf_hub_download(repo_id=REPO, filename=fname)
            shard_cache[fname] = safe_open(local_path, framework="np", device="cpu")
            download_count += 1
            print(f"  Loaded shard {fname}", flush=True)
        
        sf = shard_cache[fname]
        tensor_np = sf.get_tensor(tname)
        
        if is_qweight:
            # Load corresponding qzeros and scales
            base_name = tname[:-len('.qweight')]
            tname_z = base_name + '.qzeros'
            tname_s = base_name + '.scales'
            
            # Find these in the map
            z_fname, z_dtype, z_shape = shard_map.get(tname_z, (None, None, None))
            s_fname, s_dtype, s_shape = shard_map.get(tname_s, (None, None, None))
            
            if z_fname is None or s_fname is None:
                print(f"  WARNING: Missing qzeros/scales for {tname}, skipping")
                continue
            
            if z_fname not in shard_cache:
                local_path = hf_hub_download(repo_id=REPO, filename=z_fname)
                shard_cache[z_fname] = safe_open(local_path, framework="np", device="cpu")
            if s_fname not in shard_cache:
                local_path = hf_hub_download(repo_id=REPO, filename=s_fname)
                shard_cache[s_fname] = safe_open(local_path, framework="np", device="cpu")
            
            qzeros_np = shard_cache[z_fname].get_tensor(tname_z)
            scales_np = shard_cache[s_fname].get_tensor(tname_s)
            
            # Dequantize
            deq = DequantizeW4A16.dequantize(tensor_np, qzeros_np, scales_np)
            
            # Re-quantize to Q8_0
            out_f, in_f = deq.shape
            data_bytes_flat = deq.ravel()
            q8_bytes, _ = quantize_q8_0(data_bytes_flat)
            data_bytes = bytes(q8_bytes)
            
            # Free memory
            del deq, qzeros_np, scales_np, tensor_np
        elif qt == GGML_TYPE_F32:
            data_bytes = tensor_np.astype(np.float32).tobytes()
            del tensor_np
        elif qt == GGML_TYPE_I32:
            data_bytes = tensor_np.tobytes()
            del tensor_np
        else:
            data_bytes = tensor_np.tobytes()
            del tensor_np
        
        # Write tensor data
        pos = writer.write_tensor_data(data_bytes)
        actual_offsets.append(pos)
        del data_bytes
        gc.collect()
    
    # Now fix the tensor info section with actual offsets
    # We need to go back and update the offsets
    # Since we used placeholder 0, we need to re-write the tensor info section
    print("Updating tensor offsets...", flush=True)
    writer.file.seek(tensor_info_start)
    for (tname, dims, qt, data_size, fname, dtype, shape, is_qweight, n_elements), actual_offset in zip(tensor_info_entries, actual_offsets):
        writer.write_tensor_info(tname, dims, qt, actual_offset, n_elements)
    
    writer.close()
    
    # Cleanup
    for sf in shard_cache.values():
        sf.close()
    
    final_size = os.path.getsize(save_path) / 1e9
    print(f"\nDone! GGUF saved to {save_path} ({final_size:.1f} GB)")
    print(f"Converted {len(tensor_info_entries)} tensors")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", help="Output GGUF path")
    parser.add_argument("--max-layers", type=int, default=None, help="Max layers to convert")
    args = parser.parse_args()
    write_gguf(args.save_path, args.max_layers)
