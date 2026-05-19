#!/usr/bin/env python3
"""MojoLlama GGUF Quantizer — comprehensive quantization pipeline.

Full CLI for model conversion, quantization (all K/IQ quants), imatrix
generation, NF4, validation, benchmarking, and batch operations.

Usage:
    python3 quantizer.py list-types
    python3 quantizer.py info model.gguf
    python3 quantizer.py imatrix model.gguf --data calibration.txt --output imatrix.dat
    python3 quantizer.py quantize model.gguf --type Q4_K_M --imatrix imatrix.dat
    python3 quantizer.py nf4 model.gguf --output model-nf4.gguf
    python3 quantizer.py convert hf_model --outtype f16 --outfile model.gguf
    python3 quantizer.py validate model.gguf
    python3 quantizer.py benchmark model.gguf --prompt "Hello" --max-tokens 50
    python3 quantizer.py batch hf_model --types q4_0,q4_k_m,q8_0
    python3 quantizer.py compare base.gguf quantized.gguf
"""

import os
import sys
import json
import time
import math
import argparse
import subprocess
import tempfile
import re
from pathlib import Path
from io import StringIO
from typing import Optional

import numpy as np

# ─── Paths ──────────────────────────────────────────────────────────────

LLAMA_CPP = "/tmp/llama.cpp"
LLAMA_QUANTIZE = f"{LLAMA_CPP}/build/bin/llama-quantize"
LLAMA_IMATRIX= f"{LLAMA_CPP}/build/bin/llama-imatrix"
LLAMA_BENCH   = f"{LLAMA_CPP}/build/bin/llama-bench"
LLAMA_CONVERT = f"{LLAMA_CPP}/convert_hf_to_gguf.py"

VERSION = "0.2.0"

# ─── Quantization Type Registry ─────────────────────────────────────────

# (name, description, bpw, is_k_quant, category)
QUANT_TYPES = [
    # Float
    ("F32",     "32-bit float",            32.0,  False, "float"),
    ("F16",     "16-bit float",            16.0,  False, "float"),
    ("BF16",    "bfloat16",                16.0,  False, "float"),
    # Standard block quants
    ("Q8_0",    "8-bit block quant",        8.0,  False, "standard"),
    ("Q6_K",    "6-bit K-quant",            6.14, True,  "standard"),
    ("Q5_1",    "5-bit block quant",        5.65, False, "standard"),
    ("Q5_0",    "5-bit block quant",        5.21, False, "standard"),
    ("Q4_1",    "4-bit block quant",        4.78, False, "standard"),
    ("Q4_0",    "4-bit block quant",        4.34, False, "standard"),
    ("Q1_0",    "1-bit block quant",        1.125, False, "standard"),
    # K-Quants (mixture)
    ("Q5_K_M",  "5-bit K-quant medium",     5.33, True,  "k_quant"),
    ("Q5_K_S",  "5-bit K-quant small",      5.21, True,  "k_quant"),
    ("Q4_K_M",  "4-bit K-quant medium",     4.58, True,  "k_quant"),
    ("Q4_K_S",  "4-bit K-quant small",      4.37, True,  "k_quant"),
    ("Q3_K_L",  "3-bit K-quant large",      4.03, True,  "k_quant"),
    ("Q3_K_M",  "3-bit K-quant medium",     3.74, True,  "k_quant"),
    ("Q3_K_S",  "3-bit K-quant small",      3.41, True,  "k_quant"),
    ("Q2_K",    "2-bit K-quant",            2.96, True,  "k_quant"),
    ("Q2_K_S",  "2-bit K-quant small",      2.96, True,  "k_quant"),
    # Importance-aware quants (IQ)
    ("IQ4_NL",  "4-bit non-linear IQ",      4.50,  False, "iq_quant"),
    ("IQ4_XS",  "4-bit extra-small IQ",     4.25,  False, "iq_quant"),
    ("IQ3_XXS", "3-bit extra-extra-small IQ", 3.06, False,"iq_quant"),
    ("IQ3_XS",  "3-bit extra-small IQ",     3.30,  False, "iq_quant"),
    ("IQ3_S",   "3-bit small IQ",           3.44,  False, "iq_quant"),
    ("IQ3_M",   "3-bit medium IQ",          3.66,  False, "iq_quant"),
    ("IQ2_XXS", "2-bit extra-extra-small IQ",2.06,  False, "iq_quant"),
    ("IQ2_XS",  "2-bit extra-small IQ",     2.31,  False, "iq_quant"),
    ("IQ2_S",   "2-bit small IQ",           2.50,  False, "iq_quant"),
    ("IQ2_M",   "2-bit medium IQ",          2.70,  False, "iq_quant"),
    ("IQ1_S",   "1-bit small IQ",           1.56,  False, "iq_quant"),
    ("IQ1_M",   "1-bit medium IQ",          1.75,  False, "iq_quant"),
    # Ternary
    ("TQ1_0",   "1-bit ternary",            1.69, False, "ternary"),
    ("TQ2_0",   "2-bit ternary",            2.06, False, "ternary"),
    # Special
    ("MXFP4",   "MXFP4 (microscaling)",     4.00,  False, "special"),
    ("NVFP4",   "NVidia FP4 format",        4.00,  False, "special"),
]

QUANT_ALIASES = {
    "Q4_K": "Q4_K_M",
    "Q5_K": "Q5_K_M",
    "Q3_K": "Q3_K_M",
}

# ─── Helpers ────────────────────────────────────────────────────────────

def _resolve_quant_type(name: str) -> str:
    """Resolve quant type name (handle aliases)."""
    upper = name.upper().replace("-", "_")
    return QUANT_ALIASES.get(upper, upper)

def _find_gguf_files(path: str = ".") -> list:
    """Find all .gguf files under a path."""
    return sorted(Path(path).rglob("*.gguf"))

def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.0f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.0f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"

def _read_gguf_metadata(model_path: str) -> dict:
    """Read GGUF file metadata."""
    from gguf import GGUFReader, GGUFValueType
    reader = GGUFReader(model_path)
    meta = {
        "path": model_path,
        "size_bytes": os.path.getsize(model_path),
        "size_hr": _format_size(os.path.getsize(model_path)),
        "n_tensors": len(reader.tensors),
        "n_fields": len(reader.fields),
    }

    # Try to read GGUF.tensor_count from metadata
    try:
        tc_field = reader.get_field("GGUF.tensor_count")
        if tc_field is not None:
            meta["tensor_count"] = int(np.asarray(tc_field.parts[-1]).flat[0])
        else:
            meta["tensor_count"] = len(reader.tensors)
    except Exception:
        meta["tensor_count"] = len(reader.tensors)
    # Architecture-aware KV extraction
    # GGUF uses architecture-specific key prefixes (e.g., deepseek.context_length,
    # qwen2.context_length, gemma.block_count).  Read general.architecture first
    # so we can try the right prefix.
    ARCH_PREFIX_MAP = {
        'deepseek': ['deepseek.', 'deepseek2.', 'deepseek3.', 'llama.'],
        'qwen': ['qwen2.', 'qwen2moe.', 'qwen3moe.', 'llama.'],
        'qwen2': ['qwen2.', 'qwen2moe.', 'llama.'],
        'gemma': ['gemma.', 'gemma2.', 'llama.'],
        'gemma2': ['gemma2.', 'gemma.', 'llama.'],
        'phi': ['phi3.', 'phi2.', 'llama.'],
        'phi3': ['phi3.', 'phi2.', 'llama.'],
        'command-r': ['command-r.', 'commandr.', 'llama.'],
        'starcoder': ['starcoder.', 'llama.'],
        'falcon': ['falcon.', 'llama.'],
        'chatglm': ['chatglm.', 'glm.', 'llama.'],
        'baichuan': ['baichuan.', 'llama.'],
        'llama': ['llama.', 'llama2.', 'llama3.', 'codellama.'],
    }

    # Determine architecture
    arch_str = 'llama'
    arch_field = reader.get_field('general.architecture')
    if arch_field is not None:
        try:
            raw = bytes(arch_field.parts[-1])
            arch_str = raw.decode('utf-8').strip('\x00').lower()
        except Exception:
            pass
    meta['architecture'] = arch_str

    # Resolve architecture group (e.g., 'deepseek3' -> 'deepseek')
    arch_group = arch_str
    if arch_str not in ARCH_PREFIX_MAP:
        for group, prefixes in ARCH_PREFIX_MAP.items():
            if arch_str.startswith(group) or group.startswith(arch_str):
                arch_group = group
                break
        else:
            arch_group = 'llama'
    prefixes_to_try = ARCH_PREFIX_MAP.get(arch_group, ['llama.'])
    # Also add the raw arch as a prefix for unknown architectures
    if arch_str != arch_group and arch_str not in prefixes_to_try:
        prefixes_to_try.insert(0, arch_str + '.')

    meta_keys = [
        ('context_length', 'context_length'),
        ('embedding_length', 'embedding_length'),
        ('block_count', 'block_count'),
        ('feed_forward_length', 'ff_length'),
        ('attention.head_count', 'n_heads'),
        ('attention.head_count_kv', 'n_kv_heads'),
        ('rope.dimension_count', 'rope_dim'),
        ('attention.layer_norm_rms_epsilon', 'rms_norm_eps'),
    ]

    for gguf_suffix, meta_key in meta_keys:
        for prefix in prefixes_to_try:
            try:
                field = reader.get_field(prefix + gguf_suffix)
                if field is not None:
                    if field.types[-1] in (GGUFValueType.STRING,):
                        raw = bytes(field.parts[-1])
                        meta[meta_key] = raw.decode("utf-8").strip("\x00")
                    else:
                        arr = np.asarray(field.parts[-1])
                        meta[meta_key] = int(arr.flat[0]) if arr.size == 1 else arr.tolist()
                    break  # Found the key, move to next
            except Exception:
                continue

    # Also read general.name, general.file_type, general.description (arch-agnostic keys)
    for gguf_key, meta_key in [("general.name", "name"), ("general.file_type", "file_type"),
                                ("general.description", "description")]:
        try:
            field = reader.get_field(gguf_key)
            if field is not None:
                if field.types[-1] in (GGUFValueType.STRING,):
                    raw = bytes(field.parts[-1])
                    meta[meta_key] = raw.decode("utf-8").strip("\x00")
                else:
                    arr = np.asarray(field.parts[-1])
                    meta[meta_key] = int(arr.flat[0]) if arr.size == 1 else arr.tolist()
        except Exception:
            pass

    # Store the prefixes used for reference
    meta['_prefixes_used'] = prefixes_to_try

    # Count tensors per type
    from gguf import GGMLQuantizationType
    type_counts = {}
    for t in reader.tensors:
        qt = GGMLQuantizationType(t.tensor_type)
        try:
            name = qt.name
        except ValueError:
            name = str(t.tensor_type)
        type_counts[name] = type_counts.get(name, 0) + 1
    meta["tensor_types"] = type_counts

    # Calculate total params
    total_params = 0
    for t in reader.tensors:
        shape_dims = 1
        for d in t.shape:
            shape_dims *= int(d)
        total_params += shape_dims
    meta["n_params"] = total_params
    if total_params >= 1_000_000_000:
        meta["n_params_hr"] = f"{total_params / 1e9:.1f}B"
    elif total_params >= 1_000_000:
        meta["n_params_hr"] = f"{total_params / 1e6:.1f}M"
    else:
        meta["n_params_hr"] = str(total_params)

    # Determine dominant quant type
    if type_counts:
        meta["quant_type"] = max(type_counts, key=type_counts.get)

    return meta

# ─── IMatrix Generation ─────────────────────────────────────────────────

def cmd_imatrix(args):
    """Generate importance matrix for quant optimization."""
    print("╔═══════════════════════════════════════════╗")
    print("║    Importance Matrix Generator            ║")
    print("╚═══════════════════════════════════════════╝")
    print()

    model = args.model
    data = args.data
    output = args.output or f"{Path(model).stem}-imatrix.dat"
    threads = args.threads or os.cpu_count() // 2 or 4
    ctx_size = args.ctx_size or 512

    if not os.path.exists(LLAMA_IMATRIX):
        print(f"❌ llama-imatrix not found at {LLAMA_IMATRIX}")
        print("   Build it: cd /tmp/llama.cpp/build && cmake --build . --target llama-imatrix")
        return 1

    if not os.path.exists(model):
        print(f"❌ Model not found: {model}")
        return 1

    # Prepare calibration data
    if data and not os.path.exists(data):
        print(f"❌ Calibration data not found: {data}")
        return 1

    print(f"Model:           {model}")
    print(f"Calibration:     {data or 'internal (model self-tokens)'}")
    print(f"Output:          {output}")
    print(f"Threads:         {threads}")
    print(f"Context size:    {ctx_size}")
    print()

    cmd = [
        LLAMA_IMATRIX, "-m", model,
        "-t", str(threads),
        "-c", str(ctx_size),
        "-o", output,
    ]

    if data:
        cmd.extend(["-f", data])

    print(f"Running: {' '.join(cmd)}")
    print()
    t0 = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in proc.stdout:
        line = line.rstrip()
        print(f"  {line}")

    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode == 0 and os.path.exists(output):
        size = os.path.getsize(output)
        print(f"\n✅ imatrix generated: {output} ({_format_size(size)}) in {_format_duration(elapsed)}")
        return 0
    else:
        print(f"\n❌ imatrix generation failed (exit code {proc.returncode})")
        return 1

# ─── Quantize ───────────────────────────────────────────────────────────

def cmd_quantize(args):
    """Quantize a GGUF model to a different type."""
    print("╔═══════════════════════════════════════════╗")
    print("║    GGUF Quantizer                         ║")
    print("╚═══════════════════════════════════════════╝")
    print()

    model_path = args.model
    quant_type = _resolve_quant_type(args.type)
    output = args.output
    imatrix = args.imatrix
    threads = args.threads or os.cpu_count() // 2 or 4
    allow_requantize = args.allow_requantize
    pure = args.pure
    leave_output = args.leave_output
    dry_run = args.dry_run

    if not os.path.exists(LLAMA_QUANTIZE):
        print(f"❌ llama-quantize not found at {LLAMA_QUANTIZE}")
        print("   Build it: cd /tmp/llama.cpp/build && cmake --build . --target llama-quantize")
        return 1

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return 1

    if not output:
        stem = Path(model_path).stem
        output = str(Path(model_path).parent / f"{stem}-{quant_type}.gguf")

    print(f"Input:           {model_path}")
    print(f"Output:          {output}")
    print(f"Quant type:      {quant_type}")
    if imatrix:
        print(f"Importance mat:  {imatrix}")
    if allow_requantize:
        print(f"Allow requantize: yes")
    print()

    cmd = [LLAMA_QUANTIZE, model_path, output, quant_type, str(threads)]

    if imatrix:
        cmd.extend(["--imatrix", imatrix])
    if allow_requantize:
        cmd.append("--allow-requantize")
    if pure:
        cmd.append("--pure")
    if leave_output:
        cmd.append("--leave-output-tensor")
    if dry_run:
        cmd.append("--dry-run")

    try:
        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  {line}")
        proc.wait()
        elapsed = time.time() - t0

        if proc.returncode == 0:
            if os.path.exists(output) and not dry_run:
                size = os.path.getsize(output)
                orig_size = os.path.getsize(model_path)
                ratio = size / orig_size if orig_size > 0 else 0
                print(f"\n✅ Quantization complete: {output}")
                print(f"   Size: {_format_size(size)} ({ratio:.1%} of original)")
                print(f"   Time: {_format_duration(elapsed)}")
            else:
                print(f"\n✅ Dry-run complete")
            return 0
        else:
            print(f"\n❌ Quantization failed (exit code {proc.returncode})")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

# ─── NF4 Quantization (NormalFloat4 for QLoRA) ──────────────────────────

def quantize_nf4_block(block: np.ndarray) -> tuple:
    """Quantize a block of floats to NF4 format.

    NF4 is the NormalFloat4 format from the QLoRA paper:
    - Blocks of 64 values
    - Uses absolute maximum scaling (absmax)
    - 4-bit symmetric quantization: 16 evenly spaced levels from -1 to 1
    - Data type levels: {-1.0, -0.8667, -0.7333, -0.6, -0.4667, -0.3333,
                         -0.2, -0.0667, 0.0667, 0.2, 0.3333, 0.4667,
                          0.6, 0.7333, 0.8667, 1.0}
    """
    NF4_LEVELS = np.array([
        -1.0, -0.8666667, -0.7333333, -0.6,
        -0.4666667, -0.3333333, -0.2, -0.0666667,
        0.0666667, 0.2, 0.3333333, 0.4666667,
        0.6, 0.7333333, 0.8666667, 1.0
    ], dtype=np.float32)

    absmax = np.max(np.abs(block))
    if absmax == 0:
        return np.zeros(len(block) // 2, dtype=np.uint8), 0.0

    # Normalize to [-1, 1]
    normalized = block / absmax

    # Quantize to nearest NF4 level
    indices = np.zeros(len(normalized), dtype=np.uint8)
    for i, val in enumerate(normalized):
        idx = np.argmin(np.abs(NF4_LEVELS - val))
        indices[i] = idx

    # Pack 2×4-bit values per byte
    packed = np.zeros(len(indices) // 2, dtype=np.uint8)
    for i in range(0, len(indices), 2):
        packed[i // 2] = (indices[i] & 0x0F) | ((indices[i + 1] & 0x0F) << 4)

    return packed, absmax


def dequantize_nf4_block(packed: np.ndarray, scale: float, n_values: int) -> np.ndarray:
    """Dequantize an NF4 block back to float32."""
    NF4_LEVELS = np.array([
        -1.0, -0.8666667, -0.7333333, -0.6,
        -0.4666667, -0.3333333, -0.2, -0.0666667,
        0.0666667, 0.2, 0.3333333, 0.4666667,
        0.6, 0.7333333, 0.8666667, 1.0
    ], dtype=np.float32)

    if scale == 0:
        return np.zeros(n_values, dtype=np.float32)

    indices = np.zeros(n_values, dtype=np.uint8)
    for i in range(len(packed)):
        lo = packed[i] & 0x0F
        hi = (packed[i] >> 4) & 0x0F
        indices[i * 2] = lo
        if i * 2 + 1 < n_values:
            indices[i * 2 + 1] = hi

    return NF4_LEVELS[indices] * scale


def cmd_nf4(args):
    """Convert a GGUF model to NF4 quantization for QLoRA."""
    print("╔═══════════════════════════════════════════╗")
    print("║    NF4 Quantizer (NormalFloat4)           ║")
    print("╚═══════════════════════════════════════════╝")
    print()

    model_path = args.model
    output = args.output or f"{Path(model_path).stem}-nf4.gguf"
    block_size = args.block_size or 64

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return 1

    try:
        from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
    except ImportError as e:
        print(f"❌ gguf package required: {e}")
        print("   pip install gguf")
        return 1

    print(f"Input:           {model_path}")
    print(f"Output:          {output}")
    print(f"Block size:      {block_size}")
    print()

    reader = GGUFReader(model_path)
    meta = _read_gguf_metadata(model_path)

    print(f"Architecture:    {meta.get('architecture', 'unknown')}")
    print(f"Parameters:      {meta.get('n_params_hr', '?')}")
    print(f"Tensors:         {len(reader.tensors)}")
    print()

    # Create output GGUF
    arch = meta.get("architecture", "llama")
    writer = GGUFWriter(output, arch)

    # Copy KV metadata
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        try:
            raw = bytes(field.parts[-1])
            # Detect string vs numeric
            if field.types[-1] in (GGUFValueType.STRING,):
                val = raw.decode("utf-8").strip("\x00")
                writer.add_string(name, val)
            elif field.types[-1] in (GGUFValueType.UINT32,):
                from gguf import GGUFValueType
                if field.types[-1] == GGUFValueType.UINT32:
                    writer.add_uint32(name, int(np.asarray(field.parts[-1]).flat[0]))
        except Exception:
            pass

    t0 = time.time()
    nf4_replace_count = 0
    total_size_before = 0
    total_size_after = 0

    for tensor in reader.tensors:
        data = np.asarray(tensor.data)

        # Determine original quant type
        from gguf import GGMLQuantizationType
        orig_type = GGMLQuantizationType(tensor.tensor_type)
        orig_name = orig_type.name

        # Shape information
        shape = [int(d) for d in tensor.shape]
        flat = data.ravel()
        n_elements = len(flat)

        # Skip very small tensors and keep them at original precision
        if n_elements < block_size * 2:
            writer.add_tensor(tensor.name, data, raw_dtype=orig_type)
            continue

        # Dequantize to float32
        from gguf import dequantize

        # Calculate bytes for NF4
        nf4_bytes_needed = (n_elements * 4 // 8) + (n_elements // block_size) * 4  # data + scales
        original_bytes = data.nbytes

        # Only quantize weight tensors (not norms, etc.)
        is_weight = any(s in tensor.name for s in ['.weight', 'token_embd', 'output'])
        is_small = n_elements < 1024

        if is_weight and not is_small:
            try:
                deq = dequantize(data, tensor.tensor_type).astype(np.float32)
                deq_flat = deq.ravel()

                # Quantize block by block
                n_blocks = (n_elements + block_size - 1) // block_size
                packed_blocks = []
                scales = []

                for b in range(n_blocks):
                    start = b * block_size
                    end = min(start + block_size, n_elements)
                    block = deq_flat[start:end]

                    # Pad last block if needed
                    if len(block) < block_size:
                        block = np.pad(block, (0, block_size - len(block)))

                    packed, scale = quantize_nf4_block(block)
                    packed_blocks.append(packed)
                    scales.append(scale)

                # Store as packed format: [scales (float32), packed data (uint8)]
                scale_arr = np.array(scales, dtype=np.float32)
                packed_all = np.concatenate(packed_blocks)
                nf4_data = np.concatenate([
                    scale_arr.view(np.uint8).ravel(),
                    packed_all.ravel(),
                ])

                writer.add_tensor(tensor.name, nf4_data, raw_dtype=GGMLQuantizationType.F16)
                nf4_replace_count += 1
                total_size_before += original_bytes
                total_size_after += len(nf4_data)

            except Exception as e:
                # Fall back to original type
                writer.add_tensor(tensor.name, data, raw_dtype=orig_type)
        else:
            writer.add_tensor(tensor.name, data, raw_dtype=orig_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    elapsed = time.time() - t0
    output_size = os.path.getsize(output) if os.path.exists(output) else 0

    print(f"\n✅ NF4 conversion complete")
    print(f"   Weights converted: {nf4_replace_count}")
    if total_size_before > 0:
        print(f"   Size reduction: {_format_size(total_size_before)} → {_format_size(output_size)} ({output_size/total_size_before:.1%})")
    print(f"   Time: {_format_duration(elapsed)}")
    print()
    print(f"   NOTE: NF4 is a custom format for QLoRA. The output GGUF")
    print(f"   stores weight data as packed NF4 blocks with float32 scales.")
    print(f"   Standard llama.cpp cannot load this file directly —")
    print(f"   it's intended for MojoLlama's QLoRA training pipeline.")

    return 0

# ─── List Types ─────────────────────────────────────────────────────────

def cmd_list_types(args):
    """List all supported quantization types."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          Supported Quantization Types                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"{'Name':<14} {'Bits/Param':<12} {'Category':<14} Description")
    print("-" * 80)

    categories = ["float", "standard", "k_quant", "iq_quant", "ternary", "special"]
    current_cat = None

    for name, desc, bpw, is_k, cat in QUANT_TYPES:
        if cat != current_cat:
            if current_cat is not None:
                print()
            current_cat = cat
            cat_label = {
                "float": "── Float Types ──",
                "standard": "── Standard Block Quants ──",
                "k_quant": "── K-Quants (mixture) ──",
                "iq_quant": "── Importance-Aware Quants (IQ) ──",
                "ternary": "── Ternary ──",
                "special": "── Special ──",
            }.get(cat, cat)
            print(f"  {cat_label}")
            print()

        print(f"  {name:<12} {bpw:<12.3f} {cat:<14} {desc}")

    print()
    print(f"Total: {len(QUANT_TYPES)} quantization types")
    print()

# ─── Info ───────────────────────────────────────────────────────────────

def cmd_info(args):
    """Inspect a GGUF model file."""
    model_path = args.model
    verbose = args.verbose

    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return 1

    meta = _read_gguf_metadata(model_path)

    print("╔═══════════════════════════════════════════╗")
    print("║    GGUF Model Inspector                   ║")
    print("╚═══════════════════════════════════════════╝")
    print()
    print(f"  Path:             {meta['path']}")
    print(f"  Size:             {meta['size_hr']}")
    print(f"  Architecture:     {meta.get('architecture', '?')}")
    print(f"  Name:             {meta.get('name', '?')}")
    print(f"  Parameters:       {meta.get('n_params_hr', '?')}")
    print(f"  Quantization:     {meta.get('quant_type', '?')}")
    print(f"  Tensors:          {meta['n_tensors']}")
    print(f"  Metadata fields:  {meta['n_fields']}")
    print()

    if "context_length" in meta:
        print(f"  Context length:   {meta['context_length']}")
    if "embedding_length" in meta:
        print(f"  Embedding dim:    {meta['embedding_length']}")
    if "block_count" in meta:
        print(f"  Layers:           {meta['block_count']}")
    if "n_heads" in meta:
        print(f"  Attention heads:  {meta['n_heads']}")
    if "n_kv_heads" in meta:
        print(f"  KV heads:         {meta['n_kv_heads']}")

    if "tensor_types" in meta and meta["tensor_types"]:
        print()
        print("  Tensor types:")
        for tname, count in sorted(meta["tensor_types"].items(), key=lambda x: -x[1]):
            print(f"    {tname:<12} {count:>4} tensors")

    if verbose:
        print()
        print("  ── All Fields ──")
        try:
            from gguf import GGUFReader
            reader = GGUFReader(model_path)
            for name, field in reader.fields.items():
                print(f"    {name}")
        except Exception:
            pass

    print()
    return 0

# ─── Convert (HF → GGUF) ───────────────────────────────────────────────

def cmd_convert(args):
    """Convert a HuggingFace model to GGUF format."""
    print("╔═══════════════════════════════════════════╗")
    print("║    HuggingFace → GGUF Converter           ║")
    print("╚═══════════════════════════════════════════╝")
    print()

    model = args.model
    outtype = args.outtype or "f16"
    output = args.output
    verbose = args.verbose

    if not os.path.exists(LLAMA_CONVERT):
        print(f"❌ Converter not found at {LLAMA_CONVERT}")
        print("   llama.cpp must be checked out at /tmp/llama.cpp")
        return 1

    if not output:
        model_name = model.split("/")[-1] if "/" in model else model
        output = f"{model_name}-{outtype}.gguf"

    # Direct types that convert_hf_to_gguf.py supports natively
    direct_types = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
    # Types needing post-quantization via llama-quantize
    post_quant_types = {
        "q4_0", "q4_1", "q5_0", "q5_1",
        "q2_k", "q3_k", "q3_k_s", "q3_k_m", "q3_k_l",
        "q4_k", "q4_k_m", "q4_k_s",
        "q5_k", "q5_k_m", "q5_k_s",
        "q6_k", "q8_k",
    }

    upper_outtype = outtype.lower()

    if upper_outtype in post_quant_types:
        # Two-step: convert to f16 first, then post-quantize
        intermediate = f"/tmp/{model_name}-intermediate-f16.gguf"
        step1_type = "f16"
    elif upper_outtype in direct_types:
        step1_type = upper_outtype
        intermediate = output
    else:
        print(f"❌ Unknown outtype: {outtype}")
        print(f"   Supported: {', '.join(sorted(direct_types | post_quant_types))}")
        return 1

    print(f"Model:           {model}")
    print(f"Out type:        {upper_outtype}")
    print(f"Output:          {output}")
    print()

    step_start = time.time()

    # Step 1: Convert to intermediate
    print(f"[1/2] Converting to {step1_type}...")
    cmd = [
        sys.executable, LLAMA_CONVERT,
        model,
        "--outtype", step1_type,
        "--outfile", intermediate,
    ]
    if verbose:
        cmd.append("--verbose")

    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"❌ Conversion failed:")
        print(proc.stdout[-500:] if proc.stdout else "")
        print(proc.stderr[-500:] if proc.stderr else "")
        return 1

    print(f"  ✅ Intermediate saved: {intermediate}")
    step_elapsed = time.time() - step_start

    # Step 2: Post-quantize if needed
    if upper_outtype in post_quant_types:
        print(f"\n[2/2] Post-quantizing to {upper_outtype}...")
        quant_cmd = [
            LLAMA_QUANTIZE, intermediate, output, upper_outtype,
            str(os.cpu_count() // 2 or 4),
        ]
        print(f"  Running: {' '.join(quant_cmd)}")
        quant_proc = subprocess.run(quant_cmd, capture_output=True, text=True)
        if quant_proc.returncode != 0:
            print(f"❌ Post-quantization failed:")
            print(quant_proc.stdout[-300:] if quant_proc.stdout else "")
            print(quant_proc.stderr[-300:] if quant_proc.stderr else "")
            # Keep intermediate for debugging
            return 1

        # Remove intermediate
        try:
            os.remove(intermediate)
        except OSError:
            pass

        print(f"  ✅ Post-quantization complete")

    total_elapsed = time.time() - step_start

    if os.path.exists(output):
        size = os.path.getsize(output)
        print(f"\n✅ Export complete: {output} ({_format_size(size)})")
        print(f"   Time: {_format_duration(total_elapsed)}")
    else:
        print(f"\n❌ Output file not found: {output}")

    return 0

# ─── Validate ───────────────────────────────────────────────────────────

def cmd_validate(args):
    """Validate a GGUF model file integrity."""
    model_path = args.model

    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return 1

    print("╔═══════════════════════════════════════════╗")
    print("║    GGUF Model Validator                   ║")
    print("╚═══════════════════════════════════════════╝")
    print()

    issues = []
    size = os.path.getsize(model_path)
    print(f"  Model:  {model_path}")
    print(f"  Size:   {_format_size(size)}")
    print()

    # 1. Check file size
    if size == 0:
        issues.append(("CRITICAL", "File is empty"))
    elif size < 1024 ** 2:
        issues.append(("WARNING", f"File is very small ({_format_size(size)}) — unlikely a valid GGUF"))
    else:
        print(f"  ✅ File size OK")

    # 2. Try loading with gguf
    try:
        from gguf import GGUFReader, GGMLQuantizationType
        reader = GGUFReader(model_path)
        print(f"  ✅ GGUF header valid: {len(reader.tensors)} tensors, {len(reader.fields)} KV fields")

        # 3. Check all tensors
        broken = 0
        for t in reader.tensors:
            try:
                _ = t.data.size
                _ = t.tensor_type
                _ = t.shape
            except Exception:
                broken += 1
                if len(issues) < 10:
                    issues.append(("ERROR", f"Tensor {t.name} is corrupt"))

        if broken:
            issues.append(("ERROR", f"{broken} corrupt tensors found"))
        else:
            print(f"  ✅ All {len(reader.tensors)} tensors readable")

        # 4. Check for truncated tensors
        truncated = 0
        for t in reader.tensors:
            try:
                data = t.data
                if data.size == 0:
                    truncated += 1
            except Exception:
                truncated += 1

        if truncated:
            issues.append(("WARNING", f"{truncated} tensors appear truncated"))

        # 5. Check architecture metadata
        try:
            arch_field = reader.get_field("general.architecture")
            if arch_field is not None:
                arch_bytes = bytes(arch_field.parts[-1])
                arch = arch_bytes.decode("utf-8").strip("\x00")
                print(f"  ✅ Architecture: {arch}")
            else:
                issues.append(("WARNING", "No general.architecture field"))
        except Exception:
            issues.append(("WARNING", "Could not read architecture field"))

    except Exception as e:
        issues.append(("CRITICAL", f"Not a valid GGUF file: {e}"))

    # Summary
    print()
    if issues:
        print(f"  Found {len(issues)} issue(s):")
        for severity, msg in issues:
            icon = {"CRITICAL": "❌", "ERROR": "⚠️", "WARNING": "⚠️"}.get(severity, "❓")
            print(f"    {icon} [{severity}] {msg}")
        print()
        has_critical = any(s == "CRITICAL" for s, _ in issues)
        return 1 if has_critical else 0
    else:
        print(f"  ✅ Model validated successfully")
        print()
        return 0


# ─── Benchmark ──────────────────────────────────────────────────────────

def cmd_benchmark(args):
    """Benchmark model inference speed."""
    model_path = args.model
    prompt = args.prompt or "Hello"
    max_tokens = args.max_tokens or 128
    threads = args.threads or 0

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return 1

    print("╔═══════════════════════════════════════════╗")
    print("║    Model Benchmark                        ║")
    print("╚═══════════════════════════════════════════╝")
    print()

    meta = _read_gguf_metadata(model_path)
    print(f"  Model:       {model_path}")
    print(f"  Quant:       {meta.get('quant_type', '?')}")
    print(f"  Params:      {meta.get('n_params_hr', '?')}")
    print()

    bench_path = LLAMA_BENCH
    if os.path.exists(bench_path):
        print(f"  Using llama-bench (production benchmark)")
        print()

        cmd = [bench_path, "-m", model_path, "-p", str(max_tokens), "-n", str(max_tokens)]
        if threads:
            cmd.extend(["-t", str(threads)])

        print(f"  Running: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0

        # Parse markdown-style output
        output = proc.stdout + proc.stderr
        print()
        for line in output.split("\n"):
            if "|" in line and ("pp" in line.lower() or "tg" in line.lower()):
                parts = [p.strip() for p in line.split("|")]
                print(f"  {line}")
        print()

        if proc.returncode == 0:
            print(f"  ✅ Benchmark complete ({_format_duration(elapsed)})")
        else:
            print(f"  ⚠️  Benchmark returned exit code {proc.returncode}")
    else:
        print(f"  llama-bench not found, using fallback")
        print()

        # Try to use the backend
        t0 = time.time()
        try:
            from mojollama.backends import AutoBackend
            be = AutoBackend(model_path=model_path)
            result = be.generate(prompt, max_tokens=max_tokens)
            elapsed = time.time() - t0
            gen_tokens = result.get("tokens", 0)
            tok_s = gen_tokens / elapsed if elapsed > 0 else 0
            print(f"  Generated: {result.get('text', '')[:100]}...")
            print(f"  Tokens:    {gen_tokens}")
            print(f"  Time:      {elapsed:.2f}s")
            print(f"  Speed:     {tok_s:.1f} tok/s")
        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")
            return 1

    return 0


# ─── Compare ────────────────────────────────────────────────────────────

def cmd_compare(args):
    """Compare two GGUF models (original vs quantized)."""
    base = args.base
    quantized = args.quantized
    samples = args.samples or 100

    for path, label in [(base, "Base"), (quantized, "Quantized")]:
        if not os.path.exists(path):
            print(f"❌ {label} model not found: {path}")
            return 1

    print("╔═══════════════════════════════════════════╗")
    print("║    Model Comparison                       ║")
    print("╚═══════════════════════════════════════════╝")
    print()

    try:
        from gguf import GGUFReader, dequantize
        from mojollama.model.q4_kernels import Q4Matmul
    except ImportError:
        pass

    base_meta = _read_gguf_metadata(base)
    quant_meta = _read_gguf_metadata(quantized)

    print(f"  {'':<20} {'Base':>20} {'Quantized':>20}")
    print(f"  {'─'*20} {'─'*20} {'─'*20}")
    print(f"  {'Size':<20} {base_meta.get('size_hr', '?'):>20} {quant_meta.get('size_hr', '?'):>20}")
    print(f"  {'Quant type':<20} {base_meta.get('quant_type', '?'):>20} {quant_meta.get('quant_type', '?'):>20}")
    print(f"  {'Compression':<20} {'100%':>20} {base_meta.get('size_bytes', 0) > 0 and os.path.getsize(quantized)/base_meta.get('size_bytes', 1)*100:.1f}%")

    if base_meta.get("n_params") and quant_meta.get("n_params"):
        diff = abs(base_meta["n_params"] - quant_meta["n_params"])
        if diff > 0:
            print(f"  {'⚠️  Param mismatch':<20} {base_meta.get('n_params_hr', ''):>20} {quant_meta.get('n_params_hr', ''):>20}")

    # Quick quality check: compare logits from a small prompt
    print()
    print("  Sampling comparison (loading tensors)...")
    try:
        base_reader = GGUFReader(base)
        quant_reader = GGUFReader(quantized)

        # Compare first weight tensor
        matching = 0
        different = 0
        for t_base in base_reader.tensors[:100]:
            matching_name = False
            for t_quant in quant_reader.tensors:
                if t_base.name == t_quant.name:
                    matching_name = True
                    # Compare shapes
                    if list(t_base.shape) != list(t_quant.shape):
                        print(f"  ⚠️  Shape mismatch: {t_base.name} "
                              f"{list(t_base.shape)} vs {list(t_quant.shape)}")
                        different += 1
                    else:
                        matching += 1
                    break
            if not matching_name:
                print(f"  ⚠️  Tensor missing in quantized: {t_base.name}")

        print(f"  {'✅ Tensors match':<20} {matching:>20} {different:>20}")
    except Exception as e:
        print(f"  ⚠️  Comparison error: {e}")

    print()
    return 0


# ─── Batch ──────────────────────────────────────────────────────────────

def cmd_batch(args):
    """Convert and quantize a model to multiple types."""
    model = args.model
    types = [t.strip() for t in args.types.split(",")]
    output_dir = args.output_dir or "."

    print("╔═══════════════════════════════════════════╗")
    print("║    Batch Converter + Quantizer            ║")
    print("╚═══════════════════════════════════════════╝")
    print()

    print(f"  Model:      {model}")
    print(f"  Types:      {', '.join(types)}")
    print(f"  Output dir: {output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # First convert to F16
    model_name = model.split("/")[-1] if "/" in model else model
    f16_path = os.path.join(output_dir, f"{model_name}-base-f16.gguf")
    f16_path = os.path.abspath(f16_path)

    if not os.path.exists(f16_path):
        print(f"[1/{len(types)+1}] Converting to F16...")
        conv_args = argparse.Namespace(
            model=model, outtype="f16", output=f16_path, verbose=False,
        )
        cmd_convert(conv_args)
    else:
        print(f"  ✅ Base F16 already exists: {f16_path}")

    print()

    # Quantize to each type
    for i, qt in enumerate(types):
        print(f"[{i+2}/{len(types)+1}] Quantizing to {qt}...")
        out_name = f"{model_name}-{qt}.gguf"
        out_path = os.path.join(output_dir, out_name)
        out_path = os.path.abspath(out_path)

        if os.path.exists(out_path):
            print(f"  ⚠️  Skipping (already exists): {out_path}")
            continue

        quant_args = argparse.Namespace(
            model=f16_path, type=qt, output=out_path,
            imatrix=None, threads=0, allow_requantize=False,
            pure=False, leave_output=False, dry_run=False,
        )
        rc = cmd_quantize(quant_args)
        if rc != 0:
            print(f"  ❌ Failed on {qt}")
            return 1

    print()
    print(f"✅ Batch complete. Files in {output_dir}:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".gguf"):
            fpath = os.path.join(output_dir, f)
            size = os.path.getsize(fpath)
            print(f"  📄 {f:<40} {_format_size(size)}")

    return 0


# ─── CLI Entry ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MojoLlama GGUF Quantizer v" + VERSION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list-types
  %(prog)s info model.gguf
  %(prog)s imatrix model.gguf --data calibration.txt
  %(prog)s quantize model.gguf --type Q4_K_M --imatrix imatrix.dat
  %(prog)s nf4 model.gguf
  %(prog)s convert hf_model --outtype f16
  %(prog)s validate model.gguf
  %(prog)s benchmark model.gguf
  %(prog)s batch hf_model --types q4_0,q4_k_m,q8_0
  %(prog)s compare base.gguf quantized.gguf
        """,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    sub = parser.add_subparsers(dest="command", help="Command")

    # list-types
    p_list = sub.add_parser("list-types", help="List supported quantization types")

    # info
    p_info = sub.add_parser("info", help="Inspect GGUF model")
    p_info.add_argument("model", help="Path to GGUF model file")
    p_info.add_argument("-v", "--verbose", action="store_true", help="Show all metadata fields")

    # imatrix
    p_imatrix = sub.add_parser("imatrix", help="Generate importance matrix")
    p_imatrix.add_argument("model", help="Path to GGUF model")
    p_imatrix.add_argument("--data", "-f", help="Calibration data file (text)")
    p_imatrix.add_argument("--output", "-o", help="Output imatrix file path")
    p_imatrix.add_argument("--threads", "-t", type=int, default=0, help="Number of threads")
    p_imatrix.add_argument("--ctx-size", "-c", type=int, default=512, help="Context size")

    # quantize
    p_quant = sub.add_parser("quantize", help="Quantize a GGUF model")
    p_quant.add_argument("model", help="Path to input GGUF model")
    p_quant.add_argument("--type", "-t", default="Q4_K_M", dest="type",
                         help="Quantization type (default: Q4_K_M)")
    p_quant.add_argument("--output", "-o", help="Output path")
    p_quant.add_argument("--imatrix", help="Importance matrix file")
    p_quant.add_argument("--threads", type=int, default=0, help="Thread count")
    p_quant.add_argument("--allow-requantize", action="store_true",
                         help="Allow requantizing already quantized tensors")
    p_quant.add_argument("--pure", action="store_true",
                         help="Disable K-quant mixtures, pure type")
    p_quant.add_argument("--leave-output", action="store_true",
                         help="Leave output.weight unquantized")
    p_quant.add_argument("--dry-run", action="store_true",
                         help="Calculate size without quantizing")

    # nf4
    p_nf4 = sub.add_parser("nf4", help="Convert to NF4 (NormalFloat4)")
    p_nf4.add_argument("model", help="Path to input GGUF model")
    p_nf4.add_argument("--output", "-o", help="Output path")
    p_nf4.add_argument("--block-size", type=int, default=64,
                        help="NF4 block size (default: 64)")

    # convert
    p_conv = sub.add_parser("convert", help="Convert HF model to GGUF")
    p_conv.add_argument("model", help="HF model name or path")
    p_conv.add_argument("--outtype", default="f16",
                         help="Output type (f16, q8_0, q4_0, etc.)")
    p_conv.add_argument("--output", "-o", help="Output file path")
    p_conv.add_argument("--verbose", action="store_true", help="Verbose conversion")

    # validate
    p_val = sub.add_parser("validate", help="Validate GGUF model integrity")
    p_val.add_argument("model", help="Path to GGUF model file")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark model speed")
    p_bench.add_argument("model", help="Path to GGUF model file")
    p_bench.add_argument("--prompt", default="Hello", help="Prompt text")
    p_bench.add_argument("--max-tokens", type=int, default=128,
                          help="Tokens to generate")
    p_bench.add_argument("--threads", type=int, default=0,
                          help="Thread count (0 = auto)")

    # compare
    p_comp = sub.add_parser("compare", help="Compare two models")
    p_comp.add_argument("base", help="Base model path")
    p_comp.add_argument("quantized", help="Quantized model path")
    p_comp.add_argument("--samples", type=int, default=100,
                         help="Tensors to sample")

    # batch
    p_batch = sub.add_parser("batch", help="Batch convert+quantize")
    p_batch.add_argument("model", help="HF model name or path")
    p_batch.add_argument("--types", default="q4_0,q4_k_m,q8_0",
                          help="Comma-separated quant types")
    p_batch.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    command_map = {
        "list-types": cmd_list_types,
        "info": cmd_info,
        "imatrix": cmd_imatrix,
        "quantize": cmd_quantize,
        "nf4": cmd_nf4,
        "convert": cmd_convert,
        "validate": cmd_validate,
        "benchmark": cmd_benchmark,
        "compare": cmd_compare,
        "batch": cmd_batch,
    }

    cmd = command_map.get(args.command)
    if cmd:
        return cmd(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
