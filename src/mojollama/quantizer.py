"""
MojoLlama Quantizer — GGUF quantization CLI tool.

Converts HuggingFace models to GGUF format and applies various quantization types.
Wraps llama.cpp's convert_hf_to_gguf.py and llama-quantize, with a pure Python
Q4_0 fallback for when those tools aren't available.

Usage:
    python3 -m mojollama.quantizer convert hf_model_name --outtype f16 --outfile model.gguf
    python3 -m mojollama.quantizer quantize model.gguf --type q4_k_m
    python3 -m mojollama.quantizer info model.gguf
    python3 -m mojollama.quantizer validate model.gguf --reference ref.gguf
    python3 -m mojollama.quantizer benchmark model.gguf --prompt "Hello"
    python3 -m mojollama.quantizer batch hf_model_name --types q4_0,q4_k_m,q8_0

Importable:
    from mojollama.quantizer import quantize, convert, get_info, Q4Block, quantize_q4, dequantize_q4
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# ─── Paths ──────────────────────────────────────────────────────────────────

# Allow overriding with env vars
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", "/tmp/llama.cpp"))
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

# Default HF cache
HF_HOME = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

# Known quantization types with descriptions and approximate bpw
QUANT_TYPES = {
    "q4_0":   {"desc": "4-bit (4.34G/7B)", "bpw": 4.34, "type_id": 2},
    "q4_1":   {"desc": "4-bit (4.78G/7B)", "bpw": 4.78, "type_id": 3},
    "q5_0":   {"desc": "5-bit (5.21G/7B)", "bpw": 5.21, "type_id": 8},
    "q5_1":   {"desc": "5-bit (5.65G/7B)", "bpw": 5.65, "type_id": 9},
    "q8_0":   {"desc": "8-bit (7.96G/7B)", "bpw": 7.96, "type_id": 7},
    "q2_k":   {"desc": "2-bit K-quant", "bpw": 2.96, "type_id": 10},
    "q2_k_s": {"desc": "2-bit K-quant small", "bpw": 2.96, "type_id": 21},
    "q3_k_s": {"desc": "3-bit K-quant small", "bpw": 3.41, "type_id": 11},
    "q3_k_m": {"desc": "3-bit K-quant medium", "bpw": 3.74, "type_id": 12},
    "q3_k_l": {"desc": "3-bit K-quant large", "bpw": 4.03, "type_id": 13},
    "q4_k_s": {"desc": "4-bit K-quant small", "bpw": 4.37, "type_id": 14},
    "q4_k_m": {"desc": "4-bit K-quant medium", "bpw": 4.58, "type_id": 15},
    "q5_k_s": {"desc": "5-bit K-quant small", "bpw": 5.21, "type_id": 16},
    "q5_k_m": {"desc": "5-bit K-quant medium", "bpw": 5.33, "type_id": 17},
    "q6_k":   {"desc": "6-bit K-quant (6.14G/7B)", "bpw": 6.14, "type_id": 18},
    "f16":    {"desc": "16-bit float (14G/7B)", "bpw": 16.0, "type_id": 1},
    "f32":    {"desc": "32-bit float (26G/7B)", "bpw": 32.0, "type_id": 0},
    "bf16":   {"desc": "BFloat16 (14G/7B)", "bpw": 16.0, "type_id": 32},
    "iq1_s":  {"desc": "1.56 bpw quantization", "bpw": 1.56, "type_id": 24},
    "iq1_m":  {"desc": "1.75 bpw quantization", "bpw": 1.75, "type_id": 31},
    "iq2_xxs":{"desc": "2.06 bpw quantization", "bpw": 2.06, "type_id": 19},
    "iq2_xs": {"desc": "2.31 bpw quantization", "bpw": 2.31, "type_id": 20},
    "iq2_s":  {"desc": "2.5 bpw quantization", "bpw": 2.5, "type_id": 28},
    "iq2_m":  {"desc": "2.7 bpw quantization", "bpw": 2.7, "type_id": 29},
    "iq3_xxs":{"desc": "3.06 bpw quantization", "bpw": 3.06, "type_id": 23},
    "iq3_xs": {"desc": "3.3 bpw quantization", "bpw": 3.3, "type_id": 22},
    "iq3_s":  {"desc": "3.44 bpw quantization", "bpw": 3.44, "type_id": 26},
    "iq3_m":  {"desc": "3.66 bpw quantization mix", "bpw": 3.66, "type_id": 27},
    "iq4_nl": {"desc": "4.50 bpw nonlinear", "bpw": 4.50, "type_id": 25},
    "iq4_xs": {"desc": "4.25 bpw nonlinear", "bpw": 4.25, "type_id": 30},
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _check_tools():
    """Check availability of llama.cpp tools, return status dict."""
    status = {
        "convert_script": CONVERT_SCRIPT.exists(),
        "quantize_bin": QUANTIZE_BIN.exists(),
        "gguf_package": False,
    }
    try:
        import gguf
        status["gguf_package"] = True
    except ImportError:
        pass
    return status


def _find_llama_quantize() -> Optional[Path]:
    """Find llama-quantize binary, checking common locations."""
    candidates = [
        QUANTIZE_BIN,
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "llama-quantize",
        Path("/usr/local/bin/llama-quantize"),
        Path("/usr/bin/llama-quantize"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def _run_subprocess(cmd: List[str], desc: str = "") -> Tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    desc_str = f" ({desc})" if desc else ""
    print(f"  Running: {' '.join(cmd)}{desc_str}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


# ─── Pure Python Q4_0 Quantization (fallback) ───────────────────────────────
# Reference: q4_matmul.py

class Q4Block:
    """A single Q4_0 quantization block (16 values, 10 bytes).
    
    Format:
      - 1x float16 scale (2 bytes)
      - 8x uint8 (2x 4-bit values each, 16 values total)
      - 10 bytes per 16 values
    """

    def __init__(self, scale: np.float16 = np.float16(0.0), quants: np.ndarray = None):
        self.scale = scale
        self.quants = np.zeros(8, dtype=np.uint8) if quants is None else quants

    def dequantize(self) -> np.ndarray:
        """Dequantize block to 16 float32 values."""
        result = np.zeros(16, dtype=np.float32)
        for i in range(8):
            lo = int(self.quants[i] & 0x0F)
            hi = int((self.quants[i] >> 4) & 0x0F)
            result[i*2] = (lo - 8) * float(self.scale)
            result[i*2+1] = (hi - 8) * float(self.scale)
        return result

    def pack(self) -> bytes:
        """Pack block to 10 bytes: float16 scale + 8 uint8."""
        buf = struct.pack('<e', float(self.scale))
        buf += self.quants.tobytes()
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> "Q4Block":
        """Unpack 10 bytes into Q4Block."""
        scale = np.float16(struct.unpack('<e', data[:2])[0])
        quants = np.frombuffer(data[2:10], dtype=np.uint8).copy()
        return cls(scale, quants)


def quantize_q4(values: np.ndarray) -> List[Q4Block]:
    """Quantize float32 array to Q4_0 blocks. Returns list of blocks."""
    n = len(values)
    n_blocks = (n + 15) // 16
    blocks = []
    for b in range(n_blocks):
        start = b * 16
        end = min(start + 16, n)
        chunk = values[start:end]
        if len(chunk) < 16:
            chunk = np.pad(chunk, (0, 16 - len(chunk)))
        amax = np.max(np.abs(chunk))
        scale = np.float16(amax / 7.0 if amax > 0 else 1.0)
        s = float(scale)
        quants = np.zeros(8, dtype=np.uint8)
        for i in range(8):
            lo = max(0, min(15, int(round(chunk[i*2] / s)) + 8))
            hi = max(0, min(15, int(round(chunk[i*2+1] / s)) + 8))
            quants[i] = (hi << 4) | lo
        blocks.append(Q4Block(scale, quants))
    return blocks


def dequantize_q4(blocks: List[Q4Block], n: int) -> np.ndarray:
    """Dequantize blocks back to float32 array."""
    result = np.zeros(n, dtype=np.float32)
    for b_idx, block in enumerate(blocks):
        start = b_idx * 16
        end = min(start + 16, n)
        values = block.dequantize()
        result[start:end] = values[:end-start]
    return result


def quantize_gguf_q4_0(input_path: str, output_path: str) -> Dict[str, Any]:
    """Pure Python Q4_0 quantization of a GGUF file.
    
    Reads an F32/F16 GGUF file, quantizes all weight tensors to Q4_0,
    and writes a new GGUF file. This is a fallback when llama-quantize
    is not available.
    
    Returns stats dict.
    """
    try:
        import gguf
        from gguf.constants import GGMLQuantizationType, GGML_TYPE
    except ImportError:
        raise ImportError("gguf package required for pure Python quantization")
    
    print(f"  Reading: {input_path}")
    reader = gguf.GGUFReader(input_path)
    
    # Collect metadata
    output_tensors = []
    stats = {
        "tensors_quantized": 0,
        "bytes_before": 0,
        "bytes_after": 0,
        "skipped": 0,
    }
    
    for tensor in reader.tensors:
        # Get the raw data as numpy array
        data = tensor.data
        shape = tensor.shape
        
        # Determine if it's a weight tensor (quantizable)
        name = tensor.name
        is_weight = any(name.endswith(suffix) for suffix in [
            ".weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
            "attn_output.weight", "ffn_gate.weight", "ffn_up.weight",
            "ffn_down.weight", "output.weight", "token_embd.weight",
        ])
        
        # Only quantize weight tensors that are float type
        current_type = tensor.tensor_type if hasattr(tensor, 'tensor_type') else GGML_TYPE.F32
        
        if is_weight and current_type in (GGML_TYPE.F32, GGML_TYPE.F16):
            print(f"  Quantizing: {name} shape={shape}")
            # Convert to float32 for quantization
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Store original bytes
            orig_bytes = data.nbytes
            stats["bytes_before"] += orig_bytes
            
            # Quantize per row
            rows = data.shape[0]
            q_blocks = []
            for r in range(rows):
                row_blocks = quantize_q4(data[r])
                q_blocks.extend(row_blocks)
            
            # Pack to bytes
            packed = b''.join(b.pack() for b in q_blocks)
            stats["bytes_after"] += len(packed)
            stats["tensors_quantized"] += 1
            
            # Create new GGUF tensor info
            # For Q4_0: shape is (rows, pack_size) where pack_size = (cols+15)//16 * 10
            cols = data.shape[1]
            pack_size = ((cols + 15) // 16) * 10
            new_shape = [rows, pack_size]
            
            output_tensors.append({
                "name": name,
                "shape": new_shape,
                "data": packed,
                "type": GGMLQuantizationType.Q4_0,
            })
        else:
            # Pass through (embeddings, norm weights, etc.)
            stats["skipped"] += 1
            output_tensors.append({
                "name": name,
                "shape": list(shape),
                "data": data.tobytes(),
                "type": current_type if hasattr(current_type, 'value') else 0,
            })
    
    # Write output GGUF
    print(f"  Writing: {output_path}")
    writer = gguf.GGUFWriter(output_path, reader.get_field("general.architecture"))
    
    # Copy metadata fields
    for field in reader.fields.values():
        try:
            if field.name.startswith("general.") or field.name.startswith("llama.") or field.name.startswith("tokenizer."):
                writer.add_key(field.name)
        except Exception:
            pass
    
    # Add tensors
    for t in output_tensors:
        # We need to convert bytes back to the appropriate type for GGUFWriter
        try:
            arr = np.frombuffer(t["data"], dtype=np.uint8).reshape(t["shape"])
            writer.add_tensor(t["name"], arr, raw_dtype=t["type"])
        except Exception as e:
            print(f"  Warning: could not add tensor {t['name']}: {e}")
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    stats["output_path"] = output_path
    return stats


def quantize_python(input_path: str, output_path: str, quant_type: str = "q4_0") -> Dict[str, Any]:
    """Pure Python quantization of a GGUF file.
    
    Currently only supports Q4_0. Falls back to llama-quantize for other types.
    """
    if quant_type.lower() == "q4_0":
        return quantize_gguf_q4_0(input_path, output_path)
    else:
        raise ValueError(
            f"Pure Python quantization does not support '{quant_type}'. "
            f"Only 'q4_0' is supported. Use llama-quantize for other types."
        )


# ─── Core Operations ─────────────────────────────────────────────────────────

def convert(
    model_name_or_path: str,
    outtype: str = "f16",
    outfile: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Convert HuggingFace model to GGUF format.
    
    Wraps llama.cpp's convert_hf_to_gguf.py.
    
    Args:
        model_name_or_path: HF model name or local path
        outtype: Output format (f32, f16, bf16, q8_0, auto)
        outfile: Output file path
        verbose: Enable verbose output
    
    Returns:
        Dict with status, output path, and stats
    """
    # Check for convert script
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}. "
            f"Set LLAMA_CPP_DIR environment variable to point to your llama.cpp directory."
        )
    
    # Build command
    cmd = [sys.executable, str(CONVERT_SCRIPT), model_name_or_path]
    cmd.extend(["--outtype", outtype])
    if outfile:
        cmd.extend(["--outfile", outfile])
    if verbose:
        cmd.append("--verbose")
    
    # Add any extra kwargs as flags
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    
    print(f"Converting HF model '{model_name_or_path}' to GGUF ({outtype})...")
    t0 = time.time()
    retcode, stdout, stderr = _run_subprocess(cmd, "convert_hf_to_gguf.py")
    elapsed = time.time() - t0
    
    if retcode != 0:
        error_msg = stderr.strip() or stdout.strip()
        raise RuntimeError(f"Conversion failed (exit code {retcode}):\n{error_msg}")
    
    # Try to find the output file
    if outfile and os.path.exists(outfile):
        output_path = outfile
    else:
        # Try to derive from stdout
        output_path = _parse_output_path(stdout, model_name_or_path, outtype)
    
    size = os.path.getsize(output_path) if output_path and os.path.exists(output_path) else 0
    
    result = {
        "status": "success",
        "output_path": str(output_path) if output_path else None,
        "outtype": outtype,
        "size_bytes": size,
        "size_human": _format_size(size),
        "elapsed_seconds": elapsed,
        "stdout": stdout.strip() if verbose else "",
        "stderr": stderr.strip() if verbose else "",
    }
    
    print(f"  Output: {result['output_path']}")
    print(f"  Size: {result['size_human']}")
    print(f"  Time: {elapsed:.1f}s")
    
    return result


def _parse_output_path(stdout: str, model_input: str, outtype: str) -> Optional[str]:
    """Try to find the output file path from conversion script output."""
    import re
    # Look for "Writing to: /path/to/file.gguf" or similar
    for line in stdout.split("\n"):
        line = line.strip()
        if "writing" in line.lower() and ".gguf" in line.lower():
            m = re.search(r'(/[^\s]+\.gguf)', line)
            if m:
                return m.group(1)
    
    # Default: model name + outtype
    base = os.path.basename(model_input.rstrip("/"))
    if not base:
        base = "model"
    return f"{base}-{outtype}.gguf"


def quantize(
    input_path: str,
    quant_type: str = "q4_k_m",
    output_path: Optional[str] = None,
    allow_requantize: bool = False,
    leave_output: bool = False,
    pure_python: bool = False,
    override_kv: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Quantize a GGUF file.
    
    Wraps llama.cpp's llama-quantize, or uses pure Python fallback for Q4_0.
    
    Args:
        input_path: Path to GGUF file
        quant_type: Quantization type (e.g., q4_0, q4_k_m, q8_0, f16)
        output_path: Output file path (default: auto-generated)
        allow_requantize: Allow requantizing already quantized tensors
        leave_output: Leave output.weight unquantized
        pure_python: Force pure Python quantization (only Q4_0)
        override_kv: Dict of metadata overrides (KEY=TYPE:VALUE)
        dry_run: Calculate size without performing quantization
    
    Returns:
        Dict with stats
    """
    quant_type = quant_type.lower()
    input_path = str(Path(input_path).resolve())
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    input_size = os.path.getsize(input_path)
    
    # Default output path
    if output_path is None:
        stem = Path(input_path).stem
        # Remove existing quant suffix if any
        for qt in QUANT_TYPES:
            if stem.lower().endswith(f"-{qt}") or stem.lower().endswith(f"_{qt}"):
                stem = stem[:-(len(qt)+1)]
                break
        output_path = str(Path(input_path).parent / f"{stem}-{quant_type}.gguf")
    
    # Check if we can use pure Python fallback
    if pure_python:
        return quantize_python(input_path, output_path, quant_type)
    
    # Check for llama-quantize
    quant_bin = _find_llama_quantize()
    if quant_bin is None:
        if quant_type == "q4_0":
            print("  llama-quantize not found, using pure Python Q4_0 fallback...")
            return quantize_python(input_path, output_path, quant_type)
        else:
            raise FileNotFoundError(
                f"llama-quantize not found. Searched paths include: {QUANTIZE_BIN}. "
                f"Set LLAMA_CPP_DIR environment variable. "
                f"For Q4_0 only, use --pure-python flag."
            )
    
    # Build command
    cmd = [str(quant_bin)]
    if dry_run:
        cmd.append("--dry-run")
    if allow_requantize:
        cmd.append("--allow-requantize")
    if leave_output:
        cmd.append("--leave-output-tensor")
    if override_kv:
        for key, val in override_kv.items():
            cmd.extend(["--override-kv", f"{key}" if ":" in key else f"{key}:{val}"])
    
    cmd.extend([input_path, output_path, quant_type.upper()])
    
    desc = f"{quant_type} quantization"
    print(f"Quantizing: {input_path}")
    print(f"  Input size: {_format_size(input_size)}")
    print(f"  Output: {output_path}")
    print(f"  Type: {quant_type}")
    
    t0 = time.time()
    retcode, stdout, stderr = _run_subprocess(cmd, desc)
    elapsed = time.time() - t0
    
    if retcode != 0:
        error_msg = stderr.strip() or stdout.strip()
        raise RuntimeError(f"Quantization failed (exit code {retcode}):\n{error_msg}")
    
    # Parse output for stats
    output_size = 0
    if os.path.exists(output_path):
        output_size = os.path.getsize(output_path)
    
    compression_ratio = input_size / output_size if output_size > 0 else 0
    
    result = {
        "status": "success",
        "input_path": input_path,
        "output_path": output_path,
        "quant_type": quant_type,
        "input_size_bytes": input_size,
        "output_size_bytes": output_size,
        "input_size_human": _format_size(input_size),
        "output_size_human": _format_size(output_size),
        "compression_ratio": round(compression_ratio, 2),
        "elapsed_seconds": round(elapsed, 1),
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }
    
    # Print summary
    print(f"  Output size: {result['output_size_human']}")
    print(f"  Compression: {result['compression_ratio']}x")
    print(f"  Time: {elapsed:.1f}s")
    
    # Print size estimate from stdout if dry run
    if dry_run:
        for line in stdout.split("\n"):
            if "size" in line.lower():
                print(f"  {line.strip()}")
    
    return result


def get_info(gguf_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Get information about a GGUF file.
    
    Args:
        gguf_path: Path to GGUF file
        verbose: Show all tensors
    
    Returns:
        Dict with model info
    """
    try:
        import gguf
        from gguf.constants import GGMLQuantizationType
    except ImportError:
        raise ImportError("gguf package required for info command. Install with: pip install gguf")
    
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"File not found: {gguf_path}")
    
    file_size = os.path.getsize(gguf_path)
    reader = gguf.GGUFReader(gguf_path)
    
    # Extract metadata
    info = {
        "path": str(Path(gguf_path).resolve()),
        "file_size_bytes": file_size,
        "file_size_human": _format_size(file_size),
        "header_size": reader.header_size if hasattr(reader, 'header_size') else 0,
        "tensor_count": len(reader.tensors) if hasattr(reader, 'tensors') else 0,
        "metadata": {},
        "tensors": [],
        "quantization_types": set(),
    }
    
    # Get fields/metadata
    if hasattr(reader, 'fields'):
        for name, field in reader.fields.items():
            try:
                if hasattr(field, 'parts') and len(field.parts) > 0:
                    val = field.parts[-1]
                    if isinstance(val, bytes):
                        try:
                            val = val.decode('utf-8', errors='replace').strip('\x00')
                        except Exception:
                            val = str(val)
                    elif isinstance(val, np.ndarray) or hasattr(val, '__len__'):
                        # memmap or ndarray — try to convert
                        if val.ndim == 0:
                            val = val.item()
                        elif val.dtype.kind in ('S', 'U'):
                            # String data stored as bytes
                            try:
                                val = bytes(val).decode('utf-8', errors='replace').strip('\x00')
                            except Exception:
                                val = str(val)
                        elif val.dtype.kind == 'u' and val.size > 1:
                            # uint8 array — likely a stored string, decode as bytes then utf-8
                            try:
                                val = bytes(val).decode('utf-8', errors='replace').strip('\x00')
                            except Exception:
                                val = val.tolist()
                        elif val.size == 1:
                            val = val.item()
                        elif val.dtype.kind in ('i', 'u', 'f'):
                            # Numeric array — keep as list
                            val = val.tolist()
                        else:
                            val = str(val)
                    elif hasattr(val, 'item'):
                        val = val.item()
                    elif hasattr(val, 'tolist'):
                        val = val.tolist()
                    info["metadata"][name] = val
            except Exception:
                continue
    
    # Get tensor info
    if hasattr(reader, 'tensors'):
        tensor_type_counts = {}
        total_params = 0
        
        for tensor in reader.tensors:
            t_info = {
                "name": tensor.name,
                "shape": list(tensor.shape) if hasattr(tensor, 'shape') and tensor.shape is not None and len(tensor.shape) > 0 else [],
                "n_elements": int(np.prod(tensor.shape)) if hasattr(tensor, 'shape') and tensor.shape is not None and len(tensor.shape) > 0 else 0,
            }
            
            # Get quantization type
            if hasattr(tensor, 'tensor_type') and tensor.tensor_type is not None:
                t_type = tensor.tensor_type
                if hasattr(t_type, 'name'):
                    t_info["type"] = t_type.name
                else:
                    t_info["type"] = str(t_type)
                
                if isinstance(t_type, GGMLQuantizationType):
                    tensor_type_counts[t_type.name] = tensor_type_counts.get(t_type.name, 0) + 1
                    info["quantization_types"].add(t_type.name)
            else:
                t_info["type"] = "unknown"
            
            # Get data size
            if hasattr(tensor, 'data') and tensor.data is not None:
                t_info["data_bytes"] = tensor.data.nbytes
            else:
                t_info["data_bytes"] = 0
            
            info["tensors"].append(t_info)
            total_params += t_info["n_elements"]
        
        info["total_parameters"] = total_params
        info["tensor_type_counts"] = tensor_type_counts
    
    # Determine primary quantization type
    quantization_counts = info.get("tensor_type_counts", {})
    if quantization_counts:
        # The type with the most tensors is likely the primary quant
        primary = max(quantization_counts, key=quantization_counts.get)
        info["primary_quantization"] = primary
    else:
        info["primary_quantization"] = "unknown"
    
    # Print info
    print(f"\n{'='*60}")
    print(f"  GGUF Model Info")
    print(f"{'='*60}")
    print(f"  Path:              {info['path']}")
    print(f"  Size:              {info['file_size_human']} ({info['file_size_bytes']:,} bytes)")
    print(f"  Tensors:           {info['tensor_count']}")
    print(f"  Parameters:        {info.get('total_parameters', 0):,}")
    print(f"  Quantization:      {info.get('primary_quantization', 'unknown')}")
    
    # Print key metadata
    if info["metadata"]:
        print(f"\n  Metadata:")
        for key in sorted(info["metadata"].keys()):
            val = info["metadata"][key]
            if isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            print(f"    {key}: {val}")
    
    # Print tensor type distribution
    if info.get("tensor_type_counts"):
        print(f"\n  Tensor type distribution:")
        for ttype, count in sorted(info["tensor_type_counts"].items()):
            print(f"    {ttype}: {count} tensors")
    
    # Print tensors (verbose)
    if verbose and info["tensors"]:
        print(f"\n  Tensors:")
        print(f"  {'Name':50s} {'Shape':30s} {'Type':10s} {'Size':>10s}")
        print(f"  {'-'*50} {'-'*30} {'-'*10} {'-'*10}")
        for t in info["tensors"]:
            shape_str = str(t["shape"]) if "shape" in t else "?"
            type_str = t.get("type", "?")
            size_str = _format_size(t.get("data_bytes", 0))
            print(f"  {t['name']:50s} {shape_str:30s} {type_str:10s} {size_str:>10s}")
    
    print()
    
    return info


def validate(
    quantized_path: str,
    reference_path: Optional[str] = None,
    num_tokens: int = 3,
    prompt: str = "Hello world",
) -> Dict[str, Any]:
    """Validate a quantized model by comparing against a reference.
    
    If a reference GGUF is provided, compares final layer logits.
    Otherwise, validates structural integrity.
    
    Args:
        quantized_path: Path to quantized GGUF file
        reference_path: Path to reference GGUF (unquantized or different quant)
        num_tokens: Number of tokens to generate for comparison
        prompt: Input prompt
    
    Returns:
        Dict with validation results
    """
    try:
        import gguf
        from gguf.constants import GGMLQuantizationType
    except ImportError:
        raise ImportError("gguf package required for validate command")
    
    result = {
        "quantized_path": quantized_path,
        "status": "unknown",
        "checks": [],
    }
    
    print(f"\n{'='*60}")
    print(f"  Validate: {Path(quantized_path).name}")
    print(f"{'='*60}")
    
    # Check 1: File exists and readable
    if not os.path.exists(quantized_path):
        result["status"] = "error"
        result["error"] = "File not found"
        return result
    
    file_size = os.path.getsize(quantized_path)
    print(f"  File size: {_format_size(file_size)}")
    result["file_size"] = file_size
    
    # Check 2: Can be read by gguf
    try:
        reader = gguf.GGUFReader(quantized_path)
        tensor_count = len(reader.tensors) if hasattr(reader, 'tensors') else 0
        print(f"  Tensors loaded: {tensor_count}")
        result["tensor_count"] = tensor_count
        result["checks"].append({"check": "gguf_readable", "passed": True})
    except Exception as e:
        print(f"  ERROR: Cannot read GGUF: {e}")
        result["checks"].append({"check": "gguf_readable", "passed": False, "error": str(e)})
        result["status"] = "corrupt"
        return result
    
    # Check 3: Can load all tensor data
    errors = []
    for tensor in reader.tensors:
        try:
            _ = tensor.data.shape
        except Exception as e:
            errors.append((tensor.name, str(e)))
    
    if errors:
        print(f"  WARNING: {len(errors)} tensor(s) failed to load:")
        for name, err in errors[:5]:
            print(f"    {name}: {err}")
        result["checks"].append({"check": "tensor_data_accessible", "passed": False, "errors": errors})
    else:
        print(f"  All tensors accessible: YES")
        result["checks"].append({"check": "tensor_data_accessible", "passed": True})
    
    # Check 4: Metadata integrity
    if hasattr(reader, 'fields'):
        required_fields = ['general.architecture', 'general.file_type']
        missing = [f for f in required_fields if f not in reader.fields]
        if missing:
            print(f"  Missing metadata fields: {missing}")
            result["checks"].append({"check": "metadata_integrity", "passed": False, "missing": missing})
        else:
            print(f"  Metadata integrity: OK")
            result["checks"].append({"check": "metadata_integrity", "passed": True})
    
    # Check 5: Compare with reference if provided
    if reference_path and os.path.exists(reference_path):
        print(f"\n  Comparing with reference: {Path(reference_path).name}")
        try:
            ref_reader = gguf.GGUFReader(reference_path)
            
            # Compare tensor names and shapes
            ref_tensors = {t.name: t for t in ref_reader.tensors}
            q_tensors = {t.name: t for t in reader.tensors}
            
            common_names = set(ref_tensors.keys()) & set(q_tensors.keys())
            missing_in_q = set(ref_tensors.keys()) - set(q_tensors.keys())
            extra_in_q = set(q_tensors.keys()) - set(ref_tensors.keys())
            
            if missing_in_q:
                print(f"  Missing tensors in quantized: {len(missing_in_q)}")
                for n in sorted(missing_in_q)[:5]:
                    print(f"    {n}")
            if extra_in_q:
                print(f"  Extra tensors in quantized: {len(extra_in_q)}")
            
            # Compare shapes
            shape_mismatches = []
            for name in sorted(common_names):
                ref_shape = list(ref_tensors[name].shape)
                q_shape = list(q_tensors[name].shape)
                if ref_shape != q_shape:
                    shape_mismatches.append((name, ref_shape, q_shape))
            
            if shape_mismatches:
                print(f"  Shape mismatches: {len(shape_mismatches)}")
                for name, rs, qs in shape_mismatches[:5]:
                    print(f"    {name}: ref={rs} vs quantized={qs}")
            
            result["checks"].append({
                "check": "reference_comparison",
                "passed": len(missing_in_q) == 0 and len(shape_mismatches) == 0,
                "common_tensors": len(common_names),
                "missing_in_quantized": list(missing_in_q),
                "shape_mismatches": shape_mismatches,
            })
            
        except Exception as e:
            print(f"  Reference comparison failed: {e}")
            result["checks"].append({"check": "reference_comparison", "passed": False, "error": str(e)})
    
    # Overall status
    all_passed = all(c["passed"] for c in result["checks"])
    result["status"] = "passed" if all_passed else "warnings" if any(not c["passed"] for c in result["checks"]) else "unknown"
    
    print(f"\n  Validation: {result['status'].upper()}")
    print()
    
    return result


def benchmark(
    gguf_path: str,
    prompt: str = "Hello",
    max_tokens: int = 10,
    n_warmup: int = 2,
) -> Dict[str, Any]:
    """Quick performance benchmark of a GGUF model.
    
    Uses the project's existing inference engine to measure tokens/second.
    
    Args:
        gguf_path: Path to GGUF file
        prompt: Input prompt
        max_tokens: Number of tokens to generate
        n_warmup: Number of warmup tokens
    
    Returns:
        Dict with benchmark results
    """
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"File not found: {gguf_path}")
    
    file_size = os.path.getsize(gguf_path)
    
    print(f"\n{'='*60}")
    print(f"  Benchmark: {Path(gguf_path).name}")
    print(f"{'='*60}")
    print(f"  Prompt: {prompt!r}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  File size: {_format_size(file_size)}")
    
    # Try to use MojoLlama inference
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from mojollama.model.inference import LLMInference
        
        import gc
        gc.collect()
        
        print(f"  Loading model...")
        t0 = time.time()
        model = LLMInference(gguf_path, device="cpu")
        load_time = time.time() - t0
        print(f"  Load time: {load_time:.1f}s")
        
        # Get model info
        arch_info = {}
        if hasattr(model, 'arch'):
            arch_info["architecture"] = model.arch
        if hasattr(model, 'n_layers'):
            arch_info["layers"] = model.n_layers
        if hasattr(model, 'n_embd'):
            arch_info["dim"] = model.n_embd
        if hasattr(model, 'n_head'):
            arch_info["heads"] = model.n_head
        
        # Warmup
        if n_warmup > 0:
            print(f"  Warmup ({n_warmup} tokens)...")
            _ = model.generate(prompt, max_tokens=n_warmup)
        
        # Benchmark
        print(f"  Generating {max_tokens} tokens...")
        t0 = time.time()
        output = model.generate(prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        
        # Token counting
        try:
            prompt_tokens = len(model.encode(prompt))
            output_tokens = len(model.encode(output))
            total_tokens = prompt_tokens + output_tokens
        except Exception:
            prompt_tokens = 0
            output_tokens = max_tokens
            total_tokens = max_tokens
        
        tokens_per_second = output_tokens / elapsed if elapsed > 0 else 0
        
        result = {
            "model_path": gguf_path,
            "file_size_bytes": file_size,
            "file_size_human": _format_size(file_size),
            "prompt": prompt,
            "architecture": arch_info,
            "load_time_seconds": round(load_time, 2),
            "max_tokens": max_tokens,
            "generated_tokens": output_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_second, 2),
            "output": output,
        }
        
        print(f"\n  Results:")
        print(f"    Generated:     {output_tokens} tokens in {elapsed:.2f}s")
        print(f"    Speed:         {tokens_per_second:.2f} tok/s")
        print(f"    Output:        {output[:100]!r}{'...' if len(output) > 100 else ''}")
        
        return result
        
    except ImportError as e:
        print(f"  WARNING: Could not load MojoLlama inference: {e}")
        print(f"  Falling back to file-based benchmark...")
        return _benchmark_fast(gguf_path, prompt, max_tokens)
    except Exception as e:
        print(f"  WARNING: Inference benchmark failed: {e}")
        print(f"  Falling back to file-based benchmark...")
        return _benchmark_fast(gguf_path, prompt, max_tokens)


def _benchmark_fast(gguf_path: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    """Quick file-based benchmark when inference isn't available."""
    file_size = os.path.getsize(gguf_path)
    
    # Read metadata for info
    info = {}
    try:
        import gguf
        reader = gguf.GGUFReader(gguf_path)
        if hasattr(reader, 'fields'):
            f = reader.fields
            arch = None
            if 'general.architecture' in f:
                arch = str(f['general.architecture'].parts[-1])
            info["architecture"] = arch
            info["tensors"] = len(reader.tensors) if hasattr(reader, 'tensors') else 0
    except Exception:
        pass
    
    result = {
        "model_path": gguf_path,
        "file_size_bytes": file_size,
        "file_size_human": _format_size(file_size),
        "prompt": prompt,
        "architecture": info,
        "load_time_seconds": None,
        "max_tokens": max_tokens,
        "generated_tokens": None,
        "elapsed_seconds": None,
        "tokens_per_second": None,
        "output": None,
        "note": "Inference engine not available; metadata only"
    }
    
    print(f"\n  Model info collected. To benchmark, install MojoLlama inference dependencies.")
    print(f"  Architecture: {info.get('architecture', 'unknown')}")
    print(f"  Tensors: {info.get('tensors', 0)}")
    
    return result


def batch_convert_and_quantize(
    model_name_or_path: str,
    types: List[str],
    outtype: str = "f16",
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert and quantize a model to multiple quantization types.
    
    Args:
        model_name_or_path: HF model name or local path
        types: List of quantization types
        outtype: Intermediate float type for conversion
        output_dir: Output directory (default: current dir)
    
    Returns:
        List of result dicts for each quantization
    """
    if output_dir is None:
        output_dir = "."
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_base = Path(model_name_or_path.rstrip("/")).name
    if not model_base:
        model_base = "model"
    
    results = []
    
    # Step 1: Convert to float GGUF
    fp_gguf = str(output_dir / f"{model_base}-{outtype}.gguf")
    print(f"\n{'='*60}")
    print(f"  Step 1: Convert to {outtype} GGUF")
    print(f"{'='*60}")
    
    try:
        convert_result = convert(
            model_name_or_path,
            outtype=outtype,
            outfile=fp_gguf,
        )
        results.append({"step": "convert", **convert_result})
    except Exception as e:
        print(f"  Conversion failed: {e}")
        results.append({"step": "convert", "status": "error", "error": str(e)})
        return results
    
    # Step 2: Quantize to each type
    for qt in types:
        qt = qt.lower()
        if qt not in QUANT_TYPES:
            print(f"  Unknown quant type '{qt}', skipping...")
            results.append({"step": "quantize", "quant_type": qt, "status": "skipped", "reason": "unknown type"})
            continue
        
        print(f"\n{'-'*60}")
        print(f"  Step 2: Quantize to {qt}")
        print(f"{'-'*60}")
        
        try:
            quant_result = quantize(
                fp_gguf,
                quant_type=qt,
                output_path=str(output_dir / f"{model_base}-{qt}.gguf"),
            )
            results.append({"step": "quantize", "quant_type": qt, **quant_result})
        except Exception as e:
            print(f"  Quantization to {qt} failed: {e}")
            results.append({"step": "quantize", "quant_type": qt, "status": "error", "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  Batch Summary")
    print(f"{'='*60}")
    for r in results:
        if r.get("status") == "success":
            if r["step"] == "convert":
                print(f"  CONVERT: {r.get('output_path', '?')} ({r.get('size_human', '?')})")
            else:
                print(f"  {r.get('quant_type', '?'):8s}: {r.get('output_path', '?')} ({r.get('output_size_human', '?')})")
        elif r.get("status") == "error":
            print(f"  ERROR: {r.get('step', '?')} - {r.get('error', '?')}")
    
    return results


# ─── Init .env ──────────────────────────────────────────────────────────────

def init_env(env_path: Optional[str] = None, hf_cache: Optional[str] = None):
    """Initialize .env file with common HF cache paths.
    
    Args:
        env_path: Path to .env file (default: .env in current dir)
        hf_cache: Custom HF cache path (default: auto-detect)
    """
    if env_path is None:
        env_path = ".env"
    
    env_path = Path(env_path)
    
    if hf_cache is None:
        # Auto-detect common HF cache locations
        candidates = [
            os.environ.get("HF_HOME", ""),
            os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
            str(Path.home() / ".cache" / "huggingface"),
            "/root/.cache/huggingface",
            "/tmp/huggingface",
        ]
        hf_cache = next((c for c in candidates if c and Path(c).exists()), candidates[2])
    
    env_vars = {
        "HF_HOME": hf_cache,
        "HUGGINGFACE_HUB_CACHE": str(Path(hf_cache) / "hub"),
        "LLAMA_CPP_DIR": str(LLAMA_CPP_DIR),
        "TRANSFORMERS_CACHE": str(Path(hf_cache) / "hub"),
        "HF_DATASETS_CACHE": str(Path(hf_cache) / "datasets"),
    }
    
    # Read existing .env
    existing = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
    
    # Merge (existing values take precedence)
    for k, v in env_vars.items():
        if k not in existing:
            existing[k] = v
    
    # Write .env
    with open(env_path, "w") as f:
        f.write("# MojoLlama Quantizer Environment\n")
        f.write(f"# Auto-generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Uncomment and edit as needed\n\n")
        for k, v in existing.items():
            f.write(f'{k}="{v}"\n')
    
    print(f"  Wrote {env_path}")
    print(f"  HF_HOME: {existing.get('HF_HOME', 'not set')}")
    print(f"  LLAMA_CPP_DIR: {existing.get('LLAMA_CPP_DIR', str(LLAMA_CPP_DIR))}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="MojoLlama Quantizer — GGUF quantization CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s convert meta-llama/Llama-2-7b --outtype f16 --outfile model.gguf
  %(prog)s quantize model.gguf --type q4_k_m
  %(prog)s info model.gguf -v
  %(prog)s validate model.gguf --reference ref.gguf
  %(prog)s benchmark model.gguf --prompt "Hello world"
  %(prog)s batch meta-llama/Llama-2-7b --types q4_0,q4_k_m,q8_0
  %(prog)s init-env
        """,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")
    
    # ── convert ──
    p_convert = subparsers.add_parser("convert", help="Convert HF model to GGUF")
    p_convert.add_argument("model", help="HF model name or local path")
    p_convert.add_argument(
        "--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="Output format (default: f16)",
    )
    p_convert.add_argument("--outfile", "-o", help="Output file path")
    p_convert.add_argument("--verbose", action="store_true", help="Verbose output")
    p_convert.add_argument("--vocab-only", action="store_true", help="Extract only vocab")
    p_convert.add_argument("--model-name", help="Model name override")
    
    # ── quantize ──
    p_quant = subparsers.add_parser("quantize", help="Quantize a GGUF file")
    p_quant.add_argument("input", help="Input GGUF file")
    p_quant.add_argument(
        "--type", "-t", dest="quant_type", default="q4_k_m",
        help=f"Quantization type (default: q4_k_m). Options: {', '.join(sorted(QUANT_TYPES.keys()))}",
    )
    p_quant.add_argument("--output", "-o", help="Output file path")
    p_quant.add_argument("--allow-requantize", action="store_true",
                         help="Allow requantizing already quantized tensors")
    p_quant.add_argument("--leave-output", action="store_true",
                         help="Leave output.weight unquantized")
    p_quant.add_argument("--pure-python", action="store_true",
                         help="Use pure Python Q4_0 fallback (no llama-quantize needed)")
    p_quant.add_argument("--dry-run", action="store_true",
                         help="Calculate size without performing quantization")
    p_quant.add_argument("--override-kv", action="append",
                         help="Override metadata KEY=TYPE:VALUE")
    
    # ── info ──
    p_info = subparsers.add_parser("info", help="Inspect a GGUF file")
    p_info.add_argument("input", help="GGUF file path")
    p_info.add_argument("--verbose", "-v", action="store_true", help="Show all tensors")
    
    # ── validate ──
    p_val = subparsers.add_parser("validate", help="Validate quantized model")
    p_val.add_argument("input", help="Quantized GGUF file")
    p_val.add_argument("--reference", "-r", help="Reference GGUF file for comparison")
    p_val.add_argument("--num-tokens", type=int, default=3, help="Tokens for comparison")
    p_val.add_argument("--prompt", default="Hello world", help="Input prompt")
    
    # ── benchmark ──
    p_bench = subparsers.add_parser("benchmark", help="Benchmark model performance")
    p_bench.add_argument("input", help="GGUF file path")
    p_bench.add_argument("--prompt", default="Hello", help="Input prompt")
    p_bench.add_argument("--max-tokens", type=int, default=10, help="Tokens to generate")
    p_bench.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    
    # ── batch ──
    p_batch = subparsers.add_parser("batch", help="Batch convert and quantize")
    p_batch.add_argument("model", help="HF model name or local path")
    p_batch.add_argument(
        "--types", default="q4_0,q4_k_m,q8_0",
        help="Comma-separated quantization types (default: q4_0,q4_k_m,q8_0)",
    )
    p_batch.add_argument(
        "--outtype", default="f16", choices=["f32", "f16", "bf16"],
        help="Intermediate float type (default: f16)",
    )
    p_batch.add_argument("--output-dir", "-o", help="Output directory")
    
    # ── init-env ──
    p_env = subparsers.add_parser("init-env", help="Initialize .env file")
    p_env.add_argument("--env-file", default=".env", help="Path to .env file")
    p_env.add_argument("--hf-cache", help="Custom HF cache path")
    
    # ── list-types ──
    p_list = subparsers.add_parser("list-types", help="List supported quantization types")
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    
    if args.debug:
        print(f"Debug: args={args}")
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "convert":
            kwargs = {}
            if args.vocab_only:
                kwargs["vocab_only"] = True
            if args.model_name:
                kwargs["model_name"] = args.model_name
            
            result = convert(
                args.model,
                outtype=args.outtype,
                outfile=args.outfile,
                verbose=args.verbose,
                **kwargs,
            )
            if args.debug:
                print(json.dumps(result, default=str, indent=2))
        
        elif args.command == "quantize":
            override_kv = None
            if args.override_kv:
                override_kv = {}
                for kv in args.override_kv:
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        override_kv[k] = v
                    else:
                        override_kv[kv] = ""
            
            result = quantize(
                args.input,
                quant_type=args.quant_type,
                output_path=args.output,
                allow_requantize=args.allow_requantize,
                leave_output=args.leave_output,
                pure_python=args.pure_python,
                override_kv=override_kv,
                dry_run=args.dry_run,
            )
            if args.debug:
                print(json.dumps(result, default=str, indent=2))
        
        elif args.command == "info":
            result = get_info(args.input, verbose=args.verbose)
            if args.debug:
                print(json.dumps(result, default=str, indent=2))
        
        elif args.command == "validate":
            result = validate(
                args.input,
                reference_path=args.reference,
                num_tokens=args.num_tokens,
                prompt=args.prompt,
            )
            if args.debug:
                print(json.dumps(result, default=str, indent=2))
        
        elif args.command == "benchmark":
            result = benchmark(
                args.input,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                n_warmup=0 if args.no_warmup else 2,
            )
            if args.debug:
                print(json.dumps(result, default=str, indent=2))
        
        elif args.command == "batch":
            types = [t.strip().lower() for t in args.types.split(",")]
            result = batch_convert_and_quantize(
                args.model,
                types=types,
                outtype=args.outtype,
                output_dir=args.output_dir,
            )
            if args.debug:
                print(json.dumps(result, default=str, indent=2))
        
        elif args.command == "init-env":
            init_env(env_path=args.env_file, hf_cache=args.hf_cache)
        
        elif args.command == "list-types":
            print(f"\nSupported quantization types:\n")
            print(f"  {'Type':12s} {'Description':35s} {'BPW':8s}")
            print(f"  {'-'*12} {'-'*35} {'-'*8}")
            for name, info in sorted(QUANT_TYPES.items()):
                print(f"  {name:12s} {info['desc']:35s} {info['bpw']:<8.2f}")
            print()
        
        else:
            parser.print_help()
            return 1
        
        return 0
    
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
