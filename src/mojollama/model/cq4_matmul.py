"""
C-backed Q4_0 quantized matmul for GGUF models.
Dequantizes weights at load time via gguf, then re-quantizes to canonical
Q4_0 format for the C kernel. Zero float32 allocation during inference.

The double conversion is needed because GGUF's internal byte layout differs
from the canonical Q4_0 block format used by the C kernel.
"""
import ctypes
import numpy as np
import os
from gguf.quants import Q4_0
from gguf.constants import GGMLQuantizationType

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        lib_path = os.path.join(os.path.dirname(__file__), 'libq4matmul.so')
        _lib = ctypes.CDLL(lib_path)
        _lib.q4_matmul_forward_t.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        _lib.q4_matmul_forward_t.restype = None
    return _lib


def _canonical_q4_bytes(tensor) -> bytes:
    """Convert a GGUF Q4_0 tensor to canonical Q4_0 bytes for C kernel.

    Step 1: gguf.dequantize(tensor) → float32
    Step 2: Q4_0.quantize_blocks(float32) → canonical Q4_0 uint8 array
    """
    from gguf.quants import Q4_0
    from gguf.constants import GGMLQuantizationType
    # The GGUF data shape is (out_rows, N) where N = type bytes
    # After dequantize, shape is (out_rows, in_cols)
    # We need to reshape to (n_blocks, block_size) for quantize_blocks
    import gguf
    arr = gguf.dequantize(tensor.data, tensor.tensor_type)  # (out_rows, in_cols)

    # Q4_0.block_size = 32, need arr as (total_values,) then reshape to (n_blocks, 32)
    out_rows, in_cols = arr.shape
    total_vals = out_rows * in_cols
    n_blocks = (total_vals + 31) // 32
    flat = arr.ravel()
    if len(flat) < n_blocks * 32:
        flat = np.pad(flat, (0, n_blocks * 32 - len(flat)))

    blocks_2d = flat.reshape(n_blocks, 32)
    q4_blocks = Q4_0.quantize_blocks(blocks_2d)  # (n_blocks, 18) uint8

    # Reshape to (out_rows, -1) — one row per output feature
    blocks_per_row = (in_cols + 31) // 32
    return q4_blocks.reshape(out_rows, blocks_per_row * 18).tobytes()


class CQ4Matmul:
    """Q4_0 matmul using C kernel. Zero float32 weight allocation during inference."""

    def __init__(self, raw_bytes: bytes, out_rows: int, in_cols: int):
        self.out_rows = out_rows
        self.in_cols = in_cols
        self._raw = (ctypes.c_uint8 * len(raw_bytes)).from_buffer_copy(raw_bytes)
        self._lib = _get_lib()

    @classmethod
    def from_gguf_tensor(cls, tensor) -> "CQ4Matmul":
        """Create from a GGUF reader tensor.

        GGUF shape is [in_cols, out_rows] but dequantized result is (out_rows, in_cols).
        """
        arr = __import__('gguf').dequantize(tensor.data, tensor.tensor_type)
        out_rows, in_cols = arr.shape
        raw_bytes = _canonical_q4_bytes(tensor)
        return cls(raw_bytes, out_rows, in_cols)

    def forward_t(self, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ W.T. x: (batch, in_cols) or (in_cols,)."""
        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(1, -1)
        batch = x.shape[0]
        out = np.zeros((batch, self.out_rows), dtype=np.float32)
        self._lib.q4_matmul_forward_t(
            self._raw,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.out_rows, self.in_cols, batch,
        )
        return out.reshape(-1) if is_1d else out
