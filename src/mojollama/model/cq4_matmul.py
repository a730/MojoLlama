"""
C-backed Q4_0 quantized matmul for GGUF models — AVX2+FMA+OpenMP.

Uses the optimized q4_kernel_omp.so with:
- Full AVX2 SIMD dequant pipeline (15-instruction nibble unpack)
- FMA chain for dot products (1 MUL + 3 FMAs per block)
- Row-outer accumulation (1 write per row, not per block)
- 4-row register blocking variant
- OpenMP multi-threading on Threadripper 3970X (32 physical cores)
- Vectorized RMS norm, SiLU, softmax
- Thread control (get/set OMP threads)
"""
import ctypes
import numpy as np
import os
from pathlib import Path

_kernel_dir = Path(__file__).parent.parent / "kernels"
_lib_omp = None
_lib_batch = None


def _get_omp_lib():
    global _lib_omp
    if _lib_omp is None:
        lib_path = _kernel_dir / "q4_kernel_omp.so"
        _lib_omp = ctypes.CDLL(str(lib_path))
        # Matmul: (w, x, out, n_rows, n_cols, type_size)
        _lib_omp.q4_matmul_omp.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        _lib_omp.q4_matmul_omp.restype = None
        # Blocked variant
        _lib_omp.q4_matmul_omp_blocked.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        _lib_omp.q4_matmul_omp_blocked.restype = None
        # AVX2 single-threaded (start, end rows)
        _lib_omp.q4_matmul_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        _lib_omp.q4_matmul_avx2.restype = None
        # Norm, activation, softmax
        _lib_omp.q4_rms_norm.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        _lib_omp.q4_rms_norm.restype = None
        _lib_omp.q4_silu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib_omp.q4_silu.restype = None
        _lib_omp.q4_softmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib_omp.q4_softmax.restype = None
        # Thread control
        _lib_omp.q4_get_max_threads.argtypes = []
        _lib_omp.q4_get_max_threads.restype = ctypes.c_int
        _lib_omp.q4_set_num_threads.argtypes = [ctypes.c_int]
        _lib_omp.q4_set_num_threads.restype = None
        # Auto-detect: physical cores only (no SMT)
        n_phys = _detect_physical_cores()
        _lib_omp.q4_set_num_threads(n_phys)
    return _lib_omp


def _get_batch_lib():
    global _lib_batch
    if _lib_batch is None:
        lib_path = _kernel_dir / "q4_kernel_batch.so"
        _lib_batch = ctypes.CDLL(str(lib_path))
        MAX_PROJ = 7
        _PP = ctypes.POINTER(ctypes.c_uint8) * MAX_PROJ
        _PF = ctypes.POINTER(ctypes.c_float) * MAX_PROJ
        _PI = ctypes.c_int * MAX_PROJ
        _lib_batch.batch_matmul.argtypes = [
            _PP, ctypes.POINTER(ctypes.c_float), _PF, _PI, _PI, _PI, ctypes.c_int,
        ]
        _lib_batch.batch_matmul.restype = None
    return _lib_batch


def _detect_physical_cores():
    """Detect physical cores, excluding SMT. On Threadripper 3970X: 32."""
    try:
        cores = set()
        import glob
        for p in glob.glob("/sys/devices/system/cpu/cpu*/topology/core_id"):
            try:
                with open(p) as f:
                    cores.add(int(f.read().strip()))
            except (ValueError, OSError):
                pass
        if cores:
            return len(cores)
    except (FileNotFoundError, OSError):
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _canonical_q4_bytes(tensor) -> bytes:
    """Convert a GGUF Q4_0 tensor to canonical Q4_0 bytes for C kernel."""
    from gguf.quants import Q4_0
    import gguf
    arr = gguf.dequantize(tensor.data, tensor.tensor_type)
    out_rows, in_cols = arr.shape
    total_vals = out_rows * in_cols
    n_blocks = (total_vals + 31) // 32
    flat = arr.ravel()
    if len(flat) < n_blocks * 32:
        flat = np.pad(flat, (0, n_blocks * 32 - len(flat)))
    blocks_2d = flat.reshape(n_blocks, 32)
    q4_blocks = Q4_0.quantize_blocks(blocks_2d)
    blocks_per_row = (in_cols + 31) // 32
    return q4_blocks.reshape(out_rows, blocks_per_row * 18).tobytes()


class CQ4Matmul:
    """Q4_0 matmul using C AVX2+FMA+OpenMP kernel.

    Zero float32 weight allocation during inference.
    Uses row-outer accumulation, FMA chain, 4-row register blocking.
    Auto-tunes thread count to physical cores (not logical HT cores).
    """
    TYPE_Q4_0 = 18
    TYPE_Q4_1 = 20

    def __init__(self, raw_bytes: bytes, out_rows: int, in_cols: int,
                 type_size: int = 18):
        self.out_rows = out_rows
        self.in_cols = in_cols
        self.type_size = type_size
        self._raw = (ctypes.c_uint8 * len(raw_bytes)).from_buffer_copy(raw_bytes)
        self._lib = _get_omp_lib()

    @classmethod
    def from_gguf_tensor(cls, tensor, type_size: int = 18) -> "CQ4Matmul":
        arr = __import__('gguf').dequantize(tensor.data, tensor.tensor_type)
        out_rows, in_cols = arr.shape
        raw_bytes = _canonical_q4_bytes(tensor)
        return cls(raw_bytes, out_rows, in_cols, type_size)

    def forward(self, x: np.ndarray, blocked: bool = True) -> np.ndarray:
        """Compute y = x @ W^T. Row-outer + FMA, optionally 4-row blocked."""
        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(1, -1)
        batch = x.shape[0]
        out = np.zeros((batch, self.out_rows), dtype=np.float32)
        fn = self._lib.q4_matmul_omp_blocked if blocked else self._lib.q4_matmul_omp
        for b in range(batch):
            fn(
                self._raw,
                x[b].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out[b].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.out_rows, self.in_cols, self.type_size,
            )
        return out.reshape(-1) if is_1d else out

    def forward_t(self, x: np.ndarray) -> np.ndarray:
        """Legacy API: compute y = x @ W^T with batch support."""
        return self.forward(x, blocked=False)


class CQ4RMSNorm:
    """AVX2+FMA vectorized RMS normalization."""
    def __init__(self):
        self._lib = _get_omp_lib()

    def forward(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        n = len(x)
        self._lib.q4_rms_norm(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )
        return out


class CQ4SiLU:
    """AVX2 vectorized SiLU activation (scalar exp, vectorized multiply)."""
    def __init__(self):
        self._lib = _get_omp_lib()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        self._lib.q4_silu(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x))
        return x


class CQ4Softmax:
    """AVX2 vectorized softmax (vectorized max/reduce, scalar exp)."""
    def __init__(self):
        self._lib = _get_omp_lib()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        self._lib.q4_softmax(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x))
        return x


class CQ4BatchMatmul:
    """Batched Q4_X matmul — up to 7 projections in one OpenMP call.

    Shares input vector across all projections for maximum cache reuse.
    Perfect for fused QKV or QKV+O+gate+up+down.
    """
    TYPE_Q4_0 = 18
    TYPE_Q4_1 = 20
    MAX_PROJ = 7

    def __init__(self, projections: list):
        """Initialize with a list of (raw_bytes, out_rows, in_cols, type_size) tuples."""
        assert len(projections) <= self.MAX_PROJ
        self.projections = []
        lib = _get_omp_lib()
        for raw_bytes, out_rows, in_cols, type_size in projections:
            raw_arr = (ctypes.c_uint8 * len(raw_bytes)).from_buffer_copy(raw_bytes)
            self.projections.append((raw_arr, out_rows, in_cols, type_size))

    @classmethod
    def from_gguf_tensors(cls, tensors: list) -> "CQ4BatchMatmul":
        """Create from a list of GGUF tensors.

        Each tensor is (gguf_tensor, type_size) tuple.
        """
        projections = []
        for tensor, type_size in tensors:
            arr = __import__('gguf').dequantize(tensor.data, tensor.tensor_type)
            out_rows, in_cols = arr.shape
            raw_bytes = _canonical_q4_bytes(tensor)
            projections.append((raw_bytes, out_rows, in_cols, type_size))
        return cls(projections)

    def forward(self, x: np.ndarray) -> list:
        """Compute all projections sharing the same input vector x.

        Returns a list of output arrays, one per projection.
        """
        batch_lib = _get_batch_lib()
        n_proj = len(self.projections)
        w_arr = (ctypes.POINTER(ctypes.c_uint8) * self.MAX_PROJ)()
        out_arr = (ctypes.POINTER(ctypes.c_float) * self.MAX_PROJ)()
        nrows_arr = (ctypes.c_int * self.MAX_PROJ)()
        ncols_arr = (ctypes.c_int * self.MAX_PROJ)()
        ts_arr = (ctypes.c_int * self.MAX_PROJ)()
        outputs = []

        for i, (raw_arr, out_rows, in_cols, type_size) in enumerate(self.projections):
            w_arr[i] = ctypes.cast(raw_arr, ctypes.POINTER(ctypes.c_uint8))
            nrows_arr[i] = out_rows
            ncols_arr[i] = in_cols
            ts_arr[i] = type_size
            out = np.zeros(out_rows, dtype=np.float32)
            outputs.append(out)
            out_arr[i] = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        batch_lib.batch_matmul(
            w_arr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_arr, nrows_arr, ncols_arr, ts_arr, n_proj,
        )
        return outputs


def set_num_threads(n: int):
    """Set OpenMP thread count for C kernels. Use physical cores for best perf."""
    _get_omp_lib().q4_set_num_threads(n)


def get_max_threads() -> int:
    """Get max available OpenMP threads."""
    return _get_omp_lib().q4_get_max_threads()


def get_physical_cores() -> int:
    """Get number of physical CPU cores (excl. SMT)."""
    return _detect_physical_cores()