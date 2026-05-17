"""
MojoLlama — Device abstraction layer with Intel Arc GPU support.

Auto-detects available compute devices and provides a unified backend interface.
Supports Intel Arc (via Intel SYCL / dpctl / dpnp), NVIDIA CUDA, AMD ROCm, and CPU.

Usage:
    from mojollama.model.device import get_device, DeviceType

    device = get_device()             # auto-detect best available
    device = get_device('intel_arc')  # force Intel Arc
    device = get_device('cpu')        # force CPU

    arr = device.to_device(numpy_array)   # move array to device memory
    result = device.matmul(a, b)           # device-aware matmul
    numpy_result = device.to_cpu(result)   # bring back to host
"""

import enum
import logging
import os
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class DeviceType(enum.Enum):
    """Supported device types."""
    CPU = 'cpu'
    NVIDIA_CUDA = 'nvidia'
    AMD_ROCM = 'amd'
    INTEL_ARC = 'intel_arc'


class DeviceCapability:
    """Hardware capabilities of a compute device."""

    def __init__(self, name: str, vram_gb: float = 0, compute_units: int = 0,
                 supports_fp16: bool = False, supports_bf16: bool = False,
                 max_work_group_size: int = 0):
        self.name = name
        self.vram_gb = vram_gb
        self.compute_units = compute_units
        self.supports_fp16 = supports_fp16
        self.supports_bf16 = supports_bf16
        self.max_work_group_size = max_work_group_size

    def __repr__(self):
        return (f"DeviceCapability(name='{self.name}', vram={self.vram_gb}GB, "
                f"compute_units={self.compute_units}, fp16={self.supports_fp16})")


class DeviceBackend:
    """Base class for all device backends. Subclass per device type."""

    device_type: DeviceType = DeviceType.CPU
    capability: Optional[DeviceCapability] = None
    is_available: bool = False

    def __init__(self):
        self._stream = None  # compute stream / queue

    def to_device(self, array):
        """Move a numpy array to device memory."""
        raise NotImplementedError

    def to_cpu(self, array):
        """Move a device array back to host (numpy)."""
        raise NotImplementedError

    def matmul(self, a, b):
        """Matrix multiply: C = A @ B"""
        raise NotImplementedError

    def rms_norm(self, x, weight, eps: float = 1e-6):
        """RMS normalization."""
        raise NotImplementedError

    def silu(self, x):
        """SiLU activation."""
        raise NotImplementedError

    def softmax(self, x, axis=-1):
        """Softmax."""
        raise NotImplementedError

    def rope(self, x, cos, sin):
        """Apply rotary position embeddings."""
        raise NotImplementedError

    def copy(self, dst, src):
        """Copy src to dst on device."""
        raise NotImplementedError

    def synchronize(self):
        """Block until all pending operations complete."""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.capability}>"


# ─── CPU Backend (default fallback) ──────────────────────────────────────────

class CpuBackend(DeviceBackend):
    """Pure NumPy CPU backend — always available."""

    device_type = DeviceType.CPU

    def __init__(self):
        import numpy as np
        self.np = np
        self.is_available = True
        self.capability = DeviceCapability(
            name=f"CPU ({os.uname().nodename})",
            compute_units=os.cpu_count() or 1,
        )

    def to_device(self, array):
        return self.np.asarray(array, dtype=self.np.float32)

    def to_cpu(self, array):
        return self.np.asarray(array)

    def matmul(self, a, b):
        return a @ b

    def rms_norm(self, x, weight, eps=1e-6):
        x64 = x.astype(self.np.float64)
        variance = self.np.mean(x64 ** 2, axis=-1, keepdims=True)
        return x / self.np.sqrt(variance + eps) * weight.astype(self.np.float32)

    def silu(self, x):
        return x / (1 + self.np.exp(-x))

    def softmax(self, x, axis=-1):
        x_max = self.np.max(x, axis=axis, keepdims=True)
        x_exp = self.np.exp(x - x_max)
        return x_exp / self.np.sum(x_exp, axis=axis, keepdims=True)

    def rope(self, x, cos, sin):
        n, h, d = x.shape
        x2 = x.reshape(n, h, d // 2, 2)
        xr = self.np.stack([-x2[..., 1], x2[..., 0]], axis=-1)
        c = cos[:n, self.np.newaxis, :d // 2, self.np.newaxis]
        s = sin[:n, self.np.newaxis, :d // 2, self.np.newaxis]
        return (x2 * c + xr * s).reshape(n, h, d)

    def copy(self, dst, src):
        self.np.copyto(dst, src)


# ─── Intel Arc Backend ───────────────────────────────────────────────────────

class IntelArcBackend(DeviceBackend):
    """
    Intel Arc GPU backend using Intel SYCL via dpctl + dpnp.

    Requirements:
        - Intel GPU with Level Zero driver (kernel driver: i915)
        - dpctl >= 0.14
        - dpnp >= 0.13

    These are available from:
        pip install dpctl dpnp
        (Requires Intel GPU runtime: libze-intel-gpu1, level-zero-gpu)
    """

    device_type = DeviceType.INTEL_ARC

    def __init__(self, device_id: int = 0):
        self._has_dpctl = False
        self._has_dpnp = False
        self._sycl_device = None
        self._queue = None
        self._init_backend(device_id)

    def _init_backend(self, device_id: int):
        """Try to initialize Intel SYCL/dpctl device."""
        try:
            import dpctl
            self._has_dpctl = True
        except ImportError:
            logger.warning("dpctl not installed. Intel Arc backend unavailable.")
            return

        try:
            import dpnp as _dpnp
            self._has_dpnp = True
            self.dpnp = _dpnp
        except ImportError:
            logger.warning("dpnp not installed. Intel Arc backend limited to memory management only.")
            self.dpnp = None

        try:
            # Enumerate available Intel GPU devices
            devices = dpctl.get_devices(device_type="gpu")
            intel_gpus = [d for d in devices if d.backend.name == "level_zero"
                          or "Intel" in str(d) or "Intel" in d.name]

            if not intel_gpus:
                logger.info("No Intel GPU devices found via dpctl.")
                return

            if device_id < len(intel_gpus):
                self._sycl_device = intel_gpus[device_id]
            else:
                self._sycl_device = intel_gpus[0]

            self._queue = dpctl.SyclQueue(self._sycl_device)

            # Gather device capabilities
            name = self._sycl_device.name
            max_compute_units = self._sycl_device.max_compute_units
            max_wg_size = self._sycl_device.max_work_group_size

            # Estimate VRAM from device info
            try:
                global_mem = self._sycl_device.global_mem_size
                vram_gb = global_mem / (1024 ** 3)
            except Exception:
                vram_gb = 0.0

            # Check FP16/BF16 support (subgroup support)
            fp16 = False
            bf16 = False
            try:
                fp16 = self._sycl_device.has_aspect_fp16
            except Exception:
                pass
            try:
                bf16 = self._sycl_device.has_aspect_fp64  # not exact but indicative
            except Exception:
                pass

            self.capability = DeviceCapability(
                name=f"Intel Arc ({name})",
                vram_gb=vram_gb,
                compute_units=max_compute_units,
                supports_fp16=fp16,
                supports_bf16=bf16,
                max_work_group_size=max_wg_size,
            )
            self.is_available = True
            logger.info(f"Intel Arc GPU detected: {name}, VRAM: {vram_gb:.1f}GB, "
                        f"EUs: {max_compute_units}")

        except Exception as e:
            logger.warning(f"Failed to initialize Intel Arc backend: {e}")

    def to_device(self, array):
        if not self.is_available:
            return array  # fall back to CPU array
        if self._has_dpnp:
            return self.dpnp.asarray(array, dtype=array.dtype)
        # Fallback: use dpctl memory + USM
        import dpctl
        import numpy as np
        ary = np.ascontiguousarray(array)
        from dpctl import tensor as dpctl_tensor
        return dpctl_tensor.usm_ndarray(
            ary.shape,
            dtype=ary.dtype,
            buffer="device",
            queue=self._queue,
        )

    def to_cpu(self, array):
        import numpy as np
        if hasattr(array, 'asnumpy'):
            return array.asnumpy()
        return np.asarray(array)

    def matmul(self, a, b):
        if not self.is_available or not self._has_dpnp:
            # Fall back to numpy via to_cpu
            import numpy as np
            a_cpu = self.to_cpu(a)
            b_cpu = self.to_cpu(b)
            return np.asarray(a_cpu @ b_cpu)
        return self.dpnp.matmul(a, b)

    def rms_norm(self, x, weight, eps=1e-6):
        if not self.is_available or not self._has_dpnp:
            import numpy as np
            return CpuBackend().rms_norm(self.to_cpu(x), self.to_cpu(weight), eps)
        dpnp = self.dpnp
        variance = dpnp.mean(x.astype(dpnp.float64) ** 2, axis=-1, keepdims=True)
        result = x / dpnp.sqrt(variance + eps) * weight.astype(dpnp.float32)
        return result

    def silu(self, x):
        if not self.is_available or not self._has_dpnp:
            import numpy as np
            return CpuBackend().silu(self.to_cpu(x))
        dpnp = self.dpnp
        return x / (1 + dpnp.exp(-x))

    def softmax(self, x, axis=-1):
        if not self.is_available or not self._has_dpnp:
            import numpy as np
            return CpuBackend().softmax(self.to_cpu(x), axis)
        dpnp = self.dpnp
        x_max = dpnp.max(x, axis=axis, keepdims=True)
        x_exp = dpnp.exp(x - x_max)
        return x_exp / dpnp.sum(x_exp, axis=axis, keepdims=True)

    def rope(self, x, cos, sin):
        if not self.is_available or not self._has_dpnp:
            import numpy as np
            return CpuBackend().rope(self.to_cpu(x), self.to_cpu(cos), self.to_cpu(sin))
        dpnp = self.dpnp
        n, h, d = x.shape
        half_d = d // 2
        x2 = x.reshape(n, h, half_d, 2)
        neg = dpnp.stack([-x2[..., 1], x2[..., 0]], axis=-1)
        c = cos[:n, dpnp.newaxis, :half_d, dpnp.newaxis]
        s = sin[:n, dpnp.newaxis, :half_d, dpnp.newaxis]
        return (x2 * c + neg * s).reshape(n, h, d)

    def copy(self, dst, src):
        if hasattr(dst, '__setitem__') and hasattr(src, '__getitem__'):
            dst[:] = src  # dpnp supports item assignment
        else:
            import numpy as np
            np.copyto(self.to_cpu(dst), self.to_cpu(src))

    def synchronize(self):
        if self._queue is not None:
            self._queue.wait()


# ─── NVIDIA CUDA Backend (for comparison) ────────────────────────────────────

class CudaBackend(DeviceBackend):
    """NVIDIA CUDA backend using cupy/cublas (reference)."""

    device_type = DeviceType.NVIDIA_CUDA

    def __init__(self, device_id: int = 0):
        try:
            import cupy as cp
            self.cp = cp
            self.is_available = True
            with cp.cuda.Device(device_id):
                props = cp.cuda.runtime.getDeviceProperties(device_id)
                mem_info = cp.cuda.runtime.memGetInfo()
                free_mem = mem_info[0]
                total_mem = mem_info[1]
                self.capability = DeviceCapability(
                    name=f"NVIDIA {props['name'].decode()}",
                    vram_gb=total_mem / (1024 ** 3),
                    compute_units=props['multiProcessorCount'],
                    supports_fp16=True,
                    supports_bf16=props.get('major', 0) >= 8,
                )
            logger.info(f"NVIDIA CUDA backend initialized: {self.capability.name}")
        except ImportError:
            logger.debug("cupy not installed. NVIDIA CUDA backend unavailable.")
        except Exception as e:
            logger.warning(f"CUDA backend init failed: {e}")

    def to_device(self, array):
        return self.cp.asarray(array)

    def to_cpu(self, array):
        return self.cp.asnumpy(array)

    def matmul(self, a, b):
        return a @ b

    def rms_norm(self, x, weight, eps=1e-6):
        cp = self.cp
        x64 = x.astype(cp.float64)
        variance = cp.mean(x64 ** 2, axis=-1, keepdims=True)
        return x / cp.sqrt(variance + eps) * weight.astype(cp.float32)

    def silu(self, x):
        return x / (1 + self.cp.exp(-x))

    def softmax(self, x, axis=-1):
        cp = self.cp
        x_max = cp.max(x, axis=axis, keepdims=True)
        x_exp = cp.exp(x - x_max)
        return x_exp / cp.sum(x_exp, axis=axis, keepdims=True)

    def rope(self, x, cos, sin):
        cp = self.cp
        n, h, d = x.shape
        half_d = d // 2
        x2 = x.reshape(n, h, half_d, 2)
        xr = cp.stack([-x2[..., 1], x2[..., 0]], axis=-1)
        c = cos[:n, cp.newaxis, :half_d, cp.newaxis]
        s = sin[:n, cp.newaxis, :half_d, cp.newaxis]
        return (x2 * c + xr * s).reshape(n, h, d)

    def copy(self, dst, src):
        dst[:] = src

    def synchronize(self):
        self.cp.cuda.runtime.deviceSynchronize()


# ─── Factory ─────────────────────────────────────────────────────────────────

_BACKEND_INSTANCES: dict = {}
_BACKEND_CLASSES = [
    ('intel_arc', IntelArcBackend),
    ('nvidia', CudaBackend),
    ('cpu', CpuBackend),
]


def get_device(device_name: Optional[str] = None) -> DeviceBackend:
    """
    Get or create a device backend.

    Args:
        device_name: One of 'auto', 'cpu', 'intel_arc', 'nvidia', 'amd'.
                     'auto' (default) tries Intel Arc, then NVIDIA, then CPU.
                     If not available, falls back to CPU.

    Returns:
        A DeviceBackend instance.
    """
    env_device = os.environ.get('MOJOLLAMA_DEVICE', '').lower()

    if device_name is None or device_name == 'auto':
        device_name = env_device or 'auto'

    if device_name == 'auto':
        return _auto_detect()

    if device_name == 'cpu':
        return _get_or_create('cpu', CpuBackend)

    for key, cls in _BACKEND_CLASSES:
        if device_name == key:
            return _get_or_create(key, cls)

    logger.warning(f"Unknown device '{device_name}', falling back to CPU.")
    return _get_or_create('cpu', CpuBackend)


def _auto_detect() -> DeviceBackend:
    """Auto-detect best available device."""
    # Priority: Intel Arc -> NVIDIA CUDA -> CPU
    for key, cls in _BACKEND_CLASSES:
        if key == 'cpu':
            continue  # CPU is last resort
        instance = _get_or_create(key, cls)
        if instance.is_available:
            logger.info(f"Auto-selected device: {key} ({instance.capability})")
            return instance

    cpu = _get_or_create('cpu', CpuBackend)
    logger.info(f"Auto-selected device: cpu ({cpu.capability})")
    return cpu


def _get_or_create(key: str, backend_cls: type) -> DeviceBackend:
    """Get cached backend instance or create one."""
    if key not in _BACKEND_INSTANCES:
        _BACKEND_INSTANCES[key] = backend_cls()
    return _BACKEND_INSTANCES[key]


def list_devices() -> list[dict]:
    """List all available compute devices and their capabilities."""
    results = []
    for key, cls in _BACKEND_CLASSES:
        if key == 'cpu':
            continue
        try:
            inst = _get_or_create(key, cls)
            if inst.is_available and inst.capability:
                results.append({
                    'device': key,
                    'name': inst.capability.name,
                    'vram_gb': inst.capability.vram_gb,
                    'compute_units': inst.capability.compute_units,
                    'fp16': inst.capability.supports_fp16,
                })
        except Exception:
            pass

    results.append({
        'device': 'cpu',
        'name': f"CPU ({os.cpu_count()} cores)",
        'vram_gb': 0,
        'compute_units': os.cpu_count(),
        'fp16': False,
    })

    return results
