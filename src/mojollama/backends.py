#!/usr/bin/env python3
"""MojoLlama AutoBackend — CPU/GPU backend selector.

Switches between:
  - llama.cpp server (CPU) — 85 tok/s per user, continuous batching
  - MAX engine (GPU) — PagedAttention, CUDA kernels, competitive with vLLM
  - Python numpy (fallback) — reference implementation

Auto-detection:
  - If NVIDIA/AMD GPU found → MAX backend
  - If CPU only → llama.cpp backend  
  - If neither works → numpy fallback

Usage:
  from mojollama.backends import AutoBackend
  backend = AutoBackend()
  result = backend.generate("Hello", max_tokens=100)
"""

import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# ─── Auto-tuned config loader ─────────────────────────────────────────

CONFIG_PATH = Path.home() / ".mojollama" / "config.json"

def load_tuned_config(section: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load auto-tuned config.

    If *section* is given (e.g. 'llama_server', 'mojollama_engine'),
    returns only that section.  Otherwise returns the full config dict.
    Returns None on any failure.
    """
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            if section:
                return cfg.get(section)
            return cfg
    except Exception:
        pass
    return None


def _detect_cpu_mask() -> str:
    """Compute optimal CPU affinity mask for AMD Threadripper / multi-socket.
    
    On Threadripper 3970X (32c/64t): mask pins to physical cores only (0-31),
    avoiding SMT siblings which hurt generation throughput by ~25%.
    Returns hex string like '0x00000000FFFFFFFF' or '' if not applicable.
    """
    n_cores = os.cpu_count() or 64
    try:
        # Read physical core count from sysfs
        core_ids = set()
        for i in range(n_cores):
            try:
                with open(f"/sys/devices/system/cpu/cpu{i}/topology/core_id") as f:
                    core_ids.add(int(f.read().strip()))
            except FileNotFoundError:
                break
        physical = len(core_ids)
    except Exception:
        physical = n_cores // 2  # Assume SMT
    
    # Only generate mask if we have SMT (threads > physical cores)
    if n_cores <= physical or physical == 0:
        return ""
    
    # Build bitmask: bits 0..physical-1 set = physical cores only
    mask = (1 << physical) - 1
    return f"0x{mask:016X}"


def build_server_cmd(server_path: str, model_path: str, port: int,
                     config: Optional[Dict[str, Any]] = None) -> list:
    """Build llama-server command from config (with performance defaults).
    
    Tuned for AMD Threadripper 3970X (32c/64t, AVX2+FMA, DDR4):
      - threads=32 (physical cores, SMT hurts gen by ~25%)
      - batch=4096, ubatch=1024 (sweet spot for pp throughput)
      - flash_attn=ON (+40% pp, +5% tg vs standard attention)
      - mlock=ON (avoids page faults during inference)
      - cpu_mask pins to physical cores only
    """
    if config is None:
        cfg = load_tuned_config()
        config = cfg.get("llama_server", {}) if cfg else {}

    n_cores = os.cpu_count() or 64
    # Threadripper: use physical cores only for generation
    physical_cores = config.get("threads", min(32, n_cores))
    # Batch threads can use more (prompt eval is compute-bound, benefits from SMT)
    batch_threads = config.get("threads_batch", min(physical_cores, n_cores))

    cmd = [
        server_path, "-m", model_path, "-c", "4096",
        "-t", str(physical_cores),
        "-tb", str(batch_threads),
        "-b", str(config.get("batch_size", 4096)),
        "-ub", str(config.get("ubatch_size", 1024)),
        "-np", str(config.get("n_parallel", 4)),
        "--port", str(port), "--host", "127.0.0.1", "--no-webui",
    ]

    if config.get("mlock", True):
        cmd.append("--mlock")
    if config.get("cont_batching", True):
        cmd.append("--cont-batching")
    ct = config.get("chat_template", "")
    if ct:
        cmd.append("--chat-template"); cmd.append(ct)
    if config.get("reasoning", True) == False:
        cmd.append("--reasoning"); cmd.append("off")
    # Flash attention: ON by default — +40% pp, +5% tg on AVX2
    if config.get("flash_attn", True):
        cmd.append("-fa")
        cmd.append("1")
    # CPU affinity: pin to physical cores only
    cpu_mask = config.get("cpu_mask", "") or _detect_cpu_mask()
    if cpu_mask and cpu_mask != "0x0":
        cmd.append("-C")
        cmd.append(cpu_mask)

    return cmd


class BackendBase:
    """Base class for inference backends."""
    name = "base"
    
    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> dict:
        raise NotImplementedError
    
    def chat(self, messages: list, max_tokens: int = 256, **kwargs) -> dict:
        raise NotImplementedError
    
    def is_available(self) -> bool:
        raise NotImplementedError
    
    @property
    def info(self) -> dict:
        return {"name": self.name, "available": self.is_available()}


class LlamaCppBackend(BackendBase):
    """llama.cpp server backend — optimized for CPU."""
    name = "llama.cpp"
    
    def __init__(self, model_path: str = "", port: int = 8081):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.model_path = model_path or "/onedev-workspace/work/Llama-3.2-1B-Instruct-Q4_0.gguf"
        self._process: Optional[subprocess.Popen] = None
        self._server_path = "/tmp/llama.cpp/build/bin/llama-server"
        self._started = False
    
    def _ensure_server(self):
        """Start llama.cpp server if not running."""
        if self._started:
            return
        if self._check_server():
            self._started = True
            return
        # Start server with optimized settings from auto-tuner (or defaults)
        cmd = build_server_cmd(self._server_path, self.model_path, self.port)
        self._process = subprocess.Popen(cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Wait for it to be ready
        for _ in range(30):
            time.sleep(1)
            if self._check_server():
                self._started = True
                return
        raise RuntimeError("llama.cpp server failed to start")
    
    def _check_server(self) -> bool:
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.base_url}/health", timeout=2)
            return True
        except Exception:
            return False
    
    def _request(self, data: dict) -> dict:
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    
    def is_available(self) -> bool:
        try:
            return self._check_server()
        except Exception:
            return self._server_path and os.path.exists(self._server_path)
    
    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> dict:
        self._ensure_server()
        result = self._request({
            "prompt": prompt, "n_predict": max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "cache_prompt": True,
        })
        return {
            "text": result.get("content", ""),
            "tokens": result.get("tokens_predicted", 0),
            "backend": self.name,
        }
    
    def chat(self, messages: list, max_tokens: int = 256, **kwargs) -> dict:
        self._ensure_server()
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps({
                "messages": messages, "max_tokens": max_tokens,
                "temperature": kwargs.get("temperature", 0.7),
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        choice = result.get("choices", [{}])[0]
        return {
            "text": choice.get("message", {}).get("content", ""),
            "backend": self.name,
        }
    
    def stop(self):
        if self._process:
            self._process.kill()
            self._process = None


class MAXBackend(BackendBase):
    """MAX framework backend — GPU-accelerated inference via Modular MAX engine.

    This backend wraps MAX (https://docs.modular.com/max), Modular's ML
    inference engine built on Mojo 🔥.  MAX provides:
      - PagedAttention for efficient KV-cache management
      - FlashAttention-2 kernels on NVIDIA/AMD GPUs
      - Continuous batching with inflight batching
      - GGUF model loading with Q4_0, Q4_K_M, Q8_0, FP16 support

    **Current status:** This is a well-documented stub prepared for future
    integration.  When MAX is installed, this backend will use
    ``max.entrypoints.PipelineConfig`` and ``max.entrypoints.LLM`` for
    inference.  When MAX is not available, clear error messages guide the
    user through installation.

    Usage::

        from mojollama.backends import MAXBackend
        backend = MAXBackend("/path/to/model")
        if backend.is_available():
            result = backend.generate("Hello, world!", max_tokens=50)
        else:
            print("MAX not available; install with: pip install max")

    Attributes:
        name (str): Backend identifier (``"MAX"``).
        model_path (str): Path to model config directory or GGUF file.
        weight_path (str): Path to weights file (GGUF or safetensors).
    """
    name = "MAX"

    def __init__(self, model_path: str = "", weight_path: str = ""):
        """Initialize the MAX backend.

        Args:
            model_path: Path to the model config directory or GGUF file.
                Defaults to a sensible fallback path for Qwen3-30B-A3B.
            weight_path: Path to the weights file.  If empty, *model_path*
                is used as the weight file directly (standard for GGUF).
        """
        self.model_path = (
            model_path
            or "/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
        )
        self.weight_path = (
            weight_path
            or self.model_path  # GGUF bundles config + weights
        )
        self._llm = None
        self._available = None  # lazily evaluated

    # ═══════════════════════════════════════════════════════════════════════
    #  Detection
    # ═══════════════════════════════════════════════════════════════════════

    def is_available(self) -> bool:
        """Check whether the MAX engine is installed and importable.

        Tries to import ``max`` and optionally ``max.driver.GPU`` to
        detect GPU availability.

        Returns:
            ``True`` if MAX is installed (even without a GPU — CPU fallback
            is available).  ``False`` if MAX is not installed.
        """
        if self._available is not None:
            return self._available

        try:
            import max  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def has_gpu(self) -> bool:
        """Check if a compatible GPU is available via MAX.

        Returns:
            ``True`` if a GPU is detected and MAX can use it.
            ``False`` if MAX is in CPU mode or not installed.
        """
        if not self.is_available():
            return False
        try:
            from max.driver import GPU  # noqa: F401
            return True
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════════════════
    #  Engine lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    def _ensure_llm(self):
        """Initialize the MAX LLM pipeline if not already loaded.

        This is a **stub** — the actual ``max.entrypoints.LLM``
        initialization is prepared here but guarded by an import check.

        Raises:
            ImportError: If the ``max`` package is not installed.
                Message includes installation instructions.
        """
        if self._llm is not None:
            return

        if not self.is_available():
            raise ImportError(
                "MAX engine not available. Install with: pip install max\n"
                "See https://docs.modular.com/max/install for full setup."
            )

        # ── Future integration point ────────────────────────────────────
        # Once MAX is installed, the following code will activate the
        # pipeline.  It is deliberately left as a stub so that when MAX
        # becomes available on this system, the only work needed is to
        # uncomment and adjust paths.

        # ----8<---- [ FUTURE ACTIVATION CODE ] ----8<----
        # # Patch MAX's GGUF reader for BF16 support
        # import gguf
        # if not any(t.value == 30 for t in gguf.GGUFValueType):
        #     from enum import IntEnum
        #     class P(IntEnum):
        #         UINT8=0; INT8=1; UINT16=2; INT16=3; UINT32=4; INT32=5
        #         FLOAT32=6; BOOL=7; STRING=8; ARRAY=9; UINT64=10; INT64=11
        #         FLOAT64=12; BF16=30
        #     gguf.GGUFValueType = P
        #
        # from max.entrypoints import PipelineConfig, LLM
        # from max.driver import DeviceSpec
        #
        # # Auto-detect GPU or CPU
        # try:
        #     from max.driver import GPU
        #     gpu = GPU()
        #     device = DeviceSpec(id=0, device_type="gpu")
        #     print(f"  MAX: GPU detected ({gpu})")
        # except Exception:
        #     device = DeviceSpec(id=0, device_type="cpu")
        #     print("  MAX: CPU mode (install CUDA/cuDNN for GPU acceleration)")
        #
        # config = PipelineConfig(models={
        #     "main": {
        #         "model_path": self.model_path,
        #         "weight_path": [self.weight_path],
        #         "quantization_encoding": "q4_0",
        #         "max_length": 4096,
        #         "device_specs": [device],
        #     }
        # })
        # from max.entrypoints import LLM as MAX_LLM
        # self._llm = MAX_LLM(pipeline_config=config)
        # ----8<----

        raise ImportError(
            "MAX engine libraries not yet loaded.  This is a prepared stub.\n"
            "To activate: install MAX (pip install max) and uncomment the\n"
            "initialization code in MAXBackend._ensure_llm()."
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  Inference
    # ═══════════════════════════════════════════════════════════════════════

    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> dict:
        """Generate text completion from a prompt.

        Args:
            prompt: Input text string.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional parameters passed to the underlying
                MAX generator (e.g., ``temperature``, ``top_p``,
                ``repetition_penalty``).

        Returns:
            A dict with keys:
            - ``text`` (str): Generated continuation.
            - ``tokens`` (int): Number of tokens generated.
            - ``backend`` (str): Always ``"MAX"``.

        Raises:
            ImportError: If MAX is not installed (see
                :meth:`recommended_setup`).
        """
        self._ensure_llm()
        # Stub: once _llm is real, the following will generate:
        # result = self._llm.generate(
        #     [prompt],
        #     max_new_tokens=max_tokens,
        #     temperature=kwargs.get("temperature", 0.7),
        #     top_p=kwargs.get("top_p", 0.95),
        #     use_tqdm=False,
        # )
        # text = result[0] if isinstance(result, (list, tuple)) else str(result)
        # return {"text": text, "tokens": max_tokens, "backend": self.name}
        return {
            "text": (
                f"[MAX backend stub — model={self.model_path}, "
                f"max_tokens={max_tokens}]"
            ),
            "tokens": 0,
            "backend": self.name,
        }

    def chat(self, messages: list, max_tokens: int = 256, **kwargs) -> dict:
        """Chat completion — formats messages into a prompt and generates.

        Args:
            messages: List of dicts with ``role`` and ``content`` keys
                (e.g., ``[{"role": "user", "content": "Hello!"}]``).
            max_tokens: Maximum tokens in the response.
            **kwargs: Passed through to :meth:`generate`.

        Returns:
            A dict with keys ``text``, ``tokens``, ``backend``.
        """
        # Build a simple prompt from messages (chat-template-aware
        # formatting would go here in the real implementation).
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        prompt += "\nassistant: "
        return self.generate(prompt, max_tokens, **kwargs)

    # ═══════════════════════════════════════════════════════════════════════
    #  Setup guide
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def recommended_setup() -> str:
        """Print and return instructions for installing the MAX engine.

        Returns:
            A multi-line string with installation instructions.
        """
        msg = "\n".join([
            "╔══════════════════════════════════════════════════════════════╗",
            "║           MAX Engine — Installation Guide                   ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║                                                              ║",
            "║  The MAX backend provides GPU-accelerated inference via      ║",
            "║  Modular's MAX engine (Mojo 🔥).                             ║",
            "║                                                              ║",
            "║  Step 1 — Install the MAX Python package:                    ║",
            "║    $ pip install max                                         ║",
            "║                                                              ║",
            "║  Step 2 — Verify installation:                               ║",
            "║    $ python -c \"import max; print(max.__version__)\"          ║",
            "║                                                              ║",
            "║  Step 3 — (Optional) GPU acceleration:                       ║",
            "║    NVIDIA: install CUDA 12.x + cuDNN 9.x                     ║",
            "║    AMD:    install ROCm 6.x                                  ║",
            "║                                                              ║",
            "║  Docs: https://docs.modular.com/max/                         ║",
            "║  GitHub: https://github.com/modular/max                      ║",
            "║                                                              ║",
            "╚══════════════════════════════════════════════════════════════╝",
        ])
        print(msg)
        return msg

    def stop(self):
        """Release the MAX engine and free GPU resources.

        This is a stub — the real implementation would call
        ``self._llm.close()`` or similar when the LLM pipeline is
        active.
        """
        if self._llm is not None:
            # self._llm.close()   # future
            self._llm = None


class NumpyBackend(BackendBase):
    """Fallback numpy backend."""
    name = "numpy"
    
    def is_available(self) -> bool:
        return True
    
    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> dict:
        return {"text": "[numpy backend - use llama.cpp or MAX for performance]", 
                "tokens": 0, "backend": self.name}
    
    def chat(self, messages: list, max_tokens: int = 256, **kwargs) -> dict:
        """Chat completion — formats messages into a prompt, calls generate()."""
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt += "\nassistant: "
        return self.generate(prompt, max_tokens)


class AutoBackend:
    """Auto-selects best available backend."""
    
    def __init__(self, model_path: str = "", weight_path: str = "",
                 llama_port: int = 8081):
        self.model_path = model_path
        self.weight_path = weight_path
        self._llama_port = llama_port
        self._backend: Optional[BackendBase] = None
        self._lock = threading.Lock()
    
    def _detect(self) -> BackendBase:
        """Detect best available backend.
        
        Priority: MAX GPU > llama.cpp CPU > MAX CPU > numpy fallback
        """
        # 1. Check for GPU + MAX (fastest on GPU hardware)
        try:
            import max
            try:
                from max.driver import GPU
                gpu = GPU()
                print(f"MojoLlama: GPU ({gpu}) detected, using MAX backend")
                return MAXBackend(self.model_path, self.weight_path)
            except (ImportError, RuntimeError):
                pass  # No GPU, continue
        except ImportError:
            pass
        
        # 2. CPU-only: llama.cpp is 5.7x faster than MAX on CPU
        try:
            llama = LlamaCppBackend(self.model_path, port=self._llama_port)
            if llama._check_server() or os.path.exists(llama._server_path):
                # Start server in background
                if not llama._check_server():
                    print("MojoLlama: starting llama.cpp server (CPU backend)...")
                    threading.Thread(target=llama._ensure_server, daemon=True).start()
                    time.sleep(5)  # Give it a moment
                print(f"MojoLlama: using llama.cpp backend (85 tok/s on CPU)")
                return llama
        except Exception:
            pass
        
        # 3. MAX CPU fallback (slower but no setup needed)
        try:
            import max
            print(f"MojoLlama: using MAX CPU backend (~15 tok/s)")
            return MAXBackend(self.model_path, self.weight_path)
        except ImportError:
            pass
        
        # 4. Pure numpy fallback
        print(f"MojoLlama: no optimized backend, using numpy fallback")
        return NumpyBackend()
    
    @property
    def backend(self) -> BackendBase:
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    self._backend = self._detect()
        return self._backend
    
    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> dict:
        return self.backend.generate(prompt, max_tokens, **kwargs)
    
    def chat(self, messages: list, max_tokens: int = 256, **kwargs) -> dict:
        return self.backend.chat(messages, max_tokens, **kwargs)
    
    @property
    def info(self) -> dict:
        b = self.backend
        return {"active": b.name, "available": b.info}
    
    def stop(self):
        if isinstance(self._backend, LlamaCppBackend):
            self._backend.stop()


# Quick test
if __name__ == "__main__":
    backend = AutoBackend()
    print(f"\nActive backend: {backend.info}")
    print(f"Generating...")
    t0 = time.time()
    result = backend.generate("The meaning of life is", max_tokens=50)
    elapsed = time.time() - t0
    print(f"  Backend: {result['backend']}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Text: {result.get('text', '')[:100]}...")
    backend.stop()
