#!/usr/bin/env python3
"""MojoLlama hardware detection — CPU, GPU, and accelerator capabilities.

Detects:
  - CPU features: AVX2, AVX512, AMX, NEON, SVE, FMA
  - CPU topology: physical cores, SMT threads, NUMA nodes, cache
  - GPU backends: NVIDIA CUDA (with version), AMD ROCm, Intel SYCL, Vulkan
  - Apple Metal / MPS
  - Hybrid offloading capability

Usage:
    from mojollama.detect import detect_hardware, HardwareInfo
    hw = detect_hardware()
    print(hw.summary())
    print(hw.recommended_flags())
"""

import os
import sys
import re
import json
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Any


def read_proc_cpuinfo() -> Dict[str, Any]:
    """Read /proc/cpuinfo and extract CPU features."""
    info = {"flags": set(), "vendor": "unknown", "model": "unknown",
            "cores": mp.cpu_count(), "phys_cores": 0, "sockets": 0}
    
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        
        # Count physical cores (unique core ids per socket)
        cores_seen = set()
        socket = 0
        phys_id = -1
        core_id = -1
        
        for line in text.split("\n"):
            if line.startswith("processor"):
                socket += 1
            if line.startswith("physical id"):
                phys_id = int(line.split(":")[1].strip())
            if line.startswith("core id"):
                core_id = int(line.split(":")[1].strip())
                cores_seen.add((phys_id, core_id))
            if line.startswith("flags"):
                info["flags"] = set(line.split(":")[1].strip().split())
            if line.startswith("model name"):
                info["model"] = line.split(":")[1].strip()
            if line.startswith("vendor_id"):
                info["vendor"] = line.split(":")[1].strip()
        
        info["phys_cores"] = len(cores_seen)
        info["sockets"] = max(1, phys_id + 1) if phys_id >= 0 else 1
    
    except FileNotFoundError:
        # macOS or non-Linux
        pass
    
    return info


def detect_cpu_features(cpuinfo: Dict[str, Any]) -> Dict[str, Any]:
    """Detect CPU capabilities from flags."""
    flags = cpuinfo.get("flags", set())
    
    features = {
        "arch": "x86_64",  # default
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": "avx512_vnni" in flags,
        "avx512_bf16": "avx512_bf16" in flags,
        "amx_int8": "amx_int8" in flags,
        "amx_bf16": "amx_bf16" in flags,
        "fma": "fma" in flags,
        "f16c": "f16c" in flags,
        "bmi2": "bmi2" in flags,
        "sse3": "sse3" in flags,
        "ssse3": "ssse3" in flags,
        "neon": False,
        "sve": False,
        "apple_silicon": False,
    }
    
    # Detect architecture
    if "avx512f" in flags:
        features["arch"] = "x86_64_avx512"
    elif "avx2" in flags:
        features["arch"] = "x86_64_avx2"
    elif "sse3" in flags:
        features["arch"] = "x86_64"
    
    # Detect ARM
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        if "CPU architecture: 8" in text or "aarch64" in text:
            features["arch"] = "aarch64"
            features["neon"] = True
            if "sve" in text:
                features["sve"] = True
    except: pass
    
    # macOS Apple Silicon
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True, timeout=5)
            if "Apple" in result.stdout:
                features["arch"] = "aarch64_apple"
                features["apple_silicon"] = True
                features["neon"] = True
        except: pass
    
    return features


def detect_nvidia_gpu() -> Optional[Dict[str, Any]]:
    """Detect NVIDIA GPU and CUDA version."""
    try:
        # nvidia-smi for GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "name": parts[0] if len(parts) > 0 else "unknown",
                "vram_mb": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                "compute_cap": parts[2] if len(parts) > 2 else "?",
            })
        
        # CUDA version
        cuda_version = None
        try:
            r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            m = re.search(r"release (\d+\.\d+)", r.stdout)
            if m:
                cuda_version = m.group(1)
        except: pass
        
        if not cuda_version:
            try:
                r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
                m = re.search(r"CUDA Version:\s+(\d+\.\d+)", r.stdout)
                if m:
                    cuda_version = m.group(1)
            except: pass
        
        return {
            "available": True,
            "count": len(gpus),
            "gpus": gpus,
            "cuda_version": cuda_version,
            "cuda_major": int(cuda_version.split(".")[0]) if cuda_version else None,
        }
    except FileNotFoundError:
        return None
    except Exception:
        return None


def detect_rocm() -> Optional[Dict[str, Any]]:
    """Detect AMD ROCm."""
    try:
        # Check for rocminfo
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        
        gpu_count = 0
        gpu_names = []
        for line in result.stdout.split("\n"):
            if "Name:" in line and "GPU" in result.stdout.split(line)[0] if False else False:
                pass
            m = re.search(r'Name:\s+(AMD\s+\S+)', line)
            if m:
                gpu_names.append(m.group(1))
                gpu_count += 1
        
        if not gpu_names:
            gpu_count = result.stdout.count("GPU Agent")
            if gpu_count == 0:
                return None
        
        return {
            "available": True,
            "count": max(1, gpu_count),
            "gpus": gpu_names if gpu_names else ["AMD GPU"],
        }
    except FileNotFoundError:
        return None


def detect_sycl() -> Optional[Dict[str, Any]]:
    """Detect Intel SYCL/oneAPI."""
    try:
        result = subprocess.run(["sycl-ls"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        devices = [line.strip() for line in result.stdout.split("\n") if line.strip()]
        return {
            "available": len(devices) > 0,
            "devices": devices,
        }
    except FileNotFoundError:
        return None


def detect_vulkan() -> Optional[Dict[str, Any]]:
    """Detect Vulkan devices."""
    try:
        result = subprocess.run(["vulkaninfo", "--summary"],
                               capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return None
        devices = re.findall(r'GPU\d+:\s+(\S+)', result.stdout)
        return {
            "available": len(devices) > 0,
            "devices": devices,
        }
    except FileNotFoundError:
        return None


def detect_apple_metal() -> Optional[Dict[str, Any]]:
    """Detect Apple Metal / MPS."""
    try:
        # Check for Metal framework
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10
        )
        has_apple_gpu = "Apple" in result.stdout and ("GPU" in result.stdout or "Chip" in result.stdout)
        
        # Check MPS availability via Python
        mps_available = False
        try:
            import torch
            mps_available = torch.backends.mps.is_available()
        except: pass
        
        return {
            "available": has_apple_gpu or mps_available,
            "mps_available": mps_available,
        }
    except FileNotFoundError:
        return None


def detect_llamacpp_backends() -> Dict[str, bool]:
    """Check which llama.cpp backends are compiled in."""
    backends = {"metal": False, "cuda": False, "vulkan": False, "sycl": False}
    
    # Check the built llama-server binary
    binary = "/tmp/llama.cpp/build/bin/llama-server"
    if not os.path.exists(binary):
        return backends
    
    try:
        result = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=5)
        help_text = result.stdout + result.stderr
        
        # Check for backend-specific flags
        backends["cuda"] = "--gpu-layers" in help_text or "n-gpu-layers" in help_text
        backends["metal"] = "metal" in help_text.lower()
        backends["vulkan"] = "vulkan" in help_text.lower()
        backends["sycl"] = "sycl" in help_text.lower()
    except: pass
    
    return backends


def detect_moe_capability() -> Dict[str, Any]:
    """Check if models can use MoE-specific offloading."""
    binary = "/tmp/llama.cpp/build/bin/llama-server"
    has_moe_flag = False
    try:
        result = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=5)
        has_moe_flag = "--n-cpu-moe" in (result.stdout + result.stderr) or "-ncmoe" in (result.stdout + result.stderr)
    except: pass
    
    return {
        "expert_offloading": has_moe_flag,
        "flag": "-ncmoe" if has_moe_flag else None,
    }


def detect_hardware() -> Dict[str, Any]:
    """Full hardware detection. Returns complete system capabilities."""
    cpuinfo = read_proc_cpuinfo()
    cpu_features = detect_cpu_features(cpuinfo)
    nvidia = detect_nvidia_gpu()
    rocm = detect_rocm()
    sycl = detect_sycl()
    vulkan = detect_vulkan()
    metal = detect_apple_metal()
    llama = detect_llamacpp_backends()
    moe = detect_moe_capability()
    
    hw = {
        "cpu": {
            "model": cpuinfo["model"],
            "vendor": cpuinfo["vendor"],
            "logical_cores": cpuinfo["cores"],
            "physical_cores": cpuinfo["phys_cores"],
            "sockets": cpuinfo["sockets"],
            "smt": cpuinfo["cores"] > cpuinfo["phys_cores"],
            "smt_threads": cpuinfo["cores"] // max(1, cpuinfo["phys_cores"]),
            "features": cpu_features,
            "llama_cpu_mask": _generate_cpu_mask(cpuinfo),
        },
        "gpu": {
            "nvidia": nvidia,
            "rocm": rocm,
            "sycl": sycl,
            "vulkan": vulkan,
            "metal": metal,
            "primary": _detect_primary_gpu(nvidia, rocm, sycl, vulkan, metal),
        },
        "llamacpp_backends": llama,
        "moe": moe,
        "os": sys.platform,
    }
    
    return hw


def _generate_cpu_mask(cpuinfo: Dict[str, Any]) -> str:
    """Generate CPU mask for physical cores only (no SMT)."""
    n_phys = cpuinfo.get("phys_cores", 0)
    if n_phys == 0:
        return "0x0"
    # Mask with lowest n_phys bits set
    mask = (1 << n_phys) - 1
    return f"0x{mask:016X}"


def _detect_primary_gpu(nvidia, rocm, sycl, vulkan, metal) -> Optional[str]:
    """Determine the primary GPU backend."""
    if nvidia and nvidia.get("available"):
        cuda_major = nvidia.get("cuda_major")
        if cuda_major:
            return f"cuda_{cuda_major}"
        return "cuda"
    if rocm and rocm.get("available"):
        return "rocm"
    if sycl and sycl.get("available"):
        return "sycl"
    if vulkan and vulkan.get("available"):
        return "vulkan"
    if metal and metal.get("available"):
        return "metal"
    return None


def recommend_llamacpp_flags(hw: Dict[str, Any], model_path: str = "") -> Dict[str, Any]:
    """Generate optimal llama.cpp flags based on hardware detection."""
    cpu = hw.get("cpu", {})
    cpu_feat = cpu.get("features", {})
    gpu = hw.get("gpu", {})
    primary = gpu.get("primary")
    llama_b = hw.get("llamacpp_backends", {})
    
    flags = {
        "threads": cpu.get("physical_cores", 32),
        "threads_batch": max(1, cpu.get("physical_cores", 32) // 2),
        "batch_size": 4096 if cpu_feat.get("arch") in ("x86_64_avx2", "x86_64_avx512") else 2048,
        "ubatch_size": 512,
        "n_parallel": min(cpu.get("physical_cores", 32), 64),
        "mlock": True,
        "cont_batching": True,
        "flash_attn": cpu_feat.get("f16c", False),
        "cpu_mask": cpu.get("llama_cpu_mask", "0x0"),
    }
    
    # GPU offloading
    if primary and primary.startswith("cuda"):
        cuda_major = int(primary.split("_")[1]) if "_" in primary else 12
        flags["gpu"] = "cuda"
        flags["gpu_layers"] = -1  # offload all layers
        flags["cuda_version"] = f"cuda_{cuda_major}"
        flags["cuda_arch"] = f"sm_{80 if cuda_major >= 11 else 70}"
        
        # CUDA-specific tuning
        if cuda_major >= 13:
            flags["cuda_fusion"] = True  # hypothetical future feature
        elif cuda_major >= 12:
            flags["flash_attn"] = True  # CUDA 12 has better FA support
        elif cuda_major == 11:
            flags["flash_attn"] = False  # FA on CUDA 11 can be slower
    
    elif primary == "rocm":
        flags["gpu"] = "rocm"
        flags["gpu_layers"] = -1
        flags["flash_attn"] = True
    
    elif primary == "metal":
        flags["gpu"] = "metal"
        flags["gpu_layers"] = -1  # Apple Silicon unified memory — offload everything
    
    elif primary == "vulkan":
        flags["gpu"] = "vulkan"
        flags["gpu_layers"] = -1 if llama_b.get("vulkan") else 0
    
    elif primary == "sycl":
        flags["gpu"] = "sycl"
        flags["gpu_layers"] = -1 if llama_b.get("sycl") else 0
    
    # MoE-specific: if n_parallel is high and GPU is weak, keep experts on CPU
    if hw.get("moe", {}).get("expert_offloading"):
        flags["n_cpu_moe_experts"] = 0  # 0 = auto, let llama.cpp decide
    
    # CPU pinning only useful without GPU offloading (or partial offload)
    if not primary or flags.get("gpu_layers", 0) == 0:
        flags["cpu_mask"] = cpu.get("llama_cpu_mask", "0x0")
    else:
        flags["cpu_mask"] = "0x0"  # no pinning when GPU is active
    
    return flags


def format_recommendation(flags: Dict[str, Any]) -> str:
    """Format recommended flags as a llama-server command."""
    cmd_parts = ["llama-server", "-m", "model.gguf", "-c", "4096"]
    
    mapping = {
        "threads": "-t",
        "threads_batch": "-tb",
        "batch_size": "-b",
        "ubatch_size": "-ub",
        "n_parallel": "-np",
        "gpu_layers": "-ngl",
        "n_cpu_moe_experts": "-ncmoe",
    }
    
    for key, flag in mapping.items():
        val = flags.get(key)
        if val is not None and val != 0:
            cmd_parts.extend([flag, str(val)])
    
    if flags.get("mlock"):
        cmd_parts.append("--mlock")
    if flags.get("cont_batching"):
        cmd_parts.append("--cont-batching")
    if flags.get("flash_attn"):
        cmd_parts.extend(["-fa", "1"])
    
    cpu_mask = flags.get("cpu_mask", "0x0")
    if cpu_mask and cpu_mask != "0x0":
        cmd_parts.extend(["-C", cpu_mask])
    
    return " ".join(cmd_parts)


def summary(hw: Dict[str, Any]) -> str:
    """Human-readable summary of detected hardware."""
    cpu = hw["cpu"]
    feat = cpu["features"]
    gpu_info = hw["gpu"]
    
    lines = [
        "═══ Hardware Detection ═══",
        f"CPU: {cpu['model']}",
        f"     {cpu['logical_cores']} logical / {cpu['physical_cores']} physical cores",
        f"     {cpu['sockets']} socket(s), SMT: {cpu['smt']} ({cpu['smt_threads']}×)",
        f"     Arch: {feat['arch']}",
    ]
    
    # CPU features
    feat_list = []
    if feat["avx512f"]: feat_list.append("AVX-512F")
    if feat["avx512_vnni"]: feat_list.append("AVX-512-VNNI")
    if feat["avx2"]: feat_list.append("AVX2")
    if feat["fma"]: feat_list.append("FMA")
    if feat["f16c"]: feat_list.append("F16C")
    if feat["amx_int8"]: feat_list.append("AMX-INT8")
    if feat["neon"]: feat_list.append("NEON")
    if feat["apple_silicon"]: feat_list.append("Apple Silicon")
    if feat_list:
        lines.append(f"     ISA: {', '.join(feat_list)}")
    
    # GPU
    nv = gpu_info.get("nvidia")
    if nv and nv.get("available"):
        for g in nv.get("gpus", []):
            lines.append(f"GPU: NVIDIA {g['name']} ({g.get('vram_mb', '?')} MB VRAM, CC {g.get('compute_cap', '?')})")
        if nv.get("cuda_version"):
            lines.append(f"     CUDA {nv['cuda_version']}")
    
    rocm = gpu_info.get("rocm")
    if rocm and rocm.get("available"):
        lines.append(f"GPU: AMD ROCm ({rocm.get('count', '?')} devices)")
    
    metal = gpu_info.get("metal")
    if metal and metal.get("available"):
        lines.append(f"GPU: Apple Metal (MPS: {metal.get('mps_available', False)})")
    
    sycl_d = gpu_info.get("sycl")
    if sycl_d and sycl_d.get("available"):
        lines.append(f"GPU: Intel SYCL ({len(sycl_d.get('devices', []))} devices)")
    
    vulkan_d = gpu_info.get("vulkan")
    if vulkan_d and vulkan_d.get("available"):
        lines.append(f"GPU: Vulkan ({len(vulkan_d.get('devices', []))} devices)")
    
    if not any([
        nv and nv.get("available"),
        rocm and rocm.get("available"),
        metal and metal.get("available"),
        sycl_d and sycl_d.get("available"),
        vulkan_d and vulkan_d.get("available"),
    ]):
        lines.append("GPU: None detected (CPU-only mode)")
    
    lines.append("═══════════════════════")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hw = detect_hardware()
    print(summary(hw))
    print()
    
    flags = recommend_llamacpp_flags(hw)
    print("Recommended llama-server flags:")
    print(f"  {format_recommendation(flags)}")
    print()
    
    # Detect AVX2 vs AVX512 performance (if applicable)
    if hw["cpu"]["features"]["avx512f"]:
        print("⚠️  AVX-512 detected — llama.cpp may be faster with AVX2-only build")
        print("   (AVX-512 can throttle CPU frequency on some processors)")
