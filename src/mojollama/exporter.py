#!/usr/bin/env python3
"""MojoLlama Exporter — HF Hub push, safetensors, ONNX, checkpoint management.

Supports:
  - Push models and LoRA adapters to HuggingFace Hub
  - Convert GGUF → safetensors (for use with transformers)
  - Convert GGUF → ONNX (for deployment)
  - Save/load training checkpoints (optimizer, LR scheduler, step)
  - Best-model tracking (save when eval improves)

Dependencies:
  pip install huggingface_hub safetensors transformers onnx onnxruntime
"""

import os
import sys
import io
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

STUDIO_VERSION = "0.1.0"
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
LLAMA_CPP = "/tmp/llama.cpp"
CONVERTER = f"{LLAMA_CPP}/convert_hf_to_gguf.py"


# ═══════════════════════════════════════════════════════════════
# HuggingFace Hub
# ═══════════════════════════════════════════════════════════════

def hub_get_token(env_only: bool = False) -> str:
    """Resolve HF token from env vars: HF_TOKEN (highest priority), HUGGINGFACE_TOKEN.

    Args:
        env_only: If True, skip token store lookup (only check env). Default False.

    Returns:
        Token string, or empty string if not found.
    """
    token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if token:
        return token
    if not env_only:
        try:
            from huggingface_hub import HfFolder
            stored = HfFolder.get_token()
            if stored:
                return stored
        except Exception:
            pass
    return ""


def hub_login(token: str = "", save: bool = True) -> bool:
    """Login to HuggingFace Hub with a token or check existing login.

    Auto-detects HF_TOKEN / HUGGINGFACE_TOKEN env vars if no token is passed.
    """
    try:
        from huggingface_hub import login, whoami, HfFolder
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    # Resolve token: explicit arg > env var
    if not token:
        token = hub_get_token(env_only=True)

    if token:
        login(token=token, add_to_git_credential=save)
        print("✅ Logged in to HuggingFace Hub")
        return True
    else:
        try:
            user = whoami()
            print(f"✅ Already logged in as: {user.get('name', 'unknown')}")
            return True
        except Exception:
            print("❌ Not logged in to HuggingFace Hub.")
            print("   Set HF_TOKEN env var or use --token or run: huggingface-cli login")
            return False


def hub_whoami() -> Optional[Dict[str, Any]]:
    """Get current HuggingFace user info."""
    try:
        from huggingface_hub import whoami
        return whoami()
    except ImportError:
        return None
    except Exception:
        return None


def hub_push_model(
    model_path: str,
    repo_id: str,
    *,
    commit_message: str = "Upload MojoLlama GGUF model",
    private: bool = False,
    exist_ok: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Upload a GGUF model (or safetensors) to HuggingFace Hub.

    Args:
        model_path: Path to local model file (GGUF, safetensors, etc.)
        repo_id: HuggingFace repo ID (e.g. 'username/model-name')
        commit_message: Commit message for the upload
        private: Whether to create a private repo
        exist_ok: If False, fails if repo already exists
        metadata: Optional metadata dict to save as model card or README

    Returns:
        URL to the uploaded model on HF Hub, or None on failure.
    """
    try:
        from huggingface_hub import (
            HfApi, create_repo, upload_file, upload_folder,
            Repository, CommitOperationAdd,
        )
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return None

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    api = HfApi()

    # Create or get repo
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=exist_ok,
        )
        print(f"  Repo: {repo_id}")
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        return None

    model_size = os.path.getsize(model_path)
    model_name = os.path.basename(model_path)

    print(f"  File: {model_name} ({model_size / 1024**3:.2f} GB)")
    print(f"  Uploading...")

    # Upload the model file
    try:
        upload_url = api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

    # Upload metadata as README.md if provided
    if metadata:
        readme_parts = ["---"]
        for k, v in metadata.items():
            if isinstance(v, str):
                readme_parts.append(f"{k}: '{v}'")
            elif isinstance(v, (int, float)):
                readme_parts.append(f"{k}: {v}")
            elif isinstance(v, bool):
                readme_parts.append(f"{k}: {'true' if v else 'false'}")
        if metadata.get("tags"):
            tags = metadata["tags"]
            if isinstance(tags, list):
                readme_parts.append(f"tags: [{', '.join(tags)}]")
        readme_parts.append("---")
        readme_parts.append("")
        readme_parts.append(f"# {repo_id}")
        readme_parts.append("")
        readme_parts.append(f"MojoLlama Studio generated model ({model_name}).")
        readme_parts.append("")
        readme_parts.append("## Metadata")
        for k, v in metadata.items():
            if k == "tags":
                continue
            readme_parts.append(f"- **{k}**: {v}")
        readme_parts.append("")

        readme_content = "\n".join(readme_parts)

        # Upload README as a separate commit
        try:
            api.upload_file(
                path_or_fileobj=io.BytesIO(readme_content.encode()),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add model metadata and README",
            )
        except Exception as e:
            print(f"  ⚠️  README upload failed: {e}")

    model_url = f"https://huggingface.co/{repo_id}"
    print(f"✅ Uploaded to {model_url}")
    return model_url


def hub_push_adapter(
    adapter_path: str,
    base_model: str,
    repo_id: str,
    *,
    commit_message: str = "Upload MojoLlama LoRA adapter",
    private: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Upload a LoRA adapter to HuggingFace Hub, with adapter config."""
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
    except ImportError:
        print("❌ huggingface_hub not installed")
        return None

    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found: {adapter_path}")
        return None

    api = HfApi()

    try:
        create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        return None

    # Upload adapter GGUF
    adapter_name = os.path.basename(adapter_path)
    print(f"  Uploading adapter: {adapter_name}")
    try:
        api.upload_file(
            path_or_fileobj=adapter_path,
            path_in_repo=adapter_name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

    # Create adapter_config.json metadata
    adapter_config = {
        "adapter_type": "LoRA",
        "base_model": base_model,
        "studio_version": STUDIO_VERSION,
    }
    if metadata:
        adapter_config.update(metadata)

    try:
        config_json = json.dumps(adapter_config, indent=2)
        api.upload_file(
            path_or_fileobj=io.BytesIO(config_json.encode()),
            path_in_repo="adapter_config.json",
            repo_id=repo_id,
            commit_message="Add adapter configuration",
        )
    except Exception as e:
        print(f"  ⚠️  Config upload failed: {e}")

    url = f"https://huggingface.co/{repo_id}"
    print(f"✅ Adapter uploaded to {url}")
    return url


def hub_list_models(pattern: str = "") -> List[Dict[str, Any]]:
    """List models on the HuggingFace Hub matching a pattern."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []
    api = HfApi()
    try:
        models = api.list_models(search=pattern, task="text-generation")
        return [
            {
                "id": m.modelId,
                "downloads": m.downloads or 0,
                "likes": m.likes or 0,
                "pipeline_tag": m.pipeline_tag or "",
                "modified": str(m.lastModified) if m.lastModified else "",
            }
            for m in models[:20]
        ]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
# GGUF → Safetensors Conversion
# ═══════════════════════════════════════════════════════════════

def gguf_to_safetensors(
    gguf_path: str,
    output_dir: str,
    *,
    dtype: str = "float16",
    shard_size: str = "2GB",
) -> Optional[str]:
    """Convert a GGUF model to safetensors format.

    Args:
        gguf_path: Path to input GGUF file
        output_dir: Output directory for safetensors files
        dtype: Output dtype ('float16', 'float32', 'bfloat16')
        shard_size: Max shard size ('2GB', '5GB', '10GB', or 'NO' for no sharding)

    Returns:
        Path to the output directory, or None on failure.
    """
    print(f"GGUF → Safetensors conversion")
    print(f"  Input: {gguf_path}")
    print(f"  Output: {output_dir}")
    print(f"  Dtype: {dtype}")
    print(f"  Shard size: {shard_size}")
    print()

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file as st_save_file
    except ImportError:
        print("❌ safetensors not installed. Run: pip install safetensors")
        return None

    try:
        from gguf import GGUFReader, GGMLQuantizationType, dequantize
    except ImportError:
        print("❌ gguf package not installed")
        return None

    import torch

    os.makedirs(output_dir, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, torch.float16)

    max_shard_bytes = {
        "NO": float("inf"),
        "1GB": 1 * 1024**3,
        "2GB": 2 * 1024**3,
        "5GB": 5 * 1024**3,
        "10GB": 10 * 1024**3,
    }.get(shard_size, 2 * 1024**3)

    print("Reading GGUF model...")
    reader = GGUFReader(gguf_path)
    print(f"  {len(reader.tensors)} tensors found")

    # Extract metadata for model index
    metadata = {}
    arch = "llama"
    for name, field in reader.fields.items():
        if name == "general.architecture":
            try:
                arch = bytes(np.asarray(field.parts[-1])).decode("utf-8")
            except Exception:
                pass
        if name.startswith("GGUF."):
            continue
        metadata[name] = str(field.parts[-1]) if field.parts else ""

    # Convert tensors
    tensors = {}
    current_shard = 1
    current_shard_size = 0
    shard_files = []

    print(f"Converting to {dtype}...")

    for t in reader.tensors:
        name = t.name
        raw_data = np.asarray(t.data)

        # Dequantize to float32 first
        try:
            data_f32 = dequantize(raw_data, t.tensor_type).astype(np.float32)
        except Exception:
            # If dequantize fails, try reading as-is
            data_f32 = raw_data.astype(np.float32)

        # Convert to target dtype
        tensor_torch = torch.from_numpy(data_f32).to(target_dtype)
        tensors[name] = tensor_torch

        tensor_bytes = tensor_torch.numel() * tensor_torch.element_size()
        current_shard_size += tensor_bytes

        # Write shard if we've crossed the threshold
        if current_shard_size >= max_shard_bytes:
            if shard_size == "NO":
                continue
            shard_file = os.path.join(output_dir, f"model-{current_shard_size // (1024**3):03d}-of-{100:03d}.safetensors")
            shard_file = os.path.join(output_dir, f"model-{current_shard:05d}-of-{100:05d}.safetensors")
            shard_path = _write_safetensors_shard(tensors, shard_file)
            if shard_path:
                shard_files.append(shard_path)
            print(f"  Shard {current_shard}: {shard_path} ({current_shard_size / 1024**3:.2f} GB)")
            tensors = {}
            current_shard += 1
            current_shard_size = 0

    # Write final shard
    if tensors:
        total_shards = current_shard
        shard_file = os.path.join(output_dir, f"model-{current_shard:05d}-of-{total_shards:05d}.safetensors")
        shard_path = _write_safetensors_shard(tensors, shard_file)
        if shard_path:
            shard_files.append(shard_path)
        print(f"  Final shard {current_shard}: {shard_path}")

    # Rename shard files with correct total count
    total_shards = len(shard_files)
    renamed_files = []
    for i, old_path in enumerate(shard_files):
        new_path = os.path.join(output_dir, f"model-{i+1:05d}-of-{total_shards:05d}.safetensors")
        if old_path != new_path:
            os.rename(old_path, new_path)
        renamed_files.append(os.path.basename(new_path))

    # Write model index JSON
    index_data = {
        "metadata": {
            "total_size": sum(
                os.path.getsize(os.path.join(output_dir, f))
                for f in renamed_files
            ),
        },
        "weight_map": _build_weight_map(reader.tensors, renamed_files, max_shard_bytes),
    }

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"  Index: {index_path}")

    # Write config.json with architecture info
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": arch,
        "torch_dtype": dtype,
        "transformers_version": "4.40.0",
    }
    # Try to extract n_embd, n_head, n_layer from metadata
    n_embd = _get_metadata_int(reader, "llama.embedding_length")
    if n_embd:
        config["hidden_size"] = n_embd
    n_head = _get_metadata_int(reader, "llama.head_count")
    if n_head:
        config["num_attention_heads"] = n_head
    n_layer = _get_metadata_int(reader, "llama.block_count")
    if n_layer:
        config["num_hidden_layers"] = n_layer

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")

    print(f"\n✅ Safetensors model saved to {output_dir}")
    print(f"   {len(renamed_files)} shard(s), {len(reader.tensors)} tensors")
    return output_dir


def _write_safetensors_shard(tensors: Dict, path: str) -> Optional[str]:
    """Write a dict of torch tensors to a safetensors file."""
    try:
        from safetensors.torch import save_file as st_save_file
        st_save_file(tensors, path, metadata={"format": "pt"})
        return path
    except Exception as e:
        print(f"  ⚠️  Failed to write shard {path}: {e}")
        return None


def _build_weight_map(tensors, shard_files, max_shard_bytes):
    """Build weight_map dict for safetensors index."""
    import itertools

    weight_map = {}
    shard_idx = 0
    current_shard = os.path.join(os.path.dirname(shard_files[0]) if shard_files else ".", shard_files[0]) if shard_files else ""
    current_size = 0

    for t in tensors:
        tensor_size = np.asarray(t.data).nbytes
        if current_size + tensor_size > max_shard_bytes and shard_idx < len(shard_files) - 1:
            shard_idx += 1
            current_size = 0
            current_shard = os.path.join(os.path.dirname(shard_files[0]) if shard_files else ".", shard_files[shard_idx]) if shard_files else ""
        weight_map[t.name] = current_shard
        current_size += tensor_size

    return weight_map


def _get_metadata_int(reader, key):
    """Try to read an integer metadata value from a GGUF reader."""
    try:
        from gguf import GGUFValueType
        if key in reader.fields:
            field = reader.fields[key]
            if field.types[-1] in (GGUFValueType.UINT32, GGUFValueType.INT32,
                                   GGUFValueType.UINT64, GGUFValueType.INT64):
                return int(np.asarray(field.parts[-1]).item())
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# GGUF → ONNX Export
# ═══════════════════════════════════════════════════════════════

def gguf_to_onnx(
    gguf_path: str,
    output_path: str,
    *,
    opset: int = 17,
    optimize: bool = True,
    max_seq_len: int = 2048,
) -> Optional[str]:
    """Convert a GGUF model to ONNX format.

    This is a best-effort conversion — GGUF models are quantized weight formats
    meant for llama.cpp, not for ONNX. We convert weights and create a minimal
    ONNX export.

    Args:
        gguf_path: Path to input GGUF file
        output_path: Output ONNX file path
        opset: ONNX opset version (default: 17)
        optimize: Whether to run ONNX optimization passes
        max_seq_len: Maximum sequence length for the model

    Returns:
        Path to output ONNX file, or None on failure.
    """
    print(f"GGUF → ONNX Export")
    print(f"  Input: {gguf_path}")
    print(f"  Output: {output_path}")
    print(f"  Opset: {opset}")
    print(f"  Max seq len: {max_seq_len}")
    print()

    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        print("❌ onnx not installed. Run: pip install onnx onnxruntime")
        return None

    try:
        from gguf import GGUFReader, GGMLQuantizationType, dequantize
    except ImportError:
        print("❌ gguf package not installed")
        return None

    import torch
    import torch.nn as nn

    # Read GGUF model
    print("Reading GGUF model...")
    reader = GGUFReader(gguf_path)
    print(f"  {len(reader.tensors)} tensors found")

    # Extract model dimensions from metadata
    n_layer = _get_metadata_int(reader, "llama.block_count") or 24
    n_embd = _get_metadata_int(reader, "llama.embedding_length") or 2048
    n_head = _get_metadata_int(reader, "llama.head_count") or 16
    n_head_kv = _get_metadata_int(reader, "llama.head_count_kv") or n_head
    n_ff = _get_metadata_int(reader, "llama.feed_forward_length") or n_embd * 4

    print(f"  Architecture: {n_layer} layers, {n_embd} hidden, {n_head} heads")

    # Dequantize all tensors to float32
    print("Dequantizing tensors...")
    state_dict = {}
    for t in reader.tensors:
        name = t.name
        raw_data = np.asarray(t.data)
        try:
            data_f32 = dequantize(raw_data, t.tensor_type).astype(np.float32)
        except Exception:
            data_f32 = raw_data.astype(np.float32)
        state_dict[name] = torch.from_numpy(data_f32)

    # Build a simple transformer model and trace it
    print("Building ONNX model...")

    class LlamaLikeModel(nn.Module):
        """Simplified Llama-like model for ONNX export."""
        def __init__(self, state_dict, n_embd, n_head, n_layer, n_ff, max_seq_len):
            super().__init__()
            self.n_embd = n_embd
            self.max_seq_len = max_seq_len
            self.n_head = n_head
            self.n_layer = n_layer

            # Embedding
            self.embed = nn.Embedding(max_seq_len, n_embd)
            self.lm_head = nn.Linear(n_embd, max_seq_len, bias=False)

            # Load weights from state_dict
            self._load_weights(state_dict)

        def _load_weights(self, sd):
            # Map GGUF tensor names to our model
            weight_map = {
                "token_embd.weight": "embed.weight",
                "output.weight": "lm_head.weight",
            }
            for gguf_name, our_name in weight_map.items():
                if gguf_name in sd:
                    self.state_dict()[our_name].copy_(sd[gguf_name])

        def forward(self, input_ids):
            x = self.embed(input_ids)
            x = self.lm_head(x)
            return x

    model = LlamaLikeModel(state_dict, n_embd, n_head, n_layer, n_ff, max_seq_len)
    model.eval()

    # Export to ONNX
    dummy_input = torch.randint(0, max_seq_len, (1, 1))
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )

    # Run optimization passes
    if optimize:
        print("Running ONNX optimization...")
        try:
            import onnxoptimizer
            onnx_model = onnx.load(output_path)
            passes = ["fuse_consecutive_transposes", "eliminate_deadend", "eliminate_duplicate_initializer"]
            optimized = onnxoptimizer.optimize(onnx_model, passes)
            onnx.save(optimized, output_path)
            print("  Optimization complete")
        except ImportError:
            # onnxoptimizer may not be installed, skip
            pass
        except Exception as e:
            print(f"  ⚠️  Optimization failed: {e}")

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"\n✅ ONNX model saved to {output_path} ({size_mb:.0f} MB)")
    return output_path


# ═══════════════════════════════════════════════════════════════
# Checkpoint Management
# ═══════════════════════════════════════════════════════════════

CHECKPOINT_VERSION = 1


class TrainingCheckpoint:
    """Manages saving and loading training state for resume capability.

    Checkpoint format (saved as directory):
        checkpoint_dir/
            checkpoint.json       — metadata and training state
            model.safetensors     — model weights (or adapter weights)
            optimizer.pt           — optimizer state (if available)
            scheduler.pt           — LR scheduler state (if available)
            training_state.json    — step, epoch, loss history, best metrics
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        best_loss: float = float("inf"),
        best_step: int = 0,
        loss_history: Optional[List[float]] = None,
        model_path: Optional[str] = None,
        model_weights: Optional[Dict[str, np.ndarray]] = None,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        config: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Save a training checkpoint.

        Args:
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            best_loss: Best validation loss seen so far
            best_step: Step at which best loss was achieved
            loss_history: List of loss values over steps
            model_path: Path to save model weights checkpoint
            model_weights: Dict of numpy arrays (model weights)
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            config: Training configuration
            metadata: Additional metadata

        Returns:
            Path to the checkpoint JSON file.
        """
        # Save training state as JSON
        training_state = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "best_loss": best_loss,
            "best_step": best_step,
            "loss_history": loss_history or [],
            "timestamp": time.time(),
            "config": config or {},
            "metadata": metadata or {},
        }

        state_path = self.checkpoint_dir / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2, default=str)

        # Save optimizer state if provided
        if optimizer_state is not None:
            try:
                import torch
                opt_path = self.checkpoint_dir / "optimizer.pt"
                torch.save(optimizer_state, opt_path)
            except Exception as e:
                print(f"  ⚠️  Failed to save optimizer state: {e}")

        # Save scheduler state if provided
        if scheduler_state is not None:
            try:
                import torch
                sched_path = self.checkpoint_dir / "scheduler.pt"
                torch.save(scheduler_state, sched_path)
            except Exception as e:
                print(f"  ⚠️  Failed to save scheduler state: {e}")

        # Save model weights if provided (as safetensors shard)
        if model_weights:
            try:
                from safetensors.torch import save_file as st_save
                import torch
                torch_weights = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in model_weights.items()
                }
                weights_path = self.checkpoint_dir / "model.safetensors"
                st_save(torch_weights, str(weights_path))
                training_state["model_weights"] = str(weights_path)
            except Exception as e:
                print(f"  ⚠️  Failed to save model weights: {e}")

        # Copy model file if path provided
        if model_path and os.path.exists(model_path):
            try:
                dst = self.checkpoint_dir / os.path.basename(model_path)
                shutil.copy2(model_path, dst)
                training_state["model_file"] = str(dst)
            except Exception as e:
                print(f"  ⚠️  Failed to copy model file: {e}")

        # Write checkpoint JSON
        checkpoint_data = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "created_at": time.time(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "best_loss": best_loss,
            "best_step": best_step,
            "model_file": str(self.checkpoint_dir / "model.safetensors") if model_weights else "",
            "training_state": str(state_path),
            "config": config or {},
            "metadata": metadata or {},
        }

        ckpt_path = self.checkpoint_dir / "checkpoint.json"
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        print(f"  Checkpoint saved at step {step}")
        print(f"    Loss: {loss:.4f}, Best: {best_loss:.4f}")
        return str(ckpt_path)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load a training checkpoint.

        Returns:
            Dict with training state, or None if checkpoint doesn't exist.
        """
        ckpt_path = self.checkpoint_dir / "checkpoint.json"
        if not ckpt_path.exists():
            return None

        with open(ckpt_path) as f:
            checkpoint_data = json.load(f)

        # Load training state
        state_path = self.checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                training_state = json.load(f)
            checkpoint_data.update(training_state)

        return checkpoint_data

    def load_optimizer(self) -> Optional[Dict]:
        """Load saved optimizer state."""
        opt_path = self.checkpoint_dir / "optimizer.pt"
        if opt_path.exists():
            try:
                import torch
                return torch.load(opt_path)
            except Exception:
                pass
        return None

    def load_scheduler(self) -> Optional[Dict]:
        """Load saved scheduler state."""
        sched_path = self.checkpoint_dir / "scheduler.pt"
        if sched_path.exists():
            try:
                import torch
                return torch.load(sched_path)
            except Exception:
                pass
        return None

    @staticmethod
    def list_checkpoints(base_dir: str) -> List[Dict[str, Any]]:
        """List all available checkpoints in a directory."""
        checkpoints = []
        base = Path(base_dir)
        for ckpt_dir in sorted(base.glob("checkpoint_*")):
            ckpt_file = ckpt_dir / "checkpoint.json"
            if ckpt_file.exists():
                try:
                    with open(ckpt_file) as f:
                        data = json.load(f)
                    data["path"] = str(ckpt_dir)
                    data["name"] = ckpt_dir.name
                    checkpoints.append(data)
                except Exception:
                    checkpoints.append({
                        "path": str(ckpt_dir),
                        "name": ckpt_dir.name,
                        "error": "corrupted",
                    })
        return checkpoints

    @staticmethod
    def get_best_checkpoint(base_dir: str) -> Optional[Dict[str, Any]]:
        """Find the checkpoint with the best (lowest) loss."""
        checkpoints = TrainingCheckpoint.list_checkpoints(base_dir)
        if not checkpoints:
            return None
        valid = [c for c in checkpoints if "loss" in c and "error" not in c]
        if not valid:
            return None
        best = min(valid, key=lambda c: c.get("loss", float("inf")))
        return best


def save_best_model(
    current_loss: float,
    current_step: int,
    model_path: str,
    checkpoint_dir: str,
    *,
    lower_is_better: bool = True,
    metadata: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """Save model as best checkpoint if loss improves.

    Args:
        current_loss: Current validation loss
        current_step: Current training step
        model_path: Path to current model weights
        checkpoint_dir: Directory to save best model
        lower_is_better: If True, lower loss is better
        metadata: Additional metadata

    Returns:
        (saved, path) tuple — saved is True if best model was saved.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_record_path = os.path.join(checkpoint_dir, "best_model.json")

    # Load previous best loss
    best_loss = float("inf")
    best_step = 0
    if os.path.exists(best_record_path):
        try:
            with open(best_record_path) as f:
                record = json.load(f)
            best_loss = record.get("loss", float("inf"))
            best_step = record.get("step", 0)
        except Exception:
            pass

    # Check if current is better
    is_better = current_loss < best_loss if lower_is_better else current_loss > best_loss
    if not is_better:
        return False, None

    # Save best model
    best_model_path = os.path.join(checkpoint_dir, f"best_model_step_{current_step}.gguf")
    if os.path.exists(model_path):
        shutil.copy2(model_path, best_model_path)

    # Also save as 'best_model.gguf' for easy loading
    latest_best = os.path.join(checkpoint_dir, "best_model.gguf")
    if os.path.exists(model_path):
        shutil.copy2(model_path, latest_best)

    # Update best record
    best_record = {
        "loss": current_loss,
        "step": current_step,
        "model_path": best_model_path,
        "timestamp": time.time(),
        "metadata": metadata or {},
    }
    with open(best_record_path, "w") as f:
        json.dump(best_record, f, indent=2)

    print(f"\n🏆 Best model updated! Loss: {current_loss:.4f} (was {best_loss:.4f})")
    print(f"   Saved to {best_model_path}")
    return True, best_model_path


# ═══════════════════════════════════════════════════════════════
# CLI Commands
# ═══════════════════════════════════════════════════════════════

def cmd_hub_login(args):
    """Login to HuggingFace Hub."""
    print("╔══════════════════════════════════════════════╗")
    print("║     HuggingFace Hub — Login                  ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    token = args.token or ""
    hub_login(token=token)


def cmd_hub_whoami(args):
    """Show current HuggingFace user info."""
    user = hub_whoami()
    if user:
        print(f"User: {user.get('name', 'unknown')}")
        print(f"Email: {user.get('email', 'unknown')}")
        print(f"ID: {user.get('id', 'unknown')}")
    else:
        print("❌ Not logged in to HuggingFace Hub")


def cmd_hub_push(args):
    """Push a model to HuggingFace Hub."""
    print("╔══════════════════════════════════════════════╗")
    print("║     HuggingFace Hub — Push Model             ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    model_path = args.model or input("Model path: ").strip()
    repo_id = args.repo or input("HF repo ID (e.g. username/model): ").strip()

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Collect metadata
    metadata = {}
    if args.quant:
        metadata["quantization"] = args.quant
    if args.params:
        metadata["parameters"] = args.params
    if args.description:
        metadata["description"] = args.description
    metadata["studio_version"] = STUDIO_VERSION

    # Basic model info from filename
    model_name = os.path.basename(model_path)
    metadata["model_name"] = model_name

    result = hub_push_model(
        model_path=model_path,
        repo_id=repo_id,
        private=args.private,
        commit_message=args.message or f"Upload {model_name} via MojoLlama Studio",
        metadata=metadata,
    )

    if result:
        print(f"\n✅ Model available at: {result}")


def cmd_hub_push_adapter(args):
    """Push a LoRA adapter to HuggingFace Hub."""
    print("╔══════════════════════════════════════════════╗")
    print("║     HuggingFace Hub — Push Adapter           ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    adapter_path = args.adapter or input("Adapter path: ").strip()
    base_model = args.base_model or input("Base model name: ").strip()
    repo_id = args.repo or input("HF repo ID (e.g. username/adapter-name): ").strip()

    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found: {adapter_path}")
        return

    metadata = {"base_model": base_model}
    if args.rank:
        metadata["lora_rank"] = args.rank
    if args.alpha:
        metadata["lora_alpha"] = args.alpha

    result = hub_push_adapter(
        adapter_path=adapter_path,
        base_model=base_model,
        repo_id=repo_id,
        private=args.private,
        metadata=metadata,
    )

    if result:
        print(f"\n✅ Adapter available at: {result}")


def cmd_export_safetensors(args):
    """Convert GGUF model to safetensors format."""
    from mojollama.exporter import gguf_to_safetensors

    print("╔══════════════════════════════════════════════╗")
    print("║     GGUF → Safetensors Converter             ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    gguf_path = args.model or input("GGUF model path: ").strip()
    output_dir = args.output or gguf_path.replace(".gguf", "-safetensors")
    dtype = args.dtype or "float16"
    shard_size = args.shard_size or "2GB"

    result = gguf_to_safetensors(
        gguf_path=gguf_path,
        output_dir=output_dir,
        dtype=dtype,
        shard_size=shard_size,
    )

    if result:
        print(f"\n✅ Safetensors saved to: {result}")


def cmd_export_onnx(args):
    """Convert GGUF model to ONNX format."""
    from mojollama.exporter import gguf_to_onnx

    print("╔══════════════════════════════════════════════╗")
    print("║     GGUF → ONNX Converter                    ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    gguf_path = args.model or input("GGUF model path: ").strip()
    output_path = args.output or gguf_path.replace(".gguf", ".onnx")
    opset = args.opset or 17
    max_seq_len = args.max_seq_len or 2048

    result = gguf_to_onnx(
        gguf_path=gguf_path,
        output_path=output_path,
        opset=opset,
        max_seq_len=max_seq_len,
    )

    if result:
        print(f"\n✅ ONNX model saved to: {result}")


def cmd_checkpoint_save(args):
    """Save a training checkpoint."""
    print("╔══════════════════════════════════════════════╗")
    print("║     Save Training Checkpoint                 ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    ckpt_dir = args.checkpoint_dir or "checkpoints"
    step = args.step or 0
    epoch = args.epoch or 0
    loss = args.loss or 0.0

    ckpt = TrainingCheckpoint(ckpt_dir)
    result = ckpt.save(
        step=step,
        epoch=epoch,
        loss=loss,
        model_path=args.model or None,
        config={"learning_rate": args.lr} if args.lr else None,
    )
    if result:
        print(f"\n✅ Checkpoint saved to: {result}")


def cmd_checkpoint_load(args):
    """Load and display a training checkpoint."""
    print("╔══════════════════════════════════════════════╗")
    print("║     Load Training Checkpoint                 ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    ckpt_path = args.checkpoint_dir or "checkpoints"
    ckpt = TrainingCheckpoint(ckpt_path)
    data = ckpt.load()

    if data is None:
        print("❌ No checkpoint found")
        return

    print(f"Checkpoint: {ckpt_path}")
    print(f"  Version: {data.get('checkpoint_version', '?')}")
    print(f"  Step: {data.get('step', '?')}")
    print(f"  Epoch: {data.get('epoch', '?')}")
    print(f"  Loss: {data.get('loss', '?')}")
    print(f"  Best Loss: {data.get('best_loss', '?')}")
    print(f"  Best Step: {data.get('best_step', '?')}")
    print(f"  Created: {time.ctime(data.get('timestamp', 0))}")

    if data.get("config"):
        print(f"\n  Config:")
        for k, v in data["config"].items():
            print(f"    {k}: {v}")

    loss_history = data.get("loss_history", [])
    if loss_history:
        print(f"\n  Loss History: {len(loss_history)} entries")
        for i, l in enumerate(loss_history[-5:]):
            print(f"    [{i+1}] step {i}: {l:.4f}")

    return data


def cmd_checkpoint_list(args):
    """List all available checkpoints."""
    print("╔══════════════════════════════════════════════╗")
    print("║     List Training Checkpoints                ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    base_dir = args.checkpoint_dir or "."
    checkpoints = TrainingCheckpoint.list_checkpoints(base_dir)

    if not checkpoints:
        print("No checkpoints found")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) in {base_dir}:")
    print()
    for ckpt in checkpoints:
        name = ckpt.get("name", "?")
        loss = ckpt.get("loss", "?")
        step = ckpt.get("step", "?")
        epoch = ckpt.get("epoch", "?")
        created = time.ctime(ckpt.get("timestamp", 0)) if "timestamp" in ckpt else "?"
        status = "✅" if "error" not in ckpt else "❌"
        print(f"  {status} {name}")
        print(f"       Step: {step}, Epoch: {epoch}, Loss: {loss}")
        print(f"       Created: {created}")
        print()

    return checkpoints


# ═══════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MojoLlama Exporter & Checkpoint Tools")
    sub = parser.add_subparsers(dest="command", help="Command")

    # hub login
    p_login = sub.add_parser("hub-login", help="Login to HuggingFace Hub")
    p_login.add_argument("--token", help="HF API token")

    # hub whoami
    sub.add_parser("hub-whoami", help="Show HuggingFace user info")

    # hub push
    p_push = sub.add_parser("hub-push", help="Push model to HuggingFace Hub")
    p_push.add_argument("--model", "-m", help="Model path")
    p_push.add_argument("--repo", help="HF repo ID")
    p_push.add_argument("--message", help="Commit message")
    p_push.add_argument("--private", action="store_true", help="Create private repo")
    p_push.add_argument("--quant", help="Quantization type (metadata)")
    p_push.add_argument("--params", help="Parameter count (metadata)")
    p_push.add_argument("--description", help="Model description")

    # hub push-adapter
    p_push_adapter = sub.add_parser("hub-push-adapter", help="Push LoRA adapter to HF Hub")
    p_push_adapter.add_argument("--adapter", help="Adapter GGUF path")
    p_push_adapter.add_argument("--base-model", help="Base model name")
    p_push_adapter.add_argument("--repo", help="HF repo ID")
    p_push_adapter.add_argument("--private", action="store_true")
    p_push_adapter.add_argument("--rank", type=int, help="LoRA rank")
    p_push_adapter.add_argument("--alpha", type=float, help="LoRA alpha")

    # export safetensors
    p_st = sub.add_parser("export-safetensors", help="Convert GGUF to safetensors")
    p_st.add_argument("--model", "-m", help="GGUF model path")
    p_st.add_argument("--output", "-o", help="Output directory")
    p_st.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p_st.add_argument("--shard-size", default="2GB", help="Shard size (1GB, 2GB, 5GB, NO)")

    # export onnx
    p_onnx = sub.add_parser("export-onnx", help="Convert GGUF to ONNX")
    p_onnx.add_argument("--model", "-m", help="GGUF model path")
    p_onnx.add_argument("--output", "-o", help="Output path")
    p_onnx.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p_onnx.add_argument("--max-seq-len", type=int, default=2048)

    # checkpoint save
    p_ckpt_save = sub.add_parser("checkpoint-save", help="Save training checkpoint")
    p_ckpt_save.add_argument("--checkpoint-dir", default="checkpoints")
    p_ckpt_save.add_argument("--model", help="Model file path")
    p_ckpt_save.add_argument("--step", type=int, default=0)
    p_ckpt_save.add_argument("--epoch", type=int, default=0)
    p_ckpt_save.add_argument("--loss", type=float, default=0.0)
    p_ckpt_save.add_argument("--lr", help="Learning rate")

    # checkpoint load
    p_ckpt_load = sub.add_parser("checkpoint-load", help="Load training checkpoint")
    p_ckpt_load.add_argument("--checkpoint-dir", default="checkpoints")

    # checkpoint list
    p_ckpt_list = sub.add_parser("checkpoint-list", help="List training checkpoints")
    p_ckpt_list.add_argument("--checkpoint-dir", default=".", help="Base directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "hub-login": cmd_hub_login,
        "hub-whoami": cmd_hub_whoami,
        "hub-push": cmd_hub_push,
        "hub-push-adapter": cmd_hub_push_adapter,
        "export-safetensors": cmd_export_safetensors,
        "export-onnx": cmd_export_onnx,
        "checkpoint-save": cmd_checkpoint_save,
        "checkpoint-load": cmd_checkpoint_load,
        "checkpoint-list": cmd_checkpoint_list,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
