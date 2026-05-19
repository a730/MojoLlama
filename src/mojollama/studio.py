#!/usr/bin/env python3
"""MojoLlama Studio — train, export, dataset, merge, chat, benchmark.

Usage:
  mojollama-studio train --model model.gguf --data dataset.jsonl
  mojollama-studio export --hf meta-llama/Llama-3.2-1B --outtype q4_0
  mojollama-studio dataset create --output data.jsonl
  mojollama-studio dataset auto-label --model model.gguf --input prompts.jsonl
  mojollama-studio merge --base model.gguf --lora adapter.gguf --output merged.gguf
  mojollama-studio chat --model model.gguf
  mojollama-studio serve --model model.gguf --port 9000
  mojollama-studio benchmark --model model.gguf
"""

import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

# Auto-tuner for server optimization
from mojollama.autotune import autotune, save_config, print_config, load_config
from mojollama.dataset import (
    load_dataset, StreamingDataset, get_dataset_info, find_datasets,
    convert_dataset, compute_stats, split_dataset, write_jsonl,
    AutoLabeler, compute_confidence, ALL_FORMATS, FORMAT_DESCRIPTIONS,
    detect_format,
)

STUDIO_VERSION = "0.1.0"
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
LLAMA_CPP = "/tmp/llama.cpp"
CONVERTER = f"{LLAMA_CPP}/convert_hf_to_gguf.py"
FINETUNE_BIN = f"{LLAMA_CPP}/build/examples/training/llama-finetune"
SERVER_BIN = f"{LLAMA_CPP}/build/bin/llama-server"
CHAT_HTML = f"{BASE_DIR}/www/chat.html"
INDEX_HTML = f"{BASE_DIR}/www/index.html"


def print_banner():
    print("╔═══════════════════════════════════════════╗")
    print("║     MojoLlama Studio v" + STUDIO_VERSION + "               ║")
    print("║     Train · Export · Dataset · Chat       ║")
    print("╚═══════════════════════════════════════════╝")
    print()


# ─── Train ─────────────────────────────────────────────────────────────

TRAINING_METHODS = {
    "lora":    "Low-Rank Adaptation (standard, via llama-finetune)",
    "qlora":   "Quantized LoRA (NF4 base + LoRA adapters)",
    "dora":    "Weight-Decomposed Low-Rank Adaptation",
    "galore":  "Gradient Low-Rank Projection (memory-efficient full FT)",
    "dpo":     "Direct Preference Optimization (preference pairs)",
    "orpo":    "Odds Ratio Preference Optimization",
    "kto":     "Kahneman-Tversky Optimization (unpaired preferences)",
    "simpo":   "Simple Preference Optimization",
    "grpo":    "Group Relative Policy Optimization (RL fine-tuning)",
}

def cmd_train(args):
    """Fine-tune a model using one of the supported training methods."""
    from mojollama.trainer import MojoLlamaTrainer, emit_metric
    
    if args.list_methods:
        print_banner()
        print("[Train] Available training methods:\n")
        for name, desc in TRAINING_METHODS.items():
            print(f"  {name:8s} — {desc}")
        return
    
    print_banner()
    
    if args.method and args.method not in TRAINING_METHODS:
        print(f"❌ Unknown method: {args.method}")
        print("Available methods: lora, qlora, dora, galore, dpo, orpo, kto, simpo, grpo")
        return

    model = args.model or input("Model path (GGUF): ").strip()
    data = args.data or input("Training data path: ").strip()
    method = args.method or "lora"
    lora_out = args.lora_out or "adapter.gguf"
    
    if not os.path.exists(data):
        print(f"\n❌ Training data not found: {data}")
        print("Create one with: mojollama-studio dataset create")
        return
    
    # Build kwargs for method-specific params
    kwargs = {}
    if method == "dpo":
        kwargs["dpo_beta"] = args.dpo_beta
        kwargs["dpo_lr"] = args.lr
    elif method == "orpo":
        kwargs["orpo_lambda"] = args.orpo_lambda
        kwargs["orpo_lr"] = args.lr
    elif method == "simpo":
        kwargs["simpo_gamma"] = args.simpo_gamma
        kwargs["simpo_lr"] = args.lr
    elif method == "grpo":
        kwargs["grpo_group_size"] = args.grpo_group_size
        kwargs["grpo_clip"] = args.grpo_clip
        kwargs["grpo_lr"] = args.lr
    elif method == "galore":
        kwargs["galore_lr"] = args.lr
        kwargs["galore_rank"] = args.galore_rank
    
    trainer = MojoLlamaTrainer(
        model_path=model,
        data_path=data,
        method=method,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch,
        max_seq_length=args.max_seq_length,
        output_path=lora_out,
        save_steps=args.save_steps,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        dataset_format=args.dataset_format or "auto",
        max_samples=args.max_samples,
        template=args.template or "alpaca",
        **kwargs,
    )
    
    trainer.train()


# ─── Export ────────────────────────────────────────────────────────────

def cmd_export(args):
    """Convert HF model → GGUF."""
    print_banner()
    print("[Export] HuggingFace → GGUF converter")
    print()
    
    model = args.hf or args.model or input("HF model ID or path: ").strip()
    outtype = args.outtype or "q4_0"
    outfile = args.outfile or f"{model.split('/')[-1]}-{outtype}.gguf"
    
    # Supported direct types vs types needing post-quantization
    direct_types = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
    post_quant_types = {"q4_0", "q4_1", "q5_0", "q5_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k"}
    
    if outtype in direct_types:
        intermediate = outfile
    elif outtype in post_quant_types:
        # Step 1: convert to f16 first
        intermediate = outfile.replace(f"-{outtype}.gguf", "-f16.gguf")
        if intermediate == outfile:
            intermediate = outfile.replace(".gguf", "-f16.gguf")
        print(f"Step 1: Converting to F16 first...")
    else:
        print(f"Unsupported outtype: {outtype}")
        print(f"Supported: {', '.join(sorted(direct_types | post_quant_types))}")
        return
    
    cmd = [sys.executable, CONVERTER, model, "--outtype", "f16" if outtype in post_quant_types else outtype, 
           "--outfile", intermediate]
    if args.remote:
        cmd.append("--remote")
    if args.vocab_only:
        cmd.append("--vocab-only")
    
    print(f"Source: {model}")
    print(f"Intermediate: {intermediate}")
    print(f"Final: {outfile}")
    print()
    
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    
    # Step 2: post-quantize if needed
    if outtype in post_quant_types:
        print(f"\nStep 2: Quantizing {intermediate} → {outtype}...")
        from gguf import quantize as gguf_quantize
        from gguf import GGMLQuantizationType, GGUFReader
        
        qt_map = {
            "q4_0": GGMLQuantizationType.Q4_0,
            "q4_1": GGMLQuantizationType.Q4_1,
            "q5_0": GGMLQuantizationType.Q5_0,
            "q5_1": GGMLQuantizationType.Q5_1,
            "q2_k": GGMLQuantizationType.Q2_K,
            "q3_k": GGMLQuantizationType.Q3_K,
            "q4_k": GGMLQuantizationType.Q4_K,
            "q5_k": GGMLQuantizationType.Q5_K,
            "q6_k": GGMLQuantizationType.Q6_K,
            "q8_k": GGMLQuantizationType.Q8_K,
        }
        target_qt = qt_map.get(outtype)
        
        if target_qt:
            reader = GGUFReader(intermediate)
            print(f"  Read {len(reader.tensors)} tensors from intermediate")
            
            from gguf import GGUFWriter
            writer = GGUFWriter(outfile, "llama")
            
            # Copy KV metadata (simplified)
            for name, field in reader.fields.items():
                if name.startswith("GGUF."):
                    continue
                # Copy known types
                from gguf import GGUFValueType
                if field.types[-1] == GGUFValueType.STRING:
                    val = bytes(field.parts[-1]).decode('utf-8') if hasattr(field.parts[-1], 'tobytes') else str(field.parts[-1])
                    writer.add_string(name, val)
                elif field.types[-1] in (GGUFValueType.UINT32, GGUFValueType.INT32, GGUFValueType.UINT64, GGUFValueType.INT64):
                    val = int(field.parts[-1].item()) if hasattr(field.parts[-1], 'item') else int(field.parts[-1])
                    writer.add_uint32(name, val)
                elif field.types[-1] == GGUFValueType.FLOAT32:
                    val = float(field.parts[-1].item()) if hasattr(field.parts[-1], 'item') else float(field.parts[-1])
                    writer.add_float32(name, val)
            
            # Copy tensors with quantization
            for t in reader.tensors:
                data = np.asarray(t.data)
                new_data = gguf_quantize(data, target_qt) if t.tensor_type != target_qt.value else data
                writer.add_tensor(t.name, new_data, raw_dtype=target_qt)
            
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
            writer.close()
            os.remove(intermediate)
            print(f"  Quantized to {outtype}")
        else:
            # Fall back to copy
            import shutil
            shutil.copy(intermediate, outfile)
    
    size_mb = os.path.getsize(outfile) / 1024**2 if os.path.exists(outfile) else 0
    elapsed = time.time() - t0
    print(f"\n✅ Exported to {outfile} ({size_mb:.0f} MB, {elapsed:.0f}s)")


# ─── Merge ──────────────────────────────────────────────────────────────

def cmd_merge(args):
    """Merge LoRA/QLoRA adapter into base GGUF model.
    
    Supports both standard LoRA and QLoRA (NF4) adapters.
    Uses Python-based merge from gguf library for full control,
    or falls back to llama-export-lora if available.
    """
    print_banner()
    print("[Merge] LoRA/QLoRA adapter merge")
    print()

    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType, dequantize, quantize, Keys

    base_path = args.base
    lora_path = args.lora
    output_path = args.output
    out_quant = args.type or "q4_0"
    verbose = args.verbose

    # Read base model
    print(f"Reading base model: {base_path}")
    base_reader = GGUFReader(base_path)
    print(f"  {len(base_reader.tensors)} tensors")
    print(f"  Base type: {GGMLQuantizationType(base_reader.tensors[0].tensor_type).name if hasattr(base_reader.tensors[0], 'tensor_type') else '?'}")

    # Read LoRA adapter
    print(f"Reading LoRA adapter: {lora_path}")
    lora_reader = GGUFReader(lora_path)
    print(f"  {len(lora_reader.tensors)} tensors")

    # Get alpha from adapter metadata
    lora_alpha = 1.0
    if Keys.Adapter.LORA_ALPHA in lora_reader.fields:
        alpha_field = lora_reader.fields[Keys.Adapter.LORA_ALPHA]
        lora_alpha = float(np.asarray(alpha_field.parts[-1]).item())
    print(f"  LoRA alpha: {lora_alpha}")

    # Build LoRA tensor lookup: base_tensor_name -> {"a": tensor, "b": tensor}
    lora_map = {}
    for t in lora_reader.tensors:
        name = t.name
        if name.endswith(".lora_a"):
            base_name = name[:-7]  # strip ".lora_a"
            lora_map.setdefault(base_name, {})["a"] = t
        elif name.endswith(".lora_b"):
            base_name = name[:-7]  # strip ".lora_b"
            lora_map.setdefault(base_name, {})["b"] = t
        elif not name.endswith(".lora_a") and not name.endswith(".lora_b"):
            # Non-LoRA tensors in adapter (e.g. norm layers)
            lora_map.setdefault(name, {})["copy"] = t

    # Output quantization type map
    qt_map = {
        "f32": GGMLQuantizationType.F32,
        "f16": GGMLQuantizationType.F16,
        "q4_0": GGMLQuantizationType.Q4_0,
        "q4_1": GGMLQuantizationType.Q4_1,
        "q5_0": GGMLQuantizationType.Q5_0,
        "q5_1": GGMLQuantizationType.Q5_1,
        "q8_0": GGMLQuantizationType.Q8_0,
        "q2_k": GGMLQuantizationType.Q2_K,
        "q3_k": GGMLQuantizationType.Q3_K,
        "q4_k": GGMLQuantizationType.Q4_K,
        "q5_k": GGMLQuantizationType.Q5_K,
        "q6_k": GGMLQuantizationType.Q6_K,
        "q8_k": GGMLQuantizationType.Q8_K,
    }
    target_qt = qt_map.get(out_quant, GGMLQuantizationType.F16)

    # Get architecture from base model
    arch = "llama"
    if "general.architecture" in base_reader.fields:
        arch_field = base_reader.fields["general.architecture"]
        arch = bytes(np.asarray(arch_field.parts[-1])).decode("utf-8")

    print(f"  Output: {output_path} ({out_quant})")
    print()

    writer = GGUFWriter(output_path, arch)

    # Copy KV metadata from base model
    n_kv = 0
    for name, field in base_reader.fields.items():
        if name.startswith("GGUF."):
            continue
        try:
            if field.types[-1] == GGUFValueType.STRING:
                val = bytes(np.asarray(field.parts[-1])).decode("utf-8")
                writer.add_string(name, val)
                n_kv += 1
            elif field.types[-1] in (GGUFValueType.UINT32, GGUFValueType.INT32,
                                     GGUFValueType.UINT64, GGUFValueType.INT64):
                val = int(np.asarray(field.parts[-1]).item())
                writer.add_uint32(name, val)
                n_kv += 1
            elif field.types[-1] == GGUFValueType.FLOAT32:
                val = float(np.asarray(field.parts[-1]).item())
                writer.add_float32(name, val)
                n_kv += 1
            elif field.types[-1] == GGUFValueType.ARRAY:
                arr_field = field
                sub_type = arr_field.types[-1] if len(arr_field.types) > 1 else GGUFValueType.STRING
                if sub_type == GGUFValueType.STRING:
                    vals = [bytes(np.asarray(arr_field.parts[idx])).decode("utf-8") for idx in arr_field.data]
                    writer.add_string_array(name, vals)
                    n_kv += 1
                elif sub_type in (GGUFValueType.UINT32, GGUFValueType.INT32):
                    vals = [int(np.asarray(arr_field.parts[idx]).item()) for idx in arr_field.data]
                    writer.add_uint32_array(name, vals)
                    n_kv += 1
        except (ValueError, TypeError, IndexError, KeyError):
            pass  # skip fields we can't copy

    print(f"  Copied {n_kv} KV metadata entries")

    # Process tensors
    n_merged = 0
    n_copied = 0

    for t in base_reader.tensors:
        name = t.name
        raw_data = np.asarray(t.data)

        if name in lora_map and "a" in lora_map[name] and "b" in lora_map[name]:
            # --- Merge LoRA ---
            t_a = lora_map[name]["a"]
            t_b = lora_map[name]["b"]
            lora_a = np.asarray(t_a.data).astype(np.float32)
            lora_b = np.asarray(t_b.data).astype(np.float32)

            # Dequantize base tensor to F32
            base_f32 = dequantize(raw_data, t.tensor_type).astype(np.float32)

            # Compute LoRA delta: ΔW = B @ A
            # lora_a shape: (rank, input_dim)
            # lora_b shape: (output_dim, rank)
            lora_delta = lora_b @ lora_a

            rank = lora_a.shape[0]
            scaling = lora_alpha / rank
            merged_data = base_f32 + lora_delta * scaling

            # Quantize output
            if target_qt != GGMLQuantizationType.F32:
                out_data = quantize(merged_data, target_qt)
            else:
                out_data = merged_data

            writer.add_tensor(name, out_data, raw_dtype=target_qt)
            n_merged += 1
            print(f"  ✓ Merge: {name}")

        elif name in lora_map and "copy" in lora_map[name]:
            # Copy from adapter (e.g. norm layers overwritten by adapter)
            t_copy = lora_map[name]["copy"]
            copy_data = np.asarray(t_copy.data)
            writer.add_tensor(name, copy_data, raw_dtype=t_copy.tensor_type)
            n_copied += 1
            print(f"  ◉ Copy (adapter): {name}")

        else:
            # Copy base tensor as-is (optionally requantize)
            if target_qt != t.tensor_type:
                data_f32 = dequantize(raw_data, t.tensor_type).astype(np.float32)
                if target_qt != GGMLQuantizationType.F32:
                    out_data = quantize(data_f32, target_qt)
                else:
                    out_data = data_f32
                writer.add_tensor(name, out_data, raw_dtype=target_qt)
            else:
                writer.add_tensor(name, raw_data, raw_dtype=t.tensor_type)
            n_copied += 1

    print(f"\n  ─────────────────────────────")
    print(f"  Merged:  {n_merged} tensors")
    print(f"  Copied:  {n_copied} tensors")
    print(f"  ─────────────────────────────")

    # Write output file
    print("  Writing merged model...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    size_mb = os.path.getsize(output_path) / 1024**2 if os.path.exists(output_path) else 0
    print(f"\n✅ Merged model saved to {output_path} ({size_mb:.0f} MB)")


# ─── Dataset ───────────────────────────────────────────────────────────

def cmd_dataset(args):
    """Create, view, convert, compute stats, stream, and auto-label datasets."""
    print_banner()
    print("[Dataset] Training data management")
    print()

    actions = {
        "create": _cmd_dataset_create,
        "view": _cmd_dataset_view,
        "convert": _cmd_dataset_convert,
        "info": _cmd_dataset_info,
        "stats": _cmd_dataset_stats,
        "stream": _cmd_dataset_stream,
        "list": _cmd_dataset_list,
        "split": _cmd_dataset_split,
        "auto-label": _cmd_dataset_autolabel,
    }

    if args.action in actions:
        actions[args.action](args)
    else:
        print("Dataset commands:")
        print("  create     — Create a new dataset interactively")
        print("  view       — View sample entries from a dataset")
        print("  convert    — Convert between dataset formats")
        print("  info       — Show dataset metadata and format detection")
        print("  stats      — Compute comprehensive dataset statistics")
        print("  stream     — Stream a large dataset (memory-efficient)")
        print("  list       — List available datasets in the workspace")
        print("  split      — Split dataset into train/val/test")
        print("  auto-label — Auto-generate completions with confidence scoring")


def _cmd_dataset_create(args):
    """Create a new dataset interactively with format selection."""
    output = args.output or "dataset.jsonl"
    fmt = args.format or "alpaca"

    print(f"Creating {fmt} dataset: {output}")
    print("Format details:")
    print(f"  {FORMAT_DESCRIPTIONS.get(fmt, fmt)}")
    print()

    entries = []
    print("Enter samples. Empty prompt to finish.\n")
    try:
        while True:
            if fmt == "alpaca":
                instruction = input("Instruction: ").strip()
                if not instruction:
                    break
                input_text = input("Input (optional): ").strip()
                output_text = input("Output: ").strip()
                from mojollama.dataset import DatasetEntry, write_jsonl
                entries.append(DatasetEntry(
                    source_format=fmt,
                    instruction=instruction,
                    input_text=input_text,
                    output=output_text,
                ))
            elif fmt == "sharegpt":
                convs = []
                while True:
                    role = input("  Role (human/gpt, empty to finish): ").strip()
                    if not role:
                        break
                    val = input("  Message: ").strip()
                    if val:
                        convs.append({"from": role, "value": val})
                if not convs:
                    break
                from mojollama.dataset import DatasetEntry
                entries.append(DatasetEntry(
                    source_format=fmt,
                    conversations=convs,
                ))
            elif fmt == "openai":
                msgs = []
                while True:
                    role = input("  Role (system/user/assistant, empty to finish): ").strip()
                    if not role:
                        break
                    content = input("  Content: ").strip()
                    if content:
                        msgs.append({"role": role, "content": content})
                if not msgs:
                    break
                from mojollama.dataset import DatasetEntry
                entries.append(DatasetEntry(
                    source_format=fmt,
                    messages=msgs,
                ))
            else:
                prompt = input("Prompt: ").strip()
                if not prompt:
                    break
                completion = input("Completion: ").strip()
                from mojollama.dataset import DatasetEntry
                entries.append(DatasetEntry(
                    source_format="jsonl",
                    prompt=prompt,
                    completion=completion or "(pending)",
                ))

            print(f"  → Sample {len(entries)} saved\n")
    except (EOFError, KeyboardInterrupt):
        print()

    if entries:
        from mojollama.dataset import write_jsonl
        write_jsonl(entries, output, format=fmt)
        print(f"\n✅ Saved {len(entries)} samples to {output}")
    else:
        print("\n⚠️  No samples saved")


def _cmd_dataset_view(args):
    """View samples from a dataset."""
    path = args.dataset or args.input or input("Dataset path: ").strip()
    count = args.count or 10
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    from mojollama.dataset import get_dataset_info, get_reader
    info = get_dataset_info(path)
    print(f"Dataset: {path}")
    print(f"Format:   {info.get('format', 'unknown')}")
    print(f"Samples:  {info.get('line_count', '?')}")
    print(f"Size:     {info.get('size_display', '?')}")
    print()

    reader = get_reader(path, format=info.get("format"))
    entries = reader.read()[:count]

    for i, entry in enumerate(entries):
        print(f"[{i+1}] ", end="")
        if entry.instruction:
            print(f"Instruction: {entry.instruction[:80]}...")
        elif entry.prompt:
            print(f"Prompt: {entry.prompt[:80]}...")
        if entry.messages:
            roles = set(m.get("role", "") for m in entry.messages)
            print(f"    Messages: {len(entry.messages)} turns ({', '.join(roles)})")
        if entry.conversations:
            print(f"    Conversations: {len(entry.conversations)} turns")
        if entry.chosen and entry.rejected:
            print(f"    Chosen: {len(entry.chosen)} / Rejected: {len(entry.rejected)} turns")
        if entry.output:
            print(f"    Output: {entry.output[:80]}...")
        elif entry.completion:
            print(f"    Completion: {entry.completion[:80]}...")
        print()


def _cmd_dataset_convert(args):
    """Convert between dataset formats."""
    src = args.input or input("Source file: ").strip()
    dst = args.output or "converted.jsonl"
    fmt = args.format or "openai"

    if not os.path.exists(src):
        print(f"❌ File not found: {src}")
        return

    if fmt not in ALL_FORMATS:
        print(f"❌ Unknown format: {fmt}")
        print(f"Supported: {', '.join(ALL_FORMATS)}")
        return

    result = convert_dataset(src, dst, target_format=fmt)
    print(f"✅ Conversion complete:")
    print(f"  Input:  {result['input_path']} ({result['input_format']})")
    print(f"  Output: {result['output_path']} ({result['output_format']})")
    print(f"  Samples: {result['samples']}")


def _cmd_dataset_info(args):
    """Show comprehensive dataset info with format detection."""
    path = args.dataset or args.input or input("Dataset path: ").strip()
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    info = get_dataset_info(path)
    print(f"File:       {info['name']}")
    print(f"Path:       {info['path']}")
    print(f"Size:       {info['size_display']}")
    print(f"Format:     {info['format']}")
    print(f"Lines:      {info.get('line_count', 0)}")
    print()

    # Show sample preview
    samples = info.get("samples_preview", [])
    if samples:
        print("Sample data:")
        for i, s in enumerate(samples[:3]):
            print(f"  [{i+1}] {json.dumps(s, ensure_ascii=False)[:200]}")
            print()


def _cmd_dataset_stats(args):
    """Compute comprehensive dataset statistics."""
    path = args.dataset or args.input or input("Dataset path: ").strip()
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    print(f"Computing stats for: {path}")
    print()

    from mojollama.dataset import get_reader

    fmt = detect_format(path)
    print(f"Format: {FORMAT_DESCRIPTIONS.get(fmt, fmt)}")
    print()

    # Read all entries (may take a moment for large datasets)
    reader = get_reader(path, format=fmt, max_samples=args.max_samples)
    entries = reader.read()
    stats = compute_stats(entries)

    print(f"Total samples:      {stats.total_samples}")
    print(f"Estimated tokens:   {stats.estimated_tokens:,}")
    print(f"Vocabulary size:    {stats.vocab_size:,}")
    print()
    print("Length distribution (chars):")
    print(f"  Mean:   {stats.avg_length:.1f}")
    print(f"  Median: {stats.median_length:.1f}")
    print(f"  Std:    {stats.std_length:.1f}")
    print(f"  Min:    {stats.min_length}")
    print(f"  Max:    {stats.max_length}")
    print()
    print("Prompt length (chars):")
    print(f"  Mean:   {stats.to_dict()['prompt_length']['mean']}")
    print(f"  Median: {stats.to_dict()['prompt_length']['median']}")
    print()
    print("Completion length (chars):")
    print(f"  Mean:   {stats.to_dict()['completion_length']['mean']}")
    print(f"  Median: {stats.to_dict()['completion_length']['median']}")
    print()

    if stats.format_distribution:
        print("Format distribution:")
        for fmt_name, count in stats.format_distribution.items():
            print(f"  {fmt_name}: {count}")
        print()

    # Show histogram
    hist = stats.length_distribution(bins=8)
    if hist.get("counts"):
        edges = hist["bins"]
        counts = hist["counts"]
        max_count = max(counts)
        print("Length histogram:")
        for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            bar = "█" * int((counts[i] / max_count) * 30) if max_count > 0 else ""
            print(f"  {int(start):>6}–{int(end):<6} │ {bar} {counts[i]}")


def _cmd_dataset_stream(args):
    """Stream a large dataset (memory-efficient preview)."""
    path = args.dataset or args.input or input("Dataset path: ").strip()
    count = args.count or 20
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    from mojollama.dataset import StreamingDataset
    ds = StreamingDataset(path)
    total = ds.count_lines()
    print(f"Streaming {path}")
    print(f"Total lines: {total:,}")
    print(f"Previewing {min(count, total)} entries...")
    print()

    streamed = 0
    for i, entry in enumerate(ds.stream_entries()):
        if i >= count:
            break
        streamed += 1
        # Show a compact preview
        text = entry.get_text()[:100]
        print(f"  [{i+1}] {text}...")
        # Show format hints
        hints = []
        if entry.messages:
            hints.append(f"{len(entry.messages)} messages")
        if entry.conversations:
            hints.append(f"{len(entry.conversations)} convs")
        if entry.chosen or entry.rejected:
            hints.append(f"preference")
        if hints:
            print(f"       ({', '.join(hints)})")
        print()

    print(f"Previewed {streamed} entries (of {total:,} total)")
    if total > count:
        print(f"Use --count N to show more entries")


def _cmd_dataset_list(args):
    """List available datasets."""
    directory = args.directory or os.getcwd()
    datasets = find_datasets(directory)

    if not datasets:
        print("No datasets found.")
        print(f"Looked in: {directory}")
        print("Supported: .jsonl, .jsonl.gz, .json, .json.gz")
        return

    print(f"Found {len(datasets)} datasets in {directory}:")
    print()
    print(f"  {'Name':<30} {'Format':<14} {'Size':<10}")
    print(f"  {'─'*30} {'─'*14} {'─'*10}")
    for ds in datasets:
        print(f"  {ds['name']:<30} {ds['format_description']:<14} {ds['size_display']:<10}")


def _cmd_dataset_split(args):
    """Split dataset into train/val/test sets."""
    path = args.dataset or args.input or input("Dataset path: ").strip()
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    train_pct = args.train_ratio or 0.8
    val_pct = args.val_ratio or 0.1
    test_pct = args.test_ratio or 0.1
    out_prefix = args.output_prefix or os.path.splitext(path)[0]
    shuffle = not args.no_shuffle

    from mojollama.dataset import get_reader
    fmt = detect_format(path)
    reader = get_reader(path, format=fmt)
    entries = reader.read()
    total = len(entries)

    splits = split_dataset(entries, train_ratio=train_pct, val_ratio=val_pct,
                            test_ratio=test_pct, shuffle=shuffle)
    print(f"Splitting {total} entries:")
    print(f"  Train: {len(splits['train'])} ({train_pct*100:.0f}%)")
    print(f"  Val:   {len(splits['val'])} ({val_pct*100:.0f}%)")
    print(f"  Test:  {len(splits['test'])} ({test_pct*100:.0f}%)")
    print()

    for split_name, split_entries in splits.items():
        if not split_entries:
            continue
        out_path = f"{out_prefix}-{split_name}.jsonl"
        write_jsonl(split_entries, out_path, format=fmt)
        print(f"  ✅ {split_name}: {len(split_entries)} → {out_path}")


def _cmd_dataset_autolabel(args):
    """Auto-generate completions with confidence scoring."""
    input_file = args.input or input("Input dataset (JSONL): ").strip()
    output = args.output or input_file.replace(".jsonl", "-labeled.jsonl")
    api_base = args.api_base or "http://127.0.0.1:9000"
    model = args.model or ""
    max_tokens = args.max_tokens or 256
    temperature = args.temperature or 0.7
    batch_size = args.batch_size or 1
    threshold = args.confidence_threshold or 0.0

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    from mojollama.dataset import AutoLabeler, get_reader, compute_stats

    # Determine format
    fmt = detect_format(input_file)
    print(f"Input format: {FORMAT_DESCRIPTIONS.get(fmt, fmt)}")
    print(f"Model API: {api_base}")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    print(f"Batch size: {batch_size}")
    print()

    # Read entries
    reader = get_reader(input_file, format=fmt)
    entries = reader.read()
    total = len(entries)
    print(f"Loaded {total} entries")
    print()

    # Auto-label
    labeler = AutoLabeler(
        api_base=api_base,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    def _progress(done, total):
        print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)", end="\r", flush=True)

    print("Auto-labeling...")
    labeled = labeler.label_batch(entries, batch_size=batch_size, callback=_progress)
    print("\n")

    # Compute confidence scores
    from mojollama.dataset import compute_confidence
    for entry in labeled:
        entry.metadata["confidence"] = compute_confidence(entry)

    # Filter by confidence threshold
    high_conf = [e for e in labeled if e.metadata.get("confidence", 0) >= threshold]
    low_conf = [e for e in labeled if e.metadata.get("confidence", 0) < threshold]

    print(f"Results:")
    print(f"  Total entries:     {len(labeled)}")
    print(f"  High confidence:   {len(high_conf)} (≥{threshold:.2f})")
    print(f"  Low confidence:    {len(low_conf)} (<{threshold:.2f})")
    print()

    avg_conf = (
        sum(e.metadata.get("confidence", 0) for e in labeled) / max(len(labeled), 1)
    )
    print(f"  Average confidence: {avg_conf:.3f}")
    print()

    # Stats on the labeled data
    stats = compute_stats(labeled)
    print(f"  Estimated tokens: {stats.estimated_tokens:,}")
    print()

    # Save all
    from mojollama.dataset import write_jsonl
    write_jsonl(labeled, output, format="openai")
    print(f"✅ Saved {len(labeled)} labeled entries to {output}")

    # Save high-confidence subset
    if len(high_conf) != len(labeled):
        hc_out = output.replace(".jsonl", "-highconf.jsonl")
        write_jsonl(high_conf, hc_out, format="openai")
        print(f"✅ High-confidence subset ({len(high_conf)}) → {hc_out}")


# ─── Chat ──────────────────────────────────────────────────────────────

def cmd_chat(args):
    """Interactive chat with a model."""
    print_banner()
    print("[Chat] Interactive model chat")
    print()
    
    model = args.model or input("Model path (GGUF): ").strip()
    port = args.port or 9000
    
    if not os.path.exists(model):
        print(f"❌ Model not found: {model}")
        return
    
    # Check if already running
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        print(f"Server already running on port {port}")
        print(f"Open {CHAT_HTML} or use curl")
        return
    except:
        pass
    
    # Start server
    print(f"Starting llama.cpp server on port {port}...")
    print(f"Model: {model}")
    print()
    
    logfile = f"/tmp/mojollama-chat-{port}.log"
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(
            [SERVER_BIN, "-m", model, "-c", "4096",
             "-t", "64", "-tb", "32", "-b", "4096", "-ub", "4096",
             "-np", "8", "--mlock", "--cont-batching",
             "--port", str(port), "--host", "0.0.0.0", "--no-webui"],
            stdout=lf, stderr=subprocess.STDOUT
        )
    
    print("Waiting for server...")
    for _ in range(30):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            print(f"\n✅ Server ready at http://0.0.0.0:{port}")
            print(f"   Chat UI: {CHAT_HTML}")
            print("   API: curl http://127.0.0.1:{port}/v1/chat/completions")
            print("   -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}'")
            print("\nPress Ctrl+C to stop server")
            proc.wait()
            return
        except:
            pass
    
    with open(logfile) as lf:
        last_lines = lf.read().splitlines()[-5:]
    print(f"❌ Server failed to start (see {logfile})")
    for line in last_lines:
        print(f"  {line}")
    proc.kill()


# ─── Serve ─────────────────────────────────────────────────────────────

def cmd_serve(args):
    """Start the full MojoLlama API server with AutoBackend."""
    print_banner()
    print("[Serve] MojoLlama API server")
    print()
    
    os.environ["PORT"] = str(args.port or 9000)
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    
    sys.path.insert(0, f"{BASE_DIR}/src")
    from mojollama.server import main
    main()


# ─── Benchmark ─────────────────────────────────────────────────────────

def cmd_benchmark(args):
    """Benchmark model inference speed."""
    print_banner()
    print("[Benchmark] Model speed test")
    print()

    model = args.model or input("Model path (GGUF): ").strip()
    if not os.path.exists(model):
        print(f"❌ Model not found: {model}")
        return

    port = args.port or 8092
    prompt = args.prompt or "The meaning of life is"
    n_predict = args.n_predict or 128

    print(f"Model: {model}")
    print(f"Prompt: \"{prompt}\"")
    print(f"Tokens to generate: {n_predict}")
    print()

    # Start server
    print(f"Starting llama.cpp server on port {port}...")
    proc = subprocess.Popen(
        [SERVER_BIN, "-m", model, "-c", "4096",
         "-t", "64", "-tb", "32", "-b", "4096", "-ub", "4096",
         "-np", "8", "--mlock", "--cont-batching",
         "--port", str(port), "--host", "127.0.0.1", "--no-webui",
         "--metrics"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print("Waiting for server...", end=" ", flush=True)
    ready = False
    for _ in range(45):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            ready = True
            break
        except Exception:
            print(".", end="", flush=True)
    print()

    if not ready:
        print("\n❌ Server failed to start")
        proc.kill()
        return

    print(f"\n✅ Server ready at http://127.0.0.1:{port}")

    # Warmup run
    print("\nWarming up...")
    data = json.dumps({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 4, "temperature": 0, "stream": False
    }).encode()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=data, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=60)
    except Exception:
        pass

    # Benchmark run
    print(f"Benchmarking ({n_predict} tokens)...")
    data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict, "temperature": 0, "stream": False
    }).encode()

    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
        elapsed = time.time() - t0

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", len(content) // 4 or 1)

        prompt_tok_s = prompt_tokens / elapsed if elapsed > 0 else 0
        gen_tok_s = completion_tokens / elapsed if elapsed > 0 else 0

        print(f"\n  ─────────────────────────────")
        print(f"  Prompt tokens:       {prompt_tokens}")
        print(f"  Generated tokens:    {completion_tokens}")
        print(f"  Total time:          {elapsed:.2f}s")
        print(f"  Prompt processing:   {prompt_tok_s:.1f} tok/s")
        print(f"  Text generation:     {gen_tok_s:.1f} tok/s")
        print(f"  ─────────────────────────────")
        print(f"  Response preview:    {content[:100]}...")
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")

    proc.kill()
    print(f"\\n✅ Benchmark complete")


# ─── Evaluate ─────────────────────────────────────────────────────────

def cmd_evaluate(args):
    """Run benchmark evaluations (MMLU, GSM8K, CEval, etc.)."""
    print_banner()
    print("║     Evaluation Framework                      ║")
    print()

    if args.action == "download":
        return cmd_evaluate_download(args)

    if args.action == "list":
        from mojollama.eval.orchestrator import list_benchmarks
        print("Available benchmarks:")
        for b in list_benchmarks():
            print(f"  {b['id']:15s} — {b['description']}")
        print()
        print("Cached datasets:")
        from mojollama.eval.dataset import list_available_datasets
        cached = list_available_datasets()
        if cached:
            for name, info in cached.items():
                size_mb = info.get("size", 0) / 1024**2
                subjects = info.get("subjects", 0)
                extra = f" ({subjects} subjects)" if subjects else ""
                print(f"  \u2705 {name:15s} {size_mb:.1f} MB{extra}")
        else:
            print("  (none cached \u2014 run `evaluate download` first)")
        return

    # Ensure we have a backend running
    llama_port = args.port or 8081
    backend_url = f"http://127.0.0.1:{llama_port}"

    from mojollama.eval.base import check_backend_alive
    if not check_backend_alive(backend_url):
        print(f"\u274c Backend at {backend_url} is not responding.")
        print(f"   Start a server first: mojollama-studio serve --model model.gguf")
        return

    # Determine which benchmarks to run
    benchmarks_to_run = []
    if args.benchmarks:
        benchmarks_to_run = [b.lower() for b in args.benchmarks]
    elif args.all:
        from mojollama.eval.orchestrator import BENCHMARKS
        benchmarks_to_run = list(BENCHMARKS.keys())
    else:
        print("Specify --benchmarks or --all. Use `evaluate list` to see available benchmarks.")
        return

    from mojollama.eval.orchestrator import run_benchmarks, format_results

    results = run_benchmarks(benchmarks_to_run, backend_url,
                             args.model or "unknown",
                             max_samples=args.max_samples or 0)

    print()
    print(format_results(results))

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\\n\u2705 Results saved to {args.output}")


def cmd_evaluate_download(args):
    """Download evaluation datasets."""
    from mojollama.eval.orchestrator import download_all_datasets
    results = download_all_datasets()
    ok_count = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\\n{'─'*40}")
    print(f"  {ok_count}/{total} datasets downloaded successfully")
    return results


# ─── Auto-tune ─────────────────────────────────────────────────────────

def cmd_autotune(args):
    """Auto-tune server settings for this hardware."""
    print("╔══════════════════════════════════════════════╗")
    print("║      MojoLlama AutoTuner v0.2.0              ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Deep mode: detect hardware first
    if args.deep:
        from mojollama.detect import detect_hardware, summary, recommend_llamacpp_flags, format_recommendation
        hw = detect_hardware()
        print(summary(hw))
        print()
        flags = recommend_llamacpp_flags(hw)
        print("Recommended flags from hardware detection:")
        print(f"  {format_recommendation(flags)}")
        print()

    config = autotune(args.model or 
                      "/tmp/tl-Q4_0.gguf",
                      quick=args.quick)

    if config:
        # Merge hardware detection findings into config
        if args.deep:
            from mojollama.detect import detect_hardware, recommend_llamacpp_flags
            hw = detect_hardware()
            flags = recommend_llamacpp_flags(hw)
            config["hardware_detection"] = {
                "cpu": hw["cpu"],
                "gpu": hw["gpu"]["primary"],
            }
            # Apply hardware-recommended settings
            hs = config["llama_server"]
            for k in ["threads", "batch_size", "n_parallel", "mlock", "flash_attn", "cpu_mask"]:
                if k in flags:
                    hs[k] = flags[k]
            if hw["gpu"]["primary"]:
                hs["gpu"] = flags.get("gpu")
                hs["gpu_layers"] = flags.get("gpu_layers", -1)

        save_config(config)
        print_config(config)

        # Show the generated server config
        from mojollama.backends import build_server_cmd
        cmd = build_server_cmd(
            "/tmp/llama.cpp/build/bin/llama-server",
            args.model or "model.gguf",
            8081
        )
        print()
        print("Recommended llama-server command:")
        print(f"  {' '.join(cmd)}")
    else:
        print("❌ Auto-tuning failed")


# ─── Info ──────────────────────────────────────────────────────────────

def cmd_info(args):
    """Show system info and available resources."""
    print_banner()
    
    print("System:")
    import psutil
    print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.0f} GB total, "
          f"{psutil.virtual_memory().available / 1024**3:.0f} GB free")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print()
    
    print("Backends:")
    backends = []
    if os.path.exists(SERVER_BIN):
        backends.append("llama.cpp ✓")
    try:
        import max
        backends.append("MAX ✓")
    except:
        backends.append("MAX ✗")
    if os.path.exists(FINETUNE_BIN):
        backends.append("finetune ✓")
    for b in backends:
        print(f"  {b}")
    print()
    
    print("Models:")
    for f in sorted(Path(BASE_DIR).glob("*.gguf")):
        size = f.stat().st_size / 1024**3
        print(f"  {f.name} ({size:.1f} GB)")
    
    print()
    print("Studio commands:")
    print("  train     Fine-tune a model with LoRA")
    print("  export    Convert HF model to GGUF")
    print("  dataset   Create/manage training data")
    print("  merge     Merge LoRA adapter into base model")
    print("  chat      Interactive chat")
    print("  serve     Full API server")
    print("  benchmark Benchmark model inference speed")
    print("  info      This info")


# ─── Quantizer Wrappers ────────────────────────────────────────────────

def cmd_quantize(args):
    """Quantize GGUF → GGUF using llama-quantize."""
    quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"

    model = args.model
    outtype = args.type
    outfile = args.output or model.replace(".gguf", f"-{outtype}.gguf")
    nthreads = args.threads

    cmd = [quantize_bin]
    if args.allow_requantize:
        cmd.append("--allow-requantize")
    if args.pure:
        cmd.append("--pure")
    if args.leave_output:
        cmd.append("--leave-output-tensor")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.imatrix:
        cmd.extend(["--imatrix", args.imatrix])
    if args.override_kv:
        for kv in args.override_kv:
            cmd.extend(["--override-kv", kv])
    cmd.extend([model, outfile, outtype])
    if nthreads and nthreads > 0:
        cmd.append(str(nthreads))

    print_banner()
    print(f"[Quantize] {model} → {outfile} ({outtype})")
    print()
    subprocess.run(cmd, check=True)


def cmd_imatrix(args):
    """Generate importance matrix using llama-imatrix."""
    imatrix_bin = "/tmp/llama.cpp/build/bin/llama-imatrix"

    cmd = [imatrix_bin, "-m", args.model]
    if args.data:
        cmd.extend(["-f", args.data])
    if args.output:
        cmd.extend(["-o", args.output])
    if args.threads and args.threads > 0:
        cmd.extend(["-t", str(args.threads)])
    if args.ctx_size:
        cmd.extend(["-c", str(args.ctx_size)])

    print_banner()
    print("[IMatrix] Generating importance matrix...")
    print()
    subprocess.run(cmd, check=True)


def cmd_dynamic_quant(args):
    """Dynamic quantization: select per-tensor quant types based on importance matrix."""
    import re, math, tempfile
    
    quant_types = {
        'Q2_K': 2.5, 'Q3_K_S': 3.0, 'Q3_K_M': 3.4, 'Q4_0': 4.0,
        'Q4_K_S': 4.4, 'Q4_K_M': 4.5, 'Q5_0': 5.0, 'Q5_K_M': 5.5,
        'Q6_K': 6.0, 'Q8_0': 8.0,
    }
    refined = {'Q6_K': 6.0, 'Q5_K_M': 5.5, 'Q4_K_M': 4.5, 'Q3_K_S': 3.0, 'Q2_K': 2.5}
    
    # Parse imatrix file
    tensor_importance = {}
    with open(args.imatrix) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    tensor_importance[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    
    if not tensor_importance:
        print("[DynamicQuant] ERROR: No valid entries in imatrix file")
        return
    
    # Group tensors by type
    groups = {'attention': [], 'ffn': [], 'output': [], 'embedding': [], 'norm': [], 'other': []}
    for tname, imp in tensor_importance.items():
        base = os.path.splitext(tname)[0]
        base_clean = re.sub(r'blk\.\d+\.', '', base)
        if any(x in base_clean for x in ['attn_q', 'attn_k', 'attn_v', 'attn_o', 'attention', 'self_attn']):
            groups['attention'].append((tname, imp))
        elif any(x in base_clean for x in ['ffn_gate', 'ffn_up', 'ffn_down', 'mlp', 'feed_forward']):
            groups['ffn'].append((tname, imp))
        elif any(x in base_clean for x in ['output', 'lm_head']):
            groups['output'].append((tname, imp))
        elif any(x in base_clean for x in ['embed', 'tok_embeddings', 'token_embd']):
            groups['embedding'].append((tname, imp))
        elif 'norm' in base_clean:
            groups['norm'].append((tname, imp))
        else:
            groups['other'].append((tname, imp))
    
    # Assign quant types
    assignments = {}
    if args.target_bpw:
        # Target BPW mode: grid search for best threshold split
        all_sorted = sorted(tensor_importance.items(), key=lambda x: x[1])
        n = len(all_sorted)
        best_error = float('inf')
        best_thresholds = None
        best_assignment = {}
        bpw_target = float(args.target_bpw)
        
        for t1 in range(0, 101, 5):
            for t2 in range(t1, 101, 5):
                for t3 in range(t2, 101, 5):
                    p1, p2, p3 = t1 / 100, t2 / 100, t3 / 100
                    total_bpw = 0
                    temp_assign = {}
                    for i, (name, imp) in enumerate(all_sorted):
                        pct = i / n
                        if pct <= p1:
                            qt = 'Q6_K'
                        elif pct <= p2:
                            qt = 'Q5_K_M'
                        elif pct <= p3:
                            qt = 'Q4_K_M'
                        else:
                            qt = 'Q3_K_S' if args.extreme else 'Q2_K'
                        temp_assign[name] = qt
                        total_bpw += refined[qt]
                    avg_bpw = total_bpw / n
                    error = abs(avg_bpw - bpw_target)
                    if error < best_error:
                        best_error = error
                        best_thresholds = (p1, p2, p3)
                        best_assignment = temp_assign
        assignments = best_assignment
        print(f"[DynamicQuant] Target BPW: {bpw_target}, achieved: {bpw_target - best_error:.2f} (error: {best_error:.2f})")
        p1, p2, p3 = best_thresholds
        print(f"[DynamicQuant] Thresholds: Q6_K<={p1*100:.0f}%, Q5_K_M<={p2*100:.0f}%, Q4_K_M<={p3*100:.0f}%, Q3_K_S>{p3*100:.0f}%")
    else:
        # Standard mode: per-group 30/40/30 split
        for group_name, group_tensors in groups.items():
            if not group_tensors:
                continue
            sorted_tensors = sorted(group_tensors, key=lambda x: x[1])
            n = len(sorted_tensors)
            for i, (tname, _) in enumerate(sorted_tensors):
                pct = i / n
                if pct <= 0.3:
                    assignments[tname] = 'Q6_K'
                elif pct <= 0.7:
                    assignments[tname] = 'Q4_K_M'
                else:
                    assignments[tname] = 'Q3_K_S'
    
    # Count
    qt_counts = {}
    for qt in assignments.values():
        qt_counts[qt] = qt_counts.get(qt, 0) + 1
    print(f"[DynamicQuant] Quantization plan: {', '.join(f'{k}: {v}' for k, v in sorted(qt_counts.items()))}")
    
    if args.verbose or args.dry_run:
        print(f"\n  Tensor assignments:")
        for tname, qt in sorted(assignments.items()):
            print(f"    {tname} -> {qt}")
    
    if args.dry_run:
        print(f"\n[DynamicQuant] Dry-run mode. Would run:")
        print(f"  llama-quantize --tensor-type-file <temp_file> {args.model} {args.output} COPY {args.threads or 32}")
        return
    
    # Write tensor-type file and run quantization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, prefix='dynquant_') as f:
        for tname, qt in sorted(assignments.items()):
            f.write(f"{tname} {qt}\n")
        ttf_path = f.name
    
    quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"
    outfile = args.output or args.model.replace('.gguf', '-dynamic.gguf')
    cmd = [quantize_bin, "--allow-requantize", "--tensor-type-file", ttf_path,
           args.model, outfile, "COPY"]
    if args.threads and args.threads > 0:
        cmd.append(str(args.threads))
    
    print(f"[DynamicQuant] Quantizing -> {outfile}")
    subprocess.run(cmd, check=True)
    os.unlink(ttf_path)
    print(f"[DynamicQuant] Done: {outfile}")


def cmd_nf4_wrapper(args):
    """Convert to NF4 (wraps quantizer.py)."""
    from mojollama.quantizer import main as quantizer_main
    import sys
    sys.argv = ["quantizer.py", "nf4", args.model,
                "--output", args.output or "",
                "--block-size", str(args.block_size)]
    sys.argv = [a for a in sys.argv if a]
    return quantizer_main()


def cmd_quant_types(args):
    """List supported quantization types."""
    from mojollama.quantizer import main as quantizer_main
    import sys
    sys.argv = ["quantizer.py", "list-types"]
    return quantizer_main()


# ─── Hub commands (from exporter) ───────────────────────────────────

def cmd_hub_login(args):
    """Login to HuggingFace Hub."""
    from mojollama.exporter import hub_login
    print_banner()
    print("║     HuggingFace Hub — Login                  ║")
    print()
    hub_login(token=args.token or "")


def cmd_hub_whoami(args):
    """Show HuggingFace user info."""
    from mojollama.exporter import hub_whoami
    user = hub_whoami()
    if user:
        print(f"User: {user.get('name', 'unknown')}")
        print(f"Email: {user.get('email', 'unknown')}")
    else:
        print("❌ Not logged in to HuggingFace Hub")


def cmd_hub_push(args):
    """Push model to HuggingFace Hub."""
    from mojollama.exporter import hub_push_model
    print_banner()
    print("║     HuggingFace Hub — Push Model             ║")
    print()
    model_path = args.model or input("Model path: ").strip()
    repo_id = args.repo or input("HF repo ID (e.g. username/model): ").strip()
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    metadata = {"studio_version": STUDIO_VERSION}
    if args.quant: metadata["quantization"] = args.quant
    if args.params: metadata["parameters"] = args.params
    if args.description: metadata["description"] = args.description
    result = hub_push_model(
        model_path=model_path, repo_id=repo_id,
        private=args.private,
        commit_message=args.message or f"Upload via MojoLlama Studio",
        metadata=metadata,
    )
    if result:
        print(f"\n✅ Model available at: {result}")


def cmd_hub_push_adapter(args):
    """Push LoRA adapter to HuggingFace Hub."""
    from mojollama.exporter import hub_push_adapter
    print_banner()
    print("║  HuggingFace Hub — Push Adapter              ║")
    print()
    adapter_path = args.adapter or input("Adapter path: ").strip()
    base_model = args.base_model or input("Base model name: ").strip()
    repo_id = args.repo or input("HF repo ID: ").strip()
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found: {adapter_path}")
        return
    metadata = {"base_model": base_model}
    if args.rank: metadata["lora_rank"] = args.rank
    if args.alpha: metadata["lora_alpha"] = args.alpha
    result = hub_push_adapter(
        adapter_path=adapter_path, base_model=base_model,
        repo_id=repo_id, private=args.private, metadata=metadata,
    )
    if result:
        print(f"\n✅ Adapter available at: {result}")


# ─── Export format commands ────────────────────────────────────────

def cmd_export_safetensors(args):
    """Convert GGUF model to safetensors format."""
    from mojollama.exporter import gguf_to_safetensors
    print_banner()
    print("[Export] GGUF → Safetensors")
    print()
    gguf_path = args.model or input("GGUF model path: ").strip()
    output_dir = args.output or gguf_path.replace(".gguf", "-safetensors")
    result = gguf_to_safetensors(
        gguf_path=gguf_path, output_dir=output_dir,
        dtype=args.dtype or "float16", shard_size=args.shard_size or "2GB",
    )
    if result:
        print(f"\n✅ Saved to: {result}")


def cmd_export_onnx(args):
    """Convert GGUF model to ONNX format."""
    from mojollama.exporter import gguf_to_onnx
    print_banner()
    print("[Export] GGUF → ONNX")
    print()
    gguf_path = args.model or input("GGUF model path: ").strip()
    output_path = args.output or gguf_path.replace(".gguf", ".onnx")
    result = gguf_to_onnx(
        gguf_path=gguf_path, output_path=output_path,
        opset=args.opset or 17, max_seq_len=args.max_seq_len or 2048,
    )
    if result:
        print(f"\n✅ Saved to: {result}")


# ─── Checkpoint commands ───────────────────────────────────────────

def cmd_checkpoint_save(args):
    """Save a training checkpoint."""
    from mojollama.exporter import TrainingCheckpoint
    print_banner()
    print("[Checkpoint] Save")
    print()
    ckpt = TrainingCheckpoint(args.checkpoint_dir or "checkpoints")
    result = ckpt.save(
        step=args.step or 0, epoch=args.epoch or 0, loss=args.loss or 0.0,
        model_path=args.model or None,
    )
    if result:
        print(f"\n✅ Checkpoint saved to: {result}")


def cmd_checkpoint_load(args):
    """Load and display training checkpoint info."""
    from mojollama.exporter import TrainingCheckpoint
    print_banner()
    print("[Checkpoint] Load")
    print()
    ckpt = TrainingCheckpoint(args.checkpoint_dir or "checkpoints")
    data = ckpt.load()
    if data is None:
        print("❌ No checkpoint found")
        return
    print(f"  Step: {data.get('step', '?')}")
    print(f"  Epoch: {data.get('epoch', '?')}")
    print(f"  Loss: {data.get('loss', '?')}")
    print(f"  Best Loss: {data.get('best_loss', '?')}")
    if data.get("config"):
        print(f"  Config:")
        for k, v in data["config"].items():
            print(f"    {k}: {v}")


def cmd_checkpoint_list(args):
    """List all training checkpoints."""
    from mojollama.exporter import TrainingCheckpoint
    print_banner()
    print("[Checkpoint] List")
    print()
    checkpoints = TrainingCheckpoint.list_checkpoints(args.checkpoint_dir or ".")
    if not checkpoints:
        print("No checkpoints found")
        return
    print(f"Found {len(checkpoints)} checkpoint(s):")
    print()
    for ckpt in checkpoints:
        name = ckpt.get("name", "?")
        loss = ckpt.get("loss", "?")
        step = ckpt.get("step", "?")
        status = "✅" if "error" not in ckpt else "❌"
        print(f"  {status} {name} — step {step}, loss {loss}")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MojoLlama Studio")
    parser.add_argument("--version", action="store_true", help="Show version")
    
    sub = parser.add_subparsers(dest="command", help="Command")
    
    # train
    p_train = sub.add_parser("train", help="Fine-tune a model (LoRA, QLoRA, DoRA, GaLore, DPO, GRPO, etc.)")
    p_train.add_argument("--model", help="Model path (GGUF)")
    p_train.add_argument("--data", help="Training data (JSONL, JSON, or text)")
    p_train.add_argument("--method", choices=list(TRAINING_METHODS.keys()),
                        default="lora", help="Training method (default: lora)")
    p_train.add_argument("--lora-out", default="adapter.gguf")
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight-decay", type=float, default=0.0)
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch", type=int, default=4)
    p_train.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length")
    p_train.add_argument("--save-steps", type=int, default=0,
                        help="Save checkpoint every N steps")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--warmup-steps", type=int, default=0)
    p_train.add_argument("--dataset-format", default="auto",
                        choices=["auto", "alpaca", "sharegpt", "jsonl", "text"])
    p_train.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to use (0 = all)")
    p_train.add_argument("--template", default="alpaca",
                        choices=["alpaca", "sharegpt", "preference", "text"])
    # Method-specific options
    p_train.add_argument("--dpo-beta", type=float, default=0.1, help="DPO KL penalty")
    p_train.add_argument("--orpo-lambda", type=float, default=0.1, help="ORPO lambda")
    p_train.add_argument("--simpo-gamma", type=float, default=0.5, help="SimPO reward margin")
    p_train.add_argument("--grpo-group-size", type=int, default=8, help="GRPO group size")
    p_train.add_argument("--grpo-clip", type=float, default=0.2, help="GRPO clip epsilon")
    p_train.add_argument("--galore-rank", type=int, default=128,
                        help="GaLore projection rank")
    p_train.add_argument("--list-methods", action="store_true",
                        help="List available training methods")
    p_train.add_argument("--resume-from", help="Resume from checkpoint directory")
    p_train.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for saving checkpoints")
    p_train.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N steps")
    
    # export
    p_export = sub.add_parser("export", help="Convert HF model to GGUF")
    p_export.add_argument("--hf", help="HF model ID or path")
    p_export.add_argument("--model", help="Alias for --hf")
    p_export.add_argument("--outtype", default="q4_0")
    p_export.add_argument("--outfile", help="Output GGUF path")
    p_export.add_argument("--remote", action="store_true", help="Download from HF")
    p_export.add_argument("--vocab-only", action="store_true")
    p_export.add_argument("--mojo-bin", action="store_true", help="Also generate .bin")

    # convert (alias for export)
    p_convert = sub.add_parser("convert", help="Alias for export: Convert HF model to GGUF")
    p_convert.add_argument("--hf", help="HF model ID or path")
    p_convert.add_argument("--model", help="Alias for --hf")
    p_convert.add_argument("--outtype", default="q4_0")
    p_convert.add_argument("--outfile", help="Output GGUF path")
    p_convert.add_argument("--remote", action="store_true", help="Download from HF")
    p_convert.add_argument("--vocab-only", action="store_true")
    p_convert.add_argument("--mojo-bin", action="store_true", help="Also generate .bin")
    
    # merge
    p_merge = sub.add_parser("merge", help="Merge LoRA adapter into base model")
    p_merge.add_argument("--base", required=True, help="Base model (GGUF)")
    p_merge.add_argument("--lora", required=True, help="LoRA adapter (GGUF)")
    p_merge.add_argument("--output", default="merged.gguf", help="Output path")
    p_merge.add_argument("--type", default="q4_0", help="Output quantization type")
    p_merge.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # dataset
    p_data = sub.add_parser("dataset", help="Manage datasets")
    p_data.add_argument("action", choices=[
        "create", "view", "convert", "info", "stats",
        "stream", "list", "split", "auto-label",
    ], help="Dataset action")
    p_data.add_argument("--output", help="Output file")
    p_data.add_argument("--dataset", help="Dataset path")
    p_data.add_argument("--input", help="Input file")
    p_data.add_argument("--format", help="Dataset format (alpaca/sharegpt/openai/preference/jsonl)")
    p_data.add_argument("--model", help="Model name for auto-label")
    p_data.add_argument("--port", type=int, default=9000, help="Server port for auto-label API")
    p_data.add_argument("--api-base", help="API base URL for auto-label (default: http://127.0.0.1:9000)")
    p_data.add_argument("--max-tokens", type=int, default=256, help="Max tokens for auto-label")
    p_data.add_argument("--temperature", type=float, default=0.7, help="Temperature for auto-label")
    p_data.add_argument("--count", type=int, default=10, help="Number of samples to show (view/stream)")
    p_data.add_argument("--max-samples", type=int, help="Max samples to read for stats")
    p_data.add_argument("--batch-size", type=int, default=1, help="Batch size for auto-label")
    p_data.add_argument("--confidence-threshold", type=float, default=0.0,
                        help="Min confidence threshold for auto-label")
    p_data.add_argument("--directory", help="Directory to scan for list")
    p_data.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    p_data.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p_data.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    p_data.add_argument("--output-prefix", help="Output prefix for split files")
    p_data.add_argument("--no-shuffle", action="store_true", help="Disable shuffle for split")

    # quantize (wrapper around quantizer.py)
    p_quant = sub.add_parser("quantize", help="Quantize a GGUF model to a different type (all K/IQ quants, imatrix, NF4)")
    p_quant.add_argument("model", help="Path to input GGUF model")
    p_quant.add_argument("--type", "-t", default="Q4_K_M", dest="type",
                         help="Quantization type (default: Q4_K_M)")
    p_quant.add_argument("--output", "-o", help="Output path")
    p_quant.add_argument("--imatrix", help="Importance matrix file for guided quantization")
    p_quant.add_argument("--threads", type=int, default=0, help="Thread count (0 = auto)")
    p_quant.add_argument("--allow-requantize", action="store_true",
                         help="Allow requantizing already quantized tensors")
    p_quant.add_argument("--pure", action="store_true",
                         help="Disable K-quant mixtures, pure type")
    p_quant.add_argument("--leave-output", action="store_true",
                         help="Leave output.weight unquantized")
    p_quant.add_argument("--dry-run", action="store_true",
                         help="Calculate size without quantizing")
    p_quant.add_argument("--override-kv", action="append",
                         help="Override model metadata key=type:val (can be repeated)")

    # imatrix
    p_imatrix = sub.add_parser("imatrix", help="Generate importance matrix for better quantization")
    p_imatrix.add_argument("model", help="Path to GGUF model")
    p_imatrix.add_argument("--data", "-f", help="Calibration data file (text)")
    p_imatrix.add_argument("--output", "-o", help="Output imatrix file path")
    p_imatrix.add_argument("--threads", "-t", type=int, default=0, help="Number of threads")
    p_imatrix.add_argument("--ctx-size", "-c", type=int, default=512, help="Context size")

    p_dyn = sub.add_parser("dynamic-quantize", help="Dynamic per-tensor quantization guided by importance matrix")
    p_dyn.add_argument("model", help="Path to input GGUF model")
    p_dyn.add_argument("--imatrix", required=True, help="Importance matrix file (.dat)")
    p_dyn.add_argument("--output", "-o", help="Output GGUF path")
    p_dyn.add_argument("--target-bpw", type=float, help="Target bits-per-weight (auto thresholds)")
    p_dyn.add_argument("--extreme", action="store_true", help="Use Q2_K for lowest importance (instead of Q3_K_S)")
    p_dyn.add_argument("--dry-run", action="store_true", help="Preview assignments without quantizing")
    p_dyn.add_argument("--verbose", action="store_true", help="Print per-tensor assignments")
    p_dyn.add_argument("--threads", "-t", type=int, default=0, help="Number of threads")

    p_nf4 = sub.add_parser("nf4", help="Convert to NF4 (NormalFloat4) for QLoRA")
    p_nf4.add_argument("model", help="Path to input GGUF model")
    p_nf4.add_argument("--output", "-o", help="Output path")
    p_nf4.add_argument("--block-size", type=int, default=64,
                        help="NF4 block size (default: 64)")

    # quant-types
    sub.add_parser("quant-types", help="List all supported quantization types")
    
    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--model", help="Model path (GGUF)")
    p_chat.add_argument("--port", type=int, default=9000)

    # serve
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--port", type=int, default=9000)
    p_serve.add_argument("--model", help="Model path override")
    
    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark model inference speed")
    p_bench.add_argument("--model", help="Model path (GGUF)")
    p_bench.add_argument("--port", type=int, default=8092, help="Server port")
    p_bench.add_argument("--prompt", default="The meaning of life is", help="Test prompt")
    p_bench.add_argument("--n-predict", type=int, default=128, help="Tokens to generate")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run benchmark evaluations (MMLU, GSM8K, CEval, etc.)")
    p_eval.add_argument("action", nargs="?", choices=["download", "list"], default=None,
                        help="'download' to fetch datasets, 'list' to show available")
    p_eval.add_argument("--benchmarks", "-b", nargs="+",
                        help="Benchmarks to run: mmlu gsm8k ceval hellaswag arc bbh humaneval")
    p_eval.add_argument("--all", "-a", action="store_true",
                        help="Run all available benchmarks")
    p_eval.add_argument("--model", "-m", help="Model name (for display)")
    p_eval.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per category (0 = all)")
    p_eval.add_argument("--port", type=int, default=8081,
                        help="llama.cpp backend port")
    p_eval.add_argument("--output", "-o", help="Save results to JSON file")
    
    # autotune
    p_tune = sub.add_parser("autotune", help="Auto-tune server settings for this hardware")
    p_tune.add_argument("--model", "-m", help="Model to benchmark with (default: TinyLlama Q4_0)")
    p_tune.add_argument("--quick", "-q", action="store_true", help="Faster sweep (fewer combos)")
    p_tune.add_argument("--deep", "-d", action="store_true",
                        help="Deep hardware detection (CPU features, GPU, CUDA, ROCm, etc.)")
    
    # info
    sub.add_parser("info", help="System info")
    
    # ── Hub commands ─────────────────────────────────────────────
    p_hub_login = sub.add_parser("hub-login", help="Login to HuggingFace Hub")
    p_hub_login.add_argument("--token", help="HF API token")
    
    sub.add_parser("hub-whoami", help="Show HuggingFace user info")
    
    p_hub_push = sub.add_parser("hub-push", help="Push model to HuggingFace Hub")
    p_hub_push.add_argument("--model", "-m", help="Model path")
    p_hub_push.add_argument("--repo", help="HF repo ID")
    p_hub_push.add_argument("--message", help="Commit message")
    p_hub_push.add_argument("--private", action="store_true", help="Create private repo")
    p_hub_push.add_argument("--quant", help="Quantization type (metadata)")
    p_hub_push.add_argument("--params", help="Parameter count (metadata)")
    p_hub_push.add_argument("--description", help="Model description")
    
    p_hub_adapter = sub.add_parser("hub-push-adapter", help="Push LoRA adapter to HF Hub")
    p_hub_adapter.add_argument("--adapter", help="Adapter GGUF path")
    p_hub_adapter.add_argument("--base-model", help="Base model name")
    p_hub_adapter.add_argument("--repo", help="HF repo ID")
    p_hub_adapter.add_argument("--private", action="store_true")
    p_hub_adapter.add_argument("--rank", type=int, help="LoRA rank")
    p_hub_adapter.add_argument("--alpha", type=float, help="LoRA alpha")
    
    # ── Export format commands ────────────────────────────────────
    p_st = sub.add_parser("export-safetensors", help="Convert GGUF to safetensors")
    p_st.add_argument("--model", "-m", help="GGUF model path")
    p_st.add_argument("--output", "-o", help="Output directory")
    p_st.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p_st.add_argument("--shard-size", default="2GB", help="Shard size (1GB, 2GB, 5GB, NO)")
    
    p_onnx = sub.add_parser("export-onnx", help="Convert GGUF to ONNX")
    p_onnx.add_argument("--model", "-m", help="GGUF model path")
    p_onnx.add_argument("--output", "-o", help="Output path")
    p_onnx.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p_onnx.add_argument("--max-seq-len", type=int, default=2048)
    
    # ── Checkpoint commands ──────────────────────────────────────
    p_ckpt_save = sub.add_parser("checkpoint-save", help="Save training checkpoint")
    p_ckpt_save.add_argument("--checkpoint-dir", default="checkpoints")
    p_ckpt_save.add_argument("--model", help="Model file path")
    p_ckpt_save.add_argument("--step", type=int, default=0)
    p_ckpt_save.add_argument("--epoch", type=int, default=0)
    p_ckpt_save.add_argument("--loss", type=float, default=0.0)
    
    p_ckpt_load = sub.add_parser("checkpoint-load", help="Load training checkpoint")
    p_ckpt_load.add_argument("--checkpoint-dir", default="checkpoints")
    
    p_ckpt_list = sub.add_parser("checkpoint-list", help="List training checkpoints")
    p_ckpt_list.add_argument("--checkpoint-dir", default=".", help="Base directory")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"MojoLlama Studio v{STUDIO_VERSION}")
        return
    
    if not args.command:
        parser.print_help()
        print("\nCommands:")
        print("  train     Fine-tune a model with LoRA")
        print("  export    Convert HuggingFace model to GGUF")
        print("  convert   Alias for export")
        print("  dataset   Create, view, auto-label, and manage training datasets")
        print("  merge     Merge LoRA adapter into base GGUF model")
        print("  quantize  Quantize GGUF to different type (K/IQ quants, imatrix)")
        print("  imatrix   Generate importance matrix for guided quantization")
        print("  nf4       Convert to NF4 (NormalFloat4) for QLoRA")
        print("  quant-types  List supported quantization types")
        print("  chat      Interactive chat with a model")
        print("  serve     Start the full MojoLlama API server")
        print("  benchmark Benchmark model inference speed")
        print("  evaluate  Run benchmark evaluations (MMLU, GSM8K, CEval)")
        print("  autotune  Auto-tune server settings for this hardware")
        print("  info      Show system info")
        print("  hub-login          Login to HuggingFace Hub")
        print("  hub-whoami         Show HuggingFace user info")
        print("  hub-push           Push model to HuggingFace Hub")
        print("  hub-push-adapter   Push LoRA adapter to HF Hub")
        print("  export-safetensors Convert GGUF to safetensors")
        print("  export-onnx        Convert GGUF to ONNX")
        print("  checkpoint-save    Save training checkpoint")
        print("  checkpoint-load    Load training checkpoint")
        print("  checkpoint-list    List training checkpoints")
        return
    
    commands = {
        "train": cmd_train,
        "export": cmd_export,
        "convert": cmd_export,
        "dataset": cmd_dataset,
        "merge": cmd_merge,
        "quantize": cmd_quantize,
        "dynamic-quantize": cmd_dynamic_quant,
        "imatrix": cmd_imatrix,
        "nf4": cmd_nf4_wrapper,
        "quant-types": cmd_quant_types,
        "chat": cmd_chat,
        "serve": cmd_serve,
        "benchmark": cmd_benchmark,
        "evaluate": cmd_evaluate,
        "autotune": cmd_autotune,
        "info": cmd_info,
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
