#!/usr/bin/env python3
"""MojoLlama Studio — train, export, dataset, merge, chat, benchmark.

Usage:
  mojollama-studio train --model model.gguf --data dataset.jsonl
  mojollama-studio export --hf meta-llama/Llama-3.2-1B --outtype q4_0
  mojollama-studio dataset create --output data.jsonl
  mojollama-studio dataset auto-label --model model.gguf --input prompts.jsonl
  mojollama-studio merge --base model.gguf --lora adapter.gguf --output merged.gguf
  mojollama-studio chat --model model.gguf
  mojollama-studio serve --model model.gguf --port 8080
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

def cmd_train(args):
    """Fine-tune a model using llama.cpp finetune."""
    print_banner()
    print("[Train] Fine-tuning pipeline")
    print()
    
    if not os.path.exists(FINETUNE_BIN):
        print("Building llama.cpp training tools...")
        subprocess.run(
            ["cmake", "--build", f"{LLAMA_CPP}/build", "--target", "llama-finetune"],
            cwd=f"{LLAMA_CPP}/build", check=True
        )
    
    model = args.model or input("Model path (GGUF): ").strip()
    data = args.data or input("Training data (JSONL): ").strip()
    lora_out = args.lora_out or "lora-adapter.gguf"
    lora_rank = args.lora_rank or 16
    lora_alpha = args.lora_alpha or 32
    lr = args.lr or "1e-4"
    steps = args.steps or 100
    batch = args.batch or 4
    
    print(f"\nModel: {model}")
    print(f"Data: {data}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"LR: {lr}, steps: {steps}, batch size: {batch}")
    print(f"Output adapter: {lora_out}")
    
    if not os.path.exists(data):
        print(f"\n❌ Training data not found: {data}")
        print("Create one with: mojollama-studio dataset create")
        return
    
    cmd = [
        FINETUNE_BIN, "--model", model, "--train-data", data,
        "--lora-rank", str(lora_rank), "--lora-alpha", str(lora_alpha),
        "--learning-rate", lr, "--steps", str(steps),
        "--batch-size", str(batch), "--lora-out", lora_out
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd)
    print(f"\n✅ LoRA adapter saved to: {lora_out}")


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
    """Merge LoRA adapter into base GGUF model."""
    print_banner()
    print("[Merge] LoRA adapter merge")
    print()

    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType, dequantize, quantize, Keys

    base_path = args.base
    lora_path = args.lora
    output_path = args.output
    out_quant = args.type

    # Read base model
    print(f"Reading base model: {base_path}")
    base_reader = GGUFReader(base_path)
    print(f"  {len(base_reader.tensors)} tensors")

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
    """Create, view, and manage training datasets."""
    print_banner()
    print("[Dataset] Training data management")
    print()
    
    if args.action == "create":
        output = args.output or "dataset.jsonl"
        print(f"Creating dataset: {output}")
        print("Enter prompts one per line. Empty line to finish.\n")
        
        samples = []
        try:
            while True:
                inp = input("Prompt: ").strip()
                if not inp:
                    break
                # Use model for completion
                completion = input("Completion: ").strip()
                if not completion:
                    # Use model to generate completion (placeholder)
                    completion = "(pending)"
                samples.append({"prompt": inp, "completion": completion})
                print(f"  → Sample {len(samples)} saved\n")
        except (EOFError, KeyboardInterrupt):
            print()
        
        if samples:
            with open(output, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            print(f"\n✅ Saved {len(samples)} samples to {output}")
        else:
            print("\n⚠️  No samples saved")
    
    elif args.action == "view":
        path = args.dataset or args.input or input("Dataset path: ").strip()
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"Dataset: {path}")
        print(f"Samples: {len(samples)}")
        print()
        for i, s in enumerate(samples[:5]):
            print(f"[{i+1}] Prompt: {s.get('prompt','')[:60]}...")
            print(f"    Completion: {s.get('completion','')[:60]}...")
            print()
    
    elif args.action == "convert":
        src = args.input or input("Source format (csv/jsonl/alpaca): ").strip()
        dst = args.output or "converted.jsonl"
        fmt = args.format or "alpaca"
        print(f"Converting {src} → {dst} (format: {fmt})")
        print("Supported formats: alpaca, sharegpt, csv, json")
        print("Coming soon: automated format detection and conversion")

    elif args.action == "auto-label":
        model = args.model or input("Model path (GGUF): ").strip()
        input_file = args.input or input("Prompts file (JSONL with 'prompt' field): ").strip()
        output = args.output or input_file.replace(".jsonl", "-labeled.jsonl")
        port = args.port or 8090

        if not os.path.exists(model):
            print(f"❌ Model not found: {model}")
            return
        if not os.path.exists(input_file):
            print(f"❌ Prompts file not found: {input_file}")
            return

        print(f"Model: {model}")
        print(f"Input: {input_file}")
        print(f"Output: {output}")
        print()

        # Start llama.cpp server in background
        print(f"Starting llama.cpp server on port {port}...")
        proc = subprocess.Popen(
            [SERVER_BIN, "-m", model, "-c", "4096",
             "-t", "64", "-tb", "32", "-b", "4096", "-ub", "4096",
             "-np", "8", "--mlock", "--cont-batching",
             "--port", str(port), "--host", "127.0.0.1", "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Wait for server to be ready
        print("Waiting for server...", end=" ", flush=True)
        ready = False
        for _ in range(30):
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

        # Read prompts
        samples = []
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        print(f"Generating completions for {len(samples)} prompts...")
        for i, s in enumerate(samples):
            prompt = s.get("prompt", "")
            if not prompt:
                continue
            print(f"  [{i+1}/{len(samples)}] {prompt[:60]}...", end=" ", flush=True)

            data = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens or 256,
                "temperature": args.temperature or 0.7,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                s["completion"] = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                print("✓")
            except Exception as e:
                print(f"✗ ({e})")
                s["completion"] = ""

        # Save
        with open(output, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        proc.kill()
        print(f"\n✅ Saved {len(samples)} labeled samples to {output}")

    else:
        print("Dataset commands:")
        print("  create     — Create a new dataset interactively")
        print("  view       — View a dataset")
        print("  convert    — Convert between formats")
        print("  auto-label — Auto-generate completions using a model")


# ─── Chat ──────────────────────────────────────────────────────────────

def cmd_chat(args):
    """Interactive chat with a model."""
    print_banner()
    print("[Chat] Interactive model chat")
    print()
    
    model = args.model or input("Model path (GGUF): ").strip()
    port = args.port or 8080
    
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
    
    os.environ["PORT"] = str(args.port or 8080)
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
        print(f"\n❌ Benchmark failed: {e}")

    proc.kill()
    print(f"\n✅ Benchmark complete")


# ─── Auto-tune ─────────────────────────────────────────────────────────

def cmd_autotune(args):
    """Auto-tune server settings for this hardware."""
    print("╔══════════════════════════════════════════════╗")
    print("║      MojoLlama AutoTuner v0.1.0              ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    config = autotune(args.model or 
                      "/tmp/tl-Q4_0.gguf",
                      quick=args.quick)

    if config:
        save_config(config)
        print_config(config)

        # Also show the generated server config
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


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MojoLlama Studio")
    parser.add_argument("--version", action="store_true", help="Show version")
    
    sub = parser.add_subparsers(dest="command", help="Command")
    
    # train
    p_train = sub.add_parser("train", help="Fine-tune a model")
    p_train.add_argument("--model", help="Model path (GGUF)")
    p_train.add_argument("--data", help="Training data (JSONL)")
    p_train.add_argument("--lora-out", default="lora-adapter.gguf")
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--lr", default="1e-4")
    p_train.add_argument("--steps", type=int, default=100)
    p_train.add_argument("--batch", type=int, default=4)
    
    # export
    p_export = sub.add_parser("export", help="Convert HF model to GGUF")
    p_export.add_argument("--hf", help="HF model ID or path")
    p_export.add_argument("--model", help="Alias for --hf")
    p_export.add_argument("--outtype", default="q4_0")
    p_export.add_argument("--outfile", help="Output GGUF path")
    p_export.add_argument("--remote", action="store_true", help="Download from HF")
    p_export.add_argument("--vocab-only", action="store_true")
    p_export.add_argument("--mojo-bin", action="store_true", help="Also generate .bin")
    
    # merge
    p_merge = sub.add_parser("merge", help="Merge LoRA adapter into base model")
    p_merge.add_argument("--base", required=True, help="Base model (GGUF)")
    p_merge.add_argument("--lora", required=True, help="LoRA adapter (GGUF)")
    p_merge.add_argument("--output", default="merged.gguf", help="Output path")
    p_merge.add_argument("--type", default="q4_0", help="Output quantization type")
    
    # dataset
    p_data = sub.add_parser("dataset", help="Manage datasets")
    p_data.add_argument("action", choices=["create", "view", "convert", "auto-label"],
                        help="Dataset action")
    p_data.add_argument("--output", help="Output file")
    p_data.add_argument("--dataset", help="Dataset path")
    p_data.add_argument("--input", help="Input file for conversion or auto-label")
    p_data.add_argument("--format", help="Dataset format")
    p_data.add_argument("--model", help="Model path for auto-label")
    p_data.add_argument("--port", type=int, default=8090, help="Server port for auto-label")
    p_data.add_argument("--max-tokens", type=int, default=256, help="Max tokens for auto-label")
    p_data.add_argument("--temperature", type=float, default=0.7, help="Temperature for auto-label")
    
    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--model", help="Model path (GGUF)")
    p_chat.add_argument("--port", type=int, default=8080)
    
    # serve
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--model", help="Model path override")
    
    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark model inference speed")
    p_bench.add_argument("--model", help="Model path (GGUF)")
    p_bench.add_argument("--port", type=int, default=8092, help="Server port")
    p_bench.add_argument("--prompt", default="The meaning of life is", help="Test prompt")
    p_bench.add_argument("--n-predict", type=int, default=128, help="Tokens to generate")
    
    # autotune
    p_tune = sub.add_parser("autotune", help="Auto-tune server settings for this hardware")
    p_tune.add_argument("--model", "-m", help="Model to benchmark with (default: TinyLlama Q4_0)")
    p_tune.add_argument("--quick", "-q", action="store_true", help="Faster sweep (fewer combos)")
    
    # info
    sub.add_parser("info", help="System info")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"MojoLlama Studio v{STUDIO_VERSION}")
        return
    
    if not args.command:
        parser.print_help()
        print("\nCommands:")
        print("  train     Fine-tune a model with LoRA")
        print("  export    Convert HuggingFace model to GGUF")
        print("  dataset   Create, view, auto-label, and manage training datasets")
        print("  merge     Merge LoRA adapter into base GGUF model")
        print("  chat      Interactive chat with a model")
        print("  serve     Start the full MojoLlama API server")
        print("  benchmark Benchmark model inference speed")
        print("  autotune  Auto-tune server settings for this hardware")
        print("  info      Show system info")
        return
    
    commands = {
        "train": cmd_train,
        "export": cmd_export,
        "dataset": cmd_dataset,
        "merge": cmd_merge,
        "chat": cmd_chat,
        "serve": cmd_serve,
        "benchmark": cmd_benchmark,
        "autotune": cmd_autotune,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
