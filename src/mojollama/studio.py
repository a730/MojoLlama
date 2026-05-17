#!/usr/bin/env python3
"""MojoLlama Studio — train, export, dataset, chat.

Usage:
  mojollama-studio train --model model.gguf --data dataset.jsonl
  mojollama-studio export --hf meta-llama/Llama-3.2-1B --outtype q4_0
  mojollama-studio dataset create --output data.jsonl
  mojollama-studio chat --model model.gguf
  mojollama-studio serve --model model.gguf --port 8080
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

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
        import numpy as np
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
        path = args.dataset or input("Dataset path: ").strip()
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
    
    else:
        print("Dataset commands:")
        print("  create   — Create a new dataset interactively")
        print("  view     — View a dataset")
        print("  convert  — Convert between formats")


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
    import urllib.request
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
            [SERVER_BIN, "-m", model, "-c", "4096", "-t", "32",
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
    print("  chat      Interactive chat")
    print("  serve     Full API server")
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
    
    # dataset
    p_data = sub.add_parser("dataset", help="Manage datasets")
    p_data.add_argument("action", choices=["create", "view", "convert"],
                        help="Dataset action")
    p_data.add_argument("--output", help="Output file")
    p_data.add_argument("--dataset", help="Dataset path")
    p_data.add_argument("--input", help="Input file for conversion")
    p_data.add_argument("--format", help="Dataset format")
    
    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--model", help="Model path (GGUF)")
    p_chat.add_argument("--port", type=int, default=8080)
    
    # serve
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--model", help="Model path override")
    
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
        print("  dataset   Create, view, and manage training datasets")
        print("  chat      Interactive chat with a model")
        print("  serve     Start the full MojoLlama API server")
        print("  info      Show system info")
        return
    
    commands = {
        "train": cmd_train,
        "export": cmd_export,
        "dataset": cmd_dataset,
        "chat": cmd_chat,
        "serve": cmd_serve,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
