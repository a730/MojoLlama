#!/usr/bin/env python3
"""MojoLlama unified CLI — as simple as llama.cpp.

Usage:
  mojollama <command> [options]

Commands (match llama.cpp naming):
  chat        Interactive chat with a model
  serve       Start API server (OpenAI-compatible, SSE streaming)
  bench       Benchmark inference performance
  quantize    Quantize a GGUF model to a different type
  imatrix     Generate importance matrix for guided quantization
  convert     Convert HuggingFace model to GGUF
  info        Show system and model info
  autotune    Auto-tune server settings for this hardware
  eval        Run evaluations (MMLU, GSM8K, etc.)
  train       Fine-tune a model (LoRA, QLoRA, DPO, GRPO, etc.)
  export      Export GGUF to other formats (safetensors, ONNX)
  hub         HuggingFace Hub operations (login, push, whoami)

Examples:
  mojollama chat -m ~/models/qwen.gguf
  mojollama serve -m ~/models/qwen.gguf -p 8080
  mojollama bench -m ~/models/qwen.gguf -n 128
  mojollama quantize model.gguf -t Q4_K_M
  mojollama convert --hf meta-llama/Llama-3.2-1B --outtype q4_0
  mojollama info
"""

import os
import sys
import argparse

# Ensure we can import from the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


VERSION = "0.5.0"
BANNER = f"""╔═══════════════════════════════════════════╗
║     MojoLlama v{VERSION}                    ║
║     Unified CLI — as simple as llama.cpp    ║
╚═══════════════════════════════════════════╝"""


def print_commands():
    """Print available commands grouped by category."""
    print(BANNER)
    print()
    print("USAGE: mojollama <command> [options]")
    print()
    print("INFERENCE:")
    print("  chat        Interactive chat with a model")
    print("  serve       Start API server (OpenAI-compatible)")
    print()
    print("BENCHMARK & TUNE:")
    print("  bench       Benchmark inference performance (tok/s, latency)")
    print("  autotune    Auto-tune server settings for this hardware")
    print("  eval        Run evaluations (MMLU, GSM8K, CEval, etc.)")
    print()
    print("MODEL OPS:")
    print("  quantize    Quantize a GGUF model (supports K/IQ quants)")
    print("  imatrix     Generate importance matrix for guided quantization")
    print("  convert     Convert HuggingFace model to GGUF")
    print("  export      Export GGUF to safetensors, ONNX")
    print()
    print("HUB:")
    print("  hub         HuggingFace Hub: login, whoami, push")
    print()
    print("SYSTEM:")
    print("  info        Show system info (CPU, RAM, GPU, models)")
    print()
    print("DEVELOPMENT:")
    print("  train       Fine-tune a model (LoRA, QLoRA, DPO, GRPO, etc.)")
    print()
    print(f"Run 'mojollama <command> --help' for detailed options.")
    print()


def cmd_chat(args):
    """Interactive chat with a model."""
    from mojollama.studio import cmd_chat as _chat
    _chat(args)


def cmd_serve(args):
    """Start the API server."""
    if args.backend == "studio":
        from mojollama.studio import cmd_serve as _serve
        _serve(args)
    else:
        from mojollama.server import main as _server_main
        sys.argv = ["server.py",
                     "--model", args.model or "",
                     "--port", str(args.port)]
        _server_main()


def cmd_bench(args):
    """Benchmark inference performance."""
    if not args.model:
        print("ERROR: --model is required for benchmarking")
        print("Usage: mojollama bench -m model.gguf [-n 128] [-t 32]")
        sys.exit(1)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    from mojollama.bench_tok import bench_mojollama_qwen3
    bench_mojollama_qwen3(args.model)


def cmd_quantize(args):
    """Quantize a GGUF model."""
    from mojollama.studio import cmd_quantize as _quant
    _quant(args)


def cmd_imatrix(args):
    """Generate importance matrix."""
    from mojollama.studio import cmd_imatrix as _imatrix
    _imatrix(args)


def cmd_convert(args):
    """Convert HuggingFace model to GGUF."""
    from mojollama.studio import cmd_export as _export
    _export(args)


def cmd_info(args):
    """Show system and model info."""
    from mojollama.studio import cmd_info as _info
    _info(args)


def cmd_autotune(args):
    """Auto-tune server settings."""
    from mojollama.studio import cmd_autotune as _tune
    _tune(args)


def cmd_hub(args):
    """HuggingFace Hub operations."""
    if not args.hub_action:
        print("Usage: mojollama hub <login|whoami|push> [options]")
        print("  mojollama hub login --token <token>")
        print("  mojollama hub whoami")
        print("  mojollama hub push -m model.gguf --repo user/repo")
        sys.exit(1)
    if args.hub_action == "login":
        from mojollama.studio import cmd_hub_login as _login
        _login(args)
    elif args.hub_action == "whoami":
        from mojollama.studio import cmd_hub_whoami as _whoami
        _whoami(args)
    elif args.hub_action == "push":
        from mojollama.studio import cmd_hub_push as _push
        _push(args)


def cmd_eval(args):
    """Run evaluations."""
    from mojollama.studio import cmd_evaluate as _eval
    _eval(args)


def cmd_export(args):
    """Export to other formats."""
    if not args.export_format:
        print("Usage: mojollama export <safetensors|onnx> [options]")
        print("  mojollama export safetensors -m model.gguf -o ./export/")
        print("  mojollama export onnx -m model.gguf -o model.onnx")
        sys.exit(1)
    if args.export_format == "safetensors":
        from mojollama.studio import cmd_export_safetensors as _st
        _st(args)
    elif args.export_format == "onnx":
        from mojollama.studio import cmd_export_onnx as _onnx
        _onnx(args)


def cmd_train(args):
    """Fine-tune a model."""
    from mojollama.studio import cmd_train as _train
    _train(args)


def build_parser():
    """Build the unified argument parser."""
    parser = argparse.ArgumentParser(
        description="MojoLlama — unified CLI for LLM inference, serving, and model ops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mojollama chat -m model.gguf
  mojollama serve -m model.gguf -p 8080
  mojollama bench -m model.gguf -n 128
  mojollama quantize model.gguf -t Q4_K_M
  mojollama convert --hf meta-llama/Llama-3.2-1B --outtype q4_0
  mojollama info
        """)
    parser.add_argument("--version", "-V", action="store_true",
                        help="Show version and exit")

    sub = parser.add_subparsers(dest="command", help="Command")

    # ── chat ──────────────────────────────────────────────────────
    p_chat = sub.add_parser("chat", help="Interactive chat with a model",
                            description="Chat with an LLM interactively")
    p_chat.add_argument("-m", "--model", help="Model path (GGUF)")
    p_chat.add_argument("-p", "--port", type=int, default=9000,
                        help="Backend server port (default: 9000)")
    p_chat.set_defaults(func=cmd_chat)

    # ── serve ─────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start API server",
                             description="Start an OpenAI-compatible API server with SSE streaming")
    p_serve.add_argument("-m", "--model", help="Model path (GGUF)")
    p_serve.add_argument("-p", "--port", type=int, default=8080,
                         help="Server port (default: 8080)")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="Bind address (default: 127.0.0.1)")
    p_serve.add_argument("-t", "--threads", type=int, default=0,
                         help="CPU threads (default: auto)")
    p_serve.add_argument("--backend", choices=["native", "studio"], default="native",
                         help="Backend: native (mojollama.server) or studio (default: native)")
    p_serve.set_defaults(func=cmd_serve)

    # ── bench ─────────────────────────────────────────────────────
    p_bench = sub.add_parser("bench", aliases=["benchmark"],
                             help="Benchmark inference performance",
                             description="Benchmark model inference speed (tok/s, ms/tok)")
    p_bench.add_argument("-m", "--model", required=True,
                         help="Model path (GGUF)")
    p_bench.add_argument("-n", "--n-predict", type=int, default=128,
                         help="Tokens to generate (default: 128)")
    p_bench.add_argument("-t", "--threads", type=int, default=32,
                         help="CPU threads (default: 32)")
    p_bench.add_argument("--warmup", type=int, default=10,
                         help="Warmup tokens (default: 10)")
    p_bench.add_argument("-p", "--prompt", default="The meaning of life is",
                         help="Test prompt")
    p_bench.set_defaults(func=cmd_bench)

    # ── quantize ──────────────────────────────────────────────────
    p_quant = sub.add_parser("quantize", aliases=["q"],
                             help="Quantize a GGUF model",
                             description="Quantize a GGUF model to a different quantization type")
    p_quant.add_argument("model", help="Path to input GGUF model")
    p_quant.add_argument("-t", "--type", default="Q4_K_M", dest="type",
                         help="Quantization type (default: Q4_K_M)")
    p_quant.add_argument("-o", "--output", help="Output path (default: auto)")
    p_quant.add_argument("--imatrix", help="Importance matrix file for guided quantization")
    p_quant.add_argument("--threads", type=int, default=0, help="Thread count (0=auto)")
    p_quant.add_argument("--allow-requantize", action="store_true",
                         help="Allow requantizing already quantized tensors")
    p_quant.add_argument("--dry-run", action="store_true",
                         help="Calculate size without quantizing")
    p_quant.add_argument("--pure", action="store_true",
                         help="Disable K-quant mixtures, pure type")
    p_quant.add_argument("--leave-output", action="store_true",
                         help="Leave output.weight unquantized")
    p_quant.set_defaults(func=cmd_quantize)

    # ── imatrix ──────────────────────────────────────────────────
    p_imatrix = sub.add_parser("imatrix",
                               help="Generate importance matrix",
                               description="Generate importance matrix for guided quantization")
    p_imatrix.add_argument("model", help="Path to GGUF model")
    p_imatrix.add_argument("-f", "--data", help="Calibration data file (text)")
    p_imatrix.add_argument("-o", "--output", help="Output imatrix file path")
    p_imatrix.add_argument("-t", "--threads", type=int, default=0,
                           help="Number of threads")
    p_imatrix.add_argument("-c", "--ctx-size", type=int, default=512,
                           help="Context size (default: 512)")
    p_imatrix.set_defaults(func=cmd_imatrix)

    # ── convert ───────────────────────────────────────────────────
    p_conv = sub.add_parser("convert", aliases=["cvt"],
                            help="Convert HF model to GGUF",
                            description="Convert a HuggingFace model to GGUF format")
    p_conv.add_argument("--hf", help="HF model ID or path")
    p_conv.add_argument("-m", "--model", help="Alias for --hf (local path or HF ID)")
    p_conv.add_argument("--outtype", default="q4_0",
                        help="Output quantization type (default: q4_0)")
    p_conv.add_argument("-o", "--outfile", help="Output GGUF path")
    p_conv.add_argument("--remote", action="store_true",
                        help="Download from HuggingFace Hub")
    p_conv.add_argument("--vocab-only", action="store_true")
    p_conv.set_defaults(func=cmd_convert)

    # ── info ──────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show system info",
                            description="Show system information: CPU, RAM, GPU, available models")
    p_info.set_defaults(func=cmd_info)

    # ── autotune ──────────────────────────────────────────────────
    p_tune = sub.add_parser("autotune",
                            help="Auto-tune server settings",
                            description="Auto-tune server settings for this hardware")
    p_tune.add_argument("-m", "--model", help="Model to benchmark with")
    p_tune.add_argument("-q", "--quick", action="store_true",
                        help="Faster sweep (fewer combinations)")
    p_tune.add_argument("-d", "--deep", action="store_true",
                        help="Deep hardware detection (CPU features, GPU, etc.)")
    p_tune.set_defaults(func=cmd_autotune)

    # ── hub ───────────────────────────────────────────────────────
    p_hub = sub.add_parser("hub", help="HuggingFace Hub operations",
                           description="Login, whoami, push models to HuggingFace Hub")
    p_hub.add_argument("hub_action", nargs="?", choices=["login", "whoami", "push"],
                       help="Hub action")
    p_hub.add_argument("--token", help="HF API token (for login)")
    p_hub.add_argument("-m", "--model", help="Model path (for push)")
    p_hub.add_argument("--repo", help="HF repo ID (for push)")
    p_hub.add_argument("--message", help="Commit message (for push)")
    p_hub.add_argument("--private", action="store_true",
                       help="Create private repo (for push)")
    p_hub.add_argument("--quant", help="Quantization type metadata (for push)")
    p_hub.add_argument("--params", help="Parameter count metadata (for push)")
    p_hub.set_defaults(func=cmd_hub)

    # ── eval ──────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", aliases=["evaluate"],
                            help="Run evaluations",
                            description="Run benchmark evaluations (MMLU, GSM8K, CEval, etc.)")
    p_eval.add_argument("action", nargs="?", choices=["download", "list"], default=None,
                        help="'download' to fetch datasets, 'list' to show available")
    p_eval.add_argument("-b", "--benchmarks", nargs="+",
                        help="Benchmarks to run: mmlu gsm8k ceval hellaswag arc bbh humaneval")
    p_eval.add_argument("-a", "--all", action="store_true",
                        help="Run all available benchmarks")
    p_eval.add_argument("-m", "--model", help="Model name (for display)")
    p_eval.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per category (0=all)")
    p_eval.add_argument("--port", type=int, default=8081,
                        help="Backend port")
    p_eval.add_argument("-o", "--output", help="Save results to JSON file")
    p_eval.set_defaults(func=cmd_eval)

    # ── export ────────────────────────────────────────────────────
    p_export = sub.add_parser("export",
                              help="Export to safetensors, ONNX",
                              description="Export GGUF to safetensors or ONNX format")
    p_export.add_argument("export_format", nargs="?",
                          choices=["safetensors", "onnx"],
                          help="Export format")
    p_export.add_argument("-m", "--model", help="GGUF model path")
    p_export.add_argument("-o", "--output", help="Output path or directory")
    p_export.add_argument("--dtype", default="float16",
                          choices=["float16", "float32", "bfloat16"],
                          help="Data type (safetensors, default: float16)")
    p_export.add_argument("--shard-size", default="2GB",
                          help="Shard size (safetensors, default: 2GB)")
    p_export.add_argument("--opset", type=int, default=17,
                          help="ONNX opset (ONNX, default: 17)")
    p_export.add_argument("--max-seq-len", type=int, default=2048,
                          help="Max sequence length (ONNX, default: 2048)")
    p_export.set_defaults(func=cmd_export)

    # ── train ─────────────────────────────────────────────────────
    p_train = sub.add_parser("train",
                             help="Fine-tune a model",
                             description="Fine-tune a model using LoRA, QLoRA, DoRA, GaLore, DPO, GRPO, etc.")
    p_train.add_argument("--model", help="Model path (GGUF)")
    p_train.add_argument("--data", help="Training data (JSONL, JSON, or text)")
    p_train.add_argument("--method", choices=["lora", "qlora", "dora", "galore",
                                               "dpo", "orpo", "kto", "simpo", "grpo"],
                         default="lora", help="Training method (default: lora)")
    p_train.add_argument("--lora-out", default="adapter.gguf")
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight-decay", type=float, default=0.0)
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch", type=int, default=4)
    p_train.add_argument("--max-seq-length", type=int, default=512)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--warmup-steps", type=int, default=0)
    p_train.add_argument("--dataset-format", default="auto",
                         choices=["auto", "alpaca", "sharegpt", "jsonl", "text"])
    p_train.add_argument("--list-methods", action="store_true",
                         help="List available training methods")
    p_train.add_argument("--dpo-beta", type=float, default=0.1)
    p_train.add_argument("--orpo-lambda", type=float, default=0.5)
    p_train.add_argument("--simpo-gamma", type=float, default=0.5)
    p_train.add_argument("--grpo-group-size", type=int, default=8)
    p_train.add_argument("--grpo-clip", type=float, default=0.2)
    p_train.add_argument("--galore-rank", type=int, default=128)
    p_train.set_defaults(func=cmd_train)

    return parser


def main():
    parser = build_parser()

    # If no args, show banner + command list
    if len(sys.argv) == 1:
        print_commands()
        return

    args = parser.parse_args()

    if args.version:
        print(f"MojoLlama v{VERSION}")
        return

    if not hasattr(args, "func"):
        print_commands()
        return

    # Set OMP_NUM_THREADS if common threads arg exists
    if hasattr(args, "threads") and args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if os.environ.get("MOJOLLAMA_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
