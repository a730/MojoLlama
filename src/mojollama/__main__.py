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
    """Benchmark inference performance. Better than llama-bench — thread sweep,
    multi-model, profile mode, A/B comparison, historical results."""
    if not args.model:
        print("ERROR: --model is required for benchmarking")
        print()
        print("Examples:")
        print("  mojollama bench -m model.gguf                          # single benchmark")
        print("  mojollama bench -m model.gguf -t 1,2,4,8,16,32        # thread sweep")
        print("  mojollama bench -m m1.gguf,m2.gguf                    # multi-model")
        print("  mojollama bench -m model.gguf --compare                # A/B vs llama.cpp")
        print("  mojollama bench -m model.gguf --profile                # component profile")
        print("  mojollama bench -m model.gguf --json                   # JSON output")
        print("  mojollama bench -m model.gguf --save results.json      # save to file")
        print("  mojollama bench --load a.json --load b.json            # historical compare")
        sys.exit(1)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    # Add paths needed for engine imports
    kernels_dir = os.path.join(os.path.dirname(__file__), 'kernels')
    if kernels_dir not in sys.path:
        sys.path.insert(0, kernels_dir)
    from mojollama.benchmarks.bench_tok import main as _bench_main
    # Forward all relevant flags
    sys.argv = ["bench"]
    if args.model: sys.argv.extend(["--model", args.model])
    sys.argv.extend(["--threads", str(args.threads)])
    sys.argv.extend(["--prompt-len", str(args.prompt_len)])
    sys.argv.extend(["--gen-len", str(args.gen_len)])
    sys.argv.extend(["--warmup", str(args.warmup)])
    sys.argv.extend(["--measured", str(args.measured)])
    if args.compare: sys.argv.append("--compare")
    if args.profile: sys.argv.append("--profile")
    if args.output_json: sys.argv.extend(["--output", "json"])
    if args.save: sys.argv.extend(["--save", args.save])
    if args.concurrent: sys.argv.extend(["--concurrent", args.concurrent])
    if args.concurrent_quick: sys.argv.append("--concurrent-quick")
    if args.load_paths:
        for path in args.load_paths:
            sys.argv.extend(["--load", path])
    _bench_main()


def cmd_quantize(args):
    """Quantize a GGUF model. Supports multi-quant cascade, target BPW, benchmark."""
    import time

    # --list-types
    Q_TYPES = [
        ("Q2_K",   2.5,  "2-bit K-quant (smallest, lowest quality)"),
        ("Q3_K_S", 3.0,  "3-bit K-quant small"),
        ("Q3_K_M", 3.4,  "3-bit K-quant medium (balanced)"),
        ("Q3_K_L", 3.6,  "3-bit K-quant large"),
        ("Q4_0",   4.0,  "4-bit (no K-quant, fast but large)"),
        ("Q4_K_S", 4.4,  "4-bit K-quant small"),
        ("Q4_K_M", 4.5,  "4-bit K-quant medium (recommended)"),
        ("Q5_0",   5.0,  "5-bit (no K-quant)"),
        ("Q5_K_S", 5.1,  "5-bit K-quant small"),
        ("Q5_K_M", 5.5,  "5-bit K-quant medium"),
        ("Q6_K",   6.0,  "6-bit K-quant (high quality)"),
        ("Q8_0",   8.0,  "8-bit (lossless-ish)"),
        ("F16",    16.0, "16-bit float (lossless)"),
    ]
    if args.list_types:
        dash = "─" * 10; dash5 = "─" * 5; dash40 = "─" * 40
        print(f"\n{'Type':>10s}  {'BPW':>5s}  Description")
        print(f"{dash}  {dash5}  {dash40}")
        for name, bpw, desc in Q_TYPES:
            print(f"{name:>10s}  {bpw:>5.1f}  {desc}")
        print(f"\n  Use: mojollama quantize model.gguf -t TYPE")
        print(f"  Multi: mojollama quantize model.gguf --multi Q8_0,Q6_K,Q4_K_M")
        print(f"  Target BPW: mojollama quantize model.gguf --target-bpw 3.5")
        return

    # --target-bpw: auto-select best type
    if args.target_bpw:
        best = None
        for name, bpw, desc in Q_TYPES:
            if bpw <= args.target_bpw + 0.1:
                best = name
            else:
                break
        if best:
            best_bpw = next(b for n,b,d in Q_TYPES if n==best)
            print(f"[Quantize] Target BPW={args.target_bpw}: selected {best} ({best_bpw:.1f} bpw)")
            args.type = best
        else:
            print(f"[Quantize] No type fits target BPW={args.target_bpw}, using Q2_K")
            args.type = "Q2_K"

    # --multi cascade
    if args.multi:
        types = [t.strip() for t in args.multi.split(",")]
        print(f"\n[Quantize] Multi-quant cascade: {' → '.join(types)}")
        print(f"[Quantize] Base model: {args.model}")
        print()
        prev_model = args.model
        results = []
        for i, qt in enumerate(types):
            outfile = args.model.replace(".gguf", f"-{qt}.gguf")
            print(f"  [{i+1}/{len(types)}] Quantizing to {qt}...")
            print(f"    Input:  {prev_model}")
            print(f"    Output: {outfile}")
            t0 = time.time()
            _run_llama_quantize(prev_model, outfile, qt, args)
            elapsed = time.time() - t0
            size_mb = os.path.getsize(outfile) / (1024*1024)
            print(f"    Done: {elapsed:.0f}s  Size: {size_mb:.0f} MB")
            results.append((qt, outfile, elapsed, size_mb))

            # Benchmark each output if --bench
            if args.bench and os.path.exists(outfile):
                print(f"    Benchmarking {qt}...")
                try:
                    from mojollama.benchmarks.bench_tok import bench_native
                    pp_ms, pp_tok, tg_ms, tg_tok = bench_native(outfile)
                    print(f"    tg128: {tg_tok:.1f} tok/s")
                    results[-1] = (qt, outfile, elapsed, size_mb, tg_tok)
                except Exception as e:
                    print(f"    Benchmark FAILED: {e}")

            prev_model = outfile
            print()
        print(f"[Quantize] Cascade complete. {' → '.join(types)}")
        return

    # Standard single quantization
    outfile = args.output or args.model.replace(".gguf", f"-{args.type}.gguf")
    print(f"\n[Quantize] {args.model} → {outfile} ({args.type})")
    print()
    t0 = time.time()
    _run_llama_quantize(args.model, outfile, args.type, args)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(outfile) / (1024*1024)
    print(f"\nDone: {elapsed:.0f}s  Size: {size_mb:.0f} MB")

    # Benchmark if --bench
    if args.bench:
        print(f"\n[Quantize] Benchmarking {args.type}...")
        try:
            from mojollama.benchmarks.bench_tok import bench_native
            pp_ms, pp_tok, tg_ms, tg_tok = bench_native(outfile)
            print(f"  pp512: {pp_tok:.1f} tok/s  tg128: {tg_tok:.1f} tok/s")
        except Exception as e:
            print(f"  Benchmark FAILED: {e}")
    print()


def _run_llama_quantize(inpath, outpath, qtype, args):
    """Run llama-quantize with progress monitoring."""
    quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        print(f"  ERROR: llama-quantize not found at {quantize_bin}")
        print(f"  Build it: cd /tmp/llama.cpp && cmake -B build && cmake --build build -j32 --target llama-quantize")
        sys.exit(1)
    cmd = [quantize_bin, "--allow-requantize" if args.allow_requantize else None,
           "--pure" if args.pure else None,
           "--leave-output-tensor" if args.leave_output else None,
           "--dry-run" if args.dry_run else None, inpath, outpath, qtype]
    if args.threads and args.threads > 0:
        cmd.append(str(args.threads))
    if args.imatrix:
        cmd.extend(["--imatrix", args.imatrix])
    cmd = [c for c in cmd if c is not None]
    # Run with progress display
    import subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            # Show progress lines (not debug spam)
            if any(x in line.lower() for x in ['size', 'quantizing', 'done', 'error', 'warning', 'processing', '%']):
                print(f"    {line}", flush=True)
            elif line.startswith('['):
                print(f"    {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"  ERROR: Quantization failed (exit code {proc.returncode})")
        sys.exit(proc.returncode)


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
    """Auto-tune server settings for this hardware."""
    # Default to MojoLlama native tuning if no llama-bench available
    import shutil
    lb = shutil.which("llama-bench") or "/tmp/llama.cpp/build/bin/llama-bench"
    use_mojollama = args.mojollama or not os.path.exists(lb)
    if not args.model:
        print("ERROR: --model is required for autotune")
        print("Usage: mojollama autotune -m model.gguf [--quick] [--llamacpp-only]")
        sys.exit(1)

    from mojollama.autotune import autotune
    config = autotune(args.model, quick=args.quick, mojollama=use_mojollama)
    if config:
        from mojollama.autotune import save_config, print_config
        save_config(config)
        print_config(config)


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
                             help="Benchmark inference performance (better than llama-bench)",
                             description="Comprehensive benchmark: thread sweep, multi-model, "
                                         "A/B comparison, profile mode, historical results.")
    p_bench.add_argument("-m", "--model", type=str,
                         help="Model path(s) — comma-separated for multi-model comparison")
    p_bench.add_argument("-t", "--threads", type=str, default="32",
                         help="Thread count(s) — comma-separated for sweep (default: 32)")
    p_bench.add_argument("--prompt-len", type=int, default=512,
                         help="Prompt length in tokens (default: 512)")
    p_bench.add_argument("--gen-len", type=int, default=128,
                         help="Tokens to generate (default: 128)")
    p_bench.add_argument("--warmup", type=int, default=5,
                         help="Warmup tokens (default: 5)")
    p_bench.add_argument("--measured", type=int, default=30,
                         help="Measured tokens (default: 30)")
    p_bench.add_argument("--compare", action="store_true",
                         help="A/B comparison with llama.cpp")
    p_bench.add_argument("--profile", action="store_true",
                         help="Component-level profiling")
    p_bench.add_argument("--concurrent", type=str,
                         help="Concurrency benchmark: comma-sep levels, e.g. '1,2,4,8,16'")
    p_bench.add_argument("--concurrent-quick", action="store_true",
                         help="Quick concurrency test (levels 1,4,8)")
    p_bench.add_argument("--json", dest="output_json", action="store_true",
                         help="Output results as JSON")
    p_bench.add_argument("--save", type=str,
                         help="Save results to JSON file")
    p_bench.add_argument("--load", type=str, action="append", dest="load_paths",
                         help="Load previous results for comparison (can be used multiple times)")
    p_bench.set_defaults(func=cmd_bench)

    # ── quantize ──────────────────────────────────────────────────
    p_quant = sub.add_parser("quantize", aliases=["q"],
                             help="Quantize a GGUF model",
                             description="Quantize a GGUF model to a different quantization type. "
                                         "Supports multi-quant cascade with --multi and target-BPW selection.")
    p_quant.add_argument("model", nargs="?", help="Path to input GGUF model (optional with --list-types)")
    p_quant.add_argument("-t", "--type", default="Q4_K_M", dest="type",
                         help="Quantization type (default: Q4_K_M). Use --list-types to see options")
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
    p_quant.add_argument("--multi", type=str,
                         help="Multi-quant cascade: comma-separated types, e.g. 'Q8_0,Q6_K,Q4_K_M,Q3_K_M' "
                              "(runs sequentially, each from previous)")
    p_quant.add_argument("--target-bpw", type=float,
                         help="Target bits-per-weight (auto-selects best type)")
    p_quant.add_argument("--bench", action="store_true",
                         help="Benchmark each output after quantization")
    p_quant.add_argument("--list-types", action="store_true",
                         help="List available quantization types and exit")
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
                            description="Auto-tune MojoLlama native engine or llama.cpp settings for this hardware")
    p_tune.add_argument("-m", "--model", required=True, help="Model path (GGUF) to benchmark")
    p_tune.add_argument("-q", "--quick", action="store_true",
                        help="Faster sweep (fewer combinations)")
    p_tune.add_argument("--mojollama", action="store_true",
                        help="Force MojoLlama native engine tuning (auto-detected by default)")
    p_tune.add_argument("--deep", action="store_true",
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
