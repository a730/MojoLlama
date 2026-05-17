#!/usr/bin/env python3
"""MojoLlama AutoTuner — finds optimal llama.cpp settings for the current hardware.

Sweeps thread counts, batch sizes, and parallel slots using llama-bench.
Saves best config to ~/.mojollama/config.json for AutoBackend to consume.

Usage:
  python3 autotune.py --model /path/to/model.gguf
  python3 autotune.py --model /tmp/tl-Q4_0.gguf --quick   (faster sweep)
"""

import os
import sys
import json
import time
import re
import subprocess
import argparse
import multiprocessing as mp
from pathlib import Path

CONFIG_DIR = Path.home() / ".mojollama"
CONFIG_PATH = CONFIG_DIR / "config.json"
BENCH_BIN = "/tmp/llama.cpp/build/bin/llama-bench"


def detect_cpu():
    """Detect CPU info for display."""
    info = {"cores": mp.cpu_count(), "name": "unknown"}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["name"] = line.split(":")[1].strip()
                    break
    except: pass
    return info


def run_bench(model, threads, threads_batch, batch_size, ubatch_size,
              n_parallel, mlock, n_prompt=512, n_gen=128):
    """Run a single llama-bench test and return results."""
    cmd = [
        BENCH_BIN, "-m", str(model),
        "-p", str(n_prompt), "-n", str(n_gen),
        "-t", str(threads), "-b", str(batch_size), "-ub", str(ubatch_size),
        "-r", "1",  # single rep for speed during sweep
    ]
    if threads_batch:
        # llama-bench doesn't have -tb, but we note it for server tuning
        pass

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        print(f"  ❌ llama-bench not found at {BENCH_BIN}")
        return None

    # Parse results from markdown table — find the last column (t/s)
    pp_speed = None
    tg_speed = None
    for line in output.split("\n"):
        if "|" in line and ("pp" in line or "tg" in line):
            parts = [p.strip() for p in line.split("|")]
            # Last column contains "797.50 ± 0.00" or "92.37 ± 0.00"
            # Table has trailing |, so the value is second-to-last
            last_col = parts[-2].strip() if len(parts) > 2 else ""
            m = re.match(r"^([\d.]+)", last_col)
            if m:
                val = float(m.group(1))
                if "pp" in line:
                    pp_speed = val
                elif "tg" in line:
                    tg_speed = val

    if pp_speed is None and tg_speed is None:
        return None

    return {
        "threads": threads,
        "batch": batch_size,
        "ubatch": ubatch_size,
        "n_parallel": n_parallel,
        "mlock": mlock,
        "pp_tok_s": pp_speed or 0,
        "tg_tok_s": tg_speed or 0,
    }


def autotune(model, quick=False):
    """Sweep configurations and find optimal settings."""
    cpu = detect_cpu()
    n_cores = cpu["cores"]
    print(f"CPU: {cpu['name']} ({n_cores} cores)")
    print(f"Model: {model}")
    print()

    if not os.path.exists(model):
        print(f"❌ Model not found: {model}")
        return None

    if not os.path.exists(BENCH_BIN):
        print(f"❌ llama-bench not found at {BENCH_BIN}")
        return None

    # ─── Phase 1: Thread count sweep ───
    print("Phase 1: Sweeping thread counts...")
    thread_configs = [16, 32, n_cores, n_cores * 2] if not quick else [32, n_cores]
    # Filter to reasonable values
    thread_configs = sorted(set(t for t in thread_configs if t >= 1 and t <= n_cores * 2))
    
    results = []
    for t in thread_configs:
        r = run_bench(model, threads=t, threads_batch=t//2,
                       batch_size=2048, ubatch_size=512,
                       n_parallel=4, mlock=True)
        if r:
            results.append(r)
            print(f"  {t:3d} threads → pp={r['pp_tok_s']:.0f}  tg={r['tg_tok_s']:.0f} tok/s")
        else:
            print(f"  {t:3d} threads → FAILED")

    if not results:
        print("❌ All thread tests failed")
        return None

    # Find best thread count for generation (primary) and prompt (secondary)
    best_tg = max(results, key=lambda r: r["tg_tok_s"])
    best_pp = max(results, key=lambda r: r["pp_tok_s"])
    best_all = max(results, key=lambda r: r["tg_tok_s"] + r["pp_tok_s"])

    optimal_threads = best_all["threads"]

    print(f"\n  Best gen thread count: {best_tg['threads']} ({best_tg['tg_tok_s']:.0f} tok/s)")
    print(f"  Best prompt thread count: {best_pp['threads']} ({best_pp['pp_tok_s']:.0f} tok/s)")
    print(f"  Best overall: {best_all['threads']} (combined {best_all['tg_tok_s']+best_all['pp_tok_s']:.0f} tok/s)")

    # ─── Phase 2: Batch size sweep ───
    print(f"\nPhase 2: Sweeping batch sizes (threads={optimal_threads})...")
    batch_configs = [512, 1024, 2048, 4096] if not quick else [1024, 4096]

    results_batch = []
    for b in batch_configs:
        r = run_bench(model, threads=optimal_threads, threads_batch=optimal_threads//2,
                       batch_size=b, ubatch_size=b//4,
                       n_parallel=4, mlock=True)
        if r:
            results_batch.append(r)
            print(f"  batch={b:5d} → pp={r['pp_tok_s']:.0f}  tg={r['tg_tok_s']:.0f} tok/s")

    best_batch = max(results_batch, key=lambda r: r["tg_tok_s"] + r["pp_tok_s"]) if results_batch else None

    # ─── Phase 3: Parallel slots ───
    print(f"\nPhase 3: Sweeping parallel slots (threads={optimal_threads})...")
    parallel_configs = [1, 4, 8, 16] if not quick else [4, 8]
    # Filter to reasonable values
    parallel_configs = [p for p in parallel_configs if p <= n_cores // 4 or p <= 8]

    results_par = []
    for np_val in parallel_configs:
        r = run_bench(model, threads=optimal_threads, threads_batch=optimal_threads//2,
                       batch_size=best_batch["batch"] if best_batch else 2048,
                       ubatch_size=best_batch["ubatch"] if best_batch else 512,
                       n_parallel=np_val, mlock=True)
        if r:
            results_par.append(r)
            print(f"  np={np_val:2d} → pp={r['pp_tok_s']:.0f}  tg={r['tg_tok_s']:.0f} tok/s")

    best_par = max(results_par, key=lambda r: r["tg_tok_s"] + r["pp_tok_s"]) if results_par else None

    # ─── Phase 4: mlock comparison ───
    print(f"\nPhase 4: mlock comparison...")
    r_lock = run_bench(model, threads=optimal_threads, threads_batch=optimal_threads//2,
                        batch_size=best_batch["batch"] if best_batch else 2048,
                        ubatch_size=best_batch["ubatch"] if best_batch else 512,
                        n_parallel=best_par["n_parallel"] if best_par else 4,
                        mlock=True)
    r_nolock = run_bench(model, threads=optimal_threads, threads_batch=optimal_threads//2,
                          batch_size=best_batch["batch"] if best_batch else 2048,
                          ubatch_size=best_batch["ubatch"] if best_batch else 512,
                          n_parallel=best_par["n_parallel"] if best_par else 4,
                          mlock=False)
    mlock_better = True
    if r_lock and r_nolock:
        mlock_better = r_lock["tg_tok_s"] >= r_nolock["tg_tok_s"]
        print(f"  mlock:  {r_lock['tg_tok_s']:.0f} tok/s")
        print(f"  nomlock: {r_nolock['tg_tok_s']:.0f} tok/s")
        print(f"  → {'mlock ON' if mlock_better else 'mlock OFF'} wins")

    # ─── Build final config ───
    config = {
        "autotune_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu,
        "model_tuned_on": str(model),
        "llama_server": {
            "threads": optimal_threads,
            "threads_batch": optimal_threads // 2,
            "batch_size": best_batch["batch"] if best_batch else 2048,
            "ubatch_size": best_batch["ubatch"] if best_batch else 512,
            "n_parallel": best_par["n_parallel"] if best_par else 4,
            "mlock": mlock_better,
            "cont_batching": True,
        },
        "results": {
            "thread_sweep": [(r["threads"], r["tg_tok_s"]) for r in results],
            "batch_sweep": [(r["batch"], r["tg_tok_s"]) for r in results_batch],
            "parallel_sweep": [(r["n_parallel"], r["tg_tok_s"]) for r in results_par],
            "best_gen_tok_s": best_tg["tg_tok_s"],
            "best_prompt_tok_s": best_pp["pp_tok_s"],
        },
    }

    return config


def save_config(config):
    """Save config to ~/.mojollama/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Config saved to {CONFIG_PATH}")


def load_config():
    """Load saved config, or return defaults."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return None


def print_config(config):
    """Pretty-print the config."""
    s = config.get("llama_server", {})
    print(f"\n{'='*50}")
    print(f"  MojoLlama Server Configuration")
    print(f"  Tuned: {config.get('autotune_date', 'unknown')}")
    print(f"  CPU: {config.get('cpu', {}).get('name', 'unknown')}")
    print(f"{'='*50}")
    print(f"  --threads         {s.get('threads', '?')}")
    print(f"  --threads-batch   {s.get('threads_batch', '?')}")
    print(f"  --batch-size      {s.get('batch_size', '?')}")
    print(f"  --ubatch-size     {s.get('ubatch_size', '?')}")
    print(f"  --parallel-slots  {s.get('n_parallel', '?')}")
    print(f"  --mlock           {'yes' if s.get('mlock', True) else 'no'}")
    print(f"  --cont-batching   {'yes' if s.get('cont_batching', True) else 'no'}")
    print()
    print(f"  Performance (on tune model):")
    r = config.get("results", {})
    print(f"    Generation: {r.get('best_gen_tok_s', 0):.0f} tok/s")
    print(f"    Prompt:     {r.get('best_prompt_tok_s', 0):.0f} tok/s")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="MojoLlama AutoTuner — find optimal server settings"
    )
    parser.add_argument("--model", "-m", 
                        default="/tmp/tl-Q4_0.gguf",
                        help="GGUF model to benchmark with (default: TinyLlama Q4_0)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode — fewer config combos")
    parser.add_argument("--show", action="store_true",
                        help="Show saved config and exit")
    parser.add_argument("--serve-cmd", action="store_true",
                        help="Print the optimized llama-server command and exit")
    args = parser.parse_args()

    if args.show:
        config = load_config()
        if config:
            print_config(config)
        else:
            print("No saved config. Run autotune first.")
        return

    if args.serve_cmd:
        config = load_config()
        if not config:
            print("# No saved config. Run autotune first.")
            print("# Using defaults...")
            print("llama-server -m model.gguf -c 4096 -t 32 -b 2048 -ub 512 -np 4 --mlock --cont-batching")
            return
        
        s = config.get("llama_server", {})
        mlock_flag = " --mlock" if s.get('mlock', True) else ""
        cmd = (
            f"llama-server -m model.gguf -c 4096"
            f" -t {s.get('threads', 32)}"
            f" -b {s.get('batch_size', 2048)}"
            f" -ub {s.get('ubatch_size', 512)}"
            f" -np {s.get('n_parallel', 4)}"
            f"{mlock_flag}"
            f" --cont-batching"
        )
        print(cmd)
        return

    print("╔══════════════════════════════════════════════╗")
    print("║      MojoLlama AutoTuner v0.1.0              ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    config = autotune(args.model, quick=args.quick)

    if config:
        save_config(config)
        print_config(config)
    else:
        print("❌ Auto-tuning failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
