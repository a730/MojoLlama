#!/usr/bin/env python3
"""MojoLlama AutoTuner — finds optimal llama.cpp settings for the current hardware.

Sweeps thread counts, batch sizes, parallel slots, flash attention,
and mlock using llama-bench. Saves best config to ~/.mojollama/config.json.

Tuned specifically for AMD Threadripper 3970X (32c/64t, AVX2+FMA):
  - Physical-core-only thread count (SMT hurts gen ~25%)
  - CPU affinity mask to pin to physical cores
  - Flash attention ON by default (+40% pp, +5% tg)
  - Batch=4096 ubatch=1024 baseline

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
    """Detect CPU info for display and CPU mask generation."""
    info = {"cores": mp.cpu_count(), "name": "unknown", "physical_cores": mp.cpu_count()}
    try:
        core_ids = set()
        n = mp.cpu_count() or 64
        for i in range(n):
            try:
                with open(f"/sys/devices/system/cpu/cpu{i}/topology/core_id") as f:
                    core_ids.add(int(f.read().strip()))
            except FileNotFoundError:
                break
        if core_ids:
            info["physical_cores"] = len(core_ids)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["name"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    return info


def compute_cpu_mask(physical_cores):
    """Compute CPU affinity hex mask for physical cores only.
    
    On Threadripper 3970X: 32 physical cores → mask 0x00000000FFFFFFFF
    This avoids SMT siblings which hurt generation throughput by ~25%.
    """
    if physical_cores <= 0:
        return ""
    mask = (1 << physical_cores) - 1
    return f"0x{mask:016X}"


def run_bench(model, threads, threads_batch, batch_size, ubatch_size,
              n_parallel, mlock, flash_attn=False, n_prompt=512, n_gen=128,
              cpu_mask=""):
    """Run a single llama-bench test and return results."""
    cmd = [
        BENCH_BIN, "-m", str(model),
        "-p", str(n_prompt), "-n", str(n_gen),
        "-t", str(threads), "-b", str(batch_size), "-ub", str(ubatch_size),
        "-r", "1",  # single rep for speed during sweep
    ]
    if mlock:
        cmd.append("--mlock")
    if flash_attn:
        cmd.extend(["-fa", "1"])
    if cpu_mask and cpu_mask != "0x0":
        cmd.extend(["-C", cpu_mask])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        print(f"  ❌ llama-bench not found at {BENCH_BIN}")
        return None

    # Parse results from markdown table
    pp_speed = None
    tg_speed = None
    for line in output.split("\n"):
        if "|" in line and ("pp" in line or "tg" in line):
            parts = [p.strip() for p in line.split("|")]
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
        "flash_attn": flash_attn,
        "pp_tok_s": pp_speed or 0,
        "tg_tok_s": tg_speed or 0,
    }


def autotune(model, quick=False):
    """Sweep configurations and find optimal settings."""
    cpu = detect_cpu()
    n_cores = cpu["cores"]
    physical = cpu["physical_cores"]
    cpu_mask = compute_cpu_mask(physical)
    print(f"CPU: {cpu['name']} ({physical} physical, {n_cores} logical)")
    print(f"CPU mask: {cpu_mask}")
    print(f"Model: {model}")
    print()

    if not os.path.exists(model):
        print(f"❌ Model not found: {model}")
        return None

    if not os.path.exists(BENCH_BIN):
        print(f"❌ llama-bench not found at {BENCH_BIN}")
        return None

    # ─── Phase 1: Thread count sweep ───
    print("Phase 1: Sweeping thread counts (physical cores only)...")
    # On Threadripper: physical cores = sweet spot, SMT hurts generation
    thread_configs = [physical//2, physical, physical+4, n_cores] if not quick else [physical]
    thread_configs = sorted(set(t for t in thread_configs if 1 <= t <= n_cores))
    
    results = []
    for t in thread_configs:
        r = run_bench(model, threads=t, threads_batch=min(t, physical),
                       batch_size=4096, ubatch_size=1024,
                       n_parallel=4, mlock=True, flash_attn=True,
                       cpu_mask=cpu_mask)
        if r:
            results.append(r)
            print(f"  {t:3d} threads → pp={r['pp_tok_s']:.0f}  tg={r['tg_tok_s']:.0f} tok/s")
        else:
            print(f"  {t:3d} threads → FAILED")

    if not results:
        print("❌ All thread tests failed")
        return None

    best_all = max(results, key=lambda r: r["tg_tok_s"] + r["pp_tok_s"])
    optimal_threads = best_all["threads"]
    print(f"\n  Best overall threads: {optimal_threads} (combined {best_all['tg_tok_s']+best_all['pp_tok_s']:.0f} tok/s)")

    # ─── Phase 2: Batch size sweep ───
    print(f"\nPhase 2: Sweeping batch sizes (threads={optimal_threads})...")
    batch_configs = [1024, 2048, 4096, 8192] if not quick else [2048, 4096]

    results_batch = []
    for b in batch_configs:
        ub = max(256, b // 4)
        r = run_bench(model, threads=optimal_threads, threads_batch=min(optimal_threads, physical),
                       batch_size=b, ubatch_size=ub,
                       n_parallel=4, mlock=True, flash_attn=True,
                       cpu_mask=cpu_mask)
        if r:
            results_batch.append(r)
            print(f"  batch={b:5d} ub={ub:5d} → pp={r['pp_tok_s']:.0f}  tg={r['tg_tok_s']:.0f} tok/s")

    best_batch = max(results_batch, key=lambda r: r["tg_tok_s"] + r["pp_tok_s"]) if results_batch else None

    # ─── Phase 3: Flash attention comparison ───
    print(f"\nPhase 3: Flash attention comparison...")
    base_args = dict(
        model=model, threads=optimal_threads, threads_batch=min(optimal_threads, physical),
        batch_size=best_batch["batch"] if best_batch else 4096,
        ubatch_size=best_batch["ubatch"] if best_batch else 1024,
        n_parallel=4, mlock=True, cpu_mask=cpu_mask,
    )
    r_fa_on = run_bench(flash_attn=True, **base_args)
    r_fa_off = run_bench(flash_attn=False, **base_args)
    fa_better = True
    if r_fa_on and r_fa_off:
        fa_better = r_fa_on["tg_tok_s"] + r_fa_on["pp_tok_s"] >= r_fa_off["tg_tok_s"] + r_fa_off["pp_tok_s"]
        print(f"  flash-attn ON:  pp={r_fa_on['pp_tok_s']:.0f}  tg={r_fa_on['tg_tok_s']:.0f}")
        print(f"  flash-attn OFF: pp={r_fa_off['pp_tok_s']:.0f}  tg={r_fa_off['tg_tok_s']:.0f}")
        delta_pp = ((r_fa_on['pp_tok_s'] / r_fa_off['pp_tok_s']) - 1) * 100 if r_fa_off['pp_tok_s'] > 0 else 0
        delta_tg = ((r_fa_on['tg_tok_s'] / r_fa_off['tg_tok_s']) - 1) * 100 if r_fa_off['tg_tok_s'] > 0 else 0
        print(f"  → flash-attn {'ON' if fa_better else 'OFF'} wins (pp {delta_pp:+.0f}%, tg {delta_tg:+.0f}%)")
    elif r_fa_on:
        print(f"  flash-attn ON: pp={r_fa_on['pp_tok_s']:.0f}  tg={r_fa_on['tg_tok_s']:.0f}  (OFF test failed)")

    # ─── Phase 4: Parallel slots ───
    print(f"\nPhase 4: Sweeping parallel slots (threads={optimal_threads})...")
    parallel_configs = [1, 2, 4, 8] if not quick else [4]
    parallel_configs = [p for p in parallel_configs if p <= max(4, physical // 4)]

    results_par = []
    for np_val in parallel_configs:
        r = run_bench(model, threads=optimal_threads, threads_batch=min(optimal_threads, physical),
                       batch_size=best_batch["batch"] if best_batch else 4096,
                       ubatch_size=best_batch["ubatch"] if best_batch else 1024,
                       n_parallel=np_val, mlock=True, flash_attn=fa_better,
                       cpu_mask=cpu_mask)
        if r:
            results_par.append(r)
            print(f"  np={np_val:2d} → pp={r['pp_tok_s']:.0f}  tg={r['tg_tok_s']:.0f} tok/s")

    best_par = max(results_par, key=lambda r: r["tg_tok_s"] + r["pp_tok_s"]) if results_par else None

    # ─── Phase 5: mlock comparison ───
    print(f"\nPhase 5: mlock comparison...")
    r_lock = run_bench(model, threads=optimal_threads, threads_batch=min(optimal_threads, physical),
                        batch_size=best_batch["batch"] if best_batch else 4096,
                        ubatch_size=best_batch["ubatch"] if best_batch else 1024,
                        n_parallel=best_par["n_parallel"] if best_par else 4,
                        mlock=True, flash_attn=fa_better, cpu_mask=cpu_mask)
    r_nolock = run_bench(model, threads=optimal_threads, threads_batch=min(optimal_threads, physical),
                          batch_size=best_batch["batch"] if best_batch else 4096,
                          ubatch_size=best_batch["ubatch"] if best_batch else 1024,
                          n_parallel=best_par["n_parallel"] if best_par else 4,
                          mlock=False, flash_attn=fa_better, cpu_mask=cpu_mask)
    mlock_better = True
    if r_lock and r_nolock:
        mlock_better = r_lock["tg_tok_s"] >= r_nolock["tg_tok_s"]
        print(f"  mlock:    {r_lock['tg_tok_s']:.0f} tok/s (pp={r_lock['pp_tok_s']:.0f})")
        print(f"  no-mlock: {r_nolock['tg_tok_s']:.0f} tok/s (pp={r_nolock['pp_tok_s']:.0f})")
        print(f"  → {'mlock ON' if mlock_better else 'mlock OFF'} wins")

    # ─── Build final config ───
    optimal_np = best_par["n_parallel"] if best_par else 4
    config = {
        "autotune_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu,
        "model_tuned_on": str(model),
        "llama_server": {
            "threads": optimal_threads,
            "threads_batch": min(optimal_threads, physical),
            "batch_size": best_batch["batch"] if best_batch else 4096,
            "ubatch_size": best_batch["ubatch"] if best_batch else 1024,
            "n_parallel": optimal_np,
            "mlock": mlock_better,
            "cont_batching": True,
            "flash_attn": fa_better,
            "cpu_mask": cpu_mask,
        },
        "proxy_server": {
            "max_workers": optimal_np * 8,  # 8× parallel slots for headroom
            "queue_size": optimal_np * 32,  # 32× for burst absorption
        },
        "results": {
            "thread_sweep": [(r["threads"], r["tg_tok_s"]) for r in results],
            "batch_sweep": [(r["batch"], r["tg_tok_s"]) for r in results_batch],
            "parallel_sweep": [(r["n_parallel"], r["tg_tok_s"]) for r in results_par],
            "flash_attn_on": r_fa_on,
            "flash_attn_off": r_fa_off,
            "best_gen_tok_s": max(r["tg_tok_s"] for r in results) if results else 0,
            "best_prompt_tok_s": max(r["pp_tok_s"] for r in results) if results else 0,
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
    p = config.get("proxy_server", {})
    print(f"\n{'='*50}")
    print(f"  MojoLlama Server Configuration")
    print(f"  Tuned: {config.get('autotune_date', 'unknown')}")
    cpu = config.get("cpu", {})
    print(f"  CPU: {cpu.get('name', 'unknown')} ({cpu.get('physical_cores', '?')} physical / {cpu.get('cores', '?')} logical)")
    print(f"{'='*50}")
    print(f"  llama.cpp backend:")
    print(f"    --threads         {s.get('threads', '?')}")
    print(f"    --threads-batch   {s.get('threads_batch', '?')}")
    print(f"    --batch-size      {s.get('batch_size', '?')}")
    print(f"    --ubatch-size     {s.get('ubatch_size', '?')}")
    print(f"    --parallel-slots  {s.get('n_parallel', '?')}")
    print(f"    --mlock           {'yes' if s.get('mlock', True) else 'no'}")
    print(f"    --cont-batching   {'yes' if s.get('cont_batching', True) else 'no'}")
    print(f"    --flash-attn      {'yes' if s.get('flash_attn', True) else 'no'}")
    print(f"    -C (cpu mask)     {s.get('cpu_mask', 'auto')}")
    print(f"  Proxy server:")
    print(f"    --max-workers     {p.get('max_workers', '?')}")
    print(f"    --queue-size      {p.get('queue_size', '?')}")
    print()
    r = config.get("results", {})
    print(f"  Performance (on tune model):")
    print(f"    Generation:     {r.get('best_gen_tok_s', 0):.0f} tok/s")
    print(f"    Prompt:         {r.get('best_prompt_tok_s', 0):.0f} tok/s")
    print(f"    Max throughput: ~{s.get('n_parallel', 4) * 12:.0f} req/s (est.)")
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
            print("llama-server -m model.gguf -c 4096 -t 32 -b 4096 -ub 1024 -np 4 --mlock --cont-batching -fa 1 -C 0x00000000FFFFFFFF")
            return
        
        from mojollama.backends import build_server_cmd
        cmd = build_server_cmd("llama-server", "model.gguf", 8081, config.get("llama_server"))
        print(' '.join(cmd))
        return

    print("╔══════════════════════════════════════════════╗")
    print("║      MojoLlama AutoTuner v0.2.0              ║")
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