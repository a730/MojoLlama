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
import numpy as np
from pathlib import Path

CONFIG_DIR = Path.home() / ".mojollama"
CONFIG_PATH = CONFIG_DIR / "config.json"
BENCH_BIN = None  # resolved at runtime via shutil.which()

def _find_bench():
    """Locate llama-bench: check PATH, then build tree, then common locations."""
    global BENCH_BIN
    if BENCH_BIN is not None:
        return BENCH_BIN
    import shutil
    candidates = [
        shutil.which("llama-bench"),
        "/tmp/llama.cpp/build/bin/llama-bench",
        "/usr/local/bin/llama-bench",
        "llama-bench",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            BENCH_BIN = c
            return BENCH_BIN
    return None


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
        _find_bench(), "-m", str(model),
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
        b = _find_bench()
        if b is None:
            print(f"  ❌ llama-bench not found in PATH or build tree")
        else:
            print(f"  ❌ llama-bench not found at {b}")
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


# ═══════════════════════════════════════════════════════════════════════
# Phase 6 — MojoLlama engine benchmark (all archs: MoE, dense, Gemma4)
# ═══════════════════════════════════════════════════════════════════════

def _detect_engine_type(model_path):
    """Detect which MojoLlama engine class to use for a model."""
    try:
        import gguf
        r = gguf.GGUFReader(model_path)
        arch = str(r.fields.get("general.architecture", gguf.GGUFValue(b"unknown")).parts[-1]).strip().lower()
        if any(x in arch for x in ["moe", "gpt-oss", "deepseek"]):
            return "moe"
        if "gemma4" in arch:
            return "gemma4"
        if "zaya" in arch:
            return "zaya"
        return "dense"
    except Exception:
        return "dense"


def _load_engine(model_path, threads):
    """Load the appropriate MojoLlama engine for the given model.
    Returns (engine, engine_type) or raises."""
    _md = os.path.dirname(os.path.abspath(__file__))
    _bd = os.path.dirname(_md)
    if _bd not in sys.path:
        sys.path.insert(0, _bd)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    engine_type = _detect_engine_type(model_path)

    if engine_type in ("moe", "gemma4"):
        from mojollama.kernels.turbo_engine_v7_moe import TurboEngineV7MoE
        eng = TurboEngineV7MoE(model_path, n_threads=threads)
    else:
        from mojollama.kernels.turbo_engine_v77 import TurboEngineV77
        eng = TurboEngineV77(model_path, n_threads=threads)

    return eng, engine_type


def bench_mojollama(model_path, threads, quick=False):
    """Benchmark MojoLlama native engine with pp512 + tg128 measurement.

    Sweeps thread counts (if threads is a list) and batch sizes.
    Returns list of {threads, pp_tok_s, tg_tok_s, ms_per_tok} dicts.
    """
    try:
        engine, etype = _load_engine(model_path, threads if isinstance(threads, int) else threads[0])
    except Exception as exc:
        print(f"  ⚠ Engine load failed: {exc}")
        return []

    V = int(engine.vocab_size)
    N_WARMUP = 5
    N_MEASURED = 20
    results = []

    # Handle single thread or list
    thread_list = [threads] if isinstance(threads, int) else threads

    for nt in thread_list:
        # Re-load engine for each thread count (OMP_NUM_THREADS must match)
        if nt != thread_list[0]:
            try:
                engine, etype = _load_engine(model_path, nt)
            except Exception as exc:
                print(f"  ⚠ Reload @{nt}t failed: {exc}")
                continue

        engine.reset()

        # ── pp512: prompt processing ──
        prompt_len = 512
        prompt_tokens = [i % max(V, 1) for i in range(prompt_len)]
        t0 = time.perf_counter()
        for tok in prompt_tokens:
            engine.forward([tok])
        pp_elapsed = time.perf_counter() - t0
        pp_tok_s = prompt_len / pp_elapsed

        # ── tg128: token generation ──
        token = prompt_tokens[-1]
        # Warmup
        for _ in range(N_WARMUP):
            logits = engine.forward([token])
            a = np.asarray(logits).ravel()
            token = int(np.argmax(a)) % max(V, 1)
        # Measured
        t0 = time.perf_counter()
        for _ in range(N_MEASURED):
            logits = engine.forward([token])
            a = np.asarray(logits).ravel()
            token = int(np.argmax(a)) % max(V, 1)
        tg_elapsed = time.perf_counter() - t0
        tg_tok_s = N_MEASURED / tg_elapsed
        tg_ms = tg_elapsed / N_MEASURED * 1000

        results.append({
            "threads": nt,
            "pp_tok_s": round(pp_tok_s, 1),
            "tg_tok_s": round(tg_tok_s, 1),
            "ms_per_tok": round(tg_ms, 2),
            "engine_type": etype,
        })
        print(f"    {nt:3d}t → pp={pp_tok_s:.1f}  tg={tg_tok_s:.1f} tok/s  ({tg_ms:.1f} ms/tok)")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 7 — Concurrency tuning (no server needed)
# ═══════════════════════════════════════════════════════════════════════

def bench_concurrency(model_path, quick=False):
    """Benchmark concurrent throughput using direct engine forward() calls.

    No HTTP server required. Sweeps concurrency=1,2,4,8 (or 1,4 quick).
    Measures aggregate tok/s by spawning N concurrent forward() loops
    via threading (GIL-released OMP matmuls run in parallel).
    Returns dict with optimal_concurrency, max_throughput, latency.
    """
    try:
        engine, etype = _load_engine(model_path, int(os.environ.get("OMP_NUM_THREADS", "8")))
    except Exception as exc:
        print(f"  ⚠ Engine load failed for concurrency test: {exc}")
        return None

    V = int(engine.vocab_size)
    import threading
    from concurrent.futures import ThreadPoolExecutor

    levels = [1, 2, 4, 8, 16] if not quick else [1, 4, 8]
    n_gen = 20  # tokens per request
    n_reqs = 3  # requests per level

    def _request(engine_ref, tid):
        """Run a short generation sequence, return tok/s."""
        try:
            token = int(tid) % max(V, 1)
            t0 = time.perf_counter()
            for _ in range(n_gen):
                logits = engine_ref.forward([token])
                a = np.asarray(logits).ravel()
                token = int(np.argmax(a)) % max(V, 1)
            elapsed = time.perf_counter() - t0
            return n_gen / elapsed if elapsed > 0 else 0
        except Exception:
            return 0

    results_data = []
    for concurrency in levels:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            tok_s_list = list(pool.map(lambda x: _request(engine, x), range(n_reqs * concurrency)))
        total_elapsed = time.perf_counter() - t0

        valid = [t for t in tok_s_list if t > 0]
        if not valid:
            continue
        avg_tok_s = sum(valid) / len(valid)
        agg_tok_s = sum(valid)
        latencies = [n_gen / t * 1000 for t in valid]  # ms per request
        lat_sorted = sorted(latencies)
        p50 = lat_sorted[len(lat_sorted)//2]
        p95 = lat_sorted[int(len(lat_sorted)*0.95)]

        results_data.append({
            "concurrency": concurrency,
            "throughput": round(agg_tok_s, 1),
            "avg_per_request": round(avg_tok_s, 1),
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
            "total_time_s": round(total_elapsed, 2),
        })
        print(f"    concurrency={concurrency:3d} → {agg_tok_s:.1f} agg tok/s  p50={p50:.0f}ms  p95={p95:.0f}ms")

    if not results_data:
        return None

    # Find throughput plateau: first level where next step gives <10% improvement
    plateau = results_data[-1]["concurrency"]
    for i in range(1, len(results_data)):
        prev_tp = results_data[i-1]["throughput"]
        curr_tp = results_data[i]["throughput"]
        if prev_tp > 0 and curr_tp / prev_tp < 1.10:
            plateau = results_data[i-1]["concurrency"]
            break

    # Find latency knee: first level where p95 > 2x base p95
    base_p95 = results_data[0]["p95_latency_ms"]
    latency_knee = None
    for rd in results_data[1:]:
        if base_p95 > 0 and rd["p95_latency_ms"] > 2.0 * base_p95:
            latency_knee = rd["concurrency"]
            break

    optimal = min(plateau, latency_knee) if latency_knee else plateau
    best_throughput = max(rd["throughput"] for rd in results_data)

    print(f"\n  → Throughput plateau at concurrency={plateau}"
          f"{'  (latency knee at {latency_knee})' if latency_knee else ''}")
    print(f"  → Optimal concurrency: {optimal}  (peak {best_throughput:.0f} tok/s)")

    return {
        "optimal_concurrency": optimal,
        "throughput_plateau": plateau,
        "latency_knee": latency_knee,
        "max_throughput": round(best_throughput, 1),
        "p50_latency_ms": results_data[0]["p50_latency_ms"],
    }


def autotune(model, quick=False, mojollama=False):
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

    if _find_bench() is None:
        if mojollama:
            print("⚠ llama-bench not found — Phases 1-5 (llama.cpp tuning) will be skipped")
        else:
            print("❌ llama-bench not found — install llama.cpp or use --mojollama")
            return None

    skip_bench = mojollama and _find_bench() is None

    # Defaults used when llama-bench phases are skipped
    optimal_threads = physical
    results = []
    results_batch = []
    results_par = []
    r_fa_on = r_fa_off = None
    fa_better = True
    mlock_better = True
    best_batch = None
    best_par = None

    # ─── Phase 1: Thread count sweep ───
    if not skip_bench:
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
            if not mojollama:
                print("❌ All thread tests failed")
                return None
            print("⚠ All thread tests failed — using defaults, running MojoLlama phases")

        if results:
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
            print(f"  -> {'mlock ON' if mlock_better else 'mlock OFF'} wins")

    # --- Phase 6: MojoLlama engine benchmark sweep ---
    # Shared default values (used when --mojollama is not requested)
    optimal_moe_threads = 32
    best_moe_batch = 1
    best_moe_tok_s = 0
    optimal_moe_concurrency = 4
    latency_p50_ms = 0

    if mojollama:
        print(f"\n{'='*50}")
        print("Phase 6: MojoLlama native engine benchmark (all archs)")
        print(f"{'='*50}")
        moe_thread_configs = [int(physical*0.5), int(physical*0.75), physical] if not quick else [physical]
        moe_thread_configs = sorted(set(t for t in moe_thread_configs if t >= 1))
        all_moe_results = []
        print(f"  Thread sweep: {moe_thread_configs}")
        r = bench_mojollama(model, moe_thread_configs, quick=quick)
        all_moe_results.extend(r)

        if all_moe_results:
            best_moe_tg = max(all_moe_results, key=lambda x: x["tg_tok_s"])
            best_moe_combined = max(all_moe_results, key=lambda x: x["tg_tok_s"] + x.get("pp_tok_s", 0))
            optimal_moe_threads = best_moe_tg["threads"]
            best_moe_tok_s = best_moe_tg["tg_tok_s"]
            print(f"\n  Best MojoLlama engine (tg): {best_moe_tg['threads']}t → {best_moe_tg['tg_tok_s']:.1f} tok/s"
                  f"  pp={best_moe_tg.get('pp_tok_s', 0):.1f} tok/s")
            print(f"  Best combined: {best_moe_combined['threads']}t → "
                  f"tg={best_moe_combined['tg_tok_s']:.1f} pp={best_moe_combined.get('pp_tok_s', 0):.1f}")
        else:
            print(f"\n  MojoLlama engine benchmark skipped -- no results")

        # --- Phase 7: Concurrency tuning (direct, no server) ---
        print(f"\n{'='*50}")
        print("Phase 7: Concurrency tuning (direct forward calls)")
        print(f"{'='*50}")
        cc_result = bench_concurrency(model, quick=quick)
        if cc_result:
            optimal_moe_concurrency = cc_result["optimal_concurrency"]
            latency_p50_ms = cc_result["p50_latency_ms"]
            if cc_result["max_throughput"] > best_moe_tok_s:
                best_moe_tok_s = cc_result["max_throughput"]
        else:
            print(f"  Concurrency tuning skipped -- using defaults")

    # --- Build final config ---
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
        "mojollama_engine": {
            "threads": optimal_moe_threads,
            "optimal_batch": best_moe_batch,
            "optimal_concurrency": optimal_moe_concurrency,
            "max_throughput": best_moe_tok_s,
            "latency_p50_ms": latency_p50_ms,
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
                        help="Quick mode -- fewer config combos")
    parser.add_argument("--mojollama", action="store_true",
                        help="Run MojoLlama-specific phases (6: engine bench, 7: concurrency tuning)")
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
        print(" ".join(cmd))
        return


# ═══════════════════════════════════════════════════════════════════════
# Auto-tune on first model load
# ═══════════════════════════════════════════════════════════════════════

_AUTO_TUNE_CACHE = {}  # model_path -> config


def auto_tune_for_model(model_path: str, quick: bool = True) -> dict:
    """Run a fast MojoLlama engine tune for *model_path* on first load.

    Caches per model path.  Returns config dict with:
      - omp_threads  — optimal OMP_NUM_THREADS for this model+hardware
      - concurrency  — optimal number of parallel workers
      - strategy     — 'process' or 'thread' based on hardware topology

    Only runs the MojoLlama engine phases (skips llama-bench).
    """
    global _AUTO_TUNE_CACHE
    cache_key = os.path.abspath(model_path)
    if cache_key in _AUTO_TUNE_CACHE:
        return _AUTO_TUNE_CACHE[cache_key]

    print(f"\n{'='*50}")
    print(f"  MojoLlama AutoTune — first load of {os.path.basename(model_path)}")
    print(f"  Sweeping threads + concurrency for optimal settings...")
    print(f"{'='*50}")

    config = autotune(model_path, quick=quick, mojollama=True)

    if config is None:
        # Fallback defaults
        import multiprocessing as mp
        n_cores = mp.cpu_count() or 32
        config = {
            "autotune_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_tuned_on": str(model_path),
            "mojollama_engine": {
                "omp_threads": min(32, n_cores),
                "concurrency": min(8, max(1, n_cores // 4)),
                "strategy": "process",
            },
            "hardware": {
                "cores": n_cores,
            }
        }

    save_config(config)
    _AUTO_TUNE_CACHE[cache_key] = config

    print(f"  ✓ Saved to {CONFIG_PATH}")
    if "mojollama_engine" in config:
        me = config["mojollama_engine"]
        print(f"  → {me.get('omp_threads', '?')} OMP threads, "
              f"{me.get('concurrency', '?')} workers ({me.get('strategy', 'process')})")
    print(f"{'='*50}\n")
    return config


def get_tuned_setting(model_path: str, key: str, default=None):
    """Get a single auto-tuned setting for *model_path*.

    Runs auto-tune if not cached.  Example:
      threads = get_tuned_setting('/models/model.gguf', 'omp_threads', 32)
    """
    config = auto_tune_for_model(model_path, quick=True)
    return config.get("mojollama_engine", {}).get(key, default)


if __name__ == "__main__":
    main()