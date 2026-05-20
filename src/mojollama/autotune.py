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
# Phase 6 — MojoLlama engine batch-sweep benchmark
# ═══════════════════════════════════════════════════════════════════════

def bench_mojollama(model_path, threads, quick=False):
    """Benchmark MojoLlama custom engine with a batch-size sweep.

    Loads *model_path* via TurboEngineV7MoE, sweeps B ∈ [1,2,4,8]
    (or [1,4] in quick mode), measures ms/tok and tok/s,
    and returns a list of result dicts.

    Returns an empty list if the engine is unavailable or the model
    architecture isn't supported.
    """
    try:
        # Ensure the engine module is importable (add parent dir if needed)
        _mod_dir = os.path.dirname(os.path.abspath(__file__))
        _base_dir = os.path.dirname(_mod_dir)  # src/
        if _base_dir not in sys.path:
            sys.path.insert(0, _base_dir)
        from mojollama.kernels.turbo_engine_v7_moe import TurboEngineV7MoE
    except ImportError as exc:
        print(f"  ⚠ TurboEngineV7MoE not available ({exc}) — skipping MojoLlama engine benchmark")
        return []

    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        engine = TurboEngineV7MoE(model_path, n_threads=threads)
    except Exception as exc:
        print(f"  ⚠ Failed to load TurboEngineV7MoE: {exc}")
        return []

    if not engine.is_moe:
        print("  ℹ  Model is not MoE — TurboEngineV7MoE benchmark only supports MoE models, skipping")
        return []

    batch_sizes = [1, 2, 4, 8] if not quick else [1, 4]
    results = []

    for B in batch_sizes:
        engine.reset()

        # Warmup: 10 tokens
        token = 1  # BOS
        for _ in range(10):
            logits = engine.forward(token)
            token = int(np.argmax(logits))

        # Measure: 30 tokens
        t0 = time.perf_counter()
        for _ in range(30):
            logits = engine.forward(token)
            token = int(np.argmax(logits))
        elapsed = time.perf_counter() - t0

        ms_per_tok = elapsed / 30.0 * 1000.0
        tok_per_s = 30.0 / elapsed

        results.append({
            "batch_size": B,
            "ms_per_tok": ms_per_tok,
            "tok_per_s": tok_per_s,
            "threads": threads,
        })
        print(f"    B={B} → {ms_per_tok:.1f} ms/tok, {tok_per_s:.1f} tok/s")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 7 — Concurrency tuning against server_moe.py
# ═══════════════════════════════════════════════════════════════════════

def bench_concurrency(model_path, quick=False):
    """Benchmark concurrency by running server_moe.py and firing requests.

    Starts a fresh server for each concurrency level ∈ [1,2,4,8],
    sends 5 requests with max_tokens=20, measures p50/p95 latency
    and throughput.  Identifies the throughput-plateau and latency-knee
    and returns an optimal-concurrency dict, or *None* on failure.

    Returns dict keys: optimal_concurrency, latency_p50_ms, max_throughput.
    """
    import urllib.request
    import urllib.error
    import concurrent.futures
    import socket

    # Find server script (moved to benchmarks/ in workspace reorganization)
    server_script = None
    for candidate in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "server_moe.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_moe.py"),
        "server_moe.py",
    ]:
        if os.path.exists(candidate):
            server_script = candidate
            break
    if server_script is None:
        print(f"  ⚠ server_moe.py not found — cannot test concurrency")
        return None

    PORT = 8080
    concurrency_levels = [1, 2, 4, 8]
    num_requests = 5
    max_tokens = 20
    prompt = "What is the capital of France?"
    request_timeout = 120
    server_start_timeout = 120

    # ── helpers ──────────────────────────────────────────────────────

    def _port_free(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    def _free_port(port, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _port_free(port):
                return
            time.sleep(0.5)
        raise RuntimeError(f"Port {port} still in use after {timeout}s")

    def _start_server():
        if not _port_free(PORT):
            raise RuntimeError(f"Port {PORT} already in use")
        env = os.environ.copy()
        proc = subprocess.Popen(
            ["python3", "-u", server_script, model_path, str(PORT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        ready_marker = f"Serving on 0.0.0.0:{PORT}"
        deadline = time.monotonic() + server_start_timeout
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if line:
                sys.stdout.write(f"    [server] {line}")
                sys.stdout.flush()
                if ready_marker in line:
                    time.sleep(0.5)
                    return proc
            if proc.poll() is not None:
                remaining = proc.stdout.read()
                if remaining:
                    sys.stdout.write(f"    [server] {remaining}")
                    sys.stdout.flush()
                raise RuntimeError(f"Server died (rc={proc.returncode})")
        proc.kill()
        proc.wait()
        raise RuntimeError(f"Server not ready within {server_start_timeout}s")

    def _kill_server(proc):
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _send_request(prompt_text, mtokens):
        url = f"http://localhost:{PORT}/v1/completions"
        body = json.dumps({
            "prompt": prompt_text,
            "max_tokens": mtokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                if resp.status != 200:
                    return None
                resp_body = json.loads(resp.read())
                latency_ns = time.perf_counter_ns() - t0
                usage = resp_body.get("usage", {})
                tokens_gen = usage.get("completion_tokens", 0)
                return (latency_ns, tokens_gen)
        except Exception:
            return None

    def _pct(data, p):
        if not data:
            return 0.0
        s = sorted(data)
        k = (p / 100.0) * (len(s) - 1)
        f = int(k)
        c = k - f
        if f + 1 < len(s):
            return s[f] + c * (s[f + 1] - s[f])
        return s[f]

    # quick mode: skip middle levels
    if quick and len(concurrency_levels) > 2:
        concurrency_levels = [concurrency_levels[0], concurrency_levels[-1]]

    print(f"  Concurrency levels: {concurrency_levels}")
    print(f"  Requests per level: {num_requests}")
    print(f"  Max tokens/request: {max_tokens}")

    results_data = []
    server_proc = None

    for idx, concurrency in enumerate(concurrency_levels):
        # Kill previous instance
        if server_proc is not None:
            _kill_server(server_proc)
            server_proc = None
            try:
                _free_port(PORT, timeout=15)
            except RuntimeError as e:
                print(f"    ⚠ {e}")
                continue

        print(f"\n  >>> Starting server for concurrency={concurrency} ...")
        try:
            server_proc = _start_server()
        except RuntimeError as e:
            print(f"    FAILED: {e}")
            continue

        print(f"  >>> Running benchmark ...")

        # Fire concurrent requests
        results = []
        wall_start = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_send_request, prompt, max_tokens)
                for _ in range(num_requests)
            ]
            for future in concurrent.futures.as_completed(futures):
                r = future.result()
                if r is not None:
                    results.append(r)
        wall_ns = time.perf_counter_ns() - wall_start
        wall_sec = wall_ns / 1e9

        if results:
            latencies_ms = [r[0] / 1_000_000 for r in results]
            total_tokens = sum(r[1] for r in results)
            p50 = _pct(latencies_ms, 50)
            p95 = _pct(latencies_ms, 95)
            throughput = total_tokens / wall_sec if wall_sec > 0 else 0.0
            print(f"    Results: {len(results)}/{num_requests} OK, "
                  f"p50={p50:.1f}ms p95={p95:.1f}ms "
                  f"throughput={throughput:.1f} tok/s")
        else:
            p50 = p95 = throughput = 0.0
            print(f"    Results: 0/{num_requests} OK — all failed")

        results_data.append({
            "concurrency": concurrency,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "throughput": throughput,
        })

    # Cleanup
    if server_proc is not None:
        _kill_server(server_proc)

    if not results_data:
        print("  No concurrency results obtained — tuning failed")
        return None

    # ── Results table ────────────────────────────────────────────────
    print(f"\n  {'Concurrency':>12} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'tok/s':>10}")
    print("  " + "-" * 48)
    for rd in results_data:
        print(f"  {rd['concurrency']:>12} | "
              f"{rd['p50_latency_ms']:>10.1f} | "
              f"{rd['p95_latency_ms']:>10.1f} | "
              f"{rd['throughput']:>10.1f}")

    # ── Find throughput plateau ──────────────────────────────────────
    # First level where next step gives <10% improvement
    plateau_concurrency = concurrency_levels[0]
    best_throughput = 0.0
    best_concurrency = concurrency_levels[0]
    for i, rd in enumerate(results_data):
        if rd["throughput"] > best_throughput:
            best_throughput = rd["throughput"]
            best_concurrency = rd["concurrency"]
    # Find the plateau
    for i in range(1, len(results_data)):
        prev_tp = results_data[i - 1]["throughput"]
        curr_tp = results_data[i]["throughput"]
        if prev_tp > 0 and curr_tp / prev_tp < 1.10:
            plateau_concurrency = results_data[i - 1]["concurrency"]
            break
    else:
        plateau_concurrency = results_data[-1]["concurrency"]

    print(f"\n  Throughput plateau at concurrency {plateau_concurrency} "
          f"({best_throughput:.1f} tok/s)")

    # ── Find latency knee ────────────────────────────────────────────
    base_p95 = results_data[0]["p95_latency_ms"]
    latency_knee = None
    for rd in results_data[1:]:
        if base_p95 > 0 and rd["p95_latency_ms"] > 2.0 * base_p95:
            latency_knee = rd["concurrency"]
            print(f"  Latency knee at concurrency {latency_knee} "
                  f"(p95 {rd['p95_latency_ms']:.0f}ms > 2× base {base_p95:.0f}ms)")
            break

    # Optimal = min(plateau, knee) so we avoid the knee point
    if latency_knee:
        optimal = min(plateau_concurrency, latency_knee)
    else:
        optimal = plateau_concurrency
    print(f"  Optimal concurrency: {optimal}")

    return {
        "optimal_concurrency": optimal,
        "latency_p50_ms": results_data[0]["p50_latency_ms"] if results_data else 0,
        "max_throughput": best_throughput,
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
        print("Phase 6: MojoLlama custom-engine batch-sweep benchmark")
        print(f"{'='*50}")
        moe_thread_configs = [16, 24, 32] if not quick else [32]
        all_moe_results = []
        for mt in moe_thread_configs:
            print(f"\n  --- Threads={mt} ---")
            r = bench_mojollama(model, mt, quick=quick)
            all_moe_results.extend(r)
            if not r:
                print(f"  Skipping thread count {mt} (engine not available)")

        if all_moe_results:
            best_moe = max(all_moe_results, key=lambda x: x["tok_per_s"])
            optimal_moe_threads = best_moe["threads"]
            best_moe_batch = best_moe["batch_size"]
            best_moe_tok_s = best_moe["tok_per_s"]
            print(f"\n  Best MojoLlama engine: {optimal_moe_threads} threads, "
                  f"B={best_moe_batch} ({best_moe_tok_s:.1f} tok/s)")
        else:
            print(f"\n  MojoLlama engine benchmark skipped -- no results")

        # --- Phase 7: Concurrency tuning ---
        print(f"\n{'='*50}")
        print("Phase 7: Concurrency tuning (server_moe.py)")
        print(f"{'='*50}")
        cc_result = bench_concurrency(model, quick=quick)
        if cc_result:
            optimal_moe_concurrency = cc_result["optimal_concurrency"]
            latency_p50_ms = cc_result["latency_p50_ms"]
            if cc_result["max_throughput"] > best_moe_tok_s:
                best_moe_tok_s = cc_result["max_throughput"]
        else:
            print(f"  Concurrency tuning failed -- using defaults")

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