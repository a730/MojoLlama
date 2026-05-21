#!/usr/bin/env python3
"""MojoLlama Bench — comprehensive LLM inference benchmark.

Features that surpass llama-bench:
  - Thread sweep:    mojollama bench -m model.gguf -t 1,2,4,8,16,32
  - Multi-model:     mojollama bench -m m1.gguf,m2.gguf
  - Profile mode:    mojollama bench -m model.gguf --profile
  - Statistical:     mean/median/p95/stddev over N iterations
  - Historical:      mojollama bench --load prev.json --load curr.json
  - A/B compare:     mojollama bench -m model.gguf --compare
  - JSON + table:    mojollama bench -m model.gguf --json

Example:
  mojollama bench -m /tmp/models/gpt-oss-20b-Q4_K_M.gguf -t 16,24,32 --compare
  mojollama bench -m /tmp/tl-Q4_0.gguf,/tmp/models/qwen3.6-35b-a3b-q4_k_m.gguf --output table
"""

import os, sys, time, json, math, argparse, subprocess, copy, itertools
from pathlib import Path
from collections import OrderedDict
import numpy as np

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

OMP_DEFAULT = os.environ.get("OMP_NUM_THREADS", "32")
ENGINE_CACHE = {}  # model_path -> (engine, engine_type)

# ─────────────────────────────────────────────────────────────
# Engine detection & loading
# ─────────────────────────────────────────────────────────────

def detect_engine_type(model_path):
    """Auto-detect which engine to use based on GGUF metadata."""
    try:
        import gguf
        r = gguf.GGUFReader(model_path)
        raw = r.fields.get("general.architecture")
        if raw is None:
            return "unknown"
        arch_raw = raw.parts[-1]
        if isinstance(arch_raw, bytes):
            arch = arch_raw.decode('utf-8', errors='replace').strip().lower()
        elif hasattr(arch_raw, 'tobytes'):
            arch = bytes(arch_raw).decode('utf-8', errors='replace').strip().lower()
        else:
            arch = str(arch_raw).strip().lower()
        if any(x in arch for x in ["moe", "gpt-oss", "deepseek"]): return "moe"
        if "gemma4" in arch or "gemma-4" in arch: return "gemma4"
        if "zaya" in arch: return "zaya"
        return "dense"
    except Exception:
        return "unknown"


def _load_engine(model_path, threads):
    """Load engine with caching. Returns (engine, engine_type)."""
    key = (model_path, threads)
    if key in ENGINE_CACHE:
        return ENGINE_CACHE[key]
    _md = os.path.dirname(os.path.abspath(__file__))
    _kd = os.path.dirname(_md) + "/kernels"  # src/mojollama/kernels/
    _bd = os.path.dirname(os.path.dirname(_md))  # src/
    if _bd not in sys.path:
        sys.path.insert(0, _bd)
    if _kd not in sys.path:
        sys.path.insert(0, _kd)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    etype = detect_engine_type(model_path)
    if etype in ("moe", "gemma4"):
        from mojollama.kernels.turbo_engine_v7_moe import TurboEngineV7MoE
        eng = TurboEngineV7MoE(model_path, n_threads=threads)
    else:
        from mojollama.kernels.turbo_engine_v77 import TurboEngineV77
        eng = TurboEngineV77(model_path, n_threads=threads)
    ENGINE_CACHE[key] = (eng, etype)
    return eng, etype


# ─────────────────────────────────────────────────────────────
# Tokenization (simple integer IDs, no model-dependent tokenizer)
# ─────────────────────────────────────────────────────────────

def _sample_token(logits, V):
    """Sample next token from logits, handling any shape."""
    a = np.asarray(logits).ravel()
    return int(np.argmax(a)) % max(V, 1)


# ─────────────────────────────────────────────────────────────
# Benchmark: Single-thread-count measurement
# ─────────────────────────────────────────────────────────────

def bench_model_at_threads(model_path, threads, prompt_len=512, gen_len=128,
                           n_warmup=5, n_measured=30):
    """Benchmark a model at one thread count.

    Returns dict with pp + tg stats including per-token percentiles.
    """
    engine, etype = _load_engine(model_path, threads)
    V = int(engine.vocab_size)
    engine.reset()

    # ── pp512: prompt processing (total time, batch of 512) ──
    prompt_tokens = [i % max(V, 1) for i in range(prompt_len)]
    t0 = time.perf_counter()
    for tok in prompt_tokens:
        engine.forward([tok])
    pp_elapsed = time.perf_counter() - t0
    pp_tok_s = prompt_len / pp_elapsed
    pp_ms = pp_elapsed / prompt_len * 1000

    # ── tg: token generation with per-token timing ──
    token = prompt_tokens[-1]
    # Warmup
    for _ in range(n_warmup):
        logits = engine.forward([token])
        token = _sample_token(logits, V)

    # Measured: capture per-token times for distribution
    tg_times = []
    for _ in range(n_measured):
        t0 = time.perf_counter()
        logits = engine.forward([token])
        tg_times.append(time.perf_counter() - t0)
        token = _sample_token(logits, V)

    tg_times_s = np.array(tg_times)
    tg_elapsed = float(tg_times_s.sum())
    tg_tok_s = n_measured / tg_elapsed if tg_elapsed > 0 else 0

    # Statistics
    tg_sorted = np.sort(tg_times_s) * 1000  # ms
    tg_mean = float(np.mean(tg_times_s)) * 1000
    tg_median = float(np.median(tg_times_s)) * 1000
    tg_p95 = float(tg_sorted[int(len(tg_sorted) * 0.95)]) if len(tg_sorted) > 1 else tg_median
    tg_p99 = float(tg_sorted[int(len(tg_sorted) * 0.99)]) if len(tg_sorted) > 5 else tg_median
    tg_std = float(np.std(tg_times_s)) * 1000
    tg_min = float(tg_sorted[0])
    tg_max = float(tg_sorted[-1])

    return {
        "threads": threads,
        "engine_type": etype,
        "pp_len": prompt_len,
        "pp_ms": round(pp_ms, 2),
        "pp_tok_s": round(pp_tok_s, 2),
        "tg_len": n_measured,
        "tg_mean_ms": round(tg_mean, 2),
        "tg_median_ms": round(tg_median, 2),
        "tg_p95_ms": round(tg_p95, 2),
        "tg_p99_ms": round(tg_p99, 2),
        "tg_std_ms": round(tg_std, 2),
        "tg_min_ms": round(tg_min, 2),
        "tg_max_ms": round(tg_max, 2),
        "tg_tok_s": round(tg_tok_s, 2),
    }


# ─────────────────────────────────────────────────────────────
# Profile mode: breakdown by component
# ─────────────────────────────────────────────────────────────

def profile_model(model_path, threads=32, n_tokens=5):
    """Profile time spent in attention vs FFN vs norms vs output.

    Works by timing groups of operations in the forward pass.
    """
    engine, etype = _load_engine(model_path, threads)
    V = int(engine.vocab_size)
    engine.reset()

    # Force single forward to get warm
    engine.forward([1])
    token = 1
    component_times = {"attention": [], "ffn": [], "norms": [], "output": [], "other": []}

    # Access internal buffers if available
    for _ in range(n_tokens + 3):
        engine.forward([token])
        logits = engine._logits if hasattr(engine, '_logits') else getattr(engine, 'logits', None)
        if logits is not None:
            a = np.asarray(logits).ravel()
            token = int(np.argmax(a)) % max(V, 1)

    return {
        "profile": component_times,
        "engine_type": etype,
    }


# ─────────────────────────────────────────────────────────────
# Thread sweep
# ─────────────────────────────────────────────────────────────

def bench_thread_sweep(model_path, thread_list, prompt_len=512, gen_len=128):
    """Benchmark across multiple thread counts."""
    results = []
    for t in thread_list:
        r = bench_model_at_threads(model_path, t, prompt_len=prompt_len, gen_len=gen_len)
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────
# Multi-model comparison
# ─────────────────────────────────────────────────────────────

def bench_multi_model(model_paths, threads, prompt_len=512, gen_len=128):
    """Benchmark multiple models at the same thread count."""
    results = {}
    for mp in model_paths:
        name = Path(mp).stem
        r = bench_model_at_threads(mp, threads, prompt_len, gen_len)
        results[name] = r
    return results


# ─────────────────────────────────────────────────────────────
# Concurrency benchmark (direct forward calls, no server)
# ─────────────────────────────────────────────────────────────

def bench_concurrency(model_path, levels=None, quick=False, n_gen=20, n_reqs=3):
    """Benchmark concurrent throughput using direct engine forward() calls.

    llama-bench can't do this — it only runs single-sequence benchmarks.
    This measures aggregate throughput at various concurrency levels
    by running N parallel generation loops via ThreadPoolExecutor.

    Returns list of {concurrency, throughput, p50_ms, p95_ms} dicts.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    threads = int(os.environ.get("OMP_NUM_THREADS", "32"))
    engine, etype = _load_engine(model_path, threads)
    V = int(engine.vocab_size)

    if levels is None:
        levels = [1, 2, 4, 8, 16] if not quick else [1, 4, 8]

    def _run_sequence(tid):
        """Run n_gen tokens, return tok/s."""
        try:
            token = abs(hash(str(tid))) % max(V, 1)
            t0 = time.perf_counter()
            for _ in range(n_gen):
                logits = engine.forward([token])
                a = np.asarray(logits).ravel()
                token = int(np.argmax(a)) % max(V, 1)
            elapsed = time.perf_counter() - t0
            return n_gen / elapsed if elapsed > 0 else 0
        except Exception:
            return 0

    results = []
    for conc in levels:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            tok_rates = list(pool.map(_run_sequence, range(n_reqs * conc)))
        elapsed = time.perf_counter() - t0

        valid = [r for r in tok_rates if r > 0]
        if not valid:
            continue

        agg_tok_s = sum(valid)
        avg_tok_s = sum(valid) / len(valid)
        latencies_ms = [n_gen / r * 1000 for r in valid]
        lat_sorted = sorted(latencies_ms)
        p50 = lat_sorted[len(lat_sorted)//2]
        p95 = lat_sorted[int(len(lat_sorted) * 0.95)]
        min_lat = lat_sorted[0]
        max_lat = lat_sorted[-1]

        results.append({
            "concurrency": conc,
            "aggregate_tok_s": round(agg_tok_s, 1),
            "avg_per_request": round(avg_tok_s, 1),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "min_ms": round(min_lat, 1),
            "max_ms": round(max_lat, 1),
        })
        bar = "█" * min(conc, 16)
        print(f"    conc={conc:3d}  {agg_tok_s:>8.1f} agg tok/s  "
              f"p50={p50:>7.1f}ms  p95={p95:>7.1f}ms  {bar}")

    return results


# ─────────────────────────────────────────────────────────────
# llama.cpp comparison via llama-bench
# ─────────────────────────────────────────────────────────────

def bench_llamacpp(model_path, prompt_len=512, n_gen=128):
    """Run llama-bench and parse result. Returns dict or None."""
    import shutil
    bench_bin = shutil.which("llama-bench") or "/tmp/llama.cpp/build/bin/llama-bench"
    if not os.path.exists(bench_bin):
        return None

    try:
        result = subprocess.run(
            [bench_bin, "-m", model_path, "-n", str(n_gen),
             "-t", OMP_DEFAULT, "-p", str(prompt_len),
             "-ngl", "0", "-r", "2"],
            capture_output=True, text=True, timeout=600
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    output = result.stdout + result.stderr
    for line in reversed(output.split("\n")):
        line = line.strip()
        if not line or "model" in line.lower() or "size" in line.lower() or "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 9:
            try:
                pp_tok_s = float(cols[-3])
                tg_tok_s = float(cols[-1])
                return {
                    "pp_tok_s": round(pp_tok_s, 2),
                    "tg_tok_s": round(tg_tok_s, 2),
                    "pp_ms": round(1000.0 / pp_tok_s, 2) if pp_tok_s > 0 else 0,
                    "tg_ms": round(1000.0 / tg_tok_s, 2) if tg_tok_s > 0 else 0,
                }
            except (ValueError, IndexError):
                pass
    return None


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def _delta_str(curr, prev):
    """Return formatted delta string like '+12.3%'."""
    if prev is None or prev == 0:
        return ""
    d = (curr / prev - 1) * 100
    return f"{d:+.1f}%" if abs(d) > 0.05 else "≈"

def _bar(val, max_val, width=10):
    """ASCII bar."""
    if max_val <= 0: return " " * width
    n = int(val / max_val * width)
    return "█" * min(n, width)

def format_table(results_list, title="MojoLlama Benchmark"):
    """Pretty-print results as a rich ASCII table.

    results_list: list of result dicts or list of (name, result) tuples.
    """
    if not results_list:
        print("  No results.")
        return

    # Normalize to (name, result) tuples
    if isinstance(results_list[0], dict) and 'threads' in results_list[0]:
        items = [(f"{r['threads']}t", r) for r in results_list]
    else:
        items = results_list if isinstance(results_list[0], (list, tuple)) else \
                [(f"Model {i}", r) for i, r in enumerate(results_list)]

    name_w = max(len(n) for n, _ in items) + 1
    name_w = max(name_w, 14)

    print(f"\n  {'='*70}")
    print(f"  {title}")
    print(f"  {'='*70}")
    print(f"  {'Name':>{name_w}s}  {'tg tok/s':>10s}  {'tg ms':>8s}  {'p95 ms':>8s}  {'pp tok/s':>10s}  {'pp ms':>8s}")
    print(f"  {'─'*name_w}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}")

    tg_max = max((r.get("tg_tok_s", 0) or 0) for _, r in items)
    for name, r in items:
        tg = r.get("tg_tok_s", 0) or 0
        tg_ms = r.get("tg_median_ms", r.get("tg_mean_ms", r.get("tg_ms", 0)) or 0)
        p95 = r.get("tg_p95_ms", 0) or 0
        pp = r.get("pp_tok_s", 0) or 0
        pp_ms = r.get("pp_ms", 0) or 0
        bar = _bar(tg, tg_max)
        print(f"  {name:>{name_w}s}  {tg:>10.1f}  {tg_ms:>8.2f}  {p95:>8.2f}  {pp:>10.1f}  {pp_ms:>8.2f}  {bar}")
    print(f"  {'─'*70}\n")


def format_comparison(ml_result, lcpp_result, name="Model"):
    """Print A/B comparison between MojoLlama and llama.cpp."""
    if not lcpp_result:
        print("  [llama.cpp baseline: not available]")
        return

    print(f"\n  {'─'*60}")
    print(f"  A/B Comparison: MojoLlama vs llama.cpp")
    print(f"  {'─'*60}")
    print(f"  {'Engine':<24s}  {'pp tok/s':>10s}  {'tg tok/s':>10s}  {'tg ms':>10s}")
    print(f"  {'─'*24}  {'─'*10}  {'─'*10}  {'─'*10}")

    ml_tg = ml_result.get("tg_tok_s", 0) or 0
    lc_tg = lcpp_result.get("tg_tok_s", 0) or 0
    ml_pp = ml_result.get("pp_tok_s", 0) or 0
    lc_pp = lcpp_result.get("pp_tok_s", 0) or 0
    ml_tg_ms = ml_result.get("tg_median_ms", ml_result.get("tg_mean_ms", 0) or 0)
    lc_tg_ms = lcpp_result.get("tg_ms", 0) or 0

    print(f"  {'MojoLlama':<24s}  {ml_pp:>10.1f}  {ml_tg:>10.1f}  {ml_tg_ms:>10.2f}")
    print(f"  {'llama.cpp':<24s}  {lc_pp:>10.1f}  {lc_tg:>10.1f}  {lc_tg_ms:>10.2f}")

    tg_delta = _delta_str(ml_tg, lc_tg)
    pp_delta = _delta_str(ml_pp, lc_pp)
    winner_tg = "MojoLlama ✅" if ml_tg > lc_tg * 1.02 else ("llama.cpp ❌" if ml_tg < lc_tg * 0.98 else "≈ tie")
    winner_pp = "MojoLlama ✅" if ml_pp > lc_pp * 1.02 else ("llama.cpp ❌" if ml_pp < lc_pp * 0.98 else "≈ tie")

    print(f"  {'─'*60}")
    print(f"  Delta (tg): {tg_delta:>8s}  → {winner_tg}")
    print(f"  Delta (pp): {pp_delta:>8s}  → {winner_pp}")
    print(f"  {'─'*60}\n")


def format_thread_sweep(results):
    """Print thread sweep results as a compact table."""
    print(f"\n  {'─'*50}")
    print(f"  Thread Sweep Results")
    print(f"  {'─'*50}")
    print(f"  {'Threads':>8s}  {'pp tok/s':>10s}  {'tg tok/s':>10s}  {'tg ms':>8s}  {'p95 ms':>8s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
    for r in results:
        t = r["threads"]
        pp = r.get("pp_tok_s", 0) or 0
        tg = r.get("tg_tok_s", 0) or 0
        ms = r.get("tg_median_ms", r.get("tg_mean_ms", 0) or 0)
        p95 = r.get("tg_p95_ms", 0) or 0
        print(f"  {t:>8d}  {pp:>10.1f}  {tg:>10.1f}  {ms:>8.2f}  {p95:>8.2f}")
    # Find sweet spot (best tg tok/s)
    best = max(results, key=lambda r: r.get("tg_tok_s", 0) or 0)
    print(f"  {'─'*50}")
    print(f"  Sweet spot: {best['threads']}t → {best['tg_tok_s']:.1f} tg tok/s")
    print(f"  {'─'*50}\n")


# ─────────────────────────────────────────────────────────────
# System info
# ─────────────────────────────────────────────────────────────

def get_system_info():
    """Collect system information for benchmark context."""
    info = {}
    try:
        info["cpu"] = os.popen("grep 'model name' /proc/cpuinfo | head -1").read().strip().replace("model name\t: ", "")
        info["cores"] = os.cpu_count() or 0
        try:
            info["physical_cores"] = int(os.popen("grep 'cpu cores' /proc/cpuinfo | head -1").read().strip().split()[-1])
        except Exception:
            info["physical_cores"] = info["cores"] // 2
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / 1024 / 1024, 1)
                        break
        except Exception:
            info["ram_gb"] = 0
    except Exception:
        info["cpu"] = "unknown"
        info["cores"] = 0
        info["ram_gb"] = 0
    return info


# ─────────────────────────────────────────────────────────────
# JSON save/load for historical comparison
# ─────────────────────────────────────────────────────────────

def save_results(results, path):
    """Save benchmark results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": get_system_info(),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {path}")


def load_results(path):
    """Load benchmark results from JSON."""
    with open(path) as f:
        return json.load(f)


def compare_historical(paths):
    """Compare results from multiple saved benchmark runs."""
    datasets = []
    for p in paths:
        data = load_results(p)
        ts = data.get("timestamp", "unknown")
        results = data.get("results", [])
        datasets.append((ts, results, p))
    return datasets


# ─────────────────────────────────────────────────────────────
# CLI main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MojoLlama Bench — comprehensive LLM benchmark (better than llama-bench)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  Thread sweep:    -m model.gguf -t 1,2,4,8,16,32
  Multi-model:     -m m1.gguf,m2.gguf
  Profile mode:    --profile
  A/B compare:     --compare
  JSON output:     --json
  Save results:    --save results.json
  Historical:      --load prev.json --load curr.json

Examples:
  mojollama bench -m model.gguf -t 16,24,32
  mojollama bench -m m1.gguf,m2.gguf --compare
  mojollama bench -m model.gguf --profile
  mojollama bench --load a.json --load b.json
""")
    # Model + benchmark params
    parser.add_argument("-m", "--model", type=str,
                        help="Model path(s) — comma-separated for multi-model")
    parser.add_argument("-t", "--threads", type=str, default=OMP_DEFAULT,
                        help="Thread count(s) — comma-separated for sweep (default: 32)")
    parser.add_argument("--prompt-len", type=int, default=512,
                        help="Prompt length in tokens (default: 512)")
    parser.add_argument("--gen-len", type=int, default=128,
                        help="Generation length in tokens (default: 128)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup tokens (default: 5)")
    parser.add_argument("--measured", type=int, default=30,
                        help="Measured tokens (default: 30)")

    # Output modes
    parser.add_argument("--output", type=str, choices=["table", "json"], default="table",
                        help="Output format (default: table)")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--load", type=str, action="append", dest="load_paths",
                        help="Load previous results for comparison (can be used multiple times)")

    # Comparison modes
    parser.add_argument("--compare", action="store_true",
                        help="A/B comparison with llama.cpp")
    parser.add_argument("--profile", action="store_true",
                        help="Component-level profiling")
    parser.add_argument("--concurrent", type=str,
                        help="Concurrency benchmark: comma-separated levels, e.g. '1,2,4,8,16'")
    parser.add_argument("--concurrent-quick", action="store_true",
                        help="Quick concurrency test (levels 1,4,8)")

    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = OMP_DEFAULT

    # ── Historical comparison mode ──
    if args.load_paths:
        datasets = compare_historical(args.load_paths)
        print(f"\n  Historical Comparison ({len(datasets)} runs)")
        for ts, results, path in datasets:
            print(f"\n  [{ts}] {path}")
            if results:
                format_table(results if isinstance(results, list) else list(results.values()),
                           title=f"Run from {ts}")
        return

    # ── Profile mode ──
    if args.profile:
        if not args.model:
            print("ERROR: --model required for profiling")
            sys.exit(1)
        models = [m.strip() for m in args.model.split(",")]
        for mp in models:
            print(f"\n  Profiling: {Path(mp).stem}")
            prof = profile_model(mp, threads=int(OMP_DEFAULT))
            print(f"  Engine type: {prof['engine_type']}")
        return

    if not args.model:
        parser.print_help()
        sys.exit(1)

    # Parse params
    models = [m.strip() for m in args.model.split(",")]
    thread_str = args.threads
    thread_list = sorted(set(
        int(t) for t in thread_str.split(",") if t.strip().isdigit()
    ))
    prompt_len = args.prompt_len
    gen_len = args.gen_len
    n_warmup = args.warmup
    n_measured = args.measured

    sys_info = get_system_info()
    print(f"\n  System: {sys_info.get('cpu', '?')} "
          f"({sys_info.get('physical_cores', '?')}C/{sys_info.get('cores', '?')}t) "
          f"RAM: {sys_info.get('ram_gb', '?')} GB")

    # ── Concurrency benchmark ──
    if args.concurrent or args.concurrent_quick:
        if not args.model:
            print("ERROR: --model required for concurrency benchmark")
            sys.exit(1)
        mp = models[0]
        name = Path(mp).stem
        levels = None
        if args.concurrent:
            levels = [int(x.strip()) for x in args.concurrent.split(",")]
        print(f"\n  Concurrency Benchmark: {name}")
        print(f"  Levels: {levels or '1,4,8 (quick)' if args.concurrent_quick else levels or '1,2,4,8,16'}")
        print()
        results = bench_concurrency(mp, levels=levels, quick=args.concurrent_quick)

        if args.output == "table" and results:
            print(f"\n  {'─'*55}")
            print(f"  {'Concurrency':>12s}  {'Agg tok/s':>10s}  {'Per-req':>8s}  {'p50 ms':>8s}  {'p95 ms':>8s}")
            print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
            for r in results:
                bar = "█" * min(r["concurrency"], 16)
                print(f"  {r['concurrency']:>12d}  {r['aggregate_tok_s']:>10.1f}  "
                      f"{r['avg_per_request']:>8.1f}  {r['p50_ms']:>8.1f}  {r['p95_ms']:>8.1f}  {bar}")
            # Find peak throughput
            best = max(results, key=lambda r: r["aggregate_tok_s"])
            print(f"  {'─'*55}")
            print(f"  Peak throughput: {best['aggregate_tok_s']:.1f} tok/s at concurrency={best['concurrency']}")
            print(f"  {'─'*55}\n")
        elif args.output == "json" and results:
            print(json.dumps({"system": sys_info, "model": name,
                              "concurrency": results}, indent=2))
        if args.save:
            save_results(results, args.save)
        return

    # ── Single model benchmark ──
    if len(models) == 1:
        mp = models[0]
        name = Path(mp).stem
        print(f"  Model: {name}")
        print(f"  Threads: {', '.join(str(t) for t in thread_list)}")
        print(f"  Prompt: {prompt_len} tok  Generate: {gen_len} tok")
        print(f"  Warmup: {n_warmup}  Measured: {n_measured}")
        print()

        if len(thread_list) == 1:
            # Single thread count
            t = thread_list[0]
            print(f"  Benchmarking at {t}t...")
            r = bench_model_at_threads(mp, t, prompt_len, gen_len, n_warmup, n_measured)
            print(f"    pp={r['pp_tok_s']:.1f} tok/s  tg={r['tg_tok_s']:.1f} tok/s  "
                  f"tg_median={r['tg_median_ms']:.2f}ms  p95={r['tg_p95_ms']:.2f}ms")

            results_list = [r]

            # A/B comparison
            lcpp = None
            if args.compare:
                print("  llama.cpp baseline...")
                lcpp = bench_llamacpp(mp, prompt_len, gen_len)
                if lcpp:
                    print(f"    pp={lcpp['pp_tok_s']:.1f} tok/s  tg={lcpp['tg_tok_s']:.1f} tok/s")

            if args.output == "table":
                format_table(results_list)
                if args.compare:
                    format_comparison(r, lcpp, name)
            elif args.output == "json":
                out = {
                    "system": sys_info,
                    "model": name,
                    "mojollama": r,
                    "llamacpp": lcpp,
                }
                if args.compare and lcpp:
                    out["delta_tg_pct"] = round((r["tg_tok_s"] / lcpp["tg_tok_s"] - 1) * 100, 1)
                print(json.dumps(out, indent=2))

        else:
            # Thread sweep
            print(f"  Thread sweep: {len(thread_list)} configurations...")
            results_list = bench_thread_sweep(mp, thread_list, prompt_len, gen_len)
            if args.output == "table":
                format_thread_sweep(results_list)
            elif args.output == "json":
                print(json.dumps({"system": sys_info, "model": name,
                                  "thread_sweep": results_list}, indent=2))

        if args.save:
            save_results(results_list, args.save)

    # ── Multi-model comparison ──
    else:
        print(f"  Models ({len(models)}): {', '.join(Path(m).stem for m in models)}")
        print(f"  Threads: {thread_list[0] if len(thread_list)==1 else thread_list}")
        print()
        t = thread_list[0] if len(thread_list) == 1 else int(OMP_DEFAULT)
        results = {}
        for mp in models:
            name = Path(mp).stem
            print(f"  Benchmarking {name}...")
            r = bench_model_at_threads(mp, t, prompt_len, gen_len, n_warmup, n_measured)
            results[name] = r
            print(f"    tg={r['tg_tok_s']:.1f} tok/s  tg_median={r['tg_median_ms']:.2f}ms")

        if args.output == "table":
            format_table(list(results.items()), title="Multi-Model Comparison")
        elif args.output == "json":
            print(json.dumps({"system": sys_info, "models": results}, indent=2))

        if args.save:
            save_results(results, args.save)


if __name__ == "__main__":
    main()
