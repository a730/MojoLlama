#!/usr/bin/env python3
"""MojoLlama Concurrent Benchmark — measure throughput under load.

Tests how many concurrent requests the server can handle at various
concurrency levels. Reports req/s, latency, and error rates.

Usage:
  # Start the server first:
  python3 -m mojollama.server --model model.gguf --port 8080

  # Then run benchmark:
  python3 bench_concurrency.py --url http://127.0.0.1:8080
  python3 bench_concurrency.py --url http://127.0.0.1:8080 --concurrency 4,8,16,32
"""

import json
import time
import argparse
import urllib.request
import concurrent.futures
from statistics import median, stdev


def send_request(url, prompt="What is 2+2? Answer concisely.", max_tokens=50):
    """Send one chat completion request, return timing."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()

    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - t0
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return {
            "ok": True,
            "time": elapsed,
            "len": len(content),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {"ok": False, "time": elapsed, "error": str(e)}


def bench_concurrent(url, n_requests, concurrency, prompt=None):
    """Run benchmark at a given concurrency level."""
    prompts = [
        "What is 2+2?",
        "Explain quantum computing.",
        "Write a haiku about AI.",
        "What is the capital of France?",
        "Define machine learning.",
        "Hello, how are you?",
        "What is the speed of light?",
        "Tell me a joke.",
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(send_request, url, prompt=prompts[i % len(prompts)])
            for i in range(n_requests)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    times = [r["time"] for r in ok]

    if not times:
        return {
            "concurrency": concurrency,
            "requests": n_requests,
            "ok": 0,
            "fail": len(fail),
            "throughput": 0,
            "avg_ms": 0,
            "p50_ms": 0,
            "p95_ms": 0,
            "p99_ms": 0,
            "total_time": 0,
        }

    total_time = max(times)
    sorted_times = sorted(times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]

    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "ok": len(ok),
        "fail": len(fail),
        "throughput": len(ok) / total_time if total_time > 0 else 0,
        "avg_ms": (sum(times) / len(times)) * 1000,
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
        "p99_ms": p99 * 1000,
        "total_time": total_time,
        "errors": [r["error"] for r in fail[:3]] if fail else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="MojoLlama concurrent benchmark"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="Server URL")
    parser.add_argument("--concurrency", "-c",
                        default="1,2,4,8,16",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--requests", "-n", type=int, default=32,
                        help="Requests per concurrency level")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt (optional)")
    args = parser.parse_args()

    concurrency_levels = [int(c) for c in args.concurrency.split(",")]

    # Warmup
    print("Warming up...")
    for _ in range(3):
        send_request(args.url, prompt=args.prompt or "Hi")
    print()

    print(f"Server: {args.url}")
    print(f"Requests per test: {args.requests}")
    print(f"Concurrency levels: {args.concurrency}")
    print()
    print(f"  {'conc':>5} | {'reqs':>5} | {'ok':>5} | {'fail':>5} | "
          f"{'avg_ms':>7} | {'p50_ms':>7} | {'p95_ms':>7} | "
          f"{'req/s':>8} | {'total_s':>8}")
    print("-" * 85)

    for conc in concurrency_levels:
        result = bench_concurrent(
            args.url, args.requests, conc, prompt=args.prompt
        )
        print(f"  {result['concurrency']:>5} | {result['requests']:>5} | "
              f"{result['ok']:>5} | {result['fail']:>5} | "
              f"{result['avg_ms']:>7.1f} | {result['p50_ms']:>7.1f} | "
              f"{result['p95_ms']:>7.1f} | "
              f"{result['throughput']:>8.1f} | {result['total_time']:>8.2f}")

    # Summary
    print()
    print("Tip: For best latency, use low concurrency (1-4).")
    print("     For best throughput, use higher concurrency (8-32).")
    print("     The 'knee' in the latency curve shows optimal concurrency.")


if __name__ == "__main__":
    main()
