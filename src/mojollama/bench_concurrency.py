#!/usr/bin/env python3
"""Concurrent throughput benchmark for MojoLlama server.

Starts server as a subprocess, fires concurrent requests at various
concurrency levels, measures latency and throughput.

Usage:
    OMP_NUM_THREADS=32 python3 -u bench_concurrency.py

Or just:
    python3 -u bench_concurrency.py
"""

import os
import sys
import time
import json
import subprocess
import signal
import socket
import statistics
import concurrent.futures
import urllib.request
import urllib.error

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = '/tmp/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
PORT = 8080
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'server_moe.py')
OMP_THREADS = 32
CONCURRENCY_LEVELS = [1, 2, 4, 8]
NUM_REQUESTS = 10          # total requests per concurrency level
PROMPT = "What is the capital of France?"
MAX_TOKENS = 20
REQUEST_TIMEOUT = 120       # seconds per individual request
SERVER_START_TIMEOUT = 120  # seconds to wait for server to become ready


# ── Helpers ───────────────────────────────────────────────────────────────────

def port_free(port: int) -> bool:
    """Return True if *port* is not in use on all interfaces."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False


def free_port(port: int, timeout: float = 5.0) -> None:
    """Wait up to *timeout* seconds for *port* to become free, then raise."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_free(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Port {port} is still in use after {timeout}s timeout")


def start_server() -> subprocess.Popen:
    """Start MojoLlama server, wait until it's ready, return the Popen handle.

    Readiness is detected by scanning stdout for the log line:
        'Serving on 0.0.0.0:{PORT}'
    """
    if not port_free(PORT):
        raise RuntimeError(
            f"Port {PORT} is already in use — cannot start server"
        )

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS)

    proc = subprocess.Popen(
        ['python3', '-u', SERVER_SCRIPT, MODEL_PATH, str(PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    ready_marker = f"Serving on 0.0.0.0:{PORT}"
    deadline = time.monotonic() + SERVER_START_TIMEOUT

    while time.monotonic() < deadline:
        # Read a line (blocking, but we have a deadline check around the loop)
        line = proc.stdout.readline()
        if line:
            sys.stdout.write(f"  [server] {line}")
            sys.stdout.flush()
            if ready_marker in line:
                # Server is ready — short sleep to let everything settle
                time.sleep(0.5)
                return proc

        # Did the process die?
        if proc.poll() is not None:
            # Drain remaining output for diagnostics
            remaining = proc.stdout.read()
            if remaining:
                sys.stdout.write(f"  [server] {remaining}")
                sys.stdout.flush()
            raise RuntimeError(
                f"Server process exited prematurely (rc={proc.returncode})"
            )

    # Timed out — kill and raise
    kill_server(proc)
    raise RuntimeError(
        f"Server did not become ready within {SERVER_START_TIMEOUT}s"
    )


def kill_server(proc: subprocess.Popen) -> None:
    """Send SIGTERM to *proc*; escalate to SIGKILL after 10 s."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()          # SIGTERM
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()            # SIGKILL
        proc.wait()


def send_request(prompt: str, max_tokens: int) -> tuple | None:
    """Send one completion request, return (latency_ns, tokens_generated).

    Returns *None* on any failure (timeout, HTTP error, connection error, …).
    """
    url = f"http://localhost:{PORT}/v1/completions"
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method='POST',
    )

    t0 = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            resp_body = json.loads(resp.read())
            latency_ns = time.perf_counter_ns() - t0
            usage = resp_body.get('usage', {})
            tokens_gen = usage.get('completion_tokens', 0)
            return (latency_ns, tokens_gen)
    except Exception:
        return None


# ── Benchmark core ────────────────────────────────────────────────────────────

def benchmark_concurrency(
    concurrency_level: int, num_requests: int = 10
) -> tuple[list, float]:
    """Run *num_requests* completions with up to *concurrency_level* in-flight.

    Returns (results_list, wall_clock_seconds).

    Each result is a (latency_ns, tokens_generated) tuple.
    Results only include successful (HTTP 200) responses.
    """
    results: list[tuple[int, int]] = []
    wall_start = time.perf_counter_ns()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency_level
    ) as pool:
        futures = [
            pool.submit(send_request, PROMPT, MAX_TOKENS)
            for _ in range(num_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    wall_ns = time.perf_counter_ns() - wall_start
    return results, wall_ns / 1e9


def percentile(data: list[float], p: float) -> float:
    """Compute the *p*-th percentile of *data* (0–100)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (p / 100.0) * (len(s) - 1)
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] + c * (s[f + 1] - s[f])
    return s[f]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72, flush=True)
    print("  MojoLlama — Concurrent Throughput Benchmark")
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Port:     {PORT}")
    print(f"  Server:   {SERVER_SCRIPT}")
    print(f"  OMP_NUM_THREADS: {OMP_THREADS}")
    print(f"  Prompt:   {PROMPT!r}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Requests per level: {NUM_REQUESTS}")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print("=" * 72, flush=True)

    # Table header
    header = (
        f"  {'Concurrency':>12} | {'p50 lat (ms)':>13} | "
        f"{'p95 lat (ms)':>13} | {'Throughput (tok/s)':>18}"
    )
    print(header, flush=True)
    print("  " + "-" * len(header), flush=True)

    server_proc = None

    for cl_idx, concurrency in enumerate(CONCURRENCY_LEVELS):
        # ── Start a fresh server instance ────────────────────────────────
        # Kill previous instance (if any)
        if server_proc is not None:
            kill_server(server_proc)
            server_proc = None

        # Make sure port is free
        try:
            free_port(PORT, timeout=15)
        except RuntimeError as e:
            print(f"  ERROR: {e}", flush=True)
            sys.exit(1)

        # Start new server
        print(f"\n  >>> Starting server for concurrency={concurrency} ...",
              flush=True)
        try:
            server_proc = start_server()
        except RuntimeError as e:
            print(f"  FAILED to start server: {e}", flush=True)
            sys.exit(1)

        print(f"  >>> Server ready. Running benchmark ...", flush=True)

        # ── Run the benchmark ────────────────────────────────────────────
        try:
            results, wall_sec = benchmark_concurrency(
                concurrency, num_requests=NUM_REQUESTS
            )
        except Exception as e:
            print(f"  Benchmark FAILED: {e}", flush=True)
            results, wall_sec = [], 0.0

        # ── Compute metrics ──────────────────────────────────────────────
        successful = len(results)
        if results:
            latencies_ms = [r[0] / 1_000_000 for r in results]
            total_tokens = sum(r[1] for r in results)
            p50_lat = percentile(latencies_ms, 50)
            p95_lat = percentile(latencies_ms, 95)
            throughput = total_tokens / wall_sec if wall_sec > 0 else 0.0

            print(
                f"  {f'Results:':>12} {successful}/{NUM_REQUESTS} OK, "
                f"total_wall={wall_sec:.2f}s, "
                f"total_tokens={total_tokens}", flush=True
            )
        else:
            p50_lat = 0.0
            p95_lat = 0.0
            throughput = 0.0
            print(f"  {f'Results:':>12} 0/{NUM_REQUESTS} OK — all failed",
                  flush=True)

        # ── Print table row ──────────────────────────────────────────────
        print(
            f"  {concurrency:>12} | {p50_lat:>13.1f} | "
            f"{p95_lat:>13.1f} | {throughput:>18.1f}",
            flush=True
        )

    # ── Cleanup final server ─────────────────────────────────────────────
    if server_proc is not None:
        kill_server(server_proc)
        server_proc = None

    print("\n  Done.", flush=True)


if __name__ == '__main__':
    main()
