"""Concurrency benchmark for MojoLlama API server.

Sends multiple concurrent requests and measures throughput.
Uses stdlib only (urllib, threading, time).
"""
import json
import time
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


SERVER_URL = "http://localhost:8080"


def chat_request(prompt: str, max_tokens: int = 10) -> dict:
    """Send a single chat completion request."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def benchmark_concurrency(num_requests: int = 4, max_workers: int = 4):
    """Run concurrent requests and report throughput."""
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
        "What is 2+2?",
        "Name three colors.",
        "Say hello in French.",
        "What is the speed of light?",
        "Define recursion.",
    ] * 10  # enough prompts

    print(f"\n{'='*60}")
    print(f"Concurrency Benchmark: {num_requests} requests, {max_workers} workers")
    print(f"{'='*60}")

    # Warmup
    print("Warming up...")
    try:
        chat_request("Hello", 2)
    except Exception as e:
        print(f"Warmup failed — is the server running at {SERVER_URL}?")
        print(f"Error: {e}")
        return

    successes = 0
    failures = 0
    total_tokens = 0
    latencies = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            futures.append(pool.submit(chat_request, prompt, 10))

        for f in as_completed(futures):
            try:
                result = f.result()
                successes += 1
                usage = result.get("usage", {})
                total_tokens += usage.get("total_tokens", 0)
            except Exception as e:
                failures += 1
                print(f"  Request failed: {e}")

    elapsed = time.time() - t0

    print(f"\nResults:")
    print(f"  Successes: {successes}")
    print(f"  Failures:  {failures}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Requests/sec: {successes/elapsed:.2f}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Tokens/sec: {total_tokens/elapsed:.2f}")


if __name__ == "__main__":
    benchmark_concurrency(4, 4)
