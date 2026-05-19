#!/usr/bin/env python3
"""Continuous batching load test for server_batch_moe.py"""
import urllib.request, json, time, threading, sys

SERVER = "http://localhost:8081"
results = []
errors = []

def send_request(prompt, idx):
    t0 = time.time()
    data = json.dumps({"prompt": prompt, "max_tokens": 20, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"{SERVER}/v1/completions",
                                 data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        body = json.loads(resp.read())
        elapsed = time.time() - t0
        gen_text = body.get("choices", [{}])[0].get("text", "")
        finish = body.get("choices", [{}])[0].get("finish_reason", "?")
        results.append({"idx": idx, "elapsed": elapsed, "text": gen_text[:40], "finish": finish, "tokens": body.get("usage", {}).get("completion_tokens", 0)})
        print(f"  [{idx}] {elapsed:.1f}s finish={finish} tokens={body.get('usage',{}).get('completion_tokens',0)} '{gen_text[:40]}'")
    except Exception as e:
        errors.append({"idx": idx, "error": str(e)})
        print(f"  [{idx}] ERROR: {e}")

print("=" * 60)
print("ITEM 6: Continuous batching load test")
print("=" * 60)

# Test 1: 5 concurrent requests
print("\n--- Test 1: 5 concurrent requests ---")
threads = []
prompts = [
    "The capital of France is",
    "The theory of relativity states that",
    "Machine learning is a field of",
    "Python is a programming language that",
    "The mitochondria is the powerhouse of",
]
t0 = time.time()
for i, prompt in enumerate(prompts):
    t = threading.Thread(target=send_request, args=(prompt, i))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
t_total = time.time() - t0

print(f"\nResults: {len(results)}/{len(prompts)} OK, {len(errors)} errors")
if results:
    avg_lat = sum(r["elapsed"] for r in results) / len(results)
    max_lat = max(r["elapsed"] for r in results)
    min_lat = min(r["elapsed"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    print(f"  Avg latency: {avg_lat*1000:.0f} ms")
    print(f"  Min latency: {min_lat*1000:.0f} ms")
    print(f"  Max latency: {max_lat*1000:.0f} ms")
    print(f"  Total generation tokens: {total_tokens}")
    print(f"  Total wall time: {t_total:.1f}s")
if errors:
    for e in errors:
        print(f"  Error in request {e['idx']}: {e['error']}")

# Test 2: Streaming test
print("\n--- Test 2: Streaming test ---")
try:
    data = json.dumps({"prompt": "Once upon a time,", "max_tokens": 10, "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(f"{SERVER}/v1/completions",
                                 data=data,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=30)
    chunks = []
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data:"):
            chunks.append(line)
    print(f"  Received {len(chunks)} SSE chunks")
    for c in chunks[:5]:
        print(f"    {c[:80]}")
    if len(chunks) > 5:
        print(f"    ... ({len(chunks)-5} more)")
    print(f"  Streaming: OK ({len(chunks)} chunks)")
except Exception as e:
    print(f"  Streaming ERROR: {e}")

print(f"\nContinuous batching: {len(results)}/{len(prompts)} requests OK, avg latency {avg_lat*1000:.0f} ms, max concurrent {len(prompts)}")
