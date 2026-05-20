#!/usr/bin/env python3
"""GPT-OSS-20B concurrent throughput: MojoLlama engine vs llama.cpp server."""
import sys, os, time, json, subprocess, urllib.request, signal
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "/tmp/models/gpt-oss-20b-Q4_K_M.gguf"
LLAMA_PORT = 8092
PROMPT = "What is the capital of France?"
MAX_TOKENS = 20
N_USERS = 10
N_REQUESTS = 5  # requests per user

def send_request(url, data, timeout=120):
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        elapsed = time.perf_counter() - t0
        tok = result.get('usage', {}).get('completion_tokens', MAX_TOKENS)
        if not tok: tok = MAX_TOKENS
        return elapsed, tok, True
    except Exception as e:
        return 0, 0, False

def bench_llama_cpp_concurrent():
    """Benchmark llama.cpp server with 10 concurrent users, each sending 5 requests."""
    print(f"\n{'='*60}")
    print(f"  llama.cpp: 10 users × {N_REQUESTS} requests each (GPT-OSS)")
    print(f"{'='*60}")
    
    cmd = ["/tmp/llama.cpp/build/bin/llama-server", "-m", MODEL,
           "-c", "4096", "-t", "32", "--port", str(LLAMA_PORT),
           "--cont-batching", "--no-webui", "--mlock"]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    
    url = f"http://localhost:{LLAMA_PORT}"
    for attempt in range(60):
        try:
            r = urllib.request.urlopen(f"{url}/health", timeout=2)
            if r.status == 200: break
        except: pass
        time.sleep(2)
    else:
        proc.kill(); return None
    
    total_latencies = []
    total_tokens = 0
    wall_start = time.perf_counter()
    
    def user_bench(_):
        user_lats = []
        user_toks = 0
        for _ in range(N_REQUESTS):
            data = {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0}
            lat, tok, ok = send_request(f"{url}/v1/completions", data)
            if ok:
                user_lats.append(lat)
                user_toks += tok
        return user_lats, user_toks
    
    with ThreadPoolExecutor(max_workers=N_USERS) as pool:
        futures = [pool.submit(user_bench, i) for i in range(N_USERS)]
        for f in as_completed(futures):
            lats, toks = f.result()
            total_latencies.extend(lats)
            total_tokens += toks
    
    wall_elapsed = time.perf_counter() - wall_start
    proc.terminate()
    try: proc.wait(timeout=5)
    except: proc.kill()
    
    total_latencies.sort()
    n = len(total_latencies)
    p50 = total_latencies[n//2] * 1000
    p95 = total_latencies[int(n*0.95)] * 1000 if n > 1 else total_latencies[-1] * 1000
    throughput = total_tokens / wall_elapsed
    
    print(f"  Results:")
    print(f"    Requests:   {len(total_latencies)}/{N_USERS*N_REQUESTS}")
    print(f"    Wall time:  {wall_elapsed*1000:.0f} ms")
    print(f"    p50:        {p50:.0f} ms  p95: {p95:.0f} ms")
    print(f"    Tokens:     {total_tokens}")
    print(f"    Throughput: {throughput:.1f} tok/s")
    return {"p50": p50, "p95": p95, "throughput": throughput, "requests": len(total_latencies)}


def bench_mojollama_direct():
    """MojoLlama: direct engine benchmark simulating batch concurrency.
    
    MojoLlama's batch_forward supports B=1..N. For concurrent users,
    we run them sequentially (since the engine is single-threaded for
    decode). We measure how fast N individual sequential requests finish.
    This gives us the throughput under a simple round-robin load.
    """
    print(f"\n{'='*60}")
    print(f"  MojoLlama: {N_USERS} sequential requests (GPT-OSS)")
    print(f"{'='*60}")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernels'))
    from turbo_engine_v7_moe import TurboEngineV7MoE
    
    e = TurboEngineV7MoE(MODEL, 32)
    print(f"  Engine loaded: {e.n_layers}L, {e.n_experts} experts")
    
    total_latencies = []
    total_tokens = 0
    wall_start = time.perf_counter()
    import numpy as np
    
    for user in range(N_USERS):
        for req in range(N_REQUESTS):
            t0 = time.perf_counter()
            tokens = np.array([[1]], dtype=np.int32)  # BOS
            tok_count = 0
            for _ in range(MAX_TOKENS):
                logits = e.forward(tokens)
                token = int(np.argmax(logits))
                tok_count += 1
                tokens = np.array([[token]], dtype=np.int32)
                if token == 2:  # EOS
                    break
            elapsed = time.perf_counter() - t0
            total_latencies.append(elapsed)
            total_tokens += tok_count
    
    wall_elapsed = time.perf_counter() - wall_start
    
    total_latencies.sort()
    n = len(total_latencies)
    p50 = total_latencies[n//2] * 1000
    p95 = total_latencies[int(n*0.95)] * 1000 if n > 1 else total_latencies[-1] * 1000
    throughput = total_tokens / wall_elapsed
    
    print(f"  Results:")
    print(f"    Requests:   {len(total_latencies)}/{N_USERS*N_REQUESTS}")
    print(f"    Wall time:  {wall_elapsed*1000:.0f} ms")
    print(f"    p50:        {p50:.0f} ms  p95: {p95:.0f} ms")
    print(f"    Tokens:     {total_tokens}")
    print(f"    Throughput: {throughput:.1f} tok/s")
    return {"p50": p50, "p95": p95, "throughput": throughput, "requests": len(total_latencies)}


# Run benchmarks
llama = bench_llama_cpp_concurrent()
mojo = bench_mojollama_direct()

print(f"\n{'='*60}")
print(f"  COMPARISON: GPT-OSS-20B — {N_USERS} users × {N_REQUESTS} requests")
print(f"{'='*60}")
print(f"  {'':>20} {'llama.cpp':>14} {'MojoLlama':>14} {'Ratio':>10}")
print(f"  {'─'*20} {'─'*14} {'─'*14} {'─'*10}")

if llama and mojo:
    ratio = mojo['throughput'] / llama['throughput']
    print(f"  {'Throughput':>20} {llama['throughput']:>14.1f} {mojo['throughput']:>14.1f} {ratio:>9.2f}x")
    print(f"  {'p50':>20} {llama['p50']:>14.0f}ms {mojo['p50']:>14.0f}ms")
    print(f"  {'p95':>20} {llama['p95']:>14.0f}ms {mojo['p95']:>14.0f}ms")
    if ratio > 1:
        print(f"\n  ✅ MojoLlama is {ratio:.2f}x faster than llama.cpp")
    else:
        print(f"\n  ⚠️  llama.cpp is {1/ratio:.2f}x faster than MojoLlama")
else:
    if not llama: print("  llama.cpp FAILED")
    if not mojo: print("  MojoLlama FAILED")
print(f"{'='*60}")
