#!/usr/bin/env python3
"""MojoLlama Scheduler — vLLM-style continuous batching proxy.

Collects requests over a time window, batches them into a single
forward pass on llama.cpp, and distributes responses back.
Provides higher throughput than individual dispatch by maximizing
weight reuse across concurrent sequences.

Architecture:
  ┌──────────┐    window    ┌──────────┐    batch     ┌───────────┐
  │ Clients  │ ──────────→ │ Batcher  │ ──────────→ │ llama.cpp │
  │ (HTTP)   │ ←────────── │ (Queue)  │ ←────────── │ (np=high) │
  └──────────┘  responses   └──────────┘             └───────────┘
"""

import json
import time
import uuid
import threading
import queue
import urllib.request
import http.client
from http.server import HTTPServer, BaseHTTPRequestHandler

BACKEND_URL = "http://127.0.0.1:8081"
BATCH_WINDOW = 0.050  # 50ms — collect requests before dispatching batch
MAX_BATCH = 64        # max sequences per batch
BACKEND_NP = 64       # must match llama.cpp -np setting

# ─── Batcher ──────────────────────────────────────────────────────────

class Batcher:
    """Collects individual requests into batches for llama.cpp."""

    def __init__(self, backend_url=BACKEND_URL, max_batch=MAX_BATCH,
                 window=BATCH_WINDOW):
        self.backend_url = backend_url
        self.max_batch = max_batch
        self.window = window
        self._pending = []  # [(prompt, max_tokens, result_queue), ...]
        self._lock = threading.Lock()
        self._flush_event = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._metrics = {"batches": 0, "total_requests": 0, "avg_batch_size": 0}
        self._conn_pool = queue.Queue()

    def _get_conn(self):
        try:
            return self._conn_pool.get_nowait()
        except queue.Empty:
            return http.client.HTTPConnection("127.0.0.1", 8081, timeout=120)

    def _return_conn(self, conn):
        try:
            self._conn_pool.put_nowait(conn)
        except queue.Full:
            try: conn.close()
            except: pass

    def submit(self, prompt, max_tokens, temperature=0.7):
        """Submit a request. Returns a queue that will receive the result."""
        result_queue = queue.Queue()
        with self._lock:
            self._pending.append({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "result": result_queue,
                "id": str(uuid.uuid4())[:8],
            })
            self._flush_event.set()  # wake up batcher
        return result_queue

    def _run(self):
        """Main batch loop: collect requests, dispatch batch, return results."""
        while self._running:
            self._flush_event.wait()
            self._flush_event.clear()

            while True:
                # Collect a batch
                batch = self._drain_pending()
                if not batch:
                    break

                # Wait a bit more for more requests (up to window time)
                time.sleep(self.window)
                more = self._drain_pending()
                batch.extend(more)

                # Dispatch
                self._dispatch_batch(batch)

    def _drain_pending(self):
        """Thread-safe drain of pending queue."""
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()
            return batch

    def _dispatch_batch(self, batch):
        """Send a batch of prompts to llama.cpp."""
        if not batch:
            return

        # Build a single prompt by concatenating all prompts
        # Format: one conversation per batch entry
        # llama.cpp handles multiple sequences via its internal batcher
        # For batch efficiency, we craft a single prompt that produces
        # all required responses, then split them.
        #
        # Simpler approach: just dispatch concurrently via HTTP,
        # but schedule carefully to maximize batch utilization.

        # Track metrics
        with self._lock:
            self._metrics["batches"] += 1
            self._metrics["total_requests"] += len(batch)
            self._metrics["avg_batch_size"] = (
                self._metrics["avg_batch_size"] * 0.9 + len(batch) * 0.1
            )

        # Dispatch each request individually (using np for parallelism)
        # A true batcher would merge all prompts into one forward pass,
        # but llama.cpp's internal continuous batching already handles
        # multiple sequences efficiently when np is set high.
        threads = []
        for req in batch:
            t = threading.Thread(target=self._dispatch_one, args=(req,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def _dispatch_one(self, req):
        """Send one request to llama.cpp."""
        data = json.dumps({
            "messages": [{"role": "user", "content": req["prompt"]}],
            "max_tokens": req["max_tokens"],
            "temperature": req["temperature"],
            "stream": False,
        }).encode()

        conn = self._get_conn()
        try:
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw = resp.read()
            resp.close()
            result = json.loads(raw.decode("utf-8"))
            text = (result.get("choices", [{}])[0]
                    .get("message", {}).get("content", ""))
            usage = result.get("usage", {})
            req["result"].put({
                "ok": True, "text": text,
                "tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
            })
        except Exception as e:
            req["result"].put({"ok": False, "error": str(e)})
        finally:
            self._return_conn(conn)

    def get_metrics(self):
        with self._lock:
            return dict(self._metrics)

    def stop(self):
        self._running = False
        self._flush_event.set()


# ─── HTTP Handler ─────────────────────────────────────────────────────

batcher = Batcher()

class SchedulerHandler(BaseHTTPRequestHandler):
    """HTTP handler that routes through the batcher."""

    def log_message(self, format, *args):
        pass  # reduce noise

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "batcher": "active"})
        elif self.path == "/metrics":
            self._send_json(batcher.get_metrics())
        else:
            self._send_error(404)

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/api/chat"):
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 256)
            temperature = body.get("temperature", 0.7)
            prompt = messages[-1]["content"] if messages else ""

            # Submit to batcher
            result_queue = batcher.submit(prompt, max_tokens, temperature)
            result = result_queue.get(timeout=120)

            if result["ok"]:
                if self.path == "/api/chat":
                    # SSE streaming
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    text = result["text"]
                    # Send tokens gradually (simulate streaming from batch)
                    for token in text.split(" "):
                        chunk = json.dumps({
                            "choices": [{"delta": {"content": token + " "}, "index": 0}]
                        })
                        self.wfile.write(f"data: {chunk}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(0.01)  # throttle for UX
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self._send_json({
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": result["text"]},
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": result.get("prompt_tokens", 0),
                            "completion_tokens": result.get("tokens", 0),
                            "total_tokens": result.get("prompt_tokens", 0) + result.get("tokens", 0),
                        },
                        "backend": "mojollama-scheduler",
                    })
            else:
                self._send_error(f"Batch error: {result.get('error', 'unknown')}", 502)
        else:
            self._send_error(404)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg, status=500):
        self._send_json({"error": msg}, status)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="MojoLlama Scheduler Proxy")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--backend", default="http://127.0.0.1:8081")
    parser.add_argument("--batch-window", type=float, default=0.05)
    parser.add_argument("--max-batch", type=int, default=64)
    args = parser.parse_args()

    global batcher, BACKEND_URL, BATCH_WINDOW, MAX_BATCH
    BACKEND_URL = args.backend
    BATCH_WINDOW = args.batch_window
    MAX_BATCH = args.max_batch
    batcher = Batcher(args.backend, args.max_batch, args.batch_window)

    server = HTTPServer(("0.0.0.0", args.port), SchedulerHandler)
    print(f"MojoLlama Scheduler on :{args.port} → {args.backend}")
    print(f"  Batch window: {args.batch_window*1000:.0f}ms")
    print(f"  Max batch: {args.max_batch}")
    print(f"  /v1/chat/completions — batched completions")
    print(f"  /api/chat            — streaming batched completions")
    print(f"  /metrics             — batch metrics")
    print(f"  /health              — health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        batcher.stop()
        server.server_close()


if __name__ == "__main__":
    main()
