#!/usr/bin/env python3
"""MojoLlama server — unified API server with SSE streaming, studio API,
concurrent handling, and auto-backend selection.

Starts an OpenAI-compatible API server that routes inference through:
  - MAX GPU (if GPU available)
  - llama.cpp (if CPU only)
  - MAX CPU (fallback)

SSE streaming, CORS, and studio endpoints are all built in.

Usage:
    python3 server.py
    python3 server.py --model /path/to/model.gguf --port 8080
"""

import os
import sys
import json
import time
import uuid
import argparse
import threading
import subprocess
import re
import queue
import concurrent.futures
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from io import BytesIO

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mojollama.backends import AutoBackend

# ─── Globals ────────────────────────────────────────────────────────────

backend = None
prompts_served = 0
start_time = time.time()

# Export job tracking: job_id -> {"status": str, "log": str, "thread": Thread}
_export_jobs = {}
_export_jobs_lock = threading.Lock()

# Training metrics subscription: list of queues
_training_queues = []
_training_queues_lock = threading.Lock()

_metrics_lock = threading.Lock()
_request_times = []  # rolling window of request latencies (seconds)
_metrics_max_samples = 1000

def _record_latency(t0):
    """Record request latency for metrics."""
    elapsed = time.time() - t0
    with _metrics_lock:
        _request_times.append(elapsed)
        if len(_request_times) > _metrics_max_samples:
            _request_times.pop(0)

LLAMA_SERVER_PATH = "/tmp/llama.cpp/build/bin/llama-server"
CONVERTER_PATH = "/tmp/llama.cpp/convert_hf_to_gguf.py"
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
WORK_DIR = BASE_DIR  # /onedev-workspace/work


# ─── Threaded HTTP Server ──────────────────────────────────────────────

class ThreadPoolHTTPServer(HTTPServer):
    """HTTP server with bounded thread pool for concurrent requests.
    
    Instead of spawning an unbounded thread per request (ThreadingMixIn),
    uses a ThreadPoolExecutor with a configurable max_workers limit.
    When the queue is full, new connections get 503 Service Unavailable.
    """
    
    def __init__(self, server_address, RequestHandlerClass,
                 max_workers=32, queue_size=64):
        self.allow_reuse_address = True
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        self._queue_size = queue_size
        self._active_requests = threading.BoundedSemaphore(queue_size)
        HTTPServer.__init__(self, server_address, RequestHandlerClass)
    
    def process_request(self, request, client_address):
        """Submit request to thread pool, or reject if queue is full."""
        if not self._active_requests.acquire(blocking=False):
            # Queue full — send 503
            try:
                request.sendall(b'HTTP/1.1 503 Service Unavailable\r\n'
                               b'Content-Length: 0\r\nConnection: close\r\n\r\n')
            except: pass
            request.close()
            return
        
        self._executor.submit(self._handle_request,
                              request, client_address)
    
    def _handle_request(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self._active_requests.release()
            self.shutdown_request(request)
    
    def server_close(self):
        self._executor.shutdown(wait=True)
        HTTPServer.server_close(self)


# ─── Request Handler ───────────────────────────────────────────────────

class MojoLlamaHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible API handler with SSE streaming and studio API."""

    # ── Helpers ─────────────────────────────────────────────────────

    def log_message(self, fmt, *args):
        sys.stderr.write(
            f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]} {args[2]}\n"
        )

    def _parse_path(self):
        """Parse self.path into path, query dict."""
        parsed = urlparse(self.path)
        return parsed.path, parse_qs(parsed.query)

    def _read_body(self):
        """Read and parse JSON request body."""
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            return {}
        raw = self.rfile.read(content_len)
        if not raw:
            return {}
        return json.loads(raw)

    def _send_json(self, data, status=200):
        """Send a JSON response."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message, status=400):
        """Send a JSON error response."""
        self._send_json({"error": message}, status)

    def _set_cors(self):
        """Add CORS headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization")

    def _sse_send(self, data_obj):
        """Send a single SSE message (data: json\\n\\n)."""
        msg = f"data: {json.dumps(data_obj)}\n\n".encode()
        self.wfile.write(msg)
        self.wfile.flush()

    def _sse_end(self):
        """Send the [DONE] signal to terminate the SSE stream."""
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _build_tools_prompt(self, tools, tool_choice="auto"):
        """Build a system-level instruction from OpenAI tool definitions.

        Injects available functions into the conversation so the model
        can call them. Returns a system message dict or empty dict.
        """
        if not tools:
            return {}

        # Build a function list description
        func_descs = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool.get("function", {})
                params = fn.get("parameters", {})
                props_desc = ", ".join(params.get("properties", {}).keys())
                required = params.get("required", [])
                func_descs.append(
                    f"- {fn.get('name')}: {fn.get('description', '')} "
                    f"(parameters: {props_desc}, required: {', '.join(required)})"
                )

        instruction = (
            "You have access to the following functions. Use them when the user's "
            "request requires calling an external tool or retrieving information.\n\n"
            "Available functions:\n"
            + "\n".join(func_descs) +
            "\n\n"
            "To call a function, respond with ONLY a valid JSON object in this exact "
            'format, with no other text: {"name": "function_name", "arguments": {"arg1": "value1"}}\n'
            'If you need to call a function, respond with ONLY that JSON. Do not include '
            "any other explanation or text. If you don't need to call a function, "
            "respond normally with your answer."
        )

        if tool_choice == "none":
            return {}  # Skip function calling entirely
        elif isinstance(tool_choice, dict):
            # Force a specific function: {"type":"function","function":{"name":"..."}}
            fn_name = tool_choice.get("function", {}).get("name", "")
            instruction += (
                f"\n\nIMPORTANT: You MUST use the '{fn_name}' function for this request. "
                f"Respond with ONLY the function call JSON."
            )
            return {"role": "system", "content": instruction, "_force_tool": fn_name}
        else:
            return {"role": "system", "content": instruction}

    @staticmethod
    def _parse_tool_call_response(text):
        """Parse a model response for function call JSON.

        Returns (tool_name, args_dict) if a function call is detected,
        or (None, None) if it's a normal response.
        """
        text = text.strip()
        # Try to parse the entire response as JSON
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                if "name" in obj and "arguments" in obj:
                    args = obj["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    return obj["name"], args
            except (json.JSONDecodeError, TypeError):
                pass
        # Try to find JSON embedded in text
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]*"\s*,[^{}]*"arguments"\s*:', text, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                if "name" in obj and "arguments" in obj:
                    args = obj["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    return obj["name"], args
            except (json.JSONDecodeError, TypeError):
                pass
        return None, None

    # ── HTTP Methods ────────────────────────────────────────────────

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self._set_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ── GET ─────────────────────────────────────────────────────────

    def do_GET(self):
        path, query = self._parse_path()

        # ── Existing endpoints ──────────────────────────────────
        if path == "/v1/models":
            models_list = []
            for f in sorted(Path(WORK_DIR).glob("*.gguf")):
                models_list.append({
                    "id": f.stem,
                    "object": "model",
                    "created": int(f.stat().st_mtime),
                    "owned_by": "mojollama",
                })
            if not models_list:
                models_list.append({
                    "id": "mojollama-llama-3.2-1b",
                    "object": "model",
                    "created": int(start_time),
                    "owned_by": "mojollama",
                })
            self._send_json({"object": "list", "data": models_list})

        elif path == "/health":
            self._send_json({
                "status": "ok",
                "backend": backend.info["active"] if hasattr(backend, 'info') else "unknown",
                "uptime": f"{time.time() - start_time:.0f}s",
            })

        elif path == "/backend":
            self._send_json(backend.info if hasattr(backend, 'info') else {"active": "unknown"})

        # ── Studio API: /api/backend ─────────────────────────────
        elif path == "/api/backend":
            info = backend.info if hasattr(backend, 'info') else {"active": "unknown"}
            info["prompts_served"] = prompts_served
            info["uptime"] = f"{time.time() - start_time:.0f}s"
            self._send_json(info)

        # ── Studio API: /api/models ──────────────────────────────
        elif path == "/api/models":
            models = []
            for f in sorted(Path(WORK_DIR).glob("*.gguf")):
                size_bytes = f.stat().st_size
                models.append({
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": size_bytes,
                    "size_hr": self._format_size(size_bytes),
                    "modified": f.stat().st_mtime,
                })
            self._send_json({"models": models, "count": len(models)})

        # ── Studio API: /api/dataset ─────────────────────────────
        elif path == "/api/dataset":
            datasets = []
            for f in sorted(Path(WORK_DIR).glob("*.jsonl")) + sorted(Path(WORK_DIR).glob("*.json")):
                datasets.append({
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                })
            self._send_json({"datasets": datasets, "count": len(datasets)})

        # ── Studio API: export status ──────────────────────────
        elif path.startswith("/api/export/") and path.endswith("/status"):
            job_id = path.split("/")[3]
            with _export_jobs_lock:
                job = _export_jobs.get(job_id)
            if job is None:
                self._send_error(f"Job {job_id} not found", 404)
                return
            self._send_json({
                "job_id": job_id,
                "status": job["status"],
                "log": job.get("log", ""),
            })

        elif path.startswith("/api/export/"):
            # GET /api/export/{job_id} (without /status suffix)
            job_id = path.split("/")[3]
            with _export_jobs_lock:
                job = _export_jobs.get(job_id)
            if job is None:
                self._send_error(f"Job {job_id} not found", 404)
                return
            self._send_json({
                "job_id": job_id,
                "status": job["status"],
                "log": job.get("log", ""),
            })

        # ── Studio API: training metrics SSE ───────────────────
        elif path == "/api/train/metrics":
            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = queue.Queue()
            with _training_queues_lock:
                _training_queues.append(q)

            try:
                # Keep connection open, sending heartbeats
                while True:
                    try:
                        metrics = q.get(timeout=5)
                        self._sse_send(metrics)
                    except queue.Empty:
                        # Send a heartbeat comment to keep connection alive
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with _training_queues_lock:
                    if q in _training_queues:
                        _training_queues.remove(q)

        # ── Metrics endpoint ──────────────────────────────────
        elif path == "/metrics":
            with _metrics_lock:
                n = len(_request_times)
                avg = sum(_request_times) / n if n else 0
                p95 = sorted(_request_times)[int(n * 0.95)] if n > 10 else 0
            self._send_json({
                "prompts_served": prompts_served,
                "uptime_seconds": time.time() - start_time,
                "requests_total": prompts_served,
                "avg_latency_seconds": round(avg, 3),
                "p95_latency_seconds": round(p95, 3),
                "samples": n,
                "backend": backend.info["active"] if hasattr(backend, 'info') else "unknown",
                "connection_pool_size": _llama_pool_size,
                "pool_available": _llama_conn_pool.qsize(),
            })

        # ── Model info endpoint ───────────────────────────────
        elif path == "/api/model-info":
            model_path = query.get("path", [None])[0] or ""
            if not model_path or not os.path.exists(model_path):
                self._send_error("No model path specified or file not found", 400)
                return
            try:
                from mojollama.quantizer import get_info as _gi
                info = _gi(model_path)
                self._send_json({
                    "status": "ok",
                    "path": info.get("path"),
                    "file_size_human": info.get("file_size_human"),
                    "file_size_bytes": info.get("file_size_bytes"),
                    "tensor_count": info.get("tensor_count"),
                    "total_parameters": info.get("total_parameters"),
                    "primary_quantization": info.get("primary_quantization"),
                    "metadata": info.get("metadata"),
                    "tensor_type_counts": info.get("tensor_type_counts"),
                })
            except Exception as e:
                self._send_error(str(e), 500)

        # ── Serve static UI files ─────────────────────────────
        elif path in ("/", "/index.html", "/studio.html", "/chat.html"):
            www_dir = Path(__file__).resolve().parent.parent.parent / "www"
            # Fallback: relative to cwd (for build containers)
            if not www_dir.exists():
                www_dir = Path.cwd() / "www"
            filename = "index.html" if path == "/" else path.lstrip("/")
            filepath = www_dir / filename
            if filepath.exists():
                content = filepath.read_bytes()
                ext = filename.rsplit(".", 1)[-1] if "." in filename else "html"
                mime = {"html": "text/html", "js": "application/javascript",
                        "css": "text/css", "png": "image/png", "svg": "image/svg+xml"}.get(ext, "text/plain")
                self.send_response(200)
                self._set_cors()
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_error("not found", 404)

# ── Not found ──────────────────────────────────────────────────────────────
        else:
            self._send_error("not found", 404)

    # ── POST ────────────────────────────────────────────────────────

    def do_POST(self):
        global prompts_served
        path, query = self._parse_path()
        body = self._read_body()

        # ── Existing: /v1/completions ─────────────────────────────
        if path in ("/v1/completions", "/completion"):
            prompt = body.get("prompt", "")
            max_tokens = body.get("max_tokens", body.get("n_predict", 128))
            temperature = body.get("temperature", 0.7)
            stream = body.get("stream", False)

            t0 = time.time()

            if stream:
                # Streaming response
                return self._handle_streaming_completion(prompt, max_tokens, temperature)

            result = backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            elapsed = time.time() - t0
            prompts_served += 1
            _record_latency(t0)

            self._send_json({
                "id": f"cmpl-{prompts_served}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": "mojollama-llama-3.2-1b",
                "choices": [{"text": result.get("text", ""), "index": 0, "finish_reason": "stop"}],
                "usage": {
                    "completion_tokens": result.get("tokens", 0),
                    "total_tokens": result.get("tokens", 0),
                },
                "backend": result.get("backend"),
                "timings": {"total": f"{elapsed:.2f}s"},
            })

        # ── Updated: /v1/chat/completions (with tool calling + latency tracking) ──
        elif path == "/v1/chat/completions":
            messages = list(body.get("messages", []))
            max_tokens = body.get("max_tokens", 256)
            temperature = body.get("temperature", 0.7)
            stream = body.get("stream", False)
            tools = body.get("tools", [])
            tool_choice = body.get("tool_choice", "auto")

            t0 = time.time()

            # Handle tool calling: inject function definitions into the conversation
            if tools:
                tool_msg = self._build_tools_prompt(tools, tool_choice)
                if tool_msg:
                    # Insert the tool system message at the beginning
                    messages.insert(0, tool_msg)

            if stream:
                # SSE streaming response
                result = self._handle_streaming_chat_completion(
                    messages, max_tokens, temperature
                )
                _record_latency(t0)
                return result

            # Non-streaming: use backend.chat()
            try:
                result = backend.chat(messages, max_tokens=max_tokens, temperature=temperature)
                prompts_served += 1

                # Build OpenAI-format response
                response = {
                    "id": f"chatcmpl-{prompts_served}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "mojollama",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": result.get("text", "")},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "completion_tokens": result.get("tokens", 0),
                        "total_tokens": result.get("tokens", 0),
                    },
                    "backend": result.get("backend"),
                }

                # Check if tools were provided — try to parse a function call
                if tools:
                    content = response["choices"][0]["message"]["content"]
                    fn_name, fn_args = self._parse_tool_call_response(content)
                    if fn_name and fn_args:
                        response["choices"][0]["message"]["content"] = None
                        response["choices"][0]["message"]["tool_calls"] = [{
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": json.dumps(fn_args),
                            },
                        }]
                        response["choices"][0]["finish_reason"] = "tool_calls"

                self._send_json(response)
            except Exception as e:
                self._send_error(f"Chat completion failed: {e}", 502)
            finally:
                _record_latency(t0)

        # ── Studio API: /api/chat (streaming for web UI) ─────────
        elif path == "/api/chat":
            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 256)
            temperature = body.get("temperature", 0.7)

            return self._handle_streaming_chat_completion(
                messages, max_tokens, temperature
            )

        # ── Studio API: /api/export ──────────────────────────────
        elif path == "/api/export":
            model = body.get("model", "")
            outtype = body.get("outtype", "q4_0")
            outfile = body.get("outfile", "")

            if not model:
                self._send_error("'model' field is required")
                return

            if not outfile:
                model_name = model.split("/")[-1] if "/" in model else model
                outfile = f"{model_name}-{outtype}.gguf"

            job_id = str(uuid.uuid4())[:8]

            with _export_jobs_lock:
                _export_jobs[job_id] = {
                    "status": "running",
                    "log": f"Starting export: {model} -> {outtype} -> {outfile}\n",
                    "thread": None,
                }

            def _run_export(jid, hf_model, ot, of):
                log_lines = []
                def log(msg):
                    log_lines.append(msg)
                    with _export_jobs_lock:
                        job = _export_jobs.get(jid)
                        if job:
                            job["log"] = "\n".join(log_lines)

                log(f"Model: {hf_model}")
                log(f"Out type: {ot}")
                log(f"Out file: {of}")
                log(f"Converter: {CONVERTER_PATH}")
                log("")

                if not os.path.exists(CONVERTER_PATH):
                    log(f"ERROR: Converter not found at {CONVERTER_PATH}")
                    with _export_jobs_lock:
                        if jid in _export_jobs:
                            _export_jobs[jid]["status"] = "error"
                    return

                try:
                    cmd = [
                        sys.executable, CONVERTER_PATH,
                        hf_model,
                        "--outtype", ot,
                        "--outfile", of,
                    ]
                    log(f"Running: {' '.join(cmd)}")

                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )

                    for line in proc.stdout:
                        line = line.rstrip()
                        log(line)

                    proc.wait()

                    if proc.returncode == 0:
                        log(f"\n✅ Export complete: {of}")
                        if os.path.exists(of):
                            size_mb = os.path.getsize(of) / 1024 ** 2
                            log(f"   Size: {size_mb:.0f} MB")
                        with _export_jobs_lock:
                            if jid in _export_jobs:
                                _export_jobs[jid]["status"] = "done"
                    else:
                        log(f"\n❌ Export failed (exit code {proc.returncode})")
                        with _export_jobs_lock:
                            if jid in _export_jobs:
                                _export_jobs[jid]["status"] = "error"

                except Exception as e:
                    log(f"ERROR: {e}")
                    with _export_jobs_lock:
                        if jid in _export_jobs:
                            _export_jobs[jid]["status"] = "error"

            thread = threading.Thread(
                target=_run_export,
                args=(job_id, model, outtype, outfile),
                daemon=True,
            )
            thread.start()

            with _export_jobs_lock:
                _export_jobs[job_id]["thread"] = thread

            self._send_json({
                "job_id": job_id,
                "status": "running",
                "message": f"Export job {job_id} started",
            })

        # ── Quantizer API: /api/quantize ───────────────────────────
        elif path == "/api/quantize":
            input_path = body.get("input", "")
            quant_type = body.get("type", "q4_k_m")
            output_path = body.get("output", "")

            if not input_path or not os.path.exists(input_path):
                self._send_error(f"input file not found: {input_path}", 400)
                return

            try:
                from mojollama.quantizer import quantize as _q
                result = _q(input_path, quant_type=quant_type, output_path=output_path or None)
                self._send_json({
                    "status": "ok",
                    "output_path": result.get("output_path"),
                    "quant_type": result.get("quant_type"),
                    "input_size_bytes": result.get("input_size_bytes"),
                    "output_size_bytes": result.get("output_size_bytes"),
                    "compression_ratio": result.get("compression_ratio"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                })
            except Exception as e:
                self._send_error(str(e), 500)

        # ── Not found ─────────────────────────────────────────
        else:
            self._send_error("not found", 404)

    # ── Streaming Helpers ─────────────────────────────────────────

    def _handle_streaming_chat_completion(self, messages, max_tokens, temperature):
        """Handle SSE streaming for chat completions using backend.chat_stream()."""
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            for chunk in backend.chat_stream(messages, max_tokens=max_tokens,
                                              temperature=temperature):
                self._sse_send(chunk)
        except Exception as e:
            try:
                self._sse_send({
                    "choices": [{"delta": {"content": f"\n[Error: {e}]"}, "index": 0}]
                })
            except (BrokenPipeError, ConnectionResetError):
                pass

        try:
            self._sse_end()
            global prompts_served
            prompts_served += 1
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_streaming_completion(self, prompt, max_tokens, temperature):
        """Handle SSE streaming for text completions using backend.generate_stream()."""
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            for chunk in backend.generate_stream(prompt, max_tokens=max_tokens,
                                                  temperature=temperature):
                self._sse_send(chunk)
        except Exception as e:
            try:
                self._sse_send({
                    "choices": [{"text": f"\n[Error: {e}]", "index": 0}]
                })
            except (BrokenPipeError, ConnectionResetError):
                pass

        try:
            self._sse_end()
            global prompts_served
            prompts_served += 1
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _messages_to_prompt(self, messages):
        """Convert OpenAI-format messages to a simple prompt string.

        For chat models this sends the whole conversation history.
        llama.cpp's /completion endpoint handles chat templates internally,
        but we format as a simple conversation for the proxy.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    @staticmethod
    def _format_size(size_bytes):
        """Format bytes into human-readable size."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.0f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / 1024 ** 2:.0f} MB"
        else:
            return f"{size_bytes / 1024 ** 3:.2f} GB"


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    global backend

    parser = argparse.ArgumentParser(description="MojoLlama inference server")
    parser.add_argument("--model", help="Path to GGUF model file", default="")
    parser.add_argument("--port", type=int, help="HTTP server port", default=8080)
    parser.add_argument("--llama-port", type=int, help="llama.cpp backend port", default=8081)
    parser.add_argument("--max-workers", type=int, help="Max concurrent requests", default=32)
    parser.add_argument("--queue-size", type=int, help="Max queued requests", default=128)
    parser.add_argument("--weight", help="Path to MAX weight file", default="")
    args, _ = parser.parse_known_args()

    port = args.port or int(os.environ.get("PORT", 8080))
    model_path = args.model or os.environ.get("MODEL_PATH", "")
    weight_path = args.weight or os.environ.get("WEIGHT_PATH", "")
    llama_port = args.llama_port
    max_workers = args.max_workers
    queue_size = args.queue_size

    print("╔══════════════════════════════════════════════╗")
    print("║         MojoLlama — Inference Server         ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    backend = AutoBackend(model_path=model_path, weight_path=weight_path,
                         llama_port=llama_port)
    print(f"\nBackend: {backend.info['active']}\n")

    server = ThreadPoolHTTPServer(("0.0.0.0", port), MojoLlamaHandler,
                                  max_workers=max_workers,
                                  queue_size=queue_size)
    print(f"Serving on http://0.0.0.0:{port}")
    print(f"  Max workers: {max_workers}, Queue: {queue_size}")
    print(f"  /v1/models              — list models")
    print(f"  /v1/completions          — text completion")
    print(f"  /v1/chat/completions     — chat completion (SSE streaming)")
    print(f"  /health                  — health check")
    print(f"  /backend                 — backend info")
    print(f"  /api/backend             — studio backend info")
    print(f"  /api/models              — list GGUF models")
    print(f"  /api/export              — trigger HF→GGUF export")
    print(f"  /api/export/<job_id>     — export job status")
    print(f"  /api/dataset             — list datasets")
    print(f"  /api/chat                — streaming chat (web UI)")
    print(f"  /api/train/metrics       — training metrics SSE")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.stop()
        server.server_close()


if __name__ == "__main__":
    main()
