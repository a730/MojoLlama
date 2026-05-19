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
import http.client
import concurrent.futures
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from io import BytesIO

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mojollama.backends import AutoBackend
from mojollama.experiment import get_tracker, TrainingSession

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

# Training job tracking: job_id -> {"status": str, "log": str}
_train_jobs = {}
_train_jobs_lock = threading.Lock()

# Connection pool to llama.cpp backend — bounded, health-checked
_llama_pool_size = 32  # max concurrent connections (matches n_parallel * 8)
_llama_conn_pool = queue.Queue(maxsize=_llama_pool_size)

def _get_llama_conn():
    """Get a persistent HTTP connection from the pool (or create new).
    
    Connections are recycled after _MAX_POOL_AGE seconds to avoid stale sockets.
    """
    conn = None
    try:
        conn = _llama_conn_pool.get_nowait()
        # Discard stale connections (older than 60s idle)
        if hasattr(conn, '_pool_age') and time.time() - conn._pool_age > 60:
            try: conn.close()
            except: pass
            conn = None
    except queue.Empty:
        pass
    if conn is None:
        llama_port = backend.backend.port if hasattr(backend.backend, 'port') else 8081
        conn = http.client.HTTPConnection("127.0.0.1", llama_port, timeout=120)
    conn._pool_age = time.time()
    return conn

def _return_llama_conn(conn):
    """Return connection to pool, or close if pool is full / connection is broken."""
    try:
        _llama_conn_pool.put_nowait(conn)
    except queue.Full:
        try: conn.close()
        except: pass

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

    def _proxy_stream_from_llamacpp(self, request_data):
        """Proxy streaming through llama.cpp /completion?stream=true.

        Reads the SSE stream from llama.cpp line-by-line (buffered readline)
        and re-emits as OpenAI-format SSE events. Uses pooled HTTP connection.
        """
        prompt = request_data.get("prompt", "")
        max_tokens = request_data.get("max_tokens", 256)
        temperature = request_data.get("temperature", 0.7)

        llama_data = json.dumps({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": True,
            "cache_prompt": True,
        }).encode()

        conn = _get_llama_conn()
        conn.request(
            "POST", "/completion",
            body=llama_data,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                # Parse llama.cpp SSE format: data: {"content":"...","stop":false}
                if line_str.startswith("data: "):
                    payload = line_str[6:]
                    if payload.strip():
                        try:
                            inner = json.loads(payload)
                            token = inner.get("content", "")
                            stop = inner.get("stop", False)

                            # Emit OpenAI-format SSE
                            self._sse_send({
                                "choices": [{
                                    "delta": {"content": token},
                                    "index": 0,
                                }]
                            })

                            if stop:
                                break
                        except json.JSONDecodeError:
                            pass
        finally:
            resp.close()
            _return_llama_conn(conn)

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
                entry = {
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": size_bytes,
                    "size_gb": round(size_bytes / 1024**3, 2),
                    "size_hr": self._format_size(size_bytes),
                    "modified": f.stat().st_mtime,
                }
                # Read architecture info from GGUF metadata
                try:
                    from mojollama.quantizer import _read_gguf_metadata
                    meta = _read_gguf_metadata(str(f))
                    entry["architecture"] = meta.get("architecture", "unknown")
                    entry["n_params"] = meta.get("n_params", 0)
                    entry["n_params_hr"] = meta.get("n_params_hr", "?")
                    entry["quant_type"] = meta.get("quant_type", "?")
                    entry["context_length"] = meta.get("context_length", 0)
                    entry["block_count"] = meta.get("block_count", 0)
                    entry["n_heads"] = meta.get("n_heads", 0)
                    entry["embedding_length"] = meta.get("embedding_length", 0)
                except Exception:
                    entry["architecture"] = "unknown"
                models.append(entry)
            self._send_json({"models": models, "count": len(models)})

        # ── Studio API: /api/templates ──────────────────
        elif path == "/api/templates":
            # Architecture-aware training templates — clients can filter by arch
            templates = [
                {
                    "id": "llama3",
                    "name": "Llama 3 / 3.1 / 3.2",
                    "arch": ["llama", "llama2", "llama3", "codellama"],
                    "description": "LoRA fine-tune for Llama models. Alpaca format, cosine schedule.",
                    "rank": 16, "alpha": 32, "lr": "2e-4",
                    "scheduler": "cosine", "steps": 500, "batch": 4,
                    "grad_accum": 1, "warmup": 50, "weight_decay": "0.01",
                    "modules": "q_proj,v_proj,k_proj,o_proj",
                    "dropout": "0.05", "ctx_len": 2048, "flash_attn": True,
                    "format": "alpaca",
                    "system_prompt": "You are a helpful AI assistant based on Llama 3.",
                    "tags": ["r=16", "cosine", "alpaca"],
                },
                {
                    "id": "mistral",
                    "name": "Mistral / Mixtral",
                    "arch": ["mistral", "mistral3", "mixtral"],
                    "description": "Mistral-7B and Mixtral MoE fine-tuning. Sliding window attention, higher rank.",
                    "rank": 32, "alpha": 64, "lr": "1e-4",
                    "scheduler": "cosine", "steps": 1000, "batch": 4,
                    "grad_accum": 2, "warmup": 100, "weight_decay": "0.01",
                    "modules": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
                    "dropout": "0.05", "ctx_len": 4096, "flash_attn": True,
                    "format": "sharegpt",
                    "system_prompt": "You are a helpful AI assistant based on Mistral.",
                    "tags": ["r=32", "cosine", "sharegpt"],
                },
                {
                    "id": "qwen",
                    "name": "Qwen2 / Qwen3",
                    "arch": ["qwen", "qwen2", "qwen2moe", "qwen3moe"],
                    "description": "Qwen models with MoE support. Both dense and MoE architectures.",
                    "rank": 16, "alpha": 32, "lr": "2e-4",
                    "scheduler": "linear", "steps": 500, "batch": 4,
                    "grad_accum": 1, "warmup": 50, "weight_decay": "0.01",
                    "modules": "q_proj,v_proj",
                    "dropout": "0.05", "ctx_len": 2048, "flash_attn": True,
                    "format": "alpaca",
                    "system_prompt": "You are a helpful AI assistant based on Qwen.",
                    "tags": ["r=16", "linear", "alpaca"],
                },
                {
                    "id": "gemma",
                    "name": "Gemma / Gemma 2 / Gemma 4",
                    "arch": ["gemma", "gemma2", "gemma4"],
                    "description": "Google Gemma fine-tuning. GeLU activation, RoPE scaling changes in v2/v4.",
                    "rank": 16, "alpha": 32, "lr": "2e-4",
                    "scheduler": "cosine", "steps": 500, "batch": 4,
                    "grad_accum": 1, "warmup": 50, "weight_decay": "0.01",
                    "modules": "q_proj,v_proj,k_proj,o_proj",
                    "dropout": "0.05", "ctx_len": 2048, "flash_attn": True,
                    "format": "alpaca",
                    "system_prompt": "You are a helpful AI assistant based on Gemma.",
                    "tags": ["r=16", "cosine", "gemma"],
                },
                {
                    "id": "phi",
                    "name": "Phi-3 / Phi-4",
                    "arch": ["phi", "phi3", "phi2"],
                    "description": "Microsoft Phi models. Compact but capable — lower rank recommended.",
                    "rank": 8, "alpha": 16, "lr": "5e-4",
                    "scheduler": "constant", "steps": 300, "batch": 4,
                    "grad_accum": 1, "warmup": 20, "weight_decay": "0.01",
                    "modules": "q_proj,v_proj",
                    "dropout": "0.1", "ctx_len": 2048, "flash_attn": True,
                    "format": "alpaca",
                    "system_prompt": "You are a helpful AI assistant based on Phi-3.",
                    "tags": ["r=8", "constant", "instruct"],
                },
                {
                    "id": "deepseek",
                    "name": "DeepSeek V2/V3/R1",
                    "arch": ["deepseek", "deepseek2", "deepseek3"],
                    "description": "DeepSeek models with MLA attention. Higher rank for reasoning tasks.",
                    "rank": 64, "alpha": 128, "lr": "1e-4",
                    "scheduler": "cosine", "steps": 2000, "batch": 2,
                    "grad_accum": 4, "warmup": 100, "weight_decay": "0.005",
                    "modules": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
                    "dropout": "0.05", "ctx_len": 4096, "flash_attn": True,
                    "format": "sharegpt",
                    "system_prompt": "You are a reasoning AI assistant. Think step by step.",
                    "tags": ["r=64", "cosine", "sharegpt", "MLA"],
                },
                {
                    "id": "chatglm",
                    "name": "ChatGLM / GLM-4",
                    "arch": ["chatglm", "glm"],
                    "description": "ChatGLM prefix-encoder architecture. Different attention masking, custom tokenizer.",
                    "rank": 16, "alpha": 32, "lr": "2e-4",
                    "scheduler": "cosine", "steps": 500, "batch": 4,
                    "grad_accum": 2, "warmup": 50, "weight_decay": "0.01",
                    "modules": "q_proj,v_proj,k_proj,o_proj",
                    "dropout": "0.05", "ctx_len": 2048, "flash_attn": True,
                    "format": "alpaca",
                    "system_prompt": "You are a helpful AI assistant based on ChatGLM.",
                    "tags": ["r=16", "cosine", "prefix-encoder"],
                },
                {
                    "id": "command-r",
                    "name": "Command R / R+",
                    "arch": ["command-r", "commandr"],
                    "description": "Cohere Command R models. Large FFN layers, high capacity.",
                    "rank": 32, "alpha": 64, "lr": "1e-4",
                    "scheduler": "cosine", "steps": 1000, "batch": 2,
                    "grad_accum": 2, "warmup": 100, "weight_decay": "0.01",
                    "modules": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
                    "dropout": "0.05", "ctx_len": 4096, "flash_attn": True,
                    "format": "sharegpt",
                    "system_prompt": "You are a helpful AI assistant by Cohere. Provide accurate and grounded responses.",
                    "tags": ["r=32", "cosine", "large-ffn"],
                },
            ]

            # Optional arch filter: /api/templates?arch=deepseek
            arch_filter = query.get("arch", [None])[0]
            if arch_filter:
                arch_filter = arch_filter.lower()
                filtered = [t for t in templates if arch_filter in t["arch"]]
                self._send_json({"templates": filtered, "count": len(filtered)})
            else:
                self._send_json({"templates": templates, "count": len(templates)})

        # ── Studio API: /api/quant-types ──────────────────
        elif path == "/api/quant-types":
            from mojollama.quantizer import QUANT_TYPES
            types_list = []
            for name, desc, bpw, is_k, cat in QUANT_TYPES:
                types_list.append({
                    "name": name,
                    "description": desc,
                    "bits_per_weight": bpw,
                    "is_k_quant": is_k,
                    "category": cat,
                })
            self._send_json({"types": types_list, "count": len(types_list)})

        # ── Studio API: /api/model-info ──────────────────
        elif path == "/api/model-info":
            model_name = query.get("model", [None])[0]
            if not model_name:
                self._send_error("'model' query param required")
                return
            # Resolve
            model_path = None
            for f in Path(WORK_DIR).glob("*.gguf"):
                if f.name == model_name or str(f) == model_name:
                    model_path = str(f)
                    break
            if not model_path:
                self._send_error(f"Model not found: {model_name}", 404)
                return
            try:
                from mojollama.quantizer import _read_gguf_metadata
                meta = _read_gguf_metadata(model_path)
                self._send_json(meta)
            except Exception as e:
                self._send_error(f"Failed to read model info: {e}")

        # ── Studio API: /api/dataset ─────────────────────────────
        elif path == "/api/dataset":
            datasets = []
            for f in sorted(Path(WORK_DIR).glob("*.jsonl")) + sorted(Path(WORK_DIR).glob("*.json")):
                size_mb = f.stat().st_size / 1024 / 1024
                try:
                    from mojollama.dataset import detect_format, _count_lines
                    fmt = detect_format(str(f))
                    lines = _count_lines(str(f))
                except Exception:
                    fmt = "unknown"
                    lines = 0
                datasets.append({
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "size_mb": round(size_mb, 1),
                    "format": fmt,
                    "samples": lines,
                })
            self._send_json({"datasets": datasets, "count": len(datasets)})

        # ── Studio API: /api/dataset/info ────────────────────
        elif path == "/api/dataset/info":
            dataset = query.get("dataset", [None])[0]
            if not dataset:
                self._send_error("'dataset' query parameter required")
                return
            dataset_path = Path(WORK_DIR) / dataset
            if not dataset_path.exists():
                dataset_path = Path(dataset)
            if not dataset_path.exists():
                self._send_error(f"Dataset not found: {dataset}", 404)
                return
            try:
                from mojollama.dataset import get_dataset_info
                info = get_dataset_info(str(dataset_path))
                self._send_json(info)
            except Exception as e:
                self._send_error(f"Failed to read dataset info: {e}")

        # ── Studio API: /api/dataset/stats ───────────────────
        elif path == "/api/dataset/stats":
            dataset = query.get("dataset", [None])[0]
            max_samples = query.get("max_samples", [None])[0]
            if max_samples:
                try:
                    max_samples = int(max_samples)
                except (ValueError, TypeError):
                    max_samples = None
            if not dataset:
                self._send_error("'dataset' query parameter required")
                return
            dataset_path = Path(WORK_DIR) / dataset
            if not dataset_path.exists():
                dataset_path = Path(dataset)
            if not dataset_path.exists():
                self._send_error(f"Dataset not found: {dataset}", 404)
                return
            try:
                from mojollama.dataset import get_reader, detect_format, compute_stats
                fmt = detect_format(str(dataset_path))
                reader = get_reader(str(dataset_path), format=fmt, max_samples=max_samples)
                entries = reader.read()
                stats = compute_stats(entries)
                self._send_json(stats.to_dict())
            except Exception as e:
                self._send_error(f"Failed to compute stats: {e}")

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

        # ── Studio API: /api/train/metrics ───────────────────
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

        # ── Experiment Tracking API ──────────────────────────────
        elif path == "/api/experiments":
            tracker = get_tracker()
            status = query.get("status", [None])[0]
            limit = int(query.get("limit", [50])[0])
            offset = int(query.get("offset", [0])[0])
            runs = tracker.list_runs(status=status, limit=limit, offset=offset)
            self._send_json({"experiments": runs, "count": len(runs)})

        elif path.startswith("/api/experiments/compare"):
            # GET /api/experiments/compare?runs=a,b,c&metric=loss
            run_ids_str = query.get("runs", [""])[0]
            metric = query.get("metric", ["loss"])[0]
            if not run_ids_str:
                self._send_error("'runs' query param required (comma-separated IDs)")
                return
            run_ids = [rid.strip() for rid in run_ids_str.split(",")]
            tracker = get_tracker()
            result = tracker.compare_metrics(run_ids, metric)
            self._send_json(result)

        elif path.startswith("/api/experiments/"):
            parts = path.split("/")
            # /api/experiments/<id>
            # /api/experiments/<id>/metrics
            # /api/experiments/<id>/chart
            if len(parts) >= 4:
                exp_id = parts[3]
                sub_path = "/".join(parts[4:]) if len(parts) > 4 else ""
                tracker = get_tracker()

                if not sub_path:
                    run = tracker.get_run(exp_id)
                    if run:
                        self._send_json(run)
                    else:
                        self._send_error(f"Experiment {exp_id} not found", 404)
                elif sub_path == "metrics":
                    metric_name = query.get("metric", [None])[0]
                    metrics_data = tracker.get_metrics(exp_id, metric_name=metric_name)
                    self._send_json({"run_id": exp_id, "metric": metric_name, "points": metrics_data, "count": len(metrics_data)})
                elif sub_path == "chart":
                    metric_name = query.get("metric", [""])[0]
                    if not metric_name:
                        self._send_error("'metric' query param required")
                        return
                    data = tracker.get_metrics_chart_data(exp_id, metric_name)
                    self._send_json(data)
                else:
                    self._send_error(f"Unknown sub-path: {sub_path}", 404)
            else:
                self._send_error("Invalid experiment path", 404)

        # ── Model Cards API ──────────────────────────────────────
        elif path == "/api/model-cards":
            tracker = get_tracker()
            cards = tracker.list_model_cards()
            self._send_json({"model_cards": cards, "count": len(cards)})

        elif path.startswith("/api/model-cards/"):
            parts = path.split("/")
            if len(parts) >= 4:
                card_id = parts[3]
                sub_path = "/".join(parts[4:]) if len(parts) > 4 else ""
                tracker = get_tracker()

                if not sub_path:
                    card = tracker.get_model_card(card_id)
                    if card:
                        self._send_json(card)
                    else:
                        self._send_error(f"Model card {card_id} not found", 404)
                elif sub_path == "readme":
                    card = tracker.get_model_card(card_id)
                    if not card:
                        self._send_error(f"Model card {card_id} not found", 404)
                        return
                    readme = card.get("readme", "")
                    if not readme:
                        readme = tracker.generate_model_card_readme(card_id)
                    self._send_json({"card_id": card_id, "readme": readme})
                else:
                    self._send_error(f"Unknown sub-path: {sub_path}", 404)
            else:
                self._send_error("Invalid model card path", 404)

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

        # ── Hub ping (GET: check hub login status) ──────────────
        elif path == "/api/hub/whoami":
            try:
                from mojollama.exporter import hub_whoami
                user = hub_whoami()
                if user:
                    self._send_json({"logged_in": True, "user": user.get("name", "unknown"), "email": user.get("email", "")})
                else:
                    self._send_json({"logged_in": False, "user": None})
            except Exception as e:
                self._send_json({"logged_in": False, "error": str(e)})

        # ── List checkpoints (GET) ──────────────────────────────
        elif path == "/api/checkpoint":
            try:
                from mojollama.exporter import TrainingCheckpoint
                checkpoints = TrainingCheckpoint.list_checkpoints(str(WORK_DIR))
                self._send_json({"checkpoints": checkpoints, "count": len(checkpoints)})
            except Exception as e:
                self._send_error(f"Failed to list checkpoints: {e}")

        # ── Load checkpoint (GET) ───────────────────────────────
        elif path == "/api/checkpoint/load":
            ckpt_dir = query.get("dir", [None])[0] or str(WORK_DIR)
            try:
                from mojollama.exporter import TrainingCheckpoint
                ckpt = TrainingCheckpoint(ckpt_dir)
                data = ckpt.load()
                if data:
                    self._send_json(data)
                else:
                    self._send_error("No checkpoint found", 404)
            except Exception as e:
                self._send_error(f"Failed to load checkpoint: {e}")

        # ── Training job status (GET) ──────────────────────────
        elif path.startswith("/api/train/") and path != "/api/train/metrics":
            parts = path.split("/")
            if len(parts) >= 4:
                job_id = parts[3]
                with _train_jobs_lock:
                    job = _train_jobs.get(job_id, {"status": "not_found"})
                self._send_json({"job_id": job_id, **job})
            else:
                self._send_error("Invalid training job path", 404)

        # ── Training methods list (GET) ──────────────────────────
        elif path == "/api/train/methods":
            methods = [
                {"id": "lora", "name": "LoRA", "description": "Low-Rank Adaptation"},
                {"id": "qlora", "name": "QLoRA", "description": "NF4 Quantized LoRA"},
                {"id": "dora", "name": "DoRA", "description": "Weight-Decomposed LoRA"},
                {"id": "galore", "name": "GaLore", "description": "Gradient Low-Rank Projection"},
                {"id": "dpo", "name": "DPO", "description": "Direct Preference Optimization"},
                {"id": "orpo", "name": "ORPO", "description": "Odds Ratio Preference Optimization"},
                {"id": "kto", "name": "KTO", "description": "Kahneman-Tversky Optimization"},
                {"id": "simpo", "name": "SimPO", "description": "Simple Preference Optimization"},
                {"id": "grpo", "name": "GRPO", "description": "Group Relative Policy Optimization"},
            ]
            self._send_json({"methods": methods, "count": len(methods)})

        # ── Not found (do_GET) ─────────────────────────────────────────
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

            if stream:
                # Streaming response
                return self._handle_streaming_completion(prompt, max_tokens, temperature)

            t0 = time.time()
            result = backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            elapsed = time.time() - t0
            prompts_served += 1

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

        # ── Existing: /v1/chat/completions (with SSE streaming) ──
        elif path == "/v1/chat/completions":
            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 256)
            temperature = body.get("temperature", 0.7)
            stream = body.get("stream", False)

            if stream:
                # SSE streaming response
                return self._handle_streaming_chat_completion(
                    messages, max_tokens, temperature
                )

            # Non-streaming: proxy directly to llama.cpp's /v1/chat/completions
            # This gets proper chat template handling and is faster
            llama_chat_data = json.dumps({
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }).encode()

            conn = _get_llama_conn()
            try:
                conn.request(
                    "POST", "/v1/chat/completions",
                    body=llama_chat_data,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                raw = resp.read()
                resp.close()

                if resp.status >= 400:
                    self._send_error(
                        f"llama.cpp error: {resp.status} {raw.decode('utf-8', errors='replace')}",
                        resp.status,
                    )
                    return

                result = json.loads(raw.decode("utf-8"))
                prompts_served += 1
                # Preserve the model name from the backend response
                if "model" not in result:
                    result["model"] = "mojollama-llama-3.2-1b"
                self._send_json(result)
            except Exception as e:
                self._send_error(f"Failed to proxy chat completion: {e}", 502)
            finally:
                _return_llama_conn(conn)

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

        # ── Studio API: /api/dataset/convert ─────────────────
        elif path == "/api/dataset/convert":
            dataset = body.get("dataset", body.get("input", ""))
            output_file = body.get("output", body.get("output_file", ""))
            target_format = body.get("format", "openai")

            if not dataset:
                self._send_error("'dataset' field required")
                return
            if not output_file:
                output_file = f"converted-{target_format}.jsonl"

            dataset_path = Path(WORK_DIR) / dataset
            if not dataset_path.exists():
                dataset_path = Path(dataset)
            if not dataset_path.exists():
                self._send_error(f"Dataset not found: {dataset}", 404)
                return

            output_path = Path(WORK_DIR) / output_file
            try:
                from mojollama.dataset import convert_dataset
                result = convert_dataset(
                    str(dataset_path),
                    str(output_path),
                    target_format=target_format,
                )
                self._send_json(result)
            except Exception as e:
                self._send_error(f"Conversion failed: {e}")

        # ── Studio API: /api/dataset/auto-label ─────────────────
        elif path == "/api/dataset/auto-label":
            dataset = body.get("dataset", body.get("input", ""))
            output_file = body.get("output", "")
            model_name = body.get("model", "")
            prompt_field = body.get("prompt_field", "prompt")
            completion_field = body.get("completion_field", "completion")
            max_tokens = body.get("max_tokens", 64)

            if not dataset:
                self._send_error("'dataset' field is required")
                return

            dataset_path = Path(WORK_DIR) / dataset
            if not dataset_path.exists():
                dataset_path = Path(dataset)
            if not dataset_path.exists():
                self._send_error(f"Dataset not found: {dataset}", 404)
                return

            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                if backend is None:
                    self.wfile.write(b"ERROR: Backend not initialized\n")
                    return
                with open(dataset_path) as f:
                    lines = f.readlines()
                total = len(lines)
                self.wfile.write(f"Auto-labeling {total} samples...\n".encode())
                self.wfile.flush()

                labeled = 0
                for i, line in enumerate(lines):
                    try:
                        entry = json.loads(line)
                        if not entry.get(completion_field):
                            # Generate completion using the backend
                            prompt_text = entry.get(prompt_field, "")
                            if prompt_text:
                                result = backend.generate(prompt_text, max_tokens=max_tokens, temperature=0.5)
                                entry[completion_field] = result.get("text", "")
                                lines[i] = json.dumps(entry) + "\n"
                                labeled += 1
                                if labeled % 5 == 0:
                                    self.wfile.write(f"  Labeled {labeled}/{total}...\n".encode())
                                    self.wfile.flush()
                    except (json.JSONDecodeError, KeyError):
                        continue

                # Write updated dataset
                with open(dataset_path, "w") as f:
                    f.writelines(lines)

                self.wfile.write(f"\n✅ Auto-labeling complete: {labeled}/{total} samples labeled\n".encode())
            except Exception as e:
                self.wfile.write(f"\nERROR: {e}\n".encode())

        # ── Experiment Tracking API: POST ─────────────────────
        elif path == "/api/experiments/start":
            tracker = get_tracker()
            name = body.get("name", "untitled")
            description = body.get("description", "")
            tags = body.get("tags", {})
            params = body.get("params", {})
            auto_wandb = body.get("auto_wandb", True)
            auto_mlflow = body.get("auto_mlflow", True)
            try:
                run = tracker.start_run(
                    name=name,
                    description=description,
                    tags=tags,
                    params=params,
                    auto_wandb=auto_wandb,
                    auto_mlflow=auto_mlflow,
                )
                self._send_json(run, 201)
            except Exception as e:
                self._send_error(f"Failed to start experiment: {e}")

        elif path.startswith("/api/experiments/") and path.endswith("/log"):
            parts = path.split("/")
            exp_id = parts[3]
            tracker = get_tracker()
            run = tracker.get_run(exp_id)
            if not run:
                self._send_error(f"Experiment {exp_id} not found", 404)
                return
            metric_name = body.get("name", "")
            metric_value = body.get("value", 0)
            step = body.get("step", None)
            if not metric_name:
                self._send_error("'name' field required")
                return
            try:
                tracker.log_metric(metric_name, float(metric_value), step=step, run_id=exp_id)
                self._send_json({"status": "logged", "run_id": exp_id, "metric": metric_name, "value": float(metric_value)})
            except Exception as e:
                self._send_error(f"Failed to log metric: {e}")

        elif path.startswith("/api/experiments/") and path.endswith("/log-multi"):
            parts = path.split("/")
            exp_id = parts[3]
            tracker = get_tracker()
            metrics = body.get("metrics", {})
            step = body.get("step", None)
            if not metrics:
                self._send_error("'metrics' dict required")
                return
            for name, value in metrics.items():
                tracker.log_metric(name, float(value), step=step, run_id=exp_id)
            self._send_json({"status": "logged", "run_id": exp_id, "metrics_count": len(metrics)})

        elif path.startswith("/api/experiments/") and path.endswith("/params"):
            parts = path.split("/")
            exp_id = parts[3]
            tracker = get_tracker()
            params = body.get("params", {})
            if not params:
                self._send_error("'params' dict required")
                return
            tracker.log_params(params, run_id=exp_id)
            self._send_json({"status": "logged", "run_id": exp_id, "params_count": len(params)})

        elif path.startswith("/api/experiments/") and path.endswith("/stop"):
            parts = path.split("/")
            exp_id = parts[3]
            status = body.get("status", "completed")
            tracker = get_tracker()
            # Temporarily set current run to this one
            old_current = tracker._current_run_id
            tracker._current_run_id = exp_id
            result = tracker.stop_run(status=status)
            tracker._current_run_id = old_current
            if result:
                self._send_json(result)
            else:
                self._send_error(f"Experiment {exp_id} not found", 404)

        elif path.startswith("/api/experiments/") and path.endswith("/artifact"):
            parts = path.split("/")
            exp_id = parts[3]
            tracker = get_tracker()
            artifact_path = body.get("path", "")
            artifact_type = body.get("type", "model")
            description = body.get("description", "")
            if not artifact_path:
                self._send_error("'path' field required")
                return
            tracker.log_artifact(artifact_path, artifact_type=artifact_type, description=description, run_id=exp_id)
            self._send_json({"status": "logged", "run_id": exp_id, "artifact": artifact_path})

        # ── Model Cards API: POST ────────────────────────────────
        elif path == "/api/model-cards":
            tracker = get_tracker()
            card = tracker.create_model_card(
                model_name=body.get("model_name", "unknown"),
                architecture=body.get("architecture", "llama"),
                experiment_id=body.get("experiment_id"),
                description=body.get("description", ""),
                base_model=body.get("base_model", ""),
                dataset=body.get("dataset", ""),
                training_config=body.get("training_config", {}),
                metrics=body.get("metrics", {}),
                quant_type=body.get("quant_type", ""),
                params=body.get("params", {}),
                tags=body.get("tags", []),
                license=body.get("license", "MIT"),
            )
            self._send_json(card, 201)

        elif path.startswith("/api/model-cards/") and path.endswith("/readme"):
            parts = path.split("/")
            card_id = parts[3]
            tracker = get_tracker()
            readme = tracker.generate_model_card_readme(card_id)
            if readme:
                self._send_json({"card_id": card_id, "readme": readme})
            else:
                self._send_error(f"Model card {card_id} not found", 404)

        # ── Studio API: /api/benchmark ──────────────────────
        elif path == "/api/benchmark":
            model = body.get("model", "")
            max_tokens = body.get("max_tokens", 128)
            prompt_text = body.get("prompt", "Hello")
            threads = body.get("threads", 0)

            if not model:
                self._send_error("'model' is required")
                return

            # Resolve model path
            model_path = None
            for f in Path(WORK_DIR).glob("*.gguf"):
                if f.name == model or str(f) == model:
                    model_path = str(f)
                    break
            if not model_path and os.path.exists(model):
                model_path = model
            if not model_path:
                for f in Path(WORK_DIR).glob("*.gguf"):
                    if model in f.name:
                        model_path = str(f)
                        break

            if not model_path:
                self._send_error(f"Model not found: {model}")
                return

            # Use llama-bench if available, otherwise time a generation
            bench_path = os.path.join(os.path.dirname(LLAMA_SERVER_PATH), "llama-bench")
            if os.path.exists(bench_path):
                cmd = [bench_path, "-m", model_path, "-p", str(max_tokens),
                       "-n", str(max_tokens)]
                if threads:
                    cmd.extend(["-t", str(threads)])
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    lines = proc.stdout.strip().split('\n')
                    result = {"model": model, "backend": "llama-bench", "raw": proc.stdout}
                    for line in lines:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            try:
                                if "pp" in parts[1].lower() or "prompt" in parts[1].lower():
                                    result["prompt_tokens_per_second"] = float(parts[-2])
                                elif "tg" in parts[1].lower() or "gen" in parts[1].lower():
                                    result["tokens_per_second"] = float(parts[-2])
                            except (ValueError, IndexError):
                                pass
                    self._send_json(result)
                except subprocess.TimeoutExpired:
                    self._send_error("Benchmark timed out (300s)")
                except Exception as e:
                    self._send_error(f"Benchmark failed: {e}")
            else:
                # Fallback: time a generation via the llama backend
                t0 = time.time()
                try:
                    conn = _get_llama_conn()
                    prompt = prompt_text if prompt_text else "Hello"
                    data = json.dumps({
                        "prompt": prompt,
                        "n_predict": max_tokens,
                        "temperature": 0.0,
                        "stream": False,
                    }).encode()
                    conn.request("POST", "/completion", body=data,
                                 headers={"Content-Type": "application/json"})
                    resp = conn.getresponse()
                    raw = resp.read()
                    resp.close()
                    _return_llama_conn(conn)

                    result = json.loads(raw.decode("utf-8"))
                    elapsed = time.time() - t0
                    prompt_tokens = result.get("prompt_tokens", 0)
                    gen_tokens = result.get("tokens_evaluated", result.get("tokens", 0))
                    pp_speed = prompt_tokens / elapsed if elapsed > 0 and prompt_tokens else 0
                    tg_speed = gen_tokens / elapsed if elapsed > 0 and gen_tokens else 0

                    self._send_json({
                        "model": model,
                        "backend": "llama.cpp",
                        "prompt_tokens_per_second": round(pp_speed, 1),
                        "tokens_per_second": round(tg_speed, 1),
                        "total_time_seconds": round(elapsed, 2),
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": gen_tokens,
                        "response": result.get("content", "")[:200],
                    })
                except Exception as e:
                    self._send_error(f"Benchmark failed: {e}")

        # ── Studio API: /api/merge ────────────────────────
        elif path == "/api/merge":
            base = body.get("base", "")
            lora_adapter = body.get("lora", "")
            output = body.get("output", "merged.gguf")
            merge_type = body.get("type", "q4_0")

            if not base or not lora_adapter:
                self._send_error("'base' and 'lora' are required")
                return

            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            llama_server = LLAMA_SERVER_PATH
            if not os.path.exists(llama_server):
                self.wfile.write(f"ERROR: llama-server not found at {llama_server}\n".encode())
                return

            # Use llama-export-lora if available
            export_lora = os.path.join(os.path.dirname(llama_server), "llama-export-lora")
            if os.path.exists(export_lora):
                cmd = [export_lora, "-m", base, "-l", lora_adapter, "-o", output, "-t", merge_type]
            else:
                cmd = [llama_server, "-m", base, "--lora", lora_adapter, "--save", output]

            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                proc.wait()
                if proc.returncode == 0:
                    self.wfile.write(f"\n✅ Merge complete: {output}\n".encode())
                else:
                    self.wfile.write(f"\n❌ Merge failed (exit code {proc.returncode})\n".encode())
            except Exception as e:
                self.wfile.write(f"\nERROR: {e}\n".encode())

        # ── Studio API: /api/quantize ──────────────────────
        elif path == "/api/quantize":
            model = body.get("model", "")
            target = body.get("target", "q4_0")
            output = body.get("output", "")
            imatrix_file = body.get("imatrix", "")
            allow_requantize = body.get("allow_requantize", False)
            pure = body.get("pure", False)
            leave_output = body.get("leave_output", False)

            if not model:
                self._send_error("'model' is required")
                return

            # Resolve model path
            model_path = None
            for f in Path(WORK_DIR).glob("*.gguf"):
                if f.name == model or str(f) == model:
                    model_path = str(f)
                    break
            if not model_path and os.path.exists(model):
                model_path = model
            if not model_path:
                self._send_error(f"Model not found: {model}")
                return

            if not output:
                base_name = Path(model_path).stem
                output = str(Path(WORK_DIR) / f"{base_name}-{target}.gguf")

            llama_quantize = os.path.join(os.path.dirname(LLAMA_SERVER_PATH), "llama-quantize")
            if not os.path.exists(llama_quantize):
                self._send_error(f"llama-quantize not found at {llama_quantize}")
                return

            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            cmd = [llama_quantize, model_path, output, target, "4"]
            if imatrix_file:
                cmd.extend(["--imatrix", imatrix_file])
            if allow_requantize:
                cmd.append("--allow-requantize")
            if pure:
                cmd.append("--pure")
            if leave_output:
                cmd.append("--leave-output-tensor")

            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                proc.wait()
                if proc.returncode == 0:
                    size_mb = os.path.getsize(output) / 1024**2 if os.path.exists(output) else 0
                    orig_size_mb = os.path.getsize(model_path) / 1024**2
                    ratio = f"{size_mb/orig_size_mb:.1%}" if orig_size_mb > 0 else "?"
                    self.wfile.write(f"\n✅ Quantization complete: {output} ({size_mb:.0f} MB, {ratio} of original)\n".encode())
                else:
                    self.wfile.write(f"\n❌ Quantization failed (exit code {proc.returncode})\n".encode())
            except Exception as e:
                self.wfile.write(f"\nERROR: {e}\n".encode())

        # ── Studio API: /api/evaluate ──────────────────────
        elif path == "/api/evaluate":
            model = body.get("model", "")
            eval_type = body.get("type", "perplexity")
            eval_data = body.get("data", "")
            max_samples = body.get("max_samples", 100)
            benchmarks = body.get("benchmarks", [])

            if not model:
                self._send_error("'model' is required")
                return

            # Standard benchmarks use the llama.cpp backend URL
            BENCHMARK_TYPES = {"mmlu", "gsm8k", "ceval", "hellaswag", "arc", "bbh", "humaneval"}

            if eval_type in BENCHMARK_TYPES or (eval_type == "benchmark" and benchmarks):
                # ── Benchmark evaluation mode ──
                llama_port = backend.backend.port if hasattr(backend, 'backend') and hasattr(backend.backend, 'port') else 8081
                backend_url = f"http://127.0.0.1:{llama_port}"

                # Ensure llama.cpp server is running
                if not backend or not backend.backend:
                    self._send_error("Backend not running. Start the server with --model first.")
                    return

                self.send_response(200)
                self._set_cors()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                # Import eval framework
                from mojollama.eval.orchestrator import run_benchmark, run_benchmarks, serve_stream_results, get_benchmark

                if eval_type == "benchmark":
                    run_list = [b.lower() for b in benchmarks if b.lower() in BENCHMARK_TYPES]
                else:
                    run_list = [eval_type]

                if not run_list:
                    self.wfile.write(b"data: {\"type\":\"error\",\"message\":\"No valid benchmarks specified\"}\n\n")
                    self.wfile.flush()
                    return

                self.wfile.write(f"data: {json.dumps({'type':'progress','message':f'Starting {len(run_list)} benchmark(s) on {backend_url}'})}\n\n".encode())
                self.wfile.flush()

                try:
                    results = run_benchmarks(run_list, backend_url, model, max_samples)
                    serve_stream_results(results, self.wfile)
                except Exception as e:
                    self.wfile.write(f"data: {json.dumps({'type':'error','message':str(e)})}\n\n".encode())
                    self.wfile.flush()
                return

            # ── Classic evaluation modes (perplexity, accuracy) ──
            # Resolve model path
            model_path = None
            for f in Path(WORK_DIR).glob("*.gguf"):
                if f.name == model or str(f) == model:
                    model_path = str(f)
                    break
            if not model_path and os.path.exists(model):
                model_path = model
            if not model_path:
                self._send_error(f"Model not found: {model}")
                return

            llama_server = LLAMA_SERVER_PATH
            if not os.path.exists(llama_server):
                self._send_error(f"llama-server not found at {llama_server}")
                return

            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Perplexity evaluation using llama-perplexity
            perplexity_bin = os.path.join(os.path.dirname(llama_server), "llama-perplexity")
            if eval_type == "perplexity" and os.path.exists(perplexity_bin):
                cmd = [perplexity_bin, "-m", model_path, "-t", "4"]
                if eval_data and os.path.exists(eval_data):
                    cmd.extend(["-f", eval_data])
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    full_output = ""
                    for line in proc.stdout:
                        self.wfile.write(line.encode())
                        self.wfile.flush()
                        full_output += line
                    proc.wait()
                    ppl_match = re.search(r'perplexity\s*[:=]\s*([\d.]+)', full_output, re.IGNORECASE)
                    if not ppl_match:
                        ppl_match = re.search(r'([\d.]+)\s*\(perplexity', full_output)
                    result = {
                        "perplexity": float(ppl_match.group(1)) if ppl_match else None,
                        "type": "perplexity",
                        "model": model,
                    }
                    self.wfile.write(f"\n{json.dumps(result)}\n".encode())
                except Exception as e:
                    self.wfile.write(f"\nERROR: {e}\n".encode())
            else:
                # Fallback: simple generation-based evaluation
                self.wfile.write("Evaluation mode: generation-based\n".encode())
                self.wfile.write(f"Model: {model}\n".encode())
                try:
                    conn = _get_llama_conn()
                    prompt = "The capital of France is Paris. The capital of Germany is"
                    data = json.dumps({
                        "prompt": prompt, "n_predict": 16,
                        "temperature": 0.0, "stream": False,
                    }).encode()
                    t0 = time.time()
                    conn.request("POST", "/completion", body=data,
                                 headers={"Content-Type": "application/json"})
                    resp = conn.getresponse()
                    raw = resp.read()
                    resp.close()
                    _return_llama_conn(conn)
                    elapsed = time.time() - t0
                    result = json.loads(raw.decode("utf-8"))
                    self.wfile.write(f"Generated: {result.get('content', '')}\n".encode())
                    self.wfile.write(f"Time: {elapsed:.2f}s\n".encode())
                    self.wfile.write(f"{json.dumps({'type': 'accuracy', 'accuracy': 0, 'model': model, 'tokens': result.get('tokens', 0), 'total': 1, 'correct': 0, 'loss': 0})}\n".encode())
                except Exception as e:
                    self.wfile.write(f"ERROR: {e}\n".encode())

        # ── Studio API: /api/train ────────────────────────
        elif path == "/api/train":
            method = body.get("method", "lora")
            model = body.get("model", "")
            data = body.get("data", "")
            lora_rank = body.get("lora_rank", 16)
            lora_alpha = body.get("lora_alpha", 32)
            lr = body.get("lr", 1e-4)
            epochs = body.get("epochs", 2)
            batch_size = body.get("batch_size", 4)
            output = body.get("output", "adapter.gguf")
            dataset_format = body.get("dataset_format", "auto")
            template = body.get("template", "alpaca")
            max_samples = body.get("max_samples", 0)
            weight_decay = body.get("weight_decay", 0.0)
            warmup_steps = body.get("warmup_steps", 0)
            max_seq_length = body.get("max_seq_length", 512)
            seed = body.get("seed", 42)
            save_steps = body.get("save_steps", 0)

            if not model or not data:
                self._send_error("'model' and 'data' fields are required")
                return

            # Resolve model path
            model_path = None
            for f in Path(WORK_DIR).glob("*.gguf"):
                if f.name == model or model in f.name:
                    model_path = str(f)
                    break
            if not model_path and os.path.exists(model):
                model_path = model
            if not model_path:
                self._send_error(f"Model not found: {model}")
                return

            data_path = data
            if not os.path.exists(data_path):
                data_path = str(Path(WORK_DIR) / data)
            if not os.path.exists(data_path):
                self._send_error(f"Data not found: {data}")
                return

            # Build method-specific kwargs
            kwargs = {}
            if method == "dpo":
                kwargs["dpo_beta"] = body.get("dpo_beta", 0.1)
                kwargs["dpo_lr"] = lr
            elif method == "orpo":
                kwargs["orpo_lambda"] = body.get("orpo_lambda", 0.1)
                kwargs["orpo_lr"] = lr
            elif method == "simpo":
                kwargs["simpo_gamma"] = body.get("simpo_gamma", 0.5)
                kwargs["simpo_lr"] = lr
            elif method == "grpo":
                kwargs["grpo_group_size"] = body.get("grpo_group_size", 8)
                kwargs["grpo_clip"] = body.get("grpo_clip", 0.2)
                kwargs["grpo_lr"] = lr
            elif method == "galore":
                kwargs["galore_lr"] = lr
                kwargs["galore_rank"] = body.get("galore_rank", 128)

            # Start training in background thread
            job_id = str(uuid.uuid4())[:8]

            from mojollama.trainer import MojoLlamaTrainer

            def _run_training(jid, mp, dp, mtd, **tkwargs):
                log_lines = []
                def log(msg):
                    log_lines.append(msg)
                    with _export_jobs_lock:
                        _train_jobs[jid] = {"log": "\n".join(log_lines), "status": "running"}

                log(f"Training method: {mtd}")
                log(f"Model: {mp}")
                log(f"Data: {dp}")
                log(f"Parameters: rank={tkwargs.get('lora_rank')}, alpha={tkwargs.get('lora_alpha')}, "
                    f"lr={tkwargs.get('lr')}, epochs={tkwargs.get('epochs')}")
                log("")
                log("Starting training...")

                try:
                    trainer = MojoLlamaTrainer(
                        model_path=mp,
                        data_path=dp,
                        method=mtd,
                        **tkwargs,
                    )
                    trainer.train()
                    log(f"\n✅ Training complete! Output: {tkwargs.get('output_path')}")
                    with _export_jobs_lock:
                        _train_jobs[jid] = {"log": "\n".join(log_lines), "status": "done"}
                except Exception as e:
                    log(f"\n❌ Training failed: {e}")
                    import traceback
                    log(traceback.format_exc())
                    with _export_jobs_lock:
                        _train_jobs[jid] = {"log": "\n".join(log_lines), "status": "error"}

            from mojollama.trainer import subscribe_metrics, unsubscribe_metrics

            thread_kwargs = {
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "output_path": output,
                "dataset_format": dataset_format,
                "template": template,
                "max_samples": max_samples,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "max_seq_length": max_seq_length,
                "seed": seed,
                "save_steps": save_steps,
                **kwargs,
            }

            thread = threading.Thread(
                target=_run_training,
                args=(job_id, model_path, data_path, method),
                kwargs=thread_kwargs,
                daemon=True,
            )
            thread.start()

            self._send_json({
                "job_id": job_id,
                "status": "running",
                "method": method,
                "message": f"Training job {job_id} started ({method})",
            })

        # ── Studio API: /api/hub/login ───────────────────────────
        elif path == "/api/hub/login":
            try:
                from mojollama.exporter import hub_login
                token = body.get("token", "")
                success = hub_login(token=token)
                self._send_json({"success": success, "message": "Logged in successfully" if success else "Login failed"})
            except Exception as e:
                self._send_error(f"Login failed: {e}")

        # ── Studio API: /api/hub/push ────────────────────────────
        elif path == "/api/hub/push":
            from mojollama.exporter import hub_push_model
            model_path = body.get("model", "")
            repo_id = body.get("repo", "")
            private = body.get("private", False)
            commit_message = body.get("message", "Upload via MojoLlama Studio")
            metadata = body.get("metadata", {})

            if not model_path or not repo_id:
                self._send_error("'model' and 'repo' are required")
                return

            def _run_hub_push():
                self.send_response(200)
                self._set_cors()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    url = hub_push_model(
                        model_path=model_path, repo_id=repo_id,
                        private=private, commit_message=commit_message,
                        metadata=metadata,
                    )
                    if url:
                        self.wfile.write(f"\n✅ Uploaded to {url}\n".encode())
                    else:
                        self.wfile.write("\n❌ Upload failed\n".encode())
                except Exception as e:
                    self.wfile.write(f"\nERROR: {e}\n".encode())

            thread = threading.Thread(target=_run_hub_push, daemon=True)
            thread.start()

        # ── Studio API: /api/hub/push-adapter ─────────────────────
        elif path == "/api/hub/push-adapter":
            from mojollama.exporter import hub_push_adapter
            adapter_path = body.get("adapter", "")
            base_model = body.get("base_model", "")
            repo_id = body.get("repo", "")
            private = body.get("private", False)
            metadata = body.get("metadata", {})

            if not adapter_path or not base_model or not repo_id:
                self._send_error("'adapter', 'base_model', and 'repo' are required")
                return

            def _run_hub_push_adapter():
                self.send_response(200)
                self._set_cors()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    url = hub_push_adapter(
                        adapter_path=adapter_path, base_model=base_model,
                        repo_id=repo_id, private=private, metadata=metadata,
                    )
                    if url:
                        self.wfile.write(f"\n✅ Adapter uploaded to {url}\n".encode())
                    else:
                        self.wfile.write("\n❌ Upload failed\n".encode())
                except Exception as e:
                    self.wfile.write(f"\nERROR: {e}\n".encode())

            thread = threading.Thread(target=_run_hub_push_adapter, daemon=True)
            thread.start()

        # ── Studio API: /api/export/safetensors ─────────────────
        elif path == "/api/export/safetensors":
            from mojollama.exporter import gguf_to_safetensors
            gguf_path = body.get("model", "")
            output_dir = body.get("output", "")
            dtype = body.get("dtype", "float16")
            shard_size = body.get("shard_size", "2GB")

            if not gguf_path:
                self._send_error("'model' (GGUF path) is required")
                return

            # Resolve model path
            for f in Path(WORK_DIR).glob("*.gguf"):
                if f.name == gguf_path or str(f) == gguf_path:
                    gguf_path = str(f)
                    break
            if not os.path.exists(gguf_path):
                self._send_error(f"Model not found: {gguf_path}")
                return

            if not output_dir:
                output_dir = gguf_path.replace(".gguf", "-safetensors")

            def _run_st():
                self.send_response(200)
                self._set_cors()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    result = gguf_to_safetensors(
                        gguf_path=gguf_path, output_dir=output_dir,
                        dtype=dtype, shard_size=shard_size,
                    )
                    if result:
                        self.wfile.write(f"\n✅ Saved to {result}\n".encode())
                    else:
                        self.wfile.write("❌ Conversion failed\n".encode())
                except Exception as e:
                    self.wfile.write(f"\nERROR: {e}\n".encode())

            thread = threading.Thread(target=_run_st, daemon=True)
            thread.start()

        # ── Studio API: /api/export/onnx ─────────────────────────
        elif path == "/api/export/onnx":
            from mojollama.exporter import gguf_to_onnx
            gguf_path = body.get("model", "")
            output_path = body.get("output", "")
            opset = body.get("opset", 17)
            max_seq_len = body.get("max_seq_len", 2048)

            if not gguf_path:
                self._send_error("'model' (GGUF path) is required")
                return

            # Resolve model path
            for f in Path(WORK_DIR).glob("*.gguf"):
                if f.name == gguf_path or str(f) == gguf_path:
                    gguf_path = str(f)
                    break
            if not os.path.exists(gguf_path):
                self._send_error(f"Model not found: {gguf_path}")
                return

            if not output_path:
                output_path = gguf_path.replace(".gguf", ".onnx")

            def _run_onnx():
                self.send_response(200)
                self._set_cors()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    result = gguf_to_onnx(
                        gguf_path=gguf_path, output_path=output_path,
                        opset=opset, max_seq_len=max_seq_len,
                    )
                    if result:
                        self.wfile.write(f"\n✅ ONNX saved to {result}\n".encode())
                    else:
                        self.wfile.write("❌ Conversion failed\n".encode())
                except Exception as e:
                    self.wfile.write(f"\nERROR: {e}\n".encode())

            thread = threading.Thread(target=_run_onnx, daemon=True)
            thread.start()

        # ── Studio API: /api/checkpoint/save ─────────────────────
        elif path == "/api/checkpoint/save":
            try:
                from mojollama.exporter import TrainingCheckpoint
                ckpt_dir = body.get("dir", "checkpoints")
                step = body.get("step", 0)
                epoch = body.get("epoch", 0)
                loss = body.get("loss", 0.0)
                model_path = body.get("model", "")

                ckpt = TrainingCheckpoint(ckpt_dir)
                result = ckpt.save(step=step, epoch=epoch, loss=loss, model_path=model_path or None)
                if result:
                    self._send_json({"success": True, "path": result, "step": step, "loss": loss})
                else:
                    self._send_error("Failed to save checkpoint")
            except Exception as e:
                self._send_error(f"Failed to save checkpoint: {e}")

        # ── Not found (do_POST) ─────────────────────────────────────────
        else:
            self._send_error("not found", 404)

    # ── DELETE ──────────────────────────────────────────────────────

    def do_DELETE(self):
        path, query = self._parse_path()

        if path.startswith("/api/experiments/"):
            parts = path.split("/")
            if len(parts) >= 4:
                exp_id = parts[3]
                tracker = get_tracker()
                if tracker.delete_run(exp_id):
                    self._send_json({"status": "deleted", "run_id": exp_id})
                else:
                    self._send_error(f"Experiment {exp_id} not found", 404)
            else:
                self._send_error("Invalid experiment path", 404)
        elif path.startswith("/api/model-cards/"):
            parts = path.split("/")
            if len(parts) >= 4:
                card_id = parts[3]
                tracker = get_tracker()
                if tracker.delete_model_card(card_id):
                    self._send_json({"status": "deleted", "card_id": card_id})
                else:
                    self._send_error(f"Model card {card_id} not found", 404)
            else:
                self._send_error("Invalid model card path", 404)
        else:
            self._send_error("not found", 404)

    # ── Streaming Helpers ─────────────────────────────────────────

    def _handle_streaming_chat_completion(self, messages, max_tokens, temperature):
        """Handle SSE streaming for chat completions."""
        # Build a prompt from messages for llama.cpp
        prompt = self._messages_to_prompt(messages)

        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            self._proxy_stream_from_llamacpp({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
        except Exception as e:
            try:
                self._sse_send({
                    "choices": [{
                        "delta": {"content": f"\n[Error: {e}]"},
                        "index": 0,
                    }]
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
        """Handle SSE streaming for text completions."""
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            self._proxy_stream_from_llamacpp({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
        except Exception as e:
            try:
                self._sse_send({
                    "choices": [{
                        "text": f"\n[Error: {e}]",
                        "index": 0,
                    }]
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
    print(f"  /api/dataset             — list datasets with format detection")
    print(f"  /api/dataset/info        — dataset metadata and format info")
    print(f"  /api/dataset/stats       — dataset statistics (samples, tokens, lengths)")
    print(f"  /api/dataset/convert     — convert between dataset formats")
    print(f"  /api/dataset/auto-label  — auto-label dataset entries")
    print(f"  /api/chat                — streaming chat (web UI)")
    print(f"  /api/benchmark           — run benchmark")
    print(f"  /api/templates           — training templates (filter: ?arch=gemma)")
    print(f"  /api/quantize            — quantize GGUF model")
    print(f"  /api/evaluate            — evaluate model (perplexity, mmlu, gsm8k, ceval, hellaswag, arc, bbh, humaneval)")
    print(f"  /api/merge               — merge LoRA into base model")
    print(f"  /api/train/metrics       — training metrics SSE")
    print(f"  /api/experiments          — list experiments")
    print(f"  /api/experiments/start    — start experiment")
    print(f"  /api/experiments/compare  — compare metrics across runs")
    print(f"  /api/experiments/<id>     — experiment details")
    print(f"  /api/experiments/<id>/log    — log metric")
    print(f"  /api/experiments/<id>/params — log params")
    print(f"  /api/experiments/<id>/stop   — stop experiment")
    print(f"  /api/experiments/<id>/metrics  — get metrics")
    print(f"  /api/model-cards          — list/create model cards")
    print(f"  /api/model-cards/<id>     — model card details")
    print(f"  /api/model-cards/<id>/readme  — generate README")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.stop()
        server.server_close()


if __name__ == "__main__":
    main()
