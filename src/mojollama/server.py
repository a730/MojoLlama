#!/usr/bin/env python3
"""MojoLlama server — unified API server with auto-backend selection.

Starts an OpenAI-compatible API server that routes inference through:
  - MAX GPU (if GPU available)
  - llama.cpp (if CPU only)
  - MAX CPU (fallback)

Usage: python3 server.py
  # Starts on port 8080 with auto-detected backend
"""

import os
import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mojollama.backends import AutoBackend

backend = None
prompts_served = 0
start_time = time.time()


class MojoLlamaHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible API handler."""
    
    def log_message(self, format, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]} {args[2]}\n")
    
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": "mojollama-llama-3.2-1b",
                    "object": "model",
                    "created": int(start_time),
                    "owned_by": "mojollama",
                    "backend": backend.info["active"],
                }]
            })
        elif self.path == "/health":
            self._send_json({
                "status": "ok",
                "backend": backend.info["active"],
                "uptime": f"{time.time() - start_time:.0f}s",
            })
        elif self.path == "/backend":
            self._send_json(backend.info)
        else:
            self._send_json({"error": "not found"}, 404)
    
    def do_POST(self):
        global prompts_served
        if self.path in ("/v1/completions", "/completion"):
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            prompt = body.get("prompt", "")
            max_tokens = body.get("max_tokens", body.get("n_predict", 128))
            
            t0 = time.time()
            result = backend.generate(prompt, max_tokens=max_tokens)
            elapsed = time.time() - t0
            prompts_served += 1
            
            self._send_json({
                "id": f"cmpl-{prompts_served}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": "mojollama-llama-3.2-1b",
                "choices": [{"text": result.get("text", ""), "index": 0}],
                "usage": {
                    "completion_tokens": result.get("tokens", 0),
                    "total_tokens": result.get("tokens", 0),
                },
                "backend": result.get("backend"),
                "timings": {"total": f"{elapsed:.2f}s"},
            })
        
        elif self.path == "/v1/chat/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 256)
            
            t0 = time.time()
            result = backend.chat(messages, max_tokens=max_tokens)
            elapsed = time.time() - t0
            prompts_served += 1
            
            self._send_json({
                "id": f"chatcmpl-{prompts_served}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mojollama-llama-3.2-1b",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.get("text", ""),
                    },
                }],
                "usage": {
                    "completion_tokens": result.get("tokens", 0),
                    "total_tokens": result.get("tokens", 0),
                },
                "backend": result.get("backend"),
            })
        
        else:
            self._send_json({"error": "not found"}, 404)


def main():
    global backend
    port = int(os.environ.get("PORT", 8080))
    model_path = os.environ.get("MODEL_PATH", "")
    weight_path = os.environ.get("WEIGHT_PATH", "")
    
    print("╔══════════════════════════════════════════════╗")
    print("║         MojoLlama — Inference Server         ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    
    backend = AutoBackend(model_path=model_path, weight_path=weight_path)
    print(f"\nBackend: {backend.info['active']}\n")
    
    server = HTTPServer(("0.0.0.0", port), MojoLlamaHandler)
    print(f"Serving on http://0.0.0.0:{port}")
    print(f"  /v1/models        — list models")
    print(f"  /v1/completions   — text completion")
    print(f"  /v1/chat/completions — chat completion")
    print(f"  /health           — health check")
    print(f"  /backend          — backend info")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.stop()
        server.server_close()


if __name__ == "__main__":
    main()
