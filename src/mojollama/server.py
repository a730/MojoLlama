"""
MojoLlama — OpenAI-compatible API server (stdlib only).

Endpoints:
  GET  /v1/models           List loaded models
  POST /v1/chat/completions  Chat completion
  POST /v1/completions       Text completion

Usage:
  python -m mojollama.server --model /path/to/model.gguf --port 8080
"""
import argparse
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from mojollama.bridge import MojoLlamaBridge


# ─── Engine ──────────────────────────────────────────────────────────────────

class MojoLlamaEngine:
    """Generation engine wrapping MojoLlamaBridge."""

    def __init__(self, model_path: str):
        self.model = MojoLlamaBridge(model_path)
        self.eos_id = self.model._eos_id()

    def tokenize(self, text: str) -> list[int]:
        return self.model.tokenize(text)

    def detokenize(self, ids: list[int]) -> str:
        return self.model.detokenize(ids)

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        ids = self.tokenize(prompt)
        out = []
        self.model._kv_cache = None
        self.model._cached_len = 0
        for _ in range(max_tokens):
            logits = self.model.forward(ids)
            if logits is None:
                break
            nid = int(logits[-1].argmax())
            if nid == self.eos_id:
                break
            out.append(nid)
            ids.append(nid)
        return self.detokenize(out)

    def chat_template(self, messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")
        parts.append("Assistant: ")
        return "\n".join(parts)


# ─── HTTP Handler ────────────────────────────────────────────────────────────

class MojoLlamaHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""

    engine: MojoLlamaEngine = None  # set by server
    model_path: str = ""

    def _json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({
                "object": "list",
                "data": [{
                    "id": self.model_path.split("/")[-1].replace(".gguf", ""),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mojollama",
                }],
            })
        elif self.path == "/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        try:
            body = self._read_body()
        except Exception:
            self._json({"error": "Invalid JSON"}, 400)
            return

        if self.path == "/v1/chat/completions":
            self._handle_chat(body)
        elif self.path == "/v1/completions":
            self._handle_completion(body)
        else:
            self._json({"error": "Not found"}, 404)

    def _handle_chat(self, body: dict):
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 128)
        stream = body.get("stream", False)

        prompt = self.engine.chat_template(messages)
        t0 = time.time()
        text = self.engine.generate(prompt, max_tokens)
        elapsed = time.time() - t0

        prompt_ids = self.engine.tokenize(prompt)
        completion_ids = self.engine.tokenize(text)

        if stream:
            # Simplified streaming (single response for now)
            pass

        self._json({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path.split("/")[-1].replace(".gguf", ""),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
            },
        })
        print(f"[MojoLlama] Chat: {len(prompt_ids)} prompt → {len(completion_ids)} tokens ({elapsed:.1f}s)")

    def _handle_completion(self, body: dict):
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 128)

        t0 = time.time()
        text = self.engine.generate(prompt, max_tokens)
        elapsed = time.time() - t0

        self._json({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_path.split("/")[-1].replace(".gguf", ""),
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }],
        })
        print(f"[MojoLlama] Completion: {len(prompt)} chars → {elapsed:.1f}s")

    def log_message(self, format, *args):
        pass  # Quiet


# ─── Server ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MojoLlama API Server")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"[MojoLlama] Loading model: {args.model}")
    t0 = time.time()
    MojoLlamaHandler.engine = MojoLlamaEngine(args.model)
    MojoLlamaHandler.model_path = args.model
    print(f"[MojoLlama] Model loaded in {time.time() - t0:.1f}s")
    print(f"[MojoLlama] Server: http://{args.host}:{args.port}")
    print(f"[MojoLlama] Endpoints:")
    print(f"  GET  /v1/models")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/completions")

    server = HTTPServer((args.host, args.port), MojoLlamaHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MojoLlama] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
