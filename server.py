"""Mojo Hybrid Serve — OpenAI-compatible API server with web UI.
Usage:
  python server.py                       # Uses default model path
  MODEL_PATH=model.gguf python server.py # Custom model
  python server.py --help                # All options
"""
import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, request, jsonify, render_template_string, send_from_directory

from src.mojollama.model.inference import LLMInference
from src.mojollama.model.device import list_devices
from src.mojollama.quantizer import convert, quantize, get_info, validate, benchmark

# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000
DEFAULT_MAX_TOKENS = 256

app = Flask(__name__)
model = None
model_lock = threading.Lock()
current_model_path = None

# ─── HTML Template (embedded for single-file deployment) ────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mojo Hybrid Serve</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; height: 100vh; display: flex; flex-direction: column; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 20px; display: flex;
           align-items: center; gap: 12px; flex-shrink: 0; }
  header h1 { font-size: 18px; font-weight: 600; color: #f0f6fc; }
  header span { color: #8b949e; font-size: 13px; }
  .model-badge { background: #1f6feb22; border: 1px solid #1f6feb44; color: #58a6ff;
                 padding: 2px 10px; border-radius: 12px; font-size: 12px; }
  .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 12px 16px; border-radius: 8px; line-height: 1.5; white-space: pre-wrap; font-size: 14px; }
  .msg.user { align-self: flex-end; background: #1f6feb; color: #fff; }
  .msg.assistant { align-self: flex-start; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; }
  .msg.system { align-self: center; background: #1c2128; border: 1px solid #30363d; color: #8b949e; font-style: italic; font-size: 12px; }
  .msg .meta { font-size: 11px; color: #8b949e; margin-top: 4px; }
  .input-area { background: #161b22; border-top: 1px solid #30363d; padding: 12px 20px; display: flex; gap: 8px; flex-shrink: 0; }
  .input-area textarea { flex: 1; background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
                         color: #c9d1d9; padding: 10px 12px; font-size: 14px; resize: none; min-height: 44px;
                         font-family: inherit; outline: none; }
  .input-area textarea:focus { border-color: #1f6feb; }
  .input-area button { background: #238636; color: #fff; border: none; border-radius: 6px;
                       padding: 0 20px; font-size: 14px; font-weight: 600; cursor: pointer; white-space: nowrap; }
  .input-area button:hover { background: #2ea043; }
  .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #30363d;
             border-top-color: #58a6ff; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .typing { color: #8b949e; font-size: 13px; padding: 8px 16px; }
  pre { background: #161b22; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>🔥 Mojo Hybrid Serve</h1>
  <span class="model-badge" id="model-name">Loading...</span>
  <span id="perf-stats" style="font-size:12px;color:#8b949e;margin-left:auto;"></span>
</header>
<div class="chat-container" id="chat"></div>
<div class="input-area">
  <textarea id="input" placeholder="Type a message..." rows="1"
    onkeydown="if(event.key=='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
  <button id="send-btn" onclick="send()">Send</button>
</div>

<script>
let messages = [{"role": "system", "content": "You are a helpful AI assistant."}];
let generating = false;
let perfData = {};

async function fetchModels() {
  const r = await fetch('/v1/models');
  const data = await r.json();
  const name = data.data?.[0]?.id || 'unknown';
  document.getElementById('model-name').textContent = name;
}
fetchModels();

function addMsg(role, content) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = content;
  if (role === 'assistant') {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.id = 'perf-meta';
    chat.appendChild(div);
    chat.appendChild(meta);
  } else {
    chat.appendChild(div);
  }
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function showTyping() {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'typing';
  div.id = 'typing-indicator';
  div.innerHTML = '<span class="spinner"></span> Thinking...';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

async function send() {
  if (generating) return;
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  generating = true;
  document.getElementById('send-btn').disabled = true;

  messages.push({"role": "user", "content": text});
  addMsg('user', text);
  showTyping();

  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: '', messages: messages, max_tokens: 50, stream: false})
    });
    const data = await r.json();
    hideTyping();
    const reply = data.choices?.[0]?.message?.content || '[no response]';
    messages.push({"role": "assistant", "content": reply});
    addMsg('assistant', reply);
    if (data.usage) {
      document.getElementById('perf-stats').textContent =
        `${data.usage.total_tokens || '?'} tokens | ${data.usage.completion_tokens || '?'} generated`;
    }
  } catch(e) {
    hideTyping();
    addMsg('system', 'Error: ' + e.message);
  }
  generating = false;
  document.getElementById('send-btn').disabled = false;
  input.focus();
}
</script>
</body>
</html>"""

# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def web_ui():
    return render_template_string(HTML_TEMPLATE)


@app.route("/v1/models")
def list_models():
    with model_lock:
        path = current_model_path or "none"
        name = os.path.splitext(os.path.basename(path))[0] if path != "none" else "unknown"
    return jsonify({
        "object": "list",
        "data": [{
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mojollama",
        }]
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    temperature = float(data.get("temperature", 0.7))
    stream = data.get("stream", False)

    # Extract the last user message as prompt
    prompt = ""
    for msg in messages:
        if msg.get("role") == "user":
            prompt = msg.get("content", "")
        elif msg.get("role") == "assistant" and msg.get("content"):
            prompt += "\n" + msg.get("content", "")

    if not prompt:
        prompt = messages[-1].get("content", "") if messages else ""

    # Format as instruct prompt
    full_prompt = _format_chat_prompt(messages)

    with model_lock:
        if model is None:
            return jsonify({"error": "No model loaded"}), 500

        # Generate
        t0 = time.time()
        try:
            output = model.generate(full_prompt, max_tokens=max_tokens)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        elapsed = time.time() - t0

    usage = {
        "prompt_tokens": len(model.encode(full_prompt)),
        "completion_tokens": len(model.encode(output)),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    result = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": os.path.splitext(os.path.basename(current_model_path or ""))[0],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output.strip()},
            "finish_reason": "stop",
        }],
        "usage": usage,
    }
    return jsonify(result)


@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))

    with model_lock:
        if model is None:
            return jsonify({"error": "No model loaded"}), 500

        t0 = time.time()
        try:
            output = model.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        elapsed = time.time() - t0

    return jsonify({
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": os.path.splitext(os.path.basename(current_model_path or ""))[0],
        "choices": [{
            "index": 0,
            "text": output,
            "finish_reason": "stop",
        }],
    })


# ─── Export / Quantize API ──────────────────────────────────────────────────


@app.route("/api/export", methods=["POST"])
def api_export():
    """Convert a HuggingFace model to GGUF format."""
    data = request.get_json(force=True)
    model_name = data.get("model", "")
    outtype = data.get("outtype", "f16")
    outfile = data.get("outfile", "")

    if not model_name:
        return jsonify({"error": "model is required"}), 400
    if outtype not in ("f32", "f16", "bf16", "q8_0", "auto"):
        return jsonify({"error": f"unsupported outtype: {outtype}"}), 400

    try:
        result = convert(
            model_name,
            outtype=outtype,
            outfile=outfile if outfile else None,
        )
        return jsonify({
            "status": "ok",
            "output_path": result.get("output_path"),
            "size_bytes": result.get("size_bytes"),
            "size_human": result.get("size_human"),
            "elapsed_seconds": result.get("elapsed_seconds"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quantize", methods=["POST"])
def api_quantize():
    """Quantize an existing GGUF file."""
    data = request.get_json(force=True)
    input_path = data.get("input", "")
    quant_type = data.get("type", "q4_k_m")
    output_path = data.get("output", "")

    if not input_path or not os.path.exists(input_path):
        return jsonify({"error": f"input file not found: {input_path}"}), 400

    try:
        result = quantize(
            input_path,
            quant_type=quant_type,
            output_path=output_path if output_path else None,
        )
        return jsonify({
            "status": "ok",
            "output_path": result.get("output_path"),
            "quant_type": result.get("quant_type"),
            "input_size_bytes": result.get("input_size_bytes"),
            "output_size_bytes": result.get("output_size_bytes"),
            "compression_ratio": result.get("compression_ratio"),
            "elapsed_seconds": result.get("elapsed_seconds"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Get info about the currently loaded or specified model."""
    model_path = request.args.get("path", current_model_path or "")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": "No model loaded and no path specified"}), 400

    try:
        info = get_info(model_path)
        return jsonify({
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
        return jsonify({"error": str(e)}), 500


@app.route("/api/validate", methods=["POST"])
def api_validate():
    """Validate a quantized model against a reference."""
    data = request.get_json(force=True)
    input_path = data.get("input", "")
    reference_path = data.get("reference", "")

    if not input_path or not os.path.exists(input_path):
        return jsonify({"error": f"input file not found: {input_path}"}), 400

    try:
        result = validate(
            input_path,
            reference_path=reference_path if reference_path and os.path.exists(reference_path) else None,
        )
        return jsonify({
            "status": result.get("status"),
            "checks": result.get("checks"),
            "tensor_count": result.get("tensor_count"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """Benchmark a GGUF model."""
    data = request.get_json(force=True)
    model_path = data.get("model", current_model_path or "")
    prompt = data.get("prompt", "Hello")
    max_tokens = int(data.get("max_tokens", 10))

    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": f"model not found: {model_path}"}), 400

    try:
        result = benchmark(
            model_path,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return jsonify({
            "status": "ok",
            "tokens_per_second": result.get("tokens_per_second"),
            "generated_tokens": result.get("generated_tokens"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "output": result.get("output"),
            "architecture": result.get("architecture"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _format_chat_prompt(messages: list) -> str:
    """Format chat messages into a prompt string for Llama 3 instruct."""
    # Llama 3 chat template
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


# ─── Server Init ──────────────────────────────────────────────────────────────

def load_model_file(path: str, device: str = 'auto'):
    global model, current_model_path
    path = str(path)
    if not os.path.exists(path):
        # Try to find .gguf files
        import glob
        files = glob.glob("*.gguf") or glob.glob(os.path.join(os.path.dirname(path) or ".", "*.gguf"))
        if files:
            path = files[0]
            print(f"Auto-discovered model: {path}")
        else:
            print(f"ERROR: Model not found at {path}")
            sys.exit(1)

    print(f"Loading model: {path}")
    print(f"Device: {device}")
    t0 = time.time()
    model = LLMInference(path, device=device)
    elapsed = time.time() - t0
    current_model_path = path
    print(f"Model loaded in {elapsed:.1f}s")
    print(f"  Architecture: {model.arch}")
    print(f"  Parameters: {model.n_layers} layers, {model.n_embd} dim, {model.n_head} heads")
    if model.device.device_type.value != 'cpu':
        print(f"  Accelerator: {model.device.capability}")
    else:
        print(f"  Accelerator: CPU")


def main():
    parser = argparse.ArgumentParser(description="Mojo Hybrid Serve — LLM inference server")
    parser.add_argument("--model", "-m", default=os.environ.get("MODEL_PATH", ""),
                        help="Path to GGUF model file")
    parser.add_argument("--device", "-d", default=os.environ.get("MOJOLLAMA_DEVICE", "auto"),
                        choices=['auto', 'cpu', 'intel_arc', 'nvidia'],
                        help="Compute device: auto, cpu, intel_arc, nvidia (default: auto)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available compute devices and exit")
    parser.add_argument("--host", default=os.environ.get("HOST", DEFAULT_HOST),
                        help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", "-p", type=int,
                        default=int(os.environ.get("PORT", DEFAULT_PORT)),
                        help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if args.list_devices:
        devices = list_devices()
        print("Available compute devices:")
        for d in devices:
            print(f"  {d['device']:12s} {d['name']:40s} VRAM: {d['vram_gb']:.1f}GB  CUs: {d['compute_units']}")
        sys.exit(0)

    load_model_file(args.model, args.device)
    print(f"\n🔥 Mojo Hybrid Serve running at http://{args.host}:{args.port}")
    print(f"   API: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"   Web UI: http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
