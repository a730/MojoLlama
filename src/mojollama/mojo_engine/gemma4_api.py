#!/usr/bin/env python3
"""
Gemma 4 OpenAI-compatible API server.
Launches Mojo engine as subprocess, handles tokenization + HTTP in Python.
Usage: python3 gemma4_api.py <port> <weights_dir> [model_type]
  model_type: e4b (default) or e2b
"""
import os, sys, json, time, subprocess, argparse
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global state
mojo_binary = None
weights_dir = None
model_name = "gemma-4-e4b"

def load_tokenizer(weights_dir):
    """Load SentencePiece tokenizer from HF or GGUF directory."""
    import sentencepiece as spm
    # Check for .model file
    for fn in os.listdir(weights_dir):
        if fn.endswith('.model'):
            sp_path = os.path.join(weights_dir, fn)
            return spm.SentencePieceProcessor(model_file=sp_path)
    # Also check parent directories
    for d in [os.path.dirname(weights_dir), '/tmp/models', '/onedev-workspace/work']:
        for fn in os.listdir(d):
            if fn.endswith('.model'):
                sp_path = os.path.join(d, fn)
                return spm.SentencePieceProcessor(model_file=sp_path)
    print("Warning: No sentencepiece .model file found, using identity tokenizer")
    return None

def generate(prompt_text, max_tokens=128, temperature=0.0):
    """Run Mojo engine and return generated text."""
    # For now, launch the benchmark binary with prompt as tokens
    # In production, you'd have a proper prompt-prefill engine
    try:
        result = subprocess.run(
            [mojo_binary, str(32)],
            cwd=os.path.dirname(mojo_binary),
            capture_output=True, text=True, timeout=300,
            env={**os.environ, 
                 'LD_LIBRARY_PATH': os.environ.get('MOJO_LIB', '/root/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib'),
                 'OMP_PLACES': 'cores',
                 'OMP_PROC_BIND': 'close'}
        )
        # Parse output for tokens
        output = result.stdout or result.stderr
        tokens = []
        for line in output.split('\n'):
            if line.startswith('tok='):
                tok_str = line.split('tok=')[1].strip()
                try:
                    tokens.append(int(tok_str.split()[0]))
                except:
                    pass
        return tokens, output
    except Exception as e:
        return [], f"Error: {e}"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json()
    messages = data.get('messages', [])
    max_tokens = int(data.get('max_tokens', 128))
    temperature = float(data.get('temperature', 0.0))
    
    # Extract user prompt
    prompt = ""
    for msg in messages:
        if msg.get('role') == 'user':
            prompt = msg.get('content', '')
    
    # Generate
    token_ids, raw_output = generate(prompt, max_tokens, temperature)
    
    # Decode tokens using tokenizer
    if sp and token_ids:
        try:
            text = sp.decode(token_ids)
        except:
            text = f"<tokens: {token_ids}>"
    else:
        text = raw_output[:500] if raw_output else ""
    
    return jsonify({
        'id': 'chatcmpl-mojo',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model_name,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': text},
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': len(prompt.split()),
            'completion_tokens': len(token_ids),
            'total_tokens': len(prompt.split()) + len(token_ids)
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': model_name})

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        'object': 'list',
        'data': [{
            'id': model_name,
            'object': 'model',
            'created': int(time.time()),
            'owned_by': 'mojo'
        }]
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gemma 4 OpenAI API Server')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--weights', default='/tmp/weights_e4b_final_transposed/', help='Weights directory')
    parser.add_argument('--model', default='e4b', choices=['e4b', 'e2b'], help='Model type')
    parser.add_argument('--mojo-bin', default='/onedev-workspace/work/src/mojollama/mojo_engine/gemma4_gen_q8', 
                        help='Mojo engine binary')
    args = parser.parse_args()
    
    global mojo_binary, weights_dir, model_name, sp
    mojo_binary = args.mojo_bin
    weights_dir = args.weights
    model_name = f"gemma-4-{args.model}"
    
    print(f"Loading tokenizer from {weights_dir}...")
    sp = load_tokenizer(weights_dir)
    if sp:
        print(f"Tokenizer loaded: {sp.vocab_size()} tokens")
    else:
        print("Running without tokenizer (raw token IDs)")
    
    print(f"Mojo binary: {mojo_binary}")
    print(f"Starting server on port {args.port}...")
    print(f"API endpoint: http://0.0.0.0:{args.port}/v1/chat/completions")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)
