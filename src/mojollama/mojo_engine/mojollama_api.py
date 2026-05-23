#!/usr/bin/env python3
"""MojoLlama Chat API — Python tokenizer + Mojo engine subprocess.
   Architecture:
     Client → Flask API (port 8080) → subprocess Mojo engine → tokens → detokenize → respond

Usage:
  python3 mojollama_api.py --port 8080 --weights /tmp/weights_e4b_final_transposed/ --nw 32
"""
import os, sys, json, time, struct, subprocess, argparse
from flask import Flask, request, jsonify

app = Flask(__name__)
engine_bin = None
weights_dir = None
nw = 32
model_name = "mojollama"
sp = None

def find_tokenizer(wdir):
    """Find sentencepiece model file."""
    for root in [wdir, os.path.dirname(wdir), '/tmp/models', '/tmp']:
        if os.path.isdir(root):
            for fn in os.listdir(root):
                if fn.endswith('.model'):
                    return os.path.join(root, fn)
    # Also check hf_cache
    import glob
    for f in glob.glob('/tmp/hf_cache/**/*.model', recursive=True):
        return f
    return None

def tokenize(text):
    """Tokenize text. Returns list of token IDs."""
    if sp is None:
        return [2] + [ord(c) for c in text[:64]]  # BOS + byte fallback
    return sp.encode(text, add_bos=True, out_type=int)

def detokenize(ids):
    """Decode token IDs to text."""
    if sp is None:
        # Try to decode as bytes
        chars = []
        for tid in ids:
            if tid < 256: chars.append(chr(tid))
        return ''.join(chars)
    return sp.decode(ids)

def run_inference(prompt_tokens, max_tokens=128):
    """Run Mojo engine as subprocess with prompt tokens on stdin."""
    # Write prompt tokens to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        prompt_path = f.name
        f.write(struct.pack(f'<{len(prompt_tokens)}i', *prompt_tokens))
    
    output_path = prompt_path.replace('.bin', '_out.bin')
    
    env = os.environ.copy()
    env['OMP_PLACES'] = 'cores'
    env['OMP_PROC_BIND'] = 'close'
    env['MOJO_PROMPT_FILE'] = prompt_path
    env['MOJO_OUTPUT_FILE'] = output_path
    env['MOJO_MAX_TOKENS'] = str(max_tokens)
    
    try:
        result = subprocess.run(
            [engine_bin, '0', str(nw), weights_dir],
            env=env, capture_output=True, text=True, timeout=300
        )
        
        # Read output tokens
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                data = f.read()
            count = len(data) // 4
            tokens = list(struct.unpack(f'<{count}i', data))
            os.unlink(output_path)
            os.unlink(prompt_path)
            return tokens
    except:
        pass
    
    # Cleanup
    if os.path.exists(prompt_path): os.unlink(prompt_path)
    if os.path.exists(output_path): os.unlink(output_path)
    return [2]

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json()
    messages = data.get('messages', [])
    max_tokens = int(data.get('max_tokens', 128))
    
    prompt = ""
    for msg in messages:
        if msg.get('role') == 'user':
            prompt = msg.get('content', '')
    
    t0 = time.time()
    prompt_tokens = tokenize(prompt)
    output_tokens = run_inference(prompt_tokens, max_tokens)
    gen_tokens = output_tokens[:max_tokens]
    text = detokenize(gen_tokens)
    elapsed = time.time() - t0
    
    return jsonify({
        'id': f'chatcmpl-{int(time.time())}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model_name,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': text},
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': len(prompt_tokens),
            'completion_tokens': len(gen_tokens),
            'total_tokens': len(prompt_tokens) + len(gen_tokens)
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': model_name})

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        'object': 'list',
        'data': [{'id': model_name, 'object': 'model', 'created': int(time.time()), 'owned_by': 'mojo'}]
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MojoLlama Chat API')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--weights', default='/tmp/weights_e4b_final_transposed/')
    parser.add_argument('--engine', default='/onedev-workspace/work/src/mojollama/mojo_engine/universal_engine')
    parser.add_argument('--nw', type=int, default=32)
    parser.add_argument('--model', default='mojollama')
    args = parser.parse_args()
    
    global engine_bin, weights_dir, nw, model_name, sp
    engine_bin = args.engine
    weights_dir = args.weights
    nw = args.nw
    model_name = args.model
    
    tok_path = find_tokenizer(weights_dir)
    if tok_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=tok_path)
        print(f"Tokenizer: {sp.vocab_size()} tokens ({tok_path})")
    else:
        print("WARNING: No tokenizer - using byte fallback")
        sp = None
    
    print(f"Engine: {engine_bin}")
    print(f"Server: http://0.0.0.0:{args.port}/v1/chat/completions")
    app.run(host='0.0.0.0', port=args.port, debug=False)
