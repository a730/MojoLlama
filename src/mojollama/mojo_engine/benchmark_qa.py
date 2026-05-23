#!/usr/bin/env python3
"""MojoLlama Q&A Benchmark — single session per model.

Each model loads weights once and answers all questions in one continuous generation.
Measures real-world latency and accuracy.
"""
import struct, subprocess, os, time, re
from transformers import AutoTokenizer

MLIB = "/root/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"
ENV = {**os.environ, 'LD_LIBRARY_PATH': MLIB,
       'OMP_PLACES': 'cores', 'OMP_PROC_BIND': 'close'}
WORK = "/onedev-workspace/work/src/mojollama/mojo_engine"

# ── 5 Questions (mix of difficulty) ──
QUESTIONS = [
    ("2+2",           ["4", "four"]),
    ("Capital of France",   ["Paris", "paris"]),
    ("Color of the sky",    ["blue", "Blue"]),
    ("3+5",           ["8", "eight"]),
    ("Speed of light",      ["299792", "300000", "3e8"]),
]

def decode_tokens(output_text: str, tokenizer, model_name: str) -> str:
    """Extract and decode response tokens from engine output."""
    # Different engines print differently
    if "tinyllama" in model_name.lower():
        # TinyLlama prints decoded text via decode_token()
        # The output contains the decoded response directly
        # Look for text after the prompt print
        lines = output_text.split('\n')
        # Find the last meaningful line
        text_parts = []
        for line in lines:
            # Lines with decoded text (no special markers)
            if line and not line.startswith('B=') and not line.startswith('Load') and not line.startswith('Q8'):
                text_parts.append(line)
        return ' '.join(text_parts)
    elif "zaya" in model_name.lower():
        # ZAYA prints decoded text via decode_token_quick()
        # The output is the decoded text directly
        lines = output_text.split('\n')
        text_parts = []
        for line in lines:
            if line and not line.startswith('ZAYA') and not line.startswith('Load') and not line.startswith('B='):
                text_parts.append(line)
        return ' '.join(text_parts)
    else:
        # GPT-OSS prints token IDs
        # Extract token IDs using regex
        tokens = re.findall(r't\s+(\d+)', output_text)
        # Try to decode
        if tokens:
            ids = [int(t) for t in tokens[:40] if int(t) < len(tokenizer)]
            if ids:
                return tokenizer.decode(ids)
        return output_text[:200]

def run_questions(model_name: str, engine_path: str, prompt_file: str,
                  tokenizer, bos_id: int):
    """Run all questions through a single engine invocation (one per question)."""
    results = []
    
    for q_text, expected in QUESTIONS:
        # Tokenize with BOS
        toks = tokenizer.encode(q_text)
        if toks[0] != bos_id:
            toks = [bos_id] + toks
        
        # Write prompt
        with open(prompt_file, 'wb') as f:
            for t in toks: f.write(struct.pack('<i', t))
        
        # Run engine (loads weights, generates response)
        start = time.time()
        r = subprocess.run([engine_path, '32'], capture_output=True,
                          text=True, timeout=120, env=ENV)
        elapsed = time.time() - start
        output = r.stdout
        
        # Decode response
        response = decode_tokens(output, tokenizer, model_name)
        correct = any(ans.lower() in response.lower() for ans in expected)
        
        results.append({
            'question': q_text,
            'expected': expected,
            'latency': elapsed,
            'response': response[:100],
            'correct': correct,
        })
        
        status = "✅" if correct else "❌"
        print(f"  {status} {q_text:25s} → {elapsed:5.2f}s  {response[:60]}")
    
    return results


if __name__ == '__main__':
    print("="*65)
    print("  MojoLlama Q&A Benchmark — Realistic Inference Simulation")
    print("="*65)
    print(f"  Questions: {len(QUESTIONS)}  |  Threadripper 3970X | Q8.0 weights")
    print()
    
    all_results = {}
    
    # ── 1. TinyLlama Q8_0 ──
    print("[1/3] TinyLlama Q8_0 (B=4)...")
    tl_tok = AutoTokenizer.from_pretrained('/tmp/tinyllama-tokenizer', trust_remote_code=True)
    # Llama tokenizer adds BOS with add_special_tokens
    class TLTokenizer:
        def encode(self, text):
            return tl_tok.encode(text, add_special_tokens=True)
        def decode(self, ids):
            return tl_tok.decode(ids)
        def __len__(self):
            return len(tl_tok)
    tl = TLTokenizer()
    
    tl_results = run_questions("tinyllama",
        f"{WORK}/tinyllama_gen_q8", "/tmp/prompt_tinyllama.bin",
        tl, bos_id=1)
    all_results['TinyLlama'] = tl_results
    
    # ── 2. ZAYA1-8B Q8_0 ──
    print("\n[2/3] ZAYA1-8B Q8_0 (B=4)...")
    qw = AutoTokenizer.from_pretrained('/tmp/qwen3-tokenizer', trust_remote_code=True)
    class QWTokenizer:
        def encode(self, text):
            return [2] + qw.encode(text, add_special_tokens=False)
        def decode(self, ids):
            return qw.decode(ids)
        def __len__(self):
            return len(qw)
    zt = QWTokenizer()
    
    zaya_results = run_questions("zaya",
        f"{WORK}/zaya_gen_q8", "/tmp/prompt_zaya.bin",
        zt, bos_id=2)
    all_results['ZAYA1-8B'] = zaya_results
    
    # ── 3. GPT-OSS-20B Q8_0 ──
    print("\n[3/3] GPT-OSS-20B Q8_0 (B=1)...")
    gt = QWTokenizer()  # reuses Qwen tokenizer
    gt.encode = lambda text: [199998] + qw.encode(text, add_special_tokens=False)
    
    gptoss_results = run_questions("gptoss",
        f"{WORK}/gptoss_gen_q8", "/tmp/prompt_gptoss.bin",
        gt, bos_id=199998)
    all_results['GPT-OSS-20B'] = gptoss_results
    
    # ── Summary Table ──
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    header = f"  {'Model':<18s} {'Acc':>6s} {'Avg Lat':>9s} {'Min':>6s} {'Max':>6s} {'Q/sec':>7s}"
    print(header)
    print(f"  {'─'*18} {'─'*6} {'─'*9} {'─'*6} {'─'*6} {'─'*7}")
    
    for name, results in all_results.items():
        correct = sum(1 for r in results if r['correct'])
        lats = [r['latency'] for r in results]
        avg_lat = sum(lats) / len(lats)
        qps = len(lats) / sum(lats)
        acc_pct = 100 * correct / len(results)
        print(f"  {name:<18s} {acc_pct:5.0f}%  {avg_lat:7.2f}s  {min(lats):5.2f}  {max(lats):5.2f}  {qps:6.2f}")
