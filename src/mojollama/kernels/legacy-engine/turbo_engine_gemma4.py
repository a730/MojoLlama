#!/usr/bin/env python3
"""TurboEngine for Gemma 4 — per-layer projections, GeGLU, sliding window, Q/K norms."""
import sys, os, time, ctypes, numpy as np
import gguf
from gguf.constants import GGMLQuantizationType as QT

GGML_Q4_0 = 2; GGML_Q8_0 = 8; GGML_Q4_K = 12; GGML_Q5_K = 13; GGML_Q6_K = 14
C_KERNEL_TYPES = {GGML_Q4_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0}

# Load C kernels
_kern_dir = os.path.dirname(os.path.abspath(__file__))
_kern = None
for so_name in ['quant_kernels_omp.so']:
    p = os.path.join(_kern_dir, so_name)
    if os.path.exists(p):
        _kern = ctypes.CDLL(p)
        break

_simd = None
for so_name in ['simd_ops.so']:
    p = os.path.join(_kern_dir, so_name)
    if os.path.exists(p):
        _simd = ctypes.CDLL(p)
        break

class TurboEngineGemma4:
    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        self.n_threads = n_threads
        self.model_path = model_path
        
        t0 = time.perf_counter()
        self.reader = gguf.GGUFReader(model_path)
        self._parse_metadata()
        self._load_weights()
        self._init_buffers()
        self._setup_pointers()
        self.pos = 0
        print(f"TurboEngine Gemma4: {self.n_layers}L/{self.n_embd}D/{self.n_head}H/{self.n_kv_head}KV | t={n_threads} | vocab={self.vocab_size}", flush=True)
    
    def _parse_metadata(self):
        f = self.reader.fields
        def g(name):
            for k, v in f.items():
                if k == name and hasattr(v, 'parts') and v.parts:
                    d = v.parts[-1]
                    if hasattr(d, '__iter__') and len(d) == 1: return int(d[0])
                    if hasattr(d, 'tolist'): return bytes(d.tolist()).decode('utf-8')
                    return d
        self.arch = g('general.architecture') or 'gemma4'
        a = self.arch
        self.n_layers = int(g(f'{a}.block_count') or 35)
        self.n_embd = int(g(f'{a}.embedding_length') or 1536)
        self.n_ff = int(g(f'{a}.feed_forward_length') or 12288)
        self.n_head = int(g(f'{a}.attention.head_count') or 8)
        self.n_kv_head = int(g(f'{a}.attention.head_count_kv') or 1)
        self.head_dim = int(g(f'{a}.attention.key_length') or (self.n_embd // self.n_head))
        self.head_dim_v = int(g(f'{a}.attention.value_length') or self.head_dim)
        self.rope_dim = int(g(f'{a}.rope.dimension_count') or self.head_dim)
        self.rope_freq_base = float(g(f'{a}.rope.freq_base') or 10000.0)
        self.eps = float(g(f'{a}.attention.layer_norm_rms_epsilon') or 1e-6)
        self.vocab_size = int(g(f'{a}.vocab_size') or 262144)
        self.emb_per_layer = int(g(f'{a}.embedding_length_per_layer_input') or 256)
        self.shared_kv = int(g(f'{a}.attention.shared_kv_layers') or 20)
        self.sliding_window = int(g(f'{a}.attention.sliding_window') or 512)
        self.logit_cap = float(g(f'{a}.final_logit_softcapping') or 30.0)
    
    def _matmul(self, w_raw, qt, x, out, n_rows, n_cols):
        """C quant matmul: w[n_rows, n_cols] * x[n_cols] -> out[n_rows]"""
        if _kern is not None:
            _kern.quant_matmul_omp(
                ctypes.cast(w_raw, ctypes.POINTER(ctypes.c_uint8)),
                ctypes.cast(x.ctypes.data, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(out.ctypes.data, ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n_rows), ctypes.c_int(n_cols), ctypes.c_int(qt))
        else:
            out[:] = 0  # fallback
    
    def _load_weights(self):
        """Load weights: quantized tensors stay raw for C matmul; F32 norms."""
        self.weights = {}; self.raw = {}
        for t in self.reader.tensors:
            name = t.name; qtype = QT(t.tensor_type).value
            shape = tuple(int(s) for s in t.shape)
            if qtype in C_KERNEL_TYPES and len(shape) >= 2:
                raw = np.ascontiguousarray(t.data.reshape(-1), dtype=np.uint8)
                self.raw[name] = (raw, qtype, shape)
            else:
                f32 = gguf.dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
                if len(shape) == 2: f32 = f32.reshape(shape[1], shape[0])
                elif len(shape) == 1: f32 = f32.reshape(-1)
                self.weights[name] = np.ascontiguousarray(f32)
        
        # Embedding
        self.emb = self.weights.get('token_embd.weight', np.zeros((self.n_embd, self.vocab_size)))
        self.output_w = self.weights.get('output.weight', None)
        self.output_norm_w = self.weights.get('output_norm.weight', np.ones(self.n_embd))
        
        # Per-layer projections
        self.per_layer_proj = self.weights.get('per_layer_model_proj.weight',
            np.zeros((self.n_layers * self.emb_per_layer, self.n_embd)))
        self.per_layer_tok_embd = self.weights.get('per_layer_token_embd.weight',
            np.zeros((self.n_layers * self.emb_per_layer, self.vocab_size)))
        
        # Rope freqs (precomputed)
        self.rope_freqs = self.weights.get('rope_freqs.weight', None)
    
    def _init_buffers(self):
        N = self.n_embd; PL = self.emb_per_layer; NH = self.n_head
        NKH = self.n_kv_head; HD = self.head_dim; FF = self.n_ff
        MAX_CTX = 4096
        
        self.kv_k = np.zeros((self.n_layers, MAX_CTX, NKH * HD), dtype=np.float32)
        self.kv_v = np.zeros((self.n_layers, MAX_CTX, NKH * HD_v), dtype=np.float32) if hasattr(self, 'head_dim_v') and self.head_dim_v != HD else self.kv_k.copy()
        self.kv_len = np.zeros(self.n_layers, dtype=np.int32)
        
        self._x = np.zeros(N, dtype=np.float32)          # Full embedding
        self._xl = np.zeros(PL, dtype=np.float32)        # Per-layer input
        self._xn = np.zeros(PL, dtype=np.float32)        # Normalized
        self._xr = np.zeros(PL, dtype=np.float32)        # Residual
        self._q = np.zeros(NH * HD, dtype=np.float32)    # Q
        self._k = np.zeros(NKH * HD, dtype=np.float32)   # K
        self._v = np.zeros(NKH * HD, dtype=np.float32)   # V
        self._att = np.zeros(NH * HD, dtype=np.float32)  # Attention output
        self._gate = np.zeros(FF, dtype=np.float32)      # FFN gate
        self._up = np.zeros(FF, dtype=np.float32)        # FFN up
        self._ffn = np.zeros(PL, dtype=np.float32)       # FFN output
        self._oproj = np.zeros(N, dtype=np.float32)      # Full output projection
        self._logits = np.zeros(self.vocab_size, dtype=np.float32)
    
    def _setup_pointers(self):
        """Wire layer weights into list for fast access."""
        self._layers = []
        for i in range(self.n_layers):
            p = f'blk.{i}'
            lw = {}
            
            def _get_raw(suffix):
                name = f'{p}.{suffix}.weight'
                if name in self.raw:
                    return self.raw[name][0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.raw[name][1]
                return None, 0
            
            def _get_w(suffix):
                name = f'{p}.{suffix}.weight'
                return self.weights.get(name)
            
            lw['q_raw'], lw['q_qt'] = _get_raw('attn_q')
            lw['k_raw'], lw['k_qt'] = _get_raw('attn_k')
            lw['v_raw'], lw['v_qt'] = _get_raw('attn_v')
            lw['o_raw'], lw['o_qt'] = _get_raw('attn_output')
            lw['g_raw'], lw['g_qt'] = _get_raw('ffn_gate')
            lw['u_raw'], lw['u_qt'] = _get_raw('ffn_up')
            lw['d_raw'], lw['d_qt'] = _get_raw('ffn_down')
            
            lw['ann'] = _get_w('attn_norm')
            lw['q_norm'] = _get_w('attn_q_norm')
            lw['k_norm'] = _get_w('attn_k_norm')
            lw['pan'] = _get_w('post_attention_norm')
            lw['ffn_norm'] = _get_w('ffn_norm')
            lw['pfn'] = _get_w('post_ffw_norm')
            lw['inp_gate'] = _get_w('inp_gate')
            lw['out_scale'] = _get_w('layer_output_scale')
            lw['post_norm'] = _get_w('post_norm')
            lw['proj'] = _get_w('proj')
            
            # Q/K dims
            lw['nq'] = self.n_head * self.head_dim
            lw['nk'] = self.n_kv_head * self.head_dim
            lw['nv'] = self.n_kv_head * (self.head_dim_v if hasattr(self, 'head_dim_v') else self.head_dim)
            
            self._layers.append(lw)
    
    def _geglu(self, gate, up):
        """GeGLU: gate * gelu(up)"""
        return gate * (0.5 * up * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (up + 0.044715 * up**3))))
    
    def _rms(self, x, w):
        """RMS norm"""
        ss = np.mean(x * x) + self.eps
        return x / np.sqrt(ss) * w
    
    def _apply_rope(self, x, pos, n_heads):
        """Apply RoPE with precomputed freqs or compute on the fly."""
        hd = self.head_dim
        rope_d = self.rope_dim
        x2d = x.reshape(n_heads, hd)
        
        if self.rope_freqs is not None:
            # Precomputed freqs [n_heads, rope_dim]
            freqs = self.rope_freqs.reshape(n_heads, rope_d)
            cos_a = np.cos(pos * freqs[:, :rope_d//2])
            sin_a = np.sin(pos * freqs[:, :rope_d//2])
        else:
            t = np.arange(0, rope_d, 2, dtype=np.float32)
            inv_freq = 1.0 / (self.rope_freq_base ** (t / rope_d))
            freqs = pos * inv_freq
            cos_a = np.cos(freqs); sin_a = np.sin(freqs)
        
        half = rope_d // 2
        out = x2d.copy()
        out[:, :half] = x2d[:, :half] * cos_a - x2d[:, half:rope_d] * sin_a
        out[:, half:rope_d] = x2d[:, half:rope_d] * cos_a + x2d[:, :half] * sin_a
        return out.reshape(-1)
    
    def _gqa(self, q, k_cache, v_cache, seq_len):
        """Grouped query attention."""
        nh = self.n_head; nkh = self.n_kv_head; hd = self.head_dim
        gqa = nh // nkh
        q_2d = q.reshape(nh, hd)
        q_g = q_2d.reshape(nkh, gqa, hd)
        
        # Scores: [nkh, gqa, seq_len]
        k_T = k_cache[:seq_len].T  # [hd, seq_len]
        scores = np.dot(q_g.reshape(nh, hd), k_T) / np.sqrt(hd)  # [nh, seq_len]
        scores -= np.max(scores, axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)
        
        # Weighted sum: [nh, hd]
        v_T = v_cache[:seq_len].T  # [hd, seq_len]
        att = np.dot(scores, v_cache[:seq_len])  # [nh, hd]
        return att.reshape(-1)
    
    def _logit_softcap(self, logits, cap=30.0):
        """Logit softcapping: cap * tanh(x/cap)"""
        return cap * np.tanh(logits / cap)
    
    def forward(self, token_id):
        """Gemma 4 forward pass."""
        L = self.n_layers; N = self.n_embd; PL = self.emb_per_layer
        NH = self.n_head; NKH = self.n_kv_head; HD = self.head_dim
        FF = self.n_ff; pos = self.pos
        
        # 1. Embedding: get full embedding
        np.copyto(self._x, self.emb[:, token_id[0,0]])
        
        # 2. Per-layer projection and bias
        # per_layer_proj: [L*PL, N], per_layer_tok_embd: [L*PL, vocab]
        # We can compute all layers' inputs at once
        for i in range(L):
            off = i * PL
            proj = self.per_layer_proj[off:off+PL]  # [PL, N]
            np.copyto(self._xl, proj @ self._x)
        
        # 3. Layer loop
        for i in range(L):
            lw = self._layers[i]
            PL = self.emb_per_layer
            
            # Input gate
            if lw['inp_gate'] is not None:
                self._xl *= lw['inp_gate']
            
            # Residual copy
            np.copyto(self._xr, self._xl)
            np.clip(self._xl, -1000.0, 1000.0, out=self._xl)
            
            # Pre-attention RMS
            if lw['ann'] is not None:
                self._xn[:] = self._rms(self._xl, lw['ann'])
            else:
                np.copyto(self._xn, self._xl)
            
            # Q projection
            ikv = (i // self.shared_kv) * self.shared_kv  # shared KV layer index
            lw_kv = self._layers[ikv]
            
            if lw['q_raw'] is not None:
                self._matmul(lw['q_raw'], lw['q_qt'], self._xn, self._q, lw['nq'], PL)
            if lw_kv['k_raw'] is not None:
                self._matmul(lw_kv['k_raw'], lw_kv['k_qt'], self._xn, self._k, lw_kv['nk'], PL)
            if lw_kv['v_raw'] is not None:
                self._matmul(lw_kv['v_raw'], lw_kv['v_qt'], self._xn, self._v, lw_kv['nv'], PL)
            
            # Q/K normalization
            if lw['q_norm'] is not None:
                self._q[:] = self._rms(self._q, lw['q_norm'])
            if lw['k_norm'] is not None:
                self._k[:] = self._rms(self._k, lw['k_norm'])
            
            # RoPE
            self._q[:] = self._apply_rope(self._q, pos, NH)
            self._k[:] = self._apply_rope(self._k, pos, NKH)
            
            # KV cache (sliding window)
            self.kv_k[ikv, pos % self.sliding_window] = self._k[:]
            self.kv_v[ikv, pos % self.sliding_window] = self._v[:]
            seq_len = min(self.kv_len[ikv] + 1, self.sliding_window)
            
            # GQA attention
            # Build full cache with sliding window
            if self.kv_len[ikv] < self.sliding_window:
                k_full = self.kv_k[ikv, :pos+1]
                v_full = self.kv_v[ikv, :pos+1]
            else:
                # Reorder sliding window to chronological
                start = (pos + 1) % self.sliding_window
                idx = np.arange(start, self.sliding_window)
                idx = np.append(idx, np.arange(start))
                k_full = self.kv_k[ikv, idx[:self.sliding_window]]
                v_full = self.kv_v[ikv, idx[:self.sliding_window]]
            
            self._att[:] = self._gqa(self._q, k_full, v_full, seq_len)
            self.kv_len[ikv] += 1
            
            # Output projection
            if lw['o_raw'] is not None:
                self._matmul(lw['o_raw'], lw['o_qt'], self._att, self._oproj[:PL], PL, lw['nq'])
            
            # Post-attention norm + residual
            if lw['pan'] is not None:
                self._xl[:] = self._xr[:] + self._rms(self._oproj[:PL], lw['pan'])
            else:
                self._xl[:] = self._xr[:] + self._oproj[:PL]
            np.clip(self._xl, -1000.0, 1000.0, out=self._xl)
            
            # Pre-FFN RMS
            np.copyto(self._xr, self._xl)
            if lw['ffn_norm'] is not None:
                self._xn[:] = self._rms(self._xl, lw['ffn_norm'])
            else:
                np.copyto(self._xn, self._xl)
            
            # FFN (GeGLU: gate * gelu(up))
            if lw['g_raw'] is not None and lw['u_raw'] is not None:
                self._matmul(lw['g_raw'], lw['g_qt'], self._xn, self._gate, FF, PL)
                self._matmul(lw['u_raw'], lw['u_qt'], self._xn, self._up, FF, PL)
                self._ffn[:] = self._geglu(self._gate[:FF], self._up[:FF])
            
            # FFN down
            if lw['d_raw'] is not None:
                self._matmul(lw['d_raw'], lw['d_qt'], self._ffn, self._oproj[:PL], PL, FF)
            
            # Post-FFN norm + residual
            if lw['pfn'] is not None:
                self._xl[:] = self._xr[:] + self._rms(self._oproj[:PL], lw['pfn'])
            else:
                self._xl[:] = self._xr[:] + self._oproj[:PL]
            
            # Layer output scale
            if lw['out_scale'] is not None:
                self._xl *= lw['out_scale'][0]
            
            # Post-norm
            if lw['post_norm'] is not None:
                self._xl[:] = self._rms(self._xl, lw['post_norm'])
            
            # Output projection back to full embedding
            if lw['proj'] is not None:
                self._oproj[:] = lw['proj'] @ self._xl
        
        # Final norm
        self._xn[:] = self._rms(self._xl, self.output_norm_w)
        
        # Output projection (logits)
        if self.output_w is not None:
            self._logits[:] = self.output_w @ self._xn
        else:
            # Use embedding
            self._logits[:] = self.emb.T @ self._xn
        
        # Logit softcapping
        self._logits[:] = self._logit_softcap(self._logits, self.logit_cap)
        
        self.pos += 1
        return self._logits

if __name__ == '__main__':
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else '/onedev-workspace/work/gemma-4-E2B-it-Q4_K_M.gguf'
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    e = TurboEngineGemma4(model, t)
    logits = e.forward(np.array([[1]], dtype=np.int32))
    print(f"OK: max={logits.max():.1f}, nan={np.isnan(logits).sum()}", flush=True)
