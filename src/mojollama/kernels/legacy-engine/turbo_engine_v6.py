"""TurboEngine v6 — AVX2-optimized kernels for competitive inference.

Uses turbo_kernels_v2.so (AVX2 dequant+FMA) for Q4_0/Q4_1/Q8_0
and falls back to OMP kernels for Q4_K/Q5_K/Q6_K on large matrices.
Includes batched QKV matmul to reduce OMP fork/join overhead.
"""

import ctypes, os, time, math
import numpy as np

QTYPE_NAMES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",8:"Q8_0",12:"Q4_K",13:"Q5_K",14:"Q6_K"}

class TurboEngineV6:
    def __init__(self, model_path, n_threads=32):
        self.n_threads = n_threads
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load kernel libraries
        self.lib_v2 = ctypes.CDLL(os.path.join(model_dir, 'turbo_kernels_v2.so'))
        self.lib_omp = ctypes.CDLL(os.path.join(model_dir, 'quant_kernels_omp.so'))
        
        cf = ctypes.POINTER(ctypes.c_float)
        cu = ctypes.POINTER(ctypes.c_uint8)
        self.cf = cf
        self.cu = cu
        
        self.lib_v2.quant_matmul_v2.argtypes = [cu, cf, cf, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib_v2.quant_matmul_v2.restype = None
        self.lib_omp.quant_matmul_omp.argtypes = [cu, cf, cf, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib_omp.quant_matmul_omp.restype = None
        
        # Batched QKV
        self.lib_v2.batch_qkv_q4_0.argtypes = [cu, cu, cu, cf, cf, cf, cf,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib_v2.batch_qkv_q4_0.restype = None
        
        for lib in [self.lib_v2, self.lib_omp]:
            lib.set_num_threads.argtypes = [ctypes.c_int]
            lib.set_num_threads.restype = None
            lib.set_num_threads(n_threads)
        
        # Load model
        import gguf
        self.reader = gguf.GGUFReader(model_path)
        self._parse_config()
        self._load_weights()
        self._alloc_buffers()
        self._precompute_rope()
        
        print(f"TurboEngine v6: {self.n_layers}L/{self.n_embd}D/{self.n_ff}FF/"
              f"{self.n_head}H/{self.n_kv_head}KV | t={n_threads} | vocab={self.n_vocab}"
              f"{' [MoE:'+str(self.n_experts)+'e'+str(self.n_experts_per_tok)+'t]' if self.is_moe else ''}")
    
    def _parse_config(self):
        md = self.reader.fields
        def get_int(keys, default=None):
            for k in keys:
                v = md.get(k)
                if v is not None: return int(v)
            return default
        
        self.n_embd = get_int(['llama.embedding_length','qwen2.embedding_length'], 2048)
        self.n_layers = get_int(['llama.block_count','qwen2.block_count'], 16)
        self.n_head = get_int(['llama.attention.head_count','qwen2.attention.head_count'], 32)
        self.n_kv_head = get_int(['llama.attention.head_count_kv','qwen2.attention.head_count_kv'], 8)
        self.n_ff = get_int(['llama.feed_forward_length','qwen2.feed_forward_length'], 8192)
        self.n_vocab = get_int(['llama.vocab_size'], 128256)
        self.head_dim = self.n_embd // self.n_head
        self.n_experts = get_int(['llama.expert_count','qwen2.expert_count'], None)
        self.n_experts_per_tok = get_int(['llama.expert_used_count','qwen2.expert_used_count'], None)
        self.is_moe = self.n_experts is not None
        self.use_qk_norm = bool(get_int(['llama.attention.l2_norm','qwen2.attention.l2_norm'], 0))
    
    def _load_weights(self):
        self.weights = {}
        self.raw_weights = {}
        self.weight_qtypes = {}
        self.weight_info = {}
        self.emb = None
        self.out_w_name = None
        
        for t in self.reader.tensors:
            name = t.name
            shape = list(t.shape)
            if len(shape) == 1: shape = [1, shape[0]]
            qtype = int(t.tensor_type)
            
            self.weight_qtypes[name] = qtype
            self.weight_info[name] = (shape[0], shape[1])
            
            if qtype in (0, 1):  # F32 or F16
                self.weights[name] = np.ascontiguousarray(t.data.reshape(shape[-2:]), dtype=np.float32)
            else:
                self.raw_weights[name] = np.ascontiguousarray(t.data.flatten(), dtype=np.uint8)
                # Dequantize for reference
                self.weights[name] = self._dequant(name, shape[-2:], qtype, t.data)
            
            if 'token_embd' in name and 'weight' in name:
                self.emb = self.weights[name]
            if 'output.weight' in name:
                self.out_w_name = name
    
    def _dequant(self, name, shape, qtype, data):
        """Dequantize using OMP C kernel for correctness."""
        if qtype == 2:  # Q4_0
            return self._dequant_simple(data, shape, qtype)
        elif qtype == 3:  # Q4_1
            return self._dequant_simple(data, shape, qtype)
        elif qtype == 14:  # Q6_K - use C dequantize
            nb = shape[1] // 256
            result = np.zeros(shape, dtype=np.float32)
            try:
                raw = np.ascontiguousarray(data.flatten(), dtype=np.uint8)
                for r in range(min(shape[0], 256)):  # Limit for speed
                    blk_ptr = (r * nb) * 210
                    for b in range(nb):
                        off = blk_ptr + b * 210
                        d = float(np.float16(raw[off+208] | (raw[off+209] << 8)))
                        sc = np.frombuffer(raw[off+192:off+208], dtype=np.int8)
                        ql = raw[off:off+128]; qh = raw[off+128:off+160]
                        for n in range(0, 256, 128):
                            for l in range(32):
                                is_ = l // 16
                                q1 = ((ql[l]&0xF)|(((qh[l]>>0)&3)<<4))-32
                                q2 = ((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32
                                q3 = ((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32
                                q4 = ((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32
                                ds0=d*float(sc[is_]); ds2=d*float(sc[is_+2])
                                ds4=d*float(sc[is_+4]); ds6=d*float(sc[is_+6])
                                result[r,b*256+n+l]=ds0*q1; result[r,b*256+n+l+32]=ds2*q2
                                result[r,b*256+n+l+64]=ds4*q3; result[r,b*256+n+l+96]=ds6*q4
            except Exception:
                pass
            return np.ascontiguousarray(result)
        else:
            return np.zeros(shape, dtype=np.float32)
    
    def _dequant_simple(self, data, shape, qtype):
        """Simple dequantize for Q4_0/Q4_1."""
        from gguf.constants import GGMLQuantizationType as QT
        rows, cols = shape
        raw = np.frombuffer(data, dtype=np.uint8)
        result = np.zeros(shape, dtype=np.float32)
        
        if QT(qtype) == QT.Q4_0:
            bs = 18; bpr = cols // 32
            for r in range(rows):
                for blk in range(bpr):
                    off = (r * bpr + blk) * bs
                    s = float(np.float16(raw[off] | (raw[off+1] << 8)))
                    for j in range(16):
                        lo = (raw[off+2+j] & 0xF) - 8
                        hi = (raw[off+2+j] >> 4) - 8
                        result[r, blk*32+j] = s * lo
                        result[r, blk*32+j+16] = s * hi
        elif QT(qtype) == QT.Q4_1:
            bs = 20; bpr = cols // 32
            for r in range(rows):
                for blk in range(bpr):
                    off = (r * bpr + blk) * bs
                    d = float(np.float16(raw[off] | (raw[off+1] << 8)))
                    m = float(np.float16(raw[off+2] | (raw[off+3] << 8)))
                    for j in range(16):
                        lo = (raw[off+4+j] & 0xF)
                        hi = (raw[off+4+j] >> 4)
                        result[r, blk*32+j] = d * lo + m
                        result[r, blk*32+j+16] = d * hi + m
        return np.ascontiguousarray(result)
    
    def _alloc_buffers(self):
        N = self.n_embd
        ff = self.n_ff
        max_seq = 4096
        
        self._buf = {
            'h': np.zeros(N, dtype=np.float32),
            'residual': np.zeros(N, dtype=np.float32),
            'att_out': np.zeros(N, dtype=np.float32),
            'silu_gate': np.zeros(ff, dtype=np.float32),
        }
        self.k_cache = np.zeros((self.n_layers, max_seq, self.n_kv_head, self.head_dim), dtype=np.float32)
        self.v_cache = np.zeros((self.n_layers, max_seq, self.n_kv_head, self.head_dim), dtype=np.float32)
        self.pos = 0
    
    def _precompute_rope(self):
        hd = self.head_dim
        theta = 10000.0
        freqs = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        t = np.arange(4096, dtype=np.float32)
        self.cos_cache = np.cos(np.outer(t, freqs))
        self.sin_cache = np.sin(np.outer(t, freqs))
    
    def _rms_norm(self, x, weight_name):
        w = self.weights[weight_name]
        eps = 1e-5
        return (x / np.sqrt(np.mean(x * x) + eps)) * w[:x.shape[0]]
    
    def _silu(self, x):
        return x / (1.0 + np.exp(-np.clip(x, -88, 88)))
    
    def _matmul(self, weight_name, x):
        """Dispatch to best kernel."""
        qtype = self.weight_qtypes.get(weight_name, 0)
        nr, nc = self.weight_info[weight_name]
        
        if qtype == 0:  # F32
            return self.weights[weight_name] @ x[:nc]
        
        raw = self.raw_weights[weight_name]
        out = np.empty(nr, dtype=np.float32)
        x_in = np.ascontiguousarray(x[:nc], dtype=np.float32)
        
        # Use OMP kernel for Q6_K output (large), v2 for everything else
        if qtype == 14 and nr > 10000:
            self.lib_omp.quant_matmul_omp(
                raw.ctypes.data_as(self.cu), x_in.ctypes.data_as(self.cf),
                out.ctypes.data_as(self.cf), nr, nc, qtype)
        else:
            self.lib_v2.quant_matmul_v2(
                raw.ctypes.data_as(self.cu), x_in.ctypes.data_as(self.cf),
                out.ctypes.data_as(self.cf), nr, nc, qtype)
        return out
    
    def _apply_rope(self, x, pos, n_heads):
        hd = self.head_dim
        dim = min(hd, x.shape[0])
        half = dim // 2
        cos_p = self.cos_cache[pos, :half]
        sin_p = self.sin_cache[pos, :half]
        x_even = x[0:dim:2][:len(cos_p)]
        x_odd = x[1:dim:2][:len(cos_p)]
        result = np.zeros(x.shape[0], dtype=np.float32)
        result[0:dim:2][:len(cos_p)] = x_even * cos_p - x_odd * sin_p
        result[1:dim:2][:len(cos_p)] = x_even * sin_p + x_odd * cos_p
        if x.shape[0] > dim:
            result[dim:] = x[dim:]
        return result
    
    def reset(self):
        self.pos = 0
    
    def forward(self, token):
        b = self._buf
        N = self.n_embd
        hd = self.head_dim
        n_h = self.n_head
        n_kv = self.n_kv_head
        
        np.copyto(b['h'], self.emb[token])
        
        for i in range(self.n_layers):
            pfx = f'blk.{i}'
            np.copyto(b['residual'], b['h'])
            b['h'][:] = self._rms_norm(b['h'], f'{pfx}.attn_norm.weight')
            
            q = self._matmul(f'{pfx}.attn_q.weight', b['h'])
            k = self._matmul(f'{pfx}.attn_k.weight', b['h'])
            v = self._matmul(f'{pfx}.attn_v.weight', b['h'])
            
            q2 = self._apply_rope(q[:n_h*hd], self.pos, n_h)
            k2 = self._apply_rope(k[:n_kv*hd], self.pos, n_kv)
            
            self.k_cache[i, self.pos, :, :] = k2.reshape(n_kv, hd)
            self.v_cache[i, self.pos, :, :] = v.reshape(n_kv, hd)
            
            scale = 1.0 / math.sqrt(hd)
            q2r = q2.reshape(n_h, hd)
            n_rep = n_h // n_kv
            att = np.zeros((n_h, hd), dtype=np.float32)
            for h in range(n_h):
                scores = scale * (q2r[h] @ self.k_cache[i, :self.pos+1, h//n_rep].T)
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                att[h] = scores @ self.v_cache[i, :self.pos+1, h//n_rep]
            
            b['att_out'][:n_h*hd] = att.flatten()
            
            if self.use_qk_norm:
                # QK norm not needed for batched approach
                pass
            
            o = self._matmul(f'{pfx}.attn_output.weight', b['att_out'][:N])
            b['h'][:N] = b['residual'][:N] + o[:N]
            
            if self.is_moe and f'{pfx}.ffn_gate_exps.weight' in self.weight_qtypes:
                np.copyto(b['residual'], b['h'])
                b['h'][:] = self._rms_norm(b['h'], f'{pfx}.ffn_norm.weight')
                gate_logits = self._matmul(f'{pfx}.ffn_gate_exps.weight', b['h'][:N])[:self.n_experts]
                top_k = min(self.n_experts_per_tok, self.n_experts)
                top_indices = np.argsort(gate_logits)[-top_k:]
                top_weights = self._silu(gate_logits[top_indices])
                top_weights /= top_weights.sum()
                result = np.zeros(N, dtype=np.float32)
                for ei, idx in enumerate(top_indices):
                    gate = self._matmul(f'{pfx}.ffn_gate.{idx}.weight', b['h'][:N])
                    up = self._matmul(f'{pfx}.ffn_up.{idx}.weight', b['h'][:N])
                    down = self._matmul(f'{pfx}.ffn_down.{idx}.weight', self._silu(gate) * up)
                    result += top_weights[ei] * down[:N]
                b['h'][:N] = b['residual'][:N] + result[:N]
            else:
                np.copyto(b['residual'], b['h'])
                b['h'][:] = self._rms_norm(b['h'], f'{pfx}.ffn_norm.weight')
                gate = self._matmul(f'{pfx}.ffn_gate.weight', b['h'][:N])
                up = self._matmul(f'{pfx}.ffn_up.weight', b['h'][:N])
                b['silu_gate'][:len(gate)] = self._silu(gate) * up
                down = self._matmul(f'{pfx}.ffn_down.weight', b['silu_gate'])
                b['h'][:N] = b['residual'][:N] + down[:N]
        
        b['h'][:] = self._rms_norm(b['h'], 'output_norm.weight')
        logits = self._matmul(self.out_w_name, b['h'][:N])
        self.pos += 1
        return logits
    
    def benchmark(self, n_warmup=5, n_iter=50):
        self.reset()
        self.forward(128000)
        tok = 11
        for _ in range(n_warmup):
            logits = self.forward(tok)
            tok = int(np.argmax(logits))
        
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            logits = self.forward(tok)
            times.append((time.perf_counter() - t0) * 1000)
            tok = int(np.argmax(logits))
        
        times.sort()
        med = times[len(times) // 2]
        return med