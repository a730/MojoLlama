"""MojoLlama Graph Operations — architecture definition.

This file defines the op graph architecture for MojoLlama.
Each op struct specifies WHAT to compute and its parameters.

BACKEND NOTE (Mojo 0.26.2):
- Python backend (numpy) — works NOW, called from bridge.py
- Mojo SIMD backend — when Mojo's heap/pointer APIs mature
- MAX GPU backend (CUDA/SYCL/Vulkan) — when MAX is installable

The op definitions stay the same regardless of backend.
"""

# ─── Tensor (shape + data type descriptor) ─────────────────────────────

struct TensorDesc:
    """Describes a tensor's shape and dtype. No data — just metadata.
    The actual data lives in the backend (numpy array, Mojo SIMD buffer, GPU buffer).
    """
    var shape: String     # "batch,seq,dim" — symbolic shape description
    var dtype: String     # "f32", "q4_0", etc.

    fn __init__(out self, shape: String, dtype: String = "f32"):
        self.shape = shape
        self.dtype = dtype


# ─── Operation definitions for Llama3 ──────────────────────────────────
# Each op maps to a MAX graph op and has a numpy/Python implementation
# in the backend.

struct MatmulOp:
    """y = x @ W.T"""
    var name: String
    var in_features: Int
    var out_features: Int
    
    fn __init__(out self, name: String, in_features: Int, out_features: Int):
        self.name = name
        self.in_features = in_features
        self.out_features = out_features


struct AttentionOp:
    """Multi-head attention with optional KV cache.
    softmax(Q @ K.T / sqrt(d)) @ V
    
    Backends: numpy → Mojo SIMD MHA → MAX MHA kernel (CPU/GPU dispatch)
    """
    var n_head: Int
    var n_kv_head: Int
    var head_dim: Int
    
    fn __init__(out self, n_head: Int, n_kv_head: Int, head_dim: Int):
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim


struct RMSNormOp:
    """x / sqrt(mean(x^2) + eps) * weight"""
    var eps: Float32
    
    fn __init__(out self, eps: Float32 = 1e-6):
        self.eps = eps


struct RoPEOp:
    """Rotary Position Embedding"""
    pass


struct SiLUOp:
    """x * sigmoid(x)"""
    pass


struct EmbedOp:
    """Token embedding lookup: embed[input_ids]"""
    var vocab_size: Int
    var dim: Int
    
    fn __init__(out self, vocab_size: Int, dim: Int):
        self.vocab_size = vocab_size
        self.dim = dim


struct AddOp:
    """Element-wise addition (residual connection)"""
    pass


# ─── Transformer Block ─────────────────────────────────────────────────

struct TransformerBlock:
    """A single transformer block = Attention + FFN with residuals."""
    var attention: AttentionOp
    var attn_norm: RMSNormOp
    var ffn_norm: RMSNormOp
    var q_proj: MatmulOp
    var k_proj: MatmulOp
    var v_proj: MatmulOp
    var o_proj: MatmulOp
    var gate_proj: MatmulOp
    var up_proj: MatmulOp
    var down_proj: MatmulOp
    
    fn __init__(out self, n_embd: Int, n_head: Int, n_kv_head: Int, n_ff: Int):
        var hd = n_embd // n_head
        self.attention = AttentionOp(n_head, n_kv_head, hd)
        self.attn_norm = RMSNormOp()
        self.ffn_norm = RMSNormOp()
        self.q_proj = MatmulOp("q", n_embd, n_embd)
        self.k_proj = MatmulOp("k", n_embd, n_kv_head * hd)
        self.v_proj = MatmulOp("v", n_embd, n_kv_head * hd)
        self.o_proj = MatmulOp("o", n_embd, n_embd)
        self.gate_proj = MatmulOp("gate", n_embd, n_ff)
        self.up_proj = MatmulOp("up", n_embd, n_ff)
        self.down_proj = MatmulOp("down", n_ff, n_embd)


# ─── Llama3 Model Graph ────────────────────────────────────────────────

struct Llama3Graph:
    """Complete Llama3 model as a graph of operations."""
    var embed: EmbedOp
    var output_norm: RMSNormOp
    var lm_head: MatmulOp
    
    fn __init__(out self, n_layers: Int, n_embd: Int, n_head: Int, 
                n_kv_head: Int, n_ff: Int, vocab_size: Int):
        self.embed = EmbedOp(vocab_size, n_embd)
        self.output_norm = RMSNormOp()
        self.lm_head = MatmulOp("lm_head", n_embd, vocab_size)
