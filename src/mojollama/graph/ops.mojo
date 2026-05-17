"""MojoLlama Graph Operations — architecture definition.

This file defines the op graph architecture for MojoLlama.
Each op struct specifies WHAT to compute and its parameters.

BACKEND NOTE (Mojo 0.26.2):
- Python backend (numpy) — works NOW, called from bridge.py
- Mojo SIMD backend — when Mojo's heap/pointer APIs mature
- MAX GPU backend (CUDA/SYCL/Vulkan) — when MAX is installable

The op definitions stay the same regardless of backend.

ARCHITECTURES DEFINED:
- Llama3: Dense transformer with GQA, SiLU-gated FFN, full RoPE
- ZAYA1: MoE++ with interleaved attention + Mamba-2 SSM, CCA, MoD
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


# ─── Shared primitive ops (architecture-agnostic) ──────────────────────
# These ops are reused across all model architectures.

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
    
    Backends: numpy -> Mojo SIMD MHA -> MAX MHA kernel (CPU/GPU dispatch)
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
    """Rotary Position Embedding — applied to full head dimension."""
    var theta: Float32
    
    fn __init__(out self, theta: Float32 = 10000.0):
        self.theta = theta


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


# ─── ZAYA1-specific primitive ops ──────────────────────────────────────
# ZAYA1 (Zyphra) is a MoE++ architecture combining attention, Mamba-2 SSM,
# Cross-Layer Attention, and Mixture of Depths.

struct PartialRoPEOp:
    """Partial rotary position embedding — applies RoPE to first `dim` dims.
    ZAYA1 uses dim=64 with theta=10000.0 (partial rotary factor = 0.5).
    The remaining (head_dim - dim) dimensions pass through unrotated.
    """
    var dim: Int                # Number of dimensions to rotate
    var theta: Float32
    
    fn __init__(out self, dim: Int = 64, theta: Float32 = 10000.0):
        self.dim = dim
        self.theta = theta


struct Mamba2Op:
    """Mamba-2 selective state-space model (SSM) layer.
    
    Architecture:
      x -> conv1d (silu) -> SSM scan (A, B, C, dt) -> y
    
    Selective scan parameters:
      - state_dim:    D, the state expansion factor (SSM state size per channel)
      - conv_kernel:  Width of the local 1D convolution (ZAYA1: kernel=2)
      - dt_rank:      Rank of the time-step projection (default: state_dim)
    
    ZAYA1 uses state_dim=16 (expansion_factor=0.5 of embed=2048 -> 1024 / 64 channels).
    """
    var state_dim: Int          # SSM state dimension (D)
    var conv_kernel: Int        # Local convolution kernel size
    var dt_rank: Int            # Discretization time-step projection rank
    
    fn __init__(out self, state_dim: Int, conv_kernel: Int = 2, dt_rank: Int = -1):
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.dt_rank = dt_rank if dt_rank > 0 else state_dim


struct MoERouterOp:
    """Mixture of Experts routing/gating.
    
    Computes expert logits per token: softmax(W_router @ x).
    Determines which expert(s) each token is routed to.
    ZAYA1 uses top-1 routing (each token goes to exactly 1 expert).
    
    Load balancing: auxiliary loss encourages uniform expert utilization.
    """
    var n_experts: Int          # Total number of experts (ZAYA1-8B: 16)
    var top_k: Int              # Top-k experts per token (ZAYA1: 1)
    var dim: Int                # Input dimension for router projection
    
    fn __init__(out self, n_experts: Int, dim: Int, top_k: Int = 1):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim


struct ExpertFFNOp:
    """A single expert FFN within an MoE layer.
    SwiGLU gated FFN: (silu(x @ gate.T) * (x @ up.T)) @ down.T
    Same structure as dense FFN, but per-expert.
    """
    var intermediate_dim: Int   # Hidden dimension within the expert
    
    fn __init__(out self, intermediate_dim: Int):
        self.intermediate_dim = intermediate_dim


struct MoEFFNOp:
    """Mixture of Experts FFN — router + N parallel expert FFNs.
    
    For each token:
      1. Router computes expert weights -> selects top_k experts
      2. Token is dispatched to selected experts
      3. Expert outputs are combined via weighted sum
    
    ZAYA1-8B: 16 experts, top-1 routing, expert_intermediate_dim varies.
    """
    var router: MoERouterOp
    var expert: ExpertFFNOp    # All experts share the same intermediate structure
    var n_experts: Int
    
    fn __init__(out self, n_experts: Int, dim: Int, expert_intermediate_dim: Int, 
                 top_k: Int = 1):
        self.router = MoERouterOp(n_experts, dim, top_k)
        self.expert = ExpertFFNOp(expert_intermediate_dim)
        self.n_experts = n_experts


struct ModRouterOp:
    """Mixture of Depths routing — token-level sub-layer gating.
    
    Each token independently decides whether to pass through the sub-layer
    (attention or SSM) or take the residual shortcut.
    
    MoD enables dynamic computation: tokens that already have sufficient
    context skip expensive sub-layer computation.
    """
    var dim: Int                # Input dimension for router
    
    fn __init__(out self, dim: Int):
        self.dim = dim


# ─── Llama3 Transformer Block ──────────────────────────────────────────

struct TransformerBlock:
    """A single transformer block = Attention + FFN with residuals.
    Dense architecture (no MoE): all tokens pass through all sub-layers.
    """
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


# ─── ZAYA1 Layer Types ─────────────────────────────────────────────────
# ZAYA1 interleaves two block types:
#   Type A: Attention sub-layer (GQA with partial RoPE + CCA)
#   Type B: Mamba-2 SSM sub-layer
# Each uses MoE FFN with MoD routing.

struct Zaya1AttentionSubLayer:
    """Attention sub-layer for ZAYA1.
    
    GQA attention with:
    - Partial RoPE (first `rope_dim` dimensions only)
    - Cross-Layer Attention (CCA): KV pairs shared between adjacent layers
    - MoD: per-token routing through this sub-layer
    
    CCA means blk.N and blk.N+1 share the same KV projections.
    Only the Q projection differs between paired layers.
    """
    var attention: AttentionOp
    var rope: PartialRoPEOp
    
    fn __init__(out self, n_head: Int, n_kv_head: Int, head_dim: Int,
                 rope_dim: Int = 64, rope_theta: Float32 = 10000.0):
        self.attention = AttentionOp(n_head, n_kv_head, head_dim)
        self.rope = PartialRoPEOp(rope_dim, rope_theta)


struct Zaya1SSMSubLayer:
    """Mamba-2 SSM sub-layer for ZAYA1.
    
    Selective state-space model with:
    - 1D convolution (silu activation)
    - Selective scan with learned A, B, C, dt parameters
    - MoD: per-token routing through this sub-layer
    """
    var ssm: Mamba2Op
    
    fn __init__(out self, state_dim: Int, conv_kernel: Int = 2):
        self.ssm = Mamba2Op(state_dim, conv_kernel)


struct Zaya1Layer:
    """A single ZAYA1 layer = norm + sub-layer (attention or SSM) + MoE FFN + MoD.
    
    Each layer in ZAYA1 has:
      1. Input normalization (RMSNorm)
      2. Sub-layer: Attention (with CCA + partial RoPE) or Mamba-2 SSM
      3. MoD routing: per-token decision to keep sub-layer output or skip
      4. MoE FFN: 16 experts, top-1 routing with router
      5. Residual connection
    
    The `is_attention` field distinguishes the type at the graph level
    (no runtime dispatch — the graph IS the architecture).
    """
    var sublayer_type: String         # "attention" or "ssm"
    var input_norm: RMSNormOp
    var attention: Zaya1AttentionSubLayer  # Valid when sublayer_type == "attention"
    var ssm: Zaya1SSMSubLayer             # Valid when sublayer_type == "ssm"
    var mod_router: ModRouterOp
    var moe_ffn: MoEFFNOp
    
    fn __init__(out self, sublayer_type: String, n_embd: Int, n_head: Int, 
                n_kv_head: Int, head_dim: Int, n_experts: Int, 
                expert_intermediate_dim: Int, state_dim: Int,
                rope_dim: Int = 64, top_k: Int = 1):
        self.sublayer_type = sublayer_type
        self.input_norm = RMSNormOp()
        self.attention = Zaya1AttentionSubLayer(n_head, n_kv_head, head_dim, rope_dim)
        self.ssm = Zaya1SSMSubLayer(state_dim)
        self.mod_router = ModRouterOp(n_embd)
        self.moe_ffn = MoEFFNOp(n_experts, n_embd, expert_intermediate_dim, top_k)


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


# ─── ZAYA1 Model Graph ─────────────────────────────────────────────────
#
# ZAYA1-8B architecture (80 layers):
#   Pattern: [ATTN, ATTN, SSM, SSM] × 20
#   - Attention layers use Cross-Layer Attention (CCA pairs: layer 0&1, 4&5, ...)
#   - SSM layers use Mamba-2
#   - All layers use MoE FFN (16 experts, top-1) + MoD routing
#
# CCA: layers at indices (0,1), (4,5), (8,9), ... share KV pairs.
# Within each pair, block N produces KV, block N+1 reuses.
#
# Default configuration (ZAYA1-8B):
#   n_layers=80, n_embd=2048, n_head=8, n_kv_head=2, head_dim=256
#   rope_dim=64 (partial rotary, factor=0.5)
#   n_experts=16, top_k=1, expert_intermediate_dim=?
#   state_dim=?, conv_kernel=2 (for Mamba-2)
#   max_seq_len=131072
#   vocab_size=256000 (Gemma4 tokenizer)
#   norm_eps=?, rope_theta=10000.0

struct Zaya1Graph:
    """Complete ZAYA1 model as a graph of operations.
    
    Contains:
    - Token embedding
    - 80 interleaved layers (attention / SSM blocks)
    - Output normalization
    - LM head (MoE router at output)
    
    Layer pattern (20 repeats of [ATTN, ATTN, SSM, SSM]):
      idx 0:   ATTN (CCA pair with idx 1)
      idx 1:   ATTN (CCA pair with idx 0)
      idx 2:   SSM
      idx 3:   SSM
      idx 4:   ATTN (CCA pair with idx 5)
      idx 5:   ATTN (CCA pair with idx 4)
      ...
    """
    var embed: EmbedOp
    var output_norm: RMSNormOp
    var lm_head: MatmulOp
    # NOTE: Layer array would live here when Mojo supports dynamic arrays of structs.
    # For Mojo 0.26.2: use a backend-side representation (the Python inference class).
    # The graph constants below document the exact layer topology.

    fn __init__(out self, vocab_size: Int, n_embd: Int):
        self.embed = EmbedOp(vocab_size, n_embd)
        self.output_norm = RMSNormOp()
        self.lm_head = MatmulOp("lm_head", n_embd, vocab_size)


# ─── Architecture constants ────────────────────────────────────────────
# These constants document the exact graph topology for each architecture.
# The Mojo 0.26.2 backend (Python bridge) reads these to construct the
# correct computation DAG. Future backends will use them as compile-time
# graph definitions.

# ZAYA1-8B layer topology: returns a list of layer-type strings
# describing the full 80-layer interleaved pattern.
# Pattern: [ATTN, ATTN, SSM, SSM] × 20
fn zaya1_layer_schedule() -> String:
    """Returns the 80-layer interleaved schedule as a comma-separated string.
    
    Format: "attention,attention,ssm,ssm,attention,attention,..."
    Backends parse this to build the layer DAG.
    """
    var result: String = ""
    var i: Int = 0
    while i < 20:
        if i > 0:
            result += ","
        result += "attention,attention,ssm,ssm"
        i += 1
    return result


# ZAYA1 CCA pairing: returns a mapping of which layers share KV cache.
# Format: "L0:L1,L4:L5,..." where paired layers share KV projections.
fn zaya1_cca_pairs() -> String:
    """Returns CCA pairs as comma-separated 'L:R' tuples.
    
    Layers at indices (0,1), (4,5), (8,9), ..., (76,77) are paired.
    Layers 2,3,6,7,etc (SSM layers) are unpaired.
    """
    var result: String = ""
    var i: Int = 0
    while i < 80:
        # Check if this is the first of an ATTN pair (indices 0,4,8,...,76)
        if i % 4 == 0:
            if len(result) > 0:
                result += ","
            result += "L" + String(i) + ":L" + String(i + 1)
        i += 1
    return result
