"""
architectures.py — Dynamic model architecture detection for MojoLlama.

Reads GGUF metadata (general.architecture field) or Hugging Face config.json
(model_type field) and returns a standardized ForwardPassType enum value.

This is the single source of truth for determining which forward pass
implementation the MojoLlama engine should use (DENSE, MOE, GEMMA, etc.).

Usage:
    from mojollama.model.architectures import (
        detect_architecture,
        get_forward_pass_type,
        get_model_params,
        print_supported_models,
        ForwardPassType,
    )

    # GGUF reader
    reader = gguf.GGUFReader(path)
    arch = detect_architecture(reader)
    fwd_type = get_forward_pass_type(arch)

    # HF config
    import json
    with open("config.json") as f:
        cfg = json.load(f)
    fwd_type = get_forward_pass_type(cfg)
"""

import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Conditional imports so this module can work without every dependency
try:
    import gguf
    from gguf import MODEL_ARCH, MODEL_ARCH_NAMES, MODEL_TENSORS, MODEL_TENSOR
    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Forward Pass Type Enum
# ──────────────────────────────────────────────


class ForwardPassType(str, Enum):
    """Standardized forward-pass categories for MojoLlama.

    Each value maps to a specific forward-pass implementation in the engine.
    """
    DENSE = "dense"
    """Standard dense transformer (LLaMA, Mistral, Qwen2, etc.).
       Uses: RMSNorm → attention → RoPE → SwiGLU FFN."""

    MOE = "moe"
    """Mixture-of-Experts transformer (Qwen3 MoE, DeepSeek V2, Mixtral, GPT-OSS).
       Uses: RMSNorm → attention → RoPE → router → top-k experts."""

    DEEPSEEK_V4 = "deepseek_v4"
    """DeepSeek V4 Flash (MLA, HC, sparse attention, hash+learned MoE routing).
       43L/4096D/64H/1KV, Q_lora=1024, O_lora=1024, 256 experts×6.
       Uses: embed → HC-expand → (HC-pre → attn_norm → MLA → HC-post → 
              HC-pre → ffn_norm → MoE-FFN → HC-post)x43 → HC-head → norm → lm_head."""

    GEMMA = "gemma"
    """Google Gemma/Gemma2 (GeGLU activation, pre-norm, no RoPE on K).
       Uses: RMSNorm → attention → GeGLU FFN (or gating variant)."""

    GEMMA3 = "gemma3"
    """Google Gemma 3 (with sliding window attention, multi-layer kv heads)."""

    QWEN2 = "qwen2"
    """Qwen2 dense (SwiGLU, advanced RoPE with mrope/dual-ft, TNT embedding)."""

    QWEN3 = "qwen3"
    """Qwen3 dense variant (SwiGLU, TNT, mrope, optional prefix-lm)."""

    STARCODER = "starcoder"
    """StarCoder / StarCoder2 (Falcon-style parallel attention + MLP)."""

    DEEPSEEK = "deepseek"
    """DeepSeek V1/V2 dense (SwiGLU, MLA attention for V2)."""

    COMMAND_R = "command-r"
    """Cohere Command-R dense (SwiGLU, tanh-gated RoPE, layer-wise LR)."""

    BERT = "bert"
    """Encoder-only BERT-style (masked language modelling)."""

    UNKNOWN = "unknown"
    """Unrecognised architecture; fall back to DENSE."""

    # ── Classification helpers ──

    @property
    def is_moe(self) -> bool:
        return self in (ForwardPassType.MOE,)

    @property
    def is_dense(self) -> bool:
        return self not in (ForwardPassType.MOE, ForwardPassType.UNKNOWN)

    @property
    def is_gemma_family(self) -> bool:
        return self in (ForwardPassType.GEMMA, ForwardPassType.GEMMA3)


# ──────────────────────────────────────────────
#  Architecture string → ForwardPassType mapping
# ──────────────────────────────────────────────

# Maps GGUF general.architecture values to ForwardPassType
GGUF_ARCH_MAP: Dict[str, ForwardPassType] = {
    # ── Dense ──
    "llama":      ForwardPassType.DENSE,
    "llama2":     ForwardPassType.DENSE,
    "llama3":     ForwardPassType.DENSE,
    "llama4":     ForwardPassType.DENSE,
    "codellama":  ForwardPassType.DENSE,
    "mistral":    ForwardPassType.DENSE,
    "mistral3":   ForwardPassType.DENSE,
    "mistral4":   ForwardPassType.DENSE,
    "qwen2":      ForwardPassType.QWEN2,
    "qwen2vl":    ForwardPassType.DENSE,
    "qwen3":      ForwardPassType.QWEN3,
    "qwen35":     ForwardPassType.QWEN3,
    "starcoder2": ForwardPassType.STARCODER,
    "starcoder":  ForwardPassType.STARCODER,
    "command-r":  ForwardPassType.COMMAND_R,
    "cohere2":    ForwardPassType.COMMAND_R,
    "deepseek":   ForwardPassType.DEEPSEEK,
    "internlm2":  ForwardPassType.DENSE,
    "phi3":       ForwardPassType.DENSE,
    "gpt2":       ForwardPassType.DENSE,
    "gptneox":    ForwardPassType.DENSE,
    "falcon":     ForwardPassType.DENSE,
    "stablelm":   ForwardPassType.DENSE,
    "xverse":     ForwardPassType.DENSE,
    "olmo":       ForwardPassType.DENSE,
    "olmo2":      ForwardPassType.DENSE,
    "baichuan":   ForwardPassType.DENSE,
    "llama-embed": ForwardPassType.DENSE,

    # ── MoE ──
    "qwen2moe":   ForwardPassType.MOE,
    "qwen3moe":   ForwardPassType.MOE,
    "qwen35moe":  ForwardPassType.MOE,
    "gpt-oss":    ForwardPassType.MOE,
    "mixtral":    ForwardPassType.MOE,
    "deepseek2":  ForwardPassType.MOE,
    "deepseek3":  ForwardPassType.MOE,
    "deepseek_v4": ForwardPassType.DEEPSEEK_V4,
    "dbrx":       ForwardPassType.MOE,
    "qwen3next":  ForwardPassType.MOE,
    "olmoe":      ForwardPassType.MOE,
    "arctic":     ForwardPassType.MOE,
    "phimoe":     ForwardPassType.MOE,
    "grokt":      ForwardPassType.MOE,
    "granitemoe": ForwardPassType.MOE,
    "bailingmoe": ForwardPassType.MOE,
    "afmoe":      ForwardPassType.MOE,
    "llada-moe":  ForwardPassType.MOE,
    "seed_oss":   ForwardPassType.MOE,
    "grovemoe":   ForwardPassType.MOE,
    "hunyuan-moe": ForwardPassType.MOE,
    "ernie4_5-moe": ForwardPassType.MOE,
    "smallthinker": ForwardPassType.MOE,

    # ── Gemma family ──
    "gemma":      ForwardPassType.GEMMA,
    "gemma2":     ForwardPassType.GEMMA,
    "gemma3":     ForwardPassType.GEMMA3,
    "gemma3n":    ForwardPassType.GEMMA3,
    "gemma4":     ForwardPassType.GEMMA3,
}

# Maps HuggingFace model_type strings (from config.json) to ForwardPassType
HF_MODEL_TYPE_MAP: Dict[str, ForwardPassType] = {
    # ── Dense ──
    "llama":              ForwardPassType.DENSE,
    "mistral":            ForwardPassType.DENSE,
    "qwen2":              ForwardPassType.QWEN2,
    "qwen2_vl":           ForwardPassType.DENSE,
    "qwen2_5_vl":         ForwardPassType.DENSE,
    "qwen2_moe":          ForwardPassType.MOE,
    "qwen3":              ForwardPassType.QWEN3,
    "starcoder2":         ForwardPassType.STARCODER,
    "cohere":             ForwardPassType.COMMAND_R,
    "cohere2":            ForwardPassType.COMMAND_R,
    "deepseek_v2":        ForwardPassType.MOE,
    "deepseek":           ForwardPassType.DEEPSEEK,
    "internlm2":          ForwardPassType.DENSE,
    "phi3":               ForwardPassType.DENSE,
    "stablelm":           ForwardPassType.DENSE,
    "falcon":             ForwardPassType.DENSE,
    "baichuan":           ForwardPassType.DENSE,
    "olmo":               ForwardPassType.DENSE,
    "olmo2":              ForwardPassType.DENSE,
    "xverse":             ForwardPassType.DENSE,
    "gpt2":               ForwardPassType.DENSE,
    "gpt_neox":           ForwardPassType.DENSE,
    "bloom":              ForwardPassType.DENSE,
    "refact":             ForwardPassType.DENSE,
    "codegen":            ForwardPassType.DENSE,
    "gemma":              ForwardPassType.GEMMA,
    "gemma2":             ForwardPassType.GEMMA,
    "gemma3":             ForwardPassType.GEMMA3,
    "gemma4":             ForwardPassType.GEMMA3,
    "command-r":          ForwardPassType.COMMAND_R,
    "phi":                ForwardPassType.DENSE,
    "phi2":               ForwardPassType.DENSE,
    "phi-2":              ForwardPassType.DENSE,
    "phi-3":              ForwardPassType.DENSE,
    "minicpm":            ForwardPassType.DENSE,
    "minicpm3":           ForwardPassType.DENSE,
    "minicpm":            ForwardPassType.DENSE,

    # ── MoE ──
    "qwen3_moe":          ForwardPassType.MOE,
    "mixtral":            ForwardPassType.MOE,
    "deepseek":           ForwardPassType.DEEPSEEK,  # V1 dense
    "deepseek_v2":        ForwardPassType.MOE,
    "jetmoe":             ForwardPassType.MOE,
    "dbry":               ForwardPassType.MOE,
    "olmoe":              ForwardPassType.MOE,
    "grok":               ForwardPassType.MOE,
    "phimoe":             ForwardPassType.MOE,
    "granite_moe":        ForwardPassType.MOE,
    "gpt_oss":            ForwardPassType.MOE,
    "qwen2_moe":          ForwardPassType.MOE,

    # ── Other ──
    "bert":               ForwardPassType.BERT,
    "bert-large":         ForwardPassType.BERT,
    "roberta":            ForwardPassType.BERT,
    "albert":             ForwardPassType.BERT,
}


# ──────────────────────────────────────────────
#  Core detection functions
# ──────────────────────────────────────────────


def _get_gguf_field_str(reader, field_name: str) -> Optional[str]:
    """Extract a string value from a GGUF metadata field."""
    if not _HAS_GGUF:
        return None
    try:
        field = reader.get_field(field_name)
        if field is None:
            return None
        # GGUFReader fields have .parts — the last element is the value
        part = field.parts[-1]
        if isinstance(part, bytes):
            return part.decode("utf-8", errors="replace").strip("\x00")
        if hasattr(part, "tobytes"):
            return part.tobytes().decode("utf-8", errors="replace").strip("\x00")
        return str(part).strip()
    except Exception:
        return None


def _get_gguf_arch(reader) -> Optional[str]:
    """Extract architecture string from a GGUF reader."""
    return _get_gguf_field_str(reader, "general.architecture")


def _get_hf_config_type(config: Dict[str, Any]) -> Optional[str]:
    """Extract model_type from an HF config.json dict."""
    return config.get("model_type", None)


def detect_architecture(
    source: Union["gguf.GGUFReader", Dict[str, Any], str],
) -> ForwardPassType:
    """Detect architecture from a GGUF reader, HF config dict, or arch string.

    Args:
        source: One of:
            - A ``gguf.GGUFReader`` instance (reads general.architecture field)
            - A ``dict`` loaded from an HF ``config.json`` (reads model_type key)
            - A raw architecture string (e.g. ``"qwen3moe"``, ``"llama"``)

    Returns:
        The detected ``ForwardPassType`` (falls back to UNKNOWN).

    Examples:
        # From GGUF file
        reader = gguf.GGUFReader("model.gguf")
        arch = detect_architecture(reader)

        # From HF config
        with open("config.json") as f:
            cfg = json.load(f)
        arch = detect_architecture(cfg)

        # From string
        arch = detect_architecture("qwen3moe")
    """
    # ForwardPassType passthrough (must check before str, since ForwardPassType inherits str)
    if isinstance(source, ForwardPassType):
        return source

    arch_str: Optional[str] = None

    if isinstance(source, str):
        # Raw architecture string provided directly
        arch_str = source

    elif isinstance(source, dict):
        # HF config.json dict
        arch_str = _get_hf_config_type(source)

    elif _HAS_GGUF and isinstance(source, gguf.GGUFReader):
        # GGUF reader — try the metadata field
        arch_str = _get_gguf_arch(source)

        # If we still didn't get an arch string, scan tensor names for MoE patterns
        if arch_str is None or arch_str == "unknown":
            try:
                # Look at the gguf package's built-in MODEL_ARCH detection
                # (the reader might have it cached)
                if hasattr(source, "arch") and source.arch:
                    arch_str = source.arch
            except Exception:
                pass

    else:
        logger.warning(
            "detect_architecture: unsupported source type %s",
            type(source).__name__,
        )
        return ForwardPassType.UNKNOWN

    if not arch_str:
        return ForwardPassType.UNKNOWN

    # Normalise: strip, lowercase
    arch_str = arch_str.strip().lower().replace("-", "_").replace(" ", "_")

    # Direct lookup in GGUF map (covers the known list)
    result = GGUF_ARCH_MAP.get(arch_str)
    if result is not None:
        return result

    # Also try without underscores (for normalized strings like "gpt_oss" → check "gpt-oss")
    result = GGUF_ARCH_MAP.get(arch_str.replace("_", "-"))
    if result is not None:
        return result

    # Try HF model type map before prefix matching (HF model_types can use
    # underscore separators like "qwen3_moe" that should not be swallowed by
    # a short prefix like "qwen3")
    result = HF_MODEL_TYPE_MAP.get(arch_str)
    if result is not None:
        return result

    # Try fuzzy prefix matching for versioned archs (e.g. "deepseek3" -> MOE)
    # Sort by length descending so more specific prefixes match first
    for pattern, fptype in sorted(
        _PREFIX_MAP.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if arch_str.startswith(pattern):
            return fptype

    # Last resort: use gguf's MODEL_ARCH enum to classify by analysing
    # required tensors — if the arch has expert weight tensors it's MoE
    if _HAS_GGUF:
        try:
            arch_enum_val = None
            for name, val in MODEL_ARCH.__members__.items():
                if name.lower().replace("_", "") == arch_str.replace("_", ""):
                    arch_enum_val = val
                    break
                # Also try with common naming conventions
                name_norm = MODEL_ARCH_NAMES.get(val, "")
                if name_norm and name_norm.replace("-", "_").replace(".", "_") == arch_str:
                    arch_enum_val = val
                    break
            if arch_enum_val is not None:
                tensors = MODEL_TENSORS.get(arch_enum_val, set())
                required_tensor_names = {t.name for t in tensors}
                moe_tensors = {"FFN_GATE_EXP", "FFN_UP_EXP", "FFN_DOWN_EXP"}
                if moe_tensors.intersection(required_tensor_names):
                    return ForwardPassType.MOE
                return ForwardPassType.DENSE
        except Exception:
            pass

    return ForwardPassType.UNKNOWN


# Prefix-based fallback map for architectures that may have version suffixes
_PREFIX_MAP: Dict[str, ForwardPassType] = {
    "llama":     ForwardPassType.DENSE,
    "mistral":   ForwardPassType.DENSE,
    "qwen2moe":  ForwardPassType.MOE,
    "qwen3moe":  ForwardPassType.MOE,
    "qwen35moe": ForwardPassType.MOE,
    "qwen2":     ForwardPassType.QWEN2,
    "qwen2vl":   ForwardPassType.DENSE,
    "qwen3":     ForwardPassType.QWEN3,
    "qwen":      ForwardPassType.QWEN2,
    "deepseek":  ForwardPassType.MOE,
    "mixtral":   ForwardPassType.MOE,
    "gemma":     ForwardPassType.GEMMA,
    "starcoder": ForwardPassType.STARCODER,
    "command":   ForwardPassType.COMMAND_R,
    "cohere":    ForwardPassType.COMMAND_R,
    "gpt":       ForwardPassType.DENSE,
    "phi":       ForwardPassType.DENSE,
    "falcon":    ForwardPassType.DENSE,
    "bloom":     ForwardPassType.DENSE,
    "bert":      ForwardPassType.BERT,
}


# ──────────────────────────────────────────────
#  Tensor-name-based MoE detection (runtime)
# ──────────────────────────────────────────────


def _has_moe_tensors(source) -> bool:
    """Check if a GGUF reader has expert weight tensors (MoE indicator).

    Looks for the presence of ``ffn_gate_exps.weight``,
    ``ffn_up_exps.weight``, or ``ffn_down_exps.weight`` in any layer.
    """
    if not _HAS_GGUF:
        return False
    if not isinstance(source, gguf.GGUFReader):
        return False
    try:
        tensors = source.tensors
        for t in tensors:
            name = t.name
            if any(
                pat in name
                for pat in (".ffn_gate_exps.", ".ffn_up_exps.", ".ffn_down_exps.")
            ):
                return True
    except Exception:
        pass
    return False


# ──────────────────────────────────────────────
#  Forward pass type resolution
# ──────────────────────────────────────────────


def get_forward_pass_type(
    arch_or_config: Union[ForwardPassType, "gguf.GGUFReader", Dict[str, Any], str],
) -> ForwardPassType:
    """Resolve the forward pass type from any supported input.

    This is a convenience wrapper around ``detect_architecture`` that also
    accepts an already-resolved ``ForwardPassType`` (passes it through).

    Args:
        arch_or_config: One of:
            - ``ForwardPassType`` (returned as-is)
            - ``gguf.GGUFReader``
            - ``dict`` (HF config.json)
            - ``str`` (architecture name)

    Returns:
        The matching ``ForwardPassType``.
    """
    if isinstance(arch_or_config, ForwardPassType):
        return arch_or_config
    return detect_architecture(arch_or_config)


# ──────────────────────────────────────────────
#  Model parameter extraction
# ──────────────────────────────────────────────


def _get_gguf_scalar(reader, key: str, default=0) -> Any:
    """Read a scalar (int/float) from a GGUF metadata field."""
    try:
        field = reader.get_field(key)
        if field is None:
            return default
        part = field.parts[-1]
        if hasattr(part, "item"):
            return part.item()
        if isinstance(part, (int, float)):
            return part
        if isinstance(part, bytes):
            try:
                return int(part)
            except ValueError:
                return float(part)
        return int(part)
    except Exception:
        return default


def _get_gguf_scalar_with_prefixes(
    reader, key_suffix: str, prefixes: list, default=0
) -> Any:
    """Try reading *prefix* + *key_suffix* across a list of prefixes."""
    for prefix in prefixes:
        val = _get_gguf_scalar(reader, f"{prefix}{key_suffix}")
        if val:
            return val
    return default


def get_model_params(
    gguf_reader_or_config: Union["gguf.GGUFReader", Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract common model hyperparameters from GGUF metadata or HF config.

    Returns a dict with keys:
        ``arch``, ``n_layers``, ``n_embd``, ``n_head``, ``n_kv_head``,
        ``n_ff``, ``max_seq_len``, ``norm_eps``, ``rope_theta``,
        ``vocab_size``, ``forward_pass_type``

    Args:
        gguf_reader_or_config: Either a ``gguf.GGUFReader`` or a ``dict``
            (as loaded from ``config.json``).

    Returns:
        A dict of model parameters, or an empty dict on failure.
    """
    params: Dict[str, Any] = {}

    try:
        if _HAS_GGUF and isinstance(gguf_reader_or_config, gguf.GGUFReader):
            reader = gguf_reader_or_config
            arch_str = _get_gguf_arch(reader) or "unknown"
            params["arch"] = arch_str
            params["forward_pass_type"] = detect_architecture(reader)

            # Resolve prefixes to try for GGUF key lookup
            # (Architecture-specific key prefixes like "qwen2.block_count")
            prefixes = _get_prefixes_for_arch(arch_str)

            params["n_layers"] = int(
                _get_gguf_scalar_with_prefixes(reader, "block_count", prefixes, 0)
            )
            params["n_embd"] = int(
                _get_gguf_scalar_with_prefixes(reader, "embedding_length", prefixes, 0)
            )
            params["n_head"] = int(
                _get_gguf_scalar_with_prefixes(
                    reader, "attention.head_count", prefixes, 0
                )
            )
            params["n_kv_head"] = int(
                _get_gguf_scalar_with_prefixes(
                    reader, "attention.head_count_kv", prefixes, 0
                )
            )
            params["n_ff"] = int(
                _get_gguf_scalar_with_prefixes(
                    reader, "feed_forward_length", prefixes, 0
                )
            )
            params["max_seq_len"] = int(
                _get_gguf_scalar_with_prefixes(reader, "context_length", prefixes, 32768)
            )
            # Also try max_position_embeddings as fallback
            if params["max_seq_len"] <= 0:
                params["max_seq_len"] = int(
                    _get_gguf_scalar_with_prefixes(
                        reader, "max_position_embeddings", prefixes, 32768
                    )
                )

            params["norm_eps"] = float(
                _get_gguf_scalar_with_prefixes(
                    reader,
                    "attention.layer_norm_rms_epsilon",
                    prefixes,
                    1e-6,
                )
            )
            # Also fallback to attn_layer_norm_rms_epsilon (older GGUF convention)
            if params["norm_eps"] <= 0:
                params["norm_eps"] = float(
                    _get_gguf_scalar_with_prefixes(
                        reader, "attn_layer_norm_rms_epsilon", prefixes, 1e-6
                    )
                )

            base = _get_gguf_scalar_with_prefixes(
                reader, "rope.freq_base", prefixes, None
            )
            params["rope_theta"] = float(base) if base is not None else 1000000.0

            # Vocab size from tokenizer metadata
            try:
                token_field = reader.get_field("tokenizer.ggml.tokens")
                if token_field is not None:
                    parts = list(token_field.parts)
                    n_tokens = int(parts[4].item()) if len(parts) > 4 else 0
                    params["vocab_size"] = n_tokens
                else:
                    params["vocab_size"] = int(
                        _get_gguf_scalar_with_prefixes(
                            reader, "vocab_size", prefixes, 0
                        )
                    )
            except Exception:
                params["vocab_size"] = 0

            # MoE-specific parameters
            fwd = params["forward_pass_type"]
            if fwd == ForwardPassType.MOE or fwd.is_moe:
                params["n_experts"] = int(
                    _get_gguf_scalar_with_prefixes(
                        reader, "attention.expert_count", prefixes, 0
                    )
                )
                if params["n_experts"] <= 0:
                    params["n_experts"] = int(
                        _get_gguf_scalar_with_prefixes(
                            reader, "expert_count", prefixes, 0
                        )
                    )
                params["n_experts_per_tok"] = int(
                    _get_gguf_scalar_with_prefixes(
                        reader, "attention.expert_used_count", prefixes, 1
                    )
                )
                if params["n_experts_per_tok"] <= 0:
                    params["n_experts_per_tok"] = int(
                        _get_gguf_scalar_with_prefixes(
                            reader, "expert_used_count", prefixes, 1
                        )
                    )
            else:
                params["n_experts"] = 0
                params["n_experts_per_tok"] = 0

            # Rope type
            params["rope_type"] = str(
                _get_gguf_scalar_with_prefixes(reader, "rope.type", prefixes, "default")
            )

            # Parallel residual (for Falcon-style)
            params["parallel_residual"] = bool(
                _get_gguf_scalar_with_prefixes(
                    reader, "use_parallel_residual", prefixes, 0
                )
            )

        elif isinstance(gguf_reader_or_config, dict):
            cfg = gguf_reader_or_config
            model_type = cfg.get("model_type", "unknown")
            params["arch"] = model_type
            params["forward_pass_type"] = detect_architecture(cfg)
            params["n_layers"] = int(cfg.get("num_hidden_layers", 0))
            params["n_embd"] = int(cfg.get("hidden_size", 0))
            params["n_head"] = int(
                cfg.get("num_attention_heads", 0)
            )
            params["n_kv_head"] = int(
                cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 0))
            )
            params["n_ff"] = int(
                cfg.get("intermediate_size", 0)
            )
            params["max_seq_len"] = int(
                cfg.get("max_position_embeddings", cfg.get("max_seq_len", 32768))
            )
            params["norm_eps"] = float(
                cfg.get("rms_norm_eps", cfg.get("layer_norm_epsilon", 1e-6))
            )
            params["rope_theta"] = float(
                cfg.get("rope_theta", cfg.get("rope_theta", 1000000.0))
            )
            params["vocab_size"] = int(
                cfg.get("vocab_size", 0)
            )
            params["rope_type"] = str(cfg.get("rope_type", "default"))
            params["parallel_residual"] = bool(
                cfg.get("parallel_residual", cfg.get("use_parallel_residual", False))
            )

            # MoE-specific
            fwd = params["forward_pass_type"]
            if fwd == ForwardPassType.MOE or fwd.is_moe:
                num_experts = cfg.get("num_experts", 0) or cfg.get("num_local_experts", 0)
                params["n_experts"] = int(num_experts)
                params["n_experts_per_tok"] = int(
                    cfg.get("num_experts_per_tok", cfg.get("top_k", 1))
                )
            else:
                params["n_experts"] = 0
                params["n_experts_per_tok"] = 0
        else:
            logger.error(
                "get_model_params: unsupported type %s",
                type(gguf_reader_or_config).__name__,
            )
            return {}

        # Compute head_dim
        n_embd = params.get("n_embd", 0)
        n_head = params.get("n_head", 1)
        params["head_dim"] = n_embd // n_head if n_head > 0 else 0

    except Exception as exc:
        logger.error("get_model_params failed: %s", exc)
        return {}

    return params


def _get_prefixes_for_arch(arch_str: str) -> list:
    """Return a list of GGUF key prefixes to try for the given architecture.

    Different archs store metadata under different prefixes in GGUF files:
    e.g. ``qwen2.block_count``, ``gemma.block_count``.
    """
    base = arch_str.lower().replace("-", "_")

    # Architecture-to-prefix mappings
    prefix_map = {
        "llama": ["llama.", "llama2.", "llama3.", "codellama."],
        "mistral": ["mistral.", "llama."],
        "qwen2": ["qwen2.", "qwen2moe.", "llama."],
        "qwen2vl": ["qwen2vl.", "qwen2.", "llama."],
        "qwen3": ["qwen3.", "qwen3moe.", "llama."],
        "qwen2moe": ["qwen2moe.", "qwen2.", "llama."],
        "qwen3moe": ["qwen3moe.", "qwen3.", "llama."],
        "qwen": ["qwen2.", "qwen2moe.", "llama."],
        "gemma": ["gemma.", "gemma2.", "llama."],
        "gemma2": ["gemma2.", "gemma.", "llama."],
        "gemma3": ["gemma3.", "gemma2.", "gemma.", "llama."],
        "deepseek": ["deepseek.", "deepseek2.", "llama."],
        "deepseek2": ["deepseek2.", "deepseek.", "llama."],
        "mixtral": ["mixtral.", "llama."],
        "command": ["command-r.", "commandr.", "llama."],
        "starcoder": ["starcoder.", "llama."],
        "falcon": ["falcon.", "llama."],
        "phi": ["phi3.", "phi2.", "llama."],
        "gpt_oss": ["gpt-oss.", "llama."],
    }

    # Try exact match
    if base in prefix_map:
        prefixes = list(prefix_map[base])
    else:
        # Try partial match
        prefixes = []
        for key, pfx_list in prefix_map.items():
            if base.startswith(key) or key.startswith(base):
                prefixes.extend(pfx_list)
        if not prefixes:
            prefixes = [f"{base}.", "llama."]

    # Always add the base arch as first try
    if prefixes and prefixes[0] != f"{base}.":
        prefixes.insert(0, f"{base}.")
    elif not prefixes:
        prefixes = [f"{base}.", "llama."]

    return prefixes


# ──────────────────────────────────────────────
#  CLI helper
# ──────────────────────────────────────────────


def print_supported_models() -> None:
    """Print a formatted table of supported architectures to stdout.

    This is designed for CLI usage::

        python -c "from mojollama.model.architectures import print_supported_models; print_supported_models()"
    """
    max_name_width = 20

    header = f"{'Architecture Key':<{max_name_width}}  {'Forward Pass Type':<18}  {'Notes'}"
    sep = "-" * len(header)

    # Group and sort
    dense_archs: list[tuple[str, str]] = []
    moe_archs: list[tuple[str, str]] = []
    gemma_archs: list[tuple[str, str]] = []
    other_archs: list[tuple[str, str]] = []

    for arch_key, fptype in sorted(GGUF_ARCH_MAP.items()):
        note = ""
        if fptype == ForwardPassType.DENSE:
            if arch_key in ("llama", "mistral", "qwen2"):
                note = "Base dense (LLaMA/Mistral/Qwen2)"
            else:
                note = "Dense transformer"
            dense_archs.append((arch_key, note))
        elif fptype == ForwardPassType.MOE:
            if arch_key in ("qwen3moe", "deepseek2", "mixtral", "gpt-oss"):
                note = "MoE (router + top-k experts)"
            else:
                note = "Mixture of Experts"
            moe_archs.append((arch_key, note))
        elif fptype in (ForwardPassType.GEMMA, ForwardPassType.GEMMA3):
            note = "Gemma family (GeGLU)"
            gemma_archs.append((arch_key, note))
        else:
            if fptype == ForwardPassType.COMMAND_R:
                note = "Cohere Command-R dense"
            elif fptype == ForwardPassType.STARCODER:
                note = "StarCoder dense"
            elif fptype == ForwardPassType.QWEN2:
                note = "Qwen2 dense"
            elif fptype == ForwardPassType.QWEN3:
                note = "Qwen3 dense"
            elif fptype == ForwardPassType.DEEPSEEK:
                note = "DeepSeek V1 dense"
            else:
                note = "Other"
            other_archs.append((arch_key, note))

    def _print_section(title: str, items: list) -> None:
        if not items:
            return
        print(f"\n  [{title}]\n")
        for key, note in items:
            fptype_name = GGUF_ARCH_MAP.get(key, ForwardPassType.UNKNOWN).value
            print(f"    {key:<{max_name_width}}  {fptype_name:<18}  {note}")

    print(f"\n{'MojoLlama — Supported Model Architectures':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    _print_section("Dense", dense_archs)
    _print_section("MoE", moe_archs)
    _print_section("Gemma", gemma_archs)
    _print_section("Other", other_archs)

    print()
    print("  ForwardPassType enum values can be used to select the correct")
    print("  forward pass implementation in the MojoLlama engine.")
    print()

    # Also print the ForwardPassType enum summary
    print(f"  {'ForwardPassType':<30}  {'Description'}")
    print(f"  {'─'*30:<30}  {'─'*40}")
    for fptype in ForwardPassType:
        print(f"  {fptype.value:<30}  {fptype.name}")
    print()


# ──────────────────────────────────────────────
#  Quick self-test
# ──────────────────────────────────────────────


def _self_test() -> None:
    """Run a quick smoke test of architecture detection."""
    test_cases = [
        # (input, expected type, description)
        ("llama", ForwardPassType.DENSE, "LLaMA GGUF"),
        ("mistral", ForwardPassType.DENSE, "Mistral GGUF"),
        ("qwen2", ForwardPassType.QWEN2, "Qwen2 GGUF"),
        ("qwen3moe", ForwardPassType.MOE, "Qwen3 MoE GGUF"),
        ("gpt-oss", ForwardPassType.MOE, "GPT-OSS MoE"),
        ("mixtral", ForwardPassType.MOE, "Mixtral MoE"),
        ("gemma", ForwardPassType.GEMMA, "Gemma"),
        ("gemma2", ForwardPassType.GEMMA, "Gemma2"),
        ("deepseek2", ForwardPassType.MOE, "DeepSeek V2 MoE"),
        ("starcoder2", ForwardPassType.STARCODER, "StarCoder2"),
        ("command-r", ForwardPassType.COMMAND_R, "Cohere Command-R"),
        # HF config.json model_type values
        ({"model_type": "llama"}, ForwardPassType.DENSE, "HF LLaMA config"),
        ({"model_type": "qwen2"}, ForwardPassType.QWEN2, "HF Qwen2 config"),
        ({"model_type": "qwen3_moe"}, ForwardPassType.MOE, "HF Qwen3 MoE config"),
        ({"model_type": "mixtral"}, ForwardPassType.MOE, "HF Mixtral config"),
        ({"model_type": "gemma2"}, ForwardPassType.GEMMA, "HF Gemma2 config"),
        # ForwardPassType passthrough
        (ForwardPassType.MOE, ForwardPassType.MOE, "Passthrough MoE"),
        (ForwardPassType.DENSE, ForwardPassType.DENSE, "Passthrough DENSE"),
    ]

    all_ok = True
    for inp, expected, desc in test_cases:
        try:
            result = detect_architecture(inp)
            if result == expected:
                print(f"  OK  {desc:<35} -> {result.value}")
            else:
                print(
                    f"  FAIL {desc:<35} expected={expected.value} got={result.value}"
                )
                all_ok = False
        except Exception as e:
            print(f"  ERR  {desc:<35} exception={e}")
            all_ok = False

    print()
    if all_ok:
        print("  All self-tests passed!")
    else:
        print("  Some self-tests FAILED!")


if __name__ == "__main__":
    print_supported_models()
    print()
    _self_test()
