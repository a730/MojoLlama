from abc import ABC, abstractmethod

class ArchitectureForwardPass(ABC):
    """Base class for per-architecture forward passes."""

    def __init__(self, engine):
        self.engine = engine  # TurboEngineV7MoE reference

    @abstractmethod
    def forward(self, token_id):
        """Single-token forward pass. Returns logits."""
        ...

    def init_weights(self, weights, raw_weights, weight_info, weight_qtypes):
        """Override to load architecture-specific weights. Called during _load_weights."""
        pass

    def init_pointers(self, layer_idx, pfx, lw):
        """Override to pre-compute ctypes pointers for a layer. Called during _preload_pointers."""
        pass

    def init_buffers(self):
        """Override to allocate per-architecture buffers. Called during _init_buffers."""
        pass

    def reset_state(self):
        """Override to reset per-architecture state. Called during reset/reset_state."""
        pass
