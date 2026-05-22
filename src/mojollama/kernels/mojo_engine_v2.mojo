# MojoLlama Mojo Engine v2 — C FFI kernel calls
# Mojo orchestrates forward pass; C kernels do heavy math.
# Python interop only for GGUF model loading.
from std.prelude import *
from std.python import Python
from std import time

# ── C kernel FFI declarations ─────────────────────────────────
# These are in combined_engine.so (compiled from quant_kernels_omp.c)
@extern
fn rms_norm(x: Pointer[Float32, _], w: Pointer[Float32, _], 
            out: Pointer[Float32, _], n: Int32, eps: Float32): ...

@extern
fn quant_matmul_omp(W: Pointer[UInt8, _], x: Pointer[Float32, _],
                    out: Pointer[Float32, _], n_rows: Int32, 
                    n_cols: Int32, quant_type: Int32): ...

@extern
fn set_num_threads(n: Int32): ...

# ── Engine type ──────────────────────────────────────────────
comptime MAX_CTX: Int = 4096

struct MojoEngine:
    var model: PythonObject  # Python engine for loading
    var N: Int
    var n_head: Int
    var n_kv_head: Int
    var head_dim: Int
    var n_layers: Int
    var vocab_size: Int
    var is_moe: Bool
    var n_experts: Int
    var n_experts_per_tok: Int
    var n_ff: Int
    var n_ff_expert: Int
    
    # Buffers (raw memory)
    var x: Pointer[Float32, _]
    var xn: Pointer[Float32, _]
    var residual: Pointer[Float32, _]
    var q: Pointer[Float32, _]
    var k: Pointer[Float32, _]
    var att_out: Pointer[Float32, _]
    var gate: Pointer[Float32, _]
    var up: Pointer[Float32, _]
    var o_proj: Pointer[Float32, _]
    var ffn_out: Pointer[Float32, _]
    
    var _inited: Bool

    fn __init__(inout self, model_path: String, n_threads: Int32):
        # Load via Python interop
        var np = Python.import_module("numpy")
        var sys = Python.import_module("sys")
        sys.path.insert(0, "/onedev-workspace/work/src/mojollama/kernels")
        var te = Python.import_module("turbo_engine_v7_moe")
        var TurboEngineV7MoE = te.TurboEngineV7MoE
        
        self.model = TurboEngineV7MoE(model_path, Int(n_threads))
        
        # Copy config from Python engine
        var e = self.model
        self.N = e.n_embd.__index__()
        self.n_head = e.n_head.__index__()
        self.n_kv_head = e.n_kv_head.__index__()
        self.head_dim = e.head_dim.__index__()
        self.n_layers = e.n_layers.__index__()
        self.vocab_size = e.vocab_size.__index__()
        self.n_ff = e.n_ff.__index__() if Python.is_type(e.n_ff, "int") else 0
        self.is_moe = e.is_moe.__bool__() if Python.is_type(e.is_moe, "bool") else False
        self.n_experts = e.n_experts.__index__() if hasattr(e, "n_experts") else 0
        self.n_experts_per_tok = e.n_experts_per_tok.__index__() if hasattr(e, "n_experts_per_tok") else 0
        self.n_ff_expert = e.n_ff_expert.__index__() if hasattr(e, "n_ff_expert") else self.n_ff
        
        set_num_threads(n_threads)
        
        # Allocate buffers (use Python for now — bridge to Mojo later)
        var Nf = self.N
        # Store Python numpy arrays for now
        self.x = Pointer[Float32, _].alloc(Nf)
        self.xn = Pointer[Float32, _].alloc(Nf)
        self.residual = Pointer[Float32, _].alloc(Nf)
        self.q = Pointer[Float32, _].alloc(self.n_head * self.head_dim)
        self.k = Pointer[Float32, _].alloc(self.n_kv_head * self.head_dim)
        self.att_out = Pointer[Float32, _].alloc(self.n_head * self.head_dim)
        self.gate = Pointer[Float32, _].alloc(self.n_ff)
        self.up = Pointer[Float32, _].alloc(self.n_ff)
        self.o_proj = Pointer[Float32, _].alloc(Nf)
        self.ffn_out = Pointer[Float32, _].alloc(Nf)
        self._inited = True
    
    fn __del__(inout self):
        if self._inited:
            self.x.free()
            self.xn.free()
            self.residual.free()
            self.q.free()
            self.k.free()
            self.att_out.free()
            self.gate.free()
            self.up.free()
            self.o_proj.free()
            self.ffn_out.free()


fn main() raises:
    print("Mojo Engine v2 — C FFI")
    print("Loading model...")
    
    var eng = MojoEngine(
        "/tmp/models/gpt-oss-20b-Q4_K_M.gguf", Int32(8))
    
    print("Model: ", eng.n_layers, "L, ", eng.N, "D, ", 
           eng.n_head, "H, vocab=", eng.vocab_size)
    print("Engine ready!")
