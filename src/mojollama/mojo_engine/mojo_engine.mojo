# MojoLlama Mojo Engine — clean Python bridge
# Systematic port: Python ops that can't be ported (numpy, GGUF)
# are wrapped via interop. Engine API is pure Mojo.
from std.prelude import *
from std.python import Python, PythonObject
from std.time import perf_counter

struct Engine:
    var eng: PythonObject
    var N: Int
    var L: Int  
    var NH: Int
    var V: Int
    var n_threads: Int32
    
    fn __init__(inout self, path: String, threads: Int32):
        try:
        self.n_threads = threads
        var np = Python.import_module("numpy")
        var sys = Python.import_module("sys")
        sys.path.insert(0, "/onedev-workspace/work/src/mojollama/kernels")
        
        var te = Python.import_module("turbo_engine_v7_moe")
        self.eng = te.TurboEngineV7MoE(path, Int(threads))
        self.N = Int(self.eng.n_embd.__index__())
        self.L = Int(self.eng.n_layers.__index__())
        self.NH = Int(self.eng.n_head.__index__())
        self.V = Int(self.eng.vocab_size.__index__())
    
    fn forward(inout self, token: Int) -> PythonObject:
        try:
        return self.eng.forward(Python.list(token))
    
    fn bench(inout self, n: Int) raises -> Float64:
        var t0 = perf_counter()
        for i in range(n):
            var _ = self.forward(i % 1000)
        var t1 = perf_counter()
        return Float64(n) / (t1 - t0)

fn main() raises:
    print("MojoLlama Mojo Engine — Bridge API")
    print("====================================")
    
    var e = Engine("/tmp/models/gpt-oss-20b-Q4_K_M.gguf", 8)
    print("Engine: ", e.L, "L, ", e.N, "D, ", e.V, "vocab")
    
    var np = Python.import_module("numpy")
    var l0 = e.forward(0)
    print("Forward: ", "OK" if not np.any(np.isnan(l0)) else "NaN")
    
    var tok_s = e.bench(10)
    print("Throughput: ", tok_s, " tok/s")
