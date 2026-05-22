from std.prelude import *
from std.python import Python
from std.time import perf_counter

fn main() raises:
    var np = Python.import_module("numpy")
    print("NumPy version: ", np.__version__)
    
    var sys = Python.import_module("sys")
    sys.path.insert(0, "/onedev-workspace/work/src/mojollama/kernels")
    
    var te = Python.import_module("turbo_engine_v7_moe")
    print("Module loaded: ", te.__name__)
    
    var TurboEngineV7MoE = te.TurboEngineV7MoE
    var e = TurboEngineV7MoE("/tmp/models/gpt-oss-20b-Q4_K_M.gguf", 8)
    
    var l0 = e.forward(0)
    var has_nan = np.any(np.isnan(l0))
    print("Forward OK, NaN: ", has_nan)
    
    var t0 = perf_counter()
    for _ in range(10):
        var _ = e.forward(0)
    var t1 = perf_counter()
    print("Throughput: ", Float64(10) / (t1 - t0), " tok/s")
