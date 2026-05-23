# Mojo Benchmark Tool: General-Purpose Design

Research findings from mojolang.org on how to make a benchmark that works
for any future model architecture.

## 1. Use `Bench` + `BenchConfig` (std.benchmark)

The `std.benchmark` module has a full framework:

```mojo
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, ThroughputMeasure, BenchMetric

var config = BenchConfig(
    min_runtime_secs=0.1,    # run at least 100ms
    max_runtime_secs=5.0,    # cap at 5s per benchmark
    num_warmup_iters=2,      # warmup iterations
    max_iters=100,           # run up to 100 iterations
    num_repetitions=3,       # repeat 3 times for statistics
    format=Format.table,     # pretty-print table
)
var bench = Bench(config)
```

**Issue found:** `Bencher` uses `alloc` internally, which crashes after
weight loading exhausts Mojo's stack allocator. **Workaround:** Use
`bench_function` with `fixed_iterations` instead of `Bencher`, or
use `time.perf_counter()` directly (current working approach).

## 2. Generic Architecture via `comptime` Parameters

Make model architecture a compile-time parameter:

```mojo
comptime NE: Int = 2048   # n_embd (feature dim)
comptime NH: Int = 32     # n_heads
comptime NK: Int = 4      # n_kv_heads
comptime HD: Int = 64     # head_dim
comptime NL: Int = 22     # n_layers
comptime NF: Int = 5632   # ff_hidden (intermediate)
comptime NV: Int = 32000  # vocab_size
```

The benchmark reads these comptime values and selects appropriate
matmul shapes — no hardcoded model-specific logic needed.

## 3. `@parameter` Decorator for Compile-Time Specialization

Use `@parameter` for benchmark functions to enable compile-time
specialization by the compiler:

```mojo
@parameter
def bench_matmul[T: Int](nr: Int, nc: Int, nw: Int):
    # T is the SIMD width, specialized at compile time
    ...
```

## 4. Auto-Detect with `simd_width_of`

```mojo
from std.sys.info import simd_width_of

comptime SIMD_W = simd_width_of[DType.float32]()
# Returns 4, 8, or 16 depending on HW
```

This makes the benchmark adapt to the host CPU's SIMD capabilities
without manual configuration.

## 5. Multi-Shape Matmul Sweep (Not Just LM Head)

For a general-purpose benchmark, time ALL key matmul shapes:

| Matmul | Shape | % of total |
|--------|-------|-----------|
| Q_proj | NH*HD × NE | ~3% |
| K_proj | NK*HD × NE | ~1% |
| V_proj | NK*HD × NE | ~1% |
| O_proj | NE × NH*HD | ~3% |
| FFN_gate | NF × NE | ~10% |
| FFN_up | NF × NE | ~10% |
| FFN_down | NE × NF | ~10% |
| LM_head | NV × NE | ~6% |

Compute weighted tok/s estimate: `tok_s = 1.0 / sum(ms_per_matmul * weight)`

## 6. Cold-Cache Methodology (More Realistic)

Use a weight pool larger than L3 cache (Threadripper: 128MB) and
advance pointer per matmul to force cold reads:

```mojo
var pool_size = 256_000_000  # 256MB > L3 cache
var pool = _alc(pool_size)
var wp = pool
for each matmul:
    time f16_matmul(wp, x, o, nr, nc, nw)
    wp += nr * nc * 2  # advance to next region
```

## 7. Batch Size Sweep via Comptime

Since B is a comptime constant, build multiple versions:

```bash
# Build with different B values
sed 's/comptime B: Int = 1/comptime B: Int = 4/' benchmark.mojo > bench_b4.mojo
mojo build bench_b4.mojo
```

Or use `comptime` to generate B=1, B=2, B=4 kernels in one binary:

```mojo
comptime for b in [1, 2, 4, 8]:
    # Each iteration generates specialized kernel for batch size B
    run_bench[b]()
```

## 8. Structured Output (CSV/JSON)

Use `Bench.dump_report()` to write structured results to a file:

```mojo
bench.config.out_file = Path("/tmp/bench_results.csv")
bench.config.format = Format.tabular
bench.dump_report()
```

## 9. Architecture Database via Comptime Struct

Define model architectures as comptime data:

```mojo
struct ModelArch:
    var name: String
    var NE: Int
    var NH: Int
    var NK: Int
    var HD: Int
    var NL: Int
    var FF: Int
    var NV: Int
    var is_moe: Bool
    var n_exp: Int
    var n_act: Int

comptime MODELS = ModelArch(
    "TinyLlama", 2048, 32, 4, 64, 22, 5632, 32000, False, 0, 0
),
```

Benchmark iterates over all models in the comptime database.

## Summary: Next-Gen Benchmark Design

```
┌─────────────────────────────────────────────┐
│           benchmark_all.mojo                  │
│                                               │
│  comptime MODEL_DB = [                       │
│    {"TinyLlama", 2048, 32, 4, 64, 22,...},  │
│    {"ZAYA1-8B", 2048, 8, 2, 128, 80,...},   │
│    {"GPT-OSS", 2880, 64, 8, 64, 24,...},    │
│  ]                                            │
│                                               │
│  for model in MODEL_DB:                       │
│    allocate_synthetic_weights(model)          │
│    for nw in [1,2,4,8,16,24,32]:             │
│      for shape in [Q,K,V,O,G,U,D,LM]:        │
│        bench.bench_function(...)              │
│    bench.dump_report()                        │
│                                               │
│  Output: CSV/JSON table with                  │
│  model, batch, threads, matmul, ms, GB/s     │
└─────────────────────────────────────────────┘
```

## Limitations in Mojo 1.0.0b1

1. **`Bencher.alloc` crash** — `Bencher` uses Mojo `alloc` internally.
   After weight loading (which exhausts Mojo's stack allocator), Bencher
   crashes. Workaround: use `time.perf_counter()` directly.
2. **No runtime `B` changes** — Batch size is comptime. Need separate
   build per batch size or use `comptime for` loop.
3. **`String` broken** — Mojo 1.0.0b1 has issues with `String` indexing
   and UTF-8. Use `UnsafePointer[UInt8]` for raw data.
4. **No file I/O at comptime** — Can't read model config. Must hardcode.
