# Phase 3: Q4_0 Direct Matmul Kernel

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Eliminate float32 weight dequantization from the inference loop. Compute matmuls directly on Q4_0 quantized data, reducing per-tensor memory from 8x to 1x and keeping working sets within available RAM (2.6 GB).

**Architecture:** Python prototype with NumPy vectorized Q4_0 matmul → Mojo kernel port. The Python prototype validates the approach and provides immediate speedup from reduced memory pressure; the Mojo port adds SIMD-level performance to approach llama.cpp's 82 tok/s.

**Current bottleneck per layer:**
1. `_tensor()` calls `gguf.dequantize(t.data, tt)` → expands 9 MB Q4_0 → 72 MB float32 (8x)
2. numpy `@` matmul uses the float32 version
3. gc.collect() frees it
- For 16 layers × 7 weight tensors = ~8 GB total float32 allocated sequentially in 2.6 GB RAM → swap thrashing

**Target approach:**
1. Keep weights in raw Q4_0 bytes (9 MB per tensor)
2. Compute `out[b][r] = Σ_block Σ_i dequant(block[i]) * x[b][block.start + i]`
3. Never allocate float32 weight tensors

**Tech Stack:** Python 3.11 + NumPy (prototype), Mojo 0.26.2 (kernel port)

---

### Task 1: Create `q4_kernels.py` — Raw Q4_0 matmul

**Objective:** A Python class that loads raw Q4_0 GGUF bytes and computes `x @ W` without dequantizing W to float32.

**Files:**
- Create: `src/mojollama/model/q4_kernels.py`
- Verify: existing `q4_matmul.py` (reference, kept for comparison)

**Step 1: Write `Q4Matmul` class**

```python
"""
Q4_0 quantized matmul — operates directly on raw Q4_0 blocks.
Never allocates the full float32 weight matrix.

Q4_0 block format (10 bytes per 16 values):
  bytes 0-1:   float16 scale (little-endian)
  bytes 2-9:   8 × uint8, each storing 2 × 4-bit values (low nibble = index 2i, high nibble = index 2i+1)
  dequant: value[i] = (nibble - 8) * scale
"""
import numpy as np
import struct

class Q4Matmul:
    """Compute y = x @ W where W is stored in Q4_0 format (per-row blocks)."""

    def __init__(self, raw_bytes: bytes, out_rows: int, in_cols: int):
        self.out_rows = out_rows
        self.in_cols = in_cols
        self.blocks_per_row = (in_cols + 15) // 16
        self.stride = self.blocks_per_row * 10  # bytes per row
        self.raw = np.frombuffer(raw_bytes, dtype=np.uint8)

    @classmethod
    def from_gguf_tensor(cls, tensor) -> "Q4Matmul":
        """Create from a GGUF reader tensor (Q4_0 type)."""
        raw = np.array(tensor.data)
        out_rows, in_cols = tensor.shape[0], tensor.shape[1]
        return cls(raw.tobytes(), out_rows, in_cols)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ W (or x @ W.T depending on how W is stored).

        Args:
            x: Input (batch, in_cols) or (in_cols,)
        Returns:
            Output (batch, out_rows) or (out_rows,)
        """
        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(1, -1)
        batch = x.shape[0]
        out = np.zeros((batch, self.out_rows), dtype=np.float32)

        batch_stride = 16  # process in batches of blocks for cache efficiency
        for r in range(self.out_rows):
            row_start = r * self.stride
            row_bytes = self.raw[row_start:row_start + self.stride]

            for b in range(batch):
                total = 0.0
                # Process blocks in SIMD-friendly batches
                for block_idx in range(self.blocks_per_row):
                    boff = block_idx * 10
                    scale = row_bytes[boff:boff+2].view(np.float16)[0]
                    nibbles = row_bytes[boff+2:boff+10]

                    # Decode nibbles to int4 and dequantize
                    lo = (nibbles & 0x0F).astype(np.float32) - 8.0
                    hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0

                    # Interleave: [lo0, hi0, lo1, hi1, ...]
                    values = np.empty(16, dtype=np.float32)
                    values[0::2] = lo
                    values[1::2] = hi
                    values *= float(scale)

                    col_start = block_idx * 16
                    total += np.dot(values, x[b, col_start:col_start + 16])

                out[b, r] = total

        return out.reshape(-1) if is_1d else out

    def forward_t(self, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ W.T where W is (out_rows, in_cols).
        
        Each output row r = dot(x, W[r]) where W[r] is row r of the weight matrix.
        Since blocks are stored per-row of W, forward() already computes x @ W.
        For x @ W.T, we need x @ W_col where W_col is column r of W.
        But W is stored row-major in Q4_0, so x @ W.T means:
        out[r] = sum_j x[j] * W[r][j]
        which is exactly what forward() gives us!
        
        Actually wait - forward computes out[b][r] = Σ_j x[b][j] * W[r][j] = (x @ W.T)[b][r]
        So forward() IS x @ W.T. We need a different function for x @ W.

        Use forward() for x @ W.T (the common case in transformer inference).
        """
        return self.forward(x)
```

**Step 2: Write unit test**

```python
"""Test q4_kernels.py against float32 reference."""
import numpy as np
from mojollama.model.q4_kernels import Q4Matmul

def test_q4_matmul():
    np.random.seed(42)
    out_rows, in_cols = 128, 256
    x = np.random.randn(1, in_cols).astype(np.float32)

    # Reference: float32 matmul
    W_f32 = np.random.randn(out_rows, in_cols).astype(np.float32)
    ref = x @ W_f32.T

    # Quantize to Q4_0 manually and create Q4Matmul
    # ... (quantization logic from q4_matmul.py)
    blocks_per_row = (in_cols + 15) // 16
    raw = bytearray(out_rows * blocks_per_row * 10)
    for r in range(out_rows):
        row = W_f32[r]
        for b in range(blocks_per_row):
            start = b * 16
            chunk = row[start:start + 16]
            if len(chunk) < 16:
                chunk = np.pad(chunk, (0, 16 - len(chunk)))
            amax = np.max(np.abs(chunk))
            scale = np.float16(amax / 7.0 if amax > 0 else 1.0)
            boff = (r * blocks_per_row + b) * 10
            struct.pack_into('<e', raw, boff, scale)
            s = float(scale)
            for i in range(8):
                lo = max(0, min(15, int(round(chunk[i*2] / s) + 8)))
                hi = max(0, min(15, int(round(chunk[i*2+1] / s) + 8)))
                raw[boff + 2 + i] = (hi << 4) | lo

    q4 = Q4Matmul(bytes(raw), out_rows, in_cols)
    result = q4.forward_t(x)

    max_err = np.max(np.abs(result - ref))
    rel_err = max_err / np.max(np.abs(ref))
    print(f"Max error: {max_err:.4f}, relative: {rel_err:.4f}")
    assert rel_err < 0.1, f"Q4_0 error too large: {rel_err:.4f}"
    print("PASS")

if __name__ == "__main__":
    test_q4_matmul()
```

**Step 3: Run the test**

Run: `cd /onedev-workspace/work && python3 -c "from mojollama.model.q4_kernels import Q4Matmul; import numpy as np, struct; <test code>"`

Expected: PASS (relative error < 0.1)

---

### Task 2: Integrate Q4Matmul into inference.py

**Objective:** Replace dequant-then-matmul with direct Q4_0 matmul for all Q4_0 weight tensors.

**Files:**
- Modify: `src/mojollama/model/inference.py`
- Test: `src/mojollama/model/inference.py` via `python3 src/mojollama/model/inference.py <model.gguf>`

**Key changes:**

1. Import Q4Matmul at top:
```python
from mojollama.model.q4_kernels import Q4Matmul
```

2. Add `_q4_matmul` cache dict to `__init__`:
```python
self._q4_matmuls = {}  # gguf_name -> Q4Matmul
```

3. Add method to get or create Q4Matmul:
```python
def _get_q4_matmul(self, gguf_name: str):
    if gguf_name not in self._q4_matmuls:
        t = self._tensors_by_name.get(gguf_name)
        if t is None:
            return None
        self._q4_matmuls[gguf_name] = Q4Matmul.from_gguf_tensor(t)
    return self._q4_matmuls[gguf_name]
```

4. Modify `_tensor()`: only dequantize non-Q4_0 tensors (norms, biases, embed). For Q4_0 weights, skip dequant and return None — the matmul happens via Q4Matmul.

Actually, a cleaner approach: keep `_tensor()` for non-weight tensors (norms, biases, output) and create a new method for weight matmul:

```python
def _apply_weight(self, x: np.ndarray, gguf_name: str, transposed: bool = True) -> np.ndarray:
    """Compute x @ W or x @ W.T where W is a Q4_0 weight matrix."""
    q4 = self._get_q4_matmul(gguf_name)
    if transposed:
        return q4.forward_t(x)
    return q4.forward(x)
```

5. Replace all `x @ q_w.T` with `self._apply_weight(x, f'blk.{i}.attn_q.weight')` etc.

**Note on performance:** The `_tensor()` method is no longer called for weight tensors (just norms/embed), so the 8x dequant expansion is completely eliminated. Weights stay in 9 MB Q4_0 format permanently.

---

### Task 3: Run inference end-to-end and verify correctness

**Objective:** Verify the model produces the same (approximately) output.

**Files:**
- Run: `src/mojollama/model/inference.py`

**Steps:**
1. Run with a simple prompt and compare output tokens with baseline
2. Verify KV cache still works correctly
3. Measure timing

---

### Task 4: Benchmark against baseline and llama.cpp

**Objective:** Measure speedup from Q4_0 matmul.

**Steps:**
1. Time 10-token generation (prefill + generation steps)
2. Compare with previous baseline (~57.8s for 6-token prefill + 5-token gen)
3. Compare with llama.cpp at 82 tok/s
4. Report results

---

### Task 5: Profile and optimize hot paths (within scope)

**Objective:** Find and fix the slowest parts of the Q4_0 matmul.

**Potential optimizations:**
1. Pre-compute dequantized scales for each block (scale is float16, 2 bytes per block)
2. Use `np.dot` with pre-dequantized scales to avoid Python loop per block
3. Process multiple rows in parallel via vectorized operations

---

### Task 6: Mojo kernel prototype (stretch goal)

**Objective:** Port the inner loop to Mojo for SIMD acceleration.

**Files:**
- Create: `src/mojollama/model/q4_kernel.mojo`

**Approach:**
- Write a Mojo function that takes raw Q4_0 block data + float32 input
- Uses Mojo's SIMD vector types for block processing
- Called from Python via `Python.evaluate()` or compiled to shared library
