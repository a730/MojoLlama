"""
Q4_0 quantized matmul — operates directly on raw Q4_0 blocks.
Never allocates the full float32 weight matrix.

Q4_0 block format (10 bytes per 16 values):
  bytes 0-1:   float16 scale (little-endian)
  bytes 2-9:   8 × uint8, each storing 2 × 4-bit values
               low nibble  = value at index 2i (even positions)
               high nibble = value at index 2i+1 (odd positions)
  dequant: value[i] = (nibble - 8) * scale

Usage:
    q4 = Q4Matmul.from_gguf_tensor(gguf_tensor)
    output = q4.forward_t(x)  # y = x @ W.T
"""
import numpy as np


class Q4Matmul:
    """Compute y = x @ W or x @ W.T where W is Q4_0 quantized per-row."""

    def __init__(self, raw_bytes: bytes, out_rows: int, in_cols: int):
        self.out_rows = out_rows
        self.in_cols = in_cols
        self.blocks_per_row = (in_cols + 15) // 16
        self.stride = self.blocks_per_row * 10  # bytes per row of blocks
        self.raw = np.frombuffer(raw_bytes, dtype=np.uint8)

    @classmethod
    def from_gguf_tensor(cls, tensor) -> "Q4Matmul":
        """Create from a GGUF reader tensor (Q4_0 type)."""
        raw = np.array(tensor.data)
        out_rows, in_cols = tensor.shape[0], tensor.shape[1]
        return cls(raw.tobytes(), out_rows, in_cols)

    # ─── Block processing helpers ──────────────────────────────────────────

    def _process_row_blocks(self, row_idx: int) -> tuple:
        """Dequantize all blocks for one row of the weight matrix.

        Returns:
            weight_vals: (blocks_per_row, 16) float32 — dequantized weight row
        """
        offset = row_idx * self.stride
        blocks_flat = self.raw[offset:offset + self.stride]
        blocks = blocks_flat.reshape(-1, 10)  # (n_blocks, 10)

        # Extract scales (first 2 bytes as float16)
        scales = blocks[:, :2].view(np.float16).ravel().astype(np.float32)

        # Extract and decode nibbles
        nibbles = blocks[:, 2:]  # (n_blocks, 8)
        lo = (nibbles & 0x0F).astype(np.float32) - 8.0
        hi = ((nibbles >> 4) & 0x0F).astype(np.float32) - 8.0

        # Interleave: lo_0, hi_0, lo_1, hi_1, ..., lo_7, hi_7
        weight_vals = np.empty((self.blocks_per_row, 16), dtype=np.float32)
        weight_vals[:, 0::2] = lo
        weight_vals[:, 1::2] = hi
        weight_vals *= scales[:, np.newaxis]  # broadcast: (n_blocks, 16)

        return weight_vals

    def _process_row_blocks_batched(self, rows: np.ndarray) -> np.ndarray:
        """Dequantize blocks for multiple rows at once.

        Args:
            rows: 1D array of row indices
        Returns:
            (len(rows), blocks_per_row, 16) float32 tensor
        """
        n_rows = len(rows)
        result = np.empty((n_rows, self.blocks_per_row, 16), dtype=np.float32)

        for i, r in enumerate(rows):
            result[i] = self._process_row_blocks(r)

        return result

    # ─── Forward passes ────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ W where W is stored row-by-row as Q4_0 blocks.

        W shape: (out_rows, in_cols)
        x shape: (batch, in_cols) or (in_cols,)

        Returns: (batch, out_rows) or (out_rows,)
        """
        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(1, -1)

        batch, in_cols = x.shape
        assert in_cols == self.in_cols, f"Input dim {in_cols} != weight dim {self.in_cols}"

        out = np.zeros((batch, self.out_rows), dtype=np.float32)
        x_blocks = x.reshape(batch, self.blocks_per_row, 16)

        for r in range(self.out_rows):
            w_row = self._process_row_blocks(r)  # (blocks_per_row, 16)
            # For each batch: sum over blocks of dot product
            out[:, r] = np.sum(w_row[np.newaxis, :, :] * x_blocks, axis=(1, 2))

        return out.reshape(-1) if is_1d else out

    def forward_t(self, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ W.T where W is (out_rows, in_cols) per-row Q4_0.

        forward() already computes y[b][r] = Σ_j x[b][j] * W[r][j]
        which equals (x @ W.T)[b][r]. So forward() is x @ W.T.

        For x @ W, use forward() with properly shaped x.
        """
        return self.forward(x)


# ─── Quantize helper (for testing) ──────────────────────────────────────────

def pack_q4_weight(matrix: np.ndarray) -> bytes:
    """Quantize a float32 weight matrix to Q4_0 format (byte string).

    Used for testing against float32 reference.
    """
    import struct
    out_rows, in_cols = matrix.shape
    blocks_per_row = (in_cols + 15) // 16
    stride = blocks_per_row * 10
    raw = bytearray(out_rows * stride)

    for r in range(out_rows):
        row = matrix[r]
        for b in range(blocks_per_row):
            start = b * 16
            chunk = row[start:start + 16]
            if len(chunk) < 16:
                chunk = np.pad(chunk, (0, 16 - len(chunk)))

            # Q4_0: scale = absmax / 7
            amax = np.max(np.abs(chunk))
            scale = np.float16(amax / 7.0 if amax > 0 else 1.0)
            s = float(scale)

            boff = (r * blocks_per_row + b) * 10
            struct.pack_into('<e', raw, boff, scale)

            for i in range(8):
                lo = max(0, min(15, int(round(chunk[i*2] / s)) + 8))
                hi = max(0, min(15, int(round(chunk[i*2+1] / s)) + 8))
                raw[boff + 2 + i] = (hi << 4) | lo

    return bytes(raw)


# ─── Test ───────────────────────────────────────────────────────────────────

def test_q4_matmul():
    """Test Q4Matmul against float32 reference."""
    np.random.seed(42)

    for out_rows, in_cols in [(64, 128), (128, 256), (256, 512)]:
        x = np.random.randn(1, in_cols).astype(np.float32)
        W_f32 = np.random.randn(out_rows, in_cols).astype(np.float32)

        # Reference: float32 matmul
        ref = x @ W_f32.T

        # Quantize to Q4_0
        raw = pack_q4_weight(W_f32)
        q4 = Q4Matmul(raw, out_rows, in_cols)
        result = q4.forward_t(x)

        max_err = np.max(np.abs(result - ref))
        rel_err = max_err / np.max(np.abs(ref))
        print(f"  {out_rows}×{in_cols}: max_err={max_err:.4f} rel_err={rel_err:.6f}")
        assert rel_err < 0.15, f"Q4_0 error too large: {rel_err:.4f}"

    # Test batched
    x_batch = np.random.randn(4, 128).astype(np.float32)
    W_f32 = np.random.randn(64, 128).astype(np.float32)
    ref = x_batch @ W_f32.T
    raw = pack_q4_weight(W_f32)
    q4 = Q4Matmul(raw, 64, 128)
    result = q4.forward_t(x_batch)
    rel_err = np.max(np.abs(result - ref)) / np.max(np.abs(ref))
    print(f"  batch=4 64×128: rel_err={rel_err:.6f}")
    assert rel_err < 0.15

    print("All tests PASS")


def benchmark_q4_matmul():
    """Benchmark Q4Matmul vs float32 dequant + matmul."""
    np.random.seed(42)

    sizes = [
        ("FFN gate", 2048, 8192),
        ("FFN down", 8192, 2048),
        ("QKV proj", 2048, 2048),
        ("Output",   2048, 128256),
    ]

    print(f"{'Name':<15} {'Shape':<18} {'F32 matmul':<14} {'Q4 matmul':<14} {'Speedup':<10} {'RelErr':<10}")
    print("-" * 81)

    for name, out_r, in_c in sizes:
        if out_r * in_c * 4 > 512 * 1024 * 1024:
            print(f"  {name:<15} {out_r}×{in_c:<10} SKIP (too large for RAM)")
            continue

        # Reference float32
        W_f32 = np.random.randn(out_r, in_c).astype(np.float32)
        x = np.random.randn(1, in_c).astype(np.float32)

        # Benchmark float32
        t0 = __import__('time').time()
        for _ in range(10):
            ref = x @ W_f32.T
        t1 = __import__('time').time()
        f32_ms = (t1 - t0) * 100  # ms per call

        # Q4_0 version
        raw = pack_q4_weight(W_f32)
        q4 = Q4Matmul(raw, out_r, in_c)

        t0 = __import__('time').time()
        for _ in range(10):
            result = q4.forward_t(x)
        t1 = __import__('time').time()
        q4_ms = (t1 - t0) * 100

        rel_err = np.max(np.abs(result - ref)) / np.max(np.abs(ref))

        speedup = f32_ms / q4_ms if q4_ms > 0 else float('inf')
        print(f"  {name:<15} {out_r}×{in_c:<12} {f32_ms:<12.2f} {q4_ms:<12.2f} {speedup:<8.2f}x {rel_err:.4f}")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        benchmark_q4_matmul()
    else:
        test_q4_matmul()
