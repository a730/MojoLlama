"""Q4_0 quantized matmul — reference implementation.
Demonstrates block-by-block dot product without full dequantization.

Q4_0 block format:
  - 1× float16 scale (2 bytes)
  - 8× uint8 (2× 4-bit values each, 16 values total)
  - 10 bytes per 16 values

Usage:
    from q4_matmul import q4_matmul, Q4Block, quantize_q4, dequantize_q4
"""
import numpy as np
import struct
import time

# ─── Q4_0 Block ──────────────────────────────────────────────────────────────

class Q4Block:
    """A single Q4_0 quantization block (16 values, 10 bytes)."""
    
    def __init__(self, scale: np.float16 = np.float16(0.0), quants: np.ndarray = None):
        self.scale = scale
        # 8 bytes, each storing 2× 4-bit values
        self.quants = np.zeros(8, dtype=np.uint8) if quants is None else quants
    
    def dequantize(self) -> np.ndarray:
        """Dequantize block to 16 float32 values."""
        result = np.zeros(16, dtype=np.float32)
        for i in range(8):
            lo = self.quants[i] & 0x0F
            hi = (self.quants[i] >> 4) & 0x0F
            result[i*2] = (lo - 8) * float(self.scale)
            result[i*2+1] = (hi - 8) * float(self.scale)
        return result
    
    def dot(self, vec: np.ndarray, start: int = 0) -> float:
        """Compute dot product with 16 values starting at vec[start]."""
        total = 0.0
        s = float(self.scale)
        for i in range(8):
            lo = (self.quants[i] & 0x0F) - 8
            hi = (self.quants[i] >> 4) - 8
            total += (lo * vec[start + i*2] + hi * vec[start + i*2 + 1]) * s
        return total


def quantize_q4(values: np.ndarray) -> list[Q4Block]:
    """Quantize float32 array to Q4_0 blocks. Returns list of blocks."""
    n = len(values)
    n_blocks = (n + 15) // 16
    blocks = []
    for b in range(n_blocks):
        start = b * 16
        end = min(start + 16, n)
        chunk = values[start:end]
        if len(chunk) < 16:
            chunk = np.pad(chunk, (0, 16 - len(chunk)))
        # Find scale (absmax)
        amax = np.max(np.abs(chunk))
        scale = np.float16(amax / 7.0 if amax > 0 else 1.0)
        s = float(scale)
        # Quantize
        quants = np.zeros(8, dtype=np.uint8)
        for i in range(8):
            lo = max(0, min(15, int(round(chunk[i*2] / s)) + 8))
            hi = max(0, min(15, int(round(chunk[i*2+1] / s)) + 8))
            quants[i] = (hi << 4) | lo
        blocks.append(Q4Block(scale, quants))
    return blocks


def dequantize_q4(blocks: list[Q4Block], n: int) -> np.ndarray:
    """Dequantize blocks back to float32 array."""
    result = np.zeros(n, dtype=np.float32)
    for b_idx, block in enumerate(blocks):
        start = b_idx * 16
        end = min(start + 16, n)
        values = block.dequantize()
        result[start:end] = values[:end-start]
    return result


# ─── Q4_0 Matmul ─────────────────────────────────────────────────────────────

def pack_q4_weight(matrix: np.ndarray) -> list[list[Q4Block]]:
    """Convert float32 weight matrix to Q4_0 blocks per row.
    Returns list of rows, each row is list of blocks.
    """
    rows, cols = matrix.shape
    packed = []
    for r in range(rows):
        packed.append(quantize_q4(matrix[r]))
    return packed


def q4_matmul(x: np.ndarray, weight_blocks: list[list[Q4Block]], 
              out: np.ndarray = None) -> np.ndarray:
    """Compute y = x @ W where W is stored as Q4_0 blocks.
    
    Args:
        x: Input vector (batch, dim) or (dim,)
        weight_blocks: Weight matrix as packed Q4_0 blocks [rows][blocks]
        out: Pre-allocated output (optional)
    
    Returns:
        Output vector (batch, out_dim) or (out_dim,)
    """
    is_vector = x.ndim == 1
    if is_vector:
        x = x.reshape(1, -1)
    
    batch, dim = x.shape
    n_rows = len(weight_blocks)
    
    if out is None:
        out = np.zeros((batch, n_rows), dtype=np.float32)
    
    for b in range(batch):
        for r in range(n_rows):
            total = 0.0
            for block_idx, block in enumerate(weight_blocks[r]):
                start = block_idx * 16
                total += block.dot(x[b], start)
            out[b, r] = total
    
    return out.reshape(-1) if is_vector else out


def q4_matmul_transposed(x: np.ndarray, weight_blocks: list[list[Q4Block]],
                          out: np.ndarray = None) -> np.ndarray:
    """Compute y = x @ W.T where W is stored as Q4_0 blocks (per-row).
    This is the common case: W is (out_dim, in_dim), we want x @ W.T.
    
    Here weight_blocks[r] = blocks for output neuron r.
    So W.T has columns = blocks rows, dot with x columns.
    """
    # Same as q4_matmul since blocks are per-row of W
    return q4_matmul(x, weight_blocks, out)


# ─── Benchmark ───────────────────────────────────────────────────────────────

def benchmark():
    """Benchmark Q4_0 matmul vs float32 dequant + matmul."""
    np.random.seed(42)
    
    sizes = [
        (2048, 8192),   # FFN up/gate
        (2048, 2048),   # QKV projections
        (8192, 2048),   # FFN down
    ]
    
    print("=" * 70)
    print("Q4_0 Quantized Matmul Benchmark")
    print("=" * 70)
    
    for in_dim, out_dim in sizes:
        print(f"\n--- Matrix {out_dim}×{in_dim} ---")
        
        # Create float32 weight
        W_f32 = np.random.randn(out_dim, in_dim).astype(np.float32)
        print(f"  Float32 size: {W_f32.nbytes/1024/1024:.1f} MB")
        
        # Pack to Q4_0
        t0 = time.time()
        blocks = pack_q4_weight(W_f32)
        t1 = time.time()
        # Estimate Q4_0 size
        q4_bytes = out_dim * ((in_dim + 15)//16) * 10
        print(f"  Q4_0 size:    {q4_bytes/1024/1024:.1f} MB (packing: {t1-t0:.3f}s)")
        
        # Input vector
        x = np.random.randn(1, in_dim).astype(np.float32)
        
        # Warmup
        _ = x @ W_f32.T
        _ = q4_matmul(x, blocks)
        
        # Benchmark float32 matmul
        t0 = time.time()
        for _ in range(100):
            ref = x @ W_f32.T
        t1 = time.time()
        ref_ms = (t1 - t0) * 10  # ms per call
        
        # Benchmark Q4_0 matmul
        t0 = time.time()
        for _ in range(100):
            out = q4_matmul(x, blocks)
        t1 = time.time()
        q4_ms = (t1 - t0) * 10
        
        # Verify accuracy
        max_err = np.max(np.abs(out - ref))
        rel_err = max_err / np.max(np.abs(ref))
        
        print(f"  Float32:    {ref_ms:.1f}ms")
        print(f"  Q4_0 loop:  {q4_ms:.1f}ms")
        print(f"  Speedup:    {ref_ms/q4_ms:.1f}x" if q4_ms > 0 else "  Speedup: inf")
        print(f"  Max error:  {max_err:.4f} (relative: {rel_err:.4f})")
    
    # Compare full forward approach too
    print("\n\n--- Full Forward Comparison ---")
    print("Current: dequant entire tensor → float32 matmul")
    print("Q4_0:    dequant block-by-block → float32 dot → accumulate")
    
    in_dim, out_dim = 2048, 8192  # FFN layer
    W_f32 = np.random.randn(out_dim, in_dim).astype(np.float32)
    blocks = pack_q4_weight(W_f32)
    x = np.random.randn(1, in_dim).astype(np.float32)
    
    # Pre-dequant (current approach)
    t0 = time.time()
    for _ in range(100):
        W_q = blocks  # Simulate reading from cache
        _ = x @ W_f32.T
    t1 = time.time()
    predequant_ms = (t1 - t0) * 10
    print(f"  Pre-dequant + matmul: {predequant_ms:.1f}ms")
    
    # Q4_0 block-by-block (no full dequant)
    t0 = time.time()
    for _ in range(100):
        _ = q4_matmul(x, blocks)
    t1 = time.time()
    q4_block_ms = (t1 - t0) * 10
    print(f"  Q4_0 block-by-block:  {q4_block_ms:.1f}ms")
    print(f"  Ratio: {predequant_ms/q4_block_ms:.1f}x")
    
    print("\nDone.")


if __name__ == '__main__':
    benchmark()
