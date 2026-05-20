#!/usr/bin/env python3
"""MojoLlama parallel Q4_0 matmul — numpy + multiprocessing via shared mem.

Uses multiprocessing.shared_memory (Python 3.8+) for zero-copy data sharing.
Workers attach to the same shared memory blocks and write their row chunks.
"""

import os
import sys
import time
import json
import argparse
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np


def generate_q4_weights(n_rows, n_cols, seed=42):
    """Generate synthetic Q4_0 weight data for benchmarking."""
    rng = np.random.RandomState(seed)
    n_blocks = n_rows * (n_cols // 32)
    weights = np.zeros(n_blocks * 18, dtype=np.uint8)
    for b in range(n_blocks):
        off = b * 18
        scale_f16 = np.float16(rng.randn())
        scale_bytes = np.frombuffer(scale_f16.tobytes(), dtype=np.uint8)
        weights[off:off+2] = scale_bytes
        nibbles = rng.randint(0, 256, size=16, dtype=np.uint8)
        weights[off+2:off+18] = nibbles
    return weights


def dequant_block(block_bytes):
    """Dequantize one Q4_0 block: 32 × 4-bit → 32 × float32."""
    scale = np.frombuffer(block_bytes[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
    nibbles = np.frombuffer(block_bytes[2:18].tobytes(), dtype=np.uint8)
    lo = nibbles & 0x0f
    hi = (nibbles >> 4) & 0x0f
    vals = np.empty(32, dtype=np.float32)
    vals[0::2] = (lo.astype(np.int32) - 8).astype(np.float32)
    vals[1::2] = (hi.astype(np.int32) - 8).astype(np.float32)
    return vals * scale


def q4_matmul_row(weights, inp_vec, n_cols, row_idx):
    """Q4_0 matmul for one output row."""
    blocks_per_row = n_cols // 32
    total = 0.0
    for blk in range(blocks_per_row):
        off = (row_idx * blocks_per_row + blk) * 18
        block = weights[off:off+18]
        dequantized = dequant_block(block)
        x = inp_vec[blk*32:(blk+1)*32]
        total += np.dot(dequantized, x)
    return total


def worker(shm_w_name, shm_inp_name, shm_out_name, w_size, inp_size, out_size,
           n_rows, n_cols, start_row, end_row, worker_id):
    """Worker process: attach to shared memory, compute chunk."""
    # Attach to existing shared memory
    shm_w = shared_memory.SharedMemory(name=shm_w_name)
    shm_inp = shared_memory.SharedMemory(name=shm_inp_name)
    shm_out = shared_memory.SharedMemory(name=shm_out_name)

    weights = np.frombuffer(shm_w.buf[:w_size], dtype=np.uint8)
    inp_vec = np.frombuffer(shm_inp.buf[:inp_size], dtype=np.float32)
    out_buf = np.frombuffer(shm_out.buf[:out_size], dtype=np.float32)

    for row in range(start_row, end_row):
        out_buf[row] = q4_matmul_row(weights, inp_vec, n_cols, row)

    shm_w.close()
    shm_inp.close()
    shm_out.close()


def q4_matmul_parallel(weights, inp_vec, n_rows, n_cols, n_threads=None):
    """Parallel Q4_0 matmul via multiprocessing with shared memory."""
    if n_threads is None:
        n_threads = mp.cpu_count()

    w_size = len(weights)
    inp_size = n_cols * 4  # float32
    out_size = n_rows * 4

    try:
        shm_w = shared_memory.SharedMemory(create=True, size=w_size)
        shm_inp = shared_memory.SharedMemory(create=True, size=inp_size)
        shm_out = shared_memory.SharedMemory(create=True, size=out_size)
    except FileExistsError:
        # Cleanup stale shm
        shm_w = shared_memory.SharedMemory(name="mojo_q4_weights")
        shm_w.unlink()
        shm_inp = shared_memory.SharedMemory(name="mojo_q4_input")
        shm_inp.unlink()
        shm_out = shared_memory.SharedMemory(name="mojo_q4_output")
        shm_out.unlink()
        shm_w = shared_memory.SharedMemory(create=True, size=w_size)
        shm_inp = shared_memory.SharedMemory(create=True, size=inp_size)
        shm_out = shared_memory.SharedMemory(create=True, size=out_size)

    # Copy data into shared memory
    shm_w.buf[:w_size] = weights.tobytes()
    shm_inp.buf[:inp_size] = inp_vec.tobytes()
    shm_out.buf[:out_size] = b'\x00' * out_size

    # Split into chunks
    chunk_size = max(1, n_rows // n_threads)
    chunks = []
    start = 0
    while start < n_rows:
        end = min(start + chunk_size, n_rows)
        chunks.append((start, end))
        start = end

    # Launch workers
    processes = []
    for i, (s, e) in enumerate(chunks):
        p = mp.Process(target=worker, args=(
            shm_w.name, shm_inp.name, shm_out.name,
            w_size, inp_size, out_size, n_rows, n_cols, s, e, i
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Read result (copy detaches from shared memory)
    result = np.frombuffer(shm_out.buf[:out_size], dtype=np.float32).copy()

    # Cleanup
    shm_w.close(); shm_inp.close(); shm_out.close()
    for s in [shm_w, shm_inp, shm_out]:
        try: s.unlink()
        except: pass

    return result


def benchmark(n_rows=2048, n_cols=2048, n_threads=None):
    """Run Q4_0 matmul benchmark."""
    if n_threads is None:
        n_threads = mp.cpu_count()

    print(f"Q4_0 Matmul Benchmark: {n_rows}×{n_cols}")
    print(f"System: {mp.cpu_count()} logical CPUs, using {n_threads} workers")
    print(f"Weight size: {n_rows * (n_cols // 32) * 18 / 1024**2:.1f} MB")
    print()

    # Generate weights
    print("Generating test data...")
    weights = generate_q4_weights(n_rows, n_cols)
    inp_vec = np.random.randn(n_cols).astype(np.float32)

    # Sequential baseline
    print(f"\nSequential (1 thread)...")
    t0 = time.time()
    seq_result = np.zeros(n_rows, dtype=np.float32)
    for i in range(n_rows):
        seq_result[i] = q4_matmul_row(weights, inp_vec, n_cols, i)
    seq_time = time.time() - t0
    seq_throughput = 1.0 / seq_time
    print(f"  Time: {seq_time*1000:.1f} ms")
    print(f"  Throughput: {seq_throughput:.2f} matmul/s")
    print(f"  Result[0]={seq_result[0]:.4f}, Result[-1]={seq_result[-1]:.4f}")

    # Parallel
    print(f"\nParallel ({n_threads} workers)...")
    t0 = time.time()
    par_result = q4_matmul_parallel(weights, inp_vec, n_rows, n_cols, n_threads)
    par_time = time.time() - t0
    par_throughput = 1.0 / par_time
    speedup = seq_time / par_time if par_time > 0 else 0

    print(f"  Time: {par_time*1000:.1f} ms")
    print(f"  Throughput: {par_throughput:.2f} matmul/s")
    print(f"  Speedup vs sequential: {speedup:.1f}x")
    print(f"  Result[0]={par_result[0]:.4f}, Result[-1]={par_result[-1]:.4f}")

    # Verify
    max_diff = np.max(np.abs(seq_result - par_result))
    print(f"\nMax difference: {max_diff:.6f}")
    print("✅ Results match!" if max_diff < 0.01 else "⚠  Results differ")

    return {
        "n_rows": n_rows, "n_cols": n_cols, "n_threads": n_threads,
        "seq_time_ms": seq_time * 1000, "par_time_ms": par_time * 1000,
        "speedup": speedup,
        "seq_throughput": seq_throughput, "par_throughput": par_throughput,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MojoLlama parallel Q4_0 matmul")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    result = benchmark(args.rows, args.cols, args.threads)
    if result:
        print(f"\n{'='*50}")
        print(f"Benchmark: {result['n_rows']}x{result['n_cols']} Q4_0")
        print(f"  Sequential: {result['seq_time_ms']:.1f} ms ({result['seq_throughput']:.2f} matmul/s)")
        print(f"  Parallel ({result['n_threads']} workers): {result['par_time_ms']:.1f} ms ({result['par_throughput']:.2f} matmul/s)")
        print(f"  Speedup: {result['speedup']:.1f}x")
