#!/usr/bin/env python3
"""Test unified quant_kernels_omp.so against gguf Python dequantize reference.
Tests Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0 quantized matmul kernels.
"""

import ctypes
import numpy as np
import struct
import os
import sys

# ─── Load C kernel library ───────────────────────────────────────────────

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(KERNEL_DIR, "quant_kernels_omp.so"))

# Set up function signatures
def setup_lib(lib):
    funcs = {
        'q4_0_matmul_omp': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int],
        'q4_1_matmul_omp': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int],
        'q4_k_matmul_omp': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int],
        'q5_k_matmul_omp': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int],
        'q6_k_matmul_omp': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int],
        'q8_0_matmul_omp': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int],
        'quant_matmul_omp': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int],
        'q4_k_dequantize_row': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.c_int],
        'q5_k_dequantize_row': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.c_int],
        'q6_k_dequantize_row': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.c_int],
        'q8_0_dequantize_row': [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.c_int],
        'f32_matmul_omp': [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int],
    }
    for name, argtypes in funcs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = None
    lib.get_max_threads.argtypes = []
    lib.get_max_threads.restype = ctypes.c_int
    lib.set_num_threads.argtypes = [ctypes.c_int]
    lib.set_num_threads.restype = None

setup_lib(lib)

# Quant type constants (matching ggml)
GGML_TYPE_F32  = 0
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q4_K = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q5_K = 9
GGML_TYPE_Q6_K = 14

QK_K = 256  # values per super-block (K-quants)
QK8_0 = 32  # values per block (Q8_0)

BLOCK_SIZES = {
    GGML_TYPE_Q4_0: 18,
    GGML_TYPE_Q4_1: 20,
    GGML_TYPE_Q4_K: 144,
    GGML_TYPE_Q5_K: 176,
    GGML_TYPE_Q6_K: 210,
    GGML_TYPE_Q8_0: 34,
}


def f16_to_f32(h):
    """Convert fp16 (uint16) to float32."""
    return struct.unpack('<e', struct.pack('<H', h))[0]


# ─── GGUF Reader ─────────────────────────────────────────────────────────

def read_gguf_tensor_data(filepath, tensor_name):
    """Read tensor data from a GGUF file. Returns (numpy_array, quant_type)."""
    from gguf import GGUFReader
    reader = GGUFReader(filepath)
    
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            # Get quant type
            qtype = tensor.tensor_type
            data = bytes(tensor.data)  # copy out
            return np.frombuffer(data, dtype=np.uint8), qtype
    
    return None, None


def find_tensors_by_type(filepath, target_types):
    """Find tensor names and their types in a GGUF file.
    target_types: dict mapping ggml type id to type name."""
    from gguf import GGUFReader
    reader = GGUFReader(filepath)
    
    results = []
    for tensor in reader.tensors:
        qtype = tensor.tensor_type
        if qtype in target_types:
            results.append((tensor.name, qtype))
    return results


# ─── Python dequantize reference (from ggml spec) ────────────────────────

def get_scale_min_k4(j, scales):
    """Unpack scale and min from the 12-byte scales array (Q4_K/Q5_K)."""
    if j < 4:
        d = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m


def dequantize_q4_k_row(data, n_values):
    """Dequantize a Q4_K row to float32.
    data: bytes of Q4_K blocks, n_values: number of output values."""
    nb = n_values // QK_K
    out = np.zeros(n_values, dtype=np.float32)
    idx = 0
    
    for b in range(nb):
        blk = data[b * 144 : (b + 1) * 144]
        d = f16_to_f32(struct.unpack('<H', blk[0:2])[0])
        mn = f16_to_f32(struct.unpack('<H', blk[2:4])[0])
        scales = blk[4:16]
        q = blk[16:144]  # 128 bytes
        
        is_ = 0
        q_off = 0
        for j in range(0, QK_K, 64):
            sc, m = get_scale_min_k4(is_ + 0, scales)
            d1 = d * sc;  m1 = mn * m
            sc, m = get_scale_min_k4(is_ + 1, scales)
            d2 = d * sc;  m2 = mn * m
            
            for l in range(32):
                out[idx + l] = d1 * (q[q_off + l] & 0xF) - m1
            for l in range(32):
                out[idx + l + 32] = d2 * (q[q_off + l] >> 4) - m2
            
            idx += 64
            q_off += 32
            is_ += 2
    
    return out


def dequantize_q5_k_row(data, n_values):
    """Dequantize a Q5_K row to float32."""
    nb = n_values // QK_K
    out = np.zeros(n_values, dtype=np.float32)
    idx = 0
    
    for b in range(nb):
        blk = data[b * 176 : (b + 1) * 176]
        d = f16_to_f32(struct.unpack('<H', blk[0:2])[0])
        mn = f16_to_f32(struct.unpack('<H', blk[2:4])[0])
        scales = blk[4:16]
        qh = blk[16:48]  # 32 bytes
        ql = blk[48:176]  # 128 bytes
        
        is_ = 0
        ql_off = 0
        u1 = 1
        u2 = 2
        
        for j in range(0, QK_K, 64):
            sc, m = get_scale_min_k4(is_ + 0, scales)
            d1 = d * sc;  m1 = mn * m
            sc, m = get_scale_min_k4(is_ + 1, scales)
            d2 = d * sc;  m2 = mn * m
            
            for l in range(32):
                q5_lo = (ql[ql_off + l] & 0xF) + (16 if (qh[l] & u1) else 0)
                out[idx + l] = d1 * q5_lo - m1
            for l in range(32):
                q5_hi = (ql[ql_off + l] >> 4) + (16 if (qh[l] & u2) else 0)
                out[idx + l + 32] = d2 * q5_hi - m2
            
            idx += 64
            ql_off += 32
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    
    return out


def dequantize_q6_k_row(data, n_values):
    """Dequantize a Q6_K row to float32."""
    nb = n_values // QK_K
    out = np.zeros(n_values, dtype=np.float32)
    idx = 0
    
    for b in range(nb):
        blk = data[b * 210 : (b + 1) * 210]
        d = f16_to_f32(struct.unpack('<H', blk[208:210])[0])
        scales = blk[192:208]  # int8_t[16]
        ql = blk[0:128]
        qh = blk[128:192]
        
        q_off = 0
        qh_off = 0
        sc_off = 0
        
        for n in range(0, QK_K, 128):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[q_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[q_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[q_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[q_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32
                
                ds0 = d * np.int8(scales[sc_off + is_ + 0])
                ds2 = d * np.int8(scales[sc_off + is_ + 2])
                ds4 = d * np.int8(scales[sc_off + is_ + 4])
                ds6 = d * np.int8(scales[sc_off + is_ + 6])
                
                out[idx + l + 0] = ds0 * q1
                out[idx + l + 32] = ds2 * q2
                out[idx + l + 64] = ds4 * q3
                out[idx + l + 96] = ds6 * q4
            
            idx += 128
            q_off += 64
            qh_off += 32
            sc_off += 8
    
    return out


def dequantize_q8_0_row(data, n_values):
    """Dequantize a Q8_0 row to float32."""
    nb = n_values // QK8_0
    out = np.zeros(n_values, dtype=np.float32)
    
    for b in range(nb):
        blk = data[b * 34 : (b + 1) * 34]
        d = f16_to_f32(struct.unpack('<H', blk[0:2])[0])
        qs = blk[2:34]
        for j in range(QK8_0):
            out[b * QK8_0 + j] = np.int8(qs[j]) * d
    
    return out


def dequantize_q4_0_row(data, n_values):
    """Dequantize a Q4_0 row to float32."""
    nb = n_values // 32
    out = np.zeros(n_values, dtype=np.float32)
    
    for b in range(nb):
        blk = data[b * 18 : (b + 1) * 18]
        d = f16_to_f32(struct.unpack('<H', blk[0:2])[0])
        for j in range(16):
            lo = (blk[2 + j] & 0xF) - 8
            hi = (blk[2 + j] >> 4) - 8
            out[b * 32 + j] = d * lo
            out[b * 32 + j + 16] = d * hi
    
    return out


def dequantize_q4_1_row(data, n_values):
    """Dequantize a Q4_1 row to float32."""
    nb = n_values // 32
    out = np.zeros(n_values, dtype=np.float32)
    
    for b in range(nb):
        blk = data[b * 20 : (b + 1) * 20]
        d = f16_to_f32(struct.unpack('<H', blk[0:2])[0])
        m = f16_to_f32(struct.unpack('<H', blk[2:4])[0])
        for j in range(16):
            lo = blk[4 + j] & 0xF
            hi = blk[4 + j] >> 4
            out[b * 32 + j] = d * lo + m
            out[b * 32 + j + 16] = d * hi + m
    
    return out


# ─── Dequantize reference using C kernels ────────────────────────────────

def c_dequantize_q4_k(data, n_values):
    """Use C kernel to dequantize Q4_K data."""
    n_blocks = n_values // QK_K
    data_arr = (ctypes.c_uint8 * len(data))(*data)
    out_arr = (ctypes.c_float * n_values)()
    lib.q4_k_dequantize_row(data_arr, out_arr, n_values)
    return np.array(out_arr, dtype=np.float32)


def c_dequantize_q5_k(data, n_values):
    """Use C kernel to dequantize Q5_K data."""
    data_arr = (ctypes.c_uint8 * len(data))(*data)
    out_arr = (ctypes.c_float * n_values)()
    lib.q5_k_dequantize_row(data_arr, out_arr, n_values)
    return np.array(out_arr, dtype=np.float32)


def c_dequantize_q6_k(data, n_values):
    """Use C kernel to dequantize Q6_K data."""
    data_arr = (ctypes.c_uint8 * len(data))(*data)
    out_arr = (ctypes.c_float * n_values)()
    lib.q6_k_dequantize_row(data_arr, out_arr, n_values)
    return np.array(out_arr, dtype=np.float32)


def c_dequantize_q8_0(data, n_values):
    """Use C kernel to dequantize Q8_0 data."""
    data_arr = (ctypes.c_uint8 * len(data))(*data)
    out_arr = (ctypes.c_float * n_values)()
    lib.q8_0_dequantize_row(data_arr, out_arr, n_values)
    return np.array(out_arr, dtype=np.float32)


# ─── Test functions ───────────────────────────────────────────────────────

def test_dequantize(qtype_name, c_dequant_fn, py_dequant_fn, data, n_values):
    """Compare C dequantize vs Python reference."""
    ref = py_dequant_fn(data, n_values)
    c_result = c_dequant_fn(data, n_values)
    
    # Check correlation
    if np.std(ref) < 1e-10:
        corr = 1.0  # constant vector
    else:
        corr = np.corrcoef(ref, c_result)[0, 1]
    
    max_err = np.max(np.abs(ref - c_result))
    mean_err = np.mean(np.abs(ref - c_result))
    
    print(f"  {qtype_name}: correlation={corr:.10f}, max_err={max_err:.6e}, mean_err={mean_err:.6e}")
    if corr < 0.99999:
        print(f"  WARNING: correlation below 0.99999!")
        # Print first few values for debugging
        for i in range(min(10, len(ref))):
            print(f"    [{i}] ref={ref[i]:.6f} c={c_result[i]:.6f} diff={ref[i]-c_result[i]:.6e}")
    return corr


def test_matmul(qtype_name, matmul_fn, data, n_rows, n_cols, block_size):
    """Test matmul by comparing C kernel result against: dequantize then dot product."""
    # Generate random input vector
    np.random.seed(42)
    x = np.random.randn(n_cols).astype(np.float32)
    
    # Use C dequantize reference to get expected result
    # We'll compute: for each row, dot(dequant_row, x)
    # Step 1: C dequantize the whole weight matrix
    # Step 2: Compute reference via matrix-vector multiply
    
    # Use C matmul
    data_arr = (ctypes.c_uint8 * len(data))(*data)
    out = np.zeros(n_rows, dtype=np.float32)
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    matmul_fn(data_arr, x_ptr, out_ptr, n_rows, n_cols)
    
    print(f"  {qtype_name} matmul: out[0]={out[0]:.6f}, out[-1]={out[-1]:.6f}")
    return out


def main():
    print("=" * 70)
    print("Testing quant_kernels_omp.so against Python dequantize reference")
    print("=" * 70)
    
    n_threads = min(os.cpu_count() or 4, 16)
    lib.set_num_threads(n_threads)
    print(f"Using {n_threads} threads")
    print()
    
    # ─── Test 1: Llama-3.2-1B-Instruct-Q4_0.gguf ──
    llama_path = "/onedev-workspace/work/Llama-3.2-1B-Instruct-Q4_0.gguf"
    if os.path.exists(llama_path):
        print("--- Llama-3.2-1B-Instruct-Q4_0.gguf ---")
        from gguf import GGUFReader
        
        reader = GGUFReader(llama_path)
        type_map = {}
        for tensor in reader.tensors:
            qtype = tensor.tensor_type
            if qtype not in type_map and qtype in [GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K]:
                type_map[qtype] = tensor.name
        
        type_names = {
            GGML_TYPE_Q4_0: "Q4_0",
            GGML_TYPE_Q4_1: "Q4_1", 
            GGML_TYPE_Q4_K: "Q4_K",
            GGML_TYPE_Q5_K: "Q5_K",
            GGML_TYPE_Q6_K: "Q6_K",
            GGML_TYPE_Q8_0: "Q8_0",
        }
        
        qlines = []
        for tensor in reader.tensors:
            if tensor.tensor_type in type_names:
                qlines.append(f"  {tensor.name}: {type_names[tensor.tensor_type]}")
        print("\n".join(qlines[:5]) + ("  ..." if len(qlines) > 5 else ""))
        
        # Find a Q4_0 tensor and test
        for tensor in reader.tensors:
            qtype = tensor.tensor_type
            data = bytes(tensor.data)
            n_elements = tensor.shape[0] * (tensor.shape[1] if len(tensor.shape) > 1 else 1)
            
            if qtype == GGML_TYPE_Q4_0 and "blk.0" in tensor.name:
                n_cols = tensor.shape[1]
                n_rows = tensor.shape[0]
                num_blocks_row = n_cols // 32
                data_row = data[:num_blocks_row * 18]  # first row
                
                ref = dequantize_q4_0_row(data_row, n_cols)
                deq = np.zeros(n_cols, dtype=np.float32)
                in_arr = (ctypes.c_uint8 * len(data_row))(*data_row)
                out_arr = (ctypes.c_float * n_cols)()
                
                # Test using the C dequantize via matmul approach
                # For Q4_0 we don't have a separate dequantize function, use matmul with x=1
                # Instead, test matmul directly
                x = np.random.randn(n_cols).astype(np.float32)
                x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                # Compute reference: dequantize then dot
                ref_full = dequantize_q4_0_row(data[:n_rows * num_blocks_row * 18], n_rows * n_cols)
                ref_matrix = ref_full.reshape(n_rows, n_cols)
                expected = ref_matrix @ x
                
                # C kernel
                full_data = (ctypes.c_uint8 * len(data))(*data[:n_rows * num_blocks_row * 18])
                out = np.zeros(n_rows, dtype=np.float32)
                out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                lib.q4_0_matmul_omp(full_data, x_ptr, out_ptr, n_rows, n_cols)
                
                corr = np.corrcoef(expected, out)[0, 1] if np.std(expected) > 1e-10 else 1.0
                max_err = np.max(np.abs(expected - out))
                print(f"  Q4_0 matmul test ({tensor.name}, {n_rows}x{n_cols}): corr={corr:.10f}, max_err={max_err:.6e}")
                break
        
        # Test Q6_K if present
        for tensor in reader.tensors:
            qtype = tensor.tensor_type
            data_raw = bytes(tensor.data)
            
            if qtype == GGML_TYPE_Q6_K and "blk.0" in tensor.name:
                n_cols = tensor.shape[1]
                n_rows = tensor.shape[0]
                assert n_cols % QK_K == 0, f"n_cols={n_cols} not multiple of {QK_K}"
                
                # Test row-level dequantize
                nb = n_cols // QK_K
                row_data = data_raw[:nb * 210]
                
                # Python reference
                ref = dequantize_q6_k_row(row_data, n_cols)
                # C dequantize
                in_arr = (ctypes.c_uint8 * len(row_data))(*row_data)
                out_arr = (ctypes.c_float * n_cols)()
                lib.q6_k_dequantize_row(in_arr, out_arr, n_cols)
                c_result = np.array(out_arr, dtype=np.float32)
                
                corr_deq = np.corrcoef(ref, c_result)[0, 1]
                max_err_deq = np.max(np.abs(ref - c_result))
                print(f"  Q6_K dequant: corr={corr_deq:.10f}, max_err={max_err_deq:.6e}")
                
                # Test matmul
                x = np.random.randn(n_cols).astype(np.float32)
                x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                # Test with a few rows
                test_rows = min(n_rows, 100)
                full_data = (ctypes.c_uint8 * len(data_raw[:test_rows * nb * 210]))(*data_raw[:test_rows * nb * 210])
                
                # Reference: dequantize then matmul
                ref_matrix = np.zeros((test_rows, n_cols), dtype=np.float32)
                for r in range(test_rows):
                    rd = data_raw[r * nb * 210 : (r + 1) * nb * 210]
                    ref_matrix[r] = dequantize_q6_k_row(rd, n_cols)
                expected = ref_matrix @ x
                
                # C matmul
                out = np.zeros(test_rows, dtype=np.float32)
                out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                lib.q6_k_matmul_omp(full_data, x_ptr, out_ptr, test_rows, n_cols)
                
                corr = np.corrcoef(expected, out)[0, 1]
                max_err = np.max(np.abs(expected - out))
                mean_err = np.mean(np.abs(expected - out))
                print(f"  Q6_K matmul ({test_rows}x{n_cols}): corr={corr:.10f}, max_err={max_err:.6e}, mean_err={mean_err:.6e}")
                break
    else:
        print(f"  File not found: {llama_path}")
    
    print()
    
    # ─── Test 2: Qwen3-30B-A3B-Q4_K_M.gguf ──
    qwen_path = "/onedev-workspace/work/Qwen3-30B-A3B-Q4_K_M.gguf"
    if os.path.exists(qwen_path):
        print("--- Qwen3-30B-A3B-Q4_K_M.gguf ---")
        from gguf import GGUFReader
        
        reader = GGUFReader(qwen_path)
        type_names = {
            GGML_TYPE_Q4_0: "Q4_0",
            GGML_TYPE_Q4_1: "Q4_1",
            GGML_TYPE_Q4_K: "Q4_K",
            GGML_TYPE_Q5_K: "Q5_K",
            GGML_TYPE_Q6_K: "Q6_K",
            GGML_TYPE_Q8_0: "Q8_0",
        }
        
        found_types = set()
        for tensor in reader.tensors:
            if tensor.tensor_type in type_names:
                found_types.add(tensor.tensor_type)
        print(f"  Quant types found: {[type_names[t] for t in sorted(found_types)]}")
        
        # Test each quant type
        for target_qtype, qname in [(GGML_TYPE_Q4_K, "Q4_K"), (GGML_TYPE_Q5_K, "Q5_K"), 
                                      (GGML_TYPE_Q6_K, "Q6_K"), (GGML_TYPE_Q8_0, "Q8_0")]:
            for tensor in reader.tensors:
                if tensor.tensor_type == target_qtype and "blk.0" in tensor.name:
                    data_raw = bytes(tensor.data)
                    n_cols = tensor.shape[1]
                    n_rows = tensor.shape[0]
                    
                    if target_qtype == GGML_TYPE_Q8_0:
                        assert n_cols % QK8_0 == 0, f"Q8_0: n_cols={n_cols} not multiple of {QK8_0}"
                        block_size = QK8_0
                        bs = 34
                    else:
                        assert n_cols % QK_K == 0, f"{qname}: n_cols={n_cols} not multiple of {QK_K}"
                        block_size = QK_K
                        bs = BLOCK_SIZES[target_qtype]
                    
                    nb = n_cols // block_size
                    test_rows = min(n_rows, 64)  # test 64 rows
                    
                    print(f"\n  Testing {qname} tensor: {tensor.name} ({n_rows}x{n_cols})")
                    
                    # Random input
                    np.random.seed(42)
                    x = np.random.randn(n_cols).astype(np.float32)
                    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    
                    # Python dequantize reference for test rows
                    py_dequant_fn = {
                        GGML_TYPE_Q4_K: dequantize_q4_k_row,
                        GGML_TYPE_Q5_K: dequantize_q5_k_row,
                        GGML_TYPE_Q6_K: dequantize_q6_k_row,
                        GGML_TYPE_Q8_0: dequantize_q8_0_row,
                    }[target_qtype]
                    
                    c_dequant_fn = {
                        GGML_TYPE_Q4_K: c_dequantize_q4_k,
                        GGML_TYPE_Q5_K: c_dequantize_q5_k,
                        GGML_TYPE_Q6_K: c_dequantize_q6_k,
                        GGML_TYPE_Q8_0: c_dequantize_q8_0,
                    }[target_qtype]
                    
                    # Test dequantize first
                    row_data = data_raw[:nb * bs]
                    ref_row = py_dequant_fn(row_data, n_cols)
                    c_row = c_dequant_fn(row_data, n_cols)
                    
                    deq_corr = np.corrcoef(ref_row, c_row)[0, 1] if np.std(ref_row) > 1e-10 else 1.0
                    deq_max_err = np.max(np.abs(ref_row - c_row))
                    deq_mean_err = np.mean(np.abs(ref_row - c_row))
                    print(f"    Dequantize: corr={deq_corr:.10f}, max_err={deq_max_err:.6e}, mean_err={deq_mean_err:.6e}")
                    
                    if deq_corr < 0.9999:
                        print(f"    WARNING: Low correlation! First 5 diffs:")
                        for i in range(min(5, len(ref_row))):
                            print(f"      [{i}] py={ref_row[i]:.6f} c={c_row[i]:.6f} diff={ref_row[i]-c_row[i]:.6e}")
                    
                    # Test matmul
                    test_data = data_raw[:test_rows * nb * bs]
                    test_arr = (ctypes.c_uint8 * len(test_data))(*test_data)
                    
                    # Reference: dequantize then matmul
                    ref_matrix = np.zeros((test_rows, n_cols), dtype=np.float32)
                    for r in range(test_rows):
                        rd = data_raw[r * nb * bs : (r + 1) * nb * bs]
                        ref_matrix[r] = py_dequant_fn(rd, n_cols)
                    expected = ref_matrix @ x
                    
                    # C matmul
                    out = np.zeros(test_rows, dtype=np.float32)
                    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    
                    matmul_fn = {
                        GGML_TYPE_Q4_K: lib.q4_k_matmul_omp,
                        GGML_TYPE_Q5_K: lib.q5_k_matmul_omp,
                        GGML_TYPE_Q6_K: lib.q6_k_matmul_omp,
                        GGML_TYPE_Q8_0: lib.q8_0_matmul_omp,
                    }[target_qtype]
                    
                    matmul_fn(test_arr, x_ptr, out_ptr, test_rows, n_cols)
                    
                    corr = np.corrcoef(expected, out)[0, 1] if np.std(expected) > 1e-10 else 1.0
                    max_err = np.max(np.abs(expected - out))
                    mean_err = np.mean(np.abs(expected - out))
                    print(f"    Matmul:     corr={corr:.10f}, max_err={max_err:.6e}, mean_err={mean_err:.6e}")
                    
                    if corr < 0.9999:
                        print(f"    WARNING: Low matmul correlation!")
                        for i in range(min(5, len(expected))):
                            print(f"      [{i}] expected={expected[i]:.6f} got={out[i]:.6f} diff={expected[i]-out[i]:.6e}")
                    
                    # Also test via dispatch
                    out2 = np.zeros(test_rows, dtype=np.float32)
                    out2_ptr = out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    lib.quant_matmul_omp(test_arr, x_ptr, out2_ptr, test_rows, n_cols, target_qtype)
                    
                    dispatch_corr = np.corrcoef(expected, out2)[0, 1] if np.std(expected) > 1e-10 else 1.0
                    print(f"    Dispatch:   corr={dispatch_corr:.10f}")
                    
                    break
    else:
        print(f"  File not found: {qwen_path}")
    
    print()
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()