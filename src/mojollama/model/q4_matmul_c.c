/* Q4_0 quantized matmul — GGUF format (block_size=32, type_size=18).
 * gcc -O3 -march=native -shared -fPIC -o libq4matmul.so q4_matmul_c.c
 *
 * Q4_0 block: [f16 scale (2 bytes)][16 bytes of nibbles (32 x 4-bit values)]
 *   scale: float16 in bytes 0-1 (little-endian)
 *   nibbles: bytes 2-17, each byte packs 2 values:
 *     byte[i] = (value[2i+1] << 4) | value[2i]   (low 4 bits = even index, high 4 bits = odd index)
 *   dequant: val[i] = (nibble - 8) * scale
 *   block_size = 32 values, type_size = 18 bytes
 */
#include <stdint.h>
#include <string.h>

/* Convert float16 bits (uint16) to float32 via IEEE 754 */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp = (uint32_t)((h >> 10) & 0x1f);
    uint32_t mant = (uint32_t)(h & 0x3ff);
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) { f32 = sign << 31; }
        else {
            int shift = 24 - __builtin_clz(mant);
            f32 = (sign << 31) | ((uint32_t)(113 - shift) << 23) | ((mant << (shift + 13)) & 0x7fffff);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

/* Block constants */
#define Q4_BLOCK_SIZE 32
#define Q4_TYPE_SIZE  18

/* Compute dot product of one Q4_0 block with 32 float32 values */
static inline float q4_block_dot(const uint8_t* block, const float* x) {
    uint16_t scale_bits;
    memcpy(&scale_bits, block, 2);
    float scale = f16_to_f32(scale_bits);

    float total = 0.0f;
    for (int i = 0; i < 16; i++) {
        uint8_t b = block[2 + i];
        float lo = (float)((int)(b & 0x0F) - 8);
        float hi = (float)((int)((b >> 4) & 0x0F) - 8);
        total += lo * scale * x[i * 2];
        total += hi * scale * x[i * 2 + 1];
    }
    return total;
}

/* Compute y = x @ W.T where W is stored as GGUF Q4_0 per row.
 * 
 * w_raw: raw bytes from GGUF tensor.data, shape (out_rows, N) where N = blocks_per_row * 18
 * x: float32 input (batch, in_cols) 
 * out: float32 output (batch, out_rows)
 * out_rows: number of output features (rows of W)
 * in_cols: number of input features (columns of W, must be multiple of 32)
 * batch: number of input vectors
 */
void q4_matmul_forward_t(const uint8_t* w_raw, const float* x, float* out,
                          int out_rows, int in_cols, int batch) {
    int blocks_per_row = (in_cols + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int row_stride = blocks_per_row * Q4_TYPE_SIZE;
    int cols_per_block = Q4_BLOCK_SIZE;
    
    for (int r = 0; r < out_rows; r++) {
        const uint8_t* row_ptr = w_raw + (size_t)r * row_stride;
        for (int b = 0; b < batch; b++) {
            const float* x_row = x + (size_t)b * in_cols;
            float sum = 0.0f;
            for (int blk = 0; blk < blocks_per_row; blk++) {
                sum += q4_block_dot(row_ptr + (size_t)blk * Q4_TYPE_SIZE,
                                    x_row + (size_t)blk * cols_per_block);
            }
            out[(size_t)b * out_rows + r] = sum;
        }
    }
}

/* Dequantize GGUF Q4_0 blocks to float32 */
void q4_dequant(const uint8_t* w_raw, float* out, int out_rows, int in_cols) {
    int blocks_per_row = (in_cols + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int row_stride = blocks_per_row * Q4_TYPE_SIZE;
    
    for (int r = 0; r < out_rows; r++) {
        const uint8_t* row_ptr = w_raw + (size_t)r * row_stride;
        float* out_row = out + (size_t)r * in_cols;
        for (int blk = 0; blk < blocks_per_row; blk++) {
            const uint8_t* block = row_ptr + (size_t)blk * Q4_TYPE_SIZE;
            uint16_t scale_bits;
            memcpy(&scale_bits, block, 2);
            float scale = f16_to_f32(scale_bits);
            for (int i = 0; i < 16; i++) {
                uint8_t b = block[2 + i];
                out_row[(size_t)blk * Q4_BLOCK_SIZE + (size_t)i * 2]     = (float)((int)(b & 0x0F) - 8) * scale;
                out_row[(size_t)blk * Q4_BLOCK_SIZE + (size_t)i * 2 + 1] = (float)((int)((b >> 4) & 0x0F) - 8) * scale;
            }
        }
    }
}
