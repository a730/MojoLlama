/* Q4_0 quantized matmul — GGUF format, multi-threaded via OpenMP.
 * gcc -O3 -march=native -fopenmp -shared -fPIC -o libq4matmul.so q4_matmul_c.c
 *
 * Q4_0 block: [f16 scale (2 bytes)][16 bytes nibbles (32 x 4-bit)]
 *   byte[i] = (value[2i+1] << 4) | value[2i]
 *   dequant: val[i] = (nibble - 8) * scale
 *   block_size = 32, type_size = 18
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define Q4_BLOCK_SIZE 32
#define Q4_TYPE_SIZE  18

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15), exp = (uint32_t)((h >> 10) & 0x1f), mant = (uint32_t)(h & 0x3ff), f32;
    if (exp == 0) {
        if (mant == 0) { f32 = sign << 31; }
        else { int shift = 24 - __builtin_clz(mant); f32 = (sign << 31) | ((uint32_t)(113 - shift) << 23) | ((mant << (shift + 13)) & 0x7fffff); }
    } else if (exp == 31) { f32 = (sign << 31) | 0x7f800000 | (mant << 13); }
    else { f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13); }
    float r; memcpy(&r, &f32, sizeof(r)); return r;
}

static inline float q4_block_dot(const uint8_t* block, const float* x) {
    uint16_t sb; memcpy(&sb, block, 2);
    float s = f16_to_f32(sb), total = 0.0f;
    for (int i = 0; i < 16; i++) {
        uint8_t b = block[2 + i];
        total += (float)((int)(b & 0x0F) - 8) * s * x[i * 2];
        total += (float)((int)((b >> 4) & 0x0F) - 8) * s * x[i * 2 + 1];
    }
    return total;
}

/* Multi-threaded Q4_0 matmul: y = x @ W.T */
void q4_matmul_forward_t(const uint8_t* w_raw, const float* x, float* out,
                          int out_rows, int in_cols, int batch) {
    int bpr = (in_cols + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int stride = bpr * Q4_TYPE_SIZE;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < out_rows; r++) {
        const uint8_t* row = w_raw + (size_t)r * stride;
        for (int b = 0; b < batch; b++) {
            const float* xr = x + (size_t)b * in_cols;
            float sum = 0.0f;
            for (int blk = 0; blk < bpr; blk++)
                sum += q4_block_dot(row + (size_t)blk * Q4_TYPE_SIZE, xr + (size_t)blk * Q4_BLOCK_SIZE);
            out[(size_t)b * out_rows + r] = sum;
        }
    }
}

/* Multi-threaded attention scores: scores[h][k] = q[h] @ K[k][h] / sqrt(d) */
void q4_attn_scores(const float* q, const float* k, float* out,
                     int n_head, int n_keys, int head_dim) {
    float rs = 1.0f / sqrtf((float)head_dim);
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < n_head; h++) {
        for (int key = 0; key < n_keys; key++) {
            float sum = 0.0f;
            const float* qh = q + (size_t)h * head_dim;
            const float* kh = k + (size_t)key * n_head * head_dim + (size_t)h * head_dim;
            for (int d = 0; d < head_dim; d++) sum += qh[d] * kh[d];
            out[(size_t)h * n_keys + key] = sum * rs;
        }
    }
}

/* In-place softmax */
void q4_softmax(float* scores, int n_head, int n_keys) {
    #pragma omp parallel for
    for (int h = 0; h < n_head; h++) {
        float* s = scores + (size_t)h * n_keys;
        float mx = s[0]; for (int k = 1; k < n_keys; k++) if (s[k] > mx) mx = s[k];
        float sum = 0.0f; for (int k = 0; k < n_keys; k++) { s[k] = expf(s[k] - mx); sum += s[k]; }
        float inv = 1.0f / sum; for (int k = 0; k < n_keys; k++) s[k] *= inv;
    }
}

/* Attention apply: out[h][d] = sum_k att[h][k] * v[k][h][d] */
void q4_attn_apply(const float* att, const float* v, float* out,
                    int n_head, int n_keys, int head_dim) {
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < n_head; h++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            const float* ah = att + (size_t)h * n_keys;
            for (int k = 0; k < n_keys; k++)
                sum += ah[k] * v[(size_t)k * n_head * head_dim + (size_t)h * head_dim + d];
            out[(size_t)h * head_dim + d] = sum;
        }
    }
}

/* RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight */
void q4_rms_norm(const float* x, const float* weight, float* out,
                  int rows, int cols, float eps) {
    #pragma omp parallel for
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * cols;
        float* orow = out + (size_t)r * cols;
        float ss = 0.0f;
        for (int c = 0; c < cols; c++) ss += xr[c] * xr[c];
        float inv = 1.0f / sqrtf(ss / (float)cols + eps);
        for (int c = 0; c < cols; c++) orow[c] = xr[c] * inv * weight[c];
    }
}

/* SiLU activation: out = x / (1 + exp(-x)) */
void q4_silu(const float* x, float* out, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) out[i] = x[i] / (1.0f + expf(-x[i]));
}
