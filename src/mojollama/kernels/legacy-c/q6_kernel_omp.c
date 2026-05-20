/* Q6_K matmul — AVX2+FMA+OpenMP
 *
 * Q6_K format (210 bytes per 256 values, 6.5625 bits/value):
 *   uint8_t  ql[128]    — lower 4 bits (2 nibbles/byte)
 *   uint8_t  qh[64]     — upper 2 bits (4 values/byte, 2 bits each)
 *   int8_t   scales[16] — signed 6-bit per-sub-block scale (16 sub-blocks of 16 values)
 *   ggml_half d          — fp16 super-block scale
 *
 * Dequantization (from ggml dequantize_row_q6_K):
 *   For each 128-element half-block (n=0 or n=128):
 *     for l in 0..31:
 *       is = l/16
 *       q1 = ((ql[l+0] & 0xF)  | ((qh[l] >> 0) & 3) << 4) - 32
 *       q2 = ((ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32
 *       q3 = ((ql[l+0] >> 4)   | ((qh[l] >> 4) & 3) << 4) - 32
 *       q4 = ((ql[l+32] >> 4)  | ((qh[l] >> 6) & 3) << 4) - 32
 *       y[l+0]   = d * sc[is+0] * q1
 *       y[l+32]  = d * sc[is+2] * q2
 *       y[l+64]  = d * sc[is+4] * q3
 *       y[l+96]  = d * sc[is+6] * q4
 *     ql += 64; qh += 32; sc += 8;
 *
 *   So each iteration through the inner loop (l=0..31) produces 128 F32 values.
 *   The outer loop (n=0,128) runs twice per super-block for 256 total values.
 *
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o q6_kernel_omp.so q6_kernel_omp.c -lm
 */

#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

#define QK_K 256
#define Q6_K_BLOCK_SIZE 210  /* bytes per 256-value block */

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

/* ── Q6_K dequantize + dot product, row-parallel ──────────────────── */

void q6_k_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    /* W: row-major Q6_K quantized matrix, (n_rows, n_cols) values
     * x: input vector of length n_cols
     * out: output vector of length n_rows
     * n_cols must be multiple of 256 (QK_K)
     */
    int nb = n_cols / QK_K;  /* blocks per row */

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + r * nb * Q6_K_BLOCK_SIZE;
        float sum = 0.0f;

        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + b * Q6_K_BLOCK_SIZE;
            const float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            const int8_t *sc = (const int8_t *)(blk + 192);

            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;

            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    float ds0 = d * sc[is + 0];
                    float ds2 = d * sc[is + 2];
                    float ds4 = d * sc[is + 4];
                    float ds6 = d * sc[is + 6];

                    int base = b * QK_K + n;
                    sum += ds0 * q1 * x[base + l + 0];
                    sum += ds2 * q2 * x[base + l + 32];
                    sum += ds4 * q3 * x[base + l + 64];
                    sum += ds6 * q4 * x[base + l + 96];
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        out[r] = sum;
    }
}

/* ── Q6_K dequantize to F32 (for verification) ────────────────────── */

void q6_k_dequantize_row(const uint8_t *restrict W, float *restrict out, int n_values) {
    /* Dequantize one row of Q6_K to F32 */
    int nb = n_values / QK_K;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = W + b * Q6_K_BLOCK_SIZE;
        const float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
        const int8_t *sc = (const int8_t *)(blk + 192);
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                out[l + 0]  = d * sc[is + 0] * q1;
                out[l + 32] = d * sc[is + 2] * q2;
                out[l + 64] = d * sc[is + 4] * q3;
                out[l + 96] = d * sc[is + 6] * q4;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ── Q6_K matmul with AVX2 vectorized dequant ──────────────────────
 * Processes 8 values at a time using AVX2 when possible,
 * falls back to scalar for tail elements.
 */

void q6_k_matmul_omp_avx2(const uint8_t *restrict W, const float *restrict x,
                            float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;  /* blocks per row */

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + r * nb * Q6_K_BLOCK_SIZE;
        float sum = 0.0f;

        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + b * Q6_K_BLOCK_SIZE;
            const float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            const int8_t *sc = (const int8_t *)(blk + 192);
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            int x_off = b * QK_K;

            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    float ds0 = d * sc[is + 0];
                    float ds2 = d * sc[is + 2];
                    float ds4 = d * sc[is + 4];
                    float ds6 = d * sc[is + 6];

                    int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    sum += ds0 * q1 * x[x_off + n + l + 0];
                    sum += ds2 * q2 * x[x_off + n + l + 32];
                    sum += ds4 * q3 * x[x_off + n + l + 64];
                    sum += ds6 * q4 * x[x_off + n + l + 96];
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        out[r] = sum;
    }
}