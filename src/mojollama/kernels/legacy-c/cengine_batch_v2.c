/* cengine_batch.c — Unified batched inference engine with K-quant + MoE support.
 * Supports: Q4_0, Q8_0, Q4_K, Q6_K quant formats, dense and MoE architectures.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#define Q4_0_BS 18
#define Q8_0_BS 34
#define BLOCK_SIZE 64
#define MAX_BLOCKS 1024
#define QK_K 256

/* K-quant block structures (from llama.cpp) */
typedef struct { uint16_t d; uint16_t dm; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_K;

static inline float gf16(uint16_t h) { return _cvtsh_ss(h); }
static inline float hsum_ps(__m256 v) {
    __m128 l=_mm256_castps256_ps128(v),h=_mm256_extractf128_ps(v,1);
    l=_mm_add_ps(l,h);l=_mm_hadd_ps(l,l);l=_mm_hadd_ps(l,l);return _mm_cvtss_f32(l);
}

/* Q4_K scale extraction (matches llama.cpp get_scale_min_k4) */
static inline void k4_scale(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

/* ── AVX2 Batch Matmuls ──────────────────────── */
void q4_0_batch_matmul(const uint8_t *W, const float *x, float *out, int n_rows, int nc, int B);
void q8_0_batch_matmul(const uint8_t *W, const float *x, float *out, int n_rows, int nc, int B);
void q4_k_batch_matmul(const uint8_t *W, const float *x, float *out, int n_rows, int nc, int B);
void q6_k_batch_matmul(const uint8_t *W, const float *x, float *out, int n_rows, int nc, int B);

/* Dispatch batch matmul by quant type */
/* Simple FP32 batch matmul for dequantized weights */
void f32_batch_matmul(const float *W, const float *x, float *out,
                      int n_rows, int nc, int B) {
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        for (int b = 0; b < B; b++) {
            float dot = 0;
            for (int i = 0; i < nc; i++)
                dot += W[(size_t)r * nc + i] * x[(size_t)b * nc + i];
            out[(size_t)b * n_rows + r] = dot;
        }
    }
}

static inline void batch_matmul(int qt, const uint8_t *W, const float *x, float *out,
                                int n_rows, int nc, int B) {
    if (qt == 0) f32_batch_matmul((const float*)W, x, out, n_rows, nc, B);
    else if (qt == 14) q6_k_batch_matmul(W, x, out, n_rows, nc, B);
    else if (qt == 12) q4_k_batch_matmul(W, x, out, n_rows, nc, B);
    else if (qt == 8) q8_0_batch_matmul(W, x, out, n_rows, nc, B);
    else q4_0_batch_matmul(W, x, out, n_rows, nc, B);
}
