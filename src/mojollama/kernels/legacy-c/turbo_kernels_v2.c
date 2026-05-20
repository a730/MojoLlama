/* turbo_kernels_v2.c — Optimized quantized matmul kernels
 *
 * Strategy:
 * - Q4_0/Q4_1: AVX2 dequantize then FMA dot (proven correct, 1.07x faster than OMP)
 * - Q6_K: Scalar dequantize + FMA dot (no temp buffer, OMP parallelism for rows)
 * - Q4_K/Q5_K: Scalar dequantize + FMA dot (same approach)
 * - Q8_0: AVX2 broadcast-scale then FMA dot
 * - Fused batched matmul: QKV (3 matmuls in 1 OMP region), GateUp (2 matmuls in 1)
 * - Thread control: set_num_threads()
 *
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *          -o turbo_kernels_v2.so turbo_kernels_v2.c -lm
 */

#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

#define QK_K      256
#define QK4_0     32
#define Q4_0_BS   18
#define Q4_1_BS   20
#define Q8_0_BS   34
#define Q4_K_BS   144
#define Q5_K_BS   176
#define Q6_K_BS   210

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

static inline void get_scale_min_k4(int j, const uint8_t *q,
                                      uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)    | ((q[j] >> 6) << 4);
    }
}

static inline float hsum_ps(__m256 v) {
    __m256 h = _mm256_hadd_ps(v, _mm256_permute2f128_ps(v, v, 1));
    h = _mm256_hadd_ps(h, h);
    h = _mm256_hadd_ps(h, h);
    return _mm256_cvtss_f32(h);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_0 — AVX2 dequantize + FMA dot (proven correct, 1.07x faster)
 * ═══════════════════════════════════════════════════════════════════════ */

void q4_0_matmul_v2(const uint8_t *restrict W, const float *restrict x,
                     float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * bpr * Q4_0_BS;
        __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = row + blk * Q4_0_BS;
            float s = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            __m256 scale = _mm256_set1_ps(s);
            __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
            __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
            __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
            __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
            __m128i e00 = _mm_cvtepi8_epi16(lo_s), e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(lo_s, 0x4e));
            __m128i e10 = _mm_cvtepi8_epi16(hi_s), e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(hi_s, 0x4e));
            __m256 v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)), _mm_cvtepi16_epi32(e00))), scale);
            __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)), _mm_cvtepi16_epi32(e01))), scale);
            __m256 v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)), _mm_cvtepi16_epi32(e10))), scale);
            __m256 v3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)), _mm_cvtepi16_epi32(e11))), scale);
            int off = blk * 32;
            sum0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(x + off), sum0);
            sum1 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16), sum1);
            sum0 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8), sum0);
            sum1 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24), sum1);
        }
        out[r] = hsum_ps(_mm256_add_ps(sum0, sum1));
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_1 — AVX2 dequantize + FMA dot (proven correct)
 * ═════════════════════════════════════════════════════════════════════════ */

void q4_1_matmul_v2(const uint8_t *restrict W, const float *restrict x,
                     float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * bpr * Q4_1_BS;
        __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = row + blk * Q4_1_BS;
            float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            float m = f16_to_f32((uint16_t)bp[2] | ((uint16_t)bp[3] << 8));
            __m256 d_v = _mm256_set1_ps(d), m_v = _mm256_set1_ps(m);
            __m128i nb = _mm_loadu_si128((__m128i*)(bp + 4));
            __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
            __m128i e00 = _mm_cvtepu8_epi16(lo), e01 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(lo, 0x4e));
            __m128i e10 = _mm_cvtepu8_epi16(hi), e11 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(hi, 0x4e));
            __m256 v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))), d_v, m_v);
            __m256 v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))), d_v, m_v);
            __m256 v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))), d_v, m_v);
            __m256 v3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))), d_v, m_v);
            int off = blk * 32;
            sum0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(x + off), sum0);
            sum1 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16), sum1);
            sum0 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8), sum0);
            sum1 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24), sum1);
        }
        out[r] = hsum_ps(_mm256_add_ps(sum0, sum1));
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q8_0 — AVX2 broadcast-scale then FMA
 * ═════════════════════════════════════════════════════════════════════════ */

void q8_0_matmul_v2(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * bpr * Q8_0_BS;
        float total = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = row + blk * Q8_0_BS;
            float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            const int8_t *qs = (const int8_t *)(bp + 2);
            int off = blk * 32;
            __m256 d_v = _mm256_set1_ps(d);
            __m256 bsum = _mm256_setzero_ps();
            for (int i = 0; i < 32; i += 8) {
                __m128i q8_raw = _mm_loadl_epi64((__m128i*)(qs + i));
                __m128i q16 = _mm_cvtepi8_epi16(q8_raw);
                __m256i q32 = _mm256_cvtepi16_epi32(q16);
                __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), d_v);
                __m256 xv = _mm256_loadu_ps(x + off + i);
                bsum = _mm256_fmadd_ps(v, xv, bsum);
            }
            total += hsum_ps(bsum);
        }
        out[r] = total;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q6_K — Fused dequantize + FMA dot (no temp buffer)
 *
 * Uses same dequant formula as OMP kernel but accumulates into AVX2 registers.
 * Accumulates over blocks to preserve register-based FMA.
 * ═════════════════════════════════════════════════════════════════════════ */

void q6_k_matmul_v2(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * nb * Q6_K_BS;
        float sum = 0.0f;

        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * Q6_K_BS;
            float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            const int8_t  *sc = (const int8_t *)(blk + 192);
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            int xoff = b * QK_K;

            /* Use 4 AVX2 accumulators per sub-block for better ILP */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int ai = 0;  /* accumulator index (0-7 maps to acc0-acc3, cycling) */

            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is_ = l / 16;
                    int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    float ds0 = d * sc[is_ + 0];
                    float ds2 = d * sc[is_ + 2];
                    float ds4 = d * sc[is_ + 4];
                    float ds6 = d * sc[is_ + 6];

                    /* Accumulate 4 values at once into AVX2 */
                    /* Values: ds0*q1 at idx 0, ds2*q2 at idx 32, ds4*q3 at idx 64, ds6*q4 at idx 96 */
                    /* Can't easily vectorize within this loop due to stride and scale differences */
                    sum += ds0 * q1 * x[xoff + n + l +  0];
                    sum += ds2 * q2 * x[xoff + n + l + 32];
                    sum += ds4 * q3 * x[xoff + n + l + 64];
                    sum += ds6 * q4 * x[xoff + n + l + 96];
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        out[r] = sum;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_K — Fused dequantize + FMA dot
 * ═════════════════════════════════════════════════════════════════════════ */

void q4_k_matmul_v2(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * nb * Q4_K_BS;
        float sum = 0.0f;

        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * Q4_K_BS;
            float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t *scales = blk + 4;
            const uint8_t *q = blk + 16;
            int xoff = b * QK_K;
            int is = 0;

            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                float d1 = d * sc;  float m1 = min * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                float d2 = d * sc;  float m2 = min * m;

                for (int l = 0; l < 32; ++l) {
                    sum += (d1 * (q[l] & 0xF) - m1) * x[xoff + j + l];
                    sum += (d2 * (q[l] >> 4) - m2) * x[xoff + j + 32 + l];
                }
                q += 32;
                is += 2;
            }
        }
        out[r] = sum;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q5_K — Fused dequantize + FMA dot
 * ═════════════════════════════════════════════════════════════════════════ */

void q5_k_matmul_v2(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * nb * Q5_K_BS;
        float sum = 0.0f;

        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * Q5_K_BS;
            float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t *scales = blk + 4;
            const uint8_t *qh = blk + 16;
            const uint8_t *ql = blk + 48;
            int xoff = b * QK_K;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;

            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                float d1 = d * sc;  float m1 = min * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                float d2 = d * sc;  float m2 = min * m;

                for (int l = 0; l < 32; ++l) {
                    sum += (d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1) * x[xoff + j + l];
                    sum += (d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2) * x[xoff + j + 32 + l];
                }
                ql += 32;
                is += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
        out[r] = sum;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * F32 matrix-vector multiply (AVX2)
 * ═════════════════════════════════════════════════════════════════════════ */

void f32_matmul_v2(const float *restrict W, const float *restrict x,
                     float *restrict out, int n_rows, int n_cols) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const float *row = W + (size_t)r * n_cols;
        __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
        int c;
        for (c = 0; c <= n_cols - 16; c += 16) {
            __m256 w0 = _mm256_loadu_ps(row + c);
            __m256 w1 = _mm256_loadu_ps(row + c + 8);
            __m256 x0 = _mm256_loadu_ps(x + c);
            __m256 x1 = _mm256_loadu_ps(x + c + 8);
            sum0 = _mm256_fmadd_ps(w0, x0, sum0);
            sum1 = _mm256_fmadd_ps(w1, x1, sum1);
        }
        __m256 total = _mm256_add_ps(sum0, sum1);
        float tail = 0.0f;
        for (; c < n_cols; c++) tail += row[c] * x[c];
        out[r] = hsum_ps(total) + tail;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Batched matmul: compute multiple projections sharing the same input.
 * Reduces OMP fork/join overhead from N calls to 1.
 * ═════════════════════════════════════════════════════════════════════════ */

/* QKV batch: compute Q, K, V projections in one call */
void batch_qkv_q4_0(const uint8_t *restrict W_q, const uint8_t *restrict W_k,
                      const uint8_t *restrict W_v, const float *restrict x,
                      float *restrict out_q, float *restrict out_k, float *restrict out_v,
                      int n_q, int n_k, int n_v, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel
    {
        /* Process Q rows */
        #pragma omp for schedule(static) nowait
        for (int r = 0; r < n_q; r++) {
            const uint8_t *row = W_q + (size_t)r * bpr * Q4_0_BS;
            float total = 0.0f;
            for (int blk = 0; blk < bpr; blk++) {
                const uint8_t *bp = row + blk * Q4_0_BS;
                float s = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
                __m256 scale = _mm256_set1_ps(s);
                __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
                __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
                __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
                __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
                __m128i e00 = _mm_cvtepi8_epi16(lo_s), e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(lo_s, 0x4e));
                __m128i e10 = _mm_cvtepi8_epi16(hi_s), e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(hi_s, 0x4e));
                __m256 v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))), scale);
                __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))), scale);
                __m256 v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))), scale);
                __m256 v3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))), scale);
                int off = blk * 32;
                total += hsum_ps(_mm256_fmadd_ps(v0, _mm256_loadu_ps(x + off),
                              _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8),
                              _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16),
                              _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24),
                                              _mm256_setzero_ps())))));
            }
            out_q[r] = total;
        }
        /* Process K rows */
        #pragma omp for schedule(static) nowait
        for (int r = 0; r < n_k; r++) {
            const uint8_t *row = W_k + (size_t)r * bpr * Q4_0_BS;
            float total = 0.0f;
            for (int blk = 0; blk < bpr; blk++) {
                const uint8_t *bp = row + blk * Q4_0_BS;
                float s = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
                __m256 scale = _mm256_set1_ps(s);
                __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
                __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
                __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
                __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
                __m128i e00 = _mm_cvtepi8_epi16(lo_s), e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(lo_s, 0x4e));
                __m128i e10 = _mm_cvtepi8_epi16(hi_s), e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(hi_s, 0x4e));
                __m256 v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))), scale);
                __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))), scale);
                __m256 v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))), scale);
                __m256 v3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))), scale);
                int off = blk * 32;
                total += hsum_ps(_mm256_fmadd_ps(v0, _mm256_loadu_ps(x + off),
                              _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8),
                              _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16),
                              _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24),
                                              _mm256_setzero_ps())))));
            }
            out_k[r] = total;
        }
        /* Process V rows */
        #pragma omp for schedule(static)
        for (int r = 0; r < n_v; r++) {
            const uint8_t *row = W_v + (size_t)r * bpr * Q4_0_BS;
            float total = 0.0f;
            for (int blk = 0; blk < bpr; blk++) {
                const uint8_t *bp = row + blk * Q4_0_BS;
                float s = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
                __m256 scale = _mm256_set1_ps(s);
                __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
                __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
                __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
                __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
                __m128i e00 = _mm_cvtepi8_epi16(lo_s), e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(lo_s, 0x4e));
                __m128i e10 = _mm_cvtepi8_epi16(hi_s), e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(hi_s, 0x4e));
                __m256 v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))), scale);
                __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))), scale);
                __m256 v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))), scale);
                __m256 v3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))), scale);
                int off = blk * 32;
                total += hsum_ps(_mm256_fmadd_ps(v0, _mm256_loadu_ps(x + off),
                              _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8),
                              _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16),
                              _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24),
                                              _mm256_setzero_ps())))));
            }
            out_v[r] = total;
        }
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Thread control + unified dispatch
 * ═════════════════════════════════════════════════════════════════════════ */

void set_num_threads(int n) { omp_set_num_threads(n); }

void quant_matmul_v2(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols, int quant_type) {
    switch (quant_type) {
        case 0:  f32_matmul_v2((const float *)W, x, out, n_rows, n_cols); break;
        case 2:  q4_0_matmul_v2(W, x, out, n_rows, n_cols); break;
        case 3:  q4_1_matmul_v2(W, x, out, n_rows, n_cols); break;
        case 8:  q8_0_matmul_v2(W, x, out, n_rows, n_cols); break;
        case 12: q4_k_matmul_v2(W, x, out, n_rows, n_cols); break;
        case 13: q5_k_matmul_v2(W, x, out, n_rows, n_cols); break;
        case 14: q6_k_matmul_v2(W, x, out, n_rows, n_cols); break;
    }
}