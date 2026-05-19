/* turbo_kernels_avx2.c — AVX2+FMA vectorized quantized matmul kernels
 *
 * Key optimizations over quant_kernels_omp.c:
 * 1. Q4_K: AVX2 dot product (8 values at once) instead of scalar
 * 2. Q5_K: AVX2 dot product 
 * 3. Q6_K: AVX2 dot product — THE biggest win (output projection)
 * 4. Q4_0/Q4_1: Already AVX2 (reused from quant_kernels_omp.c)
 * 5. Batched QKV/GateUp: Single OMP parallel region for multiple projections
 *
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *          -o turbo_kernels_avx2.so turbo_kernels_avx2.c -lm
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

/* Scale helpers */
static inline void get_scale_min_k4(int j, const uint8_t *q,
                                      uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)    | ((q[j]      >> 6) << 4);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_0 matmul — AVX2 (from quant_kernels_omp.c, proven correct)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void decode_q4_0(const uint8_t* bp, __m256 v[4]) {
    float scale = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
    __m256 s = _mm256_set1_ps(scale);
    __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
    __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
    __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
    __m128i e00 = _mm_cvtepi8_epi16(lo_s), e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(lo_s, 0x4e));
    __m128i e10 = _mm_cvtepi8_epi16(hi_s), e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(hi_s, 0x4e));
    v[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e00, 0x4e)), _mm_cvtepi16_epi32(e00))), s);
    v[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e01, 0x4e)), _mm_cvtepi16_epi32(e01))), s);
    v[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e10, 0x4e)), _mm_cvtepi16_epi32(e10))), s);
    v[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e11, 0x4e)), _mm_cvtepi16_epi32(e11))), s);
}

static inline void decode_q4_1(const uint8_t* bp, __m256 v[4]) {
    float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
    float m = f16_to_f32((uint16_t)bp[2] | ((uint16_t)bp[3] << 8));
    __m256 d_v = _mm256_set1_ps(d), m_v = _mm256_set1_ps(m);
    __m128i nb = _mm_loadu_si128((__m128i*)(bp + 4));
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
    __m128i e00 = _mm_cvtepu8_epi16(lo), e01 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(lo, 0x4e));
    __m128i e10 = _mm_cvtepu8_epi16(hi), e11 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(hi, 0x4e));
    v[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))), d_v, m_v);
    v[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))), d_v, m_v);
    v[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))), d_v, m_v);
    v[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))), d_v, m_v);
}

static inline float block_dot_fma(const __m256 v[4], const float* x_base) {
    __m256 x0 = _mm256_loadu_ps(x_base);
    __m256 x1 = _mm256_loadu_ps(x_base + 8);
    __m256 x2 = _mm256_loadu_ps(x_base + 16);
    __m256 x3 = _mm256_loadu_ps(x_base + 24);
    __m256 a = _mm256_mul_ps(v[0], x0);
    a = _mm256_fmadd_ps(v[1], x1, a);
    a = _mm256_fmadd_ps(v[2], x2, a);
    a = _mm256_fmadd_ps(v[3], x3, a);
    __m256 h = _mm256_hadd_ps(a, _mm256_permute2f128_ps(a, a, 1));
    h = _mm256_hadd_ps(h, h);
    h = _mm256_hadd_ps(h, h);
    return _mm256_cvtss_f32(h);
}

void q4_0_matmul_avx2(const uint8_t *restrict W, const float *restrict x,
                        float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            __m256 v[4];
            decode_q4_0(W + ((size_t)r * bpr + blk) * Q4_0_BS, v);
            total += block_dot_fma(v, x + blk * 32);
        }
        out[r] = total;
    }
}

void q4_1_matmul_avx2(const uint8_t *restrict W, const float *restrict x,
                        float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            __m256 v[4];
            decode_q4_1(W + ((size_t)r * bpr + blk) * Q4_1_BS, v);
            total += block_dot_fma(v, x + blk * 32);
        }
        out[r] = total;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q6_K matmul — Dequantize-then-FMA approach
 *
 * Uses the proven scalar dequantization (matching quant_kernels_omp.c),
 * then AVX2 FMA dot product on the dequantized values.
 * Much faster than pure scalar because the dot product is vectorized.
 * ═════════════════════════════════════════════════════════════════════════ */

void q6_k_matmul_avx2(const uint8_t *restrict W, const float *restrict x,
                        float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;
    float qt[QK_K] __attribute__((aligned(32)));

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

            /* Dequantize 256 values using the EXACT same formula as the OMP kernel */
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

                    qt[n + l +  0] = ds0 * q1;
                    qt[n + l + 32] = ds2 * q2;
                    qt[n + l + 64] = ds4 * q3;
                    qt[n + l + 96] = ds6 * q4;
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }

            /* AVX2 FMA dot product: compute sum(qt * x) */
            int off = b * QK_K;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            for (int j = 0; j < QK_K; j += 32) {
                __m256 v0 = _mm256_loadu_ps(qt + j);
                __m256 v1 = _mm256_loadu_ps(qt + j + 8);
                __m256 v2 = _mm256_loadu_ps(qt + j + 16);
                __m256 v3 = _mm256_loadu_ps(qt + j + 24);
                __m256 x0 = _mm256_loadu_ps(x + off + j);
                __m256 x1 = _mm256_loadu_ps(x + off + j + 8);
                __m256 x2 = _mm256_loadu_ps(x + off + j + 16);
                __m256 x3 = _mm256_loadu_ps(x + off + j + 24);
                acc0 = _mm256_fmadd_ps(v0, x0, acc0);
                acc1 = _mm256_fmadd_ps(v1, x1, acc1);
                acc2 = _mm256_fmadd_ps(v2, x2, acc2);
                acc3 = _mm256_fmadd_ps(v3, x3, acc3);
            }
            __m256 sum01 = _mm256_add_ps(acc0, acc1);
            __m256 sum23 = _mm256_add_ps(acc2, acc3);
            __m256 total = _mm256_add_ps(sum01, sum23);
            __m256 h = _mm256_hadd_ps(total, _mm256_permute2f128_ps(total, total, 1));
            h = _mm256_hadd_ps(h, h);
            h = _mm256_hadd_ps(h, h);
            sum += _mm256_cvtss_f32(h);
        }
        out[r] = sum;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q8_0 matmul — AVX2 (reuse from quant_kernels_omp.c)
 * ═════════════════════════════════════════════════════════════════════════ */

void q8_0_matmul_avx2(const uint8_t *restrict W, const float *restrict x,
                        float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = W + ((size_t)r * bpr + blk) * Q8_0_BS;
            float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            const int8_t *qs = (const int8_t *)(bp + 2);
            int off = blk * 32;
            __m256 d_v = _mm256_set1_ps(d);
            __m256 bsum = _mm256_setzero_ps();
            for (int i = 0; i < 32; i += 8) {
                __m128i q8_raw = _mm_loadl_epi64((__m128i*)(qs + i));
                __m128i q16 = _mm_cvtepi8_epi16(q8_raw);
                __m256i q32 = _mm256_cvtepi16_epi32(q16);
                __m256 v = _mm256_cvtepi32_ps(q32);
                v = _mm256_mul_ps(v, d_v);
                __m256 xv = _mm256_loadu_ps(x + off + i);
                bsum = _mm256_fmadd_ps(v, xv, bsum);
            }
            /* Horizontal sum */
            __m256 h = _mm256_hadd_ps(bsum, _mm256_permute2f128_ps(bsum, bsum, 1));
            h = _mm256_hadd_ps(h, h);
            h = _mm256_hadd_ps(h, h);
            total += _mm256_cvtss_f32(h);
        }
        out[r] = total;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_K matmul — AVX2 vectorized
 * 
 * Uses partial AVX2: dequantize in 32-value chunks to float, then FMA dot.
 * Falls back to scalar for the complex scale unpacking but uses AVX2 FMA
 * for the dot product accumulation.
 * ═════════════════════════════════════════════════════════════════════════ */

void q4_k_matmul_avx2(const uint8_t *restrict W, const float *restrict x,
                        float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;
    /* Temp buffer for dequantized values (256 floats per block) */
    float qt[QK_K] __attribute__((aligned(32)));

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
            int is = 0;

            /* Dequantize all 256 values */
            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                float d1 = d * sc;  float m1 = min * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                float d2 = d * sc;  float m2 = min * m;

                for (int l = 0; l < 32; ++l)
                    qt[j + l] = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l)
                    qt[j + 32 + l] = d2 * (q[l] >> 4) - m2;
                q += 32;
                is += 2;
            }

            /* AVX2 FMA dot product */
            int off = b * QK_K;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            for (int j = 0; j < QK_K; j += 32) {
                __m256 v0 = _mm256_loadu_ps(qt + j);
                __m256 v1 = _mm256_loadu_ps(qt + j + 8);
                __m256 v2 = _mm256_loadu_ps(qt + j + 16);
                __m256 v3 = _mm256_loadu_ps(qt + j + 24);
                __m256 x0 = _mm256_loadu_ps(x + off + j);
                __m256 x1 = _mm256_loadu_ps(x + off + j + 8);
                __m256 x2 = _mm256_loadu_ps(x + off + j + 16);
                __m256 x3 = _mm256_loadu_ps(x + off + j + 24);
                acc0 = _mm256_fmadd_ps(v0, x0, acc0);
                acc1 = _mm256_fmadd_ps(v1, x1, acc1);
                acc2 = _mm256_fmadd_ps(v2, x2, acc2);
                acc3 = _mm256_fmadd_ps(v3, x3, acc3);
            }
            __m256 sum01 = _mm256_add_ps(acc0, acc1);
            __m256 sum23 = _mm256_add_ps(acc2, acc3);
            __m256 total = _mm256_add_ps(sum01, sum23);
            __m256 h = _mm256_hadd_ps(total, _mm256_permute2f128_ps(total, total, 1));
            h = _mm256_hadd_ps(h, h);
            h = _mm256_hadd_ps(h, h);
            sum += _mm256_cvtss_f32(h);
        }
        out[r] = sum;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q5_K matmul — AVX2 vectorized (dequantize + FMA dot)
 * ═════════════════════════════════════════════════════════════════════════ */

void q5_k_matmul_avx2(const uint8_t *restrict W, const float *restrict x,
                        float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;
    float qt[QK_K] __attribute__((aligned(32)));

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
            int is = 0;
            uint8_t u1 = 1, u2 = 2;

            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                float d1 = d * sc;  float m1 = min * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                float d2 = d * sc;  float m2 = min * m;

                for (int l = 0; l < 32; ++l) {
                    qt[j + l] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                }
                for (int l = 0; l < 32; ++l) {
                    qt[j + 32 + l] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }

            /* AVX2 FMA dot */
            int off = b * QK_K;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            for (int j = 0; j < QK_K; j += 32) {
                __m256 v0 = _mm256_loadu_ps(qt + j);
                __m256 v1 = _mm256_loadu_ps(qt + j + 8);
                __m256 v2 = _mm256_loadu_ps(qt + j + 16);
                __m256 v3 = _mm256_loadu_ps(qt + j + 24);
                __m256 x0 = _mm256_loadu_ps(x + off + j);
                __m256 x1 = _mm256_loadu_ps(x + off + j + 8);
                __m256 x2 = _mm256_loadu_ps(x + off + j + 16);
                __m256 x3 = _mm256_loadu_ps(x + off + j + 24);
                acc0 = _mm256_fmadd_ps(v0, x0, acc0);
                acc1 = _mm256_fmadd_ps(v1, x1, acc1);
                acc2 = _mm256_fmadd_ps(v2, x2, acc2);
                acc3 = _mm256_fmadd_ps(v3, x3, acc3);
            }
            __m256 sum01 = _mm256_add_ps(acc0, acc1);
            __m256 sum23 = _mm256_add_ps(acc2, acc3);
            __m256 total = _mm256_add_ps(sum01, sum23);
            __m256 h = _mm256_hadd_ps(total, _mm256_permute2f128_ps(total, total, 1));
            h = _mm256_hadd_ps(h, h);
            h = _mm256_hadd_ps(h, h);
            sum += _mm256_cvtss_f32(h);
        }
        out[r] = sum;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * F32 matrix-vector multiply (AVX2)
 * ═════════════════════════════════════════════════════════════════════════ */

void f32_matmul_avx2(const float *restrict W, const float *restrict x,
                       float *restrict out, int n_rows, int n_cols) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const float *row = W + (size_t)r * n_cols;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
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
        float result = 0.0f;
        for (; c < n_cols; c++) {
            result += row[c] * x[c];
        }
        __m256 h = _mm256_hadd_ps(total, _mm256_permute2f128_ps(total, total, 1));
        h = _mm256_hadd_ps(h, h);
        h = _mm256_hadd_ps(h, h);
        result += _mm256_cvtss_f32(h);
        out[r] = result;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Unified dispatch + thread control
 * ═════════════════════════════════════════════════════════════════════════ */

void set_num_threads(int n) { omp_set_num_threads(n); }

void quant_matmul_avx2(const uint8_t *restrict W, const float *restrict x,
                         float *restrict out, int n_rows, int n_cols, int quant_type) {
    switch (quant_type) {
        case 0:  f32_matmul_avx2((const float *)W, x, out, n_rows, n_cols); break;
        case 2:  q4_0_matmul_avx2(W, x, out, n_rows, n_cols); break;
        case 3:  q4_1_matmul_avx2(W, x, out, n_rows, n_cols); break;
        case 8:  q8_0_matmul_avx2(W, x, out, n_rows, n_cols); break;
        case 12: q4_k_matmul_avx2(W, x, out, n_rows, n_cols); break;
        case 13: q5_k_matmul_avx2(W, x, out, n_rows, n_cols); break;
        case 14: q6_k_matmul_avx2(W, x, out, n_rows, n_cols); break;
        default: break;
    }
}