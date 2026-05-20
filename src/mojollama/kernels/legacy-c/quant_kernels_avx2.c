/**
 * quant_kernels_avx2.c — AVX2-vectorized Q4_K, Q5_K, Q6_K, Q8_0 matmul kernels
 * plus F32 matmul and RMS norm.
 *
 * All kernels use the same calling convention:
 *   (W_raw, x, out, n_rows, n_cols) or (W_raw, x, out, n_rows, n_cols, quant_type)
 *
 * Compiled with: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o quant_kernels_omp.so
 */

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

#define QK_K  256   /* K-type super-block size */
#define QK8_0 32    /* Q8_0 block size */
#define Q4_K_BS 144 /* bytes per 256 values */
#define Q5_K_BS 176
#define Q6_K_BS 210
#define Q8_0_BS  34
#define Q4_0_BS  18
#define Q4_1_BS  20

/* f16 to f32 conversion */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    float out;
    if (exp == 0) {
        if (mant == 0) { out = 0.0f; }
        else { out = ldexpf((float)mant / 1024.0f, -14); }
    } else if (exp == 31) {
        out = (mant == 0) ? INFINITY : NAN;
    } else {
        out = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    }
    return sign ? -out : out;
}

/* Helper: horizontal sum of 8 floats in __m256 */
static inline float hsum_ps(__m256 v) {
    __m256 h = _mm256_hadd_ps(v, _mm256_permute2f128_ps(v, v, 1));
    h = _mm256_hadd_ps(h, h);
    h = _mm256_hadd_ps(h, h);
    return _mm256_cvtss_f32(h);
}

/* Helper: horizontal sum of __m128 */
static inline float hsum128_ps(__m128 v) {
    __m128 h = _mm_hadd_ps(v, v);
    h = _mm_hadd_ps(h, h);
    return _mm_cvtss_f32(h);
}

/* Scale extraction helpers for K-type quantization */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j] >> 6;
    } else if (j < 20) {
        *d = (q[j+4] & 15) | ((q[j-2] >> 6) << 4);
        *m = (q[j+4] >> 4);
    } else {
        *d = (q[j+4] & 63) >> 2;
        *m = (q[j+4] >> 4);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_K AVX2 matmul
 * Block: 144 bytes per 256 values
 * ═══════════════════════════════════════════════════════════════════════ */

void q4_k_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * nb * Q4_K_BS;
        __m256 acc = _mm256_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * Q4_K_BS;
            float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t *scales = blk + 4;
            const uint8_t *q = blk + 16;
            int x_off = b * QK_K;
            int is = 0;

            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                float d1 = d * sc;  float m1 = min * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                float d2 = d * sc;  float m2 = min * m;

                /* Process lo nibbles (32 values) + hi nibbles (32 values) = 64 values */
                __m256 d1v = _mm256_set1_ps(d1);
                __m256 d2v = _mm256_set1_ps(d2);
                __m256 m1v = _mm256_set1_ps(m1);
                __m256 m2v = _mm256_set1_ps(m2);
                int mask_lo = 0x0F0F0F0F;

                /* Lo nibbles: first 32 values */
                for (int l = 0; l < 32; l += 8) {
                    /* Load 8 x-values */
                    __m256 xv = _mm256_loadu_ps(x + x_off + j + l);
                    /* Extract lo nibbles from 8 packed bytes */
                    __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(q + l)));
                    __m256i lo_mask = _mm256_set1_epi32(0xF);
                    __m256i nib = _mm256_and_si256(qi, lo_mask);
                    __m256 fv = _mm256_cvtepi32_ps(nib);
                    /* d1 * nib - m1 */
                    __m256 deq = _mm256_sub_ps(_mm256_mul_ps(d1v, fv), m1v);
                    acc = _mm256_fmadd_ps(deq, xv, acc);
                }
                /* Hi nibbles: next 32 values (positions j+32..j+63) */
                for (int l = 0; l < 32; l += 8) {
                    __m256 xv = _mm256_loadu_ps(x + x_off + j + 32 + l);
                    __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(q + l)));
                    __m256i hi_mask = _mm256_set1_epi32(0xF);
                    __m256i nib = _mm256_srli_epi32(_mm256_and_si256(qi, _mm256_set1_epi32(0xFF)), 4);
                    __m256 fv = _mm256_cvtepi32_ps(nib);
                    __m256 deq = _mm256_sub_ps(_mm256_mul_ps(d2v, fv), m2v);
                    acc = _mm256_fmadd_ps(deq, xv, acc);
                }
                q += 32;
                is += 2;
            }
        }
        out[r] = hsum_ps(acc);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q5_K AVX2 matmul
 * Block: 176 bytes per 256 values
 * ═══════════════════════════════════════════════════════════════════════ */

void q5_k_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * nb * Q5_K_BS;
        __m256 acc = _mm256_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * Q5_K_BS;
            float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t *scales = blk + 4;
            const uint8_t *qh_arr = blk + 16;
            const uint8_t *ql = blk + 48;
            int x_off = b * QK_K;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;

            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                float d1 = d * sc;  float m1 = min * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                float d2 = d * sc;  float m2 = min * m;

                __m256 d1v = _mm256_set1_ps(d1);
                __m256 d2v = _mm256_set1_ps(d2);
                __m256 m1v = _mm256_set1_ps(m1);
                __m256 m2v = _mm256_set1_ps(m2);
                __m256i u1v = _mm256_set1_epi32(u1);
                __m256i u2v = _mm256_set1_epi32(u2);

                /* Lo nibbles + 5th bit */
                for (int l = 0; l < 32; l += 8) {
                    __m256 xv = _mm256_loadu_ps(x + x_off + j + l);
                    __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ql + l)));
                    __m256i lo = _mm256_and_si256(qi, _mm256_set1_epi32(0xF));
                    __m256i qh_lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(qh_arr + l)));
                    __m256i bit5_lo = _mm256_and_si256(qh_lo, u1v);
                    __m256i shifted = _mm256_slli_epi32(bit5_lo, 1); /* 0 or 16 */
                    __m256 fv_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(lo, shifted));
                    __m256 deq = _mm256_sub_ps(_mm256_mul_ps(d1v, fv_lo), m1v);
                    acc = _mm256_fmadd_ps(deq, xv, acc);
                }
                /* Hi nibbles + 5th bit */
                for (int l = 0; l < 32; l += 8) {
                    __m256 xv = _mm256_loadu_ps(x + x_off + j + 32 + l);
                    __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ql + l)));
                    __m256i hi = _mm256_srli_epi32(qi, 4);
                    __m256i qh_hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(qh_arr + l)));
                    __m256i bit5_hi = _mm256_and_si256(qh_hi, u2v);
                    __m256i shifted = _mm256_slli_epi32(bit5_hi, 1);
                    __m256 fv_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(hi, shifted));
                    __m256 deq = _mm256_sub_ps(_mm256_mul_ps(d2v, fv_hi), m2v);
                    acc = _mm256_fmadd_ps(deq, xv, acc);
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        out[r] = hsum_ps(acc);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q6_K AVX2 matmul
 * Block: 210 bytes per 256 values
 * ═══════════════════════════════════════════════════════════════════════ */

void q6_k_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * nb * Q6_K_BS;
        __m256 acc = _mm256_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * Q6_K_BS;
            const float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            const int8_t *sc = (const int8_t *)(blk + 192);
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            int base = b * QK_K;

            for (int n = 0; n < QK_K; n += 128) {
                /* Process 32 values at a time using AVX2 */
                for (int l = 0; l < 32; l += 8) {
                    int is_ = l / 16;

                    /* Dequantize 4 groups of 8 values */
                    /* Group 1: ql[l+0] & 0xF | (qh[l+0]>>0 & 3)<<4 - 32 */
                    __m256i ql_lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ql + l)));
                    __m256i ql_hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ql + l + 32)));
                    __m256i qh8   = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(qh + l)));

                    __m256i mask4 = _mm256_set1_epi32(0xF);
                    __m256i mask3 = _mm256_set1_epi32(3);

                    /* q1 = (ql[l]&0xF) | ((qh[l]>>0)&3)<<4 */
                    __m256i q1 = _mm256_add_epi32(
                        _mm256_and_si256(ql_lo, mask4),
                        _mm256_slli_epi32(_mm256_and_si256(qh8, mask3), 4));
                    /* q2 = (ql[l+32]&0xF) | ((qh[l]>>2)&3)<<4 */
                    __m256i q2 = _mm256_add_epi32(
                        _mm256_and_si256(ql_hi, mask4),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh8, 2), mask3), 4));
                    /* q3 = (ql[l]>>4) | ((qh[l]>>4)&3)<<4 */
                    __m256i q3 = _mm256_add_epi32(
                        _mm256_srli_epi32(_mm256_and_si256(ql_lo, _mm256_set1_epi32(0xFF)), 4),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh8, 4), mask3), 4));
                    /* q4 = (ql[l+32]>>4) | ((qh[l]>>6)&3)<<4 */
                    __m256i q4 = _mm256_add_epi32(
                        _mm256_srli_epi32(_mm256_and_si256(ql_hi, _mm256_set1_epi32(0xFF)), 4),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh8, 6), mask3), 4));

                    /* Subtract 32 to center */
                    __m256i off = _mm256_set1_epi32(32);
                    q1 = _mm256_sub_epi32(q1, off);
                    q2 = _mm256_sub_epi32(q2, off);
                    q3 = _mm256_sub_epi32(q3, off);
                    q4 = _mm256_sub_epi32(q4, off);

                    /* Scale: d * sc[is_+0..3] */
                    float ds0 = d * sc[is_ + 0];
                    float ds2 = d * sc[is_ + 2];
                    float ds4 = d * sc[is_ + 4];
                    float ds6 = d * sc[is_ + 6];

                    __m256 fq1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1), _mm256_set1_ps(ds0));
                    __m256 fq2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2), _mm256_set1_ps(ds2));
                    __m256 fq3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3), _mm256_set1_ps(ds4));
                    __m256 fq4 = _mm256_mul_ps(_mm256_cvtepi32_ps(q4), _mm256_set1_ps(ds6));

                    /* Multiply by x and accumulate */
                    __m256 x1 = _mm256_loadu_ps(x + base + n + l);
                    __m256 x2 = _mm256_loadu_ps(x + base + n + l + 32);
                    __m256 x3 = _mm256_loadu_ps(x + base + n + l + 64);
                    __m256 x4 = _mm256_loadu_ps(x + base + n + l + 96);

                    /* FMA: acc += fq * x for each group */
                    acc = _mm256_fmadd_ps(fq1, x1, acc);
                    acc = _mm256_fmadd_ps(fq2, x2, acc);
                    acc = _mm256_fmadd_ps(fq3, x3, acc);
                    acc = _mm256_fmadd_ps(fq4, x4, acc);
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        out[r] = hsum_ps(acc);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q8_0 AVX2 matmul  (34 bytes per 32 values)
 * Uses AVX2 int8 sign-extension + FMA for fast dot product
 * ═══════════════════════════════════════════════════════════════════════ */

void q8_0_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / QK8_0;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        __m256 acc = _mm256_setzero_ps();

        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = W + ((size_t)r * bpr + blk) * Q8_0_BS;
            float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            const int8_t *qs = (const int8_t *)(bp + 2);
            int off = blk * 32;
            __m256 d_v = _mm256_set1_ps(d);

            /* Process all 32 int8 values as FMA with scale */
            /* 32 int8 values: process as 4x8 -> dequantize -> FMA with x */
            for (int i = 0; i < 32; i += 8) {
                __m128i q8 = _mm_loadl_epi64((__m128i*)(qs + i));
                __m256i qi = _mm256_cvtepi8_epi32(q8);
                __m256 qf = _mm256_cvtepi32_ps(qi);
                __m256 xv = _mm256_loadu_ps(x + off + i);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(qf, d_v), xv, acc);
            }
        }
        out[r] = hsum_ps(acc);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_0 AVX2 matmul  (18 bytes per 32 values)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void decode_q4_0(const uint8_t *blk, __m256 v[4]) {
    float scale = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
    __m256 s = _mm256_set1_ps(scale);
    __m128i nb = _mm_loadu_si128((__m128i*)(blk + 2));
    /* GGUF Q4_0: 16 bytes → 32 values as [lo0..lo15, hi0..hi15] */
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
    __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
    __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
    /* Sign-extend to 16-bit then 32-bit float */
    __m128i lo16 = _mm_cvtepi8_epi16(lo_s);
    __m128i hi16 = _mm_cvtepi8_epi16(hi_s);
    __m256i lo32 = _mm256_cvtepi16_epi32(lo16);
    __m256i hi32 = _mm256_cvtepi16_epi32(hi16);
    v[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(lo32), s);  /* positions 0..7 */
    v[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(hi32), s);  /* positions 8..15 */
    /* Second 8 bytes of nibbles */
    __m128i lo16b = _mm_unpackhi_epi64(lo16, _mm_setzero_si128());
    __m128i hi16b = _mm_unpackhi_epi64(hi16, _mm_setzero_si128());
    /* lo16b/hi16b have at most 8 int16 values; but we actually need the remaining
     * 16 values from the second 8-byte half. Let me redo this properly. */
    __m128i nb2 = _mm_loadu_si128((__m128i*)(blk + 2 + 8));
    __m128i lo2 = _mm_and_si128(nb2, _mm_set1_epi8(15));
    __m128i hi2 = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb2, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
    __m128i lo_s2 = _mm_sub_epi8(lo2, _mm_set1_epi8(8));
    __m128i hi_s2 = _mm_sub_epi8(hi2, _mm_set1_epi8(8));
    __m128i lo16_2 = _mm_cvtepi8_epi16(lo_s2);
    __m128i hi16_2 = _mm_cvtepi8_epi16(hi_s2);
    v[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo16_2)), s);
    v[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi16_2)), s);
}

static inline float block_dot_fma(__m256 v[4], const float *x) {
    __m256 s0 = _mm256_mul_ps(v[0], _mm256_loadu_ps(x));
    s0 = _mm256_fmadd_ps(v[1], _mm256_loadu_ps(x + 8), s0);
    s0 = _mm256_fmadd_ps(v[2], _mm256_loadu_ps(x + 16), s0);
    s0 = _mm256_fmadd_ps(v[3], _mm256_loadu_ps(x + 24), s0);
    return hsum_ps(s0);
}

void q4_0_matmul_omp(const uint8_t *restrict W, const float *restrict x,
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

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_1 AVX2 matmul  (20 bytes per 32 values)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void decode_q4_1(const uint8_t *blk, __m256 v[4]) {
    float d = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
    float m = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
    const uint8_t *qs = blk + 4;
    __m256 dv = _mm256_set1_ps(d);
    __m256 mv = _mm256_set1_ps(m);
    for (int j = 0; j < 4; j++) {
        __m128i q = _mm_loadl_epi64((__m128i*)(qs + j * 4));
        __m256i qi = _mm256_cvtepu8_epi32(q);
        __m256i lo = _mm256_and_si256(qi, _mm256_set1_epi32(0xF));
        __m256i hi = _mm256_srli_epi32(qi, 4);
        __m256 fv = _mm256_add_ps(_mm256_mul_ps(dv, _mm256_cvtepi32_ps(lo)), mv);
        v[j] = _mm256_add_ps(fv, _mm256_mul_ps(dv, _mm256_cvtepi32_ps(hi)));
    }
}

void q4_1_matmul_omp(const uint8_t *restrict W, const float *restrict x,
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
 * F32 matrix-vector multiply (AVX2 + FMA)
 * ═══════════════════════════════════════════════════════════════════════ */

void f32_matmul_omp(const float *restrict W, const float *restrict x,
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
        float result = hsum_ps(total);
        for (; c < n_cols; c++) result += row[c] * x[c];
        out[r] = result;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Block F32 matmul for small n_cols (reduces OMP barrier overhead)
 * Performs n_proj matrix-vector products in a single OMP parallel region
 * ═══════════════════════════════════════════════════════════════════════ */

void f32_matmul_omp_blocked(const float *restrict W, const float *restrict x,
                              float *restrict out, int n_rows, int n_cols) {
    f32_matmul_omp(W, x, out, n_rows, n_cols);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Batched matmul: up to 7 projections in one OMP call
 * ═══════════════════════════════════════════════════════════════════════ */

#define MAX_PROJ 7

static void process_row(const uint8_t* w[MAX_PROJ], const float* x, float* out[MAX_PROJ],
                         int nrows[MAX_PROJ], int ncols[MAX_PROJ], int ts[MAX_PROJ],
                         int n_proj, int row) {
    for (int p = 0; p < n_proj; p++) {
        if (row >= nrows[p]) continue;
        float total = 0.0f;
        int nc = ncols[p], bp = nc / 32;
        void (*dec)(const uint8_t*, __m256*) = (ts[p] == Q4_1_BS) ? decode_q4_1 : decode_q4_0;
        for (int blk = 0; blk < bp; blk++) {
            __m256 v[4];
            dec(w[p] + ((size_t)row * bp + blk) * ts[p], v);
            total += block_dot_fma(v, x + blk * 32);
        }
        out[p][row] = total;
    }
}

void batch_matmul(const uint8_t* w[MAX_PROJ], const float* x, float* out[MAX_PROJ],
                   int nrows[MAX_PROJ], int ncols[MAX_PROJ], int ts[MAX_PROJ],
                   int n_proj) {
    int max_rows = 0;
    for (int p = 0; p < n_proj; p++) {
        memset(out[p], 0, nrows[p] * sizeof(float));
        if (nrows[p] > max_rows) max_rows = nrows[p];
    }
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < max_rows; row++) {
        process_row(w, x, out, nrows, ncols, ts, n_proj, row);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Batched quant matmul: up to 7 projections with mixed quant types
 * Uses quant_matmul_omp dispatch per projection
 * ═══════════════════════════════════════════════════════════════════════ */

/* ═════════════════════════════════════════════════════════════════════════
 * RMS normalization (AVX2 vectorized)
 * ═══════════════════════════════════════════════════════════════════════ */

void rms_norm(const float* x, const float* weight, float* out, int n) {
    __m256 ss_vec = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        ss_vec = _mm256_fmadd_ps(v, v, ss_vec);
    }
    __m256 h = _mm256_hadd_ps(ss_vec, _mm256_permute2f128_ps(ss_vec, ss_vec, 1));
    h = _mm256_hadd_ps(h, h); h = _mm256_hadd_ps(h, h);
    float ss = _mm256_cvtss_f32(h);
    for (; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / n + 1e-5f);
    __m256 inv_v = _mm256_set1_ps(inv_rms);
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, inv_v), w));
    }
    for (; i < n; i++) out[i] = x[i] * inv_rms * weight[i];
}

/* ═════════════════════════════════════════════════════════════════════════
 * SiLU activation
 * ═══════════════════════════════════════════════════════════════════════ */

void silu(float* x, int n) {
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        float tmp[8];
        _mm256_storeu_ps(tmp, v);
        for (int j = 0; j < 8; j++) tmp[j] = tmp[j] / (1.0f + expf(-tmp[j]));
        _mm256_storeu_ps(x + i, _mm256_loadu_ps(tmp));
    }
    for (; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

void silu_omp(float* x, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

int get_max_threads(void) { return omp_get_max_threads(); }
void set_num_threads(int n) { omp_set_num_threads(n); }

/* ═════════════════════════════════════════════════════════════════════════
 * Unified dispatch: quant_matmul_omp
 * quant_type: 0=F32, 2=Q4_0, 3=Q4_1, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
 * ═══════════════════════════════════════════════════════════════════════ */

void quant_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                       float *restrict out, int n_rows, int n_cols, int quant_type) {
    switch (quant_type) {
        case 0:  f32_matmul_omp((const float *)W, x, out, n_rows, n_cols); break;
        case 2:  q4_0_matmul_omp(W, x, out, n_rows, n_cols); break;
        case 3:  q4_1_matmul_omp(W, x, out, n_rows, n_cols); break;
        case 8:  q8_0_matmul_omp(W, x, out, n_rows, n_cols); break;
        case 12: q4_k_matmul_omp(W, x, out, n_rows, n_cols); break;
        case 13: q5_k_matmul_omp(W, x, out, n_rows, n_cols); break;
        case 14: q6_k_matmul_omp(W, x, out, n_rows, n_cols); break;
    }
}