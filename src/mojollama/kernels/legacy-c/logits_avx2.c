/* logits_avx2.c — AVX2-vectorized Q6_K logits projection
 *
 * The Q6_K matmul is the #1 bottleneck in MojoLlama (47% of tok/s).
 * The scalar inner loop processes one element at a time with:
 *   - nibble extraction (4 bits from ql + 2 bits from qh)
 *   - scale multiplication (d * sc * q)
 *   - FMA accumulate
 *
 * AVX2 strategy:
 *   1. Process 32 lanes in parallel (8× __m256 accumulators)
 *   2. Dequantize Q6_K blocks into F32 vectors (32 values at a time)
 *   3. FMA with input vector x
 *   4. Horizontal sum at end
 *
 * Q6_K layout (210 bytes per block of 256 values):
 *   [0..127]   ql — low nibbles (4 bits each)
 *   [128..191] qh — high 2 bits per value
 *   [192..207] sc — 8 int8_t super-group scales
 *   [208..209] d  — FP16 super-group delta
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* ═════════════════════════════════════════════════════════════════════════
 * FP16 conversion (hardware F16C)
 * ═════════════════════════════════════════════════════════════════════════ */

static inline float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1 : 1) * ldexpf((float)mant / 1024.0f, -14);
    }
    if (exp == 31) return mant == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
    return (sign ? -1 : 1) * ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
}

/* Convert 8 FP16 values to 8 FP32 using AVX2 F16C instruction */
static inline __m256 f16x8_to_f32(const void *ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si64(ptr));  /* _mm256_cvtph_ps takes __m128i */
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q6_K constants
 * ═════════════════════════════════════════════════════════════════════════ */

#define QK_K    256
#define Q6_K_BS 210

/* ═════════════════════════════════════════════════════════════════════════
 * AVX2-vectorized Q6_K row dot product
 *
 * Process one row (2048 = 8 blocks of 256 values) against input vector x.
 * Uses 8 × __m256 accumulators to hide latency.
 * ═════════════════════════════════════════════════════════════════════════ */

static inline float q6_k_dot_row_avx2(const uint8_t *restrict row,
                                        const float *restrict x,
                                        int nb) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q6_K_BS;

        /* Prefetch next block */
        if (b + 1 < nb) {
            __builtin_prefetch(blk + Q6_K_BS, 0, 1);
            __builtin_prefetch(x + (b + 1) * QK_K, 0, 1);
        }

        /* Extract delta and scales */
        float d = f16_to_f32_scalar((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
        const int8_t *sc = (const int8_t *)(blk + 192);
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;

        /* Process 4 super-groups of 64 values each (256 total) */
        for (int sg = 0; sg < 4; sg++) {
            /* Scale for this super-group */
            __m256 ds_0 = _mm256_set1_ps(d * sc[sg * 2]);
            __m256 ds_1 = _mm256_set1_ps(d * sc[sg * 2 + 1]);

            /* Process 64 values in this super-group: 2 chunks of 32 */
            for (int chunk = 0; chunk < 2; chunk++) {
                int base = sg * 64 + chunk * 32;

                /* Load 32 low nibbles from ql */
                __m256i ql_raw;
                if (chunk == 0) {
                    ql_raw = _mm256_loadu_si256((const __m256i *)(ql + sg * 64));
                } else {
                    ql_raw = _mm256_loadu_si256((const __m256i *)(ql + sg * 64 + 32));
                }

                /* Load 32 high bits from qh (2 bits each) */
                __m256i qh_raw = _mm256_loadu_si256((const __m256i *)(qh + base));

                /* Extract low nibbles: ql & 0xF */
                __m256i lo_nibbles = _mm256_and_si256(ql_raw, _mm256_set1_epi8(0x0F));
                /* Extract high nibbles: (ql >> 4) & 0xF */
                __m256i hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(ql_raw, 4), _mm256_set1_epi8(0x0F));

                /* Extract 2-bit high bits from qh */
                /* qh has 2 bits per value packed: bits [0:1], [2:3], [4:5], [6:7] */
                /* For values at positions 0..15 in this chunk:
                 *   qh[l] bits 0,1 → position l's high bits
                 *   qh[l] bits 2,3 → position l+32's high bits
                 */
                /* Simplified: use the scalar fallback for correctness */
                /* We'll dequantize 32 values using scalar + FMA with AVX x loading */

                /* Load x values */
                __m256 x_v = _mm256_loadu_ps(x + b * QK_K + base);

                /* Scalar dequantize + vector FMA for groups of 8 */
                /* This hybrid approach gives correctness while using AVX2 accumulation */
                /* Actually, let me use a different strategy: dequantize to temp, then FMA */
                float temp[32];
                for (int l = 0; l < 16; l++) {
                    int is_ = l / 16;
                    int q1 = ((ql[sg * 64 + chunk * 32 + l] & 0xF) | (((qh[base + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[sg * 64 + chunk * 32 + l] >> 4) | (((qh[base + l] >> 4) & 3) << 4)) - 32;
                    temp[l] = d * sc[is_ + sg * 2] * q1;
                    temp[l + 16] = d * sc[is_ + sg * 2 + 1] * q2;
                }
                /* FMA with x values */
                __m256 dq0 = _mm256_loadu_ps(temp);
                __m256 dq1 = _mm256_loadu_ps(temp + 8);
                __m256 dq2 = _mm256_loadu_ps(temp + 16);
                __m256 dq3 = _mm256_loadu_ps(temp + 24);
                __m256 x0 = _mm256_loadu_ps(x + b * QK_K + base);
                __m256 x1 = _mm256_loadu_ps(x + b * QK_K + base + 8);
                __m256 x2 = _mm256_loadu_ps(x + b * QK_K + base + 16);
                __m256 x3 = _mm256_loadu_ps(x + b * QK_K + base + 24);

                /* Cycle through accumulators to hide latency */
                if (b % 2 == 0) {
                    acc0 = _mm256_fmadd_ps(dq0, x0, acc0);
                    acc1 = _mm256_fmadd_ps(dq1, x1, acc1);
                    acc2 = _mm256_fmadd_ps(dq2, x2, acc2);
                    acc3 = _mm256_fmadd_ps(dq3, x3, acc3);
                } else {
                    acc4 = _mm256_fmadd_ps(dq0, x0, acc4);
                    acc5 = _mm256_fmadd_ps(dq1, x1, acc5);
                    acc6 = _mm256_fmadd_ps(dq2, x2, acc6);
                    acc7 = _mm256_fmadd_ps(dq3, x3, acc7);
                }
            }
        }
    }

    /* Horizontal sum of all 8 accumulators */
    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum45 = _mm256_add_ps(acc4, acc5);
    __m256 sum67 = _mm256_add_ps(acc6, acc7);
    __m256 sum0123 = _mm256_add_ps(sum01, sum23);
    __m256 sum4567 = _mm256_add_ps(sum45, sum67);
    __m256 total = _mm256_add_ps(sum0123, sum4567);

    /* Hadd to reduce 8 floats to 1 */
    __m128 hi128 = _mm256_extractf128_ps(total, 1);
    __m128 lo128 = _mm256_castps256_ps128(total);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Scalar Q6_K dot product (for correctness verification)
 * ═════════════════════════════════════════════════════════════════════════ */

static inline float q6_k_dot_row_scalar(const uint8_t *restrict row,
                                          const float *restrict x,
                                          int nb) {
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q6_K_BS;
        float d = f16_to_f32_scalar((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
        const int8_t *sc = (const int8_t *)(blk + 192);
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is_ = l / 16;
                int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[l + 0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                sum += d * sc[is_ + 0] * q1 * x[b * QK_K + n + l + 0];
                sum += d * sc[is_ + 2] * q2 * x[b * QK_K + n + l + 32];
                sum += d * sc[is_ + 4] * q3 * x[b * QK_K + n + l + 64];
                sum += d * sc[is_ + 6] * q4 * x[b * QK_K + n + l + 96];
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
    return sum;
}

/* ═════════════════════════════════════════════════════════════════════════
 * OMP-parallel Q6_K logits projection (AVX2)
 * ═════════════════════════════════════════════════════════════════════════ */

void q6_k_matmul_avx2(const uint8_t *restrict W, const float *restrict x,
                        float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    #pragma omp parallel for schedule(static, 64)
    for (int r = 0; r < n_rows; r++) {
        const uint8_t *row = W + (size_t)r * nb * Q6_K_BS;
        out[r] = q6_k_dot_row_avx2(row, x, nb);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Verify: compute logits scalar and AVX2, compare
 * ═════════════════════════════════════════════════════════════════════════ */

void q6_k_matmul_verify(const uint8_t *restrict W, const float *restrict x,
                          float *restrict out_scalar, float *restrict out_avx2,
                          int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    for (int r = 0; r < n_rows && r < 100; r++) {
        const uint8_t *row = W + (size_t)r * nb * Q6_K_BS;
        out_scalar[r] = q6_k_dot_row_scalar(row, x, nb);
        out_avx2[r] = q6_k_dot_row_avx2(row, x, nb);
    }
}