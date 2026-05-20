/* Unified quantized matmul kernels — AVX2+FMA+OpenMP
 *
 * Supports: F32, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0
 * Plus: rms_norm, silu, softmax, batch_matmul, thread control
 *
 * Quant type codes (ggml/GGUF convention):
 *   0  = F32
 *   2  = Q4_0   (18 bytes/32 vals)
 *   3  = Q4_1   (20 bytes/32 vals)
 *   8  = Q8_0   (34 bytes/32 vals)
 *  12  = Q4_K   (144 bytes/256 vals)
 *  13  = Q5_K   (176 bytes/256 vals)
 *  14  = Q6_K   (210 bytes/256 vals)
 *  39  = MXFP4  (17 bytes/32 vals)
 *
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *          -o quant_kernels_omp.so quant_kernels_omp.c -lm
 */

#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

#define QK_K      256
#define QK8_0     32

/* Block sizes in bytes */
#define Q4_0_BS   18
#define Q4_1_BS   20
#define Q4_K_BS   144
#define Q5_K_BS   176
#define Q6_K_BS   210
#define Q8_0_BS   34
#define MXFP4_BS  17

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

/* Forward declarations for functions called before definition */
int quantize_row_q8_0(const float *restrict x, uint8_t *restrict q8, int n_cols);
void q4_k_q8_0_matmul_omp(const uint8_t *restrict W, const uint8_t *restrict x_q8,
                           float *restrict out, int n_rows, int n_cols);
void q6_k_matmul_avx2_omp(const uint8_t *restrict W, const float *restrict x,
                           float *restrict out, int n_rows, int n_cols);
float q4_k_row_dot_avx2(const uint8_t *restrict W, const float *restrict x, int n_cols, int row);
float q5_k_row_dot_avx2(const uint8_t *restrict W, const float *restrict x, int n_cols, int row);
float q6_k_row_dot_avx2(const uint8_t *restrict W, const float *restrict x, int n_cols, int row);
float q4_k_row_dot_q8(const uint8_t *restrict W, const uint8_t *restrict x_q8, int n_cols, int row);

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_K scale/min unpacking  (from ggml get_scale_min_k4)
 *
 * The 12-byte scales[] array packs 8 (scale, min) pairs using 6 bits each.
 * Pairs 0..3: scale = scales[0..3] & 63,  min = scales[4..7] & 63
 * Pairs 4..7: scale and min reconstructed from bits of scales[8..11]
 *             plus upper 2 bits of scales[0..3] and scales[4..7].
 * ═══════════════════════════════════════════════════════════════════════ */

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
 * Q4_0 / Q4_1 AVX2 decode + dot (from q4_kernel_omp.c)
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

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_0 matmul
 * ═══════════════════════════════════════════════════════════════════════ */

void q4_0_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        const uint8_t *row = W + (size_t)r * bpr * Q4_0_BS;
        for (int blk = 0; blk < bpr; blk++) {
            /* Prefetch next weight block and next activation chunk */
            if (blk + 1 < bpr) {
                __builtin_prefetch(row + (blk + 1) * Q4_0_BS, 0, 1);
                __builtin_prefetch(x + (blk + 1) * 32, 0, 1);
            }
            __m256 v[4];
            decode_q4_0(row + blk * Q4_0_BS, v);
            total += block_dot_fma(v, x + blk * 32);
        }
        out[r] = total;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_1 matmul
 * ═══════════════════════════════════════════════════════════════════════ */

void q4_1_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        const uint8_t *row = W + (size_t)r * bpr * Q4_1_BS;
        for (int blk = 0; blk < bpr; blk++) {
            if (blk + 1 < bpr) {
                __builtin_prefetch(row + (blk + 1) * Q4_1_BS, 0, 1);
                __builtin_prefetch(x + (blk + 1) * 32, 0, 1);
            }
            __m256 v[4];
            decode_q4_1(row + blk * Q4_1_BS, v);
            total += block_dot_fma(v, x + blk * 32);
        }
        out[r] = total;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_K matmul  (144 bytes per 256 values)
 *
 * Block layout:
 *   offset 0:  d      (fp16, 2 bytes) — super-block scale
 *   offset 2:  dmin   (fp16, 2 bytes) — super-block min scale
 *   offset 4:  scales  (12 bytes)      — packed 6-bit scale/min pairs
 *   offset 16: qs      (128 bytes)     — 4-bit quantized values
 *
 * Dequantization (per group of 64 values, 4 groups per super-block):
 *   For each group (j = 0, 64, 128, 192):
 *     scale_idx = 2*(j/64)
 *     get_scale_min_k4(scale_idx+0, scales, &sc, &m) → d1=d*sc, m1=min*m
 *     get_scale_min_k4(scale_idx+1, scales, &sc, &m) → d2=d*sc, m2=min*m
 *     pos j+0..j+31:  d1 * (qs[l] & 0xF) - m1
 *     pos j+32..j+63: d2 * (qs[l] >> 4)  - m2
 *     qs advances by 32 per group
 * ═══════════════════════════════════════════════════════════════════════ */

void q4_k_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int nb_blocks = n_cols / 32;
    uint8_t x_q8_buf[2048 / 32 * Q8_0_BS];  /* stack alloc for up to 2048 dims */
    int q8_sz = (size_t)nb_blocks * Q8_0_BS;
    uint8_t *x_q8 = q8_sz <= sizeof(x_q8_buf) ? x_q8_buf : (uint8_t*)malloc(q8_sz);
    if (!x_q8) { for (int r = 0; r < n_rows; r++) out[r] = 0.0f; return; }
    quantize_row_q8_0(x, x_q8, n_cols);
    q4_k_q8_0_matmul_omp(W, x_q8, out, n_rows, n_cols);
    if (x_q8 != x_q8_buf) free(x_q8);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q5_K matmul  (176 bytes per 256 values)
 *
 * Block layout:
 *   offset 0:  d      (fp16, 2 bytes)
 *   offset 2:  dmin   (fp16, 2 bytes)
 *   offset 4:  scales (12 bytes)
 *   offset 16: qh     (32 bytes)  — 5th bit per value (1 bit/value for 256 values)
 *   offset 48: qs     (128 bytes) — lower 4 bits per value (nibbles)
 *
 * Dequantization: same scales as Q4_K, plus 5th bit from qh.
 *   For each group j (j = 0, 64, 128, 192), with shift counters u1, u2:
 *     u1 starts at 1, u2 starts at 2; after each group u1<<=2, u2<<=2
 *     pos j+0..j+31:   d1 * ((qs[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1
 *     pos j+32..j+63:  d2 * ((qs[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2
 * ═══════════════════════════════════════════════════════════════════════ */

/* ═════════════════════════════════════════════════════════════════════════
 * Q5_K matmul — AVX2 vectorized dequant + FMA
 *
 * Block layout (176 bytes per 256 values):
 *   offset 0:  d      (fp16, 2 bytes)
 *   offset 2:  dmin   (fp16, 2 bytes)
 *   offset 4:  scales (12 bytes) — packed 6-bit pairs (same as Q4_K)
 *   offset 16: qh     (32 bytes) — 1 extra bit per value
 *   offset 48: qs     (128 bytes) — lower 4 bits (2 nibbles/byte)
 *
 * Each super-block has 4 sub-blocks of 64 values. Each sub-block has 2
 * scale/min pairs (d1,m1 for lo-nibble, d2,m2 for hi-nibble).
 *
 * Q5_K value = (qs nibble) + ((qh & mask) ? 16 : 0)
 *   lo-nibble group mask = u1 (1, 4, 16, 64)
 *   hi-nibble group mask = u2 (2, 8, 32, 128)
 *
 * Uses identity: sum((d1*val - m1)*x) = d1*sum(val*x) - m1*sum(x)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Q5_K row-dot with AVX2 dequant + FMA (extracted for batch_qkv_omp) */
float q5_k_row_dot_avx2(const uint8_t *restrict W, const float *restrict x,
                                 int n_cols, int row) {
    int nb = n_cols / QK_K;
    const uint8_t *row_ptr = W + (size_t)row * nb * Q5_K_BS;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row_ptr + (size_t)b * Q5_K_BS;
        float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t *scales = blk + 4;
        const uint8_t *qh_arr = blk + 16;
        const uint8_t *qs = blk + 48;
        int x_off = b * QK_K;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;

        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * (float)sc;  float m1 = min * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * (float)sc;  float m2 = min * (float)m;

            /* ── AVX2: lo-nibble group (32 values) ── */
            __m256 acc_nx = _mm256_setzero_ps();
            __m256 acc_x  = _mm256_setzero_ps();

            for (int l = 0; l < 32; l += 16) {
                __m128i qs16 = _mm_loadu_si128((const __m128i*)(qs + l));
                __m128i nib = _mm_and_si128(qs16, _mm_set1_epi8(0x0F));
                __m128i qh16 = _mm_loadu_si128((const __m128i*)(qh_arr + l));
                __m128i qh_test = _mm_and_si128(qh16, _mm_set1_epi8((char)u1));
                __m128i qh_cond = _mm_cmpeq_epi8(qh_test, _mm_setzero_si128());
                qh_cond = _mm_xor_si128(qh_cond, _mm_set1_epi8(0xFF));
                qh_cond = _mm_and_si128(qh_cond, _mm_set1_epi8(16));
                __m128i val = _mm_or_si128(nib, qh_cond);
                __m128i v16a = _mm_cvtepu8_epi16(val);
                __m128i v16b = _mm_cvtepu8_epi16(_mm_shuffle_epi32(val, 0x4e));
                __m256 vfa = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    _mm_cvtepi16_epi32(_mm_shuffle_epi32(v16a, 0x4e)),
                    _mm_cvtepi16_epi32(v16a)));
                __m256 vfb = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    _mm_cvtepi16_epi32(_mm_shuffle_epi32(v16b, 0x4e)),
                    _mm_cvtepi16_epi32(v16b)));
                __m256 xa = _mm256_loadu_ps(x + x_off + j + l);
                __m256 xb = _mm256_loadu_ps(x + x_off + j + l + 8);
                acc_nx = _mm256_fmadd_ps(vfa, xa, acc_nx);
                acc_nx = _mm256_fmadd_ps(vfb, xb, acc_nx);
                acc_x  = _mm256_add_ps(xa, acc_x);
                acc_x  = _mm256_add_ps(xb, acc_x);
            }
            __m256 h_nx = _mm256_hadd_ps(acc_nx, _mm256_permute2f128_ps(acc_nx, acc_nx, 1));
            h_nx = _mm256_hadd_ps(h_nx, h_nx); h_nx = _mm256_hadd_ps(h_nx, h_nx);
            __m256 h_x  = _mm256_hadd_ps(acc_x,  _mm256_permute2f128_ps(acc_x,  acc_x,  1));
            h_x  = _mm256_hadd_ps(h_x,  h_x);  h_x  = _mm256_hadd_ps(h_x,  h_x);
            sum += d1 * _mm256_cvtss_f32(h_nx) - m1 * _mm256_cvtss_f32(h_x);

            /* ── AVX2: hi-nibble group (32 values) ── */
            acc_nx = _mm256_setzero_ps();
            acc_x  = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 16) {
                __m128i qs16 = _mm_loadu_si128((const __m128i*)(qs + l));
                __m128i nib = _mm_and_si128(_mm_srli_epi16(qs16, 4), _mm_set1_epi8(0x0F));
                __m128i qh16 = _mm_loadu_si128((const __m128i*)(qh_arr + l));
                __m128i qh_test = _mm_and_si128(qh16, _mm_set1_epi8((char)u2));
                __m128i qh_cond = _mm_cmpeq_epi8(qh_test, _mm_setzero_si128());
                qh_cond = _mm_xor_si128(qh_cond, _mm_set1_epi8(0xFF));
                qh_cond = _mm_and_si128(qh_cond, _mm_set1_epi8(16));
                __m128i val = _mm_or_si128(nib, qh_cond);
                __m128i v16a = _mm_cvtepu8_epi16(val);
                __m128i v16b = _mm_cvtepu8_epi16(_mm_shuffle_epi32(val, 0x4e));
                __m256 vfa = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    _mm_cvtepi16_epi32(_mm_shuffle_epi32(v16a, 0x4e)),
                    _mm_cvtepi16_epi32(v16a)));
                __m256 vfb = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    _mm_cvtepi16_epi32(_mm_shuffle_epi32(v16b, 0x4e)),
                    _mm_cvtepi16_epi32(v16b)));
                __m256 xa = _mm256_loadu_ps(x + x_off + j + 32 + l);
                __m256 xb = _mm256_loadu_ps(x + x_off + j + 32 + l + 8);
                acc_nx = _mm256_fmadd_ps(vfa, xa, acc_nx);
                acc_nx = _mm256_fmadd_ps(vfb, xb, acc_nx);
                acc_x  = _mm256_add_ps(xa, acc_x);
                acc_x  = _mm256_add_ps(xb, acc_x);
            }
            h_nx = _mm256_hadd_ps(acc_nx, _mm256_permute2f128_ps(acc_nx, acc_nx, 1));
            h_nx = _mm256_hadd_ps(h_nx, h_nx); h_nx = _mm256_hadd_ps(h_nx, h_nx);
            h_x  = _mm256_hadd_ps(acc_x,  _mm256_permute2f128_ps(acc_x,  acc_x,  1));
            h_x  = _mm256_hadd_ps(h_x,  h_x);  h_x  = _mm256_hadd_ps(h_x,  h_x);
            sum += d2 * _mm256_cvtss_f32(h_nx) - m2 * _mm256_cvtss_f32(h_x);

            qs += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
    return sum;
}

/* Q5_K matmul — AVX2 vectorized, uses q5_k_row_dot_avx2 per row */
void q5_k_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        out[r] = q5_k_row_dot_avx2(W, x, n_cols, r);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q8_0 matmul  (34 bytes per 32 values)
 *
 * Block layout:
 *   offset 0: d    (fp16, 2 bytes) — scale
 *   offset 2: qs   (int8_t[32])     — quantized values
 *
 * Dequantization: value = d * qs[i]
 * ═══════════════════════════════════════════════════════════════════════ */

void q8_0_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / QK8_0;  /* blocks of 32 per row */

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        const uint8_t *row = W + (size_t)r * bpr * Q8_0_BS;
        for (int blk = 0; blk < bpr; blk++) {
            if (blk + 1 < bpr) {
                __builtin_prefetch(row + (blk + 1) * Q8_0_BS, 0, 1);
                __builtin_prefetch(x + (blk + 1) * 32, 0, 1);
            }
            const uint8_t *bp = row + blk * Q8_0_BS;
            float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            const int8_t *qs = (const int8_t *)(bp + 2);
            int off = blk * 32;
            __m256 d_v = _mm256_set1_ps(d);

            /* Process 32 int8 values: 4 groups of 8.
             * Each group: load 8 int8, sign-extend to 8 int32 via int16,
             * convert to float, multiply by scale, dot with x */
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < 32; i += 8) {
                /* Load 8 int8 values and sign-extend to 8 int32 */
                __m128i q8 = _mm_loadl_epi64((__m128i*)(qs + i));  /* load 8 bytes */
                __m128i q16lo = _mm_cvtepi8_epi16(q8);              /* sign-extend to 8 int16 */
                __m128i q16hi = _mm_cvtepi16_epi32(q16lo);          /* low 4 int16 → int32 */
                __m128i q16lo2 = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(q16lo, q16lo));  /* high 4 int16 → int32 */
                __m256 qf = _mm256_cvtepi32_ps(_mm256_set_m128i(q16lo2, q16hi));
                __m256 xv = _mm256_loadu_ps(x + off + i);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(qf, d_v), xv, acc);
            }
            /* Horizontal sum of 8 floats */
            __m128 hi128 = _mm256_extractf128_ps(acc, 1);
            __m128 lo128 = _mm256_castps256_ps128(acc);
            __m128 sum = _mm_add_ps(lo128, hi128);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            total += _mm_cvtss_f32(sum);
        }
        out[r] = total;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * MXFP4 matmul  (17 bytes per 32 values)
 *
 * Block layout:
 *   q[16]    — 32 × 4-bit twos-complement mantissas packed into 16 bytes
 *   e[1]     — E8M0 scale exponent (unsigned, bias=127)
 *
 * Dequantization: value = nibble_sign_ext * 2^(e - 127)
 * Optimizations: int-bit-trick scale, FMA, vertical accum, prefetch
 * ═══════════════════════════════════════════════════════════════════════ */

/* MXFP4 block structure */
typedef struct {
    uint8_t q[16];
    uint8_t e;
} block_mxfp4;

void mxfp4_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;  /* blocks of 32 per row */

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        const uint8_t *row = W + (size_t)r * bpr * sizeof(block_mxfp4);
        for (int blk = 0; blk < bpr; blk++) {
            int next = blk + 1;
            if (next < bpr) {
                __builtin_prefetch(row + next * sizeof(block_mxfp4), 0, 1);
            }

            const uint8_t *bp = row + blk * sizeof(block_mxfp4);
            /* E8M0 exponent → IEEE754 float32: 2^(e-127) = float with bits (e << 23) */
            union { uint32_t u; float f; } sc = { .u = ((uint32_t)bp[16]) << 23 };
            __m256 sv = _mm256_set1_ps(sc.f);

            __m128i packed = _mm_loadu_si128((const __m128i*)bp);
            /* Extract lower/upper 4-bit nibbles */
            __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
            /* Sign extend 4-bit to signed int8: nibble >= 8 → nibble - 16 */
            __m128i sign_lo = _mm_cmpgt_epi8(lo, _mm_set1_epi8(7));
            __m128i sign_hi = _mm_cmpgt_epi8(hi, _mm_set1_epi8(7));
            lo = _mm_sub_epi8(lo, _mm_and_si128(sign_lo, _mm_set1_epi8(16)));
            hi = _mm_sub_epi8(hi, _mm_and_si128(sign_hi, _mm_set1_epi8(16)));
            /* Interleave to restore natural order */
            __m128i vals_lo = _mm_unpacklo_epi8(lo, hi);
            __m128i vals_hi = _mm_unpackhi_epi8(lo, hi);

            /* Dequant + vertical FMA accumulate across 4 quarter-blocks */
            int o = blk * 32;
            __m256 d0 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_lo)), sv);
            acc0 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(x + o), acc0);

            __m256 d1 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_srli_si128(vals_lo, 8))), sv);
            acc1 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(x + o + 8), acc1);

            __m256 d2 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_hi)), sv);
            acc2 = _mm256_fmadd_ps(d2, _mm256_loadu_ps(x + o + 16), acc2);

            __m256 d3 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_srli_si128(vals_hi, 8))), sv);
            acc3 = _mm256_fmadd_ps(d3, _mm256_loadu_ps(x + o + 24), acc3);
        }

        /* Single horizontal reduction per row */
        __m128 hi0 = _mm256_extractf128_ps(acc0, 1);
        __m128 lo0 = _mm256_castps256_ps128(acc0);
        __m128 hi1 = _mm256_extractf128_ps(acc1, 1);
        __m128 lo1 = _mm256_castps256_ps128(acc1);
        __m128 hi2 = _mm256_extractf128_ps(acc2, 1);
        __m128 lo2 = _mm256_castps256_ps128(acc2);
        __m128 hi3 = _mm256_extractf128_ps(acc3, 1);
        __m128 lo3 = _mm256_castps256_ps128(acc3);

        __m128 sum01 = _mm_add_ps(_mm_add_ps(lo0, hi0), _mm_add_ps(lo1, hi1));
        __m128 sum23 = _mm_add_ps(_mm_add_ps(lo2, hi2), _mm_add_ps(lo3, hi3));
        __m128 total = _mm_add_ps(sum01, sum23);
        total = _mm_hadd_ps(total, total);
        total = _mm_hadd_ps(total, total);
        out[r] = _mm_cvtss_f32(total);
    }
}

/* Scalar row dot for row_range dispatcher (bpr = n_cols / 32) */
static float mxfp4_row_dot(const uint8_t *restrict W, const float *restrict x,
                            int bpr, int row) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const uint8_t *row_ptr = W + (size_t)row * bpr * sizeof(block_mxfp4);
    for (int blk = 0; blk < bpr; blk++) {
        const uint8_t *bp = row_ptr + blk * sizeof(block_mxfp4);
        union { uint32_t u; float f; } sc = { .u = ((uint32_t)bp[16]) << 23 };
        __m256 sv = _mm256_set1_ps(sc.f);
        __m128i packed = _mm_loadu_si128((const __m128i*)bp);
        __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
        __m128i sign_lo = _mm_cmpgt_epi8(lo, _mm_set1_epi8(7));
        __m128i sign_hi = _mm_cmpgt_epi8(hi, _mm_set1_epi8(7));
        lo = _mm_sub_epi8(lo, _mm_and_si128(sign_lo, _mm_set1_epi8(16)));
        hi = _mm_sub_epi8(hi, _mm_and_si128(sign_hi, _mm_set1_epi8(16)));
        __m128i vals_lo = _mm_unpacklo_epi8(lo, hi);
        __m128i vals_hi = _mm_unpackhi_epi8(lo, hi);
        int o = blk * 32;
        __m256 d0 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_lo)), sv);
        acc0 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(x + o), acc0);
        __m256 d1 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_lo, 8))), sv);
        acc1 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(x + o + 8), acc1);
        __m256 d2 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_hi)), sv);
        acc2 = _mm256_fmadd_ps(d2, _mm256_loadu_ps(x + o + 16), acc2);
        __m256 d3 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_hi, 8))), sv);
        acc3 = _mm256_fmadd_ps(d3, _mm256_loadu_ps(x + o + 24), acc3);
    }
    __m128 hi0 = _mm256_extractf128_ps(acc0, 1);
    __m128 lo0 = _mm256_castps256_ps128(acc0);
    __m128 hi1 = _mm256_extractf128_ps(acc1, 1);
    __m128 lo1 = _mm256_castps256_ps128(acc1);
    __m128 hi2 = _mm256_extractf128_ps(acc2, 1);
    __m128 lo2 = _mm256_castps256_ps128(acc2);
    __m128 hi3 = _mm256_extractf128_ps(acc3, 1);
    __m128 lo3 = _mm256_castps256_ps128(acc3);
    __m128 sum01 = _mm_add_ps(_mm_add_ps(lo0, hi0), _mm_add_ps(lo1, hi1));
    __m128 sum23 = _mm_add_ps(_mm_add_ps(lo2, hi2), _mm_add_ps(lo3, hi3));
    __m128 total = _mm_add_ps(sum01, sum23);
    total = _mm_hadd_ps(total, total);
    total = _mm_hadd_ps(total, total);
    return _mm_cvtss_f32(total);
}

/* MXFP4 row dot using Q8_0 quantized input — uses VPMADDWD for integer dot product.
 * x_q8 has been quantized to Q8_0 format (34 bytes per 32 values) once upstream.
 * Combined scale = mxfp4_scale * q8_scale applied per block via FMA. */
static float mxfp4_row_dot_q8(const uint8_t *restrict W, const uint8_t *restrict x_q8,
                                int n_cols, int row) {
    int bpr = n_cols / 32;
    const uint8_t *rp = W + (size_t)row * bpr * sizeof(block_mxfp4);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (int blk = 0; blk < bpr; blk++) {
        const uint8_t *bp = rp + blk * sizeof(block_mxfp4);
        const uint8_t *xbp = x_q8 + blk * Q8_0_BS;
        float q8_d = f16_to_f32(*(const uint16_t*)xbp);
        union { uint32_t u; float f; } sc = { .u = ((uint32_t)bp[16]) << 23 };
        __m256 cv = _mm256_set1_ps(sc.f * q8_d);
        __m128i p = _mm_loadu_si128((const __m128i*)bp);
        __m128i lo = _mm_and_si128(p, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(p, 4), _mm_set1_epi8(0x0F));
        __m128i sl = _mm_cmpgt_epi8(lo, _mm_set1_epi8(7));
        __m128i sh = _mm_cmpgt_epi8(hi, _mm_set1_epi8(7));
        lo = _mm_sub_epi8(lo, _mm_and_si128(sl, _mm_set1_epi8(16)));
        hi = _mm_sub_epi8(hi, _mm_and_si128(sh, _mm_set1_epi8(16)));
        __m128i vlo = _mm_unpacklo_epi8(lo, hi);
        __m128i vhi = _mm_unpackhi_epi8(lo, hi);
        __m256i q8v = _mm256_loadu_si256((const __m256i*)(xbp + 2));
        /* int16 × int16 → int32 via VPMADDWD */
        __m256i dl = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(vlo),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8v)));
        __m256i dh = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(vhi),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8v, 1)));
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dl), cv, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dh), cv, acc1);
    }
    __m256 t = _mm256_add_ps(acc0, acc1);
    __m128 th = _mm256_extractf128_ps(t, 1);
    __m128 tl = _mm256_castps256_ps128(t);
    __m128 s = _mm_add_ps(tl, th);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* MXFP4 row dot using F32 input — vertical accum + FMA + int-bit scale.
 * For use when input is F32 (not pre-quantized), like the down projection
 * which takes SiLU'd expert output as input. */
static float mxfp4_row_dot_f32(const uint8_t *restrict W, const float *restrict x,
                                 int n_cols, int row) {
    int bpr = n_cols / 32;
    const uint8_t *rp = W + (size_t)row * bpr * sizeof(block_mxfp4);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (int blk = 0; blk < bpr; blk++) {
        const uint8_t *bp = rp + blk * sizeof(block_mxfp4);
        union { uint32_t u; float f; } sc = { .u = ((uint32_t)bp[16]) << 23 };
        __m256 sv = _mm256_set1_ps(sc.f);
        __m128i p = _mm_loadu_si128((const __m128i*)bp);
        __m128i lo = _mm_and_si128(p, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(p, 4), _mm_set1_epi8(0x0F));
        __m128i sl = _mm_cmpgt_epi8(lo, _mm_set1_epi8(7));
        __m128i sh = _mm_cmpgt_epi8(hi, _mm_set1_epi8(7));
        lo = _mm_sub_epi8(lo, _mm_and_si128(sl, _mm_set1_epi8(16)));
        hi = _mm_sub_epi8(hi, _mm_and_si128(sh, _mm_set1_epi8(16)));
        __m128i vlo = _mm_unpacklo_epi8(lo, hi);
        __m128i vhi = _mm_unpackhi_epi8(lo, hi);
        int o = blk * 32;
        __m256 d0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vlo)), sv);
        __m256 d1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vlo, 8))), sv);
        __m256 d2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vhi)), sv);
        __m256 d3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vhi, 8))), sv);
        acc0 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(x + o), acc0);
        acc1 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(x + o + 8), acc1);
        acc2 = _mm256_fmadd_ps(d2, _mm256_loadu_ps(x + o + 16), acc2);
        acc3 = _mm256_fmadd_ps(d3, _mm256_loadu_ps(x + o + 24), acc3);
    }
    __m256 t = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 th = _mm256_extractf128_ps(t, 1);
    __m128 tl = _mm256_castps256_ps128(t);
    __m128 s = _mm_add_ps(tl, th);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* MXFP4 full matmul using Q8_0 pre-quantized input — calls row_dot_q8
 * in a single OMP parallel region. Much faster than per-row dispatch
 * from moe_forward_omp because OMP overhead is amortized across all rows. */
static void mxfp4_matmul_omp_q8(const uint8_t *restrict W, const uint8_t *restrict x_q8,
                                  float *restrict out, int n_rows, int n_cols) {
    int bpr = n_cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        out[r] = mxfp4_row_dot_q8(W, x_q8, n_cols, r);
    }
}

void q6_k_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                      float *restrict out, int n_rows, int n_cols) {
    q6_k_matmul_avx2_omp(W, x, out, n_rows, n_cols);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q6_K dequantize to F32 (for verification and output requantization)
 *
 * Q6_K layout (210 bytes per 256 values):
 *   offset 0:   ql[128]   — lower 4 bits (2 nibbles/byte)
 *   offset 128: qh[64]    — upper 2 bits (4 groups of 2 bits per byte)
 *   offset 192: sc[16]    — signed 6-bit per-sub-block scales
 *   offset 208: d         (fp16, 2 bytes) — super-block scale
 *
 * Dequantization:
 *   For each 128-element half-block (h=0,128):
 *     for l in 0..31:
 *       is = l/16  (0 or 1, selects sc[is*2+0..7] within half)
 *       q1 = ((ql[l] & 0xF) | ((qh[l] >> 0) & 0x3) << 4) - 32
 *       q2 = ((ql[l+32] & 0xF) | ((qh[l] >> 2) & 0x3) << 4) - 32
 *       q3 = ((ql[l] >> 4)  | ((qh[l] >> 4) & 0x3) << 4) - 32
 *       q4 = ((ql[l+32] >> 4) | ((qh[l] >> 6) & 0x3) << 4) - 32
 *       val = d * sc[is*2+0] * q1,
 *             d * sc[is*2+2] * q2,
 *             d * sc[is*2+4] * q3,
 *             d * sc[is*2+6] * q4
 * ═══════════════════════════════════════════════════════════════════════ */

void q6_k_dequantize_row(const uint8_t *restrict W, float *restrict out, int n_values) {
    int nb = n_values / QK_K;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = W + (size_t)b * Q6_K_BS;
        const uint8_t *ql = blk;              /* offset 0: ql[128] */
        const uint8_t *qh = blk + 128;        /* offset 128: qh[64] */
        const int8_t *sc = (const int8_t *)(blk + 192); /* offset 192: scales[16] */
        float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));

        for (int h = 0; h < 2; h++) {
            int ql_off = h * 64;
            int qh_off = h * 32;
            int sc_off = h * 8;

            for (int l = 0; l < 32; l++) {
                int is = l / 16;  /* 0 or 1 */

                int q1 = ((ql[ql_off + l]       & 0x0F) | ((qh[qh_off + l]       & 0x03) << 4)) - 32;
                int q2 = ((ql[ql_off + l + 32]  & 0x0F) | ((qh[qh_off + l]       & 0x0C) << 2)) - 32;
                int q3 = ((ql[ql_off + l]       >> 4)   | ((qh[qh_off + l]       & 0x30)     )) - 32;
                int q4 = ((ql[ql_off + l + 32]  >> 4)   | ((qh[qh_off + l]       & 0xC0) >> 2)) - 32;

                float ds1 = d * (float)sc[sc_off + is + 0];
                float ds2 = d * (float)sc[sc_off + is + 2];
                float ds3 = d * (float)sc[sc_off + is + 4];
                float ds4 = d * (float)sc[sc_off + is + 6];

                out[h * 128 + l]        = ds1 * (float)q1;
                out[h * 128 + l + 32]   = ds2 * (float)q2;
                out[h * 128 + l + 64]   = ds3 * (float)q3;
                out[h * 128 + l + 96]   = ds4 * (float)q4;
            }
        }
        out += QK_K;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q5_K dequantize to F32 (for verification)
 * ═══════════════════════════════════════════════════════════════════════ */


void q4_k_dequantize_row(const uint8_t *restrict W, float *restrict out, int n_values) {
    int nb = n_values / QK_K;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = W + (size_t)b * Q4_K_BS;
        float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t *scales = blk + 4;
        const uint8_t *q = blk + 16;
        int is = 0;

        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc;  float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc;  float m2 = min * m;

            for (int l = 0; l < 32; ++l) out[l]      = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) out[l + 32]  = d2 * (q[l] >> 4)  - m2;
            out += 64;
            q += 32;
            is += 2;
        }
    }
}

void q5_k_dequantize_row(const uint8_t *restrict W, float *restrict out, int n_values) {
    int nb = n_values / QK_K;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = W + (size_t)b * Q5_K_BS;
        float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t *scales = blk + 4;
        const uint8_t *qh_arr = blk + 16;
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
                out[l]      = d1 * ((ql[l] & 0xF) + ((qh_arr[l] & u1) ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                out[l + 32] = d2 * ((ql[l] >> 4) + ((qh_arr[l] & u2) ? 16 : 0)) - m2;
            }
            out += 64;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q8_0 dequantize to F32 (for verification)
 * ═══════════════════════════════════════════════════════════════════════ */

void q8_0_dequantize_row(const uint8_t *restrict W, float *restrict out, int n_values) {
    int nb = n_values / QK8_0;
    for (int b = 0; b < nb; b++) {
        const uint8_t *bp = W + (size_t)b * Q8_0_BS;
        float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
        const int8_t *qs = (const int8_t *)(bp + 2);
        for (int j = 0; j < QK8_0; ++j) {
            out[b * QK8_0 + j] = qs[j] * d;
        }
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * F32 matrix-vector multiply (OMP parallel)
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
        for (; c < n_cols; c++) {
            /* handled below */
        }
        __m256 h = _mm256_hadd_ps(total, _mm256_permute2f128_ps(total, total, 1));
        h = _mm256_hadd_ps(h, h);
        h = _mm256_hadd_ps(h, h);
        float result = _mm256_cvtss_f32(h);
        for (c = (n_cols / 16) * 16; c < n_cols; c++) {
            result += row[c] * x[c];
        }
        out[r] = result;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Batched matmul: up to 7 projections in one OpenMP call (Q4_0/Q4_1 only)
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
            const uint8_t* bp_w = w[p] + ((size_t)row * bp + blk) * ts[p];
            __m256 v[4]; dec(bp_w, v);
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
    for (; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Softmax
 * ═══════════════════════════════════════════════════════════════════════ */

void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ═════════════════════════════════════════════════════════════════════════
 * OMP SiLU (parallel)
 * ═══════════════════════════════════════════════════════════════════════ */

void silu_omp(float* x, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Thread control
 * ═══════════════════════════════════════════════════════════════════════ */

int get_max_threads(void) { return omp_get_max_threads(); }
void set_num_threads(int n) { omp_set_num_threads(n); }

/* ═════════════════════════════════════════════════════════════════════════
 * Batched layer matmul — computes attention QKV + output in a single
 * OMP parallel region, eliminating 6 fork/join barriers per layer.
 *
 * Compared to 7 separate quant_matmul_omp calls, this saves ~0.1-0.3ms
 * per layer from reduced thread synchronization overhead.
 * ═════════════════════════════════════════════════════════════════════════ */

/* Internal: compute a single matmul row for Q4_0 */
static float q4_0_row_dot(const uint8_t *restrict W, const float *restrict x,
                           int bpr, int row) {
    float total = 0.0f;
    for (int blk = 0; blk < bpr; blk++) {
        __m256 v[4];
        decode_q4_0(W + ((size_t)row * bpr + blk) * Q4_0_BS, v);
        total += block_dot_fma(v, x + blk * 32);
    }
    return total;
}

/* Internal: compute a single matmul row for Q4_1 */
static float q4_1_row_dot(const uint8_t *restrict W, const float *restrict x,
                           int bpr, int row) {
    float total = 0.0f;
    for (int blk = 0; blk < bpr; blk++) {
        __m256 v[4];
        decode_q4_1(W + ((size_t)row * bpr + blk) * Q4_1_BS, v);
        total += block_dot_fma(v, x + blk * 32);
    }
    return total;
}

/* Internal: compute a single matmul row for Q8_0 — uses same logic as q8_0_matmul_omp */
static float q8_0_row_dot(const uint8_t *restrict W, const float *restrict x,
                           int bpr, int row) {
    float total = 0.0f;
    for (int blk = 0; blk < bpr; blk++) {
        const uint8_t *bp = W + ((size_t)row * bpr + blk) * Q8_0_BS;
        float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
        const int8_t *qs = (const int8_t *)(bp + 2);
        int off = blk * 32;
        __m256 d_v = _mm256_set1_ps(d);
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < 32; i += 8) {
            __m128i q8 = _mm_loadl_epi64((__m128i*)(qs + i));
            __m128i q16lo = _mm_cvtepi8_epi16(q8);
            __m128i q16hi = _mm_cvtepi16_epi32(q16lo);
            __m128i q16lo2 = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(q16lo, q16lo));
            __m256 qf = _mm256_cvtepi32_ps(_mm256_set_m128i(q16lo2, q16hi));
            __m256 xv = _mm256_loadu_ps(x + off + i);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(qf, d_v), xv, acc);
        }
        __m128 hi128 = _mm256_extractf128_ps(acc, 1);
        __m128 lo128 = _mm256_castps256_ps128(acc);
        __m128 sum = _mm_add_ps(lo128, hi128);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        total += _mm_cvtss_f32(sum);
    }
    return total;
}

/* Internal: compute a range of rows [row_start, row_end) for quant_matmul */
static void matmul_row_range(const uint8_t *restrict W, const float *restrict x,
                              float *restrict out, int n_rows, int n_cols,
                              int quant_type, int row_start, int row_end) {
    int bpr32 = n_cols / 32;  /* blocks per row for 32-val quants */
    int bpr256 = n_cols / 256;
    for (int r = row_start; r < row_end; r++) {
        float total = 0.0f;
        switch (quant_type) {
            case 2: /* Q4_0 */
                total = q4_0_row_dot(W, x, bpr32, r);
                break;
            case 3: /* Q4_1 */
                total = q4_1_row_dot(W, x, bpr32, r);
                break;
            case 8: /* Q8_0 */
                total = q8_0_row_dot(W, x, bpr32, r);
                break;
            case 12: /* Q4_K */
                total = q4_k_row_dot_avx2(W, x, n_cols, r);
                break;
            case 13: /* Q5_K */
                total = q5_k_row_dot_avx2(W, x, n_cols, r);
                break;
            case 14: /* Q6_K */
                total = q6_k_row_dot_avx2(W, x, n_cols, r);
                break;
            case 39: /* MXFP4 */
                total = mxfp4_row_dot(W, x, bpr32, r);
                break;
            default:
                total = 0.0f;
                break;
        }
        out[r] = total;
    }
}

/* Batch QKV projections: compute Q, K, V matmuls in a single OMP region.
 * Since Q, K, V all use the same input vector (x_norm), we can parallelize
 * across all output rows of all three projections simultaneously.
 */
void batch_qkv_omp(const uint8_t *restrict Wq, const uint8_t *restrict Wk,
                    const uint8_t *restrict Wv,
                    const float *restrict x,
                    float *restrict out_q, float *restrict out_k, float *restrict out_v,
                    int nq, int nk, int nv, int n_cols,
                    int qtype_q, int qtype_k, int qtype_v) {
    /* Quantize x to Q8_0 once if any projection uses Q4_K — saves 30× x reads */
    int has_q4k = (qtype_q == 12 || qtype_k == 12 || qtype_v == 12);
    int nb_blocks = n_cols / 32;
    int q8_sz = (size_t)nb_blocks * Q8_0_BS;
    uint8_t x_q8_buf[2048 / 32 * Q8_0_BS];
    uint8_t *x_q8 = NULL;
    if (has_q4k) {
        x_q8 = q8_sz <= (int)sizeof(x_q8_buf) ? x_q8_buf : (uint8_t*)malloc((size_t)q8_sz);
        if (x_q8) quantize_row_q8_0(x, x_q8, n_cols);
    }

    int total_rows = nq + nk + nv;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_rows; i++) {
        if (i < nq) {
            if (qtype_q == 12 && x_q8) {
                out_q[i] = q4_k_row_dot_q8(Wq, x_q8, n_cols, i);
            } else {
                matmul_row_range(Wq, x, out_q, nq, n_cols, qtype_q, i, i + 1);
            }
        } else if (i < nq + nk) {
            int r = i - nq;
            if (qtype_k == 12 && x_q8) {
                out_k[r] = q4_k_row_dot_q8(Wk, x_q8, n_cols, r);
            } else {
                matmul_row_range(Wk, x, out_k, nk, n_cols, qtype_k, r, r + 1);
            }
        } else {
            int r = i - nq - nk;
            if (qtype_v == 12 && x_q8) {
                out_v[r] = q4_k_row_dot_q8(Wv, x_q8, n_cols, r);
            } else {
                matmul_row_range(Wv, x, out_v, nv, n_cols, qtype_v, r, r + 1);
            }
        }
    }

    if (x_q8 && x_q8 != x_q8_buf) free(x_q8);
}

/* Batch Gate+Up projections: compute both with single fork/join */
void batch_gate_up_omp(const uint8_t *restrict Wg, const uint8_t *restrict Wu,
                       const float *restrict x,
                       float *restrict out_gate, float *restrict out_up,
                       int ng, int nu, int n_cols,
                       int qtype_g, int qtype_u) {
    /* Quantize x to Q8_0 once if Q4_K — saves 30× x reads for gate+up */
    int has_q4k = (qtype_g == 12 || qtype_u == 12);
    int nb_blocks = n_cols / 32;
    int q8_sz = (size_t)nb_blocks * Q8_0_BS;
    uint8_t x_q8_buf[2048 / 32 * Q8_0_BS];
    uint8_t *x_q8 = NULL;
    if (has_q4k) {
        x_q8 = q8_sz <= (int)sizeof(x_q8_buf) ? x_q8_buf : (uint8_t*)malloc((size_t)q8_sz);
        if (x_q8) quantize_row_q8_0(x, x_q8, n_cols);
    }

    int total_rows = ng + nu;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_rows; i++) {
        if (i < ng) {
            if (qtype_g == 12 && x_q8) {
                out_gate[i] = q4_k_row_dot_q8(Wg, x_q8, n_cols, i);
            } else {
                matmul_row_range(Wg, x, out_gate, ng, n_cols, qtype_g, i, i + 1);
            }
        } else {
            int r = i - ng;
            if (qtype_u == 12 && x_q8) {
                out_up[r] = q4_k_row_dot_q8(Wu, x_q8, n_cols, r);
            } else {
                matmul_row_range(Wu, x, out_up, nu, n_cols, qtype_u, r, r + 1);
            }
        }
    }

    if (x_q8 && x_q8 != x_q8_buf) free(x_q8);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Unified dispatch: quant_matmul_omp
 *
 * quant_type: 0=F32, 2=Q4_0, 3=Q4_1, 7=Q4_K, 8=Q8_0, 9=Q5_K, 14=Q6_K
 * ═══════════════════════════════════════════════════════════════════════ */

void quant_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                       float *restrict out, int n_rows, int n_cols, int quant_type) {
    switch (quant_type) {
        case 0:  /* F32 */
            f32_matmul_omp((const float *)W, x, out, n_rows, n_cols);
            break;
        case 2:  /* Q4_0 */
            q4_0_matmul_omp(W, x, out, n_rows, n_cols);
            break;
        case 3:  /* Q4_1 */
            q4_1_matmul_omp(W, x, out, n_rows, n_cols);
            break;
        case 8:  /* Q8_0 */
            q8_0_matmul_omp(W, x, out, n_rows, n_cols);
            break;
        case 12: /* Q4_K */
            q4_k_matmul_omp(W, x, out, n_rows, n_cols);
            break;
        case 13: /* Q5_K */
            q5_k_matmul_omp(W, x, out, n_rows, n_cols);
            break;
        case 14: /* Q6_K */
            q6_k_matmul_omp(W, x, out, n_rows, n_cols);
            break;
        case 39: /* MXFP4 */
            mxfp4_matmul_omp(W, x, out, n_rows, n_cols);
            break;
        default:
            break;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * QUANTIZED-ACTIVATION PATH — the key llama.cpp optimization
 *
 * Instead of dequantizing weights to FP32, we:
 *   1. Quantize the activation vector x to Q8_0 (tiny, ~34 bytes per 32 values)
 *   2. Compute packed integer dot products using _mm256_maddubs_epi16
 *   3. Apply per-block scales with a single FMA at the end
 *
 * This halves memory bandwidth because weight data stays in packed form
 * (4-6 bits/value) instead of being expanded to FP32 (32 bits/value).
 * ═════════════════════════════════════════════════════════════════════════ */

/* Quantize FP32 vector to Q8_0 (34 bytes per 32 values: 2-byte fp16 scale + 32 int8 values)
 * AVX2 vectorized: processes 8 floats per iteration for absmax + quantize.
 * Returns number of Q8_0 blocks written (= n_cols / 32)
 */
int quantize_row_q8_0(const float *restrict x, uint8_t *restrict q8, int n_cols) {
    int nb = n_cols / 32;
    __m256 clamp_min = _mm256_set1_ps(-127.0f);
    __m256 clamp_max = _mm256_set1_ps(127.0f);
    __m256i mask_abs = _mm256_set1_epi32(0x7FFFFFFF);

    for (int b = 0; b < nb; b++) {
        const float *src = x + b * 32;
        uint8_t *dst = q8 + (size_t)b * Q8_0_BS;

        /* AVX2 absmax: process 8 floats at a time, 4 iterations for 32 total */
        __m256 absmax = _mm256_setzero_ps();
        for (int i = 0; i < 32; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            __m256 av = _mm256_and_ps(v, _mm256_castsi256_ps(mask_abs));
            absmax = _mm256_max_ps(absmax, av);
        }
        /* Horizontal max of 8 lanes */
        __m256 tmp = _mm256_max_ps(absmax, _mm256_permute2f128_ps(absmax, absmax, 1));
        tmp = _mm256_max_ps(tmp, _mm256_shuffle_ps(tmp, tmp, 0x4E));
        tmp = _mm256_max_ps(tmp, _mm256_shuffle_ps(tmp, tmp, 0xB1));
        float amax = _mm256_cvtss_f32(tmp);
        float d = amax / 127.0f;
        if (d == 0.0f) d = 1.0f;
        float id = 1.0f / d;

        /* Store fp16 scale */
        __m128 fv = _mm_load_ss(&d);
        __m128i hv = _mm_cvtps_ph(fv, _MM_FROUND_TO_NEAREST_INT);
        uint16_t d16;
        memcpy(&d16, &hv, 2);
        memcpy(dst, &d16, 2);

        /* AVX2 quantize: 8 floats at a time, saturate to int8 */
        int8_t *qs = (int8_t *)(dst + 2);
        __m256 id_v = _mm256_set1_ps(id);
        for (int i = 0; i < 32; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            v = _mm256_mul_ps(v, id_v);
            v = _mm256_min_ps(_mm256_max_ps(v, clamp_min), clamp_max);
            __m256i vi = _mm256_cvtps_epi32(v);  /* round to nearest (MXCSR default) */
            __m128i lo32 = _mm256_castsi256_si128(vi);
            __m128i hi32 = _mm256_extractf128_si256(vi, 1);
            __m128i i16 = _mm_packs_epi32(lo32, hi32);   /* signed sat 4×int32 → 8×int16 */
            __m128i i8  = _mm_packs_epi16(i16, i16);      /* signed sat 8×int16 → 8×int8 */
            _mm_storel_epi64((__m128i*)(qs + i), i8);
        }
    }
    return nb;
}

/* Q4_0 × Q8_0 dot product (32 values = 1 Q4_0 block × 1 Q8_0 block)
/* Q4_0 × Q8_0 dot product (32 values = 1 Q4_0 block × 1 Q8_0 block)
 *
 * Q4_0 layout: 18 bytes = [d:fp16][qs:16 bytes packed 4-bit]
 *   lo nibble of byte i → value[i]     (positions 0..15)
 *   hi nibble of byte i → value[i+16]  (positions 16..31)
 * Q4_0 value = d * (nibble - 8)  where nibble is unsigned 0..15
 *
 * Q8_0 layout: 34 bytes = [d8:fp16][qs:32 int8]
 *   value[j] = d8 * qs8[j]
 *
 * dot = d4 * d8 * (sum(nibble_i * q8_i) - 8 * sum(q8_i))
 */
static inline float dot_q4_0_q8_0(const uint8_t *restrict q4, const uint8_t *restrict q8) {
    float d4 = f16_to_f32((uint16_t)q4[0] | ((uint16_t)q4[1] << 8));
    float d8 = f16_to_f32((uint16_t)q8[0] | ((uint16_t)q8[1] << 8));
    const uint8_t *q4_qs = q4 + 2;
    const int8_t  *q8_qs = (const int8_t *)(q8 + 2);

    /* AVX2 implementation */
    __m128i q4_raw = _mm_loadu_si128((const __m128i *)q4_qs);
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i q4_lo = _mm_and_si128(q4_raw, lo_mask);                     /* unsigned 0-15, positions 0-15 */
    __m128i q4_hi = _mm_and_si128(_mm_srli_epi16(q4_raw, 4), lo_mask); /* unsigned 0-15, positions 16-31 */

    __m128i q8v0 = _mm_loadu_si128((const __m128i *)(q8_qs));        /* q8[0..15]  */
    __m128i q8v1 = _mm_loadu_si128((const __m128i *)(q8_qs + 16));   /* q8[16..31] */

    /* maddubs: unsigned × signed → i16 pair-sums
     * madd_lo[j] = q4_lo[2j]*q8[2j] + q4_lo[2j+1]*q8[2j+1] (8 i16 values)
     * madd_hi[j] = q4_hi[2j]*q8[2j+16] + q4_hi[2j+1]*q8[2j+17] (8 i16 values)
     * Total acc = sum(madd_lo) + sum(madd_hi) = sum(all nibble*q8 pairs)
     */
    __m128i madd_lo = _mm_maddubs_epi16(q4_lo, q8v0);
    __m128i madd_hi = _mm_maddubs_epi16(q4_hi, q8v1);

    /* Horizontal sum of madd_lo and madd_hi: each has 8 i16 values
     * Use madd with ones to get 4 i32, then hadd */
    __m128i ones = _mm_set1_epi16(1);
    __m128i s0 = _mm_madd_epi16(madd_lo, ones);
    __m128i s1 = _mm_madd_epi16(madd_hi, ones);
    __m128i s01 = _mm_hadd_epi32(s0, s1);
    __m128i s_all = _mm_hadd_epi32(s01, _mm_setzero_si128());
    int32_t acc = _mm_cvtsi128_si32(s_all) + _mm_extract_epi32(s_all, 1);

    /* Sum of all q8 values for the -8 offset correction */
    /* Extend int8 → int16 → int32, then hadd sum */
    __m128i q8v0_16lo = _mm_cvtepi8_epi16(q8v0);
    __m128i q8v0_16hi = _mm_cvtepi8_epi16(_mm_shuffle_epi32(q8v0, 0x4e));
    __m128i q8v1_16lo = _mm_cvtepi8_epi16(q8v1);
    __m128i q8v1_16hi = _mm_cvtepi8_epi16(_mm_shuffle_epi32(q8v1, 0x4e));
    /* Sum all 4 × 8 = 32 int8 values as int16, then hadd */
    __m128i q8_s0 = _mm_hadd_epi16(q8v0_16lo, q8v0_16hi);
    __m128i q8_s1 = _mm_hadd_epi16(q8v1_16lo, q8v1_16hi);
    __m128i q8_sum16 = _mm_hadd_epi16(q8_s0, q8_s1);
    __m128i q8_sum32 = _mm_cvtepi16_epi32(q8_sum16);
    __m128i q8_sum32b = _mm_cvtepi16_epi32(_mm_shuffle_epi32(q8_sum16, 0x4e));
    __m128i q8_total = _mm_hadd_epi32(q8_sum32, q8_sum32b);
    q8_total = _mm_hadd_epi32(q8_total, _mm_setzero_si128());
    int32_t q8_sum = _mm_cvtsi128_si32(q8_total) + _mm_extract_epi32(q8_total, 1);

    return d4 * d8 * ((float)acc - 8.0f * (float)q8_sum);
}

/* Q4_0 × Q8_0 matmul — quantized activation path
 * Computes W[Q4_0] @ x_quantized[Q8_0] using packed integer dot products
 */
void q4_0_q8_0_matmul_omp(const uint8_t *restrict W, const uint8_t *restrict x_q8,
                            float *restrict out, int n_rows, int n_cols,
                            int n_blocks_x) {
    /* n_blocks_x = n_cols / 32 = number of Q8_0 blocks in x
     * Each weight row has n_cols/32 Q4_0 blocks
     * Each activation has n_blocks_x Q8_0 blocks */
    int bpr = n_cols / 32;  /* blocks per row for Q4_0 (32 values per block) */

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        const uint8_t *row = W + (size_t)r * bpr * Q4_0_BS;
        for (int b = 0; b < bpr; b++) {
            total += dot_q4_0_q8_0(row + (size_t)b * Q4_0_BS,
                                     x_q8 + (size_t)b * Q8_0_BS);
        }
        out[r] = total;
    }
}

/* Batch QKV with Q8_0 quantized activation — eliminates 3 dequantize-to-float steps */
void batch_qkv_q8_omp(const uint8_t *restrict Wq, const uint8_t *restrict Wk,
                        const uint8_t *restrict Wv,
                        const uint8_t *restrict x_q8,
                        float *restrict out_q, float *restrict out_k, float *restrict out_v,
                        int nq, int nk, int nv, int n_cols,
                        int qtype_q, int qtype_k, int qtype_v,
                        int n_blocks_x) {
    /* For now, only Q4_0 × Q8_0 is implemented; K-quant types fall back to
     * the float activation path via matmul_row_range */
    int total_rows = nq + nk + nv;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_rows; i++) {
        if (i < nq) {
            if (qtype_q == 2) {
                /* Q4_0 × Q8_0 fast path */
                int bpr = nq / nq * (n_cols / 32);  /* just n_cols/32 */
                float total = 0.0f;
                const uint8_t *row = Wq + (size_t)i * (n_cols/32) * Q4_0_BS;
                for (int b = 0; b < n_cols/32; b++) {
                    total += dot_q4_0_q8_0(row + (size_t)b * Q4_0_BS,
                                             x_q8 + (size_t)b * Q8_0_BS);
                }
                out_q[i] = total;
            } else {
                /* Fallback: need float activation */
                /* This shouldn't be called for non-Q4_0 types */
                out_q[i] = 0.0f;
            }
        } else if (i < nq + nk) {
            int r = i - nq;
            if (qtype_k == 2) {
                float total = 0.0f;
                const uint8_t *row = Wk + (size_t)r * (n_cols/32) * Q4_0_BS;
                for (int b = 0; b < n_cols/32; b++) {
                    total += dot_q4_0_q8_0(row + (size_t)b * Q4_0_BS,
                                             x_q8 + (size_t)b * Q8_0_BS);
                }
                out_k[r] = total;
            } else {
                out_k[r] = 0.0f;
            }
        } else {
            int r = i - nq - nk;
            if (qtype_v == 2) {
                float total = 0.0f;
                const uint8_t *row = Wv + (size_t)r * (n_cols/32) * Q4_0_BS;
                for (int b = 0; b < n_cols/32; b++) {
                    total += dot_q4_0_q8_0(row + (size_t)b * Q4_0_BS,
                                             x_q8 + (size_t)b * Q8_0_BS);
                }
                out_v[r] = total;
            } else {
                out_v[r] = 0.0f;
            }
        }
    }
}

/* Batch Gate+Up with Q8_0 quantized activation */
void batch_gate_up_q8_omp(const uint8_t *restrict Wg, const uint8_t *restrict Wu,
                           const uint8_t *restrict x_q8,
                           float *restrict out_gate, float *restrict out_up,
                           int ng, int nu, int n_cols,
                           int qtype_g, int qtype_u,
                           int n_blocks_x) {
    int total_rows = ng + nu;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_rows; i++) {
        if (i < ng) {
            if (qtype_g == 2) {
                float total = 0.0f;
                const uint8_t *row = Wg + (size_t)i * (n_cols/32) * Q4_0_BS;
                for (int b = 0; b < n_cols/32; b++) {
                    total += dot_q4_0_q8_0(row + (size_t)b * Q4_0_BS,
                                             x_q8 + (size_t)b * Q8_0_BS);
                }
                out_gate[i] = total;
            } else {
                out_gate[i] = 0.0f;
            }
        } else {
            int r = i - ng;
            if (qtype_u == 2) {
                float total = 0.0f;
                const uint8_t *row = Wu + (size_t)r * (n_cols/32) * Q4_0_BS;
                for (int b = 0; b < n_cols/32; b++) {
                    total += dot_q4_0_q8_0(row + (size_t)b * Q4_0_BS,
                                             x_q8 + (size_t)b * Q8_0_BS);
                }
                out_up[r] = total;
            } else {
                out_up[r] = 0.0f;
            }
        }
    }
}
/* ═════════════════════════════════════════════════════════════════════════
 * Q4_K × Q8_0 QUANTIZED-ACTIVATION DOT — AVX2 vectorized
 *
 * Uses _mm_maddubs_epi16 for packed integer dot products.
 * 8 Q8_0 blocks × 32 values each, matching Q4_K's 8 sub-groups.
 * ═══════════════════════════════════════════════════════════════════════ */
static inline float dot_q4_k_q8_0(const uint8_t *restrict qk, const uint8_t *restrict q8) {
    float d   = f16_to_f32(*(const uint16_t*)qk);
    float min = f16_to_f32(*(const uint16_t*)(qk + 2));
    const uint8_t *scales = qk + 4;
    const uint8_t *qs_nib = qk + 16;

    /* Pre-extract 8 scale+min pairs using llama.cpp's bit-unpacking */
    uint32_t utmp[4];
    memcpy(utmp, scales, 12);
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    /* 8 int16 scales in low 128 bits */
    __m128i scales128 = _mm_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
    /* Duplicate to both AVX lanes */
    __m256i scales_v = _mm256_insertf128_si256(_mm256_castsi128_si256(scales128), scales128, 1);

    /* Scale shuffle masks (from llama.cpp) for sub-block 2*j+0 (lo nibbles) */
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    __m256i mask_lo = _mm256_set1_epi8(0x0F);
    __m256i ones16 = _mm256_set1_epi16(1);
    __m256i sumi = _mm256_setzero_si256();
    float sum_min = 0.0f;

    for (int i = 0; i < 8; i++) {
        int grp = i >> 1;
        int sub = i & 1;

        const uint8_t *q8b = q8 + (size_t)i * Q8_0_BS;
        float d8 = f16_to_f32(*(const uint16_t*)q8b);
        const int8_t *q8qs = (const int8_t *)(q8b + 2);

        const uint8_t *nib = qs_nib + (size_t)grp * 32;

        /* Load 32 nibbles, extract lo or hi */
        __m256i nv = _mm256_loadu_si256((const __m256i*)nib);
        nv = (sub == 0)
            ? _mm256_and_si256(nv, mask_lo)
            : _mm256_and_si256(_mm256_srli_epi16(nv, 4), mask_lo);

        /* Load 32 Q8_0 int8 values */
        __m256i q8v = _mm256_loadu_si256((const __m256i*)q8qs);

        /* maddubs: unsigned(nibble) × signed(q8) → 16 int16 pair-sums */
        __m256i p16 = _mm256_maddubs_epi16(nv, q8v);

        /* Apply scale using shuffle: sub-block 2*grp+sub picks scale[2*grp+sub] */
        __m256i scl = _mm256_loadu_si256((const __m256i*)k_shuffle + (2*grp + sub));
        p16 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales_v, scl), p16);

        sumi = _mm256_add_epi32(sumi, p16);

        /* Sum of Q8 int8 for min*Σ(q8) correction */
        __m256i e_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8v));
        __m256i e_hi = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(q8v, 1));
        __m128i q8ps128 = _mm_hadd_epi32(
            _mm_hadd_epi32(
                _mm256_castsi256_si128(_mm256_add_epi32(
                    _mm256_madd_epi16(e_lo, ones16),
                    _mm256_madd_epi16(e_hi, ones16))),
                _mm_setzero_si128()),
            _mm_setzero_si128());
        int32_t q8_total = _mm_cvtsi128_si32(q8ps128);
        sum_min += min * d8 * (float)q8_total;
    }

    /* Final horizontal reduction of sumi */
    __m128i lo = _mm256_castsi256_si128(sumi);
    __m128i hi = _mm256_extractf128_si256(sumi, 1);
    __m128i h = _mm_hadd_epi32(lo, hi);
    h = _mm_hadd_epi32(h, h);
    int32_t int_sum = _mm_cvtsi128_si32(h);

    return d * (float)int_sum - sum_min;
}

/* Q4_K matrix × Q8_0 activation row-dot */
float q4_k_row_dot_q8(const uint8_t *restrict W, const uint8_t *restrict x_q8,
                               int n_cols, int row) {
    int nb = n_cols / QK_K;
    const uint8_t *row_ptr = W + (size_t)row * nb * Q4_K_BS;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        sum += dot_q4_k_q8_0(row_ptr + (size_t)b * Q4_K_BS,
                              x_q8 + (size_t)b * 8 * Q8_0_BS);
    }
    return sum;
}

/* OMP Q4_K × Q8_0 matmul */
void q4_k_q8_0_matmul_omp(const uint8_t *restrict W, const uint8_t *restrict x_q8,
                           float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        out[r] = q4_k_row_dot_q8(W, x_q8, n_cols, r);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q6_K AVX2 DEQUANT + FMA — vectorized inner loop
 *
 * Q6_K layout per 256-value super-block:
 *   ql[128]:  nibbles for lower 4 bits
 *   qh[64]:   upper 2 bits (4 packed per byte)
 *   scales[16]: signed 6-bit per-sub-block scales
 *   d[2]: fp16 super-block scale
 *
 * AVX2 processes the 4 quadrants in parallel, 8 values at a time per quadrant.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Dequant 8 Q6_K values from one quadrant into 8 floats.
 * ql_ptr: points to ql[l] for this quadrant
 * qh_ptr: points to qh[l] for this quadrant
 * shift: bit shift for qh (0, 2, 4, or 6 for q1,q2,q3,q4)
 * mask: bit mask for qh (0x03 for q1/q2, 0x30 for q3/q4 with special handling)
 * Uses SSE for byte processing, then extends to AVX2 floats. */
static inline __m256 q6k_dequant_8(const uint8_t *ql_ptr, const uint8_t *qh_ptr, int shift) {
    /* Load 8 bytes from each source */
    __m128i ql8 = _mm_loadl_epi64((const __m128i*)(ql_ptr));
    __m128i qh8 = _mm_loadl_epi64((const __m128i*)(qh_ptr));

    /* Extract low nibble: ql & 0x0F */
    __m128i ql_nib = _mm_and_si128(ql8, _mm_set1_epi8(0x0F));

    /* Extract upper 2 bits from qh at the right shift position */
    __m128i qh_shifted;
    if (shift == 0) {
        qh_shifted = _mm_and_si128(qh8, _mm_set1_epi8(0x03));
    } else if (shift == 2) {
        qh_shifted = _mm_and_si128(_mm_srli_epi16(qh8, 2), _mm_set1_epi8(0x03));
    } else if (shift == 4) {
        qh_shifted = _mm_and_si128(_mm_srli_epi16(qh8, 4), _mm_set1_epi8(0x03));
    } else { /* shift == 6 */
        qh_shifted = _mm_and_si128(_mm_srli_epi16(qh8, 6), _mm_set1_epi8(0x03));
    }
    qh_shifted = _mm_slli_epi16(qh_shifted, 4); /* shift upper bits to position 4 */

    /* Combine: (ql & 0x0F) | ((qh >> shift) & 0x03) << 4 */
    __m128i combined = _mm_or_si128(ql_nib, qh_shifted);

    /* Subtract 32 to get signed -32..31 */
    __m128i q6_s8 = _mm_sub_epi8(combined, _mm_set1_epi8(32));

    /* Sign-extend: int8 → int16 → int32 → float via SSE then AVX2 */
    /* cvtepi8_epi16 reads LOW 8 bytes → 8 int16. No data in upper 8 bytes. */
    __m128i q6_16 = _mm_cvtepi8_epi16(q6_s8);     /* 8 int8 → 8 int16, all valid */
    __m128i q6_32a = _mm_cvtepi16_epi32(q6_16);    /* LOW 4 int16 → 4 int32 */
    /* Shuffle q6_16 (all valid data), then read LOW 4 int16 = original's HIGH 4 */
    __m128i q6_32b = _mm_cvtepi16_epi32(_mm_shuffle_epi32(q6_16, 0x4e));
    return _mm256_cvtepi32_ps(_mm256_set_m128i(q6_32b, q6_32a));
}

/* Process one Q6_K super-block (256 values) with AVX2 dequant + FMA.
 * Returns the dot product: Σ d * sc[i] * q6[i] * x[i]
 *
 * v2 optimization: accumulate per-quadrant sums in 4 separate __m256
 * accumulators, apply per-quadrant scales at the end, then do a single
 * horizontal sum per sub-block (4 hsums/super-block vs 32 before). */
static inline float dot_q6_k_f32_avx2(const uint8_t *restrict blk,
                                       const float *restrict x, int x_off) {
    float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
    const int8_t *sc = (const int8_t *)(blk + 192);
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;

    float block_sum = 0.0f;

    /* Two halves of 128 values each */
    for (int half = 0; half < 2; half++) {
        int ql_h = half * 64;   /* ql offset: 0 or 64 */
        int qh_h = half * 32;   /* qh offset: 0 or 32 */
        int sc_h = half * 8;    /* scale offset: 0 or 8 */

        /* Two sub-blocks per half (l=0..15 and l=16..31) */
        for (int sub = 0; sub < 2; sub++) {
            /* Per-quadrant scales for this sub-block */
            float ds_q1 = d * (float)sc[sc_h + sub + 0];
            float ds_q2 = d * (float)sc[sc_h + sub + 2];
            float ds_q3 = d * (float)sc[sc_h + sub + 4];
            float ds_q4 = d * (float)sc[sc_h + sub + 6];

            __m256 ds1 = _mm256_set1_ps(ds_q1);
            __m256 ds2 = _mm256_set1_ps(ds_q2);
            __m256 ds3 = _mm256_set1_ps(ds_q3);
            __m256 ds4 = _mm256_set1_ps(ds_q4);

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();

            /* Process 8 values at a time (AVX2 width), 2 iterations for 16 ql */
            for (int inner = 0; inner < 2; inner++) {
                int l = sub * 16 + inner * 8;

                /* 4 quadrants in parallel */
                __m256 q1f = q6k_dequant_8(ql + ql_h + l, qh + qh_h + l, 0);
                __m256 q2f = q6k_dequant_8(ql + ql_h + l + 32, qh + qh_h + l, 2);
                __m256 q3f = q6k_dequant_8(ql + ql_h + l, qh + qh_h + l, 4);
                __m256 q4f = q6k_dequant_8(ql + ql_h + l + 32, qh + qh_h + l, 6);

                /* Load x (quadrants are interleaved at +0, +32, +64, +96) */
                int base = x_off + half * 128 + l;
                __m256 x1 = _mm256_loadu_ps(x + base);
                __m256 x2 = _mm256_loadu_ps(x + base + 32);
                __m256 x3 = _mm256_loadu_ps(x + base + 64);
                __m256 x4 = _mm256_loadu_ps(x + base + 96);

                /* Accumulate per-quadrant (no hsum yet) */
                acc1 = _mm256_fmadd_ps(q1f, x1, acc1);
                acc2 = _mm256_fmadd_ps(q2f, x2, acc2);
                acc3 = _mm256_fmadd_ps(q3f, x3, acc3);
                acc4 = _mm256_fmadd_ps(q4f, x4, acc4);
            }

            /* Apply per-quadrant scales and combine into one sum */
            __m256 sum = _mm256_mul_ps(acc1, ds1);
            sum = _mm256_fmadd_ps(acc2, ds2, sum);
            sum = _mm256_fmadd_ps(acc3, ds3, sum);
            sum = _mm256_fmadd_ps(acc4, ds4, sum);

            /* Single horizontal sum for this sub-block (64 values) */
            __m256 h = _mm256_hadd_ps(sum, _mm256_permute2f128_ps(sum, sum, 1));
            h = _mm256_hadd_ps(h, h);
            h = _mm256_hadd_ps(h, h);
            block_sum += _mm256_cvtss_f32(h);
        }
    }
    return block_sum;
}

/* Q6_K row dot using AVX2 dequant + FMA */
float q6_k_row_dot_avx2(const uint8_t *restrict W, const float *restrict x,
                                int n_cols, int row) {
    int nb = n_cols / QK_K;
    const uint8_t *row_ptr = W + (size_t)row * nb * Q6_K_BS;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        sum += dot_q6_k_f32_avx2(row_ptr + (size_t)b * Q6_K_BS, x, b * QK_K);
    }
    return sum;
}

/* OMP Q6_K matmul using AVX2 dequant + FMA */
void q6_k_matmul_avx2_omp(const uint8_t *restrict W, const float *restrict x,
                           float *restrict out, int n_rows, int n_cols) {
    int nb = n_cols / QK_K;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        out[r] = q6_k_row_dot_avx2(W, x, n_cols, r);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * MOE FUSED KERNEL — AVX2-accelerated
 *
 * Phase 1: Quantize x_norm to Q8_0 once for Q4_K gate/up matmuls
 * Phase 2: Gate + Up using Q4_K × Q8_0 quantized-activation path
 * Phase 3: Down using Q6_K AVX2 dequant + FMA
 * ═════════════════════════════════════════════════════════════════════════ */

/* ─── Q4_K row-dot with float activation — AVX2 vectorized ───
 *
 * Uses the identity: sum((d1*nibble - m1)*x) = d1*sum(nibble*x) - m1*sum(x)
 * This lets us compute two accumulators (nibble*x and x) with AVX2,
 * then apply d1/m1 once at the end per 32-value group.
 */
float q4_k_row_dot_avx2(const uint8_t *restrict W, const float *restrict x,
                                 int n_cols, int row) {
    int nb = n_cols / QK_K;
    const uint8_t *row_ptr = W + (size_t)row * nb * Q4_K_BS;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row_ptr + (size_t)b * Q4_K_BS;
        float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t *scales = blk + 4;
        const uint8_t *q = blk + 16;
        int x_off = b * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * (float)sc;  float m1 = min * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * (float)sc;  float m2 = min * (float)m;

            /* Process 32 lo-nibble values: sum += d1*nibble*x - m1*x */
            __m256 acc_nx = _mm256_setzero_ps();
            __m256 acc_x  = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 16) {
                /* Load 16 bytes, extract lo nibbles (0-15) */
                __m128i q16 = _mm_loadu_si128((const __m128i*)(q + l));
                __m128i nib = _mm_and_si128(q16, _mm_set1_epi8(0x0F));
                __m128i n16a = _mm_cvtepu8_epi16(nib);
                __m128i n16b = _mm_cvtepu8_epi16(_mm_shuffle_epi32(nib, 0x4e));
                __m256 nf_a = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    _mm_cvtepi16_epi32(_mm_shuffle_epi32(n16a, 0x4e)),
                    _mm_cvtepi16_epi32(n16a)));
                __m256 nf_b = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    _mm_cvtepi16_epi32(_mm_shuffle_epi32(n16b, 0x4e)),
                    _mm_cvtepi16_epi32(n16b)));
                __m256 xa = _mm256_loadu_ps(x + x_off + j + l);
                __m256 xb = _mm256_loadu_ps(x + x_off + j + l + 8);
                acc_nx = _mm256_fmadd_ps(nf_a, xa, acc_nx);
                acc_nx = _mm256_fmadd_ps(nf_b, xb, acc_nx);
                acc_x  = _mm256_add_ps(xa, acc_x);
                acc_x  = _mm256_add_ps(xb, acc_x);
            }
            __m256 h_nx = _mm256_hadd_ps(acc_nx, _mm256_permute2f128_ps(acc_nx, acc_nx, 1));
            h_nx = _mm256_hadd_ps(h_nx, h_nx); h_nx = _mm256_hadd_ps(h_nx, h_nx);
            __m256 h_x  = _mm256_hadd_ps(acc_x,  _mm256_permute2f128_ps(acc_x,  acc_x,  1));
            h_x  = _mm256_hadd_ps(h_x,  h_x);  h_x  = _mm256_hadd_ps(h_x,  h_x);
            float nx_sum = _mm256_cvtss_f32(h_nx);
            float  x_sum = _mm256_cvtss_f32(h_x);
            sum += d1 * nx_sum - m1 * x_sum;

            /* Process 32 hi-nibble values: sum += d2*nibble*x - m2*x */
            acc_nx = _mm256_setzero_ps();
            acc_x  = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 16) {
                __m128i q16 = _mm_loadu_si128((const __m128i*)(q + l));
                __m128i nib = _mm_and_si128(_mm_srli_epi16(q16, 4), _mm_set1_epi8(0x0F));
                __m128i n16a = _mm_cvtepu8_epi16(nib);
                __m128i n16b = _mm_cvtepu8_epi16(_mm_shuffle_epi32(nib, 0x4e));
                __m256 nf_a = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    _mm_cvtepi16_epi32(_mm_shuffle_epi32(n16a, 0x4e)),
                    _mm_cvtepi16_epi32(n16a)));
                __m256 nf_b = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    _mm_cvtepi16_epi32(_mm_shuffle_epi32(n16b, 0x4e)),
                    _mm_cvtepi16_epi32(n16b)));
                __m256 xa = _mm256_loadu_ps(x + x_off + j + 32 + l);
                __m256 xb = _mm256_loadu_ps(x + x_off + j + 32 + l + 8);
                acc_nx = _mm256_fmadd_ps(nf_a, xa, acc_nx);
                acc_nx = _mm256_fmadd_ps(nf_b, xb, acc_nx);
                acc_x  = _mm256_add_ps(xa, acc_x);
                acc_x  = _mm256_add_ps(xb, acc_x);
            }
            h_nx = _mm256_hadd_ps(acc_nx, _mm256_permute2f128_ps(acc_nx, acc_nx, 1));
            h_nx = _mm256_hadd_ps(h_nx, h_nx); h_nx = _mm256_hadd_ps(h_nx, h_nx);
            h_x  = _mm256_hadd_ps(acc_x,  _mm256_permute2f128_ps(acc_x,  acc_x,  1));
            h_x  = _mm256_hadd_ps(h_x,  h_x);  h_x  = _mm256_hadd_ps(h_x,  h_x);
            nx_sum = _mm256_cvtss_f32(h_nx);
             x_sum = _mm256_cvtss_f32(h_x);
            sum += d2 * nx_sum - m2 * x_sum;

            q += 32; is += 2;
        }
    }
    return sum;
}


void moe_forward_omp(
    const uint8_t** gate_raw, const uint8_t** up_raw, const uint8_t** down_raw,
    const float* x_norm, int n_ff_expert, int n_embd,
    int qt_gate, int qt_up, int qt_down,
    const int* top_indices, const float* top_weights, int top_k,
    float* combined,
    float* prealloc_buf, uint8_t* prealloc_q8
) {
    size_t exp_buf_sz = (size_t)top_k * n_ff_expert;
    float *all_bufs = prealloc_buf ? prealloc_buf : (float*)malloc(3 * exp_buf_sz * sizeof(float));
    float *gate_buf = all_bufs;
    float *up_buf   = all_bufs + exp_buf_sz;
    float *silu_buf = all_bufs + 2 * exp_buf_sz;

    if (!all_bufs) return;

    /* ── Quantize x_norm to Q8_0 once for all Q4_K/MXFP4 quantized-activation rows ── */
    int n_blocks_x = n_embd / 32;
    uint8_t x_q8_buf[4096 / 32 * Q8_0_BS];  /* stack alloc for up to 4096 dims */
    int q8_sz = (size_t)n_blocks_x * Q8_0_BS;
    uint8_t *x_q8 = q8_sz <= sizeof(x_q8_buf) ? x_q8_buf : (uint8_t*)malloc(q8_sz);
    uint8_t *x_q8_alloced = (x_q8 != x_q8_buf) ? x_q8 : NULL;
    (void)prealloc_q8;
    quantize_row_q8_0(x_norm, x_q8, n_embd);

    int total_gate    = top_k * n_ff_expert;
    int total_down    = top_k * n_embd;

    /* ── Phase 1: Gate + Up ── */
    if (qt_gate == 39 && qt_up == 39) {
        /* MXFP4 fast path: fused OMP over ALL gate+up rows for all experts.
         * Single parallel region per layer avoids 8× OMP fork-join overhead. */
        int total = total_gate;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total * 2; i++) {
            int is_up = i >= total;
            int idx = is_up ? i - total : i;
            int ei = idx / n_ff_expert;
            int row = idx % n_ff_expert;
            int e = top_indices[ei];
            const uint8_t *W = is_up ? up_raw[e] : gate_raw[e];
            float *buf = is_up ? up_buf : gate_buf;
            buf[idx] = mxfp4_row_dot_q8(W, x_q8, n_embd, row);
        }
    } else {
        /* Per-row dispatch for other quant types */
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < total_gate; i++) {
            int ei = i / n_ff_expert;
            int row = i % n_ff_expert;
            int e = top_indices[ei];
            float gv, uv;
            if (qt_gate == 12) {
                gv = q4_k_row_dot_q8(gate_raw[e], x_q8, n_embd, row);
            } else if (qt_gate == 14) {
                gv = q6_k_row_dot_avx2(gate_raw[e], x_norm, n_embd, row);
            } else if (qt_gate == 2) {
                int bpr = n_embd / 32;
                gv = q4_0_row_dot(gate_raw[e], x_norm, bpr, row);
            } else if (qt_gate == 3) {
                int bpr = n_embd / 32;
                gv = q4_1_row_dot(gate_raw[e], x_norm, bpr, row);
            } else {
                gv = 0.0f;
            }
            if (qt_up == 12) {
                uv = q4_k_row_dot_q8(up_raw[e], x_q8, n_embd, row);
            } else if (qt_up == 14) {
                uv = q6_k_row_dot_avx2(up_raw[e], x_norm, n_embd, row);
            } else if (qt_up == 2) {
                int bpr = n_embd / 32;
                uv = q4_0_row_dot(up_raw[e], x_norm, bpr, row);
            } else if (qt_up == 3) {
                int bpr = n_embd / 32;
                uv = q4_1_row_dot(up_raw[e], x_norm, bpr, row);
            } else {
                uv = 0.0f;
            }
            gate_buf[i] = gv;
            up_buf[i] = uv;
        }
    }

    /* ── Phase 1.5: SiLU(gate) * up (parallel) ── */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < top_k * n_ff_expert; i++) {
        float gv = gate_buf[i];
        silu_buf[i] = (gv / (1.0f + expf(-gv))) * up_buf[i];
    }

    /* ── Phase 2: Down matmuls + weighted accumulation ── */
    memset(combined, 0, n_embd * sizeof(float));
    if (qt_down == 39) {
        /* MXFP4 down path: fused OMP, one region per layer */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total_down; i++) {
            int exp_idx = i / n_embd;
            int row = i % n_embd;
            int e = top_indices[exp_idx];
            const float *s = silu_buf + exp_idx * n_ff_expert;
            float dot = mxfp4_row_dot_f32(down_raw[e], s, n_ff_expert, row);
            #pragma omp atomic
            combined[row] += top_weights[exp_idx] * dot;
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < total_down; i++) {
            int exp_idx = i / n_embd;
            int row     = i % n_embd;
            int e = top_indices[exp_idx];
            const float *s = silu_buf + exp_idx * n_ff_expert;
            float dot;
            if (qt_down == 14) {
                dot = q6_k_row_dot_avx2(down_raw[e], s, n_ff_expert, row);
            } else if (qt_down == 12) {
                dot = q4_k_row_dot_avx2(down_raw[e], s, n_ff_expert, row);
            } else if (qt_down == 2) {
                int bpr = n_ff_expert / 32;
                dot = q4_0_row_dot(down_raw[e], s, bpr, row);
            } else {
                dot = 0.0f;
            }
            #pragma omp atomic
            combined[row] += top_weights[exp_idx] * dot;
        }
    }

    if (!prealloc_buf) free(all_bufs);
    if (x_q8_alloced)  free(x_q8_alloced);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Gemma4 fused matmuls — all 6 projections in one OMP region with shared Q8_0
 */
void gemma4_batch_matmuls(
    const float *x_norm,
    float *q, float *k, float *v,
    int N, int nq, int nk, int nv,
    const uint8_t *w_q, const uint8_t *w_k, const uint8_t *w_v,
    int qt_q, int qt_k, int qt_v
) {
    int nb = N / 32;
    uint8_t x_q8[4096 / 32 * Q8_0_BS];
    quantize_row_q8_0(x_norm, x_q8, N);
    int total = nq + nk + nv;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++) {
        if (i < nq) {
            q[i] = q4_k_row_dot_q8(w_q, x_q8, N, i);
        } else if (i < nq + nk) {
            k[i - nq] = q4_k_row_dot_q8(w_k, x_q8, N, i - nq);
        } else {
            v[i - nq - nk] = q6_k_row_dot_avx2(w_v, x_norm, N, i - nq - nk);
        }
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Q4_K matmul replacement — auto-quantizes activation to Q8_0
 * Backward-compatible replacement for the old scalar q4_k_matmul_omp.
 * ═══════════════════════════════════════════════════════════════════════ */
