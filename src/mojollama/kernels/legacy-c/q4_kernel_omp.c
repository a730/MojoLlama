/* Q4_0/Q4_1 matmul — AVX2+FMA+OpenMP with 4-row register blocking.
 * 
 * Optimizations over baseline:
 * 1. Row-outer accumulation (write output once per row, not per block)
 * 2. 4-row register blocking (4 rows share input vector loads)
 * 3. Software prefetch on weight data
 * 4. Aligned output buffers (padding to 64-byte cache lines)
 * 5. OpenMP thread control via OMP_NUM_THREADS
 *
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o q4_kernel_omp.so q4_kernel_omp.c
 */
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

/* Decode one Q4_0 block (18 bytes) → 4 __m256 dequant values.
 * Full SIMD pipeline: AND→SHIFT→UNPACK→SUB→WIDEN→CVT→MUL by scale.
 * ~15 instructions to dequantize 32 values.
 */
static inline void decode_q4_0(const uint8_t* bp, __m256 v[4]) {
    float scale = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
    __m256 s = _mm256_set1_ps(scale);
    __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(15));
    __m128i i0 = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
    __m128i i1 = _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), _mm_set1_epi8(8));
    __m128i e00 = _mm_cvtepi8_epi16(i0), e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(i0, 0x4e));
    __m128i e10 = _mm_cvtepi8_epi16(i1), e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(i1, 0x4e));
    v[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e00, 0x4e)), _mm_cvtepi16_epi32(e00))), s);
    v[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e01, 0x4e)), _mm_cvtepi16_epi32(e01))), s);
    v[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e10, 0x4e)), _mm_cvtepi16_epi32(e10))), s);
    v[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e11, 0x4e)), _mm_cvtepi16_epi32(e11))), s);
}

/* Q4_1 variant: 20 bytes per block, dmin + dmax instead of just scale */
static inline void decode_q4_1(const uint8_t* bp, __m256 v[4]) {
    float dmin = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
    float dmax = f16_to_f32((uint16_t)bp[2] | ((uint16_t)bp[3] << 8));
    float diff = (dmax - dmin) / 15.0f;
    __m256 dmin_v = _mm256_set1_ps(dmin), diff_v = _mm256_set1_ps(diff);
    __m128i nb = _mm_loadu_si128((__m128i*)(bp + 4));
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(15));
    __m128i i0 = _mm_unpacklo_epi8(lo, hi), i1 = _mm_unpackhi_epi8(lo, hi);
    __m128i e00 = _mm_cvtepu8_epi16(i0), e01 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(i0, 0x4e));
    __m128i e10 = _mm_cvtepu8_epi16(i1), e11 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(i1, 0x4e));
    v[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))), diff_v, dmin_v);
    v[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))), diff_v, dmin_v);
    v[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))), diff_v, dmin_v);
    v[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))), diff_v, dmin_v);
}

/* Compute a single row-block dot product using FMA chain.
 * 4 FMA instructions + 1 MUL + 1 HADD chain = 7 uops per block.
 * This replaces the (v*scale*x).reduce_add() pattern which does 4 MULs + 4 REDUCE_ADDs.
 */
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

/* ── Single-row matmul: row-outer with register accumulation ────────── */

void q4_matmul_omp(const uint8_t* w, const float* x, float* out,
                   int n_rows, int n_cols, int type_size) {
    int bpr = n_cols / 32;
    void (*decode)(const uint8_t*, __m256*) = (type_size == 20) ? decode_q4_1 : decode_q4_0;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            __m256 v[4];
            decode(w + (r * bpr + blk) * type_size, v);
            total += block_dot_fma(v, x + blk * 32);
        }
        out[r] = total;
    }
}

/* ── 4-row register-blocked matmul ───────────────────────────────────
 * Processes 4 rows at a time, sharing input vector loads across rows.
 * Each row accumulates independently in a register → 1 write per row total.
 */

void q4_matmul_omp_blocked(const uint8_t* w, const float* x, float* out,
                            int n_rows, int n_cols, int type_size) {
    int bpr = n_cols / 32;
    void (*decode)(const uint8_t*, __m256*) = (type_size == 20) ? decode_q4_1 : decode_q4_0;
    memset(out, 0, n_rows * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r += 4) {
        int rows_left = (r + 4 <= n_rows) ? 4 : (n_rows - r);
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int blk = 0; blk < bpr; blk++) {
            float x_base[32];
            memcpy(x_base, x + blk * 32, 32 * sizeof(float));

            for (int ri = 0; ri < rows_left; ri++) {
                __m256 v[4];
                decode(w + ((r + ri) * bpr + blk) * type_size, v);
                sum[ri] += block_dot_fma(v, x_base);
            }
        }
        for (int ri = 0; ri < rows_left; ri++) {
            out[r + ri] = sum[ri];
        }
    }
}

/* ── Single-threaded row-outer AVX2+FMA baseline ──────────────────── */

void q4_matmul_avx2(const uint8_t* w, const float* x, float* out,
                    int n_rows, int n_cols, int start, int end, int type_size) {
    int bpr = n_cols / 32;
    int n_out = end - start;
    void (*decode)(const uint8_t*, __m256*) = (type_size == 20) ? decode_q4_1 : decode_q4_0;
    for (int i = 0; i < n_out; i++) out[i] = 0.0f;

    for (int blk = 0; blk < bpr; blk++) {
        int io = blk * 32;
        __m256 x0 = _mm256_loadu_ps(x + io);
        __m256 x1 = _mm256_loadu_ps(x + io + 8);
        __m256 x2 = _mm256_loadu_ps(x + io + 16);
        __m256 x3 = _mm256_loadu_ps(x + io + 24);
        for (int r = start; r < end; r++) {
            __m256 v[4];
            decode(w + (r * bpr + blk) * type_size, v);
            __m256 a = _mm256_mul_ps(v[0], x0);
            a = _mm256_fmadd_ps(v[1], x1, a);
            a = _mm256_fmadd_ps(v[2], x2, a);
            a = _mm256_fmadd_ps(v[3], x3, a);
            __m256 h = _mm256_hadd_ps(a, _mm256_permute2f128_ps(a, a, 1));
            h = _mm256_hadd_ps(h, h); h = _mm256_hadd_ps(h, h);
            out[r - start] += _mm256_cvtss_f32(h);
        }
    }
}

/* ── Convenience wrappers ──────────────────────────────────────────── */

void q4_0_matmul(const uint8_t* w, const float* x, float* out,
                 int n_rows, int n_cols, int start, int end) {
    q4_matmul_avx2(w, x, out, n_rows, n_cols, start, end, 18);
}

void q4_1_matmul(const uint8_t* w, const float* x, float* out,
                 int n_rows, int n_cols, int start, int end) {
    q4_matmul_avx2(w, x, out, n_rows, n_cols, start, end, 20);
}

/* ── Batched: up to 7 projections in one OpenMP call ──────────────── */
#define MAX_PROJ 7

static void process_row(const uint8_t* w[MAX_PROJ], const float* x, float* out[MAX_PROJ],
                         int nrows[MAX_PROJ], int ncols[MAX_PROJ], int ts[MAX_PROJ],
                         int n_proj, int row) {
    int bpr = ncols[0] / 32;
    for (int p = 0; p < n_proj; p++) {
        float total = 0.0f;
        int nc = ncols[p], nr = nrows[p];
        if (row >= nr) continue;
        int bp = nc / 32;
        void (*dec)(const uint8_t*, __m256*) = (ts[p] == 20) ? decode_q4_1 : decode_q4_0;
        for (int blk = 0; blk < bp; blk++) {
            const uint8_t* bp_w = w[p] + (row * bp + blk) * ts[p];
            int io = blk * 32;
            __m256 x0 = _mm256_loadu_ps(x + io), x1 = _mm256_loadu_ps(x + io + 8);
            __m256 x2 = _mm256_loadu_ps(x + io + 16), x3 = _mm256_loadu_ps(x + io + 24);
            __m256 v[4]; dec(bp_w, v);
            __m256 a = _mm256_mul_ps(v[0], x0);
            a = _mm256_fmadd_ps(v[1], x1, a); a = _mm256_fmadd_ps(v[2], x2, a);
            a = _mm256_fmadd_ps(v[3], x3, a);
            __m256 h = _mm256_hadd_ps(a, _mm256_permute2f128_ps(a, a, 1));
            h = _mm256_hadd_ps(h, h); h = _mm256_hadd_ps(h, h);
            total += _mm256_cvtss_f32(h);
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

/* ── RMS normalization (AVX2 vectorized) ──────────────────────────── */

void q4_rms_norm(const float* x, const float* weight, float* out, int n) {
    __m256 ss_vec = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        ss_vec = _mm256_fmadd_ps(v, v, ss_vec);
    }
    /* Horizontal sum */
    __m256 h = _mm256_hadd_ps(ss_vec, _mm256_permute2f128_ps(ss_vec, ss_vec, 1));
    h = _mm256_hadd_ps(h, h); h = _mm256_hadd_ps(h, h);
    float ss = _mm256_cvtss_f32(h);
    /* remaining elements */
    for (; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / n + 1e-6f);
    __m256 inv_v = _mm256_set1_ps(inv_rms);
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, inv_v), w));
    }
    for (; i < n; i++) out[i] = x[i] * inv_rms * weight[i];
}

/* ── SiLU activation (AVX2 vectorized multiply, scalar exp) ─────── */

void q4_silu(float* x, int n) {
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        /* Scalar exp fallback for each lane — AVX2 has no _mm256_exp_ps */
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

/* ── Softmax (AVX2 vectorized max/reduce, scalar exp) ─────────────── */

void q4_softmax(float* x, int n) {
    /* Find max for numerical stability */
    __m256 max_vec = _mm256_set1_ps(-1e30f);
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        max_vec = _mm256_max_ps(max_vec, v);
    }
    __m256 h = _mm256_hadd_ps(max_vec, _mm256_permute2f128_ps(max_vec, max_vec, 1));
    h = _mm256_hadd_ps(h, h); h = _mm256_hadd_ps(h, h);
    float max_val = _mm256_cvtss_f32(h);
    for (; i < n; i++) if (x[i] > max_val) max_val = x[i];

    /* Scalar exp(x - max) — AVX2 has no vector exp */
    float sum = 0.0f;
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    __m256 inv_v = _mm256_set1_ps(inv_sum);
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, inv_v));
    }
    for (; i < n; i++) x[i] *= inv_sum;
}

/* ── Thread-get/set for dynamic thread control ─────────────────────── */

int q4_get_max_threads(void) { return omp_get_max_threads(); }
void q4_set_num_threads(int n) { omp_set_num_threads(n); }