/* fused_layer.c — All per-layer Q4_0 matmuls in ONE OMP parallel region.
 *
 * Fuses 5 projections (Q, K, V, Gate, Up) that share the x_norm input
 * into a single OMP parallel for. This eliminates 1 OMP fork/join per layer
 * and maximizes cache reuse of the input activation.
 *
 * Plus: fused SiLU + gate*up (C SIMD) to avoid numpy overhead.
 *
 * Compile:
 *   gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *       -o fused_layer.so fused_layer.c -lm
 */
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define Q4_0_BS 18

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* Compute one row of Q4_0 matmul: W[row, :] · x[:] */
static inline float q4_0_dot_row(const uint8_t *row_base, const float *x, int n_cols) {
    int bpr = n_cols / 32;
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    for (int blk = 0; blk < bpr; blk++) {
        const uint8_t *bp = row_base + (size_t)blk * Q4_0_BS;
        float s = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
        __m256 scale = _mm256_set1_ps(s);
        __m128i nb = _mm_loadu_si128((const __m128i*)(bp + 2));
        __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(15));
        __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
        __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
        __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(lo_s)));
        __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_shuffle_epi32(lo_s, 0x4e))));
        __m256 v2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(hi_s)));
        __m256 v3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_shuffle_epi32(hi_s, 0x4e))));
        v0 = _mm256_mul_ps(v0, scale); v1 = _mm256_mul_ps(v1, scale);
        v2 = _mm256_mul_ps(v2, scale); v3 = _mm256_mul_ps(v3, scale);
        int off = blk * 32;
        sum0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(x + off), sum0);
        sum0 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8), sum0);
        sum1 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16), sum1);
        sum1 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24), sum1);
    }
    return hsum_ps(_mm256_add_ps(sum0, sum1));
}

/* Fused Q/K/V/Gate/Up matmuls in ONE OMP parallel region.
 * All 5 projections use the same input x (x_norm).
 * max_rows = max(nr_q, nr_k, nr_v, nr_gate, nr_up)
 * Threads iterate rows 0..max_rows-1; projections without that row are skipped.
 */
void fused_qkv_gate_up(const uint8_t *w_q, const uint8_t *w_k, const uint8_t *w_v,
                        const uint8_t *w_gate, const uint8_t *w_up,
                        const float *x,
                        float *out_q, float *out_k, float *out_v,
                        float *out_gate, float *out_up,
                        int nr_q, int nr_k, int nr_v,
                        int nr_gate, int nr_up,
                        int n_cols) {
    // Determine max row count
    int max_rows = nr_q;
    if (nr_k > max_rows) max_rows = nr_k;
    if (nr_v > max_rows) max_rows = nr_v;
    if (nr_gate > max_rows) max_rows = nr_gate;
    if (nr_up > max_rows) max_rows = nr_up;

    int bpr = n_cols / 32;
    size_t stride_q = (size_t)bpr * Q4_0_BS;
    size_t stride_k = (size_t)bpr * Q4_0_BS;
    size_t stride_v = (size_t)bpr * Q4_0_BS;
    size_t stride_g = (size_t)bpr * Q4_0_BS;
    size_t stride_u = (size_t)bpr * Q4_0_BS;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < max_rows; row++) {
        if (row < nr_q) {
            out_q[row] = q4_0_dot_row(w_q + (size_t)row * stride_q, x, n_cols);
        }
        if (row < nr_k) {
            out_k[row] = q4_0_dot_row(w_k + (size_t)row * stride_k, x, n_cols);
        }
        if (row < nr_v) {
            out_v[row] = q4_0_dot_row(w_v + (size_t)row * stride_v, x, n_cols);
        }
        if (row < nr_gate) {
            out_gate[row] = q4_0_dot_row(w_gate + (size_t)row * stride_g, x, n_cols);
        }
        if (row < nr_up) {
            out_up[row] = q4_0_dot_row(w_up + (size_t)row * stride_u, x, n_cols);
        }
    }
}
