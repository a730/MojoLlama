/* avx2_batch_kernels.c — Q4_0 batch kernels with inline AVX2 dequant+FMA.
 *
 * Replaces quant_kernels_omp.c's batch_qkv_omp and batch_gate_up_omp
 * with Q4_0-only versions that inline the AVX2 dequant+FMA loop
 * (no function pointers, no switch dispatch, no per-row function call).
 *
 * Expected: 1.3-1.5x faster on Q4_0 batch matmuls = ~60% of matmul time
 * → overall ~10-15% engine speedup → 82 → 90+ tok/s
 *
 * Compile:
 *   gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *       -o avx2_batch_kernels.so avx2_batch_kernels.c -lm
 */
#include <stdint.h>
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

/* Compute Q4_0 dot product for row r of weight matrix W against input x.
 * Inline AVX2 dequant+FMA — no function calls, no switch.
 */
static inline float q4_0_dot_inline(const uint8_t *W, const float *x,
                                     int n_cols, int row) {
    int bpr = n_cols / 32;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (int blk = 0; blk < bpr; blk++) {
        const uint8_t *bp = W + ((size_t)row * bpr + blk) * Q4_0_BS;
        float s = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
        __m256 scale = _mm256_set1_ps(s);
        __m128i nb = _mm_loadu_si128((const __m128i*)(bp + 2));
        __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(15));
        __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
        __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
        // Dequantize 32 values into 4 x __m256
        __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(lo_s)));
        __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_shuffle_epi32(lo_s, 0x4e))));
        __m256 v2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(hi_s)));
        __m256 v3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_shuffle_epi32(hi_s, 0x4e))));
        // Scale
        v0 = _mm256_mul_ps(v0, scale); v1 = _mm256_mul_ps(v1, scale);
        v2 = _mm256_mul_ps(v2, scale); v3 = _mm256_mul_ps(v3, scale);
        // FMA dot with input
        int off = blk * 32;
        sum0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(x + off), sum0);
        sum0 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8), sum0);
        sum1 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16), sum1);
        sum1 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24), sum1);
    }
    return hsum_ps(_mm256_add_ps(sum0, sum1));
}

/* Fused Q+K+V batch matmul — Q4_0 only, inline AVX2 dequant.
 * 
 * Processes all rows of Q, K, V projections in a single OMP parallel region.
 * Uses row-level dispatch based on accumulated row offsets.
 */
void batch_qkv_q4_0(const uint8_t *Wq, const uint8_t *Wk, const uint8_t *Wv,
                     const float *x,
                     float *out_q, float *out_k, float *out_v,
                     int nq, int nk, int nv, int n_cols) {
    int total = nq + nk + nv;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++) {
        if (i < nq) {
            out_q[i] = q4_0_dot_inline(Wq, x, n_cols, i);
        } else if (i < nq + nk) {
            int r = i - nq;
            out_k[r] = q4_0_dot_inline(Wk, x, n_cols, r);
        } else {
            int r = i - nq - nk;
            out_v[r] = q4_0_dot_inline(Wv, x, n_cols, r);
        }
    }
}

/* Fused Gate+Up batch matmul — Q4_0 only, inline AVX2 dequant. */
void batch_gate_up_q4_0(const uint8_t *Wg, const uint8_t *Wu,
                         const float *x,
                         float *out_gate, float *out_up,
                         int ng, int nu, int n_cols) {
    int total = ng + nu;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++) {
        if (i < ng) {
            out_gate[i] = q4_0_dot_inline(Wg, x, n_cols, i);
        } else {
            int r = i - ng;
            out_up[r] = q4_0_dot_inline(Wu, x, n_cols, r);
        }
    }
}
