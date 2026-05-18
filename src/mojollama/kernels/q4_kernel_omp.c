/* Q4_0/Q4_1 matmul AVX2+OpenMP kernel.
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o q4_kernel_omp.so q4_kernel_omp.c
 * Single call, internally parallel via OpenMP. No Python threading overhead.
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

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
    v[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00, 0x4e)), _mm_cvtepi16_epi32(e00))), s);
    v[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01, 0x4e)), _mm_cvtepi16_epi32(e01))), s);
    v[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10, 0x4e)), _mm_cvtepi16_epi32(e10))), s);
    v[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11, 0x4e)), _mm_cvtepi16_epi32(e11))), s);
}

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
    v[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00, 0x4e)), _mm_cvtepi16_epi32(e00))), diff_v, dmin_v);
    v[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01, 0x4e)), _mm_cvtepi16_epi32(e01))), diff_v, dmin_v);
    v[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10, 0x4e)), _mm_cvtepi16_epi32(e10))), diff_v, dmin_v);
    v[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11, 0x4e)), _mm_cvtepi16_epi32(e11))), diff_v, dmin_v);
}

/* Full matmul with OpenMP parallel + AVX2. Single call, internally threaded. */
void q4_matmul_omp(const uint8_t* w, const float* x, float* out,
                    int n_rows, int n_cols, int type_size) {
    int bpr = n_cols / 32;
    void (*decode)(const uint8_t*, __m256*) = (type_size == 20) ? decode_q4_1 : decode_q4_0;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n_rows; r++) {
        float total = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t* bp = w + (r * bpr + blk) * type_size;
            int io = blk * 32;
            __m256 x0 = _mm256_loadu_ps(x + io), x1 = _mm256_loadu_ps(x + io + 8);
            __m256 x2 = _mm256_loadu_ps(x + io + 16), x3 = _mm256_loadu_ps(x + io + 24);
            __m256 v[4]; decode(bp, v);
            __m256 a = _mm256_mul_ps(v[0], x0);
            a = _mm256_fmadd_ps(v[1], x1, a);
            a = _mm256_fmadd_ps(v[2], x2, a);
            a = _mm256_fmadd_ps(v[3], x3, a);
            __m256 h = _mm256_hadd_ps(a, _mm256_permute2f128_ps(a, a, 1));
            h = _mm256_hadd_ps(h, h); h = _mm256_hadd_ps(h, h);
            total += _mm256_cvtss_f32(h);
        }
        out[r] = total;
    }
}
