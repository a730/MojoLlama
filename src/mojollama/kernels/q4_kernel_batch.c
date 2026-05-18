/* Batched Q4_X matmul — 7 projections in one OpenMP call.
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC -o q4_kernel_batch.so q4_kernel_batch.c
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

#define MAX_PROJ 7

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

static void decode_q4_0(const uint8_t* bp, __m256 v[4]) {
    float scale = f16_to_f32(bp[0] | ((uint16_t)bp[1] << 8));
    __m256 s = _mm256_set1_ps(scale);
    __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(15));
    __m128i i0 = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
    __m128i i1 = _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), _mm_set1_epi8(8));
    __m128i e00 = _mm_cvtepi8_epi16(i0), e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(i0, 0x4e));
    __m128i e10 = _mm_cvtepi8_epi16(i1), e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(i1, 0x4e));
    v[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))),s);
    v[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))),s);
    v[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))),s);
    v[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))),s);
}

static void decode_q4_1(const uint8_t* bp, __m256 v[4]) {
    float dmin = f16_to_f32(bp[0] | ((uint16_t)bp[1] << 8));
    float dmax = f16_to_f32(bp[2] | ((uint16_t)bp[3] << 8));
    float diff = (dmax - dmin) / 15.0f;
    __m256 dm = _mm256_set1_ps(dmin), df = _mm256_set1_ps(diff);
    __m128i nb = _mm_loadu_si128((__m128i*)(bp + 4));
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(15));
    __m128i i0 = _mm_unpacklo_epi8(lo, hi), i1 = _mm_unpackhi_epi8(lo, hi);
    __m128i e00 = _mm_cvtepu8_epi16(i0), e01 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(i0,0x4e));
    __m128i e10 = _mm_cvtepu8_epi16(i1), e11 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(i1,0x4e));
    v[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))),df,dm);
    v[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))),df,dm);
    v[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))),df,dm);
    v[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))),df,dm);
}

/* Process one row: compute all projections that have data for this row */
static void process_row(const uint8_t* w[MAX_PROJ], const float* x, float* out[MAX_PROJ],
                         int nrows[MAX_PROJ], int ncols[MAX_PROJ], int ts[MAX_PROJ],
                         int n_proj, int row) {
    int bpr = ncols[0] / 32;  // all projections share same n_cols pattern
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

/* Batched: 7 projections in one call. OpenMP parallelizes over rows.
 * w[p], out[p], nrows[p], ncols[p], ts[p] for each projection.
 */
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
