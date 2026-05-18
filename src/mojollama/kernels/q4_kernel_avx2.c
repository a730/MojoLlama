/* Q4_0 and Q4_1 matmul AVX2 kernel.
 * Supports both quantization formats.
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -shared -fPIC -o q4_kernel_avx2.so q4_kernel_avx2.c
 */
#include <stdint.h>
#include <immintrin.h>

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

/* Decode one Q4_0 block (18 bytes) -> 4 F32x8 dequant values */
static inline void decode_q4_0(const uint8_t* bp, __m256 v[4]) {
    float scale = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
    __m256 s = _mm256_set1_ps(scale);
    __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(15));
    __m128i i0 = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
    __m128i i1 = _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), _mm_set1_epi8(8));
    __m128i e00 = _mm_cvtepi8_epi16(i0);
    __m128i e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(i0, 0x4e));
    __m128i e10 = _mm_cvtepi8_epi16(i1);
    __m128i e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(i1, 0x4e));
    v[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e00, 0x4e)), _mm_cvtepi16_epi32(e00))), s);
    v[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e01, 0x4e)), _mm_cvtepi16_epi32(e01))), s);
    v[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e10, 0x4e)), _mm_cvtepi16_epi32(e10))), s);
    v[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e11, 0x4e)), _mm_cvtepi16_epi32(e11))), s);
}

/* Decode one Q4_1 block (20 bytes) -> 4 F32x8 dequant values.
 * Q4_1 format: [2 bytes f16 dmin][2 bytes f16 dmax][16 bytes nibbles] = 20 bytes
 * Dequant: val = dmin + (dmax - dmin) * nibble / 15.0
 */
static inline void decode_q4_1(const uint8_t* bp, __m256 v[4]) {
    float dmin = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
    float dmax = f16_to_f32((uint16_t)bp[2] | ((uint16_t)bp[3] << 8));
    float diff = dmax - dmin;
    __m256 dmin_v = _mm256_set1_ps(dmin);
    __m256 diff_v = _mm256_set1_ps(diff / 15.0f);
    
    __m128i nb = _mm_loadu_si128((__m128i*)(bp + 4));
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(15));
    __m128i i0 = _mm_unpacklo_epi8(lo, hi);
    __m128i i1 = _mm_unpackhi_epi8(lo, hi);
    __m128i e00 = _mm_cvtepu8_epi16(i0);
    __m128i e01 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(i0, 0x4e));
    __m128i e10 = _mm_cvtepu8_epi16(i1);
    __m128i e11 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(i1, 0x4e));
    v[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e00, 0x4e)), _mm_cvtepi16_epi32(e00))), diff_v, dmin_v);
    v[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e01, 0x4e)), _mm_cvtepi16_epi32(e01))), diff_v, dmin_v);
    v[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e10, 0x4e)), _mm_cvtepi16_epi32(e10))), diff_v, dmin_v);
    v[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(
        _mm_cvtepi16_epi32(_mm_shuffle_epi32(e11, 0x4e)), _mm_cvtepi16_epi32(e11))), diff_v, dmin_v);
}

/* Generic Q4_X matmul with configurable type_size and block decoder.
 * type_size=18 for Q4_0, type_size=20 for Q4_1.
 */
void q4_matmul_avx2(
    const uint8_t* w, const float* x, float* out,
    int n_rows, int n_cols, int start, int end, int type_size
) {
    int bpr = n_cols / 32;
    int n_out = end - start;
    for (int i = 0; i < n_out; i++) out[i] = 0.0f;
    
    void (*decode)(const uint8_t*, __m256*) = 
        (type_size == 20) ? decode_q4_1 : decode_q4_0;
    
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

/* Convenience wrappers */
void q4_0_matmul(const uint8_t* w, const float* x, float* out,
                  int n_rows, int n_cols, int start, int end) {
    q4_matmul_avx2(w, x, out, n_rows, n_cols, start, end, 18);
}

void q4_1_matmul(const uint8_t* w, const float* x, float* out,
                  int n_rows, int n_cols, int start, int end) {
    q4_matmul_avx2(w, x, out, n_rows, n_cols, start, end, 20);
}
