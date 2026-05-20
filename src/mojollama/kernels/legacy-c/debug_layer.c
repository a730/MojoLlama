/* debug_layer.c — Print first layer intermediate values for debugging.
 * Compile: gcc -O0 -g -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *          -o debug_layer.so debug_layer.c -lm
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define Q4_0_BS 18

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static inline float hsum_ps(__m256 v) {
    __m128 l=_mm256_castps256_ps128(v),h=_mm256_extractf128_ps(v,1);
    l=_mm_add_ps(l,h);l=_mm_hadd_ps(l,l);l=_mm_hadd_ps(l,l);return _mm_cvtss_f32(l);
}

void test_q4_matmul(const uint8_t *W, const float *x, float *out,
                     int n_rows, int n_cols, int layer, const char *name) {
    int bp=n_cols/32;
    #pragma omp parallel for schedule(static)
    for(int r=0;r<n_rows;r++){
        __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps();
        for(int b=0;b<bp;b++){
            const uint8_t *p=W+((size_t)r*bp+b)*Q4_0_BS;
            float sc=f16_to_f32((uint16_t)p[0]|((uint16_t)p[1]<<8));
            __m256 sv=_mm256_set1_ps(sc);
            __m128i nb=_mm_loadu_si128((const __m128i*)(p+2));
            __m128i lo=_mm_and_si128(nb,_mm_set1_epi8(15));
            __m128i hi=_mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb,_mm_set1_epi8((char)0xF0)),4),_mm_set1_epi8(15));
            __m128i ls=_mm_sub_epi8(lo,_mm_set1_epi8(8)),hs=_mm_sub_epi8(hi,_mm_set1_epi8(8));
            __m256 v0=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(ls)));
            __m256 v1=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_shuffle_epi32(ls,0x4e))));
            __m256 v2=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(hs)));
            __m256 v3=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_shuffle_epi32(hs,0x4e))));
            v0=_mm256_mul_ps(v0,sv);v1=_mm256_mul_ps(v1,sv);v2=_mm256_mul_ps(v2,sv);v3=_mm256_mul_ps(v3,sv);
            int o=b*32;
            s0=_mm256_fmadd_ps(v0,_mm256_loadu_ps(x+o),s0); s0=_mm256_fmadd_ps(v1,_mm256_loadu_ps(x+o+8),s0);
            s1=_mm256_fmadd_ps(v2,_mm256_loadu_ps(x+o+16),s1); s1=_mm256_fmadd_ps(v3,_mm256_loadu_ps(x+o+24),s1);
        }
        out[r]=hsum_ps(_mm256_add_ps(s0,s1));
    }
    /* Print first 5 values */
    printf("Layer %d %s [0:5]: %.4f %.4f %.4f %.4f %.4f\n",
           layer, name, out[0], out[1], out[2], out[3], out[4]);
}
