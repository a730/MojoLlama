/* gqa_attention.c — GQA attention with cblas_sgemm + single-threaded OpenBLAS.
 *
 * Uses cblas_sgemm for the attention score computation and V aggregation.
 * OpenBLAS is set to single-thread (openblas_set_num_threads(1)) to avoid
 * nested OMP parallelism. The outer OMP loop handles head parallelism instead.
 *
 * For each KV head:
 *   scores[gqa_rep, seq_len] = Q[gqa_rep, HD] @ K[kv_h, seq_len, HD]^T * scale
 *   softmax each head's scores
 *   out[gqa_rep, HD] = scores * V[kv_h, seq_len, HD]
 *
 * Compile:
 *   gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC -o gqa_attention.so \
 *       gqa_attention.c -lopenblas -lm
 */
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <cblas.h>
#include <omp.h>
#include <immintrin.h>

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline __m256 exp_ps_simd(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 t = _mm256_mul_ps(x, log2e);
    __m256i n = _mm256_cvtps_epi32(_mm256_add_ps(t, _mm256_set1_ps(0.5f)));
    __m256 f = _mm256_sub_ps(t, _mm256_cvtepi32_ps(n));
    __m256 p = _mm256_set1_ps(1.0f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.6931471805599453f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.2402265069591007f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.0555041086648216f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.0096181291076285f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.0013333558146428f));
    __m256i biased = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    biased = _mm256_max_epi32(biased, _mm256_setzero_si256());
    biased = _mm256_min_epi32(biased, _mm256_set1_epi32(255));
    __m256 two_n = _mm256_castsi256_ps(_mm256_slli_epi32(biased, 23));
    return _mm256_mul_ps(p, two_n);
}

static inline void softmax_simd(float *scores, int seq_len) {
    int s;
    __m256 vmax = _mm256_set1_ps(-1e30f);
    for (s = 0; s <= seq_len - 8; s += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(scores + s));
    float max_val;
    {
        __m128 v = _mm256_castps256_ps128(vmax);
        __m128 v2 = _mm256_extractf128_ps(vmax, 1);
        v = _mm_max_ps(v, v2);
        v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0x4e));
        v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0xb1));
        max_val = _mm_cvtss_f32(v);
    }
    for (; s < seq_len; s++)
        if (scores[s] > max_val) max_val = scores[s];

    __m256 vsum = _mm256_setzero_ps();
    __m256 vmaxv = _mm256_set1_ps(max_val);
    for (s = 0; s <= seq_len - 8; s += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(scores + s), vmaxv);
        v = _mm256_min_ps(_mm256_max_ps(v, _mm256_set1_ps(-80.0f)), _mm256_set1_ps(80.0f));
        __m256 e = exp_ps_simd(v);
        _mm256_storeu_ps(scores + s, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float sum_exp = hsum_ps(vsum);
    for (; s < seq_len; s++) {
        float e = expf(scores[s] - max_val);
        if (e > 1e38f) e = 1e38f;
        scores[s] = e; sum_exp += e;
    }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    __m256 invv = _mm256_set1_ps(inv_sum);
    for (s = 0; s <= seq_len - 8; s += 8)
        _mm256_storeu_ps(scores + s, _mm256_mul_ps(_mm256_loadu_ps(scores + s), invv));
    for (; s < seq_len; s++) scores[s] *= inv_sum;
}

/* GQA attention decode with cblas_sgemm.
 * Sets OpenBLAS to single-thread to avoid nested OMP conflicts.
 * Uses OMP parallelism over kv_heads instead.
 */
void gqa_attention_decode(const float *q, const float *k_cache, const float *v_cache,
                           float *out, int seq_len, int n_head, int n_kv_head, int head_dim,
                           float *workspace) {
    int gqa_rep = n_head / n_kv_head;
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_stride = n_kv_head * head_dim;
    
    /* Set OpenBLAS to single-thread (avoid nested OMP) */
    openblas_set_num_threads(1);
    
    float *scores_all = workspace;
    float *tmp_scores = workspace + (size_t)n_head * seq_len;

    #pragma omp parallel for schedule(static)
    for (int kv_h = 0; kv_h < n_kv_head; kv_h++) {
        const float *qh = q + (size_t)kv_h * gqa_rep * head_dim;
        const float *kh = k_cache + (size_t)kv_h * head_dim;
        const float *vh = v_cache + (size_t)kv_h * head_dim;
        
        /* scores[gqa_rep, seq_len] = Q[gqa_rep, HD] * K[kv_h, seq_len, HD]^T * scale */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            gqa_rep, seq_len, head_dim,
            scale, qh, head_dim, kh, kv_stride,
            0.0f, tmp_scores, seq_len);
        
        /* Softmax each of the gqa_rep heads */
        for (int qi = 0; qi < gqa_rep; qi++) {
            float *sh = scores_all + (size_t)(kv_h * gqa_rep + qi) * seq_len;
            memcpy(sh, tmp_scores + (size_t)qi * seq_len, seq_len * sizeof(float));
            softmax_simd(sh, seq_len);
        }
        
        /* out[gqa_rep, HD] = scores * V[kv_h, seq_len, HD] */
        for (int qi = 0; qi < gqa_rep; qi++) {
            float *sh = scores_all + (size_t)(kv_h * gqa_rep + qi) * seq_len;
            float *oh = out + (size_t)(kv_h * gqa_rep + qi) * head_dim;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, head_dim, seq_len,
                1.0f, sh, seq_len, vh, kv_stride,
                0.0f, oh, head_dim);
        }
    }
}
