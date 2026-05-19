/* gqa_attention.c — GQA attention kernel for decode step.
 * 
 * Computes attention for 1 query token against K/V cache:
 *   scores = softmax(Q @ K^T / sqrt(d))  ->  (n_head, seq_len)
 *   out = scores @ V                      ->  (n_head, head_dim)
 * 
 * GQA: q_heads / kv_heads = gqa_rep, each q head maps to kv_heads[q // gqa_rep]
 * 
 * Compile:
 *   gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC -o gqa_attention.so gqa_attention.c -lm
 */
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

// AVX2 horizontal sum
static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// GQA decode attention: 1 query token against K/V cache
// q:          [n_head * head_dim]  F32  — query for current token (already RoPE'd)
// k_cache:    [seq_len * n_kv_head * head_dim]  F32  — key cache (row-major)
// v_cache:    [seq_len * n_kv_head * head_dim]  F32  — value cache (row-major)
// out:        [n_head * head_dim]  F32  — output (attention result)
// seq_len:    number of cached positions
// n_head:     number of query heads
// n_kv_head:  number of key/value heads
// head_dim:   dimension per head
// gqa_rep:    n_head / n_kv_head  (e.g., 8 for TinyLlama)

void gqa_attention_decode(const float *q, const float *k_cache, const float *v_cache,
                           float *out, int seq_len, int n_head, int n_kv_head, int head_dim) {
    int gqa_rep = n_head / n_kv_head;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < n_head; h++) {
        int kv_h = h / gqa_rep;
        const float *qh = q + h * head_dim;
        const float *kh_base = k_cache + kv_h * head_dim;
        const float *vh_base = v_cache + kv_h * head_dim;
        int kv_stride = n_kv_head * head_dim;
        float *oh = out + h * head_dim;

        // Compute scores: Q · K_cache / sqrt(d)
        float scores[4096];  // max seq_len; will stack-allocate
        float max_score = -1e30f;
        float scale = 1.0f / sqrtf((float)head_dim);

        for (int s = 0; s < seq_len; s++) {
            const float *ks = kh_base + s * kv_stride;
            // AVX2 dot product
            __m256 sum = _mm256_setzero_ps();
            int d;
            for (d = 0; d <= head_dim - 8; d += 8) {
                __m256 qv = _mm256_loadu_ps(qh + d);
                __m256 kv = _mm256_loadu_ps(ks + d);
                sum = _mm256_fmadd_ps(qv, kv, sum);
            }
            float dot = hsum_ps(sum);
            for (; d < head_dim; d++) dot += qh[d] * ks[d];
            dot *= scale;
            scores[s] = dot;
            if (dot > max_score) max_score = dot;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum_exp += scores[s];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int s = 0; s < seq_len; s++) {
            scores[s] *= inv_sum;
        }

        // Weighted V sum: out[h] = sum_s scores[s] * V[s, kv_h, :]
        memset(oh, 0, head_dim * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            const float *vs = vh_base + s * kv_stride;
            float w = scores[s];
            int d;
            __m256 wv = _mm256_set1_ps(w);
            for (d = 0; d <= head_dim - 8; d += 8) {
                __m256 vv = _mm256_loadu_ps(vs + d);
                __m256 acc = _mm256_loadu_ps(oh + d);
                _mm256_storeu_ps(oh + d, _mm256_fmadd_ps(wv, vv, acc));
            }
            for (; d < head_dim; d++) oh[d] += w * vs[d];
        }
    }
}
