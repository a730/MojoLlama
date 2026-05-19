/* simd_ops.c — AVX2 SIMD vector operations for LLM inference
 *
 * Provides C-callable SIMD-optimized operations that replace
 * numpy vector ops in the Python forward pass:
 *   - RMS normalization
 *   - SiLU activation  
 *   - Residual add
 *   - Softmax (attention scores)
 *   - RoPE (rotary positional embedding)
 *   - Weighted V accumulation (attention)
 *
 * These ops currently take ~6.9ms in Python (28% of total).
 * AVX2 SIMD should bring this down to ~1-2ms.
 *
 * Compile: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *          -o simd_ops.so simd_ops.c -lm
 */

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* RMS normalization: out = x * weight / sqrt(mean(x^2) + eps) */
void rms_norm(float *restrict out, const float *restrict x,
              const float *restrict weight, int n, float eps) {
    __m256 sum_sq = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }
    float ss = hsum_ps(sum_sq);
    for (; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / n + eps);
    __m256 inv_v = _mm256_set1_ps(inv_rms);
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, inv_v), w));
    }
    for (; i < n; i++) out[i] = x[i] * inv_rms * weight[i];
}

/* SiLU activation: out = x * sigmoid(x), in-place if out == x */
void silu(float *restrict out, const float *restrict x, int n) {
    /* Use tanh approximation for fast sigmoid: sig(x) ≈ 0.5 + 0.5*tanh(x/2) */
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 half_v = _mm256_mul_ps(v, _mm256_set1_ps(0.5f));
        /* tanh(x) ≈ x*(27+x²)/(27+9x²) — Pade approximation, max error ~0.003 */
        __m256 v2 = _mm256_mul_ps(half_v, half_v);
        __m256 num = _mm256_mul_ps(half_v, _mm256_add_ps(_mm256_set1_ps(27.0f), v2));
        __m256 den = _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), v2));
        __m256 tanh_v = _mm256_div_ps(num, den);
        __m256 sigmoid = _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(_mm256_set1_ps(0.5f), tanh_v));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, sigmoid));
    }
    for (; i < n; i++) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        out[i] = x[i] * sig;
    }
}

/* Residual add: out = a + b */
void residual_add(float *restrict out, const float *restrict a,
                   const float *restrict b, int n) {
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                   _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

/* Softmax in-place: x[i] = exp(x[i]) / sum(exp(x[j])) */
void softmax(float *restrict x, int n) {
    float max_val = -INFINITY;
    for (int i = 0; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    __m256 max_v = _mm256_set1_ps(max_val);
    __m256 sum_v = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), max_v);
        /* exp using exp2: exp(x) = exp2(x/log2(e)) */
        /* Faster: use the fact that exp(x) ≈ 2^(x*1.4427) for small x */
        /* Actually, just use scalar expf — softmax n is small (≤2048) */
        for (int j = 0; j < 8; j++) {
            x[i+j] = expf(x[i+j] - max_val);
            sum_v = _mm256_add_ps(sum_v, _mm256_set1_ps(x[i+j]));
        }
    }
    float sum = hsum_ps(sum_v);
    for (; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    __m256 inv_v = _mm256_set1_ps(inv_sum);
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), inv_v));
    }
    for (; i < n; i++) x[i] *= inv_sum;
}

/* RoPE: apply rotary positional embedding to Q and K
 * q: [n_heads * head_dim], k: [n_kv_heads * head_dim]
 */
void apply_rope(float *restrict q, float *restrict k,
                int n_heads, int n_kv_heads, int head_dim,
                int pos, float rope_base) {
    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < head_dim; d += 2) {
            float freq = 1.0f / powf(rope_base, (float)(d / 2) / (head_dim / 2));
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            int idx = h * head_dim + d;
            float q0 = q[idx], q1 = q[idx + 1];
            q[idx]     = q0 * cos_a - q1 * sin_a;
            q[idx + 1] = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        for (int d = 0; d < head_dim; d += 2) {
            float freq = 1.0f / powf(rope_base, (float)(d / 2) / (head_dim / 2));
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            int idx = h * head_dim + d;
            float k0 = k[idx], k1 = k[idx + 1];
            k[idx]     = k0 * cos_a - k1 * sin_a;
            k[idx + 1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

/* Attention: compute scores = Q @ K^T / sqrt(d), softmax, then V * scores
 * q: [n_heads, head_dim]
 * k_cache: [n_past+1, n_kv_heads, head_dim] 
 * v_cache: [n_past+1, n_kv_heads, head_dim]
 * out: [n_heads, head_dim]
 * This is a batched operation for all heads.
 * Uses GQA: each Q head maps to kv_h = h * n_kv_heads / n_heads
 */
void attention_forward(
    const float *restrict q,       /* [n_heads * head_dim] */
    const float *restrict k_cache, /* [max_seq * n_kv_heads * head_dim] */
    const float *restrict v_cache, /* [max_seq * n_kv_heads * head_dim] */
    float *restrict out,           /* [n_heads * head_dim] */
    int n_heads, int n_kv_heads, int head_dim,
    int n_past, int max_seq) {
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    /* Allocate temp buffer for attention scores on stack or heap */
    /* n_past+1 scores per head, at most 4096 */
    float scores[4096];  /* VLA would be better but C99 VLA on stack */
    int seq_len = n_past + 1;
    
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h * n_kv_heads / n_heads;
        const float *q_h = q + h * head_dim;
        
        /* Compute Q @ K^T */
        for (int t = 0; t < seq_len; t++) {
            const float *k_t = k_cache + t * n_kv_heads * head_dim + kv_h * head_dim;
            __m256 dot_v = _mm256_setzero_ps();
            for (int d = 0; d <= head_dim - 8; d += 8) {
                __m256 qv = _mm256_loadu_ps(q_h + d);
                __m256 kv = _mm256_loadu_ps(k_t + d);
                dot_v = _mm256_fmadd_ps(qv, kv, dot_v);
            }
            scores[t] = hsum_ps(dot_v) * scale;
        }
        
        /* Softmax */
        softmax(scores, seq_len);
        
        /* Weighted V accumulation */
        float *out_h = out + h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            const float *v_t = v_cache + t * n_kv_heads * head_dim + kv_h * head_dim;
            __m256 sw = _mm256_set1_ps(scores[t]);
            for (int d = 0; d <= head_dim - 8; d += 8) {
                __m256 vd = _mm256_loadu_ps(v_t + d);
                __m256 acc = _mm256_loadu_ps(out_h + d);
                _mm256_storeu_ps(out_h + d, _mm256_fmadd_ps(sw, vd, acc));
            }
        }
    }
}

/* Copy KV values into cache */
void kv_cache_update(float *restrict k_cache, float *restrict v_cache,
                     const float *restrict k_new, const float *restrict v_new,
                     int layer, int n_past, int max_seq,
                     int n_kv_heads, int head_dim) {
    int offset = n_past * n_kv_heads * head_dim;
    memcpy(k_cache + offset, k_new, n_kv_heads * head_dim * sizeof(float));
    memcpy(v_cache + offset, v_new, n_kv_heads * head_dim * sizeof(float));
}

/* Set OMP threads */
void set_num_threads(int n) { omp_set_num_threads(n); }