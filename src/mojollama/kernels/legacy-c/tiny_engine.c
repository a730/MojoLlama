/* tiny_engine.c — Minimal C forward pass calling proven avx2 batch kernels.
 * Uses the EXACT same setup as v7.7, only the per-layer loop is in C.
 * 
 * Compile:
 *   gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *       -o tiny_engine.so tiny_engine.c -lm
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

/* External kernels (proven correct from v7.7) */
extern void batch_qkv_q4_0(const uint8_t*, const uint8_t*, const uint8_t*,
                            const float*, float*, float*, float*,
                            int, int, int, int);
extern void batch_gate_up_q4_0(const uint8_t*, const uint8_t*,
                                const float*, float*, float*, int, int, int);
extern void gqa_attention_decode(const float*, const float*, const float*,
                                  float*, int, int, int, int);
extern void rms_norm(const float*, const float*, float*, int, float);
extern void silu(float*, int);

/* ═════════════════════════════════════════════════════════════════════════
 * forward_pass — drop-in replacement for v7.7 Python for loop
 * 
 * All pointers are pre-loaded by Python exactly as v7.7 does it.
 * This function JUST runs the 22-layer loop in C.
 * ═════════════════════════════════════════════════════════════════════════ */
void forward_pass(
    int token_id,
    /* Model dimensions */
    int L, int N, int NH, int NKH, int HD, int FF, float eps,
    /* Per-layer weight pointers [L] — raw Q4_0 data */
    const uint8_t **w_q, const uint8_t **w_k, const uint8_t **w_v,
    const uint8_t **w_o, const uint8_t **w_g, const uint8_t **w_u, const uint8_t **w_d,
    /* Per-layer norm weights [L] */
    const float **w_an, const float **w_fn,
    /* Per-layer row counts [L] */
    const int *n_q, const int *n_k, const int *n_v,
    const int *n_o, const int *n_g, const int *n_u, const int *n_d,
    int n_cols,
    /* Embedding + output norm */
    const float *embed, const float *out_norm_w,
    /* Output projection */
    const uint8_t *w_out, int out_nr, int out_nc,
    /* Mutable KV cache: [L, MP, NKH*HD], [L] lengths */
    float *kv_k, float *kv_v, int *kv_len, int max_pos,
    /* Pre-allocated buffers (size = max(N, FF, NH*HD)) */
    float *x, float *xn, float *res,
    float *q, float *k, float *v,
    float *att, float *gate, float *up,
    float *silu_buf, float *o_proj, float *ffn,
    /* Output logits [vocab] */
    float *logits
) {
    /* Embedding */
    int nc = n_cols, bpr = nc / 32;
    memcpy(x, embed + (size_t)token_id * N, N * sizeof(float));

    for (int l = 0; l < L; l++) {
        /* 1. Residual + RMS norm 1 */
        memcpy(res, x, N * sizeof(float));
        rms_norm(xn, x, w_an[l], N, eps);

        /* 2. QKV batch (proven kernel) */
        batch_qkv_q4_0(w_q[l], w_k[l], w_v[l], xn, q, k, v,
                       n_q[l], n_k[l], n_v[l], n_cols);

        /* 3. RoPE (on-the-fly, matches v7.7's apply_rope_fast) */
        int pos = kv_len[l];
        for (int h = 0; h < NH; h++) {
            float *qh = q + h * HD;
            for (int j = 0; j < HD / 2; j++) {
                double ang = (double)pos / pow(10000.0, 2.0 * j / HD);
                double cs = cos(ang), sn = sin(ang);
                float a = qh[j], b = qh[j + HD / 2];
                qh[j]        = (float)(a * cs - b * sn);
                qh[j + HD/2] = (float)(b * cs + a * sn);
            }
        }
        for (int h = 0; h < NKH; h++) {
            float *kh = k + h * HD;
            for (int j = 0; j < HD / 2; j++) {
                double ang = (double)pos / pow(10000.0, 2.0 * j / HD);
                double cs = cos(ang), sn = sin(ang);
                float a = kh[j], b = kh[j + HD / 2];
                kh[j]        = (float)(a * cs - b * sn);
                kh[j + HD/2] = (float)(b * cs + a * sn);
            }
        }

        /* 4. KV store */
        memcpy(kv_k + ((size_t)l * max_pos + kv_len[l]) * NKH * HD,
               k, NKH * HD * sizeof(float));
        memcpy(kv_v + ((size_t)l * max_pos + kv_len[l]) * NKH * HD,
               v, NKH * HD * sizeof(float));

        /* 5. Attention */
        int seq_len = kv_len[l] + 1;
        gqa_attention_decode(q,
            kv_k + (size_t)l * max_pos * NKH * HD,
            kv_v + (size_t)l * max_pos * NKH * HD,
            att, seq_len, NH, NKH, HD);
        kv_len[l]++;

        /* 6. Output projection + residual */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            /* We need a Q4_0 dot product here. Use proven kernel from avx2_batch.
             * The batch_qkv_q4_0 approach but for a single weight matrix. */
            /* Actually, re-use the batch_qkv logic by treating it as batch of 1 */
        }
        /* Fallback: compute dot using same method as batch_qkv_q4_0 */
        /* C doesn't have a "single row" function exposed, so inline it: */
        /* For now, use the q4_row from our previous attempt — it's the same
         * code as the proven batch_qkv_q4_0 kernel, just inlined. */
        
        /* ═══ Inline Q4_0 matmul for O projection ═══ */
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < n_o[l]; row++) {
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            for (int blk = 0; blk < bpr; blk++) {
                const uint8_t *bp = w_o[l] + ((size_t)row * bpr + blk) * 18;
                float sc = _cvtsh_ss((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
                __m256 sv = _mm256_set1_ps(sc);
                __m128i nb = _mm_loadu_si128((const __m128i*)(bp + 2));
                __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
                __m128i hi = _mm_and_si128(
                    _mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4),
                    _mm_set1_epi8(15));
                __m128i ls = _mm_sub_epi8(lo, _mm_set1_epi8(8));
                __m128i hs = _mm_sub_epi8(hi, _mm_set1_epi8(8));
                __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(ls)));
                __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_cvtepi8_epi16(_mm_shuffle_epi32(ls, 0x4e))));
                __m256 v2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(hs)));
                __m256 v3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_cvtepi8_epi16(_mm_shuffle_epi32(hs, 0x4e))));
                v0 = _mm256_mul_ps(v0, sv); v1 = _mm256_mul_ps(v1, sv);
                v2 = _mm256_mul_ps(v2, sv); v3 = _mm256_mul_ps(v3, sv);
                int o = blk * 32;
                s0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(att + o), s0);
                s0 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(att + o + 8), s0);
                s1 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(att + o + 16), s1);
                s1 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(att + o + 24), s1);
            }
            __m128 lo128 = _mm256_castps256_ps128(_mm256_add_ps(s0, s1));
            __m128 hi128 = _mm256_extractf128_ps(_mm256_add_ps(s0, s1), 1);
            lo128 = _mm_add_ps(lo128, hi128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            o_proj[row] = _mm_cvtss_f32(lo128);
        }
        for (int i = 0; i < N; i++) x[i] = res[i] + o_proj[i];

        /* 7. FFN: norm 2 */
        memcpy(res, x, N * sizeof(float));
        rms_norm(xn, x, w_fn[l], N, eps);

        /* 8. Gate + Up batch (proven kernel) */
        batch_gate_up_q4_0(w_g[l], w_u[l], xn, gate, up,
                           n_g[l], n_u[l], n_cols);

        /* 9. SiLU(gate) * up */
        for (int i = 0; i < FF; i++) {
            float g = gate[i];
            silu_buf[i] = (g / (1.0f + expf(-g))) * up[i];
        }

        /* 10. Down projection + residual (inline Q4_0 matmul) */
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < n_d[l]; row++) {
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            for (int blk = 0; blk < bpr; blk++) {
                const uint8_t *bp = w_d[l] + ((size_t)row * bpr + blk) * 18;
                float sc = _cvtsh_ss((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
                __m256 sv = _mm256_set1_ps(sc);
                __m128i nb = _mm_loadu_si128((const __m128i*)(bp + 2));
                __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
                __m128i hi = _mm_and_si128(
                    _mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4),
                    _mm_set1_epi8(15));
                __m128i ls = _mm_sub_epi8(lo, _mm_set1_epi8(8));
                __m128i hs = _mm_sub_epi8(hi, _mm_set1_epi8(8));
                __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(ls)));
                __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_cvtepi8_epi16(_mm_shuffle_epi32(ls, 0x4e))));
                __m256 v2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(hs)));
                __m256 v3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_cvtepi8_epi16(_mm_shuffle_epi32(hs, 0x4e))));
                v0 = _mm256_mul_ps(v0, sv); v1 = _mm256_mul_ps(v1, sv);
                v2 = _mm256_mul_ps(v2, sv); v3 = _mm256_mul_ps(v3, sv);
                int o = blk * 32;
                s0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(silu_buf + o), s0);
                s0 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(silu_buf + o + 8), s0);
                s1 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(silu_buf + o + 16), s1);
                s1 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(silu_buf + o + 24), s1);
            }
            __m128 lo128 = _mm256_castps256_ps128(_mm256_add_ps(s0, s1));
            __m128 hi128 = _mm256_extractf128_ps(_mm256_add_ps(s0, s1), 1);
            lo128 = _mm_add_ps(lo128, hi128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            lo128 = _mm_hadd_ps(lo128, lo128);
            ffn[row] = _mm_cvtss_f32(lo128);
        }
        for (int i = 0; i < N; i++) x[i] = res[i] + ffn[i];
    }

    /* Final norm + output projection */
    rms_norm(xn, x, out_norm_w, N, eps);
    
    bpr = out_nc / 32;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < out_nr; row++) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = w_out + ((size_t)row * bpr + blk) * 34;
            float d = _cvtsh_ss((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            __m256 dv = _mm256_set1_ps(d);
            const int8_t *qs = (const int8_t*)(bp + 2);
            int o = blk * 32;
            for (int i = 0; i < 32; i += 8) {
                __m128i q8 = _mm_loadl_epi64((const __m128i*)(qs + i));
                __m128i q16 = _mm_cvtepi8_epi16(q8);
                __m256i q32 = _mm256_cvtepi16_epi32(q16);
                __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), dv);
                if (i < 16) {
                    s0 = _mm256_fmadd_ps(v, _mm256_loadu_ps(xn + o + i), s0);
                } else {
                    s1 = _mm256_fmadd_ps(v, _mm256_loadu_ps(xn + o + i), s1);
                }
            }
        }
        float *out_row = logits + row;
        *out_row = 0.0f;  /* placeholder */
        float total = 0.0f;
        __m128 l128 = _mm256_castps256_ps128(_mm256_add_ps(s0, s1));
        __m128 h128 = _mm256_extractf128_ps(_mm256_add_ps(s0, s1), 1);
        l128 = _mm_add_ps(l128, h128);
        l128 = _mm_hadd_ps(l128, l128);
        l128 = _mm_hadd_ps(l128, l128);
        logits[row] = _mm_cvtss_f32(l128);
    }
}
