/**
 * turbo_forward.c — Full transformer forward pass in C
 * 
 * Eliminates all Python per-layer overhead by keeping the entire
 * forward pass in C. Python only calls once per token.
 * 
 * Architecture supports: Llama (dense) and Qwen3 (MoE)
 * Quant types: Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, F32
 */

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define QK_K   256
#define QK8_0  32

/* ── Quant type codes (GGML convention) ── */
#define GGML_F32   0
#define GGML_F16   1
#define GGML_Q4_0  2
#define GGML_Q4_1  3
#define GGML_Q8_0  8
#define GGML_Q4_K  12
#define GGML_Q5_K  13
#define GGML_Q6_K  14

/* ── Weight descriptor: one per model weight tensor ── */
typedef struct {
    void*    data;        /* Pointer to raw quantized bytes or F32 data */
    int      nrows;       /* out_dim */
    int      ncols;       /* in_dim */
    int      qtype;       /* GGML quant type code */
    int      block_size;  /* bytes per block (18 for Q4_0, 144 for Q4_K, etc) */
    int      vals_per_block; /* values per block (32 for Q4_0, 256 for Q4_K) */
} WeightDesc;

/* ── Layer descriptor: all weights for one transformer layer ── */
typedef struct {
    WeightDesc attn_norm;       /* RMSNorm weight [n_embd] */
    WeightDesc attn_q;          /* Q projection [n_embd, n_embd] */
    WeightDesc attn_k;          /* K projection [n_kv_head*head_dim, n_embd] */
    WeightDesc attn_v;          /* V projection [n_kv_head*head_dim, n_embd] */
    WeightDesc attn_out;        /* Output projection [n_embd, n_embd] */
    WeightDesc attn_q_norm;     /* Q per-head norm [head_dim] (Qwen3 GQA norm) */
    WeightDesc attn_k_norm;     /* K per-head norm [head_dim] */
    WeightDesc ffn_norm;        /* FFN RMSNorm [n_embd] */

    /* Dense FFN */
    WeightDesc ffn_gate;        /* [n_ff, n_embd] */
    WeightDesc ffn_up;         /* [n_ff, n_embd] */  
    WeightDesc ffn_down;       /* [n_embd, n_ff] */

    /* MoE */
    int        is_moe;
    WeightDesc moe_gate_inp;    /* Router [n_experts, n_embd] */
    int        n_experts;
    int        n_experts_per_tok;
    int        n_ff_expert;
    WeightDesc* moe_gate_exps;  /* [n_experts] each [n_ff_expert, n_embd] */
    WeightDesc* moe_up_exps;    /* [n_experts] each [n_ff_expert, n_embd] */
    WeightDesc* moe_down_exps;  /* [n_experts] each [n_embd, n_ff_expert] */

    /* Shared expert (Qwen3 MoE) */
    int        has_shared_expert;
    WeightDesc shared_gate;     /* [2*n_ff_expert, n_embd] */
    WeightDesc shared_up;       /* [2*n_ff_expert, n_embd] */
    WeightDesc shared_down;     /* [n_embd, 2*n_ff_expert] */
} LayerDesc;

/* ── Model descriptor: full transformer ── */
typedef struct {
    int        n_layers;
    int        n_embd;
    int        n_head;
    int        n_kv_head;
    int        head_dim;
    int        n_ff;
    float      rope_freq_base;
    float      eps;
    int        vocab_size;
    int        is_moe;

    WeightDesc token_embd;      /* [vocab_size, n_embd] */
    WeightDesc output_norm;     /* [n_embd] */
    WeightDesc output_weight;    /* [vocab_size, n_embd] or tied with token_embd */

    LayerDesc*  layers;          /* [n_layers] */

    /* Precomputed rope tables */
    float*     rope_cos;        /* [max_pos, head_dim/2] */
    float*     rope_sin;        /* [max_pos, head_dim/2] */
    int        max_pos;
} ModelDesc;

/* ── Forward pass state ── */
typedef struct {
    float* kv_k;         /* [n_layers, max_pos, n_kv_head*head_dim] */
    float* kv_v;         /* [n_layers, max_pos, n_kv_head*head_dim] */
    int*   kv_len;       /* [n_layers] */
    int    pos;

    /* Pre-allocated work buffers */
    float* h;            /* [n_embd] */
    float* residual;     /* [n_embd] */
    float* q;            /* [n_head*head_dim] */
    float* k;            /* [n_kv_head*head_dim] */
    float* v;            /* [n_kv_head*head_dim] */
    float* att_out;      /* [n_head*head_dim] */
    float* gate;         /* [n_ff] */
    float* up;           /* [n_ff] */
    float* silu_gate;    /* [n_ff] */
    float* ffn_out;      /* [n_embd] */
    float* logits;       /* [vocab_size] */
    float* scores;       /* [n_head, max_pos] for attention */
    float* expert_result;/* [n_embd] for MoE accumulation */

    int    max_pos;
    int    n_embd;
    int    n_head;
    int    n_kv_head;
    int    head_dim;
    int    n_ff;
    int    n_layers;
} ForwardState;

/* ── f16 to f32 conversion ── */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    float out;
    if (exp == 0) {
        if (mant == 0) { out = 0.0f; }
        else { out = ldexpf((float)mant / 1024.0f, -14); }
    } else if (exp == 31) {
        out = (mant == 0) ? INFINITY : NAN;
    } else {
        out = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    }
    return sign ? -out : out;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Quantized matmul kernels (AVX2 + FMA)
 * ══════════════════════════════════════════════════════════════════════════ */

static inline float hsum_ps(__m256 v) {
    __m256 h = _mm256_hadd_ps(v, _mm256_permute2f128_ps(v, v, 1));
    h = _mm256_hadd_ps(h, h);
    h = _mm256_hadd_ps(h, h);
    return _mm256_cvtss_f32(h);
}

/* ── F32 matmul (AVX2 + FMA) ── */
static void f32_matmul(const float* W, const float* x, float* out, int nrows, int ncols) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        const float* row = W + (size_t)r * ncols;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        int c;
        for (c = 0; c <= ncols - 16; c += 16) {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c),     _mm256_loadu_ps(x + c), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8),  _mm256_loadu_ps(x + c + 8), s1);
        }
        __m256 total = _mm256_add_ps(s0, s1);
        float result = hsum_ps(total);
        for (; c < ncols; c++) result += row[c] * x[c];
        out[r] = result;
    }
}

/* ── RMS Norm (AVX2) ── */
static void rms_norm(const float* x, const float* w, float* out, int n, float eps) {
    __m256 ss_vec = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        ss_vec = _mm256_fmadd_ps(v, v, ss_vec);
    }
    float ss = hsum_ps(ss_vec);
    for (; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / n + eps);
    __m256 inv_v = _mm256_set1_ps(inv_rms);
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), inv_v), _mm256_loadu_ps(w + i)));
    }
    for (; i < n; i++) out[i] = x[i] * inv_rms * w[i];
}

/* ── SiLU activation (vectorized) ── */
static void silu_inplace(float* x, int n) {
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        /* silu(x) = x / (1 + exp(-x)) = x * sigmoid(x) */
        /* Compute -x */
        __m256 neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
        /* Approximate exp: use expf in a loop for now — AVX2 doesn't have native exp */
        float tmp[8], neg_tmp[8], result[8];
        _mm256_storeu_ps(tmp, v);
        _mm256_storeu_ps(neg_tmp, neg_v);
        for (int j = 0; j < 8; j++) {
            result[j] = tmp[j] / (1.0f + expf(neg_tmp[j]));
        }
        _mm256_storeu_ps(x + i, _mm256_loadu_ps(result));
    }
    for (; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ── Q4_0 matmul (18 bytes per 32 values) ── */
static void q4_0_matmul(const uint8_t* W, const float* x, float* out, int nrows, int ncols) {
    int bpr = ncols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        float total = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t* bp = W + ((size_t)r * bpr + blk) * 18;
            float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            __m256 dv = _mm256_set1_ps(d);
            __m128i nb = _mm_loadu_si128((__m128i*)(bp + 2));
            __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
            __m128i lo_s = _mm_sub_epi8(lo, _mm_set1_epi8(8));
            __m128i hi_s = _mm_sub_epi8(hi, _mm_set1_epi8(8));
            /* Expand lo[0:7], lo[8:15], hi[0:7], hi[8:15] to 4 x __m256 */
            __m128i e00 = _mm_cvtepi8_epi16(lo_s), e01 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(lo_s, 0x4e));
            __m128i e10 = _mm_cvtepi8_epi16(hi_s), e11 = _mm_cvtepi8_epi16(_mm_shuffle_epi32(hi_s, 0x4e));
            __m256 v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00, 0x4e)), _mm_cvtepi16_epi32(e00))), dv);
            __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01, 0x4e)), _mm_cvtepi16_epi32(e01))), dv);
            __m256 v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10, 0x4e)), _mm_cvtepi16_epi32(e10))), dv);
            __m256 v3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11, 0x4e)), _mm_cvtepi16_epi32(e11))), dv);
            int off = blk * 32;
            __m256 a = _mm256_mul_ps(v0, _mm256_loadu_ps(x + off));
            a = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8), a);
            a = _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16), a);
            a = _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24), a);
            total += hsum_ps(a);
        }
        out[r] = total;
    }
}

/* ── Q4_1 matmul (20 bytes per 32 values) ── */
static void q4_1_matmul(const uint8_t* W, const float* x, float* out, int nrows, int ncols) {
    int bpr = ncols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        float total = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t* bp = W + ((size_t)r * bpr + blk) * 20;
            float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            float m = f16_to_f32((uint16_t)bp[2] | ((uint16_t)bp[3] << 8));
            __m256 d_v = _mm256_set1_ps(d), m_v = _mm256_set1_ps(m);
            __m128i nb = _mm_loadu_si128((__m128i*)(bp + 4));
            __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(15));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(nb, _mm_set1_epi8((char)0xF0)), 4), _mm_set1_epi8(15));
            __m128i e00 = _mm_cvtepu8_epi16(lo), e01 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(lo, 0x4e));
            __m128i e10 = _mm_cvtepu8_epi16(hi), e11 = _mm_cvtepu8_epi16(_mm_shuffle_epi32(hi, 0x4e));
            __m256 v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e00,0x4e)),_mm_cvtepi16_epi32(e00))), d_v, m_v);
            __m256 v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e01,0x4e)),_mm_cvtepi16_epi32(e01))), d_v, m_v);
            __m256 v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e10,0x4e)),_mm_cvtepi16_epi32(e10))), d_v, m_v);
            __m256 v3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi16_epi32(_mm_shuffle_epi32(e11,0x4e)),_mm_cvtepi16_epi32(e11))), d_v, m_v);
            int off = blk * 32;
            __m256 a = _mm256_mul_ps(v0, _mm256_loadu_ps(x + off));
            a = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x + off + 8), a);
            a = _mm256_fmadd_ps(v2, _mm256_loadu_ps(x + off + 16), a);
            a = _mm256_fmadd_ps(v3, _mm256_loadu_ps(x + off + 24), a);
            total += hsum_ps(a);
        }
        out[r] = total;
    }
}

/* ── Q8_0 matmul (34 bytes per 32 values) ── */
static void q8_0_matmul(const uint8_t* W, const float* x, float* out, int nrows, int ncols) {
    int bpr = ncols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        __m256 acc = _mm256_setzero_ps();
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t* bp = W + ((size_t)r * bpr + blk) * 34;
            float d = f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
            const int8_t* qs = (const int8_t*)(bp + 2);
            __m256 dv = _mm256_set1_ps(d);
            int off = blk * 32;
            for (int i = 0; i < 32; i += 8) {
                __m256i qi = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)(qs + i)));
                __m256 qf = _mm256_cvtepi32_ps(qi);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(qf, dv), _mm256_loadu_ps(x + off + i), acc);
            }
        }
        out[r] = hsum_ps(acc);
    }
}

/* ── K-Quant scale extraction helpers ── */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j] >> 6;
    } else if (j < 20) {
        *d = (q[j+4] & 15) | ((q[j-2] >> 6) << 4); *m = (q[j+4] >> 4);
    } else {
        *d = (q[j+4] & 63) >> 2; *m = (q[j+4] >> 4);
    }
}

/* ── Q4_K matmul (144 bytes per 256 values) ── */
static void q4_k_matmul(const uint8_t* W, const float* x, float* out, int nrows, int ncols) {
    int nb = ncols / 256;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        const uint8_t* row = W + (size_t)r * nb * 144;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* scales = blk + 4;
            const uint8_t* q = blk + 16;
            int x_off = b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                float d1, d2, m1, m2;
                get_scale_min_k4(is+0, scales, &sc, &m); d1 = d*sc; m1 = min*m;
                get_scale_min_k4(is+1, scales, &sc, &m); d2 = d*sc; m2 = min*m;
                __m256 d1v = _mm256_set1_ps(d1), d2v = _mm256_set1_ps(d2);
                __m256 m1v = _mm256_set1_ps(m1), m2v = _mm256_set1_ps(m2);
                __m256i mask4 = _mm256_set1_epi32(0xF);
                /* Lo nibbles */
                for (int l = 0; l < 32; l += 8) {
                    __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(q+l)));
                    __m256 fv = _mm256_sub_ps(_mm256_mul_ps(d1v, _mm256_cvtepi32_ps(_mm256_and_si256(qi, mask4))), m1v);
                    acc = _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + x_off + j + l), acc);
                }
                /* Hi nibbles */
                for (int l = 0; l < 32; l += 8) {
                    __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(q+l)));
                    __m256 fv = _mm256_sub_ps(_mm256_mul_ps(d2v, _mm256_cvtepi32_ps(_mm256_srli_epi32(_mm256_and_si256(qi, _mm256_set1_epi32(0xFF)), 4))), m2v);
                    acc = _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + x_off + j + 32 + l), acc);
                }
                q += 32; is += 2;
            }
        }
        out[r] = hsum_ps(acc);
    }
}

/* ── Q5_K matmul (176 bytes per 256 values) ── */
static void q5_k_matmul(const uint8_t* W, const float* x, float* out, int nrows, int ncols) {
    int nb = ncols / 256;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        const uint8_t* row = W + (size_t)r * nb * 176;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 176;
            float d   = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float min = f16_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* scales = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* ql = blk + 48;
            int x_off = b * 256;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                float d1, d2, m1, m2;
                get_scale_min_k4(is+0, scales, &sc, &m); d1 = d*sc; m1 = min*m;
                get_scale_min_k4(is+1, scales, &sc, &m); d2 = d*sc; m2 = min*m;
                __m256 d1v = _mm256_set1_ps(d1), d2v = _mm256_set1_ps(d2);
                __m256 m1v = _mm256_set1_ps(m1), m2v = _mm256_set1_ps(m2);
                __m256i u1v = _mm256_set1_epi32(u1), u2v = _mm256_set1_epi32(u2);
                /* Lo+5th bit */
                for (int l = 0; l < 32; l += 8) {
                    __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ql+l)));
                    __m256i qh8 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(qh+l)));
                    __m256 fv = _mm256_sub_ps(_mm256_mul_ps(d1v, _mm256_cvtepi32_ps(
                        _mm256_add_epi32(_mm256_and_si256(qi, _mm256_set1_epi32(0xF)),
                                         _mm256_slli_epi32(_mm256_and_si256(qh8, u1v), 1)))), m1v);
                    acc = _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + x_off + j + l), acc);
                }
                /* Hi+5th bit */
                for (int l = 0; l < 32; l += 8) {
                    __m256i qi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ql+l)));
                    __m256i qh8 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(qh+l)));
                    __m256 fv = _mm256_sub_ps(_mm256_mul_ps(d2v, _mm256_cvtepi32_ps(
                        _mm256_add_epi32(_mm256_srli_epi32(_mm256_and_si256(qi, _mm256_set1_epi32(0xFF)), 4),
                                         _mm256_slli_epi32(_mm256_and_si256(qh8, u2v), 1)))), m2v);
                    acc = _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + x_off + j + 32 + l), acc);
                }
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
        }
        out[r] = hsum_ps(acc);
    }
}

/* ── Q6_K matmul (210 bytes per 256 values) ── */
static void q6_k_matmul(const uint8_t* W, const float* x, float* out, int nrows, int ncols) {
    int nb = ncols / 256;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        const uint8_t* row = W + (size_t)r * nb * 210;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 210;
            float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            const int8_t* sc = (const int8_t*)(blk + 192);
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            int base = b * 256;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; l += 8) {
                    int is_ = l / 16;
                    __m256i ql_lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ql + l)));
                    __m256i ql_hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ql + l + 32)));
                    __m256i qh8 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(qh + l)));
                    __m256i mask4 = _mm256_set1_epi32(0xF);
                    __m256i mask3 = _mm256_set1_epi32(3);
                    __m256i off32 = _mm256_set1_epi32(32);

                    __m256i q1 = _mm256_add_epi32(_mm256_and_si256(ql_lo, mask4),
                                    _mm256_slli_epi32(_mm256_and_si256(qh8, mask3), 4));
                    __m256i q2 = _mm256_add_epi32(_mm256_and_si256(ql_hi, mask4),
                                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh8, 2), mask3), 4));
                    __m256i q3 = _mm256_add_epi32(_mm256_srli_epi32(_mm256_and_si256(ql_lo, _mm256_set1_epi32(0xFF)), 4),
                                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh8, 4), mask3), 4));
                    __m256i q4 = _mm256_add_epi32(_mm256_srli_epi32(_mm256_and_si256(ql_hi, _mm256_set1_epi32(0xFF)), 4),
                                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh8, 6), mask3), 4));

                    q1 = _mm256_sub_epi32(q1, off32);
                    q2 = _mm256_sub_epi32(q2, off32);
                    q3 = _mm256_sub_epi32(q3, off32);
                    q4 = _mm256_sub_epi32(q4, off32);

                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q1), _mm256_set1_ps(d * sc[is_+0])),
                                          _mm256_loadu_ps(x + base + n + l), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q2), _mm256_set1_ps(d * sc[is_+2])),
                                          _mm256_loadu_ps(x + base + n + l + 32), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q3), _mm256_set1_ps(d * sc[is_+4])),
                                          _mm256_loadu_ps(x + base + n + l + 64), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q4), _mm256_set1_ps(d * sc[is_+6])),
                                          _mm256_loadu_ps(x + base + n + l + 96), acc);
                }
                ql += 64; qh += 32; sc += 8;
            }
        }
        out[r] = hsum_ps(acc);
    }
}

/* ── Unified quant matmul dispatch ── */
static void quant_matmul(const WeightDesc* w, const float* x, float* out) {
    switch (w->qtype) {
        case GGML_F32:  f32_matmul((const float*)w->data, x, out, w->nrows, w->ncols); break;
        case GGML_Q4_0: q4_0_matmul((const uint8_t*)w->data, x, out, w->nrows, w->ncols); break;
        case GGML_Q4_1: q4_1_matmul((const uint8_t*)w->data, x, out, w->nrows, w->ncols); break;
        case GGML_Q8_0: q8_0_matmul((const uint8_t*)w->data, x, out, w->nrows, w->ncols); break;
        case GGML_Q4_K: q4_k_matmul((const uint8_t*)w->data, x, out, w->nrows, w->ncols); break;
        case GGML_Q5_K: q5_k_matmul((const uint8_t*)w->data, x, out, w->nrows, w->ncols); break;
        case GGML_Q6_K: q6_k_matmul((const uint8_t*)w->data, x, out, w->nrows, w->ncols); break;
    }
}

/* ── Apply RoPE to Q and K vectors ── */
static void apply_rope(float* vec, int pos, int n_heads, int head_dim,
                       float rope_freq_base) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* v = vec + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = powf(rope_freq_base, -2.0f * i / head_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float x0 = v[i];
            float x1 = v[i + half];
            v[i]       = x0 * cos_a - x1 * sin_a;
            v[i + half] = x1 * cos_a + x0 * sin_a;
        }
    }
}

/* ── Per-head RMS norm (Qwen3 QK norm) ── */
static void per_head_rms_norm(float* vec, const float* w, int n_heads, int head_dim, float eps) {
    for (int h = 0; h < n_heads; h++) {
        float* v = vec + h * head_dim;
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
        float inv_rms = 1.0f / sqrtf(ss / head_dim + eps);
        for (int i = 0; i < head_dim; i++) v[i] = v[i] * inv_rms * w[i];
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * Full transformer forward pass — single C call per token
 * ══════════════════════════════════════════════════════════════════════════ */

void turbo_forward(ForwardState* st, ModelDesc* model, int token_id, float* logits_out) {
    int N   = model->n_embd;
    int NH  = model->n_head;
    int NKH = model->n_kv_head;
    int HD  = model->head_dim;
    int L   = model->n_layers;
    int pos = st->pos;

    /* ── Embedding lookup ── */
    /* token_embd is [vocab_size, n_embd] */
    float* embd = (float*)model->token_embd.data;
    memcpy(st->h, embd + (size_t)token_id * N, N * sizeof(float));

    for (int i = 0; i < L; i++) {
        LayerDesc* layer = &model->layers[i];

        /* ── Attention ── */
        memcpy(st->residual, st->h, N * sizeof(float));
        rms_norm(st->h, (float*)layer->attn_norm.data, st->h, N, model->eps);

        /* Q/K/V projections */
        quant_matmul(&layer->attn_q, st->h, st->q);
        quant_matmul(&layer->attn_k, st->h, st->k);

        /* Per-head QK norm (Qwen3) */
        if (layer->attn_q_norm.data) {
            per_head_rms_norm(st->q, (float*)layer->attn_q_norm.data, NH, HD, model->eps);
        }
        if (layer->attn_k_norm.data) {
            per_head_rms_norm(st->k, (float*)layer->attn_k_norm.data, NKH, HD, model->eps);
        }

        /* V projection */
        quant_matmul(&layer->attn_v, st->h, st->v);

        /* RoPE */
        apply_rope(st->q, pos, NH, HD, model->rope_freq_base);
        apply_rope(st->k, pos, NKH, HD, model->rope_freq_base);

        /* KV cache */
        int kv_dim = NKH * HD;
        memcpy(st->kv_k + (size_t)i * st->max_pos * kv_dim + (size_t)st->kv_len[i] * kv_dim,
               st->k, kv_dim * sizeof(float));
        memcpy(st->kv_v + (size_t)i * st->max_pos * kv_dim + (size_t)st->kv_len[i] * kv_dim,
               st->v, kv_dim * sizeof(float));

        /* GQA attention */
        int seq_len = st->kv_len[i] + 1;
        float scale = 1.0f / sqrtf((float)HD);
        int n_rep = NH / NKH;

        memset(st->att_out, 0, NH * HD * sizeof(float));

        for (int h = 0; h < NH; h++) {
            int kv_h = h / n_rep;  /* which KV head this Q head maps to */
            float* q_vec = st->q + h * HD;

            /* Compute attention scores for this head */
            float* scores_h = st->scores + h * st->max_pos;
            float max_score = -1e30f;

            for (int s = 0; s < seq_len; s++) {
                float* k_vec = st->kv_k + (size_t)i * st->max_pos * kv_dim + s * kv_dim + kv_h * HD;
                float dot = 0.0f;
                for (int d = 0; d < HD; d++) dot += q_vec[d] * k_vec[d];
                scores_h[s] = dot * scale;
                if (scores_h[s] > max_score) max_score = scores_h[s];
            }

            /* Softmax */
            float sum = 0.0f;
            for (int s = 0; s < seq_len; s++) {
                scores_h[s] = expf(scores_h[s] - max_score);
                sum += scores_h[s];
            }
            for (int s = 0; s < seq_len; s++) scores_h[s] /= sum;

            /* Weighted sum of V */
            float* out_vec = st->att_out + h * HD;
            for (int s = 0; s < seq_len; s++) {
                float* v_vec = st->kv_v + (size_t)i * st->max_pos * kv_dim + s * kv_dim + kv_h * HD;
                float w = scores_h[s];
                for (int d = 0; d < HD; d++) out_vec[d] += w * v_vec[d];
            }
        }

        /* Output projection + residual */
        quant_matmul(&layer->attn_out, st->att_out, st->h);  /* reuse h buffer */
        for (int d = 0; d < N; d++) st->h[d] = st->residual[d] + st->h[d];

        /* ── FFN / MoE ── */
        memcpy(st->residual, st->h, N * sizeof(float));
        rms_norm(st->h, (float*)layer->ffn_norm.data, st->h, N, model->eps);

        if (layer->is_moe) {
            /* ── MoE routing ── */
            float* gate_scores = st->gate;  /* reuse gate buffer for scores */
            quant_matmul(&layer->moe_gate_inp, st->h, gate_scores);

            /* Softmax */
            int n_exp = layer->n_experts;
            float max_g = gate_scores[0];
            for (int e = 1; e < n_exp; e++) if (gate_scores[e] > max_g) max_g = gate_scores[e];
            float sum_g = 0.0f;
            for (int e = 0; e < n_exp; e++) { gate_scores[e] = expf(gate_scores[e] - max_g); sum_g += gate_scores[e]; }
            for (int e = 0; e < n_exp; e++) gate_scores[e] /= sum_g;

            /* Top-K selection (simple O(n*k) for small k) */
            int top_k = layer->n_experts_per_tok;
            int top_indices[8];    /* max top_k = 8 */
            float top_weights[8];
            for (int k = 0; k < top_k; k++) {
                int best = -1;
                float best_val = -1e30f;
                for (int e = 0; e < n_exp; e++) {
                    if (gate_scores[e] > best_val) { best_val = gate_scores[e]; best = e; }
                }
                top_indices[k] = best;
                top_weights[k] = best_val;
                gate_scores[best] = -1e30f;  /* remove from consideration */
            }
            /* Re-normalize */
            float tw_sum = 0.0f;
            for (int k = 0; k < top_k; k++) tw_sum += top_weights[k];
            for (int k = 0; k < top_k; k++) top_weights[k] /= tw_sum;

            /* Compute selected experts */
            int ff_exp = layer->n_ff_expert;
            memset(st->ffn_out, 0, N * sizeof(float));

            for (int k = 0; k < top_k; k++) {
                int eid = top_indices[k];
                float w = top_weights[k];

                /* gate_proj */
                quant_matmul(&layer->moe_gate_exps[eid], st->h, st->gate);
                /* up_proj */
                quant_matmul(&layer->moe_up_exps[eid], st->h, st->up);

                /* SiLU(gate) * up */
                for (int j = 0; j < ff_exp && j < ff_exp; j++) {
                    st->silu_gate[j] = st->gate[j] / (1.0f + expf(-st->gate[j])) * st->up[j];
                }

                /* down_proj */
                quant_matmul(&layer->moe_down_exps[eid], st->silu_gate, st->expert_result);

                /* Accumulate weighted */
                for (int d = 0; d < N; d++) st->ffn_out[d] += w * st->expert_result[d];
            }

            /* Shared expert (Qwen3 MoE) */
            if (layer->has_shared_expert) {
                int shared_ff = layer->shared_down.ncols;  /* = 2 * n_ff_expert */
                quant_matmul(&layer->shared_gate, st->h, st->gate);
                quant_matmul(&layer->shared_up, st->h, st->up);
                int sff = layer->shared_gate.nrows;
                for (int j = 0; j < sff && j < sff; j++) {
                    st->gate[j] = st->gate[j] / (1.0f + expf(-st->gate[j]));
                    st->gate[j] *= st->up[j];
                }
                quant_matmul(&layer->shared_down, st->gate, st->expert_result);
                for (int d = 0; d < N; d++) st->ffn_out[d] += st->expert_result[d];
            }
        } else {
            /* ── Dense FFN (SwiGLU) ── */
            int FF = layer->ffn_gate.nrows;
            quant_matmul(&layer->ffn_gate, st->h, st->gate);
            quant_matmul(&layer->ffn_up, st->h, st->up);
            silu_inplace(st->gate, FF);
            for (int j = 0; j < FF; j++) st->silu_gate[j] = st->gate[j] * st->up[j];
            quant_matmul(&layer->ffn_down, st->silu_gate, st->ffn_out);
        }

        /* Residual connection */
        for (int d = 0; d < N; d++) st->h[d] = st->residual[d] + st->ffn_out[d];

        st->kv_len[i]++;
    }

    /* ── Final norm + output projection ── */
    rms_norm(st->h, (float*)model->output_norm.data, st->h, N, model->eps);
    quant_matmul(&model->output_weight, st->h, logits_out);
    st->pos++;
}

/* ── Helper: compute rope table ── */
void compute_rope_table(float* cos_table, float* sin_table,
                         int max_pos, int head_dim, float freq_base) {
    for (int p = 0; p < max_pos; p++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = powf(freq_base, -2.0f * i / head_dim);
            float angle = p * freq;
            cos_table[p * (head_dim/2) + i] = cosf(angle);
            sin_table[p * (head_dim/2) + i] = sinf(angle);
        }
    }
}

/* ── Utility: get/set OMP threads ── */
int get_max_threads(void) { return omp_get_max_threads(); }
void set_num_threads(int n) { omp_set_num_threads(n); }