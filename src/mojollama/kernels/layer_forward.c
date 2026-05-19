/* layer_forward.c — Complete per-layer forward pass in C
 *
 * Eliminates all Python overhead from the hot path by performing:
 *   RMS norm, QKV matmul, RoPE, KV cache store,
 *   GQA attention (softmax QK^T, weighted V sum),
 *   O projection + residual, RMS norm 2,
 *   Gate+Up matmul, SiLU × multiply, Down projection + residual
 *
 * All in a single C call per layer.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* ═════════════════════════════════════════════════════════════════════════
 * SIMD helpers (from simd_ops.c)
 * ═════════════════════════════════════════════════════════════════════════ */

static inline float f16_to_f32(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((uint32_t)h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

static inline void rms_norm_f32(float *restrict out, const float *restrict x,
                                 const float *restrict w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * w[i];
}

static inline void silu_f32(float *restrict out, const float *restrict x, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static inline void vec_mul_f32(float *restrict out, const float *restrict a,
                                const float *restrict b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}

static inline void vec_add_f32(float *restrict out, const float *restrict a,
                                const float *restrict b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static inline void apply_rope_f32(float *restrict q, float *restrict k,
                                   int pos, int n_head, int n_kv_head, int head_dim) {
    int half = head_dim / 2;
    for (int h = 0; h < n_head; h++) {
        float *qh = q + h * head_dim;
        for (int j = 0; j < half; j++) {
            double freq = 1.0 / pow(10000.0, (2.0 * j) / (double)head_dim);
            double angle = (double)pos * freq;
            double cos_a = cos(angle);
            double sin_a = sin(angle);
            float x0 = qh[j];
            float x1 = qh[j + half];  /* NOTE: RoPE pairs (j, j+half) NOT (2j, 2j+1) */
            qh[j]       = (float)((double)x0 * cos_a - (double)x1 * sin_a);
            qh[j + half] = (float)((double)x1 * cos_a + (double)x0 * sin_a);
        }
    }
    for (int h = 0; h < n_kv_head; h++) {
        float *kh = k + h * head_dim;
        for (int j = 0; j < half; j++) {
            double freq = 1.0 / pow(10000.0, (2.0 * j) / (double)head_dim);
            double angle = (double)pos * freq;
            double cos_a = cos(angle);
            double sin_a = sin(angle);
            float x0 = kh[j];
            float x1 = kh[j + half];
            kh[j]       = (float)((double)x0 * cos_a - (double)x1 * sin_a);
            kh[j + half] = (float)((double)x1 * cos_a + (double)x0 * sin_a);
        }
    }
}

/* GQA attention in C — eliminates numpy einsum overhead
 * q:   [n_head, head_dim]     (row-major)
 * k_cache: [seq_len, n_kv_head, head_dim]
 * v_cache: [seq_len, n_kv_head, head_dim]
 * out: [n_head * head_dim]
 * gqa_rep: number of Q heads per KV head (e.g., 4 for Llama-3.2-1B)
 */
static inline void gqa_attention_f32(
    const float *restrict q, const float *restrict k_cache,
    const float *restrict v_cache, float *restrict out,
    int seq_len, int n_head, int n_kv_head, int head_dim, int gqa_rep) {

    float scale = 1.0f / sqrtf((float)head_dim);
    int nq = n_head;
    int nkv = n_kv_head;

    /* Temporary storage for scores: n_head × seq_len */
    /* We process one Q head at a time to keep memory footprint small */
    float *scores = (float *)malloc(seq_len * sizeof(float));

    for (int h = 0; h < nq; h++) {
        int kv_h = h / gqa_rep;  /* which KV head this Q head attends to */
        const float *qh = q + h * head_dim;
        const float *kh_base = k_cache + kv_h * head_dim;  /* [seq_len, head_dim] stride=nkv*HD */
        int kv_stride = nkv * head_dim;

        /* Compute scores: Q × K^T / sqrt(d) */
        float max_score = -1e30f;
        for (int s = 0; s < seq_len; s++) {
            const float *ks = kh_base + s * kv_stride;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += qh[d] * ks[d];
            }
            scores[s] = dot * scale;
            if (scores[s] > max_score) max_score = scores[s];
        }

        /* Softmax */
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        float inv_sum = 1.0f / sum;
        for (int s = 0; s < seq_len; s++) {
            scores[s] *= inv_sum;
        }

        /* Weighted V sum: out[h] = sum_s scores[s] * V[s, kv_h, :] */
        float *oh = out + h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));
        const float *vh_base = v_cache + kv_h * head_dim;
        for (int s = 0; s < seq_len; s++) {
            const float *vs = vh_base + s * kv_stride;
            float w = scores[s];
            for (int d = 0; d < head_dim; d++) {
                oh[d] += w * vs[d];
            }
        }
    }
    free(scores);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Extern declarations — matmul kernels from quant_kernels_omp.so
 * ═════════════════════════════════════════════════════════════════════════ */

extern void quant_matmul_omp(const uint8_t *W, const float *x, float *out,
                              int n_rows, int n_cols, int quant_type);
extern void batch_qkv_omp(const uint8_t *Wq, const uint8_t *Wk, const uint8_t *Wv,
                            const float *x, float *out_q, float *out_k, float *out_v,
                            int nr_q, int nr_k, int nr_v, int n_cols,
                            int qt_q, int qt_k, int qt_v);
extern void batch_gate_up_omp(const uint8_t *Wgate, const uint8_t *Wup,
                               const float *x, float *out_gate, float *out_up,
                               int nr_gate, int nr_up, int n_cols,
                               int qt_gate, int qt_up);

/* ═════════════════════════════════════════════════════════════════════════
 * Layer forward parameters struct
 * ═════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Weight pointers (quantized) */
    const uint8_t *attn_q_raw;
    const uint8_t *attn_k_raw;
    const uint8_t *attn_v_raw;
    const uint8_t *attn_out_raw;
    const uint8_t *ffn_gate_raw;
    const uint8_t *ffn_up_raw;
    const uint8_t *ffn_down_raw;

    /* Norm weights */
    const float *attn_norm_w;
    const float *ffn_norm_w;

    /* Quant types */
    int qt_q, qt_k, qt_v, qt_out, qt_gate, qt_up, qt_down;

    /* Dimensions */
    int nr_q, nr_k, nr_v, nr_out, nr_gate, nr_up, nr_down;
    int nc;  /* n_embd (same for all) */

    /* Flags */
    int use_c_qkv;  /* all 3 QKV use C kernels */
    int use_c_gate_up;  /* both gate+up use C kernels */
} LayerParams;

/* ═════════════════════════════════════════════════════════════════════════
 * Full layer forward pass — single C call per layer
 * ═════════════════════════════════════════════════════════════════════════ */

void layer_forward(
    const LayerParams *lp,
    /* Buffers — caller pre-allocates */
    float *restrict x,          /* [n_embd] input/output */
    float *restrict x_norm,     /* [n_embd] temp */
    float *restrict residual,   /* [n_embd] temp */
    float *restrict q,          /* [n_head * head_dim] */
    float *restrict k,          /* [n_kv_head * head_dim] */
    float *restrict v,          /* [n_kv_head * head_dim] */
    float *restrict att_out,    /* [n_head * head_dim] */
    float *restrict o_proj,     /* [n_embd] */
    float *restrict gate,       /* [n_ff] */
    float *restrict up,         /* [n_ff] */
    float *restrict silu_gate,  /* [n_ff] */
    float *restrict ffn_out,    /* [n_embd] */
    /* KV cache */
    float *restrict kv_k,       /* [max_seq, n_kv_head, head_dim] */
    float *restrict kv_v,       /* [max_seq, n_kv_head, head_dim] */
    /* Model parameters */
    int pos,                    /* current position */
    int n_embd,
    int n_head,
    int n_kv_head,
    int head_dim,
    int n_ff,
    float eps,
    int kv_len                   /* current KV cache length (before this token) */
) {
    int NKH = n_kv_head * head_dim;
    int NH = n_head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* 1. Residual copy */
    memcpy(residual, x, n_embd * sizeof(float));

    /* 2. RMS norm 1 */
    rms_norm_f32(x_norm, x, lp->attn_norm_w, n_embd, eps);

    /* 3. QKV projections */
    if (lp->use_c_qkv) {
        batch_qkv_omp(lp->attn_q_raw, lp->attn_k_raw, lp->attn_v_raw,
                       x_norm, q, k, v,
                       lp->nr_q, lp->nr_k, lp->nr_v, lp->nc,
                       lp->qt_q, lp->qt_k, lp->qt_v);
    }
    /* Fallback for mixed quant types not implemented — use Python path */

    /* 4. RoPE */
    apply_rope_f32(q, k, pos, n_head, n_kv_head, head_dim);

    /* 5. Store K, V in cache */
    int kv_idx = kv_len;
    memcpy(kv_k + kv_idx * NKH, k, NKH * sizeof(float));
    memcpy(kv_v + kv_idx * NKH, v, NKH * sizeof(float));

    /* 6. GQA Attention */
    int seq_len = kv_len + 1;
    int gqa_rep = n_head / n_kv_head;
    gqa_attention_f32(q, kv_k, kv_v, att_out,
                      seq_len, n_head, n_kv_head, head_dim, gqa_rep);

    /* 7. O projection + residual */
    quant_matmul_omp(lp->attn_out_raw, att_out, o_proj,
                     lp->nr_out, lp->nc, lp->qt_out);
    vec_add_f32(x, residual, o_proj, n_embd);

    /* 8. Residual copy */
    memcpy(residual, x, n_embd * sizeof(float));

    /* 9. RMS norm 2 */
    rms_norm_f32(x_norm, x, lp->ffn_norm_w, n_embd, eps);

    /* 10. Gate + Up projections */
    if (lp->use_c_gate_up) {
        batch_gate_up_omp(lp->ffn_gate_raw, lp->ffn_up_raw,
                          x_norm, gate, up,
                          lp->nr_gate, lp->nr_up, lp->nc,
                          lp->qt_gate, lp->qt_up);
    }

    /* 11. SiLU(gate) * up */
    silu_f32(silu_gate, gate, n_ff);
    vec_mul_f32(silu_gate, silu_gate, up, n_ff);

    /* 12. Down projection + residual */
    quant_matmul_omp(lp->ffn_down_raw, silu_gate, ffn_out,
                     lp->nr_down, n_ff, lp->qt_down);
    vec_add_f32(x, residual, ffn_out, n_embd);
}