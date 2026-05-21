/* gptoss_batch_forward.c — Minimal C forward pass for gpt-oss-20b.
 * Eliminates Python overhead by running the full layer loop in C.
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o gptoss_batch_forward.so gptoss_batch_forward.c quant_kernels_omp.c gqa_attention.c -lm
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* Forward declarations from quant_kernels_omp.c */
extern void mxfp4_matmul_omp(const uint8_t *W, const float *x, float *out, int n_rows, int n_cols);
extern void moe_forward_omp(const uint8_t** gate_raw, const uint8_t** up_raw, const uint8_t** down_raw,
    const float* x_norm, int n_ff_expert, int n_embd,
    int qt_gate, int qt_up, int qt_down,
    const int* top_indices, const float* top_weights, int top_k,
    float* combined, float* prealloc_buf, uint8_t* prealloc_q8);
extern int quantize_row_q8_0(const float *restrict x, uint8_t *restrict q8, int n_cols);
extern void gqa_attention_decode(const float *q, const float *k_cache, const float *v_cache,
    float *out, int seq_len, int n_head, int n_kv_head, int head_dim, float *workspace);

#define MAX_EXPERTS 64
#define MAX_TOP_K 16

/* RMS norm */
static void rms_norm(float *out, const float *x, const float *w, int N, float eps) {
    float ss = 0;
    for (int i = 0; i < N; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / N + eps);
    for (int i = 0; i < N; i++) out[i] = (x[i] / rms) * w[i];
}

/* RoPE apply */
static void rope_apply(float *buf, int nh, int hd, int pos, const float *cos_t, const float *sin_t, int rope_dim) {
    if (rope_dim <= 0) rope_dim = hd;
    int hd2 = rope_dim / 2;
    const float *cos_row = cos_t + (size_t)pos * hd2;
    const float *sin_row = sin_t + (size_t)pos * hd2;
    for (int h = 0; h < nh; h++) {
        float *b = buf + (size_t)h * hd;
        for (int j = 0; j < hd2; j++) {
            float a0 = b[j], a1 = b[j + hd2];
            float cs = cos_row[j], sn = sin_row[j];
            b[j] = a0 * cs - a1 * sn;
            b[j + hd2] = a1 * cs + a0 * sn;
        }
    }
}

/* Router: matmul + softmax + top-k */
static void router_topk(const float *w_router, const float *x_norm, int n_experts, int N,
    int top_k, int *top_indices, float *top_weights, float *router_buf) {
    /* Router matmul: w_router is [n_experts, N], x_norm is [N] */
    for (int e = 0; e < n_experts; e++) {
        float dot = 0;
        const float *wr = w_router + (size_t)e * N;
        for (int i = 0; i < N; i++) dot += wr[i] * x_norm[i];
        router_buf[e] = dot;
    }
    /* Softmax */
    float mx = router_buf[0];
    for (int e = 1; e < n_experts; e++) if (router_buf[e] > mx) mx = router_buf[e];
    float sum = 0;
    for (int e = 0; e < n_experts; e++) {
        router_buf[e] = expf(router_buf[e] - mx);
        sum += router_buf[e];
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int e = 0; e < n_experts; e++) router_buf[e] *= inv_sum;
    /* Top-k */
    for (int k = 0; k < top_k; k++) { top_indices[k] = -1; top_weights[k] = -1e30f; }
    for (int e = 0; e < n_experts; e++) {
        float v = router_buf[e];
        for (int k = 0; k < top_k; k++) {
            if (v > top_weights[k]) {
                for (int k2 = top_k - 1; k2 > k; k2--) {
                    top_weights[k2] = top_weights[k2-1];
                    top_indices[k2] = top_indices[k2-1];
                }
                top_weights[k] = v;
                top_indices[k] = e;
                break;
            }
        }
    }
    /* Renormalize top weights */
    float tw_sum = 0;
    for (int k = 0; k < top_k; k++) tw_sum += top_weights[k];
    float tw_inv = 1.0f / (tw_sum + 1e-10f);
    for (int k = 0; k < top_k; k++) top_weights[k] *= tw_inv;
}

/* GPT-OSS forward pass */
void gptoss_forward(
    const int *token, int pos,
    const float *emb, int V,
    const uint8_t **wQ, const uint8_t **wK, const uint8_t **wV, const uint8_t **wO,
    const float **wAN, const float **wFN,
    const int *nQ, const int *nK, const int *nV, const int *nO,
    const int *q_quant, const int *k_quant, const int *v_quant, const int *o_quant,
    const uint8_t *wOut, int outNR, int outNC, int outQuant,
    float *k_cache, float *v_cache, int *kv_len,
    const float *cos_table, const float *sin_table, int max_ctx, int rope_dim,
    float *workspace, int ws_size,
    int L, int N, int NH, int NKH, int HD, int FF, float eps,
    int n_experts, int n_experts_per_tok, int moe_intermediate,
    const float **w_gate_inp, const uint8_t **w_gate_exps, const uint8_t **w_up_exps, const uint8_t **w_down_exps,
    int gate_exp_quant, int up_exp_quant, int down_exp_quant,
    float *logits,
    float *moe_prealloc_buf, uint8_t *moe_prealloc_q8,
    float *gqa_ws
) {
    /* Workspace layout: x, x_norm, residual, q, k, v, att_out, gate, up, silu, o_proj, ffn_out, router_buf */
    int inner = NH * HD;
    int S = N; if (inner > S) S = inner; if (FF > S) S = FF; if (NKH * HD > S) S = NKH * HD;
    if (S < 8192) S = 8192;
    float *x = workspace;
    float *xn = x + S;
    float *res = xn + S;
    float *q = res + S;
    float *k = q + inner;
    float *v = k + NKH * HD;
    float *att = v + NKH * HD;
    float *gate = att + inner;
    float *up = gate + FF;
    float *silu = up + FF;
    float *oproj = silu + FF;
    float *ffn = oproj + N;
    float *router_buf = ffn + N;

    int top_indices[MAX_TOP_K];
    float top_weights[MAX_TOP_K];

    /* Embedding lookup */
    memcpy(x, emb + (size_t)(*token) * N, N * sizeof(float));

    for (int l = 0; l < L; l++) {
        /* Copy to residual */
        memcpy(res, x, N * sizeof(float));

        /* RMS norm 1 */
        rms_norm(xn, x, wAN[l], N, eps);

        /* QKV matmuls */
        mxfp4_matmul_omp(wQ[l], xn, q, nQ[l], N);
        mxfp4_matmul_omp(wK[l], xn, k, nK[l], N);
        mxfp4_matmul_omp(wV[l], xn, v, nV[l], N);

        /* RoPE */
        rope_apply(q, NH, HD, pos, cos_table, sin_table, rope_dim);
        rope_apply(k, NKH, HD, pos, cos_table, sin_table, rope_dim);

        /* KV cache store */
        int sl = kv_len[l];
        memcpy(k_cache + (size_t)l * max_ctx * NKH * HD + (size_t)sl * NKH * HD, k, NKH * HD * sizeof(float));
        memcpy(v_cache + (size_t)l * max_ctx * NKH * HD + (size_t)sl * NKH * HD, v, NKH * HD * sizeof(float));
        kv_len[l]++;
        sl++;

        /* GQA Attention */
        gqa_attention_decode(q, k_cache + (size_t)l * max_ctx * NKH * HD,
            v_cache + (size_t)l * max_ctx * NKH * HD, att, sl, NH, NKH, HD, gqa_ws);

        /* Output projection + residual */
        mxfp4_matmul_omp(wO[l], att, oproj, nO[l], inner);
        for (int i = 0; i < N; i++) x[i] = res[i] + oproj[i];

        /* RMS norm 2 */
        rms_norm(xn, x, wFN[l], N, eps);

        /* MoE: router + top-k */
        router_topk(w_gate_inp[l], xn, n_experts, N, n_experts_per_tok,
            top_indices, top_weights, router_buf);

        /* MoE FFN */
        moe_forward_omp(w_gate_exps + l, w_up_exps + l, w_down_exps + l,
            xn, moe_intermediate, N,
            gate_exp_quant, up_exp_quant, down_exp_quant,
            top_indices, top_weights, n_experts_per_tok, ffn, moe_prealloc_buf, moe_prealloc_q8);

        /* Residual */
        for (int i = 0; i < N; i++) x[i] = res[i] + ffn[i];
    }

    /* Output projection */
    mxfp4_matmul_omp(wOut, xn, logits, outNR, N);
}
