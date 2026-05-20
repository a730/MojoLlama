/* Test gemma4_forward_c with correct dimensions */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void gemma4_forward_c(
    const int *tokens, int B,
    int L, int N, int NH, int NKH, int V, int PL, int SW, int FF_half,
    float eps, float logit_cap,
    const float *emb,
    const unsigned char *w_out, int qt_out, int nr_out, int nc_out,
    const float *out_norm_w,
    const float **attn_norm_w, const float **ffn_norm_w,
    const float **post_attn_norm_w, const float **post_ffw_norm_w,
    const float **post_norm_w, const float **layer_scale,
    const float **q_norm_w, const float **k_norm_w,
    const float **proj_w, const float **inp_gate_w,
    const unsigned char **wq, const int *q_nr, const int *q_qt,
    const unsigned char **wk, const int *k_nr, const int *k_qt,
    const unsigned char **wv, const int *v_nr, const int *v_qt,
    const unsigned char **wo, const int *o_nr, const int *o_qt,
    const unsigned char **wg, const int *g_nr, const int *g_qt,
    const unsigned char **wu, const int *u_nr, const int *u_qt,
    const unsigned char **wd, const int *d_nr, const int *d_qt,
    const int *head_dims, const int *is_swa_arr,
    const int *kv_idx_arr, const int *rope_dim_arr,
    const float *freq_base_arr,
    float *kv_k, float *kv_v, int *kv_lens, int max_ctx,
    const float *cos_rope, const float *sin_rope,
    float *logits,
    float *ws);

int main() {
    int L=4; /* Just 4 layers for testing */
    int N=1536, NH=8, NKH=1, V=262144, PL=256, SW=512, FF_half=6144;
    int hd_default = N / NH; /* 192 */
    
    fprintf(stderr, "1. alloc\n");
    float *emb = calloc(N, sizeof(float));
    float *out_norm = calloc(N, sizeof(float));
    for (int i = 0; i < N; i++) out_norm[i] = 1.0f;
    emb[0] = 1.0f; /* token 1 -> first element */
    
    /* Per-layer arrays */
    float **attn_norm = calloc(L, sizeof(float*));
    float **ffn_norm = calloc(L, sizeof(float*));
    float **q_norm = calloc(L, sizeof(float*));
    float **k_norm = calloc(L, sizeof(float*));
    unsigned char **wq = calloc(L, sizeof(unsigned char*));
    unsigned char **wk = calloc(L, sizeof(unsigned char*));
    unsigned char **wv = calloc(L, sizeof(unsigned char*));
    unsigned char **wo = calloc(L, sizeof(unsigned char*));
    unsigned char **wg = calloc(L, sizeof(unsigned char*));
    unsigned char **wu = calloc(L, sizeof(unsigned char*));
    unsigned char **wd = calloc(L, sizeof(unsigned char*));
    unsigned char **wnull = calloc(L, sizeof(unsigned char*));
    float **fnull = calloc(L, sizeof(float*));
    
    /* Output dimensions */
    int *nq = calloc(L, sizeof(int));
    int *nk = calloc(L, sizeof(int));
    int *nv = calloc(L, sizeof(int));
    int *no = calloc(L, sizeof(int));
    int *ng = calloc(L, sizeof(int));
    int *nu = calloc(L, sizeof(int));
    int *nd = calloc(L, sizeof(int));
    int *qt_v = calloc(L, sizeof(int));
    int *hd = calloc(L, sizeof(int));
    int *sw = calloc(L, sizeof(int));
    int *kvi = calloc(L, sizeof(int));
    int *rd = calloc(L, sizeof(int));
    float *fb = calloc(L, sizeof(float));
    
    fprintf(stderr, "2. fill metadata\n");
    for (int i = 0; i < L; i++) {
        int hdi = (i < 2) ? 192 : 128;
        attn_norm[i] = calloc(N, sizeof(float));
        ffn_norm[i] = calloc(N, sizeof(float));
        q_norm[i] = calloc(NH * hdi, sizeof(float));
        k_norm[i] = calloc(NKH * hdi, sizeof(float));
        for (int j = 0; j < N; j++) { attn_norm[i][j] = 1.0f; ffn_norm[i][j] = 1.0f; }
        for (int j = 0; j < NH * hdi; j++) q_norm[i][j] = 1.0f;
        for (int j = 0; j < NKH * hdi; j++) k_norm[i][j] = 1.0f;
        
        /* Weight buffers (Q8_0 = type 2) */
        int wq_size = (NH*hdi)*N/32*34;
        int wk_size = (NKH*hdi)*N/32*34;
        int wv_size = (NKH*hdi)*N/32*34;
        int wo_size = N*(NH*hdi)/32*34;
        int wg_size = FF_half*N/32*34;
        int wu_size = FF_half*N/32*34;
        int wd_size = N*FF_half/32*34;
        
        wq[i] = calloc(wq_size, 1); wk[i] = calloc(wk_size, 1);
        wv[i] = calloc(wv_size, 1); wo[i] = calloc(wo_size, 1);
        wg[i] = calloc(wg_size, 1); wu[i] = calloc(wu_size, 1);
        wd[i] = calloc(wd_size, 1);
        
        /* Set Q8_0 block headers: scale = 0 (so dot=0 output=0) */
        unsigned char *p;
        p = wq[i]; for (int b = 0; b < wq_size/34; b++) *(float*)(p+b*34+32) = 0.0f;
        p = wk[i]; for (int b = 0; b < wk_size/34; b++) *(float*)(p+b*34+32) = 0.0f;
        p = wv[i]; for (int b = 0; b < wv_size/34; b++) *(float*)(p+b*34+32) = 0.0f;
        p = wo[i]; for (int b = 0; b < wo_size/34; b++) *(float*)(p+b*34+32) = 0.0f;
        p = wg[i]; for (int b = 0; b < wg_size/34; b++) *(float*)(p+b*34+32) = 0.0f;
        p = wu[i]; for (int b = 0; b < wu_size/34; b++) *(float*)(p+b*34+32) = 0.0f;
        p = wd[i]; for (int b = 0; b < wd_size/34; b++) *(float*)(p+b*34+32) = 0.0f;
        
        nq[i] = NH * hdi;  /* 8 * hdi */
        nk[i] = NKH * hdi; /* 1 * hdi */
        nv[i] = NKH * hdi;
        no[i] = N;
        ng[i] = FF_half;  /* 6144 */
        nu[i] = FF_half;
        nd[i] = N;
        qt_v[i] = 2; /* Q8_0 */
        
        hd[i] = hdi;
        sw[i] = (i < 2) ? 1 : 0;
        kvi[i] = i;
        rd[i] = hdi;
        fb[i] = 1000000.0f;
    }
    
    /* Output weights */
    unsigned char *w_out = calloc(V * N / 32 * 34, 1);
    /* TODO: verify correct size */
    
    fprintf(stderr, "3. KV\n");
    float *kv_k = calloc(35 * 8192 * 192, sizeof(float));
    float *kv_v = calloc(35 * 8192 * 192, sizeof(float));
    int *kv_len = calloc(35, sizeof(int));
    
    fprintf(stderr, "4. RoPE\n");
    int mpos = 8192;
    float *cos_t = calloc(mpos * 512, sizeof(float));
    float *sin_t = calloc(mpos * 512, sizeof(float));
    for (int p = 0; p < mpos; p++)
        for (int j = 0; j < 256; j++) {
            float t = p / powf(1000000.0f, 2.0f*j/512.0f);
            cos_t[p*512+j] = cosf(t); sin_t[p*512+j] = sinf(t);
        }
    
    fprintf(stderr, "5. out\n");
    float *logits = calloc(V, sizeof(float));
    int S = 8192;
    float *ws = calloc(14*S, sizeof(float));
    
    fprintf(stderr, "6. ptr arrays\n");
    const float **ca_an = calloc(L, sizeof(float*));
    const float **ca_fn = calloc(L, sizeof(float*));
    const float **ca_qn = calloc(L, sizeof(float*));
    const float **ca_kn = calloc(L, sizeof(float*));
    const float **ca_nul = calloc(L, sizeof(float*));
    const unsigned char **ca_wq = calloc(L, sizeof(unsigned char*));
    const unsigned char **ca_wk = calloc(L, sizeof(unsigned char*));
    const unsigned char **ca_wv = calloc(L, sizeof(unsigned char*));
    const unsigned char **ca_wo = calloc(L, sizeof(unsigned char*));
    const unsigned char **ca_wg = calloc(L, sizeof(unsigned char*));
    const unsigned char **ca_wu = calloc(L, sizeof(unsigned char*));
    const unsigned char **ca_wd = calloc(L, sizeof(unsigned char*));
    
    for (int i = 0; i < L; i++) {
        ca_an[i] = attn_norm[i];
        ca_fn[i] = ffn_norm[i];
        ca_qn[i] = q_norm[i];
        ca_kn[i] = k_norm[i];
        ca_nul[i] = NULL;
        ca_wq[i] = wq[i]; ca_wk[i] = wk[i];
        ca_wv[i] = wv[i]; ca_wo[i] = wo[i];
        ca_wg[i] = wg[i]; ca_wu[i] = wu[i];
        ca_wd[i] = wd[i];
    }
    
    fprintf(stderr, "7. call gemma4_forward_c (L=%d)...\n", L); fflush(stderr);
    
    gemma4_forward_c(
        (const int[]){1}, 1,
        L, N, NH, NKH, V, PL, SW, FF_half,
        1e-6f, 30.0f,
        emb,
        w_out, 2, V*N, N,  /* Q8_0, nr=V*N, nc=N */
        out_norm,
        ca_an, ca_fn, ca_nul, ca_nul, ca_nul, ca_nul,
        ca_qn, ca_kn, ca_nul, ca_nul,
        ca_wq, nq, qt_v,
        ca_wk, nk, qt_v,
        ca_wv, nv, qt_v,
        ca_wo, no, qt_v,
        ca_wg, ng, qt_v,
        ca_wu, nu, qt_v,
        ca_wd, nd, qt_v,
        hd, sw, kvi, rd, fb,
        kv_k, kv_v, kv_len, 8192,
        cos_t, sin_t,
        logits, ws);
    
    fprintf(stderr, "8. DONE! logits[0]=%f max=%f\n", logits[0], logits[0]);
    return 0;
}
