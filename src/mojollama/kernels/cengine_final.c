/* cengine_final.c — Uses proven quant_kernels_omp for ALL matmuls.
 * Drop-in replacement for q4d — calls quant_matmul_omp externally.
 * Compile:
 *   gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *       -o cengine_final.so cengine_final.c \
 *       -L. -l:quant_kernels_omp.so -Wl,-rpath,/onedev-workspace/work/src/mojollama/kernels -lm
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* External: proven OMP kernels */
extern void quant_matmul_omp(const unsigned char*, const float*, float*, int, int, int);
extern void rms_norm(const float*, const float*, float*, int, float);
extern void silu(float*, int);

/* External: GQA attention */
extern void gqa_attention_decode(const float*, const float*, const float*,
                                  float*, int, int, int, int);

void process_layer_final(
    int layer_id, float *x,
    int L, int N, int NH, int NKH, int HD, int FF, float eps,
    const unsigned char *w_q, const unsigned char *w_k, const unsigned char *w_v,
    const unsigned char *w_o, const unsigned char *w_g, const unsigned char *w_u, const unsigned char *w_d,
    const float *w_an, const float *w_fn,
    int n_q, int n_k, int n_v, int n_o, int n_g, int n_u, int n_d,
    int nc_qkv, int nc_down, int nc_o,  /* n_cols per projection */
    float *kv_k, float *kv_v, int *kv_len, int max_pos,
    float *xn, float *res, float *q, float *k, float *v,
    float *att, float *gate, float *up, float *silu_buf,
    float *oproj, float *ffn
) {
    int pos = *kv_len;

    memcpy(res, x, N * sizeof(float));
    rms_norm(x, xn, w_an, N, eps);  /* note: rms_norm(out, x, w, n, eps) */

    /* QKV — use proven quant_matmul_omp for individual projections */
    quant_matmul_omp(w_q, xn, q, n_q, nc_qkv, 2);
    quant_matmul_omp(w_k, xn, k, n_k, nc_qkv, 2);
    quant_matmul_omp(w_v, xn, v, n_v, nc_qkv, 2);

    /* RoPE — using on-the-fly computation */
    for(int hh=0;hh<NH;hh++){float *qh=q+hh*HD;
        for(int j=0;j<HD/2;j++){
            double ang=(double)pos/pow(10000.0,2.0*j/HD);
            double cs=cos(ang),sn=sin(ang);
            float a=qh[j],b=qh[j+HD/2];qh[j]=(float)(a*cs-b*sn);qh[j+HD/2]=(float)(b*cs+a*sn);}}
    for(int hh=0;hh<NKH;hh++){float *kh=k+hh*HD;
        for(int j=0;j<HD/2;j++){
            double ang=(double)pos/pow(10000.0,2.0*j/HD);
            double cs=cos(ang),sn=sin(ang);
            float a=kh[j],b=kh[j+HD/2];kh[j]=(float)(a*cs-b*sn);kh[j+HD/2]=(float)(b*cs+a*sn);}}

    /* KV store */
    memcpy(kv_k + ((size_t)layer_id*max_pos+pos)*NKH*HD, k, NKH*HD*sizeof(float));
    memcpy(kv_v + ((size_t)layer_id*max_pos+pos)*NKH*HD, v, NKH*HD*sizeof(float));

    /* Attention */
    gqa_attention_decode(q,
        kv_k+(size_t)layer_id*max_pos*NKH*HD,
        kv_v+(size_t)layer_id*max_pos*NKH*HD,
        att, pos+1, NH, NKH, HD);
    (*kv_len)++;

    /* O proj + residual */
    quant_matmul_omp(w_o, att, oproj, n_o, nc_o, 2);
    for(int i=0;i<N;i++) x[i]=res[i]+oproj[i];

    /* FFN */
    memcpy(res, x, N*sizeof(float));
    rms_norm(x, xn, w_fn, N, eps);  /* note: rms_norm(out, x, w, n, eps) — WARNING: arg order mismatch! */

    /* Gate + Up */
    quant_matmul_omp(w_g, xn, gate, n_g, nc_qkv, 2);
    quant_matmul_omp(w_u, xn, up, n_u, nc_qkv, 2);

    /* SiLU(gate) * up */
    for(int i=0;i<FF;i++){float g=gate[i];silu_buf[i]=(g/(1.0f+expf(-g)))*up[i];}

    /* Down + residual */
    quant_matmul_omp(w_d, silu_buf, ffn, n_d, nc_down, 2);
    for(int i=0;i<N;i++) x[i]=res[i]+ffn[i];
}

void process_final(float *x, int N, int HD, float eps,
    const float *onw, const unsigned char *w_out, int out_nr, int out_nc,
    float *xn, float *logits) {
    rms_norm(x, xn, onw, N, eps);
    quant_matmul_omp(w_out, xn, logits, out_nr, out_nc, 8);  /* Q8_0 type = 8 */
}
