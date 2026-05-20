/* cengine_debug.c — Per-layer forward with debug prints. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#define Q4_0_BS 18
#define Q8_0_BS 34
static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static inline float hsum_ps(__m256 v) {
    __m128 l=_mm256_castps256_ps128(v),h=_mm256_extractf128_ps(v,1);
    l=_mm_add_ps(l,h);l=_mm_hadd_ps(l,l);l=_mm_hadd_ps(l,l);return _mm_cvtss_f32(l);
}
static inline float q4d(const uint8_t *W, const float *x, int nc, int r) {
    int bp=nc/32; float total=0.0f;
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
        int o=b*32; __m256 a=_mm256_mul_ps(v0,_mm256_loadu_ps(x+o));
        a=_mm256_fmadd_ps(v1,_mm256_loadu_ps(x+o+8),a);
        a=_mm256_fmadd_ps(v2,_mm256_loadu_ps(x+o+16),a);
        a=_mm256_fmadd_ps(v3,_mm256_loadu_ps(x+o+24),a);
        __m128 l=_mm256_castps256_ps128(a),h=_mm256_extractf128_ps(a,1);
        l=_mm_add_ps(l,h);l=_mm_hadd_ps(l,l);l=_mm_hadd_ps(l,l);
        total+=_mm_cvtss_f32(l);
    }
    return total;
}
static inline float q8d(const uint8_t *W, const float *x, int nc, int r) {
    int bp=nc/32; float t=0;
    for(int b=0;b<bp;b++){
        const uint8_t *p=W+((size_t)r*bp+b)*Q8_0_BS;
        float d=f16_to_f32((uint16_t)p[0]|((uint16_t)p[1]<<8));
        const int8_t *qs=(const int8_t*)(p+2);
        __m256 dv=_mm256_set1_ps(d),bs=_mm256_setzero_ps(); int o=b*32;
        for(int i=0;i<32;i+=8){
            __m128i q8=_mm_loadl_epi64((const __m128i*)(qs+i));
            __m256 v=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(q8))),dv);
            bs=_mm256_fmadd_ps(v,_mm256_loadu_ps(x+o+i),bs);}
        t+=hsum_ps(bs);
    }
    return t;
}
static inline void rms(float *o, const float *x, const float *w, int n, float e){
    __m256 ss=_mm256_setzero_ps(); int i;
    for(i=0;i<=n-8;i+=8){__m256 v=_mm256_loadu_ps(x+i);ss=_mm256_fmadd_ps(v,v,ss);}
    __m256 hh=_mm256_hadd_ps(ss,_mm256_permute2f128_ps(ss,ss,1));
    hh=_mm256_hadd_ps(hh,hh);hh=_mm256_hadd_ps(hh,hh);float sse=_mm256_cvtss_f32(hh);
    for(;i<n;i++)sse+=x[i]*x[i];float ir=1.0f/sqrtf(sse/n+e);__m256 iv=_mm256_set1_ps(ir);
    for(i=0;i<=n-8;i+=8){__m256 xv=_mm256_loadu_ps(x+i),wv=_mm256_loadu_ps(w+i);
        _mm256_storeu_ps(o+i,_mm256_mul_ps(_mm256_mul_ps(xv,iv),wv));}
    for(;i<n;i++)o[i]=x[i]*ir*w[i];
}
static void gqa(float *o, const float *q, const float *kc, const float *vc,
                int sl, int nh, int nkh, int hd){
    int gr=nh/nkh;
    #pragma omp parallel for schedule(static)
    for(int h=0;h<nh;h++){
        int kv=h/gr;const float *qh=q+h*hd,*kb=kc+kv*hd,*vb=vc+kv*hd;
        int ks=nkh*hd;float *oh=o+h*hd,sc[4096],mx=-1e30f,inv=1.0f/sqrtf(hd);
        for(int s=0;s<sl;s++){const float*ks_=kb+s*ks;__m256 su=_mm256_setzero_ps();int d;
            for(d=0;d<=hd-8;d+=8)su=_mm256_fmadd_ps(_mm256_loadu_ps(qh+d),_mm256_loadu_ps(ks_+d),su);
            float dot=hsum_ps(su);for(;d<hd;d++)dot+=qh[d]*ks_[d];dot*=inv;sc[s]=dot;if(dot>mx)mx=dot;}
        float se=0;for(int s=0;s<sl;s++){sc[s]=expf(sc[s]-mx);se+=sc[s];}float is=1.0f/se;
        memset(oh,0,hd*sizeof(float));
        for(int s=0;s<sl;s++){const float*vs_=vb+s*ks;float w=sc[s]*is;__m256 wv=_mm256_set1_ps(w);int d;
            for(d=0;d<=hd-8;d+=8){__m256 vv=_mm256_loadu_ps(vs_+d);
                __m256 ac=_mm256_loadu_ps(oh+d);_mm256_storeu_ps(oh+d,_mm256_fmadd_ps(wv,vv,ac));}
            for(;d<hd;d++)oh[d]+=w*vs_[d];}
    }
}
void process_layer(
    int layer_id, float *x,
    int L, int N, int NH, int NKH, int HD, int FF, float eps,
    const uint8_t *w_q, const uint8_t *w_k, const uint8_t *w_v,
    const uint8_t *w_o, const uint8_t *w_g, const uint8_t *w_u, const uint8_t *w_d,
    const float *w_an, const float *w_fn,
    int n_q, int n_k, int n_v, int n_o,    int n_g, int n_u, int n_d, int nc,
    int nc_down,
    float *kv_k, float *kv_v, int *kv_len, int max_pos,
    float *xn, float *res, float *q, float *k, float *v,
    float *att, float *gate, float *up, float *silu_buf,
    float *oproj, float *ffn
) {
    int pos = *kv_len;
    memcpy(res, x, N * sizeof(float));
    rms(xn, x, w_an, N, eps);
    fprintf(stderr, "L%d xn: %.6f %.6f\n", layer_id, xn[0], xn[1]);
    int tq = n_q + n_k + n_v;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<tq;i++){
        if(i<n_q) q[i]=q4d(w_q,xn,nc,i);
        else if(i<n_q+n_k) k[i-n_q]=q4d(w_k,xn,nc,i-n_q);
        else v[i-n_q-n_k]=q4d(w_v,xn,nc,i-n_q-n_k);
    }
    fprintf(stderr, "L%d q: %.6f %.6f\n", layer_id, q[0], q[1]);
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
    fprintf(stderr, "L%d q_rope: %.6f %.6f\n", layer_id, q[0], q[1]);
    memcpy(kv_k + ((size_t)layer_id*max_pos+pos)*NKH*HD, k, NKH*HD*sizeof(float));
    memcpy(kv_v + ((size_t)layer_id*max_pos+pos)*NKH*HD, v, NKH*HD*sizeof(float));
    gqa(att, q, kv_k+(size_t)layer_id*max_pos*NKH*HD, kv_v+(size_t)layer_id*max_pos*NKH*HD,
        pos+1, NH, NKH, HD);
    (*kv_len)++;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n_o;i++) oproj[i]=q4d(w_o,att,nc,i);
    for(int i=0;i<N;i++) x[i]=res[i]+oproj[i];
    fprintf(stderr, "L%d x_after_o: %.6f %.6f\n", layer_id, x[0], x[1]);
    memcpy(res, x, N*sizeof(float));
    rms(xn, x, w_fn, N, eps);
    int tg=n_g+n_u;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<tg;i++){
        if(i<n_g) gate[i]=q4d(w_g,xn,nc,i);
        else up[i-n_g]=q4d(w_u,xn,nc,i-n_g);}
    fprintf(stderr, "L%d gate[0:3]: %.6f %.6f %.6f\n", layer_id, gate[0], gate[1], gate[2]);
    for(int i=0;i<FF;i++){float g=gate[i];silu_buf[i]=(g/(1.0f+expf(-g)))*up[i];}
    fprintf(stderr, "L%d silu[0:3]: %.6f %.6f %.6f\n", layer_id, silu_buf[0], silu_buf[1], silu_buf[2]);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n_d;i++) ffn[i]=q4d(w_d,silu_buf,nc_down,i);
    fprintf(stderr, "L%d ffn[0:3]: %.6f %.6f %.6f\n", layer_id, ffn[0], ffn[1], ffn[2]);
    for(int i=0;i<N;i++) x[i]=res[i]+ffn[i];
    fprintf(stderr, "L%d x_final: %.6f %.6f\n", layer_id, x[0], x[1]);
}
void process_final(float *x, int N, int HD, float eps, const float *onw,
    const uint8_t *w_out, int out_nr, int out_nc,
    float *xn, float *logits) {
    rms(xn, x, onw, N, eps);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<out_nr;i++) logits[i]=q8d(w_out,xn,out_nc,i);
}
