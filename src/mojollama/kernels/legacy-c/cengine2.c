/* cengine2.c — Clean C forward pass. No struct, flat arrays.
 * 
 * engine_forward(tok, ...all weight pointers and dims as flat arrays..., logits)
 *
 * Uses only the call stack and explicit parameters — no struct layout worries.
 *
 * Compile:
 *   gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *       -o cengine2.so cengine2.c -lm
 */
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

/* Q4_0 dot — inline AVX2 */
static inline float q4d(const uint8_t *W, const float *x, int nc, int r) {
    int bp=nc/32; float total=0.0f;
    for(int b=0;b<bp;b++){
        const uint8_t *p=W+((size_t)r*bp+b)*Q4_0_BS;
        __builtin_prefetch(p + Q4_0_BS, 0, 0);  /* prefetch next block */
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
        int o=b*32; __m256 acc=_mm256_mul_ps(v0,_mm256_loadu_ps(x+o));
        acc=_mm256_fmadd_ps(v1,_mm256_loadu_ps(x+o+8),acc);
        acc=_mm256_fmadd_ps(v2,_mm256_loadu_ps(x+o+16),acc);
        acc=_mm256_fmadd_ps(v3,_mm256_loadu_ps(x+o+24),acc);
        __m128 lo128=_mm256_castps256_ps128(acc),hi128=_mm256_extractf128_ps(acc,1);
        lo128=_mm_add_ps(lo128,hi128);lo128=_mm_hadd_ps(lo128,lo128);lo128=_mm_hadd_ps(lo128,lo128);
        total+=_mm_cvtss_f32(lo128);
    }
    return total;
}

/* Q8_0 dot */
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

/* RMS norm */
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

/* GQA attention */
static void gqa(float *o, const float *q, const float *kc, const float *vc,
                int sl, int nh, int nkh, int hd){
    int gr=nh/nkh;
    #pragma omp parallel for schedule(static, 64)
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

/* ═════════════════════════════════════════════════════════════════════════
 * engine_forward — flat arrays API
 * All pointer arrays are [L] length
 * ═════════════════════════════════════════════════════════════════════════ */
void engine_forward(
    int tok,
    /* Model dims */
    int L, int N, int NH, int NKH, int HD, int FF, int V,
    float eps,
    /* Per-layer weight pointers [L] */
    const uint8_t **wQ, const uint8_t **wK, const uint8_t **wV,
    const uint8_t **wO, const uint8_t **wG, const uint8_t **wU, const uint8_t **wD,
    const float **wAN, const float **wFN,
    /* Per-layer row counts [L] */
    const int *nQ, const int *nK, const int *nV,
    const int *nO, const int *nG, const int *nU, const int *nD,
    int nc,
    /* Embed + output */
    const float *emb, const float *onw,
    const uint8_t *wOut, int outNR, int outNC,
    /* KV cache + mutable state */
    float *kvK, float *kvV, int *kvL, int MP,
    /* Workspace */
    float *x, float *xn, float *res,
    float *q, float *k, float *v,
    float *att, float *gate, float *up, float *silu,
    float *oproj, float *ffn,
    /* Output */
    float *logits
) {
    memcpy(x, emb + (size_t)tok * N, N*sizeof(float));

    for(int l=0;l<L;l++){
        memcpy(res, x, N*sizeof(float));
        rms(xn, x, wAN[l], N, eps);

        /* QKV batch */
        int tq=nQ[l]+nK[l]+nV[l];
        #pragma omp parallel for schedule(static, 64)
        for(int i=0;i<tq;i++){
            if(i<nQ[l]) q[i]=q4d(wQ[l],xn,nc,i);
            else if(i<nQ[l]+nK[l]) k[i-nQ[l]]=q4d(wK[l],xn,nc,i-nQ[l]);
            else v[i-nQ[l]-nK[l]]=q4d(wV[l],xn,nc,i-nQ[l]-nK[l]);
        }

        /* RoPE (on-the-fly with precomputed freq) */
        int pos=kvL[l];
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
        memcpy(kvK+((size_t)l*MP+kvL[l])*NKH*HD, k, NKH*HD*sizeof(float));
        memcpy(kvV+((size_t)l*MP+kvL[l])*NKH*HD, v, NKH*HD*sizeof(float));

        /* GQA attention */
        gqa(att, q, kvK+(size_t)l*MP*NKH*HD, kvV+(size_t)l*MP*NKH*HD, kvL[l]+1, NH, NKH, HD);
        kvL[l]++;

        /* O proj + residual */
        #pragma omp parallel for schedule(static, 64)
        for(int i=0;i<nO[l];i++) oproj[i]=q4d(wO[l],att,nc,i);
        for(int i=0;i<N;i++) x[i]=res[i]+oproj[i];

        /* FFN */
        memcpy(res, x, N*sizeof(float));
        rms(xn, x, wFN[l], N, eps);

        int tg=nG[l]+nU[l];
        #pragma omp parallel for schedule(static, 64)
        for(int i=0;i<tg;i++){
            if(i<nG[l]) gate[i]=q4d(wG[l],xn,nc,i);
            else up[i-nG[l]]=q4d(wU[l],xn,nc,i-nG[l]);}

        for(int i=0;i<FF;i++){float g=gate[i];silu[i]=(g/(1.0f+expf(-g)))*up[i];}

        #pragma omp parallel for schedule(static, 64)
        for(int i=0;i<nD[l];i++) ffn[i]=q4d(wD[l],silu,FF,i);
        for(int i=0;i<N;i++) x[i]=res[i]+ffn[i];
    }

    rms(xn, x, onw, N, eps);
    #pragma omp parallel for schedule(static, 64)
    for(int i=0;i<outNR;i++) logits[i]=q8d(wOut,xn,outNC,i);
}
