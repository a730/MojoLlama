/* c_engine.c — Complete transformer forward pass with struct API.
 * 
 * engine_forward() does 22-layer TinyLlama forward pass in C.
 * Uses EngineConfig struct for clean API.
 *
 * Compile:
 *   gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *       -o c_engine.so c_engine.c -lm
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define Q4_0_BS 18
#define Q8_0_BS 34

/* ═══ AVX2 helpers ═════════════════════════════════════════════════════ */
static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static inline float hsum_ps(__m256 v) {
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); lo=_mm_hadd_ps(lo,lo); lo=_mm_hadd_ps(lo,lo);
    return _mm_cvtss_f32(lo);
}

/* ═══ Q4_0 dot (inline AVX2, no function pointers) ═══════════════════ */
static inline float q4_row(const uint8_t *W, const float *x, int nc, int r) {
    int bp=nc/32; __m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps();
    for(int b=0;b<bp;b++){
        const uint8_t *p=W+((size_t)r*bp+b)*Q4_0_BS;
        float sc=f16_to_f32((uint16_t)p[0]|((uint16_t)p[1]<<8));
        __m256 sv=_mm256_set1_ps(sc);
        __m128i nb=_mm_loadu_si128((const __m128i*)(p+2));
        __m128i lo=_mm_and_si128(nb,_mm_set1_epi8(15));
        __m128i hi=_mm_and_si128(_mm_srli_epi16(nb,4),_mm_set1_epi8(15));
        __m128i ls=_mm_sub_epi8(lo,_mm_set1_epi8(8)),hs=_mm_sub_epi8(hi,_mm_set1_epi8(8));
        __m256 v0=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(ls)));
        __m256 v1=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_shuffle_epi32(ls,0x4e))));
        __m256 v2=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(hs)));
        __m256 v3=_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_shuffle_epi32(hs,0x4e))));
        v0=_mm256_mul_ps(v0,sv);v1=_mm256_mul_ps(v1,sv);v2=_mm256_mul_ps(v2,sv);v3=_mm256_mul_ps(v3,sv);
        int o=b*32;
        s0=_mm256_fmadd_ps(v0,_mm256_loadu_ps(x+o),s0);
        s0=_mm256_fmadd_ps(v1,_mm256_loadu_ps(x+o+8),s0);
        s1=_mm256_fmadd_ps(v2,_mm256_loadu_ps(x+o+16),s1);
        s1=_mm256_fmadd_ps(v3,_mm256_loadu_ps(x+o+24),s1);
    }
    return hsum_ps(_mm256_add_ps(s0,s1));
}

/* Q8_0 dot for output projection */
static inline float q8_row(const uint8_t *W, const float *x, int nc, int r) {
    int bp=nc/32; float t=0;
    for(int b=0;b<bp;b++){
        const uint8_t *p=W+((size_t)r*bp+b)*Q8_0_BS;
        float d=f16_to_f32((uint16_t)p[0]|((uint16_t)p[1]<<8));
        const int8_t *qs=(const int8_t*)(p+2);
        __m256 dv=_mm256_set1_ps(d),bs=_mm256_setzero_ps(); int o=b*32;
        for(int i=0;i<32;i+=8){
            __m128i q8=_mm_loadl_epi64((const __m128i*)(qs+i));
            __m256 v=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(q8))),dv);
            bs=_mm256_fmadd_ps(v,_mm256_loadu_ps(x+o+i),bs);
        }
        t+=hsum_ps(bs);
    }
    return t;
}

/* ═══ RMS norm ══════════════════════════════════════════════════════ */
static inline void rms(float *o, const float *x, const float *w, int n, float e){
    __m256 ss=_mm256_setzero_ps(); int i;
    for(i=0;i<=n-8;i+=8){__m256 v=_mm256_loadu_ps(x+i);ss=_mm256_fmadd_ps(v,v,ss);}
    __m256 hh=_mm256_hadd_ps(ss,_mm256_permute2f128_ps(ss,ss,1));
    hh=_mm256_hadd_ps(hh,hh);hh=_mm256_hadd_ps(hh,hh);
    float sse=_mm256_cvtss_f32(hh); for(;i<n;i++)sse+=x[i]*x[i];
    float ir=1.0f/sqrtf(sse/n+e);__m256 iv=_mm256_set1_ps(ir);
    for(i=0;i<=n-8;i+=8){__m256 xv=_mm256_loadu_ps(x+i),wv=_mm256_loadu_ps(w+i);
        _mm256_storeu_ps(o+i,_mm256_mul_ps(_mm256_mul_ps(xv,iv),wv));}
    for(;i<n;i++)o[i]=x[i]*ir*w[i];
}

/* ═══ GQA attention ═══════════════════════════════════════════════════ */
static void gqa(float *o, const float *q, const float *kc, const float *vc,
                int sl, int nh, int nkh, int hd){
    int gr=nh/nkh;
    #pragma omp parallel for schedule(static)
    for(int h=0;h<nh;h++){
        int kv=h/gr;const float *qh=q+h*hd,*kb=kc+kv*hd,*vb=vc+kv*hd;
        int ks=nkh*hd;float *oh=o+h*hd,sc[4096],mx=-1e30f,inv=1.0f/sqrtf(hd);
        for(int s=0;s<sl;s++){const float*ks_=kb+s*ks;__m256 su=_mm256_setzero_ps();
            int d;for(d=0;d<=hd-8;d+=8)su=_mm256_fmadd_ps(_mm256_loadu_ps(qh+d),_mm256_loadu_ps(ks_+d),su);
            float dot=hsum_ps(su)*inv;for(;d<hd;d++)dot+=qh[d]*ks_[d];
            sc[s]=dot;if(dot>mx)mx=dot;}
        float se=0;for(int s=0;s<sl;s++){sc[s]=expf(sc[s]-mx);se+=sc[s];}
        float is=1.0f/se;memset(oh,0,hd*sizeof(float));
        for(int s=0;s<sl;s++){const float*vs_=vb+s*ks;float w=sc[s]*is;__m256 wv=_mm256_set1_ps(w);
            int d;for(d=0;d<=hd-8;d+=8){__m256 vv=_mm256_loadu_ps(vs_+d);
                __m256 ac=_mm256_loadu_ps(oh+d);_mm256_storeu_ps(oh+d,_mm256_fmadd_ps(wv,vv,ac));}
            for(;d<hd;d++)oh[d]+=w*vs_[d];}
    }
}

/* ═══ EngineConfig struct ════════════════════════════════════════════ */
typedef struct {
    int L,N,NH,NKH,HD,FF,V;
    float eps;
    /* Per-layer pointers [L] */
    const uint8_t **wQ,**wK,**wV,**wO,**wG,**wU,**wD;
    const float **wAN,**wFN;
    const int *nQ,*nK,*nV,*nO,*nG,*nU,*nD;
    int nc;
    /* Embed + output */
    const float *emb,*onw;
    const uint8_t *wOut; int outNR,outNC;
    /* KV cache + workspace (mutable) */
    float *kvK,*kvV; int *kvL; int MP;
    float *x,*xn,*res,*q,*k,*v,*att,*gate,*up,*silu,*oproj,*ffn;
} EC;

/* ═══ engine_forward — one token, no Python ═════════════════════════ */
void engine_forward(int tok, EC *c, float *logits) {
    int N=c->N,NH=c->NH,NKH=c->NKH,HD=c->HD,FF=c->FF,L=c->L,nc=c->nc;
    float *x=c->x,*xn=c->xn,*res=c->res,*q=c->q,*k=c->k,*v=c->v;
    float *att=c->att,*gate=c->gate,*up=c->up,*silu=c->silu,*o=c->oproj,*ffn=c->ffn;

    memcpy(x, c->emb + (size_t)tok * N, N*sizeof(float));

    for(int l=0;l<L;l++){
        memcpy(res, x, N*sizeof(float));
        rms(xn, x, c->wAN[l], N, c->eps);

        /* QKV batch */
        int tq=c->nQ[l]+c->nK[l]+c->nV[l];
        #pragma omp parallel for schedule(static)
        for(int i=0;i<tq;i++){
            if(i<c->nQ[l]) q[i]=q4_row(c->wQ[l],xn,nc,i);
            else if(i<c->nQ[l]+c->nK[l]) k[i-c->nQ[l]]=q4_row(c->wK[l],xn,nc,i-c->nQ[l]);
            else v[i-c->nQ[l]-c->nK[l]]=q4_row(c->wV[l],xn,nc,i-c->nQ[l]-c->nK[l]);
        }

        /* RoPE (pre-computed tables — cos_tbl/sin_tbl stored at end of EC) */
        int pos=c->kvL[l], hd=HD, h=hd/2;
        const float *ct=c->x+4096*h;  /* FIXME: use proper cos/sin pointers */
        /* Simple RoPE with on-the-fly computation (fast enough for <100 pos) */
        for(int hh=0;hh<NH;hh++){float *qh=q+hh*hd;
            for(int j=0;j<h;j++){
                double ang=(double)pos/pow(10000.0,2.0*j/hd);
                double cs=cos(ang),sn=sin(ang);
                float a=qh[j],b=qh[j+h];qh[j]=(float)(a*cs-b*sn);qh[j+h]=(float)(b*cs+a*sn);}}
        for(int hh=0;hh<NKH;hh++){float *kh=k+hh*hd;
            for(int j=0;j<h;j++){
                double ang=(double)pos/pow(10000.0,2.0*j/hd);
                double cs=cos(ang),sn=sin(ang);
                float a=kh[j],b=kh[j+h];kh[j]=(float)(a*cs-b*sn);kh[j+h]=(float)(b*cs+a*sn);}}

        /* KV store */
        memcpy(c->kvK+((size_t)l*c->MP+c->kvL[l])*NKH*HD, k, NKH*HD*sizeof(float));
        memcpy(c->kvV+((size_t)l*c->MP+c->kvL[l])*NKH*HD, v, NKH*HD*sizeof(float));

        /* GQA attention */
        int sl=c->kvL[l]+1;
        gqa(att, q, c->kvK+(size_t)l*c->MP*NKH*HD, c->kvV+(size_t)l*c->MP*NKH*HD,
            sl, NH, NKH, HD);
        c->kvL[l]++;

        /* O proj + residual */
        #pragma omp parallel for schedule(static)
        for(int i=0;i<c->nO[l];i++) o[i]=q4_row(c->wO[l],att,nc,i);
        for(int i=0;i<N;i++) x[i]=res[i]+o[i];

        /* FFN: norm2 */
        memcpy(res, x, N*sizeof(float));
        rms(xn, x, c->wFN[l], N, c->eps);

        /* Gate+Up batch */
        int tg=c->nG[l]+c->nU[l];
        #pragma omp parallel for schedule(static)
        for(int i=0;i<tg;i++){
            if(i<c->nG[l]) gate[i]=q4_row(c->wG[l],xn,nc,i);
            else up[i-c->nG[l]]=q4_row(c->wU[l],xn,nc,i-c->nG[l]);
        }

        /* SiLU * up */
        for(int i=0;i<FF;i++){float g=gate[i];silu[i]=(g/(1.0f+expf(-g)))*up[i];}

        /* Down + residual */
        #pragma omp parallel for schedule(static)
        for(int i=0;i<c->nD[l];i++) ffn[i]=q4_row(c->wD[l],silu,nc,i);
        for(int i=0;i<N;i++) x[i]=res[i]+ffn[i];
    }

    /* Final norm + output */
    rms(xn, x, c->onw, N, c->eps);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<c->outNR;i++) logits[i]=q8_row(c->wOut,xn,c->outNC,i);
}
