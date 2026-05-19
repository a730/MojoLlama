/* cengine_batch.c — vLLM-style concurrent batched inference.
 * Uses q4_0_batch_matmul for all weight projections across B tokens.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#define Q4_0_BS 18
#define Q8_0_BS 34
#define BLOCK_SIZE 64
#define MAX_BLOCKS 1024

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static inline float hsum_ps(__m256 v) {
    __m128 l=_mm256_castps256_ps128(v),h=_mm256_extractf128_ps(v,1);
    l=_mm_add_ps(l,h);l=_mm_hadd_ps(l,l);l=_mm_hadd_ps(l,l);return _mm_cvtss_f32(l);
}

/* AVX2-accelerated Batched Q4_0 matmul */
void q4_0_batch_matmul(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B) {
    int bpr = nc / 32;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float acc[1024];
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = W + ((size_t)r * bpr + blk) * Q4_0_BS;
            float sc = f16_to_f32((uint16_t)bp[0]|((uint16_t)bp[1]<<8));
            
            // SIMD dequant: 16 bytes of nibbles -> 32 floats
            __m128i nib = _mm_loadu_si128((const __m128i*)(bp + 2));
            __m128i lo = _mm_and_si128(nib, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_srli_epi16(nib, 4);
            hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
            __m256 sv = _mm256_set1_ps(sc);
            
            // lo has 16 nibbles (elements 0-15, value 0-15). Split into [0:7] and [8:15]
            __m128i lo_lo = lo;
            __m128i lo_hi = _mm_srli_si128(lo, 8);
            __m256i i32_0 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(lo_lo), _mm256_set1_epi32(8));
            __m256i i32_1 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(lo_hi), _mm256_set1_epi32(8));
            __m256 blk0 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_0), sv);
            __m256 blk1 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_1), sv);
            
            // hi has 16 nibbles (elements 16-31)
            __m128i hi_lo = hi;
            __m128i hi_hi = _mm_srli_si128(hi, 8);
            __m256i i32_2 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(hi_lo), _mm256_set1_epi32(8));
            __m256i i32_3 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(hi_hi), _mm256_set1_epi32(8));
            __m256 blk2 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_2), sv);
            __m256 blk3 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_3), sv);
            
            int o = blk * 32;
            for (int b = 0; b < B; b++) {
                const float *xb = x + (size_t)b * nc + o;
                __m256 p0 = _mm256_mul_ps(blk0, _mm256_loadu_ps(xb));
                __m256 p1 = _mm256_mul_ps(blk1, _mm256_loadu_ps(xb + 8));
                __m256 p2 = _mm256_mul_ps(blk2, _mm256_loadu_ps(xb + 16));
                __m256 p3 = _mm256_mul_ps(blk3, _mm256_loadu_ps(xb + 24));
                __m256 s01 = _mm256_add_ps(p0, p1);
                __m256 s23 = _mm256_add_ps(p2, p3);
                __m256 s = _mm256_add_ps(s01, s23);
                acc[b] += hsum_ps(s);
            }
        }
        for (int b = 0; b < B; b++) out[(size_t)b * n_rows + r] = acc[b];
    }
}

/* Scalar Q4_0 dot (for single-token fallback) */
static inline float q4d(const uint8_t *W, const float *x, int nc, int r) {
    int bp=nc/32; float total=0.0f;
    for(int b=0;b<bp;b++){
        const uint8_t *p=W+((size_t)r*bp+b)*Q4_0_BS;
        float sc=f16_to_f32((uint16_t)p[0]|((uint16_t)p[1]<<8));
        int o=b*32; float d=0;
        for (int j=0;j<32;j++) {
            int by=j/16,bx=j%16;
            int nib=(by==0)?(p[2+bx]&0x0F):((p[2+bx]>>4)&0x0F);
            d += (float)(nib-8)*sc*x[o+j];
        }
        total+=d;
    }
    return total;
}
static inline float q8d(const uint8_t *W, const float *x, int nc, int r) {
    int bp=nc/32; float t=0;
    for(int b=0;b<bp;b++){
        const uint8_t *p=W+((size_t)r*bp+b)*Q8_0_BS;
        float d=f16_to_f32((uint16_t)p[0]|((uint16_t)p[1]<<8));
        const int8_t *qs=(const int8_t*)(p+2); int o=b*32;
        for(int i=0;i<32;i++) t += d * (float)qs[i] * x[o+i];
    }
    return t;
}
static inline void rms(float *o, const float *x, const float *w, int n, float e){
    float ss=0; for(int i=0;i<n;i++) ss+=x[i]*x[i];
    float ir=1.0f/sqrtf(ss/n+e); for(int i=0;i<n;i++) o[i]=x[i]*ir*w[i];
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

/* Block-based KV cache */
typedef struct {
    float *k, *v; int n_blocks, seq_len[64], block_map[64][1024];
} KVBlock;
void kv_init(KVBlock *kv, int L, int NKH, int HD) {
    size_t sz = (size_t)L * MAX_BLOCKS * BLOCK_SIZE * NKH * HD * sizeof(float);
    kv->k = (float*)calloc(1, sz); kv->v = (float*)calloc(1, sz); kv->n_blocks = 0;
    memset(kv->block_map, -1, sizeof(kv->block_map)); memset(kv->seq_len, 0, sizeof(kv->seq_len));
}
int kv_alloc(KVBlock *kv) { return kv->n_blocks < MAX_BLOCKS ? kv->n_blocks++ : -1; }

/* AVX2-accelerated Batched Q8_0 matmul */
void q8_0_batch_matmul(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B) {
    int bpr = nc / 32;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float acc[1024];
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = W + ((size_t)r * bpr + blk) * Q8_0_BS;
            float sc = f16_to_f32((uint16_t)bp[0]|((uint16_t)bp[1]<<8));
            const int8_t *qs = (const int8_t*)(bp + 2);
            __m256 sv = _mm256_set1_ps(sc);
            
            // SIMD: load 32 int8 values, convert to float, multiply by scale
            __m128i q8 = _mm_loadu_si128((const __m128i*)(qs));      // bytes 0-15
            __m128i q8b = _mm_loadu_si128((const __m128i*)(qs + 16)); // bytes 16-31
            __m128i q8_low = q8;
            __m128i q8_high = _mm_srli_si128(q8, 8);
            __m128i q8b_low = q8b;
            __m128i q8b_high = _mm_srli_si128(q8b, 8);
            __m256 blk0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8_low)), sv);
            __m256 blk1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8_high)), sv);
            __m256 blk2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8b_low)), sv);
            __m256 blk3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8b_high)), sv);
            
            int o = blk * 32;
            for (int b = 0; b < B; b++) {
                const float *xb = x + (size_t)b * nc + o;
                __m256 p0 = _mm256_mul_ps(blk0, _mm256_loadu_ps(xb));
                __m256 p1 = _mm256_mul_ps(blk1, _mm256_loadu_ps(xb + 8));
                __m256 p2 = _mm256_mul_ps(blk2, _mm256_loadu_ps(xb + 16));
                __m256 p3 = _mm256_mul_ps(blk3, _mm256_loadu_ps(xb + 24));
                __m256 s01 = _mm256_add_ps(p0, p1);
                __m256 s23 = _mm256_add_ps(p2, p3);
                __m256 s = _mm256_add_ps(s01, s23);
                acc[b] += hsum_ps(s);
            }
        }
        for (int b = 0; b < B; b++) out[(size_t)b * n_rows + r] = acc[b];
    }
}

/* Config struct */
typedef struct {
    int L,N,NH,NKH,HD,FF,V; float eps;
    const uint8_t **wQ,**wK,**wV,**wO,**wG,**wU,**wD;
    const float **wAN,**wFN; const int *nQ,*nK,*nV,*nO,*nG,*nU,*nD;
    int nc; const float *emb,*onw; const uint8_t *wOut; int outNR,outNC,outQuant;
    KVBlock **kv_array; float *logits;  // kv_array[B] per-sequence KV, logits [B * V]
} BC;

/* Batched forward: B tokens through all layers using q4_0_batch_matmul */
void batch_forward(const BC *c, const int *tokens, int B, float *ws) {
    int N=c->N, NH=c->NH, NKH=c->NKH, HD=c->HD, FF=c->FF, L=c->L, nc=c->nc;
    int S = N; if (NH*HD > S) S = NH*HD; if (FF > S) S = FF; if (NKH*HD > S) S = NKH*HD;
    
    // Workspace pointers from flat buffer (size = B * (12*S + N + NKH*HD))
    float *x = ws, *xn = ws + B*S, *res = ws + 2*B*S;
    float *q = ws + 3*B*S, *k = ws + 4*B*S, *v = ws + 5*B*S;
    float *att = ws + 6*B*S, *gate = ws + 7*B*S, *up = ws + 8*B*S;
    float *silu = ws + 9*B*S, *oproj = ws + 10*B*S, *ffn = ws + 11*B*S;
    
    for (int b = 0; b < B; b++)
        memcpy(x + b*N, c->emb + (size_t)tokens[b] * N, N * sizeof(float));
    
    for (int l = 0; l < L; l++) {
        for (int b = 0; b < B; b++) {
            memcpy(res + b*N, x + b*N, N * sizeof(float));
            rms(xn + b*N, x + b*N, c->wAN[l], N, c->eps);
        }
        q4_0_batch_matmul(c->wQ[l], xn, q, c->nQ[l], nc, B);
        q4_0_batch_matmul(c->wK[l], xn, k, c->nK[l], nc, B);
        q4_0_batch_matmul(c->wV[l], xn, v, c->nV[l], nc, B);
        
        for (int b = 0; b < B; b++) {
            float *qb = q + b*c->nQ[l], *kb = k + b*c->nK[l], *vb = v + b*c->nV[l];
            KVBlock *kv = c->kv_array[b];
            int pos = kv->seq_len[0];
            for (int hh=0;hh<NH;hh++){float*qh=qb+hh*HD;
                for(int j=0;j<HD/2;j++){double a=(double)pos/pow(10000.0,2.0*j/HD);
                    double cs=cos(a),sn=sin(a);float a0=qh[j],a1=qh[j+HD/2];
                    qh[j]=(float)(a0*cs-a1*sn);qh[j+HD/2]=(float)(a1*cs+a0*sn);}}
            for (int hh=0;hh<NKH;hh++){float*kh=kb+hh*HD;
                for(int j=0;j<HD/2;j++){double a=(double)pos/pow(10000.0,2.0*j/HD);
                    double cs=cos(a),sn=sin(a);float a0=kh[j],a1=kh[j+HD/2];
                    kh[j]=(float)(a0*cs-a1*sn);kh[j+HD/2]=(float)(a1*cs+a0*sn);}}
            
            size_t base=(size_t)l*MAX_BLOCKS*BLOCK_SIZE*NKH*HD;
            int blk=pos/BLOCK_SIZE,off=pos%BLOCK_SIZE;
            int bid=kv->block_map[0][blk];
            if(bid<0){bid=kv_alloc(kv);kv->block_map[0][blk]=bid;}
            memcpy(kv->k+base+(size_t)bid*BLOCK_SIZE*NKH*HD+off*NKH*HD, kb, NKH*HD*sizeof(float));
            memcpy(kv->v+base+(size_t)bid*BLOCK_SIZE*NKH*HD+off*NKH*HD, vb, NKH*HD*sizeof(float));
            
            int sl = pos + 1;
            float *kct=(float*)__builtin_alloca(sl*NKH*HD*sizeof(float));
            float *vct=(float*)__builtin_alloca(sl*NKH*HD*sizeof(float));
            for(int s=0;s<sl;s++){
                int sb=s/BLOCK_SIZE,so=s%BLOCK_SIZE,sbid=kv->block_map[0][sb];
                size_t src=base+(size_t)sbid*BLOCK_SIZE*NKH*HD+(size_t)so*NKH*HD;
                memcpy(kct+s*NKH*HD, kv->k+src, NKH*HD*sizeof(float));
                memcpy(vct+s*NKH*HD, kv->v+src, NKH*HD*sizeof(float));
            }
            gqa(att+b*N, qb, kct, vct, sl, NH, NKH, HD);
        }
        
        q4_0_batch_matmul(c->wO[l], att, oproj, c->nO[l], nc, B);
        for(int b=0;b<B;b++)for(int i=0;i<N;i++)x[b*N+i]=res[b*N+i]+oproj[b*N+i];
        
        for(int b=0;b<B;b++){
            memcpy(res+b*N, x+b*N, N*sizeof(float));
            rms(xn+b*N, x+b*N, c->wFN[l], N, c->eps);
        }
        q4_0_batch_matmul(c->wG[l], xn, gate, c->nG[l], nc, B);
        q4_0_batch_matmul(c->wU[l], xn, up, c->nU[l], nc, B);
        for(int b=0;b<B;b++)
            for(int i=0;i<FF;i++){float g=gate[b*FF+i];silu[b*FF+i]=(g/(1+expf(-g)))*up[b*FF+i];}
        
        q4_0_batch_matmul(c->wD[l], silu, ffn, c->nD[l], FF, B);
        for(int b=0;b<B;b++)for(int i=0;i<N;i++)x[b*N+i]=res[b*N+i]+ffn[b*N+i];
    }
    for(int b=0;b<B;b++) c->kv_array[b]->seq_len[0]++;  // increment once per forward pass
    for(int b=0;b<B;b++) rms(xn+b*N, x+b*N, c->onw, N, c->eps);
    if (c->outQuant)
        q8_0_batch_matmul(c->wOut, xn, c->logits, c->outNR, c->outNC, B);
    else
        q4_0_batch_matmul(c->wOut, xn, c->logits, c->outNR, c->outNC, B);
}
