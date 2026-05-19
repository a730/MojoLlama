/* cengine_batch_instr.c — instrumented version to find hang location */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#define Q4_0_BS 18
#define Q8_0_BS 34
#define BLOCK_SIZE 64
#define MAX_BLOCKS 1024
#define QK_K 256

/* K-quant block structures (from llama.cpp) */
typedef struct {
    uint16_t d;
    uint16_t dm;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
} block_q6_K;

typedef struct {
    uint8_t q[16];
    uint8_t e;
} block_mxfp4;

typedef struct {
    uint16_t d;
    uint16_t dm;
    uint8_t qh[32];
    uint8_t ql[128];
    uint8_t scales[12];
} block_q5_K;

static inline float gf16(uint16_t h) { return _cvtsh_ss(h); }

static inline void k4_scale(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static inline float hsum_ps(__m256 v) {
    __m128 l=_mm256_castps256_ps128(v),h=_mm256_extractf128_ps(v,1);
    l=_mm_add_ps(l,h);l=_mm_hadd_ps(l,l);l=_mm_hadd_ps(l,l);return _mm_cvtss_f32(l);
}

void q4_0_batch_matmul(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B) {
    int bpr = nc / 32;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float *acc = (float*)__builtin_alloca(B * sizeof(float));
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = W + ((size_t)r * bpr + blk) * Q4_0_BS;
            float sc = f16_to_f32((uint16_t)bp[0]|((uint16_t)bp[1]<<8));
            __m128i nib = _mm_loadu_si128((const __m128i*)(bp + 2));
            __m128i lo = _mm_and_si128(nib, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_srli_epi16(nib, 4);
            hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
            __m256 sv = _mm256_set1_ps(sc);
            __m128i lo_lo = lo;
            __m128i lo_hi = _mm_srli_si128(lo, 8);
            __m256i i32_0 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(lo_lo), _mm256_set1_epi32(8));
            __m256i i32_1 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(lo_hi), _mm256_set1_epi32(8));
            __m256 blk0 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_0), sv);
            __m256 blk1 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_1), sv);
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

/* ── PagedAttention: PageTable struct ── */
typedef struct {
    int page_size;       /* KV slots per page (e.g., 64) */
    int n_pages;         /* total physical pages allocated */
    float *pages;        /* flat: [L][n_pages][page_size][NKH*HD] for K then V */
    int *table;          /* logical → physical page mapping (MAX_BLOCKS) */
    int *free_pages;     /* free list of physical page IDs */
    int free_count;      /* number of free pages */
    int _nkh;            /* stored for convenience */
    int _hd;
    int _n_layers;
} PageTable;

void page_table_init(PageTable *pt, int max_pages, int page_size, int nkh, int hd, int n_layers) {
    pt->page_size = page_size;
    pt->n_pages = max_pages;
    pt->_nkh = nkh;
    pt->_hd = hd;
    pt->_n_layers = n_layers;
    size_t page_data = (size_t)page_size * nkh * hd;
    size_t total = (size_t)n_layers * max_pages * page_data * 2; /* K + V */
    pt->pages = (float*)calloc(1, total * sizeof(float));
    pt->table = (int*)malloc(MAX_BLOCKS * sizeof(int));
    memset(pt->table, -1, MAX_BLOCKS * sizeof(int));
    pt->free_pages = (int*)malloc(max_pages * sizeof(int));
    for (int i = 0; i < max_pages; i++) pt->free_pages[i] = max_pages - 1 - i;
    pt->free_count = max_pages;
}

int page_table_alloc(PageTable *pt) {
    if (pt->free_count <= 0) return -1;
    return pt->free_pages[--pt->free_count];
}

void page_table_free(PageTable *pt, int page_id) {
    if (pt->free_count < pt->n_pages)
        pt->free_pages[pt->free_count++] = page_id;
}

void map_page(PageTable *pt, int logical_page, int physical_page) {
    if (logical_page >= 0 && logical_page < MAX_BLOCKS)
        pt->table[logical_page] = physical_page;
}

typedef struct {
    float *k, *v; int n_blocks, seq_len[64], block_map[64][1024];
    PageTable pt;  /* PagedAttention page table (embedded) */
} KVBlock;
void kv_init(KVBlock *kv, int L, int NKH, int HD) {
    size_t sz = (size_t)L * MAX_BLOCKS * BLOCK_SIZE * NKH * HD * sizeof(float);
    kv->k = (float*)calloc(1, sz); kv->v = (float*)calloc(1, sz); kv->n_blocks = 0;
    memset(kv->block_map, -1, sizeof(kv->block_map)); memset(kv->seq_len, 0, sizeof(kv->seq_len));
    page_table_init(&kv->pt, MAX_BLOCKS, BLOCK_SIZE, NKH, HD, L);
}
int kv_alloc(KVBlock *kv) { return page_table_alloc(&kv->pt); }

void q8_0_batch_matmul(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B) {
    int bpr = nc / 32;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float *acc = (float*)__builtin_alloca(B * sizeof(float));
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = W + ((size_t)r * bpr + blk) * Q8_0_BS;
            float sc = f16_to_f32((uint16_t)bp[0]|((uint16_t)bp[1]<<8));
            const int8_t *qs = (const int8_t*)(bp + 2);
            __m256 sv = _mm256_set1_ps(sc);
            __m128i q8 = _mm_loadu_si128((const __m128i*)(qs));
            __m128i q8b = _mm_loadu_si128((const __m128i*)(qs + 16));
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

/* Batched Q4_K matmul: 256-element super-blocks, 4-bit K-quant (AVX2) */
void q4_k_batch_matmul(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B) {
    int bpr = nc / QK_K;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float *acc = (float*)__builtin_alloca(B * sizeof(float));
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const block_q4_K *bp = (const block_q4_K*)(W + ((size_t)r * bpr + blk) * sizeof(block_q4_K));
            float d = gf16(bp->d), dm = gf16(bp->dm);
            const uint8_t *q = bp->qs;
            int is = 0;
            float deq[QK_K];
            for (int s = 0; s < QK_K; s += 64, q += 32, is += 2) {
                uint8_t sc1, sc2, m1, m2;
                k4_scale(is, bp->scales, &sc1, &m1);
                float d1 = d * sc1, mm1 = dm * m1;
                k4_scale(is+1, bp->scales, &sc2, &m2);
                float d2 = d * sc2, mm2 = dm * m2;
                __m128i qb = _mm_loadu_si128((const __m128i*)q);
                __m128i qb2 = _mm_loadu_si128((const __m128i*)(q+16));
                __m128i lo1 = _mm_and_si128(qb, _mm_set1_epi8(0x0F));
                __m128i lo2 = _mm_and_si128(qb2, _mm_set1_epi8(0x0F));
                __m128i hi1 = _mm_and_si128(_mm_srli_epi16(qb, 4), _mm_set1_epi8(0x0F));
                __m128i hi2 = _mm_and_si128(_mm_srli_epi16(qb2, 4), _mm_set1_epi8(0x0F));
                __m256 d1v = _mm256_set1_ps(d1), mm1v = _mm256_set1_ps(mm1);
                __m256 d2v = _mm256_set1_ps(d2), mm2v = _mm256_set1_ps(mm2);
                for (int k = 0; k < 4; k++) {
                    __m128i src = (k < 2) ? lo1 : lo2;
                    __m128i src_split = (k & 1) ? _mm_srli_si128(src, 8) : src;
                    int out_off = s + k * 8;
                    _mm256_storeu_ps(deq + out_off, _mm256_sub_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(src_split)), d1v), mm1v));
                }
                for (int k = 0; k < 4; k++) {
                    __m128i src = (k < 2) ? hi1 : hi2;
                    __m128i src_split = (k & 1) ? _mm_srli_si128(src, 8) : src;
                    int out_off = s + 32 + k * 8;
                    _mm256_storeu_ps(deq + out_off, _mm256_sub_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(src_split)), d2v), mm2v));
                }
            }
            int o = blk * QK_K;
            for (int b = 0; b < B; b++) {
                const float *xb = x + (size_t)b * nc + o;
                __m256 sum = _mm256_setzero_ps();
                for (int j = 0; j < QK_K; j += 8) {
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(deq + j),
                                         _mm256_loadu_ps(xb + j), sum);
                }
                acc[b] += hsum_ps(sum);
            }
        }
        for (int b = 0; b < B; b++) out[(size_t)b * n_rows + r] = acc[b];
    }
}

/* AVX2 Batched Q6_K matmul: 256-element blocks, 6-bit K-quant (fused dequant+dot) */
void q6_k_batch_matmul(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B) {
    int bpr = nc / QK_K;
    /* LUT for _mm_shuffle_epi8 nibble deinterleave:
       maps [n0,n2,n4,n6,n8,n10,n12,n14, n1,n3,n5,n7,n9,n11,n13,n15]
         to [n0,n1,n2,n3,n4,n5,n6,n7, n8,n9,n10,n11,n12,n13,n14,n15] */
    static const uint8_t kShufNib[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    __m128i shuf_nib = _mm_loadu_si128((const __m128i*)kShufNib);
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float *acc = (float*)__builtin_alloca(B * sizeof(float));
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const block_q6_K *bp = (const block_q6_K*)(W + ((size_t)r * bpr + blk) * sizeof(block_q6_K));
            float d = gf16(bp->d);
            int o = blk * QK_K;
            for (int b = 0; b < B; b++) {
                const float *xb = x + (size_t)b * nc + o;
                __m256 sum = _mm256_setzero_ps();
                for (int j = 0; j < QK_K; j += 16) {
                    float scale = d * bp->scales[j/16];
                    __m256 sv = _mm256_set1_ps(scale);
                    /* --- QL: Extract 16 nibbles from 8 packed bytes via SIMD --- */
                    __m128i ql8 = _mm_loadl_epi64((const __m128i*)(bp->ql + j/2));
                    __m128i lo = _mm_and_si128(ql8, _mm_set1_epi8(0x0F));
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(ql8, 4), _mm_set1_epi8(0x0F));
                    /* Combine lo in lower lane, hi in upper lane, shuffle to order */
                    __m128i nib_all = _mm_shuffle_epi8(_mm_unpacklo_epi64(lo, hi), shuf_nib);
                    __m128i nib_lo = nib_all;
                    __m128i nib_hi = _mm_srli_si128(nib_all, 8);
                    /* --- QH: Extract 16 x 2-bit pairs from 4 bytes (shifts+masks) --- */
                    uint32_t qh4;
                    memcpy(&qh4, bp->qh + j/4, 4);
                    uint8_t qh16[16];
                    for (int k = 0; k < 4; k++) {
                        uint8_t byte = (qh4 >> (k * 8)) & 0xFF;
                        qh16[k*4 + 0] = byte & 3;
                        qh16[k*4 + 1] = (byte >> 2) & 3;
                        qh16[k*4 + 2] = (byte >> 4) & 3;
                        qh16[k*4 + 3] = (byte >> 6) & 3;
                    }
                    __m128i up = _mm_loadu_si128((const __m128i*)qh16);
                    __m128i up_lo = up;
                    __m128i up_hi = _mm_srli_si128(up, 8);
                    /* --- Combine 6-bit value: nibble | (upper << 4), zero-center --- */
                    __m128i val_lo = _mm_sub_epi8(
                        _mm_or_si128(nib_lo, _mm_slli_epi16(up_lo, 4)), _mm_set1_epi8(32));
                    __m128i val_hi = _mm_sub_epi8(
                        _mm_or_si128(nib_hi, _mm_slli_epi16(up_hi, 4)), _mm_set1_epi8(32));
                    /* --- Convert to float, scale, FMA with x --- */
                    sum = _mm256_fmadd_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(val_lo)), sv),
                        _mm256_loadu_ps(xb + j), sum);
                    sum = _mm256_fmadd_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(val_hi)), sv),
                        _mm256_loadu_ps(xb + j + 8), sum);
                }
                acc[b] += hsum_ps(sum);
            }
        }
        for (int b = 0; b < B; b++) out[(size_t)b * n_rows + r] = acc[b];
    }
}

/* Batched Q5_K matmul: 256-element super-blocks, 5-bit K-quant (scalar fallback) */
void q5_k_batch_matmul(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B) {
    int bpr = nc / QK_K;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float *acc = (float*)__builtin_alloca(B * sizeof(float));
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const block_q5_K *bp = (const block_q5_K*)(W + ((size_t)r * bpr + blk) * sizeof(block_q5_K));
            float d = gf16(bp->d), dm = gf16(bp->dm);
            float deq[QK_K];
            for (int s = 0; s < QK_K; s += 64) {
                int is = s / 32;
                uint8_t sc1, sc2, m1, m2;
                k4_scale(is, bp->scales, &sc1, &m1);
                float d1 = d * sc1, mm1 = dm * m1;
                k4_scale(is+1, bp->scales, &sc2, &m2);
                float d2 = d * sc2, mm2 = dm * m2;
                for (int j = 0; j < 32; j++) {
                    int idx = s + j;
                    int lo = (bp->ql[idx/2] >> ((idx%2)*4)) & 0x0F;
                    int hi = ((bp->qh[idx/8] >> (idx%8)) & 0x01) << 4;
                    deq[idx] = d1 * (float)(lo | hi) - mm1;
                }
                for (int j = 32; j < 64; j++) {
                    int idx = s + j;
                    int lo = (bp->ql[idx/2] >> ((idx%2)*4)) & 0x0F;
                    int hi = ((bp->qh[idx/8] >> (idx%8)) & 0x01) << 4;
                    deq[idx] = d2 * (float)(lo | hi) - mm2;
                }
            }
            int o = blk * QK_K;
            for (int b = 0; b < B; b++) {
                const float *xb = x + (size_t)b * nc + o;
                __m256 sum = _mm256_setzero_ps();
                for (int j = 0; j < QK_K; j += 8) {
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(deq + j),
                                         _mm256_loadu_ps(xb + j), sum);
                }
                acc[b] += hsum_ps(sum);
            }
        }
        for (int b = 0; b < B; b++) out[(size_t)b * n_rows + r] = acc[b];
    }
}

/* Batched MXFP4 matmul: 32-element blocks, 4-bit two's complement mantissas + E8M0 scale */
void mxfp4_batch_matmul(const uint8_t *W, const float *x, float *out,
                         int n_rows, int nc, int B) {
    int bpr = nc / 32;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float *acc = (float*)__builtin_alloca(B * sizeof(float));
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const uint8_t *bp = W + ((size_t)r * bpr + blk) * sizeof(block_mxfp4);
            float scale = powf(2.0f, (int)bp[16] - 127);
            __m128i packed = _mm_loadu_si128((const __m128i*)bp);
            __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
            __m128i sign_lo = _mm_cmpgt_epi8(lo, _mm_set1_epi8(7));
            __m128i sign_hi = _mm_cmpgt_epi8(hi, _mm_set1_epi8(7));
            lo = _mm_sub_epi8(lo, _mm_and_si128(sign_lo, _mm_set1_epi8(16)));
            hi = _mm_sub_epi8(hi, _mm_and_si128(sign_hi, _mm_set1_epi8(16)));
            __m256 sv = _mm256_set1_ps(scale);
            __m128i lo_lo = lo;
            __m128i lo_hi = _mm_srli_si128(lo, 8);
            __m256i i32_0 = _mm256_cvtepi8_epi32(lo_lo);
            __m256i i32_1 = _mm256_cvtepi8_epi32(lo_hi);
            __m256 blk0 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_0), sv);
            __m256 blk1 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_1), sv);
            __m128i hi_lo = hi;
            __m128i hi_hi = _mm_srli_si128(hi, 8);
            __m256i i32_2 = _mm256_cvtepi8_epi32(hi_lo);
            __m256i i32_3 = _mm256_cvtepi8_epi32(hi_hi);
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

typedef struct {
    int L,N,NH,NKH,HD,FF,V; float eps;
    const uint8_t **wQ,**wK,**wV,**wO,**wG,**wU,**wD;
    const float **wAN,**wFN; const int *nQ,*nK,*nV,*nO,*nG,*nU,*nD;
    int nc; const float *emb,*onw; const uint8_t *wOut; int outNR,outNC,outQuant;
    KVBlock **kv_array; float *logits;
    int n_experts, n_experts_per_tok, moe_intermediate;
    const float **w_gate_inp;
    const uint8_t **w_gate_exps;
    const uint8_t **w_up_exps;
    const uint8_t **w_down_exps;
    int gate_exp_quant, up_exp_quant, down_exp_quant;
    const int *q_quant, *k_quant, *v_quant, *o_quant;
    const int *g_quant, *u_quant, *d_quant;
    int emb_quant;
    const uint8_t **wQK;   /* fused Q+K weight pointer (NULL if no fusion) */
    const int *qk_quant;   /* quant type for fused Q+K weight */
    float *cos_table;
    float *sin_table;
    int max_ctx;
} BC;

void moe_ffn(const BC *c, int l, float *x, float *gate_buf, float *up_buf, float *ffn_buf, int B) {
    int N = c->N, NE = c->n_experts, NK = c->n_experts_per_tok, M = c->moe_intermediate;
    size_t gate_stride = (size_t)M * (N / QK_K) * sizeof(block_q4_K);
    size_t up_stride = (size_t)M * (N / QK_K) * sizeof(block_q4_K);
    size_t down_stride = (size_t)N * (M / QK_K) * sizeof(block_q6_K);
    if (c->gate_exp_quant == 14) gate_stride = (size_t)M * (N / QK_K) * sizeof(block_q6_K);
    if (c->up_exp_quant == 14) up_stride = (size_t)M * (N / QK_K) * sizeof(block_q6_K);
    if (c->down_exp_quant == 12) down_stride = (size_t)N * (M / QK_K) * sizeof(block_q4_K);
    float router_logits[256];
    int top_idx[16]; float top_val[16];
    const float *w_router = c->w_gate_inp[l];
    
    for (int b = 0; b < B; b++) {
        float *xb = x + b*N;
        
        // Router
        for (int e = 0; e < NE; e++) {
            float dot = 0;
            for (int i = 0; i < N; i++) dot += xb[i] * w_router[e * N + i];
            router_logits[e] = dot;
        }
        
        // Softmax
        float mx = router_logits[0];
        for (int e = 1; e < NE; e++) if (router_logits[e] > mx) mx = router_logits[e];
        float sum = 0;
        for (int e = 0; e < NE; e++) { router_logits[e] = expf(router_logits[e] - mx); sum += router_logits[e]; }
        float inv_sum = 1.0f / (sum + 1e-10f);
        for (int e = 0; e < NE; e++) router_logits[e] *= inv_sum;
        
        // Top-K
        for (int k = 0; k < NK; k++) { top_val[k] = -1e30f; top_idx[k] = -1; }
        for (int e = 0; e < NE; e++) {
            float v = router_logits[e];
            for (int k = 0; k < NK; k++) {
                if (v > top_val[k]) {
                    for (int k2 = NK-1; k2 > k; k2--) { top_val[k2] = top_val[k2-1]; top_idx[k2] = top_idx[k2-1]; }
                    top_val[k] = v; top_idx[k] = e; break;
                }
            }
        }
        
        // Sparse expert FFN
        memset(ffn_buf, 0, N * sizeof(float));
        for (int k = 0; k < NK; k++) {
            int e = top_idx[k];
            if (e < 0) continue;
            float weight = top_val[k];
            
            const uint8_t *gate_w = c->w_gate_exps[l] + (size_t)e * gate_stride;
            if (c->gate_exp_quant == 12)
                q4_k_batch_matmul(gate_w, xb, gate_buf, M, N, 1);
            else if (c->gate_exp_quant == 2)
                q4_0_batch_matmul(gate_w, xb, gate_buf, M, N, 1);
            
            const uint8_t *up_w = c->w_up_exps[l] + (size_t)e * up_stride;
            if (c->up_exp_quant == 12)
                q4_k_batch_matmul(up_w, xb, up_buf, M, N, 1);
            else if (c->up_exp_quant == 2)
                q4_0_batch_matmul(up_w, xb, up_buf, M, N, 1);
            
            for (int i = 0; i < M; i++) {
                float g = gate_buf[i];
                float sg = g / (1.0f + expf(-g));
                gate_buf[i] = sg * up_buf[i];
            }
            
            const uint8_t *down_w = c->w_down_exps[l] + (size_t)e * down_stride;
            if (c->down_exp_quant == 14)
                q6_k_batch_matmul(down_w, gate_buf, up_buf, N, M, 1);
            else if (c->down_exp_quant == 2)
                q4_0_batch_matmul(down_w, gate_buf, up_buf, N, M, 1);
            
            for (int i = 0; i < N; i++) ffn_buf[i] += weight * up_buf[i];
        }
    }
}

static inline void batch_matmul(int qt, const uint8_t *W, const float *x, float *out,
                                int n_rows, int nc, int B) {
    if (qt == 12)      q4_k_batch_matmul(W, x, out, n_rows, nc, B);
    else if (qt == 13) q5_k_batch_matmul(W, x, out, n_rows, nc, B);
    else if (qt == 14) q6_k_batch_matmul(W, x, out, n_rows, nc, B);
    else if (qt == 39) mxfp4_batch_matmul(W, x, out, n_rows, nc, B);
    else if (qt == 8)  q8_0_batch_matmul(W, x, out, n_rows, nc, B);
    else if (qt == 10) q4_0_batch_matmul(W, x, out, n_rows, nc, B);  // Q2_K fallback
    else if (qt == 11) q4_0_batch_matmul(W, x, out, n_rows, nc, B);  // Q3_K fallback
    else               q4_0_batch_matmul(W, x, out, n_rows, nc, B);
}

void rope_init(float *cos_table, float *sin_table, int max_ctx, int hd) {
    int hd2 = hd / 2;
    for (int pos = 0; pos < max_ctx; pos++) {
        for (int j = 0; j < hd2; j++) {
            double theta = (double)pos / pow(10000.0, 2.0 * j / hd);
            cos_table[pos * hd2 + j] = (float)cos(theta);
            sin_table[pos * hd2 + j] = (float)sin(theta);
        }
    }
}

static inline void rope_apply(float *buf, int nh, int hd, int pos,
                               const float *cos_t, const float *sin_t) {
    int hd2 = hd / 2;
    const float *cos_row = cos_t + (size_t)pos * hd2;
    const float *sin_row = sin_t + (size_t)pos * hd2;
    for (int h = 0; h < nh; h++) {
        float *b = buf + (size_t)h * hd;
        int j = 0;
        for (; j <= hd2 - 8; j += 8) {
            __m256 v0 = _mm256_loadu_ps(b + j);
            __m256 v1 = _mm256_loadu_ps(b + j + hd2);
            __m256 cv = _mm256_loadu_ps(cos_row + j);
            __m256 sv = _mm256_loadu_ps(sin_row + j);
            __m256 nv0 = _mm256_sub_ps(_mm256_mul_ps(v0, cv), _mm256_mul_ps(v1, sv));
            __m256 nv1 = _mm256_add_ps(_mm256_mul_ps(v0, sv), _mm256_mul_ps(v1, cv));
            _mm256_storeu_ps(b + j, nv0);
            _mm256_storeu_ps(b + j + hd2, nv1);
        }
        for (; j < hd2; j++) {
            float a0 = b[j], a1 = b[j + hd2];
            float cs = cos_row[j], sn = sin_row[j];
            b[j] = a0 * cs - a1 * sn;
            b[j + hd2] = a1 * cs + a0 * sn;
        }
    }
}

void emb_lookup(const uint8_t *emb, int token, float *out, int N, int quant) {
    if (quant == 12) {
        int bpr = N / QK_K, blk_size = sizeof(block_q4_K);
        for (int bi = 0; bi < bpr; bi++) {
            const block_q4_K *bp = (const block_q4_K*)(emb + ((size_t)token * bpr + bi) * blk_size);
            float d = gf16(bp->d), dm = gf16(bp->dm);
            int o = bi * QK_K;
            for (int g = 0; g < 8; g++) {
                uint8_t sc, mn; k4_scale(g, bp->scales, &sc, &mn);
                float d_sc = d * ((int)sc - 32) / 8.0f;
                float d_mn = dm * ((int)mn - 32) / 8.0f;
                for (int j = g*32; j < (g+1)*32; j++) {
                    int s = (bp->qs[j/2] >> ((j & 1) * 4)) & 0xF;
                    out[o + j] = d_sc * (s - 8) - d_mn * 6.0f;
                }
            }
        }
    } else {
        memcpy(out, emb + (size_t)token * N * sizeof(float), N * sizeof(float));
    }
}

void batch_forward(const BC *c, const int *tokens, int B, float *ws) {
    int N=c->N, NH=c->NH, NKH=c->NKH, HD=c->HD, FF=c->FF, L=c->L, nc=c->nc;
    int S = N; if (NH*HD > S) S = NH*HD; if (FF > S) S = FF; if (NKH*HD > S) S = NKH*HD;
    int qk_fused = NH*HD + NKH*HD;  /* for fused QK output (if fusion enabled) */
    if (qk_fused > S) S = qk_fused;
    
    float *x = ws, *xn = ws + B*S, *res = ws + 2*B*S;
    float *q = ws + 3*B*S, *k = ws + 4*B*S, *v = ws + 5*B*S;
    float *att = ws + 6*B*S, *gate = ws + 7*B*S, *up = ws + 8*B*S;
    float *silu = ws + 9*B*S, *oproj = ws + 10*B*S, *ffn = ws + 11*B*S;
    
    for (int b = 0; b < B; b++) {
        if (c->emb_quant == 12)
            emb_lookup((const uint8_t*)c->emb, tokens[b], x + b*N, N, 12);
        else
            memcpy(x + b*N, c->emb + (size_t)tokens[b] * N, N * sizeof(float));
    }
    
    for (int l = 0; l < L; l++) {
        
    for (int b = 0; b < B; b++) {
        float *xb = x + b*N;
        float *rb = res + b*N;
        for(int i=0;i<=N-8;i+=8){
            __m256 xv = _mm256_loadu_ps(xb+i);
            xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
            _mm256_storeu_ps(xb+i, xv);
        }
        for(int i=N-(N%8);i<N;i++){
            float v = xb[i];
            if(v != v) v = 0; else if(v > 1000.0f) v = 1000.0f; else if(v < -1000.0f) v = -1000.0f;
            xb[i] = v;
        }
        memcpy(rb, xb, N * sizeof(float));
        rms(xn + b*N, xb, c->wAN[l], N, c->eps);
    }
        
        /* Fused Q+K matmul when available (types match, weight built), else 3 separate */
        int use_fused_qk = (c->wQK && c->wQK[l] != NULL);
        if (use_fused_qk) {
            int nqk = c->nQ[l] + c->nK[l];
            batch_matmul(c->qk_quant[l], c->wQK[l], xn, q, nqk, nc, B);
            for (int b = 0; b < B; b++) {
                memcpy(k + (size_t)b * c->nK[l],
                       q + (size_t)b * nqk + c->nQ[l],
                       c->nK[l] * sizeof(float));
            }
            batch_matmul(c->v_quant[l], c->wV[l], xn, v, c->nV[l], nc, B);
        } else {
            batch_matmul(c->q_quant[l], c->wQ[l], xn, q, c->nQ[l], nc, B);
            batch_matmul(c->k_quant[l], c->wK[l], xn, k, c->nK[l], nc, B);
            batch_matmul(c->v_quant[l], c->wV[l], xn, v, c->nV[l], nc, B);
        }
        
        for (int b = 0; b < B; b++) {
            float *qb = q + b*c->nQ[l], *kb = k + b*c->nK[l], *vb = v + b*c->nV[l];
            KVBlock *kv = c->kv_array[b];
            int pos = kv->seq_len[0];
            rope_apply(qb, NH, HD, pos, c->cos_table, c->sin_table);
            rope_apply(kb, NKH, HD, pos, c->cos_table, c->sin_table);
            
            size_t base=(size_t)l*MAX_BLOCKS*BLOCK_SIZE*NKH*HD;
            int blk=pos/BLOCK_SIZE,off=pos%BLOCK_SIZE;
            /* PagedAttention: use page table for KV cache */
            PageTable *pt = &kv->pt;
            int bid = (blk < MAX_BLOCKS) ? pt->table[blk] : -1;
            if(bid<0){bid=kv_alloc(kv);if(blk<MAX_BLOCKS)pt->table[blk]=bid;}
            /* Write to both old arrays (backward compat) and page table */
            memcpy(kv->k+base+(size_t)bid*BLOCK_SIZE*NKH*HD+off*NKH*HD, kb, NKH*HD*sizeof(float));
            memcpy(kv->v+base+(size_t)bid*BLOCK_SIZE*NKH*HD+off*NKH*HD, vb, NKH*HD*sizeof(float));
            /* Also write to page table flat array */
            {
                size_t page_data = (size_t)pt->page_size * NKH * HD;
                size_t layer_stride = (size_t)pt->n_pages * page_data;
                size_t k_base_pt = (size_t)l * layer_stride;
                size_t v_base_pt = (size_t)pt->_n_layers * layer_stride + k_base_pt;
                memcpy(pt->pages + k_base_pt + (size_t)bid * page_data + (size_t)off * NKH * HD,
                       kb, NKH * HD * sizeof(float));
                memcpy(pt->pages + v_base_pt + (size_t)bid * page_data + (size_t)off * NKH * HD,
                       vb, NKH * HD * sizeof(float));
            }

            int sl = pos + 1;
            float *kct=(float*)__builtin_alloca(sl*NKH*HD*sizeof(float));
            float *vct=(float*)__builtin_alloca(sl*NKH*HD*sizeof(float));
            for(int s=0;s<sl;s++){
                int sb=s/BLOCK_SIZE,so=s%BLOCK_SIZE;
                int sbid = (sb < MAX_BLOCKS) ? pt->table[sb] : -1;
                if(sbid<0) continue;
                /* Read from page table flat array (primary) */
                size_t page_data = (size_t)pt->page_size * NKH * HD;
                size_t layer_stride = (size_t)pt->n_pages * page_data;
                size_t k_base_pt = (size_t)l * layer_stride;
                size_t v_base_pt = (size_t)pt->_n_layers * layer_stride + k_base_pt;
                size_t src_k = k_base_pt + (size_t)sbid * page_data + (size_t)so * NKH * HD;
                size_t src_v = v_base_pt + (size_t)sbid * page_data + (size_t)so * NKH * HD;
                memcpy(kct+s*NKH*HD, pt->pages+src_k, NKH*HD*sizeof(float));
                memcpy(vct+s*NKH*HD, pt->pages+src_v, NKH*HD*sizeof(float));
            }
            gqa(att+b*N, qb, kct, vct, sl, NH, NKH, HD);
        }
        
        batch_matmul(c->o_quant[l], c->wO[l], att, oproj, c->nO[l], c->NH * c->HD, B);
        for(int b=0;b<B;b++)for(int i=0;i<N;i++)x[b*N+i]=res[b*N+i]+oproj[b*N+i];
        
        for(int b=0;b<B;b++){
            float *xb = x + b*N;
            float *rb = res + b*N;
            for(int i=0;i<=N-8;i+=8){
                __m256 xv = _mm256_loadu_ps(xb+i);
                xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
                _mm256_storeu_ps(xb+i, xv);
            }
            for(int i=N-(N%8);i<N;i++){
                float v = xb[i];
                if(v != v) v = 0; else if(v > 1000.0f) v = 1000.0f; else if(v < -1000.0f) v = -1000.0f;
                xb[i] = v;
            }
            memcpy(rb, xb, N * sizeof(float));
            rms(xn+b*N, xb, c->wFN[l], N, c->eps);
        }
        
        if (c->n_experts > 0 && c->n_experts_per_tok > 0) {
            for(int b=0;b<B;b++){
                float *gate_buf = gate; float *up_buf = silu; float *ffn_buf = ffn;
                memset(ffn_buf, 0, N * sizeof(float));
                moe_ffn(c, l, xn + b*N, gate_buf, up_buf, ffn_buf, 1);
                for(int i=0;i<N;i++)x[b*N+i]=res[b*N+i]+ffn_buf[i];
            }
        } else {
            batch_matmul(c->g_quant[l], c->wG[l], xn, gate, c->nG[l], nc, B);
            batch_matmul(c->u_quant[l], c->wU[l], xn, up, c->nU[l], nc, B);
            for(int b=0;b<B;b++)
                for(int i=0;i<FF;i++){float g=gate[b*FF+i];silu[b*FF+i]=(g/(1+expf(-g)))*up[b*FF+i];}
            batch_matmul(c->d_quant[l], c->wD[l], silu, ffn, c->nD[l], FF, B);
            for(int b=0;b<B;b++)for(int i=0;i<N;i++)x[b*N+i]=res[b*N+i]+ffn[b*N+i];
        }
    }
    for(int b=0;b<B;b++) c->kv_array[b]->seq_len[0]++;
    for(int b=0;b<B;b++){
        float *xb = x + b*N;
        float *rb = res + b*N;
        for(int i=0;i<=N-8;i+=8){
            __m256 xv = _mm256_loadu_ps(xb+i);
            xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
            _mm256_storeu_ps(xb+i, xv);
        }
        for(int i=N-(N%8);i<N;i++){
            float v = xb[i];
            if(v != v) v = 0; else if(v > 1000.0f) v = 1000.0f; else if(v < -1000.0f) v = -1000.0f;
            xb[i] = v;
        }
        memcpy(rb, xb, N * sizeof(float));
        rms(xn+b*N, xb, c->onw, N, c->eps);
    }
    batch_matmul(c->outQuant, c->wOut, xn, c->logits, c->outNR, c->outNC, B);
}
