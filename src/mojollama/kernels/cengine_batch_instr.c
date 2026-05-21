/* cengine_batch_instr.c — instrumented version to find hang location */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>

/* External Q8_0-activation matmul from quant_kernels_omp.c (compiled together into combined_engine.so) */
extern void quant_matmul_omp(const uint8_t *restrict W, const float *restrict x,
                              float *restrict out, int n_rows, int n_cols, int quant_type);
extern int quantize_row_q8_0(const float *restrict x, uint8_t *restrict q8, int n_cols);
/* Fused MoE forward (quant_kernels_omp.c) */
extern void moe_forward_omp(const uint8_t** gate_raw, const uint8_t** up_raw, const uint8_t** down_raw,
    const float* x_norm, int n_ff_expert, int n_embd,
    int qt_gate, int qt_up, int qt_down,
    const int* top_indices, const float* top_weights, int top_k,
    float* combined, float* prealloc_buf, uint8_t* prealloc_q8);
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

/* Batched MXFP4 matmul: 32-element blocks, 4-bit two's complement mantissas + E8M0 scale
 *
 * Optimizations:
 *   1. powf(2.0f, n) → integer bit manipulation (E8M0 exponent maps directly to IEEE754 float32)
 *   2. Vertical accumulation (B=1) — one hsum_ps per row, not per block
 *   3. FMA instructions (_mm256_fmadd_ps) instead of mul + add
 *   4. Software prefetch of next weight blocks
 */
void mxfp4_batch_matmul(const uint8_t *W, const float *x, float *out,
                         int n_rows, int nc, int B) {
    int bpr = nc / 32;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        if (B == 1) {
            /* ── Fast path (B=1): vertical accumulation, 1 hsum per row ── */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int blk = 0; blk < bpr; blk++) {
                const uint8_t *bp = W + ((size_t)r * bpr + blk) * sizeof(block_mxfp4);
                /* Prefetch weight blocks 2 ahead (17 B/block → ~34 B ahead) */
                _mm_prefetch(bp + 2 * (int)sizeof(block_mxfp4), _MM_HINT_NTA);

                /* E8M0 exponent → IEEE754 float32: 2^(e-127) = float with bits (e << 23) */
                union { uint32_t u; float f; } sc = { .u = ((uint32_t)bp[16]) << 23 };
                __m256 sv = _mm256_set1_ps(sc.f);

                __m128i packed = _mm_loadu_si128((const __m128i*)bp);
                /* Extract lower/upper 4-bit nibbles from 16 packed bytes */
                __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
                /* Sign extend 4-bit to signed int8: nibble >= 8 → nibble - 16 */
                __m128i sign_lo = _mm_cmpgt_epi8(lo, _mm_set1_epi8(7));
                __m128i sign_hi = _mm_cmpgt_epi8(hi, _mm_set1_epi8(7));
                lo = _mm_sub_epi8(lo, _mm_and_si128(sign_lo, _mm_set1_epi8(16)));
                hi = _mm_sub_epi8(hi, _mm_and_si128(sign_hi, _mm_set1_epi8(16)));
                /* Interleave lo/hi to restore natural order */
                __m128i vals_lo = _mm_unpacklo_epi8(lo, hi);
                __m128i vals_hi = _mm_unpackhi_epi8(lo, hi);
                /* Dequant: sign-extend int8 → int32 → float, multiply by scale */
                __m256 d0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_lo)), sv);
                __m256 d1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_lo, 8))), sv);
                __m256 d2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_hi)), sv);
                __m256 d3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_hi, 8))), sv);

                int o = blk * 32;
                const float *xb = x + o;
                /* FMA + vertical accumulate into 4 registers */
                acc0 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(xb), acc0);
                acc1 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(xb + 8), acc1);
                acc2 = _mm256_fmadd_ps(d2, _mm256_loadu_ps(xb + 16), acc2);
                acc3 = _mm256_fmadd_ps(d3, _mm256_loadu_ps(xb + 24), acc3);
            }
            /* Single reduction per row */
            __m256 total = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                         _mm256_add_ps(acc2, acc3));
            out[r] = hsum_ps(total);

        } else {
            /* ── Batch path (B>1): per-block hsum, but still benefits from powf fix + FMA + prefetch ── */
            float *acc = (float*)__builtin_alloca(B * sizeof(float));
            for (int b = 0; b < B; b++) acc[b] = 0.0f;

            for (int blk = 0; blk < bpr; blk++) {
                const uint8_t *bp = W + ((size_t)r * bpr + blk) * sizeof(block_mxfp4);
                _mm_prefetch(bp + 2 * (int)sizeof(block_mxfp4), _MM_HINT_NTA);

                union { uint32_t u; float f; } sc = { .u = ((uint32_t)bp[16]) << 23 };
                __m256 sv = _mm256_set1_ps(sc.f);

                __m128i packed = _mm_loadu_si128((const __m128i*)bp);
                __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
                __m128i sign_lo = _mm_cmpgt_epi8(lo, _mm_set1_epi8(7));
                __m128i sign_hi = _mm_cmpgt_epi8(hi, _mm_set1_epi8(7));
                lo = _mm_sub_epi8(lo, _mm_and_si128(sign_lo, _mm_set1_epi8(16)));
                hi = _mm_sub_epi8(hi, _mm_and_si128(sign_hi, _mm_set1_epi8(16)));
                __m128i vals_lo = _mm_unpacklo_epi8(lo, hi);
                __m128i vals_hi = _mm_unpackhi_epi8(lo, hi);

                __m256 d0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_lo)), sv);
                __m256 d1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_lo, 8))), sv);
                __m256 d2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_hi)), sv);
                __m256 d3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_hi, 8))), sv);

                int o = blk * 32;
                for (int b = 0; b < B; b++) {
                    const float *xb = x + (size_t)b * nc + o;
                    __m256 p0 = _mm256_fmadd_ps(d0, _mm256_loadu_ps(xb), _mm256_setzero_ps());
                    __m256 p1 = _mm256_fmadd_ps(d1, _mm256_loadu_ps(xb + 8), _mm256_setzero_ps());
                    __m256 p2 = _mm256_fmadd_ps(d2, _mm256_loadu_ps(xb + 16), _mm256_setzero_ps());
                    __m256 p3 = _mm256_fmadd_ps(d3, _mm256_loadu_ps(xb + 24), _mm256_setzero_ps());
                    __m256 s01 = _mm256_add_ps(p0, p1);
                    __m256 s23 = _mm256_add_ps(p2, p3);
                    __m256 s = _mm256_add_ps(s01, s23);
                    acc[b] += hsum_ps(s);
                }
            }
            for (int b = 0; b < B; b++) out[(size_t)b * n_rows + r] = acc[b];
        }
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
    float *workspace;   /* pre-allocated temp buffer, replaces __builtin_alloca */
    int ws_size;        /* total float count in workspace */
    /* Qwen3.6 hybrid SSM+attention support */
    int rope_dim;
    int full_attn_interval;
    const uint8_t **wQKV;
    const int *qkv_quant;
    const uint8_t **wAttnG;
    const int *attnG_quant;
    const float **ssm_conv1d;
    const float **ssm_a;
    const float **ssm_dt_bias;
    const float **ssm_alpha;
    const float **ssm_beta;
    const float **ssm_norm;
    const uint8_t **wSsmOut;
    const int *ssm_out_quant;
    float *ssm_state;
    const uint8_t **wSHexpG, **wSHexpU, **wSHexpD;
    const int *shexp_g_quant, *shexp_u_quant, *shexp_d_quant;
    const float **wShexpRouter;
    const int *layer_types;
    int n_layers_actual;
    /* ZAYA-specific */
    const uint8_t **w_zaya_ffn_gate_inp, **w_zaya_ffn_gate, **w_zaya_mlp2, **w_zaya_mlp4;
    const float **w_zaya_router_bias, **w_zaya_ffn_norm;
    const float **w_zaya_res_hs_w, **w_zaya_res_hs_b, **w_zaya_res_res_w, **w_zaya_res_res_b;
    int zaya_ffn_gate_inp_quant, zaya_ffn_gate_quant, zaya_mlp2_quant, zaya_mlp4_quant;
    int *has_moe_layer; /* 1 for odd layers, 0 for even */
    int zaya_expert_n; int zaya_moe_intermediate;
    /* CCA attention */
    const uint8_t **w_cca_v1, **w_cca_v2, **w_ssm_conv1d;
    int cca_v1_quant, cca_v2_quant;
} BC;

void moe_ffn(const BC *c, int l, float *x, float *gate_buf, float *up_buf, float *ffn_buf, int B) {
    int N = c->N, NE = c->n_experts, NK = c->n_experts_per_tok, M = c->moe_intermediate;
    size_t gate_stride = (size_t)M * (N / QK_K) * sizeof(block_q4_K);
    size_t up_stride = (size_t)M * (N / QK_K) * sizeof(block_q4_K);
    size_t down_stride = (size_t)N * (M / QK_K) * sizeof(block_q6_K);
    if (c->gate_exp_quant == 13) gate_stride = (size_t)M * (N / QK_K) * sizeof(block_q5_K);
    if (c->gate_exp_quant == 14) gate_stride = (size_t)M * (N / QK_K) * sizeof(block_q6_K);
    if (c->up_exp_quant == 13) up_stride = (size_t)M * (N / QK_K) * sizeof(block_q5_K);
    if (c->up_exp_quant == 14) up_stride = (size_t)M * (N / QK_K) * sizeof(block_q6_K);
    if (c->down_exp_quant == 12) down_stride = (size_t)N * (M / QK_K) * sizeof(block_q4_K);
    if (c->down_exp_quant == 13) down_stride = (size_t)N * (M / QK_K) * sizeof(block_q5_K);
    if (c->down_exp_quant == 14) down_stride = (size_t)N * (M / QK_K) * sizeof(block_q6_K);
    if (c->gate_exp_quant == 39) gate_stride = (size_t)M * (N / 32) * sizeof(block_mxfp4);
    if (c->up_exp_quant == 39) up_stride = (size_t)M * (N / 32) * sizeof(block_mxfp4);
    if (c->down_exp_quant == 39) down_stride = (size_t)N * (M / 32) * sizeof(block_mxfp4);
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
            quant_matmul_omp(gate_w, xb, gate_buf, M, N, c->gate_exp_quant);

            const uint8_t *up_w = c->w_up_exps[l] + (size_t)e * up_stride;
            quant_matmul_omp(up_w, xb, up_buf, M, N, c->up_exp_quant);

            for (int i = 0; i < M; i++) {
                float g = gate_buf[i];
                float sg = g / (1.0f + expf(-g));
                gate_buf[i] = sg * up_buf[i];
            }

            const uint8_t *down_w = c->w_down_exps[l] + (size_t)e * down_stride;
            quant_matmul_omp(down_w, gate_buf, up_buf, N, M, c->down_exp_quant);
            
            for (int i = 0; i < N; i++) ffn_buf[i] += weight * up_buf[i];
        }
    }
}

static inline void batch_matmul(int qt, const uint8_t *W, const float *x, float *out,
                                int n_rows, int nc, int B) {
    for (int b = 0; b < B; b++) {
        quant_matmul_omp(W, x + b*nc, out + b*n_rows, n_rows, nc, qt);
    }
}

void rope_init(float *cos_table, float *sin_table, int max_ctx, int hd, int rope_dim) {
    if (rope_dim <= 0) rope_dim = hd;
    int hd2 = rope_dim / 2;
    for (int pos = 0; pos < max_ctx; pos++) {
        for (int j = 0; j < hd2; j++) {
            double theta = (double)pos / pow(10000.0, 2.0 * j / hd);
            cos_table[pos * hd2 + j] = (float)cos(theta);
            sin_table[pos * hd2 + j] = (float)sin(theta);
        }
    }
}

static inline void rope_apply(float *buf, int nh, int hd, int pos,
                               const float *cos_t, const float *sin_t, int rope_dim) {
    if (rope_dim <= 0) rope_dim = hd;
    int hd2 = rope_dim / 2;
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

void ssm_decode_step(
    const float *x,           // input [B, N]
    float *ssm_intermediate,  // output [B, inner] — gated SSM intermediate before ssm_out
    const float *conv1d_w,    // [conv_kernel, 2*inner]  (8192)
    const float *a_param,     // [groups]
    const float *dt_bias,     // [groups]
    const float *alpha,       // [N, dt_rank]
    const float *beta,        // [N, dt_rank]
    const float *ssm_norm_w,  // [state_size]
    float *state,              // [groups, state_size] — MODIFIED in place
    int B, int N, int inner, int groups, int state_size,
    int conv_kernel, int dt_rank
) {
    for (int b = 0; b < B; b++) {
        const float *xb = x + b * N;
        float *interm = ssm_intermediate + b * inner;
        
        // 1. Conv1d (single tap — seq_len=1 for decode)
        float conv_buf[8192]; // temp for expanded conv output (max 2*inner)
        for (int i = 0; i < 2*inner; i++) {
            conv_buf[i] = xb[i % N] * conv1d_w[i];  // first tap only
        }
        
        // 2. Split into gate and ssm_input
        float *gate_in = conv_buf;              // first half
        float *ssm_in = conv_buf + inner;       // second half
        
        // 3. SSM step per group
        for (int g = 0; g < groups; g++) {
            float *sg = state + g * state_size;
            
            // Compute dt for this group
            float dt = dt_bias[g];
            for (int j = 0; j < dt_rank; j++) {
                dt += alpha[g * dt_rank + j] * xb[j % N];
            }
            // softplus
            dt = dt > 20.0f ? dt : logf(1.0f + expf(dt));
            
            // Discretize A
            float a_disc = expf(a_param[g] * dt);
            
            // B projection: Bx = beta[g] * x * dt
            float bx = 0;
            for (int j = 0; j < dt_rank; j++) {
                bx += beta[g * dt_rank + j] * xb[j % N];
            }
            bx *= dt;
            
            // State update: h = a_disc * h + bx (simplified — B is scalar per group)
            for (int s = 0; s < state_size; s++) {
                sg[s] = a_disc * sg[s] + (s == 0 ? bx : 0.0f);
            }
        }
        
        // 4. Apply SiLU to gate
        for (int i = 0; i < inner; i++) {
            float g = gate_in[i];
            interm[i] = (g / (1.0f + expf(-g))) * ssm_in[i];
        }
    }
}

/* ── Fused SSM layer forward for Qwen3.6 hybrid ──
 * Replaces 3 separate ctypes calls (ssm_decode + gate_matmul + ssm_out_matmul)
 * with one C call. Fuses: ssm_decode → gate_matmul → element_mul → ssm_out_matmul → residual.
 */
void ssm_layer_fused(
    const float *x,            // input [B, N] — already RMS-normed
    float *residual,           // [B, N] — pre-norm residual for add
    float *output,             // [B, N] — output = residual + oproj
    // SSM weights
    const float *conv1d_w, const float *a_param, const float *dt_bias,
    const float *alpha, const float *beta, const float *ssm_norm_w,
    float *state,              // [groups, state_size] — mutated in place
    // Gate matmul weights
    const uint8_t *gate_raw, int gate_nr, int gate_nc, int gate_qt,
    // SSM out matmul weights
    const uint8_t *ssm_out_raw, int ssm_out_nr, int ssm_out_nc, int ssm_out_qt,
    // Dimensions
    int B, int N, int inner, int groups, int state_size,
    int conv_kernel, int dt_rank
) {
    /* Step 1: ssm_decode_step — produces intermediate in temp buffer */
    float *ssm_intermediate = output;  /* reuse output as temp (size B*inner) */
    ssm_decode_step(x, ssm_intermediate,
                    conv1d_w, a_param, dt_bias, alpha, beta, ssm_norm_w,
                    state, B, N, inner, groups, state_size, conv_kernel, dt_rank);
    
    for (int b = 0; b < B; b++) {
        const float *xb = x + b * N;
        float *rb = residual + b * N;
        float *ob = output + b * N;
        float *interm = ssm_intermediate + b * inner;
        
        /* Step 2: gate_matmul — xn * gate_w → gate_buf */
        float gate_buf[8192];  /* max inner = 4096 */
        quant_matmul_omp(gate_raw, xb, gate_buf, gate_nr, gate_nc, gate_qt);
        
        /* Step 3: element-wise multiply */
        for (int i = 0; i < inner; i++) {
            interm[i] = gate_buf[i] * interm[i];
        }
        
        /* Step 4: ssm_out matmul — gate_buf * ssm_out_w → oproj */
        float oproj_buf[8192];  /* max N = 2048 */
        quant_matmul_omp(ssm_out_raw, interm, oproj_buf, ssm_out_nr, ssm_out_nc, ssm_out_qt);
        
        /* Step 5: residual add */
        for (int i = 0; i < N; i++) {
            ob[i] = rb[i] + oproj_buf[i];
        }
    }
}

/* ── Lightweight Qwen3.6 batch forward (no BC struct needed) ──
 * Processes ALL L layers in a single C call. Uses quant_matmul_omp for ALL matmuls.
 * Python passes flat pointer arrays + scalar params. Per-layer data is indexed
 * by the C code via the arrays of length L.
 *
 * EXTENDED signature: added moe_router_raw, moe_router_qt and shexp_*_qt params
 * that were missing from the original stub but are required for full MoE support.
 */
void qwen36_batch_forward(
    /* Scalars */
    int B, int L, int N, int NH, int n_kv_h, int HD, int FF,
    int n_experts, int top_k, int n_ff_expert, float eps,
    int rope_dim, int max_ctx, int ssm_inner, int ssm_groups,
    int ssm_state_size, int ssm_conv_kernel, int ssm_dt_rank,
    /* Embedding */
    const float *emb, int emb_quant,
    /* Layer type flags (1=SSM, 0=Attention) */
    const int *layer_types,
    /* Token IDs */
    const int *tokens,
    /* Output logits */
    float *logits,
    /* Per-layer weight pointers */
    const float **attn_norm_w, const float **ffn_norm_w,
    const uint8_t **attn_qkv_raw, const int *qkv_nr, const int *qkv_nc, const int *qkv_qt,
    const uint8_t **attn_gate_raw, const int *gate_nr, const int *gate_nc, const int *gate_qt,
    const uint8_t **ssm_out_raw, const int *ssm_out_nr, const int *ssm_out_nc, const int *ssm_out_qt,
    const uint8_t **attn_out_raw, const int *out_nr, const int *out_nc, const int *out_qt,
    uint8_t **gate_exp_raw, const int *gate_exp_qt,
    uint8_t **up_exp_raw, const int *up_exp_qt,
    uint8_t **down_exp_raw, const int *down_exp_qt,
    const float **shexp_gate_raw, const float **shexp_up_raw, const float **shexp_down_raw,
    const float **shexp_router,
    const int *shexp_int,
    /* ADDED: MoE router and shared expert quant types */
    const uint8_t **moe_router_raw, const int *moe_router_qt,
    const int *shexp_gate_qt, const int *shexp_up_qt, const int *shexp_down_qt,
    const float *out_norm_w,
    const uint8_t *out_w, int out_nr_final, int out_nc_final, int out_qt_final,
    /* SSM per-layer pointers */
    const float **ssm_conv1d, const float **ssm_a, const float **ssm_dt_bias,
    const float **ssm_alpha, const float **ssm_beta, const float **ssm_norm,
    float *ssm_state,
    /* KV cache (pre-allocated flat arrays) */
    float *kv_k, float *kv_v, int *kv_lens,
    const float *cos_table, const float *sin_table,
    /* Workspace buffers (size = B * S * 12 where S is max(N, inner, FF)+...) */
    float *ws, uint8_t *q8_ws
) {
    /* ── Workspace layout (12 slices of B*S floats) ── */
    int inner = NH * HD;           /* Q dimension = 4096 for Qwen3.6-35B */
    int n_kv = n_kv_h * HD;       /* K/V dimension per layer = 512 */
    int qk_dim = inner + 2 * n_kv; /* fused QKV: Q(4096) + K(512) + V(512) = 5120 */
    int shexp_inner = shexp_int ? shexp_int[0] : 512; /* shared expert intermediate dim */

    /* S = max(N, inner, FF, n_kv, qk_dim, n_ff_expert, shexp_inner) */
    int S = N;
    if (inner > S) S = inner;
    if (FF > S) S = FF;
    if (n_kv > S) S = n_kv;
    if (qk_dim > S) S = qk_dim;
    if (n_ff_expert > S) S = n_ff_expert;
    if (shexp_inner > S) S = shexp_inner;
    if (S < 8192) S = 8192;

    float *x      = ws;             /* [B,N] running hidden state */
    float *xn     = ws + B * S;     /* [B,N] RMS-normed input */
    float *res    = ws + 2 * B * S; /* [B,N] residual */
    float *qkv    = ws + 3 * B * S; /* [B,qk_dim] fused QKV buffer */
    float *k_tmp  = ws + 4 * B * S; /* [B,n_kv] K temp */
    float *v_tmp  = ws + 5 * B * S; /* [B,n_kv] V temp */
    float *att    = ws + 6 * B * S; /* [B,inner] attention/SSM intermediate */
    float *gate   = ws + 7 * B * S; /* [B,max(inner,n_ff_expert,shexp)] gate buf */
    float *up     = ws + 8 * B * S; /* [B,max(inner,n_ff_expert,shexp)] up buf */
    float *silu   = ws + 9 * B * S; /* [B,max(inner,n_ff_expert,shexp)] SiLU buf */
    float *oproj  = ws + 10 * B * S;/* [B,N] output projection */
    float *ffn    = ws + 11 * B * S;/* [B,N] FFN combined output */

    /* MoE local buffers */
    int top_idx_buf[16];  /* max top_k */
    float top_wt_buf[16]; /* max top_k */
    float router_scores[256]; /* max n_experts */
    const uint8_t *gate_ptrs[64]; /* per-expert gate pointers */
    const uint8_t *up_ptrs[64];
    const uint8_t *down_ptrs[64];

    fprintf(stderr, "DEBUG: qwen36_batch_forward: B=%d L=%d N=%d NH=%d n_kv_h=%d HD=%d FF=%d\n", B, L, N, NH, n_kv_h, HD, FF);
    fprintf(stderr, "DEBUG: n_experts=%d top_k=%d n_ff_expert=%d eps=%f\n", n_experts, top_k, n_ff_expert, eps);
    fprintf(stderr, "DEBUG: inner=%d n_kv=%d qk_dim=%d S=%d\n", inner, n_kv, qk_dim, S);
    fprintf(stderr, "DEBUG: emb=%p tokens=%p logits=%p\n", (void*)emb, (const void*)tokens, (void*)logits);
    fprintf(stderr, "DEBUG: ws=%p q8_ws=%p ssm_state=%p\n", (void*)ws, (void*)q8_ws, (void*)ssm_state);
    fprintf(stderr, "DEBUG: layer_types=%p kv_k=%p kv_v=%p kv_lens=%p\n", (const void*)layer_types, (void*)kv_k, (void*)kv_v, (const void*)kv_lens);

    /* ── Embedding lookup ── */
    for (int b = 0; b < B; b++) {
        if (emb_quant == 12)
            emb_lookup((const uint8_t*)emb, tokens[b], x + b * N, N, 12);
        else
            memcpy(x + b * N, emb + (size_t)tokens[b] * N, N * sizeof(float));
    }
    fprintf(stderr, "DEBUG: embedding done, x[0]=%f\n", x[0]);

    /* Helper: inline RMS norm with SIMD clip + store to xn, save to res */
#define RMS_NORM_SIMD(out_ptr, in_ptr, w_ptr, n, eps_val) do { \
    float *__out = (out_ptr); \
    const float *__in = (in_ptr); \
    const float *__w = (w_ptr); \
    int __n = (n); \
    float __ss = 0.0f; \
    int __i; \
    for (__i = 0; __i <= __n - 8; __i += 8) { \
        __m256 __xv = _mm256_loadu_ps(__in + __i); \
        __xv = _mm256_min_ps(_mm256_max_ps(__xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f)); \
        _mm256_storeu_ps(__out + __i, __xv); \
        __ss += hsum_ps(_mm256_mul_ps(__xv, __xv)); \
    } \
    for (; __i < __n; __i++) { \
        float __v = __in[__i]; \
        if (__v != __v) __v = 0.0f; else if (__v > 1000.0f) __v = 1000.0f; else if (__v < -1000.0f) __v = -1000.0f; \
        __out[__i] = __v; \
        __ss += __v * __v; \
    } \
    float __rms = sqrtf(__ss / __n + (eps_val)); \
    for (int __j = 0; __j < __n; __j++) { __out[__j] = (__out[__j] / __rms) * __w[__j]; } \
} while(0)

    /* Helper: compute per-expert weight stride based on quant type */
#define EXPERT_STRIDE(n_rows, n_cols, qt) \
    (((qt) == 12) ? ((size_t)(n_rows) * ((n_cols) / QK_K) * sizeof(block_q4_K)) : \
     ((qt) == 13) ? ((size_t)(n_rows) * ((n_cols) / QK_K) * sizeof(block_q5_K)) : \
     ((qt) == 14) ? ((size_t)(n_rows) * ((n_cols) / QK_K) * sizeof(block_q6_K)) : \
     ((qt) == 39) ? ((size_t)(n_rows) * ((n_cols) / 32) * sizeof(block_mxfp4)) : \
     ((qt) == 2)  ? ((size_t)(n_rows) * ((n_cols) / 32) * Q4_0_BS) : \
     ((qt) == 8)  ? ((size_t)(n_rows) * ((n_cols) / 32) * Q8_0_BS) : \
     0)

    /* ════════════════════════════════════════════════════════════════════
     * MAIN LAYER LOOP — processes all L layers in a single C call
     * ════════════════════════════════════════════════════════════════════ */
    for (int l = 0; l < L; l++) {
        int is_ssm = (layer_types && layer_types[l] == 1);
        fprintf(stderr, "DEBUG: layer=%d is_ssm=%d\n", l, is_ssm);
        fprintf(stderr, "DEBUG:   attn_norm_w[%d]=%p attn_qkv_raw[%d]=%p attn_gate_raw[%d]=%p ssm_out_raw[%d]=%p attn_out_raw[%d]=%p\n",
                l, (const void*)attn_norm_w[l], l, (const void*)attn_qkv_raw[l],
                l, (const void*)attn_gate_raw[l], l, (const void*)ssm_out_raw[l], l, (const void*)attn_out_raw[l]);

        /* ── STEP 1: Pre-Attention RMS Norm ──
         * xn = rms_norm(x, attn_norm_w[l]), res = x (pre-norm save) */
        for (int b = 0; b < B; b++) {
            memcpy(res + b * N, x + b * N, N * sizeof(float));
            RMS_NORM_SIMD(xn + b * N, x + b * N, attn_norm_w[l], N, eps);
        }

        /* ── STEP 2: SSM or Attention Block ── */
        if (is_ssm && ssm_conv1d && ssm_conv1d[l] != NULL) {
            /* ═══ SSM path ═══ */

            /* Step 2a: SSM decode → produces [B, ssm_inner] intermediate in 'att' */
            ssm_decode_step(xn, att,
                ssm_conv1d[l], ssm_a[l], ssm_dt_bias[l],
                ssm_alpha[l], ssm_beta[l], ssm_norm[l],
                ssm_state + (size_t)l * ssm_groups * ssm_state_size,
                B, N, ssm_inner, ssm_groups, ssm_state_size,
                ssm_conv_kernel, ssm_dt_rank);

            /* Step 2b: Gate matmul — xn * attn_gate_raw[l] → gate (N→inner) */
            for (int b = 0; b < B; b++) {
                quant_matmul_omp(attn_gate_raw[l], xn + b * N,
                    gate + b * ssm_inner, gate_nr[l], gate_nc[l], gate_qt[l]);
            }

            /* Step 2c: Element-wise gating — att = gate * att */
            for (int b = 0; b < B; b++) {
                float *gb = gate + b * ssm_inner;
                float *ab = att + b * ssm_inner;
                for (int i = 0; i < ssm_inner; i++) ab[i] = gb[i] * ab[i];
            }

            /* Step 2d: SSM out projection — att * ssm_out_raw[l] → oproj (inner→N) */
            for (int b = 0; b < B; b++) {
                quant_matmul_omp(ssm_out_raw[l], att + b * ssm_inner,
                    oproj + b * N, ssm_out_nr[l], ssm_out_nc[l], ssm_out_qt[l]);
            }

            /* Step 2e: Residual add — x = res + oproj */
            for (int b = 0; b < B; b++) {
                float *rb = res + b * N;
                float *op = oproj + b * N;
                float *xb = x + b * N;
                for (int i = 0; i < N; i++) xb[i] = rb[i] + op[i];
            }

        } else {
            /* ═══ Attention path ═══ */

            fprintf(stderr, "DEBUG: attention path layer=%d\n", l);
            fprintf(stderr, "DEBUG:   qkv_nr=%d qkv_nc=%d qkv_qt=%d\n", qkv_nr[l], qkv_nc[l], qkv_qt[l]);
            int n_qkv = qkv_nr[l];
            for (int b = 0; b < B; b++) {
                fprintf(stderr, "DEBUG:   b=%d qkv matmul\n", b);
                quant_matmul_omp(attn_qkv_raw[l], xn + b * N,
                    qkv, n_qkv, qkv_nc[l], qkv_qt[l]);
                fprintf(stderr, "DEBUG:   b=%d qkv done, inner=%d n_kv=%d\n", b, inner, n_kv);

                /* Split: Q = qkv[0:inner], K = qkv[inner:inner+N], V = qkv[inner+N:inner+2N] */
                float *q = qkv;
                float *k_src = qkv + inner;
                float *v_src = qkv + inner + N;

                /* Copy first NKH*HD elements of K and V to temp buffers */
                memcpy(k_tmp + b * n_kv, k_src, n_kv * sizeof(float));
                memcpy(v_tmp + b * n_kv, v_src, n_kv * sizeof(float));

                /* RoPE applied to Q and K */
                rope_apply(q, NH, HD, kv_lens[l], cos_table, sin_table, rope_dim);
                rope_apply(k_tmp + b * n_kv, n_kv_h, HD, kv_lens[l], cos_table, sin_table, rope_dim);

                /* KV cache store (flat arrays: [L][max_ctx][n_kv]) */
                int pos = kv_lens[l];
                size_t kv_offset = (size_t)l * max_ctx * n_kv + (size_t)pos * n_kv;
                memcpy(kv_k + kv_offset, k_tmp + b * n_kv, n_kv * sizeof(float));
                memcpy(kv_v + kv_offset, v_tmp + b * n_kv, n_kv * sizeof(float));

                /* GQA attention — reads from flat KV cache */
                int seq_len = pos + 1;
                float *k_cache = kv_k + (size_t)l * max_ctx * n_kv;
                float *v_cache = kv_v + (size_t)l * max_ctx * n_kv;
                gqa(att + b * inner, q, k_cache, v_cache, seq_len, NH, n_kv_h, HD);
            }

            /* Step 2b: Output projection — att * attn_out_raw[l] → oproj (inner→N) */
            for (int b = 0; b < B; b++) {
                quant_matmul_omp(attn_out_raw[l], att + b * inner,
                    oproj + b * N, out_nr[l], out_nc[l], out_qt[l]);
            }

            /* Step 2c: Residual add — x = res + oproj */
            for (int b = 0; b < B; b++) {
                float *rb = res + b * N;
                float *op = oproj + b * N;
                float *xb = x + b * N;
                for (int i = 0; i < N; i++) xb[i] = rb[i] + op[i];
            }
        }

        /* ── STEP 3: Post-Attention RMS Norm (FFN norm) ──
         * xn = rms_norm(x, ffn_norm_w[l]), res = x (pre-norm save) */
        for (int b = 0; b < B; b++) {
            memcpy(res + b * N, x + b * N, N * sizeof(float));
            RMS_NORM_SIMD(xn + b * N, x + b * N, ffn_norm_w[l], N, eps);
        }

        /* ── STEP 4: MoE FFN + Shared Expert ── */
        int has_moe = (n_experts > 0 && top_k > 0 && gate_exp_raw &&
                       gate_exp_raw[l] != NULL && moe_router_raw && moe_router_raw[l] != NULL);

        if (has_moe) {
            for (int b = 0; b < B; b++) {
                float *xnb = xn + b * N;
                float *xb = x + b * N;
                float *rb = res + b * N;

                /* ── Router: quant_matmul_omp(xn, moe_router[l]) → router_scores ── */
                quant_matmul_omp(moe_router_raw[l], xnb, router_scores,
                    n_experts, N, moe_router_qt[l]);

                /* ── Softmax ── */
                float mx = router_scores[0];
                for (int e = 1; e < n_experts; e++)
                    if (router_scores[e] > mx) mx = router_scores[e];
                float sum = 0.0f;
                for (int e = 0; e < n_experts; e++) {
                    router_scores[e] = expf(router_scores[e] - mx);
                    sum += router_scores[e];
                }
                float inv_sum = 1.0f / (sum + 1e-10f);
                for (int e = 0; e < n_experts; e++) router_scores[e] *= inv_sum;

                /* ── Top-K selection ── */
                for (int k = 0; k < top_k; k++) {
                    top_wt_buf[k] = -1e30f;
                    top_idx_buf[k] = -1;
                }
                for (int e = 0; e < n_experts; e++) {
                    float v = router_scores[e];
                    for (int k = 0; k < top_k; k++) {
                        if (v > top_wt_buf[k]) {
                            for (int k2 = top_k - 1; k2 > k; k2--) {
                                top_wt_buf[k2] = top_wt_buf[k2-1];
                                top_idx_buf[k2] = top_idx_buf[k2-1];
                            }
                            top_wt_buf[k] = v;
                            top_idx_buf[k] = e;
                            break;
                        }
                    }
                }
                /* Renormalize top weights */
                float tw_sum = 0.0f;
                for (int k = 0; k < top_k; k++) tw_sum += top_wt_buf[k];
                float tw_inv = 1.0f / (tw_sum + 1e-10f);
                for (int k = 0; k < top_k; k++) top_wt_buf[k] *= tw_inv;

                /* ── Build per-expert pointer arrays ── */
                int gqt = gate_exp_qt ? gate_exp_qt[l] : 12;
                int uqt = up_exp_qt ? up_exp_qt[l] : 12;
                int dqt = down_exp_qt ? down_exp_qt[l] : 14;
                size_t g_stride = EXPERT_STRIDE(n_ff_expert, N, gqt);
                size_t u_stride = EXPERT_STRIDE(n_ff_expert, N, uqt);
                size_t d_stride = EXPERT_STRIDE(N, n_ff_expert, dqt);
                for (int e = 0; e < n_experts; e++) {
                    gate_ptrs[e] = gate_exp_raw[l] + (size_t)e * g_stride;
                    up_ptrs[e]   = up_exp_raw[l]   + (size_t)e * u_stride;
                    down_ptrs[e] = down_exp_raw[l] + (size_t)e * d_stride;
                }

                /* ── Call moe_forward_omp ── */
                memset(ffn + b * N, 0, N * sizeof(float));
                moe_forward_omp(gate_ptrs, up_ptrs, down_ptrs,
                    xnb, n_ff_expert, N,
                    gqt, uqt, dqt,
                    top_idx_buf, top_wt_buf, top_k,
                    ffn + b * N,
                    gate,  /* reuse gate buffer as prealloc */
                    q8_ws);

                /* ── Shared Expert (Qwen3.6) ── */
                if (shexp_router && shexp_router[l] != NULL) {
                    /* Router score = dot(xn, shexp_router[l]) — 1D vector [N] */
                    float sh_score = 0.0f;
                    for (int i = 0; i < N; i++) sh_score += xnb[i] * shexp_router[l][i];
                    if (sh_score > 0.0f) {
                        int si = shexp_int ? shexp_int[l] : 512;
                        int sgq = shexp_gate_qt ? shexp_gate_qt[l] : 39;
                        int suq = shexp_up_qt   ? shexp_up_qt[l]   : 39;
                        int sdq = shexp_down_qt ? shexp_down_qt[l] : 39;
                        /* Gate: xn * shexp_gate → gate_buf (N→si) */
                        quant_matmul_omp((const uint8_t*)shexp_gate_raw[l],
                            xnb, gate, si, N, sgq);
                        /* Up: xn * shexp_up → up_buf (N→si) */
                        quant_matmul_omp((const uint8_t*)shexp_up_raw[l],
                            xnb, up, si, N, suq);
                        /* SiLU gate: gate / (1+exp(-gate)) */
                        for (int i = 0; i < si; i++) {
                            float gv = gate[i];
                            if (gv < -80.0f) gv = -80.0f;
                            if (gv > 80.0f) gv = 80.0f;
                            gate[i] = (gv / (1.0f + expf(-gv))) * up[i];
                        }
                        /* Down: gate_buf * shexp_down → up_buf (si→N) */
                        quant_matmul_omp((const uint8_t*)shexp_down_raw[l],
                            gate, up, N, si, sdq);
                        /* Accumulate: ffn += sh_score * down_output */
                        for (int i = 0; i < N; i++)
                            ffn[b * N + i] += sh_score * up[i];
                    }
                }

                /* ── Residual add: x = res + ffn ── */
                for (int i = 0; i < N; i++) xb[i] = rb[i] + ffn[b * N + i];
            }
        } else if (0) {
            /* Fallback: no-op — x stays as was (residual already saved) */
        }

        /* ── Advance KV length ── */
        /* (kv_lens[l] was already used for pos above; increment for next token) */
        /* NOTE: kv_lens is updated by the caller between forward calls */
    }

    /* ── STEP 5: Final RMS Norm + Output Projection ── */
    for (int b = 0; b < B; b++) {
        RMS_NORM_SIMD(xn + b * N, x + b * N, out_norm_w, N, eps);
        quant_matmul_omp(out_w, xn + b * N,
            logits + (size_t)b * out_nr_final,
            out_nr_final, out_nc_final, out_qt_final);
    }

#undef RMS_NORM_SIMD
#undef EXPERT_STRIDE
}

void batch_forward(const BC *c, const int *tokens, int B, float *ws) {
    int N=c->N, NH=c->NH, NKH=c->NKH, HD=c->HD, FF=c->FF, L=c->L, nc=c->nc;
    int inner = NH * HD;
    int S = N; if (NH*HD > S) S = NH*HD; if (FF > S) S = FF; if (NKH*HD > S) S = NKH*HD;
    int qk_fused = NH*HD + NKH*HD;
    if (qk_fused > S) S = qk_fused;
    if (S < 8192) S = 8192;
    
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
    
    int ssm_groups = 32, ssm_state_size = 128, ssm_dt_rank = 32;
    int ssm_conv_kernel = 4;
    
    for (int l = 0; l < L; l++) {
        
    /* ── Pre-RMS Norm (attn_norm) ── */
    for (int b = 0; b < B; b++) {
        float *xb = x + b*N;
        float *rb = res + b*N;
        float ss = 0;
        for(int i=0;i<=N-8;i+=8){
            __m256 xv = _mm256_loadu_ps(xb+i);
            xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
            _mm256_storeu_ps(xb+i, xv);
            _mm256_storeu_ps(rb+i, xv);
            ss += hsum_ps(_mm256_mul_ps(xv, xv));
        }
        for(int i=N-(N%8);i<N;i++){
            float v = xb[i];
            if(v != v) v = 0; else if(v > 1000.0f) v = 1000.0f; else if(v < -1000.0f) v = -1000.0f;
            xb[i] = v; rb[i] = v;
            ss += v * v;
        }
        float rms_val = sqrtf(ss / N + c->eps);
        for(int j=0;j<N;j++) xb[j] = (xb[j] / rms_val) * c->wAN[l][j];
    }
        
    /* ── Hybrid SSM/Attention Block ── */
    int is_ssm = (c->layer_types && c->layer_types[l] == 1);
    
        if (is_ssm && c->ssm_conv1d && c->ssm_conv1d[l] != NULL) {
            /* ═══ SSM path ═══ */
        int groups = ssm_groups, state_size = ssm_state_size;
        int dt_rank = ssm_dt_rank;
        
        /* Step 1: SSM decode step → produces inner-dim (4096) intermediate in 'att' */
        ssm_decode_step(xn, att,
            c->ssm_conv1d[l], c->ssm_a[l], c->ssm_dt_bias[l],
            c->ssm_alpha[l], c->ssm_beta[l], c->ssm_norm[l],
            c->ssm_state + (size_t)l * groups * state_size,
            B, N, inner, groups, state_size, ssm_conv_kernel, dt_rank);
        
        /* Step 2: attn_gate(xn) → gate_4096 (maps N→inner) */
        if (c->wAttnG) {
            batch_matmul(c->attnG_quant[l], c->wAttnG[l], xn, gate, inner, N, B);  /* n_rows=inner=4096, nc=N=2048 */
        } else {
            memset(gate, 0, B * inner * sizeof(float));
        }
        
        /* Step 3: element-wise gating: combined = gate * intermediate */
        for (int b = 0; b < B; b++) {
            float *gb = gate + b * inner;
            float *ab = att + b * inner;
            for (int i = 0; i < inner; i++) ab[i] = gb[i] * ab[i];
        }
        
        /* Step 4: ssm_out projection (inner→N) via c->wSsmOut */
        if (c->wSsmOut) {
            /* wSsmOut[inner, N] Q8_0 — n_rows=N=2048, nc=inner=4096 */
            batch_matmul(c->ssm_out_quant[l], c->wSsmOut[l], att, oproj, N, inner, B);
        } else {
            memset(oproj, 0, B * N * sizeof(float));
        }
        
        /* Step 5: residual */
        for (int b = 0; b < B; b++)
            for (int i = 0; i < N; i++) x[b*N+i] = res[b*N+i] + oproj[b*N+i];
            
        } else {
            /* ═══ Full Attention path ═══ */
            /* Use attn_qkv fused weight when available, else separate Q/K/V */
            if (c->wQKV && c->wQKV[l] != NULL) {
            /* attn_qkv maps N→8192 (Q:4096, K:2048, V:2048) */
            int n_qkv = 8192;
            batch_matmul(c->qkv_quant[l], c->wQKV[l], xn, q, n_qkv, nc, B);
            /* Split: first inner=4096 = Q, next N=2048 = K, next N=2048 = V */
            for (int b = 0; b < B; b++) {
                float *qb = q + (size_t)b * n_qkv;
                /* Q is qb[0:inner] = already in q buffer */
                /* Copy K (first NKH*HD=512 from K section) to k buffer */
                memcpy(k + (size_t)b * NKH * HD,
                       qb + inner, NKH * HD * sizeof(float));
                /* Copy V (first NKH*HD from V section) to v buffer */
                memcpy(v + (size_t)b * NKH * HD,
                       qb + inner + N, NKH * HD * sizeof(float));
            }
        } else if (c->wQK && c->wQK[l] != NULL) {
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
            float *qb = q + b*inner; /* Q starts at beginning, dimension = NH*HD */
            float *kb = k + b*NKH*HD;
            float *vb = v + b*NKH*HD;
            KVBlock *kv = c->kv_array[b];
            int pos = kv->seq_len[0];
            rope_apply(qb, NH, HD, pos, c->cos_table, c->sin_table, c->rope_dim);
            rope_apply(kb, NKH, HD, pos, c->cos_table, c->sin_table, c->rope_dim);
            
            size_t base=(size_t)l*MAX_BLOCKS*BLOCK_SIZE*NKH*HD;
            int blk=pos/BLOCK_SIZE,off=pos%BLOCK_SIZE;
            PageTable *pt = &kv->pt;
            int bid = (blk < MAX_BLOCKS) ? pt->table[blk] : -1;
            if(bid<0){bid=kv_alloc(kv);if(blk<MAX_BLOCKS)pt->table[blk]=bid;}
            memcpy(kv->k+base+(size_t)bid*BLOCK_SIZE*NKH*HD+off*NKH*HD, kb, NKH*HD*sizeof(float));
            memcpy(kv->v+base+(size_t)bid*BLOCK_SIZE*NKH*HD+off*NKH*HD, vb, NKH*HD*sizeof(float));
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
            int kct_size = c->max_ctx * NKH * HD;
            if (c->ws_size < kct_size * 2 + 4096) {
                fprintf(stderr, "ERROR: workspace too small (need %d floats, have %d)\n", kct_size * 2 + 4096, c->ws_size);
                exit(1);
            }
            float *kct = c->workspace;
            float *vct = c->workspace + kct_size;
            for(int s=0;s<sl;s++){
                int sb=s/BLOCK_SIZE,so=s%BLOCK_SIZE;
                int sbid = (sb < MAX_BLOCKS) ? pt->table[sb] : -1;
                if(sbid<0) continue;
                size_t page_data = (size_t)pt->page_size * NKH * HD;
                size_t layer_stride = (size_t)pt->n_pages * page_data;
                size_t k_base_pt = (size_t)l * layer_stride;
                size_t v_base_pt = (size_t)pt->_n_layers * layer_stride + k_base_pt;
                size_t src_k = k_base_pt + (size_t)sbid * page_data + (size_t)so * NKH * HD;
                size_t src_v = v_base_pt + (size_t)sbid * page_data + (size_t)so * NKH * HD;
                memcpy(kct+s*NKH*HD, pt->pages+src_k, NKH*HD*sizeof(float));
                memcpy(vct+s*NKH*HD, pt->pages+src_v, NKH*HD*sizeof(float));
            }
            gqa(att+b*inner, qb, kct, vct, sl, NH, NKH, HD);
        }
        
        /* ═══ attn_gate: map attention output (inner=4096) → N (2048) ═══ */
        if (c->wAttnG && c->wAttnG[l] != NULL) {
            batch_matmul(c->attnG_quant[l], c->wAttnG[l], att, oproj, N, inner, B);
        } else {
            batch_matmul(c->o_quant[l], c->wO[l], att, oproj, c->nO[l], inner, B);
        }
        for(int b=0;b<B;b++)for(int i=0;i<N;i++)x[b*N+i]=res[b*N+i]+oproj[b*N+i];
    }
    
    /* ── Post-attention RMS Norm (ffn_norm / post_attention_norm) ── */
    for(int b=0;b<B;b++){
        float *xb = x + b*N;
        float *rb = res + b*N;
        float ss = 0;
        for(int i=0;i<=N-8;i+=8){
            __m256 xv = _mm256_loadu_ps(xb+i);
            xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
            _mm256_storeu_ps(xb+i, xv);
            _mm256_storeu_ps(rb+i, xv);
            ss += hsum_ps(_mm256_mul_ps(xv, xv));
        }
        for(int i=N-(N%8);i<N;i++){
            float v = xb[i];
            if(v != v) v = 0; else if(v > 1000.0f) v = 1000.0f; else if(v < -1000.0f) v = -1000.0f;
            xb[i] = v; rb[i] = v;
            ss += v * v;
        }
        float rms_val = sqrtf(ss / N + c->eps);
        for(int j=0;j<N;j++) xb[j] = (xb[j] / rms_val) * c->wFN[l][j];
    }
    
    /* ── MoE FFN + Shared Expert ── */
    if (c->n_experts > 0 && c->n_experts_per_tok > 0) {
        for(int b=0;b<B;b++){
            float *gate_buf = gate; float *up_buf = silu; float *ffn_buf = ffn;
            memset(ffn_buf, 0, N * sizeof(float));
            moe_ffn(c, l, xn + b*N, gate_buf, up_buf, ffn_buf, 1);
            
            /* Shared expert — always active (Qwen3.6) */
            if (c->wSHexpG && c->wSHexpG[l] != NULL) {
                float *xb = xn + b*N;
                /* Router score: dot product with wShexpRouter[l] (1D vector [N]) */
                float shexp_score = 0;
                for (int i = 0; i < N; i++) shexp_score += xb[i] * c->wShexpRouter[l][i];
                if (shexp_score > 0) {
                    int shexp_int = 512; /* shared expert FF dim */
                    batch_matmul(c->shexp_g_quant[l], c->wSHexpG[l], xb, gate_buf, shexp_int, N, 1);
                    batch_matmul(c->shexp_u_quant[l], c->wSHexpU[l], xb, up_buf, shexp_int, N, 1);
                    for (int i = 0; i < shexp_int; i++)
                        up_buf[i] = (up_buf[i] / (1.0f + expf(-up_buf[i]))) * gate_buf[i];
                    batch_matmul(c->shexp_d_quant[l], c->wSHexpD[l], up_buf, gate_buf, N, shexp_int, 1);
                    for (int i = 0; i < N; i++) ffn_buf[i] += shexp_score * gate_buf[i];
                }
            }
            
            for(int i=0;i<N;i++)x[b*N+i]=res[b*N+i]+ffn_buf[i];
        }
    } else if (c->wG && c->wG[l]) {
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
        float ss = 0;
        for(int i=0;i<=N-8;i+=8){
            __m256 xv = _mm256_loadu_ps(xb+i);
            xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
            _mm256_storeu_ps(xb+i, xv);
            _mm256_storeu_ps(rb+i, xv);
            ss += hsum_ps(_mm256_mul_ps(xv, xv));
        }
        for(int i=N-(N%8);i<N;i++){
            float v = xb[i];
            if(v != v) v = 0; else if(v > 1000.0f) v = 1000.0f; else if(v < -1000.0f) v = -1000.0f;
            xb[i] = v; rb[i] = v;
            ss += v * v;
        }
        float rms_val = sqrtf(ss / N + c->eps);
        for(int j=0;j<N;j++) xb[j] = (xb[j] / rms_val) * c->onw[j];
    }
    batch_matmul(c->outQuant, c->wOut, x, c->logits, c->outNR, c->outNC, B);
}

/* ═════════════════════════════════════════════════════════════════════════
 * ZAYA batch_forward — interleaved attention/MoE, CCA, res_scales
 */
void zaya_batch_forward(const BC *c, const int *tokens, int B, float *ws) {
    int N=c->N, NH=c->NH, NKH=c->NKH, HD=c->HD, L=c->L;
    int nq=NH*HD, nk=NKH*HD;
    int S = nq; if (nk > S) S = nk; if (N > S) S = N;
    if (S < 8192) S = 8192;
    float *x = ws, *xn = ws + 4*S, *res = ws + 8*S;
    float *q = ws + 12*S, *k_ = ws + 16*S;
    float *att = ws + 20*S, *gate_buf = ws + 24*S, *up_buf = ws + 28*S;
    float *oproj = ws + 32*S, *ffn = ws + 36*S, *h_buf = ws + 40*S;
    int moe_ff = c->zaya_moe_intermediate;
    for (int b = 0; b < B; b++)
        memcpy(x + b*N, c->emb + (size_t)tokens[b]*N, N*sizeof(float));
    for (int l = 0; l < L; l++) {
        int is_moe = c->has_moe_layer ? c->has_moe_layer[l] : (l % 2);
        for (int b = 0; b < B; b++) {
            float *xb = x + b*N, *rb = res + b*N; float ss = 0;
            for (int i = 0; i <= N-8; i+=8) {
                __m256 xv = _mm256_loadu_ps(xb+i);
                xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
                _mm256_storeu_ps(xb+i, xv); _mm256_storeu_ps(rb+i, xv);
                ss += hsum_ps(_mm256_mul_ps(xv, xv));
            }
            for (int i = N-(N%8); i < N; i++) { float v=xb[i];if(v!=v)v=0;else if(v>1000)v=1000;else if(v<-1000)v=-1000;xb[i]=v;rb[i]=v;ss+=v*v; }
            float rms_val = sqrtf(ss/N + c->eps);
            for (int j = 0; j < N; j++) xb[j] = (xb[j]/rms_val)*c->wAN[l][j];
        }
        if (!is_moe) {
            batch_matmul(c->q_quant[l],c->wQ[l],xn,q,nq,N,B);
            batch_matmul(c->k_quant[l],c->wK[l],xn,k_,nk,N,B);
            for (int b = 0; b < B; b++) {
                float *qb=q+b*nq,*kb=k_+b*nk;
                rope_apply(qb,NH,HD,c->max_ctx,c->cos_table,c->sin_table,c->rope_dim);
                rope_apply(kb, NKH,HD,c->max_ctx,c->cos_table,c->sin_table,c->rope_dim);
                KVBlock *kv = c->kv_array[l]; int pos = kv->seq_len[0]++;
                memcpy(kv->k+(size_t)pos*nk,kb,nk*sizeof(float));
                memcpy(kv->v+(size_t)pos*nk,kb,nk*sizeof(float));
                gqa(att+b*nq,qb,kv->k,kv->v,NH,NKH,HD,pos+1);
                batch_matmul(c->o_quant[l],c->wO[l],att+b*nq,oproj+b*N,N,nq,1);
                float *xbo=x+b*N,*rbo=res+b*N;
                for(int i=0;i<N;i++) xbo[i]=rbo[i]*c->w_zaya_res_res_w[l][i]+c->w_zaya_res_res_b[l][i]+oproj[b*N+i]*c->w_zaya_res_hs_w[l][i]+c->w_zaya_res_hs_b[l][i];
            }
        } else {
            int rn = 256;
            for (int b = 0; b < B; b++) {
                float *xb = xn + b*N; int ne = c->zaya_expert_n;
                float h[256], scores[32];
                batch_matmul(c->zaya_ffn_gate_inp_quant,c->w_zaya_ffn_gate_inp[l],xb,h,rn,N,1);
                batch_matmul(c->zaya_ffn_gate_quant,c->w_zaya_ffn_gate[l],h,h_buf,rn,rn,1);
                for(int i=0;i<rn;i++){float v=h_buf[i];h[i]=0.5f*v*(1.0f+tanhf(0.7978845608028654f*(v+0.044715f*v*v*v)));}
                batch_matmul(c->zaya_mlp2_quant,c->w_zaya_mlp2[l],h,h_buf,rn,rn,1);
                batch_matmul(c->zaya_mlp4_quant,c->w_zaya_mlp4[l],h_buf,scores,ne+1,rn,1);
                if(c->w_zaya_router_bias[l]) for(int e=0;e<ne+1;e++) scores[e] += c->w_zaya_router_bias[l][e];
                float mx=scores[0];for(int e=1;e<ne+1;e++)if(scores[e]>mx)mx=scores[e];
                float sum=0;for(int e=0;e<ne+1;e++){scores[e]=expf(scores[e]-mx);sum+=scores[e];}
                float inv=1.0f/(sum+1e-10f); for(int e=0;e<ne+1;e++) scores[e]*=inv;
                int ec=0;float ew=scores[0];for(int e=1;e<ne;e++){if(scores[e]>ew){ew=scores[e];ec=e;}}
                memset(ffn+b*N,0,N*sizeof(float));
                if(ec<ne&&ew>0.01f){
                    float renorm=0;for(int e=0;e<ne;e++)renorm+=scores[e];ew/=renorm;
                    int f2=moe_ff/2;
                    batch_matmul(c->gate_exp_quant,c->w_gate_exps[l]+(size_t)ec*moe_ff*(N/32)*34,xb,up_buf,moe_ff,N,1);
                    for(int i=0;i<f2;i++){float g=up_buf[i];up_buf[i]=(g/(1.0f+expf(-g)))*up_buf[f2+i];}
                    batch_matmul(c->down_exp_quant,c->w_down_exps[l]+(size_t)ec*N*(moe_ff/32)*34,up_buf,ffn+b*N,N,f2,1);
                    for(int i=0;i<N;i++) ffn[b*N+i]*=ew;
                }
                float *xbo=x+b*N,*rbo=res+b*N;
                for(int i=0;i<N;i++) xbo[i]=rbo[i]*c->w_zaya_res_res_w[l][i]+c->w_zaya_res_res_b[l][i]+ffn[b*N+i]*c->w_zaya_res_hs_w[l][i]+c->w_zaya_res_hs_b[l][i];
            }
        }
    }
    for(int b=0;b<B;b++){
        float *xb=x+b*N;float ss=0;
        for(int i=0;i<=N-8;i+=8){__m256 xv=_mm256_loadu_ps(xb+i);xv=_mm256_min_ps(_mm256_max_ps(xv,_mm256_set1_ps(-1000.0f)),_mm256_set1_ps(1000.0f));_mm256_storeu_ps(xb+i,xv);ss+=hsum_ps(_mm256_mul_ps(xv,xv));}
        for(int i=N-(N%8);i<N;i++){float v=xb[i];ss+=v*v;}
        float rms_val = sqrtf(ss/N+c->eps);
        for(int j=0;j<N;j++) xb[j]=(xb[j]/rms_val)*c->onw[j];
        batch_matmul(c->outQuant,c->wOut,xb,c->logits+(size_t)b*c->V,c->V,N,1);
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Gemma4 full forward pass — all 35 layers in C, single call from Python.
 * Eliminates ~35ms Python overhead per token.
 *
 * Parameters are passed as flat arrays (indexed by layer) from Python ctypes.
 */
void gemma4_forward_c(
    /* Input tokens */
    const int *tokens, int B,
    /* Architecture */
    int L, int N, int NH, int NKH, int V, int PL, int SW, int FF_half,
    float eps, float logit_cap,
    /* Embedding */
    const float *emb,
    /* Output weights */
    const uint8_t *w_out, int qt_out, int nr_out, int nc_out,
    const float *out_norm_w,
    /* Per-layer arrays of pointers (L elements each) */
    const float **attn_norm_w, const float **ffn_norm_w,
    const float **post_attn_norm_w, const float **post_ffw_norm_w,
    const float **post_norm_w, const float **layer_scale,
    const float **q_norm_w, const float **k_norm_w,
    const float **proj_w, const float **inp_gate_w,
    const uint8_t **wq, const int *q_nr, const int *q_qt,
    const uint8_t **wk, const int *k_nr, const int *k_qt,
    const uint8_t **wv, const int *v_nr, const int *v_qt,
    const uint8_t **wo, const int *o_nr, const int *o_qt,
    const uint8_t **wg, const int *g_nr, const int *g_qt,
    const uint8_t **wu, const int *u_nr, const int *u_qt,
    const uint8_t **wd, const int *d_nr, const int *d_qt,
    /* Per-layer metadata */
    const int *head_dims, const int *is_swa_arr,
    const int *kv_idx_arr, const int *rope_dim_arr,
    const float *freq_base_arr,
    /* KV cache */
    float *kv_k, float *kv_v, int *kv_lens, int max_ctx,
    /* RoPE tables (flat, max_ctx * rope_dim/2 per table) */
    const float *cos_rope, const float *sin_rope,
    /* Output */
    float *logits,
    /* Workspace: needs 14 * max(N, NH*512, FF_half, PL) floats */
    float *ws
) {
    /* Workspace layout */
    int S = N > NH*512 ? N : NH*512;
    if (FF_half > S) S = FF_half;
    if (PL > S) S = PL;
    S = (S + 31) & ~31; /* align to 32 */
    if (S < 2048) S = 2048;

    float *x    = ws;         /* [B, N] hidden state */
    float *xn   = ws + 1*B*N; /* [B, N] normed state */
    float *res  = ws + 2*B*N; /* [B, N] residual */
    float *q    = ws + 3*B*S;   /* [B, NH*max_hd] query */
    float *k_   = ws + 4*B*S;   /* [B, NKH*max_hd] key */
    float *v    = ws + 5*B*S;   /* [B, NKH*max_hd] value */
    float *att  = ws + 6*B*S;   /* [B, NH*max_hd] attention out */
    float *oproj= ws + 7*B*S;   /* [B, N] O projection */
    float *gate = ws + 8*B*S;   /* [B, FF_half] gate */
    float *up   = ws + 9*B*S;   /* [B, FF_half] up */
    float *silu = ws + 10*B*S;  /* [B, FF_half] gelu(gate)*up */
    float *ffn  = ws + 11*B*S;  /* [B, N] FFN output */
    float *per  = ws + 12*B*S;  /* [B, PL] per-layer signal */
    float *pg   = ws + 13*B*S;  /* [B, N] per-layer gated */

    /* For each token (B=1 for decode) */
    fprintf(stderr, "C: B=%d L=%d N=%d NH=%d NKH=%d V=%d PL=%d SW=%d FF=%d\n", B, L, N, NH, NKH, V, PL, SW, FF_half);
    for (int b = 0; b < B; b++) {
        float *bx = x + b*N;
        float *bxn = xn + b*N;
        float *br = res + b*N;
        float *bq = q + b*S;
        float *bk = k_ + b*S;
        float *bv = v + b*S;
        float *batt = att + b*S;
        float *bop = oproj + b*N;
        float *bg = gate + b*S;
        float *bu = up + b*S;
        float *bs = silu + b*S;
        float *bf = ffn + b*N;
        float *bp = per + b*S;
        float *bpg = pg + b*N;

        /* Embedding */
        memcpy(bx, emb + (size_t)tokens[b] * N, N * sizeof(float));

        for (int l = 0; l < L; l++) {
            fprintf(stderr, "C: L%d hd=%d\n", l, head_dims[l]);
            int hd = head_dims[l];
            int nq = q_nr[l], nk = k_nr[l], nv = v_nr[l];
            int no = o_nr[l], ng = g_nr[l], nu = u_nr[l], nd = d_nr[l];
            int swa = is_swa_arr[l];
            int kvi = kv_idx_arr[l];
            int rd = rope_dim_arr[l];
            float fb = freq_base_arr[l];

            /* ── 1. Pre-attention RMS norm ── */
            float ss = 0;
            for (int i = 0; i < N; i += 8) {
                __m256 xv = _mm256_loadu_ps(bx + i);
                xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
                _mm256_storeu_ps(bx + i, xv);
                _mm256_storeu_ps(br + i, xv);
                ss += hsum_ps(_mm256_mul_ps(xv, xv));
            }
            for (int i = N - (N % 8); i < N; i++) {
                float vv = bx[i];
                if (vv != vv) vv = 0; else if (vv > 1000) vv = 1000; else if (vv < -1000) vv = -1000;
                bx[i] = vv; br[i] = vv; ss += vv * vv;
            }
            float ir = 1.0f / sqrtf(ss / N + eps);
            for (int j = 0; j < N; j++) bxn[j] = bx[j] * ir * attn_norm_w[l][j];
            fprintf(stderr, "  rms OK ss=%f\n", ss);

            /* ── 2. Per-layer signal ── */
            if (proj_w[l] && inp_gate_w[l]) {
                for (int j = 0; j < PL; j++) {
                    float dot = 0;
                    for (int i = 0; i < N; i++) dot += bxn[i] * proj_w[l][j * N + i];
                    bp[j] = dot;
                }
                for (int j = 0; j < N; j++) {
                    float dot = 0;
                    for (int i = 0; i < PL; i++) dot += bp[i] * inp_gate_w[l][j * PL + i];
                    bpg[j] = dot;
                }
            } else { memset(bpg, 0, N * sizeof(float)); }
            fprintf(stderr, "  per sig OK\n");

            /* ── 3. QKV projections ── */
            fprintf(stderr, "  Q nq=%d N=%d qt=%d\n", nq, N, q_qt[l]);
            batch_matmul(q_qt[l], wq[l], bxn, bq, nq, N, 1);
            batch_matmul(k_qt[l], wk[l], bxn, bk, nk, N, 1);
            batch_matmul(v_qt[l], wv[l], bxn, bv, nv, N, 1);
            fprintf(stderr, "  QKV done\n");

            /* ── 4. Q/K per-head RMS norm ── */
            if (q_norm_w[l]) {
                for (int h = 0; h < NH; h++) {
                    float *qh = bq + h * hd;
                    float s = 0;
                    for (int i = 0; i < hd; i++) s += qh[i] * qh[i];
                    float irq = 1.0f / sqrtf(s / hd + eps);
                    for (int i = 0; i < hd; i++) qh[i] *= irq * q_norm_w[l][h * hd + i];
                }
            }
            if (k_norm_w[l]) {
                for (int h = 0; h < NKH; h++) {
                    float *kh = bk + h * hd;
                    float s = 0;
                    for (int i = 0; i < hd; i++) s += kh[i] * kh[i];
                    float irk = 1.0f / sqrtf(s / hd + eps);
                    for (int i = 0; i < hd; i++) kh[i] *= irk * k_norm_w[l][h * hd + i];
                }
            }

            /* ── 5. RoPE ── */
            rope_apply(bq, NH, hd, kv_lens[kvi], cos_rope, sin_rope, rd);
            rope_apply(bk, NKH, hd, kv_lens[kvi], cos_rope, sin_rope, rd);

            /* ── 6. KV cache ── */
            int pos = kv_lens[kvi];
            if (swa) {
                int kpos = pos % SW;
                memcpy(kv_k + ((size_t)kvi * max_ctx + kpos) * nk, bk, nk * sizeof(float));
                memcpy(kv_v + ((size_t)kvi * max_ctx + kpos) * nv, bv, nv * sizeof(float));
                pos = pos < SW ? pos + 1 : SW; /* sliding window length */
            } else {
                if (pos < max_ctx) {
                    memcpy(kv_k + ((size_t)kvi * max_ctx + pos) * nk, bk, nk * sizeof(float));
                    memcpy(kv_v + ((size_t)kvi * max_ctx + pos) * nv, bv, nv * sizeof(float));
                }
                pos += 1;
            }

            /* ── 7. GQA attention ── */
            int sl = swa ? (kv_lens[kvi] < SW ? kv_lens[kvi] + 1 : SW) : pos;
            float *kct = kv_k + (size_t)kvi * max_ctx * nk;
            float *vct = kv_v + (size_t)kvi * max_ctx * nv;
            if (!swa) {
                gqa(batt, bq, kct, vct, sl, NH, NKH, hd);
            } else {
                /* Sliding window attention via Python fallback for now */
                memset(batt, 0, nq * sizeof(float));
            }
            kv_lens[kvi] = pos;

            /* ── 8. O projection ── */
            batch_matmul(o_qt[l], wo[l], batt, bop, no, nq, 1);

            /* ── 9. Post-attention norm + residual + per-layer gate ── */
            if (post_attn_norm_w[l]) {
                float ss2 = 0;
                for (int i = 0; i < N; i += 8) {
                    __m256 xv = _mm256_loadu_ps(bop + i); ss2 += hsum_ps(_mm256_mul_ps(xv, xv));
                }
                for (int i = N - (N % 8); i < N; i++) ss2 += bop[i] * bop[i];
                float ir2 = 1.0f / sqrtf(ss2 / N + eps);
                for (int j = 0; j < N; j++)
                    bx[j] = br[j] + bop[j] * ir2 * post_attn_norm_w[l][j] + bpg[j];
            } else {
                for (int j = 0; j < N; j++) bx[j] = br[j] + bop[j] + bpg[j];
            }

            /* ── 10. Pre-FFN RMS norm ── */
            ss = 0;
            for (int i = 0; i < N; i += 8) {
                __m256 xv = _mm256_loadu_ps(bx + i);
                xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
                _mm256_storeu_ps(bx + i, xv); _mm256_storeu_ps(br + i, xv);
                ss += hsum_ps(_mm256_mul_ps(xv, xv));
            }
            for (int i = N - (N % 8); i < N; i++) {
                float vv = bx[i]; if (vv != vv) vv = 0; bx[i] = vv; br[i] = vv; ss += vv * vv;
            }
            ir = 1.0f / sqrtf(ss / N + eps);
            for (int j = 0; j < N; j++) bxn[j] = bx[j] * ir * ffn_norm_w[l][j];

            /* ── 11. Gate + Up projections (fused via batch_matmul) ── */
            batch_matmul(g_qt[l], wg[l], bxn, bg, ng, N, 1);
            batch_matmul(u_qt[l], wu[l], bxn, bu, nu, N, 1);

            /* ── 12. GeGLU: GELU(gate) * up ── */
            for (int i = 0; i < FF_half; i++) {
                float gv = bg[i];
                float g = 0.5f * gv * (1.0f + tanhf(0.7978845608028654f * (gv + 0.044715f * gv * gv * gv)));
                bs[i] = g * bu[i];
            }

            /* ── 13. Down projection ── */
            batch_matmul(d_qt[l], wd[l], bs, bf, nd, FF_half, 1);

            /* ── 14. Post-FFW norm + residual ── */
            if (post_ffw_norm_w[l]) {
                float ss3 = 0;
                for (int i = 0; i < N; i += 8) ss3 += hsum_ps(_mm256_mul_ps(_mm256_loadu_ps(bf + i), _mm256_loadu_ps(bf + i)));
                for (int i = N - (N % 8); i < N; i++) ss3 += bf[i] * bf[i];
                float ir3 = 1.0f / sqrtf(ss3 / N + eps);
                for (int j = 0; j < N; j++) bx[j] = br[j] + bf[j] * ir3 * post_ffw_norm_w[l][j];
            } else {
                for (int j = 0; j < N; j++) bx[j] = br[j] + bf[j];
            }

            /* ── 15. Post-norm (final layer only) ── */
            if (post_norm_w[l]) {
                float ss4 = 0;
                for (int i = 0; i < N; i += 8) ss4 += hsum_ps(_mm256_mul_ps(_mm256_loadu_ps(bx + i), _mm256_loadu_ps(bx + i)));
                for (int i = N - (N % 8); i < N; i++) ss4 += bx[i] * bx[i];
                float ir4 = 1.0f / sqrtf(ss4 / N + eps);
                for (int j = 0; j < N; j++) bxn[j] = bx[j] * ir4 * post_norm_w[l][j];
                memcpy(bx, bxn, N * sizeof(float));
            }

            /* ── 16. Layer output scale ── */
            if (layer_scale[l]) {
                float ls = layer_scale[l][0];
                for (int j = 0; j < N; j++) bx[j] *= ls;
            }
            fprintf(stderr, "  layer %d done\n", l);
        }

        /* ═══════════ Final RMS norm + Output projection + Softcapping ═══════════ */
        fprintf(stderr, "  final rms begin\n");
        float ss = 0;
        for (int i = 0; i < N; i += 8) {
            __m256 xv = _mm256_loadu_ps(bx + i);
            xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
            _mm256_storeu_ps(bx + i, xv);
            ss += hsum_ps(_mm256_mul_ps(xv, xv));
        }
        for (int i = N - (N % 8); i < N; i++) { float vv = bx[i]; ss += vv * vv; }
        float ir = 1.0f / sqrtf(ss / N + eps);
        for (int j = 0; j < N; j++) bxn[j] = bx[j] * ir * out_norm_w[j];
        fprintf(stderr, "  final rms done, output matmul V=%d N=%d qt=%d\n", V, N, qt_out);

        /* Output projection */
        batch_matmul(qt_out, w_out, bxn, logits + (size_t)b * V, V, N, 1);
        fprintf(stderr, "  output matmul done\n");

        /* Logit softcapping */
        fprintf(stderr, "  softcap V=%d\n", V);
        float cap = logit_cap;

        for (int i = 0; i < V; i++) {
            float v = logits[i];
            if (v > cap) v = cap; else if (v < -cap) v = -cap;
            logits[i] = tanhf(v / cap) * cap;
        }
    }
}
