/* cengine_batch_diag.c — diagnostic version that dumps expert weight data */
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

/* DIAGNOSTIC Q4_K matmul with detailed NaN tracking */
void q4_k_batch_matmul_diag(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B, const char *tag) {
    int bpr = nc / QK_K;
    int nan_count = 0, first_nan_row = -1, first_nan_blk = -1;
    uint16_t bad_d = 0, bad_dm = 0;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float *acc = (float*)__builtin_alloca(B * sizeof(float));
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const block_q4_K *bp = (const block_q4_K*)(W + ((size_t)r * bpr + blk) * sizeof(block_q4_K));
            float d = gf16(bp->d), dm = gf16(bp->dm);
            
            /* Check for NaN in d/dm */
            if (isnan(d) || isnan(dm)) {
                #pragma omp critical
                {
                    if (nan_count == 0) {
                        first_nan_row = r; first_nan_blk = blk;
                        bad_d = bp->d; bad_dm = bp->dm;
                    }
                    nan_count++;
                }
            }
            
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
    if (nan_count > 0) {
        #pragma omp critical
        {
            fprintf(stderr, "  DIAG [%s]: %d NaN d/dm in %d rows (first at row=%d blk=%d d=0x%04x(%f) dm=0x%04x(%f))\n",
                    tag, nan_count, n_rows, first_nan_row, first_nan_blk,
                    bad_d, gf16(bad_d), bad_dm, gf16(bad_dm));
            /* Dump first few bytes of the problematic block */
            size_t bad_offset = ((size_t)first_nan_row * bpr + first_nan_blk) * sizeof(block_q4_K);
            const uint8_t *bad_ptr = W + bad_offset;
            fprintf(stderr, "    bytes at offset %zu: ", bad_offset);
            for (int i = 0; i < 16 && i < (int)sizeof(block_q4_K); i++) {
                fprintf(stderr, "%02x ", bad_ptr[i]);
            }
            fprintf(stderr, "\n");
            fflush(stderr);
        }
    }
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

/* Batched Q4_K matmul (standard, no diag) */
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

/* AVX2 Batched Q6_K matmul */
void q6_k_batch_matmul(const uint8_t *W, const float *x, float *out,
                       int n_rows, int nc, int B) {
    int bpr = nc / QK_K;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float *acc = (float*)__builtin_alloca(B * sizeof(float));
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const block_q6_K *bp = (const block_q6_K*)(W + ((size_t)r * bpr + blk) * sizeof(block_q6_K));
            float d = gf16(bp->d);
            float deq[QK_K];
            for (int j = 0; j < QK_K; j++) {
                uint8_t ql = bp->ql[j/2];
                uint8_t qh = bp->qh[j/4];
                int s = (j & 1) ? (ql >> 4) : (ql & 0xF);
                int sh = (qh >> ((j & 3) * 2)) & 3;
                s |= (sh << 4); s -= 32;
                deq[j] = d * bp->scales[j/16] * s;
            }
            int o = blk * QK_K;
            for (int b = 0; b < B; b++) {
                const float *xb = x + (size_t)b * nc + o;
                __m256 sum = _mm256_setzero_ps();
                for (int j = 0; j < QK_K; j += 8)
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(deq + j),
                                         _mm256_loadu_ps(xb + j), sum);
                acc[b] += hsum_ps(sum);
            }
        }
        for (int b = 0; b < B; b++) out[(size_t)b * n_rows + r] = acc[b];
    }
}

typedef struct {
    float *k, *v; int n_blocks, seq_len[64], block_map[64][1024];
} KVBlock;
void kv_init(KVBlock *kv, int L, int NKH, int HD) {
    size_t sz = (size_t)L * MAX_BLOCKS * BLOCK_SIZE * NKH * HD * sizeof(float);
    kv->k = (float*)calloc(1, sz); kv->v = (float*)calloc(1, sz); kv->n_blocks = 0;
    memset(kv->block_map, -1, sizeof(kv->block_map)); memset(kv->seq_len, 0, sizeof(kv->seq_len));
}
int kv_alloc(KVBlock *kv) { return kv->n_blocks < MAX_BLOCKS ? kv->n_blocks++ : -1; }

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
    
    /* Dump stride info */
    fprintf(stderr, "  moe: N=%d M=%d gate_stride=%zu up_stride=%zu down_stride=%zu gate_quant=%d\n",
            N, M, gate_stride, up_stride, down_stride, c->gate_exp_quant);
    
    for (int b = 0; b < B; b++) {
        float *xb = x + b*N;
        
        // Router
        for (int e = 0; e < NE; e++) {
            float dot = 0;
            for (int i = 0; i < N; i++) dot += xb[i] * w_router[e * N + i];
            router_logits[e] = dot;
        }
        
        float mx = router_logits[0];
        for (int e = 1; e < NE; e++) if (router_logits[e] > mx) mx = router_logits[e];
        float sum = 0;
        for (int e = 0; e < NE; e++) { router_logits[e] = expf(router_logits[e] - mx); sum += router_logits[e]; }
        float inv_sum = 1.0f / (sum + 1e-10f);
        for (int e = 0; e < NE; e++) router_logits[e] *= inv_sum;
        
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
        
        memset(ffn_buf, 0, N * sizeof(float));
        for (int k = 0; k < NK; k++) {
            int e = top_idx[k];
            if (e < 0) continue;
            float weight = top_val[k];
            
            /* Compute expected pointer and dump first block for each expert */
            const uint8_t *gate_w = c->w_gate_exps[l] + (size_t)e * gate_stride;
            
            /* Dump first 16 bytes of weight data for diagnostic */
            uint16_t first_d = *(const uint16_t*)gate_w;
            uint16_t first_dm = *(const uint16_t*)(gate_w + 2);
            fprintf(stderr, "  E%03d gate_w=%p d=0x%04x(%f) dm=0x%04x(%f) ",
                    e, (void*)gate_w, first_d, gf16(first_d), first_dm, gf16(first_dm));
            fprintf(stderr, "bytes: ");
            for (int i = 0; i < 16; i++) fprintf(stderr, "%02x ", gate_w[i]);
            fprintf(stderr, "\n");
            
            if (isnan(gf16(first_d)) || isnan(gf16(first_dm))) {
                fprintf(stderr, "  *** EXPERT %d HAS NaN IN SUPER-BLOCK HEADER! ***\n", e);
            }
            
            if (c->gate_exp_quant == 12)
                q4_k_batch_matmul(gate_w, xb, gate_buf, M, N, 1);
            else if (c->gate_exp_quant == 2)
                q4_0_batch_matmul(gate_w, xb, gate_buf, M, N, 1);
            
            for (int i = 0; i < M; i++) if (isnan(gate_buf[i])) {
                fprintf(stderr, "    GATE NaN at expert %d pos %d\n", e, i); fflush(stderr); break;
            }
            
            const uint8_t *up_w = c->w_up_exps[l] + (size_t)e * up_stride;
            if (c->up_exp_quant == 12)
                q4_k_batch_matmul(up_w, xb, up_buf, M, N, 1);
            else if (c->up_exp_quant == 2)
                q4_0_batch_matmul(up_w, xb, up_buf, M, N, 1);
            
            for (int i = 0; i < M; i++) {
                float g = gate_buf[i];
                gate_buf[i] = (g / (1.0f + expf(-g))) * up_buf[i];
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
    else if (qt == 14) q6_k_batch_matmul(W, x, out, n_rows, nc, B);
    else if (qt == 8)  q8_0_batch_matmul(W, x, out, n_rows, nc, B);
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
    for (int h = 0; h < nh; h++) {
        float *b = buf + h * hd;
        for (int j = 0; j < hd2; j++) {
            float x = b[j], y = b[j + hd2];
            float c = cos_t[pos * hd2 + j], s = sin_t[pos * hd2 + j];
            b[j] = x * c - y * s;
            b[j + hd2] = x * s + y * c;
        }
    }
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

void batch_forward(const BC *c, const int *tokens, int B, float *ws) {
    int N=c->N, NH=c->NH, NKH=c->NKH, HD=c->HD, FF=c->FF, L=c->L, nc=c->nc;
    int S = N; if (NH*HD > S) S = NH*HD; if (FF > S) S = FF; if (NKH*HD > S) S = NKH*HD;
    
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
            memcpy(res + b*N, x + b*N, N * sizeof(float));
            rms(xn + b*N, x + b*N, c->wAN[l], N, c->eps);
        }
        batch_matmul(c->q_quant[l], c->wQ[l], xn, q, c->nQ[l], nc, B);
        batch_matmul(c->k_quant[l], c->wK[l], xn, k, c->nK[l], nc, B);
        batch_matmul(c->v_quant[l], c->wV[l], xn, v, c->nV[l], nc, B);
        
        for (int b = 0; b < B; b++) {
            float *qb = q + b*c->nQ[l], *kb = k + b*c->nK[l], *vb = v + b*c->nV[l];
            KVBlock *kv = c->kv_array[b];
            int pos = kv->seq_len[0];
            rope_apply(qb, NH, HD, pos, c->cos_table, c->sin_table);
            rope_apply(kb, NKH, HD, pos, c->cos_table, c->sin_table);
            
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
        
        batch_matmul(c->o_quant[l], c->wO[l], att, oproj, c->nO[l], c->NH * c->HD, B);
        for(int b=0;b<B;b++)for(int i=0;i<N;i++)x[b*N+i]=res[b*N+i]+oproj[b*N+i];
        
        for(int b=0;b<B;b++){
            memcpy(res+b*N, x+b*N, N*sizeof(float));
            rms(xn+b*N, x+b*N, c->wFN[l], N, c->eps);
        }
        if (c->n_experts > 0 && c->n_experts_per_tok > 0) {
            fprintf(stderr, "\n--- Layer %d MoE FFN ---\n", l);
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
}

void set_num_threads(int n) { omp_set_num_threads(n); }

/* ADDED: emb_lookup stub for compilation */
void emb_lookup(const uint8_t *emb, int token, float *out, int N, int quant) {
    if (quant == 12) {
        int bpr = N / QK_K, blk_size = sizeof(block_q4_K);
        for (int bi = 0; bi < bpr; bi++) {
            const block_q4_K *bp = (const block_q4_K*)(emb + ((size_t)token * bpr + bi) * blk_size);
            float d = gf16(bp->d), dm = gf16(bp->dm);
            int o = bi * QK_K;
            for (int j = 0; j < QK_K; j++) {
                int nib = (bp->qs[j/2] >> ((j & 1) * 4)) & 0xF;
                out[o + j] = d * nib - dm;
            }
        }
    } else {
        memcpy(out, emb + (size_t)token * N * sizeof(float), N * sizeof(float));
    }
}
