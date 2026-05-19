/* cengine3.c — Pure C forward pass using proven quant_matmul_omp.
 * Calls quant_kernels_omp.so for ALL matmuls — bit-identical to v7.7.
 * Inline rms_norm and gqa_attention (also proven).
 *
 * Compile:
 *   gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
 *       -o cengine3.so cengine3.c \
 *       -L. -l:quant_kernels_omp.so \
 *       -Wl,-rpath,/onedev-workspace/work/src/mojollama/kernels -lm
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

/* External: proven OMP kernels */
extern void quant_matmul_omp(const unsigned char*, const float*, float*, int, int, int);
extern void batch_qkv_omp(const unsigned char*, const unsigned char*, const unsigned char*,
                           const float*, float*, float*, float*,
                           int, int, int, int, int, int, int);
extern void batch_gate_up_omp(const unsigned char*, const unsigned char*,
                               const float*, float*, float*,
                               int, int, int, int, int);

/* ═══ Inline RMS norm (AVX2, matches quant_kernels_omp.c) ══════════ */
static inline void rms_norm_f32(float *out, const float *x, const float *w, int n, float eps) {
    __m256 ss = _mm256_setzero_ps(); int i;
    for (i = 0; i <= n - 8; i += 8) { __m256 v = _mm256_loadu_ps(x + i); ss = _mm256_fmadd_ps(v, v, ss); }
    __m256 h = _mm256_hadd_ps(ss, _mm256_permute2f128_ps(ss, ss, 1));
    h = _mm256_hadd_ps(h, h); h = _mm256_hadd_ps(h, h);
    float sse = _mm256_cvtss_f32(h); for (; i < n; i++) sse += x[i] * x[i];
    float inv = 1.0f / sqrtf(sse / n + eps); __m256 iv = _mm256_set1_ps(inv);
    for (i = 0; i <= n - 8; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i), wv = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(xv, iv), wv));
    }
    for (; i < n; i++) out[i] = x[i] * inv * w[i];
}

/* ═══ Inline GQA attention (AVX2, matches gqa_attention.c) ═════════ */
static inline float hsum_ps(__m256 v) {
    __m128 l = _mm256_castps256_ps128(v), h = _mm256_extractf128_ps(v, 1);
    l = _mm_add_ps(l, h); l = _mm_hadd_ps(l, l); l = _mm_hadd_ps(l, l);
    return _mm_cvtss_f32(l);
}
static void gqa_attn(float *out, const float *q, const float *kc, const float *vc,
                      int sl, int nh, int nkh, int hd) {
    int gr = nh / nkh;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < nh; h++) {
        int kv = h / gr; const float *qh = q + h * hd, *kb = kc + kv * hd, *vb = vc + kv * hd;
        int ks = nkh * hd; float *oh = out + h * hd, sc[4096], mx = -1e30f, inv = 1.0f / sqrtf(hd);
        for (int s = 0; s < sl; s++) {
            const float *ks_ = kb + s * ks; __m256 su = _mm256_setzero_ps(); int d;
            for (d = 0; d <= hd - 8; d += 8) su = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d), _mm256_loadu_ps(ks_ + d), su);
            float dot = hsum_ps(su); for (; d < hd; d++) dot += qh[d] * ks_[d];
            sc[s] = dot * inv; if (sc[s] > mx) mx = sc[s];
        }
        float se = 0; for (int s = 0; s < sl; s++) { sc[s] = expf(sc[s] - mx); se += sc[s]; }
        float is = 1.0f / se; memset(oh, 0, hd * sizeof(float));
        for (int s = 0; s < sl; s++) {
            const float *vs_ = vb + s * ks; float w = sc[s] * is; __m256 wv = _mm256_set1_ps(w); int d;
            for (d = 0; d <= hd - 8; d += 8) {
                __m256 vv = _mm256_loadu_ps(vs_ + d);
                __m256 ac = _mm256_loadu_ps(oh + d);
                _mm256_storeu_ps(oh + d, _mm256_fmadd_ps(wv, vv, ac));
            }
            for (; d < hd; d++) oh[d] += w * vs_[d];
        }
    }
}

/* ═══ Engine config struct ═════════════════════════════════════════ */
typedef struct {
    int L, N, NH, NKH, HD, FF, V; float eps;
    /* Per-layer weight pointers [L] */
    const unsigned char **wQ, **wK, **wV, **wO, **wG, **wU, **wD;
    const float **wAN, **wFN;
    const int *nQ, *nK, *nV, *nO, *nG, *nU, *nD;
    int nc;  /* n_cols for QKV/GateUp/O (always n_embd) */
    const float *emb, *onw;
    const unsigned char *wOut; int outNR, outNC;
    float *kvK, *kvV; int *kvL; int MP;
    float *x, *xn, *res, *q, *k, *v, *att, *gate, *up, *silu, *oproj, *ffn;
} EngineConfig3;

/* ═══ engine_forward — pure C, no Python, uses proven quant_matmul_omp ═══ */
void engine_forward(int tok, const EngineConfig3 *c, float *logits) {
    int N=c->N, NH=c->NH, NKH=c->NKH, HD=c->HD, FF=c->FF, L=c->L, nc=c->nc;
    float *x=c->x, *xn=c->xn, *res=c->res, *q=c->q, *k=c->k, *v=c->v;
    float *att=c->att, *gate=c->gate, *up=c->up, *silu=c->silu, *o=c->oproj, *ffn=c->ffn;

    memcpy(x, c->emb + (size_t)tok * N, N * sizeof(float));

    for (int l = 0; l < L; l++) {
        memcpy(res, x, N * sizeof(float));
        rms_norm_f32(xn, x, c->wAN[l], N, c->eps);

        /* Q, K, V — batched OMP (single fork/join for all 3) */
        batch_qkv_omp(c->wQ[l], c->wK[l], c->wV[l], xn, q, k, v,
                      c->nQ[l], c->nK[l], c->nV[l], nc, 2, 2, 2);

        /* RoPE — on-the-fly */
        int pos = c->kvL[l];
        for (int hh = 0; hh < NH; hh++) { float *qh = q + hh * HD;
            for (int j = 0; j < HD/2; j++) {
                double ang = (double)pos / pow(10000.0, 2.0*j/HD);
                double cs = cos(ang), sn = sin(ang);
                float a = qh[j], b = qh[j+HD/2];
                qh[j] = (float)(a*cs - b*sn); qh[j+HD/2] = (float)(b*cs + a*sn);
            }
        }
        for (int hh = 0; hh < NKH; hh++) { float *kh = k + hh * HD;
            for (int j = 0; j < HD/2; j++) {
                double ang = (double)pos / pow(10000.0, 2.0*j/HD);
                double cs = cos(ang), sn = sin(ang);
                float a = kh[j], b = kh[j+HD/2];
                kh[j] = (float)(a*cs - b*sn); kh[j+HD/2] = (float)(b*cs + a*sn);
            }
        }

        /* KV store */
        memcpy(c->kvK + ((size_t)l*c->MP + pos) * NKH * HD, k, NKH * HD * sizeof(float));
        memcpy(c->kvV + ((size_t)l*c->MP + pos) * NKH * HD, v, NKH * HD * sizeof(float));

        /* GQA attention */
        gqa_attn(att, q, c->kvK + (size_t)l*c->MP*NKH*HD, c->kvV + (size_t)l*c->MP*NKH*HD,
                 pos+1, NH, NKH, HD);
        c->kvL[l]++;

        /* O projection — proven quant_matmul_omp, n_cols = nc */
        quant_matmul_omp(c->wO[l], att, o, c->nO[l], nc, 2);
        for (int i = 0; i < N; i++) x[i] = res[i] + o[i];

        /* FFN */
        memcpy(res, x, N * sizeof(float));
        rms_norm_f32(xn, x, c->wFN[l], N, c->eps);

        /* Gate + Up — batched OMP */
        batch_gate_up_omp(c->wG[l], c->wU[l], xn, gate, up,
                          c->nG[l], c->nU[l], nc, 2, 2);

        /* SiLU(gate) * up */
        for (int i = 0; i < FF; i++) { float g = gate[i]; silu[i] = (g / (1.0f + expf(-g))) * up[i]; }

        /* Down — n_cols = FF (n_ff), NOT nc (n_embd)! */
        quant_matmul_omp(c->wD[l], silu, ffn, c->nD[l], FF, 2);
        for (int i = 0; i < N; i++) x[i] = res[i] + ffn[i];
    }

    /* Final norm + output */
    rms_norm_f32(xn, x, c->onw, N, c->eps);
    quant_matmul_omp(c->wOut, xn, logits, c->outNR, c->outNC, 8);
}
