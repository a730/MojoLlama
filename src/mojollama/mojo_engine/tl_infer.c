/* tl_infer.c — Correct TinyLlama inference with f16 .bin weights
 * Compile: gcc -O2 -mavx2 -mfma -mf16c tl_infer.c -o tl_infer -lm -lpthread
 * Usage: ./tl_infer /tmp/weights_tl/
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <x86intrin.h>

#define NE 2048
#define NH 32
#define NK 4
#define HD 64
#define NL 22
#define NF 5632
#define NV 32000

static void* load(const char *dir, const char *name) {
    char p[1024]; snprintf(p, sizeof(p), "%s/%s", dir, name);
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(sz); fread(b, 1, sz, f); fclose(f); return b;
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) { float r = (float)m * 5.960464477539063e-8f; return s ? -r : r; }
    if (e == 31) return 0.0f;
    uint32_t b = (s << 31) | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &b, 4); return r;
}

static void mm_f16(const void *w, const float *x, float *o, int M, int N) {
    #pragma omp parallel for
    for (int r = 0; r < M; r++) {
        __m256 a = _mm256_setzero_ps();
        const uint16_t *wt = (const uint16_t*)w + r * N;
        for (int c = 0; c < N; c += 8) {
            __m128i w16 = _mm_loadu_si128((const __m128i*)(wt + c));
            __m256 wf = _mm256_cvtph_ps(w16);
            a = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + c), a);
        }
        __m128 hi = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        hi = _mm_hadd_ps(hi, hi); hi = _mm_hadd_ps(hi, hi);
        o[r] = _mm_cvtss_f32(hi);
    }
}

int main(int argc, char **argv) {
    const char *d = argc > 1 ? argv[1] : "/tmp/weights_tl";
    struct timespec ts0, ts1, ts2, ts3, ts4; clock_gettime(CLOCK_MONOTONIC, &ts0);
    void *w[201] = {0}; int wi = 0;
    w[wi++] = load(d, "token_embd_weight.bin");
    w[wi++] = load(d, "output_norm_weight.bin");
    w[wi++] = load(d, "output_weight.bin");
    char sfx[9][32] = {"_attn_norm_weight.bin","_ffn_norm_weight.bin",
        "_attn_q_weight.bin","_attn_k_weight.bin","_attn_v_weight.bin",
        "_attn_output_weight.bin","_ffn_gate_weight.bin","_ffn_up_weight.bin","_ffn_down_weight.bin"};
    char nm[256];
    for (int l = 0; l < NL; l++)
        for (int f = 0; f < 9; f++)
            { snprintf(nm, sizeof(nm), "blk_%d%s", l, sfx[f]); w[wi++] = load(d, nm); }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    printf("Load: %.0f ms\n", (double)(ts1.tv_sec-ts0.tv_sec)*1000 + (double)(ts1.tv_nsec-ts0.tv_nsec)/1e6);
    
    float hp[NE], bp[NE], qp[NH*HD], kp[NK*HD], vp[NK*HD], gp[NF], up[NF], dp[NE], lp[NV];
    const uint16_t *emb = (const uint16_t*)w[0];
    int tok = 1;
    for (int i = 0; i < NE; i++) hp[i] = f16_to_f32(emb[tok * NE + i]);
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    printf("Embed: %.0f ms\n", (double)(ts2.tv_sec-ts1.tv_sec)*1000 + (double)(ts2.tv_nsec-ts1.tv_nsec)/1e6);
    
    for (int l = 0; l < NL; l++) {
        int lw = 3 + l * 9;
        float ss = 0;
        for (int i = 0; i < NE; i++) ss += hp[i] * hp[i];
        float inv = 1.0f / sqrtf(ss / NE + 1e-6f);
        const float *anp = (const float*)w[lw];
        for (int i = 0; i < NE; i++) bp[i] = hp[i] * anp[i] * inv;
        mm_f16(w[lw+2], bp, qp, NH*HD, NE);
        mm_f16(w[lw+3], bp, kp, NK*HD, NE);
        mm_f16(w[lw+4], bp, vp, NK*HD, NE);
        for (int h = 0; h < NH; h++) for (int d2 = 0; d2 < HD; d2+=2) {
            float f = tok / powf(10000, (float)d2/HD), c = cosf(f), s = sinf(f);
            float x0 = qp[h*HD+d2], x1 = qp[h*HD+d2+1];
            qp[h*HD+d2] = x0*c - x1*s; qp[h*HD+d2+1] = x0*s + x1*c;
        }
        for (int h = 0; h < NK; h++) for (int d2 = 0; d2 < HD; d2+=2) {
            float f = tok / powf(10000, (float)d2/HD), c = cosf(f), s = sinf(f);
            float x0 = kp[h*HD+d2], x1 = kp[h*HD+d2+1];
            kp[h*HD+d2] = x0*c - x1*s; kp[h*HD+d2+1] = x0*s + x1*c;
        }
        for (int hq = 0; hq < NH; hq++) {
            int hkv = hq / (NH/NK); float sc = 0;
            for (int d = 0; d < HD; d++) sc += qp[hq*HD+d] * kp[hkv*HD+d];
            float wt = expf(sc / sqrtf(HD)) / expf(sc / sqrtf(HD));
            for (int d = 0; d < HD; d++) qp[hq*HD+d] = vp[hkv*HD+d] * wt;
        }
        mm_f16(w[lw+5], qp, bp, NE, NH*HD);
        for (int i = 0; i < NE; i++) hp[i] += bp[i];
        ss = 0; for (int i = 0; i < NE; i++) ss += hp[i] * hp[i];
        inv = 1.0f / sqrtf(ss / NE + 1e-6f);
        const float *fnp = (const float*)w[lw+1];
        for (int i = 0; i < NE; i++) bp[i] = hp[i] * fnp[i] * inv;
        mm_f16(w[lw+6], bp, gp, NF, NE);
        mm_f16(w[lw+7], bp, up, NF, NE);
        for (int i = 0; i < NF; i++) {
            float gv = gp[i]; if (gv < -80) gv = -80; if (gv > 80) gv = 80;
            gp[i] = (gv / (1 + expf(-gv))) * up[i];
        }
        mm_f16(w[lw+8], gp, dp, NE, NF);
        for (int i = 0; i < NE; i++) hp[i] += dp[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &ts3);
    ss = 0; for (int i = 0; i < NE; i++) ss += hp[i] * hp[i];
    inv = 1.0f / sqrtf(ss / NE + 1e-6f);
    const float *onp = (const float*)w[1];
    for (int i = 0; i < NE; i++) bp[i] = hp[i] * onp[i] * inv;
    mm_f16(w[2], bp, lp, NV, NE);
    clock_gettime(CLOCK_MONOTONIC, &ts4);
    int best = 0; float bv = lp[0];
    for (int i = 1; i < NV; i++) if (lp[i] > bv) { bv = lp[i]; best = i; }
    double layers_ms = (double)(ts3.tv_sec-ts2.tv_sec)*1000 + (double)(ts3.tv_nsec-ts2.tv_nsec)/1e6;
double total_ms = (double)(ts4.tv_sec-ts2.tv_sec)*1000 + (double)(ts4.tv_nsec-ts2.tv_nsec)/1e6;
printf("First 5 logits: %.1f %.1f %.1f %.1f %.1f\n", lp[0], lp[1], lp[2], lp[3], lp[4]);
    printf("Layers: %.0f ms (%.1f ms/layer)\n", layers_ms, layers_ms/NL);
    printf("LM head: %.0f ms\n", (double)(ts4.tv_sec-ts3.tv_sec)*1000 + (double)(ts4.tv_nsec-ts3.tv_nsec)/1e6);
    printf("Total: %.0f ms  tok/s: %.1f\n", total_ms, 1.0/(total_ms/1000.0));
    printf("Best token: %d val: %.1f\n", best, bv);
    return 0;
}
