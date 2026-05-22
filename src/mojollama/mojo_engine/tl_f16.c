/* tl_f16.c — Correct TinyLlama f16 .bin inference reference.
 * Loads raw f16 weights (not Q4_0), runs forward pass, measures real tok/s.
 * Compile: gcc -O2 -mavx2 -mfma -mf16c -fopenmp tl_f16.c -o tl_f16 -lm
 * Usage: ./tl_f16 /tmp/weights_tl/
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

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

static float f16_to_f32(uint16_t h) {
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
        int c;
        for (c = 0; c + 8 <= N; c += 8) {
            __m128i w16 = _mm_loadu_si128((const __m128i*)(wt + c));
            a = _mm256_fmadd_ps(_mm256_cvtph_ps(w16), _mm256_loadu_ps(x + c), a);
        }
        float s = 0;
        for (; c < N; c++) s += f16_to_f32(wt[c]) * x[c];
        __m128 hi = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        hi = _mm_hadd_ps(hi, hi); hi = _mm_hadd_ps(hi, hi);
        o[r] = _mm_cvtss_f32(hi) + s;
    }
}

int main(int argc, char **argv) {
    const char *d = argc > 1 ? argv[1] : "/tmp/weights_tl";
    double t0 = omp_get_wtime();
    
    // Load weights
    void *w_emb = load(d, "token_embd_weight.bin");
    void *w_on = load(d, "output_norm_weight.bin");
    void *w_lm = load(d, "output_weight.bin");
    if (!w_emb || !w_on || !w_lm) { fprintf(stderr, "FAIL: base weights\n"); return 1; }
    
    void *wl[NL][9];
    char sfx[9][32] = {"_attn_norm_weight.bin","_ffn_norm_weight.bin",
        "_attn_q_weight.bin","_attn_k_weight.bin","_attn_v_weight.bin",
        "_attn_output_weight.bin","_ffn_gate_weight.bin","_ffn_up_weight.bin","_ffn_down_weight.bin"};
    char nm[256];
    for (int l = 0; l < NL; l++)
        for (int f = 0; f < 9; f++)
            { snprintf(nm, sizeof(nm), "blk_%d%s", l, sfx[f]); wl[l][f] = load(d, nm);
              if (!wl[l][f]) { fprintf(stderr, "FAIL: %s\n", nm); return 1; } }
    
    double t1 = omp_get_wtime();
    printf("Load: %.0f ms\n", (t1-t0)*1000);
    
    // Buffers
    float hp[NE], bp[NE], qp[NH*HD], kp[NK*HD], vp[NK*HD], gp[NF], up[NF], dp[NE], lp[NV];
    
    // Embed BOS token (1)
    const uint16_t *emb = (const uint16_t*)w_emb;
    for (int i = 0; i < NE; i++) hp[i] = f16_to_f32(emb[1 * NE + i]);
    double t2 = omp_get_wtime();
    printf("Embed: %.0f ms\n", (t2-t1)*1000);
    
    // Forward pass
    for (int l = 0; l < NL; l++) {
        // RMS Norm pre-attention
        float ss = 0; for (int i = 0; i < NE; i++) ss += hp[i] * hp[i];
        float inv = 1.0f / sqrtf(ss / NE + 1e-6f);
        const float *anp = (const float*)wl[l][0];
        for (int i = 0; i < NE; i++) bp[i] = hp[i] * anp[i] * inv;
        
        // QKV
        mm_f16(wl[l][2], bp, qp, NH*HD, NE);
        mm_f16(wl[l][3], bp, kp, NK*HD, NE);
        mm_f16(wl[l][4], bp, vp, NK*HD, NE);
        
        // RoPE
        for (int h = 0; h < NH; h++) for (int d2 = 0; d2 < HD; d2 += 2) {
            float f = 1 / powf(10000, (float)d2/HD);
            float c = cosf(f), s = sinf(f);
            float x0 = qp[h*HD+d2], x1 = qp[h*HD+d2+1];
            qp[h*HD+d2] = x0*c - x1*s; qp[h*HD+d2+1] = x0*s + x1*c;
        }
        for (int h = 0; h < NK; h++) for (int d2 = 0; d2 < HD; d2 += 2) {
            float f = 1 / powf(10000, (float)d2/HD);
            float c = cosf(f), s = sinf(f);
            float x0 = kp[h*HD+d2], x1 = kp[h*HD+d2+1];
            kp[h*HD+d2] = x0*c - x1*s; kp[h*HD+d2+1] = x0*s + x1*c;
        }
        
        // GQA (single token, no KV cache)
        int khr = NH / NK;
        for (int hq = 0; hq < NH; hq++) {
            int hkv = hq / khr;
            float sc = 0;
            for (int d = 0; d < HD; d++) sc += qp[hq*HD+d] * kp[hkv*HD+d];
            sc = expf(sc / sqrtf(HD)) / expf(sc / sqrtf(HD)); // softmax(1-element) = 1.0
            for (int d = 0; d < HD; d++) qp[hq*HD+d] = vp[hkv*HD+d] * sc;
        }
        
        // O proj
        mm_f16(wl[l][5], qp, bp, NE, NH*HD);
        for (int i = 0; i < NE; i++) hp[i] += bp[i];
        
        // RMS Norm pre-FFN
        ss = 0; for (int i = 0; i < NE; i++) ss += hp[i] * hp[i];
        inv = 1.0f / sqrtf(ss / NE + 1e-6f);
        const float *fnp = (const float*)wl[l][1];
        for (int i = 0; i < NE; i++) bp[i] = hp[i] * fnp[i] * inv;
        
        // FFN gate + up
        mm_f16(wl[l][6], bp, gp, NF, NE);
        mm_f16(wl[l][7], bp, up, NF, NE);
        for (int i = 0; i < NF; i++) {
            float gv = gp[i];
            if (gv < -80) gv = -80; if (gv > 80) gv = 80;
            gp[i] = (gv / (1 + expf(-gv))) * up[i];
        }
        
        // FFN down
        mm_f16(wl[l][8], gp, dp, NE, NF);
        for (int i = 0; i < NE; i++) hp[i] += dp[i];
    }
    
    double t3 = omp_get_wtime();
    
    // Final norm + LM head
    float ss = 0; for (int i = 0; i < NE; i++) ss += hp[i] * hp[i];
    float inv = 1.0f / sqrtf(ss / NE + 1e-6f);
    const float *onp = (const float*)w_on;
    for (int i = 0; i < NE; i++) bp[i] = hp[i] * onp[i] * inv;
    mm_f16(w_lm, bp, lp, NV, NE);
    
    double t4 = omp_get_wtime();
    
    int best = 0; float bv = lp[0];
    for (int i = 1; i < NV; i++) if (lp[i] > bv) { bv = lp[i]; best = i; }
    
    printf("First 5 logits: %.2f %.2f %.2f %.2f %.2f\n", lp[0], lp[1], lp[2], lp[3], lp[4]);
    printf("Best token: %d (val=%.1f)\n", best, bv);
    printf("Layers: %.0f ms (%.2f ms/layer)\n", (t3-t2)*1000, (t3-t2)*1000/NL);
    printf("LM head: %.0f ms\n", (t4-t3)*1000);
    printf("Total forward: %.0f ms  tok/s: %.1f\n", (t4-t2)*1000, 1.0/(t4-t2));
    return 0;
}
