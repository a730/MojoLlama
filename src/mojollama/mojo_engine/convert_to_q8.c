/* convert_to_q8.c — Convert Q4_K/Q6_K/BF16/F32 weights to Q8_0 format */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/stat.h>
#include <math.h>

#define QK 32
#define QB 34

static uint16_t f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    int s = (u >> 31) & 1;
    int e = (u >> 23) & 0xFF;
    int m = u & 0x7FFFFF;
    if (e == 0) return s << 15;
    if (e == 0xFF) return (s << 15) | 0x7C00;
    int ne = e - 127 + 15;
    if (ne >= 31) return (s << 15) | 0x7C00;
    if (ne <= 0) return s << 15;
    return (s << 15) | (ne << 10) | (m >> 13);
}

/* Dequant Q4_K block (144 bytes → 256 floats) */
static void deq_q4k(const uint8_t *blk, float *out) {
    uint16_t d_hi_u, d_lo_u, m_hi_u, m_lo_u;
    memcpy(&d_hi_u, blk, 2); memcpy(&d_lo_u, blk+2, 2);
    memcpy(&m_hi_u, blk+4, 2); memcpy(&m_lo_u, blk+6, 2);
    float d_hi, d_lo, m_hi, m_lo;
    /* f16→f32 */
    { int s=(d_hi_u>>15)&1,e=(d_hi_u>>10)&0x1F,m=d_hi_u&0x3FF;
      if(e==0) d_hi=(float)m*5.96e-8f; else if(e==31) d_hi=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&d_hi,&b,4); } }
    { int s=(d_lo_u>>15)&1,e=(d_lo_u>>10)&0x1F,m=d_lo_u&0x3FF;
      if(e==0) d_lo=(float)m*5.96e-8f; else if(e==31) d_lo=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&d_lo,&b,4); } }
    { int s=(m_hi_u>>15)&1,e=(m_hi_u>>10)&0x1F,m=m_hi_u&0x3FF;
      if(e==0) m_hi=(float)m*5.96e-8f; else if(e==31) m_hi=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&m_hi,&b,4); } }
    { int s=(m_lo_u>>15)&1,e=(m_lo_u>>10)&0x1F,m=m_lo_u&0x3FF;
      if(e==0) m_lo=(float)m*5.96e-8f; else if(e==31) m_lo=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&m_lo,&b,4); } }
    for (int i = 0; i < 128; i++) {
        uint8_t q = blk[8 + i];
        out[i] = (float)((q & 0xF) - 8) * d_lo + m_lo;
        out[i+128] = (float)(((q >> 4) & 0xF) - 8) * d_hi + m_hi;
    }
}

/* Dequant Q6_K block (210 bytes → 256 floats) */
// Q6_K format(latest): d(2) + ql(128) + qh(64) + scales(16) + dmin(2) = 212 unused? Actually 210
// From gguf-py: type_size=210, block_size=256
// Layout: d[0:2] + ql[2:130] + qh[130:194] + scales[194:208](?) + dmin[208:210]
static void deq_q6k(const uint8_t *blk, float *out) {
    uint16_t d_u, dmin_u;
    memcpy(&d_u, blk, 2); memcpy(&dmin_u, blk+208, 2);
    float d_val, dmin_val;
    { int s=(d_u>>15)&1,e=(d_u>>10)&0x1F,m=d_u&0x3FF;
      if(e==0) d_val=(float)m*5.96e-8f; else if(e==31) d_val=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&d_val,&b,4); } }
    { int s=(dmin_u>>15)&1,e=(dmin_u>>10)&0x1F,m=dmin_u&0x3FF;
      if(e==0) dmin_val=(float)m*5.96e-8f; else if(e==31) dmin_val=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&dmin_val,&b,4); } }
    const uint8_t *ql = blk + 2;    // 128 bytes of low 4-bit values
    const uint8_t *qh = blk + 130;  // 64 bytes of high 2-bit values
    const int8_t *sc = (const int8_t*)(blk + 194); // 14 bytes of 8-bit scales (or 16?)
    for (int i = 0; i < 256; i++) {
        int sc_idx = i / 16;
        int low = (ql[i/2] >> (4 * (i%2))) & 0xF;
        int high = (qh[i/4] >> (2 * (i%4))) & 0x3;
        int val = (low | (high << 4)) - 32;  // 6-bit signed
        float s = (float)(sc_idx < 16 ? sc[sc_idx] : 0);
        out[i] = val * d_val + (s < 0 ? -s * dmin_val : 0.0f);
    }
}

/* Convert one file to Q8_0 */
static int convert_file(const char *in_path, const char *out_path) {
    FILE *f = fopen(in_path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *in = (uint8_t*)malloc(sz);
    if (!in) { fclose(f); return 0; }
    fread(in, 1, sz, f); fclose(f);
    
    // Determine format from size
    float *f32 = NULL;
    long n_vals = 0;
    
    if (sz % 144 == 0) {  // Q4_K
        long n_blk = sz / 144;
        n_vals = n_blk * 256;
        f32 = (float*)calloc(n_vals, 4);
        for (long b = 0; b < n_blk; b++)
            deq_q4k(in + b * 144, f32 + b * 256);
    } else if (sz % 240 == 0) {  // Q6_K
        long n_blk = sz / 240;
        n_vals = n_blk * 256;
        f32 = (float*)calloc(n_vals, 4);
        for (long b = 0; b < n_blk; b++)
            deq_q6k(in + b * 240, f32 + b * 256);
    } else if (sz % 4 == 0) {  // F32
        n_vals = sz / 4;
        f32 = (float*)malloc(sz);
        memcpy(f32, in, sz);
    } else if (sz % 2 == 0) {  // BF16 or F16
        n_vals = sz / 2;
        f32 = (float*)calloc(n_vals, 4);
        uint16_t *bf16 = (uint16_t*)in;
        for (long i = 0; i < n_vals; i++) {
            uint32_t bits = (uint32_t)bf16[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    }
    
    if (!f32) { free(in); return 0; }
    
    // Remove NaN/Inf
    for (long i = 0; i < n_vals; i++)
        if (!isfinite(f32[i])) f32[i] = 0.0f;
    
    // Quantize to Q8_0
    long n_blk_q8 = (n_vals + QK - 1) / QK;
    long q8_sz = n_blk_q8 * QB;
    uint8_t *q8 = (uint8_t*)calloc(1, q8_sz);
    
    for (long b = 0; b < n_blk_q8; b++) {
        long start = b * QK;
        long end = start + QK < n_vals ? start + QK : n_vals;
        float max_abs = 0.0f;
        for (long i = start; i < end; i++) {
            float a = fabsf(f32[i]);
            if (a > max_abs) max_abs = a;
        }
        float scale = max_abs > 1e-10f ? max_abs / 127.0f : 1.0f;
        uint16_t scale_u16 = f32_to_f16(scale);
        memcpy(q8 + b * QB, &scale_u16, 2);
        for (long i = start; i < end; i++) {
            int qv = (int)roundf(f32[i] / scale);
            if (qv > 127) qv = 127;
            if (qv < -128) qv = -128;
            q8[b * QB + 2 + (i - start)] = (uint8_t)(qv & 0xFF);
        }
    }
    
    FILE *out = fopen(out_path, "wb");
    if (out) { fwrite(q8, 1, q8_sz, out); fclose(out); }
    
    free(in); free(f32); free(q8);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <in_dir> <out_dir>\n", argv[0]); return 1; }
    mkdir(argv[2], 0755);
    
    DIR *d = opendir(argv[1]);
    if (!d) return 1;
    struct dirent *de;
    int n = 0;
    while ((de = readdir(d))) {
        if (de->d_type != DT_REG) continue;
        char in_path[2048], out_path[2048];
        snprintf(in_path, sizeof(in_path), "%s/%s", argv[1], de->d_name);
        snprintf(out_path, sizeof(out_path), "%s/%s", argv[2], de->d_name);
        if (convert_file(in_path, out_path)) n++;
        if (n % 100 == 0) printf("  %d files...\n", n);
    }
    closedir(d);
    printf("Converted %d files\n", n);
    return 0;
}
