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
/* Correct layout (from gguf-py Q4_K.dequantize_blocks):
   bytes [0:2] = d (f16 scale)
   bytes [2:4] = dmin (f16 min)
   bytes [4:16] = scales (12 bytes, 6-bit packed scale+min for 8 groups)
   bytes [16:144] = qs (128 bytes, nibble-packed quants)
   Each group: 32 elements, formula = (quant-8) * d * scale_i + dmin * min_i
*/
static void deq_q4k(const uint8_t *blk, float *out) {
    /* Read d and dmin from bytes 0-3 */
    uint16_t d_u, dm_u;
    memcpy(&d_u, blk, 2); memcpy(&dm_u, blk+2, 2);
    float d_val, dm_val;
    { int s=(d_u>>15)&1,e=(d_u>>10)&0x1F,m=d_u&0x3FF;
      if(e==0) d_val=(float)m*5.96e-8f; else if(e==31) d_val=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&d_val,&b,4); } }
    { int s=(dm_u>>15)&1,e=(dm_u>>10)&0x1F,m=dm_u&0x3FF;
      if(e==0) dm_val=(float)m*5.96e-8f; else if(e==31) dm_val=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&dm_val,&b,4); } }
    
    /* Read 12 scale bytes */
    uint8_t sc[12];
    memcpy(sc, blk+4, 12);
    
    /* Unpack 6-bit scales and mins (matching gguf-py exactly) */
    float scale_arr[8], min_arr[8];
    for (int i = 0; i < 4; i++) {
        uint8_t d_byte = sc[i];      /* scales low part */
        uint8_t m_byte = sc[i+4];    /* mins low part */
        uint8_t md_byte = sc[i+8];   /* mixed high bits */
        scale_arr[i]   = (float)(d_byte & 0x3F);
        min_arr[i]     = (float)(m_byte & 0x3F);
        scale_arr[i+4] = (float)((md_byte & 0x0F) | ((d_byte >> 2) & 0x30));
        min_arr[i+4]   = (float)((md_byte >> 4) | ((m_byte >> 2) & 0x30));
    }
    
    /* Read 128 nibble-packed quants from bytes 16-143 */
    for (int i = 0; i < 256; i++) {
        int g = i / 32;
        int byte_idx = 16 + (i / 2);  /* 2 nibbles per byte */
        int nibble = (i & 1) ? (blk[byte_idx] >> 4) : (blk[byte_idx] & 0xF);
        float q = (float)((int)nibble - 8);
        out[i] = q * d_val * scale_arr[g] + dm_val * min_arr[g];
    }
}

/* Dequant Q6_K block (210 bytes → 256 floats) */
// Layout: ql[128] + qh[64] + scales[16 as int8] + d[2 as f16] = 210
// Dequant: val = (lo + hi<<4) - 32; result = val * d * scales[group]
static void deq_q6k(const uint8_t *blk, float *out) {
    const uint8_t *ql = blk;       // 128 bytes low nibbles
    const uint8_t *qh = blk + 128; // 64 bytes high bits
    const int8_t *sc = (const int8_t*)(blk + 192); // 16 x int8 scales
    uint16_t d_u; memcpy(&d_u, blk + 208, 2);
    float d_val;
    { int s=(d_u>>15)&1,e=(d_u>>10)&0x1F,m=d_u&0x3FF;
      if(e==0) d_val=(float)m*5.96e-8f; else if(e==31) d_val=0; else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&d_val,&b,4); } }
    for (int i = 0; i < 256; i++) {
        int sg = i / 16;
        int lo = (ql[i/2] >> (4*(i%2))) & 0xF;
        int hi = (qh[i/4] >> (2*(i%4))) & 0x3;
        int val = (lo | (hi << 4)) - 32;
        out[i] = (float)val * d_val * (float)sc[sg];
    }
    // Clamp all values to prevent Inf/NaN in Q8_0 conversion
    for (int i = 0; i < 256; i++) {
        if (!isfinite(out[i])) out[i] = 0.0f;
        if (out[i] > 1e10f) out[i] = 1e10f;
        if (out[i] < -1e10f) out[i] = -1e10f;
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
    
    if (sz % 210 == 0) {  // Q6_K first (prevents false Q4_K match)
        long n_blk = sz / 210;
        n_vals = n_blk * 256;
        f32 = (float*)calloc(n_vals, 4);
        for (long b = 0; b < n_blk; b++)
            deq_q6k(in + b * 210, f32 + b * 256);
    } else if (sz % 144 == 0) {  // Q4_K
        long n_blk = sz / 144;
        n_vals = n_blk * 256;
        f32 = (float*)calloc(n_vals, 4);
        for (long b = 0; b < n_blk; b++)
            deq_q4k(in + b * 144, f32 + b * 256);
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
