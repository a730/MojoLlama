/* transpose_q8.c — Transpose Q8_0 matrix files from [d0,d1] to [d1,d0].
   Reads a shape map: each line is "<filename>,<d0>,<d1>" (1D/2D).
   Usage: ./transpose_q8 <in_dir> <out_dir> <shapes.csv>
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/stat.h>
#include <math.h>

#define QK 32
#define QB 34
#define MAX_PATH 4096
#define MAX_LINE 512

static uint16_t f32_to_f16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    int s = (u >> 31) & 1, e = (u >> 23) & 0xFF, m = u & 0x7FFFFF;
    if (e == 0) return s << 15;
    if (e == 0xFF) return (s << 15) | 0x7C00;
    int ne = e - 127 + 15;
    if (ne >= 31) return (s << 15) | 0x7C00;
    if (ne <= 0) return s << 15;
    return (s << 15) | (ne << 10) | (m >> 13);
}

/* Dequantize Q8_0 block (34 bytes → 32 floats) */
static void deq_q8_block(const uint8_t *blk, float *out) {
    uint16_t s_u; memcpy(&s_u, blk, 2);
    float scale;
    { int s=(s_u>>15)&1,e=(s_u>>10)&0x1F,m=s_u&0x3FF;
      if(e==0) scale=(float)m*5.96e-8f; else if(e==31) scale=0;
      else { uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); memcpy(&scale,&b,4); } }
    for (int i = 0; i < QK; i++)
        out[i] = (float)((int8_t)blk[2+i]) * scale;
}

/* Quantize float32 row → Q8_0 bytes (appended to buf) */
static void q8_quant_row(const float *row, int n, uint8_t *buf, long *off) {
    int n_blk = (n + QK - 1) / QK;
    for (int b = 0; b < n_blk; b++) {
        int s = b * QK, e = s + QK < n ? s + QK : n;
        float max_abs = 0;
        for (int i = s; i < e; i++) { float a = fabsf(row[i]); if (a > max_abs) max_abs = a; }
        float scale = max_abs > 1e-10f ? max_abs / 127.0f : 1.0f;
        uint16_t su = f32_to_f16(scale);
        memcpy(buf + *off, &su, 2); *off += 2;
        for (int i = s; i < e; i++) {
            int qv = (int)roundf(row[i] / scale);
            if (qv > 127) qv = 127; if (qv < -128) qv = -128;
            buf[(*off)++] = (uint8_t)(qv & 0xFF);
        }
        // Pad to QK
        for (int i = e; i < s + QK; i++) buf[(*off)++] = 0;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <in_dir> <out_dir> <shapes.csv>\n", argv[0]);
        fprintf(stderr, "  shapes.csv: each line 'filename,d0,d1' where d1=0 means 1D (copy)\n");
        return 1;
    }
    
    mkdir(argv[2], 0755);
    
    // Read shape map
    FILE *sf = fopen(argv[3], "r");
    if (!sf) { perror("shapes.csv"); return 1; }
    
    char line[MAX_LINE];
    int n_total = 0, n_transpose = 0;
    
    while (fgets(line, sizeof(line), sf)) {
        // Strip newline
        char *nl = strchr(line, '\n'); if (nl) *nl = 0;
        if (line[0] == 0) continue;
        
        char fname[MAX_PATH];
        long d0, d1;
        if (sscanf(line, "%[^,],%ld,%ld", fname, &d0, &d1) != 3) continue;
        
        char in_path[MAX_PATH], out_path[MAX_PATH];
        snprintf(in_path, sizeof(in_path), "%s/%s", argv[1], fname);
        snprintf(out_path, sizeof(out_path), "%s/%s", argv[2], fname);
        
        // Check if output already exists
        struct stat st;
        if (stat(out_path, &st) == 0) continue;  // Already processed
        
        // Check input exists
        if (stat(in_path, &st) != 0) {
            fprintf(stderr, "  SKIP: %s not found\n", fname);
            continue;
        }
        
        // Read input file
        FILE *f = fopen(in_path, "rb");
        if (!f) { fprintf(stderr, "  FAIL: cannot open %s\n", fname); continue; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *in_data = (uint8_t*)malloc(sz);
        if (!in_data) { fclose(f); fprintf(stderr, "  FAIL: malloc %s\n", fname); continue; }
        fread(in_data, 1, sz, f); fclose(f);
        
        if (d1 == 0 || d0 == d1) {
            // 1D or square: just copy
            FILE *out = fopen(out_path, "wb");
            if (out) { fwrite(in_data, 1, sz, out); fclose(out); }
            free(in_data);
            n_total++;
            if (n_total % 100 == 0) fprintf(stderr, "  copied %d files...\r", n_total);
            continue;
        }
        
        // Transpose: [d0, d1] → [d1, d0]
        // Dequantize entire matrix
        long n_vals = d0 * d1;
        float *f32 = (float*)malloc(n_vals * sizeof(float));
        if (!f32) { free(in_data); fprintf(stderr, "  FAIL: malloc f32 %s\n", fname); continue; }
        
        long n_q8_blk = sz / QB;
        long n_q8_elems = n_q8_blk * QK;
        // Ensure we don't read past data
        long n_read = n_q8_elems < n_vals ? n_q8_elems : n_vals;
        for (long b = 0; b < n_read / QK; b++)
            deq_q8_block(in_data + b * QB, f32 + b * QK);
        
        free(in_data);
        
        // Transpose: f32[d0][d1] → f32_T[d1][d0]
        float *f32_T = (float*)malloc(n_vals * sizeof(float));
        if (!f32_T) { free(f32); fprintf(stderr, "  FAIL: malloc T %s\n", fname); continue; }
        
        for (long r = 0; r < d0; r++)
            for (long c = 0; c < d1; c++)
                f32_T[c * d0 + r] = f32[r * d1 + c];
        
        free(f32);
        
        // Quantize back to Q8_0
        long out_sz = ((d1 * d0 + QK - 1) / QK) * QB;
        uint8_t *q8 = (uint8_t*)calloc(1, out_sz);
        if (!q8) { free(f32_T); fprintf(stderr, "  FAIL: malloc q8 %s\n", fname); continue; }
        
        long off = 0;
        for (long r = 0; r < d1; r++)
            q8_quant_row(f32_T + r * d0, d0, q8, &off);
        
        free(f32_T);
        
        FILE *out = fopen(out_path, "wb");
        if (out) { fwrite(q8, 1, out_sz, out); fclose(out); }
        free(q8);
        
        n_total++; n_transpose++;
        if (n_total % 50 == 0)
            fprintf(stderr, "  %d files (%d transposed)...\r", n_total, n_transpose);
    }
    fclose(sf);
    
    fprintf(stderr, "\nDone: %d total, %d transposed\n", n_total, n_transpose);
    return 0;
}
