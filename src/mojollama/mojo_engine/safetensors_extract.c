/* safetensors_extract.c — Safetensors → raw .bin extractor
 * WHAT:  Reads any .safetensors file, extracts all tensors to raw .bin files.
 * WHY:   Mojo 1.0.0b1 can't parse JSON headers (no String, no dicts).
 * WHEN:  2026-05-22 — first version.
 *
 * Format: [8 bytes header_len LE] [JSON header] [binary tensor data...]
 * JSON: {"__metadata__":{}, "tensor_name":{"dtype":"F16","shape":[N,M],"data_offsets":[S,E]}}
 *
 * Usage: ./safetensors_extract model.safetensors [out_dir]
 * Default out_dir: ./weights/
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/* ── Read 8 bytes (little-endian) ── */
static uint64_t r8(const void *p) {
    const unsigned char *b = (const unsigned char*)p;
    return (uint64_t)b[0] | ((uint64_t)b[1]<<8) | ((uint64_t)b[2]<<16) | ((uint64_t)b[3]<<24)
         | ((uint64_t)b[4]<<32) | ((uint64_t)b[5]<<40) | ((uint64_t)b[6]<<48) | ((uint64_t)b[7]<<56);
}

/* ── Human-readable size ── */
static void fmt_sz(char *buf, size_t sz) {
    if (sz < 1024)           sprintf(buf, "%zuB", sz);
    else if (sz < 1048576)   sprintf(buf, "%.0fKB", (double)sz/1024);
    else                     sprintf(buf, "%.0fMB", (double)sz/1048576);
}

/* ── Convert HF tensor name → file-safe name ── */
static void mangle(const char *src, char *dst, size_t cap) {
    if (strncmp(src, "model.", 6) == 0) src += 6;
    size_t j = 0;
    for (; *src && j+5 < cap; src++) {
        if      (*src == '.') dst[j++] = '_';
        else if (*src == '/') dst[j++] = '_';
        else                  dst[j++] = *src;
    }
    dst[j] = 0;
}

/* ── Find a key's value string in a sub-JSON buffer ── */
/* Scans {"key":"val", ...} and copies val into out */
static int sj_str(const unsigned char *sj, size_t len, const char *key, char *out, size_t cap) {
    size_t kl = strlen(key), i = 0;
    while (i < len) {
        if (sj[i] != '"') { i++; continue; }
        size_t ks = i+1, ke = ks;
        while (ke < len && sj[ke] != '"') ke++;
        size_t fl = ke - ks;
        if (fl == kl && memcmp(sj+ks, key, kl) == 0) {
            size_t p = ke+1;
            while (p < len && (sj[p]==':'||sj[p]==' '||sj[p]=='\t'||sj[p]=='\n')) p++;
            if (p >= len || sj[p] != '"') return 0;
            size_t vs = p+1, ve = vs;
            while (ve < len && sj[ve] != '"') ve++;
            size_t vl = ve - vs;
            if (vl >= cap) vl = cap-1;
            memcpy(out, sj+vs, vl); out[vl] = 0;
            return 1;
        }
        i = ke+1;
    }
    return 0;
}

/* ── Find a key's integer array value in sub-JSON ── */
/* Scans {"key":[N0,N1,...]} and returns nth integer */
static int sj_int(const unsigned char *sj, size_t len, const char *key, int nth, int64_t *out) {
    size_t kl = strlen(key), i = 0;
    while (i < len) {
        if (sj[i] != '"') { i++; continue; }
        size_t ks = i+1, ke = ks;
        while (ke < len && sj[ke] != '"') ke++;
        size_t fl = ke - ks;
        if (fl == kl && memcmp(sj+ks, key, kl) == 0) {
            size_t p = ke+1;
            while (p < len && (sj[p]==':'||sj[p]==' '||sj[p]=='\t'||sj[p]=='\n')) p++;
            if (p >= len || sj[p] != '[') return 0;
            p++; int cnt = 0;
            while (p < len && sj[p] != ']') {
                while (p < len && (sj[p]==' '||sj[p]==','||sj[p]=='\t'||sj[p]=='\n')) p++;
                if (p >= len || sj[p] == ']') break;
                if (sj[p] >= '0' && sj[p] <= '9') {
                    int64_t n = 0;
                    while (p < len && sj[p] >= '0' && sj[p] <= '9') { n = n*10 + (sj[p]-'0'); p++; }
                    if (cnt == nth) { *out = n; return 1; }
                    cnt++;
                } else p++;
            }
            return 0;
        }
        i = ke+1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.safetensors [out_dir]\n", argv[0]);
        return 1;
    }
    const char *inpath = argv[1];
    const char *outdir = (argc > 2) ? argv[2] : "./weights";
    
    /* Create output dir */
    struct stat st = {0};
    if (stat(outdir, &st) == -1)
        if (mkdir(outdir, 0755) != 0 && errno != EEXIST)
            { fprintf(stderr, "ERROR: can't create '%s'\n", outdir); return 1; }
    
    /* Open file */
    FILE *f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "ERROR: can't open '%s'\n", inpath); return 1; }
    
    /* Read 8-byte header = JSON length */
    unsigned char h8[8];
    if (fread(h8, 1, 8, f) != 8) { fprintf(stderr, "ERROR: short header\n"); fclose(f); return 1; }
    uint64_t jlen = r8(h8);
    
    if (jlen > 100*1024*1024) {
        fprintf(stderr, "ERROR: JSON header too large (%llu MB)\n", (unsigned long long)(jlen/1048576));
        fclose(f); return 1;
    }
    
    /* Read JSON header */
    unsigned char *j = (unsigned char*)malloc(jlen + 1);
    if (!j) { fprintf(stderr, "ERROR: out of memory\n"); fclose(f); return 1; }
    if (fread(j, 1, jlen, f) != jlen) { fprintf(stderr, "ERROR: can't read JSON\n"); free(j); fclose(f); return 1; }
    j[jlen] = 0;
    
    /* Read data section */
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    size_t doff = 8 + jlen;
    size_t dlen = (size_t)(fsz - (long)doff);
    
    unsigned char *d = (unsigned char*)malloc(dlen);
    if (!d) { fprintf(stderr, "ERROR: out of memory (%zu bytes)\n", dlen); free(j); fclose(f); return 1; }
    fseek(f, (long)doff, SEEK_SET);
    if (fread(d, 1, dlen, f) != dlen) { fprintf(stderr, "ERROR: can't read data\n"); free(j); free(d); fclose(f); return 1; }
    fclose(f);
    
    printf("Safetensors: JSON=%lluKB Data=%zuMB\n",
           (unsigned long long)(jlen/1024), dlen/1048576);
    
    /* Scan JSON for tensor entries */
    size_t pos = 0;
    int count = 0;
    char name[256], mang[256], dtype[32];
    
    while (pos < jlen) {
        /* Find '"' */
        if (j[pos] != '"') { pos++; continue; }
        size_t ks = pos+1, ke = ks;
        while (ke < jlen && j[ke] != '"') ke++;
        if (ke >= jlen) break;
        size_t nlen = ke - ks;
        
        /* Check if this key is followed by :{ (tensor entry) */
        size_t cp = ke+1;
        while (cp < jlen && (j[cp]==' '||j[cp]=='\t'||j[cp]=='\n')) cp++;
        if (cp >= jlen || j[cp] != ':') { pos = ke+1; continue; }
        cp++;
        while (cp < jlen && (j[cp]==' '||j[cp]=='\t'||j[cp]=='\n')) cp++;
        if (cp >= jlen || j[cp] != '{') { pos = ke+1; continue; }
        
        /* Find matching '}' for this tensor's JSON object */
        size_t os = cp+1; /* past '{' */
        size_t oe = os; int depth = 1;
        while (oe < jlen && depth > 0) {
            if (j[oe] == '{') depth++;
            else if (j[oe] == '}') depth--;
            oe++;
        }
        /* oe is past the matching '}' */
        
        /* Skip __metadata__ */
        if (nlen == 12 && memcmp(j+ks, "__metadata__", 12) == 0) {
            pos = oe; continue;
        }
        
        /* Copy tensor name */
        size_t cp_n = (nlen < 255) ? nlen : 254;
        memcpy(name, j+ks, cp_n); name[cp_n] = 0;
        
        /* Extract from sub-JSON [os, oe-2] */
        size_t sj_start = os;
        size_t sj_len = (oe > os) ? (oe - os - 1) : 0;
        
        if (sj_len > 0) {
            int64_t off0 = 0, off1 = 0;
            if (sj_int(j+sj_start, sj_len, "data_offsets", 0, &off0) &&
                sj_int(j+sj_start, sj_len, "data_offsets", 1, &off1)) {
                
                size_t ts = (size_t)off0;
                size_t te = (size_t)off1;
                size_t tsz = te - ts;
                
                if (tsz > 0 && ts + tsz <= dlen) {
                    mangle(name, mang, sizeof(mang));
                    char fpath[1024];
                    snprintf(fpath, sizeof(fpath), "%s/%s.bin", outdir, mang);
                    
                    FILE *fw = fopen(fpath, "wb");
                    if (fw) {
                        size_t wr = fwrite(d+ts, 1, tsz, fw);
                        fclose(fw);
                        char sb[24]; fmt_sz(sb, wr);
                        printf("  [%d] %s (%s)\n", ++count, mang, sb);
                    } else {
                        fprintf(stderr, "  [%d] FAIL: can't write %s\n", count+1, fpath);
                    }
                }
            }
        }
        pos = oe;
    }
    
    printf("Extracted %d tensors to '%s/'\n", count, outdir);
    free(j); free(d);
    return 0;
}
