/* gguf_extract.c — GGUF → raw .bin extractor (standalone CLI)
   WHAT:  Reads any GGUF file, extracts all tensors to <out_dir>/<name>.bin
   WHY:   Standalone binary called from Mojo. Avoids @extern("write") stdlib conflict.
   USAGE: ./gguf_extract <model.gguf> <output_dir>
   WHEN:  May 2026.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#define GGUF_ALIGN 32

static uint32_t r4(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static uint64_t r8(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (i*8);
    return v;
}
static uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: gguf_extract <model.gguf> [output_dir]\n");
        return 1;
    }
    const char *path = argv[1];
    char out_dir[1024];
    if (argc >= 3) {
        snprintf(out_dir, sizeof(out_dir), "%s", argv[2]);
    } else {
        // Default output dir: /tmp/<modelname>_weights/
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        snprintf(out_dir, sizeof(out_dir), "/tmp/%.*s_weights", (int)(strlen(base)-5), base);
    }

    // Read up to 4MB for headers
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    uint64_t fsz = (uint64_t)lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    uint64_t chunk = min_u64(fsz, 4194304ULL);
    uint8_t *buf = (uint8_t*)malloc(chunk);
    if (!buf) { close(fd); return 1; }
    if (read(fd, buf, chunk) != (ssize_t)chunk) { free(buf); close(fd); return 1; }
    close(fd);

    if (r4(buf) != 0x46554747) { fprintf(stderr, "Not GGUF\n"); free(buf); return 1; }
    uint64_t n_tensors = r8(buf + 8);
    uint64_t n_kv = r8(buf + 16);
    printf("GGUF: %lu tensors, %lu KV, %.0fMB\n", n_tensors, n_kv, fsz/1048576.0);

    // Create output dir
    mkdir(out_dir, 0755);

    uint64_t pos = 24;
    // Skip KV pairs
    for (uint64_t k = 0; k < n_kv; k++) {
        uint64_t klen = r8(buf + pos); pos += 8;
        pos += klen;
        uint32_t vtype = r4(buf + pos); pos += 4;
        switch (vtype) {
            case 0: case 1: pos += 1; break;
            case 2: case 3: pos += 2; break;
            case 4: case 5: case 6: pos += 4; break;
            case 7: pos += 1; break;
            case 8: { uint64_t sl = r8(buf + pos); pos += 8 + sl; break; }
            case 9: {
                uint32_t at = r4(buf + pos); pos += 4;
                uint64_t al = r8(buf + pos); pos += 8;
                for (uint64_t a = 0; a < al; a++) {
                    if (at == 8) { uint64_t sl = r8(buf + pos); pos += 8 + sl; }
                    else { int esz = (at <= 1) ? 1 : (at <= 3) ? 2 : 4; pos += esz; }
                }
                break;
            }
            case 10: case 11: case 12: pos += 8; break;
            default: pos += 8; break;
        }
    }
    uint64_t ti_start = pos;

    // First pass: find total tensor info size
    uint64_t ti_end = ti_start;
    for (uint64_t t = 0; t < n_tensors; t++) {
        if (ti_end + 8 > chunk) break;
        uint64_t nl = r8(buf + ti_end); ti_end += 8;
        ti_end += nl; // no alignment (GGUF v3)
        if (ti_end + 4 > chunk) break;
        uint32_t nd = r4(buf + ti_end); ti_end += 4;
        ti_end += nd * 8;
        ti_end += 12; // dtype(4) + offset(8)
    }
    uint64_t ti_total = ti_end - ti_start;
    uint64_t data_start = (ti_start + ti_total + GGUF_ALIGN - 1) & ~(GGUF_ALIGN - 1);
    printf("Data at offset %lu\n", data_start);

    // Reopen file for reading tensor data
    fd = open(path, O_RDONLY);
    if (fd < 0) { free(buf); return 1; }

    // Second pass: extract each tensor
    uint64_t tp = ti_start;
    uint64_t extracted = 0, errors = 0;
    uint64_t total_bytes = 0;

    for (uint64_t t = 0; t < n_tensors; t++) {
        uint64_t nl = r8(buf + tp); tp += 8;
        char name[1024];
        if (nl > 1023) nl = 1023;
        memcpy(name, buf + tp, nl);
        name[nl] = '\0';
        tp += nl;

        uint32_t nd = r4(buf + tp); tp += 4;
        uint64_t dims[4] = {1,1,1,1};
        for (int d = 0; d < nd && d < 4; d++) { dims[d] = r8(buf + tp); tp += 8; }
        uint32_t dtype = r4(buf + tp); tp += 4;
        uint64_t toff = r8(buf + tp); tp += 8;

        // Sanity check offsets
        if (toff >= fsz) { fprintf(stderr, "  SKIP %s: offset %lu > file size %ld\\n", name, toff, fsz); errors++; continue; }

        uint64_t ne = dims[0]*dims[1]*dims[2]*dims[3];
        uint64_t dsz = 0;
        switch (dtype) {
            case 0: dsz = ne * 4; break;          // F32
            case 1: dsz = ne * 2; break;          // F16
            case 2: dsz = ((ne+31)/32)*18; break; // Q4_0
            case 3: dsz = ((ne+31)/32)*20; break; // Q4_1
            case 6: dsz = ((ne+31)/32)*34; break; // Q8_0
            case 12: dsz = ((ne+255)/256)*144; break; // Q4_K
            case 14: dsz = ((ne+255)/256)*240; break; // Q6_K
            case 30: dsz = ne * 2; break;            // BF16
            case 39: dsz = ((ne+31)/32)*34; break; // MXFP4
            case 47: dsz = ((ne+31)/32)*34; break; // MXFP4
            default: dsz = ne * 2; break;
        }

        // Build output filename: replace '.' and '/' with '_'
        char fpath[2048];
        int nw = snprintf(fpath, sizeof(fpath), "%s/", out_dir);
        for (int i = 0; name[i]; i++) {
            if (name[i] == '.' || name[i] == '/') fpath[nw++] = '_';
            else fpath[nw++] = name[i];
        }
        memcpy(fpath + nw, ".bin", 5);

        // Read tensor data from file
        uint64_t file_pos = data_start + toff;
        FILE *out = fopen(fpath, "wb");
        if (!out) { errors++; continue; }

        uint8_t tmp[131072]; // 128KB chunks
        uint64_t remaining = dsz;
        while (remaining > 0) {
            uint64_t chunk = remaining > sizeof(tmp) ? sizeof(tmp) : remaining;
            uint64_t got = (uint64_t)pread(fd, tmp, chunk, file_pos);
            if (got != chunk) { fprintf(stderr, "  short read %s\n", name); break; }
            fwrite(tmp, 1, got, out);
            file_pos += got; remaining -= got;
        }
        fclose(out);
        extracted++; total_bytes += dsz;

        char sz_str[32];
        if (dsz > 1048576) snprintf(sz_str, sizeof(sz_str), "%.1fMB", dsz/1048576.0);
        else if (dsz > 1024) snprintf(sz_str, sizeof(sz_str), "%.0fKB", dsz/1024.0);
        else snprintf(sz_str, sizeof(sz_str), "%luB", dsz);
        printf("  [%lu/%lu] %s (%s)\n", t+1, n_tensors, fpath + strlen(out_dir) + 1, sz_str);
    }
    close(fd); free(buf);

    printf("\n[DONE] %lu tensors extracted to %s/ (%.1fMB total)\n",
           extracted, out_dir, total_bytes/1048576.0);
    return errors ? 1 : 0;
}
