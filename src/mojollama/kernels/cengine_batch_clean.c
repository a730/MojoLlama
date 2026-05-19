/* cengine_batch.c — vLLM-style concurrent batched inference.
 * Uses q4_0_batch_matmul for all weight projections across B tokens.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#define Q4_0_BS 18
#define Q8_0_BS 34
#define BLOCK_SIZE 64
#define MAX_BLOCKS 1024
#define QK_K 256

/* K-quant block structures (from llama.cpp) */
typedef struct {
    uint16_t d;    // half: super-block scale
    uint16_t dm;   // half: super-block min
    uint8_t scales[12];  // 8 × 6-bit scale/min pairs
    uint8_t qs[128];     // 4-bit nibbles (256 elements)
} block_q4_K;

typedef struct {
    uint8_t ql[128];     // lower 4 bits of quants
    uint8_t qh[64];      // upper 2 bits of quants
    int8_t  scales[16];  // 16 × int8 scales (one per 16 elements)
    uint16_t d;          // half: super-block scale
} block_q6_K;

/* GGML half <-> float conversion */
static inline float gf16(uint16_t h) {
    return _cvtsh_ss(h);
}

/* Q4_K scale extraction */
static inline void k4_scale(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}
