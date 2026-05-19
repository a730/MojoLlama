/* Scalar Q4_K batch matmul matching llama.cpp exactly */
void q4_k_batch_matmul_scalar(const uint8_t *W, const float *x, float *out,
                               int n_rows, int nc, int B) {
    int bpr = nc / QK_K;
    #pragma omp parallel for schedule(static, 8)
    for (int r = 0; r < n_rows; r++) {
        float acc[1024];
        for (int b = 0; b < B; b++) acc[b] = 0.0f;
        for (int blk = 0; blk < bpr; blk++) {
            const block_q4_K *bp = (const block_q4_K*)(W + ((size_t)r * bpr + blk) * sizeof(block_q4_K));
            const float d = gf16(bp->d);
            const float min = gf16(bp->dm);
            uint8_t sc, m;
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                k4_scale(is + 0, bp->scales, &sc, &m);
                const float d1 = d * sc;
                const float m1 = min * m;
                k4_scale(is + 1, bp->scales, &sc, &m);
                const float d2 = d * sc;
                const float m2 = min * m;
                // Process 32 lower nibbles
                int o = blk * QK_K + j;
                for (int b = 0; b < B; b++) {
                    const float *xb = x + (size_t)b * nc + o;
                    float s1 = 0, s2 = 0;
                    for (int l = 0; l < 32; l++) {
                        uint8_t nib_low = (bp->qs[(j/2 + l)] >> 0) & 0xF;
                        uint8_t nib_high = (bp->qs[(j/2 + l)] >> 4) & 0xF;
                        s1 += (d1 * nib_low - m1) * xb[l];
                        s2 += (d2 * nib_high - m1) * xb[l + 32];
                    }
                    acc[b] += s1 + s2;
                }
                q += 32; is += 2;
            }
        }
        for (int b = 0; b < B; b++) out[(size_t)b * n_rows + r] = acc[b];
    }
}
