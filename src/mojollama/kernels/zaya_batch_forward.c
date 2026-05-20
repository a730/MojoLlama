/* ═════════════════════════════════════════════════════════════════════════
 * ZAYA batch_forward — interleaved attention/MoE, CCA, res_scales
 */
void zaya_batch_forward(const BC *c, const int *tokens, int B, float *ws) {
    int N=c->N, NH=c->NH, NKH=c->NKH, HD=c->HD, L=c->L;
    int nq=NH*HD, nk=NKH*HD;
    int S = nq > N ? nq : N; if (nk > S) S = nk;
    if (S < 8192) S = 8192;
    
    float *x = ws, *xn = ws + 4*S, *res = ws + 8*S;
    float *q = ws + 12*S, *k_ = ws + 16*S;
    float *att = ws + 20*S, *gate_buf = ws + 24*S, *up_buf = ws + 28*S;
    float *oproj = ws + 32*S, *ffn = ws + 36*S;
    float *h_buf = ws + 40*S;
    int moe_ff = c->zaya_moe_intermediate;
    
    for (int b = 0; b < B; b++)
        memcpy(x + b*N, c->emb + (size_t)tokens[b]*N, N*sizeof(float));
    
    for (int l = 0; l < L; l++) {
        int is_moe = c->has_moe_layer ? c->has_moe_layer[l] : (l % 2);
        for (int b = 0; b < B; b++) {
            float *xb = x + b*N, *rb = res + b*N;
            float ss = 0;
            for (int i = 0; i <= N-8; i+=8) {
                __m256 xv = _mm256_loadu_ps(xb+i);
                xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
                _mm256_storeu_ps(xb+i, xv); _mm256_storeu_ps(rb+i, xv);
                ss += hsum_ps(_mm256_mul_ps(xv, xv));
            }
            for (int i = N-(N%8); i < N; i++) {
                float v = xb[i]; if(v!=v)v=0;else if(v>1000)v=1000;else if(v<-1000)v=-1000;
                xb[i]=v; rb[i]=v; ss += v*v;
            }
            float rms_val = sqrtf(ss/N + c->eps);
            for (int j = 0; j < N; j++) xb[j] = (xb[j]/rms_val) * c->wAN[l][j];
        }
        
        if (!is_moe) {
            /* attention */
            batch_matmul(c->q_quant[l], c->wQ[l], xn, q, nq, N, B);
            batch_matmul(c->k_quant[l], c->wK[l], xn, k_, nk, N, B);
            
            for (int b = 0; b < B; b++) {
                float *qb = q + b*nq, *kb = k_ + b*nk;
                rope_apply(qb, NH, HD, c->max_ctx, c->cos_table, c->sin_table, c->rope_dim);
                rope_apply(kb, NKH, HD, c->max_ctx, c->cos_table, c->sin_table, c->rope_dim);
                KVBlock *kv = c->kv_array[l];
                int pos = kv->pos++;
                memcpy(kv->k + (size_t)pos * nk, kb, nk * sizeof(float));
                memcpy(kv->v + (size_t)pos * nk, kb, nk * sizeof(float));
                int sl = pos + 1;
                gqa(att + b*nq, qb, kv->k, kv->v, NH, NKH, HD, sl);
                batch_matmul(c->o_quant[l], c->wO[l], att + b*nq, oproj + b*N, N, nq, 1);
                float *xbo = x + b*N;
                for (int i = 0; i < N; i++)
                    xbo[i] = rb[i]*c->w_zaya_res_res_w[l][i] + c->w_zaya_res_res_b[l][i] +
                             oproj[b*N+i]*c->w_zaya_res_hs_w[l][i] + c->w_zaya_res_hs_b[l][i];
            }
        } else {
            /* MoE */
            int rn = 256;
            for (int b = 0; b < B; b++) {
                float *xb = xn + b*N;
                int ne = c->zaya_expert_n;
                float h[256], scores[32];
                batch_matmul(c->zaya_ffn_gate_inp_quant, c->w_zaya_ffn_gate_inp[l], xb, h, rn, N, 1);
                batch_matmul(c->zaya_ffn_gate_quant, c->w_zaya_ffn_gate[l], h, h_buf, rn, rn, 1);
                for (int i = 0; i < rn; i++) {
                    float v = h_buf[i];
                    h[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f*v*v*v)));
                }
                batch_matmul(c->zaya_mlp2_quant, c->w_zaya_mlp2[l], h, h_buf, rn, rn, 1);
                batch_matmul(c->zaya_mlp4_quant, c->w_zaya_mlp4[l], h_buf, scores, ne+1, rn, 1);
                if (c->w_zaya_router_bias[l]) {
                    for (int e = 0; e < ne+1; e++) scores[e] += c->w_zaya_router_bias[l][e];
                }
                float mx = scores[0]; for (int e = 1; e < ne+1; e++) if (scores[e] > mx) mx = scores[e];
                float sum = 0; for (int e = 0; e < ne+1; e++) { scores[e] = expf(scores[e]-mx); sum += scores[e]; }
                float inv = 1.0f/(sum+1e-10f);
                for (int e = 0; e < ne+1; e++) scores[e] *= inv;
                int ec = 0; float ew = scores[0];
                for (int e = 1; e < ne; e++) { if (scores[e] > ew) { ew = scores[e]; ec = e; } }
                memset(ffn + b*N, 0, N*sizeof(float));
                if (ec < ne && ew > 0.01f) {
                    float renorm = 0; for (int e = 0; e < ne; e++) renorm += scores[e];
                    ew /= renorm;
                    int f2 = moe_ff / 2;
                    batch_matmul(c->gate_exp_quant, c->w_gate_exps[l] + (size_t)ec * moe_ff * (N/32)*34,
                                 xb, up_buf, moe_ff, N, 1);
                    for (int i = 0; i < f2; i++) {
                        float g = up_buf[i];
                        up_buf[i] = (g / (1.0f + expf(-g))) * up_buf[f2 + i];
                    }
                    batch_matmul(c->down_exp_quant, c->w_down_exps[l] + (size_t)ec * N * (moe_ff/32)*34,
                                 up_buf, ffn + b*N, N, f2, 1);
                    for (int i = 0; i < N; i++) ffn[b*N+i] *= ew;
                }
                float *xbo = x + b*N;
                for (int i = 0; i < N; i++)
                    xbo[i] = rb[i]*c->w_zaya_res_res_w[l][i] + c->w_zaya_res_res_b[l][i] +
                              ffn[b*N+i]*c->w_zaya_res_hs_w[l][i] + c->w_zaya_res_hs_b[l][i];
            }
        }
    }
    
    /* final norm + logits */
    for (int b = 0; b < B; b++) {
        float *xb = x + b*N; float ss = 0;
        for (int i = 0; i <= N-8; i+=8) {
            __m256 xv = _mm256_loadu_ps(xb+i);
            xv = _mm256_min_ps(_mm256_max_ps(xv, _mm256_set1_ps(-1000.0f)), _mm256_set1_ps(1000.0f));
            _mm256_storeu_ps(xb+i, xv); ss += hsum_ps(_mm256_mul_ps(xv, xv));
        }
        for (int i = N-(N%8); i < N; i++) { float v=xb[i]; ss += v*v; }
        float rms_val = sqrtf(ss/N + c->eps);
        for (int j = 0; j < N; j++) xb[j] = (xb[j]/rms_val) * c->onw[j];
        batch_matmul(c->outQuant, c->wOut, xb, c->logits + (size_t)b * c->V, c->V, N, 1);
    }
}
