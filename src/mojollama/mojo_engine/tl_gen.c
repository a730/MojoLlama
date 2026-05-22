/* tl_gen.c — TinyLlama autoregressive generation with f16 .bin weights */
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
#define MAX_SEQ 128

static void* load_file(const char *dir, const char *name) {
    char p[1024]; snprintf(p, sizeof(p), "%s/%s", dir, name);
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(sz); fread(b, 1, sz, f); fclose(f); return b;
}

static float h2f(uint16_t h) {
    uint32_t s=(h>>15)&1, e=(h>>10)&0x1F, m=h&0x3FF;
    if (!e) { float r=(float)m*5.960464477539063e-8f; return s?-r:r; }
    if (e==31) return 0;
    uint32_t b=(s<<31)|((e+112)<<23)|(m<<13); float r; memcpy(&r,&b,4); return r;
}

static void mm_f16(const void *w, const float *x, float *o, int M, int N) {
    #pragma omp parallel for
    for (int r=0; r<M; r++) {
        __m256 a=_mm256_setzero_ps();
        const uint16_t *wt=(const uint16_t*)w + r*N;
        int c; for (c=0; c+8<=N; c+=8) {
            a=_mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(wt+c))),
                              _mm256_loadu_ps(x+c), a);
        }
        float tail=0; for (; c<N; c++) tail+=h2f(wt[c])*x[c];
        __m128 hi=_mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a,1));
        hi=_mm_hadd_ps(hi,hi); hi=_mm_hadd_ps(hi,hi);
        o[r]=_mm_cvtss_f32(hi)+tail;
    }
}

typedef struct { int id; char *text; int len; } Entry;

static Entry* load_vocab(const char *path, int *n) {
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fread(n,4,1,f);
    Entry *v=calloc(*n,sizeof(Entry));
    for (int i=0;i<*n;i++) {
        int pl; fread(&pl,4,1,f);
        v[i].text=malloc(pl+1); fread(v[i].text,1,pl,f); v[i].text[pl]=0; v[i].len=pl;
        fread(&v[i].id,4,1,f);
    }
    fclose(f); return v;
}

static void decode(FILE *out, Entry *voc, int n, int tok) {
    for (int i=0;i<n;i++) if (voc[i].id==tok) {
        const char *p=voc[i].text; int l=voc[i].len;
        if (l>=3 && (unsigned char)p[0]==0xE2 && (unsigned char)p[1]==0x96 && (unsigned char)p[2]==0x81)
            { p+=3; l-=3; }
        fwrite(p,1,l,out);
        return;
    }
}

static void decode_raw(FILE *out, Entry *voc, int n, int tok) {
    for (int i=0;i<n;i++) if (voc[i].id==tok) {
        fwrite(voc[i].text,1,voc[i].len,out);
        return;
    }
}

int main(int argc,char**argv) {
    const char *wdir=argc>1?argv[1]:"/tmp/weights_tl";
    const char *vpath=argc>2?argv[2]:"/tmp/vocab.bin";

    int nv; Entry *voc=load_vocab(vpath,&nv);
    if(!voc){fprintf(stderr,"FAIL: vocab\n");return 1;}
    printf("Vocab: %d\n",nv);

    double t0=omp_get_wtime();
    void *emb=load_file(wdir,"token_embd_weight.bin");
    void *onw=load_file(wdir,"output_norm_weight.bin");
    void *lmw=load_file(wdir,"output_weight.bin");
    void *wl[NL][9];
    char*sfx[9]={"_attn_norm_weight.bin","_ffn_norm_weight.bin",
        "_attn_q_weight.bin","_attn_k_weight.bin","_attn_v_weight.bin",
        "_attn_output_weight.bin","_ffn_gate_weight.bin","_ffn_up_weight.bin","_ffn_down_weight.bin"};
    char nm[256];
    for(int l=0;l<NL;l++) for(int f=0;f<9;f++)
        {snprintf(nm,256,"blk_%d%s",l,sfx[f]); wl[l][f]=load_file(wdir,nm);}
    printf("Load: %.0f ms\n",(omp_get_wtime()-t0)*1000);

    // Prompt "2+2="
    int pr[]={1,29871,29906,29974,29906,29922}, np=6, ng=10;
    int toks[MAX_SEQ]; for(int i=0;i<np;i++) toks[i]=pr[i];
    int nt=np;

    float *kc=calloc(NL*NK*MAX_SEQ*HD,4);
    float *vc=calloc(NL*NK*MAX_SEQ*HD,4);
    float hp[NE],bp[NE],qp[NH*HD],kp[NK*HD],vbuf[NK*HD],gp[NF],up[NF],dp[NE],lp[NV];

    printf("Prompt: "); for(int i=0;i<np;i++) decode_raw(stdout,voc,nv,toks[i]);
    printf("\nOutput: "); fflush(stdout);

    t0=omp_get_wtime();
    for(int pos=0;pos<ng&&nt<MAX_SEQ;pos++){
        int tok=toks[pos];
        const uint16_t *e=(const uint16_t*)emb;
        for(int i=0;i<NE;i++) hp[i]=h2f(e[tok*NE+i]);

        for(int l=0;l<NL;l++){
            float ss=0; for(int i=0;i<NE;i++) ss+=hp[i]*hp[i];
            float in=1.0f/sqrtf(ss/NE+1e-6f);
            const float *an=(const float*)wl[l][0];
            for(int i=0;i<NE;i++) bp[i]=hp[i]*an[i]*in;

            mm_f16(wl[l][2],bp,qp,NH*HD,NE);
            mm_f16(wl[l][3],bp,kp,NK*HD,NE);
            mm_f16(wl[l][4],bp,vbuf,NK*HD,NE);

            for(int h=0;h<NH;h++) for(int d2=0;d2<HD;d2+=2){
                float f=pos/powf(10000,(float)d2/HD),c=cosf(f),s=sinf(f);
                float x0=qp[h*HD+d2],x1=qp[h*HD+d2+1];
                qp[h*HD+d2]=x0*c-x1*s; qp[h*HD+d2+1]=x0*s+x1*c;
            }
            for(int h=0;h<NK;h++) for(int d2=0;d2<HD;d2+=2){
                float f=pos/powf(10000,(float)d2/HD),c=cosf(f),s=sinf(f);
                float x0=kp[h*HD+d2],x1=kp[h*HD+d2+1];
                kp[h*HD+d2]=x0*c-x1*s; kp[h*HD+d2+1]=x0*s+x1*c;
            }

            int lo=l*NK*MAX_SEQ*HD;
            for(int h=0;h<NK;h++) for(int d=0;d<HD;d++){
                kc[lo+h*MAX_SEQ*HD+pos*HD+d]=kp[h*HD+d];
                vc[lo+h*MAX_SEQ*HD+pos*HD+d]=vbuf[h*HD+d];
            }

            int kr=NH/NK;
            for(int hq=0;hq<NH;hq++){
                int hk=hq/kr;
                float sc[MAX_SEQ],sm=-1e9;
                for(int p=0;p<=pos;p++){
                    float s=0;
                    for(int d=0;d<HD;d++) s+=qp[hq*HD+d]*kc[lo+hk*MAX_SEQ*HD+p*HD+d];
                    s/=sqrtf(HD); sc[p]=s; if(s>sm)sm=s;
                }
                float su=0; for(int p=0;p<=pos;p++){sc[p]=expf(sc[p]-sm); su+=sc[p];}
                for(int d=0;d<HD;d++){
                    float o=0;
                    for(int p=0;p<=pos;p++) o+=vc[lo+hk*MAX_SEQ*HD+p*HD+d]*(sc[p]/su);
                    qp[hq*HD+d]=o;
                }
            }

            mm_f16(wl[l][5],qp,bp,NE,NH*HD);
            for(int i=0;i<NE;i++) hp[i]+=bp[i];

            ss=0; for(int i=0;i<NE;i++) ss+=hp[i]*hp[i];
            in=1.0f/sqrtf(ss/NE+1e-6f);
            const float *fn=(const float*)wl[l][1];
            for(int i=0;i<NE;i++) bp[i]=hp[i]*fn[i]*in;

            mm_f16(wl[l][6],bp,gp,NF,NE);
            mm_f16(wl[l][7],bp,up,NF,NE);
            for(int i=0;i<NF;i++){float gv=gp[i];if(gv<-80)gv=-80;if(gv>80)gv=80;
                gp[i]=(gv/(1+expf(-gv)))*up[i];}
            mm_f16(wl[l][8],gp,dp,NE,NF);
            for(int i=0;i<NE;i++) hp[i]+=dp[i];
        }

        float ss=0; for(int i=0;i<NE;i++) ss+=hp[i]*hp[i];
        float in=1.0f/sqrtf(ss/NE+1e-6f);
        const float *on=(const float*)onw;
        for(int i=0;i<NE;i++) bp[i]=hp[i]*on[i]*in;
        mm_f16(lmw,bp,lp,NV,NE);

        int best=0; float bv=lp[0];
        for(int i=1;i<NV;i++) if(lp[i]>bv){bv=lp[i];best=i;}
        toks[nt++]=best;
        if(best==2) break;
        decode(stdout,voc,nv,best);
        fflush(stdout);
    }
    printf("\n");
    double t1=omp_get_wtime();
    printf("Generated %d tokens in %.0f ms (%.1f tok/s)\n",nt-np,(t1-t0)*1000,(nt-np)/(t1-t0));
    return 0;
}
