/* tokenizer.c — SentencePiece tokenizer for Mojo FFI
   Loads vocabulary from GGUF tokenizer data, encodes text to token IDs.
   
   Tokenizer format (stored in weights dir as vocab.bin):
   - 4 bytes: num_tokens (int32)
   - For each token: 4 bytes len, len bytes string, 4 bytes score (float32 as int32)  
   
   Encoding: longest-prefix-match with unigram scoring fallback to byte fallback.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#define MAX_VOCAB 300000
#define MAX_TOKEN_LEN 256

static char **vocab = NULL;
static float *scores = NULL;
static int vocab_size = 0;
static int *token_lens = NULL;
static int bos_id = 2;   /* default BOS */
static int eos_id = 1;   /* default EOS */
static int unk_id = 0;   /* default UNK */

/* ─── Load vocabulary from file ─── */
int load_tokenizer(const char *path) {
    /* Free old data */
    if (vocab) {
        for (int i = 0; i < vocab_size; i++) free(vocab[i]);
        free(vocab); free(scores); free(token_lens);
        vocab = NULL; scores = NULL; token_lens = NULL;
    }
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st; fstat(fd, &st);
    int sz = st.st_size;
    char *data = malloc(sz);
    if (!data) { close(fd); return -1; }
    read(fd, data, sz); close(fd);
    
    int pos = 0;
    if (pos + 4 > sz) { free(data); return -1; }
    vocab_size = *(int*)(data + pos); pos += 4;
    
    if (vocab_size > MAX_VOCAB) vocab_size = MAX_VOCAB;
    
    vocab = calloc(vocab_size, sizeof(char*));
    scores = calloc(vocab_size, sizeof(float));
    token_lens = calloc(vocab_size, sizeof(int));
    if (!vocab || !scores || !token_lens) { free(data); return -1; }
    
    for (int i = 0; i < vocab_size && pos < sz; i++) {
        if (pos + 4 > sz) break;
        int len = *(int*)(data + pos); pos += 4;
        if (len < 0 || len > MAX_TOKEN_LEN || pos + len > sz) break;
        vocab[i] = malloc(len + 1);
        if (vocab[i]) { memcpy(vocab[i], data + pos, len); vocab[i][len] = 0; token_lens[i] = len; }
        pos += len;
        if (pos + 4 > sz) break;
        scores[i] = *(float*)(data + pos); pos += 4;
    }
    
    free(data);
    return vocab_size;
}

/* ─── Load special token IDs ─── */
void set_special_tokens(int bos, int eos, int unk) {
    bos_id = bos; eos_id = eos; unk_id = unk;
}

/* ─── Encode text to token IDs ─── */
/* Returns malloc'd array of IDs, sets *count. Caller must free(). */
int* tokenize(const char *text, int *count) {
    if (!vocab || vocab_size == 0) { *count = 0; return NULL; }
    
    int cap = 4096;
    int *ids = malloc(cap * sizeof(int));
    int n = 0;
    int text_len = strlen(text);
    int pos = 0;
    
    /* Add BOS token */
    if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
    ids[n++] = bos_id;
    
    while (pos < text_len) {
        int best_id = -1;
        int best_len = 0;
        
        /* Find longest matching token starting at pos */
        for (int i = 0; i < vocab_size; i++) {
            if (!vocab[i]) continue;
            int tlen = token_lens[i];
            if (tlen <= best_len || pos + tlen > text_len) continue;
            if (strncmp(text + pos, vocab[i], tlen) == 0) {
                best_id = i;
                best_len = tlen;
            }
        }
        
        if (best_id >= 0) {
            if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n++] = best_id;
            pos += best_len;
        } else {
            /* Byte fallback: encode as <0xXX> or use a byte-level token */
            /* SentencePiece byte tokens are at specific IDs */
            /* Fallback: use the raw byte value as token (assuming bytes in vocab) */
            if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n++] = (unsigned char)text[pos] + 3; /* Simple byte mapping */
            pos++;
        }
    }
    
    *count = n;
    return ids;
}

/* ─── Decode token IDs back to text ─── */
char* detokenize(const int *ids, int count) {
    if (!vocab || count == 0) return strdup("");
    
    int cap = 4096;
    char *result = malloc(cap);
    if (!result) return NULL;
    int pos = 0;
    
    for (int i = 0; i < count; i++) {
        int id = ids[i];
        if (id < 0 || id >= vocab_size || !vocab[id]) continue;
        /* Skip special tokens */
        if (id == bos_id || id == eos_id) continue;
        
        int len = token_lens[id];
        if (pos + len + 1 >= cap) {
            cap = cap * 2 + len;
            result = realloc(result, cap);
            if (!result) return NULL;
        }
        memcpy(result + pos, vocab[id], len);
        pos += len;
    }
    result[pos] = 0;
    return result;
}

/* ─── Free resources ─── */
void free_tokenizer(void) {
    if (vocab) {
        for (int i = 0; i < vocab_size; i++) free(vocab[i]);
        free(vocab); free(scores); free(token_lens);
        vocab = NULL; scores = NULL; token_lens = NULL;
        vocab_size = 0;
    }
}
