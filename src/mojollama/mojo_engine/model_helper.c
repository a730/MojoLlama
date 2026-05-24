/* model_helper.c — Mojo FFI: HTTP server + JSON chat parser + tokenizer + OpenAI response */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <poll.h>
#include <ctype.h>
#include <time.h>

#define MAX_BODY 65536
#define MAX_QUEUE 64
#define MAX_VOCAB 300000
#define MAX_TOKEN_LEN 256

/* ─── HTTP Server ─── */
typedef struct { int id; char body[MAX_BODY]; int client_fd; int active; } Request;
static Request queue[MAX_QUEUE];
static int qhead = 0, qtail = 0, next_id = 1;
static pthread_mutex_t qmutex = PTHREAD_MUTEX_INITIALIZER;
static int srv_fd = -1;
static volatile int running = 0;
static pthread_t srv_thread;

static void send_http(int fd, int status, const char *st, const char *body) {
    char h[4096]; int n = snprintf(h, sizeof(h),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", status, st, strlen(body));
    write(fd, h, n); write(fd, body, strlen(body)); close(fd);
}

static char* skip_ws(char *p) { while (*p == ' '||*p=='\t'||*p=='\n'||*p=='\r') p++; return p; }
static char* parse_str(char **pp) {
    char *p = skip_ws(*pp); if (*p != '"') return NULL; p++;
    int len = 0; char *q = p; while (*q && *q != '"') { if (*q == '\\') q++; q++; len++; }
    char *r = malloc(len + 1); if (!r) return NULL; int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\') { p++;
            switch (*p) { case 'n': r[i++]='\n';break; case 't': r[i++]='\t';break; case 'r': r[i++]='\r';break; default: r[i++]=*p;break; }
            if (*p) p++;
        } else r[i++] = *p++;
    }
    r[i] = 0; if (*p == '"') p++; *pp = p; return r;
}

/* Parse chat request: find last {"role":"user","content":"..."} and return content. Caller free. */
char* extract_user_message(const char *body) {
    const char *p = body; const char *last_user_obj = NULL;
    while (*p) {
        p = skip_ws((char*)p);
        if (*p != '"') { p++; continue; }
        char *k = parse_str((char**)&p); p = skip_ws((char*)p);
        if (*p != ':') { free(k); p++; continue; } p++; p = skip_ws((char*)p);
        if (k && strcmp(k, "role") == 0 && *p == '"') {
            char *v = parse_str((char**)&p);
            if (v && strcmp(v, "user") == 0) {
                last_user_obj = p - 1; /* mark position */
            }
            free(v);
        }
        free(k);
        /* Find next field or object end */
        if (*p == ',' || *p == '}') { if (*p == ',') p++; continue; }
        if (*p == '{') { int d=1; p++; while(*p&&d>0){if(*p=='{')d++;if(*p=='}')d--;if(*p=='"'){char *s=parse_str((char**)&p);free(s);continue;}p++;} }
        else if (*p == '[') { int d=1; p++; while(*p&&d>0){if(*p=='[')d++;if(*p==']')d--;if(*p=='"'){char *s=parse_str((char**)&p);free(s);continue;}p++;} }
        else if (*p == '"') { char *s = parse_str((char**)&p); free(s); }
        else p++;
    }
    if (!last_user_obj) return NULL;
    /* Find "content" field in the user message object */
    p = last_user_obj; int depth = 1; if (*p == '}') p++;
    p = strstr(p, "\"content\"");
    if (!p) return NULL;
    p = strchr(p, ':'); if (!p) return NULL; p = skip_ws((char*)p + 1);
    return parse_str((char**)&p);
}

/* Extract integer from JSON */
int json_get_int(const char *json, const char *key, int def) {
    char *p = strstr(json, key); if (!p) return def;
    p = strchr(p, ':'); if (!p) return def; p = skip_ws(p+1);
    return atoi(p);
}

/* ─── Tokenizer (GGUF vocabulary) ─── */
static char **vocab = NULL; static int *vocab_lens = NULL; static int vocab_sz = 0;
static int bos_id = 2, eos_id = 1;

int tokenizer_load(const char *dir) {
    if (vocab) { for (int i=0; i<vocab_sz; i++) free(vocab[i]); free(vocab); free(vocab_lens); }
    char path[1024]; snprintf(path, sizeof(path), "%s/vocab.bin", dir);
    int fd = open(path, O_RDONLY); if (fd < 0) return -1;
    struct stat st; fstat(fd, &st); int sz = st.st_size;
    char *data = malloc(sz); if (!data) { close(fd); return -1; }
    read(fd, data, sz); close(fd);
    if (sz < 4) { free(data); return -1; }
    vocab_sz = *(int*)data; int pos = 4;
    if (vocab_sz > MAX_VOCAB) vocab_sz = MAX_VOCAB;
    vocab = calloc(vocab_sz, sizeof(char*)); vocab_lens = calloc(vocab_sz, sizeof(int));
    if (!vocab || !vocab_lens) { free(data); return -1; }
    for (int i = 0; i < vocab_sz && pos < sz; i++) {
        if (pos + 4 > sz) break; int len = *(int*)(data + pos); pos += 4;
        if (len < 0 || len > MAX_TOKEN_LEN || pos + len > sz) break;
        vocab[i] = malloc(len + 1); if (vocab[i]) { memcpy(vocab[i], data + pos, len); vocab[i][len] = 0; vocab_lens[i] = len; }
        pos += len; if (pos + 4 > sz) break;
        float score; memcpy(&score, data + pos, 4); pos += 4; (void)score;
    }
    free(data);
    /* Load special tokens */
    snprintf(path, sizeof(path), "%s/special_tokens.txt", dir);
    fd = open(path, O_RDONLY);
    if (fd >= 0) { char buf[64]; int n = read(fd, buf, sizeof(buf)-1); buf[n]=0; close(fd); sscanf(buf, "%d %d", &bos_id, &eos_id); }
    return vocab_sz;
}

/* Tokenize text → malloc'd int array. Sets *count. */
int* tokenize(const char *text, int *count) {
    if (!vocab || vocab_sz == 0) { *count = 0; return NULL; }
    int cap = 4096; int *ids = malloc(cap * sizeof(int)); int n = 0;
    int len = strlen(text); int pos = 0;
    /* Always add BOS */
    if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
    ids[n++] = bos_id;
    while (pos < len) {
        int best_id = -1, best_len = 0;
        for (int i = 0; i < vocab_sz; i++) {
            if (!vocab[i]) continue;
            int tl = vocab_lens[i];
            if (tl <= best_len || pos + tl > len) continue;
            if (strncmp(text + pos, vocab[i], tl) == 0) { best_id = i; best_len = tl; }
        }
        if (best_id >= 0) {
            if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n++] = best_id; pos += best_len;
        } else { /* Byte fallback */
            if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n++] = (unsigned char)text[pos] + 3; pos++;
        }
    }
    *count = n; return ids;
}

/* Detokenize IDs → malloc'd string. */
char* detokenize(const int *ids, int count) {
    if (!vocab || count == 0) return strdup("");
    int cap = 4096; char *res = malloc(cap); int pos = 0;
    for (int i = 0; i < count; i++) {
        int id = ids[i]; if (id < 0 || id >= vocab_sz || !vocab[id]) continue;
        if (id == bos_id || id == eos_id) continue;
        int tl = vocab_lens[id];
        if (pos + tl + 1 >= cap) { cap = cap * 2 + tl; res = realloc(res, cap); if (!res) return NULL; }
        memcpy(res + pos, vocab[id], tl); pos += tl;
    }
    res[pos] = 0; return res;
}

/* ─── HTTP Handler ─── */
static void handle_client(int fd, const char *buf) {
    char method[64] = {0}, path[1024] = {0};
    sscanf(buf, "%63s %1023s", method, path);
    if (strcmp(method, "OPTIONS") == 0) { send_http(fd, 204, "No Content", ""); return; }
    if (strcmp(path, "/health") == 0) { send_http(fd, 200, "OK", "{\"status\":\"ok\"}"); return; }
    if (strcmp(path, "/v1/chat/completions") == 0 && strcmp(method, "POST") == 0) {
        char *body = strstr(buf, "\r\n\r\n"); if (!body) { send_http(fd, 400, "Bad Request", "{\"error\":\"no_body\"}"); return; }
        body += 4;
        pthread_mutex_lock(&qmutex);
        int i = qtail % MAX_QUEUE; queue[i].id = next_id++; queue[i].client_fd = fd;
        strncpy(queue[i].body, body, MAX_BODY - 1); queue[i].active = 1; qtail++;
        pthread_mutex_unlock(&qmutex);
        return; /* Response sent by Mojo via send_response() */
    }
    send_http(fd, 404, "Not Found", "{\"error\":\"not_found\"}");
}

static void *server_loop(void *arg) {
    (void)arg; struct sockaddr_in a; socklen_t al = sizeof(a);
    while (running) {
        struct pollfd p = {srv_fd, POLLIN, 0};
        if (poll(&p, 1, 500) <= 0) continue;
        int c = accept(srv_fd, (struct sockaddr*)&a, &al);
        if (c < 0) continue;
        char buf[MAX_BODY]; int n = read(c, buf, sizeof(buf) - 1);
        if (n > 0) { buf[n] = 0; handle_client(c, buf); } else close(c);
    }
    return NULL;
}

/* ─── Public API for Mojo ─── */
int start_model_server(int port) {
    if (running) return -1;
    srv_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv_fd < 0) return -1;
    int opt = 1; setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons(port);
    if (bind(srv_fd, (struct sockaddr*)&a, sizeof(a)) < 0) { close(srv_fd); return -1; }
    if (listen(srv_fd, 5) < 0) { close(srv_fd); return -1; }
    running = 1; pthread_create(&srv_thread, NULL, server_loop, NULL); pthread_detach(srv_thread);
    return 0;
}

char* poll_request(void) {
    pthread_mutex_lock(&qmutex);
    if (qhead >= qtail || !queue[qhead % MAX_QUEUE].active) { pthread_mutex_unlock(&qmutex); return NULL; }
    int i = qhead % MAX_QUEUE; char *p = strdup(queue[i].body);
    pthread_mutex_unlock(&qmutex); return p;
}

void send_response(const char *resp) {
    pthread_mutex_lock(&qmutex);
    if (qhead < qtail) {
        int i = qhead % MAX_QUEUE;
        if (queue[i].active) { send_http(queue[i].client_fd, 200, "OK", resp); queue[i].active = 0; }
        qhead++;
    }
    pthread_mutex_unlock(&qmutex);
}

char* build_openai_response(const char *content, int pt, int ct, const char *model) {
    char *esc = malloc(strlen(content) * 2 + 1); if (!esc) return NULL; int j = 0;
    for (int i = 0; content[i]; i++) {
        char c = content[i];
        if (c == '"' || c == '\\') { esc[j++] = '\\'; esc[j++] = c; }
        else if (c == '\n') { esc[j++] = '\\'; esc[j++] = 'n'; }
        else if (c == '\t') { esc[j++] = '\\'; esc[j++] = 't'; }
        else esc[j++] = c;
    }
    esc[j] = 0;
    char *r; asprintf(&r,
        "{\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
        "\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
        rand(), (long)time(NULL), model ? model : "mojollama", esc, pt, ct, pt + ct);
    free(esc); return r;
}

/* Read arch.json integer */
int read_arch_int(const char *dir, const char *key) {
    char path[1024]; snprintf(path, sizeof(path), "%s/arch.json", dir);
    int fd = open(path, O_RDONLY); if (fd < 0) return 0;
    char buf[4096]; int n = read(fd, buf, sizeof(buf) - 1); close(fd);
    if (n <= 0) return 0; buf[n] = 0;
    char search[256]; snprintf(search, sizeof(search), "\"%s\":", key);
    char *p = strstr(buf, search); if (!p) return 0;
    p += strlen(search); while (*p && (*p == ' ' || *p == '\t')) p++;
    int val = 0, neg = 0; if (*p == '-') { neg = 1; p++; }
    while (*p && isdigit(*p)) { val = val * 10 + (*p - '0'); p++; }
    return neg ? -val : val;
}

/* Write binary token IDs to stdout (for mojollama_api.py subprocess mode) */
int mojo_write_tokens(int fd, const void *buf, int count) {
    return (int)write(fd, buf, (size_t)count);
}
