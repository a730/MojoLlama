/* server_helper.c — HTTP server, JSON parser, and tokenizer for Mojo FFI */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <poll.h>
#include <errno.h>
#include <math.h>

#include "server_helper.h"

/* ─── HTTP Server (raw POSIX sockets) ─── */
static int server_fd = -1;
static volatile int server_running = 0;
static pthread_t server_thread;
static request_handler_t global_handler = NULL;
static void *global_ctx = NULL;

/* Simple response: HTTP/1.1 200 OK */
static void send_http_response(int client_fd, int status, const char *status_text, 
                                const char *content_type, const char *body, int body_len) {
    char header[4096];
    int n = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    write(client_fd, header, n);
    write(client_fd, body, body_len);
}

static void handle_client(int client_fd) {
    char buf[65536];
    int n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = 0;
    
    /* CORS preflight */
    if (strncmp(buf, "OPTIONS", 7) == 0) {
        send_http_response(client_fd, 204, "No Content", "text/plain", "", 0);
        close(client_fd);
        return;
    }
    
    /* Parse method and path */
    char method[64] = {0}, path[1024] = {0};
    sscanf(buf, "%63s %1023s", method, path);
    
    /* Find body (after \r\n\r\n) */
    char *body_start = strstr(buf, "\r\n\r\n");
    char *body = body_start ? body_start + 4 : "";
    
    int resp_len = 0;
    char *response = global_handler(global_ctx, method, path, body, &resp_len);
    if (!response) {
        char *err = "{\"error\":\"internal_error\"}";
        send_http_response(client_fd, 500, "Internal Server Error", "application/json", err, strlen(err));
    } else {
        send_http_response(client_fd, 200, "OK", "application/json", response, resp_len);
        free(response);
    }
    close(client_fd);
}

static void *server_loop(void *arg) {
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    
    while (server_running) {
        struct pollfd pfd = {server_fd, POLLIN, 0};
        int ret = poll(&pfd, 1, 500); /* 500ms timeout */
        if (ret < 0) break;
        if (ret == 0) continue;
        
        int client_fd = accept(server_fd, (struct sockaddr*)&addr, &addr_len);
        if (client_fd < 0) continue;
        handle_client(client_fd);
    }
    return NULL;
}

int server_start(int port, request_handler_t handler, void *ctx) {
    if (server_running) return -1;
    
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return -1;
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(server_fd); return -1;
    }
    if (listen(server_fd, 5) < 0) {
        close(server_fd); return -1;
    }
    
    global_handler = handler;
    global_ctx = ctx;
    server_running = 1;
    
    pthread_create(&server_thread, NULL, server_loop, NULL);
    pthread_detach(server_thread);
    
    return server_fd;
}

void server_stop(void) {
    server_running = 0;
    if (server_fd >= 0) {
        close(server_fd);
        server_fd = -1;
    }
}

/* ─── Minimal JSON Parser ─── */
static char* skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return (char*)p;
}

/* Allocate and copy a JSON string value (including escape handling) */
static char* parse_json_string(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++; /* skip opening quote */
    
    /* Count length first */
    int len = 0;
    const char *q = p;
    while (*q && *q != '"') {
        if (*q == '\\') { q++; if (*q) q++; len++; }
        else { q++; len++; }
    }
    
    char *result = malloc(len + 1);
    if (!result) return NULL;
    
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case 'n': result[i++] = '\n'; break;
                case 't': result[i++] = '\t'; break;
                case 'r': result[i++] = '\r'; break;
                case '\\': result[i++] = '\\'; break;
                case '"': result[i++] = '"'; break;
                default: result[i++] = *p; break;
            }
            if (*p) p++;
        } else {
            result[i++] = *p++;
        }
    }
    result[i] = 0;
    if (*p == '"') p++;
    *pp = p;
    return result;
}

/* Skip any JSON value */
static void skip_value(const char **pp) {
    const char *p = skip_ws(*pp);
    if (!*p) return;
    if (*p == '"') { char *s = parse_json_string(&p); free(s); }
    else if (*p == '{') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') { char *s = parse_json_string(&p); free(s); continue; }
            p++;
        }
        if (*p == '}') p++;
    }
    else if (*p == '[') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') { char *s = parse_json_string(&p); free(s); continue; }
            p++;
        }
        if (*p == ']') p++;
    }
    else {
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    }
    *pp = p;
}

char* json_get_string(const char *json, const char *key) {
    const char *p = json;
    while (*p) {
        p = skip_ws(p);
        if (*p != '"') { p++; continue; }
        const char *ks = p;
        char *k = parse_json_string(&p);
        p = skip_ws(p);
        if (*p != ':') { free(k); continue; }
        p++; p = skip_ws(p);
        if (k && strcmp(k, key) == 0) {
            if (*p == '"') {
                char *val = parse_json_string(&p);
                free(k);
                return val;
            }
        }
        free(k);
        if (*p == ',' || *p == '}') { if (*p == ',') p++; continue; }
        skip_value(&p);
        if (*p == ',') p++;
    }
    return NULL;
}

int json_get_int(const char *json, const char *key, int default_val) {
    const char *p = json;
    while (*p) {
        p = skip_ws(p);
        if (*p != '"') { p++; continue; }
        char *k = parse_json_string(&p);
        p = skip_ws(p);
        if (*p != ':') { free(k); continue; }
        p++; p = skip_ws(p);
        if (k && strcmp(k, key) == 0) {
            free(k);
            long val = strtol(p, (char**)&p, 10);
            return (int)val;
        }
        free(k);
        skip_value(&p);
        if (*p == ',') p++;
    }
    return default_val;
}

double json_get_double(const char *json, const char *key, double default_val) {
    const char *p = json;
    while (*p) {
        p = skip_ws(p);
        if (*p != '"') { p++; continue; }
        char *k = parse_json_string(&p);
        p = skip_ws(p);
        if (*p != ':') { free(k); continue; }
        p++; p = skip_ws(p);
        if (k && strcmp(k, key) == 0) {
            free(k);
            return strtod(p, (char**)&p);
        }
        free(k);
        skip_value(&p);
        if (*p == ',') p++;
    }
    return default_val;
}

/* Get the "content" field from the last message in the "messages" array */
char* json_get_last_message(const char *json) {
    /* Find "messages" array */
    const char *p = strstr(json, "\"messages\"");
    if (!p) return NULL;
    p = strchr(p, '[');
    if (!p) return NULL;
    p++; /* skip [ */
    
    /* Find last object in array */
    const char *last_obj_start = NULL;
    int depth = 0;
    while (*p) {
        p = skip_ws(p);
        if (*p == '{') {
            if (!last_obj_start || depth == 0) last_obj_start = p;
            depth++;
            p++;
        } else if (*p == '}') {
            depth--;
            if (depth == 0) { p++; /* skip past the } */ break; }
            p++;
        } else if (*p == '[') { depth++; p++; }
        else if (*p == ']') { break; }
        else if (*p == '"') { char *s = parse_json_string(&p); free(s); }
        else p++;
    }
    
    if (!last_obj_start) return NULL;
    
    /* Find content field in this object */
    /* Copy the object to a null-terminated string */
    /* Actually just search for "content" in the last object */
    char obj_buf[32768];
    int obj_len = p - last_obj_start;
    if (obj_len > 32767) obj_len = 32767;
    memcpy(obj_buf, last_obj_start, obj_len);
    obj_buf[obj_len] = 0;
    
    return json_get_string(obj_buf, "content");
}

/* ─── JSON Builder ─── */
char* build_chat_response(const char *model, const char *content,
                          int prompt_tokens, int completion_tokens) {
    /* Escape content for JSON */
    int content_len = strlen(content);
    int escaped_len = 0;
    for (int i = 0; i < content_len; i++) {
        char c = content[i];
        if (c == '"' || c == '\\' || c == '\n' || c == '\t' || c == '\r') escaped_len += 2;
        else if (c < 32) escaped_len += 6; /* \u00XX */
        else escaped_len++;
    }
    
    char *escaped = malloc(escaped_len + 1);
    if (!escaped) return NULL;
    int j = 0;
    for (int i = 0; i < content_len; i++) {
        char c = content[i];
        if (c == '"') { escaped[j++] = '\\'; escaped[j++] = '"'; }
        else if (c == '\\') { escaped[j++] = '\\'; escaped[j++] = '\\'; }
        else if (c == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
        else if (c == '\t') { escaped[j++] = '\\'; escaped[j++] = 't'; }
        else if (c == '\r') { escaped[j++] = '\\'; escaped[j++] = 'r'; }
        else if (c < 32) { j += snprintf(escaped + j, 7, "\\u%04x", (unsigned char)c); }
        else { escaped[j++] = c; }
    }
    escaped[j] = 0;
    
    char *result;
    int n = asprintf(&result,
        "{\"id\":\"chatcmpl-mojo\",\"object\":\"chat.completion\","
        "\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
        "\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
        (long)time(NULL), model ? model : "gemma-4",
        escaped, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    
    free(escaped);
    return result;
}

/* ─── Tokenizer (Simple longest-prefix-match) ─── */

/* Load tokenizer data from memory (from GGUF) */
static char **vocab_tokens = NULL;
static int *vocab_scores = NULL;
static int vocab_size = 0;

/* 
 * Load tokenizer data: expects token_strings as a large block of:
 *   - first 4 bytes: num_tokens (int)
 *   - for each token: 4 bytes length, then that many bytes of string, then 4 bytes score
 */
int tokenizer_load_data(const char *data, int data_len) {
    /* Free old data */
    if (vocab_tokens) {
        for (int i = 0; i < vocab_size; i++) free(vocab_tokens[i]);
        free(vocab_tokens); free(vocab_scores);
    }
    
    const unsigned char *p = (const unsigned char*)data;
    int remaining = data_len;
    
    if (remaining < 4) return -1;
    vocab_size = *(int*)p; p += 4; remaining -= 4;
    
    vocab_tokens = calloc(vocab_size, sizeof(char*));
    vocab_scores = calloc(vocab_size, sizeof(int));
    if (!vocab_tokens || !vocab_scores) return -1;
    
    for (int i = 0; i < vocab_size && remaining > 4; i++) {
        int tlen = *(int*)p; p += 4; remaining -= 4;
        if (tlen > remaining) { tlen = remaining; }
        vocab_tokens[i] = malloc(tlen + 1);
        if (vocab_tokens[i]) { memcpy(vocab_tokens[i], p, tlen); vocab_tokens[i][tlen] = 0; }
        p += tlen; remaining -= tlen;
        if (remaining >= 4) {
            vocab_scores[i] = *(int*)p; p += 4; remaining -= 4;
        }
    }
    return vocab_size;
}

void tokenizer_free(void) {
    if (vocab_tokens) {
        for (int i = 0; i < vocab_size; i++) free(vocab_tokens[i]);
        free(vocab_tokens); free(vocab_scores);
        vocab_tokens = NULL; vocab_scores = NULL; vocab_size = 0;
    }
}

/* Find the longest matching token at position pos in text */
static int find_longest_token(const char *text, int pos, int text_len) {
    int best_id = -1;
    int best_len = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        if (!vocab_tokens[i]) continue;
        int tlen = strlen(vocab_tokens[i]);
        if (tlen <= best_len || pos + tlen > text_len) continue;
        if (strncmp(text + pos, vocab_tokens[i], tlen) == 0) {
            best_id = i;
            best_len = tlen;
        }
    }
    return best_id;
}

token_array* tokenizer_encode(const char *text) {
    token_array *ta = calloc(1, sizeof(token_array));
    if (!ta) return NULL;
    ta->capacity = 1024;
    ta->ids = malloc(ta->capacity * sizeof(int));
    if (!ta->ids) { free(ta); return NULL; }
    
    int text_len = strlen(text);
    int pos = 0;
    
    while (pos < text_len) {
        int tid = find_longest_token(text, pos, text_len);
        if (tid >= 0) {
            if (ta->count >= ta->capacity) {
                ta->capacity *= 2;
                ta->ids = realloc(ta->ids, ta->capacity * sizeof(int));
            }
            ta->ids[ta->count++] = tid;
            pos += strlen(vocab_tokens[tid]);
        } else {
            /* Unknown byte: use byte-fallback with <0xF7> prefix (SentencePiece byte fallback) */
            if (ta->count + 2 >= ta->capacity) {
                ta->capacity *= 2;
                ta->ids = realloc(ta->ids, ta->capacity * sizeof(int));
            }
            /* Byte token in SentencePiece: <0xXX> where XX is the hex byte value */
            /* We use token ID 3 + byte_value as a simple fallback */
            ta->ids[ta->count++] = 3 + (unsigned char)text[pos];
            pos++;
        }
    }
    
    return ta;
}

char* tokenizer_decode(const int *ids, int count) {
    int cap = 4096;
    char *result = malloc(cap);
    if (!result) return NULL;
    int pos = 0;
    
    for (int i = 0; i < count; i++) {
        int id = ids[i];
        if (id >= 0 && id < vocab_size && vocab_tokens[id]) {
            int tlen = strlen(vocab_tokens[id]);
            if (pos + tlen >= cap) {
                cap = cap * 2 + tlen;
                result = realloc(result, cap);
                if (!result) return NULL;
            }
            memcpy(result + pos, vocab_tokens[id], tlen);
            pos += tlen;
        }
    }
    result[pos] = 0;
    return result;
}

void token_array_free(token_array *ta) {
    if (ta) { free(ta->ids); free(ta); }
}
