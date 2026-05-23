/* server_helper.c — HTTP server with poll-based request queue for Mojo FFI
   Design: C handles networking in a background thread. Mojo polls for requests.
   Flow: C receives HTTP POST → parses JSON → queues request 
         → Mojo calls poll_request() → processes → C sends response */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <poll.h>
#include <errno.h>
#include <time.h>

#define MAX_BODY 65536
#define MAX_QUEUE 64
#define MAX_PATH 1024

/* ─── Request Queue ─── */
typedef struct {
    int id;
    char method[64];
    char path[MAX_PATH];
    char body[MAX_BODY];
    int client_fd;
    int active;
} Request;

static Request request_queue[MAX_QUEUE];
static int queue_head = 0, queue_tail = 0, next_id = 1;
static pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t queue_cond = PTHREAD_COND_INITIALIZER;
static int server_fd = -1;
static volatile int server_running = 0;
static pthread_t server_thread;

/* ─── JSON helpers ─── */
static char* skip_ws(char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static char* parse_json_string(char **pp) {
    char *p = skip_ws(*pp);
    if (*p != '"') return NULL;
    p++;
    int len = 0;
    char *q = p;
    while (*q && *q != '"') { if (*q == '\\') q++; q++; len++; }
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
        } else result[i++] = *p++;
    }
    result[i] = 0;
    if (*p == '"') p++;
    *pp = p;
    return result;
}

/* Extract the "content" from the last message in a messages array */
char* extract_prompt(const char *body) {
    /* Find "messages" array */
    const char *p = strstr(body, "\"messages\"");
    if (!p) return NULL;
    p = strchr(p, '[');
    if (!p) return NULL;
    p++;
    
    /* Find the last object in the array */
    const char *last_obj = NULL;
    int depth = 0;
    while (*p) {
        p = skip_ws((char*)p);
        if (*p == '{') { if (!last_obj) last_obj = p; depth++; p++; }
        else if (*p == '}') { depth--; if (depth <= 0) { p++; break; } p++; }
        else if (*p == '[') { depth++; p++; }
        else if (*p == ']') break;
        else if (*p == '"') { char *s = parse_json_string((char**)&p); free(s); }
        else p++;
    }
    
    if (!last_obj) return NULL;
    int obj_len = p - last_obj;
    if (obj_len > 65535) obj_len = 65535;
    char obj[65536];
    memcpy(obj, last_obj, obj_len);
    obj[obj_len] = 0;
    
    /* Find "content" field */
    p = strstr(obj, "\"content\"");
    if (!p) return NULL;
    p = strchr(p, ':');
    if (!p) return NULL;
    p = skip_ws((char*)p + 1);
    
    char *content = parse_json_string((char**)&p);
    return content;  /* Caller must free */
}

/* Extract integer field from JSON */
int extract_int(const char *body, const char *key, int default_val) {
    char *kp = strstr(body, key);
    if (!kp) return default_val;
    kp = strchr(kp, ':');
    if (!kp) return default_val;
    kp = skip_ws(kp + 1);
    return atoi(kp);
}

/* ─── HTTP Response ─── */
static void send_http_response(int fd, int status, const char *status_text,
                                const char *content_type, const char *body) {
    char header[4096];
    int n = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n", status, status_text, content_type, strlen(body));
    write(fd, header, n);
    write(fd, body, strlen(body));
    close(fd);
}

/* Build OpenAI chat completion response JSON. Caller must free. */
char* build_openai_response(const char *content, int prompt_toks, int gen_toks, const char *model) {
    char *escaped = malloc(strlen(content) * 2 + 1);
    if (!escaped) return NULL;
    int j = 0;
    for (int i = 0; content[i]; i++) {
        char c = content[i];
        if (c == '"' || c == '\\') { escaped[j++] = '\\'; escaped[j++] = c; }
        else if (c == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
        else if (c == '\t') { escaped[j++] = '\\'; escaped[j++] = 't'; }
        else escaped[j++] = c;
    }
    escaped[j] = 0;
    
    char *result;
    asprintf(&result,
        "{\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion\","
        "\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
        "\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
        rand(), (long)time(NULL), model ? model : "gemma-4",
        escaped, prompt_toks, gen_toks, prompt_toks + gen_toks);
    free(escaped);
    return result;
}

/* ─── HTTP Handler Thread ─── */
static void handle_client(int client_fd) {
    char buf[MAX_BODY];
    int n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = 0;
    
    /* Parse method and path */
    char method[64] = {0}, path[MAX_PATH] = {0};
    sscanf(buf, "%63s %1023s", method, path);
    
    /* CORS preflight */
    if (strcmp(method, "OPTIONS") == 0) {
        send_http_response(client_fd, 204, "No Content", "text/plain", "");
        return;
    }
    
    /* Health check */
    if (strcmp(path, "/health") == 0 || strcmp(path, "/") == 0) {
        send_http_response(client_fd, 200, "OK", "application/json",
            "{\"status\":\"ok\",\"model\":\"gemma-4-mojo\"}");
        return;
    }
    
    /* Models list */
    if (strcmp(path, "/v1/models") == 0) {
        send_http_response(client_fd, 200, "OK", "application/json",
            "{\"object\":\"list\",\"data\":[{\"id\":\"gemma-4-mojo\",\"object\":\"model\",\"created\":0,\"owned_by\":\"mojo\"}]}");
        return;
    }
    
    /* Chat completions */
    if (strcmp(path, "/v1/chat/completions") == 0 && strcmp(method, "POST") == 0) {
        /* Find body */
        char *body = strstr(buf, "\r\n\r\n");
        if (!body) { send_http_response(client_fd, 400, "Bad Request", "application/json", "{\"error\":\"no body\"}"); return; }
        body += 4;
        
        /* Extract prompt and max_tokens */
        char *prompt = extract_prompt(body);
        int max_tokens = extract_int(body, "max_tokens", 128);
        
        if (!prompt) {
            send_http_response(client_fd, 400, "Bad Request", "application/json",
                "{\"error\":\"no messages\"}");
            return;
        }
        
        /* Queue the request for Mojo to process */
        pthread_mutex_lock(&queue_mutex);
        int idx = queue_tail % MAX_QUEUE;
        request_queue[idx].id = next_id++;
        request_queue[idx].client_fd = client_fd;
        strncpy(request_queue[idx].body, prompt, MAX_BODY - 1);
        request_queue[idx].body[MAX_BODY - 1] = 0;
        request_queue[idx].active = 1;
        queue_tail++;
        pthread_cond_signal(&queue_cond);
        pthread_mutex_unlock(&queue_mutex);
        
        free(prompt);
        /* Note: response is sent by Mojo via send_response() */
        return;
    }
    
    /* 404 */
    send_http_response(client_fd, 404, "Not Found", "application/json",
        "{\"error\":\"not_found\"}");
}

static void *server_loop(void *arg) {
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    (void)arg;
    
    while (server_running) {
        struct pollfd pfd = {server_fd, POLLIN, 0};
        int ret = poll(&pfd, 1, 500);
        if (ret <= 0) continue;
        
        int client_fd = accept(server_fd, (struct sockaddr*)&addr, &addr_len);
        if (client_fd < 0) continue;
        handle_client(client_fd);
    }
    return NULL;
}

/* ─── Public API for Mojo ─── */

/* Start the HTTP server on the given port. Returns 0 on success. */
int start_server(int port) {
    if (server_running) return -1;
    
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return -1;
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { close(server_fd); return -1; }
    if (listen(server_fd, 10) < 0) { close(server_fd); return -1; }
    
    server_running = 1;
    pthread_create(&server_thread, NULL, server_loop, NULL);
    pthread_detach(server_thread);
    return 0;
}

/* Poll for the next request. Returns the prompt text (caller must free), or NULL if none. */
char* poll_request(void) {
    pthread_mutex_lock(&queue_mutex);
    if (queue_head >= queue_tail || !request_queue[queue_head % MAX_QUEUE].active) {
        pthread_mutex_unlock(&queue_mutex);
        return NULL;
    }
    int idx = queue_head % MAX_QUEUE;
    char *prompt = strdup(request_queue[idx].body);
    pthread_mutex_unlock(&queue_mutex);
    return prompt;
}

/* Send the response for the current request. */
void send_response(const char *response) {
    pthread_mutex_lock(&queue_mutex);
    if (queue_head < queue_tail) {
        int idx = queue_head % MAX_QUEUE;
        if (request_queue[idx].active) {
            int fd = request_queue[idx].client_fd;
            send_http_response(fd, 200, "OK", "application/json", response);
            request_queue[idx].active = 0;
        }
        queue_head++;
    }
    pthread_mutex_unlock(&queue_mutex);
}

/* Stop the server */
void stop_server(void) {
    server_running = 0;
    if (server_fd >= 0) { close(server_fd); server_fd = -1; }
}
