/* server_helper.c — Multi-model HTTP server with subprocess routing
   Routes /v1/chat/completions to the appropriate model binary.
   Model binaries are spawned on-demand per request type.
   
   Design: Single HTTP listener. Each request identifies the model.
   Server spawns the matching model binary and pipes token IDs through STDIN/STDOUT.
*/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <pthread.h>
#include <poll.h>
#include <errno.h>
#include <time.h>
#include <signal.h>

#define MAX_BODY 65536
#define MAX_PATH 1024
#define MAX_MODELS 8
#define MAX_LINE 4096

/* ─── Model Registry ─── */
typedef struct {
    char name[64];
    char binary_path[512];
    char weights_path[512];
    int port;  /* internal port for this model instance */
    pid_t pid; /* running process PID, 0 if not started */
} ModelEntry;

static ModelEntry models[MAX_MODELS];
static int num_models = 0;
static int server_fd = -1;
static volatile int server_running = 0;
static pthread_t server_thread;

/* ─── JSON Helpers ─── */
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
        if (*p == '\\') { p++;
            switch (*p) { case 'n': result[i++] = '\n'; break; case 't': result[i++] = '\t'; break; case 'r': result[i++] = '\r'; break; default: result[i++] = *p; break; }
            if (*p) p++;
        } else result[i++] = *p++;
    }
    result[i] = 0;
    if (*p == '"') p++;
    *pp = p;
    return result;
}

static char* json_get_string(const char *json, const char *key) {
    const char *p = json;
    while (*p) {
        p = (char*)skip_ws((char*)p);
        if (*p != '"') { p++; continue; }
        char *k = parse_json_string((char**)&p);
        p = (char*)skip_ws((char*)p);
        if (*p != ':') { free(k); continue; }
        p++; p = (char*)skip_ws((char*)p);
        if (k && strcmp(k, key) == 0) {
            if (*p == '"') { char *val = parse_json_string((char**)&p); free(k); return val; }
            /* Handle non-string values by reading until comma/brace */
            const char *start = p;
            while (*p && *p != ',' && *p != '}' && *p != ']') p++;
            int len = p - start;
            char *val = malloc(len + 1);
            memcpy(val, start, len); val[len] = 0;
            free(k); return val;
        }
        free(k);
        /* Skip value */
        if (*p == '{' || *p == '[') {
            int depth = 1; p++;
            while (*p && depth > 0) {
                if (*p == '{' || *p == '[') depth++;
                else if (*p == '}' || *p == ']') depth--;
                else if (*p == '"') { char *s = parse_json_string((char**)&p); free(s); continue; }
                p++;
            }
        } else {
            while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\t') p++;
        }
        if (*p == ',') p++;
    }
    return NULL;
}

/* ─── Send HTTP Response ─── */
static void send_http(int fd, int status, const char *status_text, const char *body) {
    char header[4096];
    int n = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n", status, status_text, strlen(body));
    write(fd, header, n);
    write(fd, body, strlen(body));
    close(fd);
}

/* ─── Model Process Management ─── */
static int start_model_instance(ModelEntry *model) {
    if (model->pid > 0) {
        /* Check if still alive */
        if (kill(model->pid, 0) == 0) return 0; /* Still running */
        model->pid = 0;
    }
    
    /* Start the model binary with the weights path */
    pid_t pid = fork();
    if (pid == 0) {
        /* Child: exec the model binary */
        char port_str[16];
        snprintf(port_str, sizeof(port_str), "%d", model->port);
        
        /* Set up environment */
        setenv("OMP_PLACES", "cores", 1);
        setenv("OMP_PROC_BIND", "close", 1);
        
        execl(model->binary_path, model->binary_path, port_str, "32", model->weights_path, NULL);
        /* If exec fails */
        _exit(1);
    } else if (pid > 0) {
        model->pid = pid;
        /* Give it a moment to start */
        usleep(2000000);
        return 0;
    }
    return -1;
}

/* ─── Route a chat request to the appropriate model ─── */
static void handle_chat_request(int client_fd, const char *body) {
    /* Extract "model" field from the request */
    char *model_name = json_get_string(body, "model");
    if (!model_name) {
        /* Default to first model */
        if (num_models == 0) {
            send_http(client_fd, 400, "Bad Request", "{\"error\":\"no_models_configured\"}");
            return;
        }
        model_name = strdup(models[0].name);
    }
    
    /* Find matching model */
    ModelEntry *model = NULL;
    for (int i = 0; i < num_models; i++) {
        if (strcmp(model_name, models[i].name) == 0) {
            model = &models[i];
            break;
        }
    }
    
    if (!model) {
        char err[256];
        snprintf(err, sizeof(err), "{\"error\":\"unknown_model\",\"available\":[");
        for (int i = 0; i < num_models; i++) {
            if (i > 0) strcat(err, ",");
            strcat(err, "\""); strcat(err, models[i].name); strcat(err, "\"");
        }
        strcat(err, "]}");
        send_http(client_fd, 404, "Not Found", err);
        free(model_name);
        return;
    }
    
    /* Ensure model instance is running */
    if (start_model_instance(model) != 0) {
        send_http(client_fd, 500, "Error", "{\"error\":\"failed_to_start_model\"}");
        free(model_name);
        return;
    }
    
    /* Forward request to model's internal HTTP port */
    /* Build proxy request to the internal model server */
    char proxy_req[32768];
    int n = snprintf(proxy_req, sizeof(proxy_req),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n%s",
        model->port, strlen(body), body);
    
    /* Connect to internal model server */
    int proxy_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (proxy_fd < 0) {
        send_http(client_fd, 502, "Bad Gateway", "{\"error\":\"cannot_connect\"}");
        free(model_name);
        return;
    }
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(0x7F000001); /* 127.0.0.1 */
    addr.sin_port = htons(model->port);
    
    if (connect(proxy_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(proxy_fd);
        send_http(client_fd, 502, "Bad Gateway", "{\"error\":\"connection_refused\"}");
        free(model_name);
        return;
    }
    
    /* Send request and get response */
    write(proxy_fd, proxy_req, n);
    char resp_buf[65536];
    int resp_n = read(proxy_fd, resp_buf, sizeof(resp_buf) - 1);
    close(proxy_fd);
    
    if (resp_n <= 0) {
        send_http(client_fd, 502, "Bad Gateway", "{\"error\":\"no_response\"}");
        free(model_name);
        return;
    }
    resp_buf[resp_n] = 0;
    
    /* Extract body from HTTP response */
    char *body_start = strstr(resp_buf, "\r\n\r\n");
    if (body_start) {
        body_start += 4;
        send_http(client_fd, 200, "OK", body_start);
    } else {
        send_http(client_fd, 502, "Bad Gateway", "{\"error\":\"bad_response\"}");
    }
    
    free(model_name);
}

/* ─── HTTP Handler ─── */
static void handle_client(int client_fd) {
    char buf[MAX_BODY];
    int n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = 0;
    
    char method[64] = {0}, path[MAX_PATH] = {0};
    sscanf(buf, "%63s %1023s", method, path);
    
    if (strcmp(method, "OPTIONS") == 0) {
        send_http(client_fd, 204, "No Content", "");
        return;
    }
    
    if (strcmp(path, "/health") == 0 || strcmp(path, "/") == 0) {
        char resp[1024];
        snprintf(resp, sizeof(resp), 
            "{\"status\":\"ok\",\"models\":[");
        for (int i = 0; i < num_models; i++) {
            if (i > 0) strcat(resp, ",");
            strcat(resp, "\""); strcat(resp, models[i].name); strcat(resp, "\"");
        }
        strcat(resp, "]}");
        send_http(client_fd, 200, "OK", resp);
        return;
    }
    
    if (strcmp(path, "/v1/models") == 0) {
        char resp[4096];
        snprintf(resp, sizeof(resp), "{\"object\":\"list\",\"data\":[");
        for (int i = 0; i < num_models; i++) {
            if (i > 0) strcat(resp, ",");
            char entry[256];
            snprintf(entry, sizeof(entry),
                "{\"id\":\"%s\",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"mojo\"}",
                models[i].name, (long)time(NULL));
            strcat(resp, entry);
        }
        strcat(resp, "]}");
        send_http(client_fd, 200, "OK", resp);
        return;
    }
    
    if (strcmp(path, "/v1/chat/completions") == 0 && strcmp(method, "POST") == 0) {
        char *body = strstr(buf, "\r\n\r\n");
        if (!body) { send_http(client_fd, 400, "Bad Request", "{\"error\":\"no_body\"}"); return; }
        body += 4;
        handle_chat_request(client_fd, body);
        return;
    }
    
    send_http(client_fd, 404, "Not Found", "{\"error\":\"not_found\"}");
}

static void *server_loop(void *arg) {
    (void)arg;
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    
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

/* ─── Public API ─── */

/* Register a model. Call before start_server(). */
int register_model(const char *name, const char *binary, const char *weights, int port) {
    if (num_models >= MAX_MODELS) return -1;
    int i = num_models++;
    strncpy(models[i].name, name, sizeof(models[i].name) - 1);
    strncpy(models[i].binary_path, binary, sizeof(models[i].binary_path) - 1);
    strncpy(models[i].weights_path, weights, sizeof(models[i].weights_path) - 1);
    models[i].port = port;
    models[i].pid = 0;
    return 0;
}

/* Start the HTTP server */
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

void stop_server(void) {
    server_running = 0;
    if (server_fd >= 0) { close(server_fd); server_fd = -1; }
    /* Kill model processes */
    for (int i = 0; i < num_models; i++) {
        if (models[i].pid > 0) {
            kill(models[i].pid, SIGTERM);
            waitpid(models[i].pid, NULL, WNOHANG);
        }
    }
}
