/* mojo_engine.c — Minimal inference engine binary.
 *
 * Links against ../kernels/q4_kernel_omp.so for quantised matmul kernels.
 * Exposes an HTTP /health endpoint for readiness checks.
 *
 * Usage:  ./mojo_engine --model <path> --port <num> --threads <num>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <omp.h>

/* ── Kernel symbols from q4_kernel_omp.so ────────────────────────── */
extern void q4_matmul_omp(const uint8_t *w, const float *x, float *out,
                           int n_rows, int n_cols, int type_size);
extern void q4_set_num_threads(int n);
extern int  q4_get_max_threads(void);

/* ── Global configuration ───────────────────────────────────────── */
static int g_port    = 8080;
static int g_threads = 4;
static const char *g_model_path = NULL;
static volatile int g_running = 1;

/* ── Signal handler ──────────────────────────────────────────────── */
static void handle_signal(int sig) {
    (void)sig;
    g_running = 0;
}

/* ── Simple HTTP response helpers ─────────────────────────────────── */
static const char HTTP_200_JSON[] =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: application/json\r\n"
    "Content-Length: %zu\r\n"
    "Connection: close\r\n\r\n%s";

static const char HTTP_404[] =
    "HTTP/1.1 404 Not Found\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n\r\n";

static void send_response(int client_fd, const char *header, const char *body) {
    char buf[4096];
    int n = snprintf(buf, sizeof(buf), header, strlen(body), body);
    send(client_fd, buf, n, 0);
}

/* ── HTTP server thread ───────────────────────────────────────────── */
static void *http_server(void *arg) {
    (void)arg;
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return NULL;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(g_port),
    };

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return NULL;
    }

    if (listen(server_fd, 16) < 0) {
        perror("listen");
        close(server_fd);
        return NULL;
    }

    fprintf(stderr, "[engine] listening on port %d (threads=%d, model=%s)\n",
            g_port, g_threads, g_model_path ? g_model_path : "(none)");

    while (g_running) {
        fd_set rfds;
        struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
        FD_ZERO(&rfds);
        FD_SET(server_fd, &rfds);
        int sel = select(server_fd + 1, &rfds, NULL, NULL, &tv);
        if (sel <= 0) continue;

        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;

        char req[2048] = {0};
        recv(client_fd, req, sizeof(req) - 1, 0);

        /* Very minimal routing */
        if (strncmp(req, "GET /health", 11) == 0) {
            const char *body = "{\"status\":\"ok\"}";
            send_response(client_fd, HTTP_200_JSON, body);
        } else {
            send(client_fd, HTTP_404, strlen(HTTP_404), 0);
        }
        close(client_fd);
    }

    close(server_fd);
    return NULL;
}

/* ── Argument parsing ────────────────────────────────────────────── */
static void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            g_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            g_threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s --model <path> --port <num> --threads <num>\n",
                    argv[0]);
            exit(1);
        }
    }
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    parse_args(argc, argv);

    /* Configure OpenMP threads via kernel helper */
    q4_set_num_threads(g_threads);
    fprintf(stderr, "[engine] OMP max threads: %d\n", q4_get_max_threads());

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    pthread_t server_tid;
    pthread_create(&server_tid, NULL, http_server, NULL);

    /* Block until shutdown */
    pthread_join(server_tid, NULL);

    fprintf(stderr, "[engine] shut down.\n");
    return 0;
}