/* gemma4_router_main.c — Multi-model API router entry point
   Routes requests to model instances based on model name.
   
   Usage: ./gemma4_router <port> [model1=port1:weights1] [model2=port2:weights2] ...
   
   Example:
     ./gemma4_router 8080 e4b=8081:/tmp/weights_e4b_final_transposed/ e2b=8082:/tmp/weights_e2b_final/
   
   Each model instance's binary is auto-detected from the current directory.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

/* Declare API functions from server_helper.c */
int register_model(const char *name, const char *binary, const char *weights, int port);
int start_server(int port);
void stop_server(void);

static volatile int keep_running = 1;
void handle_signal(int sig) {
    (void)sig;
    keep_running = 0;
    stop_server();
    printf("\nServer shutting down...\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <port> [model=port:weights ...]\n", argv[0]);
        fprintf(stderr, "  Default models:\n");
        fprintf(stderr, "    e4b -> port 8081, /tmp/weights_e4b_final_transposed/\n");
        fprintf(stderr, "    e2b -> port 8082, /tmp/weights_e2b_final/\n");
        return 1;
    }
    
    int port = atoi(argv[1]);
    const char *binary = "./gemma4_model_server";
    
    /* Register models from args or use defaults */
    if (argc >= 4) {
        /* Parse model=port:weights arguments */
        for (int i = 2; i < argc; i++) {
            char name[64], *p = argv[i];
            int mport = 0;
            char weights[1024] = {0};
            
            /* Parse "name=port:weights" */
            char *eq = strchr(p, '=');
            if (!eq) { fprintf(stderr, "Bad model spec: %s (expected name=port:weights)\n", p); continue; }
            int nlen = eq - p;
            if (nlen > 63) nlen = 63;
            strncpy(name, p, nlen); name[nlen] = 0;
            
            char *colon = strchr(eq + 1, ':');
            if (colon) {
                mport = atoi(eq + 1);
                strncpy(weights, colon + 1, sizeof(weights) - 1);
            } else {
                mport = atoi(eq + 1);
                snprintf(weights, sizeof(weights), "/tmp/weights_%s_final/", name);
            }
            
            register_model(name, binary, weights, mport);
            printf("  Model '%s': port %d, weights %s\n", name, mport, weights);
        }
    } else {
        /* Default models */
        register_model("e4b", binary, "/tmp/weights_e4b_final_transposed/", 8081);
        register_model("e2b", binary, "/tmp/weights_e2b_final/", 8082);
        printf("  Default models registered:\n");
        printf("    e4b -> port 8081\n");
        printf("    e2b -> port 8082\n");
    }
    
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    
    printf("Starting router on port %d...\n", port);
    if (start_server(port) < 0) {
        fprintf(stderr, "Failed to start server on port %d\n", port);
        return 1;
    }
    
    printf("Router ready at http://0.0.0.0:%d/v1/chat/completions\n", port);
    printf("Endpoints: /health, /v1/models, /v1/chat/completions\n");
    printf("Press Ctrl+C to stop.\n");
    
    /* Keep running */
    while (keep_running) {
        sleep(1);
    }
    
    return 0;
}
