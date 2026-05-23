/* model_helper.c — Model instance HTTP server + arch.json reader for Mojo FFI */
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

#define MAX_BODY 65536
#define MAX_QUEUE 64

typedef struct { int id; char body[MAX_BODY]; int client_fd; int active; } Request;
static Request queue[MAX_QUEUE];
static int qhead = 0, qtail = 0, next_id = 1;
static pthread_mutex_t qmutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t qcond = PTHREAD_COND_INITIALIZER;
static int srv_fd = -1;
static volatile int running = 0;
static pthread_t srv_thread;

static void send_resp(int fd, const char *body) {
    char h[4096]; int n = snprintf(h, sizeof(h),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", strlen(body));
    write(fd, h, n); write(fd, body, strlen(body)); close(fd);
}

static void handle(int fd) {
    char buf[MAX_BODY]; int n = read(fd, buf, sizeof(buf)-1); if (n<=0) { close(fd); return; }
    buf[n]=0;
    char method[64]={0},path[1024]={0}; sscanf(buf,"%63s %1023s",method,path);
    if (strcmp(method,"OPTIONS")==0) { send_resp(fd,""); return; }
    if (strcmp(path,"/health")==0||strcmp(path,"/")==0) { send_resp(fd,"{\"status\":\"ok\"}"); return; }
    if (strcmp(path,"/v1/chat/completions")==0&&strcmp(method,"POST")==0) {
        char *body = strstr(buf,"\r\n\r\n"); if (!body) { close(fd); return; }
        body += 4;
        pthread_mutex_lock(&qmutex);
        int i = qtail % MAX_QUEUE;
        queue[i].id = next_id++; queue[i].client_fd = fd;
        strncpy(queue[i].body, body, MAX_BODY-1); queue[i].active = 1;
        qtail++; pthread_cond_signal(&qcond);
        pthread_mutex_unlock(&qmutex);
        return;
    }
    send_resp(fd,"{\"error\":\"not_found\"}");
}

static void *loop(void *arg) {
    (void)arg; struct sockaddr_in a; socklen_t al = sizeof(a);
    while (running) {
        struct pollfd p = {srv_fd, POLLIN, 0};
        if (poll(&p,1,500)<=0) continue;
        int c = accept(srv_fd,(struct sockaddr*)&a,&al);
        if (c>=0) handle(c);
    }
    return NULL;
}

int start_model_server(int port) {
    if (running) return -1;
    srv_fd = socket(AF_INET,SOCK_STREAM,0);
    if (srv_fd<0) return -1;
    int opt=1; setsockopt(srv_fd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in a={0}; a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(port);
    if (bind(srv_fd,(struct sockaddr*)&a,sizeof(a))<0) { close(srv_fd); return -1; }
    if (listen(srv_fd,5)<0) { close(srv_fd); return -1; }
    running=1; pthread_create(&srv_thread,NULL,loop,NULL); pthread_detach(srv_thread);
    return 0;
}

char* poll_request(void) {
    pthread_mutex_lock(&qmutex);
    if (qhead>=qtail||!queue[qhead%MAX_QUEUE].active) { pthread_mutex_unlock(&qmutex); return NULL; }
    int i = qhead % MAX_QUEUE;
    char *p = strdup(queue[i].body);
    pthread_mutex_unlock(&qmutex);
    return p;
}

void send_response(const char *resp) {
    pthread_mutex_lock(&qmutex);
    if (qhead<qtail) {
        int i = qhead % MAX_QUEUE;
        if (queue[i].active) { send_resp(queue[i].client_fd, resp); queue[i].active=0; }
        qhead++;
    }
    pthread_mutex_unlock(&qmutex);
}

char* build_openai_response(const char *content, int pt, int ct, const char *model) {
    char *esc = malloc(strlen(content)*2+1); if(!esc) return NULL;
    int j=0;
    for(int i=0;content[i];i++) {
        char c=content[i];
        if(c=='"'||c=='\\'){esc[j++]='\\';esc[j++]=c;}
        else if(c=='\n'){esc[j++]='\\';esc[j++]='n';}
        else if(c=='\t'){esc[j++]='\\';esc[j++]='t';}
        else esc[j++]=c;
    }
    esc[j]=0;
    char *r; asprintf(&r,
        "{\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
        "\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
        rand(),(long)time(NULL),model?model:"gemma-4",esc,pt,ct,pt+ct);
    free(esc); return r;
}

void stop_model_server(void) { running=0; if(srv_fd>=0){close(srv_fd);srv_fd=-1;} }

/* ─── Read arch.json integer value ─── */
int read_arch_int(const char *dir_path, const char *key) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/arch.json", dir_path);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    char buf[4096];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = 0;
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *p = strstr(buf, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
    int val = 0, neg = 0;
    if (*p == '-') { neg = 1; p++; }
    while (*p && isdigit(*p)) { val = val * 10 + (*p - '0'); p++; }
    return neg ? -val : val;
}

/* Wrapper for write() to avoid Mojo @extern conflict */
int file_write(int fd, const void *buf, int count) {
    return (int)write(fd, buf, (size_t)count);
}

int mojo_write_tokens(int fd, const void *buf, int count) {
    /* Debug: verify fd */
    if (fd == 2) { write(2, "stderr\n", 7); }
    return (int)write(fd, buf, (size_t)count);
}
