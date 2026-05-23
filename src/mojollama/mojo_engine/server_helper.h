/* server_helper.h — C FFI functions for HTTP server + JSON + tokenizer */
#ifndef SERVER_HELPER_H
#define SERVER_HELPER_H

#include <stdint.h>

/* ─── HTTP Server ─── */
/* Start server on port, returns fd or -1. Callback: fn(ctx, method, path, body) -> response_text */
typedef char* (*request_handler_t)(void *ctx, const char *method, const char *path, const char *body, int *resp_len);
int server_start(int port, request_handler_t handler, void *ctx);
void server_stop(void);

/* ─── JSON Parsing (minimal for OpenAI API) ─── */
/* Get string field from JSON: {"key":"value"} -> "value". Returns NULL if not found. */
char* json_get_string(const char *json, const char *key);
/* Get int field: {"key":123} -> 123. Returns default_val if not found. */
int json_get_int(const char *json, const char *key, int default_val);
/* Get double field: {"key":0.5} -> 0.5. Returns default_val if not found. */
double json_get_double(const char *json, const char *key, double default_val);
/* Get last message content from messages array */
char* json_get_last_message(const char *json);

/* ─── JSON Building ─── */
/* Build chat completion response JSON. Caller must free() the result. */
char* build_chat_response(const char *model, const char *content, 
                          int prompt_tokens, int completion_tokens);

/* ─── Tokenizer (GGUF) ─── */
typedef struct {
    int *ids;       /* token IDs array */
    int count;      /* number of tokens */
    int capacity;   /* allocated capacity */
} token_array;

/* Load tokenizer data from GGUF file */
int tokenizer_load(const char *gguf_path);

/* Encode text to token IDs (simple BPE via merging tokens) */
token_array* tokenizer_encode(const char *text);

/* Decode token IDs to text */
char* tokenizer_decode(const int *ids, int count);

/* Free token array */
void token_array_free(token_array *ta);

#endif
