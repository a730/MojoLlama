/* perf_helper.c — Performance helpers for MojoLlama (cache-aligned alloc, prefetch, thread pinning)
 * WHAT:  Wrappers for AVX2 optimizations Mojo can't express directly.
 * WHY:   Mojo 1.0.0b1 can't emit x86 intrinsics or set thread affinity.
 *         C handles the unavoidable I/O + platform operations.
 * WHEN:  2026-05-22
 *
 * Functions:
 *   _ml_memalign(sz, align)     — cache-aligned memory allocation (64-byte boundary)
 *   _ml_free(ptr)               — free aligned memory
 *   _ml_prefetch_L1(addr)       — software prefetch to L1 cache (T0 hint)
 *   _ml_prefetch_L2(addr)       — software prefetch to L2 cache (T1 hint)
 *   _ml_prefetch_NTA(addr)      — software prefetch non-temporal (skip cache)
 *   _ml_pin_to_cpu(cpu)         — pin current thread to specific CPU core
 *   _ml_pin_range(start, end)   — pin current thread to CPU range [start, end]
 */

#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <xmmintrin.h>  /* _mm_prefetch */

/* ── Aligned memory allocation (64-byte cache line) ── */
void* _ml_memalign(size_t sz, size_t align) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, sz) != 0) return NULL;
    return ptr;
}

void _ml_free(void *ptr) { free(ptr); }

/* ── Software prefetch (x86 SSE intrinsics) ── */
void _ml_prefetch_L1(const void *addr) {
    _mm_prefetch((const char*)addr, _MM_HINT_T0);
}
void _ml_prefetch_L2(const void *addr) {
    _mm_prefetch((const char*)addr, _MM_HINT_T1);
}
void _ml_prefetch_NTA(const void *addr) {
    _mm_prefetch((const char*)addr, _MM_HINT_NTA);
}

/* ── Thread/process pinning ── */
void _ml_pin_to_cpu(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void _ml_pin_range(int start, int end) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int c = start; c <= end; c++) CPU_SET(c, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

/* ── Process-level pinning: pin all threads to physical cores 0-31 ── */
void _ml_pin_process(void) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int c = 0; c < 32; c++) CPU_SET(c, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}
