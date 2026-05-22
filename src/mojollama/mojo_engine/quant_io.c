/* I/O bridge for MojoLlama Quantizer
   Wraps POSIX write() so Mojo can call it without @extern("write") stdlib conflict.
   WHY: Mojo 1.0.0b1 std/ffi declares @extern("write") and @extern("close") which
   conflicts with Mojo code declaring the same extern. Our C helper uses different
   function names ("qwrite", "qclose") that don't conflict.
*/
#include <unistd.h>

long qwrite(int fd, const void *buf, unsigned long n) {
    return write(fd, buf, n);
}

int qclose(int fd) {
    return close(fd);
}
