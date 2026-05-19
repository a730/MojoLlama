# ── Makefile for mojo_engine ──────────────────────────────────────────
# Compiles mojo_engine.c, links against ../kernels/q4_kernel_omp.so,
# and produces the mojo_engine binary.
#
# Usage:
#   make            — build the engine
#   make clean      — remove build artefacts
#   make run        — build & run with default args
# ─────────────────────────────────────────────────────────────────────

CC      ?= gcc
CFLAGS  := -O3 -mavx2 -mfma -mf16c -fopenmp -Wall -Wextra
LDFLAGS := -L../kernels -l:q4_kernel_omp.so -Wl,-rpath,'$$ORIGIN/../kernels'
LDLIBS  := -lpthread -lm

# ── Targets ─────────────────────────────────────────────────────────

TARGET  := mojo_engine
SRC     := mojo_engine.c
OBJ     := $(SRC:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Convenience run (override with MODEL/PORT/THREADS env vars) ────

MODEL   ?= /dev/null
PORT    ?= 8080
THREADS ?= 4

run: $(TARGET)
	./$(TARGET) --model $(MODEL) --port $(PORT) --threads $(THREADS)

# ── Clean ───────────────────────────────────────────────────────────

clean:
	rm -f $(OBJ) $(TARGET)