# =============================================================================
#  MojoLlama — Universal Dockerfile
#  Multi-stage build:
#    Stage 1 (builder):  builds llama-server (llama.cpp) + AVX2 C kernels
#    Stage 2 (runtime):  Python 3.11 + mojollama code + all binaries
# =============================================================================
# syntax=docker/dockerfile:1

ARG GPU_BACKEND=cpu

# ── Stage 1: Build llama.cpp + C kernels ──────────────────────────────────
FROM ubuntu:22.04 AS builder

LABEL stage="builder"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates gcc g++ libomp-dev \
        && rm -rf /var/lib/apt/lists/*

# ── Build llama-server ──────────────────────────────────────────────────
ARG LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMACPP_BRANCH=master
RUN git clone --depth 1 --branch "${LLAMACPP_BRANCH}" "${LLAMACPP_REPO}" /tmp/llama.cpp
WORKDIR /tmp/llama.cpp
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF -DLLAMA_SERVER_VERBOSE=OFF \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_CCACHE=OFF \
        && cmake --build . --target llama-server -- -j"$(nproc)"
RUN chmod +x /tmp/llama.cpp/convert_hf_to_gguf.py

# ── Build TurboEngine AVX2 C kernels ────────────────────────────────────
COPY src/mojollama/kernels/ /build/kernels/
COPY src/mojollama/kernels/legacy-c/turbo_kernels_v2.c /build/kernels/
WORKDIR /build/kernels
RUN gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o quant_kernels_omp.so quant_kernels_omp.c -lm && \
    gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o cengine_batch_instr.so cengine_batch_instr.c -lm && \
    gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o simd_ops.so simd_ops.c -lm && \
    gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o gqa_attention.so gqa_attention.c -lm && \
    gcc -O3 -mavx2 -mfma -mf16c -shared -fPIC \
    -o turbo_kernels_v2.so turbo_kernels_v2.c -lm

# ── Stage 2: Runtime image ───────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="MojoLlama"
LABEL org.opencontainers.image.description="Universal LLM inference server — TurboEngine CPU + llama.cpp GPU backends"
LABEL org.opencontainers.image.source="https://git.bamse.cloud/a730/MojoLlama"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates libomp-dev libgomp1 \
        && rm -rf /var/lib/apt/lists/*

RUN groupadd -r mojollama && \
    useradd -r -g mojollama -d /home/mojollama -m -s /sbin/nologin mojollama

# ── Copy llama-server binary ────────────────────────────────────────────
COPY --from=builder --chown=root:root \
    /tmp/llama.cpp/build/bin/llama-server \
    /tmp/llama.cpp/build/bin/llama-server
COPY --from=builder --chown=root:root \
    /tmp/llama.cpp/convert_hf_to_gguf.py \
    /tmp/llama.cpp/convert_hf_to_gguf.py
RUN chmod 755 /tmp/llama.cpp/build/bin/llama-server /tmp/llama.cpp/convert_hf_to_gguf.py

# ── Copy MojoLlama source + Python files first ─────────────────────────
WORKDIR /app
COPY src/mojollama/ /app/src/mojollama/

# ── Overwrite with freshly compiled TurboEngine AVX2 kernels (.so files) ──
# Must come AFTER COPY src/mojollama/ so that stale checked-in .so files
# in the source tree are overwritten by the freshly built ones.
COPY --from=builder --chown=root:root \
    /build/kernels/quant_kernels_omp.so \
    /build/kernels/cengine_batch_instr.so \
    /build/kernels/simd_ops.so \
    /build/kernels/gqa_attention.so \
    /build/kernels/turbo_kernels_v2.so \
    /app/src/mojollama/kernels/

# ── Python deps ─────────────────────────────────────────────────────────
RUN pip install --no-cache-dir gguf numpy transformers requests

# ── Pre-download ZAYA1-8B model + tokenizer for testing ──────────────────
RUN mkdir -p /models && \
    curl -fSL --retry 3 --retry-delay 5 --progress-bar \
        -o /models/ZAYA1-8B-Q4_K_M.gguf \
        "https://huggingface.co/Abiray/ZAYA1-8B-GGUF/resolve/main/ZAYA1-8B-Q4_K_M.gguf" \
    || echo "[WARN] ZAYA GGUF download failed"
# Pre-download ZAYA tokenizer so server works offline
RUN mkdir -p /models/zaya-tokenizer && \
    for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do \
        curl -fSL --retry 3 --retry-delay 5 \
            -o "/models/zaya-tokenizer/$f" \
            "https://huggingface.co/Zyphra/ZAYA1-8B/resolve/main/$f" \
            || echo "[WARN] ZAYA tokenizer $f download failed"; \
    done
COPY www/ /app/www/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod 755 /app/docker-entrypoint.sh

RUN mkdir -p /models /home/mojollama/.mojollama && \
    chown -R mojollama:mojollama /app /models /home/mojollama/.mojollama

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

EXPOSE 8080 8081

USER mojollama

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
