# =============================================================================
#  MojoLlama — Production Dockerfile
#  Multi-stage build:
#    Stage 1 (builder):  builds llama-server from source (llama.cpp)
#    Stage 2 (runtime):  Python 3.11-slim + mojollama code + llama-server binary
# =============================================================================
# syntax=docker/dockerfile:1

# ── Stage 1: Build llama.cpp from source ──────────────────────────────────
FROM ubuntu:22.04 AS builder

LABEL stage="builder"

# Build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp (shallow clone for speed)
ARG LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMACPP_BRANCH=master
RUN git clone --depth 1 --branch "${LLAMACPP_BRANCH}" "${LLAMACPP_REPO}" /tmp/llama.cpp

# Build llama-server with release flags
WORKDIR /tmp/llama.cpp
RUN mkdir -p build && \
    cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        -DLLAMA_SERVER_VERBOSE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CCACHE=OFF \
        && cmake --build . --target llama-server -- -j"$(nproc)"

# Also keep the HF→GGUF converter script for the export feature
# (it's a standalone Python script with no build needed)
RUN chmod +x /tmp/llama.cpp/convert_hf_to_gguf.py

# ── Stage 2: Runtime image ───────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="MojoLlama"
LABEL org.opencontainers.image.description="High-throughput LLM inference server with GGUF support"
LABEL org.opencontainers.image.source="https://git.bamse.cloud/a730/MojoLlama"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="MojoLlama"
# OCI standard labels
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.ref.name="mojollama-server"

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# ── System dependencies ──────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# ── Create non-root user for security ────────────────────────────────────
RUN groupadd -r mojollama && \
    useradd -r -g mojollama \
        -d /home/mojollama \
        -m \
        -s /sbin/nologin \
        mojollama

# ── Copy artifacts from builder ──────────────────────────────────────────
# IMPORTANT: paths match hardcoded locations in server.py and backends.py:
#   LLAMA_SERVER_PATH = "/tmp/llama.cpp/build/bin/llama-server"
#   CONVERTER_PATH     = "/tmp/llama.cpp/convert_hf_to_gguf.py"
COPY --from=builder --chown=root:root \
    /tmp/llama.cpp/build/bin/llama-server \
    /tmp/llama.cpp/build/bin/llama-server

COPY --from=builder --chown=root:root \
    /tmp/llama.cpp/convert_hf_to_gguf.py \
    /tmp/llama.cpp/convert_hf_to_gguf.py

RUN chmod 755 /tmp/llama.cpp/build/bin/llama-server /tmp/llama.cpp/convert_hf_to_gguf.py

# ── Install Python dependencies ─────────────────────────────────────────
# gguf:       reading/converting GGUF model files
# numpy:      fallback backend and kernel ops
# requests:   optional HTTP client (not used by server itself, but useful)
RUN pip install --no-cache-dir \
        gguf \
        numpy \
        requests

# ── Copy application code ────────────────────────────────────────────────
WORKDIR /app

# Python package
COPY src/mojollama/ /app/src/mojollama/
# Static web UI files
COPY www/ /app/www/
# Entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod 755 /app/docker-entrypoint.sh

# ── Create mount points and fix permissions ──────────────────────────────
# /models — volume mount for GGUF model files
# ~/.mojollama — config directory (auto-created by entrypoint)
RUN mkdir -p /models /home/mojollama/.mojollama && \
    chown -R mojollama:mojollama \
        /app \
        /models \
        /home/mojollama/.mojollama

# ── Health check ─────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

# ── Ports ────────────────────────────────────────────────────────────────
# 8080 — MojoLlama API server
# 8081 — llama.cpp backend (internal, exposed for debugging/tuning)
EXPOSE 8080 8081

# ── Runtime user ─────────────────────────────────────────────────────────
USER mojollama

# ── Default command (entrypoint handles model resolution) ────────────────
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--port", "8080"]
