# ============================================================
# MojoLlama Dockerfile — Multi-stage build
# ============================================================
# Produces a single image with:
#   - MojoLlama engine (Python + C kernels + Mojo/MAX)
#   - llama.cpp as fallback backend
#   - Studio web UI (Flask)
#   - Quantization engine
#   - OpenAI-compatible API server
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM debian:bookworm-20250408 AS builder

LABEL org.opencontainers.image.title="MojoLlama"
LABEL org.opencontainers.image.description="High-throughput CPU LLM inference engine — GGUF native, MoE-optimized"
LABEL org.opencontainers.image.version="0.5.0"
LABEL org.opencontainers.image.licenses="MIT"

SHELL ["/bin/bash", "-c"]

# ── Build dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    ca-certificates \
    pkg-config \
    libomp-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ── Install MAX / Mojo SDK ──
RUN curl -fsSL https://get.modular.com | bash -s -- --install-version 26.3.0 \
    && pip install max-mojo-libs==26.3.0 max-mojo-mogg-libs==26.3.0 \
    && rm -rf ~/.cache/pip

# ── Build llama.cpp (fallback backend) ──
RUN git clone --depth=1 --branch b4729 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp \
    && cd /tmp/llama.cpp \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_NATIVE=OFF \
        -DLLAMA_AVX2=ON \
        -DLLAMA_FMA=ON \
        -DLLAMA_F16C=ON \
        -DLLAMA_OPENMP=ON \
        -DLLAMA_CURL=OFF \
    && cmake --build build --config Release -j$(nproc) --target llama-cli llama-server llama-bench \
    && strip build/bin/llama-cli build/bin/llama-server build/bin/llama-bench \
    && cp build/bin/llama-* /usr/local/bin/ \
    && rm -rf /tmp/llama.cpp

# ── Build MojoLlama C kernels ──
COPY kernels/ /build/kernels/
WORKDIR /build/kernels
RUN gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o quant_kernels_omp.so quant_kernels_omp.c -lm \
    && gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o cengine_batch_instr.so cengine_batch_instr.c -lm \
    && gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o simd_ops.so simd_ops.c -lm \
    && gcc -O3 -mavx2 -mfma -mf16c -fopenmp -shared -fPIC \
    -o gqa_attention.so gqa_attention.c -lm

# ── Build Mojo engine ──
COPY engine/ /build/engine/
WORKDIR /build/engine
RUN make clean && make && strip mojo_engine && cp mojo_engine /usr/local/bin/mojo_engine

# ── Build Mojo modules ──
COPY model/ /build/model/
COPY gguf/ /build/gguf/
COPY graph/ /build/graph/
COPY inference/ /build/inference/
COPY runtime/ /build/runtime/
COPY kernels/*.mojo /build/kernels/
WORKDIR /build
RUN max build --release -o /usr/local/lib/mojollama/mojo_engine_infer mojollama_infer.mojo \
    && max build --release -o /usr/local/lib/mojollama/mojo_gguf_reader gguf/reader.mojo \
    || echo "Mojo build skipped (optional)"

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM debian:bookworm-20250408-slim AS runtime

SHELL ["/bin/bash", "-c"]

# ── Runtime dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libomp-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ──
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# ── Install Python dependencies ──
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy<2 \
        gguf \
        flask \
        fastapi \
        uvicorn \
        pydantic \
        sse-starlette \
        huggingface_hub \
        jinja2

# ── Copy built artifacts from builder ──
COPY --from=builder /build/kernels/*.so /opt/mojollama/kernels/
COPY --from=builder /usr/local/bin/llama-cli /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-server /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-bench /usr/local/bin/
COPY --from=builder /usr/local/bin/mojo_engine /usr/local/bin/
COPY --from=builder /usr/local/lib/mojollama/ /usr/local/lib/mojollama/

# ── Copy MojoLlama source ──
COPY . /opt/mojollama/src/
WORKDIR /opt/mojollama

# Create symlinks for C kernels so engine can find them
RUN ln -sf /opt/mojollama/kernels /opt/mojollama/src/mojollama/kernels && \
    mkdir -p /opt/mojollama/models

# Install mojollama as a pip package
RUN pip install --no-cache-dir -e /opt/mojollama/src/

# ── Wrapper: auto-tune on first model load ──
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ── Expose ports ──
EXPOSE 8080 8081

# ── Entry point ──
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--help"]
