#!/bin/bash
# =============================================================================
#  MojoLlama Docker Entrypoint
#
#  Handles:
#    - Model path resolution
#    - Port configuration
#    - Config file initialization (~/.mojollama/config.json)
#    - AutoBackend detection (CPU → TurboEngine, GPU → llama.cpp)
#    - Optional benchmark mode (MOJOLLAMA_BENCHMARK=true)
# =============================================================================
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[mojollama]${NC} $1"; }
warn() { echo -e "${YELLOW}[mojollama]${NC} $1"; }
err()  { echo -e "${RED}[mojollama]${NC} $1" >&2; }

: "${MODEL_PATH:=}"
: "${PORT:=8080}"
: "${OMP_NUM_THREADS:=32}"
: "${MOJOLLAMA_BENCHMARK:=false}"

export OMP_NUM_THREADS
export MOJOLLAMA_BENCHMARK

# ── Source oneAPI environment (SYCL backend) ─────────────────────────────
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    log "Sourcing Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
fi

# ── Model path resolution ────────────────────────────────────────────────
if [[ -n "$HF_REPO" ]]; then
    log "HF model download requested: ${BLUE}${HF_REPO}${NC}"
    # Server will handle download — don't auto-detect a local .gguf
elif [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
    for search_dir in /models /app; do
        if [[ -d "$search_dir" ]]; then
            models=("$search_dir"/*.gguf)
            if [[ -f "${models[0]}" ]]; then
                MODEL_PATH="${models[0]}"
                log "Auto-detected model: ${BLUE}${MODEL_PATH}${NC}"
                break
            fi
        fi
    done
fi

if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -f "$MODEL_PATH" ]]; then
        warn "Model file not found: ${MODEL_PATH}"
        warn "Mount GGUF models at ${YELLOW}/models${NC} or set MODEL_PATH env var."
    else
        log "Using model: ${BLUE}${MODEL_PATH}${NC}"
        # Auto-detect tokenizer directory next to the model
        MODEL_DIR="$(dirname "$MODEL_PATH")"
        MODEL_BASE="$(basename "$MODEL_PATH" .gguf)"
        if [[ -z "$TOKENIZER_PATH" ]]; then
            for tok_dir in "${MODEL_DIR}/tokenizer" "${MODEL_DIR}/${MODEL_BASE}-tokenizer" "/models/tokenizer" "/models/zaya-tokenizer" "/models/tinyllama-tokenizer"; do
                if [[ -d "$tok_dir" && -f "${tok_dir}/tokenizer.json" ]]; then
                    log "Auto-detected tokenizer: ${BLUE}${tok_dir}${NC}"
                    export TOKENIZER_PATH="$tok_dir"
                    break
                fi
            done
            if [[ -z "$TOKENIZER_PATH" ]]; then
                log "No tokenizer directory found, will auto-download from HuggingFace"
            fi
        fi
    fi
    export MODEL_PATH
fi

# ── Config initialization ────────────────────────────────────────────────
CONFIG_DIR="${HOME}/.mojollama"
CONFIG_FILE="${CONFIG_DIR}/config.json"

mkdir -p "${CONFIG_DIR}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    log "Creating default config at ${BLUE}${CONFIG_FILE}${NC}"
    cat > "${CONFIG_FILE}" <<'EOF'
{
  "llama_server": {
    "threads": 4,
    "threads_batch": 2,
    "batch_size": 2048,
    "ubatch_size": 512,
    "n_parallel": 4,
    "mlock": true,
    "cont_batching": true
  }
}
EOF
fi

# ── Start the server ─────────────────────────────────────────────────────
log "Starting MojoLlama server..."
log "  API port:   ${BLUE}${PORT}${NC}"
log "  Model:      ${BLUE}${MODEL_PATH:-<auto-detect>}${NC}"
log "  Threads:    ${BLUE}${OMP_NUM_THREADS}${NC}"
log "  Benchmark:  ${BLUE}${MOJOLLAMA_BENCHMARK}${NC}"

exec python3 -m mojollama.server
