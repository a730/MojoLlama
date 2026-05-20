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
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
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
