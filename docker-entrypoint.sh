#!/bin/bash
# =============================================================================
#  MojoLlama Docker Entrypoint
#
#  Handles:
#    - Model path resolution (env var MODEL_PATH, defaults to Qwen3-30B-A3B)
#    - Port configuration (env var PORT, defaults to 8080)
#    - MojoLlama engine selection (server_moe.py or server_batch_moe.py)
#    - Config file initialization (~/.mojollama/config.json)
#    - Signal forwarding for graceful shutdown
#    - Pass-through of all server arguments
# =============================================================================
set -e

# ── Color helpers ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[mojollama]${NC} $1"; }
warn() { echo -e "${YELLOW}[mojollama]${NC} $1"; }
err()  { echo -e "${RED}[mojollama]${NC} $1" >&2; }

# ── Environment variable defaults ────────────────────────────────────────
# MODEL_PATH: path to GGUF model file
#   Default: pre-downloaded Qwen3-30B-A3B in /models
: "${MODEL_PATH:=/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"

# PORT: HTTP server port
#   Default: 8080 (MojoLlama API)
: "${PORT:=8080}"

# ENGINE: which MojoLlama server to run
#   Options: server_moe, server_batch_moe
#   Default: server_batch_moe (higher throughput)
: "${ENGINE:=server_batch_moe}"

# OMP_NUM_THREADS: OpenMP thread count for MoE engine
: "${OMP_NUM_THREADS:=32}"
export OMP_NUM_THREADS

# ── Parse known arguments (for backward compat with old CLI style) ────────
MODEL_ARG=""
PORT_ARG=""
ENGINE_ARG=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ -n "$2" && "$2" != --* ]]; then
                MODEL_ARG="$2"
                shift 2
            else
                err "Missing value for --model"
                exit 1
            fi
            ;;
        --port)
            if [[ -n "$2" && "$2" != --* ]]; then
                PORT_ARG="$2"
                shift 2
            else
                err "Missing value for --port"
                exit 1
            fi
            ;;
        --engine)
            if [[ -n "$2" && "$2" != --* ]]; then
                ENGINE_ARG="$2"
                shift 2
            else
                err "Missing value for --engine"
                exit 1
            fi
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# CLI args override env vars
MODEL_PATH="${MODEL_ARG:-${MODEL_PATH}}"
PORT="${PORT_ARG:-${PORT}}"
ENGINE="${ENGINE_ARG:-${ENGINE}}"

# ── Model path resolution ────────────────────────────────────────────────
# Priority:
#   1. --model CLI argument or MODEL_PATH env var
#   2. First .gguf found in /models (runtime volume mount)
#   3. First .gguf found in /app (context models)
#   4. Fallback: default Qwen3 path

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
    # Auto-detect from volume mounts
    for search_dir in /models /app; do
        if [[ -d "$search_dir" ]]; then
            # shellcheck disable=SC2012
            models=("$search_dir"/*.gguf)
            if [[ -f "${models[0]}" ]]; then
                MODEL_PATH="${models[0]}"
                log "Auto-detected model: ${BLUE}${MODEL_PATH}${NC}"
                break
            fi
        fi
    done
fi

# If we still don't have a valid model, warn but don't fail — let the server error
if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -f "$MODEL_PATH" ]]; then
        warn "Model file not found: ${MODEL_PATH}"
        warn "Mount your GGUF models at ${YELLOW}/models${NC} or set MODEL_PATH env var."
        warn "Continuing anyway — server will report the error."
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
  "mojollama_engine": {
    "threads": 32,
    "optimal_concurrency": 4,
    "batch_size": 2048
  },
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

# ── Signal handling ──────────────────────────────────────────────────────
# Using `exec` below means Python receives signals directly as PID 1.
# No trap needed — Docker's default SIGTERM flows straight to the server.

# ── Select the engine script ────────────────────────────────────────────
case "${ENGINE}" in
    server_moe)
        ENGINE_SCRIPT="server_moe.py"
        ;;
    server_batch_moe)
        ENGINE_SCRIPT="server_batch_moe.py"
        ;;
    *)
        ENGINE_SCRIPT="server_batch_moe.py"
        warn "Unknown engine '${ENGINE}', defaulting to server_batch_moe"
        ;;
esac

ENGINE_PATH="/app/src/mojollama/${ENGINE_SCRIPT}"

if [[ ! -f "${ENGINE_PATH}" ]]; then
    err "Engine script not found: ${ENGINE_PATH}"
    err "Available engines: server_moe.py, server_batch_moe.py"
    exit 1
fi

# ── Start the server ─────────────────────────────────────────────────────
log "Starting MojoLlama engine..."
log "  Engine:     ${BLUE}${ENGINE_SCRIPT}${NC}"
log "  API port:   ${BLUE}${PORT}${NC}"
log "  Model:      ${BLUE}${MODEL_PATH}${NC}"
log "  Threads:    ${BLUE}${OMP_NUM_THREADS}${NC}"
log ""

# server_moe.py / server_batch_moe.py take positional args:
#   argv[1] = model_path
#   argv[2] = port
# Use exec to replace the shell with the Python process — this ensures
# signals are delivered directly to the server.
exec python3 "${ENGINE_PATH}" "${MODEL_PATH}" "${PORT}"
