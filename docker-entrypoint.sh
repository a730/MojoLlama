#!/bin/bash
# =============================================================================
#  MojoLlama Docker Entrypoint
#
#  Handles:
#    - Model path resolution (auto-detect GGUF in /models or /app)
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

# ── Parse known arguments ────────────────────────────────────────────────
MODEL_ARG=""
PORT_ARG=""
LLAMA_PORT_ARG=""
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
        --llama-port)
            if [[ -n "$2" && "$2" != --* ]]; then
                LLAMA_PORT_ARG="$2"
                shift 2
            else
                err "Missing value for --llama-port"
                exit 1
            fi
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Environment variable defaults ────────────────────────────────────────
export PORT="${PORT_ARG:-${PORT:-8080}}"
export LLAMA_PORT="${LLAMA_PORT_ARG:-${LLAMA_PORT:-8081}}"

# ── Model path resolution ────────────────────────────────────────────────
# Priority:
#   1. --model CLI argument
#   2. MODEL_PATH env var
#   3. First .gguf found in /models (runtime volume mount)
#   4. First .gguf found in /app (context models)
#   5. Fallback: let the server try its default (will print a useful error)

MODEL_PATH="${MODEL_ARG:-${MODEL_PATH:-}}"

if [[ -z "$MODEL_PATH" ]]; then
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

# If found, inject --model into the argument list
if [[ -n "$MODEL_PATH" ]]; then
    log "Using model: ${BLUE}${MODEL_PATH}${NC}"
    # Verify it exists
    if [[ ! -f "$MODEL_PATH" ]]; then
        err "Model file not found: ${MODEL_PATH}"
        err "Mount your GGUF models at ${YELLOW}/models${NC} or pass --model explicitly."
        exit 1
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

# ── Signal handling ──────────────────────────────────────────────────────
# Using `exec` below means Python receives signals directly as PID 1.
# No trap needed — Docker's default SIGTERM flows straight to the server.

# ── Build the server command ─────────────────────────────────────────────
# Construct the final argument list:
#   If we resolved a model, prepend --model so user --model doesn't conflict
SERVER_ARGS=()

if [[ -n "$MODEL_PATH" ]]; then
    SERVER_ARGS+=("--model" "$MODEL_PATH")
fi

SERVER_ARGS+=("--port" "$PORT")
SERVER_ARGS+=("--llama-port" "$LLAMA_PORT")

# Append any remaining passthrough arguments
SERVER_ARGS+=("${PASSTHROUGH_ARGS[@]}")

# ── Start the server ─────────────────────────────────────────────────────
log "Starting MojoLlama server..."
log "  API port:   ${BLUE}${PORT}${NC}"
log "  Backend:    ${BLUE}llama.cpp on port ${LLAMA_PORT}${NC}"
if [[ -n "$MODEL_PATH" ]]; then
    log "  Model:      ${BLUE}${MODEL_PATH}${NC}"
fi
log ""

# Use exec to replace the shell with the Python process — this ensures
# signals are delivered directly to the server.
exec python3 -m mojollama.server "${SERVER_ARGS[@]}"
