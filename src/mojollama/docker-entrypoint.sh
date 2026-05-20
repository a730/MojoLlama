#!/bin/bash
# ============================================================
# MojoLlama Docker Entrypoint
# ============================================================
# Features:
#   - Auto-tunes MojoLlama engine on first model load
#   - Caches per-model tuning in ~/.mojollama/config.json
#   - Sets OMP_NUM_THREADS from tuned config
#   - Falls back to sensible defaults
# ============================================================

set -e

# ── Locate model file ──────────────────────────────────
MODEL=""
for arg in "$@"; do
    case "$arg" in
        -m|--model)
            shift
            MODEL="$1"
            ;;
        *.gguf)
            if [ -z "$MODEL" ] && [ -f "$arg" ]; then
                MODEL="$arg"
            fi
            ;;
    esac
done

# ── Auto-tune on first model load ──────────────────────
# Skip if AUTO_TUNE=false or the model doesn't exist
if [ "${AUTO_TUNE:-true}" = "true" ] && [ -n "$MODEL" ] && [ -f "$MODEL" ]; then
    MODEL_HASH=$(md5sum "$MODEL" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
    TUNE_CACHE="$HOME/.mojollama/tune_cache_${MODEL_HASH}.done"

    if [ ! -f "$TUNE_CACHE" ]; then
        echo "=================================================="
        echo "  MojoLlama AutoTune"
        echo "  Model: $(basename "$MODEL")"
        echo "  Running fast engine + concurrency sweep..."
        echo "=================================================="
        python3 -m mojollama autotune --model "$MODEL" --quick --mojollama 2>&1
        mkdir -p "$HOME/.mojollama"
        touch "$TUNE_CACHE"
        echo "  ✓ Tuning cached. Delete $TUNE_CACHE to re-tune."
        echo ""
    fi

    # ── Apply tuned OMP threads ──────────────────────────
    if [ -f "$HOME/.mojollama/config.json" ]; then
        TUNED_THREADS=$(python3 -c "
import json
cfg = json.load(open('$HOME/.mojollama/config.json'))
print(cfg.get('mojollama_engine', {}).get('omp_threads', ''))
" 2>/dev/null || echo "")
        if [ -n "$TUNED_THREADS" ]; then
            export OMP_NUM_THREADS="$TUNED_THREADS"
            echo "  OMP_NUM_THREADS=$TUNED_THREADS (auto-tuned)"
        fi
    fi
fi

# ── Fallback OMP threads ──────────────────────────────
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 32)}"

# ── Launch mojollama ──────────────────────────────────
exec python3 -m mojollama "$@"
