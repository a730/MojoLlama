#!/usr/bin/env bash
# ── test_health.sh — Verify mojo_engine starts and responds to /health ──
# Usage: ./test_health.sh [port]
#   port defaults to 18080 (alt port to avoid conflicts)
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

PORT="${1:-18080}"
ENGINE_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_BIN="${ENGINE_DIR}/mojo_engine"
TIMEOUT=10

# ── 1. Build ────────────────────────────────────────────────────────────
echo "[test] Building mojo_engine ..."
make -C "$ENGINE_DIR" clean && make -C "$ENGINE_DIR"
echo "[test] Build succeeded."

# ── 2. Start engine in background ──────────────────────────────────────
echo "[test] Starting mojo_engine on port ${PORT} ..."
"$ENGINE_BIN" --model /dev/null --port "$PORT" --threads 2 &
ENGINE_PID=$!

# Ensure we kill the engine on exit (even on failure)
cleanup() {
    echo "[test] Stopping engine (PID ${ENGINE_PID}) ..."
    kill "$ENGINE_PID" 2>/dev/null || true
    wait "$ENGINE_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── 3. Wait for readiness (poll /health with curl) ─────────────────────
echo "[test] Waiting up to ${TIMEOUT}s for /health ..."
elapsed=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
    if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[test] /health responded, engine is ready."
        break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "[FAIL] Engine did not respond within ${TIMEOUT}s."
    exit 1
fi

# ── 4. Verify response body ────────────────────────────────────────────
RESPONSE=$(curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health")
echo "[test] Response: ${RESPONSE}"

if echo "$RESPONSE" | grep -q '"status"'; then
    echo "[PASS] /health returned expected JSON."
else
    echo "[FAIL] /health response missing expected key."
    exit 1
fi

echo "[test] All checks passed."
exit 0