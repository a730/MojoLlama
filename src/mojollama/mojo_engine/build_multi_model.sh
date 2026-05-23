#!/bin/bash
# Build multi-model API server
# Router: C binary (gemma4_router) — dispatches requests to model instances
# Models: Mojo binaries (gemma4_server_*) — one per architecture

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building Multi-Model API ==="

# Build C router
gcc -O2 -o gemma4_router server_helper.c -lpthread -lm
echo "  Router: gemma4_router ✓"

# Build model server (pure Mojo)
echo "  Building model server (this takes a moment)..."
MLIB="${MOJO_LIB:-/root/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib}"
mojo build gemma4_server.mojo -o gemma4_model_server 2>&1 | grep -v "warning:" | grep -v "Crashpad" || true
if [ -f gemma4_model_server ]; then
    echo "  Model server: gemma4_model_server ✓"
else
    echo "  Build failed, trying alternate approach..."
    mojo build gemma4_server.mojo --emit object -o gemma4_server.o
    gcc -o gemma4_model_server gemma4_server.o -L"$MLIB" \
        -lKGENCompilerRTShared -lAsyncRTRuntimeGlobals \
        -lMSupportGlobals -lAsyncRTMojoBindings -lm -lpthread -ldl -lstdc++
fi

echo ""
echo "=== Usage ==="
echo "1. Start model instances on different ports:"
echo "   ./gemma4_model_server 8081 32 /tmp/weights_e4b_final_transposed/"
echo "   ./gemma4_model_server 8082 32 /tmp/weights_e2b_final/"
echo ""
echo "2. Start the router:"
echo "   ./gemma4_router 8080"
echo ""
echo "3. Send requests to the router:"
echo '   curl http://localhost:8080/v1/chat/completions \'
echo '     -H "Content-Type: application/json" \'
echo '     -d "{\"model\":\"e4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"'
echo ""
echo "=== Available Models ==="
echo "  e4b: /tmp/weights_e4b_final_transposed/ (port 8081)"
echo "  e2b: /tmp/weights_e2b_final/ (port 8082)"
