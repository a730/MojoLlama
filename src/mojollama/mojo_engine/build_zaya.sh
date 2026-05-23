#!/bin/bash
# Build ZAYA1-8B Q8_0 Mojo engine
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"
mojo build zaya_gen_q8.mojo --emit object -o zaya_gen_q8.o
gcc -o zaya_gen_q8 zaya_gen_q8.o \
    -L"$MOJO_LIB" \
    -lKGENCompilerRTShared \
    -lAsyncRTRuntimeGlobals \
    -lMSupportGlobals \
    -lAsyncRTMojoBindings \
    -lm -lpthread -ldl -lstdc++
echo "=== Done: zaya_gen_q8 ==="
echo "Run: LD_LIBRARY_PATH=\"$MOJO_LIB\" ./zaya_gen_q8"
