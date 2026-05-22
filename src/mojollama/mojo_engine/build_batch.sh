#!/bin/bash
set -e
MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"

echo "=== Step 1: Mojo compile to object ==="
mojo build tinyllama_gen_batch.mojo --emit object -o tinyllama_gen_batch.o

echo "=== Step 2: Link with GCC ==="
gcc -o tinyllama_gen_batch tinyllama_gen_batch.o \
    -L"$MOJO_LIB" \
    -lKGENCompilerRTShared \
    -lAsyncRTRuntimeGlobals \
    -lMSupportGlobals \
    -lAsyncRTMojoBindings \
    -lm -lpthread -ldl -lstdc++

echo "=== Done: tinyllama_gen_batch ==="
echo "Run: LD_LIBRARY_PATH=\"$MOJO_LIB\" ./tinyllama_gen_batch"
