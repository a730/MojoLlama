#!/bin/bash
set -e
MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"
mojo build tinyllama_gen_q8.mojo --emit object -o tinyllama_gen_q8.o
gcc -o tinyllama_gen_q8 tinyllama_gen_q8.o \
    -L"$MOJO_LIB" \
    -lKGENCompilerRTShared \
    -lAsyncRTRuntimeGlobals \
    -lMSupportGlobals \
    -lAsyncRTMojoBindings \
    -lm -lpthread -ldl -lstdc++
echo "=== Done: tinyllama_gen_q8 ==="
echo "Run: LD_LIBRARY_PATH=\"$MOJO_LIB\" ./tinyllama_gen_q8"
