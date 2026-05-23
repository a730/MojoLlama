#!/bin/bash
set -e
MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"
echo "=== Building gemma4_gen_q8 ==="
mojo build gemma4_gen_q8.mojo --emit object -o gemma4_gen_q8.o
gcc -o gemma4_gen_q8 gemma4_gen_q8.o \
    -L"$MOJO_LIB" \
    -lKGENCompilerRTShared \
    -lAsyncRTRuntimeGlobals \
    -lMSupportGlobals \
    -lAsyncRTMojoBindings \
    -lm -lpthread -ldl -lstdc++
echo "=== Done ==="
echo "Run: LD_LIBRARY_PATH=\"$MOJO_LIB\" ./gemma4_gen_q8"
