#!/bin/bash
set -e
MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"
SRC="gptoss_gen_q8.mojo"
OBJ="${SRC%.mojo}.o"
BIN="${SRC%.mojo}"
echo "=== Building $BIN ==="
mojo build "$SRC" --emit object -o "$OBJ" 2>&1
gcc -o "$BIN" "$OBJ" \
    -L"$MOJO_LIB" \
    -lKGENCompilerRTShared \
    -lAsyncRTRuntimeGlobals \
    -lMSupportGlobals \
    -lAsyncRTMojoBindings \
    -lm -lpthread -ldl -lstdc++ 2>&1
echo "=== Done ==="
ls -lh "$BIN"
echo "Run: LD_LIBRARY_PATH=\"$MOJO_LIB\" OMP_PLACES=cores OMP_PROC_BIND=close ./$BIN [nw]"
