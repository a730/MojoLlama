#!/bin/bash
set -e
MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"
SRC="tinyllama_gen_batch.mojo"

for B in 2 4 8; do
    echo ""
    echo "=============================="
    echo " Building B=$B"
    echo "=============================="
    sed -i "s/comptime B: Int = [0-9]*/comptime B: Int = $B/" "$SRC"
    mojo build "$SRC" --emit object -o "tinyllama_gen_batch.o"
    gcc -o "tinyllama_gen_batch_${B}" "tinyllama_gen_batch.o" \
        -L"$MOJO_LIB" \
        -lKGENCompilerRTShared \
        -lAsyncRTRuntimeGlobals \
        -lMSupportGlobals \
        -lAsyncRTMojoBindings \
        -lm -lpthread -ldl -lstdc++

    echo ""
    echo "--- Running B=$B (3 runs) ---"
    for run in 1 2 3; do
        LD_LIBRARY_PATH="$MOJO_LIB" "./tinyllama_gen_batch_${B}" 2>&1 | grep "B="
    done
done

# Restore to B=4
sed -i "s/comptime B: Int = [0-9]*/comptime B: Int = 4/" "$SRC"
echo ""
echo "Done."
