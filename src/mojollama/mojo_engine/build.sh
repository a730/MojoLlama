# MojoLlama Mojo Engine — Build Script
# Builds the pure Mojo engine from source.
#
# Prerequisites:
#   - Mojo 0.26.2 installed
#   - GCC
#   - Model weights dumped (via dump_weights.py)
#
# Usage:
#   bash build.sh              # Build mojo_engine_v6
#   LD_LIBRARY_PATH=<mojo_lib> ./mojo_engine_v6

MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"

# 1. Build the C bridge (file I/O + raw memory access)
gcc -c -fPIC -o read_helpers.o read_helpers.c

# 2. Build Mojo engine to object file
mojo build mojo_engine_v6_final.mojo --emit object -o mojo_v6.o

# 3. Link with Mojo runtime
gcc -o mojo_engine_v6 mojo_v6.o read_helpers.o \
    -L"$MOJO_LIB" \
    -lKGENCompilerRTShared \
    -lAsyncRTRuntimeGlobals \
    -lMSupportGlobals \
    -lAsyncRTMojoBindings \
    -lm -lpthread -ldl -lstdc++

echo "Built: mojo_engine_v6"
echo "Run: LD_LIBRARY_PATH=\"$MOJO_LIB\" ./mojo_engine_v6"
