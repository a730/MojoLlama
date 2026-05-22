#!/bin/bash
# Fast benchmark: thread × batch sweep
set -e

SRC="tinyllama_gen_q8.mojo"
cd "$(dirname "$0")"

MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"
NW_VALUES=(1 2 4 8 12 16 24 32)
B_VALUES=(2 4 8)
RPW=8

echo "=== Thread × Batch Sweep (Q8_0, RPW=$RPW) ==="
printf "%-4s %-4s %s\n" "B" "nw" "tok/s"
printf "%-4s %-4s %s\n" "--" "---" "-----"

for b in "${B_VALUES[@]}"; do
  # Build for this B
  cp "$SRC" "${SRC}.bak"
  sed -i "s/comptime B: Int = [0-9]*/comptime B: Int = $b/" "$SRC"
  sed -i "s/comptime RPW: Int = [0-9]*/comptime RPW: Int = $RPW/" "$SRC"

  OBJ="${SRC%.mojo}_B${b}.o"
  BIN="${SRC%.mojo}_B${b}"

  echo "Building B=$b..."
  mojo build "$SRC" --emit object -o "$OBJ" 2>/dev/null
  gcc -o "$BIN" "$OBJ" \
    -L"$MOJO_LIB" \
    -lKGENCompilerRTShared \
    -lAsyncRTRuntimeGlobals \
    -lMSupportGlobals \
    -lAsyncRTMojoBindings \
    -lm -lpthread -ldl -lstdc++ 2>/dev/null

  for nw in "${NW_VALUES[@]}"; do
    out=$(LD_LIBRARY_PATH="$MOJO_LIB" timeout 120 "./$BIN" "$nw" 2>&1 | grep "tok/s" | tail -1)
    tok_s=$(echo "$out" | grep -oP '\(\s*[\d.]+' | tail -1 | sed 's/(//')
    printf "%-4s %-4s %s\n" "$b" "$nw" "${tok_s:-ERR}"
  done

  rm -f "$OBJ" "$BIN"
  mv "${SRC}.bak" "$SRC"
done

echo ""
echo "=== Sequential Baseline (nw=1) ==="
echo "B=2 nw=1 provides single-stream throughput"
echo ""
echo "=== Done ==="
