#!/bin/bash
# Full benchmark: thread × batch sweep + concurrency test
set -e

SRC="tinyllama_gen_q8.mojo"
BASE_DIR="$(dirname "$0")"
cd "$BASE_DIR"

MOJO_LIB="$HOME/.local/share/uv/tools/mojo/lib/python3.11/site-packages/modular/lib"
LD_LIBRARY_PATH="$MOJO_LIB"

NW_VALUES=(1 2 4 8 16 32)
B_VALUES=(1 2 4 8)
RPW_VALUES=(4 8 16)

echo "=== Thread × Batch × UBATCH Sweep ==="
echo "B,RPW,nw,tok_s"
printf "%-4s %-5s %-4s %s\n" "B" "RPW" "nw" "tok/s"
echo "---- ---- ---- ------"

for rpw in "${RPW_VALUES[@]}"; do
  for b in "${B_VALUES[@]}"; do
    # Build for this B, RPW
    cp "$SRC" "${SRC}.bak"
    sed -i "s/comptime B: Int = [0-9]*/comptime B: Int = $b/" "$SRC"
    sed -i "s/comptime RPW: Int = [0-9]*/comptime RPW: Int = $rpw/" "$SRC"
    
    OBJ="${SRC%.mojo}_B${b}_R${rpw}.o"
    BIN="${SRC%.mojo}_B${b}_R${rpw}"
    
    mojo build "$SRC" --emit object -o "$OBJ" 2>/dev/null
    gcc -o "$BIN" "$OBJ" \
      -L"$MOJO_LIB" \
      -lKGENCompilerRTShared \
      -lAsyncRTRuntimeGlobals \
      -lMSupportGlobals \
      -lAsyncRTMojoBindings \
      -lm -lpthread -ldl -lstdc++ 2>/dev/null
    
    for nw in "${NW_VALUES[@]}"; do
      # Run 3 times, take median
      results=()
      for trial in 1 2 3; do
        out=$(LD_LIBRARY_PATH="$MOJO_LIB" timeout 90 "./$BIN" "$nw" 2>&1 | grep "tok/s" | tail -1)
        tok_s=$(echo "$out" | grep -oP '\(\s*[\d.]+' | tail -1 | sed 's/(//')
        if [ -n "$tok_s" ]; then
          results+=("$tok_s")
        fi
      done
      if [ ${#results[@]} -ge 1 ]; then
        IFS=$'\n' sorted=($(sort -n <<< "${results[*]}")); unset IFS
        mid=$(( ${#sorted[@]} / 2 ))
        median="${sorted[$mid]}"
        printf "%-4s %-5s %-4s %s\n" "$b" "$rpw" "$nw" "$median"
      fi
    done
    
    rm -f "$OBJ" "$BIN"
    mv "${SRC}.bak" "$SRC"
  done
done

echo ""
echo "=== Concurrency Test (B=4, nw=16, 2 instances) ==="
echo "Building baseline..."
cp "$SRC" "${SRC}.bak"
sed -i "s/comptime B: Int = [0-9]*/comptime B: Int = 4/" "$SRC"
sed -i "s/comptime RPW: Int = [0-9]*/comptime RPW: Int = 8/" "$SRC"
mojo build "$SRC" --emit object -o tinyllama_q8_conc.o 2>/dev/null
gcc -o tinyllama_q8_conc tinyllama_q8_conc.o \
  -L"$MOJO_LIB" \
  -lKGENCompilerRTShared \
  -lAsyncRTRuntimeGlobals \
  -lMSupportGlobals \
  -lAsyncRTMojoBindings \
  -lm -lpthread -ldl -lstdc++ 2>/dev/null

echo "Single instance (baseline):"
LD_LIBRARY_PATH="$MOJO_LIB" timeout 90 ./tinyllama_q8_conc 16 2>&1 | grep "tok/s"

echo "Two concurrent instances:"
LD_LIBRARY_PATH="$MOJO_LIB" timeout 90 ./tinyllama_q8_conc 8 &
PID1=$!
LD_LIBRARY_PATH="$MOJO_LIB" timeout 90 ./tinyllama_q8_conc 8 &
PID2=$!
wait $PID1 $PID2
# Print results
echo "Concurrent results:"

mv "${SRC}.bak" "$SRC"
rm -f tinyllama_q8_conc.o tinyllama_q8_conc
echo "=== Done ==="
