#!/bin/bash
# Differential check of x86 instruction emulation against real hardware.
#
# cases.h lists instruction byte sequences. native.c runs each one on the host
# under Rosetta 2 (x86-64) with random register/memory state; emu.cpp runs the
# same cases through CpuExecutor (interpreter, then JIT). Any line that differs
# is an emulation bug. Requires Rosetta 2 and a built build/libps4_core.a.
#
# usage: scripts/x86_diff/run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/scripts/x86_diff"
OUT="$(mktemp -d)"
clang -arch x86_64 -O1 -I"$HERE" "$HERE/native.c" -o "$OUT/native"
clang++ -std=c++20 -O0 -I"$ROOT/include" -I"$HERE" "$HERE/emu.cpp" "$ROOT/build/libps4_core.a" \
    -framework Metal -framework Foundation -framework AppKit -framework QuartzCore \
    -framework CoreAudio -framework AudioToolbox -framework GameController -o "$OUT/emu"
arch -x86_64 "$OUT/native" > "$OUT/native.txt"
"$OUT/emu" 2>/dev/null | grep -v '^#' | grep '^[0-9]' > "$OUT/interp.txt"
"$OUT/emu" jit 2>/dev/null | grep -v '^#' | grep '^[0-9]' > "$OUT/jit.txt"
status=0
for mode in interp jit; do
    if diff -q "$OUT/native.txt" "$OUT/$mode.txt" >/dev/null; then
        echo "$mode: all $(wc -l < "$OUT/native.txt" | tr -d ' ') cases match x86"
    else
        status=1
        echo "$mode: MISMATCH (case.trial lines):"
        diff "$OUT/native.txt" "$OUT/$mode.txt" | grep '^[<>]' | cut -c1-12 | sort -u | head -20
    fi
done
echo "outputs in $OUT"
exit $status
