#!/usr/bin/env bash
# Build and run the CPU/JIT tests as Linux/aarch64 binaries under qemu-aarch64.
#
# Lets the x86_64 -> ARM64 JIT be developed and tested on any Linux box
# (including x86_64 CI and cloud sessions) without a Mac. Only the CPU-side
# sources are built; Metal/AppKit/CoreAudio code is left out.
#
# usage: scripts/cpu_harness.sh [test ...]     (default: 5 13 15)
#   tests are phase numbers (tests/test_phase<N>.cpp) or paths to .cpp files
#   that only need the CPU sources (e.g. tests/test_jit_diff.cpp).
#
# env: CXX (default aarch64-linux-gnu-g++), QEMU (default qemu-aarch64),
#      OPT (default -O1), OUT (default build/aarch64-harness)
#
# requires: apt-get install g++-aarch64-linux-gnu qemu-user
set -euo pipefail

cd "$(dirname "$0")/.."
CXX=${CXX:-aarch64-linux-gnu-g++}
QEMU=${QEMU:-qemu-aarch64}
OPT=${OPT:--O1}
OUT=${OUT:-build/aarch64-harness}
FLAGS=(-std=c++20 "$OPT" -g -Iinclude -Wall -Wextra)

if [[ "$(uname -m)" == "aarch64" ]]; then
    CXX=${CXX_NATIVE:-g++}
    QEMU=""
fi
command -v "$CXX" >/dev/null || { echo "missing $CXX (apt-get install g++-aarch64-linux-gnu)" >&2; exit 2; }
[[ -z "$QEMU" ]] || command -v "$QEMU" >/dev/null || { echo "missing $QEMU (apt-get install qemu-user)" >&2; exit 2; }

SRCS=(src/core/elf_loader.cpp src/core/memory.cpp src/core/syscall_table.cpp
      src/core/vfs.cpp src/cpu/executor.cpp src/cpu/jit_compiler.cpp)

mkdir -p "$OUT"
objs=()
for src in "${SRCS[@]}"; do
    obj="$OUT/$(basename "$src").o"
    if [[ ! -f "$obj" || "$src" -nt "$obj" || -n "$(find include -newer "$obj" -print -quit)" ]]; then
        "$CXX" "${FLAGS[@]}" -c "$src" -o "$obj"
    fi
    objs+=("$obj")
done

tests=("$@")
[[ ${#tests[@]} -gt 0 ]] || tests=(5 13 15)

# QEMU timing is not representative; tests gate perf asserts on this.
export PS4_NO_PERF_ASSERTS=1
fail=0
for t in "${tests[@]}"; do
    [[ "$t" == *.cpp ]] && src=$t || src=tests/test_phase$t.cpp
    name=$(basename "$src" .cpp)
    "$CXX" "${FLAGS[@]}" "$src" "${objs[@]}" -static -pthread -o "$OUT/$name"
    if ${QEMU:+"$QEMU"} "$OUT/$name" >"$OUT/$name.log" 2>&1; then
        echo "PASS $name"
    else
        echo "FAIL $name (log: $OUT/$name.log)"; tail -5 "$OUT/$name.log"; fail=1
    fi
done
exit $fail
