#!/usr/bin/env bash
# Run the JIT-vs-interpreter differential fuzzer (tests/test_jit_diff.cpp)
# across several seeds under the aarch64/QEMU CPU harness and summarize:
# pass/fail per seed, summed per-group JIT coverage, and the unique minimized
# failures across all seeds.
#
# usage: scripts/jit_fuzz.sh [seed ...]       (default seeds: 1 2 3 4 5)
#
# env:   SEEDS="1 2 3"     alternative to positional seeds
#        JIT_DIFF_*        passed through (GROUPS, TRIALS, LEN, CHECK_FLAGS,
#                          CHECK_AF, MAX_REPORTS, ...; see the test's header)
#        QEMU, OUT         as in scripts/cpu_harness.sh
#
# Per-seed logs (with full mismatch reports): $OUT/jit_fuzz.seed<N>.log
set -uo pipefail

cd "$(dirname "$0")/.."
OUT=${OUT:-build/aarch64-harness}
QEMU=${QEMU:-qemu-aarch64}
[[ "$(uname -m)" == "aarch64" ]] && QEMU=""

seeds=("$@")
[[ ${#seeds[@]} -gt 0 ]] || read -ra seeds <<<"${SEEDS:-1 2 3 4 5}"

# Build via the harness (it also runs the binary once; one trial keeps it quick).
bin="$OUT/test_jit_diff"
rm -f "$bin"
JIT_DIFF_TRIALS=1 scripts/cpu_harness.sh tests/test_jit_diff.cpp >/dev/null 2>&1
[[ -x "$bin" ]] || { echo "build failed: run scripts/cpu_harness.sh tests/test_jit_diff.cpp" >&2; exit 2; }

fail=0
logs=()
for s in "${seeds[@]}"; do
    log="$OUT/jit_fuzz.seed$s.log"
    logs+=("$log")
    if JIT_DIFF_SEED=$s ${QEMU:+"$QEMU"} "$bin" >"$log" 2>&1; then st=PASS; else st=FAIL; fail=1; fi
    echo "$st seed=$s  $(grep '^jit_diff:' "$log" || echo '(no summary; crashed?)')"
done

echo
echo "coverage (summed over ${#seeds[@]} seeds):"
awk '
    /^group / { intable = 1; next }
    intable && NF == 0 { intable = 0; next }
    intable && NF == 7 {
        if (!($1 in insns)) order[n++] = $1
        insns[$1] += $2; jit[$1] += $3; interp[$1] += $4; fails[$1] += $7
    }
    END {
        printf "  %-10s %8s %8s %8s %7s %6s\n", "group", "insns", "jit", "interp", "jit%", "fails"
        for (i = 0; i < n; i++) {
            g = order[i]
            printf "  %-10s %8d %8d %8d %6.1f%% %6d\n", g, insns[g], jit[g], interp[g],
                   insns[g] ? 100.0 * jit[g] / insns[g] : 0, fails[g]
        }
    }' "${logs[@]}"

echo
if grep -qh '^  x[0-9]' "${logs[@]}"; then
    echo "unique minimized failures across seeds (count, example bytes, sequence):"
    grep -h '^  x[0-9]' "${logs[@]}" | awk '
        {
            count = substr($1, 2) + 0
            lb = index($0, "["); rb = index($0, "]")
            bytes = substr($0, lb + 1, rb - lb - 1)
            sig = substr($0, rb + 3)
            if (!(sig in total)) { example[sig] = bytes; order[n++] = sig }
            total[sig] += count
        }
        END { for (i = 0; i < n; i++) printf "  x%-5d [%s]  %s\n", total[order[i]], example[order[i]], order[i] }'
    echo
    echo "details: ${logs[*]}"
fi
exit $fail
