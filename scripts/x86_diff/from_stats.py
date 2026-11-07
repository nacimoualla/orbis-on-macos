#!/usr/bin/env python3
"""Coverage-driven case generator for x86_diff.

Reads a PS4_JIT_STATS JSON file (produced with PS4_JIT_STATS=1), identifies the
top interpreted instruction encodings that keep guest execution in the
interpreter, and generates differential test cases (standard registers, high
registers, aliased operands, memory operands, and edge immediates) into
scripts/x86_diff/gen_cases.py.

Usage:
    python3 scripts/x86_diff/from_stats.py STATS.json [--top N] [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_CASES_PATH = os.path.join(HERE, "gen_cases.py")

# Arithmetic flag masks matching gen_cases.py
CF, PF, AF, ZF, SF, OF = 1, 4, 0x10, 0x40, 0x80, 0x800
ALL = CF | PF | ZF | SF | OF

TERMINATORS = {
    ("", "05"), ("0F", "05"), ("", "0B"), ("0F", "0B"), ("", "C3"), ("", "C2"),
    ("", "CC"), ("", "CD"), ("", "CE"), ("", "CF"), ("", "E8"), ("", "E9"),
    ("", "EB"), ("", "F4")
}

IMM_EDGE_VALUES = [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 255]


def load_stats(paths):
    """Load and aggregate interp_opcodes from one or more stats files."""
    totals = {}
    interp = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d.get("totals", {}).items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
        for e in d.get("interp_opcodes", []):
            key = e.get("key", "")
            if not key:
                continue
            cur = interp.setdefault(key, dict(e, count=0))
            cur["count"] += e.get("count", 0)
    return totals, interp


def is_terminator(entry):
    m = entry.get("map", "")
    op = entry.get("op", "").upper()
    return (m, op) in TERMINATORS


def has_immediate(entry):
    m = entry.get("map", "")
    op = int(entry.get("op", "0"), 16)
    if m == "0F3A":
        return True
    if m in ("", "0F"):
        if op in (0x70, 0x71, 0x72, 0x73, 0xC2, 0xC4, 0xC5, 0xC6):
            return True
        if op in (0x69, 0x6B, 0x80, 0x81, 0x83, 0xC0, 0xC1, 0xC6, 0xC7):
            return True
        if m == "0F" and op == 0xBA:
            return True
    return False


def generate_cases_for_entry(entry):
    """Generate (case_name, python_add_line) for an interpreted opcode entry."""
    lines = []
    key = entry.get("key", "")
    m = entry.get("map", "")
    op_hex = entry.get("op", "0")
    op = int(op_hex, 16)
    pfx = entry.get("pfx", "")
    vex_flag = entry.get("vex", False)
    ext = entry.get("ext")
    name = entry.get("key", f"{pfx} {m} {op_hex}".strip())
    safe_name = re.sub(r"[^a-zA-Z0-9_.]", "_", name)

    # VEX-encoded instruction
    if vex_flag:
        mm = 1 if m == "0F" else 2 if m == "0F38" else 3 if m == "0F3A" else 0
        pp = 1 if pfx == "66" else 2 if pfx == "F3" else 3 if pfx == "F2" else 0
        takes_imm = has_immediate(entry)

        if takes_imm:
            for imm in [0, 1, 15, 16, 255]:
                # 128-bit
                lines.append(f'add("v_{safe_name}_x5_x2_{imm}", vex(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 5}, 5, 2, imm={imm}))')
                # High registers
                lines.append(f'add("v_{safe_name}_x9_x14_{imm}", vex(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 9}, 9, 14, imm={imm}))')
                # Aliased
                lines.append(f'add("v_{safe_name}_x2_x2_{imm}", vex(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 2}, 2, 2, imm={imm}))')
            # Memory
            lines.append(f'add("v_{safe_name}_x1_mem_1", vexm(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 1}, 2, imm=1), mem=1)')
        else:
            # Standard reg
            lines.append(f'add("v_{safe_name}_x5_x4_x2", vex(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 5}, 4, 2))')
            # 256-bit reg
            lines.append(f'add("v_{safe_name}_y5_y4_y2", vex(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 5}, 4, 2, L=1))')
            # High registers
            lines.append(f'add("v_{safe_name}_x9_x9_x14", vex(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 9}, 9, 14))')
            # Aliased
            lines.append(f'add("v_{safe_name}_x2_x2_x2", vex(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 2}, 2, 2))')
            # Memory
            lines.append(f'add("v_{safe_name}_x1_x2_mem", vexm(0x{op:02X}, {pp}, {mm}, {ext if ext is not None else 1}, 2), mem=1)')

    # Legacy SSE instruction (0F or prefix-selected)
    elif m in ("0F", "0F38", "0F3A") or pfx in ("66", "F3", "F2"):
        pfx_list = [f"0x{pfx}"] if pfx else []
        op_list = []
        if m == "0F38":
            op_list = [0x38, op]
        elif m == "0F3A":
            op_list = [0x3A, op]
        else:
            op_list = [op]

        op_str = f"[{', '.join(f'0x{b:02X}' for b in op_list)}]"
        pfx_str = f"[{', '.join(pfx_list)}]"
        takes_imm = has_immediate(entry)

        if takes_imm:
            for imm in [0, 1, 15, 16, 255]:
                lines.append(f'add("sse_{safe_name}_x1_x3_{imm}", sse({pfx_str}, {op_str}, {ext if ext is not None else 1}, 3, imm={imm}))')
                lines.append(f'add("sse_{safe_name}_x9_x14_{imm}", sse({pfx_str}, {op_str}, {ext if ext is not None else 9}, 14, imm={imm}))')
                lines.append(f'add("sse_{safe_name}_x2_x2_{imm}", sse({pfx_str}, {op_str}, {ext if ext is not None else 2}, 2, imm={imm}))')
            lines.append(f'add("sse_{safe_name}_x1_mem_1", ssem({pfx_str}, {op_str}, {ext if ext is not None else 1}, imm=1), mem=1)')
        else:
            lines.append(f'add("sse_{safe_name}_x1_x3", sse({pfx_str}, {op_str}, {ext if ext is not None else 1}, 3))')
            lines.append(f'add("sse_{safe_name}_x9_x14", sse({pfx_str}, {op_str}, {ext if ext is not None else 9}, 14))')
            lines.append(f'add("sse_{safe_name}_x2_x2", sse({pfx_str}, {op_str}, {ext if ext is not None else 2}, 2))')
            lines.append(f'add("sse_{safe_name}_x1_mem", ssem({pfx_str}, {op_str}, {ext if ext is not None else 1}), mem=1)')

    # General-purpose / ALU instructions
    else:
        modrm_reg = (ext << 3) if ext is not None else 0
        pre = f"[0x{pfx}]" if pfx else "[]"
        rex_w = entry.get("rex_w", 0) > 0
        setup = 0
        fmask = "ALL"
        if op == 0xF7 and ext in (6, 7): # div / idiv
            setup = 2 if ext == 7 else 1
            fmask = "0"
        elif op in (0xD0, 0xD1, 0xD2, 0xD3, 0xC0, 0xC1): # shifts
            setup = 3

        # 32-bit and 64-bit variants
        for is_64 in ([False, True] if rex_w else [False]):
            sfx = "64" if is_64 else "32"
            rex_byte = 0x48 if is_64 else 0x40
            rex_pre = f"[0x{rex_byte:02X}]"

            # Standard reg
            lines.append(f'add("gpr_{safe_name}_{sfx}_rax_rcx", {rex_pre} + [0x{op:02X}, 0x{0xC0 | modrm_reg | 1:02X}], {fmask}, setup={setup})')
            # High registers (REX.R and REX.B)
            lines.append(f'add("gpr_{safe_name}_{sfx}_r9_r14", [0x{rex_byte | 0x05:02X}, 0x{op:02X}, 0x{0xC0 | modrm_reg | 6:02X}], {fmask}, setup={setup})')
            # Aliased
            lines.append(f'add("gpr_{safe_name}_{sfx}_rax_rax", {rex_pre} + [0x{op:02X}, 0x{0xC0 | modrm_reg:02X}], {fmask}, setup={setup})')
            # Memory
            lines.append(f'add("gpr_{safe_name}_{sfx}_mem", {rex_pre} + [0x{op:02X}, 0x{0x84 | modrm_reg:02X}, 0xB0, 0xB0, 0x1A, 0, 0], {fmask}, mem=1, setup={setup})')

    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stats_files", nargs="+", help="Path(s) to PS4_JIT_STATS JSON file(s)")
    parser.add_argument("--top", type=int, default=10, help="Number of top interpreted encodings to generate cases for (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print generated cases without modifying gen_cases.py")
    args = parser.parse_args()

    totals, interp = load_stats(args.stats_files)
    if not interp:
        print("No interpreter opcodes found in provided stats files.")
        return 0

    # Read current gen_cases.py
    with open(GEN_CASES_PATH, "r", encoding="utf-8") as f:
        gen_cases_content = f.read()

    # Filter and rank opcodes
    ranked = sorted(interp.values(), key=lambda e: (-e.get("count", 0), e.get("key", "")))
    candidates = [e for e in ranked if not is_terminator(e)]

    top_entries = candidates[:args.top]
    print(f"Top {len(top_entries)} interpreted opcodes from {len(args.stats_files)} stat file(s):")
    for i, e in enumerate(top_entries, 1):
        print(f"  {i}. {e.get('key'):<25} count={e.get('count', 0):,}")

    new_case_lines = []
    emitted_keys = set()
    for e in top_entries:
        k = e.get("key", "")
        # Check if already generated in gen_cases
        safe_name = re.sub(r"[^a-zA-Z0-9_.]", "_", k)
        if safe_name in gen_cases_content:
            print(f"  [skip] Cases for '{k}' already present in gen_cases.py")
            continue

        case_code = generate_cases_for_entry(e)
        if case_code:
            new_case_lines.append(f"\n# Coverage-driven cases for {k} (count={e.get('count', 0):,})")
            new_case_lines.extend(case_code)
            emitted_keys.add(k)

    if not new_case_lines:
        print("All top interpreted opcodes already have cases in gen_cases.py.")
        return 0

    block_str = "\n".join(new_case_lines) + "\n"

    if args.dry_run:
        print("\n--- Dry Run: Generated Code ---")
        print(block_str)
        return 0

    # Inject right before the file writing section in gen_cases.py
    target = 'with open("cases.h", "w")'
    if target not in gen_cases_content:
        target = 'with open(os.path.join(HERE, "cases.h"), "w")'
    if target not in gen_cases_content:
        # Fallback to appending
        updated_content = gen_cases_content + "\n" + block_str
    else:
        parts = gen_cases_content.split(target, 1)
        updated_content = parts[0] + block_str + target + parts[1]

    with open(GEN_CASES_PATH, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"\nSuccessfully added test cases for {len(emitted_keys)} opcode(s) into {GEN_CASES_PATH}")

    # Regenerate cases.h
    print("Regenerating cases.h...")
    res = subprocess.run([sys.executable, GEN_CASES_PATH], cwd=HERE, capture_output=True, text=True)
    if res.returncode != 0:
        print("Failed to regenerate cases.h:", res.stderr)
        return 1
    print(res.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
