#!/usr/bin/env python3
"""Rank the x86 instructions that keep guest code out of the JIT.

Reads one or more JSON files written by the emulator with PS4_JIT_STATS set
(see include/ps4/jit_stats.hpp), sums them, and prints:

  1. totals: guest instructions retired by JIT blocks vs the interpreter;
  2. interpreter-executed opcodes ranked by dynamic count -- the instructions
     to implement next in the JIT, in order of impact;
  3. compile-time block ends: where the block compiler stopped and why.

usage: jit_coverage.py STATS.json [STATS.json ...] [--top N] [--markdown]
                       [--no-block-ends]

Opcode keys are normalized: legacy prefixes and REX are stripped (REX.W,
fs/gs and lock are reported as notes); 66/F3/F2 are kept since they select
the instruction for 0F opcodes; "/n" is the ModRM.reg extension of group
opcodes; "reg"/"mem" is the ModRM r/m form.

"kind" is the JIT's reason for stopping there: "terminator" (deliberate block
end: syscall/call/ret/jcc/hlt/...) or "unsupported" (decoder gave up). A
trailing "?" means the instruction never ended a block in these runs and the
kind is a static guess; "jit-ok?" marks opcodes the JIT already translates
(they were interpreted because no JIT was attached, e.g. --no-jit or direct
CpuExecutor::step() calls).
"""
import argparse
import json
import re
import sys

CC = ["o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g"]
ALU = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"]
SHIFT = ["rol", "ror", "rcl", "rcr", "shl", "shr", "sal", "sar"]
GRP3 = ["test", "test", "not", "neg", "mul", "imul", "div", "idiv"]
GRP5 = ["inc", "dec", "call (indirect)", "callf", "jmp (indirect)", "jmpf", "push", "?"]
GRP15 = ["fxsave", "fxrstor", "ldmxcsr", "stmxcsr", "xsave", "lfence/xrstor", "mfence/xsaveopt",
         "sfence/clflush"]
GRP8 = ["?", "?", "?", "?", "bt", "bts", "btr", "btc"]
REG8 = ["al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil"]
REG32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
REG64 = ["rax/r8", "rcx/r9", "rdx/r10", "rbx/r11", "rsp/r12", "rbp/r13", "rsi/r14", "rdi/r15"]


def _build_table():
    # key: (pfx, map, op, ext) -> mnemonic; ext None = any. map: "", "0F", "0F38", "0F3A"
    t = {}

    def put(op, name, pfx="", m="", ext=None):
        t[(pfx, m, op, ext)] = name

    for i, n in enumerate(ALU):
        base = i * 8
        put(base + 0, f"{n} r/m8, r8")
        put(base + 1, f"{n} r/m, r")
        put(base + 2, f"{n} r8, r/m8")
        put(base + 3, f"{n} r, r/m")
        put(base + 4, f"{n} al, imm8")
        put(base + 5, f"{n} eax, imm32")
        put(0x80, f"{n} r/m8, imm8", ext=i)
        put(0x81, f"{n} r/m, imm32", ext=i)
        put(0x83, f"{n} r/m, imm8", ext=i)
    for r in range(8):
        put(0x50 + r, f"push {REG64[r]}")
        put(0x58 + r, f"pop {REG64[r]}")
        put(0x90 + r, "nop" if r == 0 else f"xchg eax, {REG32[r]}")
        put(0xB0 + r, f"mov {REG8[r]}, imm8")
        put(0xB8 + r, f"mov {REG32[r]}, imm32/64")
        put(0xC0, f"{SHIFT[r]} r/m8, imm8", ext=r)
        put(0xC1, f"{SHIFT[r]} r/m, imm8", ext=r)
        put(0xD0, f"{SHIFT[r]} r/m8, 1", ext=r)
        put(0xD1, f"{SHIFT[r]} r/m, 1", ext=r)
        put(0xD2, f"{SHIFT[r]} r/m8, cl", ext=r)
        put(0xD3, f"{SHIFT[r]} r/m, cl", ext=r)
        put(0xF6, f"{GRP3[r]} r/m8", ext=r)
        put(0xF7, f"{GRP3[r]} r/m", ext=r)
        put(0xFF, GRP5[r] + ("" if r in (2, 4) else " r/m"), ext=r)
    for i, c in enumerate(CC):
        put(0x70 + i, f"j{c} rel8")
        put(0x80 + i, f"j{c} rel32", m="0F")
        put(0x40 + i, f"cmov{c}", m="0F")
        put(0x90 + i, f"set{c}", m="0F")
    for op, n in {
        0x63: "movsxd r, r/m32", 0x68: "push imm32", 0x69: "imul r, r/m, imm32", 0x6A: "push imm8",
        0x6B: "imul r, r/m, imm8", 0x84: "test r/m8, r8", 0x85: "test r/m, r", 0x86: "xchg r/m8, r8",
        0x87: "xchg r/m, r", 0x88: "mov r/m8, r8", 0x89: "mov r/m, r", 0x8A: "mov r8, r/m8",
        0x8B: "mov r, r/m", 0x8D: "lea", 0x98: "cdqe/cwde", 0x99: "cdq/cqo", 0x9C: "pushf",
        0x9D: "popf", 0xA4: "movsb", 0xA5: "movsd/q", 0xA8: "test al, imm8", 0xA9: "test eax, imm32",
        0xAA: "stosb", 0xAB: "stosd/q", 0xC2: "ret imm16", 0xC3: "ret", 0xC9: "leave", 0xCC: "int3",
        0xCD: "int imm8", 0xE8: "call rel32", 0xE9: "jmp rel32", 0xEB: "jmp rel8", 0xF4: "hlt",
        0xFC: "cld", 0xFD: "std",
    }.items():
        put(op, n)
    put(0x8F, "pop r/m", ext=0)
    put(0xC6, "mov r/m8, imm8", ext=0)
    put(0xC7, "mov r/m, imm32", ext=0)
    put(0xFE, "inc r/m8", ext=0)
    put(0xFE, "dec r/m8", ext=1)
    for op in range(0xD8, 0xE0):
        put(op, "x87")
    put(0xC3, "rep ret", pfx="F3")
    put(0xA4, "rep movsb", pfx="F3")
    put(0xA5, "rep movsd/q", pfx="F3")
    put(0xAA, "rep stosb", pfx="F3")
    put(0xAB, "rep stosd/q", pfx="F3")
    put(0x89, "mov r/m16, r16", pfx="66")
    put(0x8B, "mov r16, r/m16", pfx="66")
    put(0xC7, "mov r/m16, imm16", pfx="66", ext=0)

    # 0F map, scalar/packed SSE with prefix-selected variants
    for op, base in {0x10: "movu", 0x11: "movu", 0x51: "sqrt", 0x58: "add", 0x59: "mul",
                     0x5C: "sub", 0x5D: "min", 0x5E: "div", 0x5F: "max"}.items():
        for pfx, sfx in (("", "ps"), ("66", "pd"), ("F3", "ss"), ("F2", "sd")):
            name = base + sfx
            if base == "movu" and pfx in ("F3", "F2"):
                name = "mov" + sfx
            put(op, name + (" (store)" if op == 0x11 else ""), pfx=pfx, m="0F")
    for op, base in {0x28: "mova", 0x29: "mova", 0x54: "and", 0x55: "andn", 0x56: "or",
                     0x57: "xor", 0x14: "unpckl", 0x15: "unpckh", 0xC6: "shuf"}.items():
        put(op, base + "ps" + (" (store)" if op == 0x29 else ""), m="0F")
        put(op, base + "pd" + (" (store)" if op == 0x29 else ""), pfx="66", m="0F")
    for (pfx, op), n in {
        ("", 0x05): "syscall", ("", 0x0B): "ud2", ("", 0x1F): "nop r/m", ("", 0x18): "prefetch",
        ("", 0x12): "movlps/movhlps", ("", 0x13): "movlps (store)", ("", 0x16): "movhps/movlhps",
        ("", 0x17): "movhps (store)", ("66", 0x12): "movlpd", ("66", 0x13): "movlpd (store)",
        ("66", 0x16): "movhpd", ("F3", 0x2A): "cvtsi2ss", ("F2", 0x2A): "cvtsi2sd",
        ("F3", 0x2C): "cvttss2si", ("F2", 0x2C): "cvttsd2si", ("F3", 0x2D): "cvtss2si",
        ("F2", 0x2D): "cvtsd2si", ("", 0x2E): "ucomiss", ("66", 0x2E): "ucomisd",
        ("", 0x2F): "comiss", ("66", 0x2F): "comisd", ("", 0x31): "rdtsc", ("", 0xA2): "cpuid",
        ("", 0x5A): "cvtps2pd", ("66", 0x5A): "cvtpd2ps", ("F3", 0x5A): "cvtss2sd",
        ("F2", 0x5A): "cvtsd2ss", ("", 0x5B): "cvtdq2ps", ("66", 0x5B): "cvtps2dq",
        ("F3", 0x5B): "cvttps2dq", ("66", 0x60): "punpcklbw", ("66", 0x61): "punpcklwd",
        ("66", 0x62): "punpckldq", ("66", 0x6C): "punpcklqdq", ("66", 0x6D): "punpckhqdq",
        ("66", 0x6E): "movd/movq xmm, r/m", ("", 0x6F): "movq mm", ("66", 0x6F): "movdqa",
        ("F3", 0x6F): "movdqu", ("66", 0x70): "pshufd", ("F3", 0x70): "pshufhw",
        ("F2", 0x70): "pshuflw", ("66", 0x74): "pcmpeqb", ("66", 0x75): "pcmpeqw",
        ("66", 0x76): "pcmpeqd", ("", 0x77): "emms/vzeroupper", ("66", 0x7E): "movd/movq r/m, xmm",
        ("F3", 0x7E): "movq xmm, xmm/m64", ("66", 0x7F): "movdqa (store)",
        ("F3", 0x7F): "movdqu (store)", ("", 0xA3): "bt r/m, r", ("", 0xA4): "shld imm8",
        ("", 0xA5): "shld cl", ("", 0xAB): "bts r/m, r", ("", 0xAC): "shrd imm8",
        ("", 0xAD): "shrd cl", ("", 0xAF): "imul r, r/m", ("", 0xB0): "cmpxchg r/m8",
        ("", 0xB1): "cmpxchg r/m", ("", 0xB3): "btr r/m, r", ("", 0xB6): "movzx r, r/m8",
        ("", 0xB7): "movzx r, r/m16", ("", 0xBB): "btc r/m, r", ("", 0xBC): "bsf",
        ("F3", 0xBC): "tzcnt", ("", 0xBD): "bsr", ("F3", 0xBD): "lzcnt", ("F3", 0xB8): "popcnt",
        ("", 0xBE): "movsx r, r/m8", ("", 0xBF): "movsx r, r/m16", ("", 0xC0): "xadd r/m8",
        ("", 0xC1): "xadd r/m", ("", 0xC2): "cmpps", ("66", 0xC2): "cmppd", ("F3", 0xC2): "cmpss",
        ("F2", 0xC2): "cmpsd", ("66", 0xD4): "paddq", ("66", 0xD6): "movq (store)",
        ("66", 0xD7): "pmovmskb", ("66", 0xDB): "pand", ("66", 0xDF): "pandn", ("66", 0xEB): "por",
        ("66", 0xEF): "pxor", ("66", 0xFA): "psubd", ("66", 0xFB): "psubq", ("66", 0xFE): "paddd",
        ("66", 0xD3): "psrlq", ("66", 0xF3): "psllq", ("66", 0xE6): "cvttpd2dq",
        ("F3", 0xE6): "cvtdq2pd", ("66", 0xF4): "pmuludq",
    }.items():
        put(op, n, pfx=pfx, m="0F")
    for r in range(8):
        put(0xC8 + r, f"bswap {REG64[r]}", m="0F")
        put(0xAE, GRP15[r], m="0F", ext=r)
        put(0xBA, f"{GRP8[r]} r/m, imm8", m="0F", ext=r)
    put(0x72, "psrld imm8", pfx="66", m="0F", ext=2)
    put(0x72, "psrad imm8", pfx="66", m="0F", ext=4)
    put(0x72, "pslld imm8", pfx="66", m="0F", ext=6)
    put(0x73, "psrlq imm8", pfx="66", m="0F", ext=2)
    put(0x73, "psrldq imm8", pfx="66", m="0F", ext=3)
    put(0x73, "psllq imm8", pfx="66", m="0F", ext=6)
    put(0x73, "pslldq imm8", pfx="66", m="0F", ext=7)
    for (pfx, m, op), n in {
        ("66", "0F38", 0x00): "pshufb", ("66", "0F38", 0x17): "ptest", ("66", "0F38", 0x2B): "packusdw",
        ("66", "0F38", 0x30): "pmovzxbw", ("66", "0F38", 0x40): "pmulld", ("", "0F38", 0xF0): "movbe",
        ("66", "0F3A", 0x0F): "palignr", ("66", "0F3A", 0x16): "pextrd/q", ("66", "0F3A", 0x22): "pinsrd/q",
        ("66", "0F3A", 0x20): "pinsrb", ("66", "0F3A", 0x21): "insertps", ("66", "0F3A", 0x0B): "roundsd",
        ("66", "0F3A", 0x0A): "roundss", ("66", "0F3A", 0x14): "pextrb",
    }.items():
        put(op, n, pfx=pfx, m=m)
    return t


TABLE = _build_table()

# Instructions the block compiler ends a block on deliberately.
STATIC_TERMINATORS = {("", 0xC3), ("", 0xE8), ("", 0xE9), ("", 0xEB), ("", 0xF4), ("", 0xCC),
                      ("0F", 0x05), ("0F", 0x0B)} | {("", 0x70 + i) for i in range(16)} \
    | {("0F", 0x80 + i) for i in range(16)}


# Opcode keys the current block compiler (src/cpu/jit_compiler.cpp) translates.
# Keep in sync with JIT_SUPPORTED in scripts/opcode_census.py.
JIT_SUPPORTED = {"89 reg", "8B reg", "C7 /0 reg", "31 reg", "33 reg", "01 reg", "03 reg",
                 "29 reg", "2B reg", "21 reg", "23 reg", "09 reg", "0B reg", "83 /0 reg",
                 "83 /5 reg", "90", "0F 1F reg", "0F 1F mem", "EB", "E9"} \
    | {"B8+r"}

# Register-in-opcode instructions (push/pop/mov imm/xchg/bswap r) are one
# instruction each; fold the 8 opcodes into one "+r" row.
FOLDS = {"": {0x50: 0x50, 0x58: 0x58, 0xB0: 0xB0, 0xB8: 0xB8}, "0F ": {0xC8: 0xC8}}


def fold_key(key):
    """'53' -> '50+r', '0F CB' -> '0F C8+r'; other keys unchanged."""
    m = re.fullmatch(r"((?:66 |F3 |F2 )?)(0F )?([0-9A-F]{2})", key)
    if not m:
        return key
    op = int(m.group(3), 16)
    base = op & 0xF8
    if base not in FOLDS[m.group(2) or ""]:
        return key
    return f"{m.group(1)}{m.group(2) or ''}{base:02X}+r"


FOLDED_NAMES = {0x50: "push r64", 0x58: "pop r64", 0xB0: "mov r8, imm8",
                0xB8: "mov r32, imm32 / movabs r64, imm64"}


def mnemonic(e):
    op = int(e["op"], 16)
    if e["key"].endswith("+r"):
        if e["map"] == "0F":
            return "bswap r"
        return FOLDED_NAMES.get(op, "?")
    m, pfx, ext = e["map"], e["pfx"], e.get("ext")
    for key in ((pfx, m, op, ext), (pfx, m, op, None), ("", m, op, ext), ("", m, op, None)):
        if key in TABLE:
            name = TABLE[key]
            if key[0] != pfx and pfx:
                name += f" ({pfx})"
            break
    else:
        name = "?"
    if e.get("vex"):
        name = "v" + name if name != "?" else "VEX ?"
        if m == "0F" and op == 0x77:
            name = "vzeroupper/vzeroall"
    return name


def static_kind(e):
    if e["mod"] == "bad":
        return "unsupported"
    if e["key"] in JIT_SUPPORTED:
        return "jit-ok"  # interpreted only because no JIT ran (e.g. step() / --no-jit)
    op = int(e["op"], 16)
    return "terminator" if (e["map"], op) in STATIC_TERMINATORS and not e.get("vex") else "unsupported"


def refold(e):
    key = fold_key(e["key"])
    if key == e["key"]:
        return e
    return dict(e, key=key, op=f"{int(e['op'], 16) & 0xF8:02X}")


def load(paths):
    totals, interp, ends = {}, {}, {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for k, v in d.get("totals", {}).items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
        for e in d.get("interp_opcodes", []):
            e = refold(e)
            cur = interp.setdefault(e["key"], dict(e, count=0, rex_w=0, seg_fs_gs=0, lock=0))
            for f_ in ("count", "rex_w", "seg_fs_gs", "lock"):
                cur[f_] += e.get(f_, 0)
        for e in d.get("block_ends", []):
            e = refold(e)
            cur = ends.setdefault((e["key"], e["reason"]), dict(e, blocks=0, failed_compiles=0))
            cur["blocks"] += e.get("blocks", 0)
            cur["failed_compiles"] += e.get("failed_compiles", 0)
    return totals, interp, ends


def render(headers, rows, markdown, align):
    if markdown:
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join("---:" if a == ">" else "---" for a in align) + "|"]
        out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
        return "\n".join(out)
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]

    def fmt(r):
        return "  ".join(f"{str(c):{a}{w}}" for c, a, w in zip(r, align, widths)).rstrip()
    return "\n".join([fmt(headers), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows])


def pct(n, d):
    if not d:
        return "-"
    v = 100.0 * n / d
    return "<0.01" if 0 < v < 0.01 else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="PS4_JIT_STATS JSON file(s)")
    ap.add_argument("--top", type=int, default=30, help="rows per table (0 = all)")
    ap.add_argument("--markdown", action="store_true", help="emit Markdown tables")
    ap.add_argument("--no-block-ends", action="store_true", help="skip the block-end table")
    args = ap.parse_args()

    totals, interp, ends = load(args.files)
    jit = totals.get("jit_retired", 0)
    itp = totals.get("interp_retired", 0)
    total = jit + itp
    execs = totals.get("jit_block_executions", 0)
    h = "### " if args.markdown else ""
    print(f"{h}JIT coverage ({len(args.files)} run(s))\n")
    li = "- " if args.markdown else ""
    print(f"{li}guest instructions : {total:,}")
    print(f"{li}  via JIT blocks   : {jit:,} ({pct(jit, total)}%)")
    print(f"{li}  via interpreter  : {itp:,} ({pct(itp, total)}%)")
    print(f"{li}JIT block execs    : {execs:,} (avg {jit / execs if execs else 0:.2f} insns/block)")
    print(f"{li}blocks compiled    : {totals.get('blocks_compiled', 0):,}")
    print(f"{li}compile failures   : {totals.get('compile_failures', 0):,}"
          " (first instruction unsupported/terminator; retried on every visit)\n")

    # dominant compile-time reason per key
    reason = {}
    for (key, r), e in ends.items():
        w = e["blocks"] + e["failed_compiles"]
        if w > reason.get(key, ("", -1))[1]:
            reason[key] = (r, w)

    ranked = sorted(interp.values(), key=lambda e: (-e["count"], e["key"]))
    if args.top:
        ranked_shown = ranked[:args.top]
    else:
        ranked_shown = ranked
    rows, cum = [], 0
    for i, e in enumerate(ranked_shown, 1):
        cum += e["count"]
        kind = reason[e["key"]][0] if e["key"] in reason else static_kind(e) + "?"
        notes = []
        if e["rex_w"]:
            notes.append(f"W {pct(e['rex_w'], e['count'])}%")
        if e["seg_fs_gs"]:
            notes.append(f"fs/gs {pct(e['seg_fs_gs'], e['count'])}%")
        if e["lock"]:
            notes.append(f"lock {pct(e['lock'], e['count'])}%")
        rows.append([i, e["key"], mnemonic(e), f"{e['count']:,}", pct(e["count"], itp),
                     pct(cum, itp), kind, ", ".join(notes)])
    print(f"{h}Interpreted instructions by dynamic count (implement these next)\n")
    print(render(["#", "opcode", "mnemonic", "count", "% interp", "cum %", "kind", "notes"],
                 rows, args.markdown, [">", "<", "<", ">", ">", ">", "<", "<"]))
    if args.top and len(ranked) > args.top:
        rest = sum(e["count"] for e in ranked[args.top:])
        print(f"... {len(ranked) - args.top} more opcodes, {rest:,} executions ({pct(rest, itp)}%)")

    if not args.no_block_ends and ends:
        er = sorted(ends.values(), key=lambda e: (-(e["blocks"] + e["failed_compiles"]), e["key"]))
        if args.top:
            er = er[:args.top]
        rows = [[i, e["key"], mnemonic(e), e["reason"], f"{e['blocks']:,}", f"{e['failed_compiles']:,}"]
                for i, e in enumerate(er, 1)]
        print(f"\n{h}JIT block ends (compile time: where the block compiler stopped)\n")
        print(render(["#", "opcode", "mnemonic", "reason", "blocks", "failed compiles"],
                     rows, args.markdown, [">", "<", "<", "<", ">", ">"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
