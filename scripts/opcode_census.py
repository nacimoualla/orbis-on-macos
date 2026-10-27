#!/usr/bin/env python3
"""Static opcode census of a guest binary, and how much of it the JIT covers.

Disassembles the executable segments of one or more x86-64 ELF files (Orbis
samples/*.elf, an unpacked eboot.bin / .prx, or any x86-64 ELF) and prints:

  1. coverage: static instructions the block compiler translates today,
     deliberate block terminators, and unsupported instructions;
  2. unsupported opcodes ranked by static count -- what to implement next;
  3. the mnemonic histogram.

usage: opcode_census.py ELF [ELF ...] [--top N] [--markdown]
                        [--assume KEY ...] [--json OUT]

Opcode keys use the same normalization as the runtime telemetry
(PS4_JIT_STATS / scripts/jit_coverage.py), e.g. "8B mem", "83 /7 reg",
"66 0F 6F mem". --assume KEY (repeatable; fnmatch globs allowed) adds keys to
the supported set to answer "what if the JIT handled X?".

Uses the capstone package when available (pip install capstone), otherwise
`objdump -D`. Static counts are a proxy only: hot loops matter more than code
size, so prefer PS4_JIT_STATS data from a real run when you have it. SELF
files (eboot.bin straight from a dump) must be unpacked first with
scripts/unpack_self.py.
"""
import argparse
import collections
import fnmatch
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    import capstone  # type: ignore
except ImportError:  # pragma: no cover - exercised when capstone is absent
    capstone = None

# Opcode keys the current block compiler (src/cpu/jit_compiler.cpp) translates.
# Keep in sync with JIT_SUPPORTED in scripts/jit_coverage.py.
JIT_SUPPORTED = {"89 reg", "8B reg", "C7 /0 reg", "31 reg", "33 reg", "01 reg", "03 reg",
                 "29 reg", "2B reg", "21 reg", "23 reg", "09 reg", "0B reg", "83 /0 reg",
                 "83 /5 reg", "90", "0F 1F reg", "0F 1F mem", "EB", "E9"} \
    | {"B8+r"}
# Instructions the block compiler ends a block on (left to the interpreter).
JIT_TERMINATORS = {"0F 05", "0F 0B", "F4", "CC", "C3", "E8"} \
    | {f"{0x70 + i:02X}" for i in range(16)} | {f"0F {0x80 + i:02X}" for i in range(16)}

# ---------------------------------------------------------------------------
# Opcode normalization (Python port of normalize_opcode in src/cpu/jit_stats.cpp)
# ---------------------------------------------------------------------------
LEGACY_GROUPS = {0x80, 0x81, 0x82, 0x83, 0x8F, 0xC0, 0xC1, 0xC6, 0xC7, 0xD0, 0xD1, 0xD2, 0xD3,
                 0xF6, 0xF7, 0xFE, 0xFF} | set(range(0xD8, 0xE0))
TWOBYTE_GROUPS = {0x00, 0x01, 0x0D, 0x18, 0x71, 0x72, 0x73, 0xAE, 0xB9, 0xBA, 0xC7}


def legacy_has_modrm(op):
    if op < 0x40:
        return (op & 7) < 4
    return (op in (0x62, 0x63, 0x69, 0x6B, 0xC0, 0xC1, 0xC6, 0xC7, 0xF6, 0xF7, 0xFE, 0xFF)
            or 0x80 <= op <= 0x8F or 0xD0 <= op <= 0xD3 or 0xD8 <= op <= 0xDF)


def twobyte_has_modrm(op):
    if 0x05 <= op <= 0x0C or op == 0x0E or 0x30 <= op <= 0x3F or op == 0x77:
        return False
    if 0x80 <= op <= 0x8F or op in (0xA0, 0xA1, 0xA2, 0xA8, 0xA9, 0xAA):
        return False
    return not 0xC8 <= op <= 0xCF

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


def normalize(code):
    """Return (key, rex_w) for the instruction at the start of `code`."""
    saw66, rep, rex_w, i = False, 0, False, 0
    while i < len(code) and i < 14:
        b = code[i]
        if b == 0x66:
            saw66 = True
        elif b in (0xF2, 0xF3):
            rep = b
        elif b in (0xF0, 0x64, 0x65, 0x2E, 0x36, 0x3E, 0x26, 0x67):
            pass
        elif 0x40 <= b <= 0x4F:
            rex_w = bool(b & 8)
        else:
            break
        i += 1
    pfx = 2 if rep == 0xF3 else 3 if rep == 0xF2 else 1 if saw66 else 0
    if i >= len(code):
        return "<undecodable>", rex_w
    vex, b0 = False, code[i]
    if b0 in (0xC4, 0xC5):
        vex = True
        if b0 == 0xC5:
            if i + 2 >= len(code):
                return "<undecodable>", rex_w
            pfx, m, i = code[i + 1] & 3, 1, i + 2
        else:
            if i + 3 >= len(code) or not 1 <= (code[i + 1] & 0x1F) <= 3:
                return "<undecodable>", rex_w
            m, rex_w, pfx, i = code[i + 1] & 0x1F, bool(code[i + 2] & 0x80), code[i + 2] & 3, i + 3
        op = code[i]
        i += 1
        has_modrm = not (m == 1 and op == 0x77)
    elif b0 == 0x0F:
        if i + 1 >= len(code):
            return "<undecodable>", rex_w
        b1 = code[i + 1]
        if b1 in (0x38, 0x3A):
            if i + 2 >= len(code):
                return "<undecodable>", rex_w
            m, op, i, has_modrm = (2 if b1 == 0x38 else 3), code[i + 2], i + 3, True
        else:
            m, op, i, has_modrm = 1, b1, i + 2, twobyte_has_modrm(b1)
    else:
        m, op, i, has_modrm = 0, b0, i + 1, legacy_has_modrm(b0)
    suffix = ""
    if has_modrm:
        if i >= len(code):
            return "<undecodable>", rex_w
        modrm = code[i]
        group = (m == 0 and op in LEGACY_GROUPS) or (m == 1 and op in TWOBYTE_GROUPS)
        if group:
            suffix += f" /{(modrm >> 3) & 7}"
        suffix += " reg" if modrm >> 6 == 3 else " mem"
    p = ["", "66", "F3", "F2"][pfx]
    if vex:
        mp = ["", "0F", "0F38", "0F3A"][m]
        return f"VEX.{p + '.' if p else ''}{mp} {op:02X}{suffix}", rex_w
    mp = ["", "0F ", "0F 38 ", "0F 3A "][m]
    return f"{p + ' ' if p else ''}{mp}{op:02X}{suffix}", rex_w


# ---------------------------------------------------------------------------
# ELF loading
# ---------------------------------------------------------------------------
def exec_regions(path):
    """Yield (vaddr, bytes) for each executable region of an x86-64 ELF."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] == b"\x4f\x15\x3d\x1d":
        raise SystemExit(f"{path}: SELF container; unpack it first (scripts/unpack_self.py)")
    if data[:4] != b"\x7fELF" or data[4] != 2:
        raise SystemExit(f"{path}: not a 64-bit ELF")
    (e_phoff, e_shoff) = struct.unpack_from("<QQ", data, 0x20)
    (e_phentsize, e_phnum, e_shentsize, e_shnum) = struct.unpack_from("<HHHH", data, 0x36)
    regions = []
    for n in range(e_phnum):
        p_type, p_flags, p_offset, p_vaddr, _, p_filesz = struct.unpack_from(
            "<IIQQQQ", data, e_phoff + n * e_phentsize)
        if p_type == 1 and p_flags & 1 and p_filesz:  # PT_LOAD, PF_X
            regions.append((p_vaddr, data[p_offset:p_offset + p_filesz]))
    if not regions and e_shoff:  # relocatable objects: use SHF_EXECINSTR sections
        for n in range(e_shnum):
            sh = struct.unpack_from("<IIQQQQ", data, e_shoff + n * e_shentsize)
            _, sh_type, sh_flags, sh_addr, sh_offset, sh_size = sh
            if sh_flags & 4 and sh_type == 1 and sh_size:  # SHF_EXECINSTR, SHT_PROGBITS
                regions.append((sh_addr, data[sh_offset:sh_offset + sh_size]))
    return regions


def trim_padding(blob):
    """Drop trailing zero padding (page-aligned segments) so it doesn't count as code."""
    end = len(blob)
    while end and blob[end - 1] == 0:
        end -= 1
    return blob[:end]


# ---------------------------------------------------------------------------
# Disassembly backends: both yield (mnemonic, bytes)
# ---------------------------------------------------------------------------
def disasm_capstone(vaddr, blob):
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.skipdata = True
    for addr, size, mnem, _ in md.disasm_lite(blob, vaddr):
        if mnem == ".byte":
            yield "(data)", None
        else:
            yield mnem, blob[addr - vaddr:addr - vaddr + size]


OBJDUMP_LINE = re.compile(r"^\s*([0-9a-f]+):\t([0-9a-f ]+?)\s*(?:\t(\S+).*)?$")


def disasm_objdump(vaddr, blob):
    objdump = shutil.which("objdump")
    if not objdump:
        raise SystemExit("neither capstone (pip install capstone) nor objdump is available")
    if not blob:
        return
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        tmp.write(blob)
        tmp.flush()
        out = subprocess.run(
            [objdump, "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
             f"--adjust-vma={vaddr:#x}", tmp.name],
            check=True, capture_output=True, text=True).stdout
    cur = None
    for line in out.splitlines():
        m = OBJDUMP_LINE.match(line)
        if not m:
            continue
        raw = bytes.fromhex(m.group(2).replace(" ", ""))
        if m.group(3) is None:  # continuation line of a long instruction
            if cur:
                cur[1] += raw
            continue
        if cur:
            yield cur[0], bytes(cur[1])
        mnem = m.group(3)
        cur = ["(data)" if mnem in ("(bad)", ".byte") else mnem, bytearray(raw)]
    if cur:
        yield cur[0], bytes(cur[1])


# ---------------------------------------------------------------------------
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
    return f"{100.0 * n / d:.2f}" if d else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="x86-64 ELF files")
    ap.add_argument("--top", type=int, default=25, help="rows per table (0 = all)")
    ap.add_argument("--markdown", action="store_true", help="emit Markdown tables")
    ap.add_argument("--assume", action="append", default=[], metavar="KEY",
                    help="treat opcode KEY (glob) as JIT-supported, e.g. '8B mem'")
    ap.add_argument("--objdump", action="store_true", help="force the objdump backend")
    ap.add_argument("--json", metavar="OUT", help="also write the census as JSON")
    args = ap.parse_args()

    use_capstone = capstone is not None and not args.objdump
    disasm = disasm_capstone if use_capstone else disasm_objdump

    def supported(key):
        return key in JIT_SUPPORTED or any(fnmatch.fnmatchcase(key, a) for a in args.assume)

    keys = collections.Counter()
    mnems = collections.Counter()
    key_mnems = collections.defaultdict(collections.Counter)
    kinds = collections.Counter()
    data_bytes = 0
    runs, run = [], 0  # lengths of consecutive JIT-translatable instruction runs
    for path in args.files:
        for vaddr, blob in exec_regions(path):
            for mnem, raw in disasm(vaddr, trim_padding(blob)):
                if raw is None or mnem == "(data)":
                    data_bytes += len(raw) if raw else 1
                    continue
                key = fold_key(normalize(raw)[0])
                keys[key] += 1
                mnems[mnem] += 1
                key_mnems[key][mnem] += 1
                if supported(key):
                    kinds["supported"] += 1
                    run += 1
                    continue
                kinds["terminator" if key in JIT_TERMINATORS else "unsupported"] += 1
                if run:
                    runs.append(run)
                run = 0
        if run:
            runs.append(run)
            run = 0

    total = sum(keys.values())
    h = "### " if args.markdown else ""
    li = "- " if args.markdown else ""
    print(f"{h}Opcode census: {', '.join(os.path.basename(p) for p in args.files)}"
          f" ({'capstone ' + capstone.__version__ if use_capstone else 'objdump'})\n")
    print(f"{li}static instructions : {total:,}" + (f" (+{data_bytes} data bytes skipped)"
                                                    if data_bytes else ""))
    for k in ("supported", "terminator", "unsupported"):
        print(f"{li}  {k:<18}: {kinds[k]:,} ({pct(kinds[k], total)}%)")
    if runs:
        print(f"{li}mean JIT-able run   : {sum(runs) / len(runs):.2f} instructions"
              f" ({len(runs)} runs)")
    if args.assume:
        print(f"{li}assumed supported   : {', '.join(args.assume)}")
    print()

    cand = [(k, c) for k, c in keys.most_common() if not supported(k)]
    shown = cand[:args.top] if args.top else cand
    rows, cum = [], 0
    for i, (k, c) in enumerate(shown, 1):
        cum += c
        ex = ", ".join(m for m, _ in key_mnems[k].most_common(3))
        rows.append([i, k, ex, f"{c:,}", pct(c, total), pct(cum, total),
                     "terminator" if k in JIT_TERMINATORS else "unsupported"])
    print(f"{h}Not JIT-translated, by static count (implement next)\n")
    print(render(["#", "opcode", "mnemonic(s)", "count", "% all", "cum %", "kind"], rows,
                 args.markdown, [">", "<", "<", ">", ">", ">", "<"]))

    mshown = mnems.most_common(args.top or None)
    print(f"\n{h}Mnemonic histogram\n")
    print(render(["#", "mnemonic", "count", "%"],
                 [[i, m, f"{c:,}", pct(c, total)] for i, (m, c) in enumerate(mshown, 1)],
                 args.markdown, [">", "<", ">", ">"]))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"files": args.files, "total": total, "coverage": dict(kinds),
                       "opcodes": [{"key": k, "count": c, "supported": supported(k),
                                    "mnemonics": dict(key_mnems[k])}
                                   for k, c in keys.most_common()],
                       "mnemonics": dict(mnems.most_common())}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
