#!/usr/bin/env python3
"""Audit every NID registered in src/hle/nid_registry.cpp against ps4libdoc.

Why this exists: a mislabeled NID is invisible until it misleads. scePadInit sat
on top of pthread_mutexattr_init (0x7501D612C26DA04E) for a full run and only
showed up as a nonsense "scePadInit x42" trace; sceKernelSignalSema was once
swallowed by an sceGnmWaitLabel label stub. The NID itself is always right
(Sony NIDs are SHA-1(name + salt)[:8] little-endian), so the name can be
*proven* wrong instead of guessed at.

Authorities, strongest first:
  1. known_names.txt    ps4libdoc's 42k-name corpus: resolves any NID to a
                        symbol name, across every module.
  2. *.sprx.json        per-module symbol table: same names plus the library
                        that exports them (libkernel / libScePosix / ...).
  3. Orbis hash         SHA-1(name + salt)[:8] little-endian. Catches mislabels
                        for NIDs ps4libdoc does not know (libScePad, mono-ps4,
                        libSceGnm, ...) and for placeholder labels.

Usage:
    python3 scripts/audit_nids_libdoc.py
    python3 scripts/audit_nids_libdoc.py --json report.json
    python3 scripts/audit_nids_libdoc.py --libdoc data/libdoc/libkernel.sprx.json
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

# Official Orbis salt: raw 16 binary bytes (NOT the 32 ASCII characters).
SONY_SALT = bytes.fromhex("518D64A635DED8C1E6B039B1C3E55230")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "src" / "hle" / "nid_registry.cpp"
LIBDOC_DIR = REPO_ROOT / "data" / "libdoc"

# register_symbol(0x7501D612C26DA04EULL, "pthread_mutexattr_init", "libScePosix",
RE_DIRECT = re.compile(
    r"register_symbol\(\s*(0x[0-9A-Fa-f]+)(?:[uUlL])*\s*,\s*"
    r'"([^"]+)"\s*,\s*"([^"]+)"'
)
# kSyncStubs table entry: {0xB6D1CD7D4FAA0C15ULL, "pthread_mutex_init", "libScePosix"},
RE_STUB_TABLE = re.compile(
    r'\{\s*(0x[0-9A-Fa-f]+)(?:[uUlL])*\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,?\s*\}'
)
# kMonoStubs entry: {0x9BF8387D4F9CFB29ULL, "mono_class_get_field_from_name", MonoShape::Cookie, 0,
RE_MONO_TABLE = re.compile(
    r'\{\s*(0x[0-9A-Fa-f]+)(?:[uUlL])*\s*,\s*"([^"]+)"\s*,\s*MonoShape::\w+\s*,'
)
# Legacy NID-list stub loop: for (Nid stub_nid : { 0x....ULL, // note })
RE_STUB_BLOCK = re.compile(r"for\s*\(\s*Nid\s+stub_nid\s*:\s*\{(.*?)\}\s*\)", re.DOTALL)
RE_STUB_NID = re.compile(r"(0x[0-9A-Fa-f]+)(?:[uUlL])*\s*,?\s*(?://([^\n]*))?")
# Legacy alternate-NID form: register_symbol(0xNNN, "name", "module", handler);
# with a comment above it. Captured separately so we can report on the comment.
RE_COMMENT_SYMBOL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,63})\b")

# Labels that are deliberately not symbol names.
PLACEHOLDER_PREFIXES = ("unknown_",)
PLACEHOLDER_NAMES = {"sceKernelSyncStub"}

# Words that show up in "// NID: 0x..." notes and are never symbol names.
COMMENT_STOPWORDS = {
    "NID", "NIDs", "ULL", "NULL", "OK", "hash", "hashes", "hash-verified",
    "name", "names", "module", "modules", "NOTE", "TODO", "FIXME", "The",
    "this", "that", "guest", "libkernel", "libc", "mono", "sce", "Hash",
}

# Our shorthand module labels vs the library names ps4libdoc reports.
MODULE_ALIASES = {"libc": "libSceLibcInternal"}

# Sony 64-character base64 alphabet, used by the 11-char NID tags.
SONY_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"


def is_sony_tag(text: str) -> bool:
    """True for an 11-char Sony NID tag such as "HcurEgsfF80"."""
    return len(text) == 11 and all(c in SONY_ALPHA for c in text)

DEFAULT_GAME_DIR = (
    REPO_ROOT / "games" / "flappy_bird" / "ED1633-NPXX51362_00-TER1520000000000"
)


def compute_nid(name: str) -> int:
    """SHA-1(name + salt)[:8] as an unsigned little-endian uint64."""
    return struct.unpack("<Q", hashlib.sha1(name.encode("ascii") + SONY_SALT).digest()[:8])[0]


def is_placeholder(name: str) -> bool:
    return name in PLACEHOLDER_NAMES or name.startswith(PLACEHOLDER_PREFIXES)


def load_libdoc_json(paths: list[Path]) -> dict[int, list[dict]]:
    """hex NID -> [{module, library, name, type}] from ps4libdoc *.sprx.json dumps."""
    table: dict[int, list[dict]] = defaultdict(list)
    for path in paths:
        with open(path, "r", encoding="utf-8-sig") as handle:  # dumps carry a BOM
            doc = json.load(handle)
        for module in doc.get("modules", []):
            for library in module.get("libraries", []):
                for sym in library.get("symbols", []):
                    hex_id = str(sym.get("hex_id", "")).upper()
                    if len(hex_id) != 16:
                        continue
                    table[int(hex_id, 16)].append(
                        {
                            "module": module.get("name", ""),
                            "library": library.get("name", ""),
                            "name": sym.get("name", ""),
                            "source": path.name,
                        }
                    )
    return table


def load_known_names(path: Path) -> dict[int, list[str]]:
    """hex NID -> [names] from ps4libdoc's known_names.txt corpus."""
    table: dict[int, list[str]] = defaultdict(list)
    if not path.exists():
        return table
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            name = line.strip()
            if not name:
                continue
            table[compute_nid(name)].append(name)
    return table


def load_imported_nids(game_dir: Path) -> set[int]:
    """Every NID imported by the guest: eboot.bin plus every PRX we ship.

    Severity hinges on this. A wrong label on an imported NID hands the guest
    the wrong behaviour; the same wrong label on a NID nothing imports is only
    misleading in the trace.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from verify_nids import parse_eboot_imports
    except ImportError:
        return set()
    binaries = [game_dir / "eboot.bin"]
    for sub in ("sce_module", "Media/Modules"):
        binaries += sorted((game_dir / sub).glob("*.prx")) if (game_dir / sub).is_dir() else []
    imported: set[int] = set()
    for path in binaries:
        if not path.exists():
            continue
        try:
            table = parse_eboot_imports(path)
        except Exception:  # a malformed PRX must not abort the audit
            continue
        for entries in table.by_nid.values():
            for sym in entries:
                imported.add(sym.nid)
    return imported


def parse_registry(path: Path) -> list[dict]:
    """Every registration: direct calls, the SyncStub table, legacy stub loops."""
    content = path.read_text(encoding="utf-8")
    regs: list[dict] = []

    # The stub table first, then skip anything the direct regex would also match.
    stub_spans = []
    for match in RE_STUB_TABLE.finditer(content):
        nid, name, module = match.groups()
        regs.append(
            {
                "nid": int(nid, 16),
                "name": name,
                "module": module,
                "kind": "stub-table",
                "comment": None,
                "line": content.count("\n", 0, match.start()) + 1,
            }
        )
        stub_spans.append((match.start(), match.end()))

    for match in RE_MONO_TABLE.finditer(content):
        nid, name = match.groups()
        regs.append(
            {
                "nid": int(nid, 16),
                "name": name,
                "module": "mono-ps4",
                "kind": "mono-table",
                "comment": None,
                "line": content.count("\n", 0, match.start()) + 1,
            }
        )
        stub_spans.append((match.start(), match.end()))

    def in_stub_span(pos: int) -> bool:
        return any(s <= pos < e for s, e in stub_spans)

    for match in RE_DIRECT.finditer(content):
        if in_stub_span(match.start()):
            continue
        nid, name, module = match.groups()
        line = content.count("\n", 0, match.start()) + 1
        comment = ""
        head = content.rfind("\n", 0, max(0, match.start() - 1))
        for prev in content[max(0, head - 400) : head].splitlines():
            text = prev.strip()
            if text.startswith("//") and re.search(r"[A-Za-z_]", text):
                comment = text[2:].strip()
        regs.append(
            {
                "nid": int(nid, 16),
                "name": name,
                "module": module,
                "kind": "direct",
                "comment": comment or None,
                "line": line,
            }
        )

    for block in RE_STUB_BLOCK.finditer(content):
        for entry in RE_STUB_NID.finditer(block.group(1)):
            regs.append(
                {
                    "nid": int(entry.group(1), 16),
                    "name": "sceKernelSyncStub",
                    "module": "libkernel",
                    "kind": "stub-loop",
                    "comment": (entry.group(2) or "").strip() or None,
                    "line": content.count("\n", 0, block.start(1) + entry.start()) + 1,
                }
            )

    return regs


def audit(registry: Path, libdoc: dict, known: dict, imported: set[int] | None = None) -> dict:
    regs = parse_registry(registry)
    imported = imported or set()

    mislabels: list[dict] = []
    hash_mismatches: list[dict] = []
    module_drift: list[dict] = []
    confirmed: list[dict] = []
    placeholders: list[dict] = []
    comment_drift: list[dict] = []

    by_nid: dict[int, set[str]] = defaultdict(set)
    for reg in regs:
        by_nid[reg["nid"]].add(reg["name"])

    for reg in regs:
        truth = libdoc.get(reg["nid"])
        names = known.get(reg["nid"])
        reg["true_names"] = sorted(set(names)) if names else []
        reg["imported"] = reg["nid"] in imported
        if truth:
            reg["true_libraries"] = sorted({e["library"] for e in truth})

        # Authority 1+2: ps4libdoc names this NID something else.
        if reg["true_names"] and reg["name"] not in reg["true_names"]:
            if is_placeholder(reg["name"]):
                reg["resolved_from"] = reg["true_names"][0]
                placeholders.append(reg)
            else:
                mislabels.append(reg)
            continue

        # Placeholders are honest by construction, but a stale comment on a
        # placeholder that ps4libdoc CAN now name is worth surfacing.
        if is_placeholder(reg["name"]):
            if reg["comment"]:
                for cand in RE_COMMENT_SYMBOL.findall(reg["comment"]):
                    if cand in PLACEHOLDER_NAMES or cand in COMMENT_STOPWORDS:
                        continue
                    if cand in reg["true_names"]:
                        reg["resolved_from"] = cand
                        break
            placeholders.append(reg)
            continue

        # Authority 3: the name must hash to the NID.
        if compute_nid(reg["name"]) != reg["nid"]:
            hash_mismatches.append(dict(reg, expected=compute_nid(reg["name"])))
            continue

        # Right NID, right name, wrong module label (informational).
        lib_label = MODULE_ALIASES.get(reg["module"], reg["module"])
        if truth and lib_label not in reg["true_libraries"]:
            module_drift.append(reg)
            continue

        # A comment naming a symbol the NID does not hash to. An 11-char Sony
        # tag is the NID's own encoding, not a competing symbol name.
        if reg["comment"] and not reg["true_names"]:
            for cand in RE_COMMENT_SYMBOL.findall(reg["comment"]):
                if cand in PLACEHOLDER_NAMES or cand in COMMENT_STOPWORDS:
                    continue
                if is_sony_tag(cand) or compute_nid(cand) == reg["nid"]:
                    break
                comment_drift.append(dict(reg, comment_symbol=cand))
                break

        confirmed.append(reg)

    return {
        "registrations": regs,
        "total": len(regs),
        "mislabels": mislabels,
        "hash_mismatches": hash_mismatches,
        "module_drift": module_drift,
        "confirmed": confirmed,
        "placeholders": placeholders,
        "comment_drift": comment_drift,
        "duplicate_nids": {n: s for n, s in by_nid.items() if len(s) > 1},
    }


def print_report(result: dict, libdoc_files: list[Path], known_path: Path, registry: Path) -> None:
    print("=" * 92)
    print("NID AUDIT vs ps4libdoc")
    print("=" * 92)
    print(f"registry : {registry}")
    for path in libdoc_files:
        print(f"libdoc   : {path}")
    print(f"names    : {known_path}")
    bugs = [r for r in result["mislabels"] + result["hash_mismatches"] if r.get("imported")]
    stale = [
        r
        for r in result["mislabels"] + result["hash_mismatches"]
        if not r.get("imported")
    ]
    print(
        f"\nregistrations: {result['total']}   confirmed: {len(result['confirmed'])}   "
        f"placeholders: {len(result['placeholders'])}\n"
        f"BUGS (mislabeled + imported): {len(bugs)}   "
        f"cleanup (mislabeled, not imported): {len(stale)}\n"
        f"module drift: {len(result['module_drift'])}   "
        f"comment drift: {len(result['comment_drift'])}   "
        f"duplicate NIDs: {len(result['duplicate_nids'])}"
    )

    if result["mislabels"]:
        print("\n--- MISLABELED: ps4libdoc names this NID something else ---")
        print(f"  {'line':>6}  {'guest':>7}  {'registered as':30s} {'NID':20s} ps4libdoc says")
        print("  " + "-" * 92)
        for reg in result["mislabels"]:
            mark = "IMPORT" if reg.get("imported") else "-"
            print(
                f"  {reg['line']:>6}  {mark:>7}  {reg['name']:30s} 0x{reg['nid']:016X} "
                f"{', '.join(reg['true_names'])}"
            )
            if reg.get("true_libraries"):
                print(f"  {'':6}  {'':7}  {'':30s} {'':20s} library: {', '.join(reg['true_libraries'])}")

    if result["hash_mismatches"]:
        print("\n--- HASH MISMATCH: label does not hash to its NID, and ps4libdoc is silent ---")
        print(f"  {'line':>6}  {'guest':>7}  {'NID':20s} label / what that label really hashes to")
        print("  " + "-" * 92)
        for reg in result["hash_mismatches"]:
            mark = "IMPORT" if reg.get("imported") else "-"
            print(
                f"  {reg['line']:>6}  {mark:>7}  0x{reg['nid']:016X} "
                f"'{reg['name']}' -> 0x{reg['expected']:016X}  ({reg['kind']})"
            )

    if result["duplicate_nids"]:
        print("\n--- DUPLICATE NIDs: more than one name registered under the same NID ---")
        for nid, names in sorted(result["duplicate_nids"].items()):
            print(f"  0x{nid:016X}: {sorted(names)}")

    if result["module_drift"]:
        print("\n--- MODULE DRIFT: NID and name right, module label off ---")
        for reg in result["module_drift"]:
            print(
                f"  {reg['line']:>6}  {reg['name']:34s} [{reg['module']}] -> "
                f"{', '.join(reg['true_libraries'])}"
            )

    resolved = [r for r in result["placeholders"] if r.get("resolved_from")]
    if resolved:
        print("\n--- PLACEHOLDERS ps4libdoc CAN now name ---")
        for reg in resolved:
            print(
                f"  {reg['line']:>6}  0x{reg['nid']:016X} {reg['name']:28s} -> {reg['resolved_from']}"
            )

    if result["comment_drift"]:
        print("\n--- STALE COMMENTS: note names a symbol this NID does not hash to ---")
        for reg in result["comment_drift"]:
            print(
                f"  {reg['line']:>6}  0x{reg['nid']:016X} note mentions "
                f"'{reg['comment_symbol']}' but '{reg['name']}' does not hash to this NID"
            )

    print()
    bad = len(result["mislabels"]) + len(result["hash_mismatches"])
    if not bad:
        print("[SUCCESS] No mislabeled NIDs: every registration agrees with ps4libdoc")
        print("          or with the Orbis NID hash.")
    else:
        print(f"[FAIL] {bad} mislabeled registration(s) -- handler labels in the trace")
        print("       are untrustworthy until these are fixed.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit HLE NID registrations vs ps4libdoc")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--libdoc", action="append", default=None,
                        help="ps4libdoc *.sprx.json (repeatable). Default: all in data/libdoc.")
    parser.add_argument("--known-names", default=str(LIBDOC_DIR / "known_names.txt"))
    parser.add_argument("--game-dir", default=str(DEFAULT_GAME_DIR),
                        help="Guest dir whose imports mark a mislabel as a live bug.")
    parser.add_argument("--no-imports", action="store_true",
                        help="Skip the guest import cross-reference.")
    parser.add_argument("--json", help="Write the full audit result as JSON.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    registry = Path(args.registry)
    if not registry.exists():
        print(f"[ERROR] registry not found: {registry}", file=sys.stderr)
        return 2

    libdoc_files = [Path(p) for p in args.libdoc] if args.libdoc else sorted(
        LIBDOC_DIR.glob("*.sprx.json")
    )
    missing = [p for p in libdoc_files if not p.exists()]
    if missing:
        for path in missing:
            print(f"[ERROR] libdoc file not found: {path}", file=sys.stderr)
        return 2
    if not libdoc_files:
        print(f"[WARN] no *.sprx.json in {LIBDOC_DIR}; library attribution disabled.",
              file=sys.stderr)

    known_path = Path(args.known_names)
    known = load_known_names(known_path)
    if not known:
        print(f"[WARN] no known_names.txt at {known_path}; name authority is hash-only.",
              file=sys.stderr)

    imported: set[int] = set()
    if not args.no_imports:
        imported = load_imported_nids(Path(args.game_dir))
        if not imported:
            print(f"[WARN] no guest imports parsed from {args.game_dir}; "
                  "severity will treat every finding as cleanup.", file=sys.stderr)

    result = audit(registry, load_libdoc_json(libdoc_files), known, imported)
    if not args.quiet:
        print_report(result, libdoc_files, known_path, registry)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    live = [r for r in result["mislabels"] + result["hash_mismatches"] if r.get("imported")]
    return 1 if live else 0


if __name__ == "__main__":
    sys.exit(main())