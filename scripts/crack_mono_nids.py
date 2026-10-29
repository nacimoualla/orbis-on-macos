#!/usr/bin/env python3
"""Resolve Mono NIDs (mono_*) against the official Mono embedding API.

The HLE registry labels four Mono NIDs that the guest calls constantly:

    0x47F74CE8E2982CC4  mono_get_corlib
    0x1147722929EB0A4C  mono_class_is_subclass_of
    0xA6B7BD0639320E1B  mono_runtime_invoke
    0x4CCB2C2760985FAA  mono_class_get_parent

Those labels are only as good as the evidence behind them, and a Mono label
that is off by one letter silently describes the wrong behaviour. This script
proves them the same way everything else is proven: hash the name against the
Orbis salt and compare with the NID.

Candidate sources, in order:
  1. data/libdoc/mono_embedding_api.txt -- every mono_* identifier in the
     official Mono headers (2.8k symbols). Cached so this runs offline.
     --refresh re-scrapes from github.com/mono/mono.
  2. Strings mined from the guest's own binaries (eboot.bin, mono-ps4.prx,
     MonoAssembliesPS4.prx): catches PS4-specific Mono additions that never
     shipped upstream.
  3. ps4libdoc known_names.txt, for the same reason at corpus scale.

Usage:
    python3 scripts/crack_mono_nids.py
    python3 scripts/crack_mono_nids.py --nid 0x47f74ce8e2982cc4
    python3 scripts/crack_mono_nids.py --refresh
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import struct
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBDOC_DIR = REPO_ROOT / "data" / "libdoc"
MONO_LIST = LIBDOC_DIR / "mono_embedding_api.txt"
KNOWN_NAMES = LIBDOC_DIR / "known_names.txt"
MONO_HEADER_CACHE = Path(
    "/private/var/folders/1m/xxd5xm911wl6kj07r0dcjpy80000gn/T/opencode/mono_hdrs"
)

SONY_SALT = bytes.fromhex("518D64A635DED8C1E6B039B1C3E55230")
GAME_DIR = REPO_ROOT / "games" / "flappy_bird" / "ED1633-NPXX51362_00-TER1520000000000"
GUEST_BINARIES = [
    "eboot.bin",
    "Media/Modules/mono-ps4.prx",
    "Media/Modules/MonoAssembliesPS4.prx",
]

# The four Mono NIDs this was built to crack, with the label under test.
TARGETS = {
    0x47F74CE8E2982CC4: "mono_get_corlib",
    0x1147722929EB0A4C: "mono_class_is_subclass_of",
    0xA6B7BD0639320E1B: "mono_runtime_invoke",
    0x4CCB2C2760985FAA: "mono_class_get_parent",
    # Swept opportunistically from earlier turns.
    0x2C5314E29C000914: "mono_class_get_methods",
}

RE_IDENT = re.compile(r"\b(mono_[A-Za-z0-9_]{2,60})\b")
MONO_TREE_API = "https://api.github.com/repos/mono/mono/contents/{subdir}?ref=main"
MONO_RAW = "https://raw.githubusercontent.com/mono/mono/main/{path}"


def compute_nid(name: str) -> int:
    return struct.unpack("<Q", hashlib.sha1(name.encode("ascii") + SONY_SALT).digest()[:8])[0]


def load_cached_api() -> set[str]:
    if not MONO_LIST.exists():
        return set()
    out = set()
    for line in MONO_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def refresh_api(cache_dir: Path = MONO_HEADER_CACHE) -> set[str]:
    """Re-scrape every mono_*/jit header Mono ships. Needs network."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbols: set[str] = set()
    paths: list[str] = []
    for subdir in ("mono/metadata", "mono/jit"):
        listing = subprocess.run(
            ["curl", "-s", "-m", "30", MONO_TREE_API.format(subdir=subdir)],
            capture_output=True, text=True, check=True,
        ).stdout
        try:
            import json
            for entry in json.loads(listing):
                if entry["name"].endswith((".h", ".c")):
                    paths.append(f"{subdir}/{entry['name']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] could not list {subdir}: {exc}", file=sys.stderr)
    for path in sorted(set(paths)):
        out = cache_dir / path.replace("/", "__")
        if not (out.exists() and out.stat().st_size > 100):
            subprocess.run(
                ["curl", "-s", "-m", "25", "-o", str(out), MONO_RAW.format(path=path)],
                check=False,
            )
        if out.exists() and out.stat().st_size > 100:
            symbols.update(RE_IDENT.findall(out.read_text(errors="ignore")))
    if symbols:
        MONO_LIST.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Mono embedding API symbol list, scraped from the official Mono headers\n"
            "# (mono/metadata/*.h + mono/jit/*.h at github.com/mono/mono, branch main).\n"
            "# Used by scripts/crack_mono_nids.py to resolve mono_* NIDs by hashing each\n"
            "# name against the Orbis salt. Regenerate with: crack_mono_nids.py --refresh\n"
        )
        MONO_LIST.write_text(header + "\n".join(sorted(symbols)) + "\n", encoding="utf-8")
    return symbols


def mine_guest_symbols() -> set[str]:
    """Identifier-looking strings from the guest's own Mono binaries."""
    found: set[str] = set()
    ident = re.compile(rb"[A-Za-z_][A-Za-z0-9_]{3,80}")
    for rel in GUEST_BINARIES:
        path = GAME_DIR / rel
        if not path.exists():
            continue
        blob = path.read_bytes()
        for match in ident.finditer(blob):
            found.add(match.group(0).decode("ascii", "ignore"))
    return found


def load_known_names() -> dict[int, list[str]]:
    table: dict[int, list[str]] = defaultdict(list)
    if not KNOWN_NAMES.exists():
        return table
    for line in KNOWN_NAMES.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()
        if name:
            table[compute_nid(name)].append(name)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve Mono NIDs via the Mono embedding API")
    parser.add_argument("--nid", action="append", default=[],
                        help="Extra NID(s) to resolve, e.g. 0x47f74ce8e2982cc4")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-scrape the Mono headers (needs network).")
    parser.add_argument("--all", action="store_true",
                        help="List every NID the guest imports that matches a Mono symbol.")
    args = parser.parse_args()

    if args.refresh:
        api = refresh_api()
        print(f"[INFO] scraped {len(api)} mono_* symbols -> {MONO_LIST}")
    else:
        api = load_cached_api()
        if not api:
            print(f"[WARN] no cached symbol list at {MONO_LIST}; run with --refresh",
                  file=sys.stderr)

    print(f"Mono embedding API symbols : {len(api)}")

    mined = {s for s in mine_guest_symbols() if s.startswith("mono_")}
    print(f"mono_* strings in guest    : {len(mined)}")

    known = load_known_names()
    print(f"ps4libdoc known_names      : {len(known)} NIDs")

    # Hash everything we know of once, then answer from a reverse table.
    reverse: dict[int, set[str]] = defaultdict(set)
    universe = set(api) | mined
    for name in universe:
        reverse[compute_nid(name)].add(name)
    for nid, names in known.items():
        for name in names:
            if name.startswith("mono_"):
                reverse[nid].add(name)
    print(f"total hashed candidates    : {len(universe)}")

    targets = dict(TARGETS)
    for extra in args.nid:
        targets.setdefault(int(extra, 16), None)

    print("\n" + "=" * 78)
    print(f"{'NID':20s} {'registry label':32s} verdict")
    print("=" * 78)
    failures = 0
    for nid, label in sorted(targets.items()):
        hits = sorted(reverse.get(nid, []))
        if not hits:
            verdict = "UNRESOLVED"
            failures += 1
        elif label is None:
            verdict = f"cracked: {hits[0]}"
        elif label in hits:
            verdict = "HASH-VERIFIED"
        else:
            verdict = f"MISMATCH: registry says {label}"
            failures += 1
        print(f"0x{nid:016X}   {str(label):32s} {verdict}")
        if len(hits) > 1:
            print(f"{'':20s} {'':32s} also: {', '.join(h for h in hits if h != label)}")

    if args.all:
        print("\n--- every guest-imported NID that matches a Mono symbol ---")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from verify_nids import parse_eboot_imports
            imported: dict[int, str] = {}
            for rel in GUEST_BINARIES:
                path = GAME_DIR / rel
                if not path.exists():
                    continue
                try:
                    table = parse_eboot_imports(path)
                except Exception:  # noqa: BLE001
                    continue
                for entries in table.by_nid.values():
                    for sym in entries:
                        imported[sym.nid] = rel
        except ImportError:
            imported = {}
        for nid, names in sorted(reverse.items()):
            if nid in imported:
                print(f"0x{nid:016X}  {sorted(names)[0]:40s} ({imported[nid]})")

    print()
    if failures:
        print(f"[WARN] {failures} target(s) not hash-verified -- do not trust those labels.")
    else:
        print("[SUCCESS] Every target Mono label is hash-verified against the Orbis salt.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())