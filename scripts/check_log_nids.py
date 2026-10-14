#!/usr/bin/env python3
"""Check '[HLE CALL] <label> (NID=0x...' lines in a ps4-runner log against the
Orbis NID hash: first 8 bytes of SHA-1(name + salt), read little-endian.

A label whose hash differs from its NID is a misregistered handler.

usage: check_log_nids.py LOG [names.txt]
  names.txt (optional): one candidate symbol name per line; used to try to
  resolve the real name of each mismatched NID.
"""
import collections
import hashlib
import re
import sys

SALT = bytes.fromhex("518D64A635DED8C1E6B039B1C3E55230")
CALL = re.compile(r"\[HLE CALL\] (\S+) \(NID=0x([0-9a-fA-F]+), caller RIP=0x([0-9a-fA-F]+)")


def nid(name: str) -> int:
    return int.from_bytes(hashlib.sha1(name.encode() + SALT).digest()[:8], "little")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    seen = collections.OrderedDict()  # (label, nid) -> [count, first_caller]
    with open(sys.argv[1], errors="replace") as f:
        for line in f:
            m = CALL.search(line)
            if not m:
                continue
            key = (m.group(1), int(m.group(2), 16))
            if key not in seen:
                seen[key] = [0, m.group(3)]
            seen[key][0] += 1

    candidates = {}
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            for name in (l.strip() for l in f):
                if name:
                    candidates.setdefault(nid(name), []).append(name)

    bad = [(k, v) for k, v in seen.items() if nid(k[0]) != k[1]]
    print(f"{len(seen)} distinct label/NID pairs, {len(bad)} where hash(label) != NID")
    for (label, n), (count, caller) in sorted(bad, key=lambda kv: -kv[1][0]):
        real = candidates.get(n)
        print(f"  {label:36s} 0x{n:016X}  x{count:<4d} first caller 0x{caller}"
              + (f"  -> really {real}" if real else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
