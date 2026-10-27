#!/usr/bin/env python3
"""Summarise native-vs-emulator differences per instruction case."""
import re, sys
names = re.findall(r'\{"([^"]+)"', open(sys.argv[1]).read())
def load(p):
    out = {}
    for line in open(p):
        parts = line.split()
        if not parts or not re.match(r'^\d+\.\d+$', parts[0]): continue
        out[parts[0]] = dict(x.split('=', 1) for x in parts[1:])
    return out
nat = load(sys.argv[2])
bad = 0
for mode, path in zip(("interp", "jit"), sys.argv[3:]):
    emu = load(path); per = {}
    for k, a in nat.items():
        b = emu.get(k)
        if b is None: per.setdefault(int(k.split('.')[0]), set()).add('missing'); continue
        d = [f for f in a if a[f] != b.get(f)]
        if d: per.setdefault(int(k.split('.')[0]), set()).update(d)
    print(f"{mode}: {len(nat) - sum(1 for k in nat if k.split('.')[0] in map(str, per))} / {len(nat)} trials match")
    for c in sorted(per):
        print(f"  case {c:3} {names[c]:28} differs in {sorted(per[c])}")
    bad += len(per)
sys.exit(1 if bad else 0)
