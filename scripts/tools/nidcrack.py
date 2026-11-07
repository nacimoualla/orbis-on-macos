#!/usr/bin/env python3
"""Hash symbol names, or search a word list for a hexadecimal/encoded Orbis NID."""
import argparse
import base64
import hashlib
import sys

SUFFIX = bytes.fromhex('518D64A635DED8C1E6B039B1C3E55230')


def nid(name):
    digest = hashlib.sha1(name.encode('utf-8') + SUFFIX).digest()[:8]
    return int.from_bytes(digest, 'little')


def encoded(value):
    return base64.b64encode(value.to_bytes(8, 'big')).decode().rstrip('=').replace('/', '-')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='*', help='symbols to hash (stdin if omitted)')
    parser.add_argument('--target', help='hexadecimal or 11-character encoded NID to crack')
    parser.add_argument('--wordlist', type=argparse.FileType('r'), help='one candidate name per line')
    args = parser.parse_args()
    names = args.names if args.names else (args.wordlist or sys.stdin)
    target = args.target
    if target:
        try:
            target = int(target, 16)
        except ValueError:
            try:
                raw = base64.b64decode(target.replace('-', '/') + '=', validate=True)
                if len(raw) != 8:
                    raise ValueError('encoded NID must contain eight bytes')
                target = int.from_bytes(raw, 'big')
            except ValueError as error:
                parser.error(str(error))
    found = False
    for line in names:
        name = line.strip()
        if not name:
            continue
        value = nid(name)
        if target is None or target == value:
            print(f'{value:016x}\t{encoded(value)}\t{name}')
            found = True
    return 0 if found else 1


if __name__ == '__main__':
    raise SystemExit(main())
