#!/usr/bin/env python3
"""Convert binary P6 PPM (8-bit RGB) to PNG without third-party packages."""
import argparse
from pathlib import Path
import re
import struct
import zlib


def convert(source, destination):
    data = Path(source).read_bytes()
    pos = 0

    def token():
        nonlocal pos
        while pos < len(data):
            if data[pos:pos + 1] == b'#':
                end = data.find(b'\n', pos)
                pos = len(data) if end < 0 else end + 1
            elif data[pos] in b' \t\r\n\v\f':
                pos += 1
            else:
                break
        match = re.match(rb'[^\s#]+', data[pos:])
        if not match:
            raise ValueError('truncated PPM header')
        pos += match.end()
        return match.group()

    if token() != b'P6':
        raise ValueError('expected binary P6 PPM')
    width, height, maximum = int(token()), int(token()), int(token())
    if width <= 0 or height <= 0 or maximum != 255:
        raise ValueError('expected positive dimensions and maxval 255')
    if pos >= len(data) or data[pos] not in b' \t\r\n\v\f':
        raise ValueError('missing raster separator')
    pos += 2 if data[pos:pos + 2] == b'\r\n' else 1
    raster = data[pos:]
    if len(raster) != width * height * 3:
        raise ValueError('PPM raster length does not match dimensions')

    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))

    stride = width * 3
    scanlines = b''.join(b'\0' + raster[y * stride:(y + 1) * stride] for y in range(height))
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b'')
    Path(destination).write_bytes(png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', nargs='?', type=Path)
    args = parser.parse_args()
    try:
        convert(args.source, args.destination or args.source.with_suffix('.png'))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
