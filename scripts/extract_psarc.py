#!/usr/bin/env python3
"""Extract Sony PSARC (PlayStation Archive) files."""

import os
import sys
import struct
import zlib
from pathlib import Path


def extract_psarc(psarc_path: str, out_dir: str):
    psarc_file = Path(psarc_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(psarc_file, "rb") as f:
        header = struct.unpack(">4sL4sLLLLL", f.read(32))
        magic, version, compression, toc_size, entry_size, n_entries, block_size, flags = header
        if magic != b"PSAR":
            raise ValueError(f"Not a PSARC file (magic={magic})")

        toc_data = f.read(toc_size - 32)
        entries = []
        pos = 0
        for _ in range(n_entries):
            d = toc_data[pos : pos + entry_size]
            pos += entry_size
            zindex = struct.unpack(">I", d[16:20])[0]
            length = struct.unpack(">Q", b"\x00" * 3 + d[20:25])[0]
            offset = struct.unpack(">Q", b"\x00" * 3 + d[25:30])[0]
            entries.append({"zindex": zindex, "length": length, "offset": offset})

        zlengths = []
        num_zlengths = (len(toc_data) - pos) // 2
        for _ in range(num_zlengths):
            zlengths.append(struct.unpack(">H", toc_data[pos : pos + 2])[0])
            pos += 2

        def read_entry_data(entry):
            zl_list = zlengths[entry["zindex"] :]
            f.seek(entry["offset"])
            out = bytearray()
            idx = 0
            while len(out) < entry["length"]:
                zl = zl_list[idx]
                if zl == 0:
                    out.extend(f.read(min(block_size, entry["length"] - len(out))))
                else:
                    chunk = f.read(zl)
                    try:
                        decomp = zlib.decompress(chunk)
                        out.extend(decomp)
                    except Exception:
                        out.extend(chunk)
                idx += 1
            return bytes(out[: entry["length"]])

        # Entry 0 is file listing manifest
        manifest_data = read_entry_data(entries[0])
        filepaths = manifest_data.decode("utf-8").strip().splitlines()

        print(f"Extracting {len(filepaths)} files from {psarc_file.name} to {out_path}...")
        for entry, rel_path in zip(entries[1:], filepaths):
            target_path = out_path / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            data = read_entry_data(entry)
            with open(target_path, "wb") as out_f:
                out_f.write(data)
            print(f"  Extracted: {rel_path} ({len(data)} bytes)")

    print(f"Extraction complete! {len(filepaths)} files written.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <archive.psarc> <output_dir>")
        sys.exit(1)
    extract_psarc(sys.argv[1], sys.argv[2])
