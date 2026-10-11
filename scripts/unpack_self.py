#!/usr/bin/env python3
"""
Unpack Sony PlayStation 4 SELF (Signed/Fake ELF) into standard ELF64 binary.
"""
import sys
import struct
from pathlib import Path

def unpack_self(self_path: Path, elf_path: Path):
    with open(self_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'O\x15=\x1d':
            raise ValueError(f"Not a SELF file (magic: {magic.hex()})")
        
        f.seek(0x18)
        entry_cnt = struct.unpack('<H', f.read(2))[0]
        
        entries = []
        f.seek(0x20)
        for i in range(entry_cnt):
            flags, offset, comp_sz, uncomp_sz = struct.unpack('<QQQQ', f.read(32))
            entries.append((flags, offset, comp_sz, uncomp_sz))
            
        elf_hdr_off = 0x20 + entry_cnt * 32
        f.seek(elf_hdr_off)
        ehdr = f.read(64)
        if ehdr[:4] != b'\x7fELF':
            raise ValueError(f"Invalid inner ELF magic: {ehdr[:4]}")
            
        e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum = struct.unpack('<QQIHHH', ehdr[32:58])
        
        phs = []
        f.seek(elf_hdr_off + e_phoff)
        for i in range(e_phnum):
            ph_raw = f.read(56)
            p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack('<IIQQQQQQ', ph_raw)
            phs.append((i, p_type, p_flags, p_offset, p_vaddr, p_filesz, p_memsz))
            
        max_end = max(p_offset + p_filesz for _, _, _, p_offset, _, p_filesz, _ in phs)
        elf_buf = bytearray(max_end)
        
        # Write ELF Header + Program Headers
        f.seek(elf_hdr_off)
        hdr_total = e_phoff + e_phnum * 56
        elf_buf[:hdr_total] = f.read(hdr_total)
        
        # Filter loadable / data program headers in order
        target_types = {0x1, 0x61000010, 0x61000000, 0x6fffff00}
        target_phs = [p for p in phs if p[1] in target_types]
        
        # In uncompressed SELF, segment data entries are odd entries (1, 3, 5, 7, 9...)
        data_entries = [e for i, e in enumerate(entries) if i % 2 == 1]
        
        print(f"[self2elf] Extracted {len(target_phs)} target segments, mapping {len(data_entries)} data chunks...")
        
        for idx in range(min(len(target_phs), len(data_entries))):
            p = target_phs[idx]
            e = data_entries[idx]
            p_offset = p[3]
            comp_sz = e[2]
            f.seek(e[1])
            data = f.read(comp_sz)
            elf_buf[p_offset : p_offset + comp_sz] = data
            print(f"  Segment {p[0]} (Type 0x{p[1]:x}): copied 0x{comp_sz:x} bytes -> ELF offset 0x{p_offset:x}")
            
    with open(elf_path, 'wb') as out_f:
        out_f.write(elf_buf)
    print(f"[self2elf] Reconstructed {elf_path} ({len(elf_buf)} bytes)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: self2elf.py <input.bin> <output.elf>")
        sys.exit(1)
    unpack_self(Path(sys.argv[1]), Path(sys.argv[2]))
