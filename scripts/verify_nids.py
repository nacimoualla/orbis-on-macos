#!/usr/bin/env python3
"""
PS4 Sony NID Verification & Import Scanner Tool

Milestone 1 Implementation:
1. Compute 64-bit Sony NIDs using SHA-1(symbol_name + raw 16 bytes "518D64A635DED8C1E6B039B1C3E55230")
   formatted as little-endian 8 bytes (<Q), and Sony 11-character base64 tag encoding/decoding.
2. Validate against known test vector: strcmp <-> 0x3af6f675224e02e1 <-> Ovb2dSJOAuE.
3. Scan src/hle/nid_registry.cpp to compare every registered symbol name against its registered NID,
   reporting all matches, mismatches, and duplicates. For symbol names that are 11-char Sony tags,
   decodes the tag rather than hashing the tag string as a function name.
4. Scan against PS4 ELF/PRX eboot import tables (eboot.bin and eboot.elf) using pure Python
   with zero external dependencies (no pyelftools, no readelf).
   - Unpacks in-memory SELF (.bin) or parses raw ELF (.elf).
   - Parses PT_DYNAMIC, DT_SCE_STRTAB, DT_SCE_SYMTAB, DT_SCE_JMPREL, DT_SCE_PLTRELSZ, DT_SCE_RELA.
   - Decodes imported symbols (<11-char-tag>#<lib>#<mod>) to 64-bit NIDs.
   - Cross-references against src/hle/nid_registry.cpp and reports imported, resolved, and unresolved NIDs.
5. Clean execution with 0 exit code on audit and test vector pass.
"""

import sys
import os
import re
import struct
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

# Official Sony salt: raw 16 binary bytes (NOT 32 ASCII characters)
SONY_SALT = bytes.fromhex("518D64A635DED8C1E6B039B1C3E55230")

# Sony 64-character base64 alphabet
SONY_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"


def compute_sony_nid(symbol_name: str) -> int:
    """
    Compute 64-bit Sony NID using SHA-1(symbol_name + raw 16-byte Sony salt)
    unpacked as unsigned little-endian 64-bit integer (<Q).
    """
    h = hashlib.sha1(symbol_name.encode("ascii") + SONY_SALT).digest()
    return struct.unpack("<Q", h[:8])[0]


def encode_sony_tag(nid: int) -> str:
    """
    Encode a 64-bit unsigned integer NID into an 11-character Sony base64 tag.
    The 64-bit value is shifted left by 2 bits to form 66 bits, then divided into
    11 chunks of 6 bits each, mapped to SONY_ALPHA.
    """
    v = (nid & 0xFFFFFFFFFFFFFFFF) << 2
    chars = []
    for i in range(11):
        shift = 60 - (i * 6)
        chars.append(SONY_ALPHA[(v >> shift) & 0x3F])
    return "".join(chars)


def decode_sony_tag(tag: str) -> int:
    """
    Decode an 11-character Sony base64 tag into a 64-bit unsigned integer NID.
    If the tag contains a '#' suffix (e.g., 'tag#lib#mod'), only the tag portion is used.
    """
    if "#" in tag:
        tag = tag.split("#", 1)[0]
    tag = tag.strip()
    if len(tag) < 11:
        raise ValueError(f"Tag too short to decode: '{tag}' (expected at least 11 chars)")
    val = 0
    for c in tag[:11]:
        idx = SONY_ALPHA.find(c)
        if idx == -1:
            raise ValueError(f"Invalid character '{c}' in Sony tag '{tag}'")
        val = (val << 6) | idx
    return (val >> 2) & 0xFFFFFFFFFFFFFFFF


def is_sony_tag(s: str) -> bool:
    """
    Determine if a string is a valid 11-character Sony tag.
    A valid tag has length 11, all characters in SONY_ALPHA, and preserves
    bijective round-trip identity: encode_sony_tag(decode_sony_tag(s)) == s.
    """
    if not isinstance(s, str) or len(s) != 11:
        return False
    if any(c not in SONY_ALPHA for c in s):
        return False
    try:
        return encode_sony_tag(decode_sony_tag(s)) == s
    except Exception:
        return False


def test_vector():
    """
    Validate Sony NID calculation, base64 tag encoding, and decoding against
    official known test vectors (including strcmp <-> 0x3af6f675224e02e1 <-> Ovb2dSJOAuE).
    """
    test_vectors = [
        ("strcmp", 0x3AF6F675224E02E1, "Ovb2dSJOAuE"),
        ("sceFiosInitialize", 0xC00299FDD7ADFB2A, "wAKZ-det+yo"),
        ("sceUserServiceGetEvent", 0xC87D7B43A356B558, "yH17Q6NWtVg"),
        ("sceVideoOutOpen", 0x529DFA3D393AF3B1, "Up36PTk687E"),
        ("sceKernelClockGettime", 0x4018BB1C22B4DE1C, "QBi7HCK03hw"),
    ]
    print("=" * 78)
    print("PS4 SONY NID TEST VECTORS")
    print("=" * 78)
    for sym_name, expected_nid, expected_tag in test_vectors:
        actual_nid = compute_sony_nid(sym_name)
        actual_tag = encode_sony_tag(actual_nid)
        decoded_nid = decode_sony_tag(expected_tag)
        decoded_with_suffix = decode_sony_tag(f"{expected_tag}#l#l")

        assert actual_nid == expected_nid, (
            f"NID mismatch for '{sym_name}': 0x{actual_nid:016X} != 0x{expected_nid:016X}"
        )
        assert actual_tag == expected_tag, (
            f"Tag mismatch for '{sym_name}': '{actual_tag}' != '{expected_tag}'"
        )
        assert decoded_nid == expected_nid, (
            f"Decode mismatch for '{expected_tag}': 0x{decoded_nid:016X} != 0x{expected_nid:016X}"
        )
        assert decoded_with_suffix == expected_nid, (
            f"Suffix decode mismatch for '{expected_tag}#l#l': 0x{decoded_with_suffix:016X} != 0x{expected_nid:016X}"
        )
        print(f"  [PASS] {sym_name:25s} <-> 0x{actual_nid:016X} <-> {actual_tag}")
    print("[SUCCESS] All test vectors validated without assertion failure.\n")


def audit_nid_registry(registry_path: str, verbose: bool = False) -> dict:
    """
    Scan src/hle/nid_registry.cpp to compare every registered symbol name against
    its registered NID, reporting matches, mismatches, and duplicates.
    For symbol names that are 11-char Sony tags, decodes the tag rather than hashing
    the tag string as a function name.
    """
    print("=" * 78)
    print(f"AUDITING HLE NID REGISTRY: {registry_path}")
    print("=" * 78)

    with open(registry_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r'register_symbol\(\s*(0x[0-9A-Fa-f]+[uU]?[lL]{0,2}|\d+[uU]?[lL]{0,2})\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"'
    )
    raw_matches = pattern.findall(content)
    print(f"Total symbol registration calls found: {len(raw_matches)}")

    matches = []
    mismatches = []
    nid_to_symbols = defaultdict(list)
    sym_to_registrations = defaultdict(list)

    for nid_str, name, mod in raw_matches:
        clean_nid_str = nid_str.rstrip("uUlL")
        reg_nid = int(clean_nid_str, 16 if clean_nid_str.startswith("0x") else 10)

        nid_to_symbols[reg_nid].append((name, mod))
        sym_to_registrations[name].append((reg_nid, mod))

        if is_sony_tag(name):
            expected_nid = decode_sony_tag(name)
            is_tag = True
        else:
            expected_nid = compute_sony_nid(name)
            is_tag = False

        if reg_nid == expected_nid:
            matches.append({
                "name": name,
                "module": mod,
                "nid": reg_nid,
                "tag": encode_sony_tag(reg_nid),
                "is_tag": is_tag,
            })
        else:
            mismatches.append({
                "name": name,
                "module": mod,
                "registered_nid": reg_nid,
                "registered_tag": encode_sony_tag(reg_nid),
                "expected_nid": expected_nid,
                "expected_tag": encode_sony_tag(expected_nid),
                "is_tag": is_tag,
            })

    duplicate_nids = {nid: syms for nid, syms in nid_to_symbols.items() if len(syms) > 1}
    duplicate_symbols = {sym: nids for sym, nids in sym_to_registrations.items() if len(nids) > 1}

    print(f"Exact Sony NID matches:     {len(matches)}")
    print(f"Mismatched registrations:   {len(mismatches)}")
    print(f"Duplicate registered NIDs:  {len(duplicate_nids)}")
    print(f"Duplicate symbol names:     {len(duplicate_symbols)}")

    if duplicate_nids:
        print("\n--- DUPLICATE REGISTERED NIDS ---")
        for nid, syms in duplicate_nids.items():
            sym_list = ", ".join(f"'{name}' [{mod}]" for name, mod in syms)
            print(f"  0x{nid:016X} ({encode_sony_tag(nid)}): {sym_list}")

    if duplicate_symbols:
        print("\n--- DUPLICATE SYMBOL NAMES ---")
        for sym, reg_list in sorted(duplicate_symbols.items()):
            nids_str = ", ".join(f"0x{r[0]:016X} [{r[1]}]" for r in reg_list)
            print(f"  {sym:30s} -> {nids_str}")

    if mismatches:
        print(f"\n--- MISMATCHED REGISTRATIONS ({len(mismatches)}) ---")
        print(f"  {'Symbol Name':34s} {'Module':18s} {'Registered NID (Tag)':32s} {'Expected NID (Tag)':32s}")
        print("  " + "-" * 118)
        for m in mismatches:
            reg_info = f"0x{m['registered_nid']:016X} ({m['registered_tag']})"
            exp_info = f"0x{m['expected_nid']:016X} ({m['expected_tag']})"
            print(f"  {m['name']:34s} {m['module']:18s} {reg_info:32s} {exp_info:32s}")
    else:
        print("\n[SUCCESS] All registered symbols match calculated Sony NIDs exactly!")

    print()
    return {
        "total": len(raw_matches),
        "matches": matches,
        "mismatches": mismatches,
        "duplicate_nids": duplicate_nids,
        "duplicate_symbols": duplicate_symbols,
        "nid_to_symbols": nid_to_symbols,
    }


def unpack_self(data: bytes) -> bytes:
    """
    Unpack a Sony PlayStation 4 SELF container into standard ELF64 binary bytes in memory.
    If data does not begin with SELF magic (0x1D3D154F), returns data as-is.
    """
    if len(data) < 0x20 or data[:4] != b"O\x15=\x1d":
        return data

    entry_cnt = struct.unpack("<H", data[0x18:0x1A])[0]
    entries = []
    for i in range(entry_cnt):
        off = 0x20 + i * 32
        flags, offset, comp_sz, uncomp_sz = struct.unpack("<QQQQ", data[off : off + 32])
        entries.append((flags, offset, comp_sz, uncomp_sz))

    elf_hdr_off = 0x20 + entry_cnt * 32
    if elf_hdr_off + 64 > len(data):
        raise ValueError("Corrupt SELF container: header offset beyond EOF")

    ehdr = data[elf_hdr_off : elf_hdr_off + 64]
    if ehdr[:4] != b"\x7fELF":
        raise ValueError(f"Invalid inner ELF magic in SELF: {ehdr[:4]!r}")

    e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum = struct.unpack("<QQIHHH", ehdr[32:58])

    phs = []
    for i in range(e_phnum):
        ph_raw = data[elf_hdr_off + e_phoff + i * 56 : elf_hdr_off + e_phoff + (i + 1) * 56]
        p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack(
            "<IIQQQQQQ", ph_raw
        )
        phs.append((i, p_type, p_flags, p_offset, p_vaddr, p_filesz, p_memsz))

    max_end = max(p_offset + p_filesz for _, _, _, p_offset, _, p_filesz, _ in phs)
    elf_buf = bytearray(max_end)
    hdr_total = e_phoff + e_phnum * 56
    elf_buf[:hdr_total] = data[elf_hdr_off : elf_hdr_off + hdr_total]

    target_types = {0x1, 0x61000010, 0x61000000, 0x6fffff00}
    target_phs = [p for p in phs if p[1] in target_types]
    data_entries = [e for i, e in enumerate(entries) if i % 2 == 1]

    for idx in range(min(len(target_phs), len(data_entries))):
        p = target_phs[idx]
        e = data_entries[idx]
        p_offset = p[3]
        comp_sz = e[2]
        elf_buf[p_offset : p_offset + comp_sz] = data[e[1] : e[1] + comp_sz]

    return bytes(elf_buf)


class ImportedSymbol(int):
    """
    Representation of an imported symbol NID (subclass of int),
    supporting numeric integer comparisons, dict-like indexing, and metadata attributes.
    """
    def __new__(
        cls,
        nid: int,
        raw_name: str = "",
        tag: str = "",
        lib: str = "",
        mod: str = "",
        reloc_type: int = 0,
        offset: int = 0,
        section: str = "",
    ):
        obj = super().__new__(cls, nid)
        obj.nid = nid
        obj.raw_name = raw_name
        obj.tag = tag
        obj.lib = lib
        obj.mod = mod
        obj.reloc_type = reloc_type
        obj.offset = offset
        obj.section = section
        return obj

    def __getitem__(self, key: str):
        if key == "nid": return self.nid
        if key in ("name", "raw_name"): return self.raw_name
        if key == "tag": return self.tag
        if key == "lib": return self.lib
        if key == "mod": return self.mod
        if key == "reloc_type": return self.reloc_type
        if key == "offset": return self.offset
        if key == "section": return self.section
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        return {
            "nid": self.nid,
            "raw_name": self.raw_name,
            "tag": self.tag,
            "lib": self.lib,
            "mod": self.mod,
            "reloc_type": self.reloc_type,
            "offset": self.offset,
            "section": self.section,
        }

    def __repr__(self):
        return f"ImportedSymbol(0x{self.nid:016X}, '{self.raw_name}')"


class ImportTable(dict):
    """
    Dictionary mapping raw_symbol_name -> ImportedSymbol.
    Supports lookup by integer NID as well as symbol name string.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jmprel_symbols = {}
        self.rela_symbols = {}
        self.by_nid = {}

    def add_symbol(self, sym: ImportedSymbol, is_jmprel: bool = True):
        self[sym.raw_name] = sym
        if sym.nid not in self.by_nid:
            self.by_nid[sym.nid] = []
        self.by_nid[sym.nid].append(sym)
        if is_jmprel:
            self.jmprel_symbols[sym.raw_name] = sym
        else:
            self.rela_symbols[sym.raw_name] = sym

    def __getitem__(self, key):
        if isinstance(key, int):
            matches = self.by_nid.get(key)
            if matches:
                return matches[0]
            raise KeyError(f"NID 0x{key:016X} not found in import table")
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def parse_eboot_imports(path_or_data) -> ImportTable:
    """
    Parse PS4 ELF/SELF dynamic import tables with zero external dependencies.
    Accepts either a file path or binary bytes.
    Parses PT_DYNAMIC, PT_SCE_PROGLIB / PT_SCE_DYNLIBDATA, extracting:
      DT_SCE_STRTAB / DT_STRTAB
      DT_SCE_SYMTAB / DT_SYMTAB
      DT_SCE_JMPREL / DT_JMPREL
      DT_SCE_PLTRELSZ / DT_PLTRELSZ
      DT_SCE_RELA / DT_RELA
      DT_SCE_RELASZ / DT_RELASZ
    Decodes imported symbols (<tag>#<lib>#<mod>) to 64-bit NIDs.
    Returns an ImportTable mapping symbol names to ImportedSymbol objects.
    """
    if isinstance(path_or_data, (str, os.PathLike)):
        with open(path_or_data, "rb") as f:
            raw_data = f.read()
    else:
        raw_data = bytes(path_or_data)

    if raw_data[:4] == b"O\x15=\x1d":
        raw_data = unpack_self(raw_data)

    if raw_data[:4] != b"\x7fELF":
        raise ValueError("Invalid ELF magic (expected \\x7fELF)")

    e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum = struct.unpack("<QQIHHH", raw_data[32:58])

    phs = []
    for i in range(e_phnum):
        ph_raw = raw_data[e_phoff + i * 56 : e_phoff + (i + 1) * 56]
        p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack(
            "<IIQQQQQQ", ph_raw
        )
        phs.append({
            "index": i, "type": p_type, "flags": p_flags, "offset": p_offset,
            "vaddr": p_vaddr, "filesz": p_filesz, "memsz": p_memsz
        })

    PT_LOAD = 1
    PT_DYNAMIC = 2
    PT_SCE_PROGLIB = 0x61000000
    PT_SCE_DYNLIBDATA = 0x61000001
    PT_SCE_RELRO = 0x61000010

    proglib_off = 0
    proglib_sz = 0
    for ph in phs:
        if ph["type"] == PT_SCE_PROGLIB or (ph["type"] == PT_SCE_DYNLIBDATA and ph["filesz"] > 0x100):
            if ph["filesz"] > 0 and ph["offset"] + ph["filesz"] <= len(raw_data):
                proglib_off = ph["offset"]
                proglib_sz = ph["filesz"]
                break

    dyn_entries = []
    for ph in phs:
        if ph["type"] in (PT_DYNAMIC, PT_SCE_DYNLIBDATA):
            num_dyn = ph["filesz"] // 16
            for i in range(num_dyn):
                entry = raw_data[ph["offset"] + i * 16 : ph["offset"] + (i + 1) * 16]
                d_tag, d_val = struct.unpack("<qQ", entry)
                if d_tag == 0:
                    break
                dyn_entries.append((d_tag, d_val))

    dyn_map = {}
    for tag, val in dyn_entries:
        dyn_map[tag] = val

    DT_PLTRELSZ = 2
    DT_STRTAB = 5
    DT_SYMTAB = 6
    DT_RELA = 7
    DT_RELASZ = 8
    DT_STRSZ = 10
    DT_JMPREL = 23

    DT_SCE_JMPREL = 0x61000029
    DT_SCE_PLTRELSZ = 0x6100002D
    DT_SCE_RELA = 0x6100002F
    DT_SCE_RELASZ = 0x61000031
    DT_SCE_STRTAB = 0x61000035
    DT_SCE_STRSZ = 0x61000037
    DT_SCE_SYMTAB = 0x61000039

    strtab_addr = dyn_map.get(DT_SCE_STRTAB, dyn_map.get(DT_STRTAB, 0))
    strsz = dyn_map.get(DT_SCE_STRSZ, dyn_map.get(DT_STRSZ, 0))
    symtab_addr = dyn_map.get(DT_SCE_SYMTAB, dyn_map.get(DT_SYMTAB, 0))
    jmprel_addr = dyn_map.get(DT_SCE_JMPREL, dyn_map.get(DT_JMPREL, 0))
    pltrel_sz = dyn_map.get(DT_SCE_PLTRELSZ, dyn_map.get(DT_PLTRELSZ, 0))
    rela_addr = dyn_map.get(DT_SCE_RELA, dyn_map.get(DT_RELA, 0))
    rela_sz = dyn_map.get(DT_SCE_RELASZ, dyn_map.get(DT_RELASZ, 0))

    def read_elf_data(addr, sz):
        if sz <= 0:
            return b""
        if proglib_off != 0 and (addr + sz) <= proglib_sz:
            return raw_data[proglib_off + addr : proglib_off + addr + sz]
        for ph in phs:
            if ph["type"] in (PT_LOAD, PT_SCE_RELRO, PT_DYNAMIC):
                if ph["vaddr"] <= addr and (addr + sz) <= (ph["vaddr"] + ph["filesz"]):
                    off = ph["offset"] + (addr - ph["vaddr"])
                    return raw_data[off : off + sz]
        if addr + sz <= len(raw_data):
            return raw_data[addr : addr + sz]
        return b""

    strtab = read_elf_data(strtab_addr, strsz)

    def get_sym_name(st_name):
        if not strtab or st_name >= len(strtab):
            return ""
        end = strtab.find(b"\x00", st_name)
        if end == -1:
            end = len(strtab)
        return strtab[st_name:end].decode("ascii", errors="replace")

    import_table = ImportTable()

    # Process JMPREL relocations (PLT jump slots)
    if jmprel_addr != 0 and pltrel_sz != 0:
        count = pltrel_sz // 24
        jmprel_bytes = read_elf_data(jmprel_addr, pltrel_sz)
        for i in range(count):
            entry = jmprel_bytes[i * 24 : (i + 1) * 24]
            if len(entry) < 24:
                continue
            r_offset, r_info, r_addend = struct.unpack("<QQq", entry)
            sym_idx = r_info >> 32
            reloc_type = r_info & 0xFFFFFFFF

            sym_bytes = read_elf_data(symtab_addr + sym_idx * 24, 24)
            if len(sym_bytes) < 24:
                continue
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack("<IBBHQQ", sym_bytes)
            sname = get_sym_name(st_name)
            if not sname:
                continue

            parts = sname.split("#")
            tag = parts[0]
            lib = parts[1] if len(parts) > 1 else ""
            mod = parts[2] if len(parts) > 2 else ""

            if st_value != 0:
                nid = st_value
            elif is_sony_tag(tag):
                nid = decode_sony_tag(tag)
            else:
                nid = compute_sony_nid(tag)

            sym_obj = ImportedSymbol(
                nid,
                raw_name=sname,
                tag=tag,
                lib=lib,
                mod=mod,
                reloc_type=reloc_type,
                offset=r_offset,
                section="JMPREL",
            )
            import_table.add_symbol(sym_obj, is_jmprel=True)

    # Process RELA relocations
    if rela_addr != 0 and rela_sz != 0:
        count = rela_sz // 24
        rela_bytes = read_elf_data(rela_addr, rela_sz)
        for i in range(count):
            entry = rela_bytes[i * 24 : (i + 1) * 24]
            if len(entry) < 24:
                continue
            r_offset, r_info, r_addend = struct.unpack("<QQq", entry)
            sym_idx = r_info >> 32
            reloc_type = r_info & 0xFFFFFFFF

            if sym_idx == 0:
                continue

            sym_bytes = read_elf_data(symtab_addr + sym_idx * 24, 24)
            if len(sym_bytes) < 24:
                continue
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack("<IBBHQQ", sym_bytes)
            sname = get_sym_name(st_name)
            if not sname:
                continue

            parts = sname.split("#")
            tag = parts[0]
            lib = parts[1] if len(parts) > 1 else ""
            mod = parts[2] if len(parts) > 2 else ""

            if st_value != 0:
                nid = st_value
            elif is_sony_tag(tag):
                nid = decode_sony_tag(tag)
            else:
                nid = compute_sony_nid(tag)

            sym_obj = ImportedSymbol(
                nid,
                raw_name=sname,
                tag=tag,
                lib=lib,
                mod=mod,
                reloc_type=reloc_type,
                offset=r_offset,
                section="RELA",
            )
            if sname not in import_table:
                import_table.add_symbol(sym_obj, is_jmprel=False)

    return import_table


def cross_reference_imports(imports: ImportTable, registry_path: str, eboot_label: str = ""):
    """
    Cross-reference parsed eboot imports against src/hle/nid_registry.cpp
    and print a structured breakdown of resolved and unresolved symbols.
    """
    print("=" * 78)
    title = f"CROSS-REFERENCE: {eboot_label}" if eboot_label else "CROSS-REFERENCE EBOOT IMPORTS"
    print(f"{title} VS REGISTRY")
    print("=" * 78)

    with open(registry_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r'register_symbol\(\s*(0x[0-9A-Fa-f]+[uU]?[lL]{0,2}|\d+[uU]?[lL]{0,2})\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"'
    )
    raw_matches = pattern.findall(content)
    reg_nids = {}
    for nid_str, name, mod in raw_matches:
        clean_nid_str = nid_str.rstrip("uUlL")
        nid_val = int(clean_nid_str, 16 if clean_nid_str.startswith("0x") else 10)
        reg_nids[nid_val] = (name, mod)

    total_imports = len(imports)
    jmprel_count = len(imports.jmprel_symbols)
    rela_count = len(imports.rela_symbols)

    resolved = {}
    unresolved = {}

    for sym_name, sym in imports.items():
        if sym.nid in reg_nids:
            resolved[sym_name] = (sym, reg_nids[sym.nid])
        else:
            unresolved[sym_name] = sym

    print(f"Total unique imported symbols: {total_imports} (JMPREL: {jmprel_count}, RELA: {rela_count})")
    print(f"Resolved in HLE registry:      {len(resolved)} ({(len(resolved)/total_imports*100):.1f}%)")
    print(f"Unresolved in HLE registry:    {len(unresolved)} ({(len(unresolved)/total_imports*100):.1f}%)")

    # Group resolved by module
    resolved_by_mod = defaultdict(list)
    for sym_name, (sym, (reg_name, reg_mod)) in resolved.items():
        resolved_by_mod[reg_mod].append((sym, reg_name))

    print("\n--- RESOLVED SYMBOLS BY MODULE ---")
    for mod in sorted(resolved_by_mod.keys()):
        syms = resolved_by_mod[mod]
        print(f"  {mod:25s}: {len(syms)} symbols resolved")

    # Sample resolved
    print("\n--- SAMPLE RESOLVED IMPORTS (FIRST 10) ---")
    print(f"  {'Imported Symbol':25s} {'Decoded NID':18s} {'HLE Handler Name':30s} {'Module':15s}")
    print("  " + "-" * 90)
    for sym_name in list(resolved.keys())[:10]:
        sym, (reg_name, reg_mod) = resolved[sym_name]
        print(f"  {sym_name:25s} 0x{sym.nid:016X} {reg_name:30s} {reg_mod:15s}")

    # Sample unresolved
    print(f"\n--- SAMPLE UNRESOLVED IMPORTS (FIRST 10 OF {len(unresolved)}) ---")
    print(f"  {'Imported Symbol':25s} {'Decoded NID':18s} {'Tag':14s} {'Lib/Mod Code':15s}")
    print("  " + "-" * 75)
    for sym_name in list(unresolved.keys())[:10]:
        sym = unresolved[sym_name]
        lib_mod = f"{sym.lib}#{sym.mod}" if sym.lib or sym.mod else ""
        print(f"  {sym_name:25s} 0x{sym.nid:016X} {sym.tag:14s} {lib_mod:15s}")

    print()
    return {
        "total": total_imports,
        "resolved": resolved,
        "unresolved": unresolved,
        "resolved_by_module": resolved_by_mod,
    }


def main():
    parser = argparse.ArgumentParser(
        description="PS4 Sony NID Verification, Registry Audit & Import Scanner"
    )
    parser.add_argument(
        "--eboot",
        type=str,
        default=None,
        help="Path to PS4 eboot binary (.bin SELF or .elf ELF64)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to src/hle/nid_registry.cpp",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # 1. Run test vectors
    test_vector()

    # 2. Audit HLE NID registry
    reg_path = Path(args.registry) if args.registry else repo_root / "src" / "hle" / "nid_registry.cpp"
    if reg_path.exists():
        audit_nid_registry(str(reg_path), verbose=args.verbose)
    else:
        print(f"[WARN] NID registry not found at: {reg_path}")

    # 3. Scan eboot imports
    eboot_candidates = []
    if args.eboot:
        eboot_candidates.append(Path(args.eboot))
    else:
        # Default known locations: Flappy Bird eboot.bin and eboot.elf
        default_dir = repo_root / "games" / "flappy_bird" / "ED1633-NPXX51362_00-TER1520000000000"
        candidate_bin = default_dir / "eboot.bin"
        candidate_elf = default_dir / "eboot.elf"
        if candidate_bin.exists():
            eboot_candidates.append(candidate_bin)
        if candidate_elf.exists():
            eboot_candidates.append(candidate_elf)

    if not eboot_candidates:
        print("[INFO] No eboot binary specified and default fixture not found.")
    else:
        for eboot_path in eboot_candidates:
            if not eboot_path.exists():
                print(f"[WARN] Specified eboot file does not exist: {eboot_path}")
                continue
            print(f"Scanning imports from: {eboot_path}")
            imports = parse_eboot_imports(eboot_path)
            if reg_path.exists():
                cross_reference_imports(imports, str(reg_path), eboot_label=eboot_path.name)
            else:
                print(f"Parsed {len(imports)} imported symbols from {eboot_path.name}.")

    print("=" * 78)
    print("VERIFICATION COMPLETE - EXIT CODE 0")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
