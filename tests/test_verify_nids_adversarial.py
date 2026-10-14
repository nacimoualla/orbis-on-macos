#!/usr/bin/env python3
"""
Adversarial Stress Test Suite for scripts/verify_nids.py

Empirically tests:
1. Tag roundtripping across:
   - 0, 0xFFFFFFFFFFFFFFFF, powers of 2 (0..63), inverted powers of 2
   - MSB set, alternating bit patterns (0xAAAAAAAAAAAAAAAA, 0x5555555555555555)
   - 1,000,000 randomized uint64 values
2. Tag codec edge cases:
   - Bijective identity: encode_sony_tag(decode_sony_tag(t)) == t for valid tags
   - Suffix handling: tag#lib#mod, tag#
   - Invalid characters, short tags, empty tags, non-strings
   - Strings with invalid lowest 2 bits (distinguishing valid Sony tags from non-tags)
3. Known test vectors:
   - strcmp <-> 0x3af6f675224e02e1 <-> Ovb2dSJOAuE
   - sceFiosInitialize <-> 0xC00299FDD7ADFB2A <-> wAKZ-det+yo
   - sceUserServiceGetEvent <-> 0xC87D7B43A356B558 <-> yH17Q6NWtVg
   - sceVideoOutOpen <-> 0x529DFA3D393AF3B1 <-> Up36PTk687E
   - sceKernelClockGettime <-> 0x4018BB1C22B4DE1C <-> QBi7HCK03hw
4. Malformed inputs & robustness:
   - parse_eboot_imports with missing file, empty file, 1-byte file, truncated ELF
   - unpack_self with truncated SELF headers, corrupt segment counts, invalid magic
   - audit_nid_registry with empty file, missing file, non-existent path
   - CLI execution with missing args, invalid paths, and real eboot fixtures
"""

import sys
import os
import random
import struct
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

import verify_nids
from verify_nids import (
    compute_sony_nid,
    encode_sony_tag,
    decode_sony_tag,
    is_sony_tag,
    test_vector,
    audit_nid_registry,
    unpack_self,
    parse_eboot_imports,
    cross_reference_imports,
    ImportedSymbol,
    ImportTable,
    SONY_ALPHA,
)


class TestSonyTagRoundtrip(unittest.TestCase):
    """Stress test Sony tag encoding, decoding, and round-trip identity."""

    def test_edge_case_nids(self):
        """Test specific edge case NIDs: 0, max uint64, powers of 2, alternating bits."""
        edge_cases = [
            0,
            1,
            2,
            0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF,
            0x8000000000000000,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            0x0123456789ABCDEF,
            0xFEDCBA9876543210,
        ]
        # Powers of 2: bit 0 through bit 63
        for bit in range(64):
            edge_cases.append(1 << bit)
            edge_cases.append((~(1 << bit)) & 0xFFFFFFFFFFFFFFFF)
            edge_cases.append(((1 << bit) - 1) & 0xFFFFFFFFFFFFFFFF)

        for nid in edge_cases:
            tag = encode_sony_tag(nid)
            self.assertEqual(len(tag), 11, f"Tag length must be 11 for NID 0x{nid:016X}")
            for c in tag:
                self.assertIn(c, SONY_ALPHA, f"Char '{c}' not in SONY_ALPHA for NID 0x{nid:016X}")

            decoded = decode_sony_tag(tag)
            self.assertEqual(decoded, nid, f"Roundtrip failed for NID 0x{nid:016X}: got 0x{decoded:016X}")
            self.assertTrue(is_sony_tag(tag), f"is_sony_tag returned False for valid encoded tag '{tag}'")

    def test_exhaustive_bit_patterns(self):
        """Walk 1, 2, 3, 4, and 8 bits across all 64 positions, both positive and inverted."""
        patterns = []
        for bit in range(64):
            patterns.append(1 << bit)
            patterns.append((~(1 << bit)) & 0xFFFFFFFFFFFFFFFF)
            for width in [2, 3, 4, 8, 16, 32]:
                if bit + width <= 64:
                    mask = ((1 << width) - 1) << bit
                    patterns.append(mask)
                    patterns.append((~mask) & 0xFFFFFFFFFFFFFFFF)

        for p in patterns:
            tag = encode_sony_tag(p)
            self.assertEqual(len(tag), 11)
            decoded = decode_sony_tag(tag)
            self.assertEqual(decoded, p, f"Bit pattern 0x{p:016X} roundtrip failed")
            self.assertTrue(is_sony_tag(tag))

    def test_random_1m_uint64_roundtrip(self):
        """Stress test 1,000,000 random uint64 values."""
        rng = random.Random(0x1337C0DE)
        count = 1000000
        for _ in range(count):
            nid = rng.getrandbits(64)
            tag = encode_sony_tag(nid)
            decoded = decode_sony_tag(tag)
            self.assertEqual(decoded, nid)

    def test_tag_with_suffixes(self):
        """Test decoding tags with various library/module suffixes."""
        nid = 0x3AF6F675224E02E1
        base_tag = encode_sony_tag(nid)
        suffixes = [
            "#lib#mod",
            "#l#l",
            "#libkernel#libkernel",
            "#",
            "##",
            "   #padded",
        ]
        for s in suffixes:
            full = f"{base_tag}{s}"
            decoded = decode_sony_tag(full)
            self.assertEqual(decoded, nid, f"Suffix decoding failed for '{full}'")

    def test_invalid_tags(self):
        """Test error handling on malformed tag strings."""
        # Short tags
        with self.assertRaises(ValueError):
            decode_sony_tag("")
        with self.assertRaises(ValueError):
            decode_sony_tag("1234567890")  # 10 chars
        with self.assertRaises(ValueError):
            decode_sony_tag("short#lib#m")

        # Invalid characters
        with self.assertRaises(ValueError):
            decode_sony_tag("Ovb2dSJOAu_")  # '_' is not in SONY_ALPHA
        with self.assertRaises(ValueError):
            decode_sony_tag("Ovb2dSJOAu=")
        with self.assertRaises(ValueError):
            decode_sony_tag("Ovb2dSJOAu!")
        with self.assertRaises(ValueError):
            decode_sony_tag("Ovb2dSJOAu ")

    def test_is_sony_tag_predicate(self):
        """Verify is_sony_tag correctly identifies valid vs invalid tags."""
        self.assertTrue(is_sony_tag("Ovb2dSJOAuE"))
        self.assertTrue(is_sony_tag("wAKZ-det+yo"))
        self.assertTrue(is_sony_tag("yH17Q6NWtVg"))

        # Non-11 length
        self.assertFalse(is_sony_tag(""))
        self.assertFalse(is_sony_tag("Ovb2dSJOAu"))   # 10 chars
        self.assertFalse(is_sony_tag("Ovb2dSJOAuEE")) # 12 chars
        self.assertFalse(is_sony_tag(None))
        self.assertFalse(is_sony_tag(12345))

        # Invalid characters
        self.assertFalse(is_sony_tag("Ovb2dSJOAu_"))

        # String with invalid bottom 2 bits
        # In SONY_ALPHA, index of 'E' is 4 (binary 000100 -> lowest 2 bits are 00).
        # If we change the 11th char so that lowest 2 bits are not 00:
        # e.g. SONY_ALPHA[5] is 'F' (binary 000101 -> lowest 2 bits are 01)
        tag_invalid_bits = "Ovb2dSJOAuF"
        self.assertFalse(
            is_sony_tag(tag_invalid_bits),
            "Tag with non-zero trailing bits must not be considered valid Sony tag"
        )


class TestKnownTestVectors(unittest.TestCase):
    """Verify official Sony test vectors and hash computation."""

    def test_official_test_vectors(self):
        """Verify strcmp, sceFiosInitialize, sceUserServiceGetEvent, sceVideoOutOpen, sceKernelClockGettime."""
        vectors = [
            ("strcmp", 0x3AF6F675224E02E1, "Ovb2dSJOAuE"),
            ("sceFiosInitialize", 0xC00299FDD7ADFB2A, "wAKZ-det+yo"),
            ("sceUserServiceGetEvent", 0xC87D7B43A356B558, "yH17Q6NWtVg"),
            ("sceVideoOutOpen", 0x529DFA3D393AF3B1, "Up36PTk687E"),
            ("sceKernelClockGettime", 0x4018BB1C22B4DE1C, "QBi7HCK03hw"),
        ]
        for name, expected_nid, expected_tag in vectors:
            computed_nid = compute_sony_nid(name)
            self.assertEqual(computed_nid, expected_nid, f"NID mismatch for {name}")
            computed_tag = encode_sony_tag(computed_nid)
            self.assertEqual(computed_tag, expected_tag, f"Tag mismatch for {name}")
            decoded_nid = decode_sony_tag(expected_tag)
            self.assertEqual(decoded_nid, expected_nid, f"Decode mismatch for {expected_tag}")

    def test_empty_string_nid(self):
        """Verify compute_sony_nid handles empty symbol string."""
        nid = compute_sony_nid("")
        self.assertIsInstance(nid, int)
        self.assertTrue(0 <= nid <= 0xFFFFFFFFFFFFFFFF)


class TestImportTableAndSymbol(unittest.TestCase):
    """Test ImportedSymbol and ImportTable container behaviors."""

    def test_imported_symbol_properties(self):
        sym = ImportedSymbol(
            0x3AF6F675224E02E1,
            raw_name="Ovb2dSJOAuE#libc#libc",
            tag="Ovb2dSJOAuE",
            lib="libc",
            mod="libc",
            reloc_type=7,
            offset=0x1000,
            section="JMPREL",
        )
        # Inherits from int
        self.assertEqual(sym, 0x3AF6F675224E02E1)
        self.assertEqual(int(sym), 0x3AF6F675224E02E1)
        # Dict-like access
        self.assertEqual(sym["nid"], 0x3AF6F675224E02E1)
        self.assertEqual(sym["raw_name"], "Ovb2dSJOAuE#libc#libc")
        self.assertEqual(sym["tag"], "Ovb2dSJOAuE")
        self.assertEqual(sym["lib"], "libc")
        self.assertEqual(sym["mod"], "libc")
        self.assertEqual(sym["reloc_type"], 7)
        self.assertEqual(sym["offset"], 0x1000)
        self.assertEqual(sym["section"], "JMPREL")
        # Attributes
        self.assertEqual(sym.nid, 0x3AF6F675224E02E1)
        self.assertEqual(sym.tag, "Ovb2dSJOAuE")
        # to_dict
        d = sym.to_dict()
        self.assertEqual(d["nid"], 0x3AF6F675224E02E1)

    def test_import_table_dual_lookup(self):
        table = ImportTable()
        sym = ImportedSymbol(0x3AF6F675224E02E1, raw_name="Ovb2dSJOAuE#libc#libc")
        table.add_symbol(sym, is_jmprel=True)

        # Lookup by string key
        self.assertEqual(table["Ovb2dSJOAuE#libc#libc"], sym)
        # Lookup by int NID
        self.assertEqual(table[0x3AF6F675224E02E1], sym)
        # Missing int key raises KeyError
        with self.assertRaises(KeyError):
            _ = table[0x123456789]


class TestMalformedAndCorruptInputs(unittest.TestCase):
    """Stress test robustness against corrupt binaries, missing files, and malformed files."""

    def test_missing_files(self):
        """Missing file paths should raise FileNotFoundError cleanly."""
        with self.assertRaises(FileNotFoundError):
            parse_eboot_imports("/path/to/definitely/nonexistent/eboot.bin")
        with self.assertRaises(FileNotFoundError):
            audit_nid_registry("/path/to/definitely/nonexistent/registry.cpp")

    def test_empty_and_truncated_binary_inputs(self):
        """Binary parser with empty or truncated bytes must raise ValueError or struct.error."""
        with self.assertRaises(ValueError):
            parse_eboot_imports(b"")

        with self.assertRaises(ValueError):
            parse_eboot_imports(b"\x00" * 3)

        with self.assertRaises(ValueError):
            parse_eboot_imports(b"NOT_AN_ELF_BINARY")

        with self.assertRaises((ValueError, struct.error)):
            parse_eboot_imports(b"\x7fELF_BUT_TOO_SHORT")

    def test_corrupt_self_container(self):
        """Test malformed SELF container inputs."""
        # Starts with SELF magic but < 32 bytes
        corrupt_self_short = b"O\x15=\x1d\x00\x00\x00\x00"
        with self.assertRaises(ValueError):  # fails inner ELF check
            parse_eboot_imports(corrupt_self_short)

        # SELF magic with corrupt entry_cnt that points past EOF
        corrupt_self_bad_entries = b"O\x15=\x1d" + b"\x00" * 20 + struct.pack("<H", 100) + b"\x00" * 6
        with self.assertRaises((ValueError, struct.error)):
            parse_eboot_imports(corrupt_self_bad_entries)

    def test_empty_registry_file(self):
        """Auditing an empty or non-matching registry file should report 0 symbols cleanly without exception."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".cpp") as f:
            f.write("// empty file\n")
            f_path = f.name
        try:
            result = audit_nid_registry(f_path)
            self.assertEqual(result["total"], 0)
            self.assertEqual(len(result["matches"]), 0)
            self.assertEqual(len(result["mismatches"]), 0)
        finally:
            os.remove(f_path)

    def test_synthetic_registry_audit(self):
        """Test registry auditor against synthetic matching, mismatching, and duplicate registrations."""
        synthetic_cpp = """
        void register_all(NidRegistry& reg) {
            reg.register_symbol(0x3AF6F675224E02E1ULL, "strcmp", "libc");
            reg.register_symbol(0xC00299FDD7ADFB2AULL, "sceFiosInitialize", "libSceFios2");
            reg.register_symbol(0x1111111111111111ULL, "wrong_nid_symbol", "libtest");
            reg.register_symbol(0x3AF6F675224E02E1ULL, "Ovb2dSJOAuE", "libc"); // tag name
            reg.register_symbol(0x3AF6F675224E02E1ULL, "strcmp_alias", "libc"); // duplicate NID
            reg.register_symbol(0x2222222222222222ULL, "strcmp", "libc"); // duplicate symbol name
        }
        """
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".cpp") as f:
            f.write(synthetic_cpp)
            f_path = f.name
        try:
            result = audit_nid_registry(f_path)
            self.assertEqual(result["total"], 6)
            self.assertIn(0x3AF6F675224E02E1, result["duplicate_nids"])
            self.assertIn("strcmp", result["duplicate_symbols"])
            mismatch_names = [m["name"] for m in result["mismatches"]]
            self.assertIn("wrong_nid_symbol", mismatch_names)
            match_names = [m["name"] for m in result["matches"]]
            self.assertIn("sceFiosInitialize", match_names)
            self.assertIn("Ovb2dSJOAuE", match_names)
        finally:
            os.remove(f_path)


class TestRealFixtures(unittest.TestCase):
    """Test actual project files (eboot.bin, eboot.elf, src/hle/nid_registry.cpp)."""

    def test_real_eboot_bin_and_elf(self):
        bin_path = repo_root / "games" / "flappy_bird" / "ED1633-NPXX51362_00-TER1520000000000" / "eboot.bin"
        elf_path = repo_root / "games" / "flappy_bird" / "ED1633-NPXX51362_00-TER1520000000000" / "eboot.elf"

        if bin_path.exists():
            imports_bin = parse_eboot_imports(bin_path)
            self.assertGreater(len(imports_bin), 700)
            self.assertGreater(len(imports_bin.jmprel_symbols), 700)

        if elf_path.exists():
            imports_elf = parse_eboot_imports(elf_path)
            self.assertGreater(len(imports_elf), 700)
            self.assertGreater(len(imports_elf.jmprel_symbols), 700)

        if bin_path.exists() and elf_path.exists():
            # Bin (unpacked SELF) and ELF should parse the exact same set of symbols
            self.assertEqual(set(imports_bin.keys()), set(imports_elf.keys()))

    def test_real_nid_registry_audit(self):
        reg_path = repo_root / "src" / "hle" / "nid_registry.cpp"
        self.assertTrue(reg_path.exists(), "src/hle/nid_registry.cpp must exist")
        res = audit_nid_registry(str(reg_path))
        self.assertGreater(res["total"], 100, "Registry must contain registered symbols")


class TestCliInvocation(unittest.TestCase):
    """Test verify_nids.py CLI execution under various argument configurations."""

    def test_cli_help(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "verify_nids.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("PS4 Sony NID Verification", res.stdout)

    def test_cli_default_run(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "verify_nids.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("VERIFICATION COMPLETE - EXIT CODE 0", res.stdout)
        self.assertIn("[PASS] strcmp", res.stdout)

    def test_cli_missing_options_graceful(self):
        import subprocess
        res = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "verify_nids.py"),
                "--registry", "nonexistent_registry.cpp",
                "--eboot", "nonexistent_eboot.bin",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("[WARN] NID registry not found", res.stdout)
        self.assertIn("[WARN] Specified eboot file does not exist", res.stdout)
        self.assertIn("VERIFICATION COMPLETE - EXIT CODE 0", res.stdout)

    def test_cli_invalid_flag(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "verify_nids.py"), "--unrecognized-flag"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main()

