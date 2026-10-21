#!/usr/bin/env python3
"""
tests/test_triage.py - Unit tests for scripts/triage_crash.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

import triage_crash
from triage_crash import (
    Log,
    Finding,
    CpuFaultAnalyzer,
    decode_unhandled_opcode_bytes,
)


class TestStage1LogParsingCore(unittest.TestCase):
    def test_banner_mode_hle_and_note(self):
        log_content = """[prx] mode=hle
[CPU] Instructions Executed: 50000000
[CPU] Execution Status: Instruction executed successfully
"""
        log = Log(log_content.splitlines(keepends=True))
        b = log.banner_dict()
        self.assertEqual(b["mode"], "hle")
        self.assertEqual(b["final_instruction_count"], 50000000)
        self.assertEqual(b["stop_reason"], "Instruction executed successfully")
        self.assertIn("Mono never executed", b["note"])
        self.assertEqual(b["instruction_budget"], "not present in log")
        self.assertEqual(b["wall_time"], "not present in log")

    def test_banner_mode_unknown(self):
        log_content = """[CPU] Instructions Executed: 1234
[CPU] Execution Status: Halted
"""
        log = Log(log_content.splitlines(keepends=True))
        b = log.banner_dict()
        self.assertEqual(b["mode"], "unknown")
        self.assertEqual(b["note"], "mode unknown")

    def test_banner_budget_and_wall_time(self):
        log_content = """[prx] mode=real
--budget 1000000
elapsed: 4.5s
[CPU] Instructions Executed: 999999
[CPU] Execution Status: Budget exceeded
"""
        log = Log(log_content.splitlines(keepends=True))
        b = log.banner_dict()
        self.assertEqual(b["mode"], "real")
        self.assertEqual(b["instruction_budget"], 1000000)
        self.assertEqual(b["wall_time"], "4.5s")
        self.assertEqual(b["final_instruction_count"], 999999)
        self.assertIsNone(b["note"])

    def test_unparsed_lines_counting(self):
        log_content = """[prx] mode=real
RANDOM JUNK LINE 1
RANDOM JUNK LINE 2
[CPU] Instructions Executed: 100
"""
        log = Log(log_content.splitlines(keepends=True))
        b = log.banner_dict()
        self.assertEqual(b["unparsed_lines"], 2)
        self.assertEqual(b["total_lines"], 4)

    def test_dcb_dumps_indexed(self):
        log_content = """[DCB FULL DUMP START 0x229309e0 count=3]
DWORD[229309e0][0]=0xc0055800
DWORD[229309e0][1]=0x88c40000
DWORD[229309e0][2]=0x00000001
[DCB FULL DUMP END 0x229309e0]
"""
        log = Log(log_content.splitlines(keepends=True))
        self.assertEqual(len(log.dcb_dumps), 1)
        dcb = log.dcb_dumps[0]
        self.assertEqual(dcb.addr, 0x229309e0)
        self.assertEqual(dcb.count, 3)
        self.assertEqual(dcb.dwords, [0xc0055800, 0x88c40000, 0x00000001])


class TestStage2CpuFaultAnalyzer(unittest.TestCase):
    def test_decode_d9_7d_f6_fnstcw(self):
        raw = bytes.fromhex("d9 7d f6")
        decoded, encoding_label, op_class = decode_unhandled_opcode_bytes(raw)
        self.assertEqual(encoding_label, "D9 /7")
        self.assertEqual(op_class, "x87")
        self.assertEqual(decoded, "FNSTCW word ptr [rbp-0xA] (D9 /7)")

    def test_plt_got_slot_arithmetic(self):
        # Return address = 0x106840b.
        # Call instruction at ret-5 (0x1068406): E8 a5 4c 7a ff (offset = -0x85b35b)
        # target = 0x106840b + (-0x85b35b) = 0x80d0b0
        # At target 0x80d0b0: ff 25 5a 2e 03 00 (jmp qword ptr [rip + 0x32e5a])
        # slot = target + 6 + 0x32e5a = 0x80d0b0 + 6 + 0x32e5a = 0x83ff10
        ret = 0x106840b
        call_bytes = bytes.fromhex("e8 a5 4c 7a ff")
        target_bytes = bytes.fromhex("ff 25 5a 2e 03 00")

        log_content = f"""[CPU FAULT] RIP=0x909090909090900b, RSP=0x1ffffb7b8 (return=0x{ret:x})
[CPU FAULT] module=<none: not inside any loaded module>
[CPU FAULT] instruction before return: @0x1068406  {call_bytes.hex(' ')}
[CPU FAULT]   bytes at call target: {target_bytes.hex(' ')}
"""
        log = Log(log_content.splitlines(keepends=True))
        cfa = CpuFaultAnalyzer(log)
        self.assertEqual(cfa.rip, 0x909090909090900b)
        self.assertEqual(cfa.return_addr, ret)
        res = cfa._resolve_bad_indirect_call()
        self.assertIsNotNone(res)
        target, disp32, got_slot, _ = res
        self.assertEqual(target, 0x80d0b0)
        self.assertEqual(disp32, 0x32e5a)
        self.assertEqual(got_slot, 0x83ff10)

        # Check finding message
        indirect_findings = [f for f in cfa.findings if "Indirect call via PLT" in f.message]
        self.assertEqual(len(indirect_findings), 1)
        self.assertIn("GOT slot=0x83FF10", indirect_findings[0].message)

    def test_fault_address_classification(self):
        log = Log([])
        cfa = CpuFaultAnalyzer(log)
        # NULL-ish (< 0x10000)
        self.assertIn("NULL-ish", cfa.classify_address(0x10))
        # looks like two u32s
        self.assertIn("looks like two u32s", cfa.classify_address(0x909090909090900b))
        # ASCII
        ascii_addr = int.from_bytes(b"testdata", "little")
        self.assertIn("looks like ASCII", cfa.classify_address(ascii_addr))
        # Libc heap range (0x20000000 - 0x40000000)
        self.assertIn("guest libc heap", cfa.classify_address(0x25000000))
        # Direct memory range (0x40000000 - 0x1c0000000)
        self.assertIn("guest direct/GPU memory", cfa.classify_address(0x50000000))
        # Unmapped
        self.assertIn("unmapped", cfa.classify_address(0x300000000))

    def test_unhandled_opcode_log_parsing(self):
        log_content = """[CPU] Unhandled opcode at RIP=0x123456: d9 7d f6 00 00 00 00 00
"""
        log = Log(log_content.splitlines(keepends=True))
        cfa = CpuFaultAnalyzer(log)
        self.assertEqual(cfa.unhandled_opcode_rip, 0x123456)
        findings = [f for f in cfa.findings if "Unhandled opcode" in f.message]
        self.assertEqual(len(findings), 1)
        self.assertIn("FNSTCW word ptr [rbp-0xA] (D9 /7)", findings[0].message)
        self.assertIn("[class: x87", findings[0].message)

    def test_segfault_vs_guest_oob(self):
        log_segv = Log(["Segmentation fault (address 0x1234 size 8)\n"])
        cfa_segv = CpuFaultAnalyzer(log_segv)
        self.assertTrue(cfa_segv.segfault_detected)
        self.assertFalse(cfa_segv.guest_oob_detected)
        self.assertIn("Host segfault", str(cfa_segv.findings[-1]))

        log_oob = Log(["out of bounds at 0x5678 (size 4)\n"])
        cfa_oob = CpuFaultAnalyzer(log_oob)
        self.assertTrue(cfa_oob.guest_oob_detected)
        self.assertFalse(cfa_oob.segfault_detected)
        self.assertIn("Guest out-of-bounds", str(cfa_oob.findings[-1]))


class TestStage3HleAnalyzer(unittest.TestCase):
    def test_nid_self_check(self):
        self.assertFalse(triage_crash.RESOLVER_BROKEN)
        le, _ = triage_crash.compute_sony_nid("scePthreadCondInit")
        self.assertEqual(le, 0xd936fddaaba9ae5d)

    def test_known_nid_match(self):
        resolver = triage_crash.NidResolver()
        status, details, name = resolver.resolve(0xd936fddaaba9ae5d)
        self.assertEqual(status, "MATCH")
        self.assertEqual(name, "scePthreadCondInit")
        self.assertTrue("repo_registry" in details or "salted_sha1_le" in details)

    def test_hash_match_from_names_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("custom_game_func\n")
            names_path = f.name

        try:
            le, _ = triage_crash.compute_sony_nid("custom_game_func")
            resolver = triage_crash.NidResolver(names_file=names_path)
            status, details, name = resolver.resolve(le)
            self.assertEqual(status, "MATCH")
            self.assertEqual(name, "custom_game_func")
            self.assertIn("names_file", details)
            self.assertIn("salted_sha1_le", details)
        finally:
            if os.path.isfile(names_path):
                os.remove(names_path)

    def test_unknown_nid_unidentified(self):
        resolver = triage_crash.NidResolver()
        unknown_nid = 0x1111222233334444
        status, details, name = resolver.resolve(unknown_nid)
        self.assertEqual(status, "UNIDENTIFIED")
        self.assertIsNone(name)
        self.assertIn("UNIDENTIFIED", details)
        self.assertIn("candidates", details)

    def test_hle_fallback_parsing(self):
        log_content = """[HLE] sceKernelLoadStartModule('/app0/test.prx') -> handle 0x2001
[HLE WARNING #13 TID=1 rip=0x101f95] Unimplemented stub called: NID=0xacfa3ab55f03f5b3 (caller RIP=0x10674e6, rdi=0x1603e88, rsi=0x0, rdx=0x800000) returning 0 (0x0)
[HLE WARNING #14 TID=1 rip=0x101f95] Unimplemented stub called: NID=0xacfa3ab55f03f5b3 (caller RIP=0x10674e6, rdi=0x1603e88, rsi=0x0, rdx=0x800000) returning 0 (0x0)
"""
        log = Log(log_content.splitlines(keepends=True))
        resolver = triage_crash.NidResolver()
        hle = triage_crash.HleAnalyzer(log, resolver)
        self.assertIn(0xacfa3ab55f03f5b3, hle.fallback_stubs)
        rec = hle.fallback_stubs[0xacfa3ab55f03f5b3]
        self.assertEqual(rec.count, 2)
        self.assertEqual(rec.first_hit_caller_rip, 0x10674e6)
        self.assertEqual(rec.first_hit_args.get("rdi"), "0x1603e88")

    def test_inferred_state_waiting_vs_running(self):
        # Waiting state
        log_wait = Log(["[HLE] sceKernelWaitSema(1, 1)\n"])
        hle_wait = triage_crash.HleAnalyzer(log_wait, triage_crash.NidResolver())
        inferred_findings = [f for f in hle_wait.findings if f.kind == "INFERRED"]
        self.assertTrue(len(inferred_findings) >= 1)
        self.assertIn("waiting", inferred_findings[0].message)

        # Running state
        log_run = Log(["[HLE] sceKernelLoadStartModule('/app0/eboot.bin')\n"])
        hle_run = triage_crash.HleAnalyzer(log_run, triage_crash.NidResolver())
        inferred_findings = [f for f in hle_run.findings if f.kind == "INFERRED"]
        self.assertTrue(len(inferred_findings) >= 1)
        self.assertIn("actively running", inferred_findings[0].message)

    def test_benign_stub_allowlist(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write('[benign]\n0x6f3404c72d7cf592 = "Known benign stub"\n')
            allow_path = f.name

        try:
            log_content = """[HLE WHITELIST] Benign stub executed: NID=0x6f3404c72d7cf592 (caller RIP=0x99)
"""
            log = Log(log_content.splitlines(keepends=True))
            hle = triage_crash.HleAnalyzer(log, triage_crash.NidResolver(), allowlist_file=allow_path)
            self.assertEqual(len(hle.known_benign_findings), 1)
            self.assertEqual(hle.known_benign_findings[0][2], "Known benign stub")
        finally:
            if os.path.isfile(allow_path):
                os.remove(allow_path)


if __name__ == "__main__":
    unittest.main()
