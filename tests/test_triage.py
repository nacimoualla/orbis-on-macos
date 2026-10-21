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


class TestStage4LoaderAnalyzer(unittest.TestCase):
    def test_mapping_overlap_detection(self):
        log_content = """[map] registered guest mappings (2):
  owner                              start            end         size
  libkernel_code               0x000100000 0x000200000 0x000100000
  mono_code                    0x000180000 0x000280000 0x000100000
"""
        log = Log(log_content.splitlines(keepends=True))
        loader = triage_crash.LoaderAnalyzer(log)
        self.assertEqual(len(loader.overlap_pairs), 1)
        m1, m2 = loader.overlap_pairs[0]
        self.assertEqual(m1.owner, "libkernel_code")
        self.assertEqual(m2.owner, "mono_code")
        findings = [f for f in loader.findings if "Mapping overlap detected" in f.message]
        self.assertEqual(len(findings), 1)
        self.assertIn("libkernel_code", findings[0].message)
        self.assertIn("mono_code", findings[0].message)

    def test_module_layout_span_check(self):
        log_content = """[module] 0x1 modA: base=0x100000 span=0x200000
[module] 0x2 modB: base=0x250000 span=0x100000
"""
        log = Log(log_content.splitlines(keepends=True))
        loader = triage_crash.LoaderAnalyzer(log)
        self.assertEqual(len(loader.layout_violations), 1)
        findings = [f for f in loader.findings if "Module layout overlap" in f.message]
        self.assertEqual(len(findings), 1)
        self.assertIn("modA", findings[0].message)
        self.assertIn("extends past modB", findings[0].message)

    def test_export_counters_no_nid_skipped_hole(self):
        log_content = """[exports]   eboot.bin                base=0x0 span=0x1500000 | in_table=100 defined=90 undef_skipped=5 no_nid_skipped=5 dup_not_recorded=0 | recorded=90
"""
        log = Log(log_content.splitlines(keepends=True))
        loader = triage_crash.LoaderAnalyzer(log)
        self.assertEqual(len(loader.export_stats), 1)
        hole_findings = [f for f in loader.findings if "possible symbol resolution hole" in f.message]
        self.assertEqual(len(hole_findings), 1)
        self.assertEqual(hole_findings[0].kind, "INFERRED")
        self.assertIn("no_nid_skipped=5", hole_findings[0].message)

    def test_imports_absent_explains_0_unread(self):
        log_content = """[module] RELA: 34441 relocs, 34441 read, 0 unread
"""
        log = Log(log_content.splitlines(keepends=True))
        loader = triage_crash.LoaderAnalyzer(log)
        imp_findings = [f for f in loader.findings if "Imports per module by class: not present in log" in f.message]
        self.assertEqual(len(imp_findings), 1)
        self.assertIn("0 unread", imp_findings[0].message)

    def test_bss_segment_fault_address(self):
        log_content = """[0  ] PT_LOAD           0x4000        0x1000000         0x10000       0x20000      RW-     0x4000
"""
        # BSS range: 0x1000000 + 0x10000 to 0x1000000 + 0x20000 = [0x1010000, 0x1020000)
        fault_rip = 0x1015000
        log = Log(log_content.splitlines(keepends=True))
        loader = triage_crash.LoaderAnalyzer(log, fault_rip=fault_rip)
        bss_findings = [f for f in loader.findings if "is in BSS region" in f.message]
        self.assertEqual(len(bss_findings), 1)
        self.assertIn("0x1015000", bss_findings[0].message)

    def test_write_watch_unrecorded_change_anomaly(self):
        log_content = """[watch] #0   rip=0x100 (0x100) rsp=0x200 -> 0x3000 (8 B) old=0x0000000000000000 new=0x0000000000000001
[watch] #1   rip=0x100 (0x100) rsp=0x200 -> 0x3000 (8 B) old=0x0000000000000005 new=0x0000000000000006
"""
        log = Log(log_content.splitlines(keepends=True))
        loader = triage_crash.LoaderAnalyzer(log)
        self.assertEqual(len(loader.watch_records), 2)
        anomaly_findings = [f for f in loader.findings if "with no recorded write" in f.message]
        self.assertEqual(len(anomaly_findings), 1)
        self.assertEqual(anomaly_findings[0].kind, "INFERRED")
        self.assertIn("changed from 0x0000000000000001 to 0x0000000000000005", anomaly_findings[0].message)


class TestStage5GpuPm4Analyzer(unittest.TestCase):
    def test_pm4_header_decoding(self):
        # 0xC0001000: Type 3, count=0 (body=1, total=2), opcode 0x10 (NOP)
        t, count, op, total_len = triage_crash.decode_pm4_header(0xC0001000)
        self.assertEqual(t, 3)
        self.assertEqual(count, 0)
        self.assertEqual(op, 0x10)
        self.assertEqual(total_len, 2)
        self.assertIn("NOP", triage_crash.get_pm4_opcode_name(op))

        # 0xC0012D00: Type 3, count=1 (body=2, total=3), opcode 0x2D (DRAW_INDEX_AUTO)
        t, count, op, total_len = triage_crash.decode_pm4_header(0xC0012D00)
        self.assertEqual(t, 3)
        self.assertEqual(count, 1)
        self.assertEqual(op, 0x2D)
        self.assertEqual(total_len, 3)
        self.assertIn("DRAW_INDEX_AUTO", triage_crash.get_pm4_opcode_name(op))

    def test_zero_gap_detection_excludes_nop_bodies(self):
        # DCB with a NOP packet of 4 dwords where body has 0s:
        # DWORD 0: 0xC0021000 (Type 3 NOP, count=2 -> 4 dwords total)
        # DWORD 1: 0x00000000 (NOP body word 1)
        # DWORD 2: 0x00000000 (NOP body word 2)
        # DWORD 3: 0x00000000 (NOP body word 3)
        # Followed by a real packet:
        # DWORD 4: 0xC0002D00 (DRAW_INDEX_AUTO, 2 dwords)
        # DWORD 5: 0x12345678
        # Followed by unpadded zeroes (gap):
        # DWORD 6: 0x00000000
        # DWORD 7: 0x00000000
        # Followed by another packet:
        # DWORD 8: 0xC0001000 (NOP, 2 dwords)
        # DWORD 9: 0x00000000
        # Followed by trailing zeroes:
        # DWORD 10: 0x00000000
        # DWORD 11: 0x00000000
        log_content = """[DCB FULL DUMP START 0x1000 count=12]
DWORD[1000][0]=0xc0021000
DWORD[1000][1]=0x00000000
DWORD[1000][2]=0x00000000
DWORD[1000][3]=0x00000000
DWORD[1000][4]=0xc0002d00
DWORD[1000][5]=0x12345678
DWORD[1000][6]=0x00000000
DWORD[1000][7]=0x00000000
DWORD[1000][8]=0xc0001000
DWORD[1000][9]=0x00000000
DWORD[1000][10]=0x00000000
DWORD[1000][11]=0x00000000
[DCB FULL DUMP END 0x1000]
"""
        log = Log(log_content.splitlines(keepends=True))
        gpu = triage_crash.GpuPm4Analyzer(log)
        self.assertEqual(len(gpu.dcb_results), 1)
        res = gpu.dcb_results[0]
        # Only the gap at DWORD index 6 (length 2) should be reported, NOT the NOP body at indices 1..3
        self.assertEqual(len(res.zero_runs_unpadded), 1)
        self.assertEqual(res.zero_runs_unpadded[0], (6, 2))
        # Trailing zero run at index 10 (length 2)
        self.assertEqual(res.trailing_zero_run, (10, 2))

    def test_slice_matching(self):
        # DCB starting at 0x1000:
        # Slot 0 (addr 0x1000, 4 dwords): NOP padded
        # Slot 1 (addr 0x1010, 4 dwords): all-zero
        # Slot 2 (addr 0x1020, 3 dwords): real DRAW_INDEX_AUTO packet
        log_content = """[DCB FULL DUMP START 0x1000 count=11]
DWORD[1000][0]=0xc0021000
DWORD[1000][1]=0x00000000
DWORD[1000][2]=0x00000000
DWORD[1000][3]=0x00000000
DWORD[1000][4]=0x00000000
DWORD[1000][5]=0x00000000
DWORD[1000][6]=0x00000000
DWORD[1000][7]=0x00000000
DWORD[1000][8]=0xc0012d00
DWORD[1000][9]=0x11111111
DWORD[1000][10]=0x22222222
[DCB FULL DUMP END 0x1000]
"""
        log = Log(log_content.splitlines(keepends=True))
        gpu = triage_crash.GpuPm4Analyzer(log)
        dcb = log.dcb_dumps[0]

        # Slot 0: NOP padded
        status, desc = gpu.match_slice(dcb, rdi=0x1000, rsi=4)
        self.assertEqual(status, "ok")
        self.assertIn("NOP of exactly 4 DWORDs (stub padded)", desc)

        # Slot 1: all-zero
        status, desc = gpu.match_slice(dcb, rdi=0x1010, rsi=4)
        self.assertEqual(status, "ok")
        self.assertIn("all-zero (stub never wrote)", desc)

        # Slot 2: real packets
        status, desc = gpu.match_slice(dcb, rdi=0x1020, rsi=3)
        self.assertEqual(status, "ok")
        self.assertIn("real packets", desc)
        self.assertIn("DRAW_INDEX_AUTO", desc)

        # Out of bounds or rsi > 65536
        status, desc = gpu.match_slice(dcb, rdi=0x9000, rsi=4)
        self.assertEqual(status, "not_a_slot")
        status, desc = gpu.match_slice(dcb, rdi=0x1000, rsi=70000)
        self.assertEqual(status, "not_a_slot")

    def test_pm4_label_mismatch_detection(self):
        log_content = """[PM4] IT_LOAD_CONFIG_REG (0x79)
[PM4] IT_RELEASE_MEM (0x47)
"""
        log = Log(log_content.splitlines(keepends=True))
        gpu = triage_crash.GpuPm4Analyzer(log)
        self.assertEqual(len(gpu.label_mismatches), 2)
        mismatch_findings = [f for f in gpu.findings if "PM4 label mismatch" in f.message]
        self.assertEqual(len(mismatch_findings), 2)
        self.assertIn("SET_UCONFIG_REG", mismatch_findings[0].message)
        self.assertIn("EVENT_WRITE_EOP", mismatch_findings[1].message)

    def test_timeout_and_stall_detection(self):
        log_content = """[Gnm] SubmitAndFlip(count=1, dcbAddrs=0x1000, dcbSizes=0x100)
[shutdown] watchdog: threads still parked after 2s, timeout stall detected
"""
        log = Log(log_content.splitlines(keepends=True))
        gpu = triage_crash.GpuPm4Analyzer(log)
        self.assertEqual(len(gpu.timeout_lines), 1)
        timeout_findings = [f for f in gpu.findings if "Flip/wait/timeout/stall events" in f.message]
        self.assertEqual(len(timeout_findings), 1)
        self.assertIn("F#1", timeout_findings[0].message)


class TestStage6HotLoopAndThreads(unittest.TestCase):
    def test_hot_loop_from_histogram(self):
        log_content = """[module] 0x1 eboot.bin: base=0x500000 span=0x200000
=== RIP Histogram ===
0x588df6: 6000
0x58d28e: 3000
0x58d550: 1000
"""
        log = Log(log_content.splitlines(keepends=True))
        loader = triage_crash.LoaderAnalyzer(log)
        hot = triage_crash.HotLoopAnalyzer(log, modules=loader.modules)
        self.assertTrue(hot.hot_loop_detected)
        self.assertEqual(hot.total_samples, 10000)
        # Top 1 address (0x588df6) has 6000/10000 = 60% (>50%)
        self.assertEqual(len(hot.hot_addrs), 1)
        self.assertAlmostEqual(hot.hot_share, 0.60)
        self.assertEqual(len(hot.findings), 1)
        self.assertIn("eboot.bin+0x88DF6", hot.findings[0].message)
        self.assertIn("60.0%", hot.findings[0].message)

    def test_hot_loop_from_trace_lines(self):
        trace_lines = ["[CPU TRACE] RIP=0x1000\n"] * 60 + ["[CPU TRACE] RIP=0x2000\n"] * 40
        log = Log(trace_lines)
        hot = triage_crash.HotLoopAnalyzer(log)
        self.assertTrue(hot.hot_loop_detected)
        self.assertEqual(len(hot.hot_addrs), 1)
        self.assertAlmostEqual(hot.hot_share, 0.60)
        self.assertIn("0x1000", hot.findings[0].message)

    def test_thread_activity_and_semaphore_balance(self):
        log_content = """[Thread] Created guest thread 'main' (TID 1): entry=0x100, stack=0x1000-0x2000, rsp=0x1fff, fs_base=0x0, arg=0x0
[HLE TID=1] sceKernelCreateSema('sema1')
[Sema] CreateSema -> id=1 name='sema1' attr=0x1 init=0 max=1
[Sema] CreateSema -> id=2 name='sema2' attr=0x1 init=0 max=1
[Sema] SignalSema(1, 1) -> count=1
=== Thread States Dump (Total active: 2) ===
  [Thread 1] 'main' state=WaitingSync, RIP=0x8d2720, instrs=100
  [Thread 2] 'worker' state=WaitingSync, RIP=0x8d2720, instrs=200
[shutdown] watchdog: threads still parked after 2s, hard-exiting
"""
        log = Log(log_content.splitlines(keepends=True))
        threads = triage_crash.ThreadAnalyzer(log)
        self.assertEqual(threads.sema_created_count, 2)
        self.assertEqual(threads.sema_released_count, 1)
        sema_findings = [f for f in threads.findings if "Semaphore balance" in f.message]
        self.assertEqual(len(sema_findings), 1)
        self.assertIn("unreleased balance=1", sema_findings[0].message)

        deadlock_findings = [f for f in threads.findings if "Potential deadlock" in f.message]
        self.assertEqual(len(deadlock_findings), 1)
        self.assertEqual(deadlock_findings[0].kind, "INFERRED")


class TestStage7DiffMode(unittest.TestCase):
    def test_diff_no_behavioural_change(self):
        log_old_content = """[prx] mode=real
[CPU FAULT] RIP=0x123456, RSP=0x1ffffb000 (return=0x106840b)
[CPU] Execution Status: Memory fault / invalid instruction pointer
[CPU] Instructions Executed: 1000000
"""
        log_new_content = """[prx] mode=real
[CPU FAULT] RIP=0x123456, RSP=0x1ffffb000 (return=0x106840b)
[CPU] Execution Status: Memory fault / invalid instruction pointer
[CPU] Instructions Executed: 1200000
"""
        old_log = Log(log_old_content.splitlines(keepends=True))
        new_log = Log(log_new_content.splitlines(keepends=True))
        differ = triage_crash.DiffAnalyzer(old_log, new_log)
        self.assertTrue(differ.no_behavioural_change)
        table = differ.format_table()
        self.assertIn("no behavioural change", table)

    def test_diff_with_changes(self):
        log_old_content = """[prx] mode=hle
[CPU FAULT] RIP=0x111111, RSP=0x1ffffb000
[CPU] Execution Status: Instruction executed successfully
[CPU] Instructions Executed: 1000000
[HLE WARNING #1 TID=1 rip=0x100] Unimplemented stub called: NID=0xaaaaaaaaaaaaaaaa
[HLE] sceKernelUsleep(100)
[HLE] sceKernelUsleep(100)
"""
        log_new_content = """[prx] mode=real
[CPU FAULT] RIP=0x222222, RSP=0x1ffffb000
[CPU] Execution Status: Memory fault / invalid instruction pointer
[CPU] Instructions Executed: 2000000
[HLE WARNING #1 TID=1 rip=0x100] Unimplemented stub called: NID=0xbbbbbbbbbbbbbbbb
[HLE] sceKernelUsleep(100)
[HLE] sceKernelUsleep(100)
[HLE] sceKernelUsleep(100)
[HLE] sceKernelUsleep(100)
[DCB FULL DUMP START 0x1000 count=3]
DWORD[1000][0]=0xc0012d00
DWORD[1000][1]=0x00000000
DWORD[1000][2]=0x00000000
[DCB FULL DUMP END 0x1000]
"""
        old_log = Log(log_old_content.splitlines(keepends=True))
        new_log = Log(log_new_content.splitlines(keepends=True))
        differ = triage_crash.DiffAnalyzer(old_log, new_log)
        self.assertFalse(differ.no_behavioural_change)

        d = differ.diff_data
        self.assertTrue(d["mode"]["changed"])
        self.assertEqual(d["instructions"]["delta"], 1000000)
        self.assertEqual(len(d["fallback_nids"]["added"]), 1)
        self.assertEqual(d["fallback_nids"]["added"][0]["nid"], "0xBBBBBBBBBBBBBBBB")
        self.assertEqual(len(d["fallback_nids"]["removed"]), 1)
        self.assertEqual(d["fallback_nids"]["removed"][0]["nid"], "0xAAAAAAAAAAAAAAAA")
        self.assertEqual(d["dcb_draws"]["delta"], 1)

        # HLE call count change: sceKernelUsleep changed from 2 to 4 (+100.0% > 20%)
        usleep_chgs = [c for c in d["hle_call_changes_gt_20pct"] if "sceKernelUsleep" in c["call"]]
        self.assertEqual(len(usleep_chgs), 1)
        self.assertEqual(usleep_chgs[0]["old"], 2)
        self.assertEqual(usleep_chgs[0]["new"], 4)


if __name__ == "__main__":
    unittest.main()
