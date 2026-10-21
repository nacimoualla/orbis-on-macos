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
from triage_crash import Log, Finding


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


if __name__ == "__main__":
    unittest.main()
