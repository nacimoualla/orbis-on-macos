#!/usr/bin/env python3
"""
scripts/triage_crash.py - Triage crash logs from x86-64 to ARM64 PS4 emulator.

Python 3.9+, standard library only (capstone optional if importable).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# Capstone is optional and only used if importable
try:
    import capstone
    HAVE_CAPSTONE = True
except ImportError:
    HAVE_CAPSTONE = False


@dataclass
class Finding:
    kind: str  # "OBSERVED" or "INFERRED"
    message: str
    lines: List[int] = field(default_factory=list)
    rule: Optional[str] = None

    def __str__(self) -> str:
        if self.kind == "OBSERVED":
            lines_str = ",".join(str(l) for l in sorted(self.lines)) if self.lines else "not present in log"
            return f"[OBSERVED L{lines_str}] {self.message}"
        else:
            rule_str = f" by {self.rule}" if self.rule else ""
            return f"[INFERRED{rule_str}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": self.kind,
            "message": self.message,
        }
        if self.lines:
            d["lines"] = sorted(self.lines)
        if self.rule:
            d["rule"] = self.rule
        return d


@dataclass
class DcbDump:
    start_line: int
    end_line: int
    addr: int
    count: int
    dwords: List[int] = field(default_factory=list)


class Log:
    """
    Single-pass streaming parser for PS4 emulator logs.
    Indexes lines by regex into typed records without loading the whole file into memory.
    """

    def __init__(self, filepath_or_stream: Any):
        self.mode: str = "unknown"
        self.mode_lines: List[int] = []
        self.budget: Optional[int] = None
        self.budget_lines: List[int] = []
        self.final_instructions: Optional[int] = None
        self.final_instructions_lines: List[int] = []
        self.stop_reason: Optional[str] = None
        self.stop_reason_lines: List[int] = []
        self.wall_time: Optional[str] = None
        self.wall_time_lines: List[int] = []

        # Typed records
        self.cpu_fault_lines: List[Tuple[int, str]] = []
        self.unhandled_opcode_lines: List[Tuple[int, str]] = []
        self.hle_lines: List[Tuple[int, str]] = []
        self.dcb_dumps: List[DcbDump] = []
        self.pm4_lines: List[Tuple[int, str]] = []
        self.loader_lines: List[Tuple[int, str]] = []
        self.watch_lines: List[Tuple[int, str]] = []
        self.thread_lines: List[Tuple[int, str]] = []
        self.trace_lines: List[Tuple[int, str]] = []
        self.status_lines: List[Tuple[int, str]] = []
        self.other_recognized_lines: List[Tuple[int, str]] = []

        self.total_lines: int = 0
        self.unparsed_lines: int = 0
        self.unparsed_sample: List[Tuple[int, str]] = []

        self._parse(filepath_or_stream)

    def _parse(self, source: Any) -> None:
        if isinstance(source, (str, os.PathLike)):
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                self._stream_lines(f)
        else:
            self._stream_lines(source)

    def _stream_lines(self, stream: Any) -> None:
        # Regexes for indexing
        re_mode = re.compile(r"\[prx\]\s+mode=(real|hle)", re.IGNORECASE)
        re_budget = re.compile(r"(?:budget[:=]\s*(\d+)|--budget\s+(\d+))", re.IGNORECASE)
        re_cpu_status = re.compile(r"\[CPU\]\s+Execution Status:\s*(.+)", re.IGNORECASE)
        re_cpu_instrs = re.compile(r"\[CPU\]\s+Instructions Executed:\s*(\d+)", re.IGNORECASE)
        re_wall_time = re.compile(r"(?:wall[\s_-]?time|elapsed|duration)[:=]\s*([\d\.]+(?:s|ms|m)?)", re.IGNORECASE)

        # DCB dump markers
        re_dcb_start = re.compile(r"\[DCB FULL DUMP START 0x([0-9a-fA-F]+)\s+count=(\d+)\]")
        re_dcb_dword = re.compile(r"DWORD\[([0-9a-fA-F]+)\]\[(\d+)\]=0x([0-9a-fA-F]+)")
        re_dcb_end = re.compile(r"\[DCB FULL DUMP END 0x([0-9a-fA-F]+)\]")

        current_dcb: Optional[DcbDump] = None

        for line_no, raw_line in enumerate(stream, 1):
            self.total_lines += 1
            line = raw_line.strip()
            recognized = False

            # Check DCB active dump first
            if current_dcb is not None:
                m_dword = re_dcb_dword.match(line)
                if m_dword:
                    val = int(m_dword.group(3), 16)
                    current_dcb.dwords.append(val)
                    continue
                m_end = re_dcb_end.match(line)
                if m_end:
                    current_dcb.end_line = line_no
                    self.dcb_dumps.append(current_dcb)
                    current_dcb = None
                    continue
                # If neither matched inside DCB dump, fall through to regular checks

            # DCB dump start
            m_dcb_start = re_dcb_start.match(line)
            if m_dcb_start:
                addr = int(m_dcb_start.group(1), 16)
                count = int(m_dcb_start.group(2))
                current_dcb = DcbDump(start_line=line_no, end_line=line_no, addr=addr, count=count, dwords=[])
                continue

            # Mode
            m_mode = re_mode.search(line)
            if m_mode:
                self.mode = m_mode.group(1).lower()
                self.mode_lines.append(line_no)
                recognized = True

            # Budget
            m_b = re_budget.search(line)
            if m_b:
                self.budget = int(m_b.group(1) or m_b.group(2))
                self.budget_lines.append(line_no)
                recognized = True

            # Execution status
            m_st = re_cpu_status.search(line)
            if m_st:
                self.stop_reason = m_st.group(1).strip()
                self.stop_reason_lines.append(line_no)
                recognized = True

            # Instructions executed
            m_ins = re_cpu_instrs.search(line)
            if m_ins:
                self.final_instructions = int(m_ins.group(1))
                self.final_instructions_lines.append(line_no)
                recognized = True

            # Wall time
            m_wt = re_wall_time.search(line)
            if m_wt:
                self.wall_time = m_wt.group(1).strip()
                self.wall_time_lines.append(line_no)
                recognized = True

            # CPU Fault
            if "[CPU FAULT]" in line or "CPU FAULT" in line:
                self.cpu_fault_lines.append((line_no, line))
                recognized = True

            # Unhandled opcode
            if "unhandled opcode" in line.lower():
                self.unhandled_opcode_lines.append((line_no, line))
                recognized = True

            # HLE
            if (line.startswith("[HLE") or line.startswith("[HLE CALL") or
                    line.startswith("[HLE WARNING") or line.startswith("[HLE WHITELIST") or
                    "Unimplemented stub called:" in line):
                self.hle_lines.append((line_no, line))
                recognized = True

            # GPU / PM4
            if (line.startswith("[PM4") or line.startswith("[Gnm]") or
                    line.startswith("[Gnm Warning]") or line.startswith("[PM4 Warning]")):
                self.pm4_lines.append((line_no, line))
                recognized = True

            # Loader / Module / Map
            if (line.startswith("[module]") or line.startswith("[exports]") or
                    line.startswith("[dyn]") or line.startswith("[map]") or
                    line.startswith("[elf]")):
                self.loader_lines.append((line_no, line))
                recognized = True

            # Watch
            if line.startswith("[watch]"):
                self.watch_lines.append((line_no, line))
                recognized = True

            # Thread / Sema
            if (line.startswith("[Thread") or line.startswith("[Threads]") or
                    line.startswith("[Sema]") or "=== Thread States Dump" in line or
                    "=== [SEMA] table snapshot" in line):
                self.thread_lines.append((line_no, line))
                recognized = True

            # CPU Trace / Progress
            if line.startswith("[CPU TRACE]") or line.startswith("[CPU Progress]"):
                self.trace_lines.append((line_no, line))
                recognized = True

            # CPU Status / Registers
            if (line.startswith("[CPU]") or line.startswith("-------------------- CPU Register Context") or
                    line.startswith("RIP:") or line.startswith("RAX:") or line.startswith("RSI:") or
                    line.startswith("R8:") or line.startswith("R12:") or line.startswith("FS_BASE:")):
                self.status_lines.append((line_no, line))
                recognized = True

            # Common emulator header / guest lines that are recognized
            if (line.startswith("====") or line.startswith(" ps4-to-macos:") or
                    line.startswith("Loading:") or line.startswith("Entry Point:") or
                    line.startswith("Type:") or line.startswith("Program Headers:") or
                    line.startswith("[Idx]") or line.startswith("-----") or
                    line.startswith("[0") or line.startswith("[1") or line.startswith("[2") or
                    line.startswith("[Mono]") or line.startswith("[FIOS2]") or
                    line.startswith("[LIBC") or line.startswith("[Guest") or
                    line.startswith("[ScreenManager]") or line.startswith("[DirectMem]") or
                    line.startswith("[DEBUG]") or line.startswith("[shutdown]") or
                    line.startswith("[xfer]") or line.startswith("Forcing submitDone") or
                    not line):
                self.other_recognized_lines.append((line_no, line))
                recognized = True

            if not recognized:
                self.unparsed_lines += 1
                if len(self.unparsed_sample) < 5:
                    self.unparsed_sample.append((line_no, line))

        # Close unclosed DCB dump if EOF reached
        if current_dcb is not None:
            current_dcb.end_line = self.total_lines
            self.dcb_dumps.append(current_dcb)

    def banner_dict(self) -> Dict[str, Any]:
        """
        Returns structured banner data:
        mode (real/hle/unknown), instruction budget, final instruction count, stop reason, wall time.
        """
        note = None
        if self.mode == "hle":
            note = "Mono never executed in that run (HLE mode)"
        elif self.mode == "unknown":
            note = "mode unknown"

        return {
            "mode": self.mode,
            "mode_lines": self.mode_lines,
            "instruction_budget": self.budget if self.budget is not None else "not present in log",
            "instruction_budget_lines": self.budget_lines,
            "final_instruction_count": self.final_instructions if self.final_instructions is not None else "not present in log",
            "final_instruction_lines": self.final_instructions_lines,
            "stop_reason": self.stop_reason if self.stop_reason is not None else "not present in log",
            "stop_reason_lines": self.stop_reason_lines,
            "wall_time": self.wall_time if self.wall_time is not None else "not present in log",
            "wall_time_lines": self.wall_time_lines,
            "note": note,
            "total_lines": self.total_lines,
            "unparsed_lines": self.unparsed_lines,
        }

    def banner_str(self) -> str:
        b = self.banner_dict()
        lines = [
            "=== BANNER ===",
            f"Mode:                    {b['mode']}" + (f" (L{','.join(map(str, b['mode_lines']))})" if b['mode_lines'] else ""),
            f"Instruction Budget:      {b['instruction_budget']}",
            f"Final Instruction Count: {b['final_instruction_count']}",
            f"Stop Reason:             {b['stop_reason']}",
            f"Wall Time:               {b['wall_time']}",
            f"Total Lines:             {b['total_lines']} (unparsed: {b['unparsed_lines']})",
        ]
        if b["note"]:
            lines.append(f"NOTE:                    {b['note']}")
        return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage PS4-to-macOS emulator logs")
    parser.add_argument("log", nargs="?", help="Log file to triage")
    args = parser.parse_args()

    if not args.log:
        parser.print_help()
        return 0

    log = Log(args.log)
    print(log.banner_str())
    return 0


if __name__ == "__main__":
    sys.exit(main())
