#!/usr/bin/env python3
"""
scripts/triage_crash.py - Triage crash logs from x86-64 to ARM64 PS4 emulator.

Python 3.9+, standard library only (capstone optional if importable).
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from dataclasses import dataclass, field
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set

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


# Built-in SDM Table for x87 instructions (Memory forms: mod != 3)
X87_MODRM_MEM: Dict[int, Dict[int, str]] = {
    0xD8: {0: "FADD m32fp", 1: "FMUL m32fp", 2: "FCOM m32fp", 3: "FCOMP m32fp", 4: "FSUB m32fp", 5: "FSUBR m32fp", 6: "FDIV m32fp", 7: "FDIVR m32fp"},
    0xD9: {0: "FLD m32fp", 1: "UNKNOWN", 2: "FST m32fp", 3: "FSTP m32fp", 4: "FLDENV m14/28byte", 5: "FLDCW", 6: "FNSTENV m14/28byte", 7: "FNSTCW"},
    0xDA: {0: "FIADD m32int", 1: "FIMUL m32int", 2: "FICOM m32int", 3: "FICOMP m32int", 4: "FISUB m32int", 5: "FISUBR m32int", 6: "FIDIV m32int", 7: "FIDIVR m32int"},
    0xDB: {0: "FILD m32int", 1: "FISTTP m32int", 2: "FIST m32int", 3: "FISTP m32int", 4: "UNKNOWN", 5: "FLD m80", 6: "UNKNOWN", 7: "FSTP m80"},
    0xDC: {0: "FADD m64fp", 1: "FMUL m64fp", 2: "FCOM m64fp", 3: "FCOMP m64fp", 4: "FSUB m64fp", 5: "FSUBR m64fp", 6: "FDIV m64fp", 7: "FDIVR m64fp"},
    0xDD: {0: "FLD m64fp", 1: "FISTTP m64int", 2: "FST m64fp", 3: "FSTP m64fp", 4: "FRSTOR m94/108byte", 5: "UNKNOWN", 6: "FSAVE m94/108byte", 7: "FNSTSW m2byte"},
    0xDE: {0: "FIADD m16int", 1: "FIMUL m16int", 2: "FICOM m16int", 3: "FICOMP m16int", 4: "FISUB m16int", 5: "FISUBR m16int", 6: "FIDIV m16int", 7: "FIDIVR m16int"},
    0xDF: {0: "FILD m16int", 1: "FISTTP m16int", 2: "FIST m16int", 3: "FISTP m16int", 4: "FBLD m80dec", 5: "FILD m64int", 6: "FBSTP", 7: "FISTP m64int"},
}

# Register forms (mod == 3)
X87_MODRM_REG: Dict[int, Dict[int, str]] = {
    0xD8: {i: f"FADD ST(0), ST({i})" for i in range(8)},
    0xD9: {0: "FLD ST(0)", 1: "FLD ST(1)", 2: "FLD ST(2)", 3: "FLD ST(3)", 4: "FLD ST(4)", 5: "FLD ST(5)", 6: "FLD ST(6)", 7: "FLD ST(7)"},
    0xDB: {0xE3: "FNINIT"},
    0xDD: {0: "FFREE ST(0)", 1: "FFREE ST(1)", 2: "FFREE ST(2)", 3: "FFREE ST(3)", 4: "FFREE ST(4)", 5: "FFREE ST(5)", 6: "FFREE ST(6)", 7: "FFREE ST(7)"},
    0xDF: {0xE0: "FNSTSW AX"},
}


def decode_x86_rm_operand(mod: int, rm: int, code_bytes: bytes, offset: int, default_size: str = "") -> Tuple[str, int]:
    """
    Decodes the memory/register operand from ModR/M and subsequent disp bytes.
    Returns (operand_str, bytes_consumed).
    """
    regs64 = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"]
    prefix = f"{default_size} " if default_size else ""

    if mod == 3:
        return regs64[rm], 0

    disp_str = ""
    consumed = 0

    # SIB byte check
    if rm == 4:
        if offset < len(code_bytes):
            sib = code_bytes[offset]
            consumed += 1
            base_idx = sib & 7
            index_idx = (sib >> 3) & 7
            scale = 1 << ((sib >> 6) & 3)
            base_name = regs64[base_idx] if not (mod == 0 and base_idx == 5) else ""
            index_name = regs64[index_idx] if index_idx != 4 else ""
            parts = []
            if base_name:
                parts.append(base_name)
            if index_name:
                parts.append(f"{index_name}*{scale}" if scale > 1 else index_name)
            base_part = "+".join(parts) if parts else ""
        else:
            base_part = "rsp"
    elif mod == 0 and rm == 5:
        base_part = "rip"
    else:
        base_part = regs64[rm]

    if mod == 0:
        if rm == 5 or (rm == 4 and (code_bytes[offset - 1] & 7) == 5):
            if offset + consumed + 4 <= len(code_bytes):
                d = struct.unpack("<i", code_bytes[offset + consumed:offset + consumed + 4])[0]
                consumed += 4
                disp_str = f"+0x{d:X}" if d >= 0 else f"-0x{-d:X}"
    elif mod == 1:
        if offset + consumed + 1 <= len(code_bytes):
            d = struct.unpack("<b", code_bytes[offset + consumed:offset + consumed + 1])[0]
            consumed += 1
            disp_str = f"+0x{d:X}" if d >= 0 else f"-0x{-d:X}"
    elif mod == 2:
        if offset + consumed + 4 <= len(code_bytes):
            d = struct.unpack("<i", code_bytes[offset + consumed:offset + consumed + 4])[0]
            consumed += 4
            disp_str = f"+0x{d:X}" if d >= 0 else f"-0x{-d:X}"

    if not base_part and disp_str:
        inner = disp_str.lstrip("+")
    else:
        inner = f"{base_part}{disp_str}"

    return f"{prefix}[{inner}]", consumed


def decode_unhandled_opcode_bytes(raw_bytes: bytes) -> Tuple[str, str, str]:
    """
    Decodes an unhandled opcode sequence.
    Returns (decoded_string, encoding_label, opcode_class).
    """
    if not raw_bytes:
        return "empty opcode", "", "unknown"

    first = raw_bytes[0]

    # Check x87 (D8..DF)
    if 0xD8 <= first <= 0xDF and len(raw_bytes) >= 2:
        modrm = raw_bytes[1]
        mod = (modrm >> 6) & 3
        reg = (modrm >> 3) & 7
        rm = modrm & 7
        encoding_label = f"{first:02X} /{reg}"

        if mod == 3:
            mnemonic = X87_MODRM_REG.get(first, {}).get(modrm, None)
            if not mnemonic:
                mnemonic = X87_MODRM_REG.get(first, {}).get(reg, f"x87_reg_{first:02X}")
            return mnemonic, encoding_label, "x87"
        else:
            mnemonic = X87_MODRM_MEM.get(first, {}).get(reg, f"x87_mem_{first:02X}")
            size_prefix = ""
            if "m32" in mnemonic or "m32fp" in mnemonic or "m32int" in mnemonic:
                size_prefix = "dword ptr"
            elif "m64" in mnemonic or "m64fp" in mnemonic or "m64int" in mnemonic:
                size_prefix = "qword ptr"
            elif "m80" in mnemonic or "m80fp" in mnemonic:
                size_prefix = "tbyte ptr"
            elif "m16" in mnemonic or "m16int" in mnemonic or "m2byte" in mnemonic or "CW" in mnemonic or "SW" in mnemonic:
                size_prefix = "word ptr"

            base_mnemonic = mnemonic.split()[0]
            op_mem, _ = decode_x86_rm_operand(mod, rm, raw_bytes, 2, default_size=size_prefix)
            decoded_str = f"{base_mnemonic} {op_mem} ({encoding_label})"
            return decoded_str, encoding_label, "x87"

    # Multi-byte NOPs (0F 1F)
    if len(raw_bytes) >= 2 and raw_bytes[0] == 0x0F and raw_bytes[1] == 0x1F:
        reg = (raw_bytes[2] >> 3) & 7 if len(raw_bytes) >= 3 else 0
        return "NOP (multi-byte)", f"0F 1F /{reg}", "system"

    # SSE / AVX
    if first in (0xC4, 0xC5):
        return f"VEX-encoded instruction ({raw_bytes[:4].hex()})", f"{first:02X}", "SSE/AVX"
    if first == 0x62:
        return f"EVEX-encoded instruction ({raw_bytes[:4].hex()})", "62", "SSE/AVX"

    if first in (0x66, 0xF2, 0xF3) and len(raw_bytes) >= 2 and raw_bytes[1] == 0x0F:
        op = raw_bytes[2] if len(raw_bytes) >= 3 else raw_bytes[1]
        return f"SSE instruction prefix=0x{first:02x} op=0x{op:02x}", f"{first:02X} 0F {op:02X}", "SSE/AVX"
    if first == 0x0F and len(raw_bytes) >= 2:
        op = raw_bytes[1]
        if op in (0x01, 0x05, 0x34, 0x06, 0x08, 0x09, 0x20, 0x21, 0x22, 0x23, 0x77):
            return f"System instruction 0F {op:02X}", f"0F {op:02X}", "system"
        return f"2-byte opcode 0F {op:02X}", f"0F {op:02X}", "SSE/AVX"

    # String operations (MOVS, CMPS, STOS, LODS, SCAS)
    if first in (0xA4, 0xA5, 0xA6, 0xA7, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF):
        return f"String operation 0x{first:02X}", f"{first:02X}", "string op"
    if first in (0xF2, 0xF3) and len(raw_bytes) >= 2 and raw_bytes[1] in (0xA4, 0xA5, 0xA6, 0xA7, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF):
        return f"REP-prefixed string operation 0x{first:02X} 0x{raw_bytes[1]:02X}", f"{first:02X} {raw_bytes[1]:02X}", "string op"

    return f"raw bytes: {raw_bytes[:8].hex(' ')}", f"{first:02X}", "unknown"


class CpuFaultAnalyzer:
    """
    STAGE 2: CpuFaultAnalyzer
    Parses RIP, RSP, return address, registers, faulting module, bytes at RIP,
    classifies the fault address, resolves bad indirect calls (call rel32 + PLT jmp),
    decodes unhandled opcodes, and checks segfault vs guest OOB.
    """

    def __init__(self, log: Log, repo_dir: Optional[str] = None):
        self.log = log
        self.repo_dir = repo_dir
        self.rip: Optional[int] = None
        self.rip_lines: List[int] = []
        self.rsp: Optional[int] = None
        self.rsp_lines: List[int] = []
        self.return_addr: Optional[int] = None
        self.return_addr_lines: List[int] = []
        self.registers: Dict[str, int] = {}
        self.register_lines: List[int] = []

        self.fault_module_name: Optional[str] = None
        self.fault_module_offset: Optional[int] = None
        self.fault_module_lines: List[int] = []

        self.bytes_at_rip: Optional[bytes] = None
        self.bytes_at_rip_lines: List[int] = []

        self.call_site: Optional[int] = None
        self.call_bytes: Optional[bytes] = None
        self.call_target_bytes: Optional[bytes] = None
        self.indirect_call_lines: List[int] = []

        self.unhandled_opcode_raw: Optional[bytes] = None
        self.unhandled_opcode_rip: Optional[int] = None
        self.unhandled_opcode_lines: List[int] = []

        self.segfault_detected: bool = False
        self.guest_oob_detected: bool = False
        self.mem_fault_addr: Optional[int] = None
        self.mem_fault_size: Optional[int] = None
        self.mem_fault_lines: List[int] = []

        self.findings: List[Finding] = []
        self._analyze()

    def _analyze(self) -> None:
        self._parse_fault_card()
        self._parse_unhandled_opcodes()
        self._parse_segfault_oob()
        self._evaluate()

    def _parse_fault_card(self) -> None:
        re_fault_rip = re.compile(
            r"\[CPU FAULT\]\s+(?:Invalid execution target\s+)?RIP=0x([0-9a-fA-F]+)(?:,\s*RSP=0x([0-9a-fA-F]+))?(?:\s*\(return=0x([0-9a-fA-F]+)\))?",
            re.IGNORECASE
        )
        re_return_addr = re.compile(r"Return address on stack:\s*0x([0-9a-fA-F]+)", re.IGNORECASE)
        re_fault_mod = re.compile(r"\[CPU FAULT\]\s+module handle=0x[0-9a-fA-F]+\s+path=([^\s]+)\s+base=0x([0-9a-fA-F]+)\s+module\+0x([0-9a-fA-F]+)", re.IGNORECASE)
        re_fault_mod_none = re.compile(r"\[CPU FAULT\]\s+module=<none:\s*not inside any loaded module>", re.IGNORECASE)

        re_instr_before = re.compile(r"\[CPU FAULT\]\s+instruction before return:\s*@0x([0-9a-fA-F]+)\s+((?:[0-9a-fA-F]{2}\s*)+)", re.IGNORECASE)
        re_target_bytes = re.compile(r"\[CPU FAULT\]\s+bytes at call target:\s*((?:[0-9a-fA-F]{2}\s*)+)", re.IGNORECASE)

        re_reg_line = re.compile(r"^\s*(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r8|r9|r10|r11|r12|r13|r14|r15|rip|rflags|fs_base|gs_base)\s*[:=]\s*0x([0-9a-fA-F]+)", re.IGNORECASE)

        code_at_rip_following = False
        for line_no, line in self.log.cpu_fault_lines:
            m_rip = re_fault_rip.search(line)
            if m_rip:
                self.rip = int(m_rip.group(1), 16)
                self.rip_lines.append(line_no)
                if m_rip.group(2):
                    self.rsp = int(m_rip.group(2), 16)
                    self.rsp_lines.append(line_no)
                if m_rip.group(3):
                    self.return_addr = int(m_rip.group(3), 16)
                    self.return_addr_lines.append(line_no)

            m_ret = re_return_addr.search(line)
            if m_ret and self.return_addr is None:
                self.return_addr = int(m_ret.group(1), 16)
                self.return_addr_lines.append(line_no)

            m_mod = re_fault_mod.search(line)
            if m_mod:
                self.fault_module_name = m_mod.group(1)
                self.fault_module_offset = int(m_mod.group(3), 16)
                self.fault_module_lines.append(line_no)
            elif re_fault_mod_none.search(line):
                self.fault_module_name = "not in any loaded module"
                self.fault_module_lines.append(line_no)

            m_ib = re_instr_before.search(line)
            if m_ib:
                self.call_site = int(m_ib.group(1), 16)
                hex_str = "".join(m_ib.group(2).split())
                self.call_bytes = bytes.fromhex(hex_str)
                self.indirect_call_lines.append(line_no)

            m_tb = re_target_bytes.search(line)
            if m_tb:
                hex_str = "".join(m_tb.group(1).split())
                self.call_target_bytes = bytes.fromhex(hex_str)
                self.indirect_call_lines.append(line_no)

            if "[CPU FAULT] code AT rip:" in line:
                code_at_rip_following = True
                continue
            if code_at_rip_following:
                hex_part = "".join(re.findall(r"[0-9a-fA-F]{2}", line))
                if hex_part:
                    self.bytes_at_rip = bytes.fromhex(hex_part)
                    self.bytes_at_rip_lines.append(line_no)
                code_at_rip_following = False

            for m in re_reg_line.finditer(line):
                reg_name = m.group(1).lower()
                reg_val = int(m.group(2), 16)
                self.registers[reg_name] = reg_val
                self.register_lines.append(line_no)

        for line_no, line in self.log.status_lines:
            for m in re_reg_line.finditer(line):
                reg_name = m.group(1).lower()
                reg_val = int(m.group(2), 16)
                if reg_name not in self.registers:
                    self.registers[reg_name] = reg_val
                    self.register_lines.append(line_no)
                if reg_name == "rip" and self.rip is None:
                    self.rip = reg_val
                    self.rip_lines.append(line_no)
                if reg_name == "rsp" and self.rsp is None:
                    self.rsp = reg_val
                    self.rsp_lines.append(line_no)

        re_bytes_rip = re.compile(r"Bytes at RIP:\s*((?:[0-9a-fA-F]{2}\s*)+)", re.IGNORECASE)
        for line_no, line in self.log.other_recognized_lines:
            m = re_bytes_rip.search(line)
            if m and self.bytes_at_rip is None:
                hex_str = "".join(m.group(1).split())
                self.bytes_at_rip = bytes.fromhex(hex_str)
                self.bytes_at_rip_lines.append(line_no)

    def _parse_unhandled_opcodes(self) -> None:
        re_unhandled = re.compile(
            r"Unhandled opcode at RIP=0x([0-9a-fA-F]+):\s*((?:[0-9a-fA-F]{2}\s*)+)",
            re.IGNORECASE
        )
        for line_no, line in self.log.unhandled_opcode_lines:
            m = re_unhandled.search(line)
            if m:
                self.unhandled_opcode_rip = int(m.group(1), 16)
                raw_hex = "".join(m.group(2).split())
                self.unhandled_opcode_raw = bytes.fromhex(raw_hex)
                self.unhandled_opcode_lines.append(line_no)
                break

    def _parse_segfault_oob(self) -> None:
        re_segv = re.compile(r"(?:segmentation fault|sigsegv|segfault)(?:.*?(?:address|at)\s*0x([0-9a-fA-F]+))?(?:.*?(?:size)\s*(\d+))?", re.IGNORECASE)
        re_oob = re.compile(r"(?:out-of-bounds|out of bounds)(?:.*?(?:address|at)\s*0x([0-9a-fA-F]+))?(?:.*?(?:size)\s*(\d+))?", re.IGNORECASE)

        for line_no, line in self.log.status_lines + self.log.other_recognized_lines + self.log.cpu_fault_lines + self.log.mem_fault_raw_lines:
            m_s = re_segv.search(line)
            if m_s:
                self.segfault_detected = True
                self.mem_fault_lines.append(line_no)
                if m_s.group(1):
                    self.mem_fault_addr = int(m_s.group(1), 16)
                if m_s.group(2):
                    self.mem_fault_size = int(m_s.group(2))

            m_o = re_oob.search(line)
            if m_o:
                self.guest_oob_detected = True
                self.mem_fault_lines.append(line_no)
                if m_o.group(1):
                    self.mem_fault_addr = int(m_o.group(1), 16)
                if m_o.group(2):
                    self.mem_fault_size = int(m_o.group(2))

    def classify_address(self, addr: int) -> str:
        if addr < 0x10000:
            return f"NULL-ish (<0x10000, 0x{addr:X})"

        if self.fault_module_name and self.fault_module_name != "not in any loaded module":
            moff = f"+0x{self.fault_module_offset:X}" if self.fault_module_offset is not None else ""
            return f"inside module {self.fault_module_name}{moff}"

        try:
            b_le = addr.to_bytes(8, "little")
            printable = all(32 <= c <= 126 or c == 0 for c in b_le)
            non_zero_printable = [c for c in b_le if 32 <= c <= 126]
            if printable and len(non_zero_printable) >= 4:
                decoded = bytes(non_zero_printable).decode("ascii", errors="replace")
                return f"looks like ASCII ('{decoded}', hex=0x{addr:016X})"
        except Exception:
            pass

        hi = (addr >> 32) & 0xFFFFFFFF
        lo = addr & 0xFFFFFFFF
        if hi != 0 and lo != 0 and (hi == lo or (hi > 0x1000 and lo > 0x1000)):
            return f"looks like two u32s (lo=0x{lo:08X}, hi=0x{hi:08X})"

        if 0x20000000 <= addr < 0x40000000:
            return f"guest libc heap range (0x{addr:X})"
        if 0x40000000 <= addr < 0x1C0000000:
            return f"guest direct/GPU memory range (0x{addr:X})"

        return f"unmapped (0x{addr:X})"

    def _resolve_bad_indirect_call(self) -> Optional[Tuple[int, int, int, List[int]]]:
        if self.call_bytes is not None and len(self.call_bytes) >= 5 and self.return_addr is not None:
            if self.call_bytes[0] == 0xE8:
                rel32 = struct.unpack("<i", self.call_bytes[1:5])[0]
                target = self.return_addr + rel32

                if self.call_target_bytes is not None and len(self.call_target_bytes) >= 6:
                    if self.call_target_bytes[0] == 0xFF and (self.call_target_bytes[1] & 0xC7) == 0x05:
                        disp32 = struct.unpack("<i", self.call_target_bytes[2:6])[0]
                        got_slot = target + 6 + disp32
                        return target, disp32, got_slot, self.indirect_call_lines

                return target, 0, 0, self.indirect_call_lines
        return None

    def _check_repo_handles_opcode(self, encoding_label: str) -> Optional[str]:
        if not self.repo_dir:
            return None
        executor_cpp = os.path.join(self.repo_dir, "src", "cpu", "executor.cpp")
        if not os.path.isfile(executor_cpp):
            return None

        parts = encoding_label.split()
        if not parts:
            return None
        primary_hex = parts[0]
        try:
            with open(executor_cpp, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                pattern = f"0x{primary_hex.lower()}"
                if pattern in content.lower():
                    return f"repo executor.cpp contains references to {primary_hex}"
                else:
                    return f"repo executor.cpp does not appear to handle {primary_hex}"
        except Exception as e:
            return f"error searching repo: {e}"

    def _evaluate(self) -> None:
        if self.rip is not None:
            all_lines = sorted(list(set(self.rip_lines + self.rsp_lines + self.return_addr_lines)))
            rsp_str = f"0x{self.rsp:X}" if self.rsp is not None else "not present in log"
            ret_str = f"0x{self.return_addr:X}" if self.return_addr is not None else "not present in log"
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Fault context: RIP=0x{self.rip:X}, RSP={rsp_str}, return={ret_str}",
                lines=all_lines
            ))

            classification = self.classify_address(self.rip)
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Fault address classification: {classification}",
                lines=self.rip_lines
            ))

        if self.fault_module_name:
            if self.fault_module_name == "not in any loaded module":
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message="Faulting module: not in any loaded module",
                    lines=self.fault_module_lines
                ))
            else:
                moff = f"+0x{self.fault_module_offset:X}" if self.fault_module_offset is not None else ""
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=f"Faulting module: {self.fault_module_name}{moff}",
                    lines=self.fault_module_lines
                ))

        if self.bytes_at_rip is not None:
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Bytes at RIP: {self.bytes_at_rip.hex(' ')}",
                lines=self.bytes_at_rip_lines
            ))

        resolved = self._resolve_bad_indirect_call()
        if resolved is not None:
            target, disp32, got_slot, lines = resolved
            if got_slot != 0:
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=(
                        f"Indirect call via PLT: call rel32 target=0x{target:X}, "
                        f"PLT jmp disp32=0x{disp32:X}, computed GOT slot=0x{got_slot:X}"
                    ),
                    lines=lines
                ))
            else:
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=f"Call target from rel32: 0x{target:X}",
                    lines=lines
                ))

        if self.unhandled_opcode_raw is not None:
            decoded, encoding_label, op_class = decode_unhandled_opcode_bytes(self.unhandled_opcode_raw)
            repo_status = self._check_repo_handles_opcode(encoding_label)
            msg = (
                f"Unhandled opcode at RIP=0x{self.unhandled_opcode_rip:X}: "
                f"{self.unhandled_opcode_raw.hex(' ')} -> {decoded} "
                f"[class: {op_class}, encoding: {encoding_label}]"
            )
            if repo_status:
                msg += f" ({repo_status})"
            self.findings.append(Finding(
                kind="OBSERVED",
                message=msg,
                lines=self.unhandled_opcode_lines
            ))

        if self.segfault_detected:
            addr_str = f"0x{self.mem_fault_addr:X}" if self.mem_fault_addr is not None else "not present in log"
            size_str = f"{self.mem_fault_size} bytes" if self.mem_fault_size is not None else "not present in log"
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Memory fault type: Host segfault (address={addr_str}, size={size_str})",
                lines=self.mem_fault_lines
            ))
        elif self.guest_oob_detected:
            addr_str = f"0x{self.mem_fault_addr:X}" if self.mem_fault_addr is not None else "not present in log"
            size_str = f"{self.mem_fault_size} bytes" if self.mem_fault_size is not None else "not present in log"
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Memory fault type: Guest out-of-bounds (address={addr_str}, size={size_str})",
                lines=self.mem_fault_lines
            ))
        else:
            self.findings.append(Finding(
                kind="OBSERVED",
                message="Memory fault type: not present in log",
                lines=[]
            ))


SONY_SALT = bytes.fromhex("518D64A635DED8C1E6B039B1C3E55230")


def compute_sony_nid(name: str) -> Tuple[int, int]:
    """
    Computes (le_u64, be_u64) Sony NID using SHA-1(name + SONY_SALT).
    """
    h = hashlib.sha1(name.encode("utf-8") + SONY_SALT).digest()
    le = struct.unpack("<Q", h[:8])[0]
    be = struct.unpack(">Q", h[:8])[0]
    return le, be


# Startup self-check: scePthreadCondInit must hash to 0xd936fddaaba9ae5d (little-endian)
_self_check_le, _ = compute_sony_nid("scePthreadCondInit")
RESOLVER_BROKEN = (_self_check_le != 0xd936fddaaba9ae5d)


COMMON_CANDIDATE_NAMES = [
    # libc / runtime
    "malloc", "free", "realloc", "calloc", "memset", "memcpy", "memmove",
    "strcmp", "strncmp", "strlen", "strcpy", "strncpy", "strcat", "strncat",
    "printf", "sprintf", "snprintf", "puts", "putchar",
    "fopen", "fclose", "fread", "fwrite", "fseek", "ftell", "abort", "exit",
    "qsort", "bsearch", "sin", "cos", "tan", "sqrt", "pow", "floor", "ceil",
    "__stack_chk_fail", "__cxa_pure_virtual", "__cxa_atexit", "gettimeofday", "clock_gettime",
    # pthread
    "pthread_create", "pthread_join", "pthread_detach", "pthread_exit", "pthread_self",
    "pthread_mutex_init", "pthread_mutex_destroy", "pthread_mutex_lock", "pthread_mutex_trylock", "pthread_mutex_unlock",
    "pthread_cond_init", "pthread_cond_destroy", "pthread_cond_wait", "pthread_cond_timedwait", "pthread_cond_signal", "pthread_cond_broadcast",
    "pthread_rwlock_init", "pthread_rwlock_rdlock", "pthread_rwlock_wrlock", "pthread_rwlock_unlock",
    "pthread_key_create", "pthread_setspecific", "pthread_getspecific",
    # sceKernel / orbis
    "sceKernelExitProcess", "sceKernelLoadStartModule", "sceKernelAllocateDirectMemory",
    "sceKernelMapDirectMemory", "sceKernelReleaseDirectMemory", "sceKernelUsleep", "sceKernelNanosleep",
    "sceKernelVirtualQuery", "sceKernelCreateSema", "sceKernelWaitSema", "sceKernelPollSema",
    "sceKernelSignalSema", "sceKernelDeleteSema", "sceKernelCreateEqueue", "sceKernelDeleteEqueue",
    "sceKernelAddUserEvent", "sceKernelDeleteUserEvent", "sceKernelGetEvent",
    "sceKernelCreateEventFlag", "sceKernelSetEventFlag", "sceKernelWaitEventFlag",
    "sceKernelPollEventFlag", "sceKernelDeleteEventFlag",
    "scePthreadCreate", "scePthreadJoin", "scePthreadDetach", "scePthreadExit", "scePthreadSelf",
    "scePthreadMutexInit", "scePthreadMutexDestroy", "scePthreadMutexLock", "scePthreadMutexTrylock", "scePthreadMutexUnlock",
    "scePthreadCondInit", "scePthreadCondDestroy", "scePthreadCondWait", "scePthreadCondTimedwait",
    "scePthreadCondSignal", "scePthreadCondBroadcast",
    "sceKernelGetProcessTime", "sceKernelGetCurrentCpu", "sceKernelIccSetBuzzer", "sceKernelGetIdTable",
    # sceGnm / video
    "sceGnmSubmitAndFlipCommandBuffers", "sceGnmSubmitCommandBuffers", "sceGnmAreCommandBuffersSubmitted",
    "sceGnmInsertPushMarker", "sceGnmInsertPopMarker", "sceGnmDrawIndex", "sceGnmDrawIndexAuto",
    "sceGnmDispatchDirect", "sceGnmDispatchIndirectOnCe",
    "sceVideoOutOpen", "sceVideoOutClose", "sceVideoOutSetBufferAttribute2", "sceVideoOutRegisterBuffers2", "sceVideoOutGetResolutionStatus",
    # mono
    "mono_jit_init", "mono_jit_cleanup", "mono_domain_get", "mono_assembly_open", "mono_assembly_get_image",
    "mono_image_get_table_rows", "mono_class_from_name", "mono_class_get_method_from_name",
    "mono_runtime_invoke", "mono_object_new", "mono_string_new", "mono_string_to_utf8", "mono_thread_attach", "mono_thread_detach",
    # audio / user / sysmodule / fios
    "sceAudioOutOpen", "sceAudioOutClose", "sceAudioOutOutput", "sceAudioOutSetVolume",
    "scePadInit", "scePadOpen", "scePadClose", "scePadReadState", "scePadSetLightBar",
    "sceUserServiceInitialize", "sceUserServiceGetInitialUser", "sceUserServiceGetLoginIdList", "sceUserServiceGetUserName",
    "sceSysmoduleLoadModule", "sceSysmoduleUnloadModule", "sceSysmoduleIsLoaded",
    "sceFiosInitialize", "sceFiosTerminate",
    "sceNetInit", "sceNetTerm", "sceNetSocket", "sceNetConnect", "sceNetSend", "sceNetRecv", "sceNetClose",
]


class NidResolver:
    """
    STAGE 3: NID Resolver
    Resolves 64-bit Sony NIDs using:
    (a) salted SHA1: sha1(name + SONY_SALT), checking big-endian and little-endian
    (b) repo registry: src/hle/nid_registry.cpp
    (c) built-in common names list
    (d) user-supplied names file
    With prefix/suffix variant generation (sceKernel/scePthread vs pthread_, with/without '_').
    Never invents names. Reports MATCH name (source) or UNIDENTIFIED (tried N candidates).
    """

    def __init__(self, repo_dir: Optional[str] = None, names_file: Optional[str] = None):
        self.repo_dir = repo_dir
        self.names_file = names_file
        self.direct_nid_map: Dict[int, Tuple[str, str]] = {}  # nid -> (name, source)
        self.candidate_sources: Dict[str, str] = {}  # name -> source
        self._load_candidates()

    def _load_candidates(self) -> None:
        # 1. Built-in list
        for name in COMMON_CANDIDATE_NAMES:
            self.candidate_sources[name] = "builtin"

        # 2. Repo registry
        if self.repo_dir:
            reg_path = os.path.join(self.repo_dir, "src", "hle", "nid_registry.cpp")
            if os.path.isfile(reg_path):
                self._load_from_repo_registry(reg_path)
        else:
            # Try to auto-locate src/hle/nid_registry.cpp
            cur = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(cur, "..", "src", "hle", "nid_registry.cpp")
            if os.path.isfile(cand):
                self._load_from_repo_registry(cand)

        # 3. User names file
        if self.names_file and os.path.isfile(self.names_file):
            try:
                with open(self.names_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            self.candidate_sources[line] = "names_file"
            except Exception:
                pass

    def _load_from_repo_registry(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                # Patterns:
                # register_symbol(0x6424D00705E3BEBEULL, "sceKernelExitProcess", "libkernel", ...)
                # {0x6424D00705E3BEBEULL, "sceKernelExitProcess", "libkernel"}
                # // NID: 0x3AEDE22F569BBE78ULL ("8Op3OmHl3zY")
                p1 = re.findall(r"register_symbol\s*\(\s*0x([0-9a-fA-F]+)ULL\s*,\s*\"([^\"]+)\"", content)
                for nid_str, name in p1:
                    nid = int(nid_str, 16)
                    self.direct_nid_map[nid] = (name, "repo_registry")
                    self.candidate_sources[name] = "repo_registry"

                p2 = re.findall(r"\{\s*0x([0-9a-fA-F]+)ULL\s*,\s*\"([^\"]+)\"", content)
                for nid_str, name in p2:
                    nid = int(nid_str, 16)
                    self.direct_nid_map[nid] = (name, "repo_registry")
                    self.candidate_sources[name] = "repo_registry"

                p3 = re.findall(r"//\s*NID:\s*0x([0-9a-fA-F]+)ULL", content)
                for nid_str in p3:
                    # NID comments might not always have name on same line, keep if parsed
                    pass
        except Exception:
            pass

    def _generate_variants(self, base_name: str) -> Set[str]:
        vars_set = {base_name}
        # with and without leading underscore
        if base_name.startswith("_"):
            vars_set.add(base_name.lstrip("_"))
        else:
            vars_set.add(f"_{base_name}")

        # scePthread vs pthread_
        if base_name.startswith("scePthread"):
            suffix = base_name[len("scePthread"):]
            if suffix:
                pthread_name = f"pthread_{suffix[0].lower()}{suffix[1:]}"
                vars_set.add(pthread_name)
                vars_set.add(f"_{pthread_name}")
        elif base_name.startswith("pthread_"):
            suffix = base_name[len("pthread_"):]
            if suffix:
                sce_name = f"scePthread{suffix[0].upper()}{suffix[1:]}"
                vars_set.add(sce_name)
                vars_set.add(f"_{sce_name}")

        # sceKernel vs libc / base
        if base_name.startswith("sceKernel"):
            k_suffix = base_name[len("sceKernel"):]
            if k_suffix:
                c_name = f"{k_suffix[0].lower()}{k_suffix[1:]}"
                vars_set.add(c_name)
                vars_set.add(f"_{c_name}")

        return vars_set

    def resolve(self, nid: int) -> Tuple[str, str, Optional[str]]:
        """
        Resolves a 64-bit NID.
        Returns (status: "MATCH"|"UNIDENTIFIED", details: str, name_if_matched: Optional[str]).
        """
        if RESOLVER_BROKEN:
            return "UNIDENTIFIED", "resolver broken (self-check failed: scePthreadCondInit does not hash to 0xd936fddaaba9ae5d)", None

        # 1. Direct registry lookup
        if nid in self.direct_nid_map:
            name, source = self.direct_nid_map[nid]
            return "MATCH", f"{name} (source: {source})", name

        # 2. Hash search across candidates and variants
        candidates_tested = 0
        for cand_name, source in list(self.candidate_sources.items()):
            variants = self._generate_variants(cand_name)
            for v in variants:
                candidates_tested += 1
                le, be = compute_sony_nid(v)
                if le == nid:
                    return "MATCH", f"{v} (source: {source}, salted_sha1_le)", v
                if be == nid:
                    return "MATCH", f"{v} (source: {source}, salted_sha1_be)", v

        return "UNIDENTIFIED", f"UNIDENTIFIED (tried {candidates_tested} candidates)", None


@dataclass
class FallbackStubRecord:
    nid: int
    count: int = 0
    first_hit_line: Optional[int] = None
    first_hit_caller_rip: Optional[int] = None
    first_hit_args: Dict[str, str] = field(default_factory=dict)


class HleAnalyzer:
    """
    STAGE 3: HleAnalyzer
    Analyzes HLE calls:
    - Per-name call counts and class (registered vs generic fallback vs unresolved)
    - Top 40 by count
    - Every fallback NID with hit count, first-hit caller RIP, and first-hit args (rdi, rsi, rdx, rcx, r8, r9)
    - Last 30 HLE calls before run ended
    - Inferred final state: guest code running vs waiting in sleep/mutex/cond/sema
    - Benign stub traps allowlist (triage_allow.toml)
    """

    def __init__(self, log: Log, resolver: NidResolver, allowlist_file: Optional[str] = None):
        self.log = log
        self.resolver = resolver
        self.allowlist_file = allowlist_file
        self.benign_allowlist: Dict[str, str] = {}  # hex_nid or name -> reason

        self.call_counts: Dict[str, int] = {}
        self.call_classes: Dict[str, str] = {}  # name/nid -> "registered" | "generic fallback" | "unresolved"
        self.fallback_stubs: Dict[int, FallbackStubRecord] = {}
        self.all_hle_events: List[Tuple[int, str]] = []  # (line_no, name/call_desc)

        self.known_benign_findings: List[Tuple[int, str, str]] = []  # (line_no, stub, reason)
        self.findings: List[Finding] = []

        self._load_allowlist()
        self._analyze()

    def _load_allowlist(self) -> None:
        path = self.allowlist_file or "triage_allow.toml"
        if not os.path.isfile(path):
            cur = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(cur, "..", "triage_allow.toml")
            if os.path.isfile(cand):
                path = cand

        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    # Simple stdlib TOML key-value extraction
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            key = k.strip().strip('"').strip("'").lower()
                            val = v.strip().strip('"').strip("'")
                            self.benign_allowlist[key] = val
            except Exception:
                pass

    def _analyze(self) -> None:
        # Regexes for HLE lines
        # Unimplemented stub:
        re_unimpl = re.compile(
            r"Unimplemented stub called:\s+NID=0x([0-9a-fA-F]+)(?:.*caller RIP=0x([0-9a-fA-F]+))?",
            re.IGNORECASE
        )
        # Named HLE call lines:
        # [HLE] sceKernelLoadStartModule('/app0/...') -> handle 0x2001
        # [HLE CALL] foo
        # [HLE WHITELIST] Benign stub executed: NID=0x...
        # [FIOS2] sceFiosInitialize() -> SCE_OK
        re_named_hle = re.compile(
            r"\[(?:HLE|HLE CALL|HLE WHITELIST|FIOS2)\]\s*(?:Benign stub executed:\s+NID=0x([0-9a-fA-F]+)|([A-Za-z0-9_]+))",
            re.IGNORECASE
        )

        re_arg = re.compile(r"(rdi|rsi|rdx|rcx|r8|r9)=(0x[0-9a-fA-F]+|'[^\']*')", re.IGNORECASE)

        for line_no, line in self.log.hle_lines:
            # Check unimplemented stub
            m_unimpl = re_unimpl.search(line)
            if m_unimpl:
                nid = int(m_unimpl.group(1), 16)
                caller_rip = int(m_unimpl.group(2), 16) if m_unimpl.group(2) else None
                nid_key = f"NID:0x{nid:016X}"

                if nid not in self.fallback_stubs:
                    rec = FallbackStubRecord(nid=nid, count=1, first_hit_line=line_no, first_hit_caller_rip=caller_rip)
                    # Extract args
                    for m_a in re_arg.finditer(line):
                        reg_n = m_a.group(1).lower()
                        reg_v = m_a.group(2)
                        rec.first_hit_args[reg_n] = reg_v
                    self.fallback_stubs[nid] = rec
                else:
                    self.fallback_stubs[nid].count += 1

                self.call_counts[nid_key] = self.call_counts.get(nid_key, 0) + 1
                self.call_classes[nid_key] = "generic fallback"
                self.all_hle_events.append((line_no, nid_key))

                # Check allowlist
                hex_nid = f"0x{nid:016x}".lower()
                hex_nid_short = f"0x{nid:x}".lower()
                if hex_nid in self.benign_allowlist:
                    self.known_benign_findings.append((line_no, nid_key, self.benign_allowlist[hex_nid]))
                elif hex_nid_short in self.benign_allowlist:
                    self.known_benign_findings.append((line_no, nid_key, self.benign_allowlist[hex_nid_short]))
                continue

            # Check named HLE lines
            m_named = re_named_hle.search(line)
            if m_named:
                if m_named.group(1):
                    # Benign stub with NID
                    w_nid = int(m_named.group(1), 16)
                    w_key = f"NID:0x{w_nid:016X}"
                    self.call_counts[w_key] = self.call_counts.get(w_key, 0) + 1
                    self.call_classes[w_key] = "registered"
                    self.all_hle_events.append((line_no, w_key))

                    hex_w = f"0x{w_nid:016x}".lower()
                    if hex_w in self.benign_allowlist:
                        self.known_benign_findings.append((line_no, w_key, self.benign_allowlist[hex_w]))
                elif m_named.group(2):
                    name = m_named.group(2)
                    self.call_counts[name] = self.call_counts.get(name, 0) + 1
                    self.call_classes[name] = "registered"
                    self.all_hle_events.append((line_no, name))
                    if name.lower() in self.benign_allowlist:
                        self.known_benign_findings.append((line_no, name, self.benign_allowlist[name.lower()]))

        self._evaluate()

    def _evaluate(self) -> None:
        # Top 40 by count
        sorted_calls = sorted(self.call_counts.items(), key=lambda x: x[1], reverse=True)
        top_40 = sorted_calls[:40]
        if top_40:
            top_strs = []
            for name, count in top_40:
                cls = self.call_classes.get(name, "registered")
                top_strs.append(f"{name} ({cls}): {count}")
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Top HLE calls (by count): {'; '.join(top_strs[:10])} ... ({len(top_40)} shown)",
                lines=[ev[0] for ev in self.all_hle_events[:40]]
            ))

        # Every fallback NID with details
        if self.fallback_stubs:
            for nid, rec in sorted(self.fallback_stubs.items(), key=lambda x: x[1].count, reverse=True):
                status, details, resolved_name = self.resolver.resolve(nid)
                args_str = " ".join(f"{k}={v}" for k, v in rec.first_hit_args.items()) or "none"
                rip_str = f"0x{rec.first_hit_caller_rip:X}" if rec.first_hit_caller_rip is not None else "unknown"
                msg = (
                    f"Fallback stub NID 0x{nid:016X}: hits={rec.count}, first caller RIP={rip_str}, "
                    f"args=({args_str}) -> {status} {details}"
                )
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=msg,
                    lines=[rec.first_hit_line] if rec.first_hit_line else []
                ))

        # Known benign
        if self.known_benign_findings:
            benign_lines = [b[0] for b in self.known_benign_findings]
            benign_descs = [f"{b[1]} ({b[2]})" for b in self.known_benign_findings[:10]]
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Known benign stub executions: {'; '.join(benign_descs)}",
                lines=benign_lines
            ))

        # Last 30 HLE calls
        last_30 = self.all_hle_events[-30:] if self.all_hle_events else []
        if last_30:
            last_names = [ev[1] for ev in last_30]
            last_lines = [ev[0] for ev in last_30]

            # Inferred final state:
            # Does it look like guest code running vs waiting in a sleep/mutex/cond/sema?
            waiting_keywords = [
                "sleep", "wait", "sema", "cond", "mutex", "lock", "usleep",
                "waitsync", "waitingsync", "waitingsleep"
            ]
            last_call = last_names[-1].lower() if last_names else ""
            is_waiting = any(kw in last_call for kw in waiting_keywords)

            if is_waiting:
                self.findings.append(Finding(
                    kind="INFERRED",
                    message="Final execution state looks like waiting in sleep/mutex/cond/sema call",
                    rule="last HLE call was a synchronization or sleep primitive",
                    lines=last_lines[-1:]
                ))
            else:
                self.findings.append(Finding(
                    kind="INFERRED",
                    message="Final execution state looks like guest code actively running",
                    rule="last HLE call was not a waiting or synchronization primitive",
                    lines=last_lines[-1:]
                ))


@dataclass
class MappingEntry:
    owner: str
    start: int
    end: int
    size: int
    line: int


@dataclass
class ModuleSpan:
    path: str
    base: int
    span: int
    line: int


@dataclass
class ModuleExportStats:
    path: str
    base: int
    span: int
    in_table: int
    defined: int
    undef_skipped: int
    no_nid_skipped: int
    dup_not_recorded: int
    recorded: int
    line: int


@dataclass
class SegmentEntry:
    idx: int
    type_name: str
    offset: int
    vaddr: int
    filesz: int
    memsz: int
    flags: str
    line: int


@dataclass
class WatchRecord:
    idx: int
    rip: int
    addr: int
    size: int
    old_val: int
    new_val: int
    line: int


class LoaderAnalyzer:
    """
    STAGE 4: LoaderAnalyzer
    - Parses mapping table lines and detects overlaps (both ranges and both owners)
    - Parses module base/span lines and checks base[i]+span[i] <= base[i+1]
    - Parses per-module counters (in_table, defined, undef_skipped, no_nid_skipped, dup_not_recorded)
      and flags no_nid_skipped > 0 as a possible hole (INFERRED)
    - Imports per module by class (resolved to export, registered HLE, fallback, unresolved);
      if absent, explicitly reports 'not present in log' and clarifies '0 unread' is not resolution success
    - Segment info: flags faulting or hot address that is in BSS (memsz > filesz region)
    - Detects write-watch records and timeline; flags watched value changes with no recorded write (INFERRED)
    """

    def __init__(self, log: Log, fault_rip: Optional[int] = None):
        self.log = log
        self.fault_rip = fault_rip

        self.mappings: List[MappingEntry] = []
        self.modules: List[ModuleSpan] = []
        self.export_stats: List[ModuleExportStats] = []
        self.segments: List[SegmentEntry] = []
        self.watch_records: List[WatchRecord] = []

        self.overlap_pairs: List[Tuple[MappingEntry, MappingEntry]] = []
        self.layout_violations: List[Tuple[ModuleSpan, ModuleSpan]] = []
        self.import_counts: Optional[Dict[str, int]] = None

        self.findings: List[Finding] = []
        self._analyze()

    def _analyze(self) -> None:
        self._parse_loader_and_map_lines()
        self._parse_watch_lines()
        self._parse_segments()
        self._evaluate()

    def _parse_loader_and_map_lines(self) -> None:
        # 1. Mapping table lines
        # Format:
        # [map] registered guest mappings (%zu):
        #   owner start end size
        #   %-34s 0x%09llx 0x%09llx 0x%08llx
        # or [map] OVERLAP: cannot map 'A' [0x...,0x...) because it intersects 'B' [0x...,0x...)
        re_map_line = re.compile(r"^([^\s].*?)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)")
        re_map_overlap = re.compile(
            r"\[map\]\s+OVERLAP:\s+cannot map '([^']+)'\s+\[0x([0-9a-fA-F]+),\s*0x([0-9a-fA-F]+)\)\s+because it intersects\s+'([^']+)'\s+\[0x([0-9a-fA-F]+),\s*0x([0-9a-fA-F]+)\)",
            re.IGNORECASE
        )

        # 2. Module base/span lines:
        # [module] 0x%x %s: base=0x%llx span=0x%llx (ends 0x%llx)
        re_module_line = re.compile(
            r"\[module\]\s+0x[0-9a-fA-F]+\s+([^:]+):\s+base=0x([0-9a-fA-F]+)\s+span=0x([0-9a-fA-F]+)",
            re.IGNORECASE
        )

        # 3. Export counters:
        # [exports]   %-24s base=0x%llx span=0x%llx | in_table=%zu defined=%zu undef_skipped=%zu no_nid_skipped=%zu dup_not_recorded=%zu | recorded=%zu
        re_export_line = re.compile(
            r"\[exports\]\s+([^\s]+)\s+base=0x([0-9a-fA-F]+)\s+span=0x([0-9a-fA-F]+)\s+\|\s+in_table=(\d+)\s+defined=(\d+)\s+undef_skipped=(\d+)\s+no_nid_skipped=(\d+)\s+dup_not_recorded=(\d+)\s+\|\s+recorded=(\d+)",
            re.IGNORECASE
        )

        # 4. Imports classification counters:
        re_import_class = re.compile(r"\[(?:imports|module)\]\s+(?:imports by class:)?\s*resolved_to_export=(\d+)\s+registered_hle=(\d+)\s+fallback=(\d+)\s+unresolved=(\d+)", re.IGNORECASE)

        in_mapping_table = False
        for line_no, line in self.log.loader_lines:
            # Overlap explicit report
            m_ov = re_map_overlap.search(line)
            if m_ov:
                owner1, s1, e1 = m_ov.group(1), int(m_ov.group(2), 16), int(m_ov.group(3), 16)
                owner2, s2, e2 = m_ov.group(4), int(m_ov.group(5), 16), int(m_ov.group(6), 16)
                self.overlap_pairs.append((
                    MappingEntry(owner1, s1, e1, e1 - s1, line_no),
                    MappingEntry(owner2, s2, e2, e2 - s2, line_no)
                ))
                continue

            if "[map] registered guest mappings" in line:
                in_mapping_table = True
                continue

            if in_mapping_table:
                if line.startswith("---") or line.startswith("owner") or not line:
                    continue
                m_m = re_map_line.match(line)
                if m_m:
                    owner = m_m.group(1).strip()
                    start = int(m_m.group(2), 16)
                    end = int(m_m.group(3), 16)
                    sz = int(m_m.group(4), 16)
                    self.mappings.append(MappingEntry(owner, start, end, sz, line_no))
                    continue
                elif line.startswith("["):
                    in_mapping_table = False

            # Module base/span
            m_mod = re_module_line.search(line)
            if m_mod:
                path = m_mod.group(1).strip()
                base = int(m_mod.group(2), 16)
                span = int(m_mod.group(3), 16)
                self.modules.append(ModuleSpan(path, base, span, line_no))

            # Exports counters
            m_exp = re_export_line.search(line)
            if m_exp:
                path = m_exp.group(1).strip()
                base = int(m_exp.group(2), 16)
                span = int(m_exp.group(3), 16)
                in_table = int(m_exp.group(4))
                defined = int(m_exp.group(5))
                undef = int(m_exp.group(6))
                no_nid = int(m_exp.group(7))
                dup = int(m_exp.group(8))
                recorded = int(m_exp.group(9))
                self.export_stats.append(ModuleExportStats(
                    path, base, span, in_table, defined, undef, no_nid, dup, recorded, line_no
                ))
                # Also record module span if not present
                if not any(m.path == path for m in self.modules):
                    self.modules.append(ModuleSpan(path, base, span, line_no))

            # Import class
            m_imp = re_import_class.search(line)
            if m_imp:
                self.import_counts = {
                    "resolved_to_export": int(m_imp.group(1)),
                    "registered_hle": int(m_imp.group(2)),
                    "fallback": int(m_imp.group(3)),
                    "unresolved": int(m_imp.group(4)),
                }

    def _parse_watch_lines(self) -> None:
        # [watch] #%-3zu rip=%s (0x%llx) rsp=... -> 0x%llx (%zu B) ... old=0x%016llx new=0x%016llx...
        re_watch = re.compile(
            r"\[watch\]\s+#(\d+)\s+rip=.*?\s*\(0x([0-9a-fA-F]+)\).*?->\s*0x([0-9a-fA-F]+)\s+\((\d+)\s*B\).*?old=0x([0-9a-fA-F]+)\s+new=0x([0-9a-fA-F]+)",
            re.IGNORECASE
        )
        for line_no, line in self.log.watch_lines:
            m = re_watch.search(line)
            if m:
                idx = int(m.group(1))
                rip = int(m.group(2), 16)
                addr = int(m.group(3), 16)
                sz = int(m.group(4))
                old_val = int(m.group(5), 16)
                new_val = int(m.group(6), 16)
                self.watch_records.append(WatchRecord(idx, rip, addr, sz, old_val, new_val, line_no))

    def _parse_segments(self) -> None:
        # Format:
        # [Idx] Type Offset VirtAddr FileSz MemSz Flags Align
        # [0 ] PT_LOAD 0x4000 0x0 0x14b62c4 0x14b62c4 R-E 0x4000
        re_seg = re.compile(
            r"^\[(\d+)\s*\]\s+(PT_[A-Za-z0-9_]+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+([RWE\s\-]{3})",
            re.IGNORECASE
        )
        for line_no, line in self.log.other_recognized_lines + self.log.loader_lines:
            m = re_seg.search(line)
            if m:
                idx = int(m.group(1))
                t_name = m.group(2).strip()
                offset = int(m.group(3), 16)
                vaddr = int(m.group(4), 16)
                filesz = int(m.group(5), 16)
                memsz = int(m.group(6), 16)
                flags = m.group(7).strip()
                self.segments.append(SegmentEntry(idx, t_name, offset, vaddr, filesz, memsz, flags, line_no))

    def _check_mapping_overlaps(self) -> None:
        # Check computed overlaps from parsed mapping table
        n = len(self.mappings)
        for i in range(n):
            for j in range(i + 1, n):
                m1 = self.mappings[i]
                m2 = self.mappings[j]
                # Intersection: start1 < end2 and start2 < end1
                if m1.start < m2.end and m2.start < m1.end:
                    # Avoid duplicate if already reported
                    if not any((p[0].owner == m1.owner and p[1].owner == m2.owner) for p in self.overlap_pairs):
                        self.overlap_pairs.append((m1, m2))

    def _check_module_layout(self) -> None:
        # Sort modules by base address
        sorted_mods = sorted(self.modules, key=lambda m: m.base)
        for i in range(len(sorted_mods) - 1):
            curr_mod = sorted_mods[i]
            next_mod = sorted_mods[i + 1]
            if curr_mod.base + curr_mod.span > next_mod.base:
                self.layout_violations.append((curr_mod, next_mod))

    def _evaluate(self) -> None:
        self._check_mapping_overlaps()
        self._check_module_layout()

        # 1. Overlaps
        if self.overlap_pairs:
            for m1, m2 in self.overlap_pairs:
                lines = [l for l in (m1.line, m2.line) if l]
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=(
                        f"Mapping overlap detected: '{m1.owner}' [0x{m1.start:09X}, 0x{m1.end:09X}) "
                        f"intersects '{m2.owner}' [0x{m2.start:09X}, 0x{m2.end:09X})"
                    ),
                    lines=lines
                ))

        # 2. Module layout overlaps
        if self.layout_violations:
            for m1, m2 in self.layout_violations:
                lines = [l for l in (m1.line, m2.line) if l]
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=(
                        f"Module layout overlap: {m1.path} [0x{m1.base:X}, 0x{m1.base + m1.span:X}) "
                        f"extends past {m2.path} base 0x{m2.base:X}"
                    ),
                    lines=lines
                ))

        # 3. Export counters & no_nid_skipped
        if self.export_stats:
            for s in self.export_stats:
                line_info = [s.line] if s.line else []
                stat_msg = (
                    f"Module {s.path} loader counters: base=0x{s.base:X} span=0x{s.span:X} "
                    f"in_table={s.in_table} defined={s.defined} undef_skipped={s.undef_skipped} "
                    f"no_nid_skipped={s.no_nid_skipped} dup_not_recorded={s.dup_not_recorded} recorded={s.recorded}"
                )
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=stat_msg,
                    lines=line_info
                ))

                if s.no_nid_skipped > 0:
                    self.findings.append(Finding(
                        kind="INFERRED",
                        message=f"Module {s.path} has no_nid_skipped={s.no_nid_skipped} > 0, possible symbol resolution hole",
                        rule="loader skipped exported symbols with no valid NID",
                        lines=line_info
                    ))

        # 4. Imports classification
        if self.import_counts is not None:
            counts_str = ", ".join(f"{k}={v}" for k, v in self.import_counts.items())
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Imports by class: {counts_str}",
                lines=[]
            ))
        else:
            self.findings.append(Finding(
                kind="OBSERVED",
                message="Imports per module by class: not present in log (note: '0 unread' relocation lines do not indicate resolution success)",
                lines=[]
            ))

        # 5. Segment info (BSS check)
        if self.fault_rip is not None:
            for seg in self.segments:
                if seg.memsz > seg.filesz:
                    bss_start = seg.vaddr + seg.filesz
                    bss_end = seg.vaddr + seg.memsz
                    if bss_start <= self.fault_rip < bss_end:
                        self.findings.append(Finding(
                            kind="OBSERVED",
                            message=(
                                f"Faulting address RIP=0x{self.fault_rip:X} is in BSS region "
                                f"[0x{bss_start:X}, 0x{bss_end:X}) (segment #{seg.idx} {seg.type_name} "
                                f"vaddr=0x{seg.vaddr:X} filesz=0x{seg.filesz:X} memsz=0x{seg.memsz:X})"
                            ),
                            lines=[seg.line] if seg.line else []
                        ))

        # 6. Write-watch records timeline and anomaly detection
        if self.watch_records:
            self.watch_records.sort(key=lambda r: r.idx)
            lines_all = [r.line for r in self.watch_records]
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Write-watch timeline: {len(self.watch_records)} records captured",
                lines=lines_all
            ))

            # Group by watched address to detect unrecorded modifications
            by_addr: Dict[int, List[WatchRecord]] = {}
            for r in self.watch_records:
                by_addr.setdefault(r.addr, []).append(r)

            for addr, recs in by_addr.items():
                for i in range(len(recs) - 1):
                    r_curr = recs[i]
                    r_next = recs[i + 1]
                    if r_curr.new_val != r_next.old_val:
                        self.findings.append(Finding(
                            kind="INFERRED",
                            message=(
                                f"Watched address 0x{addr:X} changed from 0x{r_curr.new_val:016X} to 0x{r_next.old_val:016X} "
                                f"between watch #{r_curr.idx} and #{r_next.idx} with no recorded write"
                            ),
                            rule="watchpoint observed old value differing from previously recorded new value (watch may miss a store path)",
                            lines=[r_curr.line, r_next.line]
                        ))


# Canonical GFX7/CIK PM4 Type-3 Opcodes Table in ONE module-level dict
PM4_OPCODE_TABLE: Dict[int, str] = {
    0x10: "NOP (verify against SDK)",
    0x26: "INDEX_BASE (verify against SDK)",
    0x27: "DRAW_INDEX_2 (verify against SDK)",
    0x28: "CONTEXT_CONTROL (verify against SDK)",
    0x2A: "INDEX_TYPE (verify against SDK)",
    0x2D: "DRAW_INDEX_AUTO (verify against SDK)",
    0x2F: "NUM_INSTANCES (verify against SDK)",
    0x3C: "WAIT_REG_MEM (verify against SDK)",
    0x46: "EVENT_WRITE (verify against SDK)",
    0x47: "EVENT_WRITE_EOP (verify against SDK)",
    0x49: "RELEASE_MEM (verify against SDK)",
    0x50: "DMA_DATA (verify against SDK)",
    0x58: "ACQUIRE_MEM (verify against SDK)",
    0x68: "SET_CONFIG_REG (verify against SDK)",
    0x69: "SET_CONTEXT_REG (verify against SDK)",
    0x76: "SET_SH_REG (verify against SDK)",
    0x79: "SET_UCONFIG_REG (verify against SDK)",
}


def get_pm4_opcode_name(op: int) -> str:
    if op in PM4_OPCODE_TABLE:
        return PM4_OPCODE_TABLE[op]
    return f"0x{op:02X} UNKNOWN"


def decode_pm4_header(hdr: int) -> Tuple[int, int, int, int]:
    """
    Decodes a PM4 packet header DWORD.
    Returns (packet_type, count, opcode, total_packet_dwords).
    """
    pkt_type = (hdr >> 30) & 3
    count = (hdr >> 16) & 0x3FFF
    opcode = (hdr >> 8) & 0xFF

    if pkt_type == 3:
        # Type 3: count+1 body DWORDs -> count+2 total DWORDs
        total_len = count + 2
    elif pkt_type == 0:
        # Type 0: count+1 body DWORDs -> count+2 total DWORDs
        total_len = count + 2
    elif pkt_type == 2:
        # Type 2: no-op filler, 1 DWORD
        total_len = 1
    else:
        # Type 1: usually 2 or 3 DWORDs
        total_len = 3

    return pkt_type, count, opcode, total_len


@dataclass
class PacketWalkResult:
    dcb_addr: int
    total_dwords: int
    type_counts: Dict[int, int]
    packets: List[Tuple[int, int, int, int]]  # (start_idx, pkt_type, opcode, length)
    packet_boundaries: Set[int]
    first_type0_or_2: Optional[Tuple[int, int]]  # (type, index)
    zero_runs_unpadded: List[Tuple[int, int]]  # (start, length)
    trailing_zero_run: Optional[Tuple[int, int]]  # (start, length)
    draw_count_27: int
    draw_count_2d: int
    set_sh_reg_count: int
    set_context_reg_count: int
    nop_ranges: List[Tuple[int, int]]


class GpuPm4Analyzer:
    """
    STAGE 5: GpuPm4Analyzer and DCB walker
    - Opcode table in ONE module-level dict, GFX7/CIK names, each entry marked 'verify against SDK'
    - Labels mismatch detection between log names and table names
    - Packet walk of each DCB dump: header type, count, opcode, total dwords
    - Reports packets per type, first Type-0 or Type-2, runs of >=2 zero dwords not inside NOP bodies, and trailing zeroes
    - Slice matching for sceGnm* calls (rdi=start, rsi=count): all-zero, padded NOP, or real packets; flags rsi>65536 or rdi outside
    - Draw summary per DCB: 0x27 and 0x2D draws, SET_SH_REG, SET_CONTEXT_REG; flags 'no draw packets' and shader slots NOP/zero
    - Flip and wait timeouts: reports lines with flip/wait/timeout/stall with line numbers and frame number
    """

    def __init__(self, log: Log):
        self.log = log
        self.dcb_results: List[PacketWalkResult] = []
        self.label_mismatches: List[Tuple[int, int, str, str]] = []  # (line, op, log_label, table_label)
        self.timeout_lines: List[Tuple[int, int, str]] = []  # (line, frame_idx, text)
        self.findings: List[Finding] = []

        self._analyze()

    def _walk_dcb(self, dcb: DcbDump) -> PacketWalkResult:
        dwords = dcb.dwords
        n = len(dwords)
        type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        packets: List[Tuple[int, int, int, int]] = []
        packet_boundaries: Set[int] = {0, n}
        first_t0_or_t2: Optional[Tuple[int, int]] = None
        nop_ranges: List[Tuple[int, int]] = []

        draw_27 = 0
        draw_2d = 0
        set_sh = 0
        set_ctx = 0

        i = 0
        while i < n:
            hdr = dwords[i]
            # Check trailing zeroes
            if hdr == 0:
                # Check how many consecutive zeroes remain
                z_len = 0
                while i + z_len < n and dwords[i + z_len] == 0:
                    z_len += 1
                if i + z_len == n:
                    # Trailing zero run
                    break
                else:
                    # Non-trailing zeroes encountered outside packet header
                    i += 1
                    continue

            pkt_type, count, opcode, total_len = decode_pm4_header(hdr)
            type_counts[pkt_type] = type_counts.get(pkt_type, 0) + 1

            if pkt_type in (0, 2) and first_t0_or_t2 is None:
                first_t0_or_t2 = (pkt_type, i)

            if pkt_type == 3:
                if opcode == 0x10:
                    nop_ranges.append((i, i + total_len))
                elif opcode == 0x27:
                    draw_27 += 1
                elif opcode == 0x2D:
                    draw_2d += 1
                elif opcode == 0x76:
                    set_sh += 1
                elif opcode == 0x69:
                    set_ctx += 1

            packets.append((i, pkt_type, opcode, total_len))
            packet_boundaries.add(i)
            packet_boundaries.add(i + total_len)

            if total_len <= 0:
                i += 1
            else:
                i += total_len

        # Detect unpadded zero runs and trailing zeroes
        max_packet_end = max((s + l for s, _, _, l in packets), default=0)
        trailing_zero_run: Optional[Tuple[int, int]] = None
        if max_packet_end < n and all(dwords[z] == 0 for z in range(max_packet_end, n)):
            trailing_zero_run = (max_packet_end, n - max_packet_end)

        zero_runs_unpadded: List[Tuple[int, int]] = []

        def is_in_nop_body(idx: int) -> bool:
            return any(s < idx < e for s, e in nop_ranges)

        k = 0
        limit = max_packet_end if trailing_zero_run else n
        while k < limit:
            if dwords[k] == 0:
                run_start = k
                while k < limit and dwords[k] == 0:
                    k += 1
                run_len = k - run_start
                if run_len >= 2:
                    if not all(is_in_nop_body(idx) for idx in range(run_start, k)):
                        zero_runs_unpadded.append((run_start, run_len))
            else:
                k += 1

        return PacketWalkResult(
            dcb_addr=dcb.addr,
            total_dwords=n,
            type_counts=type_counts,
            packets=packets,
            packet_boundaries=packet_boundaries,
            first_type0_or_2=first_t0_or_t2,
            zero_runs_unpadded=zero_runs_unpadded,
            trailing_zero_run=trailing_zero_run,
            draw_count_27=draw_27,
            draw_count_2d=draw_2d,
            set_sh_reg_count=set_sh,
            set_context_reg_count=set_ctx,
            nop_ranges=nop_ranges,
        )

    def _check_label_mismatches(self) -> None:
        # Check if the emulator logs label an opcode incorrectly
        # E.g. 0x79 labeled as LOAD_CONFIG_REG, 0x47 as RELEASE_MEM, 0x28 as DRAW_INDEX_2
        re_labeled = re.compile(r"\[PM4\]\s+(IT_[A-Za-z0-9_]+|DRAW_[A-Za-z0-9_]+|LOAD_[A-Za-z0-9_]+|RELEASE_[A-Za-z0-9_]+)", re.IGNORECASE)
        re_op_explicit = re.compile(r"op(?:code)?=0x([0-9a-fA-F]{2})", re.IGNORECASE)

        known_emulator_drifts = {
            "IT_LOAD_CONFIG_REG": (0x79, "LOAD_CONFIG_REG"),
            "IT_RELEASE_MEM": (0x47, "RELEASE_MEM"),
            "IT_DRAW_INDEX_2": (0x28, "DRAW_INDEX_2"),
        }

        for line_no, line in self.log.pm4_lines:
            for emu_label, (op, bad_name) in known_emulator_drifts.items():
                if emu_label in line:
                    canonical = PM4_OPCODE_TABLE.get(op, "UNKNOWN").split()[0]
                    self.label_mismatches.append((line_no, op, bad_name, canonical))

    def _check_timeouts(self) -> None:
        re_timeout = re.compile(r"\b(wait|timeout|stall)\b|flip.*?(?:timeout|stall|failed|error)", re.IGNORECASE)
        frame_idx = 0
        all_lines = sorted(self.log.status_lines + self.log.other_recognized_lines + self.log.pm4_lines, key=lambda x: x[0])
        for line_no, line in all_lines:
            if "SubmitAndFlip" in line:
                frame_idx += 1
                continue
            if re_timeout.search(line):
                # Exclude benign initialization lines
                if not ("disable_stall_watchdog" in line or "g_disable_stall" in line):
                    self.timeout_lines.append((line_no, frame_idx, line.strip()))

    def _analyze(self) -> None:
        self._check_label_mismatches()
        self._check_timeouts()

        for dcb in self.log.dcb_dumps:
            res = self._walk_dcb(dcb)
            self.dcb_results.append(res)

        self._evaluate()

    def match_slice(self, dcb: DcbDump, rdi: int, rsi: int) -> Tuple[str, str]:
        """
        Slice matching for a sceGnm call:
        Returns (status: 'ok'|'not_a_slot'|'boundary_mismatch', description: str).
        """
        dcb_base = dcb.addr
        dcb_size = len(dcb.dwords) * 4

        if rsi > 65536 or rdi < dcb_base or (rdi + rsi * 4) > (dcb_base + dcb_size):
            return "not_a_slot", f"not a slot (rdi=0x{rdi:X} outside DCB 0x{dcb_base:X}..0x{dcb_base + dcb_size:X} or rsi={rsi} > 65536)"

        idx = (rdi - dcb_base) // 4
        res = self._walk_dcb(dcb)
        if idx not in res.packet_boundaries or (idx + rsi) not in res.packet_boundaries:
            return "boundary_mismatch", f"boundary mismatch at slot [{idx}, {idx + rsi})"

        slice_dwords = dcb.dwords[idx : idx + rsi]
        if all(w == 0 for w in slice_dwords):
            return "ok", "all-zero (stub never wrote)"

        # Check if exactly padded NOP
        hdr = slice_dwords[0]
        pkt_type, count, opcode, total_len = decode_pm4_header(hdr)
        if pkt_type == 3 and opcode == 0x10 and total_len == rsi:
            return "ok", f"a NOP of exactly {rsi} DWORDs (stub padded)"

        # Real packets: collect opcodes
        ops: List[str] = []
        cur = 0
        while cur < len(slice_dwords):
            p_hdr = slice_dwords[cur]
            p_type, p_cnt, p_op, p_len = decode_pm4_header(p_hdr)
            if p_type == 3:
                ops.append(f"0x{p_op:02X}:{get_pm4_opcode_name(p_op).split()[0]}")
            else:
                ops.append(f"Type-{p_type}")
            cur += max(p_len, 1)

        return "ok", f"real packets (opcodes: {', '.join(ops)})"

    def _evaluate(self) -> None:
        # 1. Label mismatches
        if self.label_mismatches:
            for line_no, op, log_lbl, can_lbl in self.label_mismatches:
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=(
                        f"PM4 label mismatch on opcode 0x{op:02X}: log labeled packet as '{log_lbl}', "
                        f"canonical CIK/GFX7 table is '{can_lbl}'"
                    ),
                    lines=[line_no]
                ))

        # 2. DCB Dumps Analysis
        for i, res in enumerate(self.dcb_results, 1):
            dcb_lines = [self.log.dcb_dumps[i - 1].start_line, self.log.dcb_dumps[i - 1].end_line]

            # Packet walk summary
            t_str = ", ".join(f"Type-{k}: {v}" for k, v in sorted(res.type_counts.items()) if v > 0)
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"DCB #{i} (addr=0x{res.dcb_addr:X}, {res.total_dwords} DWORDs): packets ({t_str})",
                lines=dcb_lines
            ))

            # Desync indication: Type-0 or Type-2 packet
            if res.first_type0_or_2 is not None:
                t, idx = res.first_type0_or_2
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=f"DCB #{i} desync warning: first Type-{t} packet at DWORD index {idx}",
                    lines=dcb_lines
                ))

            # Zero runs not inside NOP body
            if res.zero_runs_unpadded:
                runs_str = ", ".join(f"index {s} (len={l})" for s, l in res.zero_runs_unpadded[:5])
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=f"DCB #{i} contains {len(res.zero_runs_unpadded)} unpadded zero gap(s): {runs_str}",
                    lines=dcb_lines
                ))

            if res.trailing_zero_run:
                s, l = res.trailing_zero_run
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=f"DCB #{i} trailing zero run: index {s} to {s + l} ({l} DWORDs)",
                    lines=dcb_lines
                ))

            # Draw summary
            total_draws = res.draw_count_27 + res.draw_count_2d
            draw_msg = (
                f"DCB #{i} draw summary: 0x27(DRAW_INDEX_2)={res.draw_count_27}, "
                f"0x2D(DRAW_INDEX_AUTO)={res.draw_count_2d}, SET_SH_REG={res.set_sh_reg_count}, "
                f"SET_CONTEXT_REG={res.set_context_reg_count}"
            )
            self.findings.append(Finding(kind="OBSERVED", message=draw_msg, lines=dcb_lines))

            if total_draws == 0:
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=f"DCB #{i}: no draw packets present",
                    lines=dcb_lines
                ))
            elif res.set_sh_reg_count == 0 and res.set_context_reg_count == 0:
                self.findings.append(Finding(
                    kind="OBSERVED",
                    message=f"DCB #{i}: draws present but all shader slots NOP/zero",
                    lines=dcb_lines
                ))

        # 3. Timeout / stall lines
        if self.timeout_lines:
            lines = [t[0] for t in self.timeout_lines]
            sample = self.timeout_lines[:5]
            sample_str = "; ".join(f"F#{f}: '{txt}'" for _, f, txt in sample)
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Flip/wait/timeout/stall events ({len(self.timeout_lines)} detected): {sample_str}",
                lines=lines
            ))


class HotLoopAnalyzer:
    """
    STAGE 6: HotLoopAnalyzer
    - RIP histogram: parses existing histogram or builds one from per-RIP trace lines
    - Flags any set of <= 16 addresses holding > 50% of instructions
    - Prints addresses as module+offset, the share, and register values logged at loop entry
    """

    def __init__(self, log: Log, modules: Optional[List[ModuleSpan]] = None, registers: Optional[Dict[str, int]] = None):
        self.log = log
        self.modules = modules or []
        self.registers = registers or {}
        self.rip_counts: Dict[int, int] = {}
        self.rip_lines: Dict[int, List[int]] = {}
        self.total_samples: int = 0

        self.hot_loop_detected: bool = False
        self.hot_addrs: List[Tuple[int, int, float]] = []
        self.hot_share: float = 0.0

        self.findings: List[Finding] = []
        self._analyze()

    def _format_addr(self, rip: int) -> str:
        for m in sorted(self.modules, key=lambda x: x.base, reverse=True):
            if m.base <= rip < m.base + m.span:
                return f"{m.path}+0x{rip - m.base:X}"
        return f"0x{rip:X}"

    def _analyze(self) -> None:
        re_hist = re.compile(r"(?:\[hist\]\s+)?(?:RIP=)?0x([0-9a-fA-F]+)[:=]\s*(\d+)", re.IGNORECASE)
        re_trace = re.compile(r"\[CPU TRACE\]\s+RIP=0x([0-9a-fA-F]+)", re.IGNORECASE)
        re_last_rips = re.compile(r"^\s*#\d+:\s*0x([0-9a-fA-F]+)", re.IGNORECASE)
        re_rip_hist_num = re.compile(r"^\s*\[\d+\]\s+RIP=0x([0-9a-fA-F]+)", re.IGNORECASE)

        has_explicit_hist = False
        in_hist_section = False
        for line_no, line in self.log.trace_lines:
            if "Histogram" in line or "[hist]" in line:
                in_hist_section = True
            m_h = re_hist.search(line)
            if m_h and (in_hist_section or "[hist]" in line or "Histogram" in line):
                rip = int(m_h.group(1), 16)
                cnt = int(m_h.group(2))
                self.rip_counts[rip] = cnt
                self.rip_lines.setdefault(rip, []).append(line_no)
                has_explicit_hist = True
            elif line.startswith("[") and not line.startswith("[hist]"):
                in_hist_section = False

        if not has_explicit_hist:
            for line_no, line in self.log.trace_lines:
                m_t = re_trace.search(line) or re_last_rips.search(line) or re_rip_hist_num.search(line)
                if m_t:
                    rip = int(m_t.group(1), 16)
                    self.rip_counts[rip] = self.rip_counts.get(rip, 0) + 1
                    self.rip_lines.setdefault(rip, []).append(line_no)

        self.total_samples = sum(self.rip_counts.values())
        if self.total_samples > 0:
            sorted_rips = sorted(self.rip_counts.items(), key=lambda x: x[1], reverse=True)
            running_cnt = 0
            selected = []
            for rip, cnt in sorted_rips[:16]:
                running_cnt += cnt
                selected.append((rip, cnt, cnt / self.total_samples))
                if running_cnt / self.total_samples > 0.50:
                    self.hot_loop_detected = True
                    self.hot_addrs = selected
                    self.hot_share = running_cnt / self.total_samples
                    break

        self._evaluate()

    def _evaluate(self) -> None:
        if self.hot_loop_detected:
            addrs_desc = []
            ev_lines = []
            for rip, cnt, sh in self.hot_addrs:
                addrs_desc.append(f"{self._format_addr(rip)} ({cnt} hits, {sh*100:.1f}%)")
                ev_lines.extend(self.rip_lines.get(rip, []))

            regs_str = ""
            if self.registers:
                regs_sample = ", ".join(f"{k}=0x{v:X}" for k, v in list(self.registers.items())[:6])
                regs_str = f" [Registers at entry: {regs_sample}]"

            self.findings.append(Finding(
                kind="OBSERVED",
                message=(
                    f"Hot loop detected: {len(self.hot_addrs)} address(es) account for "
                    f"{self.hot_share*100:.1f}% of instructions: {'; '.join(addrs_desc)}{regs_str}"
                ),
                lines=ev_lines
            ))


class ThreadAnalyzer:
    """
    STAGE 6: ThreadAnalyzer
    - Tracks per-TID last HLE call and call counts
    - Tracks parked / timeout lines
    - Computes semaphores created minus released (only if both appear in the log)
    - Reports facts, labels any deadlock reading as INFERRED with the rule
    """

    def __init__(self, log: Log):
        self.log = log
        self.tid_call_counts: Dict[int, int] = {}
        self.tid_last_call: Dict[int, str] = {}
        self.tid_states: Dict[int, str] = {}

        self.parked_lines: List[Tuple[int, str]] = []

        self.sema_created_count: int = 0
        self.sema_created_lines: List[int] = []
        self.sema_released_count: int = 0
        self.sema_released_lines: List[int] = []

        self.findings: List[Finding] = []
        self._analyze()

    def _analyze(self) -> None:
        re_hle_tid = re.compile(r"\[HLE\s+(?:WARNING\s+#[0-9a-fA-F]+\s+)?TID=(\d+).*?\]\s*(.*)", re.IGNORECASE)
        re_thread_state = re.compile(r"\[Thread\s+(\d+)\]\s+.*?\s+state=([A-Za-z0-9_]+)", re.IGNORECASE)
        re_parked = re.compile(r"threads still parked after|watchdog: threads still parked|parked", re.IGNORECASE)

        re_sema_create = re.compile(r"\[Sema\]\s+CreateSema", re.IGNORECASE)
        re_sema_release = re.compile(r"\[Sema\]\s+(?:SignalSema|DeleteSema)", re.IGNORECASE)

        for line_no, line in self.log.thread_lines + self.log.hle_lines:
            m_t = re_hle_tid.search(line)
            if m_t:
                tid = int(m_t.group(1))
                desc = m_t.group(2).strip()
                self.tid_call_counts[tid] = self.tid_call_counts.get(tid, 0) + 1
                self.tid_last_call[tid] = desc

            m_st = re_thread_state.search(line)
            if m_st:
                tid = int(m_st.group(1))
                st = m_st.group(2).strip()
                self.tid_states[tid] = st

            if re_parked.search(line):
                self.parked_lines.append((line_no, line))

            if re_sema_create.search(line):
                self.sema_created_count += 1
                self.sema_created_lines.append(line_no)

            if re_sema_release.search(line):
                self.sema_released_count += 1
                self.sema_released_lines.append(line_no)

        self._evaluate()

    def _evaluate(self) -> None:
        if self.tid_call_counts:
            tid_descs = []
            for tid, count in sorted(self.tid_call_counts.items()):
                last = self.tid_last_call.get(tid, "none")
                state = self.tid_states.get(tid, "unknown")
                tid_descs.append(f"TID {tid} (calls={count}, state={state}, last='{last[:30]}')")
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Thread activity: {'; '.join(tid_descs[:8])}",
                lines=[]
            ))

        if self.parked_lines:
            lines = [p[0] for p in self.parked_lines]
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Parked thread events detected: '{self.parked_lines[0][1]}'",
                lines=lines
            ))

        if self.sema_created_count > 0 and self.sema_released_count > 0:
            diff = self.sema_created_count - self.sema_released_count
            lines = self.sema_created_lines[:2] + self.sema_released_lines[:2]
            self.findings.append(Finding(
                kind="OBSERVED",
                message=(
                    f"Semaphore balance: created={self.sema_created_count}, "
                    f"released/deleted={self.sema_released_count}, unreleased balance={diff}"
                ),
                lines=lines
            ))
        elif self.sema_created_count > 0:
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Semaphores created={self.sema_created_count} (semaphore releases not present in log)",
                lines=self.sema_created_lines[:2]
            ))
        elif self.sema_released_count > 0:
            self.findings.append(Finding(
                kind="OBSERVED",
                message=f"Semaphores released={self.sema_released_count} (semaphore creations not present in log)",
                lines=self.sema_released_lines[:2]
            ))

        waiting_count = sum(1 for st in self.tid_states.values() if "wait" in st.lower())
        if self.parked_lines or (waiting_count > 0 and waiting_count >= len(self.tid_states) - 1 and len(self.tid_states) > 1):
            rule_msg = "threads parked at termination or all worker threads in wait state"
            self.findings.append(Finding(
                kind="INFERRED",
                message=f"Potential deadlock: {len(self.parked_lines)} parked event(s), {waiting_count} thread(s) in wait states",
                rule=rule_msg,
                lines=[p[0] for p in self.parked_lines] if self.parked_lines else []
            ))


class DiffAnalyzer:
    """
    STAGE 7: DiffAnalyzer
    Compares OLD and NEW logs:
    - mode
    - final instruction count
    - stop reason
    - fault RIP module+offset
    - fallback NID set (added/removed with counts)
    - HLE call count changes > 20%
    - DCB draw counts
    - overlap findings
    - hot-loop share
    Outputs a short table of changes.
    If stop reason is unchanged and fault RIP is identical, reports 'no behavioural change'.
    """

    def __init__(self, old_log: Log, new_log: Log, repo_dir: Optional[str] = None):
        self.old_log = old_log
        self.new_log = new_log
        self.repo_dir = repo_dir

        self.old_cfa = CpuFaultAnalyzer(old_log, repo_dir=repo_dir)
        self.new_cfa = CpuFaultAnalyzer(new_log, repo_dir=repo_dir)

        resolver = NidResolver(repo_dir=repo_dir)
        self.old_hle = HleAnalyzer(old_log, resolver)
        self.new_hle = HleAnalyzer(new_log, resolver)

        self.old_loader = LoaderAnalyzer(old_log, fault_rip=self.old_cfa.rip)
        self.new_loader = LoaderAnalyzer(new_log, fault_rip=self.new_cfa.rip)

        self.old_gpu = GpuPm4Analyzer(old_log)
        self.new_gpu = GpuPm4Analyzer(new_log)

        self.old_hot = HotLoopAnalyzer(old_log, modules=self.old_loader.modules, registers=self.old_cfa.registers)
        self.new_hot = HotLoopAnalyzer(new_log, modules=self.new_loader.modules, registers=self.new_cfa.registers)

        self.diff_data: Dict[str, Any] = {}
        self.no_behavioural_change: bool = False
        self._compute_diff()

    def _format_rip(self, cfa: CpuFaultAnalyzer, loader: LoaderAnalyzer) -> str:
        if cfa.rip is None:
            return "none"
        for m in sorted(loader.modules, key=lambda x: x.base, reverse=True):
            if m.base <= cfa.rip < m.base + m.span:
                return f"{m.path}+0x{cfa.rip - m.base:X}"
        return f"0x{cfa.rip:X}"

    def _compute_diff(self) -> None:
        # 1. Mode
        old_mode = self.old_log.mode
        new_mode = self.new_log.mode

        # 2. Final instruction count
        old_ins = self.old_log.final_instructions
        new_ins = self.new_log.final_instructions

        # 3. Stop reason
        old_stop = self.old_log.stop_reason or "not present in log"
        new_stop = self.new_log.stop_reason or "not present in log"

        # 4. Fault RIP
        old_rip_str = self._format_rip(self.old_cfa, self.old_loader)
        new_rip_str = self._format_rip(self.new_cfa, self.new_loader)

        # Behavioral check: stop reason unchanged and fault RIP identical
        if (old_stop == new_stop) and (self.old_cfa.rip == self.new_cfa.rip):
            self.no_behavioural_change = True

        # 5. Fallback NIDs
        old_nids = set(self.old_hle.fallback_stubs.keys())
        new_nids = set(self.new_hle.fallback_stubs.keys())

        added_nids = [
            {"nid": f"0x{nid:016X}", "count": self.new_hle.fallback_stubs[nid].count}
            for nid in sorted(new_nids - old_nids)
        ]
        removed_nids = [
            {"nid": f"0x{nid:016X}", "count": self.old_hle.fallback_stubs[nid].count}
            for nid in sorted(old_nids - new_nids)
        ]

        # 6. HLE call count changes > 20%
        all_call_names = set(self.old_hle.call_counts.keys()) | set(self.new_hle.call_counts.keys())
        hle_changes = []
        for name in sorted(all_call_names):
            cnt_old = self.old_hle.call_counts.get(name, 0)
            cnt_new = self.new_hle.call_counts.get(name, 0)
            if cnt_old > 0 and cnt_new > 0:
                pct = ((cnt_new - cnt_old) / cnt_old) * 100.0
                if abs(pct) > 20.0:
                    hle_changes.append({"call": name, "old": cnt_old, "new": cnt_new, "pct_change": round(pct, 1)})
            elif cnt_old == 0 and cnt_new > 0:
                hle_changes.append({"call": name, "old": 0, "new": cnt_new, "pct_change": "+new"})
            elif cnt_old > 0 and cnt_new == 0:
                hle_changes.append({"call": name, "old": cnt_old, "new": 0, "pct_change": "-100.0%"})

        # 7. DCB draw counts
        old_draws = sum(res.draw_count_27 + res.draw_count_2d for res in self.old_gpu.dcb_results)
        new_draws = sum(res.draw_count_27 + res.draw_count_2d for res in self.new_gpu.dcb_results)

        # 8. Overlaps
        old_overlaps = len(self.old_loader.overlap_pairs)
        new_overlaps = len(self.new_loader.overlap_pairs)

        # 9. Hot-loop share
        old_hot_share = round(self.old_hot.hot_share * 100.0, 1)
        new_hot_share = round(self.new_hot.hot_share * 100.0, 1)

        self.diff_data = {
            "mode": {"old": old_mode, "new": new_mode, "changed": old_mode != new_mode},
            "instructions": {
                "old": old_ins if old_ins is not None else "not present in log",
                "new": new_ins if new_ins is not None else "not present in log",
                "delta": (new_ins - old_ins) if (old_ins is not None and new_ins is not None) else "unknown"
            },
            "stop_reason": {"old": old_stop, "new": new_stop, "changed": old_stop != new_stop},
            "fault_rip": {"old": old_rip_str, "new": new_rip_str, "identical": old_rip_str == new_rip_str},
            "fallback_nids": {"added": added_nids, "removed": removed_nids},
            "hle_call_changes_gt_20pct": hle_changes,
            "dcb_draws": {"old": old_draws, "new": new_draws, "delta": new_draws - old_draws},
            "overlap_findings": {"old": old_overlaps, "new": new_overlaps},
            "hot_loop_share": {"old_pct": old_hot_share, "new_pct": new_hot_share},
            "no_behavioural_change": self.no_behavioural_change,
        }

    def format_table(self) -> str:
        d = self.diff_data
        lines = [
            "=== DIFF SUMMARY ===",
            f"{'Metric':<24} {'OLD':<28} {'NEW':<28} {'Change'}",
            "-" * 90,
            f"{'Mode':<24} {str(d['mode']['old']):<28} {str(d['mode']['new']):<28} {'changed' if d['mode']['changed'] else 'unchanged'}",
            f"{'Instructions':<24} {str(d['instructions']['old']):<28} {str(d['instructions']['new']):<28} {str(d['instructions']['delta'])}",
            f"{'Stop Reason':<24} {str(d['stop_reason']['old'])[:26]:<28} {str(d['stop_reason']['new'])[:26]:<28} {'changed' if d['stop_reason']['changed'] else 'unchanged'}",
            f"{'Fault RIP':<24} {str(d['fault_rip']['old']):<28} {str(d['fault_rip']['new']):<28} {'identical' if d['fault_rip']['identical'] else 'changed'}",
            f"{'Fallback NIDs':<24} {len(self.old_hle.fallback_stubs):<28} {len(self.new_hle.fallback_stubs):<28} +{len(d['fallback_nids']['added'])}, -{len(d['fallback_nids']['removed'])}",
            f"{'DCB Draws':<24} {d['dcb_draws']['old']:<28} {d['dcb_draws']['new']:<28} {d['dcb_draws']['delta']:+d}",
            f"{'Overlap Findings':<24} {d['overlap_findings']['old']:<28} {d['overlap_findings']['new']:<28} {d['overlap_findings']['new'] - d['overlap_findings']['old']:+d}",
            f"{'Hot-loop Share':<24} {str(d['hot_loop_share']['old_pct']) + '%':<28} {str(d['hot_loop_share']['new_pct']) + '%':<28} {d['hot_loop_share']['new_pct'] - d['hot_loop_share']['old_pct']:+.1f}%",
            "-" * 90,
        ]

        if d["hle_call_changes_gt_20pct"]:
            lines.append("HLE Call Changes (>20%):")
            for chg in d["hle_call_changes_gt_20pct"][:10]:
                lines.append(f"  {chg['call']}: {chg['old']} -> {chg['new']} ({chg['pct_change']}%)")
            lines.append("-" * 90)

        if self.no_behavioural_change:
            lines.append("RESULT: no behavioural change")

        return "\n".join(lines)


class RemediationGenerator:
    """
    STAGE 8: Remediation Templates (only after 1-7 are tested)
    - C++ stub templates for fallback NIDs with CONFIRMED name (registration, comment with source, TODOs for out-pointers)
    - For UNIDENTIFIED NIDs: // UNIDENTIFIED NID 0x... hits=N first caller=...
    - For unhandled PM4 opcodes: switch-case template for src/gpu/pm4_translator.mm with header decode,
      count-based skip so cursor stays in sync, and 'verify packet layout against SDK'
    - For unhandled x87/other opcodes: opcode, encoding, and pointer to area via triage_map.toml (no generated emulation code)
    - Every template starts with '// GENERATED TEMPLATE - VERIFY BEFORE USE'
    """

    def __init__(self, map_file: Optional[str] = None):
        self.map_file = map_file
        self.area_map: Dict[str, str] = {}
        self._load_map()

    def _load_map(self) -> None:
        path = self.map_file or "triage_map.toml"
        if not os.path.isfile(path):
            cur = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(cur, "..", "triage_map.toml")
            if os.path.isfile(cand):
                path = cand

        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            self.area_map[k.strip().lower()] = v.strip().strip('"').strip("'")
            except Exception:
                pass

    def get_area_pointer(self, key: str, default: str) -> str:
        return self.area_map.get(key.lower(), default)

    def generate_nid_stub(self, nid: int, rec: FallbackStubRecord, resolver: NidResolver) -> str:
        status, details, confirmed_name = resolver.resolve(nid)
        first_caller = f"0x{rec.first_hit_caller_rip:X}" if rec.first_hit_caller_rip is not None else "unknown"

        if status == "MATCH" and confirmed_name:
            args_comment = ", ".join(f"{k}={v}" for k, v in rec.first_hit_args.items()) or "none"
            return (
                "// GENERATED TEMPLATE - VERIFY BEFORE USE\n"
                f"// Target: src/hle/nid_registry.cpp\n"
                f"// Name: {confirmed_name} | Source: {details}\n"
                f"// First caller: {first_caller} | Observed args: {args_comment}\n"
                f'register_symbol(0x{nid:016X}ULL, "{confirmed_name}", "libkernel",\n'
                "    [](ExecutionContext& ctx, MemoryManager& mem) {\n"
                "        // TODO: Validate input arguments and perform out-pointer writes\n"
                "        // TODO: Map error codes according to SDK documentation\n"
                "        ctx.rax = 0; // SCE_OK\n"
                "    });"
            )
        else:
            return (
                "// GENERATED TEMPLATE - VERIFY BEFORE USE\n"
                f"// UNIDENTIFIED NID 0x{nid:016X} hits={rec.count} first caller={first_caller}"
            )

    def generate_pm4_template(self, opcode: int) -> str:
        name = PM4_OPCODE_TABLE.get(opcode, "UNKNOWN").split()[0]
        area = self.get_area_pointer("pm4", "src/gpu/pm4_translator.mm (process_packet switch)")
        return (
            "// GENERATED TEMPLATE - VERIFY BEFORE USE\n"
            f"// Target: src/gpu/pm4_translator.mm (area: {area})\n"
            f"// Opcode: 0x{opcode:02X} ({name})\n"
            f"case 0x{opcode:02X}: {{ // {name} - verify packet layout against SDK\n"
            "    const uint32_t count = (header >> 16) & 0x3FFF;\n"
            "    // Count-based skip so the packet cursor stays in sync\n"
            "    cursor += count + 1;\n"
            "    break;\n"
            "}"
        )

    def generate_opcode_template(self, opcode_raw: bytes, rip: Optional[int] = None) -> str:
        decoded, encoding_label, op_class = decode_unhandled_opcode_bytes(opcode_raw)
        first_hex = f"{opcode_raw[0]:02X}" if opcode_raw else "UNKNOWN"
        area = self.get_area_pointer(op_class.lower(), self.get_area_pointer(first_hex.lower(), "src/cpu/executor.cpp (x87/unhandled opcode decode)"))
        rip_str = f" @ 0x{rip:X}" if rip is not None else ""
        return (
            "// GENERATED TEMPLATE - VERIFY BEFORE USE\n"
            f"// Unhandled Opcode: {opcode_raw.hex(' ')}{rip_str}\n"
            f"// Decoding: {decoded} (class: {op_class}, encoding: {encoding_label})\n"
            f"// Pointer to implementation area: {area}\n"
            "// Note: verify instruction semantics against Intel 64 and IA-32 Architectures Software Developer Manual (SDM)."
        )


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
        self.mem_fault_raw_lines: List[Tuple[int, str]] = []
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
        re_mode = re.compile(r"\[prx\]\s+mode=(real|hle)", re.IGNORECASE)
        re_budget = re.compile(r"(?:budget[:=]\s*(\d+)|--budget\s+(\d+))", re.IGNORECASE)
        re_cpu_status = re.compile(r"\[CPU\]\s+Execution Status:\s*(.+)", re.IGNORECASE)
        re_cpu_instrs = re.compile(r"\[CPU\]\s+Instructions Executed:\s*(\d+)", re.IGNORECASE)
        re_wall_time = re.compile(r"(?:wall[\s_-]?time|elapsed|duration)[:=]\s*([\d\.]+(?:s|ms|m)?)", re.IGNORECASE)

        re_dcb_start = re.compile(r"\[DCB FULL DUMP START 0x([0-9a-fA-F]+)\s+count=(\d+)\]")
        re_dcb_dword = re.compile(r"DWORD\[([0-9a-fA-F]+)\]\[(\d+)\]=0x([0-9a-fA-F]+)")
        re_dcb_end = re.compile(r"\[DCB FULL DUMP END 0x([0-9a-fA-F]+)\]")

        current_dcb: Optional[DcbDump] = None
        in_map_block = False

        for line_no, raw_line in enumerate(stream, 1):
            self.total_lines += 1
            line = raw_line.strip()
            recognized = False

            if in_map_block:
                if raw_line.startswith("  ") or line.startswith("owner") or line.startswith("---"):
                    self.loader_lines.append((line_no, line))
                    recognized = True
                elif not line.startswith("[map]"):
                    in_map_block = False

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

            m_dcb_start = re_dcb_start.match(line)
            if m_dcb_start:
                addr = int(m_dcb_start.group(1), 16)
                count = int(m_dcb_start.group(2))
                current_dcb = DcbDump(start_line=line_no, end_line=line_no, addr=addr, count=count, dwords=[])
                continue

            m_mode = re_mode.search(line)
            if m_mode:
                self.mode = m_mode.group(1).lower()
                self.mode_lines.append(line_no)
                recognized = True

            m_b = re_budget.search(line)
            if m_b:
                self.budget = int(m_b.group(1) or m_b.group(2))
                self.budget_lines.append(line_no)
                recognized = True

            m_st = re_cpu_status.search(line)
            if m_st:
                self.stop_reason = m_st.group(1).strip()
                self.stop_reason_lines.append(line_no)
                recognized = True

            m_ins = re_cpu_instrs.search(line)
            if m_ins:
                self.final_instructions = int(m_ins.group(1))
                self.final_instructions_lines.append(line_no)
                recognized = True

            m_wt = re_wall_time.search(line)
            if m_wt:
                self.wall_time = m_wt.group(1).strip()
                self.wall_time_lines.append(line_no)
                recognized = True

            if "[CPU FAULT]" in line or "CPU FAULT" in line:
                self.cpu_fault_lines.append((line_no, line))
                recognized = True

            if "unhandled opcode" in line.lower():
                self.unhandled_opcode_lines.append((line_no, line))
                recognized = True

            if (line.startswith("[HLE") or line.startswith("[HLE CALL") or
                    line.startswith("[HLE WARNING") or line.startswith("[HLE WHITELIST") or
                    "Unimplemented stub called:" in line):
                self.hle_lines.append((line_no, line))
                recognized = True

            if (line.startswith("[PM4") or line.startswith("[Gnm]") or
                    line.startswith("[Gnm Warning]") or line.startswith("[PM4 Warning]")):
                self.pm4_lines.append((line_no, line))
                recognized = True

            if (line.startswith("[module]") or line.startswith("[exports]") or
                    line.startswith("[dyn]") or line.startswith("[map]") or
                    line.startswith("[elf]")):
                self.loader_lines.append((line_no, line))
                if "[map] registered guest mappings" in line:
                    in_map_block = True
                recognized = True

            if line.startswith("[watch]"):
                self.watch_lines.append((line_no, line))
                recognized = True

            if (line.startswith("[Thread") or line.startswith("[Threads]") or
                    line.startswith("[Sema]") or "=== Thread States Dump" in line or
                    "=== [SEMA] table snapshot" in line):
                self.thread_lines.append((line_no, line))
                recognized = True

            # Memory fault / segfault / OOB
            if re.search(r"segmentation fault|sigsegv|segfault|out-of-bounds|out of bounds", line, re.IGNORECASE):
                self.mem_fault_raw_lines.append((line_no, line))
                recognized = True

            if (line.startswith("[CPU TRACE]") or line.startswith("[CPU Progress]") or
                    line.startswith("[hist]") or "RIP Histogram" in line or
                    line.startswith("Last RIPs:") or re.match(r"^\s*#\d+:\s*0x[0-9a-fA-F]+", line) or
                    re.match(r"^\s*\[\d+\]\s+RIP=0x[0-9a-fA-F]+", line) or
                    re.match(r"^0x[0-9a-fA-F]+:\s*\d+", line)):
                self.trace_lines.append((line_no, line))
                recognized = True

            if (line.startswith("[CPU]") or line.startswith("-------------------- CPU Register Context") or
                    line.startswith("RIP:") or line.startswith("RAX:") or line.startswith("RSI:") or
                    line.startswith("R8:") or line.startswith("R12:") or line.startswith("FS_BASE:")):
                self.status_lines.append((line_no, line))
                recognized = True

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

        if current_dcb is not None:
            current_dcb.end_line = self.total_lines
            self.dcb_dumps.append(current_dcb)

    def banner_dict(self) -> Dict[str, Any]:
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


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Bare 'triage_crash.py LOG' -> 'triage LOG'
    subcommands = {"triage", "diff", "nid", "dcb", "hot", "history"}
    if argv and argv[0] not in subcommands and not argv[0].startswith("-"):
        argv = ["triage"] + argv

    parser = argparse.ArgumentParser(description="Triage PS4-to-macOS emulator logs")
    subparsers = parser.add_subparsers(dest="subcommand")

    # triage
    p_triage = subparsers.add_parser("triage", help="Triage an emulator log")
    p_triage.add_argument("log", help="Path to log file")
    p_triage.add_argument("--json", action="store_true", help="Emit JSON output")
    p_triage.add_argument("--map", dest="map_file", default=None, help="Mapping table TOML file")
    p_triage.add_argument("--repo", default=None, help="Path to ps4-to-macos repository")

    # diff
    p_diff = subparsers.add_parser("diff", help="Diff two crash logs")
    p_diff.add_argument("old_log", help="Path to OLD log file")
    p_diff.add_argument("new_log", help="Path to NEW log file")
    p_diff.add_argument("--json", action="store_true", help="Emit JSON output")
    p_diff.add_argument("--repo", default=None, help="Path to ps4-to-macos repository")

    # nid
    p_nid = subparsers.add_parser("nid", help="Resolve Sony NIDs or compute NID for symbol names")
    p_nid.add_argument("items", nargs="+", help="NID (hex) or symbol name(s)")
    p_nid.add_argument("--names", default=None, help="File containing candidate symbol names")
    p_nid.add_argument("--repo", default=None, help="Path to ps4-to-macos repository")

    # dcb
    p_dcb = subparsers.add_parser("dcb", help="Inspect and walk DCB dumps")
    p_dcb.add_argument("log", help="Path to log file")
    p_dcb.add_argument("--frame", type=int, default=None, help="Frame / DCB index (1-based)")

    # hot
    p_hot = subparsers.add_parser("hot", help="Analyze hot loops and RIP distribution")
    p_hot.add_argument("log", help="Path to log file")

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "diff":
        old_log = Log(args.old_log)
        new_log = Log(args.new_log)
        differ = DiffAnalyzer(old_log, new_log, repo_dir=args.repo)
        if args.json:
            import json
            print(json.dumps(differ.diff_data, indent=2))
        else:
            print(differ.format_table())
        return 0

    if args.subcommand == "hot":
        log = Log(args.log)
        loader = LoaderAnalyzer(log)
        cfa = CpuFaultAnalyzer(log)
        hot = HotLoopAnalyzer(log, modules=loader.modules, registers=cfa.registers)
        if hot.findings:
            for f in hot.findings:
                print(str(f))
        else:
            print("No hot loop (>50% instructions in <=16 addresses) detected.")
        return 0

    if args.subcommand == "dcb":
        log = Log(args.log)
        gpu = GpuPm4Analyzer(log)
        if not log.dcb_dumps:
            print("No DCB dumps found in log.")
            return 0
        indices = [args.frame - 1] if (args.frame is not None and 1 <= args.frame <= len(log.dcb_dumps)) else range(len(log.dcb_dumps))
        for idx in indices:
            dcb = log.dcb_dumps[idx]
            res = gpu._walk_dcb(dcb)
            print(f"=== DCB #{idx + 1} @ 0x{res.dcb_addr:X} ({res.total_dwords} DWORDs) ===")
            print(f"  Type counts: {res.type_counts}")
            print(f"  Draws: 0x27={res.draw_count_27}, 0x2D={res.draw_count_2d}, SET_SH={res.set_sh_reg_count}, SET_CTX={res.set_context_reg_count}")
            if res.zero_runs_unpadded:
                print(f"  Unpadded zero runs: {res.zero_runs_unpadded}")
            if res.trailing_zero_run:
                print(f"  Trailing zero run: {res.trailing_zero_run}")
        return 0

    if args.subcommand == "triage":
        log = Log(args.log)
        print(log.banner_str())

        cfa = CpuFaultAnalyzer(log, repo_dir=args.repo)
        if cfa.findings:
            print("\n=== CPU FAULT ANALYSIS ===")
            for f in cfa.findings:
                print(str(f))

        resolver = NidResolver(repo_dir=args.repo)
        hle = HleAnalyzer(log, resolver)
        if hle.findings:
            print("\n=== HLE ANALYSIS ===")
            for f in hle.findings:
                print(str(f))

        loader = LoaderAnalyzer(log, fault_rip=cfa.rip)
        if loader.findings:
            print("\n=== LOADER ANALYSIS ===")
            for f in loader.findings:
                print(str(f))

        gpu = GpuPm4Analyzer(log)
        if gpu.findings:
            print("\n=== GPU PM4 & DCB ANALYSIS ===")
            for f in gpu.findings:
                print(str(f))

        hot = HotLoopAnalyzer(log, modules=loader.modules, registers=cfa.registers)
        if hot.findings:
            print("\n=== HOT LOOP ANALYSIS ===")
            for f in hot.findings:
                print(str(f))

        threads = ThreadAnalyzer(log)
        if threads.findings:
            print("\n=== THREADING ANALYSIS ===")
            for f in threads.findings:
                print(str(f))

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
