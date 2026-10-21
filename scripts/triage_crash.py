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

        for line_no, raw_line in enumerate(stream, 1):
            self.total_lines += 1
            line = raw_line.strip()
            recognized = False

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

            if line.startswith("[CPU TRACE]") or line.startswith("[CPU Progress]"):
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

    # nid
    p_nid = subparsers.add_parser("nid", help="Resolve Sony NIDs or compute NID for symbol names")
    p_nid.add_argument("items", nargs="+", help="NID (hex) or symbol name(s)")
    p_nid.add_argument("--names", default=None, help="File containing candidate symbol names")
    p_nid.add_argument("--repo", default=None, help="Path to ps4-to-macos repository")

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "nid":
        resolver = NidResolver(repo_dir=args.repo, names_file=args.names)
        for item in args.items:
            # Check if hex NID
            if item.lower().startswith("0x") or (len(item) == 16 and all(c in "0123456789abcdefABCDEF" for c in item)):
                try:
                    val = int(item, 16)
                    status, details, _ = resolver.resolve(val)
                    print(f"0x{val:016X}: {status} {details}")
                except ValueError:
                    print(f"Invalid hex NID: {item}")
            else:
                # Symbol name: compute NIDs
                le, be = compute_sony_nid(item)
                print(f"{item} -> NID_LE=0x{le:016X}, NID_BE=0x{be:016X}")
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

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
