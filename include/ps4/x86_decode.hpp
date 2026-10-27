#pragma once

// Minimal x86-64 operand decoder shared by the JIT front end.
//
// It decodes legacy prefixes, REX, ModRM, SIB and displacements into a
// structured form; it does not know about individual opcodes. Callers read
// the opcode byte(s) themselves, call decode_modrm() when the opcode has a
// ModRM byte, then read any immediate and call X86ModRM::resolve_rip() once
// the full instruction length is known (RIP-relative operands are relative to
// the address of the *next* instruction).

#include "ps4/common.hpp"
#include <cstring>

namespace ps4::x86 {

inline constexpr usize MAX_INSN_LEN = 15;

struct Prefixes {
  bool opsize16{false}; // 0x66
  bool addr32{false};   // 0x67
  bool lock{false};     // 0xF0
  u8 rep{0};            // 0xF2 / 0xF3 (last one seen), 0 = none
  u8 seg{0};            // 0x64 (FS) / 0x65 (GS), 0 = none. CS/DS/ES/SS are
                        // ignored in 64-bit mode and not recorded.
  bool has_rex{false};
  u8 rex{0};
  bool rex_w{false}, rex_r{false}, rex_x{false}, rex_b{false};
  u8 length{0}; // bytes consumed by prefixes (incl. REX)

  /// Operand size of a non-byte instruction (REX.W wins over 0x66).
  [[nodiscard]] u8 op_bits() const noexcept {
    return rex_w ? 64 : (opsize16 ? 16 : 32);
  }
};

/// Decode legacy prefixes followed by an optional REX prefix. Matches the
/// interpreter: REX must be the last prefix (decoding stops right after it).
/// Returns false if the bytes run out before an opcode byte.
inline bool decode_prefixes(const u8 *code, usize avail, Prefixes &p) {
  usize i = 0;
  while (i < avail && i < MAX_INSN_LEN) {
    const u8 b = code[i];
    if (b == 0x66) {
      p.opsize16 = true;
    } else if (b == 0x67) {
      p.addr32 = true;
    } else if (b == 0xF0) {
      p.lock = true;
    } else if (b == 0xF2 || b == 0xF3) {
      p.rep = b;
    } else if (b == 0x64 || b == 0x65) {
      p.seg = b;
    } else if (b == 0x2E || b == 0x3E || b == 0x26 || b == 0x36) {
      // null segment overrides in 64-bit mode
    } else if ((b & 0xF0) == 0x40) {
      p.has_rex = true;
      p.rex = b;
      p.rex_w = (b & 8) != 0;
      p.rex_r = (b & 4) != 0;
      p.rex_x = (b & 2) != 0;
      p.rex_b = (b & 1) != 0;
      i++;
      break;
    } else {
      break;
    }
    i++;
  }
  p.length = static_cast<u8>(i);
  return i < avail;
}

struct ModRM {
  u8 mod{0};
  u8 reg{0};          // ModRM.reg | REX.R << 3 (register or /digit)
  bool is_reg{false}; // mod == 3
  u8 rm{0};           // register number when is_reg (ModRM.rm | REX.B << 3)

  // Memory form: EA = [base] + [index << scale] + disp (+ FS/GS base)
  i8 base{-1};  // GPR number or -1
  i8 index{-1}; // GPR number or -1 (SIB index 100 without REX.X = none)
  u8 scale{0};  // shift amount 0..3
  i32 disp{0};
  bool rip_relative{false};
  // SIB base=101 with mod=00 and REX.B=1: architecturally "no base, disp32"
  // (REX.B is ignored), but the interpreter decodes it as [r13 + index].
  // Flagged so translators can refuse it rather than diverge.
  bool sib_nobase_with_rex_b{false};

  u8 length{0}; // ModRM + SIB + displacement bytes

  /// Absolute address of a RIP-relative operand, given the address of the
  /// following instruction.
  [[nodiscard]] u64 rip_target(u64 next_rip) const noexcept {
    return next_rip + static_cast<u64>(static_cast<i64>(disp));
  }
};

/// Decode ModRM (+SIB, +displacement) starting at `code`.
inline bool decode_modrm(const u8 *code, usize avail, const Prefixes &p,
                         ModRM &m) {
  if (avail < 1)
    return false;
  const u8 b = code[0];
  m.mod = (b >> 6) & 3;
  m.reg = static_cast<u8>(((b >> 3) & 7) | (p.rex_r ? 8 : 0));
  const u8 rm = b & 7;
  usize off = 1;

  if (m.mod == 3) {
    m.is_reg = true;
    m.rm = static_cast<u8>(rm | (p.rex_b ? 8 : 0));
    m.length = 1;
    return true;
  }

  bool disp32_no_base = false;
  if (rm == 4) {
    if (avail < off + 1)
      return false;
    const u8 sib = code[off++];
    m.scale = (sib >> 6) & 3;
    const u8 idx = static_cast<u8>(((sib >> 3) & 7) | (p.rex_x ? 8 : 0));
    if (idx != 4) // 100 without REX.X: no index (r12 *can* be an index)
      m.index = static_cast<i8>(idx);
    const u8 base = sib & 7;
    if (m.mod == 0 && base == 5) {
      disp32_no_base = true;
      m.sib_nobase_with_rex_b = p.rex_b;
    } else {
      m.base = static_cast<i8>(base | (p.rex_b ? 8 : 0));
    }
  } else if (m.mod == 0 && rm == 5) {
    m.rip_relative = true;
  } else {
    m.base = static_cast<i8>(rm | (p.rex_b ? 8 : 0));
  }

  if (m.mod == 1) {
    if (avail < off + 1)
      return false;
    m.disp = static_cast<i8>(code[off]);
    off += 1;
  } else if (m.mod == 2 || m.rip_relative || disp32_no_base) {
    if (avail < off + 4)
      return false;
    std::memcpy(&m.disp, code + off, 4);
    off += 4;
  }
  m.length = static_cast<u8>(off);
  return true;
}

/// Read a little-endian immediate of `size` bytes (1, 2, 4 or 8),
/// sign-extended to 64 bits.
inline bool read_imm_s(const u8 *code, usize avail, usize size, i64 &out) {
  if (avail < size)
    return false;
  switch (size) {
  case 1:
    out = static_cast<i8>(code[0]);
    return true;
  case 2: {
    i16 v;
    std::memcpy(&v, code, 2);
    out = v;
    return true;
  }
  case 4: {
    i32 v;
    std::memcpy(&v, code, 4);
    out = v;
    return true;
  }
  case 8:
    std::memcpy(&out, code, 8);
    return true;
  default:
    return false;
  }
}

} // namespace ps4::x86
