#include "ps4/jit_compiler.hpp"
#include "arm64_simd.hpp"

#include <atomic>
#include "ps4/jit_stats.hpp"
#include "ps4/x86_decode.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <iostream>
#if defined(__APPLE__)
#include <libkern/OSCacheControl.h>
#else
// Non-Apple hosts (Linux/aarch64 under QEMU for the CPU test harness): no
// MAP_JIT or per-thread W^X toggle; flush the icache with the builtin.
#ifndef MAP_JIT
#define MAP_JIT 0
#endif
static inline void pthread_jit_write_protect_np(int) {}
static inline void sys_icache_invalidate(void *start, size_t len) {
  __builtin___clear_cache(static_cast<char *>(start),
                          static_cast<char *>(start) + len);
}
#endif
#include <pthread.h>
#include <sys/mman.h>

namespace ps4 {

// =============================================================================
// Arm64Emitter Implementation
// =============================================================================

void Arm64Emitter::nop() { emit(0xd503201f); }

void Arm64Emitter::ret() { emit(0xd65f03c0); }

void Arm64Emitter::mov_reg(u8 rd, u8 rm) {
  // ORR rd, xzr, rm
  emit(0xaa0003e0 | (static_cast<u32>(rm) << 16) | rd);
}

void Arm64Emitter::mov_imm64(u8 rd, u64 val) {
  u16 chunk[4];
  int zero_chunks = 0, ones_chunks = 0;
  for (int i = 0; i < 4; ++i) {
    chunk[i] = static_cast<u16>(val >> (16 * i));
    zero_chunks += (chunk[i] == 0);
    ones_chunks += (chunk[i] == 0xFFFF);
  }
  // Start from MOVN when most halfwords are 0xFFFF (negative immediates),
  // otherwise from MOVZ; MOVK patches the remaining halfwords.
  const bool inverted = ones_chunks > zero_chunks;
  const u16 fill = inverted ? 0xFFFF : 0x0000;
  bool first = true;
  for (u32 i = 0; i < 4; ++i) {
    if (chunk[i] == fill && !(i == 3 && first))
      continue;
    if (first) {
      if (inverted) // movn rd, ~chunk, lsl 16*i
        emit(0x92800000 | (i << 21) |
             (static_cast<u32>(static_cast<u16>(~chunk[i])) << 5) | rd);
      else // movz rd, chunk, lsl 16*i
        emit(0xd2800000 | (i << 21) | (static_cast<u32>(chunk[i]) << 5) | rd);
      first = false;
    } else {
      // movk rd, chunk, lsl 16*i
      emit(0xf2800000 | (i << 21) | (static_cast<u32>(chunk[i]) << 5) | rd);
    }
  }
}

void Arm64Emitter::str_reg(u8 rt, u8 rn, u32 offset_bytes) {
  const u32 imm12 = (offset_bytes / 8) & 0xFFF;
  emit(0xf9000000 | (imm12 << 10) | (static_cast<u32>(rn) << 5) | rt);
}

void Arm64Emitter::ldr_reg(u8 rt, u8 rn, u32 offset_bytes) {
  const u32 imm12 = (offset_bytes / 8) & 0xFFF;
  emit(0xf9400000 | (imm12 << 10) | (static_cast<u32>(rn) << 5) | rt);
}

void Arm64Emitter::str_w(u8 rt, u8 rn, u32 offset_bytes) {
  const u32 imm12 = (offset_bytes / 4) & 0xFFF;
  emit(0xb9000000 | (imm12 << 10) | (static_cast<u32>(rn) << 5) | rt);
}

void Arm64Emitter::ldr_w(u8 rt, u8 rn, u32 offset_bytes) {
  const u32 imm12 = (offset_bytes / 4) & 0xFFF;
  emit(0xb9400000 | (imm12 << 10) | (static_cast<u32>(rn) << 5) | rt);
}

void Arm64Emitter::str_zero(u8 rn, u32 offset_bytes) {
  str_reg(31, rn, offset_bytes); // xzr = 31
}

void Arm64Emitter::add_reg(u8 rd, u8 rn, u8 rm) {
  emit(0x8b000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) |
       rd);
}

void Arm64Emitter::sub_reg(u8 rd, u8 rn, u8 rm) {
  emit(0xcb000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) |
       rd);
}

void Arm64Emitter::adds_reg(u8 rd, u8 rn, u8 rm) {
  emit(0xab000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) |
       rd);
}

void Arm64Emitter::subs_reg(u8 rd, u8 rn, u8 rm) {
  emit(0xeb000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) |
       rd);
}

void Arm64Emitter::b(i32 offset_words) {
  emit(0x14000000 | (static_cast<u32>(offset_words) & 0x03ffffff));
}

void Arm64Emitter::b_cond(u8 cond, i32 offset_words) {
  emit(0x54000000 | ((static_cast<u32>(offset_words) & 0x7ffff) << 5) |
       (cond & 0xf));
}

void Arm64Emitter::add_imm(u8 rd, u8 rn, u32 imm12) {
  emit(0x91000000 | ((imm12 & 0xFFF) << 10) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::sub_imm(u8 rd, u8 rn, u32 imm12) {
  emit(0xd1000000 | ((imm12 & 0xFFF) << 10) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::and_reg(u8 rd, u8 rn, u8 rm) {
  emit(0x8a000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) |
       rd);
}

void Arm64Emitter::orr_reg(u8 rd, u8 rn, u8 rm) {
  emit(0xaa000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) |
       rd);
}

void Arm64Emitter::eor_reg(u8 rd, u8 rn, u8 rm) {
  emit(0xca000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) |
       rd);
}

// --- Data processing (shifted register) ------------------------------------
// sf | opc | 01010/01011 | shift | N | Rm | imm6 | Rn | Rd

namespace {
inline u32 dp_shifted(u32 base, bool sf, u8 rd, u8 rn, u8 rm, u8 shift,
                      u8 amount) {
  return base | (sf ? 0x80000000u : 0u) | (static_cast<u32>(shift & 3) << 22) |
         (static_cast<u32>(rm & 31) << 16) |
         (static_cast<u32>(amount & 63) << 10) |
         (static_cast<u32>(rn & 31) << 5) | (rd & 31);
}
} // namespace

void Arm64Emitter::add_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift,
                               u8 amount) {
  emit(dp_shifted(0x0b000000, sf, rd, rn, rm, shift, amount));
}

void Arm64Emitter::adds_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift,
                                u8 amount) {
  emit(dp_shifted(0x2b000000, sf, rd, rn, rm, shift, amount));
}

void Arm64Emitter::subs_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift,
                                u8 amount) {
  emit(dp_shifted(0x6b000000, sf, rd, rn, rm, shift, amount));
}

void Arm64Emitter::ands_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift,
                                u8 amount) {
  emit(dp_shifted(0x6a000000, sf, rd, rn, rm, shift, amount));
}

void Arm64Emitter::orr_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift,
                               u8 amount) {
  emit(dp_shifted(0x2a000000, sf, rd, rn, rm, shift, amount));
}

void Arm64Emitter::eor_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift,
                               u8 amount) {
  emit(dp_shifted(0x4a000000, sf, rd, rn, rm, shift, amount));
}

void Arm64Emitter::bic_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift,
                               u8 amount) {
  emit(dp_shifted(0x0a200000, sf, rd, rn, rm, shift, amount));
}

void Arm64Emitter::add_w(u8 rd, u8 rn, u8 rm) {
  // ADD Wd, Wn, Wm (32-bit, zero-extends result into 64-bit Xd)
  emit(0x0b000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::sub_w(u8 rd, u8 rn, u8 rm) {
  // SUB Wd, Wn, Wm (32-bit, zero-extends result into 64-bit Xd)
  emit(0x4b000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::blr(u8 rn) {
  // BLR Xn (indirect branch with link)
  emit(0xd63f0000 | (static_cast<u32>(rn) << 5));
}

void Arm64Emitter::mov_w(u8 rd, u8 rm) {
  // ORR wd, wzr, wm
  emit(0x2a0003e0 | (static_cast<u32>(rm) << 16) | rd);
}

// --- Bitfield ------------------------------------------------------------------
// sf | opc | 100110 | N(=sf) | immr | imms | Rn | Rd

void Arm64Emitter::ubfm(bool sf, u8 rd, u8 rn, u8 immr, u8 imms) {
  emit((sf ? 0xd3400000u : 0x53000000u) | (static_cast<u32>(immr & 63) << 16) |
       (static_cast<u32>(imms & 63) << 10) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::sbfm(bool sf, u8 rd, u8 rn, u8 immr, u8 imms) {
  emit((sf ? 0x93400000u : 0x13000000u) | (static_cast<u32>(immr & 63) << 16) |
       (static_cast<u32>(imms & 63) << 10) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::bfm(bool sf, u8 rd, u8 rn, u8 immr, u8 imms) {
  emit((sf ? 0xb3400000u : 0x33000000u) | (static_cast<u32>(immr & 63) << 16) |
       (static_cast<u32>(imms & 63) << 10) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::lsl_imm(bool sf, u8 rd, u8 rn, u8 shift) {
  const u8 size = sf ? 64 : 32;
  ubfm(sf, rd, rn, static_cast<u8>((size - shift) % size),
       static_cast<u8>(size - 1 - shift));
}

void Arm64Emitter::lsr_imm(bool sf, u8 rd, u8 rn, u8 shift) {
  ubfm(sf, rd, rn, shift, sf ? 63 : 31);
}

void Arm64Emitter::asr_imm(bool sf, u8 rd, u8 rn, u8 shift) {
  sbfm(sf, rd, rn, shift, sf ? 63 : 31);
}

void Arm64Emitter::ubfx(bool sf, u8 rd, u8 rn, u8 lsb, u8 width) {
  ubfm(sf, rd, rn, lsb, static_cast<u8>(lsb + width - 1));
}

void Arm64Emitter::bfi(bool sf, u8 rd, u8 rn, u8 lsb, u8 width) {
  const u8 size = sf ? 64 : 32;
  bfm(sf, rd, rn, static_cast<u8>((size - lsb) % size),
      static_cast<u8>(width - 1));
}

void Arm64Emitter::sxt(bool sf, u8 rd, u8 rn, u8 from_bits) {
  sbfm(sf, rd, rn, 0, static_cast<u8>(from_bits - 1));
}

// --- Misc ------------------------------------------------------------------------

void Arm64Emitter::lsrv(bool sf, u8 rd, u8 rn, u8 rm) {
  emit((sf ? 0x9ac02400u : 0x1ac02400u) | (static_cast<u32>(rm) << 16) |
       (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::movz(bool sf, u8 rd, u16 imm16, u8 hw) {
  emit((sf ? 0xd2800000u : 0x52800000u) | (static_cast<u32>(hw & 3) << 21) |
       (static_cast<u32>(imm16) << 5) | rd);
}

void Arm64Emitter::cset(bool sf, u8 rd, u8 cond) {
  // CSINC rd, zr, zr, invert(cond)
  emit((sf ? 0x9a800400u : 0x1a800400u) | (31u << 16) |
       (static_cast<u32>((cond ^ 1) & 0xf) << 12) | (31u << 5) | rd);
}

void Arm64Emitter::mrs_nzcv(u8 rt) { emit(0xd53b4200 | rt); }

void Arm64Emitter::add_imm64(u8 rd, u8 rn, i64 imm, u8 scratch) {
  const bool neg = imm < 0;
  const u64 mag = neg ? (0ULL - static_cast<u64>(imm)) : static_cast<u64>(imm);
  if (mag == 0) {
    if (rd != rn)
      add_imm(rd, rn, 0);
    return;
  }
  if (mag < (1ULL << 24)) {
    // add/sub (immediate), optionally twice: imm12 and imm12 << 12
    const u32 lo = static_cast<u32>(mag & 0xFFF);
    const u32 hi = static_cast<u32>(mag >> 12);
    u8 src = rn;
    if (hi != 0) {
      emit((neg ? 0xd1400000u : 0x91400000u) | (hi << 10) |
           (static_cast<u32>(src) << 5) | rd);
      src = rd;
    }
    if (lo != 0 || hi == 0) {
      emit((neg ? 0xd1000000u : 0x91000000u) | (lo << 10) |
           (static_cast<u32>(src) << 5) | rd);
    }
    return;
  }
  mov_imm64(scratch, static_cast<u64>(imm));
  add_reg(rd, rn, scratch);
}

// --- Loads / stores ----------------------------------------------------------------

void Arm64Emitter::ldrb_imm(u8 rt, u8 rn, u32 offset_bytes) {
  emit(0x39400000 | ((offset_bytes & 0xFFF) << 10) |
       (static_cast<u32>(rn) << 5) | rt);
}

void Arm64Emitter::ldrh_imm(u8 rt, u8 rn, u32 offset_bytes) {
  emit(0x79400000 | (((offset_bytes / 2) & 0xFFF) << 10) |
       (static_cast<u32>(rn) << 5) | rt);
}

void Arm64Emitter::strb_imm(u8 rt, u8 rn, u32 offset_bytes) {
  emit(0x39000000 | ((offset_bytes & 0xFFF) << 10) |
       (static_cast<u32>(rn) << 5) | rt);
}

void Arm64Emitter::strh_imm(u8 rt, u8 rn, u32 offset_bytes) {
  emit(0x79000000 | (((offset_bytes / 2) & 0xFFF) << 10) |
       (static_cast<u32>(rn) << 5) | rt);
}

// Register offset: size | 111000 | opc | 1 | Rm | option=011 (LSL) | S=0 | 10
namespace {
inline u32 ldst_idx(u32 base, u8 rt, u8 rn, u8 rm) {
  return base | 0x00206800u | (static_cast<u32>(rm) << 16) |
         (static_cast<u32>(rn) << 5) | rt;
}
} // namespace

void Arm64Emitter::ldrb_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0x38400000, rt, rn, rm));
}
void Arm64Emitter::ldrh_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0x78400000, rt, rn, rm));
}
void Arm64Emitter::ldr_w_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0xb8400000, rt, rn, rm));
}
void Arm64Emitter::ldr_x_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0xf8400000, rt, rn, rm));
}
void Arm64Emitter::ldrsb_x_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0x38800000, rt, rn, rm));
}
void Arm64Emitter::ldrsh_x_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0x78800000, rt, rn, rm));
}
void Arm64Emitter::ldrsw_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0xb8800000, rt, rn, rm));
}
void Arm64Emitter::strb_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0x38000000, rt, rn, rm));
}
void Arm64Emitter::strh_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0x78000000, rt, rn, rm));
}
void Arm64Emitter::str_w_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0xb8000000, rt, rn, rm));
}
void Arm64Emitter::str_x_idx(u8 rt, u8 rn, u8 rm) {
  emit(ldst_idx(0xf8000000, rt, rn, rm));
}

void Arm64Emitter::patch_b_cond(usize at, usize target) {
  const i32 offset_words = static_cast<i32>(static_cast<isize>(target) -
                                            static_cast<isize>(at));
  m_code[at] = (m_code[at] & ~(0x7ffffu << 5)) |
               ((static_cast<u32>(offset_words) & 0x7ffff) << 5);
}

void Arm64Emitter::emit_prologue() {
  // stp x19, x30, [sp, #-32]!
  emit(0xa9be7bf3);
  // stp x20, x21, [sp, #16]
  emit(0xa90157f4);
  // mov x19, x0 (guest CpuContext*)
  emit(0xaa0003f3);
  // mov x20, x1 (host base of guest memory)
  emit(0xaa0103f4);
  // sub x21, x2, #8 (highest valid start address of an 8-byte access)
  emit(0xd1002055);
}

void Arm64Emitter::emit_epilogue() {
  // ldp x20, x21, [sp, #16]
  emit(0xa94157f4);
  // ldp x19, x30, [sp], #32
  emit(0xa8c27bf3);
  // ret
  emit(0xd65f03c0);
}

// =============================================================================
// JitCompiler Implementation
// =============================================================================

namespace {

u32 get_gpr_offset(u8 x86_reg_idx) {
  switch (x86_reg_idx) {
  case 0:
    return offsetof(CpuContext, rax);
  case 1:
    return offsetof(CpuContext, rcx);
  case 2:
    return offsetof(CpuContext, rdx);
  case 3:
    return offsetof(CpuContext, rbx);
  case 4:
    return offsetof(CpuContext, rsp);
  case 5:
    return offsetof(CpuContext, rbp);
  case 6:
    return offsetof(CpuContext, rsi);
  case 7:
    return offsetof(CpuContext, rdi);
  case 8:
    return offsetof(CpuContext, r8);
  case 9:
    return offsetof(CpuContext, r9);
  case 10:
    return offsetof(CpuContext, r10);
  case 11:
    return offsetof(CpuContext, r11);
  case 12:
    return offsetof(CpuContext, r12);
  case 13:
    return offsetof(CpuContext, r13);
  case 14:
    return offsetof(CpuContext, r14);
  case 15:
    return offsetof(CpuContext, r15);
  default:
    return offsetof(CpuContext, rax);
  }
}

// Telemetry (PS4_JIT_STATS): attribute the block end to the instruction that
// caused it. Terminators (incl. JMP, which is compiled) end with block_ended
// set and are at insn_rip; otherwise the decoder bailed at current_rip, or the
// block hit the instruction cap.
void record_block_end_stats(const MemoryManager &memory, bool block_ended,
                            GuestAddr insn_rip, GuestAddr current_rip,
                            usize instructions, usize max_instructions) {
  if (block_ended) {
    jit_stats_record_block_end(memory, insn_rip, JitStopReason::Terminator,
                               instructions);
  } else if (instructions >= max_instructions) {
    jit_stats_record_block_end(memory, current_rip, JitStopReason::Limit,
                               instructions);
  } else {
    jit_stats_record_block_end(memory, current_rip, JitStopReason::Unsupported,
                               instructions);
  }
}

// -----------------------------------------------------------------------------
// Block translator
//
// Register usage inside a compiled block:
//   x19 = CpuContext*   x20 = guest memory base   x21 = guest size - 8
//   x0/x1 = operands / result   x2-x5 = flag temps
//   x9 = guest effective address   x10/x11 = address temps
// Guest registers live in CpuContext; every instruction loads what it needs
// and stores its results back, so a side exit never has state to flush.
// -----------------------------------------------------------------------------

constexpr u8 R_CTX = 19;
constexpr u8 R_GBASE = 20;
constexpr u8 R_GLIMIT = 21;
constexpr u8 R_EA = 9;

constexpr u8 COND_CS = 2; // C set   (x86 CF after ADD)
constexpr u8 COND_CC = 3; // C clear (x86 CF after SUB/CMP: borrow)
constexpr u8 COND_HI = 8;

// The arithmetic flags the interpreter computes. AF is never written by the
// interpreter for these instructions, so it is left untouched here as well.
constexpr u64 FLAGS_ARITH = CpuContext::FLAG_CF | CpuContext::FLAG_PF |
                            CpuContext::FLAG_ZF | CpuContext::FLAG_SF |
                            CpuContext::FLAG_OF;

// x86 ALU group numbering (opcode bits 5:3 / ModRM.reg of 80-83)
enum : u8 {
  ALU_ADD = 0,
  ALU_OR = 1,
  ALU_AND = 4,
  ALU_SUB = 5,
  ALU_XOR = 6,
  ALU_CMP = 7
};

enum class Kind : u8 {
  Nop,
  MovRegImm, // B0+r / B8+r
  MovRmReg,  // 88 / 89
  MovRegRm,  // 8A / 8B
  MovRmImm,  // C6 /0, C7 /0
  Lea,       // 8D
  Extend,    // 0F B6/B7/BE/BF, 63
  Alu,       // 00-3B (ADD/OR/AND/SUB/XOR/CMP), 84/85 TEST
  AluImm,    // 80/81/83, 04/05.., A8/A9, F6/F7 /0
  IncDec,    // FE/FF /0 /1
  ShiftImm,  // C0/C1 /4 /5 /6 /7 imm8, D0/D1 (count 1)
  Push,      // 50+r
  PushImm,   // 6A / 68
  PushRm,    // FF /6
  Pop,       // 58+r
  Jmp,       // EB / E9
  Jcc,       // 70-7F rel8, 0F 80-8F rel32 (block end; reg = condition code)
  Cmov,      // 0F 40-4F (reg = condition code)
  Setcc,     // 0F 90-9F (reg = condition code)
  Vec,       // SSE / AVX (vop selects the operation)
  CallRel,   // E8 rel32
  Ret,       // C3 / C2 imm16 (imm = extra bytes popped)
  CallInd,   // FF /2
  JmpInd,    // FF /4
};

/// Control transfers that end the block and store the next RIP themselves.
bool writes_rip(Kind k) {
  return k == Kind::CallRel || k == Kind::Ret || k == Kind::CallInd || k == Kind::JmpInd;
}

/// CALL/RET/indirect JMP are translated unless PS4_JIT_NO_CALLS is set; the
/// interpreter then executes them and keeps its control-transfer history
/// (dump_control_xfers) complete for crash reports.
bool jit_translates_calls() {
  static const bool on = std::getenv("PS4_JIT_NO_CALLS") == nullptr;
  return on;
}

// SSE/AVX operations. Operands: dest = ModRM.reg, src1 = VEX.vvvv (legacy
// SSE: dest), src2 / rm = ModRM.rm or memory. VEX.128 writes zero bits
// 255:128 of the destination; legacy SSE writes leave them alone.
enum class VecOp : u8 {
  Load,        // dest = rm: movups/movaps/movupd/movapd/movdqu/movdqa
  Store,       // rm = dest register: the 11/29/7F store forms
  LoadScalar,  // movss/movsd xmm, xmm/m (F3/F2 10)
  StoreScalar, // movss/movsd xmm/m, xmm (F3/F2 11)
  Movq,        // movq xmm, xmm/m64 (F3 7E): zero-extended low qword
  MovqStore,   // movq xmm/m64, xmm (66 D6)
  MovdToVec,   // movd/movq xmm, r/m (66 6E)
  MovdFromVec, // movd/movq r/m, xmm (66 7E)
  LoadLow,     // movlps/movlpd xmm, m64 (12 mem)
  LoadHigh,    // movhps/movhpd xmm, m64 (16 mem)
  Movhlps,     // 0F 12 reg
  Movlhps,     // 0F 16 reg
  StoreLow,    // movlps/movlpd m64, xmm (13)
  StoreHigh,   // movhps/movhpd m64, xmm (17)
  Moveldup,    // movsldup (F3 12)
  Movehdup,    // movshdup (F3 16)
  Movddup,     // movddup (F2 12)
  Packed,      // vbin on every lane, per 128-bit half
  Scalar,      // vbin on lane 0; the other lanes come from src1
  Ucomi,       // (u)comiss / (u)comisd -> ZF, PF, CF
  Shuf,        // shufps / shufpd imm8
  Pshufd,      // pshufd / vpermilps imm8 (one source)
  Blend,       // blendps / blendpd imm8
  Blendv,      // vblendvps / vblendvpd (mask register in imm8[7:4])
  Insertps,
  Broadcast,   // vbroadcastss
  Pextr,       // pextrd / pextrq
  Pinsr,       // pinsrd / pinsrq
  CvtF2F,      // cvtss2sd / cvtsd2ss
  CvtI2F,      // cvtsi2ss / cvtsi2sd
  CvttF2I,     // cvttss2si / cvttsd2si
};

enum class VecBin : u8 {
  Add, Sub, Mul, Div, Min, Max, Sqrt, And, Andn, Or, Xor,
  UnpckLo, UnpckHi, AddSub, IAdd, ISub, ICmpEq, Cmp,
};

struct JitInsn {
  GuestAddr rip{0};
  u8 len{0};
  Kind kind{Kind::Nop};
  x86::Prefixes p{};
  x86::ModRM m{};  // valid when the opcode has a ModRM byte
  u8 width{0};     // operand size in bits
  u8 src_width{0}; // Extend: source size
  bool sign_extend{false};
  u8 alu_op{0}; // ALU_* ; IncDec: ALU_ADD (inc) / ALU_SUB (dec)
  bool rm_is_dst{false};
  bool writes_result{true}; // false for CMP / TEST
  u8 reg{0};                // register operand of opcode-embedded forms
  u64 imm{0};
  GuestAddr target{0};
  u64 flags_written{0};
  u64 flags_read{0}; // must be in ctx->rflags before this instruction

  // Kind::Vec
  VecOp vop{VecOp::Load};
  VecBin vbin{VecBin::Add};
  bool vex{false};
  bool l256{false}; // VEX.L: 256-bit operands
  bool dbl{false};  // double / 64-bit lanes (pd, sd; integer qword forms)
  bool w64{false};  // REX.W / VEX.W: 64-bit general register operand
  u8 vvvv{0};       // VEX.vvvv

  [[nodiscard]] bool has_mem() const noexcept {
    switch (kind) {
    case Kind::MovRmReg:
    case Kind::MovRegRm:
    case Kind::MovRmImm:
    case Kind::Extend:
    case Kind::Alu:
    case Kind::AluImm:
    case Kind::IncDec:
    case Kind::ShiftImm:
    case Kind::PushRm:
    case Kind::Cmov:
    case Kind::Setcc:
    case Kind::CallInd:
    case Kind::JmpInd:
      return !m.is_reg;
    default:
      return false;
    }
  }
};

u64 truncate(u64 v, u8 width) {
  return width >= 64 ? v : (v & ((1ULL << width) - 1));
}

/// x2 = 1 if x86 condition `cc` holds for the flags in ctx->rflags, else 0.
/// Same predicate as the interpreter's check_condition(). Clobbers x1, x3.
void emit_cond(Arm64Emitter &e, u8 cc) {
  constexpr u8 CF = 0, PF = 2, ZF = 6, SF = 7, OF = 11;
  e.ldr_reg(1, 19, offsetof(CpuContext, rflags));
  auto bit = [&](u8 rd, u8 lsb) { e.ubfx(true, rd, 1, lsb, 1); };
  switch ((cc >> 1) & 7) {
  case 0: bit(2, OF); break;
  case 1: bit(2, CF); break;
  case 2: bit(2, ZF); break;
  case 3: bit(2, CF); bit(3, ZF); e.orr_reg(2, 2, 3); break;
  case 4: bit(2, SF); break;
  case 5: bit(2, PF); break;
  case 6: bit(2, SF); bit(3, OF); e.eor_reg(2, 2, 3); break;
  case 7:
    bit(2, SF); bit(3, OF); e.eor_reg(2, 2, 3);
    bit(3, ZF); e.orr_reg(2, 2, 3);
    break;
  }
  if (cc & 1) {
    e.mov_imm64(3, 1);
    e.eor_reg(2, 2, 3);
  }
}

/// x0 = next RIP after a block-final Jcc, branch-free:
/// x0 = fallthrough + (-cond & (target - fallthrough)). Clobbers x1-x4.
void emit_jcc_rip(Arm64Emitter &e, const JitInsn &d) {
  emit_cond(e, d.reg);
  const GuestAddr fallthrough = d.rip + d.len;
  e.sub_reg(2, 31, 2); // all ones when taken
  e.mov_imm64(3, d.target - fallthrough);
  e.and_reg(3, 3, 2);
  e.mov_imm64(4, fallthrough);
  e.add_reg(0, 4, 3);
}

/// Control transfers and traps: the interpreter executes them, and the block
/// ends in front of them without that counting as "unsupported".
bool is_block_terminator(const u8 *code, usize avail, const x86::Prefixes &p) {
  const usize c = p.length;
  const u8 op0 = code[c];
  const u8 op1 = (c + 1 < avail) ? code[c + 1] : 0;
  if (op0 == 0xF4 || op0 == 0xCC)
    return true;
  if ((op0 == 0xC3 || op0 == 0xC2 || op0 == 0xE8) && !jit_translates_calls())
    return true;
  if (op0 == 0x0F && (op1 == 0x05 || op1 == 0x0B))
    return true;
  return false;
}

/// Opcodes routed to decode_vec (map 1 = 0F, 2 = 0F 38, 3 = 0F 3A). None of
/// them is handled by the integer decoder.
bool is_vec_opcode(u8 map, u8 op) {
  if (map == 1) {
    switch (op) {
    case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16: case 0x17:
    case 0x28: case 0x29: case 0x2A: case 0x2C: case 0x2E: case 0x2F:
    case 0x51: case 0x54: case 0x55: case 0x56: case 0x57: case 0x58: case 0x59: case 0x5A:
    case 0x5C: case 0x5D: case 0x5E: case 0x5F:
    case 0x62: case 0x6A: case 0x6C: case 0x6D: case 0x6E: case 0x6F:
    case 0x70: case 0x76: case 0x7E: case 0x7F:
    case 0xC2: case 0xC6: case 0xD0: case 0xD4: case 0xD6: case 0xDB: case 0xDF:
    case 0xEB: case 0xEF: case 0xFA: case 0xFB: case 0xFE:
      return true;
    default:
      return false;
    }
  }
  if (map == 2)
    return op == 0x18;
  return op == 0x04 || op == 0x0C || op == 0x0D || op == 0x16 || op == 0x21 ||
         op == 0x22 || op == 0x4A || op == 0x4B;
}

/// Decode an SSE/AVX instruction whose opcode byte is at code[pos] (after
/// 0F [38|3A] or the VEX prefix). pp: 0 = none, 1 = 66, 2 = F3, 3 = F2.
/// The forms not listed are left to the interpreter.
bool decode_vec(const u8 *code, usize avail, usize pos, GuestAddr rip, u8 map,
                u8 pp, JitInsn &d) {
  if (pos >= avail)
    return false;
  const u8 op = code[pos++];
  if (!x86::decode_modrm(code + pos, avail - pos, d.p, d.m))
    return false;
  pos += d.m.length;
  const bool mem = !d.m.is_reg;
  const bool vex = d.vex;
  d.kind = Kind::Vec;
  d.w64 = d.p.rex_w;
  d.dbl = pp == 1 || pp == 3; // pd / sd
  bool imm8 = false;
  bool wide_ok = false; // VEX.L = 1 accepted
  auto set = [&](VecOp v) { d.vop = v; return true; };
  auto packed = [&](VecBin b) { d.vbin = b; wide_ok = true; return set(VecOp::Packed); };
  auto scalar = [&](VecBin b) { d.vbin = b; return set(VecOp::Scalar); };
  auto arith = [&](VecBin b) { return pp <= 1 ? packed(b) : scalar(b); };

  bool ok = false;
  if (map == 1) {
    switch (op) {
    case 0x10: case 0x11:
      if (pp <= 1) { wide_ok = true; ok = set(op == 0x10 ? VecOp::Load : VecOp::Store); }
      else ok = set(op == 0x10 ? VecOp::LoadScalar : VecOp::StoreScalar);
      break;
    case 0x12:
      if (pp == 0) ok = set(mem ? VecOp::LoadLow : VecOp::Movhlps);
      else if (pp == 1 && mem) ok = set(VecOp::LoadLow);
      else if (pp == 2) { wide_ok = true; ok = set(VecOp::Moveldup); }
      else if (pp == 3) { wide_ok = true; ok = set(VecOp::Movddup); }
      break;
    case 0x16:
      if (pp == 0) ok = set(mem ? VecOp::LoadHigh : VecOp::Movlhps);
      else if (pp == 1 && mem) ok = set(VecOp::LoadHigh);
      else if (pp == 2) { wide_ok = true; ok = set(VecOp::Movehdup); }
      break;
    case 0x13: case 0x17:
      if (pp <= 1 && mem) ok = set(op == 0x13 ? VecOp::StoreLow : VecOp::StoreHigh);
      break;
    case 0x14: case 0x15:
      if (pp <= 1) ok = packed(op == 0x14 ? VecBin::UnpckLo : VecBin::UnpckHi);
      break;
    case 0x28: case 0x29:
      if (pp <= 1) { wide_ok = true; ok = set(op == 0x28 ? VecOp::Load : VecOp::Store); }
      break;
    case 0x2A:
      if (pp >= 2) { d.dbl = pp == 3; ok = set(VecOp::CvtI2F); }
      break;
    case 0x2C:
      if (pp >= 2) { d.dbl = pp == 3; ok = set(VecOp::CvttF2I); }
      break;
    case 0x2E: case 0x2F:
      if (pp <= 1) { d.flags_written = FLAGS_ARITH; ok = set(VecOp::Ucomi); }
      break;
    case 0x51: ok = arith(VecBin::Sqrt); break;
    case 0x54: case 0x55: case 0x56: case 0x57:
      if (pp <= 1) {
        static constexpr VecBin kLogic[4] = {VecBin::And, VecBin::Andn, VecBin::Or, VecBin::Xor};
        ok = packed(kLogic[op - 0x54]);
      }
      break;
    case 0x58: ok = arith(VecBin::Add); break;
    case 0x59: ok = arith(VecBin::Mul); break;
    case 0x5C: ok = arith(VecBin::Sub); break;
    case 0x5D: ok = arith(VecBin::Min); break;
    case 0x5E: ok = arith(VecBin::Div); break;
    case 0x5F: ok = arith(VecBin::Max); break;
    case 0x5A:
      if (pp >= 2) ok = set(VecOp::CvtF2F); // dbl: source is double
      break;
    case 0x62: case 0x6A: case 0x6C: case 0x6D: // punpck{l,h}{dq,qdq}
      if (pp == 1) {
        d.dbl = op >= 0x6C;
        ok = packed((op & 0x0F) == 0x02 || op == 0x6C ? VecBin::UnpckLo : VecBin::UnpckHi);
      }
      break;
    case 0x6E: if (pp == 1) ok = set(VecOp::MovdToVec); break;
    case 0x7E:
      if (pp == 1) ok = set(VecOp::MovdFromVec);
      else if (pp == 2) ok = set(VecOp::Movq);
      break;
    case 0x6F: case 0x7F:
      if (pp == 1 || pp == 2) { wide_ok = true; ok = set(op == 0x6F ? VecOp::Load : VecOp::Store); }
      break;
    case 0x70:
      if (pp == 1) { imm8 = wide_ok = true; ok = set(VecOp::Pshufd); }
      break;
    case 0x76: if (pp == 1) { d.dbl = false; ok = packed(VecBin::ICmpEq); } break;
    case 0xC2: imm8 = true; ok = arith(VecBin::Cmp); break;
    case 0xC6:
      if (pp <= 1) { imm8 = wide_ok = true; ok = set(VecOp::Shuf); }
      break;
    case 0xD0: // addsubpd (66) / addsubps (F2)
      if (pp == 1 || pp == 3) { d.dbl = pp == 1; ok = packed(VecBin::AddSub); }
      break;
    case 0xD4: case 0xFE: case 0xFA: case 0xFB:
      if (pp == 1) {
        d.dbl = op == 0xD4 || op == 0xFB;
        ok = packed(op == 0xD4 || op == 0xFE ? VecBin::IAdd : VecBin::ISub);
      }
      break;
    case 0xD6: if (pp == 1) ok = set(VecOp::MovqStore); break;
    case 0xDB: case 0xDF: case 0xEB: case 0xEF:
      if (pp == 1) {
        ok = packed(op == 0xDB ? VecBin::And : op == 0xDF ? VecBin::Andn
                    : op == 0xEB ? VecBin::Or : VecBin::Xor);
      }
      break;
    default: break;
    }
  } else if (map == 2) {
    if (op == 0x18 && vex && pp == 1 && !d.w64) { wide_ok = true; ok = set(VecOp::Broadcast); }
  } else {
    imm8 = true;
    if (pp == 1) {
      switch (op) {
      case 0x04: if (vex && !d.w64) { wide_ok = true; ok = set(VecOp::Pshufd); } break;
      case 0x0C: case 0x0D: d.dbl = op == 0x0D; wide_ok = true; ok = set(VecOp::Blend); break;
      case 0x16: d.dbl = d.w64; ok = set(VecOp::Pextr); break;
      case 0x21: ok = set(VecOp::Insertps); break;
      case 0x22: d.dbl = d.w64; ok = set(VecOp::Pinsr); break;
      case 0x4A: case 0x4B:
        if (vex && !d.w64) { d.dbl = op == 0x4B; wide_ok = true; ok = set(VecOp::Blendv); }
        break;
      default: break;
      }
    }
  }
  if (!ok || (d.l256 && !wide_ok))
    return false;
  if (imm8) {
    if (pos >= avail)
      return false;
    d.imm = code[pos++];
  }
  // Compare predicates beyond the eight SSE ones (VEX imm 8-31) stay in the
  // interpreter.
  if ((d.vop == VecOp::Packed || d.vop == VecOp::Scalar) && d.vbin == VecBin::Cmp && d.imm >= 8)
    return false;
  d.rip = rip;
  d.len = static_cast<u8>(pos);
  return true;
}

/// Decode one instruction into a JitInsn. Returns false when the JIT does not
/// translate it (the block then ends in front of it).
bool decode_insn(const u8 *code, usize avail, GuestAddr rip, JitInsn &d) {
  x86::Prefixes &p = d.p;
  if (!x86::decode_prefixes(code, avail, p))
    return false;
  // LOCK (atomic RMW), address-size override and REP/REPNE-modified forms are
  // left to the interpreter.
  if (p.lock || p.addr32)
    return false;

  usize pos = p.length;
  const u8 op0 = code[pos++];
  const u8 op_bits = p.op_bits();

  // VEX prefix (C4/C5 are never LES/LDS in 64-bit mode). A 66/F2/F3/REX
  // prefix in front of it is #UD: leave that to the interpreter.
  if (op0 == 0xC4 || op0 == 0xC5) {
    if (p.opsize16 || p.rep || p.has_rex)
      return false;
    const usize vex_len = op0 == 0xC5 ? 1 : 2;
    if (pos + vex_len >= avail)
      return false;
    const u8 b1 = code[pos];
    const u8 last = code[pos + vex_len - 1];
    u8 map = 1;
    p.rex_r = !(b1 & 0x80);
    if (op0 == 0xC4) {
      p.rex_x = !(b1 & 0x40);
      p.rex_b = !(b1 & 0x20);
      map = b1 & 0x1F;
      p.rex_w = (last & 0x80) != 0;
    }
    pos += vex_len;
    d.vex = true;
    d.vvvv = static_cast<u8>((~last >> 3) & 0x0F);
    d.l256 = (last & 4) != 0;
    if (map < 1 || map > 3)
      return false;
    return decode_vec(code, avail, pos, rip, map, last & 3, d);
  }
  // Legacy SSE: 0F xx, 0F 38 xx, 0F 3A xx with a 66/F3/F2 mandatory prefix.
  if (op0 == 0x0F && pos < avail) {
    u8 map = 1;
    usize q = pos;
    if (code[q] == 0x38 || code[q] == 0x3A) {
      map = code[q] == 0x38 ? 2 : 3;
      if (++q >= avail)
        return false;
    }
    if (is_vec_opcode(map, code[q])) {
      const u8 pp = p.rep == 0xF3 ? 2 : p.rep == 0xF2 ? 3 : p.opsize16 ? 1 : 0;
      return decode_vec(code, avail, q, rip, map, pp, d);
    }
  }

  if (p.rep && op0 != 0x90) // F3 90 = PAUSE
    return false;

  auto modrm = [&]() -> bool {
    if (!x86::decode_modrm(code + pos, avail - pos, p, d.m))
      return false;
    pos += d.m.length;
    return true;
  };
  auto imm = [&](usize size) -> bool {
    i64 v = 0;
    if (!x86::read_imm_s(code + pos, avail - pos, size, v))
      return false;
    d.imm = static_cast<u64>(v);
    pos += size;
    return true;
  };
  auto imm_for = [](u8 width) -> usize {
    return width == 8 ? 1 : width == 16 ? 2 : 4;
  };
  auto finish = [&]() {
    d.rip = rip;
    d.len = static_cast<u8>(pos);
    return true;
  };

  // CALL rel32, RET [imm16], CALL r/m64, JMP r/m64: block-ending transfers.
  // 66-prefixed (16-bit) forms stay in the interpreter.
  if (jit_translates_calls() && !p.opsize16) {
    if (op0 == 0xE8) {
      if (!imm(4))
        return false;
      d.kind = Kind::CallRel;
      finish();
      d.target = rip + d.len + d.imm;
      return true;
    }
    if (op0 == 0xC3 || op0 == 0xC2) {
      if (op0 == 0xC2) {
        if (!imm(2))
          return false;
        d.imm &= 0xFFFF;
      } else {
        d.imm = 0;
      }
      d.kind = Kind::Ret;
      return finish();
    }
    if (op0 == 0xFF && pos < avail) {
      x86::ModRM peek;
      if (x86::decode_modrm(code + pos, avail - pos, p, peek) &&
          ((peek.reg & 7) == 2 || (peek.reg & 7) == 4)) {
        modrm();
        d.kind = (d.m.reg & 7) == 2 ? Kind::CallInd : Kind::JmpInd;
        d.width = 64;
        return finish();
      }
    }
  }

  // NOP / PAUSE (REX.B 90 is XCHG r8, rAX)
  if (op0 == 0x90 && !p.rex_b) {
    d.kind = Kind::Nop;
    return finish();
  }

  // MOV r8, imm8
  if (op0 >= 0xB0 && op0 <= 0xB7) {
    if (!imm(1))
      return false;
    d.kind = Kind::MovRegImm;
    d.width = 8;
    d.reg = static_cast<u8>((op0 & 7) | (p.rex_b ? 8 : 0));
    return finish();
  }

  // MOV r16/32, imm16/32 ; MOVABS r64, imm64
  if (op0 >= 0xB8 && op0 <= 0xBF) {
    if (!imm(op_bits == 64 ? 8 : imm_for(op_bits)))
      return false;
    d.kind = Kind::MovRegImm;
    d.width = op_bits;
    d.reg = static_cast<u8>((op0 & 7) | (p.rex_b ? 8 : 0));
    return finish();
  }

  // MOV r/m, r ; MOV r, r/m
  if (op0 >= 0x88 && op0 <= 0x8B) {
    if (!modrm())
      return false;
    d.kind = (op0 & 2) ? Kind::MovRegRm : Kind::MovRmReg;
    d.width = (op0 & 1) ? op_bits : 8;
    return finish();
  }

  // MOV r/m8, imm8 ; MOV r/m, imm16/32 (sign-extended for 64-bit)
  if (op0 == 0xC6 || op0 == 0xC7) {
    if (!modrm() || (d.m.reg & 7) != 0)
      return false;
    d.width = (op0 == 0xC6) ? 8 : op_bits;
    if (!imm(imm_for(d.width)))
      return false;
    d.kind = Kind::MovRmImm;
    return finish();
  }

  // LEA. The interpreter adds the FS/GS base to LEA's result when a segment
  // override is present (architecturally LEA ignores it): leave those to it.
  if (op0 == 0x8D) {
    if (!modrm() || d.m.is_reg || p.seg)
      return false;
    d.kind = Kind::Lea;
    d.width = op_bits;
    return finish();
  }

  // MOVSXD r64, r/m32. Without REX.W the interpreter still sign-extends to
  // 64 bits (architecturally a 32-bit move), so only translate REX.W.
  if (op0 == 0x63) {
    if (!p.rex_w || !modrm())
      return false;
    d.kind = Kind::Extend;
    d.width = 64;
    d.src_width = 32;
    d.sign_extend = true;
    return finish();
  }

  // ALU r/m, r and r, r/m: ADD/OR/AND/SUB/XOR/CMP (ADC/SBB not translated)
  if (op0 <= 0x3B && (op0 & 7) <= 3) {
    const u8 major = (op0 >> 3) & 7;
    if (major == 2 || major == 3 || !modrm())
      return false;
    d.kind = Kind::Alu;
    d.alu_op = major;
    d.width = (op0 & 1) ? op_bits : 8;
    d.rm_is_dst = (op0 & 2) == 0;
    d.writes_result = major != ALU_CMP;
    d.flags_written = FLAGS_ARITH;
    return finish();
  }

  // ALU AL/eAX, imm
  if (op0 <= 0x3D && ((op0 & 7) == 4 || (op0 & 7) == 5)) {
    const u8 major = (op0 >> 3) & 7;
    if (major == 2 || major == 3)
      return false;
    d.width = (op0 & 1) ? op_bits : 8;
    if (!imm(imm_for(d.width)))
      return false;
    d.kind = Kind::AluImm;
    d.m.is_reg = true;
    d.m.rm = 0; // rAX
    d.alu_op = major;
    d.writes_result = major != ALU_CMP;
    d.flags_written = FLAGS_ARITH;
    return finish();
  }

  // ALU r/m, imm
  if (op0 == 0x80 || op0 == 0x81 || op0 == 0x83) {
    if (!modrm())
      return false;
    const u8 sub = d.m.reg & 7;
    if (sub == 2 || sub == 3)
      return false;
    d.width = (op0 == 0x80) ? 8 : op_bits;
    if (!imm(op0 == 0x81 ? imm_for(op_bits) : 1))
      return false;
    d.kind = Kind::AluImm;
    d.alu_op = sub;
    d.writes_result = sub != ALU_CMP;
    d.flags_written = FLAGS_ARITH;
    return finish();
  }

  // TEST r/m, r
  if (op0 == 0x84 || op0 == 0x85) {
    if (!modrm())
      return false;
    d.kind = Kind::Alu;
    d.alu_op = ALU_AND;
    d.width = (op0 == 0x85) ? op_bits : 8;
    d.rm_is_dst = true;
    d.writes_result = false;
    d.flags_written = FLAGS_ARITH;
    return finish();
  }

  // TEST AL/eAX, imm
  if (op0 == 0xA8 || op0 == 0xA9) {
    d.width = (op0 == 0xA9) ? op_bits : 8;
    if (!imm(imm_for(d.width)))
      return false;
    d.kind = Kind::AluImm;
    d.m.is_reg = true;
    d.m.rm = 0;
    d.alu_op = ALU_AND;
    d.writes_result = false;
    d.flags_written = FLAGS_ARITH;
    return finish();
  }

  // TEST r/m, imm (F6 /0, F7 /0)
  if (op0 == 0xF6 || op0 == 0xF7) {
    if (!modrm() || (d.m.reg & 7) != 0)
      return false;
    d.width = (op0 == 0xF6) ? 8 : op_bits;
    if (!imm(imm_for(d.width)))
      return false;
    // The interpreter zero-extends imm32 for TEST r/m64, imm32 instead of
    // sign-extending it; only translate when both agree.
    if (d.width == 64 && (d.imm >> 31) != 0)
      return false;
    d.kind = Kind::AluImm;
    d.alu_op = ALU_AND;
    d.writes_result = false;
    d.flags_written = FLAGS_ARITH;
    return finish();
  }

  // SHL/SAL/SHR/SAR r/m, imm8 / 1. The count is known at compile time, so
  // the interpreter's semantics (incl. count 0 = no-op and counts >= the
  // operand size for 8/16-bit) are reproduced exactly. Rotates and shifts by
  // CL are not translated.
  if (op0 == 0xC0 || op0 == 0xC1 || op0 == 0xD0 || op0 == 0xD1) {
    if (!modrm())
      return false;
    const u8 sub = d.m.reg & 7;
    if (sub < 4)
      return false;
    d.width = (op0 & 1) ? op_bits : 8;
    if (op0 <= 0xC1) {
      if (!imm(1))
        return false;
    } else {
      d.imm = 1;
    }
    d.imm &= (d.width == 64) ? 0x3F : 0x1F;
    d.kind = Kind::ShiftImm;
    d.alu_op = (sub == 6) ? 4 : sub; // SAL == SHL
    d.flags_written = d.imm ? FLAGS_ARITH : 0;
    return finish();
  }

  // INC / DEC r/m (CF preserved) ; PUSH r/m64
  if (op0 == 0xFE || op0 == 0xFF) {
    if (!modrm())
      return false;
    const u8 sub = d.m.reg & 7;
    if (op0 == 0xFF && sub == 6) {
      if (p.opsize16) // the interpreter always pushes 8 bytes
        return false;
      d.kind = Kind::PushRm;
      d.width = 64;
      return finish();
    }
    if (sub > 1)
      return false;
    d.kind = Kind::IncDec;
    d.width = (op0 == 0xFE) ? 8 : op_bits;
    d.alu_op = (sub == 0) ? ALU_ADD : ALU_SUB;
    d.flags_written = FLAGS_ARITH & ~CpuContext::FLAG_CF;
    return finish();
  }

  // PUSH r64 / POP r64. 16-bit forms are not translated (the interpreter
  // always moves 8 bytes), nor is PUSH RSP (the interpreter stores the
  // already-decremented RSP).
  if (op0 >= 0x50 && op0 <= 0x5F) {
    if (p.opsize16)
      return false;
    d.reg = static_cast<u8>((op0 & 7) | (p.rex_b ? 8 : 0));
    d.kind = (op0 < 0x58) ? Kind::Push : Kind::Pop;
    if (d.kind == Kind::Push && d.reg == 4)
      return false;
    return finish();
  }

  // PUSH imm8 / imm32 (sign-extended)
  if (op0 == 0x6A || op0 == 0x68) {
    if (p.opsize16 || !imm(op0 == 0x6A ? 1 : 4))
      return false;
    d.kind = Kind::PushImm;
    return finish();
  }

  // JMP rel8 / rel32
  if (op0 == 0xEB || op0 == 0xE9) {
    if (!imm(op0 == 0xEB ? 1 : 4))
      return false;
    d.kind = Kind::Jmp;
    finish();
    d.target = rip + d.len + d.imm;
    return true;
  }

  // Jcc rel8. Reads the flags the block has already stored to ctx->rflags.
  if (op0 >= 0x70 && op0 <= 0x7F) {
    if (!imm(1))
      return false;
    d.kind = Kind::Jcc;
    d.reg = op0 & 0x0F;
    d.flags_read = FLAGS_ARITH;
    finish();
    d.target = rip + d.len + d.imm;
    return true;
  }

  if (op0 == 0x0F && pos < avail) {
    const u8 op1 = code[pos++];

    // Jcc rel32 (66-prefixed rel16 forms are left to the interpreter)
    if (op1 >= 0x80 && op1 <= 0x8F) {
      if (p.opsize16 || !imm(4))
        return false;
      d.kind = Kind::Jcc;
      d.reg = op1 & 0x0F;
      d.flags_read = FLAGS_ARITH;
      finish();
      d.target = rip + d.len + d.imm;
      return true;
    }

    // CMOVcc r32/r64, r/m. 16-bit forms are left to the interpreter.
    if (op1 >= 0x40 && op1 <= 0x4F) {
      if (op_bits == 16 || !modrm())
        return false;
      d.kind = Kind::Cmov;
      d.reg = op1 & 0x0F;
      d.width = op_bits;
      d.flags_read = FLAGS_ARITH;
      return finish();
    }

    // SETcc r/m8
    if (op1 >= 0x90 && op1 <= 0x9F) {
      if (!modrm())
        return false;
      d.kind = Kind::Setcc;
      d.reg = op1 & 0x0F;
      d.width = 8;
      d.flags_read = FLAGS_ARITH;
      return finish();
    }

    // Multi-byte NOP: 0F 1F /0
    if (op1 == 0x1F) {
      if (!modrm())
        return false;
      d.kind = Kind::Nop;
      return finish();
    }

    // MOVZX / MOVSX
    if (op1 == 0xB6 || op1 == 0xB7 || op1 == 0xBE || op1 == 0xBF) {
      const bool word_src = (op1 & 1) != 0;
      // 66-prefixed MOVZX/MOVSX r16, r/m16: the interpreter writes 32 bits.
      if (word_src && op_bits == 16)
        return false;
      if (!modrm())
        return false;
      d.kind = Kind::Extend;
      d.width = op_bits;
      d.src_width = word_src ? 16 : 8;
      d.sign_extend = op1 >= 0xBE;
      return finish();
    }
  }

  return false;
}

class BlockTranslator {
public:
  explicit BlockTranslator(Arm64Emitter &e) : m_e(e) {}

  /// Emit one decoded instruction. `retired` is the number of guest
  /// instructions completed before it (the side-exit return value).
  void emit(const JitInsn &d, u64 flags_needed, u64 retired) {
    m_cur = &d;
    m_retired = retired;
    if (d.has_mem()) {
      emit_ea(d.m, d.rip + d.len, d.p.seg);
      emit_bounds_check();
    }
    const u8 w = d.width;
    switch (d.kind) {
    case Kind::Nop:
    case Kind::Jmp:
    case Kind::Jcc: // the block epilogue selects the next RIP
      break;
    case Kind::Cmov:
      // Branch-free select; a 32-bit CMOV zero-extends even when not taken.
      // The source is read either way, as on hardware.
      emit_cond(m_e, d.reg);
      load_gpr(0, d.m.reg, w);
      load_rm(5, w);
      m_e.eor_reg(6, 0, 5);
      m_e.sub_reg(2, 31, 2);
      m_e.and_reg(6, 6, 2);
      m_e.eor_reg(0, 0, 6);
      store_gpr(d.m.reg, w, 0);
      break;
    case Kind::Setcc:
      emit_cond(m_e, d.reg);
      store_rm(2, 8);
      break;
    case Kind::MovRegImm:
      m_e.mov_imm64(0, truncate(d.imm, w));
      store_gpr(d.reg, w, 0);
      break;
    case Kind::MovRmReg:
      load_gpr(0, d.m.reg, w);
      store_rm(0, w);
      break;
    case Kind::MovRegRm:
      load_rm(0, w);
      store_gpr(d.m.reg, w, 0);
      break;
    case Kind::MovRmImm:
      m_e.mov_imm64(0, truncate(d.imm, w));
      store_rm(0, w);
      break;
    case Kind::Lea:
      emit_ea(d.m, d.rip + d.len, 0);
      store_gpr(d.m.reg, w, R_EA);
      break;
    case Kind::Extend:
      if (d.m.is_reg) {
        load_gpr(0, d.m.rm, d.src_width);
        if (d.sign_extend)
          m_e.sxt(true, 0, 0, d.src_width);
      } else {
        load_mem(0, d.src_width, d.sign_extend);
      }
      store_gpr(d.m.reg, w, 0);
      break;
    case Kind::Alu:
      if (d.rm_is_dst) {
        load_rm(0, w);
        load_gpr(1, d.m.reg, w);
      } else {
        load_gpr(0, d.m.reg, w);
        load_rm(1, w);
      }
      emit_alu(d.alu_op, w, d.flags_written, flags_needed);
      if (d.writes_result) {
        if (d.rm_is_dst)
          store_rm(0, w);
        else
          store_gpr(d.m.reg, w, 0);
      }
      break;
    case Kind::AluImm:
      load_rm(0, w);
      m_e.mov_imm64(1, truncate(d.imm, w));
      emit_alu(d.alu_op, w, d.flags_written, flags_needed);
      if (d.writes_result)
        store_rm(0, w);
      break;
    case Kind::IncDec:
      load_rm(0, w);
      m_e.movz(false, 1, 1);
      emit_alu(d.alu_op, w, d.flags_written, flags_needed);
      store_rm(0, w);
      break;
    case Kind::ShiftImm:
      emit_shift(d, flags_needed);
      break;
    case Kind::Push:
    case Kind::PushImm:
    case Kind::PushRm:
      if (d.kind == Kind::PushRm)
        load_rm(0, 64); // source read with the old RSP (also for PUSH [rsp])
      m_e.ldr_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      m_e.sub_imm(R_EA, R_EA, 8);
      emit_bounds_check();
      if (d.kind == Kind::Push)
        load_gpr(0, d.reg, 64);
      else if (d.kind == Kind::PushImm)
        m_e.mov_imm64(0, d.imm);
      m_e.str_x_idx(0, R_GBASE, R_EA);
      m_e.str_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      break;
    case Kind::Vec:
      emit_vec(d);
      break;
    case Kind::CallRel:
    case Kind::CallInd:
      // The indirect target is read first (with the old RSP, as on hardware).
      if (d.kind == Kind::CallInd)
        load_rm(0, 64);
      else
        m_e.mov_imm64(0, d.target);
      m_e.ldr_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      m_e.sub_imm(R_EA, R_EA, 8);
      emit_bounds_check();
      m_e.mov_imm64(1, d.rip + d.len);
      m_e.str_x_idx(1, R_GBASE, R_EA);
      m_e.str_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      m_e.str_reg(0, R_CTX, offsetof(CpuContext, rip));
      break;
    case Kind::JmpInd:
      load_rm(0, 64);
      m_e.str_reg(0, R_CTX, offsetof(CpuContext, rip));
      break;
    case Kind::Ret:
      m_e.ldr_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      emit_bounds_check();
      m_e.ldr_x_idx(0, R_GBASE, R_EA);
      m_e.add_imm(R_EA, R_EA, 8);
      if (d.imm)
        m_e.add_imm64(R_EA, R_EA, static_cast<i64>(d.imm), 11);
      m_e.str_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      m_e.str_reg(0, R_CTX, offsetof(CpuContext, rip));
      break;
    case Kind::Pop:
      m_e.ldr_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      emit_bounds_check();
      m_e.ldr_x_idx(0, R_GBASE, R_EA);
      m_e.add_imm(R_EA, R_EA, 8);
      m_e.str_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      store_gpr(d.reg, 64, 0); // after RSP, so POP RSP keeps the popped value
      break;
    }
  }

  /// Emit the side-exit stubs collected so far (after the block epilogue).
  void emit_side_exits() {
    for (const SideExit &x : m_exits) {
      m_e.patch_b_cond(x.branch_pos, m_e.pos());
      m_e.mov_imm64(0, x.rip);
      m_e.str_reg(0, R_CTX, offsetof(CpuContext, rip));
      m_e.mov_imm64(0, x.retired);
      m_e.emit_epilogue();
    }
    m_exits.clear();
  }

private:
  struct SideExit {
    usize branch_pos;
    GuestAddr rip;
    u64 retired;
  };

  Arm64Emitter &m_e;
  const JitInsn *m_cur{nullptr};
  u64 m_retired{0};
  std::vector<SideExit> m_exits;

  [[nodiscard]] bool high_byte_reg(u8 reg) const {
    return !m_cur->p.has_rex && reg >= 4 && reg <= 7; // AH/CH/DH/BH
  }

  // x86 GPR -> zero-extended value in xd
  void load_gpr(u8 rd, u8 reg, u8 width) {
    switch (width) {
    case 64:
      m_e.ldr_reg(rd, R_CTX, get_gpr_offset(reg));
      break;
    case 32:
      m_e.ldr_w(rd, R_CTX, get_gpr_offset(reg));
      break;
    case 16:
      m_e.ldrh_imm(rd, R_CTX, get_gpr_offset(reg));
      break;
    default:
      if (high_byte_reg(reg))
        m_e.ldrb_imm(rd, R_CTX, get_gpr_offset(reg - 4) + 1);
      else
        m_e.ldrb_imm(rd, R_CTX, get_gpr_offset(reg));
      break;
    }
  }

  // Write an x86 GPR: 32-bit writes zero bits 63:32, 8/16-bit writes merge.
  // (The host is little-endian: a field's low byte/word is at its offset.)
  void store_gpr(u8 reg, u8 width, u8 rs) {
    switch (width) {
    case 64:
      m_e.str_reg(rs, R_CTX, get_gpr_offset(reg));
      break;
    case 32:
      m_e.str_w(rs, R_CTX, get_gpr_offset(reg));
      m_e.str_w(31, R_CTX, get_gpr_offset(reg) + 4); // wzr
      break;
    case 16:
      m_e.strh_imm(rs, R_CTX, get_gpr_offset(reg));
      break;
    default:
      if (high_byte_reg(reg))
        m_e.strb_imm(rs, R_CTX, get_gpr_offset(reg - 4) + 1);
      else
        m_e.strb_imm(rs, R_CTX, get_gpr_offset(reg));
      break;
    }
  }

  // Guest memory at x9 (already bounds-checked)
  void load_mem(u8 rd, u8 width, bool sign_extend = false) {
    switch (width) {
    case 64:
      m_e.ldr_x_idx(rd, R_GBASE, R_EA);
      break;
    case 32:
      if (sign_extend)
        m_e.ldrsw_idx(rd, R_GBASE, R_EA);
      else
        m_e.ldr_w_idx(rd, R_GBASE, R_EA);
      break;
    case 16:
      if (sign_extend)
        m_e.ldrsh_x_idx(rd, R_GBASE, R_EA);
      else
        m_e.ldrh_idx(rd, R_GBASE, R_EA);
      break;
    default:
      if (sign_extend)
        m_e.ldrsb_x_idx(rd, R_GBASE, R_EA);
      else
        m_e.ldrb_idx(rd, R_GBASE, R_EA);
      break;
    }
  }

  void store_mem(u8 rs, u8 width) {
    switch (width) {
    case 64:
      m_e.str_x_idx(rs, R_GBASE, R_EA);
      break;
    case 32:
      m_e.str_w_idx(rs, R_GBASE, R_EA);
      break;
    case 16:
      m_e.strh_idx(rs, R_GBASE, R_EA);
      break;
    default:
      m_e.strb_idx(rs, R_GBASE, R_EA);
      break;
    }
  }

  void load_rm(u8 rd, u8 width) {
    if (m_cur->m.is_reg)
      load_gpr(rd, m_cur->m.rm, width);
    else
      load_mem(rd, width);
  }

  void store_rm(u8 rs, u8 width) {
    if (m_cur->m.is_reg)
      store_gpr(m_cur->m.rm, width, rs);
    else
      store_mem(rs, width);
  }

  /// x9 = guest effective address of a memory ModRM operand.
  void emit_ea(const x86::ModRM &m, GuestAddr next_rip, u8 seg) {
    if (m.rip_relative) {
      m_e.mov_imm64(R_EA, m.rip_target(next_rip));
    } else if (m.base < 0 && m.index < 0) {
      m_e.mov_imm64(R_EA, static_cast<u64>(static_cast<i64>(m.disp)));
    } else {
      if (m.base >= 0)
        m_e.ldr_reg(R_EA, R_CTX, get_gpr_offset(static_cast<u8>(m.base)));
      if (m.index >= 0) {
        m_e.ldr_reg(10, R_CTX, get_gpr_offset(static_cast<u8>(m.index)));
        // add x9, {x9 | xzr}, x10, lsl #scale
        m_e.add_shifted(true, R_EA, m.base >= 0 ? R_EA : 31, 10,
                        Arm64Emitter::SHIFT_LSL, m.scale);
      }
      m_e.add_imm64(R_EA, R_EA, m.disp, 11);
    }
    if (seg == 0x64 || seg == 0x65) {
      m_e.ldr_reg(10, R_CTX,
                  seg == 0x64 ? offsetof(CpuContext, fs_base)
                              : offsetof(CpuContext, gs_base));
      m_e.add_reg(R_EA, R_EA, 10);
    }
  }

  /// if (x9 > guest_size - 8) side-exit with RIP at the current instruction.
  /// Accesses starting in the last 8 bytes of guest memory also take the
  /// exit; the interpreter then performs them with its own bounds checks.
  /// Wider accesses (`bytes` > 8) check their last 8 bytes the same way.
  void emit_bounds_check(u32 bytes = 8) {
    if (bytes > 8) {
      m_e.add_imm(10, R_EA, bytes - 8);
      m_e.cmp_reg(true, 10, R_GLIMIT);
    } else {
      m_e.cmp_reg(true, R_EA, R_GLIMIT);
    }
    m_exits.push_back({m_e.pos(), m_cur->rip, m_retired});
    m_e.b_cond(COND_HI, 0); // patched in emit_side_exits()
  }

  // ---- SSE / AVX ---------------------------------------------------------
  // Guest XMM/YMM registers live in ctx->ymm; v0-v3 and v31 are scratch
  // (caller-saved, so the block needs no NEON save/restore).

  static u32 xmm_off(u8 r, u32 byte = 0) {
    return static_cast<u32>(offsetof(CpuContext, ymm)) + 32u * r + byte;
  }
  void vld(u8 vt, u8 r, u32 half) { m_e.emit(a64::ldr_q(vt, R_CTX, xmm_off(r, 16 * half))); }
  void vst(u8 vt, u8 r, u32 half) { m_e.emit(a64::str_q(vt, R_CTX, xmm_off(r, 16 * half))); }
  void zero_upper(u8 r) {
    m_e.emit(a64::movi_zero(31));
    vst(31, r, 1);
  }
  // Register holding EA + off (x9 itself, or x10).
  u8 ea_plus(u32 off) {
    if (off == 0)
      return R_EA;
    m_e.add_imm(10, R_EA, off);
    return 10;
  }
  void mld_q(u8 vt, u32 off = 0) { m_e.emit(a64::ldr_q_x(vt, R_GBASE, ea_plus(off))); }
  void mst_q(u8 vt, u32 off = 0) { m_e.emit(a64::str_q_x(vt, R_GBASE, ea_plus(off))); }
  // Scalar load/store of lane 0 (4 or 8 bytes); a load zeroes the rest of vt.
  void mld_lane(bool dbl, u8 vt) {
    m_e.emit(dbl ? a64::ldr_d_x(vt, R_GBASE, R_EA) : a64::ldr_s_x(vt, R_GBASE, R_EA));
  }
  void mst_lane(bool dbl, u8 vt) {
    m_e.emit(dbl ? a64::str_d_x(vt, R_GBASE, R_EA) : a64::str_s_x(vt, R_GBASE, R_EA));
  }
  void ctx_ld_lane(bool dbl, u8 vt, u8 r, u32 byte = 0) {
    m_e.emit(dbl ? a64::ldr_d(vt, R_CTX, xmm_off(r, byte)) : a64::ldr_s(vt, R_CTX, xmm_off(r, byte)));
  }
  void ins(bool dbl, u8 d, u32 i, u8 n, u32 j) { m_e.emit(a64::ins_elem(dbl, d, i, n, j)); }
  // vt = half `half` of the rm operand (register or memory).
  void ld_rm_q(u8 vt, u32 half) {
    if (m_cur->m.is_reg)
      vld(vt, m_cur->m.rm, half);
    else
      mld_q(vt, 16 * half);
  }
  // vt lane 0 = the scalar rm operand.
  void ld_rm_lane(bool dbl, u8 vt) {
    if (m_cur->m.is_reg)
      ctx_ld_lane(dbl, vt, m_cur->m.rm);
    else
      mld_lane(dbl, vt);
  }

  static u32 vec_mem_bytes(const JitInsn &d) {
    switch (d.vop) {
    case VecOp::Load: case VecOp::Store: case VecOp::Packed: case VecOp::Shuf:
    case VecOp::Pshufd: case VecOp::Blend: case VecOp::Blendv: case VecOp::Moveldup:
    case VecOp::Movehdup:
      return d.l256 ? 32 : 16;
    case VecOp::Movddup:
      return d.l256 ? 32 : 8;
    case VecOp::LoadScalar: case VecOp::StoreScalar: case VecOp::Scalar: case VecOp::Ucomi:
    case VecOp::CvtF2F: case VecOp::CvttF2I:
      return d.dbl ? 8 : 4;
    case VecOp::Insertps: case VecOp::Broadcast:
      return 4;
    case VecOp::MovdToVec: case VecOp::MovdFromVec: case VecOp::CvtI2F: case VecOp::Pextr:
    case VecOp::Pinsr:
      return d.w64 ? 8 : 4;
    default: // Movq, MovqStore, LoadLow/High, StoreLow/High
      return 8;
    }
  }

  /// x86 returns the negative "QNaN indefinite" where an operation creates a
  /// NaN from non-NaN inputs (0/0, sqrt(-1), inf - inf); ARM's default NaN is
  /// positive. Set the sign of such lanes of v2 (inputs v0, v1; `unary`: v1
  /// only). Clobbers v3-v5.
  void emit_x86_nan(bool dbl, bool unary) {
    m_e.emit(a64::fcmeq_v(dbl, 3, 2, 2));            // result is a number
    m_e.emit(a64::fcmeq_v(dbl, 4, unary ? 1 : 0, unary ? 1 : 0));
    m_e.emit(a64::fcmeq_v(dbl, 5, 1, 1));
    m_e.emit(a64::and_v(4, 4, 5));                   // inputs are numbers
    m_e.emit(a64::bic_v(4, 4, 3));                   // ... but the result is not
    m_e.emit(a64::shl_sign(dbl, 4, 4));
    m_e.emit(a64::orr_v(2, 2, 4));
  }

  /// v2 = v0 OP v1 (all lanes; scalar callers use lane 0). Clobbers v3-v5.
  void emit_vbin(const JitInsn &d) {
    const bool dbl = d.dbl;
    switch (d.vbin) {
    case VecBin::Add: m_e.emit(a64::fadd_v(dbl, 2, 0, 1)); emit_x86_nan(dbl, false); break;
    case VecBin::Sub: m_e.emit(a64::fsub_v(dbl, 2, 0, 1)); emit_x86_nan(dbl, false); break;
    case VecBin::Mul: m_e.emit(a64::fmul_v(dbl, 2, 0, 1)); emit_x86_nan(dbl, false); break;
    case VecBin::Div: m_e.emit(a64::fdiv_v(dbl, 2, 0, 1)); emit_x86_nan(dbl, false); break;
    // x86 MIN/MAX return the second operand unless the comparison holds
    // (NaNs and equal zeros included): a < b ? a : b, a > b ? a : b.
    case VecBin::Min:
      m_e.emit(a64::fcmgt_v(dbl, 2, 1, 0));
      m_e.emit(a64::bsl_v(2, 0, 1));
      break;
    case VecBin::Max:
      m_e.emit(a64::fcmgt_v(dbl, 2, 0, 1));
      m_e.emit(a64::bsl_v(2, 0, 1));
      break;
    case VecBin::Sqrt: m_e.emit(a64::fsqrt_v(dbl, 2, 1)); emit_x86_nan(dbl, true); break;
    case VecBin::And: m_e.emit(a64::and_v(2, 0, 1)); break;
    case VecBin::Andn: m_e.emit(a64::bic_v(2, 1, 0)); break; // ~a & b
    case VecBin::Or: m_e.emit(a64::orr_v(2, 0, 1)); break;
    case VecBin::Xor: m_e.emit(a64::eor_v(2, 0, 1)); break;
    case VecBin::UnpckLo: m_e.emit(a64::zip1(dbl, 2, 0, 1)); break;
    case VecBin::UnpckHi: m_e.emit(a64::zip2(dbl, 2, 0, 1)); break;
    case VecBin::AddSub: // even lanes a - b, odd lanes a + b
      m_e.emit(a64::fsub_v(dbl, 2, 0, 1));
      m_e.emit(a64::fadd_v(dbl, 3, 0, 1));
      ins(dbl, 2, 1, 3, 1);
      if (!dbl)
        ins(false, 2, 3, 3, 3);
      emit_x86_nan(dbl, false);
      break;
    case VecBin::IAdd: m_e.emit(a64::add_v(dbl, 2, 0, 1)); break;
    case VecBin::ISub: m_e.emit(a64::sub_v(dbl, 2, 0, 1)); break;
    case VecBin::ICmpEq: m_e.emit(a64::cmeq_v(2, 0, 1)); break;
    case VecBin::Cmp: {
      const u32 pred = d.imm & 7;
      switch (pred & 3) {
      case 0: m_e.emit(a64::fcmeq_v(dbl, 2, 0, 1)); break; // EQ / NEQ
      case 1: m_e.emit(a64::fcmgt_v(dbl, 2, 1, 0)); break; // LT / NLT
      case 2: m_e.emit(a64::fcmge_v(dbl, 2, 1, 0)); break; // LE / NLE
      default:                                            // ORD / UNORD
        m_e.emit(a64::fcmeq_v(dbl, 2, 0, 0));
        m_e.emit(a64::fcmeq_v(dbl, 3, 1, 1));
        m_e.emit(a64::and_v(2, 2, 3));
        break;
      }
      // Predicates 3-6 are the negations of 7, 0, 1, 2.
      if (pred >= 3 && pred <= 6)
        m_e.emit(a64::not_v(2, 2));
      break;
    }
    }
  }

  void emit_vec(const JitInsn &d) {
    const bool mem = !d.m.is_reg;
    if (mem) {
      emit_ea(d.m, d.rip + d.len, d.p.seg);
      emit_bounds_check(vec_mem_bytes(d));
    }
    const u8 dst = d.m.reg, rm = d.m.rm;
    const u8 src1 = d.vex ? d.vvvv : dst;
    const u32 halves = d.l256 ? 2 : 1;
    const bool dbl = d.dbl;
    const u32 imm = static_cast<u32>(d.imm & 0xFF);
    // A VEX.128 write zeroes bits 255:128; VEX.256 wrote both halves.
    auto done = [&](u8 r) {
      if (d.vex && !d.l256)
        zero_upper(r);
    };
    switch (d.vop) {
    case VecOp::Load:
      for (u32 h = 0; h < halves; ++h) {
        ld_rm_q(0, h);
        vst(0, dst, h);
      }
      done(dst);
      break;
    case VecOp::Store:
      for (u32 h = 0; h < halves; ++h) {
        vld(0, dst, h);
        if (mem)
          mst_q(0, 16 * h);
        else
          vst(0, rm, h);
      }
      if (!mem)
        done(rm);
      break;
    case VecOp::LoadScalar:
      if (mem) {
        mld_lane(dbl, 0); // zero-extended to 128 bits
      } else {
        vld(0, src1, 0);
        vld(1, rm, 0);
        ins(dbl, 0, 0, 1, 0);
      }
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::StoreScalar:
      if (mem) {
        ctx_ld_lane(dbl, 0, dst);
        mst_lane(dbl, 0);
      } else {
        vld(0, d.vex ? d.vvvv : rm, 0);
        vld(1, dst, 0);
        ins(dbl, 0, 0, 1, 0);
        vst(0, rm, 0);
        done(rm);
      }
      break;
    case VecOp::Movq:
      if (mem)
        mld_lane(true, 0);
      else
        ctx_ld_lane(true, 0, rm);
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::MovqStore:
      ctx_ld_lane(true, 0, dst);
      if (mem) {
        mst_lane(true, 0);
      } else {
        vst(0, rm, 0);
        done(rm);
      }
      break;
    case VecOp::MovdToVec:
      load_rm(0, d.w64 ? 64 : 32);
      m_e.emit(d.w64 ? a64::fmov_d_x(0, 0) : a64::fmov_s_w(0, 0)); // zero-extends
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::MovdFromVec:
      if (d.w64)
        m_e.ldr_reg(0, R_CTX, xmm_off(dst));
      else
        m_e.ldr_w(0, R_CTX, xmm_off(dst));
      store_rm(0, d.w64 ? 64 : 32);
      break;
    case VecOp::LoadLow:
    case VecOp::LoadHigh:
      vld(0, src1, 0);
      mld_lane(true, 1);
      ins(true, 0, d.vop == VecOp::LoadHigh ? 1 : 0, 1, 0);
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::Movhlps:
    case VecOp::Movlhps:
      vld(0, src1, 0);
      vld(1, rm, 0);
      if (d.vop == VecOp::Movhlps)
        ins(true, 0, 0, 1, 1);
      else
        ins(true, 0, 1, 1, 0);
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::StoreLow:
    case VecOp::StoreHigh:
      ctx_ld_lane(true, 0, dst, d.vop == VecOp::StoreHigh ? 8 : 0);
      mst_lane(true, 0);
      break;
    case VecOp::Moveldup:
    case VecOp::Movehdup:
      for (u32 h = 0; h < halves; ++h) {
        ld_rm_q(1, h);
        m_e.emit(d.vop == VecOp::Moveldup ? a64::trn1(2, 1, 1) : a64::trn2(2, 1, 1));
        vst(2, dst, h);
      }
      done(dst);
      break;
    case VecOp::Movddup:
      for (u32 h = 0; h < halves; ++h) {
        if (mem && !d.l256)
          mld_lane(true, 1);
        else
          ld_rm_q(1, h);
        m_e.emit(a64::dup_elem(true, 2, 1, 0));
        vst(2, dst, h);
      }
      done(dst);
      break;
    case VecOp::Packed:
      for (u32 h = 0; h < halves; ++h) {
        if (d.vbin != VecBin::Sqrt)
          vld(0, src1, h);
        ld_rm_q(1, h);
        emit_vbin(d);
        vst(2, dst, h);
      }
      done(dst);
      break;
    case VecOp::Scalar:
      vld(0, src1, 0);
      ld_rm_lane(dbl, 1);
      emit_vbin(d);
      ins(dbl, 0, 0, 2, 0);
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::Ucomi: {
      // ZF, PF, CF = 111 unordered, 000 greater, 001 less, 100 equal;
      // OF, SF and AF are cleared. After FCMP: less or unordered = LT,
      // unordered = VS, equal = EQ.
      ctx_ld_lane(dbl, 0, dst);
      ld_rm_lane(dbl, 1);
      m_e.emit(a64::fcmp_s(dbl, 0, 1));
      m_e.cset(true, 3, 11); // LT -> CF
      m_e.cset(true, 4, 6);  // VS -> PF
      m_e.cset(true, 5, 0);  // EQ
      m_e.orr_reg(5, 5, 4);  // ZF = equal or unordered
      m_e.bfi(true, 3, 4, 2, 1);
      m_e.bfi(true, 3, 5, 6, 1);
      m_e.ldr_reg(4, R_CTX, offsetof(CpuContext, rflags));
      m_e.mov_imm64(5, FLAGS_ARITH | CpuContext::FLAG_AF);
      m_e.bic_shifted(true, 4, 4, 5);
      m_e.orr_shifted(true, 4, 4, 3);
      m_e.str_reg(4, R_CTX, offsetof(CpuContext, rflags));
      break;
    }
    case VecOp::Shuf:
      for (u32 h = 0; h < halves; ++h) {
        vld(0, src1, h);
        ld_rm_q(1, h);
        if (!dbl) {
          ins(false, 2, 0, 0, imm & 3);
          ins(false, 2, 1, 0, (imm >> 2) & 3);
          ins(false, 2, 2, 1, (imm >> 4) & 3);
          ins(false, 2, 3, 1, (imm >> 6) & 3);
        } else {
          ins(true, 2, 0, 0, (imm >> (2 * h)) & 1);
          ins(true, 2, 1, 1, (imm >> (2 * h + 1)) & 1);
        }
        vst(2, dst, h);
      }
      done(dst);
      break;
    case VecOp::Pshufd:
      for (u32 h = 0; h < halves; ++h) {
        ld_rm_q(1, h);
        for (u32 i = 0; i < 4; ++i)
          ins(false, 2, i, 1, (imm >> (2 * i)) & 3);
        vst(2, dst, h);
      }
      done(dst);
      break;
    case VecOp::Blend: {
      const u32 lanes = dbl ? 2 : 4;
      for (u32 h = 0; h < halves; ++h) {
        vld(0, src1, h);
        ld_rm_q(1, h);
        m_e.emit(a64::mov_v(2, 0));
        for (u32 i = 0; i < lanes; ++i)
          if ((imm >> (i + lanes * h)) & 1)
            ins(dbl, 2, i, 1, i);
        vst(2, dst, h);
      }
      done(dst);
      break;
    }
    case VecOp::Blendv: {
      const u8 mask = static_cast<u8>((imm >> 4) & 15);
      for (u32 h = 0; h < halves; ++h) {
        vld(0, src1, h);
        ld_rm_q(1, h);
        vld(3, mask, h);
        m_e.emit(a64::sshr_sign(dbl, 3, 3));
        m_e.emit(a64::bsl_v(3, 1, 0)); // sign set ? src2 : src1
        vst(3, dst, h);
      }
      done(dst);
      break;
    }
    case VecOp::Insertps: {
      vld(0, src1, 0);
      u32 from = 0;
      if (mem) {
        mld_lane(false, 1);
      } else {
        vld(1, rm, 0);
        from = (imm >> 6) & 3;
      }
      ins(false, 0, (imm >> 4) & 3, 1, from);
      if (imm & 15) {
        m_e.emit(a64::movi_zero(31));
        for (u32 k = 0; k < 4; ++k)
          if ((imm >> k) & 1)
            ins(false, 0, k, 31, 0);
      }
      vst(0, dst, 0);
      done(dst);
      break;
    }
    case VecOp::Broadcast:
      ld_rm_lane(false, 1);
      m_e.emit(a64::dup_elem(false, 2, 1, 0));
      for (u32 h = 0; h < halves; ++h)
        vst(2, dst, h);
      done(dst);
      break;
    case VecOp::Pextr: // ModRM.reg is the XMM source, rm the destination
      if (d.w64)
        m_e.ldr_reg(0, R_CTX, xmm_off(dst, 8 * (imm & 1)));
      else
        m_e.ldr_w(0, R_CTX, xmm_off(dst, 4 * (imm & 3)));
      store_rm(0, d.w64 ? 64 : 32);
      break;
    case VecOp::Pinsr:
      vld(0, src1, 0);
      load_rm(0, d.w64 ? 64 : 32);
      m_e.emit(a64::ins_gpr(d.w64, 0, d.w64 ? (imm & 1) : (imm & 3), 0));
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::CvtF2F: // dbl: cvtsd2ss, else cvtss2sd
      vld(0, src1, 0);
      ld_rm_lane(dbl, 1);
      if (dbl) {
        m_e.emit(a64::fcvt_s_d(2, 1));
        ins(false, 0, 0, 2, 0);
      } else {
        m_e.emit(a64::fcvt_d_s(2, 1));
        ins(true, 0, 0, 2, 0);
      }
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::CvtI2F:
      vld(0, src1, 0);
      load_rm(0, d.w64 ? 64 : 32);
      m_e.emit(a64::scvtf(dbl, d.w64, 2, 0));
      ins(dbl, 0, 0, 2, 0);
      vst(0, dst, 0);
      done(dst);
      break;
    case VecOp::CvttF2I: {
      // x86 returns the "integer indefinite" (only the sign bit set) for NaN
      // and out-of-range inputs; FCVTZS saturates and maps NaN to 0.
      ld_rm_lane(dbl, 1);
      m_e.emit(a64::fcvtzs(dbl, true, 0, 1));
      m_e.emit(a64::fcmp_s(dbl, 1, 1));
      if (d.w64) {
        m_e.mov_imm64(2, 0x8000000000000000ull);
        m_e.emit(a64::csel(true, 0, 2, 0, 6)); // NaN (VS)
        m_e.mov_imm64(3, 0x7FFFFFFFFFFFFFFFull);
        m_e.cmp_reg(true, 0, 3);
        m_e.emit(a64::csel(true, 0, 2, 0, 0)); // saturated high (EQ)
        store_gpr(dst, 64, 0);
      } else {
        m_e.mov_imm64(2, 0x80000000ull);
        m_e.emit(a64::csel(true, 0, 2, 0, 6)); // NaN (VS)
        m_e.sxt(true, 3, 0, 32);
        m_e.cmp_reg(true, 3, 0);
        m_e.emit(a64::csel(true, 0, 2, 0, 1)); // not an int32 (NE)
        store_gpr(dst, 32, 0);
      }
      break;
    }
    }
  }

  /// x0 = x0 OP x1 at `width` bits (result zero-extended); if any flag it
  /// writes is live, merge CF/PF/ZF/SF/OF (as selected by flags_written)
  /// into ctx.rflags. 8/16-bit operations run in the top bits of a W
  /// register so that ARM's N/Z/C/V describe the narrow result exactly.
  void emit_alu(u8 op, u8 width, u64 flags_written, u64 flags_needed) {
    const bool sf = width == 64;
    const u8 sh = width < 32 ? static_cast<u8>(32 - width) : 0;
    if (sh)
      m_e.lsl_imm(false, 0, 0, sh);
    bool borrow = false;
    switch (op) {
    case ALU_ADD:
      m_e.adds_shifted(sf, 0, 0, 1, Arm64Emitter::SHIFT_LSL, sh);
      break;
    case ALU_SUB:
    case ALU_CMP:
      m_e.subs_shifted(sf, 0, 0, 1, Arm64Emitter::SHIFT_LSL, sh);
      borrow = true;
      break;
    case ALU_AND:
      m_e.ands_shifted(sf, 0, 0, 1, Arm64Emitter::SHIFT_LSL, sh);
      break;
    case ALU_OR:
      m_e.orr_shifted(sf, 0, 0, 1, Arm64Emitter::SHIFT_LSL, sh);
      if (flags_needed)
        m_e.ands_shifted(sf, 31, 0, 0); // tst: N/Z from result, C = V = 0
      break;
    case ALU_XOR:
      m_e.eor_shifted(sf, 0, 0, 1, Arm64Emitter::SHIFT_LSL, sh);
      if (flags_needed)
        m_e.ands_shifted(sf, 31, 0, 0);
      break;
    default:
      break;
    }
    if (sh)
      m_e.lsr_imm(false, 0, 0, sh); // UBFM leaves NZCV alone
    if (flags_needed)
      emit_flags(borrow, flags_written, 0);
  }

  /// SHL/SHR/SAR by a constant, matching the interpreter: ZF/SF/PF from the
  /// result, OF = 0, CF = last bit shifted out (0 once the count exceeds the
  /// operand size); count 0 changes nothing.
  void emit_shift(const JitInsn &d, u64 flags_needed) {
    const u8 w = d.width;
    const u8 c = static_cast<u8>(d.imm);
    if (c == 0) {
      // No shift, but a 32-bit register destination is still written, which
      // zero-extends it (as on hardware and in the interpreter).
      if (w == 32 && d.m.is_reg) {
        load_rm(0, 32);
        store_rm(0, 32);
      }
      return;
    }
    const bool sf = w == 64;
    load_rm(0, w); // zero-extended
    if (flags_needed) {
      // bit index of the last bit shifted out, if any
      int p = -1;
      if (d.alu_op == 4)
        p = c < w ? w - c : (c == w ? 0 : -1);
      else if (d.alu_op == 7) // SAR: shifting past the width keeps shifting out the sign
        p = c <= w ? c - 1 : w - 1;
      else
        p = c <= w ? c - 1 : -1;
      if (p >= 0)
        m_e.ubfx(true, 6, 0, static_cast<u8>(p), 1);
      else
        m_e.movz(true, 6, 0);
    }
    // OF for a 1-bit shift (x10): SHR -> MSB of the source; SHL -> MSB of
    // the result ^ CF (below); SAR -> 0. Other counts leave OF = 0.
    const bool of1 = c == 1 && (flags_needed & CpuContext::FLAG_OF);
    if (of1 && d.alu_op == 5)
      m_e.ubfx(true, 10, 0, static_cast<u8>(w - 1), 1);
    switch (d.alu_op) {
    case 4: // SHL
      if (c >= w) {
        m_e.movz(true, 0, 0);
      } else {
        m_e.lsl_imm(sf, 0, 0, c);
        if (w < 32)
          m_e.ubfx(false, 0, 0, 0, w);
      }
      break;
    case 5: // SHR
      if (c >= w)
        m_e.movz(true, 0, 0);
      else
        m_e.lsr_imm(sf, 0, 0, c);
      break;
    default: // 7: SAR
      if (w < 64)
        m_e.sxt(true, 0, 0, w);
      m_e.asr_imm(true, 0, 0, std::min<u8>(c, static_cast<u8>(w - 1)));
      if (w < 64)
        m_e.ubfx(true, 0, 0, 0, w);
      break;
    }
    if (flags_needed) {
      if (w < 32) {
        m_e.lsl_imm(false, 7, 0, static_cast<u8>(32 - w));
        m_e.ands_shifted(false, 31, 7, 7); // N/Z of the narrow result, C=V=0
      } else {
        m_e.ands_shifted(sf, 31, 0, 0);
      }
      emit_flags(false, d.flags_written, 0, 6);
      if (of1) {
        if (d.alu_op == 4) {
          m_e.ubfx(true, 10, 0, static_cast<u8>(w - 1), 1);
          m_e.eor_reg(10, 10, 6);
        } else if (d.alu_op == 7) {
          m_e.movz(true, 10, 0);
        }
        m_e.ldr_reg(11, R_CTX, offsetof(CpuContext, rflags));
        m_e.bfi(true, 11, 10, 11, 1); // OF -> bit 11
        m_e.str_reg(11, R_CTX, offsetof(CpuContext, rflags));
      }
    }
    store_rm(0, w);
  }

  /// Convert NZCV (plus the parity of the low byte of `res`) to x86 flags and
  /// merge the bits selected by `mask` into ctx.rflags. CF comes from the C
  /// flag, or from bit 0 of register `cf_reg` when given.
  void emit_flags(bool borrow, u64 mask, u8 res, int cf_reg = -1) {
    m_e.mrs_nzcv(2);
    if ((mask & CpuContext::FLAG_CF) && cf_reg >= 0)
      m_e.mov_reg(3, static_cast<u8>(cf_reg));
    else if (mask & CpuContext::FLAG_CF)
      m_e.cset(true, 3, borrow ? COND_CC : COND_CS); // CF -> bit 0
    else
      m_e.movz(true, 3, 0);
    // PF = even parity of the low byte. 0x9669 is a 16-entry table of
    // "nibble has even parity", indexed by (low nibble ^ high nibble).
    m_e.eor_shifted(false, 4, res, res, Arm64Emitter::SHIFT_LSR, 4);
    m_e.ubfx(false, 4, 4, 0, 4);
    m_e.movz(false, 5, 0x9669);
    m_e.lsrv(false, 5, 5, 4);
    m_e.bfi(true, 3, 5, 2, 1);   // PF -> bit 2
    m_e.lsr_imm(true, 4, 2, 30); // bit0 = Z, bit1 = N
    m_e.bfi(true, 3, 4, 6, 2);   // ZF -> bit 6, SF -> bit 7
    m_e.lsr_imm(true, 4, 2, 28); // bit0 = V
    m_e.bfi(true, 3, 4, 11, 1);  // OF -> bit 11
    m_e.ldr_reg(4, R_CTX, offsetof(CpuContext, rflags));
    m_e.mov_imm64(5, mask);
    m_e.bic_shifted(true, 4, 4, 5);
    m_e.orr_shifted(true, 4, 4, 3);
    m_e.str_reg(4, R_CTX, offsetof(CpuContext, rflags));
  }
};

} // namespace

namespace {
std::atomic<u64> g_jit_epoch{1};
} // namespace

JitCompiler::JitCompiler(usize code_cache_size)
    : m_code_buffer_size(code_cache_size) {
  m_epoch.store(g_jit_epoch.fetch_add(1, std::memory_order_relaxed), std::memory_order_release);
  m_code_buffer = static_cast<u8 *>(
      ::mmap(nullptr, m_code_buffer_size, PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0));

  if (m_code_buffer == MAP_FAILED) {
    m_code_buffer = nullptr;
  }
}

JitCompiler::~JitCompiler() {
  if (m_code_buffer && m_code_buffer != MAP_FAILED) {
    ::munmap(m_code_buffer, m_code_buffer_size);
    m_code_buffer = nullptr;
  }
}

JitCompiler::JitCompiler(JitCompiler &&other) noexcept
    : m_code_buffer(other.m_code_buffer),
      m_code_buffer_size(other.m_code_buffer_size),
      m_code_buffer_offset(other.m_code_buffer_offset),
      m_total_invocations(other.m_total_invocations),
      m_blocks(std::move(other.m_blocks)),
      m_metrics(std::move(other.m_metrics)),
      m_failed(std::move(other.m_failed)) {
  other.m_code_buffer = nullptr;
  other.m_code_buffer_size = 0;
  other.m_code_buffer_offset = 0;
  m_epoch.store(g_jit_epoch.fetch_add(1, std::memory_order_relaxed), std::memory_order_release);
  other.m_epoch.store(g_jit_epoch.fetch_add(1, std::memory_order_relaxed), std::memory_order_release);
}

JitCompiler &JitCompiler::operator=(JitCompiler &&other) noexcept {
  if (this != &other) {
    if (m_code_buffer && m_code_buffer != MAP_FAILED) {
      ::munmap(m_code_buffer, m_code_buffer_size);
    }
    m_code_buffer = other.m_code_buffer;
    m_code_buffer_size = other.m_code_buffer_size;
    m_code_buffer_offset = other.m_code_buffer_offset;
    m_total_invocations = other.m_total_invocations;
    m_blocks = std::move(other.m_blocks);
    m_metrics = std::move(other.m_metrics);
    m_failed = std::move(other.m_failed);

    other.m_code_buffer = nullptr;
    other.m_code_buffer_size = 0;
    other.m_code_buffer_offset = 0;
    m_epoch.store(g_jit_epoch.fetch_add(1, std::memory_order_relaxed), std::memory_order_release);
    other.m_epoch.store(g_jit_epoch.fetch_add(1, std::memory_order_relaxed), std::memory_order_release);
  }
  return *this;
}

void *JitCompiler::allocate_code(usize size) {
  if (!m_code_buffer)
    return nullptr;

  // Align to 16 bytes
  m_code_buffer_offset = (m_code_buffer_offset + 15) & ~15ULL;
  if (m_code_buffer_offset + size > m_code_buffer_size) {
    static std::atomic<bool> s_warned{false};
    if (!s_warned.exchange(true))
      std::fprintf(stderr, "[JIT] code cache exhausted (%zu bytes); new blocks will run interpreted\n",
                   m_code_buffer_size);
    return nullptr;
  }

  void *ptr = m_code_buffer + m_code_buffer_offset;
  m_code_buffer_offset += size;
  return ptr;
}

// MAP_JIT write protection is per thread and only commit_code() switches it
// off (and back on); toggling it before every block was a measurable cost.
static thread_local bool t_jit_executable = false;

void JitCompiler::commit_code(void *dest, const u32 *src, usize size_bytes) {
  // 1. Enable write permission for Apple Silicon JIT thread
  pthread_jit_write_protect_np(0);

  // 2. Copy machine instructions
  std::memcpy(dest, src, size_bytes);

  // 3. Restore executable permission
  pthread_jit_write_protect_np(1);
  t_jit_executable = true;

  // 4. Invalidate instruction cache
  sys_icache_invalidate(dest, size_bytes);
}

bool JitCompiler::is_block_cached(GuestAddr guest_rip) const {
  std::shared_lock lock(m_cache_mutex);
  return m_blocks.find(guest_rip) != m_blocks.end();
}

void JitCompiler::clear_cache() {
  std::unique_lock lock(m_cache_mutex);
  m_blocks.clear();
  m_metrics.clear();
  m_failed.clear();
  m_code_buffer_offset = 0;
  m_epoch.store(g_jit_epoch.fetch_add(1, std::memory_order_relaxed), std::memory_order_release);
}

usize JitCompiler::cached_block_count() const noexcept {
  std::shared_lock lock(m_cache_mutex);
  return m_blocks.size();
}

usize JitCompiler::code_buffer_used() const noexcept {
  std::shared_lock lock(m_cache_mutex);
  return m_code_buffer_offset;
}

usize JitCompiler::total_invocations() const noexcept {
  std::shared_lock lock(m_cache_mutex);
  return std::atomic_ref<usize>(const_cast<usize &>(m_total_invocations))
      .load(std::memory_order_relaxed);
}

const JitBlockMetrics *
JitCompiler::get_block_metrics(GuestAddr guest_rip) const {
  std::shared_lock lock(m_cache_mutex);
  auto it = m_metrics.find(guest_rip);
  if (it != m_metrics.end()) {
    return &it->second;
  }
  return nullptr;
}

JittedBlockFunc JitCompiler::compile_block(GuestAddr guest_rip,
                                           const MemoryManager &memory) {
  std::unique_lock lock(m_cache_mutex);
  return compile_block_locked(guest_rip, memory);
}

JittedBlockFunc JitCompiler::compile_block_locked(GuestAddr guest_rip,
                                                  const MemoryManager &memory) {
  if (auto it = m_blocks.find(guest_rip); it != m_blocks.end()) {
    return it->second;
  }

  if (!m_code_buffer || !memory.is_valid_address(guest_rip, 1) ||
      m_failed.count(guest_rip) != 0) {
    return nullptr;
  }

  // Pass 1: decode the block.
  constexpr usize MAX_BLOCK_INSNS = 128;
  std::vector<JitInsn> insns;
  GuestAddr current_rip = guest_rip;
  bool block_ended = false;
  GuestAddr insn_rip = guest_rip; // start of the instruction being decoded

  while (!block_ended && insns.size() < MAX_BLOCK_INSNS) {
    insn_rip = current_rip;
    if (!memory.is_valid_address(current_rip, 1)) {
      break;
    }
    const usize avail = static_cast<usize>(std::min<GuestSize>(
        x86::MAX_INSN_LEN, memory.total_size() - current_rip));
    const u8 *code = memory.to_host_ptr<u8>(current_rip);
    if (!code)
      break;

    // HLT / SYSCALL / UD2 / INT3 / RET / CALL / Jcc: end the block without
    // advancing RIP so the interpreter steps into the instruction.
    x86::Prefixes pfx;
    if (x86::decode_prefixes(code, avail, pfx) &&
        is_block_terminator(code, avail, pfx)) {
      block_ended = true;
      break;
    }

    JitInsn insn;
    if (!decode_insn(code, avail, current_rip, insn)) {
      // Unknown / unsupported instruction: stop compiling block
      break;
    }

    insns.push_back(insn);
    if (insn.kind == Kind::Jcc || writes_rip(insn.kind)) {
      block_ended = true;
      break;
    }
    if (insn.kind == Kind::Jmp) {
      current_rip = insn.target;
      block_ended = true;
      break;
    }
    current_rip += insn.len;
  }

  if (jit_stats_enabled()) {
    record_block_end_stats(memory, block_ended, insn_rip, current_rip,
                           insns.size(), MAX_BLOCK_INSNS);
  }

  if (insns.empty()) {
    // Interpreter fallback. Remember the failure so the next visit does not
    // decode the same instruction again.
    m_failed.insert(guest_rip);
    return nullptr;
  }

  // Pass 2: flag liveness. Every flag is live at the block exit; a flag write
  // is dead if a later instruction in the block overwrites that flag before
  // anything could read it. (Jcc/CMOVcc/SETcc read flags from ctx->rflags,
  // so they keep earlier writes live; INC/DEC simply do not write CF.) Side exits stay exact: the interpreter
  // re-executes the exiting instruction, which rewrites those same flags.
  std::vector<u64> flags_needed(insns.size(), 0);
  u64 live = FLAGS_ARITH;
  for (usize i = insns.size(); i-- > 0;) {
    flags_needed[i] = insns[i].flags_written & live;
    live &= ~insns[i].flags_written;
    live |= insns[i].flags_read;
  }

  // Pass 3: emit.
  m_emitter.clear();
  m_emitter.emit_prologue();
  BlockTranslator translator(m_emitter);
  for (usize i = 0; i < insns.size(); ++i) {
    translator.emit(insns[i], flags_needed[i], i);
  }

  const usize instructions = insns.size();

  // Update RIP at the end of the block (CALL/RET/indirect JMP stored it
  // already); return the retired-instruction count.
  if (insns.back().kind == Kind::Jcc) {
    emit_jcc_rip(m_emitter, insns.back());
    m_emitter.str_reg(0, 19, offsetof(CpuContext, rip));
  } else if (!writes_rip(insns.back().kind)) {
    m_emitter.mov_imm64(0, current_rip);
    m_emitter.str_reg(0, 19, offsetof(CpuContext, rip));
  }
  m_emitter.mov_imm64(0, instructions);
  m_emitter.emit_epilogue();

  translator.emit_side_exits();

  const usize code_size = m_emitter.size_bytes();
  void *code_ptr = allocate_code(code_size);
  if (!code_ptr) {
    return nullptr;
  }

  commit_code(code_ptr, m_emitter.code().data(), code_size);

  auto fn = reinterpret_cast<JittedBlockFunc>(code_ptr);
  m_blocks[guest_rip] = fn;
  m_metrics[guest_rip] = JitBlockMetrics{
      .instructions_compiled = instructions,
      .bytes_emitted = code_size,
      .execution_count = 0,
  };

  return fn;
}

namespace {

struct ThreadBlockCache {
  struct Entry {
    JittedBlockFunc fn;      // nullptr: compiling this rip failed
    usize *execution_count;  // the block's JitBlockMetrics counter
  };
  // Direct-mapped front for the hot loop: one compare instead of a hash-map
  // probe per block. Misses fall back to `map`.
  static constexpr usize kFastSlots = 1u << 16;
  struct Slot {
    GuestAddr rip;
    Entry entry;
  };
  u64 epoch = 0;
  std::unique_ptr<Slot[]> fast;
  std::unordered_map<GuestAddr, Entry> map;
  static usize slot_of(GuestAddr rip) { return static_cast<usize>((rip ^ (rip >> 16)) & (kFastSlots - 1)); }
};
thread_local ThreadBlockCache t_block_cache;
} // namespace

bool JitCompiler::execute_block(CpuContext &ctx, const MemoryManager &memory,
                                usize *retired) {
  if (retired)
    *retired = 0;
  // RIP 0 (a guest thread returning from its entry point) and the other
  // invalid low targets are the interpreter's to report.
  if (ctx.rip < 0x20)
    return false;
  JittedBlockFunc fn = nullptr;
  usize *exec_count = nullptr;
  ThreadBlockCache &tc = t_block_cache;
  const u64 epoch = m_epoch.load(std::memory_order_acquire);
  if (tc.epoch != epoch || !tc.fast) {
    tc.map.clear();
    if (!tc.fast)
      tc.fast = std::make_unique<ThreadBlockCache::Slot[]>(ThreadBlockCache::kFastSlots);
    for (usize i = 0; i < ThreadBlockCache::kFastSlots; ++i)
      tc.fast[i].rip = ~GuestAddr{0};
    tc.epoch = epoch;
  }
  ThreadBlockCache::Slot &slot = tc.fast[ThreadBlockCache::slot_of(ctx.rip)];
  if (slot.rip == ctx.rip) {
    if (!slot.entry.fn)
      return false;
    fn = slot.entry.fn;
    exec_count = slot.entry.execution_count;
  } else if (auto it = tc.map.find(ctx.rip); it != tc.map.end()) {
    // Fastest path: this thread has seen the block (or its failure) before.
    slot = {ctx.rip, it->second};
    if (!it->second.fn)
      return false;
    fn = it->second.fn;
    exec_count = it->second.execution_count;
  } else {
    {
      // Another thread may have compiled it: look it up under the shared lock.
      std::shared_lock lock(m_cache_mutex);
      if (auto b = m_blocks.find(ctx.rip); b != m_blocks.end()) {
        fn = b->second;
        exec_count = &m_metrics.find(ctx.rip)->second.execution_count;
      } else if (m_failed.count(ctx.rip) != 0) {
        tc.map.emplace(ctx.rip, ThreadBlockCache::Entry{nullptr, nullptr});
        return false;
      }
    }
    if (!fn) {
      std::unique_lock lock(m_cache_mutex);
      fn = compile_block_locked(ctx.rip, memory);
      if (!fn) {
        // Remember only permanent failures; a full code cache is reported
        // by allocate_code and may be cleared later.
        if (m_failed.count(ctx.rip) != 0)
          tc.map.emplace(ctx.rip, ThreadBlockCache::Entry{nullptr, nullptr});
        return false;
      }
      exec_count = &m_metrics[ctx.rip].execution_count;
    }
    tc.map.emplace(ctx.rip, ThreadBlockCache::Entry{fn, exec_count});
    slot = {ctx.rip, ThreadBlockCache::Entry{fn, exec_count}};
  }
  if (m_collect_metrics) {
    std::atomic_ref<usize>(*exec_count).fetch_add(1, std::memory_order_relaxed);
    std::atomic_ref<usize>(m_total_invocations).fetch_add(1, std::memory_order_relaxed);
  }

  // Enforce execute mode for MAP_JIT on Apple Silicon (once per thread).
  if (!t_jit_executable) {
    pthread_jit_write_protect_np(1);
    t_jit_executable = true;
  }

  // Compiled code stores to guest memory: the MemoryManager is only const
  // because compilation does not modify it.
  u8 *guest_base = const_cast<u8 *>(memory.base_address());
  const u64 n = fn(&ctx, guest_base, memory.total_size());
  if (retired)
    *retired = static_cast<usize>(n);
  // A block that side-exited on its first instruction made no progress: the
  // caller must let the interpreter execute that instruction.
  return n != 0;
}

} // namespace ps4
