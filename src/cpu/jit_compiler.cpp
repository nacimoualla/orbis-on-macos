#include "ps4/jit_compiler.hpp"

#include <atomic>
#include "ps4/jit_stats.hpp"
#include "ps4/x86_decode.hpp"
#include <algorithm>
#include <cstring>
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
  if (op0 == 0xF4 || op0 == 0xCC || op0 == 0xC3 || op0 == 0xC2 ||
      op0 == 0xE8)
    return true;
  if (op0 == 0x0F && (op1 == 0x05 || op1 == 0x0B))
    return true;
  return false;
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
  void emit_bounds_check() {
    m_e.cmp_reg(true, R_EA, R_GLIMIT);
    m_exits.push_back({m_e.pos(), m_cur->rip, m_retired});
    m_e.b_cond(COND_HI, 0); // patched in emit_side_exits()
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

JitCompiler::JitCompiler(usize code_cache_size)
    : m_code_buffer_size(code_cache_size) {
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
  }
  return *this;
}

void *JitCompiler::allocate_code(usize size) {
  if (!m_code_buffer)
    return nullptr;

  // Align to 16 bytes
  m_code_buffer_offset = (m_code_buffer_offset + 15) & ~15ULL;
  if (m_code_buffer_offset + size > m_code_buffer_size) {
    return nullptr; // Code cache exhausted
  }

  void *ptr = m_code_buffer + m_code_buffer_offset;
  m_code_buffer_offset += size;
  return ptr;
}

void JitCompiler::commit_code(void *dest, const u32 *src, usize size_bytes) {
  // 1. Enable write permission for Apple Silicon JIT thread
  pthread_jit_write_protect_np(0);

  // 2. Copy machine instructions
  std::memcpy(dest, src, size_bytes);

  // 3. Restore executable permission
  pthread_jit_write_protect_np(1);

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
    if (insn.kind == Kind::Jcc) {
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

  // Always update RIP at end of block; return the retired-instruction count
  if (insns.back().kind == Kind::Jcc) {
    emit_jcc_rip(m_emitter, insns.back());
  } else {
    m_emitter.mov_imm64(0, current_rip);
  }
  m_emitter.str_reg(0, 19, offsetof(CpuContext, rip));
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

bool JitCompiler::execute_block(CpuContext &ctx, const MemoryManager &memory,
                                usize *retired) {
  if (retired)
    *retired = 0;
  JittedBlockFunc fn = nullptr;
  {
    // Fast path: cache hit (or a known compile failure) under the shared lock.
    std::shared_lock lock(m_cache_mutex);
    if (auto it = m_blocks.find(ctx.rip); it != m_blocks.end()) {
      fn = it->second;
      std::atomic_ref<usize>(m_metrics.find(ctx.rip)->second.execution_count)
          .fetch_add(1, std::memory_order_relaxed);
      std::atomic_ref<usize>(m_total_invocations)
          .fetch_add(1, std::memory_order_relaxed);
    } else if (m_failed.count(ctx.rip) != 0) {
      return false;
    }
  }
  if (!fn) {
    std::unique_lock lock(m_cache_mutex);
    fn = compile_block_locked(ctx.rip, memory);
    if (!fn) {
      return false;
    }
    m_metrics[ctx.rip].execution_count++;
    m_total_invocations++;
  }

  // Enforce execute mode for MAP_JIT on Apple Silicon
  pthread_jit_write_protect_np(1);

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
