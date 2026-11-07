#include "ps4/jit_compiler.hpp"

#include <atomic>
#include "ps4/jit_stats.hpp"
#include "ps4/x86_decode.hpp"
#include <algorithm>
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

// --- Variable shifts, select, multiply, rotate -----------------------------------

namespace {
inline u32 rrr(u32 base, bool sf, u8 rd, u8 rn, u8 rm) {
  return base | (sf ? 0x80000000u : 0u) | (static_cast<u32>(rm) << 16) |
         (static_cast<u32>(rn) << 5) | rd;
}
} // namespace

void Arm64Emitter::lslv(bool sf, u8 rd, u8 rn, u8 rm) { emit(rrr(0x1ac02000u, sf, rd, rn, rm)); }
void Arm64Emitter::asrv(bool sf, u8 rd, u8 rn, u8 rm) { emit(rrr(0x1ac02800u, sf, rd, rn, rm)); }

void Arm64Emitter::csel(bool sf, u8 rd, u8 rn, u8 rm, u8 cond) {
  emit(rrr(0x1a800000u, sf, rd, rn, rm) | (static_cast<u32>(cond & 0xf) << 12));
}

void Arm64Emitter::cmp_imm(bool sf, u8 rn, u16 imm12) {
  // SUBS zr, rn, #imm12
  emit((sf ? 0xf1000000u : 0x71000000u) | (static_cast<u32>(imm12 & 0xfff) << 10) |
       (static_cast<u32>(rn) << 5) | 31u);
}

void Arm64Emitter::mul(bool sf, u8 rd, u8 rn, u8 rm) {
  emit(rrr(0x1b000000u, sf, rd, rn, rm) | (31u << 10)); // MADD rd, rn, rm, zr
}
void Arm64Emitter::smull(u8 rd, u8 rn, u8 rm) {
  emit(rrr(0x9b200000u, false, rd, rn, rm) | (31u << 10)); // SMADDL xd, wn, wm, xzr
}
void Arm64Emitter::smulh(u8 rd, u8 rn, u8 rm) {
  emit(rrr(0x9b400000u, false, rd, rn, rm) | (31u << 10));
}

void Arm64Emitter::ror_imm(bool sf, u8 rd, u8 rn, u8 shift) {
  emit((sf ? 0x93c00000u : 0x13800000u) | (static_cast<u32>(rn) << 16) |
       (static_cast<u32>(shift & (sf ? 63 : 31)) << 10) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::cbz(bool sf, u8 rt, i32 offset_words) {
  emit((sf ? 0xb4000000u : 0x34000000u) | ((static_cast<u32>(offset_words) & 0x7ffff) << 5) | rt);
}

// --- SIMD&FP -------------------------------------------------------------------------

namespace {
// LDR/STR (SIMD&FP, unsigned offset) and (register, LSL #0), indexed by
// size_log2 - 2 (S, D, Q).
constexpr u32 kFpLdrImm[3] = {0xbd400000u, 0xfd400000u, 0x3dc00000u};
constexpr u32 kFpStrImm[3] = {0xbd000000u, 0xfd000000u, 0x3d800000u};
constexpr u32 kFpLdrIdx[3] = {0xbc606800u, 0xfc606800u, 0x3ce06800u};
constexpr u32 kFpStrIdx[3] = {0xbc206800u, 0xfc206800u, 0x3ca06800u};
} // namespace

void Arm64Emitter::fp_ldr(u8 size_log2, u8 vt, u8 rn, u32 offset_bytes) {
  emit(kFpLdrImm[size_log2 - 2] | (((offset_bytes >> size_log2) & 0xfff) << 10) |
       (static_cast<u32>(rn) << 5) | vt);
}
void Arm64Emitter::fp_str(u8 size_log2, u8 vt, u8 rn, u32 offset_bytes) {
  emit(kFpStrImm[size_log2 - 2] | (((offset_bytes >> size_log2) & 0xfff) << 10) |
       (static_cast<u32>(rn) << 5) | vt);
}
void Arm64Emitter::fp_ldr_idx(u8 size_log2, u8 vt, u8 rn, u8 rm) {
  emit(kFpLdrIdx[size_log2 - 2] | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | vt);
}
void Arm64Emitter::fp_str_idx(u8 size_log2, u8 vt, u8 rn, u8 rm) {
  emit(kFpStrIdx[size_log2 - 2] | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | vt);
}

void Arm64Emitter::fp_arith(bool dbl, u8 op, u8 vd, u8 vn, u8 vm) {
  emit(0x1e200800u | (dbl ? 0x400000u : 0u) | (static_cast<u32>(op & 3) << 12) |
       (static_cast<u32>(vm) << 16) | (static_cast<u32>(vn) << 5) | vd);
}
void Arm64Emitter::fcmp(bool dbl, u8 vn, u8 vm) {
  emit(0x1e202000u | (dbl ? 0x400000u : 0u) | (static_cast<u32>(vm) << 16) | (static_cast<u32>(vn) << 5));
}
void Arm64Emitter::fcsel(bool dbl, u8 vd, u8 vn, u8 vm, u8 cond) {
  emit(0x1e200c00u | (dbl ? 0x400000u : 0u) | (static_cast<u32>(vm) << 16) |
       (static_cast<u32>(cond & 0xf) << 12) | (static_cast<u32>(vn) << 5) | vd);
}
void Arm64Emitter::scvtf(bool dbl, bool src64, u8 vd, u8 rn) {
  emit(0x1e220000u | (src64 ? 0x80000000u : 0u) | (dbl ? 0x400000u : 0u) |
       (static_cast<u32>(rn) << 5) | vd);
}
void Arm64Emitter::ins_lane0(bool dbl, u8 vd, u8 vn) {
  emit((dbl ? 0x6e080400u : 0x6e040400u) | (static_cast<u32>(vn) << 5) | vd);
}
void Arm64Emitter::eor_v16(u8 vd, u8 vn, u8 vm) {
  emit(0x6e201c00u | (static_cast<u32>(vm) << 16) | (static_cast<u32>(vn) << 5) | vd);
}
void Arm64Emitter::fmov_to_gpr(bool dbl, u8 rd, u8 vn) {
  emit((dbl ? 0x9e660000u : 0x1e260000u) | (static_cast<u32>(vn) << 5) | rd);
}
void Arm64Emitter::fmov_from_gpr(bool dbl, u8 vd, u8 rn) {
  emit((dbl ? 0x9e670000u : 0x1e270000u) | (static_cast<u32>(rn) << 5) | vd);
}

void Arm64Emitter::patch_b_cond(usize at, usize target) {
  const i32 offset_words = static_cast<i32>(static_cast<isize>(target) -
                                            static_cast<isize>(at));
  m_code[at] = (m_code[at] & ~(0x7ffffu << 5)) |
               ((static_cast<u32>(offset_words) & 0x7ffff) << 5);
}

namespace {
// STP/LDP (64-bit): pre-index, signed offset, post-index. imm in bytes.
inline u32 ldst_pair(u32 base, u8 rt, u8 rt2, u8 rn, i32 imm) {
  return base | ((static_cast<u32>(imm / 8) & 0x7f) << 15) | (static_cast<u32>(rt2) << 10) |
         (static_cast<u32>(rn) << 5) | rt;
}
} // namespace

void Arm64Emitter::emit_prologue() {
  emit(ldst_pair(0xa9800000u, 19, 30, 31, -64)); // stp x19, x30, [sp, #-64]!
  emit(ldst_pair(0xa9000000u, 20, 21, 31, 16));  // stp x20, x21, [sp, #16]
  emit(ldst_pair(0xa9000000u, 22, 23, 31, 32));  // stp x22, x23, [sp, #32]
  emit(ldst_pair(0xa9000000u, 24, 25, 31, 48));  // stp x24, x25, [sp, #48]
  // mov x19, x0 (guest CpuContext*)
  emit(0xaa0003f3);
  // mov x20, x1 (host base of guest memory)
  emit(0xaa0103f4);
  // sub x21, x2, #8 (highest valid start address of an 8-byte access)
  emit(0xd1002055);
  // x22 = 0 (instructions retired by chained blocks), x23 = x3 (chain budget)
  movz(true, 22, 0);
  mov_reg(23, 3);
}

void Arm64Emitter::emit_epilogue() {
  emit(ldst_pair(0xa9400000u, 24, 25, 31, 48)); // ldp x24, x25, [sp, #48]
  emit(ldst_pair(0xa9400000u, 22, 23, 31, 32)); // ldp x22, x23, [sp, #32]
  emit(ldst_pair(0xa9400000u, 20, 21, 31, 16)); // ldp x20, x21, [sp, #16]
  emit(ldst_pair(0xa8c00000u, 19, 30, 31, 64)); // ldp x19, x30, [sp], #64
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
  ShiftCl,   // D3 /4 /5 /6 /7, 32/64-bit (count = CL)
  RotImm,    // C0/C1/D0/D1 /0 /1 (ROL/ROR by a constant)
  Imul,      // 0F AF, 69, 6B, 32/64-bit
  Neg,       // F6/F7 /3
  Not,       // F6/F7 /2
  SignExt,   // 98 CWDE/CDQE, 99 CDQ/CQO (32/64-bit)
  Bextr,     // VEX.LZ.0F38 F7, pp=0
  ShiftX,    // VEX.LZ.0F38 F7, pp=1/2/3 (SHLX/SARX/SHRX; alu_op = pp)
  Andn,      // VEX.LZ.0F38 F2
  Sse,       // VEX.0F scalar float / xor subset (vex_op + vex_pp select it)
  CallRel,   // E8 (block end; target known)
  CallInd,   // FF /2 (block end; target in x12)
  JmpInd,    // FF /4 (block end; target in x12)
  Ret,       // C3 (block end; target in x12)
  Div,       // F7 /6 /7, 32/64-bit (alu_op = 6 DIV / 7 IDIV); #DE cases side-exit
  Vec,       // VEX.128 packed-integer / move subset (vex_op, vex_pp, reg = map)
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
  // VEX fields (Bextr/ShiftX/Andn/Sse)
  u8 vex_op{0};
  u8 vex_pp{0};
  u8 vex_vvvv{0};
  bool vex_l{false};

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
    case Kind::ShiftCl:
    case Kind::RotImm:
    case Kind::Imul:
    case Kind::Neg:
    case Kind::Not:
    case Kind::Bextr:
    case Kind::ShiftX:
    case Kind::Andn:
    case Kind::Sse:
    case Kind::CallInd:
    case Kind::JmpInd:
    case Kind::Div:
    case Kind::Vec:
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
  if (op0 == 0xF4 || op0 == 0xCC || op0 == 0xC2)
    return true;
  if (op0 == 0x0F && (op1 == 0x05 || op1 == 0x0B))
    return true;
  return false;
}

/// The VEX packed-integer / move forms emit_vec() translates (128-bit unless
/// noted). Everything else stays with the interpreter.
bool vec_supported(u8 map, u8 op, u8 pp, bool l, const x86::ModRM &m) {
  if (map == 2) {
    if (pp != 1)
      return false;
    if ((op >= 0x20 && op <= 0x25) || (op >= 0x30 && op <= 0x35)) // VPMOVSX / VPMOVZX
      return !(l == false && (op & 0x0F) == 2); // BQ from 2 bytes stays interpreted
    return !l && (op == 0x00 || op == 0x40 || op == 0x2B); // VPSHUFB, VPMULLD, VPACKUSDW
  }
  if (map == 3) {
    if (pp != 1)
      return false;
    if (op == 0x18 || op == 0x19) // VINSERTF128 / VEXTRACTF128
      return l;
    if (op == 0x0E) // VPBLENDW
      return !l;
    if (op == 0x4A || op == 0x4B) // VBLENDVPS / VBLENDVPD
      return true;
    return false;
  }
  if (map != 1)
    return false;
  switch (op) {
  case 0x6F: case 0x7F: // VMOVDQA (66) / VMOVDQU (F3), 128 or 256-bit
    return pp == 1 || pp == 2;
  case 0x10: case 0x11: case 0x28: case 0x29: // VMOVUPS/UPD/APS/APD
    return pp <= 1;
  case 0x6E: // VMOVD/Q xmm, r/m
    return pp == 1 && !l;
  case 0x7E: // VMOVD/Q r/m, xmm (66) ; VMOVQ xmm, xmm/m64 (F3)
    return (pp == 1 || pp == 2) && !l;
  case 0xD6: // VMOVQ xmm/m64, xmm
    return pp == 1 && !l;
  case 0x70: // VPSHUFD (66), VPSHUFHW (F3), VPSHUFLW (F2)
    return pp >= 1 && !l;
  case 0x63: case 0x67: case 0x6B: // VPACKSSWB, VPACKUSWB, VPACKSSDW
    return pp == 1 && !l;
  case 0x71: case 0x72: case 0x73: // shifts by immediate (/2 /4 /6; no /4 for q)
    return pp == 1 && !l && m.is_reg && ((m.reg & 7) == 2 || (m.reg & 7) == 6 || ((m.reg & 7) == 4 && op != 0x73));
  case 0xFC: case 0xFD: case 0xFE: case 0xD4: // VPADDB/W/D/Q
  case 0xF8: case 0xF9: case 0xFA: case 0xFB: // VPSUBB/W/D/Q
  case 0xDB: case 0xDF: case 0xEB:            // VPAND / VPANDN / VPOR
  case 0xD5:                                  // VPMULLW
  case 0x60: case 0x61: case 0x62: case 0x6C: // VPUNPCKL BW/WD/DQ/QDQ
  case 0x68: case 0x69: case 0x6A: case 0x6D: // VPUNPCKH BW/WD/DQ/QDQ
    return pp == 1 && !l;
  case 0x54: case 0x55: case 0x56: // VANDPS/PD, VANDNPS/PD, VORPS/PD (also 256-bit)
    return pp <= 1;
  default:
    return false;
  }
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

  // VEX (C4 / C5): BMI1/BMI2 GPR ops and the scalar SSE subset. Legacy
  // 66/F2/F3/REX prefixes before VEX are #UD; leave those to the interpreter.
  if ((op0 == 0xC4 || op0 == 0xC5) && !p.opsize16 && !p.rep && !p.has_rex && !p.lock) {
    if (pos + 2 > avail)
      return false;
    u8 map = 1;
    bool w = false;
    const u8 b1 = code[pos++];
    p.rex_r = !(b1 & 0x80);
    u8 b2 = b1;
    if (op0 == 0xC4) {
      p.rex_x = !(b1 & 0x40);
      p.rex_b = !(b1 & 0x20);
      map = b1 & 0x1F;
      if (pos >= avail)
        return false;
      b2 = code[pos++];
      w = (b2 & 0x80) != 0;
    }
    p.rex_w = w;
    d.vex_vvvv = static_cast<u8>((~b2 >> 3) & 15);
    d.vex_l = (b2 & 4) != 0;
    d.vex_pp = b2 & 3;
    if (pos >= avail)
      return false;
    d.vex_op = code[pos++];
    if (!modrm())
      return false;
    const u8 op = d.vex_op, pp = d.vex_pp;
    if (map == 2 && op == 0xF7 && !d.vex_l) {
      d.width = w ? 64 : 32;
      if (pp == 0) {
        d.kind = Kind::Bextr;
        d.flags_written = CpuContext::FLAG_ZF | CpuContext::FLAG_CF | CpuContext::FLAG_OF;
      } else {
        d.kind = Kind::ShiftX;
        d.alu_op = pp;
      }
      return finish();
    }
    if (map == 2 && op == 0xF2 && pp == 0 && !d.vex_l) {
      d.kind = Kind::Andn;
      d.width = w ? 64 : 32;
      d.flags_written = CpuContext::FLAG_SF | CpuContext::FLAG_ZF | CpuContext::FLAG_CF |
                        CpuContext::FLAG_OF;
      return finish();
    }
    if (vec_supported(map, op, pp, d.vex_l, d.m)) {
      d.kind = Kind::Vec;
      d.reg = map;
      d.width = w ? 64 : 32; // VMOVD/VMOVQ GPR width
      if (map == 3 || (map == 1 && (op == 0x70 || op == 0x71 || op == 0x72 || op == 0x73))) {
        if (pos >= avail)
          return false;
        d.imm = code[pos++];
      }
      return finish();
    }
    if (map != 1)
      return false;
    const bool scalar = pp == 2 || pp == 3;
    bool ok = false;
    if ((op == 0x10 || op == 0x11 || op == 0x2A) && scalar)
      ok = true;
    else if ((op == 0x58 || op == 0x59 || (op >= 0x5C && op <= 0x5F)) && scalar)
      ok = true;
    else if ((op == 0x57 && pp <= 1) || (op == 0xEF && pp == 1))
      ok = !d.vex_l || d.m.is_reg; // 256-bit only from registers
    else if ((op == 0x2E || op == 0x2F) && pp <= 1) {
      ok = true;
      d.flags_written = FLAGS_ARITH | CpuContext::FLAG_AF;
    }
    if (!ok)
      return false;
    d.kind = Kind::Sse;
    d.width = w ? 64 : 32; // VCVTSI2SS/SD source width
    return finish();
  }

  // NOP / PAUSE (REX.B 90 is XCHG r8, rAX)
  if (op0 == 0x90 && !p.rex_b) {
    d.kind = Kind::Nop;
    return finish();
  }

  // CWDE/CDQE (98), CDQ/CQO (99). 16-bit forms are left to the interpreter.
  if ((op0 == 0x98 || op0 == 0x99) && op_bits != 16) {
    d.kind = Kind::SignExt;
    d.width = op_bits;
    d.reg = op0 & 1;
    return finish();
  }

  // IMUL r, r/m, imm32 / imm8 (32/64-bit)
  if ((op0 == 0x69 || op0 == 0x6B) && op_bits != 16) {
    if (!modrm() || !imm(op0 == 0x6B ? 1 : 4))
      return false;
    d.kind = Kind::Imul;
    d.width = op_bits;
    d.rm_is_dst = true; // marks the immediate form
    d.flags_written = CpuContext::FLAG_CF | CpuContext::FLAG_OF;
    return finish();
  }

  // SHL/SHR/SAR r/m32/64, CL. The count is only known at run time: a zero
  // count leaves every flag alone, so the previous flags must be in rflags.
  if (op0 == 0xD3 && op_bits != 16) {
    if (!modrm())
      return false;
    const u8 sub = d.m.reg & 7;
    if (sub < 4)
      return false;
    d.kind = Kind::ShiftCl;
    d.width = op_bits;
    d.alu_op = (sub == 6) ? 4 : sub;
    d.flags_written = FLAGS_ARITH;
    d.flags_read = FLAGS_ARITH;
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

  // ALU r/m, r and r, r/m: ADD/OR/AND/SUB/XOR/CMP; ADC/SBB at 32/64 bits
  if (op0 <= 0x3B && (op0 & 7) <= 3) {
    const u8 major = (op0 >> 3) & 7;
    if ((major == 2 || major == 3) && ((op0 & 1) == 0 || op_bits == 16))
      return false;
    if (!modrm())
      return false;
    if (major == 2 || major == 3)
      d.flags_read = CpuContext::FLAG_CF;
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
    if ((sub == 2 || sub == 3) && (op0 == 0x80 || op_bits == 16))
      return false;
    if (sub == 2 || sub == 3)
      d.flags_read = CpuContext::FLAG_CF;
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

  // DIV / IDIV r/m32/64 (F7 /6 /7). Flags are left alone, as in the interpreter.
  if (op0 == 0xF7 && op_bits != 16 && pos < avail && ((code[pos] >> 3) & 7) >= 6) {
    if (!modrm())
      return false;
    d.kind = Kind::Div;
    d.width = op_bits;
    d.alu_op = d.m.reg & 7;
    return finish();
  }

  // NOT / NEG r/m (F6/F7 /2 /3)
  if ((op0 == 0xF6 || op0 == 0xF7) && pos < avail && (((code[pos] >> 3) & 7) == 2 || ((code[pos] >> 3) & 7) == 3)) {
    if (!modrm())
      return false;
    d.width = (op0 == 0xF6) ? 8 : op_bits;
    if ((d.m.reg & 7) == 2) {
      d.kind = Kind::Not;
    } else {
      d.kind = Kind::Neg;
      d.flags_written = FLAGS_ARITH;
    }
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
    if (sub == 2 || sub == 3) // RCL / RCR
      return false;
    d.width = (op0 & 1) ? op_bits : 8;
    if (op0 <= 0xC1) {
      if (!imm(1))
        return false;
    } else {
      d.imm = 1;
    }
    d.imm &= (d.width == 64) ? 0x3F : 0x1F;
    if (sub < 2) {
      // ROL / ROR: CF always (once the count is non-zero), OF for count 1.
      d.kind = Kind::RotImm;
      d.alu_op = sub;
      d.flags_written = d.imm ? (CpuContext::FLAG_CF | (d.imm == 1 ? CpuContext::FLAG_OF : 0)) : 0;
      return finish();
    }
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
    if (op0 == 0xFF && (sub == 2 || sub == 4)) { // CALL / JMP r/m64
      if (p.opsize16)
        return false;
      d.kind = sub == 2 ? Kind::CallInd : Kind::JmpInd;
      d.width = 64;
      return finish();
    }
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

  // CALL rel32 / RET (66-prefixed forms are left to the interpreter)
  if (op0 == 0xE8 && !p.opsize16) {
    if (!imm(4))
      return false;
    d.kind = Kind::CallRel;
    finish();
    d.target = rip + d.len + d.imm;
    return true;
  }
  if (op0 == 0xC3 && !p.opsize16) {
    d.kind = Kind::Ret;
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

    // IMUL r, r/m (32/64-bit)
    if (op1 == 0xAF && op_bits != 16) {
      if (!modrm())
        return false;
      d.kind = Kind::Imul;
      d.width = op_bits;
      d.flags_written = CpuContext::FLAG_CF | CpuContext::FLAG_OF;
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
    case Kind::ShiftCl:
      emit_shift_cl(d, flags_needed);
      break;
    case Kind::RotImm:
      emit_rotate(d, flags_needed);
      break;
    case Kind::Imul:
      emit_imul(d, flags_needed);
      break;
    case Kind::Neg:
      m_e.movz(true, 0, 0);
      load_rm(1, w);
      emit_alu(ALU_SUB, w, d.flags_written, flags_needed);
      store_rm(0, w);
      break;
    case Kind::Not:
      load_rm(0, w);
      m_e.mov_imm64(1, truncate(~0ULL, w));
      m_e.eor_reg(0, 0, 1);
      store_rm(0, w);
      break;
    case Kind::SignExt:
      if (d.reg == 0) { // CWDE / CDQE
        load_gpr(0, 0, w / 2);
        m_e.sxt(true, 0, 0, static_cast<u8>(w / 2));
        store_gpr(0, w, 0);
      } else {          // CDQ / CQO
        load_gpr(0, 0, w);
        if (w == 32)
          m_e.sxt(true, 0, 0, 32);
        m_e.asr_imm(true, 0, 0, 63);
        store_gpr(2, w, 0);
      }
      break;
    case Kind::Bextr:
      emit_bextr(d, flags_needed);
      break;
    case Kind::ShiftX: {
      const bool sf = w == 64;
      load_rm(0, w);
      load_gpr(1, d.vex_vvvv, w); // the count is masked by the shift itself
      if (d.alu_op == 1)
        m_e.lslv(sf, 0, 0, 1);
      else if (d.alu_op == 2)
        m_e.asrv(sf, 0, 0, 1);
      else
        m_e.lsrv(sf, 0, 0, 1);
      store_gpr(d.m.reg, w, 0);
      break;
    }
    case Kind::Andn: {
      const bool sf = w == 64;
      load_rm(0, w);
      load_gpr(1, d.vex_vvvv, w);
      m_e.bic_shifted(sf, 0, 0, 1); // rm & ~src1
      if (flags_needed) {
        m_e.ands_shifted(sf, 31, 0, 0);
        emit_flags(false, d.flags_written, 0);
      }
      store_gpr(d.m.reg, w, 0);
      break;
    }
    case Kind::Sse:
      emit_sse(d, flags_needed);
      break;
    case Kind::CallRel:
    case Kind::CallInd:
      if (d.kind == Kind::CallInd)
        load_rm(12, 64); // target, read before RSP moves
      m_e.ldr_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      m_e.sub_imm(R_EA, R_EA, 8);
      emit_bounds_check();
      m_e.mov_imm64(0, d.rip + d.len);
      m_e.str_x_idx(0, R_GBASE, R_EA);
      m_e.str_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      break;
    case Kind::JmpInd:
      load_rm(12, 64);
      break;
    case Kind::Div:
      emit_div(d);
      break;
    case Kind::Vec:
      emit_vec(d);
      break;
    case Kind::Ret:
      m_e.ldr_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
      emit_bounds_check();
      m_e.ldr_x_idx(12, R_GBASE, R_EA);
      m_e.add_imm(R_EA, R_EA, 8);
      m_e.str_reg(R_EA, R_CTX, offsetof(CpuContext, rsp));
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
      m_e.add_reg(0, 0, 22); // plus what chained blocks before this one retired
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
    case 2:   // ADC (32/64-bit only): ARM C = x86 CF, then ADCS
    case 3: { // SBB: ARM C = !CF (ARM borrows on C clear), then SBCS
      m_e.ldr_reg(2, R_CTX, offsetof(CpuContext, rflags));
      m_e.ubfx(true, 2, 2, 0, 1);
      if (op == 3)
        m_e.emit(0xd2400042u); // eor x2, x2, #1
      m_e.cmp_imm(true, 2, 1); // C = (x2 >= 1)
      m_e.emit((op == 2 ? 0x3a000000u : 0x7a000000u) | (sf ? 0x80000000u : 0u) | (1u << 16) | (0u << 5) | 0u);
      borrow = op == 3;
      break;
    }
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

  /// SHL/SHR/SAR r32/64, CL with the interpreter's semantics: count masked to
  /// 5/6 bits; a zero count writes nothing but still zero-extends a 32-bit
  /// register; otherwise ZF/SF/PF from the result, CF = last bit out, OF =
  /// (count 1 only) SHL: MSB(result) ^ CF, SHR: MSB(source), SAR: 0.
  void emit_shift_cl(const JitInsn &d, u64 flags_needed) {
    const u8 w = d.width;
    const bool sf = w == 64;
    load_rm(0, w);
    m_e.ldrb_imm(7, R_CTX, get_gpr_offset(1));
    m_e.ubfx(true, 7, 7, 0, sf ? 6 : 5); // count
    switch (d.alu_op) {
    case 4: m_e.lslv(sf, 1, 0, 7); break;
    case 5: m_e.lsrv(sf, 1, 0, 7); break;
    default: m_e.asrv(sf, 1, 0, 7); break;
    }
    store_rm(1, w);
    if (!flags_needed)
      return;
    const usize skip = m_e.pos();
    m_e.cbz(true, 7, 0); // count 0: flags unchanged
    // CF: SHL -> bit (w - count) of the source, SHR/SAR -> bit (count - 1).
    if (d.alu_op == 4) {
      m_e.movz(true, 8, w);
      m_e.sub_reg(8, 8, 7);
    } else {
      m_e.sub_imm(8, 7, 1);
    }
    m_e.lsrv(true, 6, 0, 8);
    m_e.ubfx(true, 6, 6, 0, 1);
    // OF candidate (x10), applied only when count == 1.
    if (d.alu_op == 4) {
      m_e.ubfx(true, 10, 1, static_cast<u8>(w - 1), 1);
      m_e.eor_reg(10, 10, 6);
    } else if (d.alu_op == 5) {
      m_e.ubfx(true, 10, 0, static_cast<u8>(w - 1), 1);
    } else {
      m_e.movz(true, 10, 0);
    }
    m_e.ands_shifted(sf, 31, 1, 1); // N/Z of the result, C = V = 0
    emit_flags(false, FLAGS_ARITH, 1, 6);
    m_e.cmp_imm(true, 7, 1);
    m_e.csel(true, 10, 10, 31, 0 /* EQ */);
    m_e.ldr_reg(11, R_CTX, offsetof(CpuContext, rflags));
    m_e.bfi(true, 11, 10, 11, 1); // OF -> bit 11
    m_e.str_reg(11, R_CTX, offsetof(CpuContext, rflags));
    m_e.patch_b_cond(skip, m_e.pos());
  }

  /// ROL/ROR by a constant: CF = LSB (ROL) / MSB (ROR) of the result; for a
  /// count of 1, OF = MSB ^ CF (ROL) / MSB ^ next bit (ROR). Other flags kept.
  void emit_rotate(const JitInsn &d, u64 flags_needed) {
    const u8 w = d.width;
    const u8 c = static_cast<u8>(d.imm);
    if (c == 0) {
      if (w == 32 && d.m.is_reg) {
        load_rm(0, 32);
        store_rm(0, 32);
      }
      return;
    }
    const u8 n = c % w;
    const bool rol = d.alu_op == 0;
    load_rm(0, w);
    if (n != 0) {
      if (w >= 32) {
        m_e.ror_imm(w == 64, 0, 0, rol ? static_cast<u8>(w - n) : n);
      } else {
        const u8 left = rol ? n : static_cast<u8>(w - n);
        m_e.lsl_imm(false, 1, 0, left);
        m_e.lsr_imm(false, 0, 0, static_cast<u8>(w - left));
        m_e.orr_reg(0, 0, 1);
        m_e.ubfx(false, 0, 0, 0, w);
      }
    }
    store_rm(0, w);
    if (!flags_needed)
      return;
    m_e.ldr_reg(11, R_CTX, offsetof(CpuContext, rflags));
    m_e.ubfx(true, 6, 0, rol ? 0 : static_cast<u8>(w - 1), 1); // CF
    m_e.bfi(true, 11, 6, 0, 1);
    if (c == 1) {
      m_e.ubfx(true, 10, 0, static_cast<u8>(w - 1), 1);
      if (rol)
        m_e.eor_reg(10, 10, 6);
      else {
        m_e.ubfx(true, 8, 0, static_cast<u8>(w - 2), 1);
        m_e.eor_reg(10, 10, 8);
      }
      m_e.bfi(true, 11, 10, 11, 1);
    }
    m_e.str_reg(11, R_CTX, offsetof(CpuContext, rflags));
  }

  /// IMUL r, r/m [, imm]: the low half of the signed product; CF = OF = the
  /// product does not fit the destination. Other flags are left alone.
  void emit_imul(const JitInsn &d, u64 flags_needed) {
    const u8 w = d.width;
    load_rm(1, w);
    if (d.rm_is_dst)
      m_e.mov_imm64(0, d.imm);
    else
      load_gpr(0, d.m.reg, w);
    if (w == 32) {
      m_e.smull(2, 0, 1);
      m_e.sxt(true, 3, 2, 32);
      m_e.cmp_reg(true, 3, 2);
    } else {
      m_e.mul(true, 2, 0, 1);
      m_e.smulh(3, 0, 1);
      m_e.asr_imm(true, 4, 2, 63);
      m_e.cmp_reg(true, 3, 4);
    }
    store_gpr(d.m.reg, w, 2);
    if (!flags_needed)
      return;
    m_e.cset(true, 5, 1 /* NE */);
    m_e.ldr_reg(11, R_CTX, offsetof(CpuContext, rflags));
    m_e.bfi(true, 11, 5, 0, 1);  // CF
    m_e.bfi(true, 11, 5, 11, 1); // OF
    m_e.str_reg(11, R_CTX, offsetof(CpuContext, rflags));
  }

  /// BEXTR dst, src, ctrl: start = ctrl[7:0], len = ctrl[15:8]; ZF from the
  /// result, CF = OF = 0 (the interpreter leaves SF/PF alone).
  void emit_bextr(const JitInsn &d, u64 flags_needed) {
    const u8 w = d.width;
    const bool sf = w == 64;
    load_rm(0, w);
    load_gpr(1, d.vex_vvvv, w);
    m_e.ubfx(true, 2, 1, 0, 8);    // start
    m_e.ubfx(true, 3, 1, 8, 8);    // len
    m_e.lsrv(true, 4, 0, 2);       // src >> (start & 63)
    m_e.cmp_imm(true, 2, w);
    m_e.csel(true, 4, 31, 4, 2 /* HS */); // start >= w -> 0
    m_e.movz(true, 5, 1);
    m_e.lslv(true, 5, 5, 3);
    m_e.sub_imm(5, 5, 1);          // (1 << len) - 1
    m_e.mov_imm64(6, ~0ULL);
    m_e.cmp_imm(true, 3, 64);
    m_e.csel(true, 5, 6, 5, 2 /* HS */); // len >= 64 -> all ones
    m_e.and_reg(0, 4, 5);
    store_gpr(d.m.reg, w, 0);
    if (flags_needed) {
      m_e.ands_shifted(sf, 31, 0, 0);
      emit_flags(false, d.flags_written, 0);
    }
  }

  /// DIV / IDIV r/m32/64 with the interpreter's results; divide by zero and a
  /// quotient that does not fit side-exit, and the interpreter raises #DE.
  void emit_div(const JitInsn &d) {
    const u8 w = d.width;
    const bool sgn = d.alu_op == 7;
    auto exit_if = [&](u8 cond) {
      m_exits.push_back({m_e.pos(), m_cur->rip, m_retired});
      m_e.b_cond(cond, 0);
    };
    auto exit_if_zero = [&](u8 reg) {
      m_exits.push_back({m_e.pos(), m_cur->rip, m_retired});
      m_e.cbz(true, reg, 0);
    };
    load_rm(1, w); // divisor (zero-extended)
    exit_if_zero(1);
    if (w == 32) {
      // x0 = EDX:EAX
      m_e.ldr_w(0, R_CTX, get_gpr_offset(0));
      m_e.ldr_w(2, R_CTX, get_gpr_offset(2));
      m_e.orr_shifted(true, 0, 0, 2, Arm64Emitter::SHIFT_LSL, 32);
      if (sgn) {
        m_e.sxt(true, 1, 1, 32);
        m_e.emit(0x9ac00c00u | (1u << 16) | (0u << 5) | 3u); // sdiv x3, x0, x1
        m_e.sxt(true, 4, 3, 32);
      } else {
        m_e.emit(0x9ac00800u | (1u << 16) | (0u << 5) | 3u); // udiv x3, x0, x1
        m_e.mov_w(4, 3);
      }
      m_e.cmp_reg(true, 4, 3);
      exit_if(1 /* NE: quotient overflows 32 bits */);
    } else {
      m_e.ldr_reg(0, R_CTX, get_gpr_offset(0));
      m_e.ldr_reg(2, R_CTX, get_gpr_offset(2));
      // Only RDX = sign/zero extension of RAX (CQO / xor edx,edx) is handled
      // here; a real 128-bit dividend goes to the interpreter.
      if (sgn) {
        m_e.asr_imm(true, 4, 0, 63);
        m_e.cmp_reg(true, 2, 4);
        exit_if(1);
        // INT64_MIN / -1 overflows
        m_e.mov_imm64(5, 0x8000000000000000ULL);
        m_e.cmp_reg(true, 0, 5);
        m_e.cset(true, 6, 0 /* EQ */);
        m_e.mov_imm64(5, ~0ULL);
        m_e.cmp_reg(true, 1, 5);
        m_e.cset(true, 7, 0);
        m_e.and_reg(6, 6, 7);
        exit_if_zero(6); // placeholder: patched below to exit when x6 != 0
        m_e.invert_last_cbz(); // cbnz: exit when non-zero
        m_e.emit(0x9ac00c00u | (1u << 16) | (0u << 5) | 3u); // sdiv x3, x0, x1
      } else {
        exit_if_zero(2);
        m_e.invert_last_cbz(); // cbnz: exit when non-zero
        m_e.emit(0x9ac00800u | (1u << 16) | (0u << 5) | 3u); // udiv x3, x0, x1
      }
    }
    // remainder = dividend - quotient * divisor
    m_e.emit(0x9b008000u | (1u << 16) | (0u << 10) | (3u << 5) | 5u); // msub x5, x3, x1, x0
    store_gpr(0, w, 3);
    store_gpr(2, w, 5);
  }

  /// The VEX packed-integer / move subset (see vec_supported), with NEON.
  /// v0 = src1 (vvvv), v1 = r/m; the result is written to the destination
  /// and bits 255:128 are cleared, as every VEX.128 write does.
  void emit_vec(const JitInsn &d) {
    const u8 op = d.vex_op, pp = d.vex_pp, map = d.reg;
    const u8 dst = d.m.reg & 15, src1 = d.vex_vvvv & 15;
    auto v3 = [&](u32 base, u8 vd, u8 vn, u8 vm) {
      m_e.emit(base | (static_cast<u32>(vm) << 16) | (static_cast<u32>(vn) << 5) | vd);
    };
    auto load_rm_q = [&](u8 v, u32 half) {
      if (d.m.is_reg) {
        m_e.fp_ldr(4, v, R_CTX, ymm_off(d.m.rm) + half);
      } else if (half == 0) {
        m_e.fp_ldr_idx(4, v, R_GBASE, R_EA);
      } else {
        m_e.add_imm(10, R_EA, half);
        m_e.fp_ldr_idx(4, v, R_GBASE, 10);
      }
    };
    auto store_dst = [&](u8 v) {
      m_e.fp_str(4, v, R_CTX, ymm_off(dst));
      zero_upper(dst);
    };

    if (map == 1) {
      const bool load = op == 0x6F || op == 0x10 || op == 0x28;
      const bool store = op == 0x7F || op == 0x11 || op == 0x29;
      if (load || store) { // full-register moves, 128 or 256-bit
        const u32 bytes = d.vex_l ? 32 : 16;
        if (!d.m.is_reg)
          emit_bounds_check_wide(bytes);
        for (u32 half = 0; half < bytes; half += 16) {
          if (load) {
            load_rm_q(0, half);
            m_e.fp_str(4, 0, R_CTX, ymm_off(dst) + half);
          } else {
            m_e.fp_ldr(4, 0, R_CTX, ymm_off(dst) + half);
            if (d.m.is_reg) {
              m_e.fp_str(4, 0, R_CTX, ymm_off(d.m.rm) + half);
            } else if (half == 0) {
              m_e.fp_str_idx(4, 0, R_GBASE, R_EA);
            } else {
              m_e.add_imm(10, R_EA, half);
              m_e.fp_str_idx(4, 0, R_GBASE, 10);
            }
          }
        }
        if (!d.vex_l) {
          if (load)
            zero_upper(dst);
          else if (d.m.is_reg)
            zero_upper(d.m.rm);
        }
        return;
      }
      const bool q64 = d.width == 64;
      switch (op) {
      case 0x6E: // VMOVD/VMOVQ xmm, r/m
        load_rm(0, q64 ? 64 : 32);
        m_e.fmov_from_gpr(q64, 0, 0); // zeroes the rest of v0
        store_dst(0);
        return;
      case 0x7E:
        if (pp == 1) { // VMOVD/VMOVQ r/m, xmm
          m_e.fp_ldr(q64 ? 3 : 2, 0, R_CTX, ymm_off(dst));
          m_e.fmov_to_gpr(q64, 0, 0);
          store_rm(0, q64 ? 64 : 32);
        } else {       // VMOVQ xmm, xmm/m64
          if (d.m.is_reg)
            m_e.fp_ldr(3, 0, R_CTX, ymm_off(d.m.rm));
          else
            m_e.fp_ldr_idx(3, 0, R_GBASE, R_EA);
          store_dst(0);
        }
        return;
      case 0xD6: // VMOVQ xmm/m64, xmm
        m_e.fp_ldr(3, 0, R_CTX, ymm_off(dst));
        if (d.m.is_reg) {
          m_e.fp_str(4, 0, R_CTX, ymm_off(d.m.rm));
          zero_upper(d.m.rm);
        } else {
          m_e.fp_str_idx(3, 0, R_GBASE, R_EA);
        }
        return;
      case 0x70: { // VPSHUFD (66) / VPSHUFHW (F3) / VPSHUFLW (F2)
        if (!d.m.is_reg)
          emit_bounds_check_wide(16);
        load_rm_q(1, 0);
        if (pp == 1) {
          for (u32 i = 0; i < 4; ++i) {
            const u32 j = (d.imm >> (2 * i)) & 3;
            m_e.emit(0x6e000400u | (((i << 3) | 4u) << 16) | ((j << 2) << 11) | (1u << 5) | 0u);
          }
        } else {
          // INS Vd.H[i], Vn.H[j]: imm5 = i:010, imm4 = j:0
          const u32 base = pp == 2 ? 4 : 0; // the shuffled half
          v3(0x4ea01c00u, 0, 1, 1);         // mov v0, v1 (keeps the other half)
          for (u32 i = 0; i < 4; ++i) {
            const u32 j = base + ((d.imm >> (2 * i)) & 3);
            m_e.emit(0x6e000400u | ((((base + i) << 2) | 2u) << 16) | ((j << 1) << 11) | (1u << 5) | 0u);
          }
        }
        store_dst(0);
        return;
      }
      case 0x71: case 0x72: case 0x73: { // shifts by immediate (dest = vvvv)
        const u32 esize = op == 0x71 ? 16 : op == 0x72 ? 32 : 64;
        const u32 sub = d.m.reg & 7;
        const u32 sh = static_cast<u32>(d.imm);
        m_e.fp_ldr(4, 0, R_CTX, ymm_off(d.m.rm));
        if (sh == 0) {
          // unchanged
        } else if (sub == 6) { // left
          if (sh >= esize)
            m_e.eor_v16(0, 0, 0);
          else
            m_e.emit(0x4f005400u | ((esize + sh) << 16) | (0u << 5) | 0u);
        } else {               // 2: logical right, 4: arithmetic right
          if (sub == 2 && sh >= esize) {
            m_e.eor_v16(0, 0, 0);
          } else {
            const u32 s = std::min(sh, esize);
            m_e.emit((sub == 2 ? 0x6f000400u : 0x4f000400u) | ((2 * esize - s) << 16) | (0u << 5) | 0u);
          }
        }
        m_e.fp_str(4, 0, R_CTX, ymm_off(src1));
        zero_upper(src1);
        return;
      }
      default:
        break;
      }
    }

    if (map == 2 && ((op >= 0x20 && op <= 0x25) || (op >= 0x30 && op <= 0x35))) {
      // VPMOVSX / VPMOVZX: widen the low elements of r/m. kind: 0 BW, 1 BD,
      // 2 BQ, 3 WD, 4 WQ, 5 DQ; each step doubles the element size.
      static constexpr u8 kFrom[6] = {8, 8, 8, 16, 16, 32};
      static constexpr u8 kSteps[6] = {1, 2, 3, 1, 2, 1};
      const u8 kind = op & 0x0F;
      const u32 out = d.vex_l ? 32 : 16, in = out >> kSteps[kind];
      const bool sgn = op < 0x30;
      auto widen = [&](u8 vd, u8 vn, u32 esize, bool upper) {
        m_e.emit((sgn ? 0x0f00a400u : 0x2f00a400u) | (upper ? 0x40000000u : 0u) | ((esize / 8) << 19) |
                 (static_cast<u32>(vn) << 5) | vd);
      };
      const u8 lg = in == 16 ? 4 : in == 8 ? 3 : 2;
      if (d.m.is_reg) {
        m_e.fp_ldr(lg, 1, R_CTX, ymm_off(d.m.rm));
      } else {
        if (in > 8)
          emit_bounds_check_wide(in);
        m_e.fp_ldr_idx(lg, 1, R_GBASE, R_EA);
      }
      u32 esize = kFrom[kind];
      for (u8 st = 1; st < kSteps[kind]; ++st, esize *= 2)
        widen(1, 1, esize, false);
      widen(0, 1, esize, false);
      if (d.vex_l) {
        widen(2, 1, esize, true);
        m_e.fp_str(4, 0, R_CTX, ymm_off(dst));
        m_e.fp_str(4, 2, R_CTX, ymm_off(dst) + 16);
      } else {
        store_dst(0);
      }
      return;
    }

    if (map == 3 && (op == 0x18 || op == 0x19)) { // VINSERTF128 / VEXTRACTF128
      const u32 lane = (d.imm & 1) * 16;
      if (!d.m.is_reg)
        emit_bounds_check_wide(16);
      if (op == 0x18) {
        m_e.fp_ldr(4, 0, R_CTX, ymm_off(src1));
        m_e.fp_ldr(4, 1, R_CTX, ymm_off(src1) + 16);
        load_rm_q(2, 0);
        m_e.fp_str(4, lane ? 0 : 2, R_CTX, ymm_off(dst));
        m_e.fp_str(4, lane ? 2 : 1, R_CTX, ymm_off(dst) + 16);
      } else {
        m_e.fp_ldr(4, 0, R_CTX, ymm_off(dst) + lane);
        if (d.m.is_reg) {
          m_e.fp_str(4, 0, R_CTX, ymm_off(d.m.rm));
          zero_upper(d.m.rm);
        } else {
          m_e.fp_str_idx(4, 0, R_GBASE, R_EA);
        }
      }
      return;
    }

    if (map == 3 && (op == 0x4A || op == 0x4B)) { // VBLENDVPS / VBLENDVPD
      // dest = src2 where the mask element's sign bit is set, else src1.
      const u8 mask = static_cast<u8>((d.imm >> 4) & 15);
      const u32 halves = d.vex_l ? 2 : 1;
      if (!d.m.is_reg)
        emit_bounds_check_wide(16 * halves);
      for (u32 h = 0; h < halves; ++h) {
        m_e.fp_ldr(4, 0, R_CTX, ymm_off(src1) + 16 * h);
        load_rm_q(1, 16 * h);
        m_e.fp_ldr(4, 2, R_CTX, ymm_off(mask) + 16 * h);
        // sshr v2, v2, #esize-1: all ones where the sign bit is set
        m_e.emit(op == 0x4A ? 0x4f210442u : 0x4f410442u);
        v3(0x6ea01c00u, 0, 1, 2); // bit v0, v1, v2
        m_e.fp_str(4, 0, R_CTX, ymm_off(dst) + 16 * h);
      }
      if (!d.vex_l)
        zero_upper(dst);
      return;
    }

    if (map == 1 && d.vex_l) { // 256-bit VANDPS / VANDNPS / VORPS (and PD)
      if (!d.m.is_reg)
        emit_bounds_check_wide(32);
      for (u32 h = 0; h < 2; ++h) {
        m_e.fp_ldr(4, 0, R_CTX, ymm_off(src1) + 16 * h);
        load_rm_q(1, 16 * h);
        if (op == 0x54)
          v3(0x4e201c00u, 0, 0, 1);
        else if (op == 0x56)
          v3(0x4ea01c00u, 0, 0, 1);
        else
          v3(0x4e601c00u, 0, 1, 0);
        m_e.fp_str(4, 0, R_CTX, ymm_off(dst) + 16 * h);
      }
      return;
    }

    // Two-source ops: dest = src1 OP r/m (128-bit).
    if (!d.m.is_reg)
      emit_bounds_check_wide(16);
    m_e.fp_ldr(4, 0, R_CTX, ymm_off(src1));
    load_rm_q(1, 0);
    if (map == 3) { // VPBLENDW: word i from r/m when imm bit i is set
      u64 lo = 0, hi = 0;
      for (u32 i = 0; i < 8; ++i)
        if (d.imm & (1u << i))
          (i < 4 ? lo : hi) |= 0xFFFFULL << ((i & 3) * 16);
      m_e.mov_imm64(2, lo);
      m_e.mov_imm64(3, hi);
      m_e.fmov_from_gpr(true, 2, 2);
      m_e.emit(0x4e181c62u); // ins v2.d[1], x3
      v3(0x6ea01c00u, 0, 1, 2); // bit v0, v1, v2
      store_dst(0);
      return;
    }
    if (map == 1 && (op == 0x63 || op == 0x67 || op == 0x6B)) {
      // Saturating packs: low half from src1, high half from r/m.
      const u32 narrow = op == 0x67 ? 0x2e212800u   // sqxtun .8b, .8h
                       : op == 0x63 ? 0x0e214800u   // sqxtn  .8b, .8h
                                    : 0x0e614800u;  // sqxtn  .4h, .4s
      m_e.emit(narrow | (0u << 5) | 2u);
      m_e.emit(narrow | 0x40000000u | (1u << 5) | 2u);
      store_dst(2);
      return;
    }
    if (map == 2 && op == 0x2B) { // VPACKUSDW: sqxtun .4h / sqxtun2 .8h from .4s
      m_e.emit(0x2e612800u | (0u << 5) | 2u);
      m_e.emit(0x6e612800u | (1u << 5) | 2u);
      store_dst(2);
      return;
    }
    if (map == 2) {
      if (op == 0x00) { // VPSHUFB: index bit 7 -> 0, else src1[idx & 15]
        m_e.emit(0x4f00e400u | (4u << 16) | (0x0fu << 5) | 2u); // movi v2.16b, #0x8f
        v3(0x4e201c00u, 1, 1, 2);                              // and v1, v1, v2
        v3(0x4e000000u, 0, 0, 1);                              // tbl v0, {v0}, v1
      } else {          // VPMULLD
        v3(0x4ea09c00u, 0, 0, 1);
      }
      store_dst(0);
      return;
    }
    u32 base = 0;
    bool swap = false; // operands as (r/m, src1)
    switch (op) {
    case 0xFC: base = 0x4e208400u; break; // add .16b
    case 0xFD: base = 0x4e608400u; break; // add .8h
    case 0xFE: base = 0x4ea08400u; break; // add .4s
    case 0xD4: base = 0x4ee08400u; break; // add .2d
    case 0xF8: base = 0x6e208400u; break; // sub .16b
    case 0xF9: base = 0x6e608400u; break; // sub .8h
    case 0xFA: base = 0x6ea08400u; break; // sub .4s
    case 0xFB: base = 0x6ee08400u; break; // sub .2d
    case 0xDB: case 0x54: base = 0x4e201c00u; break; // and
    case 0xEB: case 0x56: base = 0x4ea01c00u; break; // orr
    case 0xDF: case 0x55: base = 0x4e601c00u; swap = true; break; // ~src1 & rm = bic(rm, src1)
    case 0xD5: base = 0x4e609c00u; break; // mul .8h
    case 0x60: base = 0x4e003800u; break; // zip1 .16b
    case 0x61: base = 0x4e403800u; break; // zip1 .8h
    case 0x62: base = 0x4e803800u; break; // zip1 .4s
    case 0x6C: base = 0x4ec03800u; break; // zip1 .2d
    case 0x68: base = 0x4e007800u; break; // zip2 .16b
    case 0x69: base = 0x4e407800u; break; // zip2 .8h
    case 0x6A: base = 0x4e807800u; break; // zip2 .4s
    case 0x6D: base = 0x4ec07800u; break; // zip2 .2d
    default: break;
    }
    if (swap)
      v3(base, 0, 1, 0);
    else
      v3(base, 0, 0, 1);
    store_dst(0);
  }

  static u32 ymm_off(u8 reg) {
    return static_cast<u32>(offsetof(CpuContext, ymm) + (reg & 15) * sizeof(CpuContext::YmmReg));
  }

  /// Zero bits 255:128 of ymm[reg] (every VEX.128 / scalar write does).
  void zero_upper(u8 reg) {
    m_e.str_zero(R_CTX, ymm_off(reg) + 16);
    m_e.str_zero(R_CTX, ymm_off(reg) + 24);
  }

  /// Side exit unless [x9, x9 + bytes) lies inside guest memory (the generic
  /// check covers 8 bytes).
  void emit_bounds_check_wide(u32 bytes) {
    m_e.add_imm(10, R_EA, bytes - 8);
    m_e.cmp_reg(true, 10, R_GLIMIT);
    m_exits.push_back({m_e.pos(), m_cur->rip, m_retired});
    m_e.b_cond(COND_HI, 0);
  }

  /// The VEX scalar float / xor subset, with the interpreter's semantics:
  /// scalar ops take bits 127:N from src1 (vvvv) and zero 255:128; created
  /// NaNs are x86's negative "indefinite"; MIN/MAX return src2 unless src1
  /// compares strictly less/greater; UCOMISS/COMISS set ZF/PF/CF and clear
  /// OF/SF/AF.
  void emit_sse(const JitInsn &d, u64 flags_needed) {
    const u8 op = d.vex_op, pp = d.vex_pp;
    const u8 dst = d.m.reg & 15, src1 = d.vex_vvvv & 15;
    const bool dbl = pp == 3 || ((op == 0x2E || op == 0x2F) && pp == 1);
    const u8 lg = dbl ? 3 : 2; // element size log2
    auto load_src2 = [&](u8 v, u8 size_log2) {
      if (d.m.is_reg)
        m_e.fp_ldr(size_log2, v, R_CTX, ymm_off(d.m.rm));
      else
        m_e.fp_ldr_idx(size_log2, v, R_GBASE, R_EA);
    };

    if (op == 0x10 || op == 0x11) { // VMOVSS / VMOVSD
      const bool load = op == 0x10;
      if (!d.m.is_reg) {
        if (load) {
          m_e.fp_ldr_idx(lg, 0, R_GBASE, R_EA); // zeroes the rest of v0
          m_e.fp_str(4, 0, R_CTX, ymm_off(dst));
          zero_upper(dst);
        } else {
          m_e.fp_ldr(lg, 0, R_CTX, ymm_off(dst));
          m_e.fp_str_idx(lg, 0, R_GBASE, R_EA);
        }
        return;
      }
      // Register form: dest = src1[127:N] : other[N-1:0].
      const u8 to = load ? dst : static_cast<u8>(d.m.rm & 15);
      const u8 from = load ? static_cast<u8>(d.m.rm & 15) : dst;
      m_e.fp_ldr(4, 0, R_CTX, ymm_off(src1));
      m_e.fp_ldr(lg, 1, R_CTX, ymm_off(from));
      m_e.ins_lane0(dbl, 0, 1);
      m_e.fp_str(4, 0, R_CTX, ymm_off(to));
      zero_upper(to);
      return;
    }

    if (op == 0x57 || op == 0xEF) { // VXORPS / VXORPD / VPXOR
      if (!d.m.is_reg)
        emit_bounds_check_wide(16);
      for (u8 half = 0; half < (d.vex_l ? 2 : 1); ++half) {
        m_e.fp_ldr(4, 0, R_CTX, ymm_off(src1) + half * 16);
        if (d.m.is_reg)
          m_e.fp_ldr(4, 1, R_CTX, ymm_off(d.m.rm) + half * 16);
        else
          m_e.fp_ldr_idx(4, 1, R_GBASE, R_EA);
        m_e.eor_v16(0, 0, 1);
        m_e.fp_str(4, 0, R_CTX, ymm_off(dst) + half * 16);
      }
      if (!d.vex_l)
        zero_upper(dst);
      return;
    }

    if (op == 0x2E || op == 0x2F) { // (V)UCOMISS/SD, (V)COMISS/SD
      if (!flags_needed)
        return;
      m_e.fp_ldr(lg, 0, R_CTX, ymm_off(dst));
      load_src2(1, lg);
      m_e.fcmp(dbl, 0, 1);
      m_e.cset(true, 3, 11 /* LT: less or unordered */);
      m_e.cset(true, 4, 0 /* EQ */);
      m_e.cset(true, 5, 6 /* VS: unordered */);
      m_e.orr_reg(4, 4, 5);
      m_e.bfi(true, 3, 5, 2, 1); // PF
      m_e.bfi(true, 3, 4, 6, 1); // ZF
      m_e.ldr_reg(4, R_CTX, offsetof(CpuContext, rflags));
      m_e.mov_imm64(5, FLAGS_ARITH | CpuContext::FLAG_AF);
      m_e.bic_shifted(true, 4, 4, 5);
      m_e.orr_reg(4, 4, 3);
      m_e.str_reg(4, R_CTX, offsetof(CpuContext, rflags));
      return;
    }

    m_e.fp_ldr(4, 0, R_CTX, ymm_off(src1)); // v0 = src1
    if (op == 0x2A) {                         // VCVTSI2SS / VCVTSI2SD
      load_rm(1, d.width);
      m_e.scvtf(dbl, d.width == 64, 2, 1);
    } else {
      load_src2(1, lg);
      switch (op) {
      case 0x5D: // MIN: src1 < src2 ? src1 : src2
        m_e.fcmp(dbl, 0, 1);
        m_e.fcsel(dbl, 2, 0, 1, 4 /* MI */);
        break;
      case 0x5F: // MAX: src1 > src2 ? src1 : src2
        m_e.fcmp(dbl, 0, 1);
        m_e.fcsel(dbl, 2, 0, 1, 12 /* GT */);
        break;
      default: {
        const u8 fop = op == 0x58 ? Arm64Emitter::FP_ADD
                     : op == 0x59 ? Arm64Emitter::FP_MUL
                     : op == 0x5C ? Arm64Emitter::FP_SUB : Arm64Emitter::FP_DIV;
        m_e.fp_arith(dbl, fop, 2, 0, 1);
        // A NaN created from two non-NaN inputs is x86's negative QNaN.
        m_e.fcmp(dbl, 0, 1);
        m_e.cset(true, 3, 7 /* VC: inputs ordered */);
        m_e.fcmp(dbl, 2, 2);
        m_e.cset(true, 4, 6 /* VS: result NaN */);
        m_e.and_reg(3, 3, 4);
        m_e.fmov_to_gpr(dbl, 5, 2);
        m_e.mov_imm64(6, dbl ? 0xFFF8000000000000ULL : 0xFFC00000ULL);
        m_e.cmp_imm(true, 3, 0);
        m_e.csel(true, 5, 6, 5, 1 /* NE */);
        m_e.fmov_from_gpr(dbl, 2, 5);
        break;
      }
      }
    }
    m_e.ins_lane0(dbl, 0, 2);
    m_e.fp_str(4, 0, R_CTX, ymm_off(dst));
    zero_upper(dst);
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
  m_chain.resize(kChainSlots);
  chain_clear();
}

void JitCompiler::add_chain_barrier(GuestAddr rip) {
  std::unique_lock lock(m_cache_mutex);
  m_barriers.insert(rip);
}

void JitCompiler::chain_clear() {
  for (ChainSlot &s : m_chain) {
    __atomic_store_n(&s.rip, ~0ULL, __ATOMIC_RELAXED);
    __atomic_store_n(&s.entry, 0ULL, __ATOMIC_RELAXED);
  }
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// Readers load {rip, entry} with one atomic LDP; invalidate the slot before
// changing its entry so no reader can pair a new entry with an old rip.
void JitCompiler::chain_insert(GuestAddr rip, const void *entry) {
  ChainSlot &s = m_chain[(rip ^ (rip >> 16)) & (kChainSlots - 1)];
  __atomic_store_n(&s.rip, ~0ULL, __ATOMIC_RELEASE);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __atomic_store_n(&s.entry, reinterpret_cast<u64>(entry), __ATOMIC_RELEASE);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __atomic_store_n(&s.rip, rip, __ATOMIC_RELEASE);
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
      m_failed(std::move(other.m_failed)),
      m_chain(std::move(other.m_chain)) {
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
    m_chain = std::move(other.m_chain);

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
  chain_clear();
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
  return __atomic_load_n(&m_total_invocations, __ATOMIC_RELAXED);
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
    if (current_rip != guest_rip && m_barriers.count(current_rip) != 0)
      break; // a barrier starts its own block
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
    if (insn.kind == Kind::Jmp || insn.kind == Kind::CallRel) {
      current_rip = insn.target;
      block_ended = true;
      break;
    }
    if (insn.kind == Kind::Ret || insn.kind == Kind::JmpInd || insn.kind == Kind::CallInd) {
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
  m_emitter.mov_imm64(24, reinterpret_cast<u64>(m_chain.data()));
  const usize chain_entry_pos = m_emitter.pos(); // chained blocks enter here
  BlockTranslator translator(m_emitter);
  for (usize i = 0; i < insns.size(); ++i) {
    translator.emit(insns[i], flags_needed[i], i);
  }

  const usize instructions = insns.size();

  // Always update RIP at end of block.
  const Kind last = insns.back().kind;
  if (last == Kind::Jcc) {
    emit_jcc_rip(m_emitter, insns.back());
  } else if (last == Kind::Ret || last == Kind::JmpInd || last == Kind::CallInd) {
    m_emitter.mov_reg(0, 12);
  } else {
    m_emitter.mov_imm64(0, current_rip);
  }
  m_emitter.str_reg(0, 19, offsetof(CpuContext, rip));
  // x22 += retired; while under the chain budget (x23), continue straight
  // into the next block if it is compiled: chain[(rip ^ rip >> 16) & 0xffff].
  m_emitter.add_imm(22, 22, static_cast<u32>(instructions));
  m_emitter.cmp_reg(true, 22, 23);
  const usize to_return = m_emitter.pos();
  m_emitter.b_cond(2 /* HS */, 0);
  m_emitter.eor_shifted(true, 1, 0, 0, Arm64Emitter::SHIFT_LSR, 16);
  m_emitter.ubfx(true, 1, 1, 0, 16);
  m_emitter.add_shifted(true, 1, 24, 1, Arm64Emitter::SHIFT_LSL, 4);
  m_emitter.emit(0xa9400c22); // ldp x2, x3, [x1]
  m_emitter.cmp_reg(true, 2, 0);
  const usize miss = m_emitter.pos();
  m_emitter.b_cond(1 /* NE */, 0);
  m_emitter.emit(0xd61f0060); // br x3
  m_emitter.patch_b_cond(to_return, m_emitter.pos());
  m_emitter.patch_b_cond(miss, m_emitter.pos());
  m_emitter.mov_reg(0, 22);
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
  if (m_barriers.count(guest_rip) == 0)
    chain_insert(guest_rip, static_cast<const u8 *>(code_ptr) + chain_entry_pos * sizeof(u32));
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
                                usize *retired, usize chain_budget) {
  if (retired)
    *retired = 0;
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
    // __atomic builtins: the macos-14 CI runner's libc++ has no C++20
    // atomic reference type.
    __atomic_fetch_add(exec_count, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&m_total_invocations, 1, __ATOMIC_RELAXED);
  }

  // Enforce execute mode for MAP_JIT on Apple Silicon (once per thread).
  if (!t_jit_executable) {
    pthread_jit_write_protect_np(1);
    t_jit_executable = true;
  }

  // Compiled code stores to guest memory: the MemoryManager is only const
  // because compilation does not modify it.
  u8 *guest_base = const_cast<u8 *>(memory.base_address());
  const u64 n = fn(&ctx, guest_base, memory.total_size(), chain_budget);
  if (retired)
    *retired = static_cast<usize>(n);
  // A block that side-exited on its first instruction made no progress: the
  // caller must let the interpreter execute that instruction.
  return n != 0;
}

} // namespace ps4
