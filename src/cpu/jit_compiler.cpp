#include "ps4/jit_compiler.hpp"
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
  const u16 w0 = static_cast<u16>(val & 0xFFFF);
  const u16 w1 = static_cast<u16>((val >> 16) & 0xFFFF);
  const u16 w2 = static_cast<u16>((val >> 32) & 0xFFFF);
  const u16 w3 = static_cast<u16>((val >> 48) & 0xFFFF);

  // movz rd, w0, lsl 0
  emit(0xd2800000 | (static_cast<u32>(w0) << 5) | rd);
  if (w1 != 0 || w2 != 0 || w3 != 0) {
    // movk rd, w1, lsl 16
    emit(0xf2a00000 | (static_cast<u32>(w1) << 5) | rd);
  }
  if (w2 != 0 || w3 != 0) {
    // movk rd, w2, lsl 32
    emit(0xf2c00000 | (static_cast<u32>(w2) << 5) | rd);
  }
  if (w3 != 0) {
    // movk rd, w3, lsl 48
    emit(0xf2e00000 | (static_cast<u32>(w3) << 5) | rd);
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

void Arm64Emitter::emit_prologue() {
  // stp x19, x30, [sp, -16]!
  emit(0xa9bf7bf3);
  // mov x19, x0 (save context pointer in callee-saved register x19)
  emit(0xaa0003f3);
}

void Arm64Emitter::emit_epilogue() {
  // ldp x19, x30, [sp], 16
  emit(0xa8c17bf3);
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
      m_metrics(std::move(other.m_metrics)) {
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
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  return m_blocks.find(guest_rip) != m_blocks.end();
}

void JitCompiler::clear_cache() {
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  m_blocks.clear();
  m_metrics.clear();
  m_code_buffer_offset = 0;
}

usize JitCompiler::cached_block_count() const noexcept {
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  return m_blocks.size();
}

usize JitCompiler::code_buffer_used() const noexcept {
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  return m_code_buffer_offset;
}

usize JitCompiler::total_invocations() const noexcept {
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  return m_total_invocations;
}

const JitBlockMetrics *
JitCompiler::get_block_metrics(GuestAddr guest_rip) const {
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  auto it = m_metrics.find(guest_rip);
  if (it != m_metrics.end()) {
    return &it->second;
  }
  return nullptr;
}

JittedBlockFunc JitCompiler::compile_block(GuestAddr guest_rip,
                                           const MemoryManager &memory) {
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  return compile_block_locked(guest_rip, memory);
}

JittedBlockFunc JitCompiler::compile_block_locked(GuestAddr guest_rip,
                                                  const MemoryManager &memory) {
  if (auto it = m_blocks.find(guest_rip); it != m_blocks.end()) {
    return it->second;
  }

  if (!m_code_buffer || !memory.is_valid_address(guest_rip, 1)) {
    return nullptr;
  }

  m_emitter.clear();
  m_emitter.emit_prologue();

  GuestAddr current_rip = guest_rip;
  usize instructions = 0;
  bool block_ended = false;

  while (!block_ended && instructions < 128) {
    if (!memory.is_valid_address(current_rip, 1)) {
      break;
    }

    const u8 *code = memory.to_host_ptr<u8>(current_rip);
    if (!code)
      break;

    usize cursor = 0;
    bool rex_w = false, rex_r = false, rex_b = false;
    bool has_rex = false;

    // Decode legacy prefixes and REX prefix
    while (cursor < 15 && memory.is_valid_address(current_rip + cursor, 1)) {
      const u8 byte = code[cursor];
      if (byte >= 0x40 && byte <= 0x4F) {
        has_rex = true;
        rex_w = (byte & 0x08) != 0;
        rex_r = (byte & 0x04) != 0;
        rex_b = (byte & 0x01) != 0;
        cursor++;
      } else {
        break;
      }
    }

    if (!memory.is_valid_address(current_rip + cursor, 1)) {
      break;
    }

    const u8 op0 = code[cursor];

    // 0x90: NOP
    if (op0 == 0x90 && !has_rex) {
      m_emitter.nop();
      current_rip += cursor + 1;
      instructions++;
      continue;
    }

    // Multi-byte NOP: 0x0F 0x1F /0
    if (op0 == 0x0F && memory.is_valid_address(current_rip + cursor, 3) &&
        code[cursor + 1] == 0x1F) {
      const u8 modrm = code[cursor + 2];
      const u8 mod = (modrm >> 6) & 3;
      usize len = 3;
      if (mod == 1)
        len = 4;
      else if (mod == 2)
        len = 7;
      if (memory.is_valid_address(current_rip + cursor, len)) {
        m_emitter.nop();
        current_rip += cursor + len;
        instructions++;
        continue;
      }
    }

    // 0xF4: HLT (end block without advancing RIP so interpreter steps into it)
    if (op0 == 0xF4) {
      block_ended = true;
      break;
    }

    // 0x0F 0x05: SYSCALL (end block without advancing RIP so interpreter steps
    // into it)
    if (op0 == 0x0F && memory.is_valid_address(current_rip + cursor, 2) &&
        code[cursor + 1] == 0x05) {
      block_ended = true;
      break;
    }

    // 0x0F 0x0B: UD2 (end block without advancing RIP so interpreter steps into
    // it)
    if (op0 == 0x0F && memory.is_valid_address(current_rip + cursor, 2) &&
        code[cursor + 1] == 0x0B) {
      block_ended = true;
      break;
    }

    // 0xCC: INT3 (end block without advancing RIP so interpreter steps into it)
    if (op0 == 0xCC) {
      block_ended = true;
      break;
    }

    // 0xC3: RET (end block without advancing RIP so interpreter steps into it)
    if (op0 == 0xC3) {
      block_ended = true;
      break;
    }

    // 0xE8: CALL rel32 (end block without advancing RIP so interpreter steps
    // into it)
    if (op0 == 0xE8) {
      block_ended = true;
      break;
    }

    // Conditional jumps: 0x70 .. 0x7F and 0x0F 0x80 .. 0x8F (end block so
    // interpreter steps)
    if ((op0 >= 0x70 && op0 <= 0x7F) ||
        (op0 == 0x0F && memory.is_valid_address(current_rip + cursor, 2) &&
         code[cursor + 1] >= 0x80 && code[cursor + 1] <= 0x8F)) {
      block_ended = true;
      break;
    }

    // 0xEB: JMP rel8
    if (op0 == 0xEB && memory.is_valid_address(current_rip + cursor, 2)) {
      const i8 rel = static_cast<i8>(code[cursor + 1]);
      const GuestAddr target = current_rip + cursor + 2 + rel;
      current_rip = target;
      instructions++;
      block_ended = true;
      break;
    }

    // 0xE9: JMP rel32
    if (op0 == 0xE9 && memory.is_valid_address(current_rip + cursor, 5)) {
      i32 rel = 0;
      std::memcpy(&rel, code + cursor + 1, 4);
      const GuestAddr target = current_rip + cursor + 5 + rel;
      current_rip = target;
      instructions++;
      block_ended = true;
      break;
    }

    // 0xB8 - 0xBF: MOV reg32, imm32 OR (with REX.W) MOVABS reg64, imm64
    if (op0 >= 0xB8 && op0 <= 0xBF) {
      const u8 reg_idx = (op0 - 0xB8) | (rex_b ? 8 : 0);
      const u32 off = get_gpr_offset(reg_idx);

      if (rex_w) {
        // 64-bit immediate (9 bytes after cursor)
        if (memory.is_valid_address(current_rip + cursor, 9)) {
          u64 imm64 = 0;
          std::memcpy(&imm64, code + cursor + 1, sizeof(imm64));
          m_emitter.mov_imm64(0, imm64);
          m_emitter.str_reg(0, 19, off);
          current_rip += cursor + 9;
          instructions++;
          continue;
        }
      } else {
        // 32-bit immediate (5 bytes after cursor)
        if (memory.is_valid_address(current_rip + cursor, 5)) {
          u32 imm32 = 0;
          std::memcpy(&imm32, code + cursor + 1, sizeof(imm32));
          m_emitter.mov_imm64(0, imm32); // zero-extends to 64-bit
          m_emitter.str_reg(0, 19, off);
          current_rip += cursor + 5;
          instructions++;
          continue;
        }
      }
    }

    // 0xC7 /0: MOV r/m, imm32
    if (op0 == 0xC7 && memory.is_valid_address(current_rip + cursor, 6)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3 && ((modrm >> 3) & 7) == 0) { // mod == 3, reg == 0
        const u8 reg_idx = (modrm & 7) | (rex_b ? 8 : 0);
        const u32 off = get_gpr_offset(reg_idx);
        i32 imm32 = 0;
        std::memcpy(&imm32, code + cursor + 2, sizeof(imm32));
        const u64 imm_val = rex_w ? static_cast<u64>(static_cast<i64>(imm32))
                                  : static_cast<u64>(static_cast<u32>(imm32));
        m_emitter.mov_imm64(0, imm_val);
        m_emitter.str_reg(0, 19, off);
        current_rip += cursor + 6;
        instructions++;
        continue;
      }
    }

    // 0x89: MOV r/m, reg (reg to reg)
    if (op0 == 0x89 && memory.is_valid_address(current_rip + cursor, 2)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3) {
        const u8 dst_idx = (modrm & 7) | (rex_b ? 8 : 0);
        const u8 src_idx = ((modrm >> 3) & 7) | (rex_r ? 8 : 0);
        const u32 off_dst = get_gpr_offset(dst_idx);
        const u32 off_src = get_gpr_offset(src_idx);

        if (rex_w)
          m_emitter.ldr_reg(0, 19, off_src);
        else
          m_emitter.ldr_w(0, 19,
                          off_src); // Zero-extends 32-bit W0 to 64-bit X0
        m_emitter.str_reg(0, 19, off_dst);
        current_rip += cursor + 2;
        instructions++;
        continue;
      }
    }

    // 0x8B: MOV reg, r/m (reg to reg)
    if (op0 == 0x8B && memory.is_valid_address(current_rip + cursor, 2)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3) {
        const u8 dst_idx = ((modrm >> 3) & 7) | (rex_r ? 8 : 0);
        const u8 src_idx = (modrm & 7) | (rex_b ? 8 : 0);
        const u32 off_dst = get_gpr_offset(dst_idx);
        const u32 off_src = get_gpr_offset(src_idx);

        if (rex_w)
          m_emitter.ldr_reg(0, 19, off_src);
        else
          m_emitter.ldr_w(0, 19,
                          off_src); // Zero-extends 32-bit W0 to 64-bit X0
        m_emitter.str_reg(0, 19, off_dst);
        current_rip += cursor + 2;
        instructions++;
        continue;
      }
    }
    [[maybe_unused]] auto next_insn_reads_flags = [&](usize insn_len) -> bool {
      GuestAddr next_rip = current_rip + cursor + insn_len;
      if (!memory.is_valid_address(next_rip, 2))
        return true;
      const u8 *next_code = memory.to_host_ptr<u8>(next_rip);
      usize nc = 0;
      while (nc < 4 && (next_code[nc] == 0x66 ||
                        (next_code[nc] >= 0x40 && next_code[nc] <= 0x4F)))
        nc++;
      const u8 n0 = next_code[nc];
      const u8 n1 = next_code[nc + 1];
      if (n0 >= 0x70 && n0 <= 0x7F)
        return true; // Jcc rel8
      if (n0 == 0x0F &&
          ((n1 >= 0x80 && n1 <= 0x8F) || (n1 >= 0x90 && n1 <= 0x9F) ||
           (n1 >= 0x40 && n1 <= 0x4F)))
        return true;
      return false;
    };

    // 0x31 / 0x33: XOR
    if ((op0 == 0x31 || op0 == 0x33) &&
        memory.is_valid_address(current_rip + cursor, 2)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3) {
        const u8 dst_idx = (op0 == 0x31)
                               ? ((modrm & 7) | (rex_b ? 8 : 0))
                               : (((modrm >> 3) & 7) | (rex_r ? 8 : 0));
        const u8 src_idx = (op0 == 0x31)
                               ? (((modrm >> 3) & 7) | (rex_r ? 8 : 0))
                               : ((modrm & 7) | (rex_b ? 8 : 0));
        const u32 off_dst = get_gpr_offset(dst_idx);
        const u32 off_src = get_gpr_offset(src_idx);

        if (dst_idx == src_idx) {
          m_emitter.str_zero(19, off_dst);
        } else {
          m_emitter.ldr_reg(0, 19, off_dst);
          m_emitter.ldr_reg(1, 19, off_src);
          m_emitter.eor_reg(0, 0, 1);
          m_emitter.str_reg(0, 19, off_dst);
        }
        current_rip += cursor + 2;
        instructions++;
        continue;
      }
    }

    // 0x01 / 0x03: ADD
    if ((op0 == 0x01 || op0 == 0x03) &&
        memory.is_valid_address(current_rip + cursor, 2)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3) {
        const u8 dst_idx = (op0 == 0x01)
                               ? ((modrm & 7) | (rex_b ? 8 : 0))
                               : (((modrm >> 3) & 7) | (rex_r ? 8 : 0));
        const u8 src_idx = (op0 == 0x01)
                               ? (((modrm >> 3) & 7) | (rex_r ? 8 : 0))
                               : ((modrm & 7) | (rex_b ? 8 : 0));
        const u32 off_dst = get_gpr_offset(dst_idx);
        const u32 off_src = get_gpr_offset(src_idx);

        m_emitter.ldr_reg(0, 19, off_dst);
        m_emitter.ldr_reg(1, 19, off_src);
        m_emitter.add_reg(0, 0, 1);
        m_emitter.str_reg(0, 19, off_dst);
        current_rip += cursor + 2;
        instructions++;
        continue;
      }
    }

    // 0x29 / 0x2B: SUB
    if ((op0 == 0x29 || op0 == 0x2B) &&
        memory.is_valid_address(current_rip + cursor, 2)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3) {
        const u8 dst_idx = (op0 == 0x29)
                               ? ((modrm & 7) | (rex_b ? 8 : 0))
                               : (((modrm >> 3) & 7) | (rex_r ? 8 : 0));
        const u8 src_idx = (op0 == 0x29)
                               ? (((modrm >> 3) & 7) | (rex_r ? 8 : 0))
                               : ((modrm & 7) | (rex_b ? 8 : 0));
        const u32 off_dst = get_gpr_offset(dst_idx);
        const u32 off_src = get_gpr_offset(src_idx);

        m_emitter.ldr_reg(0, 19, off_dst);
        m_emitter.ldr_reg(1, 19, off_src);
        m_emitter.sub_reg(0, 0, 1);
        m_emitter.str_reg(0, 19, off_dst);
        current_rip += cursor + 2;
        instructions++;
        continue;
      }
    }

    // 0x21 / 0x23: AND
    if ((op0 == 0x21 || op0 == 0x23) &&
        memory.is_valid_address(current_rip + cursor, 2)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3) {
        const u8 dst_idx = (op0 == 0x21)
                               ? ((modrm & 7) | (rex_b ? 8 : 0))
                               : (((modrm >> 3) & 7) | (rex_r ? 8 : 0));
        const u8 src_idx = (op0 == 0x21)
                               ? (((modrm >> 3) & 7) | (rex_r ? 8 : 0))
                               : ((modrm & 7) | (rex_b ? 8 : 0));
        const u32 off_dst = get_gpr_offset(dst_idx);
        const u32 off_src = get_gpr_offset(src_idx);

        m_emitter.ldr_reg(0, 19, off_dst);
        m_emitter.ldr_reg(1, 19, off_src);
        m_emitter.and_reg(0, 0, 1);
        m_emitter.str_reg(0, 19, off_dst);
        current_rip += cursor + 2;
        instructions++;
        continue;
      }
    }

    // 0x09 / 0x0B: OR
    if ((op0 == 0x09 || op0 == 0x0B) &&
        memory.is_valid_address(current_rip + cursor, 2)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3) {
        const u8 dst_idx = (op0 == 0x09)
                               ? ((modrm & 7) | (rex_b ? 8 : 0))
                               : (((modrm >> 3) & 7) | (rex_r ? 8 : 0));
        const u8 src_idx = (op0 == 0x09)
                               ? (((modrm >> 3) & 7) | (rex_r ? 8 : 0))
                               : ((modrm & 7) | (rex_b ? 8 : 0));
        const u32 off_dst = get_gpr_offset(dst_idx);
        const u32 off_src = get_gpr_offset(src_idx);

        m_emitter.ldr_reg(0, 19, off_dst);
        m_emitter.ldr_reg(1, 19, off_src);
        m_emitter.orr_reg(0, 0, 1);
        m_emitter.str_reg(0, 19, off_dst);
        current_rip += cursor + 2;
        instructions++;
        continue;
      }
    }

    // 0x83: ALU r/m with imm8
    if (op0 == 0x83 && memory.is_valid_address(current_rip + cursor, 3)) {
      const u8 modrm = code[cursor + 1];
      if ((modrm >> 6) == 3) {
        const u8 reg_idx = (modrm & 7) | (rex_b ? 8 : 0);
        const u8 subop = (modrm >> 3) & 7;
        const i8 imm8 = static_cast<i8>(code[cursor + 2]);
        const u32 off = get_gpr_offset(reg_idx);

        if (subop == 0) { // ADD
          m_emitter.ldr_reg(0, 19, off);
          if (imm8 >= 0)
            m_emitter.add_imm(0, 0, static_cast<u32>(imm8));
          else
            m_emitter.sub_imm(0, 0, static_cast<u32>(-imm8));
          m_emitter.str_reg(0, 19, off);
          current_rip += cursor + 3;
          instructions++;
          continue;
        } else if (subop == 5) { // SUB
          m_emitter.ldr_reg(0, 19, off);
          if (imm8 >= 0)
            m_emitter.sub_imm(0, 0, static_cast<u32>(imm8));
          else
            m_emitter.add_imm(0, 0, static_cast<u32>(-imm8));
          m_emitter.str_reg(0, 19, off);
          current_rip += cursor + 3;
          instructions++;
          continue;
        }
      }
    }

    // Unknown / unsupported instruction: stop compiling block
    break;
  }

  if (instructions == 0) {
    return nullptr; // Interpreter fallback
  }

  // Always update RIP at end of block
  m_emitter.mov_imm64(0, current_rip);
  m_emitter.str_reg(0, 19, offsetof(CpuContext, rip));

  m_emitter.emit_epilogue();

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

bool JitCompiler::execute_block(CpuContext &ctx, const MemoryManager &memory) {
  JittedBlockFunc fn = nullptr;
  {
    std::lock_guard<std::mutex> lock(m_cache_mutex);
    fn = compile_block_locked(ctx.rip, memory);
    if (!fn) {
      return false;
    }

    m_metrics[ctx.rip].execution_count++;
    m_total_invocations++;
  }

  // Enforce execute mode for MAP_JIT on Apple Silicon
  pthread_jit_write_protect_np(1);

  fn(&ctx);
  return true;
}

} // namespace ps4
