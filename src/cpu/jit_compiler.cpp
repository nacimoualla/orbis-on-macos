#include "ps4/jit_compiler.hpp"
#include <cstring>
#include <iostream>
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>

namespace ps4 {

// =============================================================================
// Arm64Emitter Implementation
// =============================================================================

void Arm64Emitter::nop() {
    emit(0xd503201f);
}

void Arm64Emitter::ret() {
    emit(0xd65f03c0);
}

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

void Arm64Emitter::str_zero(u8 rn, u32 offset_bytes) {
    str_reg(31, rn, offset_bytes); // xzr = 31
}

void Arm64Emitter::add_reg(u8 rd, u8 rn, u8 rm) {
    emit(0x8b000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::sub_reg(u8 rd, u8 rn, u8 rm) {
    emit(0xcb000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::add_imm(u8 rd, u8 rn, u32 imm12) {
    emit(0x91000000 | ((imm12 & 0xFFF) << 10) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::sub_imm(u8 rd, u8 rn, u32 imm12) {
    emit(0xd1000000 | ((imm12 & 0xFFF) << 10) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::and_reg(u8 rd, u8 rn, u8 rm) {
    emit(0x8a000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::orr_reg(u8 rd, u8 rn, u8 rm) {
    emit(0xaa000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | rd);
}

void Arm64Emitter::eor_reg(u8 rd, u8 rn, u8 rm) {
    emit(0xca000000 | (static_cast<u32>(rm) << 16) | (static_cast<u32>(rn) << 5) | rd);
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
        case 0: return offsetof(CpuContext, rax);
        case 1: return offsetof(CpuContext, rcx);
        case 2: return offsetof(CpuContext, rdx);
        case 3: return offsetof(CpuContext, rbx);
        case 4: return offsetof(CpuContext, rsp);
        case 5: return offsetof(CpuContext, rbp);
        case 6: return offsetof(CpuContext, rsi);
        case 7: return offsetof(CpuContext, rdi);
        case 8: return offsetof(CpuContext, r8);
        case 9: return offsetof(CpuContext, r9);
        case 10: return offsetof(CpuContext, r10);
        case 11: return offsetof(CpuContext, r11);
        case 12: return offsetof(CpuContext, r12);
        case 13: return offsetof(CpuContext, r13);
        case 14: return offsetof(CpuContext, r14);
        case 15: return offsetof(CpuContext, r15);
        default: return offsetof(CpuContext, rax);
    }
}

} // namespace

JitCompiler::JitCompiler(usize code_cache_size)
    : m_code_buffer_size(code_cache_size) {
    m_code_buffer = static_cast<u8*>(::mmap(
        nullptr,
        m_code_buffer_size,
        PROT_READ | PROT_WRITE | PROT_EXEC,
        MAP_ANON | MAP_PRIVATE | MAP_JIT,
        -1,
        0
    ));

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

JitCompiler::JitCompiler(JitCompiler&& other) noexcept
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

JitCompiler& JitCompiler::operator=(JitCompiler&& other) noexcept {
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

void* JitCompiler::allocate_code(usize size) {
    if (!m_code_buffer) return nullptr;

    // Align to 16 bytes
    m_code_buffer_offset = (m_code_buffer_offset + 15) & ~15ULL;
    if (m_code_buffer_offset + size > m_code_buffer_size) {
        return nullptr; // Code cache exhausted
    }

    void* ptr = m_code_buffer + m_code_buffer_offset;
    m_code_buffer_offset += size;
    return ptr;
}

void JitCompiler::commit_code(void* dest, const u32* src, usize size_bytes) {
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
    return m_blocks.find(guest_rip) != m_blocks.end();
}

void JitCompiler::clear_cache() {
    m_blocks.clear();
    m_metrics.clear();
    m_code_buffer_offset = 0;
}

const JitBlockMetrics* JitCompiler::get_block_metrics(GuestAddr guest_rip) const {
    auto it = m_metrics.find(guest_rip);
    if (it != m_metrics.end()) {
        return &it->second;
    }
    return nullptr;
}

JittedBlockFunc JitCompiler::compile_block(GuestAddr guest_rip, const MemoryManager& memory) {
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

        const u8* code = memory.to_host_ptr<u8>(current_rip);
        if (!code) break;

        const u8 op0 = code[0];

        // 0x90: NOP
        if (op0 == 0x90) {
            m_emitter.nop();
            current_rip += 1;
            instructions++;
            continue;
        }

        // 0xF4: HLT (end block without advancing RIP so interpreter steps into it)
        if (op0 == 0xF4) {
            block_ended = true;
            break;
        }

        // 0x0F 0x05: SYSCALL (end block without advancing RIP so interpreter steps into it)
        if (op0 == 0x0F && memory.is_valid_address(current_rip, 2) && code[1] == 0x05) {
            block_ended = true;
            break;
        }

        // 0x31: XOR reg, reg
        if (op0 == 0x31 && memory.is_valid_address(current_rip, 2)) {
            const u8 modrm = code[1];
            if (modrm == 0xC0) { // xor eax, eax
                m_emitter.str_zero(19, offsetof(CpuContext, rax));
                current_rip += 2; instructions++; continue;
            }
            if (modrm == 0xFF) { // xor edi, edi
                m_emitter.str_zero(19, offsetof(CpuContext, rdi));
                current_rip += 2; instructions++; continue;
            }
            if (modrm == 0xF6) { // xor esi, esi
                m_emitter.str_zero(19, offsetof(CpuContext, rsi));
                current_rip += 2; instructions++; continue;
            }
            if (modrm == 0xD2) { // xor edx, edx
                m_emitter.str_zero(19, offsetof(CpuContext, rdx));
                current_rip += 2; instructions++; continue;
            }
        }

        // 0xB8 - 0xBF: MOV reg32, imm32 (5 bytes)
        if (op0 >= 0xB8 && op0 <= 0xBF && memory.is_valid_address(current_rip, 5)) {
            u32 imm32 = 0;
            std::memcpy(&imm32, code + 1, sizeof(imm32));
            u32 off = offsetof(CpuContext, rax);
            switch (op0) {
                case 0xB8: off = offsetof(CpuContext, rax); break;
                case 0xB9: off = offsetof(CpuContext, rcx); break;
                case 0xBA: off = offsetof(CpuContext, rdx); break;
                case 0xBB: off = offsetof(CpuContext, rbx); break;
                case 0xBE: off = offsetof(CpuContext, rsi); break;
                case 0xBF: off = offsetof(CpuContext, rdi); break;
                default: break;
            }
            m_emitter.mov_imm64(0, imm32);
            m_emitter.str_reg(0, 19, off);
            current_rip += 5;
            instructions++;
            continue;
        }

        // REX.W prefix (0x48) instructions
        if (op0 == 0x48 && memory.is_valid_address(current_rip, 2)) {
            const u8 op1 = code[1];

            // 0x48 0xC7: MOV r/m64, imm32 (sign-extended, 7 bytes)
            if (op1 == 0xC7 && memory.is_valid_address(current_rip, 7)) {
                const u8 modrm = code[2];
                i32 imm32 = 0;
                std::memcpy(&imm32, code + 3, sizeof(imm32));
                const u64 imm64 = static_cast<u64>(static_cast<i64>(imm32));

                u32 off = offsetof(CpuContext, rax);
                if (modrm == 0xC0) off = offsetof(CpuContext, rax);
                else if (modrm == 0xC1) off = offsetof(CpuContext, rcx);
                else if (modrm == 0xC2) off = offsetof(CpuContext, rdx);
                else if (modrm == 0xC3) off = offsetof(CpuContext, rbx);
                else if (modrm == 0xC6) off = offsetof(CpuContext, rsi);
                else if (modrm == 0xC7) off = offsetof(CpuContext, rdi);
                else break;

                m_emitter.mov_imm64(0, imm64);
                m_emitter.str_reg(0, 19, off);
                current_rip += 7;
                instructions++;
                continue;
            }

            // 0x48 0xB8 - 0xBF: MOVABS reg64, imm64 (10 bytes)
            if (op1 >= 0xB8 && op1 <= 0xBF && memory.is_valid_address(current_rip, 10)) {
                u64 imm64 = 0;
                std::memcpy(&imm64, code + 2, sizeof(imm64));
                u32 off = offsetof(CpuContext, rax);
                switch (op1) {
                    case 0xB8: off = offsetof(CpuContext, rax); break;
                    case 0xB9: off = offsetof(CpuContext, rcx); break;
                    case 0xBA: off = offsetof(CpuContext, rdx); break;
                    case 0xBB: off = offsetof(CpuContext, rbx); break;
                    case 0xBE: off = offsetof(CpuContext, rsi); break;
                    case 0xBF: off = offsetof(CpuContext, rdi); break;
                    default: break;
                }
                m_emitter.mov_imm64(0, imm64);
                m_emitter.str_reg(0, 19, off);
                current_rip += 10;
                instructions++;
                continue;
            }

            // 0x48 0x01: ADD r/m64, r64 (3 bytes)
            if (op1 == 0x01 && memory.is_valid_address(current_rip, 3)) {
                const u8 modrm = code[2];
                const u8 rm_idx = (modrm >> 3) & 0x7;
                const u8 rn_idx = modrm & 0x7;
                const u32 off_rm = get_gpr_offset(rm_idx);
                const u32 off_rn = get_gpr_offset(rn_idx);

                m_emitter.ldr_reg(0, 19, off_rn); // x0 = dest
                m_emitter.ldr_reg(1, 19, off_rm); // x1 = src
                m_emitter.add_reg(0, 0, 1);
                m_emitter.str_reg(0, 19, off_rn);

                current_rip += 3;
                instructions++;
                continue;
            }

            // 0x48 0x29: SUB r/m64, r64 (3 bytes)
            if (op1 == 0x29 && memory.is_valid_address(current_rip, 3)) {
                const u8 modrm = code[2];
                const u8 rm_idx = (modrm >> 3) & 0x7;
                const u8 rn_idx = modrm & 0x7;
                const u32 off_rm = get_gpr_offset(rm_idx);
                const u32 off_rn = get_gpr_offset(rn_idx);

                m_emitter.ldr_reg(0, 19, off_rn); // x0 = dest
                m_emitter.ldr_reg(1, 19, off_rm); // x1 = src
                m_emitter.sub_reg(0, 0, 1);
                m_emitter.str_reg(0, 19, off_rn);

                current_rip += 3;
                instructions++;
                continue;
            }
        }

        // 0xC3: RET (end block without advancing RIP so interpreter step handles the stack pop)
        if (op0 == 0xC3) {
            block_ended = true;
            break;
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
    void* code_ptr = allocate_code(code_size);
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

bool JitCompiler::execute_block(CpuContext& ctx, const MemoryManager& memory) {
    auto fn = compile_block(ctx.rip, memory);
    if (!fn) {
        return false;
    }

    m_metrics[ctx.rip].execution_count++;
    m_total_invocations++;

    fn(&ctx);
    return true;
}

} // namespace ps4
