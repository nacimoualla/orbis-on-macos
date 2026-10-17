#pragma once

#include "ps4/common.hpp"
#include <iomanip>
#include <iostream>

namespace ps4 {

/// CPU execution register context for x86_64 architecture
struct CpuContext {
    // 64-bit General-Purpose Registers (SysV ABI / Orbis OS)
    Reg64 rax{0};
    Reg64 rbx{0};
    Reg64 rcx{0};
    Reg64 rdx{0};
    Reg64 rsi{0};
    Reg64 rdi{0};
    Reg64 rsp{0}; // Stack Pointer
    Reg64 rbp{0}; // Base/Frame Pointer
    Reg64 r8{0};
    Reg64 r9{0};
    Reg64 r10{0};
    Reg64 r11{0};
    Reg64 r12{0};
    Reg64 r13{0};
    Reg64 r14{0};
    Reg64 r15{0};

    // Program Counter / Instruction Pointer
    Reg64 rip{0};

    // Flags Register
    Reg64 rflags{0x202}; // Standard initial flags (IF bit 9 set)

    static constexpr u64 FLAG_CF = 1ULL << 0;
    static constexpr u64 FLAG_PF = 1ULL << 2;
    static constexpr u64 FLAG_AF = 1ULL << 4;
    static constexpr u64 FLAG_ZF = 1ULL << 6;
    static constexpr u64 FLAG_SF = 1ULL << 7;
    static constexpr u64 FLAG_IF = 1ULL << 9;
    static constexpr u64 FLAG_DF = 1ULL << 10;
    static constexpr u64 FLAG_OF = 1ULL << 11;

    [[nodiscard]] bool flag_cf() const noexcept { return (rflags & FLAG_CF) != 0; }
    [[nodiscard]] bool flag_pf() const noexcept { return (rflags & FLAG_PF) != 0; }
    [[nodiscard]] bool flag_zf() const noexcept { return (rflags & FLAG_ZF) != 0; }
    [[nodiscard]] bool flag_sf() const noexcept { return (rflags & FLAG_SF) != 0; }
    [[nodiscard]] bool flag_df() const noexcept { return (rflags & FLAG_DF) != 0; }
    [[nodiscard]] bool flag_of() const noexcept { return (rflags & FLAG_OF) != 0; }

    void set_flag(u64 flag, bool val) noexcept {
        if (val) rflags |= flag;
        else rflags &= ~flag;
    }

    void update_flags_logic(u64 result, usize bits) noexcept {
        const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
        const u64 masked = result & mask;
        set_flag(FLAG_CF, false);
        set_flag(FLAG_OF, false);
        set_flag(FLAG_ZF, (masked == 0));
        set_flag(FLAG_PF, __builtin_parity(static_cast<u8>(masked & 0xFF)) == 0);
        const u64 sign_bit = (bits == 64) ? (1ULL << 63) : (bits == 32) ? (1ULL << 31) : (bits == 16) ? (1ULL << 15) : (1ULL << 7);
        set_flag(FLAG_SF, (masked & sign_bit) != 0);
    }

    void update_flags_add(u64 a, u64 b, u64 result, usize bits) noexcept {
        const u64 sign_mask = (bits == 64) ? (1ULL << 63) : (bits == 32) ? (1ULL << 31) : (bits == 16) ? (1ULL << 15) : (1ULL << 7);
        const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
        const u64 masked = result & mask;
        set_flag(FLAG_ZF, (masked == 0));
        set_flag(FLAG_SF, (masked & sign_mask) != 0);
        set_flag(FLAG_PF, __builtin_parity(static_cast<u8>(masked & 0xFF)) == 0);
        set_flag(FLAG_CF, masked < (a & mask));
        const bool a_sign = (a & sign_mask) != 0;
        const bool b_sign = (b & sign_mask) != 0;
        const bool res_sign = (masked & sign_mask) != 0;
        set_flag(FLAG_OF, (a_sign == b_sign) && (a_sign != res_sign));
    }

    void update_flags_sub(u64 a, u64 b, u64 result, usize bits) noexcept {
        const u64 sign_mask = (bits == 64) ? (1ULL << 63) : (bits == 32) ? (1ULL << 31) : (bits == 16) ? (1ULL << 15) : (1ULL << 7);
        const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
        const u64 masked = result & mask;
        set_flag(FLAG_ZF, (masked == 0));
        set_flag(FLAG_SF, (masked & sign_mask) != 0);
        set_flag(FLAG_PF, __builtin_parity(static_cast<u8>(masked & 0xFF)) == 0);
        set_flag(FLAG_CF, (a & mask) < (b & mask));
        const bool a_sign = (a & sign_mask) != 0;
        const bool b_sign = (b & sign_mask) != 0;
        const bool res_sign = (masked & sign_mask) != 0;
        set_flag(FLAG_OF, (a_sign != b_sign) && (a_sign != res_sign));
    }

    // Segment base registers (for TLS emulation)
    Reg64 fs_base{0};
    Reg64 gs_base{0};

    // 256-bit SIMD / AVX registers (ymm0 - ymm15)
    struct alignas(32) YmmReg {
        union {
            u8  b[32];
            u16 w[16];
            u32 d[8];
            u64 q[4];
            float f[8];
            double df[4];
        };
        YmmReg() : q{0, 0, 0, 0} {}
    };
    YmmReg ymm[16]{};

    void reset() noexcept {
        *this = CpuContext{};
    }

    void dump() const {
        std::cout << "-------------------- CPU Register Context --------------------\n";
        std::cout << "RIP: 0x" << std::hex << rip << "  RFLAGS: 0x" << rflags << "\n";
        std::cout << "RAX: 0x" << std::setw(16) << rax << "  RBX: 0x" << std::setw(16) << rbx << "\n";
        std::cout << "RCX: 0x" << std::setw(16) << rcx << "  RDX: 0x" << std::setw(16) << rdx << "\n";
        std::cout << "RSI: 0x" << std::setw(16) << rsi << "  RDI: 0x" << std::setw(16) << rdi << "\n";
        std::cout << "RBP: 0x" << std::setw(16) << rbp << "  RSP: 0x" << std::setw(16) << rsp << "\n";
        std::cout << "R8:  0x" << std::setw(16) << r8  << "  R9:  0x" << std::setw(16) << r9  << "\n";
        std::cout << "R10: 0x" << std::setw(16) << r10 << "  R11: 0x" << std::setw(16) << r11 << "\n";
        std::cout << "R12: 0x" << std::setw(16) << r12 << "  R13: 0x" << std::setw(16) << r13 << "\n";
        std::cout << "R14: 0x" << std::setw(16) << r14 << "  R15: 0x" << std::setw(16) << r15 << "\n";
        std::cout << "FS_BASE: 0x" << std::setw(16) << fs_base << "  GS_BASE: 0x" << std::setw(16) << gs_base << "\n" << std::dec;
        std::cout << "--------------------------------------------------------------\n";
    }
};

} // namespace ps4
