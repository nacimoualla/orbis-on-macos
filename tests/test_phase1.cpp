#include <cassert>
#include <iostream>
#include <limits>
#include <type_traits>
#include "ps4/common.hpp"

int main() {
    std::cout << "[PHASE 1] Starting type validation and sanity checks...\n";

    // 1. Primitive Sizes
    static_assert(sizeof(ps4::u8) == 1, "u8 must be 1 byte");
    static_assert(sizeof(ps4::u16) == 2, "u16 must be 2 bytes");
    static_assert(sizeof(ps4::u32) == 4, "u32 must be 4 bytes");
    static_assert(sizeof(ps4::u64) == 8, "u64 must be 8 bytes");
    static_assert(sizeof(ps4::u128) == 16, "u128 must be 16 bytes");

    static_assert(sizeof(ps4::i8) == 1, "i8 must be 1 byte");
    static_assert(sizeof(ps4::i16) == 2, "i16 must be 2 bytes");
    static_assert(sizeof(ps4::i32) == 4, "i32 must be 4 bytes");
    static_assert(sizeof(ps4::i64) == 8, "i64 must be 8 bytes");
    static_assert(sizeof(ps4::i128) == 16, "i128 must be 16 bytes");

    static_assert(sizeof(ps4::f32) == 4, "f32 must be 4 bytes");
    static_assert(sizeof(ps4::f64) == 8, "f64 must be 8 bytes");

    // 2. Signedness checks
    static_assert(std::is_unsigned_v<ps4::u8>, "u8 must be unsigned");
    static_assert(std::is_unsigned_v<ps4::u16>, "u16 must be unsigned");
    static_assert(std::is_unsigned_v<ps4::u32>, "u32 must be unsigned");
    static_assert(std::is_unsigned_v<ps4::u64>, "u64 must be unsigned");

    static_assert(std::is_signed_v<ps4::i8>, "i8 must be signed");
    static_assert(std::is_signed_v<ps4::i16>, "i16 must be signed");
    static_assert(std::is_signed_v<ps4::i32>, "i32 must be signed");
    static_assert(std::is_signed_v<ps4::i64>, "i64 must be signed");

    // 3. Guest and Host Address Type Integrity
    static_assert(sizeof(ps4::GuestAddr) == 8, "GuestAddr must be 64-bit integer");
    static_assert(sizeof(ps4::GuestSize) == 8, "GuestSize must be 64-bit integer");
    static_assert(sizeof(ps4::HostAddr) == sizeof(void*), "HostAddr must match native host pointer width");

    // 4. Orbis OS Primitives
    static_assert(sizeof(ps4::Nid) == 8, "Nid must be 64-bit integer");
    static_assert(sizeof(ps4::OrbisResult) == 4, "OrbisResult must be 32-bit signed integer");
    static_assert(std::is_signed_v<ps4::OrbisResult>, "OrbisResult must be signed");

    // 5. Standard ELF64 Types
    static_assert(sizeof(ps4::Elf64_Addr) == 8, "Elf64_Addr must be 64-bit");
    static_assert(sizeof(ps4::Elf64_Off) == 8, "Elf64_Off must be 64-bit");
    static_assert(sizeof(ps4::Elf64_Half) == 2, "Elf64_Half must be 16-bit");
    static_assert(sizeof(ps4::Elf64_Word) == 4, "Elf64_Word must be 32-bit");
    static_assert(sizeof(ps4::Elf64_Sword) == 4, "Elf64_Sword must be 32-bit signed");
    static_assert(sizeof(ps4::Elf64_Xword) == 8, "Elf64_Xword must be 64-bit");
    static_assert(sizeof(ps4::Elf64_Sxword) == 8, "Elf64_Sxword must be 64-bit signed");

    // 6. CPU Register Types
    static_assert(sizeof(ps4::Reg64) == 8, "Reg64 must be 64-bit");
    static_assert(sizeof(ps4::Reg32) == 4, "Reg32 must be 32-bit");
    static_assert(sizeof(ps4::V128) == 16, "V128 must be 128-bit");

    // 7. Runtime boundary tests
    ps4::GuestAddr test_addr = 0x00400000ULL;
    ps4::GuestSize page_size = 0x4000ULL; // 16 KB
    assert(test_addr % page_size == 0);

    ps4::OrbisResult ok_result = 0;
    ps4::OrbisResult err_result = -1;
    assert(ok_result == 0);
    assert(err_result < 0);

    std::cout << "[PHASE 1] PASS: All compile-time and runtime type checks passed successfully.\n";
    return 0;
}
