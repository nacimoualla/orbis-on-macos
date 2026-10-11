#pragma once

#include <cstdint>
#include <cstddef>

namespace ps4 {

// Unsigned integers
using u8   = std::uint8_t;
using u16  = std::uint16_t;
using u32  = std::uint32_t;
using u64  = std::uint64_t;
using u128 = unsigned __int128;

// Signed integers
using i8   = std::int8_t;
using i16  = std::int16_t;
using i32  = std::int32_t;
using i64  = std::int64_t;
using i128 = __int128;

// Floating point
using f32  = float;
using f64  = double;

// Size & pointer arithmetic
using usize = std::size_t;
using isize = std::ptrdiff_t;

// Guest vs Host address spaces
using GuestAddr = u64;
using GuestSize = u64;
using HostAddr  = std::uintptr_t;

// PS4 Memory Constants
inline constexpr GuestSize GUEST_MEMORY_SIZE    = 8ULL * 1024 * 1024 * 1024; // 8 GB unified memory
inline constexpr GuestSize GUEST_PAGE_SIZE      = 0x4000;                     // 16 KB standard page
inline constexpr GuestSize GUEST_HUGE_PAGE_SIZE = 0x200000;                   // 2 MB huge page

// Orbis OS primitives
using Nid              = u64;
using OrbisResult      = i32;
using OrbisPid         = i32;
using OrbisTid         = i32;
using OrbisKernelModule = i32;
using OrbisHandle       = i32;

inline constexpr OrbisResult SCE_OK = 0;
inline constexpr OrbisResult SCE_ERROR_GENERIC = -1;

// Standard ELF64 types
using Elf64_Addr   = u64;
using Elf64_Off    = u64;
using Elf64_Half   = u16;
using Elf64_Word   = u32;
using Elf64_Sword  = i32;
using Elf64_Xword  = u64;
using Elf64_Sxword = i64;

// CPU register types
using Reg64 = u64;
using Reg32 = u32;
using V128  = u128;

} // namespace ps4