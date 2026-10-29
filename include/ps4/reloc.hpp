#pragma once

#include "ps4/common.hpp"

namespace ps4 {

// Standard ELF Dynamic Array Tags (d_tag)
inline constexpr i64 DT_NULL        = 0;
inline constexpr i64 DT_NEEDED      = 1;
inline constexpr i64 DT_PLTRELSZ    = 2;
inline constexpr i64 DT_PLTGOT      = 3;
inline constexpr i64 DT_HASH        = 4;
inline constexpr i64 DT_STRTAB      = 5;
inline constexpr i64 DT_SYMTAB      = 6;
inline constexpr i64 DT_RELA        = 7;
inline constexpr i64 DT_RELASZ      = 8;
inline constexpr i64 DT_RELAENT     = 9;
inline constexpr i64 DT_STRSZ       = 10;
inline constexpr i64 DT_SYMENT      = 11;
inline constexpr i64 DT_INIT        = 12;
inline constexpr i64 DT_FINI        = 13;
inline constexpr i64 DT_SONAME      = 14;
inline constexpr i64 DT_RPATH       = 15;
inline constexpr i64 DT_SYMBOLIC    = 16;
inline constexpr i64 DT_REL         = 17;
inline constexpr i64 DT_PLTREL      = 20;
inline constexpr i64 DT_DEBUG       = 21;
inline constexpr i64 DT_TEXTREL     = 22;
inline constexpr i64 DT_JMPREL      = 23;
inline constexpr i64 DT_BIND_NOW    = 24;
inline constexpr i64 DT_INIT_ARRAY  = 25;
inline constexpr i64 DT_FINI_ARRAY  = 26;
inline constexpr i64 DT_INIT_ARRAYSZ = 27;
inline constexpr i64 DT_FINI_ARRAYSZ = 28;

// Sony Orbis OS custom dynamic tags (PT_SCE_DYNLIBDATA)
inline constexpr i64 DT_SCE_FINGERPRINT          = 0x61000007;
inline constexpr i64 DT_SCE_ORIGINAL_FILENAME    = 0x61000009;
inline constexpr i64 DT_SCE_MODULE_INFO          = 0x6100000D;
inline constexpr i64 DT_SCE_NEEDED_MODULE        = 0x6100000F;
inline constexpr i64 DT_SCE_MODULE_ATTR          = 0x61000011;
inline constexpr i64 DT_SCE_EXPORT_LIB           = 0x61000013;
inline constexpr i64 DT_SCE_IMPORT_LIB           = 0x61000015;
inline constexpr i64 DT_SCE_EXPORT_LIB_ATTR      = 0x61000017;
inline constexpr i64 DT_SCE_IMPORT_LIB_ATTR      = 0x61000019;
inline constexpr i64 DT_SCE_STUB_MODULE_NAME     = 0x6100001D;
inline constexpr i64 DT_SCE_STUB_MODULE_VERSION  = 0x6100001F;
inline constexpr i64 DT_SCE_STUB_LIBRARY_NAME    = 0x61000021;
inline constexpr i64 DT_SCE_STUB_LIBRARY_VERSION = 0x61000023;
inline constexpr i64 DT_SCE_HASH                 = 0x61000025;
inline constexpr i64 DT_SCE_PLTGOT               = 0x61000027;
inline constexpr i64 DT_SCE_JMPREL               = 0x61000029;
inline constexpr i64 DT_SCE_PLTREL               = 0x6100002B;
inline constexpr i64 DT_SCE_PLTRELSZ             = 0x6100002D;
inline constexpr i64 DT_SCE_RELA                 = 0x6100002F;
inline constexpr i64 DT_SCE_RELASZ               = 0x61000031;
inline constexpr i64 DT_SCE_RELAENT              = 0x61000033;
inline constexpr i64 DT_SCE_STRTAB               = 0x61000035;
inline constexpr i64 DT_SCE_STRSZ                = 0x61000037;
inline constexpr i64 DT_SCE_SYMTAB               = 0x61000039;
inline constexpr i64 DT_SCE_SYMENT               = 0x6100003B;

// DT_SCE_SYMTAB (0x61000039) is the symbol table OFFSET and the tag past
// SYMENT is its byte SIZE, so the entry count is d_val / sizeof(Elf64_Sym).
// Verified on both shipped modules: mono-ps4.prx gives 0x38e8 / 0x5400, and
// 0x5400 is exactly 896 * 24; MonoAssembliesPS4.prx gives 0x268 / 0x2a0 = 28
// symbols. Reading the size as an offset lands mid-table and silently shifts
// every st_name, which is what makes a back-scan appear necessary.
inline constexpr i64 DT_SCE_SYMTABSZ             = 0x6100003F;



// x86_64 Relocation Types (ELF64_R_TYPE)
inline constexpr u32 R_X86_64_NONE      = 0;
inline constexpr u32 R_X86_64_64        = 1;
inline constexpr u32 R_X86_64_PC32      = 2;
inline constexpr u32 R_X86_64_GOT32     = 3;
inline constexpr u32 R_X86_64_PLT32     = 4;
inline constexpr u32 R_X86_64_COPY      = 5;
inline constexpr u32 R_X86_64_GLOB_DAT  = 6; // Set GOT entry to data symbol address
inline constexpr u32 R_X86_64_JUMP_SLOT = 7; // Set GOT entry to function symbol address
inline constexpr u32 R_X86_64_RELATIVE  = 8; // Adjust address by base load address
inline constexpr u32 R_X86_64_GOTPCREL  = 9;

inline constexpr u32 ELF64_R_SYM(u64 info) noexcept {
    return static_cast<u32>(info >> 32);
}

inline constexpr u32 ELF64_R_TYPE(u64 info) noexcept {
    return static_cast<u32>(info & 0xFFFFFFFFULL);
}

inline constexpr u64 ELF64_R_INFO(u32 sym, u32 type) noexcept {
    return (static_cast<u64>(sym) << 32) | static_cast<u64>(type);
}

#pragma pack(push, 1)

/// ELF64 Dynamic Array Entry (16 bytes)
struct Elf64_Dyn {
    Elf64_Sxword d_tag;
    union {
        Elf64_Xword d_val;
        Elf64_Addr  d_ptr;
    } d_un;
};

/// ELF64 Relocation Entry with Addend (24 bytes)
struct Elf64_Rela {
    Elf64_Addr   r_offset; // Virtual address where relocation is applied
    Elf64_Xword  r_info;   // Symbol index and relocation type
    Elf64_Sxword r_addend; // Addend value
};

/// ELF64 Symbol Table Entry (24 bytes)
struct Elf64_Sym {
    Elf64_Word  st_name;  // Symbol name (offset into strtab) or NID index
    u8          st_info;  // Symbol type and binding
    u8          st_other; // Symbol visibility
    Elf64_Half  st_shndx; // Section table index
    Elf64_Addr  st_value; // Symbol value / address
    Elf64_Xword st_size;  // Symbol size
};

#pragma pack(pop)

static_assert(sizeof(Elf64_Dyn) == 16, "Elf64_Dyn must be exactly 16 bytes");
static_assert(sizeof(Elf64_Rela) == 24, "Elf64_Rela must be exactly 24 bytes");
static_assert(sizeof(Elf64_Sym) == 24, "Elf64_Sym must be exactly 24 bytes");

} // namespace ps4
