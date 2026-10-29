#pragma once

#include "ps4/common.hpp"
#include <string_view>

namespace ps4 {

// ELF Identification indexes
inline constexpr usize EI_MAG0       = 0;
inline constexpr usize EI_MAG1       = 1;
inline constexpr usize EI_MAG2       = 2;
inline constexpr usize EI_MAG3       = 3;
inline constexpr usize EI_CLASS      = 4;
inline constexpr usize EI_DATA       = 5;
inline constexpr usize EI_VERSION    = 6;
inline constexpr usize EI_OSABI      = 7;
inline constexpr usize EI_ABIVERSION = 8;
inline constexpr usize EI_PAD        = 9;
inline constexpr usize EI_NIDENT     = 16;

// ELF Magic bytes
inline constexpr u8 ELFMAG0 = 0x7F;
inline constexpr u8 ELFMAG1 = 'E';
inline constexpr u8 ELFMAG2 = 'L';
inline constexpr u8 ELFMAG3 = 'F';

// ELF Class
inline constexpr u8 ELFCLASSNONE = 0;
inline constexpr u8 ELFCLASS32   = 1;
inline constexpr u8 ELFCLASS64   = 2;

// ELF Data encoding
inline constexpr u8 ELFDATANONE = 0;
inline constexpr u8 ELFDATA2LSB = 1; // 2's complement, little endian
inline constexpr u8 ELFDATA2MSB = 2; // 2's complement, big endian

// ELF OS ABI
inline constexpr u8 ELFOSABI_SYSV    = 0;
inline constexpr u8 ELFOSABI_FREEBSD = 9; // FreeBSD 9 (Orbis OS base)

// ELF Machine architectures
inline constexpr u16 EM_NONE   = 0;
inline constexpr u16 EM_X86_64 = 0x3E; // AMD x86-64 architecture (62)

// ELF Object file types
inline constexpr u16 ET_NONE          = 0;
inline constexpr u16 ET_REL           = 1;
inline constexpr u16 ET_EXEC          = 2;
inline constexpr u16 ET_DYN           = 3;
inline constexpr u16 ET_CORE          = 4;
inline constexpr u16 ET_SCE_EXEC      = 0xFE00; // PlayStation 4 executable
inline constexpr u16 ET_SCE_RELEXEC   = 0xFE04; // PlayStation 4 relocatable executable
inline constexpr u16 ET_SCE_DYNAMIC   = 0xFE18; // PlayStation 4 PRX dynamic module
// eboot.elf reports 0xFE10 for its fixed-address executable, which is not the
// 0xFE00 above. Identify relocatability positively rather than negating an
// exec value that has more than one spelling in shipped Orbis binaries.
inline constexpr u16 ET_SCE_EXEC_ALT  = 0xFE10;

// ELF Program Header Segment types
inline constexpr u32 PT_NULL          = 0;
inline constexpr u32 PT_LOAD          = 1;
inline constexpr u32 PT_DYNAMIC       = 2;
inline constexpr u32 PT_INTERP        = 3;
inline constexpr u32 PT_NOTE          = 4;
inline constexpr u32 PT_SHLIB         = 5;
inline constexpr u32 PT_PHDR          = 6;
inline constexpr u32 PT_TLS           = 7;

// Sony Orbis OS custom segment types
inline constexpr u32 PT_SCE_RELA       = 0x60000000; // Relocations
inline constexpr u32 PT_SCE_PROGLIB    = 0x61000000; // Program library info
inline constexpr u32 PT_SCE_DYNLIBDATA = 0x61000001; // Dynamic library data
inline constexpr u32 PT_SCE_RELRO      = 0x61000010; // Read-only after relocations
inline constexpr u32 PT_SCE_COMMENT    = 0x6FFFFF00; // SDK comments/metadata
inline constexpr u32 PT_SCE_LIBVERSION = 0x6FFFFF01; // Library versioning

// Program Header Segment flags
inline constexpr u32 PF_X = 1 << 0; // Execute
inline constexpr u32 PF_W = 1 << 1; // Write
inline constexpr u32 PF_R = 1 << 2; // Read

#pragma pack(push, 1)
/// Standard 64-bit ELF File Header layout (64 bytes)
struct Elf64Header {
    u8          e_ident[EI_NIDENT]; // 16 bytes: magic & identification
    Elf64_Half  e_type;             // Object file type
    Elf64_Half  e_machine;          // Architecture
    Elf64_Word  e_version;          // Object file version
    Elf64_Addr  e_entry;            // Entry point virtual address
    Elf64_Off   e_phoff;            // Program header table file offset
    Elf64_Off   e_shoff;            // Section header table file offset
    Elf64_Word  e_flags;            // Processor-specific flags
    Elf64_Half  e_ehsize;           // ELF header size in bytes
    Elf64_Half  e_phentsize;        // Program header table entry size
    Elf64_Half  e_phnum;            // Program header table entry count
    Elf64_Half  e_shentsize;        // Section header table entry size
    Elf64_Half  e_shnum;            // Section header table entry count
    Elf64_Half  e_shstrndx;         // Section header string table index
};

/// Standard 64-bit ELF Program Header entry layout (56 bytes)
struct Elf64ProgramHeader {
    Elf64_Word  p_type;   // Segment type
    Elf64_Word  p_flags;  // Segment flags (PF_R, PF_W, PF_X)
    Elf64_Off   p_offset; // Segment file offset
    Elf64_Addr  p_vaddr;  // Segment virtual address in memory
    Elf64_Addr  p_paddr;  // Segment physical address (unused)
    Elf64_Xword p_filesz; // Segment size in file
    Elf64_Xword p_memsz;  // Segment size in memory (p_memsz >= p_filesz)
    Elf64_Xword p_align;  // Segment alignment
};
#pragma pack(pop)

static_assert(sizeof(Elf64Header) == 64, "Elf64Header must be exactly 64 bytes");
static_assert(sizeof(Elf64ProgramHeader) == 56, "Elf64ProgramHeader must be exactly 56 bytes");

/// Helper to get a human-readable string for segment type
inline std::string_view segment_type_to_string(u32 type) {
    switch (type) {
        case PT_NULL:          return "PT_NULL";
        case PT_LOAD:          return "PT_LOAD";
        case PT_DYNAMIC:       return "PT_DYNAMIC";
        case PT_INTERP:        return "PT_INTERP";
        case PT_NOTE:          return "PT_NOTE";
        case PT_SHLIB:         return "PT_SHLIB";
        case PT_PHDR:          return "PT_PHDR";
        case PT_TLS:           return "PT_TLS";
        case PT_SCE_RELA:       return "PT_SCE_RELA";
        case PT_SCE_PROGLIB:    return "PT_SCE_PROGLIB";
        case PT_SCE_DYNLIBDATA: return "PT_SCE_DYNLIBDATA";
        case PT_SCE_RELRO:      return "PT_SCE_RELRO";
        case PT_SCE_COMMENT:    return "PT_SCE_COMMENT";
        case PT_SCE_LIBVERSION: return "PT_SCE_LIBVERSION";
        default:               return "UNKNOWN";
    }
}

} // namespace ps4
