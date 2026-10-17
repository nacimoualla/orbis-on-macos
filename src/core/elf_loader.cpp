#include "ps4/elf_loader.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace ps4 {

std::string_view elf_error_to_string(ElfError error) {
    switch (error) {
        case ElfError::None:
            return "Success";
        case ElfError::FileNotFound:
            return "File not found or cannot be opened";
        case ElfError::FileTooSmall:
            return "File is smaller than the 64-byte ELF header";
        case ElfError::InvalidMagic:
            return "Invalid ELF magic number (expected 0x7F 'E' 'L' 'F')";
        case ElfError::Not64Bit:
            return "ELF file is not 64-bit (EI_CLASS != ELFCLASS64)";
        case ElfError::NotLittleEndian:
            return "ELF file is not little-endian (EI_DATA != ELFDATA2LSB)";
        case ElfError::InvalidArchitecture:
            return "Unsupported architecture (expected x86-64 / 0x3E)";
        case ElfError::InvalidProgramHeadersOffset:
            return "Program header table offset is out of file bounds";
        case ElfError::InvalidProgramHeaderSize:
            return "Program header entry size is smaller than standard Elf64ProgramHeader";
        case ElfError::ReadError:
            return "I/O error while reading file data";
    }
    return "Unknown error";
}

bool ElfLoader::load_from_file(const std::filesystem::path& path) {
    m_loaded = false;
    m_program_headers.clear();

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        m_error = ElfError::FileNotFound;
        return false;
    }

    const auto file_size = file.tellg();
    if (file_size < static_cast<std::streampos>(sizeof(Elf64Header))) {
        m_error = ElfError::FileTooSmall;
        return false;
    }

    file.seekg(0, std::ios::beg);
    std::vector<u8> buffer(static_cast<usize>(file_size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        m_error = ElfError::ReadError;
        return false;
    }

    return parse_internal(buffer);
}

bool ElfLoader::load_from_memory(std::span<const u8> buffer) {
    m_loaded = false;
    m_program_headers.clear();
    return parse_internal(buffer);
}

bool ElfLoader::parse_internal(std::span<const u8> buffer) {
    // A shipped PS4 module is a 352-byte SELF container wrapped around a
    // plaintext ELF. Detect and peel it before anything reads program headers,
    // otherwise the ELF magic check below rejects every .prx outright.
    if (buffer.size() >= SELF_HEADER_SIZE) {
        u32 magic = 0;
        std::memcpy(&magic, buffer.data(), sizeof(magic));
        if (magic == SELF_MAGIC) {
            return parse_self(buffer);
        }
    }

    const usize ph_table_size = static_cast<usize>(m_header.e_phnum) * static_cast<usize>(m_header.e_phentsize);
    if (m_header.e_phoff > buffer.size() || ph_table_size > buffer.size() - m_header.e_phoff) {
        m_error = ElfError::InvalidProgramHeadersOffset;
        return false;
    }
    if (buffer.size() < sizeof(Elf64Header)) {
        m_error = ElfError::FileTooSmall;
        return false;
    }

    std::memcpy(&m_header, buffer.data(), sizeof(Elf64Header));

    // Validate magic bytes
    if (m_header.e_ident[EI_MAG0] != ELFMAG0 ||
        m_header.e_ident[EI_MAG1] != ELFMAG1 ||
        m_header.e_ident[EI_MAG2] != ELFMAG2 ||
        m_header.e_ident[EI_MAG3] != ELFMAG3) {
        m_error = ElfError::InvalidMagic;
        return false;
    }

    // Validate 64-bit class
    if (m_header.e_ident[EI_CLASS] != ELFCLASS64) {
        m_error = ElfError::Not64Bit;
        return false;
    }

    // Validate little-endian data encoding
    if (m_header.e_ident[EI_DATA] != ELFDATA2LSB) {
        m_error = ElfError::NotLittleEndian;
        return false;
    }

    // Validate x86-64 target architecture
    if (m_header.e_machine != EM_X86_64) {
        m_error = ElfError::InvalidArchitecture;
        return false;
    }

    // Validate program header table bounds
    if (m_header.e_phnum > 0) {
        if (m_header.e_phentsize < sizeof(Elf64ProgramHeader)) {
            m_error = ElfError::InvalidProgramHeaderSize;
            return false;
        }

        const usize ph_table_end = static_cast<usize>(m_header.e_phoff) +
            (static_cast<usize>(m_header.e_phnum) * m_header.e_phentsize);

        if (ph_table_end > buffer.size() || ph_table_end < m_header.e_phoff) {
            m_error = ElfError::InvalidProgramHeadersOffset;
            return false;
        }

        m_program_headers.reserve(m_header.e_phnum);
        for (usize i = 0; i < m_header.e_phnum; ++i) {
            Elf64ProgramHeader phdr{};
            const usize offset = static_cast<usize>(m_header.e_phoff) + (i * m_header.e_phentsize);
            std::memcpy(&phdr, buffer.data() + offset, sizeof(Elf64ProgramHeader));
            m_program_headers.push_back(phdr);
        }
    }

    m_error = ElfError::None;
    m_loaded = true;
    return true;
}

// ---------------------------------------------------------------------------
// SELF container
// ---------------------------------------------------------------------------

bool ElfLoader::parse_self(std::span<const u8> raw) {
    u16 count = 0;
    std::memcpy(&count, raw.data() + SELF_ENTRY_COUNT_OFF, sizeof(count));
    if (count == 0 || count > 64) {
        std::fprintf(stderr, "[elf] SELF entry count %u out of range\n", count);
        m_error = ElfError::InvalidProgramHeaderSize;
        return false;
    }

    m_is_self = true;
    m_elf_file_offset = SELF_ELF_OFFSET;
    m_self_entries.clear();
    m_self_entries.reserve(count);
    for (u16 i = 0; i < count; ++i) {
        const usize off = SELF_ENTRY_TABLE_OFF + (static_cast<usize>(i) * 32);
        if (off + 32 > raw.size()) {
            std::fprintf(stderr, "[elf] SELF entry %u truncated\n", i);
            m_error = ElfError::FileTooSmall;
            return false;
        }
        SelfEntry e{};
        std::memcpy(&e.type,     raw.data() + off + 0,  sizeof(u32));
        std::memcpy(&e.reserved, raw.data() + off + 4,  sizeof(u32));
        std::memcpy(&e.offset,   raw.data() + off + 8,  sizeof(u64));
        std::memcpy(&e.filesz,   raw.data() + off + 16, sizeof(u64));
        std::memcpy(&e.memsz,    raw.data() + off + 24, sizeof(u64));
        // A truncated dump would otherwise hand out offsets past EOF.
        if (e.offset + e.filesz > raw.size()) {
            std::fprintf(stderr, "[elf] SELF entry %u out of file bounds "
                                "(off=0x%llx filesz=0x%llx, file=0x%llzx) -- ignoring\n",
                         i, (unsigned long long)e.offset, (unsigned long long)e.filesz, raw.size());
            continue;
        }
        m_self_entries.push_back(e);
    }

    // Re-enter the normal path on the ELF that starts at SELF_ELF_OFFSET.
    // Its magic is \x7fELF, so this terminates instead of recursing.
    m_error = ElfError::None;
    m_loaded = false;
    if (!parse_internal(raw.subspan(SELF_ELF_OFFSET))) {
        return false;
    }

    // Cross-check: the entry whose size matches a segment's p_filesz is that
    // segment's real location. Report the divergence once, since it is the
    // thing that makes naive PT_DYNAMIC walking fail.
    if (const auto* proglib = find_phdr(PT_SCE_PROGLIB)) {
        if (const usize real_off = segment_file_offset(*proglib);
            real_off != 0 && real_off != static_cast<usize>(proglib->p_offset)) {
            std::fprintf(stderr,
                         "[elf] SELF: PT_SCE_PROGLIB p_offset=0x%llx is stale; "
                         "real file offset 0x%zx (delta %lld)\n",
                         (unsigned long long)proglib->p_offset, real_off,
                         (long long)static_cast<isize>(proglib->p_offset) -
                         (long long)static_cast<isize>(real_off));
        }
        if (const usize dyn = dynamic_offset_file_offset(); dyn != 0) {
            std::fprintf(stderr,
                         "[elf] SELF: PT_DYNAMIC p_offset=0x%llx is past payload; "
                         "real file offset 0x%zx\n",
                         (unsigned long long)find_phdr(PT_DYNAMIC)->p_offset, dyn);
        }
    }
    return true;
}

const Elf64ProgramHeader* ElfLoader::find_phdr(u32 p_type) const {
    for (const auto& ph : m_program_headers) {
        if (ph.p_type == p_type) return &ph;
    }
    return nullptr;
}

usize ElfLoader::segment_file_offset(const Elf64ProgramHeader& ph, bool* found) const {
    if (found) *found = false;

    if (m_is_self) {
        // Match on size: the entry table carries no p_type, but each segment has
        // a distinct p_filesz, which identifies its entry uniquely.
        for (const auto& e : m_self_entries) {
            if (e.filesz == ph.p_filesz) {
                if (found) *found = true;
                return static_cast<usize>(e.offset);
            }
        }
        return 0;
    }

    if (found) *found = true;
    return static_cast<usize>(ph.p_offset);
}

usize ElfLoader::proglib_file_offset() const {
    const Elf64ProgramHeader* ph = find_phdr(PT_SCE_PROGLIB);
    if (ph == nullptr) return 0;
    bool found = false;
    return segment_file_offset(*ph, &found);
}

usize ElfLoader::proglib_file_size() const {
    const Elf64ProgramHeader* ph = find_phdr(PT_SCE_PROGLIB);
    if (ph == nullptr) return 0;
    return static_cast<usize>(ph->p_filesz);
}

usize ElfLoader::dynamic_offset_file_offset() const {
    if (!m_is_self) {
        const Elf64ProgramHeader* ph = find_phdr(PT_DYNAMIC);
        return ph ? static_cast<usize>(ph->p_offset) : 0;
    }

    const Elf64ProgramHeader* dyn = find_phdr(PT_DYNAMIC);
    const Elf64ProgramHeader* proglib = find_phdr(PT_SCE_PROGLIB);
    if (dyn == nullptr || proglib == nullptr) return 0;

    // Both offsets are measured from the same (stale) origin, so subtracting
    // cancels the error and leaves the true distance from PROGLIB's real start.
    if (dyn->p_offset <= proglib->p_offset) return 0;
    const u64 delta = dyn->p_offset - proglib->p_offset;
    const usize base = proglib_file_offset();
    if (base == 0) return 0;
    return base + static_cast<usize>(delta);
}

bool ElfLoader::needs_load_base() const noexcept {
    // Positively identify the relocatable types. Orbis modules do not use the
    // standard ET_DYN(3): mono-ps4.prx records 0xFE18 and eboot.elf records
    // 0xFE10, neither of which ET_DYN matches. Testing only ET_DYN makes every
    // PRX look like a fixed-address executable, and it is then mapped at 0 --
    // directly on top of the base module.
    switch (m_header.e_type) {
        case ET_DYN:
        case ET_SCE_RELEXEC:
        case ET_SCE_DYNAMIC:
            return true;
        default:
            return false;
    }
}

std::vector<Elf64ProgramHeader> ElfLoader::load_segments() const {
    std::vector<Elf64ProgramHeader> loads;
    for (const auto& phdr : m_program_headers) {
        if (phdr.p_type == PT_LOAD || phdr.p_type == PT_SCE_RELRO) {
            loads.push_back(phdr);
        }
    }
    return loads;
}

void ElfLoader::dump_info() const {
    if (!m_loaded) {
        std::cout << "[ElfLoader] No valid ELF loaded (Error: " << error_message() << ")\n";
        return;
    }

    std::cout << "========================================================\n";
    std::cout << " ELF64 Binary Information\n";
    std::cout << "========================================================\n";
    std::cout << "Entry Point:      0x" << std::hex << m_header.e_entry << std::dec << "\n";
    std::cout << "Type:             0x" << std::hex << m_header.e_type << std::dec;
    if (m_header.e_type == ET_SCE_EXEC) std::cout << " (Sony Orbis Executable)";
    else if (m_header.e_type == ET_SCE_DYNAMIC) std::cout << " (Sony Orbis Dynamic PRX)";
    else if (m_header.e_type == ET_EXEC) std::cout << " (Standard Executable)";
    else if (m_header.e_type == ET_DYN) std::cout << " (Shared Object / PIE)";
    std::cout << "\n";

    std::cout << "Program Headers:  " << m_program_headers.size() << "\n\n";

    std::cout << std::left
              << std::setw(6)  << "[Idx]"
              << std::setw(18) << "Type"
              << std::setw(14) << "Offset"
              << std::setw(18) << "VirtAddr"
              << std::setw(14) << "FileSz"
              << std::setw(14) << "MemSz"
              << std::setw(8)  << "Flags"
              << std::setw(10) << "Align"
              << "\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    for (usize i = 0; i < m_program_headers.size(); ++i) {
        const auto& ph = m_program_headers[i];
        std::string flags;
        flags += (ph.p_flags & PF_R) ? 'R' : '-';
        flags += (ph.p_flags & PF_W) ? 'W' : '-';
        flags += (ph.p_flags & PF_X) ? 'E' : '-';

        std::cout << std::left
                  << "[" << std::setw(3) << i << "] "
                  << std::setw(18) << segment_type_to_string(ph.p_type)
                  << "0x" << std::hex << std::setw(12) << ph.p_offset
                  << "0x" << std::setw(16) << ph.p_vaddr
                  << "0x" << std::setw(12) << ph.p_filesz
                  << "0x" << std::setw(12) << ph.p_memsz
                  << std::dec << std::setw(8) << flags
                  << "0x" << std::hex << ph.p_align << std::dec
                  << "\n";
    }
    std::cout << "========================================================\n";
}

} // namespace ps4
