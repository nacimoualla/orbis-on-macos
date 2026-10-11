#include "ps4/elf_loader.hpp"
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
