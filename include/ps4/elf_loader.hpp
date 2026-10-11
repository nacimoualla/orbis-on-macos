#pragma once

#include "ps4/common.hpp"
#include "ps4/elf.hpp"
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ps4 {

enum class ElfError {
    None,
    FileNotFound,
    FileTooSmall,
    InvalidMagic,
    Not64Bit,
    NotLittleEndian,
    InvalidArchitecture,
    InvalidProgramHeadersOffset,
    InvalidProgramHeaderSize,
    ReadError
};

std::string_view elf_error_to_string(ElfError error);

class ElfLoader {
public:
    ElfLoader() = default;

    /// Load and validate an ELF/PRX binary from disk
    bool load_from_file(const std::filesystem::path& path);

    /// Load and validate an ELF binary from an in-memory buffer
    bool load_from_memory(std::span<const u8> buffer);

    /// Retrieve the parsed ELF64 header
    [[nodiscard]] const Elf64Header& header() const noexcept { return m_header; }

    /// Retrieve all parsed program headers
    [[nodiscard]] const std::vector<Elf64ProgramHeader>& program_headers() const noexcept {
        return m_program_headers;
    }

    /// Retrieve only loadable segments (PT_LOAD)
    [[nodiscard]] std::vector<Elf64ProgramHeader> load_segments() const;

    /// Retrieve the virtual memory entry point
    [[nodiscard]] GuestAddr entry_point() const noexcept { return m_header.e_entry; }

    /// Return the error code if loading failed
    [[nodiscard]] ElfError error() const noexcept { return m_error; }

    /// Return a human-readable description of the last error
    [[nodiscard]] std::string_view error_message() const noexcept {
        return elf_error_to_string(m_error);
    }

    /// Check if an ELF has been successfully loaded
    [[nodiscard]] bool is_valid() const noexcept { return m_loaded; }

    /// Print detailed information about headers and segments to stdout
    void dump_info() const;

private:
    bool parse_internal(std::span<const u8> buffer);

    Elf64Header m_header{};
    std::vector<Elf64ProgramHeader> m_program_headers;
    ElfError m_error{ElfError::None};
    bool m_loaded{false};
};

} // namespace ps4
