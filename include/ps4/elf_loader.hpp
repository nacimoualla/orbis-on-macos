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

// ---------------------------------------------------------------------------
// PS4 SELF/PRX container.
//
// A shipped .prx (and eboot.bin) is not a bare ELF: it is a 352-byte SELF
// header followed by the ELF image. The magic below is what the dump actually
// starts with, and the plaintext ELF begins at kSelfElfOffset.
//
// The program headers inside that ELF are NOT trustworthy for reading the file.
// They describe the layout the module had before Sony's packing; the SELF entry
// table describes where the bytes actually are. The two disagree by a different
// amount per segment (measured on mono-ps4.prx: -10,064 for PT_LOAD[0], -11,616
// for PT_LOAD[1], -11,280 for PT_SCE_PROGLIB), so there is no single shift to
// apply -- the entry table has to be consulted per segment.
//
// The worst case is PT_DYNAMIC: its p_offset points ~10 KB past the end of the
// module's own payload, which is why walking PT_DYNAMIC as-is finds garbage or
// nothing at all. See dynamic_offset_file_offset() for the correct derivation.
// ---------------------------------------------------------------------------

inline constexpr u32 SELF_MAGIC = 0x1D3D154FULL;
inline constexpr usize SELF_HEADER_SIZE = 0x160;   // 352
inline constexpr usize SELF_ELF_OFFSET  = 0x160;   // plaintext ELF starts here
inline constexpr usize SELF_ENTRY_COUNT_OFF = 0x18; // u16 number of entries
inline constexpr usize SELF_ENTRY_TABLE_OFF = 0x20; // 32-byte entries

struct SelfEntry {
    u32 type;
    u32 reserved;
    u64 offset;   // authoritative file offset of this segment's data
    u64 filesz;
    u64 memsz;

    /// Segment class: the high nibble of bit 8..11, used to pair each data
    /// segment with its metadata companion.
    [[nodiscard]] u32 segment_class() const noexcept { return (type >> 8) & 0xF; }
};

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

    // --- PS4 SELF/PRX support ------------------------------------------------

    /// True when the buffer/file began with the SELF magic rather than \x7fELF.
    [[nodiscard]] bool is_self_container() const noexcept { return m_is_self; }

    /// Offset within the raw file at which the plaintext ELF image begins
    /// (0 for a bare ELF, kSelfElfOffset for a SELF container).
    [[nodiscard]] usize elf_file_offset() const noexcept { return m_elf_file_offset; }

    /// The SELF entry table (empty for a bare ELF).
    [[nodiscard]] const std::vector<SelfEntry>& self_entries() const noexcept {
        return m_self_entries;
    }

    /// True when this ELF is position-independent (ET_DYN), i.e. a PRX that
    /// needs a load base and relocation before any of its addresses mean
    /// anything. A bare ET_EXEC is already linked at absolute addresses.
    [[nodiscard]] bool needs_load_base() const noexcept;

    /// Authoritative file offset for a specific segment's data.
    ///
    /// For a SELF container this comes from the entry table matched by p_filesz,
    /// because p_offset is stale. For a bare ELF it is simply p_offset.
    ///
    /// Takes the header rather than just its type on purpose: a module has two
    /// PT_LOAD segments with different file offsets, so a type-keyed lookup would
    /// silently hand the second one the first one's offset. Sets *found = false
    /// and returns 0 when nothing is known.
    [[nodiscard]] usize segment_file_offset(const Elf64ProgramHeader& ph,
                                            bool* found = nullptr) const;

    /// File offset of PT_DYNAMIC inside a SELF container.
    ///
    /// PT_DYNAMIC is not its own SELF segment -- it is a slice at the tail of
    /// PT_SCE_PROGLIB, so its p_offset is relative to that segment's stale
    /// origin. Both offsets carry the same error, so the difference survives:
    ///
    ///   proglib_real + (PT_DYNAMIC.p_offset - PT_SCE_PROGLIB.p_offset)
    ///
    /// which lands on a DT_NULL-terminated array right before the final two
    /// SELF entries. Returns 0 if the module is not a SELF container or the
    /// segment is absent.
    [[nodiscard]] usize dynamic_offset_file_offset() const;

    /// Offset of PT_SCE_PROGLIB's data within the raw file.
    [[nodiscard]] usize proglib_file_offset() const;

    /// Size of PT_SCE_PROGLIB's data.
    [[nodiscard]] usize proglib_file_size() const;

private:
    /// First program header with the given type, or nullptr.
    [[nodiscard]] const Elf64ProgramHeader* find_phdr(u32 p_type) const;

    bool parse_internal(std::span<const u8> buffer);
    bool parse_self(std::span<const u8> raw);

    Elf64Header m_header{};
    std::vector<Elf64ProgramHeader> m_program_headers;
    std::vector<SelfEntry> m_self_entries;
    usize m_elf_file_offset{0};
    ElfError m_error{ElfError::None};
    bool m_loaded{false};
    bool m_is_self{false};
};

void dump_module_dynamic(const std::filesystem::path& path);
} // namespace ps4
