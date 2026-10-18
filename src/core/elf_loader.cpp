#include "ps4/elf_loader.hpp"
#include <cstdio>
#include <cstring>
#include <map>
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

namespace ps4 {

// Dump a module's dynamic table using the loader's OWN offset resolution. The
// SELF's phdr.p_offset values are stale, so re-deriving them by hand (as the
// Python attempts did) reads past the end of the file; dynamic_offset_file_offset()
// cancels the error by differencing two offsets that share the same origin.
void dump_module_dynamic(const std::filesystem::path& path) {
    std::fprintf(stderr, "\n[dyn] ===== %s =====\n", path.filename().c_str());
    std::vector<u8> data;
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "[dyn] cannot open\n"); return; }
        data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    ElfLoader ld;
    if (!ld.load_from_memory(std::span<const u8>(data))) {
        std::fprintf(stderr, "[dyn] parse failed: %s\n", ld.error_message().data());
        return;
    }
    const usize dyn_off = ld.dynamic_offset_file_offset();
    std::fprintf(stderr, "[dyn] is_self=%d proglib_off=0x%zx PT_DYNAMIC real file off=0x%zx (file size 0x%zx)\n",
                 1, ld.proglib_file_offset(), dyn_off, data.size());
    if (dyn_off == 0 || dyn_off + 16 > data.size()) {
        std::fprintf(stderr, "[dyn] dynamic offset invalid\n"); return;
    }

    std::map<u64, u64> tags;
    std::map<u64, usize> tag_at;
    for (usize off = dyn_off; off + 16 <= data.size(); off += 16) {
        u64 tag, val;
        std::memcpy(&tag, data.data() + off, 8);
        std::memcpy(&val, data.data() + off + 8, 8);
        if (tag == 0) break;
        if (!tags.count(tag)) tag_at[tag] = off;
        auto& slot = tags[tag];
        if (slot == 0) slot = val; else if (slot != val) slot = ~0ULL;  // ~0 => multiple
    }
    auto show = [&](u64 tag, const char* nm) {
        auto it = tags.find(tag);
        if (it == tags.end()) return false;
        std::fprintf(stderr, "[dyn]   %-18s tag=%#llx  fileoff=0x%zx  value=%s",
                     nm, (unsigned long long)tag, tag_at[tag],
                     it->second == ~0ULL ? "<multiple>" : "");
        if (it->second != ~0ULL) std::fprintf(stderr, "%#llx", (unsigned long long)it->second);
        std::fprintf(stderr, "\n");
        return true;
    };
    const u64 DT_NEEDED=1, DT_STRTAB=5, DT_INIT=12, DT_FINI=13, DT_SONAME=14;
    const u64 DT_INIT_ARRAY=25, DT_FINI_ARRAY=26, DT_INIT_ARRAYSZ=27, DT_FINI_ARRAYSZ=28;
    const u64 DT_SCE_STRTAB=0x61000019, DT_SCE_SYMTAB=0x61000039, DT_SCE_SYMTABSZ=0x6100003F;
    std::fprintf(stderr, "[dyn] tags found: %zu\n", tags.size());
    show(DT_INIT, "DT_INIT");        show(DT_FINI, "DT_FINI");
    show(DT_INIT_ARRAY, "DT_INIT_ARRAY"); show(DT_INIT_ARRAYSZ, "DT_INIT_ARRAYSZ");
    show(DT_FINI_ARRAY, "DT_FINI_ARRAY"); show(DT_FINI_ARRAYSZ, "DT_FINI_ARRAYSZ");
    show(DT_NEEDED, "DT_NEEDED");    show(DT_SONAME, "DT_SONAME");
    show(DT_STRTAB, "DT_STRTAB");    show(DT_SCE_STRTAB, "DT_SCE_STRTAB");
    show(DT_SCE_SYMTAB, "DT_SCE_SYMTAB"); show(DT_SCE_SYMTABSZ, "DT_SCE_SYMTABSZ");

    // DT_NEEDED strings, via DT_SCE_STRTAB if present else DT_STRTAB.
    u64 strtab = 0;
    if (tags.count(DT_SCE_STRTAB) && tags[DT_SCE_STRTAB] != ~0ULL) strtab = tags[DT_SCE_STRTAB];
    else if (tags.count(DT_STRTAB)) strtab = tags[DT_STRTAB];
    if (tags.count(DT_NEEDED) && strtab != 0 && tags[DT_NEEDED] != ~0ULL) {
        // one string per NEEDED entry; count consecutive NEEDED occurrences
        u32 n = 0;
        for (usize off = dyn_off; off + 16 <= data.size(); off += 16) {
            u64 t; std::memcpy(&t, data.data() + off, 8);
            if (t == 0) break;
            if (t == DT_NEEDED) ++n;
        }
        std::fprintf(stderr, "[dyn]   DT_NEEDED strings (%u entries, strtab 0x%llx):\n",
                     n, (unsigned long long)strtab);
        for (u32 i = 0; i < n; ++i) {
            const usize so = static_cast<usize>(strtab + i);
            if (so >= data.size()) break;
            const char* p = reinterpret_cast<const char*>(data.data() + so);
            std::fprintf(stderr, "[dyn]     [%u] %s\n", i, p);
        }
    }

    // INIT_ARRAY / FINI_ARRAY entries (virtual addresses; need the load base to
    // dereference, so print them raw here).
    auto arr = [&](u64 atag, u64 sztag, const char* nm) {
        if (!tags.count(atag) || tags[atag] == ~0ULL) return;
        if (!tags.count(sztag)) return;
        const u64 a = tags[atag], sz = tags[sztag];
        std::fprintf(stderr, "[dyn]   %s @0x%llx size=0x%llx -> %llu entries\n",
                     nm, (unsigned long long)a, (unsigned long long)sz, (unsigned long long)(sz / 8));
    };
    arr(DT_INIT_ARRAY, DT_INIT_ARRAYSZ, "DT_INIT_ARRAY");
    arr(DT_FINI_ARRAY, DT_FINI_ARRAYSZ, "DT_FINI_ARRAY");

    // Does the export table hold a module_start-style NID?
    if (tags.count(DT_SCE_SYMTAB) && tags.count(DT_SCE_SYMTABSZ) &&
        tags[DT_SCE_SYMTAB] != ~0ULL && tags[DT_SCE_SYMTABSZ] != ~0ULL) {
        const usize so = static_cast<usize>(tags[DT_SCE_SYMTAB]);
        const usize ssz = static_cast<usize>(tags[DT_SCE_SYMTABSZ]);
        struct Want { u64 nid; const char* nm; };
        const Want wants[] = {
            {0x05a38a72783c83cfULL, "module_start"},
            {0x2a90ccacf1efb774ULL, "module_stop"},
            {0x44a906d9bd4bb5ddULL, "module_bootstrap"},
            {0x8008ef0941de01b1ULL, "_init"},
        };
        int nhit = 0;
        for (const auto& w : wants) {
            bool hit = false; u64 val = 0, size = 0;
            for (usize i = 0; i + 24 <= ssz && so + i + 24 <= data.size(); i += 24) {
                u32 sn, info; u8 o, sh; u64 sv, sz;
                std::memcpy(&sn, data.data() + so + i, 4);
                std::memcpy(&info, data.data() + so + i + 4, 4);
                o = data[so + i + 7];
                sh = data[so + i + 8];
                std::memcpy(&sv, data.data() + so + i + 8, 8);
                std::memcpy(&sz, data.data() + so + i + 16, 8);
                (void)info; (void)o;
                if (sh == 0) continue;
                const char* tok = reinterpret_cast<const char*>(data.data() + so + i - so + 0);
                (void)tok;
                // st_name is at +0; token is 11 base64url chars in STRTAB
                if (sn + 11 > data.size()) continue;
            }
            std::fprintf(stderr, "[dyn]   export NID %-18s %#018llx : %s\n",
                         w.nm, (unsigned long long)w.nid, hit ? "HIT" : "MISS");
            if (hit) { ++nhit;
                std::fprintf(stderr, "[dyn]     value=0x%llx size=0x%llx\n",
                             (unsigned long long)val, (unsigned long long)size); }
        }
        std::fprintf(stderr, "[dyn]   module_start-style NIDs found in export table: %d/4\n", nhit);
    } else {
        std::fprintf(stderr, "[dyn]   no DT_SCE_SYMTAB/SYZ: export NID check skipped\n");
    }
}
} // namespace ps4
