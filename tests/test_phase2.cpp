#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "ps4/elf_loader.hpp"

namespace {

// Helper to construct a synthetic valid ELF64 buffer
std::vector<ps4::u8> create_test_elf(
    ps4::u8 mag0 = ps4::ELFMAG0,
    ps4::u8 mag1 = ps4::ELFMAG1,
    ps4::u8 mag2 = ps4::ELFMAG2,
    ps4::u8 mag3 = ps4::ELFMAG3,
    ps4::u8 elf_class = ps4::ELFCLASS64,
    ps4::u8 elf_data = ps4::ELFDATA2LSB,
    ps4::u16 machine = ps4::EM_X86_64,
    ps4::u64 phoff = sizeof(ps4::Elf64Header),
    ps4::u16 phnum = 3
) {
    const ps4::usize total_size = static_cast<ps4::usize>(phoff) + (phnum * sizeof(ps4::Elf64ProgramHeader)) + 0x1000;
    std::vector<ps4::u8> buffer(total_size, 0);

    ps4::Elf64Header hdr{};
    hdr.e_ident[ps4::EI_MAG0]    = mag0;
    hdr.e_ident[ps4::EI_MAG1]    = mag1;
    hdr.e_ident[ps4::EI_MAG2]    = mag2;
    hdr.e_ident[ps4::EI_MAG3]    = mag3;
    hdr.e_ident[ps4::EI_CLASS]   = elf_class;
    hdr.e_ident[ps4::EI_DATA]    = elf_data;
    hdr.e_ident[ps4::EI_VERSION] = 1;
    hdr.e_ident[ps4::EI_OSABI]   = ps4::ELFOSABI_FREEBSD;

    hdr.e_type      = ps4::ET_SCE_EXEC;
    hdr.e_machine   = machine;
    hdr.e_version   = 1;
    hdr.e_entry     = 0x400000ULL;
    hdr.e_phoff     = phoff;
    hdr.e_ehsize    = sizeof(ps4::Elf64Header);
    hdr.e_phentsize = sizeof(ps4::Elf64ProgramHeader);
    hdr.e_phnum     = phnum;

    std::memcpy(buffer.data(), &hdr, sizeof(hdr));

    if (phnum >= 1 && (phoff + sizeof(ps4::Elf64ProgramHeader) <= buffer.size())) {
        // Segment 0: Code (PT_LOAD, RX)
        ps4::Elf64ProgramHeader ph0{};
        ph0.p_type   = ps4::PT_LOAD;
        ph0.p_flags  = ps4::PF_R | ps4::PF_X;
        ph0.p_offset = 0x1000;
        ph0.p_vaddr  = 0x400000ULL;
        ph0.p_filesz = 0x800;
        ph0.p_memsz  = 0x800;
        ph0.p_align  = 0x4000;
        std::memcpy(buffer.data() + phoff, &ph0, sizeof(ph0));
    }

    if (phnum >= 2 && (phoff + 2 * sizeof(ps4::Elf64ProgramHeader) <= buffer.size())) {
        // Segment 1: Data / BSS (PT_LOAD, RW)
        ps4::Elf64ProgramHeader ph1{};
        ph1.p_type   = ps4::PT_LOAD;
        ph1.p_flags  = ps4::PF_R | ps4::PF_W;
        ph1.p_offset = 0x2000;
        ph1.p_vaddr  = 0x600000ULL;
        ph1.p_filesz = 0x400;
        ph1.p_memsz  = 0x600; // memsz > filesz indicates .bss
        ph1.p_align  = 0x4000;
        std::memcpy(buffer.data() + phoff + sizeof(ps4::Elf64ProgramHeader), &ph1, sizeof(ph1));
    }

    if (phnum >= 3 && (phoff + 3 * sizeof(ps4::Elf64ProgramHeader) <= buffer.size())) {
        // Segment 2: Sony Orbis custom segment (PT_SCE_PROGLIB)
        ps4::Elf64ProgramHeader ph2{};
        ph2.p_type   = ps4::PT_SCE_PROGLIB;
        ph2.p_flags  = ps4::PF_R;
        ph2.p_offset = 0x3000;
        ph2.p_vaddr  = 0x700000ULL;
        ph2.p_filesz = 0x200;
        ph2.p_memsz  = 0x200;
        ph2.p_align  = 0x10;
        std::memcpy(buffer.data() + phoff + 2 * sizeof(ps4::Elf64ProgramHeader), &ph2, sizeof(ph2));
    }

    return buffer;
}

} // namespace

int main() {
    std::cout << "[PHASE 2] Starting ELF Loader unit tests...\n";

    // Test 1: Valid ELF64 parsing in memory
    {
        auto valid_buffer = create_test_elf();
        ps4::ElfLoader loader;
        bool ok = loader.load_from_memory(valid_buffer);
        assert(ok);
        assert(loader.is_valid());
        assert(loader.error() == ps4::ElfError::None);
        assert(loader.entry_point() == 0x400000ULL);
        assert(loader.header().e_type == ps4::ET_SCE_EXEC);
        assert(loader.header().e_machine == ps4::EM_X86_64);
        assert(loader.program_headers().size() == 3);

        auto loads = loader.load_segments();
        assert(loads.size() == 2);
        assert(loads[0].p_vaddr == 0x400000ULL);
        assert(loads[0].p_flags == (ps4::PF_R | ps4::PF_X));
        assert(loads[1].p_vaddr == 0x600000ULL);
        assert(loads[1].p_filesz == 0x400);
        assert(loads[1].p_memsz == 0x600);

        std::cout << "  ✓ Test 1: Valid in-memory ELF64 parsed successfully\n";
    }

    // Test 2: File too small
    {
        std::vector<ps4::u8> small_buffer(32, 0);
        ps4::ElfLoader loader;
        bool ok = loader.load_from_memory(small_buffer);
        assert(!ok);
        assert(!loader.is_valid());
        assert(loader.error() == ps4::ElfError::FileTooSmall);
        std::cout << "  ✓ Test 2: FileTooSmall rejected correctly\n";
    }

    // Test 3: Invalid magic bytes
    {
        auto bad_magic = create_test_elf('B', 'A', 'D', '!');
        ps4::ElfLoader loader;
        bool ok = loader.load_from_memory(bad_magic);
        assert(!ok);
        assert(!loader.is_valid());
        assert(loader.error() == ps4::ElfError::InvalidMagic);
        std::cout << "  ✓ Test 3: InvalidMagic rejected correctly\n";
    }

    // Test 4: Non-64-bit class
    {
        auto elf32 = create_test_elf(ps4::ELFMAG0, ps4::ELFMAG1, ps4::ELFMAG2, ps4::ELFMAG3, ps4::ELFCLASS32);
        ps4::ElfLoader loader;
        bool ok = loader.load_from_memory(elf32);
        assert(!ok);
        assert(loader.error() == ps4::ElfError::Not64Bit);
        std::cout << "  ✓ Test 4: Not64Bit rejected correctly\n";
    }

    // Test 5: Big-endian encoding
    {
        auto big_endian = create_test_elf(ps4::ELFMAG0, ps4::ELFMAG1, ps4::ELFMAG2, ps4::ELFMAG3, ps4::ELFCLASS64, ps4::ELFDATA2MSB);
        ps4::ElfLoader loader;
        bool ok = loader.load_from_memory(big_endian);
        assert(!ok);
        assert(loader.error() == ps4::ElfError::NotLittleEndian);
        std::cout << "  ✓ Test 5: NotLittleEndian rejected correctly\n";
    }

    // Test 6: Invalid architecture (e.g. ARM64 0xB7 instead of x86-64 0x3E)
    {
        auto arm_elf = create_test_elf(ps4::ELFMAG0, ps4::ELFMAG1, ps4::ELFMAG2, ps4::ELFMAG3, ps4::ELFCLASS64, ps4::ELFDATA2LSB, 0xB7);
        ps4::ElfLoader loader;
        bool ok = loader.load_from_memory(arm_elf);
        assert(!ok);
        assert(loader.error() == ps4::ElfError::InvalidArchitecture);
        std::cout << "  ✓ Test 6: InvalidArchitecture rejected correctly\n";
    }

    // Test 7: Program header offset out of bounds
    {
        auto oob_phoff = create_test_elf();
        ps4::Elf64_Off corrupted_phoff = oob_phoff.size() + 0x1000;
        std::memcpy(oob_phoff.data() + offsetof(ps4::Elf64Header, e_phoff), &corrupted_phoff, sizeof(corrupted_phoff));

        ps4::ElfLoader loader;
        bool ok = loader.load_from_memory(oob_phoff);
        assert(!ok);
        assert(loader.error() == ps4::ElfError::InvalidProgramHeadersOffset);
        std::cout << "  ✓ Test 7: InvalidProgramHeadersOffset rejected correctly\n";
    }

    // Test 8: File I/O loading
    {
        const std::filesystem::path temp_file = "test_synthetic.elf";
        auto buf = create_test_elf();
        {
            std::ofstream out(temp_file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        }

        ps4::ElfLoader loader;
        bool ok = loader.load_from_file(temp_file);
        assert(ok);
        assert(loader.is_valid());
        assert(loader.entry_point() == 0x400000ULL);
        assert(loader.program_headers().size() == 3);

        std::filesystem::remove(temp_file);

        // Test non-existent file
        bool ok_missing = loader.load_from_file("non_existent_file.elf");
        assert(!ok_missing);
        assert(loader.error() == ps4::ElfError::FileNotFound);

        std::cout << "  ✓ Test 8: Disk file I/O and FileNotFound tested successfully\n";
    }

    std::cout << "[PHASE 2] PASS: All ELF Loader tests passed!\n";
    return 0;
}
