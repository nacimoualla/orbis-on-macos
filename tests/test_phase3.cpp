#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include "ps4/elf_loader.hpp"
#include "ps4/memory.hpp"

namespace {

std::vector<ps4::u8> create_test_elf_with_payload() {
    const ps4::usize total_size = 0x4000;
    std::vector<ps4::u8> buffer(total_size, 0);

    ps4::Elf64Header hdr{};
    hdr.e_ident[ps4::EI_MAG0]    = ps4::ELFMAG0;
    hdr.e_ident[ps4::EI_MAG1]    = ps4::ELFMAG1;
    hdr.e_ident[ps4::EI_MAG2]    = ps4::ELFMAG2;
    hdr.e_ident[ps4::EI_MAG3]    = ps4::ELFMAG3;
    hdr.e_ident[ps4::EI_CLASS]   = ps4::ELFCLASS64;
    hdr.e_ident[ps4::EI_DATA]    = ps4::ELFDATA2LSB;
    hdr.e_ident[ps4::EI_VERSION] = 1;
    hdr.e_ident[ps4::EI_OSABI]   = ps4::ELFOSABI_FREEBSD;

    hdr.e_type      = ps4::ET_SCE_EXEC;
    hdr.e_machine   = ps4::EM_X86_64;
    hdr.e_version   = 1;
    hdr.e_entry     = 0x400000ULL;
    hdr.e_phoff     = sizeof(ps4::Elf64Header);
    hdr.e_ehsize    = sizeof(ps4::Elf64Header);
    hdr.e_phentsize = sizeof(ps4::Elf64ProgramHeader);
    hdr.e_phnum     = 2;

    std::memcpy(buffer.data(), &hdr, sizeof(hdr));

    // Segment 0: Code (PT_LOAD, RX, offset 0x1000, vaddr 0x400000, size 0x100)
    ps4::Elf64ProgramHeader ph0{};
    ph0.p_type   = ps4::PT_LOAD;
    ph0.p_flags  = ps4::PF_R | ps4::PF_X;
    ph0.p_offset = 0x1000;
    ph0.p_vaddr  = 0x400000ULL;
    ph0.p_filesz = 0x100;
    ph0.p_memsz  = 0x100;
    ph0.p_align  = 0x4000;
    std::memcpy(buffer.data() + hdr.e_phoff, &ph0, sizeof(ph0));

    // Write code payload: string "HELLO_ORBIS_CODE"
    const char code_payload[] = "HELLO_ORBIS_CODE";
    std::memcpy(buffer.data() + 0x1000, code_payload, sizeof(code_payload));

    // Segment 1: Data + BSS (PT_LOAD, RW, offset 0x2000, vaddr 0x600000, filesz 0x50, memsz 0x100)
    ps4::Elf64ProgramHeader ph1{};
    ph1.p_type   = ps4::PT_LOAD;
    ph1.p_flags  = ps4::PF_R | ps4::PF_W;
    ph1.p_offset = 0x2000;
    ph1.p_vaddr  = 0x600000ULL;
    ph1.p_filesz = 0x50;
    ph1.p_memsz  = 0x100; // Extra 0xB0 bytes for .bss
    ph1.p_align  = 0x4000;
    std::memcpy(buffer.data() + hdr.e_phoff + sizeof(ph0), &ph1, sizeof(ph1));

    // Write data payload: string "DATA_PAYLOAD"
    const char data_payload[] = "DATA_PAYLOAD";
    std::memcpy(buffer.data() + 0x2000, data_payload, sizeof(data_payload));

    return buffer;
}

} // namespace

int main() {
    std::cout << "[PHASE 3] Starting Virtual Memory Subsystem unit tests...\n";

    // Test 1: 8GB allocation and basic properties
    {
        ps4::MemoryManager mem;
        assert(!mem.is_initialized());
        bool ok = mem.initialize();
        assert(ok);
        assert(mem.is_initialized());
        assert(mem.base_address() != nullptr);
        assert(mem.total_size() == ps4::GUEST_MEMORY_SIZE);

        // Cannot initialize twice
        assert(!mem.initialize());
        assert(mem.last_error() == ps4::MemoryError::AlreadyInitialized);
        std::cout << "  ✓ Test 1: 8GB Virtual Memory allocation and initialization verified\n";
    }

    // Test 2: Address translation and bounds checking
    {
        ps4::MemoryManager mem;
        assert(mem.initialize());

        ps4::GuestAddr addr = 0x00400000ULL;
        void* host_ptr = mem.to_host_ptr(addr);
        assert(host_ptr != nullptr);
        assert(host_ptr == static_cast<void*>(mem.base_address() + addr));

        ps4::GuestAddr round_trip = mem.to_guest_addr(host_ptr);
        assert(round_trip == addr);

        // Valid boundary addresses
        assert(mem.is_valid_address(0, 1));
        assert(mem.is_valid_address(ps4::GUEST_MEMORY_SIZE - 1, 1));
        assert(mem.is_valid_address(0, ps4::GUEST_MEMORY_SIZE));

        // Out-of-bounds addresses
        assert(!mem.is_valid_address(ps4::GUEST_MEMORY_SIZE, 1));
        assert(!mem.is_valid_address(ps4::GUEST_MEMORY_SIZE + 0x1000, 1));
        assert(!mem.is_valid_address(ps4::GUEST_MEMORY_SIZE - 10, 20));
        assert(mem.to_host_ptr(ps4::GUEST_MEMORY_SIZE) == nullptr);

        std::cout << "  ✓ Test 2: Address translation and boundary checks verified\n";
    }

    // Test 3: Memory read, write, and zeroing
    {
        ps4::MemoryManager mem;
        assert(mem.initialize());

        // Write typed value
        const ps4::u64 expected_val = 0xCAFEBABE12345678ULL;
        assert(mem.write_value(0x100000ULL, expected_val));

        ps4::u64 read_val = 0;
        assert(mem.read_value(0x100000ULL, read_val));
        assert(read_val == expected_val);

        // Zero memory
        assert(mem.zero_memory(0x100000ULL, sizeof(ps4::u64)));
        assert(mem.read_value(0x100000ULL, read_val));
        assert(read_val == 0);

        // Out of bounds write fails
        assert(!mem.write_value(ps4::GUEST_MEMORY_SIZE, expected_val));
        assert(mem.last_error() == ps4::MemoryError::OutOfBounds);

        std::cout << "  ✓ Test 3: Read/Write primitives and zero_memory verified\n";
    }

    // Test 4: ELF segment loading into guest memory
    {
        auto elf_buffer = create_test_elf_with_payload();
        ps4::ElfLoader loader;
        assert(loader.load_from_memory(elf_buffer));

        ps4::MemoryManager mem;
        assert(mem.initialize());
        assert(mem.load_elf_segments(loader, elf_buffer));

        // Check code segment at 0x400000
        const char* code_str = mem.to_host_ptr<char>(0x400000ULL);
        assert(code_str != nullptr);
        assert(std::strcmp(code_str, "HELLO_ORBIS_CODE") == 0);

        // Check data segment at 0x600000
        const char* data_str = mem.to_host_ptr<char>(0x600000ULL);
        assert(data_str != nullptr);
        assert(std::strcmp(data_str, "DATA_PAYLOAD") == 0);

        // Check .bss segment expansion is zeroed
        // filesz was 0x50, memsz was 0x100 (so offset 0x600050 to 0x600100 must be 0)
        const ps4::u8* bss_region = mem.to_host_ptr<ps4::u8>(0x600050ULL);
        assert(bss_region != nullptr);
        for (ps4::usize i = 0; i < (0x100 - 0x50); ++i) {
            assert(bss_region[i] == 0);
        }

        std::cout << "  ✓ Test 4: ELF PT_LOAD segments and .bss zero-fill loaded successfully\n";
    }

    // Test 5: Move semantics
    {
        ps4::MemoryManager mem1;
        assert(mem1.initialize());
        ps4::u8* base_ptr = mem1.base_address();

        ps4::MemoryManager mem2 = std::move(mem1);
        assert(!mem1.is_initialized());
        assert(mem1.base_address() == nullptr);
        assert(mem2.is_initialized());
        assert(mem2.base_address() == base_ptr);

        std::cout << "  ✓ Test 5: Move semantics verified\n";
    }

    std::cout << "[PHASE 3] PASS: All Virtual Memory Subsystem tests passed!\n";
    return 0;
}
