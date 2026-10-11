#include "ps4/dynamic_linker.hpp"
#include <cstring>
#include <iostream>

namespace ps4 {

DynamicLinker::DynamicLinker(MemoryManager& memory, SyscallTable& syscalls, NidRegistry& registry)
    : m_memory(memory), m_syscalls(syscalls), m_registry(registry) {}

bool DynamicLinker::initialize(GuestAddr trampoline_base) {
    m_trampoline_base = trampoline_base;
    m_next_trampoline_id = 0;
    m_nid_to_trampoline.clear();
    m_id_to_nid.clear();

    // Reserve 64KB space for trampolines
    constexpr GuestSize TRAMPOLINE_POOL_SIZE = 0x10000;
    if (!m_memory.is_valid_address(m_trampoline_base, TRAMPOLINE_POOL_SIZE)) {
        return false;
    }

    m_memory.zero_memory(m_trampoline_base, TRAMPOLINE_POOL_SIZE);
    return true;
}

GuestAddr DynamicLinker::get_or_create_trampoline(Nid nid) {
    auto it = m_nid_to_trampoline.find(nid);
    if (it != m_nid_to_trampoline.end()) {
        return it->second;
    }

    const u32 id = m_next_trampoline_id++;
    const GuestAddr stub_addr = m_trampoline_base + (id * 16);

    // Emit 8-byte x86_64 stub:
    // B8 <imm32> : mov eax, (0x80000000 | id)  (5 bytes)
    // 0F 05      : syscall                      (2 bytes)
    // C3         : ret                          (1 byte)
    const u32 pseudo_syscall = HLE_SYSCALL_DISPATCH_BASE | id;
    u8 stub[8];
    stub[0] = 0xB8;
    std::memcpy(stub + 1, &pseudo_syscall, sizeof(pseudo_syscall));
    stub[5] = 0x0F;
    stub[6] = 0x05;
    stub[7] = 0xC3;

    m_memory.write_bytes(stub_addr, stub, sizeof(stub));

    // Hook pseudo-syscall in SyscallTable
    m_syscalls.register_handler(pseudo_syscall, [this, nid](SyscallContext& sctx) -> OrbisResult {
        const HleSymbol* sym = m_registry.find_by_nid(nid);
        if (sym && sym->handler) {
            std::fprintf(stderr, "[HLE CALL] %s (NID=0x%llx, rdi=0x%llx, rsi=0x%llx)\n",
                         sym->name.c_str(), (unsigned long long)nid, (unsigned long long)sctx.args[0], (unsigned long long)sctx.args[1]);
            CpuContext mock_cpu{};
            mock_cpu.rdi = sctx.args[0];
            mock_cpu.rsi = sctx.args[1];
            mock_cpu.rdx = sctx.args[2];
            mock_cpu.rcx = sctx.rcx ? sctx.rcx : sctx.args[3]; // Function call ABI uses RCX
            mock_cpu.r8  = sctx.args[4];
            mock_cpu.r9  = sctx.args[5];
            mock_cpu.rax = sctx.rax;
            mock_cpu.rsp = sctx.rsp;

            sym->handler(mock_cpu, m_memory);
            sctx.rax = mock_cpu.rax;

            // Trigger clean exit on sceKernelExitProcess
            constexpr Nid NID_EXIT_PROCESS = 0x6424D00705E3BEBEULL;
            if (nid == NID_EXIT_PROCESS) {
                m_syscalls.dispatch(SYS_exit, sctx);
            }
            return 0;
        }

        // Return SCE_OK (0) for unimplemented stubs to permit process initialization
        std::fprintf(stderr, "[HLE UNIMPLEMENTED] NID=0x%llx\n", (unsigned long long)nid);
        sctx.rax = 0;
        return 0;
    });

    m_nid_to_trampoline[nid] = stub_addr;
    m_id_to_nid[id] = nid;
    return stub_addr;
}

bool DynamicLinker::patch_jump_slot(GuestAddr got_slot_addr, Nid nid) {
    const GuestAddr trampoline_addr = get_or_create_trampoline(nid);
    return m_memory.write_value(got_slot_addr, trampoline_addr);
}

namespace {

inline Nid decode_nid_from_string(const char* s) {
    if (!s || !*s) return 0;
    std::string_view sv(s);
    auto hash = sv.find('#');
    if (hash != std::string_view::npos) {
        sv = sv.substr(0, hash);
    }
    if (sv.length() == 11) {
        static constexpr std::string_view alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
        __uint128_t val = 0;
        bool ok = true;
        for (char c : sv) {
            auto pos = alpha.find(c);
            if (pos == std::string_view::npos) { ok = false; break; }
            val = (val << 6) | static_cast<__uint128_t>(pos);
        }
        if (ok) return static_cast<u64>(val >> 2);
    }
    // FNV-1a hash
    u64 h = 14695981039346656037ULL;
    for (char c : sv) {
        h ^= static_cast<u8>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace

bool DynamicLinker::resolve_relocations(std::span<const u8> file_data, const ElfLoader& loader, GuestAddr base_load_addr) {
    GuestAddr rela_addr = 0;
    GuestSize rela_size = 0;
    GuestAddr jmprel_addr = 0;
    GuestSize pltrel_size = 0;
    GuestAddr symtab_addr = 0;
    GuestAddr strtab_addr = 0;
    [[maybe_unused]] GuestSize strtab_size = 0;

    // Scan for PT_SCE_PROGLIB / PT_SCE_DYNLIBDATA file offset
    usize proglib_file_offset = 0;
    usize proglib_file_size = 0;
    for (const auto& ph : loader.program_headers()) {
        if (ph.p_type == PT_SCE_PROGLIB || (ph.p_type == PT_SCE_DYNLIBDATA && ph.p_filesz > 0x100)) {
            if (ph.p_filesz > 0 && ph.p_offset + ph.p_filesz <= file_data.size()) {
                proglib_file_offset = static_cast<usize>(ph.p_offset);
                proglib_file_size = static_cast<usize>(ph.p_filesz);
                break;
            }
        }
    }

    // Scan for PT_DYNAMIC or PT_SCE_DYNLIBDATA
    for (const auto& ph : loader.program_headers()) {
        if (ph.p_type == PT_DYNAMIC || ph.p_type == PT_SCE_DYNLIBDATA) {
            const usize num_dyn = static_cast<usize>(ph.p_filesz / sizeof(Elf64_Dyn));
            if (ph.p_offset + ph.p_filesz <= file_data.size()) {
                const auto* dyn_table = reinterpret_cast<const Elf64_Dyn*>(file_data.data() + ph.p_offset);
                for (usize i = 0; i < num_dyn; ++i) {
                    const auto& dyn = dyn_table[i];
                    if (dyn.d_tag == DT_NULL) break;
                    if (dyn.d_tag == DT_RELA || dyn.d_tag == DT_SCE_RELA) rela_addr = dyn.d_un.d_ptr;
                    if (dyn.d_tag == DT_RELASZ || dyn.d_tag == DT_SCE_RELASZ) rela_size = dyn.d_un.d_val;
                    if (dyn.d_tag == DT_JMPREL || dyn.d_tag == DT_SCE_JMPREL) jmprel_addr = dyn.d_un.d_ptr;
                    if (dyn.d_tag == DT_PLTRELSZ || dyn.d_tag == DT_SCE_PLTRELSZ) pltrel_size = dyn.d_un.d_val;
                    if (dyn.d_tag == DT_SYMTAB || dyn.d_tag == DT_SCE_SYMTAB) symtab_addr = dyn.d_un.d_ptr;
                    if (dyn.d_tag == DT_STRTAB || dyn.d_tag == DT_SCE_STRTAB) strtab_addr = dyn.d_un.d_ptr;
                    if (dyn.d_tag == DT_STRSZ || dyn.d_tag == DT_SCE_STRSZ) strtab_size = dyn.d_un.d_val;
                }
            }
        }
    }

    auto read_data = [&](GuestAddr vaddr, void* dest, usize size) -> bool {
        // 1. If PROGLIB table is present and vaddr is within its bounds
        if (proglib_file_offset != 0 && (vaddr + size) <= proglib_file_size) {
            if (proglib_file_offset + vaddr + size <= file_data.size()) {
                std::memcpy(dest, file_data.data() + proglib_file_offset + vaddr, size);
                return true;
            }
        }

        // 2. Try to find the virtual address within any ELF program header in file_data
        for (const auto& ph : loader.program_headers()) {
            if (ph.p_filesz > 0 && (ph.p_type == PT_LOAD || ph.p_type == PT_SCE_RELRO || ph.p_type == PT_DYNAMIC)) {
                // If ELF is relocated at base_load_addr
                const GuestAddr seg_start = base_load_addr + ph.p_vaddr;
                if (vaddr >= seg_start && (vaddr + size) <= (seg_start + ph.p_filesz)) {
                    const usize offset_in_ph = static_cast<usize>(vaddr - seg_start);
                    const usize file_offset = static_cast<usize>(ph.p_offset + offset_in_ph);
                    if (file_offset + size <= file_data.size()) {
                        std::memcpy(dest, file_data.data() + file_offset, size);
                        return true;
                    }
                }
                // Check raw vaddr without base_load_addr
                if (base_load_addr != 0 && vaddr >= ph.p_vaddr && (vaddr + size) <= (ph.p_vaddr + ph.p_filesz)) {
                    const usize offset_in_ph = static_cast<usize>(vaddr - ph.p_vaddr);
                    const usize file_offset = static_cast<usize>(ph.p_offset + offset_in_ph);
                    if (file_offset + size <= file_data.size()) {
                        std::memcpy(dest, file_data.data() + file_offset, size);
                        return true;
                    }
                }
            }
        }

        // 3. Fallback to guest memory
        if (m_memory.read_bytes(vaddr, dest, size)) {
            return true;
        }

        // 4. Fallback: treat vaddr as direct file offset if within file bounds
        if (vaddr + size <= file_data.size()) {
            std::memcpy(dest, file_data.data() + vaddr, size);
            return true;
        }

        return false;
    };

    auto process_rela_table = [&](GuestAddr table_addr, GuestSize table_size) {
        if (table_addr == 0 || table_size == 0) return;
        const usize count = static_cast<usize>(table_size / sizeof(Elf64_Rela));
        for (usize i = 0; i < count; ++i) {
            const GuestAddr entry_addr = table_addr + (i * sizeof(Elf64_Rela));
            Elf64_Rela rela{};
            if (!read_data(entry_addr, &rela, sizeof(rela))) continue;

            const u32 type = ELF64_R_TYPE(rela.r_info);
            const u32 sym_idx = ELF64_R_SYM(rela.r_info);
            const GuestAddr patch_location = base_load_addr + rela.r_offset;

            if (type == R_X86_64_RELATIVE) {
                const GuestAddr value = base_load_addr + rela.r_addend;
                m_memory.write_value(patch_location, value);
            } else if (type == R_X86_64_64) {
                if (symtab_addr != 0) {
                    Elf64_Sym sym{};
                    const GuestAddr sym_entry_addr = symtab_addr + (sym_idx * sizeof(Elf64_Sym));
                    if (read_data(sym_entry_addr, &sym, sizeof(sym))) {
                        const GuestAddr value = sym.st_value + rela.r_addend;
                        m_memory.write_value(patch_location, value);
                    }
                }
            } else if (type == R_X86_64_JUMP_SLOT || type == R_X86_64_GLOB_DAT) {
                if (symtab_addr != 0) {
                    Elf64_Sym sym{};
                    const GuestAddr sym_entry_addr = symtab_addr + (sym_idx * sizeof(Elf64_Sym));
                    if (read_data(sym_entry_addr, &sym, sizeof(sym))) {
                        Nid nid = sym.st_value;
                        if (nid == 0 && strtab_addr != 0 && sym.st_name != 0) {
                            char sym_str[128]{0};
                            if (read_data(strtab_addr + sym.st_name, sym_str, sizeof(sym_str) - 1)) {
                                nid = decode_nid_from_string(sym_str);
                            }
                        }
                        if (nid == 0) {
                            nid = 0x8000000000000000ULL | static_cast<u64>(sym_idx);
                        }
                        patch_jump_slot(patch_location, nid);
                    }
                }
            }
        }
    };

    process_rela_table(rela_addr, rela_size);
    process_rela_table(jmprel_addr, pltrel_size);
    return true;
}

} // namespace ps4
