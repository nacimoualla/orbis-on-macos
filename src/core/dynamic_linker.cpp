#include "ps4/dynamic_linker.hpp"
#include "ps4/thread.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ps4 {

DynamicLinker::DynamicLinker(MemoryManager& memory, SyscallTable& syscalls, NidRegistry& registry)
    : m_memory(memory), m_syscalls(syscalls), m_registry(registry) {
    // Initialize default benign whitelist with CRT bootstrap stubs:
    // 0x6F3404C72D7CF592ULL (_init_env / bzQExy189ZI)
    // 0xF06D8B07E037AF38ULL (atexit / 8G2LB+A3rzg)
    m_whitelist.insert(0x6F3404C72D7CF592ULL);
    m_whitelist.insert(0xF06D8B07E037AF38ULL);
}

void DynamicLinker::set_strict_hle(bool enabled) noexcept {
    m_strict_hle = enabled;
}

bool DynamicLinker::is_strict_hle() const noexcept {
    return m_strict_hle;
}

void DynamicLinker::add_to_whitelist(Nid nid) {
    m_whitelist.insert(nid);
}

bool DynamicLinker::is_whitelisted(Nid nid) const noexcept {
    return m_whitelist.contains(nid);
}

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
    m_registry.set_syscall_table(&m_syscalls);
    return true;
}

namespace {

// Unimplemented-NID warning throttle state. Poll loops can hit one stub
// millions of times; the dispatch gate below caps warning+backtrace output per
// (NID, caller) so a single hot loop can never balloon the log again.
// Mutex-guarded: dispatches run on many host threads concurrently.
struct UnimplWarnEntry { Nid nid{0}; GuestAddr caller{0}; unsigned printed{0}; };
std::mutex s_unimpl_warn_mtx;
std::unordered_map<u64, UnimplWarnEntry> s_unimpl_warned;
unsigned long long s_unimpl_suppressed_total{0};

u64 unimpl_warn_key(Nid nid, GuestAddr caller) {
    return nid ^ (caller * 0x9e3779b97f4a7c15ULL);
}

} // namespace

void dump_hle_warning_stats() {
    struct Row { Nid nid; GuestAddr caller; unsigned printed; };
    std::vector<Row> rows;
    unsigned long long suppressed = 0;
    {
        std::lock_guard<std::mutex> lock(s_unimpl_warn_mtx);
        suppressed = s_unimpl_suppressed_total;
        for (const auto& [key, e] : s_unimpl_warned) {
            (void)key;
            rows.push_back({e.nid, e.caller, e.printed});
        }
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.printed > b.printed; });
    std::fprintf(stderr, "=== [HLE WARNING] suppression stats: %llu silenced after cap ===\n", suppressed);
    for (usize i = 0; i < rows.size() && i < 10; ++i) {
        std::fprintf(stderr, "  NID=0x%016llx caller=0x%llx printed=%u\n",
                     (unsigned long long)rows[i].nid,
                     (unsigned long long)rows[i].caller, rows[i].printed);
    }
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
        note_hle_activity_for(current_hle_thread_id());
        const HleSymbol* sym = m_registry.find_by_nid(nid);
        if (sym && sym->handler) {
            GuestAddr ret_addr = 0;
            m_memory.read_value(sctx.rsp, ret_addr);
            // Post-stall verbose: after the first stall dump, log everything
            // except keepalive TID 5's loop (4th worker spawned; stable order
            // in every run observed). Costs nothing pre-stall (flag off).
            const ScePthreadId hle_tid = current_hle_thread_id();
            const bool verbose = post_stall_verbose() && hle_tid != 5;
            // Suppressed high-frequency names still count toward hles=, so the
            // two totals differ by design; report the suppressed tally alongside
            // the line so the gap is auditable instead of looking like a
            // cross-run counter mismatch.
            static std::atomic<u64> s_suppressed{0};
            static std::atomic<u64> s_logged{0};
            static const bool s_log_hle = (std::getenv("PS4_LOG_HLE") != nullptr && std::strcmp(std::getenv("PS4_LOG_HLE"), "1") == 0);
            if ((s_log_hle || verbose) && (sym->name != "powf" && sym->name != "memcmp" && sym->name != "memset" &&
                sym->name != "strlen" && sym->name != "memcpy" && sym->name != "memmove" &&
                sym->name != "strcmp" && sym->name != "strncmp" && sym->name != "strncpy" &&
                sym->name != "malloc" && sym->name != "free" && sym->name != "memalign" &&
                sym->name != "mono_add_internal_call" &&
                sym->name != "sceKernelClockGettime" &&
                sym->name != "sceKernelGetProcessTimeCounter" &&
                sym->name != "scePthreadMutexLock" &&
                sym->name != "scePthreadMutexUnlock" &&
                sym->name != "scePthreadMutexTrylock" &&
                sym->name != "scePthreadSelf" &&
                sym->name != "sceAudio3dObjectSetAttributes" &&
                sym->name != "sceAudio3dPortFlush" &&
                sym->name != "sceKernelUsleep" &&
                sym->name != "sceKernelWaitSema" &&
                sym->name != "sceKernelSignalSema" &&
                sym->name != "sceKernelSyncStub")) {
                const u64 n = s_logged.fetch_add(1, std::memory_order_relaxed) + 1;
                // Log the head densely, then sample. Unbounded, this trace produced
                // 16 GB of log on a 30 B-instruction run: mono_runtime_invoke alone
                // fires ~400 M times when nothing consumes its return value, which
                // buried the watch ring we armed. Set PS4_LOG_HLE=all to restore
                // the full trace when that is genuinely what you want.
                static const bool s_log_all = [] {
                    const char* v = std::getenv("PS4_LOG_HLE");
                    return v && std::string_view(v) == "all";
                }();
                const bool is_gnm = (sym->name.rfind("sceGnm", 0) == 0);
                if (!s_log_all && !is_gnm && n > 2000 && (n % 100000) != 0) {
                    s_suppressed.fetch_add(1, std::memory_order_relaxed);
                    // fall through to the handler without printing
                } else {
                std::fprintf(stderr, "[HLE CALL] T%-3u %s (NID=0x%llx, caller RIP=0x%llx, rdi=0x%llx, rsi=0x%llx, rdx=0x%llx, rcx=0x%llx, r8=0x%llx, r9=0x%llx) [logged=%llu suppressed=%llu]\n",
                             (unsigned)hle_tid,
                             sym->name.c_str(), (unsigned long long)nid, (unsigned long long)ret_addr,
                             (unsigned long long)sctx.args[0], (unsigned long long)sctx.args[1],
                             (unsigned long long)sctx.args[2], (unsigned long long)(sctx.rcx ? sctx.rcx : sctx.args[3]),
                             (unsigned long long)sctx.args[4], (unsigned long long)sctx.args[5],
                             (unsigned long long)n,
                             (unsigned long long)s_suppressed.load(std::memory_order_relaxed));
                }
            } else {
                s_suppressed.fetch_add(1, std::memory_order_relaxed);
            }
            CpuContext* cur_cpu = sctx.cpu;
            CpuContext mock_cpu{};
            if (!cur_cpu) {
                mock_cpu.rdi = sctx.args[0];
                mock_cpu.rsi = sctx.args[1];
                mock_cpu.rdx = sctx.args[2];
                mock_cpu.rcx = sctx.rcx ? sctx.rcx : sctx.args[3]; // Function call ABI uses RCX
                mock_cpu.r8  = sctx.args[4];
                mock_cpu.r9  = sctx.args[5];
                mock_cpu.rax = sctx.rax;
                mock_cpu.rsp = sctx.rsp;
                mock_cpu.rbp = sctx.rbp;
                cur_cpu = &mock_cpu;
            }

            const GuestAddr initial_rip = cur_cpu->rip;
            m_registry.note_hle_dispatch(nid, ret_addr);
            t_guest_in_hle = true;
            sym->handler(*cur_cpu, m_memory);
            t_guest_in_hle = false;
            m_registry.note_hle_return();
            sctx.rax = cur_cpu->rax;
            sctx.rsp = cur_cpu->rsp;
            sctx.rbp = cur_cpu->rbp;

            if (cur_cpu->rip != initial_rip) {
                sctx.cpu_rip_modified = true;
            }

            // Trigger clean exit on sceKernelExitProcess
            constexpr Nid NID_EXIT_PROCESS = 0x6424D00705E3BEBEULL;
            if (nid == NID_EXIT_PROCESS) {
                m_syscalls.dispatch(SYS_exit, sctx);
            }
            return 0;
        }

        // 1. If NID is in m_whitelist: benign stub execution
        if (is_whitelisted(nid)) {
            GuestAddr ret_addr = 0;
            m_memory.read_value(sctx.rsp, ret_addr);
            std::fprintf(stderr, "[HLE WHITELIST] Benign stub executed: NID=0x%016llx (caller RIP=0x%llx)\n",
                         (unsigned long long)nid, (unsigned long long)ret_addr);
            sctx.rax = SCE_OK;
            if (sctx.cpu) {
                sctx.cpu->rax = SCE_OK;
            }
            return 0;
        }

        // 2. If NID is NOT in m_whitelist and strict HLE is enabled: crash diagnostically
        if (m_strict_hle) {
            GuestAddr ret_addr = 0;
            m_memory.read_value(sctx.rsp, ret_addr);
            std::fprintf(stderr, "\n======================================================================\n");
            std::fprintf(stderr, "[HLE CRASH] Strict HLE violation: Unregistered/unimplemented NID=0x%016llx\n",
                         (unsigned long long)nid);
            std::fprintf(stderr, "Caller RIP=0x%llx, RSP=0x%llx, RBP=0x%llx\n",
                         (unsigned long long)ret_addr, (unsigned long long)sctx.rsp, (unsigned long long)sctx.rbp);
            std::fprintf(stderr, "======================================================================\n");
            std::fprintf(stderr, "[HLE CRASH] Register State:\n");
            if (sctx.cpu != nullptr) {
                sctx.cpu->dump();
            } else {
                std::fprintf(stderr, "  RAX: 0x%016llx  RBX: 0x%016llx  RCX: 0x%016llx  RDX: 0x%016llx\n",
                             (unsigned long long)sctx.rax, 0ULL, (unsigned long long)sctx.rcx, (unsigned long long)sctx.args[2]);
                std::fprintf(stderr, "  RSI: 0x%016llx  RDI: 0x%016llx  RBP: 0x%016llx  RSP: 0x%016llx\n",
                             (unsigned long long)sctx.args[1], (unsigned long long)sctx.args[0], (unsigned long long)sctx.rbp, (unsigned long long)sctx.rsp);
                std::fprintf(stderr, "  R8 : 0x%016llx  R9 : 0x%016llx\n",
                             (unsigned long long)sctx.args[4], (unsigned long long)sctx.args[5]);
            }
            std::fprintf(stderr, "[HLE CRASH] Call Stack:\n");
            GuestAddr cur_rbp = sctx.rbp;
            for (int f = 0; f < 16 && cur_rbp > 0x1000 && m_memory.is_valid_address(cur_rbp, 16); ++f) {
                GuestAddr saved_rip = 0;
                GuestAddr saved_rbp = 0;
                m_memory.read_value(cur_rbp + 8, saved_rip);
                m_memory.read_value(cur_rbp, saved_rbp);
                std::fprintf(stderr, "  #%d: RIP=0x%llx, RBP=0x%llx\n", f, (unsigned long long)saved_rip, (unsigned long long)saved_rbp);
                cur_rbp = saved_rbp;
            }
            std::fprintf(stderr, "======================================================================\n");
            std::fprintf(stderr, "Aborting execution due to --strict-hle policy.\n");
            std::fflush(stderr);
            std::abort();
        }

        // 3. Normal mode (non-strict): warn and return 0 (null-bail fallback).
        // 0 is SCE_OK for Sony calls and NULL for pointer/iterator APIs: mono_*
        // walks bail cleanly on null, while EINVAL was stored as a handle and
        // chased as a pointer (see the 0x99ad0a poll loop).
        const char* str_arg0 = (sctx.args[0] > 0x1000 && m_memory.is_valid_address(sctx.args[0], 1))
                                ? m_memory.to_host_ptr<char>(sctx.args[0]) : nullptr;
        const char* str_arg1 = (sctx.args[1] > 0x1000 && m_memory.is_valid_address(sctx.args[1], 1))
                                ? m_memory.to_host_ptr<char>(sctx.args[1]) : nullptr;
        GuestAddr ret_addr = 0;
        m_memory.read_value(sctx.rsp, ret_addr);
        // Throttle: first 5 prints per (NID, caller); the rest only bump the
        // suppressed counter reported by dump_hle_warning_stats().
        bool print_warning = false;
        {
            std::lock_guard<std::mutex> lock(s_unimpl_warn_mtx);
            UnimplWarnEntry& e = s_unimpl_warned[unimpl_warn_key(nid, ret_addr)];
            if (e.printed == 0) {
                e.nid = nid;
                e.caller = ret_addr;
            }
            if (e.printed < 5) {
                ++e.printed;
                print_warning = true;
            } else {
                ++s_unimpl_suppressed_total;
            }
        }
        if (print_warning) {
        // Sequence number and calling thread: without them it is impossible to
        // tell whether a warning is a cause or a consequence of the abort path
        // (the teardown sweep signals semas that shutdown already erased).
        static std::atomic<unsigned long long> s_warn_seq{0};
        std::fprintf(stderr, "[HLE WARNING #%llu TID=%d rip=0x%llx] Unimplemented stub called: NID=0x%016llx (caller RIP=0x%llx, rdi=0x%llx '%s', rsi=0x%llx '%s', rdx=0x%llx) returning 0 (0x0)\n",
                     (unsigned long long)(++s_warn_seq),
                     (int)t_guest_tid, (unsigned long long)t_guest_rip,
                     (unsigned long long)nid, (unsigned long long)ret_addr, (unsigned long long)sctx.args[0],
                     str_arg0 ? str_arg0 : "", (unsigned long long)sctx.args[1],
                     str_arg1 ? str_arg1 : "", (unsigned long long)sctx.args[2]);
        GuestAddr cur_rbp = sctx.rbp;
        for (int f = 0; f < 8 && cur_rbp > 0x1000 && m_memory.is_valid_address(cur_rbp, 16); ++f) {
            GuestAddr saved_rip = 0;
            GuestAddr saved_rbp = 0;
            m_memory.read_value(cur_rbp + 8, saved_rip);
            m_memory.read_value(cur_rbp, saved_rbp);
            std::fprintf(stderr, "  frame #%d: RIP=0x%llx, RBP=0x%llx\n", f, (unsigned long long)saved_rip, (unsigned long long)saved_rbp);
            cur_rbp = saved_rbp;
        }
        } // if (print_warning)
        sctx.rax = 0;
        if (sctx.cpu) {
            sctx.cpu->rax = 0;
        }
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
                        if (sym.st_value != 0) {
                            const GuestAddr value = base_load_addr + sym.st_value + rela.r_addend;
                            m_memory.write_value(patch_location, value);
                        } else {
                            char sym_str[128]{0};
                            if (strtab_addr != 0 && sym.st_name != 0) {
                                read_data(strtab_addr + sym.st_name, sym_str, sizeof(sym_str) - 1);
                            }
                            Nid nid = decode_nid_from_string(sym_str);
                            if (nid == 0) {
                                nid = 0x8000000000000000ULL | static_cast<u64>(sym_idx);
                            }

                            if (std::string_view(sym_str).find("f7uOxY9mM1U") != std::string_view::npos ||
                                nid == 0x7fbb8ec58f663355ULL || nid == 0x5533668fc58ebb7fULL) {
                                constexpr GuestAddr STACK_CANARY_ADDR = 0x3FF0000ULL;
                                constexpr u64 CANARY_VALUE = 0x595E9FBD94FDA766ULL;
                                m_memory.write_value(STACK_CANARY_ADDR, CANARY_VALUE);
                                m_memory.write_value(patch_location, STACK_CANARY_ADDR);
                            } else {
                                const GuestAddr tramp = get_or_create_trampoline(nid);
                                m_memory.write_value(patch_location, tramp + rela.r_addend);
                            }
                        }
                    }
                }
            } else if (type == R_X86_64_JUMP_SLOT || type == R_X86_64_GLOB_DAT) {
                if (symtab_addr != 0) {
                    Elf64_Sym sym{};
                    const GuestAddr sym_entry_addr = symtab_addr + (sym_idx * sizeof(Elf64_Sym));
                    if (read_data(sym_entry_addr, &sym, sizeof(sym))) {
                        Nid nid = sym.st_value;
                        char sym_str[128]{0};
                        if (strtab_addr != 0 && sym.st_name != 0) {
                            read_data(strtab_addr + sym.st_name, sym_str, sizeof(sym_str) - 1);
                        }
                        if (nid == 0 && sym_str[0] != '\0') {
                            nid = decode_nid_from_string(sym_str);
                        }
                        if (nid == 0) {
                            nid = 0x8000000000000000ULL | static_cast<u64>(sym_idx);
                        }
                        if (std::string_view(sym_str).find("f7uOxY9mM1U") != std::string_view::npos ||
                            nid == 0x7fbb8ec58f663355ULL || nid == 0x5533668fc58ebb7fULL) {
                            constexpr GuestAddr STACK_CANARY_ADDR = 0x3FF0000ULL;
                            constexpr u64 CANARY_VALUE = 0x595E9FBD94FDA766ULL;
                            m_memory.write_value(STACK_CANARY_ADDR, CANARY_VALUE);
                            m_memory.write_value(patch_location, STACK_CANARY_ADDR);
                        } else {
                            patch_jump_slot(patch_location, nid);
                        }
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
