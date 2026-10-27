#include "ps4/dynamic_linker.hpp"
#include "ps4/thread.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace ps4 {
// HLE dispatch counts, independent of the PS4_LOG_HLE sampling in the
// dispatcher below. Keyed by NID; the name is filled in on first sight.
struct HleCounts { u64 registered = 0; u64 fallback = 0; };
static std::map<std::string, HleCounts> g_hle_hist;
// Per-NID call counts for trampolines that route into a loaded PRX export.
static std::map<Nid, u64> g_prx_call_count;
static u64 g_prx_total = 0;

// "mono-ps4+0x120ed0" style attribution for a raw guest address, or "?" when no
// loaded module contains it. Used by diagnostics that must not assume module
// ranges.
std::string describe_guest_addr(GuestAddr a) {
    if (auto* dl = active_linker()) {
        GuestAddr off = 0;
        if (const LoadedModule* m = dl->module_for_addr(a, &off)) {
            const std::string base_name =
                m->path.size() ? m->path.substr(m->path.find_last_of('/') + 1) : std::string("?");
            char buf[160];
            std::snprintf(buf, sizeof(buf), "%s+0x%llx", base_name.c_str(),
                          (unsigned long long)off);
            return buf;
        }
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "0x%llx", (unsigned long long)a);
    return buf;
}

// PS4_DUMP_EXPORTS=1: record each export as the loader resolves it. Re-parsing
// PT_DYNAMIC by hand ignores the PT_DYNAMIC p_filesz bound and mis-indexes the
// strtab, which produced <multiple> for DT_STRTAB and a 655-vs-896 mismatch.
struct ExportRow { Nid nid; GuestAddr addr; u64 value; const char* type; char tok[12]; u32 handle; GuestAddr base; GuestSize span; };
struct ModStats { std::string path; GuestAddr base; GuestSize span;
                  usize total=0, undef=0, no_nid=0, dup=0, recorded=0, objs=0, funcs=0; };
static std::vector<ModStats> g_mod_stats;
static std::vector<const char*> g_skip_notes;
static std::vector<ExportRow> g_export_rows;
static const bool s_dump_exports = [] {
    const char* v = std::getenv("PS4_DUMP_EXPORTS");
    return v && *v && std::string_view(v) != "0";
}();

void dump_prx_calls() {
    if (g_prx_call_count.empty()) return;
    std::vector<std::pair<u64, Nid>> v;
    u64 tot = 0;
    for (const auto& e : g_prx_call_count) { v.emplace_back(e.second, e.first); tot += e.second; }
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::fprintf(stderr, "[PRX CALL] distinct=%zu total_calls=%llu\n", v.size(),
                 (unsigned long long)tot);
    const usize show = v.size() < 40 ? v.size() : 40;
    for (usize i = 0; i < show; ++i)
        std::fprintf(stderr, "[PRX CALL] %10llu  NID=0x%016llx\n",
                     (unsigned long long)v[i].first, (unsigned long long)v[i].second);
}

void dump_hle_histogram() {
    if (g_hle_hist.empty()) return;
    struct Row { std::string name; u64 reg; u64 fb; u64 total; };
    std::vector<Row> rows;
    u64 tr = 0, tf = 0;
    for (const auto& e : g_hle_hist) {
        rows.push_back({e.first, e.second.registered, e.second.fallback,
                        e.second.registered + e.second.fallback});
        tr += e.second.registered; tf += e.second.fallback;
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.total > b.total; });
    std::fprintf(stderr, "[HLEHIST] names=%zu registered_dispatches=%llu fallback_dispatches=%llu\n",
                 rows.size(), (unsigned long long)tr, (unsigned long long)tf);
    const usize show = rows.size() < 40 ? rows.size() : 40;
    for (usize i = 0; i < show; ++i)
        std::fprintf(stderr, "[HLEHIST] %8llu  reg=%-8llu fb=%-8llu  %s\n",
                     (unsigned long long)rows[i].total,
                     (unsigned long long)rows[i].reg,
                     (unsigned long long)rows[i].fb,
                     rows[i].name.c_str());
}



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
    m_last_error.clear();
    m_unresolved_nids.clear();
    m_trampoline_base = trampoline_base;
    m_next_trampoline_id = 0;
    m_nid_to_trampoline.clear();
    m_id_to_nid.clear();

    // Reserve 64KB space for trampolines
    constexpr GuestSize TRAMPOLINE_POOL_SIZE = 0x10000;
    if (!m_memory.is_valid_address(m_trampoline_base, TRAMPOLINE_POOL_SIZE)) {
        m_last_error = "trampoline pool outside guest memory";
        return false;
    }

    m_memory.zero_memory(m_trampoline_base, TRAMPOLINE_POOL_SIZE);
    m_registry.set_syscall_table(&m_syscalls);
    return true;
}

std::string DynamicLinker::relocation_diagnostics() const {
    std::string result = "trampolines=" + std::to_string(m_nid_to_trampoline.size()) +
                         ", loaded_modules=" + std::to_string(m_modules.size());
    if (!m_last_error.empty()) result += ", error=" + m_last_error;
    if (!m_unresolved_nids.empty()) {
        result += ", unresolved NIDs=";
        const usize limit = std::min<usize>(m_unresolved_nids.size(), 8);
        for (usize i = 0; i < limit; ++i) {
            if (i != 0) result += ",";
            char nid[32];
            std::snprintf(nid, sizeof(nid), "0x%016llx",
                          static_cast<unsigned long long>(m_unresolved_nids[i]));
            result += nid;
        }
        if (m_unresolved_nids.size() > limit) result += ",...";
    }
    return result;
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

        // A loaded module's export wins over our HLE stub.
        //
        // This is safe to do here rather than by patching: the stub carries only
        // a dispatch id, not a target, and the NID is captured in this lambda,
        // so the lookup happens on every call. The executable's GOT slots were
        // bound to these stubs before instruction 0, long before any
        // sceKernelLoadStartModule runs -- but because dispatch is late, a module
        // loaded at that point is picked up immediately with no rebinding.
        if (const GuestAddr exported = find_export(nid); exported != 0) {
            // Log the first few takeovers per NID so it is visible whether the
            // module is actually being entered, and which NIDs it claims.
            static std::unordered_map<Nid, u32> s_takeovers;
            static std::mutex s_takeover_mtx;
            bool report = false;
            {
                std::lock_guard<std::mutex> lk(s_takeover_mtx);
                u32& n = s_takeovers[nid];
                if (n < 3) { ++n; report = true; }
            }
            if (report) {
                std::fprintf(stderr,
                             "[module] NID 0x%016llx now resolves into loaded module "
                             "at 0x%llx (HLE stub bypassed)\n",
                             (unsigned long long)nid, (unsigned long long)exported);
            }
            // Tail-jump: leave the return address we did not push in place, so
            // the module's own `ret` lands directly back in the caller. ctx.rip
            // already points past the `syscall`, so the stub's `ret` never runs.
            // cpu_rip_modified tells the executor not to advance RIP or copy
            // sctx.rax back. Without it the jump was discarded: every call that
            // reached a loaded module through a trampoline (e.g. the 50
            // mono_aot_register_module calls from MonoAssembliesPS4.prx)
            // returned SCE_OK without running a single instruction.
            if (sctx.cpu != nullptr) {
                sctx.cpu->rip = exported;
                sctx.cpu_rip_modified = true;
            }
            // PRX export call tracking: which loaded-module entry points the
            // guest actually reaches, and with what arguments.
            {
                GuestAddr caller = 0;
                m_memory.read_value(sctx.rsp, caller);
                const u64 cnt = ++g_prx_call_count[nid];
                if (cnt <= 50) {
                    std::fprintf(stderr,
                        "[PRX CALL] %s -> 0x%llx (rdi=0x%llx rsi=0x%llx rdx=0x%llx) "
                        "caller=0x%llx count=%llu\n",
                        nid >= 0x100000000ull ? "" : "?", (unsigned long long)exported,
                        (unsigned long long)sctx.args[0], (unsigned long long)sctx.args[1],
                        (unsigned long long)sctx.args[2], (unsigned long long)caller,
                        (unsigned long long)cnt);
                }
                ++g_prx_total;
            }
            return SCE_OK;
        }

        const HleSymbol* sym = m_registry.find_by_nid(nid);
        // Classify EVERY dispatch here, before any logging gate. The previous
        // counters lived inside the (s_log_hle || verbose || is_gnm) branch, so
        // with logging off they counted "this call was not logged" and reported
        // it as "hit the generic fallback", while real fallback hits were never
        // counted at all.
        if (sym && sym->handler) ++g_hle_hist[sym->name].registered;
        else if (sym)            ++g_hle_hist[sym->name].fallback;
        else {
            char key[64];
            std::snprintf(key, sizeof(key), "<unresolved NID=0x%016llx>",
                          (unsigned long long)nid);
            ++g_hle_hist[key].fallback;
        }
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
            const bool is_gnm = (sym->name.rfind("sceGnm", 0) == 0);
            if ((s_log_hle || verbose || is_gnm) && (sym->name != "powf" && sym->name != "memcmp" && sym->name != "memset" &&
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
            if (e.printed < 100) {
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
    m_last_error.clear();
    GuestAddr rela_addr = 0;
    GuestSize rela_size = 0;
    GuestAddr jmprel_addr = 0;
    GuestSize pltrel_size = 0;
    GuestAddr symtab_addr = 0;
    GuestAddr strtab_addr = 0;
    [[maybe_unused]] GuestSize strtab_size = 0;
    usize synthetic_imports = 0;

    // Scan for PT_SCE_PROGLIB / PT_SCE_DYNLIBDATA file offset.
    //
    // Use the loader's SELF-aware accessor, not ph.p_offset. In a SELF-wrapped
    // module p_offset describes the pre-packing layout: on mono-ps4.prx it is
    // 11,280 bytes past where the bytes actually are, so reading with it lands
    // outside the segment entirely.
    usize proglib_file_offset = 0;
    usize proglib_file_size = 0;
    for (const auto& ph : loader.program_headers()) {
        if (ph.p_type == PT_SCE_PROGLIB || (ph.p_type == PT_SCE_DYNLIBDATA && ph.p_filesz > 0x100)) {
            bool found = false;
            const usize off = loader.segment_file_offset(ph, &found);
            if (found && off != 0 && off + static_cast<usize>(ph.p_filesz) <= file_data.size()) {
                proglib_file_offset = off;
                proglib_file_size = static_cast<usize>(ph.p_filesz);
                break;
            }
        }
    }

    // Walk the dynamic table. PT_DYNAMIC is not its own segment in a SELF
    // module -- it is a slice at the tail of PT_SCE_PROGLIB, so its p_offset
    // is ~10 KB past EOF. ElfLoader::dynamic_offset_file_offset() recovers it
    // by subtracting the two stale offsets, whose errors cancel.
    if (const usize dyn_off = loader.dynamic_offset_file_offset();
        dyn_off != 0 && dyn_off < file_data.size()) {
        const Elf64ProgramHeader* dyn_ph = nullptr;
        for (const auto& ph : loader.program_headers()) {
            if (ph.p_type == PT_DYNAMIC) { dyn_ph = &ph; break; }
        }
        const usize dyn_size = dyn_ph ? static_cast<usize>(dyn_ph->p_filesz) : 0;
        const usize num_dyn = dyn_size / sizeof(Elf64_Dyn);
        if (dyn_off + (num_dyn * sizeof(Elf64_Dyn)) <= file_data.size()) {
            const auto* dyn_table = reinterpret_cast<const Elf64_Dyn*>(file_data.data() + dyn_off);
            for (usize i = 0; i < num_dyn; ++i) {
                const auto& dyn = dyn_table[i];
                if (dyn.d_tag == DT_NULL) break;
                if (dyn.d_tag == DT_RELA || dyn.d_tag == DT_SCE_RELA) rela_addr = dyn.d_un.d_ptr;
                if (dyn.d_tag == DT_RELASZ || dyn.d_tag == DT_SCE_RELASZ) rela_size = dyn.d_un.d_val;
                if (dyn.d_tag == DT_JMPREL || dyn.d_tag == DT_SCE_JMPREL) jmprel_addr = dyn.d_un.d_ptr;
                if (dyn.d_tag == DT_PLTRELSZ || dyn.d_tag == DT_SCE_PLTRELSZ) pltrel_size = dyn.d_un.d_val;
                // DT_SCE_SYMTAB (0x61000039) is the symbol table OFFSET; the tag
                // past SYMENT is its byte SIZE. Only the offset belongs here --
                // treating the size as an offset lands mid-table and shifts every
                // st_name. See reloc.hpp for the measurement.
                if (dyn.d_tag == DT_SYMTAB || dyn.d_tag == DT_SCE_SYMTAB) {
                    symtab_addr = dyn.d_un.d_ptr;
                }
                if (dyn.d_tag == DT_STRTAB || dyn.d_tag == DT_SCE_STRTAB) strtab_addr = dyn.d_un.d_ptr;
                if (dyn.d_tag == DT_STRSZ || dyn.d_tag == DT_SCE_STRSZ) strtab_size = dyn.d_un.d_val;
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

        // 2. Try to find the virtual address within any ELF program header in file_data.
        // PT_DYNAMIC belongs here too: in a plain ELF it is the only segment
        // covering the symbol/relocation tables, so dropping it makes every
        // JMPREL lookup miss.
        for (const auto& ph : loader.program_headers()) {
            if (ph.p_filesz > 0 &&
                (ph.p_type == PT_LOAD || ph.p_type == PT_SCE_RELRO || ph.p_type == PT_DYNAMIC)) {
                bool seg_found = false;
                const usize seg_off = loader.segment_file_offset(ph, &seg_found);
                if (!seg_found || seg_off == 0) continue;
                // If ELF is relocated at base_load_addr
                const GuestAddr seg_start = base_load_addr + ph.p_vaddr;
                if (vaddr >= seg_start && (vaddr + size) <= (seg_start + ph.p_filesz)) {
                    const usize offset_in_ph = static_cast<usize>(vaddr - seg_start);
                    const usize file_offset = seg_off + offset_in_ph;
                    if (file_offset + size <= file_data.size()) {
                        std::memcpy(dest, file_data.data() + file_offset, size);
                        return true;
                    }
                }
                // Check raw vaddr without base_load_addr
                if (base_load_addr != 0 && vaddr >= ph.p_vaddr && (vaddr + size) <= (ph.p_vaddr + ph.p_filesz)) {
                    const usize offset_in_ph = static_cast<usize>(vaddr - ph.p_vaddr);
                    const usize file_offset = seg_off + offset_in_ph;
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

    auto process_rela_table = [&](GuestAddr table_addr, GuestSize table_size, const char* what) -> bool {
        if (table_addr == 0 || table_size == 0) return true;
        const usize count = static_cast<usize>(table_size / sizeof(Elf64_Rela));
        usize applied = 0, unread = 0;
        GuestAddr first_failure = 0;
        auto read_failed = [&](GuestAddr address) {
            ++unread;
            if (first_failure == 0) first_failure = address;
        };
        auto patch_failed = [&](GuestAddr address) {
            ++unread;
            if (first_failure == 0) first_failure = address;
        };
        // Undefined symbols whose name did not decode to a real NID get a
        // synthetic one. Counted so a module's import table can be audited.
        for (usize i = 0; i < count; ++i) {
            const GuestAddr entry_addr = table_addr + (i * sizeof(Elf64_Rela));
            Elf64_Rela rela{};
            if (!read_data(entry_addr, &rela, sizeof(rela))) {
                read_failed(entry_addr);
                continue;
            }
            ++applied;

            const u32 type = ELF64_R_TYPE(rela.r_info);
            const u32 sym_idx = ELF64_R_SYM(rela.r_info);
            const GuestAddr patch_location = base_load_addr + rela.r_offset;

            if (type == R_X86_64_RELATIVE) {
                const GuestAddr value = base_load_addr + rela.r_addend;
                if (!m_memory.write_value(patch_location, value)) patch_failed(patch_location);
            } else if (type == R_X86_64_64) {
                if (symtab_addr == 0) {
                    read_failed(symtab_addr);
                    continue;
                }
                Elf64_Sym sym{};
                const GuestAddr sym_entry_addr = symtab_addr + (sym_idx * sizeof(Elf64_Sym));
                if (!read_data(sym_entry_addr, &sym, sizeof(sym))) {
                    read_failed(sym_entry_addr);
                    continue;
                }
                if (sym.st_value != 0) {
                    const GuestAddr value = base_load_addr + sym.st_value + rela.r_addend;
                    if (!m_memory.write_value(patch_location, value)) patch_failed(patch_location);
                    continue;
                }
                char sym_str[128]{0};
                if (strtab_addr != 0 && sym.st_name != 0)
                    read_data(strtab_addr + sym.st_name, sym_str, sizeof(sym_str) - 1);
                Nid nid = decode_nid_from_string(sym_str);
                if (nid == 0) {
                    ++synthetic_imports;
                    nid = 0x8000000000000000ULL | static_cast<u64>(sym_idx);
                }
                if (!m_registry.find_by_nid(nid) && !m_module_exports.contains(nid))
                    m_unresolved_nids.push_back(nid);
                if (std::string_view(sym_str).find("f7uOxY9mM1U") != std::string_view::npos ||
                    nid == 0x7fbb8ec58f663355ULL || nid == 0x5533668fc58ebb7fULL) {
                    constexpr GuestAddr STACK_CANARY_ADDR = 0x3FF0000ULL;
                    constexpr u64 CANARY_VALUE = 0x595E9FBD94FDA766ULL;
                    if (!m_memory.write_value(STACK_CANARY_ADDR, CANARY_VALUE))
                        patch_failed(STACK_CANARY_ADDR);
                    if (!m_memory.write_value(patch_location, STACK_CANARY_ADDR))
                        patch_failed(patch_location);
                } else {
                    const GuestAddr tramp = get_or_create_trampoline(nid);
                    if (!m_memory.write_value(patch_location, tramp + rela.r_addend))
                        patch_failed(patch_location);
                    m_nid_to_slots[nid].push_back(patch_location);
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
                            ++synthetic_imports;
                            nid = 0x8000000000000000ULL | static_cast<u64>(sym_idx);
                        }
                        if (!m_registry.find_by_nid(nid) && !m_module_exports.contains(nid)) {
                            m_unresolved_nids.push_back(nid);
                        }
                        if (std::string_view(sym_str).find("f7uOxY9mM1U") != std::string_view::npos ||
                            nid == 0x7fbb8ec58f663355ULL || nid == 0x5533668fc58ebb7fULL) {
                            constexpr GuestAddr STACK_CANARY_ADDR = 0x3FF0000ULL;
                            constexpr u64 CANARY_VALUE = 0x595E9FBD94FDA766ULL;
                            if (!m_memory.write_value(STACK_CANARY_ADDR, CANARY_VALUE))
                                patch_failed(STACK_CANARY_ADDR);
                            if (!m_memory.write_value(patch_location, STACK_CANARY_ADDR))
                                patch_failed(patch_location);
                        } else {
                            if (!patch_jump_slot(patch_location, nid))
                                patch_failed(patch_location);
                            m_nid_to_slots[nid].push_back(patch_location);
                        }
                    } else {
                        read_failed(sym_entry_addr);
                    }
                } else {
                    read_failed(symtab_addr);
                }
            }
        }
        if (count != 0) {
            std::fprintf(stderr, "[module] %s: %zu relocs, %zu read, %zu unread\n",
                         what, count, applied, unread);
        }
        if (unread != 0) {
            char detail[256];
            std::snprintf(detail, sizeof(detail),
                          "%s relocation table read/patch failure at 0x%llx "
                          "(table=0x%llx, unread=%zu)",
                          what, static_cast<unsigned long long>(first_failure),
                          static_cast<unsigned long long>(table_addr), unread);
            m_last_error = detail;
            return false;
        }
        return true;
    };

    if (!process_rela_table(rela_addr, rela_size, "RELA")) return false;
    if (!process_rela_table(jmprel_addr, pltrel_size, "JMPREL")) return false;
    if (synthetic_imports != 0) {
        std::fprintf(stderr,
                     "[module] relocation: %zu imported symbols had no decodable NID and "
                     "fell back to a synthetic one\n", synthetic_imports);
    }
    return true;
}

// ---------------------------------------------------------------------------
// PS4 export names
//
// A shipped module does not store C identifiers. Each exported symbol's name is
// a 16-byte record: 11 base64 characters, "#XX#YY", NUL. The 11 characters are
// base64url of the 64-bit NID in BIG-endian order (verified: encoding our own
// NIDs big-endian matches 137 of 370 in the registry against 1 little-endian).
//
// st_name does not point at byte 0 of that token -- it points 8 bytes in, at the
// "#XX#YY" tail. Scanning backwards for an 11-character base64 run resolves every
// entry of mono-ps4.prx's 6024-symbol table, so do that rather than assume.
// ---------------------------------------------------------------------------

namespace {

/// SHN_UNDEF. Spelled locally because this file does not pull in <elf.h>.
constexpr u16 SHN_UNDEF_LOCAL = 0;
constexpr u8 ELF64_ST_TYPE_FUNC = 2;

// Sony's NID token alphabet: '+' is 62 and '-' is 63, the same alphabet
// decode_nid_from_string() uses for imports. The URL-safe alphabet used here
// before ('-' = 62, '_' = 63) gave every export whose token contains '-' a
// wrong NID, so imports of it (e.g. mono_parse_default_optimizations) never
// found the module's export and fell through to HLE stubs.
constexpr std::string_view kBase64Url =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

/// Decode exactly 11 base64url characters into a 64-bit big-endian NID.
///
/// The alphabet in the PRX is STANDARD base64, not base64url: mono-ps4.prx
/// really contains tokens like "O4ZcR+K4CUE" and "+6h47p15FxM". Rejecting '+'
/// and '/' silently dropped 99 of mono-ps4's 556 exports and would equally
/// drop any import whose NID uses those two characters, which is a plausible
/// source of unresolved imports. '+' is the standard spelling of base64url's
/// '-' (both are 62) and '/' of '_' (both 63), so accept both spellings.
bool decode_nid_token(const char* tok11, Nid* out) {
    if (tok11 == nullptr) return false;
    u64 val = 0;
    u8 bits = 0;
    u64 acc = 0;
    for (int i = 0; i < 11; ++i) {
        std::string_view::size_type pos = kBase64Url.find(tok11[i]);
        if (pos == std::string_view::npos) return false;
        acc = (acc << 6) | static_cast<u64>(pos);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            val = (val << 8) | ((acc >> bits) & 0xFF);
        }
    }
    *out = val;
    return true;
}

} // namespace

void DynamicLinker::build_export_table(std::span<const u8> file_data, const ElfLoader& loader,
                                       GuestAddr base, u32 handle, GuestSize span) {
    const usize proglib_off = loader.proglib_file_offset();
    const usize proglib_size = loader.proglib_file_size();
    if (proglib_off == 0 || proglib_size == 0 ||
        proglib_off + proglib_size > file_data.size()) {
        std::fprintf(stderr, "[module] 0x%x: no usable PT_SCE_PROGLIB (off=0x%zx size=0x%zx)\n",
                     handle, proglib_off, proglib_size);
        return;
    }
    const u8* proglib = file_data.data() + proglib_off;

    // Which dynamic tag carries the symbol table is NOT consistent across
    // modules. In mono-ps4.prx it is 0x6100003F; in MonoAssembliesPS4.prx the
    // same 0x6100003F holds 0x2a0, which is not a symbol table, and 0x61000039
    // holds the real one at 0x268. So gather every candidate and pick the one
    // that actually parses as Elf64_Sym rather than trusting a tag.
    usize symtab_off = 0, symtab_sz = 0, strtab_off = 0;
    GuestAddr init = 0, fini = 0;
    const usize dyn_off = loader.dynamic_offset_file_offset();
    if (dyn_off == 0 || dyn_off + sizeof(Elf64_Dyn) > file_data.size()) {
        std::fprintf(stderr, "[module] 0x%x: no usable PT_DYNAMIC\n", handle);
        return;
    }
    usize dyn_size = 0;
    for (const auto& ph : loader.program_headers()) {
        if (ph.p_type == PT_DYNAMIC) { dyn_size = static_cast<usize>(ph.p_filesz); break; }
    }
    const auto* dyn = reinterpret_cast<const Elf64_Dyn*>(file_data.data() + dyn_off);
    for (usize i = 0; i * sizeof(Elf64_Dyn) < dyn_size; ++i) {
        const auto& d = dyn[i];
        if (d.d_tag == DT_NULL) break;
        // Offset and size are separate tags. Reading the size as an offset lands
        // partway into the table and shifts every st_name, which is what made a
        // backwards scan for the name token look necessary.
        if (d.d_tag == DT_SYMTAB || d.d_tag == DT_SCE_SYMTAB) {
            symtab_off = static_cast<usize>(d.d_un.d_ptr);
        } else if (d.d_tag == DT_SCE_SYMTABSZ) {
            symtab_sz = static_cast<usize>(d.d_un.d_val);
        } else if (d.d_tag == DT_STRTAB || d.d_tag == DT_SCE_STRTAB) {
            strtab_off = static_cast<usize>(d.d_un.d_ptr);
        } else if (d.d_tag == DT_INIT) {
            init = d.d_un.d_ptr;
        } else if (d.d_tag == DT_FINI) {
            fini = d.d_un.d_ptr;
        }
    }

    if (symtab_off == 0 || symtab_sz < sizeof(Elf64_Sym) ||
        symtab_off + symtab_sz > proglib_size) {
        std::fprintf(stderr, "[module] 0x%x: bad symbol table (off=0x%zx size=0x%zx "
                            "proglib=0x%zx)\n",
                     handle, symtab_off, symtab_sz, proglib_size);
        auto it0 = m_modules.find(handle);
        if (it0 != m_modules.end()) { it0->second.init = init; it0->second.fini = fini; }
        return;
    }
    const usize total = symtab_sz / sizeof(Elf64_Sym);

    // st_name indexes the string table, so the token sits at
    // PROGLIB + strtab_off + st_name -- at byte 0, not partway in.
    // Split the four reasons token_at() returns 0, so a benign "st_name==0"
    // is not confused with a strtab indexing bug that would also corrupt import
    // resolution.
    u64 c_no_strtab = 0, c_name0 = 0, c_past_end = 0, c_decode_fail = 0;
    auto token_at = [&](u32 st_name) -> Nid {
        if (strtab_off == 0) { ++c_no_strtab; return 0; }
        if (st_name == 0) { ++c_name0; return 0; }
        const usize at = strtab_off + st_name;
        if (at + 11 > proglib_size) { ++c_past_end; return 0; }
        Nid nid = 0;
        if (!decode_nid_token(reinterpret_cast<const char*>(proglib + at), &nid)) {
            ++c_decode_fail; return 0;
        }
        return nid;
    };

    ModStats* g_ms = nullptr;
    if (s_dump_exports) {
        const std::string nm = "handle-" + std::to_string(handle);
        g_mod_stats.push_back({nm, base, 0});
        g_ms = &g_mod_stats.back();
        g_ms->span = m_modules.count(handle) ? m_modules.at(handle).size : 0;
    }
    usize defined = 0, objs = 0, funcs = 0;
    for (usize i = 0; i < total; ++i) {
        Elf64_Sym sym{};
        std::memcpy(&sym, proglib + symtab_off + (i * sizeof(Elf64_Sym)), sizeof(sym));
        if (g_ms) ++g_ms->total;
        // SHN_UNDEF means "imported from somewhere else" -- not ours to export.
        if (sym.st_shndx == SHN_UNDEF_LOCAL) {
            if (g_ms) ++g_ms->undef;
            continue;
        }
        const Nid nid = token_at(sym.st_name);
        if (nid == 0) { if (g_ms) ++g_ms->no_nid; continue; }
        // Take every defined symbol, not just STT_FUNC. AOT module descriptor
        // tables are STT_OBJECT data; filtering on FUNC alone is what made
        // MonoAssembliesPS4.prx look like it exported nothing.
        const u8 stype = sym.st_info & 0xF;
        if (stype == ELF64_ST_TYPE_FUNC) ++funcs;
        else ++objs;
        // First definition wins, so an earlier module is not shadowed.
        const GuestAddr real = base + sym.st_value;
        if (!m_module_exports.emplace(nid, real).second && g_ms) ++g_ms->dup;
        ++defined;

        // Replace any trampoline already written for this NID.
        //
        // A stub address is only correct for a symbol the guest CALLS. For a data
        // import the guest reads the slot and dereferences it, so a trampoline
        // makes it consume `B8 xx xx xx xx 0F 05 C3` as a struct pointer. eboot
        // imports 19 STT_OBJECT AOT descriptor tables from
        // MonoAssembliesPS4.prx through R_X86_64_64, so all 19 are affected.
        // Rewriting to the module address is correct for both kinds: a called
        // function now jumps straight into the module instead of trapping.
        if (s_dump_exports) {
            ExportRow r{};
            r.nid = nid; r.addr = real; r.value = sym.st_value;
            r.type = (stype == ELF64_ST_TYPE_FUNC) ? "FUNC" : "OBJECT";
            r.handle = handle; r.base = base;
            r.span = m_modules.count(handle) ? m_modules.at(handle).size : 0;
            if (g_ms) { ++g_ms->recorded; if (std::strcmp(r.type, "OBJECT") == 0) ++g_ms->objs; else ++g_ms->funcs; }
            std::snprintf(r.tok, sizeof(r.tok), "%.11s",
                          reinterpret_cast<const char*>(proglib + strtab_off + sym.st_name));
            g_export_rows.push_back(r);
        }
        const auto slots = m_nid_to_slots.find(nid);
        if (slots != m_nid_to_slots.end()) {
            usize patched = 0;
            for (const GuestAddr slot : slots->second) {
                if (m_memory.write_value(slot, real)) ++patched;
            }
            std::fprintf(stderr,
                         "[module] 0x%x export 0x%016llx (%s) at 0x%llx rewrote %zu "
                         "import slot(s) previously holding a trampoline\n",
                         handle, (unsigned long long)nid,
                         stype == ELF64_ST_TYPE_FUNC ? "func" : "data",
                         (unsigned long long)real, patched);
        }
    }

    auto it = m_modules.find(handle);
    if (it != m_modules.end()) {
        it->second.export_count = defined;
        it->second.init = init;
        it->second.fini = fini;
    }
    std::fprintf(stderr,
                 "[module] 0x%x %s: base=0x%llx span=0x%llx (ends 0x%llx) | %zu syms, "
                 "%zu exported, DT_INIT=0x%llx DT_FINI=0x%llx\n",
                 handle, loader.is_self_container() ? "SELF/PRX" : "ELF",
                 (unsigned long long)base, (unsigned long long)span,
                 (unsigned long long)(base + span),
                 total, defined, (unsigned long long)init, (unsigned long long)fini);
    if (s_dump_exports) {
        std::fprintf(stderr,
            "[tokfail] handle=0x%x strtab_off=0x%zx proglib_size=0x%zx | "
            "no_strtab=%llu st_name==0=%llu past_proglib_end=%llu decode_fail=%llu total=%llu\n",
            handle, strtab_off, proglib_size,
            (unsigned long long)c_no_strtab, (unsigned long long)c_name0,
            (unsigned long long)c_past_end, (unsigned long long)c_decode_fail,
            (unsigned long long)(c_no_strtab + c_name0 + c_past_end + c_decode_fail));
    }
}

bool DynamicLinker::map_segments(std::span<const u8> file_data, const ElfLoader& loader,
                                 GuestAddr base, GuestSize* span_out, u32 handle) {
    GuestSize span = 0;
    for (const auto& ph : loader.program_headers()) {
        if (ph.p_type != PT_LOAD && ph.p_type != PT_SCE_RELRO) continue;
        bool found = false;
        const usize seg_off = loader.segment_file_offset(ph, &found);
        if (!found) {
            std::fprintf(stderr, "[module] no SELF entry for segment type 0x%x\n", ph.p_type);
            return false;
        }
        const GuestAddr dst = base + ph.p_vaddr;
        char label[96];
        std::snprintf(label, sizeof(label), "module@0x%x:seg@0x%llx", handle,
                      (unsigned long long)ph.p_vaddr);
        if (!m_memory.reserve_mapping(dst, ph.p_memsz, label)) {
            std::fprintf(stderr, "[module] 0x%x: refusing to map segment at 0x%llx\n",
                         handle, (unsigned long long)dst);
            return false;
        }
        if (ph.p_filesz > 0) {
            if (seg_off + static_cast<usize>(ph.p_filesz) > file_data.size()) {
                std::fprintf(stderr, "[module] segment 0x%llx+0x%llx runs past end of file\n",
                             (unsigned long long)dst, (unsigned long long)ph.p_filesz);
                return false;
            }
            if (!m_memory.write_bytes(dst, file_data.data() + seg_off,
                                      static_cast<usize>(ph.p_filesz))) {
                std::fprintf(stderr, "[module] write of segment at 0x%llx failed\n",
                             (unsigned long long)dst);
                return false;
            }
        }
        // memsz > filesz is .bss: it must be zeroed or the module reads stale
        // memory from whatever the allocator handed us.
        if (ph.p_memsz > ph.p_filesz) {
            const u8 zeros[256]{};
            GuestAddr p = dst + ph.p_filesz;
            GuestAddr end = dst + ph.p_memsz;
            while (p < end) {
                const usize n = static_cast<usize>(std::min<GuestAddr>(sizeof(zeros), end - p));
                if (!m_memory.write_bytes(p, zeros, n)) return false;
                p += n;
            }
        }
        span = std::max<GuestSize>(span, ph.p_vaddr + ph.p_memsz);
    }
    if (span_out) *span_out = span;
    return span != 0;
}

u32 DynamicLinker::load_module(const std::string& path, GuestAddr base, bool run_init) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "[module] cannot open %s\n", path.c_str());
        return 0;
    }
    std::vector<u8> data((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    if (data.empty()) {
        std::fprintf(stderr, "[module] %s is empty\n", path.c_str());
        return 0;
    }

    ElfLoader loader;
    if (!loader.load_from_memory(data)) {
        std::fprintf(stderr, "[module] %s: %s\n", path.c_str(),
                     std::string(loader.error_message()).c_str());
        return 0;
    }

    // Compute the span from the program headers before choosing a base, so the
    // reservation is the real footprint.
    GuestSize needed = 0;
    for (const auto& ph : loader.program_headers()) {
        if (ph.p_type != PT_LOAD && ph.p_type != PT_SCE_RELRO) continue;
        needed = std::max<GuestSize>(needed, ph.p_vaddr + ph.p_memsz);
    }
    if (needed == 0) {
        std::fprintf(stderr, "[module] %s: no loadable segments\n", path.c_str());
        return 0;
    }

    if (base == 0) {
        if (loader.needs_load_base()) {
            base = module_base_allocator(needed);
            if (base == 0) return 0;
        }
        // else: fixed-address executable, maps where its headers say
    } else if (!loader.needs_load_base()) {
        std::fprintf(stderr, "[module] %s has fixed addresses, ignoring base 0x%llx\n",
                     path.c_str(), (unsigned long long)base);
        base = 0;
    }

    // Refuse to map on top of an existing module. This is a loader bug if it
    // ever fires, and silently overlapping produces a GOT that reads as
    // arbitrary bytes rather than an address.
    if (base != 0) {
        for (const auto& [h, m] : m_modules) {
            if (m.base == 0) continue;
            if (base < m.base + m.size && m.base < base + needed) {
                std::fprintf(stderr,
                             "[module] REFUSING to map %s at 0x%llx: overlaps module 0x%x "
                             "[0x%llx,0x%llx)\n",
                             path.c_str(), (unsigned long long)base, h,
                             (unsigned long long)m.base,
                             (unsigned long long)(m.base + m.size));
                return 0;
            }
        }
    }

    const u32 handle = ++m_next_module_handle;
    LoadedModule mod{};
    mod.handle = handle;
    mod.path = path;
    mod.base = base;
    m_modules.emplace(handle, mod);

    const auto rollback_mapping = [&]() {
        for (const auto& ph : loader.program_headers()) {
            if (ph.p_type == PT_LOAD || ph.p_type == PT_SCE_RELRO) {
                m_memory.release_mapping(base + ph.p_vaddr);
            }
        }
        m_modules.erase(handle);
        for (auto it = m_nid_to_slots.begin(); it != m_nid_to_slots.end();) {
            auto& slots = it->second;
            slots.erase(std::remove_if(slots.begin(), slots.end(),
                                       [&](GuestAddr slot) {
                                           return slot >= base && slot < base + needed;
                                       }),
                        slots.end());
            if (slots.empty()) it = m_nid_to_slots.erase(it);
            else ++it;
        }
    };

    GuestSize span = 0;
    if (!map_segments(data, loader, base, &span, handle)) {
        rollback_mapping();
        return 0;
    }
    m_modules.find(handle)->second.size = span;

    // Relocate before reading exports: the symbol values we record are the
    // pre-relocation ones, but the import bindings written here are what make
    // the module's own calls land somewhere sane.
    if (!resolve_relocations(data, loader, base)) {
        std::fprintf(stderr, "[module] 0x%x %s: relocation failed: %s\n",
                     handle, path.c_str(), relocation_diagnostics().c_str());
        rollback_mapping();
        return 0;
    }
    build_export_table(data, loader, base, handle, span);

    if (run_init) run_module_init(handle);
    return handle;
}

bool DynamicLinker::run_module_init(u32 handle) {
    auto it = m_modules.find(handle);
    if (it == m_modules.end()) return false;
    if (it->second.init == 0) {
        std::fprintf(stderr, "[module] 0x%x has no DT_INIT; not calling an entry point\n", handle);
        return false;
    }
    const GuestAddr entry = it->second.base + it->second.init;
    std::fprintf(stderr, "[module] 0x%x %s: DT_INIT -> 0x%llx\n",
                 handle, it->second.path.c_str(), (unsigned long long)entry);
    return true;
}

GuestAddr DynamicLinker::find_export(Nid nid) const noexcept {
    const auto it = m_module_exports.find(nid);
    return it == m_module_exports.end() ? 0 : it->second;
}

const LoadedModule* DynamicLinker::module(u32 handle) const noexcept {
    const auto it = m_modules.find(handle);
    return it == m_modules.end() ? nullptr : &it->second;
}

const LoadedModule* DynamicLinker::module_for_addr(GuestAddr addr, GuestAddr* offset) const noexcept {
    for (const auto& [handle, mod] : m_modules) {
        if (mod.base == 0) continue;
        const GuestAddr size = mod.size ? mod.size : 0x1000;
        if (addr >= mod.base && addr < mod.base + size) {
            if (offset) *offset = addr - mod.base;
            return &mod;
        }
    }
    if (offset) *offset = addr;
    return nullptr;
}

GuestAddr DynamicLinker::first_module_base() const noexcept {
    GuestAddr lo = 0;
    for (const auto& [handle, mod] : m_modules) {
        if (mod.base != 0 && (lo == 0 || mod.base < lo)) lo = mod.base;
    }
    return lo;
}

namespace {
DynamicLinker* g_active_linker = nullptr;

// Modules get the otherwise unused gap between the eboot image (ends ~0x1652000)
// and the anonymous-mmap arena at 0x10000000 (CPU heap from 0x20000000).
// They used to start at 0x160000000, which is both the base of the HLE mspace
// arena and inside the 1:1 direct-memory window (0x40000000..0x1C0000000), so
// with PS4_LOAD_PRX the first mspace or direct-memory block landing there
// overwrote mono-ps4.prx's code and GOT.
GuestAddr g_next_module_base = 0x04000000ULL;
constexpr GuestAddr kModuleBaseCeiling = 0x10000000ULL;
constexpr GuestAddr kModuleBaseAlign = 0x200000ULL;
} // namespace

DynamicLinker* active_linker() noexcept { return g_active_linker; }
void set_active_linker(DynamicLinker* linker) noexcept { g_active_linker = linker; }

GuestAddr module_base_allocator(GuestSize need) noexcept {
    // Reserve the module's ACTUAL span, then advance past it with a guard gap.
    //
    // A fixed 2 MB bump overlapped mono-ps4.prx, whose span is 0x24ab68. The next
    // module landed at 0x160200000 -- inside the first module's RELRO -- and its
    // LOAD[0] copy (0x114d8b0 bytes) overwrote the GOT, including a jump slot
    // that had just been correctly patched to a trampoline. A later PLT
    // `jmp qword ptr [rip+disp]` through that slot then landed on raw file bytes.
    //
    // next_base = align_up(base + span, 0x4000) + kModuleGuardGap
    constexpr GuestAddr kModuleAlign = 0x4000;
    constexpr GuestAddr kModuleGuardGap = 0x40000; // 256 KB of unmapped slack

    const GuestAddr base = (g_next_module_base + kModuleAlign - 1) & ~(kModuleAlign - 1);
    const GuestAddr end = base + need;
    if (end >= kModuleBaseCeiling) {
        std::fprintf(stderr, "[module] module base space exhausted (need 0x%llx at 0x%llx)\n",
                     (unsigned long long)need, (unsigned long long)base);
        return 0;
    }
    g_next_module_base = ((end + kModuleAlign - 1) & ~(kModuleAlign - 1)) + kModuleGuardGap;
    return base;
}


void dump_export_tables() {
    if (!g_mod_stats.empty()) {
        std::fprintf(stderr, "[exports] per module (loader counters):\n");
        for (const auto& m : g_mod_stats) {
            std::fprintf(stderr,
                "[exports]   %-24s base=0x%llx span=0x%llx | in_table=%zu defined=%zu "
                "undef_skipped=%zu no_nid_skipped=%zu dup_not_recorded=%zu | recorded=%zu "
                "(FUNC=%zu OBJECT=%zu) end=%s\n",
                m.path.c_str(), (unsigned long long)m.base, (unsigned long long)m.span,
                m.total, m.total - m.undef - m.no_nid, m.undef, m.no_nid, m.dup,
                m.recorded, m.funcs, m.objs,
                m.recorded == m.total - m.undef - m.no_nid ? "complete" : "DUPES DROPPED");
        }
    }
    if (g_export_rows.empty()) return;
    std::vector<ExportRow> v = g_export_rows;
    std::sort(v.begin(), v.end(), [](const ExportRow& a, const ExportRow& b) { return a.addr < b.addr; });
    u64 nfun = 0, nobj = 0;
    for (const auto& r : v) { if (std::strcmp(r.type, "FUNC") == 0) ++nfun; else ++nobj; }
    std::fprintf(stderr, "[exports] total=%zu (FUNC=%llu OBJECT=%llu)\n",
                 v.size(), (unsigned long long)nfun, (unsigned long long)nobj);
    std::fprintf(stderr, "[exports] lowest 10 by address:\n");
    for (usize i = 0; i < v.size() && i < 10; ++i)
        std::fprintf(stderr, "[exports]   #%-2zu 0x%016llx  module+0x%06llx %-6s tok=%s\n",
                     i, (unsigned long long)v[i].addr, (unsigned long long)v[i].value,
                     v[i].type, v[i].tok);
    std::fprintf(stderr, "[exports] highest 3 by address:\n");
    for (usize i = v.size() > 3 ? v.size() - 3 : 0; i < v.size(); ++i)
        std::fprintf(stderr, "[exports]   #%-2zu 0x%016llx  module+0x%06llx %-6s tok=%s\n",
                     i, (unsigned long long)v[i].addr, (unsigned long long)v[i].value,
                     v[i].type, v[i].tok);
    struct Q { u64 nid; const char* nm; };
    const Q qs[] = {{0x05a38a72783c83cfULL,"module_start"},
                    {0x2a90ccacf1efb774ULL,"module_stop"},
                    {0x44a906d9bd4bb5ddULL,"module_bootstrap"},
                    {0x8008ef0941de01b1ULL,"_init"}};
    for (const auto& q : qs) {
        bool hit = false; u64 a = 0; const char* t = "";
        for (const auto& r : v) if (r.nid == q.nid) { hit = true; a = r.addr; t = r.tok; break; }
        if (hit) std::fprintf(stderr, "[exports] NID %#018llx %-16s HIT addr=0x%llx tok=%s\n",
                              (unsigned long long)q.nid, q.nm, (unsigned long long)a, t);
        else      std::fprintf(stderr, "[exports] NID %#018llx %-16s MISS\n",
                              (unsigned long long)q.nid, q.nm);
    }
}

} // namespace ps4
