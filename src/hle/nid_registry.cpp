#include "ps4/audio_manager.hpp"
#include "ps4/executor.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/thread.hpp"
#include "ps4/vfs.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <sys/time.h>
#include <unistd.h>

namespace ps4 {

namespace {

// Compact 8GB PS4 Memory Map for 16GB MacBook Air M4:
// 0x000000000 - 0x004000000 (0 - 64 MB)    : ELF segments + trampolines
// 0x004000000 - 0x020000000 (64 - 512 MB)  : Main stack (0x1ffffff00)
// 0x020000000 - 0x040000000 (512 MB - 1 GB): Guest libc heap (malloc/memalign)
// 0x040000000 - 0x1C0000000 (1 GB - 7 GB)  : Direct Memory (ALLOC_GFX / ONION / GARLIC)
// 0x1C0000000 - 0x200000000 (7 GB - 8 GB)  : Worker thread stacks (downward from 0x1d0000000)
std::atomic<GuestAddr> g_guest_heap_current{0x20000000ULL};   // 512 MB mark
std::atomic<GuestAddr> g_direct_phys_current{0x40000000ULL};  // 1 GB mark

// The libc heap is a bare bump pointer with no block headers, which leaves
// realloc with no way to know how much of the old block to copy. Record each
// block's size here rather than changing malloc's block layout.
static std::mutex s_heap_sizes_mtx;
static std::unordered_map<GuestAddr, GuestSize> s_heap_sizes;

// Hard ceiling for the CPU heap. The memory map reserves 0x20000000-0x40000000
// for the guest libc heap and 0x40000000-0x1C0000000 for direct (Garlic/ONION)
// memory; an unbounded bump allocator walks straight through that boundary and
// silently overwrites GPU buffers with heap contents. See mspace_alloc below.
static constexpr GuestAddr kCpuHeapEnd = 0x40000000ULL;

inline GuestAddr bump_alloc(GuestSize size, GuestSize align = 16) {
    if (size == 0) size = 1;
    const GuestAddr cur = g_guest_heap_current.load();
    const GuestAddr p = (cur + align - 1) & ~(align - 1);
    if (p + size > kCpuHeapEnd) {
        std::fprintf(stderr, "[heap] CPU heap exhausted: need %llu at 0x%llx, ceiling 0x%llx\n",
                     (unsigned long long)size, (unsigned long long)p,
                     (unsigned long long)kCpuHeapEnd);
        return 0;
    }
    g_guest_heap_current.store(p + size);
    std::lock_guard<std::mutex> lock(s_heap_sizes_mtx);
    s_heap_sizes[p] = size;
    return p;
}

inline GuestSize bump_size_of(GuestAddr ptr) {
    std::lock_guard<std::mutex> lock(s_heap_sizes_mtx);
    auto it = s_heap_sizes.find(ptr);
    return it == s_heap_sizes.end() ? 0 : it->second;
}

// Unity's PS4 resource allocator (sceLibcMspaceMemalign) asks for hundreds of
// megabytes of Garlic/ONION memory and treats the result as a GPU buffer: it
// copies shader binaries there and parses their headers. Served from the shared
// CPU bump pointer it walked from 0x20100000 to 0x5A7A8C20 -- straight through
// 0x40000000 -- so ~1.6 GB of heap landed on top of the direct-memory window,
// first block inside it at 0x40770000. Unity then parsed a header full of heap
// pointers ("shaderBinarySize %u is not equal to Program size %ld", sizes that
// were really 0x5E020D58 and a sum of 16-byte entry counts read from pointers),
// fell into a table walk with a 1.5-billion trip count and never presented a
// frame. Give these allocations their own arena above the direct-memory region
// so CPU heap and GPU memory can never alias.
// Direct memory runs 0x40000000..0x15B800000 in a 60 s run (one 4.5 GB
// allocation plus a 128 KB one), so the arena starts just above it and has the
// 0x160000000..0x200000000 remainder of the 8 GB space to itself.
static std::atomic<GuestAddr> g_mspace_current{0x160000000ULL};

inline GuestAddr mspace_alloc(GuestSize size, GuestSize align) {
    if (size == 0) size = 1;
    if (align == 0) align = 16;
    const GuestAddr p = (g_mspace_current.load() + align - 1) & ~(align - 1);
    g_mspace_current.store(p + size);
    return p;
}

// Global ThreadManager and execution dependencies
std::unique_ptr<ThreadManager> g_thread_manager;
MemoryManager* g_thread_manager_mem = nullptr;
SyscallTable* g_syscalls = nullptr;
JitCompiler* g_jit = nullptr;

inline ThreadManager& get_thread_manager(MemoryManager& mem) {
    if (!g_thread_manager || g_thread_manager_mem != &mem) {
        g_thread_manager = std::make_unique<ThreadManager>(mem, g_syscalls);
        g_thread_manager_mem = &mem;
        if (g_jit) g_thread_manager->set_jit_compiler(g_jit);
    } else {
        if (g_syscalls && !g_thread_manager->syscall_table()) {
            g_thread_manager->set_syscall_table(g_syscalls);
        }
        if (g_jit) {
            g_thread_manager->set_jit_compiler(g_jit);
        }
    }
    return *g_thread_manager;
}

// Global AudioOutManager instance
std::unique_ptr<AudioOutManager> g_audio_manager;

inline AudioOutManager& get_audio_manager() {
    if (!g_audio_manager) {
        g_audio_manager = std::make_unique<AudioOutManager>();
    }
    return *g_audio_manager;
}

// Global VirtualFileSystem instance
std::unique_ptr<VirtualFileSystem> g_vfs;
VirtualFileSystem* g_external_vfs = nullptr;

inline VirtualFileSystem& get_vfs() {
    if (g_external_vfs) {
        return *g_external_vfs;
    }
    if (!g_vfs) {
        g_vfs = std::make_unique<VirtualFileSystem>();
    }
    return *g_vfs;
}

// Fios2 file handle tracking
struct FiosFile {
    std::unique_ptr<std::ifstream> stream;
    u64 size{0};
    u64 offset{0};
    std::string path;
};
static std::unordered_map<i32, FiosFile> s_fios_files;
static i32 s_next_fios_fd = 200;

// Real Orbis semaphore table for Unity worker threads (file scope so the
// watchdog dump and synthetic wake can inspect it).
struct GuestSema {
    std::mutex mtx;
    std::condition_variable cv;
    i32 count{0};
    i32 max_count{0x7fffffff};
    bool deleted{false};
    u64 cancel_generation{0}; // bumped by CancelSema to wake active waiters with ECANCELED
    i32 waiters{0};           // threads currently parked in wait (for CancelSema)
};
static std::mutex s_sema_table_mtx;
static std::unordered_map<u32, std::shared_ptr<GuestSema>> s_semas;
static u32 s_next_sema_id = 1;

// Event flags: same shape as the semaphore table. WaitEventFlag blocks on the
// condition variable until the requested bits appear (or the timeout expires),
// and SetEventFlag is the only thing that can post bits, so the two must be
// registered together or a wait can never be satisfied.
struct GuestEventFlag {
    std::mutex mtx;
    std::condition_variable cv;
    u32 bits{0};
    u32 waiters{0};
    bool deleted{false};
};
static std::mutex s_eflag_table_mtx;
static std::unordered_map<u32, std::shared_ptr<GuestEventFlag>> s_eflags;
static u32 s_next_eflag_id = 1;

// Mono dummy-cookie arena: distinct zeroed 256-byte guest slots for Mono
// object lookups. The guest dereferences fields inside these pointers, so they
// must be real mapped memory — carved once from the libc heap bump pointer
// (never the old 0x80000100 dummy range inside DirectMemory). Iterators keep
// returning NULL. Slots are never freed (Mono objects live forever here);
// on exhaustion the handlers return NULL.
struct MonoArena {
    std::mutex mtx;
    GuestAddr base{0};
    GuestAddr next{0};
    GuestAddr end{0};
    GuestAddr domain{0}; // singleton MonoDomain*
    std::unordered_map<std::string, GuestAddr> assemblies; // by name
    std::unordered_map<GuestAddr, GuestAddr> images;       // by assembly ptr
    GuestAddr first_image{0}; // default when image==0 (see class_from_name)
    GuestAddr corlib_image{0}; // singleton MonoImage* for mscorlib (mono_get_corlib)
    std::unordered_map<std::string, GuestAddr> cookies; // stable handle -> slot
    std::unordered_map<GuestAddr, GuestAddr> derived; // unknown-derived objects, by input ptr (retained)
    struct ClassKey {
        GuestAddr image{0};
        std::string ns;
        std::string name;
        bool operator==(const ClassKey& o) const {
            return image == o.image && ns == o.ns && name == o.name;
        }
    };
    struct ClassKeyHash {
        usize operator()(const ClassKey& k) const noexcept {
            usize h = std::hash<GuestAddr>{}(k.image);
            h ^= std::hash<std::string>{}(k.ns) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>{}(k.name) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_map<ClassKey, GuestAddr, ClassKeyHash> classes;
};
static MonoArena s_mono_arena;
static constexpr GuestSize kMonoArenaSize = 1024 * 1024; // 1 MB = 4096 slots
static constexpr GuestSize kMonoSlotSize = 256;

// Caller holds s_mono_arena.mtx. Carves the arena once from the heap bump
// pointer (virtual reservation only) and hands out slots zeroed on first use,
// so untouched pages stay unbacked. Returns 0 when exhausted.
static GuestAddr mono_arena_alloc(MemoryManager& mem) {
    if (s_mono_arena.base == 0) {
        GuestAddr cur = (g_guest_heap_current.load() + 15) & ~15ULL;
        if (!mem.is_valid_address(cur, kMonoArenaSize)) {
            std::fprintf(stderr, "[Mono] arena carve failed: heap exhausted\n");
            return 0;
        }
        s_mono_arena.base = cur;
        s_mono_arena.next = cur;
        s_mono_arena.end = cur + kMonoArenaSize;
        g_guest_heap_current.store(cur + kMonoArenaSize);
    }
    if (s_mono_arena.next + kMonoSlotSize > s_mono_arena.end) {
        std::fprintf(stderr, "[Mono] arena exhausted (4096 slots)\n");
        return 0;
    }
    const GuestAddr slot = s_mono_arena.next;
    s_mono_arena.next += kMonoSlotSize;
    mem.zero_memory(slot, kMonoSlotSize);
    return slot;
}

// Stable handle for a (function, argument-tuple) pair. Two lookups of the same
// field from the same class must return the same pointer, or every guest
// identity comparison fails and the caller retries forever. Keyed on the
// arguments, so distinct lookups get distinct zeroed slots.
// Caller must hold s_mono_arena.mtx.
static GuestAddr mono_cookie(MemoryManager& mem, const char* what, std::string_view args) {
    const std::string key = std::string(what) + "|" + std::string(args);
    auto it = s_mono_arena.cookies.find(key);
    if (it != s_mono_arena.cookies.end()) return it->second;
    const GuestAddr slot = mono_arena_alloc(mem);
    if (slot) s_mono_arena.cookies.emplace(key, slot);
    return slot;
}

static std::string mono_args(const CpuContext& ctx) {
    char buf[80];
    std::snprintf(buf, sizeof(buf), "%llx:%llx:%llx",
                  static_cast<unsigned long long>(ctx.rdi),
                  static_cast<unsigned long long>(ctx.rsi),
                  static_cast<unsigned long long>(ctx.rdx));
    return buf;
}

// ---------------------------------------------------------------------------
// Imports that must not return to the guest
// ---------------------------------------------------------------------------
// A body that only sets rax and returns is *not* a conservative stub for these.
// The guest has no code after the call that can make progress: what sits there
// is the callee's alignment sled, then the next function's prologue, so control
// walks into unrelated code until it branches to a garbage pointer. That is how
// a canary failure was reported as "Memory fault / invalid instruction pointer"
// at RIP=0x3b480151ed5a1d8d with the canary itself invisible (commit 9b8365c).
//
// So terminate, and print enough to name the failure. 134 = 128 + SIGABRT, the
// conventional abort() status, which also lets a run's exit code tell a guest
// abort apart from a clean exit.
[[noreturn]] static void guest_abort(std::string_view reason, const CpuContext& ctx,
                                     MemoryManager& mem) {
    // The return address of the call site is [rsp]; ctx.rip is wherever the
    // trampoline landed and is NOT the call site.
    GuestAddr ret = 0;
    mem.read_value(ctx.rsp, ret);
    std::fprintf(stderr,
                 "[Guest abort] %.*s\n"
                 "  call site (return addr) = 0x%llx\n"
                 "  rbp = 0x%llx   rsp = 0x%llx\n"
                 "  rdi = 0x%llx   rsi = 0x%llx   rdx = 0x%llx\n",
                 static_cast<int>(reason.size()), reason.data(),
                 static_cast<unsigned long long>(ret),
                 static_cast<unsigned long long>(ctx.rbp),
                 static_cast<unsigned long long>(ctx.rsp),
                 static_cast<unsigned long long>(ctx.rdi),
                 static_cast<unsigned long long>(ctx.rsi),
                 static_cast<unsigned long long>(ctx.rdx));
    dump_control_xfers((int)t_guest_tid);
    dump_write_watch_ring(ctx.rbp - 0x30); // the canary slot itself
    std::fprintf(stderr, "[Guest abort] terminating emulator\n");
    std::fflush(stderr);
    std::exit(134);
}

// exit(status) / _exit(status).
// Runs the process teardown the runner and every guest waiter loop key off
// (SyscallTable::has_exited, which is also what release of untimed POSIX
// waiters keys off), then stops *this* thread: there is nothing to return into
// once the process is gone. rip=0 is the same "stop executing" convention
// scePthreadExit already uses.
//
// quick_exit is also noreturn but the guest does not import it, so it is not
// registered: a dead registration with a name nothing imports is noise in the
// trace, not a fix.
static void libc_exit_handler(CpuContext& ctx, MemoryManager& mem) {
    const i32 status = static_cast<i32>(ctx.rdi);
    std::fprintf(stderr, "[HLE] Guest called exit(%d)\n", status);
    if (g_syscalls) {
        SyscallContext sctx(mem);
        sctx.args[0] = static_cast<u64>(status);
        g_syscalls->dispatch(SYS_exit, sctx);
    }
    ctx.rip = 0;
    ctx.rax = 0;
}

} // namespace

NidRegistry::NidRegistry() = default;

void NidRegistry::set_vfs(VirtualFileSystem* vfs) {
    g_external_vfs = vfs;
}

static MetalBackend* g_metal_backend = nullptr;
static std::atomic<u32> g_pad_buttons{0};

void NidRegistry::set_metal_backend(MetalBackend* metal) {
    g_metal_backend = metal;
}

void NidRegistry::set_syscall_table(SyscallTable* syscalls) {
    g_syscalls = syscalls;
    if (g_thread_manager && !g_thread_manager->syscall_table()) {
        g_thread_manager->set_syscall_table(syscalls);
    }
}

void NidRegistry::set_jit_compiler(JitCompiler* jit) {
    g_jit = jit;
    if (g_thread_manager) {
        g_thread_manager->set_jit_compiler(jit);
    }
}

NidRegistry::~NidRegistry() {
    shutdown();
}

// Defined next to the semaphore condition variable in
// register_default_libraries(); declared here because shutdown() appears earlier
// in the translation unit.
void request_semaphore_shutdown();

static std::mutex s_sem_cv_mtx;
static std::condition_variable s_sem_cv;
static std::atomic<bool> s_sem_shutdown{false};

// Unpark every untimed POSIX waiter so thread teardown can complete. has_exited()
// alone does not cover the HLT, instruction-budget, or fault exits.
void request_semaphore_shutdown() {
    s_sem_shutdown.store(true, std::memory_order_relaxed);
    s_sem_cv.notify_all();
}

void NidRegistry::shutdown() {
    // Teardown watchdog. Guest threads park indefinitely in untimed waits
    // (WaitSema with timeout=infinite, mutex_lock, sceKernelWaitEventFlag) and
    // none of them observe has_exited(), so ThreadManager's join() never
    // returns and a single run can wedge for the whole session. Guest threads do
    // not need orderly teardown -- once main has stopped, nothing depends on
    // their state -- so arm a detached timer that hard-exits shortly after.
    // If the joins happen to finish first, the normal exit wins and the timer
    // thread dies with the process.
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::fflush(nullptr);
        std::fprintf(stderr, "[shutdown] watchdog: threads still parked after 2s, hard-exiting\n");
        std::fflush(stderr);
        std::_Exit(0);
    }).detach();
    // Release every thread parked in an untimed POSIX wait before joining.
    // sem_wait blocks on a condition variable until it is posted or until the
    // emulator is shutting down; has_exited() is NOT enough, because it is only
    // set by an exit syscall. A guest that ends any other way -- HLT
    // (StepResult::Halted), the headless instruction budget, or a fault -- leaves
    // has_exited() false, so idle pool workers stayed parked and join() hung
    // forever. That is what turned a 46 s run into an unkillable one.
    request_semaphore_shutdown();
    // 1. Join and destroy all guest threads while MemoryManager, SyscallTable, and JitCompiler are still alive
    if (g_thread_manager) {
        g_thread_manager.reset();
        g_thread_manager_mem = nullptr;
    }
    // 2. Close audio backend and FIOS handles
    if (g_audio_manager) {
        g_audio_manager.reset();
    }
    s_fios_files.clear();

    // 3. Clear pointers to stack-allocated subsystems in main()
    g_syscalls = nullptr;
    g_jit = nullptr;
    g_external_vfs = nullptr;
    g_metal_backend = nullptr;
}

ThreadManager& NidRegistry::thread_manager(MemoryManager& mem) {
    return get_thread_manager(mem);
}

ThreadManager* NidRegistry::thread_manager() const noexcept {
    return g_thread_manager.get();
}

void NidRegistry::note_hle_dispatch(Nid nid, GuestAddr caller_rip) const {
    if (g_thread_manager) {
        const ScePthreadId tid = g_thread_manager->current_thread_id();
        if (GuestThread* th = g_thread_manager->get_thread(tid)) {
            th->last_nid.store(nid, std::memory_order_relaxed);
            th->hle_caller_rip.store(caller_rip, std::memory_order_relaxed);
            th->hle_calls.fetch_add(1, std::memory_order_relaxed);
            th->in_hle.store(true, std::memory_order_relaxed);
        }
    }
}

void NidRegistry::note_hle_return() const {
    if (g_thread_manager) {
        if (GuestThread* th = g_thread_manager->get_thread(g_thread_manager->current_thread_id())) {
            th->in_hle.store(false, std::memory_order_relaxed);
        }
    }
}

void NidRegistry::set_pad_buttons(u32 buttons) {
    g_pad_buttons.store(buttons, std::memory_order_relaxed);
}

void NidRegistry::register_symbol(Nid nid, std::string_view name, std::string_view module_name, HleFunction handler) {
    HleSymbol sym;
    sym.nid = nid;
    sym.name = std::string(name);
    sym.module_name = std::string(module_name);
    sym.handler = std::move(handler);

    m_symbols_by_nid[nid] = sym;
    m_nids_by_name[sym.name] = nid;
}

const HleSymbol* NidRegistry::find_by_nid(Nid nid) const {
    auto it = m_symbols_by_nid.find(nid);
    if (it != m_symbols_by_nid.end()) {
        return &it->second;
    }
    return nullptr;
}

const HleSymbol* NidRegistry::find_by_name(std::string_view name) const {
    auto it = m_nids_by_name.find(std::string(name));
    if (it != m_nids_by_name.end()) {
        return find_by_nid(it->second);
    }
    return nullptr;
}

// Helper: safely format a guest format string using up to 4 guest GP arguments
static std::string format_guest_string(const MemoryManager& mem, const char* fmt,
                                       u64 a0, u64 a1, u64 a2, u64 a3) {
    if (!fmt) return "";
    const u64 args[4] = {a0, a1, a2, a3};
    usize arg_idx = 0;
    std::string out;
    out.reserve(256);

    for (usize i = 0; fmt[i] != '\0'; ++i) {
        if (fmt[i] != '%') {
            out.push_back(fmt[i]);
            continue;
        }
        ++i;
        if (fmt[i] == '\0') break;
        if (fmt[i] == '%') {
            out.push_back('%');
            continue;
        }

        // Skip flags, width, precision, and length modifiers
        while (fmt[i] == '-' || fmt[i] == '+' || fmt[i] == ' ' || fmt[i] == '#' || fmt[i] == '0' ||
               (fmt[i] >= '1' && fmt[i] <= '9') || fmt[i] == '.') {
            ++i;
        }
        while (fmt[i] == 'l' || fmt[i] == 'h' || fmt[i] == 'z' || fmt[i] == 't') {
            ++i;
        }
        const char spec = fmt[i];
        if (spec == '\0') break;

        const u64 val = (arg_idx < 4) ? args[arg_idx++] : 0;
        char buf[64]{0};
        if (spec == 's') {
            if (val > 0x1000 && mem.is_valid_address(val, 1)) {
                const char* s = mem.to_host_ptr<char>(val);
                if (s) out.append(s);
            }
        } else if (spec == 'd' || spec == 'i') {
            std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(static_cast<i64>(val)));
            out.append(buf);
        } else if (spec == 'u') {
            std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(val));
            out.append(buf);
        } else if (spec == 'x' || spec == 'X' || spec == 'p') {
            std::snprintf(buf, sizeof(buf), "%llx", static_cast<unsigned long long>(val));
            out.append(buf);
        } else if (spec == 'c') {
            out.push_back(static_cast<char>(val & 0xFF));
        }
    }
    return out;
}

// Simulated GPU fence + legacy label slot for libSceGnmDriver. The slot address
// below was once handed to the guest; keep it fresh on every fence advance so a
// guest polling it never spins on a stale value. (It sits in the heap zone, so a
// proper reservation is still TBD.)
static std::atomic<u32> g_gnm_completed_label{1};
static constexpr GuestAddr kGnmLabelSlotAddr = 0x20003000ULL;

static void sync_gnm_label_slot(MemoryManager& mem) {
    if (mem.is_valid_address(kGnmLabelSlotAddr, sizeof(u32))) {
        mem.write_value(kGnmLabelSlotAddr, g_gnm_completed_label.load());
    }
}

// Sema-table snapshot for the stall watchdog (id, count/max per semaphore).
void dump_sema_states() {
    std::lock_guard<std::mutex> lock(s_sema_table_mtx);
    std::fprintf(stderr, "=== [SEMA] table snapshot (%zu entries) ===\n", s_semas.size());
    for (const auto& [id, sema] : s_semas) {
        if (!sema) continue;
        std::lock_guard<std::mutex> slock(sema->mtx);
        std::fprintf(stderr, "  [Sema %u] count=%d max=%d%s\n",
                     id, sema->count, sema->max_count,
                     sema->deleted ? " DELETED" : "");
    }
}

// One-shot synthetic wake for the stall experiment: nudge the two semaphores
// the gfx handshake is parked on. Fires once ever (see s_synthetic_wake_done).
void debug_wake_stall_semas() {
    for (u32 id : {0x33u, 0x38u}) {
        std::shared_ptr<GuestSema> sema;
        {
            std::lock_guard<std::mutex> lock(s_sema_table_mtx);
            auto it = s_semas.find(id);
            if (it != s_semas.end()) sema = it->second;
        }
        if (sema) {
            std::lock_guard<std::mutex> lock(sema->mtx);
            sema->count = std::min(sema->max_count, sema->count + 1);
            sema->cv.notify_all();
            std::fprintf(stderr, "[WATCHDOG] synthetic wake: SignalSema(0x%x) count now %d\n",
                         id, sema->count);
        } else {
            std::fprintf(stderr, "[WATCHDOG] synthetic wake: sema 0x%x not present\n", id);
        }
    }
}

void NidRegistry::register_default_libraries() {
    // =========================================================================
    // libkernel: Core Orbis OS runtime services
    // =========================================================================

    // sceKernelExitProcess(int status)
    register_symbol(0x6424D00705E3BEBEULL, "sceKernelExitProcess", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const i32 status = static_cast<i32>(ctx.rdi);
            std::cout << "[HLE] sceKernelExitProcess(" << status << ")\n" << std::flush;
            ctx.rax = 0;
        });

    // __stack_chk_fail(void)
    // NID: 0x3AEDE22F569BBE78ULL ("8Op3OmHl3zY") -- imported by eboot.bin;
    // ps4libdoc lists it as the stack-protector runtime's canary failure.
    //
    // This MUST NOT RETURN. The guest calls it when the canary at [rbp-0x30]
    // no longer matches the guard at *0x14fd360. It was registered in the
    // unimplemented-stub table, whose body returns 0, so the guest fell
    // through into the nop sled at 0x58d595 and kept executing until it
    // branched to a garbage pointer: RIP=0x3b480151ed5a1d8d, reported only as
    // "Memory fault / invalid instruction pointer" with the canary failure
    // invisible. Real __stack_chk_fail calls __fortify_fail and aborts, so
    // terminating is the faithful behaviour -- and it turns an unactionable
    // wild jump into a report that names the actual failure. See the shared
    // noreturn helper above.
    register_symbol(0x3AEDE22F569BBE78ULL, "__stack_chk_fail", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            // Canaries: the frame's copy lives at [rbp-0x30], the guard is the
            // pointer at 0x14fd360. Print both plus the 64 bytes around the
            // frame canary -- ASCII points at a string copy overrun, a run of
            // zeros at a store-width bug, a pointer-shaped word at a stray write.
            GuestAddr guard_ptr = 0, guard_val = 0, frame_canary = 0;
            mem.read_value(static_cast<GuestAddr>(0x14FD360), guard_ptr);
            if (guard_ptr) mem.read_value(guard_ptr, guard_val);
            mem.read_value(ctx.rbp - 0x30, frame_canary);
            std::fprintf(stderr,
                         "[Guest abort] __stack_chk_fail: canary mismatch\n"
                         "  rbp                      = 0x%llx  (canary slot = 0x%llx)\n"
                         "  guard ptr  *0x14fd360   = 0x%llx\n"
                         "  guard value             = 0x%016llx\n"
                         "  frame canary            = 0x%016llx\n",
                         (unsigned long long)ctx.rbp,
                         (unsigned long long)(ctx.rbp - 0x30),
                         (unsigned long long)guard_ptr,
                         (unsigned long long)guard_val,
                         (unsigned long long)frame_canary);
            // Full frame as qwords, [rbp-0x80 .. rbp+0x10). The canary slot
            // is [rbp-0x30]; reading the whole window settles whether the frame
            // holds an array of heap pointers (overflowed) or a struct.
            std::fprintf(stderr, "  frame [rbp-0x80 .. rbp+0x10) as qwords:\n");
            for (GuestAddr a = ctx.rbp - 0x80; a < ctx.rbp + 0x10; a += 8) {
                u64 v = 0;
                const bool ok = mem.read_value(a, v);
                std::fprintf(stderr, "   %s 0x%016llx : 0x%016llx%s\n",
                             (a == ctx.rbp - 0x30) ? "->" : "  ",
                             (unsigned long long)a, (unsigned long long)v,
                             ok ? "" : "  (unreadable)");
            }
            guest_abort("__stack_chk_fail: canary mismatch", ctx, mem);
        });

    // sceKernelLoadStartModule
    // NID: 0xC33BEA4F852A297FULL
    register_symbol(0xC33BEA4F852A297FULL, "sceKernelLoadStartModule", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            static u32 s_module_id = 0x2000;
            const GuestAddr path_addr = ctx.rdi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : "unknown";
            const u32 handle = ++s_module_id;
            std::fprintf(stderr, "[HLE] sceKernelLoadStartModule('%s') -> handle 0x%x\n", path, handle);
            ctx.rax = handle;
        });

    // sceKernelAllocateDirectMemory(off_t search_start, off_t search_end, size_t length, size_t align, int type, off_t* out_addr)
    // Canonical NID: 0xAD35F0EB9C662C80ULL ("rTXw65xmLIA"), Legacy test alias: 0x51EAECE91C9A12B0ULL
    auto alloc_direct_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestSize requested_len = ctx.rdx;
        const GuestSize align  = ctx.rcx > 0 ? ctx.rcx : GUEST_PAGE_SIZE;
        const GuestAddr out_ptr = ctx.r9;

        GuestAddr cur = g_direct_phys_current.load();
        cur = (cur + align - 1) & ~(align - 1);

        // Keep all direct memory allocations strictly below the 7GB thread-stack boundary (0x1C0000000)
        constexpr GuestAddr kDirectMemCeiling = 0x1C0000000ULL;
        GuestSize effective_advance = requested_len;
        if (cur + effective_advance > kDirectMemCeiling) {
            // Reserve virtual sub-range without overflowing into the 7GB..8GB stack region
            effective_advance = (cur < kDirectMemCeiling) ? std::min<GuestSize>(requested_len, 0x40000000ULL) : 0;
        }
        g_direct_phys_current.store(cur + effective_advance);

        std::fprintf(stderr, "[DirectMem] Allocated 0x%llx bytes at phys 0x%llx (next=0x%llx)\n",
                     (unsigned long long)requested_len,
                     (unsigned long long)cur,
                     (unsigned long long)g_direct_phys_current.load());

        if (mem.is_valid_address(out_ptr, sizeof(GuestAddr))) {
            mem.write_value(out_ptr, cur);
            ctx.rax = SCE_OK;
        } else {
            ctx.rax = static_cast<u64>(-14); // -EFAULT
        }
    };
    register_symbol(0x51EAECE91C9A12B0ULL, "sceKernelAllocateDirectMemory", "libkernel", alloc_direct_handler);
    register_symbol(0xAD35F0EB9C662C80ULL, "sceKernelAllocateDirectMemory", "libkernel", alloc_direct_handler);

    // sceKernelMapDirectMemory(void** out_addr, size_t length, int protection, int flags, off_t phys_addr, size_t align)
    // Canonical NID: 0x2FF4372C48C86E00ULL ("L-Q3LEjIbgA"), Legacy test alias: 0x6002A3859BC62DFBULL
    auto map_direct_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr out_ptr  = ctx.rdi;
        const GuestAddr phys_addr = ctx.r8;

        // In our unified memory architecture, virtual address maps 1:1 to direct memory
        if (mem.is_valid_address(out_ptr, sizeof(GuestAddr))) {
            mem.write_value(out_ptr, phys_addr);
            ctx.rax = SCE_OK;
        } else {
            ctx.rax = static_cast<u64>(-14);
        }
    };
    register_symbol(0x6002A3859BC62DFBULL, "sceKernelMapDirectMemory", "libkernel", map_direct_handler);
    register_symbol(0x2FF4372C48C86E00ULL, "sceKernelMapDirectMemory", "libkernel", map_direct_handler);

    // sceKernelUsleep(unsigned int useconds)
    // NID: 0x3230BAEFBC36C18BULL and 0xD637D72D15738AC7ULL ("1jfXLRVzisc")
    auto usleep_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const u32 usec = static_cast<u32>(ctx.rdi);
        auto& tm = get_thread_manager(mem);
        GuestThread* cur_th = tm.current_thread();
        if (cur_th) {
            cur_th->state.store(ThreadState::WaitingSleep);
            const u64 now_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            cur_th->sleep_until_us.store(now_us + usec);
        }

        // Unblock job queue flush loops if caller is 0x85ccbc / 0x85ccc6
        // JobQueue object is in %r15 at 0x8082f570; clearing pending counter [r15 + 0x98] completes flush
        if (ctx.r15 > 0x1000 && mem.is_valid_address(ctx.r15 + 0x98, sizeof(u64))) {
            mem.write_value(ctx.r15 + 0x98, static_cast<u64>(0));
        }

        if (usec > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(usec));
        } else {
            std::this_thread::yield();
        }

        if (cur_th) {
            cur_th->sleep_until_us.store(0);
            cur_th->state.store(ThreadState::Running);
        }
        ctx.rax = SCE_OK;
    };
    register_symbol(0x3230BAEFBC36C18BULL, "sceKernelUsleep", "libkernel", usleep_handler);
    register_symbol(0xD637D72D15738AC7ULL, "sceKernelUsleep", "libkernel", usleep_handler);

    // sceKernelVirtualQuery(const void* addr, int flags, void* info, size_t size)
    // NID: 0xAD58D1BC72745FA7ULL ("rVjRvHJ0X6c")
    register_symbol(0xAD58D1BC72745FA7ULL, "sceKernelVirtualQuery", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr addr = ctx.rdi;
            const GuestAddr info_ptr = ctx.rdx;
            const GuestSize size = ctx.rcx;

            if (addr >= 0x2000000ULL || !mem.is_valid_address(info_ptr, size)) {
                // Report SCE_KERNEL_ERROR_EACCES to signify end of mapped range
                ctx.rax = 0x8002000dULL;
                return;
            }

            mem.zero_memory(info_ptr, size);

            const u64 start = addr;
            const u64 end = 0x2000000ULL; // 32MB mapped range
            const u32 prot = 7;           // RWX
            const u8 flags = 1;           // Committed

            mem.write_value(info_ptr + 0x00, start);
            mem.write_value(info_ptr + 0x08, end);
            mem.write_value(info_ptr + 0x18, prot);
            mem.write_value(info_ptr + 0x20, flags);

            ctx.rax = SCE_OK;
        });

    // sceKernelDirectMemoryQuery(off_t phys_addr, int flags, void* info, size_t size)
    // NID: 0x047A2E2D0CE1D17DULL ("BHouLQzh0X0")
    register_symbol(0x047A2E2D0CE1D17DULL, "sceKernelDirectMemoryQuery", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr addr = ctx.rdi;
            const GuestAddr info_ptr = ctx.rdx;
            const GuestSize size = ctx.rcx;

            if (addr >= 0x2000000ULL || !mem.is_valid_address(info_ptr, size)) {
                // Report SCE_KERNEL_ERROR_EACCES to terminate the direct memory scan
                ctx.rax = 0x8002000dULL;
                return;
            }

            mem.zero_memory(info_ptr, size);

            const u64 start = addr;
            const u64 end = 0x2000000ULL;
            const u64 flags = 1;

            mem.write_value(info_ptr + 0x00, start);
            mem.write_value(info_ptr + 0x08, end);
            mem.write_value(info_ptr + 0x10, flags);

            ctx.rax = SCE_OK;
        });

    // Host-side full attr store keyed by guest attr slot. The guest slot holds a
    // single u64 (stack size); detach state and priority live here instead.
    static std::mutex s_pthread_attr_mtx;
    static std::unordered_map<GuestAddr, ScePthreadAttr> s_pthread_attrs;

    // scePthreadCreate(ScePthread* thread, const ScePthreadAttr* attr, void* (*entry)(void*), void* arg, const char* name)
    auto pthread_create_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr out_thread_ptr = ctx.rdi;
        const GuestAddr attr_ptr = ctx.rsi;
        const GuestAddr entry = ctx.rdx;
        const GuestAddr arg = ctx.rcx;
        const GuestAddr name_ptr = ctx.r8;
        const char* name = (name_ptr > 0x1000 && mem.is_valid_address(name_ptr, 1))
                               ? mem.to_host_ptr<char>(name_ptr)
                               : "ps4_thread";

        auto& tm = get_thread_manager(mem);
        ScePthreadAttr attr{};
        bool have_full_attr = false;
        {
            std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
            auto it = s_pthread_attrs.find(attr_ptr);
            if (it != s_pthread_attrs.end()) {
                attr = it->second;
                have_full_attr = true;
            }
        }
        if (!have_full_attr && attr_ptr != 0 && mem.is_valid_address(attr_ptr, sizeof(u64))) {
            u64 stored_stack_sz = 0;
            mem.read_value(attr_ptr, stored_stack_sz);
            if (stored_stack_sz >= 65536 && stored_stack_sz <= 16 * 1024 * 1024) {
                attr.stack_size = stored_stack_sz;
            }
        }
        const ScePthreadId tid = tm.create_thread(entry, arg, &attr, name ? name : "ps4_thread");
        if (out_thread_ptr != 0 && mem.is_valid_address(out_thread_ptr, sizeof(u64))) {
            mem.write_value(out_thread_ptr, static_cast<u64>(tid));
        }
        ctx.rax = SCE_OK;
    };
    register_symbol(0x7FD4FA3050DCDE51ULL, "scePthreadCreate", "libkernel", pthread_create_handler);
    register_symbol(0xE9482DC15FB4CDBEULL, "scePthreadCreate", "libkernel", pthread_create_handler);

    // scePthreadAttrInit NID: 0x9EC628351CB0C0D8ULL ("nsYoNRywwNg")
    // Orbis pthread_attr_t is an 8-byte opaque handle on the stack! Writing >8 bytes clobbers saved RBP!
    register_symbol(0x9EC628351CB0C0D8ULL, "scePthreadAttrInit", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr attr_ptr = ctx.rdi;
            if (attr_ptr != 0 && mem.is_valid_address(attr_ptr, sizeof(u64))) {
                mem.write_value(attr_ptr, static_cast<u64>(2 * 1024 * 1024));
                std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
                s_pthread_attrs[attr_ptr] = ScePthreadAttr{};
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetstacksize NID: 0x5135F325B5A18531ULL ("UTXzJbWhhTE")
    register_symbol(0x5135F325B5A18531ULL, "scePthreadAttrSetstacksize", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr attr_ptr = ctx.rdi;
            const u64 stack_sz = ctx.rsi;
            if (attr_ptr != 0 && mem.is_valid_address(attr_ptr, sizeof(u64))) {
                mem.write_value(attr_ptr, stack_sz);
                std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
                s_pthread_attrs[attr_ptr].stack_size = stack_sz;
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrGetschedparam NID: 0x1573D61CD93C39FDULL
    register_symbol(0x1573D61CD93C39FDULL, "scePthreadAttrGetschedparam", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr param_ptr = ctx.rsi;
            if (param_ptr != 0 && mem.is_valid_address(param_ptr, sizeof(i32))) {
                mem.write_value(param_ptr, static_cast<i32>(0));
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetdetachstate NID: 0xFD6ADEA6BB6ED10BULL
    register_symbol(0xFD6ADEA6BB6ED10BULL, "scePthreadAttrSetdetachstate", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const GuestAddr attr_ptr = ctx.rdi;
            const i32 state = static_cast<i32>(ctx.rsi);
            if (attr_ptr != 0) {
                std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
                s_pthread_attrs[attr_ptr].detachstate = state;
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetschedparam NID: 0x0F3112F61405E1FEULL
    register_symbol(0x0F3112F61405E1FEULL, "scePthreadAttrSetschedparam", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr attr_ptr = ctx.rdi;
            const GuestAddr param_ptr = ctx.rsi;
            i32 prio = 256;
            if (param_ptr > 0x1000 && mem.is_valid_address(param_ptr, sizeof(i32))) {
                mem.read_value(param_ptr, prio);
            }
            if (attr_ptr != 0) {
                std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
                s_pthread_attrs[attr_ptr].priority = prio;
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetinheritsched NID: 0x7976D44A911A4EC0ULL ("eXbUSpEaTsA")
    register_symbol(0x7976D44A911A4EC0ULL, "scePthreadAttrSetinheritsched", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetschedpolicy NID: 0xE3E87D133C0A1782ULL ("4+h9EzwKF4I")
    register_symbol(0xE3E87D133C0A1782ULL, "scePthreadAttrSetschedpolicy", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetaffinity NID: 0xDEAC603387B31130ULL ("3qxgM4ezETA")
    register_symbol(0xDEAC603387B31130ULL, "scePthreadAttrSetaffinity", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrDestroy NID: 0xEB6282C04326CDC3ULL ("62KCwEMmzcM")
    register_symbol(0xEB6282C04326CDC3ULL, "scePthreadAttrDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadSetaffinity NID: 0x6EDDC24C12A61B22ULL ("bt3CTBKmGyI")
    register_symbol(0x6EDDC24C12A61B22ULL, "scePthreadSetaffinity", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadJoin(ScePthread thread, void** value_ptr)
    // NID: 0xA27358F41CA7FD6FULL ("onNY9Byn-W8")
    register_symbol(0xA27358F41CA7FD6FULL, "scePthreadJoin", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const ScePthreadId tid = static_cast<ScePthreadId>(ctx.rdi);
            const GuestAddr val_ptr = ctx.rsi;
            auto& tm = get_thread_manager(mem);
            u64 ret_val = 0;
            if (tm.join_thread(tid, &ret_val)) {
                if (val_ptr != 0 && mem.is_valid_address(val_ptr, sizeof(u64))) {
                    mem.write_value(val_ptr, ret_val);
                }
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-3); // ESRCH
            }
        });

    // scePthreadSelf()
    // NID: 0x688F8E782CFCC6B4ULL ("aI+OeCz8xrQ")
    register_symbol(0x688F8E782CFCC6B4ULL, "scePthreadSelf", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            auto& tm = get_thread_manager(mem);
            ctx.rax = static_cast<u64>(tm.current_thread_id());
        });

    // scePthreadMutexInit(ScePthreadMutex* mutex, const ScePthreadMutexattr* attr, const char* name)
    register_symbol(0x429E57A7F66472D5ULL, "scePthreadMutexInit", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            const u32 mid = tm.mutex_create();
            if (mtx_ptr != 0 && mem.is_valid_address(mtx_ptr, sizeof(u32))) {
                mem.write_value(mtx_ptr, mid);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14); // EFAULT
            }
        });

    // Resolve the guest's mutex handle to one of ours.
    //
    // Unity's PlatformMutex (PlatformMutex.cpp) keeps a *guest-owned* struct:
    // copies of it carry whatever the game stored, not a handle we issued. A
    // handle we do not recognise is indistinguishable from an uninitialised
    // mutex, so adopt it: mint a real mutex and write the handle back. Before
    // this, the gfx worker spun on EINVAL forever:
    //
    //   [SCE] scePthreadMutexLock(&m_Mutex) returned 0xffffffea in
    //         .\PlatformDependent\PS4\Source\Threads\PlatformMutex.cpp(38)
    //   [SCE] sceKernelUsleep(0xf4240)          ; 1 ms, repeat
    auto resolve_mutex_id = [](CpuContext& ctx, MemoryManager& mem) -> u32 {
        const GuestAddr mtx_ptr = ctx.rdi;
        auto& tm = get_thread_manager(mem);
        u32 mid = 0;
        if (mtx_ptr != 0 && mem.read_value(mtx_ptr, mid) && mid != 0 && tm.mutex_exists(mid)) {
            return mid;
        }
        const u32 fresh = tm.mutex_create();
        if (mtx_ptr != 0 && mem.is_valid_address(mtx_ptr, sizeof(u32))) {
            mem.write_value(mtx_ptr, fresh);
        }
        return fresh;
    };

    // scePthreadMutexLock(ScePthreadMutex* mutex)
    register_symbol(0xE86895318FDC64A0ULL, "scePthreadMutexLock", "libkernel",
        [resolve_mutex_id](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = get_thread_manager(mem).mutex_lock(resolve_mutex_id(ctx, mem)) ? SCE_OK
                                                                                      : static_cast<u64>(-22);
        });

    // scePthreadMutexTrylock(ScePthreadMutex* mutex)
    // NID: 0xBA9A15AF330715E1ULL ("upoVrzMHFeE")
    register_symbol(0xBA9A15AF330715E1ULL, "scePthreadMutexTrylock", "libkernel",
        [resolve_mutex_id](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = get_thread_manager(mem).mutex_trylock(resolve_mutex_id(ctx, mem))
                          ? SCE_OK
                          : static_cast<u64>(-16); // EBUSY
        });

    // scePthreadMutexUnlock(ScePthreadMutex* mutex)
    register_symbol(0xEFB2023594895C85ULL, "scePthreadMutexUnlock", "libkernel",
        [resolve_mutex_id](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = get_thread_manager(mem).mutex_unlock(resolve_mutex_id(ctx, mem))
                          ? SCE_OK
                          : static_cast<u64>(-22);
        });

    // scePthreadMutexLock (alternate NID: 0xF542B5BCB6507EDEULL / "9UK1vLZQft4")
    register_symbol(0xF542B5BCB6507EDEULL, "scePthreadMutexLock", "libkernel",
        [this](CpuContext& ctx, MemoryManager& mem) {
            const auto* s = find_by_nid(0xE86895318FDC64A0ULL);
            if (s && s->handler) s->handler(ctx, mem);
        });

    // scePthreadMutexUnlock (alternate NID: 0xB67DD5943D211BADULL / "tn3VlD0hG60")
    register_symbol(0xB67DD5943D211BADULL, "scePthreadMutexUnlock", "libkernel",
        [this](CpuContext& ctx, MemoryManager& mem) {
            const auto* s = find_by_nid(0xEFB2023594895C85ULL);
            if (s && s->handler) s->handler(ctx, mem);
        });

    // scePthreadYield()
    // NID: 0x4FBDA1CFA7DFAB4FULL ("T72hz6ffq08")
    register_symbol(0x4FBDA1CFA7DFAB4FULL, "scePthreadYield", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexDestroy(ScePthreadMutex* mutex)
    // NID: 0xD8E7F47FEDE68611ULL ("2Of0f+3mhhE")
    register_symbol(0xD8E7F47FEDE68611ULL, "scePthreadMutexDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 mid = 0;
            if (mem.read_value(mtx_ptr, mid)) {
                ctx.rax = tm.mutex_destroy(mid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondInit(ScePthreadCond* cond, const ScePthreadCondattr* attr, const char* name)
    // NID: 0xD936FDDAABA9AE5DULL ("2Tb92quprl0")
    register_symbol(0xD936FDDAABA9AE5DULL, "scePthreadCondInit", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            const u32 cid = tm.cond_create();
            if (cond_ptr != 0 && mem.is_valid_address(cond_ptr, sizeof(u32))) {
                mem.write_value(cond_ptr, cid);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondWait(ScePthreadCond* cond, ScePthreadMutex* mutex)
    // NID: 0x58A0172785C13D0EULL ("WKAXJ4XBPQ4")
    register_symbol(0x58A0172785C13D0EULL, "scePthreadCondWait", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            const GuestAddr mtx_ptr = ctx.rsi;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0, mid = 0;
            if (mem.read_value(cond_ptr, cid) && mem.read_value(mtx_ptr, mid)) {
                ctx.rax = tm.cond_wait(cid, mid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondSignal(ScePthreadCond* cond)
    // NID: 0x90387F35FC6032D1ULL ("kDh-NfxgMtE")
    register_symbol(0x90387F35FC6032D1ULL, "scePthreadCondSignal", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0;
            if (mem.read_value(cond_ptr, cid)) {
                ctx.rax = tm.cond_signal(cid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondBroadcast(ScePthreadCond* cond)
    // NID: 0x246823ED4BEB97E0ULL ("JGgj7Uvrl+A")
    register_symbol(0x246823ED4BEB97E0ULL, "scePthreadCondBroadcast", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0;
            if (mem.read_value(cond_ptr, cid)) {
                ctx.rax = tm.cond_broadcast(cid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondDestroy(ScePthreadCond* cond)
    // NID: 0x83E3D977686269C8ULL ("g+PZd2hiacg")
    register_symbol(0x83E3D977686269C8ULL, "scePthreadCondDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0;
            if (mem.read_value(cond_ptr, cid)) {
                ctx.rax = tm.cond_destroy(cid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondTimedwait(ScePthreadCond *cond, ScePthreadMutex *mutex, SceKernelUseconds usec)
    // NID: 0x06632363199EC35CULL ("BmMjYxmew1w")
    register_symbol(0x06632363199EC35CULL, "scePthreadCondTimedwait", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            const GuestAddr mtx_ptr = ctx.rsi;
            const u64 timeout_usec = ctx.rdx;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0, mid = 0;
            if (mem.read_value(cond_ptr, cid) && mem.read_value(mtx_ptr, mid)) {
                if (cid == 0) {
                    cid = tm.cond_create();
                    mem.write_value(cond_ptr, cid);
                }
                if (mid == 0) {
                    mid = tm.mutex_create();
                    mem.write_value(mtx_ptr, mid);
                }
                const bool ok = tm.cond_timedwait(cid, mid, timeout_usec);
                ctx.rax = ok ? SCE_OK : 0x8002003CULL; // SCE_KERNEL_ERROR_ETIMEDOUT
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadRename(ScePthread thread, const char *name)
    // NID: 0x181518ef2c1d50b1ULL ("GBUY7ywdULE")
    register_symbol(0x181518ef2c1d50b1ULL, "scePthreadRename", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const ScePthreadId tid = static_cast<ScePthreadId>(ctx.rdi);
            const GuestAddr name_ptr = ctx.rsi;
            const char* name = (name_ptr != 0 && mem.is_valid_address(name_ptr, 1))
                                   ? mem.to_host_ptr<char>(name_ptr)
                                   : "unnamed";
            auto& tm = get_thread_manager(mem);
            tm.rename_thread(tid, name);
            std::fprintf(stderr, "[HLE] scePthreadRename(tid=%u, name='%s')\n", tid, name);
            ctx.rax = SCE_OK;
        });

    // scePthreadExit(void *value_ptr)
    // NID: 0xDE483BAD3D0D408BULL. Shares the body with pthread_exit above;
    // Orbis's own thread-exit import and the POSIX one are the same operation.

    // sceKernelCreateEqueue(SceKernelEqueue *eq, const char *name)
    // NID: 0x0F439D14C8E9E3A2ULL ("D0OdFMjp46I")
    register_symbol(0x0F439D14C8E9E3A2ULL, "sceKernelCreateEqueue", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr eq_ptr = ctx.rdi;
            const GuestAddr name_ptr = ctx.rsi;
            const char* name = (name_ptr != 0 && mem.is_valid_address(name_ptr, 1))
                                   ? mem.to_host_ptr<char>(name_ptr)
                                   : "equeue";
            auto& tm = get_thread_manager(mem);
            const u32 eq_id = tm.equeue_create(name);
            std::fprintf(stderr, "[HLE] sceKernelCreateEqueue('%s') -> id=%u\n", name, eq_id);
            if (eq_ptr != 0 && mem.is_valid_address(eq_ptr, sizeof(u32))) {
                mem.write_value(eq_ptr, eq_id);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14); // -EFAULT
            }
        });

    // sceKernelDeleteEqueue(SceKernelEqueue eq)
    // NID: 0x8E91639A0002E401ULL ("jpFjmgAC5AE")
    register_symbol(0x8E91639A0002E401ULL, "sceKernelDeleteEqueue", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const u32 eq_id = static_cast<u32>(ctx.rdi);
            auto& tm = get_thread_manager(mem);
            std::fprintf(stderr, "[HLE] sceKernelDeleteEqueue(id=%u)\n", eq_id);
            ctx.rax = tm.equeue_delete(eq_id) ? SCE_OK : static_cast<u64>(-22);
        });

    // exit(int status) / _exit(int status) -- NIDs 0xB8C7A2D56F6EC8DA /
    // 0xE99F37B18585940F. Both are imported by the guest (exit by eboot.bin,
    // _exit by libc.prx) and both are noreturn; they were falling through to
    // the unimplemented stub, which returns 0 and lets the guest walk into the
    // callee's alignment sled after the process is already gone. The bodies are
    // identical because we keep no atexit registry to skip for _exit: that is a
    // consequence of the __cxa_atexit decision further down, not a body left
    // unimplemented.
    register_symbol(0xB8C7A2D56F6EC8DAULL, "exit", "libc", libc_exit_handler);
    register_symbol(0xE99F37B18585940FULL, "_exit", "libc", libc_exit_handler);

    // pthread_exit(void* retval) -- NID: 0x149AD3E4BB940405, imported by
    // eboot.bin. noreturn for the *calling thread*: it was unregistered, so the
    // stub returned 0 and the thread went on executing the function that had
    // just called pthread_exit. scePthreadExit (0xDE483BAD3D0D408B) shares the
    // body -- Orbis's own thread-exit import and the POSIX one are the same
    // operation.
    auto pthread_exit_handler = [](CpuContext& ctx, MemoryManager& mem) {
        auto& tm = get_thread_manager(mem);
        tm.exit_current_thread(ctx.rdi);
        ctx.rip = 0;
        ctx.rax = SCE_OK;
    };
    register_symbol(0x149AD3E4BB940405ULL, "pthread_exit", "libc", pthread_exit_handler);
    register_symbol(0xDE483BAD3D0D408BULL, "scePthreadExit", "libkernel", pthread_exit_handler);

    // sceKernelWaitEqueue(SceKernelEqueue eq, SceKernelEvent *ev, int num, int *out, const SceKernelUseconds *timo)
    // NID: 0x7F3C8C2ACF648A6DULL ("fzyMKs9kim0")
    register_symbol(0x7F3C8C2ACF648A6DULL, "sceKernelWaitEqueue", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const u32 eq_id = static_cast<u32>(ctx.rdi);
            const GuestAddr ev_ptr = ctx.rsi;
            const i32 num = static_cast<i32>(ctx.rdx);
            const GuestAddr out_ptr = ctx.rcx;
            const GuestAddr timo_ptr = ctx.r8;

            u64 timeout_usec = 0;
            bool has_timeout = false;
            if (timo_ptr != 0 && mem.is_valid_address(timo_ptr, sizeof(u32))) {
                u32 usec = 0;
                mem.read_value(timo_ptr, usec);
                timeout_usec = usec;
                has_timeout = true;
            }

            auto& tm = get_thread_manager(mem);
            std::vector<SceKernelEvent> out_events;
            const bool ok = tm.equeue_wait(eq_id, out_events, num > 0 ? static_cast<usize>(num) : 1, timeout_usec, has_timeout);

            if (out_ptr != 0 && mem.is_valid_address(out_ptr, sizeof(i32))) {
                mem.write_value(out_ptr, static_cast<i32>(out_events.size()));
            }

            if (ok && !out_events.empty()) {
                if (ev_ptr != 0 && mem.is_valid_address(ev_ptr, out_events.size() * sizeof(SceKernelEvent))) {
                    for (usize i = 0; i < out_events.size(); ++i) {
                        const GuestAddr slot = ev_ptr + (i * sizeof(SceKernelEvent));
                        mem.write_value(slot + 0x00, out_events[i].ident);
                        mem.write_value(slot + 0x08, out_events[i].filter);
                        mem.write_value(slot + 0x0A, out_events[i].flags);
                        mem.write_value(slot + 0x0C, out_events[i].fflags);
                        mem.write_value(slot + 0x10, out_events[i].data);
                        mem.write_value(slot + 0x18, out_events[i].udata);
                    }
                }
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = 0x8002003CULL; // SCE_KERNEL_ERROR_ETIMEDOUT
            }
        });

    // sceKernelTriggerUserEvent(SceKernelEqueue eq, int id, void *udata)
    // NID: 0x17A7B4930A387279ULL ("F6e0kwo4cnk")
    register_symbol(0x17A7B4930A387279ULL, "sceKernelTriggerUserEvent", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const u32 eq_id = static_cast<u32>(ctx.rdi);
            const i32 event_id = static_cast<i32>(ctx.rsi);
            const u64 udata = ctx.rdx;
            auto& tm = get_thread_manager(mem);
            std::fprintf(stderr, "[HLE] sceKernelTriggerUserEvent(eq=%u, event_id=%d, udata=0x%llx)\n",
                         eq_id, event_id, (unsigned long long)udata);
            ctx.rax = tm.equeue_trigger_user_event(eq_id, event_id, udata) ? SCE_OK : static_cast<u64>(-22);
        });

    // sceKernelAddUserEventEdge(SceKernelEqueue eq, int id)
    // NID: 0x583B339926D6B839ULL ("WDszmSbWuDk")
    register_symbol(0x583B339926D6B839ULL, "sceKernelAddUserEventEdge", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceKernelDeleteUserEvent(SceKernelEqueue eq, int id)
    // NID: 0x2C90F07523539C38ULL ("LJDwdSNTrDg")
    register_symbol(0x2C90F07523539C38ULL, "sceKernelDeleteUserEvent", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceKernelGettimeofday(struct timeval* tp, struct timezone* tzp)
    register_symbol(0x78D1E6F34C96C7A1ULL, "sceKernelGettimeofday", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr tp_ptr = ctx.rdi;
            if (tp_ptr != 0 && mem.is_valid_address(tp_ptr, 16)) {
                struct timeval tv{};
                ::gettimeofday(&tv, nullptr);
                mem.write_value(tp_ptr, static_cast<i64>(tv.tv_sec));
                mem.write_value(tp_ptr + 8, static_cast<i64>(tv.tv_usec));
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // =========================================================================
    // libc: Standard C library services
    // =========================================================================

    // puts(const char* s) body. ps4libdoc puts this NID under a different name
    // (0x610D276AFA7E6087 below is the real "puts"), so this registration keeps
    // an honest unknown_* label while serving as the shared implementation.
    register_symbol(0x5C0F663677A000A8ULL, "unknown_5c0f6636", "libSceLibcInternal",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr str_addr = ctx.rdi;
            if (mem.is_valid_address(str_addr)) {
                const char* str = mem.to_host_ptr<char>(str_addr);
                if (str) {
                    // Flush immediately: std::cout is block-buffered when stdout is
                    // not a TTY, so guest output was being dropped when the run is
                    // killed and ended up reordered against the unbuffered stderr
                    // [HLE CALL] trace. See also the printf handler below.
                    std::cout << str << "\n" << std::flush;
                    ctx.rax = 0;
                    return;
                }
            }
            ctx.rax = static_cast<u64>(-1);
        });

    // malloc(size_t size)
    // Note: no zeroing — the bump allocator only hands out fresh OS-zero pages
    // (free is a no-op), so untouched pages stay purely virtual until written.
    auto malloc_handler = [](CpuContext& ctx, MemoryManager&) {
        const GuestSize size = ctx.rdi;
        ctx.rax = bump_alloc(size, 16);
    };
    register_symbol(0x9E990E49EC698A4DULL, "malloc", "libc", malloc_handler);
    register_symbol(0x8105FEE060D08E93ULL, "malloc", "libc", malloc_handler);

    // free(void* ptr)
    auto free_handler = [](CpuContext& ctx, MemoryManager&) {
        ctx.rax = 0;
    };
    register_symbol(0x7E16B7B8361BD5BBULL, "free", "libc", free_handler);
    register_symbol(0xB4886CAA3D2AB051ULL, "free", "libc", free_handler);

    // realloc(void* ptr, size_t size) -- NID: 0x63B689D6EC9D3CCAULL per
    // ps4libdoc. This NID is imported by eboot.bin and mono-ps4.prx but was
    // registered as a third "free", so every realloc silently returned NULL.
    // The bump allocator never reuses blocks, so the copy is all we owe the
    // caller; bump_alloc() knows the old size without a block header.
    register_symbol(0x63B689D6EC9D3CCAULL, "realloc", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr old_ptr = ctx.rdi;
            const GuestSize size = ctx.rsi;
            if (size == 0) {
                ctx.rax = 0;
                return;
            }
            const GuestAddr allocated = bump_alloc(size, 16);
            const GuestSize old_size = bump_size_of(old_ptr);
            if (old_ptr != 0 && old_size > 0 && mem.is_valid_address(old_ptr, old_size)) {
                std::memcpy(mem.to_host_ptr<void>(allocated), mem.to_host_ptr<void>(old_ptr),
                            old_size < size ? old_size : size);
            }
            ctx.rax = allocated;
        });

    // 0xB50228F8620F925AULL is _ZSt14_Xlength_errorPKc per ps4libdoc, i.e. the
    // C++ std::length_error thrower -- it was registered here as "calloc".
    // eboot.bin and libSceSmart.prx import this NID, but only reach it when an
    // allocation failed in real firmware. Body: hand back a fresh block
    // instead of throwing, so a guest error path can never unwind through our
    // un-emulated C++ runtime.
    register_symbol(0xB50228F8620F925AULL, "_ZSt14_Xlength_errorPKc", "libstdc++",
        [](CpuContext& ctx, MemoryManager&) {
            const GuestSize num = ctx.rdi;
            const GuestSize size = ctx.rsi;
            ctx.rax = bump_alloc(num * size, 16);
        });

    // setjmp(jmp_buf env)
    // NID: 0x80D435576BDF5C31ULL / 0x315CDF6B5735D480ULL ("gNQ1V2vfXDE")
    auto setjmp_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr env = ctx.rdi;
        if (env != 0 && mem.is_valid_address(env, 64)) {
            GuestAddr ret_addr = 0;
            mem.read_value(ctx.rsp, ret_addr);
            mem.write_value(env + 0x00, ctx.rbx);
            mem.write_value(env + 0x08, ctx.rsp + 8); // caller's stack pointer after ret
            mem.write_value(env + 0x10, ctx.rbp);
            mem.write_value(env + 0x18, ctx.r12);
            mem.write_value(env + 0x20, ctx.r13);
            mem.write_value(env + 0x28, ctx.r14);
            mem.write_value(env + 0x30, ctx.r15);
            mem.write_value(env + 0x38, ret_addr);    // caller's return RIP
        }
        ctx.rax = 0;
    };
    register_symbol(0x80D435576BDF5C31ULL, "setjmp", "libc", setjmp_handler);
    register_symbol(0x315CDF6B5735D480ULL, "setjmp", "libc", setjmp_handler);

    // longjmp(jmp_buf env, int val)
    // NID: 0x94A10DD8879B809DULL / 0x9D809B87D80DA194ULL ("lKEN2IebgJ0")
    auto longjmp_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr env = ctx.rdi;
        const u32 val = static_cast<u32>(ctx.rsi);
        if (env != 0 && mem.is_valid_address(env, 64)) {
            GuestAddr saved_rbx = 0, saved_rsp = 0, saved_rbp = 0;
            GuestAddr saved_r12 = 0, saved_r13 = 0, saved_r14 = 0, saved_r15 = 0;
            GuestAddr saved_rip = 0;
            mem.read_value(env + 0x00, saved_rbx);
            mem.read_value(env + 0x08, saved_rsp);
            mem.read_value(env + 0x10, saved_rbp);
            mem.read_value(env + 0x18, saved_r12);
            mem.read_value(env + 0x20, saved_r13);
            mem.read_value(env + 0x28, saved_r14);
            mem.read_value(env + 0x30, saved_r15);
            mem.read_value(env + 0x38, saved_rip);

            ctx.rbx = saved_rbx;
            ctx.rbp = saved_rbp;
            ctx.r12 = saved_r12;
            ctx.r13 = saved_r13;
            ctx.r14 = saved_r14;
            ctx.r15 = saved_r15;
            ctx.rax = (val == 0) ? 1 : val;

            ctx.rsp = saved_rsp;
            ctx.rip = saved_rip;
        }
    };
    register_symbol(0x94A10DD8879B809DULL, "longjmp", "libc", longjmp_handler);
    register_symbol(0x9D809B87D80DA194ULL, "longjmp", "libc", longjmp_handler);

    // powf(float x, float y)
    // NID: 0xD43D07D8A363B211ULL / 0x11B263A3D8073DD4ULL ("1D0H2KNjshE")
    auto powf_handler = [](CpuContext& ctx, MemoryManager&) {
        const float x = ctx.ymm[0].f[0];
        const float y = ctx.ymm[1].f[0];
        ctx.ymm[0].f[0] = std::pow(x, y);
    };
    register_symbol(0xD43D07D8A363B211ULL, "powf", "libc", powf_handler);
    register_symbol(0x11B263A3D8073DD4ULL, "powf", "libc", powf_handler);

    // sinf(float x)
    auto sinf_handler = [](CpuContext& ctx, MemoryManager&) {
        ctx.ymm[0].f[0] = std::sin(ctx.ymm[0].f[0]);
    };
    register_symbol(0xE19F218ECFDE29A6ULL, "sinf", "libc", sinf_handler);

    // cosf(float x)
    auto cosf_handler = [](CpuContext& ctx, MemoryManager&) {
        ctx.ymm[0].f[0] = std::cos(ctx.ymm[0].f[0]);
    };
    register_symbol(0xD9C650B0145DB17CULL, "cosf", "libc", cosf_handler);

    // atan2f(float y, float x)
    auto atan2f_handler = [](CpuContext& ctx, MemoryManager&) {
        ctx.ymm[0].f[0] = std::atan2(ctx.ymm[0].f[0], ctx.ymm[1].f[0]);
    };
    register_symbol(0xC37A8C8C7F0ED898ULL, "atan2f", "libc", atan2f_handler);

    // memcmp(const void* s1, const void* s2, size_t n)
    // NID: 0x0DF8AF3C0AE1B9C8ULL / 0xC8B9E10AC0F38A0DULL ("DfivPArhucg")
    auto memcmp_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr s1 = ctx.rdi;
        const GuestAddr s2 = ctx.rsi;
        const GuestSize n  = ctx.rdx;
        if (n == 0 || s1 == s2) {
            ctx.rax = 0;
            return;
        }
        if (!mem.is_valid_address(s1, n) || !mem.is_valid_address(s2, n)) {
            ctx.rax = (s1 < s2) ? -1 : 1;
            return;
        }
        const u8* p1 = mem.to_host_ptr<u8>(s1);
        const u8* p2 = mem.to_host_ptr<u8>(s2);
        if (p1 && p2) {
            ctx.rax = static_cast<i64>(std::memcmp(p1, p2, n));
        } else {
            ctx.rax = 0;
        }
    };
    register_symbol(0x0DF8AF3C0AE1B9C8ULL, "memcmp", "libc", memcmp_handler);
    register_symbol(0xC8B9E10AC0F38A0DULL, "memcmp", "libc", memcmp_handler);

    // abort(void)
    // NID: 0x2F54814E40BE0AFCULL ("L1SBTkC+Cvw") / 0xFC0ABE404E81542FULL, both
    // imported by eboot.bin.
    //
    // This MUST NOT RETURN, and it was returning: the body set rax = -1 and
    // came back, so the guest resumed the code after abort() inside a process
    // it had just killed -- the identical failure mode __stack_chk_fail had
    // before 9b8365c. Real abort() raises SIGABRT, so terminating is faithful
    // and turns a silent walk into the guest's own alignment sled into a
    // report that names the abort and the call site that raised it.
    auto abort_handler = [](CpuContext& ctx, MemoryManager& mem) {
        guest_abort("abort(): guest raised SIGABRT", ctx, mem);
    };
    register_symbol(0x2F54814E40BE0AFCULL, "abort", "libc", abort_handler);
    register_symbol(0xFC0ABE404E81542FULL, "abort", "libc", abort_handler);

    // sceCommonDialogInitialize
    register_symbol(0xBA85292C6364CA09ULL, "sceCommonDialogInitialize", "libSceCommonDialog",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // memset(void* dest, int ch, size_t n)
    // NID: 0xF334C5BC120020DFULL ("8zTFvBIAIN8")
    register_symbol(0xF334C5BC120020DFULL, "memset", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const u8 ch          = static_cast<u8>(ctx.rsi & 0xFF);
            const GuestSize n    = ctx.rdx;
            if (dest != 0 && n > 0 && mem.is_valid_address(dest, n)) {
                void* d_ptr = mem.to_host_ptr<void>(dest);
                if (d_ptr) {
                    std::memset(d_ptr, ch, n);
                }
            }
            ctx.rax = dest;
        });

    // snprintf(char* str, size_t size, const char* format, ...)
    // NID: 0x78B743C3A974FDB5ULL ("eLdDw6l0-bU")
    register_symbol(0x78B743C3A974FDB5ULL, "snprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr buf = ctx.rdi;
            const GuestSize sz  = ctx.rsi;
            const GuestAddr fmt_addr = ctx.rdx;
            const char* fmt = (fmt_addr > 0x1000 && mem.is_valid_address(fmt_addr, 1))
                                ? mem.to_host_ptr<char>(fmt_addr) : nullptr;
            if (!fmt) {
                ctx.rax = 0;
                return;
            }

            const std::string formatted = format_guest_string(mem, fmt, ctx.rcx, ctx.r8, ctx.r9, 0);
            if (buf != 0 && sz > 0 && mem.is_valid_address(buf, sz)) {
                char* dst = mem.to_host_ptr<char>(buf);
                if (dst) {
                    const usize copy_len = std::min<usize>(formatted.size(), sz - 1);
                    std::memcpy(dst, formatted.data(), copy_len);
                    dst[copy_len] = '\0';
                }
            }
            ctx.rax = formatted.size();
        });

    // fopen(const char* path, const char* mode)
    // NID: 0xC5E60EE2EEEEC89DULL ("xeYO4u7uyJ0")
    register_symbol(0xC5E60EE2EEEEC89DULL, "fopen", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rdi;
            const GuestAddr mode_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                ? mem.to_host_ptr<char>(path_addr) : nullptr;
            const char* mode = (mode_addr > 0x1000 && mem.is_valid_address(mode_addr, 1))
                                ? mem.to_host_ptr<char>(mode_addr) : "rb";
            if (!path) {
                ctx.rax = 0;
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            const std::string host_path = resolved ? resolved->string() : std::string(path);
            FILE* fp = std::fopen(host_path.c_str(), mode);
            std::fprintf(stderr, "[LIBC fopen] fopen('%s' -> '%s', '%s') -> %p\n", path, host_path.c_str(), mode, static_cast<void*>(fp));
            ctx.rax = reinterpret_cast<u64>(fp);
        });

    // fseek(FILE* stream, long offset, int whence)
    // NID: 0xAD0155057A7F0B18ULL
    register_symbol(0xAD0155057A7F0B18ULL, "fseek", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            FILE* fp = reinterpret_cast<FILE*>(ctx.rdi);
            const long offset = static_cast<long>(ctx.rsi);
            const int whence = static_cast<int>(ctx.rdx);
            if (!fp) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            ctx.rax = static_cast<u64>(static_cast<i64>(std::fseek(fp, offset, whence)));
        });

    // ftell(FILE* stream)
    // NID: 0x41ACF2F0B9974EFCULL
    register_symbol(0x41ACF2F0B9974EFCULL, "ftell", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            FILE* fp = reinterpret_cast<FILE*>(ctx.rdi);
            if (!fp) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const long pos = std::ftell(fp);
            std::fprintf(stderr, "[LIBC ftell] ftell(%p) -> %ld\n", static_cast<void*>(fp), pos);
            ctx.rax = static_cast<u64>(pos);
        });

    // fread(void* ptr, size_t size, size_t nmemb, FILE* stream)
    // NID: 0x95B07E52566A546DULL
    register_symbol(0x95B07E52566A546DULL, "fread", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dst_addr = ctx.rdi;
            const usize size = ctx.rsi;
            const usize nmemb = ctx.rdx;
            FILE* fp = reinterpret_cast<FILE*>(ctx.rcx);
            const usize total_bytes = size * nmemb;
            if (!fp || dst_addr == 0 || total_bytes == 0 || !mem.is_valid_address(dst_addr, total_bytes)) {
                ctx.rax = 0;
                return;
            }
            void* dst = mem.to_host_ptr<void>(dst_addr);
            const usize items_read = std::fread(dst, size, nmemb, fp);
            std::fprintf(stderr, "[LIBC fread] fread(dst=0x%llx, size=%zu, nmemb=%zu, fp=%p) -> %zu\n",
                         (unsigned long long)dst_addr, size, nmemb, static_cast<void*>(fp), items_read);
            ctx.rax = items_read;
        });

    // fclose(FILE* stream)
    // NID: 0xBA874B632522A76DULL
    register_symbol(0xBA874B632522A76DULL, "fclose", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            FILE* fp = reinterpret_cast<FILE*>(ctx.rdi);
            if (!fp) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::fprintf(stderr, "[LIBC fclose] fclose(%p)\n", static_cast<void*>(fp));
            ctx.rax = static_cast<u64>(std::fclose(fp));
        });

    // vsnprintf(char* str, size_t size, const char* format, va_list ap)
    // NID: 0x43657E8AABE3802DULL ("Q2V+iqvjgC0")
    register_symbol(0x43657E8AABE3802DULL, "vsnprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestSize maxlen = ctx.rsi;
            const GuestAddr fmt_addr = ctx.rdx;
            const GuestAddr ap = ctx.rcx;

            const char* fmt = (fmt_addr > 0x1000 && mem.is_valid_address(fmt_addr, 1))
                                ? mem.to_host_ptr<char>(fmt_addr) : nullptr;
            if (!fmt) {
                ctx.rax = 0;
                return;
            }

            // Extract up to 4 GP arguments from x86_64 System V va_list
            u64 args[4] = {0, 0, 0, 0};
            if (ap != 0 && mem.is_valid_address(ap, 24)) {
                u32 gp_offset = 0;
                GuestAddr overflow_area = 0, reg_save = 0;
                mem.read_value(ap + 0, gp_offset);
                mem.read_value(ap + 8, overflow_area);
                mem.read_value(ap + 16, reg_save);

                for (int i = 0; i < 4; ++i) {
                    if (gp_offset < 48 && mem.is_valid_address(reg_save + gp_offset, 8)) {
                        mem.read_value(reg_save + gp_offset, args[i]);
                        gp_offset += 8;
                    } else if ( overflow_area > 0x1000 && mem.is_valid_address(overflow_area, 8)) {
                        mem.read_value(overflow_area, args[i]);
                        overflow_area += 8;
                    }
                }
            }

            const std::string formatted = format_guest_string(mem, fmt, args[0], args[1], args[2], args[3]);

            if (dest != 0 && maxlen > 0 && mem.is_valid_address(dest, maxlen)) {
                char* out = mem.to_host_ptr<char>(dest);
                if (out) {
                    const usize copy_len = std::min<usize>(formatted.size(), maxlen - 1);
                    std::memcpy(out, formatted.data(), copy_len);
                    out[copy_len] = '\0';
                }
            }

            if (formatted.size() > 1 && formatted != "/app0/media/boot.config") {
                std::fprintf(stderr, "[Unity Log] %s", formatted.c_str());
                if (formatted.back() != '\n') std::fprintf(stderr, "\n");
            }

            ctx.rax = formatted.size();
        });

    // strchr(const char* s, int c) -> char*   (returns the pointer, not an offset)
    // NID: 0xA1BE71016E259FFDULL (libSceLibcInternal per ps4libdoc).
    // Unity's parser calls this on every token; the generic stub returned 0,
    // which is indistinguishable from "character not found" and silently
    // truncated parsing at the first token.
    register_symbol(0xA1BE71016E259FFDULL, "strchr", "libSceLibcInternal",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr str_addr = ctx.rdi;
            const GuestAddr needle = ctx.rsi; // truncated to unsigned char
            if (str_addr == 0 || !mem.is_valid_address(str_addr, 1)) {
                ctx.rax = 0;
                return;
            }
            const char* s = mem.to_host_ptr<char>(str_addr);
            if (!s) {
                ctx.rax = 0;
                return;
            }
            const char* hit = std::strchr(s, static_cast<int>(needle & 0xFF));
            ctx.rax = hit ? (str_addr + (hit - s)) : 0;
        });

    // gettimeofday(struct timeval* tv, struct timezone* tz) -- ps4libdoc:
    // libScePosix/libkernel. Also answers the sceKernelGettimeofday NID, which
    // shares this body.
    auto gettimeofday_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr tv_ptr = ctx.rdi;
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const auto secs = std::chrono::duration_cast<std::chrono::seconds>(now);
        const auto usecs = std::chrono::duration_cast<std::chrono::microseconds>(now - secs);
        if (tv_ptr != 0 && mem.is_valid_address(tv_ptr, 16)) {
            mem.write_value(tv_ptr, static_cast<u64>(secs.count()));            // tv_sec
            mem.write_value(tv_ptr + 8, static_cast<u64>(usecs.count()));       // tv_usec
        }
        ctx.rax = SCE_OK;
    };
    register_symbol(0x9FCF2FC770B99D6FULL, "gettimeofday", "libScePosix", gettimeofday_handler);
    register_symbol(0x7A37A471A35036ADULL, "sceKernelGettimeofday", "libkernel", gettimeofday_handler);

    // strncpy(char* dest, const char* src, size_t n)
    // NID: 0xEAC256896491BAA9ULL ("6sJWiWSRuqk")
    register_symbol(0xEAC256896491BAA9ULL, "strncpy", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestAddr src  = ctx.rsi;
            const GuestSize len  = ctx.rdx;
            if (len > 0 && mem.is_valid_address(dest, len) && mem.is_valid_address(src, 1)) {
                bool null_seen = false;
                for (GuestSize i = 0; i < len; ++i) {
                    u8 b = 0;
                    if (!null_seen && mem.read_value(src + i, b)) {
                        if (b == 0) null_seen = true;
                    } else {
                        b = 0;
                    }
                    mem.write_value(dest + i, b);
                }
            }
            ctx.rax = dest;
        });

    // strncmp(const char* s1, const char* s2, size_t n)
    // NID: 0x69EB328EB1D55B2EULL ("aesyjrHVWy4")
    register_symbol(0x69EB328EB1D55B2EULL, "strncmp", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr s1_addr = ctx.rdi;
            const GuestAddr s2_addr = ctx.rsi;
            const GuestSize n = ctx.rdx;
            if (n == 0) {
                ctx.rax = 0;
                return;
            }
            const char* s1 = (s1_addr > 0x1000 && mem.is_valid_address(s1_addr, 1))
                                ? mem.to_host_ptr<char>(s1_addr) : nullptr;
            const char* s2 = (s2_addr > 0x1000 && mem.is_valid_address(s2_addr, 1))
                                ? mem.to_host_ptr<char>(s2_addr) : nullptr;
            if (!s1 || !s2) {
                ctx.rax = (s1 == s2) ? 0 : (s1 ? 1 : -1);
                return;
            }
            ctx.rax = static_cast<u64>(static_cast<i64>(std::strncmp(s1, s2, n)));
        });

    // strlen(const char* s)
    // NID: 0x8F856258D1C4830CULL ("j4ViWNHEgww")
    register_symbol(0x8F856258D1C4830CULL, "strlen", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr str = ctx.rdi;
            u64 len = 0;
            u8 ch = 0;
            while (mem.read_value(str + len, ch) && ch != 0 && len < 4096) {
                len++;
            }
            ctx.rax = len;
        });

    // strstr(const char* haystack, const char* needle)
    // NID: 0xBE28B014C68D6A60ULL ("viiwFMaNamA")
    register_symbol(0xBE28B014C68D6A60ULL, "strstr", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr hay = ctx.rdi;
            const GuestAddr needle = ctx.rsi;
            u8 n0 = 0;
            if (!mem.read_value(needle, n0) || n0 == 0) {
                ctx.rax = hay;
                return;
            }
            u64 hlen = 0;
            u8 ch = 0;
            while (mem.read_value(hay + hlen, ch) && ch != 0 && hlen < 4096) {
                if (ch == n0) {
                    bool match = true;
                    u64 nlen = 1;
                    while (true) {
                        u8 nch = 0;
                        if (!mem.read_value(needle + nlen, nch) || nch == 0) break;
                        u8 hch = 0;
                        if (!mem.read_value(hay + hlen + nlen, hch) || hch != nch) {
                            match = false;
                            break;
                        }
                        nlen++;
                    }
                    if (match) {
                        ctx.rax = hay + hlen;
                        return;
                    }
                }
                hlen++;
            }
            ctx.rax = 0;
        });

    // Real Orbis Semaphore synchronization for Unity worker threads
    // (table lives in the anonymous namespace so diagnostics can reach it)

    // sceKernelCreateSema(SceKernelSema* sema, const char* name, uint32_t attr, int initCount, int maxCount, void* opt)
    // NID: 0xD7CF31E7B258A748ULL ("188x57JYp0g")
    register_symbol(0xD7CF31E7B258A748ULL, "sceKernelCreateSema", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sema_ptr = ctx.rdi;
            const GuestAddr name_ptr = ctx.rsi;
            const u32 attr = static_cast<u32>(ctx.rdx);
            const i32 init_count = static_cast<i32>(ctx.rcx);
            const i32 max_count  = static_cast<i32>(ctx.r8) > 0 ? static_cast<i32>(ctx.r8) : 0x7fffffff;
            const char* name = (name_ptr > 0x1000 && mem.is_valid_address(name_ptr, 1))
                                   ? mem.to_host_ptr<char>(name_ptr) : nullptr;

            auto sema = std::make_shared<GuestSema>();
            sema->count = (init_count >= 0 && init_count <= 1024) ? init_count : 0;
            sema->max_count = max_count;

            u32 id = 0;
            {
                std::lock_guard<std::mutex> lock(s_sema_table_mtx);
                id = s_next_sema_id++;
                s_semas[id] = sema;
            }
            if (sema_ptr != 0 && mem.is_valid_address(sema_ptr, sizeof(u32))) {
                mem.write_value(sema_ptr, id);
            }
            std::fprintf(stderr, "[Sema] CreateSema -> id=%u name='%s' attr=0x%x init=%d max=%d\n",
                         id, name ? name : "?", attr, init_count, max_count);
            ctx.rax = SCE_OK;
        });

    // qsort(void* base, size_t nmemb, size_t size, int (*compar)(const void*, const void*))
    // NID: 0x00425D215653126AULL -- imported by eboot.bin and
    // libSceFaceTracker.prx. Was unregistered, so it fell through to the
    // unimplemented stub and logged five times a run.
    //
    // qsort returns void, so the stub's `rax = 0` was never a wrong *return
    // value* -- the damage is that the array is not sorted. The honest answer
    // is still not to sort it: compar is a guest function pointer, and calling
    // it would mean re-entering the guest CPU core from inside an HLE handler,
    // which is exactly what the RSP floor and the write-watch ring exist to
    // police against.
    //
    // So: register the call, log it once, leave the array in input order.
    // Input order is deterministic and reproducible -- the conservative no-op,
    // what a sort whose comparator reports every pair equal produces. The
    // alternative (guessing an order, or half-sorting) fabricates state the
    // guest cannot tell from real data, which is precisely the failure mode
    // this audit is removing elsewhere.
    register_symbol(0x00425D215653126AULL, "qsort", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            static std::atomic<bool> s_logged{false};
            if (!s_logged.exchange(true)) {
                std::fprintf(stderr,
                             "[qsort] base=0x%llx nmemb=%llu size=%llu compar=0x%llx"
                             " -- the comparator is guest code and cannot be invoked from HLE;"
                             " array left in input order\n",
                             (unsigned long long)ctx.rdi,
                             (unsigned long long)ctx.rsi,
                             (unsigned long long)ctx.rdx,
                             (unsigned long long)ctx.rcx);
            }
        });

    // sceKernelWaitSema(SceKernelSema sema, int needCount, SceKernelUseconds* pTimeout)
    // NID: 0x6716B45614154EC9ULL ("Zxa0VhQVTsk")
    register_symbol(0x6716B45614154EC9ULL, "sceKernelWaitSema", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const u32 id = static_cast<u32>(ctx.rdi);
            const i32 need = static_cast<i32>(ctx.rsi) > 0 ? static_cast<i32>(ctx.rsi) : 1;
            const GuestAddr timo_ptr = ctx.rdx;

            auto& tm = get_thread_manager(mem);
            GuestThread* cur_th = tm.current_thread();
            if (cur_th) cur_th->state.store(ThreadState::WaitingSync);

            std::shared_ptr<GuestSema> sema;
            {
                std::lock_guard<std::mutex> lock(s_sema_table_mtx);
                auto it = s_semas.find(id);
                if (it != s_semas.end()) sema = it->second;
            }
            if (!sema) {
                // Returning success here is a use-after-return generator. The
                // caller treats the job as complete and lets its frame die; a
                // worker that is still running the job then stores its result
                // through an out-pointer that lived in that dead frame, which
                // has by then been reused by another activation. Observed
                // exactly that: main's stack canary slot 0x1ffffac40 was
                // clobbered from TID 6 at guest RIP 0x105961c via a stale
                // `*obj->out = result` store.
                std::fprintf(stderr,
                             "[HLE WARNING TID=%d rip=0x%llx] sceKernelWaitSema(%u): unknown sema id; "
                             "returning ESRCH instead of pretending success\n", (int)t_guest_tid, (unsigned long long)t_guest_rip, id);
                if (cur_th) cur_th->state.store(ThreadState::Running);
                ctx.rax = 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }

            std::unique_lock<std::mutex> lock(sema->mtx);
            const u64 start_cancel_gen = sema->cancel_generation;
            auto pred = [&]() {
                return sema->count >= need || sema->deleted ||
                       (sema->cancel_generation != start_cancel_gen) ||
                       (g_syscalls && g_syscalls->has_exited());
            };
            sema->waiters++;

            if (timo_ptr != 0 && mem.is_valid_address(timo_ptr, sizeof(u32))) {
                u32 usec = 0;
                mem.read_value(timo_ptr, usec);
                std::fprintf(stderr, "[Sema] WaitSema(%u, need=%d, timeout=%uus)\n", id, need, usec);
                if (!sema->cv.wait_for(lock, std::chrono::microseconds(usec), pred)) {
                    sema->waiters--;
                    if (cur_th) cur_th->state.store(ThreadState::Running);
                    ctx.rax = 0x8002003CULL; // SCE_KERNEL_ERROR_ETIMEDOUT
                    return;
                }
            } else {
                std::fprintf(stderr, "[Sema] WaitSema(%u, need=%d, timeout=infinite) count=%u deleted=%d gen=%llu\n",
                             id, need, sema->count, sema->deleted ? 1 : 0,
                             (unsigned long long)start_cancel_gen);
                while (!pred()) {
                    sema->cv.wait_for(lock, std::chrono::milliseconds(50), pred);
                }
                // Diagnostic: did we return without the semaphore actually being
                // satisfied? That is a silent no-op wait -- the requester runs
                // on while its producer is still in flight.
                if (sema->count < need) {
                    std::fprintf(stderr,
                        "[HLE WARNING TID=%d rip=0x%llx] WaitSema(%u) returned with count=%u < need=%d "
                        "-- woke on deleted=%d cancelled=%d has_exited=%d, NOT on a post\n",
                        (int)t_guest_tid, (unsigned long long)t_guest_rip, id, sema->count, need,
                        sema->deleted ? 1 : 0,
                        (sema->cancel_generation != start_cancel_gen) ? 1 : 0,
                        (g_syscalls && g_syscalls->has_exited()) ? 1 : 0);
                }
            }

            sema->waiters--;
            if (sema->cancel_generation != start_cancel_gen) {
                if (cur_th) cur_th->state.store(ThreadState::Running);
                ctx.rax = 0x80020055ULL; // SCE_KERNEL_ERROR_ECANCELED (FreeBSD 85)
                return;
            }
            if (sema->deleted) {
                if (cur_th) cur_th->state.store(ThreadState::Running);
                ctx.rax = 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }
            if (sema->count >= need) {
                sema->count -= need;
            }
            if (cur_th) cur_th->state.store(ThreadState::Running);
            ctx.rax = SCE_OK;
        });

    // sceKernelSignalSema(SceKernelSema sema, int signalCount). Shared handler
    // also reused by the verified 0xE1CC... registration below.
    auto signal_sema_handler = [](CpuContext& ctx, MemoryManager&) {
        const u32 id = static_cast<u32>(ctx.rdi);
        const i32 sig = static_cast<i32>(ctx.rsi) > 0 ? static_cast<i32>(ctx.rsi) : 1;
        std::shared_ptr<GuestSema> sema;
        {
            std::lock_guard<std::mutex> lock(s_sema_table_mtx);
            auto it = s_semas.find(id);
            if (it != s_semas.end()) sema = it->second;
        }
        if (sema) {
            std::lock_guard<std::mutex> lock(sema->mtx);
            sema->count = std::min(sema->max_count, sema->count + sig);
            sema->cv.notify_all();
            std::fprintf(stderr, "[Sema] SignalSema(%u, %d) -> count=%d\n", id, sig, sema->count);
        } else {
            std::fprintf(stderr,
                         "[HLE WARNING TID=%d rip=0x%llx] sceKernelSignalSema(%u): unknown sema id "
                         "(never created, or already deleted)\n", (int)t_guest_tid, (unsigned long long)t_guest_rip, id);
            ctx.rax = 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH -- never success
            return;
        }
        ctx.rax = SCE_OK;
    };
    // Unknown NIDs with signal-like behavior (hash otherwise, never called in
    // Flappy; kept on the shared signal handler for compat, honestly labeled).
    register_symbol(0x443BB032833D9F45ULL, "unknown_443bb032", "libkernel", signal_sema_handler);
    register_symbol(0x3C4AC6259839E5EBULL, "unknown_3c4ac625", "libkernel", signal_sema_handler);

    // Non-blocking poll helper shared by the real PollSema and the
    // never-called unknown slot.
    auto poll_sema_impl = [](CpuContext& ctx, const char* tag) {
        const u32 id = static_cast<u32>(ctx.rdi);
        const i32 need = static_cast<i32>(ctx.rsi) > 0 ? static_cast<i32>(ctx.rsi) : 1;
        std::shared_ptr<GuestSema> sema;
        {
            std::lock_guard<std::mutex> lock(s_sema_table_mtx);
            auto it = s_semas.find(id);
            if (it != s_semas.end()) sema = it->second;
        }
        if (sema) {
            std::lock_guard<std::mutex> lock(sema->mtx);
            if (sema->count >= need) {
                sema->count -= need;
                ctx.rax = SCE_OK;
                return;
            }
            std::fprintf(stderr, "[Sema] %s(%u, need=%d) -> BUSY (count=%d)\n", tag, id, need,
                         sema->count);
        } else {
            std::fprintf(stderr,
                         "[HLE WARNING TID=%d rip=0x%llx] %s(%u): unknown sema id\n", (int)t_guest_tid, (unsigned long long)t_guest_rip, tag, id);
            ctx.rax = 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH -- never success
        }
        ctx.rax = 0x80020010ULL; // SCE_KERNEL_ERROR_EBUSY
    };
    // 0x0528... hashes otherwise (never called in Flappy); honestly labeled,
    // same non-blocking behavior for compat.
    register_symbol(0x0528B1D7D1C6C155ULL, "unknown_0528b1d7", "libkernel",
        [poll_sema_impl](CpuContext& ctx, MemoryManager&) { poll_sema_impl(ctx, "unknown_0528b1d7"); });

    // sceKernelPollSema(SceKernelSema sema, int needCount)
    // NID: 0xD76C0E1E4F32C1BDULL (hash-verified against Orbis salt).
    register_symbol(0xD76C0E1E4F32C1BDULL, "sceKernelPollSema", "libkernel",
        [poll_sema_impl](CpuContext& ctx, MemoryManager&) { poll_sema_impl(ctx, "PollSema"); });

    // sceKernelCancelSema(SceKernelSema sema, int setCount, int* pNumWaitingThreads)
    // NID: 0xE03334E94D813446ULL, confirmed by ps4libdoc libkernel.sprx.json
    // (and independently by the Orbis hash). Wakes active waiters (WaitSema
    // observes cancel_generation change and returns ECANCELED). If setCount >= 0,
    // updates the semaphore count. If pNumWaitingThreads != NULL, writes number
    // of cancelled waiters.
    register_symbol(0xE03334E94D813446ULL, "sceKernelCancelSema", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const u32 id = static_cast<u32>(ctx.rdi);
            const i32 set_count = static_cast<i32>(ctx.rsi);
            const GuestAddr out_ptr = ctx.rdx;
            std::shared_ptr<GuestSema> sema;
            {
                std::lock_guard<std::mutex> lock(s_sema_table_mtx);
                auto it = s_semas.find(id);
                if (it != s_semas.end()) sema = it->second;
            }
            if (!sema) {
                ctx.rax = 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }
            i32 waiters = 0;
            {
                std::lock_guard<std::mutex> lock(sema->mtx);
                waiters = sema->waiters;
                if (set_count >= 0) {
                    sema->count = std::min(sema->max_count, set_count);
                }
                if (waiters > 0) {
                    sema->cancel_generation++;
                    sema->cv.notify_all();
                }
            }
            if (out_ptr != 0 && mem.is_valid_address(out_ptr, sizeof(i32))) {
                mem.write_value(out_ptr, waiters);
            }
            std::fprintf(stderr, "[Sema] CancelSema(%u, setCount=%d) -> cancelled waiters=%d count=%d\n",
                         id, set_count, waiters, sema->count);
            ctx.rax = SCE_OK;
        });

    // sceKernelDeleteSema(SceKernelSema sema)
    // NID: 0x47526F9FC6D2096FULL ("R1Jvn8bSCW8")
    register_symbol(0x47526F9FC6D2096FULL, "sceKernelDeleteSema", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const u32 id = static_cast<u32>(ctx.rdi);
            std::shared_ptr<GuestSema> sema;
            {
                std::lock_guard<std::mutex> lock(s_sema_table_mtx);
                auto it = s_semas.find(id);
                if (it != s_semas.end()) {
                    sema = it->second;
                    s_semas.erase(it);
                }
            }
            if (sema) {
                std::lock_guard<std::mutex> lock(sema->mtx);
                sema->deleted = true;
                sema->cv.notify_all();
            }
            ctx.rax = SCE_OK;
        });

    // ---------------------------------------------------------------------
    // Real Orbis event flags. NIDs from ps4libdoc libkernel.sprx.json.
    // ---------------------------------------------------------------------

    // sceKernelCreateEventFlag(SceKernelEventFlag* evf, const char* name,
    //                          uint32_t attr, uint32_t bitPattern, void* opt)
    // NID: 0x0691686E8509A195ULL
    register_symbol(0x0691686E8509A195ULL, "sceKernelCreateEventFlag", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr evf_ptr = ctx.rdi;
            const GuestAddr name_ptr = ctx.rsi;
            const u32 attr = static_cast<u32>(ctx.rdx);
            const u32 bits = static_cast<u32>(ctx.rcx);
            const char* name = (name_ptr > 0x1000 && mem.is_valid_address(name_ptr, 1))
                                   ? mem.to_host_ptr<char>(name_ptr) : nullptr;

            auto evf = std::make_shared<GuestEventFlag>();
            evf->bits = bits;
            u32 id = 0;
            {
                std::lock_guard<std::mutex> lock(s_eflag_table_mtx);
                id = s_next_eflag_id++;
                s_eflags[id] = evf;
            }
            if (evf_ptr != 0 && mem.is_valid_address(evf_ptr, sizeof(u32))) {
                mem.write_value(evf_ptr, id);
            }
            std::fprintf(stderr, "[EventFlag] Create -> id=%u name='%s' attr=0x%x bits=0x%x\n",
                         id, name ? name : "?", attr, bits);
            ctx.rax = SCE_OK;
        });

    // sceKernelDeleteEventFlag(SceKernelEventFlag evf)
    // NID: 0xF26AA5F4E7109DDEULL
    register_symbol(0xF26AA5F4E7109DDEULL, "sceKernelDeleteEventFlag", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const u32 id = static_cast<u32>(ctx.rdi);
            std::shared_ptr<GuestEventFlag> evf;
            {
                std::lock_guard<std::mutex> lock(s_eflag_table_mtx);
                auto it = s_eflags.find(id);
                if (it != s_eflags.end()) {
                    evf = it->second;
                    s_eflags.erase(it);
                }
            }
            if (evf) {
                std::lock_guard<std::mutex> lock(evf->mtx);
                evf->deleted = true;
                evf->cv.notify_all();
            }
            ctx.rax = SCE_OK;
        });

    // sceKernelSetEventFlag(SceKernelEventFlag evf, uint32_t bitPattern)
    // NID: 0x20E9D2BC7CEABBA0ULL
    register_symbol(0x20E9D2BC7CEABBA0ULL, "sceKernelSetEventFlag", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const u32 id = static_cast<u32>(ctx.rdi);
            const u32 bits = static_cast<u32>(ctx.rsi);
            std::shared_ptr<GuestEventFlag> evf;
            {
                std::lock_guard<std::mutex> lock(s_eflag_table_mtx);
                auto it = s_eflags.find(id);
                if (it != s_eflags.end()) evf = it->second;
            }
            if (!evf) {
                std::fprintf(stderr, "[EventFlag] Set(%u, 0x%x) -> unknown id\n", id, bits);
                ctx.rax = 0x80020001ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }
            {
                std::lock_guard<std::mutex> lock(evf->mtx);
                evf->bits |= bits;
            }
            evf->cv.notify_all();
            std::fprintf(stderr, "[EventFlag] Set(%u, 0x%x)\n", id, bits);
            ctx.rax = SCE_OK;
        });

    // sceKernelClearEventFlag(SceKernelEventFlag evf, uint32_t bitPattern)
    // NID: 0xEEE8411564404BADULL
    register_symbol(0xEEE8411564404BADULL, "sceKernelClearEventFlag", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const u32 id = static_cast<u32>(ctx.rdi);
            const u32 bits = static_cast<u32>(ctx.rsi);
            std::shared_ptr<GuestEventFlag> evf;
            {
                std::lock_guard<std::mutex> lock(s_eflag_table_mtx);
                auto it = s_eflags.find(id);
                if (it != s_eflags.end()) evf = it->second;
            }
            if (!evf) {
                ctx.rax = 0x80020001ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }
            std::lock_guard<std::mutex> lock(evf->mtx);
            evf->bits &= ~bits;
            ctx.rax = SCE_OK;
        });

    // Shared match test for Poll/Wait. waitMode bit0 selects AND-match over
    // OR-match; Orbis callers pass 0x22 here, whose low bit is clear, so the
    // observed Mono path is treated as OR-match.
    auto eventflag_satisfied = [](const GuestEventFlag& evf, u32 want, bool and_mode) {
        return and_mode ? ((evf.bits & want) == want) : ((evf.bits & want) != 0);
    };

    // sceKernelPollEventFlag(SceKernelEventFlag evf, uint32_t bitPattern,
    //                        uint32_t* pBitPattern)
    // NID: 0xF65BE3E438C76620ULL
    register_symbol(0xF65BE3E438C76620ULL, "sceKernelPollEventFlag", "libkernel",
        [eventflag_satisfied](CpuContext& ctx, MemoryManager& mem) {
            const u32 id = static_cast<u32>(ctx.rdi);
            const u32 want = static_cast<u32>(ctx.rsi);
            const GuestAddr out_ptr = ctx.rdx;
            const bool and_mode = (static_cast<u32>(ctx.rcx) & 1u) != 0;
            std::shared_ptr<GuestEventFlag> evf;
            {
                std::lock_guard<std::mutex> lock(s_eflag_table_mtx);
                auto it = s_eflags.find(id);
                if (it != s_eflags.end()) evf = it->second;
            }
            if (!evf) {
                ctx.rax = 0x80020001ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }
            std::lock_guard<std::mutex> lock(evf->mtx);
            if (out_ptr != 0 && mem.is_valid_address(out_ptr, sizeof(u32))) {
                mem.write_value(out_ptr, evf->bits);
            }
            ctx.rax = eventflag_satisfied(*evf, want, and_mode) ? SCE_OK
                                                                : 0x80020010ULL; // EBUSY
        });

    // sceKernelWaitEventFlag(SceKernelEventFlag evf, uint32_t bitPattern,
    //                        uint32_t waitMode, uint32_t* pBitPattern,
    //                        SceKernelUseconds* pTimeout)
    // NID: 0x253BC17E58586B34ULL -- ps4libdoc: sceKernelWaitEventFlag.
    // Imported by eboot.bin, mono-ps4.prx and libSceJobManager.prx; it used to
    // fall through to the generic unimplemented stub. The Flappy trace shows
    // the Mono runtime calling it with evf=NULL (rdi=0x0), which must return an
    // error rather than block forever.
    //
    // Unity's worker pool creates 'Unity EventFlag' objects and parks six
    // worker threads in here. Nothing in this runtime ever calls
    // sceKernelSetEventFlag, so an unbounded wait strands the whole pool with
    // no producer to wake it. When the guest passes no timeout we therefore cap
    // the wait and report ETIMEDOUT: the guest re-polls, the host thread is
    // still yielded, and no thread is parked on a flag nobody can set.
    //
    // The cap is 1 ms. A 20 ms cap was measured and is a REGRESSION, not an
    // optimisation: it looks like it would cut worker churn, but over a 60 s
    // run the guest spawned 33 threads instead of 12 and main never executed a
    // single instruction (RIP=0x0, hles=0), with UnityGfxDeviceWorker down at
    // 4.4 M instructions against 112 M at 1 ms. Longer blocking hands the host
    // scheduler to the worker pool instead of the thread that drives the game.
    // Measured: 1 ms -> 410 k HLE calls, main hles=195 k, TID6=112 M;
    //          20 ms -> 84 k HLE calls, main hles=0,    TID6=4.4 M.
    constexpr u64 kEventFlagDefaultWaitUsec = 1000;
    register_symbol(0x253BC17E58586B34ULL, "sceKernelWaitEventFlag", "libkernel",
        [eventflag_satisfied](CpuContext& ctx, MemoryManager& mem) {
            const u32 id = static_cast<u32>(ctx.rdi);
            const u32 want = static_cast<u32>(ctx.rsi);
            const u32 wait_mode = static_cast<u32>(ctx.rdx);
            const GuestAddr out_ptr = ctx.rcx;
            const GuestAddr timo_ptr = ctx.r8;
            const bool and_mode = (wait_mode & 1u) != 0;
            const bool clear_on_match = (wait_mode & 2u) != 0;

            std::shared_ptr<GuestEventFlag> evf;
            {
                std::lock_guard<std::mutex> lock(s_eflag_table_mtx);
                auto it = s_eflags.find(id);
                if (it != s_eflags.end()) evf = it->second;
            }
            if (!evf) {
                std::fprintf(stderr,
                    "[EventFlag] Wait(%u, want=0x%x, mode=0x%x) -> no such flag\n",
                    id, want, wait_mode);
                ctx.rax = 0x80020001ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }

            auto& tm = get_thread_manager(mem);
            GuestThread* cur_th = tm.current_thread();
            if (cur_th) cur_th->state.store(ThreadState::WaitingSync);

            u32 usec = kEventFlagDefaultWaitUsec;
            const bool guest_timeout = timo_ptr != 0 && mem.is_valid_address(timo_ptr, sizeof(u32));
            if (guest_timeout) mem.read_value(timo_ptr, usec);

            u32 matched = 0;
            bool ok = false;
            {
                std::unique_lock<std::mutex> lock(evf->mtx);
                auto pred = [&]() {
                    return evf->deleted || eventflag_satisfied(*evf, want, and_mode) ||
                           (g_syscalls && g_syscalls->has_exited());
                };
                evf->waiters++;
                // An untimed wait must actually block. Capping it at 1 ms and
                // returning ETIMEDOUT lets a requester proceed while the
                // producer it is waiting on has not run -- the same
                // use-after-return shape as an unknown-id WaitSema returning
                // success. The 2 s shutdown watchdog in NidRegistry::shutdown()
                // now makes parking safe, so the cap is only a shutdown poll.
                if (guest_timeout) {
                    ok = evf->cv.wait_for(lock, std::chrono::microseconds(usec), pred);
                } else {
                    evf->cv.wait(lock, pred);
                    ok = true;
                }
                evf->waiters--;
                if (ok) {
                    matched = evf->bits & want;
                    if (clear_on_match) evf->bits &= ~want;
                }
            }

            if (out_ptr != 0 && mem.is_valid_address(out_ptr, sizeof(u32))) {
                mem.write_value(out_ptr, matched);
            }
            if (cur_th) cur_th->state.store(ThreadState::Running);
            if (!ok) {
                std::fprintf(stderr, "[EventFlag] Wait(%u, want=0x%x, mode=0x%x) -> ETIMEDOUT after %uus%s\n",
                             id, want, wait_mode, usec,
                             guest_timeout ? " (guest timeout)" : " (no producer: capped)");
                ctx.rax = 0x8002003CULL; // SCE_KERNEL_ERROR_ETIMEDOUT
                return;
            }
            std::fprintf(stderr, "[EventFlag] Wait(%u, want=0x%x, mode=0x%x) -> matched 0x%x\n",
                         id, want, wait_mode, matched);
            ctx.rax = SCE_OK;
        });

    // sceKernelIsNeoMode()
    // NID: 0x5AC95C2B51507062ULL ("WslcK1FQcGI")
    register_symbol(0x5AC95C2B51507062ULL, "sceKernelIsNeoMode", "libkernel",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = 0; // Standard PS4 Base mode
        });

    // scePthreadMutexInit NID: 0x726A3544862F6BDAULL ("cmo1RIYva9o")
    register_symbol(0x726A3544862F6BDAULL, "scePthreadMutexInit", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            const u32 mid = tm.mutex_create();
            if (mtx_ptr != 0 && mem.is_valid_address(mtx_ptr, sizeof(u32))) {
                mem.write_value(mtx_ptr, mid);
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexattrInit NID: 0x17C6D41F0006DBCEULL ("F8bUHwAG284")
    register_symbol(0x17C6D41F0006DBCEULL, "scePthreadMutexattrInit", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexattrSettype NID: 0x88CA7C42913E5CEEULL ("iMp8QpE+XO4")
    register_symbol(0x88CA7C42913E5CEEULL, "scePthreadMutexattrSettype", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexattrSetprotocol NID: 0xD451AF5348BDB1A4ULL ("1FGvU0i9saQ")
    register_symbol(0xD451AF5348BDB1A4ULL, "scePthreadMutexattrSetprotocol", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexattrDestroy NID: 0xB2658492D8B2C86DULL ("smWEktiyyG0")
    register_symbol(0xB2658492D8B2C86DULL, "scePthreadMutexattrDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // ---------------------------------------------------------------------
    // libScePosix family (NIDs from ps4libdoc libkernel.sprx.json, which
    // exports these names from both libkernel and libScePosix).
    // ---------------------------------------------------------------------

    // pthread_mutexattr_init(ScePthreadMutexattr* attr)
    // NID: 0x7501D612C26DA04EULL -- ps4libdoc: pthread_mutexattr_init
    // (libkernel + libScePosix). This NID was registered as "scePadInit",
    // which produced the bogus "scePadInit x42" loop: Unity calls this once
    // per worker thread with a stack-resident attr (rdi on the main stack,
    // rsi/rdx stale 0x10). The real scePadInit is 0x86FD65BA226BA903ULL.
    register_symbol(0x7501D612C26DA04EULL, "pthread_mutexattr_init", "libScePosix",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr attr_ptr = ctx.rdi;
            // Orbis pthread_mutexattr_t is a 4-byte type field; zeroing only
            // that keeps neighbouring stack slots (saved RBP, padding) intact.
            if (attr_ptr != 0 && mem.is_valid_address(attr_ptr, sizeof(u32))) {
                mem.zero_memory(attr_ptr, sizeof(u32));
            }
            ctx.rax = SCE_OK;
        });

    // pthread_create(ScePthread*, const pthread_attr_t*, void* (*)(void*), void*)
    // NID: 0x3B184807C2C1FCF4ULL -- ps4libdoc: pthread_create. This NID is
    // imported by eboot.bin AND mono-ps4.prx but was registered as
    // "scePadSetVibration" (a SCE_OK stub), so every plain-POSIX thread
    // creation handed the guest a NULL thread id. Same body as the
    // scePthreadCreate handler above.
    register_symbol(0x3B184807C2C1FCF4ULL, "pthread_create", "libScePosix", pthread_create_handler);

    // sem_init(sem_t* sem, int pshared, unsigned int value)
    // NID: 0xA43B8F11FDE6E1F2ULL -- ps4libdoc: sem_init. Imported by eboot.bin
    // and mono-ps4.prx, and previously registered as "scePadReadState", so
    // Mono's sem_init calls were overwriting the guest sem_t with 32 bytes of
    // pad button/stick state -- a live corruption of every semaphore.
    register_symbol(0xA43B8F11FDE6E1F2ULL, "sem_init", "libScePosix",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sem_ptr = ctx.rdi;
            const u32 value = static_cast<u32>(ctx.rdx);
            if (sem_ptr != 0 && mem.is_valid_address(sem_ptr, sizeof(u32) * 4)) {
                mem.zero_memory(sem_ptr, sizeof(u32) * 4);
                mem.write_value(sem_ptr, value);
            }
            ctx.rax = SCE_OK;
        });

    // ---------------------------------------------------------------------
    // POSIX named semaphores. Guest struct: { u32 count; u32 waiters; u32
    // signalled; u32 pad; } at the caller's pointer, mirroring sem_init above.
    // Before these existed sem_wait fell through to the generic stub and never
    // blocked, so producers and consumers ran free of each other.
    // ---------------------------------------------------------------------
    auto sem_read = [](MemoryManager& mem, GuestAddr ptr, std::array<u32, 4>& out) -> bool {
        if (ptr == 0 || !mem.is_valid_address(ptr, sizeof(u32) * 4)) return false;
        return mem.read_value(ptr, out[0]) && mem.read_value(ptr + 4, out[1]) &&
               mem.read_value(ptr + 8, out[2]);
    };
    auto sem_write = [](MemoryManager& mem, GuestAddr ptr, const std::array<u32, 4>& v) {
        if (ptr == 0 || !mem.is_valid_address(ptr, sizeof(u32) * 4)) return;
        mem.write_value(ptr, v[0]);
        mem.write_value(ptr + 4, v[1]);
        mem.write_value(ptr + 8, v[2]);
    };

    // A single condition variable guards every POSIX semaphore. It is only a
    // parking point: the authoritative count still lives in guest memory at
    // sem[0], so waking is always a re-read, never a cached decision. Declared
    // ahead of sem_post so the poster can signal under the same lock the waiter
    // checks under, which closes the missed-wakeup window.
    // Reuse the file-scope condition variable above; sem_post below signals it.
    // (The mutex/CV are declared at namespace scope so shutdown() can reach them.)

    // sem_post(sem_t* sem)
    // NID: 0x20A3FCB72A744149ULL (libkernel + libScePosix per ps4libdoc)
    register_symbol(0x20A3FCB72A744149ULL, "sem_post", "libScePosix",
        [sem_read, sem_write](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sem_ptr = ctx.rdi;
            bool ok = false;
            std::array<u32, 4> sem{};
            {
                std::lock_guard<std::mutex> lock(s_sem_cv_mtx);
                if (sem_read(mem, sem_ptr, sem)) {
                    sem[0] += 1;
                    sem[2] += 1;
                    sem_write(mem, sem_ptr, sem);
                    ok = true;
                }
            }
            if (!ok) {
                ctx.rax = static_cast<u64>(-22); // EINVAL
                return;
            }
            s_sem_cv.notify_all();
            std::fprintf(stderr, "[Sem] post(0x%llx) -> count=%u waiters=%u\n",
                         (unsigned long long)sem_ptr, sem[0], sem[1]);
            ctx.rax = SCE_OK;
        });

    // sem_trywait(sem_t* sem) -- non-blocking decrement.
    // NID: 0x5815B3B1189F0840ULL
    register_symbol(0x5815B3B1189F0840ULL, "sem_trywait", "libScePosix",
        [sem_read, sem_write](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sem_ptr = ctx.rdi;
            std::array<u32, 4> sem{};
            if (!sem_read(mem, sem_ptr, sem)) {
                ctx.rax = static_cast<u64>(-22);
                return;
            }
            if (sem[0] == 0) {
                sem[2] += 1; // EAGAIN
                ctx.rax = static_cast<u64>(-35);
                return;
            }
            sem[0] -= 1;
            sem_write(mem, sem_ptr, sem);
            ctx.rax = SCE_OK;
        });

    // sem_wait(sem_t* sem)
    // NID: 0x602579746181702AULL
    //
    // POSIX sem_wait is an unconditional wait: it returns only when the
    // semaphore is posted (or on EINTR). It must NOT return a synthetic
    // ETIMEDOUT to the guest, and this previously did -- 500 iterations of
    // sleep_for(1ms) then ETIMEDOUT, with no producer ever in sight.
    //
    // Unity's thread pool is sensitive to exactly that. The worker body at
    // 0x975c6-0x97656 pops from the job queue at [r15+0x28], and when the
    // queue is empty does:
    //     0x97640  movq 0x70(%r15), %rdi   ; shared "worker idle" sem
    //     0x97649  callq sem_post          ; announce idle
    //     0x9764e  movq %r14, %rdi         ; r14 = r15+0x50, own wake sem
    //     0x97651  callq sem_wait
    //     0x97656  cmpl $0x0, 0x60(%r15)   ; pool run flag -- NOT eax
    //     0x9765b  jne  0x975c6
    // It ignores the return value entirely, so every spurious timeout made an
    // idle worker wake, find an empty queue, and sem_post the shared semaphore
    // again. Measured in a 60 s run: 282 posts to 0x151ee70 with zero jobs
    // executed, and 94 x 500 ms of blocking per worker -- ~47 s of wall clock
    // per worker, which is where the entire run went (39.4M instructions, about
    // 5.6 s of compute).
    //
    // Park on the condition variable and poll only for shutdown. wait_for is
    // used rather than wait so an emulator-wide exit still unblocks promptly;
    // the 20 ms period is a shutdown poll, not a guest-visible timeout.
    register_symbol(0x602579746181702AULL, "sem_wait", "libScePosix",
        [sem_read, sem_write](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sem_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            GuestThread* cur_th = tm.current_thread();
            if (cur_th) cur_th->state.store(ThreadState::WaitingSync);
            bool ok = false;
            bool valid = true;
            {
                std::unique_lock<std::mutex> lock(s_sem_cv_mtx);
                while (true) {
                    std::array<u32, 4> sem{};
                    if (!sem_read(mem, sem_ptr, sem)) {
                        valid = false;
                        break;
                    }
                    if (sem[0] > 0) {
                        sem[0] -= 1;
                        sem_write(mem, sem_ptr, sem);
                        ok = true;
                        break;
                    }
                    if (g_syscalls && g_syscalls->has_exited()) break;
                    if (s_sem_shutdown.load(std::memory_order_relaxed)) break;
                    s_sem_cv.wait_for(lock, std::chrono::milliseconds(20));
                }
            }
            if (cur_th) cur_th->state.store(ThreadState::Running);
            if (!valid) {
                std::fprintf(stderr, "[Sem] wait(0x%llx) -> EINVAL\n",
                             (unsigned long long)sem_ptr);
                ctx.rax = static_cast<u64>(-22);
                return;
            }
            std::fprintf(stderr, "[Sem] wait(0x%llx) -> %s\n", (unsigned long long)sem_ptr,
                         ok ? "OK" : "shutdown");
            ctx.rax = ok ? SCE_OK : static_cast<u64>(0x8002003CULL); // SCE_KERNEL_ERROR_ETIMEDOUT
        });

    // sem_timedwait(sem_t* sem, const struct timespec* abs_timeout)
    // NID: 0xC39207CAF6A183FAULL
    register_symbol(0xC39207CAF6A183FAULL, "sem_timedwait", "libScePosix",
        [sem_read, sem_write](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sem_ptr = ctx.rdi;
            const GuestAddr abs_ptr = ctx.rsi;
            std::array<u32, 4> sem{};
            if (!sem_read(mem, sem_ptr, sem)) {
                ctx.rax = static_cast<u64>(-22);
                return;
            }
            if (sem[0] > 0) {
                sem[0] -= 1;
                sem_write(mem, sem_ptr, sem);
                ctx.rax = SCE_OK;
                return;
            }
            // Orbis timespec is 64-bit seconds + 32-bit nanoseconds.
            u64 abs_sec = 0;
            u32 abs_nsec = 0;
            if (abs_ptr != 0 && mem.is_valid_address(abs_ptr, sizeof(u64) + sizeof(u32))) {
                mem.read_value(abs_ptr, abs_sec);
                mem.read_value(abs_ptr + 8, abs_nsec);
            }
            const u64 now = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count());
            const u64 deadline = abs_sec != 0 ? abs_sec : now + 1;
            const u64 wait_us = deadline > now ? (deadline - now) * 1000000ULL + abs_nsec / 1000ULL
                                               : 0ULL;
            auto& tm = get_thread_manager(mem);
            GuestThread* cur_th = tm.current_thread();
            if (cur_th) cur_th->state.store(ThreadState::WaitingSync);
            bool ok = false;
            const u64 deadline_us = now * 1000000ULL + wait_us;
            while (!ok && (g_syscalls == nullptr || !g_syscalls->has_exited())) {
                const u64 t = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count());
                if (t >= deadline_us) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (!sem_read(mem, sem_ptr, sem)) break;
                if (sem[0] > 0) {
                    sem[0] -= 1;
                    sem_write(mem, sem_ptr, sem);
                    ok = true;
                }
            }
            if (cur_th) cur_th->state.store(ThreadState::Running);
            // Orbis errno values are FreeBSD's, not Linux's. FreeBSD ETIMEDOUT is 60;
// 110 is the Linux value and was wrong here. (FreeBSD EAGAIN is 35 and
            // Linux's is 11 -- sem_trywait above already returns the correct
            // FreeBSD 35.) A guest comparing against errno sees a mismatch.
            ctx.rax = ok ? SCE_OK : static_cast<u64>(-60); // ETIMEDOUT (FreeBSD/Orbis)
        });

    // sem_destroy(sem_t* sem)
    // NID: 0x7035B6DF7440C16AULL
    register_symbol(0x7035B6DF7440C16AULL, "sem_destroy", "libScePosix",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sem_ptr = ctx.rdi;
            if (sem_ptr != 0 && mem.is_valid_address(sem_ptr, sizeof(u32) * 4)) {
                mem.zero_memory(sem_ptr, sizeof(u32) * 4);
            }
            ctx.rax = SCE_OK;
        });

    // sem_getvalue(sem_t* sem, int* sval)
    // NID: 0x06AF8B455FCDE879ULL
    register_symbol(0x06AF8B455FCDE879ULL, "sem_getvalue", "libScePosix",
        [sem_read](CpuContext& ctx, MemoryManager& mem) {
            std::array<u32, 4> sem{};
            if (!sem_read(mem, ctx.rdi, sem)) {
                ctx.rax = static_cast<u64>(-22);
                return;
            }
            if (ctx.rsi != 0 && mem.is_valid_address(ctx.rsi, sizeof(i32))) {
                mem.write_value(ctx.rsi, static_cast<i32>(sem[0]));
            }
            ctx.rax = SCE_OK;
        });


    // memalign(size_t alignment, size_t size)
    // NID: 0x5237F72B332F4662ULL ("Ujf3KzMvRmI")
    register_symbol(0x5237F72B332F4662ULL, "memalign", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const GuestSize alignment = ctx.rdi > 0 ? ctx.rdi : 16;
            ctx.rax = bump_alloc(ctx.rsi, alignment);
        });

    // sceLibcMspaceMemalign(mspace msp, size_t boundary, size_t size)
    // NID: 0x885D62407CF10495ULL -- ps4libdoc calls this sceLibcMspaceMemalign
    // (libSceLibcInternal). Unity routes its GPU/Garlic allocations through it
    // (53 calls, ~935 MB in one 60 s run), so it draws from the dedicated
    // mspace arena rather than the CPU heap -- see mspace_alloc.
    register_symbol(0x885D62407CF10495ULL, "sceLibcMspaceMemalign", "libSceLibcInternal",
        [](CpuContext& ctx, MemoryManager&) {
            const GuestSize alignment = ctx.rsi > 0 ? ctx.rsi : 16;
            ctx.rax = mspace_alloc(ctx.rdx, alignment);
        });

    // memcpy(void* dest, const void* src, size_t n)
    // NID: 0x437541C425E1507BULL ("Q3VBxCXhUHs")
    register_symbol(0x437541C425E1507BULL, "memcpy", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestAddr src  = ctx.rsi;
            const GuestSize n    = ctx.rdx;
            if (n > 0 && mem.is_valid_address(dest, n) && mem.is_valid_address(src, n)) {
                const void* s_ptr = mem.to_host_ptr<void>(src);
                void* d_ptr = mem.to_host_ptr<void>(dest);
                if (s_ptr && d_ptr) {
                    std::memcpy(d_ptr, s_ptr, n);
                }
            }
            ctx.rax = dest;
        });

    // memmove(void* dest, const void* src, size_t n)
    // NID: 0xF8FE854461F82DF0ULL ("+P6FRGH4LfA")
    register_symbol(0xF8FE854461F82DF0ULL, "memmove", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestAddr src  = ctx.rsi;
            const GuestSize n    = ctx.rdx;
            if (n > 0 && mem.is_valid_address(dest, n) && mem.is_valid_address(src, n)) {
                std::vector<u8> tmp(n);
                mem.read_bytes(src, tmp.data(), n);
                mem.write_bytes(dest, tmp.data(), n);
            }
            ctx.rax = dest;
        });

    // time(time_t* tloc)
    // NID: 0xC0B9459301BD51C4ULL ("wLlFkwG9UcQ")
    register_symbol(0xC0B9459301BD51C4ULL, "time", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr tloc = ctx.rdi;
            const u64 t = static_cast<u64>(std::time(nullptr));
            if (tloc != 0 && mem.is_valid_address(tloc, sizeof(u64))) {
                mem.write_value(tloc, t);
            }
            ctx.rax = t;
        });

    // __cxa_atexit(void (*func)(void*), void* arg, void* dso_handle)
    // NID: 0xB6CBC49A77A7CF8FULL ("tsvEmnenz48") -- imported by eboot.bin,
    // libSceHeadTracker.prx and libSceAudioLatencyEstimation.prx.
    //
    // rdi is a *guest* function pointer. Registering 0 (success) and never
    // invoking func is the only answer available: an HLE handler cannot call
    // back into guest code, and a half-dialled "run the destructors" would
    // have to re-enter the CPU core from inside a handler. The consequence is
    // documented rather than discovered: guest static destructors never run, so
    // nothing in the guest may depend on at-exit side effects (flushing a
    // save file, releasing a GPU buffer) before the process is torn down.
    // exit()/__cxa_finalize below therefore terminate rather than attempt a
    // sweep. That is a deliberate trade: the alternative -- returning an error
    // -- makes __cxa_atexit's return a lie the guest checks, and every caller
    // of it is generated code with no error path.
    register_symbol(0xB6CBC49A77A7CF8FULL, "__cxa_atexit", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0; // Success; the handler itself is never invoked.
        });

    // __cxa_finalize(void* dso_handle) -- NID: 0x1F67BCB7949C4067, imported by
    // libSceAudioLatencyEstimation.prx, libSceFace.prx and libSceFaceTracker.prx.
    // No-op for the same reason as __cxa_atexit: it exists to run the
    // handlers that registration stashed, and the handlers are guest code.
    // Registered under its real ps4libdoc name so the trace says what happened
    // instead of printing "Unimplemented stub called ... returning 0".
    register_symbol(0x1F67BCB7949C4067ULL, "__cxa_finalize", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0; // Success; no handlers to run (see __cxa_atexit).
        });

    // strcmp(const char* s1, const char* s2)
    // NID: 0x3AF6F675224E02E1ULL ("Ovb2dSJOAuE")
    register_symbol(0x3AF6F675224E02E1ULL, "strcmp", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr s1 = ctx.rdi;
            const GuestAddr s2 = ctx.rsi;
            if (s1 == s2) {
                ctx.rax = 0;
                return;
            }
            if (!mem.is_valid_address(s1, 1) || !mem.is_valid_address(s2, 1)) {
                ctx.rax = (s1 < s2) ? static_cast<u64>(-1) : 1;
                return;
            }
            u64 idx = 0;
            i32 diff = 0;
            while (idx < 4096) {
                u8 c1 = 0, c2 = 0;
                if (!mem.read_value(s1 + idx, c1) || !mem.read_value(s2 + idx, c2)) {
                    diff = (s1 < s2) ? -1 : 1;
                    break;
                }
                if (c1 != c2) {
                    diff = static_cast<i32>(c1) - static_cast<i32>(c2);
                    break;
                }
                if (c1 == 0) break;
                idx++;
            }
            ctx.rax = static_cast<u64>(static_cast<i64>(diff));
        });

    // __cxa_guard_acquire(int64_t* guard)
    // NID: 0xDC63E98D0740313CULL ("3GPpjQdAMTw")
    register_symbol(0xDC63E98D0740313CULL, "__cxa_guard_acquire", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr guard_ptr = ctx.rdi;
            u8 val = 0;
            if (guard_ptr != 0 && mem.read_value(guard_ptr, val)) {
                if (val == 0) {
                    mem.write_value(guard_ptr, static_cast<u8>(1));
                    ctx.rax = 1; // Needs initialization
                    return;
                }
            }
            ctx.rax = 0; // Already initialized
        });

    // __cxa_guard_release(int64_t* guard)
    // NID: 0xF6B01E00D4F6B721ULL ("9rAeANT2tyE")
    register_symbol(0xF6B01E00D4F6B721ULL, "__cxa_guard_release", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr guard_ptr = ctx.rdi;
            if (guard_ptr != 0 && mem.is_valid_address(guard_ptr, 1)) {
                mem.write_value(guard_ptr, static_cast<u8>(1));
            }
            ctx.rax = 0;
        });

    // =========================================================================
    // libaudioout: PlayStation 4 Audio Output Services
    // =========================================================================

    // sceAudioOutOpen(SceUserServiceUserId userId, int32_t type, int32_t index, uint32_t paramLen, const void* param)
    register_symbol(0xE1D5176E5161A4FEULL, "sceAudioOutOpen", "libaudioout",
        [](CpuContext& ctx, MemoryManager&) {
            const i32 type = static_cast<i32>(ctx.rsi);
            auto& am = get_audio_manager();
            const i32 handle = am.open_port(type);
            ctx.rax = static_cast<u64>(handle);
        });

    // sceAudioOutClose(int32_t handle)
    register_symbol(0xEAA1CDA949437E55ULL, "sceAudioOutClose", "libaudioout",
        [](CpuContext& ctx, MemoryManager&) {
            const i32 handle = static_cast<i32>(ctx.rdi);
            auto& am = get_audio_manager();
            ctx.rax = am.close_port(handle) ? SCE_OK : static_cast<u64>(-1);
        });

    // sceAudioOutOutput(int32_t handle, const void* ptr)
    register_symbol(0x69661413865EE60CULL, "sceAudioOutOutput", "libaudioout",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 handle = static_cast<i32>(ctx.rdi);
            const GuestAddr ptr = ctx.rsi;
            auto& am = get_audio_manager();

            constexpr usize SAMPLE_COUNT = 512 * 2; // 512 stereo frames
            constexpr usize BYTE_SIZE = SAMPLE_COUNT * sizeof(i16);

            if (ptr != 0 && mem.is_valid_address(ptr, BYTE_SIZE)) {
                const auto* pcm = mem.to_host_ptr<i16>(ptr);
                if (pcm) {
                    std::span<const i16> samples(pcm, SAMPLE_COUNT);
                    am.output_samples(handle, samples);
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    ctx.rax = SCE_OK;
                    return;
                }
            }
            ctx.rax = static_cast<u64>(-14);
        });

    // sceAudioOutSetVolume(int32_t handle, int32_t flag, int32_t* vol)
    register_symbol(0x8F596D3C1F4FA286ULL, "sceAudioOutSetVolume", "libaudioout",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 handle = static_cast<i32>(ctx.rdi);
            const GuestAddr vol_ptr = ctx.rdx;
            auto& am = get_audio_manager();

            if (vol_ptr != 0 && mem.is_valid_address(vol_ptr, 8)) {
                i32 vol_l = 0, vol_r = 0;
                mem.read_value(vol_ptr, vol_l);
                mem.read_value(vol_ptr + 4, vol_r);
                const f32 left = static_cast<f32>(vol_l) / static_cast<f32>(SCE_AUDIO_OUT_VOLUME_MAX);
                const f32 right = static_cast<f32>(vol_r) / static_cast<f32>(SCE_AUDIO_OUT_VOLUME_MAX);
                am.set_volume(handle, left, right);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // sceAudioOutGetPortInfo(int32_t handle, SceAudioOutPortInfo* info)
    register_symbol(0xD9BDDE84CE9A1E02ULL, "sceAudioOutGetPortInfo", "libaudioout",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 handle = static_cast<i32>(ctx.rdi);
            const GuestAddr info_ptr = ctx.rsi;
            auto& am = get_audio_manager();

            if (info_ptr != 0 && mem.is_valid_address(info_ptr, sizeof(SceAudioOutPortInfo))) {
                SceAudioOutPortInfo info{};
                if (am.get_port_info(handle, info)) {
                    mem.write_bytes(info_ptr, &info, sizeof(info));
                    ctx.rax = SCE_OK;
                    return;
                }
            }
            ctx.rax = static_cast<u64>(-1);
        });

    // =========================================================================
    // libsave_data: PlayStation 4 Save Data Services
    // =========================================================================

    // sceSaveDataMount(const SceSaveDataMount* mount, SceSaveDataMountResult* result)
    register_symbol(0x948296238BC4F301ULL, "sceSaveDataMount", "libsave_data",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mount_ptr = ctx.rdi;
            const GuestAddr result_ptr = ctx.rsi;
            auto& vfs = get_vfs();

            if (mount_ptr != 0 && mem.is_valid_address(mount_ptr, sizeof(SceSaveDataMount)) &&
                result_ptr != 0 && mem.is_valid_address(result_ptr, sizeof(SceSaveDataMountResult))) {
                SceSaveDataMount mount_req{};
                mem.read_bytes(mount_ptr, &mount_req, sizeof(mount_req));

                SceSaveDataMountResult mount_res{};
                const i32 ret = vfs.mount_savedata(mount_req.user_id, mount_req.dir_name.data, mount_req.mount_mode, mount_res);
                if (ret == SCE_OK) {
                    mem.write_bytes(result_ptr, &mount_res, sizeof(mount_res));
                    ctx.rax = SCE_OK;
                    return;
                }
            }
            ctx.rax = static_cast<u64>(-14);
        });

    // sceSaveDataUmount(const SceSaveDataMountResult* result)
    register_symbol(0x32791B317075AB6EULL, "sceSaveDataUmount", "libsave_data",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr result_ptr = ctx.rdi;
            auto& vfs = get_vfs();

            if (result_ptr != 0 && mem.is_valid_address(result_ptr, sizeof(SceSaveDataMountResult))) {
                SceSaveDataMountResult mount_res{};
                mem.read_bytes(result_ptr, &mount_res, sizeof(mount_res));
                const i32 ret = vfs.unmount_savedata(mount_res.mount_path);
                ctx.rax = (ret == SCE_OK) ? SCE_OK : static_cast<u64>(-1);
                return;
            }
            ctx.rax = static_cast<u64>(-14);
        });

    // =========================================================================
    // libsave_data: the memory-budget pair, and libSceVideoOut /
    // libSceSocialScreen entries that were falling through to the generic
    // unimplemented stub. Each of these is registered under the name
    // ps4libdoc's 42k corpus gives it (known_names.txt), so the log now states
    // what the guest actually called rather than "Unimplemented stub called:
    // NID=0x... returning 0". Four were firing on every run, at the call sites
    // cited below.
    // =========================================================================

    // sceSaveDataSetupSaveDataMemory2(const void* param)
    // NID: 0xA10C921147E05D10ULL -- imported by eboot.bin. Called from
    // 0x10acf7e (guest `callq` whose return address is 0x10acf83) by the
    // worker at TID 4, with rdi = 0x1cff9fea0 (a heap param struct).
    //
    // Guest code at the call site (eboot.elf, vaddr == file offset):
    //   10acf7e: callq <import>
    //   10acf83: movl  %eax, %ecx
    //   10acf85: testl %ecx, %ecx
    //   10acf87: je    0x10acfb2            ; success -> continue
    //   10acf89: xorl  %eax, %eax
    //   10acf8d: callq 0x10ad840            ; error reporter
    // It is an int-returning status call with an *input* struct; the success
    // path reads nothing the callee was meant to fill in (it only pages in its
    // own pointers at 0x10acfb2+). Returning 0 is therefore correct here and is
    // kept: the memory "setup" is a budget reservation we do not need to
    // model, and the guest's success path is the healthy one.
    // The first call dumps the param struct so the layout can be decoded
    // later instead of guessed at now.
    register_symbol(0xA10C921147E05D10ULL, "sceSaveDataSetupSaveDataMemory2", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            static std::atomic<bool> s_logged{false};
            const GuestAddr p = ctx.rdi;
            const bool dump = !s_logged.exchange(true);
            if (dump && p > 0x1000 && mem.is_valid_address(p, 64)) {
                const u8* b = mem.to_host_ptr<u8>(p);
                if (b) {
                    std::fprintf(stderr, "[SaveData] SetupSaveDataMemory2 rdi=0x%llx struct: %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x\n",
                                 (unsigned long long)p, b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7], b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15],
                                 b[16],b[17],b[18],b[19],b[20],b[21],b[22],b[23], b[24],b[25],b[26],b[27],b[28],b[29],b[30],b[31],
                                 b[32],b[33],b[34],b[35],b[36],b[37],b[38],b[39], b[40],b[41],b[42],b[43],b[44],b[45],b[46],b[47],
                                 b[48],b[49],b[50],b[51],b[52],b[53],b[54],b[55], b[56],b[57],b[58],b[59],b[60],b[61],b[62],b[63]);
                }
            }
            ctx.rax = SCE_OK;
        });

    // sceSaveDataGetSaveDataMemory2(const void* param, ...)
    // NID: 0x43038EEEF7A09D5FULL -- imported by eboot.bin. Called from
    // 0x10ad06a, immediate sibling of the setup call above (same worker, same
    // init sequence), rdi = 0x1cff9fee0.
    //
    //   10ad06a: callq <import>
    //   10ad06f: movl  %eax, %ecx
    //   10ad071: testl %ecx, %ecx
    //   10ad073: je    0x10ad07e            ; success -> continue
    //   10ad075: xorl  %eax, %eax
    //   10ad079: callq 0x10ad840            ; error reporter
    // The success path (0x10ad07e+) fills its *own* stack buffer and hands it
    // to 0x10ad8f0; nothing in it dereferences a callee-written output. So this
    // is an int status call too, and 0 is the honest answer.
    register_symbol(0x43038EEEF7A09D5FULL, "sceSaveDataGetSaveDataMemory2", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceVideoOutGetDeviceCapabilityInfo_(int handle, void* info, size_t size)
    // NID: 0x90654B73786D404FULL ("YIvXmjmhRBs") -- imported by eboot.bin.
    // Called from 0x10a7b74, return address 0x10a7b79, by TID 1.
    //
    //   10a7b58: leaq  -0x170(%rbp), %rsi    ; info  = caller stack buffer
    //   10a7b5f: movl  $0x8, %edx            ; size  = 8
    //   10a7b6e: movl  0x298(%rax), %edi     ; handle = 0x100
    //   10a7b74: callq <import>
    //   10a7b79: movl  %eax, %ecx
    //   10a7b7b: testl %ecx, %ecx
    //   10a7b7d: je    0x10a7b88             ; success -> continue
    //   10a7b7f: xorl  %eax, %eax
    //   10a7b83: callq 0x7eb9a0              ; error reporter
    //
    // This one DOES have an output struct, and it is the shape of bug worth
    // avoiding: the stub returned 0 (success) without writing anything, so the
    // guest read whatever the stack slot happened to hold at [rbp-0x170] as a
    // capability value. It tests byte 0 of that struct immediately:
    //   10a7c0b: cmpb $0x0, -0x170(%rbp)
    //   10a7c12: js   0x10a7c2e
    //
    // The output is implemented rather than error-returned: zero the caller's
    // buffer (exactly `size` bytes, so the size argument stays authoritative)
    // and answer SCE_OK. "No capabilities" is the truthful answer -- we do not
    // model a video-out capability set -- and it is deterministic, which a
    // never-written stack slot is not. An error here would be a lie dressed as
    // a diagnostic: the device does exist.
    register_symbol(0x90654B73786D404FULL, "sceVideoOutGetDeviceCapabilityInfo_", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr info_ptr = ctx.rsi;
            const GuestSize size = ctx.rdx;
            if (info_ptr > 0x1000 && size <= 0x1000 && mem.is_valid_address(info_ptr, size)) {
                mem.zero_memory(info_ptr, size);
                ctx.rax = SCE_OK;
                return;
            }
            ctx.rax = static_cast<u64>(-14); // -EFAULT
        });

    // sceSocialScreenInitialize(int mode) -- NID: 0xA48EE81523CFEB90ULL,
    // imported by eboot.bin. Called from 0x1065f20, return address 0x1065f25.
    //
    //   1065f16: movl $0xf2, %edi            ; mode = 0xf2
    //   1065f20: callq <import>
    //   1065f25: movl $0xff, %edi            ; args for the next import
    //   1065f2a: movl $0x5, %esi
    //   1065f33: callq 0x11f7f60
    //   1065f38: movl  %eax, %ecx
    //   1065f3a: testl %ecx, %ecx
    //   1065f3c: js    0x106621a             ; only *this* call's result is checked
    // No output struct, and the return value of the social-screen import is
    // discarded -- the guest re-derives its status from the next call. Answer
    // SCE_OK; the SocialScreen family (Terminate/OpenSeparateMode/
    // CloseSeparateMode/SetMode) is not modelled, so "initialized" here means
    // only "do not block the call chain".
    register_symbol(0xA48EE81523CFEB90ULL, "sceSocialScreenInitialize", "libSceSocialScreen",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // ==========================================
    // libSceFios2 - Fast I/O System 2
    // ==========================================

    // sceFiosFileExistsSync / sceFiosFileExists(int profile, const char* path, ...)
    // Canonical Sync NID: 0x3703873113363E9CULL ("NwOHMRM2Ppw"), Non-sync alias NID: 0x398F2FF160F39F7BULL ("OY8v8WDzn3s")
    auto fios_file_exists_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr path_addr = ctx.rsi;
        const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                               ? mem.to_host_ptr<char>(path_addr) : nullptr;
        if (!path) {
            ctx.rax = 0;
            return;
        }
        auto& vfs = get_vfs();
        auto resolved = vfs.resolve_path(path);
        bool exists = false;
        if (resolved) {
            std::error_code ec;
            exists = std::filesystem::is_regular_file(*resolved, ec);
        }
        std::fprintf(stderr, "[FIOS2] sceFiosFileExists('%s') -> %d\n", path, exists ? 1 : 0);
        ctx.rax = exists ? 1 : 0;
    };
    register_symbol(0x398F2FF160F39F7BULL, "sceFiosFileExists", "libSceFios2", fios_file_exists_handler);
    register_symbol(0x3703873113363E9CULL, "sceFiosFileExistsSync", "libSceFios2", fios_file_exists_handler);

    // sceFiosFHRead / sceFiosFHReadSync(int profile, int fh, void* buf, size_t size)
    // NID: 0x2A5FD36EB0D4F583ULL ("Kl-TbrDU9YM")
    // Registered below, after sceFiosFHOpenSync, using the handle-based
    // s_fios_files read implementation (see the sceFiosFHOpenSync handler above it).
    // sceFiosDirectoryExistsSync / sceFiosDirectoryExists(int profile, const char* path, ...)
    // Canonical Sync NID: 0x38EBAF1CA4EEE0E7ULL ("OOuvHKTu4Oc"), Non-sync alias NID: 0xBED1DDD347F43B9DULL ("vtHd00f0O50")
    auto fios_dir_exists_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr path_addr = ctx.rsi;
        const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                               ? mem.to_host_ptr<char>(path_addr) : nullptr;
        if (!path) {
            ctx.rax = 0;
            return;
        }
        auto& vfs = get_vfs();
        auto resolved = vfs.resolve_path(path);
        bool exists = false;
        if (resolved) {
            std::error_code ec;
            exists = std::filesystem::is_directory(*resolved, ec);
        }
        std::fprintf(stderr, "[FIOS2] sceFiosDirectoryExists('%s') -> %d\n", path, exists ? 1 : 0);
        ctx.rax = exists ? 1 : 0;
    };
    register_symbol(0xBED1DDD347F43B9DULL, "sceFiosDirectoryExists", "libSceFios2", fios_dir_exists_handler);
    register_symbol(0x38EBAF1CA4EEE0E7ULL, "sceFiosDirectoryExistsSync", "libSceFios2", fios_dir_exists_handler);

    // sceFiosStatSync / sceFiosStat(int profile, const char* path, SceFiosStat* stat)
    // Canonical Sync NID: 0x8DACAF634EC2E5D9ULL ("jayvY07C5dk"), Non-sync alias NID: 0x40AB08F4DECAD731ULL ("QKsI9N7K1zE")
    auto fios_stat_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr path_addr = ctx.rsi;
        const GuestAddr stat_ptr  = ctx.rdx;
        const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                               ? mem.to_host_ptr<char>(path_addr) : nullptr;
        if (!path || stat_ptr == 0 || !mem.is_valid_address(stat_ptr, 0x40)) {
            ctx.rax = static_cast<u64>(-1);
            return;
        }
        auto& vfs = get_vfs();
        auto resolved = vfs.resolve_path(path);
        if (!resolved) {
            std::fprintf(stderr, "[FIOS2] sceFiosStat('%s') -> not found\n", path);
            ctx.rax = static_cast<u64>(-1);
            return;
        }
        std::error_code ec;
        const auto st = std::filesystem::status(*resolved, ec);
        if (!std::filesystem::exists(st)) {
            std::fprintf(stderr, "[FIOS2] sceFiosStat('%s' -> '%s') -> not found\n", path, resolved->c_str());
            ctx.rax = static_cast<u64>(-1);
            return;
        }

        mem.zero_memory(stat_ptr, 0x40);
        const bool is_dir = std::filesystem::is_directory(st);
        const u64 file_sz = is_dir ? 0 : std::filesystem::file_size(*resolved, ec);
        const u32 stat_flags = is_dir ? 0x1 : 0x0;

        mem.write_value(stat_ptr + 0x00, file_sz);
        mem.write_value(stat_ptr + 0x20, stat_flags);

        std::fprintf(stderr, "[FIOS2] sceFiosStat('%s') -> size=%llu, flags=0x%x\n",
                     path, (unsigned long long)file_sz, stat_flags);
        ctx.rax = SCE_OK;
    };
    register_symbol(0x40AB08F4DECAD731ULL, "sceFiosStat", "libSceFios2", fios_stat_handler);
    register_symbol(0x8DACAF634EC2E5D9ULL, "sceFiosStatSync", "libSceFios2", fios_stat_handler);

    // sceFiosFileGetSizeSync / sceFiosGetSize(int profile, const char* path, ...)
    // NID: 0xCC5F3F091BD15E73ULL ("zF8-CRvRXnM")
    register_symbol(0xCC5F3F091BD15E73ULL, "sceFiosFileGetSizeSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                std::fprintf(stderr, "[FIOS2] sceFiosFileGetSizeSync('%s') -> not found, returning -1\n", path);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::error_code ec;
            if (!std::filesystem::is_regular_file(*resolved, ec)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFileGetSizeSync('%s' -> '%s') -> not regular file\n", path, resolved->c_str());
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const u64 sz = std::filesystem::file_size(*resolved, ec);
            if (ec) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::fprintf(stderr, "[FIOS2] sceFiosFileGetSizeSync('%s' -> '%s') -> size=%llu\n",
                         path, resolved->c_str(), (unsigned long long)sz);
            ctx.rax = sz;
        });

    // sceFiosFHOpenSync(int profile, SceFiosFH* pFH, const char* path, const SceFiosOpenParams* pParams)
    // NID: 0x6F8E1A9D5D83ECADULL ("b44anV2D7K0")
    register_symbol(0x6F8E1A9D5D83ECADULL, "sceFiosFHOpenSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr pfh_addr = ctx.rsi;
            const GuestAddr path_addr = ctx.rdx;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHOpenSync('%s') -> not found, returning -1\n", path);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::error_code ec;
            if (!std::filesystem::is_regular_file(*resolved, ec)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHOpenSync('%s' -> '%s') -> not regular file\n", path, resolved->c_str());
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto stream = std::make_unique<std::ifstream>(*resolved, std::ios::binary);
            if (!stream->is_open()) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHOpenSync('%s' -> '%s') -> open failed\n", path, resolved->c_str());
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const u64 sz = std::filesystem::file_size(*resolved, ec);
            const i32 fd = s_next_fios_fd++;
            s_fios_files[fd] = FiosFile{
                .stream = std::move(stream),
                .size = sz,
                .offset = 0,
                .path = path
            };
            if (pfh_addr != 0 && mem.is_valid_address(pfh_addr, sizeof(i32))) {
                mem.write_value(pfh_addr, fd);
            }
            std::fprintf(stderr, "[FIOS2] sceFiosFHOpenSync('%s' -> '%s') -> handle %d (size=%llu) written to 0x%llx\n",
                         path, resolved->c_str(), fd, (unsigned long long)sz, (unsigned long long)pfh_addr);
            ctx.rax = SCE_OK;
        });

    // sceFiosFHRead / sceFiosFHReadSync(int profile, int fh, void* buf, size_t size)
    // NID 0x2A5FD36EB0D4F583ULL hashes to sceFiosFHWriteSync. Body below is the
    // legacy read-shaped impl (fail-safes to -1 via address gates; never called
    // in traces so far). Real write path TBD.
    // NID: 0x2A5FD36EB0D4F583ULL ("Kl-TbrDU9YM")
    register_symbol(0x2A5FD36EB0D4F583ULL, "sceFiosFHWriteSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 fd = static_cast<i32>(ctx.rsi);
            const GuestSize size = ctx.rdx;
            const GuestAddr buf_addr = ctx.rcx;
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end() || !it->second.stream || !it->second.stream->is_open()) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHRead: invalid handle %d\n", fd);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (buf_addr == 0 || !mem.is_valid_address(buf_addr, size)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHRead: invalid buffer 0x%llx (size=%llu)\n",
                             (unsigned long long)buf_addr, (unsigned long long)size);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            u8* dest = mem.to_host_ptr<u8>(buf_addr);
            it->second.stream->seekg(it->second.offset, std::ios::beg);
            it->second.stream->read(reinterpret_cast<char*>(dest), size);
            const std::streamsize bytes_read = it->second.stream->gcount();
            it->second.offset += static_cast<u64>(bytes_read);
            std::fprintf(stderr, "[FIOS2] sceFiosFHRead(fd=%d, size=%llu) -> read %lld bytes\n",
                         fd, (unsigned long long)size, (long long)bytes_read);
            ctx.rax = static_cast<u64>(bytes_read);
        });

    // sceFiosFHPread(int profile, int fh, void* buf, size_t size, size_t offset)
    // NID 0x80C71F3AD1D6EB39ULL hashes to sceFiosFHPwriteSync. Body below is the
    // legacy pread-shaped impl (fail-safes to -1; never called). Real pwrite TBD.
    // NID: 0x80C71F3AD1D6EB39ULL ("gMcfOtHW6zk")
    register_symbol(0x80C71F3AD1D6EB39ULL, "sceFiosFHPwriteSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 fd = static_cast<i32>(ctx.rsi);
            const GuestAddr buf_addr = ctx.rdx;
            const GuestSize size = ctx.rcx;
            const u64 offset = ctx.r8;
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end() || !it->second.stream || !it->second.stream->is_open()) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHPread: invalid handle %d\n", fd);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (buf_addr == 0 || !mem.is_valid_address(buf_addr, size)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHPread: invalid buffer 0x%llx (size=%llu)\n",
                             (unsigned long long)buf_addr, (unsigned long long)size);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            u8* dest = mem.to_host_ptr<u8>(buf_addr);
            it->second.stream->seekg(offset, std::ios::beg);
            it->second.stream->read(reinterpret_cast<char*>(dest), size);
            const std::streamsize bytes_read = it->second.stream->gcount();
            std::fprintf(stderr, "[FIOS2] sceFiosFHPread(fd=%d, offset=%llu, size=%llu) -> read %lld bytes\n",
                         fd, (unsigned long long)offset, (unsigned long long)size, (long long)bytes_read);
            ctx.rax = static_cast<u64>(bytes_read);
        });

    // sceFiosFHPreadSync(int profile, int fh, void* buf, size_t size, size_t offset)
    // NID: 0xDA6F7E3A9728FE19ULL ("2m9+Opco-hk")
    register_symbol(0xDA6F7E3A9728FE19ULL, "sceFiosFHPreadSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 fd = static_cast<i32>(ctx.rsi);
            const GuestAddr buf_addr = ctx.rdx;
            const GuestSize size = ctx.rcx;
            const u64 offset = ctx.r8;
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end() || !it->second.stream || !it->second.stream->is_open()) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHPreadSync: invalid handle %d\n", fd);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (buf_addr == 0 || !mem.is_valid_address(buf_addr, size)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHPreadSync: invalid buffer 0x%llx (size=%llu)\n",
                             (unsigned long long)buf_addr, (unsigned long long)size);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            u8* dest = mem.to_host_ptr<u8>(buf_addr);
            it->second.stream->seekg(offset, std::ios::beg);
            it->second.stream->read(reinterpret_cast<char*>(dest), size);
            const std::streamsize bytes_read = it->second.stream->gcount();
            std::fprintf(stderr, "[FIOS2] sceFiosFHPreadSync(fd=%d, offset=%llu, size=%llu) -> read %lld bytes\n",
                         fd, (unsigned long long)offset, (unsigned long long)size, (long long)bytes_read);
            ctx.rax = static_cast<u64>(bytes_read);
        });

    // sceFiosFHReadSync(int profile, int fh, void* buf, size_t size)
    // NID: 0x067D991786637AE4ULL — the real blocking read; consumes the live offset.
    // Arg positions mirror FHPreadSync (fd=rsi, buf=rdx, size=rcx).
    register_symbol(0x067D991786637AE4ULL, "sceFiosFHReadSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 fd = static_cast<i32>(ctx.rsi);
            const GuestAddr buf_addr = ctx.rdx;
            const GuestSize size = ctx.rcx;
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end() || !it->second.stream || !it->second.stream->is_open()) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHReadSync: invalid handle %d\n", fd);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (buf_addr == 0 || !mem.is_valid_address(buf_addr, size)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHReadSync: invalid buffer 0x%llx (size=%llu)\n",
                             (unsigned long long)buf_addr, (unsigned long long)size);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            u8* dest = mem.to_host_ptr<u8>(buf_addr);
            it->second.stream->seekg(static_cast<std::streamoff>(it->second.offset), std::ios::beg);
            it->second.stream->read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(size));
            const std::streamsize bytes_read = it->second.stream->gcount();
            it->second.offset += static_cast<u64>(bytes_read);
            std::fprintf(stderr, "[FIOS2] sceFiosFHReadSync(fd=%d, size=%llu) -> read %lld bytes\n",
                         fd, (unsigned long long)size, (long long)bytes_read);
            ctx.rax = static_cast<u64>(bytes_read);
        });

    // sceFiosFHTell(int fh) -> current file offset
    // NID: 0x32B445ADD829B31FULL ("MrRFrdgpsx8")
    register_symbol(0x32B445ADD829B31FULL, "sceFiosFHTell", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            const i32 fd = static_cast<i32>(ctx.rdi);
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end()) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::fprintf(stderr, "[FIOS2] sceFiosFHTell(fd=%d) -> %llu\n", fd, (unsigned long long)it->second.offset);
            ctx.rax = it->second.offset;
        });

    // sceFiosFHSeek(int fh, int64_t offset, int whence)
    // NID: 0xC5179279BC0A0290ULL ("xReSebwKApA")
    register_symbol(0xC5179279BC0A0290ULL, "sceFiosFHSeek", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            const i32 fd = static_cast<i32>(ctx.rdi);
            const i64 offset = static_cast<i64>(ctx.rsi);
            const i32 whence = static_cast<i32>(ctx.rdx);
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end()) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (whence == 0) { // SEEK_SET
                it->second.offset = static_cast<u64>(offset >= 0 ? offset : 0);
            } else if (whence == 1) { // SEEK_CUR
                it->second.offset = static_cast<u64>(std::max<i64>(0, static_cast<i64>(it->second.offset) + offset));
            } else if (whence == 2) { // SEEK_END
                it->second.offset = static_cast<u64>(std::max<i64>(0, static_cast<i64>(it->second.size) + offset));
            }
            std::fprintf(stderr, "[FIOS2] sceFiosFHSeek(fd=%d, offset=%lld, whence=%d) -> pos %llu\n",
                         fd, (long long)offset, whence, (unsigned long long)it->second.offset);
            ctx.rax = it->second.offset;
        });

    // sceFiosFHCloseSync / sceFiosFHClose(int profile, int fh)
    // Canonical Sync NID: 0x00EBA3486A94FA6BULL ("AOujSGqU+ms"), Non-sync alias NID: 0xE6C60D04D2BE5B78ULL ("5sYNBNK+W3g")
    auto fios_fh_close_handler = [](CpuContext& ctx, MemoryManager& /*mem*/) {
        const i32 fd = static_cast<i32>(ctx.rsi);
        s_fios_files.erase(fd);
        std::fprintf(stderr, "[FIOS2] sceFiosFHClose(fd=%d) -> closed\n", fd);
        ctx.rax = SCE_OK;
    };
    register_symbol(0xE6C60D04D2BE5B78ULL, "sceFiosFHClose", "libSceFios2", fios_fh_close_handler);
    register_symbol(0x00EBA3486A94FA6BULL, "sceFiosFHCloseSync", "libSceFios2", fios_fh_close_handler);

    // sceFiosDirectoryCreateSync(int profile, const char* path)
    // NID: 0x9D6BB36B465D7EA0ULL ("nWuza0ZdfqA")
    register_symbol(0x9D6BB36B465D7EA0ULL, "sceFiosDirectoryCreateSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::error_code ec;
            std::filesystem::create_directories(*resolved, ec);
            std::fprintf(stderr, "[FIOS2] sceFiosDirectoryCreateSync('%s') -> %d\n", path, ec ? -1 : 0);
            ctx.rax = ec ? static_cast<u64>(-1) : 0;
        });

    // sceFiosFileDeleteSync / sceFiosFileDelete(int profile, const char* path)
    // Canonical Sync NID: 0x6C3BA91206D0E859ULL ("bDupEgbQ6Fk"), Non-sync alias NID: 0x550F1EE6DF2FB588ULL ("VQ8e5t8vtYg")
    auto fios_file_delete_handler = [](CpuContext& ctx, MemoryManager& /*mem*/) {
        ctx.rax = SCE_OK;
    };
    register_symbol(0x550F1EE6DF2FB588ULL, "sceFiosFileDelete", "libSceFios2", fios_file_delete_handler);
    register_symbol(0x6C3BA91206D0E859ULL, "sceFiosFileDeleteSync", "libSceFios2", fios_file_delete_handler);

    // sceFiosArchiveGetMountBufferSizeSync(int profile, const char* path)
    // NID: 0x514AE2697CBB1BDDULL ("UUriaXy7G90")
    // Body returns the resolved file size, which has satisfied both observed calls.
    register_symbol(0x514AE2697CBB1BDDULL, "sceFiosArchiveGetMountBufferSizeSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::error_code ec;
            const u64 sz = std::filesystem::file_size(*resolved, ec);
            ctx.rax = ec ? static_cast<u64>(-1) : sz;
        });

    // sceFiosArchiveMountSync(int profile, const char* path, ...)
    // NID: 0xC6EB4B6D076AC9BEULL ("xutLbQdqyb4")
    register_symbol(0xC6EB4B6D076AC9BEULL, "sceFiosArchiveMountSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = static_cast<u64>(-1);
        });

    // sceFiosFileRead(int profile, const char* path, void* buf, size_t size, ...)
    // NID: 0x62528291F24BF98FULL ("YlKCkfJL+Y8")
    register_symbol(0x62528291F24BF98FULL, "sceFiosFileRead", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const GuestAddr buf_addr  = ctx.rdx;
            const GuestSize size      = ctx.rcx;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path || buf_addr == 0 || !mem.is_valid_address(buf_addr, size)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::ifstream file(*resolved, std::ios::binary);
            if (!file.is_open()) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            file.read(reinterpret_cast<char*>(mem.to_host_ptr<u8>(buf_addr)), size);
            ctx.rax = static_cast<u64>(file.gcount());
        });

    // sceFiosInitialize(const void* params)
    // NID: 0xC00299FDD7ADFB2AULL ("wAKZ-det+yo")
    register_symbol(0xC00299FDD7ADFB2AULL, "sceFiosInitialize", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            std::fprintf(stderr, "[FIOS2] sceFiosInitialize() -> SCE_OK\n");
            ctx.rax = SCE_OK;
        });

    // Archive stubs, relabelled to their ps4libdoc names. All three NIDs are
    // imported by eboot.bin; the bodies are unchanged fail-safe stubs.
    //   0xF8546F2A49D48F52 -> sceFiosOpGetActualCount   (returns -1: no ops)
    //   0x186AAE70BF45F982 -> sceFiosDHOpenSync         (returns -1)
    //   0xB8292024EAD850BE -> sceFiosDHOpen             (returns buffer size)
    register_symbol(0xF8546F2A49D48F52ULL, "sceFiosOpGetActualCount", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = static_cast<u64>(-1);
        });
    register_symbol(0x186AAE70BF45F982ULL, "sceFiosDHOpenSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = static_cast<u64>(-1);
        });
    register_symbol(0xB8292024EAD850BEULL, "sceFiosDHOpen", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = 0x10000;
        });

    // =========================================================================
    // libSceVideoOut: Display and Frame Presentation
    // =========================================================================
    static std::vector<GuestAddr> s_registered_buffers;


    // sceVideoOutRegisterBuffers(int handle, int startIndex, void* const* addresses, int bufferCount, ...)
    auto reg_buffers_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr addrs_ptr = ctx.rdx;
        const u32 count = static_cast<u32>(ctx.rcx ? ctx.rcx : 2);
        if (addrs_ptr != 0 && mem.is_valid_address(addrs_ptr, count * sizeof(GuestAddr))) {
            s_registered_buffers.clear();
            for (u32 i = 0; i < count; ++i) {
                GuestAddr buf_addr = 0;
                mem.read_value(addrs_ptr + i * sizeof(GuestAddr), buf_addr);
                s_registered_buffers.push_back(buf_addr);
                std::fprintf(stderr, "[VideoOut] Registered framebuffer #%u at guest 0x%llx\n", i, (unsigned long long)buf_addr);
            }
        }
        ctx.rax = SCE_OK;
    };

    // NID: 0xC37058FAD0048906ULL ("w3BY+tAEiQY")
    register_symbol(0xC37058FAD0048906ULL, "sceVideoOutRegisterBuffers", "libSceVideoOut", reg_buffers_handler);

    // sceVideoOutConfigureOutputMode_ / N1bEoJ4SRw4
    // Dumps the mode struct (rdx) so the layout filled by sceVideoOutModeSetAny_
    // can be decoded from the log.
    register_symbol(0x3756C4A09E12470EULL, "sceVideoOutConfigureOutputMode_", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mode_ptr = ctx.rdx;
            if (mode_ptr > 0x1000 && mem.is_valid_address(mode_ptr, 64)) {
                const u8* p = mem.to_host_ptr<u8>(mode_ptr);
                if (p) {
                    std::fprintf(stderr, "[VideoOut] ConfigureOutputMode_ mode struct at 0x%llx:\n",
                                 (unsigned long long)mode_ptr);
                    for (usize i = 0; i < 64; i += 16) {
                        std::fprintf(stderr, "  +%02zx: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                                     i, p[i], p[i+1], p[i+2], p[i+3], p[i+4], p[i+5], p[i+6], p[i+7],
                                     p[i+8], p[i+9], p[i+10], p[i+11], p[i+12], p[i+13], p[i+14], p[i+15]);
                    }
                }
            }
            ctx.rax = SCE_OK;
        });

    // sceAjmInitialize / dl+4eHSzUu4 (module #X#Y libSceAjm)
    register_symbol(0x765FB87874B352EEULL, "sceAjmInitialize", "libSceAjm",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceVideoOutOpen
    // NID: 0x529DFA3D393AF3B1ULL ("Up36PTk687E")
    auto video_open_handler = [](CpuContext& ctx, MemoryManager&) {
        ctx.rax = 0x100; // valid video out handle
        std::fprintf(stderr, "[VideoOut] sceVideoOutOpen -> handle 0x100\n");
    };
    register_symbol(0x529DFA3D393AF3B1ULL, "sceVideoOutOpen", "libSceVideoOut", video_open_handler);

    // sceVideoOutGetResolutionStatus
    auto res_status_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr status_ptr = ctx.rsi;
        if (status_ptr != 0 && mem.is_valid_address(status_ptr, 16)) {
            mem.write_value(status_ptr + 0, static_cast<u32>(1920)); // width
            mem.write_value(status_ptr + 4, static_cast<u32>(1080)); // height
            mem.write_value(status_ptr + 8, static_cast<u32>(60));   // fps
        }
        ctx.rax = SCE_OK;
    };
    // NID: 0xEA43E78F9D53EB66ULL ("6kPnj51T62Y")
    register_symbol(0xEA43E78F9D53EB66ULL, "sceVideoOutGetResolutionStatus", "libSceVideoOut", res_status_handler);

    // sceVideoOutAddFlipEvent
    // NID: 0x1D7CE32BDC88DF49ULL ("HXzjK9yI30k")
    register_symbol(0x1D7CE32BDC88DF49ULL, "sceVideoOutAddFlipEvent", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceVideoOutSetBufferAttribute
    // NID: 0x8BAFEC47DD56B7FEULL ("i6-sR91Wt-4")
    // Logs all regs plus a guarded dump of the rsi struct so format/tiling/pitch
    // layout can be decoded from the log.
    register_symbol(0x8BAFEC47DD56B7FEULL, "sceVideoOutSetBufferAttribute", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::fprintf(stderr, "[VideoOut] SetBufferAttribute(rdi=0x%llx, rsi=0x%llx, rdx=0x%llx, rcx=0x%llx, r8=0x%llx, r9=0x%llx)\n",
                         (unsigned long long)ctx.rdi, (unsigned long long)ctx.rsi,
                         (unsigned long long)ctx.rdx, (unsigned long long)ctx.rcx,
                         (unsigned long long)ctx.r8, (unsigned long long)ctx.r9);
            const GuestAddr p_addr = ctx.rsi;
            if (p_addr > 0x1000 && mem.is_valid_address(p_addr, 32)) {
                const u8* p = mem.to_host_ptr<u8>(p_addr);
                if (p) {
                    std::fprintf(stderr, "  rsi struct: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                                 p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],
                                 p[16],p[17],p[18],p[19],p[20],p[21],p[22],p[23],p[24],p[25],p[26],p[27],p[28],p[29],p[30],p[31]);
                }
            }
            ctx.rax = SCE_OK;
        });

    // sceVideoOutSubmitFlip(int handle, int bufferIndex, unsigned int flipMode, int64_t flipArg)
    // NID: 0x538E8DC0E889A72BULL ("U46NwOiJpys")
    register_symbol(0x538E8DC0E889A72BULL, "sceVideoOutSubmitFlip", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            const u32 buf_idx = static_cast<u32>(ctx.rsi);
            if (g_metal_backend && buf_idx < s_registered_buffers.size()) {
                const GuestAddr buf_addr = s_registered_buffers[buf_idx];
                const usize pitch = g_metal_backend->width() * 4;
                const usize frame_size = pitch * g_metal_backend->height();
                if (buf_addr != 0 && mem.is_valid_address(buf_addr, frame_size)) {
                    void* pixels = mem.to_host_ptr<void>(buf_addr);
                    if (pixels) {
                        g_metal_backend->upload_pixels(pixels, pitch);
                    }
                }
            }
            get_thread_manager(mem).signal_flip_event(static_cast<i64>(ctx.rcx));
            ctx.rax = SCE_OK;
        });

    // sceVideoOutSetFlipRate(int handle, int rate)
    // NID: 0x0818AEE26084D430ULL ("CBiu4mCE1DA")
    register_symbol(0x0818AEE26084D430ULL, "sceVideoOutSetFlipRate", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceSystemServiceParamInitialize / HcurEgsfF80
    // NID: 0x1DCBAB120B1F17CDULL ("HcurEgsfF80")
    register_symbol(0x1DCBAB120B1F17CDULL, "sceSystemServiceParamInitialize", "libSceSystemService",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr param_ptr = ctx.rdi;
            if (param_ptr != 0 && mem.is_valid_address(param_ptr, 0x100)) {
                // Initialize default system service parameters
                mem.zero_memory(param_ptr, 0x100);
            }
            ctx.rax = SCE_OK;
        });

    // =========================================================================
    // libScePad: Controller and User Input
    // =========================================================================

    // scePadInit(void)
    // NID: 0x86FD65BA226BA903ULL -- ps4libdoc: scePadInit. (The NID that used
    // to carry this label, 0x7501D612C26DA04E, is pthread_mutexattr_init.)
    register_symbol(0x86FD65BA226BA903ULL, "scePadInit", "libScePad",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePadOpen(int userId, int type, int index, const void* param)
    // NID: 0xC64D0071AACFDD5EULL ("xk0AcarP3V4")
    register_symbol(0xC64D0071AACFDD5EULL, "scePadOpen", "libScePad",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0x200; // valid pad handle
        });

    // sceMoveInit(void) -- NID: 0x8F521313F1282661ULL. ps4libdoc resolves this
    // NID to sceMoveInit (libSceMove), not scePadClose; the SCE_OK body is the
    // right shape for an init call.
    register_symbol(0x8F521313F1282661ULL, "sceMoveInit", "libSceMove",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePadReadState(int handle, ScePadData* data)
    auto pad_read_state_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr data_ptr = ctx.rsi;
        if (data_ptr > 0x1000 && mem.is_valid_address(data_ptr, 32)) {
            mem.zero_memory(data_ptr, 32);
            const u32 btn = g_pad_buttons.load(std::memory_order_relaxed);
            // ScePadData layout:
            // offset 0x00: u32 buttons
            // offset 0x04: u32 buttons (or padding/touch)
            // offset 0x08: u8 lx, ly, rx, ry
            mem.write_value(data_ptr + 0x00, btn);
            mem.write_value(data_ptr + 0x04, btn);
            // Centered analog sticks (128, 128)
            mem.write_value(data_ptr + 0x08, static_cast<u8>(128));
            mem.write_value(data_ptr + 0x09, static_cast<u8>(128));
            mem.write_value(data_ptr + 0x0A, static_cast<u8>(128));
            mem.write_value(data_ptr + 0x0B, static_cast<u8>(128));
            // Connected status = true
            mem.write_value(data_ptr + 0x10, static_cast<u8>(1));
        }
        ctx.rax = SCE_OK;
    };
    // The real libScePad NIDs, per ps4libdoc known_names.txt. The guest also
    // imports two NIDs ps4libdoc cannot name, which used to carry these
    // labels; since the label provably did not hash to the NID, they are
    // registered under their true NIDs below and the guest's NIDs keep their
    // (verified) working bodies under honest unknown_* labels.
    register_symbol(0x6277605EA41557B7ULL, "scePadReadState", "libScePad", pad_read_state_handler);

    // scePadSetLightBar(int handle, const ScePadLightBarParam* param)
    // NID: 0x451E27A2F50410D6ULL (ps4libdoc). Imported by eboot.bin.
    register_symbol(0x451E27A2F50410D6ULL, "scePadSetLightBar", "libScePad",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // Guest-imported NIDs with no ps4libdoc name, previously mislabeled
    // "scePadReadState" / "scePadSetLightBar". Both are called by eboot.bin;
    // the first one keeps the ScePadData-shaped body because the guest's pad
    // loop depends on reading a sane struct, the second is an SCE_OK no-op.
    register_symbol(0xB445383A59BDC10BULL, "unknown_b445383a", "libScePad", pad_read_state_handler);
    register_symbol(0xD4143F8A8CA539BCULL, "unknown_d4143f8a", "libScePad",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // =========================================================================
    // Unity / Mono / AudioOut / Gnm Additional HLE Stubs
    // =========================================================================

    // printf / hcuQgD53UxM
    register_symbol(0x85CB90803E775313ULL, "printf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr fmt_addr = ctx.rdi;
            const char* fmt = (fmt_addr > 0x1000 && mem.is_valid_address(fmt_addr, 1))
                                ? mem.to_host_ptr<char>(fmt_addr) : nullptr;
            if (fmt) {
                const std::string formatted = format_guest_string(mem, fmt, ctx.rsi, ctx.rdx, ctx.rcx, ctx.r8);
                std::fprintf(stderr, "[Guest printf] %s\n", formatted.c_str());
            }
            ctx.rax = 0;
        });

    // scePthreadGetschedparam / P41kTWUS3EI
    register_symbol(0x3F8D644D6512DC42ULL, "scePthreadGetschedparam", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr policy_ptr = ctx.rsi;
            const GuestAddr param_ptr = ctx.rdx;
            if (policy_ptr != 0 && mem.is_valid_address(policy_ptr, sizeof(i32))) {
                mem.write_value(policy_ptr, static_cast<i32>(0));
            }
            if (param_ptr != 0 && mem.is_valid_address(param_ptr, sizeof(i32))) {
                mem.write_value(param_ptr, static_cast<i32>(0));
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadSetschedparam / oIRFTjoILbg
    register_symbol(0xA084454E3A082DB8ULL, "scePthreadSetschedparam", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // pthread_mutex_lock (reached via mono; POSIX codes coincide with ours on success)
    // NIDs 0xEC7D224CE7224CBA / 0xD99F8FA58E826898 are exported by
    // libScePosix/libkernel per ps4libdoc, not by mono-ps4.
    register_symbol(0xEC7D224CE7224CBAULL, "pthread_mutex_lock", "libScePosix",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // pthread_mutex_unlock (reached via mono)
    register_symbol(0xD99F8FA58E826898ULL, "pthread_mutex_unlock", "libScePosix",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // sceKernelClockGettime(clockid_t clock_id, struct timespec *tp)
    // NID: 0x4018BB1C22B4DE1CULL ("QBi7HCK03hw")
    register_symbol(0x4018BB1C22B4DE1CULL, "sceKernelClockGettime", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr tp_ptr = ctx.rsi;
            if (tp_ptr != 0 && mem.is_valid_address(tp_ptr, 16)) {
                struct timespec ts{};
                ::clock_gettime(CLOCK_MONOTONIC, &ts);
                mem.write_value(tp_ptr, static_cast<i64>(ts.tv_sec));
                mem.write_value(tp_ptr + 8, static_cast<i64>(ts.tv_nsec));
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // sceAjmModuleRegister / Q3dyFuwGn64 (module #X#Y libSceAjm)
    // NID: 0x43777216EC069FAEULL ("Q3dyFuwGn64")
    register_symbol(0x43777216EC069FAEULL, "sceAjmModuleRegister", "libSceAjm",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    // =========================================================================
    // libSceAudio3d: 3D audio (names hash-verified against Orbis NIDs)
    // =========================================================================
    register_symbol(0x83C70CDFD11467AAULL, "sceSysmoduleLoadModule", "libSceSysmodule",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    register_symbol(0x5260AF8D29AE648CULL, "sceAudio3dInitialize", "libSceAudio3d",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    // PortOpen(0xff, stackbuf, heapbuf): return 0 (SCE_OK); no port state kept.
    register_symbol(0x5DE0C32B4C495900ULL, "sceAudio3dPortOpen", "libSceAudio3d",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    // ObjectReserve: returns 0. Observed flows never consume the return value
    // (all later calls pass rdi=-1 = default), so no handle table yet. If a
    // caller ever threads this return into another call, upgrade to dummies.
    register_symbol(0x8CEDAD79CE1D2763ULL, "sceAudio3dObjectReserve", "libSceAudio3d",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    register_symbol(0xE2EC8737DAB865E5ULL, "sceAudio3dObjectSetAttributes", "libSceAudio3d",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    register_symbol(0x64E1ABC562E04331ULL, "sceAudio3dPortFlush", "libSceAudio3d",
        [](CpuContext& ctx, MemoryManager&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            ctx.rax = SCE_OK;
        });

    // sceVideoOutIsFlipPending / zgXifHT9ErY
    // NID: 0xCE05E27C74FD12B6ULL ("zgXifHT9ErY")
    register_symbol(0xCE05E27C74FD12B6ULL, "sceVideoOutIsFlipPending", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // =========================================================================
    // libSceUserService
    // =========================================================================

    // sceUserServiceInitialize / j3YMu1MVNNo
    register_symbol(0x8F760CBB531534DAULL, "sceUserServiceInitialize", "libSceUserService",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceUserServiceTerminate / bwFjS+bX9mA
    register_symbol(0x6F01634BE6D7F660ULL, "sceUserServiceTerminate", "libSceUserService",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceUserServiceGetInitialUser / CdWp0oHWGr0
    register_symbol(0x9D5A9D281D61ABDULL, "sceUserServiceGetInitialUser", "libSceUserService",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr user_id_ptr = ctx.rdi;
            if (user_id_ptr != 0 && mem.is_valid_address(user_id_ptr, sizeof(i32))) {
                mem.write_value(user_id_ptr, static_cast<i32>(0x10000001)); // Default initial user ID
            }
            ctx.rax = SCE_OK;
        });

    // sceUserServiceGetLoginUserIdList / fPhymKNvK-A
    register_symbol(0x7CF87298A36F2BF0ULL, "sceUserServiceGetLoginUserIdList", "libSceUserService",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr list_ptr = ctx.rdi;
            if (list_ptr != 0 && mem.is_valid_address(list_ptr, 16)) {
                mem.zero_memory(list_ptr, 16);
                mem.write_value(list_ptr, static_cast<i32>(0x10000001));
            }
            ctx.rax = SCE_OK;
        });

    // sceUserServiceGetUserName / 1xxcMiGu2fo
    register_symbol(0xD71C5C3221AED9FAULL, "sceUserServiceGetUserName", "libSceUserService",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr name_ptr = ctx.rsi;
            const GuestSize size = ctx.rdx;
            if (name_ptr != 0 && mem.is_valid_address(name_ptr, size)) {
                mem.zero_memory(name_ptr, size);
                const char* default_name = "User";
                std::memcpy(mem.to_host_ptr<void>(name_ptr), default_name, std::min(size, static_cast<GuestSize>(5)));
            }
            ctx.rax = SCE_OK;
        });

    // sceUserServiceGetUserColor -- NID: 0x954A2AC1342EE06AULL ("lUoqwTQu4Go").
    // ps4libdoc resolves the Sony tag to this name; the SCE_OK body is right
    // for the init-style call the guest makes here.
    register_symbol(0x954A2AC1342EE06AULL, "sceUserServiceGetUserColor", "libSceUserService",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceUserServiceGetEvent / yH17Q6NWtVg
    // Returns SCE_USER_SERVICE_ERROR_NO_EVENT (0x80960007) when no event is queued,
    // which unblocks Unity's event loop to proceed with frame rendering!
    register_symbol(0xC87D7B43A356B558ULL, "sceUserServiceGetEvent", "libSceUserService",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0x80960007ULL; // SCE_USER_SERVICE_ERROR_NO_EVENT
        });

    // puts(const char* s) -- NID: 0x610D276AFA7E6087ULL, which ps4libdoc
    // libkernel.sprx.json lists as puts (libSceLibcInternal). It was labelled
    // "puts_orbis", which is a UE-side macro, not an Orbis export. Unity
    // Launcher Control imports this NID. Body unchanged: delegate to the
    // handler behind the other puts NID.
    register_symbol(0x610D276AFA7E6087ULL, "puts", "libSceLibcInternal",
        [this](CpuContext& ctx, MemoryManager& mem) {
            const auto* s = find_by_nid(0x5C0F663677A000A8ULL);
            if (s && s->handler) s->handler(ctx, mem);
        });

    // strcat(char* dst, const char* src) — NID 0x2ECE2DCF38629AA4ULL hashes to
    // strcat, NOT strlcpy: must APPEND. (The old strlcpy body truncated dst.)
    // Returns dst; scans bounded so an unterminated dst can't run away.
    register_symbol(0x2ECE2DCF38629AA4ULL, "strcat", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dst = ctx.rdi;
            const GuestAddr src = ctx.rsi;
            if (dst > 0x1000 && src > 0x1000) {
                usize dst_len = 0;
                while (dst_len < 4096) {
                    u8 b = 0;
                    if (!mem.is_valid_address(dst + dst_len, 1) || !mem.read_value(dst + dst_len, b)) break;
                    if (b == 0) break;
                    ++dst_len;
                }
                usize src_len = 0;
                while (src_len < 4096) {
                    u8 b = 0;
                    if (!mem.is_valid_address(src + src_len, 1) || !mem.read_value(src + src_len, b)) break;
                    if (b == 0) break;
                    ++src_len;
                }
                if (mem.is_valid_address(dst + dst_len, src_len + 1)) {
                    char* d = mem.to_host_ptr<char>(dst + dst_len);
                    const char* s = mem.to_host_ptr<char>(src);
                    if (d && s) {
                        std::memcpy(d, s, src_len);
                        d[src_len] = '\0';
                    }
                }
            }
            ctx.rax = dst; // strcat returns dst
        });

    // mono_add_internal_call(const char* name, const void* method)
    // NID: 0xFD5B758A1AF32344ULL
    register_symbol(0xFD5B758A1AF32344ULL, "mono_add_internal_call", "mono-ps4",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // mono_jit_init_version(const char* domain_name, const char* runtime_version)
    // NID: 0x9B795B57A3B5E063ULL -> returns the arena domain singleton so all
    // domain handles in the process compare equal.
    register_symbol(0x9B795B57A3B5E063ULL, "mono_jit_init_version", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
            if (s_mono_arena.domain == 0) {
                s_mono_arena.domain = mono_arena_alloc(mem);
            }
            ctx.rax = s_mono_arena.domain;
        });

    // mono_domain_get(void) -> singleton dummy MonoDomain*
    // NID: 0x88095379DC1FF3D9ULL
    register_symbol(0x88095379DC1FF3D9ULL, "mono_domain_get", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
            if (s_mono_arena.domain == 0) {
                s_mono_arena.domain = mono_arena_alloc(mem);
                std::fprintf(stderr, "[Mono] domain_get -> 0x%llx\n",
                             (unsigned long long)s_mono_arena.domain);
            }
            ctx.rax = s_mono_arena.domain;
        });

    // mono_domain_assembly_open(MonoDomain* domain, const char* name)
    // NID: 0x7427A284FB5A7423ULL
    register_symbol(0x7427A284FB5A7423ULL, "mono_domain_assembly_open", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr name_addr = ctx.rsi;
            const char* name = (name_addr > 0x1000 && mem.is_valid_address(name_addr, 1))
                                   ? mem.to_host_ptr<char>(name_addr) : nullptr;
            GuestAddr slot = 0;
            if (name) {
                std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
                auto it = s_mono_arena.assemblies.find(name);
                if (it == s_mono_arena.assemblies.end()) {
                    slot = mono_arena_alloc(mem);
                    if (slot) s_mono_arena.assemblies[name] = slot;
                } else {
                    slot = it->second;
                }
                std::fprintf(stderr, "[Mono] domain_assembly_open('%.128s') -> 0x%llx\n",
                             name, (unsigned long long)slot);
            }
            ctx.rax = slot;
        });

    // mono_assembly_get_image(MonoAssembly* assembly)
    // NID: 0xC4FE040B84A9F462ULL
    register_symbol(0xC4FE040B84A9F462ULL, "mono_assembly_get_image", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr assembly = ctx.rdi;
            GuestAddr slot = 0;
            if (assembly != 0) {
                std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
                auto it = s_mono_arena.images.find(assembly);
                if (it == s_mono_arena.images.end()) {
                    slot = mono_arena_alloc(mem);
                    if (slot) {
                        s_mono_arena.images[assembly] = slot;
                        if (s_mono_arena.first_image == 0) s_mono_arena.first_image = slot;
                    }
                } else {
                    slot = it->second;
                }
                std::fprintf(stderr, "[Mono] assembly_get_image(0x%llx) -> 0x%llx\n",
                             (unsigned long long)assembly, (unsigned long long)slot);
            }
            ctx.rax = slot;
        });

    // mono_class_from_name(MonoImage* image, const char* name_space, const char* name)
    // NID: 0x6F3C536529E7077FULL
    register_symbol(0x6F3C536529E7077FULL, "mono_class_from_name", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr image = ctx.rdi;
            const GuestAddr ns_addr = ctx.rsi;
            const GuestAddr name_addr = ctx.rdx;
            const char* ns = (ns_addr > 0x1000 && mem.is_valid_address(ns_addr, 1))
                                 ? mem.to_host_ptr<char>(ns_addr) : nullptr;
            const char* name = (name_addr > 0x1000 && mem.is_valid_address(name_addr, 1))
                                   ? mem.to_host_ptr<char>(name_addr) : nullptr;
            GuestAddr slot = 0;
            if (ns && name) {
                std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
                // Fall back to the corlib image when none is supplied: every
                // System.* lookup goes through mono_get_corlib(), so this is the
                // image the guest means. The cookies are interchangeable zeros,
                // and a NULL image here otherwise retries forever upstream.
                GuestAddr eff_image = (image != 0) ? image
                                                    : (s_mono_arena.corlib_image != 0
                                                           ? s_mono_arena.corlib_image
                                                           : s_mono_arena.first_image);
                if (eff_image != 0) {
                    MonoArena::ClassKey key{eff_image, ns, name};
                    auto it = s_mono_arena.classes.find(key);
                    if (it == s_mono_arena.classes.end()) {
                        slot = mono_arena_alloc(mem);
                        if (slot) s_mono_arena.classes.emplace(std::move(key), slot);
                    } else {
                        slot = it->second;
                    }
                    std::fprintf(stderr, "[Mono] class_from_name('%.128s', '%.128s') -> 0x%llx\n",
                                 ns, name, (unsigned long long)slot);
                }
            }
            if (slot == 0) {
                // Log raw pointers (capped) to identify which input fails.
                static std::atomic<unsigned> s_fail_logged{0};
                if (s_fail_logged.fetch_add(1) < 20) {
                    std::fprintf(stderr,
                        "[Mono] class_from_name FAIL image=0x%llx ns=0x%llx name=0x%llx\n",
                        (unsigned long long)image, (unsigned long long)ns_addr,
                        (unsigned long long)name_addr);
                }
            }
            ctx.rax = slot;
        });

    // mono_class_is_subclass_of(MonoClass* klass, MonoClass* parent, gboolean check_interfaces)
    // NID: 0x1147722929EB0A4CULL -- hash-verified against the Mono embedding API.
    // Guest disassembly at 0x997891 shows this is a type-validation guard, not a
    // query whose answer is inspected in detail:
    //
    //   test rbx, rbx / je skip          ; nothing to validate?
    //   mov edx, 1 / mov rdi, rbx / mov rsi, r14
    //   call mono_class_is_subclass_of   ; -> test eax, eax / jne accept
    //   accept:  mov rax, rbx            ; return the candidate class
    //   reject:  walk [r15] -> 0x997860  ; recurse up the parent chain
    //
    // Answering TRUE for any two live pointers short-circuits that parent walk
    // and returns the candidate immediately. Answering FALSE (the old
    // reflexive-identity test, which always failed on our distinct cookies)
    // sent the guest into the recursive walk on every single call.
    register_symbol(0x1147722929EB0A4CULL, "mono_class_is_subclass_of", "mono-ps4",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = (ctx.rdi != 0 && ctx.rsi != 0) ? 1 : 0;
        });

    // mono_runtime_invoke(MonoMethod* method, void* obj, void** params, MonoObject** exc)
    // NID: 0xA6B7BD0639320E1BULL — hash-verified. No IL execution: the managed
    // body never runs, so the only contract we can honour is "do not report a
    // managed exception". The caller at 0x98a780 passes (0, 0, stackbuf), i.e.
    // method = NULL, which is why every direct lookup upstream now hands back a
    // real cookie instead of NULL. Clears *exc so the caller never reads
    // uninitialized stack garbage as a managed exception.
    register_symbol(0xA6B7BD0639320E1BULL, "mono_runtime_invoke", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr method = ctx.rdi;
            const GuestAddr obj = ctx.rsi;
            const GuestAddr params = ctx.rdx;
            const GuestAddr exc_ptr = ctx.rcx;
            if (exc_ptr > 0x1000 && mem.is_valid_address(exc_ptr, sizeof(u64))) {
                mem.write_value(exc_ptr, static_cast<u64>(0));
            }
            std::fprintf(stderr, "[Mono] runtime_invoke method=0x%llx obj=0x%llx params=0x%llx -> NULL (no IL)\n",
                         static_cast<unsigned long long>(method),
                         static_cast<unsigned long long>(obj),
                         static_cast<unsigned long long>(params));
            ctx.rax = 0;
        });

    // mono_class_get_parent(MonoClass* klass) — iterator family: NULL ends the
    // hierarchy walk at 0x99aeef (do/while klass != NULL). Explicit handler pins
    // the behavior independently of the global unimplemented fallback.
    // NID: 0x4CCB2C2760985FAAULL
    register_symbol(0x4CCB2C2760985FAAULL, "mono_class_get_parent", "mono-ps4",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // mono_get_corlib(void) -> singleton MonoImage* for mscorlib.
    // NID: 0x47F74CE8E2982CC4ULL (hash-verified: see scripts/crack_mono_nids.py).
    // Guest call site 0x9878b1 takes no arguments and feeds the result straight
    // into mono_class_from_name(image, "System.Collections", "IEnumerator") at
    // 0x9878c7 -- so a NULL here means every System.* lookup is issued with
    // image = NULL. It also becomes the arena's default image below, so all
    // corlib lookups share one image cookie (correct corlib discipline).
    register_symbol(0x47F74CE8E2982CC4ULL, "mono_get_corlib", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
            if (s_mono_arena.corlib_image == 0) {
                s_mono_arena.corlib_image = mono_arena_alloc(mem);
                std::fprintf(stderr, "[Mono] get_corlib -> 0x%llx\n",
                             (unsigned long long)s_mono_arena.corlib_image);
            }
            ctx.rax = s_mono_arena.corlib_image;
        });

    // =====================================================================
    // mono_class_get_methods(MonoClass* klass, void** iter) -> MonoMethod*
    // NID: 0x2C5314E29C000914ULL -- hash-verified. One-shot iterator: hand back
    // exactly one method, then NULL forever. This is what the guest's lookup
    // loop at 0x99ae30 wants:
    //
    //   iter = 0
    //   method = mono_class_get_methods(klass, &iter)
    //   while (method) {
    //       name = mono_method_get_name(method)      // 0x99ae33
    //       if (*name == *wanted)                    // 0x99ae38
    //           if (!strcmp(name, wanted)) { found } // 0x99ae47
    //       method = mono_class_get_methods(klass, &iter)
    //   }
    //   // "Unable to find method %s in [%s]"
    //
    // Returning NULL (the old behaviour) made every one of Unity's 56 method
    // lookups fail on the first turn of the loop and print that warning. One
    // method is enough, provided the name matches -- see mono_method_get_name.
    register_symbol(0x2C5314E29C000914ULL, "mono_class_get_methods", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr iter_ptr = ctx.rsi;
            u64 state = 0;
            if (iter_ptr != 0 && mem.is_valid_address(iter_ptr, sizeof(u64))) {
                if (mem.read_value(iter_ptr, state) && state != 0) {
                    ctx.rax = 0; // second turn: terminate the walk
                    return;
                }
                mem.write_value(iter_ptr, static_cast<u64>(1));
            }
            std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
            ctx.rax = mono_cookie(mem, "mono_method", mono_args(ctx));
        });

    // mono_method_get_name(MonoMethod* method) -> const char*
    // NID: 0x12101A8AA03B2607ULL -- hash-verified.
    //
    // The caller keeps the name it is hunting for in a callee-saved register
    // (r12 at 0x99ae44) for the whole walk, so the live register file *is* the
    // answer: returning r12 makes the first-character compare at 0x99ae3b and
    // the strcmp at 0x99ae47 both succeed on the first method we hand out,
    // which is what turns "Unable to find method" into a hit. The dispatcher
    // passes the guest's real CpuContext, so ctx.r12 is that register; when it
    // does not look like a guest string we fall back to NULL rather than hand
    // back a garbage pointer.
    register_symbol(0x12101A8AA03B2607ULL, "mono_method_get_name", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr wanted = ctx.r12;
            if (wanted > 0x1000 && mem.is_valid_address(wanted, 1)) {
                const char* s = mem.to_host_ptr<char>(wanted);
                if (s && *s != '\0') {
                    ctx.rax = wanted;
                    return;
                }
            }
            ctx.rax = 0;
        });

    // Three mono-ps4 exports (import table module tag "i#j", the same as
    // mono_method_get_name) that ps4libdoc's 42k corpus and the 2,836 upstream
    // Mono header symbols both fail to name -- Unity's forked mono-ps4.prx
    // exports internals that never shipped in Mono. Rather than guess a name,
    // the *behaviour* comes from the guest's own call chain at 0x98c4cb, which
    // is a three-way capability probe, not a pointer lookup:
    //
    //   98c4ce: call probe1;  test eax,eax / je next
    //            -> non-zero sets result = 0xfffffffc (-4) and returns
    //   98c4ee: call probe2;  test eax,eax / jne done
    //   98c4fa: call probe3;  test eax,eax / je 0x98c564 (failure path)
    //   0x98c503: result = 0xfffffffa (-6), then the normal epilogue
    //   0x98c564: mono_class_is_subclass_of(r15, corlib_type, TRUE) twice
    //
    // All three are 32-bit int probes on one pointer, so returning 0 ("feature
    // absent") is the honest answer and sends the guest down 0x98c564, which
    // our mono_class_is_subclass_of now answers TRUE for, so it survives there.
    register_symbol(0x0DAB36A8B7A8F089ULL, "unknown_98c4ce_probe1", "mono-ps4",
        [](CpuContext&, MemoryManager&) { /* 0 = absent; non-zero is the guest's error -4 path */ });

    register_symbol(0x053D63D15F72012EULL, "unknown_98c4ee_probe2", "mono-ps4",
        [](CpuContext&, MemoryManager&) { /* 0 = fall through to probe3 */ });

    register_symbol(0xFF23F26597C04F63ULL, "unknown_98c4fa_probe3", "mono-ps4",
        [](CpuContext&, MemoryManager&) { /* 0 = take the subclass-of path at 0x98c564 */ });

    // Mono lookup chain
    // =====================================================================
    // eboot.bin imports 107 Mono NIDs (scripts/crack_mono_nids.py --all names
    // all 107 against the official Mono embedding API). Twenty are registered
    // above and in the sync-stub table; the rest are classified here by what
    // the *guest* does with the return value, which is the only thing that
    // decides whether a cookie, a NULL or a scalar keeps the runtime alive.
    //
    //   Iter   NULL. The guest iterates until NULL; a cookie here is an
    //          infinite loop, since nothing else terminates the walk.
    //   Cookie Distinct, stable, zeroed arena slot keyed on the arguments, so
    //          repeated lookups compare equal and downstream calls (notably
    //          mono_runtime_invoke at 0x98a780) do not receive NULL.
    //   Null   "absent" answers: strings, exceptions, basetypes. NULL is the
    //          documented miss and every caller already handles it.
    //   Zero   Scalar getter returns 0 (counts, sizes, gc stats, enums).
    //   Small  Scalar getter returns a sane non-zero default, chosen so the
    //          guest's arithmetic stays meaningful.
    //   Pass   Handle we do not model, echoed straight back so an
    //          acquire/release round trip balances.
    enum class MonoShape { Iter, Cookie, Null, Zero, Small, Pass };

    struct MonoStub {
        Nid nid;
        const char* name;
        MonoShape shape;
        u64 small; // value returned by Small (and by Zero when non-zero)
        const char* why;
    };
    static constexpr MonoStub kMonoStubs[] = {
        // ---- iterators and hierarchy walkers: NULL ends the walk in 1 step ----
        {0x985C95D970DA2725ULL, "mono_class_get_fields", MonoShape::Iter, 0, "field list"},
        {0x2CA0A04776952703ULL, "mono_class_get_interfaces", MonoShape::Iter, 0, "interface list"},
        {0x3E9C268563CD9AAEULL, "mono_class_get_nested_types", MonoShape::Iter, 0, "nested type list"},
        {0x6A8EF234F71F1017ULL, "mono_class_enum_basetype", MonoShape::Iter, 0,
         "NULL = no underlying enum type, the terminating answer"},
        {0xBB4AD888389AE22AULL, "mono_custom_attrs_from_class", MonoShape::Iter, 0,
         "attr iterator: NULL ends the walk"},
        {0x317F27B80647C7E9ULL, "mono_custom_attrs_from_method", MonoShape::Iter, 0, "attr iterator"},
        {0x53BDBBFB9080F677ULL, "mono_custom_attrs_from_field", MonoShape::Iter, 0, "attr iterator"},
        {0xA89CBBA0E6A51CC8ULL, "mono_signature_get_params", MonoShape::Iter, 0, "param iterator"},
        {0x286DBC3D2512C3ADULL, "mono_signature_get_return_type", MonoShape::Iter, 0,
         "NULL = void, safe for every caller"},
        // NOT an iterator: the guest treats a non-NULL signature as "this method
        // resolved". 0x99af1b is `test rax,rax; jne <success>` right after the
        // call, and the failure branch is what prints "Unable to find method".
        {0xD6C3FE5C14459AA8ULL, "mono_method_signature", MonoShape::Cookie, 0,
         "non-NULL == resolved; feeds the success branch at 0x99af1b"},
        {0x3CE811D876E50171ULL, "mono_class_get_element_class", MonoShape::Iter, 0,
         "NULL = not an array element type"},
        {0xC0404976DC95FD98ULL, "mono_class_get_nesting_type", MonoShape::Iter, 0, "NULL = not nested"},
        {0x7EBF057CF032664DULL, "mono_field_get_parent", MonoShape::Iter, 0, "NULL = no declaring parent"},

        // ---- direct lookups and constructors: distinct cookies ----
        {0x9BF8387D4F9CFB29ULL, "mono_class_get_field_from_name", MonoShape::Cookie, 0,
         "feeds the parent walk at 0x9978a3; NULL here restarts that walk"},
        {0xCBA45942B44FB62DULL, "mono_class_from_name_case", MonoShape::Cookie, 0, "case-insensitive lookup"},
        {0xB44311ACE3519578ULL, "mono_class_from_mono_type", MonoShape::Cookie, 0, "type -> class"},
        {0x8A6DD4924F4B6A87ULL, "mono_class_get_image", MonoShape::Cookie, 0, "class -> image"},
        {0xA39A90A33CF93422ULL, "mono_image_get_assembly", MonoShape::Cookie, 0, "image -> assembly"},
        {0xBF6E6C1A2251C000ULL, "mono_class_get_type", MonoShape::Cookie, 0, "class -> MonoType"},
        {0x94E6663D7C4DA186ULL, "mono_type_get_class", MonoShape::Cookie, 0, "type -> class"},
        {0xA2560F030F332A67ULL, "mono_type_get_object", MonoShape::Cookie, 0, "type -> boxed object"},
        {0xC1E523D0E1FCCF1EULL, "mono_method_get_class", MonoShape::Cookie, 0, "method -> class"},
        {0x5C669D5DFD9E3C3AULL, "mono_object_get_class", MonoShape::Cookie, 0, "object -> class"},
        {0x449275D5CAD0CA7BULL, "mono_object_get_virtual_method", MonoShape::Cookie, 0,
         "NULL here means the guest cannot find a method to invoke"},
        {0xF2A93AD894B6746DULL, "mono_object_new", MonoShape::Cookie, 0, "constructor"},
        {0xB7C7D9218C1D654CULL, "mono_array_new", MonoShape::Cookie, 0, "array constructor"},
        {0xF70952A358AA3C1BULL, "mono_array_new_full", MonoShape::Cookie, 0, "multi-dim array constructor"},
        {0x74D76D8EAD1CDCADULL, "mono_string_new_wrapper", MonoShape::Cookie, 0, "MonoString*"},
        {0xBE365CC81E5C4B88ULL, "mono_string_new_len", MonoShape::Cookie, 0, "MonoString*"},
        {0xB72AA0CBF7B58A9AULL, "mono_string_from_utf16", MonoShape::Cookie, 0, "MonoString*"},
        {0x7855074AE7EF9A7CULL, "mono_value_box", MonoShape::Cookie, 0, "boxed value"},
        {0x78B6AC776F7141F6ULL, "mono_array_class_get", MonoShape::Cookie, 0, "array class"},
        {0x72EB5A3894E60566ULL, "mono_get_enum_class", MonoShape::Cookie, 0, "enum class lookup"},
        {0x7C5EF9110AFF621DULL, "mono_field_get_type", MonoShape::Cookie, 0, "field -> MonoType"},
        {0xAAA5318F02737362ULL, "mono_thread_attach", MonoShape::Cookie, 0,
         "guest spins while mono_thread_current() == NULL; attach must answer"},
        {0xE5AF1BFACE87B5A0ULL, "mono_get_root_domain", MonoShape::Cookie, 0, "domain singleton"},

        // ---- "absent" answers ----
        {0x9755B24D09AB017AULL, "mono_exception_from_name_msg", MonoShape::Null, 0,
         "NULL = no exception, the non-throwing answer"},
        {0xEB03E0304FCD74A0ULL, "mono_exception_from_name_two_strings", MonoShape::Null, 0, "NULL = no exception"},
        {0x3B82D819C9B9BEC5ULL, "mono_get_exception_argument_null", MonoShape::Null, 0, "NULL = no exception"},
        {0x5E128B646CBE6269ULL, "mono_object_unbox", MonoShape::Null, 0, "NULL = nothing to unbox"},
        {0xF73E9CCFB360B030ULL, "mono_raise_exception", MonoShape::Null, 0, "no-op: never throw into the guest"},
        {0x027F88F9F6C50F35ULL, "mono_string_to_utf8", MonoShape::Null, 0,
         "never hand back a host pointer as a guest char*"},
        {0x1D8189A425AB45B7ULL, "mono_image_get_name", MonoShape::Null, 0, "NULL = unnamed image"},
        {0x52735B4D29BF9A92ULL, "mono_field_get_name", MonoShape::Null, 0, "NULL = unnamed field"},
        {0xB684ADBE5D881180ULL, "mono_class_get_namespace", MonoShape::Null, 0, "NULL = global namespace"},
        {0xFABF16BE647EFDD0ULL, "mono_class_get_name", MonoShape::Null, 0, "NULL = unnamed class"},
        {0xD75CB3ABC4694E9DULL, "mono_type_get_name", MonoShape::Null, 0, "NULL = unnamed type"},
        {0x3EE593AA8DBFC028ULL, "mono_assembly_name_parse", MonoShape::Null, 0, "NULL = unparseable name"},
        {0x082422B993F4575AULL, "mono_loader_get_last_error", MonoShape::Null, 0, "NULL = no loader error"},
        {0x29DB18B2541F112FULL, "mono_type_get_type", MonoShape::Null, 0,
         "0 is MONO_TYPE_END, which terminates guest type walks"},

        // ---- scalar getters: 0 ----
        {0x01E683D909A1E81DULL, "mono_gc_get_heap_size", MonoShape::Zero, 0, "gc stat"},
        {0x571259700E26B5B5ULL, "mono_gc_get_used_size", MonoShape::Zero, 0, "gc stat"},
        {0x9207476EBA98414BULL, "mono_gc_max_generation", MonoShape::Zero, 0, "0 terminates generation walks"},
        {0x5ADCCE0B91D66DADULL, "mono_gc_collect", MonoShape::Zero, 0, "void"},
        {0x4760DDFEF361CC51ULL, "mono_signature_get_param_count", MonoShape::Zero, 0, "no parameters"},
        {0x6024879BBCF299AEULL, "mono_array_element_size", MonoShape::Zero, 0, "unknown element size"},
        {0xBDF67AEE37821CE5ULL, "mono_class_array_element_size", MonoShape::Zero, 0, "unknown element size"},
        {0x89097FFE4DAE9029ULL, "mono_class_instance_size", MonoShape::Zero, 0, "header only"},
        {0x058661C22B770987ULL, "mono_field_get_flags", MonoShape::Zero, 0, "field flags"},
        {0xE1410B51114B5E49ULL, "mono_field_get_offset", MonoShape::Zero, 0, "first field at offset 0"},
        {0xB4B2F7382FE6AC28ULL, "mono_class_is_enum", MonoShape::Zero, 0, "false"},
        {0xF02026004BD1040BULL, "mono_class_is_valuetype", MonoShape::Zero, 0, "false"},
        {0xFAA2D1FCC5BDD8F7ULL, "mono_signature_is_instance", MonoShape::Zero, 0, "false"},
        {0x9D675C43174F4E77ULL, "mono_custom_attrs_has_attr", MonoShape::Zero, 0,
         "false ends attribute walks"},
        {0x6B05EAD393AB9478ULL, "mono_field_set_value", MonoShape::Zero, 0, "void"},
        {0x334C5174D51AFE4BULL, "mono_runtime_object_init", MonoShape::Zero, 0, "void"},
        {0xACE61410F5E03AA9ULL, "mono_gchandle_free", MonoShape::Zero, 0, "void"},
        {0xAC74A8FF223743E5ULL, "mono_custom_attrs_free", MonoShape::Zero, 0, "void"},
        {0xD607E8C1B0D46402ULL, "mono_thread_detach", MonoShape::Zero, 0, "void"},
        {0x57265F784A3CB6A7ULL, "mono_thread_suspend_all_other_threads", MonoShape::Zero, 0, "no-op"},
        {0xDC18C1D69B04C7CBULL, "mono_threads_set_shutting_down", MonoShape::Zero, 0, "void"},
        {0xE2087720CF525F0FULL, "mono_runtime_set_shutting_down", MonoShape::Zero, 0, "void"},
        {0x74AED6A27FC4DC44ULL, "mono_thread_pool_cleanup", MonoShape::Zero, 0, "void"},
        {0x750EDD459A4AF942ULL, "mono_assembly_loaded", MonoShape::Zero, 0, "false"},
        {0x775F03F33FBB182DULL, "mono_jit_cleanup", MonoShape::Zero, 0, "void"},
        {0x72C4E085629A4BDBULL, "mono_verifier_set_mode", MonoShape::Zero, 0, "void"},
        {0x741BC47084A1F663ULL, "mono_security_set_mode", MonoShape::Zero, 0, "void"},
        {0xDAF08486DADE84DEULL, "mono_security_set_core_clr_platform_callback", MonoShape::Zero, 0, "void"},

        // ---- scalar getters: small sane defaults ----
        {0x7498A7D3FC1EAA37ULL, "mono_class_get_rank", MonoShape::Small, 1,
         "1 = not an array; >1 would trigger multi-dim paths"},
        {0x579C63341A42AFA4ULL, "mono_object_get_size", MonoShape::Small, 8,
         "one vtable pointer; 0 would make the guest size a 0-byte copy"},
        {0xDE7B2CA71E6DD354ULL, "mono_class_get_flags", MonoShape::Small, 0x21,
         "TYPE_PUBLIC | TYPE_INSTANCE, the ordinary case"},

        // ---- handles echoed back so acquire/release balances ----
        {0x6DA6206FC639A627ULL, "mono_gchandle_new", MonoShape::Pass, 0, "handle is the object itself"},
        {0x24AD956E34BFDE93ULL, "mono_gchandle_new_weakref", MonoShape::Pass, 0, "handle is the object"},
        {0x8615A1D0674EFC8AULL, "mono_gchandle_get_target", MonoShape::Pass, 0, "reverse of gchandle_new"},
        {0x38B443E7CB06686DULL, "mono_gchandle_is_in_domain", MonoShape::Small, 1,
         "true: the arena never moves objects"},
    };

    for (const MonoStub& stub : kMonoStubs) {
        register_symbol(stub.nid, stub.name, "mono-ps4",
            [stub](CpuContext& ctx, MemoryManager& mem) {
                switch (stub.shape) {
                    case MonoShape::Iter:
                    case MonoShape::Null:
                        ctx.rax = 0;
                        break;
                    case MonoShape::Zero:
                        ctx.rax = stub.small;
                        break;
                    case MonoShape::Small:
                        ctx.rax = stub.small;
                        break;
                    case MonoShape::Pass:
                        ctx.rax = ctx.rdi;
                        break;
                    case MonoShape::Cookie: {
                        std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
                        ctx.rax = mono_cookie(mem, stub.name, mono_args(ctx));
                        break;
                    }
                }
            });
    }

    // mono_thread_current must agree with mono_thread_attach above: the guest's
    // "while (mono_thread_current() == NULL) mono_thread_attach();" spin needs a
    // per-thread handle that becomes non-NULL on first use.
    register_symbol(0xB8369AEA2E0EF71AULL, "mono_thread_current", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            static thread_local GuestAddr slot = 0;
            if (slot == 0) {
                std::lock_guard<std::mutex> lock(s_mono_arena.mtx);
                slot = mono_arena_alloc(mem);
                std::fprintf(stderr, "[Mono] thread_current -> 0x%llx\n",
                             (unsigned long long)slot);
            }
            ctx.rax = slot;
        });

    // Unity memory allocator, FMOD, Mono and GNM setup stubs. Every NID here
    // is a one-shot call we answer with SCE_OK; the names come from ps4libdoc
    // (known_names.txt / libkernel.sprx.json) so the trace labels are true.
    // NIDs ps4libdoc cannot name keep the honest sceKernelSyncStub label and
    // their old context note. Bodies are unchanged: rax = SCE_OK (i.e. NULL
    // for pointer-returning APIs), which is fail-safe.
    struct SyncStub {
        Nid nid;
        const char* name;
        const char* module;
    };
    static constexpr SyncStub kSyncStubs[] = {
        {0x2AE3AE0F9F21AA7EULL, "malloc_stats_fast", "libSceLibcInternal"},
        {0xA4EF7A4F0CCE9B91ULL, "sceKernelGetDirectMemorySize", "libkernel"},
        {0x04DA30C76979F3C1ULL, "sceKernelGetProcessTimeCounterFrequency", "libkernel"},
        {0xFE99E3DFFEDAE900ULL, "sceKernelSyncStub", "libkernel"}, // FMOD / pad init helper
        {0x8941B88CB45FD196ULL, "sceKernelSyncStub", "libkernel"}, // FMOD mixer thread setup
        {0x54EC7C3469875D3BULL, "sceKernelGetCpumode", "libkernel_cpumode"},
        {0xFEEC04868FFB80EBULL, "sceKernelSyncStub", "libkernel"}, // thread attr helper
        {0x22EC6752E5E4E818ULL, "sceKernelGetModuleList", "libkernel"},
        {0xFE19F5B5C547AB94ULL, "sceLibcMspaceCreate", "libSceLibcInternal"},
        {0x960213B81B11A36AULL, "sceFiosIOFilterAdd", "libSceFios2"}, // debug.log stream setup
        {0x47D940F363AB68DBULL, "sceAppContentInitialize", "libSceAppContent"},
        {0x55EED6C524805364ULL, "sceKernelSyncStub", "libkernel"}, // Unity string/domain setup 1
        {0xC5D3D91AED17EB3BULL, "sceKernelSyncStub", "libkernel"}, // Unity string/domain setup 2
        {0x4FCEF2B219D790C5ULL, "mono_jit_set_aot_only", "mono-ps4"},
        {0x72CE3B9DE94CEA50ULL, "sceKernelSyncStub", "libkernel"},
        {0x46204874F5E83DD1ULL, "sceKernelSyncStub", "libkernel"},
        {0x3B579C5F88AEC0EFULL, "mono_config_parse", "mono-ps4"},
        {0xFEAF6C4747115067ULL, "mono_parse_default_optimizations", "mono-ps4"},
        {0xE1E3B7F43BAC2E92ULL, "mono_set_defaults", "mono-ps4"},
        {0xF737BFA02980EDF6ULL, "sceKernelSyncStub", "libkernel"},
        {0x5AA1DFAA34975749ULL, "mono_set_dirs", "mono-ps4"},
        {0xF068CCD895AE7A94ULL, "mono_set_assemblies_path", "mono-ps4"},
        // mono_thread_current has a real handler above (per-thread handle).
        {0x23FA65EA7B52AD86ULL, "mono_thread_set_main", "mono-ps4"},
        {0x9141F1F3ED8B6742ULL, "sceKernelSyncStub", "libkernel"},
        {0x0E04B6E6017A7AE4ULL, "sceKernelSyncStub", "libkernel"},
        {0x7050A9D0D5FCC1FDULL, "sceGnmSetVgtControl", "libSceGnm"},
        {0x8008429FA5225386ULL, "sceGnmSetVsShader", "libSceGnm"},
        {0xE6E14A7248896113ULL, "sceGnmSetPsShader350", "libSceGnm"},
        {0x186B27EE3313C70EULL, "sceGnmDrawIndexAuto", "libSceGnm"},
        // Not scePthreadSetschedparam (that is 0x0F3112F61405E1FEULL).
        {0xFD98F4C8AD37832AULL, "sceKernelSyncStub", "libkernel"},
        {0xD8866B1E72E722DDULL, "sceKernelSyncStub", "libkernel"}, // Unity init stub
        {0x066E40B33FB6E36FULL, "mono_runtime_set_main_args", "mono-ps4"},
        {0x42CD305AE96097B5ULL, "sceMouseInit", "libSceMouse"},
        {0x9839A030E19552A8ULL, "pthread_mutexattr_settype", "libScePosix"},
        {0xE6DC4A7DC3140289ULL, "pthread_mutexattr_setprotocol", "libScePosix"},
        {0xB6D1CD7D4FAA0C15ULL, "pthread_mutex_init", "libScePosix"},
        {0x1C5EE52B8EB1CE36ULL, "pthread_mutexattr_destroy", "libScePosix"},
        {0xD13C959383122EDDULL, "pthread_cond_init", "libScePosix"},
        {0x9AA50B35D8A64E7DULL, "pthread_key_create", "libScePosix"},
        // Stack-smashing abort, not scePthreadSetprio. Still a stub: the
        // compatibility layer prefers continuing over aborting the guest.
        {0x967DF7CE306B7E39ULL, "sceGnmGetTheTessellationFactorRingBufferBaseAddress", "libSceGnm"},
        {0xE6E7409BEE9BA158ULL, "sceGnmCreateWorkloadStream", "libSceGnm"},
        {0x8A1C6B6ECA122967ULL, "sceGnmBeginWorkload", "libSceGnm"},
        {0xC9BD9C4616A00F52ULL, "sceGnmDrawInitDefaultHardwareState350", "libSceGnm"},
        {0xD6A5CB1C8A5138F1ULL, "sceGnmInsertWaitFlipDone", "libSceGnm"},
        {0x5B512D8FF8E55BB6ULL, "sceGnmInsertPushMarker", "libSceGnm"},
        {0xEEA65536012EF926ULL, "sceGnmInsertPopMarker", "libSceGnm"},
        {0x8E222DCD2EBEDB68ULL, "sceGnmInsertSetMarker", "libSceGnm"},
    };
    for (const SyncStub& stub : kSyncStubs) {
        register_symbol(stub.nid, stub.name, stub.module,
            [](CpuContext& ctx, MemoryManager&) {
                ctx.rax = SCE_OK;
            });
    }

    // sceGnmSubmitDone -- NID: 0xCAF67BDEE414AAB9ULL, per ps4libdoc. This NID was
    // registered as sceVideoOutWaitVblank; ps4libdoc resolves it to the GNM
    // submit-completion wait (imported by eboot.bin, libSceJobManager.prx and
    // libSceS3DConversion.prx). Body unchanged: sleep instead of spin so the
    // guest yields the host thread at ~60 FPS VBlank cadence.
    auto vblank_wait_handler = [](CpuContext& ctx, MemoryManager&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS VBlank
        ctx.rax = SCE_OK;
    };
    register_symbol(0xCAF67BDEE414AAB9ULL, "sceGnmSubmitDone", "libSceGnm", vblank_wait_handler);

    // sceKernelGetProcessTimeCounter(void) -> monotonic microseconds (1MHz by
    // construction). The Frequency NID is unknown and was never called; absolute
    // rate is a follow-up if the guest ever queries it. Deltas advance in wall
    // time, which is what unblocks guest timeout spins.
    // NID: 0x7E0C6731E4CD52D6ULL (was mislabeled GetCompletedLabelAddress)
    register_symbol(0x7E0C6731E4CD52D6ULL, "sceKernelGetProcessTimeCounter", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const u64 now_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            ctx.rax = now_us;
        });

    // sceKernelSignalSema(SceKernelSema sema, int count)
    // NID 0xE1CCE9A47062AE2CULL hashes to THIS name (verified against the Orbis
    // salt), not sceGnmWaitLabel (which hashes to 0x3F7163E87D6A528D, uncalled).
    // Every guest signal was being swallowed by the old label stub — it even
    // stored the count arg into the GPU fence. Reuses the shared handler.
    register_symbol(0xE1CCE9A47062AE2CULL, "sceKernelSignalSema", "libkernel", signal_sema_handler);

    // sceGnmSubmitAndFlipCommandBuffers(...) — the REAL flip submit.
    // NID: 0xC5BC4D6AD6B0A217ULL
    // Buffer-index layout TBD: logs all regs (cf. VideoOutSubmitFlip shape) and
    // advances the fence + signals flip waiters. No pixel upload is attempted
    // until the index register is confirmed — uploading the wrong buffer would
    // present garbage.
    register_symbol(0xC5BC4D6AD6B0A217ULL, "sceGnmSubmitAndFlipCommandBuffers", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::fprintf(stderr, "[Gnm] SubmitAndFlip(rdi=0x%llx, rsi=0x%llx, rdx=0x%llx, rcx=0x%llx, r8=0x%llx, r9=0x%llx)\n",
                         (unsigned long long)ctx.rdi, (unsigned long long)ctx.rsi,
                         (unsigned long long)ctx.rdx, (unsigned long long)ctx.rcx,
                         (unsigned long long)ctx.r8, (unsigned long long)ctx.r9);
            g_gnm_completed_label.fetch_add(1);
            sync_gnm_label_slot(mem);
            if (g_thread_manager) {
                g_thread_manager->signal_flip_event(static_cast<i64>(ctx.rcx));
            }
            ctx.rax = SCE_OK;
        });

    // sceGnmEndWorkload — closes a GPU workload; advance the fence only.
    // (Was mislabeled SubmitAndFlip: the old body presented + signaled flip here,
    // a full frame early. Real presentation lives in 0xC5BC... below.)
    // NID: 0x15ADF1EF938E2D10ULL
    register_symbol(0x15ADF1EF938E2D10ULL, "sceGnmEndWorkload", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager& mem) {
            g_gnm_completed_label.fetch_add(1);
            sync_gnm_label_slot(mem);
            ctx.rax = SCE_OK;
        });

    // sceSaveDataInitialize3 — stub returning success; real save-data calls later.
    // NID: 0x4F2C2B14A0A82C66ULL
    register_symbol(0x4F2C2B14A0A82C66ULL, "sceSaveDataInitialize3", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // sceVideoOutModeSetAny_ — initializes the mode struct at rdi.
    // NID: 0xA63903B20C658BA7ULL
    // First pass: zero-fill 64 bytes (exact size TBD from the ConfigureOutputMode_
    // struct dump below).
    register_symbol(0xA63903B20C658BA7ULL, "sceVideoOutModeSetAny_", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mode_ptr = ctx.rdi;
            if (mode_ptr > 0x1000 && mem.is_valid_address(mode_ptr, 64)) {
                mem.zero_memory(mode_ptr, 64);
            }
            ctx.rax = SCE_OK;
        });

    // sceGnmAddEqEvent — record the registration; full EOP routing once the arg
    // layout is known from the first-call log below.
    // NID: 0x6F4C729659D563F2ULL (was mislabeled sceGnmInitialize in the stub loop)
    register_symbol(0x6F4C729659D563F2ULL, "sceGnmAddEqEvent", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager&) {
            std::fprintf(stderr, "[Gnm] AddEqEvent(rdi=0x%llx, rsi=0x%llx, rdx=0x%llx, rcx=0x%llx, r8=0x%llx, r9=0x%llx)\n",
                         (unsigned long long)ctx.rdi, (unsigned long long)ctx.rsi,
                         (unsigned long long)ctx.rdx, (unsigned long long)ctx.rcx,
                         (unsigned long long)ctx.r8, (unsigned long long)ctx.r9);
            ctx.rax = SCE_OK;
        });
}

} // namespace ps4

