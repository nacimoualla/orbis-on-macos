#include "ps4/audio_manager.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/thread.hpp"
#include "ps4/vfs.hpp"
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
};
static std::mutex s_sema_table_mtx;
static std::unordered_map<u32, std::shared_ptr<GuestSema>> s_semas;
static u32 s_next_sema_id = 1;

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
static constexpr GuestSize kMonoArenaSize = 64 * 1024; // 256 slots
static constexpr GuestSize kMonoSlotSize = 256;

// Caller holds s_mono_arena.mtx. Returns a zeroed slot, or 0 when exhausted.
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
        mem.zero_memory(cur, kMonoArenaSize);
    }
    if (s_mono_arena.next + kMonoSlotSize > s_mono_arena.end) {
        std::fprintf(stderr, "[Mono] arena exhausted (256 slots)\n");
        return 0;
    }
    const GuestAddr slot = s_mono_arena.next;
    s_mono_arena.next += kMonoSlotSize;
    return slot;
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

void NidRegistry::shutdown() {
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

void NidRegistry::note_hle_dispatch(Nid nid) const {
    if (g_thread_manager) {
        const ScePthreadId tid = g_thread_manager->current_thread_id();
        if (GuestThread* th = g_thread_manager->get_thread(tid)) {
            th->last_nid.store(nid, std::memory_order_relaxed);
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
            std::cout << "[HLE] sceKernelExitProcess(" << status << ")\n";
            ctx.rax = 0;
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

    // scePthreadMutexLock(ScePthreadMutex* mutex)
    register_symbol(0xE86895318FDC64A0ULL, "scePthreadMutexLock", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 mid = 0;
            if (mem.read_value(mtx_ptr, mid)) {
                if (mid == 0) { // Auto-init on zeroed memory
                    mid = tm.mutex_create();
                    mem.write_value(mtx_ptr, mid);
                }
                ctx.rax = tm.mutex_lock(mid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadMutexTrylock(ScePthreadMutex* mutex)
    // NID: 0xBA9A15AF330715E1ULL ("upoVrzMHFeE")
    register_symbol(0xBA9A15AF330715E1ULL, "scePthreadMutexTrylock", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 mid = 0;
            if (mem.read_value(mtx_ptr, mid)) {
                if (mid == 0) {
                    mid = tm.mutex_create();
                    mem.write_value(mtx_ptr, mid);
                }
                ctx.rax = tm.mutex_trylock(mid) ? SCE_OK : static_cast<u64>(-16); // EBUSY
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadMutexUnlock(ScePthreadMutex* mutex)
    register_symbol(0xEFB2023594895C85ULL, "scePthreadMutexUnlock", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 mid = 0;
            if (mem.read_value(mtx_ptr, mid)) {
                ctx.rax = tm.mutex_unlock(mid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
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
    // NID: 0xDE483BAD3D0D408BULL
    register_symbol(0xDE483BAD3D0D408BULL, "scePthreadExit", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            auto& tm = get_thread_manager(mem);
            tm.exit_current_thread(ctx.rdi);
            ctx.rip = 0;
            ctx.rax = SCE_OK;
        });

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

    // puts(const char* s)
    register_symbol(0x5C0F663677A000A8ULL, "puts", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr str_addr = ctx.rdi;
            if (mem.is_valid_address(str_addr)) {
                const char* str = mem.to_host_ptr<char>(str_addr);
                if (str) {
                    std::cout << str << "\n";
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
        g_guest_heap_current = (g_guest_heap_current + 15) & ~15ULL;
        const GuestAddr allocated = g_guest_heap_current;
        g_guest_heap_current += size;
        ctx.rax = allocated;
    };
    register_symbol(0x9E990E49EC698A4DULL, "malloc", "libc", malloc_handler);
    register_symbol(0x8105FEE060D08E93ULL, "malloc", "libc", malloc_handler);

    // free(void* ptr)
    auto free_handler = [](CpuContext& ctx, MemoryManager&) {
        ctx.rax = 0;
    };
    register_symbol(0x7E16B7B8361BD5BBULL, "free", "libc", free_handler);
    register_symbol(0xB4886CAA3D2AB051ULL, "free", "libc", free_handler);
    register_symbol(0x63B689D6EC9D3CCAULL, "free", "libc", free_handler);

    // calloc(size_t num, size_t size)
    // Note: no explicit zeroing — see malloc above; fresh bump pages are OS-zero.
    register_symbol(0xB50228F8620F925AULL, "calloc", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const GuestSize num = ctx.rdi;
            const GuestSize size = ctx.rsi;
            const GuestSize total = num * size;
            g_guest_heap_current = (g_guest_heap_current + 15) & ~15ULL;
            const GuestAddr allocated = g_guest_heap_current;
            g_guest_heap_current += total;
            ctx.rax = allocated;
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

    // abort()
    // NID: 0x2F54814E40BE0AFCULL / 0xFC0ABE404E81542FULL ("L1SBTkC+Cvw")
    auto abort_handler = [](CpuContext& ctx, MemoryManager&) {
        std::fprintf(stderr, "[HLE] Guest called abort() at caller RIP=0x%llx\n", (unsigned long long)ctx.rip);
        ctx.rax = -1;
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
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (cur_th) cur_th->state.store(ThreadState::Running);
                ctx.rax = SCE_OK;
                return;
            }

            std::unique_lock<std::mutex> lock(sema->mtx);
            auto pred = [&]() {
                return sema->count >= need || sema->deleted || (g_syscalls && g_syscalls->has_exited());
            };

            if (timo_ptr != 0 && mem.is_valid_address(timo_ptr, sizeof(u32))) {
                u32 usec = 0;
                mem.read_value(timo_ptr, usec);
                std::fprintf(stderr, "[Sema] WaitSema(%u, need=%d, timeout=%uus)\n", id, need, usec);
                if (!sema->cv.wait_for(lock, std::chrono::microseconds(usec), pred)) {
                    if (cur_th) cur_th->state.store(ThreadState::Running);
                    ctx.rax = 0x8002003CULL; // SCE_KERNEL_ERROR_ETIMEDOUT
                    return;
                }
            } else {
                std::fprintf(stderr, "[Sema] WaitSema(%u, need=%d, timeout=infinite)\n", id, need);
                while (!pred()) {
                    sema->cv.wait_for(lock, std::chrono::milliseconds(50), pred);
                }
            }

            if (sema->count >= need) {
                sema->count -= need;
            }
            if (cur_th) cur_th->state.store(ThreadState::Running);
            ctx.rax = SCE_OK;
        });

    // sceKernelSignalSema(SceKernelSema sema, int signalCount)
    // NIDs: 0x443BB032833D9F45ULL ("RDuwMoM9n0U") and 0x3C4AC6259839E5EBULL
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
            std::fprintf(stderr, "[Sema] SignalSema(%u, %d) -> unknown id\n", id, sig);
        }
        ctx.rax = SCE_OK;
    };
    register_symbol(0x443BB032833D9F45ULL, "sceKernelSignalSema", "libkernel", signal_sema_handler);
    register_symbol(0x3C4AC6259839E5EBULL, "sceKernelSignalSema", "libkernel", signal_sema_handler);

    // sceKernelPollSema(SceKernelSema sema, int needCount)
    // NID: 0x0528B1D7D1C6C155ULL ("BSix19HGwVU")
    register_symbol(0x0528B1D7D1C6C155ULL, "sceKernelPollSema", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
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
                std::fprintf(stderr, "[Sema] PollSema(%u, need=%d) -> BUSY (count=%d)\n", id, need, sema->count);
            } else {
                std::fprintf(stderr, "[Sema] PollSema(%u, need=%d) -> unknown id\n", id, need);
            }
            ctx.rax = 0x80020010ULL; // SCE_KERNEL_ERROR_EBUSY
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


    // memalign(size_t alignment, size_t size)
    // NID: 0x5237F72B332F4662ULL ("Ujf3KzMvRmI")
    register_symbol(0x5237F72B332F4662ULL, "memalign", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const GuestSize alignment = ctx.rdi > 0 ? ctx.rdi : 16;
            const GuestSize size = ctx.rsi;
            g_guest_heap_current = (g_guest_heap_current + alignment - 1) & ~(alignment - 1);
            const GuestAddr allocated = g_guest_heap_current;
            g_guest_heap_current += size;
            ctx.rax = allocated;
        });

    // mspace_memalign(mspace msp, size_t alignment, size_t size)
    // NID: 0x885D62407CF10495ULL ("iF1iQHzxBJU")
    register_symbol(0x885D62407CF10495ULL, "mspace_memalign", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const GuestSize alignment = ctx.rsi > 0 ? ctx.rsi : 16;
            const GuestSize size = ctx.rdx;
            g_guest_heap_current = (g_guest_heap_current + alignment - 1) & ~(alignment - 1);
            const GuestAddr allocated = g_guest_heap_current;
            g_guest_heap_current += size;
            ctx.rax = allocated;
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
    // NID: 0xB6CBC49A77A7CF8FULL ("tsvEmnenz48")
    register_symbol(0xB6CBC49A77A7CF8FULL, "__cxa_atexit", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0; // Success
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

    // Archive Mount stubs (return negative so engine uses loose files on /app0)
    register_symbol(0xF8546F2A49D48F52ULL, "sceFiosArchiveMountWithOrder", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = static_cast<u64>(-1);
        });
    register_symbol(0x186AAE70BF45F982ULL, "sceFiosArchiveMountWithOrderSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = static_cast<u64>(-1);
        });
    register_symbol(0xB8292024EAD850BEULL, "sceFiosArchiveGetMountBufferSize", "libSceFios2",
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

    // scePadInit()
    register_symbol(0x7501D612C26DA04EULL, "scePadInit", "libScePad",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePadOpen(int userId, int type, int index, const void* param)
    // NID: 0xC64D0071AACFDD5EULL ("xk0AcarP3V4")
    register_symbol(0xC64D0071AACFDD5EULL, "scePadOpen", "libScePad",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0x200; // valid pad handle
        });

    // scePadClose(int handle)
    register_symbol(0x8F521313F1282661ULL, "scePadClose", "libScePad",
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
    register_symbol(0xB445383A59BDC10BULL, "scePadReadState", "libScePad", pad_read_state_handler);
    register_symbol(0xA43B8F11FDE6E1F2ULL, "scePadReadState", "libScePad", pad_read_state_handler);

    // scePadSetLightBar(int handle, const ScePadLightBarParam* param)
    register_symbol(0xD4143F8A8CA539BCULL, "scePadSetLightBar", "libScePad",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePadSetVibration(int handle, const ScePadVibrationParam* param)
    register_symbol(0x3B184807C2C1FCF4ULL, "scePadSetVibration", "libScePad",
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
    register_symbol(0xEC7D224CE7224CBAULL, "pthread_mutex_lock", "mono-ps4",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // pthread_mutex_unlock (reached via mono)
    register_symbol(0xD99F8FA58E826898ULL, "pthread_mutex_unlock", "mono-ps4",
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
    register_symbol(0x83C70CDFD11467AAULL, "g8cM39EUZ6o", "libSceAudio3d",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    register_symbol(0x5260AF8D29AE648CULL, "UmCvjSmuZIw", "libSceAudio3d",
        [](CpuContext& ctx, MemoryManager&) {
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

    // lUoqwTQu4Go
    register_symbol(0x954A2AC1342EE06AULL, "lUoqwTQu4Go", "libSceUserService",
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

    // Orbis puts alias used by Unity Launcher Control (NID: 0x610D276AFA7E6087ULL)
    register_symbol(0x610D276AFA7E6087ULL, "puts_orbis", "libc",
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
                // Fall back to the first image when none is supplied: the
                // cookies are interchangeable zeros, and a NULL image here
                // otherwise retries forever upstream.
                GuestAddr eff_image = (image != 0) ? image : s_mono_arena.first_image;
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
    // NID: 0x1147722929EB0A4CULL — reflexive identity only; distinct cookies
    // stay 0 so unrelated types are never misclassified.
    register_symbol(0x1147722929EB0A4CULL, "mono_class_is_subclass_of", "mono-ps4",
        [](CpuContext& ctx, MemoryManager&) {
            const GuestAddr klass = ctx.rdi;
            const GuestAddr parent = ctx.rsi;
            ctx.rax = (klass != 0 && klass == parent) ? 1 : 0;
        });

    // mono_runtime_invoke(MonoMethod* method, void* obj, void** params, MonoObject** exc)
    // NID: 0xA6B7BD0639320E1BULL — no IL execution; clears *exc so the caller
    // never reads uninitialized stack garbage as a managed exception.
    register_symbol(0xA6B7BD0639320E1BULL, "mono_runtime_invoke", "mono-ps4",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr exc_ptr = ctx.rcx;
            if (exc_ptr > 0x1000 && mem.is_valid_address(exc_ptr, sizeof(u64))) {
                mem.write_value(exc_ptr, static_cast<u64>(0));
            }
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

    // Unity memory allocator, FMOD, and Mono module setup stubs
    for (Nid stub_nid : {
             0x2AE3AE0F9F21AA7EULL, // onion_sys_allocator naming / query
             0xA4EF7A4F0CCE9B91ULL, // virtual query helper
             0x04DA30C76979F3C1ULL, // init helper
             0xFE99E3DFFEDAE900ULL, // FMOD / pad init helper
             0x8941B88CB45FD196ULL, // FMOD mixer thread setup
             0x54EC7C3469875D3BULL, // thread prio / attr helper
             0xFEEC04868FFB80EBULL, // thread attr helper
             0x22EC6752E5E4E818ULL, // mono-ps4.prx post-load query
             0xFE19F5B5C547AB94ULL, // ALLOC_GFX virtual range naming
             0x960213B81B11A36AULL, // debug.log stream setup
             0x47D940F363AB68DBULL, // Unity init helper
             0x55EED6C524805364ULL, // Unity string/domain setup 1
             0xC5D3D91AED17EB3BULL, // Unity string/domain setup 2
             0x4FCEF2B219D790C5ULL,
             0x72CE3B9DE94CEA50ULL,
             0x46204874F5E83DD1ULL,
             0x3B579C5F88AEC0EFULL,
             0xFEAF6C4747115067ULL,
             0xE1E3B7F43BAC2E92ULL,
             0xF737BFA02980EDF6ULL,
             0x5AA1DFAA34975749ULL,
             0xF068CCD895AE7A94ULL,
             0xB8369AEA2E0EF71AULL,
             0x23FA65EA7B52AD86ULL,
             0x9141F1F3ED8B6742ULL,
             0x0E04B6E6017A7AE4ULL,
             0x7050A9D0D5FCC1FDULL, // libSceGnm init / draw command buffer
             0x8008429FA5225386ULL, // libSceGnm state / shader setup
             0xE6E14A7248896113ULL, // libSceGnm state / constant setup
             0x186B27EE3313C70EULL, // libSceGnm context setup
             0xFD98F4C8AD37832AULL, // unknown attr helper (NOT Setschedparam; real one is 0x0F3112F61405E1FE)
             0xD8866B1E72E722DDULL, // Unity init stub
             0x066E40B33FB6E36FULL, // Unity argv/init stub
             0x86FD65BA226BA903ULL, // Audio3d / Ajm stub
             0x42CD305AE96097B5ULL, // Audio3d / Ajm stub
             0x9839A030E19552A8ULL, // mono mutexattr settype
             0xE6DC4A7DC3140289ULL, // mono mutexattr setprotocol
             0xB6D1CD7D4FAA0C15ULL, // mono mutex init
             0x1C5EE52B8EB1CE36ULL, // mono mutexattr destroy
             0xD13C959383122EDDULL, // mono cond init
             0x9AA50B35D8A64E7DULL, // mono TLS / key create
             0x3AEDE22F569BBE78ULL, // scePthreadSetprio (called right after scePthreadSetaffinity)
             0x967DF7CE306B7E39ULL, // sceGnmCreateContext
             0xE6E7409BEE9BA158ULL, // sceGnmRegisterResource
             0x8A1C6B6ECA122967ULL, // sceGnmUnregisterResource
             0xC9BD9C4616A00F52ULL, // sceGnmFlushGarlic
             0xD6A5CB1C8A5138F1ULL, // sceGnmValidation
             0x5B512D8FF8E55BB6ULL, // sceGnmGetResourceBudget
             0xEEA65536012EF926ULL, // sceGnmSetCsShader
             0x8E222DCD2EBEDB68ULL  // sceGnmUpdateResource
         }) {
        register_symbol(stub_nid, "sceKernelSyncStub", "libkernel",
            [](CpuContext& ctx, MemoryManager&) {
                ctx.rax = SCE_OK;
            });
    }

    // sceVideoOutWaitVblank — sleep instead of spinning so the guest yields the
    // host thread at ~60 FPS VBlank cadence.
    auto vblank_wait_handler = [](CpuContext& ctx, MemoryManager&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS VBlank
        ctx.rax = SCE_OK;
    };
    register_symbol(0xCAF67BDEE414AAB9ULL, "sceVideoOutWaitVblank", "libSceVideoOut", vblank_wait_handler);

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

