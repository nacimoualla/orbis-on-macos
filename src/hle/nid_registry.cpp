#include "ps4/audio_manager.hpp"
#include "ps4/dynamic_linker.hpp"
#include "ps4/executor.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/np_trophy.hpp"
#include "ps4/guest_libc_blob.hpp"
#include "ps4/pm4_translator.hpp"
#include "ps4/save_data.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/thread.hpp"
#include "ps4/vfs.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include <thread>

// PS4_LOG_PM4=1: trace every PM4 packet, draw and shader. Off by default:
// at 60 FPS it is megabytes per second, and when stderr is a pipe to a slow
// terminal the writes block the guest's GPU thread.
static bool pm4_verbose() {
    static const bool on = std::getenv("PS4_LOG_PM4") != nullptr;
    return on;
}
#define PM4_LOG(...) do { if (pm4_verbose()) std::fprintf(stderr, __VA_ARGS__); } while (0)

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwctype>

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

// The libc heap: a bump pointer plus size-keyed free lists. There are no block
// headers (realloc and free look sizes up here instead), and freed blocks are
// reused best-fit: with free() a no-op a long session walked the bump pointer
// into the 1 GB ceiling and every later malloc returned NULL.
//
// All three are leaked on purpose: guest threads keep allocating while the host
// tears down at exit, and a destroyed mutex throws "mutex lock failed: Invalid
// argument" out of bump_alloc (Flappy's Unity workers do this through the
// sceLibcMspace* handlers).
static std::mutex& s_heap_sizes_mtx = *new std::mutex;                                   // guards everything below
static auto& s_heap_sizes = *new std::unordered_map<GuestAddr, GuestSize>;              // live block -> size
static auto& s_heap_free = *new std::multimap<GuestSize, GuestAddr>;                    // size -> free block

// Hard ceiling for the CPU heap. The memory map reserves 0x20000000-0x40000000
// for the guest libc heap and 0x40000000-0x1C0000000 for direct (Garlic/ONION)
// memory; an unbounded bump allocator walks straight through that boundary and
// silently overwrites GPU buffers with heap contents. See mspace_alloc below.
static constexpr GuestAddr kCpuHeapEnd = 0x40000000ULL;

// Round to 16 bytes up to 4 KB, then to eighths of the size's power of two,
// so a freed block fits later requests of about the same size.
inline GuestSize heap_round(GuestSize size) {
    if (size <= 4096) return (std::max<GuestSize>(size, 1) + 15) & ~GuestSize{15};
    const unsigned shift = 63u - static_cast<unsigned>(__builtin_clzll(size)) - 3u;
    const GuestSize gran = GuestSize{1} << shift;
    return (size + gran - 1) & ~(gran - 1);
}

inline GuestAddr bump_alloc(GuestSize size, GuestSize align = 16) {
    if (align < 16) align = 16;
    const GuestSize rs = heap_round(size);
    std::lock_guard<std::mutex> lock(s_heap_sizes_mtx);
    // Reuse a free block of this size class (up to 25% larger) if one is
    // suitably aligned.
    int tries = 0;
    for (auto it = s_heap_free.lower_bound(rs); it != s_heap_free.end() && it->first <= rs + rs / 4 && tries < 64;
         ++it, ++tries) {
        if ((it->second & (align - 1)) != 0) continue;
        const GuestAddr p = it->second;
        s_heap_sizes[p] = it->first;
        s_heap_free.erase(it);
        return p;
    }
    const GuestAddr cur = g_guest_heap_current.load();
    const GuestAddr p = (cur + align - 1) & ~(align - 1);
    if (p + rs > kCpuHeapEnd) {
        std::fprintf(stderr, "[heap] CPU heap exhausted: need %llu at 0x%llx, ceiling 0x%llx\n",
                     (unsigned long long)size, (unsigned long long)p,
                     (unsigned long long)kCpuHeapEnd);
        return 0;
    }
    g_guest_heap_current.store(p + rs);
    s_heap_sizes[p] = rs;
    return p;
}

inline GuestSize bump_size_of(GuestAddr ptr) {
    std::lock_guard<std::mutex> lock(s_heap_sizes_mtx);
    auto it = s_heap_sizes.find(ptr);
    return it == s_heap_sizes.end() ? 0 : it->second;
}

// Return a block to the free lists. Pointers this heap did not hand out
// (mspace, mmap, module data) and double frees are ignored.
inline void heap_free(GuestAddr ptr) {
    if (ptr == 0) return;
    std::lock_guard<std::mutex> lock(s_heap_sizes_mtx);
    auto it = s_heap_sizes.find(ptr);
    if (it == s_heap_sizes.end()) return;
    s_heap_free.emplace(it->second, ptr);
    s_heap_sizes.erase(it);
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

// Anonymous/file mmap arena for loaded modules (mono-ps4.prx's GC maps its heap
// sections here). Sits between the module region (0x04000000..0x10000000) and
// the CPU bump heap at 0x20000000, clear of direct memory and the mspace arena.
// Never reused: munmap only zeroes, like the other bump arenas.
static constexpr GuestAddr kMmapArenaBase = 0x10000000ULL;
static constexpr GuestAddr kMmapArenaEnd = 0x20000000ULL;
static std::atomic<GuestAddr> g_mmap_current{kMmapArenaBase};

inline GuestAddr mmap_arena_alloc(GuestSize size) {
    const GuestSize len = (size + GUEST_PAGE_SIZE - 1) & ~(GUEST_PAGE_SIZE - 1);
    GuestAddr cur = g_mmap_current.load();
    do {
        if (cur + len > kMmapArenaEnd) return 0;
    } while (!g_mmap_current.compare_exchange_weak(cur, cur + len));
    return cur;
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

static std::mutex g_guest_stdio_mtx;
static std::unordered_set<uintptr_t> g_guest_open_files;

inline FILE* resolve_guest_stream(GuestAddr addr) {
    if (addr == 0) return nullptr;
    if (addr == 1 || (addr >= 0x3FF2200ULL && addr < 0x3FF2400ULL)) return stdout;
    if (addr == 2 || (addr >= 0x3FF2400ULL && addr < 0x3FF2600ULL)) return stderr;
    if (addr >= 0x3FF2000ULL && addr < 0x3FF2200ULL) return stdin;

    std::lock_guard<std::mutex> lock(g_guest_stdio_mtx);
    if (g_guest_open_files.contains(static_cast<uintptr_t>(addr))) {
        return reinterpret_cast<FILE*>(addr);
    }
    // Safe fallback for unmapped streams or trampolines: write to stderr rather than crashing host flockfile
    return stderr;
}

// PS4_LOG_SEMA=1: log every sceKernelWaitSema/SignalSema (FMOD hits these
// hundreds of times a second, so they are off by default).
static bool sema_log_enabled() {
    static const bool on = std::getenv("PS4_LOG_SEMA") != nullptr;
    return on;
}

class AudioDeadlinePacer {
public:
    void pace(u32 grain_samples, u32 sample_rate = 48000) {
        if (grain_samples == 0 || sample_rate == 0) return;
        const auto now = std::chrono::steady_clock::now();
        const auto step = std::chrono::nanoseconds(static_cast<u64>(grain_samples) * 1'000'000'000ULL / sample_rate);
        if (m_next.time_since_epoch().count() == 0 || (now - m_next) > std::chrono::milliseconds(100)) {
            m_next = now;
        }
        m_next += step;
        if (m_next > now) {
            std::this_thread::sleep_until(m_next);
        }
    }
private:
    std::chrono::steady_clock::time_point m_next{};
};

struct FlipState {
    std::atomic<u64> flip_count{0};
    std::atomic<i64> last_flip_arg{0};
    std::atomic<u32> current_buffer{0};
    std::atomic<u64> last_flip_time_us{0};
};
static FlipState s_flip_state;
constexpr u64 kVideoOutEventFlip = 0; // SCE_VIDEO_OUT_EVENT_FLIP (VBLANK is 1)

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

struct PMutex {
    std::mutex m;
    std::condition_variable cv;
    std::thread::id owner{};
    int count{0};
};
struct PCond {
    std::mutex m;
    std::condition_variable cv;
    u64 seq{0};
};
static std::mutex s_ptab_mtx;
static std::unordered_map<GuestAddr, std::shared_ptr<PMutex>> s_pmutexes;
static std::unordered_map<GuestAddr, std::shared_ptr<PCond>> s_pconds;

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
        std::lock_guard<std::mutex> heap_lock(s_heap_sizes_mtx);
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
        std::fprintf(stderr, "\n[FATAL] Mono arena exhausted! Hit ceiling of 4096 slots (1 MB). "
                             "Expand kMonoArenaSize in src/hle/nid_registry.cpp.\n");
        std::fflush(stderr);
        std::abort();
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
    dump_write_watch_ring(write_watch_addr());
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

// Run the calling guest thread's pending exception handlers (see
// ThreadManager::deliver_pending_exceptions). Wait loops call this with their
// locks released, so a thread blocked in an HLE wait still answers, e.g., the
// GC's suspend signal.
static void deliver_guest_exceptions(MemoryManager& mem) {
    if (CpuExecutor* ex = current_executor()) get_thread_manager(mem).deliver_pending_exceptions(*ex);
}

// True when a guest exception is queued for the calling thread.
static bool guest_signal_pending() {
    const GuestThread* th = ThreadManager::current_guest_thread();
    return th != nullptr && th->has_pending_signals.load(std::memory_order_acquire);
}

// Publishes the condition variable a wait loop sleeps on, so raise_exception
// can wake it (see GuestThread::wait_cv).
struct SignalWakeScope {
    GuestThread* th;
    explicit SignalWakeScope(std::condition_variable& cv) : th(ThreadManager::current_guest_thread()) {
        if (!th) return;
        std::lock_guard<std::mutex> lock(th->wait_cv_mtx);
        th->wait_cv = &cv;
    }
    ~SignalWakeScope() {
        if (!th) return;
        std::lock_guard<std::mutex> lock(th->wait_cv_mtx);
        th->wait_cv = nullptr;
    }
    SignalWakeScope(const SignalWakeScope&) = delete;
    SignalWakeScope& operator=(const SignalWakeScope&) = delete;
};

// True when an armed write-watch overlaps [addr, addr + n); host-side bulk
// copies route through write_bytes then so the watch records them.
static bool watch_overlaps(GuestAddr addr, GuestSize n) {
    const GuestAddr lo = write_watch_addr(), hi = write_watch_end();
    return lo != 0 && addr < hi && lo < addr + n;
}

NidRegistry::NidRegistry() = default;

void NidRegistry::set_vfs(VirtualFileSystem* vfs) {
    g_external_vfs = vfs;
}

static MetalBackend* g_metal_backend = nullptr;
static std::unique_ptr<Pm4Translator> g_pm4_translator;
static std::mutex g_input_mutex;
static PadState g_active_pad_state{};
static std::atomic<u32> g_pad_pressed_latch{0};
static MouseState g_active_mouse_state{};
// Runs after every guest flip (see NidRegistry::set_flip_hook).
static std::function<void()> g_flip_hook;

static PadState read_pad_state() {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    return g_active_pad_state;
}

// ScePadData (0x78 bytes): buttons, left/right stick (u8 x,y), analog L2/R2,
// orientation quaternion (+0x0C), acceleration (+0x1C), angular velocity
// (+0x28), touch data (+0x34: count, then two {u16 x, u16 y, u8 id} at
// +0x3C/+0x44), connected (+0x4C), timestamp (+0x50).
constexpr GuestSize kScePadDataSize = 0x78;
static void write_pad_data(MemoryManager& mem, GuestAddr p, const PadState& st) {
    mem.zero_memory(p, kScePadDataSize);
    const u32 latched_buttons = st.buttons | g_pad_pressed_latch.exchange(0, std::memory_order_relaxed);
    mem.write_value(p + 0x00, latched_buttons);

    // Left stick
    u8 final_lx = st.lx;
    u8 final_ly = st.ly;
    if (final_lx == 128) {
        if (latched_buttons & kPadRight) final_lx = 255;
        else if (latched_buttons & kPadLeft) final_lx = 0;
    }
    if (final_ly == 128) {
        if (latched_buttons & kPadUp) final_ly = 0;
        else if (latched_buttons & kPadDown) final_ly = 255;
    }
    mem.write_value(p + 0x04, final_lx);
    mem.write_value(p + 0x05, final_ly);

    // Right stick
    mem.write_value(p + 0x06, st.rx);
    mem.write_value(p + 0x07, st.ry);

    // Analog L2/R2
    mem.write_value(p + 0x08, st.l2);
    mem.write_value(p + 0x09, st.r2);

    mem.write_value(p + 0x18, 1.0f); // orientation w (identity)
    mem.write_value(p + 0x20, 1.0f); // acceleration: 1 g along y

    // Touch data
    if (st.touch_down || (latched_buttons & kPadTouchPadClick)) {
        mem.write_value(p + 0x34, static_cast<u8>(1)); // 1 active touch point
        mem.write_value(p + 0x3C, st.touch_x);
        mem.write_value(p + 0x3E, st.touch_y);
    }

    mem.write_value(p + 0x4C, static_cast<u8>(st.connected ? 1 : 0));
    const u64 ts = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    mem.write_value(p + 0x50, ts);
}

void NidRegistry::set_metal_backend(MetalBackend* metal) {
    g_metal_backend = metal;
    if (metal) {
        g_pm4_translator = std::make_unique<Pm4Translator>(metal);
    } else {
        g_pm4_translator.reset();
    }
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
    if (g_syscalls) {
        g_syscalls->set_exited(true);
    }
    s_sem_cv.notify_all();
    {
        std::lock_guard<std::mutex> lock(s_sema_table_mtx);
        for (auto& [id, sema] : s_semas) {
            if (sema) {
                std::lock_guard<std::mutex> slock(sema->mtx);
                sema->deleted = true;
                sema->cv.notify_all();
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(s_eflag_table_mtx);
        for (auto& [id, eflag] : s_eflags) {
            if (eflag) {
                std::lock_guard<std::mutex> elock(eflag->mtx);
                eflag->deleted = true;
                eflag->cv.notify_all();
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(s_ptab_mtx);
        for (auto& [addr, pm] : s_pmutexes) {
            if (pm) {
                std::lock_guard<std::mutex> mlock(pm->m);
                pm->cv.notify_all();
            }
        }
        for (auto& [addr, pc] : s_pconds) {
            if (pc) {
                std::lock_guard<std::mutex> clock(pc->m);
                pc->cv.notify_all();
            }
        }
    }
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
    if (g_pm4_translator) {
        g_pm4_translator.reset();
    }
}

ThreadManager& NidRegistry::thread_manager(MemoryManager& mem) {
    return get_thread_manager(mem);
}

ThreadManager* NidRegistry::thread_manager() const noexcept {
    return g_thread_manager.get();
}

void NidRegistry::note_hle_dispatch(Nid nid, GuestAddr caller_rip) const {
    if (GuestThread* th = ThreadManager::current_guest_thread()) {
        th->last_nid.store(nid, std::memory_order_relaxed);
        th->hle_caller_rip.store(caller_rip, std::memory_order_relaxed);
        th->hle_calls.fetch_add(1, std::memory_order_relaxed);
        th->in_hle.store(true, std::memory_order_relaxed);
    }
}

void NidRegistry::note_hle_return() const {
    if (GuestThread* th = ThreadManager::current_guest_thread()) {
        th->in_hle.store(false, std::memory_order_relaxed);
    }
}

void NidRegistry::set_flip_hook(std::function<void()> hook) {
    g_flip_hook = std::move(hook);
}

void NidRegistry::set_pad_state(const PadState& state) {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    const u32 prev_buttons = g_active_pad_state.buttons;
    g_active_pad_state = state;
    g_pad_pressed_latch.fetch_or(state.buttons & ~prev_buttons, std::memory_order_relaxed);
}

PadState NidRegistry::get_pad_state() const noexcept {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    return g_active_pad_state;
}

void NidRegistry::set_pad_buttons(u32 buttons) {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    const u32 prev = g_active_pad_state.buttons;
    g_active_pad_state.buttons = buttons;
    g_pad_pressed_latch.fetch_or(buttons & ~prev, std::memory_order_relaxed);
}

void NidRegistry::set_mouse_state(const MouseState& state) {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    g_active_mouse_state = state;
}

MouseState NidRegistry::get_mouse_state() const noexcept {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    return g_active_mouse_state;
}

u64 NidRegistry::get_flip_count() const noexcept {
    return s_flip_state.flip_count.load(std::memory_order_relaxed);
}

Pm4Translator* NidRegistry::gpu_translator() const noexcept {
    return g_pm4_translator.get();
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

// ---------------------------------------------------------------------------
// Guest printf formatting
//
// Arguments come from a GuestArgs source: a SysV va_list (vsnprintf & co.) or
// a variadic call's live registers + stack (printf & co.). Each conversion is
// handed to the host's snprintf with its own flags/width/precision, so the
// output matches libc. The old formatter took four integer arguments and
// dropped widths: Mono's GUID format ("%02X" x 16) came out unpadded and
// truncated, so every image GUID mismatched its AOT module.
// ---------------------------------------------------------------------------
class GuestArgs {
public:
    virtual ~GuestArgs() = default;
    virtual u64 next_gp() = 0;
    virtual double next_fp() = 0;
};

// SysV va_list: {u32 gp_offset, u32 fp_offset, void* overflow_arg_area,
// void* reg_save_area}. GP registers live at reg_save_area[0..48), XMM
// registers at [48..176) in 16-byte slots; the rest spill to the overflow area.
class VaListArgs final : public GuestArgs {
public:
    VaListArgs(const MemoryManager& mem, GuestAddr ap) : m_mem(mem) {
        if (ap != 0 && mem.is_valid_address(ap, 24)) {
            mem.read_value(ap + 0, m_gp);
            mem.read_value(ap + 4, m_fp);
            mem.read_value(ap + 8, m_overflow);
            mem.read_value(ap + 16, m_save);
        } else {
            m_gp = 48;
            m_fp = 176;
        }
    }
    u64 next_gp() override {
        u64 v = 0;
        if (m_gp < 48) {
            m_mem.read_value(m_save + m_gp, v);
            m_gp += 8;
        } else {
            m_mem.read_value(m_overflow, v);
            m_overflow += 8;
        }
        return v;
    }
    double next_fp() override {
        u64 bits = 0;
        if (m_fp < 176) {
            m_mem.read_value(m_save + m_fp, bits);
            m_fp += 16;
        } else {
            m_mem.read_value(m_overflow, bits);
            m_overflow += 8;
        }
        double d;
        std::memcpy(&d, &bits, sizeof(d));
        return d;
    }

private:
    const MemoryManager& m_mem;
    u32 m_gp{0}, m_fp{0};
    GuestAddr m_overflow{0}, m_save{0};
};

// The variadic part of a call that just entered an HLE handler: GP registers
// from index `first_gp` (rdi=0 .. r9=5), XMM0-7, then stack arguments above
// the return address at [rsp].
class CallArgs final : public GuestArgs {
public:
    CallArgs(const CpuContext& ctx, const MemoryManager& mem, int first_gp)
        : m_ctx(ctx), m_mem(mem), m_gp(first_gp), m_stack(ctx.rsp + 8) {}
    u64 next_gp() override {
        const u64 regs[6] = {m_ctx.rdi, m_ctx.rsi, m_ctx.rdx, m_ctx.rcx, m_ctx.r8, m_ctx.r9};
        if (m_gp < 6) return regs[m_gp++];
        return from_stack();
    }
    double next_fp() override {
        u64 bits = m_fp < 8 ? m_ctx.ymm[m_fp++].q[0] : from_stack();
        double d;
        std::memcpy(&d, &bits, sizeof(d));
        return d;
    }

private:
    u64 from_stack() {
        u64 v = 0;
        m_mem.read_value(m_stack, v);
        m_stack += 8;
        return v;
    }
    const CpuContext& m_ctx;
    const MemoryManager& m_mem;
    int m_gp;
    int m_fp{0};
    GuestAddr m_stack;
};

static std::string guest_cstring(const MemoryManager& mem, GuestAddr p, usize limit) {
    std::string out;
    if (p == 0) return "(null)";
    for (usize i = 0; i < limit; ++i) {
        char c = 0;
        if (!mem.read_value(p + i, c) || c == 0) break;
        out.push_back(c);
    }
    return out;
}

static std::string format_guest(const MemoryManager& mem, const char* fmt, GuestArgs& args) {
    std::string out;
    if (!fmt) return out;
    out.reserve(256);
    for (usize i = 0; fmt[i] != '\0'; ++i) {
        if (fmt[i] != '%') {
            out.push_back(fmt[i]);
            continue;
        }
        if (fmt[i + 1] == '%') {
            out.push_back('%');
            ++i;
            continue;
        }
        // %[flags][width][.precision][length]conv -> host spec with '*'s
        // resolved to numbers and the length modifier normalised.
        std::string spec = "%";
        ++i;
        while (std::strchr("-+ #0'", fmt[i]) != nullptr && fmt[i] != '\0') spec.push_back(fmt[i++]);
        if (fmt[i] == '*') {
            spec += std::to_string(static_cast<i32>(args.next_gp()));
            ++i;
        } else {
            while (fmt[i] >= '0' && fmt[i] <= '9') spec.push_back(fmt[i++]);
        }
        int precision = -1;
        if (fmt[i] == '.') {
            spec.push_back(fmt[i++]);
            if (fmt[i] == '*') {
                precision = static_cast<i32>(args.next_gp());
                spec += std::to_string(precision);
                ++i;
            } else {
                precision = 0;
                while (fmt[i] >= '0' && fmt[i] <= '9') {
                    precision = precision * 10 + (fmt[i] - '0');
                    spec.push_back(fmt[i++]);
                }
            }
        }
        int size = 0; // 0 = int, 1 = char, 2 = short, 8 = 64-bit
        bool long_double = false;
        for (;;) {
            const char c = fmt[i];
            if (c == 'h') { size = size == 2 ? 1 : 2; ++i; }
            else if (c == 'l' || c == 'j' || c == 'z' || c == 't' || c == 'q') { size = 8; ++i; }
            else if (c == 'L') { long_double = true; ++i; }
            else break;
        }
        const char conv = fmt[i];
        if (conv == '\0') break;
        char buf[512];
        switch (conv) {
        case 'd': case 'i': {
            i64 v = static_cast<i64>(args.next_gp());
            if (size == 0) v = static_cast<i32>(v);
            else if (size == 2) v = static_cast<i16>(v);
            else if (size == 1) v = static_cast<i8>(v);
            std::snprintf(buf, sizeof(buf), (spec + "lld").c_str(), static_cast<long long>(v));
            out += buf;
            break;
        }
        case 'u': case 'o': case 'x': case 'X': {
            u64 v = args.next_gp();
            if (size == 0) v = static_cast<u32>(v);
            else if (size == 2) v = static_cast<u16>(v);
            else if (size == 1) v = static_cast<u8>(v);
            std::snprintf(buf, sizeof(buf), (spec + "ll" + conv).c_str(), static_cast<unsigned long long>(v));
            out += buf;
            break;
        }
        case 'c':
            std::snprintf(buf, sizeof(buf), (spec + "c").c_str(), static_cast<int>(args.next_gp() & 0xFF));
            out += buf;
            break;
        case 's': {
            const GuestAddr p = args.next_gp();
            const std::string str = guest_cstring(mem, p, precision >= 0 ? static_cast<usize>(precision) : 1u << 20);
            const int n = std::snprintf(nullptr, 0, (spec + "s").c_str(), str.c_str());
            std::string tmp(static_cast<usize>(n > 0 ? n : 0) + 1, '\0');
            std::snprintf(tmp.data(), tmp.size(), (spec + "s").c_str(), str.c_str());
            tmp.pop_back();
            out += tmp;
            break;
        }
        case 'p':
            std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(args.next_gp()));
            out += buf;
            break;
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A': {
            // long double (x87 80-bit) is passed in memory; not modelled.
            const double d = long_double ? 0.0 : args.next_fp();
            std::snprintf(buf, sizeof(buf), (spec + conv).c_str(), d);
            out += buf;
            break;
        }
        case 'n':
            (void)args.next_gp(); // never write through a guest %n
            break;
        default:
            out += spec;
            out.push_back(conv);
            break;
        }
    }
    return out;
}

// Guest sscanf: parses a guest string, storing through guest pointers taken
// from `args`. Returns the number of assignments, or -1 on input failure
// before the first conversion (as C does).
static int scan_guest(MemoryManager& mem, const std::string& in, const char* fmt, GuestArgs& args) {
    usize pos = 0;
    int assigned = 0;
    bool any_conversion = false;
    auto ws = [](char c) { return c == ' ' || (c >= '\t' && c <= '\r'); };
    for (usize i = 0; fmt[i] != '\0'; ++i) {
        const char f = fmt[i];
        if (ws(f)) {
            while (pos < in.size() && ws(in[pos])) ++pos;
            continue;
        }
        if (f != '%') {
            if (pos >= in.size() || in[pos] != f) return any_conversion || pos < in.size() ? assigned : -1;
            ++pos;
            continue;
        }
        ++i;
        if (fmt[i] == '%') {
            while (pos < in.size() && ws(in[pos])) ++pos;
            if (pos >= in.size() || in[pos] != '%') return assigned;
            ++pos;
            continue;
        }
        const bool suppress = fmt[i] == '*';
        if (suppress) ++i;
        usize width = 0;
        while (fmt[i] >= '0' && fmt[i] <= '9') width = width * 10 + static_cast<usize>(fmt[i++] - '0');
        int size = 4; // bytes stored for integers
        if (fmt[i] == 'h') { size = 2; ++i; if (fmt[i] == 'h') { size = 1; ++i; } }
        else if (fmt[i] == 'l') { size = 8; ++i; if (fmt[i] == 'l') ++i; }
        else if (fmt[i] == 'j' || fmt[i] == 'z' || fmt[i] == 't' || fmt[i] == 'q') { size = 8; ++i; }
        const char conv = fmt[i];
        if (conv == '\0') break;
        if (conv == 'n') {
            if (!suppress) mem.write_bytes(args.next_gp(), &pos, static_cast<usize>(size));
            continue;
        }
        if (conv != 'c' && conv != '[')
            while (pos < in.size() && ws(in[pos])) ++pos;
        if (pos >= in.size()) return any_conversion ? assigned : -1;
        const usize limit = width ? std::min(in.size(), pos + width) : in.size();
        if (conv == 'd' || conv == 'i' || conv == 'u' || conv == 'x' || conv == 'X' || conv == 'o' || conv == 'p') {
            const int base = (conv == 'd' || conv == 'u') ? 10 : (conv == 'o') ? 8 : (conv == 'i') ? 0 : 16;
            const std::string field = in.substr(pos, limit - pos);
            char* end = nullptr;
            errno = 0;
            const unsigned long long v = (conv == 'd' || conv == 'i')
                ? static_cast<unsigned long long>(std::strtoll(field.c_str(), &end, base))
                : std::strtoull(field.c_str(), &end, base);
            if (end == field.c_str()) return assigned;
            pos += static_cast<usize>(end - field.c_str());
            any_conversion = true;
            if (!suppress) {
                mem.write_bytes(args.next_gp(), &v, static_cast<usize>(conv == 'p' ? 8 : size));
                ++assigned;
            }
        } else if (conv == 's') {
            const usize start = pos;
            while (pos < limit && !ws(in[pos])) ++pos;
            any_conversion = true;
            if (!suppress) {
                const std::string tok = in.substr(start, pos - start);
                mem.write_bytes(args.next_gp(), tok.c_str(), tok.size() + 1);
                ++assigned;
            }
        } else if (conv == 'c') {
            const usize n = width ? width : 1;
            if (pos + n > in.size()) return any_conversion ? assigned : -1;
            any_conversion = true;
            if (!suppress) {
                mem.write_bytes(args.next_gp(), in.data() + pos, n);
                ++assigned;
            }
            pos += n;
        } else if (conv == '[') {
            ++i;
            const bool negate = fmt[i] == '^';
            if (negate) ++i;
            bool in_set[256] = {};
            if (fmt[i] == ']') in_set[static_cast<u8>(fmt[i++])] = true;
            while (fmt[i] != '\0' && fmt[i] != ']') {
                const u8 a = static_cast<u8>(fmt[i]);
                if (fmt[i + 1] == '-' && fmt[i + 2] != ']' && fmt[i + 2] != '\0') {
                    for (unsigned c = a; c <= static_cast<u8>(fmt[i + 2]); ++c) in_set[c] = true;
                    i += 3;
                } else {
                    in_set[a] = true;
                    ++i;
                }
            }
            const usize start = pos;
            while (pos < limit && (in_set[static_cast<u8>(in[pos])] != negate)) ++pos;
            if (pos == start) return assigned;
            any_conversion = true;
            if (!suppress) {
                const std::string tok = in.substr(start, pos - start);
                mem.write_bytes(args.next_gp(), tok.c_str(), tok.size() + 1);
                ++assigned;
            }
            if (fmt[i] == '\0') break;
        } else {
            return assigned; // unsupported conversion (e.g. floating point)
        }
    }
    return assigned;
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

// ---------------------------------------------------------------------------
// GPU -> CPU completion signalling (Gnm EOP interrupts)
//
// sceGnmAddEqEvent(eq, kEopEventId, udata) registers an EVFILT_GRAPHICS_CORE
// knote. The GPU raises it when the command processor retires an
// EVENT_WRITE_EOP / RELEASE_MEM packet whose INT_SEL requests an interrupt --
// and only then. Unity's gfx worker waits on that queue ("EOP QUEUE") for the
// end-of-frame EOP of the previous frame before it reuses that frame's command
// memory, and pairs it with the frame label the same packet writes.
// ---------------------------------------------------------------------------
constexpr i32 kGnmEopEventId = 0x40; // GnmEventIdents::GfxEop

struct GnmEqSubscription {
    u32 eq_id{0};
    i32 event_id{0};
    u64 udata{0};
};
static std::mutex s_gnm_eq_mtx;
static std::vector<GnmEqSubscription> s_gnm_eq_subscriptions;
static std::atomic<u64> s_gnm_eop_interrupts{0};

static void deliver_gnm_eop_interrupt() {
    s_gnm_eop_interrupts.fetch_add(1, std::memory_order_relaxed);
    if (!g_thread_manager) return;
    std::lock_guard<std::mutex> lock(s_gnm_eq_mtx);
    for (const auto& sub : s_gnm_eq_subscriptions) {
        if (sub.event_id != kGnmEopEventId) continue;
        SceKernelEvent ev{};
        ev.ident = static_cast<u64>(sub.event_id);
        ev.filter = kEvFiltGraphicsCore;
        ev.flags = 0;
        ev.fflags = 0;
        ev.data = sub.event_id; // sceGnmGetEqEventType reads the event type back from data
        ev.udata = sub.udata;
        g_thread_manager->equeue_signal(sub.eq_id, ev);
    }
}

// Count the end-of-pipe packets in a DCB that request a CPU interrupt
// (EVENT_WRITE_EOP INT_SEL = dw2[25:24], RELEASE_MEM INT_SEL = dw1[26:24]).
// The PM4 translator performs the label writes; the interrupts belong to the
// HLE equeue path, raised after the stream has executed (the label is visible
// before the waiter wakes, as on hardware).
static u32 count_dcb_eop_interrupts(MemoryManager& mem, GuestAddr dcb, u32 size_bytes, bool log) {
    if (!mem.is_valid_address(dcb, size_bytes)) return 0;
    const u32* dw = mem.to_host_ptr<u32>(dcb);
    const usize n = size_bytes / sizeof(u32);
    usize off = 0;
    u32 interrupts = 0;
    while (off < n) {
        const Pm4Header hdr = decode_pm4_header(dw[off++]);
        if (hdr.type == Pm4PacketType::Type2) continue;
        if (hdr.type != Pm4PacketType::Type3 && hdr.type != Pm4PacketType::Type0) break;
        if (off + hdr.count > n) break;
        const u32* p = dw + off;
        if (hdr.type == Pm4PacketType::Type3) {
            if (hdr.opcode == IT_EVENT_WRITE_EOP && hdr.count >= 3 && ((p[2] >> 24) & 0x3) != 0) ++interrupts;
            if (hdr.opcode == IT_RELEASE_MEM && hdr.count >= 2 && ((p[1] >> 24) & 0x7) != 0) ++interrupts;
            if (log) {
                switch (hdr.opcode) {
                case IT_WAIT_REG_MEM:
                case IT_EVENT_WRITE_EOP:
                case IT_EVENT_WRITE_EOS:
                case IT_RELEASE_MEM:
                case IT_WRITE_DATA: {
                    PM4_LOG("[GnmSync] dcb=0x%llx +%zu %s:", (unsigned long long)dcb,
                                 (off - 1) * 4, pm4_opcode_to_name(hdr.opcode));
                    for (u16 i = 0; i < hdr.count && i < 8; ++i) std::fprintf(stderr, " %08x", p[i]);
                    std::fprintf(stderr, "\n");
                    break;
                }
                default: break;
                }
            }
        }
        off += hdr.count;
    }
    return interrupts;
}

// Execute the DCBs of a submit on the emulated command processor and raise the
// EOP interrupts they request. Shared by sceGnmSubmitCommandBuffers and
// sceGnmSubmitAndFlipCommandBuffers.
static void execute_submitted_dcbs(MemoryManager& mem, u32 count, GuestAddr dcb_addrs_ptr,
                                   GuestAddr dcb_sizes_ptr) {
    if (dcb_addrs_ptr == 0 || dcb_sizes_ptr == 0 || count > 64) return;
    static std::atomic<u32> s_submits_logged{0};
    const bool log = s_submits_logged.fetch_add(1, std::memory_order_relaxed) < 4;
    if (g_pm4_translator) g_pm4_translator->set_memory_manager(&mem);
    u32 interrupts = 0;
    for (u32 i = 0; i < count; ++i) {
        GuestAddr dcb_gpu_addr = 0;
        u32 dcb_size_bytes = 0;
        if (mem.is_valid_address(dcb_addrs_ptr + i * sizeof(GuestAddr), sizeof(GuestAddr))) {
            mem.read_value(dcb_addrs_ptr + i * sizeof(GuestAddr), dcb_gpu_addr);
        }
        if (mem.is_valid_address(dcb_sizes_ptr + i * sizeof(u32), sizeof(u32))) {
            mem.read_value(dcb_sizes_ptr + i * sizeof(u32), dcb_size_bytes);
        }
        if (dcb_gpu_addr <= 0x1000 || dcb_size_bytes < sizeof(u32) ||
            !mem.is_valid_address(dcb_gpu_addr, dcb_size_bytes)) {
            continue;
        }
        interrupts += count_dcb_eop_interrupts(mem, dcb_gpu_addr, dcb_size_bytes, log);
        if (g_pm4_translator) {
            const usize pkts_in_dcb = g_pm4_translator->execute_command_buffer(dcb_gpu_addr, dcb_size_bytes);
            auto cc = g_pm4_translator->clear_color();
            PM4_LOG("[Gnm] Executed DCB #%u at 0x%llx (%u bytes, %zu pkts, total_draws=%zu, reg_writes=%zu, clear=(%.2f,%.2f,%.2f,%.2f))\n",
                         i, (unsigned long long)dcb_gpu_addr, dcb_size_bytes, pkts_in_dcb,
                         g_pm4_translator->draw_count(), g_pm4_translator->context_reg_writes(),
                         cc.r, cc.g, cc.b, cc.a);
        }
    }
    for (u32 i = 0; i < interrupts; ++i) deliver_gnm_eop_interrupt();
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

// Ensure Unity ScreenManagerPS4 display configuration globals and dimensions are
// initialized to 1920x1080 (16:9), avoiding 0/0=NaN in Camera::CalculateProjectionMatrix.
static void ensure_unity_screen_resolution(MemoryManager& mem) {
    constexpr GuestAddr DISPLAY_CFG_ADDR = 0x3FF1000ULL;
    if (mem.is_valid_address(DISPLAY_CFG_ADDR, 16)) {
        mem.write_value(DISPLAY_CFG_ADDR + 0x00, static_cast<u32>(1920));
        mem.write_value(DISPLAY_CFG_ADDR + 0x04, static_cast<u32>(1920));
        mem.write_value(DISPLAY_CFG_ADDR + 0x08, static_cast<u32>(0));
        mem.write_value(DISPLAY_CFG_ADDR + 0x0C, static_cast<u32>(0));
    }
    // Note: RELRO GOT pointers for QBbo2kzANmQ, iEfUiWkJ3PI, 8HhxNMSnlT4, hMr0EtMYJbQ
    // are resolved directly to DISPLAY_CFG_ADDR by DynamicLinker at load time.

    // If Unity's ScreenManagerPS4 singleton is already instantiated at 0x1563a00,
    // ensure its cached dimensions are initialized to 1920x1080.
    GuestAddr sm_ptr = 0;
    const bool read_ok = mem.read_value(0x1563a00, sm_ptr);
    PM4_LOG("[ScreenManager] ensure_unity_screen_resolution: *0x1563a00 = 0x%llx (ok=%d)\n",
                 (unsigned long long)sm_ptr, read_ok ? 1 : 0);
    if (read_ok && sm_ptr > 0x1000 && mem.is_valid_address(sm_ptr, 0x300)) {
        u32 old_w = 0, old_h = 0;
        mem.read_value(sm_ptr + 0x2b0, old_w);
        mem.read_value(sm_ptr + 0x2b4, old_h);
        mem.write_value(sm_ptr + 0x2b0, static_cast<u32>(1920)); // m_Width
        mem.write_value(sm_ptr + 0x2b4, static_cast<u32>(1080)); // m_Height
        mem.write_value(sm_ptr + 0x2bc, static_cast<u32>(1920)); // m_TargetWidth
        mem.write_value(sm_ptr + 0x2c0, static_cast<u32>(1080)); // m_TargetHeight
        mem.write_value(sm_ptr + 0x2e4, static_cast<u32>(1920)); // m_VideoWidth
        mem.write_value(sm_ptr + 0x2e8, static_cast<u32>(1080)); // m_VideoHeight
        PM4_LOG("[ScreenManager] Fixed sm=0x%llx from %ux%u to 1920x1080\n",
                     (unsigned long long)sm_ptr, old_w, old_h);
    }
}

// ---------------------------------------------------------------------------
// Helpers for the offline-console and libc stubs registered at the end of
// register_default_libraries().
// ---------------------------------------------------------------------------

// Reads a NUL-terminated guest string into `out`. Returns false when the first
// byte is unmapped; a string that runs off mapped memory keeps what was read.
static bool read_guest_cstring(const MemoryManager& mem, GuestAddr p, std::string& out) {
    out.clear();
    if (p == 0 || !mem.is_valid_address(p, 1)) return false;
    for (GuestSize i = 0; i < (GuestSize{1} << 20); ++i) {
        char c = 0;
        if (!mem.read_value(p + i, c) || c == 0) break;
        out.push_back(c);
    }
    return true;
}

// strtol-style end pointer: *endptr = guest address of the first unparsed byte.
static void store_end_pointer(MemoryManager& mem, GuestAddr endptr, GuestAddr s,
                              const std::string& text, const char* end) {
    if (endptr == 0 || !mem.is_valid_address(endptr, sizeof(GuestAddr))) return;
    const GuestAddr at = end ? s + static_cast<GuestAddr>(end - text.c_str()) : s;
    mem.write_value<GuestAddr>(endptr, at);
}

// A guest-visible NUL-terminated copy of `text`, allocated once per distinct
// string. Used for char* results the guest may keep (strerror, setlocale, the
// zone name in struct tm). Returns 0 when the guest heap is exhausted.
static GuestAddr guest_static_string(MemoryManager& mem, const std::string& text) {
    static std::mutex mtx;
    static std::map<std::string, GuestAddr> cache;
    std::lock_guard<std::mutex> lock(mtx);
    if (const auto it = cache.find(text); it != cache.end()) return it->second;
    const GuestAddr at = bump_alloc(text.size() + 1, 16);
    if (at == 0 || !mem.write_bytes(at, text.c_str(), text.size() + 1)) return 0;
    cache.emplace(text, at);
    return at;
}

// strcasecmp core over already-read strings (ASCII case folding).
static int guest_strcasecmp(const std::string& a, const std::string& b) {
    for (usize i = 0;; ++i) {
        int ca = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        int cb = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        if (ca >= 'A' && ca <= 'Z') ca += 'a' - 'A';
        if (cb >= 'A' && cb <= 'Z') cb += 'a' - 'A';
        if (ca != cb) return ca - cb;
        if (ca == 0) return 0;
    }
}

// strtok / strtok_r core. `save` is the scan position kept between calls.
static GuestAddr guest_strtok(MemoryManager& mem, GuestAddr s, GuestAddr delim, GuestAddr& save) {
    std::string d;
    if (s == 0) s = save;
    if (s == 0 || !read_guest_cstring(mem, delim, d)) {
        save = 0;
        return 0;
    }
    auto is_delim = [&d](char c) { return d.find(c) != std::string::npos; };
    char c = 0;
    GuestAddr p = s;
    for (;;) {  // skip leading delimiters
        if (!mem.read_value(p, c) || c == 0) {
            save = 0;
            return 0;
        }
        if (!is_delim(c)) break;
        ++p;
    }
    const GuestAddr token = p;
    for (;;) {  // find the end of the token
        if (!mem.read_value(p, c) || c == 0) {
            save = 0;
            return token;
        }
        if (is_delim(c)) {
            mem.write_value<u8>(p, 0);
            save = p + 1;
            return token;
        }
        ++p;
    }
}

// Guest struct tm (LP64): nine 32-bit fields, four bytes of padding, long
// tm_gmtoff at +40 and char* tm_zone at +48; 56 bytes in all.
static bool load_guest_tm(const MemoryManager& mem, GuestAddr p, std::tm& t) {
    if (p == 0 || !mem.is_valid_address(p, 56)) return false;
    i32 f[9] = {};
    for (GuestSize i = 0; i < 9; ++i) mem.read_value(p + 4 * i, f[i]);
    t = {};
    t.tm_sec = f[0];  t.tm_min = f[1];  t.tm_hour = f[2];
    t.tm_mday = f[3]; t.tm_mon = f[4];  t.tm_year = f[5];
    t.tm_wday = f[6]; t.tm_yday = f[7]; t.tm_isdst = f[8];
    return true;
}

static void store_guest_tm(MemoryManager& mem, GuestAddr p, const std::tm& t, bool with_zone) {
    const i32 f[9] = {t.tm_sec, t.tm_min, t.tm_hour, t.tm_mday, t.tm_mon,
                      t.tm_year, t.tm_wday, t.tm_yday, t.tm_isdst};
    for (GuestSize i = 0; i < 9; ++i) mem.write_value<i32>(p + 4 * i, f[i]);
    if (!with_zone) return;
    mem.write_value<i64>(p + 40, static_cast<i64>(t.tm_gmtoff));
    mem.write_value<GuestAddr>(p + 48, t.tm_zone ? guest_static_string(mem, t.tm_zone) : 0);
}

// localtime (local = true) and gmtime. Both hand back one static struct tm,
// as libc does, so a second call overwrites the first result.
static GuestAddr guest_time_split(MemoryManager& mem, GuestAddr timer, bool local) {
    static GuestAddr result = 0;
    i64 secs = 0;
    if (!mem.read_value(timer, secs)) return 0;
    const std::time_t tt = static_cast<std::time_t>(secs);
    std::tm host{};
    if ((local ? localtime_r(&tt, &host) : gmtime_r(&tt, &host)) == nullptr) return 0;
    if (result == 0) result = bump_alloc(64, 16);
    if (result == 0) return 0;
    store_guest_tm(mem, result, host, true);
    return result;
}

// Current time as an SCE RTC tick: microseconds since 0001-01-01 00:00:00 UTC.
static u64 guest_rtc_now() {
    constexpr u64 kUnixEpochTicks = 62135596800ULL * 1000000ULL;  // 0001-01-01 -> 1970-01-01
    timeval tv{};
    gettimeofday(&tv, nullptr);
    return kUnixEpochTicks + static_cast<u64>(tv.tv_sec) * 1000000ULL + static_cast<u64>(tv.tv_usec);
}

// inet_pton over guest memory. Orbis address families follow FreeBSD:
// AF_INET = 2, AF_INET6 = 28. Returns inet_pton's result (1 ok, 0 malformed),
// or -1 for a family this layer does not map.
static u64 guest_inet_pton(MemoryManager& mem, u64 af, GuestAddr src, GuestAddr dst) {
    std::string text;
    read_guest_cstring(mem, src, text);
    int host_af = 0;
    GuestSize len = 0;
    if (af == 2) {
        host_af = AF_INET;
        len = 4;
    } else if (af == 28) {
        host_af = AF_INET6;
        len = 16;
    } else {
        return static_cast<u64>(-1);
    }
    unsigned char out[16] = {};
    const int rc = inet_pton(host_af, text.c_str(), out);
    if (rc == 1 && mem.is_valid_address(dst, len)) mem.write_bytes(dst, out, len);
    return static_cast<u64>(static_cast<i64>(rc));
}
static void register_save_data_and_trophy(NidRegistry& reg);

void NidRegistry::register_default_libraries() {
    // A new registry is a new emulation session. request_semaphore_shutdown()
    // from an earlier registry's teardown left s_sem_shutdown set, and since
    // WaitSema treats it as "cancel every wait", each later wait in the same
    // process returned ECANCELED (test_phase9's Test 11 after Test 10).
    s_sem_shutdown.store(false, std::memory_order_relaxed);
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
    //
    // Really loads the module. It used to mint an incrementing integer and hand
    // that back, so mono-ps4.prx and MonoAssembliesPS4.prx were "loaded" while
    // not one instruction of either had executed -- every mono-ps4 NID kept
    // resolving to our s_mono_arena cookie stubs, and every static constructor
    // in the game silently did nothing.
    //
    // The load is best-effort. A module that fails to map returns 0 (the real
    // failure value) rather than a fake handle, so callers can tell the
    // difference instead of jumping into an unmapped page.

    // sceKernelLoadStartModule
    // NID: 0xC33BEA4F852A297FULL
    // Call guest code at `entry` (SysV: rdi, rsi) to completion on the calling
    // thread's executor, nested inside the current HLE call, as exception
    // delivery does. It returns through a HLT sentinel; the caller's context
    // is restored afterwards. False when no executor is active or the code
    // stopped before returning.
    static auto call_guest = [](MemoryManager& mem, GuestAddr entry, u64 a0, u64 a1, u64& ret) -> bool {
        CpuExecutor* ex = current_executor();
        if (ex == nullptr || entry == 0) return false;
        static const GuestAddr s_sentinel = [&mem] {
            const GuestAddr p = bump_alloc(16, 16);
            const u8 hlt = 0xF4;
            if (p != 0) mem.write_bytes(p, &hlt, 1);
            return p;
        }();
        if (s_sentinel == 0) return false;
        const CpuContext saved = ex->context();
        CpuContext& c = ex->context();
        const GuestAddr sp = ((saved.rsp - 512) & ~GuestAddr{0xF}) - 8;  // below the red zone
        mem.write_value(sp, s_sentinel);
        c.rsp = sp;
        c.rip = entry;
        c.rdi = a0;
        c.rsi = a1;
        // A PRX start stub tail-calls a non-null third argument instead of
        // the module's own module_start.
        c.rdx = 0;
        c.rcx = 0;
        bool returned = false;
        for (int i = 0; i < 100000; ++i) {
            const RunResult rr = ex->run(200000);
            if (c.rip == s_sentinel || c.rip == s_sentinel + 1) {
                returned = true;
                break;
            }
            if (rr.status != StepResult::Ok && rr.status != StepResult::Syscall) {
                std::fprintf(stderr, "[HLE] guest call 0x%llx stopped at RIP=0x%llx (%s)\n",
                             (unsigned long long)entry, (unsigned long long)c.rip,
                             step_result_to_string(rr.status).data());
                break;
            }
        }
        ret = c.rax;
        ex->context() = saved;
        return returned;
    };
    register_symbol(0xC33BEA4F852A297FULL, "sceKernelLoadStartModule", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rdi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : "unknown";

            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            std::string host_path = resolved ? resolved->string() : std::string(path);
            if (host_path.ends_with(".sprx")) {
                std::string prx_cand = host_path.substr(0, host_path.length() - 5) + ".prx";
                if (std::filesystem::exists(prx_cand)) {
                    host_path = prx_cand;
                }
            }

            // Real PRX loading is enabled by default when the target module file exists on disk,
            // unless explicitly disabled via PS4_LOAD_PRX=0. If the PRX is missing, it falls
            // back to HLE stub mode.
            const char* prx_env = std::getenv("PS4_LOAD_PRX");
            const bool force_hle = (prx_env && std::strcmp(prx_env, "0") == 0);
            const bool prx_exists = std::filesystem::exists(host_path);

            if (force_hle || !prx_exists) {
                static u32 s_module_id = 0x2000;
                const u32 handle = ++s_module_id;
                std::fprintf(stderr, "[HLE] sceKernelLoadStartModule('%s') -> handle 0x%x (HLE mode)\n", path, handle);
                ensure_unity_screen_resolution(mem);
                ctx.rax = handle;
                return;
            }

            DynamicLinker* linker = ps4::active_linker();
            if (linker == nullptr) {
                std::fprintf(stderr, "[HLE] sceKernelLoadStartModule('%s'): no linker, failed\n", path);
                ctx.rax = 0;
                return;
            }
            // base 0: let load_module reserve the module's real span.
            const GuestAddr base = 0;
            // run_init stays false: MonoAssembliesPS4.prx has DT_INIT == 0, and
            // the guest drives these modules through its own startup path.
            // Calling a null entry point would jump straight to the module base.
            const u32 handle = linker->load_module(host_path, base, /*run_init=*/false);
            std::fprintf(stderr, "[HLE] sceKernelLoadStartModule('%s' -> '%s') -> handle 0x%x\n",
                         path, host_path.c_str(), handle);
            // The "Start": module_start(args, argp), whose result goes to
            // *pRes (r9). Unity's SavedGames plugin marks itself started there;
            // without the call every export returned early and the game's
            // startup autoload never finished. A module that does not export
            // module_start starts at its ELF entry: the SDK stub there runs the
            // initializer array, then the module's own start routine (e_entry
            // is commonly 0, i.e. the module base).
            const u64 args = ctx.rsi, argp = ctx.rdx, res_ptr = ctx.r9;
            constexpr Nid kModuleStart = 0x05A38A72783C83CFULL;
            GuestAddr start = handle ? linker->find_module_export(handle, kModuleStart) : 0;
            if (start == 0 && handle != 0)
                if (const LoadedModule* m = linker->module(handle)) start = m->base + m->entry;
            if (start != 0) {
                u64 rv = 0;
                const bool ok = call_guest(mem, start, args, argp, rv);
                std::fprintf(stderr, "[HLE] module_start 0x%llx of handle 0x%x -> %s %lld\n",
                             (unsigned long long)start, handle, ok ? "returned" : "did not return",
                             (long long)static_cast<i32>(rv));
                if (ok && res_ptr > 0x1000 && mem.is_valid_address(res_ptr, 4))
                    mem.write_value(res_ptr, static_cast<i32>(rv));
            }
            ensure_unity_screen_resolution(mem);
            ctx.rax = handle;
        });

    // sceKernelDlsym(SceKernelModule handle, const char* symbol, void** addrp)
    // NID: 0x2F01BC8379E2AB00
    //
    // Mono's pinvoke loader resolves the native plugins' entry points by name
    // through this (e.g. _OSKPlugin_GetScreenScale, then the ...A variant). The
    // name is hashed to the NID the exporting module stores, then looked up among
    // that module's own exports. A handle that is not a loaded module falls back
    // to the global export map. On a miss *addrp is left alone and ESRCH returns.
    register_symbol(0x2F01BC8379E2AB00ULL, "sceKernelDlsym", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const u32 handle = static_cast<u32>(ctx.rdi);
            const GuestAddr name_addr = ctx.rsi;
            const GuestAddr out_addr = ctx.rdx;

            // Bounded copy of the guest name; a bad pointer just ends the string.
            std::string symbol;
            for (GuestAddr p = name_addr; symbol.size() < 256 && mem.is_valid_address(p, 1); ++p) {
                const char c = *mem.to_host_ptr<char>(p);
                if (c == '\0') break;
                symbol.push_back(c);
            }

            DynamicLinker* linker = ps4::active_linker();
            GuestAddr addr = 0;
            if (linker != nullptr && !symbol.empty()) {
                const Nid nid = nid_from_name(symbol);
                addr = linker->module(handle) != nullptr
                           ? linker->find_module_export(handle, nid)
                           : linker->find_export(nid);
            }
            if (addr == 0) {
                std::fprintf(stderr, "[HLE] sceKernelDlsym(handle 0x%x, '%s') -> not found\n",
                             handle, symbol.c_str());
                ctx.rax = 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }
            if (out_addr != 0) mem.write_value(out_addr, addr);
            std::fprintf(stderr, "[HLE] sceKernelDlsym(handle 0x%x, '%s') -> 0x%llx\n",
                         handle, symbol.c_str(), (unsigned long long)addr);
            ctx.rax = SCE_OK;
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

        // (A former workaround here zeroed [r15 + 0x98] on EVERY call, for any
        // caller, to force PreloadManager's wait-for-all loop to finish while
        // Mono was stubbed. With the real runtime it erased the queued level0
        // load -- the preload thread then found its queue empty and the first
        // scene never loaded -- and it corrupted whatever else r15 pointed at.)
        // The sleep answers guest signals like the other waits: Mono's GC
        // stops the world with signal 30 and blocks until every thread has
        // acknowledged. A sleeping thread only reached its handler at the end
        // of its next instruction quantum, which for a usleep polling loop
        // took tens of seconds (the game froze on each collection).
        if (usec > 0) {
            std::mutex m;
            std::condition_variable cv;
            std::unique_lock<std::mutex> lock(m);
            SignalWakeScope wake(cv);
            cv.wait_for(lock, std::chrono::microseconds(usec), [] { return guest_signal_pending(); });
        } else {
            std::this_thread::yield();
        }

        if (cur_th) {
            cur_th->sleep_until_us.store(0);
            cur_th->state.store(ThreadState::Running);
        }
        if (guest_signal_pending()) deliver_guest_exceptions(mem);
        ctx.rax = SCE_OK;
    };
    register_symbol(0x3230BAEFBC36C18BULL, "sceKernelUsleep", "libkernel", usleep_handler);
    register_symbol(0xD637D72D15738AC7ULL, "sceKernelUsleep", "libkernel", usleep_handler);

    // nanosleep(const struct timespec* req, struct timespec* rem)
    // NID: 0xC92F14D931827B50ULL ("yS8U2TGCe1A")
    register_symbol(0xC92F14D931827B50ULL, "nanosleep", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr req_ptr = ctx.rdi;
            if (req_ptr > 0x1000 && mem.is_valid_address(req_ptr, 16)) {
                u64 sec = 0, nsec = 0;
                mem.read_value(req_ptr, sec);
                mem.read_value(req_ptr + 8, nsec);
                const u64 usec = sec * 1'000'000ULL + nsec / 1'000ULL;
                auto& tm = get_thread_manager(mem);
                GuestThread* cur_th = tm.current_thread();
                if (cur_th) {
                    cur_th->state.store(ThreadState::WaitingSleep);
                    const u64 now_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
                    cur_th->sleep_until_us.store(now_us + usec);
                }
                if (usec > 0) {
                    std::mutex m;
                    std::condition_variable cv;
                    std::unique_lock<std::mutex> lock(m);
                    SignalWakeScope wake(cv);
                    cv.wait_for(lock, std::chrono::microseconds(usec), [] { return guest_signal_pending(); });
                } else {
                    std::this_thread::yield();
                }
                if (cur_th) {
                    cur_th->sleep_until_us.store(0);
                    cur_th->state.store(ThreadState::Running);
                }
                if (guest_signal_pending()) deliver_guest_exceptions(mem);
            }
            ctx.rax = 0;
        });

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

            {
                static std::atomic<u32> s_wait_logged{0};
                if (s_wait_logged.fetch_add(1) < 40) {
                    std::fprintf(stderr, "[Equeue] T%d WaitEqueue(eq=%u, num=%d, timeout=%s%llu)\n",
                                 (int)t_guest_tid, eq_id, num, has_timeout ? "" : "inf/",
                                 (unsigned long long)timeout_usec);
                }
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

    // malloc(size_t size): not zeroed (reused blocks keep old contents, as on
    // real libc).
    auto malloc_handler = [](CpuContext& ctx, MemoryManager&) {
        const GuestSize size = ctx.rdi;
        ctx.rax = bump_alloc(size, 16);
    };
    register_symbol(0x9E990E49EC698A4DULL, "malloc", "libc", malloc_handler);
    register_symbol(0x8105FEE060D08E93ULL, "malloc", "libc", malloc_handler);

    // free(void* ptr)
    auto free_handler = [](CpuContext& ctx, MemoryManager&) {
        heap_free(ctx.rdi);
        ctx.rax = 0;
    };
    register_symbol(0x7E16B7B8361BD5BBULL, "free", "libc", free_handler);
    register_symbol(0xB4886CAA3D2AB051ULL, "free", "libc", free_handler);

    // C++ operator new / delete (libc++ mangled names). The loaded PRX plugins
    // import these; with no handler every new came back NULL. Backed by the same
    // heap as malloc, so memory from either side can be freed by the other.
    auto new_handler = [](CpuContext& ctx, MemoryManager&) {
        ctx.rax = bump_alloc(ctx.rdi, 16);
    };
    register_symbol(0x7C99E9B955416CA9ULL, "_Znwm", "libc++", new_handler);  // operator new(size_t)
    register_symbol(0x85D9B461F31AED34ULL, "_Znam", "libc++", new_handler);  // operator new[](size_t)
    auto delete_handler = [](CpuContext& ctx, MemoryManager&) {
        heap_free(ctx.rdi);
        ctx.rax = 0;
    };
    register_symbol(0xCFE3FEC429D62C19ULL, "_ZdlPv", "libc++", delete_handler);  // operator delete(void*)
    register_symbol(0x30B5A5F7448558D1ULL, "_ZdaPv", "libc++", delete_handler);  // operator delete[](void*)

    // realloc(void* ptr, size_t size) -- NID: 0x63B689D6EC9D3CCAULL per
    // ps4libdoc. This NID is imported by eboot.bin and mono-ps4.prx but was
    // registered as a third "free", so every realloc silently returned NULL.
    // bump_alloc() knows the old size without a block header: keep the block
    // if it still fits, otherwise copy and free it.
    register_symbol(0x63B689D6EC9D3CCAULL, "realloc", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr old_ptr = ctx.rdi;
            const GuestSize size = ctx.rsi;
            if (size == 0) {
                heap_free(old_ptr);
                ctx.rax = 0;
                return;
            }
            const GuestSize old_size = bump_size_of(old_ptr);
            if (old_ptr != 0 && old_size >= size) {
                ctx.rax = old_ptr;
                return;
            }
            const GuestAddr allocated = bump_alloc(size, 16);
            if (allocated != 0 && old_ptr != 0 && old_size > 0 && mem.is_valid_address(old_ptr, old_size)) {
                std::memcpy(mem.to_host_ptr<void>(allocated), mem.to_host_ptr<void>(old_ptr),
                            old_size < size ? old_size : size);
                heap_free(old_ptr);
            }
            ctx.rax = allocated;
        });

    // reallocalign(void* ptr, size_t size, size_t alignment)
    // NID: 0x386c9b56e3c08406ULL ("OGybVuPAhAY")
    auto reallocalign_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr old_ptr = ctx.rdi;
        const GuestSize size = ctx.rsi;
        const GuestSize align = ctx.rdx > 0 ? ctx.rdx : 16;
        if (size == 0) {
            heap_free(old_ptr);
            ctx.rax = 0;
            return;
        }
        if (old_ptr == 0) {
            ctx.rax = bump_alloc(size, align);
            return;
        }
        const GuestSize old_size = bump_size_of(old_ptr);
        if (old_size >= size && (old_ptr & (align - 1)) == 0) {
            ctx.rax = old_ptr;
            return;
        }
        const GuestAddr allocated = bump_alloc(size, align);
        if (allocated != 0 && old_size > 0 && mem.is_valid_address(old_ptr, old_size)) {
            std::memcpy(mem.to_host_ptr<void>(allocated), mem.to_host_ptr<void>(old_ptr),
                        old_size < size ? old_size : size);
            heap_free(old_ptr);
        }
        ctx.rax = allocated;
    };
    register_symbol(0x386c9b56e3c08406ULL, "reallocalign", "libc", reallocalign_handler);
    register_symbol(0x0684c0e3569b6c38ULL, "reallocalign", "libc", reallocalign_handler);

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

    // C++ standard stream initializers (std::ios_base::Init and std::_Winit)
    register_symbol(0xB2A5B2B678587448ULL, "_ZNSt8ios_base4InitC1Ev", "libstdc++",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = ctx.rdi; // constructor returns this
        });

    register_symbol(0x9315C2BDC6ADD5C3ULL, "_ZNSt8ios_base4InitD1Ev", "libstdc++",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = ctx.rdi;
        });

    register_symbol(0xFC197DFD26769E87ULL, "_ZNSt6_WinitC1Ev", "libstdc++",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = ctx.rdi; // constructor returns this
        });

    register_symbol(0x530DCE4D914F36DEULL, "_ZNSt6_WinitD1Ev", "libstdc++",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = ctx.rdi;
        });

    register_symbol(0x6F3404C72D7CF592ULL, "_init_env", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    register_symbol(0xF06D8B07E037AF38ULL, "atexit", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // _ZSt18_Xinvalid_argumentPKc: libc++ throw helper, never returns. There is
    // no unwinder, so returning would fall into the ud2 after the call; abort
    // with the message instead, like _Xout_of_range below.
    static auto throw_helper_abort = [](const char* kind, CpuContext& ctx, MemoryManager& mem) {
        std::string what;
        for (GuestSize i = 0; i < 256; ++i) {
            char c = 0;
            if (!mem.read_value(ctx.rdi + i, c) || c == 0) break;
            what.push_back(c);
        }
        guest_abort(std::string(kind) + " thrown: " + what, ctx, mem);
    };
    register_symbol(0x354FD3E10A304CD0ULL, "_ZSt18_Xinvalid_argumentPKc", "libstdc++",
        [](CpuContext& ctx, MemoryManager& mem) { throw_helper_abort("std::invalid_argument", ctx, mem); });

    // =========================================================================
    // POSIX entry points imported by mono-ps4.prx (PS4_LOAD_PRX)
    // =========================================================================
    // The Mono runtime imports these libkernel/libc NIDs; the eboot never did,
    // so they fell through to the "return 0" stub. Each zero was a lie with a
    // consequence: calloc -> NULL tripped eglib's 'hash != NULL' assertion,
    // open("/mono/config") -> fd 0 plus an unwritten stat buffer made Mono ask
    // for a 1.9 TB read buffer, mmap -> 0 put the GC heap at address 0, and
    // getpagesize -> 0 is a divisor. NIDs are hash-verified (name + salt).

    // calloc(size_t n, size_t size): zeroed explicitly, since freed blocks are reused.
    register_symbol(0xD97E5A8058CAC4C7ULL, "calloc", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize n = ctx.rdi, size = ctx.rsi;
            if (size != 0 && n > ~GuestSize{0} / size) {
                ctx.rax = 0;
                return;
            }
            const GuestAddr p = bump_alloc(n * size, 16);
            if (p != 0 && n * size != 0) mem.zero_memory(p, n * size);
            ctx.rax = p;
        });

    // open(const char* path, int flags, int mode) -> fd, or -1
    register_symbol(0xC2E0ABA081A3B768ULL, "open", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path = ctx.rdi;
            if (path == 0 || !mem.is_valid_address(path, 1)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const i32 fd = get_vfs().open(mem.to_host_ptr<char>(path),
                                          static_cast<i32>(ctx.rsi), static_cast<i32>(ctx.rdx));
            ctx.rax = static_cast<u64>(static_cast<i64>(fd < 0 ? -1 : fd));
        });

    // close(int fd) -> 0, or -1
    register_symbol(0x6D8FCF3BA261CE14ULL, "close", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = static_cast<u64>(get_vfs().close(static_cast<i32>(ctx.rdi)) < 0 ? -1 : 0);
        });

    // read(int fd, void* buf, size_t n) -> bytes read, or -1
    register_symbol(0x02A062A02DAF1772ULL, "read", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr buf = ctx.rsi;
            const GuestSize n = ctx.rdx;
            if (n == 0) {
                ctx.rax = 0;
                return;
            }
            if (!mem.is_valid_address(buf, n)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const i64 r = get_vfs().read(static_cast<i32>(ctx.rdi), mem.to_host_ptr<void>(buf), n);
            ctx.rax = static_cast<u64>(r < 0 ? -1 : r);
        });

    // stat(const char* path, struct stat* st) -> 0, or -1. Files and
    // directories both open read-only on the host, so open + fstat + close.
    register_symbol(0x13A6A8DF8C0FC3E5ULL, "stat", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path = ctx.rdi, out = ctx.rsi;
            if (path == 0 || !mem.is_valid_address(path, 1) ||
                !mem.is_valid_address(out, sizeof(OrbisStat))) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            const i32 fd = vfs.open(mem.to_host_ptr<char>(path), 0 /* O_RDONLY */, 0);
            if (fd < 0) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            OrbisStat st{};
            const i32 r = vfs.fstat(fd, st);
            vfs.close(fd);
            if (r != 0) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            mem.write_bytes(out, &st, sizeof(st));
            ctx.rax = 0;
        });
    // sceKernelStat(path, stat*): stat with SCE error codes (ENOENT when the
    // path does not open). SavedGames checks its slot files with it.
    register_symbol(0x795F70003DAB8880ULL, "sceKernelStat", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path = ctx.rdi, out = ctx.rsi;
            if (path == 0 || !mem.is_valid_address(path, 1) || !mem.is_valid_address(out, sizeof(OrbisStat))) {
                ctx.rax = 0x80020016ULL; // SCE_KERNEL_ERROR_EINVAL
                return;
            }
            auto& vfs = get_vfs();
            const i32 fd = vfs.open(mem.to_host_ptr<char>(path), 0 /* O_RDONLY */, 0);
            if (fd < 0) {
                ctx.rax = 0x80020002ULL; // SCE_KERNEL_ERROR_ENOENT
                return;
            }
            OrbisStat st{};
            const i32 r = vfs.fstat(fd, st);
            vfs.close(fd);
            if (r != 0) {
                ctx.rax = 0x80020005ULL; // SCE_KERNEL_ERROR_EIO
                return;
            }
            mem.write_bytes(out, &st, sizeof(st));
            ctx.rax = SCE_OK;
        });

    // strcpy(char* dst, const char* src) -> dst
    register_symbol(0x9226525C859DF6F8ULL, "strcpy", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dst = ctx.rdi, src = ctx.rsi;
            for (GuestSize i = 0;; ++i) {
                u8 b = 0;
                if (!mem.read_value(src + i, b) || !mem.write_value(dst + i, b)) break;
                if (b == 0) break;
            }
            ctx.rax = dst;
        });

    // strdup(const char* s)
    // NID: 0x83BCF3CCB0D81B0DULL ("g7zzzLDYGw0")
    register_symbol(0x83BCF3CCB0D81B0DULL, "strdup", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr str_addr = ctx.rdi;
            if (str_addr == 0 || !mem.is_valid_address(str_addr, 1)) {
                ctx.rax = 0;
                return;
            }
            std::string s;
            if (!read_guest_cstring(mem, str_addr, s)) {
                ctx.rax = 0;
                return;
            }
            const usize len = s.size() + 1;
            const GuestAddr dst = mspace_alloc(len, 16);
            if (dst != 0) {
                mem.write_bytes(dst, s.c_str(), len);
            }
            ctx.rax = dst;
        });

    // srand(unsigned int seed)
    // NID: 0x54F6C9C130A0304DULL ("VPbJwTCgME0")
    register_symbol(0x54F6C9C130A0304DULL, "srand", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            std::srand(static_cast<unsigned int>(ctx.rdi));
            ctx.rax = ctx.rdi;
        });

    // strrchr(const char* s, int c) -> last match (the terminator counts), or NULL
    register_symbol(0xF720D63311057495ULL, "strrchr", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr s = ctx.rdi;
            const u8 c = static_cast<u8>(ctx.rsi);
            GuestAddr last = 0;
            for (GuestSize i = 0;; ++i) {
                u8 b = 0;
                if (!mem.read_value(s + i, b)) break;
                if (b == c) last = s + i;
                if (b == 0) break;
            }
            ctx.rax = last;
        });

    // getpagesize() -> 16 KB on PS4
    register_symbol(0x93E017AAEDBF7817ULL, "getpagesize", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = GUEST_PAGE_SIZE; });

    // clock() -> process CPU time (FreeBSD CLOCKS_PER_SEC is 128)
    register_symbol(0x4193FA23D659C691ULL, "clock", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = static_cast<u64>(std::clock()) * 128 / CLOCKS_PER_SEC;
        });

    // mmap(void* addr, size_t len, int prot, int flags, int fd, off_t off).
    // FreeBSD flags: MAP_FIXED 0x10, MAP_ANON 0x1000. MAP_FAILED is -1.
    register_symbol(0x04F13DB3DBD0417AULL, "mmap", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr hint = ctx.rdi;
            const GuestSize len = ctx.rsi;
            const u32 flags = static_cast<u32>(ctx.rcx);
            const i32 fd = static_cast<i32>(ctx.r8);
            const i64 off = static_cast<i64>(ctx.r9);
            constexpr u32 kMapFixed = 0x10, kMapAnon = 0x1000;
            if (len == 0) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            GuestAddr addr = 0;
            if ((flags & kMapFixed) && mem.is_valid_address(hint, len)) {
                addr = hint;
                static std::atomic<u32> s_fixed_logged{0};
                if (s_fixed_logged.fetch_add(1) < 32)
                    std::fprintf(stderr, "[mmap] MAP_FIXED 0x%llx len=0x%llx (zeroed)\n",
                                 (unsigned long long)addr, (unsigned long long)len);
                std::memset(mem.to_host_ptr<void>(addr), 0, len);
            } else {
                addr = mmap_arena_alloc(len);
            }
            if (addr == 0) {
                std::fprintf(stderr, "[mmap] arena exhausted: len=0x%llx\n", (unsigned long long)len);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (!(flags & kMapAnon) && fd >= 0) {
                auto& vfs = get_vfs();
                if (vfs.lseek(fd, off, 0 /* SEEK_SET */) >= 0)
                    vfs.read(fd, mem.to_host_ptr<void>(addr), len);
            }
            ctx.rax = addr;
        });

    // munmap(void* addr, size_t len): arena space is not reused; zero it so a
    // later MAP_FIXED of the same range sees fresh pages.
    register_symbol(0x52A0C68D7039C943ULL, "munmap", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            static std::atomic<u32> s_logged{0};
            if (s_logged.fetch_add(1) < 32)
                std::fprintf(stderr, "[mmap] munmap 0x%llx len=0x%llx (zeroed)\n",
                             (unsigned long long)ctx.rdi, (unsigned long long)ctx.rsi);
            if (ctx.rsi != 0 && mem.is_valid_address(ctx.rdi, ctx.rsi))
                std::memset(mem.to_host_ptr<void>(ctx.rdi), 0, ctx.rsi);
            ctx.rax = 0;
        });

    // mprotect: guest memory is not page-protected.
    register_symbol(0x61039FC4BE107DE5ULL, "mprotect", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; });

    // pthread_self / pthread_equal: the thread id is the handle, as for
    // scePthreadSelf, so scePthreadAttrGet() below can find the thread again.
    register_symbol(0x128B51F1ADC049FEULL, "pthread_self", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = static_cast<u64>(get_thread_manager(mem).current_thread_id());
        });
    register_symbol(0xED7976E7B33854D2ULL, "pthread_equal", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = ctx.rdi == ctx.rsi ? 1 : 0; });

    // pthread_mutex_trylock: consistent with the mono-side pthread_mutex_lock
    // no-op above -- always acquired.
    register_symbol(0x2BF8D785BB76827EULL, "pthread_mutex_trylock", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; });

    // pthread_cleanup_push/pop: handlers only matter on cancellation, which is
    // never delivered.
    register_symbol(0xE1979959C32C015DULL, "pthread_cleanup_push", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; });
    register_symbol(0x455C5BD12B1AE6DDULL, "pthread_cleanup_pop", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; });

    register_symbol(0x9B9FF66EC35FBFBBULL, "scePthreadCondattrInit", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = SCE_OK; });
    register_symbol(0xC1A3DCC58891DD60ULL, "scePthreadCondattrDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = SCE_OK; });

    // scePthreadAttrGet(ScePthread thread, ScePthreadAttr* attr): the GC reads
    // each thread's stack bounds through this to scan it for roots. The main
    // thread has no GuestThread stack record; it runs on the default stack
    // from CpuExecutor::setup_default_stack().
    register_symbol(0xC755FBE9AAD83315ULL, "scePthreadAttrGet", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const ScePthreadId tid = static_cast<ScePthreadId>(ctx.rdi);
            const GuestAddr attr_ptr = ctx.rsi;
            if (attr_ptr == 0 || !mem.is_valid_address(attr_ptr, sizeof(u64))) {
                ctx.rax = static_cast<u64>(-22);
                return;
            }
            ScePthreadAttr attr{};
            attr.stack_addr = 0x1FFF00000ULL;
            attr.stack_size = 0x100000ULL;
            if (GuestThread* t = get_thread_manager(mem).get_thread(tid);
                t != nullptr && t->stack_base != 0) {
                attr.stack_addr = t->stack_base;
                attr.stack_size = t->stack_size;
            }
            mem.write_value(attr_ptr, static_cast<u64>(attr.stack_size));
            std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
            s_pthread_attrs[attr_ptr] = attr;
            ctx.rax = SCE_OK;
        });

    register_symbol(0x46EDFA7E24ED2730ULL, "scePthreadAttrGetstackaddr", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            GuestAddr addr = 0;
            {
                std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
                if (auto it = s_pthread_attrs.find(ctx.rdi); it != s_pthread_attrs.end())
                    addr = it->second.stack_addr;
            }
            if (ctx.rsi == 0 || !mem.write_value(ctx.rsi, addr)) {
                ctx.rax = static_cast<u64>(-22);
                return;
            }
            ctx.rax = SCE_OK;
        });

    register_symbol(0xFDF03EED99460D0BULL, "scePthreadAttrGetstacksize", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            GuestSize size = 2 * 1024 * 1024;
            {
                std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
                if (auto it = s_pthread_attrs.find(ctx.rdi); it != s_pthread_attrs.end())
                    size = it->second.stack_size;
            }
            if (ctx.rsi == 0 || !mem.write_value(ctx.rsi, static_cast<u64>(size))) {
                ctx.rax = static_cast<u64>(-22);
                return;
            }
            ctx.rax = SCE_OK;
        });

    // --- Thread-specific data --------------------------------------------
    // pthread_key_create used to be a sync stub that never wrote a key, and
    // get/setspecific fell to the zero stub, so Mono had no TLS at all. Guest
    // threads are not 1:1 with host threads, so values are keyed by guest
    // thread id rather than host thread_local storage. Destructors are not
    // run (guest threads are only reaped at shutdown).
    static std::mutex s_tsd_mtx;
    static std::unordered_map<u64, u64> s_tsd; // (tid << 32 | key) -> value
    static std::atomic<u32> s_next_tsd_key{1};
    auto key_create = [](CpuContext& ctx, MemoryManager& mem) {
        const u32 key = s_next_tsd_key.fetch_add(1);
        if (ctx.rdi == 0 || !mem.write_value(ctx.rdi, key)) {
            ctx.rax = 22; // EINVAL
            return;
        }
        ctx.rax = 0;
    };
    auto get_specific = [](CpuContext& ctx, MemoryManager& mem) {
        const u64 slot = (static_cast<u64>(get_thread_manager(mem).current_thread_id()) << 32) |
                         static_cast<u32>(ctx.rdi);
        std::lock_guard<std::mutex> lock(s_tsd_mtx);
        auto it = s_tsd.find(slot);
        ctx.rax = it == s_tsd.end() ? 0 : it->second;
    };
    auto set_specific = [](CpuContext& ctx, MemoryManager& mem) {
        const u64 slot = (static_cast<u64>(get_thread_manager(mem).current_thread_id()) << 32) |
                         static_cast<u32>(ctx.rdi);
        std::lock_guard<std::mutex> lock(s_tsd_mtx);
        s_tsd[slot] = ctx.rsi;
        ctx.rax = 0;
    };
    register_symbol(0x9AA50B35D8A64E7DULL, "pthread_key_create", "libScePosix", key_create);
    register_symbol(0x81E0DAAA01FD9538ULL, "scePthreadKeyCreate", "libkernel", key_create);
    register_symbol(0xD3F297692EF4C72EULL, "pthread_getspecific", "libScePosix", get_specific);
    register_symbol(0x7A886DEE640E0A6AULL, "scePthreadGetspecific", "libkernel", get_specific);
    register_symbol(0x5AB38BBC7534C903ULL, "pthread_setspecific", "libScePosix", set_specific);
    register_symbol(0xF81CD7624A9878B1ULL, "scePthreadSetspecific", "libkernel", set_specific);
    register_symbol(0xE81A4466E0D3ED82ULL, "pthread_key_delete", "libScePosix",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; });

    // --- sceKernel file API ------------------------------------------------
    // Same VFS as open/read/close above, but failures are SCE error codes
    // (0x80020000 | errno) instead of -1. Mono opens every managed assembly
    // (mscorlib.dll first) through sceKernelOpen.
    auto sce_err = [](i64 neg_errno) -> u64 {
        return 0x80020000ULL | static_cast<u64>(-neg_errno & 0xFFFF);
    };
    register_symbol(0xD46DE51751A0D64FULL, "sceKernelOpen", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path = ctx.rdi;
            if (path == 0 || !mem.is_valid_address(path, 1)) {
                ctx.rax = sce_err(-14); // EFAULT
                return;
            }
            const char* p = mem.to_host_ptr<char>(path);
            const i32 fd = get_vfs().open(p, static_cast<i32>(ctx.rsi), static_cast<i32>(ctx.rdx));
            static std::atomic<u32> s_logged{0};
            if (s_logged.fetch_add(1) < 64) {
                GuestAddr caller = 0;
                mem.read_value(ctx.rsp, caller);
                std::fprintf(stderr, "[VFS] sceKernelOpen('%s', 0x%x) -> %d (caller 0x%llx)\n", p,
                             static_cast<u32>(ctx.rsi), fd, (unsigned long long)caller);
            }
            ctx.rax = fd < 0 ? sce_err(fd) : static_cast<u64>(fd);
        });
    register_symbol(0x50AD939760D6527BULL, "sceKernelClose", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager&) {
            const i32 r = get_vfs().close(static_cast<i32>(ctx.rdi));
            ctx.rax = r < 0 ? sce_err(r) : 0;
        });
    register_symbol(0xD7F2C52E6445C713ULL, "sceKernelMkdir", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager& mem) {
            std::string path;
            if (!read_guest_cstring(mem, ctx.rdi, path)) {
                ctx.rax = sce_err(-14);
                return;
            }
            const i32 mode = static_cast<i32>(ctx.rsi);
            const i32 res = get_vfs().mkdir(path, mode);
            std::fprintf(stderr, "[VFS] sceKernelMkdir('%s', 0%o) -> %d\n", path.c_str(), mode, res);
            ctx.rax = (res < 0) ? sce_err(res) : SCE_OK;
        });
    register_symbol(0x0A0E2CAD9E9329B5ULL, "sceKernelRead", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize n = ctx.rdx;
            if (n != 0 && !mem.is_valid_address(ctx.rsi, n)) {
                ctx.rax = sce_err(-14);
                return;
            }
            const i64 r = n == 0 ? 0 : get_vfs().read(static_cast<i32>(ctx.rdi), mem.to_host_ptr<void>(ctx.rsi), n);
            ctx.rax = r < 0 ? sce_err(r) : static_cast<u64>(r);
        });
    register_symbol(0xFABDEB305C08B55EULL, "sceKernelPread", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager& mem) {
            const i32 fd = static_cast<i32>(ctx.rdi);
            const GuestSize n = ctx.rdx;
            if (n != 0 && !mem.is_valid_address(ctx.rsi, n)) {
                ctx.rax = sce_err(-14);
                return;
            }
            auto& vfs = get_vfs();
            const i64 saved = vfs.lseek(fd, 0, 1 /* SEEK_CUR */);
            if (saved < 0 || vfs.lseek(fd, static_cast<i64>(ctx.rcx), 0) < 0) {
                ctx.rax = sce_err(saved < 0 ? saved : -22);
                return;
            }
            const i64 r = n == 0 ? 0 : vfs.read(fd, mem.to_host_ptr<void>(ctx.rsi), n);
            vfs.lseek(fd, saved, 0);
            ctx.rax = r < 0 ? sce_err(r) : static_cast<u64>(r);
        });
    register_symbol(0xA226FBE85FF5D9F9ULL, "sceKernelLseek", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager&) {
            const i64 r = get_vfs().lseek(static_cast<i32>(ctx.rdi), static_cast<i64>(ctx.rsi),
                                          static_cast<i32>(ctx.rdx));
            ctx.rax = r < 0 ? sce_err(r) : static_cast<u64>(r);
        });
    register_symbol(0x901C023EC617FE6EULL, "sceKernelFstat", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager& mem) {
            if (!mem.is_valid_address(ctx.rsi, sizeof(OrbisStat))) {
                ctx.rax = sce_err(-14);
                return;
            }
            OrbisStat st{};
            const i32 r = get_vfs().fstat(static_cast<i32>(ctx.rdi), st);
            if (r < 0) {
                ctx.rax = sce_err(r);
                return;
            }
            mem.write_bytes(ctx.rsi, &st, sizeof(st));
            ctx.rax = 0;
        });

    // vprintf(const char* fmt, va_list ap): Mono's error reporter. Same va_list
    // walk as vsnprintf (first four GP arguments).
    register_symbol(0x18CA6FC4F156F76EULL, "vprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr fmt_addr = ctx.rdi, ap = ctx.rsi;
            if (fmt_addr == 0 || !mem.is_valid_address(fmt_addr, 1)) {
                ctx.rax = 0;
                return;
            }
            VaListArgs args(mem, ap);
            const std::string out = format_guest(mem, mem.to_host_ptr<char>(fmt_addr), args);
            std::cout << "[Guest printf] " << out << std::flush;
            // Mono reports g_assert failures through here; show where from.
            if (out.find("Assertion at") != std::string::npos) {
                GuestAddr ret = 0, fp = ctx.rbp;
                mem.read_value(ctx.rsp, ret);
                std::fprintf(stderr, "[assert-bt] TID=%d ret=0x%llx\n", (int)t_guest_tid, (unsigned long long)ret);
                for (int i = 0; i < 16 && fp > 0x1000 && mem.is_valid_address(fp, 16); ++i) {
                    GuestAddr next = 0, rip = 0;
                    mem.read_value(fp, next);
                    mem.read_value(fp + 8, rip);
                    std::fprintf(stderr, "[assert-bt]   #%d rip=0x%llx fp=0x%llx\n", i, (unsigned long long)rip,
                                 (unsigned long long)fp);
                    if (next <= fp) break;
                    fp = next;
                }
            }
            ctx.rax = out.size();
        });

    // vasprintf(char** strp, const char* fmt, va_list ap) -> length, or -1.
    // Mono formats GUIDs ("%02X%02X...") with it; the block is freed with free().
    register_symbol(0xAA3065C367153003ULL, "vasprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr strp = ctx.rdi, fmt_addr = ctx.rsi;
            if (strp == 0 || fmt_addr == 0 || !mem.is_valid_address(fmt_addr, 1)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            VaListArgs args(mem, ctx.rdx);
            const std::string out = format_guest(mem, mem.to_host_ptr<char>(fmt_addr), args);
            const GuestAddr buf = bump_alloc(out.size() + 1, 16);
            if (buf == 0 || !mem.write_bytes(buf, out.c_str(), out.size() + 1) ||
                !mem.write_value(strp, buf)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            ctx.rax = out.size();
        });

    // sprintf(char* buf, const char* fmt, ...) -> length
    register_symbol(0xB5C562E528AF17B4ULL, "sprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr buf = ctx.rdi, fmt_addr = ctx.rsi;
            if (fmt_addr == 0 || !mem.is_valid_address(fmt_addr, 1)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            CallArgs args(ctx, mem, 2);
            const std::string out = format_guest(mem, mem.to_host_ptr<char>(fmt_addr), args);
            if (buf == 0 || !mem.write_bytes(buf, out.c_str(), out.size() + 1)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            ctx.rax = out.size();
        });

    // _Stoul(const char* s, char** endptr, int base): Dinkumware's strtoul,
    // which libc's strtoul wraps. Parsed on the host from a bounded copy.
    // _Stoull is the same 64-bit parse (unsigned long is 64-bit on PS4).
    auto stoul_handler =
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr s = ctx.rdi, endptr = ctx.rsi;
            std::string text;
            for (GuestSize i = 0; i < 256; ++i) {
                char c = 0;
                if (!mem.read_value(s + i, c) || c == 0) break;
                text.push_back(c);
            }
            char* end = nullptr;
            const unsigned long long v = std::strtoull(text.c_str(), &end, static_cast<int>(ctx.rdx));
            if (endptr != 0) mem.write_value(endptr, static_cast<u64>(s + (end - text.c_str())));
            // 64-bit unsigned long on PS4; clamp like strtoul on overflow.
            ctx.rax = static_cast<u64>(v);
        };
    register_symbol(0xCE57C41FC166C940ULL, "_Stoul", "libc", stoul_handler);
    register_symbol(0xABEF44D17DDA5A95ULL, "_Stoull", "libc", stoul_handler);

    // Dinkumware ctype tables. _Getpctype/_Getptolower/_Getptoupper return
    // pointers into 16-bit tables indexed by (unsigned char) c, with the EOF
    // (-1) entry just before element 0; isalpha(c) is `tbl[c] & 0x212`. They
    // returned NULL, so Mono's tolower/isalpha read address 0 and mangled
    // every string it case-folded ("UTF-8" -> garbage).
    static std::once_flag s_ctype_once;
    static GuestAddr s_ctype = 0, s_tolower = 0, s_toupper = 0;
    auto ctype_tables = [](MemoryManager& mem) {
        std::call_once(s_ctype_once, [&mem] {
            constexpr u16 XD = 0x01, UP = 0x02, SP = 0x04, PU = 0x08, LO = 0x10, DI = 0x20, CN = 0x40, BB = 0x80;
            u16 ct[257] = {}, lo[257] = {}, up[257] = {};
            ct[0] = 0;
            lo[0] = up[0] = 0xFFFF; // EOF maps to EOF
            for (int c = 0; c < 256; ++c) {
                u16 f = 0;
                if (c < 0x20 || c == 0x7F) f = (c >= 0x09 && c <= 0x0D) ? CN : BB;
                else if (c == ' ') f = SP;
                else if (c >= '0' && c <= '9') f = DI | XD;
                else if (c >= 'A' && c <= 'Z') f = UP | ((c <= 'F') ? XD : 0);
                else if (c >= 'a' && c <= 'z') f = LO | ((c <= 'f') ? XD : 0);
                else if (c < 0x7F) f = PU;
                ct[c + 1] = f;
                lo[c + 1] = static_cast<u16>((c >= 'A' && c <= 'Z') ? c + 32 : c);
                up[c + 1] = static_cast<u16>((c >= 'a' && c <= 'z') ? c - 32 : c);
            }
            const GuestAddr base = bump_alloc(3 * sizeof(ct), 16);
            if (base == 0) return;
            mem.write_bytes(base, ct, sizeof(ct));
            mem.write_bytes(base + sizeof(ct), lo, sizeof(lo));
            mem.write_bytes(base + 2 * sizeof(ct), up, sizeof(up));
            s_ctype = base + 2;                   // &table[c] for c = 0
            s_tolower = base + sizeof(ct) + 2;
            s_toupper = base + 2 * sizeof(ct) + 2;
        });
    };
    register_symbol(0xB143F58416A8B8ECULL, "_Getpctype", "libc",
        [ctype_tables](CpuContext& ctx, MemoryManager& mem) { ctype_tables(mem); ctx.rax = s_ctype; });
    register_symbol(0xD6E260A15AB76D05ULL, "_Getptolower", "libc",
        [ctype_tables](CpuContext& ctx, MemoryManager& mem) { ctype_tables(mem); ctx.rax = s_tolower; });
    register_symbol(0xADC40252BD046915ULL, "_Getptoupper", "libc",
        [ctype_tables](CpuContext& ctx, MemoryManager& mem) { ctype_tables(mem); ctx.rax = s_toupper; });

    // _Iswctype(wint_t c, wctype_t desc)
    // NID: 0x0B25ECDA5FF590D0ULL ("CyXs2l_1kNA")
    register_symbol(0x0B25ECDA5FF590D0ULL, "_Iswctype", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const wint_t c = static_cast<wint_t>(ctx.rdi);
            const u32 desc = static_cast<u32>(ctx.rsi);
            bool ok = false;
            if (desc == 6) ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
            else if (desc == 10) ok = (c >= 'A' && c <= 'Z');
            else if (desc == 2) ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
            else if (desc == 4) ok = (c >= '0' && c <= '9');
            else if (c < 128) ok = std::iswalnum(c) != 0;
            ctx.rax = ok ? 1 : 0;
        });

    // _Towctrans(wint_t c, wctrans_t desc)
    // NID: 0x0DB12703E327548CULL ("DbEnA+MnVIw")
    register_symbol(0x0DB12703E327548CULL, "_Towctrans", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const wint_t c = static_cast<wint_t>(ctx.rdi);
            const u32 desc = static_cast<u32>(ctx.rsi);
            wint_t res = c;
            if (desc == 2) {
                if (c >= 'A' && c <= 'Z') res = c + ('a' - 'A');
                else res = std::towlower(c);
            } else if (desc == 1) {
                if (c >= 'a' && c <= 'z') res = c - ('a' - 'A');
                else res = std::towupper(c);
            }
            ctx.rax = static_cast<u64>(res);
        });

    // putchar(int c)
    register_symbol(0x9B9C0DF92C19391EULL, "putchar", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            std::cout << static_cast<char>(ctx.rdi & 0xFF) << std::flush;
            ctx.rax = ctx.rdi & 0xFF;
        });

    // getenv(name): the guest has no environment except what the host passes
    // explicitly as PS4_GUEST_<name> (e.g. PS4_GUEST_GC_DONT_GC=1). Values are
    // copied into guest memory once per name and stay valid, as getenv's are.
    register_symbol(0xB266D0BA47F16093ULL, "getenv", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            static std::mutex s_env_mtx;
            static std::unordered_map<std::string, GuestAddr> s_env;
            std::string name;
            for (GuestSize i = 0; i < 256; ++i) {
                char c = 0;
                if (!mem.read_value(ctx.rdi + i, c) || c == 0) break;
                name.push_back(c);
            }
            const char* host = std::getenv(("PS4_GUEST_" + name).c_str());
            static std::atomic<u32> s_logged{0};
            if (s_logged.fetch_add(1) < 32)
                std::fprintf(stderr, "[getenv] %s -> %s\n", name.c_str(), host ? host : "(unset)");
            if (!host) {
                ctx.rax = 0;
                return;
            }
            std::lock_guard<std::mutex> lock(s_env_mtx);
            GuestAddr& slot = s_env[name];
            if (slot == 0) {
                const std::string v(host);
                slot = bump_alloc(v.size() + 1, 16);
                if (slot != 0) mem.write_bytes(slot, v.c_str(), v.size() + 1);
            }
            ctx.rax = slot;
        });

    // sscanf(const char* str, const char* fmt, ...) / vsscanf(str, fmt, va_list).
    // Mono parses AOT symbol names with it ("specific_trampoline_lazy_fetch_%d").
    auto scan_input = [](MemoryManager& mem, GuestAddr str) {
        std::string in;
        for (GuestSize i = 0; i < (1u << 16); ++i) {
            char c = 0;
            if (!mem.read_value(str + i, c) || c == 0) break;
            in.push_back(c);
        }
        return in;
    };
    register_symbol(0xD4F934A99406796AULL, "sscanf", "libc",
        [scan_input](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rdi == 0 || ctx.rsi == 0 || !mem.is_valid_address(ctx.rsi, 1)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            CallArgs args(ctx, mem, 2);
            const int r = scan_guest(mem, scan_input(mem, ctx.rdi), mem.to_host_ptr<char>(ctx.rsi), args);
            ctx.rax = static_cast<u64>(static_cast<i64>(r));
        });
    register_symbol(0x513AE93952DCA0E0ULL, "vsscanf", "libc",
        [scan_input](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rdi == 0 || ctx.rsi == 0 || !mem.is_valid_address(ctx.rsi, 1)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            VaListArgs args(mem, ctx.rdx);
            const int r = scan_guest(mem, scan_input(mem, ctx.rdi), mem.to_host_ptr<char>(ctx.rsi), args);
            ctx.rax = static_cast<u64>(static_cast<i64>(r));
        });

    // fflush(FILE*): guest stdio is not buffered on our side.
    register_symbol(0x3148C2E256C7ACAEULL, "fflush", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; });

    // sceSystemServiceReportAbnormalTermination(const void* info): the game is
    // about to abort. Say so loudly instead of returning silently.
    register_symbol(0xDECF1C1E20812811ULL, "sceSystemServiceReportAbnormalTermination",
                    "libSceSystemService",
        [](CpuContext& ctx, MemoryManager&) {
            std::fprintf(stderr, "[HLE] sceSystemServiceReportAbnormalTermination(0x%llx) TID=%d "
                         "caller RIP=0x%llx -- the guest is aborting\n",
                         (unsigned long long)ctx.rdi, (int)t_guest_tid,
                         (unsigned long long)t_guest_rip);
            ctx.rax = SCE_OK;
        });

    // --- POSIX thread attributes / misc, as imported by mono-ps4.prx ---------
    // pthread_create looks attributes up by the guest pthread_attr_t's address
    // (s_pthread_attrs), exactly as the scePthreadAttr* handlers store them, so
    // the POSIX names alias those handlers. Return codes agree (0 = success).
    auto alias = [this](Nid nid, const char* name, Nid target) {
        if (const HleSymbol* t = find_by_nid(target); t != nullptr && t->handler)
            register_symbol(nid, name, "libScePosix", t->handler);
    };
    alias(0xC2D92DFED791D6CAULL, "pthread_attr_init", 0x9EC628351CB0C0D8ULL);
    alias(0xD90D33EAB9C1AD31ULL, "pthread_attr_setstacksize", 0x5135F325B5A18531ULL);
    alias(0x13EB72A37969E4BCULL, "pthread_attr_setdetachstate", 0xFD6ADEA6BB6ED10BULL);
    alias(0xCC772163C7EDE699ULL, "pthread_attr_destroy", 0xEB6282C04326CDC3ULL);
    alias(0x87D09C3F7274A153ULL, "pthread_join", 0xA27358F41CA7FD6FULL);

    register_symbol(0x5544F5652AC74F42ULL, "pthread_attr_getdetachstate", "libScePosix",
        [](CpuContext& ctx, MemoryManager& mem) {
            i32 state = 0;
            {
                std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
                if (auto it = s_pthread_attrs.find(ctx.rdi); it != s_pthread_attrs.end())
                    state = it->second.detachstate;
            }
            ctx.rax = (ctx.rsi != 0 && mem.write_value(ctx.rsi, state)) ? 0 : 22;
        });

    // scePthreadAttrGetstack(attr, void** addr, size_t* size)
    register_symbol(0xFEAB8F6B8484254CULL, "scePthreadAttrGetstack", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            ScePthreadAttr attr{};
            {
                std::lock_guard<std::mutex> lock(s_pthread_attr_mtx);
                if (auto it = s_pthread_attrs.find(ctx.rdi); it != s_pthread_attrs.end())
                    attr = it->second;
            }
            const bool ok = ctx.rsi != 0 && ctx.rdx != 0 &&
                            mem.write_value(ctx.rsi, static_cast<u64>(attr.stack_addr)) &&
                            mem.write_value(ctx.rdx, static_cast<u64>(attr.stack_size));
            ctx.rax = ok ? SCE_OK : static_cast<u64>(-22);
        });

    // Detached threads are still reaped at shutdown; nothing to do now.
    auto detach = [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; };
    register_symbol(0xF94D51E16B57BE87ULL, "pthread_detach", "libScePosix", detach);
    register_symbol(0xE2A1AB47A7A83FD6ULL, "scePthreadDetach", "libkernel", detach);

    // __pthread_cleanup_push_imp / _pop_imp: FreeBSD's out-of-line cleanup
    // handlers. They only run on cancellation, which is never delivered.
    register_symbol(0xD71BED515C75FD28ULL, "__pthread_cleanup_push_imp", "libScePosix",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; });
    register_symbol(0x896B0595831FDCACULL, "__pthread_cleanup_pop_imp", "libScePosix",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; });

    // Priorities are recorded on the thread but not used for host scheduling.
    register_symbol(0x5B41E99B65F4B8F1ULL, "scePthreadSetprio", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (GuestThread* t = get_thread_manager(mem).get_thread(static_cast<ScePthreadId>(ctx.rdi)))
                t->priority = static_cast<i32>(ctx.rsi);
            ctx.rax = SCE_OK;
        });
    register_symbol(0xD6D2B21BB465309AULL, "scePthreadGetprio", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            i32 prio = 256;
            if (GuestThread* t = get_thread_manager(mem).get_thread(static_cast<ScePthreadId>(ctx.rdi)))
                prio = t->priority;
            ctx.rax = (ctx.rsi != 0 && mem.write_value(ctx.rsi, prio)) ? SCE_OK : static_cast<u64>(-22);
        });

    // scePthreadGetname(thread, char name[32])
    register_symbol(0x1E8C3B07C39EB7A9ULL, "scePthreadGetname", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string name = "main";
            if (GuestThread* t = get_thread_manager(mem).get_thread(static_cast<ScePthreadId>(ctx.rdi)))
                name = t->name;
            name.resize(std::min<usize>(name.size(), 31));
            ctx.rax = (ctx.rsi != 0 && mem.write_bytes(ctx.rsi, name.c_str(), name.size() + 1))
                          ? SCE_OK : static_cast<u64>(-22);
        });

    // getpid(): a single fixed process.
    register_symbol(0x1E82D558D6A70417ULL, "getpid", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 1; });

    // clock_gettime / clock_getres(clockid_t, struct timespec*). FreeBSD ids:
    // REALTIME 0, MONOTONIC 4, UPTIME 5, THREAD_CPUTIME 14, PROCESS_CPUTIME 15.
    auto host_clock = [](u64 id) -> clockid_t {
        switch (id) {
        case 0: return CLOCK_REALTIME;
        case 14: return CLOCK_THREAD_CPUTIME_ID;
        case 15: return CLOCK_PROCESS_CPUTIME_ID;
        default: return CLOCK_MONOTONIC;
        }
    };
    register_symbol(0x94B313F6F240724DULL, "clock_gettime", "libkernel",
        [host_clock](CpuContext& ctx, MemoryManager& mem) {
            struct timespec ts{};
            ::clock_gettime(host_clock(ctx.rdi), &ts);
            const bool ok = ctx.rsi != 0 && mem.write_value(ctx.rsi, static_cast<i64>(ts.tv_sec)) &&
                            mem.write_value(ctx.rsi + 8, static_cast<i64>(ts.tv_nsec));
            ctx.rax = ok ? 0 : static_cast<u64>(-1);
        });
    register_symbol(0xB26223EDEAB3644FULL, "clock_getres", "libkernel",
        [host_clock](CpuContext& ctx, MemoryManager& mem) {
            struct timespec ts{};
            ::clock_getres(host_clock(ctx.rdi), &ts);
            if (ctx.rsi != 0) {
                mem.write_value(ctx.rsi, static_cast<i64>(ts.tv_sec));
                mem.write_value(ctx.rsi + 8, static_cast<i64>(ts.tv_nsec));
            }
            ctx.rax = 0;
        });

    register_symbol(0x421BF90110283847ULL, "sceKernelRemoveExceptionHandler", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            get_thread_manager(mem).install_exception_handler(static_cast<int>(ctx.rdi), 0);
            ctx.rax = SCE_OK;
        });

    // sceKernelInstallExceptionHandler: no signals are ever raised.
    // sceKernelInstallExceptionHandler(int signum, void (*handler)(int, void* ctx))
    // sceKernelRaiseException(ScePthread thread, int signum)
    // Delivered like signals by ThreadManager (Boehm GC's stop-the-world uses
    // signal 30: each GC thread's handler records its registers and stack
    // pointer, acknowledges on a semaphore and waits to be resumed).
    register_symbol(0x5A4C0477737BC346ULL, "sceKernelInstallExceptionHandler", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            static std::once_flag s_sentinel_once;
            std::call_once(s_sentinel_once, [&mem] {
                const GuestAddr hlt = bump_alloc(16, 16);
                const u8 code = 0xF4; // HLT: end of a delivered handler
                if (hlt != 0 && mem.write_bytes(hlt, &code, 1))
                    get_thread_manager(mem).set_exception_return_sentinel(hlt);
            });
            std::fprintf(stderr, "[exc] install signum=%llu handler=0x%llx TID=%d\n",
                         (unsigned long long)ctx.rdi, (unsigned long long)ctx.rsi, (int)t_guest_tid);
            get_thread_manager(mem).install_exception_handler(static_cast<int>(ctx.rdi), ctx.rsi);
            ctx.rax = SCE_OK;
        });
    register_symbol(0x8A5D379E5B8A7CC9ULL, "sceKernelRaiseException", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const bool ok = get_thread_manager(mem).raise_exception(static_cast<ScePthreadId>(ctx.rdi),
                                                                    static_cast<int>(ctx.rsi));
            static std::atomic<u32> s_logged{0};
            if (s_logged.fetch_add(1) < 16)
                std::fprintf(stderr, "[exc] raise signum=%llu -> TID %llu: %s (from TID %d)\n",
                             (unsigned long long)ctx.rsi, (unsigned long long)ctx.rdi,
                             ok ? "queued" : "no such thread", (int)t_guest_tid);
            ctx.rax = ok ? SCE_OK : 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH
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

    // _FSin(float x, unsigned int idx)
    // Dinkumware C runtime: idx=0 -> sin(x), idx=1 -> cos(x), idx=2 -> -sin(x), idx=3 -> -cos(x)
    auto fsin_handler = [](CpuContext& ctx, MemoryManager&) {
        const float x = ctx.ymm[0].f[0];
        const u32 idx = static_cast<u32>(ctx.rdi);
        float res = 0.0f;
        switch (idx & 3) {
            case 0: res = std::sin(x); break;
            case 1: res = std::cos(x); break;
            case 2: res = -std::sin(x); break;
            case 3: res = -std::cos(x); break;
        }
        ctx.ymm[0].f[0] = res;
    };
    register_symbol(0x66D8ECA64250FAFCULL, "_FSin", "libc", fsin_handler);

    // _Sin(double x, unsigned int idx)
    auto sin_dinkum_handler = [](CpuContext& ctx, MemoryManager&) {
        const double x = ctx.ymm[0].df[0];
        const u32 idx = static_cast<u32>(ctx.rdi);
        double res = 0.0;
        switch (idx & 3) {
            case 0: res = std::sin(x); break;
            case 1: res = std::cos(x); break;
            case 2: res = -std::sin(x); break;
            case 3: res = -std::cos(x); break;
        }
        ctx.ymm[0].df[0] = res;
    };
    register_symbol(0x7025E353BD99D0ECULL, "_Sin", "libc", sin_dinkum_handler);

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

    // atanf(float x)
    // NID: 0xC1E0EE83C403FE51ULL ("weDug8QD-lE")
    auto atanf_handler = [](CpuContext& ctx, MemoryManager&) {
        ctx.ymm[0].f[0] = std::atan(ctx.ymm[0].f[0]);
    };
    register_symbol(0xC1E0EE83C403FE51ULL, "atanf", "libc", atanf_handler);

    // asinf(float x) / acosf(float x). Unregistered, they returned 0 in rax and
    // left xmm0 untouched, so Dragon Ball Xenoverse 2 got a garbage float on
    // ~800 calls in a single boot.
    register_symbol(0x1995A317F6081459ULL, "asinf", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.ymm[0].f[0] = std::asin(ctx.ymm[0].f[0]);
        });
    register_symbol(0x408FF1D122FC8E1CULL, "acosf", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.ymm[0].f[0] = std::acos(ctx.ymm[0].f[0]);
        });

    // More libm that Dragon Ball Xenoverse 2 calls every frame once it is past
    // boot; unregistered, each returned 0 in rax and left xmm0 as it was.
    register_symbol(0xF33B2ED385CDB19EULL, "expf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::exp(ctx.ymm[0].f[0]); });
    register_symbol(0xC2E010B7F8FEA78AULL, "exp2f", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::exp2(ctx.ymm[0].f[0]); });
    register_symbol(0xB9545C336C8574FEULL, "fminf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::fmin(ctx.ymm[0].f[0], ctx.ymm[1].f[0]); });
    register_symbol(0x2F2C760F350BECB7ULL, "fmaxf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::fmax(ctx.ymm[0].f[0], ctx.ymm[1].f[0]); });
    // ldexpf(float x, int e) / ldexp(double x, int e): the exponent is in edi.
    register_symbol(0x927D32898784C600ULL, "ldexpf", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.ymm[0].f[0] = std::ldexp(ctx.ymm[0].f[0], static_cast<int>(ctx.rdi));
        });
    register_symbol(0x26BC0520CCCA36BDULL, "ldexp", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.ymm[0].df[0] = std::ldexp(ctx.ymm[0].df[0], static_cast<int>(ctx.rdi));
        });
    // Dinkumware _FLog(float x, int base) / _Log(double x, int base): the
    // backends of log (0), log10 (1) and log2 (-1).
    register_symbol(0x8E38D14BB97530F3ULL, "_FLog", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const float x = ctx.ymm[0].f[0];
            const int base = static_cast<int>(ctx.rdi);
            ctx.ymm[0].f[0] = base == 0 ? std::log(x) : base == 1 ? std::log10(x) : std::log2(x);
        });
    register_symbol(0xCD70A2EFC6D8AC42ULL, "_Log", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const double x = ctx.ymm[0].df[0];
            const int base = static_cast<int>(ctx.rdi);
            ctx.ymm[0].df[0] = base == 0 ? std::log(x) : base == 1 ? std::log10(x) : std::log2(x);
        });

    // Dinkumware _FDtest(const float* px): short classification code
    // _DENORM (-2), _FINITE (-1), _DZERO (0), _INFCODE (1), _NANCODE (2).
    register_symbol(0x9A7B9F3E561B9CDDULL, "_FDtest", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            float val = 0.0f;
            if (ctx.rdi != 0 && mem.is_valid_address(ctx.rdi, sizeof(float))) {
                mem.read_value(ctx.rdi, val);
            }
            const int cls = std::fpclassify(val);
            i16 ret = -1;
            switch (cls) {
                case FP_ZERO:      ret =  0; break;
                case FP_NORMAL:    ret = -1; break;
                case FP_SUBNORMAL: ret = -2; break;
                case FP_INFINITE:  ret =  1; break;
                case FP_NAN:       ret =  2; break;
                default:           ret = -1; break;
            }
            ctx.rax = static_cast<u64>(static_cast<i64>(ret));
        });

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
    // sceCommonDialogIsUsed
    register_symbol(0x050DED7B2D099903ULL, "sceCommonDialogIsUsed", "libSceCommonDialog",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0; // false: no common dialog currently in use
        });

    // memset(void* dest, int ch, size_t n)
    // NID: 0xF334C5BC120020DFULL ("8zTFvBIAIN8")
    register_symbol(0xF334C5BC120020DFULL, "memset", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const u8 ch          = static_cast<u8>(ctx.rsi & 0xFF);
            const GuestSize n    = ctx.rdx;
            if (dest != 0 && n > 0 && mem.is_valid_address(dest, n)) {
                if (watch_overlaps(dest, n)) {
                    // Slow path while a write-watch covers the range, so the
                    // store is recorded like any other guest write.
                    const std::vector<u8> fill(n, ch);
                    mem.write_bytes(dest, fill.data(), n);
                } else if (void* d_ptr = mem.to_host_ptr<void>(dest)) {
                    std::memset(d_ptr, ch, n);
                }
            }
            ctx.rax = dest;
        });

    // snprintf(char* str, size_t size, const char* format, ...)
    // NID: 0x78B743C3A974FDB5ULL ("eLdDw6l0-bU")
    // snprintf_s (C11 Annex K) has the same three fixed arguments; Dragon Ball
    // Xenoverse 2 formats floats with it ("%f"), and the unregistered stub left
    // the buffer unwritten.
    auto snprintf_handler =
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

            CallArgs args(ctx, mem, 3);
            const std::string formatted = format_guest(mem, fmt, args);
            if (buf != 0 && sz > 0 && mem.is_valid_address(buf, sz)) {
                char* dst = mem.to_host_ptr<char>(buf);
                if (dst) {
                    const usize copy_len = std::min<usize>(formatted.size(), sz - 1);
                    std::memcpy(dst, formatted.data(), copy_len);
                    dst[copy_len] = '\0';
                }
            }
            ctx.rax = formatted.size();
        };
    register_symbol(0x78B743C3A974FDB5ULL, "snprintf", "libc", snprintf_handler);
    register_symbol(0xDC1CAD3CE42054A7ULL, "snprintf_s", "libc", snprintf_handler);
    register_symbol(0xC44B332551A9C9BBULL, "sprintf_s", "libc", snprintf_handler);

    // swprintf / swprintf_s / vswprintf over 16-bit wchar_t: narrow the format
    // to UTF-8, format with the shared printf engine, widen the result. %ls
    // arguments are read as narrow strings (format_guest has no wide %s).
    auto wide_printf = [](MemoryManager& mem, GuestAddr buf, GuestSize n, GuestAddr fmt_addr,
                          auto& args) -> u64 {
        std::string fmt;
        for (GuestSize i = 0; i < 4096; ++i) {
            u16 wc = 0;
            if (!mem.read_value(fmt_addr + 2 * i, wc) || wc == 0) break;
            fmt.push_back(wc < 0x80 ? static_cast<char>(wc) : '?');
        }
        const std::string out = format_guest(mem, fmt.c_str(), args);
        std::vector<u16> wide;
        for (usize i = 0; i < out.size();) {
            const u8 b = static_cast<u8>(out[i]);
            u32 cp = b;
            usize len = 1;
            if (b >= 0xE0 && i + 2 < out.size()) {
                cp = ((b & 0x0Fu) << 12) | ((out[i + 1] & 0x3Fu) << 6) | (out[i + 2] & 0x3Fu);
                len = 3;
            } else if (b >= 0xC0 && i + 1 < out.size()) {
                cp = ((b & 0x1Fu) << 6) | (out[i + 1] & 0x3Fu);
                len = 2;
            }
            wide.push_back(static_cast<u16>(cp));
            i += len;
        }
        if (buf == 0 || n == 0) return static_cast<u64>(-1);
        const GuestSize count = std::min<GuestSize>(wide.size(), n - 1);
        if (count) mem.write_bytes(buf, wide.data(), count * 2);
        mem.write_value<u16>(buf + 2 * count, 0);
        return wide.size() < n ? wide.size() : static_cast<u64>(-1);
    };
    auto swprintf_handler = [wide_printf](CpuContext& ctx, MemoryManager& mem) {
        CallArgs args(ctx, mem, 3);
        ctx.rax = wide_printf(mem, ctx.rdi, ctx.rsi, ctx.rdx, args);
    };
    register_symbol(0x9C9CF5E89135B713ULL, "swprintf", "libc", swprintf_handler);
    register_symbol(0x226E79549FC17A47ULL, "swprintf_s", "libc", swprintf_handler);
    register_symbol(0xBB45CEB2E3A63B37ULL, "vswprintf", "libc",
        [wide_printf](CpuContext& ctx, MemoryManager& mem) {
            VaListArgs args(mem, ctx.rcx);
            ctx.rax = wide_printf(mem, ctx.rdi, ctx.rsi, ctx.rdx, args);
        });

    // strcpy_s / strcat_s (C11 Annex K): errno_t, 0 on success. Unregistered,
    // they returned 0 without copying, so Dragon Ball Xenoverse 2's CRI file
    // loader took "/app0/data.cpk" into an empty buffer and opened ''.
    auto read_guest_str = [](MemoryManager& mem, GuestAddr at, GuestSize limit, std::string& out) {
        out.clear();
        for (GuestSize i = 0; i < limit; ++i) {
            char c = 0;
            if (!mem.read_value(at + i, c)) return false;
            if (c == 0) return true;
            out.push_back(c);
        }
        return false; // no terminator within limit
    };
    constexpr u64 kEINVAL = 22, kERANGE = 34;
    // strcpy_s(char* dest, rsize_t destsz, const char* src)
    register_symbol(0xE576B600234409DAULL, "strcpy_s", "libc",
        [read_guest_str](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi, src = ctx.rdx;
            const GuestSize destsz = ctx.rsi;
            if (dest == 0 || destsz == 0 || destsz > 0x7FFFFFFF) {
                ctx.rax = kEINVAL;
                return;
            }
            std::string text;
            const bool terminated = src != 0 && read_guest_str(mem, src, destsz, text);
            if (!terminated) {
                mem.write_value<u8>(dest, 0);
                ctx.rax = src == 0 ? kEINVAL : kERANGE;
                return;
            }
            mem.write_bytes(dest, text.c_str(), text.size() + 1);
            ctx.rax = 0;
        });
    // strcat_s(char* dest, rsize_t destsz, const char* src)
    register_symbol(0x2BE81C9C51492957ULL, "strcat_s", "libc",
        [read_guest_str](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi, src = ctx.rdx;
            const GuestSize destsz = ctx.rsi;
            if (dest == 0 || destsz == 0 || destsz > 0x7FFFFFFF) {
                ctx.rax = kEINVAL;
                return;
            }
            std::string head, tail;
            if (!read_guest_str(mem, dest, destsz, head) || src == 0) {
                mem.write_value<u8>(dest, 0);
                ctx.rax = kEINVAL;
                return;
            }
            if (!read_guest_str(mem, src, destsz - head.size(), tail)) {
                mem.write_value<u8>(dest, 0);
                ctx.rax = kERANGE;
                return;
            }
            mem.write_bytes(dest + head.size(), tail.c_str(), tail.size() + 1);
            ctx.rax = 0;
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
            std::string host_path = resolved ? resolved->string() : std::string(path);
            if (host_path.ends_with(".sprx")) {
                std::string prx_cand = host_path.substr(0, host_path.length() - 5) + ".prx";
                if (std::filesystem::exists(prx_cand)) {
                    host_path = prx_cand;
                }
            }
            FILE* fp = std::fopen(host_path.c_str(), mode);
            if (fp) {
                std::lock_guard<std::mutex> lock(g_guest_stdio_mtx);
                g_guest_open_files.insert(reinterpret_cast<uintptr_t>(fp));
            }
            std::fprintf(stderr, "[LIBC fopen] fopen('%s' -> '%s', '%s') -> %p\n", path, host_path.c_str(), mode, static_cast<void*>(fp));
            ctx.rax = reinterpret_cast<u64>(fp);
        });

    // fseek(FILE* stream, long offset, int whence)
    // NID: 0xAD0155057A7F0B18ULL
    register_symbol(0xAD0155057A7F0B18ULL, "fseek", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            FILE* fp = resolve_guest_stream(ctx.rdi);
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
            FILE* fp = resolve_guest_stream(ctx.rdi);
            if (!fp) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const long pos = std::ftell(fp);
            std::fprintf(stderr, "[LIBC ftell] ftell(%p) -> %ld\n", static_cast<void*>(fp), pos);
            ctx.rax = static_cast<u64>(pos);
        });

    // fgetpos(FILE* stream, fpos_t* pos)
    // NID: 0x48796DEC484EB6A0ULL ("SHlt7EhOtqA")
    register_symbol(0x48796DEC484EB6A0ULL, "fgetpos", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            FILE* fp = resolve_guest_stream(ctx.rdi);
            const GuestAddr pos_ptr = ctx.rsi;
            if (!fp || pos_ptr == 0) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const long pos = std::ftell(fp);
            if (pos < 0) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            mem.write_value<u64>(pos_ptr, static_cast<u64>(pos));
            std::fprintf(stderr, "[LIBC fgetpos] fgetpos(%p) -> %ld\n", static_cast<void*>(fp), pos);
            ctx.rax = 0;
        });

    // setvbuf(FILE* stream, char* buf, int mode, size_t size)
    // NID: 0x40C1722E8A8DC488ULL ("QMFyLoqNxIg")
    register_symbol(0x40C1722E8A8DC488ULL, "setvbuf", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // fread(void* ptr, size_t size, size_t nmemb, FILE* stream)
    // NID: 0x95B07E52566A546DULL
    register_symbol(0x95B07E52566A546DULL, "fread", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dst_addr = ctx.rdi;
            const usize size = ctx.rsi;
            const usize nmemb = ctx.rdx;
            FILE* fp = resolve_guest_stream(ctx.rcx);
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
            FILE* fp = resolve_guest_stream(ctx.rdi);
            if (!fp || fp == stdin || fp == stdout || fp == stderr) {
                ctx.rax = 0;
                return;
            }
            {
                std::lock_guard<std::mutex> lock(g_guest_stdio_mtx);
                g_guest_open_files.erase(reinterpret_cast<uintptr_t>(fp));
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

            VaListArgs args(mem, ap);
            const std::string formatted = format_guest(mem, fmt, args);

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

                if (formatted.find("Error assigning") != std::string::npos) {
                    std::fprintf(stderr, "\n=== [DIAGNOSTIC] Material::SetTexture Error Detected ===\n");
                    std::fprintf(stderr, "  Message: %s\n", formatted.c_str());
                    // Stack backtrace
                    GuestAddr cur_rbp = ctx.rbp;
                    for (int f = 0; f < 10 && cur_rbp > 0x1000 && mem.is_valid_address(cur_rbp, 16); ++f) {
                        GuestAddr rip = 0, next_rbp = 0;
                        mem.read_value(cur_rbp + 8, rip);
                        mem.read_value(cur_rbp + 0, next_rbp);
                        std::fprintf(stderr, "  Frame #%d: RIP=0x%llx, RBP=0x%llx\n", f, (unsigned long long)rip, (unsigned long long)cur_rbp);

                        // Frame inside 0x644320..0x64444c (SetTextureProperty)
                        if (rip >= 0x644320 && rip <= 0x64444c) {
                            GuestAddr rbx_val = 0;
                            // In 0x644320: push rbp; push r15; push r14; push r13; push r12; push rbx; sub rsp, 0x78.
                            // Saved rbx is at cur_rbp - 0x28.
                            mem.read_value(cur_rbp - 0x28, rbx_val);
                            std::fprintf(stderr, "    Found SetTextureProperty frame! Saved rbx (PropertyEntry*)=0x%llx\n", (unsigned long long)rbx_val);
                            if (rbx_val > 0x1000 && mem.is_valid_address(rbx_val, 24)) {
                                GuestAddr tex_obj = 0;
                                u32 tex_dim = 0, prop_dim = 0, flag = 0;
                                mem.read_value(rbx_val + 0x00, tex_obj);
                                mem.read_value(rbx_val + 0x08, tex_dim);
                                mem.read_value(rbx_val + 0x0c, prop_dim);
                                mem.read_value(rbx_val + 0x10, flag);
                                std::fprintf(stderr, "    PropertyEntry: Texture*=0x%llx, texDim=%u, prop.texDim=%u, flag=%u\n",
                                             (unsigned long long)tex_obj, tex_dim, prop_dim, flag);
                                if (tex_obj > 0x1000 && mem.is_valid_address(tex_obj, 256)) {
                                    std::fprintf(stderr, "    Dumping Texture* object (256 bytes at 0x%llx):\n", (unsigned long long)tex_obj);
                                    for (int off = 0; off < 256; off += 16) {
                                        u32 w0 = 0, w1 = 0, w2 = 0, w3 = 0;
                                        mem.read_value(tex_obj + off + 0, w0);
                                        mem.read_value(tex_obj + off + 4, w1);
                                        mem.read_value(tex_obj + off + 8, w2);
                                        mem.read_value(tex_obj + off + 12, w3);
                                        std::fprintf(stderr, "      [+0x%02x]: 0x%08x 0x%08x 0x%08x 0x%08x\n", off, w0, w1, w2, w3);
                                    }
                                }
                            }
                        }
                        cur_rbp = next_rbp;
                    }
                    // Also inspect g_DefaultTextures at 0x1560cf0
                    if (mem.is_valid_address(0x1560cf0, 128)) {
                        GuestAddr tex2d_ptr = 0;
                        mem.read_value(0x1560cf0 + 0x50, tex2d_ptr);
                        std::fprintf(stderr, "  g_DefaultTextures (0x1560cf0) + 0x50 = 0x%llx\n", (unsigned long long)tex2d_ptr);
                        if (tex2d_ptr > 0x1000 && mem.is_valid_address(tex2d_ptr, 256)) {
                            std::fprintf(stderr, "  Dumping Default Texture2D (256 bytes at 0x%llx):\n", (unsigned long long)tex2d_ptr);
                            for (int off = 0; off < 256; off += 16) {
                                u32 w0 = 0, w1 = 0, w2 = 0, w3 = 0;
                                mem.read_value(tex2d_ptr + off + 0, w0);
                                mem.read_value(tex2d_ptr + off + 4, w1);
                                mem.read_value(tex2d_ptr + off + 8, w2);
                                mem.read_value(tex2d_ptr + off + 12, w3);
                                std::fprintf(stderr, "    [+0x%02x]: 0x%08x 0x%08x 0x%08x 0x%08x\n", off, w0, w1, w2, w3);
                            }
                        }
                    }
                    std::fprintf(stderr, "========================================================\n\n");
                }
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

    // strncat(char* dest, const char* src, size_t n)
    // NID: 0x907838E6A3C2E9FDULL ("kHg45qPC6f0")
    register_symbol(0x907838E6A3C2E9FDULL, "strncat", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestAddr src  = ctx.rsi;
            const GuestSize max_src = ctx.rdx;
            ctx.rax = dest;
            if (dest == 0 || src == 0) return;
            GuestAddr d = dest;
            u8 b = 0;
            while (mem.read_value(d, b) && b != 0) {
                ++d;
            }
            for (GuestSize i = 0; i < max_src; ++i) {
                u8 sb = 0;
                if (!mem.read_value(src + i, sb) || sb == 0) break;
                mem.write_value(d++, sb);
            }
            mem.write_value<u8>(d, 0);
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
            while (mem.read_value(str + len, ch) && ch != 0 && len < (u64{1} << 20)) {
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
            while (mem.read_value(hay + hlen, ch) && ch != 0 && hlen < (u64{1} << 20)) {
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

    // qsort / bsearch take a guest comparator, and HLE handlers never re-enter
    // the guest CPU to call it. Instead they run AS guest code: the routines
    // in src/hle/guest_libc.c are compiled to x86-64 (include/ps4/
    // guest_libc_blob.hpp), copied into guest memory on first use, and the
    // handler tail-jumps there with the caller's arguments still in place.
    // The routine's own `ret` returns to the caller, and the comparator is an
    // ordinary guest call.
    //
    // qsort used to leave the array in input order. bsearch was unregistered
    // and returned NULL: mono-ps4.prx uses it for every metadata table lookup
    // (206 calls before Unity aborted).
    static std::once_flag s_guest_libc_once;
    static GuestAddr s_guest_libc = 0;
    auto guest_libc_entry = [](MemoryManager& mem, usize offset) -> GuestAddr {
        std::call_once(s_guest_libc_once, [&mem] {
            const GuestAddr at = bump_alloc(sizeof(guest_libc::kCode), 64);
            if (at != 0 && mem.write_bytes(at, guest_libc::kCode, sizeof(guest_libc::kCode)))
                s_guest_libc = at;
        });
        return s_guest_libc == 0 ? 0 : s_guest_libc + offset;
    };

    // qsort(void* base, size_t nmemb, size_t size, int (*compar)(const void*, const void*))
    // NID: 0x00425D215653126AULL
    register_symbol(0x00425D215653126AULL, "qsort", "libc",
        [guest_libc_entry](CpuContext& ctx, MemoryManager& mem) {
            if (const GuestAddr fn = guest_libc_entry(mem, guest_libc::k_qsort_offset))
                ctx.rip = fn;
        });

    // bsearch(const void* key, const void* base, size_t n, size_t size, compar)
    // NID: 0x35EB0881399F1744ULL (hash-verified)
    register_symbol(0x35EB0881399F1744ULL, "bsearch", "libc",
        [guest_libc_entry](CpuContext& ctx, MemoryManager& mem) {
            if (const GuestAddr fn = guest_libc_entry(mem, guest_libc::k_bsearch_offset))
                ctx.rip = fn;
            else
                ctx.rax = 0;
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
                if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
                ctx.rax = 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }

            std::unique_lock<std::mutex> lock(sema->mtx);
            const u64 start_cancel_gen = sema->cancel_generation;
            auto pred = [&]() {
                return sema->count >= need || sema->deleted ||
                       (sema->cancel_generation != start_cancel_gen) ||
                       s_sem_shutdown.load(std::memory_order_relaxed) ||
                       (g_syscalls && g_syscalls->has_exited());
            };
            sema->waiters++;

            if (timo_ptr != 0 && mem.is_valid_address(timo_ptr, sizeof(u32))) {
                u32 usec = 0;
                mem.read_value(timo_ptr, usec);
                if (sema_log_enabled()) std::fprintf(stderr, "[Sema] WaitSema(%u, need=%d, timeout=%uus)\n", id, need, usec);
                if (!sema->cv.wait_for(lock, std::chrono::microseconds(usec), pred)) {
                    sema->waiters--;
                    if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
                    ctx.rax = 0x8002003CULL; // SCE_KERNEL_ERROR_ETIMEDOUT
                    return;
                }
            } else {
                if (sema_log_enabled()) std::fprintf(stderr, "[Sema] WaitSema(%u, need=%d, timeout=infinite) count=%u deleted=%d gen=%llu T%d ret=0x%llx\n",
                             id, need, sema->count, sema->deleted ? 1 : 0,
                             (unsigned long long)start_cancel_gen, (int)t_guest_tid,
                             (unsigned long long)t_guest_rip);
                SignalWakeScope wake(sema->cv);
                while (!pred()) {
                    sema->cv.wait_for(lock, std::chrono::milliseconds(50),
                                      [&] { return pred() || guest_signal_pending(); });
                    if (!pred()) {
                        lock.unlock();
                        deliver_guest_exceptions(mem);
                        lock.lock();
                    }
                }
                // Diagnostic: did we return without the semaphore actually being
                // satisfied? Only warn if this was NOT due to deletion, cancellation,
                // or emulator shutdown.
                if (sema->count < need && !sema->deleted &&
                    !s_sem_shutdown.load(std::memory_order_relaxed) &&
                    !(g_syscalls && g_syscalls->has_exited()) &&
                    sema->cancel_generation == start_cancel_gen) {
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
            if (s_sem_shutdown.load(std::memory_order_relaxed) ||
                (g_syscalls && g_syscalls->has_exited())) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
                ctx.rax = 0x80020055ULL; // SCE_KERNEL_ERROR_ECANCELED
                return;
            }
            if (sema->cancel_generation != start_cancel_gen) {
                if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
                ctx.rax = 0x80020055ULL; // SCE_KERNEL_ERROR_ECANCELED (FreeBSD 85)
                return;
            }
            if (sema->deleted) {
                if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
                ctx.rax = 0x80020003ULL; // SCE_KERNEL_ERROR_ESRCH
                return;
            }
            if (sema->count >= need) {
                sema->count -= need;
            }
            if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
            ctx.rax = SCE_OK;
        });

    // sceKernelSignalSema(SceKernelSema sema, int signalCount). Shared handler
    // also reused by the verified 0xE1CC... registration below.
    auto signal_sema_handler = [](CpuContext& ctx, MemoryManager& mem) {
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
            if (sema_log_enabled()) {
            GuestAddr ret = 0;
            if (mem.is_valid_address(ctx.rsp, sizeof(GuestAddr))) mem.read_value(ctx.rsp, ret);
            std::fprintf(stderr, "[Sema] SignalSema(%u, %d) -> count=%d (T%d ret=0x%llx)", id, sig, sema->count,
                         (int)t_guest_tid, (unsigned long long)ret);
            if (std::getenv("PS4_SEMA_CHAIN")) {
                GuestAddr fp = ctx.rbp;
                for (int f = 0; f < 10 && fp > 0x1000 && mem.is_valid_address(fp, 16); ++f) {
                    GuestAddr sfp = 0, ra = 0;
                    mem.read_value(fp, sfp); mem.read_value(fp + 8, ra);
                    std::fprintf(stderr, " <0x%llx", (unsigned long long)ra);
                    if (sfp <= fp) break;
                    fp = sfp;
                }
            }
            std::fprintf(stderr, "\n");
            }
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
            if (sema_log_enabled()) std::fprintf(stderr, "[EventFlag] Set(%u, 0x%x)\n", id, bits);
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
            if (!eventflag_satisfied(*evf, want, and_mode)) {
                ctx.rax = 0x80020010ULL; // EBUSY
                return;
            }
            // Same CLEAR_ALL (0x10) / CLEAR_PAT (0x20) bits as the wait.
            const u32 poll_mode = static_cast<u32>(ctx.rcx);
            if (poll_mode & 0x10u) evf->bits = 0;
            else if (poll_mode & 0x20u) evf->bits &= ~want;
            ctx.rax = SCE_OK;
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
            // SCE_KERNEL_EVF_WAITMODE_: AND 0x01, OR 0x02, CLEAR_ALL 0x10,
            // CLEAR_PAT 0x20. Clearing used to key off 0x02 (OR), so AND|CLEAR_ALL
            // waits (0x11) never cleared: Dragon Ball Xenoverse 2's render thread
            // sailed through its per-frame flags and rebuilt the command buffer
            // many times per flip, and the main thread submitted it half-built.
            const bool and_mode = (wait_mode & 1u) != 0;
            const bool clear_all = (wait_mode & 0x10u) != 0;
            const bool clear_pat = (wait_mode & 0x20u) != 0;

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
                           s_sem_shutdown.load(std::memory_order_relaxed) ||
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
                if (evf->deleted || s_sem_shutdown.load(std::memory_order_relaxed) ||
                    (g_syscalls && g_syscalls->has_exited())) {
                    if (s_sem_shutdown.load(std::memory_order_relaxed) ||
                        (g_syscalls && g_syscalls->has_exited())) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                    if (cur_th && cur_th->state.load() != ThreadState::Terminated) {
                        cur_th->state.store(ThreadState::Running);
                    }
                    ctx.rax = 0x80020001ULL; // SCE_KERNEL_ERROR_ESRCH
                    return;
                }
                if (ok) {
                    matched = evf->bits & want;
                    if (clear_all) evf->bits = 0;
                    else if (clear_pat) evf->bits &= ~want;
                }
            }

            if (out_ptr != 0 && mem.is_valid_address(out_ptr, sizeof(u32))) {
                mem.write_value(out_ptr, matched);
            }
            if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
            if (!ok) {
                std::fprintf(stderr, "[EventFlag] Wait(%u, want=0x%x, mode=0x%x) -> ETIMEDOUT after %uus%s\n",
                             id, want, wait_mode, usec,
                             guest_timeout ? " (guest timeout)" : " (no producer: capped)");
                ctx.rax = 0x8002003CULL; // SCE_KERNEL_ERROR_ETIMEDOUT
                return;
            }
            if (sema_log_enabled()) std::fprintf(stderr, "[EventFlag] Wait(%u, want=0x%x, mode=0x%x) -> matched 0x%x\n",
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
            static const bool s_log_sem = (std::getenv("PS4_LOG_SEM") != nullptr);
            if (s_log_sem) {
                std::fprintf(stderr, "[Sem] TID=%d post(0x%llx) -> count=%u waiters=%u\n",
                             (int)t_guest_tid, (unsigned long long)sem_ptr, sem[0], sem[1]);
            }
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
            const auto t0 = std::chrono::steady_clock::now();
            int iters = 0;
            {
                std::unique_lock<std::mutex> lock(s_sem_cv_mtx);
                bool waiter_marked = false;
                while (true) {
                    std::array<u32, 4> sem{};
                    if (!sem_read(mem, sem_ptr, sem)) {
                        valid = false;
                        break;
                    }
                    if (sem[0] > 0) {
                        sem[0] -= 1;
                        if (waiter_marked && sem[1] > 0) {
                            sem[1] -= 1;
                            waiter_marked = false;
                        }
                        sem_write(mem, sem_ptr, sem);
                        ok = true;
                        break;
                    }
                    if (g_syscalls && g_syscalls->has_exited()) break;
                    if (s_sem_shutdown.load(std::memory_order_relaxed)) break;
                    if (!waiter_marked) {
                        sem[1] += 1;
                        sem_write(mem, sem_ptr, sem);
                        waiter_marked = true;
                    }
                    ++iters;
                    {
                        SignalWakeScope wake(s_sem_cv);
                        if (!guest_signal_pending()) s_sem_cv.wait_for(lock, std::chrono::milliseconds(20));
                    }
                    if (guest_signal_pending()) {
                        lock.unlock();
                        deliver_guest_exceptions(mem);
                        lock.lock();
                    }
                }
                if (waiter_marked) {
                    std::array<u32, 4> sem{};
                    if (sem_read(mem, sem_ptr, sem) && sem[1] > 0) {
                        sem[1] -= 1;
                        sem_write(mem, sem_ptr, sem);
                    }
                }
            }
            if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
            const double waited_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (!valid) {
                std::fprintf(stderr, "[Sem] TID=%d wait(0x%llx) -> EINVAL (waited=%.2fms, iters=%d)\n",
                             (int)t_guest_tid, (unsigned long long)sem_ptr, waited_ms, iters);
                ctx.rax = static_cast<u64>(-22);
                return;
            }
            if (!ok) {
                // Process is shutting down. Park this worker thread until termination rather
                // than returning to guest code to spin endlessly.
                if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::WaitingSync);
                std::this_thread::sleep_for(std::chrono::seconds(10));
                ctx.rax = static_cast<u64>(0x8002003CULL);
                return;
            }
            static const bool s_log_sem = (std::getenv("PS4_LOG_SEM") != nullptr);
            if (s_log_sem) {
                std::fprintf(stderr, "[Sem] TID=%d wait(0x%llx) -> OK (waited=%.2fms, iters=%d)\n",
                             (int)t_guest_tid, (unsigned long long)sem_ptr, waited_ms, iters);
            }
            ctx.rax = SCE_OK;
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
            while (!ok && (g_syscalls == nullptr || !g_syscalls->has_exited()) && !s_sem_shutdown.load(std::memory_order_relaxed)) {
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
            if (cur_th && cur_th->state.load() != ThreadState::Terminated) cur_th->state.store(ThreadState::Running);
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

    // The rest of the mspace API. Every block comes from the shared CPU heap
    // (bump_alloc/heap_free) and `msp` is never dereferenced, so any non-zero
    // handle works. Create used to be a SyncStub answering 0: Unity then passed
    // a NULL mspace to sceLibcMspaceMalloc, which was unregistered and returned
    // 0, and the first allocation threw std::bad_alloc (Dragon Ball Xenoverse 2
    // aborted right after boot).

    // sceLibcMspaceCreate(...) -> mspace handle: a unique 16-byte control block.
    register_symbol(0xFE19F5B5C547AB94ULL, "sceLibcMspaceCreate", "libSceLibcInternal",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = bump_alloc(16, 16);
        });

    // sceLibcMspaceMalloc(mspace msp, size_t size)
    register_symbol(0x3898E6FD03881E52ULL, "sceLibcMspaceMalloc", "libSceLibcInternal",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = bump_alloc(ctx.rsi, 16);
        });

    // sceLibcMspaceCalloc(mspace msp, size_t n, size_t size): zeroed, since
    // freed blocks are reused.
    register_symbol(0x2D8A371A1225077FULL, "sceLibcMspaceCalloc", "libSceLibcInternal",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize n = ctx.rsi, size = ctx.rdx;
            if (size != 0 && n > ~GuestSize{0} / size) {
                ctx.rax = 0;
                return;
            }
            const GuestAddr p = bump_alloc(n * size, 16);
            if (p != 0 && n * size != 0) mem.zero_memory(p, n * size);
            ctx.rax = p;
        });

    // sceLibcMspaceFree(mspace msp, void* ptr)
    register_symbol(0x5656BF67E797971AULL, "sceLibcMspaceFree", "libSceLibcInternal",
        [](CpuContext& ctx, MemoryManager&) {
            heap_free(ctx.rsi);
            ctx.rax = 0;
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
                if (s_ptr && d_ptr && watch_overlaps(dest, n)) {
                    const std::vector<u8> tmp(static_cast<const u8*>(s_ptr), static_cast<const u8*>(s_ptr) + n);
                    mem.write_bytes(dest, tmp.data(), n);
                } else if (s_ptr && d_ptr) {
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
            while (idx < (u64{1} << 20)) {
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
            const u32 grain = (ctx.rcx > 0) ? static_cast<u32>(ctx.rcx) : 256;
            const u32 freq = (ctx.r8 > 0) ? static_cast<u32>(ctx.r8) : 48000;
            AudioPortConfig cfg{};
            cfg.grain = grain;
            cfg.sample_rate = freq;
            auto& am = get_audio_manager();
            const i32 handle = am.open_port(type, cfg);
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

            const u32 grain = am.get_port_grain(handle);
            const u32 sample_rate = am.get_port_sample_rate(handle);
            const usize sample_count = static_cast<usize>(grain) * 2; // stereo frames
            const usize byte_size = sample_count * sizeof(i16);

            if (grain == 0) {
                ctx.rax = static_cast<u64>(-14);
                return;
            }
            // A null buffer only waits for the port; either way the call blocks
            // for one grain, which is what paces the game's mixer thread.
            if (ptr != 0 && mem.is_valid_address(ptr, byte_size)) {
                if (const auto* pcm = mem.to_host_ptr<i16>(ptr))
                    am.output_samples(handle, std::span<const i16>(pcm, sample_count));
            }
            thread_local AudioDeadlinePacer s_audio_out_pacer;
            s_audio_out_pacer.pace(grain, sample_rate);
            ctx.rax = SCE_OK;
        });

    // sceAudioOutOutputs(SceAudioOutOutputParam *param, uint32_t num)
    // NID: 0xC373DD6924D2C061ULL ("w3PdaSTSwGE")
    register_symbol(0xC373DD6924D2C061ULL, "sceAudioOutOutputs", "libSceAudioOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr param_array = ctx.rdi;
            const u32 num = static_cast<u32>(ctx.rsi);
            auto& am = get_audio_manager();

            u32 max_grain = 0;
            u32 sample_rate = 48000;

            for (u32 i = 0; i < num; ++i) {
                const GuestAddr entry = param_array + (static_cast<GuestSize>(i) * 16);
                if (!mem.is_valid_address(entry, 16)) break;
                i32 handle = 0;
                GuestAddr ptr = 0;
                mem.read_value(entry + 0, handle);
                mem.read_value(entry + 8, ptr);

                const u32 grain = am.get_port_grain(handle);
                sample_rate = am.get_port_sample_rate(handle);
                if (grain > max_grain) max_grain = grain;

                const usize sample_count = static_cast<usize>(grain) * 2;
                const usize byte_size = sample_count * sizeof(i16);

                if (ptr != 0 && mem.is_valid_address(ptr, byte_size)) {
                    if (const auto* pcm = mem.to_host_ptr<i16>(ptr)) {
                        am.output_samples(handle, std::span<const i16>(pcm, sample_count));
                    }
                }
            }

            if (max_grain == 0) max_grain = 256;
            thread_local AudioDeadlinePacer s_audio_outs_pacer;
            s_audio_outs_pacer.pace(max_grain, sample_rate);
            ctx.rax = SCE_OK;
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

    // ==================================================================    // libSceAudioIn: Audio Input / Microphone
    // =========================================================================

    // sceAudioInOpen(int userId, int type, int index, uint32_t param)
    // NID: 0xE4D13C4A373B542FULL ("5NE8Sjc7VC8")
    register_symbol(0xE4D13C4A373B542FULL, "sceAudioInOpen", "libSceAudioIn",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 1; // valid audio in handle
        });

    // sceAudioInInput(int handle, void* dest)
    // NID: 0x2E8CC4394F3E6A73ULL ("LozEOU8+anM")
    register_symbol(0x2E8CC4394F3E6A73ULL, "sceAudioInInput", "libSceAudioIn",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rsi;
            // Dual channel 16-bit PCM @ 48kHz: 256 samples = 1024 bytes
            if (dest > 0x1000 && mem.is_valid_address(dest, 1024)) {
                mem.zero_memory(dest, 1024);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            ctx.rax = 256; // 256 samples read
        });

    // sceAudioInClose(int handle)
    // NID: 0x261E966C786723AFULL ("Jh6WbHhnI68")
    register_symbol(0x261E966C786723AFULL, "sceAudioInClose", "libSceAudioIn",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // libSceSaveData and libSceNpTrophy (see register_save_data_and_trophy).
    register_save_data_and_trophy(*this);

    // =========================================================================
    // libSceVideoOut / libSceSocialScreen entries that were falling through to
    // the generic unimplemented stub. Each of these is registered under the name
    // ps4libdoc's 42k corpus gives it (known_names.txt), so the log now states
    // what the guest actually called rather than "Unimplemented stub called:
    // NID=0x... returning 0". Several were firing on every run, at the call
    // sites cited below.
    // =========================================================================

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
    // =========================================================================
    // libSceVideoOut: Display and Frame Presentation
    // =========================================================================
    static std::vector<GuestAddr> s_registered_buffers;

    struct FlipSubscription {
        u32 eq_id{0};
        u64 udata{0};
    };
    static std::mutex s_flip_subs_mtx;
    static std::vector<FlipSubscription> s_flip_subscriptions;

    auto perform_flip = [](MemoryManager& mem, u32 buffer_index, i64 flip_arg) {
        // VBlank pacing. A PS4 flips at the next 60 Hz vblank and the game
        // paces itself on that; completing every flip at once let frame times
        // drift between 16 and 22 ms (judder). With a presenter attached
        // (the windowed runner sets the flip hook), hold each flip until the
        // next 16.67 ms tick; a frame that runs long resyncs instead of being
        // followed by a burst. PS4_NO_VSYNC=1 turns this off.
        static const bool s_vsync = std::getenv("PS4_NO_VSYNC") == nullptr;
        if (s_vsync && g_flip_hook) {
            using clock = std::chrono::steady_clock;
            constexpr auto kVblank = std::chrono::nanoseconds(16'666'667);
            static clock::time_point s_next_vblank{};
            const auto now = clock::now();
            if (s_next_vblank.time_since_epoch().count() == 0 || now > s_next_vblank + kVblank) {
                s_next_vblank = now; // first flip, or more than a frame late: resync
            } else if (now < s_next_vblank) {
                // sleep_until oversleeps by 1-2 ms on macOS; sleep to just short
                // of the tick and spin the remainder so frames stay evenly spaced.
                const auto coarse = s_next_vblank - std::chrono::microseconds(1500);
                if (now < coarse) std::this_thread::sleep_until(coarse);
                while (clock::now() < s_next_vblank) std::this_thread::yield();
            }
            s_next_vblank += kVblank;
        }
        const u64 count = s_flip_state.flip_count.fetch_add(1) + 1;
        const u64 now_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        s_flip_state.last_flip_time_us.store(now_us);
        s_flip_state.last_flip_arg.store(flip_arg);
        s_flip_state.current_buffer.store(buffer_index);

        PM4_LOG("[VideoOut] perform_flip #%llu: buffer=%u, flipArg=0x%llx\n",
                     (unsigned long long)count, buffer_index, (unsigned long long)flip_arg);
        // PS4_FLIP_STATS=1: once a second, the guest's frame rate and its worst
        // flip-to-flip gap (frame pacing, not just the average).
        static const bool s_flip_stats = std::getenv("PS4_FLIP_STATS") != nullptr;
        if (s_flip_stats) {
            static u64 s_prev_us = 0, s_window_start_us = 0, s_max_gap_us = 0;
            static u32 s_frames = 0, s_hitches = 0;
            if (s_prev_us != 0) {
                const u64 gap = now_us - s_prev_us;
                s_max_gap_us = std::max(s_max_gap_us, gap);
                if (gap > 20000) {
                    ++s_hitches;
                    std::fprintf(stderr, "[fps] hitch %.1f ms before flip #%llu\n", gap / 1000.0,
                                 (unsigned long long)count);
                }
            }
            s_prev_us = now_us;
            if (s_window_start_us == 0) s_window_start_us = now_us;
            ++s_frames;
            if (now_us - s_window_start_us >= 1000000) {
                std::fprintf(stderr, "[fps] guest %u fps, worst frame %.1f ms, frames >20ms: %u\n", s_frames,
                             s_max_gap_us / 1000.0, s_hitches);
                s_window_start_us = now_us;
                s_frames = s_hitches = 0;
                s_max_gap_us = 0;
            }
        }

        // If the GPU PM4 command processor executed draw calls, Metal already has
        // the rendered scene in its framebuffer. Only upload from guest memory if
        // CPU software rendering was performed (draw_count == 0) and the buffer contains non-zero data.
        const bool gpu_rendered = (g_pm4_translator && g_pm4_translator->draw_count() > 0);
        if (!gpu_rendered && g_metal_backend && buffer_index < s_registered_buffers.size()) {
            const GuestAddr buf_addr = s_registered_buffers[buffer_index];
            const usize pitch = g_metal_backend->width() * 4;
            const usize frame_size = pitch * g_metal_backend->height();
            if (buf_addr != 0 && mem.is_valid_address(buf_addr, frame_size)) {
                void* pixels = mem.to_host_ptr<void>(buf_addr);
                if (pixels) {
                    const u64* qwords = reinterpret_cast<const u64*>(pixels);
                    const usize qword_count = frame_size / sizeof(u64);
                    bool has_pixels = false;
                    for (usize q = 0; q < std::min<usize>(qword_count, 1024); ++q) {
                        if (qwords[q] != 0) {
                            has_pixels = true;
                            break;
                        }
                    }
                    if (has_pixels) {
                        g_metal_backend->upload_pixels(pixels, pitch);
                    }
                }
            }
        }

        // Flip completion: deliver SCE_VIDEO_OUT_EVENT_FLIP (ident 0,
        // EVFILT_VIDEO_OUT) to every equeue registered with
        // sceVideoOutAddFlipEvent -- and to no other queue. data packs
        // flipArg[63:16] | count[15:12] | time[11:0]; equeue_signal coalesces
        // the count while the event is still pending.
        if (g_thread_manager) {
            const u64 packed = (static_cast<u64>(flip_arg) << 16) | (1ULL << 12) | (now_us & 0xFFFULL);
            std::lock_guard<std::mutex> lock(s_flip_subs_mtx);
            for (const auto& sub : s_flip_subscriptions) {
                SceKernelEvent ev{};
                ev.ident = kVideoOutEventFlip;
                ev.filter = kEvFiltVideoOut;
                ev.flags = 0;
                ev.fflags = 0;
                ev.data = static_cast<i64>(packed);
                ev.udata = sub.udata;
                g_thread_manager->equeue_signal(sub.eq_id, ev);
            }
        }
        if (g_flip_hook) {
            g_flip_hook();
        } else if (g_pm4_translator && g_pm4_translator->frame_in_progress()) {
            g_pm4_translator->submit_frame(false);
            g_pm4_translator->begin_frame();
        }
    };

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
                if (g_metal_backend) {
                    g_metal_backend->register_display_buffer(buf_addr);
                }
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
    auto video_open_handler = [](CpuContext& ctx, MemoryManager& mem) {
        ctx.rax = 0x100; // valid video out handle
        std::fprintf(stderr, "[VideoOut] sceVideoOutOpen -> handle 0x100\n");
        ensure_unity_screen_resolution(mem);
    };
    register_symbol(0x529DFA3D393AF3B1ULL, "sceVideoOutOpen", "libSceVideoOut", video_open_handler);

    // sceVideoOutGetResolutionStatus
    auto res_status_handler = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr status_ptr = ctx.rsi;
        if (status_ptr != 0 && mem.is_valid_address(status_ptr, 16)) {
            mem.write_value(status_ptr + 0, static_cast<u32>(1920)); // width
            mem.write_value(status_ptr + 4, static_cast<u32>(1080)); // height
            mem.write_value(status_ptr + 8, static_cast<u32>(1920)); // paneWidth
            mem.write_value(status_ptr + 12, static_cast<u32>(1080)); // paneHeight
        }
        ensure_unity_screen_resolution(mem);
        ctx.rax = SCE_OK;
    };
    // NID: 0xEA43E78F9D53EB66ULL ("6kPnj51T62Y")
    register_symbol(0xEA43E78F9D53EB66ULL, "sceVideoOutGetResolutionStatus", "libSceVideoOut", res_status_handler);

    // sceVideoOutGetFlipStatus(int handle, SceVideoOutFlipStatus *status)
    // NID: 0x49B537770A7CD254ULL
    register_symbol(0x49B537770A7CD254ULL, "sceVideoOutGetFlipStatus", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr status_ptr = ctx.rsi;
            // SceVideoOutFlipStatus (64 bytes): count, processTime, tsc,
            // flipArg, submitTsc, reserved0, gcQueueNum, flipPendingNum,
            // currentBuffer, reserved1. Flips complete synchronously, so
            // nothing is ever queued or pending.
            if (status_ptr != 0 && mem.is_valid_address(status_ptr, 64)) {
                const u64 t = s_flip_state.last_flip_time_us.load();
                mem.write_value(status_ptr + 0, s_flip_state.flip_count.load());
                mem.write_value(status_ptr + 8, t);
                mem.write_value(status_ptr + 16, t);
                mem.write_value(status_ptr + 24, s_flip_state.last_flip_arg.load());
                mem.write_value(status_ptr + 32, t);
                mem.write_value(status_ptr + 40, static_cast<u64>(0));
                mem.write_value(status_ptr + 48, static_cast<i32>(0)); // gcQueueNum
                mem.write_value(status_ptr + 52, static_cast<i32>(0)); // flipPendingNum
                mem.write_value(status_ptr + 56, static_cast<i32>(s_flip_state.current_buffer.load()));
                mem.write_value(status_ptr + 60, static_cast<u32>(0));
            }
            ctx.rax = SCE_OK;
        });

    // sceVideoOutWaitVblank(int handle)
    // NID: 0x8FA45A01495A2EFDULL
    register_symbol(0x8FA45A01495A2EFDULL, "sceVideoOutWaitVblank", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            ctx.rax = SCE_OK;
        });

    // sceVideoOutAddFlipEvent(SceKernelEqueue eq, int videoHandle, void *udata)
    // NID: 0x1D7CE32BDC88DF49ULL ("HXzjK9yI30k")
    register_symbol(0x1D7CE32BDC88DF49ULL, "sceVideoOutAddFlipEvent", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager&) {
            const u32 eq_id = static_cast<u32>(ctx.rdi);
            const u64 udata = ctx.rdx;
            std::lock_guard<std::mutex> lock(s_flip_subs_mtx);
            s_flip_subscriptions.push_back({eq_id, udata});
            std::fprintf(stderr, "[VideoOut] AddFlipEvent: eq=%u, udata=0x%llx\n",
                         eq_id, (unsigned long long)udata);
            ctx.rax = SCE_OK;
        });

    // sceVideoOutSetBufferAttribute(SceVideoOutBufferAttribute *attr, int32_t format, uint32_t tmode, uint32_t aspect, uint32_t width, uint32_t height, uint32_t pitchInPixel)
    // NID: 0x8BAFEC47DD56B7FEULL ("i6-sR91Wt-4")
    register_symbol(0x8BAFEC47DD56B7FEULL, "sceVideoOutSetBufferAttribute", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr attr_ptr = ctx.rdi;
            const i32 format = static_cast<i32>(ctx.rsi);
            const u32 tmode  = static_cast<u32>(ctx.rdx);
            const u32 aspect = static_cast<u32>(ctx.rcx);
            u32 width  = static_cast<u32>(ctx.r8);
            u32 height = static_cast<u32>(ctx.r9);
            if (width == 0 || width == 8888 || width > 7680) width = 1920;
            if (height == 0 || height == 4999 || height > 4320) height = 1080;
            u32 pitch = width;
            if (ctx.rsp > 0x1000 && mem.is_valid_address(ctx.rsp + 8, sizeof(u32))) {
                u32 p = 0;
                mem.read_value(ctx.rsp + 8, p);
                if (p > 0 && p <= 7680 && p != 8888) pitch = p;
                else pitch = width;
            }
            GuestAddr caller_rip = 0;
            if (ctx.rsp > 0x1000 && mem.is_valid_address(ctx.rsp, sizeof(GuestAddr))) {
                mem.read_value(ctx.rsp, caller_rip);
            }
            std::fprintf(stderr, "[VideoOut] SetBufferAttribute(attr=0x%llx, fmt=0x%x, tmode=%u, aspect=%u, w=%u, h=%u, pitch=%u, caller=0x%llx)\n",
                         (unsigned long long)attr_ptr, format, tmode, aspect, width, height, pitch, (unsigned long long)caller_rip);
            if (attr_ptr > 0x1000 && mem.is_valid_address(attr_ptr, 32)) {
                mem.write_value(attr_ptr + 0x00, format);
                mem.write_value(attr_ptr + 0x04, tmode);
                mem.write_value(attr_ptr + 0x08, aspect);
                mem.write_value(attr_ptr + 0x0C, width);
                mem.write_value(attr_ptr + 0x10, height);
                mem.write_value(attr_ptr + 0x14, pitch);
                mem.write_value(attr_ptr + 0x18, static_cast<u64>(0));
            }
            ensure_unity_screen_resolution(mem);
            ctx.rax = SCE_OK;
        });

    // sceVideoOutSubmitFlip(int handle, int bufferIndex, unsigned int flipMode, int64_t flipArg)
    // NID: 0x538E8DC0E889A72BULL ("U46NwOiJpys")
    register_symbol(0x538E8DC0E889A72BULL, "sceVideoOutSubmitFlip", "libSceVideoOut",
        [perform_flip](CpuContext& ctx, MemoryManager& mem) {
            const u32 buf_idx = static_cast<u32>(ctx.rsi);
            const i64 flip_arg = static_cast<i64>(ctx.rcx);
            perform_flip(mem, buf_idx, flip_arg);
            ensure_unity_screen_resolution(mem);
            ctx.rax = SCE_OK;
        });

    // sceVideoOutSetFlipRate(int handle, int rate)
    // NID: 0x0818AEE26084D430ULL ("CBiu4mCE1DA")
    register_symbol(0x0818AEE26084D430ULL, "sceVideoOutSetFlipRate", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // sceVideoOutGetVblankStatus(int handle, SceVideoOutVblankStatus *status)
    // NID: 0xD456412B2F0778D5ULL ("1FZBKy8HeNU")
    register_symbol(0xD456412B2F0778D5ULL, "sceVideoOutGetVblankStatus", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr status_ptr = ctx.rsi;
            if (status_ptr != 0 && mem.is_valid_address(status_ptr, 16)) {
                using clock = std::chrono::steady_clock;
                static const auto s_epoch = clock::now();
                const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - s_epoch).count();
                const u64 vblank_count = static_cast<u64>(elapsed_us / 16666ULL);
                mem.write_value<u64>(status_ptr, vblank_count);
                mem.write_value<u64>(status_ptr + 8, static_cast<u64>(elapsed_us));
            }
            ctx.rax = SCE_OK;
        });

    register_symbol(0x0D886159B2527918ULL, "sceVideoOutColorSettingsSetGamma_", "libSceVideoOut",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    register_symbol(0xA6FF42239542F91DULL, "sceVideoOutAdjustColor_", "libSceVideoOut",
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
        if (data_ptr > 0x1000 && mem.is_valid_address(data_ptr, kScePadDataSize)) {
            write_pad_data(mem, data_ptr, read_pad_state());
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
                CallArgs args(ctx, mem, 1);
                const std::string formatted = format_guest(mem, fmt, args);
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
    // sceSysmoduleUnloadModule(id) / sceSysmoduleIsLoaded(id): every module id is
    // reported loaded and unload is a no-op, so the plugins' load/unload pairs and
    // their "is it up" checks all pass.
    register_symbol(0x791D9B6450005344ULL, "sceSysmoduleUnloadModule", "libSceSysmodule",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });
    register_symbol(0x7CC3F934750E68C9ULL, "sceSysmoduleIsLoaded", "libSceSysmodule",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // libSceImeDialog. There is no on-screen keyboard, so the dialog never waits
    // for a user: Init finishes it at once, GetStatus reports FINISHED and
    // GetResult reports a cancel. A plugin that polls GetStatus therefore leaves
    // its loop instead of spinning forever. Status values use the common-dialog
    // enum (NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3), the same enum the
    // sceXxxDialog families below use, so Init stores FINISHED and never RUNNING.
    // The USER_CANCELED end status follows the Orbis SDK as remembered; it is not
    // checked against a header in this repo.
    static std::atomic<u32> s_ime_dialog_status{0};
    // sceImeDialogInit(const SceImeDialogParam* param, ...) -- NID 0x354781ACDEE1CDFD
    register_symbol(0x354781ACDEE1CDFDULL, "sceImeDialogInit", "libSceImeDialog",
        [](CpuContext& ctx, MemoryManager&) {
            s_ime_dialog_status.store(3);   // SCE_COMMON_DIALOG_STATUS_FINISHED
            ctx.rax = SCE_OK;
        });
    // sceImeDialogTerm() -- NID 0x8324F2567F9B5CCC
    register_symbol(0x8324F2567F9B5CCCULL, "sceImeDialogTerm", "libSceImeDialog",
        [](CpuContext& ctx, MemoryManager&) {
            s_ime_dialog_status.store(0);   // NONE
            ctx.rax = SCE_OK;
        });
    // sceImeDialogGetStatus() -- NID 0x2000E60F8B527016
    register_symbol(0x2000E60F8B527016ULL, "sceImeDialogGetStatus", "libSceImeDialog",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = s_ime_dialog_status.load();
        });
    // sceImeDialogGetResult(SceImeDialogResult* result) -- NID 0xC74D63C6EFAFC657
    // Writes only the leading end-status word, USER_CANCELED (1).
    register_symbol(0xC74D63C6EFAFC657ULL, "sceImeDialogGetResult", "libSceImeDialog",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr result_addr = ctx.rdi;
            if (mem.is_valid_address(result_addr, sizeof(u32))) {
                mem.write_value<u32>(result_addr, 1);
            }
            ctx.rax = SCE_OK;
        });
    // sceImeDialogGetPanelSize(...) -- NID 0xC2AB09BD15F0979F
    // No panel is drawn, so the reported size is zero. Argument layout is not
    // confirmed; the output pointers in rsi/rdx are cleared when they look valid.
    register_symbol(0xC2AB09BD15F0979FULL, "sceImeDialogGetPanelSize", "libSceImeDialog",
        [](CpuContext& ctx, MemoryManager& mem) {
            for (const GuestAddr out : {ctx.rsi, ctx.rdx}) {
                if (out > 0x10000 && mem.is_valid_address(out, sizeof(u32))) {
                    mem.write_value<u32>(out, 0);
                }
            }
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
            thread_local AudioDeadlinePacer s_audio3d_pacer;
            s_audio3d_pacer.pace(256, 48000);
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
                // Unused slots hold SCE_USER_SERVICE_USER_ID_INVALID (-1), not 0.
                for (int i = 0; i < 4; ++i) mem.write_value(list_ptr + 4 * i, static_cast<i32>(i == 0 ? 0x10000001 : -1));
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
    // The first call reports SCE_USER_SERVICE_EVENT_TYPE_LOGIN (0) for the
    // initial user: Unity opens that user's pad on the login event, so without
    // it no scePadOpen/scePadRead ever happens and the game gets no input.
    register_symbol(0xC87D7B43A356B558ULL, "sceUserServiceGetEvent", "libSceUserService",
        [](CpuContext& ctx, MemoryManager& mem) {
            static std::atomic<bool> s_login_sent{false};
            if (ctx.rdi > 0x1000 && mem.is_valid_address(ctx.rdi, 8) && !s_login_sent.exchange(true)) {
                mem.write_value(ctx.rdi + 0, static_cast<i32>(0));          // eventType = LOGIN
                mem.write_value(ctx.rdi + 4, static_cast<i32>(0x10000001)); // userId
                ctx.rax = SCE_OK;
                return;
            }
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
                while (dst_len < (usize{1} << 20)) {
                    u8 b = 0;
                    if (!mem.is_valid_address(dst + dst_len, 1) || !mem.read_value(dst + dst_len, b)) break;
                    if (b == 0) break;
                    ++dst_len;
                }
                usize src_len = 0;
                while (src_len < (usize{1} << 20)) {
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
            // Gate the trace. Returning 0 means the caller gets nothing it can
            // advance on, so once the guest stops noticing, this re-enters
            // hundreds of millions of times -- it alone produced 4.6 GB of log
            // on a 30 B-instruction run and buried the write-watch ring.
            static std::atomic<u64> s_invoke_count{0};
            const u64 n = s_invoke_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 20 || (n % 1000000) == 0) {
                std::fprintf(stderr, "[Mono] runtime_invoke #%llu method=0x%llx obj=0x%llx params=0x%llx -> NULL (no IL)\n",
                             static_cast<unsigned long long>(n),
                             static_cast<unsigned long long>(method),
                             static_cast<unsigned long long>(obj),
                             static_cast<unsigned long long>(params));
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

    // PlayGo: the whole game is installed on the host, so every chunk is local.
    // Unregistered, scePlayGoGetLocus returned 0 without writing outLoci, so
    // Dragon Ball Xenoverse 2 read SCE_PLAYGO_LOCUS_NOT_DOWNLOADED and polled
    // it forever on a black screen.
    register_symbol(0xB6CE8695938A46B1ULL, "scePlayGoInitialize", "libScePlayGo",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = SCE_OK; });
    // scePlayGoOpen(ScePlayGoHandle* outHandle, const void* param)
    register_symbol(0x3351A66B5A1CAC61ULL, "scePlayGoOpen", "libScePlayGo",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rdi != 0) mem.write_value(ctx.rdi, static_cast<i32>(1));
            ctx.rax = SCE_OK;
        });
    // scePlayGoGetLocus(handle, const ScePlayGoChunkId* ids, u32 n, ScePlayGoLocus* out)
    register_symbol(0xB962182C5924C2A9ULL, "scePlayGoGetLocus", "libScePlayGo",
        [](CpuContext& ctx, MemoryManager& mem) {
            constexpr u8 kLocusLocalFast = 3;
            const GuestAddr out = ctx.rcx;
            const u32 n = static_cast<u32>(ctx.rdx);
            for (u32 i = 0; out != 0 && i < n; ++i) mem.write_value(out + i, kLocusLocalFast);
            ctx.rax = SCE_OK;
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
        // sceKernelGetProcessTimeCounterFrequency has a real handler: as a
        // sync stub it returned 0, and Unity divides by it to get seconds.
        {0xFE99E3DFFEDAE900ULL, "sceKernelSyncStub", "libkernel"}, // FMOD / pad init helper
        {0x8941B88CB45FD196ULL, "sceKernelSyncStub", "libkernel"}, // FMOD mixer thread setup
        {0x54EC7C3469875D3BULL, "sceKernelGetCpumode", "libkernel_cpumode"},
        {0xFEEC04868FFB80EBULL, "sceKernelSyncStub", "libkernel"}, // thread attr helper
        {0x22EC6752E5E4E818ULL, "sceKernelGetModuleList", "libkernel"},
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
        // pthread_key_create has a real handler (thread-specific data).
        // Stack-smashing abort, not scePthreadSetprio. Still a stub: the
        // compatibility layer prefers continuing over aborting the guest.
        {0x967DF7CE306B7E39ULL, "sceGnmGetTheTessellationFactorRingBufferBaseAddress", "libSceGnm"},
        {0xE6E7409BEE9BA158ULL, "sceGnmCreateWorkloadStream", "libSceGnm"},
        {0x8A1C6B6ECA122967ULL, "sceGnmBeginWorkload", "libSceGnm"},
    };
    for (const SyncStub& stub : kSyncStubs) {
        register_symbol(stub.nid, stub.name, stub.module,
            [](CpuContext& ctx, MemoryManager&) {
                ctx.rax = SCE_OK;
            });
    }

    // Helper to write PM4 NOP padding for remaining space in a DCB reservation
    auto write_dcb_nop_padding = [](MemoryManager& mem, GuestAddr addr, u32 num_dwords) {
        if (num_dwords == 0) return;
        if (num_dwords == 1) {
            mem.write_value<u32>(addr, 0x80000000U);
        } else {
            std::vector<u32> words(num_dwords, 0U);
            words[0] = 0xC0001000U | ((num_dwords - 2) << 16);
            mem.write_bytes(addr, words.data(), static_cast<GuestSize>(num_dwords) * sizeof(u32));
        }
    };

    // GNM DCB-writing stubs: the caller reserves `rsi` DWORDs at `rdi` in the DCB.
    // Each stub fills the reserved range with a valid PM4 Type-3 IT_NOP packet
    // (0xC0001000 | ((n-2) << 16)), preserving packet cursor alignment so that
    // subsequent PM4 packets in the command stream do not desynchronize.
    struct DcbWriterStub {
        Nid nid;
        const char* name;
        const char* module;
    };
    static const DcbWriterStub kDcbWriterStubs[] = {
        {0x7050A9D0D5FCC1FDULL, "sceGnmSetVgtControl", "libSceGnm"},
        {0x103F7F163AFC4DAEULL, "sceGnmDrawIndexIndirect", "libSceGnm"},
        {0x678DEF2A9E64EEBDULL, "sceGnmDispatchIndirect", "libSceGnm"},
        {0xF8016F3845EB2899ULL, "sceGnmSetEmbeddedVsShader", "libSceGnm"},
        {0x5FD3A6C3D770BF93ULL, "sceGnmSetEmbeddedPsShader", "libSceGnm"},
        {0x1541C6F2C437479FULL, "sceGnmSetEsShader", "libSceGnm"},
        {0x509C0DB8C05C5009ULL, "sceGnmSetGsShader", "libSceGnm"},
        {0x54936316DAA2179CULL, "sceGnmSetHsShader", "libSceGnm"},
        {0xD6A5CB1C8A5138F1ULL, "sceGnmInsertWaitFlipDone", "libSceGnm"},
        {0x5B512D8FF8E55BB6ULL, "sceGnmInsertPushMarker", "libSceGnm"},
        {0xEEA65536012EF926ULL, "sceGnmInsertPopMarker", "libSceGnm"},
        {0x8E222DCD2EBEDB68ULL, "sceGnmInsertSetMarker", "libSceGnm"},
        // Imported by Unity titles but never emulated: an unresolved stub
        // would leave the reserved DWORDs stale and break the PM4 stream.
        {0x68F2192535C2F9C5ULL, "sceGnmInsertPushColorMarker", "libSceGnm"},
        {0x6A3DCBFE26859B29ULL, "sceGnmInsertSetColorMarker", "libSceGnm"},
        {0x5110E025C5E140EBULL, "sceGnmInsertThreadTraceMarker", "libSceGnm"},
        {0x35FBCEACDCF2EAC9ULL, "sceGnmInsertDingDongMarker", "libSceGnm"},
        {0x31846D621A2329D0ULL, "sceGnmResetVgtControl", "libSceGnm"},
        {0x7DFACD40EB21A30BULL, "sceGnmComputeWaitOnAddress", "libSceGnm"},
        {0x1096A9365DBEA605ULL, "sceGnmComputeWaitSemaphore", "libSceGnm"},
        {0xBDC91DCDB438E922ULL, "sceGnmSetLsShader", "libSceGnm"},
        {0x9CB3368B6FBAE610ULL, "sceGnmUpdateGsShader", "libSceGnm"},
        {0x18D971FB2EF13DD1ULL, "sceGnmUpdateHsShader", "libSceGnm"},
        {0xC040F8657085253DULL, "sceGnmDispatchIndirectOnMec", "libSceGnm"},
        {0x8A6D99B88B5A6EEEULL, "sceGnmDrawInitToDefaultContextState400", "libSceGnm"},
        {0x9C5E9B1515014405ULL, "sceGnmDispatchInitDefaultHardwareState", "libSceGnm"},
        {0xE2FFA8B4821D8EA8ULL, "sceGnmDrawIndirect", "libSceGnm"},
        {0x7140A8F0EBC0AEBCULL, "sceGnmDrawIndirectCountMulti", "libSceGnm"},
        {0xB616CF706EC4EEA9ULL, "sceGnmDrawIndexIndirectCountMulti", "libSceGnm"},
        {0x8C774FBC8CE5A4A7ULL, "sceGnmDrawIndexMultiInstanced", "libSceGnm"},
        {0xB2D0D2616D920553ULL, "sceGnmDrawOpaqueAuto", "libSceGnm"},
    };
    for (const DcbWriterStub& stub : kDcbWriterStubs) {
        register_symbol(stub.nid, stub.name, stub.module,
            [write_dcb_nop_padding](CpuContext& ctx, MemoryManager& mem) {
                const GuestAddr dcb_addr = ctx.rdi;
                const u32 num_dwords = static_cast<u32>(ctx.rsi);
                if (num_dwords > 0 && num_dwords <= 65536 &&
                    mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                    write_dcb_nop_padding(mem, dcb_addr, num_dwords);
                }
                ctx.rax = SCE_OK;
            });
    }

    register_symbol(0x8E0DF7AC428B7D5BULL, "sceGnmIsUserPaEnabled", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // sceGnmDrawInitDefaultHardwareState{,175,200,350}(u32* cmd, u64 numDwords)
    // -> DWORDs written; the caller advances its command pointer by the return
    // value. As a sync stub it returned 0 and wrote nothing, so Dragon Ball
    // Xenoverse 2's command buffers lost their 0x100-DWORD init block, the
    // packets after it no longer lined up with the sizes the game submitted,
    // and every frame came out black. Reserve the same 0x100 DWORDs the real
    // driver does as one NOP; the PM4 translator keeps its own default state.
    struct HwInitStub {
        Nid nid;
        const char* name;
    };
    static constexpr HwInitStub kHwInitStubs[] = {
        {0x21D7DFC1FDF287CBULL, "sceGnmDrawInitDefaultHardwareState"},
        {0x4219F245EB5E2753ULL, "sceGnmDrawInitDefaultHardwareState175"},
        {0xD07DAF0586D32C72ULL, "sceGnmDrawInitDefaultHardwareState200"},
        {0xC9BD9C4616A00F52ULL, "sceGnmDrawInitDefaultHardwareState350"},
    };
    for (const HwInitStub& stub : kHwInitStubs) {
        register_symbol(stub.nid, stub.name, "libSceGnm",
            [write_dcb_nop_padding](CpuContext& ctx, MemoryManager& mem) {
                constexpr u32 kHwInitDwords = 0x100;
                const GuestAddr dcb_addr = ctx.rdi;
                if (ctx.rsi < kHwInitDwords ||
                    !mem.is_valid_address(dcb_addr, GuestSize{kHwInitDwords} * sizeof(u32))) {
                    ctx.rax = 0;
                    return;
                }
                write_dcb_nop_padding(mem, dcb_addr, kHwInitDwords);
                ctx.rax = kHwInitDwords;
            });
    }

    // sceGnmDrawInitToDefaultContextState(u32* cmd, u32 numDwords) -> DWORDs
    // written: the same contract with a 0x20-DWORD block.
    register_symbol(0xF251F9E2C7E37E65ULL, "sceGnmDrawInitToDefaultContextState", "libSceGnm",
        [write_dcb_nop_padding](CpuContext& ctx, MemoryManager& mem) {
            constexpr u32 kCtxInitDwords = 0x20;
            const GuestAddr dcb_addr = ctx.rdi;
            if (ctx.rsi < kCtxInitDwords ||
                !mem.is_valid_address(dcb_addr, GuestSize{kCtxInitDwords} * sizeof(u32))) {
                ctx.rax = 0;
                return;
            }
            write_dcb_nop_padding(mem, dcb_addr, kCtxInitDwords);
            ctx.rax = kCtxInitDwords;
        });

    // sceGnmDrawIndexAuto -- writes IT_DRAW_INDEX_AUTO (0x2D, 3 DWORDs) + NOP padding
    register_symbol(0x186B27EE3313C70EULL, "sceGnmDrawIndexAuto", "libSceGnm",
        [write_dcb_nop_padding](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dcb_addr = ctx.rdi;
            const u32 num_dwords = static_cast<u32>(ctx.rsi);
            const u32 index_count = static_cast<u32>(ctx.rdx);
            const u32 initiator = ctx.rcx ? static_cast<u32>(ctx.rcx) : 2U; // DI_SRC_SEL_AUTO_INDEX = 2

            if (num_dwords >= 3 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                u32 pkt[3];
                pkt[0] = 0xC0012D00U; // Type 3, count=1, opcode=0x2D (IT_DRAW_INDEX_AUTO)
                pkt[1] = index_count;
                pkt[2] = initiator;
                mem.write_bytes(dcb_addr, pkt, sizeof(pkt));
                write_dcb_nop_padding(mem, dcb_addr + sizeof(pkt), num_dwords - 3);
            } else if (num_dwords > 0 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                write_dcb_nop_padding(mem, dcb_addr, num_dwords);
            }
            ctx.rax = SCE_OK;
        });

    // sceGnmDrawIndexOffset(cmd, numDwords, indexOffset, indexCount, flags) --
    // IT_DRAW_INDEX_OFFSET_2 (0x35): [MAX_SIZE, INDEX_OFFSET, INDEX_COUNT,
    // DRAW_INITIATOR=DMA], drawing from the index base set earlier.
    register_symbol(0xA1833E6337C29B66ULL, "sceGnmDrawIndexOffset", "libSceGnm",
        [write_dcb_nop_padding](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dcb_addr = ctx.rdi;
            const u32 num_dwords = static_cast<u32>(ctx.rsi);
            const u32 index_offset = static_cast<u32>(ctx.rdx);
            const u32 index_count = static_cast<u32>(ctx.rcx);
            if (num_dwords == 0 || !mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                ctx.rax = SCE_OK;
                return;
            }
            if (num_dwords >= 5) {
                const u32 pkt[5] = {0xC0033500U, index_offset + index_count, index_offset, index_count, 0U};
                mem.write_bytes(dcb_addr, pkt, sizeof(pkt));
                write_dcb_nop_padding(mem, dcb_addr + sizeof(pkt), num_dwords - 5);
            } else {
                write_dcb_nop_padding(mem, dcb_addr, num_dwords);
            }
            ctx.rax = SCE_OK;
        });

    // sceGnmDrawIndex -- writes IT_DRAW_INDEX_2 (0x27, 6 DWORDs) + NOP padding, matching
    // Gnm: [MAX_SIZE, INDEX_BASE_LO, INDEX_BASE_HI, INDEX_COUNT, DRAW_INITIATOR=DMA].
    // Topology comes from VGT_PRIMITIVE_TYPE, not from the packet.
    register_symbol(0x1E54CFA19FE863B6ULL, "sceGnmDrawIndex", "libSceGnm",
        [write_dcb_nop_padding](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dcb_addr = ctx.rdi;
            const u32 num_dwords = static_cast<u32>(ctx.rsi);
            const u32 index_count = static_cast<u32>(ctx.rdx);
            const GuestAddr index_buffer_addr = ctx.rcx;
            const u32 base_vertex = static_cast<u32>(ctx.r8);
            const u32 instance_count = static_cast<u32>(ctx.r9 ? ctx.r9 : 1);

            PM4_LOG("[HLE sceGnmDrawIndex] dcb=0x%llx dwords=%u idx_count=%u idx_buf=0x%llx (rcx) base_vtx=%u instances=%u\n",
                         (unsigned long long)dcb_addr, num_dwords, index_count,
                         (unsigned long long)index_buffer_addr, base_vertex, instance_count);

            if (num_dwords >= 6 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                u32 pkt[6];
                pkt[0] = 0xC0042700U; // Type 3, count=4, opcode=0x27 (IT_DRAW_INDEX_2)
                pkt[1] = index_count;
                pkt[2] = static_cast<u32>(index_buffer_addr & 0xFFFFFFFFU);
                pkt[3] = static_cast<u32>((index_buffer_addr >> 32) & 0xFFFFFFFFU);
                pkt[4] = index_count;
                pkt[5] = 0U; // DRAW_INITIATOR: SOURCE_SELECT=DI_SRC_SEL_DMA
                mem.write_bytes(dcb_addr, pkt, sizeof(pkt));
                write_dcb_nop_padding(mem, dcb_addr + sizeof(pkt), num_dwords - 6);
            } else if (num_dwords > 0 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                write_dcb_nop_padding(mem, dcb_addr, num_dwords);
            }
            ctx.rax = SCE_OK;
        });

    // Shared handler for sceGnmSetVsShader & sceGnmUpdateVsShader:
    // rdx points to Gnm::VsStageRegisters in guest memory:
    // +0x00: m_spiShaderPgmLoVs, +0x04: m_spiShaderPgmHiVs
    // +0x08: m_spiShaderPgmRsrc1Vs, +0x0C: m_spiShaderPgmRsrc2Vs
    // +0x10: m_spiVsOutConfig, +0x14: m_spiShaderPosFormat, +0x18: m_paClVsOutCntl
    auto vs_shader_handler = [write_dcb_nop_padding](const char* func_name, CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr dcb_addr = ctx.rdi;
        const u32 num_dwords = static_cast<u32>(ctx.rsi);
        const GuestAddr vs_regs_addr = ctx.rdx;
        const u32 modifier = static_cast<u32>(ctx.rcx);

        u32 pgm_lo = 0, pgm_hi = 0, rsrc1 = 0, rsrc2 = 0;
        u32 out_cfg = 0, pos_fmt = 0, pa_cl = 0;
        if (vs_regs_addr && mem.is_valid_address(vs_regs_addr, 28)) {
            mem.read_value(vs_regs_addr + 0x00, pgm_lo);
            mem.read_value(vs_regs_addr + 0x04, pgm_hi);
            mem.read_value(vs_regs_addr + 0x08, rsrc1);
            mem.read_value(vs_regs_addr + 0x0C, rsrc2);
            mem.read_value(vs_regs_addr + 0x10, out_cfg);
            mem.read_value(vs_regs_addr + 0x14, pos_fmt);
            mem.read_value(vs_regs_addr + 0x18, pa_cl);
        }
        const u64 isa_addr = ((static_cast<u64>(pgm_hi) << 32) | pgm_lo) << 8;
        PM4_LOG("[HLE %s] dcb=0x%llx dwords=%u vs_regs=0x%llx mod=0x%x -> VS ISA: 0x%llx (lo=0x%08x hi=0x%08x rsrc1=0x%08x rsrc2=0x%08x out_cfg=0x%08x pos_fmt=0x%08x pa_cl=0x%08x)\n",
                     func_name, (unsigned long long)dcb_addr, num_dwords, (unsigned long long)vs_regs_addr, modifier,
                     (unsigned long long)isa_addr, pgm_lo, pgm_hi, rsrc1, rsrc2, out_cfg, pos_fmt, pa_cl);

        if (num_dwords >= 6 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
            std::vector<u32> pkt = {
                0xC0047600U, // Type 3, count=4 (5 payload dwords), opcode=0x76 (IT_SET_SH_REG)
                0x0048U,     // mmSPI_SHADER_PGM_LO_VS
                pgm_lo, pgm_hi, rsrc1, rsrc2,
            };
            // The stage's context registers, as the real library writes them.
            const std::vector<u32> ctx_regs = {
                0xC0016900U, 0x01B1U, out_cfg,   // SPI_VS_OUT_CONFIG
                0xC0016900U, 0x01C3U, pos_fmt,   // SPI_SHADER_POS_FORMAT
                0xC0016900U, 0x0207U, pa_cl,     // PA_CL_VS_OUT_CNTL
            };
            if (num_dwords >= pkt.size() + ctx_regs.size()) pkt.insert(pkt.end(), ctx_regs.begin(), ctx_regs.end());
            mem.write_bytes(dcb_addr, pkt.data(), pkt.size() * sizeof(u32));
            write_dcb_nop_padding(mem, dcb_addr + pkt.size() * sizeof(u32), num_dwords - static_cast<u32>(pkt.size()));
        } else if (num_dwords > 0 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
            write_dcb_nop_padding(mem, dcb_addr, num_dwords);
        }
        ctx.rax = SCE_OK;
    };

    register_symbol(0x8008429FA5225386ULL, "sceGnmSetVsShader", "libSceGnm",
        [vs_shader_handler](CpuContext& ctx, MemoryManager& mem) {
            vs_shader_handler("sceGnmSetVsShader", ctx, mem);
        });

    register_symbol(0x577D55D3552249C6ULL, "sceGnmUpdateVsShader", "libSceGnm",
        [vs_shader_handler](CpuContext& ctx, MemoryManager& mem) {
            vs_shader_handler("sceGnmUpdateVsShader", ctx, mem);
        });

    // sceGnmSetPsShader350 -- writes IT_SET_SH_REG (0x76, count=4, base_reg=0x0008 mmSPI_SHADER_PGM_LO_PS) + NOP padding
    // rdx points to Gnm::PsStageRegisters in guest memory:
    // +0x00: m_spiShaderPgmLoPs, +0x04: m_spiShaderPgmHiPs
    // +0x08: m_spiShaderPgmRsrc1Ps, +0x0C: m_spiShaderPgmRsrc2Ps
    // +0x10: m_spiShaderZFormat, +0x14: m_spiShaderColFormat, +0x18: m_spiPsInputEna
    // +0x1C: m_spiPsInputAddr, +0x20: m_spiPsInControl, +0x24: m_spiBarycCntl
    // +0x28: m_dbShaderControl, +0x2C: m_cbShaderMask
    // Shared by sceGnmSetPsShader(350) and sceGnmUpdatePsShader(350): the update
    // variants rewrite a block reserved earlier in the DCB, so leaving it
    // untouched (the old unresolved stub) left stale bytes that desynchronised
    // the PM4 stream and dropped the frame's EOP interrupt.
    auto ps_shader_handler = [write_dcb_nop_padding](const char* func_name, CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dcb_addr = ctx.rdi;
            const u32 num_dwords = static_cast<u32>(ctx.rsi);
            const GuestAddr ps_regs_addr = ctx.rdx;
            const u32 modifier = static_cast<u32>(ctx.rcx);

            u32 pgm_lo = 0, pgm_hi = 0, rsrc1 = 0, rsrc2 = 0;
            u32 z_fmt = 0, col_fmt = 0, in_ena = 0, in_addr = 0;
            u32 in_ctrl = 0, baryc = 0, db_ctrl = 0, cb_mask = 0;
            if (ps_regs_addr && mem.is_valid_address(ps_regs_addr, 48)) {
                mem.read_value(ps_regs_addr + 0x00, pgm_lo);
                mem.read_value(ps_regs_addr + 0x04, pgm_hi);
                mem.read_value(ps_regs_addr + 0x08, rsrc1);
                mem.read_value(ps_regs_addr + 0x0C, rsrc2);
                mem.read_value(ps_regs_addr + 0x10, z_fmt);
                mem.read_value(ps_regs_addr + 0x14, col_fmt);
                mem.read_value(ps_regs_addr + 0x18, in_ena);
                mem.read_value(ps_regs_addr + 0x1C, in_addr);
                mem.read_value(ps_regs_addr + 0x20, in_ctrl);
                mem.read_value(ps_regs_addr + 0x24, baryc);
                mem.read_value(ps_regs_addr + 0x28, db_ctrl);
                mem.read_value(ps_regs_addr + 0x2C, cb_mask);
            }
            const u64 isa_addr = ((static_cast<u64>(pgm_hi) << 32) | pgm_lo) << 8;
            PM4_LOG("[HLE %s] dcb=0x%llx dwords=%u ps_regs=0x%llx mod=0x%x -> PS ISA: 0x%llx (lo=0x%08x hi=0x%08x rsrc1=0x%08x rsrc2=0x%08x z_fmt=0x%x col_fmt=0x%x in_ena=0x%x in_addr=0x%x in_ctrl=0x%x baryc=0x%x db_ctrl=0x%x cb_mask=0x%x)\n",
                         func_name, (unsigned long long)dcb_addr, num_dwords, (unsigned long long)ps_regs_addr, modifier,
                         (unsigned long long)isa_addr, pgm_lo, pgm_hi, rsrc1, rsrc2, z_fmt, col_fmt, in_ena, in_addr, in_ctrl, baryc, db_ctrl, cb_mask);

            if (num_dwords >= 6 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                std::vector<u32> pkt = {
                    0xC0047600U, // Type 3, count=4 (5 payload dwords), opcode=0x76 (IT_SET_SH_REG)
                    0x0008U,     // mmSPI_SHADER_PGM_LO_PS
                    pgm_lo, pgm_hi, rsrc1, rsrc2,
                };
                // The stage's context registers (PS input layout, export
                // formats, DB_SHADER_CONTROL), as the real library writes them.
                const std::vector<u32> ctx_regs = {
                    0xC0026900U, 0x01C4U, z_fmt, col_fmt,    // SPI_SHADER_Z_FORMAT, SPI_SHADER_COL_FORMAT
                    0xC0026900U, 0x01B3U, in_ena, in_addr,   // SPI_PS_INPUT_ENA, SPI_PS_INPUT_ADDR
                    0xC0016900U, 0x01B6U, in_ctrl,           // SPI_PS_IN_CONTROL
                    0xC0016900U, 0x01B8U, baryc,             // SPI_BARYC_CNTL
                    0xC0016900U, 0x0203U, db_ctrl,           // DB_SHADER_CONTROL
                    0xC0016900U, 0x008FU, cb_mask,           // CB_SHADER_MASK
                };
                if (num_dwords >= pkt.size() + ctx_regs.size()) pkt.insert(pkt.end(), ctx_regs.begin(), ctx_regs.end());
                mem.write_bytes(dcb_addr, pkt.data(), pkt.size() * sizeof(u32));
                write_dcb_nop_padding(mem, dcb_addr + pkt.size() * sizeof(u32), num_dwords - static_cast<u32>(pkt.size()));
            } else if (num_dwords > 0 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                write_dcb_nop_padding(mem, dcb_addr, num_dwords);
            }
            ctx.rax = SCE_OK;
        };
    static constexpr struct { Nid nid; const char* name; } kPsShaderWriters[] = {
        {0xE6E14A7248896113ULL, "sceGnmSetPsShader350"},
        {0x98B54BECDEC15418ULL, "sceGnmUpdatePsShader350"},
        {0x6D055DE58CC26A5DULL, "sceGnmSetPsShader"},
        {0xE0C811C3F6D53505ULL, "sceGnmUpdatePsShader"},
    };
    for (const auto& w : kPsShaderWriters) {
        const char* name = w.name;
        register_symbol(w.nid, name, "libSceGnm",
            [ps_shader_handler, name](CpuContext& ctx, MemoryManager& mem) { ps_shader_handler(name, ctx, mem); });
    }

    // sceGnmSetCsShaderWithModifier -- writes IT_SET_SH_REG for COMPUTE registers + NOP padding
    // rdx points to Gnm::CsStageRegisters in guest memory:
    // +0x00: m_computePgmLo, +0x04: m_computePgmHi
    // +0x08: m_computePgmRsrc1, +0x0C: m_computePgmRsrc2
    // +0x10: m_computeNumThreadX, +0x14: m_computeNumThreadY, +0x18: m_computeNumThreadZ
    auto cs_shader_handler = [write_dcb_nop_padding](const char* func_name, CpuContext& ctx, MemoryManager& mem) {
        const GuestAddr dcb_addr = ctx.rdi;
        const u32 num_dwords = static_cast<u32>(ctx.rsi);
        const GuestAddr cs_regs_addr = ctx.rdx;
        const u32 modifier = static_cast<u32>(ctx.rcx);

        u32 pgm_lo = 0, pgm_hi = 0, rsrc1 = 0, rsrc2 = 0;
        u32 threads_x = 64, threads_y = 1, threads_z = 1;
        if (cs_regs_addr && mem.is_valid_address(cs_regs_addr, 28)) {
            mem.read_value(cs_regs_addr + 0x00, pgm_lo);
            mem.read_value(cs_regs_addr + 0x04, pgm_hi);
            mem.read_value(cs_regs_addr + 0x08, rsrc1);
            mem.read_value(cs_regs_addr + 0x0C, rsrc2);
            mem.read_value(cs_regs_addr + 0x10, threads_x);
            mem.read_value(cs_regs_addr + 0x14, threads_y);
            mem.read_value(cs_regs_addr + 0x18, threads_z);
        }
        const u64 isa_addr = ((static_cast<u64>(pgm_hi) << 32) | pgm_lo) << 8;
        PM4_LOG("[HLE %s] dcb=0x%llx dwords=%u cs_regs=0x%llx mod=0x%x -> CS ISA: 0x%llx (lo=0x%08x hi=0x%08x rsrc1=0x%08x rsrc2=0x%08x threads=%u,%u,%u)\n",
                     func_name, (unsigned long long)dcb_addr, num_dwords, (unsigned long long)cs_regs_addr, modifier,
                     (unsigned long long)isa_addr, pgm_lo, pgm_hi, rsrc1, rsrc2, threads_x, threads_y, threads_z);

        if (num_dwords >= 13 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
            std::vector<u32> pkt;
            // 1. COMPUTE_NUM_THREAD_X/Y/Z (0x0204, count=3)
            pkt.push_back(0xC0037600U); // IT_SET_SH_REG, count=3 (4 payload dwords)
            pkt.push_back(0x0204U);     // mmCOMPUTE_NUM_THREAD_X
            pkt.push_back(threads_x ? threads_x : 64);
            pkt.push_back(threads_y ? threads_y : 1);
            pkt.push_back(threads_z ? threads_z : 1);

            // 2. COMPUTE_PGM_LO/HI (0x020C, count=2)
            pkt.push_back(0xC0027600U); // IT_SET_SH_REG, count=2 (3 payload dwords)
            pkt.push_back(0x020CU);     // mmCOMPUTE_PGM_LO
            pkt.push_back(pgm_lo);
            pkt.push_back(pgm_hi);

            // 3. COMPUTE_PGM_RSRC1/RSRC2 (0x0212, count=2)
            pkt.push_back(0xC0027600U); // IT_SET_SH_REG, count=2 (3 payload dwords)
            pkt.push_back(0x0212U);     // mmCOMPUTE_PGM_RSRC1
            pkt.push_back(rsrc1);
            pkt.push_back(rsrc2);

            mem.write_bytes(dcb_addr, pkt.data(), pkt.size() * sizeof(u32));
            write_dcb_nop_padding(mem, dcb_addr + pkt.size() * sizeof(u32), num_dwords - static_cast<u32>(pkt.size()));
        } else if (num_dwords > 0 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
            write_dcb_nop_padding(mem, dcb_addr, num_dwords);
        }
        ctx.rax = SCE_OK;
    };

    register_symbol(0x2B1FE1FE759027C0ULL, "sceGnmSetCsShaderWithModifier", "libSceGnm",
        [cs_shader_handler](CpuContext& ctx, MemoryManager& mem) {
            cs_shader_handler("sceGnmSetCsShaderWithModifier", ctx, mem);
        });

    // sceGnmDispatchDirect -- writes IT_DISPATCH_DIRECT (0x15, 5 DWORDs) + NOP padding
    register_symbol(0xD01CCB1A58DCC01AULL, "sceGnmDispatchDirect", "libSceGnm",
        [write_dcb_nop_padding](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dcb_addr = ctx.rdi;
            const u32 num_dwords = static_cast<u32>(ctx.rsi);
            const u32 dim_x = static_cast<u32>(ctx.rdx);
            const u32 dim_y = static_cast<u32>(ctx.rcx);
            const u32 dim_z = static_cast<u32>(ctx.r8);
            const u32 initiator = static_cast<u32>(ctx.r9 ? ctx.r9 : 1);

            PM4_LOG("[HLE sceGnmDispatchDirect] dcb=0x%llx dwords=%u grid=(%u, %u, %u) init=0x%x\n",
                         (unsigned long long)dcb_addr, num_dwords, dim_x, dim_y, dim_z, initiator);

            if (num_dwords >= 5 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                const u32 pkt[] = {
                    0xC0031500U, // Type 3, count=3 (4 payload dwords), opcode=0x15 (IT_DISPATCH_DIRECT)
                    dim_x,
                    dim_y,
                    dim_z,
                    initiator
                };
                mem.write_bytes(dcb_addr, pkt, sizeof(pkt));
                write_dcb_nop_padding(mem, dcb_addr + sizeof(pkt), num_dwords - 5);
            } else if (num_dwords > 0 && mem.is_valid_address(dcb_addr, static_cast<GuestSize>(num_dwords) * sizeof(u32))) {
                write_dcb_nop_padding(mem, dcb_addr, num_dwords);
            }
            ctx.rax = SCE_OK;
        });

    // sceGnmSubmitDone -- NID: 0xCAF67BDEE414AAB9ULL, per ps4libdoc. This NID was
    // registered as sceVideoOutWaitVblank; ps4libdoc resolves it to the GNM
    // submit-completion wait (imported by eboot.bin, libSceJobManager.prx and
    // libSceS3DConversion.prx). Body unchanged: sleep instead of spin so the
    // guest yields the host thread at ~60 FPS VBlank cadence.
    // sceGnmSubmitDone only tells the driver that the application finished
    // submitting for this frame (TRC R4089); it does not wait for the GPU or
    // for VBlank. Every submit already ran to completion synchronously.
    register_symbol(0xCAF67BDEE414AAB9ULL, "sceGnmSubmitDone", "libSceGnm",
        [](CpuContext& ctx, MemoryManager& mem) {
            ensure_unity_screen_resolution(mem);
            ctx.rax = SCE_OK;
        });

    // sceGnmSubmitCommandBuffers(count, dcbGpuAddrs, dcbSizes, ccbGpuAddrs, ccbSizes)
    // NID: 0xCF0634615F754D32 (imported by eboot; was unregistered, so a
    // submit without flip silently dropped its DCBs and their EOP interrupts).
    register_symbol(0xCF0634615F754D32ULL, "sceGnmSubmitCommandBuffers", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager& mem) {
            execute_submitted_dcbs(mem, static_cast<u32>(ctx.rdi), ctx.rsi, ctx.rdx);
            g_gnm_completed_label.fetch_add(1);
            sync_gnm_label_slot(mem);
            ctx.rax = SCE_OK;
        });

    // sceGnmAreSubmitsAllowed() -> bool. NID: 0x6F4F0082D3E51CF8. Submits are
    // only suspended while the system has the GPU (never, here); the unset
    // stub's 0 meant "not allowed".
    register_symbol(0x6F4F0082D3E51CF8ULL, "sceGnmAreSubmitsAllowed", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 1; });

    // sceGnmGetEqEventType(const SceKernelEvent* ev) -> event type (ev->data).
    // NID: 0x528618D0358C0B45.
    register_symbol(0x528618D0358C0B45ULL, "sceGnmGetEqEventType", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager& mem) {
            i64 data = 0;
            if (ctx.rdi > 0x1000 && mem.is_valid_address(ctx.rdi + 0x10, sizeof(i64))) mem.read_value(ctx.rdi + 0x10, data);
            ctx.rax = static_cast<u64>(data);
        });

    // sceGnmDeleteEqEvent(eq, id). NID: 0x3D54FE7EEA12F605.
    register_symbol(0x3D54FE7EEA12F605ULL, "sceGnmDeleteEqEvent", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager&) {
            const u32 eq_id = static_cast<u32>(ctx.rdi);
            const i32 event_id = static_cast<i32>(ctx.rsi);
            std::lock_guard<std::mutex> lock(s_gnm_eq_mtx);
            std::erase_if(s_gnm_eq_subscriptions, [&](const GnmEqSubscription& sub) {
                return sub.eq_id == eq_id && sub.event_id == event_id;
            });
            ctx.rax = SCE_OK;
        });

    // sceKernelGetEventFilter / sceKernelGetEventId(const SceKernelEvent* ev).
    // NIDs 0xDB708F3C8D6DC816 / 0x989EDA8219A0BDF7 (both imported).
    register_symbol(0xDB708F3C8D6DC816ULL, "sceKernelGetEventFilter", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            i16 filter = 0;
            if (ctx.rdi > 0x1000 && mem.is_valid_address(ctx.rdi + 0x08, sizeof(i16))) mem.read_value(ctx.rdi + 0x08, filter);
            ctx.rax = static_cast<u64>(static_cast<i64>(filter));
        });
    register_symbol(0x989EDA8219A0BDF7ULL, "sceKernelGetEventId", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            u64 ident = 0;
            if (ctx.rdi > 0x1000 && mem.is_valid_address(ctx.rdi, sizeof(u64))) mem.read_value(ctx.rdi, ident);
            ctx.rax = ident;
        });

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

    // The counter above ticks in microseconds. Its frequency was a sync stub
    // returning 0: Unity's Time (counter / frequency) never advanced
    // sensibly, so the splash screen never finished and the first scene
    // (level0) was never loaded.
    register_symbol(0x04DA30C76979F3C1ULL, "sceKernelGetProcessTimeCounterFrequency", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 1000000; });

    // sceKernelGetProcessTime(): microseconds since the process started.
    static const auto s_process_start = std::chrono::steady_clock::now();
    register_symbol(0xE09DAC5099AE1D94ULL, "sceKernelGetProcessTime", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - s_process_start).count());
        });

    // sceKernelReadTsc / sceKernelGetTscFrequency: a nanosecond time-stamp counter.
    register_symbol(0xFF62115023BFFCF3ULL, "sceKernelReadTsc", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        });
    register_symbol(0xD63DD2DE7FED4D6EULL, "sceKernelGetTscFrequency", "libkernel",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 1000000000ULL; });

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
        [perform_flip](CpuContext& ctx, MemoryManager& mem) {
            ensure_unity_screen_resolution(mem);
            const u32 count = static_cast<u32>(ctx.rdi);
            const GuestAddr dcb_addrs_ptr = ctx.rsi;
            const GuestAddr dcb_sizes_ptr = ctx.rdx;

            u32 buffer_index = 0;
            if (mem.is_valid_address(ctx.rsp + 8, sizeof(u32))) {
                mem.read_value(ctx.rsp + 8, buffer_index);
            }
            i64 flip_arg = 0;
            if (mem.is_valid_address(ctx.rsp + 24, sizeof(i64))) {
                mem.read_value(ctx.rsp + 24, flip_arg);
            }

            PM4_LOG("[Gnm] SubmitAndFlip(count=%u, dcbAddrs=0x%llx, dcbSizes=0x%llx, bufIdx=%u, flipArg=0x%llx)\n",
                         count, (unsigned long long)dcb_addrs_ptr, (unsigned long long)dcb_sizes_ptr,
                         buffer_index, (unsigned long long)flip_arg);

            // The command processor runs the DCBs to completion here (labels
            // written, EOP interrupts raised), then the flip is performed.
            execute_submitted_dcbs(mem, count, dcb_addrs_ptr, dcb_sizes_ptr);
            g_gnm_completed_label.fetch_add(1);
            sync_gnm_label_slot(mem);

            perform_flip(mem, buffer_index, flip_arg);
            ctx.rax = SCE_OK;
        });

    // sceGnmEndWorkload — closes a GPU workload; advance the fence only.
    // (Was mislabeled SubmitAndFlip: the old body presented + signaled flip here,
    // a full frame early. Real presentation lives in 0xC5BC... below.)
    // NID: 0x15ADF1EF938E2D10ULL
    register_symbol(0x15ADF1EF938E2D10ULL, "sceGnmEndWorkload", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager& mem) {
            // No GPU work is retired here: EOP events come only from EOP
            // packets that request an interrupt (see execute_submitted_dcbs).
            g_gnm_completed_label.fetch_add(1);
            sync_gnm_label_slot(mem);
            ctx.rax = SCE_OK;
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

    // sceGnmAddEqEvent — register equeue event for EOP / fence completion.
    // NID: 0x6F4C729659D563F2ULL
    register_symbol(0x6F4C729659D563F2ULL, "sceGnmAddEqEvent", "libSceGnmDriver",
        [](CpuContext& ctx, MemoryManager&) {
            const u32 eq_id = static_cast<u32>(ctx.rdi);
            const i32 event_id = static_cast<i32>(ctx.rsi);
            const u64 udata = ctx.rdx;
            std::lock_guard<std::mutex> lock(s_gnm_eq_mtx);
            s_gnm_eq_subscriptions.push_back({eq_id, event_id, udata});
            PM4_LOG("[Gnm] AddEqEvent: eq=%u, id=%d, udata=0x%llx\n",
                         eq_id, event_id, (unsigned long long)udata);
            ctx.rax = SCE_OK;
        });

    // =========================================================================
    // POSIX mutexes / condition variables (libScePosix), as used by mono-ps4.
    // Registered last: they replace earlier no-op stubs (lock/unlock returned
    // 0 without locking, so Mono's threads raced inside the runtime, and
    // pthread_cond_* were missing, so waiters were never woken).
    //
    // Objects are keyed by the guest pthread_mutex_t / pthread_cond_t address,
    // which also covers static initialisers. Mutexes are recursive for the
    // owning thread (a superset of the default type). Blocking waits run in
    // 20 ms slices with every lock released between them, so a blocked guest
    // thread still takes pending exceptions (Boehm GC's suspend signal).
    // Return values are POSIX error numbers (FreeBSD: EPERM 1, EBUSY 16,
    // EINVAL 22, ETIMEDOUT 60).
    // =========================================================================
    auto pmutex = [](GuestAddr a) {
        std::lock_guard<std::mutex> l(s_ptab_mtx);
        auto& p = s_pmutexes[a];
        if (!p) p = std::make_shared<PMutex>();
        return p;
    };
    auto pcond = [](GuestAddr a) {
        std::lock_guard<std::mutex> l(s_ptab_mtx);
        auto& p = s_pconds[a];
        if (!p) p = std::make_shared<PCond>();
        return p;
    };
    // Acquire `count` levels of `pm` for the calling thread.
    auto pm_acquire = [](PMutex& pm, MemoryManager& mem, int count) {
        const auto self = std::this_thread::get_id();
        std::unique_lock<std::mutex> l(pm.m);
        while (pm.count > 0 && pm.owner != self) {
            if (g_syscalls && g_syscalls->has_exited()) break;
            {
                SignalWakeScope wake(pm.cv);
                if (!guest_signal_pending()) pm.cv.wait_for(l, std::chrono::milliseconds(20));
            }
            if (guest_signal_pending()) {
                l.unlock();
                deliver_guest_exceptions(mem);
                l.lock();
            }
        }
        pm.owner = self;
        pm.count += count;
    };
    // Release all levels held by the calling thread; returns how many.
    auto pm_release_all = [](PMutex& pm) {
        std::lock_guard<std::mutex> l(pm.m);
        if (pm.count == 0 || pm.owner != std::this_thread::get_id()) return 0;
        const int n = pm.count;
        pm.count = 0;
        pm.owner = {};
        pm.cv.notify_all();
        return n;
    };
    auto ok = [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0; };
    register_symbol(0xB6D1CD7D4FAA0C15ULL, "pthread_mutex_init", "libScePosix",
        [pmutex](CpuContext& ctx, MemoryManager&) { pmutex(ctx.rdi); ctx.rax = 0; });
    register_symbol(0x96D09F686AF62461ULL, "pthread_mutex_destroy", "libScePosix",
        [](CpuContext& ctx, MemoryManager&) {
            std::lock_guard<std::mutex> l(s_ptab_mtx);
            s_pmutexes.erase(ctx.rdi);
            ctx.rax = 0;
        });
    register_symbol(0xEC7D224CE7224CBAULL, "pthread_mutex_lock", "libScePosix",
        [pmutex, pm_acquire](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rdi == 0) { ctx.rax = 22; return; }
            pm_acquire(*pmutex(ctx.rdi), mem, 1);
            ctx.rax = 0;
        });
    register_symbol(0x2BF8D785BB76827EULL, "pthread_mutex_trylock", "libScePosix",
        [pmutex](CpuContext& ctx, MemoryManager&) {
            if (ctx.rdi == 0) { ctx.rax = 22; return; }
            auto pm = pmutex(ctx.rdi);
            std::lock_guard<std::mutex> l(pm->m);
            if (pm->count > 0 && pm->owner != std::this_thread::get_id()) { ctx.rax = 16; return; }
            pm->owner = std::this_thread::get_id();
            ++pm->count;
            ctx.rax = 0;
        });
    register_symbol(0xD99F8FA58E826898ULL, "pthread_mutex_unlock", "libScePosix",
        [pmutex](CpuContext& ctx, MemoryManager&) {
            if (ctx.rdi == 0) { ctx.rax = 22; return; }
            auto pm = pmutex(ctx.rdi);
            std::lock_guard<std::mutex> l(pm->m);
            if (pm->count == 0 || pm->owner != std::this_thread::get_id()) { ctx.rax = 1; return; }
            if (--pm->count == 0) {
                pm->owner = {};
                pm->cv.notify_all();
            }
            ctx.rax = 0;
        });
    register_symbol(0x7501D612C26DA04EULL, "pthread_mutexattr_init", "libScePosix", ok);
    register_symbol(0x9839A030E19552A8ULL, "pthread_mutexattr_settype", "libScePosix", ok);
    register_symbol(0xE6DC4A7DC3140289ULL, "pthread_mutexattr_setprotocol", "libScePosix", ok);
    register_symbol(0x1C5EE52B8EB1CE36ULL, "pthread_mutexattr_destroy", "libScePosix", ok);
    register_symbol(0x98AA13C74DC74560ULL, "pthread_condattr_init", "libScePosix", ok);
    register_symbol(0x74972E4159FAFC8CULL, "pthread_condattr_destroy", "libScePosix", ok);
    register_symbol(0xD13C959383122EDDULL, "pthread_cond_init", "libScePosix",
        [pcond](CpuContext& ctx, MemoryManager&) { pcond(ctx.rdi); ctx.rax = 0; });
    register_symbol(0x4575EA8B80AD17CCULL, "pthread_cond_destroy", "libScePosix",
        [](CpuContext& ctx, MemoryManager&) {
            std::lock_guard<std::mutex> l(s_ptab_mtx);
            s_pconds.erase(ctx.rdi);
            ctx.rax = 0;
        });
    register_symbol(0xD8C3B2FAB51FBA14ULL, "pthread_cond_signal", "libScePosix",
        [pcond](CpuContext& ctx, MemoryManager&) {
            auto pc = pcond(ctx.rdi);
            { std::lock_guard<std::mutex> l(pc->m); ++pc->seq; }
            pc->cv.notify_one();
            ctx.rax = 0;
        });
    register_symbol(0x9A4C767D584D32C8ULL, "pthread_cond_broadcast", "libScePosix",
        [pcond](CpuContext& ctx, MemoryManager&) {
            auto pc = pcond(ctx.rdi);
            { std::lock_guard<std::mutex> l(pc->m); ++pc->seq; }
            pc->cv.notify_all();
            ctx.rax = 0;
        });
    // pthread_cond_wait(cond, mutex) / pthread_cond_timedwait(cond, mutex, abstime)
    auto cond_wait = [pmutex, pcond, pm_acquire, pm_release_all](CpuContext& ctx, MemoryManager& mem, bool timed) {
        if (ctx.rdi == 0 || ctx.rsi == 0) { ctx.rax = 22; return; }
        auto pc = pcond(ctx.rdi);
        auto pm = pmutex(ctx.rsi);
        std::optional<std::chrono::system_clock::time_point> deadline;
        if (timed) {
            i64 sec = 0, nsec = 0;
            if (ctx.rdx == 0 || !mem.read_value(ctx.rdx, sec) || !mem.read_value(ctx.rdx + 8, nsec)) {
                ctx.rax = 22;
                return;
            }
            deadline = std::chrono::system_clock::time_point(std::chrono::seconds(sec)) +
                       std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::nanoseconds(nsec));
        }
        std::unique_lock<std::mutex> cl(pc->m);
        const u64 gen = pc->seq;
        const int held = pm_release_all(*pm); // atomically w.r.t. signalers: pc->m is held
        if (held == 0) { ctx.rax = 1; return; } // EPERM: caller does not own the mutex
        bool timed_out = false;
        while (pc->seq == gen) {
            if (g_syscalls && g_syscalls->has_exited()) break;
            if (deadline && std::chrono::system_clock::now() >= *deadline) { timed_out = true; break; }
            {
                SignalWakeScope wake(pc->cv);
                if (!guest_signal_pending()) pc->cv.wait_for(cl, std::chrono::milliseconds(20));
            }
            if (pc->seq != gen) break;
            if (guest_signal_pending()) {
                cl.unlock();
                deliver_guest_exceptions(mem);
                cl.lock();
            }
        }
        cl.unlock();
        pm_acquire(*pm, mem, held);
        ctx.rax = timed_out ? 60 : 0;
    };
    register_symbol(0x3A9F130466392878ULL, "pthread_cond_wait", "libScePosix",
        [cond_wait](CpuContext& ctx, MemoryManager& mem) { cond_wait(ctx, mem, false); });
    register_symbol(0xDBB6C08222663A1DULL, "pthread_cond_timedwait", "libScePosix",
        [cond_wait](CpuContext& ctx, MemoryManager& mem) { cond_wait(ctx, mem, true); });

    // sceAudioOutInit(): the port manager needs no global setup.
    register_symbol(0x25F10F5D5C6116A0ULL, "sceAudioOutInit", "libaudioout",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = SCE_OK; });

    // ---- libSceUserService / libSceSystemService calls Unity polls ----
    register_symbol(0x09D5A9D281D61ABDULL, "sceUserServiceGetInitialUser", "libSceUserService",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rdi > 0x1000 && mem.is_valid_address(ctx.rdi, 4)) mem.write_value(ctx.rdi, static_cast<i32>(0x10000001));
            ctx.rax = SCE_OK;
        });
    // SceSystemServiceStatus: eventNum, then isSystemUiOverlaid,
    // isInBackgroundExecution, isCpuMode7CpuNormal, isGameLiveStreamingOnAir,
    // isOutOfVrPlayArea. Foreground, nothing pending.
    register_symbol(0xACFA3AB55F03F5B3ULL, "sceSystemServiceGetStatus", "libSceSystemService",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rdi > 0x1000 && mem.is_valid_address(ctx.rdi, 12)) {
                mem.zero_memory(ctx.rdi, 12);
                mem.write_value(ctx.rdi + 6, static_cast<u8>(1)); // isCpuMode7CpuNormal
            }
            ctx.rax = SCE_OK;
        });
    register_symbol(0xEB9E8B3104AB83A5ULL, "sceSystemServiceReceiveEvent", "libSceSystemService",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 0x80A10003ULL; }); // SCE_SYSTEM_SERVICE_ERROR_NO_EVENT
    register_symbol(0x568E55F0A0300A69ULL, "sceSystemServiceHideSplashScreen", "libSceSystemService",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = SCE_OK; });
    // sceSystemServiceParamGetInt(paramId, int* value)
    register_symbol(0x7D9A38F2E9FB2CAEULL, "sceSystemServiceParamGetInt", "libSceSystemService",
        [](CpuContext& ctx, MemoryManager& mem) {
            i32 v = 0;
            switch (static_cast<i32>(ctx.rdi)) {
                case 1: v = 1; break;    // LANG: English (US)
                case 2: v = 1; break;    // DATE_FORMAT: MM/DD/YYYY
                case 3: v = 0; break;    // TIME_FORMAT: 12-hour
                case 4: v = 0; break;    // TIME_ZONE (minutes)
                case 5: v = 0; break;    // SUMMERTIME
                case 1000: v = 1; break; // ENTER_BUTTON_ASSIGN: cross
                default: break;
            }
            if (ctx.rsi > 0x1000 && mem.is_valid_address(ctx.rsi, 4)) mem.write_value(ctx.rsi, v);
            ctx.rax = SCE_OK;
        });
    // sceSystemServiceGetDisplaySafeAreaInfo(info*): ratio 1.0 (no overscan).
    register_symbol(0xD67DFBAB506F7396ULL, "sceSystemServiceGetDisplaySafeAreaInfo", "libSceSystemService",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rdi > 0x1000 && mem.is_valid_address(ctx.rdi, 4)) mem.write_value(ctx.rdi, 1.0f);
            ctx.rax = SCE_OK;
        });
    register_symbol(0xFF81827D874D175BULL, "sceImeUpdate", "libSceIme",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = SCE_OK; });
    // sceAudioOutGetPortState(handle, SceAudioOutPortState*): primary output,
    // stereo, full volume.
    register_symbol(0x1AB43DB3822B35A4ULL, "sceAudioOutGetPortState", "libSceAudioOut",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rsi > 0x1000 && mem.is_valid_address(ctx.rsi, 32)) {
                mem.zero_memory(ctx.rsi, 32);
                mem.write_value(ctx.rsi + 0, static_cast<u16>(1));  // output: primary
                mem.write_value(ctx.rsi + 2, static_cast<u8>(2));   // channels
                mem.write_value(ctx.rsi + 4, static_cast<i16>(127)); // volume
            }
            ctx.rax = SCE_OK;
        });

    // ---- libScePad: the functions Unity actually calls ----
    // scePadRead(handle, ScePadData* data, int num) -> samples written.
    register_symbol(0xAB570735F1B270B2ULL, "scePadRead", "libScePad",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 num = static_cast<i32>(ctx.rdx);
            if (num < 1 || ctx.rsi <= 0x1000 || !mem.is_valid_address(ctx.rsi, kScePadDataSize)) {
                ctx.rax = num < 1 ? 0 : 0x80920001ULL; // SCE_PAD_ERROR_INVALID_ARG
                return;
            }
            write_pad_data(mem, ctx.rsi, read_pad_state());
            ctx.rax = 1;
        });
    // scePadGetControllerInformation(handle, ScePadControllerInformation*):
    // a connected standard DualShock 4 with its touch pad geometry.
    register_symbol(0x8233FDFCA433A149ULL, "scePadGetControllerInformation", "libScePad",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rsi > 0x1000 && mem.is_valid_address(ctx.rsi, 28)) {
                mem.zero_memory(ctx.rsi, 28);
                mem.write_value(ctx.rsi + 0, 44.86f);              // touchPad.pixelDensity
                mem.write_value(ctx.rsi + 4, static_cast<u16>(1920)); // touchPad.resolution.x
                mem.write_value(ctx.rsi + 6, static_cast<u16>(943));  // touchPad.resolution.y
                mem.write_value(ctx.rsi + 8, static_cast<u8>(13));    // stick dead zones
                mem.write_value(ctx.rsi + 9, static_cast<u8>(13));
                mem.write_value(ctx.rsi + 11, static_cast<u8>(1));    // connectedCount
                mem.write_value(ctx.rsi + 12, static_cast<u8>(1));    // connected
            }
            ctx.rax = SCE_OK;
        });
    static constexpr struct { Nid nid; const char* name; } kPadNoops[] = {
        {0xEA77207B9FA5E50BULL, "scePadClose"},
        {0x0EC703D62F475F5CULL, "scePadResetLightBar"},
        {0xAC866747A792A6F9ULL, "scePadResetOrientation"},
        {0xAF8E260317521BE5ULL, "scePadSetAngularVelocityDeadbandState"},
        {0x72556F2F86439EDCULL, "scePadSetMotionSensorState"},
        {0xBC32CCA092DD7BC2ULL, "scePadSetTiltCorrectionState"},
        {0xC8556739D1B1BD96ULL, "scePadSetVibration"},
    };
    for (const auto& p : kPadNoops)
        register_symbol(p.nid, p.name, "libScePad", [](CpuContext& ctx, MemoryManager&) { ctx.rax = SCE_OK; });

    // ---- Optional input devices: none attached ----
    // Returning 0 from the open calls handed Unity "handle 0" for a Move
    // controller, which it then polled every frame.
    static constexpr struct { Nid nid; const char* name; const char* module; u64 err; } kAbsentDevices[] = {
        {0x1F30BAD0C7E32715ULL, "sceMoveOpen", "libSceMove", 0x80EE0005ULL},          // SCE_MOVE_ERROR_NO_CONTROLLER_CONNECTED-style
        {0x1965D3CB1B3841B1ULL, "sceMoveGetDeviceInfo", "libSceMove", 0x80EE0003ULL},  // invalid handle
        {0x7F66DCA4AEA425F8ULL, "sceMoveReadStateRecent", "libSceMove", 0x80EE0003ULL},
        {0x79A1578DF26FDF1BULL, "sceImeKeyboardOpen", "libSceIme", 0x80BC0005ULL},     // no keyboard
    };
    for (const auto& d : kAbsentDevices) {
        const u64 err = d.err;
        register_symbol(d.nid, d.name, d.module, [err](CpuContext& ctx, MemoryManager&) { ctx.rax = err; });
    }

    // ---- libSceMouse: active mouse input ----
    // SceMouseData (0x40 bytes approx):
    // timestamp (u64 at 0x00), buttons (u32 at 0x08), xAxis (i32 at 0x0C), yAxis (i32 at 0x10), wheel (i32 at 0x14), connected (u8 at 0x18)
    constexpr GuestSize kSceMouseDataSize = 0x40;

    // sceMouseOpen(int userId, int type, int index, const void* param)
    register_symbol(0x45AAB16487FA0EF1ULL, "sceMouseOpen", "libSceMouse",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 1; // Handle 1
        });

    // sceMouseRead(int handle, SceMouseData* data, int num) -> samples read
    register_symbol(0xC7CAA75EA87FB623ULL, "sceMouseRead", "libSceMouse",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 num = static_cast<i32>(ctx.rdx);
            const GuestAddr p = ctx.rsi;
            if (num < 1 || p <= 0x1000 || !mem.is_valid_address(p, kSceMouseDataSize)) {
                ctx.rax = num < 1 ? 0 : 0x80DF0001ULL; // SCE_MOUSE_ERROR_INVALID_ARG
                return;
            }
            mem.zero_memory(p, kSceMouseDataSize);
            MouseState ms;
            {
                std::lock_guard<std::mutex> lock(g_input_mutex);
                ms = g_active_mouse_state;
            }
            const u64 ts = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            mem.write_value(p + 0x00, ts);
            mem.write_value(p + 0x08, ms.buttons);
            mem.write_value(p + 0x0C, ms.delta_x);
            mem.write_value(p + 0x10, ms.delta_y);
            mem.write_value(p + 0x14, ms.wheel);
            mem.write_value(p + 0x18, static_cast<u8>(ms.connected ? 1 : 0));
            ctx.rax = 1;
        });
    // sceNpGetAccountIdA(userId, SceNpAccountId* out): a fixed offline id.
    register_symbol(0xADB9276948E9A96AULL, "sceNpGetAccountIdA", "libSceNpManager",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rsi > 0x1000 && mem.is_valid_address(ctx.rsi, 8)) mem.write_value(ctx.rsi, static_cast<u64>(0x1000000000000001ULL));
            ctx.rax = SCE_OK;
        });

    // =====================================================================
    // Offline console: PSN, network, HTTP, RTC, common dialogs, and the libc /
    // C++ ABI gaps left unresolved by UnityNpToolkit.prx and the other plugins.
    //
    // The console is modelled as signed out with no network connection, so the
    // game takes its offline path:
    //   - Init / Term / Register / Unregister / Set / Delete / Abort / Check
    //     calls touch only local state and return SCE_OK.
    //   - Create calls return a positive handle id, so a caller that tests for
    //     "< 0" or "== 0" takes its success branch.
    //   - Anything that would reach a server (requests, lookups, scores, room
    //     and web API traffic) returns the error a signed-out console returns.
    //     Those calls never return success with an output buffer left unset.
    //   - Common dialogs never draw a UI; see the tracker below.
    // =====================================================================
    struct HleStub { Nid nid; const char* name; };
    static constexpr u64 kNpErrSignedOut = 0x80550006ULL;         // SCE_NP_ERROR_SIGNED_OUT
    static constexpr u64 kHttpErrNetwork = 0x80431063ULL;         // SCE_HTTP_ERROR_NETWORK
    static constexpr u64 kNetCtlErrNotConnected = 0x80412108ULL;  // SCE_NET_CTL_ERROR_NOT_CONNECTED
    // Shared counter for the Create* handle ids (positive, never reused here).
    static std::atomic<u64> s_offline_handle_seq{1};
    // Each entry's handler sets rax to `rax`.
    auto add_stubs = [this](const char* module, std::initializer_list<HleStub> stubs, u64 rax) {
        for (const HleStub& s : stubs)
            register_symbol(s.nid, s.name, module, [rax](CpuContext& ctx, MemoryManager&) { ctx.rax = rax; });
    };
    // Each entry's handler returns a fresh positive handle id.
    auto add_handles = [this](const char* module, std::initializer_list<HleStub> stubs) {
        for (const HleStub& s : stubs)
            register_symbol(s.nid, s.name, module,
                [](CpuContext& ctx, MemoryManager&) { ctx.rax = s_offline_handle_seq.fetch_add(1); });
    };

    // --- PSN / network: offline console ---
    add_stubs("libSceNp", {
        {0x11CEB7CB9F65F6DCULL, "sceNpSetNpTitleId"},
        {0x036090DE4812A294ULL, "sceNpSetContentRestriction"},
        {0x55F45298F9A3F10FULL, "sceNpRegisterStateCallback"},
        {0x9A38D35E1F8D1D66ULL, "sceNpUnregisterStateCallback"},
        {0xDD997C05E3D387D6ULL, "sceNpCheckCallback"},
        {0x4BB4139FBD8FAC3CULL, "sceNpDeleteRequest"},
    }, SCE_OK);
    add_handles("libSceNp", {
        {0x1A92D00CD28809A7ULL, "sceNpCreateRequest"},
    });
    add_stubs("libSceNp", {
        {0xDABB059A519695E4ULL, "sceNpCheckNpAvailability"},
        {0xA7FA3BE029E83736ULL, "sceNpGetNpId"},
        {0x5C39DC5D02095129ULL, "sceNpGetOnlineId"},
    }, kNpErrSignedOut);
    // sceNpCmpNpId(const SceNpId* a, const SceNpId* b): 0 when the online ids
    // match. Only the 16-byte handle is compared; the struct layout is assumed.
    register_symbol(0x8BC5265D34AAECDEULL, "sceNpCmpNpId", "libSceNp",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (!mem.is_valid_address(ctx.rdi, 16) || !mem.is_valid_address(ctx.rsi, 16)) {
                ctx.rax = 1;
                return;
            }
            ctx.rax = std::strncmp(mem.to_host_ptr<char>(ctx.rdi), mem.to_host_ptr<char>(ctx.rsi), 16) == 0 ? 0 : 1;
        });

    // libSceNpMatching2: rooms and signaling need a live session.
    add_stubs("libSceNpMatching2", {
        {0xD74B777B9F893E75ULL, "sceNpMatching2Initialize"},
        {0x32AA77949FAC8F2EULL, "sceNpMatching2Terminate"},
        {0x99E12321D6E3000DULL, "sceNpMatching2SetUserInfo"},
        {0xFDFE8CE1C68D7BC9ULL, "sceNpMatching2ContextStop"},
        {0x373FD913BBABDF62ULL, "sceNpMatching2DestroyContext"},
        {0x7D041F3FCEC8EE1BULL, "sceNpMatching2RegisterContextCallback"},
        {0xA7ED849F199A00C3ULL, "sceNpMatching2RegisterRoomEventCallback"},
        {0xB81112CF3E02430BULL, "sceNpMatching2RegisterRoomMessageCallback"},
        {0xD1431E5911A764A0ULL, "sceNpMatching2RegisterSignalingCallback"},
        {0xFBC7BBC172E68DDBULL, "sceNpMatching2SetDefaultRequestOptParam"},
    }, SCE_OK);
    add_handles("libSceNpMatching2", {
        {0x61F9A95BBD7DACCAULL, "sceNpMatching2CreateContext"},
    });
    add_stubs("libSceNpMatching2", {
        {0xEEF8CD43A675A29DULL, "sceNpMatching2ContextStart"},
        {0x2E108F72D2080B14ULL, "sceNpMatching2GetServerId"},
        {0x043EA47F1E38D83AULL, "sceNpMatching2LeaveRoom"},
        {0x01455F53A6F28377ULL, "sceNpMatching2KickoutRoomMember"},
        {0x230DA1D09AEB6F95ULL, "sceNpMatching2SendRoomMessage"},
        {0xA290E997BE298B61ULL, "sceNpMatching2SendRoomChatMessage"},
        {0xAC934F26A0C2A622ULL, "sceNpMatching2GetWorldInfoList"},
        {0xCC25999975CDEB4DULL, "sceNpMatching2CreateJoinRoom"},
        {0x56A657ECF3A0D8C9ULL, "sceNpMatching2SearchRoom"},
        {0xA9E17FAB9283B407ULL, "sceNpMatching2GetUserInfoList"},
        {0x09220C0EC563B3F8ULL, "sceNpMatching2JoinRoom"},
        {0xDBABD6ACF01625F3ULL, "sceNpMatching2GetRoomDataExternalList"},
        {0x26B6B189F9A87ADEULL, "sceNpMatching2GetRoomDataInternal"},
        {0xE6586F3AA85E1410ULL, "sceNpMatching2GetRoomMemberDataInternal"},
        {0xABB18AF7CFE76121ULL, "sceNpMatching2SetRoomDataExternal"},
        {0x4BD0FC252608AE31ULL, "sceNpMatching2SetRoomDataInternal"},
        {0x1E8A93AE44BD7394ULL, "sceNpMatching2SetRoomMemberDataInternal"},
        {0xB7056EA5E69862B9ULL, "sceNpMatching2SignalingGetConnectionInfo"},
    }, kNpErrSignedOut);

    // libSceNpWebApi: requests fail; the handles and filters are local.
    add_stubs("libSceNpWebApi", {
        {0x1B70272CD7510631ULL, "sceNpWebApiInitialize"},
        {0x6ACCF74ED22A185FULL, "sceNpWebApiTerminate"},
        {0xE4C9FB4D8C29977DULL, "sceNpWebApiDeleteHandle"},
        {0xCC4F91E91731DD6DULL, "sceNpWebApiDeletePushEventFilter"},
        {0x3DF43E7FAC2CEFAEULL, "sceNpWebApiDeleteServicePushEventFilter"},
        {0x9E842095EBBE28B1ULL, "sceNpWebApiDeleteRequest"},
        {0x5D48DDB124D36775ULL, "sceNpWebApiDeleteContext"},
        {0x2738584CFD9F1B5FULL, "sceNpWebApiAbortRequest"},
        {0x3DF4930C280D3207ULL, "sceNpWebApiRegisterPushEventCallback"},
        {0x909409134B8A9B9CULL, "sceNpWebApiRegisterServicePushEventCallback"},
        {0xD9E76B92BD1CFF08ULL, "sceNpWebApiUnregisterServicePushEventCallback"},
        {0xA8AE28DBAE7A5B8CULL, "sceNpWebApiUnregisterPushEventCallback"},
    }, SCE_OK);
    add_handles("libSceNpWebApi", {
        {0xEFD33F26ABEF1A8DULL, "sceNpWebApiCreateHandle"},
        {0xCB94DAE490B34076ULL, "sceNpWebApiCreatePushEventFilter"},
        {0xB08171EF7E3EC72BULL, "sceNpWebApiCreateServicePushEventFilter"},
        {0xADD82CE59D4CC85CULL, "sceNpWebApiCreateRequest"},
        {0xC7563BCA261293B7ULL, "sceNpWebApiCreateContext"},
    });
    add_stubs("libSceNpWebApi", {
        {0x9156CBE212F72BBCULL, "sceNpWebApiSendRequest"},
        {0x936D74A0A80FF346ULL, "sceNpWebApiGetHttpStatusCode"},
        {0x090B4F45217A0ECFULL, "sceNpWebApiReadData"},
        {0xEF8DD9CC4073955FULL, "sceNpWebApiGetHttpResponseHeaderValueLength"},
        {0x5702792F41E2820DULL, "sceNpWebApiGetHttpResponseHeaderValue"},
    }, kNpErrSignedOut);

    // libSceNpScore / libSceNpTus / libSceNpTss: server-backed data.
    add_stubs("libSceNpScore", {
        {0x1B4A44F91342C1F9ULL, "sceNpScoreDeleteNpTitleCtx"},
        {0x74AF3F4A061FEABEULL, "sceNpScoreDeleteRequest"},
        {0xD62EE498A6D7EA19ULL, "sceNpScoreAbortRequest"},
    }, SCE_OK);
    add_handles("libSceNpScore", {
        {0x2A7340D53120B412ULL, "sceNpScoreCreateNpTitleCtx"},
        {0x816F2ACA362B51B9ULL, "sceNpScoreCreateRequest"},
    });
    add_stubs("libSceNpScore", {
        {0xCD3D1706D82D3922ULL, "sceNpScoreRecordScore"},
        {0x00D26CB0FCF7998DULL, "sceNpScoreRecordScoreAsync"},
        {0xF66644828884ABA6ULL, "sceNpScoreGetRankingByNpId"},
        {0x45DDBB76A505655FULL, "sceNpScoreGetRankingByNpIdAsync"},
        {0x2811F10E3CA4FE30ULL, "sceNpScoreGetRankingByRange"},
        {0xAD2866A971F0A101ULL, "sceNpScoreGetRankingByRangeAsync"},
        {0xF24B88CD4C3ABAD4ULL, "sceNpScoreGetFriendsRanking"},
        {0xED2B8C52537B43A2ULL, "sceNpScoreGetFriendsRankingAsync"},
        {0x2E854C56B8A354E9ULL, "sceNpScoreGetBoardInfo"},
        {0x9B50DF351B2D9124ULL, "sceNpScorePollAsync"},
    }, kNpErrSignedOut);
    add_stubs("libSceNpTus", {
        {0x09C207E347584BCFULL, "sceNpTusDeleteRequest"},
        {0x1F7BAAEF1D2C64E2ULL, "sceNpTusDeleteNpTitleCtx"},
    }, SCE_OK);
    add_handles("libSceNpTus", {
        {0x04890C9947CD2963ULL, "sceNpTusCreateNpTitleCtx"},
        {0xDDB876681BEF9AF3ULL, "sceNpTusCreateRequest"},
    });
    add_stubs("libSceNpTus", {
        {0xD42CF4853245CA1EULL, "sceNpTusSetMultiSlotVariableVUser"},
        {0x73A698A1AE3B6202ULL, "sceNpTusSetMultiSlotVariable"},
        {0xB85C556091247267ULL, "sceNpTusGetMultiSlotVariableVUser"},
        {0x17E79095F7246BDFULL, "sceNpTusGetMultiSlotVariable"},
        {0xF918734AEB97C31AULL, "sceNpTusSetDataVUser"},
        {0xE0DAEE7E43429221ULL, "sceNpTusSetData"},
        {0x1901C292C4BB68BBULL, "sceNpTusGetDataVUser"},
        {0x5CECECCCEE0E3565ULL, "sceNpTusGetData"},
    }, kNpErrSignedOut);
    add_stubs("libSceNpTss", {
        {0xFD2511F94A0B4BA7ULL, "sceNpTssGetData"},
    }, kNpErrSignedOut);

    // libSceNpLookup / libSceNpAuth / libSceNpWordFilter / libSceNpSnsFacebook
    add_stubs("libSceNpLookup", {
        {0x9ADA832BDCE4A081ULL, "sceNpLookupDeleteTitleCtx"},
        {0xC0B6B1721BC41279ULL, "sceNpLookupDeleteRequest"},
    }, SCE_OK);
    add_handles("libSceNpLookup", {
        {0xF39DF743E2D4EC44ULL, "sceNpLookupCreateTitleCtx"},
        {0x890AFD5313C7505BULL, "sceNpLookupCreateRequest"},
    });
    add_stubs("libSceNpLookup", {
        {0x4FAB6733552D8B88ULL, "sceNpLookupNpId"},
    }, kNpErrSignedOut);
    add_stubs("libSceNpAuth", {
        {0x1FCC06F4193F9CF7ULL, "sceNpAuthDeleteRequest"},
    }, SCE_OK);
    add_handles("libSceNpAuth", {
        {0xE9BC05928B184508ULL, "sceNpAuthCreateRequest"},
    });
    add_stubs("libSceNpAuth", {
        {0x2B11A43AB4094EA6ULL, "sceNpAuthGetAuthorizationCode"},
    }, kNpErrSignedOut);
    add_stubs("libSceNpWordFilter", {
        {0xB743F9CF9CAE14F0ULL, "sceNpWordFilterDeleteTitleCtx"},
        {0x3D8152D47EF46C3BULL, "sceNpWordFilterDeleteRequest"},
        {0xAC038EA830310489ULL, "sceNpWordFilterAbortRequest"},
    }, SCE_OK);
    add_handles("libSceNpWordFilter", {
        {0xAFD0602343DF2598ULL, "sceNpWordFilterCreateTitleCtx"},
        {0x882AB9C56E4A416EULL, "sceNpWordFilterCreateRequest"},
        {0x20407EBE056841BCULL, "sceNpWordFilterCreateAsyncRequest"},
    });
    add_stubs("libSceNpWordFilter", {
        {0xD5D32776A2FF4201ULL, "sceNpWordFilterCensorComment"},
        {0x263E2692914EDA01ULL, "sceNpWordFilterSanitizeComment"},
        {0xBABE524A1C86D1D9ULL, "sceNpWordFilterPollAsync"},
    }, kNpErrSignedOut);
    add_stubs("libSceNpSnsFacebook", {
        {0xA53BC0295D624241ULL, "sceNpSnsFacebookDeleteRequest"},
    }, SCE_OK);
    add_handles("libSceNpSnsFacebook", {
        {0x3A84E37F197CFF02ULL, "sceNpSnsFacebookCreateRequest"},
    });
    add_stubs("libSceNpSnsFacebook", {
        {0x56DC92F172C1A8D1ULL, "sceNpSnsFacebookGetAccessToken"},
    }, kNpErrSignedOut);

    // libSceNpInGameMessage / libSceNpBandwidthTest / libSceNpCommerce icons
    add_stubs("libSceNpInGameMessage", {
        {0x1858555294666C71ULL, "sceNpInGameMessageInitialize"},
        {0x6CC1B77159949AE9ULL, "sceNpInGameMessageTerminate"},
        {0xF9A9EE4B1D9ABC74ULL, "sceNpInGameMessageDeleteHandle"},
    }, SCE_OK);
    add_handles("libSceNpInGameMessage", {
        {0xB385046B988125D7ULL, "sceNpInGameMessageCreateHandle"},
    });
    add_stubs("libSceNpInGameMessage", {
        {0x561D5B8541BA992BULL, "sceNpInGameMessagePrepare"},
        {0x38DED27F95C43262ULL, "sceNpInGameMessageSendData"},
    }, kNpErrSignedOut);
    add_stubs("libSceNpBandwidthTest", {
        {0xA4BAF57C4412D73FULL, "sceNpBandwidthTestShutdown"},
    }, SCE_OK);
    add_stubs("libSceNpBandwidthTest", {
        {0x8E4B70C37C895E77ULL, "sceNpBandwidthTestInitStart"},
        {0x05821918A9BA6CEEULL, "sceNpBandwidthTestGetStatus"},
    }, kNpErrSignedOut);
    add_stubs("libSceNpCommerce", {
        {0x0C79B0B1AE92F137ULL, "sceNpCommerceShowPsStoreIcon"},
        {0x76CA8256C34CD198ULL, "sceNpCommerceHidePsStoreIcon"},
    }, SCE_OK);

    // --- Network: no interface, no servers ---
    add_stubs("libSceNet", {
        {0x3657AFECB83C9370ULL, "sceNetInit"},
        {0x7131A473AFD30652ULL, "sceNetTerm"},
        {0x2BB465AD3908FE6CULL, "sceNetPoolDestroy"},
        {0x9099581F9B8C0162ULL, "sceNetResolverDestroy"},
    }, SCE_OK);
    add_handles("libSceNet", {
        {0x76024169E2671A9AULL, "sceNetPoolCreate"},
        {0x0B85200C71CFBDDCULL, "sceNetResolverCreate"},
    });
    add_stubs("libSceNet", {
        {0x35DF7559A5A61B6CULL, "sceNetResolverStartNtoa"},
        {0x0296F8603C4AB112ULL, "sceNetResolverStartAton"},
    }, kNetCtlErrNotConnected);
    register_symbol(0x896416AF0892B7C0ULL, "sceNetHtons", "libSceNet",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = __builtin_bswap16(static_cast<u16>(ctx.rdi)); });
    register_symbol(0xF53DA90C5D91CAA8ULL, "sceNetHtonl", "libSceNet",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = __builtin_bswap32(static_cast<u32>(ctx.rdi)); });
    register_symbol(0xF0A729E5DFEAD54AULL, "sceNetInetPton", "libSceNet",
        [](CpuContext& ctx, MemoryManager& mem) { ctx.rax = guest_inet_pton(mem, ctx.rdi, ctx.rsi, ctx.rdx); });

    // libSceNetCtl: disconnected, with no callbacks ever fired.
    add_stubs("libSceNetCtl", {
        {0x824CB4FA868D3389ULL, "sceNetCtlInit"},
        {0x678C3008588110B4ULL, "sceNetCtlTerm"},
        {0x509F99ED0FB8724DULL, "sceNetCtlRegisterCallback"},
        {0x46A9B63A764C0B3DULL, "sceNetCtlUnregisterCallback"},
        {0x890C378903E1BD44ULL, "sceNetCtlCheckCallback"},
    }, SCE_OK);
    add_stubs("libSceNetCtl", {
        {0xA1BBB17538B0905FULL, "sceNetCtlGetInfo"},
        {0x24EE32B93B8CA0A2ULL, "sceNetCtlGetNatInfo"},
        {0xD1C06076E3D147E3ULL, "sceNetCtlGetResult"},
    }, kNetCtlErrNotConnected);
    // sceNetCtlGetState(int* state): SCE_NET_CTL_STATE_DISCONNECTED (0).
    register_symbol(0xB813E5AF495BBA22ULL, "sceNetCtlGetState", "libSceNetCtl",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (mem.is_valid_address(ctx.rdi, sizeof(u32))) mem.write_value<u32>(ctx.rdi, 0);
            ctx.rax = SCE_OK;
        });

    // libSceHttp / libSsl: templates, connections and requests are local
    // handles; anything that sends fails with a network error.
    add_stubs("libSceHttp", {
        {0x224FCAA4B4E57FB4ULL, "sceHttpTerm"},
        {0xE08F2F129B84859FULL, "sceHttpDeleteTemplate"},
        {0x3FA037CADA6C8987ULL, "sceHttpDeleteConnection"},
        {0xA9EEE867EBF83D60ULL, "sceHttpDeleteRequest"},
    }, SCE_OK);
    add_handles("libSceHttp", {
        {0x03D715314B44A786ULL, "sceHttpInit"},
        {0xD206233D347FE9C6ULL, "sceHttpCreateTemplate"},
        {0xAA0C43063A2B531BULL, "sceHttpCreateConnectionWithURL"},
        {0x01EBB9C152A417DCULL, "sceHttpCreateRequestWithURL"},
    });
    add_stubs("libSceHttp", {
        {0xD5ED8137023F5F31ULL, "sceHttpSendRequest"},
        {0xD1AD9304D7C4DC15ULL, "sceHttpGetStatusCode"},
    }, kHttpErrNetwork);
    add_stubs("libSceSsl", {
        {0xD0AD7243A2EFFD87ULL, "sceSslTerm"},
    }, SCE_OK);
    add_handles("libSceSsl", {
        {0x85DA551140C55B7BULL, "sceSslInit"},
    });

    // libSceNpToolkit
    register_symbol(0x65B009710E7F9CF8ULL, "ZbAJcQ5-nPg", "libSceNpToolkit",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr p = ctx.rdi;
            if (p != 0 && mem.is_valid_address(p, 0x400)) {
                mem.zero_memory(p, 0x400);
                mem.write_value<u32>(p + 0x18, 0x400);
            }
            ctx.rax = SCE_OK;
        });
    register_symbol(0x0275609908F7C918ULL, "AnVgmQj3yRg", "libSceNpToolkit",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr p = ctx.rdi;
            if (p != 0 && mem.is_valid_address(p, 0x400)) {
                mem.zero_memory(p, 0x400);
                mem.write_value<u32>(p + 0x18, 0x400);
            }
            ctx.rax = SCE_OK;
        });
    register_symbol(0x8E903E21CBC62C94ULL, "jpA+IcvGLJQ", "libSceNpToolkit",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    add_stubs("libSceGameLiveStreaming", {
        {0x92F604C369419DD9ULL, "sceGameLiveStreamingInitialize"},
        {0xFC41E753AF201315ULL, "sceGameLiveStreamingPermitServerSideRecording"},
        {0xABF931B9A17B5115ULL, "sceGameLiveStreamingSetMaxBitrate"},
    }, SCE_OK);
    add_stubs("libSceSharePlay", {
        {0x8ACAEEAAD86961CCULL, "sceSharePlayInitialize"},
        {0x728D8D0A3FFFA677ULL, "sceSharePlaySetProhibition"},
    }, SCE_OK);
    add_stubs("libSceRemoteplay", {
        {0x9354B082431238CFULL, "sceRemoteplayInitialize"},
        {0xC50788AF24D7EDD6ULL, "sceRemoteplayApprove"},
    }, SCE_OK);
    add_stubs("libSceScreenShot", {
        {0xEF7590E098F49C92ULL, "sceScreenShotSetOverlayImageWithOrigin"},
    }, SCE_OK);

    // libSceRtc: the local clock stands in for network time (no NTP offline).
    register_symbol(0xD7C076352D72F545ULL, "sceRtcGetCurrentTick", "libSceRtc",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (mem.is_valid_address(ctx.rdi, sizeof(u64))) mem.write_value<u64>(ctx.rdi, guest_rtc_now());
            ctx.rax = SCE_OK;
        });
    register_symbol(0xCCEF542F7A8820D4ULL, "sceRtcGetCurrentNetworkTick", "libSceRtc",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (mem.is_valid_address(ctx.rdi, sizeof(u64))) mem.write_value<u64>(ctx.rdi, guest_rtc_now());
            ctx.rax = SCE_OK;
        });
    register_symbol(0x3354EF16CB7F8EB3ULL, "sceRtcConvertUtcToLocalTime", "libSceRtc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr utc_ptr = ctx.rdi;
            const GuestAddr loc_ptr = ctx.rsi;
            u64 utc_tick = 0;
            if (utc_ptr != 0 && mem.is_valid_address(utc_ptr, sizeof(u64))) {
                mem.read_value(utc_ptr, utc_tick);
            }
            constexpr u64 kUnixEpochTicks = 62135596800ULL * 1000000ULL;
            u64 loc_tick = utc_tick;
            if (utc_tick >= kUnixEpochTicks) {
                const time_t sec = static_cast<time_t>((utc_tick - kUnixEpochTicks) / 1000000ULL);
                struct tm tm_loc{};
                localtime_r(&sec, &tm_loc);
                loc_tick = static_cast<u64>(static_cast<i64>(utc_tick) + static_cast<i64>(tm_loc.tm_gmtoff) * 1000000LL);
            }
            if (loc_ptr != 0 && mem.is_valid_address(loc_ptr, sizeof(u64))) {
                mem.write_value(loc_ptr, loc_tick);
            }
            ctx.rax = SCE_OK;
        });
    register_symbol(0xB9E7A06BABF7194CULL, "sceRtcSetTick", "libSceRtc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dt_ptr = ctx.rdi;
            const GuestAddr tick_ptr = ctx.rsi;
            u64 tick = 0;
            if (tick_ptr != 0 && mem.is_valid_address(tick_ptr, sizeof(u64))) {
                mem.read_value(tick_ptr, tick);
            }
            constexpr u64 kUnixEpochTicks = 62135596800ULL * 1000000ULL;
            time_t sec = 0;
            u32 usec = 0;
            if (tick >= kUnixEpochTicks) {
                sec = static_cast<time_t>((tick - kUnixEpochTicks) / 1000000ULL);
                usec = static_cast<u32>((tick - kUnixEpochTicks) % 1000000ULL);
            }
            struct tm tm_val{};
            gmtime_r(&sec, &tm_val);
            if (dt_ptr != 0 && mem.is_valid_address(dt_ptr, 16)) {
                mem.write_value<u16>(dt_ptr + 0, static_cast<u16>(tm_val.tm_year + 1900));
                mem.write_value<u16>(dt_ptr + 2, static_cast<u16>(tm_val.tm_mon + 1));
                mem.write_value<u16>(dt_ptr + 4, static_cast<u16>(tm_val.tm_mday));
                mem.write_value<u16>(dt_ptr + 6, static_cast<u16>(tm_val.tm_hour));
                mem.write_value<u16>(dt_ptr + 8, static_cast<u16>(tm_val.tm_min));
                mem.write_value<u16>(dt_ptr + 10, static_cast<u16>(tm_val.tm_sec));
                mem.write_value<u32>(dt_ptr + 12, usec);
            }
            ctx.rax = SCE_OK;
        });
    register_symbol(0xF30FC7D7D8A9E3C2ULL, "sceRtcGetTick", "libSceRtc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dt_ptr = ctx.rdi;
            const GuestAddr tick_ptr = ctx.rsi;
            u64 tick = 0;
            if (dt_ptr != 0 && mem.is_valid_address(dt_ptr, 16)) {
                u16 year = 0, mon = 0, mday = 0, hour = 0, min = 0, sec = 0;
                u32 usec = 0;
                mem.read_value(dt_ptr + 0, year);
                mem.read_value(dt_ptr + 2, mon);
                mem.read_value(dt_ptr + 4, mday);
                mem.read_value(dt_ptr + 6, hour);
                mem.read_value(dt_ptr + 8, min);
                mem.read_value(dt_ptr + 10, sec);
                mem.read_value(dt_ptr + 12, usec);

                struct tm tm_val{};
                tm_val.tm_year = year > 1900 ? year - 1900 : 0;
                tm_val.tm_mon = mon > 0 ? mon - 1 : 0;
                tm_val.tm_mday = mday > 0 ? mday : 1;
                tm_val.tm_hour = hour;
                tm_val.tm_min = min;
                tm_val.tm_sec = sec;
                time_t tsec = timegm(&tm_val);
                constexpr u64 kUnixEpochTicks = 62135596800ULL * 1000000ULL;
                tick = kUnixEpochTicks + static_cast<u64>(tsec) * 1000000ULL + usec;
            }
            if (tick_ptr != 0 && mem.is_valid_address(tick_ptr, sizeof(u64))) {
                mem.write_value(tick_ptr, tick);
            }
            ctx.rax = SCE_OK;
        });

    // --- Common dialogs: one status tracker per family ---
    // SCE_COMMON_DIALOG_STATUS_*: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3.
    // No dialog is ever drawn and nobody answers one, so Open finishes at once
    // (GetStatus then reads FINISHED), Terminate returns the family to NONE, and
    // GetResult reports SCE_COMMON_DIALOG_RESULT_USER_CANCELED (1).
    struct DialogTracker { std::atomic<u32> status{0}; };
    static DialogTracker s_dlg_invitation, s_dlg_friend_list, s_dlg_profile,
                         s_dlg_custom_data, s_dlg_commerce, s_dlg_facebook;
    static constexpr u32 kDlgStatusNone = 0;
    static constexpr u32 kDlgStatusFinished = 3;
    static constexpr u32 kDlgResultUserCanceled = 1;
    auto dlg_ok = [](CpuContext& ctx, MemoryManager&) { ctx.rax = SCE_OK; };
    auto dlg_open = [](DialogTracker& t) {
        return [&t](CpuContext& ctx, MemoryManager&) {
            t.status.store(kDlgStatusFinished);
            ctx.rax = SCE_OK;
        };
    };
    auto dlg_terminate = [](DialogTracker& t) {
        return [&t](CpuContext& ctx, MemoryManager&) {
            t.status.store(kDlgStatusNone);
            ctx.rax = SCE_OK;
        };
    };
    auto dlg_status = [](DialogTracker& t) {
        return [&t](CpuContext& ctx, MemoryManager&) { ctx.rax = t.status.load(); };
    };
    // GetResult(result*): the first word of the result struct is the end status.
    auto dlg_result = []() {
        return [](CpuContext& ctx, MemoryManager& mem) {
            if (mem.is_valid_address(ctx.rdi, sizeof(u32))) mem.write_value<u32>(ctx.rdi, kDlgResultUserCanceled);
            ctx.rax = SCE_OK;
        };
    };

    // libSceInvitationDialog
    register_symbol(0x5EF039292E7AC1CBULL, "sceInvitationDialogInitialize", "libSceInvitationDialog", dlg_ok);
    register_symbol(0xD335341BEC2254B0ULL, "sceInvitationDialogOpen", "libSceInvitationDialog", dlg_open(s_dlg_invitation));
    register_symbol(0x07A1D526D0D8C441ULL, "sceInvitationDialogTerminate", "libSceInvitationDialog", dlg_terminate(s_dlg_invitation));
    register_symbol(0xF7E83D88EABEEE48ULL, "sceInvitationDialogUpdateStatus", "libSceInvitationDialog", dlg_status(s_dlg_invitation));
    register_symbol(0xF17291EB06BAE224ULL, "sceInvitationDialogGetResult", "libSceInvitationDialog", dlg_result());
    // libSceNpFriendListDialog (no Initialize import)
    register_symbol(0xCD433F446E479B27ULL, "sceNpFriendListDialogOpen", "libSceNpFriendListDialog", dlg_open(s_dlg_friend_list));
    register_symbol(0xF0154FE7A24866E0ULL, "sceNpFriendListDialogGetStatus", "libSceNpFriendListDialog", dlg_status(s_dlg_friend_list));
    register_symbol(0x6782498570A722F6ULL, "sceNpFriendListDialogGetResult", "libSceNpFriendListDialog", dlg_result());
    register_symbol(0x10213393E2BD2F69ULL, "sceNpFriendListDialogTerminate", "libSceNpFriendListDialog", dlg_terminate(s_dlg_friend_list));
    // libSceNpProfileDialog (no Open import)
    register_symbol(0x2E0F8D084EA94F04ULL, "sceNpProfileDialogInitialize", "libSceNpProfileDialog", dlg_ok);
    register_symbol(0xDC1AA88853A34AC9ULL, "sceNpProfileDialogGetStatus", "libSceNpProfileDialog", dlg_status(s_dlg_profile));
    register_symbol(0xF2B84B975FF45BFAULL, "sceNpProfileDialogGetResult", "libSceNpProfileDialog", dlg_result());
    register_symbol(0xD12A7DBC9701D7FCULL, "sceNpProfileDialogTerminate", "libSceNpProfileDialog", dlg_terminate(s_dlg_profile));
    // libSceGameCustomDataDialog
    register_symbol(0xC6D6DBFF67FBD370ULL, "sceGameCustomDataDialogInitialize", "libSceGameCustomDataDialog", dlg_ok);
    register_symbol(0xE53BEDB7246E53CEULL, "sceGameCustomDataDialogOpen", "libSceGameCustomDataDialog", dlg_open(s_dlg_custom_data));
    register_symbol(0x1F012D1C50A95395ULL, "sceGameCustomDataDialogTerminate", "libSceGameCustomDataDialog", dlg_terminate(s_dlg_custom_data));
    register_symbol(0x3E474BB1103838DDULL, "sceGameCustomDataDialogUpdateStatus", "libSceGameCustomDataDialog", dlg_status(s_dlg_custom_data));
    register_symbol(0xB09A6D670BECD62BULL, "sceGameCustomDataDialogGetResult", "libSceGameCustomDataDialog", dlg_result());
    // libSceNpCommerceDialog
    register_symbol(0xD1A4766969906A5EULL, "sceNpCommerceDialogInitialize", "libSceNpCommerceDialog", dlg_ok);
    register_symbol(0x0DF4820D10371236ULL, "sceNpCommerceDialogOpen", "libSceNpCommerceDialog", dlg_open(s_dlg_commerce));
    register_symbol(0x9BF23DD806F9D16FULL, "sceNpCommerceDialogTerminate", "libSceNpCommerceDialog", dlg_terminate(s_dlg_commerce));
    register_symbol(0x2D1E5CC0530C0951ULL, "sceNpCommerceDialogUpdateStatus", "libSceNpCommerceDialog", dlg_status(s_dlg_commerce));
    register_symbol(0xAF8D9B59C41BB596ULL, "sceNpCommerceDialogGetResult", "libSceNpCommerceDialog", dlg_result());
    // libSceNpSnsFacebookDialog
    register_symbol(0xB5D400AD42DB3509ULL, "sceNpSnsFacebookDialogInitialize", "libSceNpSnsFacebookDialog", dlg_ok);
    register_symbol(0x1E2DE89EA1D237A3ULL, "sceNpSnsFacebookDialogOpen", "libSceNpSnsFacebookDialog", dlg_open(s_dlg_facebook));
    register_symbol(0x71DA8E172BA9452EULL, "sceNpSnsFacebookDialogTerminate", "libSceNpSnsFacebookDialog", dlg_terminate(s_dlg_facebook));
    register_symbol(0x7E357B0BC1F463C9ULL, "sceNpSnsFacebookDialogUpdateStatus", "libSceNpSnsFacebookDialog", dlg_status(s_dlg_facebook));
    register_symbol(0x0227DD708D5A29ABULL, "sceNpSnsFacebookDialogGetResult", "libSceNpSnsFacebookDialog", dlg_result());
    // libSceSaveDataDialog: Unity's SavedGames plugin shows one while it
    // loads ("no saved data" on a first boot) and waits for FINISHED before
    // reporting the load to the game; the fallback status 0 kept it waiting
    // forever, so the startup autoload never completed. Every dialog ends
    // at once with the OK button.
    static DialogTracker s_dlg_save_data;
    register_symbol(0xB3D7B7F98A519F3CULL, "sceSaveDataDialogInitialize", "libSceSaveDataDialog", dlg_ok);
    register_symbol(0xE2D3E1B0FE85A432ULL, "sceSaveDataDialogOpen", "libSceSaveDataDialog", dlg_open(s_dlg_save_data));
    register_symbol(0x7C7E3A2DA83CF176ULL, "sceSaveDataDialogClose", "libSceSaveDataDialog", dlg_open(s_dlg_save_data));
    register_symbol(0x62E1F6140EDACEA4ULL, "sceSaveDataDialogTerminate", "libSceSaveDataDialog", dlg_terminate(s_dlg_save_data));
    register_symbol(0x28ADC1760D5158ADULL, "sceSaveDataDialogUpdateStatus", "libSceSaveDataDialog", dlg_status(s_dlg_save_data));
    register_symbol(0x1112B392C6AE0090ULL, "sceSaveDataDialogGetStatus", "libSceSaveDataDialog", dlg_status(s_dlg_save_data));
    register_symbol(0x7A7EE03559E1F3BFULL, "sceSaveDataDialogIsReadyToDisplay", "libSceSaveDataDialog",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 1; });
    register_symbol(0x85ACB509F4E62F20ULL, "sceSaveDataDialogProgressBarSetValue", "libSceSaveDataDialog", dlg_ok);
    // sceSaveDataDirNameSearch: see register_save_data_and_trophy (host folders).
    // GetResult(SceSaveDataDialogResult*): mode (left as is), result 0, buttonId OK (1).
    register_symbol(0xC84889FEAAABE828ULL, "sceSaveDataDialogGetResult", "libSceSaveDataDialog",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (mem.is_valid_address(ctx.rdi, 12)) {
                mem.write_value<i32>(ctx.rdi + 4, 0);
                mem.write_value<i32>(ctx.rdi + 8, 1);
            }
            ctx.rax = SCE_OK;
        });

    // libSceMsgDialog: there is no system UI to show it in, so it is logged and
    // ends at once with OK / Yes. Dragon Ball Xenoverse 2 opens one after its
    // opening movie and polls UpdateStatus for FINISHED; the fallback's 0
    // left it on a black screen forever.
    static DialogTracker s_dlg_msg;
    register_symbol(0x943AB1698D546C4AULL, "sceMsgDialogInitialize", "libSceMsgDialog", dlg_ok);
    // sceMsgDialogOpen(const SceMsgDialogParam*): mode at +0x38 (1 = user text),
    // userMsgParam at +0x40 = { int buttonType; const char* msg at +8 }.
    register_symbol(0x6F4E878740CF11A1ULL, "sceMsgDialogOpen", "libSceMsgDialog",
        [](CpuContext& ctx, MemoryManager& mem) {
            i32 mode = 0;
            GuestAddr user = 0, msg = 0;
            std::string text;
            mem.read_value(ctx.rdi + 0x38, mode);
            if (mode == 1 && mem.read_value(ctx.rdi + 0x40, user) && user != 0 && mem.read_value(user + 8, msg))
                read_guest_cstring(mem, msg, text);
            std::fprintf(stderr, "[MsgDialog] mode %d: %s\n", mode, text.c_str());
            s_dlg_msg.status.store(kDlgStatusFinished);
            ctx.rax = SCE_OK;
        });
    register_symbol(0x1D3ADC0CA9452AE3ULL, "sceMsgDialogClose", "libSceMsgDialog", dlg_open(s_dlg_msg));
    register_symbol(0x78FC3F92A6667A5AULL, "sceMsgDialogTerminate", "libSceMsgDialog", dlg_terminate(s_dlg_msg));
    register_symbol(0xE9F202DD72ADDA4DULL, "sceMsgDialogUpdateStatus", "libSceMsgDialog", dlg_status(s_dlg_msg));
    register_symbol(0x096556EFC41CDDF2ULL, "sceMsgDialogGetStatus", "libSceMsgDialog", dlg_status(s_dlg_msg));
    // GetResult(SceMsgDialogResult*): mode (left as is), result 0, buttonId OK/Yes (1).
    register_symbol(0x2EBF28BC71FD97A0ULL, "sceMsgDialogGetResult", "libSceMsgDialog",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (mem.is_valid_address(ctx.rdi, 12)) {
                mem.write_value<i32>(ctx.rdi + 4, 0);
                mem.write_value<i32>(ctx.rdi + 8, 1);
            }
            ctx.rax = SCE_OK;
        });
    // libSceErrorDialog: logged, ends at once.
    static DialogTracker s_dlg_error;
    register_symbol(0x23CF0A0A19729D2BULL, "sceErrorDialogInitialize", "libSceErrorDialog", dlg_ok);
    // sceErrorDialogOpen(const SceErrorDialogParam*): errorCode at +4.
    register_symbol(0x336645FC294B8606ULL, "sceErrorDialogOpen", "libSceErrorDialog",
        [](CpuContext& ctx, MemoryManager& mem) {
            u32 code = 0;
            mem.read_value(ctx.rdi + 4, code);
            std::fprintf(stderr, "[ErrorDialog] error 0x%08x\n", code);
            s_dlg_error.status.store(kDlgStatusFinished);
            ctx.rax = SCE_OK;
        });
    register_symbol(0x7A45C76F5903065DULL, "sceErrorDialogClose", "libSceErrorDialog", dlg_open(s_dlg_error));
    register_symbol(0xF570312B63CCC24FULL, "sceErrorDialogTerminate", "libSceErrorDialog", dlg_terminate(s_dlg_error));
    register_symbol(0x596886BA1F577E04ULL, "sceErrorDialogUpdateStatus", "libSceErrorDialog", dlg_status(s_dlg_error));
    register_symbol(0xB7616F1D15F382A9ULL, "sceErrorDialogGetStatus", "libSceErrorDialog", dlg_status(s_dlg_error));

    // --- libc / libc++ gaps (real implementations) ---
    register_symbol(0x015EA2A4235AE11CULL, "strcasecmp", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string a, b;
            read_guest_cstring(mem, ctx.rdi, a);
            read_guest_cstring(mem, ctx.rsi, b);
            ctx.rax = static_cast<u64>(static_cast<i64>(guest_strcasecmp(a, b)));
        });
    register_symbol(0xE6336E6F0E2F9400ULL, "strnlen", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            GuestSize n = 0;
            char c = 0;
            while (n < ctx.rsi && mem.read_value(ctx.rdi + n, c) && c != 0) ++n;
            ctx.rax = n;
        });
    register_symbol(0xF2EF253F3504ABE5ULL, "memchr", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr s = ctx.rdi;
            const GuestSize n = ctx.rdx;
            ctx.rax = 0;
            if (n == 0 || !mem.is_valid_address(s, n)) return;
            const u8* base = mem.to_host_ptr<u8>(s);
            if (const void* hit = std::memchr(base, static_cast<int>(ctx.rsi & 0xFF), n))
                ctx.rax = s + static_cast<GuestAddr>(static_cast<const u8*>(hit) - base);
        });
    register_symbol(0xA15919F16F3F43C0ULL, "strtok", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            // Dinkumware keeps strtok's scan position per thread; Unity's
            // shader loader and main thread tokenize concurrently. Guest
            // threads are not 1:1 with host threads, so key by guest tid.
            static std::mutex s_strtok_mtx;
            static std::unordered_map<ScePthreadId, GuestAddr> s_strtok_save;
            const ScePthreadId tid = current_hle_thread_id();
            GuestAddr save = 0;
            {
                std::lock_guard<std::mutex> lk(s_strtok_mtx);
                save = s_strtok_save[tid];
            }
            ctx.rax = guest_strtok(mem, ctx.rdi, ctx.rsi, save);
            std::lock_guard<std::mutex> lk(s_strtok_mtx);
            s_strtok_save[tid] = save;
        });
    register_symbol(0x7A7A8F18B7E654D5ULL, "strtok_r", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            GuestAddr save = 0;
            if (!mem.read_value(ctx.rdx, save)) {
                ctx.rax = 0;
                return;
            }
            ctx.rax = guest_strtok(mem, ctx.rdi, ctx.rsi, save);
            mem.write_value<GuestAddr>(ctx.rdx, save);
        });
    register_symbol(0x9979718662CD30F8ULL, "strtol", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string text;
            read_guest_cstring(mem, ctx.rdi, text);
            char* end = nullptr;
            const long v = std::strtol(text.c_str(), &end, static_cast<int>(ctx.rdx));
            store_end_pointer(mem, ctx.rsi, ctx.rdi, text, end);
            ctx.rax = static_cast<u64>(static_cast<i64>(v));
        });
    // _Stod(const char* s, char** endptr, ...): strtod with the end pointer.
    register_symbol(0x738D5410756D8840ULL, "_Stod", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string text;
            read_guest_cstring(mem, ctx.rdi, text);
            char* end = nullptr;
            ctx.ymm[0].df[0] = std::strtod(text.c_str(), &end);
            store_end_pointer(mem, ctx.rsi, ctx.rdi, text, end);
        });
    // atof(const char* s): converts string to double in XMM0
    register_symbol(0x49123A4BD07EFDAEULL, "atof", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string text;
            if (read_guest_cstring(mem, ctx.rdi, text)) {
                ctx.ymm[0].df[0] = std::strtod(text.c_str(), nullptr);
            } else {
                ctx.ymm[0].df[0] = 0.0;
            }
        });
    // __error(): returns pointer to errno in guest memory
    register_symbol(0xF41703CA43E6A352ULL, "__error", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            constexpr GuestAddr GUEST_ERRNO_ADDR = 0x3FF2000ULL;
            mem.write_value<int>(GUEST_ERRNO_ADDR, 0);
            ctx.rax = GUEST_ERRNO_ADDR;
        });

    // _Stoll(const char* s, char** endptr, int base): strtoll with the end pointer.
    register_symbol(0xEE934A72C70AADFFULL, "_Stoll", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string text;
            read_guest_cstring(mem, ctx.rdi, text);
            char* end = nullptr;
            const long long v = std::strtoll(text.c_str(), &end, static_cast<int>(ctx.rdx));
            store_end_pointer(mem, ctx.rsi, ctx.rdi, text, end);
            ctx.rax = static_cast<u64>(v);
        });

    // ldiv(long numer, long denom): returns ldiv_t in RAX (quot) and RDX (rem) (AMD64 ABI)
    // NID: 0x81F3F48A6E59DE0DULL ("gfP0im5Z3g0")
    register_symbol(0x81F3F48A6E59DE0DULL, "ldiv", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const i64 numer = static_cast<i64>(ctx.rdi);
            const i64 denom = static_cast<i64>(ctx.rsi);
            if (denom == 0) {
                ctx.rax = 0;
                ctx.rdx = 0;
                return;
            }
            const auto res = std::ldiv(numer, denom);
            ctx.rax = static_cast<u64>(res.quot);
            ctx.rdx = static_cast<u64>(res.rem);
        });

    // lldiv(long long numer, long long denom): returns lldiv_t in RAX (quot) and RDX (rem)
    // NID: 0x89584966F00ED9A4ULL ("iViJ7vAO2aQ")
    register_symbol(0x89584966F00ED9A4ULL, "lldiv", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const i64 numer = static_cast<i64>(ctx.rdi);
            const i64 denom = static_cast<i64>(ctx.rsi);
            if (denom == 0) {
                ctx.rax = 0;
                ctx.rdx = 0;
                return;
            }
            const auto res = std::lldiv(numer, denom);
            ctx.rax = static_cast<u64>(res.quot);
            ctx.rdx = static_cast<u64>(res.rem);
        });

    // div(int numer, int denom): returns div_t packed in RAX (quot in bits 0..31, rem in bits 32..63)
    // NID: 0xDA06DC96D937B301ULL ("2gD286L3B24")
    register_symbol(0xDA06DC96D937B301ULL, "div", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            const i32 numer = static_cast<i32>(ctx.rdi);
            const i32 denom = static_cast<i32>(ctx.rsi);
            if (denom == 0) {
                ctx.rax = 0;
                return;
            }
            const auto res = std::div(numer, denom);
            ctx.rax = (static_cast<u64>(static_cast<u32>(res.rem)) << 32) | static_cast<u32>(res.quot);
        });
    register_symbol(0xA4AC2C96C3149929ULL, "fmod", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::fmod(ctx.ymm[0].df[0], ctx.ymm[1].df[0]); });
    register_symbol(0xF3C56FFC0CC7563FULL, "fmodf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::fmod(ctx.ymm[0].f[0], ctx.ymm[1].f[0]); });
    register_symbol(0xADC56FE62BCC858DULL, "lrintf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::lrint(ctx.ymm[0].f[0])); });
    register_symbol(0x54E28E811ECBFF66ULL, "lrint", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::lrint(ctx.ymm[0].df[0])); });
    register_symbol(0x28FB10034A62B3CAULL, "llrintf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::llrint(ctx.ymm[0].f[0])); });
    register_symbol(0xFF8DF503F601024BULL, "llrint", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::llrint(ctx.ymm[0].df[0])); });
    register_symbol(0x0BA81609624A33E5ULL, "lroundf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::lround(ctx.ymm[0].f[0])); });
    register_symbol(0x2775EE192FDC0B44ULL, "lround", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::lround(ctx.ymm[0].df[0])); });
    register_symbol(0x7908411674CEA78DULL, "llroundf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::llround(ctx.ymm[0].f[0])); });
    register_symbol(0xC3F06F5C5E0EEB1AULL, "llround", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::llround(ctx.ymm[0].df[0])); });
    register_symbol(0x0C31C6D5AEBEDEADULL, "roundf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::round(ctx.ymm[0].f[0]); });
    register_symbol(0x9E56A88CBF610ED0ULL, "round", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::round(ctx.ymm[0].df[0]); });
    register_symbol(0x98A8550E661C8960ULL, "floorf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::floor(ctx.ymm[0].f[0]); });
    register_symbol(0x9A971380CCE1518FULL, "floor", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::floor(ctx.ymm[0].df[0]); });
    register_symbol(0x18052E2CA1A1B02CULL, "ceilf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::ceil(ctx.ymm[0].f[0]); });
    register_symbol(0x81A71F3A63BC84DBULL, "ceil", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::ceil(ctx.ymm[0].df[0]); });
    register_symbol(0x568F2BBD6B59C378ULL, "truncf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::trunc(ctx.ymm[0].f[0]); });
    register_symbol(0x6B880B1ACA4F1033ULL, "trunc", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::trunc(ctx.ymm[0].df[0]); });
    register_symbol(0xAB95B3B9CC954A43ULL, "rintf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::rint(ctx.ymm[0].f[0]); });
    register_symbol(0x2F11886182B02987ULL, "rint", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::rint(ctx.ymm[0].df[0]); });
    register_symbol(0x73EE2BFD3FED1087ULL, "nearbyintf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::nearbyint(ctx.ymm[0].f[0]); });
    register_symbol(0x7092D3C2D2865C99ULL, "nearbyint", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::nearbyint(ctx.ymm[0].df[0]); });
    // modf(double x, double* iptr): integer part to *iptr, fraction returned.
    register_symbol(0xD163070DBE43B7DEULL, "modf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            double ip = 0.0;
            const double frac = std::modf(ctx.ymm[0].df[0], &ip);
            if (mem.is_valid_address(ctx.rdi, sizeof(double))) mem.write_value<double>(ctx.rdi, ip);
            ctx.ymm[0].df[0] = frac;
        });
    register_symbol(0xF4B0A3A56C90E597ULL, "pow", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::pow(ctx.ymm[0].df[0], ctx.ymm[1].df[0]); });
    register_symbol(0x35569D7E7CD08474ULL, "exp", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::exp(ctx.ymm[0].df[0]); });
    register_symbol(0x4FBBB236A3FBBD00ULL, "tan", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::tan(ctx.ymm[0].df[0]); });
    register_symbol(0x644E9134BF9E2DB9ULL, "tanf", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].f[0] = std::tan(ctx.ymm[0].f[0]); });
    register_symbol(0x24CE0406F593F6B7ULL, "tanh", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::tanh(ctx.ymm[0].df[0]); });
    register_symbol(0x24172062E5BC94F5ULL, "acos", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::acos(ctx.ymm[0].df[0]); });
    register_symbol(0xECBCB9DB368BE384ULL, "asin", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::asin(ctx.ymm[0].df[0]); });
    register_symbol(0x39799AB8B750F246ULL, "atan", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::atan(ctx.ymm[0].df[0]); });
    register_symbol(0x1D46D998E9D3FC38ULL, "atan2", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.ymm[0].df[0] = std::atan2(ctx.ymm[0].df[0], ctx.ymm[1].df[0]); });
    // vsprintf(char* dest, const char* fmt, va_list ap): returns the length.
    register_symbol(0x8DBCFD23DBE4AA49ULL, "vsprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string fmt;
            if (!read_guest_cstring(mem, ctx.rsi, fmt)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            VaListArgs args(mem, ctx.rdx);
            const std::string out = format_guest(mem, fmt.c_str(), args);
            if (mem.is_valid_address(ctx.rdi, out.size() + 1)) mem.write_bytes(ctx.rdi, out.c_str(), out.size() + 1);
            ctx.rax = static_cast<u64>(out.size());
        });
    // vsnprintf_s(char* buf, size_t bufsz, size_t count, const char* fmt, va_list ap).
    // Writes at most min(count, bufsz - 1) characters; returns that length, or -1
    // when the output had to be cut short.
    register_symbol(0xAD64AE4D663624DDULL, "vsnprintf_s", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestSize size = ctx.rsi;
            const GuestSize count = ctx.rdx;
            std::string fmt;
            if (size == 0 || !read_guest_cstring(mem, ctx.rcx, fmt)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            VaListArgs args(mem, ctx.r8);
            const std::string out = format_guest(mem, fmt.c_str(), args);
            const usize n = std::min<usize>({static_cast<usize>(count), static_cast<usize>(size - 1), out.size()});
            if (mem.is_valid_address(dest, n + 1)) {
                mem.write_bytes(dest, out.c_str(), n);
                mem.write_value<u8>(dest + n, 0);
            }
            ctx.rax = out.size() > n ? static_cast<u64>(-1) : static_cast<u64>(n);
        });
    // div(int numer, int denom): div_t comes back packed in rax (quot low, rem high).
    register_symbol(0xDA06DC96D937B301ULL, "div", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 num = static_cast<i32>(ctx.rdi);
            const i32 den = static_cast<i32>(ctx.rsi);
            if (den == 0 || (num == INT32_MIN && den == -1))
                guest_abort("div(): division by zero or overflow (SIGFPE on the PS4)", ctx, mem);
            const u64 q = static_cast<u32>(num / den);
            const u64 r = static_cast<u32>(num % den);
            ctx.rax = q | (r << 32);
        });
    register_symbol(0x72908E5D632085ADULL, "rand", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = static_cast<u64>(std::rand()); });
    register_symbol(0xD30BB7DE1BA735D1ULL, "sleep", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            std::this_thread::sleep_for(std::chrono::seconds(static_cast<u32>(ctx.rdi)));
            ctx.rax = 0;
        });
    // perror(const char* s): the message goes to stderr; guest errno is not modelled.
    register_symbol(0x10CBADC1A437E09AULL, "perror", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string text;
            read_guest_cstring(mem, ctx.rdi, text);
            std::fprintf(stderr, "[guest perror] %s\n", text.c_str());
            ctx.rax = 0;
        });
    // setlocale(int category, const char* locale): only the C locale exists.
    register_symbol(0x3EDB01D50F70B050ULL, "setlocale", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (ctx.rsi == 0) {
                ctx.rax = guest_static_string(mem, "C");
                return;
            }
            std::string loc;
            read_guest_cstring(mem, ctx.rsi, loc);
            const bool c_locale = loc.empty() || loc == "C" || loc == "POSIX";
            ctx.rax = c_locale ? guest_static_string(mem, "C") : 0;
        });
    register_symbol(0x4486BA1A75A9FA25ULL, "strerror", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = guest_static_string(mem, std::strerror(static_cast<int>(ctx.rdi)));
        });
    // _Syserror_map(int): the same message text, under its C++ name.
    register_symbol(0x3C35F4D73894CFEEULL, "_ZSt13_Syserror_mapi", "libstdc++",
        [](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = guest_static_string(mem, std::strerror(static_cast<int>(ctx.rdi)));
        });
    // _Fltrounds(): FLT_ROUNDS = 1, round to nearest.
    register_symbol(0x268F4E37F017F5E5ULL, "_Fltrounds", "libc",
        [](CpuContext& ctx, MemoryManager&) { ctx.rax = 1; });
    // Wide strings. The PS4 toolchain's wchar_t is 16-bit (clang's PS4 target
    // uses unsigned short; Dragon Ball Xenoverse 2 tests `word ptr [s]` before
    // wcslen and stores L"%d" as UTF-16). wcsncpy used to assume 32-bit units
    // and the rest were unregistered: wcslen and mbstowcs returned 0, a wide
    // name came back empty, and the game threw "invalid string position".
    static constexpr GuestSize kWideMax = GuestSize{1} << 24;
    auto wide_len = [](MemoryManager& mem, GuestAddr s, GuestSize limit) -> GuestSize {
        GuestSize n = 0;
        u16 wc = 0;
        while (n < limit && mem.read_value(s + 2 * n, wc) && wc != 0) ++n;
        return n;
    };
    // wcslen(const wchar_t* s)
    register_symbol(0x5A491ECB02C97205ULL, "wcslen", "libc",
        [wide_len](CpuContext& ctx, MemoryManager& mem) { ctx.rax = wide_len(mem, ctx.rdi, kWideMax); });
    // wcscpy(wchar_t* dest, const wchar_t* src) -> dest
    register_symbol(0x14CE4D3E72EA05CFULL, "wcscpy", "libc",
        [wide_len](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize n = wide_len(mem, ctx.rsi, kWideMax);
            std::vector<u16> buf(n + 1, 0);
            if (n) mem.read_bytes(ctx.rsi, buf.data(), n * 2);
            mem.write_bytes(ctx.rdi, buf.data(), (n + 1) * 2);
            ctx.rax = ctx.rdi;
        });
    // wcsncpy(wchar_t* dest, const wchar_t* src, size_t n): copies up to n units, then pads with zeros.
    register_symbol(0xD27576D498D8087FULL, "wcsncpy", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestAddr src = ctx.rsi;
            const GuestSize n = ctx.rdx;
            ctx.rax = dest;
            if (n == 0 || n > kWideMax || !mem.is_valid_address(dest, n * 2)) return;
            bool copying = true;
            for (GuestSize i = 0; i < n; ++i) {
                u16 wc = 0;
                if (copying && (!mem.read_value(src + 2 * i, wc) || wc == 0)) copying = false;
                if (!copying) wc = 0;
                mem.write_value<u16>(dest + 2 * i, wc);
            }
        });
    // wcscmp / wcsncmp: compare as unsigned 16-bit units.
    auto wide_cmp = [](MemoryManager& mem, GuestAddr a, GuestAddr b, GuestSize n) -> i64 {
        for (GuestSize i = 0; i < n; ++i) {
            u16 x = 0, y = 0;
            mem.read_value(a + 2 * i, x);
            mem.read_value(b + 2 * i, y);
            if (x != y) return x < y ? -1 : 1;
            if (x == 0) return 0;
        }
        return 0;
    };
    register_symbol(0xA4DB49744DF1E3D1ULL, "wcscmp", "libc",
        [wide_cmp](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = static_cast<u64>(wide_cmp(mem, ctx.rdi, ctx.rsi, kWideMax));
        });
    register_symbol(0x13CC02A1411B7F39ULL, "wcsncmp", "libc",
        [wide_cmp](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = static_cast<u64>(wide_cmp(mem, ctx.rdi, ctx.rsi, std::min<GuestSize>(ctx.rdx, kWideMax)));
        });
    // wcsstr(const wchar_t* haystack, const wchar_t* needle) -> match or NULL
    register_symbol(0x583A686E322601BEULL, "wcsstr", "libc",
        [wide_len](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize hn = wide_len(mem, ctx.rdi, kWideMax), nn = wide_len(mem, ctx.rsi, kWideMax);
            if (nn == 0) {
                ctx.rax = ctx.rdi;
                return;
            }
            std::vector<u16> h(hn), nd(nn);
            if (hn) mem.read_bytes(ctx.rdi, h.data(), hn * 2);
            mem.read_bytes(ctx.rsi, nd.data(), nn * 2);
            const auto it = std::search(h.begin(), h.end(), nd.begin(), nd.end());
            ctx.rax = it == h.end() ? 0 : ctx.rdi + 2 * static_cast<GuestAddr>(it - h.begin());
        });
    // wmem*: n counts 16-bit units.
    register_symbol(0x7E75048C108D4555ULL, "wmemchr", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const u16 c = static_cast<u16>(ctx.rsi);
            ctx.rax = 0;
            for (GuestSize i = 0; i < std::min<GuestSize>(ctx.rdx, kWideMax); ++i) {
                u16 wc = 0;
                if (!mem.read_value(ctx.rdi + 2 * i, wc)) break;
                if (wc == c) {
                    ctx.rax = ctx.rdi + 2 * i;
                    break;
                }
            }
        });
    register_symbol(0x409E7155F2A49E2DULL, "wmemcmp", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            ctx.rax = 0;
            for (GuestSize i = 0; i < std::min<GuestSize>(ctx.rdx, kWideMax); ++i) {
                u16 x = 0, y = 0;
                mem.read_value(ctx.rdi + 2 * i, x);
                mem.read_value(ctx.rsi + 2 * i, y);
                if (x != y) {
                    ctx.rax = static_cast<u64>(x < y ? -1 : 1);
                    break;
                }
            }
        });
    auto wmem_move = [](CpuContext& ctx, MemoryManager& mem) {
        const GuestSize bytes = std::min<GuestSize>(ctx.rdx, kWideMax) * 2;
        if (bytes && mem.is_valid_address(ctx.rsi, bytes) && mem.is_valid_address(ctx.rdi, bytes)) {
            std::vector<u8> buf(bytes);
            mem.read_bytes(ctx.rsi, buf.data(), bytes);
            mem.write_bytes(ctx.rdi, buf.data(), bytes);
        }
        ctx.rax = ctx.rdi;
    };
    register_symbol(0x7CBDCED36CA96451ULL, "wmemcpy", "libc", wmem_move);
    register_symbol(0x3688FD3EC26BB1AFULL, "wmemmove", "libc", wmem_move);
    register_symbol(0x025F0C64987FE213ULL, "wmemset", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize n = std::min<GuestSize>(ctx.rdx, kWideMax);
            if (n && mem.is_valid_address(ctx.rdi, n * 2)) {
                std::vector<u16> buf(n, static_cast<u16>(ctx.rsi));
                mem.write_bytes(ctx.rdi, buf.data(), n * 2);
            }
            ctx.rax = ctx.rdi;
        });
    // mbstowcs(wchar_t* dst, const char* src, size_t n): UTF-8 -> UTF-16, n
    // units at most; with dst NULL it returns the length it would write. Bytes
    // that are not valid UTF-8 pass through as Latin-1.
    register_symbol(0x554CE35E49CF3C1BULL, "mbstowcs", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string src;
            for (GuestSize i = 0; i < kWideMax; ++i) {
                char c = 0;
                if (!mem.read_value(ctx.rsi + i, c) || c == 0) break;
                src.push_back(c);
            }
            std::vector<u16> out;
            for (usize i = 0; i < src.size();) {
                const u8 b = static_cast<u8>(src[i]);
                u32 cp = b;
                usize len = 1;
                if (b >= 0xF0 && i + 3 < src.size()) {
                    cp = ((b & 0x07u) << 18) | ((src[i + 1] & 0x3Fu) << 12) | ((src[i + 2] & 0x3Fu) << 6) | (src[i + 3] & 0x3Fu);
                    len = 4;
                } else if (b >= 0xE0 && i + 2 < src.size()) {
                    cp = ((b & 0x0Fu) << 12) | ((src[i + 1] & 0x3Fu) << 6) | (src[i + 2] & 0x3Fu);
                    len = 3;
                } else if (b >= 0xC0 && i + 1 < src.size()) {
                    cp = ((b & 0x1Fu) << 6) | (src[i + 1] & 0x3Fu);
                    len = 2;
                }
                if (cp >= 0x10000) {
                    cp -= 0x10000;
                    out.push_back(static_cast<u16>(0xD800 + (cp >> 10)));
                    out.push_back(static_cast<u16>(0xDC00 + (cp & 0x3FF)));
                } else {
                    out.push_back(static_cast<u16>(cp));
                }
                i += len;
            }
            if (ctx.rdi == 0) {
                ctx.rax = out.size();
                return;
            }
            const GuestSize n = ctx.rdx;
            const GuestSize count = std::min<GuestSize>(out.size(), n);
            if (count) mem.write_bytes(ctx.rdi, out.data(), count * 2);
            if (count < n) mem.write_value<u16>(ctx.rdi + 2 * count, 0);
            ctx.rax = count;
        });
    // wcstombs(char* dst, const wchar_t* src, size_t n): UTF-16 -> UTF-8, n bytes at most.
    register_symbol(0xBFB4BB2E13F63897ULL, "wcstombs", "libc",
        [wide_len](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize wn = wide_len(mem, ctx.rsi, kWideMax);
            std::vector<u16> w(wn);
            if (wn) mem.read_bytes(ctx.rsi, w.data(), wn * 2);
            std::string out;
            for (usize i = 0; i < w.size(); ++i) {
                u32 cp = w[i];
                if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < w.size() && w[i + 1] >= 0xDC00 && w[i + 1] < 0xE000)
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (w[++i] - 0xDC00);
                if (cp < 0x80) out.push_back(static_cast<char>(cp));
                else if (cp < 0x800) {
                    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else if (cp < 0x10000) {
                    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else {
                    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                }
            }
            if (ctx.rdi == 0) {
                ctx.rax = out.size();
                return;
            }
            const GuestSize count = std::min<GuestSize>(out.size(), ctx.rdx);
            if (count) mem.write_bytes(ctx.rdi, out.data(), count);
            if (count < ctx.rdx) mem.write_value<u8>(ctx.rdi + count, 0);
            ctx.rax = count;
        });
    // mktime(struct tm*): normalises the fields in place and returns the time_t.
    register_symbol(0x9FB01EA70474B37EULL, "mktime", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::tm t{};
            if (!load_guest_tm(mem, ctx.rdi, t)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const std::time_t r = std::mktime(&t);
            store_guest_tm(mem, ctx.rdi, t, false);
            ctx.rax = static_cast<u64>(static_cast<i64>(r));
        });
    register_symbol(0x79F84AFD84946184ULL, "localtime", "libc",
        [](CpuContext& ctx, MemoryManager& mem) { ctx.rax = guest_time_split(mem, ctx.rdi, true); });
    register_symbol(0xD6679C3FB4602360ULL, "gmtime", "libc",
        [](CpuContext& ctx, MemoryManager& mem) { ctx.rax = guest_time_split(mem, ctx.rdi, false); });
    // difftime(time_t t1, time_t t0) -> double t1 - t0.
    register_symbol(0xFD5567EF8672844BULL, "difftime", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.ymm[0].df[0] = static_cast<double>(static_cast<i64>(ctx.rdi) - static_cast<i64>(ctx.rsi));
        });
    // strftime(char* s, size_t max, const char* fmt, const struct tm* tm)
    register_symbol(0x02FDF38D68BAE0ACULL, "strftime", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestSize max = ctx.rsi;
            std::string fmt;
            std::tm t{};
            ctx.rax = 0;
            if (max == 0 || max > (GuestSize{1} << 20) || !read_guest_cstring(mem, ctx.rdx, fmt)
                || !load_guest_tm(mem, ctx.rcx, t))
                return;
            std::vector<char> buf(static_cast<usize>(max));
            const usize n = std::strftime(buf.data(), buf.size(), fmt.c_str(), &t);
            if (n == 0 || !mem.is_valid_address(dest, n + 1)) return;
            mem.write_bytes(dest, buf.data(), n);
            mem.write_value<u8>(dest + n, 0);
            ctx.rax = n;
        });
    register_symbol(0xE27E75B34CC47F47ULL, "inet_pton", "libc",
        [](CpuContext& ctx, MemoryManager& mem) { ctx.rax = guest_inet_pton(mem, ctx.rdi, ctx.rsi, ctx.rdx); });
    // _Assert(expr, file, line): Dinkumware assertion. It always terminates.
    register_symbol(0xFD082A393E6ED959ULL, "_Assert", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string expr, file;
            read_guest_cstring(mem, ctx.rdi, expr);
            read_guest_cstring(mem, ctx.rsi, file);
            guest_abort("_Assert failed: " + expr + " (" + file + ")", ctx, mem);
        });
    // scePthreadGetaffinity(thread, SceKernelCpumask* mask): every core is allowed.
    register_symbol(0xADCAD5149B105916ULL, "scePthreadGetaffinity", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            if (mem.is_valid_address(ctx.rsi, sizeof(u64))) mem.write_value<u64>(ctx.rsi, 0xFFULL);
            ctx.rax = SCE_OK;
        });

    // stdio on host and guest FILE pointers:
    register_symbol(0x42B659749F17B17DULL, "fputs", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            FILE* fp = resolve_guest_stream(ctx.rsi);
            std::string text;
            if (!fp || !read_guest_cstring(mem, ctx.rdi, text)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            ctx.rax = static_cast<u64>(static_cast<i64>(std::fputs(text.c_str(), fp)));
        });
    register_symbol(0x6992BC94D7A2FD0CULL, "fputc", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            FILE* fp = resolve_guest_stream(ctx.rsi);
            ctx.rax = fp ? static_cast<u64>(static_cast<i64>(std::fputc(static_cast<int>(ctx.rdi), fp)))
                         : static_cast<u64>(-1);
        });
    // fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream)
    register_symbol(0x329C61321F1016BAULL, "fwrite", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr src = ctx.rdi;
            const GuestSize size = ctx.rsi;
            const GuestSize n = ctx.rdx;
            FILE* fp = resolve_guest_stream(ctx.rcx);
            ctx.rax = 0;
            if (!fp || size == 0 || n == 0 || size > (GuestSize{1} << 32) || n > (GuestSize{1} << 32)) return;
            if (!mem.is_valid_address(src, size * n)) return;
            ctx.rax = std::fwrite(mem.to_host_ptr<void>(src), size, n, fp);
        });
    // fprintf(FILE* stream, const char* fmt, ...)
    register_symbol(0x7DF7F010B5CD5450ULL, "fprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            FILE* fp = resolve_guest_stream(ctx.rdi);
            std::string fmt;
            if (!fp || !read_guest_cstring(mem, ctx.rsi, fmt)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            CallArgs args(ctx, mem, 2);
            const std::string out = format_guest(mem, fmt.c_str(), args);
            ctx.rax = std::fwrite(out.data(), 1, out.size(), fp) == out.size()
                          ? static_cast<u64>(out.size()) : static_cast<u64>(-1);
        });
    // vfprintf(FILE* stream, const char* fmt, va_list ap)
    register_symbol(0xA43043718EAE2D20ULL, "vfprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            FILE* fp = resolve_guest_stream(ctx.rdi);
            std::string fmt;
            if (!fp || !read_guest_cstring(mem, ctx.rsi, fmt)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            VaListArgs args(mem, ctx.rdx);
            const std::string out = format_guest(mem, fmt.c_str(), args);
            ctx.rax = std::fwrite(out.data(), 1, out.size(), fp) == out.size()
                          ? static_cast<u64>(out.size()) : static_cast<u64>(-1);
        });
    // fgets(char* buf, int n, FILE* stream)
    register_symbol(0x29D3FF9D42E9B86CULL, "fgets", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dst = ctx.rdi;
            const int n = static_cast<int>(ctx.rsi);
            FILE* fp = resolve_guest_stream(ctx.rdx);
            ctx.rax = 0;
            if (!fp || n <= 0 || !mem.is_valid_address(dst, static_cast<GuestSize>(n))) return;
            std::vector<char> line(static_cast<usize>(n));
            if (std::fgets(line.data(), n, fp) == nullptr) return;
            mem.write_bytes(dst, line.data(), std::strlen(line.data()) + 1);
            ctx.rax = dst;
        });
    // ferror(FILE* stream): a NULL stream counts as failed.
    register_symbol(0x007C7284DF7A772EULL, "ferror", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            FILE* fp = resolve_guest_stream(ctx.rdi);
            ctx.rax = fp ? static_cast<u64>(std::ferror(fp) != 0) : 1;
        });

    // --- C++ exception ABI: no unwinder is emulated. Any throw terminates. ---
    // A silent return would let the guest keep running on a half-built exception
    // object, so each entry names itself and the caller RIP is printed by
    // guest_abort. Consistent with abort() above.
    register_symbol(0xCEBD3DE04437F56CULL, "__cxa_pure_virtual", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            guest_abort("__cxa_pure_virtual: pure virtual method called", ctx, mem);
        });
    register_symbol(0x71F017BABBDF979AULL, "__cxa_allocate_exception", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            guest_abort("__cxa_allocate_exception: C++ exception raised, unwinder not emulated", ctx, mem);
        });
    register_symbol(0xBE4BAE2DF8674992ULL, "__cxa_throw", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            guest_abort("__cxa_throw: C++ exception thrown, unwinder not emulated", ctx, mem);
        });
    register_symbol(0x64BF455789895F1AULL, "__cxa_rethrow", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            guest_abort("__cxa_rethrow: C++ exception rethrown, unwinder not emulated", ctx, mem);
        });
    register_symbol(0xBAC29BBAFCB68508ULL, "__cxa_call_unexpected", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            guest_abort("__cxa_call_unexpected: exception specification violated", ctx, mem);
        });
    register_symbol(0x7F5CF02778C02369ULL, "_Unwind_Resume", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            guest_abort("_Unwind_Resume: resuming an unwind, unwinder not emulated", ctx, mem);
        });
    // __dynamic_cast(const void* sub, const __class_type_info* src,
    //                const __class_type_info* dst, ptrdiff_t src2dst) -- Itanium
    // C++ ABI. Unregistered, it returned NULL, so every dynamic_cast failed
    // (Dragon Ball Xenoverse 2's movie thread hits it right after opening
    // movie.cpk). The type_info vtables are data imports, so each type_info's
    // vptr is the trampoline of __class_type_info / __si_class_type_info /
    // __vmi_class_type_info plus 16; that tells the walk how to read bases.
    register_symbol(0x84C01EF93592F664ULL, "__dynamic_cast", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sub = ctx.rdi, dst = ctx.rdx;
            const i64 src2dst = static_cast<i64>(ctx.rcx);
            ctx.rax = 0;
            DynamicLinker* linker = ps4::active_linker();
            if (sub == 0 || dst == 0 || !linker) return;
            static GuestAddr s_si = 0, s_vmi = 0;
            if (!s_si) {
                s_si = linker->get_or_create_trampoline(0xA59F565DC0A53CEFULL) + 16;  // __si_class_type_info
                s_vmi = linker->get_or_create_trampoline(0xF41C9131DA3BCB08ULL) + 16; // __vmi_class_type_info
            }
            auto rd = [&](GuestAddr a) { u64 v = 0; mem.read_value(a, v); return v; };
            auto name_of = [&](GuestAddr ti) {
                std::string n;
                const GuestAddr p = rd(ti + 8);
                for (GuestSize i = 0; p && i < 512; ++i) {
                    char c = 0;
                    if (!mem.read_value(p + i, c) || c == 0) break;
                    n.push_back(c);
                }
                return n;
            };
            const std::string dst_name = name_of(dst);
            auto same = [&](GuestAddr ti) { return ti == dst || (!dst_name.empty() && name_of(ti) == dst_name); };

            const GuestAddr vptr = rd(sub);
            if (vptr == 0) return;
            const i64 offset_to_top = static_cast<i64>(rd(vptr - 16));
            const GuestAddr dyn_ti = rd(vptr - 8);
            const GuestAddr whole = sub + offset_to_top;

            // Every public dst subobject of the complete object.
            std::vector<GuestAddr> hits;
            std::function<void(GuestAddr, GuestAddr, int)> walk = [&](GuestAddr ti, GuestAddr obj, int depth) {
                if (ti == 0 || depth > 32) return;
                if (same(ti)) {
                    hits.push_back(obj);
                    return;
                }
                const GuestAddr kind = rd(ti);
                if (kind == s_si) {
                    walk(rd(ti + 16), obj, depth + 1);
                } else if (kind == s_vmi) {
                    u32 count = 0;
                    mem.read_value(ti + 20, count);
                    for (u32 b = 0; b < count && b < 64; ++b) {
                        const GuestAddr base_ti = rd(ti + 24 + 16 * b);
                        const i64 offset_flags = static_cast<i64>(rd(ti + 32 + 16 * b));
                        if (!(offset_flags & 2)) continue;   // __public_mask
                        i64 off = offset_flags >> 8;
                        if (offset_flags & 1) {              // __virtual_mask: offset is in the vtable
                            const GuestAddr obj_vptr = rd(obj);
                            off = static_cast<i64>(rd(obj_vptr + off));
                        }
                        walk(base_ti, obj + off, depth + 1);
                    }
                }
            };
            walk(dyn_ti, whole, 0);
            if (hits.empty()) return;
            std::sort(hits.begin(), hits.end());
            hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
            if (hits.size() == 1) {
                ctx.rax = hits[0];
                return;
            }
            // Ambiguous: the static hint (dst at sub - src2dst) picks one.
            if (src2dst >= 0) {
                const GuestAddr hint = sub - static_cast<GuestAddr>(src2dst);
                if (std::find(hits.begin(), hits.end(), hint) != hits.end()) ctx.rax = hint;
            }
        });

    register_symbol(0xA98867A1EBDDF5B2ULL, "_ZSt9terminatev", "libstdc++",
        [](CpuContext& ctx, MemoryManager& mem) {
            guest_abort("std::terminate() called", ctx, mem);
        });
    register_symbol(0x793D94B264DEC1B5ULL, "_ZSt11_Xbad_allocv", "libstdc++",
        [](CpuContext& ctx, MemoryManager& mem) {
            guest_abort("std::bad_alloc thrown (std::_Xbad_alloc)", ctx, mem);
        });
    register_symbol(0xA33300AF6F01C126ULL, "_ZSt14_Xout_of_rangePKc", "libstdc++",
        [](CpuContext& ctx, MemoryManager& mem) {
            std::string msg;
            read_guest_cstring(mem, ctx.rdi, msg);
            guest_abort("std::out_of_range thrown: " + msg, ctx, mem);
        });

    // Some handlers above were registered under NIDs that are not the real
    // hash of their name (the old values stay: sceKernelExitProcess's is used
    // internally). Expose each under its real NID too, or games importing it
    // fall through to the unresolved stub (sceAudioOutOutput returning at once
    // let FMOD's audio threads spin flat out).
    static constexpr struct { Nid wrong, real; } kRealNids[] = {
        {0x6424D00705E3BEBEULL, 0x16704A1E9FA0AF4EULL}, // sceKernelExitProcess
        {0xE19F218ECFDE29A6ULL, 0x438AD12F7E0211E1ULL}, // sinf
        {0xD9C650B0145DB17CULL, 0xFCFE8534CCE4D8A7ULL}, // cosf
        {0xC37A8C8C7F0ED898ULL, 0x107FF1EF5DC0F7D7ULL}, // atan2f
        {0xE1D5176E5161A4FEULL, 0x7A436FB13DB6AEC6ULL}, // sceAudioOutOpen
        {0xEAA1CDA949437E55ULL, 0xB35FFFB84F66045CULL}, // sceAudioOutClose
        {0x69661413865EE60CULL, 0x40E42D6DE0EAB13EULL}, // sceAudioOutOutput
        {0x8F596D3C1F4FA286ULL, 0x6FEB8057CF489711ULL}, // sceAudioOutSetVolume
        {0xD9BDDE84CE9A1E02ULL, 0x6FAE042D5030EB68ULL}, // sceAudioOutGetPortInfo
        {0x948296238BC4F301ULL, 0xDF61D0010770336AULL}, // sceSaveDataMount
        {0x32791B317075AB6EULL, 0x04C47817F51E9371ULL}, // sceSaveDataUmount
        {0x948296238BC4F301ULL, 0xD33E393C81FE48D2ULL}, // sceSaveDataMount (0z45PIH+SNI)
    };
    for (const auto& n : kRealNids) {
        if (m_symbols_by_nid.count(n.real)) continue;
        if (const HleSymbol* s = find_by_nid(n.wrong)) {
            const HleSymbol copy = *s;
            register_symbol(n.real, copy.name, copy.module_name, copy.handler);
            m_nids_by_name[copy.name] = n.wrong; // keep name lookups on the original
        }
    }
}

// =========================================================================
// libSceSaveData: save directories backed by host folders
// =========================================================================
//
// A mount binds "/savedataN" (N = slot 0..15, reused after unmount) in the
// VFS to <sandbox>/savedata/<user id>/<title id>/<dir name>, so the guest's
// sceKernelOpen/Write/Read on /savedata0/... hit real files and the save
// survives the process. The title id comes from INSTALL_DIR_SAVEDATA (else
// TITLE_ID) in /app0/sce_sys/param.sfo. Each save keeps its SceSaveDataParam
// as raw bytes in sce_sys/param.bin and its icon in sce_sys/icon0.png; the
// guest sees sce_sys/ inside the mount as it does on hardware. Save-data
// memory (sceSaveData*SaveDataMemory*) is a flat file in the title folder:
// sce_sdmemory[<slot>]/memory.dat. Directory names starting with "sce_" are
// never reported by sceSaveDataDirNameSearch.
//
// Behaviour follows the SDK and shadPS4's save_data library: CREATE fails
// with EXISTS when the save exists, CREATE2 creates only when missing and
// reports MOUNT_STATUS_CREATED, a mount without either fails with NOT_FOUND
// when the save is missing, and mounting the same save twice is BUSY.
// Calls made before sceSaveDataInitialize* are accepted, not rejected.

namespace {

// First call of each function only: some are polled every frame
// (sceSaveDataGetEventResult) and the log should stay readable.
void hle_log_once(const char* lib, const char* fn, const std::string& detail = {}) {
    static std::mutex s_mtx;
    static std::unordered_set<std::string> s_seen;
    {
        std::lock_guard lock(s_mtx);
        if (!s_seen.insert(fn).second) return;
    }
    std::fprintf(stderr, "[%s] %s%s%s\n", lib, fn, detail.empty() ? "" : " ", detail.c_str());
}

template <typename T>
bool read_guest(const MemoryManager& mem, GuestAddr addr, T& out) {
    return addr > 0x1000 && mem.is_valid_address(addr, sizeof(T)) && mem.read_bytes(addr, &out, sizeof(T));
}

template <typename T>
bool write_guest(MemoryManager& mem, GuestAddr addr, const T& value) {
    return addr > 0x1000 && mem.is_valid_address(addr, sizeof(T)) && mem.write_bytes(addr, &value, sizeof(T));
}

// A fixed-size guest char array that may lack its terminator.
template <usize N>
std::string fixed_string(const char (&data)[N]) {
    return std::string(data, strnlen(data, N));
}

template <usize N>
void copy_fixed_string(char (&dst)[N], std::string_view src) {
    std::memset(dst, 0, N);
    std::memcpy(dst, src.data(), std::min(src.size(), N - 1));
}

// Save directory and title names become host path components.
bool valid_save_name(std::string_view s, usize max_len) {
    if (s.empty() || s.size() > max_len || s[0] == '.') return false;
    return std::all_of(s.begin(), s.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.';
    });
}

// One string value from a PSF (param.sfo) file, or "" when absent.
std::string read_sfo_string(const std::filesystem::path& path, std::string_view key) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    const std::vector<char> sfo((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u32_at = [&](usize off) -> u32 {
        u32 v = 0;
        if (off + 4 <= sfo.size()) std::memcpy(&v, sfo.data() + off, 4);
        return v;
    };
    auto u16_at = [&](usize off) -> u16 {
        u16 v = 0;
        if (off + 2 <= sfo.size()) std::memcpy(&v, sfo.data() + off, 2);
        return v;
    };
    if (sfo.size() < 0x14 || u32_at(0) != 0x46535000) return {}; // "\0PSF"
    const u32 key_table = u32_at(0x08), data_table = u32_at(0x0C), count = u32_at(0x10);
    for (u32 i = 0; i < count; ++i) {
        const usize entry = 0x14 + usize(i) * 16;
        if (entry + 16 > sfo.size()) break;
        const usize key_off = usize(key_table) + u16_at(entry);
        const u16 fmt = u16_at(entry + 2);
        const u32 len = u32_at(entry + 4);
        const usize data_off = usize(data_table) + u32_at(entry + 12);
        if (key_off >= sfo.size() || data_off + len > sfo.size()) continue;
        const char* k = sfo.data() + key_off;
        if (std::string_view(k, strnlen(k, sfo.size() - key_off)) != key || fmt == 0x0404) continue;
        const char* v = sfo.data() + data_off;
        return std::string(v, strnlen(v, len));
    }
    return {};
}

constexpr usize kSaveDataSlots = 16;

struct SaveDataSlot {
    bool used{false};
    std::string mount_point; // "/savedataN"
    std::filesystem::path host_dir;
    bool read_only{false};
    u64 blocks{0};
};

struct SaveDataState {
    std::mutex mtx;
    std::array<SaveDataSlot, kSaveDataSlots> slots;
    std::string title_id;
    std::unordered_map<u32, std::filesystem::path> memory_files; // save-memory slot -> memory.dat
};

SaveDataState& save_data_state() {
    static SaveDataState s;
    return s;
}

// Caller holds save_data_state().mtx.
const std::string& own_title_id() {
    std::string& id = save_data_state().title_id;
    if (id.empty()) {
        if (auto sfo = get_vfs().resolve_path("/app0/sce_sys/param.sfo")) {
            id = read_sfo_string(*sfo, "INSTALL_DIR_SAVEDATA");
            if (id.empty()) id = read_sfo_string(*sfo, "TITLE_ID");
        }
        if (!valid_save_name(id, 9)) id = "CUSA00000";
    }
    return id;
}

std::filesystem::path title_save_dir(i32 user_id, const std::string& title_id) {
    return get_vfs().sandbox_root() / "savedata" / std::to_string(user_id) / title_id;
}

// Title id from an optional guest SceSaveDataTitleId*; false if malformed.
bool guest_title_id(const MemoryManager& mem, GuestAddr p, std::string& out) {
    if (p == 0) {
        out = own_title_id();
        return true;
    }
    SceSaveDataTitleId t{};
    if (!read_guest(mem, p, t)) return false;
    out = fixed_string(t.data);
    return valid_save_name(out, 9);
}

bool guest_dir_name(const MemoryManager& mem, GuestAddr p, std::string& out) {
    SceSaveDataDirName d{};
    if (!read_guest(mem, p, d)) return false;
    out = fixed_string(d.data);
    return valid_save_name(out, 31);
}

// Caller holds the lock. Null when the mount point is not mounted.
SaveDataSlot* find_mounted(const MemoryManager& mem, GuestAddr mount_point_ptr) {
    SceSaveDataMountPoint mp{};
    if (!read_guest(mem, mount_point_ptr, mp)) return nullptr;
    const std::string name = fixed_string(mp.data);
    for (auto& s : save_data_state().slots) {
        if (s.used && s.mount_point == name) return &s;
    }
    return nullptr;
}

u64 directory_bytes(const std::filesystem::path& dir) {
    std::error_code ec;
    u64 total = 0;
    for (auto it = std::filesystem::recursive_directory_iterator(dir, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec)) total += it->file_size(ec);
    }
    return total;
}

i64 last_write_unix(const std::filesystem::path& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 ? static_cast<i64>(st.st_mtime) : 0;
}

// The stored param of a save directory (zeros, apart from mtime, if never set).
SceSaveDataParam load_param(const std::filesystem::path& dir) {
    SceSaveDataParam param{};
    const auto file = dir / "sce_sys" / "param.bin";
    std::ifstream f(file, std::ios::binary);
    if (f) f.read(reinterpret_cast<char*>(&param), sizeof(param));
    param.mtime = last_write_unix(std::filesystem::exists(file) ? file : dir);
    return param;
}

bool store_param(const std::filesystem::path& dir, const SceSaveDataParam& param) {
    std::error_code ec;
    std::filesystem::create_directories(dir / "sce_sys", ec);
    std::ofstream f(dir / "sce_sys" / "param.bin", std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&param), sizeof(param));
    return static_cast<bool>(f);
}

// Copies a guest SceSaveDataIcon's image to <dir>/sce_sys/icon0.png.
bool store_icon(MemoryManager& mem, const std::filesystem::path& dir, const SceSaveDataIcon& icon) {
    const u64 n = std::min(icon.buf_size, icon.data_size ? icon.data_size : icon.buf_size);
    if (icon.buf <= 0x1000 || n > (64ull << 20) || !mem.is_valid_address(icon.buf, n)) return false;
    std::vector<char> bytes(n);
    mem.read_bytes(icon.buf, bytes.data(), n);
    std::error_code ec;
    std::filesystem::create_directories(dir / "sce_sys", ec);
    std::ofstream f(dir / "sce_sys" / "icon0.png", std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

u32 save_data_mount(MemoryManager& mem, i32 user_id, GuestAddr title_ptr, GuestAddr dir_ptr, u64 blocks,
                    u32 mode, GuestAddr result_ptr) {
    if (user_id < 0) return SCE_SAVE_DATA_ERROR_INVALID_LOGIN_USER;
    if (result_ptr <= 0x1000 || !mem.is_valid_address(result_ptr, sizeof(SceSaveDataMountResult)))
        return SCE_SAVE_DATA_ERROR_PARAMETER;
    auto& st = save_data_state();
    std::lock_guard lock(st.mtx);
    std::string title, dir;
    if (!guest_title_id(mem, title_ptr, title) || !guest_dir_name(mem, dir_ptr, dir))
        return SCE_SAVE_DATA_ERROR_PARAMETER;
    const auto host_dir = title_save_dir(user_id, title) / dir;

    SaveDataSlot* slot = nullptr;
    usize slot_index = 0;
    for (usize i = 0; i < st.slots.size(); ++i) {
        if (st.slots[i].used) {
            if (st.slots[i].host_dir == host_dir) return SCE_SAVE_DATA_ERROR_BUSY;
        } else if (!slot) {
            slot = &st.slots[i];
            slot_index = i;
        }
    }
    if (!slot) return SCE_SAVE_DATA_ERROR_MOUNT_FULL;

    std::error_code ec;
    const bool exists = std::filesystem::is_directory(host_dir, ec);
    const bool create = mode & SCE_SAVE_DATA_MOUNT_MODE_CREATE;
    const bool create2 = mode & SCE_SAVE_DATA_MOUNT_MODE_CREATE2;
    if (exists && create) return SCE_SAVE_DATA_ERROR_EXISTS;
    if (!exists && !create && !create2) return SCE_SAVE_DATA_ERROR_NOT_FOUND;
    if (!exists && !std::filesystem::create_directories(host_dir / "sce_sys", ec))
        return SCE_SAVE_DATA_ERROR_INTERNAL;

    *slot = SaveDataSlot{
        .used = true,
        .mount_point = "/savedata" + std::to_string(slot_index),
        .host_dir = host_dir,
        .read_only = (mode & SCE_SAVE_DATA_MOUNT_MODE_RDONLY) != 0,
        .blocks = blocks ? std::min(blocks, SCE_SAVE_DATA_BLOCKS_MAX) : SCE_SAVE_DATA_BLOCKS_MAX,
    };
    get_vfs().mount(slot->mount_point, host_dir, slot->read_only);

    SceSaveDataMountResult res{};
    copy_fixed_string(res.mount_point.data, slot->mount_point);
    res.mount_status = (!exists && create2) ? SCE_SAVE_DATA_MOUNT_STATUS_CREATED : SCE_SAVE_DATA_MOUNT_STATUS_NOTHING;
    mem.write_bytes(result_ptr, &res, sizeof(res));
    hle_log_once("SaveData", "mount", slot->mount_point + " -> " + host_dir.string());
    return SCE_OK;
}

// ---- save-data memory -----------------------------------------------------

std::filesystem::path save_memory_dir(i32 user_id, u32 slot_id) {
    std::lock_guard lock(save_data_state().mtx);
    const std::string name = slot_id == 0 ? "sce_sdmemory" : "sce_sdmemory" + std::to_string(slot_id);
    return title_save_dir(user_id, own_title_id()) / name;
}

std::optional<std::filesystem::path> save_memory_file(u32 slot_id) {
    auto& st = save_data_state();
    std::lock_guard lock(st.mtx);
    auto it = st.memory_files.find(slot_id);
    if (it == st.memory_files.end()) return std::nullopt;
    return it->second;
}

// Creates or opens the slot's memory.dat, at least memory_size bytes long.
// Returns the size it had before (0 when just created), or nullopt on failure.
std::optional<u64> setup_save_memory(i32 user_id, u32 slot_id, u64 memory_size) {
    const auto dir = save_memory_dir(user_id, slot_id);
    const auto file = dir / "memory.dat";
    std::error_code ec;
    std::filesystem::create_directories(dir / "sce_sys", ec);
    const u64 existed = std::filesystem::exists(file, ec) ? std::filesystem::file_size(file, ec) : 0;
    if (!std::filesystem::exists(file, ec)) std::ofstream(file, std::ios::binary);
    if (existed < memory_size) std::filesystem::resize_file(file, memory_size, ec);
    if (ec) return std::nullopt;
    auto& st = save_data_state();
    std::lock_guard lock(st.mtx);
    st.memory_files[slot_id] = file;
    return existed;
}

// Copies between memory.dat and guest memory; reads past the end give zeros.
bool save_memory_io(MemoryManager& mem, const std::filesystem::path& file, const SceSaveDataMemoryData& d,
                    bool write) {
    if (d.buf_size == 0) return true;
    if (d.buf <= 0x1000 || d.offset < 0 || d.buf_size > (1ull << 30) || !mem.is_valid_address(d.buf, d.buf_size))
        return false;
    std::vector<char> bytes(d.buf_size, 0);
    if (write) {
        mem.read_bytes(d.buf, bytes.data(), bytes.size());
        std::fstream f(file, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(d.offset);
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        return static_cast<bool>(f);
    }
    std::ifstream f(file, std::ios::binary);
    f.seekg(d.offset);
    f.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return mem.write_bytes(d.buf, bytes.data(), bytes.size());
}

u32 save_memory_get(MemoryManager& mem, u32 slot_id, GuestAddr data_ptr, GuestAddr param_ptr, GuestAddr icon_ptr) {
    const auto file = save_memory_file(slot_id);
    if (!file) return SCE_SAVE_DATA_ERROR_MEMORY_NOT_READY;
    SceSaveDataMemoryData d{};
    if (data_ptr && (!read_guest(mem, data_ptr, d) || !save_memory_io(mem, *file, d, false)))
        return SCE_SAVE_DATA_ERROR_PARAMETER;
    const auto dir = file->parent_path();
    if (param_ptr && !write_guest(mem, param_ptr, load_param(dir))) return SCE_SAVE_DATA_ERROR_PARAMETER;
    SceSaveDataIcon icon{};
    if (icon_ptr && read_guest(mem, icon_ptr, icon)) {
        std::ifstream f(dir / "sce_sys" / "icon0.png", std::ios::binary);
        const std::vector<char> png((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const u64 n = std::min<u64>(icon.buf_size, png.size());
        if (n && icon.buf > 0x1000 && mem.is_valid_address(icon.buf, n)) mem.write_bytes(icon.buf, png.data(), n);
        icon.data_size = n;
        write_guest(mem, icon_ptr, icon);
    }
    return SCE_OK;
}

u32 save_memory_set(MemoryManager& mem, u32 slot_id, GuestAddr data_ptr, u32 data_num, GuestAddr param_ptr,
                    GuestAddr icon_ptr) {
    const auto file = save_memory_file(slot_id);
    if (!file) return SCE_SAVE_DATA_ERROR_MEMORY_NOT_READY;
    for (u32 i = 0; data_ptr && i < std::max(data_num, 1u); ++i) {
        SceSaveDataMemoryData d{};
        if (!read_guest(mem, data_ptr + i * sizeof(d), d) || !save_memory_io(mem, *file, d, true))
            return SCE_SAVE_DATA_ERROR_PARAMETER;
    }
    SceSaveDataParam param{};
    if (param_ptr && read_guest(mem, param_ptr, param)) store_param(file->parent_path(), param);
    SceSaveDataIcon icon{};
    if (icon_ptr && read_guest(mem, icon_ptr, icon)) store_icon(mem, file->parent_path(), icon);
    return SCE_OK;
}

// '%' matches any run of characters, '_' exactly one (case-insensitive).
bool save_name_matches(std::string_view s, std::string_view pat) {
    if (pat.empty()) return s.empty();
    if (pat[0] == '%') {
        for (usize i = 0; i <= s.size(); ++i)
            if (save_name_matches(s.substr(i), pat.substr(1))) return true;
        return false;
    }
    if (s.empty()) return false;
    const bool same = pat[0] == '_' || std::tolower(static_cast<unsigned char>(pat[0])) ==
                                           std::tolower(static_cast<unsigned char>(s[0]));
    return same && save_name_matches(s.substr(1), pat.substr(1));
}

// =========================================================================
// libSceNpTrophy: in-memory contexts, handles and unlocks
// =========================================================================
//
// There is no trophy pack to read, so every title has an empty trophy set:
// contexts and handles behave like the real ones (ids from 1, 8 contexts and
// 4 handles at most, one context per user and service label), unlocks are
// remembered for the process, and the info queries report what was unlocked
// so far with no names or grades. Nothing here blocks or shows UI.

struct TrophyContext {
    bool used{false};
    i32 user_id{0};
    u32 service_label{0};
    bool registered{false};
};

struct TrophyState {
    std::mutex mtx;
    std::array<TrophyContext, 8> contexts;
    std::array<bool, 4> handles{};
    std::map<i32, std::array<u64, SCE_NP_TROPHY_NUM_MAX>> unlock_ticks; // user -> RTC tick, 0 = locked
};

TrophyState& trophy_state() {
    static TrophyState s;
    return s;
}

// Caller holds the lock. Checks context then handle, as the library does.
u32 trophy_lookup(i32 context, i32 handle, TrophyContext** out) {
    auto& st = trophy_state();
    if (context < 1 || context > static_cast<i32>(st.contexts.size()) || !st.contexts[context - 1].used)
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    if (handle < 1 || handle > static_cast<i32>(st.handles.size()) || !st.handles[handle - 1])
        return SCE_NP_TROPHY_ERROR_INVALID_HANDLE;
    *out = &st.contexts[context - 1];
    return SCE_OK;
}

// SceRtcTick now: microseconds since 0001-01-01.
u64 rtc_tick_now() {
    constexpr u64 kUnixEpochTicks = 62135596800ull * 1000000ull;
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return kUnixEpochTicks + static_cast<u64>(us);
}

struct TrophyCounts {
    u32 num_trophies{0}; // highest unlocked id + 1: the set has no other members we know of
    u32 unlocked{0};
};

TrophyCounts trophy_counts(const TrophyContext& c) {
    TrophyCounts counts;
    auto it = trophy_state().unlock_ticks.find(c.user_id);
    if (it == trophy_state().unlock_ticks.end()) return counts;
    for (i32 id = 0; id < SCE_NP_TROPHY_NUM_MAX; ++id) {
        if (it->second[id]) {
            counts.num_trophies = static_cast<u32>(id) + 1;
            ++counts.unlocked;
        }
    }
    return counts;
}

} // namespace

static void register_save_data_and_trophy(NidRegistry& reg) {
    // ---- libSceSaveData: lifecycle ---------------------------------------
    // sceSaveDataInitialize/2/3(const SceSaveDataInitParams*) and Terminate().
    // Spelled out one per line so scripts/verify_nids.py audits each NID.
    auto succeed = [](const char* name) {
        return [name](CpuContext& ctx, MemoryManager&) {
            hle_log_once("SaveData", name);
            ctx.rax = SCE_OK;
        };
    };
    reg.register_symbol(0x664661B2408F5C5CULL, "sceSaveDataInitialize", "libSceSaveData", succeed("sceSaveDataInitialize"));
    reg.register_symbol(0x9753660DE0E93465ULL, "sceSaveDataInitialize2", "libSceSaveData", succeed("sceSaveDataInitialize2"));
    reg.register_symbol(0x4F2C2B14A0A82C66ULL, "sceSaveDataInitialize3", "libSceSaveData", succeed("sceSaveDataInitialize3"));
    reg.register_symbol(0xC8A0F2F12E722C0DULL, "sceSaveDataTerminate", "libSceSaveData", succeed("sceSaveDataTerminate"));

    // ---- libSceSaveData: mount / unmount ---------------------------------
    // sceSaveDataMount(const SceSaveDataMount*, SceSaveDataMountResult*)
    reg.register_symbol(0xDF61D0010770336AULL, "sceSaveDataMount", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataMount");
            SceSaveDataMount m{};
            if (!read_guest(mem, ctx.rdi, m)) {
                ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                return;
            }
            ctx.rax = save_data_mount(mem, m.user_id, m.title_id, m.dir_name, m.blocks, m.mount_mode, ctx.rsi);
        });

    // sceSaveDataMount2(const SceSaveDataMount2*, SceSaveDataMountResult*)
    reg.register_symbol(0xD33E393C81FE48D2ULL, "sceSaveDataMount2", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataMount2");
            SceSaveDataMount2 m{};
            if (!read_guest(mem, ctx.rdi, m)) {
                ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                return;
            }
            ctx.rax = save_data_mount(mem, m.user_id, 0, m.dir_name, m.blocks, m.mount_mode, ctx.rsi);
        });

    // sceSaveDataUmount(const SceSaveDataMountPoint*)
    reg.register_symbol(0x04C47817F51E9371ULL, "sceSaveDataUmount", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataUmount");
            std::lock_guard lock(save_data_state().mtx);
            SaveDataSlot* slot = find_mounted(mem, ctx.rdi);
            if (!slot) {
                ctx.rax = ctx.rdi ? SCE_SAVE_DATA_ERROR_NOT_FOUND : SCE_SAVE_DATA_ERROR_PARAMETER;
                return;
            }
            get_vfs().unmount(slot->mount_point);
            *slot = SaveDataSlot{};
            ctx.rax = SCE_OK;
        });

    // sceSaveDataGetMountInfo(const SceSaveDataMountPoint*, SceSaveDataMountInfo*)
    reg.register_symbol(0xEB9547D1069ACFABULL, "sceSaveDataGetMountInfo", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataGetMountInfo");
            std::lock_guard lock(save_data_state().mtx);
            SaveDataSlot* slot = find_mounted(mem, ctx.rdi);
            if (!slot) {
                ctx.rax = SCE_SAVE_DATA_ERROR_NOT_MOUNTED;
                return;
            }
            const u64 used = (directory_bytes(slot->host_dir) + SCE_SAVE_DATA_BLOCK_SIZE - 1) / SCE_SAVE_DATA_BLOCK_SIZE;
            SceSaveDataMountInfo info{};
            info.blocks = slot->blocks;
            info.free_blocks = slot->blocks > used ? slot->blocks - used : 0;
            ctx.rax = write_guest(mem, ctx.rsi, info) ? SCE_OK : SCE_SAVE_DATA_ERROR_PARAMETER;
        });

    // ---- libSceSaveData: params and icon ---------------------------------
    // sceSaveDataGetParam(const SceSaveDataMountPoint*, SceSaveDataParamType,
    //                     void* buf, size_t bufSize, size_t* gotSize)
    reg.register_symbol(0x5E0BD2B88767325CULL, "sceSaveDataGetParam", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataGetParam");
            const u32 type = static_cast<u32>(ctx.rsi);
            const GuestAddr buf = ctx.rdx, got_ptr = ctx.r8;
            const u64 buf_size = ctx.rcx;
            std::lock_guard lock(save_data_state().mtx);
            SaveDataSlot* slot = find_mounted(mem, ctx.rdi);
            if (!slot) {
                ctx.rax = SCE_SAVE_DATA_ERROR_NOT_MOUNTED;
                return;
            }
            const SceSaveDataParam p = load_param(slot->host_dir);
            std::string text;
            const void* src = nullptr;
            u64 n = 0;
            switch (type) {
            case SCE_SAVE_DATA_PARAM_TYPE_ALL: src = &p; n = sizeof(p); break;
            case SCE_SAVE_DATA_PARAM_TYPE_TITLE: text = fixed_string(p.title); break;
            case SCE_SAVE_DATA_PARAM_TYPE_SUB_TITLE: text = fixed_string(p.sub_title); break;
            case SCE_SAVE_DATA_PARAM_TYPE_DETAIL: text = fixed_string(p.detail); break;
            case SCE_SAVE_DATA_PARAM_TYPE_USER_PARAM: src = &p.user_param; n = sizeof(p.user_param); break;
            case SCE_SAVE_DATA_PARAM_TYPE_MTIME: src = &p.mtime; n = sizeof(p.mtime); break;
            default: ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER; return;
            }
            if (!src) { // strings: truncated to the buffer, always terminated
                if (buf_size == 0) {
                    ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                    return;
                }
                text.resize(std::min<u64>(text.size(), buf_size - 1));
                src = text.c_str();
                n = text.size() + 1;
            }
            if (buf_size < n || buf <= 0x1000 || !mem.is_valid_address(buf, n)) {
                ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                return;
            }
            mem.write_bytes(buf, src, n);
            if (got_ptr) write_guest(mem, got_ptr, n);
            ctx.rax = SCE_OK;
        });

    // sceSaveDataSetParam(const SceSaveDataMountPoint*, SceSaveDataParamType,
    //                     const void* buf, size_t bufSize)
    reg.register_symbol(0xF39CEE97FFDE197BULL, "sceSaveDataSetParam", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataSetParam");
            const u32 type = static_cast<u32>(ctx.rsi);
            const GuestAddr buf = ctx.rdx;
            std::lock_guard lock(save_data_state().mtx);
            SaveDataSlot* slot = find_mounted(mem, ctx.rdi);
            if (!slot) {
                ctx.rax = SCE_SAVE_DATA_ERROR_NOT_MOUNTED;
                return;
            }
            if (slot->read_only) {
                ctx.rax = SCE_SAVE_DATA_ERROR_BAD_MOUNTED;
                return;
            }
            SceSaveDataParam p = load_param(slot->host_dir);
            auto read_text = [&](auto& field) {
                char tmp[sizeof(field)] = {};
                const u64 n = std::min<u64>(sizeof(tmp) - 1, ctx.rcx ? ctx.rcx : sizeof(tmp) - 1);
                if (buf <= 0x1000 || !mem.is_valid_address(buf, n)) return false;
                mem.read_bytes(buf, tmp, n);
                copy_fixed_string(field, std::string_view(tmp, strnlen(tmp, n)));
                return true;
            };
            bool ok = false;
            switch (type) {
            case SCE_SAVE_DATA_PARAM_TYPE_ALL: ok = read_guest(mem, buf, p); break;
            case SCE_SAVE_DATA_PARAM_TYPE_TITLE: ok = read_text(p.title); break;
            case SCE_SAVE_DATA_PARAM_TYPE_SUB_TITLE: ok = read_text(p.sub_title); break;
            case SCE_SAVE_DATA_PARAM_TYPE_DETAIL: ok = read_text(p.detail); break;
            case SCE_SAVE_DATA_PARAM_TYPE_USER_PARAM: ok = read_guest(mem, buf, p.user_param); break;
            default: break; // MTIME is not settable
            }
            if (!ok) {
                ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                return;
            }
            ctx.rax = store_param(slot->host_dir, p) ? SCE_OK : SCE_SAVE_DATA_ERROR_INTERNAL;
        });

    // sceSaveDataSaveIcon(const SceSaveDataMountPoint*, const SceSaveDataIcon*)
    reg.register_symbol(0x73CF18CB9E0CC74CULL, "sceSaveDataSaveIcon", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataSaveIcon");
            std::lock_guard lock(save_data_state().mtx);
            SaveDataSlot* slot = find_mounted(mem, ctx.rdi);
            SceSaveDataIcon icon{};
            if (!slot) {
                ctx.rax = SCE_SAVE_DATA_ERROR_NOT_MOUNTED;
            } else if (slot->read_only) {
                ctx.rax = SCE_SAVE_DATA_ERROR_BAD_MOUNTED;
            } else if (!read_guest(mem, ctx.rsi, icon) || !store_icon(mem, slot->host_dir, icon)) {
                ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
            } else {
                ctx.rax = SCE_OK;
            }
        });

    // ---- libSceSaveData: delete / search / events ------------------------
    // sceSaveDataDelete(const SceSaveDataDelete*)
    reg.register_symbol(0x4B51A478F235EF34ULL, "sceSaveDataDelete", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataDelete");
            SceSaveDataDelete d{};
            std::lock_guard lock(save_data_state().mtx);
            std::string title, dir;
            if (!read_guest(mem, ctx.rdi, d) || !guest_title_id(mem, d.title_id, title) ||
                !guest_dir_name(mem, d.dir_name, dir)) {
                ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                return;
            }
            const auto host_dir = title_save_dir(d.user_id, title) / dir;
            for (const auto& s : save_data_state().slots) {
                if (s.used && s.host_dir == host_dir) {
                    ctx.rax = SCE_SAVE_DATA_ERROR_BUSY;
                    return;
                }
            }
            std::error_code ec;
            std::filesystem::remove_all(host_dir, ec);
            ctx.rax = ec ? SCE_SAVE_DATA_ERROR_INTERNAL : SCE_OK;
        });

    // sceSaveDataDirNameSearch(const SceSaveDataDirNameSearchCond*,
    //                          SceSaveDataDirNameSearchResult*)
    reg.register_symbol(0x7722219D7ABFD123ULL, "sceSaveDataDirNameSearch", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataDirNameSearch");
            SceSaveDataDirNameSearchCond cond{};
            SceSaveDataDirNameSearchResult res{};
            std::lock_guard lock(save_data_state().mtx);
            std::string title, pattern;
            if (!read_guest(mem, ctx.rdi, cond) || !read_guest(mem, ctx.rsi, res) ||
                !guest_title_id(mem, cond.title_id, title) || cond.key > SCE_SAVE_DATA_SORT_KEY_FREE_BLOCKS ||
                cond.key == 4 || cond.order > SCE_SAVE_DATA_SORT_ORDER_DESCENT) {
                ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                return;
            }
            SceSaveDataDirName pat{};
            if (cond.dir_name && read_guest(mem, cond.dir_name, pat)) pattern = fixed_string(pat.data);

            struct Entry {
                std::string name;
                SceSaveDataParam param;
                u64 blocks, free_blocks;
            };
            std::vector<Entry> found;
            std::error_code ec;
            for (const auto& e : std::filesystem::directory_iterator(title_save_dir(cond.user_id, title), ec)) {
                const std::string name = e.path().filename().string();
                if (!e.is_directory(ec) || name.starts_with("sce_") || !valid_save_name(name, 31)) continue;
                if (!pattern.empty() && !save_name_matches(name, pattern)) continue;
                const u64 used = (directory_bytes(e.path()) + SCE_SAVE_DATA_BLOCK_SIZE - 1) / SCE_SAVE_DATA_BLOCK_SIZE;
                found.push_back({name, load_param(e.path()), SCE_SAVE_DATA_BLOCKS_MAX, SCE_SAVE_DATA_BLOCKS_MAX - used});
            }
            auto key = [&](const Entry& e) -> i64 {
                switch (cond.key) {
                case SCE_SAVE_DATA_SORT_KEY_USER_PARAM: return e.param.user_param;
                case SCE_SAVE_DATA_SORT_KEY_BLOCKS: return static_cast<i64>(e.blocks);
                case SCE_SAVE_DATA_SORT_KEY_MTIME: return e.param.mtime;
                case SCE_SAVE_DATA_SORT_KEY_FREE_BLOCKS: return static_cast<i64>(e.free_blocks);
                default: return 0;
                }
            };
            std::sort(found.begin(), found.end(), [](const Entry& a, const Entry& b) { return a.name < b.name; });
            if (cond.key != SCE_SAVE_DATA_SORT_KEY_DIRNAME)
                std::stable_sort(found.begin(), found.end(), [&](const Entry& a, const Entry& b) { return key(a) < key(b); });
            if (cond.order == SCE_SAVE_DATA_SORT_ORDER_DESCENT) std::reverse(found.begin(), found.end());

            const u32 n = static_cast<u32>(std::min<usize>(found.size(), res.dir_names_num));
            for (u32 i = 0; i < n; ++i) {
                SceSaveDataDirName dn{};
                copy_fixed_string(dn.data, found[i].name);
                if (!write_guest(mem, res.dir_names + i * sizeof(dn), dn)) {
                    ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                    return;
                }
                if (res.params) write_guest(mem, res.params + i * sizeof(SceSaveDataParam), found[i].param);
                if (res.infos) {
                    SceSaveDataSearchInfo info{};
                    info.blocks = found[i].blocks;
                    info.free_blocks = found[i].free_blocks;
                    write_guest(mem, res.infos + i * sizeof(info), info);
                }
            }
            res.hit_num = static_cast<u32>(found.size());
            res.set_num = n;
            write_guest(mem, ctx.rsi, res);
            ctx.rax = SCE_OK;
        });

    // sceSaveDataGetEventResult(const SceSaveDataEventParam*, SceSaveDataEvent*)
    // Events come from backups and save-memory syncs, which are not modelled:
    // the queue is always empty.
    reg.register_symbol(0x8FCC4AB62163D126ULL, "sceSaveDataGetEventResult", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager&) {
            hle_log_once("SaveData", "sceSaveDataGetEventResult");
            ctx.rax = ctx.rsi ? SCE_SAVE_DATA_ERROR_NOT_FOUND : SCE_SAVE_DATA_ERROR_PARAMETER;
        });

    // ---- libSceSaveData: save-data memory --------------------------------
    // sceSaveDataSetupSaveDataMemory(userId, size_t memorySize, SceSaveDataParam*)
    reg.register_symbol(0xBFB00000CA342F3EULL, "sceSaveDataSetupSaveDataMemory", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataSetupSaveDataMemory");
            const auto existed = setup_save_memory(static_cast<i32>(ctx.rdi), 0, ctx.rsi);
            if (!existed) {
                ctx.rax = SCE_SAVE_DATA_ERROR_INTERNAL;
                return;
            }
            SceSaveDataParam param{};
            if (ctx.rdx && read_guest(mem, ctx.rdx, param)) store_param(save_memory_dir(static_cast<i32>(ctx.rdi), 0), param);
            ctx.rax = SCE_OK;
        });

    // sceSaveDataSetupSaveDataMemory2(const SceSaveDataMemorySetup2*,
    //                                 SceSaveDataMemorySetupResult*)
    // Called by the Flappy eboot at 0x10acf7e on a worker thread.
    reg.register_symbol(0xA10C921147E05D10ULL, "sceSaveDataSetupSaveDataMemory2", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataSetupSaveDataMemory2");
            SceSaveDataMemorySetup2 setup{};
            if (!read_guest(mem, ctx.rdi, setup)) {
                ctx.rax = SCE_SAVE_DATA_ERROR_PARAMETER;
                return;
            }
            const auto existed = setup_save_memory(setup.user_id, setup.slot_id, setup.memory_size);
            if (!existed) {
                ctx.rax = SCE_SAVE_DATA_ERROR_INTERNAL;
                return;
            }
            if (*existed == 0) { // just created: apply the initial param and icon
                const auto dir = save_memory_dir(setup.user_id, setup.slot_id);
                SceSaveDataParam param{};
                if (setup.init_param && read_guest(mem, setup.init_param, param)) store_param(dir, param);
                SceSaveDataIcon icon{};
                if (setup.init_icon && read_guest(mem, setup.init_icon, icon)) store_icon(mem, dir, icon);
            }
            SceSaveDataMemorySetupResult result{};
            result.existed_memory_size = *existed;
            if (ctx.rsi) write_guest(mem, ctx.rsi, result);
            ctx.rax = SCE_OK;
        });

    // sceSaveDataGetSaveDataMemory(userId, void* buf, size_t bufSize, int64_t offset)
    reg.register_symbol(0xEC1B79A410BF01CAULL, "sceSaveDataGetSaveDataMemory", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataGetSaveDataMemory");
            const auto file = save_memory_file(0);
            SceSaveDataMemoryData d{};
            d.buf = ctx.rsi;
            d.buf_size = ctx.rdx;
            d.offset = static_cast<i64>(ctx.rcx);
            ctx.rax = !file ? SCE_SAVE_DATA_ERROR_MEMORY_NOT_READY
                    : save_memory_io(mem, *file, d, false) ? SCE_OK : SCE_SAVE_DATA_ERROR_PARAMETER;
        });

    // sceSaveDataGetSaveDataMemory2(SceSaveDataMemoryGet2*)
    reg.register_symbol(0x43038EEEF7A09D5FULL, "sceSaveDataGetSaveDataMemory2", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataGetSaveDataMemory2");
            SceSaveDataMemoryGet2 get{};
            ctx.rax = read_guest(mem, ctx.rdi, get) ? save_memory_get(mem, get.slot_id, get.data, get.param, get.icon)
                                                    : SCE_SAVE_DATA_ERROR_PARAMETER;
        });

    // sceSaveDataSetSaveDataMemory(userId, const void* buf, size_t bufSize, int64_t offset)
    reg.register_symbol(0x8776144735C64954ULL, "sceSaveDataSetSaveDataMemory", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataSetSaveDataMemory");
            const auto file = save_memory_file(0);
            SceSaveDataMemoryData d{};
            d.buf = ctx.rsi;
            d.buf_size = ctx.rdx;
            d.offset = static_cast<i64>(ctx.rcx);
            ctx.rax = !file ? SCE_SAVE_DATA_ERROR_MEMORY_NOT_READY
                    : save_memory_io(mem, *file, d, true) ? SCE_OK : SCE_SAVE_DATA_ERROR_PARAMETER;
        });

    // sceSaveDataSetSaveDataMemory2(const SceSaveDataMemorySet2*)
    reg.register_symbol(0x71DBB2F6FE18993EULL, "sceSaveDataSetSaveDataMemory2", "libSceSaveData",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("SaveData", "sceSaveDataSetSaveDataMemory2");
            SceSaveDataMemorySet2 set{};
            ctx.rax = read_guest(mem, ctx.rdi, set)
                    ? save_memory_set(mem, set.slot_id, set.data, set.data_num, set.param, set.icon)
                    : SCE_SAVE_DATA_ERROR_PARAMETER;
        });

    // ---- libSceNpTrophy ----------------------------------------------------
    // sceNpTrophyCreateContext(SceNpTrophyContext* out, userId, serviceLabel, u64 options)
    reg.register_symbol(0x5DB9236E86D99426ULL, "sceNpTrophyCreateContext", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("NpTrophy", "sceNpTrophyCreateContext");
            const i32 user_id = static_cast<i32>(ctx.rsi);
            const u32 label = static_cast<u32>(ctx.rdx);
            auto& st = trophy_state();
            std::lock_guard lock(st.mtx);
            if (ctx.rcx != 0 || ctx.rdi <= 0x1000 || !mem.is_valid_address(ctx.rdi, 4)) {
                ctx.rax = SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
                return;
            }
            TrophyContext* free_slot = nullptr;
            i32 id = 0;
            for (usize i = 0; i < st.contexts.size(); ++i) {
                auto& c = st.contexts[i];
                if (c.used && c.user_id == user_id && c.service_label == label) {
                    ctx.rax = SCE_NP_TROPHY_ERROR_CONTEXT_ALREADY_EXISTS;
                    return;
                }
                if (!c.used && !free_slot) {
                    free_slot = &c;
                    id = static_cast<i32>(i) + 1;
                }
            }
            if (!free_slot) {
                ctx.rax = SCE_NP_TROPHY_ERROR_CONTEXT_EXCEEDS_MAX;
                return;
            }
            *free_slot = TrophyContext{.used = true, .user_id = user_id, .service_label = label};
            mem.write_value(ctx.rdi, id);
            ctx.rax = SCE_OK;
        });

    // sceNpTrophyCreateHandle(SceNpTrophyHandle* out)
    reg.register_symbol(0xABB53AB440107FB7ULL, "sceNpTrophyCreateHandle", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("NpTrophy", "sceNpTrophyCreateHandle");
            auto& st = trophy_state();
            std::lock_guard lock(st.mtx);
            if (ctx.rdi <= 0x1000 || !mem.is_valid_address(ctx.rdi, 4)) {
                ctx.rax = SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
                return;
            }
            for (usize i = 0; i < st.handles.size(); ++i) {
                if (!st.handles[i]) {
                    st.handles[i] = true;
                    mem.write_value(ctx.rdi, static_cast<i32>(i) + 1);
                    ctx.rax = SCE_OK;
                    return;
                }
            }
            ctx.rax = SCE_NP_TROPHY_ERROR_HANDLE_EXCEEDS_MAX;
        });

    // sceNpTrophyRegisterContext(context, handle, u64 options)
    reg.register_symbol(0x4C9080C6DA3D4845ULL, "sceNpTrophyRegisterContext", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager&) {
            hle_log_once("NpTrophy", "sceNpTrophyRegisterContext");
            std::lock_guard lock(trophy_state().mtx);
            TrophyContext* c = nullptr;
            const u32 err = trophy_lookup(static_cast<i32>(ctx.rdi), static_cast<i32>(ctx.rsi), &c);
            if (err == SCE_OK) c->registered = true;
            ctx.rax = err;
        });

    // sceNpTrophyDestroyContext(context)
    reg.register_symbol(0x1355ABC1DD3B2EBFULL, "sceNpTrophyDestroyContext", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager&) {
            hle_log_once("NpTrophy", "sceNpTrophyDestroyContext");
            auto& st = trophy_state();
            std::lock_guard lock(st.mtx);
            const i32 id = static_cast<i32>(ctx.rdi);
            if (id < 1 || id > static_cast<i32>(st.contexts.size()) || !st.contexts[id - 1].used) {
                ctx.rax = SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
                return;
            }
            st.contexts[id - 1] = TrophyContext{};
            ctx.rax = SCE_OK;
        });

    // sceNpTrophyDestroyHandle(handle)
    reg.register_symbol(0x18D705E2889D6346ULL, "sceNpTrophyDestroyHandle", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager&) {
            hle_log_once("NpTrophy", "sceNpTrophyDestroyHandle");
            auto& st = trophy_state();
            std::lock_guard lock(st.mtx);
            const i32 id = static_cast<i32>(ctx.rdi);
            if (id < 1 || id > static_cast<i32>(st.handles.size()) || !st.handles[id - 1]) {
                ctx.rax = SCE_NP_TROPHY_ERROR_INVALID_HANDLE;
                return;
            }
            st.handles[id - 1] = false;
            ctx.rax = SCE_OK;
        });

    // sceNpTrophyUnlockTrophy(context, handle, SceNpTrophyId, SceNpTrophyId* platinumId)
    reg.register_symbol(0xDBCC6645415AA3AFULL, "sceNpTrophyUnlockTrophy", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 trophy = static_cast<i32>(ctx.rdx);
            hle_log_once("NpTrophy", "sceNpTrophyUnlockTrophy", "id=" + std::to_string(trophy));
            auto& st = trophy_state();
            std::lock_guard lock(st.mtx);
            TrophyContext* c = nullptr;
            u32 err = trophy_lookup(static_cast<i32>(ctx.rdi), static_cast<i32>(ctx.rsi), &c);
            if (err == SCE_OK && (trophy < 0 || trophy >= SCE_NP_TROPHY_NUM_MAX)) err = SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
            if (err == SCE_OK && (ctx.rcx <= 0x1000 || !mem.is_valid_address(ctx.rcx, 4))) err = SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
            if (err == SCE_OK && !c->registered) err = SCE_NP_TROPHY_ERROR_NOT_REGISTERED;
            if (err == SCE_OK) {
                auto& tick = st.unlock_ticks[c->user_id][trophy];
                if (tick) {
                    err = SCE_NP_TROPHY_ERROR_TROPHY_ALREADY_UNLOCKED;
                } else {
                    tick = rtc_tick_now();
                    mem.write_value(ctx.rcx, SCE_NP_TROPHY_INVALID_TROPHY_ID); // no platinum earned
                }
            }
            ctx.rax = err;
        });

    // sceNpTrophyGetTrophyUnlockState(context, handle, SceNpTrophyFlagArray*, u32* count)
    reg.register_symbol(0x2C7B9298EDD22DDFULL, "sceNpTrophyGetTrophyUnlockState", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("NpTrophy", "sceNpTrophyGetTrophyUnlockState");
            auto& st = trophy_state();
            std::lock_guard lock(st.mtx);
            TrophyContext* c = nullptr;
            u32 err = (ctx.rdx && ctx.rcx) ? SCE_OK : SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
            if (err == SCE_OK) err = trophy_lookup(static_cast<i32>(ctx.rdi), static_cast<i32>(ctx.rsi), &c);
            if (err == SCE_OK && !c->registered) err = SCE_NP_TROPHY_ERROR_NOT_REGISTERED;
            if (err != SCE_OK) {
                ctx.rax = err;
                return;
            }
            SceNpTrophyFlagArray flags{};
            if (auto it = st.unlock_ticks.find(c->user_id); it != st.unlock_ticks.end()) {
                for (i32 id = 0; id < SCE_NP_TROPHY_NUM_MAX; ++id)
                    if (it->second[id]) flags.flag_bits[id >> 5] |= 1u << (id & 31);
            }
            const u32 count = trophy_counts(*c).num_trophies;
            ctx.rax = write_guest(mem, ctx.rdx, flags) && write_guest(mem, ctx.rcx, count)
                    ? SCE_OK : SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
        });

    // sceNpTrophyGetGameInfo(context, handle, SceNpTrophyGameDetails*, SceNpTrophyGameData*)
    reg.register_symbol(0x6183F77F65B4F688ULL, "sceNpTrophyGetGameInfo", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("NpTrophy", "sceNpTrophyGetGameInfo");
            std::lock_guard lock(trophy_state().mtx);
            TrophyContext* c = nullptr;
            u32 err = trophy_lookup(static_cast<i32>(ctx.rdi), static_cast<i32>(ctx.rsi), &c);
            if (err == SCE_OK && !c->registered) err = SCE_NP_TROPHY_ERROR_NOT_REGISTERED;
            if (err != SCE_OK) {
                ctx.rax = err;
                return;
            }
            const TrophyCounts counts = trophy_counts(*c);
            SceNpTrophyGameDetails details{};
            if (ctx.rdx && read_guest(mem, ctx.rdx, details.size)) { // size is the caller's
                const u64 size = details.size;
                details = SceNpTrophyGameDetails{};
                details.size = size;
                details.num_trophies = counts.num_trophies;
                write_guest(mem, ctx.rdx, details);
            }
            SceNpTrophyGameData data{};
            if (ctx.rcx && read_guest(mem, ctx.rcx, data.size)) {
                const u64 size = data.size;
                data = SceNpTrophyGameData{};
                data.size = size;
                data.unlocked_trophies = counts.unlocked;
                data.progress_percentage = counts.num_trophies ? counts.unlocked * 100 / counts.num_trophies : 0;
                write_guest(mem, ctx.rcx, data);
            }
            ctx.rax = SCE_OK;
        });

    // sceNpTrophyGetGroupInfo(context, handle, SceNpTrophyGroupId,
    //                         SceNpTrophyGroupDetails*, SceNpTrophyGroupData*)
    reg.register_symbol(0xC1353019FB292A27ULL, "sceNpTrophyGetGroupInfo", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("NpTrophy", "sceNpTrophyGetGroupInfo");
            const i32 group = static_cast<i32>(ctx.rdx);
            std::lock_guard lock(trophy_state().mtx);
            TrophyContext* c = nullptr;
            u32 err = trophy_lookup(static_cast<i32>(ctx.rdi), static_cast<i32>(ctx.rsi), &c);
            if (err == SCE_OK && !c->registered) err = SCE_NP_TROPHY_ERROR_NOT_REGISTERED;
            if (err == SCE_OK && group < SCE_NP_TROPHY_BASE_GAME_GROUP_ID) err = SCE_NP_TROPHY_ERROR_INVALID_GROUP_ID;
            if (err != SCE_OK) {
                ctx.rax = err;
                return;
            }
            // Every unlocked trophy belongs to the base game; other groups are empty.
            const TrophyCounts counts = group == SCE_NP_TROPHY_BASE_GAME_GROUP_ID ? trophy_counts(*c) : TrophyCounts{};
            SceNpTrophyGroupDetails details{};
            if (ctx.rcx && read_guest(mem, ctx.rcx, details.size)) {
                const u64 size = details.size;
                details = SceNpTrophyGroupDetails{};
                details.size = size;
                details.group_id = group;
                details.num_trophies = counts.num_trophies;
                write_guest(mem, ctx.rcx, details);
            }
            SceNpTrophyGroupData data{};
            if (ctx.r8 && read_guest(mem, ctx.r8, data.size)) {
                const u64 size = data.size;
                data = SceNpTrophyGroupData{};
                data.size = size;
                data.group_id = group;
                data.unlocked_trophies = counts.unlocked;
                data.progress_percentage = counts.num_trophies ? counts.unlocked * 100 / counts.num_trophies : 0;
                write_guest(mem, ctx.r8, data);
            }
            ctx.rax = SCE_OK;
        });

    // sceNpTrophyGetTrophyInfo(context, handle, SceNpTrophyId,
    //                          SceNpTrophyDetails*, SceNpTrophyData*)
    reg.register_symbol(0xAAA515183810066DULL, "sceNpTrophyGetTrophyInfo", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager& mem) {
            hle_log_once("NpTrophy", "sceNpTrophyGetTrophyInfo");
            const i32 trophy = static_cast<i32>(ctx.rdx);
            auto& st = trophy_state();
            std::lock_guard lock(st.mtx);
            TrophyContext* c = nullptr;
            u32 err = trophy_lookup(static_cast<i32>(ctx.rdi), static_cast<i32>(ctx.rsi), &c);
            if (err == SCE_OK && (trophy < 0 || trophy >= SCE_NP_TROPHY_NUM_MAX)) err = SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
            if (err == SCE_OK && !c->registered) err = SCE_NP_TROPHY_ERROR_NOT_REGISTERED;
            if (err != SCE_OK) {
                ctx.rax = err;
                return;
            }
            u64 tick = 0;
            if (auto it = st.unlock_ticks.find(c->user_id); it != st.unlock_ticks.end()) tick = it->second[trophy];
            SceNpTrophyDetails details{};
            if (ctx.rcx && read_guest(mem, ctx.rcx, details.size)) {
                const u64 size = details.size;
                details = SceNpTrophyDetails{};
                details.size = size;
                details.trophy_id = trophy;
                details.group_id = SCE_NP_TROPHY_BASE_GAME_GROUP_ID;
                write_guest(mem, ctx.rcx, details);
            }
            SceNpTrophyData data{};
            if (ctx.r8 && read_guest(mem, ctx.r8, data.size)) {
                const u64 size = data.size;
                data = SceNpTrophyData{};
                data.size = size;
                data.trophy_id = trophy;
                data.unlocked = tick != 0;
                data.timestamp = tick;
                write_guest(mem, ctx.r8, data);
            }
            ctx.rax = SCE_OK;
        });

    // sceNpTrophyShowTrophyList(context, handle): no system UI to show.
    reg.register_symbol(0x77D8E974FCF97FFFULL, "sceNpTrophyShowTrophyList", "libSceNpTrophy",
        [](CpuContext& ctx, MemoryManager&) {
            hle_log_once("NpTrophy", "sceNpTrophyShowTrophyList");
            std::lock_guard lock(trophy_state().mtx);
            TrophyContext* c = nullptr;
            ctx.rax = trophy_lookup(static_cast<i32>(ctx.rdi), static_cast<i32>(ctx.rsi), &c);
        });

    // ---- libkernel: the write side of file I/O ---------------------------
    // sceKernelOpen/Read/Lseek/Close were already routed to the VFS; without
    // these a guest writing its save under /savedataN got 0 bytes written.
    auto sce_err = [](i64 neg_errno) -> u64 {
        return 0x80020000ULL | static_cast<u64>(-neg_errno & 0xFFFF);
    };
    // sceKernelWrite(int fd, const void* buf, size_t n); fds 1/2 go to the host's.
    reg.register_symbol(0xE304B37BDD8184B2ULL, "sceKernelWrite", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager& mem) {
            const i32 fd = static_cast<i32>(ctx.rdi);
            const GuestSize n = ctx.rdx;
            if (n != 0 && !mem.is_valid_address(ctx.rsi, n)) {
                ctx.rax = sce_err(-EFAULT);
                return;
            }
            if (fd == 1 || fd == 2) {
                std::fwrite(mem.to_host_ptr(ctx.rsi, n), 1, n, fd == 1 ? stdout : stderr);
                ctx.rax = n;
                return;
            }
            const i64 r = n == 0 ? 0 : get_vfs().write(fd, mem.to_host_ptr(ctx.rsi, n), n);
            ctx.rax = r < 0 ? sce_err(r) : static_cast<u64>(r);
        });
    // sceKernelMkdir(const char* path, mode_t mode)
    reg.register_symbol(0xD7F2C52E6445C713ULL, "sceKernelMkdir", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager& mem) {
            const i32 r = get_vfs().mkdir(guest_cstring(mem, ctx.rdi, 1024), static_cast<i32>(ctx.rsi));
            ctx.rax = r < 0 ? sce_err(r) : 0;
        });
    // sceKernelUnlink(const char* path)
    reg.register_symbol(0x0145D5C5678953F0ULL, "sceKernelUnlink", "libkernel",
        [sce_err](CpuContext& ctx, MemoryManager& mem) {
            const i32 r = get_vfs().unlink(guest_cstring(mem, ctx.rdi, 1024));
            ctx.rax = r < 0 ? sce_err(r) : 0;
        });
}

} // namespace ps4
