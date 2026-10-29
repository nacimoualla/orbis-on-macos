#pragma once

#include "ps4/common.hpp"
#include "ps4/cpu_context.hpp"
#include "ps4/memory.hpp"
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ps4 {

using ScePthread = u64;
using ScePthreadId = u32;

enum class ThreadState {
    Uninitialized,
    Ready,
    Running,
    Waiting,
    WaitingEqueue,
    WaitingSleep,
    WaitingSync,
    Terminated
};

struct ScePthreadAttr {
    GuestAddr stack_addr{0};
    GuestSize stack_size{2 * 1024 * 1024}; // 2MB default stack
    i32 priority{256};
    u64 affinity{0};
    char name[32]{0};
    i32 detachstate{0}; // 0 = joinable, 1 = detached (appended: never serialized to guest)
};

struct SceKernelEvent {
    u64 ident{0};
    i16 filter{0};
    u16 flags{0};
    u32 fflags{0};
    i64 data{0};
    u64 udata{0};
};

// Orbis kqueue filters (SceKernelEvent::filter).
constexpr i16 kEvFiltUser = -11;         // SCE_KERNEL_EVFILT_USER
constexpr i16 kEvFiltVideoOut = -13;     // SCE_KERNEL_EVFILT_VIDEO_OUT
constexpr i16 kEvFiltGraphicsCore = -14; // SCE_KERNEL_EVFILT_GRAPHICS_CORE

struct Equeue {
    u32 id{0};
    std::string name{"equeue"};
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<SceKernelEvent> events;
    bool deleted{false};
};

struct GuestThread {
    ScePthreadId id{0};
    std::string name{"ps4_thread"};
    std::atomic<ThreadState> state{ThreadState::Uninitialized};
    GuestAddr entry_point{0};
    GuestAddr argument{0};
    GuestAddr stack_base{0};
    GuestSize stack_size{0};
    u64 return_value{0};
    bool detached{false}; // created detached: join not required (still joined at shutdown)
    i32 priority{256};
    std::atomic<u64> sleep_until_us{0}; // host steady-clock wake deadline while WaitingSleep
    // Stall-watchdog telemetry (all atomics; lock-free for the dispatch hot path)
    std::atomic<u64> run_returns{0};   // executor.run() completions
    std::atomic<u64> hle_calls{0};     // HLE dispatches on this thread
    std::atomic<bool> in_hle{false};   // inside a handler right now
    std::atomic<u64> last_nid{0};      // most recent HLE NID
    // Return address of the call currently being serviced. live_context is only
    // published at quantum boundaries, so a thread blocked inside a handler has
    // live_context == nullptr and the dump would otherwise print the entry
    // context -- making a thread that ran 1,400 instructions look like it never
    // left its entry point. This pair disambiguates "blocked in HLE" from that.
    std::atomic<GuestAddr> hle_caller_rip{0};
    // Worker's live executor context, set after each quantum copy-in and
    // cleared on thread exit. Read by the watchdog without locking; atomicity
    // only (tearing documented) — never dereference past thread exit.
    std::atomic<const CpuContext*> live_context{nullptr};
    std::thread host_thread;
    CpuContext context{};
    std::atomic<u64> instructions_executed{0};
    // Guest exceptions raised at this thread (sceKernelRaiseException),
    // delivered on its own host thread by ThreadManager::deliver_pending_exceptions.
    std::mutex pending_signals_mtx;
    std::vector<int> pending_signals;
    std::atomic<bool> has_pending_signals{false};
    bool delivering_signal{false}; // owning host thread only
    // Condition variable this thread is blocked on inside an HLE wait (set by
    // the wait for its duration). raise_exception notifies it so a suspend
    // signal is answered at once instead of on the wait's next poll timeout;
    // the GC's stop-the-world otherwise stalled every frame for up to ~100 ms.
    std::mutex wait_cv_mtx;
    std::condition_variable* wait_cv{nullptr};
    // Host QoS class wanted for this thread (from its name), applied by the
    // thread itself at its next quantum boundary; 0 = nothing pending.
    std::atomic<unsigned> pending_qos{0};
};

class CpuExecutor;

class SyscallTable;
class JitCompiler;

/// Stall-watchdog activity stamp. Called on every HLE dispatch; the watchdog
/// (owned by ThreadManager) dumps per-thread state after ~2 s without any.
void note_hle_activity();

/// Calling thread's guest TID (thread-local; 1 on the main host thread).
[[nodiscard]] ScePthreadId current_hle_thread_id();

/// Per-thread activity stamp for the watchdog trigger (also refreshes the
/// global stamp). Cheap: thread-local read + one guarded map update.
void note_hle_activity_for(ScePthreadId tid);

/// Post-stall verbose logging: set once after the first stall-b dump; makes
/// the HLE dispatch bypass its quiet/whitelist filter (except keepalive TID 5).
void set_post_stall_verbose();
[[nodiscard]] bool post_stall_verbose();

/// Sema-table snapshot + synthetic stall wake (implemented in nid_registry.cpp).
void dump_sema_states();
void debug_wake_stall_semas();

class ThreadManager {
public:
    explicit ThreadManager(MemoryManager& memory, SyscallTable* syscalls = nullptr);
    ~ThreadManager();

    // Non-copyable
    ThreadManager(const ThreadManager&) = delete;
    ThreadManager& operator=(const ThreadManager&) = delete;

    /// Attach SyscallTable for guest thread syscall/HLE execution
    void set_syscall_table(SyscallTable* syscalls) noexcept { m_syscalls = syscalls; }
    [[nodiscard]] SyscallTable* syscall_table() noexcept { return m_syscalls; }
    [[nodiscard]] const SyscallTable* syscall_table() const noexcept { return m_syscalls; }

    /// Attach JIT compiler for tier-2 execution in threads
    void set_jit_compiler(JitCompiler* jit) noexcept { m_jit = jit; }

    /// Create and launch a new guest thread with host execution
    ScePthreadId create_thread(GuestAddr entry_point, GuestAddr arg,
                              const ScePthreadAttr* attr = nullptr,
                              std::string_view name = "ps4_thread",
                              std::function<void(GuestThread&)> host_task = nullptr);

    /// Join an existing thread and wait for completion
    bool join_thread(ScePthreadId id, u64* out_return_value = nullptr);

    /// Rename thread
    bool rename_thread(ScePthreadId id, std::string_view name);

    /// Get current executing thread ID
    [[nodiscard]] ScePthreadId current_thread_id() const;

    /// Get pointer to current thread
    [[nodiscard]] GuestThread* current_thread();
    [[nodiscard]] static GuestThread* current_guest_thread() noexcept;

    /// Get pointer to thread by ID
    [[nodiscard]] GuestThread* get_thread(ScePthreadId id);

    /// Set current thread's return value and mark terminated
    void exit_current_thread(u64 return_val);

    // --- Guest exceptions (sceKernelInstallExceptionHandler / RaiseException)
    // Delivery is asynchronous, like a signal: the target thread runs the
    // handler on its own stack at its next delivery point -- between execution
    // quanta, or inside a blocking HLE wait -- then resumes where it was.
    void install_exception_handler(int signum, GuestAddr handler);
    /// Address of an HLT the handler returns to (end of delivery).
    void set_exception_return_sentinel(GuestAddr hlt_addr) noexcept { m_exc_sentinel = hlt_addr; }
    /// Queue `signum` for thread `tid`. Returns false if no such live thread.
    bool raise_exception(ScePthreadId tid, int signum);
    /// Run the calling thread's pending handlers on `ex`. Safe to call with no
    /// locks held from a wait loop; a no-op when nothing is pending.
    void deliver_pending_exceptions(CpuExecutor& ex);

    /// Total number of active threads
    [[nodiscard]] usize active_thread_count() const;

    /// Diagnostics: dump all thread states and RIPs
    void dump_thread_states() const;

    /// Stall watchdog: wall-clock thread taking register-level snapshots after
    /// ~2 s of HLE silence. Started/stopped automatically with this object.
    void start_watchdog();
    void stop_watchdog();

    /// Point the watchdog at main's live CpuContext (stepped by the runner loop,
    /// not stored in the main GuestThread entry). Must outlive this manager.
    void set_main_context(const CpuContext* ctx);

    // Telemetry getters
    [[nodiscard]] u64 threads_created() const noexcept { return m_threads_created.load(); }
    [[nodiscard]] u64 threads_started() const noexcept { return m_threads_started.load(); }
    [[nodiscard]] u64 threads_executed_instructions() const noexcept { return m_threads_executed_instructions.load(); }

    // Mutex management
    u32 mutex_create(std::string_view name = "mutex");
    bool mutex_exists(u32 mutex_id);
    bool mutex_lock(u32 mutex_id);
    bool mutex_trylock(u32 mutex_id);
    bool mutex_unlock(u32 mutex_id);
    bool mutex_destroy(u32 mutex_id);

    // Condition variable management
    u32 cond_create(std::string_view name = "cond");
    bool cond_wait(u32 cond_id, u32 mutex_id);
    bool cond_timedwait(u32 cond_id, u32 mutex_id, u64 timeout_usec);
    bool cond_signal(u32 cond_id);
    bool cond_broadcast(u32 cond_id);
    bool cond_destroy(u32 cond_id);

    // Equeue management
    u32 equeue_create(std::string_view name = "equeue");
    bool equeue_delete(u32 eq_id);
    bool equeue_wait(u32 eq_id, std::vector<SceKernelEvent>& out_events, usize max_events,
                     u64 timeout_usec = 0, bool has_timeout = false);
    bool equeue_trigger_user_event(u32 eq_id, i32 event_id, u64 udata);
    bool equeue_signal(u32 eq_id, const SceKernelEvent& ev);

private:
    void dump_stall_snapshot(const char* tag) const;
    void watchdog_loop();

    MemoryManager& m_memory;
    SyscallTable* m_syscalls{nullptr};
    JitCompiler* m_jit{nullptr};
    mutable std::mutex m_threads_mutex;
    std::unordered_map<ScePthreadId, std::unique_ptr<GuestThread>> m_threads;
    ScePthreadId m_next_thread_id{1};

    std::mutex m_exc_mutex;
    std::unordered_map<int, GuestAddr> m_exc_handlers;
    GuestAddr m_exc_sentinel{0};

    // Stacks allocated from high guest memory downward
    GuestAddr m_next_stack_top{0x1D0000000ULL};

    // Synchronization tables
    std::mutex m_sync_mutex;
    u32 m_next_mutex_id{1};
    std::unordered_map<u32, std::unique_ptr<std::recursive_mutex>> m_mutexes;

    u32 m_next_cond_id{1};
    std::unordered_map<u32, std::unique_ptr<std::condition_variable_any>> m_conds;

    std::mutex m_equeues_mutex;
    u32 m_next_equeue_id{1};
    std::unordered_map<u32, std::shared_ptr<Equeue>> m_equeues;

    // Telemetry counters
    std::atomic<u64> m_threads_created{0};
    std::atomic<u64> m_threads_started{0};
    std::atomic<u64> m_threads_executed_instructions{0};

    // Stall watchdog (see start_watchdog)
    std::thread m_watchdog_thread;
    std::atomic<bool> m_watchdog_stop{false};
    std::atomic<const CpuContext*> m_main_context{nullptr};
};

} // namespace ps4
