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
};

struct SceKernelEvent {
    u64 ident{0};
    i16 filter{0};
    u16 flags{0};
    u32 fflags{0};
    i64 data{0};
    u64 udata{0};
};

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
    std::thread host_thread;
    CpuContext context{};
    std::atomic<u64> instructions_executed{0};
};

class SyscallTable;
class JitCompiler;

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

    /// Get pointer to thread by ID
    [[nodiscard]] GuestThread* get_thread(ScePthreadId id);

    /// Set current thread's return value and mark terminated
    void exit_current_thread(u64 return_val);

    /// Total number of active threads
    [[nodiscard]] usize active_thread_count() const;

    /// Diagnostics: dump all thread states and RIPs
    void dump_thread_states() const;

    // Telemetry getters
    [[nodiscard]] u64 threads_created() const noexcept { return m_threads_created.load(); }
    [[nodiscard]] u64 threads_started() const noexcept { return m_threads_started.load(); }
    [[nodiscard]] u64 threads_executed_instructions() const noexcept { return m_threads_executed_instructions.load(); }

    // Mutex management
    u32 mutex_create(std::string_view name = "mutex");
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
    void signal_flip_event(i64 flip_arg = 0);

private:
    MemoryManager& m_memory;
    SyscallTable* m_syscalls{nullptr};
    JitCompiler* m_jit{nullptr};
    mutable std::mutex m_threads_mutex;
    std::unordered_map<ScePthreadId, std::unique_ptr<GuestThread>> m_threads;
    ScePthreadId m_next_thread_id{1};

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
};

} // namespace ps4
