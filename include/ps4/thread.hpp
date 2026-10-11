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
    Running,
    Waiting,
    Terminated
};

struct ScePthreadAttr {
    GuestAddr stack_addr{0};
    GuestSize stack_size{2 * 1024 * 1024}; // 2MB default stack
    i32 priority{256};
    u64 affinity{0};
    char name[32]{0};
};

struct GuestThread {
    ScePthreadId id{0};
    std::string name{"ps4_thread"};
    ThreadState state{ThreadState::Uninitialized};
    GuestAddr entry_point{0};
    GuestAddr argument{0};
    GuestAddr stack_base{0};
    GuestSize stack_size{0};
    u64 return_value{0};
    std::thread host_thread;
    CpuContext context{};
};

class ThreadManager {
public:
    explicit ThreadManager(MemoryManager& memory);
    ~ThreadManager();

    // Non-copyable
    ThreadManager(const ThreadManager&) = delete;
    ThreadManager& operator=(const ThreadManager&) = delete;

    /// Create and launch a new guest thread with host execution
    ScePthreadId create_thread(GuestAddr entry_point, GuestAddr arg,
                              const ScePthreadAttr* attr = nullptr,
                              std::string_view name = "ps4_thread",
                              std::function<void(GuestThread&)> host_task = nullptr);

    /// Join an existing thread and wait for completion
    bool join_thread(ScePthreadId id, u64* out_return_value = nullptr);

    /// Get current executing thread ID
    [[nodiscard]] ScePthreadId current_thread_id() const;

    /// Set current thread's return value and mark terminated
    void exit_current_thread(u64 return_val);

    /// Total number of active threads
    [[nodiscard]] usize active_thread_count() const;

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

private:
    MemoryManager& m_memory;
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
};

} // namespace ps4
