#include "ps4/thread.hpp"
#include <chrono>

namespace ps4 {

namespace {
thread_local ScePthreadId t_current_thread_id = 0;
} // namespace

ThreadManager::ThreadManager(MemoryManager& memory) : m_memory(memory) {}

ThreadManager::~ThreadManager() {
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    for (auto& [id, thread] : m_threads) {
        if (thread && thread->host_thread.joinable()) {
            thread->host_thread.join();
        }
    }
}

ScePthreadId ThreadManager::create_thread(GuestAddr entry_point, GuestAddr arg,
                                          const ScePthreadAttr* attr,
                                          std::string_view name,
                                          std::function<void(GuestThread&)> host_task) {
    std::lock_guard<std::mutex> lock(m_threads_mutex);

    const GuestSize stack_size = (attr && attr->stack_size > 0) ? attr->stack_size : 2 * 1024 * 1024;
    m_next_stack_top -= stack_size;
    const GuestAddr stack_base = m_next_stack_top;
    const GuestAddr initial_rsp = (stack_base + stack_size - 16) & ~15ULL;

    // Zero out new stack space in virtual memory
    m_memory.zero_memory(stack_base, stack_size);

    auto thread_ptr = std::make_unique<GuestThread>();
    const ScePthreadId tid = m_next_thread_id++;
    thread_ptr->id = tid;
    thread_ptr->name = std::string(name);
    thread_ptr->state = ThreadState::Running;
    thread_ptr->entry_point = entry_point;
    thread_ptr->argument = arg;
    thread_ptr->stack_base = stack_base;
    thread_ptr->stack_size = stack_size;
    thread_ptr->context.rsp = initial_rsp;
    thread_ptr->context.rdi = arg;
    thread_ptr->context.rip = entry_point;

    GuestThread* raw_thread = thread_ptr.get();
    raw_thread->host_thread = std::thread([raw_thread, host_task]() {
        t_current_thread_id = raw_thread->id;
        if (host_task) {
            host_task(*raw_thread);
        }
        raw_thread->state = ThreadState::Terminated;
    });

    m_threads[tid] = std::move(thread_ptr);
    return tid;
}

bool ThreadManager::join_thread(ScePthreadId id, u64* out_return_value) {
    std::unique_ptr<GuestThread> thread_to_join;
    {
        std::lock_guard<std::mutex> lock(m_threads_mutex);
        auto it = m_threads.find(id);
        if (it == m_threads.end()) {
            return false;
        }
        if (it->second->host_thread.joinable()) {
            it->second->host_thread.join();
        }
        if (out_return_value) {
            *out_return_value = it->second->return_value;
        }
        it->second->state = ThreadState::Terminated;
    }
    return true;
}

ScePthreadId ThreadManager::current_thread_id() const {
    return t_current_thread_id;
}

void ThreadManager::exit_current_thread(u64 return_val) {
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    auto it = m_threads.find(t_current_thread_id);
    if (it != m_threads.end()) {
        it->second->return_value = return_val;
        it->second->state = ThreadState::Terminated;
    }
}

usize ThreadManager::active_thread_count() const {
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    usize count = 0;
    for (const auto& [id, thread] : m_threads) {
        if (thread && thread->state == ThreadState::Running) {
            count++;
        }
    }
    return count;
}

u32 ThreadManager::mutex_create(std::string_view) {
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    const u32 id = m_next_mutex_id++;
    m_mutexes[id] = std::make_unique<std::recursive_mutex>();
    return id;
}

bool ThreadManager::mutex_lock(u32 mutex_id) {
    std::recursive_mutex* mtx = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_sync_mutex);
        auto it = m_mutexes.find(mutex_id);
        if (it == m_mutexes.end()) return false;
        mtx = it->second.get();
    }
    mtx->lock();
    return true;
}

bool ThreadManager::mutex_trylock(u32 mutex_id) {
    std::recursive_mutex* mtx = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_sync_mutex);
        auto it = m_mutexes.find(mutex_id);
        if (it == m_mutexes.end()) return false;
        mtx = it->second.get();
    }
    return mtx->try_lock();
}

bool ThreadManager::mutex_unlock(u32 mutex_id) {
    std::recursive_mutex* mtx = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_sync_mutex);
        auto it = m_mutexes.find(mutex_id);
        if (it == m_mutexes.end()) return false;
        mtx = it->second.get();
    }
    mtx->unlock();
    return true;
}

bool ThreadManager::mutex_destroy(u32 mutex_id) {
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    return m_mutexes.erase(mutex_id) > 0;
}

u32 ThreadManager::cond_create(std::string_view) {
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    const u32 id = m_next_cond_id++;
    m_conds[id] = std::make_unique<std::condition_variable_any>();
    return id;
}

bool ThreadManager::cond_wait(u32 cond_id, u32 mutex_id) {
    std::condition_variable_any* cv = nullptr;
    std::recursive_mutex* mtx = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_sync_mutex);
        auto it_cv = m_conds.find(cond_id);
        auto it_mtx = m_mutexes.find(mutex_id);
        if (it_cv == m_conds.end() || it_mtx == m_mutexes.end()) return false;
        cv = it_cv->second.get();
        mtx = it_mtx->second.get();
    }
    cv->wait(*mtx);
    return true;
}

bool ThreadManager::cond_timedwait(u32 cond_id, u32 mutex_id, u64 timeout_usec) {
    std::condition_variable_any* cv = nullptr;
    std::recursive_mutex* mtx = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_sync_mutex);
        auto it_cv = m_conds.find(cond_id);
        auto it_mtx = m_mutexes.find(mutex_id);
        if (it_cv == m_conds.end() || it_mtx == m_mutexes.end()) return false;
        cv = it_cv->second.get();
        mtx = it_mtx->second.get();
    }
    const auto status = cv->wait_for(*mtx, std::chrono::microseconds(timeout_usec));
    return status == std::cv_status::no_timeout;
}

bool ThreadManager::cond_signal(u32 cond_id) {
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    auto it = m_conds.find(cond_id);
    if (it == m_conds.end()) return false;
    it->second->notify_one();
    return true;
}

bool ThreadManager::cond_broadcast(u32 cond_id) {
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    auto it = m_conds.find(cond_id);
    if (it == m_conds.end()) return false;
    it->second->notify_all();
    return true;
}

bool ThreadManager::cond_destroy(u32 cond_id) {
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    return m_conds.erase(cond_id) > 0;
}

} // namespace ps4
