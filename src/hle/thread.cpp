#include "ps4/thread.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace ps4 {

namespace {
thread_local ScePthreadId t_current_thread_id = 0;
} // namespace

ThreadManager::ThreadManager(MemoryManager& memory, SyscallTable* syscalls)
    : m_memory(memory), m_syscalls(syscalls) {}

ThreadManager::~ThreadManager() {
    // 1. Mark all active threads as terminated so worker loops exit
    {
        std::lock_guard<std::mutex> lock(m_threads_mutex);
        for (auto& [id, thread] : m_threads) {
            if (thread && thread->state.load() != ThreadState::Terminated) {
                thread->state.store(ThreadState::Terminated);
            }
        }
    }

    // 2. Unblock any threads waiting on equeues
    {
        std::lock_guard<std::mutex> lock(m_equeues_mutex);
        for (auto& [id, eq] : m_equeues) {
            if (eq) {
                std::lock_guard<std::mutex> eq_lock(eq->mutex);
                eq->deleted = true;
                eq->cv.notify_all();
            }
        }
    }

    // 3. Unblock any threads waiting on condition variables
    {
        std::lock_guard<std::mutex> lock(m_sync_mutex);
        for (auto& [id, cv] : m_conds) {
            if (cv) cv->notify_all();
        }
    }

    // 4. Safely join all host threads without holding m_threads_mutex
    std::vector<GuestThread*> threads_to_join;
    {
        std::lock_guard<std::mutex> lock(m_threads_mutex);
        for (auto& [id, thread] : m_threads) {
            if (thread && thread->host_thread.joinable()) {
                threads_to_join.push_back(thread.get());
            }
        }
    }

    for (auto* th : threads_to_join) {
        if (th->host_thread.joinable()) {
            th->host_thread.join();
        }
    }
}

ScePthreadId ThreadManager::create_thread(GuestAddr entry_point, GuestAddr arg,
                                          const ScePthreadAttr* attr,
                                          std::string_view name,
                                          std::function<void(GuestThread&)> host_task) {
    std::lock_guard<std::mutex> lock(m_threads_mutex);

    constexpr GuestSize kTlsReserveSize = 0x10000; // 64 KB reserved for TLS/TCB below stack
    GuestSize stack_size = (attr && attr->stack_size >= 65536) ? attr->stack_size : 2 * 1024 * 1024;
    GuestAddr stack_base = 0;
    GuestAddr tls_base = 0;

    if (attr && attr->stack_addr != 0 && attr->stack_size >= 65536 &&
        m_memory.is_valid_address(attr->stack_addr, stack_size)) {
        stack_base = attr->stack_addr;
        // Allocate TLS separately so we never overwrite a small user-supplied stack
        m_next_stack_top -= kTlsReserveSize;
        tls_base = (m_next_stack_top + 0x8000) & ~0xFFFULL;
        m_memory.zero_memory(m_next_stack_top, kTlsReserveSize);
    } else {
        m_next_stack_top -= (stack_size + kTlsReserveSize);
        tls_base = (m_next_stack_top + 0x8000) & ~0xFFFULL;
        stack_base = m_next_stack_top + kTlsReserveSize;
        m_memory.zero_memory(m_next_stack_top, stack_size + kTlsReserveSize);
    }

    // System V x86_64 ABI: RSP must be (16n - 8) at function entry where [RSP] holds return address
    const GuestAddr initial_rsp = ((stack_base + stack_size - 16) & ~15ULL) - 8;

    // Write return address 0 to the top of stack (standard System V entry convention)
    m_memory.write_value(initial_rsp, static_cast<u64>(0));

    auto thread_ptr = std::make_unique<GuestThread>();
    const ScePthreadId tid = m_next_thread_id++;
    thread_ptr->id = tid;
    if (!name.empty()) {
        thread_ptr->name = std::string(name);
    } else {
        thread_ptr->name = "ps4_thread";
    }
    thread_ptr->state.store(ThreadState::Ready);
    thread_ptr->entry_point = entry_point;
    thread_ptr->argument = arg;
    thread_ptr->stack_base = stack_base;
    thread_ptr->stack_size = stack_size;
    thread_ptr->context.rsp = initial_rsp;
    thread_ptr->context.rdi = arg;
    thread_ptr->context.rip = entry_point;

    thread_ptr->context.fs_base = tls_base;
    m_memory.write_value(tls_base, tls_base);

    m_threads_created.fetch_add(1);

    std::fprintf(stderr, "[Thread] Created guest thread '%s' (TID %u): entry=0x%llx, stack=0x%llx-0x%llx, rsp=0x%llx, fs_base=0x%llx, arg=0x%llx\n",
                 thread_ptr->name.c_str(), tid,
                 (unsigned long long)entry_point,
                 (unsigned long long)stack_base,
                 (unsigned long long)(stack_base + stack_size),
                 (unsigned long long)initial_rsp,
                 (unsigned long long)tls_base,
                 (unsigned long long)arg);

    GuestThread* raw_thread = thread_ptr.get();
    raw_thread->host_thread = std::thread([this, raw_thread, host_task]() {
        t_current_thread_id = raw_thread->id;
        m_threads_started.fetch_add(1);
        raw_thread->state.store(ThreadState::Running);

        std::fprintf(stderr, "[Thread] Starting execution of thread '%s' (TID %u) at RIP=0x%llx\n",
                     raw_thread->name.c_str(), raw_thread->id,
                     (unsigned long long)raw_thread->entry_point);

        if (host_task) {
            host_task(*raw_thread);
        } else if (m_syscalls && raw_thread->entry_point != 0) {
            CpuExecutor executor(m_memory, *m_syscalls);
            if (m_jit) {
                executor.set_jit_compiler(m_jit);
            }
            executor.context() = raw_thread->context;

            constexpr usize QUANTUM = 20000;
            while (raw_thread->state.load() != ThreadState::Terminated && !m_syscalls->has_exited()) {
                RunResult rr = executor.run(QUANTUM);
                raw_thread->instructions_executed.fetch_add(rr.instructions_executed);
                m_threads_executed_instructions.fetch_add(rr.instructions_executed);
                raw_thread->context = executor.context();

                if (rr.status == StepResult::Exited || rr.status == StepResult::Halted) {
                    raw_thread->return_value = executor.context().rax;
                    break;
                }
                if (rr.status != StepResult::Ok && rr.status != StepResult::Syscall) {
                    std::fprintf(stderr, "[Thread] Thread '%s' (TID %u) halted at RIP=0x%llx with %s\n",
                                 raw_thread->name.c_str(), raw_thread->id,
                                 (unsigned long long)executor.context().rip,
                                 step_result_to_string(rr.status).data());
                    break;
                }

                // Cooperative yield between quanta
                std::this_thread::yield();
            }
        }

        raw_thread->state.store(ThreadState::Terminated);
        std::fprintf(stderr, "[Thread] Terminated thread '%s' (TID %u), executed %llu instructions\n",
                     raw_thread->name.c_str(), raw_thread->id,
                     (unsigned long long)raw_thread->instructions_executed.load());
    });

    m_threads[tid] = std::move(thread_ptr);
    return tid;
}

bool ThreadManager::join_thread(ScePthreadId id, u64* out_return_value) {
    GuestThread* th = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_threads_mutex);
        auto it = m_threads.find(id);
        if (it == m_threads.end()) {
            return false;
        }
        th = it->second.get();
    }

    if (th->host_thread.joinable()) {
        th->host_thread.join();
    }

    if (out_return_value) {
        *out_return_value = th->return_value;
    }
    th->state.store(ThreadState::Terminated);
    return true;
}

bool ThreadManager::rename_thread(ScePthreadId id, std::string_view name) {
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    auto it = m_threads.find(id);
    if (it != m_threads.end()) {
        it->second->name = std::string(name);
        return true;
    }
    return false;
}

ScePthreadId ThreadManager::current_thread_id() const {
    return t_current_thread_id;
}

GuestThread* ThreadManager::current_thread() {
    return get_thread(t_current_thread_id);
}

GuestThread* ThreadManager::get_thread(ScePthreadId id) {
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    auto it = m_threads.find(id);
    if (it != m_threads.end()) {
        return it->second.get();
    }
    return nullptr;
}

void ThreadManager::exit_current_thread(u64 return_val) {
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    auto it = m_threads.find(t_current_thread_id);
    if (it != m_threads.end()) {
        it->second->return_value = return_val;
        it->second->state.store(ThreadState::Terminated);
    }
}

usize ThreadManager::active_thread_count() const {
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    usize count = 0;
    for (const auto& [id, thread] : m_threads) {
        if (thread) {
            const auto st = thread->state.load();
            if (st != ThreadState::Terminated && st != ThreadState::Uninitialized) {
                count++;
            }
        }
    }
    return count;
}

void ThreadManager::dump_thread_states() const {
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    usize active_count = 0;
    for (const auto& [id, th] : m_threads) {
        if (th) {
            const auto st = th->state.load();
            if (st != ThreadState::Terminated && st != ThreadState::Uninitialized) {
                active_count++;
            }
        }
    }
    std::fprintf(stderr, "=== Thread States Dump (Total active: %zu) ===\n", active_count);
    for (const auto& [id, th] : m_threads) {
        if (!th) continue;
        const char* st = "Unknown";
        switch (th->state.load()) {
            case ThreadState::Uninitialized: st = "Uninitialized"; break;
            case ThreadState::Ready: st = "Ready"; break;
            case ThreadState::Running: st = "Running"; break;
            case ThreadState::Waiting: st = "Waiting"; break;
            case ThreadState::WaitingEqueue: st = "WaitingEqueue"; break;
            case ThreadState::WaitingSleep: st = "WaitingSleep"; break;
            case ThreadState::WaitingSync: st = "WaitingSync"; break;
            case ThreadState::Terminated: st = "Terminated"; break;
        }
        std::fprintf(stderr, "  [Thread %u] '%s' state=%s, RIP=0x%llx, instrs=%llu\n",
                     th->id, th->name.c_str(), st,
                     (unsigned long long)th->context.rip,
                     (unsigned long long)th->instructions_executed.load());
    }
    std::fprintf(stderr, "===============================================\n");
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
    GuestThread* cur_th = current_thread();
    if (cur_th) cur_th->state.store(ThreadState::WaitingSync);
    mtx->lock();
    if (cur_th) cur_th->state.store(ThreadState::Running);
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
    GuestThread* cur_th = current_thread();
    if (cur_th) cur_th->state.store(ThreadState::WaitingSync);
    cv->wait(*mtx);
    if (cur_th) cur_th->state.store(ThreadState::Running);
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
    GuestThread* cur_th = current_thread();
    if (cur_th) cur_th->state.store(ThreadState::WaitingSync);
    const auto status = cv->wait_for(*mtx, std::chrono::microseconds(timeout_usec));
    if (cur_th) cur_th->state.store(ThreadState::Running);
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

u32 ThreadManager::equeue_create(std::string_view name) {
    std::lock_guard<std::mutex> lock(m_equeues_mutex);
    const u32 id = m_next_equeue_id++;
    auto eq = std::make_shared<Equeue>();
    eq->id = id;
    eq->name = std::string(name);
    m_equeues[id] = std::move(eq);
    return id;
}

bool ThreadManager::equeue_delete(u32 eq_id) {
    std::shared_ptr<Equeue> eq;
    {
        std::lock_guard<std::mutex> lock(m_equeues_mutex);
        auto it = m_equeues.find(eq_id);
        if (it == m_equeues.end()) return false;
        eq = it->second;
        m_equeues.erase(it);
    }
    if (eq) {
        std::lock_guard<std::mutex> eq_lock(eq->mutex);
        eq->deleted = true;
        eq->cv.notify_all();
    }
    return true;
}

bool ThreadManager::equeue_wait(u32 eq_id, std::vector<SceKernelEvent>& out_events, usize max_events,
                                u64 timeout_usec, bool has_timeout) {
    std::shared_ptr<Equeue> eq;
    {
        std::lock_guard<std::mutex> lock(m_equeues_mutex);
        auto it = m_equeues.find(eq_id);
        if (it == m_equeues.end()) return false;
        eq = it->second;
    }

    GuestThread* cur_th = current_thread();
    if (cur_th) {
        cur_th->state.store(ThreadState::WaitingEqueue);
    }

    std::unique_lock<std::mutex> lock(eq->mutex);
    auto pred = [&]() {
        return !eq->events.empty() || eq->deleted || (m_syscalls && m_syscalls->has_exited());
    };

    bool got_events = false;
    if (has_timeout) {
        got_events = eq->cv.wait_for(lock, std::chrono::microseconds(timeout_usec), pred);
    } else {
        // Cooperative wait loop: wake periodically (50ms) to check exit condition / avoid deadlock
        while (!pred()) {
            if (eq->cv.wait_for(lock, std::chrono::milliseconds(50), pred)) {
                break;
            }
            if (m_syscalls && m_syscalls->has_exited()) {
                break;
            }
        }
        got_events = !eq->events.empty();
    }

    if (got_events && !eq->events.empty()) {
        const usize count = std::min(max_events, eq->events.size());
        for (usize i = 0; i < count; ++i) {
            out_events.push_back(eq->events[i]);
        }
        eq->events.erase(eq->events.begin(), eq->events.begin() + count);
    }

    if (cur_th) {
        cur_th->state.store(ThreadState::Running);
    }

    return !out_events.empty();
}

bool ThreadManager::equeue_trigger_user_event(u32 eq_id, i32 event_id, u64 udata) {
    SceKernelEvent ev{};
    ev.ident = static_cast<u64>(event_id);
    ev.filter = -1; // EVFILT_USER
    ev.flags = 0;
    ev.fflags = 0;
    ev.data = 0;
    ev.udata = udata;
    return equeue_signal(eq_id, ev);
}

bool ThreadManager::equeue_signal(u32 eq_id, const SceKernelEvent& ev) {
    std::shared_ptr<Equeue> eq;
    {
        std::lock_guard<std::mutex> lock(m_equeues_mutex);
        auto it = m_equeues.find(eq_id);
        if (it == m_equeues.end()) return false;
        eq = it->second;
    }
    std::lock_guard<std::mutex> lock(eq->mutex);
    eq->events.push_back(ev);
    eq->cv.notify_one();
    return true;
}

void ThreadManager::signal_flip_event(i64 flip_arg) {
    std::lock_guard<std::mutex> lock(m_equeues_mutex);
    for (auto& [id, eq] : m_equeues) {
        if (!eq || eq->deleted) continue;
        std::lock_guard<std::mutex> eq_lock(eq->mutex);
        SceKernelEvent ev{};
        ev.ident = 1;
        ev.filter = 0;
        ev.flags = 0;
        ev.fflags = 0;
        ev.data = flip_arg;
        ev.udata = 0;
        eq->events.push_back(ev);
        eq->cv.notify_all();
    }
}

} // namespace ps4
