#include "ps4/thread.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace ps4 {

namespace {
thread_local ScePthreadId t_current_thread_id = 1; // Main thread is ID 1 (never 0!)

// Last HLE-dispatch time (µs, steady clock). Written by note_hle_activity(),
// read by the watchdog loop. Diagnostics only; controls nothing.
std::atomic<u64> s_last_hle_us{0};

const char* thread_state_name(ThreadState st) {
    switch (st) {
        case ThreadState::Uninitialized: return "Uninitialized";
        case ThreadState::Ready: return "Ready";
        case ThreadState::Running: return "Running";
        case ThreadState::Waiting: return "Waiting";
        case ThreadState::WaitingEqueue: return "WaitingEqueue";
        case ThreadState::WaitingSleep: return "WaitingSleep";
        case ThreadState::WaitingSync: return "WaitingSync";
        case ThreadState::Terminated: return "Terminated";
    }
    return "Unknown";
}
} // namespace

static u64 host_now_us() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void note_hle_activity() {
    s_last_hle_us.store(host_now_us(), std::memory_order_relaxed);
}

thread_local GuestThread* t_current_guest_thread{nullptr};

ScePthreadId current_hle_thread_id() {
    return t_current_thread_id;
}

// Per-thread last-HLE time for the watchdog trigger (lock-free atomic array).
static constexpr usize MAX_HLE_TRACKED_THREADS = 256;
static std::atomic<u64> s_hle_last_us[MAX_HLE_TRACKED_THREADS]{};

// Post-stall verbose logging (see post_stall_verbose): once the first stall
// episode dumps, HLE dispatch logs everything except keepalive TID 5.
std::atomic<bool> s_post_stall_verbose{false};

void set_post_stall_verbose() {
    s_post_stall_verbose.store(true, std::memory_order_relaxed);
}

bool post_stall_verbose() {
    return s_post_stall_verbose.load(std::memory_order_relaxed);
}

// First-episode marker for one-time diagnostics (caller-site code dumps).
std::atomic<bool> s_first_episode_done{false};

void note_hle_activity_for(ScePthreadId tid) {
    const u64 now = host_now_us();
    s_last_hle_us.store(now, std::memory_order_relaxed);
    if (tid < MAX_HLE_TRACKED_THREADS) {
        s_hle_last_us[tid].store(now, std::memory_order_relaxed);
    }
}

ThreadManager::ThreadManager(MemoryManager& memory, SyscallTable* syscalls)
    : m_memory(memory), m_syscalls(syscalls) {
    m_next_thread_id = 2; // Worker threads start at ID 2
    // Register the main thread as ID 1 so scePthreadSelf()/joins resolve it.
    // Safe: it owns no host thread (never joined) and active_thread_count is unused.
    auto main_thread = std::make_unique<GuestThread>();
    main_thread->id = 1;
    main_thread->name = "main";
    main_thread->state.store(ThreadState::Running);
    t_current_guest_thread = main_thread.get();
    m_threads[1] = std::move(main_thread);

    note_hle_activity(); // grace period: no stall can predate the manager itself
    start_watchdog();
}

ThreadManager::~ThreadManager() {
    // 0. Park the watchdog first: it reads thread contexts and guest memory.
    stop_watchdog();

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
    constexpr GuestSize kMinStackSize = 64 * 1024;        // 64 KB
    constexpr GuestSize kMaxStackSize = 16 * 1024 * 1024; // 16 MB strict max: canary/garbage can NEVER underflow m_next_stack_top
    constexpr GuestSize kDefaultStackSize = 2 * 1024 * 1024;

    GuestSize stack_size = (attr && attr->stack_size >= kMinStackSize && attr->stack_size <= kMaxStackSize)
                               ? ((attr->stack_size + 0xFFFULL) & ~0xFFFULL)
                               : kDefaultStackSize;
    GuestAddr stack_base = 0;
    GuestAddr tls_base = 0;

    if (attr && attr->stack_addr != 0 && attr->stack_size >= kMinStackSize &&
        attr->stack_size <= kMaxStackSize &&
        m_memory.is_valid_address(attr->stack_addr, stack_size)) {
        stack_base = attr->stack_addr;
        // Allocate TLS separately so we never overwrite a small user-supplied stack.
        // No zeroing: untouched pages stay purely virtual (0 physical RAM) until written.
        m_next_stack_top -= kTlsReserveSize;
        tls_base = (m_next_stack_top + 0x8000) & ~0xFFFULL;
    } else {
        m_next_stack_top -= (stack_size + kTlsReserveSize);
        tls_base = (m_next_stack_top + 0x8000) & ~0xFFFULL;
        stack_base = m_next_stack_top + kTlsReserveSize;
    }

    // System V x86_64 ABI: RSP must be (16n - 8) at function entry where [RSP] holds return address
    const GuestAddr initial_rsp = ((stack_base + stack_size - 16) & ~15ULL) - 8;

    // Write return address 0 to the top of stack (standard System V entry convention)
    m_memory.write_value(initial_rsp, static_cast<u64>(0));

    auto thread_ptr = std::make_unique<GuestThread>();
    const ScePthreadId tid = m_next_thread_id++;
    thread_ptr->id = tid;
    if (attr) {
        thread_ptr->detached = (attr->detachstate != 0);
        thread_ptr->priority = attr->priority;
    }
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
        t_current_guest_thread = raw_thread;
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
            raw_thread->live_context.store(&executor.context(), std::memory_order_relaxed);
        t_guest_tid = static_cast<int>(raw_thread->id);

            constexpr usize QUANTUM = 20000;
            while (raw_thread->state.load() != ThreadState::Terminated && !m_syscalls->has_exited()) {
                RunResult rr = executor.run(QUANTUM);
                raw_thread->run_returns.fetch_add(1, std::memory_order_relaxed);
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

        raw_thread->live_context.store(nullptr, std::memory_order_relaxed);
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
    if (t_current_guest_thread) return t_current_guest_thread;
    return get_thread(t_current_thread_id);
}

GuestThread* ThreadManager::current_guest_thread() noexcept {
    return t_current_guest_thread;
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
        std::fprintf(stderr, "  [Thread %u] '%s' state=%s, RIP=0x%llx, instrs=%llu, detached=%d, prio=%d, wake_at_us=%llu\n",
                     th->id, th->name.c_str(), st,
                     (unsigned long long)th->context.rip,
                     (unsigned long long)th->instructions_executed.load(),
                     th->detached ? 1 : 0, th->priority,
                     (unsigned long long)th->sleep_until_us.load());
    }
    std::fprintf(stderr, "===============================================\n");
}

void ThreadManager::start_watchdog() {
    stop_watchdog(); // idempotent; also joins any previous thread
    m_watchdog_stop.store(false, std::memory_order_relaxed);
    m_watchdog_thread = std::thread([this]() { watchdog_loop(); });
}

void ThreadManager::stop_watchdog() {
    m_watchdog_stop.store(true, std::memory_order_relaxed);
    if (m_watchdog_thread.joinable()) {
        m_watchdog_thread.join();
    }
}

void ThreadManager::watchdog_loop() {
    // Per-thread idleness: a thread is stuck when it shows no HLE activity AND
    // no RIP/instruction movement for kStallThresholdUs. Chatter from one
    // thread can no longer veto snapshots of the others. Terminated threads
    // never count (their state is frozen by design); brand-new threads get one
    // free poll to establish a baseline. Cooldown bounds dump volume.
    constexpr u64 kStallThresholdUs = 2000ULL * 1000ULL;
    constexpr u64 kDumpCooldownUs = 30ULL * 1000ULL * 1000ULL;
    constexpr auto kPollInterval = std::chrono::milliseconds(50);
    std::unordered_map<ScePthreadId, std::pair<GuestAddr, u64>> baseline;
    u64 cooldown_until_us = 0;
    for (;;) {
        std::this_thread::sleep_for(kPollInterval);
        if (m_watchdog_stop.load(std::memory_order_relaxed)) break;
        const u64 now = host_now_us();
        bool any_stuck = false;
        {
            std::lock_guard<std::mutex> lock(m_threads_mutex);
            for (const auto& [id, th] : m_threads) {
                if (!th) continue;
                if (th->state.load() == ThreadState::Terminated) {
                    baseline.erase(id);
                    continue;
                }
                GuestAddr rip = th->context.rip;
                if (id == 1) {
                    if (const CpuContext* mc = m_main_context.load(std::memory_order_relaxed)) {
                        rip = mc->rip;
                    }
                }
                const u64 instrs = th->instructions_executed.load();
                auto it = baseline.find(id);
                if (it == baseline.end()) {
                    baseline.emplace(id, std::make_pair(rip, instrs));
                    continue;
                }
                const bool frozen = (rip == it->second.first && instrs == it->second.second);
                it->second = {rip, instrs};
                if (!frozen) continue;
                if (id >= MAX_HLE_TRACKED_THREADS) continue;
                const u64 last = s_hle_last_us[id].load(std::memory_order_relaxed);
                if (last == 0 || now <= last || now - last < kStallThresholdUs) continue;
                any_stuck = true;
            }
        }
        if (any_stuck && now >= cooldown_until_us) {
            dump_stall_snapshot("stall-a");
            std::this_thread::sleep_for(kPollInterval);
            if (m_watchdog_stop.load(std::memory_order_relaxed)) break;
            dump_stall_snapshot("stall-b");
            dump_sema_states();
            set_post_stall_verbose();
            if (!s_first_episode_done.exchange(true, std::memory_order_relaxed)) {
                // One-time: disassembly context for the two WaitSema call sites
                // (guest addresses are stable: fixed arena base, no ASLR).
                for (GuestAddr site : {0x1047811ULL, 0x8d30e6ULL}) {
                    const GuestAddr base = (site > 32) ? site - 32 : 0;
                    const u8* p = m_memory.is_valid_address(base, 64)
                                      ? m_memory.to_host_ptr<u8>(base) : nullptr;
                    if (p) {
                        std::fprintf(stderr, "    caller code @0x%llx:", (unsigned long long)base);
                        for (int i = 0; i < 64; ++i) std::fprintf(stderr, " %02x", p[i]);
                        std::fprintf(stderr, "   <-- call @+%llu\n",
                                     (unsigned long long)(site - base));
                    } else {
                        std::fprintf(stderr, "    caller code @0x%llx unmapped\n",
                                     (unsigned long long)site);
                    }
                }
            }
            // Synthetic wake DISABLED: real guest SignalSema traffic is under
            // test (e1cc now routes to the signal handler). Re-enable if the
            // loop still starves with signals flowing.
            cooldown_until_us = host_now_us() + kDumpCooldownUs;
        }
    }
}

void ThreadManager::set_main_context(const CpuContext* ctx) {
    m_main_context.store(ctx, std::memory_order_relaxed);
}

void ThreadManager::dump_stall_snapshot(const char* tag) const {
    // Values may tear: workers update their contexts without holding
    // m_threads_mutex. RIP/GPRs stay exact enough to identify poll sites.
    std::lock_guard<std::mutex> lock(m_threads_mutex);
    std::fprintf(stderr, "=== [WATCHDOG %s] stall snapshot ===\n", tag);
    for (const auto& [id, th] : m_threads) {
        if (!th) continue;
        // Live worker context first (mid-quantum truth), then main's live
        // context, then the entry snapshot (quantum-boundary fossil).
        const CpuContext* live = th->live_context.load(std::memory_order_relaxed);
        const CpuContext* mc = (!live && id == 1) ? m_main_context.load(std::memory_order_relaxed) : nullptr;
        const CpuContext& c = live ? *live : (mc ? *mc : th->context);
        // live_context is only published at quantum boundaries, so a thread
        // blocked inside a handler has live_context == nullptr and `c` falls
        // back to th->context -- the entry snapshot, whose rip is still the
        // entry point. That made a thread which ran ~1,400 instructions and then
        // parked in sem_wait look identical to one that never ran at all. Report
        // the HLE it is actually parked in so the two are distinguishable.
        const bool blocked_in_hle = th->in_hle.load(std::memory_order_relaxed);
        if (blocked_in_hle && !live) {
            std::fprintf(stderr,
                         "  [Thread %u] '%s' state=%s BLOCKED-IN-HLE nid=0x%llx caller=0x%llx"
                         " instrs=%llu runs=%llu hles=%llu\n",
                         th->id, th->name.c_str(), thread_state_name(th->state.load()),
                         (unsigned long long)th->last_nid.load(std::memory_order_relaxed),
                         (unsigned long long)th->hle_caller_rip.load(std::memory_order_relaxed),
                         (unsigned long long)th->instructions_executed.load(),
                         (unsigned long long)th->run_returns.load(),
                         (unsigned long long)th->hle_calls.load());
            continue;
        }
        std::fprintf(stderr,
                     "  [Thread %u] '%s' state=%s RIP=0x%llx instrs=%llu runs=%llu hles=%llu inhle=%d lastnid=0x%llx\n"
                     "    rax=0x%llx rbx=0x%llx rcx=0x%llx rdx=0x%llx\n"
                     "    rsi=0x%llx rdi=0x%llx rsp=0x%llx rbp=0x%llx\n"
                     "    r8=0x%llx r9=0x%llx r10=0x%llx r11=0x%llx\n"
                     "    r12=0x%llx r13=0x%llx r14=0x%llx r15=0x%llx rflags=0x%llx\n",
                     th->id, th->name.c_str(), thread_state_name(th->state.load()),
                     (unsigned long long)c.rip,
                     (unsigned long long)th->instructions_executed.load(),
                     (unsigned long long)th->run_returns.load(std::memory_order_relaxed),
                     (unsigned long long)th->hle_calls.load(std::memory_order_relaxed),
                     th->in_hle.load(std::memory_order_relaxed) ? 1 : 0,
                     (unsigned long long)th->last_nid.load(std::memory_order_relaxed),
                     (unsigned long long)c.rax, (unsigned long long)c.rbx,
                     (unsigned long long)c.rcx, (unsigned long long)c.rdx,
                     (unsigned long long)c.rsi, (unsigned long long)c.rdi,
                     (unsigned long long)c.rsp, (unsigned long long)c.rbp,
                     (unsigned long long)c.r8, (unsigned long long)c.r9,
                     (unsigned long long)c.r10, (unsigned long long)c.r11,
                     (unsigned long long)c.r12, (unsigned long long)c.r13,
                     (unsigned long long)c.r14, (unsigned long long)c.r15,
                     (unsigned long long)c.rflags);
        // 32 raw bytes around RIP for immediate disassembly from the log.
        const GuestAddr code_base = (c.rip > 16) ? c.rip - 16 : 0;
        const u8* code = m_memory.is_valid_address(code_base, 32)
                             ? m_memory.to_host_ptr<u8>(code_base) : nullptr;
        if (code) {
            std::fprintf(stderr, "    code @0x%llx:", (unsigned long long)code_base);
            for (int i = 0; i < 32; ++i) std::fprintf(stderr, " %02x", code[i]);
            std::fprintf(stderr, "   <-- RIP @+%llu\n",
                         (unsigned long long)(c.rip - code_base));
        } else {
            std::fprintf(stderr, "    code: RIP 0x%llx unmapped\n", (unsigned long long)c.rip);
        }
        // u64 behind pointer-sized GPRs: poll-target candidates (address + value).
        const GuestAddr regs[] = {c.rax, c.rbx, c.rcx, c.rdx, c.rsi, c.rdi,
                                  c.r8, c.r9, c.r10, c.r11, c.r12, c.r13, c.r14, c.r15};
        const char* names[] = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
                               "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
        for (int i = 0; i < 14; ++i) {
            if (regs[i] > 0x1000 && m_memory.is_valid_address(regs[i], sizeof(u64))) {
                u64 v = 0;
                m_memory.read_value(regs[i], v);
                std::fprintf(stderr, "    Mem[%s=0x%llx] = 0x%llx\n",
                             names[i], (unsigned long long)regs[i], (unsigned long long)v);
            }
        }
    }
    std::fprintf(stderr, "========================================\n");
}

u32 ThreadManager::mutex_create(std::string_view) {
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    const u32 id = m_next_mutex_id++;
    m_mutexes[id] = std::make_unique<std::recursive_mutex>();
    return id;
}

bool ThreadManager::mutex_exists(u32 mutex_id) {
    std::lock_guard<std::mutex> lock(m_sync_mutex);
    return m_mutexes.find(mutex_id) != m_mutexes.end();
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
