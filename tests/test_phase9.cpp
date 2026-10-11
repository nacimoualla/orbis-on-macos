#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/thread.hpp"

int main() {
    std::cout << "[PHASE 9] Starting Threading & Synchronization HLE unit tests...\n";

    ps4::MemoryManager memory;
    assert(memory.initialize());

    ps4::ThreadManager thread_mgr(memory);

    // Test 1: Thread creation and joining with return value
    {
        std::atomic<bool> executed{false};
        ps4::ScePthreadId tid = thread_mgr.create_thread(
            0x401000, 0x42, nullptr, "worker_1",
            [&executed](ps4::GuestThread& th) {
                assert(th.argument == 0x42);
                executed = true;
                th.return_value = 0x1337;
            });

        assert(tid > 0);
        ps4::u64 ret = 0;
        assert(thread_mgr.join_thread(tid, &ret));
        assert(executed.load());
        assert(ret == 0x1337);

        std::cout << "  ✓ Test 1: Single thread spawned, executed with args, and joined successfully\n";
    }

    // Test 2: Concurrent multi-threading
    {
        constexpr int NUM_WORKERS = 4;
        std::atomic<int> counter{0};
        std::vector<ps4::ScePthreadId> tids;

        for (int i = 0; i < NUM_WORKERS; ++i) {
            tids.push_back(thread_mgr.create_thread(
                0x402000, i, nullptr, "worker_pool",
                [&counter](ps4::GuestThread&) {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }));
        }

        for (auto tid : tids) {
            assert(thread_mgr.join_thread(tid));
        }
        assert(counter.load() == NUM_WORKERS);

        std::cout << "  ✓ Test 2: Pool of 4 concurrent threads executed and joined without race\n";
    }

    // Test 3: Mutex mutual exclusion
    {
        const ps4::u32 mtx = thread_mgr.mutex_create("test_mutex");
        assert(mtx > 0);

        int shared_resource = 0;
        constexpr int ITERATIONS_PER_THREAD = 2500;
        constexpr int NUM_THREADS = 4;
        std::vector<ps4::ScePthreadId> tids;

        for (int i = 0; i < NUM_THREADS; ++i) {
            tids.push_back(thread_mgr.create_thread(
                0x403000, 0, nullptr, "mutex_worker",
                [&thread_mgr, mtx, &shared_resource](ps4::GuestThread&) {
                    for (int j = 0; j < ITERATIONS_PER_THREAD; ++j) {
                        assert(thread_mgr.mutex_lock(mtx));
                        shared_resource++;
                        assert(thread_mgr.mutex_unlock(mtx));
                    }
                }));
        }

        for (auto tid : tids) {
            assert(thread_mgr.join_thread(tid));
        }

        assert(shared_resource == NUM_THREADS * ITERATIONS_PER_THREAD);
        assert(thread_mgr.mutex_destroy(mtx));

        std::cout << "  ✓ Test 3: Mutex mutual exclusion protected shared variable across 10,000 operations\n";
    }

    // Test 4: Mutex trylock
    {
        const ps4::u32 mtx = thread_mgr.mutex_create("trylock_mutex");
        assert(thread_mgr.mutex_trylock(mtx));
        assert(thread_mgr.mutex_unlock(mtx));
        assert(thread_mgr.mutex_destroy(mtx));

        std::cout << "  ✓ Test 4: Mutex trylock semantics verified\n";
    }

    // Test 5: Condition variable signaling & waiting
    {
        const ps4::u32 mtx = thread_mgr.mutex_create("cond_mutex");
        const ps4::u32 cond = thread_mgr.cond_create("cond_var");
        bool ready = false;

        ps4::ScePthreadId consumer = thread_mgr.create_thread(
            0x404000, 0, nullptr, "consumer",
            [&thread_mgr, mtx, cond, &ready](ps4::GuestThread&) {
                assert(thread_mgr.mutex_lock(mtx));
                while (!ready) {
                    assert(thread_mgr.cond_wait(cond, mtx));
                }
                assert(thread_mgr.mutex_unlock(mtx));
            });

        // Let consumer start waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        // Producer signals
        assert(thread_mgr.mutex_lock(mtx));
        ready = true;
        assert(thread_mgr.cond_signal(cond));
        assert(thread_mgr.mutex_unlock(mtx));

        assert(thread_mgr.join_thread(consumer));
        assert(thread_mgr.cond_destroy(cond));
        assert(thread_mgr.mutex_destroy(mtx));

        std::cout << "  ✓ Test 5: Condition variable wait/signal pattern synchronized consumer thread\n";
    }

    // Test 6: NID Registry HLE Threading Symbols
    {
        ps4::NidRegistry registry;
        registry.register_default_libraries();

        assert(registry.find_by_name("scePthreadCreate") != nullptr);
        assert(registry.find_by_name("scePthreadJoin") != nullptr);
        assert(registry.find_by_name("scePthreadMutexLock") != nullptr);
        assert(registry.find_by_name("scePthreadCondSignal") != nullptr);

        const auto* gettime_sym = registry.find_by_name("sceKernelGettimeofday");
        assert(gettime_sym != nullptr);

        // Test sceKernelGettimeofday HLE handler
        const ps4::GuestAddr timeval_addr = 0x500000ULL;
        ps4::CpuContext ctx{};
        ctx.rdi = timeval_addr;
        gettime_sym->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);

        ps4::i64 sec = 0, usec = 0;
        assert(memory.read_value(timeval_addr, sec));
        assert(memory.read_value(timeval_addr + 8, usec));
        assert(sec > 1700000000); // Unix timestamp valid

        std::cout << "  ✓ Test 6: Threading NIDs registered and sceKernelGettimeofday verified\n";
    }

    std::cout << "[PHASE 9] PASS: All Threading & Synchronization HLE tests passed!\n";
    return 0;
}
