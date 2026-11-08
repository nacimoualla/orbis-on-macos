// Guest mutex semantics that games rely on (FreeBSD libthr behaviour).
//
// - Recursive mutexes (the default) nest; unlocking one you don't hold is EPERM.
// - Error-check / normal / adaptive mutexes answer a re-lock by the owner with
//   EDEADLK and leave the count alone. inFAMOUS Second Son locks its movie
//   mutex this way every frame ("lock; if (!depth) unlock"); a recursive
//   re-lock leaked one level per frame and starved its MovieDecoder thread.
// - pthread_cond_wait fully releases a recursive mutex locked several times
//   and restores the count afterwards.
// - scePthreadMutexattrSettype(4) reaches scePthreadMutexInit.

#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/thread.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

using TM = ps4::ThreadManager;

}  // namespace

int main() {
    ps4::MemoryManager memory;
    CHECK(memory.initialize());
    TM tm(memory);

    // 1. Recursive (default): nests, and over-unlocking is refused.
    {
        const ps4::u32 m = tm.mutex_create("rec");
        CHECK(tm.mutex_lock_errno(m) == 0);
        CHECK(tm.mutex_lock_errno(m) == 0);
        CHECK(tm.mutex_trylock_errno(m) == 0);
        CHECK(tm.mutex_unlock_errno(m) == 0);
        CHECK(tm.mutex_unlock_errno(m) == 0);
        CHECK(tm.mutex_unlock_errno(m) == 0);
        CHECK(tm.mutex_unlock_errno(m) == TM::kMutexEperm);
        CHECK(tm.mutex_destroy(m));
    }

    // 2. Non-recursive: re-lock by the owner is EDEADLK, trylock EBUSY, and a
    //    single unlock releases it.
    {
        const ps4::u32 m = tm.mutex_create("errchk", /*recursive=*/false);
        CHECK(tm.mutex_lock_errno(m) == 0);
        CHECK(tm.mutex_lock_errno(m) == TM::kMutexEdeadlk);
        CHECK(tm.mutex_trylock_errno(m) == TM::kMutexEbusy);
        CHECK(tm.mutex_unlock_errno(m) == 0);
        CHECK(tm.mutex_unlock_errno(m) == TM::kMutexEperm);

        // Another thread can take it now.
        std::atomic<int> other_lock{-1};
        std::thread t([&] {
            other_lock = tm.mutex_lock_errno(m);
            if (other_lock == 0) tm.mutex_unlock_errno(m);
        });
        t.join();
        CHECK(other_lock == 0);
        CHECK(tm.mutex_destroy(m));
    }

    // 3. Unlock by a thread that does not hold the mutex is EPERM and does not
    //    release it.
    {
        const ps4::u32 m = tm.mutex_create("owned");
        CHECK(tm.mutex_lock_errno(m) == 0);
        std::atomic<int> other_unlock{-1}, other_try{-1};
        std::thread t([&] {
            other_unlock = tm.mutex_unlock_errno(m);
            other_try = tm.mutex_trylock_errno(m);
        });
        t.join();
        CHECK(other_unlock == TM::kMutexEperm);
        CHECK(other_try == TM::kMutexEbusy);
        CHECK(tm.mutex_unlock_errno(m) == 0);
        CHECK(tm.mutex_destroy(m));
    }

    // 4. cond_wait on a recursive mutex locked twice releases it completely,
    //    so another thread can lock it, and restores both levels afterwards.
    {
        const ps4::u32 m = tm.mutex_create("cv_rec");
        const ps4::u32 cv = tm.cond_create("cv");
        std::atomic<bool> waiting{false}, ready{false};
        std::atomic<int> unlocks_ok{0}, extra_unlock{-1};
        std::thread waiter([&] {
            CHECK(tm.mutex_lock_errno(m) == 0);
            CHECK(tm.mutex_lock_errno(m) == 0);
            waiting = true;
            while (!ready) tm.cond_wait(cv, m);
            unlocks_ok += tm.mutex_unlock_errno(m) == 0;
            unlocks_ok += tm.mutex_unlock_errno(m) == 0;
            extra_unlock = tm.mutex_unlock_errno(m);
        });
        while (!waiting) std::this_thread::yield();
        // Before the fix the waiter kept one level, so this never succeeded.
        bool got = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!got && std::chrono::steady_clock::now() < deadline) {
            got = tm.mutex_trylock_errno(m) == 0;
            if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(got);
        ready = true;
        CHECK(tm.cond_broadcast(cv));
        CHECK(tm.mutex_unlock_errno(m) == 0);
        waiter.join();
        CHECK(unlocks_ok == 2);
        CHECK(extra_unlock == TM::kMutexEperm);
        CHECK(tm.cond_destroy(cv));
        CHECK(tm.mutex_destroy(m));
    }

    // 5. HLE path: an attr with type 4 (adaptive) gives a non-recursive mutex,
    //    and the default attr still gives a recursive one.
    {
        ps4::NidRegistry registry;
        registry.register_default_libraries();
        const auto* attr_init = registry.find_by_nid(0x17C6D41F0006DBCEULL);  // scePthreadMutexattrInit
        const auto* settype = registry.find_by_nid(0x88CA7C42913E5CEEULL);    // scePthreadMutexattrSettype
        const auto* init = registry.find_by_nid(0x726A3544862F6BDAULL);       // scePthreadMutexInit
        const auto* lock = registry.find_by_nid(0xE86895318FDC64A0ULL);       // scePthreadMutexLock
        const auto* unlock = registry.find_by_nid(0xEFB2023594895C85ULL);     // scePthreadMutexUnlock
        CHECK(attr_init && settype && init && lock && unlock);

        constexpr ps4::GuestAddr kAttr = 0x400100, kMutexA = 0x400200, kMutexB = 0x400300;
        auto call = [&](const ps4::HleSymbol* s, ps4::u64 a0, ps4::u64 a1 = 0) {
            ps4::CpuContext c{};
            c.rdi = a0;
            c.rsi = a1;
            s->handler(c, memory);
            return c.rax;
        };
        constexpr ps4::u64 kEdeadlk = 0x8002000BULL;

        CHECK(call(attr_init, kAttr) == 0);
        CHECK(call(settype, kAttr, 4) == 0);
        CHECK(call(settype, kAttr, 9) != 0);  // invalid type rejected
        CHECK(call(init, kMutexA, kAttr) == 0);
        CHECK(call(lock, kMutexA) == 0);
        CHECK(call(lock, kMutexA) == kEdeadlk);
        CHECK(call(unlock, kMutexA) == 0);

        CHECK(call(init, kMutexB, 0) == 0);  // no attr: recursive, as before
        CHECK(call(lock, kMutexB) == 0);
        CHECK(call(lock, kMutexB) == 0);
        CHECK(call(unlock, kMutexB) == 0);
        CHECK(call(unlock, kMutexB) == 0);
    }

    std::cout << "test_mutex_semantics: OK\n";
    return 0;
}
