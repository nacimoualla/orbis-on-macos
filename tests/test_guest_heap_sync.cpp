// HLE allocation and one-time-initialization primitives under concurrency.
//
//  - __cxa_guard_acquire/release/abort: a second thread must wait while the
//    first constructs the static, and must not see the "initialized" byte set
//    until release (setting it in acquire let other threads use half-built
//    objects).
//  - sceLibcMspaceMemalign/Free: concurrent callers get distinct,
//    non-overlapping blocks inside the mspace heap, and freed blocks are reused
//    (it was a racy, never-reusing bump pointer that grew into thread stacks).
//  - sceKernelGetDirectMemorySize / sceKernelAllocateDirectMemory: a real size,
//    and a request that does not fit fails instead of wrapping the allocator.

#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace {

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

constexpr ps4::Nid kGuardAcquire = 0xDC63E98D0740313CULL;
constexpr ps4::Nid kGuardRelease = 0xF6B01E00D4F6B721ULL;
constexpr ps4::Nid kGuardAbort = 0xD9E99A6A5B96CD4CULL;
constexpr ps4::Nid kMspaceMemalign = 0x885D62407CF10495ULL;
constexpr ps4::Nid kMspaceFree = 0x5656BF67E797971AULL;
constexpr ps4::Nid kGetDirectMemorySize = 0xA4EF7A4F0CCE9B91ULL;
constexpr ps4::Nid kAllocateDirectMemory = 0xAD35F0EB9C662C80ULL;

constexpr ps4::GuestAddr kGuardA = 0x400300;
constexpr ps4::GuestAddr kGuardB = 0x400340;
constexpr ps4::GuestAddr kOut = 0x400380;

constexpr ps4::GuestAddr kMspaceLo = 0x180000000ULL;
constexpr ps4::GuestAddr kMspaceHi = 0x1C0000000ULL;

ps4::u64 call(const ps4::HleSymbol* s, ps4::MemoryManager& mem, ps4::u64 a0, ps4::u64 a1 = 0, ps4::u64 a2 = 0,
              ps4::u64 a3 = 0, ps4::u64 a4 = 0, ps4::u64 a5 = 0) {
    ps4::CpuContext c{};
    c.rdi = a0; c.rsi = a1; c.rdx = a2; c.rcx = a3; c.r8 = a4; c.r9 = a5;
    s->handler(c, mem);
    return c.rax;
}

}  // namespace

int main() {
    ps4::MemoryManager memory;
    CHECK(memory.initialize());
    ps4::SyscallTable syscalls(memory);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false);
    ps4::NidRegistry registry;
    registry.register_default_libraries();

    const auto* acquire = registry.find_by_nid(kGuardAcquire);
    const auto* release = registry.find_by_nid(kGuardRelease);
    const auto* abort_ = registry.find_by_nid(kGuardAbort);
    const auto* memalign = registry.find_by_nid(kMspaceMemalign);
    const auto* mfree = registry.find_by_nid(kMspaceFree);
    const auto* dm_size = registry.find_by_nid(kGetDirectMemorySize);
    const auto* dm_alloc = registry.find_by_nid(kAllocateDirectMemory);
    CHECK(acquire && release && abort_ && memalign && mfree && dm_size && dm_alloc);

    // 1. Guards: the second acquirer waits for release and then sees the
    //    object as constructed.
    {
        CHECK(memory.write_value<ps4::u64>(kGuardA, 0));
        CHECK(call(acquire, memory, kGuardA) == 1);
        ps4::u8 done = 0xff;
        CHECK(memory.read_value(kGuardA, done) && done == 0);  // not "initialized" yet

        std::atomic<bool> returned{false};
        ps4::u64 other = 99;
        std::thread t([&] {
            other = call(acquire, memory, kGuardA);
            returned = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(!returned.load());  // still waiting for the constructor
        CHECK(call(release, memory, kGuardA) == 0);
        t.join();
        CHECK(other == 0);
        CHECK(memory.read_value(kGuardA, done) && done == 1);
        CHECK(call(acquire, memory, kGuardA) == 0);  // fast path once constructed
        std::cout << "  ✓ __cxa_guard_acquire waits for release\n";
    }

    // 2. Abort: a waiting thread takes over construction.
    {
        CHECK(memory.write_value<ps4::u64>(kGuardB, 0));
        CHECK(call(acquire, memory, kGuardB) == 1);
        ps4::u64 other = 99;
        std::thread t([&] { other = call(acquire, memory, kGuardB); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(call(abort_, memory, kGuardB) == 0);
        t.join();
        CHECK(other == 1);
        CHECK(call(release, memory, kGuardB) == 0);
        std::cout << "  ✓ __cxa_guard_abort hands construction to a waiter\n";
    }

    // 3. mspace blocks from concurrent threads never overlap and stay in the
    //    mspace heap.
    std::vector<std::pair<ps4::GuestAddr, ps4::GuestSize>> blocks;
    {
        constexpr int kThreads = 8, kPerThread = 2000;
        std::vector<std::vector<std::pair<ps4::GuestAddr, ps4::GuestSize>>> per(kThreads);
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; ++t) {
            ts.emplace_back([&, t] {
                for (int i = 0; i < kPerThread; ++i) {
                    const ps4::GuestSize size = 16 + ((t * 7919 + i * 104729) % 5000);
                    const ps4::GuestAddr p = call(memalign, memory, 1, 64, size);
                    per[t].emplace_back(p, size);
                }
            });
        }
        for (auto& th : ts) th.join();
        for (auto& v : per) blocks.insert(blocks.end(), v.begin(), v.end());
        std::sort(blocks.begin(), blocks.end());
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            CHECK(blocks[i].first != 0);
            CHECK(blocks[i].first % 64 == 0);
            CHECK(blocks[i].first >= kMspaceLo && blocks[i].first + blocks[i].second <= kMspaceHi);
            if (i) CHECK(blocks[i - 1].first + blocks[i - 1].second <= blocks[i].first);
        }
        std::cout << "  ✓ " << blocks.size() << " concurrent mspace blocks, no overlap\n";
    }

    // 4. Freed mspace blocks are reused.
    {
        const ps4::GuestAddr highest = blocks.back().first;
        for (const auto& [p, n] : blocks) call(mfree, memory, 1, p);
        for (const auto& [p, n] : blocks) {
            const ps4::GuestAddr q = call(memalign, memory, 1, 64, n);
            CHECK(q != 0 && q <= highest);
        }
        std::cout << "  ✓ freed mspace blocks are reused\n";
    }

    // 5. Direct memory: a real size, and an impossible request fails without
    //    moving the allocator.
    {
        CHECK(call(dm_size, memory, 0) == 0x13C000000ULL);
        CHECK(call(dm_alloc, memory, 0, 0, 0xffffffff97ff0000ULL, 0x10000, 0, kOut) == 0x80020023ULL);
        CHECK(memory.write_value<ps4::u64>(kOut, 0));
        CHECK(call(dm_alloc, memory, 0, 0, 0x100000, 0x10000, 0, kOut) == 0);
        ps4::u64 phys = 0;
        CHECK(memory.read_value(kOut, phys));
        CHECK(phys >= 0x40000000ULL && phys + 0x100000 <= 0x40000000ULL + 0x13C000000ULL);
        std::cout << "  ✓ direct memory size and overflow check\n";
    }

    std::cout << "test_guest_heap_sync: OK\n";
    return 0;
}
