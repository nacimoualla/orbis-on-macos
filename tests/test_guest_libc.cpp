// Runs the guest-resident libc routines (include/ps4/guest_libc_blob.hpp) on
// the emulated CPU, interpreter and JIT, with a guest comparator: qsort must
// sort and bsearch must find present keys and miss absent ones.

#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

#include "ps4/executor.hpp"
#include "ps4/guest_libc_blob.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"

using namespace ps4;

namespace {
constexpr GuestAddr kCode = 0x50000;    // blob
constexpr GuestAddr kCmp = 0x58000;     // int cmp(const int* a, const int* b)
constexpr GuestAddr kHalt = 0x58100;    // return address: HLT
constexpr GuestAddr kArray = 0x60000;
constexpr GuestAddr kKey = 0x61000;

// mov eax,[rdi]; mov ecx,[rsi]; cmp eax,ecx; setg al; setl cl; sub al,cl; movsx eax,al; ret
const u8 kCmpCode[] = {0x8B, 0x07, 0x8B, 0x0E, 0x39, 0xC8, 0x0F, 0x9F, 0xC0, 0x0F, 0x9C,
                       0xC1, 0x28, 0xC8, 0x0F, 0xBE, 0xC0, 0xC3};

struct Guest {
    MemoryManager memory;
    SyscallTable syscalls{memory};
    CpuExecutor cpu{memory, syscalls};
    JitCompiler jit;

    explicit Guest(bool use_jit) {
        assert(memory.initialize());
        assert(cpu.setup_default_stack());
        if (use_jit) cpu.set_jit_compiler(&jit);
        assert(memory.write_bytes(kCode, guest_libc::kCode, sizeof(guest_libc::kCode)));
        assert(memory.write_bytes(kCmp, kCmpCode, sizeof(kCmpCode)));
        const u8 hlt = 0xF4;
        assert(memory.write_bytes(kHalt, &hlt, 1));
    }

    // Call fn(args...) and run until it returns to the HLT.
    u64 call(GuestAddr fn, u64 a0, u64 a1, u64 a2, u64 a3, u64 a4) {
        auto& c = cpu.context();
        c.rdi = a0; c.rsi = a1; c.rdx = a2; c.rcx = a3; c.r8 = a4;
        c.rsp = (c.rsp & ~0xFULL) - 8;
        assert(memory.write_value(c.rsp, static_cast<u64>(kHalt)));
        c.rip = fn;
        const RunResult r = cpu.run(10'000'000);
        assert(r.status == StepResult::Halted && c.rip == kHalt);
        return c.rax;
    }
};

void run(bool use_jit) {
    Guest g(use_jit);
    std::vector<i32> v = {42, -7, 19, 0, 1000, 3, 3, -100, 77, 5, 64, -1, 8, 2, 11, 9, 13};
    const usize n = v.size();
    assert(g.memory.write_bytes(kArray, v.data(), n * 4));

    g.call(kCode + guest_libc::k_qsort_offset, kArray, n, 4, kCmp, 0);
    std::vector<i32> sorted(n);
    assert(g.memory.read_bytes(kArray, sorted.data(), n * 4));
    for (usize i = 1; i < n; ++i) assert(sorted[i - 1] <= sorted[i]);

    for (usize i = 0; i < n; ++i) {
        assert(g.memory.write_value(kKey, sorted[i]));
        const u64 hit = g.call(kCode + guest_libc::k_bsearch_offset, kKey, kArray, n, 4, kCmp);
        assert(hit != 0);
        i32 found = 0;
        assert(g.memory.read_value(hit, found) && found == sorted[i]);
    }
    for (const i32 missing : {-101, 4, 1001, 10}) {
        assert(g.memory.write_value(kKey, missing));
        assert(g.call(kCode + guest_libc::k_bsearch_offset, kKey, kArray, n, 4, kCmp) == 0);
    }
    // Degenerate inputs
    assert(g.call(kCode + guest_libc::k_bsearch_offset, kKey, kArray, 0, 4, kCmp) == 0);
    g.call(kCode + guest_libc::k_qsort_offset, kArray, 1, 4, kCmp, 0);
    std::cout << "  -> " << (use_jit ? "JIT" : "interpreter") << " passed\n";
}
} // namespace

int main() {
    std::cout << "=== Guest-resident qsort/bsearch ===\n";
    run(false);
    run(true);
    std::cout << "All guest libc tests passed.\n";
    return 0;
}
