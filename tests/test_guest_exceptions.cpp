// Guest exception delivery (sceKernelInstallExceptionHandler /
// sceKernelRaiseException): a spinning guest thread is interrupted, runs the
// handler on its own stack with (signum, ctx) and resumes where it was.

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

#include "ps4/cpu_context.hpp"
#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/syscalls.hpp"
#include "ps4/thread.hpp"

using namespace ps4;

namespace {
constexpr Nid kInstall = 0x5A4C0477737BC346ULL;
constexpr Nid kRaise = 0x8A5D379E5B8A7CC9ULL;
constexpr GuestAddr kWorker = 0x50000, kHandler = 0x51000, kData = 0x58000;

// inc qword [0x58000]; jmp $-10
const u8 kWorkerCode[] = {0x48, 0xFF, 0x04, 0x25, 0x00, 0x80, 0x05, 0x00, 0xEB, 0xF6};
// mov [0x58008], rdi; mov rax, [rsi+0xf8]; mov [0x58010], rax;
// mov rax, [0x58000]; mov [0x58018], rax; ret
const u8 kHandlerCode[] = {0x48, 0x89, 0x3C, 0x25, 0x08, 0x80, 0x05, 0x00,
                           0x48, 0x8B, 0x86, 0xF8, 0x00, 0x00, 0x00,
                           0x48, 0x89, 0x04, 0x25, 0x10, 0x80, 0x05, 0x00,
                           0x48, 0x8B, 0x04, 0x25, 0x00, 0x80, 0x05, 0x00,
                           0x48, 0x89, 0x04, 0x25, 0x18, 0x80, 0x05, 0x00,
                           0xC3};

u64 read64(MemoryManager& mem, GuestAddr a) {
    u64 v = 0;
    mem.read_value(a, v);
    return v;
}

template <typename Pred>
bool wait_until(Pred p) {
    for (int i = 0; i < 400; ++i) {
        if (p()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}
} // namespace

int main() {
    std::cout << "=== Guest exception delivery ===\n";
    MemoryManager mem;
    assert(mem.initialize());
    SyscallTable syscalls(mem);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false);
    NidRegistry registry;
    registry.set_syscall_table(&syscalls);
    registry.register_default_libraries();
    assert(mem.write_bytes(kWorker, kWorkerCode, sizeof(kWorkerCode)));
    assert(mem.write_bytes(kHandler, kHandlerCode, sizeof(kHandlerCode)));

    CpuContext ctx{};
    ctx.rdi = 30;
    ctx.rsi = kHandler;
    registry.find_by_nid(kInstall)->handler(ctx, mem);
    assert(ctx.rax == 0);

    ThreadManager& tm = registry.thread_manager(mem);
    const ScePthreadId tid = tm.create_thread(kWorker, 0, nullptr, "spinner");
    assert(tid != 0);
    assert(wait_until([&] { return read64(mem, kData) > 1000; }));

    // Unknown thread: ESRCH.
    ctx = {};
    ctx.rdi = 9999;
    ctx.rsi = 30;
    registry.find_by_nid(kRaise)->handler(ctx, mem);
    assert(ctx.rax == 0x80020003ULL);

    ctx = {};
    ctx.rdi = tid;
    ctx.rsi = 30;
    registry.find_by_nid(kRaise)->handler(ctx, mem);
    assert(ctx.rax == 0);
    assert(wait_until([&] { return read64(mem, kData + 8) == 30; }));

    // The handler saw the interrupted thread's stack pointer at ctx+0xf8.
    const GuestThread* th = tm.get_thread(tid);
    const u64 sp = read64(mem, kData + 0x10);
    assert(th != nullptr && sp > th->stack_base && sp <= th->stack_base + th->stack_size);

    // ...and the thread resumed its loop afterwards.
    const u64 at_handler = read64(mem, kData + 0x18);
    assert(wait_until([&] { return read64(mem, kData) > at_handler + 1000; }));
    std::cout << "  -> handler ran with signum 30 and rsp 0x" << std::hex << sp << std::dec
              << "; thread resumed\n";

    SyscallContext exit_ctx(mem);
    syscalls.dispatch(SYS_exit, exit_ctx);
    registry.shutdown();
    std::cout << "All guest exception tests passed.\n";
    return 0;
}
