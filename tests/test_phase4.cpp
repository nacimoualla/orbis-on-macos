#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"

int main() {
    std::cout << "[PHASE 4] Starting Syscall Dispatcher unit tests...\n";

    ps4::MemoryManager memory;
    assert(memory.initialize());

    ps4::SyscallTable syscalls(memory);
    syscalls.set_terminate_on_exit(false); // Do not terminate process in unit tests

    // Test 1: Handler registration
    {
        assert(!syscalls.is_registered(ps4::SYS_exit));
        syscalls.register_default_handlers();
        assert(syscalls.is_registered(ps4::SYS_exit));
        assert(syscalls.is_registered(ps4::SYS_write));
        assert(syscalls.is_registered(ps4::SYS_getpid));
        assert(syscalls.is_registered(ps4::SYS_mprotect));
        assert(!syscalls.is_registered(999));
        std::cout << "  ✓ Test 1: Default handlers registered correctly\n";
    }

    // Test 2: SYS_write with valid guest buffer to redirected stdout
    {
        std::ostringstream captured_output;
        syscalls.set_output_stream(&captured_output);

        const char message[] = "Hello PlayStation 4 on macOS!\n";
        const ps4::usize msg_len = std::strlen(message);
        const ps4::GuestAddr buf_addr = 0x401000ULL;

        assert(memory.write_bytes(buf_addr, message, msg_len));

        ps4::SyscallContext ctx{};
        ctx.args[0] = 1;        // fd = stdout
        ctx.args[1] = buf_addr; // buffer in guest RAM
        ctx.args[2] = msg_len;  // length

        ps4::OrbisResult res = syscalls.dispatch(ps4::SYS_write, ctx);
        assert(res == 0);
        assert(ctx.rax == msg_len);
        assert(captured_output.str() == message);

        std::cout << "  ✓ Test 2: SYS_write output redirected and captured successfully\n";
    }

    // Test 3: SYS_write with out-of-bounds guest buffer (EFAULT)
    {
        ps4::SyscallContext ctx{};
        ctx.args[0] = 1;
        ctx.args[1] = ps4::GUEST_MEMORY_SIZE + 0x100; // invalid guest address
        ctx.args[2] = 10;

        ps4::OrbisResult res = syscalls.dispatch(ps4::SYS_write, ctx);
        assert(res == -14); // -EFAULT
        assert(ctx.rax == static_cast<ps4::u64>(-14));
        std::cout << "  ✓ Test 3: SYS_write out-of-bounds address returns -EFAULT\n";
    }

    // Test 4: SYS_write with invalid file descriptor (EBADF)
    {
        ps4::SyscallContext ctx{};
        ctx.args[0] = 99; // invalid fd
        ctx.args[1] = 0x401000ULL;
        ctx.args[2] = 5;

        ps4::OrbisResult res = syscalls.dispatch(ps4::SYS_write, ctx);
        assert(res == -9); // -EBADF
        assert(ctx.rax == static_cast<ps4::u64>(-9));
        std::cout << "  ✓ Test 4: SYS_write invalid fd returns -EBADF\n";
    }

    // Test 5: SYS_exit
    {
        assert(!syscalls.has_exited());

        ps4::SyscallContext ctx{};
        ctx.args[0] = 42; // exit code 42

        ps4::OrbisResult res = syscalls.dispatch(ps4::SYS_exit, ctx);
        assert(res == 0);
        assert(syscalls.has_exited());
        assert(syscalls.exit_code() == 42);
        std::cout << "  ✓ Test 5: SYS_exit triggers exit flag with status 42\n";
    }

    // Test 6: SYS_getpid
    {
        ps4::SyscallContext ctx{};
        ps4::OrbisResult res = syscalls.dispatch(ps4::SYS_getpid, ctx);
        assert(res == 0);
        assert(ctx.rax == 1001);
        std::cout << "  ✓ Test 6: SYS_getpid returns mock process ID\n";
    }

    // Test 7: Unimplemented syscall (ENOSYS)
    {
        ps4::SyscallContext ctx{};
        ps4::OrbisResult res = syscalls.dispatch(999, ctx);
        assert(res == -38); // -ENOSYS
        assert(ctx.rax == static_cast<ps4::u64>(-38));
        std::cout << "  ✓ Test 7: Unimplemented syscall returns -ENOSYS\n";
    }

    // Test 8: Custom handler registration
    {
        constexpr ps4::u32 CUSTOM_SYSCALL = 500;
        bool custom_called = false;

        syscalls.register_handler(CUSTOM_SYSCALL, [&](ps4::SyscallContext& ctx) -> ps4::OrbisResult {
            custom_called = true;
            ctx.rax = ctx.args[0] * 2;
            return 0;
        });

        ps4::SyscallContext ctx{};
        ctx.args[0] = 21;
        ps4::OrbisResult res = syscalls.dispatch(CUSTOM_SYSCALL, ctx);
        assert(res == 0);
        assert(custom_called);
        assert(ctx.rax == 42);
        std::cout << "  ✓ Test 8: Custom syscall handler dispatched successfully\n";
    }

    std::cout << "[PHASE 4] PASS: All Syscall Dispatcher tests passed!\n";
    return 0;
}
