#include "ps4/cpu_context.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"
#include <cassert>
#include <iostream>

using namespace ps4;

void test_arm64_emitter_primitives() {
    std::cout << "[Test 13.1] Testing ARM64 instruction emitter...\n";

    Arm64Emitter emitter;
    emitter.nop();
    emitter.ret();
    emitter.mov_imm64(0, 0xDEADBEEFCAFEULL);
    emitter.add_reg(0, 1, 2);
    emitter.sub_reg(0, 1, 2);
    emitter.str_zero(19, 8);
    emitter.str_reg(0, 19, 16);
    emitter.ldr_reg(0, 19, 16);

    assert(emitter.code().size() > 5);
    assert(emitter.size_bytes() == emitter.code().size() * sizeof(u32));

    std::cout << "  -> ARM64 instruction emitter primitives passed!\n";
}

void test_jit_buffer_and_execution() {
    std::cout << "[Test 13.2] Testing MAP_JIT executable allocation and native execution...\n";

    JitCompiler jit;
    assert(jit.code_buffer_used() == 0);

    // Let's manually compile a block that sets RAX = 0x1337 and RBX = 0x42
    MemoryManager memory;
    assert(memory.initialize());

    // Write x86_64 guest instructions into memory at 0x10000:
    // mov eax, 0x1337     (B8 37 13 00 00)
    // mov ebx, 0x42       (BB 42 00 00 00)
    // hlt                 (F4)
    const GuestAddr code_addr = 0x10000;
    const u8 x86_code[] = {
        0xB8, 0x37, 0x13, 0x00, 0x00, // mov eax, 0x1337
        0xBB, 0x42, 0x00, 0x00, 0x00  // mov ebx, 0x42
    };
    memory.write_bytes(code_addr, x86_code, sizeof(x86_code));
    // hlt ends the block (the zero bytes after it would otherwise decode as
    // `add [rax], al`, which the JIT translates)
    const u8 hlt = 0xF4;
    memory.write_bytes(code_addr + sizeof(x86_code), &hlt, 1);

    CpuContext ctx{};
    ctx.rip = code_addr;

    assert(!jit.is_block_cached(code_addr));
    const bool success = jit.execute_block(ctx, memory);
    assert(success);
    assert(jit.is_block_cached(code_addr));
    assert(jit.cached_block_count() == 1);
    assert(jit.total_invocations() == 1);

    // Verify registers after native ARM64 execution
    assert(ctx.rax == 0x1337);
    assert(ctx.rbx == 0x42);
    assert(ctx.rip == code_addr + sizeof(x86_code));

    const auto* metrics = jit.get_block_metrics(code_addr);
    assert(metrics != nullptr);
    assert(metrics->instructions_compiled == 2);
    assert(metrics->bytes_emitted > 0);
    assert(metrics->execution_count == 1);

    std::cout << "  -> MAP_JIT allocation and native block execution passed!\n";
}

void test_jit_arithmetic_recompiler() {
    std::cout << "[Test 13.3] Testing JIT recompilation of 64-bit arithmetic and register moves...\n";

    MemoryManager memory;
    assert(memory.initialize());

    JitCompiler jit;

    // x86_64 machine code:
    // xor eax, eax                        (31 C0)
    // movabs rdi, 0x1000000020000000      (48 BF 00 00 00 20 00 00 00 10)
    // movabs rsi, 0x0000000000000005      (48 BE 05 00 00 00 00 00 00 00)
    // add rdi, rsi                        (48 01 F7)
    // mov rax, rdi                        (done via ret end of block)
    // ret                                 (C3)
    const GuestAddr code_addr = 0x20000;
    const u8 x86_code[] = {
        0x31, 0xC0,                                                 // xor eax, eax
        0x48, 0xBF, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x10, // mov rdi, 0x1000000020000000
        0x48, 0xBE, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rsi, 5
        0x48, 0x01, 0xF7                                            // add rdi, rsi
    };
    memory.write_bytes(code_addr, x86_code, sizeof(x86_code));

    CpuContext ctx{};
    ctx.rip = code_addr;
    ctx.rax = 0x9999; // Should be cleared to 0

    const bool ok = jit.execute_block(ctx, memory);
    assert(ok);
    assert(ctx.rax == 0);
    assert(ctx.rsi == 5);
    assert(ctx.rdi == 0x1000000020000005ULL);

    // Test sub rdi, rsi
    const GuestAddr sub_code_addr = 0x21000;
    const u8 sub_x86_code[] = {
        0x48, 0x29, 0xF7 // sub rdi, rsi
    };
    memory.write_bytes(sub_code_addr, sub_x86_code, sizeof(sub_x86_code));

    ctx.rip = sub_code_addr;
    assert(jit.execute_block(ctx, memory));
    assert(ctx.rdi == 0x1000000020000000ULL);

    std::cout << "  -> JIT 64-bit arithmetic recompilation passed!\n";
}

void test_jit_executor_integration() {
    std::cout << "[Test 13.4] Testing CpuExecutor with attached JIT recompiler...\n";

    MemoryManager memory;
    assert(memory.initialize());

    SyscallTable syscalls(memory);
    syscalls.register_default_handlers();

    CpuExecutor executor(memory, syscalls);
    JitCompiler jit;
    executor.set_jit_compiler(&jit);

    // Program:
    // mov eax, 1             (SYS_exit = 1)
    // mov edi, 42            (exit code = 42)
    // syscall                (0F 05)
    const GuestAddr prog_addr = 0x30000;
    const u8 prog[] = {
        0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0xBF, 0x2A, 0x00, 0x00, 0x00, // mov edi, 42
        0x0F, 0x05                    // syscall
    };
    memory.write_bytes(prog_addr, prog, sizeof(prog));

    executor.set_entry_point(prog_addr);
    const RunResult run_res = executor.run(100);

    assert(run_res.status == StepResult::Exited);
    assert(run_res.exit_code == 42);
    assert(executor.context().rax == 0);
    assert(executor.context().rdi == 42);
    assert(jit.total_invocations() > 0);

    std::cout << "  -> CpuExecutor with JIT dynamic recompilation passed!\n";
}

int main() {
    std::cout << "=== Phase 13: ARM64 JIT Dynamic Recompiler Tests ===\n";

    test_arm64_emitter_primitives();
    test_jit_buffer_and_execution();
    test_jit_arithmetic_recompiler();
    test_jit_executor_integration();

    std::cout << "=== All Phase 13 tests passed successfully! ===\n";
    return 0;
}
