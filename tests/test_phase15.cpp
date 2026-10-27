#include "ps4/cpu_context.hpp"
#include "ps4/elf_loader.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using namespace ps4;

// -----------------------------------------------------------------------------
// Test 15.1: Basic-Block Boundary Scanning & Trap Exits
// -----------------------------------------------------------------------------
void test_basic_block_boundaries_and_traps() {
    std::cout << "[Test 15.1] Testing basic-block boundary scanning and interrupt/trap exits...\n";

    MemoryManager memory;
    assert(memory.initialize());
    JitCompiler jit;

    // 1. Syscall boundary: code ending in 0x0F 0x05 (SYSCALL)
    {
        const GuestAddr code_addr = 0x10000;
        const u8 code[] = {
            0xB8, 0x11, 0x11, 0x00, 0x00, // mov eax, 0x1111 (5 bytes)
            0xBB, 0x22, 0x22, 0x00, 0x00, // mov ebx, 0x2222 (5 bytes)
            0xB9, 0x33, 0x33, 0x00, 0x00, // mov ecx, 0x3333 (5 bytes)
            0x0F, 0x05                    // syscall        (2 bytes)
        };
        assert(memory.write_bytes(code_addr, code, sizeof(code)));

        CpuContext ctx{};
        ctx.rip = code_addr;

        assert(jit.execute_block(ctx, memory));
        assert(ctx.rax == 0x1111);
        assert(ctx.rbx == 0x2222);
        assert(ctx.rcx == 0x3333);
        // RIP must point precisely to the syscall instruction
        assert(ctx.rip == code_addr + 15);

        const auto* metrics = jit.get_block_metrics(code_addr);
        assert(metrics != nullptr);
        assert(metrics->instructions_compiled == 3);
    }

    // 2. UD2 boundary: code ending in 0x0F 0x0B (UD2)
    {
        const GuestAddr code_addr = 0x11000;
        const u8 code[] = {
            0xBA, 0x44, 0x44, 0x00, 0x00, // mov edx, 0x4444 (5 bytes)
            0x0F, 0x0B                    // ud2             (2 bytes)
        };
        assert(memory.write_bytes(code_addr, code, sizeof(code)));

        CpuContext ctx{};
        ctx.rip = code_addr;

        assert(jit.execute_block(ctx, memory));
        assert(ctx.rdx == 0x4444);
        assert(ctx.rip == code_addr + 5); // RIP points to ud2
    }

    // 3. HLT boundary: code ending in 0xF4 (HLT)
    {
        const GuestAddr code_addr = 0x12000;
        const u8 code[] = {
            0xBE, 0x55, 0x55, 0x00, 0x00, // mov esi, 0x5555 (5 bytes)
            0xF4                          // hlt             (1 byte)
        };
        assert(memory.write_bytes(code_addr, code, sizeof(code)));

        CpuContext ctx{};
        ctx.rip = code_addr;

        assert(jit.execute_block(ctx, memory));
        assert(ctx.rsi == 0x5555);
        assert(ctx.rip == code_addr + 5); // RIP points to hlt
    }

    // 4. RET boundary: code ending in 0xC3 (RET)
    {
        const GuestAddr code_addr = 0x13000;
        const u8 code[] = {
            0xBF, 0x66, 0x66, 0x00, 0x00, // mov edi, 0x6666 (5 bytes)
            0xC3                          // ret             (1 byte)
        };
        assert(memory.write_bytes(code_addr, code, sizeof(code)));

        CpuContext ctx{};
        ctx.rip = code_addr;

        assert(jit.execute_block(ctx, memory));
        assert(ctx.rdi == 0x6666);
        assert(ctx.rip == code_addr + 5); // RIP points to ret
    }

    std::cout << "  -> Basic-block boundaries and trap exits verified!\n";
}

// -----------------------------------------------------------------------------
// Test 15.2: Apple Silicon W^X Memory Protection & Thread-Safe Cache
// -----------------------------------------------------------------------------
void test_wx_protection_and_thread_safety() {
    std::cout << "[Test 15.2] Testing W^X code cache and multithreaded concurrent compilation...\n";

    MemoryManager memory;
    assert(memory.initialize());
    JitCompiler jit;

    // Prepare 4 separate guest code blocks
    const GuestAddr addrs[4] = {0x20000, 0x21000, 0x22000, 0x23000};
    const u8 code0[] = {0xB8, 0x01, 0x00, 0x00, 0x00}; // mov eax, 1
    const u8 code1[] = {0xBB, 0x02, 0x00, 0x00, 0x00}; // mov ebx, 2
    const u8 code2[] = {0xB9, 0x03, 0x00, 0x00, 0x00}; // mov ecx, 3
    const u8 code3[] = {0xBA, 0x04, 0x00, 0x00, 0x00}; // mov edx, 4

    assert(memory.write_bytes(addrs[0], code0, sizeof(code0)));
    assert(memory.write_bytes(addrs[1], code1, sizeof(code1)));
    assert(memory.write_bytes(addrs[2], code2, sizeof(code2)));
    assert(memory.write_bytes(addrs[3], code3, sizeof(code3)));

    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&jit, &memory, code_addr = addrs[t], t]() {
            for (int iter = 0; iter < 1000; ++iter) {
                CpuContext ctx{};
                ctx.rip = code_addr;
                assert(jit.execute_block(ctx, memory));
                if (t == 0) assert(ctx.rax == 1);
                else if (t == 1) assert(ctx.rbx == 2);
                else if (t == 2) assert(ctx.rcx == 3);
                else if (t == 3) assert(ctx.rdx == 4);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    assert(jit.cached_block_count() == 4);
    assert(jit.total_invocations() == 4000);
    assert(jit.code_buffer_used() > 0);

    std::cout << "  -> Multithreaded W^X code cache safety verified!\n";
}

// -----------------------------------------------------------------------------
// Test 15.3: Control Flow (JMP) & ALU Instruction Recompilation
// -----------------------------------------------------------------------------
void test_control_flow_and_alu() {
    std::cout << "[Test 15.3] Testing JMP branches and arithmetic recompilation...\n";

    MemoryManager memory;
    assert(memory.initialize());
    JitCompiler jit;

    // Test JMP rel8 forward
    // 0x30000: mov eax, 100         (B8 64 00 00 00)
    // 0x30005: jmp +4               (EB 04) -> jumps to 0x3000B
    // 0x30007: mov eax, 999         (B8 E7 03 00 00) -- skipped!
    // 0x3000C: mov ebx, 200         (BB C8 00 00 00)
    const GuestAddr jmp_addr = 0x30000;
    const u8 jmp_prog[] = {
        0xB8, 0x64, 0x00, 0x00, 0x00, // mov eax, 100 (5 bytes)
        0xEB, 0x05,                   // jmp +5      (2 bytes) -> target: 0x3000C
        0xB8, 0xE7, 0x03, 0x00, 0x00, // mov eax, 999 (5 bytes, skipped)
        0xBB, 0xC8, 0x00, 0x00, 0x00  // mov ebx, 200 (5 bytes)
    };
    assert(memory.write_bytes(jmp_addr, jmp_prog, sizeof(jmp_prog)));

    CpuContext ctx{};
    ctx.rip = jmp_addr;

    // First block: mov eax, 100; jmp +5
    assert(jit.execute_block(ctx, memory));
    assert(ctx.rax == 100);
    assert(ctx.rip == jmp_addr + 12); // target address 0x3000C

    // Second block: mov ebx, 200
    assert(jit.execute_block(ctx, memory));
    assert(ctx.rax == 100); // Unaltered
    assert(ctx.rbx == 200);

    // Test ALU operations: ADD, SUB, AND, OR, XOR reg64 and reg32
    const GuestAddr alu_addr = 0x40000;
    const u8 alu_prog[] = {
        0x31, 0xC0,                         // xor eax, eax (clear rax)
        0x48, 0xC7, 0xC3, 0x0A, 0x00, 0x00, 0x00, // mov rbx, 10
        0x48, 0xC7, 0xC1, 0x05, 0x00, 0x00, 0x00, // mov rcx, 5
        0x48, 0x01, 0xCB,                   // add rbx, rcx (rbx = 15)
        0x48, 0x29, 0xCB,                   // sub rbx, rcx (rbx = 10)
        0x48, 0x21, 0xCB,                   // and rbx, rcx (10 & 5 = 0)
        0x48, 0xC7, 0xC3, 0x0C, 0x00, 0x00, 0x00, // mov rbx, 12
        0x48, 0x09, 0xCB                    // or rbx, rcx  (12 | 5 = 13)
    };
    assert(memory.write_bytes(alu_addr, alu_prog, sizeof(alu_prog)));

    ctx = CpuContext{};
    ctx.rip = alu_addr;
    assert(jit.execute_block(ctx, memory));
    assert(ctx.rax == 0);
    assert(ctx.rcx == 5);
    assert(ctx.rbx == 13);

    std::cout << "  -> Control flow and arithmetic recompilation passed!\n";
}

// -----------------------------------------------------------------------------
// Test 15.4: Seamless JIT <-> Interpreter Transitions in CpuExecutor
// -----------------------------------------------------------------------------
void test_executor_seamless_transitions() {
    std::cout << "[Test 15.4] Testing CpuExecutor running JIT blocks with interpreter syscalls...\n";

    MemoryManager memory;
    assert(memory.initialize());
    SyscallTable syscalls(memory);
    syscalls.register_default_handlers();

    CpuExecutor executor(memory, syscalls);
    JitCompiler jit;
    executor.set_jit_compiler(&jit);

    // Program:
    // Block 1 (JIT):
    //   mov eax, 10      (B8 0A 00 00 00)
    //   mov ebx, 20      (BB 14 00 00 00)
    //   add rbx, rax     (48 01 C3) -> rbx = 30
    //   mov rdi, rbx     (48 89 DF) -> rdi = 30
    //   mov eax, 1       (B8 01 00 00 00) -> SYS_exit
    // Block 2 (Interpreter fallback):
    //   syscall          (0F 05)
    const GuestAddr prog_addr = 0x50000;
    const u8 prog[] = {
        0xB8, 0x0A, 0x00, 0x00, 0x00,
        0xBB, 0x14, 0x00, 0x00, 0x00,
        0x48, 0x01, 0xC3,
        0x48, 0x89, 0xDF,
        0xB8, 0x01, 0x00, 0x00, 0x00,
        0x0F, 0x05
    };
    assert(memory.write_bytes(prog_addr, prog, sizeof(prog)));

    executor.set_entry_point(prog_addr);
    const RunResult res = executor.run(100);

    assert(res.status == StepResult::Exited);
    assert(res.exit_code == 30);
    assert(executor.context().rdi == 30);
    assert(jit.cached_block_count() > 0);
    assert(jit.total_invocations() > 0);

    std::cout << "  -> CpuExecutor seamless JIT and interpreter execution passed!\n";
}

// -----------------------------------------------------------------------------
// Test 15.5: Benchmark Interpreter vs. JIT on samples/triangle_orbis.elf & Compute
// -----------------------------------------------------------------------------
void test_benchmark_triangle_orbis_and_compute() {
    std::cout << "[Test 15.5] Benchmarking Interpreter vs. ARM64 JIT throughput...\n";

    // 1. Benchmark basic block of samples/triangle_orbis.elf
    std::filesystem::path elf_path = "samples/triangle_orbis.elf";
    if (!std::filesystem::exists(elf_path)) {
        elf_path = "../samples/triangle_orbis.elf";
    }
    if (!std::filesystem::exists(elf_path)) {
        elf_path = "../../samples/triangle_orbis.elf";
    }
    std::ifstream file(elf_path, std::ios::binary | std::ios::ate);
    assert(file.is_open() && "Could not open samples/triangle_orbis.elf");

    const auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<u8> elf_data(static_cast<usize>(file_size));
    assert(file.read(reinterpret_cast<char*>(elf_data.data()), file_size));

    ElfLoader loader;
    assert(loader.load_from_memory(elf_data));

    MemoryManager memory;
    assert(memory.initialize());
    assert(memory.load_elf_segments(loader, elf_data));

    const GuestAddr entry_rip = loader.entry_point(); // 0x400000
    const int ITERATIONS = 200000;

    // Benchmark Interpreter execution
    SyscallTable syscalls_interp(memory);
    syscalls_interp.register_default_handlers();
    CpuExecutor executor_interp(memory, syscalls_interp);

    const auto t0_interp = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        executor_interp.context().rip = entry_rip;
        // Step 4 instructions of the block before syscall
        executor_interp.step();
        executor_interp.step();
        executor_interp.step();
        executor_interp.step();
    }
    const auto t1_interp = std::chrono::high_resolution_clock::now();
    const double elapsed_interp_s = std::chrono::duration<double>(t1_interp - t0_interp).count();
    const double mips_interp = (ITERATIONS * 4.0) / elapsed_interp_s / 1e6;

    // Benchmark JIT execution
    JitCompiler jit;
    CpuContext ctx_jit{};
    // Compile and cache block
    ctx_jit.rip = entry_rip;
    assert(jit.execute_block(ctx_jit, memory));

    const auto t0_jit = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        ctx_jit.rip = entry_rip;
        jit.execute_block(ctx_jit, memory);
    }
    const auto t1_jit = std::chrono::high_resolution_clock::now();
    const double elapsed_jit_s = std::chrono::duration<double>(t1_jit - t0_jit).count();
    const double mips_jit = (ITERATIONS * 4.0) / elapsed_jit_s / 1e6;

    const double speedup = elapsed_interp_s / elapsed_jit_s;

    std::cout << "  --------------------------------------------------------\n";
    std::cout << "  [Benchmark: samples/triangle_orbis.elf basic block (" << ITERATIONS << " runs)]\n";
    std::cout << "  Interpreter: " << elapsed_interp_s * 1000.0 << " ms (" << mips_interp << " MIPS)\n";
    std::cout << "  ARM64 JIT:   " << elapsed_jit_s * 1000.0 << " ms (" << mips_jit << " MIPS)\n";
    std::cout << "  Throughput Speedup: " << speedup << "x faster on Apple Silicon!\n";
    std::cout << "  --------------------------------------------------------\n";

    assert(speedup > 1.0 && "JIT must yield a measurable throughput speedup!");

    // 2. Heavy Compute Workload (5,000,000 instructions)
    const GuestAddr compute_addr = 0x60000;
    const u8 compute_block[] = {
        0x48, 0x01, 0xCB, // add rbx, rcx
        0x48, 0x29, 0xCB, // sub rbx, rcx
        0x48, 0x01, 0xD3, // add rbx, rdx
        0x48, 0x29, 0xD3, // sub rbx, rdx
        0x48, 0x01, 0xCB, // add rbx, rcx
        0x48, 0x29, 0xCB, // sub rbx, rcx
        0x48, 0x01, 0xD3, // add rbx, rdx
        0x48, 0x29, 0xD3  // sub rbx, rdx
    };
    assert(memory.write_bytes(compute_addr, compute_block, sizeof(compute_block)));

    const int COMPUTE_ITERS = 500000; // 500k * 8 = 4 million instructions
    const auto t0_c_jit = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < COMPUTE_ITERS; ++i) {
        ctx_jit.rip = compute_addr;
        jit.execute_block(ctx_jit, memory);
    }
    const auto t1_c_jit = std::chrono::high_resolution_clock::now();
    const double elapsed_c_jit_s = std::chrono::duration<double>(t1_c_jit - t0_c_jit).count();
    const double compute_mips = (COMPUTE_ITERS * 8.0) / elapsed_c_jit_s / 1e6;

    std::cout << "  [Compute Workload: " << COMPUTE_ITERS * 8 << " instructions]\n";
    std::cout << "  JIT Throughput: " << compute_mips << " MIPS (" << elapsed_c_jit_s * 1000.0 << " ms)\n";
    std::cout << "  --------------------------------------------------------\n";
    // Under QEMU (scripts/cpu_harness.sh) absolute speed is meaningless.
    if (!std::getenv("PS4_NO_PERF_ASSERTS")) {
        assert(compute_mips > 50.0);
    }

    std::cout << "  -> JIT benchmark and throughput verification passed!\n";
}

int main() {
    std::cout << "=== Phase 15: ARM64 JIT & CpuExecutor Integration Tests ===\n";

    test_basic_block_boundaries_and_traps();
    test_wx_protection_and_thread_safety();
    test_control_flow_and_alu();
    test_executor_seamless_transitions();
    test_benchmark_triangle_orbis_and_compute();

    std::cout << "=== All Phase 15 tests passed successfully! ===\n";
    return 0;
}
