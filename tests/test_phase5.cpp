#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include "ps4/executor.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"

int main() {
    std::cout << "[PHASE 5] Starting CPU Execution and Instruction Dispatch unit tests...\n";

    ps4::MemoryManager memory;
    assert(memory.initialize());

    ps4::SyscallTable syscalls(memory);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false);

    // VHADDPS xmm0, xmm1, xmm2 (VEX.128.F2.0F 7C /r)
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        const ps4::GuestAddr code_addr = 0x403000ULL;
        const ps4::u8 code[] = {0xC5, 0xF3, 0x7C, 0xC2, 0xF4};
        assert(memory.write_bytes(code_addr, code, sizeof(code)));
        cpu.context().ymm[1].f[0] = 1.0f;
        cpu.context().ymm[1].f[1] = 2.0f;
        cpu.context().ymm[1].f[2] = 3.0f;
        cpu.context().ymm[1].f[3] = 4.0f;
        cpu.context().ymm[2].f[0] = 5.0f;
        cpu.context().ymm[2].f[1] = 6.0f;
        cpu.context().ymm[2].f[2] = 7.0f;
        cpu.context().ymm[2].f[3] = 8.0f;
        cpu.set_entry_point(code_addr);
        assert(cpu.step() == ps4::StepResult::Ok);
        assert(cpu.context().ymm[0].f[0] == 3.0f);
        assert(cpu.context().ymm[0].f[1] == 7.0f);
        assert(cpu.context().ymm[0].f[2] == 11.0f);
        assert(cpu.context().ymm[0].f[3] == 15.0f);
        std::cout << "  ✓ VHADDPS VEX map=1 opcode 0x7c verified\n";
    }

    // VMOVLPS m64, xmm1 (VEX.128.0F 13 /r).  The VEX.vvvv field is reserved
    // and VEX.L=1 is invalid for this 64-bit store form.
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        const ps4::GuestAddr code_addr = 0x403100ULL;
        const ps4::GuestAddr data_addr = 0x600000ULL;
        const ps4::u8 code[] = {
            0xC5, 0xF8, 0x13, 0x08, // vmovlps [rax], xmm1
            0xF4
        };
        assert(memory.write_bytes(code_addr, code, sizeof(code)));
        cpu.context().rax = data_addr;
        cpu.context().ymm[1].q[0] = 0x1122334455667788ULL;
        cpu.context().ymm[1].q[1] = 0xAABBCCDDEEFF0011ULL;
        assert(memory.write_value(data_addr, 0xDEADBEEFDEADBEEFULL));
        cpu.set_entry_point(code_addr);
        assert(cpu.step() == ps4::StepResult::Ok);
        ps4::u64 stored = 0;
        assert(memory.read_value(data_addr, stored));
        assert(stored == 0x1122334455667788ULL);

        const ps4::GuestAddr invalid_rip = cpu.context().rip;
        const ps4::u8 invalid_256[] = {
            0xC5, 0xFC, 0x13, 0x08 // VEX.L=1: invalid VMOVLPS encoding
        };
        assert(memory.write_bytes(invalid_rip, invalid_256, sizeof(invalid_256)));
        assert(cpu.step() == ps4::StepResult::IllegalInstruction);
        assert(cpu.context().rip == invalid_rip);
        assert(memory.read_value(data_addr, stored));
        assert(stored == 0x1122334455667788ULL);
        std::cout << "  ✓ VMOVLPS VEX map=1 opcode 0x13 128-bit store and 256-bit rejection verified\n";
    }

    // Test 1: Stack initialization
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());
        assert(cpu.context().rsp % 16 == 0); // 16-byte aligned
        assert(cpu.context().rsp > 0x1FFF00000ULL);
        assert(cpu.context().rsp <= ps4::GUEST_MEMORY_SIZE);
        std::cout << "  ✓ Test 1: Stack allocated and 16-byte aligned\n";
    }

    // Test 2: NOP and HLT
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        const ps4::GuestAddr code_addr = 0x400000ULL;
        // 0x90 (NOP), 0x90 (NOP), 0xF4 (HLT)
        const ps4::u8 code[] = {0x90, 0x90, 0xF4};
        assert(memory.write_bytes(code_addr, code, sizeof(code)));

        cpu.set_entry_point(code_addr);
        ps4::StepResult s1 = cpu.step();
        assert(s1 == ps4::StepResult::Ok);
        assert(cpu.context().rip == code_addr + 1);

        ps4::StepResult s2 = cpu.step();
        assert(s2 == ps4::StepResult::Ok);
        assert(cpu.context().rip == code_addr + 2);

        ps4::StepResult s3 = cpu.step();
        assert(s3 == ps4::StepResult::Halted);
        assert(cpu.context().rip == code_addr + 2);

        std::cout << "  ✓ Test 2: NOP stepping and HLT termination verified\n";
    }

    // Test 3: Register manipulation (MOV, XOR)
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        const ps4::GuestAddr code_addr = 0x401000ULL;
        std::vector<ps4::u8> code;

        // mov eax, 0x12345678 (B8 78 56 34 12)
        code.push_back(0xB8);
        code.push_back(0x78); code.push_back(0x56); code.push_back(0x34); code.push_back(0x12);

        // mov edi, 0xCAFEBABE (BF BE BA FE CA)
        code.push_back(0xBF);
        code.push_back(0xBE); code.push_back(0xBA); code.push_back(0xFE); code.push_back(0xCA);

        // movabs rsi, 0x1122334455667788 (48 BE 88 77 66 55 44 33 22 11)
        code.push_back(0x48); code.push_back(0xBE);
        code.push_back(0x88); code.push_back(0x77); code.push_back(0x66); code.push_back(0x55);
        code.push_back(0x44); code.push_back(0x33); code.push_back(0x22); code.push_back(0x11);

        // xor edx, edx (31 D2)
        code.push_back(0x31); code.push_back(0xD2);

        // hlt (F4)
        code.push_back(0xF4);

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == 0x12345678ULL);
        assert(cpu.context().rdi == 0xCAFEBABEULL);
        assert(cpu.context().rsi == 0x1122334455667788ULL);
        assert(cpu.context().rdx == 0ULL);

        std::cout << "  ✓ Test 3: MOV (32/64-bit) and XOR instructions verified\n";
    }

    // Test 4: CALL and RET control flow
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());
        const ps4::GuestAddr code_addr = 0x402000ULL;
        std::vector<ps4::u8> code(64, 0x90);

        // At 0x402000: call +10 bytes (E8 0A 00 00 00) -> jumps to 0x40200F
        code[0] = 0xE8;
        code[1] = 0x0A; code[2] = 0x00; code[3] = 0x00; code[4] = 0x00;
        // At 0x402005: hlt (F4)
        code[5] = 0xF4;

        // Subroutine at 0x40200F (offset 15):
        // mov eax, 0x42 (B8 42 00 00 00)
        code[15] = 0xB8;
        code[16] = 0x42; code[17] = 0x00; code[18] = 0x00; code[19] = 0x00;
        // ret (C3)
        code[20] = 0xC3;

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == 0x42);
        assert(cpu.context().rip == code_addr + 5); // Stopped at HLT after returning

        std::cout << "  ✓ Test 4: CALL and RET subroutines verified\n";
    }

    // Test 5: Complete guest program executing SYS_write and SYS_exit
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());

        std::ostringstream captured;
        syscalls.set_output_stream(&captured);

        const char msg[] = "Orbis Native Code Execution Success!\n";
        const ps4::usize msg_len = std::strlen(msg);
        const ps4::GuestAddr msg_addr = 0x500000ULL;
        assert(memory.write_bytes(msg_addr, msg, msg_len));

        const ps4::GuestAddr code_addr = 0x403000ULL;
        std::vector<ps4::u8> code;

        // mov eax, 4 (SYS_write)
        code.push_back(0xB8);
        code.push_back(0x04); code.push_back(0x00); code.push_back(0x00); code.push_back(0x00);

        // mov edi, 1 (stdout)
        code.push_back(0xBF);
        code.push_back(0x01); code.push_back(0x00); code.push_back(0x00); code.push_back(0x00);

        // movabs rsi, msg_addr (48 BE ...)
        code.push_back(0x48); code.push_back(0xBE);
        for (int i = 0; i < 8; ++i) {
            code.push_back(static_cast<ps4::u8>((msg_addr >> (i * 8)) & 0xFF));
        }

        // mov edx, msg_len
        code.push_back(0xBA);
        code.push_back(static_cast<ps4::u8>(msg_len & 0xFF));
        code.push_back(0x00); code.push_back(0x00); code.push_back(0x00);

        // syscall (0F 05)
        code.push_back(0x0F); code.push_back(0x05);

        // mov eax, 1 (SYS_exit)
        code.push_back(0xB8);
        code.push_back(0x01); code.push_back(0x00); code.push_back(0x00); code.push_back(0x00);

        // mov edi, 0 (exit status 0)
        code.push_back(0xBF);
        code.push_back(0x00); code.push_back(0x00); code.push_back(0x00); code.push_back(0x00);

        // syscall (0F 05)
        code.push_back(0x0F); code.push_back(0x05);

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Exited);
        assert(res.exit_code == 0);
        assert(captured.str() == msg);

        std::cout << "  ✓ Test 5: Full guest x86_64 program executed SYS_write and SYS_exit\n";
    }

    // Test 6: Stack operations (PUSH reg, PUSH imm, POP reg, LEAVE)
    {
        syscalls.reset_exit();
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());
        const ps4::GuestAddr initial_rsp = cpu.context().rsp;
        const ps4::GuestAddr code_addr = 0x404000ULL;
        std::vector<ps4::u8> code;

        // movabs rax, 0x1122334455667788 (48 B8 ...)
        code.push_back(0x48); code.push_back(0xB8);
        const ps4::u64 magic_val = 0x1122334455667788ULL;
        for (int i = 0; i < 8; ++i) code.push_back(static_cast<ps4::u8>((magic_val >> (i * 8)) & 0xFF));

        // push rax (50)
        code.push_back(0x50);
        // push 0x42 (6A 42)
        code.push_back(0x6A); code.push_back(0x42);
        // pop rbx (5B)
        code.push_back(0x5B);
        // pop rcx (59)
        code.push_back(0x59);
        // hlt (F4)
        code.push_back(0xF4);

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        std::cout << "Test 6 status: " << ps4::step_result_to_string(res.status) 
                  << " rip: 0x" << std::hex << cpu.context().rip << std::dec << "\n";
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rbx == 0x42);
        assert(cpu.context().rcx == magic_val);
        assert(cpu.context().rsp == initial_rsp); // Stack balanced

        std::cout << "  ✓ Test 6: Stack operations (PUSH, POP, stack balance) verified\n";
    }

    // Test 7: Memory moves, ModR/M, SIB, and LEA
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());
        const ps4::GuestAddr data_addr = 0x600000ULL;
        const ps4::GuestAddr code_addr = 0x405000ULL;
        std::vector<ps4::u8> code;

        // movabs rdi, data_addr (48 BF ...)
        code.push_back(0x48); code.push_back(0xBF);
        for (int i = 0; i < 8; ++i) code.push_back(static_cast<ps4::u8>((data_addr >> (i * 8)) & 0xFF));

        // mov qword ptr [rdi], 0x12345678 (48 C7 07 78 56 34 12)
        code.push_back(0x48); code.push_back(0xC7); code.push_back(0x07);
        code.push_back(0x78); code.push_back(0x56); code.push_back(0x34); code.push_back(0x12);

        // mov rax, [rdi] (48 8B 07)
        code.push_back(0x48); code.push_back(0x8B); code.push_back(0x07);

        // lea rdx, [rdi + 0x30] (48 8D 57 30)
        code.push_back(0x48); code.push_back(0x8D); code.push_back(0x57); code.push_back(0x30);

        // hlt (F4)
        code.push_back(0xF4);

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == 0x12345678ULL);
        assert(cpu.context().rdx == data_addr + 0x30);

        std::cout << "  ✓ Test 7: Memory stores, loads, and LEA addressing verified\n";
    }

    // Test 8: ALU operations (ADD, SUB, CMP, TEST) and flags
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());
        const ps4::GuestAddr code_addr = 0x406000ULL;
        std::vector<ps4::u8> code;

        // mov eax, 100 (B8 64 00 00 00)
        code.push_back(0xB8); code.push_back(100); code.push_back(0); code.push_back(0); code.push_back(0);

        // add eax, 50 (83 C0 32)
        code.push_back(0x83); code.push_back(0xC0); code.push_back(50);

        // sub eax, 30 (83 E8 1E)
        code.push_back(0x83); code.push_back(0xE8); code.push_back(30);

        // cmp eax, 120 (83 F8 78) -> ZF must be 1
        code.push_back(0x83); code.push_back(0xF8); code.push_back(120);

        // hlt (F4)
        code.push_back(0xF4);

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == 120);
        assert(cpu.context().flag_zf()); // Compare succeeded -> ZF is set

        std::cout << "  ✓ Test 8: ALU arithmetic (ADD, SUB, CMP) and flags verified\n";
    }

    // Test 9: Conditional jumps (JZ / JNZ / Jcc)
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());
        const ps4::GuestAddr code_addr = 0x407000ULL;
        std::vector<ps4::u8> code;

        // xor eax, eax (31 C0) -> ZF = 1
        code.push_back(0x31); code.push_back(0xC0);

        // jz +5 (74 05) -> skip 5 bytes
        code.push_back(0x74); code.push_back(0x05);

        // mov ebx, 999 (BB E7 03 00 00) - should be skipped!
        code.push_back(0xBB); code.push_back(0xE7); code.push_back(0x03); code.push_back(0x00); code.push_back(0x00);

        // mov ebx, 42 (BB 2A 00 00 00) - destination!
        code.push_back(0xBB); code.push_back(0x2A); code.push_back(0x00); code.push_back(0x00); code.push_back(0x00);

        // hlt (F4)
        code.push_back(0xF4);

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rbx == 42); // Successfully jumped past 999

        std::cout << "  ✓ Test 9: Conditional jumps (Jcc branching) verified\n";
    }

    // Test 10: REX prefix & Extended GPRs (R8 - R15)
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());
        const ps4::GuestAddr code_addr = 0x408000ULL;
        std::vector<ps4::u8> code;

        // mov r14, 0x1000 (49 C7 C6 00 10 00 00)
        code.push_back(0x49); code.push_back(0xC7); code.push_back(0xC6);
        code.push_back(0x00); code.push_back(0x10); code.push_back(0x00); code.push_back(0x00);

        // mov r15, 0x2000 (49 C7 C7 00 20 00 00)
        code.push_back(0x49); code.push_back(0xC7); code.push_back(0xC7);
        code.push_back(0x00); code.push_back(0x20); code.push_back(0x00); code.push_back(0x00);

        // add r14, r15 (4D 01 FE)
        code.push_back(0x4D); code.push_back(0x01); code.push_back(0xFE);

        // mov rax, r14 (4C 89 F0)
        code.push_back(0x4C); code.push_back(0x89); code.push_back(0xF0);

        // hlt (F4)
        code.push_back(0xF4);

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().r14 == 0x3000ULL);
        assert(cpu.context().rax == 0x3000ULL);

        std::cout << "  ✓ Test 10: Extended registers (R8-R15) and REX prefix verified\n";
    }
    // Test 10: 4KB Orbis sub-page protections inside a single 16KB Apple Silicon host page
    {
        ps4::MemoryManager mem;
        assert(mem.initialize());

        // 0x400000 is 16KB-aligned (.rodata -> PF_R)
        assert(mem.protect(0x400000, 0x1000, ps4::PF_R));

        // 0x401000 is 4KB-aligned, NOT 16KB-aligned (.data -> PF_R | PF_W)
        // Old code fails here on Apple Silicon with EINVAL!
        assert(mem.protect(0x401000, 0x1000, ps4::PF_R | ps4::PF_W));

        // Verify we can still write to 0x401000 without a SIGBUS/SIGSEGV because
        // the 16KB host page (0x400000..0x403FFF) unioned PF_R and PF_R|PF_W -> PROT_READ|PROT_WRITE
        uint64_t test_val = 0xCAFEBABE12345678ULL;
        assert(mem.write_value(0x401000, test_val));

        uint64_t read_back = 0;
        assert(mem.read_value(0x401000, read_back));
        assert(read_back == test_val);
    }

    // Test 11: VPCMPEQD and VINSERTF128 (all-ones mask generation)
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        const ps4::GuestAddr code_addr = 0x403000ULL;
        std::vector<ps4::u8> code = {
            // vpcmpeqd xmm0, xmm0, xmm0 (C5 F9 76 C0)
            0xC5, 0xF9, 0x76, 0xC0,
            // vinsertf128 ymm1, ymm0, xmm0, 1 (C4 E3 7D 18 C8 01)
            0xC4, 0xE3, 0x7D, 0x18, 0xC8, 0x01,
            // hlt
            0xF4
        };
        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        // All 4 dwords of xmm0 should be 0xFFFFFFFF
        for (int i = 0; i < 4; ++i) {
            assert(cpu.context().ymm[0].d[i] == 0xFFFFFFFFU);
        }
        assert(cpu.context().ymm[0].q[2] == 0);
        assert(cpu.context().ymm[0].q[3] == 0);
        // All 8 dwords of ymm1 should be 0xFFFFFFFF
        for (int i = 0; i < 8; ++i) {
            assert(cpu.context().ymm[1].d[i] == 0xFFFFFFFFU);
        }
        std::cout << "  ✓ Test 11: VPCMPEQD and VINSERTF128 all-ones mask verified\n";
    }

    // Test 12: VPACKSSDW signed saturation and unsupported SIMD failure.
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        const ps4::GuestAddr code_addr = 0x409000ULL;
        const ps4::u8 code[] = {
            // VPACKSSDW xmm0, xmm1, xmm1
            0xC5, 0xF1, 0x6B, 0xC1,
            // VPACKUSWB xmm2, xmm0, xmm0: unsigned saturation of those words
            0xC5, 0xF9, 0x67, 0xD0,
            // VPSADBW is still unimplemented and must fail explicitly.
            0xC5, 0xF9, 0xF6, 0xC0
        };
        assert(memory.write_bytes(code_addr, code, sizeof(code)));
        cpu.context().ymm[1].d[0] = 0x00010000U;
        cpu.context().ymm[1].d[1] = 0xFFFFFFFFU;
        cpu.context().ymm[1].d[2] = 0x00000002U;
        cpu.context().ymm[1].d[3] = 0x80000000U;
        cpu.set_entry_point(code_addr);

        assert(cpu.step() == ps4::StepResult::Ok);
        assert(cpu.context().ymm[0].w[0] == 0x7FFF);
        assert(cpu.context().ymm[0].w[1] == 0xFFFF);
        assert(cpu.context().ymm[0].w[2] == 2);
        assert(cpu.context().ymm[0].w[3] == 0x8000);
        assert(cpu.context().ymm[0].w[4] == 0x7FFF);
        assert(cpu.context().ymm[0].w[5] == 0xFFFF);
        assert(cpu.context().ymm[0].w[6] == 2);
        assert(cpu.context().ymm[0].w[7] == 0x8000);

        assert(cpu.step() == ps4::StepResult::Ok);
        assert(cpu.context().ymm[2].b[0] == 0xFF && cpu.context().ymm[2].b[1] == 0 &&
               cpu.context().ymm[2].b[2] == 2 && cpu.context().ymm[2].b[3] == 0);
        assert(cpu.context().ymm[2].b[8] == 0xFF && cpu.context().ymm[2].q[2] == 0);

        const ps4::GuestAddr unsupported_rip = cpu.context().rip;
        assert(cpu.step() == ps4::StepResult::IllegalInstruction);
        assert(cpu.context().rip == unsupported_rip);

        std::cout << "  ✓ Test 12: VPACKSSDW, VPACKUSWB and unsupported VEX handling verified\n";
    }

    std::cout << "[PHASE 5] PASS: All CPU Execution and Instruction Dispatch tests passed!\n";
    return 0;
}
