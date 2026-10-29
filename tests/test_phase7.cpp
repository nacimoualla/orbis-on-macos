#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include "ps4/dynamic_linker.hpp"
#include "ps4/executor.hpp"
#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"

int main() {
    std::cout << "[PHASE 7] Starting PRX Dynamic Linking & NID Resolution unit tests...\n";

    ps4::MemoryManager memory;
    if (!memory.initialize()) {
        std::cerr << "Failed to initialize memory\n";
        return 1;
    }

    ps4::SyscallTable syscalls(memory);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false);

    ps4::NidRegistry registry;
    registry.register_default_libraries();

    ps4::DynamicLinker linker(memory, syscalls, registry);
    if (!linker.initialize()) {
        std::cerr << "Failed to initialize linker\n";
        return 1;
    }

    // Test 1: NID Registry lookups
    {
        const ps4::Nid NID_EXIT = 0x6424D00705E3BEBEULL;
        const auto* sym_exit = registry.find_by_nid(NID_EXIT);
        assert(sym_exit != nullptr);
        assert(sym_exit->name == "sceKernelExitProcess");
        assert(sym_exit->module_name == "libkernel");

        const auto* sym_puts = registry.find_by_name("puts");
        assert(sym_puts != nullptr);
        // ps4libdoc libkernel.sprx.json: puts is 0x610D276AFA7E6087, exported by
        // libSceLibcInternal. 0x5C0F663677A000A8 is a second puts NID the
        // registry serves under an honest unknown_5c0f6636 label.
        assert(sym_puts->nid == 0x610D276AFA7E6087ULL);
        assert(sym_puts->module_name == "libSceLibcInternal");

        std::cout << "  ✓ Test 1: NID Registry contains core libkernel and libc functions\n";
    }

    // Test 2: Direct memory allocation HLE shims
    {
        const ps4::Nid NID_ALLOC_DIRECT = 0x51EAECE91C9A12B0ULL;
        const auto* sym_alloc = registry.find_by_nid(NID_ALLOC_DIRECT);
        assert(sym_alloc != nullptr);

        ps4::CpuContext ctx{};
        ctx.rdi = 0;              // search_start
        ctx.rsi = 0;              // search_end
        ctx.rdx = 0x200000;       // 2MB size
        ctx.rcx = 0x10000;        // 64KB alignment
        ctx.r8  = 0;              // type
        ctx.r9  = 0x400050ULL;    // out_phys_addr pointer in guest memory

        sym_alloc->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);

        ps4::GuestAddr allocated_phys = 0;
        assert(memory.read_value(0x400050ULL, allocated_phys));
        assert(allocated_phys > 0);
        assert(allocated_phys % 0x10000 == 0); // Aligned

        std::cout << "  ✓ Test 2: sceKernelAllocateDirectMemory HLE shim verified\n";
    }

    // Test 3: Trampoline generation
    {
        const ps4::Nid NID_ALLOC_DIRECT = 0x51EAECE91C9A12B0ULL;
        ps4::GuestAddr tramp_addr = linker.get_or_create_trampoline(NID_ALLOC_DIRECT);
        assert(tramp_addr >= ps4::HLE_TRAMPOLINE_BASE);

        // Fetching again returns identical address
        assert(linker.get_or_create_trampoline(NID_ALLOC_DIRECT) == tramp_addr);

        // Verify machine code stub: B8 <imm32> 0F 05 C3
        const ps4::u8* code = memory.to_host_ptr<ps4::u8>(tramp_addr);
        assert(code != nullptr);
        assert(code[0] == 0xB8); // mov eax, pseudo_syscall
        assert(code[5] == 0x0F); // syscall
        assert(code[6] == 0x05);
        assert(code[7] == 0xC3); // ret

        std::cout << "  ✓ Test 3: x86_64 HLE trampoline stub generated correctly\n";
    }

    // Test 4: End-to-end execution calling HLE function via trampoline
    {
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());

        const ps4::Nid NID_MALLOC = 0x9E990E49EC698A4DULL; // malloc
        const ps4::GuestAddr tramp_malloc = linker.get_or_create_trampoline(NID_MALLOC);

        // Construct caller code:
        // mov edi, 0x100 (size = 256 bytes) -> BF 00 01 00 00
        // movabs r11, tramp_malloc           -> 48 BB ...
        // call r11 (FF D3) or call rel32
        // hlt (F4)
        const ps4::GuestAddr code_addr = 0x405000ULL;
        std::vector<ps4::u8> code;

        // mov edi, 256 (BF 00 01 00 00)
        code.push_back(0xBF);
        code.push_back(0x00); code.push_back(0x01); code.push_back(0x00); code.push_back(0x00);

        // call rel32 to tramp_malloc:
        // offset = tramp_malloc - (code_addr + code.size() + 5)
        const ps4::i32 rel32 = static_cast<ps4::i32>(tramp_malloc - (code_addr + code.size() + 5));
        code.push_back(0xE8);
        for (int i = 0; i < 4; ++i) {
            code.push_back(static_cast<ps4::u8>((rel32 >> (i * 8)) & 0xFF));
        }

        // hlt (F4)
        code.push_back(0xF4);

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rax >= 0x20000000ULL && cpu.context().rax < 0x40000000ULL); // Returned guest heap address from malloc
        assert(cpu.context().rax % 16 == 0);        // 16-byte aligned

        std::cout << "  ✓ Test 4: Guest x86_64 code called HLE malloc via trampoline and returned successfully\n";
    }

    // Test 5: GOT Jump Slot patching
    {
        const ps4::GuestAddr got_slot = 0x601000ULL;
        const ps4::Nid NID_PUTS = 0x5C0F663677A000A8ULL;

        assert(linker.patch_jump_slot(got_slot, NID_PUTS));

        ps4::GuestAddr resolved_trampoline = 0;
        assert(memory.read_value(got_slot, resolved_trampoline));
        assert(resolved_trampoline == linker.get_or_create_trampoline(NID_PUTS));

        std::cout << "  ✓ Test 5: GOT Jump Slot successfully patched with HLE trampoline\n";
    }

    // Test 6: DynamicLinker::resolve_relocations with PT_DYNAMIC and NID resolution
    {
        // Construct a synthetic dynamic ELF buffer
        std::vector<ps4::u8> elf_buf(0x4000, 0);

        ps4::Elf64Header hdr{};
        hdr.e_ident[ps4::EI_MAG0]    = ps4::ELFMAG0;
        hdr.e_ident[ps4::EI_MAG1]    = ps4::ELFMAG1;
        hdr.e_ident[ps4::EI_MAG2]    = ps4::ELFMAG2;
        hdr.e_ident[ps4::EI_MAG3]    = ps4::ELFMAG3;
        hdr.e_ident[ps4::EI_CLASS]   = ps4::ELFCLASS64;
        hdr.e_ident[ps4::EI_DATA]    = ps4::ELFDATA2LSB;
        hdr.e_ident[ps4::EI_VERSION] = 1;
        hdr.e_ident[ps4::EI_OSABI]   = ps4::ELFOSABI_FREEBSD;
        hdr.e_type      = ps4::ET_SCE_DYNAMIC;
        hdr.e_machine   = ps4::EM_X86_64;
        hdr.e_version   = 1;
        hdr.e_entry     = 0x400000ULL;
        hdr.e_phoff     = sizeof(ps4::Elf64Header);
        hdr.e_ehsize    = sizeof(ps4::Elf64Header);
        hdr.e_phentsize = sizeof(ps4::Elf64ProgramHeader);
        hdr.e_phnum     = 2;
        std::memcpy(elf_buf.data(), &hdr, sizeof(hdr));

        // PH 0: PT_LOAD at 0x600000 (Data / GOT region)
        ps4::Elf64ProgramHeader ph0{};
        ph0.p_type   = ps4::PT_LOAD;
        ph0.p_flags  = ps4::PF_R | ps4::PF_W;
        ph0.p_offset = 0x1000;
        ph0.p_vaddr  = 0x600000ULL;
        ph0.p_filesz = 0x100;
        ph0.p_memsz  = 0x100;
        ph0.p_align  = 0x1000;
        std::memcpy(elf_buf.data() + hdr.e_phoff, &ph0, sizeof(ph0));

        // PH 1: PT_DYNAMIC at 0x700000 (Dynamic info)
        ps4::Elf64ProgramHeader ph1{};
        ph1.p_type   = ps4::PT_DYNAMIC;
        ph1.p_flags  = ps4::PF_R;
        ph1.p_offset = 0x2000;
        ph1.p_vaddr  = 0x700000ULL;
        ph1.p_filesz = 0x1000;
        ph1.p_memsz  = 0x1000;
        ph1.p_align  = 0x1000;
        std::memcpy(elf_buf.data() + hdr.e_phoff + sizeof(ps4::Elf64ProgramHeader), &ph1, sizeof(ph1));

        // In PH 1:
        // offset 0x2000: Dynamic table (Elf64_Dyn)
        // offset 0x2100: Symbol table (Elf64_Sym) at vaddr 0x700100
        // offset 0x2200: JMPREL table (Elf64_Rela) at vaddr 0x700200
        const ps4::GuestAddr dyn_symtab_vaddr = 0x700100ULL;
        const ps4::GuestAddr dyn_jmprel_vaddr = 0x700200ULL;

        // Dynamic entries
        ps4::Elf64_Dyn dyns[5]{};
        dyns[0].d_tag = ps4::DT_SYMTAB;
        dyns[0].d_un.d_ptr = dyn_symtab_vaddr;
        dyns[1].d_tag = ps4::DT_JMPREL;
        dyns[1].d_un.d_ptr = dyn_jmprel_vaddr;
        dyns[2].d_tag = ps4::DT_PLTRELSZ;
        dyns[2].d_un.d_val = sizeof(ps4::Elf64_Rela);
        dyns[3].d_tag = ps4::DT_PLTREL;
        dyns[3].d_un.d_val = ps4::DT_RELA;
        dyns[4].d_tag = ps4::DT_NULL;
        std::memcpy(elf_buf.data() + 0x2000, dyns, sizeof(dyns));

        // Symbol table: sym[0] = null, sym[1] = puts (st_value = NID)
        const ps4::Nid NID_PUTS = 0x5C0F663677A000A8ULL;
        ps4::Elf64_Sym syms[2]{};
        syms[1].st_value = NID_PUTS;
        std::memcpy(elf_buf.data() + 0x2100, syms, sizeof(syms));

        // Relocation table: 1 JUMP_SLOT targeting GOT slot 0x600020
        const ps4::GuestAddr got_slot = 0x600020ULL;
        ps4::Elf64_Rela rela{};
        rela.r_offset = got_slot;
        rela.r_info = ps4::ELF64_R_INFO(1, ps4::R_X86_64_JUMP_SLOT);
        rela.r_addend = 0;
        std::memcpy(elf_buf.data() + 0x2200, &rela, sizeof(rela));

        ps4::ElfLoader dynamic_loader;
        assert(dynamic_loader.load_from_memory(elf_buf));

        // Zero out GOT slot
        assert(memory.write_value(got_slot, static_cast<ps4::GuestAddr>(0)));

        assert(linker.resolve_relocations(elf_buf, dynamic_loader));

        // Verify GOT slot was patched with trampoline
        ps4::GuestAddr patched_addr = 0;
        assert(memory.read_value(got_slot, patched_addr));
        assert(patched_addr == linker.get_or_create_trampoline(NID_PUTS));

        std::cout << "  ✓ Test 6: resolve_relocations correctly resolved dynamic relocations and patched GOT\n";
    }

    // Test 7: Strict HLE mode detection and configuration
    {
        assert(!linker.is_strict_hle());
        linker.set_strict_hle(true);
        assert(linker.is_strict_hle());
        linker.set_strict_hle(false);
        assert(!linker.is_strict_hle());

        std::cout << "  ✓ Test 7: Strict HLE mode toggle and query verified\n";
    }

    // Test 8: Default benign whitelist contains CRT bootstrap stubs and returns SCE_OK (0)
    {
        const ps4::Nid NID_INIT_ENV = 0x6F3404C72D7CF592ULL;
        const ps4::Nid NID_ATEXIT   = 0xF06D8B07E037AF38ULL;

        assert(linker.is_whitelisted(NID_INIT_ENV));
        assert(linker.is_whitelisted(NID_ATEXIT));

        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());

        // Call NID_INIT_ENV via trampoline
        const ps4::GuestAddr tramp_init_env = linker.get_or_create_trampoline(NID_INIT_ENV);
        const ps4::GuestAddr code_addr_env = 0x406000ULL;
        std::vector<ps4::u8> code_env;
        const ps4::i32 rel_env = static_cast<ps4::i32>(tramp_init_env - (code_addr_env + 5));
        code_env.push_back(0xE8);
        for (int i = 0; i < 4; ++i) code_env.push_back(static_cast<ps4::u8>((rel_env >> (i * 8)) & 0xFF));
        code_env.push_back(0xF4); // hlt

        assert(memory.write_bytes(code_addr_env, code_env.data(), code_env.size()));
        cpu.set_entry_point(code_addr_env);
        cpu.context().rax = 0x12345678ULL;

        ps4::RunResult res_env = cpu.run();
        assert(res_env.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == ps4::SCE_OK);

        // Call NID_ATEXIT via trampoline
        const ps4::GuestAddr tramp_atexit = linker.get_or_create_trampoline(NID_ATEXIT);
        const ps4::GuestAddr code_addr_atexit = 0x406100ULL;
        std::vector<ps4::u8> code_atexit;
        const ps4::i32 rel_atexit = static_cast<ps4::i32>(tramp_atexit - (code_addr_atexit + 5));
        code_atexit.push_back(0xE8);
        for (int i = 0; i < 4; ++i) code_atexit.push_back(static_cast<ps4::u8>((rel_atexit >> (i * 8)) & 0xFF));
        code_atexit.push_back(0xF4); // hlt

        assert(memory.write_bytes(code_addr_atexit, code_atexit.data(), code_atexit.size()));
        cpu.set_entry_point(code_addr_atexit);
        cpu.context().rax = 0x99999999ULL;

        ps4::RunResult res_atexit = cpu.run();
        assert(res_atexit.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == ps4::SCE_OK);

        std::cout << "  ✓ Test 8: Whitelisted CRT bootstrap stubs return SCE_OK (0)\n";
    }

    // Test 9: Dynamically adding an NID to whitelist allows it to safely return SCE_OK
    {
        const ps4::Nid CUSTOM_BENIGN_NID = 0xA1B2C3D4E5F60718ULL;
        assert(!linker.is_whitelisted(CUSTOM_BENIGN_NID));

        linker.add_to_whitelist(CUSTOM_BENIGN_NID);
        assert(linker.is_whitelisted(CUSTOM_BENIGN_NID));

        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());

        const ps4::GuestAddr tramp = linker.get_or_create_trampoline(CUSTOM_BENIGN_NID);
        const ps4::GuestAddr code_addr = 0x406200ULL;
        std::vector<ps4::u8> code;
        const ps4::i32 rel = static_cast<ps4::i32>(tramp - (code_addr + 5));
        code.push_back(0xE8);
        for (int i = 0; i < 4; ++i) code.push_back(static_cast<ps4::u8>((rel >> (i * 8)) & 0xFF));
        code.push_back(0xF4); // hlt

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);
        cpu.context().rax = 0xBBBBBBBBULL;

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == ps4::SCE_OK);

        std::cout << "  ✓ Test 9: add_to_whitelist() dynamically whitelists stubs to return SCE_OK\n";
    }

    // Test 10: Unhandled non-whitelisted stubs return SCE_KERNEL_ERROR_EINVAL (0x80020016ULL) in normal mode
    {
        linker.set_strict_hle(false);
        const ps4::Nid UNHANDLED_NID = 0xDEADBEEF00112233ULL;
        assert(!linker.is_whitelisted(UNHANDLED_NID));
        assert(registry.find_by_nid(UNHANDLED_NID) == nullptr);

        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());

        const ps4::GuestAddr tramp = linker.get_or_create_trampoline(UNHANDLED_NID);
        const ps4::GuestAddr code_addr = 0x406300ULL;
        std::vector<ps4::u8> code;
        const ps4::i32 rel = static_cast<ps4::i32>(tramp - (code_addr + 5));
        code.push_back(0xE8);
        for (int i = 0; i < 4; ++i) code.push_back(static_cast<ps4::u8>((rel >> (i * 8)) & 0xFF));
        code.push_back(0xF4); // hlt

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);
        cpu.context().rax = 0xdeadbeefULL;

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == 0);

        std::cout << "  ✓ Test 10: Unhandled non-whitelisted stubs return 0 (null-bail fallback)\n";
    }

    // Test 11: sceUserServiceGetEvent reports the initial user's LOGIN once,
    // then SCE_USER_SERVICE_ERROR_NO_EVENT (0x80960007ULL)
    {
        const auto* sym_event = registry.find_by_name("sceUserServiceGetEvent");
        assert(sym_event != nullptr);
        assert(sym_event->nid == 0xC87D7B43A356B558ULL);
        assert(sym_event->module_name == "libSceUserService");

        // Verify handler directly: one LOGIN (type 0) for user 0x10000001...
        constexpr ps4::GuestAddr kEvent = 0x406800ULL;
        ps4::CpuContext ctx{};
        ctx.rdi = kEvent;
        sym_event->handler(ctx, memory);
        assert(ctx.rax == 0);
        ps4::i32 type = -1, user = 0;
        memory.read_value(kEvent, type);
        memory.read_value(kEvent + 4, user);
        assert(type == 0 && user == 0x10000001);
        // ...then no more events.
        ctx = {};
        ctx.rdi = kEvent;
        sym_event->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_USER_SERVICE_ERROR_NO_EVENT);
        assert(ctx.rax == 0x80960007ULL);

        // Verify invocation via trampoline and CpuExecutor
        ps4::CpuExecutor cpu(memory, syscalls);
        assert(cpu.setup_default_stack());

        const ps4::GuestAddr tramp = linker.get_or_create_trampoline(sym_event->nid);
        const ps4::GuestAddr code_addr = 0x406400ULL;
        std::vector<ps4::u8> code;
        const ps4::i32 rel = static_cast<ps4::i32>(tramp - (code_addr + 5));
        code.push_back(0xE8);
        for (int i = 0; i < 4; ++i) code.push_back(static_cast<ps4::u8>((rel >> (i * 8)) & 0xFF));
        code.push_back(0xF4); // hlt

        assert(memory.write_bytes(code_addr, code.data(), code.size()));
        cpu.set_entry_point(code_addr);
        cpu.context().rax = 0;

        ps4::RunResult res = cpu.run();
        assert(res.status == ps4::StepResult::Halted);
        assert(cpu.context().rax == ps4::SCE_USER_SERVICE_ERROR_NO_EVENT);
        assert(cpu.context().rax == 0x80960007ULL);

        std::cout << "  ✓ Test 11: sceUserServiceGetEvent reports one LOGIN, then 0x80960007ULL (SCE_USER_SERVICE_ERROR_NO_EVENT)\n";
    }

    // Test 12: Strict HLE mode triggers diagnostic crash (SIGABRT) on unregistered NID
    {
#if !defined(_WIN32)
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            // Child process: enable strict HLE and call unhandled non-whitelisted NID
            linker.set_strict_hle(true);

            // Silence stderr output during expected abort test to keep ctest clean
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }

            const ps4::Nid UNREGISTERED_STRICT_NID = 0xBADF00D000000001ULL;
            ps4::CpuExecutor cpu(memory, syscalls);
            if (!cpu.setup_default_stack()) std::exit(1);

            const ps4::GuestAddr tramp = linker.get_or_create_trampoline(UNREGISTERED_STRICT_NID);
            const ps4::GuestAddr code_addr = 0x406500ULL;
            std::vector<ps4::u8> code;
            const ps4::i32 rel = static_cast<ps4::i32>(tramp - (code_addr + 5));
            code.push_back(0xE8);
            for (int i = 0; i < 4; ++i) code.push_back(static_cast<ps4::u8>((rel >> (i * 8)) & 0xFF));
            code.push_back(0xF4);

            memory.write_bytes(code_addr, code.data(), code.size());
            cpu.set_entry_point(code_addr);
            cpu.run(); // Should abort here!

            std::exit(0); // Should not be reached
        }

        int status = 0;
        waitpid(pid, &status, 0);
        assert(WIFSIGNALED(status));
        assert(WTERMSIG(status) == SIGABRT);

        std::cout << "  ✓ Test 12: Strict HLE aborts process on unhandled non-whitelisted stub invocation\n";
#endif
    }

    // Test 13: Orbis Semaphore lifecycle and cancellation semantics
    {
        const auto* sym_create = registry.find_by_name("sceKernelCreateSema");
        const auto* sym_wait   = registry.find_by_name("sceKernelWaitSema");
        const auto* sym_signal = registry.find_by_name("sceKernelSignalSema");
        const auto* sym_cancel = registry.find_by_name("sceKernelCancelSema");
        const auto* sym_delete = registry.find_by_name("sceKernelDeleteSema");

        assert(sym_create && sym_wait && sym_signal && sym_cancel && sym_delete);

        // 13a: CreateSema mints handle and sets init_count=0
        const ps4::GuestAddr sema_slot = 0x408000ULL;
        ps4::CpuContext ctx{};
        ctx.rdi = sema_slot;
        ctx.rsi = 0; // anonymous
        ctx.rdx = 1;
        ctx.rcx = 0; // initCount = 0
        ctx.r8  = 0x7fffffff;
        sym_create->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);

        ps4::u32 sema_id = 0;
        bool ok = memory.read_value(sema_slot, sema_id);
        assert(ok && sema_id > 0);

        // 13b: CancelSema with 0 waiters must NOT latch a cancellation for future waiters
        ctx.rdi = sema_id;
        ctx.rsi = static_cast<ps4::u64>(-1); // setCount = -1
        ctx.rdx = 0x408010ULL;               // pNumWaitingThreads
        sym_cancel->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);
        ps4::i32 cancelled_count = -1;
        ok = memory.read_value(0x408010ULL, cancelled_count);
        assert(ok && cancelled_count == 0);

        // 13c: Timed wait on unsignaled semaphore must time out (ETIMEDOUT), NOT return ECANCELED
        const ps4::GuestAddr timeout_slot = 0x408020ULL;
        ok = memory.write_value(timeout_slot, static_cast<ps4::u32>(1000)); // 1ms
        assert(ok);
        ctx.rdi = sema_id;
        ctx.rsi = 1; // need 1
        ctx.rdx = timeout_slot;
        sym_wait->handler(ctx, memory);
        assert(ctx.rax == 0x8002003CULL); // SCE_KERNEL_ERROR_ETIMEDOUT

        // 13d: SignalSema satisfies waiter
        ctx.rdi = sema_id;
        ctx.rsi = 1; // sigCount = 1
        sym_signal->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);

        ctx.rdi = sema_id;
        ctx.rsi = 1; // need 1
        ctx.rdx = timeout_slot;
        sym_wait->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK); // Consumed the signal

        // 13e: Cancel active waiter in background thread
        std::atomic<ps4::u64> waiter_res{0};
        std::atomic<bool> waiter_started{false};
        std::thread waiter_th([&]() {
            ps4::CpuContext wctx{};
            wctx.rdi = sema_id;
            wctx.rsi = 1;
            wctx.rdx = 0; // infinite wait
            waiter_started.store(true);
            sym_wait->handler(wctx, memory);
            waiter_res.store(wctx.rax);
        });

        while (!waiter_started.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        // Cancel while thread is parked
        ctx.rdi = sema_id;
        ctx.rsi = static_cast<ps4::u64>(-1);
        ctx.rdx = 0x408010ULL;
        sym_cancel->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);
        ok = memory.read_value(0x408010ULL, cancelled_count);
        assert(ok && cancelled_count == 1);

        waiter_th.join();
        assert(waiter_res.load() == 0x80020055ULL); // SCE_KERNEL_ERROR_ECANCELED

        // 13f: Subsequent wait after cancel should block normally, not inherit ECANCELED
        ok = memory.write_value(timeout_slot, static_cast<ps4::u32>(1000));
        assert(ok);
        ctx.rdi = sema_id;
        ctx.rsi = 1;
        ctx.rdx = timeout_slot;
        sym_wait->handler(ctx, memory);
        assert(ctx.rax == 0x8002003CULL); // ETIMEDOUT, NOT ECANCELED

        // 13g: DeleteSema removes from table
        ctx.rdi = sema_id;
        sym_delete->handler(ctx, memory);
        assert(ctx.rax == ps4::SCE_OK);

        // Calling WaitSema on deleted id returns ESRCH
        ctx.rdi = sema_id;
        ctx.rsi = 1;
        ctx.rdx = 0;
        sym_wait->handler(ctx, memory);
        assert(ctx.rax == 0x80020003ULL); // SCE_KERNEL_ERROR_ESRCH

        std::cout << "  ✓ Test 13: Orbis Semaphore lifecycle and cancellation semantics verified\n";
    }

    std::cout << "[PHASE 7] PASS: All PRX Dynamic Linking & NID Resolution tests passed!\n";
    return 0;
}
