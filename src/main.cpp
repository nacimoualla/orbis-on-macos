#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include "ps4/common.hpp"
#include "ps4/cpu_context.hpp"
#include "ps4/display_window.hpp"
#include "ps4/dynamic_linker.hpp"
#include "ps4/elf_loader.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/vfs.hpp"

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << " ps4-to-macos: Orbis Compatibility Layer & Runtime\n";
    std::cout << "========================================================\n";

    bool enable_window = false;
    std::filesystem::path elf_path;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--window" || arg == "-w") {
            enable_window = true;
        } else if (arg == "--headless") {
            enable_window = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ps4-runner [options] <path-to-eboot.bin-or-elf>\n";
            std::cout << "Options:\n";
            std::cout << "  --window, -w     Open interactive macOS display window (60 FPS)\n";
            std::cout << "  --headless       Run headlessly in console only (default)\n";
            return 0;
        } else {
            elf_path = argv[i];
        }
    }

    if (elf_path.empty()) {
        std::cout << "Usage: ps4-runner [--window|-w] <path-to-eboot.bin-or-elf>\n";
        return 0;
    }
    std::cout << "Loading: " << elf_path << "\n\n";

    // 1. Read binary file data into memory buffer
    std::ifstream file(elf_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[Error] Cannot open file: " << elf_path << "\n";
        return 1;
    }

    const auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<ps4::u8> file_data(static_cast<ps4::usize>(file_size));
    if (!file.read(reinterpret_cast<char*>(file_data.data()), file_size)) {
        std::cerr << "[Error] Failed to read file data from disk.\n";
        return 1;
    }

    // 2. Parse ELF headers
    ps4::ElfLoader loader;
    if (!loader.load_from_memory(file_data)) {
        std::cerr << "[Error] Failed to load binary: " << loader.error_message() << "\n";
        return 1;
    }

    loader.dump_info();

    // 3. Initialize 8GB Virtual Memory Space
    std::cout << "\n[Memory] Initializing 8GB unified virtual address space...\n";
    ps4::MemoryManager memory;
    if (!memory.initialize()) {
        std::cerr << "[Error] Failed to initialize virtual memory: " << memory.error_message() << "\n";
        return 1;
    }
    std::cout << "[Memory] Base Host Address: " << static_cast<void*>(memory.base_address()) << "\n";

    // 4. Map PT_LOAD segments into guest virtual address space
    std::cout << "[Memory] Mapping " << loader.load_segments().size() << " loadable segments into guest memory...\n";
    if (!memory.load_elf_segments(loader, file_data)) {
        std::cerr << "[Error] Failed to map ELF segments into memory: " << memory.error_message() << "\n";
        return 1;
    }
    std::cout << "[Memory] All segments loaded successfully.\n";

    // 5. Initialize Syscall Dispatch Table & Virtual File System
    std::cout << "[HLE] Initializing Orbis system call table & Virtual File System...\n";
    ps4::VirtualFileSystem vfs;
    ps4::SyscallTable syscalls(memory);
    syscalls.set_vfs(&vfs);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false); // Controlled shutdown

    // 6. Initialize NID Registry & Dynamic Linker
    std::cout << "[Linker] Initializing NID Registry and Dynamic Linker...\n";
    ps4::NidRegistry nid_registry;
    nid_registry.register_default_libraries();

    ps4::DynamicLinker linker(memory, syscalls, nid_registry);
    if (!linker.initialize()) {
        std::cerr << "[Error] Failed to initialize dynamic linker trampoline pool.\n";
        return 1;
    }

    linker.resolve_relocations(file_data, loader);
    std::cout << "[Linker] Resolved symbols via NID registry (Active trampolines: "
              << linker.trampoline_count() << ")\n";

    // 7. Initialize Apple Silicon Metal GPU Subsystem
    std::cout << "[Metal] Initializing GPU backend...\n";
    ps4::MetalBackend metal;
    if (metal.initialize()) {
        std::cout << "[Metal] GPU Device: " << metal.device_name() << "\n";
        if (metal.create_framebuffer(1920, 1080)) {
            std::cout << "[Metal] 1920x1080 BGRA8Unorm Framebuffer allocated in unified memory.\n";
            // PlayStation Deep Blue clear: R=0.0f, G=0.15f, B=0.45f
            if (metal.clear_screen({0.0f, 0.15f, 0.45f, 1.0f})) {
                std::cout << "[Metal] GPU command buffer executed clear pass.\n";
            }
        }
    } else {
        std::cout << "[Metal] Note: Metal device initialization skipped or unavailable.\n";
    }

    // 8. CPU Execution
    std::cout << "\n[CPU] Initializing execution engine & ARM64 JIT dynamic recompiler...\n";
    ps4::JitCompiler jit;
    ps4::CpuExecutor cpu(memory, syscalls);
    // cpu.set_jit_compiler(&jit);
    if (!cpu.setup_default_stack()) {
        std::cerr << "[Error] Failed to allocate stack.\n";
        return 1;
    }
    cpu.set_entry_point(loader.entry_point());
    std::cout << "[CPU] RIP set to entry point: 0x" << std::hex << loader.entry_point() << std::dec << "\n";
    std::cout << "[CPU] RSP initialized to:     0x" << std::hex << cpu.context().rsp << std::dec << "\n";

    std::cout << "\n================= Guest Execution Output =================\n";
    for (int i = 0; i < 20; ++i) {
        ps4::GuestAddr cur_rip = cpu.context().rip;
        ps4::StepResult sr = cpu.step();
        std::cout << "Step " << i << ": RIP=0x" << std::hex << cur_rip << " -> 0x" << cpu.context().rip 
                  << " (" << ps4::step_result_to_string(sr) << ")\n" << std::dec;
        if (sr != ps4::StepResult::Ok) break;
    }
    ps4::RunResult run_res{};
    ps4::usize total_instructions = 0;
    const ps4::usize batch = 1000000;
    const ps4::usize max_total = 20000000;
    while (total_instructions < max_total) {
        run_res = cpu.run(batch);
        total_instructions += run_res.instructions_executed;
        if (run_res.status != ps4::StepResult::Ok) break;
        std::cout << "[CPU Progress] " << total_instructions << " instrs, RIP=0x" << std::hex << cpu.context().rip << std::dec << "\n";
    }
    std::cout << "==========================================================\n";

    std::cout << "\n[CPU] Execution Status: " << ps4::step_result_to_string(run_res.status) << "\n";
    std::cout << "[CPU] Instructions Executed: " << total_instructions << "\n";
    if (run_res.status == ps4::StepResult::Exited) {
        std::cout << "[CPU] Process Exit Code: " << run_res.exit_code << "\n";
    }

    cpu.context().dump();

    // 9. Interactive Metal Display Window Presentation
    if (enable_window && metal.is_initialized()) {
        std::cout << "\n[Display] Opening interactive macOS Metal window...\n";
        std::cout << "[Display] Running presentation loop at 60 FPS (Press ESC or close window to exit)\n";
        ps4::DisplayWindow window;
        ps4::WindowConfig win_cfg{
            .title = "ps4-to-macos: " + elf_path.filename().string() + " (Metal 3)",
            .width = 1280,
            .height = 720,
            .resizable = true,
            .headless = false
        };

        if (window.initialize(metal, win_cfg)) {
            while (window.is_open()) {
                window.poll_events();
                window.present_frame();
                std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
            }
            std::cout << "[Display] Window closed cleanly.\n";
        }
    }

    return 0;
}
