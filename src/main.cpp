#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include "ps4/common.hpp"
#include "ps4/cpu_context.hpp"
#include "ps4/display_window.hpp"
#include "ps4/pm4_translator.hpp"
#include "ps4/dynamic_linker.hpp"
#include "ps4/elf_loader.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/thread.hpp"
#include "ps4/vfs.hpp"

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << " ps4-to-macos: Orbis Compatibility Layer & Runtime\n";
    std::cout << "========================================================\n";

    bool enable_window = false;
    bool enable_jit = true;
    bool enable_trace = false;
    bool enable_strict_hle = false;
    bool budget_specified = false;
    // Headless instruction budget. The default (100M) stops inside the guest's
    // static-initializer phase, well before Unity's Mono bootstrap, so soak and
    // smoke runs that need the Mono call census must raise it.
    ps4::usize instruction_budget = 100000000;
    std::filesystem::path elf_path;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--window" || arg == "-w") {
            enable_window = true;
        } else if (arg == "--headless") {
            enable_window = false;
        } else if (arg == "--no-jit") {
            enable_jit = false;
        } else if (arg == "--trace" || arg == "-t") {
            enable_trace = true;
        } else if (arg == "--strict-hle") {
            enable_strict_hle = true;
        } else if (arg == "--budget" && i + 1 < argc) {
            instruction_budget = std::strtoull(argv[++i], nullptr, 10);
            budget_specified = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ps4-runner [options] <path-to-eboot.bin-or-elf>\n";
            std::cout << "Options:\n";
            std::cout << "  --window, -w     Open interactive macOS display window (60 FPS)\n";
            std::cout << "  --headless       Run headlessly in console only (default)\n";
            std::cout << "  --budget N       Headless instruction budget (default 100000000, 0 = unlimited)\n";
            std::cout << "  --no-jit         Disable ARM64 JIT dynamic recompiler (interpreter only)\n";
            std::cout << "  --trace, -t      Trace guest instruction execution step-by-step\n";
            std::cout << "  --strict-hle     Fail-loud strict HLE development mode (crash on missing NIDs)\n";
            return 0;
        } else {
            elf_path = argv[i];
        }
    }

    if (enable_window && !budget_specified) {
        instruction_budget = 0; // Interactive windowed mode runs indefinitely until window is closed
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
    const auto app_dir = std::filesystem::absolute(std::filesystem::path(elf_path).parent_path());
    vfs.mount("/app0", app_dir, true);
    std::cout << "[VFS] Mounted /app0 -> " << app_dir << "\n";

    ps4::SyscallTable syscalls(memory);
    syscalls.set_vfs(&vfs);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false); // Controlled shutdown

    // 6. Initialize NID Registry & Dynamic Linker
    std::cout << "[Linker] Initializing NID Registry and Dynamic Linker...\n";
    ps4::NidRegistry nid_registry;
    nid_registry.set_vfs(&vfs);
    nid_registry.set_syscall_table(&syscalls);
    nid_registry.register_default_libraries();

    ps4::DynamicLinker linker(memory, syscalls, nid_registry);
    // State up front which module-loading mode this run will use, so a log can
    // never be ambiguous about whether Mono actually executed.
    {
        const char* prx = std::getenv("PS4_LOAD_PRX");
        std::fprintf(stderr, "[prx] mode=%s\n", (prx && prx[0] && std::strcmp(prx, "0") != 0) ? "real" : "hle");
    }
    // Hand the linker to HLE handlers: sceKernelLoadStartModule needs it to map
    // PRX modules for real, and the trampoline dispatcher it owns is what makes
    // those modules' exports take over from our stubs.
    ps4::set_active_linker(&linker);
    if (enable_strict_hle) {
        linker.set_strict_hle(true);
        std::cout << "[Linker] Strict HLE mode ENABLED (fail-loud diagnostic crash on missing NIDs)\n";
    }
    if (!linker.initialize()) {
        std::cerr << "[Error] Failed to initialize dynamic linker trampoline pool.\n";
        return 1;
    }

    linker.resolve_relocations(file_data, loader);
    std::cout << "[Linker] Resolved symbols via NID registry (Active trampolines: "
              << linker.trampoline_count() << ")\n";

    // 7. Initialize Apple Silicon Metal GPU Subsystem & Window
    std::cout << "[Metal] Initializing GPU backend...\n";
    ps4::MetalBackend metal;
    std::unique_ptr<ps4::DisplayWindow> window;
    if (metal.initialize()) {
        std::cout << "[Metal] GPU Device: " << metal.device_name() << "\n";
        if (metal.create_framebuffer(1920, 1080)) {
            std::cout << "[Metal] 1920x1080 BGRA8Unorm Framebuffer allocated in unified memory.\n";
            // PlayStation Deep Blue clear: R=0.0f, G=0.15f, B=0.45f
            if (metal.clear_screen({0.0f, 0.15f, 0.45f, 1.0f})) {
                std::cout << "[Metal] GPU command buffer executed clear pass.\n";
            }
            nid_registry.set_metal_backend(&metal);
        }

        if (enable_window) {
            std::cout << "\n[Display] Opening interactive macOS Metal window...\n";
            std::cout << "[Display] Running presentation loop at 60 FPS (Press ESC or close window to exit)\n";
            window = std::make_unique<ps4::DisplayWindow>();
            ps4::WindowConfig win_cfg{
                .title = "ps4-to-macos: " + elf_path.filename().string() + " (Metal 3 - Interactive)",
                .width = 1280,
                .height = 720,
                .resizable = true,
                .headless = false
            };
            if (!window->initialize(metal, win_cfg)) {
                std::cerr << "[Display] Warning: Failed to open interactive window, falling back to console mode.\n";
                window.reset();
            }
        }
    } else {
        std::cout << "[Metal] Note: Metal device initialization skipped or unavailable.\n";
    }

    // 8. CPU Execution
    std::cout << "\n[CPU] Initializing execution engine & ARM64 JIT dynamic recompiler...\n";
    ps4::JitCompiler jit;
    ps4::CpuExecutor cpu(memory, syscalls);
    // Creating overload: the manager may not exist yet (first HLE dispatch
    // creates it lazily), and the no-arg accessor would return null here.
    nid_registry.thread_manager(memory).set_main_context(&cpu.context());
    // Optional single-address write-watch, for chasing a specific clobber:
    // PS4_WATCH_ADDR=0x1ffffac40 ./ps4-runner ... reports the first store to that
    // guest address together with the RIP and TID that made it.
    if (const char* w = std::getenv("PS4_WATCH_ADDR")) {
        // The canary slot is NOT deterministic across runs -- it lands wherever
        // main's frame happens to be -- so allow a range and filter afterwards.
        const std::size_t wsz = std::getenv("PS4_WATCH_SIZE")
                              ? std::strtoull(std::getenv("PS4_WATCH_SIZE"), nullptr, 0)
                              : 8;
        // base 0: "0x..." is hex, a bare "123" is decimal. Printing the parsed
        // value matters because a bare decimal string for a module address is a
        // silently wrong watch (160246468 == 0x98d2ac4, not 0x160246468).
        const ps4::GuestAddr waddr = std::strtoull(w, nullptr, 0);
        std::fprintf(stderr, "[watch] PS4_WATCH_ADDR=\"%s\" parsed as 0x%llx (base 0)\n",
                     w, (unsigned long long)waddr);
        ps4::set_write_watch(waddr, static_cast<ps4::usize>(wsz));
    }
    if (enable_jit) {
        cpu.set_jit_compiler(&jit);
        nid_registry.set_jit_compiler(&jit);
        std::cout << "[CPU] ARM64 JIT compiler connected to executor (Tier-2 Dynamic Binary Translation active)\n";
    } else {
        std::cout << "[CPU] Running in pure interpreter mode (--no-jit specified)\n";
    }
    if (!cpu.setup_default_stack()) {
        std::cerr << "[Error] Failed to allocate stack.\n";
        return 1;
    }
    cpu.set_entry_point(loader.entry_point());
    std::cout << "[CPU] RIP set to entry point: 0x" << std::hex << loader.entry_point() << std::dec << "\n";
    std::cout << "[CPU] RSP initialized to:     0x" << std::hex << cpu.context().rsp << std::dec << "\n";

    std::cout << "\n================= Guest Execution Output =================\n";
    ps4::RunResult run_res{};
    // Main does not run through ThreadManager's worker lambda, so t_guest_tid
    // was never published for it and the write-watch reported main as TID=0.
    // Publish it here: main is guest TID 1.
    ps4::t_guest_tid = 1;
    ps4::usize total_instructions = 0;

    if (enable_trace) {
        for (int i = 0; i < 20; ++i) {
            ps4::GuestAddr cur_rip = cpu.context().rip;
            ps4::StepResult sr = cpu.step();
            std::cout << "Step " << i << ": RIP=0x" << std::hex << cur_rip << " -> 0x" << cpu.context().rip 
                      << " (" << ps4::step_result_to_string(sr) << ")\n" << std::dec;
            total_instructions++;
            if (sr != ps4::StepResult::Ok && sr != ps4::StepResult::Syscall) {
                run_res.status = sr;
                if (sr == ps4::StepResult::Exited) {
                    run_res.exit_code = syscalls.exit_code();
                }
                break;
            }
        }
    }

    if (!syscalls.has_exited() &&
        (run_res.status == ps4::StepResult::Ok || run_res.status == ps4::StepResult::Syscall)) {
        if (window && window->is_open()) {
            // Interactive Presentation Loop:
            // Run guest execution batches each frame, pump user input events, and present via Metal
            const ps4::usize frame_slice = 2000000;
            auto t_start = std::chrono::steady_clock::now();
            auto t_last = t_start;
            ps4::usize instrs_last = 0;
            constexpr auto frame_period = std::chrono::microseconds(16667);
            auto next_deadline = std::chrono::steady_clock::now();
            while (window->is_open() && !syscalls.has_exited() && (instruction_budget == 0 || total_instructions < instruction_budget)) {
                next_deadline += frame_period;
                window->poll_events();
                nid_registry.set_pad_buttons(window->input_buttons());

                ps4::Pm4Translator* gpu = nid_registry.gpu_translator();
                if (gpu && !gpu->begin_frame()) {
                    std::cerr << "[GPU] Failed to begin translator frame; stopping guest execution.\n";
                    run_res.status = ps4::StepResult::MemoryFault;
                    break;
                }
                run_res = cpu.run(frame_slice);
                total_instructions += run_res.instructions_executed;
                if (gpu) {
                    const bool submitted = gpu->submit_frame(false);
                    if (!submitted) {
                        std::cerr << "[GPU] Failed to submit translator frame; stopping guest execution.\n";
                        if (run_res.status == ps4::StepResult::Ok ||
                            run_res.status == ps4::StepResult::Syscall) {
                            run_res.status = ps4::StepResult::MemoryFault;
                        }
                    }
                }
                if (run_res.status != ps4::StepResult::Ok) break;

                const auto now = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(now - t_last).count();
                const double dt_total = std::chrono::duration<double>(now - t_start).count();
                const double cur_mips = (dt > 0.0) ? ((total_instructions - instrs_last) / dt / 1e6) : 0.0;
                const double avg_mips = (dt_total > 0.0) ? (total_instructions / dt_total / 1e6) : 0.0;
                instrs_last = total_instructions;
                t_last = now;

                std::cout << "[CPU Progress] " << total_instructions << " instrs ("
                          << std::fixed << std::setprecision(1) << cur_mips << " M instr/s, avg "
                          << avg_mips << " M instr/s), RIP=0x" << std::hex << cpu.context().rip << std::dec << "\n" << std::flush;

                if (nid_registry.get_flip_count() > 0) {
                    window->present_frame();
                }

                const auto deadline_now = std::chrono::steady_clock::now();
                if (deadline_now > next_deadline + frame_period * 2) {
                    next_deadline = deadline_now;
                } else {
                    std::this_thread::sleep_until(next_deadline);
                }
            }
        } else {
            // Console / Headless Execution Loop
            const ps4::usize batch = 1000000;
            const ps4::usize max_total = instruction_budget;
            auto t_start = std::chrono::steady_clock::now();
            auto t_last = t_start;
            ps4::usize instrs_last = 0;
            while ((max_total == 0 || total_instructions < max_total) && !syscalls.has_exited()) {
                run_res = cpu.run(batch);
                total_instructions += run_res.instructions_executed;
                if (run_res.status != ps4::StepResult::Ok) break;

                const auto now = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(now - t_last).count();
                const double dt_total = std::chrono::duration<double>(now - t_start).count();
                const double cur_mips = (dt > 0.0) ? ((total_instructions - instrs_last) / dt / 1e6) : 0.0;
                const double avg_mips = (dt_total > 0.0) ? (total_instructions / dt_total / 1e6) : 0.0;
                instrs_last = total_instructions;
                t_last = now;

                std::cout << "[CPU Progress] " << total_instructions << " instrs ("
                          << std::fixed << std::setprecision(1) << cur_mips << " M instr/s, avg "
                          << avg_mips << " M instr/s), RIP=0x" << std::hex << cpu.context().rip << std::dec << "\n" << std::flush;
            }
        }
    }
    std::cout << "==========================================================\n";

    std::cout << "\n[CPU] Execution Status: " << ps4::step_result_to_string(run_res.status) << "\n";
    std::cout << "[CPU] Instructions Executed: " << total_instructions << "\n";
    // Dump an armed write-watch here too, not only on fault. A run that ends on
    // the instruction budget reports "executed successfully" and never reaches
    // the fault handler, so the ring would otherwise be silently discarded --
    // which is exactly the case where you want to know whether the watched slot
    // was written at all.
    if (ps4::write_watch_addr() != 0) {
        const char* wf = std::getenv("PS4_WATCH_FILTER");
        ps4::dump_write_watch_ring(wf ? std::strtoull(wf, nullptr, 0)
                                      : ps4::write_watch_addr(), &memory);
    }
    std::fprintf(stderr, "[map] final table at run exit:\n");
    // PS4_DUMP_CODE=<guest_addr>:<len> writes raw loaded guest bytes to a file.
    // The SELF's phdr.p_offset values are stale and non-uniform, so llvm-objdump
    // on the extracted inner ELF maps addresses differently from the runtime.
    // Dumping what the emulator actually mapped is the only way to disassemble
    // the image the guest really executed.
    if (const char* dc = std::getenv("PS4_DUMP_CODE")) {
        unsigned long long base = 0, len = 0;
        if (std::sscanf(dc, "%llx:%llx", &base, &len) == 2 && len > 0 && len < (1u << 22)) {
            std::vector<ps4::u8> buf(static_cast<size_t>(len));
            if (memory.read_bytes(static_cast<ps4::GuestAddr>(base), buf.data(), buf.size())) {
                char path[256];
                std::snprintf(path, sizeof(path), "dump_code_%llx.bin", base);
                std::FILE* f = std::fopen(path, "wb");
                if (f) {
                    std::fwrite(buf.data(), 1, buf.size(), f);
                    std::fclose(f);
                    std::fprintf(stderr, "[dumpcode] %llu bytes @ 0x%llx -> %s\n",
                                 len, base, path);
                }
            } else {
                std::fprintf(stderr, "[dumpcode] read FAILED at 0x%llx\n", base);
            }
        }
    }
    memory.dump_mappings();
    ps4::dump_rip_histogram();
    ps4::dump_hle_histogram();
    ps4::dump_export_tables();
    ps4::dump_prx_calls();
    if (enable_jit && jit.cached_block_count() > 0) {
        std::cout << "[CPU] JIT Blocks Cached: " << jit.cached_block_count()
                  << " | Total Invocations: " << jit.total_invocations() << "\n";
    }
    if (auto* tm = nid_registry.thread_manager()) {
        std::cout << "[Threads] Created: " << tm->threads_created()
                  << " | Started: " << tm->threads_started()
                  << " | Executed Instructions: " << tm->threads_executed_instructions() << "\n";
    }
    if (run_res.status == ps4::StepResult::Exited) {
        std::cout << "[CPU] Process Exit Code: " << run_res.exit_code << "\n";
    }

    cpu.context().dump();

    // End-of-run per-thread snapshot: state, wake deadline, last RIP, instrs.
    // Answers "what is each thread blocked in" without a live debugger.
    if (auto* tm = nid_registry.thread_manager()) {
        tm->dump_thread_states();
    }
    ps4::dump_sema_states();
    ps4::dump_hle_warning_stats();

    if (window && window->is_open()) {
        window->close();
    }

    // Join all guest pthreads before local MemoryManager, SyscallTable, and JitCompiler go out of scope
    nid_registry.shutdown();

    return 0;
}
