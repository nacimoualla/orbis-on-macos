#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "ps4/display_window.hpp"
#include "ps4/dynamic_linker.hpp"
#include "ps4/elf_loader.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/pm4_translator.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/thread.hpp"
#include "ps4/vfs.hpp"

namespace {
constexpr int kFrames = 120;
const std::filesystem::path kElf =
    std::filesystem::path(CMAKE_SOURCE_DIR) /
    "games/flappy_bird/ED1633-NPXX51362_00-TER1520000000000/eboot.elf";

int fail(const std::string& message) {
    std::cerr << "[Flappy GPU smoke] " << message << "\n";
    return 1;
}
} // namespace

int main() {
    if (!std::filesystem::is_regular_file(kElf))
        return fail("missing retail target: " + kElf.string());
    std::ifstream file(kElf, std::ios::binary | std::ios::ate);
    if (!file) return fail("cannot open retail target: " + kElf.string());
    const auto size = file.tellg();
    file.seekg(0);
    std::vector<ps4::u8> data(static_cast<ps4::usize>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size))
        return fail("failed to read retail target: " + kElf.string());

    ps4::ElfLoader loader;
    if (!loader.load_from_memory(data))
        return fail("ELF loading failed: " + std::string(loader.error_message()));
    ps4::MemoryManager memory;
    if (!memory.initialize())
        return fail("guest memory initialization failed: " + std::string(memory.error_message()));
    if (!memory.load_elf_segments(loader, data))
        return fail("ELF segment mapping failed: " + std::string(memory.error_message()));

    ps4::VirtualFileSystem vfs;
    vfs.mount("/app0", kElf.parent_path(), true);
    ps4::SyscallTable syscalls(memory);
    syscalls.set_vfs(&vfs);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false);
    ps4::NidRegistry registry;
    registry.set_vfs(&vfs);
    registry.set_syscall_table(&syscalls);
    registry.register_default_libraries();
    ps4::DynamicLinker linker(memory, syscalls, registry);
    ps4::set_active_linker(&linker);
    if (!linker.initialize())
        return fail("dynamic linker initialization failed: " + linker.relocation_diagnostics());
    if (!linker.resolve_relocations(data, loader))
        return fail("dynamic linker relocation failed: " + linker.relocation_diagnostics());

    ps4::MetalBackend metal;
    if (!metal.initialize()) return fail("Metal dependency unavailable");
    if (!metal.create_framebuffer(1920, 1080))
        return fail("Metal framebuffer creation failed");
    registry.set_metal_backend(&metal);
    ps4::Pm4Translator* translator = registry.gpu_translator();
    if (!translator) return fail("GPU translator dependency was not initialized");
    translator->set_memory_manager(&memory);
    ps4::DisplayWindow window;
    if (!window.initialize(metal, {.title = "Flappy GPU smoke", .width = 1920,
                                   .height = 1080, .resizable = false, .headless = true}))
        return fail("headless DisplayWindow initialization failed");

    ps4::JitCompiler jit;
    ps4::CpuExecutor cpu(memory, syscalls);
    cpu.set_jit_compiler(&jit);
    registry.set_jit_compiler(&jit);
    registry.thread_manager(memory).set_main_context(&cpu.context());
    if (!cpu.setup_default_stack()) return fail("guest stack initialization failed");
    cpu.set_entry_point(loader.entry_point());
    ps4::t_guest_tid = 1;

    const auto start = std::chrono::steady_clock::now();
    ps4::usize packets = 0;
    ps4::RunResult result{};
    for (int frame = 0; frame < kFrames; ++frame) {
        window.poll_events();
        if (!translator->begin_frame()) return fail("GPU frame begin failed");
        result = cpu.run(2'000'000);
        packets += translator->frame_stats().packets;
        if (!translator->submit_frame(false)) return fail("GPU frame submission failed");
        if (!window.present_frame()) return fail("headless frame presentation failed");
        if (result.status != ps4::StepResult::Ok &&
            result.status != ps4::StepResult::Syscall)
            return fail("guest execution stopped at frame " + std::to_string(frame + 1) +
                        " before the smoke interval completed (status=" +
                        std::string(ps4::step_result_to_string(result.status)) +
                        "; guest execution dependency missing; inspect the preceding "
                        "CPU/HLE diagnostic for the exact unsupported operation; "
                        "runtime: " + linker.relocation_diagnostics() +
                        "; GPU packets=" + std::to_string(translator->packets_processed()) +
                        " draws=" + std::to_string(translator->draw_count()) +
                        " unsupported=" + std::to_string(translator->unsupported_packet_count()) +
                        ". Relevant CPU/HLE/GPU diagnostics above are intentionally preserved.)");
    }

    if (!metal.wait_for_last_frame()) return fail("last GPU frame did not complete");
    ps4::u8 r = 0, g = 0, b = 0, a = 0;
    if (!metal.read_pixel(metal.width() / 2, metal.height() / 2, r, g, b, a))
        return fail("failed to read center framebuffer pixel");
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    const auto stats = translator->frame_stats();
    const auto draws = translator->draw_count();
    const auto unsupported = translator->unsupported_packet_count();
    std::cout << "Flappy GPU smoke: frames=" << kFrames
              << " draws=" << draws << " packets=" << packets
              << " unsupported_packets=" << unsupported
              << " upload_bytes=" << stats.upload_bytes
              << " elapsed_ms=" << elapsed << "\n";
    registry.shutdown();
    if (draws == 0 || packets == 0)
        return fail("retail guest produced no GPU draws or PM4 packets");
    if (r == 0 && g == 0 && b == 0)
        return fail("center framebuffer pixel is black");
    return 0;
}
