#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "ps4/display_window.hpp"
#include "ps4/dynamic_linker.hpp"
#include "ps4/elf_loader.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/jit_stats.hpp"
#include "ps4/memory.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/pm4_translator.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/thread.hpp"
#include "ps4/vfs.hpp"

namespace {
constexpr int kDefaultFrames = 120;
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

    // PS4_WATCH_ADDR / PS4_WATCH_SIZE: report guest stores to that range, as in ps4-runner.
    if (const char* w = std::getenv("PS4_WATCH_ADDR")) {
        const char* sz = std::getenv("PS4_WATCH_SIZE");
        ps4::set_write_watch(std::strtoull(w, nullptr, 0), sz ? std::strtoull(sz, nullptr, 0) : 8);
    }

    ps4::JitCompiler jit;
    ps4::CpuExecutor cpu(memory, syscalls);
    // PS4_NO_JIT=1 runs every guest thread on the interpreter (A/B a JIT bug).
    if (!std::getenv("PS4_NO_JIT")) {
        cpu.set_jit_compiler(&jit);
        registry.set_jit_compiler(&jit);
    }
    registry.thread_manager(memory).set_main_context(&cpu.context());
    if (!cpu.setup_default_stack()) return fail("guest stack initialization failed");
    cpu.set_entry_point(loader.entry_point());
    ps4::t_guest_tid = 1;

    // PS4_TRACE_AFTER=addr: once that RIP executes, trace every instruction.
    const ps4::GuestAddr trace_after =
        std::getenv("PS4_TRACE_AFTER") ? std::strtoull(std::getenv("PS4_TRACE_AFTER"), nullptr, 0) : 0;
    bool trace_armed = false;
    // PS4_TRACE_MAX: number of trace lines to emit (default 4000).
    const int trace_max = std::getenv("PS4_TRACE_MAX") ? std::atoi(std::getenv("PS4_TRACE_MAX")) : 4000;
    ps4::GuestAddr trace_lo = 0, trace_hi = 0;
    int traced = 0;
    if (const char* r = std::getenv("PS4_TRACE_RANGE")) {
        char* end = nullptr;
        trace_lo = std::strtoull(r, &end, 0);
        trace_hi = (*end == ',') ? std::strtoull(end + 1, nullptr, 0) : trace_lo + 1;
    }
    int auto_tap_start = 0, auto_tap_period = 0, auto_tap_hold = 2;
    if (const char* t = std::getenv("PS4_AUTO_TAP"))
        std::sscanf(t, "%d,%d,%d", &auto_tap_start, &auto_tap_period, &auto_tap_hold);
    const auto start = std::chrono::steady_clock::now();
    ps4::usize packets = 0;
    ps4::RunResult result{};
    // PS4_FRAMES=<n> runs longer (or shorter) than the default smoke interval.
    const int kFrames = std::getenv("PS4_FRAMES") ? std::atoi(std::getenv("PS4_FRAMES")) : kDefaultFrames;
    for (int frame = 0; frame < kFrames; ++frame) {
        const auto t_frame = std::chrono::steady_clock::now();
        double run_ms = 0;
        window.poll_events();
        // PS4_AUTO_TAP=start,period[,hold]: from frame `start`, press the touch
        // pad (and Cross) for `hold` frames (default 2) every `period` frames,
        // so a headless run can play. Otherwise forward the window's input.
        if (auto_tap_period > 0 && frame >= auto_tap_start) {
            const bool down = (frame - auto_tap_start) % auto_tap_period < auto_tap_hold;
            registry.set_pad_buttons(down ? 0x4000u | 0x100000u : 0u);
        } else {
            registry.set_pad_buttons(window.input_buttons());
        }
        if (!translator->begin_frame()) return fail("GPU frame begin failed");
        if (trace_lo != 0 || trace_after != 0) {
            // PS4_TRACE_RANGE=lo,hi: single-step the main thread and print every
            // RIP inside [lo, hi) (first 4000), for following guest control flow.
            result = {};
            for (int n = 0; n < 2'000'000; ++n) {
                const ps4::GuestAddr rip = cpu.context().rip;
                if (trace_after != 0 && rip == trace_after) trace_armed = true;
                if (((trace_after != 0 && trace_armed) || (trace_after == 0 && rip >= trace_lo && rip < trace_hi)) &&
                    traced < trace_max) {
                    const auto& c = cpu.context();
                    ps4::u64 ret = 0;
                    memory.read_value(c.rsp, ret);
                    std::fprintf(stderr, "[trace] 0x%llx rax=0x%llx rdi=0x%llx rsi=0x%llx [rsp]=0x%llx xmm0=%g/%g rcx=0x%llx "
                                 "xmm1=%g/%g xmm2=%g/%g\n",
                                 (unsigned long long)rip, (unsigned long long)c.rax, (unsigned long long)c.rdi,
                                 (unsigned long long)c.rsi, (unsigned long long)ret, c.ymm[0].f[0], c.ymm[0].df[0],
                                 (unsigned long long)c.rcx, c.ymm[1].f[0], c.ymm[1].df[0], c.ymm[2].f[0], c.ymm[2].df[0]);
                    ++traced;
                }
                result.status = cpu.step();
                ++result.instructions_executed;
                if (result.status != ps4::StepResult::Ok && result.status != ps4::StepResult::Syscall) break;
            }
        } else {
            const auto t_run = std::chrono::steady_clock::now();
            result = cpu.run(2'000'000);
            run_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_run).count();
        }
        packets += translator->frame_stats().packets;
        // PS4_SAMPLE_MAIN=1: one sample per frame of where the main thread is
        // (RIP plus its frame-pointer chain), for profiling long stretches.
        if (std::getenv("PS4_SAMPLE_MAIN")) {
            const auto& c = cpu.context();
            std::fprintf(stderr, "[sample] %d rip=0x%llx", frame, (unsigned long long)c.rip);
            ps4::GuestAddr fp = c.rbp;
            for (int d = 0; d < 10 && fp > 0x1000; ++d) {
                ps4::u64 next = 0, ret = 0;
                if (!memory.read_value(fp, next) || !memory.read_value(fp + 8, ret)) break;
                std::fprintf(stderr, " 0x%llx", (unsigned long long)ret);
                if (next <= fp) break;
                fp = next;
            }
            std::fprintf(stderr, "\n");
        }
        if (!translator->submit_frame(false)) return fail("GPU frame submission failed");
        if (!window.present_frame()) return fail("headless frame presentation failed");
        // PS4_FRAME_TIMES=1: per-frame wall time split into the main thread's
        // guest slice and everything else (GPU submit, present, readback).
        if (std::getenv("PS4_FRAME_TIMES")) {
            const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_frame).count();
            std::fprintf(stderr, "[ftime] %d total=%.1fms main_slice=%.1fms instrs=%zu flips=%llu\n", frame, total, run_ms,
                         (size_t)result.instructions_executed, (unsigned long long)registry.get_flip_count());
        }
        if (result.status != ps4::StepResult::Ok &&
            result.status != ps4::StepResult::Syscall) {
            std::fprintf(stderr, "[Flappy test] Main CPU RIP=0x%llx\n", (unsigned long long)cpu.context().rip);
            ps4::u8 b[16] = {0};
            memory.read_bytes(cpu.context().rip, b, 16);
            std::fprintf(stderr, "[Flappy test] Bytes at RIP: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                         b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
            registry.thread_manager(memory).dump_thread_states();
            ps4::dump_control_xfers(1); // how the main thread got to RIP
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
    }

    // PS4_DUMP_QWORDS=0xaddr[,0xaddr...]: print guest qwords at the end of the run.
    if (const char* dump = std::getenv("PS4_DUMP_QWORDS")) {
        for (const char* p = dump; *p;) {
            char* end = nullptr;
            const ps4::GuestAddr a = std::strtoull(p, &end, 0);
            ps4::u64 v = 0;
            const bool ok = memory.read_value(a, v);
            std::fprintf(stderr, "[dump] 0x%llx = %s0x%llx\n", (unsigned long long)a, ok ? "" : "<invalid> ",
                         (unsigned long long)v);
            p = (*end == ',') ? end + 1 : end;
            if (end == p && *p) break;
        }
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
    ps4::write_jit_stats(); // PS4_JIT_STATS; the guest's exit path can skip atexit
    ps4::dump_rip_histogram(); // PS4_RIP_HIST=1 (interpreter steps only)
    if (std::getenv("PS4_WATCH_ADDR")) ps4::dump_write_watch_ring(0, &memory);
    if (std::getenv("PS4_HLE_HIST")) ps4::dump_hle_histogram(); // every HLE dispatch by name
    registry.shutdown();
    if (draws == 0 || packets == 0)
        return fail("retail guest produced no GPU draws or PM4 packets");
    if (r == 0 && g == 0 && b == 0)
        return fail("center framebuffer pixel is black");
    return 0;
}
