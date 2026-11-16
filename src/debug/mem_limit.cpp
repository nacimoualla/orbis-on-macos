// Process memory limit.
//
// A game that makes the emulator allocate without bound (inFAMOUS Second Son
// once took a run past 40 GB through per-draw GPU copies) should stop with a
// message, not push a 16 GB machine into swap until it is unusable. The guest
// address space is an 8 GB reservation; what counts here is the physical
// footprint (dirty and compressed pages), as Activity Monitor shows it.

#include "ps4/debug/mem_limit.hpp"

#include <libproc.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace ps4::debug {

std::uint64_t process_footprint() {
    rusage_info_v2 info{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, reinterpret_cast<rusage_info_t*>(&info)) != 0) return 0;
    return info.ri_phys_footprint;
}

std::uint64_t start_memory_limit() {
    std::uint64_t limit = 0;
    if (const char* v = std::getenv("PS4_MEM_LIMIT_MB")) {
        limit = std::strtoull(v, nullptr, 10) << 20;
    } else {
        std::uint64_t ram = 0;
        size_t len = sizeof(ram);
        if (sysctlbyname("hw.memsize", &ram, &len, nullptr, 0) == 0) limit = ram / 2;
    }
    if (limit == 0) return 0;
    std::thread([limit] {
        std::uint64_t peak = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const std::uint64_t now = process_footprint();
            if (now > peak) peak = now;
            if (now > limit) {
                std::fprintf(stderr,
                             "\n[memory] stopping: the emulator is using %llu MB, over its limit of %llu MB.\n"
                             "[memory] Set PS4_MEM_LIMIT_MB to change the limit (0 turns it off).\n",
                             static_cast<unsigned long long>(now >> 20),
                             static_cast<unsigned long long>(limit >> 20));
                std::fflush(stderr);
                std::fflush(stdout);
                _exit(99);
            }
        }
    }).detach();
    return limit;
}

}  // namespace ps4::debug
