#include <cstdarg>
#include <cstdio>
#include <cstring>
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"
extern "C" {
#include "cases.h" // case table shared with native.c
}
using namespace ps4;
static int pr(const char* f, ...) { va_list a; va_start(a, f); int r = vprintf(f, a); va_end(a); return r; }
int main(int argc, char**) {
    MemoryManager mem; mem.initialize();
    SyscallTable sys(mem);
    CpuExecutor cpu(mem, sys);
    JitCompiler jit;
    if (argc > 1) cpu.set_jit_compiler(&jit);
    cpu.setup_default_stack();
    const GuestAddr code = 0x50000, gmem = 0x60000;
    for (unsigned c = 0; c < NCASES; c++) {
        for (unsigned t = 0; t < 8; t++) {
            static State s; fill(&s, c, t);
            if (kCases[c].mem) { s.rax = gmem - 0x1ab0; s.rsi = 0; }
            u8 buf[16]; std::memcpy(buf, kCases[c].bytes, kCases[c].len); buf[kCases[c].len] = 0xF4;
            mem.write_bytes(code, buf, kCases[c].len + 1);
            mem.write_bytes(gmem, s.mem, 16);
            jit.clear_cache();
            auto& x = cpu.context();
            for (int r = 0; r < 16; r++) std::memcpy(&x.ymm[r], s.ymm[r], 32);
            x.rax = s.rax; x.rcx = s.rcx; x.rdx = s.rdx; x.rsi = s.rsi; x.rip = code;
            if (argc > 1) cpu.run(1); else cpu.step();
            if (x.rip != code + kCases[c].len) pr("# rip mismatch case %u: +%llu\n", c, (unsigned long long)(x.rip - code));
            for (int r = 0; r < 16; r++) std::memcpy(s.ymm[r], &x.ymm[r], 32);
            s.rax = x.rax; s.rcx = x.rcx; s.rdx = x.rdx; s.rsi = x.rsi;
            mem.read_bytes(gmem, s.mem, 16);
            if (kCases[c].mem) s.rax -= gmem - 0x1ab0;
            print_state(&s, c, t, pr);
        }
    }
}
