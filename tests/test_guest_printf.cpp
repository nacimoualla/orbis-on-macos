// Guest printf formatting through the real vsnprintf HLE handler, with a
// hand-built SysV va_list: GP and XMM save areas plus the overflow area.

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ps4/cpu_context.hpp"
#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"

using namespace ps4;

namespace {
constexpr Nid kVsnprintf = 0x43657E8AABE3802DULL;
constexpr GuestAddr kFmt = 0x50000, kOut = 0x51000, kVaList = 0x52000, kSave = 0x53000,
                    kOverflow = 0x54000, kStr = 0x55000;

struct Harness {
    MemoryManager mem;
    NidRegistry registry;
    Harness() {
        assert(mem.initialize());
        registry.register_default_libraries();
    }

    // gp: integer/pointer args in order; fp: double args in order.
    std::string format(const char* fmt, const std::vector<u64>& gp, const std::vector<double>& fp = {}) {
        assert(mem.write_bytes(kFmt, fmt, std::strlen(fmt) + 1));
        // The va_list starts after the 3 named GP args of vsnprintf (gp_offset 24)
        // and with no named FP args (fp_offset 48).
        const u32 gp_offset = 24, fp_offset = 48;
        usize g = 0, f = 0;
        GuestAddr overflow = kOverflow;
        for (u32 off = gp_offset; off < 48 && g < gp.size(); off += 8) assert(mem.write_value(kSave + off, gp[g++]));
        for (u32 off = fp_offset; off < 176 && f < fp.size(); off += 16) {
            u64 bits;
            std::memcpy(&bits, &fp[f++], 8);
            assert(mem.write_value(kSave + off, bits));
        }
        // Remaining args spill to the overflow area in call order; this test
        // only mixes kinds where registers are exhausted for one kind at a time.
        for (; g < gp.size(); ++g, overflow += 8) assert(mem.write_value(overflow, gp[g]));
        for (; f < fp.size(); ++f, overflow += 8) {
            u64 bits;
            std::memcpy(&bits, &fp[f], 8);
            assert(mem.write_value(overflow, bits));
        }
        assert(mem.write_value(kVaList + 0, gp_offset));
        assert(mem.write_value(kVaList + 4, fp_offset));
        assert(mem.write_value(kVaList + 8, static_cast<u64>(kOverflow)));
        assert(mem.write_value(kVaList + 16, static_cast<u64>(kSave)));

        CpuContext ctx{};
        ctx.rdi = kOut;
        ctx.rsi = 512;
        ctx.rdx = kFmt;
        ctx.rcx = kVaList;
        const HleSymbol* sym = registry.find_by_nid(kVsnprintf);
        assert(sym && sym->handler);
        sym->handler(ctx, mem);
        std::string out(mem.to_host_ptr<char>(kOut));
        assert(ctx.rax == out.size());
        return out;
    }
};

void expect(const std::string& got, const std::string& want) {
    if (got != want) {
        std::cerr << "got  '" << got << "'\nwant '" << want << "'\n";
        assert(false);
    }
}
} // namespace

int main() {
    std::cout << "=== Guest printf formatting ===\n";
    Harness h;

    // Mono's GUID format: 16 byte arguments, zero-padded, most in the overflow area.
    std::vector<u64> guid;
    for (u64 b : {0x0Au, 0x1Bu, 0x2Cu, 0x3Du, 0x04u, 0x05u, 0x06u, 0x07u,
                  0x80u, 0x91u, 0xA2u, 0xB3u, 0xC4u, 0xD5u, 0xE6u, 0xF7u}) guid.push_back(b);
    expect(h.format("%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X", guid),
           "0A1B2C3D-0405-0607-8091-A2B3C4D5E6F7");

    assert(h.mem.write_bytes(kStr, "mscorlib", 9));
    expect(h.format("[%-10s|%5s|%.3s]", {kStr, kStr, kStr}), "[mscorlib  |mscorlib|msc]");
    expect(h.format("%*d|%-*d|%+d", {5, 42, 4, 7, 9}), "   42|7   |+9");
    expect(h.format("%d %u %x", {0xFFFFFFFFull, 0xFFFFFFFFull, 0x1234ABCDull}), "-1 4294967295 1234abcd");
    expect(h.format("%lld %llu %lx", {static_cast<u64>(-5), ~0ull, 0xDEADBEEFCAFEull}),
           "-5 18446744073709551615 deadbeefcafe");
    expect(h.format("%hhx %hd %c%%", {0x1FFull, 0x18000ull, 'Z'}), "ff -32768 Z%");
    expect(h.format("%.2f %e %g", {}, {3.14159, 1500.0, 0.25}), "3.14 1.500000e+03 0.25");
    expect(h.format("%s", {0}), "(null)");
    expect(h.format("%p", {0x4000000}), "0x4000000");

    // vsscanf through the real handler: pointers in the va_list GP area.
    {
        constexpr Nid kVsscanf = 0x513AE93952DCA0E0ULL;
        constexpr GuestAddr kIn = 0x56000, kOut = 0x57000;
        auto scan = [&](const char* input, const char* fmt, int nptrs) -> i64 {
            assert(h.mem.write_bytes(kIn, input, std::strlen(input) + 1));
            assert(h.mem.write_bytes(kFmt, fmt, std::strlen(fmt) + 1));
            std::vector<u8> zero(256, 0);
            assert(h.mem.write_bytes(kOut, zero.data(), zero.size()));
            for (int i = 0; i < nptrs; ++i) assert(h.mem.write_value(kSave + 16 + 8 * i, kOut + 32 * i));
            assert(h.mem.write_value(kVaList + 0, u32{16}));   // after str, fmt
            assert(h.mem.write_value(kVaList + 4, u32{48}));
            assert(h.mem.write_value(kVaList + 8, static_cast<u64>(kOverflow)));
            assert(h.mem.write_value(kVaList + 16, static_cast<u64>(kSave)));
            CpuContext ctx{};
            ctx.rdi = kIn;
            ctx.rsi = kFmt;
            ctx.rdx = kVaList;
            h.registry.find_by_nid(kVsscanf)->handler(ctx, h.mem);
            return static_cast<i64>(ctx.rax);
        };
        auto out32 = [&](int i) { u32 v = 0; h.mem.read_value(kOut + 32 * i, v); return v; };
        auto outstr = [&](int i) { return std::string(h.mem.to_host_ptr<char>(kOut + 32 * i)); };
        assert(scan("specific_trampoline_lazy_fetch_2147483648", "specific_trampoline_lazy_fetch_%d", 1) == 1);
        assert(out32(0) == 0x80000000u);
        assert(scan("  42 ff word rest", "%d %x %s", 3) == 3);
        assert(out32(0) == 42 && out32(1) == 0xff && outstr(2) == "word");
        assert(scan("abc", "%d", 1) == 0);
        assert(scan("", "%d", 1) == -1);
        assert(scan("key=val;", "%[a-z]=%[^;]", 2) == 2 && outstr(0) == "key" && outstr(1) == "val");
        assert(scan("12:34", "%*d:%d%n", 2) == 1 && out32(0) == 34 && out32(1) == 5);
        std::cout << "  -> vsscanf passed\n";
    }

    std::cout << "All guest printf tests passed.\n";
    return 0;
}
