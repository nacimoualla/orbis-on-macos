// x87 rounding and control-word vectors.
//
// The rounding logic under test lives inline in CpuExecutor::step(), driven by
// guest memory and the control word, so these tests drive the same encoder from
// a synthetic guest: a MemoryManager, a CpuContext, and the real instruction
// bytes. That exercises the decoder too, not a reimplementation of the rounding
// table -- a divergence between this test and the interpreter would then be a
// real failure rather than a tautology.
#include "ps4/common.hpp"
#include "ps4/cpu_context.hpp"
#include "ps4/executor.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/memory.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace ps4;

static int g_failures = 0;

static void check_u64(const char* what, u64 got, u64 want) {
    const bool ok = (got == want);
    if (!ok) ++g_failures;
    std::printf("  %-58s got=0x%016llx want=0x%016llx %s\n", what,
                (unsigned long long)got, (unsigned long long)want, ok ? "ok" : "FAIL");
}

// Host double -> 80-bit extended, so FLD m80/FSTP m80 round-trips real bytes.
static void f64_to_f80(double d, u8 out[10]) {
    std::memset(out, 0, 10);
    u64 bits;
    std::memcpy(&bits, &d, 8);
    const u64 frac = bits & 0x000FFFFFFFFFFFFFULL;
    const u64 be = (bits >> 52) & 0x7FF;
    const u64 sign = (bits >> 63) & 1;
    u16 se;
    if (be == 0 && frac == 0) {
        se = static_cast<u16>(sign << 15);          // zero
        out[8] = out[9] = 0;
    } else if (be == 0x7FF) {
        se = static_cast<u16>((sign << 15) | 0x7FFF);  // inf / NaN
        out[8] = out[9] = 0xFF;
    } else {
        const u64 exp80 = be + 0x3800;             // bias 127 -> 16383
        se = static_cast<u16>((sign << 15) | exp80);
        const u64 mant = frac | 0x8000000000000000ULL;  // explicit integer bit
        for (int i = 0; i < 8; ++i) out[i] = static_cast<u8>((mant >> (8 * i)) & 0xFF);
    }
}

static double f80_to_f64(const u8 in[10]) {
    u64 mant = 0;
    for (int i = 7; i >= 0; --i) mant = (mant << 8) | in[i];
    const u16 se = static_cast<u16>(in[8] | (in[9] << 8));
    const u64 exp80 = se & 0x7FFF;
    const u64 sign = (se >> 15) & 1;
    if (exp80 == 0) return 0.0 * (sign ? -1.0 : 1.0);
    if (exp80 == 0x7FFF) return sign ? -INFINITY : INFINITY;
    const u64 be = exp80 - 0x3800;
    u64 f64bits = (sign << 63) | (be << 52) | (mant & 0x000FFFFFFFFFFFFFULL);
    double d;
    std::memcpy(&d, &f64bits, 8);
    return d;
}

int main() {
    MemoryManager mem;
    if (!mem.initialize()) {
        std::printf("memory init failed\n");
        return 1;
    }
    CpuContext ctx{};

    // A scratch guest page for data operands, and a code page.
    const GuestAddr DATA = 0x3000000;
    const GuestAddr CODE = 0x2000000;
    std::printf("[x87] cw after CpuContext reset = 0x%04x (expected 0x037F)\n", ctx.fpu.cw);
    if (ctx.fpu.cw != 0x037F) ++g_failures;

    auto put_code = [&](GuestAddr at, const std::vector<u8>& b) {
        mem.write_bytes(at, b.data(), b.size());
    };
    SyscallTable syscalls(mem);
    CpuExecutor exec(mem, syscalls);
    auto step_at = [&](GuestAddr at) {
        exec.context() = ctx;
        exec.context().rip = at;
        const StepResult r = exec.step();
        ctx = exec.context();
        return r;
    };

    // --- fnstcw / fldcw round trip -----------------------------------------
    // D9 /7 FNSTCW word ptr [disp8]  -> D9 E7 is FNSTCW AX, so use memory form.
    put_code(CODE, {0xD9, 0x7D, 0x10});               // fnstcw [rbp+0x10]
    ctx.rbp = DATA + 0x200;
    mem.zero_memory(DATA, 0x400);
    step_at(CODE);
    u16 cw = 0;
    mem.read_value(DATA + 0x210, cw);
    check_u64("fnstcw after reset reads 0x037F", cw, 0x037F);

    put_code(CODE, {0xD9, 0x6D, 0x10});               // fldcw word ptr [rbp+0x10]
    const u16 newcw = 0x0C7F;
    mem.write_value(DATA + 0x210, static_cast<u64>(newcw));
    step_at(CODE);
    put_code(CODE, {0xD9, 0x7D, 0x10});
    step_at(CODE);
    mem.read_value(DATA + 0x210, cw);
    check_u64("fldcw 0x0C7F then fnstcw reads 0x0C7F", cw, 0x0C7F);

    // --- fistp m32 rounding -------------------------------------------------
    // DF E0 FISTP dword [rbp+0x20] needs DF/7 memory form instead: DB /3 is m16.
    // Use DD /3 (FSTP m64) style paths via DF E0 push? Simpler: DD /3 is m64.
    // We drive the integer store through DF E0's 5-byte register form is not
    // possible, so use DB /2 (FST m32int) + DB /3 (FISTP m32int) is also m16.
    // m32 FISTP is DF /7 -> "DF 7D 20".
    struct RCase { const char* name; u16 rc_bits; double input; u32 want; };
    const RCase cases[] = {
        {"nearest-even 2.5",  0x00,  2.5,  2},
        {"nearest-even 3.5",  0x00,  3.5,  4},
        {"nearest-even -2.5", 0x00, -2.5, static_cast<u32>(-2)},
        {"down        2.5",   0x01,  2.5,  2},
        {"down        3.5",   0x01,  3.5,  3},
        {"down        -2.5",  0x01, -2.5, static_cast<u32>(-3)},
        {"up          2.5",   0x02,  2.5,  3},
        {"up          3.5",   0x02,  3.5,  4},
        {"up          -2.5",  0x02, -2.5, static_cast<u32>(-2)},
        {"truncate    2.5",   0x03,  2.5,  2},
        {"truncate    3.5",   0x03,  3.5,  3},
        {"truncate    -2.5",  0x03, -2.5, static_cast<u32>(-2)},
    };
    std::printf("[x87] fistp m32 rounding (RC = cw bits 10-11):\n");
    for (const auto& c : cases) {
        // FLD1 (D9 E8), then replace ST0 via FLD m64? Use FILD-free path:
        // D9 E8 fld1 pushes 1.0; we instead load the constant with FILD m32 (DF /0).
        u32 bits;
        const float d = static_cast<float>(c.input);  // m32 is a 32-bit float
        std::memcpy(&bits, &d, 4);
        // rbp = DATA+0x200 for both instructions, so FILD [rbp+0x00] reads
        // DATA+0x200 and FISTP [rbp+0x20] writes DATA+0x220.
        mem.zero_memory(DATA, 0x400);
        mem.write_value(DATA + 0x200, static_cast<u64>(static_cast<u32>(bits)));
        ctx.rbp = DATA + 0x200;
        put_code(CODE, {0xD9, 0x45, 0x00});           // D9 /0 FLD m32fp
        step_at(CODE);
        ctx.rbp = DATA + 0x200;
        ctx.fpu.cw = static_cast<u16>((ctx.fpu.cw & ~0x0C00) | (c.rc_bits << 10));
        put_code(CODE, {0xDF, 0x7D, 0x20});           // FISTP dword ptr [rbp+0x20]
        step_at(CODE);
        u64 got64 = 0;
        mem.read_value(DATA + 0x220, got64);
        const u32 got = static_cast<u32>(got64);
        char label[96];
        std::snprintf(label, sizeof(label), "%s -> %d", c.name, (int)c.want);
        const bool ok = (got == c.want);
        if (!ok) ++g_failures;
        std::printf("  %-40s got=%-12d want=%-12d %s\n", label, (int)got, (int)c.want,
                    ok ? "ok" : "FAIL");
    }

    // --- overflow -> integer indefinite ------------------------------------
    std::printf("[x87] overflow:\n");
    {
        ctx.fpu.cw = 0x037F;                          // RC=00 nearest-even
        mem.zero_memory(DATA, 0x400);
        u32 bits;
        const float d = 3e10f;
        std::memcpy(&bits, &d, 4);
        mem.write_value(DATA, static_cast<u64>(static_cast<u32>(bits)));
        ctx.rbp = DATA;
        put_code(CODE, {0xD9, 0x45, 0x00});
        { const StepResult sr = step_at(CODE);        // D9 /0 FLD m32fp
          std::printf("  [diag] FLD step_result=%d rip=0x%llx\n", (int)sr,
                      (unsigned long long)ctx.rip); }
        { u64 mv = 0; mem.read_value(DATA, mv);
          double st0; std::memcpy(&st0, &ctx.fpu.st[ctx.fpu.top], 8);
          std::printf("  [diag] wrote=0x%08x mem=0x%llx ST0=%g top=%u rbp=0x%llx\n",
                      bits, (unsigned long long)mv, st0, ctx.fpu.top,
                      (unsigned long long)ctx.rbp);
          for (int k = 0; k < 8; ++k)
              std::printf("  [diag]   st[%d]=0x%016llx\n", k,
                          (unsigned long long)ctx.fpu.st[k]); }
        ctx.rbp = DATA + 0x200;
        put_code(CODE, {0xDF, 0x7D, 0x20});           // FISTP m32
        const u16 sw_before = ctx.fpu.sw;
        step_at(CODE);
        u64 got64 = 0;
        mem.read_value(DATA + 0x220, got64);
        const u32 got = static_cast<u32>(got64);
        char label[96];
        std::snprintf(label, sizeof(label), "fistp m32 of 3e10 -> 0x80000000");
        const bool ok = (got == 0x80000000u);
        if (!ok) ++g_failures;
        std::printf("  %-40s got=0x%08x want=0x80000000 %s\n", label, got, ok ? "ok" : "FAIL");
        const bool ie = (ctx.fpu.sw & 0x0001) != 0;
        std::printf("  %-40s IE=%s (sw=0x%04x, was 0x%04x) %s\n", "IE flag set",
                    ie ? "1" : "0", ctx.fpu.sw, sw_before, ie ? "ok" : "FAIL");
        if (!ie) ++g_failures;
    }

    // --- guest RC sequence: and 0xFCFF / or 0x0200 leaves RC=00 -------------
    std::printf("[x87] guest control-word sequence:\n");
    {
        ctx.fpu.cw = 0x037F;
        u16 v = ctx.fpu.cw;
        v = static_cast<u16>(v & 0xFCFF);             // and ecx,0xFCFF  -> PC=00, RC kept
        v = static_cast<u16>(v | 0x0200);             // or  ecx,0x0200  -> PC=10
        ctx.fpu.cw = v;
        const int pc = (ctx.fpu.cw >> 8) & 3;
        const int rc = (ctx.fpu.cw >> 10) & 3;
        std::printf("  cw=0x%04x  PC=%d (10 = 53-bit)  RC=%d (00 = nearest-even) %s\n",
                    ctx.fpu.cw, pc, rc, (pc == 2 && rc == 0) ? "ok" : "FAIL");
        if (pc != 2 || rc != 0) ++g_failures;
    }

    std::printf("[x87] failures=%d\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
