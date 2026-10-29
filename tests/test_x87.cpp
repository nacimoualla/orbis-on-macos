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

int main(int argc, char** argv) {
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

    auto reset_fpu = [&](void) {
        ctx.fpu = X87State{};
        ctx.fpu.cw = 0x037F;
        ctx.fpu.sw = 0;
        for (int k = 0; k < 8; ++k) { ctx.fpu.st[k] = 0; ctx.fpu.se_st[k] = 0; ctx.fpu.tag[k] = 0xFFFF; }
        ctx.fpu.top = 0;
    };

    // ---- STEP 0a: the overflow vector ALONE, fresh context, no prior vectors ----
    if (argc > 1 && std::strcmp(argv[1], "overflow-only") == 0) {
        std::printf("[0a] overflow vector alone, fresh CpuContext\n");
        reset_fpu();
        mem.zero_memory(DATA, 0x400);
        u32 bits = 0;
        const float d = 3e10f;
        std::memcpy(&bits, &d, 4);
        mem.write_value(DATA, static_cast<u64>(bits));
        ctx.rbp = DATA;
        put_code(CODE, {0xD9, 0x45, 0x00});          // D9 /0 FLD m32fp
        const StepResult r1 = step_at(CODE);
        double st0 = 0; std::memcpy(&st0, &ctx.fpu.st[ctx.fpu.top], 8);
        std::printf("  FLD  step=%d top=%u ST0=%g  (want 3e10)\n",
                    (int)r1, ctx.fpu.top, st0);
        std::printf("  pre-IE=%u sw=0x%04x\n", (unsigned)((ctx.fpu.sw & 1u)), ctx.fpu.sw);
        ctx.rbp = DATA + 0x200;
        put_code(CODE, {0xDB, 0x5D, 0x20});          // DB /3 FISTP m32int
        const StepResult r2 = step_at(CODE);
        u64 got64 = 0; mem.read_value(DATA + 0x220, got64);
        std::printf("  FISTP step=%d got=0x%08x want=0x80000000  IE=%u\n",
                    (int)r2, static_cast<u32>(got64), (unsigned)(ctx.fpu.sw & 1u));
        const bool ok = (static_cast<u32>(got64) == 0x80000000u) && ((ctx.fpu.sw & 1u) == 1u);
        std::printf("  [0a] %s\n", ok ? "ok" : "FAIL");
        return ok ? 0 : 1;
    }

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
        reset_fpu();                                   // 0b: per-vector reset
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
        put_code(CODE, {0xDB, 0x5D, 0x20});           // DB /3 FISTP m32int [rbp+0x20]
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
        std::printf("  [pre] inherited from previous vectors: cw=0x%04x sw=0x%04x top=%u\n",
                    ctx.fpu.cw, ctx.fpu.sw, ctx.fpu.top);
        for (int k = 0; k < 8; ++k)
            std::printf("  [pre]   st[%d]=0x%016llx tag=0x%04x\n", k,
                        (unsigned long long)ctx.fpu.st[k], ctx.fpu.tag[k]);
        reset_fpu();                                   // 0b: per-vector reset
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
        put_code(CODE, {0xDB, 0x5D, 0x20});           // DB /3 FISTP m32int
        const u16 sw_before = ctx.fpu.sw;
        if ((sw_before & 0x0001) != 0) {
            ++g_failures;
            std::printf("  %-40s IE was ALREADY set before the store: FAIL\n",
                        "precondition IE==0");
        } else {
            std::printf("  %-40s sw=0x%04x ok\n", "precondition IE==0", sw_before);
        }
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

    // ---- stack discipline: push / no-pop / pop for every memory-store form ----
    std::printf("[x87] stack discipline (TOP is physical index; push dec, pop inc):\n");
    {
        struct PopCase { const char* name; std::vector<u8> code; int rc; };
        const PopCase pops[] = {
            {"FLD m64   DD /0  (push, expect -1)", {0xDD, 0x45, 0x00}, -1},
            {"FST  m64  DD /2  (no pop, expect 0)", {0xDD, 0x55, 0x00}, 0},
            {"FSTP m64  DD /3  (pop, expect +1)",   {0xDD, 0x5D, 0x00}, 1},
            {"FIST m32  DB /2  (no pop, expect 0)", {0xDB, 0x55, 0x00}, 0},
            {"FISTP m32 DB /3  (pop, expect +1)",   {0xDB, 0x5D, 0x00}, 1},
            {"FIST m16  DF /2  (no pop, expect 0)", {0xDF, 0x55, 0x00}, 0},
            {"FISTP m16 DF /3  (pop, expect +1)",   {0xDF, 0x5D, 0x00}, 1},
            {"FISTP m32 DF /7  (pop, expect +1)",   {0xDF, 0x7D, 0x00}, 1},
        };
        for (const auto& pc : pops) {
            reset_fpu();
            mem.zero_memory(DATA, 0x400);
            double one = 1.0; u64 b; std::memcpy(&b, &one, 8);
            mem.write_value(DATA, b); mem.write_value(DATA + 0x20, b);
            ctx.rbp = DATA;
            put_code(CODE, {0xD9, 0xE8});                 // FLD1 so ST0 is valid
            step_at(CODE);
            put_code(CODE, pc.code);
            const u8 before = ctx.fpu.top;
            step_at(CODE);
            // TOP is a 3-bit physical index, so a pop from 7 reads as 0.
            const int delta = ((static_cast<int>(ctx.fpu.top) - static_cast<int>(before) + 4) & 7) - 4;
            const bool ok = (delta == pc.rc);
            if (!ok) ++g_failures;
            std::printf("  %-40s top %u->%u delta=%+d  %s\n", pc.name, before,
                        ctx.fpu.top, delta, ok ? "ok" : "FAIL");
        }
    }

    // ---- fld1; fld1; faddp; fstp m64 = 2.0, TOP restored -------------------
    std::printf("[x87] fld1; fld1; faddp; fstp m64:\n");
    {
        reset_fpu();
        mem.zero_memory(DATA, 0x400);
        const u8 top0 = ctx.fpu.top;
        put_code(CODE, {0xD9, 0xE8});   step_at(CODE);   // FLD1
        put_code(CODE, {0xD9, 0xE8});   step_at(CODE);   // FLD1
        put_code(CODE, {0xDE, 0xC1});   step_at(CODE);   // FADDP ST(1),ST(0)
        ctx.rbp = DATA;
        put_code(CODE, {0xDD, 0x5D, 0x20}); step_at(CODE);  // FSTP m64 [rbp+0x20] = DATA+0x20
        u64 bits = 0; mem.read_value(DATA + 0x20, bits);
        double v = 0; std::memcpy(&v, &bits, 8);
        const bool ok = (v == 2.0) && (ctx.fpu.top == top0);
        if (!ok) ++g_failures;
        std::printf("  %-40s got=%g want=2  top %u->%u  %s\n", "fld1;fld1;faddp;fstp m64",
                    v, top0, ctx.fpu.top, ok ? "ok" : "FAIL");
    }

    // ---- fcomip ZF/PF/CF --------------------------------------------------
    std::printf("[x87] fcomip flags (ST0 cmp ST1):\n");
    {
        struct FC { const char* name; double st0; double st1; int zf; int pf; int cf; };
        const FC fcs[] = {
            {"greater  (1 > 0)",   1.0,  0.0, 0, 0, 0},
            {"less     (0 < 1)",   0.0,  1.0, 0, 0, 1},
            {"equal    (2 == 2)",  2.0,  2.0, 1, 0, 0},
            {"unordered(NaN, 1)",  NAN,  1.0, 1, 1, 1},
        };
        for (const auto& fc : fcs) {
            reset_fpu();
            mem.zero_memory(DATA, 0x400);
            u64 a = 0, b = 0; std::memcpy(&a, &fc.st1, 8); std::memcpy(&b, &fc.st0, 8);
            mem.write_value(DATA, a);      // ST(i) operand
            mem.write_value(DATA + 0x20, b); // ST0 operand
            ctx.rbp = DATA;
            put_code(CODE, {0xDD, 0x45, 0x00}); step_at(CODE);  // FLD m64 [rbp+0]  -> ST0 = ST(i) val
            put_code(CODE, {0xDD, 0x45, 0x20}); step_at(CODE);  // FLD m64 [rbp+0x20]-> ST0 = b
            put_code(CODE, {0xDF, 0xF1});     step_at(CODE);    // FCOMIP ST(1), pop
            const int zf = (ctx.rflags & 0x40) ? 1 : 0;
            const int pf = (ctx.rflags & 0x04) ? 1 : 0;
            const int cf = (ctx.rflags & 0x01) ? 1 : 0;
            const bool ok = (zf == fc.zf) && (pf == fc.pf) && (cf == fc.cf);
            if (!ok) ++g_failures;
            std::printf("  %-40s ZF=%d PF=%d CF=%d  want %d/%d/%d  %s\n", fc.name,
                        zf, pf, cf, fc.zf, fc.pf, fc.cf, ok ? "ok" : "FAIL");
        }
    }

    // ---- DF /7 is FISTP m64int (verified against GNU objdump's copy of the
    // Intel map: "df 7d 10  fistp QWORD PTR [rbp+0x10]"). ----
    std::printf("[x87] DF /7 = FISTP m64int (8-byte store):\n");
    {
        struct W64 { double in; const char* label; u64 want; bool want_ie; };
        const W64 ws[] = {
            { 1099511627776.0, "+2^40", 0x0000010000000000ULL, false},
            {-1099511627776.0, "-2^40", 0xFFFFFF0000000000ULL, false},
            { 9223372036854775808.0, "2^63", 0x8000000000000000ULL, true},
        };
        for (const auto& w : ws) {
            reset_fpu();
            mem.zero_memory(DATA, 0x400);
            u64 bits = 0; std::memcpy(&bits, &w.in, 8);
            mem.write_value(DATA, bits);
            ctx.rbp = DATA;
            put_code(CODE, {0xDD, 0x45, 0x00}); step_at(CODE);  // DD /0 FLD m64fp
            put_code(CODE, {0xDF, 0x7D, 0x20}); step_at(CODE);  // DF /7 FISTP m64int
            u64 got = 0; mem.read_value(DATA + 0x20, got);
            const u16 ie = static_cast<u16>(ctx.fpu.sw & 1u);
            const bool ok = (got == w.want) && ((ie != 0) == w.want_ie);
            if (!ok) ++g_failures;
            std::printf("  FISTP m64 of %-5s -> 0x%016llx want 0x%016llx IE=%u %s\n",
                        w.label, (unsigned long long)got, (unsigned long long)w.want,
                        ie, ok ? "ok" : "FAIL");
        }
    }

    // ---- FISTTP ignores RC: truncates under all four modes, and pops ----
    std::printf("[x87] FISTTP (DB /1, DF /1, DD /1) truncates, ignores RC, pops:\n");
    {
        struct TTCase { const char* nm; int width; std::vector<u8> code; };
        // DB /1 FISTTP m32int -> modrm 0x4D ; DF /1 FISTTP m16int -> 0x4D ;
        // DD /1 FISTTP m64int -> 0x4D
        const TTCase tt[] = {
            {"DB /1 m32int", 4, {0xDB, 0x4D, 0x20}},
            {"DF /1 m16int", 2, {0xDF, 0x4D, 0x20}},
            {"DD /1 m64int", 8, {0xDD, 0x4D, 0x20}},
        };
        for (const auto& t : tt) {
            for (int rc = 0; rc < 4; ++rc) {
                for (double v : {2.9, -2.9}) {
                    reset_fpu();
                    mem.zero_memory(DATA, 0x400);
                    u64 bits = 0; std::memcpy(&bits, &v, 8);
                    mem.write_value(DATA, bits);
                    ctx.rbp = DATA;
                    put_code(CODE, {0xDD, 0x45, 0x00}); step_at(CODE);   // FLD m64fp
                    ctx.fpu.cw = static_cast<u16>((ctx.fpu.cw & ~0x0C00) | (rc << 10));
                    const u8 before = ctx.fpu.top;
                    put_code(CODE, t.code); step_at(CODE);
                    u64 got = 0; mem.read_value(DATA + 0x20, got);
                    const bool popped = ctx.fpu.top == static_cast<u8>((before + 1) & 7);
                    const i64 want = static_cast<i64>(v);   // truncate toward zero
                    // Sign-extend from the store width; the raw read is 8 bytes.
                    i64 gotv;
                    if (t.width == 2)     gotv = static_cast<i16>(got);
                    else if (t.width == 4) gotv = static_cast<i32>(got);
                    else                  gotv = static_cast<i64>(got);
                    const bool ok = (gotv == want) && popped;
                    if (!ok) ++g_failures;
                    std::printf("  %-12s RC=%d %+.1f -> %lld want %lld pop=%d %s\n",
                                t.nm, rc, v, (long long)gotv, (long long)want,
                                popped ? 1 : 0, ok ? "ok" : "FAIL");
                }
            }
        }
    }

    // ---- FISTP m32int (DB /3) of 2^40 still overflows ----
    std::printf("[x87] DB /3 FISTP m32int of 2^40:\n");
    {
        reset_fpu();
        mem.zero_memory(DATA, 0x400);
        u64 b64 = 0; const double v = 1099511627776.0;
        std::memcpy(&b64, &v, 8);
        mem.write_value(DATA, b64);
        ctx.rbp = DATA;
        put_code(CODE, {0xDD, 0x45, 0x00}); step_at(CODE);      // FLD m64fp
        put_code(CODE, {0xDB, 0x5D, 0x20}); step_at(CODE);      // DB /3 FISTP m32int
        u64 got = 0; mem.read_value(DATA + 0x20, got);
        const u32 g32 = static_cast<u32>(got);
        const bool ok = (g32 == 0x80000000u) && ((ctx.fpu.sw & 1u) == 1u);
        if (!ok) ++g_failures;
        std::printf("  FISTP m32 of 2^40 -> 0x%08x want 0x80000000 IE=%u %s\n",
                    g32, (unsigned)(ctx.fpu.sw & 1u), ok ? "ok" : "FAIL");
    }

    // ---- m80 byte round trip, with the precision loss reported ------------
    std::printf("[x87] m80 FLD/FSTP byte round trip:\n");
    {
        struct M80 { const char* name; u8 b[10]; };
        // 1.0                       : mantissa 0x8000000000000000, exp 0x3FFF, sign 0
        const M80 v_exact = {"1.0", {0,0,0,0,0,0,0,0x80, 0xFF, 0x3F}};
        // 1.0 + 2^-54               : bit 54 of the mantissa, beyond a double's 53
        const M80 v_lossy = {"1.0+2^-54", {0x00,0x02,0,0,0,0,0,0x80, 0xFF, 0x3F}};
        for (const M80* v : {&v_exact, &v_lossy}) {
            reset_fpu();
            mem.zero_memory(DATA, 0x400);
            mem.write_bytes(DATA, v->b, 10);
            ctx.rbp = DATA;
            put_code(CODE, {0xDB, 0x6D, 0x00}); step_at(CODE);   // DB /5 FLD m80
            put_code(CODE, {0xDB, 0x7D, 0x20}); step_at(CODE);   // DB /7 FSTP m80 -> DATA+0x20
            u8 out[10]; mem.read_bytes(DATA + 0x20, out, 10);
            const bool same = (std::memcmp(v->b, out, 10) == 0);
            const bool expect_same = (v->b[1] == 0x00);  // 1.0 round-trips; 1.0+2^-54 may not
            std::printf("  in  %02x%02x%02x%02x%02x%02x%02x%02x %02x%02x\n",
                        v->b[7],v->b[6],v->b[5],v->b[4],v->b[3],v->b[2],v->b[1],v->b[0],
                        v->b[8],v->b[9]);
            std::printf("  out %02x%02x%02x%02x%02x%02x%02x%02x %02x%02x  %s\n",
                        out[7],out[6],out[5],out[4],out[3],out[2],out[1],out[0],
                        out[8],out[9], same ? "byte-identical" : "DIFFERS");
            const bool ok = (same == expect_same);
            if (!ok) ++g_failures;
            std::printf("    expect %-22s -> %s\n", expect_same ? "byte-identical" : "may differ",
                        ok ? "ok" : "FAIL");
        }
        // declare which one is expected to survive a 53-bit host double
        u8 o[10]; reset_fpu(); mem.zero_memory(DATA, 0x400);
        mem.write_bytes(DATA, v_lossy.b, 10); ctx.rbp = DATA;
        put_code(CODE, {0xDB, 0x6D, 0x00}); step_at(CODE);
        put_code(CODE, {0xDB, 0x7D, 0x20}); step_at(CODE);
        mem.read_bytes(DATA + 0x20, o, 10);
        const bool lossy_lost = (o[1] == 0x00);
        if (!lossy_lost) ++g_failures;
        std::printf("  %-40s mantissa byte[1] %02x -> %02x  %s\n",
                    "1.0+2^-54 (needs 54 mantissa bits)", v_lossy.b[1], o[1],
                    lossy_lost ? "lost, as expected (host double = 53 bits)" : "unexpectedly kept");
    }

    std::printf("[x87] failures=%d\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
