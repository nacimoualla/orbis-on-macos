// Differential fuzzer: x86_64 -> ARM64 JIT vs. the interpreter (reference).
//
// Each trial generates a random straight-line sequence of x86_64 instructions
// from a table of encoding templates, terminates it with HLT, randomizes all
// 16 GPRs + RFLAGS and a scratch data region, then runs it twice from the
// identical state:
//   1. interpreter only (CpuExecutor::step until HLT)            -> reference
//   2. JIT attached (JitCompiler::execute_block, interpreter for
//      whatever the JIT declines, exactly like CpuExecutor::run)  -> candidate
// and compares GPRs, RIP, the arithmetic flags the sequence leaves defined,
// and the scratch/stack memory. Mismatches are shrunk (delta debugging) to a
// minimal failing sequence and reported with hex bytes.
//
// Run:  scripts/cpu_harness.sh tests/test_jit_diff.cpp
//       (log: build/aarch64-harness/test_jit_diff.log), or scripts/jit_fuzz.sh
//
// Environment:
//   JIT_DIFF_SEED=<n>        base seed (default 1). Trial i uses (seed, i).
//   JIT_DIFF_TRIALS=<n>      number of trials (default 2000)
//   JIT_DIFF_TRIAL=<i>       run only trial i (reproduce a reported failure)
//   JIT_DIFF_LEN=<n|a..b>    instructions per trial (default 1..8)
//   JIT_DIFF_GROUPS=a,b      only these template groups (also argv[1]);
//                            `--list` (or JIT_DIFF_LIST=1) prints all groups
//   JIT_DIFF_CHECK_FLAGS=0   ignore arithmetic flags (e.g. before the JIT
//                            computes flags) -- default 1
//   JIT_DIFF_CHECK_AF=1      also compare AF (off by default: the interpreter
//                            does not model AF for ADD/SUB/CMP)
//   JIT_DIFF_MAX_REPORTS=<n> detailed reports for the first n unique
//                            minimized failures (default 3)
//   JIT_DIFF_STOP_ON_FAIL=1  stop after the first failing trial
//   JIT_DIFF_VERBOSE=1       per-template coverage table, and keep the
//                            interpreter's stderr (muted by default: a
//                            mis-translating JIT can send it into garbage)
//
// Extending: add a Template in build_templates(). Most ModRM instructions
// need no code -- set opcode, operand roles, imm size and flag behaviour and
// the generic gen_modrm() encodes register and memory forms (all mod/SIB/disp
// /RIP-relative variants, REX.W/R/X/B). Memory forms (mem = true) are kept in
// bounds automatically: every trial that contains one reserves "pointer"
// registers (pointing into the scratch region) and an "index" register that
// no other instruction in the sequence may write.

#include "ps4/cpu_context.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace ps4;

namespace {

// -----------------------------------------------------------------------------
// Guest layout
// -----------------------------------------------------------------------------
constexpr GuestAddr CODE_BASE = 0x100000;
constexpr GuestAddr SCRATCH_BASE = 0x200000;
constexpr usize SCRATCH_SIZE = 0x4000;
constexpr GuestAddr STACK_BASE = 0x300000;
constexpr usize STACK_SIZE = 0x2000;
constexpr int MAX_STEPS = 512;
constexpr int STATUS_STEP_LIMIT = -1;

constexpr u64 F_CF = CpuContext::FLAG_CF;
constexpr u64 F_PF = CpuContext::FLAG_PF;
constexpr u64 F_AF = CpuContext::FLAG_AF;
constexpr u64 F_ZF = CpuContext::FLAG_ZF;
constexpr u64 F_SF = CpuContext::FLAG_SF;
constexpr u64 F_OF = CpuContext::FLAG_OF;
constexpr u64 ARITH_FLAGS = F_CF | F_PF | F_AF | F_ZF | F_SF | F_OF;
constexpr u64 FS_ARITH = ARITH_FLAGS;                   // ADD/SUB/CMP: all defined
constexpr u64 FS_LOGIC = F_CF | F_PF | F_ZF | F_SF | F_OF; // AND/OR/XOR/TEST
constexpr u64 FU_LOGIC = F_AF;                          //   ... AF undefined

constexpr u32 NEED_PTR = 1;   // uses memory operands -> pointer/index registers
constexpr u32 NEED_STACK = 2; // pushes/pops -> RSP must point at the stack

const char* const REG_NAMES[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                   "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
Reg64 CpuContext::* const GPR_FIELDS[16] = {
    &CpuContext::rax, &CpuContext::rcx, &CpuContext::rdx, &CpuContext::rbx,
    &CpuContext::rsp, &CpuContext::rbp, &CpuContext::rsi, &CpuContext::rdi,
    &CpuContext::r8,  &CpuContext::r9,  &CpuContext::r10, &CpuContext::r11,
    &CpuContext::r12, &CpuContext::r13, &CpuContext::r14, &CpuContext::r15};

// -----------------------------------------------------------------------------
// Environment helpers
// -----------------------------------------------------------------------------
u64 env_u64(const char* name, u64 def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    return std::strtoull(v, nullptr, 0);
}

std::string env_str(const char* name, const std::string& def = "") {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : def;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

// -----------------------------------------------------------------------------
// RNG (splitmix64: tiny, deterministic across hosts)
// -----------------------------------------------------------------------------
struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed) {}
    u64 next() {
        u64 z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    u64 below(u64 n) { return n ? next() % n : 0; }
    bool coin() { return (next() & 1) != 0; }
    i64 range(i64 lo, i64 hi) { return lo + static_cast<i64>(below(static_cast<u64>(hi - lo + 1))); }
};

// Register/immediate values biased toward flag and width boundaries.
u64 interesting_u64(Rng& r) {
    static const u64 table[] = {0,
                                1,
                                ~0ULL,
                                0x7F,
                                0x80,
                                0xFF,
                                0x7FFF,
                                0x8000,
                                0xFFFF,
                                0x7FFFFFFFULL,
                                0x80000000ULL,
                                0xFFFFFFFFULL,
                                0x100000000ULL,
                                0xFFFFFFFF80000000ULL,
                                0x7FFFFFFFFFFFFFFFULL,
                                0x8000000000000000ULL};
    switch (r.below(8)) {
        case 0:
        case 1: return table[r.below(sizeof(table) / sizeof(table[0]))];
        case 2: return static_cast<u64>(r.range(-256, 256));
        case 3: return r.next() & 0xFFFFFFFFULL;
        default: return r.next();
    }
}

u32 interesting_imm32(Rng& r) {
    switch (r.below(6)) {
        case 0: return 0;
        case 1: return static_cast<u32>(r.range(-128, 127));
        case 2: {
            static const u32 t[] = {1, 0xFFFFFFFFu, 0x7FFFFFFFu, 0x80000000u, 0xFFFFu, 0x8000u};
            return t[r.below(6)];
        }
        default: return static_cast<u32>(r.next());
    }
}

u8 interesting_imm8(Rng& r) {
    switch (r.below(5)) {
        case 0: return 0;
        case 1: return 0x7F;
        case 2: return 0x80;
        case 3: return 0xFF;
        default: return static_cast<u8>(r.next());
    }
}

void push32(std::vector<u8>& v, u32 x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<u8>(x >> (8 * i)));
}

void push64(std::vector<u8>& v, u64 x) {
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<u8>(x >> (8 * i)));
}

std::string hex_bytes(const std::vector<u8>& b) {
    std::string s;
    char buf[4];
    for (usize i = 0; i < b.size(); ++i) {
        std::snprintf(buf, sizeof(buf), "%02x", b[i]);
        if (i) s += ' ';
        s += buf;
    }
    return s;
}

std::string hex64(u64 v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(v));
    return buf;
}

std::string flag_names(u64 f) {
    std::string s;
    const std::pair<u64, const char*> names[] = {{F_CF, "CF"}, {F_PF, "PF"}, {F_AF, "AF"},
                                                 {F_ZF, "ZF"}, {F_SF, "SF"}, {F_OF, "OF"}};
    for (const auto& [bit, n] : names) {
        if (f & bit) {
            if (!s.empty()) s += ' ';
            s += n;
        }
    }
    return s.empty() ? "-" : s;
}

// -----------------------------------------------------------------------------
// Instruction templates
// -----------------------------------------------------------------------------
struct Insn {
    u16 tmpl{0};
    std::vector<u8> bytes;
};

// Per-trial register discipline handed to generators.
struct Gen {
    Rng& rng;
    std::vector<u8> free_regs; // may be written by instructions
    std::vector<u8> base_regs; // point into scratch (or RSP -> stack); read-only
    int index_reg{-1};         // small value usable as SIB index; read-only
    GuestAddr insn_addr{0};    // address the instruction will be placed at

    u8 pick_free() { return free_regs[rng.below(free_regs.size())]; }
    u8 pick_any() { return static_cast<u8>(rng.below(16)); }
};

enum Role : u8 {
    R_ANY,   // read-only register operand
    R_DST,   // written register operand (32/64-bit)
    R_DST8,  // written 8-bit register operand (handles AH..BH vs SPL..DIL)
    R_DIGIT  // ModRM.reg is an opcode extension (/digit)
};

enum WMode : u8 { W_RAND, W_0, W_1 };

struct Template;
using GenFn = void (*)(Gen&, const Template&, Insn&);

struct Template {
    std::string group;
    std::string name;
    GenFn gen{nullptr};
    u16 opcode{0};          // 0x0Fxx for two-byte opcodes
    Role reg_role{R_ANY};
    Role rm_role{R_ANY};    // register (mod=3) forms only
    bool mem{false};        // r/m is a memory operand
    u8 digit{0};
    u8 imm{0};              // immediate size in bytes (0, 1, 4)
    WMode wmode{W_RAND};
    bool byte_regs{false};  // 8-bit register semantics: no spurious REX
    u64 fset{0};            // arithmetic flags this instruction defines
    u64 fundef{0};          // arithmetic flags it leaves undefined
    u32 needs{0};
    u8 ext{0};              // free parameter for custom generators
};

// REX/opcode/tail assembly.
struct Enc {
    bool w{false};
    u8 r{0}, x{0}, b{0};
    bool force_rex{false}; // e.g. SPL..DIL byte registers
    bool no_rex{false};    // AH..BH byte registers
    std::vector<u8> prefix;
    std::vector<u8> op;
    std::vector<u8> tail;
};

void finish(Gen& g, const Enc& e, Insn& out, bool allow_spurious_rex) {
    out.bytes = e.prefix;
    const u8 rex = static_cast<u8>(0x40 | (e.w << 3) | (e.r << 2) | (e.x << 1) | e.b);
    assert(!(e.no_rex && (rex != 0x40 || e.force_rex)));
    const bool spurious = allow_spurious_rex && !e.no_rex && g.rng.below(6) == 0;
    if (rex != 0x40 || e.force_rex || spurious) out.bytes.push_back(rex);
    out.bytes.insert(out.bytes.end(), e.op.begin(), e.op.end());
    out.bytes.insert(out.bytes.end(), e.tail.begin(), e.tail.end());
}

struct MemOp {
    u8 x{0}, b{0};
    std::vector<u8> bytes; // ModRM (reg field 0) [SIB] [disp]
};

// Random in-bounds memory operand: [base], [base+d8], [base+d32],
// [base+idx*s(+d)], [idx*s+d32], [d32], [rip+d32].
MemOp gen_mem(Gen& g, bool allow_rip) {
    MemOp m;
    assert(!g.base_regs.empty() && g.index_reg >= 0);
    const int mode = static_cast<int>(g.rng.below(allow_rip ? 9 : 8));
    auto rand_disp32 = [&] { return static_cast<u32>(static_cast<i32>(g.rng.range(-0x800, 0x800))); };
    auto scratch_abs = [&] { return static_cast<u32>(SCRATCH_BASE + 0x1000 + g.rng.below(0x2000)); };

    if (mode == 8) { // RIP-relative (target approximate: margins absorb shrinking shifts)
        m.bytes = {0x05};
        const GuestAddr target = SCRATCH_BASE + 0x800 + g.rng.below(0x3000);
        push32(m.bytes, static_cast<u32>(target - (g.insn_addr + 8)));
        return m;
    }
    if (mode == 7) { // [disp32] via SIB (no base, no index)
        m.bytes = {0x04, static_cast<u8>((g.rng.below(4) << 6) | (4 << 3) | 5)};
        push32(m.bytes, scratch_abs());
        return m;
    }
    if (mode == 6) { // [index*scale + disp32], no base. REX.B kept 0: the
                     // interpreter mis-decodes base=101+REX.B+mod=00 as [r13].
        const u8 idx = static_cast<u8>(g.index_reg);
        m.x = idx >> 3;
        m.bytes = {0x04, static_cast<u8>((g.rng.below(4) << 6) | ((idx & 7) << 3) | 5)};
        push32(m.bytes, scratch_abs() - 0x400);
        return m;
    }

    const u8 base = g.base_regs[g.rng.below(g.base_regs.size())];
    m.b = base >> 3;
    u8 mod = static_cast<u8>(mode % 3);
    const bool with_index = mode >= 3;
    if (mod == 0 && (base & 7) == 5) mod = 1; // [rbp]/[r13] need a displacement
    const bool sib = with_index || (base & 7) == 4;
    if (sib) {
        u8 idx_field = 4; // "no index"
        if (with_index) {
            idx_field = static_cast<u8>(g.index_reg) & 7;
            m.x = static_cast<u8>(g.index_reg) >> 3;
        }
        m.bytes = {static_cast<u8>((mod << 6) | 4),
                   static_cast<u8>((g.rng.below(4) << 6) | (idx_field << 3) | (base & 7))};
    } else {
        m.bytes = {static_cast<u8>((mod << 6) | (base & 7))};
    }
    if (mod == 1) m.bytes.push_back(static_cast<u8>(g.rng.range(-128, 127)));
    if (mod == 2) push32(m.bytes, rand_disp32());
    return m;
}

void push_imm(Gen& g, std::vector<u8>& v, u8 size) {
    if (size == 1) v.push_back(interesting_imm8(g.rng));
    else if (size == 4) push32(v, interesting_imm32(g.rng));
}

// Generic ModRM instruction: [REX] opcode ModRM [SIB] [disp] [imm].
void gen_modrm(Gen& g, const Template& t, Insn& out) {
    Enc e;
    e.w = t.wmode == W_RAND ? g.rng.coin() : t.wmode == W_1;
    if (t.opcode > 0xFF) e.op = {0x0F, static_cast<u8>(t.opcode)};
    else e.op = {static_cast<u8>(t.opcode)};

    std::vector<u8> rm_bytes;
    if (t.mem) {
        MemOp m = gen_mem(g, true);
        e.x = m.x;
        e.b = m.b;
        rm_bytes = m.bytes;
    } else {
        const u8 rm = (t.rm_role == R_DST) ? g.pick_free() : g.pick_any();
        e.b = rm >> 3;
        rm_bytes = {static_cast<u8>(0xC0 | (rm & 7))};
    }

    u8 reg = 0;
    switch (t.reg_role) {
        case R_DIGIT: reg = t.digit; break;
        case R_ANY: reg = g.pick_any(); break;
        case R_DST: reg = g.pick_free(); break;
        case R_DST8: {
            const u8 r = g.pick_free();
            const bool rex_needed = e.w || e.x || e.b;
            if (r >= 4 && r < 8) {
                reg = r;
                e.force_rex = true; // SPL/BPL/SIL/DIL
            } else if (r < 4 && !rex_needed && g.rng.coin()) {
                reg = r + 4;        // AH/CH/DH/BH
                e.no_rex = true;
            } else {
                reg = r;
            }
            break;
        }
    }
    e.r = reg >> 3;
    rm_bytes[0] |= static_cast<u8>((reg & 7) << 3);
    e.tail = rm_bytes;
    push_imm(g, e.tail, t.imm);
    finish(g, e, out, !t.byte_regs);
}

void gen_nop90(Gen&, const Template&, Insn& out) { out.bytes = {0x90}; }

// 0F 1F /0. ext=0: forms without SIB/disp32 (what the JIT sizes today);
// ext=1: SIB and disp32 forms (e.g. the ubiquitous 0F 1F 44 00 00).
void gen_nopl(Gen& g, const Template& t, Insn& out) {
    Enc e;
    e.op = {0x0F, 0x1F};
    if (t.ext == 0) {
        static const u8 rms[] = {0, 1, 2, 3, 6, 7};
        const u8 mod = static_cast<u8>(g.rng.below(3));
        e.tail = {static_cast<u8>((mod << 6) | rms[g.rng.below(6)])};
        if (mod == 1) e.tail.push_back(static_cast<u8>(g.rng.next()));
        if (mod == 2) push32(e.tail, static_cast<u32>(g.rng.next()));
    } else {
        if (g.rng.below(4) == 0) e.prefix = {0x66};
        const u8 sib = static_cast<u8>(g.rng.next());
        switch (g.rng.below(4)) {
            case 0: // mod=00 SIB (disp32 when base=101)
                e.tail = {0x04, sib};
                if ((sib & 7) == 5) push32(e.tail, static_cast<u32>(g.rng.next()));
                break;
            case 1: e.tail = {0x44, sib, static_cast<u8>(g.rng.next())}; break;
            case 2: e.tail = {0x84, sib}; push32(e.tail, static_cast<u32>(g.rng.next())); break;
            default: e.tail = {0x05}; push32(e.tail, static_cast<u32>(g.rng.next())); break;
        }
    }
    finish(g, e, out, true);
}

// B8+r imm32 (ext=0) or REX.W B8+r imm64 (ext=1).
void gen_mov_imm(Gen& g, const Template& t, Insn& out) {
    Enc e;
    const u8 r = g.pick_free();
    e.b = r >> 3;
    e.w = t.ext == 1;
    e.op = {static_cast<u8>(0xB8 + (r & 7))};
    if (e.w) push64(e.tail, g.rng.coin() ? interesting_u64(g.rng) : g.rng.next());
    else push32(e.tail, interesting_imm32(g.rng));
    finish(g, e, out, true);
}

// TEST eAX/rAX, imm32 (A9).
void gen_test_acc(Gen& g, const Template&, Insn& out) {
    Enc e;
    e.w = g.rng.coin();
    e.op = {0xA9};
    push32(e.tail, interesting_imm32(g.rng));
    finish(g, e, out, true);
}

// JMP rel8 (EB) / rel32 (E9) forward over 0..6 filler bytes (HLT/INT3), so the
// sequence stays linear. Filler is never executed unless the jump is wrong.
void gen_jmp(Gen& g, const Template& t, Insn& out) {
    const u8 skip = static_cast<u8>(g.rng.below(7));
    out.bytes = {static_cast<u8>(t.opcode)};
    if (t.opcode == 0xEB) out.bytes.push_back(skip);
    else push32(out.bytes, skip);
    for (u8 i = 0; i < skip; ++i) out.bytes.push_back(g.rng.coin() ? 0xF4 : 0xCC);
}

// Jcc rel8 (70+cc) / rel32 (0F 80+cc) over HLT/INT3 filler, like gen_jmp: a
// wrong condition makes the JIT and the interpreter stop at different RIPs.
// `opcode` holds the condition code; `imm` selects rel32 when 4.
void gen_jcc(Gen& g, const Template& t, Insn& out) {
    const u8 skip = static_cast<u8>(g.rng.below(7));
    const u8 cc = static_cast<u8>(t.opcode & 0x0F);
    if (t.imm == 4) {
        out.bytes = {0x0F, static_cast<u8>(0x80 + cc)};
        push32(out.bytes, skip);
    } else {
        out.bytes = {static_cast<u8>(0x70 + cc), skip};
    }
    for (u8 i = 0; i < skip; ++i) out.bytes.push_back(g.rng.coin() ? 0xF4 : 0xCC);
}

void gen_push_reg(Gen& g, const Template&, Insn& out) {
    Enc e;
    const u8 r = g.pick_any();
    e.b = r >> 3;
    e.op = {static_cast<u8>(0x50 + (r & 7))};
    finish(g, e, out, true);
}

void gen_pop_reg(Gen& g, const Template&, Insn& out) {
    Enc e;
    const u8 r = g.pick_free();
    e.b = r >> 3;
    e.op = {static_cast<u8>(0x58 + (r & 7))};
    finish(g, e, out, true);
}

void gen_push_imm(Gen& g, const Template& t, Insn& out) {
    out.bytes = {static_cast<u8>(t.opcode)};
    push_imm(g, out.bytes, t.opcode == 0x6A ? 1 : 4);
}

struct AluOp {
    const char* name;
    u8 base;  // 00/08/20/28/30/38
    u8 digit; // group-1 /digit
    u64 fset;
    u64 fundef;
};

const AluOp ALU_OPS[] = {
    {"add", 0x00, 0, FS_ARITH, 0},        {"or", 0x08, 1, FS_LOGIC, FU_LOGIC},
    {"and", 0x20, 4, FS_LOGIC, FU_LOGIC}, {"sub", 0x28, 5, FS_ARITH, 0},
    {"xor", 0x30, 6, FS_LOGIC, FU_LOGIC}, {"cmp", 0x38, 7, FS_ARITH, 0},
};

std::vector<Template> build_templates() {
    std::vector<Template> v;
    auto add = [&](Template t) {
        if (!t.gen) t.gen = gen_modrm;
        if (t.mem) t.needs |= NEED_PTR;
        v.push_back(std::move(t));
    };

    // --- What the JIT translates today (compile_block_locked) ---------------
    add({.group = "nop", .name = "nop", .gen = gen_nop90});
    add({.group = "nop", .name = "nopl_modrm", .gen = gen_nopl, .ext = 0});
    add({.group = "nop", .name = "nopl_sib_disp32", .gen = gen_nopl, .ext = 1});

    add({.group = "mov_imm", .name = "mov_r32_imm32", .gen = gen_mov_imm, .ext = 0});
    add({.group = "mov_imm", .name = "mov_r64_imm64", .gen = gen_mov_imm, .ext = 1});
    add({.group = "mov_imm", .name = "mov_rm_imm32(C7)", .opcode = 0xC7, .reg_role = R_DIGIT,
         .rm_role = R_DST, .imm = 4});

    add({.group = "mov_rr", .name = "mov_rm_r(89)", .opcode = 0x89, .reg_role = R_ANY, .rm_role = R_DST});
    add({.group = "mov_rr", .name = "mov_r_rm(8B)", .opcode = 0x8B, .reg_role = R_DST, .rm_role = R_ANY});

    for (const AluOp& op : ALU_OPS) {
        const std::string n = op.name;
        const bool cmp = op.digit == 7;
        const std::string rr_group = cmp ? "cmp_test" : "alu_rr";
        add({.group = rr_group, .name = n + "_rm_r", .opcode = static_cast<u16>(op.base + 1),
             .reg_role = R_ANY, .rm_role = cmp ? R_ANY : R_DST, .fset = op.fset, .fundef = op.fundef});
        add({.group = rr_group, .name = n + "_r_rm", .opcode = static_cast<u16>(op.base + 3),
             .reg_role = cmp ? R_ANY : R_DST, .rm_role = R_ANY, .fset = op.fset, .fundef = op.fundef});
        add({.group = "alu_imm8", .name = n + "_rm_imm8(83)", .opcode = 0x83, .reg_role = R_DIGIT,
             .rm_role = cmp ? R_ANY : R_DST, .digit = op.digit, .imm = 1, .fset = op.fset,
             .fundef = op.fundef});
        add({.group = "alu_imm32", .name = n + "_rm_imm32(81)", .opcode = 0x81, .reg_role = R_DIGIT,
             .rm_role = cmp ? R_ANY : R_DST, .digit = op.digit, .imm = 4, .fset = op.fset,
             .fundef = op.fundef});
        // Memory forms
        const std::string mem_group = cmp ? "cmp_mem" : "alu_mem";
        add({.group = mem_group, .name = n + "_m_r", .opcode = static_cast<u16>(op.base + 1),
             .reg_role = R_ANY, .mem = true, .fset = op.fset, .fundef = op.fundef});
        add({.group = mem_group, .name = n + "_r_m", .opcode = static_cast<u16>(op.base + 3),
             .reg_role = cmp ? R_ANY : R_DST, .mem = true, .fset = op.fset, .fundef = op.fundef});
        add({.group = mem_group, .name = n + "_m_imm8(83)", .opcode = 0x83, .reg_role = R_DIGIT,
             .mem = true, .digit = op.digit, .imm = 1, .fset = op.fset, .fundef = op.fundef});
        add({.group = mem_group, .name = n + "_m_imm32(81)", .opcode = 0x81, .reg_role = R_DIGIT,
             .mem = true, .digit = op.digit, .imm = 4, .fset = op.fset, .fundef = op.fundef});
    }

    add({.group = "cmp_test", .name = "test_rm_r(85)", .opcode = 0x85, .reg_role = R_ANY,
         .rm_role = R_ANY, .fset = FS_LOGIC, .fundef = FU_LOGIC});
    add({.group = "cmp_test", .name = "test_acc_imm32(A9)", .gen = gen_test_acc, .fset = FS_LOGIC,
         .fundef = FU_LOGIC});
    add({.group = "cmp_mem", .name = "test_m_r(85)", .opcode = 0x85, .reg_role = R_ANY, .mem = true,
         .fset = FS_LOGIC, .fundef = FU_LOGIC});

    add({.group = "jmp", .name = "jmp_rel8", .gen = gen_jmp, .opcode = 0xEB});
    add({.group = "jmp", .name = "jmp_rel32", .gen = gen_jmp, .opcode = 0xE9});
    {
        static const char* const CCN[] = {"o", "no", "b", "ae", "e", "ne", "be", "a",
                                          "s", "ns", "p", "np", "l", "ge", "le", "g"};
        for (u8 cc = 0; cc < 16; ++cc) {
            add({.group = "jcc", .name = std::string("j") + CCN[cc] + "_rel8", .gen = gen_jcc,
                 .opcode = cc, .imm = 1});
            add({.group = "jcc", .name = std::string("j") + CCN[cc] + "_rel32", .gen = gen_jcc,
                 .opcode = cc, .imm = 4});
        }
    }

    // CMOVcc r32/r64, r/m and SETcc r/m8 for all 16 conditions. Preceding ALU
    // instructions leave varied flags, so both outcomes of each are exercised.
    {
        static const char* const CCN[] = {"o", "no", "b", "ae", "e", "ne", "be", "a",
                                          "s", "ns", "p", "np", "l", "ge", "le", "g"};
        for (u16 cc = 0; cc < 16; ++cc) {
            const std::string c = CCN[cc];
            add({.group = "cmov", .name = "cmov" + c, .opcode = static_cast<u16>(0x0F40 + cc),
                 .reg_role = R_DST, .rm_role = R_ANY});
            add({.group = "cmov_mem", .name = "cmov" + c + "_mem",
                 .opcode = static_cast<u16>(0x0F40 + cc), .reg_role = R_DST, .mem = true});
            add({.group = "setcc", .name = "set" + c, .opcode = static_cast<u16>(0x0F90 + cc),
                 .reg_role = R_DIGIT, .rm_role = R_DST, .wmode = W_0, .byte_regs = true});
            add({.group = "setcc_mem", .name = "set" + c + "_mem",
                 .opcode = static_cast<u16>(0x0F90 + cc), .reg_role = R_DIGIT, .mem = true,
                 .wmode = W_0});
        }
    }

    // --- Memory-operand forms (ready for the JIT to grow into) --------------
    add({.group = "mov_mem", .name = "mov_m_r(89)", .opcode = 0x89, .reg_role = R_ANY, .mem = true});
    add({.group = "mov_mem", .name = "mov_r_m(8B)", .opcode = 0x8B, .reg_role = R_DST, .mem = true});
    add({.group = "mov_mem", .name = "mov_m8_r8(88)", .opcode = 0x88, .reg_role = R_ANY, .mem = true,
         .wmode = W_0, .byte_regs = true});
    add({.group = "mov_mem", .name = "mov_r8_m8(8A)", .opcode = 0x8A, .reg_role = R_DST8, .mem = true,
         .wmode = W_0, .byte_regs = true});
    add({.group = "mov_mem", .name = "mov_m_imm32(C7)", .opcode = 0xC7, .reg_role = R_DIGIT, .mem = true,
         .imm = 4});
    add({.group = "mov_mem", .name = "mov_m8_imm8(C6)", .opcode = 0xC6, .reg_role = R_DIGIT, .mem = true,
         .imm = 1, .wmode = W_0});

    add({.group = "lea", .name = "lea(8D)", .opcode = 0x8D, .reg_role = R_DST, .mem = true});

    struct Movx {
        const char* name;
        u16 opcode;
        WMode w;
    };
    const Movx movx[] = {{"movzx_r_rm8", 0x0FB6, W_RAND}, {"movzx_r_rm16", 0x0FB7, W_RAND},
                         {"movsx_r_rm8", 0x0FBE, W_RAND}, {"movsx_r_rm16", 0x0FBF, W_RAND},
                         {"movsxd_r64_rm32", 0x63, W_1}};
    for (const Movx& m : movx) {
        add({.group = "movx", .name = m.name, .opcode = m.opcode, .reg_role = R_DST, .rm_role = R_ANY,
             .wmode = m.w});
        add({.group = "movx_mem", .name = std::string(m.name) + "_mem", .opcode = m.opcode,
             .reg_role = R_DST, .mem = true, .wmode = m.w});
    }

    add({.group = "stack", .name = "push_r", .gen = gen_push_reg, .needs = NEED_STACK});
    add({.group = "stack", .name = "pop_r", .gen = gen_pop_reg, .needs = NEED_STACK});
    add({.group = "stack", .name = "push_imm8(6A)", .gen = gen_push_imm, .opcode = 0x6A, .needs = NEED_STACK});
    add({.group = "stack", .name = "push_imm32(68)", .gen = gen_push_imm, .opcode = 0x68, .needs = NEED_STACK});
    add({.group = "stack", .name = "push_m(FF/6)", .opcode = 0xFF, .reg_role = R_DIGIT, .mem = true,
         .digit = 6, .wmode = W_0, .needs = NEED_STACK});
    return v;
}

// -----------------------------------------------------------------------------
// Trials
// -----------------------------------------------------------------------------
struct Trial {
    std::vector<Insn> seq;
    u64 regs[16]{};
    u64 rflags{0x202};
    u64 mem_seed{0};
};

struct Outcome {
    int status{0}; // StepResult, or STATUS_STEP_LIMIT
    CpuContext ctx{};
    std::vector<u8> scratch;
    std::vector<u8> stack;
    std::vector<bool> jit_covered; // per instruction: executed by JIT code
    usize jit_blocks{0};
};

struct Machine {
    MemoryManager memory;
    SyscallTable syscalls;
    CpuExecutor exec;
    JitCompiler jit;

    Machine() : syscalls(memory), exec(memory, syscalls) {}
};

std::vector<GuestAddr> insn_addrs(const Trial& t) {
    std::vector<GuestAddr> a;
    GuestAddr p = CODE_BASE;
    for (const Insn& i : t.seq) {
        a.push_back(p);
        p += i.bytes.size();
    }
    return a;
}

void load_trial(Machine& m, const Trial& t) {
    std::vector<u8> code;
    for (const Insn& i : t.seq) code.insert(code.end(), i.bytes.begin(), i.bytes.end());
    code.insert(code.end(), 64, 0xF4); // HLT terminator (+ HLT sled for mis-sized decodes)
    assert(m.memory.write_bytes(CODE_BASE, code.data(), code.size()));

    Rng r(t.mem_seed);
    std::vector<u8> fill(SCRATCH_SIZE);
    for (usize i = 0; i < fill.size(); i += 8) {
        const u64 v = r.next();
        std::memcpy(fill.data() + i, &v, 8);
    }
    assert(m.memory.write_bytes(SCRATCH_BASE, fill.data(), fill.size()));
    fill.assign(STACK_SIZE, 0);
    for (usize i = 0; i < fill.size(); i += 8) {
        const u64 v = r.next();
        std::memcpy(fill.data() + i, &v, 8);
    }
    assert(m.memory.write_bytes(STACK_BASE, fill.data(), fill.size()));
}

Outcome run_trial(Machine& m, const Trial& t, bool use_jit) {
    load_trial(m, t);
    CpuContext& ctx = m.exec.context();
    ctx.reset();
    for (int i = 0; i < 16; ++i) ctx.*GPR_FIELDS[i] = t.regs[i];
    ctx.rflags = t.rflags;
    ctx.rip = CODE_BASE;
    m.jit.clear_cache();

    Outcome o;
    o.jit_covered.assign(t.seq.size(), false);
    const std::vector<GuestAddr> addrs = insn_addrs(t);
    o.status = STATUS_STEP_LIMIT;
    for (int steps = 0; steps < MAX_STEPS; ++steps) {
        // Same policy as CpuExecutor::run: JIT block first, interpreter step
        // when the JIT declines (returns false).
        if (use_jit) {
            const GuestAddr rip = ctx.rip;
            if (m.jit.execute_block(ctx, m.memory)) {
                o.jit_blocks++;
                const JitBlockMetrics* met = m.jit.get_block_metrics(rip);
                const usize k = met ? met->instructions_compiled : 0;
                const auto it = std::find(addrs.begin(), addrs.end(), rip);
                if (it != addrs.end()) {
                    for (usize j = static_cast<usize>(it - addrs.begin()); j < addrs.size() && k > 0; ++j) {
                        if (j - static_cast<usize>(it - addrs.begin()) >= k) break;
                        o.jit_covered[j] = true;
                    }
                }
                continue;
            }
        }
        const StepResult r = m.exec.step();
        if (r != StepResult::Ok) {
            o.status = static_cast<int>(r);
            break;
        }
    }
    o.ctx = ctx;
    o.scratch.resize(SCRATCH_SIZE);
    o.stack.resize(STACK_SIZE);
    assert(m.memory.read_bytes(SCRATCH_BASE, o.scratch.data(), SCRATCH_SIZE));
    assert(m.memory.read_bytes(STACK_BASE, o.stack.data(), STACK_SIZE));
    return o;
}

struct Config {
    bool check_flags{true};
    bool check_af{false};
    bool verbose{false};
};

// Mutes fd 2 for its lifetime (interpreter diagnostics on garbage paths).
struct StderrMute {
    int saved{-1};
    explicit StderrMute(bool active) {
        if (!active) return;
        std::fflush(stderr);
        saved = ::dup(2);
        const int nul = ::open("/dev/null", O_WRONLY);
        if (nul >= 0) {
            ::dup2(nul, 2);
            ::close(nul);
        }
    }
    ~StderrMute() {
        if (saved < 0) return;
        std::fflush(stderr);
        ::dup2(saved, 2);
        ::close(saved);
    }
};

// Arithmetic flags whose final value is architecturally defined by the
// sequence (initial flags count as defined: both runs start identical).
u64 defined_flags(const Trial& t, const std::vector<Template>& tmpls, const Config& cfg) {
    u64 def = ARITH_FLAGS;
    for (const Insn& i : t.seq) {
        const Template& tp = tmpls[i.tmpl];
        def = (def | tp.fset) & ~tp.fundef;
    }
    if (!cfg.check_af) def &= ~F_AF;
    if (!cfg.check_flags) def = 0;
    return def;
}

std::string status_name(int s) {
    if (s == STATUS_STEP_LIMIT) return "step-limit";
    return std::string(step_result_to_string(static_cast<StepResult>(s)));
}

std::vector<std::string> compare(const Outcome& ref, const Outcome& jit, u64 flag_mask) {
    std::vector<std::string> d;
    if (ref.status != jit.status) {
        d.push_back("status interp=" + status_name(ref.status) + " jit=" + status_name(jit.status));
    }
    for (int i = 0; i < 16; ++i) {
        const u64 a = ref.ctx.*GPR_FIELDS[i];
        const u64 b = jit.ctx.*GPR_FIELDS[i];
        if (a != b) d.push_back(std::string(REG_NAMES[i]) + "  interp=" + hex64(a) + " jit=" + hex64(b));
    }
    if (ref.ctx.rip != jit.ctx.rip) {
        d.push_back("rip  interp=" + hex64(ref.ctx.rip) + " jit=" + hex64(jit.ctx.rip));
    }
    const u64 cmp_mask = ~ARITH_FLAGS | flag_mask;
    const u64 fa = ref.ctx.rflags & cmp_mask;
    const u64 fb = jit.ctx.rflags & cmp_mask;
    if (fa != fb) {
        d.push_back("rflags interp=" + hex64(ref.ctx.rflags) + " jit=" + hex64(jit.ctx.rflags) +
                    " differ[" + flag_names(fa ^ fb) + "] checked[" + flag_names(flag_mask) + "]");
    }
    auto mem_diff = [&](const std::vector<u8>& a, const std::vector<u8>& b, GuestAddr base, const char* name) {
        usize count = 0;
        for (usize i = 0; i < a.size(); ++i) {
            if (a[i] == b[i]) continue;
            if (count < 8) {
                char buf[96];
                std::snprintf(buf, sizeof(buf), "%s[0x%llx] interp=%02x jit=%02x", name,
                              static_cast<unsigned long long>(base + i), a[i], b[i]);
                d.push_back(buf);
            }
            count++;
        }
        if (count > 8) d.push_back(std::string(name) + ": " + std::to_string(count - 8) + " more differing bytes");
    };
    mem_diff(ref.scratch, jit.scratch, SCRATCH_BASE, "scratch");
    mem_diff(ref.stack, jit.stack, STACK_BASE, "stack");
    return d;
}

enum class Verdict { Pass, Fail, RefUnsupported };

struct Check {
    Verdict verdict{Verdict::Pass};
    Outcome ref, jit;
    std::vector<std::string> diff;
};

Check check_trial(Machine& m, const Trial& t, const std::vector<Template>& tmpls, const Config& cfg) {
    Check c;
    {
        StderrMute mute(!cfg.verbose);
        c.ref = run_trial(m, t, false);
        c.jit = run_trial(m, t, true);
    }
    if (c.ref.status != static_cast<int>(StepResult::Halted) || c.ref.ctx.rip != CODE_BASE + [&] {
            usize n = 0;
            for (const Insn& i : t.seq) n += i.bytes.size();
            return n;
        }()) {
        c.verdict = Verdict::RefUnsupported;
        return c;
    }
    c.diff = compare(c.ref, c.jit, defined_flags(t, tmpls, cfg));
    c.verdict = c.diff.empty() ? Verdict::Pass : Verdict::Fail;
    return c;
}

// Delta debugging: drop chunks of instructions while the trial still fails.
Trial shrink(Machine& m, Trial t, const std::vector<Template>& tmpls, const Config& cfg) {
    usize chunk = std::max<usize>(1, t.seq.size() / 2);
    while (true) {
        bool removed = false;
        for (usize i = 0; i + chunk <= t.seq.size() && t.seq.size() > 1;) {
            Trial c = t;
            c.seq.erase(c.seq.begin() + static_cast<long>(i), c.seq.begin() + static_cast<long>(i + chunk));
            if (!c.seq.empty() && check_trial(m, c, tmpls, cfg).verdict == Verdict::Fail) {
                t = std::move(c);
                removed = true;
            } else {
                i += chunk;
            }
        }
        if (!removed) {
            if (chunk == 1) break;
            chunk /= 2;
        }
    }
    return t;
}

Trial make_trial(Rng& rng, const std::vector<Template>& tmpls,
                 const std::vector<std::vector<u16>>& groups, usize len_min, usize len_max) {
    Trial t;
    const usize n = len_min + rng.below(len_max - len_min + 1);
    std::vector<u16> picks;
    u32 needs = 0;
    for (usize i = 0; i < n; ++i) {
        const auto& g = groups[rng.below(groups.size())];
        const u16 ti = g[rng.below(g.size())];
        picks.push_back(ti);
        needs |= tmpls[ti].needs;
    }

    for (int i = 0; i < 16; ++i) t.regs[i] = interesting_u64(rng);
    t.rflags = 0x202 | (rng.next() & ARITH_FLAGS);
    t.mem_seed = rng.next();

    Gen g{rng, {}, {}, -1, CODE_BASE};
    std::vector<bool> reserved(16, false);
    if (needs & (NEED_PTR | NEED_STACK)) {
        reserved[4] = true;
        t.regs[4] = STACK_BASE + 0x1000 - 8 * rng.below(16);
    }
    if (needs & NEED_PTR) {
        std::vector<u8> pool;
        for (u8 r = 0; r < 16; ++r) {
            if (r != 4) pool.push_back(r);
        }
        for (usize i = 0; i < pool.size(); ++i) std::swap(pool[i], pool[i + rng.below(pool.size() - i)]);
        const usize nbase = 2 + rng.below(3);
        for (usize i = 0; i < nbase; ++i) {
            const u8 r = pool[i];
            reserved[r] = true;
            g.base_regs.push_back(r);
            t.regs[r] = SCRATCH_BASE + 0x1000 + rng.below(0x2000);
            if (rng.below(4) != 0) t.regs[r] &= ~7ULL;
        }
        g.base_regs.push_back(4); // [rsp + ...] addresses the stack region
        g.index_reg = pool[nbase];
        reserved[pool[nbase]] = true;
        t.regs[pool[nbase]] = rng.below(0x80);
    }
    for (u8 r = 0; r < 16; ++r) {
        if (!reserved[r]) g.free_regs.push_back(r);
    }

    GuestAddr addr = CODE_BASE;
    for (const u16 ti : picks) {
        Insn insn;
        insn.tmpl = ti;
        g.insn_addr = addr;
        tmpls[ti].gen(g, tmpls[ti], insn);
        assert(!insn.bytes.empty() && insn.bytes.size() <= 15 + 6);
        addr += insn.bytes.size();
        t.seq.push_back(std::move(insn));
    }
    return t;
}

void print_sequence(const Trial& t, const std::vector<Template>& tmpls, const std::vector<bool>* jit_cov) {
    const std::vector<GuestAddr> addrs = insn_addrs(t);
    std::vector<u8> all;
    for (usize i = 0; i < t.seq.size(); ++i) {
        const Insn& in = t.seq[i];
        const Template& tp = tmpls[in.tmpl];
        std::printf("    [%zu] 0x%llx  %-28s %-14s %s%s\n", i, static_cast<unsigned long long>(addrs[i]),
                    hex_bytes(in.bytes).c_str(), tp.group.c_str(), tp.name.c_str(),
                    jit_cov ? ((*jit_cov)[i] ? "  (jit)" : "  (interp)") : "");
        all.insert(all.end(), in.bytes.begin(), in.bytes.end());
    }
    all.push_back(0xF4);
    std::printf("    bytes: %s\n", hex_bytes(all).c_str());
}

void print_state(const Trial& t) {
    std::printf("    initial:");
    for (int i = 0; i < 16; ++i) {
        if (i % 4 == 0) std::printf("\n     ");
        std::printf(" %-3s=%s", REG_NAMES[i], hex64(t.regs[i]).c_str());
    }
    std::printf("\n      rflags=%s mem_seed=%s\n", hex64(t.rflags).c_str(), hex64(t.mem_seed).c_str());
}

void print_diff(const std::vector<std::string>& d) {
    for (const auto& line : d) std::printf("    ! %s\n", line.c_str());
}

std::string signature(const Trial& t, const std::vector<Template>& tmpls) {
    std::string s;
    for (const Insn& i : t.seq) {
        if (!s.empty()) s += " ; ";
        s += tmpls[i.tmpl].group + "/" + tmpls[i.tmpl].name;
    }
    return s;
}

struct Stats {
    usize insns{0}, jit{0}, interp{0}, trials{0}, fails{0};
};

} // namespace

int main(int argc, char** argv) {
    const std::vector<Template> tmpls = build_templates();

    std::vector<std::string> group_order;
    std::map<std::string, std::vector<u16>> by_group;
    for (usize i = 0; i < tmpls.size(); ++i) {
        if (!by_group.count(tmpls[i].group)) group_order.push_back(tmpls[i].group);
        by_group[tmpls[i].group].push_back(static_cast<u16>(i));
    }

    bool list = env_u64("JIT_DIFF_LIST", 0) != 0;
    std::string group_filter = env_str("JIT_DIFF_GROUPS");
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list") list = true;
        else group_filter = a;
    }
    if (list) {
        for (const auto& gname : group_order) {
            std::printf("%s:", gname.c_str());
            for (u16 ti : by_group[gname]) std::printf(" %s", tmpls[ti].name.c_str());
            std::printf("\n");
        }
        return 0;
    }

    std::vector<std::string> enabled = group_filter.empty() ? group_order : split(group_filter, ',');
    std::vector<std::vector<u16>> groups;
    for (const auto& gname : enabled) {
        if (!by_group.count(gname)) {
            std::fprintf(stderr, "unknown group '%s'; groups:", gname.c_str());
            for (const auto& g : group_order) std::fprintf(stderr, " %s", g.c_str());
            std::fprintf(stderr, "\n");
            return 2;
        }
        groups.push_back(by_group[gname]);
    }

    const u64 seed = env_u64("JIT_DIFF_SEED", 1);
    const u64 trials = env_u64("JIT_DIFF_TRIALS", 2000);
    const bool single = std::getenv("JIT_DIFF_TRIAL") != nullptr;
    const u64 only_trial = env_u64("JIT_DIFF_TRIAL", 0);
    const u64 max_reports = env_u64("JIT_DIFF_MAX_REPORTS", 3);
    const bool stop_on_fail = env_u64("JIT_DIFF_STOP_ON_FAIL", 0) != 0;
    const bool verbose = env_u64("JIT_DIFF_VERBOSE", 0) != 0;
    Config cfg;
    cfg.check_flags = env_u64("JIT_DIFF_CHECK_FLAGS", 1) != 0;
    cfg.check_af = env_u64("JIT_DIFF_CHECK_AF", 0) != 0;
    cfg.verbose = verbose;

    usize len_min = 1, len_max = 8;
    {
        const std::string l = env_str("JIT_DIFF_LEN");
        if (!l.empty()) {
            const usize dots = l.find("..");
            if (dots != std::string::npos) {
                len_min = std::strtoull(l.substr(0, dots).c_str(), nullptr, 0);
                len_max = std::strtoull(l.substr(dots + 2).c_str(), nullptr, 0);
            } else {
                len_min = len_max = std::strtoull(l.c_str(), nullptr, 0);
            }
            len_min = std::max<usize>(1, len_min);
            len_max = std::max(len_min, len_max);
        }
    }

    std::printf("=== JIT differential fuzzer (interpreter = reference) ===\n");
    std::printf("seed=%llu trials=%llu len=%zu..%zu groups=%s check_flags=%d check_af=%d\n",
                static_cast<unsigned long long>(seed), static_cast<unsigned long long>(trials), len_min, len_max,
                group_filter.empty() ? "<all>" : group_filter.c_str(), cfg.check_flags, cfg.check_af);

    Machine m;
    assert(m.memory.initialize());

    std::map<std::string, Stats> gstats;
    std::map<u16, Stats> tstats;
    struct FailInfo {
        usize count{0};
        u64 first_trial{0};
        std::string bytes;
    };
    std::map<std::string, FailInfo> unique_fails;
    usize passed = 0, failed = 0, skipped = 0, reported = 0;
    usize full_jit = 0, partial_jit = 0, no_jit = 0;
    std::vector<std::string> skip_samples;

    const u64 first = single ? only_trial : 0;
    const u64 last = single ? only_trial + 1 : trials;
    for (u64 ti = first; ti < last; ++ti) {
        Rng rng(seed * 0x9E3779B97F4A7C15ULL ^ (ti + 1) * 0xD1B54A32D192ED03ULL);
        rng.next();
        const Trial t = make_trial(rng, tmpls, groups, len_min, len_max);
        const Check c = check_trial(m, t, tmpls, cfg);

        if (c.verdict == Verdict::RefUnsupported) {
            skipped++;
            if (skip_samples.size() < 5) {
                skip_samples.push_back("trial " + std::to_string(ti) + " status=" + status_name(c.ref.status) +
                                       " rip=" + hex64(c.ref.ctx.rip) + " : " + signature(t, tmpls));
            }
            continue;
        }

        usize njit = 0;
        for (usize i = 0; i < t.seq.size(); ++i) {
            const Template& tp = tmpls[t.seq[i].tmpl];
            Stats& gs = gstats[tp.group];
            Stats& ts = tstats[t.seq[i].tmpl];
            gs.insns++;
            ts.insns++;
            if (c.jit.jit_covered[i]) {
                gs.jit++;
                ts.jit++;
                njit++;
            } else {
                gs.interp++;
                ts.interp++;
            }
        }
        if (njit == t.seq.size()) full_jit++;
        else if (njit == 0) no_jit++;
        else partial_jit++;

        std::map<std::string, bool> in_trial;
        for (const Insn& i : t.seq) in_trial[tmpls[i.tmpl].group] = true;
        for (const auto& [gname, _] : in_trial) gstats[gname].trials++;

        if (c.verdict == Verdict::Pass) {
            passed++;
            continue;
        }

        failed++;
        const Trial small = shrink(m, t, tmpls, cfg);
        const Check sc = check_trial(m, small, tmpls, cfg);
        std::map<std::string, bool> in_small;
        for (const Insn& i : small.seq) in_small[tmpls[i.tmpl].group] = true;
        for (const auto& [gname, _] : in_small) gstats[gname].fails++;

        const std::string sig = signature(small, tmpls);
        FailInfo& fi = unique_fails[sig];
        if (fi.count == 0) {
            fi.first_trial = ti;
            std::vector<u8> all;
            for (const Insn& i : small.seq) all.insert(all.end(), i.bytes.begin(), i.bytes.end());
            fi.bytes = hex_bytes(all);
            if (reported < max_reports) {
                reported++;
                std::printf("\nMISMATCH trial %llu  (reproduce: JIT_DIFF_SEED=%llu JIT_DIFF_TRIAL=%llu)\n",
                            static_cast<unsigned long long>(ti), static_cast<unsigned long long>(seed),
                            static_cast<unsigned long long>(ti));
                std::printf("  original sequence (%zu insns):\n", t.seq.size());
                print_sequence(t, tmpls, &c.jit.jit_covered);
                print_state(t);
                print_diff(c.diff);
                std::printf("  minimized to %zu insn(s):\n", small.seq.size());
                print_sequence(small, tmpls, &sc.jit.jit_covered);
                print_diff(sc.diff);
            }
        }
        fi.count++;
        if (stop_on_fail) break;
    }

    std::printf("\n%-10s %8s %8s %8s %7s %7s %6s\n", "group", "insns", "jit", "interp", "jit%", "trials", "fails");
    for (const auto& gname : group_order) {
        if (!gstats.count(gname)) continue;
        const Stats& s = gstats[gname];
        std::printf("%-10s %8zu %8zu %8zu %6.1f%% %7zu %6zu\n", gname.c_str(), s.insns, s.jit, s.interp,
                    s.insns ? 100.0 * static_cast<double>(s.jit) / static_cast<double>(s.insns) : 0.0, s.trials,
                    s.fails);
    }
    if (verbose) {
        std::printf("\n%-32s %8s %8s %8s\n", "template", "insns", "jit", "interp");
        for (const auto& [ti, s] : tstats) {
            const std::string n = tmpls[ti].group + "/" + tmpls[ti].name;
            std::printf("%-32s %8zu %8zu %8zu\n", n.c_str(), s.insns, s.jit, s.interp);
        }
    }
    if (!skip_samples.empty()) {
        std::printf("\nreference (interpreter) could not run %zu trial(s) -- generator/interpreter gap:\n", skipped);
        for (const auto& s : skip_samples) std::printf("  %s\n", s.c_str());
    }
    if (!unique_fails.empty()) {
        std::printf("\nunique minimized failures (%zu):\n", unique_fails.size());
        for (const auto& [sig, fi] : unique_fails) {
            std::printf("  x%-5zu trial %-6llu [%s]  %s\n", fi.count, static_cast<unsigned long long>(fi.first_trial),
                        fi.bytes.c_str(), sig.c_str());
        }
    }
    std::printf("\nblocks: fully-jitted=%zu partially-jitted=%zu fell-back=%zu\n", full_jit, partial_jit, no_jit);
    std::printf("jit_diff: seed=%llu trials=%zu passed=%zu failed=%zu skipped=%zu unique_failures=%zu\n",
                static_cast<unsigned long long>(seed), passed + failed + skipped, passed, failed, skipped,
                unique_fails.size());
    if (failed) {
        std::printf("=== JIT differential fuzzer FAILED ===\n");
        return 1;
    }
    std::printf("=== JIT differential fuzzer passed ===\n");
    return 0;
}
