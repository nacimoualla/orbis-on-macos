// JIT memory-operand tests: every case runs the same guest code through the
// interpreter alone and through one JIT block, then compares all GPRs, RIP,
// RFLAGS and the touched guest memory. Each case runs with many random
// register/memory/flag states (edge values included) so flag results are
// exercised, not just the data path.
//
// Run: scripts/cpu_harness.sh tests/test_jit_mem.cpp
//   env JIT_MEM_TRIALS=<n> (default 150) trials per case

#include "ps4/cpu_context.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace ps4;

namespace {

constexpr GuestAddr CODE = 0x100000;  // code; [CODE+0x800, CODE+0x1800) is RIP-relative data
constexpr GuestAddr DATA = 0x200000;  // data window [DATA-0x1000, DATA+0x3000)
constexpr GuestAddr STACK = 0x300000; // stack window [STACK-0x1000, STACK+0x1000)
constexpr u64 GS_BASE = 0x1000;       // gs:[rbx] lands in the data window

struct Window {
  GuestAddr addr;
  usize size;
};
const Window WINDOWS[] = {
    {CODE + 0x800, 0x1000},
    {DATA - 0x1000, 0x4000},
    {STACK - 0x1000, 0x2000},
};

using Setup = std::function<void(CpuContext &, std::mt19937_64 &)>;

struct Case {
  std::string name;
  std::vector<u8> code;
  usize insns;          // instructions in `code` (all expected to be JIT'd)
  Setup setup{};        // extra per-trial register setup
  int exit_insn{-1};    // >= 0: expect a bounds side exit at this instruction
  u32 exit_off{0};      // byte offset of that instruction
};

int g_failures = 0;
int g_cases = 0;
u64 g_trials = 150;

u64 edge_or_random(std::mt19937_64 &rng) {
  static const u64 edges[] = {0,
                              1,
                              0x7F,
                              0x80,
                              0xFF,
                              0x7FFF,
                              0x8000,
                              0xFFFF,
                              0x7FFFFFFF,
                              0x80000000,
                              0xFFFFFFFF,
                              0x7FFFFFFFFFFFFFFFULL,
                              0x8000000000000000ULL,
                              ~0ULL};
  if (rng() % 4 == 0)
    return edges[rng() % (sizeof(edges) / sizeof(edges[0]))];
  return rng();
}

void random_state(CpuContext &c, std::mt19937_64 &rng) {
  c = CpuContext{};
  // data registers
  c.rax = edge_or_random(rng);
  c.rdx = edge_or_random(rng);
  c.rdi = edge_or_random(rng);
  c.r8 = edge_or_random(rng);
  c.r10 = edge_or_random(rng);
  c.r11 = edge_or_random(rng);
  c.r15 = edge_or_random(rng);
  // base registers point into the data window
  c.rbx = DATA + (rng() % 0x400);
  c.rbp = DATA + 0x400 + (rng() % 0x400);
  c.rsi = DATA + 0x800 + (rng() % 0x400);
  c.r12 = DATA + (rng() % 0x400);
  c.r13 = DATA + 0x400 + (rng() % 0x400);
  c.r14 = DATA + 0x800 + (rng() % 0x400);
  // index registers: small, sometimes negative
  c.rcx = rng() % 0x40;
  c.r9 = rng() % 0x40;
  if (rng() % 3 == 0)
    c.rcx = 0 - c.rcx;
  c.rsp = STACK - 8 * (rng() % 16);
  c.rip = CODE;
  c.fs_base = DATA;
  c.gs_base = GS_BASE;
  // random arithmetic flags (incl. AF, which must be preserved)
  constexpr u64 arith = CpuContext::FLAG_CF | CpuContext::FLAG_PF |
                        CpuContext::FLAG_AF | CpuContext::FLAG_ZF |
                        CpuContext::FLAG_SF | CpuContext::FLAG_OF;
  c.rflags = 0x202 | (rng() & arith);
}

std::vector<u8> snapshot(const MemoryManager &mem) {
  std::vector<u8> out;
  for (const Window &w : WINDOWS) {
    const usize at = out.size();
    out.resize(at + w.size);
    mem.read_bytes(w.addr, out.data() + at, w.size);
  }
  return out;
}

void restore(MemoryManager &mem, const std::vector<u8> &snap) {
  usize at = 0;
  for (const Window &w : WINDOWS) {
    mem.write_bytes(w.addr, snap.data() + at, w.size);
    at += w.size;
  }
}

const char *REG_NAMES[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp",
                             "rsi", "rdi", "r8",  "r9",  "r10", "r11",
                             "r12", "r13", "r14", "r15"};

u64 gpr(const CpuContext &c, int i) {
  const u64 *regs[16] = {&c.rax, &c.rcx, &c.rdx, &c.rbx, &c.rsp, &c.rbp,
                         &c.rsi, &c.rdi, &c.r8,  &c.r9,  &c.r10, &c.r11,
                         &c.r12, &c.r13, &c.r14, &c.r15};
  return *regs[i];
}

// Interpreter-only run from ctx.rip until `end` (bounded). Returns Ok when
// `end` was reached, otherwise the step result that stopped it (e.g. a
// MemoryFault for an out-of-bounds PUSH).
StepResult interpret_until(MemoryManager &mem, CpuContext &ctx,
                           GuestAddr end) {
  SyscallTable sys(mem);
  CpuExecutor ex(mem, sys);
  ex.context() = ctx;
  StepResult r = StepResult::Ok;
  for (int i = 0; i < 64 && ex.context().rip != end; ++i) {
    r = ex.step();
    if (r != StepResult::Ok)
      break;
  }
  ctx = ex.context();
  if (r == StepResult::Ok && ctx.rip != end)
    r = StepResult::IllegalInstruction;
  return r;
}

void run_case(MemoryManager &mem, const Case &tc) {
  ++g_cases;
  std::vector<u8> code = tc.code;
  code.push_back(0xF4); // hlt: ends the block
  mem.write_bytes(CODE, code.data(), code.size());
  const GuestAddr end = CODE + tc.code.size();

  JitCompiler jit;
  std::mt19937_64 rng(std::hash<std::string>{}(tc.name));
  for (u64 t = 0; t < g_trials; ++t) {
    CpuContext init;
    random_state(init, rng);
    if (tc.setup)
      tc.setup(init, rng);
    for (const Window &w : WINDOWS) {
      std::vector<u8> bytes(w.size);
      for (auto &b : bytes)
        b = static_cast<u8>(rng());
      mem.write_bytes(w.addr, bytes.data(), bytes.size());
    }
    const std::vector<u8> mem0 = snapshot(mem);

    // Reference: interpreter
    CpuContext ref = init;
    const StepResult ref_status = interpret_until(mem, ref, end);
    // Only side-exit cases may legitimately fault in the interpreter.
    if (ref_status != StepResult::Ok && tc.exit_insn < 0) {
      std::printf("FAIL %s trial %llu: interpreter did not reach the end\n",
                  tc.name.c_str(), (unsigned long long)t);
      ++g_failures;
      return;
    }
    const std::vector<u8> mem_ref = snapshot(mem);

    // JIT
    restore(mem, mem0);
    CpuContext got = init;
    usize retired = 999;
    const bool ok = jit.execute_block(got, mem, &retired);
    std::string err;
    if (tc.exit_insn < 0) {
      if (!ok || retired != tc.insns || got.rip != end)
        err = "block did not run to the end (ok=" + std::to_string(ok) +
              " retired=" + std::to_string(retired) + ")";
    } else {
      if (retired != static_cast<usize>(tc.exit_insn) ||
          got.rip != CODE + tc.exit_off || ok != (tc.exit_insn > 0))
        err = "expected side exit at insn " + std::to_string(tc.exit_insn) +
              " (ok=" + std::to_string(ok) +
              " retired=" + std::to_string(retired) + ")";
      else if (interpret_until(mem, got, end) != ref_status)
        err = "interpreter ended differently after the side exit";
    }
    if (err.empty()) {
      for (int r = 0; r < 16; ++r) {
        if (gpr(got, r) != gpr(ref, r)) {
          char buf[128];
          std::snprintf(buf, sizeof(buf), "%s: jit=0x%llx interp=0x%llx",
                        REG_NAMES[r], (unsigned long long)gpr(got, r),
                        (unsigned long long)gpr(ref, r));
          err = buf;
          break;
        }
      }
    }
    if (err.empty() && got.rflags != ref.rflags) {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "rflags: jit=0x%llx interp=0x%llx",
                    (unsigned long long)got.rflags,
                    (unsigned long long)ref.rflags);
      err = buf;
    }
    if (err.empty() && got.rip != ref.rip)
      err = "rip differs";
    if (err.empty() && snapshot(mem) != mem_ref)
      err = "guest memory differs";
    if (!err.empty()) {
      std::printf("FAIL %s trial %llu: %s\n", tc.name.c_str(),
                  (unsigned long long)t, err.c_str());
      ++g_failures;
      return;
    }
  }
  std::printf("ok   %s\n", tc.name.c_str());
}

std::vector<u8> le32(u32 v) {
  return {static_cast<u8>(v), static_cast<u8>(v >> 8), static_cast<u8>(v >> 16),
          static_cast<u8>(v >> 24)};
}

std::vector<u8> cat(std::initializer_list<std::vector<u8>> parts) {
  std::vector<u8> out;
  for (const auto &p : parts)
    out.insert(out.end(), p.begin(), p.end());
  return out;
}

// Integration: CpuExecutor::run with the JIT attached must make progress past
// a side exit (no livelock) and end in the same state as the interpreter.
void test_executor_side_exit_progress(MemoryManager &mem) {
  ++g_cases;
  const std::vector<u8> prog = {
      0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
      0x01, 0x03,                   // add [rbx], eax   (rbx out of bounds)
      0x48, 0x8B, 0x0E,             // mov rcx, [rsi]
      0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1 (SYS_exit)
      0x89, 0xCF,                   // mov edi, ecx
      0x0F, 0x05,                   // syscall
  };
  auto run = [&](bool with_jit, CpuContext &out) {
    mem.write_bytes(CODE, prog.data(), prog.size());
    const u64 v = 0x2A;
    mem.write_value(DATA, v);
    SyscallTable sys(mem);
    sys.register_default_handlers();
    CpuExecutor ex(mem, sys);
    JitCompiler jit;
    if (with_jit)
      ex.set_jit_compiler(&jit);
    ex.context().rip = CODE;
    ex.context().rbx = mem.total_size() + 0x100;
    ex.context().rsi = DATA;
    const RunResult r = ex.run(100);
    out = ex.context();
    return r;
  };
  CpuContext a, b;
  const RunResult ra = run(false, a);
  const RunResult rb = run(true, b);
  bool pass = ra.status == StepResult::Exited &&
              rb.status == StepResult::Exited && ra.exit_code == 0x2A &&
              rb.exit_code == 0x2A &&
              ra.instructions_executed == rb.instructions_executed;
  for (int r = 0; r < 16; ++r)
    pass = pass && gpr(a, r) == gpr(b, r);
  if (!pass) {
    std::printf("FAIL executor_side_exit_progress (status %d/%d exit %d/%d "
                "insns %zu/%zu)\n",
                (int)ra.status, (int)rb.status, ra.exit_code, rb.exit_code,
                ra.instructions_executed, rb.instructions_executed);
    ++g_failures;
  } else {
    std::printf("ok   executor_side_exit_progress\n");
  }
}

} // namespace

int main() {
  if (const char *t = std::getenv("JIT_MEM_TRIALS"))
    g_trials = std::strtoull(t, nullptr, 0);

  std::printf("=== JIT memory-operand tests (interpreter = reference) ===\n");
  MemoryManager mem;
  if (!mem.initialize()) {
    std::printf("cannot map guest memory\n");
    return 1;
  }
  const u64 SIZE = mem.total_size();

  const Setup oob_rbx = [SIZE](CpuContext &c, std::mt19937_64 &) {
    c.rbx = SIZE + 0x100;
  };

  const std::vector<Case> cases = {
      // ---- MOV, addressing modes -------------------------------------------
      {"mov [rbx], eax", {0x89, 0x03}, 1},
      {"mov [rbx+rcx*1+0x10], rax (SIB scale 1, disp8)",
       {0x48, 0x89, 0x44, 0x0B, 0x10}, 1},
      {"mov rdx, [rsi+rcx*2-8] (scale 2, negative disp8)",
       {0x48, 0x8B, 0x54, 0x4E, 0xF8}, 1},
      {"mov edx, [rbx+rcx*4+0x100] (scale 4, disp32)",
       {0x8B, 0x94, 0x8B, 0x00, 0x01, 0x00, 0x00}, 1},
      {"mov rdi, [rsi+rcx*8-0x40] (scale 8)", {0x48, 0x8B, 0x7C, 0xCE, 0xC0}, 1},
      {"mov eax, [rcx*4+DATA] (SIB no base)",
       cat({{0x8B, 0x04, 0x8D}, le32(DATA)}), 1},
      {"mov rax, [DATA+0x40] (SIB no base, no index)",
       cat({{0x48, 0x8B, 0x04, 0x25}, le32(DATA + 0x40)}), 1},
      {"mov eax, [rcx*2+disp32] REX.B (no base, not r13)",
       cat({{0x41, 0x8B, 0x04, 0x4D}, le32(DATA + 0x10)}), 1},
      {"mov eax, [rbp-0x10]", {0x8B, 0x45, 0xF0}, 1},
      {"mov rax, [rsp+8]", {0x48, 0x8B, 0x44, 0x24, 0x08}, 1},
      {"mov eax, [rbx+r12*1] (REX.X index r12)", {0x42, 0x8B, 0x04, 0x23}, 1},
      {"mov r15, [r12+r9*8+8] (REX.WRXB)", {0x4F, 0x8B, 0x7C, 0xCC, 0x08}, 1},
      {"mov [r13+0], r8d", {0x45, 0x89, 0x45, 0x00}, 1},
      {"mov r10b, [r14+rcx]", {0x45, 0x8A, 0x14, 0x0E}, 1},
      {"mov rax, [rip+0x800]", {0x48, 0x8B, 0x05, 0x00, 0x08, 0x00, 0x00}, 1},
      {"mov [rip+0x900], edx", {0x89, 0x15, 0x00, 0x09, 0x00, 0x00}, 1},
      {"mov dword [rip+0x880], imm32 (RIP-rel + imm)",
       {0xC7, 0x05, 0x80, 0x08, 0x00, 0x00, 0x78, 0x56, 0x34, 0x12}, 1},
      {"mov rax, fs:[0x10]",
       {0x64, 0x48, 0x8B, 0x04, 0x25, 0x10, 0x00, 0x00, 0x00}, 1},
      {"mov gs:[rbx], eax", {0x65, 0x89, 0x03}, 1},
      // ---- MOV, partial widths ---------------------------------------------
      {"mov ax, [rbx] (16-bit merge)", {0x66, 0x8B, 0x03}, 1},
      {"mov [rbx+1], dx", {0x66, 0x89, 0x53, 0x01}, 1},
      {"mov al, [rsi+3]", {0x8A, 0x46, 0x03}, 1},
      {"mov ah, [rsi]", {0x8A, 0x26}, 1},
      {"mov [rbx], dh", {0x88, 0x33}, 1},
      {"mov sil, [rbx] (REX byte reg)", {0x40, 0x8A, 0x33}, 1},
      {"mov ah, bl ; mov dil, al", {0x88, 0xDC, 0x40, 0x88, 0xC7}, 2},
      {"mov byte [rbx+rcx*2+5], 0xAB", {0xC6, 0x44, 0x4B, 0x05, 0xAB}, 1},
      {"mov qword [rsi], -5", {0x48, 0xC7, 0x06, 0xFB, 0xFF, 0xFF, 0xFF}, 1},
      {"mov word [rbx], 0x1234", {0x66, 0xC7, 0x03, 0x34, 0x12}, 1},
      {"mov r9d, 0x80000000 ; mov ax, 0x1234 ; mov bh, 7",
       {0x41, 0xB9, 0x00, 0x00, 0x00, 0x80, 0x66, 0xB8, 0x34, 0x12, 0xB7, 0x07},
       3},
      // ---- LEA ---------------------------------------------------------------
      {"lea rax, [rbx+rcx*8+0x12345]",
       {0x48, 0x8D, 0x84, 0xCB, 0x45, 0x23, 0x01, 0x00}, 1},
      {"lea eax, [rsi+rcx*2-1] (32-bit truncation)",
       {0x8D, 0x44, 0x4E, 0xFF}, 1},
      {"lea ax, [rbx+rdx] (16-bit merge)", {0x66, 0x8D, 0x04, 0x13}, 1},
      {"lea r11, [rip+0x1000]", {0x4C, 0x8D, 0x1D, 0x00, 0x10, 0x00, 0x00}, 1},
      {"lea rdx, [rcx*4+0]", {0x48, 0x8D, 0x14, 0x8D, 0x00, 0x00, 0x00, 0x00},
       1},
      {"lea rax, [rax+rdx-0x7fffffff] (large disp)",
       {0x48, 0x8D, 0x84, 0x10, 0x01, 0x00, 0x00, 0x80}, 1},
      // ---- MOVZX / MOVSX / MOVSXD -------------------------------------------
      {"movzx eax, byte [rbx+rcx]", {0x0F, 0xB6, 0x04, 0x0B}, 1},
      {"movzx rdx, word [rsi]", {0x48, 0x0F, 0xB7, 0x16}, 1},
      {"movsx rax, byte [rsi+1]", {0x48, 0x0F, 0xBE, 0x46, 0x01}, 1},
      {"movsx edx, word [rbx]", {0x0F, 0xBF, 0x13}, 1},
      {"movsx ax, byte [rbx] (16-bit dst)", {0x66, 0x0F, 0xBE, 0x03}, 1},
      {"movzx eax, ah", {0x0F, 0xB6, 0xC4}, 1},
      {"movsx r11, r10b", {0x4D, 0x0F, 0xBE, 0xDA}, 1},
      {"movsxd rax, dword [rbx+4]", {0x48, 0x63, 0x43, 0x04}, 1},
      {"movsxd r8, edx", {0x4C, 0x63, 0xC2}, 1},
      // ---- ALU r/m forms -------------------------------------------------------
      {"add [rbx], eax", {0x01, 0x03}, 1},
      {"add rax, [rsi+rcx*8]", {0x48, 0x03, 0x04, 0xCE}, 1},
      {"sub [rbx+8], rdx", {0x48, 0x29, 0x53, 0x08}, 1},
      {"sub dl, [rsi]", {0x2A, 0x16}, 1},
      {"and [rbx], ax", {0x66, 0x21, 0x03}, 1},
      {"or r8, [r12+8]", {0x4D, 0x0B, 0x44, 0x24, 0x08}, 1},
      {"xor [rsi], dil", {0x40, 0x30, 0x3E}, 1},
      {"cmp [rbx], rdx", {0x48, 0x39, 0x13}, 1},
      {"cmp eax, [rsi]", {0x3B, 0x06}, 1},
      {"cmp ah, [rbx]", {0x3A, 0x23}, 1},
      {"add r12d, eax (32-bit zero-extend)", {0x44, 0x03, 0xE0}, 1},
      {"xor eax, eax", {0x31, 0xC0}, 1},
      {"sub ah, bl", {0x28, 0xDC}, 1},
      {"or cx, dx", {0x66, 0x09, 0xD1}, 1},
      // ---- ALU imm forms -------------------------------------------------------
      {"cmp byte [rbx], 0x80", {0x80, 0x3B, 0x80}, 1},
      {"add dword [rbx+rcx*4], 0x7fffffff",
       {0x81, 0x04, 0x8B, 0xFF, 0xFF, 0xFF, 0x7F}, 1},
      {"sub qword [rsi], -1", {0x48, 0x83, 0x2E, 0xFF}, 1},
      {"and word [rbx], 0xF0F0", {0x66, 0x81, 0x23, 0xF0, 0xF0}, 1},
      {"or byte [rsi+2], 0x81", {0x80, 0x4E, 0x02, 0x81}, 1},
      {"xor dword [rip+0x800], 0x55",
       {0x83, 0x35, 0x00, 0x08, 0x00, 0x00, 0x55}, 1},
      {"cmp qword [rbx], 0x10", {0x48, 0x83, 0x3B, 0x10}, 1},
      {"add rdx, -0x80 (83 /0)", {0x48, 0x83, 0xC2, 0x80}, 1},
      {"sub r11d, 1 (83 /5, 32-bit)", {0x41, 0x83, 0xEB, 0x01}, 1},
      {"add al, 0x7F", {0x04, 0x7F}, 1},
      {"sub eax, 0x80000000", {0x2D, 0x00, 0x00, 0x00, 0x80}, 1},
      {"cmp rax, -2", {0x48, 0x3D, 0xFE, 0xFF, 0xFF, 0xFF}, 1},
      {"and ax, 0xFF00", {0x66, 0x25, 0x00, 0xFF}, 1},
      // ---- TEST ----------------------------------------------------------------
      {"test [rbx], eax", {0x85, 0x03}, 1},
      {"test byte [rsi], dl", {0x84, 0x16}, 1},
      {"test [rbx], ax", {0x66, 0x85, 0x03}, 1},
      {"test qword [rbx], 0x7fff0000",
       {0x48, 0xF7, 0x03, 0x00, 0x00, 0xFF, 0x7F}, 1},
      {"test byte [rbx+1], 0x81", {0xF6, 0x43, 0x01, 0x81}, 1},
      {"test al, 0x80", {0xA8, 0x80}, 1},
      {"test rax, 0x40 (A9)", {0x48, 0xA9, 0x40, 0x00, 0x00, 0x00}, 1},
      // ---- INC / DEC -----------------------------------------------------------
      {"inc dword [rbx]", {0xFF, 0x03}, 1},
      {"dec qword [rsi+rcx*8]", {0x48, 0xFF, 0x0C, 0xCE}, 1},
      {"inc byte [rbx]", {0xFE, 0x03}, 1},
      {"dec word [rsi]", {0x66, 0xFF, 0x0E}, 1},
      {"inc r9", {0x49, 0xFF, 0xC1}, 1},
      {"dec bh", {0xFE, 0xCF}, 1},
      // ---- PUSH / POP ----------------------------------------------------------
      {"push rbx ; pop rax", {0x53, 0x58}, 2},
      {"push r15 ; pop r9", {0x41, 0x57, 0x41, 0x59}, 2},
      {"push 0x7f ; push -0x1000",
       {0x6A, 0x7F, 0x68, 0x00, 0xF0, 0xFF, 0xFF}, 2},
      {"pop rsp", {0x5C}, 1},
      {"push qword [rbx+8] ; push qword [rsp]",
       {0xFF, 0x73, 0x08, 0xFF, 0x34, 0x24}, 2},
      // ---- multi-byte NOPs -------------------------------------------------------
      {"nop variants",
       {0x0F, 0x1F, 0x44, 0x00, 0x00,                         // nopl 0(%rax,%rax)
        0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, // nopw 0(...)
        0x0F, 0x1F, 0x04, 0x96,                               // nopl (%rsi,%rdx,4)
        0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xF3, 0x90}, // pause
       5},
      // ---- blocks (flag liveness across instructions) ---------------------------
      {"mov/add/store/cmp/lea block",
       {0x48, 0x8B, 0x03,             // mov rax, [rbx]
        0x48, 0x03, 0x06,             // add rax, [rsi]
        0x48, 0x89, 0x43, 0x08,       // mov [rbx+8], rax
        0x48, 0x39, 0xD0,             // cmp rax, rdx
        0x48, 0x8D, 0x0C, 0x48},      // lea rcx, [rax+rcx*2]
       5},
      {"add then inc keeps CF live", {0x01, 0x03, 0xFF, 0x06}, 2},
      {"sub then xor (dead flags) then dec",
       {0x48, 0x29, 0xD0, 0x31, 0xC9, 0xFF, 0xCA}, 3},
      // ---- bounds side exits ---------------------------------------------------
      {"side exit: first instruction", {0x48, 0x8B, 0x03}, 1, oob_rbx, 0, 0},
      {"side exit: mid-block RMW",
       {0xB8, 0x05, 0x00, 0x00, 0x00, 0x01, 0x03, 0xB9, 0x01, 0x00, 0x00, 0x00},
       3, oob_rbx, 1, 5},
      {"side exit: EA wraps below zero",
       {0x31, 0xC0, 0x8B, 0x43, 0xF8}, 2,
       [](CpuContext &c, std::mt19937_64 &) { c.rbx = 4; }, 1, 2},
      {"side exit: last bytes of guest memory",
       {0x8B, 0x03}, 1,
       [SIZE](CpuContext &c, std::mt19937_64 &) { c.rbx = SIZE - 4; }, 0, 0},
      {"side exit: push with rsp=0", {0x90, 0x53}, 2,
       [](CpuContext &c, std::mt19937_64 &) { c.rsp = 0; }, 1, 1},
  };

  for (const Case &tc : cases)
    run_case(mem, tc);
  test_executor_side_exit_progress(mem);

  std::printf("=== %d/%d cases passed (%llu trials each) ===\n",
              g_cases - g_failures, g_cases, (unsigned long long)g_trials);
  return g_failures == 0 ? 0 : 1;
}
