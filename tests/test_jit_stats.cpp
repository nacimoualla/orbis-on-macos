#include "ps4/elf_loader.hpp"
#include "ps4/executor.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/jit_stats.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ps4;

namespace {

std::string key_of(std::vector<u8> bytes) {
    return normalize_opcode(bytes.data(), bytes.size()).to_string();
}

u64 interp_count(const std::string& key) {
    for (const OpcodeCount& e : jit_stats_interp_histogram()) {
        if (e.key.to_string() == key) return e.count;
    }
    return 0;
}

// Test 1: opcode normalization (prefix/REX stripping, maps, groups, mod)
void test_normalize() {
    std::cout << "[Test 1] Opcode normalization...\n";
    assert(key_of({0x48, 0x89, 0xD8}) == "89 reg");                 // mov rax, rbx
    assert(key_of({0x48, 0x8B, 0x45, 0xF8}) == "8B mem");           // mov rax, [rbp-8]
    assert(key_of({0x48, 0x83, 0xC0, 0x01}) == "83 /0 reg");        // add rax, 1
    assert(key_of({0x83, 0x7D, 0xFC, 0x00}) == "83 /7 mem");        // cmp dword [rbp-4], 0
    assert(key_of({0xFF, 0x15, 0, 0, 0, 0}) == "FF /2 mem");        // call [rip+x]
    assert(key_of({0xC3}) == "C3");
    assert(key_of({0xF3, 0xC3}) == "F3 C3");                        // rep ret
    assert(key_of({0x0F, 0x05}) == "0F 05");                        // syscall
    assert(key_of({0x0F, 0x84, 0, 0, 0, 0}) == "0F 84");            // jz rel32
    assert(key_of({0x66, 0x0F, 0x6F, 0x07}) == "66 0F 6F mem");     // movdqa xmm0, [rdi]
    assert(key_of({0xF3, 0x0F, 0x6F, 0xC1}) == "F3 0F 6F reg");     // movdqu xmm0, xmm1
    assert(key_of({0x66, 0x0F, 0x38, 0x00, 0xC1}) == "66 0F 38 00 reg"); // pshufb
    assert(key_of({0xC5, 0xF8, 0x77}) == "VEX.0F 77");              // vzeroupper
    assert(key_of({0xC5, 0xFA, 0x7F, 0x07}) == "VEX.F3.0F 7F mem"); // vmovdqu [rdi], xmm0
    assert(key_of({0xC4, 0xE2, 0x79, 0x18, 0x07}) == "VEX.66.0F38 18 mem"); // vbroadcastss
    assert(key_of({0x0F}) == "<undecodable>");
    assert(key_of({}) == "<undecodable>");

    OpcodeNotes notes;
    const u8 tls_load[] = {0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0}; // mov rax, fs:[0]
    const OpcodeKey k = normalize_opcode(tls_load, sizeof(tls_load), &notes);
    assert(k.to_string() == "8B mem");
    assert(notes.rex_w && notes.seg_fs_gs && !notes.lock);
    assert(OpcodeKey::from_index(k.index()).to_string() == k.to_string());
    std::cout << "  -> normalization passed\n";
}

// Program used by tests 2 and 3. Loops 100 times over a body the JIT only
// partly handles (inc/dec are FF /0 and FF /1, unsupported), then exits with
// the loop count.
const GuestAddr PROG_ADDR = 0x50000;
const u8 PROG[] = {
    0xB9, 0x64, 0x00, 0x00, 0x00, // 00: mov ecx, 100
    0x31, 0xD2,                   // 05: xor edx, edx
    0x48, 0x89, 0xD8,             // 07: mov rax, rbx   <- loop
    0x48, 0x83, 0xC0, 0x01,       // 0A: add rax, 1
    0x48, 0xFF, 0xC2,             // 0E: inc rdx        (unsupported)
    0xFF, 0xC9,                   // 11: dec ecx        (unsupported)
    0x75, 0xF2,                   // 13: jnz loop       (terminator)
    0x89, 0xD7,                   // 15: mov edi, edx
    0xB8, 0x01, 0x00, 0x00, 0x00, // 17: mov eax, 1 (SYS_exit)
    0x0F, 0x05,                   // 1C: syscall        (terminator)
};

RunResult run_program(bool with_jit) {
    MemoryManager memory;
    assert(memory.initialize());
    SyscallTable syscalls(memory);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false);
    assert(memory.write_bytes(PROG_ADDR, PROG, sizeof(PROG)));

    JitCompiler jit;
    CpuExecutor cpu(memory, syscalls);
    if (with_jit) cpu.set_jit_compiler(&jit);
    assert(cpu.setup_default_stack());
    cpu.set_entry_point(PROG_ADDR);
    const RunResult res = cpu.run(100000);
    assert(res.status == StepResult::Exited);
    assert(res.exit_code == 100);
    return res;
}

// Test 2: telemetry disabled records nothing
void test_disabled() {
    std::cout << "[Test 2] Disabled telemetry is inert...\n";
    set_jit_stats_enabled(false);
    reset_jit_stats();
    run_program(true);
    const JitStatsTotals t = jit_stats_totals();
    assert(t.jit_retired == 0 && t.interp_retired == 0 && t.blocks_compiled == 0);
    std::cout << "  -> disabled passed\n";
}

// Test 3: counters for a mixed JIT/interpreter run
void test_mixed_run() {
    std::cout << "[Test 3] Mixed JIT/interpreter counters...\n";
    set_jit_stats_enabled(true);
    reset_jit_stats();
    const RunResult res = run_program(true);
    const JitStatsTotals t = jit_stats_totals();

    // Every retired instruction is attributed to exactly one engine.
    assert(t.jit_retired + t.interp_retired == res.instructions_executed);
    // Block @00: 4 insns, block @07: 2 insns x 99, block @15: 2 insns.
    assert(t.jit_retired == 4 + 99 * 2 + 2);
    // inc, dec, jnz per iteration + the final syscall.
    assert(t.interp_retired == 3 * 100 + 1);
    assert(t.jit_block_executions == 1 + 99 + 1);
    assert(t.blocks_compiled == 3);
    // Failed compiles are not cached, so they recur on every visit:
    // inc/dec/jnz retried every iteration, plus the final syscall.
    assert(t.compile_failures == 3 * 100 + 1);

    assert(interp_count("FF /0 reg") == 100);
    assert(interp_count("FF /1 reg") == 100);
    assert(interp_count("75") == 100);
    assert(interp_count("0F 05") == 1);
    assert(interp_count("89 reg") == 0);

    std::ostringstream json;
    dump_jit_stats(json);
    const std::string s = json.str();
    assert(s.find("\"jit_retired\": 204") != std::string::npos);
    assert(s.find("\"key\": \"FF /0 reg\"") != std::string::npos);
    assert(s.find("\"reason\": \"unsupported\", \"blocks\": 2, \"failed_compiles\": 100") !=
           std::string::npos);
    assert(s.find("\"reason\": \"terminator\", \"blocks\": 1, \"failed_compiles\": 1") !=
           std::string::npos); // block @15 ends at syscall, then syscall alone
    std::cout << "  -> jit " << t.jit_retired << " / interp " << t.interp_retired
              << " instructions, " << t.blocks_compiled << " blocks\n";
}

// Test 4: pure interpreter run attributes everything to the interpreter
void test_interpreter_only() {
    std::cout << "[Test 4] Interpreter-only counters...\n";
    reset_jit_stats();
    const RunResult res = run_program(false);
    const JitStatsTotals t = jit_stats_totals();
    assert(t.jit_retired == 0 && t.blocks_compiled == 0);
    assert(t.interp_retired == res.instructions_executed);
    assert(interp_count("89 reg") == 1 + 100); // mov rax, rbx x100 + mov edi, edx
    assert(interp_count("83 /0 reg") == 100);
    std::cout << "  -> interpreter-only passed\n";
}

// Test 5: a sample ELF end to end (skipped if samples/ is not reachable)
void test_sample_elf() {
    std::cout << "[Test 5] samples/triangle_orbis.elf...\n";
    std::filesystem::path path = "samples/triangle_orbis.elf";
    if (!std::filesystem::exists(path)) path = "../samples/triangle_orbis.elf";
    if (!std::filesystem::exists(path)) {
        std::cout << "  -> skipped (sample not found)\n";
        return;
    }
    std::ifstream f(path, std::ios::binary);
    std::vector<u8> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    reset_jit_stats();
    MemoryManager memory;
    assert(memory.initialize());
    ElfLoader loader;
    assert(loader.load_from_memory(data));
    assert(memory.load_elf_segments(loader, data));
    SyscallTable syscalls(memory);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false);
    JitCompiler jit;
    CpuExecutor cpu(memory, syscalls);
    cpu.set_jit_compiler(&jit);
    assert(cpu.setup_default_stack());
    cpu.set_entry_point(loader.entry_point());
    const RunResult res = cpu.run(100000);
    const JitStatsTotals t = jit_stats_totals();
    assert(t.jit_retired + t.interp_retired == res.instructions_executed);
    assert(interp_count("0F 05") >= 1);
    std::cout << "  -> " << res.instructions_executed << " instructions, jit "
              << t.jit_retired << " / interp " << t.interp_retired << "\n";
}

} // namespace

int main() {
    std::cout << "=== JIT coverage telemetry tests ===\n";
    test_normalize();
    test_disabled();
    test_mixed_run();
    test_interpreter_only();
    test_sample_elf();
    std::cout << "All JIT stats tests passed.\n";
    return 0;
}
