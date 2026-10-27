#pragma once

// JIT coverage telemetry: which guest x86 instructions keep code out of the
// JIT, weighted by how often they actually execute.
//
// Off by default. Enable with the environment variable PS4_JIT_STATS:
//   PS4_JIT_STATS=1            write ./jit_stats.json at exit
//   PS4_JIT_STATS=/tmp/x.json  write that path at exit
//   PS4_JIT_STATS=-            write the JSON to stderr at exit
// Analyse the output with scripts/jit_coverage.py.
//
// When disabled every hook is a single relaxed atomic load.

#include "ps4/common.hpp"
#include <atomic>
#include <iosfwd>
#include <string>
#include <vector>

namespace ps4 {

class MemoryManager;

/// Why the JIT stopped decoding a block at a given instruction.
enum class JitStopReason : u8 {
    Terminator,  // deliberate block end: syscall/call/ret/jcc/jmp/hlt/ud2/int3
    Unsupported, // the decoder does not handle this instruction
    Limit,       // block hit the max-instructions cap
};

/// Normalized opcode identity: prefixes/REX stripped (noted separately),
/// escape maps folded in, ModRM /reg extension and reg-vs-mem form kept.
struct OpcodeKey {
    u8 op{0};
    u8 map{0};  // 0 = one-byte, 1 = 0F, 2 = 0F 38, 3 = 0F 3A
    u8 pfx{0};  // mandatory/legacy prefix: 0 = none, 1 = 66, 2 = F3, 3 = F2
    u8 ext{0};  // ModRM.reg (meaningful only for group opcodes)
    u8 mod{0};  // 0 = no ModRM, 1 = register form, 2 = memory form, 3 = undecodable
    bool vex{false};

    [[nodiscard]] u32 index() const noexcept;
    [[nodiscard]] static OpcodeKey from_index(u32 index) noexcept;
    [[nodiscard]] bool has_group_ext() const noexcept;
    /// Canonical text form, e.g. "83 /0 reg", "66 0F 6F mem", "VEX.F3.0F 7F mem".
    [[nodiscard]] std::string to_string() const;
};

/// Per-instruction prefix notes that are not part of the key.
struct OpcodeNotes {
    bool rex_w{false};
    bool seg_fs_gs{false};
    bool lock{false};
};

/// Decode the opcode identity of the instruction at code[0..avail).
OpcodeKey normalize_opcode(const u8* code, usize avail, OpcodeNotes* notes = nullptr) noexcept;

namespace jit_stats_detail {
extern std::atomic<int> g_state; // 0 = not yet read from env, 1 = off, 2 = on
int init_from_env() noexcept;
} // namespace jit_stats_detail

[[nodiscard]] inline bool jit_stats_enabled() noexcept {
    int s = jit_stats_detail::g_state.load(std::memory_order_relaxed);
    if (s == 0) [[unlikely]] {
        s = jit_stats_detail::init_from_env();
    }
    return s == 2;
}

/// Force telemetry on/off (tests). Enabling allocates the counter tables.
void set_jit_stats_enabled(bool enabled);
/// Zero all counters.
void reset_jit_stats();

// --- Hooks (call only when jit_stats_enabled()) -----------------------------

/// Interpreter is about to execute the instruction at rip.
void jit_stats_record_interp(const MemoryManager& memory, GuestAddr rip) noexcept;
/// A JIT block retired `instructions` guest instructions.
void jit_stats_record_jit_block(usize instructions) noexcept;
/// The JIT block compiler stopped at stop_rip. `instructions` is the number of
/// instructions it did compile (0 = compile failed, interpreter fallback).
void jit_stats_record_block_end(const MemoryManager& memory, GuestAddr stop_rip,
                                JitStopReason reason, usize instructions) noexcept;

// --- Results -----------------------------------------------------------------

struct JitStatsTotals {
    u64 jit_retired{0};          // guest instructions retired inside JIT blocks
    u64 interp_retired{0};       // guest instructions executed by CpuExecutor::step()
    u64 jit_block_executions{0}; // JIT block invocations
    u64 blocks_compiled{0};      // successful block compiles
    u64 compile_failures{0};     // compile attempts that produced no block
};

struct OpcodeCount {
    OpcodeKey key;
    u64 count{0};
};

[[nodiscard]] JitStatsTotals jit_stats_totals();
/// Interpreter-executed opcodes, sorted by count (descending).
[[nodiscard]] std::vector<OpcodeCount> jit_stats_interp_histogram();

/// Write all counters as JSON (see scripts/jit_coverage.py for the schema).
void dump_jit_stats(std::ostream& os);
/// Write JSON to the PS4_JIT_STATS destination. No-op when disabled.
void write_jit_stats();

} // namespace ps4
