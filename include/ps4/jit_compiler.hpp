#pragma once

#include "ps4/common.hpp"
#include "ps4/cpu_context.hpp"
#include "ps4/memory.hpp"
#include <cstddef>
#include <memory>
#include <mutex>
#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ps4 {

// Signature for natively compiled basic blocks (AAPCS64):
//   x0 = guest CpuContext*            (kept in x19 inside the block)
//   x1 = host base of guest memory    (kept in x20; host = base + guest addr)
//   x2 = size of the guest address space in bytes (x21 = size - 8 is the
//        highest guest address a <= 8-byte access may start at)
// Returns the number of guest instructions retired. A block retires all of
// its instructions unless a memory access fails the bounds check, in which
// case it side-exits with ctx->rip at the faulting instruction (so the
// interpreter executes it) and returns the count retired before it.
using JittedBlockFunc = u64 (*)(CpuContext* ctx, u8* guest_base, u64 guest_size);

struct JitBlockMetrics {
    usize instructions_compiled{0};
    usize bytes_emitted{0};
    usize execution_count{0};
};

class Arm64Emitter {
public:
    Arm64Emitter() = default;

    void clear() { m_code.clear(); }
    [[nodiscard]] const std::vector<u32>& code() const noexcept { return m_code; }
    [[nodiscard]] usize size_bytes() const noexcept { return m_code.size() * sizeof(u32); }

    void emit(u32 insn) { m_code.push_back(insn); }

    // High-level ARM64 instruction emitters
    void nop();
    void ret();
    void mov_reg(u8 rd, u8 rm);
    void mov_imm64(u8 rd, u64 val);
    void str_reg(u8 rt, u8 rn, u32 offset_bytes);
    void ldr_reg(u8 rt, u8 rn, u32 offset_bytes);
    void str_w(u8 rt, u8 rn, u32 offset_bytes);
    void ldr_w(u8 rt, u8 rn, u32 offset_bytes);
    void str_zero(u8 rn, u32 offset_bytes);
    void add_reg(u8 rd, u8 rn, u8 rm);
    void sub_reg(u8 rd, u8 rn, u8 rm);
    void adds_reg(u8 rd, u8 rn, u8 rm);
    void subs_reg(u8 rd, u8 rn, u8 rm);
    void add_imm(u8 rd, u8 rn, u32 imm12);
    void sub_imm(u8 rd, u8 rn, u32 imm12);
    void and_reg(u8 rd, u8 rn, u8 rm);
    void orr_reg(u8 rd, u8 rn, u8 rm);
    void eor_reg(u8 rd, u8 rn, u8 rm);
    void add_w(u8 rd, u8 rn, u8 rm);
    void sub_w(u8 rd, u8 rn, u8 rm);
    void b(i32 offset_words);
    void b_cond(u8 cond, i32 offset_words);
    void blr(u8 rn);

    // --- Encodings used by the memory-operand / flags translator ----------
    // `sf` selects the 64-bit (X) form, otherwise the 32-bit (W) form.
    enum : u8 { SHIFT_LSL = 0, SHIFT_LSR = 1, SHIFT_ASR = 2 };

    // Data processing (shifted register): rd = rn OP (rm SHIFT amount)
    void add_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift = SHIFT_LSL, u8 amount = 0);
    void adds_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift = SHIFT_LSL, u8 amount = 0);
    void subs_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift = SHIFT_LSL, u8 amount = 0);
    void ands_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift = SHIFT_LSL, u8 amount = 0);
    void orr_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift = SHIFT_LSL, u8 amount = 0);
    void eor_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift = SHIFT_LSL, u8 amount = 0);
    void bic_shifted(bool sf, u8 rd, u8 rn, u8 rm, u8 shift = SHIFT_LSL, u8 amount = 0);
    void cmp_reg(bool sf, u8 rn, u8 rm) { subs_shifted(sf, 31, rn, rm); }
    void mov_w(u8 rd, u8 rm); // ORR wd, wzr, wm (zero-extends into xd)

    // Bitfield moves and their aliases
    void ubfm(bool sf, u8 rd, u8 rn, u8 immr, u8 imms);
    void sbfm(bool sf, u8 rd, u8 rn, u8 immr, u8 imms);
    void bfm(bool sf, u8 rd, u8 rn, u8 immr, u8 imms);
    void lsl_imm(bool sf, u8 rd, u8 rn, u8 shift);
    void lsr_imm(bool sf, u8 rd, u8 rn, u8 shift);
    void asr_imm(bool sf, u8 rd, u8 rn, u8 shift);
    void ubfx(bool sf, u8 rd, u8 rn, u8 lsb, u8 width);
    void bfi(bool sf, u8 rd, u8 rn, u8 lsb, u8 width);
    void sxt(bool sf, u8 rd, u8 rn, u8 from_bits); // SXTB/SXTH/SXTW

    void lsrv(bool sf, u8 rd, u8 rn, u8 rm);
    void movz(bool sf, u8 rd, u16 imm16, u8 hw = 0);
    void cset(bool sf, u8 rd, u8 cond);
    void mrs_nzcv(u8 rt);
    /// rd = rn + imm for any signed 64-bit imm (uses `scratch` if it does not
    /// fit two 12-bit immediates).
    void add_imm64(u8 rd, u8 rn, i64 imm, u8 scratch);

    // Loads/stores, unsigned scaled immediate offset (offset in bytes)
    void ldrb_imm(u8 rt, u8 rn, u32 offset_bytes);
    void ldrh_imm(u8 rt, u8 rn, u32 offset_bytes);
    void strb_imm(u8 rt, u8 rn, u32 offset_bytes);
    void strh_imm(u8 rt, u8 rn, u32 offset_bytes);

    // Loads/stores, register offset: [rn, rm] (64-bit index, no shift)
    void ldrb_idx(u8 rt, u8 rn, u8 rm);   // zero-extends
    void ldrh_idx(u8 rt, u8 rn, u8 rm);   // zero-extends
    void ldr_w_idx(u8 rt, u8 rn, u8 rm);  // zero-extends
    void ldr_x_idx(u8 rt, u8 rn, u8 rm);
    void ldrsb_x_idx(u8 rt, u8 rn, u8 rm);
    void ldrsh_x_idx(u8 rt, u8 rn, u8 rm);
    void ldrsw_idx(u8 rt, u8 rn, u8 rm);
    void strb_idx(u8 rt, u8 rn, u8 rm);
    void strh_idx(u8 rt, u8 rn, u8 rm);
    void str_w_idx(u8 rt, u8 rn, u8 rm);
    void str_x_idx(u8 rt, u8 rn, u8 rm);

    /// Current position (in instruction words), e.g. for later patching.
    [[nodiscard]] usize pos() const noexcept { return m_code.size(); }
    /// Re-target a B.cond emitted at `at` to the instruction word `target`.
    void patch_b_cond(usize at, usize target);

    // Block calling convention helpers
    void emit_prologue();
    void emit_epilogue();

private:
    std::vector<u32> m_code;
};

class JitCompiler {
public:
    // Reserved, not committed: pages are only backed once code is written.
    // 16 MB filled up during Unity gameplay, after which every new block
    // retried its compile under the exclusive lock and ran interpreted.
    static constexpr usize DEFAULT_CODE_CACHE_SIZE = 256 * 1024 * 1024;

    explicit JitCompiler(usize code_cache_size = DEFAULT_CODE_CACHE_SIZE);
    ~JitCompiler();

    JitCompiler(const JitCompiler&) = delete;
    JitCompiler& operator=(const JitCompiler&) = delete;
    JitCompiler(JitCompiler&& other) noexcept;
    JitCompiler& operator=(JitCompiler&& other) noexcept;

    /// Compile a basic block starting at guest_rip
    JittedBlockFunc compile_block(GuestAddr guest_rip, const MemoryManager& memory);

    /// Execute a block at current ctx.rip (compiling on demand if not cached)
    /// Returns true if executed via JIT, false if the caller must fall back to
    /// the interpreter (nothing compiled, or the block side-exited before
    /// retiring any instruction). `retired` receives the number of guest
    /// instructions the block retired.
    /// The block reads and writes guest memory through memory.base_address().
    bool execute_block(CpuContext& ctx, const MemoryManager& memory,
                       usize* retired = nullptr);

    /// Check if a guest address has a compiled block in cache
    [[nodiscard]] bool is_block_cached(GuestAddr guest_rip) const;

    /// Invalidate cache
    void clear_cache();

    /// Metrics and stats
    [[nodiscard]] usize cached_block_count() const noexcept;
    [[nodiscard]] usize code_buffer_used() const noexcept;
    [[nodiscard]] usize total_invocations() const noexcept;
    [[nodiscard]] const JitBlockMetrics* get_block_metrics(GuestAddr guest_rip) const;

private:
    u8* m_code_buffer{nullptr};
    usize m_code_buffer_size{0};
    usize m_code_buffer_offset{0};
    usize m_total_invocations{0};

    std::unordered_map<GuestAddr, JittedBlockFunc> m_blocks;
    std::unordered_map<GuestAddr, JitBlockMetrics> m_metrics;
    std::unordered_set<GuestAddr> m_failed; // block starts that compiled to nothing
    Arm64Emitter m_emitter;
    // Shared for cache hits (every guest thread, every block), exclusive for
    // compiles and cache-wide changes. Hit counters are bumped atomically under
    // the shared lock.
    mutable std::shared_mutex m_cache_mutex;
    // Identifies this cache's current contents. Each guest thread keeps a
    // private rip -> block map tagged with the epoch it was filled under, so
    // cache hits take no lock; a constructor or clear_cache() takes a fresh
    // epoch from a global counter, which drops every thread's copy.
    std::atomic<u64> m_epoch{0};

    JittedBlockFunc compile_block_locked(GuestAddr guest_rip, const MemoryManager& memory);
    void* allocate_code(usize size);
    void commit_code(void* dest, const u32* src, usize size_bytes);
};

} // namespace ps4
