#pragma once

#include "ps4/common.hpp"
#include "ps4/cpu_context.hpp"
#include "ps4/memory.hpp"
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ps4 {

// Signature for natively compiled basic blocks
// Argument: pointer to guest CpuContext in register x0
using JittedBlockFunc = void (*)(CpuContext* ctx);

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
    void b(i32 offset_words);
    void b_cond(u8 cond, i32 offset_words);

    // Block calling convention helpers
    void emit_prologue();
    void emit_epilogue();

private:
    std::vector<u32> m_code;
};

class JitCompiler {
public:
    static constexpr usize DEFAULT_CODE_CACHE_SIZE = 16 * 1024 * 1024; // 16 MB executable buffer

    explicit JitCompiler(usize code_cache_size = DEFAULT_CODE_CACHE_SIZE);
    ~JitCompiler();

    JitCompiler(const JitCompiler&) = delete;
    JitCompiler& operator=(const JitCompiler&) = delete;
    JitCompiler(JitCompiler&& other) noexcept;
    JitCompiler& operator=(JitCompiler&& other) noexcept;

    /// Compile a basic block starting at guest_rip
    JittedBlockFunc compile_block(GuestAddr guest_rip, const MemoryManager& memory);

    /// Execute a block at current ctx.rip (compiling on demand if not cached)
    /// Returns true if executed via JIT, false if fell back to interpreter
    bool execute_block(CpuContext& ctx, const MemoryManager& memory);

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
    Arm64Emitter m_emitter;
    mutable std::mutex m_cache_mutex;

    JittedBlockFunc compile_block_locked(GuestAddr guest_rip, const MemoryManager& memory);
    void* allocate_code(usize size);
    void commit_code(void* dest, const u32* src, usize size_bytes);
};

} // namespace ps4
