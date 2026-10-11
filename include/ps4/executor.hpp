#pragma once

#include "ps4/common.hpp"
#include "ps4/cpu_context.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscall_table.hpp"
#include <string_view>

namespace ps4 {

class SyscallTable;
class JitCompiler;

enum class StepResult {
    Ok,
    Syscall,
    Halted,
    Exited,
    IllegalInstruction,
    MemoryFault
};

std::string_view step_result_to_string(StepResult result);

struct RunResult {
    StepResult status{StepResult::Ok};
    usize instructions_executed{0};
    i32 exit_code{0};
};

class CpuExecutor {
public:
    CpuExecutor(MemoryManager& memory, SyscallTable& syscalls);

    /// Setup execution starting address
    void set_entry_point(GuestAddr entry) noexcept { m_context.rip = entry; }

    /// Allocate a 1MB guest stack region at the top of the 8GB space and initialize rsp
    bool setup_default_stack(GuestAddr stack_base = 0x1FFF00000ULL, GuestSize stack_size = 0x100000ULL);

    /// Attach JIT compiler for tier-2 dynamic recompilation
    void set_jit_compiler(JitCompiler* jit) noexcept { m_jit = jit; }
    [[nodiscard]] JitCompiler* jit_compiler() noexcept { return m_jit; }
    [[nodiscard]] const JitCompiler* jit_compiler() const noexcept { return m_jit; }

    /// Access register context
    [[nodiscard]] CpuContext& context() noexcept { return m_context; }
    [[nodiscard]] const CpuContext& context() const noexcept { return m_context; }

    /// Execute a single instruction at current RIP
    StepResult step();

    /// Run execution loop up to max_instructions (0 = until halt/exit)
    RunResult run(usize max_instructions = 0);

private:
    MemoryManager& m_memory;
    SyscallTable& m_syscalls;
    CpuContext m_context{};
    JitCompiler* m_jit{nullptr};
};

} // namespace ps4
