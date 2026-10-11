#pragma once

#include "ps4/common.hpp"
#include "ps4/memory.hpp"
#include "ps4/syscalls.hpp"
#include <functional>
#include <iostream>
#include <unordered_map>

namespace ps4 {
 
class VirtualFileSystem;

struct SyscallContext {
    u64 args[6]{0};               // x86_64 SysV ABI registers: rdi, rsi, rdx, r10, r8, r9
    u64 rax{0};                   // Return value register
    u64 rcx{0};                   // 4th argument in userspace calling convention (System V)
    u64 rsp{0};                   // Stack pointer register
    u64 rbp{0};                   // Base pointer register
    MemoryManager* memory{nullptr}; // Pointer to guest memory manager
    VirtualFileSystem* vfs{nullptr}; // Pointer to virtual file system

    SyscallContext() = default;
    explicit SyscallContext(MemoryManager& mem, VirtualFileSystem* vfs_ptr = nullptr)
        : memory(&mem), vfs(vfs_ptr) {}
};

using SyscallHandler = std::function<OrbisResult(SyscallContext&)>;

class SyscallTable {
public:
    explicit SyscallTable(MemoryManager& memory);

    /// Get associated memory manager
    [[nodiscard]] MemoryManager& memory() noexcept { return m_memory; }
    [[nodiscard]] const MemoryManager& memory() const noexcept { return m_memory; }

    /// Register a custom syscall handler
    void register_handler(u32 syscall_num, SyscallHandler handler);

    /// Dispatch a syscall by number (automatically binds m_memory if ctx.memory is null)
    OrbisResult dispatch(u32 syscall_num, SyscallContext& ctx);

    /// Check if a syscall has a registered handler
    [[nodiscard]] bool is_registered(u32 syscall_num) const;

    /// Register default HLE implementations for core system calls
    void register_default_handlers();

    /// Check if sys_exit has been triggered
    [[nodiscard]] bool has_exited() const noexcept { return m_has_exited; }
    [[nodiscard]] i32 exit_code() const noexcept { return m_exit_code; }
    void reset_exit() noexcept { m_has_exited = false; m_exit_code = 0; }

    /// Control whether sys_exit calls std::exit() immediately or flags exit
    void set_terminate_on_exit(bool terminate) noexcept { m_terminate_on_exit = terminate; }

    /// Redirect sys_write stdout/stderr output (defaults to std::cout)
    void set_output_stream(std::ostream* out) noexcept { m_out_stream = out; }

    /// Attach virtual file system
    void set_vfs(VirtualFileSystem* vfs) noexcept { m_vfs = vfs; }
    [[nodiscard]] VirtualFileSystem* vfs() noexcept { return m_vfs; }
    [[nodiscard]] const VirtualFileSystem* vfs() const noexcept { return m_vfs; }

private:
    MemoryManager& m_memory;
    std::unordered_map<u32, SyscallHandler> m_handlers;
    bool m_has_exited{false};
    i32 m_exit_code{0};
    bool m_terminate_on_exit{false};
    std::ostream* m_out_stream{nullptr};
    VirtualFileSystem* m_vfs{nullptr};
};

} // namespace ps4
