#pragma once

#include "ps4/common.hpp"
#include "ps4/cpu_context.hpp"
#include "ps4/memory.hpp"
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ps4 {

// Signature for High-Level Emulation function shims
using HleFunction = std::function<void(CpuContext&, MemoryManager&)>;

struct HleSymbol {
    Nid nid{0};
    std::string name;
    std::string module_name;
    HleFunction handler;
};

class NidRegistry {
public:
    NidRegistry();
    ~NidRegistry();

    /// Join guest worker threads and clear subsystem pointers before
    /// MemoryManager, SyscallTable, and JitCompiler are destroyed in main().
    void shutdown();

    /// Register an HLE symbol handler
    void register_symbol(Nid nid, std::string_view name, std::string_view module_name, HleFunction handler);

    /// Find an HLE symbol by NID
    [[nodiscard]] const HleSymbol* find_by_nid(Nid nid) const;

    /// Find an HLE symbol by name
    [[nodiscard]] const HleSymbol* find_by_name(std::string_view name) const;

    /// HLE dispatch accounting for the stall watchdog. Const: touches only
    /// globals and thread entries, never the registry itself.
    void note_hle_dispatch(Nid nid, GuestAddr caller_rip = 0) const;
    void note_hle_return() const;

    /// Total count of registered symbols
    [[nodiscard]] usize size() const noexcept { return m_symbols_by_nid.size(); }

    /// Register standard built-in functions for libkernel and libc
    void register_default_libraries();

    /// Attach virtual file system for HLE I/O routines
    void set_vfs(class VirtualFileSystem* vfs);

    /// Attach MetalBackend for video out display presentation
    void set_metal_backend(class MetalBackend* metal);

    /// Attach SyscallTable for guest thread execution
    void set_syscall_table(class SyscallTable* syscalls);

    /// Attach JIT compiler for guest thread execution
    void set_jit_compiler(class JitCompiler* jit);

    /// Access active ThreadManager
    class ThreadManager& thread_manager(MemoryManager& mem);
    [[nodiscard]] class ThreadManager* thread_manager() const noexcept;

    /// Set pad digital buttons bitmask (e.g. from keyboard/mouse input)
    void set_pad_buttons(u32 buttons);

    /// Get total number of video flips performed
    [[nodiscard]] u64 get_flip_count() const noexcept;

private:
    std::unordered_map<Nid, HleSymbol> m_symbols_by_nid;
    std::unordered_map<std::string, Nid> m_nids_by_name;
};

} // namespace ps4
