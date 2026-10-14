#pragma once

#include "ps4/common.hpp"
#include "ps4/elf.hpp"
#include "ps4/elf_loader.hpp"
#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/reloc.hpp"
#include "ps4/syscall_table.hpp"
#include <unordered_map>
#include <unordered_set>
#include <span>

namespace ps4 {

inline constexpr GuestAddr HLE_TRAMPOLINE_BASE = 0x00100000ULL;
inline constexpr u32 HLE_SYSCALL_DISPATCH_BASE = 0x80000000;

// Orbis OS return codes
inline constexpr u64 SCE_KERNEL_ERROR_EINVAL = 0x80020016ULL;
inline constexpr u64 SCE_USER_SERVICE_ERROR_NO_EVENT = 0x80960007ULL;

class DynamicLinker {
public:
    DynamicLinker(MemoryManager& memory, SyscallTable& syscalls, NidRegistry& registry);

    /// Initialize trampoline area in guest memory and hook HLE dispatcher
    bool initialize(GuestAddr trampoline_base = HLE_TRAMPOLINE_BASE);

    /// Get or generate an x86_64 trampoline stub address for a 64-bit NID
    GuestAddr get_or_create_trampoline(Nid nid);

    /// Resolve and patch relocations (PT_DYNAMIC or PT_SCE_RELA)
    bool resolve_relocations(std::span<const u8> file_data, const ElfLoader& loader, GuestAddr base_load_addr = 0);

    /// Patch a specific GOT jump slot directly to an NID's trampoline
    bool patch_jump_slot(GuestAddr got_slot_addr, Nid nid);

    /// Total number of generated trampolines
    [[nodiscard]] usize trampoline_count() const noexcept { return m_nid_to_trampoline.size(); }

    /// Strict HLE fail-loud development mode
    void set_strict_hle(bool enabled) noexcept;
    [[nodiscard]] bool is_strict_hle() const noexcept;

    /// Benign stubs whitelist
    void add_to_whitelist(Nid nid);
    [[nodiscard]] bool is_whitelisted(Nid nid) const noexcept;

private:
    MemoryManager& m_memory;
    SyscallTable& m_syscalls;
    NidRegistry& m_registry;
    GuestAddr m_trampoline_base{HLE_TRAMPOLINE_BASE};
    u32 m_next_trampoline_id{0};

    bool m_strict_hle{false};
    std::unordered_set<Nid> m_whitelist;

    std::unordered_map<Nid, GuestAddr> m_nid_to_trampoline;
    std::unordered_map<u32, Nid> m_id_to_nid;
};

} // namespace ps4
