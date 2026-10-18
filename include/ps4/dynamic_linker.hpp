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
#include <string>
#include <vector>

namespace ps4 {

inline constexpr GuestAddr HLE_TRAMPOLINE_BASE = 0x00100000ULL;
inline constexpr u32 HLE_SYSCALL_DISPATCH_BASE = 0x80000000;

// Orbis OS return codes
inline constexpr u64 SCE_KERNEL_ERROR_EINVAL = 0x80020016ULL;
inline constexpr u64 SCE_USER_SERVICE_ERROR_NO_EVENT = 0x80960007ULL;

// A module mapped into guest address space and known to export symbols.
//
// PS4 modules do not publish an ELF .dynsym a consumer can walk. Each module
// carries a symbol table in its PT_SCE_PROGLIB, reached through the SCE dynamic
// tags, and each name is a 16-byte record "<11-char base64>#XX#YY\0" rather than
// a C identifier. The base64 is big-endian over the 64-bit NID, so exports are
// naturally keyed by Nid -- the same key the HLE trampolines already use.
struct LoadedModule {
    u32 handle{0};
    std::string path;
    GuestAddr base{0};
    GuestAddr init{0};          // DT_INIT, 0 when the module has no entry point
    GuestAddr fini{0};          // DT_FINI
    GuestSize size{0};          // total mapped span
    usize export_count{0};
};

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

    // --- PRX module loading -------------------------------------------------

    /// Map a PRX at \p base, relocate it, and record its exports.
    ///
    /// The guest address must already be reserved (see module_base allocator);
    /// this only writes into it. Returns a handle, or 0 on failure.
    ///
    /// \p run_init is deliberately separate from loading: MonoAssembliesPS4.prx
    /// has DT_INIT == 0, so calling an entry point unconditionally would jump to
    /// the module base. Call run_module_init() for the modules that have one.
    u32 load_module(const std::string& path, GuestAddr base, bool run_init = false);

    /// Call a loaded module's DT_INIT (a no-op when it is 0).
    bool run_module_init(u32 handle);

    /// Guest address of the given NID in any loaded module, or 0.
    ///
    /// Consulted by the trampoline dispatcher before the HLE registry, so a
    /// module loaded after the executable's relocations already ran still takes
    /// over every NID it exports. The trampoline stub carries only a dispatch
    /// id, so no GOT patching or stub rewriting is needed.
    [[nodiscard]] GuestAddr find_export(Nid nid) const noexcept;

    [[nodiscard]] const LoadedModule* module(u32 handle) const noexcept;
    [[nodiscard]] usize module_count() const noexcept { return m_modules.size(); }

    /// Which loaded module contains this guest address, and at what offset.
    /// Returns nullptr when the address is not inside any module.
    [[nodiscard]] const LoadedModule* module_for_addr(GuestAddr addr,
                                                      GuestAddr* offset) const noexcept;

    /// Lowest address the NID importer assigned to a module, or 0 if none.
    /// Used to tell a module range apart from the base module's.
    [[nodiscard]] GuestAddr first_module_base() const noexcept;
    [[nodiscard]] usize module_export_count() const noexcept { return m_module_exports.size(); }

    /// Total number of generated trampolines
    [[nodiscard]] usize trampoline_count() const noexcept { return m_nid_to_trampoline.size(); }

    /// Strict HLE fail-loud development mode
    void set_strict_hle(bool enabled) noexcept;
    [[nodiscard]] bool is_strict_hle() const noexcept;

    /// Benign stubs whitelist
    void add_to_whitelist(Nid nid);
    [[nodiscard]] bool is_whitelisted(Nid nid) const noexcept;

private:
    /// Walk the SCE dynamic table and record every symbol table whose
    /// st_shndx != SHN_UNDEF into m_module_exports.
    void build_export_table(std::span<const u8> file_data, const ElfLoader& loader,
                            GuestAddr base, u32 handle, GuestSize span);

    /// Copy PT_LOAD contents to \p base + p_vaddr and zero the .bss tail.
    bool map_segments(std::span<const u8> file_data, const ElfLoader& loader, GuestAddr base, GuestSize* span_out, u32 handle);

    MemoryManager& m_memory;
    SyscallTable& m_syscalls;
    NidRegistry& m_registry;
    GuestAddr m_trampoline_base{HLE_TRAMPOLINE_BASE};
    u32 m_next_trampoline_id{0};

    bool m_strict_hle{false};
    std::unordered_set<Nid> m_whitelist;

    std::unordered_map<Nid, GuestAddr> m_nid_to_trampoline;
    std::unordered_map<u32, Nid> m_id_to_nid;

    /// Addresses already written with a trampoline for an imported NID.
    ///
    /// A trampoline is only a valid placeholder for a symbol that is *called*.
    /// For a data import the guest reads the slot and dereferences it, so leaving
    /// a stub address there hands it `B8 xx xx xx xx 0F 05 C3` -- machine code --
    /// where a struct pointer belongs. eboot imports all 19 STT_OBJECT AOT
    /// descriptor tables from MonoAssembliesPS4.prx through R_X86_64_64 this way.
    /// Once the defining module loads, every recorded slot is rewritten with the
    /// real address.
    std::unordered_map<Nid, std::vector<GuestAddr>> m_nid_to_slots;

    std::unordered_map<u32, LoadedModule> m_modules;
    std::unordered_map<Nid, GuestAddr> m_module_exports;
    u32 m_next_module_handle{0x2000};
};

/// Unimplemented-stub warning stats (suppressed-after-cap totals). Defined in
/// dynamic_linker.cpp; called from the run exit path.
void dump_hle_warning_stats();

/// The process-wide linker, so HLE handlers can reach module loading without
/// threading a reference through every registry signature. main.cpp installs it
/// once the linker is constructed; it is null before that and after teardown.
DynamicLinker* active_linker() noexcept;
void set_active_linker(DynamicLinker* linker) noexcept;

/// Next free guest address for module mapping. Sits in the mspace arena range,
/// well clear of the 0x1C0000000 direct-memory ceiling.
/// Reserve \p need bytes of module space and return the base, or 0 if the
/// region is exhausted. The reservation accounts for the real span, so
/// consecutive modules cannot overlap.
GuestAddr module_base_allocator(GuestSize need) noexcept;

void dump_hle_histogram();
void dump_export_tables();
std::string describe_guest_addr(GuestAddr a);
} // namespace ps4
