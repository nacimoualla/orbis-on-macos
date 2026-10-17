#include "ps4/memory.hpp"
#include <sys/mman.h>
#include <cstring>
#include <iostream>

namespace ps4 {

std::string_view memory_error_to_string(MemoryError error) {
    switch (error) {
        case MemoryError::None:
            return "Success";
        case MemoryError::AlreadyInitialized:
            return "MemoryManager is already initialized";
        case MemoryError::NotInitialized:
            return "MemoryManager is not initialized";
        case MemoryError::AllocationFailed:
            return "Failed to allocate 8GB virtual memory via mmap";
        case MemoryError::OutOfBounds:
            return "Guest address or size is out of 8GB address space bounds";
        case MemoryError::ProtectionFailed:
            return "Failed to change memory protection flags via mprotect";
        case MemoryError::InvalidSegment:
            return "Segment file offset or virtual address is invalid";
    }
    return "Unknown error";
}

MemoryManager::MemoryManager() : m_total_size(GUEST_MEMORY_SIZE) {}

MemoryManager::MemoryManager(GuestSize total_size) : m_total_size(total_size) {}

MemoryManager::~MemoryManager() {
    release();
}

MemoryManager::MemoryManager(MemoryManager&& other) noexcept
    : m_base(other.m_base),
      m_total_size(other.m_total_size),
      m_last_error(other.m_last_error) {
    other.m_base = nullptr;
    other.m_total_size = 0;
    other.m_last_error = MemoryError::None;
}

MemoryManager& MemoryManager::operator=(MemoryManager&& other) noexcept {
    if (this != &other) {
        release();
        m_base = other.m_base;
        m_total_size = other.m_total_size;
        m_last_error = other.m_last_error;
        other.m_base = nullptr;
        other.m_total_size = 0;
        other.m_last_error = MemoryError::None;
    }
    return *this;
}

bool MemoryManager::initialize() {
    if (m_base != nullptr) {
        m_last_error = MemoryError::AlreadyInitialized;
        return false;
    }

    void* ptr = ::mmap(nullptr, m_total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (ptr == MAP_FAILED) {
        m_last_error = MemoryError::AllocationFailed;
        return false;
    }

    m_base = static_cast<u8*>(ptr);
    m_last_error = MemoryError::None;
    return true;
}

void MemoryManager::release() {
    if (m_base != nullptr) {
        ::munmap(m_base, m_total_size);
        m_base = nullptr;
        m_page_flags.clear();
    }
}

bool MemoryManager::protect(GuestAddr addr, GuestSize size, u32 elf_flags) {
    if (!is_valid_address(addr, size)) {
        m_last_error = MemoryError::OutOfBounds;
        return false;
    }
    if (size == 0) {
        m_last_error = MemoryError::None;
        return true;
    }
    constexpr GuestSize kGuestPageSize = 0x1000;//4kb orbis OS page
    constexpr GuestSize kHostPageSize = 0x4000;//16kb apple silicon page
    const GuestAddr start_4k = addr & ~(kGuestPageSize - 1);
    const GuestAddr end_4k = (addr + size + kGuestPageSize - 1) & ~(kGuestPageSize - 1);
    for (GuestAddr page = start_4k; page < end_4k; page += kGuestPageSize){
        m_page_flags[page >> 12] = elf_flags;
    }
    //align outer range into 16kb macOS pages
    const GuestAddr start_16k = start_4k & ~(kHostPageSize - 1);
    const GuestAddr end_16k = (end_4k + kHostPageSize - 1) & ~(kHostPageSize - 1);
    //union all of 4kb pages into one 16kb page
    for(GuestAddr host_page = start_16k; host_page < end_16k; host_page += kHostPageSize){
        u32 combined_elf_flags = 0;
        for(GuestSize sub = 0; sub < kHostPageSize; sub += kGuestPageSize) {
            auto it = m_page_flags.find((host_page + sub) >> 12);
            if (it != m_page_flags.end()) {
                combined_elf_flags |= it->second;
            } else {
                combined_elf_flags |= (PF_R | PF_W);
            }
        }
        int host_prot = PROT_NONE;
        if (combined_elf_flags & (PF_R | PF_X)) host_prot |= PROT_READ;
        if (combined_elf_flags & PF_W) host_prot |=PROT_WRITE;
        if(::mprotect(m_base + host_page, kHostPageSize, host_prot) != 0) {
            m_last_error = MemoryError::ProtectionFailed;
            return false;
        }
    }
    m_last_error = MemoryError::None;
    return true;
}

// Guest address currently being executed on this host thread, published by the
// executor. MemoryManager needs it so a write-watch can name the instruction
// that clobbered a location instead of only reporting the corruption later.
thread_local GuestAddr t_guest_rip = 0;
thread_local GuestAddr t_guest_rsp = 0;
thread_local GuestAddr t_guest_rbp = 0;
thread_local GuestAddr t_guest_rbx = 0;
thread_local int t_guest_tid = 0;
thread_local bool t_guest_in_hle = false;

// Optional write-watch. Armed from PS4_WATCH_ADDR. Sits on write_bytes rather
// than in the executor so it also catches stores made by HLE handler code,
// which never pass through the instruction decoder.
//
// Records into a RING, not the first N: the canary slot is reused stack, so the
// prologue's install and later activations all write the same address, and a
// first-hit-only watch reports the benign store and hides the clobber.
static std::atomic<GuestAddr> g_watch_addr{0};
static std::atomic<usize> g_watch_size{0};
static std::atomic<bool> g_watch_armed{false};
// Large: the watched slot is shared stack, so many unrelated frames write it
// and a small ring rolls past the activation we actually care about.
static constexpr usize kWatchRing = 4096;
static std::atomic<usize> g_watch_count{0};
static WatchHit g_watch_ring[kWatchRing];

void set_write_watch(GuestAddr addr, usize size) {
    g_watch_addr.store(addr, std::memory_order_relaxed);
    g_watch_size.store(size, std::memory_order_relaxed);
    g_watch_armed.store(addr != 0, std::memory_order_relaxed);
    std::fprintf(stderr, "[watch] armed on 0x%llx..0x%llx (%zu bytes)\n",
                 (unsigned long long)addr, (unsigned long long)(addr + size), size);
}

GuestAddr write_watch_addr() noexcept {
    return g_watch_addr.load(std::memory_order_relaxed);
}

GuestAddr write_watch_end() noexcept {
    return g_watch_addr.load(std::memory_order_relaxed) +
           g_watch_size.load(std::memory_order_relaxed);
}

void dump_write_watch_ring(u64 filter_addr) {
    const usize n = g_watch_count.load(std::memory_order_relaxed);
    std::fprintf(stderr, "[watch] ring: %zu writes recorded, capacity %zu, addr filter=0x%llx\n",
                 n, kWatchRing, (unsigned long long)filter_addr);
    const usize start = (n > kWatchRing) ? (n - kWatchRing) : 0;
    usize shown = 0;
    for (usize i = start; i < n; ++i) {
        const WatchHit& h = g_watch_ring[i % kWatchRing];
        // Filter by ADDRESS, not rbp. The clobbering store comes from a callee
        // prologue whose rbp differs from the failing frame's, so an rbp filter
        // excludes precisely the culprit.
        if (filter_addr != 0) {
            if (!(h.guest_addr <= filter_addr && filter_addr < h.guest_addr + h.size)) continue;
        }
        ++shown;
        std::fprintf(stderr,
                     "[watch] #%-3zu rip=0x%016llx rsp=0x%llx rbp=0x%llx rbx=0x%llx TID=%d %-3s "
                     "-> 0x%llx (%zu B)  sampled@0x%llx  old=0x%016llx new=0x%016llx%s\n",
                     i, (unsigned long long)h.rip, (unsigned long long)h.rsp,
                     (unsigned long long)h.rbp, (unsigned long long)h.rbx, h.tid, h.from_hle ? "HLE" : "dec",
                     (unsigned long long)h.guest_addr, h.size,
                     (unsigned long long)h.sample_addr,
                     (unsigned long long)h.old_value, (unsigned long long)h.new_value,
                     (h.old_value != h.new_value) ? "  <-- CHANGED" : "");
    }
    std::fprintf(stderr, "[watch] ring: %zu writes matched filter\n", shown);
}

bool MemoryManager::write_bytes(GuestAddr addr, const void* src, usize size) {
    if (!is_valid_address(addr, size)) {
        m_last_error = MemoryError::OutOfBounds;
        return false;
    }

    // Range overlap, not exact equality: a 16/32-byte VEX or movups store that
    // merely spans the slot must be caught. This code region is full of those.
    if (g_watch_armed.load(std::memory_order_relaxed)) {
        const GuestAddr wa = g_watch_addr.load(std::memory_order_relaxed);
        const usize ws = g_watch_size.load(std::memory_order_relaxed);
        if (addr < wa + ws && wa < addr + size) {
            WatchHit h{};
            h.rip = t_guest_rip;
            h.rsp = t_guest_rsp;
            h.rbp = t_guest_rbp;
            h.rbx = t_guest_rbx;
            h.tid = t_guest_tid;
            h.from_hle = t_guest_in_hle;
            h.guest_addr = addr;
            h.size = size;

            // Sample old/new at the START OF THE OVERLAP, not at the store base.
            // A wide store (movups/VEX) that merely spans the watched slot begins
            // before it, so sampling at `addr` reads 8 bytes that are outside the
            // watch entirely -- often the same bytes the store happens to rewrite,
            // which made every range-spanning write report old == new and look
            // like a no-op. Overlap start is the only address the watch actually
            // covers, so it is the only one whose before/after is meaningful.
            const GuestAddr ov_lo = (addr > wa) ? addr : wa;
            const GuestAddr ov_hi = (wa + ws < addr + size) ? (wa + ws) : (addr + size);
            usize take = ov_hi - ov_lo;
            if (take > sizeof(h.new_value)) take = sizeof(h.new_value);
            // Never sample past the mapped region; a store near the top of memory
            // would otherwise turn instrumentation into an out-of-bounds read.
            if (ov_lo + take > m_total_size) take = m_total_size - ov_lo;
            h.sample_addr = ov_lo;
            h.old_value = 0;
            h.new_value = 0;
            std::memcpy(&h.old_value, m_base + ov_lo, take);
            // The store's source buffer is indexed from its own base, so shift by
            // the same offset to land on the same guest byte.
            std::memcpy(&h.new_value, static_cast<const u8*>(src) + (ov_lo - addr), take);
            const usize slot = g_watch_count.fetch_add(1, std::memory_order_relaxed);
            g_watch_ring[slot % kWatchRing] = h;
        }
    }

    std::memcpy(m_base + addr, src, size);
    m_last_error = MemoryError::None;
    return true;
}

bool MemoryManager::read_bytes(GuestAddr addr, void* dest, usize size) const {
    if (!is_valid_address(addr, size)) {
        return false;
    }

    std::memcpy(dest, m_base + addr, size);
    return true;
}

bool MemoryManager::zero_memory(GuestAddr addr, GuestSize size) {
    if (!is_valid_address(addr, size)) {
        m_last_error = MemoryError::OutOfBounds;
        return false;
    }

    std::memset(m_base + addr, 0, size);
    m_last_error = MemoryError::None;
    return true;
}

bool MemoryManager::load_elf_segments(const ElfLoader& loader, std::span<const u8> file_data) {
    if (!is_initialized()) {
        m_last_error = MemoryError::NotInitialized;
        return false;
    }

    for (const auto& seg : loader.load_segments()) {
        // Ask the loader where the bytes actually are. In a SELF-wrapped module
        // seg.p_offset is the pre-packing layout and is stale -- eboot.bin's
        // PT_LOAD[0] claims 0x4000 but its data starts at 0xaeb0, so copying
        // from p_offset maps the wrong 0x4000 bytes and the first guest
        // instruction lands in a string.
        bool seg_found = false;
        const usize src_off = loader.segment_file_offset(seg, &seg_found);
        if (!seg_found) {
            m_last_error = MemoryError::InvalidSegment;
            return false;
        }
        if (src_off > file_data.size() ||
            seg.p_filesz > file_data.size() - src_off ||
            seg.p_memsz < seg.p_filesz) {
            m_last_error = MemoryError::InvalidSegment;
            return false;
        }

        if (!is_valid_address(seg.p_vaddr, seg.p_memsz)) {
            m_last_error = MemoryError::OutOfBounds;
            return false;
        }

        // Copy file content to guest virtual address
        if (seg.p_filesz > 0) {
            std::memcpy(m_base + seg.p_vaddr, file_data.data() + src_off, seg.p_filesz);
        }

        // Zero out .bss region if memsz > filesz
        if (seg.p_memsz > seg.p_filesz) {
            const GuestSize bss_size = seg.p_memsz - seg.p_filesz;
            std::memset(m_base + seg.p_vaddr + seg.p_filesz, 0, bss_size);
        }
    }

    m_last_error = MemoryError::None;
    return true;
}

} // namespace ps4
