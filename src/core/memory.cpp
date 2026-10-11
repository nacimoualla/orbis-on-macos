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
    }
}

bool MemoryManager::protect(GuestAddr addr, GuestSize size, u32 elf_flags) {
    if (!is_valid_address(addr, size)) {
        m_last_error = MemoryError::OutOfBounds;
        return false;
    }

    int prot = PROT_NONE;
    if (elf_flags & PF_R) prot |= PROT_READ;
    if (elf_flags & PF_W) prot |= PROT_WRITE;
    if (elf_flags & PF_X) prot |= PROT_EXEC;

    if (::mprotect(m_base + addr, size, prot) != 0) {
        m_last_error = MemoryError::ProtectionFailed;
        return false;
    }

    m_last_error = MemoryError::None;
    return true;
}

bool MemoryManager::write_bytes(GuestAddr addr, const void* src, usize size) {
    if (!is_valid_address(addr, size)) {
        m_last_error = MemoryError::OutOfBounds;
        return false;
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
        if (seg.p_offset + seg.p_filesz > file_data.size()) {
            m_last_error = MemoryError::InvalidSegment;
            return false;
        }

        if (!is_valid_address(seg.p_vaddr, seg.p_memsz)) {
            m_last_error = MemoryError::OutOfBounds;
            return false;
        }

        // Copy file content to guest virtual address
        if (seg.p_filesz > 0) {
            std::memcpy(m_base + seg.p_vaddr, file_data.data() + seg.p_offset, seg.p_filesz);
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
