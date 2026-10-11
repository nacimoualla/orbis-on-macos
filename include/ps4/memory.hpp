#pragma once

#include "ps4/common.hpp"
#include "ps4/elf.hpp"
#include "ps4/elf_loader.hpp"
#include <cassert>
#include <cstddef>
#include <span>
#include <string_view>

namespace ps4 {

enum class MemoryError {
    None,
    AlreadyInitialized,
    NotInitialized,
    AllocationFailed,
    OutOfBounds,
    ProtectionFailed,
    InvalidSegment
};

std::string_view memory_error_to_string(MemoryError error);

class MemoryManager {
public:
    MemoryManager();
    explicit MemoryManager(GuestSize total_size);
    ~MemoryManager();

    // Disable copy semantics (unique ownership of 8GB virtual memory)
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // Enable move semantics
    MemoryManager(MemoryManager&& other) noexcept;
    MemoryManager& operator=(MemoryManager&& other) noexcept;

    /// Allocate virtual memory space via mmap
    bool initialize();

    /// Release allocated memory space via munmap
    void release();

    /// Check if virtual memory is initialized
    [[nodiscard]] bool is_initialized() const noexcept { return m_base != nullptr; }

    /// Base host pointer for the guest virtual memory space
    [[nodiscard]] u8* base_address() noexcept { return m_base; }
    [[nodiscard]] const u8* base_address() const noexcept { return m_base; }

    /// Total size of the allocated address space (8 GB by default)
    [[nodiscard]] GuestSize total_size() const noexcept { return m_total_size; }

    /// Validate if a guest address and size span is within the mapped space
    [[nodiscard]] bool is_valid_address(GuestAddr addr, GuestSize size = 1) const noexcept {
        if (!m_base) return false;
        if (size == 0) return addr < m_total_size;
        return (addr < m_total_size) && (size <= m_total_size - addr);
    }

    /// Translate a guest virtual address to a host pointer
    [[nodiscard]] void* to_host_ptr(GuestAddr addr) noexcept {
        if (!is_valid_address(addr)) return nullptr;
        return static_cast<void*>(m_base + addr);
    }

    [[nodiscard]] const void* to_host_ptr(GuestAddr addr) const noexcept {
        if (!is_valid_address(addr)) return nullptr;
        return static_cast<const void*>(m_base + addr);
    }

    /// Typed translation helper
    template <typename T>
    [[nodiscard]] T* to_host_ptr(GuestAddr addr) noexcept {
        return static_cast<T*>(to_host_ptr(addr));
    }

    template <typename T>
    [[nodiscard]] const T* to_host_ptr(GuestAddr addr) const noexcept {
        return static_cast<const T*>(to_host_ptr(addr));
    }

    /// Translate a host pointer back to a guest address
    [[nodiscard]] GuestAddr to_guest_addr(const void* host_ptr) const noexcept {
        const auto* ptr = static_cast<const u8*>(host_ptr);
        if (ptr < m_base || ptr >= m_base + m_total_size) {
            return static_cast<GuestAddr>(-1);
        }
        return static_cast<GuestAddr>(ptr - m_base);
    }

    /// Change protection flags for a page-aligned range of guest memory
    bool protect(GuestAddr addr, GuestSize size, u32 elf_flags);

    /// Write raw data into guest memory
    bool write_bytes(GuestAddr addr, const void* src, usize size);

    /// Read raw data from guest memory
    bool read_bytes(GuestAddr addr, void* dest, usize size) const;

    /// Read a typed value directly from guest memory
    template <typename T>
    bool read_value(GuestAddr addr, T& out_value) const {
        return read_bytes(addr, &out_value, sizeof(T));
    }

    /// Write a typed value directly to guest memory
    template <typename T>
    bool write_value(GuestAddr addr, const T& value) {
        return write_bytes(addr, &value, sizeof(T));
    }

    /// Zero out a region of guest memory
    bool zero_memory(GuestAddr addr, GuestSize size);

    /// Load all PT_LOAD segments of an ELF file into guest memory
    bool load_elf_segments(const ElfLoader& loader, std::span<const u8> file_data);

    /// Last error encountered
    [[nodiscard]] MemoryError last_error() const noexcept { return m_last_error; }
    [[nodiscard]] std::string_view error_message() const noexcept {
        return memory_error_to_string(m_last_error);
    }

private:
    u8* m_base{nullptr};
    GuestSize m_total_size{GUEST_MEMORY_SIZE};
    MemoryError m_last_error{MemoryError::None};
};

} // namespace ps4
