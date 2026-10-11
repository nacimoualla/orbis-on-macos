#pragma once

#include "ps4/common.hpp"
#include "ps4/pm4.hpp"
#include <initializer_list>
#include <span>
#include <vector>

namespace ps4 {

class GpuRingBuffer {
public:
    explicit GpuRingBuffer(usize capacity_dwords = 0x10000); // 64K DWORDs = 256KB default
    ~GpuRingBuffer() = default;

    /// Write a single 32-bit word into ring buffer
    bool write(u32 dword);

    /// Write a span of 32-bit words
    bool write_dwords(std::span<const u32> dwords);

    /// Write a complete PM4 Type-3 packet
    bool write_type3_packet(u8 opcode, std::span<const u32> payload);
    bool write_type3_packet(u8 opcode, std::initializer_list<u32> payload) {
        return write_type3_packet(opcode, std::span<const u32>(payload.begin(), payload.size()));
    }

    /// Parse all pending unread packets from the ring buffer
    std::vector<Pm4Packet> parse_pending_packets();

    /// Reset read and write pointers
    void reset();

    /// Ring buffer capacity in DWORDs
    [[nodiscard]] usize capacity() const noexcept { return m_buffer.size(); }

    /// Number of unread DWORDs available
    [[nodiscard]] usize available_dwords() const noexcept;

    /// Read pointer offset (in DWORDs)
    [[nodiscard]] u32 read_pointer() const noexcept { return m_rptr; }

    /// Write pointer offset (in DWORDs)
    [[nodiscard]] u32 write_pointer() const noexcept { return m_wptr; }

private:
    std::vector<u32> m_buffer;
    u32 m_rptr{0};
    u32 m_wptr{0};
};

} // namespace ps4
