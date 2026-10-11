#include "ps4/gpu_ring_buffer.hpp"
#include <algorithm>

namespace ps4 {

GpuRingBuffer::GpuRingBuffer(usize capacity_dwords)
    : m_buffer(capacity_dwords > 0 ? capacity_dwords : 0x10000, 0) {}

bool GpuRingBuffer::write(u32 dword) {
    if (m_buffer.empty()) return false;
    m_buffer[m_wptr] = dword;
    m_wptr = (m_wptr + 1) % static_cast<u32>(m_buffer.size());
    return true;
}

bool GpuRingBuffer::write_dwords(std::span<const u32> dwords) {
    for (u32 dw : dwords) {
        if (!write(dw)) return false;
    }
    return true;
}

bool GpuRingBuffer::write_type3_packet(u8 opcode, std::span<const u32> payload) {
    const u32 header = encode_pm4_type3_header(opcode, static_cast<u16>(payload.size()));
    if (!write(header)) return false;
    return write_dwords(payload);
}

std::vector<Pm4Packet> GpuRingBuffer::parse_pending_packets() {
    std::vector<Pm4Packet> packets;

    while (available_dwords() > 0) {
        const u32 header_dword = m_buffer[m_rptr];
        Pm4Header header = decode_pm4_header(header_dword);
        m_rptr = (m_rptr + 1) % static_cast<u32>(m_buffer.size());

        if (header.type == Pm4PacketType::Type2) {
            // NOP packet has no payload
            Pm4Packet pkt;
            pkt.header = header;
            packets.push_back(std::move(pkt));
            continue;
        }

        // Type-0 and Type-3 packets have header.count payload DWORDs
        if (available_dwords() < header.count) {
            // Incomplete packet in ring buffer, rewind header and break
            m_rptr = (m_rptr == 0) ? static_cast<u32>(m_buffer.size() - 1) : m_rptr - 1;
            break;
        }

        Pm4Packet pkt;
        pkt.header = header;
        pkt.payload.reserve(header.count);

        for (u16 i = 0; i < header.count; ++i) {
            pkt.payload.push_back(m_buffer[m_rptr]);
            m_rptr = (m_rptr + 1) % static_cast<u32>(m_buffer.size());
        }

        packets.push_back(std::move(pkt));
    }

    return packets;
}

void GpuRingBuffer::reset() {
    m_rptr = 0;
    m_wptr = 0;
    std::fill(m_buffer.begin(), m_buffer.end(), 0);
}

usize GpuRingBuffer::available_dwords() const noexcept {
    if (m_wptr >= m_rptr) {
        return m_wptr - m_rptr;
    }
    return (m_buffer.size() - m_rptr) + m_wptr;
}

} // namespace ps4
