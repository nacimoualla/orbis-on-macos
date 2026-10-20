#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "ps4/gpu_ring_buffer.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/pm4.hpp"
#include "ps4/pm4_translator.hpp"

using namespace ps4;

void test_pm4_rejects_short_payload_without_state_change() {
    Pm4Translator translator;
    const u32 payload[] = {PA_CL_VPORT_XSCALE};
    Pm4Packet packet{
        .header = decode_pm4_header(encode_pm4_type3_header(IT_SET_CONTEXT_REG, 1)),
        .payload = {payload, payload + 1},
    };
    const GpuViewport before = translator.viewport();
    assert(!translator.execute_packet(packet));
    assert(translator.malformed_packet_count() == 1);
    assert(translator.viewport().width == before.width);

    const u32 draw_payload[] = {3};
    Pm4Packet draw_packet{
        .header = decode_pm4_header(encode_pm4_type3_header(IT_DRAW_INDEX_2, 1)),
        .payload = {draw_payload, draw_payload + 1},
    };
    assert(!translator.execute_packet(draw_packet));
    assert(translator.malformed_packet_count() == 2);
    assert(translator.indexed_draw_count() == 0);

    const u32 unknown_payload[] = {0};
    Pm4Packet unknown_packet{
        .header = decode_pm4_header(encode_pm4_type3_header(0xFF, 1)),
        .payload = {unknown_payload, unknown_payload + 1},
    };
    assert(!translator.execute_packet(unknown_packet));
    assert(translator.unsupported_packet_count() == 1);
}

int main() {
    std::cout << "[PHASE 10] Starting PS4 PM4 GPU Command Processor & Ring Buffer tests...\n";

    // Test 1: Header encoding & decoding
    {
        // Type 3 NOP packet with 3 payload words
        const ps4::u32 hdr_t3 = ps4::encode_pm4_type3_header(ps4::IT_NOP, 3);
        const auto decoded_t3 = ps4::decode_pm4_header(hdr_t3);
        assert(decoded_t3.type == ps4::Pm4PacketType::Type3);
        assert(decoded_t3.opcode == ps4::IT_NOP);
        assert(decoded_t3.count == 3);

        // Type 0 register write packet
        const ps4::u32 hdr_t0 = ps4::encode_pm4_type0_header(0x100, 4);
        const auto decoded_t0 = ps4::decode_pm4_header(hdr_t0);
        assert(decoded_t0.type == ps4::Pm4PacketType::Type0);
        assert(decoded_t0.count == 4);

        // Type 2 NOP packet
        const ps4::u32 hdr_t2 = ps4::encode_pm4_type2_nop();
        const auto decoded_t2 = ps4::decode_pm4_header(hdr_t2);
        assert(decoded_t2.type == ps4::Pm4PacketType::Type2);
        assert(decoded_t2.count == 0);

        std::cout << "  ✓ Test 1: PM4 Type-0, Type-2, and Type-3 header encoding/decoding verified\n";
    }

    // Test 2: GpuRingBuffer circular write and read pointer management
    {
        ps4::GpuRingBuffer ring(16);
        assert(ring.capacity() == 16);
        assert(ring.available_dwords() == 0);

        // Write 10 DWORDs
        for (ps4::u32 i = 0; i < 10; ++i) {
            assert(ring.write(i * 100));
        }
        assert(ring.available_dwords() == 10);
        assert(ring.write_pointer() == 10);
        assert(ring.read_pointer() == 0);

        // Reset
        ring.reset();
        assert(ring.available_dwords() == 0);
        assert(ring.read_pointer() == 0);
        assert(ring.write_pointer() == 0);

        std::cout << "  ✓ Test 2: Circular GpuRingBuffer write and pointer tracking verified\n";
    }

    // Test 3: Packet parsing from ring buffer
    {
        ps4::GpuRingBuffer ring(64);

        // Write Type-2 NOP
        assert(ring.write(ps4::encode_pm4_type2_nop()));

        // Write Type-3 NOP with 2 payload words
        const std::vector<ps4::u32> payload_nop = {0xCAFEBABE, 0xDEADBEEF};
        assert(ring.write_type3_packet(ps4::IT_NOP, payload_nop));

        // Parse packets
        auto packets = ring.parse_pending_packets();
        assert(packets.size() == 2);
        assert(packets[0].header.type == ps4::Pm4PacketType::Type2);
        assert(packets[1].header.type == ps4::Pm4PacketType::Type3);
        assert(packets[1].header.opcode == ps4::IT_NOP);
        assert(packets[1].payload.size() == 2);
        assert(packets[1].payload[0] == 0xCAFEBABE);
        assert(packets[1].payload[1] == 0xDEADBEEF);
        assert(ring.available_dwords() == 0);

        std::cout << "  ✓ Test 3: Stream of PM4 packets parsed from ring buffer correctly\n";
    }

    // Test 4: Viewport and Scissor context register translation
    {
        ps4::Pm4Translator translator;

        // Construct IT_SET_CONTEXT_REG for viewport (1280x720)
        // xscale = width / 2.0 = 640.0f
        // yscale = height / 2.0 = 360.0f
        const ps4::f32 xscale = 640.0f;
        const ps4::f32 xoffset = 640.0f;
        const ps4::f32 yscale = 360.0f;
        const ps4::f32 yoffset = 360.0f;

        ps4::u32 dw_xscale = 0, dw_xoffset = 0, dw_yscale = 0, dw_yoffset = 0;
        std::memcpy(&dw_xscale, &xscale, sizeof(ps4::f32));
        std::memcpy(&dw_xoffset, &xoffset, sizeof(ps4::f32));
        std::memcpy(&dw_yscale, &yscale, sizeof(ps4::f32));
        std::memcpy(&dw_yoffset, &yoffset, sizeof(ps4::f32));

        ps4::Pm4Packet vp_pkt;
        vp_pkt.header.type = ps4::Pm4PacketType::Type3;
        vp_pkt.header.opcode = ps4::IT_SET_CONTEXT_REG;
        vp_pkt.header.count = 5;
        vp_pkt.payload = {
            ps4::PA_CL_VPORT_XSCALE, // base register offset
            dw_xscale,
            dw_xoffset,
            dw_yscale,
            dw_yoffset
        };

        assert(translator.execute_packet(vp_pkt));
        assert(std::abs(translator.viewport().width - 1280.0f) < 0.001f);
        assert(std::abs(translator.viewport().height - 720.0f) < 0.001f);
        assert(std::abs(translator.viewport().x - 640.0f) < 0.001f);
        assert(std::abs(translator.viewport().y - 360.0f) < 0.001f);

        // Construct IT_SET_CONTEXT_REG for scissor (0, 0 to 1280, 720)
        ps4::Pm4Packet sc_pkt;
        sc_pkt.header.type = ps4::Pm4PacketType::Type3;
        sc_pkt.header.opcode = ps4::IT_SET_CONTEXT_REG;
        sc_pkt.header.count = 3;
        sc_pkt.payload = {
            ps4::PA_SC_SCREEN_SCISSOR_TL,
            0x00000000,              // TL: (0, 0)
            (720 << 16) | 1280       // BR: (1280, 720)
        };

        assert(translator.execute_packet(sc_pkt));
        assert(translator.scissor().x == 0);
        assert(translator.scissor().y == 0);
        assert(translator.scissor().width == 1280);
        assert(translator.scissor().height == 720);

        std::cout << "  ✓ Test 4: Viewport and Scissor context registers translated accurately\n";
    }

    // Test 5: Clear color and Draw packet translation
    {
        ps4::MetalBackend metal;
        assert(metal.initialize());
        assert(metal.create_framebuffer(1920, 1080));

        ps4::Pm4Translator translator(&metal);
        assert(translator.begin_frame());

        // Set clear color to Red (255, 0, 0, 255)
        // Word 0: R (0xFF) | G (0x00 << 8) = 0x00FF
        // Word 1: B (0x00) | A (0xFF << 8) = 0xFF00
        ps4::Pm4Packet clr_pkt;
        clr_pkt.header.type = ps4::Pm4PacketType::Type3;
        clr_pkt.header.opcode = ps4::IT_SET_CONTEXT_REG;
        clr_pkt.header.count = 3;
        clr_pkt.payload = {
            ps4::CB_COLOR_CLEAR_WORD0,
            0x00FF,
            0xFF00
        };

        assert(translator.execute_packet(clr_pkt));
        assert(std::abs(translator.clear_color().r - 1.0f) < 0.01f);
        assert(std::abs(translator.clear_color().g - 0.0f) < 0.01f);
        assert(std::abs(translator.clear_color().b - 0.0f) < 0.01f);
        assert(std::abs(translator.clear_color().a - 1.0f) < 0.01f);

        // Submit IT_DRAW_INDEX_AUTO (Draw 3 vertices)
        ps4::Pm4Packet draw_pkt;
        draw_pkt.header.type = ps4::Pm4PacketType::Type3;
        draw_pkt.header.opcode = ps4::IT_DRAW_INDEX_AUTO;
        draw_pkt.header.count = 2;
        draw_pkt.payload = {
            3, // vertex count
            static_cast<ps4::u32>(ps4::PrimitiveType::TriangleList)
        };

        assert(translator.execute_packet(draw_pkt));
        assert(translator.draw_count() == 1);
        assert(translator.submit_frame(true));

        // Verify pixel readback from Metal framebuffer (should be cleared with Red)
        ps4::u8 r = 0, g = 0, b = 0, a = 0;
        assert(metal.read_pixel(100, 100, r, g, b, a));
        assert(r == 255);
        assert(g == 0);
        assert(b == 0);

        std::cout << "  ✓ Test 5: Clear color and Draw packet translated directly to Metal framebuffer\n";
    }

    // Test 6: End-to-end GpuRingBuffer batch submission
    {
        ps4::MetalBackend metal;
        assert(metal.initialize());
        assert(metal.create_framebuffer(1920, 1080));

        ps4::Pm4Translator translator(&metal);
        ps4::GpuRingBuffer ring(256);
        assert(translator.begin_frame());

        // Push:
        // 1. CLEAR_STATE (takes 1 dummy DWORD in PM4)
        assert(ring.write_type3_packet(ps4::IT_CLEAR_STATE, {0}));

        // 2. SET_CONTEXT_REG (Clear color = Blue)
        assert(ring.write_type3_packet(ps4::IT_SET_CONTEXT_REG, {
            ps4::CB_COLOR_CLEAR_WORD0,
            0x0000, // R=0, G=0
            0xFF00 | 0xFF // B=255, A=255
        }));

        // 3. DRAW_INDEX_AUTO
        assert(ring.write_type3_packet(ps4::IT_DRAW_INDEX_AUTO, {
            6, static_cast<ps4::u32>(ps4::PrimitiveType::TriangleList)
        }));

        // 4. EVENT_WRITE (Flush)
        assert(ring.write_type3_packet(ps4::IT_EVENT_WRITE, {0x01}));

        const ps4::usize executed = translator.execute_ring_buffer(ring);
        assert(executed == 4);
        assert(translator.draw_count() == 1);
        assert(translator.submit_frame(true));
        assert(ring.available_dwords() == 0);

        // Read pixel to verify Blue clear
        ps4::u8 r = 0, g = 0, b = 0, a = 0;
        assert(metal.read_pixel(50, 50, r, g, b, a));
        assert(r == 0);
        assert(b == 255);

        std::cout << "  ✓ Test 6: Full PM4 batch executed from ring buffer and verified on GPU\n";
    }

    test_pm4_rejects_short_payload_without_state_change();
    std::cout << "[PHASE 10] PASS: All PS4 PM4 GPU Command Processor & Ring Buffer tests passed!\n";
    return 0;
}
