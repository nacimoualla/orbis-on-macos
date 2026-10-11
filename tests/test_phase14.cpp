#include "ps4/gpu_ring_buffer.hpp"
#include "ps4/memory.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/pm4.hpp"
#include "ps4/pm4_translator.hpp"
#include <cassert>
#include <iostream>

using namespace ps4;

void test_pm4_draw_index_2_packet_decoding() {
    std::cout << "[Test 14.1] Testing PM4 IT_DRAW_INDEX_2 packet encoding and decoding...\n";

    // Build IT_DRAW_INDEX_2 packet
    // payload: [max_size, base_lo, base_hi, count, initiator]
    const u32 payload[] = {
        3,                    // max_size / index count
        0x1000,               // index_base_lo
        0x0000,               // index_base_hi
        3,                    // index count
        static_cast<u32>(PrimitiveType::TriangleList) // topology
    };

    const u32 header_word = encode_pm4_type3_header(IT_DRAW_INDEX_2, 5);
    const Pm4Header hdr = decode_pm4_header(header_word);

    assert(hdr.type == Pm4PacketType::Type3);
    assert(hdr.opcode == IT_DRAW_INDEX_2);
    assert(hdr.count == 5);

    Pm4Translator translator;
    Pm4Packet packet{
        .header = hdr,
        .payload = {payload, payload + 5}
    };

    assert(translator.execute_packet(packet));
    assert(translator.packets_processed() == 1);
    assert(translator.indexed_draw_count() == 1);

    std::cout << "  -> IT_DRAW_INDEX_2 packet decoding passed!\n";
}

void test_vertex_buffer_binding_via_sh_reg() {
    std::cout << "[Test 14.2] Testing vertex buffer binding via PM4 IT_SET_SH_REG...\n";

    Pm4Translator translator;

    // payload: [base_reg, vaddr_lo, vaddr_hi, size, stride]
    const u32 sh_payload[] = {
        SPI_SHADER_USER_DATA_VS_0, // base register (0x0200)
        0x50000,                   // vaddr_lo
        0x00001,                   // vaddr_hi -> 0x100050000
        1024,                      // buffer size
        sizeof(Vertex3D)           // stride
    };

    const u32 hdr_word = encode_pm4_type3_header(IT_SET_SH_REG, 5);
    Pm4Packet pkt{
        .header = decode_pm4_header(hdr_word),
        .payload = {sh_payload, sh_payload + 5}
    };

    assert(translator.execute_packet(pkt));
    assert(translator.vertex_buffer_address() == 0x100050000ULL);
    assert(translator.vertex_buffer_size() == 1024);

    std::cout << "  -> Vertex buffer binding via IT_SET_SH_REG passed!\n";
}

void test_metal_3d_indexed_triangle_rendering() {
    std::cout << "[Test 14.3] Testing 3D colored triangle rendering with Apple Silicon Metal 3...\n";

    MetalBackend metal;
    assert(metal.initialize());
    assert(metal.create_framebuffer(1920, 1080));

    // Clear to pure black
    assert(metal.clear_screen(ColorRGBA{0.0f, 0.0f, 0.0f, 1.0f}));

    // Check clear color at center
    u8 r_bg = 0, g_bg = 0, b_bg = 0, a_bg = 0;
    assert(metal.read_pixel(960, 540, r_bg, g_bg, b_bg, a_bg));
    assert(r_bg == 0 && g_bg == 0 && b_bg == 0);

    // Initialize 8GB unified memory
    MemoryManager memory;
    assert(memory.initialize());

    // Allocate vertices and index buffers in guest virtual memory
    const GuestAddr vtx_addr = 0x100000;
    const GuestAddr idx_addr = 0x101000;

    // 3D Triangle in Normalized Device Coordinates (NDC)
    const Vertex3D triangle_vertices[3] = {
        // Vertex 0: Bottom-left, pure Red
        {{-0.6f, -0.6f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
        // Vertex 1: Top-center, pure Green
        {{ 0.0f,  0.6f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        // Vertex 2: Bottom-right, pure Blue
        {{ 0.6f, -0.6f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}
    };

    const u16 triangle_indices[3] = {0, 1, 2};

    memory.write_bytes(vtx_addr, triangle_vertices, sizeof(triangle_vertices));
    memory.write_bytes(idx_addr, triangle_indices, sizeof(triangle_indices));

    // Setup PM4 command translator
    Pm4Translator translator(&metal);
    translator.set_memory_manager(&memory);
    translator.bind_vertex_buffer(vtx_addr, sizeof(triangle_vertices));

    // Setup GPU ring buffer and submit draw commands
    GpuRingBuffer ring_buffer(64 * 1024);

    // 1. Set Viewport context registers (1920x1080)
    // Scale: 960.0f, Offset: 960.0f, YScale: -540.0f, YOffset: 540.0f
    f32 xscale = 960.0f, xoffset = 960.0f, yscale = -540.0f, yoffset = 540.0f;
    u32 u_xscale = 0, u_xoffset = 0, u_yscale = 0, u_yoffset = 0;
    std::memcpy(&u_xscale, &xscale, sizeof(f32));
    std::memcpy(&u_xoffset, &xoffset, sizeof(f32));
    std::memcpy(&u_yscale, &yscale, sizeof(f32));
    std::memcpy(&u_yoffset, &yoffset, sizeof(f32));

    ring_buffer.write_type3_packet(IT_SET_CONTEXT_REG, {
        PA_CL_VPORT_XSCALE,
        u_xscale,
        u_xoffset,
        u_yscale,
        u_yoffset
    });

    // 2. Set Vertex Buffer via IT_SET_SH_REG
    ring_buffer.write_type3_packet(IT_SET_SH_REG, {
        SPI_SHADER_USER_DATA_VS_0,
        static_cast<u32>(vtx_addr & 0xFFFFFFFF),
        static_cast<u32>(vtx_addr >> 32),
        sizeof(triangle_vertices),
        sizeof(Vertex3D)
    });

    // 3. Submit IT_DRAW_INDEX_2
    ring_buffer.write_type3_packet(IT_DRAW_INDEX_2, {
        3,                                          // max_size
        static_cast<u32>(idx_addr & 0xFFFFFFFF),   // index base lo
        static_cast<u32>(idx_addr >> 32),          // index base hi
        3,                                          // count
        static_cast<u32>(PrimitiveType::TriangleList) // initiator
    });

    // Execute ring buffer
    const usize processed = translator.execute_ring_buffer(ring_buffer);
    assert(processed == 3);
    assert(translator.draw_count() == 1);
    assert(translator.indexed_draw_count() == 1);

    // Read back pixel color at center of the triangle (960, 540)
    u8 r = 0, g = 0, b = 0, a = 0;
    assert(metal.read_pixel(960, 540, r, g, b, a));

    // The center of an RGB triangle must have non-zero rasterized color!
    std::cout << "  -> Center pixel rasterized color: R=" << static_cast<int>(r)
              << " G=" << static_cast<int>(g)
              << " B=" << static_cast<int>(b)
              << " A=" << static_cast<int>(a) << "\n";
    assert(r > 0 || g > 0 || b > 0);
    assert(a == 255);

    // Pixels outside the triangle (e.g. at corner 50, 50) remain black
    u8 r_out = 0, g_out = 0, b_out = 0, a_out = 0;
    assert(metal.read_pixel(50, 50, r_out, g_out, b_out, a_out));
    assert(r_out == 0 && g_out == 0 && b_out == 0);

    std::cout << "  -> 3D indexed mesh rasterization on Metal 3 passed!\n";
}

int main() {
    std::cout << "=== Phase 14: 3D Graphics Mesh Pipeline & Sample Demo Tests ===\n";

    test_pm4_draw_index_2_packet_decoding();
    test_vertex_buffer_binding_via_sh_reg();
    test_metal_3d_indexed_triangle_rendering();

    std::cout << "=== All Phase 14 tests passed successfully! ===\n";
    return 0;
}
