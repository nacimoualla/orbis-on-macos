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

void test_pm4_index_base_and_dma_data() {
    std::cout << "[Test 14.4] Testing PM4 IT_INDEX_BASE and IT_DMA_DATA (immediate fill and memcopy)...\n";

    MemoryManager memory;
    assert(memory.initialize());

    Pm4Translator translator;
    translator.set_memory_manager(&memory);

    // 1. Test IT_INDEX_BASE
    const u32 idx_base_payload[] = {
        0x5000, // base lo (aligned)
        0x0002  // base hi -> 0x200005000
    };
    const u32 idx_hdr = encode_pm4_type3_header(IT_INDEX_BASE, 2);
    Pm4Packet idx_pkt{
        .header = decode_pm4_header(idx_hdr),
        .payload = {idx_base_payload, idx_base_payload + 2}
    };
    assert(translator.execute_packet(idx_pkt));
    assert(translator.index_base_address() == 0x200005000ULL);

    // 2. Test IT_DMA_DATA with immediate data fill (src_sel = 2)
    const GuestAddr fill_dst = 0x200000;
    const u32 fill_pattern = 0xA5A5A5A5U;
    const usize fill_bytes = 64;

    // Control word: src_sel = 2 (bits 30:29 = 0x40000000), dst_sel = 0
    const u32 ctl_immediate = (2U << 29);
    const u32 dma_fill_payload[] = {
        ctl_immediate,
        fill_pattern,                              // src_lo = constant pattern
        0,                                         // src_hi
        static_cast<u32>(fill_dst & 0xFFFFFFFF),   // dst_lo
        static_cast<u32>(fill_dst >> 32),          // dst_hi
        static_cast<u32>(fill_bytes)               // byte count
    };
    const u32 dma_fill_hdr = encode_pm4_type3_header(IT_DMA_DATA, 6);
    Pm4Packet dma_fill_pkt{
        .header = decode_pm4_header(dma_fill_hdr),
        .payload = {dma_fill_payload, dma_fill_payload + 6}
    };
    assert(translator.execute_packet(dma_fill_pkt));
    assert(translator.last_dma_dst_address() == fill_dst);
    assert(translator.last_dma_size() == fill_bytes);

    // Verify memory was filled with pattern
    const u32* filled_mem = memory.to_host_ptr<u32>(fill_dst);
    for (usize i = 0; i < fill_bytes / sizeof(u32); ++i) {
        assert(filled_mem[i] == fill_pattern);
    }

    // 3. Test IT_DMA_DATA with memory copy (src_sel = 0)
    const GuestAddr copy_src = 0x300000;
    const GuestAddr copy_dst = 0x400000;
    const u32 source_data[4] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
    memory.write_bytes(copy_src, source_data, sizeof(source_data));

    const u32 ctl_memcpy = (0U << 29);
    const u32 dma_copy_payload[] = {
        ctl_memcpy,
        static_cast<u32>(copy_src & 0xFFFFFFFF),
        static_cast<u32>(copy_src >> 32),
        static_cast<u32>(copy_dst & 0xFFFFFFFF),
        static_cast<u32>(copy_dst >> 32),
        static_cast<u32>(sizeof(source_data))
    };
    const u32 dma_copy_hdr = encode_pm4_type3_header(IT_DMA_DATA, 6);
    Pm4Packet dma_copy_pkt{
        .header = decode_pm4_header(dma_copy_hdr),
        .payload = {dma_copy_payload, dma_copy_payload + 6}
    };
    assert(translator.execute_packet(dma_copy_pkt));
    assert(translator.last_dma_dst_address() == copy_dst);

    const u32* copied_mem = memory.to_host_ptr<u32>(copy_dst);
    for (usize i = 0; i < 4; ++i) {
        assert(copied_mem[i] == source_data[i]);
    }

    std::cout << "  -> IT_INDEX_BASE and IT_DMA_DATA tests passed!\n";
}

void test_pm4_type0_index_configuration() {
    std::cout << "[Test 14.5] Testing Type-0 register writes for index type & max size...\n";

    Pm4Translator translator;

    // Write to VGT_DMA_INDEX_TYPE_REG (0x029F) with 1 (UInt32)
    const u32 type_val = 1;
    const u32 type0_hdr1 = encode_pm4_type0_header(VGT_DMA_INDEX_TYPE_REG, 1);
    Pm4Packet pkt1{
        .header = decode_pm4_header(type0_hdr1),
        .payload = {type_val}
    };
    assert(translator.execute_packet(pkt1));
    assert(translator.is_32bit_index() == true);

    // Write to VGT_DMA_MAX_SIZE_REG (0x029E) with 32768
    const u32 max_val = 32768;
    const u32 type0_hdr2 = encode_pm4_type0_header(VGT_DMA_MAX_SIZE_REG, 1);
    Pm4Packet pkt2{
        .header = decode_pm4_header(type0_hdr2),
        .payload = {max_val}
    };
    assert(translator.execute_packet(pkt2));
    assert(translator.index_buffer_size() == 32768);

    std::cout << "  -> Type-0 index configuration passed!\n";
}

void test_pm4_draw_index_indirect_execution() {
    std::cout << "[Test 14.6] Testing PM4 IT_DRAW_INDEX_INDIRECT execution with Metal 3...\n";

    MetalBackend metal;
    assert(metal.initialize());
    assert(metal.create_framebuffer(1920, 1080));
    assert(metal.clear_screen(ColorRGBA{0.0f, 0.0f, 0.0f, 1.0f}));

    MemoryManager memory;
    assert(memory.initialize());

    Pm4Translator translator(&metal);
    translator.set_memory_manager(&memory);

    const GuestAddr vtx_addr = 0x500000;
    const GuestAddr idx_addr = 0x510000;
    const GuestAddr args_addr = 0x520000;

    const Vertex3D vertices[3] = {
        {{-0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}}, // Green
        {{ 0.0f,  0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{ 0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}}
    };
    const u16 indices[3] = {0, 1, 2};

    // SceGnmDrawIndexIndirectArgs: 20 bytes
    struct SceGnmDrawIndexIndirectArgs {
        u32 index_count;
        u32 instance_count;
        u32 first_index;
        i32 vertex_offset;
        u32 first_instance;
    } draw_args = {
        .index_count = 3,
        .instance_count = 1,
        .first_index = 0,
        .vertex_offset = 0,
        .first_instance = 0
    };

    memory.write_bytes(vtx_addr, vertices, sizeof(vertices));
    memory.write_bytes(idx_addr, indices, sizeof(indices));
    memory.write_bytes(args_addr, &draw_args, sizeof(draw_args));

    // Bind vertex buffer
    translator.bind_vertex_buffer(vtx_addr, sizeof(vertices));

    // 1. IT_INDEX_BASE
    const u32 base_payload[] = {
        static_cast<u32>(idx_addr & 0xFFFFFFFF),
        static_cast<u32>(idx_addr >> 32)
    };
    Pm4Packet base_pkt{
        .header = decode_pm4_header(encode_pm4_type3_header(IT_INDEX_BASE, 2)),
        .payload = {base_payload, base_payload + 2}
    };
    assert(translator.execute_packet(base_pkt));
    assert(translator.index_base_address() == idx_addr);

    // 2. Set indirect base address
    translator.set_indirect_base_address(args_addr);

    // 3. IT_DRAW_INDEX_INDIRECT
    // payload: [data_offset, base_vtx_loc/start_indx_loc, draw_initiator]
    const u32 draw_payload[] = {
        0,                                            // data_offset = 0
        0,                                            // SGPR offsets
        static_cast<u32>(PrimitiveType::TriangleList) // topology
    };
    Pm4Packet draw_pkt{
        .header = decode_pm4_header(encode_pm4_type3_header(IT_DRAW_INDEX_INDIRECT, 3)),
        .payload = {draw_payload, draw_payload + 3}
    };
    assert(translator.execute_packet(draw_pkt));
    assert(translator.draw_count() == 1);
    assert(translator.indexed_draw_count() == 1);

    // Verify rasterization
    u8 r = 0, g = 0, b = 0, a = 0;
    assert(metal.read_pixel(960, 540, r, g, b, a));
    std::cout << "  -> IT_DRAW_INDEX_INDIRECT rasterized pixel at (960, 540): G=" << static_cast<int>(g) << "\n";
    assert(g > 0);

    // 4. Test execute_command_buffer
    const GuestAddr dcb_addr = 0x600000;
    std::vector<u32> dcb_data;
    // NOP packet
    dcb_data.push_back(encode_pm4_type3_header(IT_NOP, 1));
    dcb_data.push_back(0x68752000); // GNM tag
    // DRAW_INDEX_INDIRECT packet
    dcb_data.push_back(encode_pm4_type3_header(IT_DRAW_INDEX_INDIRECT, 3));
    dcb_data.push_back(0); // data_offset
    dcb_data.push_back(0);
    dcb_data.push_back(static_cast<u32>(PrimitiveType::TriangleList));

    memory.write_bytes(dcb_addr, dcb_data.data(), dcb_data.size() * sizeof(u32));
    const usize pkts = translator.execute_command_buffer(dcb_addr, dcb_data.size() * sizeof(u32));
    assert(pkts == 2);
    assert(translator.draw_count() == 2);

    std::cout << "  -> IT_DRAW_INDEX_INDIRECT and execute_command_buffer passed!\n";
}

void test_pm4_draw_index_auto_and_procedural_fallback() {
    std::cout << "[Test 14.7] Testing PM4 IT_DRAW_INDEX_AUTO and procedural mesh fallback...\n";

    MetalBackend metal;
    assert(metal.initialize());
    assert(metal.create_framebuffer(1920, 1080));
    assert(metal.clear_screen(ColorRGBA{0.0f, 0.0f, 0.0f, 1.0f}));

    MemoryManager memory;
    assert(memory.initialize());

    Pm4Translator translator(&metal);
    translator.set_memory_manager(&memory);

    // 1. Test IT_DRAW_INDEX_AUTO with count=3 (fullscreen background triangle)
    const u32 auto_payload_3[] = {3, 2}; // count=3, initiator=2
    Pm4Packet auto_pkt_3{
        .header = decode_pm4_header(encode_pm4_type3_header(IT_DRAW_INDEX_AUTO, 2)),
        .payload = {auto_payload_3, auto_payload_3 + 2}
    };
    assert(translator.execute_packet(auto_pkt_3));
    assert(translator.draw_count() == 1);

    // Center pixel should be colored by the procedural sky background
    u8 r = 0, g = 0, b = 0, a = 0;
    assert(metal.read_pixel(960, 540, r, g, b, a));
    assert(r > 0 || g > 0 || b > 0);
    assert(a > 0);

    // 2. Test IT_DRAW_INDEX_AUTO with count=4 (procedural quad)
    const u32 auto_payload_4[] = {4, 2}; // count=4, initiator=2
    Pm4Packet auto_pkt_4{
        .header = decode_pm4_header(encode_pm4_type3_header(IT_DRAW_INDEX_AUTO, 2)),
        .payload = {auto_payload_4, auto_payload_4 + 2}
    };
    assert(translator.execute_packet(auto_pkt_4));
    assert(translator.draw_count() == 2);

    // 3. Test procedural indexed quad fallback when vertex_buffer_addr is 0
    const GuestAddr idx_addr = 0x700000;
    const u16 quad_indices[6] = {0, 2, 1, 1, 2, 3};
    memory.write_bytes(idx_addr, quad_indices, sizeof(quad_indices));

    translator.bind_vertex_buffer(0, 0); // No vertex buffer bound!
    assert(translator.execute_indexed_draw(idx_addr, 6, PrimitiveType::TriangleList));
    assert(translator.draw_count() == 3);

    std::cout << "  -> IT_DRAW_INDEX_AUTO and procedural fallback passed!\n";
}

int main() {
    std::cout << "=== Phase 14: 3D Graphics Mesh Pipeline & Sample Demo Tests ===\n";

    test_pm4_draw_index_2_packet_decoding();
    test_vertex_buffer_binding_via_sh_reg();
    test_metal_3d_indexed_triangle_rendering();
    test_pm4_index_base_and_dma_data();
    test_pm4_type0_index_configuration();
    test_pm4_draw_index_indirect_execution();
    test_pm4_draw_index_auto_and_procedural_fallback();

    std::cout << "=== All Phase 14 tests passed successfully! ===\n";
    return 0;
}
