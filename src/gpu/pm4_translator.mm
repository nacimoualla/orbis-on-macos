#import <Metal/Metal.h>
#include "ps4/memory.hpp"
#include "ps4/pm4_translator.hpp"
#include <cstring>
#include <iostream>

namespace ps4 {

struct Pm4Translator::Impl {
    MetalBackend* metal{nullptr};
    MemoryManager* memory{nullptr};
    GpuViewport viewport{};
    GpuScissor scissor{};
    ColorRGBA clear_color{0.0f, 0.0f, 0.0f, 1.0f};

    GuestAddr vertex_buffer_addr{0};
    usize vertex_buffer_size{0};
    usize vertex_stride{sizeof(Vertex3D)};

    usize packets_processed{0};
    usize draw_count{0};
    usize indexed_draw_count{0};
    usize context_reg_writes{0};

    id<MTLRenderPipelineState> mesh_pipeline_state{nil};

    void init_mesh_pipeline();
    bool render_indexed_mesh(GuestAddr idx_addr, u32 idx_count, PrimitiveType prim_type);
};

void Pm4Translator::Impl::init_mesh_pipeline() {
    if (!metal || !metal->is_initialized()) return;
    if (mesh_pipeline_state) return;

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)metal->raw_device();
        if (!device) return;

        NSString* shader_source = @""
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "struct Vertex3D {\n"
            "    packed_float3 position;\n"
            "    packed_float4 color;\n"
            "};\n"
            "struct RasterizerData {\n"
            "    float4 position [[position]];\n"
            "    float4 color;\n"
            "};\n"
            "vertex RasterizerData vs_mesh(uint vid [[vertex_id]],\n"
            "                              constant Vertex3D* vertices [[buffer(0)]]) {\n"
            "    RasterizerData out;\n"
            "    out.position = float4(float3(vertices[vid].position), 1.0);\n"
            "    out.color = float4(vertices[vid].color);\n"
            "    return out;\n"
            "}\n"
            "fragment float4 fs_mesh(RasterizerData in [[stage_in]]) {\n"
            "    return in.color;\n"
            "}\n";

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shader_source options:nil error:&error];
        if (!library) {
            std::cerr << "[PM4] Failed to compile MSL 3D mesh shader: "
                      << (error ? [[error localizedDescription] UTF8String] : "unknown error") << "\n";
            return;
        }

        id<MTLFunction> vs = [library newFunctionWithName:@"vs_mesh"];
        id<MTLFunction> fs = [library newFunctionWithName:@"fs_mesh"];

        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vs;
        desc.fragmentFunction = fs;
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

        mesh_pipeline_state = [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (!mesh_pipeline_state) {
            std::cerr << "[PM4] Failed to create 3D mesh render pipeline state: "
                      << (error ? [[error localizedDescription] UTF8String] : "unknown error") << "\n";
        }
    }
}

bool Pm4Translator::Impl::render_indexed_mesh(GuestAddr idx_addr, u32 idx_count, PrimitiveType prim_type) {
    if (idx_count == 0) return true;

    if (!metal || !metal->is_initialized()) {
        draw_count++;
        indexed_draw_count++;
        return true;
    }

    @autoreleasepool {
        init_mesh_pipeline();
        if (!mesh_pipeline_state) return false;

        id<MTLDevice> device = (__bridge id<MTLDevice>)metal->raw_device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal->raw_command_queue();
        id<MTLTexture> framebuffer = (__bridge id<MTLTexture>)metal->raw_framebuffer();
        if (!device || !queue || !framebuffer) return false;

        const usize required_idx_size = idx_count * sizeof(u16);
        if (!memory || !memory->is_valid_address(idx_addr, required_idx_size)) {
            return false;
        }
        const u16* idx_ptr = memory->to_host_ptr<u16>(idx_addr);
        if (!idx_ptr) return false;

        // Determine actual vertex count from max index if vertex_buffer_size was not explicitly set
        usize required_vtx_size = vertex_buffer_size;
        if (required_vtx_size == 0) {
            u16 max_idx = 0;
            for (u32 i = 0; i < idx_count; ++i) {
                if (idx_ptr[i] > max_idx) max_idx = idx_ptr[i];
            }
            const usize stride = vertex_stride > 0 ? vertex_stride : sizeof(Vertex3D);
            required_vtx_size = static_cast<usize>(max_idx + 1) * stride;
        }

        if (!memory->is_valid_address(vertex_buffer_addr, required_vtx_size)) {
            return false;
        }
        const void* vtx_ptr = memory->to_host_ptr<void>(vertex_buffer_addr);
        if (!vtx_ptr) return false;

        id<MTLBuffer> vtx_buf = [device newBufferWithBytes:vtx_ptr length:required_vtx_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> idx_buf = [device newBufferWithBytes:idx_ptr length:required_idx_size options:MTLResourceStorageModeShared];

        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_desc.colorAttachments[0].texture = framebuffer;
        pass_desc.colorAttachments[0].loadAction = MTLLoadActionLoad;
        pass_desc.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass_desc];

        [enc setRenderPipelineState:mesh_pipeline_state];
        [enc setCullMode:MTLCullModeNone];

        const u32 fb_w = metal->width() > 0 ? metal->width() : 1920;
        const u32 fb_h = metal->height() > 0 ? metal->height() : 1080;

        f32 origin_x = (viewport.width > 0.0f && viewport.x >= (viewport.width / 2.0f))
                           ? (viewport.x - (viewport.width / 2.0f))
                           : viewport.x;
        f32 origin_y = (viewport.height > 0.0f && viewport.y >= (viewport.height / 2.0f))
                           ? (viewport.y - (viewport.height / 2.0f))
                           : viewport.y;

        [enc setViewport:MTLViewport{
            .originX = origin_x,
            .originY = origin_y,
            .width = viewport.width > 0.0f ? std::min<f32>(viewport.width, static_cast<f32>(fb_w)) : static_cast<f32>(fb_w),
            .height = viewport.height > 0.0f ? std::min<f32>(viewport.height, static_cast<f32>(fb_h)) : static_cast<f32>(fb_h),
            .znear = viewport.min_depth,
            .zfar = viewport.max_depth
        }];

        const u32 sc_x = std::min<u32>(scissor.x, fb_w - 1);
        const u32 sc_y = std::min<u32>(scissor.y, fb_h - 1);
        const u32 sc_w = (scissor.width > 0 && sc_x + scissor.width <= fb_w) ? scissor.width : (fb_w - sc_x);
        const u32 sc_h = (scissor.height > 0 && sc_y + scissor.height <= fb_h) ? scissor.height : (fb_h - sc_y);

        [enc setScissorRect:MTLScissorRect{
            .x = sc_x,
            .y = sc_y,
            .width = sc_w,
            .height = sc_h
        }];

        [enc setVertexBuffer:vtx_buf offset:0 atIndex:0];

        MTLPrimitiveType mtl_prim = MTLPrimitiveTypeTriangle;
        if (prim_type == PrimitiveType::TriangleStrip) {
            mtl_prim = MTLPrimitiveTypeTriangleStrip;
        } else if (prim_type == PrimitiveType::LineList) {
            mtl_prim = MTLPrimitiveTypeLine;
        } else if (prim_type == PrimitiveType::PointList) {
            mtl_prim = MTLPrimitiveTypePoint;
        }

        [enc drawIndexedPrimitives:mtl_prim
                        indexCount:idx_count
                         indexType:MTLIndexTypeUInt16
                       indexBuffer:idx_buf
                 indexBufferOffset:0];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        draw_count++;
        indexed_draw_count++;
        return true;
    }
}

Pm4Translator::Pm4Translator(MetalBackend* metal)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->metal = metal;
    reset_state();
}

Pm4Translator::~Pm4Translator() = default;

Pm4Translator::Pm4Translator(Pm4Translator&&) noexcept = default;
Pm4Translator& Pm4Translator::operator=(Pm4Translator&&) noexcept = default;

void Pm4Translator::reset_state() {
    if (!m_impl) return;
    m_impl->viewport = GpuViewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = 1920.0f,
        .height = 1080.0f,
        .min_depth = 0.0f,
        .max_depth = 1.0f
    };
    m_impl->scissor = GpuScissor{
        .x = 0,
        .y = 0,
        .width = 1920,
        .height = 1080
    };
    m_impl->clear_color = ColorRGBA{0.0f, 0.0f, 0.0f, 1.0f};
    m_impl->vertex_buffer_addr = 0;
    m_impl->vertex_buffer_size = 0;
}

void Pm4Translator::set_memory_manager(MemoryManager* memory) noexcept {
    if (m_impl) {
        m_impl->memory = memory;
    }
}

void Pm4Translator::bind_vertex_buffer(GuestAddr vaddr, usize size, usize stride) {
    if (m_impl) {
        m_impl->vertex_buffer_addr = vaddr;
        m_impl->vertex_buffer_size = size;
        m_impl->vertex_stride = stride;
    }
}

GuestAddr Pm4Translator::vertex_buffer_address() const noexcept {
    return m_impl ? m_impl->vertex_buffer_addr : 0;
}

usize Pm4Translator::vertex_buffer_size() const noexcept {
    return m_impl ? m_impl->vertex_buffer_size : 0;
}

bool Pm4Translator::execute_indexed_draw(GuestAddr index_buffer_addr, u32 index_count, PrimitiveType prim_type) {
    return m_impl ? m_impl->render_indexed_mesh(index_buffer_addr, index_count, prim_type) : false;
}

bool Pm4Translator::execute_packet(const Pm4Packet& packet) {
    if (!m_impl) return false;
    m_impl->packets_processed++;

    if (packet.header.type == Pm4PacketType::Type2) {
        // NOP packet
        return true;
    }

    if (packet.header.type == Pm4PacketType::Type0) {
        // Type 0: register writes
        m_impl->context_reg_writes += packet.payload.size();
        return true;
    }

    if (packet.header.type == Pm4PacketType::Type3) {
        switch (packet.header.opcode) {
            case IT_NOP:
                return true;

            case IT_CLEAR_STATE:
                reset_state();
                return true;

            case IT_SET_CONTEXT_REG: {
                if (packet.payload.empty()) return false;
                const u16 base_reg = static_cast<u16>(packet.payload[0] & 0xFFFF);
                for (usize i = 1; i < packet.payload.size(); ++i) {
                    const u16 reg = base_reg + static_cast<u16>(i - 1);
                    const u32 val = packet.payload[i];
                    f32 fval = 0.0f;
                    std::memcpy(&fval, &val, sizeof(fval));

                    if (reg == PA_CL_VPORT_XSCALE) {
                        m_impl->viewport.width = std::abs(fval) * 2.0f;
                    } else if (reg == PA_CL_VPORT_XOFFSET) {
                        m_impl->viewport.x = fval;
                    } else if (reg == PA_CL_VPORT_YSCALE) {
                        m_impl->viewport.height = std::abs(fval) * 2.0f;
                    } else if (reg == PA_CL_VPORT_YOFFSET) {
                        m_impl->viewport.y = fval;
                    } else if (reg == PA_SC_SCREEN_SCISSOR_TL) {
                        m_impl->scissor.x = val & 0x7FFF;
                        m_impl->scissor.y = (val >> 16) & 0x7FFF;
                    } else if (reg == PA_SC_SCREEN_SCISSOR_BR) {
                        const u32 br_x = val & 0x7FFF;
                        const u32 br_y = (val >> 16) & 0x7FFF;
                        m_impl->scissor.width = (br_x > m_impl->scissor.x) ? (br_x - m_impl->scissor.x) : 0;
                        m_impl->scissor.height = (br_y > m_impl->scissor.y) ? (br_y - m_impl->scissor.y) : 0;
                    } else if (reg == CB_COLOR_CLEAR_WORD0) {
                        m_impl->clear_color.r = static_cast<f32>(val & 0xFF) / 255.0f;
                        m_impl->clear_color.g = static_cast<f32>((val >> 8) & 0xFF) / 255.0f;
                    } else if (reg == CB_COLOR_CLEAR_WORD1) {
                        m_impl->clear_color.b = static_cast<f32>(val & 0xFF) / 255.0f;
                        m_impl->clear_color.a = static_cast<f32>((val >> 8) & 0xFF) / 255.0f;
                    }
                }
                m_impl->context_reg_writes += (packet.payload.size() - 1);
                return true;
            }

            case IT_SET_SH_REG: {
                if (packet.payload.size() >= 3) {
                    const u16 base_reg = static_cast<u16>(packet.payload[0] & 0xFFFF);
                    if (base_reg == SPI_SHADER_USER_DATA_VS_0) {
                        const u32 vaddr_lo = packet.payload[1];
                        const u32 vaddr_hi = packet.payload[2];
                        const GuestAddr vaddr = (static_cast<u64>(vaddr_hi) << 32) | vaddr_lo;
                        const usize size = packet.payload.size() >= 4 ? packet.payload[3] : 0;
                        const usize stride = packet.payload.size() >= 5 ? packet.payload[4] : sizeof(Vertex3D);
                        bind_vertex_buffer(vaddr, size, stride);
                    }
                }
                return true;
            }

            case IT_DRAW_INDEX_AUTO: {
                m_impl->draw_count++;
                if (m_impl->metal && m_impl->metal->is_initialized()) {
                    @autoreleasepool {
                        m_impl->metal->clear_screen(m_impl->clear_color);
                    }
                }
                return true;
            }

            case IT_DRAW_INDEX_2: {
                if (packet.payload.size() >= 4) {
                    const u32 idx_lo = packet.payload[1];
                    const u32 idx_hi = packet.payload[2];
                    const GuestAddr idx_addr = (static_cast<u64>(idx_hi) << 32) | idx_lo;
                    const u32 count = packet.payload[3];
                    PrimitiveType prim = PrimitiveType::TriangleList;
                    if (packet.payload.size() >= 5) {
                        const u8 prim_val = static_cast<u8>(packet.payload[4] & 0x3F);
                        if (prim_val != 0) {
                            prim = static_cast<PrimitiveType>(prim_val);
                        }
                    }
                    return m_impl->render_indexed_mesh(idx_addr, count, prim);
                }
                return false;
            }

            case IT_WRITE_DATA: {
                if (packet.payload.size() >= 4 && m_impl->memory) {
                    const u32 addr_lo = packet.payload[1];
                    const u32 addr_hi = packet.payload[2];
                    const GuestAddr dst_addr = (static_cast<u64>(addr_hi) << 32) | addr_lo;
                    const usize count = packet.payload.size() - 3;
                    const usize byte_size = count * sizeof(u32);
                    if (dst_addr > 0x1000 && m_impl->memory->is_valid_address(dst_addr, byte_size)) {
                        m_impl->memory->write_bytes(dst_addr, &packet.payload[3], byte_size);
                    }
                }
                return true;
            }

            case IT_WAIT_REG_MEM: {
                return true;
            }

            case IT_EVENT_WRITE:
                return true;

            case IT_RELEASE_MEM: {
                if (packet.payload.size() >= 5 && m_impl->memory) {
                    const u32 addr_lo = packet.payload[2];
                    const u32 addr_hi = packet.payload[3];
                    const GuestAddr label_addr = (static_cast<u64>(addr_hi) << 32) | addr_lo;
                    const u32 data_lo = packet.payload[4];
                    const u32 data_hi = packet.payload.size() >= 6 ? packet.payload[5] : 0;
                    const u64 data_val = (static_cast<u64>(data_hi) << 32) | data_lo;
                    if (label_addr > 0x1000 && m_impl->memory->is_valid_address(label_addr, 4)) {
                        m_impl->memory->write_value(label_addr, data_val);
                    }
                }
                return true;
            }

            case IT_DMA_DATA: {
                if (packet.payload.size() >= 6 && m_impl->memory) {
                    const u32 src_lo = packet.payload[1];
                    const u32 src_hi = packet.payload[2];
                    const u32 dst_lo = packet.payload[3];
                    const u32 dst_hi = packet.payload[4];
                    const GuestAddr src_addr = (static_cast<u64>(src_hi) << 32) | src_lo;
                    const GuestAddr dst_addr = (static_cast<u64>(dst_hi) << 32) | dst_lo;
                    const usize byte_size = packet.payload[5] & 0x1FFFFF;
                    if (byte_size > 0 && m_impl->memory->is_valid_address(src_addr, byte_size) &&
                        m_impl->memory->is_valid_address(dst_addr, byte_size)) {
                        const void* src = m_impl->memory->to_host_ptr<void>(src_addr);
                        void* dst = m_impl->memory->to_host_ptr<void>(dst_addr);
                        if (src && dst) {
                            std::memcpy(dst, src, byte_size);
                        }
                    }
                }
                return true;
            }

            default:
                return true;
        }
    }

    return true;
}

usize Pm4Translator::execute_ring_buffer(GpuRingBuffer& ring_buffer) {
    auto packets = ring_buffer.parse_pending_packets();
    usize count = 0;
    for (const auto& pkt : packets) {
        if (execute_packet(pkt)) {
            count++;
        }
    }
    return count;
}

const GpuViewport& Pm4Translator::viewport() const noexcept {
    static const GpuViewport s_def{};
    return m_impl ? m_impl->viewport : s_def;
}

const GpuScissor& Pm4Translator::scissor() const noexcept {
    static const GpuScissor s_def{};
    return m_impl ? m_impl->scissor : s_def;
}

ColorRGBA Pm4Translator::clear_color() const noexcept {
    return m_impl ? m_impl->clear_color : ColorRGBA{};
}

usize Pm4Translator::packets_processed() const noexcept {
    return m_impl ? m_impl->packets_processed : 0;
}

usize Pm4Translator::draw_count() const noexcept {
    return m_impl ? m_impl->draw_count : 0;
}

usize Pm4Translator::indexed_draw_count() const noexcept {
    return m_impl ? m_impl->indexed_draw_count : 0;
}

usize Pm4Translator::context_reg_writes() const noexcept {
    return m_impl ? m_impl->context_reg_writes : 0;
}

} // namespace ps4
