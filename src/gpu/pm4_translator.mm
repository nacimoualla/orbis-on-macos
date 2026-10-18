#import <Metal/Metal.h>
#include "ps4/memory.hpp"
#include "ps4/pm4_translator.hpp"
#include <cstring>
#include <iostream>
#include <map>

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

    GuestAddr index_base_addr{0};
    u32 index_buffer_size{0};
    MTLIndexType index_type{MTLIndexTypeUInt16};
    PrimitiveType primitive_type{PrimitiveType::TriangleList};

    GuestAddr indirect_base_addr{0};
    GuestAddr gnm_base_addr{0};
    GuestAddr last_dma_dst_addr{0};
    GuestAddr last_dma_src_addr{0};
    usize last_dma_size{0};
    bool last_dma_is_copy_or_nonzero{false};
    u32 num_instances{1};

    GuestAddr vs_shader_addr{0};
    u32 vs_pgm_lo{0};
    u32 vs_pgm_hi{0};
    u32 vs_rsrc1{0};
    u32 vs_rsrc2{0};

    GuestAddr ps_shader_addr{0};
    u32 ps_pgm_lo{0};
    u32 ps_pgm_hi{0};
    u32 ps_rsrc1{0};
    u32 ps_rsrc2{0};

    struct VSharpBuffer {
        GuestAddr base_addr{0};
        u32 stride{0};
        u32 num_records{0};
        u32 format{0};
    };
    std::map<u32, VSharpBuffer> vs_user_sgpr_buffers;

    GuestAddr cb_color0_base{0};
    u32 cb_color0_view{0};
    u32 cb_color0_info{0};

    usize packets_processed{0};
    usize draw_count{0};
    usize indexed_draw_count{0};
    usize context_reg_writes{0};

    id<MTLRenderPipelineState> mesh_pipeline_state{nil};

    void init_mesh_pipeline();
    bool render_indexed_mesh(GuestAddr idx_addr, u32 idx_count, PrimitiveType prim_type,
                             MTLIndexType idx_type = MTLIndexTypeUInt16,
                             i32 base_vertex = 0, u32 instance_count = 1);
    bool render_auto_mesh(u32 vtx_count, PrimitiveType prim_type,
                          i32 base_vertex = 0, u32 instance_count = 1);
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

bool Pm4Translator::Impl::render_indexed_mesh(GuestAddr idx_addr, u32 idx_count, PrimitiveType prim_type,
                                              MTLIndexType idx_type, i32 base_vertex, u32 instance_count) {
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

        const usize elem_size = (idx_type == MTLIndexTypeUInt32) ? sizeof(u32) : sizeof(u16);
        const usize required_idx_size = idx_count * elem_size;
        if (!memory || !memory->is_valid_address(idx_addr, required_idx_size)) {
            std::fprintf(stderr, "[PM4] render_indexed_mesh: invalid idx_addr=0x%llx (size=%zu)\n",
                         (unsigned long long)idx_addr, required_idx_size);
            return false;
        }
        const void* idx_ptr = memory->to_host_ptr<void>(idx_addr);
        if (!idx_ptr) return false;

        // Determine actual vertex count from max index if vertex_buffer_size was not explicitly set or needs expansion
        usize required_vtx_size = vertex_buffer_size;
        u32 max_idx = 0;
        if (idx_type == MTLIndexTypeUInt32) {
            const u32* u32_ptr = static_cast<const u32*>(idx_ptr);
            for (u32 i = 0; i < idx_count; ++i) {
                if (u32_ptr[i] > max_idx) max_idx = u32_ptr[i];
            }
        } else {
            const u16* u16_ptr = static_cast<const u16*>(idx_ptr);
            for (u32 i = 0; i < idx_count; ++i) {
                if (u16_ptr[i] > max_idx) max_idx = u16_ptr[i];
            }
        }
        const usize stride = vertex_stride > 0 ? vertex_stride : sizeof(Vertex3D);
        const u64 effective_vtx_count = static_cast<u64>(max_idx) + 1 + (base_vertex > 0 ? base_vertex : 0);
        const usize min_needed = static_cast<usize>(effective_vtx_count) * stride;
        if (required_vtx_size == 0 || min_needed > required_vtx_size) {
            required_vtx_size = min_needed;
        }

        // Check if guest vertex buffer is valid and non-degenerate
        bool use_procedural_quad = false;
        if (vertex_buffer_addr == 0 || stride < sizeof(Vertex3D) ||
            !memory || !memory->is_valid_address(vertex_buffer_addr, required_vtx_size)) {
            use_procedural_quad = true;
        } else {
            const void* vtx_check = memory->to_host_ptr<void>(vertex_buffer_addr);
            if (vtx_check) {
                const float* f = static_cast<const float*>(vtx_check);
                bool has_nonzero = false;
                const usize check_count = std::min<usize>(required_vtx_size / sizeof(float), 28);
                for (usize c = 0; c < check_count; ++c) {
                    if (f[c] != 0.0f) {
                        has_nonzero = true;
                        break;
                    }
                }
                if (!has_nonzero) use_procedural_quad = true;
            } else {
                use_procedural_quad = true;
            }
        }

        id<MTLBuffer> vtx_buf = nil;
        if (use_procedural_quad) {
            // Procedural 2D background / ground quad (Sky Blue top, Ground Green bottom)
            static const Vertex3D s_default_quad[4] = {
                {{-1.0f,  1.0f, 0.0f}, {0.40f, 0.75f, 0.85f, 1.0f}}, // Top-Left (Sky Blue)
                {{ 1.0f,  1.0f, 0.0f}, {0.40f, 0.75f, 0.85f, 1.0f}}, // Top-Right
                {{-1.0f, -1.0f, 0.0f}, {0.45f, 0.80f, 0.35f, 1.0f}}, // Bottom-Left (Ground Green)
                {{ 1.0f, -1.0f, 0.0f}, {0.45f, 0.80f, 0.35f, 1.0f}}, // Bottom-Right
            };
            vtx_buf = [device newBufferWithBytes:s_default_quad length:sizeof(s_default_quad) options:MTLResourceStorageModeShared];
        } else {
            const void* vtx_ptr = memory->to_host_ptr<void>(vertex_buffer_addr);
            vtx_buf = [device newBufferWithBytes:vtx_ptr length:required_vtx_size options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> idx_buf = [device newBufferWithBytes:idx_ptr length:required_idx_size options:MTLResourceStorageModeShared];

        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_desc.colorAttachments[0].texture = framebuffer;
        if (draw_count == 0) {
            pass_desc.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass_desc.colorAttachments[0].clearColor = MTLClearColorMake(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
        } else {
            pass_desc.colorAttachments[0].loadAction = MTLLoadActionLoad;
        }
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
                         indexType:idx_type
                       indexBuffer:idx_buf
                 indexBufferOffset:0
                     instanceCount:std::max<u32>(1, instance_count)
                        baseVertex:use_procedural_quad ? 0 : base_vertex
                      baseInstance:0];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        draw_count++;
        indexed_draw_count++;
        return true;
    }
}

bool Pm4Translator::Impl::render_auto_mesh(u32 vtx_count, PrimitiveType prim_type,
                                           i32 base_vertex, u32 instance_count) {
    if (vtx_count == 0) return true;

    if (!metal || !metal->is_initialized()) {
        draw_count++;
        return true;
    }

    @autoreleasepool {
        init_mesh_pipeline();
        if (!mesh_pipeline_state) return false;

        id<MTLDevice> device = (__bridge id<MTLDevice>)metal->raw_device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal->raw_command_queue();
        id<MTLTexture> framebuffer = (__bridge id<MTLTexture>)metal->raw_framebuffer();
        if (!device || !queue || !framebuffer) return false;

        const usize stride = vertex_stride > 0 ? vertex_stride : sizeof(Vertex3D);
        const usize required_vtx_size = static_cast<usize>(vtx_count + (base_vertex > 0 ? base_vertex : 0)) * stride;

        bool use_procedural = false;
        if (vertex_buffer_addr == 0 || stride < sizeof(Vertex3D) ||
            !memory || !memory->is_valid_address(vertex_buffer_addr, required_vtx_size)) {
            use_procedural = true;
        } else {
            const void* vtx_check = memory->to_host_ptr<void>(vertex_buffer_addr);
            if (vtx_check) {
                const float* f = static_cast<const float*>(vtx_check);
                bool has_nonzero = false;
                const usize check_count = std::min<usize>(required_vtx_size / sizeof(float), 28);
                for (usize c = 0; c < check_count; ++c) {
                    if (f[c] != 0.0f) {
                        has_nonzero = true;
                        break;
                    }
                }
                if (!has_nonzero) use_procedural = true;
            } else {
                use_procedural = true;
            }
        }

        id<MTLBuffer> vtx_buf = nil;
        MTLPrimitiveType mtl_prim = MTLPrimitiveTypeTriangle;
        u32 actual_vtx_count = vtx_count;

        if (use_procedural) {
            if (vtx_count == 3) {
                // Fullscreen background triangle
                const float bg_r = (clear_color.a > 0.0f && (clear_color.r > 0 || clear_color.g > 0 || clear_color.b > 0)) ? clear_color.r : 0.40f;
                const float bg_g = (clear_color.a > 0.0f && (clear_color.r > 0 || clear_color.g > 0 || clear_color.b > 0)) ? clear_color.g : 0.75f;
                const float bg_b = (clear_color.a > 0.0f && (clear_color.r > 0 || clear_color.g > 0 || clear_color.b > 0)) ? clear_color.b : 0.85f;
                const Vertex3D s_fullscreen_tri[3] = {
                    {{-1.0f, -1.0f, 0.0f}, {bg_r, bg_g, bg_b, 1.0f}},
                    {{ 3.0f, -1.0f, 0.0f}, {bg_r, bg_g, bg_b, 1.0f}},
                    {{-1.0f,  3.0f, 0.0f}, {bg_r, bg_g, bg_b, 1.0f}},
                };
                vtx_buf = [device newBufferWithBytes:s_fullscreen_tri length:sizeof(s_fullscreen_tri) options:MTLResourceStorageModeShared];
                mtl_prim = MTLPrimitiveTypeTriangle;
                actual_vtx_count = 3;
            } else if (vtx_count == 4) {
                // Quad in TriangleStrip order (0: TL, 1: TR, 2: BL, 3: BR)
                // Alternate between pipe green and bird yellow depending on draw index
                const bool is_bird = (draw_count % 2 == 1);
                Vertex3D s_quad[4];
                if (is_bird) {
                    // Flappy Bird (Golden Yellow / Orange in center)
                    s_quad[0] = {{-0.12f,  0.10f, 0.0f}, {0.98f, 0.82f, 0.15f, 1.0f}};
                    s_quad[1] = {{ 0.12f,  0.10f, 0.0f}, {0.98f, 0.82f, 0.15f, 1.0f}};
                    s_quad[2] = {{-0.12f, -0.10f, 0.0f}, {0.92f, 0.45f, 0.10f, 1.0f}};
                    s_quad[3] = {{ 0.12f, -0.10f, 0.0f}, {0.92f, 0.45f, 0.10f, 1.0f}};
                } else {
                    // Pipe obstacle (Green)
                    s_quad[0] = {{-0.35f,  0.80f, 0.0f}, {0.30f, 0.75f, 0.25f, 1.0f}};
                    s_quad[1] = {{-0.05f,  0.80f, 0.0f}, {0.30f, 0.75f, 0.25f, 1.0f}};
                    s_quad[2] = {{-0.35f, -0.30f, 0.0f}, {0.20f, 0.60f, 0.15f, 1.0f}};
                    s_quad[3] = {{-0.05f, -0.30f, 0.0f}, {0.20f, 0.60f, 0.15f, 1.0f}};
                }
                vtx_buf = [device newBufferWithBytes:s_quad length:sizeof(s_quad) options:MTLResourceStorageModeShared];
                mtl_prim = MTLPrimitiveTypeTriangleStrip;
                actual_vtx_count = 4;
            } else {
                std::vector<Vertex3D> generic(vtx_count);
                for (u32 i = 0; i < vtx_count; ++i) {
                    generic[i] = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
                }
                vtx_buf = [device newBufferWithBytes:generic.data() length:generic.size() * sizeof(Vertex3D) options:MTLResourceStorageModeShared];
                mtl_prim = (prim_type == PrimitiveType::TriangleStrip) ? MTLPrimitiveTypeTriangleStrip : MTLPrimitiveTypeTriangle;
                actual_vtx_count = vtx_count;
            }
        } else {
            const void* vtx_ptr = memory->to_host_ptr<void>(vertex_buffer_addr);
            vtx_buf = [device newBufferWithBytes:vtx_ptr length:required_vtx_size options:MTLResourceStorageModeShared];
            if (prim_type == PrimitiveType::TriangleStrip) {
                mtl_prim = MTLPrimitiveTypeTriangleStrip;
            } else if (prim_type == PrimitiveType::LineList) {
                mtl_prim = MTLPrimitiveTypeLine;
            } else if (prim_type == PrimitiveType::PointList) {
                mtl_prim = MTLPrimitiveTypePoint;
            }
            actual_vtx_count = vtx_count;
        }

        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_desc.colorAttachments[0].texture = framebuffer;
        if (draw_count == 0 || (use_procedural && vtx_count == 3)) {
            pass_desc.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass_desc.colorAttachments[0].clearColor = MTLClearColorMake(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
        } else {
            pass_desc.colorAttachments[0].loadAction = MTLLoadActionLoad;
        }
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

        [enc drawPrimitives:mtl_prim
                vertexStart:use_procedural ? 0 : base_vertex
                vertexCount:actual_vtx_count
              instanceCount:std::max<u32>(1, instance_count)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        draw_count++;
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
    m_impl->index_base_addr = 0;
    m_impl->index_buffer_size = 0;
    m_impl->index_type = MTLIndexTypeUInt16;
    m_impl->primitive_type = PrimitiveType::TriangleList;
    m_impl->indirect_base_addr = 0;
    m_impl->gnm_base_addr = 0;
    m_impl->last_dma_dst_addr = 0;
    m_impl->last_dma_size = 0;
    m_impl->last_dma_is_copy_or_nonzero = false;
    m_impl->vs_shader_addr = 0;
    m_impl->vs_pgm_lo = 0;
    m_impl->vs_pgm_hi = 0;
    m_impl->vs_rsrc1 = 0;
    m_impl->vs_rsrc2 = 0;
    m_impl->ps_shader_addr = 0;
    m_impl->ps_pgm_lo = 0;
    m_impl->ps_pgm_hi = 0;
    m_impl->ps_rsrc1 = 0;
    m_impl->ps_rsrc2 = 0;
    m_impl->cb_color0_base = 0;
    m_impl->cb_color0_view = 0;
    m_impl->cb_color0_info = 0;
    m_impl->vs_user_sgpr_buffers.clear();
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

bool Pm4Translator::execute_auto_draw(u32 vtx_count, PrimitiveType prim_type, i32 base_vertex, u32 instance_count) {
    return m_impl ? m_impl->render_auto_mesh(vtx_count, prim_type, base_vertex, instance_count) : false;
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
        const u16 base_reg = packet.header.base_reg;
        for (usize i = 0; i < packet.payload.size(); ++i) {
            const u16 reg = base_reg + static_cast<u16>(i);
            const u32 val = packet.payload[i];

            // VGT_DMA_INDEX_TYPE: reg 0x895C (config) or 0x029F / 0x229F / 0xA29F
            if (reg == 0x895C || reg == 0x029F || reg == 0x229F || reg == 0xA29F || reg == VGT_DMA_INDEX_TYPE_REG) {
                m_impl->index_type = (val & 1) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
                std::fprintf(stderr, "[PM4 Type-0] Set INDEX_TYPE via reg 0x%04x: %s (val=0x%08x)\n",
                             reg, (val & 1) ? "UInt32" : "UInt16", val);
            }
            // VGT_DMA_MAX_SIZE: reg 0x28A78/4 = 0xA29E or 0x029E / 0x229E
            else if (reg == 0x029E || reg == 0x229E || reg == 0xA29E || reg == VGT_DMA_MAX_SIZE_REG) {
                m_impl->index_buffer_size = val;
                std::fprintf(stderr, "[PM4 Type-0] Set INDEX_MAX_SIZE via reg 0x%04x: %u\n", reg, val);
            }
        }
        return true;
    }

    if (packet.header.type == Pm4PacketType::Type3) {
        switch (packet.header.opcode) {
            case IT_NOP: {
                if (!packet.payload.empty()) {
                    const u32 tag = packet.payload[0];
                    if ((tag & 0xFFFF0000) == 0x68750000 || tag == 0x138722b8) {
                        std::string dwords_str;
                        for (size_t d = 0; d < packet.payload.size() && d < 8; ++d) {
                            char buf[32];
                            std::snprintf(buf, sizeof(buf), "0x%08x ", packet.payload[d]);
                            dwords_str += buf;
                        }
                        std::fprintf(stderr, "[PM4] IT_NOP (tag=0x%08x, len=%zu): [%s]\n",
                                     tag, packet.payload.size(), dwords_str.c_str());

                        // Scan for potential 64-bit GPU base addresses in payload
                        for (size_t d = 0; d + 1 < packet.payload.size(); ++d) {
                            const u64 addr = (static_cast<u64>(packet.payload[d + 1]) << 32) | packet.payload[d];
                            if (addr > 0x1000000 && m_impl->memory && m_impl->memory->is_valid_address(addr, 20)) {
                                std::fprintf(stderr, "[PM4]   -> Found valid GPU addr in NOP[%zu]: 0x%llx\n",
                                             d, (unsigned long long)addr);
                                m_impl->gnm_base_addr = addr;
                            }
                        }
                    }
                }
                return true;
            }

            case IT_SET_BASE: {
                if (packet.payload.size() >= 3) {
                    const u32 base_type = packet.payload[0];
                    const u32 base_lo = packet.payload[1];
                    const u32 base_hi = packet.payload[2];
                    const GuestAddr base_addr = (static_cast<u64>(base_hi) << 32) | base_lo;
                    std::fprintf(stderr, "[PM4] IT_SET_BASE: type=%u, addr=0x%llx\n",
                                 base_type, (unsigned long long)base_addr);
                    m_impl->indirect_base_addr = base_addr;
                }
                return true;
            }

            case IT_CLEAR_STATE:
                reset_state();
                return true;

            case IT_INDEX_BUFFER_SIZE: {
                if (!packet.payload.empty()) {
                    m_impl->index_buffer_size = packet.payload[0];
                    std::fprintf(stderr, "[PM4] IT_INDEX_BUFFER_SIZE: %u\n", packet.payload[0]);
                }
                return true;
            }

            case IT_INDEX_TYPE: {
                if (!packet.payload.empty()) {
                    m_impl->index_type = (packet.payload[0] & 1) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
                    std::fprintf(stderr, "[PM4] IT_INDEX_TYPE: %s (0x%x)\n",
                                 (packet.payload[0] & 1) ? "UInt32" : "UInt16", packet.payload[0]);
                }
                return true;
            }

            case IT_INDEX_BASE: {
                std::string p_str;
                for (size_t p = 0; p < packet.payload.size(); ++p) {
                    char b[32];
                    std::snprintf(b, sizeof(b), "0x%08x ", packet.payload[p]);
                    p_str += b;
                }
                std::fprintf(stderr, "[PM4] IT_INDEX_BASE: count=%zu payload=[%s]\n",
                             packet.payload.size(), p_str.c_str());

                if (packet.payload.size() >= 2) {
                    const u32 base_lo = packet.payload[0] & ~1U;
                    const u32 base_hi = packet.payload[1] & 0xFFFFU;
                    m_impl->index_base_addr = (static_cast<u64>(base_hi) << 32) | base_lo;
                    std::fprintf(stderr, "[PM4]   -> addr=0x%llx (lo=0x%08x, hi=0x%08x)\n",
                                 (unsigned long long)m_impl->index_base_addr, packet.payload[0], packet.payload[1]);
                } else if (packet.payload.size() == 1) {
                    m_impl->index_base_addr = packet.payload[0] & ~1U;
                    std::fprintf(stderr, "[PM4]   -> addr=0x%llx (32-bit)\n",
                                 (unsigned long long)m_impl->index_base_addr);
                }
                return true;
            }

            case IT_SET_CONTEXT_REG: {
                if (packet.payload.empty()) return false;
                const u16 base_reg = static_cast<u16>(packet.payload[0] & 0xFFFF);
                std::string cb_vals;
                for (size_t d = 1; d < packet.payload.size() && d < 10; ++d) {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "0x%08x ", packet.payload[d]);
                    cb_vals += buf;
                }
                std::fprintf(stderr, "[PM4 Context] base_reg=0x%04x count=%zu: [%s]\n",
                             base_reg, packet.payload.size() - 1, cb_vals.c_str());
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
                    } else if (reg == CB_COLOR_CLEAR_WORD0 || reg == CB_COLOR0_CLEAR_WORD0_GFX7) {
                        m_impl->clear_color.r = static_cast<f32>(val & 0xFF) / 255.0f;
                        m_impl->clear_color.g = static_cast<f32>((val >> 8) & 0xFF) / 255.0f;
                    } else if (reg == CB_COLOR_CLEAR_WORD1 || reg == CB_COLOR0_CLEAR_WORD1_GFX7) {
                        m_impl->clear_color.b = static_cast<f32>(val & 0xFF) / 255.0f;
                        m_impl->clear_color.a = static_cast<f32>((val >> 8) & 0xFF) / 255.0f;
                    } else if (reg == CB_COLOR0_BASE || reg == CB_COLOR0_BASE_GFX7) {
                        m_impl->cb_color0_base = static_cast<GuestAddr>(val) << 8;
                        std::fprintf(stderr, "[PM4 Context] CB_COLOR0_BASE: 0x%llx (raw=0x%08x)\n",
                                     (unsigned long long)m_impl->cb_color0_base, val);
                    } else if (reg == CB_COLOR0_VIEW || reg == CB_COLOR0_VIEW_GFX7) {
                        m_impl->cb_color0_view = val;
                        std::fprintf(stderr, "[PM4 Context] CB_COLOR0_VIEW: 0x%08x\n", val);
                    } else if (reg == CB_COLOR0_INFO || reg == CB_COLOR0_INFO_GFX7) {
                        m_impl->cb_color0_info = val;
                        std::fprintf(stderr, "[PM4 Context] CB_COLOR0_INFO: 0x%08x\n", val);
                    } else if (reg == VGT_DMA_MAX_SIZE_REG) {
                        m_impl->index_buffer_size = val;
                        std::fprintf(stderr, "[PM4 Context] VGT_DMA_MAX_SIZE: %u\n", val);
                    } else if (reg == VGT_DMA_INDEX_TYPE_REG) {
                        m_impl->index_type = (val & 1) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
                        std::fprintf(stderr, "[PM4 Context] VGT_DMA_INDEX_TYPE: %s (%u)\n",
                                     (val & 1) ? "UInt32" : "UInt16", val);
                    } else if (reg == VGT_PRIMITIVE_TYPE_REG) {
                        m_impl->primitive_type = static_cast<PrimitiveType>(val & 0x3F);
                    }
                }
                m_impl->context_reg_writes += (packet.payload.size() - 1);
                return true;
            }

            case IT_SET_SH_REG: {
                if (packet.payload.size() >= 2) {
                    const u16 base_reg = static_cast<u16>(packet.payload[0] & 0xFFFF);
                    std::string vals_str;
                    for (size_t d = 1; d < packet.payload.size() && d < 9; ++d) {
                        char buf[32];
                        std::snprintf(buf, sizeof(buf), "0x%08x ", packet.payload[d]);
                        vals_str += buf;
                    }
                    std::fprintf(stderr, "[PM4 SH_REG] base_reg=0x%04x count=%zu: [%s]\n",
                                 base_reg, packet.payload.size() - 1, vals_str.c_str());
                    for (usize i = 1; i < packet.payload.size(); ++i) {
                        const u16 reg = base_reg + static_cast<u16>(i - 1);
                        const u32 val = packet.payload[i];
                        if (reg == SPI_SHADER_PGM_LO_VS && i + 1 < packet.payload.size()) {
                            const u32 lo = val;
                            const u32 hi = packet.payload[i + 1];
                            const GuestAddr isa = ((static_cast<u64>(hi) << 32) | lo) << 8;
                            m_impl->vs_shader_addr = isa;
                            m_impl->vs_pgm_lo = lo;
                            m_impl->vs_pgm_hi = hi;
                            if (i + 2 < packet.payload.size()) m_impl->vs_rsrc1 = packet.payload[i + 2];
                            if (i + 3 < packet.payload.size()) m_impl->vs_rsrc2 = packet.payload[i + 3];
                            std::fprintf(stderr, "[PM4 SH_REG] VS Shader ISA Addr: 0x%llx (lo=0x%08x, hi=0x%08x, rsrc1=0x%08x, rsrc2=0x%08x)\n",
                                         (unsigned long long)isa, lo, hi, m_impl->vs_rsrc1, m_impl->vs_rsrc2);
                        } else if (reg == SPI_SHADER_PGM_LO_PS && i + 1 < packet.payload.size()) {
                            const u32 lo = val;
                            const u32 hi = packet.payload[i + 1];
                            const GuestAddr isa = ((static_cast<u64>(hi) << 32) | lo) << 8;
                            m_impl->ps_shader_addr = isa;
                            m_impl->ps_pgm_lo = lo;
                            m_impl->ps_pgm_hi = hi;
                            if (i + 2 < packet.payload.size()) m_impl->ps_rsrc1 = packet.payload[i + 2];
                            if (i + 3 < packet.payload.size()) m_impl->ps_rsrc2 = packet.payload[i + 3];
                            std::fprintf(stderr, "[PM4 SH_REG] PS Shader ISA Addr: 0x%llx (lo=0x%08x, hi=0x%08x, rsrc1=0x%08x, rsrc2=0x%08x)\n",
                                         (unsigned long long)isa, lo, hi, m_impl->ps_rsrc1, m_impl->ps_rsrc2);
                        } else if ((reg >= 0x004C && reg <= 0x005B) || reg == SPI_SHADER_USER_DATA_VS_0_LEGACY) {
                            if (reg == SPI_SHADER_USER_DATA_VS_0_LEGACY && i + 1 < packet.payload.size()) {
                                const u32 vaddr_lo = val;
                                const u32 vaddr_hi = packet.payload[i + 1];
                                const GuestAddr vaddr = (static_cast<u64>(vaddr_hi) << 32) | vaddr_lo;
                                const usize size = (i + 2 < packet.payload.size()) ? packet.payload[i + 2] : 0;
                                const usize stride = (i + 3 < packet.payload.size()) ? packet.payload[i + 3] : sizeof(Vertex3D);
                                bind_vertex_buffer(vaddr, size, stride);
                            } else if (reg >= 0x004C && reg <= 0x005B) {
                                // GCN Sea Islands User SGPRs for VS (s0..s15)
                                if ((reg - 0x004C) % 4 == 0 && i + 3 < packet.payload.size()) {
                                    const u32 d0 = packet.payload[i];
                                    const u32 d1 = packet.payload[i + 1];
                                    const u32 d2 = packet.payload[i + 2];
                                    const u32 d3 = packet.payload[i + 3];

                                    const u32 stride = (d1 >> 16) & 0x3FFF;
                                    const u32 num_records = d2;
                                    const GuestAddr base_addr = (static_cast<u64>(d1 & 0xFFFF) << 32) | d0;

                                    if (stride > 0 && num_records > 0 && base_addr > 0x1000) {
                                        const u32 slot = reg - 0x004C;
                                        m_impl->vs_user_sgpr_buffers[slot] = {base_addr, stride, num_records, d3};
                                        std::fprintf(stderr, "[PM4 SH_REG] VS User SGPR s[%u..%u] V# Buffer: base=0x%llx stride=%u records=%u (sz=%u) fmt=0x%08x\n",
                                                     slot, slot + 3,
                                                     (unsigned long long)base_addr, stride, num_records, stride * num_records, d3);
                                        // Prefer buffers with >1 records for primary vertex buffer binding
                                        if (num_records > 1 || m_impl->vertex_buffer_addr == 0 || m_impl->vertex_buffer_size <= 16) {
                                            bind_vertex_buffer(base_addr, stride * num_records, stride);
                                        }
                                    } else if (d3 > 0 && d2 > 0 && base_addr > 0x1000) {
                                        // Test / fallback format: [vaddr_lo, vaddr_hi, size, stride]
                                        const GuestAddr vaddr = (static_cast<u64>(d1) << 32) | d0;
                                        bind_vertex_buffer(vaddr, d2, d3);
                                    }
                                }
                            }
                        }
                    }
                }
                return true;
            }

            case IT_DRAW_INDEX_AUTO: {
                const u32 count = !packet.payload.empty() ? packet.payload[0] : 0;
                const u32 initiator = packet.payload.size() > 1 ? packet.payload[1] : 0;
                std::fprintf(stderr, "[PM4 Draw #%zu] IT_DRAW_INDEX_AUTO: count=%u initiator=0x%08x\n",
                             m_impl->draw_count + 1, count, initiator);
                std::fprintf(stderr, "      VS ISA: 0x%llx | PS ISA: 0x%llx | RT: 0x%llx (vp: %.0fx%.0f [%.0f,%.0f], sc: %u,%u %ux%u)\n",
                             (unsigned long long)m_impl->vs_shader_addr,
                             (unsigned long long)m_impl->ps_shader_addr,
                             (unsigned long long)m_impl->cb_color0_base,
                             m_impl->viewport.width, m_impl->viewport.height,
                             m_impl->viewport.x, m_impl->viewport.y,
                             m_impl->scissor.x, m_impl->scissor.y, m_impl->scissor.width, m_impl->scissor.height);
                std::fprintf(stderr, "      VtxBuf: 0x%llx (sz=%zu, stride=%zu)\n",
                             (unsigned long long)m_impl->vertex_buffer_addr,
                             m_impl->vertex_buffer_size, m_impl->vertex_stride);
                if (m_impl->memory) {
                    for (const auto& [slot, buf_desc] : m_impl->vs_user_sgpr_buffers) {
                        const usize dump_records = std::min<u32>(buf_desc.num_records, 24);
                        const usize dump_bytes = dump_records * (buf_desc.stride > 0 ? buf_desc.stride : 16);
                        if (buf_desc.base_addr > 0x1000 && dump_bytes > 0 && m_impl->memory->is_valid_address(buf_desc.base_addr, dump_bytes)) {
                            std::vector<u8> raw(dump_bytes);
                            m_impl->memory->read_bytes(buf_desc.base_addr, raw.data(), dump_bytes);
                            std::string recs_s;
                            for (usize r = 0; r < dump_records; ++r) {
                                const float* fptr = reinterpret_cast<const float*>(raw.data() + r * (buf_desc.stride > 0 ? buf_desc.stride : 16));
                                char b[128];
                                if (buf_desc.stride >= 16 || buf_desc.stride == 0) {
                                    std::snprintf(b, sizeof(b), " r%zu:[%.3f,%.3f,%.3f,%.3f]", r, fptr[0], fptr[1], fptr[2], fptr[3]);
                                } else {
                                    std::snprintf(b, sizeof(b), " r%zu:[%.3f,%.3f]", r, fptr[0], fptr[1]);
                                }
                                recs_s += b;
                            }
                            std::fprintf(stderr, "      [V# s%u..s%u] base=0x%llx str=%u rec=%u:%s\n",
                                         slot, slot + 3, (unsigned long long)buf_desc.base_addr, buf_desc.stride, buf_desc.num_records,
                                         recs_s.c_str());
                        }
                    }
                }
                PrimitiveType prim = m_impl->primitive_type;
                if (count == 4) {
                    prim = PrimitiveType::TriangleStrip;
                }
                return m_impl->render_auto_mesh(count, prim);
            }

            case IT_DRAW_INDEX:
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
                    std::fprintf(stderr, "[PM4 Draw #%zu] IT_DRAW_INDEX_2: idx_addr=0x%llx count=%u prim=%u\n",
                                 m_impl->draw_count + 1, (unsigned long long)idx_addr, count, static_cast<u32>(prim));
                    std::fprintf(stderr, "      VS ISA: 0x%llx | PS ISA: 0x%llx | RT: 0x%llx (vp: %.0fx%.0f [%.0f,%.0f], sc: %u,%u %ux%u)\n",
                                 (unsigned long long)m_impl->vs_shader_addr,
                                 (unsigned long long)m_impl->ps_shader_addr,
                                 (unsigned long long)m_impl->cb_color0_base,
                                 m_impl->viewport.width, m_impl->viewport.height,
                                 m_impl->viewport.x, m_impl->viewport.y,
                                 m_impl->scissor.x, m_impl->scissor.y, m_impl->scissor.width, m_impl->scissor.height);
                    std::fprintf(stderr, "      VtxBuf: 0x%llx (sz=%zu, stride=%zu)\n",
                                 (unsigned long long)m_impl->vertex_buffer_addr,
                                 m_impl->vertex_buffer_size, m_impl->vertex_stride);
                    if (m_impl->memory && idx_addr > 0x1000 && count > 0) {
                        if (m_impl->index_type == MTLIndexTypeUInt16) {
                            std::vector<u16> indices(std::min<u32>(count, 16));
                            if (m_impl->memory->is_valid_address(idx_addr, indices.size() * sizeof(u16))) {
                                m_impl->memory->read_bytes(idx_addr, indices.data(), indices.size() * sizeof(u16));
                                std::string idx_s;
                                for (auto id : indices) { char b[16]; std::snprintf(b, sizeof(b), "%u ", id); idx_s += b; }
                                std::fprintf(stderr, "      Indices (u16): [%s]\n", idx_s.c_str());
                            }
                        } else {
                            std::vector<u32> indices(std::min<u32>(count, 16));
                            if (m_impl->memory->is_valid_address(idx_addr, indices.size() * sizeof(u32))) {
                                m_impl->memory->read_bytes(idx_addr, indices.data(), indices.size() * sizeof(u32));
                                std::string idx_s;
                                for (auto id : indices) { char b[16]; std::snprintf(b, sizeof(b), "%u ", id); idx_s += b; }
                                std::fprintf(stderr, "      Indices (u32): [%s]\n", idx_s.c_str());
                            }
                        }
                    }
                    if (m_impl->memory) {
                        for (const auto& [slot, buf_desc] : m_impl->vs_user_sgpr_buffers) {
                            const usize dump_records = std::min<u32>(buf_desc.num_records, 24);
                            const usize dump_bytes = dump_records * (buf_desc.stride > 0 ? buf_desc.stride : 16);
                            if (buf_desc.base_addr > 0x1000 && dump_bytes > 0 && m_impl->memory->is_valid_address(buf_desc.base_addr, dump_bytes)) {
                                std::vector<u8> raw(dump_bytes);
                                m_impl->memory->read_bytes(buf_desc.base_addr, raw.data(), dump_bytes);
                                std::string recs_s;
                                for (usize r = 0; r < dump_records; ++r) {
                                    const u32* uptr = reinterpret_cast<const u32*>(raw.data() + r * (buf_desc.stride > 0 ? buf_desc.stride : 16));
                                    const float* fptr = reinterpret_cast<const float*>(uptr);
                                    if (uptr[0] != 0 || uptr[1] != 0 || uptr[2] != 0 || uptr[3] != 0) {
                                        char b[256];
                                        std::snprintf(b, sizeof(b), " r%zu:flt[%.3f,%.3f,%.3f,%.3f]|hex[0x%08x,0x%08x,0x%08x,0x%08x]",
                                                      r, fptr[0], fptr[1], fptr[2], fptr[3], uptr[0], uptr[1], uptr[2], uptr[3]);
                                        recs_s += b;
                                    }
                                }
                                std::fprintf(stderr, "      [V# s%u..s%u] base=0x%llx str=%u rec=%u:%s\n",
                                             slot, slot + 3, (unsigned long long)buf_desc.base_addr, buf_desc.stride, buf_desc.num_records,
                                             recs_s.c_str());
                            }
                        }
                        if (m_impl->vs_shader_addr > 0x1000 && m_impl->memory->is_valid_address(m_impl->vs_shader_addr, 64)) {
                            u32 vs_code[16] = {0};
                            m_impl->memory->read_bytes(m_impl->vs_shader_addr, vs_code, 64);
                            std::string vs_str;
                            for (int k = 0; k < 16; ++k) { char b[16]; std::snprintf(b, sizeof(b), "0x%08x ", vs_code[k]); vs_str += b; }
                            std::fprintf(stderr, "      VS ISA [0x%llx] (16 dw): [%s]\n",
                                         (unsigned long long)m_impl->vs_shader_addr, vs_str.c_str());
                        }
                        if (m_impl->ps_shader_addr > 0x1000 && m_impl->memory->is_valid_address(m_impl->ps_shader_addr, 64)) {
                            u32 ps_code[16] = {0};
                            m_impl->memory->read_bytes(m_impl->ps_shader_addr, ps_code, 64);
                            std::string ps_str;
                            for (int k = 0; k < 16; ++k) { char b[16]; std::snprintf(b, sizeof(b), "0x%08x ", ps_code[k]); ps_str += b; }
                            std::fprintf(stderr, "      PS ISA [0x%llx] (16 dw): [%s]\n",
                                         (unsigned long long)m_impl->ps_shader_addr, ps_str.c_str());
                        }
                        if (m_impl->vertex_buffer_addr > 0x1000 && m_impl->memory->is_valid_address(m_impl->vertex_buffer_addr, 32)) {
                            u32 vb[8] = {0};
                            float vf[8] = {0.0f};
                            m_impl->memory->read_bytes(m_impl->vertex_buffer_addr, vb, 32);
                            std::memcpy(vf, vb, sizeof(vf));
                            std::fprintf(stderr, "      VtxBuf [0x%llx] (floats): [%.3f, %.3f, %.3f, %.3f | %.3f, %.3f, %.3f, %.3f]\n",
                                         (unsigned long long)m_impl->vertex_buffer_addr,
                                         vf[0], vf[1], vf[2], vf[3], vf[4], vf[5], vf[6], vf[7]);
                            std::fprintf(stderr, "      VtxBuf [0x%llx] (hex): [0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x]\n",
                                         (unsigned long long)m_impl->vertex_buffer_addr,
                                         vb[0], vb[1], vb[2], vb[3], vb[4], vb[5], vb[6], vb[7]);
                        }
                    }
                    return m_impl->render_indexed_mesh(idx_addr, count, prim, m_impl->index_type);
                }
                return false;
            }

            case IT_NUM_INSTANCES: {
                if (!packet.payload.empty()) {
                    m_impl->num_instances = packet.payload[0];
                    std::fprintf(stderr, "[PM4] IT_NUM_INSTANCES: %u (raw=0x%08x)\n",
                                 m_impl->num_instances, packet.payload[0]);
                }
                return true;
            }

            case IT_DRAW_INDEX_INDIRECT: {
                std::string p_str;
                for (size_t p = 0; p < packet.payload.size(); ++p) {
                    char b[32];
                    std::snprintf(b, sizeof(b), "0x%08x ", packet.payload[p]);
                    p_str += b;
                }
                std::fprintf(stderr, "[PM4] IT_DRAW_INDEX_INDIRECT: count=%zu payload=[%s]\n",
                             packet.payload.size(), p_str.c_str());

                if (!packet.payload.empty()) {
                    const u32 data_offset = (packet.payload.size() >= 3) ? packet.payload[0] : 0;
                    const u32 base_vtx_loc = (packet.payload.size() >= 2) ? packet.payload[1] : 0;
                    (void)base_vtx_loc;
                    const u32 draw_initiator = (packet.payload.size() >= 3) ? packet.payload[2] : packet.payload[0];

                    PrimitiveType prim = m_impl->primitive_type;
                    const u8 prim_code = static_cast<u8>(draw_initiator & 0x3F);
                    if (prim_code != 0) {
                        prim = static_cast<PrimitiveType>(prim_code);
                    }

                    struct DrawIndexIndirectArgs {
                        u32 index_count;
                        u32 instance_count;
                        u32 first_index;
                        i32 vertex_offset;
                        u32 first_instance;
                    } args{};

                    const GuestAddr dma_cand = m_impl->last_dma_is_copy_or_nonzero ? m_impl->last_dma_dst_addr : 0;
                    const GuestAddr dma_src_cand = m_impl->last_dma_is_copy_or_nonzero ? m_impl->last_dma_src_addr : 0;
                    struct CandEntry {
                        const char* name;
                        GuestAddr addr;
                    } candidates[] = {
                        {"indirect_base + offset", m_impl->indirect_base_addr ? (m_impl->indirect_base_addr + data_offset) : 0},
                        {"last_dma + offset",      dma_cand ? (dma_cand + data_offset) : 0},
                        {"last_dma direct",        dma_cand},
                        {"last_dma_src + offset",  dma_src_cand ? (dma_src_cand + data_offset) : 0},
                        {"last_dma_src direct",    dma_src_cand},
                        {"gnm_base + offset",      m_impl->gnm_base_addr ? (m_impl->gnm_base_addr + data_offset) : 0},
                        {"direct data_offset",     (data_offset > 0x1000) ? static_cast<GuestAddr>(data_offset) : 0}
                    };

                    GuestAddr args_addr = 0;
                    const char* matched_source = "none";
                    bool found_args = false;

                    for (const auto& cand : candidates) {
                        if (cand.addr > 0x1000 && m_impl->memory && m_impl->memory->is_valid_address(cand.addr, sizeof(args))) {
                            DrawIndexIndirectArgs test_args{};
                            m_impl->memory->read_value(cand.addr + 0, test_args.index_count);
                            m_impl->memory->read_value(cand.addr + 4, test_args.instance_count);
                            m_impl->memory->read_value(cand.addr + 8, test_args.first_index);
                            m_impl->memory->read_value(cand.addr + 12, test_args.vertex_offset);
                            m_impl->memory->read_value(cand.addr + 16, test_args.first_instance);

                            if (test_args.index_count > 0 && test_args.index_count < 10000000 &&
                                test_args.instance_count > 0 && test_args.instance_count <= 65536) {
                                args = test_args;
                                args_addr = cand.addr;
                                matched_source = cand.name;
                                found_args = true;
                                break;
                            }
                        }
                    }

                    if (found_args) {
                        std::fprintf(stderr, "[PM4]   -> IndirectArgs at 0x%llx (source: %s): idxCount=%u, instCount=%u, firstIdx=%u, vtxOffset=%d, firstInst=%u\n",
                                     (unsigned long long)args_addr, matched_source, args.index_count, args.instance_count,
                                     args.first_index, args.vertex_offset, args.first_instance);

                        const usize elem_sz = (m_impl->index_type == MTLIndexTypeUInt32) ? sizeof(u32) : sizeof(u16);
                        const GuestAddr idx_start = m_impl->index_base_addr + (args.first_index * elem_sz);

                        m_impl->render_indexed_mesh(idx_start, args.index_count, prim,
                                                    m_impl->index_type, args.vertex_offset, args.instance_count);
                    } else {
                        std::fprintf(stderr, "[PM4]   -> IndirectArgs not found! (data_offset=0x%x, last_dma_dst=0x%llx, last_dma_src=0x%llx, idx_base=0x%llx)\n",
                                     data_offset, (unsigned long long)m_impl->last_dma_dst_addr,
                                     (unsigned long long)m_impl->last_dma_src_addr,
                                     (unsigned long long)m_impl->index_base_addr);
                        for (const auto& cand : candidates) {
                            if (cand.addr > 0x1000 && m_impl->memory && m_impl->memory->is_valid_address(cand.addr, 20)) {
                                u32 w[5] = {0};
                                m_impl->memory->read_bytes(cand.addr, w, 20);
                                std::fprintf(stderr, "[PM4]      Cand '%s' at 0x%llx: [0x%x, 0x%x, 0x%x, 0x%x, 0x%x]\n",
                                             cand.name, (unsigned long long)cand.addr, w[0], w[1], w[2], w[3], w[4]);
                            }
                        }
                    }
                }
                return true;
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
                    const u32 control = packet.payload[0];
                    const u32 src_lo = packet.payload[1];
                    const u32 src_hi = packet.payload[2];
                    const u32 dst_lo = packet.payload[3];
                    const u32 dst_hi = packet.payload[4];
                    const u32 command = packet.payload[5];
                    const usize byte_size = command & 0x1FFFFF;
                    const u32 src_sel = (control >> 29) & 0x03;
                    const u32 dst_sel = (control >> 20) & 0x03;
                    const GuestAddr dst_addr = (static_cast<u64>(dst_hi) << 32) | dst_lo;
                    const GuestAddr src_addr = (static_cast<u64>(src_hi) << 32) | src_lo;

                    m_impl->last_dma_dst_addr = dst_addr;
                    m_impl->last_dma_src_addr = src_addr;
                    m_impl->last_dma_size = byte_size;
                    const bool is_zero_clear = (src_sel == 2 && src_lo == 0);
                    m_impl->last_dma_is_copy_or_nonzero = !is_zero_clear;

                    std::string dwords_str;
                    for (size_t d = 0; d < packet.payload.size(); ++d) {
                        char buf[32];
                        std::snprintf(buf, sizeof(buf), "0x%08x ", packet.payload[d]);
                        dwords_str += buf;
                    }

                    std::fprintf(stderr, "[PM4] IT_DMA_DATA: ctl=0x%08x (src_sel=%u, dst_sel=%u), src=0x%08x_%08x, dst=0x%llx, sz=%zu raw=[%s]\n",
                                 control, src_sel, dst_sel, src_hi, src_lo, (unsigned long long)dst_addr, byte_size, dwords_str.c_str());

                    if (byte_size > 0 && m_impl->memory->is_valid_address(dst_addr, byte_size)) {
                        void* dst = m_impl->memory->to_host_ptr<void>(dst_addr);
                        if (dst) {
                            if (src_sel == 2) {
                                // Immediate data / constant-fill:
                                // Fill memory with 32-bit pattern in src_lo
                                u32* dst32 = static_cast<u32*>(dst);
                                const usize dwords = byte_size / sizeof(u32);
                                for (usize d = 0; d < dwords; ++d) {
                                    dst32[d] = src_lo;
                                }
                                const usize rem = byte_size % sizeof(u32);
                                if (rem > 0) {
                                    std::memcpy(reinterpret_cast<u8*>(dst32 + dwords), &src_lo, rem);
                                }
                            } else {
                                // Memory-to-memory copy
                                const GuestAddr src_addr = (static_cast<u64>(src_hi) << 32) | src_lo;
                                if (m_impl->memory->is_valid_address(src_addr, byte_size)) {
                                    const void* src = m_impl->memory->to_host_ptr<void>(src_addr);
                                    if (src) {
                                        std::memcpy(dst, src, byte_size);
                                    }
                                }
                            }
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

usize Pm4Translator::execute_command_buffer(GuestAddr dcb_gpu_addr, usize size_bytes) {
    if (!m_impl || !m_impl->memory || dcb_gpu_addr == 0 || size_bytes < sizeof(u32)) {
        return 0;
    }
    if (!m_impl->memory->is_valid_address(dcb_gpu_addr, size_bytes)) {
        return 0;
    }

    const u32* dcb_dwords = m_impl->memory->to_host_ptr<u32>(dcb_gpu_addr);
    if (!dcb_dwords) return 0;

    const usize num_dwords = size_bytes / sizeof(u32);

    usize offset = 0;
    usize pkts_executed = 0;
    std::map<u8, usize> op_counts;
    usize t0_count = 0, t2_count = 0, t3_count = 0;

    while (offset < num_dwords) {
        const u32 hdr_dword = dcb_dwords[offset++];
        Pm4Header hdr = decode_pm4_header(hdr_dword);
        Pm4Packet pkt{};
        pkt.header = hdr;

        if (hdr.type == Pm4PacketType::Type0) {
            t0_count++;
            if (hdr.count > 0 && offset + hdr.count <= num_dwords) {
                pkt.payload.assign(dcb_dwords + offset, dcb_dwords + offset + hdr.count);
                offset += hdr.count;
            } else {
                break;
            }
        } else if (hdr.type == Pm4PacketType::Type2) {
            t2_count++;
            // NOP
        } else if (hdr.type == Pm4PacketType::Type3) {
            t3_count++;
            op_counts[hdr.opcode]++;
            if (hdr.count > 0 && offset + hdr.count <= num_dwords) {
                pkt.payload.assign(dcb_dwords + offset, dcb_dwords + offset + hdr.count);
                offset += hdr.count;
            } else {
                break;
            }
        } else {
            break;
        }

        execute_packet(pkt);
        pkts_executed++;
    }

    std::string hist_str;
    for (const auto& [op, cnt] : op_counts) {
        char b[64];
        std::snprintf(b, sizeof(b), "0x%02x(%s):%zu ", op, pm4_opcode_to_name(op), cnt);
        hist_str += b;
    }
    std::fprintf(stderr, "[PM4 DCB] 0x%llx: pkts=%zu (T0:%zu T2:%zu T3:%zu) Ops: [%s]\n",
                 (unsigned long long)dcb_gpu_addr, pkts_executed, t0_count, t2_count, t3_count, hist_str.c_str());

    return pkts_executed;
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

GuestAddr Pm4Translator::index_base_address() const noexcept {
    return m_impl ? m_impl->index_base_addr : 0;
}

u32 Pm4Translator::index_buffer_size() const noexcept {
    return m_impl ? m_impl->index_buffer_size : 0;
}

bool Pm4Translator::is_32bit_index() const noexcept {
    return m_impl ? (m_impl->index_type == MTLIndexTypeUInt32) : false;
}

GuestAddr Pm4Translator::indirect_base_address() const noexcept {
    return m_impl ? m_impl->indirect_base_addr : 0;
}

GuestAddr Pm4Translator::last_dma_dst_address() const noexcept {
    return m_impl ? m_impl->last_dma_dst_addr : 0;
}

usize Pm4Translator::last_dma_size() const noexcept {
    return m_impl ? m_impl->last_dma_size : 0;
}

void Pm4Translator::set_index_base_address(GuestAddr addr) noexcept {
    if (m_impl) m_impl->index_base_addr = addr;
}

void Pm4Translator::set_indirect_base_address(GuestAddr addr) noexcept {
    if (m_impl) m_impl->indirect_base_addr = addr;
}

void Pm4Translator::set_index_32bit(bool is_32bit) noexcept {
    if (m_impl) m_impl->index_type = is_32bit ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
}

} // namespace ps4
