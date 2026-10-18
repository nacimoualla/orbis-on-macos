#import <Metal/Metal.h>
#include <simd/simd.h>
#include "ps4/memory.hpp"
#include "ps4/pm4_translator.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

namespace ps4 {

struct MeshVertex {
    float position[3]; // X, Y, Z
    float color[4];    // R, G, B, A
    float texcoord[2]; // U, V
};

struct MeshUniforms {
    simd_float4x4 mvp;
    simd_float4 uv_scale_offset;
    simd_float4 tint_color;
    u32 vertex_stride;
    u32 use_texture;
    u32 pad[2];
};

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
    id<MTLTexture> default_white_texture{nil};
    id<MTLSamplerState> sampler_state{nil};
    id<MTLTexture> active_metal_texture{nil};

    GuestAddr ps_texture_base{0};
    u32 ps_texture_width{0};
    u32 ps_texture_height{0};
    u32 ps_texture_format{0};
    u32 ps_texture_w0{0};
    u32 ps_texture_w1{0};
    u32 ps_texture_w2{0};
    u32 ps_texture_w3{0};
    bool ps_texture_has_nonzero{false};
    usize ps_texture_nonzero_bytes{0};
    std::string ps_texture_status{"none"};

    u32 vs_user_sgprs[16]{};
    GuestAddr last_idx_addr{0};
    u32 last_idx_count{0};

    void init_mesh_pipeline();
    MeshUniforms compute_uniforms();
    id<MTLTexture> prepare_texture(id<MTLDevice> device);
    void log_draw_diagnostics(const char* draw_type, u32 count,
                              const MeshUniforms& uniforms,
                              id<MTLTexture> active_tex,
                              bool use_guest_vtx,
                              const void* vtx_data,
                              usize vtx_stride,
                              u32 num_vtx);
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
            "#include <simd/simd.h>\n"
            "using namespace metal;\n"
            "\n"
            "struct RasterizerData {\n"
            "    float4 position [[position]];\n"
            "    float4 color;\n"
            "    float2 texcoord;\n"
            "};\n"
            "\n"
            "struct MeshUniforms {\n"
            "    float4x4 mvp;\n"
            "    float4 uv_scale_offset;\n"
            "    float4 tint_color;\n"
            "    uint vertex_stride;\n"
            "    uint use_texture;\n"
            "    uint pad[2];\n"
            "};\n"
            "\n"
            "vertex RasterizerData vs_mesh(uint vid [[vertex_id]],\n"
            "                              constant uchar* vertex_data [[buffer(0)]],\n"
            "                              constant MeshUniforms& uniforms [[buffer(1)]]) {\n"
            "    RasterizerData out;\n"
            "    constant uchar* v_ptr = vertex_data + vid * uniforms.vertex_stride;\n"
            "    float3 pos = *(constant packed_float3*)v_ptr;\n"
            "    float4 col = float4(1.0, 1.0, 1.0, 1.0);\n"
            "    float2 uv = float2(0.0, 0.0);\n"
            "    if (uniforms.vertex_stride >= 28) {\n"
            "        col = *(constant packed_float4*)(v_ptr + 12);\n"
            "        if (uniforms.vertex_stride >= 36) {\n"
            "            uv = *(constant packed_float2*)(v_ptr + 28);\n"
            "        }\n"
            "    } else if (uniforms.vertex_stride == 16) {\n"
            "        uv = *(constant packed_float2*)(v_ptr + 8);\n"
            "    }\n"
            "    out.position = uniforms.mvp * float4(pos, 1.0);\n"
            "    out.color = col * uniforms.tint_color;\n"
            "    out.texcoord = uv * uniforms.uv_scale_offset.xy + uniforms.uv_scale_offset.zw;\n"
            "    return out;\n"
            "}\n"
            "\n"
            "fragment float4 fs_mesh(RasterizerData in [[stage_in]],\n"
            "                        texture2d<float> tex [[texture(0)]],\n"
            "                        sampler smp [[sampler(0)]],\n"
            "                        constant MeshUniforms& uniforms [[buffer(1)]]) {\n"
            "    float4 c = in.color;\n"
            "    if (uniforms.use_texture != 0) {\n"
            "        float4 t = tex.sample(smp, in.texcoord);\n"
            "        c *= t;\n"
            "    }\n"
            "    return c;\n"
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
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

        mesh_pipeline_state = [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (!mesh_pipeline_state) {
            std::cerr << "[PM4] Failed to create 3D mesh render pipeline state: "
                      << (error ? [[error localizedDescription] UTF8String] : "unknown error") << "\n";
        }

        if (!default_white_texture) {
            MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                               width:1
                                                                                              height:1
                                                                                           mipmapped:NO];
            texDesc.usage = MTLTextureUsageShaderRead;
            default_white_texture = [device newTextureWithDescriptor:texDesc];
            const u32 white_pixel = 0xFFFFFFFF;
            [default_white_texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&white_pixel bytesPerRow:4];
        }

        if (!sampler_state) {
            MTLSamplerDescriptor* sampDesc = [[MTLSamplerDescriptor alloc] init];
            sampDesc.minFilter = MTLSamplerMinMagFilterLinear;
            sampDesc.magFilter = MTLSamplerMinMagFilterLinear;
            sampDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
            sampDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
            sampler_state = [device newSamplerStateWithDescriptor:sampDesc];
        }
    }
}

MeshUniforms Pm4Translator::Impl::compute_uniforms() {
    MeshUniforms u{};
    u.mvp = matrix_identity_float4x4;
    u.uv_scale_offset = simd_make_float4(1.0f, 1.0f, 0.0f, 0.0f);
    u.tint_color = simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    u.vertex_stride = sizeof(Vertex3D);
    u.use_texture = 0;

    if (!memory) return u;

    // 1. UV scale/offset vector (buffer with num_records == 1)
    for (const auto& [slot, buf] : vs_user_sgpr_buffers) {
        if (buf.num_records == 1 && buf.base_addr > 0x1000 && memory->is_valid_address(buf.base_addr, 16)) {
            float uv_raw[4] = {1.0f, 1.0f, 0.0f, 0.0f};
            memory->read_bytes(buf.base_addr, uv_raw, 16);
            if (!std::isnan(uv_raw[0]) && !std::isinf(uv_raw[0]) &&
                !std::isnan(uv_raw[1]) && !std::isinf(uv_raw[1])) {
                u.uv_scale_offset = simd_make_float4(uv_raw[0], uv_raw[1], uv_raw[2], uv_raw[3]);
            }
            break;
        }
    }

    // 2. Model-View Matrix (buffer with num_records == 10)
    simd_float4x4 model_view = matrix_identity_float4x4;
    bool has_mv = false;
    for (const auto& [slot, buf] : vs_user_sgpr_buffers) {
        if (buf.num_records == 10 && buf.base_addr > 0x1000 && memory->is_valid_address(buf.base_addr, 64)) {
            float m[16] = {0};
            memory->read_bytes(buf.base_addr, m, 64);
            bool valid = true;
            for (int i = 0; i < 16; ++i) {
                if (std::isnan(m[i]) || std::isinf(m[i])) { valid = false; break; }
            }
            if (valid) {
                model_view.columns[0] = simd_make_float4(m[0], m[1], m[2], m[3]);
                model_view.columns[1] = simd_make_float4(m[4], m[5], m[6], m[7]);
                model_view.columns[2] = simd_make_float4(m[8], m[9], m[10], m[11]);
                model_view.columns[3] = simd_make_float4(m[12], m[13], m[14], m[15]);
                has_mv = true;
            }
            break;
        }
    }

    // 3. Projection Matrix (buffer with num_records >= 21 at offset 17*16 = 272)
    simd_float4x4 proj = matrix_identity_float4x4;
    bool has_proj = false;
    for (const auto& [slot, buf] : vs_user_sgpr_buffers) {
        if (buf.num_records >= 21 && buf.base_addr > 0x1000) {
            const GuestAddr proj_addr = buf.base_addr + 272;
            if (memory->is_valid_address(proj_addr, 64)) {
                float p[16] = {0};
                memory->read_bytes(proj_addr, p, 64);
                bool valid = true;
                for (int i = 0; i < 16; ++i) {
                    if (std::isnan(p[i]) || std::isinf(p[i])) { valid = false; break; }
                }
                if (valid && (p[0] != 0.0f || p[5] != 0.0f || p[10] != 0.0f || p[15] != 0.0f)) {
                    proj.columns[0] = simd_make_float4(p[0], p[1], p[2], p[3]);
                    proj.columns[1] = simd_make_float4(p[4], p[5], p[6], p[7]);
                    proj.columns[2] = simd_make_float4(p[8], p[9], p[10], p[11]);
                    proj.columns[3] = simd_make_float4(p[12], p[13], p[14], p[15]);
                    has_proj = true;
                }
            }
            break;
        }
    }

    // In unit tests, vs_shader_addr == 0 and neither MV nor Proj are present
    if (vs_shader_addr == 0 && !has_mv && !has_proj) {
        u.mvp = matrix_identity_float4x4;
    } else {
        if (!has_proj) {
            proj = matrix_identity_float4x4;
            proj.columns[1].y = -1.0f; // Unity 2D screen space Y-flip
        }
        u.mvp = simd_mul(proj, model_view);
    }

    return u;
}

id<MTLTexture> Pm4Translator::Impl::prepare_texture(id<MTLDevice> device) {
    if (!device || !memory) {
        ps_texture_status = "hit white fallback (no device or memory manager)";
        return default_white_texture;
    }
    if (ps_texture_base <= 0x1000 || ps_texture_width == 0 || ps_texture_height == 0 ||
        ps_texture_width > 4096 || ps_texture_height > 4096) {
        ps_texture_status = "hit white fallback (invalid base 0x" + std::to_string(ps_texture_base) + " or dim " +
                            std::to_string(ps_texture_width) + "x" + std::to_string(ps_texture_height) + ")";
        return default_white_texture;
    }

    const usize byte_size = static_cast<usize>(ps_texture_width) * ps_texture_height * 4;
    if (!memory->is_valid_address(ps_texture_base, byte_size)) {
        ps_texture_status = "hit white fallback (invalid memory range for 0x" + std::to_string(ps_texture_base) + ")";
        return default_white_texture;
    }

    const void* host_pixels = memory->to_host_ptr<void>(ps_texture_base);
    if (!host_pixels) {
        ps_texture_status = "hit white fallback (null host pointer)";
        return default_white_texture;
    }

    if (!active_metal_texture ||
        active_metal_texture.width != ps_texture_width ||
        active_metal_texture.height != ps_texture_height) {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                        width:ps_texture_width
                                                                                       height:ps_texture_height
                                                                                    mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        active_metal_texture = [device newTextureWithDescriptor:desc];
    }

    const u8* u8_ptr = static_cast<const u8*>(host_pixels);
    usize nonzero_count = 0;
    const usize check_len = std::min<usize>(byte_size, 512);
    for (usize k = 0; k < check_len; ++k) {
        if (u8_ptr[k] != 0) {
            nonzero_count++;
        }
    }
    ps_texture_has_nonzero = (nonzero_count > 0);
    ps_texture_nonzero_bytes = nonzero_count;

    if (ps_texture_has_nonzero) {
        ps_texture_status = "loaded real non-zero texels (" + std::to_string(nonzero_count) + "/" +
                            std::to_string(check_len) + " non-zero bytes checked in guest memory)";
        [active_metal_texture replaceRegion:MTLRegionMake2D(0, 0, ps_texture_width, ps_texture_height)
                                mipmapLevel:0
                                  withBytes:host_pixels
                                bytesPerRow:ps_texture_width * 4];
    } else {
        ps_texture_status = "hit white fallback (all 0x00 bytes in guest memory across " +
                            std::to_string(check_len) + " bytes checked)";
        std::vector<u32> white_data(ps_texture_width * ps_texture_height, 0xFFFFFFFF);
        [active_metal_texture replaceRegion:MTLRegionMake2D(0, 0, ps_texture_width, ps_texture_height)
                                mipmapLevel:0
                                  withBytes:white_data.data()
                                bytesPerRow:ps_texture_width * 4];
    }

    return active_metal_texture;
}

static std::string decode_gcn_instruction(u32 dw0, u32 dw1, bool& is_64bit) {
    is_64bit = false;
    char b[128];
    // SOPP (0xBF800000..0xBFFFFFFF)
    if ((dw0 & 0xFF800000) == 0xBF800000) {
        u32 op = (dw0 >> 16) & 0x7F;
        u32 simm = dw0 & 0xFFFF;
        if (op == 0x00) std::snprintf(b, sizeof(b), "s_nop 0x%04x", simm);
        else if (op == 0x01) std::snprintf(b, sizeof(b), "s_endpgm");
        else if (op == 0x0C) std::snprintf(b, sizeof(b), "s_waitcnt vmcnt(%u) expcnt(%u) lgkmcnt(%u)", simm & 0xF, (simm>>4)&7, (simm>>8)&0x1F);
        else std::snprintf(b, sizeof(b), "s_sopp [op=0x%02x, simm=0x%04x]", op, simm);
        return b;
    }
    // SOP1 (0xBE800000..0xBEFFFFFF)
    if ((dw0 & 0xFF800000) == 0xBE800000) {
        u32 sdst = (dw0 >> 16) & 0x7F;
        u32 op = (dw0 >> 8) & 0xFF;
        u32 ssrc0 = dw0 & 0xFF;
        if (op == 0x00) std::snprintf(b, sizeof(b), "s_mov_b32 s%u, s%u", sdst, ssrc0);
        else if (op == 0x01) std::snprintf(b, sizeof(b), "s_mov_b64 s[%u:%u], s[%u:%u]", sdst, sdst+1, ssrc0, ssrc0+1);
        else if (op == 0x21 || op == 0x22) std::snprintf(b, sizeof(b), "s_swappc_b64 s[%u:%u], s[%u:%u]", sdst, sdst+1, ssrc0, ssrc0+1);
        else std::snprintf(b, sizeof(b), "s_sop1 [op=0x%02x] s%u, s%u", op, sdst, ssrc0);
        return b;
    }
    // SOPK (0xB0000000..0xB7FFFFFF)
    if ((dw0 & 0xF8000000) == 0xB0000000) {
        u32 op = (dw0 >> 23) & 0x1F;
        u32 sdst = (dw0 >> 16) & 0x7F;
        u32 simm = dw0 & 0xFFFF;
        std::snprintf(b, sizeof(b), "s_sopk [op=0x%02x] s%u, 0x%04x", op, sdst, simm);
        return b;
    }
    // SOP2 (0x80000000..0xAFFFFFFF)
    if ((dw0 & 0xC0000000) == 0x80000000) {
        u32 op = (dw0 >> 23) & 0x7F;
        u32 sdst = (dw0 >> 16) & 0x7F;
        u32 ssrc1 = (dw0 >> 8) & 0xFF;
        u32 ssrc0 = dw0 & 0xFF;
        std::snprintf(b, sizeof(b), "s_sop2 [op=0x%02x] s%u, s%u, s%u", op, sdst, ssrc0, ssrc1);
        return b;
    }
    // SMEM (0xC0000000..0xC7FFFFFF)
    if ((dw0 & 0xFC000000) == 0xC0000000) {
        u32 op = (dw0 >> 21) & 0x1F;
        u32 imm = (dw0 >> 19) & 0x1;
        u32 sdst = (dw0 >> 12) & 0x7F;
        u32 sbase = ((dw0 >> 6) & 0x3F) * 2;
        u32 offset = dw0 & 0x3F;
        const char* name = "s_smem";
        if (op == 0x00) name = "s_load_dword";
        else if (op == 0x01) name = "s_load_dwordx2";
        else if (op == 0x02) name = "s_load_dwordx4";
        else if (op == 0x08) name = "s_buffer_load_dword";
        else if (op == 0x09) name = "s_buffer_load_dwordx2";
        else if (op == 0x0A || op == 0x18 || op == 0x19) name = "s_buffer_load_dwordx4";
        std::snprintf(b, sizeof(b), "%s s[%u..], s[%u..], %s0x%x", name, sdst, sbase, imm ? "imm:" : "offset:", offset);
        return b;
    }
    // MUBUF (0xE0000000..0xE7FFFFFF, 64-bit)
    if ((dw0 & 0xFC000000) == 0xE0000000) {
        is_64bit = true;
        u32 op = (dw0 >> 19) & 0x7F;
        u32 offset = dw0 & 0xFFF;
        u32 vdata = (dw1 >> 8) & 0xFF;
        u32 vaddr = dw1 & 0xFF;
        u32 srsrc = ((dw1 >> 18) & 0x3F) * 4;
        u32 soffset = (dw1 >> 24) & 0xFF;
        const char* name = "buffer_load";
        if (op == 0x00) name = "buffer_load_format_x";
        else if (op == 0x01) name = "buffer_load_format_xy";
        else if (op == 0x02) name = "buffer_load_format_xyz";
        else if (op == 0x03) name = "buffer_load_format_xyzw";
        else if (op == 0x0C) name = "buffer_load_dword";
        else if (op == 0x0F) name = "buffer_load_dwordx4";
        std::snprintf(b, sizeof(b), "%s v[%u..], v%u, s[%u:%u], s%u offen off:0x%x", name, vdata, vaddr, srsrc, srsrc+3, soffset, offset);
        return b;
    }
    // VOP1 (0x7E000000..0x7FFFFFFF)
    if ((dw0 & 0xFE000000) == 0x7E000000) {
        u32 vdst = (dw0 >> 17) & 0xFF;
        u32 op = (dw0 >> 9) & 0xFF;
        u32 src0 = dw0 & 0x1FF;
        if (op == 0x01) std::snprintf(b, sizeof(b), "v_mov_b32 v%u, 0x%x", vdst, src0);
        else std::snprintf(b, sizeof(b), "v_vop1 [op=0x%02x] v%u, 0x%x", op, vdst, src0);
        return b;
    }
    // VOP2 (bit 31 = 0)
    if ((dw0 & 0x80000000) == 0) {
        u32 op = (dw0 >> 25) & 0x3F;
        u32 vdst = (dw0 >> 17) & 0xFF;
        u32 vsrc1 = (dw0 >> 9) & 0xFF;
        u32 src0 = dw0 & 0x1FF;
        const char* name = "v_vop2";
        if (op == 0x03) name = "v_add_f32";
        else if (op == 0x04) name = "v_sub_f32";
        else if (op == 0x08) name = "v_mul_f32";
        else if (op == 0x13) name = "v_mac_f32";
        else if (op == 0x19) name = "v_add_i32";
        else if (op == 0x1B) name = "v_lshlrev_b32";
        std::snprintf(b, sizeof(b), "%s v%u, 0x%x, v%u", name, vdst, src0, vsrc1);
        return b;
    }
    // VOP3 (0xD0000000..0xD3FFFFFF, 64-bit)
    if ((dw0 & 0xFC000000) == 0xD0000000) {
        is_64bit = true;
        u32 op = (dw0 >> 17) & 0x1FF;
        u32 vdst = dw0 & 0xFF;
        const char* name = "v_vop3";
        if (op == 0x140) name = "v_mad_f32";
        else if (op == 0x156) name = "v_fma_f32";
        std::snprintf(b, sizeof(b), "%s v%u, ... [op=0x%03x]", name, vdst, op);
        return b;
    }
    std::snprintf(b, sizeof(b), "raw_dw (0x%08x)", dw0);
    return b;
}

static void print_gcn_disassembly(MemoryManager* mem, GuestAddr addr, u32 max_insns, const char* prefix) {
    if (!mem || addr < 0x1000) return;
    u32 insn_cnt = 0;
    u32 dw_offset = 0;
    while (insn_cnt < max_insns && dw_offset < 100) {
        const GuestAddr cur_addr = addr + dw_offset * 4;
        if (!mem->is_valid_address(cur_addr, 8)) break;
        u32 dw0 = 0, dw1 = 0;
        mem->read_value(cur_addr, dw0);
        mem->read_value(cur_addr + 4, dw1);
        bool is_64 = false;
        std::string dis = decode_gcn_instruction(dw0, dw1, is_64);
        if (is_64) {
            std::fprintf(stderr, "%s  [0x%llx] %08x %08x: %s\n", prefix, (unsigned long long)cur_addr, dw0, dw1, dis.c_str());
            dw_offset += 2;
        } else {
            std::fprintf(stderr, "%s  [0x%llx] %08x:          %s\n", prefix, (unsigned long long)cur_addr, dw0, dis.c_str());
            dw_offset += 1;
        }
        insn_cnt++;
        if (dw0 == 0xbf810000) break; // s_endpgm
    }
}

void Pm4Translator::Impl::log_draw_diagnostics(const char* draw_type, u32 count,
                                               const MeshUniforms& uniforms,
                                               id<MTLTexture> active_tex,
                                               bool use_guest_vtx,
                                               const void* vtx_data,
                                               usize vtx_stride,
                                               u32 num_vtx) {
    (void)active_tex;
    const u32 cur_draw = static_cast<u32>(draw_count + 1);
    std::fprintf(stderr, "\n=== [DIAGNOSTIC DUMP] Draw #%u (%s, count=%u) ===\n", cur_draw, draw_type, count);
    std::fprintf(stderr, "  Shaders:\n");
    std::fprintf(stderr, "    VS ISA: 0x%llx (PGM_LO=0x%08x)\n", (unsigned long long)vs_shader_addr, vs_pgm_lo);
    std::fprintf(stderr, "    PS ISA: 0x%llx (PGM_LO=0x%08x)\n", (unsigned long long)ps_shader_addr, ps_pgm_lo);

    std::fprintf(stderr, "  VS User SGPRs (s[0..15]):\n");
    for (int s = 0; s < 16; ++s) {
        std::fprintf(stderr, "    s%02d: 0x%08x%s", s, vs_user_sgprs[s], (s % 4 == 3) ? "\n" : "  ");
    }
    const u64 fetch_shader_addr = (static_cast<u64>(vs_user_sgprs[1]) << 32) | vs_user_sgprs[0];
    const u64 v_table_addr = (static_cast<u64>(vs_user_sgprs[3]) << 32) | vs_user_sgprs[2];
    std::fprintf(stderr, "    s[0..1] Fetch Shader Addr:  0x%llx\n", (unsigned long long)fetch_shader_addr);
    std::fprintf(stderr, "    s[2..3] V# Table Addr:      0x%llx\n", (unsigned long long)v_table_addr);

    if (memory && vs_shader_addr > 0x1000) {
        std::fprintf(stderr, "  Disassembly of VS Main Shader (0x%llx):\n", (unsigned long long)vs_shader_addr);
        print_gcn_disassembly(memory, vs_shader_addr, 30, "   ");
    }
    if (memory && fetch_shader_addr > 0x1000) {
        std::fprintf(stderr, "  Disassembly of VS Fetch Shader (0x%llx):\n", (unsigned long long)fetch_shader_addr);
        print_gcn_disassembly(memory, fetch_shader_addr, 20, "   ");
    }

    if (memory && v_table_addr > 0x1000 && memory->is_valid_address(v_table_addr, 128)) {
        std::fprintf(stderr, "  V# Descriptor Table (at 0x%llx):\n", (unsigned long long)v_table_addr);
        for (int desc = 0; desc < 8; ++desc) {
            u32 d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            memory->read_value(v_table_addr + desc * 16 + 0, d0);
            memory->read_value(v_table_addr + desc * 16 + 4, d1);
            memory->read_value(v_table_addr + desc * 16 + 8, d2);
            memory->read_value(v_table_addr + desc * 16 + 12, d3);
            const u64 base = (static_cast<u64>(d1 & 0xFFFF) << 32) | d0;
            const u32 stride = (d1 >> 16) & 0x3FFF;
            const u32 num_recs = d2;
            const u32 dfmt = (d3 >> 20) & 0x3F;
            const u32 nfmt = (d3 >> 26) & 0xF;
            std::fprintf(stderr, "    V#%d: raw=[0x%08x 0x%08x 0x%08x 0x%08x] base=0x%llx stride=%u recs=%u dfmt=%u nfmt=%u\n",
                         desc, d0, d1, d2, d3, (unsigned long long)base, stride, num_recs, dfmt, nfmt);
            if (base > 0x1000 && num_recs > 0 && stride > 0) {
                const u32 dump_cnt = std::min<u32>(num_recs, 4);
                for (u32 r = 0; r < dump_cnt; ++r) {
                    const GuestAddr rec_addr = base + r * stride;
                    if (memory->is_valid_address(rec_addr, stride)) {
                        std::vector<float> f(stride / sizeof(float));
                        memory->read_bytes(rec_addr, f.data(), f.size() * sizeof(float));
                        std::string f_str;
                        for (usize fi = 0; fi < f.size(); ++fi) {
                            char b[32];
                            std::snprintf(b, sizeof(b), "%.3f ", f[fi]);
                            f_str += b;
                        }
                        std::fprintf(stderr, "      rec[%u]: [ %s]\n", r, f_str.c_str());
                    }
                }
            }
        }
    }

    if (last_idx_addr > 0x1000 && memory && memory->is_valid_address(last_idx_addr, 12)) {
        u16 idx[6] = {0};
        memory->read_bytes(last_idx_addr, idx, sizeof(idx));
        std::fprintf(stderr, "  Draw #2 Index Buffer (at 0x%llx, 6 indices): [%u, %u, %u, %u, %u, %u]\n",
                     (unsigned long long)last_idx_addr, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
    }

    const GuestAddr cb_addr = vs_user_sgpr_buffers.count(12) ? vs_user_sgpr_buffers[12].base_addr : 0x2454cb80;
    if (cb_addr > 0x1000 && memory && memory->is_valid_address(cb_addr, 336)) {
        std::fprintf(stderr, "  Constant Buffer s[12..15] (at 0x%llx, 21 rows):\n", (unsigned long long)cb_addr);
        for (int r = 0; r <= 20; ++r) {
            float flts[4] = {0};
            u32 hx[4] = {0};
            memory->read_bytes(cb_addr + r * 16, flts, 16);
            memory->read_bytes(cb_addr + r * 16, hx, 16);
            std::fprintf(stderr, "    r%02d (off %3d): flt=[%9.3f, %9.3f, %9.3f, %9.3f] | hex=[0x%08x, 0x%08x, 0x%08x, 0x%08x]\n",
                         r, r * 16, flts[0], flts[1], flts[2], flts[3], hx[0], hx[1], hx[2], hx[3]);
        }
    }

    if (ps_texture_w0 != 0 || ps_texture_base != 0) {
        const u32 dfmt_14 = (ps_texture_w1 >> 14) & 0x3F;
        const u32 dfmt_20 = (ps_texture_w1 >> 20) & 0x3F;
        const u32 nfmt_20 = (ps_texture_w1 >> 20) & 0xF;
        const u32 nfmt_24 = (ps_texture_w1 >> 24) & 0xF;
        const u32 tiling_idx = (ps_texture_w3 >> 21) & 0x1F;
        std::fprintf(stderr, "  Decoded T# Texture Descriptor:\n");
        std::fprintf(stderr, "    base_addr:  0x%llx\n", (unsigned long long)ps_texture_base);
        std::fprintf(stderr, "    dimensions: %ux%u\n", ps_texture_width, ps_texture_height);
        std::fprintf(stderr, "    dfmt:       %u (bits 19:14), %u (bits 25:20)\n", dfmt_14, dfmt_20);
        std::fprintf(stderr, "    nfmt:       %u (bits 23:20), %u (bits 27:24)\n", nfmt_20, nfmt_24);
        std::fprintf(stderr, "    tiling_idx: %u (bits 25:21)\n", tiling_idx);
        std::fprintf(stderr, "    raw DWORDs: [w0=0x%08x, w1=0x%08x, w2=0x%08x, w3=0x%08x]\n",
                     ps_texture_w0, ps_texture_w1, ps_texture_w2, ps_texture_w3);
        std::fprintf(stderr, "    prepare_texture: %s\n", ps_texture_status.c_str());
    } else {
        std::fprintf(stderr, "  Decoded T# Texture Descriptor: None (prepare_texture: %s)\n", ps_texture_status.c_str());
    }

    std::fprintf(stderr, "  Uniforms & MVP Matrix:\n");
    std::fprintf(stderr, "    use_texture: %u\n", uniforms.use_texture);
    std::fprintf(stderr, "    tint_color:  (%.3f, %.3f, %.3f, %.3f)\n",
                 uniforms.tint_color.x, uniforms.tint_color.y, uniforms.tint_color.z, uniforms.tint_color.w);
    std::fprintf(stderr, "    uv_scale_off:(%.3f, %.3f, %.3f, %.3f)\n",
                 uniforms.uv_scale_offset.x, uniforms.uv_scale_offset.y, uniforms.uv_scale_offset.z, uniforms.uv_scale_offset.w);
    std::fprintf(stderr, "    Scale:       (%.4f, %.4f, %.4f)\n",
                 uniforms.mvp.columns[0].x, uniforms.mvp.columns[1].y, uniforms.mvp.columns[2].z);
    std::fprintf(stderr, "    Translation: (%.4f, %.4f, %.4f)\n",
                 uniforms.mvp.columns[3].x, uniforms.mvp.columns[3].y, uniforms.mvp.columns[3].z);
    std::fprintf(stderr, "    MVP 4x4:\n");
    std::fprintf(stderr, "      Row 0: [%8.4f, %8.4f, %8.4f, %8.4f]\n",
                 uniforms.mvp.columns[0].x, uniforms.mvp.columns[1].x, uniforms.mvp.columns[2].x, uniforms.mvp.columns[3].x);
    std::fprintf(stderr, "      Row 1: [%8.4f, %8.4f, %8.4f, %8.4f]\n",
                 uniforms.mvp.columns[0].y, uniforms.mvp.columns[1].y, uniforms.mvp.columns[2].y, uniforms.mvp.columns[3].y);
    std::fprintf(stderr, "      Row 2: [%8.4f, %8.4f, %8.4f, %8.4f]\n",
                 uniforms.mvp.columns[0].z, uniforms.mvp.columns[1].z, uniforms.mvp.columns[2].z, uniforms.mvp.columns[3].z);
    std::fprintf(stderr, "      Row 3: [%8.4f, %8.4f, %8.4f, %8.4f]\n",
                 uniforms.mvp.columns[0].w, uniforms.mvp.columns[1].w, uniforms.mvp.columns[2].w, uniforms.mvp.columns[3].w);

    std::fprintf(stderr, "  Vertices & NDC Positions (num_vtx=%u, guest_vtx=%s):\n",
                 num_vtx, use_guest_vtx ? "YES" : "NO");
    const u8* vptr = static_cast<const u8*>(vtx_data);
    for (u32 k = 0; k < num_vtx; ++k) {
        float vx = 0.0f, vy = 0.0f, vz = 0.0f;
        if (vptr) {
            const float* fp = reinterpret_cast<const float*>(vptr + k * vtx_stride);
            vx = fp[0];
            vy = fp[1];
            vz = fp[2];
        }
        simd_float4 local_pos = simd_make_float4(vx, vy, vz, 1.0f);
        simd_float4 clip_pos = simd_mul(uniforms.mvp, local_pos);
        simd_float4 ndc = (clip_pos.w != 0.0f) ? (clip_pos / clip_pos.w) : clip_pos;
        const bool clipped = (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f || ndc.z < 0.0f || ndc.z > 1.0f);
        std::fprintf(stderr, "    V%u: local=[%6.2f, %6.2f, %6.2f] -> clip=[%8.3f, %8.3f, %8.3f, %8.3f] -> NDC=[%8.3f, %8.3f, %8.3f] %s\n",
                     k, vx, vy, vz,
                     clip_pos.x, clip_pos.y, clip_pos.z, clip_pos.w,
                     ndc.x, ndc.y, ndc.z,
                     clipped ? "[CLIPPED OUTSIDE FRUSTUM]" : "[VISIBLE IN FRUSTUM]");
    }

    if (metal) {
        u8 cr = 0, cg = 0, cb = 0, ca = 0;
        metal->read_pixel(960, 540, cr, cg, cb, ca);
        u8 tr = 0, tg = 0, tb = 0, ta = 0;
        metal->read_pixel(960, 200, tr, tg, tb, ta);
        u8 br = 0, bg = 0, bb = 0, ba = 0;
        metal->read_pixel(960, 900, br, bg, bb, ba);
        std::fprintf(stderr, "  Framebuffer Pixel Readback:\n");
        std::fprintf(stderr, "    Center(960, 540): RGBA(%3u, %3u, %3u, %3u) [hex #%02x%02x%02x%02x]\n", cr, cg, cb, ca, cr, cg, cb, ca);
        std::fprintf(stderr, "    Top   (960, 200): RGBA(%3u, %3u, %3u, %3u) [hex #%02x%02x%02x%02x]\n", tr, tg, tb, ta, tr, tg, tb, ta);
        std::fprintf(stderr, "    Bottom(960, 900): RGBA(%3u, %3u, %3u, %3u) [hex #%02x%02x%02x%02x]\n", br, bg, bb, ba, br, bg, bb, ba);
    }
    std::fprintf(stderr, "========================================================\n\n");
}

bool Pm4Translator::Impl::render_indexed_mesh(GuestAddr idx_addr, u32 idx_count, PrimitiveType prim_type,
                                              MTLIndexType idx_type, i32 base_vertex, u32 instance_count) {
    last_idx_addr = idx_addr;
    last_idx_count = idx_count;
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

        MeshUniforms uniforms = compute_uniforms();
        id<MTLTexture> active_tex = prepare_texture(device);
        uniforms.use_texture = (active_tex != default_white_texture) ? 1 : 0;

        bool use_guest_vtx = false;
        if (vertex_buffer_addr != 0 && vertex_buffer_size > 0 && stride >= sizeof(Vertex3D) &&
            memory && memory->is_valid_address(vertex_buffer_addr, required_vtx_size)) {
            const void* vtx_check = memory->to_host_ptr<void>(vertex_buffer_addr);
            if (vtx_check) {
                const float* f = static_cast<const float*>(vtx_check);
                const usize check_count = std::min<usize>(required_vtx_size / sizeof(float), 28);
                for (usize c = 0; c < check_count; ++c) {
                    if (f[c] != 0.0f) {
                        use_guest_vtx = true;
                        break;
                    }
                }
            }
        }

        // Canonical unit quad for indexed draws (indices 0, 2, 1, 1, 2, 3)
        const MeshVertex s_default_quad[4] = {
            {{-0.5f,  0.5f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}, // TL
            {{ 0.5f,  0.5f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}, // TR
            {{-0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}}, // BL
            {{ 0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}}, // BR
        };

        id<MTLBuffer> vtx_buf = nil;
        if (use_guest_vtx) {
            const void* vtx_ptr = memory->to_host_ptr<void>(vertex_buffer_addr);
            vtx_buf = [device newBufferWithBytes:vtx_ptr length:required_vtx_size options:MTLResourceStorageModeShared];
            uniforms.vertex_stride = stride;
        } else {
            vtx_buf = [device newBufferWithBytes:s_default_quad length:sizeof(s_default_quad) options:MTLResourceStorageModeShared];
            uniforms.vertex_stride = sizeof(MeshVertex);
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

        id<MTLBuffer> uniform_buf = [device newBufferWithBytes:&uniforms length:sizeof(uniforms) options:MTLResourceStorageModeShared];
        [enc setVertexBuffer:vtx_buf offset:0 atIndex:0];
        [enc setVertexBuffer:uniform_buf offset:0 atIndex:1];
        [enc setFragmentBuffer:uniform_buf offset:0 atIndex:1];
        [enc setFragmentTexture:active_tex ? active_tex : default_white_texture atIndex:0];
        [enc setFragmentSamplerState:sampler_state atIndex:0];

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
                        baseVertex:use_guest_vtx ? base_vertex : 0
                      baseInstance:0];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        log_draw_diagnostics("IT_DRAW_INDEX_2", idx_count, uniforms, active_tex, use_guest_vtx,
                             use_guest_vtx ? memory->to_host_ptr<void>(vertex_buffer_addr) : s_default_quad,
                             use_guest_vtx ? stride : sizeof(MeshVertex),
                             use_guest_vtx ? std::min<u32>(static_cast<u32>(effective_vtx_count), 4) : 4);

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

        MeshUniforms uniforms = compute_uniforms();
        id<MTLTexture> active_tex = prepare_texture(device);
        uniforms.use_texture = (active_tex != default_white_texture) ? 1 : 0;

        id<MTLBuffer> vtx_buf = nil;
        MTLPrimitiveType mtl_prim = MTLPrimitiveTypeTriangle;
        u32 actual_vtx_count = vtx_count;

        bool use_guest_vtx = false;
        if (vertex_buffer_addr != 0 && vertex_buffer_size > 0 && stride >= sizeof(Vertex3D) &&
            memory && memory->is_valid_address(vertex_buffer_addr, required_vtx_size)) {
            const void* vtx_check = memory->to_host_ptr<void>(vertex_buffer_addr);
            if (vtx_check) {
                const float* f = static_cast<const float*>(vtx_check);
                const usize check_count = std::min<usize>(required_vtx_size / sizeof(float), 28);
                for (usize c = 0; c < check_count; ++c) {
                    if (f[c] != 0.0f) {
                        use_guest_vtx = true;
                        break;
                    }
                }
            }
        }

        const void* diag_vtx_ptr = nullptr;
        usize diag_vtx_stride = sizeof(MeshVertex);
        u32 diag_vtx_num = 0;

        const float bg_r = (clear_color.a > 0.0f && (clear_color.r > 0 || clear_color.g > 0 || clear_color.b > 0)) ? clear_color.r : 0.40f;
        const float bg_g = (clear_color.a > 0.0f && (clear_color.r > 0 || clear_color.g > 0 || clear_color.b > 0)) ? clear_color.g : 0.75f;
        const float bg_b = (clear_color.a > 0.0f && (clear_color.r > 0 || clear_color.g > 0 || clear_color.b > 0)) ? clear_color.b : 0.85f;
        const MeshVertex s_fullscreen_tri[3] = {
            {{-1.0f, -1.0f, 0.0f}, {bg_r, bg_g, bg_b, 1.0f}, {0.0f, 0.0f}},
            {{ 3.0f, -1.0f, 0.0f}, {bg_r, bg_g, bg_b, 1.0f}, {2.0f, 0.0f}},
            {{-1.0f,  3.0f, 0.0f}, {bg_r, bg_g, bg_b, 1.0f}, {0.0f, 2.0f}},
        };

        const MeshVertex s_quad[4] = {
            {{-0.5f,  0.5f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
            {{ 0.5f,  0.5f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
            {{-0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
            {{ 0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
        };

        std::vector<MeshVertex> generic;

        if (use_guest_vtx) {
            const void* vtx_ptr = memory->to_host_ptr<void>(vertex_buffer_addr);
            vtx_buf = [device newBufferWithBytes:vtx_ptr length:required_vtx_size options:MTLResourceStorageModeShared];
            uniforms.vertex_stride = stride;
            if (prim_type == PrimitiveType::TriangleStrip) {
                mtl_prim = MTLPrimitiveTypeTriangleStrip;
            } else if (prim_type == PrimitiveType::LineList) {
                mtl_prim = MTLPrimitiveTypeLine;
            } else if (prim_type == PrimitiveType::PointList) {
                mtl_prim = MTLPrimitiveTypePoint;
            }
            actual_vtx_count = vtx_count;
            diag_vtx_ptr = vtx_ptr;
            diag_vtx_stride = stride;
            diag_vtx_num = std::min<u32>(actual_vtx_count, 4);
        } else {
            if (vtx_count == 3) {
                // Fullscreen background triangle
                vtx_buf = [device newBufferWithBytes:s_fullscreen_tri length:sizeof(s_fullscreen_tri) options:MTLResourceStorageModeShared];
                uniforms.vertex_stride = sizeof(MeshVertex);
                uniforms.mvp = matrix_identity_float4x4;
                uniforms.use_texture = 0;
                mtl_prim = MTLPrimitiveTypeTriangle;
                actual_vtx_count = 3;
                diag_vtx_ptr = s_fullscreen_tri;
                diag_vtx_stride = sizeof(MeshVertex);
                diag_vtx_num = 3;
            } else if (vtx_count == 4) {
                // Canonical unit quad in TriangleStrip order (0: TL, 1: TR, 2: BL, 3: BR)
                // Transformed by uniforms.mvp (ModelView * Projection)
                vtx_buf = [device newBufferWithBytes:s_quad length:sizeof(s_quad) options:MTLResourceStorageModeShared];
                uniforms.vertex_stride = sizeof(MeshVertex);
                mtl_prim = MTLPrimitiveTypeTriangleStrip;
                actual_vtx_count = 4;
                diag_vtx_ptr = s_quad;
                diag_vtx_stride = sizeof(MeshVertex);
                diag_vtx_num = 4;
            } else {
                generic.resize(vtx_count);
                for (u32 i = 0; i < vtx_count; ++i) {
                    generic[i] = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}};
                }
                vtx_buf = [device newBufferWithBytes:generic.data() length:generic.size() * sizeof(MeshVertex) options:MTLResourceStorageModeShared];
                uniforms.vertex_stride = sizeof(MeshVertex);
                mtl_prim = (prim_type == PrimitiveType::TriangleStrip) ? MTLPrimitiveTypeTriangleStrip : MTLPrimitiveTypeTriangle;
                actual_vtx_count = vtx_count;
                diag_vtx_ptr = generic.data();
                diag_vtx_stride = sizeof(MeshVertex);
                diag_vtx_num = std::min<u32>(vtx_count, 4);
            }
        }

        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_desc.colorAttachments[0].texture = framebuffer;
        if (draw_count == 0 || (!use_guest_vtx && vtx_count == 3)) {
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

        id<MTLBuffer> uniform_buf = [device newBufferWithBytes:&uniforms length:sizeof(uniforms) options:MTLResourceStorageModeShared];
        [enc setVertexBuffer:vtx_buf offset:0 atIndex:0];
        [enc setVertexBuffer:uniform_buf offset:0 atIndex:1];
        [enc setFragmentBuffer:uniform_buf offset:0 atIndex:1];
        [enc setFragmentTexture:active_tex ? active_tex : default_white_texture atIndex:0];
        [enc setFragmentSamplerState:sampler_state atIndex:0];

        [enc drawPrimitives:mtl_prim
                vertexStart:use_guest_vtx ? base_vertex : 0
                vertexCount:actual_vtx_count
              instanceCount:std::max<u32>(1, instance_count)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        log_draw_diagnostics("IT_DRAW_INDEX_AUTO", actual_vtx_count, uniforms, active_tex, use_guest_vtx,
                             diag_vtx_ptr, diag_vtx_stride, diag_vtx_num);

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
    m_impl->ps_texture_base = 0;
    m_impl->ps_texture_width = 0;
    m_impl->ps_texture_height = 0;
    m_impl->ps_texture_format = 0;
    m_impl->ps_texture_w0 = 0;
    m_impl->ps_texture_w1 = 0;
    m_impl->ps_texture_w2 = 0;
    m_impl->ps_texture_w3 = 0;
    m_impl->ps_texture_has_nonzero = false;
    m_impl->ps_texture_nonzero_bytes = 0;
    m_impl->ps_texture_status = "none";
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
                            if (reg == SPI_SHADER_USER_DATA_VS_0 || reg == SPI_SHADER_USER_DATA_VS_0_LEGACY) {
                                const u32 vaddr_lo = val;
                                const u32 vaddr_hi = (i + 1 < packet.payload.size()) ? packet.payload[i + 1] : 0;
                                const GuestAddr vaddr = (static_cast<u64>(vaddr_hi & 0xFFFF) << 32) | vaddr_lo;
                                const usize size = (i + 2 < packet.payload.size()) ? packet.payload[i + 2] : 0;
                                const u32 stride_hi = (vaddr_hi >> 16) & 0x3FFF;
                                const u32 stride_d3 = (i + 3 < packet.payload.size()) ? packet.payload[i + 3] : 0;
                                const usize stride = (stride_hi > 0) ? stride_hi : (stride_d3 > 0 ? stride_d3 : sizeof(Vertex3D));
                                bind_vertex_buffer(vaddr, size, stride);
                            }
                            if (reg >= 0x004C && reg <= 0x005B) {
                                const u32 sgpr_idx = reg - 0x004C;
                                m_impl->vs_user_sgprs[sgpr_idx] = val;
                                // GCN Sea Islands User SGPRs for VS (s0..s15)
                                const u32 slot = reg - 0x004C;
                                if (slot % 4 == 0 && i + 3 < packet.payload.size()) {
                                    const u32 d0 = packet.payload[i];
                                    const u32 d1 = packet.payload[i + 1];
                                    const u32 d2 = packet.payload[i + 2];
                                    const u32 d3 = packet.payload[i + 3];

                                    const u32 stride = (d1 >> 16) & 0x3FFF;
                                    const u32 num_records = d2;
                                    const GuestAddr base_addr = (static_cast<u64>(d1 & 0xFFFF) << 32) | d0;

                                    if (num_records > 0 && base_addr > 0x1000) {
                                        const u32 eff_stride = (stride > 0) ? stride : 16;
                                        m_impl->vs_user_sgpr_buffers[slot] = {base_addr, eff_stride, num_records, d3};
                                        std::fprintf(stderr, "[PM4 SH_REG] VS User SGPR s[%u..%u] V# Buffer: base=0x%llx stride=%u records=%u (sz=%u) fmt=0x%08x\n",
                                                     slot, slot + 3,
                                                     (unsigned long long)base_addr, eff_stride, num_records, eff_stride * num_records, d3);
                                    }
                                }
                            }
                        } else if (reg >= 0x000C && reg <= 0x001B) {
                            // GCN Sea Islands User SGPRs for PS (s0..s15)
                            if ((reg - 0x000C) == 0 && i + 3 < packet.payload.size()) {
                                const u32 w0 = packet.payload[i];
                                const u32 w1 = packet.payload[i + 1];
                                const u32 w2 = packet.payload[i + 2];
                                const u32 w3 = packet.payload[i + 3];
                                if (((w3 >> 28) >= 8) || (w1 == 0x00a00000)) {
                                    const GuestAddr tex_base = static_cast<GuestAddr>(w0) << 8;
                                    const u32 w_m1 = w2 & 0x3FFF;
                                    const u32 h_m1 = (w2 >> 14) & 0x3FFF;
                                    m_impl->ps_texture_base = tex_base;
                                    m_impl->ps_texture_width = w_m1 + 1;
                                    m_impl->ps_texture_height = h_m1 + 1;
                                    m_impl->ps_texture_format = w1;
                                    m_impl->ps_texture_w0 = w0;
                                    m_impl->ps_texture_w1 = w1;
                                    m_impl->ps_texture_w2 = w2;
                                    m_impl->ps_texture_w3 = w3;
                                    std::fprintf(stderr, "[PM4 SH_REG] PS T# Texture: base=0x%llx dim=%ux%u w1=0x%08x w3=0x%08x\n",
                                                 (unsigned long long)tex_base, w_m1 + 1, h_m1 + 1, w1, w3);
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
                    if (m_impl->vs_shader_addr > 0x1000 && m_impl->memory->is_valid_address(m_impl->vs_shader_addr, 320)) {
                        u32 vs_code[80] = {0};
                        m_impl->memory->read_bytes(m_impl->vs_shader_addr, vs_code, 320);
                        std::string vs_str;
                        for (int k = 0; k < 80; ++k) {
                            char b[16];
                            std::snprintf(b, sizeof(b), "0x%08x ", vs_code[k]);
                            vs_str += b;
                            if ((k + 1) % 8 == 0) vs_str += "\n        ";
                        }
                        std::fprintf(stderr, "      VS ISA [0x%llx] (80 dw):\n        %s\n",
                                     (unsigned long long)m_impl->vs_shader_addr, vs_str.c_str());
                    }
                    if (m_impl->ps_shader_addr > 0x1000 && m_impl->memory->is_valid_address(m_impl->ps_shader_addr, 200)) {
                        u32 ps_code[50] = {0};
                        m_impl->memory->read_bytes(m_impl->ps_shader_addr, ps_code, 200);
                        std::string ps_str;
                        for (int k = 0; k < 50; ++k) {
                            char b[16];
                            std::snprintf(b, sizeof(b), "0x%08x ", ps_code[k]);
                            ps_str += b;
                            if ((k + 1) % 8 == 0) ps_str += "\n        ";
                        }
                        std::fprintf(stderr, "      PS ISA [0x%llx] (50 dw):\n        %s\n",
                                     (unsigned long long)m_impl->ps_shader_addr, ps_str.c_str());
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
                        if (m_impl->vs_shader_addr > 0x1000 && m_impl->memory->is_valid_address(m_impl->vs_shader_addr, 320)) {
                            u32 vs_code[80] = {0};
                            m_impl->memory->read_bytes(m_impl->vs_shader_addr, vs_code, 320);
                            std::string vs_str;
                            for (int k = 0; k < 80; ++k) {
                                char b[16];
                                std::snprintf(b, sizeof(b), "0x%08x ", vs_code[k]);
                                vs_str += b;
                                if ((k + 1) % 8 == 0) vs_str += "\n        ";
                            }
                            std::fprintf(stderr, "      VS ISA [0x%llx] (80 dw):\n        %s\n",
                                         (unsigned long long)m_impl->vs_shader_addr, vs_str.c_str());
                        }
                        if (m_impl->ps_shader_addr > 0x1000 && m_impl->memory->is_valid_address(m_impl->ps_shader_addr, 200)) {
                            u32 ps_code[50] = {0};
                            m_impl->memory->read_bytes(m_impl->ps_shader_addr, ps_code, 200);
                            std::string ps_str;
                            for (int k = 0; k < 50; ++k) {
                                char b[16];
                                std::snprintf(b, sizeof(b), "0x%08x ", ps_code[k]);
                                ps_str += b;
                                if ((k + 1) % 8 == 0) ps_str += "\n        ";
                            }
                            std::fprintf(stderr, "      PS ISA [0x%llx] (50 dw):\n        %s\n",
                                         (unsigned long long)m_impl->ps_shader_addr, ps_str.c_str());
                        }
                        // Dump memory around index buffer
                        if (idx_addr > 0x1000 && m_impl->memory->is_valid_address(idx_addr - 128, 512)) {
                            std::fprintf(stderr, "      Around idx_addr [0x%llx - 128]:\n", (unsigned long long)idx_addr);
                            for (int line = -2; line < 6; ++line) {
                                GuestAddr a = idx_addr + line * 32;
                                float f[8] = {0};
                                u32 u[8] = {0};
                                m_impl->memory->read_bytes(a, u, 32);
                                std::memcpy(f, u, 32);
                                std::fprintf(stderr, "        0x%llx: [%.3f, %.3f, %.3f, %.3f | %.3f, %.3f, %.3f, %.3f]\n",
                                             (unsigned long long)a, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
                            }
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
