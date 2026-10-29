#import <Metal/Metal.h>
#include <simd/simd.h>
#include "ps4/memory.hpp"
#include "ps4/pm4_translator.hpp"
#include "ps4/gcn_pipeline.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <algorithm>

// PS4_LOG_PM4=1: trace every PM4 packet, draw and shader. Off by default:
// at 60 FPS it is megabytes per second, and when stderr is a pipe to a slow
// terminal the writes block the guest's GPU thread.
static bool pm4_verbose() {
    static const bool on = std::getenv("PS4_LOG_PM4") != nullptr;
    return on;
}
#define PM4_LOG(...) do { if (pm4_verbose()) std::fprintf(stderr, __VA_ARGS__); } while (0)

#include <set>
#include <tuple>
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

struct ShaderSignature {
    GuestAddr vertex_program{0};
    GuestAddr fragment_program{0};
    u32 vertex_resource_1{0};
    u32 vertex_resource_2{0};
    u32 fragment_resource_1{0};
    u32 fragment_resource_2{0};

    bool operator<(const ShaderSignature& other) const noexcept {
        return std::tie(vertex_program, fragment_program, vertex_resource_1, vertex_resource_2,
                        fragment_resource_1, fragment_resource_2) <
               std::tie(other.vertex_program, other.fragment_program, other.vertex_resource_1,
                        other.vertex_resource_2, other.fragment_resource_1, other.fragment_resource_2);
    }
};

struct Pm4Translator::Impl {
    // The host frame loop (begin_frame/submit_frame) and the guest thread that
    // submits command buffers both drive the same Metal encoder. Recursive
    // because indirect buffers re-enter execute_command_buffer.
    std::recursive_mutex frame_mutex;
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
    // PM4 command-processor state (owned by execute_packet).
    // Shadow of the GPU register file, indexed by absolute dword register
    // address (config 0x2000.., SH 0x2C00.., context 0xA000.., uconfig 0xC000..).
    std::vector<u32> reg_file = std::vector<u32>(0x10000, 0);
    // Constant-engine RAM (48 KiB on CIK) for LOAD/WRITE/DUMP_CONST_RAM.
    std::vector<u8> ce_ram = std::vector<u8>(48 * 1024, 0);
    // Dwords the stream walker must skip (COND_EXEC / PRED_EXEC false).
    usize pending_skip_dwords{0};
    // Nesting depth of INDIRECT_BUFFER execution (guards runaway recursion).
    u32 ib_depth{0};

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
    usize unsupported_packets{0};
    usize malformed_packets{0};
    enum DirtyBits : u32 {
        DirtyPipeline = 1u << 0,
        DirtyVertexLayout = 1u << 1,
        DirtyTexture = 1u << 2,
        DirtyUniform = 1u << 3,
        DirtyRenderTarget = 1u << 4,
        DirtyScissor = 1u << 5,
    };
    u32 dirty_bits{0};
    GpuFrameStats frame_stats{};
    static constexpr usize kDefaultResourceCacheBudget = 64 * 1024 * 1024;
    struct ResourceKey {
        GuestAddr address{0};
        usize size{0};
        u32 format{0};
        u32 width{0};
        u32 height{0};
        u64 generation{0};
        bool operator<(const ResourceKey& o) const noexcept {
            return std::tie(address, size, format, width, height, generation) <
                   std::tie(o.address, o.size, o.format, o.width, o.height, o.generation);
        }
    };
    struct BufferCacheEntry {
        id<MTLBuffer> buffer{nil};
        usize bytes{0};
        u64 last_used{0};
        std::vector<u8> snapshot;
    };
    struct TextureCacheEntry {
        id<MTLTexture> texture{nil};
        usize bytes{0};
        u64 last_used{0};
        std::vector<u8> snapshot;
    };
    std::map<ResourceKey, BufferCacheEntry> buffer_cache;
    std::map<ResourceKey, TextureCacheEntry> texture_cache;
    usize resource_cache_budget{kDefaultResourceCacheBudget};
    usize resource_cache_bytes{0};
    u64 resource_generation{0};
    u64 resource_use_clock{0};
    bool frame_draw_encoded{false};
    bool draw_diagnostics_enabled{true};
    u64 frame_submission_start{0};

    id<MTLRenderPipelineState> mesh_pipeline_state{nil};
    std::map<ShaderSignature, id<MTLRenderPipelineState>> shader_pipelines;
    std::set<ShaderSignature> shader_fallbacks;
    std::string shader_pipeline_error;
    id<MTLTexture> default_white_texture{nil};
    id<MTLSamplerState> sampler_state{nil};
    id<MTLTexture> active_metal_texture{nil};

    GuestAddr ps_texture_base{0};
    u32 ps_texture_width{0};
    u32 ps_texture_height{0};
    usize ps_texture_pitch{0};
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

    // Phase 19: translated GCN shader path (gcn_pipeline.mm). User SGPRs and
    // context state are read from reg_file; the fixed mesh shader below
    // remains the fallback when a shader or state is unsupported.
    std::unique_ptr<GcnPipelineCache> gcn_cache;
    bool gcn_enabled{GcnPipelineCache::enabled_by_env()};
    bool try_gcn_draw(bool indexed, u32 count, GuestAddr idx_addr, MTLIndexType idx_type,
                      PrimitiveType prim_type, i32 base_vertex, u32 instance_count);
    void apply_viewport_scissor(id<MTLRenderCommandEncoder> enc);

    bool init_mesh_pipeline();
    id<MTLRenderPipelineState> pipeline_for_current_shader();
    ShaderSignature current_shader_signature() const noexcept;
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
    id<MTLBuffer> cached_buffer(id<MTLDevice> device, GuestAddr address, usize size,
                                const void* bytes, u32 format = 0);
    void evict_resource_cache(usize requested);
    void invalidate_resources(GuestAddr address, usize size) noexcept;

    bool require_payload(const Pm4Packet& packet, usize count, const char* operation) {
        if (packet.payload.size() >= count) return true;
        ++malformed_packets;
        std::fprintf(stderr, "[PM4] Malformed %s packet: expected at least %zu payload DWORDs, got %zu\n",
                     operation, count, packet.payload.size());
        return false;
    }

    void mark_pipeline_dirty() noexcept { dirty_bits |= DirtyPipeline; }
    void mark_resource_dirty() noexcept {
        dirty_bits |= DirtyVertexLayout | DirtyTexture | DirtyUniform | DirtyRenderTarget;
    }
};

bool Pm4Translator::Impl::init_mesh_pipeline() {
    if (!metal || !metal->is_initialized()) return false;
    if (mesh_pipeline_state) return true;

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)metal->raw_device();
        if (!device) return false;

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
            "    if (uniforms.vertex_stride == 88) {\n"
            "        pos = *(constant packed_float3*)v_ptr;\n"
            "        col = *(constant packed_float4*)(v_ptr + 24);\n"
            "        uv  = *(constant packed_float2*)(v_ptr + 40);\n"
            "    } else if (uniforms.vertex_stride == 24) {\n"
            "        pos = *(constant packed_float3*)v_ptr;\n"
            "        col = float4(1.0, 1.0, 1.0, 1.0);\n"
            "        uv  = *(constant packed_float2*)(v_ptr + 16);\n"
            "    } else if (uniforms.vertex_stride == 16) {\n"
            "        pos = *(constant packed_float3*)v_ptr;\n"
            "        col = float4(1.0, 1.0, 1.0, 1.0);\n"
            "        uv  = *(constant packed_float2*)(v_ptr + 8);\n"
            "    } else if (uniforms.vertex_stride >= 28) {\n"
            "        col = *(constant packed_float4*)(v_ptr + 12);\n"
            "        if (uniforms.vertex_stride >= 36) {\n"
            "            uv = *(constant packed_float2*)(v_ptr + 28);\n"
            "        }\n"
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
            shader_pipeline_error = error ? [[error localizedDescription] UTF8String] : "unknown shader library error";
            return false;
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
            shader_pipeline_error = error ? [[error localizedDescription] UTF8String] : "unknown pipeline error";
            return false;
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
    return mesh_pipeline_state != nil;
}

ShaderSignature Pm4Translator::Impl::current_shader_signature() const noexcept {
    return ShaderSignature{
        .vertex_program = vs_shader_addr,
        .fragment_program = ps_shader_addr,
        .vertex_resource_1 = vs_rsrc1,
        .vertex_resource_2 = vs_rsrc2,
        .fragment_resource_1 = ps_rsrc1,
        .fragment_resource_2 = ps_rsrc2,
    };
}

id<MTLRenderPipelineState> Pm4Translator::Impl::pipeline_for_current_shader() {
    const ShaderSignature signature = current_shader_signature();
    const auto cached = shader_pipelines.find(signature);
    if (cached != shader_pipelines.end()) return cached->second;

    if (!init_mesh_pipeline()) return nil;
    shader_pipelines.emplace(signature, mesh_pipeline_state);
    shader_fallbacks.insert(signature);
    std::fprintf(stderr,
                 "[PM4] Shader compatibility fallback: VS=0x%llx PS=0x%llx "
                 "VS_RES=(0x%08x,0x%08x) PS_RES=(0x%08x,0x%08x) (fallbacks=%zu)\n",
                 static_cast<unsigned long long>(signature.vertex_program),
                 static_cast<unsigned long long>(signature.fragment_program),
                 signature.vertex_resource_1,
                 signature.vertex_resource_2,
                 signature.fragment_resource_1,
                 signature.fragment_resource_2,
                 shader_fallbacks.size());
    return mesh_pipeline_state;
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

    // Ensure Unity's ScreenManagerPS4 dimensions are initialized to 1920x1080
    if (memory) {
        GuestAddr sm_ptr = 0;
        if (memory->read_value(0x1563a00, sm_ptr) && sm_ptr > 0x1000 && memory->is_valid_address(sm_ptr, 0x300)) {
            u32 cur_w = 0, cur_h = 0;
            memory->read_value(sm_ptr + 0x2b0, cur_w);
            memory->read_value(sm_ptr + 0x2b4, cur_h);
            if (cur_w != 1920 || cur_h != 1080) {
                memory->write_value(sm_ptr + 0x2b0, static_cast<u32>(1920));
                memory->write_value(sm_ptr + 0x2b4, static_cast<u32>(1080));
                memory->write_value(sm_ptr + 0x2bc, static_cast<u32>(1920));
                memory->write_value(sm_ptr + 0x2c0, static_cast<u32>(1080));
                memory->write_value(sm_ptr + 0x2e4, static_cast<u32>(1920));
                memory->write_value(sm_ptr + 0x2e8, static_cast<u32>(1080));
            }
        }
    }

    for (const auto& [slot, buf] : vs_user_sgpr_buffers) {
        if (buf.num_records >= 21 && buf.base_addr > 0x1000) {
            const GuestAddr proj_addr = buf.base_addr + 272;
            if (memory->is_valid_address(proj_addr, 64)) {
                float p[16] = {0};
                memory->read_bytes(proj_addr, p, 64);
                // Sanitize NaN values resulting from 0/0 screen dimension divisions
                if (std::isnan(p[0]) || std::isnan(p[5])) {
                    p[0]  = 2.0f / 1920.0f;
                    p[1]  = 0.0f;
                    p[2]  = 0.0f;
                    p[3]  = 0.0f;
                    p[4]  = 0.0f;
                    p[5]  = -2.0f / 1080.0f;
                    p[6]  = 0.0f;
                    p[7]  = 0.0f;
                    p[8]  = 0.0f;
                    p[9]  = 0.0f;
                    p[10] = 0.9901f;
                    p[11] = 0.0f;
                    p[12] = -1.0f;
                    p[13] = 1.0f;
                    p[14] = 0.0f;
                    p[15] = 1.0f;
                    memory->write_bytes(proj_addr, p, 64);
                }
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

void Pm4Translator::Impl::evict_resource_cache(usize requested) {
    while (resource_cache_bytes + requested > resource_cache_budget &&
           (!buffer_cache.empty() || !texture_cache.empty())) {
        bool evict_buffer = texture_cache.empty();
        ResourceKey oldest_key{};
        u64 oldest = UINT64_MAX;
        if (!buffer_cache.empty()) {
            const auto it = std::min_element(buffer_cache.begin(), buffer_cache.end(),
                [](const auto& a, const auto& b) { return a.second.last_used < b.second.last_used; });
            oldest_key = it->first;
            oldest = it->second.last_used;
        }
        if (!texture_cache.empty()) {
            const auto it = std::min_element(texture_cache.begin(), texture_cache.end(),
                [](const auto& a, const auto& b) { return a.second.last_used < b.second.last_used; });
            if (it->second.last_used < oldest) {
                evict_buffer = false;
                oldest_key = it->first;
            }
        }
        if (evict_buffer) {
            resource_cache_bytes -= buffer_cache[oldest_key].bytes;
            buffer_cache.erase(oldest_key);
        } else {
            resource_cache_bytes -= texture_cache[oldest_key].bytes;
            texture_cache.erase(oldest_key);
        }
    }
}

id<MTLBuffer> Pm4Translator::Impl::cached_buffer(id<MTLDevice> device, GuestAddr address,
                                                  usize size, const void* bytes, u32 format) {
    if (!device || !memory || size == 0 || !memory->is_valid_address(address, size) || !bytes) {
        return nil;
    }
    if (size > resource_cache_budget) return nil;
    const ResourceKey key{address, size, format, 0, 0, resource_generation};
    auto hit = buffer_cache.find(key);
    if (hit != buffer_cache.end()) {
        if (hit->second.snapshot.size() != size ||
            std::memcmp(hit->second.snapshot.data(), bytes, size) != 0) {
            resource_cache_bytes -= hit->second.bytes;
            buffer_cache.erase(hit);
        } else {
            hit->second.last_used = ++resource_use_clock;
            ++frame_stats.resource_cache_hits;
            return hit->second.buffer;
        }
    }
    evict_resource_cache(size);
    if (resource_cache_bytes + size > resource_cache_budget) return nil;
    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes length:size options:MTLResourceStorageModeShared];
    if (!buffer) return nil;
    std::vector<u8> snapshot(static_cast<const u8*>(bytes),
                             static_cast<const u8*>(bytes) + size);
    buffer_cache.emplace(key, BufferCacheEntry{buffer, size, ++resource_use_clock,
                                               std::move(snapshot)});
    resource_cache_bytes += size;
    frame_stats.upload_bytes += size;
    ++frame_stats.resource_cache_misses;
    return buffer;
}

void Pm4Translator::Impl::invalidate_resources(GuestAddr address, usize size) noexcept {
    if (size == 0) return;
    const GuestAddr end = address > UINT64_MAX - size ? UINT64_MAX : address + size;
    auto overlaps = [address, end](const ResourceKey& key) {
        const GuestAddr key_end = key.address > UINT64_MAX - key.size ? UINT64_MAX : key.address + key.size;
        return key.address < end && address < key_end;
    };
    for (auto it = buffer_cache.begin(); it != buffer_cache.end();) {
        if (overlaps(it->first)) {
            resource_cache_bytes -= it->second.bytes;
            it = buffer_cache.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = texture_cache.begin(); it != texture_cache.end();) {
        if (overlaps(it->first)) {
            resource_cache_bytes -= it->second.bytes;
            it = texture_cache.erase(it);
        } else {
            ++it;
        }
    }
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

    const u32 dfmt = (ps_texture_w1 >> 14) & 0x3F;
    const u32 nfmt = (ps_texture_w1 >> 20) & 0xF;
    enum class TextureLayout { RGBA8, BGRA8 };
    TextureLayout layout;
    if (dfmt == 10 || nfmt == 10 || (dfmt == 0 && nfmt == 0)) {
        layout = TextureLayout::RGBA8;
    } else if (dfmt == 11 || nfmt == 11) {
        layout = TextureLayout::BGRA8;
    } else {
        ps_texture_status = "unsupported texture format dfmt=" + std::to_string(dfmt) +
                            " nfmt=" + std::to_string(nfmt);
        return nil;
    }
    const usize row_bytes = static_cast<usize>(ps_texture_width) * 4;
    // The observed descriptor stores pitch in bytes in the low word. A zero
    // pitch means the documented tightly-packed layout.
    const usize pitch = ps_texture_pitch != 0 ? ps_texture_pitch : row_bytes;
    if (pitch < row_bytes) {
        ps_texture_status = "invalid texture pitch";
        return nil;
    }
    const usize byte_size = pitch * ps_texture_height;
    if (!memory->is_valid_address(ps_texture_base, byte_size)) {
        ps_texture_status = "hit white fallback (invalid memory range for 0x" + std::to_string(ps_texture_base) + ")";
        return default_white_texture;
    }

    const void* host_pixels = memory->to_host_ptr<void>(ps_texture_base);
    if (!host_pixels) {
        ps_texture_status = "hit white fallback (null host pointer)";
        return default_white_texture;
    }

    const usize texture_bytes = byte_size;
    const ResourceKey key{ps_texture_base, texture_bytes, ps_texture_format,
                          ps_texture_width, ps_texture_height, resource_generation};
    auto cached = texture_cache.find(key);
    if (cached != texture_cache.end()) {
        if (cached->second.snapshot.size() == byte_size &&
            std::memcmp(cached->second.snapshot.data(), host_pixels, byte_size) == 0) {
            cached->second.last_used = ++resource_use_clock;
            ++frame_stats.resource_cache_hits;
            active_metal_texture = cached->second.texture;
            ps_texture_status = "cache hit";
            return active_metal_texture;
        }
        resource_cache_bytes -= cached->second.bytes;
        texture_cache.erase(cached);
    }
    if (texture_bytes > resource_cache_budget) {
        ps_texture_status = "resource exceeds cache budget";
        return nil;
    }
    evict_resource_cache(texture_bytes);
    if (resource_cache_bytes + texture_bytes > resource_cache_budget) return nil;
    // A cache miss must allocate a texture for this complete key. Reusing a
    // same-dimension active texture would alias distinct guest addresses.
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:ps_texture_width
                                                                                   height:ps_texture_height
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    active_metal_texture = [device newTextureWithDescriptor:desc];

    const u8* u8_ptr = static_cast<const u8*>(host_pixels);
    std::vector<u8> packed(row_bytes * ps_texture_height);
    for (u32 y = 0; y < ps_texture_height; ++y) {
        const u8* src = u8_ptr + static_cast<usize>(y) * pitch;
        u8* dst = packed.data() + static_cast<usize>(y) * row_bytes;
        if (layout == TextureLayout::RGBA8) {
            std::memcpy(dst, src, row_bytes);
        } else {
            for (u32 x = 0; x < ps_texture_width; ++x) {
                dst[x * 4 + 0] = src[x * 4 + 2];
                dst[x * 4 + 1] = src[x * 4 + 1];
                dst[x * 4 + 2] = src[x * 4 + 0];
                dst[x * 4 + 3] = src[x * 4 + 3];
            }
        }
    }
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
                                  withBytes:packed.data()
                                bytesPerRow:row_bytes];
    } else {
        ps_texture_status = "hit white fallback (all 0x00 bytes in guest memory across " +
                            std::to_string(check_len) + " bytes checked)";
        std::vector<u32> white_data(ps_texture_width * ps_texture_height, 0xFFFFFFFF);
        [active_metal_texture replaceRegion:MTLRegionMake2D(0, 0, ps_texture_width, ps_texture_height)
                                mipmapLevel:0
                                  withBytes:white_data.data()
                                bytesPerRow:ps_texture_width * 4];
    }

    std::vector<u8> snapshot(u8_ptr, u8_ptr + byte_size);
    texture_cache.emplace(key, TextureCacheEntry{active_metal_texture, texture_bytes, ++resource_use_clock,
                                                 std::move(snapshot)});
    resource_cache_bytes += texture_bytes;
    frame_stats.upload_bytes += texture_bytes;
    ++frame_stats.resource_cache_misses;
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

void Pm4Translator::Impl::apply_viewport_scissor(id<MTLRenderCommandEncoder> enc) {
    const u32 fb_w = metal->width() > 0 ? metal->width() : 1920;
    const u32 fb_h = metal->height() > 0 ? metal->height() : 1080;
    const f32 origin_x = (viewport.width > 0.0f && viewport.x >= (viewport.width / 2.0f))
                             ? (viewport.x - (viewport.width / 2.0f)) : viewport.x;
    const f32 origin_y = (viewport.height > 0.0f && viewport.y >= (viewport.height / 2.0f))
                             ? (viewport.y - (viewport.height / 2.0f)) : viewport.y;
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
    [enc setScissorRect:MTLScissorRect{.x = sc_x, .y = sc_y, .width = sc_w, .height = sc_h}];
}

// Phase 19: draw with the game's own GCN shaders translated to MSL. Returns
// false (nothing encoded) when the translated path is disabled or anything
// is unsupported, so the caller falls through to the fixed mesh shader.
bool Pm4Translator::Impl::try_gcn_draw(bool indexed, u32 count, GuestAddr idx_addr, MTLIndexType idx_type,
                                       PrimitiveType prim_type, i32 base_vertex, u32 instance_count) {
    if (!gcn_enabled || !memory || !metal || vs_shader_addr <= 0x1000 || ps_shader_addr <= 0x1000) return false;
    @autoreleasepool {
        if (!gcn_cache) gcn_cache = std::make_unique<GcnPipelineCache>(metal->raw_device(), memory);
        gcn_cache->set_memory(memory);
        if (!metal->ensure_render_encoder(!frame_draw_encoded, clear_color)) return false;
        id<MTLRenderCommandEncoder> enc = (__bridge id<MTLRenderCommandEncoder>)metal->active_render_encoder();
        id<MTLTexture> fb = (__bridge id<MTLTexture>)metal->raw_framebuffer();
        if (!enc || !fb) return false;
        GcnDrawRequest req;
        req.vs_addr = vs_shader_addr;
        req.ps_addr = ps_shader_addr;
        req.vs_rsrc2 = vs_rsrc2;
        req.ps_rsrc2 = ps_rsrc2;
        // SPI_SHADER_USER_DATA_PS_0 = SH 0x0C, SPI_SHADER_USER_DATA_VS_0 = SH 0x4C.
        req.vs_user = reg_file.data() + kPm4ShRegBase + 0x4C;
        req.ps_user = reg_file.data() + kPm4ShRegBase + 0x0C;
        req.ctx_regs = reg_file.data() + kPm4ContextRegBase;
        req.ctx_reg_count = kPm4UconfigRegBase - kPm4ContextRegBase;
        req.prim = prim_type;
        req.indexed = indexed;
        req.count = count;
        req.index_addr = idx_addr;
        req.index32 = idx_type == MTLIndexTypeUInt32;
        req.base_vertex = base_vertex;
        req.instances = instance_count;
        apply_viewport_scissor(enc);
        std::string why;
        if (!gcn_cache->encode_draw((__bridge void*)enc, static_cast<u32>(fb.pixelFormat), req, &why)) return false;
        if (draw_diagnostics_enabled) {
            PM4_LOG("[GCN] draw #%zu %s count=%u prim=%u VS=0x%llx PS=0x%llx (translated)\n",
                         draw_count + 1, indexed ? "indexed" : "auto", count, static_cast<u32>(prim_type),
                         (unsigned long long)vs_shader_addr, (unsigned long long)ps_shader_addr);
        }
        return true;
    }
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
    if (!metal->frame_in_progress()) {
        PM4_LOG("[PM4] draw outside frame\n");
        return false;
    }

    if (try_gcn_draw(true, idx_count, idx_addr, idx_type, prim_type, base_vertex, instance_count)) {
        draw_count++;
        indexed_draw_count++;
        frame_stats.draws++;
        frame_stats.indexed_draws++;
        frame_draw_encoded = true;
        return true;
    }

    @autoreleasepool {
        id<MTLRenderPipelineState> pipeline = pipeline_for_current_shader();
        if (!pipeline) return false;

        id<MTLDevice> device = (__bridge id<MTLDevice>)metal->raw_device();
        if (!device) return false;

        const usize elem_size = (idx_type == MTLIndexTypeUInt32) ? sizeof(u32) : sizeof(u16);
        const usize required_idx_size = idx_count * elem_size;
        if (!memory || !memory->is_valid_address(idx_addr, required_idx_size)) {
            std::fprintf(stderr, "[PM4] render_indexed_mesh: invalid idx_addr=0x%llx (size=%zu)\n",
                         (unsigned long long)idx_addr, required_idx_size);
            return false;
        }
        const void* idx_ptr = memory->to_host_ptr<void>(idx_addr);
        if (!idx_ptr) return false;

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

        // Discover guest vertex buffer from V#0 in vs_user_sgprs[2:3]
        GuestAddr eff_vtx_addr = vertex_buffer_addr;
        usize eff_vtx_stride = vertex_stride > 0 ? vertex_stride : sizeof(Vertex3D);
        usize eff_vtx_size = vertex_buffer_size;

        const u64 v_table_addr = (static_cast<u64>(vs_user_sgprs[3]) << 32) | vs_user_sgprs[2];
        if (memory && v_table_addr > 0x1000 && memory->is_valid_address(v_table_addr, 16)) {
            u32 d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            memory->read_value(v_table_addr + 0, d0);
            memory->read_value(v_table_addr + 4, d1);
            memory->read_value(v_table_addr + 8, d2);
            memory->read_value(v_table_addr + 12, d3);
            const u64 b = (static_cast<u64>(d1 & 0xFFFF) << 32) | d0;
            const u32 s = (d1 >> 16) & 0x3FFF;
            const u32 r = d2;
            if (b > 0x1000 && s > 0 && r > 0) {
                eff_vtx_addr = b;
                eff_vtx_stride = s;
                eff_vtx_size = static_cast<usize>(r) * s;
            }
        }

        const usize stride = eff_vtx_stride;
        const u64 effective_vtx_count = static_cast<u64>(max_idx) + 1 + (base_vertex > 0 ? base_vertex : 0);
        const usize min_needed = static_cast<usize>(effective_vtx_count) * stride;
        usize required_vtx_size = eff_vtx_size;
        if (required_vtx_size == 0 || min_needed > required_vtx_size) {
            required_vtx_size = min_needed;
        }
        if (eff_vtx_addr != 0 && (!memory || required_vtx_size == 0 ||
                                  !memory->is_valid_address(eff_vtx_addr, required_vtx_size))) {
            std::fprintf(stderr, "[PM4] render_indexed_mesh: invalid vertex range 0x%llx (size=%zu)\n",
                         (unsigned long long)eff_vtx_addr, required_vtx_size);
            return false;
        }

        MeshUniforms uniforms = compute_uniforms();
        id<MTLTexture> active_tex = prepare_texture(device);
        if (!active_tex) {
            std::fprintf(stderr, "[PM4] indexed draw rejected: texture allocation failed\n");
            return false;
        }
        uniforms.use_texture = (active_tex != default_white_texture) ? 1 : 0;

        bool use_guest_vtx = false;
        if (eff_vtx_addr != 0 && stride >= 16 &&
            memory && memory->is_valid_address(eff_vtx_addr, required_vtx_size)) {
            const void* vtx_check = memory->to_host_ptr<void>(eff_vtx_addr);
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
            const void* vtx_ptr = memory->to_host_ptr<void>(eff_vtx_addr);
            vtx_buf = cached_buffer(device, eff_vtx_addr, required_vtx_size, vtx_ptr);
            if (!vtx_buf) return false;
            uniforms.vertex_stride = stride;
        } else {
            vtx_buf = [device newBufferWithBytes:s_default_quad length:sizeof(s_default_quad) options:MTLResourceStorageModeShared];
            uniforms.vertex_stride = sizeof(MeshVertex);
        }

        id<MTLBuffer> idx_buf = cached_buffer(device, idx_addr, required_idx_size, idx_ptr);
        if (!idx_buf) return false;

        if (!metal->ensure_render_encoder(!frame_draw_encoded, clear_color)) return false;
        id<MTLRenderCommandEncoder> enc =
            (__bridge id<MTLRenderCommandEncoder>)metal->active_render_encoder();
        if (!enc) return false;

        [enc setRenderPipelineState:pipeline];
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

        if (draw_diagnostics_enabled) {
            log_draw_diagnostics("IT_DRAW_INDEX_2", idx_count, uniforms, active_tex, use_guest_vtx,
                                 use_guest_vtx ? memory->to_host_ptr<void>(eff_vtx_addr) : s_default_quad,
                                 use_guest_vtx ? stride : sizeof(MeshVertex),
                                 use_guest_vtx ? std::min<u32>(static_cast<u32>(effective_vtx_count), 4) : 4);
        }

        draw_count++;
        indexed_draw_count++;
        if (metal && metal->frame_in_progress()) {
            frame_stats.draws++;
            frame_stats.indexed_draws++;
        }
        frame_draw_encoded = true;
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
    if (!metal->frame_in_progress()) {
        PM4_LOG("[PM4] draw outside frame\n");
        return false;
    }

    if (try_gcn_draw(false, vtx_count, 0, MTLIndexTypeUInt16, prim_type, base_vertex, instance_count)) {
        draw_count++;
        frame_stats.draws++;
        frame_draw_encoded = true;
        return true;
    }
    // Fixed-shader fallback has no RectList/QuadList lowering for guest
    // vertices: keep the pre-VGT_PRIMITIVE_TYPE behaviour (a single quad as a
    // 4-vertex strip, a rect as a plain triangle).
    if (prim_type == PrimitiveType::QuadList && vtx_count == 4) prim_type = PrimitiveType::TriangleStrip;
    if (prim_type == PrimitiveType::RectList) prim_type = PrimitiveType::TriangleList;

    @autoreleasepool {
        id<MTLRenderPipelineState> pipeline = pipeline_for_current_shader();
        if (!pipeline) return false;

        id<MTLDevice> device = (__bridge id<MTLDevice>)metal->raw_device();
        if (!device) return false;

        // Discover guest vertex buffer from V#0 in vs_user_sgprs[2:3]
        GuestAddr eff_vtx_addr = vertex_buffer_addr;
        usize eff_vtx_stride = vertex_stride > 0 ? vertex_stride : sizeof(Vertex3D);
        usize eff_vtx_size = vertex_buffer_size;

        const u64 v_table_addr = (static_cast<u64>(vs_user_sgprs[3]) << 32) | vs_user_sgprs[2];
        if (memory && v_table_addr > 0x1000 && memory->is_valid_address(v_table_addr, 16)) {
            u32 d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            memory->read_value(v_table_addr + 0, d0);
            memory->read_value(v_table_addr + 4, d1);
            memory->read_value(v_table_addr + 8, d2);
            memory->read_value(v_table_addr + 12, d3);
            const u64 b = (static_cast<u64>(d1 & 0xFFFF) << 32) | d0;
            const u32 s = (d1 >> 16) & 0x3FFF;
            const u32 r = d2;
            if (b > 0x1000 && s > 0 && r > 0) {
                eff_vtx_addr = b;
                eff_vtx_stride = s;
                eff_vtx_size = static_cast<usize>(r) * s;
            }
        }

        const usize stride = eff_vtx_stride;
        const usize min_needed = static_cast<usize>(vtx_count + (base_vertex > 0 ? base_vertex : 0)) * stride;
        const usize required_vtx_size = std::max<usize>(eff_vtx_size, min_needed);
        if (eff_vtx_addr != 0 && (!memory || required_vtx_size == 0 ||
                                  !memory->is_valid_address(eff_vtx_addr, required_vtx_size))) {
            std::fprintf(stderr, "[PM4] render_auto_mesh: invalid vertex range 0x%llx (size=%zu)\n",
                         (unsigned long long)eff_vtx_addr, required_vtx_size);
            return false;
        }

        MeshUniforms uniforms = compute_uniforms();
        id<MTLTexture> active_tex = prepare_texture(device);
        if (!active_tex) {
            std::fprintf(stderr, "[PM4] auto draw rejected: texture allocation failed\n");
            return false;
        }
        uniforms.use_texture = (active_tex != default_white_texture) ? 1 : 0;

        id<MTLBuffer> vtx_buf = nil;
        MTLPrimitiveType mtl_prim = MTLPrimitiveTypeTriangle;
        u32 actual_vtx_count = vtx_count;

        bool use_guest_vtx = false;
        if (eff_vtx_addr != 0 && stride >= 16 &&
            memory && memory->is_valid_address(eff_vtx_addr, required_vtx_size)) {
            const void* vtx_check = memory->to_host_ptr<void>(eff_vtx_addr);
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
            const void* vtx_ptr = memory->to_host_ptr<void>(eff_vtx_addr);
            vtx_buf = cached_buffer(device, eff_vtx_addr, required_vtx_size, vtx_ptr);
            if (!vtx_buf) return false;
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

        if (!metal->ensure_render_encoder(!frame_draw_encoded, clear_color)) return false;
        id<MTLRenderCommandEncoder> enc =
            (__bridge id<MTLRenderCommandEncoder>)metal->active_render_encoder();
        if (!enc) return false;

        [enc setRenderPipelineState:pipeline];
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

        if (draw_diagnostics_enabled) {
            log_draw_diagnostics("IT_DRAW_INDEX_AUTO", actual_vtx_count, uniforms, active_tex, use_guest_vtx,
                                 diag_vtx_ptr, diag_vtx_stride, diag_vtx_num);
        }

        draw_count++;
        if (metal && metal->frame_in_progress()) {
            frame_stats.draws++;
        }
        frame_draw_encoded = true;
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

bool Pm4Translator::begin_frame() {
    if (!m_impl) return false;
    std::lock_guard<std::recursive_mutex> frame_lock(m_impl->frame_mutex);
    if (!m_impl->metal) return true;
    if (!m_impl->metal->begin_frame()) return false;
    m_impl->frame_draw_encoded = false;
    const usize upload_bytes = m_impl->frame_stats.upload_bytes;
    const usize cache_hits = m_impl->frame_stats.resource_cache_hits;
    const usize cache_misses = m_impl->frame_stats.resource_cache_misses;
    m_impl->frame_stats = GpuFrameStats{};
    m_impl->frame_stats.upload_bytes = upload_bytes;
    m_impl->frame_stats.resource_cache_hits = cache_hits;
    m_impl->frame_stats.resource_cache_misses = cache_misses;
    m_impl->frame_submission_start = m_impl->metal->submitted_frame_count();
    m_impl->frame_stats.frame_index = m_impl->frame_submission_start + 1;
    return true;
}

bool Pm4Translator::submit_frame(bool wait_for_completion) {
    if (!m_impl) return false;
    std::lock_guard<std::recursive_mutex> frame_lock(m_impl->frame_mutex);
    if (!m_impl->metal) return true;
    if (!m_impl->metal->frame_in_progress()) return false;
    m_impl->metal->end_render_encoder();
    // PS4_GPU_READBACK=1: wait for frames that drew and log the framebuffer
    // centre (diagnostic for comparing the GCN path with the fixed shader).
    static const bool readback = std::getenv("PS4_GPU_READBACK") != nullptr;
    const bool drew = m_impl->frame_draw_encoded;
    const bool submitted = m_impl->metal->submit_frame(wait_for_completion || (readback && drew));
    if (readback && drew && submitted) {
        u8 r = 0, g = 0, b = 0, a = 0;
        const u32 w = m_impl->metal->width(), h = m_impl->metal->height();
        m_impl->metal->read_pixel(w / 2, h / 2, r, g, b, a);
        u8 r2 = 0, g2 = 0, b2 = 0, a2 = 0;
        m_impl->metal->read_pixel(w / 4, h / 4, r2, g2, b2, a2);
        std::fprintf(stderr, "[GPU readback] frame %llu: center RGBA(%u,%u,%u,%u) quarter RGBA(%u,%u,%u,%u) (gcn %s)\n",
                     (unsigned long long)m_impl->metal->submitted_frame_count(), r, g, b, a, r2, g2, b2, a2,
                     m_impl->gcn_enabled ? "on" : "off");
        const char* dir = std::getenv("PS4_GPU_READBACK");
        if (dir && dir[0] == '/') {   // also dump the frame as a PPM
            id<MTLTexture> fb = (__bridge id<MTLTexture>)m_impl->metal->raw_framebuffer();
            std::vector<u8> px(static_cast<usize>(w) * h * 4);
            [fb getBytes:px.data() bytesPerRow:w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
            char path[512];
            std::snprintf(path, sizeof(path), "%s/frame_%llu_%s.ppm", dir,
                          (unsigned long long)m_impl->metal->submitted_frame_count(),
                          m_impl->gcn_enabled ? "gcn" : "fixed");
            if (FILE* f = std::fopen(path, "wb")) {
                std::fprintf(f, "P6\n%u %u\n255\n", w, h);
                for (usize p = 0; p < static_cast<usize>(w) * h; ++p) {
                    const u8 rgb[3] = {px[p * 4 + 2], px[p * 4 + 1], px[p * 4 + 0]};
                    std::fwrite(rgb, 1, 3, f);
                }
                std::fclose(f);
            }
        }
    }
    m_impl->frame_stats.command_buffers_submitted =
        m_impl->metal->submitted_frame_count() - m_impl->frame_submission_start;
    return submitted;
}

bool Pm4Translator::frame_in_progress() const noexcept {
    return m_impl && (!m_impl->metal || m_impl->metal->frame_in_progress());
}

GpuFrameStats Pm4Translator::frame_stats() const noexcept {
    return m_impl ? m_impl->frame_stats : GpuFrameStats{};
}

usize Pm4Translator::shader_fallback_count() const noexcept {
    return m_impl ? m_impl->shader_fallbacks.size() : 0;
}

std::string Pm4Translator::shader_status() const {
    if (!m_impl) return "shader pipeline unavailable";
    if (!m_impl->shader_pipeline_error.empty()) {
        return "shader pipeline error: " + m_impl->shader_pipeline_error;
    }
    return "shader compatibility fallback signatures=" +
           std::to_string(m_impl->shader_fallbacks.size()) +
           " cached_pipelines=" + std::to_string(m_impl->shader_pipelines.size());
}

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
    m_impl->num_instances = 1;
    // CLEAR_STATE semantics: context registers return to their defaults (0).
    std::fill(m_impl->reg_file.begin() + kPm4ContextRegBase,
              m_impl->reg_file.begin() + kPm4UconfigRegBase, 0u);
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
    m_impl->ps_texture_pitch = 0;
    m_impl->ps_texture_format = 0;
    m_impl->ps_texture_w0 = 0;
    m_impl->ps_texture_w1 = 0;
    m_impl->ps_texture_w2 = 0;
    m_impl->ps_texture_w3 = 0;
    m_impl->ps_texture_has_nonzero = false;
    m_impl->ps_texture_nonzero_bytes = 0;
    m_impl->ps_texture_status = "none";
    m_impl->vs_user_sgpr_buffers.clear();
    m_impl->dirty_bits = Impl::DirtyPipeline | Impl::DirtyVertexLayout | Impl::DirtyTexture |
                         Impl::DirtyUniform | Impl::DirtyRenderTarget | Impl::DirtyScissor;
}

void Pm4Translator::set_memory_manager(MemoryManager* memory) noexcept {
    if (m_impl) {
        if (m_impl->memory != memory) {
            m_impl->buffer_cache.clear();
            m_impl->texture_cache.clear();
            m_impl->resource_cache_bytes = 0;
            ++m_impl->resource_generation;
        }

        m_impl->memory = memory;
    }
}

void Pm4Translator::set_draw_diagnostics_enabled(bool enabled) noexcept {
    if (m_impl) m_impl->draw_diagnostics_enabled = enabled;
}

void Pm4Translator::set_resource_cache_budget(usize bytes) noexcept {
    if (!m_impl) return;
    m_impl->resource_cache_budget = bytes;
    m_impl->evict_resource_cache(0);
}

void Pm4Translator::invalidate_guest_resources(GuestAddr address, usize size) noexcept {
    if (m_impl) m_impl->invalidate_resources(address, size);
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
    if (m_impl->metal && m_impl->metal->frame_in_progress()) {
        m_impl->frame_stats.packets++;
    }
    if (packet.header.count != packet.payload.size()) {
        ++m_impl->malformed_packets;
        std::fprintf(stderr, "[PM4] Malformed packet: header declares %u payload DWORDs, got %zu\n",
                     packet.header.count, packet.payload.size());
        return false;
    }

    if (packet.header.type == Pm4PacketType::Type2) {
        // NOP packet
        return true;
    }

    Impl& st = *m_impl;

    // ---------------------------------------------------------------------
    // Register-space helpers. Every register write (SET_*_REG, Type-0,
    // WRITE_DATA/COPY_DATA/DMA_DATA to a register, LOAD_*_REG, RMW packets)
    // lands in the shadow register file and is re-dispatched through the
    // matching SET_*_REG handler so that side effects live in one place.
    // ---------------------------------------------------------------------
    auto reg_space_for = [](u32 abs_reg, u8& op, u32& base) -> bool {
        if (abs_reg >= kPm4UconfigRegBase && abs_reg < 0x10000) { op = IT_SET_UCONFIG_REG; base = kPm4UconfigRegBase; return true; }
        if (abs_reg >= kPm4ContextRegBase && abs_reg < kPm4UconfigRegBase) { op = IT_SET_CONTEXT_REG; base = kPm4ContextRegBase; return true; }
        if (abs_reg >= kPm4ShRegBase && abs_reg < kPm4ShRegBase + 0x400) { op = IT_SET_SH_REG; base = kPm4ShRegBase; return true; }
        if (abs_reg >= kPm4ConfigRegBase && abs_reg < kPm4ShRegBase) { op = IT_SET_CONFIG_REG; base = kPm4ConfigRegBase; return true; }
        return false;
    };
    auto write_regs = [&](u32 abs_reg, const u32* vals, usize n) -> bool {
        if (n == 0) return true;
        u8 op = 0;
        u32 base = 0;
        if (!reg_space_for(abs_reg, op, base)) {
            // Outside the four SET_*_REG apertures: keep the value only.
            for (usize i = 0; i < n; ++i) {
                if (abs_reg + i < st.reg_file.size()) st.reg_file[abs_reg + i] = vals[i];
            }
            return true;
        }
        Pm4Packet sub{};
        sub.header.type = Pm4PacketType::Type3;
        sub.header.opcode = op;
        sub.header.count = static_cast<u16>(n + 1);
        sub.payload.reserve(n + 1);
        sub.payload.push_back(abs_reg - base);
        sub.payload.insert(sub.payload.end(), vals, vals + n);
        const bool counted_frame = st.metal && st.metal->frame_in_progress();
        const bool ok = execute_packet(sub);
        // An internal re-dispatch is not a separate command-stream packet.
        --st.packets_processed;
        if (counted_frame && st.frame_stats.packets > 0) st.frame_stats.packets--;
        return ok;
    };
    auto read_reg = [&](u32 abs_reg) -> u32 {
        return abs_reg < st.reg_file.size() ? st.reg_file[abs_reg] : 0u;
    };
    auto set_index_type = [&](u32 val, const char* src) {
        const u32 t = val & kIndexTypeMask;
        if (t == 2) {
            PM4_LOG("[PM4] %s: 8-bit index type is not valid on CIK; using 16-bit\n", src);
        }
        st.index_type = (t == 1) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
        st.dirty_bits |= Impl::DirtyVertexLayout;
    };
    // VGT state that CIK keeps in user-config space (SI kept it in config space).
    auto apply_vgt_reg = [&](u32 abs_reg, u32 val) {
        switch (abs_reg) {
            case kPm4UconfigRegBase + UCONFIG_VGT_PRIMITIVE_TYPE:
            case kPm4ConfigRegBase + 0x256: // SI mmVGT_PRIMITIVE_TYPE (0x8958)
                st.primitive_type = static_cast<PrimitiveType>(val & 0x3F);
                st.dirty_bits |= Impl::DirtyPipeline;
                PM4_LOG("[PM4] VGT_PRIMITIVE_TYPE = 0x%02x\n", val & 0x3F);
                break;
            case kPm4UconfigRegBase + UCONFIG_VGT_INDEX_TYPE:
                set_index_type(val, "VGT_INDEX_TYPE");
                break;
            case kPm4UconfigRegBase + UCONFIG_VGT_NUM_INSTANCES:
            case kPm4ConfigRegBase + 0x25D: // SI mmVGT_NUM_INSTANCES (0x8974)
                st.num_instances = val;
                break;
            default:
                break;
        }
    };
    auto instances = [&]() -> u32 { return std::max<u32>(1u, st.num_instances); };
    auto gpu_clock = []() -> u64 {
        // 100 MHz reference clock (PS4 GPU core counter frequency is fixed by Gnm).
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return static_cast<u64>(ns) / 10u;
    };
    // EOP/EOS/RELEASE_MEM data write. DATA_SEL: 0 discard, 1 32-bit low,
    // 2 64-bit, 3 64-bit GPU clock, 4 64-bit system clock.
    auto write_end_of_pipe = [&](const char* name, GuestAddr addr, u32 data_sel, u32 lo, u32 hi) -> bool {
        if (data_sel == 0) return true;
        if (!st.memory) return true;
        u64 value = 0;
        usize bytes = 0;
        switch (data_sel) {
            case 1: value = lo; bytes = 4; addr &= ~GuestAddr{3}; break;
            case 2: value = (static_cast<u64>(hi) << 32) | lo; bytes = 8; addr &= ~GuestAddr{7}; break;
            case 3:
            case 4: value = gpu_clock(); bytes = 8; addr &= ~GuestAddr{7}; break;
            default:
                std::fprintf(stderr, "[PM4] %s: unsupported DATA_SEL=%u\n", name, data_sel);
                return false;
        }
        if (addr == 0 || !st.memory->is_valid_address(addr, bytes)) {
            std::fprintf(stderr, "[PM4] %s: label address 0x%llx is not mapped\n", name,
                         (unsigned long long)addr);
            return true;
        }
        if (bytes == 4) {
            st.memory->write_value<u32>(addr, static_cast<u32>(value));
        } else {
            st.memory->write_value<u64>(addr, value);
        }
        return true;
    };
    // Shared DMA engine for DMA_DATA and CP_DMA.
    // src_sel: 0 = src address (memory, or register when SAS), 1 = GDS,
    //          2 = immediate DATA, 3 = src address via L2 (memory).
    // dst_sel: 0 = dst address (memory, or register when DAS), 1 = GDS,
    //          3 = nowhere (pure prefetch).
    auto run_dma = [&](const char* name, u32 src_sel, u32 dst_sel, GuestAddr src, GuestAddr dst,
                       usize bytes, bool sas, bool das, bool saic, bool daic, u32 imm) -> bool {
        st.last_dma_dst_addr = dst;
        st.last_dma_src_addr = src;
        st.last_dma_size = bytes;
        st.last_dma_is_copy_or_nonzero = !(src_sel == 2 && imm == 0);
        if (bytes == 0 || dst_sel == 3) return true;
        if (src_sel == 1 || dst_sel == 1) {
            std::fprintf(stderr, "[PM4] %s: GDS source/destination is not emulated (src_sel=%u dst_sel=%u)\n",
                         name, src_sel, dst_sel);
            return false;
        }
        const usize dwords = (bytes + 3) / 4;
        auto src_dword = [&](usize i) -> u32 {
            if (src_sel == 2) return imm;
            if (sas) return read_reg(static_cast<u32>(src >> 2) + (saic ? 0u : static_cast<u32>(i)));
            const GuestAddr a = src + (saic ? 0 : i * 4);
            u32 v = 0;
            if (st.memory && st.memory->is_valid_address(a, 4)) st.memory->read_value(a, v);
            return v;
        };
        if (das) {
            const u32 reg = static_cast<u32>(dst >> 2);
            if (daic) {
                const u32 last = src_dword(dwords - 1);
                return write_regs(reg, &last, 1);
            }
            std::vector<u32> vals(dwords);
            for (usize i = 0; i < dwords; ++i) vals[i] = src_dword(i);
            return write_regs(reg, vals.data(), vals.size());
        }
        if (!st.memory) return true;
        if (daic) {
            if (st.memory->is_valid_address(dst, 4)) st.memory->write_value<u32>(dst, src_dword(dwords - 1));
            return true;
        }
        if (!st.memory->is_valid_address(dst, bytes)) {
            std::fprintf(stderr, "[PM4] %s: destination 0x%llx+%zu is not mapped\n", name,
                         (unsigned long long)dst, bytes);
            return true;
        }
        u8* dst_ptr = st.memory->to_host_ptr<u8>(dst);
        if (src_sel != 2 && !sas && !saic && st.memory->is_valid_address(src, bytes)) {
            std::memmove(dst_ptr, st.memory->to_host_ptr<u8>(src), bytes);
        } else {
            for (usize i = 0; i < dwords; ++i) {
                const u32 v = src_dword(i);
                std::memcpy(dst_ptr + i * 4, &v, std::min<usize>(4, bytes - i * 4));
            }
        }
        st.invalidate_resources(dst, bytes);
        return true;
    };
    auto hex_payload = [](const Pm4Packet& p, usize max_dw) {
        std::string s;
        for (usize d = 0; d < p.payload.size() && d < max_dw; ++d) {
            char b[16];
            std::snprintf(b, sizeof(b), "0x%08x ", p.payload[d]);
            s += b;
        }
        return s;
    };
    // Per-draw diagnostics (VS/PS ISA, V# contents, first indices). Other
    // subsystems read these lines from logs, so they are kept verbatim.
    auto dump_draw_state = [&](const char* tag, GuestAddr idx_addr, u32 count, MTLIndexType idx_type) {
        if (!pm4_verbose()) return;
        std::fprintf(stderr, "      VS ISA: 0x%llx | PS ISA: 0x%llx | RT: 0x%llx (vp: %.0fx%.0f [%.0f,%.0f], sc: %u,%u %ux%u) prim=0x%02x inst=%u\n",
                     (unsigned long long)st.vs_shader_addr, (unsigned long long)st.ps_shader_addr,
                     (unsigned long long)st.cb_color0_base,
                     st.viewport.width, st.viewport.height, st.viewport.x, st.viewport.y,
                     st.scissor.x, st.scissor.y, st.scissor.width, st.scissor.height,
                     static_cast<u32>(st.primitive_type), instances());
        std::fprintf(stderr, "      VtxBuf: 0x%llx (sz=%zu, stride=%zu)\n",
                     (unsigned long long)st.vertex_buffer_addr, st.vertex_buffer_size, st.vertex_stride);
        if (!st.memory) return;
        (void)tag;
        if (idx_addr > 0x1000 && count > 0) {
            const usize esz = (idx_type == MTLIndexTypeUInt32) ? 4 : 2;
            const usize n = std::min<u32>(count, 16);
            if (st.memory->is_valid_address(idx_addr, n * esz)) {
                std::string idx_s;
                for (usize i = 0; i < n; ++i) {
                    u32 v = 0;
                    if (esz == 4) st.memory->read_value(idx_addr + i * 4, v);
                    else { u16 h = 0; st.memory->read_value(idx_addr + i * 2, h); v = h; }
                    char b[16];
                    std::snprintf(b, sizeof(b), "%u ", v);
                    idx_s += b;
                }
                std::fprintf(stderr, "      Indices (u%zu): [%s]\n", esz * 8, idx_s.c_str());
            }
        }
        for (const auto& [slot, buf_desc] : st.vs_user_sgpr_buffers) {
            const usize stride = buf_desc.stride > 0 ? buf_desc.stride : 16;
            const usize dump_records = std::min<u32>(buf_desc.num_records, 24);
            const usize dump_bytes = dump_records * stride;
            if (buf_desc.base_addr <= 0x1000 || dump_bytes == 0 ||
                !st.memory->is_valid_address(buf_desc.base_addr, dump_bytes)) {
                continue;
            }
            std::vector<u8> raw(dump_bytes);
            st.memory->read_bytes(buf_desc.base_addr, raw.data(), dump_bytes);
            std::string recs_s;
            for (usize r = 0; r < dump_records; ++r) {
                float f[4] = {0, 0, 0, 0};
                std::memcpy(f, raw.data() + r * stride, std::min<usize>(16, stride));
                char b[128];
                if (stride >= 16) std::snprintf(b, sizeof(b), " r%zu:[%.3f,%.3f,%.3f,%.3f]", r, f[0], f[1], f[2], f[3]);
                else std::snprintf(b, sizeof(b), " r%zu:[%.3f,%.3f]", r, f[0], f[1]);
                recs_s += b;
            }
            std::fprintf(stderr, "      [V# s%u..s%u] base=0x%llx str=%u rec=%u:%s\n",
                         slot, slot + 3, (unsigned long long)buf_desc.base_addr, buf_desc.stride,
                         buf_desc.num_records, recs_s.c_str());
        }
        auto dump_isa = [&](const char* label, GuestAddr addr, int dws) {
            if (addr <= 0x1000 || !st.memory->is_valid_address(addr, dws * 4)) return;
            std::vector<u32> code(dws);
            st.memory->read_bytes(addr, code.data(), dws * 4);
            std::string s;
            for (int k = 0; k < dws; ++k) {
                char b[16];
                std::snprintf(b, sizeof(b), "0x%08x ", code[k]);
                s += b;
                if ((k + 1) % 8 == 0) s += "\n        ";
            }
            std::fprintf(stderr, "      %s ISA [0x%llx] (%d dw):\n        %s\n", label,
                         (unsigned long long)addr, dws, s.c_str());
        };
        dump_isa("VS", st.vs_shader_addr, 80);
        dump_isa("PS", st.ps_shader_addr, 50);
    };

    if (packet.header.type == Pm4PacketType::Type0) {
        // Type 0: consecutive writes starting at an absolute dword register.
        st.context_reg_writes += packet.payload.size();
        return write_regs(packet.header.base_reg, packet.payload.data(), packet.payload.size());
    }

    if (packet.header.type == Pm4PacketType::Type3) {
        // Shadow every SET_*_REG write into the register file first.
        {
            u32 base = 0;
            switch (packet.header.opcode) {
                case IT_SET_CONFIG_REG:  base = kPm4ConfigRegBase; break;
                case IT_SET_SH_REG:      base = kPm4ShRegBase; break;
                case IT_SET_CONTEXT_REG: base = kPm4ContextRegBase; break;
                case IT_SET_UCONFIG_REG: base = kPm4UconfigRegBase; break;
                default: break;
            }
            if (base != 0 && packet.payload.size() >= 2) {
                const u32 first = base + (packet.payload[0] & 0xFFFF);
                for (usize i = 1; i < packet.payload.size(); ++i) {
                    const u32 r = first + static_cast<u32>(i - 1);
                    if (r < st.reg_file.size()) st.reg_file[r] = packet.payload[i];
                }
            }
        }
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
                        PM4_LOG("[PM4] IT_NOP (tag=0x%08x, len=%zu): [%s]\n",
                                     tag, packet.payload.size(), dwords_str.c_str());

                        // Scan for potential 64-bit GPU base addresses in payload
                        for (size_t d = 0; d + 1 < packet.payload.size(); ++d) {
                            const u64 addr = (static_cast<u64>(packet.payload[d + 1]) << 32) | packet.payload[d];
                            if (addr > 0x1000000 && m_impl->memory && m_impl->memory->is_valid_address(addr, 20)) {
                                PM4_LOG("[PM4]   -> Found valid GPU addr in NOP[%zu]: 0x%llx\n",
                                             d, (unsigned long long)addr);
                                m_impl->gnm_base_addr = addr;
                            }
                        }
                    }
                }
                return true;
            }

            case IT_SET_BASE: {
                // [BASE_INDEX, ADDR_LO, ADDR_HI]. BASE_INDEX 1 = DRAW_INDEX_BASE:
                // the base used by DRAW_(INDEX_)INDIRECT and DISPATCH_INDIRECT.
                if (!st.require_payload(packet, 3, "IT_SET_BASE")) return false;
                const u32 base_index = packet.payload[0] & 0xF;
                const GuestAddr base_addr =
                    (static_cast<u64>(packet.payload[2] & 0xFFFFu) << 32) | (packet.payload[1] & ~7u);
                PM4_LOG("[PM4] IT_SET_BASE: index=%u, addr=0x%llx\n",
                             base_index, (unsigned long long)base_addr);
                if (base_index == 1) {
                    st.indirect_base_addr = base_addr;
                }
                return true;
            }

            case IT_CLEAR_STATE:
                if (!st.require_payload(packet, 1, "IT_CLEAR_STATE")) return false;
                reset_state();
                return true;

            case IT_CONTEXT_CONTROL:
                // [LOAD_CONTROL, SHADOW_ENABLE]: controls register shadowing
                // across hardware context switches. The emulator runs a single
                // context with a full shadow register file, so there is
                // nothing to load or shadow.
                return st.require_payload(packet, 2, "IT_CONTEXT_CONTROL");

            case IT_INDEX_BUFFER_SIZE: {
                // [NUM_INDICES] -> VGT_DMA_MAX_SIZE (in indices).
                if (!st.require_payload(packet, 1, "IT_INDEX_BUFFER_SIZE")) return false;
                st.index_buffer_size = packet.payload[0];
                return true;
            }

            case IT_INDEX_TYPE: {
                // [INDEX_TYPE[1:0] (0=16-bit, 1=32-bit), SWAP_MODE[3:2]]
                if (!st.require_payload(packet, 1, "IT_INDEX_TYPE")) return false;
                set_index_type(packet.payload[0], "IT_INDEX_TYPE");
                st.reg_file[kPm4ContextRegBase + VGT_DMA_INDEX_TYPE_REG] = packet.payload[0];
                return true;
            }

            case IT_INDEX_BASE: {
                // [BASE_LO (2-byte aligned), BASE_HI[15:0]]
                if (!st.require_payload(packet, 2, "IT_INDEX_BASE")) return false;
                st.index_base_addr =
                    (static_cast<u64>(packet.payload[1] & 0xFFFFu) << 32) | (packet.payload[0] & ~1u);
                PM4_LOG("[PM4] IT_INDEX_BASE: addr=0x%llx\n",
                             (unsigned long long)st.index_base_addr);
                return true;
            }

            case IT_SET_CONFIG_REG:
            case IT_SET_UCONFIG_REG: {
                // [REG_OFFSET, values...] relative to the config/uconfig base.
                const char* name = packet.header.opcode == IT_SET_CONFIG_REG ? "IT_SET_CONFIG_REG"
                                                                              : "IT_SET_UCONFIG_REG";
                if (!st.require_payload(packet, 2, name)) return false;
                const u32 base = packet.header.opcode == IT_SET_CONFIG_REG ? kPm4ConfigRegBase
                                                                            : kPm4UconfigRegBase;
                const u32 first = base + (packet.payload[0] & 0xFFFF);
                for (usize i = 1; i < packet.payload.size(); ++i) {
                    apply_vgt_reg(first + static_cast<u32>(i - 1), packet.payload[i]);
                }
                return true;
            }

            case IT_CONTEXT_REG_RMW: {
                // [REG_OFFSET, MASK, DATA]: reg = (reg & ~MASK) | (DATA & MASK)
                if (!st.require_payload(packet, 3, "IT_CONTEXT_REG_RMW")) return false;
                const u32 reg = kPm4ContextRegBase + (packet.payload[0] & 0xFFFF);
                const u32 mask = packet.payload[1];
                const u32 val = (read_reg(reg) & ~mask) | (packet.payload[2] & mask);
                return write_regs(reg, &val, 1);
            }

            case IT_REG_RMW: {
                // [MOD_ADRS[15:0] | AND_MASK_SRC[30] | OR_MASK_SRC[31], AND_MASK, OR_MASK]
                // *_SRC=1 takes the mask from the register following MOD_ADRS.
                if (!st.require_payload(packet, 3, "IT_REG_RMW")) return false;
                const u32 reg = packet.payload[0] & 0xFFFF;
                const u32 and_mask = (packet.payload[0] & (1u << 30)) ? read_reg(reg + 1) : packet.payload[1];
                const u32 or_mask = (packet.payload[0] & (1u << 31)) ? read_reg(reg + 1) : packet.payload[2];
                const u32 val = (read_reg(reg) & and_mask) | or_mask;
                return write_regs(reg, &val, 1);
            }

            case IT_LOAD_CONFIG_REG:
            case IT_LOAD_SH_REG:
            case IT_LOAD_CONTEXT_REG:
            case IT_LOAD_UCONFIG_REG: {
                // [BASE_LO, BASE_HI, (REG_OFFSET, NUM_DWORDS)*]. Registers are
                // loaded from BASE + REG_OFFSET*4.
                if (!st.require_payload(packet, 4, pm4_opcode_to_name(packet.header.opcode))) return false;
                u32 base = kPm4ConfigRegBase;
                if (packet.header.opcode == IT_LOAD_SH_REG) base = kPm4ShRegBase;
                else if (packet.header.opcode == IT_LOAD_CONTEXT_REG) base = kPm4ContextRegBase;
                else if (packet.header.opcode == IT_LOAD_UCONFIG_REG) base = kPm4UconfigRegBase;
                const GuestAddr mem_base =
                    (static_cast<u64>(packet.payload[1] & 0xFFFFu) << 32) | (packet.payload[0] & ~3u);
                bool ok = true;
                for (usize i = 2; i + 1 < packet.payload.size(); i += 2) {
                    const u32 off = packet.payload[i] & 0xFFFF;
                    const u32 n = packet.payload[i + 1] & 0x3FFF;
                    const GuestAddr src = mem_base + static_cast<GuestAddr>(off) * 4;
                    if (n == 0) continue;
                    if (!st.memory || !st.memory->is_valid_address(src, n * 4ull)) {
                        std::fprintf(stderr, "[PM4] %s: source 0x%llx+%u dw is not mapped\n",
                                     pm4_opcode_to_name(packet.header.opcode), (unsigned long long)src, n);
                        ok = false;
                        continue;
                    }
                    std::vector<u32> vals(n);
                    st.memory->read_bytes(src, vals.data(), n * 4ull);
                    ok = write_regs(base + off, vals.data(), n) && ok;
                }
                return ok;
            }

            case IT_SET_CONTEXT_REG: {
                if (!m_impl->require_payload(packet, 2, "IT_SET_CONTEXT_REG")) return false;
                const u16 base_reg = static_cast<u16>(packet.payload[0] & 0xFFFF);
                std::string cb_vals;
                for (size_t d = 1; d < packet.payload.size() && d < 10; ++d) {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "0x%08x ", packet.payload[d]);
                    cb_vals += buf;
                }
                PM4_LOG("[PM4 Context] base_reg=0x%04x count=%zu: [%s]\n",
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
                        m_impl->dirty_bits |= Impl::DirtyScissor;
                        m_impl->scissor.x = val & 0x7FFF;
                        m_impl->scissor.y = (val >> 16) & 0x7FFF;
                    } else if (reg == PA_SC_SCREEN_SCISSOR_BR) {
                        m_impl->dirty_bits |= Impl::DirtyScissor;
                        const u32 br_x = val & 0x7FFF;
                        const u32 br_y = (val >> 16) & 0x7FFF;
                        m_impl->scissor.width = (br_x > m_impl->scissor.x) ? (br_x - m_impl->scissor.x) : 0;
                        m_impl->scissor.height = (br_y > m_impl->scissor.y) ? (br_y - m_impl->scissor.y) : 0;
                    } else if (reg == CB_COLOR_CLEAR_WORD0 || reg == CB_COLOR0_CLEAR_WORD0_GFX7) {
                        m_impl->dirty_bits |= Impl::DirtyRenderTarget;
                        m_impl->clear_color.r = static_cast<f32>(val & 0xFF) / 255.0f;
                        m_impl->clear_color.g = static_cast<f32>((val >> 8) & 0xFF) / 255.0f;
                    } else if (reg == CB_COLOR_CLEAR_WORD1 || reg == CB_COLOR0_CLEAR_WORD1_GFX7) {
                        m_impl->dirty_bits |= Impl::DirtyRenderTarget;
                        m_impl->clear_color.b = static_cast<f32>(val & 0xFF) / 255.0f;
                        m_impl->clear_color.a = static_cast<f32>((val >> 8) & 0xFF) / 255.0f;
                    } else if (reg == CB_COLOR0_BASE || reg == CB_COLOR0_BASE_GFX7) {
                        m_impl->dirty_bits |= Impl::DirtyRenderTarget;
                        m_impl->cb_color0_base = static_cast<GuestAddr>(val) << 8;
                        PM4_LOG("[PM4 Context] CB_COLOR0_BASE: 0x%llx (raw=0x%08x)\n",
                                     (unsigned long long)m_impl->cb_color0_base, val);
                    } else if (reg == CB_COLOR0_VIEW || reg == CB_COLOR0_VIEW_GFX7) {
                        m_impl->dirty_bits |= Impl::DirtyRenderTarget;
                        m_impl->cb_color0_view = val;
                        PM4_LOG("[PM4 Context] CB_COLOR0_VIEW: 0x%08x\n", val);
                    } else if (reg == CB_COLOR0_INFO || reg == CB_COLOR0_INFO_GFX7) {
                        m_impl->dirty_bits |= Impl::DirtyRenderTarget;
                        m_impl->cb_color0_info = val;
                        PM4_LOG("[PM4 Context] CB_COLOR0_INFO: 0x%08x\n", val);
                    } else if (reg == VGT_DMA_MAX_SIZE_REG) {
                        m_impl->dirty_bits |= Impl::DirtyVertexLayout;
                        m_impl->index_buffer_size = val;
                        PM4_LOG("[PM4 Context] VGT_DMA_MAX_SIZE: %u\n", val);
                    } else if (reg == VGT_DMA_INDEX_TYPE_REG) {
                        m_impl->dirty_bits |= Impl::DirtyVertexLayout;
                        m_impl->index_type = (val & 1) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
                        PM4_LOG("[PM4 Context] VGT_DMA_INDEX_TYPE: %s (%u)\n",
                                     (val & 1) ? "UInt32" : "UInt16", val);
                    }
                    // Note: VGT_PRIMITIVE_TYPE is not a context register on GCN;
                    // it is handled by SET_UCONFIG_REG (CIK) / SET_CONFIG_REG (SI).
                }
                m_impl->context_reg_writes += (packet.payload.size() - 1);
                return true;
            }

            case IT_SET_SH_REG: {
                if (!m_impl->require_payload(packet, 2, "IT_SET_SH_REG")) return false;
                {
                    const u16 base_reg = static_cast<u16>(packet.payload[0] & 0xFFFF);
                    std::string vals_str;
                    for (size_t d = 1; d < packet.payload.size() && d < 9; ++d) {
                        char buf[32];
                        std::snprintf(buf, sizeof(buf), "0x%08x ", packet.payload[d]);
                        vals_str += buf;
                    }
                    PM4_LOG("[PM4 SH_REG] base_reg=0x%04x count=%zu: [%s]\n",
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
                            PM4_LOG("[PM4 SH_REG] VS Shader ISA Addr: 0x%llx (lo=0x%08x, hi=0x%08x, rsrc1=0x%08x, rsrc2=0x%08x)\n",
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
                            PM4_LOG("[PM4 SH_REG] PS Shader ISA Addr: 0x%llx (lo=0x%08x, hi=0x%08x, rsrc1=0x%08x, rsrc2=0x%08x)\n",
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
                                        PM4_LOG("[PM4 SH_REG] VS User SGPR s[%u..%u] V# Buffer: base=0x%llx stride=%u records=%u (sz=%u) fmt=0x%08x\n",
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
                                    m_impl->ps_texture_pitch = (w3 & 0xFFFFu);
                                    m_impl->ps_texture_format = w1;
                                    m_impl->ps_texture_w0 = w0;
                                    m_impl->ps_texture_w1 = w1;
                                    m_impl->ps_texture_w2 = w2;
                                    m_impl->ps_texture_w3 = w3;
                                    PM4_LOG("[PM4 SH_REG] PS T# Texture: base=0x%llx dim=%ux%u w1=0x%08x w3=0x%08x\n",
                                                 (unsigned long long)tex_base, w_m1 + 1, h_m1 + 1, w1, w3);
                                }
                            }
                        }
                    }
                }
                m_impl->mark_pipeline_dirty();
                m_impl->mark_resource_dirty();
                return true;
            }

            case IT_DRAW_INDEX_AUTO: {
                // [INDEX_COUNT, DRAW_INITIATOR]. Topology comes from
                // VGT_PRIMITIVE_TYPE, instance count from NUM_INSTANCES.
                if (!st.require_payload(packet, 2, "IT_DRAW_INDEX_AUTO")) return false;
                const u32 count = packet.payload[0];
                const u32 initiator = packet.payload[1];
                PM4_LOG("[PM4 Draw #%zu] IT_DRAW_INDEX_AUTO: count=%u initiator=0x%08x\n",
                             st.draw_count + 1, count, initiator);
                if ((initiator & 3) != kDiSrcSelAutoIndex) {
                    PM4_LOG("[PM4]   DRAW_INDEX_AUTO with SOURCE_SELECT=%u (expected AUTO_INDEX)\n",
                                 initiator & 3);
                }
                dump_draw_state("DRAW_INDEX_AUTO", 0, 0, st.index_type);
                // RectList/QuadList are lowered in render_auto_mesh (translated
                // GCN path: in-shader expansion; fixed fallback: see there).
                return st.render_auto_mesh(count, st.primitive_type, 0, instances());
            }

            case IT_DRAW_INDEX_2: {
                // [MAX_SIZE, INDEX_BASE_LO, INDEX_BASE_HI, INDEX_COUNT, DRAW_INITIATOR]
                if (!st.require_payload(packet, 5, "IT_DRAW_INDEX_2")) return false;
                const u32 max_size = packet.payload[0];
                const GuestAddr idx_addr =
                    (static_cast<u64>(packet.payload[2] & 0xFFFFu) << 32) | (packet.payload[1] & ~1u);
                const u32 count = packet.payload[3];
                const u32 initiator = packet.payload[4];
                PM4_LOG("[PM4 Draw #%zu] IT_DRAW_INDEX_2: idx_addr=0x%llx count=%u max=%u initiator=0x%08x prim=0x%02x %s\n",
                             st.draw_count + 1, (unsigned long long)idx_addr, count, max_size, initiator,
                             static_cast<u32>(st.primitive_type),
                             st.index_type == MTLIndexTypeUInt32 ? "u32" : "u16");
                if (count > max_size) {
                    PM4_LOG("[PM4]   DRAW_INDEX_2: index_count %u exceeds MAX_SIZE %u\n", count, max_size);
                }
                st.index_buffer_size = max_size;
                dump_draw_state("DRAW_INDEX_2", idx_addr, count, st.index_type);
                return st.render_indexed_mesh(idx_addr, count, st.primitive_type, st.index_type, 0, instances());
            }

            case IT_DRAW_INDEX_OFFSET_2: {
                // [MAX_SIZE, INDEX_OFFSET, INDEX_COUNT, DRAW_INITIATOR]; indices
                // are fetched from INDEX_BASE + INDEX_OFFSET elements.
                if (!st.require_payload(packet, 4, "IT_DRAW_INDEX_OFFSET_2")) return false;
                const usize esz = (st.index_type == MTLIndexTypeUInt32) ? 4 : 2;
                const GuestAddr idx_addr = st.index_base_addr + static_cast<GuestAddr>(packet.payload[1]) * esz;
                const u32 count = packet.payload[2];
                PM4_LOG("[PM4 Draw #%zu] IT_DRAW_INDEX_OFFSET_2: idx_addr=0x%llx count=%u max=%u\n",
                             st.draw_count + 1, (unsigned long long)idx_addr, count, packet.payload[0]);
                st.index_buffer_size = packet.payload[0];
                dump_draw_state("DRAW_INDEX_OFFSET_2", idx_addr, count, st.index_type);
                return st.render_indexed_mesh(idx_addr, count, st.primitive_type, st.index_type, 0, instances());
            }

            case IT_NUM_INSTANCES: {
                if (!st.require_payload(packet, 1, "IT_NUM_INSTANCES")) return false;
                st.num_instances = packet.payload[0];
                st.reg_file[kPm4UconfigRegBase + UCONFIG_VGT_NUM_INSTANCES] = packet.payload[0];
                return true;
            }

            case IT_DRAW_INDIRECT:
            case IT_DRAW_INDEX_INDIRECT: {
                // CIK: [DATA_OFFSET, BASE_VTX_LOC, START_INST_LOC, DRAW_INITIATOR]
                // SI : [DATA_OFFSET, BASE_VTX_LOC | START_INST_LOC<<16, DRAW_INITIATOR]
                // Arguments live at the SET_BASE(DRAW_INDEX_BASE) address + DATA_OFFSET.
                const bool indexed = packet.header.opcode == IT_DRAW_INDEX_INDIRECT;
                const char* name = indexed ? "IT_DRAW_INDEX_INDIRECT" : "IT_DRAW_INDIRECT";
                if (!st.require_payload(packet, 3, name)) return false;
                const u32 data_offset = packet.payload[0];
                const GuestAddr args_addr = st.indirect_base_addr + data_offset;
                PM4_LOG("[PM4 Draw #%zu] %s: base=0x%llx offset=0x%x payload=[%s]\n",
                             st.draw_count + 1, name, (unsigned long long)st.indirect_base_addr, data_offset,
                             hex_payload(packet, 4).c_str());
                const usize args_size = indexed ? 20 : 16;
                if (st.indirect_base_addr == 0 || !st.memory ||
                    !st.memory->is_valid_address(args_addr, args_size)) {
                    std::fprintf(stderr, "[PM4]   -> indirect args at 0x%llx are not mapped (SET_BASE missing?)\n",
                                 (unsigned long long)args_addr);
                    return false;
                }
                u32 a[5] = {0, 0, 0, 0, 0};
                st.memory->read_bytes(args_addr, a, args_size);
                if (indexed) {
                    // {index_count, instance_count, start_index, base_vertex, start_instance}
                    const usize esz = (st.index_type == MTLIndexTypeUInt32) ? 4 : 2;
                    const GuestAddr idx_start = st.index_base_addr + static_cast<GuestAddr>(a[2]) * esz;
                    PM4_LOG("[PM4]   -> args@0x%llx: idxCount=%u instCount=%u firstIdx=%u vtxOffset=%d firstInst=%u\n",
                                 (unsigned long long)args_addr, a[0], a[1], a[2], static_cast<i32>(a[3]), a[4]);
                    if (a[0] == 0 || a[1] == 0) return true; // empty draw
                    dump_draw_state(name, idx_start, a[0], st.index_type);
                    return st.render_indexed_mesh(idx_start, a[0], st.primitive_type, st.index_type,
                                                  static_cast<i32>(a[3]), a[1]);
                }
                // {vertex_count, instance_count, start_vertex, start_instance}
                PM4_LOG("[PM4]   -> args@0x%llx: vtxCount=%u instCount=%u firstVtx=%u firstInst=%u\n",
                             (unsigned long long)args_addr, a[0], a[1], a[2], a[3]);
                if (a[0] == 0 || a[1] == 0) return true;
                dump_draw_state(name, 0, 0, st.index_type);
                return st.render_auto_mesh(a[0], st.primitive_type, static_cast<i32>(a[2]), a[1]);
            }

            case IT_WRITE_DATA: {
                // [CONTROL, DST_ADDR_LO, DST_ADDR_HI, DATA...]
                // CONTROL: DST_SEL[11:8] (0 register, 1 memory sync, 2 TC_L2,
                // 5 memory async), WR_ONE_ADDR[16], WR_CONFIRM[20], ENGINE[31:30].
                if (!st.require_payload(packet, 4, "IT_WRITE_DATA")) return false;
                const u32 control = packet.payload[0];
                const u32 dst_sel = (control >> 8) & 0xF;
                const bool one_addr = (control & (1u << 16)) != 0;
                const u32* data = &packet.payload[3];
                const usize n = packet.payload.size() - 3;
                if (dst_sel == 0) {
                    const u32 reg = packet.payload[1] & 0xFFFF;
                    if (one_addr) return write_regs(reg, &data[n - 1], 1);
                    return write_regs(reg, data, n);
                }
                if (dst_sel != 1 && dst_sel != 2 && dst_sel != 5) {
                    std::fprintf(stderr, "[PM4] IT_WRITE_DATA: unsupported DST_SEL=%u\n", dst_sel);
                    return false;
                }
                const GuestAddr dst =
                    (static_cast<u64>(packet.payload[2]) << 32) | (packet.payload[1] & ~3u);
                const usize bytes = one_addr ? 4 : n * 4;
                if (!st.memory || dst == 0 || !st.memory->is_valid_address(dst, bytes)) {
                    std::fprintf(stderr, "[PM4] IT_WRITE_DATA: destination 0x%llx+%zu is not mapped\n",
                                 (unsigned long long)dst, bytes);
                    return true;
                }
                if (one_addr) {
                    st.memory->write_value<u32>(dst, data[n - 1]);
                } else {
                    st.memory->write_bytes(dst, data, bytes);
                }
                st.invalidate_resources(dst, bytes);
                return true;
            }

            case IT_WAIT_REG_MEM: {
                // [CONTROL, POLL_ADDR_LO|REG, POLL_ADDR_HI, REFERENCE, MASK, POLL_INTERVAL]
                // CONTROL: FUNCTION[2:0] (0 always, 1 <, 2 <=, 3 ==, 4 !=, 5 >=, 6 >),
                // MEM_SPACE[4] (0 register, 1 memory), OPERATION[7:6], ENGINE[8].
                if (!st.require_payload(packet, 6, "IT_WAIT_REG_MEM")) return false;
                const u32 control = packet.payload[0];
                const u32 func = control & 0x7u;
                const bool is_mem = (control & 0x10u) != 0;
                const u32 ref = packet.payload[3];
                const u32 mask = packet.payload[4];
                auto passes = [func, ref, mask](u32 v) {
                    const u32 x = v & mask;
                    switch (func) {
                        case 0: return true;
                        case 1: return x < ref;
                        case 2: return x <= ref;
                        case 3: return x == ref;
                        case 4: return x != ref;
                        case 5: return x >= ref;
                        case 6: return x > ref;
                        default: return true;
                    }
                };
                if (!is_mem) {
                    const u32 reg = packet.payload[1] & 0xFFFF;
                    if (!passes(read_reg(reg))) {
                        PM4_LOG("[PM4] IT_WAIT_REG_MEM: register 0x%04x=0x%08x never satisfies func=%u ref=0x%08x mask=0x%08x; continuing\n",
                                     reg, read_reg(reg), func, ref, mask);
                    }
                    return true;
                }
                const GuestAddr poll_addr =
                    (static_cast<u64>(packet.payload[2] & 0xFFFFu) << 32) | (packet.payload[1] & ~3u);
                if (!st.memory || poll_addr == 0 || !st.memory->is_valid_address(poll_addr, 4)) {
                    std::fprintf(stderr, "[PM4] IT_WAIT_REG_MEM: poll address 0x%llx is not mapped\n",
                                 (unsigned long long)poll_addr);
                    return true;
                }
                u32 cur = 0;
                st.memory->read_value(poll_addr, cur);
                if (passes(cur)) return true;
                // The command processor executes the stream synchronously on
                // the submitting thread, so a wait that is not yet satisfied
                // can never be satisfied by a later packet of this stream. The
                // GPU never writes the polled location itself, so neither do
                // we: report the unsatisfied wait and continue.
                PM4_LOG("[PM4] IT_WAIT_REG_MEM: unsatisfied wait on 0x%llx (cur=0x%08x func=%u ref=0x%08x mask=0x%08x); continuing\n",
                             (unsigned long long)poll_addr, cur, func, ref, mask);
                return true;
            }

            case IT_EVENT_WRITE: {
                // [EVENT_TYPE[5:0] | EVENT_INDEX[11:8], (ADDR_LO, ADDR_HI)]
                // Cache flush / pipeline events have no observable effect on a
                // synchronously executed stream. Events with a memory payload
                // (ZPASS_DONE, SAMPLE_PIPELINESTAT, ...) write query results,
                // which are reported as zero.
                if (!st.require_payload(packet, 1, "IT_EVENT_WRITE")) return false;
                if (packet.payload.size() >= 3 && st.memory) {
                    const GuestAddr addr =
                        (static_cast<u64>(packet.payload[2] & 0xFFFFu) << 32) | (packet.payload[1] & ~7u);
                    const u32 event_type = packet.payload[0] & 0x3F;
                    std::fprintf(stderr, "[PM4] IT_EVENT_WRITE: event=0x%02x query addr=0x%llx (results not emulated)\n",
                                 event_type, (unsigned long long)addr);
                }
                return true;
            }

            case IT_EVENT_WRITE_EOP: {
                // [EVENT_CNTL, ADDR_LO, ADDR_HI[15:0] | INT_SEL[25:24] | DATA_SEL[31:29], DATA_LO, DATA_HI]
                if (!st.require_payload(packet, 5, "IT_EVENT_WRITE_EOP")) return false;
                const u32 dw2 = packet.payload[2];
                const GuestAddr addr = (static_cast<u64>(dw2 & 0xFFFFu) << 32) | packet.payload[1];
                const u32 data_sel = (dw2 >> 29) & 0x7;
                const u32 int_sel = (dw2 >> 24) & 0x3;
                PM4_LOG("[PM4] IT_EVENT_WRITE_EOP: event=0x%02x addr=0x%llx data_sel=%u int_sel=%u data=0x%08x_%08x\n",
                             packet.payload[0] & 0x3F, (unsigned long long)addr, data_sel, int_sel,
                             packet.payload[4], packet.payload[3]);
                // Interrupt delivery (INT_SEL) is owned by the HLE equeue path.
                return write_end_of_pipe("IT_EVENT_WRITE_EOP", addr, data_sel, packet.payload[3], packet.payload[4]);
            }

            case IT_EVENT_WRITE_EOS: {
                // [EVENT_CNTL, ADDR_LO, ADDR_HI[15:0] | CMD[31:29], DATA]
                // CMD: 1 = store GDS data, 2 = store 32-bit DATA.
                if (!st.require_payload(packet, 4, "IT_EVENT_WRITE_EOS")) return false;
                const u32 dw2 = packet.payload[2];
                const GuestAddr addr = (static_cast<u64>(dw2 & 0xFFFFu) << 32) | packet.payload[1];
                const u32 cmd = (dw2 >> 29) & 0x7;
                if (cmd == 2) return write_end_of_pipe("IT_EVENT_WRITE_EOS", addr, 1, packet.payload[3], 0);
                if (cmd == 1) {
                    std::fprintf(stderr, "[PM4] IT_EVENT_WRITE_EOS: GDS store is not emulated\n");
                    return false;
                }
                return true;
            }

            case IT_RELEASE_MEM: {
                // [EVENT_CNTL, DST_SEL[17:16] | INT_SEL[26:24] | DATA_SEL[31:29],
                //  ADDR_LO, ADDR_HI, DATA_LO, DATA_HI]
                if (!st.require_payload(packet, 6, "IT_RELEASE_MEM")) return false;
                const u32 dw1 = packet.payload[1];
                const GuestAddr addr = (static_cast<u64>(packet.payload[3]) << 32) | packet.payload[2];
                const u32 data_sel = (dw1 >> 29) & 0x7;
                PM4_LOG("[PM4] IT_RELEASE_MEM: event=0x%02x addr=0x%llx data_sel=%u int_sel=%u data=0x%08x_%08x\n",
                             packet.payload[0] & 0x3F, (unsigned long long)addr, data_sel, (dw1 >> 24) & 0x7,
                             packet.payload[5], packet.payload[4]);
                return write_end_of_pipe("IT_RELEASE_MEM", addr, data_sel, packet.payload[4], packet.payload[5]);
            }

            case IT_DMA_DATA: {
                // [CONTROL, SRC_LO|DATA, SRC_HI, DST_LO, DST_HI, COMMAND]
                // CONTROL: DST_SEL[21:20], SRC_SEL[30:29], CP_SYNC[31].
                // COMMAND: BYTE_COUNT[20:0], SAS[26], DAS[27], SAIC[28], DAIC[29].
                if (!st.require_payload(packet, 6, "IT_DMA_DATA")) return false;
                const u32 control = packet.payload[0];
                const u32 command = packet.payload[5];
                const u32 src_sel = (control >> 29) & 0x3;
                const u32 dst_sel = (control >> 20) & 0x3;
                const GuestAddr src = (static_cast<u64>(packet.payload[2]) << 32) | packet.payload[1];
                const GuestAddr dst = (static_cast<u64>(packet.payload[4]) << 32) | packet.payload[3];
                const usize bytes = command & 0x1FFFFF;
                const bool sas = (command >> 26) & 1, das = (command >> 27) & 1;
                const bool saic = (command >> 28) & 1, daic = (command >> 29) & 1;
                PM4_LOG("[PM4] IT_DMA_DATA: src_sel=%u dst_sel=%u src=0x%llx%s dst=0x%llx%s bytes=%zu\n",
                             src_sel, dst_sel, (unsigned long long)src, sas ? "(reg)" : "",
                             (unsigned long long)dst, das ? "(reg)" : "", bytes);
                return run_dma("IT_DMA_DATA", src_sel, dst_sel, src, dst, bytes, sas, das, saic, daic,
                               packet.payload[1]);
            }

            case IT_CP_DMA: {
                // [SRC_LO|DATA, SRC_HI[15:0] | DST_SEL[21:20] | SRC_SEL[30:29] | CP_SYNC[31],
                //  DST_LO, DST_HI[15:0], COMMAND (as DMA_DATA)]
                if (!st.require_payload(packet, 5, "IT_CP_DMA")) return false;
                const u32 dw1 = packet.payload[1];
                const u32 command = packet.payload[4];
                const GuestAddr src = (static_cast<u64>(dw1 & 0xFFFFu) << 32) | packet.payload[0];
                const GuestAddr dst = (static_cast<u64>(packet.payload[3] & 0xFFFFu) << 32) | packet.payload[2];
                return run_dma("IT_CP_DMA", (dw1 >> 29) & 0x3, (dw1 >> 20) & 0x3, src, dst,
                               command & 0x1FFFFF, (command >> 26) & 1, (command >> 27) & 1,
                               (command >> 28) & 1, (command >> 29) & 1, packet.payload[0]);
            }

            case IT_ACQUIRE_MEM:
            case IT_SURFACE_SYNC: {
                // ACQUIRE_MEM: [COHER_CNTL, COHER_SIZE, COHER_SIZE_HI, COHER_BASE_LO, COHER_BASE_HI, POLL]
                // SURFACE_SYNC: [COHER_CNTL, COHER_SIZE, COHER_BASE, POLL]
                // Sizes/bases are in 256-byte units; SIZE=0xFFFFFFFF means all.
                // Host-side resource caches are content-validated on use, so a
                // full-range acquire needs no work; a bounded range drops the
                // cached Metal copies covering it.
                const bool acquire = packet.header.opcode == IT_ACQUIRE_MEM;
                if (!st.require_payload(packet, acquire ? 6 : 4,
                                        acquire ? "IT_ACQUIRE_MEM" : "IT_SURFACE_SYNC")) return false;
                const u64 size_units = acquire
                    ? ((static_cast<u64>(packet.payload[2] & 0xFF) << 32) | packet.payload[1])
                    : packet.payload[1];
                const u64 base_units = acquire
                    ? ((static_cast<u64>(packet.payload[4] & 0xFF) << 32) | packet.payload[3])
                    : packet.payload[2];
                const bool full = (packet.payload[1] == 0xFFFFFFFFu);
                if (!full && size_units != 0) {
                    st.invalidate_resources(base_units << 8, static_cast<usize>(size_units << 8));
                }
                return true;
            }

            case IT_INDIRECT_BUFFER:
            case IT_INDIRECT_BUFFER_CNST: {
                // [IB_BASE_LO, IB_BASE_HI[15:0], IB_SIZE[19:0] | CHAIN[20] | VALID[23] | VMID[31:24]]
                const char* name = packet.header.opcode == IT_INDIRECT_BUFFER ? "IT_INDIRECT_BUFFER"
                                                                               : "IT_INDIRECT_BUFFER_CNST";
                if (!st.require_payload(packet, 3, name)) return false;
                const GuestAddr ib = (static_cast<u64>(packet.payload[1] & 0xFFFFu) << 32) | (packet.payload[0] & ~3u);
                const usize ib_dwords = packet.payload[2] & 0xFFFFF;
                if (st.ib_depth >= 8) {
                    std::fprintf(stderr, "[PM4] %s: nesting too deep at 0x%llx\n", name, (unsigned long long)ib);
                    return false;
                }
                ++st.ib_depth;
                const usize executed = execute_command_buffer(ib, ib_dwords * 4);
                --st.ib_depth;
                return ib_dwords == 0 || executed > 0;
            }

            case IT_COND_EXEC: {
                // [BOOL_ADDR_LO, BOOL_ADDR_HI, CONTROL, EXEC_COUNT[13:0]]:
                // skip the next EXEC_COUNT dwords when the 32-bit value is 0.
                if (!st.require_payload(packet, 4, "IT_COND_EXEC")) return false;
                const GuestAddr addr = (static_cast<u64>(packet.payload[1] & 0xFFFFu) << 32) | (packet.payload[0] & ~3u);
                u32 v = 1;
                if (st.memory && st.memory->is_valid_address(addr, 4)) st.memory->read_value(addr, v);
                if (v == 0) st.pending_skip_dwords = packet.payload[3] & 0x3FFF;
                return true;
            }

            case IT_PRED_EXEC: {
                // [EXEC_COUNT[13:0] | DEVICE_SELECT[31:24]]: single-GPU device mask bit 0.
                if (!st.require_payload(packet, 1, "IT_PRED_EXEC")) return false;
                if (((packet.payload[0] >> 24) & 1u) == 0) st.pending_skip_dwords = packet.payload[0] & 0x3FFF;
                return true;
            }

            case IT_SET_PREDICATION:
                // Render predication on occlusion results. Occlusion queries are
                // not emulated, so predicated draws always execute (the "draw"
                // outcome the hardware takes when results are unavailable).
                return st.require_payload(packet, 2, "IT_SET_PREDICATION");

            case IT_WRITE_CONST_RAM: {
                // [OFFSET[15:0] (bytes), DATA...]
                if (!st.require_payload(packet, 2, "IT_WRITE_CONST_RAM")) return false;
                const usize off = packet.payload[0] & 0xFFFC;
                const usize bytes = (packet.payload.size() - 1) * 4;
                if (off + bytes > st.ce_ram.size()) return false;
                std::memcpy(st.ce_ram.data() + off, &packet.payload[1], bytes);
                return true;
            }

            case IT_LOAD_CONST_RAM: {
                // [ADDR_LO, ADDR_HI, NUM_DW[14:0], START_ADDR[15:0] (bytes)]
                if (!st.require_payload(packet, 4, "IT_LOAD_CONST_RAM")) return false;
                const GuestAddr src = (static_cast<u64>(packet.payload[1]) << 32) | (packet.payload[0] & ~3u);
                const usize bytes = static_cast<usize>(packet.payload[2] & 0x7FFF) * 4;
                const usize off = packet.payload[3] & 0xFFFC;
                if (off + bytes > st.ce_ram.size() || !st.memory || !st.memory->is_valid_address(src, bytes)) return false;
                st.memory->read_bytes(src, st.ce_ram.data() + off, bytes);
                return true;
            }

            case IT_DUMP_CONST_RAM: {
                // [OFFSET[15:0] (bytes), NUM_DW[14:0], ADDR_LO, ADDR_HI]
                if (!st.require_payload(packet, 4, "IT_DUMP_CONST_RAM")) return false;
                const usize off = packet.payload[0] & 0xFFFC;
                const usize bytes = static_cast<usize>(packet.payload[1] & 0x7FFF) * 4;
                const GuestAddr dst = (static_cast<u64>(packet.payload[3]) << 32) | (packet.payload[2] & ~3u);
                if (off + bytes > st.ce_ram.size() || !st.memory || !st.memory->is_valid_address(dst, bytes)) return false;
                st.memory->write_bytes(dst, st.ce_ram.data() + off, bytes);
                st.invalidate_resources(dst, bytes);
                return true;
            }

            case IT_INCREMENT_CE_COUNTER:
            case IT_INCREMENT_DE_COUNTER:
            case IT_WAIT_ON_CE_COUNTER:
            case IT_WAIT_ON_DE_COUNTER_DIFF:
            case IT_SWITCH_BUFFER:
            case IT_PFP_SYNC_ME:
            case IT_PREAMBLE_CNTL:
            case IT_REWIND:
                // CE/DE and PFP/ME synchronization primitives. The emulator
                // executes a single in-order stream, so every such wait is
                // already satisfied when the packet is reached.
                return true;

            case IT_COPY_DATA: {
                // [CONTROL, SRC_LO|REG|IMM_LO, SRC_HI|IMM_HI, DST_LO|REG, DST_HI]
                // CONTROL: SRC_SEL[3:0] (0 reg, 1 mem, 2 TC_L2, 5 imm, 9 GPU clock),
                // DST_SEL[11:8] (0 reg, 1 mem, 2 TC_L2, 5 mem async), COUNT_SEL[16].
                if (!st.require_payload(packet, 5, "IT_COPY_DATA")) return false;
                const u32 control = packet.payload[0];
                const u32 src_sel = control & 0xF;
                const u32 dst_sel = (control >> 8) & 0xF;
                const bool is64 = (control & (1u << 16)) != 0;
                const usize bytes = is64 ? 8 : 4;
                const GuestAddr align = is64 ? ~GuestAddr{7} : ~GuestAddr{3};
                u64 value = 0;
                switch (src_sel) {
                    case 0: {
                        const u32 reg = packet.payload[1] & 0xFFFF;
                        value = read_reg(reg) | (is64 ? static_cast<u64>(read_reg(reg + 1)) << 32 : 0);
                        break;
                    }
                    case 1:
                    case 2: {
                        const GuestAddr src = ((static_cast<u64>(packet.payload[2]) << 32) | packet.payload[1]) & align;
                        if (!st.memory || !st.memory->is_valid_address(src, bytes)) {
                            std::fprintf(stderr, "[PM4] IT_COPY_DATA: source 0x%llx is not mapped\n",
                                         (unsigned long long)src);
                            return true;
                        }
                        st.memory->read_bytes(src, &value, bytes);
                        break;
                    }
                    case 5:
                        value = packet.payload[1] | (is64 ? static_cast<u64>(packet.payload[2]) << 32 : 0);
                        break;
                    case 9:
                        value = gpu_clock();
                        break;
                    default:
                        std::fprintf(stderr, "[PM4] IT_COPY_DATA: unsupported SRC_SEL=%u\n", src_sel);
                        return false;
                }
                if (dst_sel == 0) {
                    const u32 vals[2] = {static_cast<u32>(value), static_cast<u32>(value >> 32)};
                    return write_regs(packet.payload[3] & 0xFFFF, vals, is64 ? 2 : 1);
                }
                if (dst_sel != 1 && dst_sel != 2 && dst_sel != 5) {
                    std::fprintf(stderr, "[PM4] IT_COPY_DATA: unsupported DST_SEL=%u\n", dst_sel);
                    return false;
                }
                const GuestAddr dst = ((static_cast<u64>(packet.payload[4]) << 32) | packet.payload[3]) & align;
                if (!st.memory || !st.memory->is_valid_address(dst, bytes)) {
                    std::fprintf(stderr, "[PM4] IT_COPY_DATA: destination 0x%llx is not mapped\n",
                                 (unsigned long long)dst);
                    return true;
                }
                st.memory->write_bytes(dst, &value, bytes);
                return true;
            }

            default:
                if (st.metal && st.metal->frame_in_progress()) {
                    st.frame_stats.unsupported_packets++;
                    PM4_LOG("[PM4 UNIMPL] op=0x%02x(%s) count=%zu d0=0x%08x d1=0x%08x\n",
                                 packet.header.opcode, pm4_opcode_to_name(packet.header.opcode),
                                 packet.payload.size(),
                                 packet.payload.size() > 0 ? packet.payload[0] : 0,
                                 packet.payload.size() > 1 ? packet.payload[1] : 0);
                }
                ++st.unsupported_packets;
                return false;
        }
    }

    return true;
}

usize Pm4Translator::execute_command_buffer(GuestAddr dcb_gpu_addr, usize size_bytes) {
    if (!m_impl || !m_impl->memory || dcb_gpu_addr == 0 || size_bytes < sizeof(u32)) {
        return 0;
    }
    std::lock_guard<std::recursive_mutex> frame_lock(m_impl->frame_mutex);
    if (!m_impl->memory->is_valid_address(dcb_gpu_addr, size_bytes)) {
        return 0;
    }

    const u32* live_dwords = m_impl->memory->to_host_ptr<u32>(dcb_gpu_addr);
    if (!live_dwords) return 0;

    const usize num_dwords = size_bytes / sizeof(u32);
    // Parse a copy taken at submit: executing packets can take long enough
    // (shader translation, DMA) for the live buffer to change under us.
    const std::vector<u32> snapshot(live_dwords, live_dwords + num_dwords);
    const u32* dcb_dwords = snapshot.data();

    usize offset = 0;
    usize pkts_executed = 0;
    std::map<u8, usize> op_counts;
    usize t0_count = 0, t2_count = 0, t3_count = 0;
    m_impl->pending_skip_dwords = 0;

    while (offset < num_dwords) {
        if (m_impl->pending_skip_dwords > 0) {
            // COND_EXEC / PRED_EXEC evaluated false: skip the guarded dwords.
            const usize skip = std::min(m_impl->pending_skip_dwords, num_dwords - offset);
            offset += skip;
            m_impl->pending_skip_dwords = 0;
            continue;
        }
        const u32 hdr_dword = dcb_dwords[offset++];
        Pm4Header hdr = decode_pm4_header(hdr_dword);
        Pm4Packet pkt{};
        pkt.header = hdr;
        // A packet that runs past the buffer ends parsing; say where, since
        // everything after it (fences, EOP interrupts) is lost with it.
        auto report_truncated = [&] {
            std::fprintf(stderr, "[PM4 DCB] 0x%llx: malformed packet at +%zu (hdr=0x%08x count=%u, %zu dwords left):",
                         (unsigned long long)dcb_gpu_addr, (offset - 1) * 4, hdr_dword, (unsigned)hdr.count,
                         num_dwords - offset);
            const usize from = offset - 1 >= 16 ? offset - 1 - 16 : 0;
            for (usize i = from; i < offset - 1 + 8 && i < num_dwords; ++i)
                std::fprintf(stderr, "%s%08x", i == offset - 1 ? " | " : " ", dcb_dwords[i]);
            std::fprintf(stderr, "\n");
            usize changed = 0, first = 0;
            for (usize i = 0; i < num_dwords; ++i)
                if (live_dwords[i] != snapshot[i] && changed++ == 0) first = i;
            if (changed)
                PM4_LOG("[PM4 DCB]   %zu dwords changed in guest memory since submit (first at +%zu)\n",
                             changed, first * 4);
            // PS4_WATCH_ADDR on the buffer: show who wrote it, now, since the
            // guest usually stalls on the lost fence and never reaches the end-of-run dump.
            if (write_watch_addr() != 0) dump_write_watch_ring(0, m_impl->memory);
        };

        if (hdr.type == Pm4PacketType::Type0) {
            t0_count++;
            if (hdr.count > 0 && offset + hdr.count <= num_dwords) {
                pkt.payload.assign(dcb_dwords + offset, dcb_dwords + offset + hdr.count);
                offset += hdr.count;
            } else {
                report_truncated();
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
                report_truncated();
                break;
            }
        } else {
            break;
        }

        execute_packet(pkt);
        pkts_executed++;
    }
    m_impl->pending_skip_dwords = 0;

    std::string hist_str;
    for (const auto& [op, cnt] : op_counts) {
        char b[64];
        std::snprintf(b, sizeof(b), "0x%02x(%s):%zu ", op, pm4_opcode_to_name(op), cnt);
        hist_str += b;
    }
    PM4_LOG("[PM4 DCB] 0x%llx: pkts=%zu (T0:%zu T2:%zu T3:%zu) Ops: [%s]\n",
                 (unsigned long long)dcb_gpu_addr, pkts_executed, t0_count, t2_count, t3_count, hist_str.c_str());

    return pkts_executed;
}

usize Pm4Translator::execute_ring_buffer(GpuRingBuffer& ring_buffer) {
    auto packets = ring_buffer.parse_pending_packets();
    usize count = 0;
    if (m_impl) m_impl->pending_skip_dwords = 0;
    for (const auto& pkt : packets) {
        if (m_impl && m_impl->pending_skip_dwords > 0) {
            const usize dwords = 1 + pkt.payload.size();
            m_impl->pending_skip_dwords -= std::min(m_impl->pending_skip_dwords, dwords);
            continue;
        }
        if (execute_packet(pkt)) {
            count++;
        }
    }
    if (m_impl) m_impl->pending_skip_dwords = 0;
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

usize Pm4Translator::unsupported_packet_count() const noexcept {
    return m_impl ? m_impl->unsupported_packets : 0;
}

usize Pm4Translator::malformed_packet_count() const noexcept {
    return m_impl ? m_impl->malformed_packets : 0;
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

PrimitiveType Pm4Translator::primitive_type() const noexcept {
    return m_impl ? m_impl->primitive_type : PrimitiveType::TriangleList;
}

u32 Pm4Translator::num_instances() const noexcept {
    return m_impl ? m_impl->num_instances : 1;
}

u32 Pm4Translator::register_value(u32 abs_dword_reg) const noexcept {
    return (m_impl && abs_dword_reg < m_impl->reg_file.size()) ? m_impl->reg_file[abs_dword_reg] : 0;
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
