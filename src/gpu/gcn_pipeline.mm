#import <Metal/Metal.h>

#include "ps4/gcn_pipeline.hpp"
#include "ps4/gcn_shader.hpp"
#include "ps4/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace ps4 {

namespace {

constexpr u32 kSpiPsInputCntl0 = 0x0191;
constexpr u32 kCbTargetMask = 0x008E;
constexpr u32 kCbBlend0Control = 0x01E0;

MTLBlendFactor blend_factor(u32 f) {
    switch (f) {
    case 0: return MTLBlendFactorZero;
    case 1: return MTLBlendFactorOne;
    case 2: return MTLBlendFactorSourceColor;
    case 3: return MTLBlendFactorOneMinusSourceColor;
    case 4: return MTLBlendFactorSourceAlpha;
    case 5: return MTLBlendFactorOneMinusSourceAlpha;
    case 6: return MTLBlendFactorDestinationAlpha;
    case 7: return MTLBlendFactorOneMinusDestinationAlpha;
    case 8: return MTLBlendFactorDestinationColor;
    case 9: return MTLBlendFactorOneMinusDestinationColor;
    case 10: return MTLBlendFactorSourceAlphaSaturated;
    case 13: return MTLBlendFactorBlendColor;
    case 14: return MTLBlendFactorOneMinusBlendColor;
    case 15: return MTLBlendFactorSource1Color;
    case 16: return MTLBlendFactorOneMinusSource1Color;
    case 17: return MTLBlendFactorSource1Alpha;
    case 18: return MTLBlendFactorOneMinusSource1Alpha;
    case 19: return MTLBlendFactorBlendAlpha;
    case 20: return MTLBlendFactorOneMinusBlendAlpha;
    default: return MTLBlendFactorOne;
    }
}

MTLBlendOperation blend_op(u32 f) {
    switch (f) {
    case 1: return MTLBlendOperationSubtract;
    case 2: return MTLBlendOperationMin;
    case 3: return MTLBlendOperationMax;
    case 4: return MTLBlendOperationReverseSubtract;
    default: return MTLBlendOperationAdd;
    }
}

MTLSamplerAddressMode address_mode(u32 c) {
    switch (c) {
    case 0: return MTLSamplerAddressModeRepeat;
    case 1: return MTLSamplerAddressModeMirrorRepeat;
    case 2: return MTLSamplerAddressModeClampToEdge;
    case 3: case 5: case 7: return MTLSamplerAddressModeMirrorClampToEdge;
    default: return MTLSamplerAddressModeClampToZero;
    }
}

} // namespace

struct GcnPipelineCache::Impl {
    id<MTLDevice> device{nil};
    MemoryManager* memory{nullptr};
    GcnPipelineStats stats{};

    struct PipeKey {
        u64 vs_hash{0}, ps_hash{0}, cntl_hash{0};
        u32 remap{0}, blend{0}, mask{0}, format{0};
        bool operator<(const PipeKey& o) const noexcept {
            return std::tie(vs_hash, ps_hash, cntl_hash, remap, blend, mask, format) <
                   std::tie(o.vs_hash, o.ps_hash, o.cntl_hash, o.remap, o.blend, o.mask, o.format);
        }
    };
    struct PipeEntry {
        id<MTLRenderPipelineState> pso{nil};
        gcn::TranslatedPipeline tp;
        std::string error;
    };
    std::map<PipeKey, PipeEntry> pipes;

    struct TexEntry {
        id<MTLTexture> tex{nil};
        u64 content_hash{0};
        u64 last_used{0};
    };
    std::map<std::vector<u32>, TexEntry> textures;
    std::map<u64, id<MTLSamplerState>> samplers;
    id<MTLTexture> white{nil};
    std::set<std::string> logged;
    u64 clock{0};

    void log_once(const std::string& msg) {
        if (logged.insert(msg).second) std::fprintf(stderr, "[GCN] %s\n", msg.c_str());
    }

    gcn::MemReader reader() {
        return [this](u64 addr, void* dst, usize size) -> bool {
            return memory && memory->is_valid_address(addr, size) && memory->read_bytes(addr, dst, size);
        };
    }

    id<MTLTexture> white_texture() {
        if (white) return white;
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:1 height:1 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        white = [device newTextureWithDescriptor:d];
        const u32 px = 0xFFFFFFFFu;
        [white replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&px bytesPerRow:4];
        return white;
    }

    id<MTLSamplerState> sampler_for(const gcn::ImageDescriptors& img) {
        const gcn::SamplerDesc sd = gcn::decode_ssharp(img.ssharp);
        const u64 key = (static_cast<u64>(img.ssharp[0]) << 32) ^ img.ssharp[2];
        auto it = samplers.find(key);
        if (it != samplers.end()) return it->second;
        MTLSamplerDescriptor* d = [[MTLSamplerDescriptor alloc] init];
        d.sAddressMode = address_mode(sd.clamp_x);
        d.tAddressMode = address_mode(sd.clamp_y);
        d.rAddressMode = address_mode(sd.clamp_z);
        d.magFilter = (sd.xy_mag_filter & 1) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        d.minFilter = (sd.xy_min_filter & 1) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        d.mipFilter = sd.mip_filter == 0 ? MTLSamplerMipFilterNotMipmapped
                    : sd.mip_filter == 1 ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
        id<MTLSamplerState> s = [device newSamplerStateWithDescriptor:d];
        samplers[key] = s;
        return s;
    }

    id<MTLTexture> texture_for(const gcn::ImageDescriptors& img) {
        const gcn::TextureDesc td = gcn::decode_tsharp(img.tsharp);
        if (td.type != 9 && td.type != 8) {
            log_once("unsupported image type " + std::to_string(td.type) + " (only 1D/2D); using white");
            return white_texture();
        }
        const usize size = gcn::texture_guest_size(td);
        if (size == 0 || size > 64u * 1024 * 1024 || !memory || !memory->is_valid_address(td.base, size)) {
            log_once("texture memory unavailable/unsupported (dfmt " + std::to_string(td.dfmt) + "); using white");
            return white_texture();
        }
        const u8* src = memory->to_host_ptr<u8>(td.base);
        const u64 h = gcn::hash_words(reinterpret_cast<const u32*>(src), size / 4);
        std::vector<u32> key(img.tsharp, img.tsharp + 8);
        auto it = textures.find(key);
        if (it != textures.end() && it->second.content_hash == h) {
            it->second.last_used = ++clock;
            return it->second.tex;
        }
        std::vector<u8> rgba;
        std::string err;
        if (!gcn::texture_to_rgba8(td, src, size, rgba, err)) {
            log_once("texture upload: " + err + "; using white");
            return white_texture();
        }
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:td.width
                                                                                    height:td.height
                                                                                 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> tex = [device newTextureWithDescriptor:d];
        if (!tex) return white_texture();
        [tex replaceRegion:MTLRegionMake2D(0, 0, td.width, td.height) mipmapLevel:0 withBytes:rgba.data()
               bytesPerRow:td.width * 4];
        if (textures.size() > 256) textures.clear();
        textures[key] = TexEntry{tex, h, ++clock};
        ++stats.textures_uploaded;
        return tex;
    }

    void dump(const PipeKey& key, const gcn::Program& vs, const gcn::Program& ps, const gcn::TranslatedPipeline& tp) {
        const char* dir = std::getenv("PS4_GCN_DUMP_DIR");
        if (!dir || !*dir) return;
        char name[96];
        std::snprintf(name, sizeof(name), "%s/gcn_%016llx_%016llx", dir, (unsigned long long)key.vs_hash,
                      (unsigned long long)key.ps_hash);
        const std::string base = name;
        auto write_bin = [](const std::string& path, const std::vector<u32>& w) {
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(w.data()), static_cast<std::streamsize>(w.size() * 4));
        };
        write_bin(base + "_vs.bin", vs.code);
        if (!vs.fetch_code.empty()) write_bin(base + "_fetch.bin", vs.fetch_code);
        write_bin(base + "_ps.bin", ps.code);
        std::ofstream t(base + ".txt");
        t << "; VS\n";
        for (const auto& i : vs.insts) t << "  " << gcn::disassemble(i) << "\n";
        t << "; PS\n";
        for (const auto& i : ps.insts) t << "  " << gcn::disassemble(i) << "\n";
        if (!tp.ok) t << "; translation error: " << tp.error << "\n";
        if (tp.ok) {
            std::ofstream m(base + ".metal");
            m << tp.msl;
        }
    }

    PipeEntry& pipeline(const PipeKey& key, const gcn::Program& vs, const gcn::Program& ps, const u32* cntl,
                        gcn::TranslateOptions opts, u32 blend, u32 mask, u32 format) {
        auto it = pipes.find(key);
        if (it != pipes.end()) return it->second;
        PipeEntry e;
        const auto t_compile = std::chrono::steady_clock::now();
        e.tp = gcn::translate_pipeline(vs, ps, cntl, opts);
        dump(key, vs, ps, e.tp);
        if (!e.tp.ok) {
            e.error = "translate: " + e.tp.error;
        } else @autoreleasepool {
            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:e.tp.msl.c_str()];
            MTLCompileOptions* co = [[MTLCompileOptions alloc] init];
            id<MTLLibrary> lib = [device newLibraryWithSource:src options:co error:&err];
            if (!lib) {
                e.error = std::string("MSL compile: ") + (err ? [[err localizedDescription] UTF8String] : "?");
            } else {
                MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
                d.vertexFunction = [lib newFunctionWithName:@"gcn_vs_main"];
                d.fragmentFunction = [lib newFunctionWithName:@"gcn_ps_main"];
                d.colorAttachments[0].pixelFormat = static_cast<MTLPixelFormat>(format);
                MTLColorWriteMask wm = MTLColorWriteMaskNone;
                if (e.tp.ps_writes_color) {
                    if (mask & 1) wm |= MTLColorWriteMaskRed;
                    if (mask & 2) wm |= MTLColorWriteMaskGreen;
                    if (mask & 4) wm |= MTLColorWriteMaskBlue;
                    if (mask & 8) wm |= MTLColorWriteMaskAlpha;
                }
                d.colorAttachments[0].writeMask = wm;
                if (blend & (1u << 30)) {
                    const bool separate = (blend >> 29) & 1;
                    auto* ca = d.colorAttachments[0];
                    ca.blendingEnabled = YES;
                    ca.sourceRGBBlendFactor = blend_factor(blend & 0x1F);
                    ca.rgbBlendOperation = blend_op((blend >> 5) & 7);
                    ca.destinationRGBBlendFactor = blend_factor((blend >> 8) & 0x1F);
                    ca.sourceAlphaBlendFactor = blend_factor(separate ? ((blend >> 16) & 0x1F) : (blend & 0x1F));
                    ca.alphaBlendOperation = blend_op(separate ? ((blend >> 21) & 7) : ((blend >> 5) & 7));
                    ca.destinationAlphaBlendFactor =
                        blend_factor(separate ? ((blend >> 24) & 0x1F) : ((blend >> 8) & 0x1F));
                }
                e.pso = [device newRenderPipelineStateWithDescriptor:d error:&err];
                if (!e.pso)
                    e.error = std::string("pipeline: ") + (err ? [[err localizedDescription] UTF8String] : "?");
            }
        }
        char hdr[128];
        std::snprintf(hdr, sizeof(hdr), "pipeline VS#%016llx PS#%016llx", (unsigned long long)key.vs_hash,
                      (unsigned long long)key.ps_hash);
        if (e.pso) {
            ++stats.pipelines_compiled;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_compile).count();
            std::fprintf(stderr, "[GCN] compiled %s in %.1f ms (vs %zu insts, ps %zu insts, %u params, %zu textures)\n", hdr,
                         ms, vs.insts.size(), ps.insts.size(), e.tp.vs_param_count, e.tp.ps_images.size());
        } else {
            ++stats.pipeline_failures;
            std::fprintf(stderr, "[GCN] %s unsupported -> fixed fallback shader: %s\n", hdr, e.error.c_str());
        }
        return pipes.emplace(key, std::move(e)).first->second;
    }
};

GcnPipelineCache::GcnPipelineCache(void* mtl_device, MemoryManager* memory) : m_impl(std::make_unique<Impl>()) {
    m_impl->device = (__bridge id<MTLDevice>)mtl_device;
    m_impl->memory = memory;
}

GcnPipelineCache::~GcnPipelineCache() = default;

void GcnPipelineCache::set_memory(MemoryManager* memory) noexcept { m_impl->memory = memory; }

const GcnPipelineStats& GcnPipelineCache::stats() const noexcept { return m_impl->stats; }

bool GcnPipelineCache::enabled_by_env() noexcept {
    const char* e = std::getenv("PS4_GCN_SHADERS");
    return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
}

bool GcnPipelineCache::encode_draw(void* render_encoder, u32 color_format, const GcnDrawRequest& req,
                                   std::string* why) {
    Impl& im = *m_impl;
    auto fail = [&](const std::string& msg) {
        if (why) *why = msg;
        ++im.stats.fallback_draws;
        return false;
    };
    if (!im.device || !im.memory || !render_encoder || !req.vs_user || !req.ps_user) return fail("not configured");
    if (req.vs_addr < 0x1000 || req.ps_addr < 0x1000) return fail("no shaders bound");

    const gcn::MemReader read = im.reader();
    const u32 vs_user_count = (req.vs_rsrc2 >> 1) & 0x1F;
    const u32 ps_user_count = (req.ps_rsrc2 >> 1) & 0x1F;

    gcn::Program vs, ps;
    std::string err;
    if (!gcn::load_program(read, req.vs_addr, req.vs_user, std::min<u32>(vs_user_count, 16), vs, err)) {
        im.log_once("VS 0x" + std::to_string(req.vs_addr) + " load failed: " + err);
        return fail("VS load: " + err);
    }
    if (!gcn::load_program(read, req.ps_addr, req.ps_user, std::min<u32>(ps_user_count, 16), ps, err)) {
        im.log_once("PS 0x" + std::to_string(req.ps_addr) + " load failed: " + err);
        return fail("PS load: " + err);
    }

    // Primitive handling.
    using VR = gcn::TranslateOptions::VertexRemap;
    gcn::TranslateOptions opts;
    MTLPrimitiveType mtl_prim = MTLPrimitiveTypeTriangle;
    PrimitiveType prim = req.prim;
    // Indexed quad lists are lowered on the CPU by expanding the index
    // buffer (0,1,2,3 -> 0,1,2, 0,2,3); non-indexed rect/quad lists are
    // expanded inside the translated VS.
    const bool indexed_quads = req.indexed && prim == PrimitiveType::QuadList;
    if (indexed_quads) prim = PrimitiveType::TriangleList;
    switch (prim) {
    case PrimitiveType::None:
    case PrimitiveType::TriangleList: mtl_prim = MTLPrimitiveTypeTriangle; break;
    case PrimitiveType::TriangleStrip: mtl_prim = MTLPrimitiveTypeTriangleStrip; break;
    case PrimitiveType::LineList: mtl_prim = MTLPrimitiveTypeLine; break;
    case PrimitiveType::LineStrip: mtl_prim = MTLPrimitiveTypeLineStrip; break;
    case PrimitiveType::PointList: mtl_prim = MTLPrimitiveTypePoint; break;
    case PrimitiveType::RectList: opts.remap = VR::RectList; break;
    case PrimitiveType::QuadList: opts.remap = VR::QuadList; break;
    default:
        im.log_once("unsupported primitive type " + std::to_string(static_cast<u32>(prim)));
        return fail("primitive type");
    }
    if (opts.remap != VR::None && req.indexed) {
        im.log_once("indexed rect/quad lists unsupported");
        return fail("indexed rect/quad list");
    }

    // Context state.
    u32 cntl[32];
    for (u32 k = 0; k < 32; ++k) cntl[k] = k;
    u32 blend = 0;
    u32 mask = 0xF;
    const bool have_ctx = req.ctx_regs && req.ctx_reg_count > kSpiPsInputCntl0 + 32;
    if (have_ctx) {
        for (u32 k = 0; k < 32; ++k) cntl[k] = req.ctx_regs[kSpiPsInputCntl0 + k];
        blend = req.ctx_regs[kCbBlend0Control];
        mask = req.ctx_regs[kCbTargetMask] & 0xF;
        if (mask == 0) mask = 0xF;   // shadow not populated yet
    } else {
        blend = (1u << 30) | (5u << 8) | 4u;   // fixed path behaviour: src-alpha blending
    }

    Impl::PipeKey key;
    key.vs_hash = vs.hash;
    key.ps_hash = ps.hash;
    key.cntl_hash = gcn::hash_words(cntl, 32);
    key.remap = static_cast<u32>(opts.remap);
    key.blend = blend;
    key.mask = mask;
    key.format = color_format;
    Impl::PipeEntry& pe = im.pipeline(key, vs, ps, cntl, opts, blend, mask, color_format);
    if (!pe.pso) return fail(pe.error);

    // Resources.
    const u32 max_index = req.indexed ? 0xFFFFFu : req.count + (req.base_vertex > 0 ? req.base_vertex : 0);
    gcn::ResourceScan vs_scan = gcn::scan_resources(vs, pe.tp.vs_images, req.vs_user, std::min<u32>(vs_user_count, 16),
                                                    read, max_index);
    gcn::ResourceScan ps_scan = gcn::scan_resources(ps, pe.tp.ps_images, req.ps_user, std::min<u32>(ps_user_count, 16),
                                                    read, 0);
    if (!vs_scan.ok || !ps_scan.ok) {
        const std::string e = !vs_scan.ok ? "VS " + vs_scan.error : "PS " + ps_scan.error;
        im.log_once("resource scan: " + e);
        return fail(e);
    }

    std::vector<gcn::MemRange> ranges = vs_scan.ranges;
    ranges.insert(ranges.end(), ps_scan.ranges.begin(), ps_scan.ranges.end());
    if (ranges.size() > gcn::binding::kMaxRanges) {
        im.log_once("too many memory ranges in one draw");
        return fail("too many ranges");
    }

    gcn::DrawConsts dc{};
    std::vector<u8> arena;
    for (usize k = 0; k < ranges.size(); ++k) {
        gcn::MemRange r = ranges[k];
        // Clamp to mapped guest memory.
        while (r.size > 0 && !im.memory->is_valid_address(r.addr, r.size)) r.size /= 2;
        const usize off = (arena.size() + 15) & ~usize(15);
        arena.resize(off + r.size);
        if (r.size) im.memory->read_bytes(r.addr, arena.data() + off, r.size);
        dc.range_base[k] = r.addr;
        dc.range_size[k] = r.size;
        dc.range_off[k] = static_cast<u32>(off);
    }
    dc.range_count = static_cast<u32>(ranges.size());
    if (std::getenv("PS4_GCN_TRACE")) {
        for (u32 k = 0; k < dc.range_count; ++k) {
            std::fprintf(stderr, "[GCN trace] range %u: 0x%llx +%u:", k, (unsigned long long)dc.range_base[k],
                         dc.range_size[k]);
            for (u32 o = 0; o + 4 <= std::min<u32>(dc.range_size[k], 600); o += 4) {
                float f;
                std::memcpy(&f, arena.data() + dc.range_off[k] + o, 4);
                u32 x;
                std::memcpy(&x, &f, 4);
                if (dc.range_size[k] <= 64) std::fprintf(stderr, " %08x", x);
                else std::fprintf(stderr, " %g", f);
            }
            std::fprintf(stderr, "\n");
        }
    }
    dc.base_vertex = req.base_vertex > 0 ? static_cast<u32>(req.base_vertex) : 0;
    std::memcpy(dc.vs_user, req.vs_user, sizeof(dc.vs_user));
    std::memcpy(dc.ps_user, req.ps_user, sizeof(dc.ps_user));
    if (arena.empty()) arena.resize(16);

    @autoreleasepool {
        id<MTLRenderCommandEncoder> enc = (__bridge id<MTLRenderCommandEncoder>)render_encoder;
        id<MTLBuffer> mem = [im.device newBufferWithBytes:arena.data() length:arena.size()
                                                  options:MTLResourceStorageModeShared];
        if (!mem) return fail("arena allocation");

        id<MTLBuffer> idx_buf = nil;
        u32 index_count = req.count;
        if (req.indexed) {
            const usize isz = static_cast<usize>(req.count) * (req.index32 ? 4 : 2);
            if (!im.memory->is_valid_address(req.index_addr, isz)) return fail("index buffer out of range");
            const u8* src = im.memory->to_host_ptr<u8>(req.index_addr);
            if (indexed_quads) {
                std::vector<u32> exp;
                for (u32 q = 0; q + 4 <= req.count; q += 4) {
                    u32 v[4];
                    for (u32 k = 0; k < 4; ++k) {
                        if (req.index32) std::memcpy(&v[k], src + (q + k) * 4, 4);
                        else { u16 h; std::memcpy(&h, src + (q + k) * 2, 2); v[k] = h; }
                    }
                    const u32 order[6] = {0, 1, 2, 0, 2, 3};
                    for (u32 o : order) exp.push_back(v[o]);
                }
                if (exp.empty()) return fail("empty quad list");
                index_count = static_cast<u32>(exp.size());
                idx_buf = [im.device newBufferWithBytes:exp.data() length:exp.size() * 4
                                                options:MTLResourceStorageModeShared];
            } else {
                idx_buf = [im.device newBufferWithBytes:src length:isz options:MTLResourceStorageModeShared];
            }
            if (!idx_buf) return fail("index allocation");
        }

        [enc setRenderPipelineState:pe.pso];
        [enc setCullMode:MTLCullModeNone];
        [enc setVertexBuffer:mem offset:0 atIndex:gcn::binding::kMemBuffer];
        [enc setVertexBytes:&dc length:sizeof(dc) atIndex:gcn::binding::kDrawConsts];
        [enc setFragmentBuffer:mem offset:0 atIndex:gcn::binding::kMemBuffer];
        [enc setFragmentBytes:&dc length:sizeof(dc) atIndex:gcn::binding::kDrawConsts];
        for (usize k = 0; k < pe.tp.vs_images.size(); ++k) {
            [enc setVertexTexture:im.texture_for(vs_scan.images[k]) atIndex:k];
            if (pe.tp.vs_images[k].sampled) [enc setVertexSamplerState:im.sampler_for(vs_scan.images[k]) atIndex:k];
        }
        for (usize k = 0; k < pe.tp.ps_images.size(); ++k) {
            [enc setFragmentTexture:im.texture_for(ps_scan.images[k]) atIndex:k];
            if (pe.tp.ps_images[k].sampled) [enc setFragmentSamplerState:im.sampler_for(ps_scan.images[k]) atIndex:k];
        }

        const u32 instances = std::max<u32>(1, req.instances);
        if (req.indexed) {
            [enc drawIndexedPrimitives:mtl_prim
                            indexCount:index_count
                             indexType:(req.index32 || indexed_quads) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
                           indexBuffer:idx_buf
                     indexBufferOffset:0
                         instanceCount:instances
                            baseVertex:req.base_vertex
                          baseInstance:0];
        } else if (opts.remap == VR::RectList) {
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:(req.count / 3) * 6
                  instanceCount:instances];
        } else if (opts.remap == VR::QuadList) {
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:(req.count / 4) * 6
                  instanceCount:instances];
        } else {
            [enc drawPrimitives:mtl_prim vertexStart:dc.base_vertex vertexCount:req.count instanceCount:instances];
        }
    }
    ++im.stats.translated_draws;
    return true;
}

} // namespace ps4
