#include "ps4/debug/capture.hpp"
#import <Metal/Metal.h>

#include "ps4/gcn_pipeline.hpp"
#include "ps4/gcn_shader.hpp"
#include "ps4/gpu_state.hpp"
#include "ps4/memory.hpp"
#include "ps4/metal_backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
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

// GCN and Metal number their compare functions identically (NEVER..ALWAYS).
MTLCompareFunction compare_fn(CompareFunc f) { return static_cast<MTLCompareFunction>(f); }

MTLStencilOperation stencil_op(StencilOp op) {
    switch (op) {
    case StencilOp::Keep: return MTLStencilOperationKeep;
    case StencilOp::Zero: return MTLStencilOperationZero;
    // ONES and REPLACE_OP have no Metal equivalent; replacing with the
    // reference value is the closest single operation.
    case StencilOp::Ones:
    case StencilOp::ReplaceTest:
    case StencilOp::ReplaceOp: return MTLStencilOperationReplace;
    case StencilOp::AddClamp: return MTLStencilOperationIncrementClamp;
    case StencilOp::SubClamp: return MTLStencilOperationDecrementClamp;
    case StencilOp::Invert: return MTLStencilOperationInvert;
    case StencilOp::AddWrap: return MTLStencilOperationIncrementWrap;
    case StencilOp::SubWrap: return MTLStencilOperationDecrementWrap;
    default: return MTLStencilOperationKeep;   // bitwise ops
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
    // All of guest memory as one no-copy buffer, for compute shaders (they
    // read and write it in place). Made once: wrapping the 8 GB range again
    // for every dispatch made each command buffer wait for that range to be
    // made resident, so a game dispatching every frame (Sonic Mania) ran at
    // 4-17 fps with the GPU nearly idle.
    id<MTLBuffer> guest_buffer{nil};
    const void* guest_buffer_base{nullptr};
    id<MTLBuffer> guest_memory_buffer() {
        if (!memory || !memory->base_address()) return nil;
        if (guest_buffer && guest_buffer_base == memory->base_address()) return guest_buffer;
        const NSUInteger len = std::min<NSUInteger>(memory->total_size(), [device maxBufferLength]) & ~16383UL;
        guest_buffer = [device newBufferWithBytesNoCopy:memory->base_address()
                                                 length:len
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
        guest_buffer_base = guest_buffer ? memory->base_address() : nullptr;
        return guest_buffer;
    }
    MetalBackend* metal{nullptr};
    GcnPipelineStats stats{};

    struct PipeKey {
        u64 vs_hash{0}, ps_hash{0}, cntl_hash{0};
        u32 remap{0}, blend{0}, mask{0}, format{0}, depth_format{0}, ps_inputs{0};
        u64 mrt{0}; // formats and blend controls of attachments 1-7
        bool operator<(const PipeKey& o) const noexcept {
            return std::tie(vs_hash, ps_hash, cntl_hash, remap, blend, mask, format, depth_format, ps_inputs, mrt) <
                   std::tie(o.vs_hash, o.ps_hash, o.cntl_hash, o.remap, o.blend, o.mask, o.format, o.depth_format,
                            o.ps_inputs, o.mrt);
        }
    };
    struct PipeEntry {
        id<MTLRenderPipelineState> pso{nil};
        gcn::TranslatedPipeline tp;
        std::string error;
    };
    std::map<PipeKey, PipeEntry> pipes;

    struct ComputeKey {
        u64 cs_hash{0};
        u32 threads_x{0};
        u32 threads_y{0};
        u32 threads_z{0};
        u32 rsrc1{0};
        u32 rsrc2{0};
        bool operator<(const ComputeKey& o) const noexcept {
            return std::tie(cs_hash, threads_x, threads_y, threads_z, rsrc1, rsrc2) <
                   std::tie(o.cs_hash, o.threads_x, o.threads_y, o.threads_z, o.rsrc1, o.rsrc2);
        }
    };
    struct ComputeEntry {
        id<MTLComputePipelineState> pso{nil};
        gcn::TranslatedComputePipeline tp;
        std::string error;
    };
    std::map<ComputeKey, ComputeEntry> compute_pipes;

    struct TexEntry {
        id<MTLTexture> tex{nil};
        u64 content_hash{0};
        u64 last_used{0};
    };
    std::map<std::vector<u32>, TexEntry> textures;
    std::map<u64, id<MTLSamplerState>> samplers;
    std::map<std::array<u32, 3>, id<MTLDepthStencilState>> depth_states;

    id<MTLDepthStencilState> depth_stencil(const DepthStencilState& ds) {
        auto face_bits = [](const StencilFace& f) {
            return static_cast<u32>(f.func) | (static_cast<u32>(f.fail) << 3) | (static_cast<u32>(f.zpass) << 7) |
                   (static_cast<u32>(f.zfail) << 11) | (static_cast<u32>(f.read_mask) << 15) |
                   (static_cast<u32>(f.write_mask) << 23);
        };
        const CompareFunc zf = ds.depth_test ? ds.depth_func : CompareFunc::Always;
        const std::array<u32, 3> key{
            static_cast<u32>(ds.depth_test && ds.depth_write) | (static_cast<u32>(zf) << 1) |
                (static_cast<u32>(ds.stencil_test) << 4),
            ds.stencil_test ? face_bits(ds.front) : 0u, ds.stencil_test ? face_bits(ds.back) : 0u};
        auto it = depth_states.find(key);
        if (it != depth_states.end()) return it->second;
        MTLDepthStencilDescriptor* d = [[MTLDepthStencilDescriptor alloc] init];
        d.depthCompareFunction = compare_fn(zf);
        d.depthWriteEnabled = ds.depth_test && ds.depth_write;
        if (ds.stencil_test) {
            auto face = [](const StencilFace& f) {
                MTLStencilDescriptor* sd = [[MTLStencilDescriptor alloc] init];
                sd.stencilCompareFunction = compare_fn(f.func);
                sd.stencilFailureOperation = stencil_op(f.fail);
                sd.depthFailureOperation = stencil_op(f.zfail);
                sd.depthStencilPassOperation = stencil_op(f.zpass);
                sd.readMask = f.read_mask;
                sd.writeMask = f.write_mask;
                return sd;
            };
            d.frontFaceStencil = face(ds.front);
            d.backFaceStencil = face(ds.back);
        }
        id<MTLDepthStencilState> st = [device newDepthStencilStateWithDescriptor:d];
        depth_states[key] = st;
        return st;
    }
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

    // Content hash for the texture cache: four independent 64-bit lanes so
    // a multi-megabyte texture (a movie frame) hashes at memory speed rather
    // than one serial multiply per word.
    static u64 hash_texture(const u8* p, usize n) {
        u64 h[4] = {0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0xd6e8feb86659fd93ull};
        usize i = 0;
        for (; i + 32 <= n; i += 32) {
            for (usize k = 0; k < 4; ++k) {
                u64 w;
                std::memcpy(&w, p + i + 8 * k, 8);
                h[k] = (h[k] ^ w) * 0x100000001b3ull;
                h[k] ^= h[k] >> 29;
            }
        }
        u64 r = h[0] ^ (h[1] * 3) ^ (h[2] * 5) ^ (h[3] * 7) ^ n;
        for (; i < n; ++i) r = (r ^ p[i]) * 0x100000001b3ull;
        return r;
    }

    id<MTLTexture> direct_texture(const gcn::TextureDesc& td, const u8* src) {
        const bool linear = td.tiling_index == 8 || td.tiling_index == 31;
        if (!linear || td.nfmt != 0 || (td.dfmt != 10 && td.dfmt != 16) || td.type != 9) return nil;
        auto swz = [](u32 sel, MTLTextureSwizzle* out) {
            switch (sel) {
            case 0: *out = MTLTextureSwizzleZero; return true;
            case 1: *out = MTLTextureSwizzleOne; return true;
            case 4: *out = MTLTextureSwizzleRed; return true;
            case 5: *out = MTLTextureSwizzleGreen; return true;
            case 6: *out = MTLTextureSwizzleBlue; return true;
            case 7: *out = MTLTextureSwizzleAlpha; return true;
            default: return false;
            }
        };
        // GCN 5_6_5 keeps channel 0 in bits 4:0, which Metal's B5G6R5Unorm
        // reads as blue (red is bits 15:11), so channels 0 and 2 trade
        // places; 8_8_8_8 keeps channel k in byte k (RGBA8Unorm).
        const bool rgb565 = td.dfmt == 16;
        MTLTextureSwizzle c[4];
        for (u32 k = 0; k < 4; ++k) {
            u32 sel = td.dst_sel[k];
            if (rgb565 && (sel == 4 || sel == 6)) sel = sel == 4 ? 6 : 4;
            if (!swz(sel, &c[k])) return nil;
        }
        MTLTextureDescriptor* d = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:rgb565 ? MTLPixelFormatB5G6R5Unorm : MTLPixelFormatRGBA8Unorm
                                         width:td.width
                                        height:td.height
                                     mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        d.swizzle = MTLTextureSwizzleChannelsMake(c[0], c[1], c[2], c[3]);
        id<MTLTexture> tex = [device newTextureWithDescriptor:d];
        if (!tex) return nil;
        const u32 bpp = rgb565 ? 2 : 4;
        [tex replaceRegion:MTLRegionMake2D(0, 0, td.width, td.height) mipmapLevel:0 withBytes:src
               bytesPerRow:std::max(td.pitch, td.width) * bpp];
        return tex;
    }

    id<MTLTexture> texture_for(const gcn::ImageDescriptors& img) {
        const gcn::TextureDesc td = gcn::decode_tsharp(img.tsharp);
        if (td.type != 9 && td.type != 8) {
            log_once("unsupported image type " + std::to_string(td.type) + " (only 1D/2D); using white");
            return white_texture();
        }
        // A texture at the address of a render target (offscreen, or a display
        // buffer: the screen) samples what was rendered there; the guest copy
        // is never written by the GPU path.
        if (metal && td.base > 0x1000) {
            id<MTLTexture> offscreen = (__bridge id<MTLTexture>)metal->surface_texture(td.base);
            if (offscreen) {
                if (offscreen.width != td.width || offscreen.height != td.height)
                    log_once("texture " + std::to_string(td.width) + "x" + std::to_string(td.height) +
                             " samples a " + std::to_string(offscreen.width) + "x" +
                             std::to_string(offscreen.height) + " render target");
                return offscreen;
            }
        }
        const usize size = gcn::texture_guest_size(td);
        if (size == 0 || size > 64u * 1024 * 1024 || !memory || !memory->is_valid_address(td.base, size)) {
            log_once("texture memory unavailable/unsupported (dfmt " + std::to_string(td.dfmt) + "); using white");
            return white_texture();
        }
        const u8* src = memory->to_host_ptr<u8>(td.base);
        const u64 h = gcn::hash_memory(src, size);
        std::vector<u32> key(img.tsharp, img.tsharp + 8);
        auto it = textures.find(key);
        if (it != textures.end() && it->second.content_hash == h) {
            it->second.last_used = ++clock;
            return it->second.tex;
        }
        // Linear R8 (video planes change every frame) needs no conversion: the
        // guest bytes upload as they are and the T#'s selector becomes a swizzle.
        if (td.dfmt == 1 && td.nfmt == 0 && gcn::is_linear_tiling(td.tiling_index)) {
            auto channel = [](u32 sel) {
                switch (sel) {
                case 1: return MTLTextureSwizzleOne;
                case 4: return MTLTextureSwizzleRed;
                case 5: return MTLTextureSwizzleGreen;
                case 6: return MTLTextureSwizzleBlue;
                case 7: return MTLTextureSwizzleAlpha;
                default: return MTLTextureSwizzleZero;
                }
            };
            MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                         width:td.width
                                                                                        height:td.height
                                                                                     mipmapped:NO];
            d.usage = MTLTextureUsageShaderRead;
            d.allowGPUOptimizedContents = NO;
            d.swizzle = MTLTextureSwizzleChannelsMake(channel(td.dst_sel[0]), channel(td.dst_sel[1]),
                                                      channel(td.dst_sel[2]), channel(td.dst_sel[3]));
            id<MTLTexture> tex = [device newTextureWithDescriptor:d];
            if (!tex) return white_texture();
            [tex replaceRegion:MTLRegionMake2D(0, 0, td.width, td.height) mipmapLevel:0 withBytes:src
                   bytesPerRow:std::max(td.pitch, td.width)];
            if (textures.size() > 256) textures.clear();
            textures[key] = TexEntry{tex, h, ++clock};
            ++stats.textures_uploaded;
            return tex;
        }
        // Linear 8_8_8_8 / 5_6_5 unorm (software-rendered screens, movie
        // frames): Metal reads the guest bytes as they are, with the T#'s
        // channel selects as the texture swizzle -- no CPU conversion.
        if (id<MTLTexture> direct = direct_texture(td, src)) {
            if (textures.size() > 256) textures.clear();
            textures[key] = TexEntry{direct, h, ++clock};
            ++stats.textures_uploaded;
            return direct;
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

    /// Writable host texture for an image_store T#: the surface the GPU path
    /// already keeps at that address (render target or screen), or a new one
    /// in the T#'s format, so later draws and samples see what compute wrote.
    id<MTLTexture> storage_texture_for(const gcn::ImageDescriptors& img) {
        const gcn::TextureDesc td = gcn::decode_tsharp(img.tsharp);
        if (!metal || td.base <= 0x1000 || td.width == 0 || td.height == 0) return nil;
        MTLPixelFormat fmt = MTLPixelFormatRGBA8Unorm;
        if (td.dfmt == 12 && td.nfmt == 7) fmt = MTLPixelFormatRGBA16Float;
        else if (td.dfmt == 14 && td.nfmt == 7) fmt = MTLPixelFormatRGBA32Float;
        else if (td.dfmt == 4 && td.nfmt == 7) fmt = MTLPixelFormatR32Float;
        else if (td.dfmt != 10) log_once("image_store format " + std::to_string(td.dfmt) + "/" +
                                         std::to_string(td.nfmt) + " stored as RGBA8Unorm");
        // A linear RGBA8 target lives only on the host once compute wrote it;
        // remember its guest layout so it can be written back if the cache
        // drops it (Dragon Ball Xenoverse 2 copies its 2x1 UI ramp texture
        // this way at boot and samples it a minute later, after the movie's
        // post-processing had pushed it out of the cache).
        if (fmt == MTLPixelFormatRGBA8Unorm && td.dfmt == 10 && td.nfmt == 0 && gcn::is_linear_tiling(td.tiling_index))
            storage_layouts[td.base] = StorageLayout{td.width, td.height, std::max(td.pitch, td.width)};
        return (__bridge id<MTLTexture>)metal->storage_target(td.base, td.width, td.height,
                                                              static_cast<u32>(fmt));
    }

    struct StorageLayout {
        u32 width, height, pitch;  // pitch in texels
    };
    std::map<u64, StorageLayout> storage_layouts;

    // MetalBackend's eviction hook: copy a dropped image_store target back,
    // once the GPU has run what was encoded into it. (The cache can fill up in
    // the same frame as the dispatch that wrote the target.)
    void write_back_surface(u64 key, void* texture) {
        auto it = storage_layouts.find(key);
        if (it == storage_layouts.end()) return;
        const StorageLayout l = it->second;
        storage_layouts.erase(it);
        id<MTLTexture> t = (__bridge id<MTLTexture>)texture;
        if (!memory || !metal || !t || t.pixelFormat != MTLPixelFormatRGBA8Unorm) return;
        const u32 w = std::min<u32>(l.width, static_cast<u32>(t.width));
        const u32 h = std::min<u32>(l.height, static_cast<u32>(t.height));
        if (!memory->is_valid_address(key, static_cast<usize>(l.pitch) * 4 * h)) return;
        MemoryManager* mem = memory;
        metal->run_after_gpu([t, mem, key, l, w, h] {
            std::vector<u8> row(static_cast<usize>(w) * 4);
            for (u32 y = 0; y < h; ++y) {
                [t getBytes:row.data() bytesPerRow:w * 4 fromRegion:MTLRegionMake2D(0, y, w, 1) mipmapLevel:0];
                mem->write_bytes(key + static_cast<u64>(y) * l.pitch * 4, row.data(), row.size());
            }
        });
    }

    void dump(const PipeKey& key, const gcn::Program& vs, const gcn::Program& ps, const gcn::TranslatedPipeline& tp) {
        const char* dir = std::getenv("PS4_GCN_DUMP_DIR");
        if (!dir || !*dir) return;
        // The directory is concatenated, not formatted into a fixed buffer:
        // a long path used to be truncated and every dump silently failed.
        char leaf[64];
        std::snprintf(leaf, sizeof(leaf), "/gcn_%016llx_%016llx", (unsigned long long)key.vs_hash,
                      (unsigned long long)key.ps_hash);
        const std::string base = dir + std::string(leaf);
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

    /// formats[n] / blends[n]: MTLPixelFormat (0 = not attached) and
    /// CB_BLEND<n>_CONTROL of colour attachment n; mask: CB_TARGET_MASK.
    PipeEntry& pipeline(const PipeKey& key, const gcn::Program& vs, const gcn::Program& ps, const u32* cntl,
                        gcn::TranslateOptions opts, const u32* blends, u32 mask, const u32* formats, u32 depth_format) {
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
                // A depth-only draw has no fragment function; every colour
                // attachment is then write-masked off (ps_color_mask is 0).
                d.fragmentFunction = e.tp.has_ps ? [lib newFunctionWithName:@"gcn_ps_main"] : nil;
                if (depth_format) {
                    const MTLPixelFormat df = static_cast<MTLPixelFormat>(depth_format);
                    d.depthAttachmentPixelFormat = df;
                    if (df == MTLPixelFormatDepth32Float_Stencil8 || df == MTLPixelFormatDepth24Unorm_Stencil8)
                        d.stencilAttachmentPixelFormat = df;
                }
                // Every attachment of the pass must be declared; one the PS
                // does not export (mrt<n>) is left unwritten.
                for (u32 n = 0; n < 8; ++n) {
                    if (!formats[n]) continue;
                    auto* ca = d.colorAttachments[n];
                    ca.pixelFormat = static_cast<MTLPixelFormat>(formats[n]);
                    const u32 m = (mask >> (4 * n)) & 0xF;
                    MTLColorWriteMask wm = MTLColorWriteMaskNone;
                    if ((e.tp.ps_color_mask >> n) & 1) {
                        if (m & 1) wm |= MTLColorWriteMaskRed;
                        if (m & 2) wm |= MTLColorWriteMaskGreen;
                        if (m & 4) wm |= MTLColorWriteMaskBlue;
                        if (m & 8) wm |= MTLColorWriteMaskAlpha;
                    }
                    ca.writeMask = wm;
                    const u32 blend = blends[n];
                    if (blend & (1u << 30)) {
                        const bool separate = (blend >> 29) & 1;
                        ca.blendingEnabled = YES;
                        ca.sourceRGBBlendFactor = blend_factor(blend & 0x1F);
                        ca.rgbBlendOperation = blend_op((blend >> 5) & 7);
                        ca.destinationRGBBlendFactor = blend_factor((blend >> 8) & 0x1F);
                        ca.sourceAlphaBlendFactor = blend_factor(separate ? ((blend >> 16) & 0x1F) : (blend & 0x1F));
                        ca.alphaBlendOperation = blend_op(separate ? ((blend >> 21) & 7) : ((blend >> 5) & 7));
                        ca.destinationAlphaBlendFactor =
                            blend_factor(separate ? ((blend >> 24) & 0x1F) : ((blend >> 8) & 0x1F));
                    }
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

GcnPipelineCache::~GcnPipelineCache() {
    if (m_impl && m_impl->metal) m_impl->metal->set_surface_evict_hook({});
}

void GcnPipelineCache::set_memory(MemoryManager* memory) noexcept { m_impl->memory = memory; }
void GcnPipelineCache::set_metal_backend(MetalBackend* metal) noexcept {
    if (m_impl->metal == metal) return;
    if (m_impl->metal) m_impl->metal->set_surface_evict_hook({});
    m_impl->metal = metal;
    if (metal) {
        Impl* im = m_impl.get();
        metal->set_surface_evict_hook([im](u64 key, void* texture) { im->write_back_surface(key, texture); });
    }
}

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
    if (req.vs_addr < 0x1000) return fail("no vertex shader bound");

    const gcn::MemReader read = im.reader();
    const u32 vs_user_count = (req.vs_rsrc2 >> 1) & 0x1F;
    const u32 ps_user_count = (req.ps_rsrc2 >> 1) & 0x1F;

    gcn::Program vs, ps;
    std::string err;
    if (!gcn::load_program(read, req.vs_addr, req.vs_user, std::min<u32>(vs_user_count, 16), vs, err)) {
        im.log_once("VS 0x" + std::to_string(req.vs_addr) + " load failed: " + err);
        return fail("VS load: " + err);
    }
    // No PS address (0): a depth-only draw; `ps` stays empty.
    if (req.ps_addr >= 0x1000 &&
        !gcn::load_program(read, req.ps_addr, req.ps_user, std::min<u32>(ps_user_count, 16), ps, err)) {
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
    u32 blends[8] = {};
    u32 formats[8] = {color_format};
    for (u32 n = 1; n < 8; ++n) formats[n] = req.color_formats[n];
    const bool have_ctx = req.ctx_regs && req.ctx_reg_count > kSpiPsInputCntl0 + 32;
    if (have_ctx) {
        for (u32 k = 0; k < 32; ++k) cntl[k] = req.ctx_regs[kSpiPsInputCntl0 + k];
        if (req.ctx_reg_count > ctx_reg::kSpiPsInputAddr) {
            opts.ps_input_ena = req.ctx_regs[ctx_reg::kSpiPsInputEna] & 0xFFFF;
            opts.ps_input_addr = req.ctx_regs[ctx_reg::kSpiPsInputAddr] & 0xFFFF;
        }
        blend = req.ctx_regs[kCbBlend0Control];
        for (u32 n = 1; n < 8; ++n) blends[n] = req.ctx_regs[kCbBlend0Control + n];
        mask = req.ctx_regs[kCbTargetMask]; // 0: no colour writes (depth-only pass)
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
    key.depth_format = req.depth_format;
    key.ps_inputs = opts.ps_input_ena | (opts.ps_input_addr << 16);
    blends[0] = blend;
    {
        u32 mrt[14];
        for (u32 n = 1; n < 8; ++n) { mrt[n - 1] = formats[n]; mrt[6 + n] = blends[n]; }
        key.mrt = gcn::hash_words(mrt, 14);
    }

    // Fixed-function 3D state. Without a known render target size the
    // viewport and scissor are left as the caller set them.
    const bool have_fb = req.fb_width && req.fb_height;
    const GpuRenderState rs = decode_render_state(have_ctx ? req.ctx_regs : nullptr, have_ctx ? req.ctx_reg_count : 0,
                                                  have_fb ? req.fb_width : 16384, have_fb ? req.fb_height : 16384);
    if (rs.skip_draw) {
        ++im.stats.skipped_draws;
        return true;   // nothing would be rasterised
    }

    Impl::PipeEntry& pe = im.pipeline(key, vs, ps, cntl, opts, blends, mask, formats, req.depth_format);
    if (!pe.pso) return fail(pe.error);

    // Resources.
    const u32 max_index = req.indexed ? 0xFFFFFu : req.count + (req.base_vertex > 0 ? req.base_vertex : 0);
    gcn::ResourceScan vs_scan = gcn::scan_resources(vs, pe.tp.vs_images, req.vs_user, std::min<u32>(vs_user_count, 16),
                                                    read, max_index);
    gcn::ResourceScan ps_scan = gcn::scan_resources(ps, pe.tp.ps_images, req.ps_user, std::min<u32>(ps_user_count, 16),
                                                    read, 0);
    debug::capture_resources("VS", vs.hash, vs_scan, pe.tp.vs_images);
    debug::capture_resources("PS", ps.hash, ps_scan, pe.tp.ps_images);
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
    dc.flags = rs.viewport.flip_y ? gcn::kDrawFlipY : 0;
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
        [enc setCullMode:rs.raster.cull == CullMode::Front  ? MTLCullModeFront
                         : rs.raster.cull == CullMode::Back ? MTLCullModeBack
                                                            : MTLCullModeNone];
        [enc setFrontFacingWinding:rs.raster.front_ccw ? MTLWindingCounterClockwise : MTLWindingClockwise];
        if (req.depth_format) {
            [enc setDepthStencilState:im.depth_stencil(rs.ds)];
            [enc setStencilFrontReferenceValue:rs.ds.front.ref backReferenceValue:rs.ds.back.ref];
            [enc setDepthBias:rs.raster.depth_bias slopeScale:rs.raster.depth_bias_slope clamp:rs.raster.depth_bias_clamp];
            [enc setDepthClipMode:rs.raster.depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
        }
        if (have_fb) {
            const RenderViewport& v = rs.viewport;
            [enc setViewport:MTLViewport{v.x, v.y, v.width, v.height, v.znear, v.zfar}];
            const RenderScissor& sc = rs.scissor;
            [enc setScissorRect:MTLScissorRect{sc.x, sc.y, sc.width, sc.height}];
        }
        [enc setVertexBuffer:mem offset:0 atIndex:gcn::binding::kMemBuffer];
        [enc setVertexBytes:&dc length:sizeof(dc) atIndex:gcn::binding::kDrawConsts];
        [enc setFragmentBuffer:mem offset:0 atIndex:gcn::binding::kMemBuffer];
        [enc setFragmentBytes:&dc length:sizeof(dc) atIndex:gcn::binding::kDrawConsts];
        auto bind_uniform_loads = [](const std::vector<u32>& t, auto&& set) {
            static const u32 kEmpty = 0;
            set(t.empty() ? &kEmpty : t.data(), t.empty() ? sizeof(kEmpty) : t.size() * sizeof(u32));
        };
        bind_uniform_loads(vs_scan.uniform_loads, [&](const u32* p, usize n) {
            [enc setVertexBytes:p length:n atIndex:gcn::binding::kUniformLoads];
        });
        bind_uniform_loads(ps_scan.uniform_loads, [&](const u32* p, usize n) {
            [enc setFragmentBytes:p length:n atIndex:gcn::binding::kUniformLoads];
        });
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

bool GcnPipelineCache::encode_dispatch(void* compute_encoder, const GcnComputeRequest& req,
                                       std::string* why) {
    Impl& im = *m_impl;
    auto fail = [&](const std::string& msg) {
        if (why) *why = msg;
        return false;
    };
    if (!im.device || !compute_encoder) return fail("not configured");
    if (req.cs_addr < 0x1000) return fail("no compute shader bound");

    const gcn::MemReader read = im.reader();
    const u32 cs_user_count = (req.cs_rsrc2 >> 1) & 0x1F;

    gcn::Program cs;
    std::string err;
    if (!gcn::load_program(read, req.cs_addr, req.cs_user, std::min<u32>(cs_user_count, 16), cs, err)) {
        im.log_once("CS 0x" + std::to_string(req.cs_addr) + " load failed: " + err);
        return fail("CS load: " + err);
    }

    gcn::ComputePipelineOptions opts{};
    opts.threads_x = req.threads_x;
    opts.threads_y = req.threads_y;
    opts.threads_z = req.threads_z;
    opts.rsrc1 = req.cs_rsrc1;
    opts.rsrc2 = req.cs_rsrc2;

    Impl::ComputeKey key;
    key.cs_hash = cs.hash;
    key.threads_x = req.threads_x;
    key.threads_y = req.threads_y;
    key.threads_z = req.threads_z;
    key.rsrc1 = req.cs_rsrc1;
    key.rsrc2 = req.cs_rsrc2;

    auto it = im.compute_pipes.find(key);
    Impl::ComputeEntry* pe = nullptr;
    if (it != im.compute_pipes.end()) {
        pe = &it->second;
    } else {
        Impl::ComputeEntry e;
        const auto t_compile = std::chrono::steady_clock::now();
        e.tp = gcn::translate_compute_pipeline(cs, opts);
        if (const char* dir = std::getenv("PS4_GCN_DUMP_DIR"); dir && *dir) {
            char leaf[40];
            std::snprintf(leaf, sizeof(leaf), "/gcn_cs_%016llx", (unsigned long long)cs.hash);
            const std::string name = dir + std::string(leaf);
            std::ofstream t(name + ".txt");
            t << "; CS threads " << req.threads_x << "x" << req.threads_y << "x" << req.threads_z << " rsrc1 0x"
              << std::hex << req.cs_rsrc1 << " rsrc2 0x" << req.cs_rsrc2 << std::dec << "\n";
            for (const auto& i : cs.insts) t << "  " << gcn::disassemble(i) << "\n";
            if (!e.tp.ok) t << "; translation error: " << e.tp.error << "\n";
            if (e.tp.ok) std::ofstream(name + ".metal") << e.tp.msl;
        }
        if (!e.tp.ok) {
            e.error = "translate: " + e.tp.error;
        } else @autoreleasepool {
            NSError* compile_err = nil;
            NSString* src = [NSString stringWithUTF8String:e.tp.msl.c_str()];
            MTLCompileOptions* co = [[MTLCompileOptions alloc] init];
            id<MTLLibrary> lib = [im.device newLibraryWithSource:src options:co error:&compile_err];
            if (!lib) {
                e.error = std::string("MSL compile: ") + (compile_err ? [[compile_err localizedDescription] UTF8String] : "?");
            } else {
                id<MTLFunction> fn = [lib newFunctionWithName:@"gcn_cs_main"];
                if (!fn) {
                    e.error = "entry point gcn_cs_main not found";
                } else {
                    e.pso = [im.device newComputePipelineStateWithFunction:fn error:&compile_err];
                    if (!e.pso) {
                        e.error = std::string("compute pipeline: ") + (compile_err ? [[compile_err localizedDescription] UTF8String] : "?");
                    }
                }
            }
        }
        if (e.pso) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_compile).count();
            std::fprintf(stderr, "[GCN] compiled compute pipeline CS#%016llx in %.1f ms (cs %zu insts)\n",
                         (unsigned long long)key.cs_hash, ms, cs.insts.size());
        } else {
            std::fprintf(stderr, "[GCN] compute pipeline CS#%016llx unsupported: %s\n",
                         (unsigned long long)key.cs_hash, e.error.c_str());
        }
        pe = &im.compute_pipes.emplace(key, std::move(e)).first->second;
    }

    if (!pe->pso) return fail(pe->error);

    // Always scanned: it also evaluates the scalar loads (uniform_loads).
    gcn::ResourceScan scan =
        gcn::scan_resources(cs, pe->tp.images, req.cs_user, std::min<u32>(cs_user_count, 16), read, 0);
    if (!pe->tp.images.empty()) debug::capture_resources("CS", cs.hash, scan, pe->tp.images);
    if (!scan.ok && !pe->tp.images.empty()) {
        im.log_once("CS resource scan: " + scan.error);
        return fail("CS resource scan: " + scan.error);
    }
    if (!scan.ok) scan.uniform_loads.clear();  // the shader then loads them itself

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)compute_encoder;
        [enc setComputePipelineState:pe->pso];
        for (usize k = 0; k < pe->tp.images.size(); ++k) {
            const gcn::ImageSlot& slot = pe->tp.images[k];
            if (slot.storage) {
                id<MTLTexture> t = im.storage_texture_for(scan.images[k]);
                if (!t) return fail("image_store target unavailable");
                [enc setTexture:t atIndex:k];
            } else {
                [enc setTexture:im.texture_for(scan.images[k]) atIndex:k];
                if (slot.sampled) [enc setSamplerState:im.sampler_for(scan.images[k]) atIndex:k];
            }
        }

        id<MTLBuffer> mem = im.guest_memory_buffer();
        if (!mem) mem = [im.device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        [enc setBuffer:mem offset:0 atIndex:gcn::binding::kMemBuffer];

        gcn::ComputeConsts cc{};
        if (req.cs_user) {
            std::memcpy(cc.cs_user, req.cs_user, sizeof(cc.cs_user));
        }
        [enc setBytes:&cc length:sizeof(cc) atIndex:gcn::binding::kDrawConsts];
        static const u32 kNoTable = 0;
        if (scan.uniform_loads.empty())
            [enc setBytes:&kNoTable length:sizeof(kNoTable) atIndex:gcn::binding::kUniformLoads];
        else
            [enc setBytes:scan.uniform_loads.data() length:scan.uniform_loads.size() * sizeof(u32)
                  atIndex:gcn::binding::kUniformLoads];

        MTLSize threadgroups = MTLSizeMake(std::max<u32>(1, req.grid_x),
                                           std::max<u32>(1, req.grid_y),
                                           std::max<u32>(1, req.grid_z));
        MTLSize threadsPerGroup = MTLSizeMake(std::max<u32>(1, req.threads_x),
                                              std::max<u32>(1, req.threads_y),
                                              std::max<u32>(1, req.threads_z));
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    }
    return true;
}

} // namespace ps4
