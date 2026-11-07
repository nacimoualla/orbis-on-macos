#import <Metal/Metal.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#import <Foundation/Foundation.h>
#include "ps4/metal_backend.hpp"
#include "ps4/debug/capture.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>

namespace ps4 {

struct MetalBackend::Impl {
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> command_queue{nil};
    id<MTLTexture> framebuffer{nil};
    id<MTLCommandBuffer> active_command_buffer{nil};
    id<MTLRenderCommandEncoder> active_render_encoder{nil};
    id<MTLComputeCommandEncoder> active_compute_encoder{nil};
    id<MTLCommandBuffer> last_submitted_command_buffer{nil};
    u64 submitted_frames{0};
    bool last_frame_completed{true};
    u32 width{0};
    u32 height{0};

    // Depth/stencil buffers by guest depth-surface address (0 = none bound
    // yet). A new buffer is cleared to 0 by the first pass that uses it.
    struct DepthBuffer {
        id<MTLTexture> texture{nil};
        bool initialized{false};
    };
    std::map<u64, DepthBuffer> depth_buffers;
    u64 depth_key{0};
    u64 encoder_depth_key{0};

    // Offscreen color render targets by guest surface address.
    struct ColorBuffer {
        id<MTLTexture> texture{nil};
        u32 width{0};
        u32 height{0};
        MTLPixelFormat format{MTLPixelFormatBGRA8Unorm};
        bool initialized{false};
    };
    std::map<u64, ColorBuffer> color_buffers;
    std::array<u64, 8> color_keys{};
    std::array<u64, 8> encoder_color_keys{};
    std::set<u64> display_buffers;

    // (Re)create the texture for `key` when its size or format changes.
    // Only set_color_target() calls this; opening a pass just looks the
    // target up, so its contents survive across passes and frames.
    ColorBuffer& color_buffer(u64 key, u32 w, u32 h, MTLPixelFormat fmt) {
        ColorBuffer& cb = color_buffers[key];
        const u32 tw = w > 0 ? w : width;
        const u32 th = h > 0 ? h : height;
        if (!cb.texture || cb.width != tw || cb.height != th || cb.format != fmt) {
            MTLTextureDescriptor* d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                   width:tw
                                                                  height:th
                                                               mipmapped:NO];
            // ShaderWrite: compute image_store writes surfaces in place.
            d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            d.storageMode = MTLStorageModeShared;
            cb.texture = [device newTextureWithDescriptor:d];
            cb.width = tw;
            cb.height = th;
            cb.format = fmt;
            cb.initialized = false;
        }
        return cb;
    }
    ColorBuffer* find_color(u64 key) {
        if (key == 0 || display_buffers.count(key)) return nullptr;
        auto it = color_buffers.find(key);
        return it != color_buffers.end() && it->second.texture ? &it->second : nullptr;
    }

    // Depth buffers match the colour attachments of the pass they are used
    // with (Metal requires equal attachment sizes).
    DepthBuffer& depth_buffer(u64 key, u32 w, u32 h) {
        DepthBuffer& db = depth_buffers[key];
        if (!db.texture || db.texture.width != w || db.texture.height != h) {
            MTLTextureDescriptor* d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                                   width:w
                                                                  height:h
                                                               mipmapped:NO];
            d.usage = MTLTextureUsageRenderTarget;
            d.storageMode = MTLStorageModePrivate;
            db.texture = [device newTextureWithDescriptor:d];
            db.initialized = false;
        }
        return db;
    }

    void abort_active_frame() noexcept {
        active_render_encoder = nil;
        active_compute_encoder = nil;
        active_command_buffer = nil;
    }
};

MetalBackend::MetalBackend() : m_impl(std::make_unique<Impl>()) {}

MetalBackend::~MetalBackend() = default;

MetalBackend::MetalBackend(MetalBackend&&) noexcept = default;
MetalBackend& MetalBackend::operator=(MetalBackend&&) noexcept = default;

bool MetalBackend::initialize() {
    @autoreleasepool {
        m_impl->device = MTLCreateSystemDefaultDevice();
        if (!m_impl->device) {
            return false;
        }

        m_impl->command_queue = [m_impl->device newCommandQueue];
        if (!m_impl->command_queue) {
            return false;
        }

        return true;
    }
}

bool MetalBackend::is_initialized() const noexcept {
    return m_impl && m_impl->device != nil && m_impl->command_queue != nil;
}

std::string MetalBackend::device_name() const {
    if (!is_initialized()) {
        return "None";
    }
    @autoreleasepool {
        NSString* name = [m_impl->device name];
        return name ? [name UTF8String] : "Unknown Apple Silicon GPU";
    }
}

bool MetalBackend::create_framebuffer(u32 width, u32 height) {
    if (!is_initialized() || width == 0 || height == 0) {
        return false;
    }

    @autoreleasepool {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                         width:width
                                                                                        height:height
                                                                                     mipmapped:NO];
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        desc.storageMode = MTLStorageModeShared; // Unified memory accessible by both CPU and GPU

        m_impl->framebuffer = [m_impl->device newTextureWithDescriptor:desc];
        if (!m_impl->framebuffer) {
            return false;
        }

        m_impl->width = width;
        m_impl->height = height;
        m_impl->depth_buffers.clear();
        m_impl->color_buffers.clear();
        m_impl->color_keys.fill(0);
        m_impl->encoder_color_keys.fill(0);
        return true;
    }
}

u32 MetalBackend::width() const noexcept {
    return m_impl ? m_impl->width : 0;
}

u32 MetalBackend::height() const noexcept {
    return m_impl ? m_impl->height : 0;
}

bool MetalBackend::clear_screen(ColorRGBA color) {
    if (!begin_frame()) {
        return false;
    }

    if (!encode_clear(color)) {
        m_impl->abort_active_frame();
        return false;
    }

    return submit_frame(true);
}

bool MetalBackend::begin_frame() {
    if (!is_initialized()) {
        std::cerr << "[Metal] Cannot begin frame before initialization\n";
        return false;
    }
    if (!m_impl->framebuffer) {
        std::cerr << "[Metal] Cannot begin frame without a framebuffer\n";
        return false;
    }
    if (m_impl->active_command_buffer) {
        std::cerr << "[Metal] Cannot begin frame while another frame is active\n";
        return false;
    }

    @autoreleasepool {
        debug::capture_begin(m_impl->submitted_frames + 1, (__bridge void*)m_impl->device);
        m_impl->active_command_buffer = [m_impl->command_queue commandBuffer];
        if (!m_impl->active_command_buffer) {
            std::cerr << "[Metal] Failed to create frame command buffer\n";
            return false;
        }
        m_impl->active_render_encoder = nil;
        m_impl->active_compute_encoder = nil;
        return true;
    }
}

bool MetalBackend::encode_clear(ColorRGBA color) {
    if (!is_initialized()) {
        std::cerr << "[Metal] Cannot encode clear before initialization\n";
        return false;
    }
    if (!m_impl->framebuffer) {
        std::cerr << "[Metal] Cannot encode clear without a framebuffer\n";
        return false;
    }
    if (!m_impl->active_command_buffer || m_impl->active_render_encoder || m_impl->active_compute_encoder) {
        std::cerr << "[Metal] Cannot encode clear without an available active frame\n";
        return false;
    }

    @autoreleasepool {
        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_desc.colorAttachments[0].texture = m_impl->framebuffer;
        pass_desc.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass_desc.colorAttachments[0].clearColor = MTLClearColorMake(color.r, color.g, color.b, color.a);
        pass_desc.colorAttachments[0].storeAction = MTLStoreActionStore;

        m_impl->active_render_encoder =
            [m_impl->active_command_buffer renderCommandEncoderWithDescriptor:pass_desc];
        if (!m_impl->active_render_encoder) {
            std::cerr << "[Metal] Failed to create clear render encoder\n";
            m_impl->abort_active_frame();
            return false;
        }

        debug::capture_render_pass((__bridge void*)pass_desc, m_impl->color_keys.data());
        end_render_encoder();
        return true;
    }
}

bool MetalBackend::submit_frame(bool wait_for_completion) {
    if (!is_initialized()) {
        std::cerr << "[Metal] Cannot submit frame before initialization\n";
        return false;
    }

    if (!m_impl->active_command_buffer) {
        std::cerr << "[Metal] Cannot submit frame when no frame is active\n";
        return false;
    }

    @autoreleasepool {
        end_render_encoder();
        end_compute_encoder();
        wait_for_completion = wait_for_completion || debug::capture_active();

        id<MTLCommandBuffer> command_buffer = m_impl->active_command_buffer;
        // PS4_FLIP_STATS: how busy the GPU is, once a second.
        static const bool gpu_stats = std::getenv("PS4_FLIP_STATS") != nullptr;
        if (gpu_stats) {
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                static std::atomic<int> s_frames{0};
                static std::atomic<double> s_busy{0}, s_since{0};
                const double busy = cb.GPUEndTime - cb.GPUStartTime;
                s_frames.fetch_add(1);
                double total = s_busy.load();
                while (!s_busy.compare_exchange_weak(total, total + busy)) {}
                if (cb.GPUEndTime - s_since.load() >= 1.0) {
                    std::fprintf(stderr, "[fps] gpu %d frames, %.0f ms busy\n", s_frames.exchange(0),
                                 s_busy.exchange(0) * 1000.0);
                    s_since.store(cb.GPUEndTime);
                }
            }];
        }
        [command_buffer commit];
        if (wait_for_completion) {
            [command_buffer waitUntilCompleted];
        }

        m_impl->last_frame_completed =
            command_buffer.status == MTLCommandBufferStatusCompleted;
        if (command_buffer.status == MTLCommandBufferStatusError) {
            std::cerr << "[Metal] Frame command buffer failed\n";
        }
        m_impl->last_submitted_command_buffer =
            m_impl->last_frame_completed ? nil : command_buffer;
        if (debug::capture_active()) {
            debug::capture_texture(0, (__bridge void*)m_impl->framebuffer);
            for (const auto& [key, cb] : m_impl->color_buffers)
                if (cb.initialized) debug::capture_texture(key, (__bridge void*)cb.texture);
        }
        debug::capture_end();
        ++m_impl->submitted_frames;
        m_impl->active_command_buffer = nil;
        m_impl->active_render_encoder = nil;
        m_impl->active_compute_encoder = nil;

        debug::at_frame(m_impl->submitted_frames);
        return command_buffer.status != MTLCommandBufferStatusError &&
               (!wait_for_completion || m_impl->last_frame_completed);
    }
}

bool MetalBackend::ensure_render_encoder(bool clear, ColorRGBA color) {
    if (!is_initialized() || !m_impl->active_command_buffer || !m_impl->framebuffer) {
        return false;
    }
    if (m_impl->active_compute_encoder) {
        end_compute_encoder();
    }
    if (m_impl->active_render_encoder) {
        return true;
    }
    @autoreleasepool {
        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];

        // Colour attachments. Slot 0 is the screen framebuffer unless an
        // offscreen target is bound; only the screen takes the frame's clear.
        // Offscreen targets keep their contents (a new one starts zeroed).
        u32 pass_w = m_impl->width, pass_h = m_impl->height;
        for (u32 slot = 0; slot < 8; ++slot) {
            Impl::ColorBuffer* cb = m_impl->find_color(m_impl->color_keys[slot]);
            auto* att = pass_desc.colorAttachments[slot];
            if (cb) {
                att.texture = cb->texture;
                att.loadAction = cb->initialized ? MTLLoadActionLoad : MTLLoadActionClear;
                att.clearColor = MTLClearColorMake(0, 0, 0, 0);
                cb->initialized = true;
                if (slot == 0) { pass_w = cb->width; pass_h = cb->height; }
            } else if (slot == 0) {
                att.texture = m_impl->framebuffer;
                att.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
                att.clearColor = MTLClearColorMake(color.r, color.g, color.b, color.a);
            } else {
                continue;
            }
            att.storeAction = MTLStoreActionStore;
        }

        Impl::DepthBuffer& db = m_impl->depth_buffer(m_impl->depth_key, pass_w, pass_h);
        if (db.texture) {
            // The host buffer stands in for the guest depth surface, so its
            // contents persist across passes and frames.
            const MTLLoadAction load = db.initialized ? MTLLoadActionLoad : MTLLoadActionClear;
            pass_desc.depthAttachment.texture = db.texture;
            pass_desc.depthAttachment.loadAction = load;
            pass_desc.depthAttachment.clearDepth = 0.0;
            pass_desc.depthAttachment.storeAction = MTLStoreActionStore;
            pass_desc.stencilAttachment.texture = db.texture;
            pass_desc.stencilAttachment.loadAction = load;
            pass_desc.stencilAttachment.clearStencil = 0;
            pass_desc.stencilAttachment.storeAction = MTLStoreActionStore;
            db.initialized = true;
        }
        debug::capture_render_pass((__bridge void*)pass_desc, m_impl->color_keys.data());
        m_impl->active_render_encoder =
            [m_impl->active_command_buffer renderCommandEncoderWithDescriptor:pass_desc];
        m_impl->encoder_depth_key = m_impl->depth_key;
        m_impl->encoder_color_keys = m_impl->color_keys;
        return m_impl->active_render_encoder != nil;
    }
}

void* MetalBackend::active_render_encoder() const noexcept {
    return m_impl ? (__bridge void*)m_impl->active_render_encoder : nullptr;
}

bool MetalBackend::ensure_compute_encoder() {
    if (!is_initialized() || !m_impl->active_command_buffer) {
        return false;
    }
    if (m_impl->active_render_encoder) {
        end_render_encoder();
    }
    if (m_impl->active_compute_encoder) {
        return true;
    }
    @autoreleasepool {
        m_impl->active_compute_encoder = [m_impl->active_command_buffer computeCommandEncoder];
        return m_impl->active_compute_encoder != nil;
    }
}

void MetalBackend::end_compute_encoder() {
    if (m_impl && m_impl->active_compute_encoder) {
        [m_impl->active_compute_encoder endEncoding];
        m_impl->active_compute_encoder = nil;
        if (debug::capture_active()) {
            debug::capture_pass((__bridge void*)m_impl->active_command_buffer, "compute");
            m_impl->active_command_buffer = [m_impl->command_queue commandBuffer];
        }
    }
}

void* MetalBackend::active_compute_encoder() const noexcept {
    return m_impl ? (__bridge void*)m_impl->active_compute_encoder : nullptr;
}

u32 MetalBackend::depth_stencil_format() const noexcept {
    return static_cast<u32>(MTLPixelFormatDepth32Float_Stencil8);
}

void MetalBackend::set_depth_target(u64 key) {
    if (!m_impl || key == m_impl->depth_key) return;
    m_impl->depth_key = key;
    if (m_impl->active_render_encoder && m_impl->encoder_depth_key != key) end_render_encoder();
    // Bound the number of host depth buffers a game can make us keep.
    if (m_impl->depth_buffers.size() > 32) {
        for (auto it = m_impl->depth_buffers.begin(); it != m_impl->depth_buffers.end();) {
            if (it->first != key && it->first != m_impl->encoder_depth_key) it = m_impl->depth_buffers.erase(it);
            else ++it;
        }
    }
}

void MetalBackend::register_display_buffer(u64 addr) {
    if (m_impl && addr != 0) {
        m_impl->display_buffers.insert(addr);
    }
}

bool MetalBackend::is_display_buffer(u64 addr) const noexcept {
    return m_impl && (addr == 0 || m_impl->display_buffers.count(addr) > 0);
}

void MetalBackend::set_color_target(u32 slot, u64 key, u32 width, u32 height, u32 pixel_format) {
    if (!m_impl || slot >= 8) return;
    const bool offscreen = key != 0 && !m_impl->display_buffers.count(key);
    const MTLPixelFormat fmt = pixel_format ? static_cast<MTLPixelFormat>(pixel_format) : MTLPixelFormatBGRA8Unorm;
    // A changed format, or a size the texture cannot hold, replaces it: the
    // open pass may still reference the old one. A smaller request keeps the
    // texture. The size comes from the screen scissor, which games change
    // per pass (Dragon Ball Xenoverse 2: 1080, 1078, 1073, 1064 rows), and
    // replacing the surface each time discarded the scene drawn into it.
    const Impl::ColorBuffer* cur = offscreen ? m_impl->find_color(key) : nullptr;
    const bool reshape = offscreen && (!cur || (width && width > cur->width) || (height && height > cur->height) ||
                                       cur->format != fmt);
    if (m_impl->color_keys[slot] == key && !reshape) return;
    if (m_impl->active_render_encoder) {
        bool bound = m_impl->encoder_color_keys[slot] != key;
        for (u32 s = 0; s < 8 && !bound; ++s) bound = reshape && m_impl->encoder_color_keys[s] == key;
        if (bound) end_render_encoder();
    }
    m_impl->color_keys[slot] = key;
    if (reshape) {
        // Growing in one dimension must not shrink the other.
        const bool same_fmt = cur && cur->format == fmt;
        m_impl->color_buffer(key, same_fmt ? std::max(width, cur->width) : width,
                             same_fmt ? std::max(height, cur->height) : height, fmt);
    }
    if (m_impl->color_buffers.size() > 32) {
        for (auto it = m_impl->color_buffers.begin(); it != m_impl->color_buffers.end();) {
            bool in_use = false;
            for (u32 s = 0; s < 8; ++s) {
                if (it->first == m_impl->color_keys[s] || it->first == m_impl->encoder_color_keys[s]) {
                    in_use = true;
                    break;
                }
            }
            if (!in_use) it = m_impl->color_buffers.erase(it);
            else ++it;
        }
    }
}

void MetalBackend::reset_color_targets() {
    if (!m_impl) return;
    bool had_nonzero = false;
    for (u32 s = 0; s < 8; ++s) {
        if (m_impl->color_keys[s] != 0) had_nonzero = true;
        m_impl->color_keys[s] = 0;
    }
    if (had_nonzero && m_impl->active_render_encoder) {
        end_render_encoder();
    }
}

bool MetalBackend::resolve_color(u64 src_key, u64 dst_key) {
    if (!is_initialized() || !m_impl->active_command_buffer) return false;
    auto surface = [this](u64 key) -> id<MTLTexture> {
        if (key == 0 || m_impl->display_buffers.count(key)) return m_impl->framebuffer;
        Impl::ColorBuffer* cb = m_impl->find_color(key);
        return cb && cb->initialized ? cb->texture : nil;
    };
    id<MTLTexture> src = surface(src_key);
    if (!src) return false;
    id<MTLTexture> dst = surface(dst_key);
    if (!dst && dst_key != 0) {   // an offscreen destination nothing has drawn to yet
        Impl::ColorBuffer& cb = m_impl->color_buffer(dst_key, static_cast<u32>(src.width),
                                                     static_cast<u32>(src.height), src.pixelFormat);
        cb.initialized = true;
        dst = cb.texture;
    }
    if (!dst || dst == src) return false;
    if (dst.pixelFormat != src.pixelFormat) {
        static bool s_logged = false;
        if (!s_logged) {
            s_logged = true;
            std::fprintf(stderr, "[Metal] resolve between formats %lu and %lu is not supported\n",
                         (unsigned long)src.pixelFormat, (unsigned long)dst.pixelFormat);
        }
        return false;
    }
    end_render_encoder();
    end_compute_encoder();
    @autoreleasepool {
        id<MTLBlitCommandEncoder> blit = [m_impl->active_command_buffer blitCommandEncoder];
        if (!blit) return false;
        const NSUInteger w = std::min(src.width, dst.width), h = std::min(src.height, dst.height);
        [blit copyFromTexture:src
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(w, h, 1)
                    toTexture:dst
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
    }
    return true;
}

void* MetalBackend::find_color_target(u64 key) const noexcept {
    if (!m_impl || key == 0) return nullptr;
    auto it = m_impl->color_buffers.find(key);
    if (it != m_impl->color_buffers.end() && it->second.texture) {
        return (__bridge void*)it->second.texture;
    }
    return nullptr;
}

void* MetalBackend::surface_texture(u64 key) const noexcept {
    if (!m_impl || key == 0) return nullptr;
    if (m_impl->display_buffers.count(key)) return (__bridge void*)m_impl->framebuffer;
    return find_color_target(key);
}

void* MetalBackend::storage_target(u64 key, u32 width, u32 height, u32 pixel_format) {
    if (!is_initialized() || key == 0) return nullptr;
    if (m_impl->display_buffers.count(key)) return (__bridge void*)m_impl->framebuffer;
    if (Impl::ColorBuffer* cb = m_impl->find_color(key)) {
        cb->initialized = true;
        return (__bridge void*)cb->texture;
    }
    const MTLPixelFormat fmt = pixel_format ? static_cast<MTLPixelFormat>(pixel_format) : MTLPixelFormatRGBA8Unorm;
    Impl::ColorBuffer& cb = m_impl->color_buffer(key, width, height, fmt);
    cb.initialized = true;
    return (__bridge void*)cb.texture;
}

void* MetalBackend::color_target_texture(u32 slot) const noexcept {
    if (!m_impl || slot >= 8) return nullptr;
    if (Impl::ColorBuffer* cb = m_impl->find_color(m_impl->color_keys[slot])) return (__bridge void*)cb->texture;
    return slot == 0 ? (__bridge void*)m_impl->framebuffer : nullptr;
}

bool MetalBackend::read_color_pixel(u64 key, u32 x, u32 y, u8& out_r, u8& out_g, u8& out_b, u8& out_a) const {
    if (!is_initialized()) return false;
    if (key == 0 || (m_impl && m_impl->display_buffers.count(key))) {
        return read_pixel(x, y, out_r, out_g, out_b, out_a);
    }
    auto it = m_impl->color_buffers.find(key);
    if (it == m_impl->color_buffers.end() || !it->second.texture) return false;
    id<MTLTexture> tex = it->second.texture;
    if (x >= it->second.width || y >= it->second.height) return false;
    @autoreleasepool {
        u8 raw[4] = {0, 0, 0, 0};
        MTLRegion region = MTLRegionMake2D(x, y, 1, 1);
        [tex getBytes:raw bytesPerRow:4 fromRegion:region mipmapLevel:0];
        if (tex.pixelFormat == MTLPixelFormatBGRA8Unorm) {
            out_b = raw[0];
            out_g = raw[1];
            out_r = raw[2];
            out_a = raw[3];
        } else {
            out_r = raw[0];
            out_g = raw[1];
            out_b = raw[2];
            out_a = raw[3];
        }
        return true;
    }
}

void MetalBackend::end_render_encoder() {
    if (m_impl && m_impl->active_render_encoder) {
        [m_impl->active_render_encoder endEncoding];
        m_impl->active_render_encoder = nil;
        if (debug::capture_active()) {
            debug::capture_pass((__bridge void*)m_impl->active_command_buffer, "render");
            m_impl->active_command_buffer = [m_impl->command_queue commandBuffer];
        }
    }
}

bool MetalBackend::wait_for_last_frame() {
    if (!is_initialized()) {
        std::cerr << "[Metal] Cannot wait for a frame before initialization\n";
        return false;
    }
    if (!m_impl->framebuffer) {
        std::cerr << "[Metal] Cannot wait for a frame without a framebuffer\n";
        return false;
    }
    if (!m_impl->last_submitted_command_buffer) {
        return true;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = m_impl->last_submitted_command_buffer;
        if (command_buffer.status != MTLCommandBufferStatusCompleted &&
            command_buffer.status != MTLCommandBufferStatusError) {
            [command_buffer waitUntilCompleted];
        }
        m_impl->last_frame_completed =
            command_buffer.status == MTLCommandBufferStatusCompleted;
        if (command_buffer.status == MTLCommandBufferStatusError) {
            std::cerr << "[Metal] Last frame command buffer failed\n";
        }
        m_impl->last_submitted_command_buffer = nil;
        return m_impl->last_frame_completed;
    }
}

bool MetalBackend::frame_in_progress() const noexcept {
    return m_impl && m_impl->active_command_buffer != nil;
}

u64 MetalBackend::submitted_frame_count() const noexcept {
    return m_impl ? m_impl->submitted_frames : 0;
}

bool MetalBackend::read_pixel(u32 x, u32 y, u8& out_r, u8& out_g, u8& out_b, u8& out_a) const {
    if (!is_initialized() || !m_impl->framebuffer) {
        return false;
    }

    if (x >= m_impl->width || y >= m_impl->height) {
        return false;
    }

    @autoreleasepool {
        u8 bgra[4] = {0, 0, 0, 0};
        MTLRegion region = MTLRegionMake2D(x, y, 1, 1);
        [m_impl->framebuffer getBytes:bgra bytesPerRow:4 fromRegion:region mipmapLevel:0];

        // Format is BGRA8Unorm
        out_b = bgra[0];
        out_g = bgra[1];
        out_r = bgra[2];
        out_a = bgra[3];
        return true;
    }
}

bool MetalBackend::upload_pixels(const void* pixel_data, usize bytes_per_row) {
    if (!is_initialized() || !m_impl->framebuffer || !pixel_data) {
        return false;
    }

    @autoreleasepool {
        MTLRegion region = MTLRegionMake2D(0, 0, m_impl->width, m_impl->height);
        [m_impl->framebuffer replaceRegion:region
                              mipmapLevel:0
                                withBytes:pixel_data
                              bytesPerRow:bytes_per_row];
        return true;
    }
}

void* MetalBackend::raw_device() const noexcept {
    return m_impl ? (__bridge void*)m_impl->device : nullptr;
}

void* MetalBackend::raw_command_queue() const noexcept {
    return m_impl ? (__bridge void*)m_impl->command_queue : nullptr;
}

void* MetalBackend::raw_framebuffer() const noexcept {
    return m_impl ? (__bridge void*)m_impl->framebuffer : nullptr;
}

} // namespace ps4
