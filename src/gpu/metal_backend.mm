#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "ps4/metal_backend.hpp"
#include <iostream>
#include <map>

namespace ps4 {

struct MetalBackend::Impl {
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> command_queue{nil};
    id<MTLTexture> framebuffer{nil};
    id<MTLCommandBuffer> active_command_buffer{nil};
    id<MTLRenderCommandEncoder> active_render_encoder{nil};
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

    DepthBuffer& depth_buffer(u64 key) {
        DepthBuffer& db = depth_buffers[key];
        if (!db.texture) {
            MTLTextureDescriptor* d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                                   width:width
                                                                  height:height
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
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared; // Unified memory accessible by both CPU and GPU

        m_impl->framebuffer = [m_impl->device newTextureWithDescriptor:desc];
        if (!m_impl->framebuffer) {
            return false;
        }

        m_impl->width = width;
        m_impl->height = height;
        m_impl->depth_buffers.clear();
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
        m_impl->active_command_buffer = [m_impl->command_queue commandBuffer];
        if (!m_impl->active_command_buffer) {
            std::cerr << "[Metal] Failed to create frame command buffer\n";
            return false;
        }
        m_impl->active_render_encoder = nil;
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
    if (!m_impl->active_command_buffer || m_impl->active_render_encoder) {
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

        [m_impl->active_render_encoder endEncoding];
        m_impl->active_render_encoder = nil;
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
        if (m_impl->active_render_encoder) {
            [m_impl->active_render_encoder endEncoding];
            m_impl->active_render_encoder = nil;
        }

        id<MTLCommandBuffer> command_buffer = m_impl->active_command_buffer;
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
        ++m_impl->submitted_frames;
        m_impl->active_command_buffer = nil;
        m_impl->active_render_encoder = nil;

        return command_buffer.status != MTLCommandBufferStatusError &&
               (!wait_for_completion || m_impl->last_frame_completed);
    }
}

bool MetalBackend::ensure_render_encoder(bool clear, ColorRGBA color) {
    if (!is_initialized() || !m_impl->active_command_buffer || !m_impl->framebuffer) {
        return false;
    }
    if (m_impl->active_render_encoder) {
        return true;
    }
    @autoreleasepool {
        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_desc.colorAttachments[0].texture = m_impl->framebuffer;
        pass_desc.colorAttachments[0].loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
        pass_desc.colorAttachments[0].clearColor = MTLClearColorMake(color.r, color.g, color.b, color.a);
        pass_desc.colorAttachments[0].storeAction = MTLStoreActionStore;
        Impl::DepthBuffer& db = m_impl->depth_buffer(m_impl->depth_key);
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
        m_impl->active_render_encoder =
            [m_impl->active_command_buffer renderCommandEncoderWithDescriptor:pass_desc];
        m_impl->encoder_depth_key = m_impl->depth_key;
        return m_impl->active_render_encoder != nil;
    }
}

void* MetalBackend::active_render_encoder() const noexcept {
    return m_impl ? (__bridge void*)m_impl->active_render_encoder : nullptr;
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

void MetalBackend::end_render_encoder() {
    if (m_impl && m_impl->active_render_encoder) {
        [m_impl->active_render_encoder endEncoding];
        m_impl->active_render_encoder = nil;
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
