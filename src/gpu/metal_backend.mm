#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "ps4/metal_backend.hpp"

namespace ps4 {

struct MetalBackend::Impl {
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> command_queue{nil};
    id<MTLTexture> framebuffer{nil};
    u32 width{0};
    u32 height{0};
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
    if (!is_initialized() || !m_impl->framebuffer) {
        return false;
    }

    @autoreleasepool {
        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_desc.colorAttachments[0].texture = m_impl->framebuffer;
        pass_desc.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass_desc.colorAttachments[0].clearColor = MTLClearColorMake(color.r, color.g, color.b, color.a);
        pass_desc.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLCommandBuffer> cmd_buffer = [m_impl->command_queue commandBuffer];
        if (!cmd_buffer) {
            return false;
        }

        id<MTLRenderCommandEncoder> encoder = [cmd_buffer renderCommandEncoderWithDescriptor:pass_desc];
        if (!encoder) {
            return false;
        }

        [encoder endEncoding];
        [cmd_buffer commit];
        [cmd_buffer waitUntilCompleted];

        return cmd_buffer.status == MTLCommandBufferStatusCompleted;
    }
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
