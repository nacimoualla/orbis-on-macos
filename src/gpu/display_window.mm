#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "ps4/display_window.hpp"
#include <iostream>

namespace {

const char* DISPLAY_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut displayVertexShader(uint vertexID [[vertex_id]]) {
    // Fullscreen triangle covering clip space [-1, 1] with UV mapping [0, 1]
    float2 uvs[3] = { float2(0.0, 2.0), float2(0.0, 0.0), float2(2.0, 0.0) };
    float4 positions[3] = { float4(-1.0, 3.0, 0.0, 1.0), float4(-1.0, -1.0, 0.0, 1.0), float4(3.0, -1.0, 0.0, 1.0) };
    VertexOut out;
    out.position = positions[vertexID];
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 displayFragmentShader(VertexOut in [[stage_in]],
                                     texture2d<float> colorTexture [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    return colorTexture.sample(textureSampler, in.uv);
}
)";

} // namespace

@interface PS4WindowDelegate : NSObject <NSWindowDelegate>
@property (nonatomic, assign) bool* openFlag;
@end

@implementation PS4WindowDelegate
- (void)windowWillClose:(NSNotification *)notification {
    (void)notification;
    if (self.openFlag) {
        *self.openFlag = false;
    }
}
@end

namespace ps4 {

struct DisplayWindow::Impl {
    WindowConfig config{};
    bool is_open{false};
    u64 presented_frames{0};
    u64 skipped_presentations{0};
    std::chrono::steady_clock::time_point next_frame_deadline{};
    MetalBackend* metal{nullptr};
    id<MTLDevice> device{nil};
    id<MTLCommandQueue> command_queue{nil};
    id<MTLTexture> framebuffer_texture{nil};
    id<MTLRenderPipelineState> pipeline_state{nil};

    NSWindow* window{nil};
    CAMetalLayer* metal_layer{nil};
    PS4WindowDelegate* delegate{nil};
    u32 buttons{0};
};

DisplayWindow::DisplayWindow() : m_impl(std::make_unique<Impl>()) {}

DisplayWindow::~DisplayWindow() {
    close();
}

DisplayWindow::DisplayWindow(DisplayWindow&&) noexcept = default;
DisplayWindow& DisplayWindow::operator=(DisplayWindow&&) noexcept = default;

bool DisplayWindow::initialize(MetalBackend& metal, const WindowConfig& config) {
    if (!metal.is_initialized()) {
        return false;
    }

    m_impl->config = config;
    m_impl->metal = &metal;
    m_impl->presented_frames = 0;
    m_impl->skipped_presentations = 0;
    m_impl->next_frame_deadline = std::chrono::steady_clock::now();
    m_impl->device = (__bridge id<MTLDevice>)metal.raw_device();
    m_impl->command_queue = (__bridge id<MTLCommandQueue>)metal.raw_command_queue();
    m_impl->framebuffer_texture = (__bridge id<MTLTexture>)metal.raw_framebuffer();

    if (!m_impl->device || !m_impl->command_queue || !m_impl->framebuffer_texture) {
        return false;
    }

    @autoreleasepool {
        // 1. Compile fullscreen presentation shader
        NSError* error = nil;
        id<MTLLibrary> lib = [m_impl->device newLibraryWithSource:[NSString stringWithUTF8String:DISPLAY_SHADER_SOURCE]
                                                          options:nil
                                                            error:&error];
        if (!lib) {
            std::cerr << "[DisplayWindow] Failed to compile presentation shader: "
                      << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }

        MTLRenderPipelineDescriptor* pdesc = [[MTLRenderPipelineDescriptor alloc] init];
        pdesc.vertexFunction = [lib newFunctionWithName:@"displayVertexShader"];
        pdesc.fragmentFunction = [lib newFunctionWithName:@"displayFragmentShader"];
        pdesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

        m_impl->pipeline_state = [m_impl->device newRenderPipelineStateWithDescriptor:pdesc error:&error];
        if (!m_impl->pipeline_state) {
            std::cerr << "[DisplayWindow] Failed to create presentation pipeline: "
                      << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }

        m_impl->is_open = true;

        // If headless, skip native AppKit window creation
        if (config.headless) {
            return true;
        }

        // 2. Initialize AppKit Application
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        // 3. Create NSWindow
        NSRect frame = NSMakeRect(0, 0, config.width, config.height);
        NSWindowStyleMask styleMask = NSWindowStyleMaskTitled |
                                      NSWindowStyleMaskClosable |
                                      NSWindowStyleMaskMiniaturizable;
        if (config.resizable) {
            styleMask |= NSWindowStyleMaskResizable;
        }

        m_impl->window = [[NSWindow alloc] initWithContentRect:frame
                                                     styleMask:styleMask
                                                       backing:NSBackingStoreBuffered
                                                         defer:NO];
        if (!m_impl->window) {
            m_impl->is_open = false;
            return false;
        }

        [m_impl->window setTitle:[NSString stringWithUTF8String:config.title.c_str()]];
        [m_impl->window center];

        m_impl->delegate = [[PS4WindowDelegate alloc] init];
        m_impl->delegate.openFlag = &m_impl->is_open;
        [m_impl->window setDelegate:m_impl->delegate];

        // 4. Setup CAMetalLayer
        NSView* contentView = [m_impl->window contentView];
        m_impl->metal_layer = [CAMetalLayer layer];
        m_impl->metal_layer.device = m_impl->device;
        m_impl->metal_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        m_impl->metal_layer.framebufferOnly = YES;
        m_impl->metal_layer.drawableSize = CGSizeMake(config.width, config.height);
        m_impl->metal_layer.contentsScale = [m_impl->window backingScaleFactor];

        [contentView setWantsLayer:YES];
        [contentView setLayer:m_impl->metal_layer];

        [m_impl->window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        return true;
    }
}

bool DisplayWindow::is_open() const noexcept {
    return m_impl && m_impl->is_open;
}

void DisplayWindow::poll_events() {
    if (!m_impl || !m_impl->is_open || m_impl->config.headless) {
        return;
    }

    @autoreleasepool {
        while (true) {
            NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                untilDate:[NSDate distantPast]
                                                   inMode:NSDefaultRunLoopMode
                                                  dequeue:YES];
            if (!event) break;

            // Close on Escape key (keyCode 53)
            if ([event type] == NSEventTypeKeyDown && [event keyCode] == 53) {
                close();
                break;
            }

            // Map Space bar (keyCode 49), Enter (keyCode 36), or Left Click / Touch to PS4 Cross (0x4000)
            constexpr u32 PS4_BUTTON_CROSS = 0x4000;
            if ([event type] == NSEventTypeKeyDown) {
                const unsigned short code = [event keyCode];
                if (code == 49 || code == 36 || code == 126 /* Up */ || code == 124 /* Right */) {
                    m_impl->buttons |= PS4_BUTTON_CROSS;
                }
            } else if ([event type] == NSEventTypeKeyUp) {
                const unsigned short code = [event keyCode];
                if (code == 49 || code == 36 || code == 126 || code == 124) {
                    m_impl->buttons &= ~PS4_BUTTON_CROSS;
                }
            } else if ([event type] == NSEventTypeLeftMouseDown) {
                m_impl->buttons |= PS4_BUTTON_CROSS;
            } else if ([event type] == NSEventTypeLeftMouseUp) {
                m_impl->buttons &= ~PS4_BUTTON_CROSS;
            }

            [NSApp sendEvent:event];
        }
        [NSApp updateWindows];
    }
}

u32 DisplayWindow::input_buttons() const noexcept {
    return m_impl ? m_impl->buttons : 0;
}

bool DisplayWindow::present_frame() {
    if (!m_impl || !m_impl->is_open) {
        return false;
    }

    if (m_impl->config.headless) {
        return m_impl->pipeline_state != nil;
    }

    @autoreleasepool {
        if (!m_impl->metal_layer) {
            return false;
        }

        // Dynamically match drawable size if window resized
        NSView* contentView = [m_impl->window contentView];
        if (contentView) {
            const CGSize viewSize = contentView.bounds.size;
            const CGFloat scale = [m_impl->window backingScaleFactor];
            m_impl->metal_layer.drawableSize = CGSizeMake(viewSize.width * scale, viewSize.height * scale);
        }

        id<CAMetalDrawable> drawable = [m_impl->metal_layer nextDrawable];
        if (!drawable) {
            ++m_impl->skipped_presentations;
            poll_events();
            return true;
        }

        // The guest producer and presenter share the backend command queue.
        // If a newly submitted producer frame is still outstanding, resolve
        // that ordering before sampling it for presentation. This is after
        // nextDrawable so drawable starvation remains non-blocking.
        if (m_impl->metal &&
            m_impl->metal->submitted_frame_count() > m_impl->presented_frames &&
            !m_impl->metal->wait_for_last_frame()) {
            return false;
        }

        MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_desc.colorAttachments[0].texture = drawable.texture;
        pass_desc.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass_desc.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        pass_desc.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLCommandBuffer> cmd_buffer = [m_impl->command_queue commandBuffer];
        if (!cmd_buffer) {
            return false;
        }

        id<MTLRenderCommandEncoder> encoder = [cmd_buffer renderCommandEncoderWithDescriptor:pass_desc];
        if (!encoder) {
            return false;
        }

        [encoder setRenderPipelineState:m_impl->pipeline_state];
        [encoder setFragmentTexture:m_impl->framebuffer_texture atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];

        [cmd_buffer presentDrawable:drawable];
        [cmd_buffer commit];

        ++m_impl->presented_frames;
        poll_events();
        return true;
    }
}

void DisplayWindow::close() {
    if (m_impl && m_impl->is_open) {
        m_impl->is_open = false;
        if (m_impl->window) {
            @autoreleasepool {
                [m_impl->window setDelegate:nil];
                [m_impl->window close];
                m_impl->window = nil;
            }
        }
    }
}

u32 DisplayWindow::width() const noexcept {
    if (!m_impl) return 0;
    if (m_impl->window) {
        return static_cast<u32>([[m_impl->window contentView] bounds].size.width);
    }
    return m_impl->config.width;
}

u32 DisplayWindow::height() const noexcept {
    if (!m_impl) return 0;
    if (m_impl->window) {
        return static_cast<u32>([[m_impl->window contentView] bounds].size.height);
    }
    return m_impl->config.height;
}

const WindowConfig& DisplayWindow::config() const noexcept {
    static const WindowConfig s_default_config{};
    return m_impl ? m_impl->config : s_default_config;
}

} // namespace ps4
