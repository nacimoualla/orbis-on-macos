#import <AppKit/AppKit.h>
#import <GameController/GameController.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/QuartzCore.h>
#include <mach/mach.h>
#include "ps4/display_window.hpp"
#include <algorithm>
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
    // Fullscreen triangle covering clip space [-1, 1] with UV mapping [0, 1].
    // Clip-space y points up but texture v = 0 is the framebuffer's top row
    // (as on the PS4), so v = (1 - y) / 2; the old (y + 1) / 2 mapping showed
    // every game upside down in the window while offscreen readbacks were fine.
    float2 uvs[3] = { float2(0.0, -1.0), float2(0.0, 1.0), float2(2.0, 1.0) };
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
    PadState active_pad_state{};
    mutable MouseState active_mouse_state{};
    bool key_down[512]{false};
    bool mouse_left_down{false};
    bool mouse_right_down{false};
    u32 buttons{0};
    // Buttons pressed since the last input_buttons() query: a click whose
    // down and up arrive in the same event batch would otherwise never be seen.
    mutable u32 pressed_latch{0};

    // Telemetry HUD overlay
    bool show_overlay{true};
    CALayer* hud_layer{nil};
    CATextLayer* hud_text_layer{nil};
    std::chrono::steady_clock::time_point last_stats_time{};
    u64 last_presented_count{0};
    double current_fps{0.0};
    double current_frame_time_ms{0.0};
    float current_cpu_pct{0.0f};
    float current_ram_mb{0.0f};
    double current_gpu_ms{0.0f};

    void update_stats_overlay();
};

namespace {

// Orbis pad button bits come from ps4/input.hpp; these are the names the
// keyboard and controller maps below use.
constexpr u32 kPadOptions = kPadStart, kPadTouchPad = kPadTouchPadClick;

// Keyboard -> pad. Space and a left click press Cross and the touch pad
// together: games that say "touch pad to start" read the pad, others Cross.
[[maybe_unused]] u32 key_to_pad(unsigned short code) {
    switch (code) {
    case 123: return kPadLeft;                  // arrows
    case 124: return kPadRight;
    case 125: return kPadDown;
    case 126: return kPadUp;
    case 49: return kPadCross | kPadTouchPad;   // Space
    case 6: return kPadCross;                   // Z
    case 7: return kPadCircle;                  // X
    case 8: return kPadSquare;                  // C
    case 9: return kPadTriangle;                // V
    case 36: return kPadOptions;                // Return
    case 0: return kPadL1;                      // A
    case 1: return kPadR1;                      // S
    case 12: return kPadL2;                     // Q
    case 14: return kPadR2;                     // E
    case 48: return kPadTouchPad;               // Tab
    default: return 0;
    }
}

// Every connected game controller (DualShock 4, DualSense, Xbox, MFi): D-pad
// or left stick, face buttons by position, shoulders, triggers, Menu.
u32 gamepad_buttons() {
    u32 b = 0;
    for (GCController* c in [GCController controllers]) {
        GCExtendedGamepad* g = c.extendedGamepad;
        if (!g) continue;
        const float x = g.leftThumbstick.xAxis.value, y = g.leftThumbstick.yAxis.value;
        if (g.dpad.up.pressed || y > 0.5f) b |= kPadUp;
        if (g.dpad.down.pressed || y < -0.5f) b |= kPadDown;
        if (g.dpad.left.pressed || x < -0.5f) b |= kPadLeft;
        if (g.dpad.right.pressed || x > 0.5f) b |= kPadRight;
        if (g.buttonA.pressed) b |= kPadCross;
        if (g.buttonB.pressed) b |= kPadCircle;
        if (g.buttonX.pressed) b |= kPadSquare;
        if (g.buttonY.pressed) b |= kPadTriangle;
        if (g.leftShoulder.pressed) b |= kPadL1;
        if (g.rightShoulder.pressed) b |= kPadR1;
        if (g.leftTrigger.pressed) b |= kPadL2;
        if (g.rightTrigger.pressed) b |= kPadR2;
        if (g.buttonMenu.pressed) b |= kPadOptions;
        if (g.leftThumbstickButton.pressed) b |= kPadL3;
        if (g.rightThumbstickButton.pressed) b |= kPadR3;
        if ([g isKindOfClass:[GCDualShockGamepad class]] &&
            ((GCDualShockGamepad*)g).touchpadButton.pressed) b |= kPadTouchPad;
        if ([g isKindOfClass:[GCDualSenseGamepad class]] &&
            ((GCDualSenseGamepad*)g).touchpadButton.pressed) b |= kPadTouchPad;
    }
    return b;
}

} // namespace

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

        // 5. Setup stats HUD overlay
        const char* env_overlay = std::getenv("PS4_STATS_OVERLAY");
        if (env_overlay) {
            m_impl->show_overlay = (std::strcmp(env_overlay, "0") != 0 &&
                                    std::strcmp(env_overlay, "false") != 0 &&
                                    std::strcmp(env_overlay, "off") != 0);
        } else {
            m_impl->show_overlay = config.show_stats_overlay;
        }

        m_impl->hud_layer = [CALayer layer];
        m_impl->hud_layer.backgroundColor = CGColorCreateGenericRGB(0.06, 0.08, 0.12, 0.85);
        m_impl->hud_layer.borderColor = CGColorCreateGenericRGB(0.25, 0.38, 0.50, 0.65);
        m_impl->hud_layer.borderWidth = 1.0;
        m_impl->hud_layer.cornerRadius = 8.0;
        m_impl->hud_layer.masksToBounds = YES;
        m_impl->hud_layer.zPosition = 1000.0;

        m_impl->hud_text_layer = [CATextLayer layer];
        m_impl->hud_text_layer.font = (__bridge CFTypeRef)[NSFont monospacedSystemFontOfSize:11.0 weight:NSFontWeightMedium];
        m_impl->hud_text_layer.fontSize = 11.0;
        m_impl->hud_text_layer.contentsScale = [m_impl->window backingScaleFactor];
        m_impl->hud_text_layer.alignmentMode = kCAAlignmentLeft;
        m_impl->hud_text_layer.zPosition = 1001.0;

        NSDictionary* no_anim = @{
            @"contents": [NSNull null],
            @"bounds": [NSNull null],
            @"position": [NSNull null],
            @"string": [NSNull null],
            @"hidden": [NSNull null],
            @"frame": [NSNull null]
        };
        m_impl->hud_layer.actions = no_anim;
        m_impl->hud_text_layer.actions = no_anim;

        const CGFloat hud_w = 210.0;
        const CGFloat hud_h = 92.0;
        const CGFloat pad = 14.0;
        const CGFloat view_h = contentView.bounds.size.height > 0 ? contentView.bounds.size.height : config.height;
        m_impl->hud_layer.frame = CGRectMake(pad, view_h - hud_h - pad, hud_w, hud_h);
        m_impl->hud_text_layer.frame = CGRectMake(12, 8, hud_w - 24, hud_h - 14);

        [m_impl->hud_layer addSublayer:m_impl->hud_text_layer];
        [m_impl->hud_layer setHidden:!m_impl->show_overlay];
        [m_impl->metal_layer addSublayer:m_impl->hud_layer];

        m_impl->last_stats_time = std::chrono::steady_clock::now();
        m_impl->last_presented_count = 0;

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

            const NSEventType eventType = [event type];

            // Close on Command+W (keyCode 13) or Command+Q (keyCode 12)
            if (eventType == NSEventTypeKeyDown) {
                const unsigned short code = [event keyCode];
                const NSEventModifierFlags flags = [event modifierFlags];
                if ((flags & NSEventModifierFlagCommand) && (code == 13 || code == 12)) {
                    close();
                    break;
                }
            }

            if (eventType == NSEventTypeKeyDown) {
                const unsigned short code = [event keyCode];
                // F11 (103) or Grave/Tilde (50) toggles stats HUD overlay
                if (code == 103 || code == 50) {
                    toggle_stats_overlay();
                }
                if (code < 512) {
                    m_impl->key_down[code] = true;
                }

                u32 pressed = 0;
                if (code == 49 || code == 36 || code == 6) pressed |= kPadCross;
                if (code == 51 || code == 8  || code == 7) pressed |= kPadCircle;
                if (code == 3  || code == 7  || code == 8) pressed |= kPadSquare;
                if (code == 14 || code == 15 || code == 8) pressed |= kPadTriangle;
                if (code == 12)               pressed |= kPadL1;
                if (code == 48)               pressed |= kPadTouchPadClick;
                if (code == 53)               pressed |= kPadStart;
                if (code == 126)              pressed |= kPadUp;
                if (code == 125)              pressed |= kPadDown;
                if (code == 123)              pressed |= kPadLeft;
                if (code == 124)              pressed |= kPadRight;
                m_impl->pressed_latch |= pressed;
            } else if (eventType == NSEventTypeKeyUp) {
                const unsigned short code = [event keyCode];
                if (code < 512) {
                    m_impl->key_down[code] = false;
                }
            } else if (eventType == NSEventTypeFlagsChanged) {
                const unsigned short code = [event keyCode];
                const NSEventModifierFlags flags = [event modifierFlags];
                if (code == 56) { // Left Shift -> kPadL1
                    const bool shift_down = (flags & NSEventModifierFlagShift) != 0;
                    m_impl->key_down[56] = shift_down;
                    if (shift_down) {
                        m_impl->pressed_latch |= kPadL1;
                    }
                }
            } else if (eventType == NSEventTypeMouseMoved ||
                       eventType == NSEventTypeLeftMouseDragged ||
                       eventType == NSEventTypeRightMouseDragged ||
                       eventType == NSEventTypeOtherMouseDragged) {
                m_impl->active_mouse_state.delta_x += static_cast<i32>([event deltaX]);
                m_impl->active_mouse_state.delta_y += static_cast<i32>([event deltaY]);
            } else if (eventType == NSEventTypeScrollWheel) {
                m_impl->active_mouse_state.wheel += static_cast<i32>([event scrollingDeltaY]);
            } else if (eventType == NSEventTypeLeftMouseDown) {
                m_impl->mouse_left_down = true;
                m_impl->active_pad_state.touch_down = true;
                m_impl->active_mouse_state.buttons |= 1;
                m_impl->pressed_latch |= (kPadCross | kPadTouchPadClick);
            } else if (eventType == NSEventTypeLeftMouseUp) {
                m_impl->mouse_left_down = false;
                m_impl->active_pad_state.touch_down = false;
                m_impl->active_mouse_state.buttons &= ~1;
            } else if (eventType == NSEventTypeRightMouseDown) {
                m_impl->mouse_right_down = true;
                m_impl->active_mouse_state.buttons |= 2;
                m_impl->pressed_latch |= kPadCircle;
            } else if (eventType == NSEventTypeRightMouseUp) {
                m_impl->mouse_right_down = false;
                m_impl->active_mouse_state.buttons &= ~2;
            } else if (eventType == NSEventTypeOtherMouseDown) {
                m_impl->active_mouse_state.buttons |= 4;
            } else if (eventType == NSEventTypeOtherMouseUp) {
                m_impl->active_mouse_state.buttons &= ~4;
            }

            // Map mouse coordinates in the window to capacitive touchpad bounds [0..1920, 0..943]
            if (eventType == NSEventTypeMouseMoved ||
                eventType == NSEventTypeLeftMouseDragged ||
                eventType == NSEventTypeRightMouseDragged ||
                eventType == NSEventTypeOtherMouseDragged ||
                eventType == NSEventTypeLeftMouseDown ||
                eventType == NSEventTypeRightMouseDown ||
                eventType == NSEventTypeOtherMouseDown) {
                if (m_impl->window) {
                    NSView* cv = [m_impl->window contentView];
                    if (cv) {
                        const NSRect bounds = cv.bounds;
                        if (bounds.size.width > 0 && bounds.size.height > 0) {
                            const NSPoint loc = [event locationInWindow];
                            const CGFloat rx = std::clamp(loc.x / bounds.size.width, 0.0, 1.0);
                            const CGFloat ry = std::clamp((bounds.size.height - loc.y) / bounds.size.height, 0.0, 1.0);
                            m_impl->active_pad_state.touch_x = static_cast<u16>(rx * 1920.0);
                            m_impl->active_pad_state.touch_y = static_cast<u16>(ry * 943.0);
                        }
                    }
                }
            }

            [NSApp sendEvent:event];
        }

        // Recompute sticks from WASD
        u8 lx = 128;
        if (m_impl->key_down[0] && !m_impl->key_down[2]) lx = 0;        // A (Left)
        else if (m_impl->key_down[2] && !m_impl->key_down[0]) lx = 255; // D (Right)

        u8 ly = 128;
        if (m_impl->key_down[13] && !m_impl->key_down[1]) ly = 0;       // W (Up)
        else if (m_impl->key_down[1] && !m_impl->key_down[13]) ly = 255;// S (Down)

        m_impl->active_pad_state.lx = lx;
        m_impl->active_pad_state.ly = ly;

        // Recompute buttons mask
        u32 btns = 0;
        if (m_impl->key_down[126]) btns |= kPadUp;
        if (m_impl->key_down[125]) btns |= kPadDown;
        if (m_impl->key_down[123]) btns |= kPadLeft;
        if (m_impl->key_down[124]) btns |= kPadRight;
        if (m_impl->key_down[49] || m_impl->key_down[36] || m_impl->key_down[6]) btns |= kPadCross;
        if (m_impl->key_down[51] || m_impl->key_down[8]  || m_impl->key_down[7])  btns |= kPadCircle;
        if (m_impl->key_down[3]  || m_impl->key_down[7]  || m_impl->key_down[8])  btns |= kPadSquare;
        if (m_impl->key_down[14] || m_impl->key_down[15] || m_impl->key_down[8])  btns |= kPadTriangle;
        if (m_impl->key_down[12] || m_impl->key_down[56]) btns |= kPadL1;
        if (m_impl->key_down[48]) btns |= kPadTouchPadClick;
        if (m_impl->key_down[53]) btns |= kPadStart;

        if (m_impl->mouse_left_down) {
            btns |= (kPadCross | kPadTouchPadClick);
        }
        if (m_impl->mouse_right_down) {
            btns |= kPadCircle;
        }

        m_impl->active_pad_state.buttons = btns;
        m_impl->buttons = btns;

        [NSApp updateWindows];
    }
}

u32 DisplayWindow::input_buttons() const noexcept {
    if (!m_impl) return 0;
    const u32 b = m_impl->buttons | m_impl->pressed_latch | gamepad_buttons();
    m_impl->pressed_latch = 0;
    return b;
}

PadState DisplayWindow::input_pad_state() const noexcept {
    if (!m_impl) return PadState{};
    PadState st = m_impl->active_pad_state;
    st.buttons |= m_impl->pressed_latch | gamepad_buttons();  // keyboard, mouse and any game controller
    m_impl->pressed_latch = 0;
    return st;
}

MouseState DisplayWindow::input_mouse_state() const noexcept {
    if (!m_impl) return MouseState{};
    MouseState ms = m_impl->active_mouse_state;
    m_impl->active_mouse_state.delta_x = 0;
    m_impl->active_mouse_state.delta_y = 0;
    m_impl->active_mouse_state.wheel = 0;
    return ms;
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

        // The producer and presenter use the same command queue. Metal
        // preserves queue ordering, so this presentation command buffer will
        // sample the framebuffer after the guest's submitted work without a
        // CPU-side wait. If no drawable is available, defer this frame.

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

        m_impl->update_stats_overlay();
        ++m_impl->presented_frames;
        poll_events();
        return true;
    }
}

void DisplayWindow::Impl::update_stats_overlay() {
    if (!is_open || config.headless || !window) return;

    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - last_stats_time).count();
    if (elapsed < 0.25) {
        return;
    }

    const u64 frames = presented_frames - last_presented_count;
    current_fps = elapsed > 0.0 ? (static_cast<double>(frames) / elapsed) : 0.0;
    current_frame_time_ms = current_fps > 0.001 ? (1000.0 / current_fps) : 0.0;
    last_presented_count = presented_frames;
    last_stats_time = now;

    // 1. RAM: Resident memory in MB
    struct mach_task_basic_info t_info;
    mach_msg_type_number_t t_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&t_info, &t_count) == KERN_SUCCESS) {
        current_ram_mb = static_cast<float>(t_info.resident_size) / (1024.0f * 1024.0f);
    }

    // 2. CPU: Cumulative CPU % across all active host threads
    thread_array_t thread_list;
    mach_msg_type_number_t thread_count;
    if (task_threads(mach_task_self(), &thread_list, &thread_count) == KERN_SUCCESS) {
        float cpu_sum = 0.0f;
        for (mach_msg_type_number_t i = 0; i < thread_count; ++i) {
            thread_basic_info_data_t tb_info;
            mach_msg_type_number_t tb_count = THREAD_BASIC_INFO_COUNT;
            if (thread_info(thread_list[i], THREAD_BASIC_INFO, (thread_info_t)&tb_info, &tb_count) == KERN_SUCCESS) {
                if (!(tb_info.flags & TH_FLAGS_IDLE)) {
                    cpu_sum += static_cast<float>(tb_info.cpu_usage) / static_cast<float>(TH_USAGE_SCALE) * 100.0f;
                }
            }
            mach_port_deallocate(mach_task_self(), thread_list[i]);
        }
        vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(thread_list), thread_count * sizeof(thread_act_t));
        current_cpu_pct = cpu_sum;
    }

    // 3. GPU: Last frame GPU time in ms from MetalBackend
    if (metal) {
        current_gpu_ms = metal->last_gpu_time_ms();
    }

    // 4. Update Window Title
    NSString* title_str = [NSString stringWithFormat:@"%s — %.1f FPS | CPU: %.1f%% | RAM: %.0f MB | GPU: %.1f ms",
                           config.title.c_str(), current_fps, current_cpu_pct, current_ram_mb, current_gpu_ms];
    [window setTitle:title_str];

    // 5. Update On-Screen HUD Overlay
    if (hud_layer && hud_text_layer && show_overlay) {
        NSView* contentView = [window contentView];
        if (contentView) {
            const CGSize viewSize = contentView.bounds.size;
            const CGFloat hud_w = 210.0;
            const CGFloat hud_h = 92.0;
            const CGFloat pad = 14.0;
            [CATransaction begin];
            [CATransaction setDisableActions:YES];
            hud_layer.frame = CGRectMake(pad, viewSize.height - hud_h - pad, hud_w, hud_h);
            hud_text_layer.frame = CGRectMake(12, 8, hud_w - 24, hud_h - 14);

            NSMutableAttributedString* attr = [[NSMutableAttributedString alloc] init];
            NSFont* font = [NSFont monospacedSystemFontOfSize:11.0 weight:NSFontWeightMedium];
            NSFont* hint_font = [NSFont monospacedSystemFontOfSize:9.0 weight:NSFontWeightRegular];
            NSDictionary* label_attr = @{
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: [NSColor colorWithCalibratedWhite:0.65 alpha:1.0]
            };
            NSDictionary* hint_attr = @{
                NSFontAttributeName: hint_font,
                NSForegroundColorAttributeName: [NSColor colorWithCalibratedWhite:0.45 alpha:1.0]
            };

            auto add_line = [&](NSString* label, NSString* val, NSColor* col) {
                [attr appendAttributedString:[[NSAttributedString alloc] initWithString:label attributes:label_attr]];
                [attr appendAttributedString:[[NSAttributedString alloc] initWithString:val attributes:@{
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName: col
                }]];
                [attr appendAttributedString:[[NSAttributedString alloc] initWithString:@"\n" attributes:label_attr]];
            };

            // FPS line: bright green (or amber/red if low)
            NSColor* fps_col = current_fps >= 45.0 ? [NSColor colorWithCalibratedRed:0.0 green:0.92 blue:0.45 alpha:1.0]
                             : current_fps >= 25.0 ? [NSColor colorWithCalibratedRed:1.0 green:0.80 blue:0.20 alpha:1.0]
                             : [NSColor colorWithCalibratedRed:1.0 green:0.35 blue:0.35 alpha:1.0];
            NSString* fps_val = [NSString stringWithFormat:@"%.1f (%.1f ms)", current_fps, current_frame_time_ms];
            add_line(@"FPS:  ", fps_val, fps_col);

            // CPU line: amber/yellow
            NSString* cpu_val = [NSString stringWithFormat:@"%.1f%%", current_cpu_pct];
            add_line(@"CPU:  ", cpu_val, [NSColor colorWithCalibratedRed:1.0 green:0.82 blue:0.35 alpha:1.0]);

            // RAM line: cyan/light blue
            NSString* ram_val = [NSString stringWithFormat:@"%.0f MB", current_ram_mb];
            add_line(@"RAM:  ", ram_val, [NSColor colorWithCalibratedRed:0.40 green:0.85 blue:1.0 alpha:1.0]);

            // GPU line: light purple/magenta
            NSString* gpu_val = [NSString stringWithFormat:@"%.1f ms", current_gpu_ms];
            add_line(@"GPU:  ", gpu_val, [NSColor colorWithCalibratedRed:0.88 green:0.50 blue:1.0 alpha:1.0]);

            // Hotkey hint
            [attr appendAttributedString:[[NSAttributedString alloc] initWithString:@"[F11: Toggle Overlay]" attributes:hint_attr]];

            hud_text_layer.string = attr;
            [CATransaction commit];
        }
    }
}

void DisplayWindow::close() {
    if (m_impl && m_impl->is_open) {
        m_impl->is_open = false;
        if (m_impl->window) {
            @autoreleasepool {
                if (m_impl->hud_layer) {
                    [m_impl->hud_layer removeFromSuperlayer];
                    m_impl->hud_layer = nil;
                    m_impl->hud_text_layer = nil;
                }
                [m_impl->window setDelegate:nil];
                [m_impl->window close];
                m_impl->window = nil;
            }
        }
    }
}

void DisplayWindow::toggle_stats_overlay() {
    if (!m_impl) return;
    m_impl->show_overlay = !m_impl->show_overlay;
    if (m_impl->hud_layer) {
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        [m_impl->hud_layer setHidden:!m_impl->show_overlay];
        [CATransaction commit];
    }
}

bool DisplayWindow::is_stats_overlay_visible() const noexcept {
    return m_impl && m_impl->show_overlay;
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
