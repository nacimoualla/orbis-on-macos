#pragma once

#include "ps4/common.hpp"
#include <memory>
#include <string>

namespace ps4 {

struct ColorRGBA {
    f32 r{0.0f};
    f32 g{0.0f};
    f32 b{0.0f};
    f32 a{1.0f};
};

class MetalBackend {
public:
    MetalBackend();
    ~MetalBackend();

    // Move semantics enabled
    MetalBackend(MetalBackend&&) noexcept;
    MetalBackend& operator=(MetalBackend&&) noexcept;

    // Non-copyable (owns Metal resources)
    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    /// Initialize Apple Silicon Metal device and command queue
    bool initialize();

    /// Check if Metal device is initialized
    [[nodiscard]] bool is_initialized() const noexcept;

    /// GPU name (e.g. "Apple M1", "Apple M2 Max", etc.)
    [[nodiscard]] std::string device_name() const;

    /// Create 2D offscreen BGRA8Unorm framebuffer
    bool create_framebuffer(u32 width = 1920, u32 height = 1080);

    /// Framebuffer dimensions
    [[nodiscard]] u32 width() const noexcept;
    [[nodiscard]] u32 height() const noexcept;

    /// Submit clear-screen render pass and wait for completion
    bool clear_screen(ColorRGBA color);

    /// Begin recording a frame for deferred submission
    bool begin_frame();

    /// Encode a clear operation into the active frame
    bool encode_clear(ColorRGBA color);

    /// Submit the active frame, optionally waiting for completion
    bool submit_frame(bool wait_for_completion);
    /// Ensure the active frame has a render encoder, clearing only on its first pass.
    bool ensure_render_encoder(bool clear, ColorRGBA color);
    /// Return the active render encoder as an opaque Objective-C pointer.
    [[nodiscard]] void* active_render_encoder() const noexcept;
    /// End the active render encoder without submitting the frame.
    void end_render_encoder();

    /// Wait for the most recently submitted frame when it is incomplete
    bool wait_for_last_frame();

    /// Whether a frame is currently being recorded
    [[nodiscard]] bool frame_in_progress() const noexcept;

    /// Number of frames submitted since initialization
    [[nodiscard]] u64 submitted_frame_count() const noexcept;

    /// Read back pixel value at (x, y) into 8-bit RGBA channels
    bool read_pixel(u32 x, u32 y, u8& out_r, u8& out_g, u8& out_b, u8& out_a) const;

    /// Upload CPU/guest pixel buffer (BGRA or RGBA) directly into Metal framebuffer texture
    bool upload_pixels(const void* pixel_data, usize bytes_per_row);

    /// Raw Metal pointers for window presentation and GPU subsystems
    [[nodiscard]] void* raw_device() const noexcept;
    [[nodiscard]] void* raw_command_queue() const noexcept;
    [[nodiscard]] void* raw_framebuffer() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ps4
