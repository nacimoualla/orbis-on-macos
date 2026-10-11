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

    /// Read back pixel value at (x, y) into 8-bit RGBA channels
    bool read_pixel(u32 x, u32 y, u8& out_r, u8& out_g, u8& out_b, u8& out_a) const;

    /// Raw Metal pointers for window presentation and GPU subsystems
    [[nodiscard]] void* raw_device() const noexcept;
    [[nodiscard]] void* raw_command_queue() const noexcept;
    [[nodiscard]] void* raw_framebuffer() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ps4
