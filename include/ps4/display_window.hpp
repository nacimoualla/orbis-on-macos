#pragma once

#include "ps4/common.hpp"
#include "ps4/metal_backend.hpp"
#include <memory>
#include <string>

namespace ps4 {

struct WindowConfig {
    std::string title{"PlayStation 4 Runtime - macOS"};
    u32 width{1280};
    u32 height{720};
    bool resizable{true};
    bool headless{false}; // When true, runs offscreen/headless for automated testing
};

class DisplayWindow {
public:
    DisplayWindow();
    ~DisplayWindow();

    // Non-copyable
    DisplayWindow(const DisplayWindow&) = delete;
    DisplayWindow& operator=(const DisplayWindow&) = delete;

    // Moveable
    DisplayWindow(DisplayWindow&&) noexcept;
    DisplayWindow& operator=(DisplayWindow&&) noexcept;

    /// Initialize macOS window, CAMetalLayer, and Metal presentation pipeline
    bool initialize(MetalBackend& metal, const WindowConfig& config = {});

    /// Check if the window is currently initialized and open
    [[nodiscard]] bool is_open() const noexcept;

    /// Pump and dispatch native AppKit events non-blockingly
    void poll_events();

    /// Render and present the current MetalBackend framebuffer to the display
    bool present_frame();

    /// Request closing the window
    void close();

    /// Get current controller/keyboard buttons bitmask (Bit 14 = Cross/Space/Click)
    [[nodiscard]] u32 input_buttons() const noexcept;

    /// Window content width
    [[nodiscard]] u32 width() const noexcept;

    /// Window content height
    [[nodiscard]] u32 height() const noexcept;

    /// Configuration used to initialize the window
    [[nodiscard]] const WindowConfig& config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ps4
