#include <cassert>
#include <iostream>
#include "ps4/display_window.hpp"
#include "ps4/metal_backend.hpp"

int main() {
    std::cout << "[PHASE 8] Starting DisplayWindow & CAMetalLayer Presentation tests...\n";

    // Test 1: Default & custom WindowConfig
    {
        ps4::WindowConfig default_cfg{};
        assert(default_cfg.width == 1280);
        assert(default_cfg.height == 720);
        assert(default_cfg.resizable == true);
        assert(default_cfg.headless == false);
        assert(default_cfg.title == "PlayStation 4 Runtime - macOS");

        ps4::WindowConfig custom_cfg{
            .title = "Test Display",
            .width = 1920,
            .height = 1080,
            .resizable = false,
            .headless = true
        };
        assert(custom_cfg.width == 1920);
        assert(custom_cfg.height == 1080);
        assert(custom_cfg.resizable == false);
        assert(custom_cfg.headless == true);

        std::cout << "  ✓ Test 1: WindowConfig struct defaults and custom parameters verified\n";
    }

    // Test 2: Uninitialized MetalBackend is rejected
    {
        ps4::MetalBackend uninit_metal;
        ps4::DisplayWindow window;
        assert(!window.initialize(uninit_metal));
        assert(!window.is_open());

        std::cout << "  ✓ Test 2: Uninitialized MetalBackend rejected gracefully\n";
    }

    // Test 3: Headless DisplayWindow initialization & MSL Pipeline compilation
    ps4::MetalBackend metal;
    assert(metal.initialize());
    assert(metal.create_framebuffer(1920, 1080));

    {
        ps4::DisplayWindow window;
        ps4::WindowConfig cfg{
            .title = "Headless Test Window",
            .width = 1280,
            .height = 720,
            .resizable = true,
            .headless = true
        };

        assert(window.initialize(metal, cfg));
        assert(window.is_open());
        assert(window.width() == 1280);
        assert(window.height() == 720);
        assert(window.config().headless == true);

        std::cout << "  ✓ Test 3: Headless DisplayWindow initialized with compiled Metal pipeline\n";

        // Test 4: Headless Frame Presentation
        assert(metal.clear_screen({0.0f, 0.15f, 0.45f, 1.0f})); // PlayStation Deep Blue
        assert(window.present_frame());

        std::cout << "  ✓ Test 4: Headless frame presentation succeeded\n";

        // Test 5: Event polling
        window.poll_events();
        assert(window.is_open());

        std::cout << "  ✓ Test 5: Non-blocking event polling verified\n";

        // Test 6: Clean window closure
        window.close();
        assert(!window.is_open());
        assert(!window.present_frame());

        std::cout << "  ✓ Test 6: Clean window close and subsequent present rejection verified\n";
    }

    // Test 7: Native AppKit Window creation & CAMetalLayer setup
    {
        ps4::DisplayWindow window;
        ps4::WindowConfig cfg{
            .title = "PS4 Native Display Test",
            .width = 640,
            .height = 360,
            .resizable = true,
            .headless = false
        };

        assert(window.initialize(metal, cfg));
        assert(window.is_open());
        assert(window.width() == 640);
        assert(window.height() == 360);

        // Clear screen and present real Metal frame to the CAMetalLayer
        assert(metal.clear_screen({0.0f, 0.15f, 0.45f, 1.0f}));
        assert(window.present_frame());

        // Process any pending Cocoa window events
        window.poll_events();
        assert(window.is_open());

        window.close();
        assert(!window.is_open());

        std::cout << "  ✓ Test 7: Native AppKit NSWindow + CAMetalLayer created, presented frame, and closed cleanly\n";
    }

    std::cout << "[PHASE 8] PASS: All DisplayWindow & Presentation tests passed!\n";
    return 0;
}
