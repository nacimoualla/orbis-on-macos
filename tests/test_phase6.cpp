#include <cassert>
#include <cmath>
#include <iostream>
#include "ps4/metal_backend.hpp"

namespace {

bool color_near(ps4::u8 actual, ps4::f32 expected_norm) {
    int expected_byte = static_cast<int>(std::round(expected_norm * 255.0f));
    return std::abs(static_cast<int>(actual) - expected_byte) <= 2;
}

} // namespace

int main() {
    std::cout << "[PHASE 6] Starting Apple Silicon Metal Backend unit tests...\n";

    ps4::MetalBackend metal;

    // Test 1: Initialize device and command queue
    {
        assert(!metal.is_initialized());
        bool ok = metal.initialize();
        assert(ok);
        assert(metal.is_initialized());
        std::string name = metal.device_name();
        assert(!name.empty());
        std::cout << "  ✓ Test 1: Apple Silicon Metal device initialized: " << name << "\n";
    }

    // Test 2: Create 1080p offscreen framebuffer
    {
        assert(metal.create_framebuffer(1920, 1080));
        assert(metal.width() == 1920);
        assert(metal.height() == 1080);
        std::cout << "  ✓ Test 2: 1920x1080 BGRA8Unorm framebuffer created in unified memory\n";
    }

    // Test 3: Clear screen with Cornflower Blue and verify GPU render pass
    {
        // Cornflower Blue: R=0.392, G=0.584, B=0.929, A=1.0
        const ps4::ColorRGBA cornflower_blue{0.392f, 0.584f, 0.929f, 1.0f};
        assert(metal.clear_screen(cornflower_blue));

        ps4::u8 r = 0, g = 0, b = 0, a = 0;
        // Verify center pixel
        assert(metal.read_pixel(960, 540, r, g, b, a));
        assert(color_near(r, cornflower_blue.r));
        assert(color_near(g, cornflower_blue.g));
        assert(color_near(b, cornflower_blue.b));
        assert(a == 255);

        // Verify corner pixels
        assert(metal.read_pixel(0, 0, r, g, b, a));
        assert(color_near(r, cornflower_blue.r));
        assert(metal.read_pixel(1919, 1079, r, g, b, a));
        assert(color_near(r, cornflower_blue.r));

        std::cout << "  ✓ Test 3: Clear-screen render pass verified with pixel readback\n";
    }

    // Test 4: Clear to Solid Red (1.0, 0.0, 0.0, 1.0)
    {
        const ps4::ColorRGBA pure_red{1.0f, 0.0f, 0.0f, 1.0f};
        assert(metal.clear_screen(pure_red));

        ps4::u8 r = 0, g = 0, b = 0, a = 0;
        assert(metal.read_pixel(500, 500, r, g, b, a));
        assert(r == 255);
        assert(g == 0);
        assert(b == 0);
        assert(a == 255);

        std::cout << "  ✓ Test 4: Solid Red clear pass verified\n";
    }

    std::cout << "[PHASE 6] PASS: All Apple Metal Framebuffer tests passed!\n";
    return 0;
}
