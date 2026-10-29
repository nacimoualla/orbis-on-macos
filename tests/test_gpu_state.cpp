// Phase 20: fixed-function 3D state (depth/stencil, rasterizer, viewport,
// scissor) decoded from the context-register shadow. The base register set is
// what Flappy Bird (Unity) programs, captured with PS4_LOG_PM4=1.

#include "ps4/gpu_state.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ps4;

namespace {

int g_failures = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_failures;                                                         \
        }                                                                         \
    } while (0)

u32 fbits(f32 f) {
    u32 u;
    std::memcpy(&u, &f, 4);
    return u;
}

bool near(f32 a, f32 b) { return std::fabs(a - b) < 1e-4f; }

std::vector<u32> unity_regs() {
    std::vector<u32> c(ctx_reg::kCount, 0);
    auto set = [&](u32 base, std::initializer_list<u32> vals) {
        for (u32 v : vals) c[base++] = v;
    };
    set(0x000C, {0x00000000, 0x04380780});
    set(0x0010, {0xa0000003, 0x20000001, 0x0160ff00, 0x01618f00, 0x0160ff00, 0x01618f00, 0x000478ff, 0x00008fff});
    set(0x0090, {0x80000000, 0x3fff3fff});
    set(0x0094, {0x80000000, 0x0fff0fff});
    set(0x00B4, {0x00000000, 0x3f800000});
    set(0x010B, {0x00000333, 0xffffff00, 0xffffff00});
    set(0x010F, {fbits(960), fbits(960), fbits(-540), fbits(540), fbits(1), fbits(0)});
    set(0x0200, {0x00000062});
    set(0x0204, {0x00080000, 0x00000244, 0x0000043f});
    set(0x0292, {0x00000002});
    return c;
}

void test_unity_draw() {
    const std::vector<u32> c = unity_regs();
    const GpuRenderState s = decode_render_state(c.data(), c.size(), 1920, 1080);
    CHECK(s.depth.has_depth && s.depth.has_stencil && s.depth.z_format == 3);
    CHECK(s.depth.base == 0x160ff0000ull);
    CHECK(s.depth.width == 2048 && s.depth.height == 1152);
    // Reversed Z: test GEQUAL, no writes, stencil off.
    CHECK(s.ds.depth_test && !s.ds.depth_write && s.ds.depth_func == CompareFunc::GreaterEqual);
    CHECK(!s.ds.stencil_test && !s.ds.depth_clear);
    CHECK(near(s.viewport.x, 0) && near(s.viewport.y, 0) && near(s.viewport.width, 1920) &&
          near(s.viewport.height, 1080));
    CHECK(near(s.viewport.znear, 0) && near(s.viewport.zfar, 1) && !s.viewport.flip_y);
    CHECK(s.scissor.x == 0 && s.scissor.y == 0 && s.scissor.width == 1920 && s.scissor.height == 1080);
    CHECK(s.raster.cull == CullMode::None && !s.raster.front_ccw && !s.raster.depth_clamp);
    CHECK(s.raster.depth_bias == 0 && s.raster.depth_bias_slope == 0);
    CHECK(!s.skip_draw);
}

void test_unity_clear() {
    std::vector<u32> c = unity_regs();
    c[ctx_reg::kDbRenderControl] = 1;   // DEPTH_CLEAR_ENABLE
    c[ctx_reg::kDbDepthControl] = 0x776;
    c[ctx_reg::kDbDepthClear] = fbits(0.25f);
    GpuRenderState s = decode_render_state(c.data(), c.size(), 1920, 1080);
    CHECK(s.ds.depth_clear && s.ds.depth_test && s.ds.depth_write && s.ds.depth_func == CompareFunc::Always);
    CHECK(near(s.viewport.znear, 0.25f) && near(s.viewport.zfar, 0.25f));
    CHECK(!s.ds.stencil_clear);

    c[ctx_reg::kDbRenderControl] = 3;   // + STENCIL_CLEAR_ENABLE
    c[ctx_reg::kDbStencilClear] = 0x5A;
    s = decode_render_state(c.data(), c.size(), 1920, 1080);
    CHECK(s.ds.stencil_clear && s.ds.stencil_test);
    CHECK(s.ds.front.func == CompareFunc::Always && s.ds.front.zpass == StencilOp::ReplaceTest &&
          s.ds.front.ref == 0x5A && s.ds.back.ref == 0x5A && s.ds.front.write_mask == 0xFF);

    // Without a depth surface the clear bit does nothing.
    c[ctx_reg::kDbZInfo] = 0;
    c[ctx_reg::kDbStencilInfo] = 0;
    s = decode_render_state(c.data(), c.size(), 1920, 1080);
    CHECK(!s.ds.depth_clear && !s.ds.depth_test && !s.ds.stencil_test);
}

void test_raster_and_stencil() {
    std::vector<u32> c = unity_regs();
    // Cull back, CCW front, front polygon offset (slope 2.0 programmed x16).
    c[ctx_reg::kPaSuScModeCntl] = 0x2 | (1u << 11);
    c[ctx_reg::kPaSuPolyOffsetFrontScale] = fbits(32.0f);
    c[ctx_reg::kPaSuPolyOffsetFrontScale + 1] = fbits(4.0f);
    c[ctx_reg::kPaSuPolyOffsetClamp] = fbits(0.01f);
    c[ctx_reg::kPaClClipCntl] |= 3u << 26;
    // Stencil: front EQUAL ref 7 mask 0x0F, back (BACKFACE_ENABLE) NOTEQUAL ref 9.
    c[ctx_reg::kDbDepthControl] = 0x1 | 0x2 | 0x4 | (1u << 4) | 0x80 | (2u << 8) | (5u << 20);
    c[ctx_reg::kDbStencilControl] = 0x1 | (0x5u << 4) | (0x7u << 8) | (0x8u << 12);
    c[ctx_reg::kDbStencilRefMask] = 0x07 | (0x0F << 8) | (0xF0 << 16);
    c[ctx_reg::kDbStencilRefMaskBf] = 0x09 | (0xFF << 8) | (0xFF << 16);
    const GpuRenderState s = decode_render_state(c.data(), c.size(), 1920, 1080);
    CHECK(s.raster.cull == CullMode::Back && s.raster.front_ccw);
    CHECK(near(s.raster.depth_bias_slope, 2.0f) && near(s.raster.depth_bias, 4.0f) &&
          near(s.raster.depth_bias_clamp, 0.01f));
    CHECK(s.raster.depth_clamp);
    CHECK(s.ds.depth_test && s.ds.depth_write && s.ds.depth_func == CompareFunc::Less);
    CHECK(s.ds.stencil_test);
    CHECK(s.ds.front.func == CompareFunc::Equal && s.ds.front.fail == StencilOp::Zero &&
          s.ds.front.zpass == StencilOp::AddClamp && s.ds.front.zfail == StencilOp::Invert);
    CHECK(s.ds.front.ref == 7 && s.ds.front.read_mask == 0x0F && s.ds.front.write_mask == 0xF0);
    CHECK(s.ds.back.func == CompareFunc::NotEqual && s.ds.back.fail == StencilOp::AddWrap && s.ds.back.ref == 9);

    c[ctx_reg::kPaSuScModeCntl] = 0x3;
    CHECK(decode_render_state(c.data(), c.size(), 1920, 1080).skip_draw);
}

void test_viewport_and_scissor() {
    std::vector<u32> c = unity_regs();
    // Half-size viewport in the lower right, Y up (positive YSCALE), z [0.2, 0.6].
    c[ctx_reg::kPaClVportXscale + 0] = fbits(480);
    c[ctx_reg::kPaClVportXscale + 1] = fbits(1440);
    c[ctx_reg::kPaClVportXscale + 2] = fbits(270);
    c[ctx_reg::kPaClVportXscale + 3] = fbits(810);
    c[ctx_reg::kPaClVportXscale + 4] = fbits(0.4f);
    c[ctx_reg::kPaClVportXscale + 5] = fbits(0.2f);
    // Generic scissor with window offset (100, 50).
    c[ctx_reg::kPaScWindowOffset] = 100 | (50u << 16);
    c[ctx_reg::kPaScGenericScissorTl] = 10 | (20u << 16);
    c[ctx_reg::kPaScGenericScissorBr] = 500 | (300u << 16);
    GpuRenderState s = decode_render_state(c.data(), c.size(), 1920, 1080);
    CHECK(near(s.viewport.x, 960) && near(s.viewport.y, 540) && near(s.viewport.width, 960) &&
          near(s.viewport.height, 540));
    CHECK(s.viewport.flip_y && near(s.viewport.znear, 0.2f) && near(s.viewport.zfar, 0.6f));
    // The viewport scissor (0x0fff0fff) and screen scissor don't cut further.
    CHECK(s.scissor.x == 110 && s.scissor.y == 70 && s.scissor.width == 490 && s.scissor.height == 280);

    // Disjoint screen scissor: nothing to draw.
    c[ctx_reg::kPaScScreenScissorTl] = 1000 | (1000u << 16);
    CHECK(decode_render_state(c.data(), c.size(), 1920, 1080).skip_draw);

    // Nothing programmed: full framebuffer, depth range [0, 1], no depth.
    const std::vector<u32> zero(ctx_reg::kCount, 0);
    s = decode_render_state(zero.data(), zero.size(), 64, 32);
    CHECK(near(s.viewport.width, 64) && near(s.viewport.height, 32) && near(s.viewport.zfar, 1));
    CHECK(s.scissor.width == 64 && s.scissor.height == 32 && !s.skip_draw);
    CHECK(!s.depth.has_depth && !s.ds.depth_test && s.raster.cull == CullMode::None);
    s = decode_render_state(nullptr, 0, 64, 32);
    CHECK(s.scissor.width == 64 && !s.skip_draw);
}

} // namespace

int main() {
    test_unity_draw();
    test_unity_clear();
    test_raster_and_stencil();
    test_viewport_and_scissor();
    if (g_failures) {
        std::fprintf(stderr, "GPU state test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("GPU state test: all checks passed\n");
    return 0;
}
