#include "ps4/gpu_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ps4 {

namespace {

struct Rect {
    i64 x0, y0, x1, y1;
    void intersect(const Rect& o) {
        x0 = std::max(x0, o.x0);
        y0 = std::max(y0, o.y0);
        x1 = std::min(x1, o.x1);
        y1 = std::min(y1, o.y1);
    }
};

// TL/BR register pair (BR exclusive). Bit 31 of TL is WINDOW_OFFSET_DISABLE.
Rect scissor_rect(u32 tl, u32 br, i32 wx, i32 wy) {
    const bool offset = !(tl >> 31);
    Rect r{tl & 0x7FFF, (tl >> 16) & 0x7FFF, br & 0x7FFF, (br >> 16) & 0x7FFF};
    if (offset) {
        r.x0 += wx; r.x1 += wx;
        r.y0 += wy; r.y1 += wy;
    }
    return r;
}

StencilOp stencil_op(u32 v) { return static_cast<StencilOp>(v & 0xF); }
CompareFunc compare(u32 v) { return static_cast<CompareFunc>(v & 7); }

} // namespace

GpuRenderState decode_render_state(const u32* ctx, usize count, u32 fb_width, u32 fb_height) {
    auto reg = [&](u32 k) -> u32 { return (ctx && k < count) ? ctx[k] : 0u; };
    auto flt = [&](u32 k) -> f32 {
        const u32 v = reg(k);
        f32 f;
        std::memcpy(&f, &v, 4);
        return f;
    };
    GpuRenderState st;

    // Depth target.
    DepthTarget& dt = st.depth;
    dt.z_format = reg(ctx_reg::kDbZInfo) & 3;
    dt.has_depth = dt.z_format != 0;
    dt.has_stencil = (reg(ctx_reg::kDbStencilInfo) & 1) != 0;
    dt.base = static_cast<GuestAddr>(reg(ctx_reg::kDbZReadBase)) << 8;
    const u32 size = reg(ctx_reg::kDbDepthSize);
    dt.width = ((size & 0x7FF) + 1) * 8;
    dt.height = (((size >> 11) & 0x3FFFFF) + 1) * 8;

    // Depth/stencil test.
    DepthStencilState& ds = st.ds;
    const u32 dc = reg(ctx_reg::kDbDepthControl);
    ds.depth_test = dt.has_depth && (dc & 2);
    ds.depth_write = ds.depth_test && (dc & 4);
    ds.depth_func = compare(dc >> 4);
    ds.stencil_test = dt.has_stencil && (dc & 1);
    const u32 sc = reg(ctx_reg::kDbStencilControl);
    const u32 rm = reg(ctx_reg::kDbStencilRefMask);
    ds.front = StencilFace{compare(dc >> 8), stencil_op(sc), stencil_op(sc >> 4), stencil_op(sc >> 8),
                           static_cast<u8>(rm), static_cast<u8>(rm >> 8), static_cast<u8>(rm >> 16)};
    if (dc & 0x80) {   // BACKFACE_ENABLE
        const u32 rb = reg(ctx_reg::kDbStencilRefMaskBf);
        ds.back = StencilFace{compare(dc >> 20), stencil_op(sc >> 12), stencil_op(sc >> 16), stencil_op(sc >> 20),
                              static_cast<u8>(rb), static_cast<u8>(rb >> 8), static_cast<u8>(rb >> 16)};
    } else {
        ds.back = ds.front;
    }

    // Rasterizer.
    RasterState& rs = st.raster;
    const u32 mode = reg(ctx_reg::kPaSuScModeCntl);
    const bool cull_front = mode & 1, cull_back = mode & 2;
    rs.cull = cull_front && cull_back ? CullMode::Both
            : cull_front              ? CullMode::Front
            : cull_back               ? CullMode::Back
                                      : CullMode::None;
    rs.front_ccw = !(mode & 4);
    // POLY_OFFSET_FRONT_ENABLE / BACK_ENABLE. Metal has one bias for both
    // faces; the front one wins. Gnm programs the slope scale times 16.
    const u32 bias_reg = (mode & (1u << 11)) ? ctx_reg::kPaSuPolyOffsetFrontScale
                       : (mode & (1u << 12)) ? ctx_reg::kPaSuPolyOffsetFrontScale + 2
                                             : 0;
    if (bias_reg) {
        rs.depth_bias_slope = flt(bias_reg) / 16.0f;
        rs.depth_bias = flt(bias_reg + 1);
        rs.depth_bias_clamp = flt(ctx_reg::kPaSuPolyOffsetClamp);
    }
    const u32 clip = reg(ctx_reg::kPaClClipCntl);
    rs.depth_clamp = (clip >> 26) & 3;   // ZCLIP_NEAR_DISABLE / ZCLIP_FAR_DISABLE

    // Viewport. An unprogrammed one (scales 0) covers the framebuffer.
    RenderViewport& vp = st.viewport;
    const f32 xs = flt(ctx_reg::kPaClVportXscale), ys = flt(ctx_reg::kPaClVportXscale + 2);
    if (xs == 0.0f && ys == 0.0f) {
        vp = RenderViewport{0.0f, 0.0f, static_cast<f32>(fb_width), static_cast<f32>(fb_height), 0.0f, 1.0f, false};
    } else {
        u32 vte = reg(ctx_reg::kPaClVteCntl);
        if ((vte & 0x3F) == 0) vte |= 0x3F;   // never programmed: scale and offset enabled
        auto pick = [&](u32 bit, u32 k, f32 def) { return (vte & (1u << bit)) ? flt(k) : def; };
        const f32 sx = pick(0, ctx_reg::kPaClVportXscale, 1.0f);
        const f32 ox = pick(1, ctx_reg::kPaClVportXscale + 1, 0.0f);
        const f32 sy = pick(2, ctx_reg::kPaClVportXscale + 2, 1.0f);
        const f32 oy = pick(3, ctx_reg::kPaClVportXscale + 3, 0.0f);
        const f32 sz = pick(4, ctx_reg::kPaClVportXscale + 4, 1.0f);
        const f32 oz = pick(5, ctx_reg::kPaClVportXscale + 5, 0.0f);
        vp.x = ox - std::fabs(sx);
        vp.width = 2.0f * std::fabs(sx);
        vp.y = oy - std::fabs(sy);
        vp.height = 2.0f * std::fabs(sy);
        vp.flip_y = sy > 0.0f;
        // z_window = z * ZSCALE + ZOFFSET; Metal: znear + z * (zfar - znear).
        vp.znear = oz;
        vp.zfar = oz + sz;
    }
    f32 zmin = flt(ctx_reg::kPaScVportZmin0), zmax = flt(ctx_reg::kPaScVportZmax0);
    if (!(zmax > zmin)) { zmin = 0.0f; zmax = 1.0f; }
    zmin = std::clamp(zmin, 0.0f, 1.0f);
    zmax = std::clamp(zmax, zmin, 1.0f);
    vp.znear = std::clamp(vp.znear, zmin, zmax);
    vp.zfar = std::clamp(vp.zfar, zmin, zmax);

    // DB_RENDER_CONTROL clears: every covered pixel gets the clear value.
    const u32 rc = reg(ctx_reg::kDbRenderControl);
    ds.depth_clear = dt.has_depth && (rc & 1);
    ds.stencil_clear = dt.has_stencil && (rc & 2);
    if (ds.depth_clear) {
        ds.depth_test = true;
        ds.depth_write = true;
        ds.depth_func = CompareFunc::Always;
        const f32 z = std::clamp(flt(ctx_reg::kDbDepthClear), 0.0f, 1.0f);
        vp.znear = vp.zfar = z;
    }
    if (ds.stencil_clear) {
        ds.stencil_test = true;
        const StencilFace clear{CompareFunc::Always, StencilOp::ReplaceTest, StencilOp::ReplaceTest,
                                StencilOp::ReplaceTest, static_cast<u8>(reg(ctx_reg::kDbStencilClear)), 0xFF, 0xFF};
        ds.front = ds.back = clear;
    }

    // Scissor: framebuffer ∩ screen ∩ generic ∩ window ∩ viewport scissor.
    Rect r{0, 0, fb_width, fb_height};
    const u32 wo = reg(ctx_reg::kPaScWindowOffset);
    const i32 wx = static_cast<i16>(wo & 0xFFFF), wy = static_cast<i16>(wo >> 16);
    auto apply = [&](u32 tl_reg, bool window_offset) {
        const u32 tl = reg(tl_reg), br = reg(tl_reg + 1);
        if (br == 0) return;   // never programmed
        r.intersect(scissor_rect(window_offset ? tl : (tl | 0x80000000u), br, wx, wy));
    };
    apply(ctx_reg::kPaScScreenScissorTl, false);
    apply(ctx_reg::kPaScGenericScissorTl, true);
    apply(ctx_reg::kPaScWindowScissorTl, true);
    if (reg(ctx_reg::kPaScModeCntl0) & 2) apply(ctx_reg::kPaScVportScissor0Tl, true);   // VPORT_SCISSOR_ENABLE
    if (r.x1 <= r.x0 || r.y1 <= r.y0) {
        st.skip_draw = true;
        st.scissor = RenderScissor{};
    } else {
        st.scissor = RenderScissor{static_cast<u32>(r.x0), static_cast<u32>(r.y0), static_cast<u32>(r.x1 - r.x0),
                                   static_cast<u32>(r.y1 - r.y0)};
    }
    if (rs.cull == CullMode::Both) st.skip_draw = true;
    return st;
}

} // namespace ps4
