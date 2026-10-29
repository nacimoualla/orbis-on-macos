#pragma once

// Fixed-function 3D state of a GCN draw (depth/stencil, rasterizer,
// viewport, scissor), decoded from the context-register shadow. Plain C++ so
// it can be tested without Metal; gcn_pipeline.mm maps it onto Metal state.

#include "ps4/common.hpp"
#include <array>

namespace ps4 {

// Context register offsets (dwords relative to kPm4ContextRegBase, CIK).
namespace ctx_reg {
inline constexpr u32 kDbRenderControl = 0x000;
inline constexpr u32 kDbStencilClear = 0x00A;
inline constexpr u32 kDbDepthClear = 0x00B;
inline constexpr u32 kPaScScreenScissorTl = 0x00C;
inline constexpr u32 kPaScScreenScissorBr = 0x00D;
inline constexpr u32 kDbZInfo = 0x010;
inline constexpr u32 kDbStencilInfo = 0x011;
inline constexpr u32 kDbZReadBase = 0x012;
inline constexpr u32 kDbDepthSize = 0x016;
inline constexpr u32 kPaScWindowOffset = 0x080;
inline constexpr u32 kPaScWindowScissorTl = 0x081;
inline constexpr u32 kPaScWindowScissorBr = 0x082;
inline constexpr u32 kCbTargetMask = 0x08E;
inline constexpr u32 kPaScGenericScissorTl = 0x090;
inline constexpr u32 kPaScGenericScissorBr = 0x091;
inline constexpr u32 kPaScVportScissor0Tl = 0x094;
inline constexpr u32 kPaScVportScissor0Br = 0x095;
inline constexpr u32 kPaScVportZmin0 = 0x0B4;
inline constexpr u32 kPaScVportZmax0 = 0x0B5;
inline constexpr u32 kDbStencilControl = 0x10B;
inline constexpr u32 kDbStencilRefMask = 0x10C;
inline constexpr u32 kDbStencilRefMaskBf = 0x10D;
inline constexpr u32 kPaClVportXscale = 0x10F;   // XSCALE, XOFFSET, YSCALE, YOFFSET, ZSCALE, ZOFFSET
inline constexpr u32 kSpiPsInputEna = 0x1B3;
inline constexpr u32 kSpiPsInputAddr = 0x1B4;
inline constexpr u32 kDbDepthControl = 0x200;
inline constexpr u32 kDbShaderControl = 0x203;
inline constexpr u32 kPaClClipCntl = 0x204;
inline constexpr u32 kPaSuScModeCntl = 0x205;
inline constexpr u32 kPaClVteCntl = 0x206;
inline constexpr u32 kPaScModeCntl0 = 0x292;
inline constexpr u32 kPaSuPolyOffsetClamp = 0x2DF;
inline constexpr u32 kPaSuPolyOffsetFrontScale = 0x2E0;   // FRONT_SCALE, FRONT_OFFSET, BACK_SCALE, BACK_OFFSET
inline constexpr u32 kCbBlend0Control = 0x1E0;  // + slot
inline constexpr u32 kCbColor0BaseGfx7 = 0x318;   // CB_COLOR0_BASE; 15 registers per slot
inline constexpr u32 kCbColorStrideGfx7 = 15;
inline constexpr u32 kMaxColorTargets = 8;
inline constexpr u32 kCount = 0x400;   // registers decode_render_state may read
} // namespace ctx_reg

// GCN compare function (DB_DEPTH_CONTROL ZFUNC / STENCILFUNC).
enum class CompareFunc : u8 { Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always };

// DB_STENCIL_CONTROL operation.
enum class StencilOp : u8 {
    Keep, Zero, Ones, ReplaceTest, ReplaceOp, AddClamp, SubClamp, Invert,
    AddWrap, SubWrap, And, Or, Xor, Nand, Nor, Xnor
};

struct StencilFace {
    CompareFunc func{CompareFunc::Always};
    StencilOp fail{StencilOp::Keep};
    StencilOp zpass{StencilOp::Keep};
    StencilOp zfail{StencilOp::Keep};
    u8 ref{0};
    u8 read_mask{0xFF};
    u8 write_mask{0xFF};
};

struct DepthStencilState {
    bool depth_test{false};
    bool depth_write{false};
    CompareFunc depth_func{CompareFunc::Always};
    bool stencil_test{false};
    StencilFace front{};
    StencilFace back{};
    // DB_RENDER_CONTROL depth/stencil clear: the draw writes the clear
    // values instead of its own depth (folded into the state above and the
    // viewport depth range).
    bool depth_clear{false};
    bool stencil_clear{false};
};

struct DepthTarget {
    bool has_depth{false};     // DB_Z_INFO.FORMAT != INVALID
    bool has_stencil{false};   // DB_STENCIL_INFO.FORMAT != INVALID
    u32 z_format{0};           // 1 = Z16, 3 = Z32_FLOAT
    GuestAddr base{0};         // DB_Z_READ_BASE << 8
    u32 width{0};
    u32 height{0};
};

struct ColorTarget {
    bool enabled{false};  // base, format and a write mask are all set
    GuestAddr base{0};    // CB_COLOR_BASE << 8
    u32 width{0};         // rendered size: the tile-padded surface (pitch x
    u32 height{0};        // slice / pitch) clipped by the screen scissor
    u32 pitch{0};         // pixels per row
    u32 format{0};        // CB_COLOR_INFO.FORMAT (bits 2-6): 10 = 8_8_8_8, 12 = 16_16_16_16, ...
    u32 num_format{0};    // NUMBER_TYPE (bits 8-10): 0 UNORM, 1 SNORM, 4 UINT, 5 SINT, 6 SRGB, 7 FLOAT
    u32 comp_swap{0};     // COMP_SWAP (bits 11-12): 0 STD (RGBA), 1 ALT (BGRA), 2 STD_REV, 3 ALT_REV
    u8 write_mask{0};     // CB_TARGET_MASK nibble (R, G, B, A)
};

enum class CullMode : u8 { None, Front, Back, Both };

struct RasterState {
    CullMode cull{CullMode::None};
    bool front_ccw{true};          // PA_SU_SC_MODE_CNTL.FACE == 0
    f32 depth_bias{0.0f};          // constant offset (depth units)
    f32 depth_bias_slope{0.0f};
    f32 depth_bias_clamp{0.0f};
    bool depth_clamp{false};       // z clipping disabled (near and far)
};

struct RenderViewport {
    f32 x{0.0f}, y{0.0f}, width{0.0f}, height{0.0f};
    f32 znear{0.0f}, zfar{1.0f};
    bool flip_y{false};            // YSCALE > 0: the VS must negate y
};

struct RenderScissor {
    u32 x{0}, y{0}, width{0}, height{0};
};

struct GpuRenderState {
    std::array<ColorTarget, ctx_reg::kMaxColorTargets> color{};
    DepthTarget depth{};
    DepthStencilState ds{};
    RasterState raster{};
    RenderViewport viewport{};
    RenderScissor scissor{};
    // Nothing would be rasterised (empty scissor or front+back culling).
    bool skip_draw{false};
};

/// Decode the draw state from the context-register shadow `ctx` (indexed by
/// context offset; `count` registers valid). `fb_width`/`fb_height` bound the
/// viewport and scissor; an unprogrammed viewport covers the framebuffer.
GpuRenderState decode_render_state(const u32* ctx, usize count, u32 fb_width, u32 fb_height);

} // namespace ps4
