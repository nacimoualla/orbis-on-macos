#pragma once

// Metal side of the GCN shader path: translated-pipeline cache, per-draw
// resource upload (guest memory arena, T# textures, S# samplers) and draw
// encoding. ObjC types are hidden behind void* so plain C++ can include this.

#include "ps4/common.hpp"
#include "ps4/pm4.hpp"
#include <memory>
#include <string>

namespace ps4 {

class MemoryManager;
class MetalBackend;

struct GcnDrawRequest {
    GuestAddr vs_addr{0};
    GuestAddr ps_addr{0};
    u32 vs_rsrc2{0};
    u32 ps_rsrc2{0};
    const u32* vs_user{nullptr};      // 16 VS user SGPRs
    const u32* ps_user{nullptr};      // 16 PS user SGPRs
    const u32* ctx_regs{nullptr};     // context register shadow (may be null)
    usize ctx_reg_count{0};
    PrimitiveType prim{PrimitiveType::TriangleList};

    bool indexed{false};
    u32 count{0};                     // vertex or index count
    GuestAddr index_addr{0};
    bool index32{false};
    i32 base_vertex{0};
    u32 instances{1};

    u32 depth_format{0};              // MTLPixelFormat of the pass depth attachment, 0 = none
    u32 color_formats[8]{};           // MTLPixelFormat of colour attachments 1-7 (0 = none);
                                      // attachment 0 is encode_draw's color_format
    u32 fb_width{0};                  // render target size; 0 leaves viewport/scissor alone
    u32 fb_height{0};
};

struct GcnPipelineStats {
    usize translated_draws{0};
    usize fallback_draws{0};
    usize pipelines_compiled{0};
    usize pipeline_failures{0};
    usize textures_uploaded{0};
    usize skipped_draws{0};           // empty scissor or front+back culling
};

class GcnPipelineCache {
public:
    GcnPipelineCache(void* mtl_device, MemoryManager* memory);
    ~GcnPipelineCache();
    GcnPipelineCache(const GcnPipelineCache&) = delete;
    GcnPipelineCache& operator=(const GcnPipelineCache&) = delete;

    void set_memory(MemoryManager* memory) noexcept;
    void set_metal_backend(MetalBackend* metal) noexcept;

    /// Encode a draw with the game's translated shaders. Returns false (and
    /// leaves the encoder state untouched apart from nothing) when anything
    /// is unsupported; the caller then uses the fixed fallback shader.
    /// `render_encoder` is an id<MTLRenderCommandEncoder>, `color_format` an
    /// MTLPixelFormat.
    bool encode_draw(void* render_encoder, u32 color_format, const GcnDrawRequest& req,
                     std::string* why = nullptr);

    [[nodiscard]] const GcnPipelineStats& stats() const noexcept;
    [[nodiscard]] static bool enabled_by_env() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ps4
