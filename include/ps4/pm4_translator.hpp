#pragma once

#include "ps4/common.hpp"
#include "ps4/gpu_ring_buffer.hpp"
#include "ps4/metal_backend.hpp"
#include "ps4/pm4.hpp"
#include <memory>

namespace ps4 {

class MemoryManager;

struct GpuViewport {
    f32 x{0.0f};
    f32 y{0.0f};
    f32 width{1920.0f};
    f32 height{1080.0f};
    f32 min_depth{0.0f};
    f32 max_depth{1.0f};
};

struct GpuScissor {
    u32 x{0};
    u32 y{0};
    u32 width{1920};
    u32 height{1080};
};

struct GpuFrameStats {
    u64 frame_index{0};
    usize packets{0};
    usize draws{0};
    usize indexed_draws{0};
    usize unsupported_packets{0};
    usize upload_bytes{0};
    usize resource_cache_hits{0};
    usize resource_cache_misses{0};
    u64 command_buffers_submitted{0};
};

class Pm4Translator {
public:
    explicit Pm4Translator(MetalBackend* metal = nullptr);
    ~Pm4Translator();

    // Moveable
    Pm4Translator(Pm4Translator&&) noexcept;
    Pm4Translator& operator=(Pm4Translator&&) noexcept;

    // Non-copyable
    Pm4Translator(const Pm4Translator&) = delete;
    Pm4Translator& operator=(const Pm4Translator&) = delete;

    bool begin_frame();
    bool submit_frame(bool wait_for_completion = false);
    [[nodiscard]] bool frame_in_progress() const noexcept;
    [[nodiscard]] GpuFrameStats frame_stats() const noexcept;

    /// Execute a single PM4 packet
    bool execute_packet(const Pm4Packet& packet);

    /// Execute raw command buffer at guest virtual address
    usize execute_command_buffer(GuestAddr dcb_gpu_addr, usize size_bytes);

    /// Drain and execute all pending packets in ring buffer
    usize execute_ring_buffer(GpuRingBuffer& ring_buffer);

    /// Reset GPU context state to defaults
    void reset_state();

    /// GPU context state accessors
    [[nodiscard]] const GpuViewport& viewport() const noexcept;
    [[nodiscard]] const GpuScissor& scissor() const noexcept;
    [[nodiscard]] ColorRGBA clear_color() const noexcept;

    /// Execution statistics
    [[nodiscard]] usize packets_processed() const noexcept;
    [[nodiscard]] usize draw_count() const noexcept;
    [[nodiscard]] usize indexed_draw_count() const noexcept;
    [[nodiscard]] usize context_reg_writes() const noexcept;

    /// 3D Mesh pipeline: memory manager attachment and vertex buffer binding
    void set_memory_manager(MemoryManager* memory) noexcept;
    void bind_vertex_buffer(GuestAddr vaddr, usize size, usize stride = sizeof(Vertex3D));
    [[nodiscard]] GuestAddr vertex_buffer_address() const noexcept;
    [[nodiscard]] usize vertex_buffer_size() const noexcept;

    /// Index buffer and indirect draw state
    [[nodiscard]] GuestAddr index_base_address() const noexcept;
    [[nodiscard]] u32 index_buffer_size() const noexcept;
    [[nodiscard]] bool is_32bit_index() const noexcept;
    [[nodiscard]] GuestAddr indirect_base_address() const noexcept;
    [[nodiscard]] GuestAddr last_dma_dst_address() const noexcept;
    [[nodiscard]] usize last_dma_size() const noexcept;

    void set_index_base_address(GuestAddr addr) noexcept;
    void set_indirect_base_address(GuestAddr addr) noexcept;
    void set_index_32bit(bool is_32bit) noexcept;

    /// Direct 3D indexed draw submission
    bool execute_indexed_draw(GuestAddr index_buffer_addr, u32 index_count, PrimitiveType prim_type = PrimitiveType::TriangleList);

    /// Direct 3D auto draw submission
    bool execute_auto_draw(u32 vtx_count, PrimitiveType prim_type = PrimitiveType::TriangleList, i32 base_vertex = 0, u32 instance_count = 1);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ps4
