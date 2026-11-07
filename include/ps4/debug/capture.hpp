#pragma once
#include "ps4/gcn_shader.hpp"
#include <functional>
namespace ps4 { class MemoryManager; }

namespace ps4::debug {
bool capture_active();
void capture_begin(u64 frame, void* device);
void capture_operation(const char* kind, u32 count, const u32* regs, MemoryManager* memory,
                       u32 width, u32 height, u32 x = 0, u32 y = 0, u32 z = 0);
void capture_resources(const char* stage, u64 hash, const gcn::ResourceScan& scan,
                       const std::vector<gcn::ImageSlot>& slots);
void capture_render_pass(void* descriptor, const u64* keys);
void capture_bound();
void capture_pass(void* command_buffer, const char* kind);
void capture_texture(u64 key, void* texture);
void capture_end();
void at_frame(u64 frame);
void set_frame_dump(std::function<void()> callback);
}
