#pragma once
#include "ps4/common.hpp"
namespace ps4 { class NidRegistry; }
namespace ps4::debug {
void scenario_start(NidRegistry& registry);
void scenario_stop();
bool scenario_enabled();
bool scenario_should_stop();
void scenario_shader(const char* stage, u64 hash);
void scenario_texture(u64 address);
void scenario_begin_frame(u64 frame);
void scenario_frame(u64 frame);
bool scenario_capture_requested();
void scenario_capture_completed(const char* path);
}
