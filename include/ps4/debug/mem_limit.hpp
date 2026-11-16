#pragma once
#include <cstdint>
namespace ps4::debug {
/// Start a background thread that exits the process when its physical memory
/// footprint passes a limit: PS4_MEM_LIMIT_MB if set (0 disables), otherwise
/// half of the machine's physical memory. Returns the limit in bytes (0 = off).
std::uint64_t start_memory_limit();
/// Current physical footprint of this process in bytes (0 if unavailable).
std::uint64_t process_footprint();
}
