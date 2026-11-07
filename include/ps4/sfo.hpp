#pragma once
#include "ps4/common.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace ps4 {

enum class SfoFormat : u16 {
    Utf8Special = 0x0004,
    Utf8String  = 0x0204,
    UInt32      = 0x0404,
};

struct SfoEntry {
    std::string key;
    SfoFormat format{SfoFormat::Utf8String};
    std::string string_value;
    u32 int_value{0};
};

class SfoParser {
public:
    static bool parse(const std::vector<u8>& data, std::unordered_map<std::string, SfoEntry>& out_entries);
    static bool parse_file(const std::string& path, std::unordered_map<std::string, SfoEntry>& out_entries);
    static std::string get_string(const std::unordered_map<std::string, SfoEntry>& entries, const std::string& key, const std::string& default_val = "");
    static u32 get_uint32(const std::unordered_map<std::string, SfoEntry>& entries, const std::string& key, u32 default_val = 0);
};

} // namespace ps4
