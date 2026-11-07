#include "ps4/sfo.hpp"
#include <cstdio>
#include <cstring>

namespace ps4 {

bool SfoParser::parse(const std::vector<u8>& data, std::unordered_map<std::string, SfoEntry>& out_entries) {
    out_entries.clear();
    if (data.size() < 20) return false;

    // Check magic: \0PSF (0x00, 'P', 'S', 'F')
    if (data[0] != 0x00 || data[1] != 'P' || data[2] != 'S' || data[3] != 'F') {
        return false;
    }

    u32 key_table_off = 0;
    u32 data_table_off = 0;
    u32 entries_count = 0;
    std::memcpy(&key_table_off, &data[8], 4);
    std::memcpy(&data_table_off, &data[12], 4);
    std::memcpy(&entries_count, &data[16], 4);

    const size_t entries_header_size = 20 + static_cast<size_t>(entries_count) * 16;
    if (data.size() < entries_header_size) return false;
    if (key_table_off > data.size() || data_table_off > data.size()) return false;

    for (u32 i = 0; i < entries_count; ++i) {
        const size_t entry_base = 20 + i * 16;
        u16 key_off = 0;
        u16 param_fmt = 0;
        u32 param_len = 0;
        u32 param_max_len = 0;
        u32 d_off = 0;

        std::memcpy(&key_off, &data[entry_base + 0], 2);
        std::memcpy(&param_fmt, &data[entry_base + 2], 2);
        std::memcpy(&param_len, &data[entry_base + 4], 4);
        std::memcpy(&param_max_len, &data[entry_base + 8], 4);
        std::memcpy(&d_off, &data[entry_base + 12], 4);

        const size_t actual_key_pos = key_table_off + key_off;
        if (actual_key_pos >= data.size()) continue;

        // Read null-terminated key
        std::string key;
        for (size_t p = actual_key_pos; p < data.size() && data[p] != 0; ++p) {
            key.push_back(static_cast<char>(data[p]));
        }
        if (key.empty()) continue;

        const size_t actual_data_pos = data_table_off + d_off;
        if (actual_data_pos > data.size() || actual_data_pos + param_len > data.size()) continue;

        SfoEntry entry;
        entry.key = key;
        entry.format = static_cast<SfoFormat>(param_fmt);

        if (entry.format == SfoFormat::Utf8String || entry.format == SfoFormat::Utf8Special) {
            std::string val;
            for (size_t p = actual_data_pos; p < actual_data_pos + param_len; ++p) {
                if (data[p] == 0) break;
                val.push_back(static_cast<char>(data[p]));
            }
            entry.string_value = std::move(val);
        } else if (entry.format == SfoFormat::UInt32 && param_len >= 4) {
            std::memcpy(&entry.int_value, &data[actual_data_pos], 4);
        }

        out_entries[key] = std::move(entry);
    }

    return true;
}

bool SfoParser::parse_file(const std::string& path, std::unordered_map<std::string, SfoEntry>& out_entries) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    if (sz <= 0 || sz > 10 * 1024 * 1024) { // max 10MB sanity
        std::fclose(f);
        return false;
    }
    std::fseek(f, 0, SEEK_SET);
    std::vector<u8> buf(static_cast<size_t>(sz));
    const size_t read_bytes = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (read_bytes != buf.size()) return false;
    return parse(buf, out_entries);
}

std::string SfoParser::get_string(const std::unordered_map<std::string, SfoEntry>& entries, const std::string& key, const std::string& default_val) {
    auto it = entries.find(key);
    if (it != entries.end() && !it->second.string_value.empty()) {
        return it->second.string_value;
    }
    return default_val;
}

u32 SfoParser::get_uint32(const std::unordered_map<std::string, SfoEntry>& entries, const std::string& key, u32 default_val) {
    auto it = entries.find(key);
    if (it != entries.end() && it->second.format == SfoFormat::UInt32) {
        return it->second.int_value;
    }
    return default_val;
}

} // namespace ps4
