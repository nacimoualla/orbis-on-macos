#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>
#include "ps4/sfo.hpp"

// Helper to construct a synthetic SFO buffer
static std::vector<ps4::u8> make_synthetic_sfo(const std::string& title, const std::string& title_id, ps4::u32 app_type) {
    // Header: magic (\0PSF), version 0x00000101, key_table_off, data_table_off, entries_count (3)
    std::vector<ps4::u8> buf;
    buf.resize(20 + 3 * 16); // Header (20) + 3 entries (3 * 16 = 48) -> 68 bytes

    // Magic
    buf[0] = 0x00; buf[1] = 'P'; buf[2] = 'S'; buf[3] = 'F';
    // Version 1.1
    buf[4] = 0x01; buf[5] = 0x01; buf[6] = 0x00; buf[7] = 0x00;

    const std::string key0 = "APP_TYPE";
    const std::string key1 = "TITLE";
    const std::string key2 = "TITLE_ID";

    std::vector<ps4::u8> key_table;
    auto add_key = [&](const std::string& k) -> ps4::u16 {
        ps4::u16 off = static_cast<ps4::u16>(key_table.size());
        for (char c : k) key_table.push_back(static_cast<ps4::u8>(c));
        key_table.push_back(0);
        return off;
    };
    ps4::u16 k0_off = add_key(key0);
    ps4::u16 k1_off = add_key(key1);
    ps4::u16 k2_off = add_key(key2);

    std::vector<ps4::u8> data_table;
    auto add_str = [&](const std::string& s) -> std::pair<ps4::u32, ps4::u32> {
        ps4::u32 off = static_cast<ps4::u32>(data_table.size());
        for (char c : s) data_table.push_back(static_cast<ps4::u8>(c));
        data_table.push_back(0);
        return {off, static_cast<ps4::u32>(s.size() + 1)};
    };
    auto add_u32 = [&](ps4::u32 val) -> std::pair<ps4::u32, ps4::u32> {
        ps4::u32 off = static_cast<ps4::u32>(data_table.size());
        for (int i = 0; i < 4; ++i) data_table.push_back(static_cast<ps4::u8>((val >> (i * 8)) & 0xFF));
        return {off, 4};
    };

    auto [d0_off, d0_len] = add_u32(app_type);
    auto [d1_off, d1_len] = add_str(title);
    auto [d2_off, d2_len] = add_str(title_id);

    ps4::u32 key_table_off = static_cast<ps4::u32>(buf.size());
    ps4::u32 data_table_off = key_table_off + static_cast<ps4::u32>(key_table.size());

    // Write header offsets
    std::memcpy(&buf[8], &key_table_off, 4);
    std::memcpy(&buf[12], &data_table_off, 4);
    ps4::u32 count = 3;
    std::memcpy(&buf[16], &count, 4);

    auto write_entry = [&](size_t idx, ps4::u16 k_off, ps4::u16 fmt, ps4::u32 len, ps4::u32 max_len, ps4::u32 d_off) {
        size_t base = 20 + idx * 16;
        std::memcpy(&buf[base + 0], &k_off, 2);
        std::memcpy(&buf[base + 2], &fmt, 2);
        std::memcpy(&buf[base + 4], &len, 4);
        std::memcpy(&buf[base + 8], &max_len, 4);
        std::memcpy(&buf[base + 12], &d_off, 4);
    };

    write_entry(0, k0_off, static_cast<ps4::u16>(ps4::SfoFormat::UInt32), d0_len, 4, d0_off);
    write_entry(1, k1_off, static_cast<ps4::u16>(ps4::SfoFormat::Utf8String), d1_len, d1_len, d1_off);
    write_entry(2, k2_off, static_cast<ps4::u16>(ps4::SfoFormat::Utf8String), d2_len, d2_len, d2_off);

    buf.insert(buf.end(), key_table.begin(), key_table.end());
    buf.insert(buf.end(), data_table.begin(), data_table.end());
    return buf;
}

int main() {
    std::cout << "[SFO] Testing SFO Parser...\n";

    // Test 1: Invalid buffer rejection
    {
        std::vector<ps4::u8> bad(10, 0);
        std::unordered_map<std::string, ps4::SfoEntry> entries;
        assert(!ps4::SfoParser::parse(bad, entries));

        // Bad magic
        std::vector<ps4::u8> bad_magic(30, 0);
        bad_magic[0] = 'B'; bad_magic[1] = 'A'; bad_magic[2] = 'D'; bad_magic[3] = '!';
        assert(!ps4::SfoParser::parse(bad_magic, entries));

        // Non-existent file
        assert(!ps4::SfoParser::parse_file("nonexistent_file_path.sfo", entries));
        std::cout << "  ✓ Test 1: Corrupt/short buffer and missing files rejected\n";
    }

    // Test 2: Synthetic SFO parse
    {
        auto sfo = make_synthetic_sfo("Thomas Was Alone", "CUSA00909", 1);
        std::unordered_map<std::string, ps4::SfoEntry> entries;
        assert(ps4::SfoParser::parse(sfo, entries));
        assert(entries.size() == 3);
        assert(ps4::SfoParser::get_string(entries, "TITLE") == "Thomas Was Alone");
        assert(ps4::SfoParser::get_string(entries, "TITLE_ID") == "CUSA00909");
        assert(ps4::SfoParser::get_uint32(entries, "APP_TYPE") == 1);
        assert(ps4::SfoParser::get_string(entries, "NONEXISTENT", "default") == "default");
        assert(ps4::SfoParser::get_uint32(entries, "NONEXISTENT", 99) == 99);
        std::cout << "  ✓ Test 2: Synthetic SFO parsed successfully\n";
    }

    // Test 3: Real game param.sfo file parsing (Thomas Was Alone)
    {
        std::unordered_map<std::string, ps4::SfoEntry> entries;
        const std::string real_sfo = "games/thomas_was_alone/EP4395-CUSA00909_00-THOMASWASALONEFG/sce_sys/param.sfo";
        if (ps4::SfoParser::parse_file(real_sfo, entries)) {
            assert(ps4::SfoParser::get_string(entries, "TITLE") == "Thomas Was Alone");
            assert(ps4::SfoParser::get_string(entries, "TITLE_ID") == "CUSA00909");
            assert(ps4::SfoParser::get_string(entries, "APP_VER") == "01.00");
            std::cout << "  ✓ Test 3: Real game param.sfo file (Thomas) parsed successfully\n";
        } else {
            std::cout << "  - Note: real Thomas param.sfo not accessible\n";
        }
    }

    // Test 4: Real game param.sfo file parsing (Flappy Bird)
    {
        std::unordered_map<std::string, ps4::SfoEntry> entries;
        const std::string real_sfo = "games/flappy_bird/ED1633-NPXX51362_00-TER1520000000000/sce_sys/param.sfo";
        if (ps4::SfoParser::parse_file(real_sfo, entries)) {
            assert(ps4::SfoParser::get_string(entries, "TITLE_ID") == "NPXX51362");
            assert(!ps4::SfoParser::get_string(entries, "TITLE").empty());
            assert(ps4::SfoParser::get_string(entries, "APP_VER") == "01.00");
            std::cout << "  ✓ Test 4: Real game param.sfo file (Flappy) parsed successfully\n";
        } else {
            std::cout << "  - Note: real Flappy param.sfo not accessible\n";
        }
    }

    std::cout << "[SFO] All tests passed!\n";
    return 0;
}
