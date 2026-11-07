# Keyboard & Mouse Support and GUI App Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement comprehensive Keyboard and Mouse input mapping (WASD analog stick emulation, arrow key D-pad, face buttons, DualShock 4 touchpad emulation, and `libSceMouse` HLE) alongside major UX updates to `ps4-gui` (SFO metadata parsing, title artwork thumbnails, interactive controls guide HUD, log search/filtering, and automatic game discovery).

**Architecture:** A new pure C++ `SfoParser` extracts title names, IDs, and versions from `sce_sys/param.sfo`, which feeds `game_import` and the AppKit `ps4-gui` library frontend. `DisplayWindow` translates macOS AppKit keyboard and mouse events into unified `PadState` and `MouseState` structures, emulating DualShock 4 digital buttons, analog sticks, and touchpad touch coordinates `(0..1920, 0..943)` for games running under `ps4-runner`, while `libSceMouse` HLE exposes direct mouse events.

**Tech Stack:** C++20, macOS AppKit (Objective-C++ ARC), Metal / CAMetalLayer, CMake.

## Global Constraints

- C++20 standard (`-std=c++20`).
- macOS Apple Silicon (ARM64) target with Objective-C++ ARC (`-fobjc-arc`).
- Clean-room implementation: zero proprietary Sony SDK headers, libraries, or copyrighted dumps.
- All existing 27 CTest unit tests must pass after every task.
- Zero compiler warnings or errors (`-Wall -Wextra -Wpedantic`).
- Maintain documentation integrity and preserve existing comments.
- Do not break backward compatibility with `DisplayWindow::input_buttons()`.

---

### Task 1: SFO (System File Object) Metadata Parser

**Files:**
- Create: `include/ps4/sfo.hpp`
- Create: `src/core/sfo.cpp`
- Create: `tests/test_sfo.cpp`
- Modify: `CMakeLists.txt:42-67` (add `src/core/sfo.cpp` to `ps4_core`)
- Modify: `CMakeLists.txt:230-253` (add `test_sfo` executable and CTest test)

**Interfaces:**
- Consumes: Standard C++ file streams and vectors.
- Produces:
  ```cpp
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
  }
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_sfo.cpp`:
```cpp
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
        std::cout << "  ✓ Test 1: Corrupt/short buffer rejected\n";
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
        std::cout << "  ✓ Test 2: Synthetic SFO parsed successfully\n";
    }

    // Test 3: Real game param.sfo file parsing (if present)
    {
        std::unordered_map<std::string, ps4::SfoEntry> entries;
        const std::string real_sfo = "games/thomas_was_alone/EP4395-CUSA00909_00-THOMASWASALONEFG/sce_sys/param.sfo";
        if (ps4::SfoParser::parse_file(real_sfo, entries)) {
            assert(ps4::SfoParser::get_string(entries, "TITLE") == "Thomas Was Alone");
            assert(ps4::SfoParser::get_string(entries, "TITLE_ID") == "CUSA00909");
            std::cout << "  ✓ Test 3: Real game param.sfo file parsed successfully\n";
        }
    }

    std::cout << "[SFO] All tests passed!\n";
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_sfo`
Expected: FAIL with `No such file: ps4/sfo.hpp` or target not found.

- [ ] **Step 3: Write minimal implementation**

Create `include/ps4/sfo.hpp`:
```cpp
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
```

Create `src/core/sfo.cpp`:
```cpp
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

        const size_t actual_data_pos = data_table_off + d_off;
        if (actual_data_pos + param_len > data.size()) continue;

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
```

Update `CMakeLists.txt` to add `src/core/sfo.cpp` to `ps4_core` and add `test_sfo`:
```cmake
add_executable(test_sfo tests/test_sfo.cpp)
target_link_libraries(test_sfo PRIVATE ps4_core)
add_test(NAME SfoParser COMMAND test_sfo)
set_tests_properties(SfoParser PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_sfo && ctest --test-dir build -R SfoParser --output-on-failure`
Expected: `1/1 Test #...: SfoParser ... Passed`

- [ ] **Step 5: Commit**

```bash
git add include/ps4/sfo.hpp src/core/sfo.cpp tests/test_sfo.cpp CMakeLists.txt
git commit -m "feat(core): implement SFO metadata parser for PS4 param.sfo files"
```

---

### Task 2: Integrate SFO & Artwork Discovery in Game Import Scanner

**Files:**
- Modify: `include/ps4/game_import.hpp`
- Modify: `src/gui/game_import.cpp`
- Modify: `tests/test_phase16.cpp`

**Interfaces:**
- Consumes: `SfoParser` (`include/ps4/sfo.hpp`).
- Produces: `GameEntry::title_id`, `GameEntry::app_ver`, `GameEntry::icon_path` populated during `scan_folder()`.

- [ ] **Step 1: Write the failing test**

In `tests/test_phase16.cpp`, add checks for SFO metadata and `icon0.png` discovery in `games/flappy_bird/...` and `games/thomas_was_alone/...`:
```cpp
    // Verify param.sfo metadata and icon discovery
    {
        auto games = ps4::gui::scan_folder("games");
        for (const auto& g : games) {
            if (g.id.find("ED1633-NPXX51362") != std::string::npos) {
                check(!g.name.empty(), "flappy title name populated from param.sfo");
                check(g.title_id == "NPXX51362", "flappy title_id extracted from param.sfo");
                check(!g.icon_path.empty(), "flappy icon_path populated");
            }
            if (g.id.find("CUSA00909") != std::string::npos) {
                check(g.name == "Thomas Was Alone", "thomas title name populated from param.sfo");
                check(g.title_id == "CUSA00909", "thomas title_id extracted from param.sfo");
                check(!g.icon_path.empty(), "thomas icon_path populated");
            }
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ctest --test-dir build -R Phase16_GameImport --output-on-failure`
Expected: FAIL because `GameEntry` lacks `title_id` / `icon_path` or `g.name` is empty.

- [ ] **Step 3: Write minimal implementation**

In `include/ps4/game_import.hpp`:
```cpp
struct GameEntry {
    std::string id;        // directory name, e.g. ED1633-NPXX51362_00-...
    std::string name;      // from param.sfo TITLE, e.g. "Thomas Was Alone"
    std::string title_id;  // from param.sfo TITLE_ID, e.g. "CUSA00909"
    std::string app_ver;   // from param.sfo APP_VER, e.g. "01.00"
    std::string icon_path; // path to sce_sys/icon0.png if present
    std::string root;      // absolute path to the game directory
    std::string eboot;     // absolute path to the validated eboot (an ELF)
    u64 total_bytes{0};    // recursive size of the title directory
    u32 file_count{0};
    std::string status;    // "ok", "no eboot", "eboot not an ELF", ...
};
```

In `src/gui/game_import.cpp`:
Include `"ps4/sfo.hpp"`:
Inside `walk()` when `contains_eboot` is true:
```cpp
    if (contains_eboot(dir, eboot, status)) {
        GameEntry g;
        g.id = base_name(dir);
        g.root = dir;
        g.eboot = eboot;
        g.status = status;
        g.total_bytes = dir_size(dir, g.file_count);

        // Check for sce_sys/param.sfo
        const std::string sfo_path = dir + "/sce_sys/param.sfo";
        std::unordered_map<std::string, SfoEntry> sfo_entries;
        if (SfoParser::parse_file(sfo_path, sfo_entries)) {
            g.name = SfoParser::get_string(sfo_entries, "TITLE", g.id);
            g.title_id = SfoParser::get_string(sfo_entries, "TITLE_ID", "");
            g.app_ver = SfoParser::get_string(sfo_entries, "APP_VER", "01.00");
        } else {
            g.name = g.id;
        }

        // Check for sce_sys/icon0.png
        const std::string icon_path = dir + "/sce_sys/icon0.png";
        if (is_file(icon_path)) {
            g.icon_path = icon_path;
        }

        out.push_back(std::move(g));
        return;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_phase16 && ctest --test-dir build -R Phase16_GameImport --output-on-failure`
Expected: PASS with 100% checks passed.

- [ ] **Step 5: Commit**

```bash
git add include/ps4/game_import.hpp src/gui/game_import.cpp tests/test_phase16.cpp
git commit -m "feat(gui): integrate SFO title metadata and icon discovery into game scanner"
```

---

### Task 3: Input State Data Structures & NidRegistry Pad/Mouse Extension

**Files:**
- Create: `include/ps4/input.hpp`
- Modify: `include/ps4/nid_registry.hpp`
- Modify: `src/hle/nid_registry.cpp`
- Create: `tests/test_input.cpp`
- Modify: `CMakeLists.txt` (add `test_input` executable and CTest test)

**Interfaces:**
- Consumes: `MemoryManager`, `GuestAddr`, `u32`.
- Produces:
  ```cpp
  namespace ps4 {
  enum PadButton : u32 {
      kPadSelect        = 0x0001, // SHARE
      kPadL3            = 0x0002,
      kPadR3            = 0x0004,
      kPadStart         = 0x0008, // OPTIONS
      kPadUp            = 0x0010,
      kPadRight         = 0x0020,
      kPadDown          = 0x0040,
      kPadLeft          = 0x0080,
      kPadL2            = 0x0100, // digital
      kPadR2            = 0x0200, // digital
      kPadL1            = 0x0400,
      kPadR1            = 0x0800,
      kPadTriangle      = 0x1000,
      kPadCircle        = 0x2000,
      kPadCross         = 0x4000,
      kPadSquare        = 0x8000,
      kPadPS            = 0x00010000,
      kPadTouchPadClick = 0x00100000,
      kPadIntercepted   = 0x00200000,
  };
  struct PadState {
      u32 buttons{0};
      u8 lx{128}, ly{128};
      u8 rx{128}, ry{128};
      u8 l2{0}, r2{0};
      u16 touch_x{960}, touch_y{471};
      bool touch_down{false};
      bool connected{true};
  };
  struct MouseState {
      i32 delta_x{0};
      i32 delta_y{0};
      i32 wheel{0};
      u32 buttons{0}; // bit 0: left, bit 1: right, bit 2: middle
      bool connected{true};
  };
  }
  ```
  In `NidRegistry`:
  - `void set_pad_state(const PadState& state);`
  - `PadState get_pad_state() const;`
  - `void set_mouse_state(const MouseState& state);`
  - `MouseState get_mouse_state() const;`

- [ ] **Step 1: Write the failing test**

Create `tests/test_input.cpp`:
```cpp
#include <cassert>
#include <iostream>
#include "ps4/input.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/memory.hpp"

int main() {
    std::cout << "[INPUT] Testing PadState and MouseState in NidRegistry...\n";

    ps4::NidRegistry reg;
    ps4::MemoryManager mem;
    assert(mem.initialize());

    // Test 1: Default PadState
    ps4::PadState default_st = reg.get_pad_state();
    assert(default_st.buttons == 0);
    assert(default_st.lx == 128 && default_st.ly == 128);
    assert(default_st.rx == 128 && default_st.ry == 128);
    assert(default_st.touch_x == 960 && default_st.touch_y == 471);
    std::cout << "  ✓ Test 1: Default PadState verified\n";

    // Test 2: Custom PadState with analog sticks and touchpad
    ps4::PadState st{};
    st.buttons = ps4::kPadCross | ps4::kPadL1;
    st.lx = 0;   // Hard Left
    st.ly = 255; // Hard Down
    st.touch_x = 400;
    st.touch_y = 300;
    st.touch_down = true;

    reg.set_pad_state(st);
    ps4::PadState retrieved = reg.get_pad_state();
    assert(retrieved.buttons == (ps4::kPadCross | ps4::kPadL1));
    assert(retrieved.lx == 0 && retrieved.ly == 255);
    assert(retrieved.touch_x == 400 && retrieved.touch_y == 300);
    assert(retrieved.touch_down == true);
    std::cout << "  ✓ Test 2: Set and retrieve PadState verified\n";

    // Test 3: MouseState
    ps4::MouseState ms{};
    ms.delta_x = 15;
    ms.delta_y = -10;
    ms.buttons = 1; // Left click
    reg.set_mouse_state(ms);
    ps4::MouseState retrieved_ms = reg.get_mouse_state();
    assert(retrieved_ms.delta_x == 15);
    assert(retrieved_ms.delta_y == -10);
    assert(retrieved_ms.buttons == 1);
    std::cout << "  ✓ Test 3: Set and retrieve MouseState verified\n";

    std::cout << "[INPUT] All input state tests passed!\n";
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_input`
Expected: FAIL with `include/ps4/input.hpp` missing or `set_pad_state` missing.

- [ ] **Step 3: Write minimal implementation**

Create `include/ps4/input.hpp`:
```cpp
#pragma once

#include "ps4/common.hpp"

namespace ps4 {

enum PadButton : u32 {
    kPadSelect        = 0x0001, // SHARE
    kPadL3            = 0x0002,
    kPadR3            = 0x0004,
    kPadStart         = 0x0008, // OPTIONS
    kPadUp            = 0x0010,
    kPadRight         = 0x0020,
    kPadDown          = 0x0040,
    kPadLeft          = 0x0080,
    kPadL2            = 0x0100, // digital
    kPadR2            = 0x0200, // digital
    kPadL1            = 0x0400,
    kPadR1            = 0x0800,
    kPadTriangle      = 0x1000,
    kPadCircle        = 0x2000,
    kPadCross         = 0x4000,
    kPadSquare        = 0x8000,
    kPadPS            = 0x00010000,
    kPadTouchPadClick = 0x00100000,
    kPadIntercepted   = 0x00200000,
};

struct PadState {
    u32 buttons{0};
    u8 lx{128}, ly{128};
    u8 rx{128}, ry{128};
    u8 l2{0}, r2{0};
    u16 touch_x{960}, touch_y{471};
    bool touch_down{false};
    bool connected{true};
};

struct MouseState {
    i32 delta_x{0};
    i32 delta_y{0};
    i32 wheel{0};
    u32 buttons{0}; // bit 0: left, bit 1: right, bit 2: middle
    bool connected{true};
};

} // namespace ps4
```

In `include/ps4/nid_registry.hpp`:
Include `"ps4/input.hpp"` and add:
```cpp
    void set_pad_buttons(u32 buttons);
    void set_pad_state(const PadState& state);
    [[nodiscard]] PadState get_pad_state() const noexcept;

    void set_mouse_state(const MouseState& state);
    [[nodiscard]] MouseState get_mouse_state() const noexcept;
```

In `src/hle/nid_registry.cpp`:
Replace `g_pad_buttons` with a thread-safe `g_pad_state` and update `write_pad_data`:
```cpp
static std::mutex g_input_mutex;
static PadState g_active_pad_state{};
static std::atomic<u32> g_pad_pressed_latch{0};
static MouseState g_active_mouse_state{};

void NidRegistry::set_pad_state(const PadState& state) {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    const u32 prev_buttons = g_active_pad_state.buttons;
    g_active_pad_state = state;
    g_pad_pressed_latch.fetch_or(state.buttons & ~prev_buttons, std::memory_order_relaxed);
}

PadState NidRegistry::get_pad_state() const noexcept {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    return g_active_pad_state;
}

void NidRegistry::set_pad_buttons(u32 buttons) {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    const u32 prev = g_active_pad_state.buttons;
    g_active_pad_state.buttons = buttons;
    g_pad_pressed_latch.fetch_or(buttons & ~prev, std::memory_order_relaxed);
}

void NidRegistry::set_mouse_state(const MouseState& state) {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    g_active_mouse_state = state;
}

MouseState NidRegistry::get_mouse_state() const noexcept {
    std::lock_guard<std::mutex> lock(g_input_mutex);
    return g_active_mouse_state;
}

static void write_pad_data(MemoryManager& mem, GuestAddr p, const PadState& st) {
    mem.zero_memory(p, kScePadDataSize);
    const u32 latched_buttons = st.buttons | g_pad_pressed_latch.exchange(0, std::memory_order_relaxed);
    mem.write_value(p + 0x00, latched_buttons);

    // Left stick
    mem.write_value(p + 0x04, st.lx);
    mem.write_value(p + 0x05, st.ly);
    // Right stick
    mem.write_value(p + 0x06, st.rx);
    mem.write_value(p + 0x07, st.ry);
    // Analog L2/R2
    mem.write_value(p + 0x08, st.l2);
    mem.write_value(p + 0x09, st.r2);

    mem.write_value(p + 0x18, 1.0f); // orientation w (identity)
    mem.write_value(p + 0x20, 1.0f); // acceleration: 1 g along y

    // Touch data
    if (st.touch_down || (latched_buttons & kPadTouchPadClick)) {
        mem.write_value(p + 0x34, static_cast<u8>(1)); // 1 active touch point
        mem.write_value(p + 0x3C, st.touch_x);
        mem.write_value(p + 0x3E, st.touch_y);
    }

    mem.write_value(p + 0x4C, static_cast<u8>(st.connected ? 1 : 0));
    const u64 ts = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    mem.write_value(p + 0x50, ts);
}
```
In `pad_read_state_handler` and `scePadRead`: call `write_pad_data(mem, data_ptr, g_nid_registry ? g_nid_registry->get_pad_state() : PadState{})`.

In `CMakeLists.txt`:
```cmake
add_executable(test_input tests/test_input.cpp)
target_link_libraries(test_input PRIVATE ps4_core)
add_test(NAME InputState COMMAND test_input)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_input && ctest --test-dir build -R InputState --output-on-failure`
Expected: PASS with 100% tests passed.

- [ ] **Step 5: Commit**

```bash
git add include/ps4/input.hpp include/ps4/nid_registry.hpp src/hle/nid_registry.cpp tests/test_input.cpp CMakeLists.txt
git commit -m "feat(hle): introduce unified PadState and MouseState structures and write_pad_data mapping"
```

---

### Task 4: DisplayWindow Keyboard & Mouse Translation Engine

**Files:**
- Modify: `include/ps4/display_window.hpp`
- Modify: `src/gpu/display_window.mm`
- Modify: `src/main.cpp`
- Modify: `tests/test_phase8.cpp`

**Interfaces:**
- Consumes: macOS AppKit events (`NSEventTypeKeyDown`, `NSEventTypeKeyUp`, `NSEventTypeMouseMoved`, `NSEventTypeLeftMouseDown`, etc.).
- Produces:
  ```cpp
  [[nodiscard]] PadState input_pad_state() const noexcept;
  [[nodiscard]] MouseState input_mouse_state() const noexcept;
  ```

- [ ] **Step 1: Write the failing test**

In `tests/test_phase8.cpp`, add tests verifying `window.input_pad_state()` and `window.input_mouse_state()`:
```cpp
    // Test 9: DisplayWindow PadState and MouseState retrieval
    {
        ps4::DisplayWindow window;
        ps4::WindowConfig cfg{.headless = true};
        assert(window.initialize(metal, cfg));
        ps4::PadState ps = window.input_pad_state();
        assert(ps.lx == 128 && ps.ly == 128);
        assert(ps.touch_x == 960 && ps.touch_y == 471);
        ps4::MouseState ms = window.input_mouse_state();
        assert(ms.delta_x == 0 && ms.delta_y == 0);
        std::cout << "  ✓ Test 9: DisplayWindow input_pad_state and input_mouse_state verified\n";
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_phase8`
Expected: FAIL with `input_pad_state` or `input_mouse_state` not declared in `ps4::DisplayWindow`.

- [ ] **Step 3: Write minimal implementation**

In `include/ps4/display_window.hpp`:
```cpp
#include "ps4/input.hpp"
...
    /// Get current controller/keyboard buttons bitmask (Bit 14 = Cross/Space/Click)
    [[nodiscard]] u32 input_buttons() const noexcept;

    /// Get comprehensive current pad state (sticks, buttons, triggers, touchpad)
    [[nodiscard]] PadState input_pad_state() const noexcept;

    /// Get current mouse state (delta movement, wheel, buttons)
    [[nodiscard]] MouseState input_mouse_state() const noexcept;
```

In `src/gpu/display_window.mm`:
Add tracking for key states and mouse position in `DisplayWindow::Impl`:
```cpp
    PadState active_pad_state{};
    MouseState active_mouse_state{};
    bool key_down[512]{false};
    NSPoint last_mouse_pos{0, 0};
```
In `poll_events()`:
Handle keys:
- W (13): Up Left Stick (`ly = 0`)
- S (1): Down Left Stick (`ly = 255`)
- A (0): Left Left Stick (`lx = 0`)
- D (2): Right Left Stick (`lx = 255`)
- Up Arrow (126): `kPadUp`
- Down Arrow (125): `kPadDown`
- Left Arrow (123): `kPadLeft`
- Right Arrow (124): `kPadRight`
- Space (49), Enter (36): `kPadCross`
- Backspace (51), C (8): `kPadCircle`
- F (3), X (7): `kPadSquare`
- E (14), R (15): `kPadTriangle`
- Q (12), Left Shift (56): `kPadL1`
- Tab (48): `kPadTouchPadClick`
- Escape (53): `kPadStart` (Options menu)
- Command+W (13 with `NSEventModifierFlagCommand`) or Command+Q (12 with `NSEventModifierFlagCommand`): `close()` window! (Escape no longer kills the window).

Handle mouse:
- `NSEventTypeMouseMoved`, `NSEventTypeLeftMouseDragged`, `NSEventTypeRightMouseDragged`:
  - Calculate delta: `dx = event.deltaX`, `dy = event.deltaY`.
  - Update `active_mouse_state.delta_x += dx`, `active_mouse_state.delta_y += dy`.
  - Convert mouse window location to DualShock 4 touchpad coords:
    `NSRect bounds = window.contentView.bounds;`
    `CGFloat x_ratio = std::clamp(event.locationInWindow.x / bounds.size.width, 0.0, 1.0);`
    `CGFloat y_ratio = std::clamp((bounds.size.height - event.locationInWindow.y) / bounds.size.height, 0.0, 1.0);`
    `active_pad_state.touch_x = static_cast<u16>(x_ratio * 1920.0);`
    `active_pad_state.touch_y = static_cast<u16>(y_ratio * 943.0);`
- `NSEventTypeLeftMouseDown`:
  `active_pad_state.touch_down = true;`
  `active_pad_state.buttons |= kPadCross | kPadTouchPadClick;`
  `active_mouse_state.buttons |= 1;`
- `NSEventTypeLeftMouseUp`:
  `active_pad_state.touch_down = false;`
  `active_pad_state.buttons &= ~(kPadCross | kPadTouchPadClick);`
  `active_mouse_state.buttons &= ~1;`
- `NSEventTypeRightMouseDown`:
  `active_pad_state.buttons |= kPadCircle;`
  `active_mouse_state.buttons |= 2;`
- `NSEventTypeRightMouseUp`:
  `active_pad_state.buttons &= ~kPadCircle;`
  `active_mouse_state.buttons &= ~2;`

In `src/main.cpp`:
In the event loop, update to push both `window->input_pad_state()` and `window->input_mouse_state()`:
```cpp
    nid_registry.set_pad_state(window->input_pad_state());
    nid_registry.set_mouse_state(window->input_mouse_state());
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_phase8 && ctest --test-dir build -R Phase8_DisplayWindow --output-on-failure`
Expected: PASS with all 9 tests passing.

- [ ] **Step 5: Commit**

```bash
git add include/ps4/display_window.hpp src/gpu/display_window.mm src/main.cpp tests/test_phase8.cpp
git commit -m "feat(input): implement keyboard and mouse mapping engine in DisplayWindow"
```

---

### Task 5: libSceMouse HLE Subsystem

**Files:**
- Modify: `src/hle/nid_registry.cpp:7650-7665`
- Modify: `tests/test_input.cpp`

**Interfaces:**
- Consumes: `NidRegistry::get_mouse_state()`.
- Produces: `sceMouseOpen`, `sceMouseRead`, `sceMouseGetControllerInformation` returning device data to guest.

- [ ] **Step 1: Write the failing test**

In `tests/test_input.cpp`, add tests invoking the HLE symbols for `sceMouseOpen` and `sceMouseRead`:
```cpp
    // Test 4: libSceMouse HLE symbols
    {
        const auto* sym_open = reg.find_by_name("sceMouseOpen");
        assert(sym_open != nullptr);
        ps4::CpuContext ctx{};
        ctx.rdi = 1; // userId
        ctx.rsi = 0; // type
        ctx.rdx = 0; // index
        sym_open->handler(ctx, mem);
        assert(ctx.rax > 0); // valid mouse handle >= 1

        const auto* sym_read = reg.find_by_name("sceMouseRead");
        assert(sym_read != nullptr);
        ps4::GuestAddr mouse_buf = 0x20000;
        mem.allocate_fixed(mouse_buf, 0x1000, ps4::PROT_READ_ | ps4::PROT_WRITE_);
        ctx.rdi = 1; // handle
        ctx.rsi = mouse_buf;
        ctx.rdx = 1; // num samples
        sym_read->handler(ctx, mem);
        assert(ctx.rax == 1); // 1 sample read
        std::cout << "  ✓ Test 4: libSceMouse sceMouseOpen and sceMouseRead verified\n";
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_input && ctest --test-dir build -R InputState --output-on-failure`
Expected: FAIL because `sceMouseOpen` currently returns error `0x80DF0003ULL` (no device).

- [ ] **Step 3: Write minimal implementation**

In `src/hle/nid_registry.cpp`:
Remove `sceMouseOpen` and `sceMouseRead` from `kAbsentDevices[]`.
Register active handlers:
```cpp
    // SceMouseData (0x40 bytes approx):
    // timestamp (u64 at 0x00), buttons (u32 at 0x08), xAxis (i32 at 0x0C), yAxis (i32 at 0x10), wheel (i32 at 0x14), connected (u8 at 0x18)
    constexpr GuestSize kSceMouseDataSize = 0x40;

    // sceMouseOpen(int userId, int type, int index, const void* param)
    register_symbol(0x45AAB16487FA0EF1ULL, "sceMouseOpen", "libSceMouse",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 1; // Handle 1
        });

    // sceMouseRead(int handle, SceMouseData* data, int num) -> samples read
    register_symbol(0xC7CAA75EA87FB623ULL, "sceMouseRead", "libSceMouse",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 num = static_cast<i32>(ctx.rdx);
            const GuestAddr p = ctx.rsi;
            if (num < 1 || p <= 0x1000 || !mem.is_valid_address(p, kSceMouseDataSize)) {
                ctx.rax = num < 1 ? 0 : 0x80DF0001ULL; // SCE_MOUSE_ERROR_INVALID_ARG
                return;
            }
            mem.zero_memory(p, kSceMouseDataSize);
            const MouseState ms = g_active_mouse_state;
            const u64 ts = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            mem.write_value(p + 0x00, ts);
            mem.write_value(p + 0x08, ms.buttons);
            mem.write_value(p + 0x0C, ms.delta_x);
            mem.write_value(p + 0x10, ms.delta_y);
            mem.write_value(p + 0x14, ms.wheel);
            mem.write_value(p + 0x18, static_cast<u8>(ms.connected ? 1 : 0));
            ctx.rax = 1;
        });
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_input && ctest --test-dir build -R InputState --output-on-failure`
Expected: PASS with 100% tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/hle/nid_registry.cpp tests/test_input.cpp
git commit -m "feat(hle): implement libSceMouse sceMouseOpen and sceMouseRead HLE functions"
```

---

### Task 6: GUI App Auto-Discovery & Table View Metadata Overhaul

**Files:**
- Modify: `src/gui/game_library_gui.mm`

**Interfaces:**
- Consumes: `GameEntry::name`, `GameEntry::title_id`, `GameEntry::app_ver`, `GameEntry::icon_path`.
- Produces: Automatic discovery of `<repo_root>/games` on startup; rich table view columns displaying icon thumbnail, Title, Title ID, Version, Size, and Status.

- [ ] **Step 1: Write the failing test**

Run: `grep -E "icon_path|title_id" src/gui/game_library_gui.mm`
Expected: Empty / no matches.

- [ ] **Step 2: Implement auto-discovery and table columns**

In `src/gui/game_library_gui.mm`:
1. In `buildWindow`:
   - Increase window default size: `NSMakeRect(0, 0, 1020, 700)`.
   - Update table columns:
     - `icon` (width: 50)
     - `title` (width: 250, title: "Title")
     - `title_id` (width: 110, title: "Title ID")
     - `app_ver` (width: 70, title: "Version")
     - `size` (width: 90, title: "Size")
     - `status` (width: 100, title: "Status")
   - In `tableView:viewForTableColumn:row:` or `tableView:objectValueForTableColumn:row:`:
     - For `icon`: load `NSImage` from `g.icon_path` if available; fallback to default controller icon.
     - For `title`: return `@(g.name.c_str())`.
     - For `title_id`: return `@(g.title_id.c_str())`.
     - For `app_ver`: return `@(g.app_ver.c_str())`.
2. In `init`:
   - Locate repo root: if `repo_root(root)` succeeds and `root + "/games"` exists:
     Schedule `[self scanFolder:@((root + "/games").c_str())]` so games appear immediately on launch!

- [ ] **Step 3: Build ps4-gui**

Run: `cmake --build build --target ps4-gui ps4-gui-app`
Expected: Build succeeds with 0 errors.

- [ ] **Step 4: Verify functionality**

Run: `build/ps4-gui` in background or test with a smoke test script.
Verify table displays "Thomas Was Alone" and "★ FLAPPY BIRD ★ By TER152 ★" with icons and versions.

- [ ] **Step 5: Commit**

```bash
git add src/gui/game_library_gui.mm
git commit -m "feat(gui): auto-scan games folder and display SFO titles, IDs, versions, and icons"
```

---

### Task 7: GUI App Game Detail Sidebar & Direct Launch Pane

**Files:**
- Modify: `src/gui/game_library_gui.mm`

**Interfaces:**
- Consumes: Selected `GameEntry` from table view.
- Produces: Interactive sidebar showing large game artwork (`icon0.png`), detailed game specs, and primary Launch Game action button.

- [ ] **Step 1: Write minimal implementation**

In `src/gui/game_library_gui.mm`:
1. Add UI properties to `GameLibraryController`:
   ```objc
   @property (nonatomic, strong) NSImageView* previewIconView;
   @property (nonatomic, strong) NSTextField* previewTitleLabel;
   @property (nonatomic, strong) NSTextField* previewMetaLabel;
   @property (nonatomic, strong) NSButton* bigLaunchButton;
   ```
2. In `buildWindow`:
   - Adjust table width to `620` and place a detail panel on the right (`NSMakeRect(650, 300, 340, 250)`).
   - In detail panel:
     - 128x128 `NSImageView` with rounded corners.
     - Bold title label (`[NSFont boldSystemFontOfSize:15]`).
     - Subtitle label for Content ID & eboot path.
     - Large Play / Run button `[NSButton buttonWithTitle:@"▶ Launch Title" target:self action:@selector(onRun:)]`.
3. Implement `tableViewSelectionDidChange:(NSNotification *)notification`:
   - When selected row changes:
     - Load `g.icon_path` into `previewIconView`.
     - Set `previewTitleLabel.stringValue = @(g.name.c_str())`.
     - Set `previewMetaLabel.stringValue = [NSString stringWithFormat:@"ID: %s | Ver: %s\nSize: %@\nStatus: %s", g.title_id.c_str(), g.app_ver.c_str(), human_size(g.total_bytes), g.status.c_str()]`.
     - Enable `bigLaunchButton`.

- [ ] **Step 2: Build ps4-gui**

Run: `cmake --build build --target ps4-gui ps4-gui-app`
Expected: Build succeeds with 0 warnings/errors.

- [ ] **Step 3: Commit**

```bash
git add src/gui/game_library_gui.mm
git commit -m "feat(gui): add game detail preview pane with artwork and instant launch button"
```

---

### Task 8: GUI App Interactive Controls & Keybindings HUD

**Files:**
- Modify: `src/gui/game_library_gui.mm`

**Interfaces:**
- Consumes: User click on "Controls Guide" button.
- Produces: Modal sheet or window displaying keyboard and mouse mappings to PS4 DualShock 4 controls.

- [ ] **Step 1: Write minimal implementation**

In `src/gui/game_library_gui.mm`:
1. Add a "Controls" button next to Import button in `buildWindow`:
   ```objc
   NSButton* controlsBtn = [NSButton buttonWithTitle:@"Controls Guide…" target:self action:@selector(onShowControls:)];
   controlsBtn.frame = NSMakeRect(156, frame.size.height - 46, 130, 28);
   [root addSubview:controlsBtn];
   ```
2. Implement `- (void)onShowControls:(id)sender`:
   - Create an `NSAlert` or modal window titled `"Keyboard & Mouse Controls Guide"`.
   - Informative text formatted with clean markdown / ASCII columns:
     ```
     DualShock 4 Control     Keyboard & Mouse Mapping
     ─────────────────────────────────────────────────────────────
     Left Analog Stick       W / A / S / D
     Right Analog Stick      Mouse Movement (or I / J / K / L)
     D-Pad (Up/Down/L/R)     Arrow Keys (↑ ↓ ← →)
     Cross (✕)               Spacebar / Enter / Left Click
     Circle (○)              C / Backspace / Right Click
     Square (□)              F / X
     Triangle (△)            E / R
     L1 / R1 Bumpers         Q / Left Shift & Right Shift
     L2 / R2 Triggers        Z / Left Ctrl & V
     Touchpad Motion         Mouse Cursor inside Game Window
     Touchpad Click          Tab Key / Mouse Click
     Options / Start         Escape Key / P
     Share                   Backquote (`) / F1
     Quit / Close Window     Cmd + W or Cmd + Q
     ```
   - Present with `[alert beginSheetModalForWindow:self.window completionHandler:nil];`.

- [ ] **Step 2: Build ps4-gui**

Run: `cmake --build build --target ps4-gui ps4-gui-app`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add src/gui/game_library_gui.mm
git commit -m "feat(gui): add interactive controls and keybindings guide sheet"
```

---

### Task 9: GUI App Log Search, Filtering, Export, and Post-Run Triage Card

**Files:**
- Modify: `src/gui/game_library_gui.mm`

**Interfaces:**
- Consumes: Log output streamed from `ps4-runner`.
- Produces: Live log filter search field, "Copy Log", "Clear Log", "Reveal in Finder" buttons, and post-run status indicator.

- [ ] **Step 1: Write minimal implementation**

In `src/gui/game_library_gui.mm`:
1. Add UI properties to `GameLibraryController`:
   ```objc
   @property (nonatomic, strong) NSSearchField* logSearchField;
   @property (nonatomic, strong) NSPopUpButton* logFilterPopup;
   @property (nonatomic, copy) NSString* filterText;
   ```
2. In `buildWindow`:
   - Above the log scroll view, add:
     - `NSSearchField` (`NSMakeRect(16, 226, 220, 24)`) with action `@selector(onLogFilterChanged:)`.
     - Filter popup: `[All, GPU/PM4, JIT, HLE, Errors]`.
     - "Copy Log" button (`NSMakeRect(W - 320, 224, 90, 26)`).
     - "Clear" button (`NSMakeRect(W - 225, 224, 75, 26)`).
     - "Reveal Logs…" button (`NSMakeRect(W - 145, 224, 115, 26)`).
3. Implement actions:
   - `- (void)onCopyLog:(id)sender`: Copy `_logView.string` to `[NSPasteboard generalPasteboard]`.
   - `- (void)onClearLog:(id)sender`: Clear `_logLines`, `_pendingLogLines`, and `_logView.string`.
   - `- (void)onRevealLogs:(id)sender`:
     ```objc
     std::string root;
     if (repo_root(root)) {
         NSString* logDir = [@(root.c_str()) stringByAppendingPathComponent:@"Logs"];
         [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:logDir]];
     }
     ```
   - In `flushPendingLogs`: apply `_filterText` or selected filter category before rendering to `_logView`.
4. In `terminationHandler`:
   - Update `_statusLabel`:
     - If exit code == 0: green checkmark `[NSString stringWithFormat:@"✓ Finished cleanly (0) - %@", logName]`.
     - If exit code != 0: red alert badge `[NSString stringWithFormat:@"✗ Exited with error (%d) - %@", exitCode, logName]`.

- [ ] **Step 2: Build ps4-gui**

Run: `cmake --build build --target ps4-gui ps4-gui-app`
Expected: Build succeeds with 0 errors.

- [ ] **Step 3: Commit**

```bash
git add src/gui/game_library_gui.mm
git commit -m "feat(gui): add log search, category filtering, clipboard export, and post-run status"
```

---

### Task 10: Complete Verification & Test Suite Gate

**Files:**
- Test all modified files across core, GPU, HLE, GUI, and runner.

**Interfaces:**
- Consumes: Entire project build targets and CTest test suite.
- Produces: Clean build and 100% passing tests (all 28+ tests).

- [ ] **Step 1: Build all targets**

Run: `cmake --build build -j8`
Expected: Clean build of all targets (`ps4_core`, `ps4-runner`, `ps4-gui`, `ps4-gui-app`, all tests) with 0 errors.

- [ ] **Step 2: Run CTest test suite**

Run: `ctest --test-dir build --output-on-failure -j8`
Expected: **28 / 28 tests passed (100% pass rate)**.

- [ ] **Step 3: Run interactive verification test**

Run a headless session of Thomas Was Alone and Flappy Bird to verify input state handling:
```bash
build/ps4-runner --headless --budget 50000000 "games/thomas_was_alone/EP4395-CUSA00909_00-THOMASWASALONEFG/eboot.bin"
```
Expected: Executes cleanly, advances frames, records input pad states, and exits without assertion failures.

- [ ] **Step 4: Final commit and summary**

```bash
git commit --allow-empty -m "chore: verify keyboard/mouse input and GUI updates pass all gates"
```
