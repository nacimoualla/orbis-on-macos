// Game import scanning for the library GUI.
//
// A PS4 title as it appears in a dump is a directory named
// <content-id>_<something> containing eboot.bin. scan_folder() looks for those
// and reports enough metadata for the GUI to list them; it does not copy
// anything, so "import" here means "discover and validate", not "ingest into
// some content store".

#pragma once

#include <string>
#include <vector>

#include "ps4/common.hpp"

namespace ps4 {
namespace gui {

struct GameEntry {
    std::string id;        // directory name, e.g. ED1633-NPXX51362_00-...
    std::string name;      // from the SCE header when readable, else empty
    std::string root;      // absolute path to the game directory
    std::string eboot;     // absolute path to eboot.bin
    u64 total_bytes{0};    // recursive size of the title directory
    u32 file_count{0};
    std::string status;    // "ok", "no eboot.bin", "unreadable", ...
};

// Finds every candidate title directory directly under `folder`.
// Returns an empty vector if `folder` does not exist or is not a directory.
std::vector<GameEntry> scan_folder(const std::string& folder);

} // namespace gui
} // namespace ps4