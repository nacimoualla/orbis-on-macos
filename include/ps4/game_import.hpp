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
    std::string eboot;     // absolute path to the validated eboot (an ELF)
    u64 total_bytes{0};    // recursive size of the title directory
    u32 file_count{0};
    // "ok", "no eboot", "eboot not an ELF", ...
    std::string status;
};

// Finds every candidate title directory directly under `folder`.
// Returns an empty vector if `folder` does not exist or is not a directory.
std::vector<GameEntry> scan_folder(const std::string& folder);

// True if `path` starts with the ELF magic. Dumps in this repo carry both an
// eboot.elf and an eboot.bin and only the former is a real ELF, so choosing by
// filename alone hands the runner something it cannot parse.
bool is_elf(const std::string& path);

} // namespace gui
} // namespace ps4