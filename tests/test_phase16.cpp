// Game-import scanner tests.
//
// The important case is the real dump layout, because a name-pattern heuristic
// gets it exactly backwards:
//
//   games/<publisher>/<content-id>_<serial>/eboot.bin
//
// The publisher directory ("flappy_bird") is not a title; the content-id
// directory ("ED1633-NPXX51362_00-...") is. So the scanner locates titles by
// finding eboot.bin rather than by matching directory names.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "ps4/game_import.hpp"

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    std::printf("  %s %s\n", cond ? "\xE2\x9C\x93" : "\xE2\x9C\x97", what);
    if (!cond) ++g_failures;
}

void mkdirp(const std::string& p) { ::mkdir(p.c_str(), 0755); }

std::string temp_root() {
    char tmpl[] = "/tmp/ps4_gui_test_XXXXXX";
    char* made = ::mkdtemp(tmpl);
    return made ? std::string(made) : std::string();
}

void touch(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "wb");
    if (f) { std::fputs("x", f); std::fclose(f); }
}

} // namespace

int main() {
    std::printf("[PHASE 16] Game import scanner\n");

    // --- Test 1: real repo layout ---
    std::printf("  -- real dump layout --\n");
    {
        auto games = ps4::gui::scan_folder("games");
        bool found_flappy = false;
        for (const auto& g : games) {
            if (g.id.find("ED1633-NPXX51362") != std::string::npos) {
                found_flappy = true;
                check(!g.eboot.empty(), "flappy eboot path is non-empty");
                check(g.status == "ok", "flappy status is ok");
                check(g.total_bytes > 0, "flappy size is non-zero");
                check(g.file_count > 0, "flappy file count is non-zero");
            }
            // The publisher directory must NOT be reported as its own title.
            check(g.id != "flappy_bird", "publisher dir is not listed as a title");
        }
        check(found_flappy, "found the real content-id title directory");
    }

    // --- Test 2: synthetic nested layout ---
    std::printf("  -- synthetic nested layout --\n");
    {
        const std::string root = temp_root();
        if (root.empty()) { check(false, "mkdtemp"); return 1; }
        const std::string title = root + "/CUSA12345_00-ABC12345678";
        mkdirp(title);
        touch(title + "/eboot.bin");

        auto games = ps4::gui::scan_folder(root);
        check(games.size() == 1, "exactly one title found");
        if (!games.empty()) {
            check(games[0].id == "CUSA12345_00-ABC12345678", "id is the content-id dir");
            check(games[0].status == "ok", "status ok");
        }

        // A title's own subdirectory must not be reported as another title.
        mkdirp(title + "/data");
        touch(title + "/data/blob.bin");
        auto again = ps4::gui::scan_folder(root);
        check(again.size() == 1, "does not descend into a found title");

        // eboot.elf should also be accepted.
        const std::string root2 = temp_root();
        mkdirp(root2 + "/CUSA99999_00-X");
        touch(root2 + "/CUSA99999_00-X/eboot.elf");
        auto elf = ps4::gui::scan_folder(root2);
        check(elf.size() == 1 && elf[0].status == "ok", "accepts eboot.elf");

        std::printf("  (temp dirs left in /tmp: %s, %s)\n", root.c_str(), root2.c_str());
    }

    // --- Test 3: empty / missing folder ---
    std::printf("  -- degenerate inputs --\n");
    check(ps4::gui::scan_folder("/nonexistent/xyz").empty(), "missing folder -> empty");
    check(ps4::gui::scan_folder("").empty(), "empty path -> empty");

    std::printf("%s\n", g_failures ? "[PHASE 16] FAILED" : "[PHASE 16] PASSED");
    return g_failures ? 1 : 0;
}
