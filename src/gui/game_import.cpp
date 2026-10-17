#include "ps4/game_import.hpp"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace ps4 {
namespace gui {

namespace {

bool is_dir(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool is_file(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

u64 dir_size(const std::string& dir, u32& files) {
    u64 total = 0;
    DIR* d = ::opendir(dir.c_str());
    if (!d) return 0;
    while (dirent* e = ::readdir(d)) {
        if (std::strcmp(e->d_name, ".") == 0 || std::strcmp(e->d_name, "..") == 0)
            continue;
        const std::string child = dir + "/" + e->d_name;
        struct stat st{};
        if (::stat(child.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            total += dir_size(child, files);
        } else if (S_ISREG(st.st_mode)) {
            total += static_cast<u64>(st.st_size);
            ++files;
        }
    }
    ::closedir(d);
    return total;
}

std::string base_name(const std::string& p) {
    const size_t slash = p.find_last_of('/');
    return (slash == std::string::npos) ? p : p.substr(slash + 1);
}

bool is_elf_file(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    unsigned char m[4] = {0};
    const size_t n = std::fread(m, 1, sizeof(m), f);
    std::fclose(f);
    return n == sizeof(m) && m[0] == 0x7F && m[1] == 'E' && m[2] == 'L' && m[3] == 'F';
}

// A title is any directory that actually contains an eboot, found by walking
// rather than by pattern-matching the directory name.
//
// Naming patterns do not work here: a dump is laid out as
//     <folder>/<publisher>/<content-id>_<serial>/eboot.bin
// so the real title directory is ED1633-NPXX51362_00-TER1520000000000 while its
// parent is "flappy_bird". A "4 letters then digits" heuristic accepts the
// publisher and rejects the title, which is exactly backwards. Both eboot.bin
// and eboot.elf are accepted because dumps ship one or the other.
bool contains_eboot(const std::string& dir, std::string& eboot_out,
                    std::string& status_out) {
    // Prefer whichever candidate is actually an ELF. Naming alone is not enough:
    // this repo's title ships both eboot.elf (magic 7f454c46, loadable) and
    // eboot.bin (magic 4f153d1d, a raw blob the runner rejects with
    // "Invalid ELF magic number"). Both orders are tried so either layout works.
    const std::string candidates[2] = { dir + "/eboot.elf", dir + "/eboot.bin" };
    std::string any_present;
    for (const std::string& c : candidates) {
        if (!is_file(c)) continue;
        if (is_elf_file(c)) { eboot_out = c; status_out = "ok"; return true; }
        if (any_present.empty()) any_present = c;
    }
    if (!any_present.empty()) {
        eboot_out = any_present;
        status_out = "eboot is not an ELF";
        return true; // still a title, but not runnable
    }
    return false;
}

void walk(const std::string& dir, int depth, std::vector<GameEntry>& out) {
    DIR* d = ::opendir(dir.c_str());
    if (!d) return;

    std::vector<std::string> subdirs;
    while (dirent* e = ::readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string child = dir + "/" + e->d_name;
        if (is_dir(child)) subdirs.push_back(child);
    }
    ::closedir(d);
    std::sort(subdirs.begin(), subdirs.end());

    // A directory holding an eboot is a title: record it and do not descend,
    // otherwise a title's own subfolders would be mistaken for more titles.
    std::string eboot, status;
    if (contains_eboot(dir, eboot, status)) {
        GameEntry g;
        g.id = base_name(dir);
        g.root = dir;
        g.eboot = eboot;
        g.status = status;
        g.total_bytes = dir_size(dir, g.file_count);
        out.push_back(std::move(g));
        return;
    }

    if (depth <= 0) return;
    for (const std::string& s : subdirs) walk(s, depth - 1, out);
}

} // namespace

// Public wrapper -- the implementation stays in the anonymous namespace.
bool is_elf(const std::string& path) { return is_elf_file(path); }

std::vector<GameEntry> scan_folder(const std::string& folder) {
    std::vector<GameEntry> out;
    if (!is_dir(folder)) return out;
    // Depth 3 covers <folder>/<publisher>/<title> and <folder>/<title>.
    walk(folder, 3, out);
    std::sort(out.begin(), out.end(),
              [](const GameEntry& a, const GameEntry& b) { return a.id < b.id; });
    return out;
}

} // namespace gui
} // namespace ps4