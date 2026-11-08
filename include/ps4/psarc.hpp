#pragma once
#include "ps4/common.hpp"
#include <filesystem>
#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ps4 {

// Read-only PlayStation archive (PSARC v1.x, zlib or stored blocks), the
// format FIOS2 mounts with sceFiosArchiveMount. Entry 0 is the manifest: the
// newline-separated names of entries 1..n. Files are decompressed lazily,
// one block at a time, so large audio archives are not loaded up front.
class PsarcArchive : public std::enable_shared_from_this<PsarcArchive> {
public:
    ~PsarcArchive();
    PsarcArchive(const PsarcArchive&) = delete;
    PsarcArchive& operator=(const PsarcArchive&) = delete;

    struct Entry {
        u32 first_block{0};
        u64 size{0};    // uncompressed
        u64 offset{0};  // file offset of the first block
    };

    // Parse the header, TOC and manifest. Nullptr if the file is not a PSARC.
    static std::shared_ptr<PsarcArchive> open(const std::filesystem::path& path);

    // Look up an entry by archive path ("/dialogue/x.wem" or "dialogue/x.wem").
    // Case-insensitive, as the archives' flags request.
    const Entry* find(const std::string& name) const;
    bool is_directory(const std::string& name) const;

    // A seekable stream over one entry's uncompressed bytes.
    std::unique_ptr<std::istream> open_entry(const Entry& e) const;

    // Read up to `len` uncompressed bytes of `e` at `pos`. Returns bytes read.
    u64 read(const Entry& e, u64 pos, u8* dst, u64 len) const;

    const std::filesystem::path& path() const { return path_; }
    size_t file_count() const { return files_.size(); }
    u32 block_size() const { return block_size_; }

private:
    PsarcArchive() = default;
    static std::string normalize(const std::string& name);
    bool read_block(u32 block, u64 file_offset, std::vector<u8>& out) const;

    std::filesystem::path path_;
    u32 block_size_{0};
    int fd_{-1};  // read with pread, so concurrent reads need no lock
    std::vector<u32> zsizes_;  // stored size per block; 0 = full block_size_
    std::vector<u64> block_start_;  // prefix sums of stored block sizes
    std::unordered_map<std::string, Entry> files_;
    std::unordered_map<std::string, bool> dirs_;
};

} // namespace ps4
