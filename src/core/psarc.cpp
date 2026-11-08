#include "ps4/psarc.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <streambuf>
#include <unistd.h>
#include <zlib.h>

namespace ps4 {

namespace {

u64 be(const u8* p, int n) {
    u64 v = 0;
    for (int i = 0; i < n; ++i) v = (v << 8) | p[i];
    return v;
}

}  // namespace

PsarcArchive::~PsarcArchive() {
    if (fd_ >= 0) ::close(fd_);
}

std::string PsarcArchive::normalize(const std::string& name) {
    std::string s;
    s.reserve(name.size() + 1);
    if (name.empty() || name[0] != '/') s.push_back('/');
    for (char c : name) {
        const char l = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (l == '/' && !s.empty() && s.back() == '/') continue;  // collapse "//"
        s.push_back(l == '\\' ? '/' : l);
    }
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    return s;
}

bool PsarcArchive::read_block(u32 block, u64 file_offset, std::vector<u8>& out) const {
    const u32 stored = zsizes_[block] == 0 ? block_size_ : zsizes_[block];
    std::vector<u8> raw(stored);
    const ssize_t got = ::pread(fd_, raw.data(), stored, static_cast<off_t>(file_offset));
    if (got <= 0) return false;
    raw.resize(static_cast<size_t>(got));
    // A block shorter than block_size_ that starts with a zlib header is
    // compressed; anything else (including a full-size block) is stored as is.
    if (zsizes_[block] != 0 && raw.size() >= 2 && raw[0] == 0x78 &&
        ((raw[0] << 8) | raw[1]) % 31 == 0) {
        out.resize(block_size_);
        uLongf dlen = block_size_;
        if (uncompress(out.data(), &dlen, raw.data(), static_cast<uLong>(raw.size())) == Z_OK) {
            out.resize(dlen);
            return true;
        }
    }
    out = std::move(raw);
    return true;
}

u64 PsarcArchive::read(const Entry& e, u64 pos, u8* dst, u64 len) const {
    if (pos >= e.size) return 0;
    len = std::min(len, e.size - pos);
    u64 done = 0;
    std::vector<u8> buf;
    while (done < len) {
        const u64 at = pos + done;
        const u32 k = static_cast<u32>(at / block_size_);
        const u32 blk = e.first_block + k;
        if (blk >= zsizes_.size()) break;
        const u64 foff = e.offset + (block_start_[blk] - block_start_[e.first_block]);
        if (!read_block(blk, foff, buf)) break;
        const u64 in_blk = at - static_cast<u64>(k) * block_size_;
        if (in_blk >= buf.size()) break;
        const u64 n = std::min<u64>(len - done, buf.size() - in_blk);
        std::memcpy(dst + done, buf.data() + in_blk, n);
        done += n;
    }
    return done;
}

namespace {

// Seekable read-only streambuf over one archive entry, caching one block.
class EntryBuf : public std::streambuf {
public:
    EntryBuf(std::shared_ptr<const PsarcArchive> ar, PsarcArchive::Entry e)
        : ar_(std::move(ar)), e_(e), buf_(ar_->block_size()) {}

protected:
    int_type underflow() override {
        if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
        const u64 pos = base_ + static_cast<u64>(egptr() - eback());
        return fill(pos) ? traits_type::to_int_type(*gptr()) : traits_type::eof();
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode) override {
        const i64 cur = static_cast<i64>(base_) + static_cast<i64>(gptr() - eback());
        i64 target = off;
        if (dir == std::ios_base::cur) target += cur;
        else if (dir == std::ios_base::end) target += static_cast<i64>(e_.size);
        return seekpos(pos_type(target), std::ios_base::in);
    }

    pos_type seekpos(pos_type p, std::ios_base::openmode) override {
        const i64 target = static_cast<i64>(p);
        if (target < 0 || static_cast<u64>(target) > e_.size) return pos_type(off_type(-1));
        const u64 t = static_cast<u64>(target);
        if (t >= base_ && t < base_ + static_cast<u64>(egptr() - eback())) {
            setg(eback(), eback() + (t - base_), egptr());
        } else {
            base_ = t;
            setg(buf_.data(), buf_.data(), buf_.data());  // empty; next read refills at t
        }
        return pos_type(target);
    }

private:
    bool fill(u64 pos) {
        const u64 bs = ar_->block_size();
        const u64 start = pos - pos % bs;
        const u64 n = ar_->read(e_, start, reinterpret_cast<u8*>(buf_.data()), bs);
        if (n == 0 || pos - start >= n) {
            base_ = pos;
            setg(buf_.data(), buf_.data(), buf_.data());
            return false;
        }
        base_ = start;
        setg(buf_.data(), buf_.data() + (pos - start), buf_.data() + n);
        return true;
    }

    std::shared_ptr<const PsarcArchive> ar_;
    PsarcArchive::Entry e_;
    std::vector<char> buf_;
    u64 base_{0};  // entry offset of eback()
};

class EntryStream : public std::istream {
public:
    explicit EntryStream(std::unique_ptr<EntryBuf> b) : std::istream(b.get()), buf_(std::move(b)) {}

private:
    std::unique_ptr<EntryBuf> buf_;
};

}  // namespace

std::unique_ptr<std::istream> PsarcArchive::open_entry(const Entry& e) const {
    return std::make_unique<EntryStream>(std::make_unique<EntryBuf>(shared_from_this(), e));
}

const PsarcArchive::Entry* PsarcArchive::find(const std::string& name) const {
    auto it = files_.find(normalize(name));
    return it == files_.end() ? nullptr : &it->second;
}

bool PsarcArchive::is_directory(const std::string& name) const {
    const std::string n = normalize(name);
    return n == "/" || dirs_.contains(n);
}

std::shared_ptr<PsarcArchive> PsarcArchive::open(const std::filesystem::path& path) {
    auto ar = std::shared_ptr<PsarcArchive>(new PsarcArchive());
    ar->fd_ = ::open(path.c_str(), O_RDONLY);
    if (ar->fd_ < 0) return nullptr;
    const int fd = ar->fd_;

    u8 hdr[32];
    if (::pread(fd, hdr, sizeof(hdr), 0) != static_cast<ssize_t>(sizeof(hdr))) return nullptr;
    if (std::memcmp(hdr, "PSAR", 4) != 0) return nullptr;
    const u32 toc_len = static_cast<u32>(be(hdr + 12, 4));
    const u32 entry_size = static_cast<u32>(be(hdr + 16, 4));
    const u32 count = static_cast<u32>(be(hdr + 20, 4));
    const u32 block_size = static_cast<u32>(be(hdr + 24, 4));
    if (std::memcmp(hdr + 8, "zlib", 4) != 0 && std::memcmp(hdr + 8, "\0\0\0\0", 4) != 0) {
        std::fprintf(stderr, "[PSARC] %s: unsupported compression '%.4s'\n", path.c_str(), hdr + 8);
        return nullptr;
    }
    if (toc_len < 32 || entry_size < 30 || count == 0 || block_size == 0 ||
        static_cast<u64>(count) * entry_size > toc_len - 32) {
        return nullptr;
    }

    std::vector<u8> toc(toc_len - 32);
    if (::pread(fd, toc.data(), toc.size(), 32) != static_cast<ssize_t>(toc.size())) return nullptr;

    ar->path_ = path;
    ar->block_size_ = block_size;

    std::vector<Entry> entries(count);
    for (u32 i = 0; i < count; ++i) {
        const u8* e = toc.data() + static_cast<size_t>(i) * entry_size;
        entries[i].first_block = static_cast<u32>(be(e + 16, 4));
        entries[i].size = be(e + 20, 5);
        entries[i].offset = be(e + 25, 5);
    }
    const int zbytes = block_size <= 0x10000 ? 2 : (block_size <= 0x1000000 ? 3 : 4);
    for (size_t p = static_cast<size_t>(count) * entry_size; p + zbytes <= toc.size(); p += zbytes) {
        ar->zsizes_.push_back(static_cast<u32>(be(toc.data() + p, zbytes)));
    }
    ar->block_start_.resize(ar->zsizes_.size() + 1, 0);
    for (size_t i = 0; i < ar->zsizes_.size(); ++i) {
        ar->block_start_[i + 1] = ar->block_start_[i] + (ar->zsizes_[i] ? ar->zsizes_[i] : block_size);
    }
    for (const Entry& e : entries) {
        if (e.size > 0 && e.first_block >= ar->zsizes_.size()) return nullptr;
    }

    // Manifest: entry 0, newline-separated names for entries 1..count-1.
    std::string manifest(entries[0].size, '\0');
    if (ar->read(entries[0], 0, reinterpret_cast<u8*>(manifest.data()), manifest.size()) != manifest.size()) {
        return nullptr;
    }
    size_t idx = 1, start = 0;
    while (idx < count && start <= manifest.size()) {
        size_t end = manifest.find('\n', start);
        if (end == std::string::npos) end = manifest.size();
        std::string name = manifest.substr(start, end - start);
        if (!name.empty() && name.back() == '\r') name.pop_back();
        start = end + 1;
        if (name.empty()) continue;
        const std::string n = normalize(name);
        ar->files_[n] = entries[idx++];
        for (size_t s = n.find('/', 1); s != std::string::npos; s = n.find('/', s + 1)) {
            ar->dirs_[n.substr(0, s)] = true;
        }
    }
    return ar;
}

} // namespace ps4
