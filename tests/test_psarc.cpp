// Unit test for the PSARC reader behind sceFiosArchiveMount.
//
// Builds a small archive in a temp file: a zlib-compressed manifest, a stored
// one-block file, and a three-block file mixing compressed, stored and short
// blocks. Checks lookup (case-insensitive, with or without a leading '/'),
// directories, ranged reads across block boundaries, and stream seeking.

#include "ps4/psarc.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <zlib.h>

namespace {

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

constexpr unsigned kBlock = 0x10000;

void put_be(std::vector<unsigned char>& v, unsigned long long x, int n) {
    for (int i = n - 1; i >= 0; --i) v.push_back(static_cast<unsigned char>(x >> (8 * i)));
}

std::vector<unsigned char> zlib_compress(const std::vector<unsigned char>& in) {
    uLongf len = compressBound(static_cast<uLong>(in.size()));
    std::vector<unsigned char> out(len);
    CHECK(compress2(out.data(), &len, in.data(), static_cast<uLong>(in.size()), 9) == Z_OK);
    out.resize(len);
    return out;
}

struct Built {
    std::vector<unsigned char> data;          // blocks, in file order
    std::vector<unsigned> zsizes;             // per block, 0 = full stored block
};

// Append one entry's blocks. `compress[i]` picks compressed or stored per block.
unsigned add_blocks(Built& b, const std::vector<unsigned char>& content, const std::vector<bool>& compress) {
    const unsigned first = static_cast<unsigned>(b.zsizes.size());
    size_t pos = 0, i = 0;
    do {
        const size_t n = std::min<size_t>(kBlock, content.size() - pos);
        std::vector<unsigned char> chunk(content.begin() + pos, content.begin() + pos + n);
        const bool z = i < compress.size() && compress[i];
        std::vector<unsigned char> stored = z ? zlib_compress(chunk) : chunk;
        b.data.insert(b.data.end(), stored.begin(), stored.end());
        b.zsizes.push_back(stored.size() == kBlock ? 0u : static_cast<unsigned>(stored.size()));
        pos += n;
        ++i;
    } while (pos < content.size());
    return first;
}

std::vector<unsigned char> pattern(size_t n, unsigned seed) {
    std::vector<unsigned char> v(n);
    unsigned x = seed;
    for (auto& c : v) {
        x = x * 1103515245u + 12345u;
        c = static_cast<unsigned char>(x >> 16);
    }
    return v;
}

}  // namespace

int main() {
    const std::string manifest_text = "/Dialogue/Hello.wem\n/music/track.wem";
    const std::vector<unsigned char> manifest(manifest_text.begin(), manifest_text.end());
    const std::vector<unsigned char> hello = pattern(1000, 1);
    // 2.5 blocks: compressible first block, random (stored) second, short third.
    std::vector<unsigned char> track(kBlock * 2 + kBlock / 2, 'A');
    const auto noise = pattern(kBlock, 7);
    std::copy(noise.begin(), noise.end(), track.begin() + kBlock);

    Built b;
    const unsigned m_first = add_blocks(b, manifest, {true});
    const unsigned h_first = add_blocks(b, hello, {false});
    const unsigned t_first = add_blocks(b, track, {true, false, true});

    const unsigned count = 3, entry_size = 30;
    const unsigned toc_len = 32 + count * entry_size + static_cast<unsigned>(b.zsizes.size()) * 2;
    std::vector<unsigned char> file;
    file.insert(file.end(), {'P', 'S', 'A', 'R'});
    put_be(file, 0x00010004, 4);
    file.insert(file.end(), {'z', 'l', 'i', 'b'});
    put_be(file, toc_len, 4);
    put_be(file, entry_size, 4);
    put_be(file, count, 4);
    put_be(file, kBlock, 4);
    put_be(file, 3, 4);  // flags: ignore case | absolute paths

    // Block data starts right after the TOC; entries point at their first block.
    std::vector<unsigned long long> block_off(b.zsizes.size());
    unsigned long long off = toc_len;
    for (size_t i = 0; i < b.zsizes.size(); ++i) {
        block_off[i] = off;
        off += b.zsizes[i] ? b.zsizes[i] : kBlock;
    }
    const struct { unsigned first; size_t size; } ents[] = {
        {m_first, manifest.size()}, {h_first, hello.size()}, {t_first, track.size()}};
    for (const auto& e : ents) {
        for (int i = 0; i < 16; ++i) file.push_back(0);  // name MD5, unused by the reader
        put_be(file, e.first, 4);
        put_be(file, e.size, 5);
        put_be(file, block_off[e.first], 5);
    }
    for (unsigned z : b.zsizes) put_be(file, z, 2);
    CHECK(file.size() == toc_len);
    file.insert(file.end(), b.data.begin(), b.data.end());

    const auto path = std::filesystem::temp_directory_path() / "ps4_test_psarc.psarc";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    }

    auto ar = ps4::PsarcArchive::open(path);
    CHECK(ar != nullptr);
    CHECK(ar->file_count() == 2);

    const auto* h = ar->find("dialogue/hello.WEM");
    CHECK(h != nullptr && h->size == hello.size());
    CHECK(ar->find("/DIALOGUE/HELLO.wem") == h);
    CHECK(ar->find("/dialogue/missing.wem") == nullptr);
    CHECK(ar->is_directory("/dialogue"));
    CHECK(ar->is_directory("/Music/"));
    CHECK(!ar->is_directory("/dialogue/hello.wem"));

    std::vector<unsigned char> got(hello.size());
    CHECK(ar->read(*h, 0, got.data(), got.size()) == hello.size());
    CHECK(got == hello);

    const auto* t = ar->find("/music/track.wem");
    CHECK(t != nullptr && t->size == track.size());
    got.assign(track.size(), 0);
    CHECK(ar->read(*t, 0, got.data(), got.size()) == track.size());
    CHECK(got == track);

    // A read spanning the compressed/stored and stored/short block boundaries.
    const unsigned long long span_at = kBlock - 100;
    std::vector<unsigned char> span(kBlock + 200);
    CHECK(ar->read(*t, span_at, span.data(), span.size()) == span.size());
    CHECK(std::memcmp(span.data(), track.data() + span_at, span.size()) == 0);
    // Reads past the end are clipped.
    CHECK(ar->read(*t, track.size() - 10, span.data(), 100) == 10);
    CHECK(ar->read(*t, track.size(), span.data(), 100) == 0);

    // Stream: seek around, read across blocks, hit EOF.
    auto s = ar->open_entry(*t);
    s->seekg(static_cast<std::streamoff>(kBlock * 2 - 4));
    char buf[8];
    s->read(buf, 8);
    CHECK(s->gcount() == 8);
    CHECK(std::memcmp(buf, track.data() + kBlock * 2 - 4, 8) == 0);
    s->seekg(-3, std::ios::end);
    s->read(buf, 8);
    CHECK(s->gcount() == 3);
    CHECK(std::memcmp(buf, track.data() + track.size() - 3, 3) == 0);
    s->clear();
    s->seekg(5);
    s->read(buf, 4);
    CHECK(s->gcount() == 4 && std::memcmp(buf, track.data() + 5, 4) == 0);

    // Not a PSARC.
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "not an archive at all, just text padding it past 32 bytes";
    }
    CHECK(ps4::PsarcArchive::open(path) == nullptr);
    std::filesystem::remove(path);

    std::cout << "test_psarc: OK\n";
    return 0;
}
