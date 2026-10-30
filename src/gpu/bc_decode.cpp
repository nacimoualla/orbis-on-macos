// Block-compressed texture decoding (BC1..BC7). The BC1-BC5 formats follow the
// Direct3D block formats. The BC6H and BC7 mode logic, bit layouts and
// partition tables are ported from bcdec.h (v0.985, dual MIT / Unlicense).
// Partition-dependent modes use bcdec's tables: the Khronos BPTC tables are
// known to contain mistakes.

#include "ps4/bc_decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace ps4::bc {
namespace {

u32 le16(const u8* p) { return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8); }

u32 le32(const u8* p) { return le16(p) | (le16(p + 2) << 16); }

/// Reads a block as a bit stream, LSB first. BC6H and BC7 pack their fields
/// this way; the first field starts at bit 0 of byte 0.
class BitReader {
public:
    explicit BitReader(const u8* p) : p_(p) {}

    int read(int n) {
        int v = 0;
        for (int i = 0; i < n; ++i, ++pos_) v |= ((p_[pos_ >> 3] >> (pos_ & 7)) & 1) << i;
        return v;
    }

    /// Reads n bits and reverses their order. BC6H modes 12 and 13 store some
    /// high endpoint bits this way.
    int read_r(int n) {
        int bits = read(n);
        int r = 0;
        for (int i = 0; i < n; ++i, bits >>= 1) r = (r << 1) | (bits & 1);
        return r;
    }

private:
    const u8* p_;
    int pos_{0};
};

/// Sign-extends the low `bits` bits of v.
int sext(int v, int bits) {
    const unsigned shift = 32u - static_cast<unsigned>(bits);
    return static_cast<int>(static_cast<unsigned>(v) << shift) >> shift;
}

/// n-bit unsigned value widened to 8 bits by replicating its high bits into
/// the low end, so the full range maps onto 0..255.
u32 expand8(u32 v, u32 n) { return (v << (8 - n)) | (v >> (2 * n - 8)); }

/// IEEE half-precision bit pattern to float (no inf/NaN inputs reach here).
float half_to_float(u32 h) {
    const float sign = (h & 0x8000u) ? -1.0f : 1.0f;
    const i32 e = static_cast<i32>((h >> 10) & 31u);
    const u32 m = h & 1023u;
    if (e == 0) return sign * std::ldexp(static_cast<float>(m), -24);
    return sign * std::ldexp(static_cast<float>(1024u + m), e - 25);
}

// ---------------------------------------------------------------------------
// BC1 colour block, also the colour half of BC2 and BC3.
// ---------------------------------------------------------------------------

/// Two RGB565 endpoints and sixteen 2-bit indices. When `transparent` is set
/// and c0 <= c1, the block uses the 3-colour mode, where index 3 is transparent
/// black. BC2 and BC3 always use the 4-colour mode, so they pass false.
void decode_color_block(const u8* p, bool transparent, float out[16][4]) {
    const u32 ends[2] = {le16(p), le16(p + 2)};
    u32 pal[4][4];
    for (u32 e = 0; e < 2; ++e) {
        pal[e][0] = expand8((ends[e] >> 11) & 31u, 5);
        pal[e][1] = expand8((ends[e] >> 5) & 63u, 6);
        pal[e][2] = expand8(ends[e] & 31u, 5);
        pal[e][3] = 255;
    }
    if (ends[0] > ends[1] || !transparent) {
        for (u32 k = 0; k < 3; ++k) {
            pal[2][k] = (2 * pal[0][k] + pal[1][k]) / 3;
            pal[3][k] = (pal[0][k] + 2 * pal[1][k]) / 3;
        }
        pal[2][3] = 255;
        pal[3][3] = 255;
    } else {
        for (u32 k = 0; k < 3; ++k) {
            pal[2][k] = (pal[0][k] + pal[1][k]) / 2;
            pal[3][k] = 0;
        }
        pal[2][3] = 255;
        pal[3][3] = 0;
    }
    const u32 idx = le32(p + 4);
    for (u32 i = 0; i < 16; ++i) {
        const u32 s = (idx >> (2 * i)) & 3u;
        for (u32 k = 0; k < 4; ++k) out[i][k] = static_cast<float>(pal[s][k]) / 255.0f;
    }
}

// ---------------------------------------------------------------------------
// BC4 single-channel block: also the alpha of BC3.
// ---------------------------------------------------------------------------

/// Two endpoints and sixteen 3-bit indices. If endpoint 0 > endpoint 1 the
/// block uses an 8-entry ramp. Otherwise it uses a 6-entry ramp whose last two
/// entries are fixed: 0 and 1 for UNORM, -1 and 1 for SNORM.
void decode_channel(const u8* p, bool snorm, float out[16]) {
    i32 a0 = snorm ? static_cast<i32>(static_cast<signed char>(p[0])) : static_cast<i32>(p[0]);
    i32 a1 = snorm ? static_cast<i32>(static_cast<signed char>(p[1])) : static_cast<i32>(p[1]);
    // SNORM -128 is clamped to -127 so that both endpoints map onto [-1,1]. This
    // happens before the ramp is chosen, so (-127, -128) takes the 6-value ramp.
    if (snorm) {
        a0 = std::max(a0, -127);
        a1 = std::max(a1, -127);
    }
    float e[8];
    e[0] = snorm ? static_cast<float>(a0) / 127.0f : static_cast<float>(a0) / 255.0f;
    e[1] = snorm ? static_cast<float>(a1) / 127.0f : static_cast<float>(a1) / 255.0f;
    if (a0 > a1) {
        for (i32 j = 1; j <= 6; ++j) {
            const float fj = static_cast<float>(j);
            e[1 + j] = ((7.0f - fj) * e[0] + fj * e[1]) / 7.0f;
        }
    } else {
        for (i32 j = 1; j <= 4; ++j) {
            const float fj = static_cast<float>(j);
            e[1 + j] = ((5.0f - fj) * e[0] + fj * e[1]) / 5.0f;
        }
        e[6] = snorm ? -1.0f : 0.0f;
        e[7] = 1.0f;
    }
    u64 bits = 0;
    for (u32 k = 0; k < 6; ++k) bits |= static_cast<u64>(p[2 + k]) << (8 * k);
    for (u32 i = 0; i < 16; ++i) out[i] = e[(bits >> (3 * i)) & 7u];
}

// ---------------------------------------------------------------------------
// BC6H (ported from bcdec.h)
// ---------------------------------------------------------------------------

// partition/anchor tables from bcdec.h by Sergii Kudlai, MIT/Unlicense
constexpr u8 kBc6Partition2[32][16] = {
    {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},  // 0
    {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},  // 1
    {0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1},  // 2
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1},  // 3
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1},  // 4
    {0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},  // 5
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},  // 6
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1},  // 7
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1},  // 8
    {0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // 9
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1},  // 10
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1},  // 11
    {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // 12
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},  // 13
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // 14
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},  // 15
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1},  // 16
    {0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},  // 17
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0},  // 18
    {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0},  // 19
    {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},  // 20
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0},  // 21
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},  // 22
    {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1},  // 23
    {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},  // 24
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},  // 25
    {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0},  // 26
    {0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0},  // 27
    {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0},  // 28
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  // 29
    {0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0},  // 30
    {0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},  // 31
};

constexpr u8 kBc6Anchor2[32][2] = {
    {0, 15},  // 0
    {0, 15},  // 1
    {0, 15},  // 2
    {0, 15},  // 3
    {0, 15},  // 4
    {0, 15},  // 5
    {0, 15},  // 6
    {0, 15},  // 7
    {0, 15},  // 8
    {0, 15},  // 9
    {0, 15},  // 10
    {0, 15},  // 11
    {0, 15},  // 12
    {0, 15},  // 13
    {0, 15},  // 14
    {0, 15},  // 15
    {0, 15},  // 16
    {0, 2},  // 17
    {0, 8},  // 18
    {0, 2},  // 19
    {0, 2},  // 20
    {0, 8},  // 21
    {0, 8},  // 22
    {0, 15},  // 23
    {0, 2},  // 24
    {0, 8},  // 25
    {0, 2},  // 26
    {0, 2},  // 27
    {0, 8},  // 28
    {0, 8},  // 29
    {0, 2},  // 30
    {0, 2},  // 31
};

constexpr u8 kBc7Partition2[64][16] = {
    {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},  // 0
    {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},  // 1
    {0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1},  // 2
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1},  // 3
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1},  // 4
    {0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},  // 5
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},  // 6
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1},  // 7
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1},  // 8
    {0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // 9
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1},  // 10
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1},  // 11
    {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // 12
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},  // 13
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // 14
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},  // 15
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1},  // 16
    {0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},  // 17
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0},  // 18
    {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0},  // 19
    {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},  // 20
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0},  // 21
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},  // 22
    {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1},  // 23
    {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},  // 24
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},  // 25
    {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0},  // 26
    {0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0},  // 27
    {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0},  // 28
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  // 29
    {0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0},  // 30
    {0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},  // 31
    {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},  // 32
    {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},  // 33
    {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0},  // 34
    {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0},  // 35
    {0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0},  // 36
    {0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0},  // 37
    {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1},  // 38
    {0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1},  // 39
    {0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0},  // 40
    {0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0},  // 41
    {0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0},  // 42
    {0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0},  // 43
    {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0},  // 44
    {0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1},  // 45
    {0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1},  // 46
    {0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0},  // 47
    {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},  // 48
    {0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0},  // 49
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0},  // 50
    {0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0},  // 51
    {0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1},  // 52
    {0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1},  // 53
    {0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0},  // 54
    {0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0},  // 55
    {0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1},  // 56
    {0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1},  // 57
    {0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1},  // 58
    {0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1},  // 59
    {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},  // 60
    {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  // 61
    {0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},  // 62
    {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1},  // 63
};

constexpr u8 kBc7Anchor2[64][2] = {
    {0, 15},  // 0
    {0, 15},  // 1
    {0, 15},  // 2
    {0, 15},  // 3
    {0, 15},  // 4
    {0, 15},  // 5
    {0, 15},  // 6
    {0, 15},  // 7
    {0, 15},  // 8
    {0, 15},  // 9
    {0, 15},  // 10
    {0, 15},  // 11
    {0, 15},  // 12
    {0, 15},  // 13
    {0, 15},  // 14
    {0, 15},  // 15
    {0, 15},  // 16
    {0, 2},  // 17
    {0, 8},  // 18
    {0, 2},  // 19
    {0, 2},  // 20
    {0, 8},  // 21
    {0, 8},  // 22
    {0, 15},  // 23
    {0, 2},  // 24
    {0, 8},  // 25
    {0, 2},  // 26
    {0, 2},  // 27
    {0, 8},  // 28
    {0, 8},  // 29
    {0, 2},  // 30
    {0, 2},  // 31
    {0, 15},  // 32
    {0, 15},  // 33
    {0, 6},  // 34
    {0, 8},  // 35
    {0, 2},  // 36
    {0, 8},  // 37
    {0, 15},  // 38
    {0, 15},  // 39
    {0, 2},  // 40
    {0, 8},  // 41
    {0, 2},  // 42
    {0, 2},  // 43
    {0, 2},  // 44
    {0, 15},  // 45
    {0, 15},  // 46
    {0, 6},  // 47
    {0, 6},  // 48
    {0, 2},  // 49
    {0, 6},  // 50
    {0, 8},  // 51
    {0, 15},  // 52
    {0, 15},  // 53
    {0, 2},  // 54
    {0, 2},  // 55
    {0, 15},  // 56
    {0, 15},  // 57
    {0, 15},  // 58
    {0, 15},  // 59
    {0, 15},  // 60
    {0, 2},  // 61
    {0, 2},  // 62
    {0, 15},  // 63
};

constexpr u8 kBc7Partition3[64][16] = {
    {0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2},  // 0
    {0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1},  // 1
    {0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1},  // 2
    {0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1},  // 3
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2},  // 4
    {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2},  // 5
    {0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1},  // 6
    {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1},  // 7
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},  // 8
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2},  // 9
    {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2},  // 10
    {0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2},  // 11
    {0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2},  // 12
    {0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2},  // 13
    {0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2},  // 14
    {0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0},  // 15
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2},  // 16
    {0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0},  // 17
    {0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2},  // 18
    {0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1},  // 19
    {0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2},  // 20
    {0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1},  // 21
    {0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2},  // 22
    {0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0},  // 23
    {0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0},  // 24
    {0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2},  // 25
    {0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0},  // 26
    {0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1},  // 27
    {0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2},  // 28
    {0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2},  // 29
    {0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1},  // 30
    {0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1},  // 31
    {0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2},  // 32
    {0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1},  // 33
    {0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2},  // 34
    {0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0},  // 35
    {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0},  // 36
    {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0},  // 37
    {0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0},  // 38
    {0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1},  // 39
    {0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1},  // 40
    {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2},  // 41
    {0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1},  // 42
    {0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2},  // 43
    {0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1},  // 44
    {0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1},  // 45
    {0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1},  // 46
    {0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1},  // 47
    {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2},  // 48
    {0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1},  // 49
    {0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2},  // 50
    {0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2},  // 51
    {0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2},  // 52
    {0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2},  // 53
    {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2},  // 54
    {0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2},  // 55
    {0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2},  // 56
    {0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2},  // 57
    {0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2},  // 58
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2},  // 59
    {0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1},  // 60
    {0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2},  // 61
    {0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},  // 62
    {0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0},  // 63
};

constexpr u8 kBc7Anchor3[64][3] = {
    {0, 3, 15},  // 0
    {0, 3, 8},  // 1
    {0, 15, 8},  // 2
    {0, 15, 3},  // 3
    {0, 8, 15},  // 4
    {0, 3, 15},  // 5
    {0, 15, 3},  // 6
    {0, 15, 8},  // 7
    {0, 8, 15},  // 8
    {0, 8, 15},  // 9
    {0, 6, 15},  // 10
    {0, 6, 15},  // 11
    {0, 6, 15},  // 12
    {0, 5, 15},  // 13
    {0, 3, 15},  // 14
    {0, 3, 8},  // 15
    {0, 3, 15},  // 16
    {0, 3, 8},  // 17
    {0, 8, 15},  // 18
    {0, 15, 3},  // 19
    {0, 3, 15},  // 20
    {0, 3, 8},  // 21
    {0, 6, 15},  // 22
    {0, 10, 8},  // 23
    {0, 5, 3},  // 24
    {0, 8, 15},  // 25
    {0, 8, 6},  // 26
    {0, 6, 10},  // 27
    {0, 8, 15},  // 28
    {0, 5, 15},  // 29
    {0, 15, 10},  // 30
    {0, 15, 8},  // 31
    {0, 8, 15},  // 32
    {0, 15, 3},  // 33
    {0, 3, 15},  // 34
    {0, 5, 10},  // 35
    {0, 6, 10},  // 36
    {0, 10, 8},  // 37
    {0, 8, 9},  // 38
    {0, 15, 10},  // 39
    {0, 15, 6},  // 40
    {0, 3, 15},  // 41
    {0, 15, 8},  // 42
    {0, 5, 15},  // 43
    {0, 15, 3},  // 44
    {0, 15, 6},  // 45
    {0, 15, 6},  // 46
    {0, 15, 8},  // 47
    {0, 3, 15},  // 48
    {0, 15, 3},  // 49
    {0, 5, 15},  // 50
    {0, 5, 15},  // 51
    {0, 5, 15},  // 52
    {0, 8, 15},  // 53
    {0, 5, 15},  // 54
    {0, 10, 15},  // 55
    {0, 5, 15},  // 56
    {0, 10, 15},  // 57
    {0, 8, 15},  // 58
    {0, 13, 15},  // 59
    {0, 15, 3},  // 60
    {0, 12, 15},  // 61
    {0, 3, 15},  // 62
    {0, 3, 8},  // 63
};

/// True if texel `texel` is the anchor of a subset: its index drops one bit.
bool bc6_is_anchor(int numPartitions, int partition, int texel) {
    if (texel == 0) return true;
    return numPartitions == 1 && texel == kBc6Anchor2[partition][1];
}

/// Unquantizes an endpoint of `bits` bits to a 16-bit magnitude (signed keeps a sign).
int bc6_unquantize(int val, int bits, bool sign) {
    if (!sign) {
        if (bits >= 15) return val;
        if (val == 0) return 0;
        if (val == (1 << bits) - 1) return 0xFFFF;
        return ((val << 16) + 0x8000) >> bits;
    }
    if (bits >= 16) return val;
    int s = 0;
    if (val < 0) {
        s = 1;
        val = -val;
    }
    int unq;
    if (val == 0) unq = 0;
    else if (val >= (1 << (bits - 1)) - 1) unq = 0x7FFF;
    else unq = ((val << 15) + 0x4000) >> (bits - 1);
    return s ? -unq : unq;
}

/// Maps the interpolated 16-bit value to a half-float bit pattern and then to float.
float bc6_finish(int val, bool sign) {
    if (!sign) return half_to_float(static_cast<u32>((val * 31) >> 6));
    val = (val < 0) ? -(((-val) * 31) >> 5) : (val * 31) >> 5;
    const u32 s = val < 0 ? 0x8000u : 0u;
    return half_to_float(s | static_cast<u32>(val < 0 ? -val : val));
}

/// Adds the base endpoint to a delta and wraps to `bits`.
int bc6_transform_inverse(int val, int a0, int bits, bool sign) {
    val = (val + a0) & ((1 << bits) - 1);
    return sign ? sext(val, bits) : val;
}

bool decode_bc6h(const u8* p, bool sign, float out[16][4]) {
    static constexpr int kW[14] = {10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16};   // base bits
    static constexpr int kDR[14] = {5, 6, 5, 4, 4, 5, 6, 5, 5, 6, 10, 9, 8, 4};        // delta R
    static constexpr int kDG[14] = {5, 6, 4, 5, 4, 5, 5, 6, 5, 6, 10, 9, 8, 4};        // delta G
    static constexpr int kDB[14] = {5, 6, 4, 4, 5, 5, 5, 5, 6, 6, 10, 9, 8, 4};        // delta B
    static constexpr int kW3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
    static constexpr int kW4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

    BitReader br(p);
    int mode = br.read(2);
    if (mode > 1) mode |= br.read(3) << 2;  // 5-bit mode code

    int r[4] = {0, 0, 0, 0}, g[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};  // endpoints w,x,y,z
    int partition = 0;
    // Each case reads the endpoint fields in its own bit order, then sets `mode`
    // to the zero-based mode number.
    switch (mode) {
        /* mode 1 */
        case 0b00: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 75 bits (10.555, 10.555, 10.555) */
            g[2] |= br.read(1) << 4;       /* gy[4]   */
            b[2] |= br.read(1) << 4;       /* by[4]   */
            b[3] |= br.read(1) << 4;       /* bz[4]   */
            r[0] |= br.read(10);       /* rw[9:0] */
            g[0] |= br.read(10);       /* gw[9:0] */
            b[0] |= br.read(10);       /* bw[9:0] */
            r[1] |= br.read(5);        /* rx[4:0] */
            g[3] |= br.read(1) << 4;       /* gz[4]   */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(5);        /* gx[4:0] */
            b[3] |= br.read(1);            /* bz[0]   */
            g[3] |= br.read(4);        /* gz[3:0] */
            b[1] |= br.read(5);        /* bx[4:0] */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(5);        /* ry[4:0] */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            r[3] |= br.read(5);        /* rz[4:0] */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            partition = br.read(5);    /* d[4:0]  */
            mode = 0;
        } break;

        /* mode 2 */
        case 0b01: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 75 bits (7666, 7666, 7666) */
            g[2] |= br.read(1) << 5;       /* gy[5]   */
            g[3] |= br.read(1) << 4;       /* gz[4]   */
            g[3] |= br.read(1) << 5;       /* gz[5]   */
            r[0] |= br.read(7);        /* rw[6:0] */
            b[3] |= br.read(1);            /* bz[0]   */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(1) << 4;       /* by[4]   */
            g[0] |= br.read(7);        /* gw[6:0] */
            b[2] |= br.read(1) << 5;       /* by[5]   */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            g[2] |= br.read(1) << 4;       /* gy[4]   */
            b[0] |= br.read(7);        /* bw[6:0] */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            b[3] |= br.read(1) << 5;       /* bz[5]   */
            b[3] |= br.read(1) << 4;       /* bz[4]   */
            r[1] |= br.read(6);        /* rx[5:0] */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(6);        /* gx[5:0] */
            g[3] |= br.read(4);        /* gz[3:0] */
            b[1] |= br.read(6);        /* bx[5:0] */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(6);        /* ry[5:0] */
            r[3] |= br.read(6);        /* rz[5:0] */
            partition = br.read(5);    /* d[4:0]  */
            mode = 1;
        } break;

        /* mode 3 */
        case 0b00010: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (11.555, 11.444, 11.444) */
            r[0] |= br.read(10);       /* rw[9:0] */
            g[0] |= br.read(10);       /* gw[9:0] */
            b[0] |= br.read(10);       /* bw[9:0] */
            r[1] |= br.read(5);        /* rx[4:0] */
            r[0] |= br.read(1) << 10;      /* rw[10]  */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(4);        /* gx[3:0] */
            g[0] |= br.read(1) << 10;      /* gw[10]  */
            b[3] |= br.read(1);            /* bz[0]   */
            g[3] |= br.read(4);        /* gz[3:0] */
            b[1] |= br.read(4);        /* bx[3:0] */
            b[0] |= br.read(1) << 10;      /* bw[10]  */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(5);        /* ry[4:0] */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            r[3] |= br.read(5);        /* rz[4:0] */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            partition = br.read(5);    /* d[4:0]  */
            mode = 2;
        } break;

        /* mode 4 */
        case 0b00110: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (11.444, 11.555, 11.444) */
            r[0] |= br.read(10);       /* rw[9:0] */
            g[0] |= br.read(10);       /* gw[9:0] */
            b[0] |= br.read(10);       /* bw[9:0] */
            r[1] |= br.read(4);        /* rx[3:0] */
            r[0] |= br.read(1) << 10;      /* rw[10]  */
            g[3] |= br.read(1) << 4;       /* gz[4]   */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(5);        /* gx[4:0] */
            g[0] |= br.read(1) << 10;      /* gw[10]  */
            g[3] |= br.read(4);        /* gz[3:0] */
            b[1] |= br.read(4);        /* bx[3:0] */
            b[0] |= br.read(1) << 10;      /* bw[10]  */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(4);        /* ry[3:0] */
            b[3] |= br.read(1);            /* bz[0]   */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            r[3] |= br.read(4);        /* rz[3:0] */
            g[2] |= br.read(1) << 4;       /* gy[4]   */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            partition = br.read(5);    /* d[4:0]  */
            mode = 3;
        } break;

        /* mode 5 */
        case 0b01010: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (11.444, 11.444, 11.555) */
            r[0] |= br.read(10);       /* rw[9:0] */
            g[0] |= br.read(10);       /* gw[9:0] */
            b[0] |= br.read(10);       /* bw[9:0] */
            r[1] |= br.read(4);        /* rx[3:0] */
            r[0] |= br.read(1) << 10;      /* rw[10]  */
            b[2] |= br.read(1) << 4;       /* by[4]   */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(4);        /* gx[3:0] */
            g[0] |= br.read(1) << 10;      /* gw[10]  */
            b[3] |= br.read(1);            /* bz[0]   */
            g[3] |= br.read(4);        /* gz[3:0] */
            b[1] |= br.read(5);        /* bx[4:0] */
            b[0] |= br.read(1) << 10;      /* bw[10]  */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(4);        /* ry[3:0] */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            r[3] |= br.read(4);        /* rz[3:0] */
            b[3] |= br.read(1) << 4;       /* bz[4]   */ 
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            partition = br.read(5);    /* d[4:0]  */
            mode = 4;
        } break;

        /* mode 6 */
        case 0b01110: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (9555, 9555, 9555) */
            r[0] |= br.read(9);        /* rw[8:0] */
            b[2] |= br.read(1) << 4;       /* by[4]   */
            g[0] |= br.read(9);        /* gw[8:0] */
            g[2] |= br.read(1) << 4;       /* gy[4]   */
            b[0] |= br.read(9);        /* bw[8:0] */
            b[3] |= br.read(1) << 4;       /* bz[4]   */
            r[1] |= br.read(5);        /* rx[4:0] */
            g[3] |= br.read(1) << 4;       /* gz[4]   */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(5);        /* gx[4:0] */
            b[3] |= br.read(1);            /* bz[0]   */
            g[3] |= br.read(4);        /* gx[3:0] */
            b[1] |= br.read(5);        /* bx[4:0] */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(5);        /* ry[4:0] */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            r[3] |= br.read(5);        /* rz[4:0] */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            partition = br.read(5);    /* d[4:0]  */
            mode = 5;
        } break;

        /* mode 7 */
        case 0b10010: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (8666, 8555, 8555) */
            r[0] |= br.read(8);        /* rw[7:0] */
            g[3] |= br.read(1) << 4;       /* gz[4]   */
            b[2] |= br.read(1) << 4;       /* by[4]   */
            g[0] |= br.read(8);        /* gw[7:0] */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            g[2] |= br.read(1) << 4;       /* gy[4]   */
            b[0] |= br.read(8);        /* bw[7:0] */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            b[3] |= br.read(1) << 4;       /* bz[4]   */
            r[1] |= br.read(6);        /* rx[5:0] */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(5);        /* gx[4:0] */
            b[3] |= br.read(1);            /* bz[0]   */
            g[3] |= br.read(4);        /* gz[3:0] */
            b[1] |= br.read(5);        /* bx[4:0] */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(6);        /* ry[5:0] */
            r[3] |= br.read(6);        /* rz[5:0] */
            partition = br.read(5);    /* d[4:0]  */
            mode = 6;
        } break;

        /* mode 8 */
        case 0b10110: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (8555, 8666, 8555) */
            r[0] |= br.read(8);        /* rw[7:0] */
            b[3] |= br.read(1);            /* bz[0]   */
            b[2] |= br.read(1) << 4;       /* by[4]   */
            g[0] |= br.read(8);        /* gw[7:0] */
            g[2] |= br.read(1) << 5;       /* gy[5]   */
            g[2] |= br.read(1) << 4;       /* gy[4]   */
            b[0] |= br.read(8);        /* bw[7:0] */
            g[3] |= br.read(1) << 5;       /* gz[5]   */
            b[3] |= br.read(1) << 4;       /* bz[4]   */
            r[1] |= br.read(5);        /* rx[4:0] */
            g[3] |= br.read(1) << 4;       /* gz[4]   */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(6);        /* gx[5:0] */
            g[3] |= br.read(4);        /* zx[3:0] */
            b[1] |= br.read(5);        /* bx[4:0] */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(5);        /* ry[4:0] */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            r[3] |= br.read(5);        /* rz[4:0] */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            partition = br.read(5);    /* d[4:0]  */
            mode = 7;
        } break;

        /* mode 9 */
        case 0b11010: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (8555, 8555, 8666) */
            r[0] |= br.read(8);        /* rw[7:0] */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(1) << 4;       /* by[4]   */
            g[0] |= br.read(8);        /* gw[7:0] */
            b[2] |= br.read(1) << 5;       /* by[5]   */
            g[2] |= br.read(1) << 4;       /* gy[4]   */
            b[0] |= br.read(8);        /* bw[7:0] */
            b[3] |= br.read(1) << 5;       /* bz[5]   */
            b[3] |= br.read(1) << 4;       /* bz[4]   */
            r[1] |= br.read(5);        /* bw[4:0] */
            g[3] |= br.read(1) << 4;       /* gz[4]   */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(5);        /* gx[4:0] */
            b[3] |= br.read(1);            /* bz[0]   */
            g[3] |= br.read(4);        /* gz[3:0] */
            b[1] |= br.read(6);        /* bx[5:0] */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(5);        /* ry[4:0] */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            r[3] |= br.read(5);        /* rz[4:0] */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            partition = br.read(5);    /* d[4:0]  */
            mode = 8;
        } break;

        /* mode 10 */
        case 0b11110: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (6666, 6666, 6666) */
            r[0] |= br.read(6);        /* rw[5:0] */
            g[3] |= br.read(1) << 4;       /* gz[4]   */
            b[3] |= br.read(1);            /* bz[0]   */
            b[3] |= br.read(1) << 1;       /* bz[1]   */
            b[2] |= br.read(1) << 4;       /* by[4]   */
            g[0] |= br.read(6);        /* gw[5:0] */
            g[2] |= br.read(1) << 5;       /* gy[5]   */
            b[2] |= br.read(1) << 5;       /* by[5]   */
            b[3] |= br.read(1) << 2;       /* bz[2]   */
            g[2] |= br.read(1) << 4;       /* gy[4]   */
            b[0] |= br.read(6);        /* bw[5:0] */
            g[3] |= br.read(1) << 5;       /* gz[5]   */
            b[3] |= br.read(1) << 3;       /* bz[3]   */
            b[3] |= br.read(1) << 5;       /* bz[5]   */
            b[3] |= br.read(1) << 4;       /* bz[4]   */
            r[1] |= br.read(6);        /* rx[5:0] */
            g[2] |= br.read(4);        /* gy[3:0] */
            g[1] |= br.read(6);        /* gx[5:0] */
            g[3] |= br.read(4);        /* gz[3:0] */
            b[1] |= br.read(6);        /* bx[5:0] */
            b[2] |= br.read(4);        /* by[3:0] */
            r[2] |= br.read(6);        /* ry[5:0] */
            r[3] |= br.read(6);        /* rz[5:0] */
            partition = br.read(5);    /* d[4:0]  */
            mode = 9;
        } break;

        /* mode 11 */
        case 0b00011: {
            /* Partitition indices: 63 bits
               Partition: 0 bits
               Color Endpoints: 60 bits (10.10, 10.10, 10.10) */
            r[0] |= br.read(10);       /* rw[9:0] */
            g[0] |= br.read(10);       /* gw[9:0] */
            b[0] |= br.read(10);       /* bw[9:0] */
            r[1] |= br.read(10);       /* rx[9:0] */
            g[1] |= br.read(10);       /* gx[9:0] */
            b[1] |= br.read(10);       /* bx[9:0] */
            mode = 10;
        } break;

        /* mode 12 */
        case 0b00111: {
            /* Partitition indices: 63 bits
               Partition: 0 bits
               Color Endpoints: 60 bits (11.9, 11.9, 11.9) */
            r[0] |= br.read(10);       /* rw[9:0] */
            g[0] |= br.read(10);       /* gw[9:0] */
            b[0] |= br.read(10);       /* bw[9:0] */
            r[1] |= br.read(9);        /* rx[8:0] */
            r[0] |= br.read(1) << 10;      /* rw[10]  */
            g[1] |= br.read(9);        /* gx[8:0] */
            g[0] |= br.read(1) << 10;      /* gw[10]  */
            b[1] |= br.read(9);        /* bx[8:0] */
            b[0] |= br.read(1) << 10;      /* bw[10]  */
            mode = 11;
        } break;

        /* mode 13 */
        case 0b01011: {
            /* Partitition indices: 63 bits
               Partition: 0 bits
               Color Endpoints: 60 bits (12.8, 12.8, 12.8) */
            r[0] |= br.read(10);       /* rw[9:0] */
            g[0] |= br.read(10);       /* gw[9:0] */
            b[0] |= br.read(10);       /* bw[9:0] */
            r[1] |= br.read(8);        /* rx[7:0] */
            r[0] |= br.read_r(2) << 10;/* rx[10:11] */
            g[1] |= br.read(8);        /* gx[7:0] */
            g[0] |= br.read_r(2) << 10;/* gx[10:11] */
            b[1] |= br.read(8);        /* bx[7:0] */
            b[0] |= br.read_r(2) << 10;/* bx[10:11] */
            mode = 12;
        } break;

        /* mode 14 */
        case 0b01111: {
            /* Partitition indices: 63 bits
               Partition: 0 bits
               Color Endpoints: 60 bits (16.4, 16.4, 16.4) */
            r[0] |= br.read(10);       /* rw[9:0] */
            g[0] |= br.read(10);       /* gw[9:0] */
            b[0] |= br.read(10);       /* bw[9:0] */
            r[1] |= br.read(4);        /* rx[3:0] */
            r[0] |= br.read_r(6) << 10;/* rw[10:15] */
            g[1] |= br.read(4);        /* gx[3:0] */
            g[0] |= br.read_r(6) << 10;/* gw[10:15] */
            b[1] |= br.read(4);        /* bx[3:0] */
            b[0] |= br.read_r(6) << 10;/* bw[10:15] */
            mode = 13;
        } break;
    default:  // reserved codes decode to black (RGB zero)
        for (u32 i = 0; i < 16; ++i) {
            out[i][0] = out[i][1] = out[i][2] = 0.0f;
            out[i][3] = 1.0f;
        }
        return true;
    }

    // Modes 0-9 use two subsets (the first partition selects the shape). Modes 10-13 use one.
    const int numPartitions = mode >= 10 ? 0 : 1;
    const int numEndpoints = (numPartitions + 1) * 2;
    const int bits0 = kW[mode];
    if (sign) {
        r[0] = sext(r[0], bits0);
        g[0] = sext(g[0], bits0);
        b[0] = sext(b[0], bits0);
    }
    // Modes 10 (and 9) store the second endpoint raw. Other modes store deltas.
    if ((mode != 9 && mode != 10) || sign) {
        for (int i = 1; i < numEndpoints; ++i) {
            r[i] = sext(r[i], kDR[mode]);
            g[i] = sext(g[i], kDG[mode]);
            b[i] = sext(b[i], kDB[mode]);
        }
    }
    if (mode != 9 && mode != 10) {
        for (int i = 1; i < numEndpoints; ++i) {
            r[i] = bc6_transform_inverse(r[i], r[0], bits0, sign);
            g[i] = bc6_transform_inverse(g[i], g[0], bits0, sign);
            b[i] = bc6_transform_inverse(b[i], b[0], bits0, sign);
        }
    }
    for (int i = 0; i < numEndpoints; ++i) {
        r[i] = bc6_unquantize(r[i], bits0, sign);
        g[i] = bc6_unquantize(g[i], bits0, sign);
        b[i] = bc6_unquantize(b[i], bits0, sign);
    }

    const int* weights = mode >= 10 ? kW4 : kW3;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int texel = i * 4 + j;
            const int subset = numPartitions ? kBc6Partition2[partition][texel] : 0;
            // An anchor index has one bit less because its MSB is implied zero.
            const bool anchor = bc6_is_anchor(numPartitions, partition, texel);
            const int index = br.read(mode >= 10 ? (anchor ? 3 : 4) : (anchor ? 2 : 3));
            const int w = weights[index];
            const int e0 = subset * 2, e1 = subset * 2 + 1;
            const int rv = (r[e0] * (64 - w) + r[e1] * w + 32) >> 6;
            const int gv = (g[e0] * (64 - w) + g[e1] * w + 32) >> 6;
            const int bv = (b[e0] * (64 - w) + b[e1] * w + 32) >> 6;
            out[texel][0] = bc6_finish(rv, sign);
            out[texel][1] = bc6_finish(gv, sign);
            out[texel][2] = bc6_finish(bv, sign);
            out[texel][3] = 1.0f;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// BC7 (ported from bcdec.h)
// ---------------------------------------------------------------------------

bool decode_bc7(const u8* p, float out[16][4]) {
    static constexpr int kRgbBits[8] = {4, 6, 5, 7, 5, 7, 7, 5};
    static constexpr int kAlphaBits[8] = {0, 0, 0, 0, 6, 8, 7, 5};
    static constexpr int kW2[4] = {0, 21, 43, 64};
    static constexpr int kW3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
    static constexpr int kW4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
    constexpr int kHasPBits = 0b11001011;  // modes 0, 1, 3, 6, 7

    BitReader br(p);
    int mode = 0;
    while (mode < 8 && br.read(1) == 0) ++mode;
    if (mode >= 8) {  // reserved: transparent black
        for (u32 i = 0; i < 16; ++i)
            for (u32 k = 0; k < 4; ++k) out[i][k] = 0.0f;
        return true;
    }

    int partition = 0, numPartitions = 1, rotation = 0, indexSelectionBit = 0;
    if (mode == 0 || mode == 1 || mode == 2 || mode == 3 || mode == 7) {
        numPartitions = (mode == 0 || mode == 2) ? 3 : 2;
        partition = br.read(mode == 0 ? 4 : 6);
    }
    const int numEndpoints = numPartitions * 2;
    if (mode == 4 || mode == 5) {
        rotation = br.read(2);
        if (mode == 4) indexSelectionBit = br.read(1);
    }

    // Endpoints are stored per channel: all R, then all G, B and A.
    int endpoints[6][4] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < numEndpoints; ++j) endpoints[j][i] = br.read(kRgbBits[mode]);
    if (kAlphaBits[mode] > 0)
        for (int j = 0; j < numEndpoints; ++j) endpoints[j][3] = br.read(kAlphaBits[mode]);

    if (mode == 0 || mode == 1 || mode == 3 || mode == 6 || mode == 7) {
        for (int i = 0; i < numEndpoints; ++i)
            for (int j = 0; j < 4; ++j) endpoints[i][j] <<= 1;
        if (mode == 1) {  // one shared P-bit per subset
            const int pi = br.read(1), pj = br.read(1);
            for (int k = 0; k < 3; ++k) {
                endpoints[0][k] |= pi;
                endpoints[1][k] |= pi;
                endpoints[2][k] |= pj;
                endpoints[3][k] |= pj;
            }
        } else if (kHasPBits & (1 << mode)) {  // a P-bit per endpoint
            for (int i = 0; i < numEndpoints; ++i) {
                const int pb = br.read(1);
                for (int k = 0; k < 4; ++k) endpoints[i][k] |= pb;
            }
        }
    }

    const int pbit = (kHasPBits >> mode) & 1;
    for (int i = 0; i < numEndpoints; ++i) {
        // Replicate the top bits into the bits revealed by the shift, so the
        // endpoint spans the full 8-bit range.
        int j = kRgbBits[mode] + pbit;
        for (int k = 0; k < 3; ++k) {
            endpoints[i][k] = endpoints[i][k] << (8 - j);
            endpoints[i][k] |= endpoints[i][k] >> j;
        }
        j = kAlphaBits[mode] + pbit;
        endpoints[i][3] = endpoints[i][3] << (8 - j);
        endpoints[i][3] |= endpoints[i][3] >> j;
    }
    if (kAlphaBits[mode] == 0)
        for (int j = 0; j < numEndpoints; ++j) endpoints[j][3] = 0xFF;

    const int indexBits = (mode == 0 || mode == 1) ? 3 : (mode == 6 ? 4 : 2);
    const int indexBits2 = mode == 4 ? 3 : (mode == 5 ? 2 : 0);
    const int* weights = indexBits == 2 ? kW2 : (indexBits == 3 ? kW3 : kW4);
    const int* weights2 = indexBits2 == 2 ? kW2 : kW3;

    // Pass 1 reads all primary indices; pass 2 reads the secondary indices.
    int indices[16];
    for (int texel = 0; texel < 16; ++texel) {
        const bool anchor = texel == 0 || (numPartitions == 2 && texel == kBc7Anchor2[partition][1]) ||
                            (numPartitions == 3 && (texel == kBc7Anchor3[partition][1] || texel == kBc7Anchor3[partition][2]));
        indices[texel] = br.read(indexBits - (anchor ? 1 : 0));
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int texel = i * 4 + j;
            int subset = 0;
            if (numPartitions == 2) subset = kBc7Partition2[partition][texel];
            else if (numPartitions == 3) subset = kBc7Partition3[partition][texel];
            const int e0 = subset * 2, e1 = subset * 2 + 1;
            const int index = indices[texel];
            int r, g, b, a;
            if (indexBits2 == 0) {
                r = (endpoints[e0][0] * (64 - weights[index]) + endpoints[e1][0] * weights[index] + 32) >> 6;
                g = (endpoints[e0][1] * (64 - weights[index]) + endpoints[e1][1] * weights[index] + 32) >> 6;
                b = (endpoints[e0][2] * (64 - weights[index]) + endpoints[e1][2] * weights[index] + 32) >> 6;
                a = (endpoints[e0][3] * (64 - weights[index]) + endpoints[e1][3] * weights[index] + 32) >> 6;
            } else {
                const int index2 = br.read(texel == 0 ? indexBits2 - 1 : indexBits2);
                // The index selection bit decides which index drives colour and which drives alpha.
                const int* wc = indexSelectionBit ? weights2 : weights;
                const int* wa = indexSelectionBit ? weights : weights2;
                const int ic = indexSelectionBit ? index2 : index;
                const int ia = indexSelectionBit ? index : index2;
                auto lerp = [&](int c, const int* w, int idx) {
                    return (endpoints[e0][c] * (64 - w[idx]) + endpoints[e1][c] * w[idx] + 32) >> 6;
                };
                r = lerp(0, wc, ic);
                g = lerp(1, wc, ic);
                b = lerp(2, wc, ic);
                a = lerp(3, wa, ia);
            }
            switch (rotation) {  // rotation swaps alpha with one colour channel
                case 1: std::swap(a, r); break;
                case 2: std::swap(a, g); break;
                case 3: std::swap(a, b); break;
                default: break;
            }
            out[texel][0] = static_cast<float>(r) / 255.0f;
            out[texel][1] = static_cast<float>(g) / 255.0f;
            out[texel][2] = static_cast<float>(b) / 255.0f;
            out[texel][3] = static_cast<float>(a) / 255.0f;
        }
    }
    return true;
}

} // namespace

u32 block_bytes(u32 dfmt) {
    switch (dfmt) {
    case 35: case 38: return 8;                                 // BC1, BC4
    case 36: case 37: case 39: case 40: case 41: return 16;     // BC2, BC3, BC5, BC6H, BC7
    default: return 0;
    }
}

bool decode_block(u32 dfmt, u32 nfmt, const u8* b, float out[16][4]) {
    const bool snorm = nfmt == 1;
    switch (dfmt) {
    case 35:  // BC1: 1-bit alpha in the transparent 3-colour mode
        decode_color_block(b, true, out);
        return true;
    case 36: {  // BC2: explicit 4-bit alpha, texel i in nibble i
        decode_color_block(b + 8, false, out);
        for (u32 i = 0; i < 16; ++i) out[i][3] = static_cast<float>((b[i >> 1] >> (4 * (i & 1))) & 15u) / 15.0f;
        return true;
    }
    case 37: {  // BC3: interpolated alpha block
        decode_color_block(b + 8, false, out);
        float a[16];
        decode_channel(b, false, a);
        for (u32 i = 0; i < 16; ++i) out[i][3] = a[i];
        return true;
    }
    case 38: {  // BC4: red only; green and blue are 0, alpha 1
        float r[16];
        decode_channel(b, snorm, r);
        for (u32 i = 0; i < 16; ++i) {
            out[i][0] = r[i];
            out[i][1] = out[i][2] = 0.0f;
            out[i][3] = 1.0f;
        }
        return true;
    }
    case 39: {  // BC5: red and green; blue 0, alpha 1
        float r[16], g[16];
        decode_channel(b, snorm, r);
        decode_channel(b + 8, snorm, g);
        for (u32 i = 0; i < 16; ++i) {
            out[i][0] = r[i];
            out[i][1] = g[i];
            out[i][2] = 0.0f;
            out[i][3] = 1.0f;
        }
        return true;
    }
    case 40: return decode_bc6h(b, snorm, out);
    case 41: return decode_bc7(b, out);
    default: return false;
    }
}

} // namespace ps4::bc
