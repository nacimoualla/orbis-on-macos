// BC1-BC7 block decoding (ps4/bc_decode.hpp) and the BC path of
// texture_to_rgba8. Blocks are built with BitWriter, so each expected value
// follows from the format definition rather than from the decoder under test.

#include "ps4/bc_decode.hpp"
#include "ps4/gcn_shader.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ps4;

namespace {

int g_failures = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_failures;                                                         \
        }                                                                         \
    } while (0)

bool near(float a, float b) { return std::fabs(a - b) < 1e-3f; }

/// Appends fields LSB first, the order BC6H and BC7 use. The same packing
/// gives BC1 endpoints and index words their little-endian layout.
struct BitWriter {
    u8 b[16]{};
    u32 pos{0};

    void put(u32 v, u32 n) {
        for (u32 i = 0; i < n; ++i, ++pos)
            if ((v >> i) & 1u) b[pos >> 3] = static_cast<u8>(b[pos >> 3] | (1u << (pos & 7)));
    }
};

/// Checks one texel's RGBA against the expected values; reports the caller's line.
void check_rgba_at(int line, const float got[4], float r, float g, float b, float a) {
    const float want[4] = {r, g, b, a};
    for (u32 k = 0; k < 4; ++k)
        if (!near(got[k], want[k])) {
            std::fprintf(stderr, "check_rgba failed at line %d, channel %u: got %f want %f\n", line, k,
                         static_cast<double>(got[k]), static_cast<double>(want[k]));
            ++g_failures;
        }
}
#define check_rgba(...) check_rgba_at(__LINE__, __VA_ARGS__)

/// Packs one BC1 block: two endpoints and a 32-bit index word (2 bits per texel).
void bc1_block(u8* out, u32 c0, u32 c1, u32 idx) {
    BitWriter w;
    w.put(c0, 16);
    w.put(c1, 16);
    w.put(idx, 32);
    std::memcpy(out, w.b, 8);
}

// 2-bit index words: all texels 0, all 1, all 2, all 3.
constexpr u32 kIdx0 = 0x00000000u, kIdx1 = 0x55555555u, kIdx2 = 0xAAAAAAAAu, kIdx3 = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------

void test_bc1() {
    float o[16][4];

    // Opaque (c0 > c1): index 0..3 gives c0, c1, (2c0+c1)/3, (c0+2c1)/3.
    // Red 0xF800 and blue 0x001F give (170,0,85) and (85,0,170) in 8 bits.
    BitWriter w;
    w.put(0xF800, 16);
    w.put(0x001F, 16);
    w.put(0xE4, 32);  // texel 0..3 indices 0,1,2,3 (LSB first)
    CHECK(bc::decode_block(35, 0, w.b, o));
    check_rgba(o[0], 1.0f, 0.0f, 0.0f, 1.0f);
    check_rgba(o[1], 0.0f, 0.0f, 1.0f, 1.0f);
    check_rgba(o[2], 170.0f / 255.0f, 0.0f, 85.0f / 255.0f, 1.0f);
    check_rgba(o[3], 85.0f / 255.0f, 0.0f, 170.0f / 255.0f, 1.0f);
    check_rgba(o[15], 1.0f, 0.0f, 0.0f, 1.0f);  // unset indices are 0

    // 3-colour (c0 <= c1): index 2 is the midpoint, index 3 is transparent.
    // Blue 0x001F (c0) < red 0xF800 (c1).
    BitWriter t;
    t.put(0x001F, 16);
    t.put(0xF800, 16);
    t.put(0xE4, 32);
    CHECK(bc::decode_block(35, 0, t.b, o));
    check_rgba(o[0], 0.0f, 0.0f, 1.0f, 1.0f);
    check_rgba(o[1], 1.0f, 0.0f, 0.0f, 1.0f);
    check_rgba(o[2], 127.0f / 255.0f, 0.0f, 127.0f / 255.0f, 1.0f);
    check_rgba(o[3], 0.0f, 0.0f, 0.0f, 0.0f);

    // Equal endpoints count as the 3-colour case; index 0 is still c0.
    u8 blk[8];
    bc1_block(blk, 0xF800, 0xF800, kIdx0);
    CHECK(bc::decode_block(35, 0, blk, o));
    check_rgba(o[7], 1.0f, 0.0f, 0.0f, 1.0f);
    bc1_block(blk, 0xF800, 0xF800, kIdx3);
    CHECK(bc::decode_block(35, 0, blk, o));
    check_rgba(o[7], 0.0f, 0.0f, 0.0f, 0.0f);
}

void test_bc2() {
    BitWriter w;
    for (u32 i = 0; i < 16; ++i) w.put(i, 4);  // alpha nibble of texel i is i
    w.put(0x0000, 16);                         // c0 black
    w.put(0xF800, 16);                         // c1 red, c0 < c1 but BC2 is 4-colour
    w.put(0x3, 32);                            // texel 0 index 3
    float o[16][4];
    CHECK(bc::decode_block(36, 0, w.b, o));
    // Index 3 of 4-colour black/red is (c0 + 2*c1)/3 = 2/3 red.
    check_rgba(o[0], 2.0f / 3.0f, 0.0f, 0.0f, 0.0f);
    CHECK(near(o[5][3], 5.0f / 15.0f));
    CHECK(near(o[15][3], 1.0f));
    CHECK(near(o[5][0], 0.0f));
}

void test_bc3_alpha() {
    // 8-value alpha ramp (alpha0 > alpha1): 255, 0, then (6a0+a1)/7 ... (a0+6a1)/7.
    BitWriter w;
    w.put(255, 8);
    w.put(0, 8);
    w.put(0, 3);   // texel 0 -> alpha0 = 1.0
    w.put(1, 3);   // texel 1 -> alpha1 = 0.0
    w.put(2, 3);   // texel 2 -> 6/7
    w.put(7, 3);   // texel 3 -> 1/7
    for (u32 i = 4; i < 16; ++i) w.put(0, 3);  // the alpha index field is 48 bits
    w.put(0xFFFF, 16);  // colour: white
    w.put(0xFFFF, 16);
    w.put(0, 32);
    float o[16][4];
    CHECK(bc::decode_block(37, 0, w.b, o));
    check_rgba(o[0], 1.0f, 1.0f, 1.0f, 1.0f);
    check_rgba(o[1], 1.0f, 1.0f, 1.0f, 0.0f);
    CHECK(near(o[2][3], 6.0f / 7.0f));
    CHECK(near(o[3][3], 1.0f / 7.0f));

    // 6-value ramp (alpha0 <= alpha1): index 6 is 0 and index 7 is 255.
    BitWriter s;
    s.put(0, 8);
    s.put(255, 8);
    s.put(6, 3);   // texel 0 -> 0
    s.put(7, 3);   // texel 1 -> 1.0
    s.put(1, 3);   // texel 2 -> alpha1 = 1.0
    for (u32 i = 3; i < 16; ++i) s.put(0, 3);
    s.put(0xFFFF, 16);
    s.put(0xFFFF, 16);
    s.put(0, 32);
    CHECK(bc::decode_block(37, 0, s.b, o));
    CHECK(near(o[0][3], 0.0f));
    CHECK(near(o[1][3], 1.0f));
    CHECK(near(o[2][3], 1.0f));
}

void test_bc4_bc5() {
    float o[16][4];

    // BC4 UNORM: 8-value ramp.
    BitWriter u;
    u.put(255, 8);
    u.put(0, 8);
    u.put(0, 3);
    u.put(1, 3);
    u.put(7, 3);
    u8 blk[16]{};
    std::memcpy(blk, u.b, 8);
    CHECK(bc::decode_block(38, 0, blk, o));
    CHECK(near(o[0][0], 1.0f));
    CHECK(near(o[1][0], 0.0f));
    CHECK(near(o[2][0], 1.0f / 7.0f));
    CHECK(near(o[0][1], 0.0f) && near(o[0][2], 0.0f) && near(o[0][3], 1.0f));

    // BC4 SNORM, 8-value ramp: 127 -> 1.0, 0 -> 0.0.
    BitWriter s8;
    s8.put(127, 8);
    s8.put(0, 8);
    s8.put(0, 3);
    s8.put(1, 3);
    std::memcpy(blk, s8.b, 8);
    CHECK(bc::decode_block(38, 1, blk, o));
    CHECK(near(o[0][0], 1.0f));
    CHECK(near(o[1][0], 0.0f));

    // BC4 SNORM, 6-value ramp: index 6 -> -1.0, index 7 -> +1.0.
    BitWriter s6;
    s6.put(0, 8);
    s6.put(127, 8);
    s6.put(6, 3);
    s6.put(7, 3);
    std::memcpy(blk, s6.b, 8);
    CHECK(bc::decode_block(38, 1, blk, o));
    CHECK(near(o[0][0], -1.0f));
    CHECK(near(o[1][0], 1.0f));

    // SNORM -128 clamps to -1.0 and is the lower endpoint of the 6-value ramp.
    BitWriter neg;
    neg.put(0x80, 8);
    neg.put(0x00, 8);
    neg.put(0, 3);
    std::memcpy(blk, neg.b, 8);
    CHECK(bc::decode_block(38, 1, blk, o));
    CHECK(near(o[0][0], -1.0f));

    // BC5: R from the first block, G from the second.
    BitWriter r, g;
    r.put(255, 8);
    r.put(0, 8);
    r.put(0, 3);
    r.put(1, 3);
    g.put(0, 8);
    g.put(255, 8);
    g.put(0, 3);
    g.put(1, 3);
    u8 both[16]{};
    std::memcpy(both, r.b, 8);
    std::memcpy(both + 8, g.b, 8);
    CHECK(bc::decode_block(39, 0, both, o));
    check_rgba(o[0], 1.0f, 0.0f, 0.0f, 1.0f);
    check_rgba(o[1], 0.0f, 1.0f, 0.0f, 1.0f);

    // BC5 SNORM: R = 1.0 at index 0, G = -1.0 at index 0 (6-value ramp).
    BitWriter rs, gs;
    rs.put(127, 8);
    rs.put(0, 8);
    rs.put(0, 3);
    gs.put(0x80, 8);
    gs.put(0x00, 8);
    gs.put(0, 3);
    std::memcpy(both, rs.b, 8);
    std::memcpy(both + 8, gs.b, 8);
    CHECK(bc::decode_block(39, 1, both, o));
    CHECK(near(o[0][0], 1.0f));
    CHECK(near(o[0][1], -1.0f));
}

void test_bc7_modes() {
    float o[16][4];

    // Reserved: no mode bit set -> all zeros.
    u8 zero[16]{};
    CHECK(bc::decode_block(41, 0, zero, o));
    check_rgba(o[0], 0.0f, 0.0f, 0.0f, 0.0f);
    check_rgba(o[15], 0.0f, 0.0f, 0.0f, 0.0f);

    // Mode 6: one subset, 7-bit endpoints with P-bits, 4-bit indices.
    // Solid colour: every endpoint raw 0x40, P-bit 0 -> 128.
    BitWriter solid;
    solid.put(1u << 6, 7);  // mode marker
    for (u32 k = 0; k < 8; ++k) solid.put(0x40, 7);  // R0 R1 G0 G1 B0 B1 A0 A1
    solid.put(0, 1);
    solid.put(0, 1);
    for (u32 i = 0; i < 16; ++i) solid.put(0, i == 0 ? 3 : 4);
    CHECK(bc::decode_block(41, 0, solid.b, o));
    for (u32 i = 0; i < 16; ++i)
        for (u32 k = 0; k < 4; ++k) CHECK(near(o[i][k], 128.0f / 255.0f));

    // Gradient on red: R0 = 0, R1 = 63 with P1 = 1 -> 127. Index 8 is 34/64,
    // so (34*127 + 32) >> 6 = 67; index 15 gives 127; texel 0 gives 0.
    BitWriter grad;
    grad.put(1u << 6, 7);
    grad.put(0, 7);    // R0
    grad.put(63, 7);   // R1
    for (u32 k = 0; k < 6; ++k) grad.put(63, 7);  // G, B, A endpoints
    grad.put(0, 1);    // P0
    grad.put(1, 1);    // P1
    for (u32 i = 0; i < 16; ++i) {
        const u32 v = i == 8 ? 8u : i == 15 ? 15u : 0u;
        grad.put(v, i == 0 ? 3 : 4);
    }
    CHECK(bc::decode_block(41, 0, grad.b, o));
    CHECK(near(o[0][0], 0.0f));
    CHECK(near(o[8][0], 67.0f / 255.0f));
    CHECK(near(o[15][0], 127.0f / 255.0f));

    // Mode 5: rotation 1 swaps R with A. Pre-rotation (255, 0, 129, 64)
    // -> (64, 0, 129, 255). The 7-bit endpoints expand to 255, 0, 129.
    BitWriter rot;
    rot.put(1u << 5, 6);  // mode 5 marker
    rot.put(1, 2);        // rotation 1
    rot.put(127, 7);      // R0
    rot.put(127, 7);      // R1
    rot.put(0, 7);        // G0
    rot.put(0, 7);        // G1
    rot.put(64, 7);       // B0
    rot.put(64, 7);       // B1
    rot.put(64, 8);       // A0
    rot.put(64, 8);       // A1
    for (u32 i = 0; i < 16; ++i) rot.put(0, i == 0 ? 1 : 2);  // colour indices
    for (u32 i = 0; i < 16; ++i) rot.put(0, i == 0 ? 1 : 2);  // alpha indices
    CHECK(bc::decode_block(41, 0, rot.b, o));
    for (u32 i = 0; i < 16; ++i) check_rgba(o[i], 64.0f / 255.0f, 0.0f, 129.0f / 255.0f, 1.0f);

    // Mode 4: index selector 0, endpoints 16 (5-bit) and 32 (6-bit).
    // 16 expands to 132 and 32 to 130.
    BitWriter m4;
    m4.put(1u << 4, 5);  // mode 4 marker
    m4.put(0, 2);        // rotation 0
    m4.put(0, 1);        // index selector 0
    for (u32 c = 0; c < 3; ++c) {
        m4.put(16, 5);
        m4.put(16, 5);
    }
    m4.put(32, 6);  // A0
    m4.put(32, 6);  // A1
    for (u32 i = 0; i < 16; ++i) m4.put(0, i == 0 ? 1 : 2);  // colour indices, 2-bit
    for (u32 i = 0; i < 16; ++i) m4.put(0, i == 0 ? 2 : 3);  // alpha indices, 3-bit
    CHECK(bc::decode_block(41, 0, m4.b, o));
    check_rgba(o[7], 132.0f / 255.0f, 132.0f / 255.0f, 132.0f / 255.0f, 130.0f / 255.0f);

}

// BC7 mode 1 (two subsets, 6-bit partition). Block and RGBA8 output are from the
// bcdec reference (fixed seed, forced mode 1).
void test_bc7_mode1_vector() {
    static const u8 kBlock[16] = {0xfe, 0x12, 0xfd, 0x7a, 0xc7, 0x36, 0xd4, 0x99,
                                  0x9a, 0x38, 0xf2, 0x5b, 0x75, 0x8f, 0xf6, 0x5a};
    static const u8 kRgba[64] = {
        72, 28, 100, 255, 122, 215, 58, 255, 209, 108, 169, 255, 190, 97, 159, 255,
        111, 51, 119, 255, 141, 158, 52, 255, 190, 97, 159, 255, 170, 86, 150, 255,
        209, 108, 169, 255, 191, 14, 38, 255, 141, 158, 52, 255, 141, 158, 52, 255,
        209, 108, 169, 255, 172, 71, 44, 255, 162, 99, 46, 255, 181, 42, 41, 255};
    float o[16][4];
    CHECK(bc::decode_block(41, 0, kBlock, o));
    for (u32 i = 0; i < 16; ++i)
        for (u32 c = 0; c < 4; ++c) CHECK(std::lround(o[i][c] * 255.0f) == kRgba[i * 4 + c]);
}

void test_bc6h() {
    float o[16][4];
    // 0x3E0F = 0x3E00 + 15, and 0x3E00 = 1.5, so this is 1 + 527/1024.
    const float half512 = 1.5146484375f;

    // Single-region 10-bit mode, 5-bit code 0x03 (zero-based mode 10; bcdec's
    // comment calls it mode 11). Endpoint 0 is 512 in each channel, so 512
    // unquantizes to 32800, then to half bits 0x3E0F. Endpoint 1 is stored raw
    // (0 here), and index 0 selects endpoint 0.
    BitWriter w;
    w.put(0x03, 5);
    for (u32 c = 0; c < 3; ++c) w.put(512, 10);  // endpoint 0: R, G, B
    for (u32 c = 0; c < 3; ++c) w.put(0, 10);    // endpoint 1: R, G, B (raw)
    w.put(0, 3);                                 // anchor index
    for (u32 i = 1; i < 16; ++i) w.put(0, 4);
    CHECK(bc::decode_block(40, 7, w.b, o));
    CHECK(near(o[0][0], half512) && near(o[9][1], half512) && near(o[15][2], half512));
    CHECK(near(o[0][3], 1.0f));

    // Code 0x03 with endpoint 0 = 1023 unquantizes to 0xFFFF, the largest finite half, 65504.
    BitWriter big;
    big.put(0x03, 5);
    for (u32 c = 0; c < 3; ++c) big.put(1023, 10);
    for (u32 c = 0; c < 3; ++c) big.put(0, 10);
    big.put(0, 3);
    for (u32 i = 1; i < 16; ++i) big.put(0, 4);
    CHECK(bc::decode_block(40, 7, big.b, o));
    CHECK(near(o[0][0], 65504.0f));

    // Code 0x0F (mode 13), bcdec's bit order: base bits 10..15 of each channel
    // come after the 4-bit delta and are stored reversed. 0x8000 has bits 10..15
    // = 0b100000, so each is stored as reverse6(0b100000) = 1. Half 0x3E00 = 1.5.
    BitWriter m13;
    m13.put(0x0F, 5);
    m13.put(0, 10);   // rw[9:0]
    m13.put(0, 10);   // gw[9:0]
    m13.put(0, 10);   // bw[9:0]
    m13.put(0, 4);    // rx[3:0]
    m13.put(1, 6);    // rw[15:10], reversed
    m13.put(0, 4);    // gx[3:0]
    m13.put(1, 6);    // gw[15:10], reversed
    m13.put(0, 4);    // bx[3:0]
    m13.put(1, 6);    // bw[15:10], reversed
    m13.put(0, 3);    // anchor index
    for (u32 i = 1; i < 16; ++i) m13.put(0, 4);
    CHECK(bc::decode_block(40, 7, m13.b, o));
    CHECK(near(o[0][0], 1.5f));

    // Reserved code 0x1F decodes to black.
    u8 reserved[16]{};
    reserved[0] = 0x1F;
    CHECK(bc::decode_block(40, 7, reserved, o));
    check_rgba(o[0], 0.0f, 0.0f, 0.0f, 1.0f);
}

// BC6H mode 1 (two regions, unsigned). Block and float output are from the bcdec
// reference (fixed seed, forced mode 1).
void test_bc6h_mode1_vector() {
    static const u8 kBlock[16] = {0xd5, 0x61, 0x72, 0x6c, 0x43, 0x57, 0x54, 0xb1,
                                  0x7c, 0x5c, 0x14, 0x1d, 0x7a, 0x01, 0xb4, 0x6b};
    static const float kRgb[48] = {
        0.00411605835f, 337.25f, 0.147094727f, 0.000222444534f, 19.515625f, 0.0362548828f,
        0.000197529793f, 18.96875f, 0.0278930664f, 9.60826874e-05f, 16.1875f, 0.00730895996f,
        0.00036907196f, 686.0f, 0.299804688f, 0.000197529793f, 18.96875f, 0.0278930664f,
        9.60826874e-05f, 16.1875f, 0.00730895996f, 0.000120997429f, 17.265625f, 0.0126571655f,
        0.00036907196f, 686.0f, 0.299804688f, 0.000250816345f, 20.0625f, 0.0479736328f,
        0.000250816345f, 20.0625f, 0.0479736328f, 0.000120997429f, 17.265625f, 0.0126571655f,
        103.375f, 17.875f, 0.00772094727f, 0.000172615051f, 18.421875f, 0.0220489502f,
        0.000120997429f, 17.265625f, 0.0126571655f, 0.000222444534f, 19.515625f, 0.0362548828f};
    float o[16][4];
    CHECK(bc::decode_block(40, 0, kBlock, o));
    for (u32 i = 0; i < 16; ++i)
        for (u32 c = 0; c < 3; ++c) CHECK(o[i][c] == kRgb[i * 3 + c]);
}

void test_block_sizes() {
    CHECK(bc::block_bytes(35) == 8);
    CHECK(bc::block_bytes(38) == 8);
    for (u32 f : {36u, 37u, 39u, 40u, 41u}) CHECK(bc::block_bytes(f) == 16);
    CHECK(bc::block_bytes(10) == 0);
    CHECK(bc::block_bytes(34) == 0);
    CHECK(bc::block_bytes(42) == 0);
    CHECK(gcn::data_format_bytes(35) == 8);
    CHECK(gcn::data_format_bytes(41) == 16);
}

// ---------------------------------------------------------------------------
// texture_to_rgba8 with BC data
// ---------------------------------------------------------------------------

void test_texture_bc1_linear_edges() {
    // 8x8 linear BC1: 2x2 blocks in raster order. Block 0 red, 1 blue, 2 (170,0,85), 3 (85,0,170).
    std::vector<u8> src(4 * 8);
    bc1_block(&src[0], 0xF800, 0x001F, kIdx0);
    bc1_block(&src[8], 0xF800, 0x001F, kIdx1);
    bc1_block(&src[16], 0xF800, 0x001F, kIdx2);
    bc1_block(&src[24], 0xF800, 0x001F, kIdx3);

    gcn::TextureDesc td;
    td.width = 8; td.height = 8; td.pitch = 8; td.dfmt = 35; td.nfmt = 0; td.tiling_index = 8;
    std::vector<u8> out;
    std::string err;
    CHECK(gcn::texture_guest_size(td) == 32);
    CHECK(gcn::texture_to_rgba8(td, src.data(), src.size(), out, err));
    if (out.size() != 8 * 8 * 4) {
        std::fprintf(stderr, "texture_to_rgba8 failed: %s\n", err.c_str());
        ++g_failures;
        return;
    }
    auto px = [&](u32 x, u32 y, u32 k) { return out[(y * 8 + x) * 4 + k]; };
    CHECK(px(0, 0, 0) == 255 && px(0, 0, 1) == 0 && px(0, 0, 2) == 0 && px(0, 0, 3) == 255);
    CHECK(px(7, 0, 0) == 0 && px(7, 0, 1) == 0 && px(7, 0, 2) == 255 && px(7, 0, 3) == 255);
    CHECK(px(0, 7, 0) == 170 && px(0, 7, 1) == 0 && px(0, 7, 2) == 85 && px(0, 7, 3) == 255);
    CHECK(px(7, 7, 0) == 85 && px(7, 7, 1) == 0 && px(7, 7, 2) == 170 && px(7, 7, 3) == 255);
    CHECK(px(4, 4, 0) == 85 && px(4, 4, 2) == 170);

    // A short source is rejected before any block is read.
    CHECK(!gcn::texture_to_rgba8(td, src.data(), 31, out, err));
}

void test_texture_bc1_partial_blocks() {
    // 6x5: 2x2 blocks, cropped to 6 columns and 5 rows. Pitch rounds up to 2 blocks.
    std::vector<u8> src(4 * 8);
    bc1_block(&src[0], 0xF800, 0x001F, kIdx0);
    bc1_block(&src[8], 0xF800, 0x001F, kIdx1);
    bc1_block(&src[16], 0xF800, 0x001F, kIdx2);
    bc1_block(&src[24], 0xF800, 0x001F, kIdx3);

    gcn::TextureDesc td;
    td.width = 6; td.height = 5; td.pitch = 6; td.dfmt = 35; td.nfmt = 0; td.tiling_index = 8;
    std::vector<u8> out;
    std::string err;
    CHECK(gcn::texture_guest_size(td) == 32);
    CHECK(gcn::texture_to_rgba8(td, src.data(), src.size(), out, err));
    if (out.size() != 6 * 5 * 4) {
        std::fprintf(stderr, "texture_to_rgba8 failed: %s\n", err.c_str());
        ++g_failures;
        return;
    }
    auto px = [&](u32 x, u32 y, u32 k) { return out[(y * 6 + x) * 4 + k]; };
    CHECK(px(5, 0, 2) == 255 && px(5, 0, 0) == 0);          // block 1, column 1: blue
    CHECK(px(5, 4, 0) == 85 && px(5, 4, 2) == 170);         // block 3, the last texel row
}

void test_texture_bc1_micro_tiled() {
    // 32x32 micro tiled BC1 is one tile of 8x8 blocks. Block (2,1) sits at slot
    // micro_tile_index(2,1) = 6; slot 6 is red and every other slot is blue.
    std::vector<u8> src(64 * 8);
    for (u32 slot = 0; slot < 64; ++slot)
        bc1_block(&src[slot * 8], slot == 6 ? 0xF800 : 0x001F, slot == 6 ? 0x001F : 0xF800, kIdx0);

    gcn::TextureDesc td;
    td.width = 32; td.height = 32; td.pitch = 32; td.dfmt = 35; td.nfmt = 0; td.tiling_index = 13;
    std::vector<u8> out;
    std::string err;
    CHECK(gcn::texture_guest_size(td) == 64 * 8);
    CHECK(gcn::texture_to_rgba8(td, src.data(), src.size(), out, err));
    if (out.size() != 32 * 32 * 4) {
        std::fprintf(stderr, "texture_to_rgba8 failed: %s\n", err.c_str());
        ++g_failures;
        return;
    }
    auto px = [&](u32 x, u32 y, u32 k) { return out[(y * 32 + x) * 4 + k]; };
    CHECK(px(8, 4, 0) == 255 && px(8, 4, 2) == 0);    // block (2,1): red
    CHECK(px(0, 0, 0) == 0 && px(0, 0, 2) == 255);    // block (0,0): blue
}

void test_texture_bc_guest_sizes() {
    gcn::TextureDesc td;
    td.dfmt = 37; td.nfmt = 0; td.tiling_index = 8;
    td.width = 20; td.height = 20; td.pitch = 20;
    CHECK(gcn::texture_guest_size(td) == 5 * 5 * 16);  // linear: 5x5 blocks, no rounding
    td.tiling_index = 13;
    CHECK(gcn::texture_guest_size(td) == 8 * 8 * 16);  // micro: rounded to 8x8 blocks
}

void test_non_bc_unchanged() {
    // RGBA8 (dfmt 10) keeps the element path: linear bytes come through as-is.
    gcn::TextureDesc td;
    td.width = 3; td.height = 2; td.pitch = 3; td.dfmt = 10; td.nfmt = 0; td.tiling_index = 8;
    std::vector<u8> src(3 * 2 * 4);
    for (usize i = 0; i < src.size(); ++i) src[i] = static_cast<u8>(i * 7 + 3);
    std::vector<u8> out;
    std::string err;
    CHECK(gcn::texture_guest_size(td) == 24);
    CHECK(gcn::texture_to_rgba8(td, src.data(), src.size(), out, err));
    CHECK(out == src);
}

} // namespace

int main() {
    test_bc1();
    test_bc2();
    test_bc3_alpha();
    test_bc4_bc5();
    test_bc7_modes();
    test_bc7_mode1_vector();
    test_bc6h();
    test_bc6h_mode1_vector();
    test_block_sizes();
    test_texture_bc1_linear_edges();
    test_texture_bc1_partial_blocks();
    test_texture_bc1_micro_tiled();
    test_texture_bc_guest_sizes();
    test_non_bc_unchanged();
    if (g_failures) {
        std::fprintf(stderr, "BC decode test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("BC decode test: all checks passed\n");
    return 0;
}
