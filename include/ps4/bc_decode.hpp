#pragma once

// CPU decoding of the block-compressed (BC1..BC7) texture formats used by GCN
// textures. Each block covers 4x4 texels; the T# data format picks the codec
// (IMG_DATA_FORMAT 35 = BC1 .. 41 = BC7).
//
// All BC1-BC7 modes are decoded. BC6H and BC7 follow bcdec.h (v0.985), which
// is dual MIT / Unlicense; the partition tables come from the same source.

#include "ps4/common.hpp"

namespace ps4::bc {

/// Bytes per 4x4 block for a GCN data format: 8 for BC1 and BC4, 16 for BC2,
/// BC3, BC5, BC6H and BC7. Returns 0 for every non-BC format.
u32 block_bytes(u32 dfmt);

/// Decodes one 4x4 block into float RGBA, texel i at out[i] with i = y * 4 + x.
/// UNORM channels land in [0,1]. SNORM (BC4/BC5 with nfmt 1) lands in [-1,1].
/// BC6H (nfmt 1 = signed, otherwise unsigned) yields half-float values that can
/// exceed 1. Returns false only for a non-BC data format.
bool decode_block(u32 dfmt, u32 nfmt, const u8* block, float out[16][4]);

} // namespace ps4::bc
