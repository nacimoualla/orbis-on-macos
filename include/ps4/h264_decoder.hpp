#pragma once

// H.264 decoding behind sceVideodec, using VideoToolbox.
//
// The game feeds an Annex B byte stream (start codes, SPS/PPS in band) one
// chunk per sceVideodecDecode call and gets pictures back in display order,
// one per call. This class does the parts the PS4 hardware does for it:
//  - splits the stream into NAL units and groups them into access units
//    (a chunk may hold one AU, or the AU may only end when the next starts);
//  - decodes each AU with a VTDecompressionSession built from the in-band
//    SPS/PPS, rebuilt when they change;
//  - computes picture order counts from the slice headers and releases frames
//    in display order once more than the stream's reorder depth are queued.
//
// Output frames are NV12 packed tightly: `width` x `height` luma bytes, then
// `width` x `height / 2` interleaved CbCr bytes (pitch == width).

#include "ps4/common.hpp"

#include <memory>
#include <vector>

namespace ps4 {

struct H264Frame {
    u32 width = 0;
    u32 height = 0;
    u64 pts = 0;
    u64 attached = 0;
    std::vector<u8> nv12;  // width * height * 3 / 2 bytes
};

class H264Decoder {
public:
    H264Decoder();
    ~H264Decoder();
    H264Decoder(const H264Decoder&) = delete;
    H264Decoder& operator=(const H264Decoder&) = delete;

    /// Feed one chunk of the Annex B stream. Any access unit this completes is
    /// decoded. The chunk's pts/attached data are carried by the AU whose first
    /// NAL unit it holds. Returns false if a decode failed (the stream goes on).
    bool feed(const u8* data, usize size, u64 pts, u64 attached);

    /// Take the next frame in display order, if the reorder window allows one.
    bool pop(H264Frame& out);

    /// End of stream: decode the pending access unit and release every queued
    /// frame to pop(), regardless of the reorder depth.
    void flush();

    /// Drop all pending input and queued frames (sceVideodecReset).
    void reset();

    /// Number of access units decoded so far (for logging and tests).
    [[nodiscard]] u64 decoded_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

}  // namespace ps4
