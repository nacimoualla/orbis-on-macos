// H264Decoder (behind sceVideodec) against a real stream.
//
// Encodes 40 flat frames, each a different luma level, with VideoToolbox's
// H.264 encoder (High profile, B-frames allowed, an IDR every 16 frames), turns
// the output into an Annex B stream like the game's movies, and checks that the
// decoder returns every frame once, in display order, with the right content:
//  - one chunk per access unit, as sceVideodecDecode gets them;
//  - every access unit split in the middle of a NAL unit across two chunks;
//  - reset() part way through, then the whole stream again.

#include "ps4/h264_decoder.hpp"

#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <VideoToolbox/VideoToolbox.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

constexpr int kW = 320, kH = 240, kFrames = 40;

int luma_of(int i) { return 30 + i * 4; }

struct Au {
    std::vector<unsigned char> annexb;
    long long pts;
};

struct Encoded {
    std::vector<Au> aus;  // decode order
};

void append_start_code(std::vector<unsigned char>& v, const unsigned char* p, size_t n) {
    static const unsigned char sc[4] = {0, 0, 0, 1};
    v.insert(v.end(), sc, sc + 4);
    v.insert(v.end(), p, p + n);
}

void on_encoded(void* ref, void*, OSStatus status, VTEncodeInfoFlags, CMSampleBufferRef sb) {
    CHECK(status == noErr && sb);
    auto* enc = static_cast<Encoded*>(ref);
    Au au;
    au.pts = CMSampleBufferGetPresentationTimeStamp(sb).value;

    bool key = true;
    if (CFArrayRef att = CMSampleBufferGetSampleAttachmentsArray(sb, false); att && CFArrayGetCount(att) > 0) {
        auto d = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(att, 0));
        key = !CFDictionaryContainsKey(d, kCMSampleAttachmentKey_NotSync);
    }
    if (key) {
        CMFormatDescriptionRef fmt = CMSampleBufferGetFormatDescription(sb);
        size_t count = 0;
        CHECK(CMVideoFormatDescriptionGetH264ParameterSetAtIndex(fmt, 0, nullptr, nullptr, &count, nullptr) == noErr);
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* p = nullptr;
            size_t n = 0;
            CHECK(CMVideoFormatDescriptionGetH264ParameterSetAtIndex(fmt, i, &p, &n, nullptr, nullptr) == noErr);
            append_start_code(au.annexb, p, n);
        }
    }
    CMBlockBufferRef bb = CMSampleBufferGetDataBuffer(sb);
    std::vector<unsigned char> avcc(CMBlockBufferGetDataLength(bb));
    CHECK(CMBlockBufferCopyDataBytes(bb, 0, avcc.size(), avcc.data()) == noErr);
    for (size_t off = 0; off + 4 <= avcc.size();) {
        const size_t n = (size_t(avcc[off]) << 24) | (size_t(avcc[off + 1]) << 16) | (size_t(avcc[off + 2]) << 8) |
                         avcc[off + 3];
        CHECK(off + 4 + n <= avcc.size());
        append_start_code(au.annexb, avcc.data() + off + 4, n);
        off += 4 + n;
    }
    enc->aus.push_back(std::move(au));
}

void set_num(VTCompressionSessionRef s, CFStringRef key, int v) {
    CFNumberRef n = CFNumberCreate(nullptr, kCFNumberIntType, &v);
    VTSessionSetProperty(s, key, n);
    CFRelease(n);
}

Encoded encode() {
    Encoded enc;
    VTCompressionSessionRef s = nullptr;
    CHECK(VTCompressionSessionCreate(nullptr, kW, kH, kCMVideoCodecType_H264, nullptr, nullptr, nullptr,
                                     &on_encoded, &enc, &s) == noErr);
    VTSessionSetProperty(s, kVTCompressionPropertyKey_ProfileLevel, kVTProfileLevel_H264_High_AutoLevel);
    VTSessionSetProperty(s, kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanTrue);
    VTSessionSetProperty(s, kVTCompressionPropertyKey_RealTime, kCFBooleanFalse);
    set_num(s, kVTCompressionPropertyKey_MaxKeyFrameInterval, 16);
    CHECK(VTCompressionSessionPrepareToEncodeFrames(s) == noErr);
    for (int i = 0; i < kFrames; ++i) {
        CVPixelBufferRef pb = nullptr;
        CHECK(CVPixelBufferCreate(nullptr, kW, kH, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, nullptr, &pb) ==
              kCVReturnSuccess);
        CVPixelBufferLockBaseAddress(pb, 0);
        auto* y = static_cast<unsigned char*>(CVPixelBufferGetBaseAddressOfPlane(pb, 0));
        auto* uv = static_cast<unsigned char*>(CVPixelBufferGetBaseAddressOfPlane(pb, 1));
        for (int r = 0; r < kH; ++r) std::memset(y + r * CVPixelBufferGetBytesPerRowOfPlane(pb, 0), luma_of(i), kW);
        for (int r = 0; r < kH / 2; ++r) std::memset(uv + r * CVPixelBufferGetBytesPerRowOfPlane(pb, 1), 128, kW);
        CVPixelBufferUnlockBaseAddress(pb, 0);
        CHECK(VTCompressionSessionEncodeFrame(s, pb, CMTimeMake(i, 30), kCMTimeInvalid, nullptr, nullptr, nullptr) ==
              noErr);
        CVPixelBufferRelease(pb);
    }
    CHECK(VTCompressionSessionCompleteFrames(s, kCMTimeInvalid) == noErr);
    VTCompressionSessionInvalidate(s);
    CFRelease(s);
    return enc;
}

double mean_luma(const ps4::H264Frame& f) {
    double sum = 0;
    for (size_t i = 0; i < size_t(f.width) * f.height; ++i) sum += f.nv12[i];
    return sum / (double(f.width) * f.height);
}

// Collect what the decoder returns, checking order and content.
struct Sink {
    std::vector<long long> pts;
    void take(ps4::H264Decoder& d) {
        ps4::H264Frame f;
        while (d.pop(f)) {
            CHECK(f.width == kW && f.height == kH);
            CHECK(f.nv12.size() == size_t(kW) * kH * 3 / 2);
            CHECK(f.attached == f.pts + 1000);
            const double y = mean_luma(f);
            if (std::abs(y - luma_of(int(f.pts))) > 4.0) {
                std::cerr << "frame pts " << f.pts << ": mean luma " << y << ", want " << luma_of(int(f.pts)) << "\n";
                CHECK(false);
            }
            pts.push_back(static_cast<long long>(f.pts));
        }
    }
    void check_all() const {
        CHECK(pts.size() == size_t(kFrames));
        for (int i = 0; i < kFrames; ++i) CHECK(pts[i] == i);
    }
};

}  // namespace

int main() {
    const Encoded enc = encode();
    CHECK(enc.aus.size() == size_t(kFrames));
    int out_of_order = 0;
    for (size_t i = 1; i < enc.aus.size(); ++i) out_of_order += enc.aus[i].pts < enc.aus[i - 1].pts;
    std::cout << "encoded " << enc.aus.size() << " access units, " << out_of_order
              << " out of display order (B-frames)\n";

    // 1. One chunk per access unit.
    {
        ps4::H264Decoder d;
        Sink sink;
        for (const auto& au : enc.aus) {
            CHECK(d.feed(au.annexb.data(), au.annexb.size(), au.pts, au.pts + 1000));
            sink.take(d);
        }
        d.flush();
        sink.take(d);
        sink.check_all();
        CHECK(d.decoded_count() == size_t(kFrames));
    }

    // 2. Each access unit split mid-NAL across two chunks. The second chunk's
    //    pts is junk: an AU keeps the pts of the chunk it started in.
    {
        ps4::H264Decoder d;
        Sink sink;
        for (const auto& au : enc.aus) {
            // Cut in the middle of the largest NAL unit.
            const auto& b = au.annexb;
            size_t best = 0, best_len = 0;
            for (size_t i = 0; i + 3 <= b.size(); ++i) {
                if (b[i] != 0 || b[i + 1] != 0 || b[i + 2] != 1) continue;
                size_t j = i + 3;
                while (j + 3 <= b.size() && !(b[j] == 0 && b[j + 1] == 0 && (b[j + 2] == 1 || b[j + 2] == 0))) ++j;
                if (j + 3 > b.size()) j = b.size();
                if (j - (i + 3) > best_len) {
                    best = i + 3;
                    best_len = j - (i + 3);
                }
            }
            CHECK(best_len >= 4);
            const size_t cut = best + best_len / 2;
            CHECK(d.feed(au.annexb.data(), cut, au.pts, au.pts + 1000));
            CHECK(d.feed(au.annexb.data() + cut, au.annexb.size() - cut, 999999, 0));
            sink.take(d);
        }
        d.flush();
        sink.take(d);
        sink.check_all();
    }

    // 3. Reset part way (as on a movie restart), then play the whole stream.
    {
        ps4::H264Decoder d;
        for (int i = 0; i < 10; ++i) d.feed(enc.aus[i].annexb.data(), enc.aus[i].annexb.size(), 0, 0);
        d.reset();
        ps4::H264Frame f;
        CHECK(!d.pop(f));
        Sink sink;
        for (const auto& au : enc.aus) {
            d.feed(au.annexb.data(), au.annexb.size(), au.pts, au.pts + 1000);
            sink.take(d);
        }
        d.flush();
        sink.take(d);
        sink.check_all();
        CHECK(!d.pop(f));
    }

    std::cout << "test_h264_decoder: OK\n";
    return 0;
}
