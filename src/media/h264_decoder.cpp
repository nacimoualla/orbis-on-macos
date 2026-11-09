// H.264 decoder behind sceVideodec: Annex B parsing, display-order reordering
// and VideoToolbox decoding. See include/ps4/h264_decoder.hpp.

#include "ps4/h264_decoder.hpp"

#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <VideoToolbox/VideoToolbox.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <optional>
#include <tuple>

namespace ps4 {
namespace {

// ---------------------------------------------------------------------------
// Bitstream parsing
// ---------------------------------------------------------------------------

// NAL payload (after the header byte) with emulation prevention bytes removed.
std::vector<u8> to_rbsp(const u8* p, usize n) {
    std::vector<u8> out;
    out.reserve(n);
    usize zeros = 0;
    for (usize i = 0; i < n; ++i) {
        if (zeros >= 2 && p[i] == 3) {
            zeros = 0;
            continue;
        }
        zeros = p[i] == 0 ? zeros + 1 : 0;
        out.push_back(p[i]);
    }
    return out;
}

class BitReader {
public:
    explicit BitReader(const std::vector<u8>& d) : d_(d) {}
    bool ok() const { return !overrun_; }
    u32 u(int n) {
        u32 v = 0;
        for (int i = 0; i < n; ++i) v = (v << 1) | bit();
        return v;
    }
    u32 ue() {
        int lz = 0;
        while (!bit()) {
            if (++lz > 31 || overrun_) return 0;
        }
        return lz ? ((1u << lz) - 1) + u(lz) : 0;
    }
    i32 se() {
        const u32 k = ue();
        return (k & 1) ? static_cast<i32>((k + 1) / 2) : -static_cast<i32>(k / 2);
    }

private:
    u32 bit() {
        if (pos_ >= d_.size() * 8) {
            overrun_ = true;
            return 0;
        }
        const u32 b = (d_[pos_ / 8] >> (7 - pos_ % 8)) & 1;
        ++pos_;
        return b;
    }
    const std::vector<u8>& d_;
    usize pos_ = 0;
    bool overrun_ = false;
};

struct Sps {
    std::vector<u8> nal;  // whole NAL unit, header byte included
    u32 log2_max_frame_num = 4;
    u32 poc_type = 0;
    u32 log2_max_poc_lsb = 4;
    bool frame_mbs_only = true;
    bool separate_colour_plane = false;
    u32 width = 0, height = 0;  // after cropping
    int max_num_reorder = -1;   // from VUI bitstream_restriction, -1 = absent
    int max_dec_frame_buffering = -1;
};

struct Pps {
    std::vector<u8> nal;
    u32 sps_id = 0;
};

void skip_scaling_list(BitReader& br, int size) {
    int last = 8, next = 8;
    for (int j = 0; j < size; ++j) {
        if (next != 0) next = (last + br.se() + 256) % 256;
        last = next == 0 ? last : next;
    }
}

void skip_hrd(BitReader& br) {
    const u32 cpb_cnt = br.ue() + 1;
    br.u(4);
    br.u(4);
    for (u32 i = 0; i < cpb_cnt && br.ok(); ++i) {
        br.ue();
        br.ue();
        br.u(1);
    }
    br.u(5);
    br.u(5);
    br.u(5);
    br.u(5);
}

std::optional<std::pair<u32, Sps>> parse_sps(const u8* nal, usize n) {
    if (n < 4) return std::nullopt;
    const auto rbsp = to_rbsp(nal + 1, n - 1);
    BitReader br(rbsp);
    Sps s;
    s.nal.assign(nal, nal + n);
    const u32 profile = br.u(8);
    br.u(8);  // constraint flags
    br.u(8);  // level
    const u32 id = br.ue();
    u32 chroma_format = 1;
    if (profile == 100 || profile == 110 || profile == 122 || profile == 244 || profile == 44 ||
        profile == 83 || profile == 86 || profile == 118 || profile == 128 || profile == 138 ||
        profile == 139 || profile == 134 || profile == 135) {
        chroma_format = br.ue();
        if (chroma_format == 3) s.separate_colour_plane = br.u(1);
        br.ue();  // bit_depth_luma_minus8
        br.ue();  // bit_depth_chroma_minus8
        br.u(1);  // qpprime_y_zero_transform_bypass
        if (br.u(1)) {  // seq_scaling_matrix_present
            const int lists = chroma_format != 3 ? 8 : 12;
            for (int i = 0; i < lists; ++i)
                if (br.u(1)) skip_scaling_list(br, i < 6 ? 16 : 64);
        }
    }
    s.log2_max_frame_num = br.ue() + 4;
    s.poc_type = br.ue();
    if (s.poc_type == 0) {
        s.log2_max_poc_lsb = br.ue() + 4;
    } else if (s.poc_type == 1) {
        br.u(1);
        br.se();
        br.se();
        const u32 cycle = br.ue();
        for (u32 i = 0; i < cycle && br.ok(); ++i) br.se();
    }
    br.ue();  // max_num_ref_frames
    br.u(1);  // gaps_in_frame_num_allowed
    const u32 w_mbs = br.ue() + 1;
    const u32 h_units = br.ue() + 1;
    s.frame_mbs_only = br.u(1);
    if (!s.frame_mbs_only) br.u(1);  // mb_adaptive_frame_field
    br.u(1);                          // direct_8x8_inference
    u32 crop_l = 0, crop_r = 0, crop_t = 0, crop_b = 0;
    if (br.u(1)) {
        crop_l = br.ue();
        crop_r = br.ue();
        crop_t = br.ue();
        crop_b = br.ue();
    }
    const u32 sub_w = (chroma_format == 1 || chroma_format == 2) ? 2 : 1;
    const u32 sub_h = (chroma_format == 1) ? 2 : 1;
    const u32 frame_h_mbs = h_units * (s.frame_mbs_only ? 1 : 2);
    const u32 crop_unit_y = (chroma_format == 0 ? 1 : sub_h) * (s.frame_mbs_only ? 1 : 2);
    const u32 crop_unit_x = chroma_format == 0 ? 1 : sub_w;
    s.width = w_mbs * 16 - crop_unit_x * (crop_l + crop_r);
    s.height = frame_h_mbs * 16 - crop_unit_y * (crop_t + crop_b);

    if (br.u(1)) {  // vui_parameters_present
        if (br.u(1)) {  // aspect_ratio_info_present
            if (br.u(8) == 255) {
                br.u(16);
                br.u(16);
            }
        }
        if (br.u(1)) br.u(1);  // overscan
        if (br.u(1)) {         // video_signal_type
            br.u(3);
            br.u(1);
            if (br.u(1)) {
                br.u(8);
                br.u(8);
                br.u(8);
            }
        }
        if (br.u(1)) {  // chroma_loc_info
            br.ue();
            br.ue();
        }
        if (br.u(1)) {  // timing_info
            br.u(32);
            br.u(32);
            br.u(1);
        }
        const bool nal_hrd = br.u(1);
        if (nal_hrd) skip_hrd(br);
        const bool vcl_hrd = br.u(1);
        if (vcl_hrd) skip_hrd(br);
        if (nal_hrd || vcl_hrd) br.u(1);  // low_delay_hrd
        br.u(1);                          // pic_struct_present
        if (br.u(1)) {                    // bitstream_restriction
            br.u(1);
            br.ue();
            br.ue();
            br.ue();
            br.ue();
            const u32 reorder = br.ue();
            const u32 dpb = br.ue();
            if (br.ok()) {
                s.max_num_reorder = static_cast<int>(reorder);
                s.max_dec_frame_buffering = static_cast<int>(dpb);
            }
        }
    }
    if (!br.ok() && s.width == 0) return std::nullopt;
    return std::make_pair(id, std::move(s));
}

std::optional<std::pair<u32, Pps>> parse_pps(const u8* nal, usize n) {
    if (n < 2) return std::nullopt;
    const auto rbsp = to_rbsp(nal + 1, n - 1);
    BitReader br(rbsp);
    Pps p;
    p.nal.assign(nal, nal + n);
    const u32 id = br.ue();
    p.sps_id = br.ue();
    if (!br.ok()) return std::nullopt;
    return std::make_pair(id, std::move(p));
}

struct SliceHeader {
    u32 first_mb = 0;
    u32 pps_id = 0;
    u32 poc_lsb = 0;
};

// Reads up to pic_order_cnt_lsb. Needs the SPS for field widths.
std::optional<SliceHeader> parse_slice(const u8* nal, usize n,
                                       const std::array<std::optional<Sps>, 32>& spss,
                                       const std::array<std::optional<Pps>, 256>& ppss) {
    if (n < 2) return std::nullopt;
    // The header is short: 64 bytes of payload is plenty.
    const auto rbsp = to_rbsp(nal + 1, std::min<usize>(n - 1, 64));
    BitReader br(rbsp);
    SliceHeader h;
    h.first_mb = br.ue();
    br.ue();  // slice_type
    h.pps_id = br.ue();
    if (h.pps_id >= ppss.size() || !ppss[h.pps_id]) return h;
    const u32 sps_id = ppss[h.pps_id]->sps_id;
    if (sps_id >= spss.size() || !spss[sps_id]) return h;
    const Sps& s = *spss[sps_id];
    if (s.separate_colour_plane) br.u(2);
    br.u(static_cast<int>(s.log2_max_frame_num));  // frame_num
    if (!s.frame_mbs_only && br.u(1)) br.u(1);     // field_pic, bottom_field
    if ((nal[0] & 0x1f) == 5) br.ue();             // idr_pic_id
    if (s.poc_type == 0) h.poc_lsb = br.u(static_cast<int>(s.log2_max_poc_lsb));
    if (!br.ok()) return std::nullopt;
    return h;
}

// Splits an Annex B chunk into NAL units, start codes dropped. Bytes before the
// first start code are returned as `lead`: the tail of a NAL unit the previous
// chunk began. Trailing zero bytes are dropped only where a start code follows;
// the last NAL unit may continue in the next chunk, so its zeros may be payload
// (the decoder trims it once it knows the unit is complete).
void split_annexb(const u8* d, usize n, std::vector<std::pair<const u8*, usize>>& nals,
                  std::pair<const u8*, usize>& lead) {
    auto next_sc = [&](usize from) -> usize {
        for (usize i = from; i + 3 <= n; ++i)
            if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1) return i;
        return n;
    };
    auto trim = [&](usize b, usize e) {
        if (e < n)
            while (e > b && d[e - 1] == 0) --e;
        return std::make_pair(d + b, e - b);
    };
    usize sc = next_sc(0);
    lead = trim(0, sc);
    while (sc < n) {
        const usize begin = sc + 3;
        const usize nsc = next_sc(begin);
        auto nal = trim(begin, nsc);
        if (nal.second) nals.push_back(nal);
        sc = nsc;
    }
}

template <typename T>
struct CfRef {
    T ref = nullptr;
    CfRef() = default;
    explicit CfRef(T r) : ref(r) {}
    ~CfRef() { reset(); }
    CfRef(const CfRef&) = delete;
    CfRef& operator=(const CfRef&) = delete;
    void reset(T r = nullptr) {
        if (ref) CFRelease(ref);
        ref = r;
    }
    explicit operator bool() const { return ref != nullptr; }
};

}  // namespace

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

struct H264Decoder::Impl {
    std::array<std::optional<Sps>, 32> spss;
    std::array<std::optional<Pps>, 256> ppss;

    // Access unit being collected.
    std::vector<std::vector<u8>> au_nals;
    bool au_has_vcl = false;
    u64 au_pts = 0, au_attached = 0;

    // POC state (type 0).
    i32 prev_poc_msb = 0;
    u32 prev_poc_lsb = 0;
    u64 epoch = 0;        // bumped at every IDR; orders frames across IDRs
    u64 decode_index = 0;  // also the POC stand-in for types 1 and 2
    u64 decoded = 0;

    // VideoToolbox session and the parameter sets it was built for.
    CfRef<CMVideoFormatDescriptionRef> format;
    VTDecompressionSessionRef session = nullptr;
    std::vector<u8> session_sps, session_pps;

    struct Pending {
        u64 epoch;
        i64 poc;
        u64 order;  // decode order, tie-break
        H264Frame frame;
    };
    std::vector<Pending> queue;  // decoded, not yet released
    int reorder_depth = 16;
    bool draining = false;

    struct FrameTag {
        Impl* self;
        u64 epoch;
        i64 poc;
        u64 order;
        u64 pts, attached;
        u32 width, height;
        bool ok;
    };

    ~Impl() { destroy_session(); }

    void destroy_session() {
        if (session) {
            VTDecompressionSessionWaitForAsynchronousFrames(session);
            VTDecompressionSessionInvalidate(session);
            CFRelease(session);
            session = nullptr;
        }
        format.reset();
        session_sps.clear();
        session_pps.clear();
    }

    static void on_frame(void* /*ctx*/, void* tag_ptr, OSStatus status, VTDecodeInfoFlags /*flags*/,
                         CVImageBufferRef image, CMTime /*pts*/, CMTime /*dur*/) {
        auto* tag = static_cast<FrameTag*>(tag_ptr);
        if (status != noErr || !image) return;
        CVPixelBufferRef pb = image;
        if (CVPixelBufferGetPlaneCount(pb) < 2) return;
        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        const u32 bw = static_cast<u32>(CVPixelBufferGetWidth(pb));
        const u32 bh = static_cast<u32>(CVPixelBufferGetHeight(pb));
        const u32 w = std::min(tag->width ? tag->width : bw, bw) & ~1u;
        const u32 h = std::min(tag->height ? tag->height : bh, bh) & ~1u;
        Pending p{tag->epoch, tag->poc, tag->order, {}};
        p.frame.width = w;
        p.frame.height = h;
        p.frame.pts = tag->pts;
        p.frame.attached = tag->attached;
        p.frame.nv12.resize(static_cast<usize>(w) * h * 3 / 2);
        const auto* y = static_cast<const u8*>(CVPixelBufferGetBaseAddressOfPlane(pb, 0));
        const auto* uv = static_cast<const u8*>(CVPixelBufferGetBaseAddressOfPlane(pb, 1));
        const usize ys = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
        const usize uvs = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
        u8* dst = p.frame.nv12.data();
        for (u32 r = 0; r < h; ++r) std::memcpy(dst + static_cast<usize>(r) * w, y + r * ys, w);
        dst += static_cast<usize>(w) * h;
        for (u32 r = 0; r < h / 2; ++r) std::memcpy(dst + static_cast<usize>(r) * w, uv + r * uvs, w);
        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        tag->ok = true;
        tag->self->queue.push_back(std::move(p));
    }

    bool ensure_session(const Sps& sps, const Pps& pps) {
        if (session && session_sps == sps.nal && session_pps == pps.nal) return true;
        if (session) {
            // A new stream configuration: frames of the old one go out first.
            VTDecompressionSessionWaitForAsynchronousFrames(session);
        }
        destroy_session();
        const u8* sets[2] = {sps.nal.data(), pps.nal.data()};
        const size_t sizes[2] = {sps.nal.size(), pps.nal.size()};
        CMVideoFormatDescriptionRef fmt = nullptr;
        OSStatus st = CMVideoFormatDescriptionCreateFromH264ParameterSets(
            kCFAllocatorDefault, 2, sets, sizes, 4, &fmt);
        if (st != noErr) {
            std::fprintf(stderr, "[Videodec] format description failed (%d)\n", static_cast<int>(st));
            return false;
        }
        format.reset(fmt);

        const int32_t pixfmt = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
        CfRef<CFNumberRef> pf(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pixfmt));
        const void* keys[] = {kCVPixelBufferPixelFormatTypeKey};
        const void* vals[] = {pf.ref};
        CfRef<CFDictionaryRef> attrs(CFDictionaryCreate(kCFAllocatorDefault, keys, vals, 1,
                                                        &kCFTypeDictionaryKeyCallBacks,
                                                        &kCFTypeDictionaryValueCallBacks));
        VTDecompressionOutputCallbackRecord cb{&Impl::on_frame, this};
        st = VTDecompressionSessionCreate(kCFAllocatorDefault, format.ref, nullptr, attrs.ref, &cb, &session);
        if (st != noErr) {
            std::fprintf(stderr, "[Videodec] VTDecompressionSessionCreate failed (%d)\n", static_cast<int>(st));
            session = nullptr;
            format.reset();
            return false;
        }
        session_sps = sps.nal;
        session_pps = pps.nal;
        reorder_depth = sps.max_num_reorder >= 0 ? sps.max_num_reorder
                        : sps.poc_type == 2       ? 0
                        : sps.max_dec_frame_buffering >= 0 ? sps.max_dec_frame_buffering
                                                           : 16;
        std::fprintf(stderr, "[Videodec] session %ux%u poc_type=%u reorder_depth=%d\n", sps.width,
                     sps.height, sps.poc_type, reorder_depth);
        return true;
    }

    // The last NAL unit collected is complete: drop its trailing zero bytes
    // (trailing_zero_8bits or the first byte of a 4-byte start code).
    void seal_last_nal() {
        if (au_nals.empty()) return;
        auto& last = au_nals.back();
        while (!last.empty() && last.back() == 0) last.pop_back();
        if (last.empty()) au_nals.pop_back();
    }

    // Decode the collected access unit (if it has any slices).
    bool finish_au() {
        seal_last_nal();
        if (!au_has_vcl) {
            // Parameter sets were handled as they arrived; nothing to decode.
            au_nals.clear();
            return true;
        }
        std::vector<std::vector<u8>> nals;
        nals.swap(au_nals);
        au_has_vcl = false;

        // First slice decides the picture's parameter sets and POC.
        const std::vector<u8>* first = nullptr;
        for (const auto& n : nals) {
            const u8 t = n[0] & 0x1f;
            if (t == 1 || t == 5) {
                first = &n;
                break;
            }
        }
        const auto sh = parse_slice(first->data(), first->size(), spss, ppss);
        if (!sh || !ppss[sh->pps_id] || !spss[ppss[sh->pps_id]->sps_id]) {
            std::fprintf(stderr, "[Videodec] slice without known parameter sets, skipped\n");
            return false;
        }
        const Pps& pps = *ppss[sh->pps_id];
        const Sps& sps = *spss[pps.sps_id];
        if (!ensure_session(sps, pps)) return false;

        const bool idr = ((*first)[0] & 0x1f) == 5;
        const bool is_ref = ((*first)[0] & 0x60) != 0;
        if (idr) {
            ++epoch;
            prev_poc_msb = 0;
            prev_poc_lsb = 0;
        }
        i64 poc;
        if (sps.poc_type == 0) {
            const u32 max_lsb = 1u << sps.log2_max_poc_lsb;
            const u32 lsb = sh->poc_lsb;
            i32 msb = prev_poc_msb;
            if (lsb < prev_poc_lsb && prev_poc_lsb - lsb >= max_lsb / 2)
                msb += static_cast<i32>(max_lsb);
            else if (lsb > prev_poc_lsb && lsb - prev_poc_lsb > max_lsb / 2)
                msb -= static_cast<i32>(max_lsb);
            poc = static_cast<i64>(msb) + lsb;
            if (is_ref) {
                prev_poc_msb = msb;
                prev_poc_lsb = lsb;
            }
        } else {
            poc = static_cast<i64>(decode_index);  // output order == decode order
        }

        // AVCC sample: slices (and SEI) with 4-byte big-endian lengths.
        usize total = 0;
        for (const auto& n : nals) {
            const u8 t = n[0] & 0x1f;
            if ((t >= 1 && t <= 6)) total += 4 + n.size();
        }
        CMBlockBufferRef bb = nullptr;
        OSStatus st = CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, nullptr, total, kCFAllocatorDefault,
                                                         nullptr, 0, total, kCMBlockBufferAssureMemoryNowFlag, &bb);
        if (st != noErr) return false;
        CfRef<CMBlockBufferRef> block(bb);
        usize off = 0;
        for (const auto& n : nals) {
            const u8 t = n[0] & 0x1f;
            if (!(t >= 1 && t <= 6)) continue;
            const u32 len = static_cast<u32>(n.size());
            const u8 be[4] = {static_cast<u8>(len >> 24), static_cast<u8>(len >> 16), static_cast<u8>(len >> 8),
                              static_cast<u8>(len)};
            CMBlockBufferReplaceDataBytes(be, bb, off, 4);
            CMBlockBufferReplaceDataBytes(n.data(), bb, off + 4, n.size());
            off += 4 + n.size();
        }
        CMSampleTimingInfo timing{kCMTimeInvalid, CMTimeMake(static_cast<int64_t>(decode_index), 90000),
                                  kCMTimeInvalid};
        CMSampleBufferRef sb = nullptr;
        const size_t sample_size = total;
        st = CMSampleBufferCreateReady(kCFAllocatorDefault, bb, format.ref, 1, 1, &timing, 1, &sample_size, &sb);
        if (st != noErr) return false;
        CfRef<CMSampleBufferRef> sample(sb);

        FrameTag tag{this, epoch, poc, decode_index, au_pts, au_attached, sps.width, sps.height, false};
        ++decode_index;
        VTDecodeInfoFlags info = 0;
        st = VTDecompressionSessionDecodeFrame(session, sb, 0, &tag, &info);
        VTDecompressionSessionWaitForAsynchronousFrames(session);  // the tag lives on this stack
        ++decoded;
        if (st != noErr || !tag.ok) {
            std::fprintf(stderr, "[Videodec] decode of AU %llu failed (status %d)\n",
                         static_cast<unsigned long long>(tag.order), static_cast<int>(st));
            return false;
        }
        return true;
    }

    // Returns false if a decode failed.
    bool add_nal(const u8* p, usize n, u64 pts, u64 attached) {
        bool ok = true;
        seal_last_nal();
        const u8 type = p[0] & 0x1f;
        const bool vcl = type == 1 || type == 5;
        bool starts_au = false;
        if (vcl) {
            const auto sh = parse_slice(p, n, spss, ppss);
            starts_au = au_has_vcl && sh && sh->first_mb == 0;
        } else if (type == 6 || type == 7 || type == 8 || type == 9 || (type >= 14 && type <= 18)) {
            starts_au = au_has_vcl;
        }
        if (starts_au) ok = finish_au();

        if (type == 7) {
            if (auto s = parse_sps(p, n); s && s->first < spss.size()) spss[s->first] = std::move(s->second);
        } else if (type == 8) {
            if (auto pp = parse_pps(p, n); pp && pp->first < ppss.size()) ppss[pp->first] = std::move(pp->second);
        }
        if (au_nals.empty()) {
            au_pts = pts;
            au_attached = attached;
        }
        au_nals.emplace_back(p, p + n);
        au_has_vcl = au_has_vcl || vcl;
        if (type == 10 || type == 11) ok = finish_au() && ok;  // end of sequence / stream
        return ok;
    }

    bool releasable() const {
        if (queue.empty()) return false;
        return draining || static_cast<int>(queue.size()) > reorder_depth;
    }
};

H264Decoder::H264Decoder() : m(std::make_unique<Impl>()) {}
H264Decoder::~H264Decoder() = default;

bool H264Decoder::feed(const u8* data, usize size, u64 pts, u64 attached) {
    if (!data || !size) return true;
    m->draining = false;
    std::vector<std::pair<const u8*, usize>> nals;
    std::pair<const u8*, usize> lead{};
    split_annexb(data, size, nals, lead);
    if (lead.second && !m->au_nals.empty()) {
        // Continuation of the previous chunk's last NAL unit.
        auto& last = m->au_nals.back();
        last.insert(last.end(), lead.first, lead.first + lead.second);
    }
    bool ok = true;
    for (const auto& [p, n] : nals) ok = m->add_nal(p, n, pts, attached) && ok;
    return ok;
}

bool H264Decoder::pop(H264Frame& out) {
    if (!m->releasable()) return false;
    auto it = std::min_element(m->queue.begin(), m->queue.end(), [](const Impl::Pending& a, const Impl::Pending& b) {
        return std::tie(a.epoch, a.poc, a.order) < std::tie(b.epoch, b.poc, b.order);
    });
    out = std::move(it->frame);
    m->queue.erase(it);
    return true;
}

void H264Decoder::flush() {
    m->finish_au();
    if (m->session) VTDecompressionSessionWaitForAsynchronousFrames(m->session);
    m->draining = true;
}

void H264Decoder::reset() {
    m->au_nals.clear();
    m->au_has_vcl = false;
    m->queue.clear();
    m->prev_poc_msb = 0;
    m->prev_poc_lsb = 0;
    m->draining = false;
    m->destroy_session();
}

u64 H264Decoder::decoded_count() const noexcept { return m->decoded; }

}  // namespace ps4
