#pragma once

// GCN (Sea Islands / Liverpool, the PS4 GPU) shader ISA decoder and the
// direct GCN -> Metal Shading Language translator (ROADMAP Phase 19,
// ADR 0007 as amended).
//
// Pure C++: nothing in this header depends on Metal, so the decoder, the
// translator and the scalar resource pre-pass can be unit-tested without a
// GPU. The Metal side (pipeline cache, resource upload, binding) lives in
// gcn_pipeline.hpp / gcn_pipeline.mm.

#include "ps4/common.hpp"
#include <functional>
#include <string>
#include <vector>

namespace ps4::gcn {

enum class Encoding : u8 {
    Unknown, SOP2, SOPK, SOP1, SOPC, SOPP, SMRD,
    VOP2, VOP1, VOPC, VOP3, VINTRP, DS, MUBUF, MTBUF, MIMG, EXP,
};

const char* encoding_name(Encoding e) noexcept;

// Operand encodings shared by the scalar (8-bit) and vector (9-bit) source
// fields.
namespace operand {
inline constexpr u32 kVccLo = 106;
inline constexpr u32 kVccHi = 107;
inline constexpr u32 kM0 = 124;
inline constexpr u32 kExecLo = 126;
inline constexpr u32 kExecHi = 127;
inline constexpr u32 kZero = 128;
inline constexpr u32 kVccz = 251;
inline constexpr u32 kExecz = 252;
inline constexpr u32 kScc = 253;
inline constexpr u32 kLiteral = 255;
inline constexpr u32 kVgprBase = 256;
} // namespace operand

/// One decoded instruction. `pc` is the dword offset of the instruction in
/// its program; fields not used by an encoding are left zero.
struct Instruction {
    Encoding enc{Encoding::Unknown};
    u32 op{0};
    u32 pc{0};
    u32 size{1};       // in dwords, including a trailing literal
    u32 dw0{0};
    u32 dw1{0};
    u32 literal{0};
    bool has_literal{false};

    // Generic decoded operands (meaning depends on the encoding).
    u32 dst{0};        // SDST / VDST / VDATA
    u32 sdst{0};       // VOP3b / VOPC-in-VOP3 scalar destination
    u32 src0{0}, src1{0}, src2{0};
    u32 abs{0}, neg{0}, omod{0};
    bool clamp{false};
    i32 simm{0};       // SOPP/SOPK immediate (sign extended)

    // Memory encodings.
    u32 offset{0};     // MUBUF/MTBUF byte offset, SMRD dword/byte offset
    bool imm{false};   // SMRD: offset is an immediate (dword units)
    bool offen{false}, idxen{false}, glc{false};
    u32 vaddr{0}, srsrc{0}, ssamp{0}, soffset{0}, sbase{0};
    u32 dfmt{0}, nfmt{0};
    u32 dmask{0};
    bool unorm{false}, da{false};

    // VINTRP / EXP.
    u32 attr{0}, chan{0};
    u32 en{0}, target{0};
    bool compr{false}, done{false}, vm{false};
};

/// Decode a single instruction from `words` (which must contain at least
/// `avail` dwords). Returns false when the stream is truncated.
bool decode_instruction(const u32* words, usize avail, u32 pc, Instruction& out);

/// Human readable mnemonic for an instruction ("v_mad_f32", ...). Unknown
/// opcodes come back as "<enc>_op<N>".
std::string mnemonic(const Instruction& inst);
std::string disassemble(const Instruction& inst);

/// Callback used to read guest memory (code, descriptors, constants).
using MemReader = std::function<bool(u64 addr, void* dst, usize size)>;

struct Program {
    std::vector<Instruction> insts;
    std::vector<u32> code;           // raw dwords of the main program
    std::vector<u32> fetch_code;     // raw dwords of an inlined fetch shader
    u64 hash{0};                     // hash of code + fetch_code
    bool has_branches{false};
};

/// Read a program from guest memory starting at `addr` until (and including)
/// the s_endpgm that terminates it (taking forward branches into account).
/// When the program calls a fetch shader with s_swappc_b64 through a user SGPR
/// pair, `user_sgprs` is used to find it and its body is inlined in place of
/// the call (the trailing s_setpc_b64 is dropped).
bool load_program(const MemReader& read, u64 addr, const u32* user_sgprs,
                  u32 user_sgpr_count, Program& out, std::string& error);

/// Decode an in-memory program (no fetch-shader inlining). For tests.
bool decode_program(const u32* words, usize count, Program& out, std::string& error);

enum class Stage : u8 { Vertex, Pixel };

/// Static description of the image (T#/S#) slots a program samples from.
struct ImageSlot {
    u32 srsrc{0};   // first SGPR of the T#
    u32 ssamp{0};   // first SGPR of the S# (~0u for image_load)
    bool sampled{true};
};

struct TranslateOptions {
    /// How the VS is driven: vertex ids are remapped for RectList/QuadList
    /// expansion (the 4th rect corner is synthesised in the shader).
    enum class VertexRemap : u8 { None, RectList, QuadList } remap{VertexRemap::None};
};

struct TranslatedPipeline {
    bool ok{false};
    std::string error;               // first unsupported construct
    std::string msl;                 // full Metal source (both stages)
    std::string vs_entry{"gcn_vs_main"};
    std::string ps_entry{"gcn_ps_main"};
    std::vector<ImageSlot> ps_images;
    std::vector<ImageSlot> vs_images;
    u32 vs_param_count{0};
    u32 ps_attr_mask{0};
    bool ps_writes_color{false};
};

/// Translate a VS/PS pair to MSL. `ps_input_cntl` holds SPI_PS_INPUT_CNTL_n
/// (attribute -> VS param mapping; low 5 bits = offset), may be null.
TranslatedPipeline translate_pipeline(const Program& vs, const Program& ps,
                                      const u32* ps_input_cntl,
                                      const TranslateOptions& opts);

/// MSL binding contract shared with gcn_pipeline.mm.
namespace binding {
inline constexpr u32 kMemBuffer = 0;     // device const uchar* (resource arena)
inline constexpr u32 kDrawConsts = 1;    // constant GcnDrawConsts&
inline constexpr u32 kMaxRanges = 16;
} // namespace binding

/// Layout of the per-draw constant block (must match the MSL struct emitted
/// by the translator).
struct DrawConsts {
    u64 range_base[binding::kMaxRanges];
    u32 range_size[binding::kMaxRanges];
    u32 range_off[binding::kMaxRanges];
    u32 range_count;
    u32 base_vertex;
    u32 vs_user[16];
    u32 ps_user[16];
    u32 pad[2];
};

/// Guest memory a draw's shader will touch, discovered by evaluating the
/// scalar (uniform) part of the program on the CPU with the real user SGPRs.
struct MemRange {
    u64 addr{0};
    u32 size{0};
};

struct ImageDescriptors {
    u32 tsharp[8]{};
    u32 ssharp[4]{};
    bool sampler_known{false};
};

struct ResourceScan {
    bool ok{false};
    std::string error;
    std::vector<MemRange> ranges;
    std::vector<ImageDescriptors> images;   // parallel to ImageSlot list
};

ResourceScan scan_resources(const Program& prog, const std::vector<ImageSlot>& slots,
                            const u32* user_sgprs, u32 user_count,
                            const MemReader& read, u32 max_vertex_index);

// --- Texture descriptor helpers (T#/S#) ------------------------------------

struct TextureDesc {
    u64 base{0};
    u32 width{0}, height{0}, depth{1}, pitch{0};
    u32 dfmt{0}, nfmt{0};
    u32 dst_sel[4]{4, 5, 6, 7};
    u32 tiling_index{0};
    u32 type{0};
    u32 base_level{0}, last_level{0};
};

TextureDesc decode_tsharp(const u32* t);

struct SamplerDesc {
    u32 clamp_x{0}, clamp_y{0}, clamp_z{0};
    u32 xy_mag_filter{0}, xy_min_filter{0}, mip_filter{0};
    bool force_unnormalized{false};
};

SamplerDesc decode_ssharp(const u32* s);

/// Bytes per element for a T#/V# data format, 0 if unsupported.
u32 data_format_bytes(u32 dfmt);

/// Convert texture memory to linear RGBA8 (R in byte 0). Supports linear and
/// 1D/2D-thin micro tiled 32bpp/16bpp/8bpp layouts plus the common formats.
/// Returns false (with `error`) when the layout or format is unsupported.
bool texture_to_rgba8(const TextureDesc& td, const u8* src, usize src_size,
                      std::vector<u8>& out, std::string& error);

/// Byte size of the guest memory backing a texture's mip 0.
usize texture_guest_size(const TextureDesc& td);

u64 hash_words(const u32* data, usize count, u64 seed = 0xcbf29ce484222325ull);

} // namespace ps4::gcn
