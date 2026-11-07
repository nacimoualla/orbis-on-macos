// Direct GCN -> MSL translator (ADR 0007, amended: see the "Deviation"
// section there for why this is not GCN -> SPIR-V -> SPIRV-Cross).
//
// Execution model
// ---------------
// Each MSL invocation is one GCN lane. The scalar register file is a
// per-invocation `uint s[128]` (indexed by the raw operand encoding, so
// vcc_lo == s[106], m0 == s[124], exec_lo == s[126]) and the vector file is
// `uint v[N]`. Uniform scalar code is simply re-evaluated by every lane,
// which is correct because it only depends on uniform inputs.
//
// 64-bit lane masks (EXEC, VCC, VOPC/VOP3 compare results) are modelled by
// the lane's own bit only: bit 0 of the low dword is "this lane". Bitwise
// SALU ops on masks therefore keep working, and per-lane control flow
// (s_cbranch_execz / vccz) becomes ordinary per-invocation control flow.
//
// Memory: every guest range a draw touches is discovered on the CPU by
// scan_resources() (it evaluates the uniform scalar prefix with the real user
// SGPRs) and copied into one per-draw arena buffer. The shader translates a
// 64-bit guest address to an arena offset with a small range table, so the
// generated MSL does not depend on any data values and can be cached by ISA
// hash.

#include "ps4/bc_decode.hpp"
#include "ps4/gcn_shader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <sstream>

namespace ps4::gcn {

namespace {

const char* kPrelude = R"MSL(#include <metal_stdlib>
using namespace metal;

struct GcnDrawConsts {
    ulong range_base[16];
    uint range_size[16];
    uint range_off[16];
    uint range_count;
    uint base_vertex;
    uint vs_user[16];
    uint ps_user[16];
    uint flags;
    uint pad;
};

struct GcnComputeConsts {
    uint cs_user[16];
    uint pad[16];
};

#define F(x) as_type<float>(uint(x))
#define U(x) as_type<uint>(float(x))

static inline uint gcn_ld8(device const uchar* mem, constant GcnDrawConsts& dc, ulong a) {
    for (uint i = 0; i < dc.range_count; ++i) {
        ulong b = dc.range_base[i];
        if (a >= b && a < b + ulong(dc.range_size[i])) return uint(mem[dc.range_off[i] + uint(a - b)]);
    }
    return 0u;
}

static inline uint gcn_ld32(device const uchar* mem, constant GcnDrawConsts& dc, ulong a) {
    for (uint i = 0; i < dc.range_count; ++i) {
        ulong b = dc.range_base[i];
        if (a >= b && a + 4ul <= b + ulong(dc.range_size[i])) {
            device const uchar* p = mem + dc.range_off[i] + uint(a - b);
            return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
        }
    }
    return 0u;
}

static inline uint gcn_ld16(device const uchar* mem, constant GcnDrawConsts& dc, ulong a) {
    return gcn_ld8(mem, dc, a) | (gcn_ld8(mem, dc, a + 1) << 8);
}

static inline uint gcn_ld8(device const uchar* mem, constant GcnComputeConsts& dc, ulong a) {
    return uint(mem[a]);
}
static inline uint gcn_ld8(device uchar* mem, constant GcnComputeConsts& dc, ulong a) {
    return uint(mem[a]);
}

static inline uint gcn_ld32(device const uchar* mem, constant GcnComputeConsts& dc, ulong a) {
    device const uchar* p = mem + a;
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}
static inline uint gcn_ld32(device uchar* mem, constant GcnComputeConsts& dc, ulong a) {
    device uchar* p = mem + a;
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

static inline uint gcn_ld16(device const uchar* mem, constant GcnComputeConsts& dc, ulong a) {
    return gcn_ld8(mem, dc, a) | (gcn_ld8(mem, dc, a + 1) << 8);
}
static inline uint gcn_ld16(device uchar* mem, constant GcnComputeConsts& dc, ulong a) {
    return gcn_ld8(mem, dc, a) | (gcn_ld8(mem, dc, a + 1) << 8);
}

static inline void gcn_st8(device uchar* mem, constant GcnComputeConsts& dc, ulong a, uint v) {
    mem[a] = uchar(v);
}
static inline void gcn_st16(device uchar* mem, constant GcnComputeConsts& dc, ulong a, uint v) {
    device uchar* p = mem + a;
    p[0] = uchar(v); p[1] = uchar(v >> 8);
}
static inline void gcn_st32(device uchar* mem, constant GcnComputeConsts& dc, ulong a, uint v) {
    device uchar* p = mem + a;
    p[0] = uchar(v); p[1] = uchar(v >> 8); p[2] = uchar(v >> 16); p[3] = uchar(v >> 24);
}

static inline void gcn_st8(device uchar* mem, constant GcnDrawConsts& dc, ulong a, uint v) {
    for (uint i = 0; i < dc.range_count; ++i) {
        ulong b = dc.range_base[i];
        if (a >= b && a < b + ulong(dc.range_size[i])) { mem[dc.range_off[i] + uint(a - b)] = uchar(v); return; }
    }
}
static inline void gcn_st16(device uchar* mem, constant GcnDrawConsts& dc, ulong a, uint v) {
    gcn_st8(mem, dc, a, v & 0xffu);
    gcn_st8(mem, dc, a + 1ul, (v >> 8) & 0xffu);
}
static inline void gcn_st32(device uchar* mem, constant GcnDrawConsts& dc, ulong a, uint v) {
    for (uint i = 0; i < dc.range_count; ++i) {
        ulong b = dc.range_base[i];
        if (a >= b && a + 4ul <= b + ulong(dc.range_size[i])) {
            device uchar* p = mem + dc.range_off[i] + uint(a - b);
            p[0] = uchar(v); p[1] = uchar(v >> 8); p[2] = uchar(v >> 16); p[3] = uchar(v >> 24);
            return;
        }
    }
}

static inline ulong gcn_vbase(uint w0, uint w1) { return ulong(w0) | (ulong(w1 & 0xffffu) << 32); }
static inline uint gcn_vstride(uint w1) { return (w1 >> 16) & 0x3fffu; }

static inline int gcn_sext(uint v, uint bits) { return int(v << (32u - bits)) >> int(32u - bits); }

// Convert one raw component of `bits` width according to a GCN NUM_FORMAT.
static inline uint gcn_cvt(uint c, uint bits, uint nfmt) {
    if (bits == 32u) return c;
    switch (nfmt) {
    case 0u: return U(float(c) / float((1u << bits) - 1u));                          // unorm
    case 1u: case 6u: return U(max(float(gcn_sext(c, bits)) / float((1u << (bits - 1u)) - 1u), -1.0f)); // snorm
    case 2u: return U(float(c));                                                     // uscaled
    case 3u: return U(float(gcn_sext(c, bits)));                                     // sscaled
    case 4u: return c;                                                               // uint
    case 5u: return uint(gcn_sext(c, bits));                                         // sint
    case 7u: return bits == 16u ? U(float(as_type<half>(ushort(c)))) : c;           // float
    default: return U(float(c) / float((1u << bits) - 1u));
    }
}

template <typename MemT, typename ConstsT>
static inline uint4 gcn_fetch_format(MemT mem, constant ConstsT& dc,
                                     ulong a, uint dfmt, uint nfmt) {
    bool is_int = (nfmt == 4u || nfmt == 5u);
    uint4 r = uint4(0u, 0u, 0u, is_int ? 1u : as_type<uint>(1.0f));
    switch (dfmt) {
    case 1u: r.x = gcn_cvt(gcn_ld8(mem, dc, a), 8u, nfmt); break;
    case 2u: r.x = gcn_cvt(gcn_ld16(mem, dc, a), 16u, nfmt); break;
    case 3u: { uint w = gcn_ld16(mem, dc, a);
        r.x = gcn_cvt(w & 0xffu, 8u, nfmt); r.y = gcn_cvt(w >> 8, 8u, nfmt); break; }
    case 4u: r.x = gcn_ld32(mem, dc, a); break;
    case 5u: { uint w = gcn_ld32(mem, dc, a);
        r.x = gcn_cvt(w & 0xffffu, 16u, nfmt); r.y = gcn_cvt(w >> 16, 16u, nfmt); break; }
    case 9u: { uint w = gcn_ld32(mem, dc, a);
        r.x = gcn_cvt(w & 0x3ffu, 10u, nfmt); r.y = gcn_cvt((w >> 10) & 0x3ffu, 10u, nfmt);
        r.z = gcn_cvt((w >> 20) & 0x3ffu, 10u, nfmt); r.w = gcn_cvt(w >> 30, 2u, nfmt); break; }
    case 10u: { uint w = gcn_ld32(mem, dc, a);
        r.x = gcn_cvt(w & 0xffu, 8u, nfmt); r.y = gcn_cvt((w >> 8) & 0xffu, 8u, nfmt);
        r.z = gcn_cvt((w >> 16) & 0xffu, 8u, nfmt); r.w = gcn_cvt(w >> 24, 8u, nfmt); break; }
    case 11u: r.x = gcn_ld32(mem, dc, a); r.y = gcn_ld32(mem, dc, a + 4); break;
    case 12u: { uint w0 = gcn_ld32(mem, dc, a), w1 = gcn_ld32(mem, dc, a + 4);
        r.x = gcn_cvt(w0 & 0xffffu, 16u, nfmt); r.y = gcn_cvt(w0 >> 16, 16u, nfmt);
        r.z = gcn_cvt(w1 & 0xffffu, 16u, nfmt); r.w = gcn_cvt(w1 >> 16, 16u, nfmt); break; }
    case 13u: r.x = gcn_ld32(mem, dc, a); r.y = gcn_ld32(mem, dc, a + 4); r.z = gcn_ld32(mem, dc, a + 8); break;
    case 14u: r.x = gcn_ld32(mem, dc, a); r.y = gcn_ld32(mem, dc, a + 4);
        r.z = gcn_ld32(mem, dc, a + 8); r.w = gcn_ld32(mem, dc, a + 12); break;
    default: break;
    }
    return r;
}

static inline uint gcn_sel(uint4 r, uint sel, bool is_int) {
    switch (sel) {
    case 0u: return 0u;
    case 1u: return is_int ? 1u : as_type<uint>(1.0f);
    case 4u: return r.x;
    case 5u: return r.y;
    case 6u: return r.z;
    case 7u: return r.w;
    default: return 0u;
    }
}

// image_sample_*_o offset VGPR: signed 6-bit texel offsets, x in bits 5:0
// and y in bits 13:8.
static inline float2 gcn_texel_offset(uint o) {
    return float2(float(int(o << 26u) >> 26), float(int((o >> 8u) << 26u) >> 26));
}

// Pack one component for a format store: unorm/snorm from float, half for
// 16-bit float, raw bits otherwise (uint/sint and 32-bit float).
static inline uint gcn_pack(uint x, uint bits, uint nfmt) {
    const uint mask = bits >= 32u ? 0xffffffffu : ((1u << bits) - 1u);
    switch (nfmt) {
    case 0u: return uint(rint(clamp(as_type<float>(x), 0.0f, 1.0f) * float(mask))) & mask;
    case 1u: return uint(int(rint(clamp(as_type<float>(x), -1.0f, 1.0f) * float(mask >> 1)))) & mask;
    case 7u: return bits == 16u ? uint(as_type<ushort>(half(as_type<float>(x)))) : (x & mask);
    default: return x & mask;
    }
}

template <typename ConstsT>
static inline void gcn_store_format(device uchar* mem, constant ConstsT& dc, ulong a, uint dfmt, uint nfmt,
                                    uint4 v) {
    switch (dfmt) {
    case 1u: gcn_st8(mem, dc, a, gcn_pack(v.x, 8u, nfmt)); break;
    case 2u: gcn_st16(mem, dc, a, gcn_pack(v.x, 16u, nfmt)); break;
    case 3u: gcn_st16(mem, dc, a, gcn_pack(v.x, 8u, nfmt) | (gcn_pack(v.y, 8u, nfmt) << 8)); break;
    case 4u: gcn_st32(mem, dc, a, v.x); break;
    case 5u: gcn_st32(mem, dc, a, gcn_pack(v.x, 16u, nfmt) | (gcn_pack(v.y, 16u, nfmt) << 16)); break;
    case 10u: gcn_st32(mem, dc, a, gcn_pack(v.x, 8u, nfmt) | (gcn_pack(v.y, 8u, nfmt) << 8) |
                                   (gcn_pack(v.z, 8u, nfmt) << 16) | (gcn_pack(v.w, 8u, nfmt) << 24)); break;
    case 11u: gcn_st32(mem, dc, a, v.x); gcn_st32(mem, dc, a + 4, v.y); break;
    case 12u: gcn_st32(mem, dc, a, gcn_pack(v.x, 16u, nfmt) | (gcn_pack(v.y, 16u, nfmt) << 16));
              gcn_st32(mem, dc, a + 4, gcn_pack(v.z, 16u, nfmt) | (gcn_pack(v.w, 16u, nfmt) << 16)); break;
    case 13u: gcn_st32(mem, dc, a, v.x); gcn_st32(mem, dc, a + 4, v.y); gcn_st32(mem, dc, a + 8, v.z); break;
    case 14u: gcn_st32(mem, dc, a, v.x); gcn_st32(mem, dc, a + 4, v.y);
              gcn_st32(mem, dc, a + 8, v.z); gcn_st32(mem, dc, a + 12, v.w); break;
    default: break;
    }
}

// MUBUF/MTBUF address + bounds check. Returns false for out-of-range accesses
// (GCN returns zero for those).
static inline bool gcn_buf_addr(uint w0, uint w1, uint w2, uint index, uint offset, thread ulong& a) {
    uint stride = gcn_vstride(w1);
    if (stride != 0u) { if (index >= w2) return false; }
    else if (offset >= w2) return false;
    a = gcn_vbase(w0, w1) + ulong(index) * ulong(stride) + ulong(offset);
    return true;
}

static inline float gcn_mul_legacy(float a, float b) { return (a == 0.0f || b == 0.0f) ? 0.0f : a * b; }
static inline uint gcn_bfe_u(uint v, uint off, uint w) { off &= 31u; w &= 31u; return w == 0u ? 0u : (v >> off) & ((1u << w) - 1u); }
static inline uint gcn_bfe_i(uint v, uint off, uint w) { off &= 31u; w &= 31u; if (w == 0u) return 0u; return uint(gcn_sext((v >> off) & ((1u << w) - 1u), w)); }
static inline uint gcn_ffbh(uint v) { return v == 0u ? 0xffffffffu : clz(v); }
static inline uint gcn_ffbl(uint v) { return v == 0u ? 0xffffffffu : ctz(v); }
static inline float gcn_med3(float a, float b, float c) { return max(min(a, b), min(max(a, b), c)); }
static inline int gcn_med3i(int a, int b, int c) { return max(min(a, b), min(max(a, b), c)); }
static inline uint gcn_med3u(uint a, uint b, uint c) { return max(min(a, b), min(max(a, b), c)); }
static inline uint gcn_pkrtz(float a, float b) { return uint(as_type<ushort>(half(a))) | (uint(as_type<ushort>(half(b))) << 16); }
static inline float gcn_hlo(uint v) { return float(as_type<half>(ushort(v & 0xffffu))); }
static inline float gcn_hhi(uint v) { return float(as_type<half>(ushort(v >> 16))); }
static inline ulong gcn_s64(thread const uint* s, uint r) { return ulong(s[r]) | (ulong(s[r + 1]) << 32); }
static inline void gcn_w64(thread uint* s, uint r, ulong v) { s[r] = uint(v); s[r + 1] = uint(v >> 32); }
)MSL";

std::string hex(u32 v) {
    char b[16];
    std::snprintf(b, sizeof(b), "0x%08xu", v);
    return b;
}

class Emitter {
public:
    Emitter(const Program& p, Stage st, const u32* ps_input_cntl, u32 ps_input_ena)
        : prog(p), stage(st), input_cntl(ps_input_cntl), input_ena(ps_input_ena) {}

    bool run();

    std::ostringstream body;
    std::string error;
    std::vector<ImageSlot> images;
    std::set<u32> exported_params;      // VS
    std::map<u32, u32> ps_attrs;          // PS attr index -> locn (VS param)
    std::set<u32> flat_attrs;
    bool ps_color{false};
    u8 ps_color_mask{0};
    bool ps_depth{false};
    bool uses_exec_write{false};
    bool uses_frag_pos{false};
    u32 max_vgpr{0};
    u32 max_sgpr{0};

private:
    const Program& prog;
    Stage stage;
    std::vector<u32> uniform_slots;     // parallel to prog.insts (see uniform_load_slots)
    const u32* input_cntl;
    [[maybe_unused]] u32 input_ena;
    const Instruction* cur{nullptr};

    void fail(const std::string& why) {
        if (error.empty()) error = why;
    }

    void track_v(u32 v) { max_vgpr = std::max(max_vgpr, v + 1); }
    void track_s(u32 s) { if (s < 104) max_sgpr = std::max(max_sgpr, s + 1); }

    std::string vreg(u32 r) { track_v(r); return "v[" + std::to_string(r) + "]"; }

    // 32-bit source operand as a uint expression.
    std::string usrc(u32 s) {
        if (s >= operand::kVgprBase) return vreg(s - 256);
        if (s < 128) { track_s(s); return "s[" + std::to_string(s) + "]"; }
        if (s == operand::kZero) return "0u";
        if (s >= 129 && s <= 192) return std::to_string(s - 128) + "u";
        if (s >= 193 && s <= 208) return "uint(-" + std::to_string(s - 192) + ")";
        switch (s) {
        case 240: return "0x3f000000u";
        case 241: return "0xbf000000u";
        case 242: return "0x3f800000u";
        case 243: return "0xbf800000u";
        case 244: return "0x40000000u";
        case 245: return "0xc0000000u";
        case 246: return "0x40800000u";
        case 247: return "0xc0800000u";
        case operand::kVccz: return "((s[106] | s[107]) == 0u ? 1u : 0u)";
        case operand::kExecz: return "((s[126] | s[127]) == 0u ? 1u : 0u)";
        case operand::kScc: return "scc";
        case operand::kLiteral: return hex(cur->literal);
        default:
            fail("unsupported source operand " + std::to_string(s));
            return "0u";
        }
    }

    // 64-bit scalar source.
    std::string usrc64(u32 s) {
        if (s < 127) { track_s(s); track_s(s + 1); return "gcn_s64(s, " + std::to_string(s) + "u)"; }
        if (s == operand::kLiteral) return "ulong(" + hex(cur->literal) + ")";
        if (s >= 193 && s <= 208) return "ulong(long(-" + std::to_string(s - 192) + "))";
        return "ulong(" + usrc(s) + ")";
    }

    std::string fsrc(u32 s, u32 idx) {
        std::string e = "F(" + usrc(s) + ")";
        if (cur->enc == Encoding::VOP3) {
            if (cur->abs & (1u << idx)) e = "fabs(" + e + ")";
            if (cur->neg & (1u << idx)) e = "(-" + e + ")";
        }
        return e;
    }

    std::string isrc(u32 s) { return "int(" + usrc(s) + ")"; }

    std::string exec_guard() { return uses_exec_write ? "if ((s[126] & 1u) != 0u) " : ""; }

    void write_v(u32 d, const std::string& expr) {
        body << "    " << exec_guard() << vreg(d) << " = " << expr << ";\n";
    }

    std::string float_result(const std::string& f) {
        std::string e = f;
        if (cur->enc == Encoding::VOP3) {
            if (cur->omod == 1) e = "(" + e + ") * 2.0f";
            else if (cur->omod == 2) e = "(" + e + ") * 4.0f";
            else if (cur->omod == 3) e = "(" + e + ") * 0.5f";
            if (cur->clamp) e = "saturate(" + e + ")";
        }
        return "U(" + e + ")";
    }

    void emit_valu(const Instruction& i);
    bool valu_op2(u32 op, u32 dst, u32 a, u32 b, u32 c, bool vop3);
    bool valu_op1(u32 op, u32 dst, u32 a);
    bool valu_op3(u32 op, u32 dst, u32 a, u32 b, u32 c);
    void valu_cmp(u32 op, u32 sdst, u32 a, u32 b);
    void emit_salu(const Instruction& i);
    void emit_smrd(const Instruction& i);
    void emit_buffer(const Instruction& i);
    void emit_mimg(const Instruction& i);
    void emit_interp(const Instruction& i);
    void emit_export(const Instruction& i);
    void emit_one(const Instruction& i, bool structured);
};

void Emitter::valu_cmp(u32 op, u32 sdst, u32 a, u32 b) {
    std::string cond;
    const u32 c = op & 0xF;
    if (op < 0x20) {
        const std::string fa = fsrc(a, 0), fb = fsrc(b, 1);
        static const char* fmt[16] = {
            "false", "%s < %s", "%s == %s", "%s <= %s", "%s > %s", "(%s < %s || %s > %s)", "%s >= %s",
            "(!isnan(%s) && !isnan(%s))", "(isnan(%s) || isnan(%s))", "!(%s >= %s)", "!(%s < %s || %s > %s)",
            "!(%s > %s)", "!(%s <= %s)", "!(%s == %s)", "!(%s < %s)", "true"};
        char b2[512];
        std::snprintf(b2, sizeof(b2), fmt[c], fa.c_str(), fb.c_str(), fa.c_str(), fb.c_str());
        cond = b2;
    } else if ((op >= 0x80 && op < 0xA0 && (op & 0xF) < 8) || (op >= 0xC0 && op < 0xE0 && (op & 0xF) < 8)) {
        const bool sgn = op < 0xA0;
        const std::string x = sgn ? isrc(a) : usrc(a);
        const std::string y = sgn ? isrc(b) : usrc(b);
        static const char* ops[8] = {"false", "<", "==", "<=", ">", "!=", ">=", "true"};
        const u32 k = op & 7;
        if (k == 0) cond = "false";
        else if (k == 7) cond = "true";
        else cond = "(" + x + " " + ops[k] + " " + y + ")";
    } else {
        fail("unsupported compare " + mnemonic(*cur));
        return;
    }
    const bool cmpx = (op & 0x10) != 0;
    body << "    { bool c_ = ((s[126] & 1u) != 0u) && (" << cond << ");\n";
    if (sdst < 126 || sdst == operand::kVccLo)
        body << "      s[" << sdst << "] = c_ ? 1u : 0u; s[" << sdst + 1 << "] = 0u;\n";
    if (cmpx) body << "      s[126] = c_ ? 1u : 0u; s[127] = 0u;\n";
    body << "    }\n";
}

bool Emitter::valu_op2(u32 op, u32 d, u32 a, u32 b, u32 c, bool vop3) {
    const std::string ua = usrc(a), ub = usrc(b);
    auto fa = [&] { return fsrc(a, 0); };
    auto fb = [&] { return fsrc(b, 1); };
    const std::string carry_dst = vop3 ? std::to_string(cur->sdst) : "106";
    const std::string carry_in = vop3 ? ("(s[" + std::to_string(c) + "] & 1u)") : "(s[106] & 1u)";
    const std::string old = "v[" + std::to_string(d) + "]";
    switch (op) {
    case 0: write_v(d, "(" + std::string(vop3 ? "(s[" + std::to_string(c) + "] & 1u)" : "(s[106] & 1u)") +
                       " != 0u ? " + ub + " : " + ua + ")"); return true;
    case 3: write_v(d, float_result(fa() + " + " + fb())); return true;
    case 4: write_v(d, float_result(fa() + " - " + fb())); return true;
    case 5: write_v(d, float_result(fb() + " - " + fa())); return true;
    case 6: track_v(d); write_v(d, float_result("gcn_mul_legacy(" + fa() + ", " + fb() + ") + F(" + old + ")")); return true;
    case 7: write_v(d, float_result("gcn_mul_legacy(" + fa() + ", " + fb() + ")")); return true;
    case 8: write_v(d, float_result(fa() + " * " + fb())); return true;
    case 9: write_v(d, "uint(gcn_sext(" + ua + " & 0xffffffu, 24u) * gcn_sext(" + ub + " & 0xffffffu, 24u))"); return true;
    case 10: write_v(d, "uint((long(gcn_sext(" + ua + " & 0xffffffu, 24u)) * long(gcn_sext(" + ub + " & 0xffffffu, 24u))) >> 32)"); return true;
    case 11: write_v(d, "((" + ua + " & 0xffffffu) * (" + ub + " & 0xffffffu))"); return true;
    case 12: write_v(d, "uint((ulong(" + ua + " & 0xffffffu) * ulong(" + ub + " & 0xffffffu)) >> 32)"); return true;
    case 13: write_v(d, float_result("(" + fa() + " < " + fb() + " ? " + fa() + " : " + fb() + ")")); return true;
    case 14: write_v(d, float_result("(" + fa() + " > " + fb() + " ? " + fa() + " : " + fb() + ")")); return true;
    case 15: write_v(d, float_result("fmin(" + fa() + ", " + fb() + ")")); return true;
    case 16: write_v(d, float_result("fmax(" + fa() + ", " + fb() + ")")); return true;
    case 17: write_v(d, "uint(min(int(" + ua + "), int(" + ub + ")))"); return true;
    case 18: write_v(d, "uint(max(int(" + ua + "), int(" + ub + ")))"); return true;
    case 19: write_v(d, "min(" + ua + ", " + ub + ")"); return true;
    case 20: write_v(d, "max(" + ua + ", " + ub + ")"); return true;
    case 21: write_v(d, "(" + ua + " >> (" + ub + " & 31u))"); return true;
    case 22: write_v(d, "(" + ub + " >> (" + ua + " & 31u))"); return true;
    case 23: write_v(d, "uint(int(" + ua + ") >> int(" + ub + " & 31u))"); return true;
    case 24: write_v(d, "uint(int(" + ub + ") >> int(" + ua + " & 31u))"); return true;
    case 25: write_v(d, "(" + ua + " << (" + ub + " & 31u))"); return true;
    case 26: write_v(d, "(" + ub + " << (" + ua + " & 31u))"); return true;
    case 27: write_v(d, "(" + ua + " & " + ub + ")"); return true;
    case 28: write_v(d, "(" + ua + " | " + ub + ")"); return true;
    case 29: write_v(d, "(" + ua + " ^ " + ub + ")"); return true;
    case 30: write_v(d, "(((1u << (" + ua + " & 31u)) - 1u) << (" + ub + " & 31u))"); return true;
    case 31: track_v(d); write_v(d, float_result(fa() + " * " + fb() + " + F(" + old + ")")); return true;
    case 32: write_v(d, float_result(fa() + " * F(" + hex(cur->literal) + ") + " + fb())); return true;
    case 33: write_v(d, float_result(fa() + " * " + fb() + " + F(" + hex(cur->literal) + ")")); return true;
    case 34: write_v(d, "(popcount(" + ua + ") + " + ub + ")"); return true;
    case 35: case 36: write_v(d, ub); return true;   // single-lane model: lane id 0
    case 37: case 38: case 39: case 40: case 41: case 42: {
        std::string x = ua, y = ub;
        if (op == 39 || op == 42) std::swap(x, y);
        const bool add = (op == 37 || op == 40);
        const bool with_cin = op >= 40;
        const std::string cin = with_cin ? carry_in : "0u";
        body << "    { ulong r_ = " << (add ? "ulong(" + x + ") + ulong(" + y + ") + ulong(" + cin + ")"
                                            : "ulong(" + x + ") - ulong(" + y + ") - ulong(" + cin + ")") << ";\n";
        body << "      " << exec_guard() << vreg(d) << " = uint(r_);\n";
        body << "      s[" << carry_dst << "] = (((s[126] & 1u) != 0u) && ((r_ >> 32) != 0ul)) ? 1u : 0u; s["
             << (vop3 ? std::to_string(cur->sdst + 1) : std::string("107")) << "] = 0u; }\n";
        return true;
    }
    case 43: write_v(d, float_result("ldexp(" + fa() + ", int(" + ub + "))")); return true;
    case 44: track_v(d); write_v(d, "((" + old + " & ~(0xffu << ((" + ub + " & 3u) * 8u))) | ((uint(clamp(rint(" + fa() +
                             "), 0.0f, 255.0f)) & 0xffu) << ((" + ub + " & 3u) * 8u)))"); return true;
    case 45: write_v(d, "pack_float_to_snorm2x16(float2(" + fa() + ", " + fb() + "))"); return true;
    case 46: write_v(d, "pack_float_to_unorm2x16(float2(" + fa() + ", " + fb() + "))"); return true;
    case 47: write_v(d, "gcn_pkrtz(" + fa() + ", " + fb() + ")"); return true;
    case 48: write_v(d, "(min(" + ua + ", 65535u) | (min(" + ub + ", 65535u) << 16))"); return true;
    case 49: write_v(d, "((uint(clamp(int(" + ua + "), -32768, 32767)) & 0xffffu) | (uint(clamp(int(" + ub +
                       "), -32768, 32767)) << 16))"); return true;
    default: return false;
    }
}

bool Emitter::valu_op1(u32 op, u32 d, u32 a) {
    const std::string ua = usrc(a);
    auto fa = [&] { return fsrc(a, 0); };
    switch (op) {
    case 0: return true;
    case 1: case 9: write_v(d, ua); return true;
    case 2:   // v_readfirstlane_b32: destination is an SGPR
        body << "    s[" << d << "] = " << ua << ";\n";
        return true;
    case 5: write_v(d, float_result("float(int(" + ua + "))")); return true;
    case 6: write_v(d, float_result("float(" + ua + ")")); return true;
    case 7: write_v(d, "uint(clamp(" + fa() + ", 0.0f, 4294967040.0f))"); return true;
    case 8: write_v(d, "uint(int(clamp(" + fa() + ", -2147483648.0f, 2147483520.0f)))"); return true;
    case 10: write_v(d, "uint(as_type<ushort>(half(" + fa() + ")))"); return true;
    case 11: write_v(d, float_result("gcn_hlo(" + ua + ")")); return true;
    case 12: write_v(d, "uint(int(floor(" + fa() + " + 0.5f)))"); return true;
    case 13: write_v(d, "uint(int(floor(" + fa() + ")))"); return true;
    // v_cvt_off_f32_i4: the low nibble as a signed 4-bit value, in 1/16ths
    // (a sample offset: 0x8 -> -0.5, 0x7 -> 0.4375).
    case 14: write_v(d, float_result("float(gcn_sext(" + ua + " & 0xfu, 4u)) * 0.0625f")); return true;
    case 17: case 18: case 19: case 20:
        write_v(d, float_result("float((" + ua + " >> " + std::to_string((op - 17) * 8) + "u) & 0xffu)"));
        return true;
    case 32: write_v(d, float_result("fract(" + fa() + ")")); return true;
    case 33: write_v(d, float_result("trunc(" + fa() + ")")); return true;
    case 34: write_v(d, float_result("ceil(" + fa() + ")")); return true;
    case 35: write_v(d, float_result("rint(" + fa() + ")")); return true;
    case 36: write_v(d, float_result("floor(" + fa() + ")")); return true;
    case 37: write_v(d, float_result("exp2(" + fa() + ")")); return true;
    case 38: case 39: write_v(d, float_result("log2(" + fa() + ")")); return true;
    case 40: case 41: case 42: case 43: write_v(d, float_result("(1.0f / " + fa() + ")")); return true;
    case 44: case 45: case 46: write_v(d, float_result("rsqrt(" + fa() + ")")); return true;
    case 51: write_v(d, float_result("sqrt(" + fa() + ")")); return true;
    case 53: write_v(d, float_result("sin(" + fa() + " * 6.2831853f)")); return true;
    case 54: write_v(d, float_result("cos(" + fa() + " * 6.2831853f)")); return true;
    case 55: write_v(d, "(~" + ua + ")"); return true;
    case 56: write_v(d, "reverse_bits(" + ua + ")"); return true;
    case 57: write_v(d, "gcn_ffbh(" + ua + ")"); return true;
    case 58: write_v(d, "gcn_ffbl(" + ua + ")"); return true;
    case 59: write_v(d, "gcn_ffbh(int(" + ua + ") < 0 ? ~" + ua + " : " + ua + ")"); return true;
    case 63: write_v(d, "uint(" + fa() + " == 0.0f ? 0 : int(((" + ua + " >> 23) & 0xffu)) - 126)"); return true;
    case 64: write_v(d, float_result("(" + fa() + " == 0.0f ? 0.0f : as_type<float>((" + ua +
                                     " & 0x807fffffu) | 0x3f000000u))")); return true;
    default: return false;
    }
}

bool Emitter::valu_op3(u32 op, u32 d, u32 a, u32 b, u32 c) {
    const std::string ua = usrc(a), ub = usrc(b), uc = usrc(c);
    auto fa = [&] { return fsrc(a, 0); };
    auto fb = [&] { return fsrc(b, 1); };
    auto fc = [&] { return fsrc(c, 2); };
    switch (op) {
    case 0x140: write_v(d, float_result("gcn_mul_legacy(" + fa() + ", " + fb() + ") + " + fc())); return true;
    case 0x141: write_v(d, float_result(fa() + " * " + fb() + " + " + fc())); return true;
    case 0x142: write_v(d, "uint(gcn_sext(" + ua + " & 0xffffffu, 24u) * gcn_sext(" + ub + " & 0xffffffu, 24u) + int(" + uc + "))"); return true;
    case 0x143: write_v(d, "((" + ua + " & 0xffffffu) * (" + ub + " & 0xffffffu) + " + uc + ")"); return true;
    case 0x148: write_v(d, "gcn_bfe_u(" + ua + ", " + ub + ", " + uc + ")"); return true;
    case 0x149: write_v(d, "gcn_bfe_i(" + ua + ", " + ub + ", " + uc + ")"); return true;
    case 0x14a: write_v(d, "((" + ua + " & " + ub + ") | (~" + ua + " & " + uc + "))"); return true;
    case 0x14b: write_v(d, float_result("fma(" + fa() + ", " + fb() + ", " + fc() + ")")); return true;
    case 0x14e: write_v(d, "uint(((ulong(" + ua + ") << 32) | ulong(" + ub + ")) >> (" + uc + " & 31u))"); return true;
    case 0x14f: write_v(d, "uint(((ulong(" + ua + ") << 32) | ulong(" + ub + ")) >> ((" + uc + " & 3u) * 8u))"); return true;
    case 0x151: write_v(d, float_result("fmin(fmin(" + fa() + ", " + fb() + "), " + fc() + ")")); return true;
    case 0x152: write_v(d, "uint(min(min(int(" + ua + "), int(" + ub + ")), int(" + uc + ")))"); return true;
    case 0x153: write_v(d, "min(min(" + ua + ", " + ub + "), " + uc + ")"); return true;
    case 0x154: write_v(d, float_result("fmax(fmax(" + fa() + ", " + fb() + "), " + fc() + ")")); return true;
    case 0x155: write_v(d, "uint(max(max(int(" + ua + "), int(" + ub + ")), int(" + uc + ")))"); return true;
    case 0x156: write_v(d, "max(max(" + ua + ", " + ub + "), " + uc + ")"); return true;
    case 0x157: write_v(d, float_result("gcn_med3(" + fa() + ", " + fb() + ", " + fc() + ")")); return true;
    case 0x158: write_v(d, "uint(gcn_med3i(int(" + ua + "), int(" + ub + "), int(" + uc + ")))"); return true;
    case 0x159: write_v(d, "gcn_med3u(" + ua + ", " + ub + ", " + uc + ")"); return true;
    case 0x15d: write_v(d, "(absdiff(" + ua + ", " + ub + ") + " + uc + ")"); return true;
    case 0x15e: write_v(d, "((" + uc + " & ~(0xffu << ((" + ub + " & 3u) * 8u))) | ((uint(clamp(rint(" + fa() +
                           "), 0.0f, 255.0f)) & 0xffu) << ((" + ub + " & 3u) * 8u)))"); return true;
    case 0x15f: write_v(d, float_result(fc() + " / " + fb())); return true;   // div_fixup(q, den, num)
    case 0x169: case 0x16b: write_v(d, "(" + ua + " * " + ub + ")"); return true;
    case 0x16a: write_v(d, "mulhi(" + ua + ", " + ub + ")"); return true;
    case 0x16c: write_v(d, "uint(mulhi(int(" + ua + "), int(" + ub + ")))"); return true;
    case 0x16d:   // v_div_scale_f32: no scaling needed with IEEE fp32 divide
        write_v(d, ua);
        body << "    s[" << cur->sdst << "] = 0u; s[" << cur->sdst + 1 << "] = 0u;\n";
        return true;
    case 0x16f: write_v(d, float_result("fma(" + fa() + ", " + fb() + ", " + fc() + ")")); return true;
    default: return false;
    }
}

void Emitter::emit_valu(const Instruction& i) {
    bool ok = true;
    switch (i.enc) {
    case Encoding::VOP2: ok = valu_op2(i.op, i.dst, i.src0, i.src1, 0, false); break;
    case Encoding::VOP1: ok = valu_op1(i.op, i.dst, i.src0); break;
    case Encoding::VOPC: valu_cmp(i.op, operand::kVccLo, i.src0, i.src1); break;
    case Encoding::VOP3:
        if (i.op < 0x100) valu_cmp(i.op, i.sdst, i.src0, i.src1);
        else if (i.op < 0x140) ok = valu_op2(i.op - 0x100, i.dst, i.src0, i.src1, i.src2, true);
        else if (i.op < 0x180) ok = valu_op3(i.op, i.dst, i.src0, i.src1, i.src2);
        else ok = valu_op1(i.op - 0x180, i.dst, i.src0);
        break;
    default: ok = false; break;
    }
    if (!ok) fail("unsupported VALU " + mnemonic(i));
}

void Emitter::emit_salu(const Instruction& i) {
    if (i.enc == Encoding::SOPP) {
        switch (i.op) {
        case 10:
            if (stage == Stage::Compute) {
                body << "    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n";
            }
            return;
        case 0: case 12: case 13: case 14: case 15: case 16: case 17: case 19: case 20: case 21: case 22:
            return;
        default:
            fail("unsupported " + mnemonic(i));
            return;
        }
    }
    const std::string d = "s[" + std::to_string(i.dst) + "]";
    track_s(i.dst);
    if (i.enc == Encoding::SOPK) {
        const i32 k = i.simm;
        const u32 uk = static_cast<u32>(i.simm) & 0xFFFF;
        switch (i.op) {
        case 0: body << "    " << d << " = " << hex(static_cast<u32>(k)) << ";\n"; return;
        case 2: body << "    if (scc != 0u) " << d << " = " << hex(static_cast<u32>(k)) << ";\n"; return;
        case 3: case 4: case 5: case 6: case 7: case 8: {
            static const char* ops[] = {"==", "!=", ">", ">=", "<", "<="};
            body << "    scc = (int(" << d << ") " << ops[i.op - 3] << " " << k << ") ? 1u : 0u;\n";
            return;
        }
        case 9: case 10: case 11: case 12: case 13: case 14: {
            static const char* ops[] = {"==", "!=", ">", ">=", "<", "<="};
            body << "    scc = (" << d << " " << ops[i.op - 9] << " " << uk << "u) ? 1u : 0u;\n";
            return;
        }
        case 15:
            body << "    { int r_ = int(" << d << ") + " << k << "; scc = ((int(" << d << ") >= 0) == (" << k
                 << " >= 0) && (r_ >= 0) != (" << k << " >= 0)) ? 1u : 0u; " << d << " = uint(r_); }\n";
            return;
        case 16: body << "    " << d << " = uint(int(" << d << ") * " << k << ");\n"; return;
        default: fail("unsupported " + mnemonic(i)); return;
        }
    }
    if (i.enc == Encoding::SOPC) {
        const std::string a = usrc(i.src0), b = usrc(i.src1);
        static const char* ops[] = {"==", "!=", ">", ">=", "<", "<="};
        if (i.op <= 5) body << "    scc = (int(" << a << ") " << ops[i.op] << " int(" << b << ")) ? 1u : 0u;\n";
        else if (i.op <= 11) body << "    scc = (" << a << " " << ops[i.op - 6] << " " << b << ") ? 1u : 0u;\n";
        else if (i.op == 12) body << "    scc = ((" << a << " >> (" << b << " & 31u)) & 1u) == 0u ? 1u : 0u;\n";
        else if (i.op == 13) body << "    scc = ((" << a << " >> (" << b << " & 31u)) & 1u);\n";
        else fail("unsupported " + mnemonic(i));
        return;
    }
    if (i.enc == Encoding::SOP1) {
        const std::string a = usrc(i.src0);
        switch (i.op) {
        case 3: body << "    " << d << " = " << a << ";\n"; return;
        case 4: body << "    gcn_w64(s, " << i.dst << "u, " << usrc64(i.src0) << ");\n"; return;
        case 5: body << "    if (scc != 0u) " << d << " = " << a << ";\n"; return;
        case 6: body << "    if (scc != 0u) gcn_w64(s, " << i.dst << "u, " << usrc64(i.src0) << ");\n"; return;
        case 7: body << "    " << d << " = ~" << a << "; scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 8: body << "    gcn_w64(s, " << i.dst << "u, ~" << usrc64(i.src0) << "); scc = gcn_s64(s, " << i.dst
                     << "u) != 0ul ? 1u : 0u;\n"; return;
        case 9: case 10:   // s_wqm: lane-bit model -> whole quad active if this lane is
            body << "    { ulong m_ = " << usrc64(i.src0) << "; m_ = (m_ & 0xful) != 0ul ? (m_ | 0xful) : m_; ";
            if (i.op == 9) body << d << " = uint(m_);";
            else body << "gcn_w64(s, " << i.dst << "u, m_);";
            body << " scc = m_ != 0ul ? 1u : 0u; }\n";
            return;
        case 11: body << "    " << d << " = reverse_bits(" << a << ");\n"; return;
        case 15: body << "    " << d << " = popcount(" << a << "); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 19: body << "    " << d << " = gcn_ffbl(" << a << ");\n"; return;
        case 21: body << "    " << d << " = gcn_ffbh(" << a << ");\n"; return;
        case 25: body << "    " << d << " = uint(gcn_sext(" << a << " & 0xffu, 8u));\n"; return;
        case 26: body << "    " << d << " = uint(gcn_sext(" << a << " & 0xffffu, 16u));\n"; return;
        case 27: body << "    " << d << " &= ~(1u << (" << a << " & 31u));\n"; return;
        case 29: body << "    " << d << " |= (1u << (" << a << " & 31u));\n"; return;
        case 36: case 37: case 38: case 39: case 40: case 41: case 42: case 43: {
            uses_exec_write = true;
            static const char* f[] = {"m_ & e_", "m_ | e_", "m_ ^ e_", "m_ & ~e_", "m_ | ~e_", "~(m_ & e_)",
                                      "~(m_ | e_)", "~(m_ ^ e_)"};
            body << "    { ulong m_ = " << usrc64(i.src0) << "; ulong e_ = gcn_s64(s, 126u); gcn_w64(s, " << i.dst
                 << "u, e_); ulong r_ = " << f[i.op - 36] << "; gcn_w64(s, 126u, r_ & 1ul); scc = (r_ & 1ul) != 0ul ? 1u : 0u; }\n";
            return;
        }
        case 52: body << "    " << d << " = uint(abs(int(" << a << "))); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        default: fail("unsupported " + mnemonic(i)); return;
        }
    }
    if (i.enc == Encoding::SOP2) {
        const std::string a = usrc(i.src0), b = usrc(i.src1);
        auto bin32 = [&](const char* op) {
            body << "    " << d << " = " << a << " " << op << " " << b << "; scc = " << d << " != 0u ? 1u : 0u;\n";
        };
        auto bin64 = [&](const std::string& expr) {
            body << "    { ulong a_ = " << usrc64(i.src0) << ", b_ = " << usrc64(i.src1) << "; ulong r_ = " << expr
                 << "; gcn_w64(s, " << i.dst << "u, r_); scc = r_ != 0ul ? 1u : 0u; }\n";
        };
        if (i.dst == operand::kExecLo && i.op >= 15 && i.op <= 29 && (i.op & 1)) uses_exec_write = true;
        switch (i.op) {
        case 0: body << "    { ulong r_ = ulong(" << a << ") + ulong(" << b << "); " << d
                     << " = uint(r_); scc = uint(r_ >> 32); }\n"; return;
        case 1: body << "    scc = " << b << " > " << a << " ? 1u : 0u; " << d << " = " << a << " - " << b << ";\n"; return;
        case 2: body << "    " << d << " = uint(int(" << a << ") + int(" << b << "));\n"; return;
        case 3: body << "    " << d << " = uint(int(" << a << ") - int(" << b << "));\n"; return;
        case 4: body << "    { ulong r_ = ulong(" << a << ") + ulong(" << b << ") + ulong(scc); " << d
                     << " = uint(r_); scc = uint(r_ >> 32); }\n"; return;
        case 5: body << "    { ulong r_ = ulong(" << a << ") - ulong(" << b << ") - ulong(scc); " << d
                     << " = uint(r_); scc = (r_ >> 32) != 0ul ? 1u : 0u; }\n"; return;
        case 6: body << "    scc = int(" << a << ") < int(" << b << ") ? 1u : 0u; " << d << " = scc != 0u ? " << a << " : " << b << ";\n"; return;
        case 7: body << "    scc = " << a << " < " << b << " ? 1u : 0u; " << d << " = scc != 0u ? " << a << " : " << b << ";\n"; return;
        case 8: body << "    scc = int(" << a << ") > int(" << b << ") ? 1u : 0u; " << d << " = scc != 0u ? " << a << " : " << b << ";\n"; return;
        case 9: body << "    scc = " << a << " > " << b << " ? 1u : 0u; " << d << " = scc != 0u ? " << a << " : " << b << ";\n"; return;
        case 10: body << "    " << d << " = scc != 0u ? " << a << " : " << b << ";\n"; return;
        case 11: body << "    gcn_w64(s, " << i.dst << "u, scc != 0u ? " << usrc64(i.src0) << " : " << usrc64(i.src1) << ");\n"; return;
        case 14: bin32("&"); return;
        case 15: bin64("a_ & b_"); return;
        case 16: bin32("|"); return;
        case 17: bin64("a_ | b_"); return;
        case 18: bin32("^"); return;
        case 19: bin64("a_ ^ b_"); return;
        case 20: body << "    " << d << " = " << a << " & ~" << b << "; scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 21: bin64("a_ & ~b_"); return;
        case 22: body << "    " << d << " = " << a << " | ~" << b << "; scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 23: bin64("a_ | ~b_"); return;
        case 24: body << "    " << d << " = ~(" << a << " & " << b << "); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 25: bin64("~(a_ & b_)"); return;
        case 26: body << "    " << d << " = ~(" << a << " | " << b << "); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 27: bin64("~(a_ | b_)"); return;
        case 28: body << "    " << d << " = ~(" << a << " ^ " << b << "); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 29: bin64("~(a_ ^ b_)"); return;
        case 30: body << "    " << d << " = " << a << " << (" << b << " & 31u); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 31: bin64("a_ << (b_ & 63ul)"); return;
        case 32: body << "    " << d << " = " << a << " >> (" << b << " & 31u); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 33: bin64("a_ >> (b_ & 63ul)"); return;
        case 34: body << "    " << d << " = uint(int(" << a << ") >> int(" << b << " & 31u)); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        case 36: body << "    " << d << " = ((1u << (" << a << " & 31u)) - 1u) << (" << b << " & 31u);\n"; return;
        case 38: body << "    " << d << " = uint(int(" << a << ") * int(" << b << "));\n"; return;
        case 39: body << "    " << d << " = gcn_bfe_u(" << a << ", " << b << " & 31u, (" << b << " >> 16) & 0x7fu); scc = "
                      << d << " != 0u ? 1u : 0u;\n"; return;
        case 40: body << "    " << d << " = gcn_bfe_i(" << a << ", " << b << " & 31u, (" << b << " >> 16) & 0x7fu); scc = "
                      << d << " != 0u ? 1u : 0u;\n"; return;
        case 44: body << "    " << d << " = uint(abs(int(" << a << ") - int(" << b << "))); scc = " << d << " != 0u ? 1u : 0u;\n"; return;
        default: fail("unsupported " + mnemonic(i)); return;
        }
    }
}

constexpr u32 kNoUniformSlot = ~0u;
constexpr u32 kMaxUniformLoads = 1000;  // Metal's setBytes takes up to 4 KB

/// Where each scalar load of the shader's start goes in the table of loads
/// evaluated when the draw or dispatch is encoded (kNoUniformSlot: the shader
/// loads it itself). Slot 0 is a flag: 1 when scan_resources evaluated the
/// table, else the shader takes its own loads.
///
/// Up to the first branch target (or backward or unconditional branch) every
/// load runs at most once, in program order, with values the draw fixes: a
/// forward conditional branch can only skip some of them. scan_resources walks
/// the same instructions linearly, so it gets the same values.
///
/// Two reasons to read them from the table. Speed: on every fragment the load
/// would otherwise search the range table and read memory for each dword,
/// which dominated post-processing shaders. Correctness for compute: dispatches
/// read guest memory directly when the GPU runs them, which is later than on
/// the console; by then a game may have reused the constant memory (Dragon
/// Ball Xenoverse 2's texture copies read the next copy's size and wrote
/// nothing).
std::vector<u32> uniform_load_slots(const Program& prog) {
    static const u32 kCount[] = {1, 2, 4, 8, 16};
    std::set<u32> targets;
    for (const Instruction& i : prog.insts)
        if (i.enc == Encoding::SOPP && i.op >= 2 && i.op <= 9)
            targets.insert(static_cast<u32>(static_cast<i64>(i.pc) + i.size + i.simm));
    std::vector<u32> slots(prog.insts.size(), kNoUniformSlot);
    u32 next = 1;
    for (usize k = 0; k < prog.insts.size(); ++k) {
        const Instruction& i = prog.insts[k];
        if (targets.count(i.pc)) break;
        if (i.enc == Encoding::SOPP && i.op == 2) break;  // s_branch
        if (i.enc == Encoding::SOPP && i.op >= 4 && i.op <= 9 && i.simm < 0) break;
        if (i.enc == Encoding::SOP1 && (i.op == 32 || i.op == 33)) break;  // s_setpc, s_swappc
        if (i.enc != Encoding::SMRD || i.op == 29 || i.op == 30 || i.op == 31) continue;
        if ((i.op & 7) > 4 || (i.op & ~7u) > 8 || !(i.imm || i.offset == 255)) continue;
        const u32 n = kCount[i.op & 7];
        if (next + n > kMaxUniformLoads) break;
        slots[k] = next;
        next += n;
    }
    return slots;
}

void Emitter::emit_smrd(const Instruction& i) {
    static const u32 kCount[] = {1, 2, 4, 8, 16};
    const u32 kind = i.op & 7;
    if (i.op == 29 || i.op == 31) return;            // cache invalidations
    if (i.op == 30) {
        track_s(i.dst); track_s(i.dst + 1);
        body << "    s[" << i.dst << "] = 0u; s[" << i.dst + 1 << "] = 0u;\n";
        return;
    }
    if (kind > 4 || (i.op & ~7u) > 8) { fail("unsupported " + mnemonic(i)); return; }
    const u32 n = kCount[kind];
    for (u32 k = 0; k < n; ++k) track_s(i.dst + k);
    // Evaluated once per draw on the CPU: read the answer from the draw's table.
    const usize index = static_cast<usize>(&i - prog.insts.data());
    const bool uniform = index < uniform_slots.size() && uniform_slots[index] != kNoUniformSlot;
    if (uniform && stage != Stage::Compute) {
        // A draw whose loads cannot be evaluated is rejected before it is
        // encoded, so the table is always valid here. Keeping the shader's own
        // load as a fallback would stop Metal from optimising it away.
        for (u32 k = 0; k < n; ++k)
            body << "    s[" << i.dst + k << "] = ul[" << uniform_slots[index] + k << "];\n";
        return;
    }
    if (uniform) {
        // Compute runs even when the scan failed (ul[0] == 0).
        body << "    if (ul[0] != 0u) {\n";
        for (u32 k = 0; k < n; ++k)
            body << "      s[" << i.dst + k << "] = ul[" << uniform_slots[index] + k << "];\n";
        body << "    } else\n";
    }
    track_s(i.sbase); track_s(i.sbase + 1);
    if (!i.imm && i.offset < 128) track_s(i.offset);
    std::string off;
    if (i.imm) off = std::to_string(i.offset * 4) + "ul";
    else if (i.offset == 255) off = std::to_string(static_cast<u64>(i.literal) * 4) + "ul";
    else off = "ulong(s[" + std::to_string(i.offset) + "])";
    const u32 sb = i.sbase;
    std::string base;
    if (i.op >= 8) base = "gcn_vbase(s[" + std::to_string(sb) + "], s[" + std::to_string(sb + 1) + "])";
    else base = "(gcn_s64(s, " + std::to_string(sb) + "u) & 0xfffffffffffful)";
    body << "    { ulong a_ = " << base << " + " << off << ";\n";
    // Load into temporaries first: the destination may overlap sbase.
    body << "      uint t_[" << n << "];\n";
    body << "      for (uint k_ = 0; k_ < " << n << "u; ++k_) t_[k_] = gcn_ld32(mem, dc, a_ + ulong(k_) * 4ul);\n";
    body << "      for (uint k_ = 0; k_ < " << n << "u; ++k_) s[" << i.dst << "u + k_] = t_[k_]; }\n";
}

void Emitter::emit_buffer(const Instruction& i) {
    const bool tbuf = i.enc == Encoding::MTBUF;
    const bool is_store = tbuf ? (i.op >= 4) : ((i.op >= 4 && i.op <= 7) || i.op >= 24);
    // Format stores: tbuffer_store_format_* (MTBUF) and buffer_store_format_*
    // (MUBUF 4..7) convert through the descriptor's (or instruction's) format.
    const bool format_store = is_store && (tbuf || i.op <= 7);
    const u32 r = i.srsrc;
    track_s(r); track_s(r + 1); track_s(r + 2); track_s(r + 3);
    std::string idx = "0u", off = "0u";
    if (i.idxen) { idx = vreg(i.vaddr); if (i.offen) off = vreg(i.vaddr + 1); }
    else if (i.offen) off = vreg(i.vaddr);
    const std::string w0 = "s[" + std::to_string(r) + "]", w1 = "s[" + std::to_string(r + 1) + "]",
                      w2 = "s[" + std::to_string(r + 2) + "]", w3 = "s[" + std::to_string(r + 3) + "]";
    body << "    { ulong a_ = 0ul; bool in_ = gcn_buf_addr(" << w0 << ", " << w1 << ", " << w2 << ", " << idx << ", "
         << off << " + " << usrc(i.soffset) << " + " << i.offset << "u, a_);\n";
    const std::string g = exec_guard();
    if (format_store) {
        const u32 comps = (i.op & 3) + 1;
        const std::string dfmt = tbuf ? std::to_string(i.dfmt) + "u" : "((" + w3 + " >> 15) & 0xfu)";
        const std::string nfmt = tbuf ? std::to_string(i.nfmt) + "u" : "((" + w3 + " >> 12) & 0x7u)";
        std::string v4 = "uint4(";
        for (u32 c = 0; c < 4; ++c) v4 += (c ? ", " : "") + (c < comps ? vreg(i.dst + c) : std::string("0u"));
        v4 += ")";
        body << "      " << g << "if (in_) gcn_store_format(mem, dc, a_, " << dfmt << ", " << nfmt << ", " << v4
             << ");\n    }\n";
        return;
    }
    if (is_store) {
        switch (i.op) {
        case 24: body << "      " << g << "if (in_) gcn_st8(mem, dc, a_, " << vreg(i.dst) << ");\n"; break;
        case 26: body << "      " << g << "if (in_) gcn_st16(mem, dc, a_, " << vreg(i.dst) << ");\n"; break;
        case 28: body << "      " << g << "if (in_) gcn_st32(mem, dc, a_, " << vreg(i.dst) << ");\n"; break;
        case 29:
            body << "      " << g << "if (in_) { gcn_st32(mem, dc, a_, " << vreg(i.dst) << "); gcn_st32(mem, dc, a_ + 4ul, "
                 << vreg(i.dst + 1) << "); }\n";
            break;
        case 31:
            body << "      " << g << "if (in_) { gcn_st32(mem, dc, a_, " << vreg(i.dst) << "); gcn_st32(mem, dc, a_ + 4ul, "
                 << vreg(i.dst + 1) << "); gcn_st32(mem, dc, a_ + 8ul, " << vreg(i.dst + 2) << "); }\n";
            break;
        case 30:
            body << "      " << g << "if (in_) { gcn_st32(mem, dc, a_, " << vreg(i.dst) << "); gcn_st32(mem, dc, a_ + 4ul, "
                 << vreg(i.dst + 1) << "); gcn_st32(mem, dc, a_ + 8ul, " << vreg(i.dst + 2)
                 << "); gcn_st32(mem, dc, a_ + 12ul, " << vreg(i.dst + 3) << "); }\n";
            break;
        default: fail("unsupported " + mnemonic(i) + " (stores)"); break;
        }
        body << "    }\n";
        return;
    }
    if (tbuf || i.op <= 3) {
        const u32 comps = (i.op & 3) + 1;
        const std::string dfmt = tbuf ? std::to_string(i.dfmt) + "u" : "((" + w3 + " >> 15) & 0xfu)";
        const std::string nfmt = tbuf ? std::to_string(i.nfmt) + "u" : "((" + w3 + " >> 12) & 0x7u)";
        body << "      uint nf_ = " << nfmt << "; bool ii_ = (nf_ == 4u || nf_ == 5u);\n";
        body << "      uint4 r_ = in_ ? gcn_fetch_format(mem, dc, a_, " << dfmt << ", nf_) : uint4(0u);\n";
        static const char* comp = "xyzw";
        for (u32 c = 0; c < comps; ++c) {
            body << "      " << g << vreg(i.dst + c) << " = in_ ? gcn_sel(r_, (" << w3 << " >> " << c * 3
                 << "u) & 7u, ii_) : 0u;\n";
            (void)comp;
        }
    } else {
        switch (i.op) {
        case 8: body << "      " << g << vreg(i.dst) << " = in_ ? gcn_ld8(mem, dc, a_) : 0u;\n"; break;
        case 9: body << "      " << g << vreg(i.dst) << " = in_ ? uint(gcn_sext(gcn_ld8(mem, dc, a_), 8u)) : 0u;\n"; break;
        case 10: body << "      " << g << vreg(i.dst) << " = in_ ? gcn_ld16(mem, dc, a_) : 0u;\n"; break;
        case 11: body << "      " << g << vreg(i.dst) << " = in_ ? uint(gcn_sext(gcn_ld16(mem, dc, a_), 16u)) : 0u;\n"; break;
        case 12: case 13: case 14: case 15: {
            const u32 n = i.op == 12 ? 1 : i.op == 13 ? 2 : i.op == 14 ? 4 : 3;
            for (u32 c = 0; c < n; ++c)
                body << "      " << g << vreg(i.dst + c) << " = in_ ? gcn_ld32(mem, dc, a_ + " << c * 4 << "ul) : 0u;\n";
            break;
        }
        default: fail("unsupported " + mnemonic(i)); break;
        }
    }
    body << "    }\n";
}

void Emitter::emit_mimg(const Instruction& i) {
    const bool is_load = i.op == 0x00 || i.op == 0x01;
    const bool is_store = i.op == 0x08 || i.op == 0x09;   // image_store / image_store_mip
    // image_sample{,_l,_b,_lz}_o (0x30/0x34/0x35/0x37) are the base ops with a
    // packed texel offset in the first address VGPR.
    const bool has_offset = i.op == 0x30 || i.op == 0x34 || i.op == 0x35 || i.op == 0x37;
    const u32 sop = has_offset ? i.op - 0x10 : i.op;
    const bool is_sample = sop == 0x20 || sop == 0x24 || sop == 0x25 || sop == 0x27;
    if (!is_load && !is_sample && !is_store) { fail("unsupported " + mnemonic(i)); return; }
    if (i.da) { fail("array/cube images unsupported"); return; }
    u32 slot = 0;
    bool found = false;
    for (u32 k = 0; k < images.size(); ++k) {
        if (images[k].srsrc == i.srsrc && (!is_sample || images[k].ssamp == i.ssamp) &&
            images[k].sampled == is_sample && images[k].storage == is_store) { slot = k; found = true; break; }
    }
    if (!found) {
        slot = static_cast<u32>(images.size());
        images.push_back(ImageSlot{i.srsrc, is_sample ? i.ssamp : ~0u, is_sample, is_store});
    }
    const std::string t = "tex" + std::to_string(slot);
    const std::string g = exec_guard();
    if (is_store) {
        // vdata holds one VGPR per dmask bit, in component order; the rest
        // are zero. The mip of image_store_mip is ignored: host surfaces have
        // one level.
        static const char* comp_zero[4] = {"0.0f", "0.0f", "0.0f", "0.0f"};
        std::string c[4];
        u32 src = i.dst;
        for (u32 k = 0; k < 4; ++k) c[k] = (i.dmask & (1u << k)) ? "F(" + vreg(src++) + ")" : comp_zero[k];
        const std::string x = vreg(i.vaddr), y = vreg(i.vaddr + 1);
        body << "    " << g << "if (" << x << " < " << t << ".get_width() && " << y << " < " << t
             << ".get_height()) " << t << ".write(float4(" << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3]
             << "), uint2(" << x << ", " << y << "));\n";
        return;
    }
    body << "    { float4 r_;\n";
    if (is_load) {
        body << "      r_ = " << t << ".read(uint2(" << vreg(i.vaddr) << ", " << vreg(i.vaddr + 1) << ")"
             << (i.op == 0x01 ? ", " + vreg(i.vaddr + 2) : std::string()) << ");\n";
    } else {
        u32 c = i.vaddr;
        std::string offset;
        if (has_offset) { offset = "gcn_texel_offset(" + vreg(c) + ")"; ++c; }
        std::string bias;
        if (sop == 0x25) { bias = "F(" + vreg(c) + ")"; ++c; }
        const std::string size = "float2(" + t + ".get_width(), " + t + ".get_height())";
        std::string uv = "float2(F(" + vreg(c) + "), F(" + vreg(c + 1) + "))";
        // The offset is in texels of the sampled level; Metal's offset
        // argument must be a constant, so it is folded into the coordinate.
        if (i.unorm) uv = "((" + uv + (has_offset ? " + " + offset : std::string()) + ") / " + size + ")";
        else if (has_offset) uv = "(" + uv + " + " + offset + " / " + size + ")";
        const std::string smp = "smp" + std::to_string(slot);
        if (sop == 0x20) body << "      r_ = " << t << ".sample(" << smp << ", " << uv << ");\n";
        else if (sop == 0x24) body << "      r_ = " << t << ".sample(" << smp << ", " << uv << ", level(F(" << vreg(c + 2) << ")));\n";
        else if (sop == 0x25) body << "      r_ = " << t << ".sample(" << smp << ", " << uv << ", bias(" << bias << "));\n";
        else body << "      r_ = " << t << ".sample(" << smp << ", " << uv << ", level(0.0f));\n";
    }
    u32 out = i.dst;
    static const char* comp = "xyzw";
    for (u32 k = 0; k < 4; ++k) {
        if (i.dmask & (1u << k)) {
            body << "      " << g << vreg(out) << " = U(r_." << comp[k] << ");\n";
            ++out;
        }
    }
    body << "    }\n";
}

void Emitter::emit_interp(const Instruction& i) {
    if (stage != Stage::Pixel) { fail("v_interp in non-pixel stage"); return; }
    u32 locn = i.attr;
    if (input_cntl) {
        const u32 cntl = input_cntl[i.attr];
        locn = cntl & 0x1F;
        if ((cntl >> 10) & 1) flat_attrs.insert(i.attr);
    }
    if (i.op == 2) flat_attrs.insert(i.attr);
    ps_attrs[i.attr] = locn;
    static const char* comp = "xyzw";
    write_v(i.dst, "U(in.a" + std::to_string(i.attr) + "." + comp[i.chan] + ")");
}

void Emitter::emit_export(const Instruction& i) {
    auto src = [&](u32 k) {
        const u32 r = k == 0 ? i.src0 : k == 1 ? i.src1 : k == 2 ? i.src2 : i.dst;
        return vreg(r);
    };
    std::string val;
    if (i.compr) {
        val = "float4(gcn_hlo(" + src(0) + "), gcn_hhi(" + src(0) + "), gcn_hlo(" + src(1) + "), gcn_hhi(" + src(1) + "))";
    } else {
        val = "float4(F(" + src(0) + "), F(" + src(1) + "), F(" + src(2) + "), F(" + src(3) + "))";
    }
    std::string mask;
    if (stage == Stage::Compute) { fail("exp in compute stage"); return; }
    // Components not enabled keep their previous value (zero initialised).
    if (stage == Stage::Vertex) {
        if (i.target == 12) {
            body << "    { float4 e_ = " << val << "; out.position = float4(" << ((i.en & 1) ? "e_.x" : "0.0f")
                 << ", " << ((i.en & 2) ? "e_.y" : "0.0f") << ", " << ((i.en & 4) ? "e_.z" : "0.0f") << ", "
                 << ((i.en & 8) ? "e_.w" : "1.0f") << "); }\n";
        } else if (i.target >= 32 && i.target < 64) {
            const u32 p = i.target - 32;
            exported_params.insert(p);
            body << "    { float4 e_ = " << val << ";\n";
            static const char* comp = "xyzw";
            for (u32 k = 0; k < 4; ++k)
                if (i.en & (1u << k)) body << "      out.p" << p << "." << comp[k] << " = e_." << comp[k] << ";\n";
            body << "    }\n";
        }
        // pos1..3 / null: ignored.
        return;
    }
    if (i.target < 8) { // mrt0..7
        const std::string c = "color" + std::to_string(i.target);
        ps_color_mask |= static_cast<u8>(1u << i.target);
        if (i.target == 0) ps_color = true;
        body << "    { float4 e_ = " << val << ";\n";
        static const char* comp = "xyzw";
        const u32 en = i.compr ? ((i.en & 1 ? 3u : 0u) | (i.en & 4 ? 12u : 0u)) : i.en;
        for (u32 k = 0; k < 4; ++k)
            if (en & (1u << k)) body << "      " << c << "." << comp[k] << " = e_." << comp[k] << ";\n";
        body << "    }\n";
    }
    if (i.target == 8 && (i.en & 1) && !i.compr) {
        // mrtz: x is the fragment depth (stencil/mask in y/z/w are ignored).
        ps_depth = true;
        body << "    depth0 = F(" << src(0) << ");\n";
    }
    // null exports are ignored.
}

void Emitter::emit_one(const Instruction& i, bool structured) {
    cur = &i;
    switch (i.enc) {
    case Encoding::SOPP:
        if (i.op == 1) {
            body << (structured ? "    running = false; break;\n" : "    // s_endpgm\n");
            return;
        }
        if (i.op >= 2 && i.op <= 9) {
            const u32 target = static_cast<u32>(static_cast<i64>(i.pc) + i.size + i.simm);
            std::string cond;
            switch (i.op) {
            case 2: cond = "true"; break;
            case 4: cond = "scc == 0u"; break;
            case 5: cond = "scc != 0u"; break;
            case 6: cond = "(s[106] | s[107]) == 0u"; break;
            case 7: cond = "(s[106] | s[107]) != 0u"; break;
            case 8: cond = "(s[126] | s[127]) == 0u"; break;
            case 9: cond = "(s[126] | s[127]) != 0u"; break;
            default: cond = "false"; break;
            }
            body << "    if (" << cond << ") { pc = " << target << "u; continue; }\n";
            return;
        }
        emit_salu(i);
        return;
    case Encoding::SOP1:
        if (i.op == 32 || i.op == 33 || i.op == 31 || i.op == 34) { fail("unsupported " + mnemonic(i)); return; }
        emit_salu(i);
        return;
    case Encoding::SOP2:
    case Encoding::SOPK:
    case Encoding::SOPC:
        emit_salu(i);
        return;
    case Encoding::SMRD: emit_smrd(i); return;
    case Encoding::VOP1:
    case Encoding::VOP2:
    case Encoding::VOPC:
    case Encoding::VOP3: emit_valu(i); return;
    case Encoding::MUBUF:
    case Encoding::MTBUF: emit_buffer(i); return;
    case Encoding::MIMG: emit_mimg(i); return;
    case Encoding::VINTRP: emit_interp(i); return;
    case Encoding::EXP: emit_export(i); return;
    default:
        fail("unsupported " + mnemonic(i));
        return;
    }
}

bool Emitter::run() {
    // Pre-scan: does anything write EXEC? (decides whether VALU writes need
    // an exec guard).
    for (const Instruction& i : prog.insts) {
        if (i.enc == Encoding::SOP1 && i.op >= 36 && i.op <= 43) uses_exec_write = true;
        if ((i.enc == Encoding::SOP1 || i.enc == Encoding::SOP2 || i.enc == Encoding::SOPK) &&
            (i.dst == operand::kExecLo || i.dst == operand::kExecHi) &&
            !(i.enc == Encoding::SOP1 && (i.op == 9 || i.op == 10)))
            uses_exec_write = true;
        if (i.enc == Encoding::VOPC && (i.op & 0x10)) uses_exec_write = true;
        if (i.enc == Encoding::VOP3 && i.op < 0x100 && (i.op & 0x10)) uses_exec_write = true;
        if (i.enc == Encoding::VOP3 && i.op < 0x100 && i.sdst == operand::kExecLo) uses_exec_write = true;
    }
    uniform_slots = uniform_load_slots(prog);
    const bool structured = prog.has_branches;
    std::set<u32> labels;
    if (structured) {
        labels.insert(0);
        for (const Instruction& i : prog.insts) {
            if (i.enc == Encoding::SOPP && i.op >= 2 && i.op <= 9)
                labels.insert(static_cast<u32>(static_cast<i64>(i.pc) + i.size + i.simm));
        }
        body << "  uint pc = 0u; bool running = true;\n";
        body << "  for (uint guard_ = 0u; running && guard_ < 100000u; ++guard_) {\n  switch (pc) {\n";
    }
    for (const Instruction& i : prog.insts) {
        if (structured && labels.count(i.pc)) body << "  case " << i.pc << "u:\n";
        emit_one(i, structured);
        if (!error.empty()) return false;
    }
    if (structured) {
        body << "  default: running = false; break;\n  }\n  running = false;\n  }\n";
    }
    return error.empty();
}

// Storage slots (image_store) are write-only textures; the rest are read.
const char* image_type(const ImageSlot& slot) {
    return slot.storage ? "texture2d<float, access::write>" : "texture2d<float>";
}

std::string image_params(const std::vector<ImageSlot>& images) {
    std::string p;
    for (u32 k = 0; k < images.size(); ++k) {
        p += ",\n    " + std::string(image_type(images[k])) + " tex" + std::to_string(k) + " [[texture(" +
             std::to_string(k) + ")]]";
        if (images[k].sampled)
            p += ",\n    sampler smp" + std::to_string(k) + " [[sampler(" + std::to_string(k) + ")]]";
    }
    return p;
}

std::string image_args(const std::vector<ImageSlot>& images) {
    std::string p;
    for (u32 k = 0; k < images.size(); ++k) {
        p += ", tex" + std::to_string(k);
        if (images[k].sampled) p += ", smp" + std::to_string(k);
    }
    return p;
}

std::string image_decl_args(const std::vector<ImageSlot>& images) {
    std::string p;
    for (u32 k = 0; k < images.size(); ++k) {
        p += ", " + std::string(image_type(images[k])) + " tex" + std::to_string(k);
        if (images[k].sampled) p += ", sampler smp" + std::to_string(k);
    }
    return p;
}

} // namespace

TranslatedPipeline translate_pipeline(const Program& vs, const Program& ps, const u32* ps_input_cntl,
                                      const TranslateOptions& opts) {
    TranslatedPipeline out;
    Emitter ev(vs, Stage::Vertex, nullptr, 0);
    if (!ev.run()) {
        out.error = "VS: " + ev.error;
        return out;
    }
    // No PS program: a depth-only draw (Gnm binds no pixel shader for Z
    // prepasses and shadow maps). Only the VS is emitted.
    const bool has_ps = !ps.insts.empty();
    Emitter ep(ps, Stage::Pixel, ps_input_cntl, 0);
    if (!ep.run()) {
        out.error = "PS: " + ep.error;
        return out;
    }

    // Interface: VS exports params 0..N-1; the PS reads a subset.
    u32 nparams = 0;
    for (u32 p : ev.exported_params) nparams = std::max(nparams, p + 1);
    for (auto [attr, locn] : ep.ps_attrs) nparams = std::max(nparams, locn + 1);
    std::set<u32> used_locn;
    for (auto [attr, locn] : ep.ps_attrs) {
        if (!used_locn.insert(locn).second) {
            out.error = "two PS attributes map to VS param " + std::to_string(locn);
            return out;
        }
    }

    std::ostringstream m;
    m << kPrelude << "\n";
    m << "struct GcnVsOut {\n    float4 position [[position]];\n";
    for (u32 p = 0; p < nparams; ++p) m << "    float4 p" << p << " [[user(locn" << p << ")]];\n";
    m << "};\n\n";

    // System values the hardware preloads into PS VGPRs (SPI_PS_INPUT_ADDR
    // allocates them in this order, SPI_PS_INPUT_ENA says which are loaded).
    // Barycentrics stay zero: v_interp reads the interpolated stage_in.
    std::ostringstream preload;
    bool uses_front_face = false;
    {
        static const u32 kSlots[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        const u32 layout = opts.ps_input_addr ? opts.ps_input_addr : opts.ps_input_ena;
        const u32 loaded = opts.ps_input_ena ? opts.ps_input_ena : layout;
        const u32 ps_vgprs = std::max<u32>(ep.max_vgpr, 4);
        u32 vg = 0;
        for (u32 bit = 0; bit < 16; ++bit) {
            if (!(layout & (1u << bit))) continue;
            const u32 k = vg;
            vg += kSlots[bit];
            if (!(loaded & (1u << bit)) || k >= ps_vgprs) continue;
            const std::string dst = "  v[" + std::to_string(k) + "] = ";
            switch (bit) {
            case 8: preload << dst << "U(in.position.x);\n"; break;
            case 9: preload << dst << "U(in.position.y);\n"; break;
            case 10: preload << dst << "U(in.position.z);\n"; break;
            // POS_W_FLOAT is clip-space w; Metal's [[position]].w is 1/w.
            case 11: preload << dst << "U(1.0f / in.position.w);\n"; break;
            case 12:
                uses_front_face = true;
                preload << dst << "front_face ? 0x3f800000u : 0xbf800000u;\n";
                break;
            case 14: preload << dst << "1u;\n"; break;   // SAMPLE_COVERAGE: one sample
            case 15: preload << dst << "(uint(in.position.x) & 0xFFFFu) | (uint(in.position.y) << 16);\n"; break;
            default: break;
            }
        }
    }

    m << "struct GcnPsIn {\n    float4 position [[position]];\n";
    for (auto [attr, locn] : ep.ps_attrs) {
        m << "    float4 a" << attr << " [[user(locn" << locn << ")]]";
        if (ep.flat_attrs.count(attr)) m << " [[flat]]";
        m << ";\n";
    }
    m << "};\n\n";

    const u32 vs_vgprs = std::max<u32>(ev.max_vgpr, 4);
    m << "static void gcn_vs_body(uint vid, uint iid, constant GcnDrawConsts& dc, device const uchar* mem, constant uint* ul"
      << image_decl_args(ev.images) << ", thread GcnVsOut& out) {\n";
    m << "  uint s[128]; uint v[" << vs_vgprs << "]; uint scc = 0u;\n";
    m << "  for (uint k = 0; k < 128u; ++k) s[k] = 0u;\n";
    m << "  for (uint k = 0; k < " << vs_vgprs << "u; ++k) v[k] = 0u;\n";
    m << "  for (uint k = 0; k < 16u; ++k) s[k] = dc.vs_user[k];\n";
    m << "  s[126] = 1u;\n";
    m << "  v[0] = vid; v[1] = iid; v[2] = iid; v[3] = iid;\n";
    m << "  out.position = float4(0.0f, 0.0f, 0.0f, 1.0f);\n";
    for (u32 p = 0; p < nparams; ++p) m << "  out.p" << p << " = float4(0.0f);\n";
    m << ev.body.str() << "}\n\n";

    m << "vertex GcnVsOut gcn_vs_main(uint vid [[vertex_id]], uint iid [[instance_id]],\n"
      << "    device const uchar* mem [[buffer(0)]], constant GcnDrawConsts& dc [[buffer(1)]], constant uint* ul [[buffer(2)]]"
      << image_params(ev.images) << ") {\n";
    m << "  GcnVsOut out;\n";
    using VR = TranslateOptions::VertexRemap;
    if (opts.remap == VR::None) {
        m << "  gcn_vs_body(vid, iid, dc, mem, ul" << image_args(ev.images) << ", out);\n";
    } else if (opts.remap == VR::QuadList) {
        m << "  uint q = vid / 6u, c = vid % 6u;\n";
        m << "  const uint map_[6] = {0u, 1u, 2u, 0u, 2u, 3u};\n";
        m << "  gcn_vs_body(dc.base_vertex + q * 4u + map_[c], iid, dc, mem, ul" << image_args(ev.images) << ", out);\n";
    } else {
        // RectList: corners 0,1,2 are given, corner 3 = v1 + v2 - v0.
        m << "  uint r = vid / 6u, c = vid % 6u;\n";
        m << "  const uint map_[6] = {0u, 1u, 2u, 2u, 1u, 3u};\n";
        m << "  uint k = map_[c]; uint b = dc.base_vertex + r * 3u;\n";
        m << "  if (k < 3u) {\n    gcn_vs_body(b + k, iid, dc, mem, ul" << image_args(ev.images) << ", out);\n  } else {\n";
        m << "    GcnVsOut o0, o1, o2;\n";
        m << "    gcn_vs_body(b, iid, dc, mem, ul" << image_args(ev.images) << ", o0);\n";
        m << "    gcn_vs_body(b + 1u, iid, dc, mem, ul" << image_args(ev.images) << ", o1);\n";
        m << "    gcn_vs_body(b + 2u, iid, dc, mem, ul" << image_args(ev.images) << ", o2);\n";
        m << "    out.position = o1.position + o2.position - o0.position;\n";
        for (u32 p = 0; p < nparams; ++p) m << "    out.p" << p << " = o1.p" << p << " + o2.p" << p << " - o0.p" << p << ";\n";
        m << "  }\n";
    }
    m << "  if ((dc.flags & " << kDrawFlipY << "u) != 0u) out.position.y = -out.position.y;\n";
    m << "  return out;\n}\n\n";

    if (has_ps) {
        const u32 ps_vgprs = std::max<u32>(ep.max_vgpr, 4);
        // One float4 return for mrt0 alone; a struct for more targets or depth.
        const u8 cmask = ep.ps_color_mask | 1; // color0 is always returned
        const bool ps_struct = ep.ps_depth || cmask != 1;
        if (ps_struct) {
            m << "struct GcnPsOut {\n";
            for (u32 n = 0; n < 8; ++n)
                if ((cmask >> n) & 1) m << "    float4 c" << n << " [[color(" << n << ")]];\n";
            if (ep.ps_depth) m << "    float depth [[depth(any)]];\n";
            m << "};\n\n";
        }
        m << "fragment " << (ps_struct ? "GcnPsOut" : "float4") << " gcn_ps_main(GcnPsIn in [[stage_in]],\n"
          << "    device const uchar* mem [[buffer(0)]], constant GcnDrawConsts& dc [[buffer(1)]], constant uint* ul [[buffer(2)]]"
          << (uses_front_face ? ", bool front_face [[front_facing]]" : "")
          << image_params(ep.images) << ") {\n";
        m << "  uint s[128]; uint v[" << ps_vgprs << "]; uint scc = 0u;\n";
        m << "  for (uint k = 0; k < 128u; ++k) s[k] = 0u;\n";
        m << "  for (uint k = 0; k < " << ps_vgprs << "u; ++k) v[k] = 0u;\n";
        m << "  for (uint k = 0; k < 16u; ++k) s[k] = dc.ps_user[k];\n";
        m << "  s[126] = 1u; s[124] = 0u;\n";
        m << preload.str();
        for (u32 n = 0; n < 8; ++n)
            if ((cmask >> n) & 1) m << "  float4 color" << n << " = float4(0.0f);\n";
        if (ep.ps_depth) m << "  float depth0 = in.position.z;\n";
        m << ep.body.str();
        if (ep.uses_exec_write) m << "  if ((s[126] & 1u) == 0u) discard_fragment();\n";
        if (ps_struct) {
            m << "  GcnPsOut o_;\n";
            for (u32 n = 0; n < 8; ++n)
                if ((cmask >> n) & 1) m << "  o_.c" << n << " = color" << n << ";\n";
            if (ep.ps_depth) m << "  o_.depth = depth0;\n";
            m << "  return o_;\n}\n";
        } else {
            m << "  return color0;\n}\n";
        }

    }

    out.ok = true;
    out.msl = m.str();
    out.vs_images = ev.images;
    out.ps_images = ep.images;
    out.vs_param_count = nparams;
    for (auto [attr, locn] : ep.ps_attrs) out.ps_attr_mask |= 1u << (attr & 31);
    out.ps_writes_color = ep.ps_color;
    out.ps_color_mask = ep.ps_color_mask;
    out.ps_writes_depth = ep.ps_depth;
    out.has_ps = has_ps;
    return out;
}

TranslatedComputePipeline translate_compute_pipeline(const Program& cs,
                                                    const ComputePipelineOptions& opts) {
    TranslatedComputePipeline out;
    Emitter ec(cs, Stage::Compute, nullptr, 0);
    if (!ec.run()) {
        out.error = "CS: " + ec.error;
        return out;
    }

    u32 cs_vgprs = std::max<u32>(ec.max_vgpr, 4);
    u32 cs_sgprs = std::max<u32>({ec.max_sgpr, 16u, 3u});
    if (opts.rsrc1 != 0) {
        const u32 rsrc1_vgprs = (((opts.rsrc1 >> 0) & 0x3fu) + 1u) * 4u;
        const u32 rsrc1_sgprs = (((opts.rsrc1 >> 6) & 0x0fu) + 1u) * 8u;
        cs_vgprs = std::max(cs_vgprs, rsrc1_vgprs);
        cs_sgprs = std::max(cs_sgprs, rsrc1_sgprs);
    }
    // COMPUTE_PGM_RSRC2.USER_SGPR (bits 1..5) is the number of user SGPRs;
    // the workgroup ids follow them. It was doubled, so tgid landed in
    // s[2n..] and kernels read garbage for their group position.
    if (opts.rsrc2 != 0) {
        const u32 user_sgprs = (opts.rsrc2 >> 1) & 0x1fu;
        cs_sgprs = std::max(cs_sgprs, user_sgprs);
    }

    std::ostringstream m;
    m << kPrelude << "\n";
    m << "kernel void gcn_cs_main(\n"
      << "    uint3 tid_in_tg [[thread_position_in_threadgroup]],\n"
      << "    uint3 tgid [[threadgroup_position_in_grid]],\n"
      << "    uint3 global_tid [[thread_position_in_grid]],\n"
      << "    device uchar* mem [[buffer(0)]],\n"
      << "    constant GcnComputeConsts& dc [[buffer(1)]],\n"
      << "    constant uint* ul [[buffer(2)]]"
      << image_params(ec.images) << ") {\n";
    m << "  uint s[128];\n";
    m << "  uint v[" << cs_vgprs << "];\n";
    m << "  uint scc = 0u;\n";
    m << "  for (uint k = 0; k < 128u; ++k) s[k] = 0u;\n";
    m << "  for (uint k = 0; k < " << cs_vgprs << "u; ++k) v[k] = 0u;\n";
    m << "  for (uint k = 0; k < 16u; ++k) s[k] = dc.cs_user[k];\n";
    m << "  s[126] = 1u; s[124] = 0u;\n\n";

    m << "  // Setup workgroup ID in SGPRs (by default s[0..2] = tgid.xyz)\n";
    if (opts.rsrc2 == 0) {
        m << "  s[0] = tgid.x;\n";
        m << "  s[1] = tgid.y;\n";
        m << "  s[2] = tgid.z;\n";
    } else {
        const u32 user_sgprs = (opts.rsrc2 >> 1) & 0x1fu;
        u32 s_idx = user_sgprs;
        if (opts.rsrc2 & (1u << 7)) { // TGID_X_EN
            m << "  s[" << s_idx++ << "] = tgid.x;\n";
        } else if (user_sgprs == 0 && (opts.rsrc2 & 0x380u) == 0) {
            m << "  s[0] = tgid.x;\n";
        }
        if (opts.rsrc2 & (1u << 8)) { // TGID_Y_EN
            m << "  s[" << s_idx++ << "] = tgid.y;\n";
        } else if (user_sgprs == 0 && (opts.rsrc2 & 0x380u) == 0) {
            m << "  s[1] = tgid.y;\n";
        }
        if (opts.rsrc2 & (1u << 9)) { // TGID_Z_EN
            m << "  s[" << s_idx++ << "] = tgid.z;\n";
        } else if (user_sgprs == 0 && (opts.rsrc2 & 0x380u) == 0) {
            m << "  s[2] = tgid.z;\n";
        }
    }

    m << "\n  // Setup thread ID in VGPR (v[0] = tid_in_tg.x)\n";
    m << "  v[0] = tid_in_tg.x;\n";
    if (opts.rsrc2 != 0) {
        const u32 tidig = (opts.rsrc2 >> 11) & 3u;
        if (tidig >= 1u) m << "  v[1] = tid_in_tg.y;\n";
        if (tidig >= 2u) m << "  v[2] = tid_in_tg.z;\n";
    }

    m << ec.body.str();
    m << "}\n";

    out.ok = true;
    out.msl = m.str();
    out.images = ec.images;
    out.max_vgpr = cs_vgprs;
    out.max_sgpr = cs_sgprs;
    out.uses_exec_write = ec.uses_exec_write;
    return out;
}

// ---------------------------------------------------------------------------
// Scalar resource pre-pass
// ---------------------------------------------------------------------------

ResourceScan scan_resources(const Program& prog, const std::vector<ImageSlot>& slots, const u32* user_sgprs,
                            u32 user_count, const MemReader& read, u32 max_vertex_index) {
    ResourceScan rs;
    rs.images.resize(slots.size());
    const std::vector<u32> uniform_slots = uniform_load_slots(prog);
    for (u32 slot : uniform_slots)
        if (slot != kNoUniformSlot) rs.uniform_loads.resize(slot + 16, 0);
    u32 val[128] = {};
    bool known[128] = {};
    for (u32 k = 0; k < user_count && k < 16; ++k) { val[k] = user_sgprs[k]; known[k] = true; }
    val[operand::kExecLo] = 1;
    known[operand::kExecLo] = known[operand::kExecHi] = true;
    val[operand::kM0] = 0;
    known[operand::kM0] = true;

    auto get = [&](u32 s, const Instruction& i, u32& out) -> bool {
        if (s < 128) { out = val[s]; return known[s]; }
        if (s == operand::kZero) { out = 0; return true; }
        if (s >= 129 && s <= 192) { out = s - 128; return true; }
        if (s >= 193 && s <= 208) { out = static_cast<u32>(-static_cast<i32>(s - 192)); return true; }
        if (s == operand::kLiteral) { out = i.literal; return true; }
        return false;
    };
    auto set = [&](u32 d, u32 v, bool k) { if (d < 128) { val[d] = v; known[d] = k; } };
    auto add_range = [&](u64 addr, u64 size) {
        if (size == 0) return;
        size = std::min<u64>(size, 16u * 1024 * 1024);
        rs.ranges.push_back(MemRange{addr, static_cast<u32>(size)});
    };
    auto need = [&](u32 first, u32 n, const char* what) -> bool {
        for (u32 k = 0; k < n; ++k)
            if (first + k >= 128 || !known[first + k]) {
                rs.error = std::string("cannot resolve ") + what + " descriptor in s[" + std::to_string(first) + "]";
                return false;
            }
        return true;
    };
    auto buffer_extent = [&](u32 w1, u32 w2) -> u64 {
        const u32 stride = (w1 >> 16) & 0x3FFF;
        if (stride == 0) return w2;
        const u64 recs = std::min<u64>(w2, static_cast<u64>(max_vertex_index) + 1 > w2 ? w2 : w2);
        return recs * stride + 16;
    };

    for (const Instruction& i : prog.insts) {
        switch (i.enc) {
        case Encoding::SMRD: {
            if (i.op == 29 || i.op == 31) break;
            const usize index = static_cast<usize>(&i - prog.insts.data());
            static const u32 kCount[] = {1, 2, 4, 8, 16};
            const u32 kind = i.op & 7;
            if (kind > 4 || i.op == 30) {
                for (u32 k = 0; k < 2; ++k) set(i.dst + k, 0, false);
                break;
            }
            const u32 n = kCount[kind];
            u32 offv = 0;
            bool off_known = true;
            if (i.imm) offv = i.offset * 4;
            else if (i.offset == 255) offv = i.literal * 4;
            else off_known = get(i.offset, i, offv);
            u64 base = 0;
            bool ok = false;
            if (i.op >= 8) {
                if (need(i.sbase, 4, "s_buffer_load")) {
                    base = static_cast<u64>(val[i.sbase]) | (static_cast<u64>(val[i.sbase + 1] & 0xFFFF) << 32);
                    const u64 extent = std::max<u64>(buffer_extent(val[i.sbase + 1], val[i.sbase + 2]),
                                                     static_cast<u64>(offv) + n * 4);
                    add_range(base, std::max<u64>(extent, 16));
                    ok = off_known;
                } else {
                    return rs;
                }
            } else {
                if (need(i.sbase, 2, "s_load")) {
                    base = (static_cast<u64>(val[i.sbase]) | (static_cast<u64>(val[i.sbase + 1]) << 32)) & 0xFFFFFFFFFFFFull;
                    if (!off_known) { rs.error = "s_load with unknown offset"; return rs; }
                    add_range(base + offv, n * 4);
                    ok = true;
                } else {
                    return rs;
                }
            }
            for (u32 k = 0; k < n; ++k) {
                u32 w = 0;
                const bool got = ok && read(base + offv + k * 4, &w, 4);
                set(i.dst + k, w, got);
                if (uniform_slots[index] != kNoUniformSlot) rs.uniform_loads[uniform_slots[index] + k] = w;
            }
            break;
        }
        case Encoding::SOP1: {
            u32 a = 0;
            if (i.op == 3) { const bool k = get(i.src0, i, a); set(i.dst, a, k); }
            else if (i.op == 4) {
                u32 hi = 0;
                bool k = get(i.src0, i, a);
                if (i.src0 < 127) k = k && get(i.src0 + 1, i, hi);
                else hi = (i.src0 >= 193 && i.src0 <= 208) ? 0xFFFFFFFFu : 0;
                set(i.dst, a, k);
                set(i.dst + 1, hi, k);
            } else if (i.op >= 36 && i.op <= 43) {
                set(i.dst, 0, false); set(i.dst + 1, 0, false);
                known[operand::kExecLo] = known[operand::kExecHi] = false;
            } else {
                set(i.dst, 0, false);
                if (i.op == 4 || i.op == 6 || i.op == 8 || i.op == 10) set(i.dst + 1, 0, false);
            }
            break;
        }
        case Encoding::SOPK:
            if (i.op == 0) set(i.dst, static_cast<u32>(i.simm), true);
            else if (i.op == 15) { set(i.dst, val[i.dst] + static_cast<u32>(i.simm), known[i.dst]); }
            else if (i.op == 16) { set(i.dst, static_cast<u32>(static_cast<i32>(val[i.dst]) * i.simm), known[i.dst]); }
            else if (i.op < 3 || i.op > 14) set(i.dst, 0, false);
            break;
        case Encoding::SOP2: {
            u32 a = 0, b = 0;
            const bool k = get(i.src0, i, a) && get(i.src1, i, b);
            u32 r = 0;
            bool handled = true;
            switch (i.op) {
            case 0: case 2: r = a + b; break;
            case 1: case 3: r = a - b; break;
            case 14: r = a & b; break;
            case 16: r = a | b; break;
            case 18: r = a ^ b; break;
            case 30: r = a << (b & 31); break;
            case 32: r = a >> (b & 31); break;
            case 38: r = a * b; break;
            case 39: { const u32 off = b & 31, w = (b >> 16) & 0x7F; r = w == 0 ? 0 : (a >> off) & (w >= 32 ? ~0u : ((1u << w) - 1)); break; }
            default: handled = false; break;
            }
            set(i.dst, r, k && handled);
            const bool is64 = (i.op & 1) && i.op >= 11 && i.op <= 37;
            if (is64 || i.op == 11) set(i.dst + 1, 0, false);
            break;
        }
        case Encoding::VOP1:
            if (i.op == 2) set(i.dst, 0, false);
            break;
        case Encoding::VOP2:
            if (i.op == 1) set(i.dst, 0, false);  // v_readlane writes an SGPR
            if (i.op >= 37 && i.op <= 42) { known[operand::kVccLo] = known[operand::kVccHi] = false; }
            break;
        case Encoding::VOP3:
            if (i.op < 0x100 || (i.op >= 0x125 && i.op <= 0x12A) || i.op == 0x16D) {
                set(i.sdst, 0, false); set(i.sdst + 1, 0, false);
            }
            if (i.op == 0x101 || i.op == 0x182) set(i.dst, 0, false);
            break;
        case Encoding::VOPC:
            known[operand::kVccLo] = known[operand::kVccHi] = false;
            break;
        case Encoding::MUBUF:
        case Encoding::MTBUF: {
            if (!need(i.srsrc, 4, "buffer")) return rs;
            const u32 w0 = val[i.srsrc], w1 = val[i.srsrc + 1], w2 = val[i.srsrc + 2];
            const u64 base = static_cast<u64>(w0) | (static_cast<u64>(w1 & 0xFFFF) << 32);
            u32 soff = 0;
            get(i.soffset, i, soff);
            add_range(base, buffer_extent(w1, w2) + soff + i.offset + 16);
            break;
        }
        case Encoding::MIMG: {
            const bool store = i.op == 0x08 || i.op == 0x09;
            const u32 sop = (i.op == 0x30 || i.op == 0x34 || i.op == 0x35 || i.op == 0x37) ? i.op - 0x10 : i.op;
            const bool sample = sop == 0x20 || sop == 0x24 || sop == 0x25 || sop == 0x27;
            for (u32 k = 0; k < slots.size(); ++k) {
                const ImageSlot& sl = slots[k];
                if (sl.srsrc != i.srsrc) continue;
                // A shader can reuse the T# registers (load from one image,
                // reload s[0:7], store to another): fill only the slot of
                // this instruction's kind.
                if (sl.storage != store || sl.sampled != sample) continue;
                if (sl.sampled && sl.ssamp != i.ssamp) continue;
                if (!need(i.srsrc, 8, "image T#")) return rs;
                for (u32 w = 0; w < 8; ++w) rs.images[k].tsharp[w] = val[i.srsrc + w];
                if (sl.sampled) {
                    if (!need(i.ssamp, 4, "sampler S#")) return rs;
                    for (u32 w = 0; w < 4; ++w) rs.images[k].ssharp[w] = val[i.ssamp + w];
                    rs.images[k].sampler_known = true;
                }
            }
            break;
        }
        default:
            break;
        }
    }

    // Coalesce overlapping / nearby ranges.
    std::sort(rs.ranges.begin(), rs.ranges.end(), [](const MemRange& a, const MemRange& b) { return a.addr < b.addr; });
    std::vector<MemRange> merged;
    for (const MemRange& r : rs.ranges) {
        if (!merged.empty()) {
            MemRange& m = merged.back();
            const u64 m_end = m.addr + m.size;
            if (r.addr <= m_end + 256) {
                const u64 end = std::max<u64>(m_end, r.addr + r.size);
                m.size = static_cast<u32>(end - m.addr);
                continue;
            }
        }
        merged.push_back(r);
    }
    while (merged.size() > binding::kMaxRanges) {
        // Merge the pair with the smallest gap.
        usize best = 0;
        u64 best_gap = ~0ull;
        for (usize k = 0; k + 1 < merged.size(); ++k) {
            const u64 gap = merged[k + 1].addr - (merged[k].addr + merged[k].size);
            if (gap < best_gap) { best_gap = gap; best = k; }
        }
        const u64 end = std::max<u64>(merged[best].addr + merged[best].size, merged[best + 1].addr + merged[best + 1].size);
        merged[best].size = static_cast<u32>(end - merged[best].addr);
        merged.erase(merged.begin() + static_cast<isize>(best) + 1);
    }
    rs.ranges = std::move(merged);
    rs.ok = true;
    if (!rs.uniform_loads.empty()) rs.uniform_loads[0] = 1;  // table evaluated
    return rs;
}

// ---------------------------------------------------------------------------
// T# / S# and texture conversion
// ---------------------------------------------------------------------------

TextureDesc decode_tsharp(const u32* t) {
    TextureDesc d;
    // Word 0 holds bits 39:8 of the base address; Word 1 bits 3:0 hold bits 43:40.
    d.base = (static_cast<u64>(t[0]) << 8) | (static_cast<u64>(t[1] & 0x0F) << 40);
    d.dfmt = (t[1] >> 20) & 0x3F;
    d.nfmt = (t[1] >> 26) & 0xF;
    d.width = (t[2] & 0x3FFF) + 1;
    d.height = ((t[2] >> 14) & 0x3FFF) + 1;
    for (u32 k = 0; k < 4; ++k) d.dst_sel[k] = (t[3] >> (k * 3)) & 7;
    d.base_level = (t[3] >> 12) & 0xF;
    d.last_level = (t[3] >> 16) & 0xF;
    d.tiling_index = (t[3] >> 20) & 0x1F;
    d.type = (t[3] >> 28) & 0xF;
    d.depth = (t[4] & 0x1FFF) + 1;
    d.pitch = ((t[4] >> 13) & 0x3FFF) + 1;
    return d;
}

SamplerDesc decode_ssharp(const u32* s) {
    SamplerDesc d;
    d.clamp_x = s[0] & 7;
    d.clamp_y = (s[0] >> 3) & 7;
    d.clamp_z = (s[0] >> 6) & 7;
    d.force_unnormalized = (s[0] >> 15) & 1;
    d.xy_mag_filter = (s[2] >> 20) & 3;
    d.xy_min_filter = (s[2] >> 22) & 3;
    d.mip_filter = (s[2] >> 26) & 3;
    return d;
}

u32 data_format_bytes(u32 dfmt) {
    switch (dfmt) {
    case 1: return 1;
    case 2: case 3: case 16: case 17: case 18: case 19: return 2;
    case 4: case 5: case 6: case 7: case 8: case 9: case 10: case 20: case 21: return 4;
    case 11: case 12: return 8;
    case 13: return 12;
    case 14: return 16;
    default: return bc::block_bytes(dfmt);  // BC1-BC7: bytes per 4x4 block
    }
}

/// Tiling index 8 (and 31) is linear; the other supported index is 8x8 micro tiled.
bool is_linear_tiling(u32 tiling_index) { return tiling_index == 8 || tiling_index == 31; }

usize texture_guest_size(const TextureDesc& td) {
    if (const u32 block = bc::block_bytes(td.dfmt)) {
        // Block-compressed: pitch and height count 4x4 blocks. Micro tiles are
        // 8x8 blocks, mirroring the element path below.
        const u32 pitch_b = (std::max(td.pitch, td.width) + 3) / 4;
        u32 h_b = (td.height + 3) / 4;
        u32 p_b = pitch_b;
        if (!is_linear_tiling(td.tiling_index)) {
            p_b = (pitch_b + 7) & ~7u;
            h_b = (h_b + 7) & ~7u;
        }
        return static_cast<usize>(p_b) * h_b * block;
    }
    const u32 bpp = data_format_bytes(td.dfmt);
    const u32 pitch = std::max(td.pitch, td.width);
    u32 h = td.height;
    if (td.tiling_index != 8) h = (h + 7) & ~7u;
    const u32 p = td.tiling_index != 8 ? ((pitch + 7) & ~7u) : pitch;
    return static_cast<usize>(p) * h * bpp;
}

namespace {

u8 to_u8(float f) {
    f = std::min(std::max(f, 0.0f), 1.0f);
    return static_cast<u8>(f * 255.0f + 0.5f);
}

float half_to_float(u16 h) {
    const u32 s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    u32 bits;
    if (e == 0) {
        if (m == 0) bits = s << 31;
        else {
            float f = static_cast<float>(m) / 1024.0f / 16384.0f;
            return s ? -f : f;
        }
    } else if (e == 31) bits = (s << 31) | 0x7F800000 | (m << 13);
    else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// Decode one element to RGBA float.
bool decode_element(u32 dfmt, u32 nfmt, const u8* p, float out[4]) {
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    auto unorm = [](u32 v, u32 bits) { return static_cast<float>(v) / static_cast<float>((1u << bits) - 1); };
    u32 w = 0;
    std::memcpy(&w, p, std::min<u32>(4, std::max<u32>(1, data_format_bytes(dfmt))));
    const bool is_float = nfmt == 7;
    switch (dfmt) {
    case 1: out[0] = unorm(p[0], 8); return true;
    case 3: out[0] = unorm(p[0], 8); out[1] = unorm(p[1], 8); return true;
    case 10: for (u32 k = 0; k < 4; ++k) out[k] = unorm(p[k], 8); return true;
    case 2: {
        const u16 v = static_cast<u16>(w & 0xFFFF);
        out[0] = is_float ? half_to_float(v) : unorm(v, 16);
        return true;
    }
    case 5: {
        for (u32 k = 0; k < 2; ++k) {
            const u16 v = static_cast<u16>((w >> (16 * k)) & 0xFFFF);
            out[k] = is_float ? half_to_float(v) : unorm(v, 16);
        }
        return true;
    }
    case 12: {
        for (u32 k = 0; k < 4; ++k) {
            u16 v;
            std::memcpy(&v, p + 2 * k, 2);
            out[k] = is_float ? half_to_float(v) : unorm(v, 16);
        }
        return true;
    }
    case 4: case 11: case 13: case 14: {
        const u32 n = dfmt == 4 ? 1 : dfmt == 11 ? 2 : dfmt == 13 ? 3 : 4;
        for (u32 k = 0; k < n; ++k) {
            u32 v;
            std::memcpy(&v, p + 4 * k, 4);
            if (is_float) std::memcpy(&out[k], &v, 4);
            else out[k] = static_cast<float>(v) / 4294967295.0f;
        }
        return true;
    }
    case 9:
        out[0] = unorm(w & 0x3FF, 10); out[1] = unorm((w >> 10) & 0x3FF, 10);
        out[2] = unorm((w >> 20) & 0x3FF, 10); out[3] = unorm(w >> 30, 2);
        return true;
    case 16: {   // 5_6_5
        const u16 v = static_cast<u16>(w);
        out[0] = unorm(v & 0x1F, 5); out[1] = unorm((v >> 5) & 0x3F, 6); out[2] = unorm(v >> 11, 5);
        return true;
    }
    case 17: {   // 1_5_5_5
        const u16 v = static_cast<u16>(w);
        out[0] = unorm(v & 0x1F, 5); out[1] = unorm((v >> 5) & 0x1F, 5); out[2] = unorm((v >> 10) & 0x1F, 5);
        out[3] = static_cast<float>(v >> 15);
        return true;
    }
    case 19: {   // 4_4_4_4
        const u16 v = static_cast<u16>(w);
        for (u32 k = 0; k < 4; ++k) out[k] = unorm((v >> (4 * k)) & 0xF, 4);
        return true;
    }
    default:
        return false;
    }
}

/// Element offset of (x, y) inside an 8x8 thin micro tile (non-displayable
/// ordering: bit interleave x0 y0 x1 y1 x2 y2).
u32 micro_tile_index(u32 x, u32 y) {
    return (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
}

/// Writes one texel's RGBA floats as 8 bits per channel, applying the T# dst_sel swizzle.
void store_texel(u8* o, const u32* dst_sel, const float c[4]) {
    for (u32 k = 0; k < 4; ++k) {
        const u32 sel = dst_sel[k];
        const float v = sel == 0 ? 0.0f : sel == 1 ? 1.0f : (sel >= 4 ? c[sel - 4] : 0.0f);
        o[k] = to_u8(v);
    }
}

/// BC1-BC7 texture. Tiling works as on the element path, with a 4x4 block as
/// the element (micro tiles are 8x8 blocks). Edge blocks are cropped to the
/// texture size.
bool bc_texture_to_rgba8(const TextureDesc& td, const u8* src, bool linear, std::vector<u8>& out,
                         std::string& error) {
    const u32 bb = bc::block_bytes(td.dfmt);
    const u32 blocks_w = (td.width + 3) / 4;
    const u32 blocks_h = (td.height + 3) / 4;
    const u32 pitch_b = (std::max(td.pitch, td.width) + 3) / 4;
    const u32 tiles_per_row = (pitch_b + 7) / 8;
    // RGBA8Unorm cannot hold negative values, so signed BC4/BC5 data is stored
    // remapped from [-1,1] to [0,1]. Shaders that expect raw SNORM will see an
    // offset of 0.5. BC6H is clamped to [0,1] by to_u8 (HDR range is lost), and
    // SRGB data is kept as-is because the upload is RGBA8Unorm.
    const bool remap_signed = td.nfmt == 1 && (td.dfmt == 38 || td.dfmt == 39);
    out.assign(static_cast<usize>(td.width) * td.height * 4, 0);
    float block[16][4];
    for (u32 by = 0; by < blocks_h; ++by) {
        for (u32 bx = 0; bx < blocks_w; ++bx) {
            usize b;
            if (linear) {
                b = static_cast<usize>(by) * pitch_b + bx;
            } else {
                const usize tile = static_cast<usize>(by / 8) * tiles_per_row + (bx / 8);
                b = tile * 64 + micro_tile_index(bx & 7, by & 7);
            }
            if (!bc::decode_block(td.dfmt, td.nfmt, src + b * bb, block)) {
                error = "unsupported BC block, format " + std::to_string(td.dfmt) + "/" +
                        std::to_string(td.nfmt);
                return false;
            }
            for (u32 py = 0; py < 4; ++py) {
                for (u32 px = 0; px < 4; ++px) {
                    const u32 x = bx * 4 + px, y = by * 4 + py;
                    if (x >= td.width || y >= td.height) continue;
                    float c[4];
                    for (u32 k = 0; k < 4; ++k) {
                        const float v = block[py * 4 + px][k];
                        c[k] = remap_signed ? v * 0.5f + 0.5f : v;
                    }
                    store_texel(out.data() + (static_cast<usize>(y) * td.width + x) * 4, td.dst_sel, c);
                }
            }
        }
    }
    return true;
}

} // namespace

bool texture_to_rgba8(const TextureDesc& td, const u8* src, usize src_size, std::vector<u8>& out,
                      std::string& error) {
    const u32 bpp = data_format_bytes(td.dfmt);
    if (bpp == 0) {
        error = "unsupported texture data format " + std::to_string(td.dfmt);
        return false;
    }
    const bool linear = is_linear_tiling(td.tiling_index);
    const bool micro = td.tiling_index == 13 || td.tiling_index == 9;
    if (!linear && !micro) {
        error = "unsupported tiling index " + std::to_string(td.tiling_index);
        return false;
    }
    if (texture_guest_size(td) > src_size) {
        error = "texture memory out of range";
        return false;
    }
    if (bc::block_bytes(td.dfmt) != 0) return bc_texture_to_rgba8(td, src, linear, out, error);
    const u32 pitch = std::max(td.pitch, td.width);
    const u32 tiles_per_row = (pitch + 7) / 8;
    out.assign(static_cast<usize>(td.width) * td.height * 4, 0);

    // Fast paths for the common linear unorm formats (a game's software-
    // rendered screen or a decoded movie frame is converted every frame):
    // integer channel selects instead of the per-texel float decode.
    // dst_sel: 0 -> 0, 1 -> 1.0 (255), 4..7 -> source channel 0..3.
    if (linear && td.nfmt == 0 && (td.dfmt == 10 || td.dfmt == 16)) {
        u8 sel[4];
        bool ok = true;
        for (u32 k = 0; k < 4; ++k) {
            sel[k] = static_cast<u8>(td.dst_sel[k]);
            ok = ok && (sel[k] == 0 || sel[k] == 1 || (sel[k] >= 4 && sel[k] <= 7));
        }
        if (ok) {
            for (u32 y = 0; y < td.height; ++y) {
                u8* o = out.data() + static_cast<usize>(y) * td.width * 4;
                if (td.dfmt == 10) {
                    const u8* row = src + static_cast<usize>(y) * pitch * 4;
                    for (u32 x = 0; x < td.width; ++x) {
                        const u8* t = row + x * 4;
                        for (u32 k = 0; k < 4; ++k)
                            o[x * 4 + k] = sel[k] >= 4 ? t[sel[k] - 4] : (sel[k] == 1 ? 255 : 0);
                    }
                } else { // 5_6_5: channel 0 = bits 4:0, 1 = 10:5, 2 = 15:11 (A = 1)
                    const u16* row = reinterpret_cast<const u16*>(src) + static_cast<usize>(y) * pitch;
                    for (u32 x = 0; x < td.width; ++x) {
                        u16 v;
                        std::memcpy(&v, row + x, 2);
                        const u32 r5 = v & 0x1F, g6 = (v >> 5) & 0x3F, b5 = v >> 11;
                        const u8 c[4] = {static_cast<u8>((r5 * 527 + 23) >> 6), static_cast<u8>((g6 * 259 + 33) >> 6),
                                         static_cast<u8>((b5 * 527 + 23) >> 6), 255};
                        for (u32 k = 0; k < 4; ++k)
                            o[x * 4 + k] = sel[k] >= 4 ? c[sel[k] - 4] : (sel[k] == 1 ? 255 : 0);
                    }
                }
            }
            return true;
        }
    }
    float c[4];
    for (u32 y = 0; y < td.height; ++y) {
        for (u32 x = 0; x < td.width; ++x) {
            usize elem;
            if (linear) {
                elem = static_cast<usize>(y) * pitch + x;
            } else {
                const usize tile = static_cast<usize>(y / 8) * tiles_per_row + (x / 8);
                elem = tile * 64 + micro_tile_index(x & 7, y & 7);
            }
            if (!decode_element(td.dfmt, td.nfmt, src + elem * bpp, c)) {
                error = "unsupported texture format " + std::to_string(td.dfmt) + "/" + std::to_string(td.nfmt);
                return false;
            }
            store_texel(out.data() + (static_cast<usize>(y) * td.width + x) * 4, td.dst_sel, c);
        }
    }
    return true;
}

} // namespace ps4::gcn
