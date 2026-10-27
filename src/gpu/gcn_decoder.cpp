// GCN (Sea Islands / PS4 "Liverpool") instruction decoder + disassembler.
//
// Encodings and opcode numbers follow the AMD "Sea Islands Series
// Instruction Set Architecture" reference (GCN 1.1). The PS4 GPU uses the
// CI encodings, which differ from GCN3/Vega (VI) for most VALU opcodes.

#include "ps4/gcn_shader.hpp"

#include <cstdio>
#include <cstring>
#include <map>

namespace ps4::gcn {

namespace {

struct OpName {
    u32 op;
    const char* name;
};

const char* lookup(const OpName* table, usize n, u32 op) {
    for (usize i = 0; i < n; ++i)
        if (table[i].op == op) return table[i].name;
    return nullptr;
}

#define TABLE_LOOKUP(t, op) lookup(t, sizeof(t) / sizeof(t[0]), op)

const OpName kSop2[] = {
    {0, "s_add_u32"}, {1, "s_sub_u32"}, {2, "s_add_i32"}, {3, "s_sub_i32"},
    {4, "s_addc_u32"}, {5, "s_subb_u32"}, {6, "s_min_i32"}, {7, "s_min_u32"},
    {8, "s_max_i32"}, {9, "s_max_u32"}, {10, "s_cselect_b32"}, {11, "s_cselect_b64"},
    {14, "s_and_b32"}, {15, "s_and_b64"}, {16, "s_or_b32"}, {17, "s_or_b64"},
    {18, "s_xor_b32"}, {19, "s_xor_b64"}, {20, "s_andn2_b32"}, {21, "s_andn2_b64"},
    {22, "s_orn2_b32"}, {23, "s_orn2_b64"}, {24, "s_nand_b32"}, {25, "s_nand_b64"},
    {26, "s_nor_b32"}, {27, "s_nor_b64"}, {28, "s_xnor_b32"}, {29, "s_xnor_b64"},
    {30, "s_lshl_b32"}, {31, "s_lshl_b64"}, {32, "s_lshr_b32"}, {33, "s_lshr_b64"},
    {34, "s_ashr_i32"}, {35, "s_ashr_i64"}, {36, "s_bfm_b32"}, {37, "s_bfm_b64"},
    {38, "s_mul_i32"}, {39, "s_bfe_u32"}, {40, "s_bfe_i32"}, {41, "s_bfe_u64"},
    {42, "s_bfe_i64"}, {43, "s_cbranch_g_fork"}, {44, "s_absdiff_i32"},
};
const OpName kSopk[] = {
    {0, "s_movk_i32"}, {2, "s_cmovk_i32"}, {3, "s_cmpk_eq_i32"}, {4, "s_cmpk_lg_i32"},
    {5, "s_cmpk_gt_i32"}, {6, "s_cmpk_ge_i32"}, {7, "s_cmpk_lt_i32"}, {8, "s_cmpk_le_i32"},
    {9, "s_cmpk_eq_u32"}, {10, "s_cmpk_lg_u32"}, {11, "s_cmpk_gt_u32"}, {12, "s_cmpk_ge_u32"},
    {13, "s_cmpk_lt_u32"}, {14, "s_cmpk_le_u32"}, {15, "s_addk_i32"}, {16, "s_mulk_i32"},
    {18, "s_getreg_b32"}, {19, "s_setreg_b32"}, {21, "s_setreg_imm32_b32"},
};
const OpName kSop1[] = {
    {3, "s_mov_b32"}, {4, "s_mov_b64"}, {5, "s_cmov_b32"}, {6, "s_cmov_b64"},
    {7, "s_not_b32"}, {8, "s_not_b64"}, {9, "s_wqm_b32"}, {10, "s_wqm_b64"},
    {11, "s_brev_b32"}, {12, "s_brev_b64"}, {13, "s_bcnt0_i32_b32"}, {15, "s_bcnt1_i32_b32"},
    {16, "s_bcnt1_i32_b64"}, {17, "s_ff0_i32_b32"}, {19, "s_ff1_i32_b32"}, {20, "s_ff1_i32_b64"},
    {21, "s_flbit_i32_b32"}, {25, "s_sext_i32_i8"}, {26, "s_sext_i32_i16"},
    {27, "s_bitset0_b32"}, {29, "s_bitset1_b32"}, {31, "s_getpc_b64"}, {32, "s_setpc_b64"},
    {33, "s_swappc_b64"}, {34, "s_rfe_b64"}, {36, "s_and_saveexec_b64"},
    {37, "s_or_saveexec_b64"}, {38, "s_xor_saveexec_b64"}, {39, "s_andn2_saveexec_b64"},
    {40, "s_orn2_saveexec_b64"}, {41, "s_nand_saveexec_b64"}, {42, "s_nor_saveexec_b64"},
    {43, "s_xnor_saveexec_b64"}, {44, "s_quadmask_b32"}, {45, "s_quadmask_b64"},
    {46, "s_movrels_b32"}, {47, "s_movrels_b64"}, {48, "s_movreld_b32"},
    {49, "s_movreld_b64"}, {52, "s_abs_i32"},
};
const OpName kSopc[] = {
    {0, "s_cmp_eq_i32"}, {1, "s_cmp_lg_i32"}, {2, "s_cmp_gt_i32"}, {3, "s_cmp_ge_i32"},
    {4, "s_cmp_lt_i32"}, {5, "s_cmp_le_i32"}, {6, "s_cmp_eq_u32"}, {7, "s_cmp_lg_u32"},
    {8, "s_cmp_gt_u32"}, {9, "s_cmp_ge_u32"}, {10, "s_cmp_lt_u32"}, {11, "s_cmp_le_u32"},
    {12, "s_bitcmp0_b32"}, {13, "s_bitcmp1_b32"},
};
const OpName kSopp[] = {
    {0, "s_nop"}, {1, "s_endpgm"}, {2, "s_branch"}, {4, "s_cbranch_scc0"},
    {5, "s_cbranch_scc1"}, {6, "s_cbranch_vccz"}, {7, "s_cbranch_vccnz"},
    {8, "s_cbranch_execz"}, {9, "s_cbranch_execnz"}, {10, "s_barrier"},
    {12, "s_waitcnt"}, {13, "s_sethalt"}, {14, "s_sleep"}, {15, "s_setprio"},
    {16, "s_sendmsg"}, {17, "s_sendmsghalt"}, {18, "s_trap"}, {19, "s_icache_inv"},
    {20, "s_incperflevel"}, {21, "s_decperflevel"}, {22, "s_ttracedata"},
};
const OpName kSmrd[] = {
    {0, "s_load_dword"}, {1, "s_load_dwordx2"}, {2, "s_load_dwordx4"}, {3, "s_load_dwordx8"},
    {4, "s_load_dwordx16"}, {8, "s_buffer_load_dword"}, {9, "s_buffer_load_dwordx2"},
    {10, "s_buffer_load_dwordx4"}, {11, "s_buffer_load_dwordx8"}, {12, "s_buffer_load_dwordx16"},
    {29, "s_dcache_inv_vol"}, {30, "s_memtime"}, {31, "s_dcache_inv"},
};
const OpName kVop2[] = {
    {0, "v_cndmask_b32"}, {1, "v_readlane_b32"}, {2, "v_writelane_b32"}, {3, "v_add_f32"},
    {4, "v_sub_f32"}, {5, "v_subrev_f32"}, {6, "v_mac_legacy_f32"}, {7, "v_mul_legacy_f32"},
    {8, "v_mul_f32"}, {9, "v_mul_i32_i24"}, {10, "v_mul_hi_i32_i24"}, {11, "v_mul_u32_u24"},
    {12, "v_mul_hi_u32_u24"}, {13, "v_min_legacy_f32"}, {14, "v_max_legacy_f32"},
    {15, "v_min_f32"}, {16, "v_max_f32"}, {17, "v_min_i32"}, {18, "v_max_i32"},
    {19, "v_min_u32"}, {20, "v_max_u32"}, {21, "v_lshr_b32"}, {22, "v_lshrrev_b32"},
    {23, "v_ashr_i32"}, {24, "v_ashrrev_i32"}, {25, "v_lshl_b32"}, {26, "v_lshlrev_b32"},
    {27, "v_and_b32"}, {28, "v_or_b32"}, {29, "v_xor_b32"}, {30, "v_bfm_b32"},
    {31, "v_mac_f32"}, {32, "v_madmk_f32"}, {33, "v_madak_f32"}, {34, "v_bcnt_u32_b32"},
    {35, "v_mbcnt_lo_u32_b32"}, {36, "v_mbcnt_hi_u32_b32"}, {37, "v_add_i32"},
    {38, "v_sub_i32"}, {39, "v_subrev_i32"}, {40, "v_addc_u32"}, {41, "v_subb_u32"},
    {42, "v_subbrev_u32"}, {43, "v_ldexp_f32"}, {44, "v_cvt_pkaccum_u8_f32"},
    {45, "v_cvt_pknorm_i16_f32"}, {46, "v_cvt_pknorm_u16_f32"}, {47, "v_cvt_pkrtz_f16_f32"},
    {48, "v_cvt_pk_u16_u32"}, {49, "v_cvt_pk_i16_i32"},
};
const OpName kVop1[] = {
    {0, "v_nop"}, {1, "v_mov_b32"}, {2, "v_readfirstlane_b32"}, {5, "v_cvt_f32_i32"},
    {6, "v_cvt_f32_u32"}, {7, "v_cvt_u32_f32"}, {8, "v_cvt_i32_f32"}, {9, "v_mov_fed_b32"},
    {10, "v_cvt_f16_f32"}, {11, "v_cvt_f32_f16"}, {12, "v_cvt_rpi_i32_f32"},
    {13, "v_cvt_flr_i32_f32"}, {14, "v_cvt_off_f32_i4"}, {17, "v_cvt_f32_ubyte0"},
    {18, "v_cvt_f32_ubyte1"}, {19, "v_cvt_f32_ubyte2"}, {20, "v_cvt_f32_ubyte3"},
    {32, "v_fract_f32"}, {33, "v_trunc_f32"}, {34, "v_ceil_f32"}, {35, "v_rndne_f32"},
    {36, "v_floor_f32"}, {37, "v_exp_f32"}, {38, "v_log_clamp_f32"}, {39, "v_log_f32"},
    {40, "v_rcp_clamp_f32"}, {41, "v_rcp_legacy_f32"}, {42, "v_rcp_f32"},
    {43, "v_rcp_iflag_f32"}, {44, "v_rsq_clamp_f32"}, {45, "v_rsq_legacy_f32"},
    {46, "v_rsq_f32"}, {51, "v_sqrt_f32"}, {53, "v_sin_f32"}, {54, "v_cos_f32"},
    {55, "v_not_b32"}, {56, "v_bfrev_b32"}, {57, "v_ffbh_u32"}, {58, "v_ffbl_b32"},
    {59, "v_ffbh_i32"}, {63, "v_frexp_exp_i32_f32"}, {64, "v_frexp_mant_f32"},
    {66, "v_movreld_b32"}, {67, "v_movrels_b32"}, {68, "v_movrelsd_b32"},
};
const OpName kVop3Only[] = {
    {0x140, "v_mad_legacy_f32"}, {0x141, "v_mad_f32"}, {0x142, "v_mad_i32_i24"},
    {0x143, "v_mad_u32_u24"}, {0x144, "v_cubeid_f32"}, {0x145, "v_cubesc_f32"},
    {0x146, "v_cubetc_f32"}, {0x147, "v_cubema_f32"}, {0x148, "v_bfe_u32"},
    {0x149, "v_bfe_i32"}, {0x14a, "v_bfi_b32"}, {0x14b, "v_fma_f32"}, {0x14c, "v_fma_f64"},
    {0x14d, "v_lerp_u8"}, {0x14e, "v_alignbit_b32"}, {0x14f, "v_alignbyte_b32"},
    {0x150, "v_mullit_f32"}, {0x151, "v_min3_f32"}, {0x152, "v_min3_i32"},
    {0x153, "v_min3_u32"}, {0x154, "v_max3_f32"}, {0x155, "v_max3_i32"},
    {0x156, "v_max3_u32"}, {0x157, "v_med3_f32"}, {0x158, "v_med3_i32"},
    {0x159, "v_med3_u32"}, {0x15a, "v_sad_u8"}, {0x15b, "v_sad_hi_u8"},
    {0x15c, "v_sad_u16"}, {0x15d, "v_sad_u32"}, {0x15e, "v_cvt_pk_u8_f32"},
    {0x15f, "v_div_fixup_f32"}, {0x169, "v_mul_lo_u32"}, {0x16a, "v_mul_hi_u32"},
    {0x16b, "v_mul_lo_i32"}, {0x16c, "v_mul_hi_i32"}, {0x16d, "v_div_scale_f32"},
    {0x16f, "v_div_fmas_f32"},
};
const OpName kMubuf[] = {
    {0, "buffer_load_format_x"}, {1, "buffer_load_format_xy"}, {2, "buffer_load_format_xyz"},
    {3, "buffer_load_format_xyzw"}, {4, "buffer_store_format_x"}, {5, "buffer_store_format_xy"},
    {6, "buffer_store_format_xyz"}, {7, "buffer_store_format_xyzw"}, {8, "buffer_load_ubyte"},
    {9, "buffer_load_sbyte"}, {10, "buffer_load_ushort"}, {11, "buffer_load_sshort"},
    {12, "buffer_load_dword"}, {13, "buffer_load_dwordx2"}, {14, "buffer_load_dwordx4"},
    {15, "buffer_load_dwordx3"}, {24, "buffer_store_byte"}, {26, "buffer_store_short"},
    {28, "buffer_store_dword"}, {29, "buffer_store_dwordx2"}, {30, "buffer_store_dwordx4"},
    {31, "buffer_store_dwordx3"},
};
const OpName kMtbuf[] = {
    {0, "tbuffer_load_format_x"}, {1, "tbuffer_load_format_xy"}, {2, "tbuffer_load_format_xyz"},
    {3, "tbuffer_load_format_xyzw"}, {4, "tbuffer_store_format_x"}, {5, "tbuffer_store_format_xy"},
    {6, "tbuffer_store_format_xyz"}, {7, "tbuffer_store_format_xyzw"},
};
const OpName kMimg[] = {
    {0x00, "image_load"}, {0x01, "image_load_mip"}, {0x08, "image_store"},
    {0x09, "image_store_mip"}, {0x0e, "image_get_resinfo"}, {0x20, "image_sample"},
    {0x21, "image_sample_cl"}, {0x22, "image_sample_d"}, {0x23, "image_sample_d_cl"},
    {0x24, "image_sample_l"}, {0x25, "image_sample_b"}, {0x26, "image_sample_b_cl"},
    {0x27, "image_sample_lz"}, {0x28, "image_sample_c"}, {0x2c, "image_sample_c_l"},
    {0x2d, "image_sample_c_b"}, {0x2f, "image_sample_c_lz"}, {0x30, "image_sample_o"},
    {0x37, "image_sample_lz_o"}, {0x40, "image_gather4"}, {0x47, "image_gather4_lz"},
    {0x60, "image_get_lod"},
};

const char* vopc_name(u32 op) {
    static const char* kCond[16] = {"f", "lt", "eq", "le", "gt", "lg", "ge", "o",
                                    "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"};
    static const char* kIntCond[8] = {"f", "lt", "eq", "le", "gt", "ne", "ge", "t"};
    static thread_local char buf[48];
    const u32 group = op & ~0xFu;
    if (op < 0x20) {
        std::snprintf(buf, sizeof(buf), "v_cmp%s_%s_f32", (op & 0x10) ? "x" : "", kCond[op & 0xF]);
    } else if (op >= 0x20 && op < 0x40) {
        std::snprintf(buf, sizeof(buf), "v_cmp%s_%s_f64", (op & 0x10) ? "x" : "", kCond[op & 0xF]);
    } else if (group == 0x80 || group == 0x90 || group == 0xA0 || group == 0xB0 ||
               group == 0xC0 || group == 0xD0 || group == 0xE0 || group == 0xF0) {
        if ((op & 0xF) == 8) {
            std::snprintf(buf, sizeof(buf), "v_cmp%s_class_%s", (op & 0x10) ? "x" : "",
                          (op & 0x20) ? "f64" : "f32");
        } else {
            const char* ty = (group == 0x80 || group == 0x90) ? "i32"
                           : (group == 0xA0 || group == 0xB0) ? "i64"
                           : (group == 0xC0 || group == 0xD0) ? "u32" : "u64";
            std::snprintf(buf, sizeof(buf), "v_cmp%s_%s_%s", (op & 0x10) ? "x" : "",
                          kIntCond[op & 7], ty);
        }
    } else {
        std::snprintf(buf, sizeof(buf), "v_cmp_op%u", op);
    }
    return buf;
}

std::string sreg(u32 r) {
    char b[24];
    if (r < 104) std::snprintf(b, sizeof(b), "s%u", r);
    else if (r == operand::kVccLo) return "vcc_lo";
    else if (r == operand::kVccHi) return "vcc_hi";
    else if (r == operand::kM0) return "m0";
    else if (r == operand::kExecLo) return "exec_lo";
    else if (r == operand::kExecHi) return "exec_hi";
    else std::snprintf(b, sizeof(b), "sreg%u", r);
    return b;
}

std::string src_str(u32 s, const Instruction& inst) {
    char b[32];
    if (s >= operand::kVgprBase) { std::snprintf(b, sizeof(b), "v%u", s - 256); return b; }
    if (s < 128) return sreg(s);
    if (s == operand::kZero) return "0";
    if (s >= 129 && s <= 192) { std::snprintf(b, sizeof(b), "%u", s - 128); return b; }
    if (s >= 193 && s <= 208) { std::snprintf(b, sizeof(b), "-%u", s - 192); return b; }
    switch (s) {
    case 240: return "0.5";
    case 241: return "-0.5";
    case 242: return "1.0";
    case 243: return "-1.0";
    case 244: return "2.0";
    case 245: return "-2.0";
    case 246: return "4.0";
    case 247: return "-4.0";
    case operand::kVccz: return "vccz";
    case operand::kExecz: return "execz";
    case operand::kScc: return "scc";
    case operand::kLiteral:
        std::snprintf(b, sizeof(b), "0x%08x", inst.literal);
        return b;
    default:
        std::snprintf(b, sizeof(b), "src%u", s);
        return b;
    }
}

bool needs_literal32(const Instruction& i) {
    switch (i.enc) {
    case Encoding::SOP2:
    case Encoding::SOPC:
        return i.src0 == operand::kLiteral || i.src1 == operand::kLiteral;
    case Encoding::SOP1:
    case Encoding::VOP1:
        return i.src0 == operand::kLiteral;
    case Encoding::VOP2:
        return i.src0 == operand::kLiteral || i.op == 32 || i.op == 33;
    case Encoding::VOPC:
        return i.src0 == operand::kLiteral;
    case Encoding::SOPK:
        return i.op == 21;
    case Encoding::SMRD:
        return !i.imm && i.offset == 255;
    default:
        return false;
    }
}

} // namespace

const char* encoding_name(Encoding e) noexcept {
    switch (e) {
    case Encoding::SOP2: return "sop2";
    case Encoding::SOPK: return "sopk";
    case Encoding::SOP1: return "sop1";
    case Encoding::SOPC: return "sopc";
    case Encoding::SOPP: return "sopp";
    case Encoding::SMRD: return "smrd";
    case Encoding::VOP2: return "vop2";
    case Encoding::VOP1: return "vop1";
    case Encoding::VOPC: return "vopc";
    case Encoding::VOP3: return "vop3";
    case Encoding::VINTRP: return "vintrp";
    case Encoding::DS: return "ds";
    case Encoding::MUBUF: return "mubuf";
    case Encoding::MTBUF: return "mtbuf";
    case Encoding::MIMG: return "mimg";
    case Encoding::EXP: return "exp";
    default: return "unknown";
    }
}

bool decode_instruction(const u32* words, usize avail, u32 pc, Instruction& out) {
    if (avail == 0) return false;
    Instruction i{};
    i.pc = pc;
    const u32 w = words[0];
    i.dw0 = w;
    i.size = 1;

    if ((w & 0x80000000u) == 0) {
        if ((w >> 25) == 0x3F) {            // VOP1
            i.enc = Encoding::VOP1;
            i.dst = (w >> 17) & 0xFF;
            i.op = (w >> 9) & 0xFF;
            i.src0 = w & 0x1FF;
        } else if ((w >> 25) == 0x3E) {     // VOPC
            i.enc = Encoding::VOPC;
            i.op = (w >> 17) & 0xFF;
            i.src1 = ((w >> 9) & 0xFF) + operand::kVgprBase;
            i.src0 = w & 0x1FF;
        } else {                             // VOP2
            i.enc = Encoding::VOP2;
            i.op = (w >> 25) & 0x3F;
            i.dst = (w >> 17) & 0xFF;
            i.src1 = ((w >> 9) & 0xFF) + operand::kVgprBase;
            i.src0 = w & 0x1FF;
        }
    } else if ((w >> 30) == 0x2) {
        const u32 top9 = w >> 23;
        if (top9 == 0x17D) {                 // SOP1
            i.enc = Encoding::SOP1;
            i.dst = (w >> 16) & 0x7F;
            i.op = (w >> 8) & 0xFF;
            i.src0 = w & 0xFF;
        } else if (top9 == 0x17E) {          // SOPC
            i.enc = Encoding::SOPC;
            i.op = (w >> 16) & 0x7F;
            i.src1 = (w >> 8) & 0xFF;
            i.src0 = w & 0xFF;
        } else if (top9 == 0x17F) {          // SOPP
            i.enc = Encoding::SOPP;
            i.op = (w >> 16) & 0x7F;
            i.simm = static_cast<i16>(w & 0xFFFF);
        } else if ((w >> 28) == 0xB) {       // SOPK
            i.enc = Encoding::SOPK;
            i.op = (w >> 23) & 0x1F;
            i.dst = (w >> 16) & 0x7F;
            i.simm = static_cast<i16>(w & 0xFFFF);
        } else {                             // SOP2
            i.enc = Encoding::SOP2;
            i.op = (w >> 23) & 0x7F;
            i.dst = (w >> 16) & 0x7F;
            i.src1 = (w >> 8) & 0xFF;
            i.src0 = w & 0xFF;
        }
    } else if ((w >> 27) == 0x18) {          // SMRD
        i.enc = Encoding::SMRD;
        i.op = (w >> 22) & 0x1F;
        i.dst = (w >> 15) & 0x7F;
        i.sbase = ((w >> 9) & 0x3F) * 2;
        i.imm = (w >> 8) & 1;
        i.offset = w & 0xFF;
    } else {
        const u32 top6 = w >> 26;
        if (top6 == 0x32) {                  // VINTRP (32-bit)
            i.enc = Encoding::VINTRP;
            i.dst = (w >> 18) & 0xFF;
            i.op = (w >> 16) & 0x3;
            i.attr = (w >> 10) & 0x3F;
            i.chan = (w >> 8) & 0x3;
            i.src0 = (w & 0xFF) + operand::kVgprBase;
            out = i;
            return true;
        }
        if (avail < 2) return false;
        const u32 w1 = words[1];
        i.dw1 = w1;
        i.size = 2;
        switch (top6) {
        case 0x34: {                          // VOP3
            i.enc = Encoding::VOP3;
            i.op = (w >> 17) & 0x1FF;
            i.dst = w & 0xFF;
            const bool vop3b = (i.op >= 0x125 && i.op <= 0x12A) || i.op == 0x16D || i.op == 0x16E;
            const bool is_cmp = i.op < 0x100;
            if (is_cmp) {
                i.sdst = w & 0xFF;
                i.abs = (w >> 8) & 0x7;
            } else if (vop3b) {
                i.sdst = (w >> 8) & 0x7F;
            } else {
                i.abs = (w >> 8) & 0x7;
                i.clamp = (w >> 11) & 1;
            }
            i.src0 = w1 & 0x1FF;
            i.src1 = (w1 >> 9) & 0x1FF;
            i.src2 = (w1 >> 18) & 0x1FF;
            i.omod = (w1 >> 27) & 0x3;
            i.neg = (w1 >> 29) & 0x7;
            break;
        }
        case 0x32:                            // VINTRP (32-bit)
            i.enc = Encoding::VINTRP;
            i.size = 1;
            i.dw1 = 0;
            i.dst = (w >> 18) & 0xFF;
            i.op = (w >> 16) & 0x3;
            i.attr = (w >> 10) & 0x3F;
            i.chan = (w >> 8) & 0x3;
            i.src0 = (w & 0xFF) + operand::kVgprBase;
            break;
        case 0x36:                            // DS
            i.enc = Encoding::DS;
            i.op = (w >> 18) & 0xFF;
            break;
        case 0x38:                            // MUBUF
        case 0x3A:                            // MTBUF
            i.enc = top6 == 0x38 ? Encoding::MUBUF : Encoding::MTBUF;
            i.offset = w & 0xFFF;
            i.offen = (w >> 12) & 1;
            i.idxen = (w >> 13) & 1;
            i.glc = (w >> 14) & 1;
            if (top6 == 0x38) {
                i.op = (w >> 18) & 0x7F;
            } else {
                i.op = (w >> 16) & 0x7;
                i.dfmt = (w >> 19) & 0xF;
                i.nfmt = (w >> 23) & 0x7;
            }
            i.vaddr = w1 & 0xFF;
            i.dst = (w1 >> 8) & 0xFF;
            i.srsrc = ((w1 >> 16) & 0x1F) * 4;
            i.soffset = (w1 >> 24) & 0xFF;
            break;
        case 0x3C:                            // MIMG
            i.enc = Encoding::MIMG;
            i.dmask = (w >> 8) & 0xF;
            i.unorm = (w >> 12) & 1;
            i.glc = (w >> 13) & 1;
            i.da = (w >> 14) & 1;
            i.op = (w >> 18) & 0x7F;
            i.vaddr = w1 & 0xFF;
            i.dst = (w1 >> 8) & 0xFF;
            i.srsrc = ((w1 >> 16) & 0x1F) * 4;
            i.ssamp = ((w1 >> 21) & 0x1F) * 4;
            break;
        case 0x3E:                            // EXP
            i.enc = Encoding::EXP;
            i.en = w & 0xF;
            i.target = (w >> 4) & 0x3F;
            i.compr = (w >> 10) & 1;
            i.done = (w >> 11) & 1;
            i.vm = (w >> 12) & 1;
            i.src0 = w1 & 0xFF;
            i.src1 = (w1 >> 8) & 0xFF;
            i.src2 = (w1 >> 16) & 0xFF;
            i.dst = (w1 >> 24) & 0xFF;      // vsrc3
            break;
        default:
            i.enc = Encoding::Unknown;
            i.size = 1;
            break;
        }
    }

    if (needs_literal32(i)) {
        if (avail < i.size + 1) return false;
        i.literal = words[i.size];
        i.has_literal = true;
        i.size += 1;
    }
    out = i;
    return true;
}

std::string mnemonic(const Instruction& i) {
    const char* n = nullptr;
    switch (i.enc) {
    case Encoding::SOP2: n = TABLE_LOOKUP(kSop2, i.op); break;
    case Encoding::SOPK: n = TABLE_LOOKUP(kSopk, i.op); break;
    case Encoding::SOP1: n = TABLE_LOOKUP(kSop1, i.op); break;
    case Encoding::SOPC: n = TABLE_LOOKUP(kSopc, i.op); break;
    case Encoding::SOPP: n = TABLE_LOOKUP(kSopp, i.op); break;
    case Encoding::SMRD: n = TABLE_LOOKUP(kSmrd, i.op); break;
    case Encoding::VOP2: n = TABLE_LOOKUP(kVop2, i.op); break;
    case Encoding::VOP1: n = TABLE_LOOKUP(kVop1, i.op); break;
    case Encoding::VOPC: n = vopc_name(i.op); break;
    case Encoding::VOP3:
        if (i.op < 0x100) n = vopc_name(i.op);
        else if (i.op < 0x140) n = TABLE_LOOKUP(kVop2, i.op - 0x100);
        else if (i.op < 0x180) n = TABLE_LOOKUP(kVop3Only, i.op);
        else n = TABLE_LOOKUP(kVop1, i.op - 0x180);
        break;
    case Encoding::VINTRP:
        n = i.op == 0 ? "v_interp_p1_f32" : i.op == 1 ? "v_interp_p2_f32" : "v_interp_mov_f32";
        break;
    case Encoding::MUBUF: n = TABLE_LOOKUP(kMubuf, i.op); break;
    case Encoding::MTBUF: n = TABLE_LOOKUP(kMtbuf, i.op); break;
    case Encoding::MIMG: n = TABLE_LOOKUP(kMimg, i.op); break;
    case Encoding::EXP: n = "exp"; break;
    default: break;
    }
    if (n) return n;
    char b[40];
    std::snprintf(b, sizeof(b), "%s_op%u", encoding_name(i.enc), i.op);
    return b;
}

std::string disassemble(const Instruction& i) {
    std::string m = mnemonic(i);
    char b[160];
    switch (i.enc) {
    case Encoding::SOP2:
        std::snprintf(b, sizeof(b), "%s %s, %s, %s", m.c_str(), sreg(i.dst).c_str(),
                      src_str(i.src0, i).c_str(), src_str(i.src1, i).c_str());
        break;
    case Encoding::SOP1:
        std::snprintf(b, sizeof(b), "%s %s, %s", m.c_str(), sreg(i.dst).c_str(), src_str(i.src0, i).c_str());
        break;
    case Encoding::SOPK:
        std::snprintf(b, sizeof(b), "%s %s, %d", m.c_str(), sreg(i.dst).c_str(), i.simm);
        break;
    case Encoding::SOPC:
        std::snprintf(b, sizeof(b), "%s %s, %s", m.c_str(), src_str(i.src0, i).c_str(), src_str(i.src1, i).c_str());
        break;
    case Encoding::SOPP:
        std::snprintf(b, sizeof(b), "%s 0x%04x", m.c_str(), static_cast<u32>(i.simm) & 0xFFFF);
        break;
    case Encoding::SMRD:
        std::snprintf(b, sizeof(b), "%s s%u, s[%u:%u], %s0x%x", m.c_str(), i.dst, i.sbase, i.sbase + 1,
                      i.imm ? "" : (i.offset == 255 ? "lit:" : "s"), i.offset == 255 && !i.imm ? i.literal : i.offset);
        break;
    case Encoding::VOP1:
        std::snprintf(b, sizeof(b), "%s v%u, %s", m.c_str(), i.dst, src_str(i.src0, i).c_str());
        break;
    case Encoding::VOP2:
        std::snprintf(b, sizeof(b), "%s v%u, %s, %s", m.c_str(), i.dst, src_str(i.src0, i).c_str(),
                      src_str(i.src1, i).c_str());
        break;
    case Encoding::VOPC:
        std::snprintf(b, sizeof(b), "%s vcc, %s, %s", m.c_str(), src_str(i.src0, i).c_str(), src_str(i.src1, i).c_str());
        break;
    case Encoding::VOP3:
        std::snprintf(b, sizeof(b), "%s %s%u, %s, %s, %s%s%s", m.c_str(), i.op < 0x100 ? "s" : "v",
                      i.op < 0x100 ? i.sdst : i.dst, src_str(i.src0, i).c_str(), src_str(i.src1, i).c_str(),
                      src_str(i.src2, i).c_str(), i.neg ? " neg" : "", i.abs ? " abs" : "");
        break;
    case Encoding::VINTRP:
        std::snprintf(b, sizeof(b), "%s v%u, v%u, attr%u.%c", m.c_str(), i.dst, i.src0 - 256, i.attr, "xyzw"[i.chan]);
        break;
    case Encoding::MUBUF:
    case Encoding::MTBUF:
        std::snprintf(b, sizeof(b), "%s v%u, v%u, s[%u:%u], %s%s%s offset:%u%s", m.c_str(), i.dst, i.vaddr,
                      i.srsrc, i.srsrc + 3, src_str(i.soffset, i).c_str(), i.idxen ? " idxen" : "",
                      i.offen ? " offen" : "", i.offset, i.enc == Encoding::MTBUF ? " (tbuf)" : "");
        break;
    case Encoding::MIMG:
        std::snprintf(b, sizeof(b), "%s v%u, v%u, s[%u:%u], s[%u:%u] dmask:0x%x", m.c_str(), i.dst, i.vaddr,
                      i.srsrc, i.srsrc + 7, i.ssamp, i.ssamp + 3, i.dmask);
        break;
    case Encoding::EXP: {
        char tgt[16];
        if (i.target < 8) std::snprintf(tgt, sizeof(tgt), "mrt%u", i.target);
        else if (i.target == 8) std::snprintf(tgt, sizeof(tgt), "mrtz");
        else if (i.target == 9) std::snprintf(tgt, sizeof(tgt), "null");
        else if (i.target >= 12 && i.target < 16) std::snprintf(tgt, sizeof(tgt), "pos%u", i.target - 12);
        else if (i.target >= 32) std::snprintf(tgt, sizeof(tgt), "param%u", i.target - 32);
        else std::snprintf(tgt, sizeof(tgt), "tgt%u", i.target);
        std::snprintf(b, sizeof(b), "exp %s, v%u, v%u, v%u, v%u en:0x%x%s%s%s", tgt, i.src0, i.src1, i.src2,
                      i.dst, i.en, i.compr ? " compr" : "", i.done ? " done" : "", i.vm ? " vm" : "");
        break;
    }
    default:
        std::snprintf(b, sizeof(b), "%s (0x%08x)", m.c_str(), i.dw0);
        break;
    }
    return b;
}

u64 hash_words(const u32* data, usize count, u64 seed) {
    u64 h = seed;
    for (usize i = 0; i < count; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ull;
        h ^= h >> 29;
    }
    return h;
}

namespace {

bool decode_words(const std::vector<u32>& words, u32 pc_base, std::vector<Instruction>& out,
                  bool& has_branches, std::string& error, bool stop_at_setpc) {
    u32 pc = 0;
    while (pc < words.size()) {
        Instruction inst;
        if (!decode_instruction(words.data() + pc, words.size() - pc, pc + pc_base, inst)) {
            error = "truncated instruction stream";
            return false;
        }
        if (inst.enc == Encoding::SOPP && inst.op >= 2 && inst.op <= 9) has_branches = true;
        if (stop_at_setpc && inst.enc == Encoding::SOP1 && inst.op == 32) return true;
        out.push_back(inst);
        pc += inst.size;
    }
    return true;
}

/// Scan words starting at `addr` until the terminating instruction. Returns
/// the dword count.
bool read_until_end(const MemReader& read, u64 addr, bool fetch_shader, std::vector<u32>& words,
                    std::string& error) {
    constexpr usize kMaxDwords = 16384;
    usize max_target = 0;
    words.clear();
    usize pc = 0;
    u32 buf[3];
    while (pc < kMaxDwords) {
        // Read up to three dwords for the current instruction.
        while (words.size() < pc + 3) {
            u32 w = 0;
            if (!read(addr + words.size() * 4, &w, 4)) {
                if (words.size() > pc) break;
                error = "shader code not readable";
                return false;
            }
            words.push_back(w);
        }
        const usize avail = std::min<usize>(3, words.size() - pc);
        for (usize k = 0; k < avail; ++k) buf[k] = words[pc + k];
        Instruction inst;
        if (!decode_instruction(buf, avail, static_cast<u32>(pc), inst)) {
            error = "truncated instruction";
            return false;
        }
        const usize next = pc + inst.size;
        if (inst.enc == Encoding::SOPP && inst.op >= 2 && inst.op <= 9) {
            const i64 tgt = static_cast<i64>(next) + inst.simm;
            if (tgt > static_cast<i64>(max_target)) max_target = static_cast<usize>(tgt);
        }
        const bool is_end = fetch_shader ? (inst.enc == Encoding::SOP1 && inst.op == 32)
                                         : (inst.enc == Encoding::SOPP && inst.op == 1);
        if (is_end && next > max_target) {
            words.resize(next);
            return true;
        }
        pc = next;
    }
    error = "no s_endpgm within 64KB";
    return false;
}

} // namespace

bool decode_program(const u32* words, usize count, Program& out, std::string& error) {
    out = Program{};
    out.code.assign(words, words + count);
    if (!decode_words(out.code, 0, out.insts, out.has_branches, error, false)) return false;
    out.hash = hash_words(out.code.data(), out.code.size());
    return true;
}

bool load_program(const MemReader& read, u64 addr, const u32* user_sgprs, u32 user_sgpr_count,
                  Program& out, std::string& error) {
    out = Program{};
    if (!read_until_end(read, addr, false, out.code, error)) return false;
    std::vector<Instruction> main_insts;
    bool branches = false;
    if (!decode_words(out.code, 0, main_insts, branches, error, false)) return false;
    out.has_branches = branches;

    for (const Instruction& inst : main_insts) {
        if (inst.enc == Encoding::SOP1 && inst.op == 33) {   // s_swappc_b64
            const u32 src = inst.src0;
            if (!user_sgprs || src + 1 >= user_sgpr_count) {
                error = "s_swappc_b64 through a non-user SGPR pair";
                return false;
            }
            const u64 target = (static_cast<u64>(user_sgprs[src + 1]) << 32) | user_sgprs[src];
            if (!read_until_end(read, target, true, out.fetch_code, error)) {
                error = "fetch shader: " + error;
                return false;
            }
            std::vector<Instruction> fetch;
            bool fb = false;
            if (!decode_words(out.fetch_code, 0x100000, fetch, fb, error, true)) return false;
            if (fb) {
                error = "branches inside fetch shader";
                return false;
            }
            out.insts.insert(out.insts.end(), fetch.begin(), fetch.end());
            continue;
        }
        out.insts.push_back(inst);
    }
    out.hash = hash_words(out.code.data(), out.code.size());
    out.hash = hash_words(out.fetch_code.data(), out.fetch_code.size(), out.hash);
    return true;
}

} // namespace ps4::gcn
