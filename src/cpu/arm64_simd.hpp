#pragma once

// ARM64 FP/NEON (and a few integer) instruction encodings used by the JIT.
// Each function returns one instruction word; register arguments are
// V-register numbers (vd/vn/vm), X/W-register numbers (xd/xn/...), byte
// offsets for loads/stores and element indices for lane moves. Encodings are
// from the Arm ARM; every form used here was checked against Apple's
// assembler (clang -arch arm64).

#include "ps4/common.hpp"

namespace ps4::a64 {

constexpr u32 rd_rn(u32 base, u32 d, u32 n) { return base | (n << 5) | d; }
constexpr u32 rd_rn_rm(u32 base, u32 d, u32 n, u32 m) { return base | (m << 16) | (n << 5) | d; }

// Loads/stores, unsigned scaled immediate offset: [xn, #off].
constexpr u32 ldr_q(u32 vt, u32 xn, u32 off) { return rd_rn(0x3DC00000 | ((off / 16) << 10), vt, xn); }
constexpr u32 str_q(u32 vt, u32 xn, u32 off) { return rd_rn(0x3D800000 | ((off / 16) << 10), vt, xn); }
constexpr u32 ldr_d(u32 vt, u32 xn, u32 off) { return rd_rn(0xFD400000 | ((off / 8) << 10), vt, xn); }
constexpr u32 str_d(u32 vt, u32 xn, u32 off) { return rd_rn(0xFD000000 | ((off / 8) << 10), vt, xn); }
constexpr u32 ldr_s(u32 vt, u32 xn, u32 off) { return rd_rn(0xBD400000 | ((off / 4) << 10), vt, xn); }
constexpr u32 str_s(u32 vt, u32 xn, u32 off) { return rd_rn(0xBD000000 | ((off / 4) << 10), vt, xn); }
// Register offset: [xn, xm].
constexpr u32 ldr_q_x(u32 vt, u32 xn, u32 xm) { return rd_rn_rm(0x3CE06800, vt, xn, xm); }
constexpr u32 str_q_x(u32 vt, u32 xn, u32 xm) { return rd_rn_rm(0x3CA06800, vt, xn, xm); }
constexpr u32 ldr_d_x(u32 vt, u32 xn, u32 xm) { return rd_rn_rm(0xFC606800, vt, xn, xm); }
constexpr u32 str_d_x(u32 vt, u32 xn, u32 xm) { return rd_rn_rm(0xFC206800, vt, xn, xm); }
constexpr u32 ldr_s_x(u32 vt, u32 xn, u32 xm) { return rd_rn_rm(0xBC606800, vt, xn, xm); }
constexpr u32 str_s_x(u32 vt, u32 xn, u32 xm) { return rd_rn_rm(0xBC206800, vt, xn, xm); }

// Vector floating point, 4S (dbl = false) or 2D (dbl = true).
constexpr u32 fsz(bool dbl) { return dbl ? (1u << 22) : 0u; }
constexpr u32 fadd_v(bool dbl, u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E20D400 | fsz(dbl), d, n, m); }
constexpr u32 fsub_v(bool dbl, u32 d, u32 n, u32 m) { return rd_rn_rm(0x4EA0D400 | fsz(dbl), d, n, m); }
constexpr u32 fmul_v(bool dbl, u32 d, u32 n, u32 m) { return rd_rn_rm(0x6E20DC00 | fsz(dbl), d, n, m); }
constexpr u32 fdiv_v(bool dbl, u32 d, u32 n, u32 m) { return rd_rn_rm(0x6E20FC00 | fsz(dbl), d, n, m); }
constexpr u32 fcmeq_v(bool dbl, u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E20E400 | fsz(dbl), d, n, m); }
constexpr u32 fcmge_v(bool dbl, u32 d, u32 n, u32 m) { return rd_rn_rm(0x6E20E400 | fsz(dbl), d, n, m); }
constexpr u32 fcmgt_v(bool dbl, u32 d, u32 n, u32 m) { return rd_rn_rm(0x6EA0E400 | fsz(dbl), d, n, m); }
constexpr u32 fsqrt_v(bool dbl, u32 d, u32 n) { return rd_rn(0x6EA1F800 | fsz(dbl), d, n); }

// Scalar floating point: S (dbl = false) or D registers.
constexpr u32 fsq(bool dbl) { return dbl ? (1u << 22) : 0u; }
constexpr u32 fsqrt_s(bool dbl, u32 d, u32 n) { return rd_rn(0x1E21C000 | fsq(dbl), d, n); }
constexpr u32 fcmp_s(bool dbl, u32 n, u32 m) { return rd_rn_rm(0x1E202000 | fsq(dbl), 0, n, m); }
constexpr u32 fcvt_d_s(u32 d, u32 n) { return rd_rn(0x1E22C000, d, n); }   // double <- single
constexpr u32 fcvt_s_d(u32 d, u32 n) { return rd_rn(0x1E624000, d, n); }   // single <- double
// Integer <-> FP; `x` selects the 64-bit general register.
constexpr u32 scvtf(bool dbl, bool x, u32 vd, u32 rn) {
    return rd_rn(0x1E220000 | fsq(dbl) | (x ? 0x80000000u : 0u), vd, rn);
}
constexpr u32 fcvtzs(bool dbl, bool x, u32 rd, u32 vn) {
    return rd_rn(0x1E380000 | fsq(dbl) | (x ? 0x80000000u : 0u), rd, vn);
}
constexpr u32 fmov_s_w(u32 vd, u32 wn) { return rd_rn(0x1E270000, vd, wn); }   // zeroes the rest of vd
constexpr u32 fmov_d_x(u32 vd, u32 xn) { return rd_rn(0x9E670000, vd, xn); }

// Bitwise, 16B.
constexpr u32 and_v(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E201C00, d, n, m); }
constexpr u32 bic_v(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E601C00, d, n, m); }   // n & ~m
constexpr u32 orr_v(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4EA01C00, d, n, m); }
constexpr u32 eor_v(u32 d, u32 n, u32 m) { return rd_rn_rm(0x6E201C00, d, n, m); }
constexpr u32 bsl_v(u32 d, u32 n, u32 m) { return rd_rn_rm(0x6E601C00, d, n, m); }   // d ? n : m (bitwise)
constexpr u32 not_v(u32 d, u32 n) { return rd_rn(0x6E205800, d, n); }
constexpr u32 mov_v(u32 d, u32 n) { return orr_v(d, n, n); }
constexpr u32 movi_zero(u32 d) { return 0x6F00E400 | d; }   // movi vd.2d, #0

// Integer vector: 4S (q = false) or 2D (q = true).
constexpr u32 isz(bool q) { return q ? (3u << 22) : (2u << 22); }
constexpr u32 add_v(bool q, u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E208400 | isz(q), d, n, m); }
constexpr u32 sub_v(bool q, u32 d, u32 n, u32 m) { return rd_rn_rm(0x6E208400 | isz(q), d, n, m); }
constexpr u32 cmeq_v(u32 d, u32 n, u32 m) { return rd_rn_rm(0x6E208C00 | isz(false), d, n, m); }   // 4S
constexpr u32 sshr_sign(bool q, u32 d, u32 n) {   // sshr vd, vn, #31 (4S) / #63 (2D): lane sign masks
    return rd_rn(q ? 0x4F410400 : 0x4F210400, d, n);
}
constexpr u32 shl_sign(bool q, u32 d, u32 n) {    // shl vd, vn, #31 (4S) / #63 (2D): mask -> sign bit
    return rd_rn(q ? 0x4F7F5400 : 0x4F3F5400, d, n);
}

// Integer vector, any lane size: sz = 0 (16B), 1 (8H), 2 (4S), 3 (2D). The
// JIT stores these bases per opcode and adds the registers when it emits.
constexpr u32 lanes(u32 base, u32 sz) { return base | (sz << 22); }
inline constexpr u32 ADD_V = 0x4E208400, SUB_V = 0x6E208400, MUL_V = 0x4E209C00;
inline constexpr u32 SQADD_V = 0x4E200C00, UQADD_V = 0x6E200C00;
inline constexpr u32 SQSUB_V = 0x4E202C00, UQSUB_V = 0x6E202C00;
inline constexpr u32 CMEQ_V = 0x6E208C00, CMGT_V = 0x4E203400;
inline constexpr u32 SMAX_V = 0x4E206400, UMAX_V = 0x6E206400;
inline constexpr u32 SMIN_V = 0x4E206C00, UMIN_V = 0x6E206C00;
inline constexpr u32 URHADD_V = 0x6E201400;
inline constexpr u32 ZIP1_V = 0x4E003800, ZIP2_V = 0x4E007800;
// Saturating narrow into the low half of vd (OR in NARROW_HIGH for the high half).
inline constexpr u32 SQXTN_V = 0x0E214800, SQXTUN_V = 0x2E212800, NARROW_HIGH = 0x40000000;
constexpr u32 smull_h(u32 d, u32 n, u32 m) { return rd_rn_rm(0x0E60C000, d, n, m); }    // 4S <- 4H * 4H
constexpr u32 smull2_h(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E60C000, d, n, m); }   // 4S <- high 4H
constexpr u32 umull_h(u32 d, u32 n, u32 m) { return rd_rn_rm(0x2E60C000, d, n, m); }
constexpr u32 umull2_h(u32 d, u32 n, u32 m) { return rd_rn_rm(0x6E60C000, d, n, m); }
constexpr u32 umull_s(u32 d, u32 n, u32 m) { return rd_rn_rm(0x2EA0C000, d, n, m); }    // 2D <- 2S * 2S
constexpr u32 xtn_s(u32 d, u32 n) { return rd_rn(0x0EA12800, d, n); }                    // 2S <- 2D
constexpr u32 uzp2_h(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E405800, d, n, m); }
constexpr u32 addp_s(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4EA0BC00, d, n, m); }
constexpr u32 tbl_16b(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E000000, d, n, m); }
constexpr u32 ext_16b(u32 d, u32 n, u32 m, u32 i) { return rd_rn_rm(0x6E000000 | (i << 11), d, n, m); }
constexpr u32 movi_16b(u32 d, u32 imm) {   // all bytes = imm
    return 0x4F00E400 | ((imm >> 5) << 16) | ((imm & 31) << 5) | d;
}
// Shift by immediate; esize = 16, 32 or 64. shl: sh < esize; shr: 1 <= sh <= esize.
constexpr u32 shl_imm(u32 esize, u32 d, u32 n, u32 sh) { return rd_rn(0x4F005400 | ((esize + sh) << 16), d, n); }
constexpr u32 sshr_imm(u32 esize, u32 d, u32 n, u32 sh) { return rd_rn(0x4F000400 | ((2 * esize - sh) << 16), d, n); }
constexpr u32 ushr_imm(u32 esize, u32 d, u32 n, u32 sh) { return rd_rn(0x6F000400 | ((2 * esize - sh) << 16), d, n); }
// Widen the low half of vn; esize = source lane bits (8, 16, 32).
constexpr u32 sxtl(u32 esize, u32 d, u32 n) { return rd_rn(0x0F00A400 | (esize << 16), d, n); }
constexpr u32 uxtl(u32 esize, u32 d, u32 n) { return rd_rn(0x2F00A400 | (esize << 16), d, n); }
constexpr u32 ins_h(u32 d, u32 i, u32 n, u32 j) {   // vd.h[i] = vn.h[j]
    return 0x6E000400 | (((i << 2) | 2) << 16) | ((j << 1) << 11) | (n << 5) | d;
}
constexpr u32 ldr_h_x(u32 vt, u32 xn, u32 xm) { return rd_rn_rm(0x7C606800, vt, xn, xm); }

// Permutes: 4S (q = false) or 2D (q = true).
constexpr u32 zip1(bool q, u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E003800 | isz(q), d, n, m); }
constexpr u32 zip2(bool q, u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E007800 | isz(q), d, n, m); }
constexpr u32 trn1(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E802800, d, n, m); }   // 4S
constexpr u32 trn2(u32 d, u32 n, u32 m) { return rd_rn_rm(0x4E806800, d, n, m); }   // 4S

// Lane moves. `dbl` selects 64-bit lanes (D) over 32-bit (S).
constexpr u32 imm5(bool dbl, u32 i) { return dbl ? ((i << 4) | 8) : ((i << 3) | 4); }
constexpr u32 ins_elem(bool dbl, u32 d, u32 i, u32 n, u32 j) {   // vd.[i] = vn.[j]
    return 0x6E000400 | (imm5(dbl, i) << 16) | ((dbl ? j << 3 : j << 2) << 11) | (n << 5) | d;
}
constexpr u32 ins_gpr(bool dbl, u32 d, u32 i, u32 rn) {           // vd.[i] = wn / xn
    return 0x4E001C00 | (imm5(dbl, i) << 16) | (rn << 5) | d;
}
constexpr u32 dup_elem(bool dbl, u32 d, u32 n, u32 j) {           // all lanes = vn.[j]
    return 0x4E000400 | (imm5(dbl, j) << 16) | (n << 5) | d;
}

// General purpose: csel xd/wd, xn, xm, cond.
constexpr u32 csel(bool x, u32 d, u32 n, u32 m, u32 cond) {
    return (x ? 0x9A800000u : 0x1A800000u) | (m << 16) | (cond << 12) | (n << 5) | d;
}
// Variable shifts/rotates (amount taken modulo the register size).
constexpr u32 gsz(bool x) { return x ? 0x80000000u : 0u; }
constexpr u32 lslv(bool x, u32 d, u32 n, u32 m) { return rd_rn_rm(0x1AC02000 | gsz(x), d, n, m); }
constexpr u32 asrv(bool x, u32 d, u32 n, u32 m) { return rd_rn_rm(0x1AC02800 | gsz(x), d, n, m); }
constexpr u32 rorv(bool x, u32 d, u32 n, u32 m) { return rd_rn_rm(0x1AC02C00 | gsz(x), d, n, m); }
constexpr u32 mul(u32 d, u32 n, u32 m) { return rd_rn_rm(0x9B007C00, d, n, m); }     // 64-bit
constexpr u32 smulh(u32 d, u32 n, u32 m) { return rd_rn_rm(0x9B407C00, d, n, m); }

} // namespace ps4::a64
