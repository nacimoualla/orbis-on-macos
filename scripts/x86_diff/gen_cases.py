#!/usr/bin/env python3
"""Generate cases.h: instruction encodings checked against real x86.

Register-only forms use xmm0-7 / rax(0) rcx(1) rdx(2) rsi(6); the memory
store case addresses [rax+4*rsi+0x1ab0], which the harnesses point at a
32-byte scratch buffer. FMASK is the set of arithmetic flags the instruction
defines (undefined flags are not compared).
"""
CF, PF, AF, ZF, SF, OF = 1, 4, 0x10, 0x40, 0x80, 0x800
ALL = CF | PF | ZF | SF | OF
cases = []

def vex(op, pp, mm, d, v, r, L=0, W=0, imm=None, mod_rm=None):
    """VEX-encoded reg-reg form: ModRM.reg=d, ModRM.rm=r, VEX.vvvv=v."""
    R, B = (0 if d >= 8 else 1), (0 if r >= 8 else 1)
    vv = (~v) & 15
    if mm == 1 and W == 0 and B == 1:
        b = [0xC5, (R << 7) | (vv << 3) | (L << 2) | pp]
    else:
        b = [0xC4, (R << 7) | (1 << 6) | (B << 5) | mm, (W << 7) | (vv << 3) | (L << 2) | pp]
    b += [op, mod_rm if mod_rm is not None else 0xC0 | ((d & 7) << 3) | (r & 7)]
    if imm is not None:
        b.append(imm)
    return b

def add(name, b, fmask=0, mem=0, setup=0):
    """setup: 0 random, 1 = unsigned-div safe (rcx != 0, rdx < rcx),
    2 = signed-div safe (rdx = sign of rax, rcx != 0 and not -1)."""
    cases.append((name, b, fmask, mem, setup))

# Existing GC size-map loop
add("vpaddq x5,x4,x2", [0xc5,0xd9,0xd4,0xea]); add("vmovq rsi,x4", [0xc4,0xe1,0xf9,0x7e,0xe6])
add("vpblendw x5,x5,x3,cc", [0xc4,0xe3,0x51,0x0e,0xeb,0xcc]); add("vpsrlq x5,3", [0xc5,0xd1,0x73,0xd5,0x03])
add("vpshufd x5,x5,e8", [0xc5,0xf9,0x70,0xed,0xe8]); add("vmovq [rax+4rsi+1ab0],x5", [0xc5,0xf9,0xd6,0xac,0xb0,0xb0,0x1a,0,0], mem=1)
add("vmovd x2,ecx", [0xc5,0xf9,0x6e,0xd1]); add("vpxor x3,x3,x3", [0xc5,0xe1,0xef,0xdb]); add("vpand x7,x7,x5", [0xc5,0xc1,0xdb,0xfd])

# Packed integer add/sub (VEX.66.0F), 128 and 256 bit, incl. high regs and aliasing
for op, nm in ((0xF8,"psubb"),(0xF9,"psubw"),(0xFA,"psubd"),(0xFB,"psubq"),
               (0xFC,"paddb"),(0xFD,"paddw"),(0xFE,"paddd"),(0xD4,"paddq")):
    add(f"v{nm} x5,x4,x2", vex(op, 1, 1, 5, 4, 2))
    add(f"v{nm} y5,y4,y2", vex(op, 1, 1, 5, 4, 2, L=1))
    add(f"v{nm} x3,x7,x15", vex(op, 1, 1, 3, 7, 15))
    add(f"v{nm} x2,x2,x2", vex(op, 1, 1, 2, 2, 2))

# VEX integer ops added for Unity's math/text code
for op in (0xD1,0xD2,0xD3,0xE1,0xE2,0xF1,0xF2,0xF3,0xD5,0xE4,0xE5,0xF4,0xF5,0xD8,0xD9,0xDC,0xDD,
           0xE8,0xE9,0xEC,0xED,0xE0,0xE3,0xDA,0xDE,0xEA,0xEE,0x63,0x67):
    add(f"vex66.0f.{op:02x} x5,x4,x2", vex(op, 1, 1, 5, 4, 2))
    add(f"vex66.0f.{op:02x} y5,y4,y2", vex(op, 1, 1, 5, 4, 2, L=1))
for op in (0x08,0x09,0x0A,0x1C,0x1D,0x1E,0x2B):
    add(f"vex66.0f38.{op:02x} x5,x4,x2", vex(op, 1, 2, 5, 4 if op < 0x1C or op == 0x2B else 0, 2))
    add(f"vex66.0f38.{op:02x} y5,y4,y2", vex(op, 1, 2, 5, 4 if op < 0x1C or op == 0x2B else 0, 2, L=1))
add("vpmovmskb ecx,x2", vex(0xD7, 1, 1, 1, 0, 2)); add("vpmovmskb ecx,y2", vex(0xD7, 1, 1, 1, 0, 2, L=1))
add("vaddsubps x5,x4,x2", vex(0xD0, 3, 1, 5, 4, 2)); add("vaddsubps y5,y4,y2", vex(0xD0, 3, 1, 5, 4, 2, L=1))
add("vpalignr x5,x4,x2,5", vex(0x0F, 1, 3, 5, 4, 2, imm=5)); add("vpalignr y5,y4,y2,19", vex(0x0F, 1, 3, 5, 4, 2, L=1, imm=19))
add("vperm2f128 y5,y4,y2,31", vex(0x06, 1, 3, 5, 4, 2, L=1, imm=0x31)); add("vperm2f128 y5,y4,y2,8a", vex(0x06, 1, 3, 5, 4, 2, L=1, imm=0x8a))
add("vpblendvb x5,x4,x2,x3", vex(0x4C, 1, 3, 5, 4, 2, imm=0x30)); add("vpblendvb y5,y4,y2,x3", vex(0x4C, 1, 3, 5, 4, 2, L=1, imm=0x30))
add("vcvtps2ph x2,x5,0", vex(0x1D, 1, 3, 5, 0, 2, imm=0)); add("vcvtps2ph x2,y5,3", vex(0x1D, 1, 3, 5, 0, 2, L=1, imm=3))
add("vcvtph2ps x5,x2", vex(0x13, 1, 2, 5, 0, 2)); add("vcvtph2ps y5,x2", vex(0x13, 1, 2, 5, 0, 2, L=1))
add("vcvtsd2si ecx,x2", vex(0x2D, 3, 1, 1, 0, 2)); add("vcvtss2si rcx,x2", vex(0x2D, 2, 1, 1, 0, 2, W=1))
add("vpblendw y5,y4,y2,a5", vex(0x0E, 1, 3, 5, 4, 2, L=1, imm=0xa5))

# Horizontal add/sub: 66 = packed double, F2 = packed single (Unity's u64->double tick conversion)
for op, nm in ((0x7C, "hadd"), (0x7D, "hsub")):
    for pp, sfx in ((1, "pd"), (3, "ps")):
        add(f"v{nm}{sfx} x5,x4,x2", vex(op, pp, 1, 5, 4, 2))
        add(f"v{nm}{sfx} y5,y4,y2", vex(op, pp, 1, 5, 4, 2, L=1))

# Legacy SSE float -> int conversions (cvtsd2si was skipped as a NOP)
add("cvtsd2si eax,xmm1", [0xF2,0x0F,0x2D,0xC1]); add("cvtsd2si rax,xmm1", [0xF2,0x48,0x0F,0x2D,0xC1])
add("cvtss2si eax,xmm1", [0xF3,0x0F,0x2D,0xC1]); add("cvttsd2si eax,xmm1", [0xF2,0x0F,0x2C,0xC1])
add("cvttss2si rax,xmm1", [0xF3,0x48,0x0F,0x2C,0xC1])

# Scalar / packed float (map 0F)
for pp, sfx in ((2, "ss"), (3, "sd")):
    for op, nm in ((0x58,"add"),(0x59,"mul"),(0x5C,"sub"),(0x5D,"min"),(0x5E,"div"),(0x5F,"max"),(0x51,"sqrt"),(0x5A,"cvt")):
        add(f"v{nm}{sfx} x1,x2,x3", vex(op, pp, 1, 1, 2, 3))
for op, nm in ((0x54,"andps"),(0x55,"andnps"),(0x57,"xorps"),(0x58,"addps"),(0x59,"mulps")):
    add(f"v{nm} x1,x2,x3", vex(op, 0, 1, 1, 2, 3)); add(f"v{nm} y1,y2,y3", vex(op, 0, 1, 1, 2, 3, L=1))
add("vmovaps x1,x3", vex(0x28, 0, 1, 1, 0, 3)); add("vmovaps y1,y3", vex(0x28, 0, 1, 1, 0, 3, L=1))
add("vucomiss x1,x2", vex(0x2E, 0, 1, 1, 0, 2), ALL); add("vucomisd x1,x2", vex(0x2E, 1, 1, 1, 0, 2), ALL)
for imm in range(8):
    add(f"vcmpss x1,x2,x3,{imm}", vex(0xC2, 2, 1, 1, 2, 3, imm=imm))
add("vcvtsi2ss x1,x2,ecx", vex(0x2A, 2, 1, 1, 2, 1)); add("vcvtsi2ss x1,x2,rcx", vex(0x2A, 2, 1, 1, 2, 1, W=1))
add("vcvtsi2sd x1,x2,ecx", vex(0x2A, 3, 1, 1, 2, 1)); add("vcvtsi2sd x1,x2,rcx", vex(0x2A, 3, 1, 1, 2, 1, W=1))
add("vcvttss2si eax,x3", vex(0x2C, 2, 1, 0, 0, 3)); add("vcvttss2si rax,x3", vex(0x2C, 2, 1, 0, 0, 3, W=1))
add("vcvttsd2si eax,x3", vex(0x2C, 3, 1, 0, 0, 3)); add("vcvttsd2si rax,x3", vex(0x2C, 3, 1, 0, 0, 3, W=1))

# Integer SIMD (66 0F)
for op, nm in ((0x62,"punpckldq"),(0x6A,"punpckhdq"),(0x61,"punpcklwd"),(0x69,"punpckhwd"),(0x6C,"punpcklqdq"),
               (0x6D,"punpckhqdq"),(0xFE,"paddd"),(0xD4,"paddq"),(0xDF,"pandn"),(0x76,"pcmpeqd"),(0xDB,"pand"),(0xEF,"pxor")):
    add(f"v{nm} x1,x2,x3", vex(op, 1, 1, 1, 2, 3))
add("vmovd x1,ecx", vex(0x6E, 1, 1, 1, 0, 1)); add("vmovq x1,rcx", vex(0x6E, 1, 1, 1, 0, 1, W=1))
add("vmovd ecx,x1", vex(0x7E, 1, 1, 1, 0, 1)); add("vmovq rcx,x1", vex(0x7E, 1, 1, 1, 0, 1, W=1))
add("vmovq x1,x3 (F3 7E)", vex(0x7E, 2, 1, 1, 0, 3))
for imm in range(4):
    add(f"vpextrd ecx,x1,{imm}", vex(0x16, 1, 3, 1, 0, 1, imm=imm))
    add(f"vpinsrd x1,x2,ecx,{imm}", vex(0x22, 1, 3, 1, 2, 1, imm=imm))
for imm in range(2):
    add(f"vpextrq rcx,x1,{imm}", vex(0x16, 1, 3, 1, 0, 1, W=1, imm=imm))
    add(f"vpinsrq x1,x2,rcx,{imm}", vex(0x22, 1, 3, 1, 2, 1, W=1, imm=imm))
    add(f"vinsertf128 y1,y2,x3,{imm}", vex(0x18, 1, 3, 1, 2, 3, L=1, imm=imm))
    add(f"vextractf128 x3,y1,{imm}", vex(0x19, 1, 3, 1, 0, 3, L=1, imm=imm))
add("vblendvps x1,x2,x3,x4", vex(0x4A, 1, 3, 1, 2, 3, imm=0x40)); add("vblendvps y1,y2,y3,y4", vex(0x4A, 1, 3, 1, 2, 3, L=1, imm=0x40))
add("vblendvpd x1,x2,x3,x4", vex(0x4B, 1, 3, 1, 2, 3, imm=0x40))

# BMI1/BMI2 (0F38), dest eax/rax, src ecx/rcx, count/control edx/rdx
for W in (0, 1):
    w = "r" if W else "e"
    add(f"shlx {w}ax,{w}cx,{w}dx", vex(0xF7, 1, 2, 0, 2, 1, W=W))
    add(f"sarx {w}ax,{w}cx,{w}dx", vex(0xF7, 2, 2, 0, 2, 1, W=W))
    add(f"shrx {w}ax,{w}cx,{w}dx", vex(0xF7, 3, 2, 0, 2, 1, W=W))
    add(f"bextr {w}ax,{w}cx,{w}dx", vex(0xF7, 0, 2, 0, 2, 1, W=W), ZF | CF | OF)
    add(f"andn {w}ax,{w}dx,{w}cx", vex(0xF2, 0, 2, 0, 2, 1, W=W), ZF | SF | CF | OF)
# LZCNT / TZCNT / POPCNT / BSF / BSR, 32 and 64 bit
for pre, nm, fl in (([0xF3], "lzcnt", CF | ZF), ([0xF3], "tzcnt", CF | ZF), ([], "bsr", ZF), ([], "bsf", ZF)):
    op = 0xBD if nm in ("lzcnt", "bsr") else 0xBC
    add(f"{nm} eax,ecx", pre + [0x0F, op, 0xC1], fl); add(f"{nm} rax,rcx", pre + [0x48, 0x0F, op, 0xC1], fl)
add("popcnt eax,ecx", [0xF3, 0x0F, 0xB8, 0xC1], ALL); add("popcnt rax,rcx", [0xF3, 0x48, 0x0F, 0xB8, 0xC1], ALL)
for cc, nm in ((0x45, "ne"), (0x46, "be"), (0x4D, "ge")):
    add(f"cmov{nm} ax,cx", [0x66, 0x0F, cc, 0xC1])


# ---- Legacy SSE moves. Memory forms address [rax+4*rsi+0x1ab0] (scratch).
MEMOP = [0x84, 0xB0, 0xB0, 0x1A, 0, 0]  # modrm reg=0 (xmm0), SIB rsi*4+rax, disp32
for pre, nm in (([], "ups"), ([0x66], "upd"), ([0xF3], "ss"), ([0xF2], "sd")):
    add(f"mov{nm} x1,x3", pre + [0x0F, 0x10, 0xCB]); add(f"mov{nm} x3<-x1 (11)", pre + [0x0F, 0x11, 0xCB])
    add(f"mov{nm} x0,[mem]", pre + [0x0F, 0x10] + MEMOP, mem=1)
    add(f"mov{nm} [mem],x0", pre + [0x0F, 0x11] + MEMOP, mem=1)
for op, nm in ((0x51, "sqrt"), (0x58, "add"), (0x59, "mul"), (0x5C, "sub"), (0x5D, "min"), (0x5E, "div"), (0x5F, "max")):
    for pre, sfx in (([], "ps"), ([0x66], "pd"), ([0xF3], "ss"), ([0xF2], "sd")):
        add(f"{nm}{sfx} x1,x3", pre + [0x0F, op, 0xCB])
    add(f"{nm}sd x0,[mem]", [0xF2, 0x0F, op] + MEMOP, mem=1)
add("movaps x1,x3", [0x0F, 0x28, 0xCB]); add("movaps [mem],x0", [0x0F, 0x29] + MEMOP, mem=1)

# ---- Integer ALU (interpreter paths). rax=0 dst, rcx=1 src, rdx=2. --------
NOAF = CF | PF | ZF | SF | OF
W = {8: ([], 0), 16: ([0x66], 1), 32: ([], 1), 64: ([0x48], 1)}
def rr(width, op8, op, modrm=0xC8):  # op r/m, reg  (rm=rax, reg=rcx)
    pre, wide = W[width]
    return pre + [op if wide else op8, modrm]
for nm, base in (("add",0x00),("or",0x08),("adc",0x10),("sbb",0x18),("and",0x20),("sub",0x28),("xor",0x30),("cmp",0x38)):
    for width in (8, 16, 32, 64):
        add(f"{nm}{width} rax,rcx", rr(width, base, base + 1), NOAF)
for width in (8, 16, 32, 64):
    pre, wide = W[width]
    g = 0xF7 if wide else 0xF6
    add(f"neg{width}", pre + [g, 0xD8], NOAF); add(f"not{width}", pre + [g, 0xD0])
    add(f"inc{width}", pre + [0xFF if wide else 0xFE, 0xC0], PF | ZF | SF | OF)
    add(f"dec{width}", pre + [0xFF if wide else 0xFE, 0xC8], PF | ZF | SF | OF)
    add(f"mul{width} rcx", pre + [g, 0xE1], CF | OF); add(f"imul1op{width} rcx", pre + [g, 0xE9], CF | OF)
    add(f"div{width} rcx", pre + [g, 0xF1], 0, setup=1); add(f"idiv{width} rcx", pre + [g, 0xF9], 0, setup=2)
    for sub, nm in ((0,"rol"),(1,"ror"),(2,"rcl"),(3,"rcr"),(4,"shl"),(5,"shr"),(7,"sar")):
        rot = sub < 4
        add(f"{nm}{width} rax,1", pre + [0xD1 if wide else 0xD0, 0xC0 | (sub << 3)], (CF | OF) if rot else NOAF)
        add(f"{nm}{width} rax,cl", pre + [0xD3 if wide else 0xD2, 0xC0 | (sub << 3)], CF if rot else (CF | PF | ZF | SF))
        add(f"{nm}{width} rax,5", pre + [0xC1 if wide else 0xC0, 0xC0 | (sub << 3), 5], CF if rot else (CF | PF | ZF | SF))
for width in (16, 32, 64):
    pre, _ = W[width]
    add(f"imul{width} rax,rcx", pre + [0x0F, 0xAF, 0xC1], CF | OF)
    add(f"imul{width} rax,rcx,-7", pre + [0x6B, 0xC1, 0xF9], CF | OF)
    add(f"shld{width} rax,rcx,5", pre + [0x0F, 0xA4, 0xC8, 5], CF | PF | ZF | SF)
    add(f"shrd{width} rax,rcx,5", pre + [0x0F, 0xAC, 0xC8, 5], CF | PF | ZF | SF)
    if width != 16:  # 16-bit SHLD with CL > 16 is undefined in the SDM
        add(f"shld{width} rax,rcx,cl", pre + [0x0F, 0xA5, 0xC8], 0)
    if width != 16:  # likewise 16-bit SHRD with CL > 16
        add(f"shrd{width} rax,rcx,cl", pre + [0x0F, 0xAD, 0xC8], 0)
    for op, nm in ((0xA3,"bt"),(0xAB,"bts"),(0xB3,"btr"),(0xBB,"btc")):
        add(f"{nm}{width} rax,rcx", pre + [0x0F, op, 0xC8], CF)
    for sub, nm in ((4,"bt"),(5,"bts"),(6,"btr"),(7,"btc")):
        add(f"{nm}{width} rax,13", pre + [0x0F, 0xBA, 0xC0 | (sub << 3), 13], CF)
    add(f"xadd{width} rax,rcx", pre + [0x0F, 0xC1, 0xC8], NOAF)
    # Only ZF: Rosetta 2 reports CF/SF/PF/OF of CMPXCHG as (dest - acc), while
    # the SDM, QEMU and Bochs all give CMP acc, dest; we follow the latter.
    add(f"cmpxchg{width} rdx,rcx", pre + [0x0F, 0xB1, 0xCA], ZF)
    add(f"movsx{width} rax,cl", pre + [0x0F, 0xBE, 0xC1]); add(f"movzx{width} rax,cl", pre + [0x0F, 0xB6, 0xC1])
add("movsx32 eax,cx", [0x0F, 0xBF, 0xC1]); add("movsx64 rax,cx", [0x48, 0x0F, 0xBF, 0xC1]); add("movsxd rax,ecx", [0x48, 0x63, 0xC1])
add("cbw", [0x66, 0x98]); add("cwde", [0x98]); add("cdqe", [0x48, 0x98]); add("cwd", [0x66, 0x99]); add("cdq", [0x99]); add("cqo", [0x48, 0x99])
add("lahf", [0x9F]); add("sahf", [0x9E], CF | PF | ZF | SF | AF)
add("xchg rax,rcx", [0x48, 0x91]); add("lea rax,[rcx+rdx*4+9]", [0x48, 0x8D, 0x44, 0x91, 0x09])
add("bswap ecx", [0x0F, 0xC9]); add("bswap rcx", [0x48, 0x0F, 0xC9])
add("adc rax,0x7f", [0x48, 0x83, 0xD0, 0x7F], NOAF); add("sbb eax,-1", [0x83, 0xD8, 0xFF], NOAF)

# RSDKv5 (Sonic Mania) scene loader: 16-byte name hashes read 4 bytes at a
# time and assembled with vpinsrb [mem] / vpmovzxbd / vpslld / vpor.
for k in (0, 1, 3, 9):
    add(f"vpinsrb x0,x0,[mem],{k}", vex(0x20, 1, 3, 0, 0, 0, mod_rm=0x84) + MEMOP[1:] + [k], mem=1)
add("vpinsrb x3,x5,ecx,2", vex(0x20, 1, 3, 3, 5, 1, imm=2))
add("vpmovzxbd x3,x3", vex(0x31, 1, 2, 3, 0, 3)); add("vpmovzxbd x5,x2", vex(0x31, 1, 2, 5, 0, 2))
add("vpmovzxbd y5,x2", vex(0x31, 1, 2, 5, 0, 2, L=1))
# 0F38 20-25 sign-extend, 30-35 zero-extend (the two were swapped once).
for op, nm in ((0x20,"sxbw"),(0x21,"sxbd"),(0x22,"sxbq"),(0x23,"sxwd"),(0x24,"sxwq"),(0x25,"sxdq"),
               (0x30,"zxbw"),(0x32,"zxbq"),(0x33,"zxwd"),(0x34,"zxwq"),(0x35,"zxdq")):
    add(f"vpmov{nm} x5,x2", vex(op, 1, 2, 5, 0, 2))
for sh in (8, 16, 24):
    add(f"vpslld x3,x3,{sh}", vex(0x72, 1, 1, 6, 3, 3, imm=sh))
add("vpor x2,x3,x2", vex(0xEB, 1, 1, 2, 3, 2))

# Scalar SSE forms the JIT translates (VEX.F3/F2.0F 10/11 reg and memory,
# memory-source arithmetic and xor, COMISS / UCOMISD).
for pp, nm in ((2, "ss"), (3, "sd")):
    add(f"vmov{nm} x5,x4,x2", vex(0x10, pp, 1, 5, 4, 2))
    add(f"vmov{nm} x2,x4,x5 (11)", vex(0x11, pp, 1, 5, 4, 2))
    add(f"vmov{nm} x0,[mem]", vex(0x10, pp, 1, 0, 0, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
    add(f"vmov{nm} [mem],x0", vex(0x11, pp, 1, 0, 0, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
    add(f"vadd{nm} x0,x3,[mem]", vex(0x58, pp, 1, 0, 3, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
add("vxorps x0,x3,[mem]", vex(0x57, 0, 1, 0, 3, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
add("vcomiss x5,x2", vex(0x2F, 0, 1, 5, 0, 2), CF | PF | ZF | SF | OF)
add("vucomisd x5,x2", vex(0x2E, 1, 1, 5, 0, 2), CF | PF | ZF | SF | OF)

# Packed-integer moves and ops the JIT translates with NEON.
for op, pp, nm in ((0x6F, 1, "vmovdqa x5,x2"), (0x6F, 2, "vmovdqu x5,x2"), (0x7F, 1, "vmovdqa x2,x5 (7f)"),
                   (0x10, 0, "vmovups x5,x2"), (0x28, 0, "vmovaps x5,x2"), (0x29, 1, "vmovapd x2,x5 (29)"),
                   (0x70, 1, None)):
    if nm: add(nm, vex(op, pp, 1, 5, 0, 2))
add("vmovdqu y5,y2", vex(0x6F, 2, 1, 5, 0, 2, L=1)); add("vmovups x2,x5 (11)", vex(0x11, 0, 1, 5, 0, 2))
add("vmovdqu x0,[mem]", vex(0x6F, 2, 1, 0, 0, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
add("vmovdqu [mem],x0", vex(0x7F, 2, 1, 0, 0, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
add("vmovups [mem],x0", vex(0x11, 0, 1, 0, 0, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
add("vmovd x5,ecx", vex(0x6E, 1, 1, 5, 0, 1)); add("vmovq x5,rcx", vex(0x6E, 1, 1, 5, 0, 1, W=1))
add("vmovd ecx,x5", vex(0x7E, 1, 1, 5, 0, 1)); add("vmovq rcx,x5", vex(0x7E, 1, 1, 5, 0, 1, W=1))
add("vmovq x5,x2 (f3 7e)", vex(0x7E, 2, 1, 5, 0, 2)); add("vmovq x2,x5 (d6)", vex(0xD6, 1, 1, 5, 0, 2))
add("vmovq [mem],x0", vex(0xD6, 1, 1, 0, 0, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
for imm in (0x1B, 0xE4, 0x4E, 0x00):
    add(f"vpshufd x5,x2,{imm:#x}", vex(0x70, 1, 1, 5, 0, 2, imm=imm))
for op, nm in ((0x71, "w"), (0x72, "d"), (0x73, "q")):
    for sub, sn in ((2, "srl"), (4, "sra"), (6, "sll")):
        if op == 0x73 and sub == 4: continue
        for sh in (0, 1, 7, 15, 31, 33, 63, 70):
            add(f"vp{sn}{nm} x5,x2,{sh}", vex(op, 1, 1, sub, 5, 2, imm=sh))
for op in (0xFC, 0xFD, 0xFE, 0xD4, 0xF8, 0xF9, 0xFA, 0xFB, 0xDB, 0xDF, 0xEB, 0xD5,
           0x60, 0x61, 0x62, 0x6C, 0x68, 0x69, 0x6A, 0x6D):
    add(f"vex66.0f.{op:02x}b x5,x4,x2", vex(op, 1, 1, 5, 4, 2))
add("vpaddd x0,x3,[mem]", vex(0xFE, 1, 1, 0, 3, 0, mod_rm=0x84) + MEMOP[1:], mem=1)
for op in (0x54, 0x55, 0x56):
    add(f"vex.0f.{op:02x} x5,x4,x2", vex(op, 0, 1, 5, 4, 2))
add("vpshufb x5,x4,x2", vex(0x00, 1, 2, 5, 4, 2)); add("vpmulld x5,x4,x2", vex(0x40, 1, 2, 5, 4, 2))

# Ops the opening movie's decoder / colour conversion uses.
M = MEMOP[1:]
add("vpackuswb x5,x4,x2", vex(0x67, 1, 1, 5, 4, 2)); add("vpackuswb y5,y4,y2", vex(0x67, 1, 1, 5, 4, 2, L=1))
add("vpacksswb x5,x4,x2", vex(0x63, 1, 1, 5, 4, 2)); add("vpackssdw x5,x4,x2", vex(0x6B, 1, 1, 5, 4, 2))
add("vpackusdw x5,x4,x2", vex(0x2B, 1, 2, 5, 4, 2))
add("vpmovzxbd x0,[mem]", vex(0x31, 1, 2, 0, 0, 0, mod_rm=0x84) + M, mem=1)
add("vpmovzxbd y0,[mem]", vex(0x31, 1, 2, 0, 0, 0, L=1, mod_rm=0x84) + M, mem=1)
add("vpmovsxwd x0,[mem]", vex(0x23, 1, 2, 0, 0, 0, mod_rm=0x84) + M, mem=1)
add("vpmovsxwd y0,[mem]", vex(0x23, 1, 2, 0, 0, 0, L=1, mod_rm=0x84) + M, mem=1)
add("vpmovzxbd y5,x2", vex(0x31, 1, 2, 5, 0, 2, L=1)); add("vpmovsxwd y5,x2", vex(0x23, 1, 2, 5, 0, 2, L=1))
for k in (0, 1):
    add(f"vinsertf128 y5,y4,x2,{k}", vex(0x18, 1, 3, 5, 4, 2, L=1, imm=k))
    add(f"vinsertf128 y0,y3,[mem],{k}", vex(0x18, 1, 3, 0, 3, 0, L=1, mod_rm=0x84) + M + [k], mem=1)
    add(f"vextractf128 x2,y5,{k}", vex(0x19, 1, 3, 5, 0, 2, L=1, imm=k))
    add(f"vextractf128 [mem],y0,{k}", vex(0x19, 1, 3, 0, 0, 0, L=1, mod_rm=0x84) + M + [k], mem=1)
add("vbroadcastsd y0,[mem]", vex(0x19, 1, 2, 0, 0, 0, L=1, mod_rm=0x84) + M, mem=1)
add("vbroadcastss x0,[mem]", vex(0x18, 1, 2, 0, 0, 0, mod_rm=0x84) + M, mem=1)
add("vmovlpd [mem],x0", vex(0x13, 1, 1, 0, 0, 0, mod_rm=0x84) + M, mem=1)
add("vblendvpd x5,x4,x2,x3", vex(0x4B, 1, 3, 5, 4, 2, imm=0x30)); add("vblendvpd y5,y4,y2,y3", vex(0x4B, 1, 3, 5, 4, 2, L=1, imm=0x30))
add("vblendvps y5,y4,y2,y3", vex(0x4A, 1, 3, 5, 4, 2, L=1, imm=0x30))
add("vpcmpeqd x5,x4,x2", vex(0x76, 1, 1, 5, 4, 2)); add("vpcmpeqd x5,x4,x4", vex(0x76, 1, 1, 5, 4, 4))
add("vpinsrd x0,x3,[mem],2", vex(0x22, 1, 3, 0, 3, 0, mod_rm=0x84) + M + [2], mem=1)
add("vorps x0,x3,[mem]", vex(0x56, 0, 1, 0, 3, 0, mod_rm=0x84) + M, mem=1)
add("vorps y5,y4,y2", vex(0x56, 0, 1, 5, 4, 2, L=1)); add("vandps y5,y4,y2", vex(0x54, 0, 1, 5, 4, 2, L=1))
for pp, nm in ((0, "ps"), (1, "pd"), (2, "ss"), (3, "sd")):
    for imm in (0, 1, 4, 5, 6, 9, 13, 14, 0x1D):
        add(f"vcmp{nm} x5,x4,x2,{imm:#x}", vex(0xC2, pp, 1, 5, 4, 2, imm=imm))
    add(f"vcmp{nm} x5,x4,x4,0", vex(0xC2, pp, 1, 5, 4, 4, imm=0))
add("adc32 ecx,-1", [0x83, 0xD1, 0xFF], ALL); add("adc32 ecx,5", [0x83, 0xD1, 0x05], ALL)
add("vpaddusb x5,x4,x2", vex(0xDC, 1, 1, 5, 4, 2)); add("vpsubusb x5,x4,x2", vex(0xD8, 1, 1, 5, 4, 2))
add("vpmulhw x5,x4,x2", vex(0xE5, 1, 1, 5, 4, 2)); add("vpmaddwd x5,x4,x2", vex(0xF5, 1, 1, 5, 4, 2))
add("vpavgb x5,x4,x2", vex(0xE0, 1, 1, 5, 4, 2)); add("vpsraw x5,x2,3", vex(0x71, 1, 1, 4, 5, 2, imm=3))

# The rest of what the movie path interprets.
add("vmovsldup x5,x2", vex(0x12, 2, 1, 5, 0, 2)); add("vmovshdup x5,x2", vex(0x16, 2, 1, 5, 0, 2))
add("vmovsldup y5,y2", vex(0x12, 2, 1, 5, 0, 2, L=1)); add("vmovddup x5,x2", vex(0x12, 3, 1, 5, 0, 2))
add("vmovhlps x5,x4,x2", vex(0x12, 0, 1, 5, 4, 2)); add("vmovlhps x5,x4,x2", vex(0x16, 0, 1, 5, 4, 2))
for imm in (0x1B, 0xE4, 0x00):
    add(f"vpshuflw x5,x2,{imm:#x}", vex(0x70, 3, 1, 5, 0, 2, imm=imm)); add(f"vpshufhw x5,x2,{imm:#x}", vex(0x70, 2, 1, 5, 0, 2, imm=imm))
for op in (0x64, 0x65, 0x66, 0x74, 0x75, 0xF4, 0xD1, 0xD2, 0xD3, 0xE1, 0xE2, 0xF1, 0xF2, 0xF3, 0xE0, 0xE3, 0xE4, 0xE5, 0xE8, 0xE9, 0xEC, 0xED, 0xDA, 0xDE, 0xEA, 0xEE, 0xD8, 0xD9, 0xDC, 0xDD, 0xF6):
    add(f"vex66.0f.{op:02x}c x5,x4,x2", vex(op, 1, 1, 5, 4, 2))
for op in (0x02, 0x01, 0x03, 0x05, 0x06, 0x3B, 0x3A, 0x39, 0x38, 0x3F, 0x3E, 0x3D, 0x3C, 0x2B, 0x0B, 0x04, 0x1C, 0x1D, 0x1E, 0x29, 0x37):
    add(f"vex66.0f38.{op:02x}c x5,x4,x2", vex(op, 1, 2, 5, 0 if 0x1C <= op <= 0x1E else 4, 2))
for op in (0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25):
    add(f"vpmov.{op:02x} x5,x2", vex(op, 1, 2, 5, 0, 2))
add("vpmovzxbw x0,[mem]", vex(0x30, 1, 2, 0, 0, 0, mod_rm=0x84) + M, mem=1)
for imm in (0x00, 0x0F, 0xAA, 0x5C):
    add(f"vpblendw x5,x4,x2,{imm:#x}", vex(0x0E, 1, 3, 5, 4, 2, imm=imm))
for imm in (0x00, 0x1D, 0x50, 0xC8, 0x3F):
    add(f"vinsertps x5,x4,x2,{imm:#x}", vex(0x21, 1, 3, 5, 4, 2, imm=imm))
for imm in (0, 1, 2, 3, 4, 9):
    add(f"vroundss x5,x4,x2,{imm}", vex(0x0A, 1, 3, 5, 4, 2, imm=imm)); add(f"vroundsd x5,x4,x2,{imm}", vex(0x0B, 1, 3, 5, 4, 2, imm=imm))
    add(f"vroundps x5,x2,{imm}", vex(0x08, 1, 3, 5, 0, 2, imm=imm))
for sh in (0, 1, 4, 15, 16, 20):
    add(f"vpslldq x5,x2,{sh}", vex(0x73, 1, 1, 7, 5, 2, imm=sh)); add(f"vpsrldq x5,x2,{sh}", vex(0x73, 1, 1, 3, 5, 2, imm=sh))
add("vcvtdq2ps x5,x2", vex(0x5B, 0, 1, 5, 0, 2)); add("vcvttps2dq x5,x2", vex(0x5B, 2, 1, 5, 0, 2))
add("vcvtps2dq x5,x2", vex(0x5B, 1, 1, 5, 0, 2)); add("vcvtdq2ps y5,y2", vex(0x5B, 0, 1, 5, 0, 2, L=1))
add("vcvttss2si ecx,x2", vex(0x2C, 2, 1, 1, 0, 2)); add("vcvttsd2si ecx,x2", vex(0x2C, 3, 1, 1, 0, 2))
add("vcvtss2sd x5,x4,x2", vex(0x5A, 2, 1, 5, 4, 2)); add("vcvtsd2ss x5,x4,x2", vex(0x5A, 3, 1, 5, 4, 2))
add("vsqrtss x5,x4,x2", vex(0x51, 2, 1, 5, 4, 2))
for op, nm in ((0x58, "add"), (0x59, "mul"), (0x5C, "sub"), (0x54, "and"), (0x55, "andn")):
    add(f"v{nm}ps x5,x4,x2 (b)", vex(op, 0, 1, 5, 4, 2)); add(f"v{nm}ps y5,y4,y2 (b)", vex(op, 0, 1, 5, 4, 2, L=1))
add("vpextrd ecx,x5,2 (b)", vex(0x16, 1, 3, 5, 0, 1, imm=2)); add("vpinsrd x5,x4,ecx,3 (b)", vex(0x22, 1, 3, 5, 4, 1, imm=3))
add("vpinsrb x0,x3,[mem],5", vex(0x20, 1, 3, 0, 3, 0, mod_rm=0x84) + M + [5], mem=1)
add("lzcnt32 ecx,edx", [0xF3, 0x0F, 0xBD, 0xCA], CF | ZF); add("lzcnt64 rcx,rdx", [0xF3, 0x48, 0x0F, 0xBD, 0xCA], CF | ZF)
add("cmove16 cx,dx", [0x66, 0x0F, 0x44, 0xCA]); add("cmovs16 cx,dx", [0x66, 0x0F, 0x48, 0xCA])
add("rol32 ecx,cl (b)", [0xD3, 0xC2], CF); add("bt32 edx,ecx", [0x0F, 0xA3, 0xCA], CF)
add("sbb8 dl,cl", [0x18, 0xCA], NOAF); add("mul8 cl", [0xF6, 0xE1], CF | OF); add("div8 cl", [0xF6, 0xF1], 0, setup=1)

# Second batch of JIT translations (movie path): 256-bit widening loads,
# lane insert/extract, blends, packs, ADC/SBB.
for op in (0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35):
    add(f"vpmov.{op:02x} y5,x2", vex(op, 1, 2, 5, 0, 2, L=1))
    add(f"vpmov.{op:02x} y0,[mem]", vex(op, 1, 2, 0, 0, 0, L=1, mod_rm=0x84) + M, mem=1)
    add(f"vpmov.{op:02x} x0,[mem]", vex(op, 1, 2, 0, 0, 0, mod_rm=0x84) + M, mem=1)
add("vinsertf128 y5,y5,x2,1", vex(0x18, 1, 3, 5, 5, 2, L=1, imm=1)); add("vinsertf128 y2,y5,x2,0", vex(0x18, 1, 3, 2, 5, 2, L=1, imm=0))
add("vblendvps x5,x4,[mem],x3", vex(0x4A, 1, 3, 5, 4, 0, mod_rm=0x84 | (5 << 3) & 0x38) + M + [0x30], mem=1)
for op, nm in ((0x54, "and"), (0x55, "andn"), (0x56, "or")):
    add(f"v{nm}ps y0,y3,[mem]", vex(op, 0, 1, 0, 3, 0, L=1, mod_rm=0x84) + M, mem=1)
add("vpackusdw x0,x3,[mem]", vex(0x2B, 1, 2, 0, 3, 0, mod_rm=0x84) + M, mem=1)
add("vpshuflw x0,[mem],0x93", vex(0x70, 3, 1, 0, 0, 0, mod_rm=0x84) + M + [0x93], mem=1)
for w, pre in ((32, []), (64, [0x48])):
    add(f"adc{w} rcx,rdx", pre + [0x11, 0xD1], NOAF); add(f"sbb{w} rcx,rdx", pre + [0x19, 0xD1], NOAF)
    add(f"adc{w} rdx,rcx (13)", pre + [0x13, 0xD1], NOAF); add(f"sbb{w} rdx,rcx (1b)", pre + [0x1B, 0xD1], NOAF)
    add(f"adc{w} rcx,-1", pre + [0x83, 0xD1, 0xFF], NOAF); add(f"sbb{w} rcx,7", pre + [0x83, 0xD9, 0x07], NOAF)
    add(f"adc{w} rcx,0x12345", pre + [0x81, 0xD1, 0x45, 0x23, 0x01, 0x00], NOAF)
    add(f"adc{w} [mem],ecx", pre + [0x11, 0x8C, 0xB0, 0xB0, 0x1A, 0, 0], NOAF, mem=1)

# VMASKMOVPS/PD masked loads (0F38 2C/2D) and stores (2E/2F) -- Press Garden.
for op, nm in ((0x2C, "vmaskmovps x5,x4,[mem]"), (0x2D, "vmaskmovpd x5,x4,[mem]")):
    add(nm, vex(op, 1, 2, 5, 4, 0, mod_rm=0x84 | (5 << 3) & 0x38) + M, mem=1)
for op, nm in ((0x2E, "vmaskmovps [mem],x4,x5"), (0x2F, "vmaskmovpd [mem],x4,x5")):
    add(nm, vex(op, 1, 2, 5, 4, 0, mod_rm=0x84 | (5 << 3) & 0x38) + M, mem=1)
add("vmaskmovps y5,y4,[mem]", vex(0x2C, 1, 2, 5, 4, 0, L=1, mod_rm=0x84 | (5 << 3) & 0x38) + M, mem=1)
add("vmaskmovps [mem],y4,y5", vex(0x2E, 1, 2, 5, 4, 0, L=1, mod_rm=0x84 | (5 << 3) & 0x38) + M, mem=1)
add("vpcmpgtd x5,x4,x2", vex(0x66, 1, 1, 5, 4, 2))

with open("cases.h", "w") as f:
    f.write("// Generated by gen_cases.py -- shared by native.c and emu.cpp.\n#include <stdint.h>\n")
    f.write("struct Case { const char* name; uint8_t len; uint8_t bytes[15]; uint32_t fmask; int mem; int setup; };\n")
    f.write("static const struct Case kCases[] = {\n")
    for nm, b, fm, mem, setup in cases:
        f.write('  {"%s", %d, {%s}, 0x%x, %d, %d},\n' % (nm, len(b), ",".join("0x%02x" % x for x in b), fm, mem, setup))
    f.write("};\n#define NCASES (sizeof(kCases)/sizeof(kCases[0]))\n")
    f.write(open("state.h.in").read())
print(len(cases), "cases")
