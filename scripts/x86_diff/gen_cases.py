#!/usr/bin/env python3
"""Generate cases.h: instruction encodings checked against real x86.

Register-only forms use xmm0-7 / rax(0) rcx(1) rdx(2) rsi(6); the memory
store case addresses [rax+4*rsi+0x1ab0], which the harnesses point at a
16-byte scratch buffer. FMASK is the set of arithmetic flags the instruction
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
    2 = signed-div safe (rdx = sign of rax, rcx != 0 and not -1),
    3 = cl takes the counts 0, 1, 8, 9, 16, 17, 31, 63 (one per trial)."""
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

# ---- JIT SSE/AVX coverage: memory operands, 3-operand moves, shuffles and
# blends across immediates, 256-bit and high-register forms, legacy encodings.
def vexm(op, pp, mm, d, v, L=0, W=0, imm=None):
    """VEX form with a memory operand at [rax+4*rsi+0x1ab0] (16-byte scratch)."""
    b = vex(op, pp, mm, d, v, 0, L=L, W=W, mod_rm=0x84 | ((d & 7) << 3))
    return b + [0xB0, 0xB0, 0x1A, 0, 0] + ([imm] if imm is not None else [])
def sse(pre, op, d, r, imm=None, rex=0):
    """Legacy SSE reg-reg form; op is the byte(s) after 0F."""
    ops = op if isinstance(op, list) else [op]
    rx = rex | (4 if d >= 8 else 0) | (1 if r >= 8 else 0)
    return pre + ([0x40 | rx] if rx else []) + [0x0F] + ops + [0xC0 | ((d & 7) << 3) | (r & 7)] + ([imm] if imm is not None else [])
def ssem(pre, op, d, imm=None, rex=0):
    ops = op if isinstance(op, list) else [op]
    return pre + ([0x40 | rex] if rex else []) + [0x0F] + ops + [0x84 | ((d & 7) << 3), 0xB0, 0xB0, 0x1A, 0, 0] + ([imm] if imm is not None else [])

# moves
for pp, nm in ((2, "ss"), (3, "sd")):
    add(f"vmov{nm} x1,x2,x3", vex(0x10, pp, 1, 1, 2, 3)); add(f"vmov{nm} x3,x2,x1 (11)", vex(0x11, pp, 1, 1, 2, 3))
    add(f"vmov{nm} x1,[mem]", vexm(0x10, pp, 1, 1, 0), mem=1); add(f"vmov{nm} [mem],x1", vexm(0x11, pp, 1, 1, 0), mem=1)
add("vmovups x1,[mem]", vexm(0x10, 0, 1, 1, 0), mem=1); add("vmovups [mem],x1", vexm(0x11, 0, 1, 1, 0), mem=1)
add("vmovaps [mem],x1", vexm(0x29, 0, 1, 1, 0), mem=1); add("vmovapd x1,[mem]", vexm(0x28, 1, 1, 1, 0), mem=1)
add("vmovdqu x1,[mem]", vexm(0x6F, 2, 1, 1, 0), mem=1); add("vmovdqu [mem],x1", vexm(0x7F, 2, 1, 1, 0), mem=1)
add("vmovdqa x1,x3", vex(0x6F, 1, 1, 1, 0, 3)); add("vmovdqa y1,y3", vex(0x6F, 1, 1, 1, 0, 3, L=1))
add("vmovdqu y3<-y1 (7f)", vex(0x7F, 2, 1, 1, 0, 3, L=1)); add("vmovups y1,y3", vex(0x10, 0, 1, 1, 0, 3, L=1))
add("vmovups y3<-y1 (11)", vex(0x11, 0, 1, 1, 0, 3, L=1)); add("vmovaps x9,x14", vex(0x28, 0, 1, 9, 0, 14))
add("vmovq x1,[mem]", vexm(0x7E, 2, 1, 1, 0), mem=1); add("vmovq x3<-x1 (d6)", vex(0xD6, 1, 1, 1, 0, 3))
add("vmovd x1,[mem]", vexm(0x6E, 1, 1, 1, 0), mem=1); add("vmovd [mem],x1", vexm(0x7E, 1, 1, 1, 0), mem=1)
add("vmovq x1,[mem] (6e W1)", vexm(0x6E, 1, 1, 1, 0, W=1), mem=1)
for op, nm in ((0x12, "lps"), (0x16, "hps")):
    add(f"vmov{nm} x1,x2,[mem]", vexm(op, 0, 1, 1, 2), mem=1); add(f"vmov{nm[0]}pd x1,x2,[mem]", vexm(op, 1, 1, 1, 2), mem=1)
add("vmovlps [mem],x1", vexm(0x13, 0, 1, 1, 0), mem=1); add("vmovhps [mem],x1", vexm(0x17, 0, 1, 1, 0), mem=1)
add("vmovhlps x1,x2,x3", vex(0x12, 0, 1, 1, 2, 3)); add("vmovlhps x1,x2,x3", vex(0x16, 0, 1, 1, 2, 3))
for pp, op, nm in ((2, 0x12, "sldup"), (2, 0x16, "shdup"), (3, 0x12, "ddup")):
    add(f"vmov{nm} x1,x3", vex(op, pp, 1, 1, 0, 3)); add(f"vmov{nm} y1,y3", vex(op, pp, 1, 1, 0, 3, L=1))
add("vmovddup x1,[mem]", vexm(0x12, 3, 1, 1, 0), mem=1)

# packed / scalar arithmetic
for op, nm in ((0x58,"add"),(0x59,"mul"),(0x5C,"sub"),(0x5D,"min"),(0x5E,"div"),(0x5F,"max"),(0x51,"sqrt")):
    v = 0 if op == 0x51 else 2
    add(f"v{nm}ps x1,x2,x3", vex(op, 0, 1, 1, v, 3)); add(f"v{nm}ps y1,y2,y3", vex(op, 0, 1, 1, v, 3, L=1))
    add(f"v{nm}pd x1,x2,x3", vex(op, 1, 1, 1, v, 3)); add(f"v{nm}pd y1,y2,y3", vex(op, 1, 1, 1, v, 3, L=1))
    add(f"v{nm}ss x1,x2,[mem]", vexm(op, 2, 1, 1, 2), mem=1); add(f"v{nm}ps x1,x2,[mem]", vexm(op, 0, 1, 1, v), mem=1)
add("vaddps y9,y12,y14", vex(0x58, 0, 1, 9, 12, 14, L=1)); add("vmulss x15,x8,x10", vex(0x59, 2, 1, 15, 8, 10))
for op, nm in ((0x55,"andnpd"),(0x56,"orps"),(0x57,"xorpd"),(0x14,"unpcklps"),(0x15,"unpckhps")):
    pp = 1 if nm.endswith("pd") else 0
    add(f"v{nm} x1,x2,x3", vex(op, pp, 1, 1, 2, 3)); add(f"v{nm} y1,y2,y3", vex(op, pp, 1, 1, 2, 3, L=1))
add("vunpcklpd x1,x2,x3", vex(0x14, 1, 1, 1, 2, 3)); add("vunpckhpd y1,y2,y3", vex(0x15, 1, 1, 1, 2, 3, L=1))
for imm in range(8):
    add(f"vcmpps x1,x2,x3,{imm}", vex(0xC2, 0, 1, 1, 2, 3, imm=imm)); add(f"vcmpsd x1,x2,x3,{imm}", vex(0xC2, 3, 1, 1, 2, 3, imm=imm))
add("vcmppd x1,x2,x3,1", vex(0xC2, 1, 1, 1, 2, 3, imm=1)); add("vcmpps y1,y2,y3,2", vex(0xC2, 0, 1, 1, 2, 3, L=1, imm=2))
add("vcmpss x1,x2,[mem],1", vexm(0xC2, 2, 1, 1, 2, imm=1), mem=1); add("vaddsubpd x1,x2,x3", vex(0xD0, 1, 1, 1, 2, 3))

# shuffles, blends, inserts, broadcasts
for imm in (0x1B, 0xE4, 0x4E, 0xB1, 0x00):
    add(f"vshufps x1,x2,x3,{imm:02x}", vex(0xC6, 0, 1, 1, 2, 3, imm=imm))
    add(f"vpermilps x1,x3,{imm:02x}", vex(0x04, 1, 3, 1, 0, 3, imm=imm))
add("vshufps y1,y2,y3,1b", vex(0xC6, 0, 1, 1, 2, 3, L=1, imm=0x1B)); add("vpermilps y1,y3,93", vex(0x04, 1, 3, 1, 0, 3, L=1, imm=0x93))
for imm in range(4):
    add(f"vshufpd x1,x2,x3,{imm}", vex(0xC6, 1, 1, 1, 2, 3, imm=imm))
add("vshufpd y1,y2,y3,5", vex(0xC6, 1, 1, 1, 2, 3, L=1, imm=5)); add("vpshufd x1,x3,1b", vex(0x70, 1, 1, 1, 0, 3, imm=0x1B))
add("vshufps x1,x2,[mem],4e", vexm(0xC6, 0, 1, 1, 2, imm=0x4E), mem=1)
for imm in (0x05, 0x0A, 0x0F):
    add(f"vblendps x1,x2,x3,{imm:02x}", vex(0x0C, 1, 3, 1, 2, 3, imm=imm))
add("vblendps y1,y2,y3,a5", vex(0x0C, 1, 3, 1, 2, 3, L=1, imm=0xA5)); add("vblendpd x1,x2,x3,2", vex(0x0D, 1, 3, 1, 2, 3, imm=2))
add("vblendpd y1,y2,y3,9", vex(0x0D, 1, 3, 1, 2, 3, L=1, imm=9)); add("vblendvpd y1,y2,y3,y4", vex(0x4B, 1, 3, 1, 2, 3, L=1, imm=0x40))
add("vblendvps x1,x2,[mem],x4", vexm(0x4A, 1, 3, 1, 2, imm=0x40), mem=1)
for imm in (0x00, 0x10, 0x20, 0x30, 0x40, 0x0F, 0xC8, 0x5A, 0x1E, 0xF1):
    add(f"vinsertps x1,x2,x3,{imm:02x}", vex(0x21, 1, 3, 1, 2, 3, imm=imm))
add("vinsertps x1,x2,[mem],20", vexm(0x21, 1, 3, 1, 2, imm=0x20), mem=1); add("vinsertps x1,x2,[mem],39", vexm(0x21, 1, 3, 1, 2, imm=0x39), mem=1)
add("vbroadcastss x1,[mem]", vexm(0x18, 1, 2, 1, 0), mem=1); add("vbroadcastss y1,[mem]", vexm(0x18, 1, 2, 1, 0, L=1), mem=1)
add("vpextrd [mem],x1,2", vexm(0x16, 1, 3, 1, 0, imm=2), mem=1); add("vpinsrd x1,x2,[mem],1", vexm(0x22, 1, 3, 1, 2, imm=1), mem=1)

# compares to flags, conversions
add("vcomiss x1,x2", vex(0x2F, 0, 1, 1, 0, 2), ALL); add("vcomisd x1,x2", vex(0x2F, 1, 1, 1, 0, 2), ALL)
add("vucomiss x1,[mem]", vexm(0x2E, 0, 1, 1, 0), ALL, mem=1); add("vucomisd x1,[mem]", vexm(0x2E, 1, 1, 1, 0), ALL, mem=1)
add("vucomiss x1,x1", vex(0x2E, 0, 1, 1, 0, 1), ALL)
add("vcvtss2sd x1,x2,[mem]", vexm(0x5A, 2, 1, 1, 2), mem=1); add("vcvtsd2ss x1,x2,[mem]", vexm(0x5A, 3, 1, 1, 2), mem=1)
add("vcvttss2si ecx,[mem]", vexm(0x2C, 2, 1, 1, 0), mem=1); add("vcvttsd2si rcx,[mem]", vexm(0x2C, 3, 1, 1, 0, W=1), mem=1)
add("vcvtsi2ss x1,x2,[mem]", vexm(0x2A, 2, 1, 1, 2), mem=1); add("vcvtsi2sd x1,x2,[mem] (W1)", vexm(0x2A, 3, 1, 1, 2, W=1), mem=1)

# legacy SSE encodings of the same operations
for pre, nm in (([], "ucomiss"), ([0x66], "ucomisd")):
    add(f"{nm} x1,x3", sse(pre, 0x2E, 1, 3), ALL); add(f"{nm[1:]} x1,x3", sse(pre, 0x2F, 1, 3), ALL)
add("ucomiss x0,[mem]", ssem([], 0x2E, 0), ALL, mem=1)
for op, nm in ((0x54,"andps"),(0x55,"andnps"),(0x56,"orps"),(0x57,"xorps"),(0x14,"unpcklps"),(0x15,"unpckhps")):
    add(f"{nm} x1,x3", sse([], op, 1, 3)); add(f"{nm[:-2]}pd x1,x3", sse([0x66], op, 1, 3))
for imm in range(8):
    add(f"cmpps x1,x3,{imm}", sse([], 0xC2, 1, 3, imm)); add(f"cmpss x1,x3,{imm}", sse([0xF3], 0xC2, 1, 3, imm))
add("cmpsd x0,[mem],2", ssem([0xF2], 0xC2, 0, 2), mem=1)
add("shufps x1,x3,1b", sse([], 0xC6, 1, 3, 0x1B)); add("shufpd x1,x3,1", sse([0x66], 0xC6, 1, 3, 1))
add("pshufd x1,x3,b1", sse([0x66], 0x70, 1, 3, 0xB1)); add("addsubps x1,x3", sse([0xF2], 0xD0, 1, 3))
add("movd x1,ecx", sse([0x66], 0x6E, 1, 1)); add("movq x1,rcx", sse([0x66], 0x6E, 1, 1, rex=8))
add("movd ecx,x1", sse([0x66], 0x7E, 1, 1)); add("movq rcx,x1", sse([0x66], 0x7E, 1, 1, rex=8))
add("movq x1,x3 (f3 7e)", sse([0xF3], 0x7E, 1, 3)); add("movq x3<-x1 (66 d6)", sse([0x66], 0xD6, 1, 3))
add("movq x0,[mem]", ssem([0xF3], 0x7E, 0), mem=1); add("movdqu x0,[mem]", ssem([0xF3], 0x6F, 0), mem=1)
add("movdqa [mem],x0", ssem([0x66], 0x7F, 0), mem=1)
add("movhlps x1,x3", sse([], 0x12, 1, 3)); add("movlhps x1,x3", sse([], 0x16, 1, 3))
add("movlps x0,[mem]", ssem([], 0x12, 0), mem=1); add("movhps x0,[mem]", ssem([], 0x16, 0), mem=1)
add("movhps [mem],x0", ssem([], 0x17, 0), mem=1)
add("movsldup x1,x3", sse([0xF3], 0x12, 1, 3)); add("movshdup x1,x3", sse([0xF3], 0x16, 1, 3)); add("movddup x1,x3", sse([0xF2], 0x12, 1, 3))
add("insertps x1,x3,9c", sse([0x66], [0x3A, 0x21], 1, 3, 0x9C)); add("blendps x1,x3,6", sse([0x66], [0x3A, 0x0C], 1, 3, 6))
add("pextrd ecx,x1,3", sse([0x66], [0x3A, 0x16], 1, 1, 3)); add("pinsrq x1,rcx,1", sse([0x66], [0x3A, 0x22], 1, 1, 1, rex=8))
add("cvtsi2ss x1,ecx", sse([0xF3], 0x2A, 1, 1)); add("cvtsi2sd x1,rcx", sse([0xF2], 0x2A, 1, 1, rex=8))
add("cvtss2sd x1,x3", sse([0xF3], 0x5A, 1, 3)); add("cvtsd2ss x1,x3", sse([0xF2], 0x5A, 1, 3))
add("cvttss2si ecx,[mem]", ssem([0xF3], 0x2C, 1), mem=1)
add("addps x9,x12", sse([], 0x58, 9, 12)); add("mulss x15,x2", sse([0xF3], 0x59, 15, 2))
add("paddd x1,x3", sse([0x66], 0xFE, 1, 3)); add("psubq x1,x3", sse([0x66], 0xFB, 1, 3)); add("pcmpeqd x1,x3", sse([0x66], 0x76, 1, 3))
add("punpcklqdq x1,x3", sse([0x66], 0x6C, 1, 3)); add("pxor x1,x3", sse([0x66], 0xEF, 1, 3))

# ---- JIT integer coverage: shifts/rotates by CL at edge counts, memory forms
for width in (8, 16, 32, 64):
    pre, wide = W[width]
    for sub, nm in ((0,"rol"),(1,"ror"),(4,"shl"),(5,"shr"),(7,"sar")):
        rot = sub < 4
        add(f"{nm}{width} rax,cl (edge)", pre + [0xD3 if wide else 0xD2, 0xC0 | (sub << 3)],
            CF if rot else (CF | PF | ZF | SF), setup=3)
    add(f"shl{width} [mem],cl", pre + [0xD3 if wide else 0xD2, 0xA4] + MEMOP[1:], CF | PF | ZF | SF, mem=1, setup=3)
    add(f"neg{width} [mem]", pre + [0xF7 if wide else 0xF6, 0x9C] + MEMOP[1:], NOAF, mem=1)
    add(f"not{width} [mem]", pre + [0xF7 if wide else 0xF6, 0x94] + MEMOP[1:], mem=1)
for width in (16, 32, 64):
    pre, _ = W[width]
    add(f"imul{width} rcx,[mem]", pre + [0x0F, 0xAF, 0x8C] + MEMOP[1:], CF | OF, mem=1)
    add(f"imul{width} rcx,[mem],0x1234", pre + [0x69, 0x8C] + MEMOP[1:] + [0x34, 0x12] + ([0, 0] if width != 16 else []), CF | OF, mem=1)
    add(f"imul{width} rcx,rdx,127", pre + [0x6B, 0xCA, 0x7F], CF | OF)
    add(f"bt{width} [mem],13", pre + [0x0F, 0xBA, 0xA4] + MEMOP[1:] + [13], CF, mem=1)
    add(f"btc{width} [mem],2", pre + [0x0F, 0xBA, 0xBC] + MEMOP[1:] + [2], CF, mem=1)
    add(f"bt{width} rax,rcx (rcx>=width)", pre + [0x0F, 0xA3, 0xC8], CF)
add("neg8 ah", [0xF6, 0xDC], NOAF); add("not8 ch", [0xF6, 0xD5]); add("imul32 eax,eax", [0x0F, 0xAF, 0xC0], CF | OF)

# ---- AVX forms from Dragon Ball Xenoverse 2's vectorised binding count
# (0xcbb180): 64-bit lane extract to memory, byte inserts, horizontal add.
MEM_RM = lambda reg: [0x84 | (reg << 3)] + MEMOP[1:]
add("vpextrq [mem],x2,1", [0xC4, 0xE3, 0xF9, 0x16] + MEM_RM(2) + [0x01], mem=1)
add("vpextrq [mem],x2,0", [0xC4, 0xE3, 0xF9, 0x16] + MEM_RM(2) + [0x00], mem=1)
add("vpextrq rsi,x2,1", [0xC4, 0xE3, 0xF9, 0x16, 0xD6, 0x01])
add("vpinsrb x2,x0,[mem],4", [0xC4, 0xE3, 0x79, 0x20] + MEM_RM(2) + [0x04], mem=1)
add("vpinsrb x2,x0,[mem],c", [0xC4, 0xE3, 0x79, 0x20] + MEM_RM(2) + [0x0C], mem=1)
add("vpinsrb x2,x0,esi,8", [0xC4, 0xE3, 0x79, 0x20, 0xD6, 0x08])
add("vpinsrb x2,x2,esi,0", [0xC4, 0xE3, 0x69, 0x20, 0xD6, 0x00])
add("vphaddd x0,x0,x0", [0xC4, 0xE2, 0x79, 0x02, 0xC0])
add("vphaddd x1,x2,x3", [0xC4, 0xE2, 0x69, 0x02, 0xCB])
add("vpslldq x7,x2,8", [0xC5, 0xC1, 0x73, 0xFA, 0x08])
add("vpshufd x6,x6,44", [0xC5, 0xF9, 0x70, 0xF6, 0x44])
add("vpshufd x1,x0,4e", [0xC5, 0xF9, 0x70, 0xC8, 0x4E])
add("vpcmpeqd x2,x2,x5", [0xC5, 0xE9, 0x76, 0xD5])

# ---- Interpreter-only AVX/BMI2 ops of the game's CRI video decoder (IDCT /
# motion compensation), ranked by PS4_JIT_STATS during the opening movie.
for nm, op in (("vpmaddwd", 0xF5), ("vpaddsw", 0xED), ("vpsubsw", 0xE9), ("vpavgb", 0xE0),
               ("vpackssdw", 0x6B), ("vpmulhw", 0xE5), ("vpaddw", 0xFD), ("vpackuswb", 0x67),
               ("vpsubb", 0xF8), ("vpmulhuw", 0xE4), ("vpmullw", 0xD5), ("vpaddusb", 0xDC),
               ("vpsubusb", 0xD8), ("vpackusdw_like_67", 0x63), ("vpavgw", 0xE3)):
    add(f"{nm} x5,x4,x2", vex(op, 1, 1, 5, 4, 2))
    add(f"{nm} x3,x7,[mem]", vex(op, 1, 1, 3, 7, 0, mod_rm=0x84 | (3 << 3)) + MEMOP[1:], mem=1)
for cnt in (1, 3, 15, 16, 20, 31, 40):
    add(f"vpsraw x5,x2,{cnt}", vex(0x71, 1, 1, 4, 5, 2, imm=cnt))
    add(f"vpsrad x5,x2,{cnt}", vex(0x72, 1, 1, 4, 5, 2, imm=cnt))
    add(f"vpsrlw x5,x2,{cnt}", vex(0x71, 1, 1, 2, 5, 2, imm=cnt))
    add(f"vpsllw x5,x2,{cnt}", vex(0x71, 1, 1, 6, 5, 2, imm=cnt))
for imm in (0x1B, 0xE4, 0x00, 0x4E):
    add(f"vpshuflw x5,x2,{imm:x}", vex(0x70, 3, 1, 5, 0, 2, imm=imm))
    add(f"vpshufhw x5,x2,{imm:x}", vex(0x70, 2, 1, 5, 0, 2, imm=imm))
add("vpshuflw x5,[mem],1b", vex(0x70, 3, 1, 5, 0, 0, imm=None, mod_rm=0x84 | (5 << 3)) + MEMOP[1:] + [0x1B], mem=1)
add("vpmovzxbw x5,x2", vex(0x30, 1, 2, 5, 0, 2))
add("vpmovzxbw x5,[mem]", vex(0x30, 1, 2, 5, 0, 0, mod_rm=0x84 | (5 << 3)) + MEMOP[1:], mem=1)
add("vpmovsxbw x5,x2", vex(0x20, 1, 2, 5, 0, 2))
add("vpshufb x5,x4,x2", vex(0x00, 1, 2, 5, 4, 2))
add("vinsertf128 y5,y4,x2,1", vex(0x18, 1, 3, 5, 4, 2, L=1, imm=1))
add("vinsertf128 y5,y4,x2,0", vex(0x18, 1, 3, 5, 4, 2, L=1, imm=0))
add("vpermilpd x5,x2,1", vex(0x05, 1, 3, 5, 0, 2, imm=1))
add("vpermilpd y5,y2,5", vex(0x05, 1, 3, 5, 0, 2, L=1, imm=5))
for w in (0, 1):
    add(f"shlx{32 << w} rcx,rax,rdx", vex(0xF7, 1, 2, 1, 2, 0, W=w))
    add(f"sarx{32 << w} rcx,rax,rdx", vex(0xF7, 2, 2, 1, 2, 0, W=w))
    add(f"shrx{32 << w} rcx,rax,rdx", vex(0xF7, 3, 2, 1, 2, 0, W=w))
    add(f"andn{32 << w} rcx,rdx,rax", vex(0xF2, 0, 2, 1, 2, 0, W=w), SF | ZF | OF | CF)

# ---- Legacy SSE forms of the word shuffles and zero/sign extensions.
for imm in (0x1B, 0x4E):
    add(f"pshuflw x1,x3,{imm:x}", [0xF2, 0x0F, 0x70, 0xCB, imm])
    add(f"pshufhw x1,x3,{imm:x}", [0xF3, 0x0F, 0x70, 0xCB, imm])
for nm, op in (("pmovsxbw", 0x20), ("pmovsxbd", 0x21), ("pmovsxwd", 0x23), ("pmovsxdq", 0x25),
               ("pmovzxbw", 0x30), ("pmovzxbd", 0x31), ("pmovzxwd", 0x33), ("pmovzxdq", 0x35)):
    add(f"{nm} x1,x3", [0x66, 0x0F, 0x38, op, 0xCB])
    add(f"v{nm} x5,x2", vex(op, 1, 2, 5, 0, 2))
    add(f"v{nm} y5,x2", vex(op, 1, 2, 5, 0, 2, L=1))

with open("cases.h", "w") as f:
    f.write("// Generated by gen_cases.py -- shared by native.c and emu.cpp.\n#include <stdint.h>\n")
    f.write("struct Case { const char* name; uint8_t len; uint8_t bytes[15]; uint32_t fmask; int mem; int setup; };\n")
    f.write("static const struct Case kCases[] = {\n")
    for nm, b, fm, mem, setup in cases:
        f.write('  {"%s", %d, {%s}, 0x%x, %d, %d},\n' % (nm, len(b), ",".join("0x%02x" % x for x in b), fm, mem, setup))
    f.write("};\n#define NCASES (sizeof(kCases)/sizeof(kCases[0]))\n")
    f.write(open("state.h.in").read())
print(len(cases), "cases")
