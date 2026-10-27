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
    2 = signed-div safe (rdx = sign of rax, rcx != 0 and not -1)."""
    cases.append((name, b, fmask, mem, setup))

# Existing GC size-map loop
add("vpaddq x5,x4,x2", [0xc5,0xd9,0xd4,0xea]); add("vmovq rsi,x4", [0xc4,0xe1,0xf9,0x7e,0xe6])
add("vpblendw x5,x5,x3,cc", [0xc4,0xe3,0x51,0x0e,0xeb,0xcc]); add("vpsrlq x5,3", [0xc5,0xd1,0x73,0xd5,0x03])
add("vpshufd x5,x5,e8", [0xc5,0xf9,0x70,0xed,0xe8]); add("vmovq [rax+4rsi+1ab0],x5", [0xc5,0xf9,0xd6,0xac,0xb0,0xb0,0x1a,0,0], mem=1)
add("vmovd x2,ecx", [0xc5,0xf9,0x6e,0xd1]); add("vpxor x3,x3,x3", [0xc5,0xe1,0xef,0xdb]); add("vpand x7,x7,x5", [0xc5,0xc1,0xdb,0xfd])

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
add("xchg rax,rcx", [0x48, 0x91]); add("lea rax,[rcx+rdx*4+9]", [0x48, 0x8D, 0x44, 0x91, 0x09])
add("bswap ecx", [0x0F, 0xC9]); add("bswap rcx", [0x48, 0x0F, 0xC9])
add("adc rax,0x7f", [0x48, 0x83, 0xD0, 0x7F], NOAF); add("sbb eax,-1", [0x83, 0xD8, 0xFF], NOAF)

with open("cases.h", "w") as f:
    f.write("// Generated by gen_cases.py -- shared by native.c and emu.cpp.\n#include <stdint.h>\n")
    f.write("struct Case { const char* name; uint8_t len; uint8_t bytes[15]; uint32_t fmask; int mem; int setup; };\n")
    f.write("static const struct Case kCases[] = {\n")
    for nm, b, fm, mem, setup in cases:
        f.write('  {"%s", %d, {%s}, 0x%x, %d, %d},\n' % (nm, len(b), ",".join("0x%02x" % x for x in b), fm, mem, setup))
    f.write("};\n#define NCASES (sizeof(kCases)/sizeof(kCases[0]))\n")
    f.write(open("state.h.in").read())
print(len(cases), "cases")
