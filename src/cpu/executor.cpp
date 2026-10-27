#include "ps4/executor.hpp"
#include "ps4/dynamic_linker.hpp"
#include "ps4/jit_compiler.hpp"
#include "ps4/jit_stats.hpp"
#include <chrono>
#include <cstring>
#include <cmath>
#include <iostream>
#include <mutex>

namespace ps4 {

std::string_view step_result_to_string(StepResult result) {
    switch (result) {
        case StepResult::Ok:
            return "Instruction executed successfully";
        case StepResult::Syscall:
            return "Syscall executed";
        case StepResult::Halted:
            return "CPU Halted (HLT)";
        case StepResult::Exited:
            return "Process exited (SYS_exit)";
        case StepResult::IllegalInstruction:
            return "Illegal or unimplemented instruction";
        case StepResult::MemoryFault:
            return "Memory fault / invalid instruction pointer";
    }
    return "Unknown step result";
}

namespace {

// VEX.L for the instruction currently being decoded. The 0F38/0F3A handlers
// sit in a different scope from the VEX prefix parse, so the bit is published
// at file scope rather than threaded through as a local.
static thread_local bool g_vex_L = false;
// VEX R/X/B extension bits. The 0F38 handlers are in a different scope from the
// VEX prefix parser, so these are published at file scope too. Passing the REX
// bits instead made `vbroadcastsd ymm0, [r14+0xfce8]` (VEX byte c2 sets ~B=0,
// i.e. vex_b=1, so rm=6+8=14) address [rsi+0xfce8] instead.
static thread_local u8 g_vex_map = 1, g_vex_op = 0;
static thread_local u8 g_vex_r = 1, g_vex_x = 1, g_vex_b = 1;
static thread_local u8 g_vex_vvvv = 0;

inline u64& get_gpr64(CpuContext& ctx, u8 reg_idx) {
    switch (reg_idx & 0x0F) {
        case 0:  return ctx.rax;
        case 1:  return ctx.rcx;
        case 2:  return ctx.rdx;
        case 3:  return ctx.rbx;
        case 4:  return ctx.rsp;
        case 5:  return ctx.rbp;
        case 6:  return ctx.rsi;
        case 7:  return ctx.rdi;
        case 8:  return ctx.r8;
        case 9:  return ctx.r9;
        case 10: return ctx.r10;
        case 11: return ctx.r11;
        case 12: return ctx.r12;
        case 13: return ctx.r13;
        case 14: return ctx.r14;
        case 15: return ctx.r15;
        default: return ctx.rax;
    }
}

inline const u64& get_gpr64(const CpuContext& ctx, u8 reg_idx) {
    switch (reg_idx & 0x0F) {
        case 0:  return ctx.rax;
        case 1:  return ctx.rcx;
        case 2:  return ctx.rdx;
        case 3:  return ctx.rbx;
        case 4:  return ctx.rsp;
        case 5:  return ctx.rbp;
        case 6:  return ctx.rsi;
        case 7:  return ctx.rdi;
        case 8:  return ctx.r8;
        case 9:  return ctx.r9;
        case 10: return ctx.r10;
        case 11: return ctx.r11;
        case 12: return ctx.r12;
        case 13: return ctx.r13;
        case 14: return ctx.r14;
        case 15: return ctx.r15;
        default: return ctx.rax;
    }
}

inline u8 get_gpr8(const CpuContext& ctx, u8 reg_idx, bool has_rex) {
    if (!has_rex && reg_idx >= 4 && reg_idx <= 7) {
        switch (reg_idx) {
            case 4: return static_cast<u8>((ctx.rax >> 8) & 0xFF); // AH
            case 5: return static_cast<u8>((ctx.rcx >> 8) & 0xFF); // CH
            case 6: return static_cast<u8>((ctx.rdx >> 8) & 0xFF); // DH
            case 7: return static_cast<u8>((ctx.rbx >> 8) & 0xFF); // BH
            default: break;
        }
    }
    return static_cast<u8>(get_gpr64(ctx, reg_idx) & 0xFF);
}

inline void set_gpr8(CpuContext& ctx, u8 reg_idx, bool has_rex, u8 val) {
    if (!has_rex && reg_idx >= 4 && reg_idx <= 7) {
        switch (reg_idx) {
            case 4: ctx.rax = (ctx.rax & ~0xFF00ULL) | (static_cast<u64>(val) << 8); return;
            case 5: ctx.rcx = (ctx.rcx & ~0xFF00ULL) | (static_cast<u64>(val) << 8); return;
            case 6: ctx.rdx = (ctx.rdx & ~0xFF00ULL) | (static_cast<u64>(val) << 8); return;
            case 7: ctx.rbx = (ctx.rbx & ~0xFF00ULL) | (static_cast<u64>(val) << 8); return;
            default: break;
        }
    }
    u64& reg = get_gpr64(ctx, reg_idx);
    reg = (reg & ~0xFFULL) | val;
}

inline void set_gpr16(CpuContext& ctx, u8 reg_idx, u16 val) {
    u64& reg = get_gpr64(ctx, reg_idx);
    reg = (reg & ~0xFFFFULL) | val;
}

inline void set_gpr32(CpuContext& ctx, u8 reg_idx, u32 val) {
    // 32-bit register writes in x86_64 zero-extend to 64-bit
    get_gpr64(ctx, reg_idx) = static_cast<u64>(val);
}

inline void set_gpr64(CpuContext& ctx, u8 reg_idx, u64 val) {
    get_gpr64(ctx, reg_idx) = val;
}

inline void update_flags_logic(CpuContext& ctx, u64 result, usize bits) {
    ctx.update_flags_logic(result, bits);
}

inline void update_flags_add(CpuContext& ctx, u64 a, u64 b, u64 result, usize bits) {
    ctx.update_flags_add(a, b, result, bits);
}

inline void update_flags_sub(CpuContext& ctx, u64 a, u64 b, u64 result, usize bits) {
    ctx.update_flags_sub(a, b, result, bits);
}

inline bool check_condition(const CpuContext& ctx, u8 cc) {
    const bool cf = ctx.flag_cf();
    const bool pf = ctx.flag_pf();
    const bool zf = ctx.flag_zf();
    const bool sf = ctx.flag_sf();
    const bool of = ctx.flag_of();
    bool cond = false;
    switch ((cc >> 1) & 0x07) {
        case 0: cond = of; break;                 // O
        case 1: cond = cf; break;                 // B/C/NAE
        case 2: cond = zf; break;                 // E/Z
        case 3: cond = cf || zf; break;           // BE/NA
        case 4: cond = sf; break;                 // S
        case 5: cond = pf; break;                 // P/PE (Fixed!)
        case 6: cond = (sf != of); break;         // L/NGE
        case 7: cond = zf || (sf != of); break;   // LE/NG
        default: break;
    }
    return (cc & 1) ? !cond : cond;
}

// x86 float -> integer conversion: NaN and out-of-range inputs produce the
// "integer indefinite" value (0x80000000 / 0x8000000000000000). A plain C++
// cast is undefined there and saturates on the ARM64 host.
inline u32 x86_cvt_i32(double v) {
    if (!(v > -2147483649.0 && v < 2147483648.0)) return 0x80000000u;
    return static_cast<u32>(static_cast<i32>(v));
}
inline u64 x86_cvt_i64(double v) {
    if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0)) return 0x8000000000000000ull;
    return static_cast<u64>(static_cast<i64>(v));
}

u64 execute_alu_op(CpuContext& ctx, u8 op, u64 a, u64 b, usize bits) {
    const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
    a &= mask;
    b &= mask;
    u64 res = 0;
    switch (op) {
        case 0: // ADD
            res = (a + b) & mask;
            update_flags_add(ctx, a, b, res, bits);
            return res;
        case 1: // OR
            res = (a | b) & mask;
            update_flags_logic(ctx, res, bits);
            return res;
        case 2: { // ADC
            const u64 cf = ctx.flag_cf() ? 1 : 0;
            res = (a + b + cf) & mask;
            const bool new_cf = ((res & mask) < (a & mask)) || (((res & mask) == (a & mask)) && (cf != 0));
            update_flags_add(ctx, a, b + cf, res, bits);
            ctx.set_flag(CpuContext::FLAG_CF, new_cf);
            return res;
        }
        case 3: { // SBB
            const u64 cf = ctx.flag_cf() ? 1 : 0;
            res = (a - b - cf) & mask;
            const bool new_cf = ((a & mask) < (b & mask)) || (((a & mask) == (b & mask)) && (cf != 0));
            update_flags_sub(ctx, a, b, res, bits);
            ctx.set_flag(CpuContext::FLAG_CF, new_cf);
            return res;
        }
        case 4: // AND
            res = (a & b) & mask;
            update_flags_logic(ctx, res, bits);
            return res;
        case 5: // SUB
            res = (a - b) & mask;
            update_flags_sub(ctx, a, b, res, bits);
            return res;
        case 6: // XOR
            res = (a ^ b) & mask;
            update_flags_logic(ctx, res, bits);
            return res;
        case 7: // CMP
            res = (a - b) & mask;
            update_flags_sub(ctx, a, b, res, bits);
            return a;
        default:
            return a;
    }
}

struct ModRmResult {
    bool is_reg{false};
    u8 reg_id{0};
    u8 rm_id{0};
    GuestAddr mem_addr{0};
    usize bytes_consumed{0};
    bool valid{false};
};

inline ModRmResult decode_modrm(const u8* code, usize available, GuestAddr next_ip_base,
                                u8 rex_r, u8 rex_x, u8 rex_b, u8 seg_override,
                                const CpuContext& ctx, usize extra_imm_bytes = 0) {
    ModRmResult res{};
    if (available < 1) return res;

    const u8 modrm = code[0];
    const u8 mod = (modrm >> 6) & 3;
    const u8 reg = (modrm >> 3) & 7;
    const u8 rm  = modrm & 7;

    res.reg_id = (rex_r ? 8 : 0) | reg;
    usize offset = 1;

    if (mod == 3) {
        res.is_reg = true;
        res.rm_id = (rex_b ? 8 : 0) | rm;
        res.bytes_consumed = 1;
        res.valid = true;
        return res;
    }

    res.is_reg = false;
    GuestAddr base_addr = 0;
    bool has_base = false;

    if (rm == 4) {
        // SIB byte follows
        if (available < offset + 1) return res;
        const u8 sib = code[offset++];
        const u8 scale = (sib >> 6) & 3;
        const u8 index = (sib >> 3) & 7;
        const u8 base  = sib & 7;

        const u8 index_reg = (rex_x ? 8 : 0) | index;
        if (index_reg != 4) { // RSP cannot be index
            base_addr += get_gpr64(ctx, index_reg) << scale;
        }

        const u8 base_reg = (rex_b ? 8 : 0) | base;
        // SIB base=101 with mod=00 means "no base, disp32" whatever REX.B
        // says (r13 as a base needs mod=01 with disp8=0).
        if (mod == 0 && base == 5) {
            has_base = false; // [index*scale + disp32] (no base register)
        } else {
            base_addr += get_gpr64(ctx, base_reg);
            has_base = true;
        }
    } else if (mod == 0 && rm == 5) {
        // RIP-relative
        has_base = false;
    } else {
        const u8 base_reg = (rex_b ? 8 : 0) | rm;
        base_addr += get_gpr64(ctx, base_reg);
        has_base = true;
    }

    if (mod == 0) {
        if (rm == 5) {
            if (available < offset + 4) return res;
            i32 disp32 = 0;
            std::memcpy(&disp32, code + offset, 4);
            offset += 4;
            // RIP-relative calculation: next_ip_base + offset + extra_imm_bytes + disp32
            res.mem_addr = next_ip_base + offset + extra_imm_bytes + disp32;
        } else if (rm == 4 && !has_base) {
            if (available < offset + 4) return res;
            i32 disp32 = 0;
            std::memcpy(&disp32, code + offset, 4);
            offset += 4;
            // SIB with no base: [index*scale + disp32]
            res.mem_addr = base_addr + static_cast<i64>(disp32);
        } else {
            res.mem_addr = base_addr;
        }
    } else if (mod == 1) {
        if (available < offset + 1) return res;
        const i8 disp8 = static_cast<i8>(code[offset++]);
        res.mem_addr = base_addr + disp8;
    } else if (mod == 2) {
        if (available < offset + 4) return res;
        i32 disp32 = 0;
        std::memcpy(&disp32, code + offset, 4);
        offset += 4;
        res.mem_addr = base_addr + disp32;
    }

    if (seg_override == 0x64) {
        res.mem_addr += ctx.fs_base;
    } else if (seg_override == 0x65) {
        res.mem_addr += ctx.gs_base;
    }

    res.bytes_consumed = offset;
    res.valid = true;
    return res;
}

// 80-bit extended <-> host double.
//
// PRECISION LOST: the x87 stack slot holds an 80-bit value with a 64-bit
// mantissa and an explicit integer bit. Each slot here is a host double, so
// only 53 mantissa bits survive, and the explicit integer bit is folded into
// the implicit leading one. An FLD m80 / FSTP m80 pair therefore round-trips
// any value representable in binary64 exactly, but a value needing bits 54..64
// of the extended mantissa comes back rounded. The sign and exponent survive
// intact, including denormals, infinities and NaN payloads only insofar as a
// double can express them.
double f80_to_double(const u8 in[10]) {
    u64 mant = 0;
    for (int i = 7; i >= 0; --i) mant = (mant << 8) | in[i];
    const u16 se = static_cast<u16>(in[8] | (in[9] << 8));
    const u64 exp80 = se & 0x7FFF;
    const u64 sign = (se >> 15) & 1;
    if (exp80 == 0 && mant == 0) return sign ? -0.0 : 0.0;
    if (exp80 == 0x7FFF) {
        // Preserve NaN-ness; the payload cannot fit a double's NaN space fully,
        // so map any non-zero low mantissa to a quiet NaN with that sign.
        const bool is_nan = (mant & 0x7FFFFFFFFFFFFFFFULL) != 0;
        if (!is_nan) return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }
    // Unbias: extended bias 16383, double bias 1023, so the biased double
    // exponent is (exp80 - 16383) + 1023 = exp80 - 15360 = exp80 - 0x3C00.
    // (Subtracting 0x3800 alone left 2047 too large and every normal value
    // overflowed to infinity.)
    const u64 be = exp80 - 0x3C00;
    // Extended mantissa bit k has weight 2^(k-63). The binary64 fraction field
    // holds bits 62..11 of the extended mantissa, so shift right by 11 and keep
    // 52 bits. Dropping only the explicit integer bit and re-ORing a low 1 was
    // wrong twice over: it kept 52 bits but at the wrong exponents, and it
    // fabricated a fraction bit.
    const u64 frac = (mant >> 11) & 0x000FFFFFFFFFFFFFULL;
    u64 bits = (sign << 63) | (be << 52) | frac;
    double d;
    std::memcpy(&d, &bits, 8);
    return d;
}

void double_to_f80(double d, u8 out[10]) {
    std::memset(out, 0, 10);
    u64 bits;
    std::memcpy(&bits, &d, 8);
    const u64 frac = bits & 0x000FFFFFFFFFFFFFULL;
    const u64 be = (bits >> 52) & 0x7FF;
    const u64 sign = (bits >> 63) & 1;
    u16 se;
    if (be == 0 && frac == 0) {
        se = static_cast<u16>(sign << 15);
    } else if (be == 0x7FF) {
        se = static_cast<u16>((sign << 15) | 0x7FFF);
        out[8] = out[9] = 0xFF;
    } else {
        // Extended bias is 16383 and double bias is 1023, so the extended
        // exponent is (be - 1023) + 16383 = be + 15360 = be + 0x3C00.
        // (Subtracting 1023 and adding only 0x3800 left 1.0 at 0x3800, the
        // smallest normal exponent, instead of 0x3FFF.)
        se = static_cast<u16>((sign << 15) | (be + 0x3C00));
        // Inverse of the >>11 shift above: restore bits 62..11 and re-attach the
        // explicit integer bit at bit 63.
        const u64 mant = ((frac & 0x000FFFFFFFFFFFFFULL) << 11) | 0x8000000000000000ULL;
        for (int i = 0; i < 8; ++i) out[i] = static_cast<u8>((mant >> (8 * i)) & 0xFF);
    }
    out[8] = static_cast<u8>(se & 0xFF);
    out[9] = static_cast<u8>((se >> 8) & 0xFF);
}

} // namespace

CpuExecutor::CpuExecutor(MemoryManager& memory, SyscallTable& syscalls)
    : m_memory(memory), m_syscalls(syscalls) {}

bool CpuExecutor::setup_default_stack(GuestAddr stack_base, GuestSize stack_size) {
    if (!m_memory.is_valid_address(stack_base, stack_size)) {
        return false;
    }

    m_memory.zero_memory(stack_base, stack_size);

    // Standard FreeBSD / System V ABI Initial Stack Layout
    const GuestAddr sp = (stack_base + stack_size - 0x100) & ~0xFULL;
    
    // Store program path string
    const char prog_name[] = "/app0/eboot.bin";
    const GuestAddr str_addr = sp + 0x80;
    m_memory.write_bytes(str_addr, prog_name, sizeof(prog_name));

    // Arguments: [sp] = argc (1), [sp+8] = argv[0], [sp+16] = NULL, [sp+24] = envp NULL
    const u64 argc = 1;
    m_memory.write_value(sp, argc);
    m_memory.write_value(sp + 8, str_addr);
    m_memory.write_value(sp + 16, 0ULL);
    m_memory.write_value(sp + 24, 0ULL);

    m_context.rdi = sp;
    m_context.rsi = 0;
    m_context.rdx = 0;

    m_context.rsp = sp;
    m_context.rbp = sp;

    // Standard x86_64 ELF TLS / TCB initial layout:
    // fs_base points to TCB, where (%fs:0) points to fs_base itself (tcb->tcb_self).
    // Static TLS variables with negative offsets live immediately below fs_base.
    const GuestAddr tls_base = (stack_base + 0x10000) & ~0xFFFULL;
    m_context.fs_base = tls_base;
    m_memory.write_value(tls_base, tls_base);
    return true;
}

// ---------------------------------------------------------------------------
// Guest LOCK-prefixed read-modify-write.
//
// Every guest thread runs on its own host thread over ONE shared mmap of guest
// memory. The LOCK prefix was decoded into `lock_prefix` and then never read, so
// `lock` instructions fell through to read_rm()/write_rm() -- a plain
// load-then-store that loses updates whenever two guests touch the same word.
//
// That is not hypothetical here. Unity drives both of its job systems from
// exactly these instructions:
//     0x975ba  lock incq (%rax)               job-done counter
//     0x975c2  lock decq (%rax)               outstanding--
//     0x975eb  lock decq (%r13)               queue pending--
//     0x98d30a7 lock cmpxchgl %r15d, (%rcx)   ThreadedStreamBuffer producer CAS
//     0x975b0  xchgq %rax, (%r15)             queue slot swap
// so the pool's task queue and the render ringbuffer were being corrupted by
// lost updates. That is consistent with every symptom seen so far: stack canary
// failures whose faulting frame moves between runs, job queues that never drain,
// and wild indirect jumps through a corrupted pointer.
//
// These helpers issue a single host atomic instead. They return -1 when the
// operand cannot be handled this way (register destination, or an address with
// no host mapping) so the caller can fall back to its existing path.
// ---------------------------------------------------------------------------
// Striped mutex pool for atomic memory ops (misaligned accesses and CMPXCHG16B)
static constexpr usize ATOMIC_STRIPE_COUNT = 64;
struct alignas(64) StripedMutex {
    std::mutex mtx;
};
static StripedMutex s_atomic_mutexes[ATOMIC_STRIPE_COUNT];

static inline std::mutex& get_atomic_mutex(GuestAddr addr) {
    return s_atomic_mutexes[(addr >> 3) % ATOMIC_STRIPE_COUNT].mtx;
}

template <typename T>
static int guest_atomic_cas(MemoryManager& mem, GuestAddr addr, T expected, T desired, T* out_actual) {
    if ((addr & (sizeof(T) - 1)) != 0) {
        std::lock_guard<std::mutex> lock(get_atomic_mutex(addr));
        T* p = mem.to_host_ptr<T>(addr);
        if (!p) return -1;
        T current;
        std::memcpy(&current, p, sizeof(T));
        *out_actual = current;
        if (current == expected) {
            std::memcpy(p, &desired, sizeof(T));
            return 1;
        }
        return 0;
    }
    T* p = mem.to_host_ptr<T>(addr);
    if (!p) return -1;
    T exp = expected;
    if (__atomic_compare_exchange_n(p, &exp, desired, /*weak=*/false,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        *out_actual = expected;
        return 1;
    }
    *out_actual = exp;
    return 0;
}

template <typename T>
static int guest_atomic_xadd(MemoryManager& mem, GuestAddr addr, T operand, T* out_old) {
    if ((addr & (sizeof(T) - 1)) != 0) {
        std::lock_guard<std::mutex> lock(get_atomic_mutex(addr));
        T* p = mem.to_host_ptr<T>(addr);
        if (!p) return -1;
        T cur;
        std::memcpy(&cur, p, sizeof(T));
        *out_old = cur;
        T next = cur + operand;
        std::memcpy(p, &next, sizeof(T));
        return 1;
    }
    T* p = mem.to_host_ptr<T>(addr);
    if (!p) return -1;
    *out_old = __atomic_fetch_add(p, operand, __ATOMIC_SEQ_CST);
    return 1;
}

template <typename T>
static int guest_atomic_xchg(MemoryManager& mem, GuestAddr addr, T operand, T* out_old) {
    if ((addr & (sizeof(T) - 1)) != 0) {
        std::lock_guard<std::mutex> lock(get_atomic_mutex(addr));
        T* p = mem.to_host_ptr<T>(addr);
        if (!p) return -1;
        T cur;
        std::memcpy(&cur, p, sizeof(T));
        *out_old = cur;
        std::memcpy(p, &operand, sizeof(T));
        return 1;
    }
    T* p = mem.to_host_ptr<T>(addr);
    if (!p) return -1;
    *out_old = __atomic_exchange_n(p, operand, __ATOMIC_SEQ_CST);
    return 1;
}

template <typename T>
static int guest_atomic_and(MemoryManager& mem, GuestAddr addr, T operand, T* out_old) {
    if ((addr & (sizeof(T) - 1)) != 0) {
        std::lock_guard<std::mutex> lock(get_atomic_mutex(addr));
        T* p = mem.to_host_ptr<T>(addr);
        if (!p) return -1;
        T cur;
        std::memcpy(&cur, p, sizeof(T));
        *out_old = cur;
        T next = cur & operand;
        std::memcpy(p, &next, sizeof(T));
        return 1;
    }
    T* p = mem.to_host_ptr<T>(addr);
    if (!p) return -1;
    *out_old = __atomic_fetch_and(p, operand, __ATOMIC_SEQ_CST);
    return 1;
}

template <typename T>
static int guest_atomic_or(MemoryManager& mem, GuestAddr addr, T operand, T* out_old) {
    if ((addr & (sizeof(T) - 1)) != 0) {
        std::lock_guard<std::mutex> lock(get_atomic_mutex(addr));
        T* p = mem.to_host_ptr<T>(addr);
        if (!p) return -1;
        T cur;
        std::memcpy(&cur, p, sizeof(T));
        *out_old = cur;
        T next = cur | operand;
        std::memcpy(p, &next, sizeof(T));
        return 1;
    }
    T* p = mem.to_host_ptr<T>(addr);
    if (!p) return -1;
    *out_old = __atomic_fetch_or(p, operand, __ATOMIC_SEQ_CST);
    return 1;
}

template <typename T>
static int guest_atomic_xor(MemoryManager& mem, GuestAddr addr, T operand, T* out_old) {
    if ((addr & (sizeof(T) - 1)) != 0) {
        std::lock_guard<std::mutex> lock(get_atomic_mutex(addr));
        T* p = mem.to_host_ptr<T>(addr);
        if (!p) return -1;
        T cur;
        std::memcpy(&cur, p, sizeof(T));
        *out_old = cur;
        T next = cur ^ operand;
        std::memcpy(p, &next, sizeof(T));
        return 1;
    }
    T* p = mem.to_host_ptr<T>(addr);
    if (!p) return -1;
    *out_old = __atomic_fetch_xor(p, operand, __ATOMIC_SEQ_CST);
    return 1;
}

template <typename T>
static int guest_atomic_neg(MemoryManager& mem, GuestAddr addr, T* out_old) {
    std::lock_guard<std::mutex> lock(get_atomic_mutex(addr));
    T* p = mem.to_host_ptr<T>(addr);
    if (!p) return -1;
    T cur;
    std::memcpy(&cur, p, sizeof(T));
    *out_old = cur;
    T next = static_cast<T>(0) - cur;
    std::memcpy(p, &next, sizeof(T));
    return 1;
}

// ---------------------------------------------------------------------------
// Control-transfer ring (diagnostic).
//
// The stack canary at [rbp-0x30] is clobbered by a callee prologue, but a
// frame canary alone cannot say HOW control reached that callee. This records
// the last 64 control transfers per thread -- CALL, RET, indirect JMP and
// SYSCALL -- with the guest stack pointer and frame pointer, so a dump at the
// abort shows the exact call/ret sequence and F1's real entry point.
// ---------------------------------------------------------------------------
struct Xfer {
    GuestAddr from_rip;
    GuestAddr to_rip;
    GuestAddr rsp;
    GuestAddr rbp;
    u8 kind; // 0=CALL 1=RET 2=IND-JMP 3=SYSCALL 4=IRET
};
static constexpr usize kXferRing = 64;
static thread_local Xfer t_xfer[kXferRing];
static thread_local usize t_xfer_n = 0;

static void record_xfer(GuestAddr from, GuestAddr to, GuestAddr rsp, GuestAddr rbp, u8 kind) {
    Xfer& e = t_xfer[t_xfer_n % kXferRing];
    e = Xfer{from, to, rsp, rbp, kind};
    ++t_xfer_n;
}

void dump_control_xfers(int tid) {
    static const char* kNames[] = {"CALL ", "RET  ", "IJMP ", "SYSC ", "IRET "};
    const usize n = t_xfer_n;
    std::fprintf(stderr, "[xfer] thread %d: %zu transfers recorded\n", tid, n);
    const usize start = (n > kXferRing) ? (n - kXferRing) : 0;
    for (usize i = start; i < n; ++i) {
        const Xfer& e = t_xfer[i % kXferRing];
        std::fprintf(stderr, "[xfer] #%-3zu %s 0x%016llx -> 0x%016llx  rsp=0x%llx rbp=0x%llx\n",
                     i, kNames[e.kind < 5 ? e.kind : 0],
                     (unsigned long long)e.from_rip, (unsigned long long)e.to_rip,
                     (unsigned long long)e.rsp, (unsigned long long)e.rbp);
    }
}

// RIP occupancy histogram, keyed by raw guest address. Enabled with PS4_RIP_HIST=1.
// The caller maps addresses back to module+offset; the executor does not know
// about module ranges.
static std::unordered_map<GuestAddr, u64> s_rip_hist;
static bool s_rip_hist_on = [] {
    const char* v = std::getenv("PS4_RIP_HIST");
    return v && *v && std::string_view(v) != "0";
}();

void dump_rip_histogram() {
    if (s_rip_hist.empty()) return;
    std::vector<std::pair<GuestAddr, u64>> v(s_rip_hist.begin(), s_rip_hist.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    u64 total = 0;
    for (const auto& e : v) total += e.second;
    std::fprintf(stderr, "[RIPHIST] distinct=%zu total=%llu\n", v.size(),
                 (unsigned long long)total);
    const usize show = v.size() < 40 ? v.size() : 40;
    for (usize i = 0; i < show; ++i)
        std::fprintf(stderr, "[RIPHIST] %6llu %5.2f%%  rip=0x%llx\n",
                     (unsigned long long)v[i].second, 100.0 * (double)v[i].second / (double)total,
                     (unsigned long long)v[i].first);
    // Also emit the raw map so an external script can attribute addresses.
    std::fprintf(stderr, "[RIPHIST] RAW\n");
    for (const auto& e : v)
        std::fprintf(stderr, "[RIPHIST] %llx %llu\n", (unsigned long long)e.first,
                     (unsigned long long)e.second);
}

StepResult CpuExecutor::step() {
    // Publish the executing RIP so a MemoryManager write-watch can name the
    // instruction responsible for a clobbered address.
    t_guest_rip = m_context.rip;
    t_guest_rsp = m_context.rsp;
    t_guest_rbp = m_context.rbp;
    t_guest_rbx = m_context.rbx;
    if (m_syscalls.has_exited()) {
        return StepResult::Exited;
    }

    if (m_context.rip == 0) {
        return StepResult::Halted;
    }

    thread_local GuestAddr s_rip_history[128] = {0};
    thread_local usize s_rip_hist_idx = 0;

    // Record RIP for debugging
    s_rip_history[s_rip_hist_idx++ % 128] = m_context.rip;

    if (m_context.rip < 0x20) {
        u64 ret_addr = 0;
        m_memory.read_value(m_context.rsp, ret_addr);
        std::fprintf(stderr, "[CPU FAULT] Invalid execution target RIP=0x%llx! Return address on stack: 0x%llx\n",
                     (unsigned long long)m_context.rip, (unsigned long long)ret_addr);
        std::fprintf(stderr, "[CPU FAULT] Recent RIP history (last 64 instructions):\n");
        for (usize i = 0; i < 64; ++i) {
            const GuestAddr h_rip = s_rip_history[(s_rip_hist_idx + 64 + i) % 128];
            if (h_rip != 0) {
                std::fprintf(stderr, "  [%02zu] RIP=0x%llx\n", i, (unsigned long long)h_rip);
            }
        }
        return StepResult::MemoryFault;
    }

    if (!m_memory.is_valid_address(m_context.rip, 1) || m_context.rip == 0) {
        GuestAddr ret = 0;
        m_memory.read_value(m_context.rsp, ret);
        std::fprintf(stderr, "[CPU FAULT] RIP=0x%llx, RSP=0x%llx (return=0x%llx)\n",
                     (unsigned long long)m_context.rip, (unsigned long long)m_context.rsp, (unsigned long long)ret);

        // Which loaded module, if any, contains the faulting address.
        GuestAddr moff = 0;
        const LoadedModule* mod = nullptr;
        if (DynamicLinker* lk = active_linker()) mod = lk->module_for_addr(m_context.rip, &moff);
        if (mod != nullptr) {
            std::fprintf(stderr, "[CPU FAULT] module handle=0x%x path=%s base=0x%llx "
                                 "module+0x%llx (span 0x%llx)\n",
                         mod->handle, mod->path.c_str(), (unsigned long long)mod->base,
                         (unsigned long long)moff, (unsigned long long)mod->size);
        } else {
            std::fprintf(stderr, "[CPU FAULT] module=<none: not inside any loaded module>\n");
        }

        // 64 bytes ending at RIP (the instruction stream that led here) plus 16
        // bytes AT the faulting RIP.
        if (m_context.rip >= 64) {
            u8 pre[64], at[16];
            if (m_memory.read_bytes(m_context.rip - 64, pre, sizeof(pre)) &&
                m_memory.read_bytes(m_context.rip, at, sizeof(at))) {
                std::fprintf(stderr, "[CPU FAULT] code 64 bytes BEFORE rip:\n  ");
                for (int i = 0; i < 64; i += 16) {
                    std::fprintf(stderr, "0x%016llx: ", (unsigned long long)(m_context.rip - 64 + i));
                    for (int j = 0; j < 16; ++j) std::fprintf(stderr, "%02x ", pre[i + j]);
                    std::fprintf(stderr, "\n  ");
                }
                std::fprintf(stderr, "\n[CPU FAULT] code AT rip:\n  ");
                for (int j = 0; j < 16; ++j) std::fprintf(stderr, "%02x ", at[j]);
                std::fprintf(stderr, "\n");
            }
        }

        // The operand that produced the bad target. RIP itself is usually
        // unreadable (it IS the bad address), so walk back to the call that got
        // us here: a `call rel32` at ret-5, then whatever is at its target --
        // for a PLT that is `jmp qword ptr [rip+disp]`, whose slot is the thing
        // that actually had to hold a real address.
        {
            GuestAddr moff2 = 0;
            const LoadedModule* caller = nullptr;
            if (DynamicLinker* lk = active_linker()) caller = lk->module_for_addr(ret, &moff2);
            std::fprintf(stderr, "[CPU FAULT] caller module=%s module+0x%llx\n",
                         caller ? caller->path.c_str() : "<none>", (unsigned long long)moff2);
            const GuestAddr site = ret - 5;
            u8 cb[5] = {0};
            if (m_memory.read_bytes(site, cb, sizeof(cb))) {
                std::fprintf(stderr, "[CPU FAULT] instruction before return: @0x%llx  %02x %02x %02x %02x %02x",
                             (unsigned long long)site, cb[0], cb[1], cb[2], cb[3], cb[4]);
                if (cb[0] == 0xE8) {  // call rel32
                    i32 d = 0;
                    std::memcpy(&d, cb + 1, 4);
                    const GuestAddr target = ret + d;
                    GuestAddr toff = 0;
                    const LoadedModule* tm = nullptr;
                    if (DynamicLinker* lk = active_linker()) tm = lk->module_for_addr(target, &toff);
                    std::fprintf(stderr, "   = CALL rel32 -> 0x%llx (%s module+0x%llx)\n",
                                 (unsigned long long)target,
                                 tm ? tm->path.c_str() : "<not in a module>",
                                 (unsigned long long)toff);
                    u8 tb[8] = {0};
                    if (m_memory.read_bytes(target, tb, sizeof(tb))) {
                        std::fprintf(stderr, "[CPU FAULT]   bytes at call target: %02x %02x %02x %02x %02x %02x %02x %02x",
                                     tb[0], tb[1], tb[2], tb[3], tb[4], tb[5], tb[6], tb[7]);
                        if (tb[0] == 0xFF && (tb[1] & 0xC7) == 0x05) {
                            i32 gd = 0;
                            std::memcpy(&gd, tb + 2, 4);
                            const GuestAddr slot = target + 6 + gd;
                            u64 sv = 0;
                            m_memory.read_value(slot, sv);
                            std::fprintf(stderr,
                                         "[CPU FAULT]   PLT jmp qword ptr [rip+0x%x] -> SLOT @0x%llx\n"
                                         "[CPU FAULT]   slot holds 0x%016llx  (u32 lo=0x%08x hi=0x%08x)\n",
                                         gd, (unsigned long long)slot,
                                         (unsigned long long)sv,
                                         (unsigned)(sv & 0xFFFFFFFF), (unsigned)(sv >> 32));
                            // Walk back from the PLT entry to the owning symbol table
                            // by dumping the raw bytes for offline disassembly.
                            if (DynamicLinker* lk = active_linker()) {
                                GuestAddr e2 = 0;
                                if (const LoadedModule* em = lk->module_for_addr(slot, &e2)) {
                                    std::fprintf(stderr, "[CPU FAULT]   slot is in %s at module+0x%llx\n",
                                                 em->path.c_str(), (unsigned long long)e2);
                                }
                            }
                        } else {
                            std::fprintf(stderr, "[CPU FAULT]   call target is not a PLT jmp [rip+disp]\n");
                        }
                    }
                } else if (cb[0] == 0xFF) {
                    std::fprintf(stderr, "   = indirect FF %02x (reg field=%d)\n", cb[1], (cb[1] >> 3) & 7);
                } else {
                    std::fprintf(stderr, "   = not a call rel32\n");
                }
            } else {
                std::fprintf(stderr, "[CPU FAULT] could not read instruction before return\n");
            }
        }

        // Call stack by walking rbp frames.
        std::fprintf(stderr, "[CPU FAULT] register set:\n");
        std::fprintf(stderr, "  rax=0x%llx rbx=0x%llx rcx=0x%llx rdx=0x%llx\n",
                     (unsigned long long)m_context.rax, (unsigned long long)m_context.rbx,
                     (unsigned long long)m_context.rcx, (unsigned long long)m_context.rdx);
        std::fprintf(stderr, "  rsi=0x%llx rdi=0x%llx rbp=0x%llx rsp=0x%llx\n",
                     (unsigned long long)m_context.rsi, (unsigned long long)m_context.rdi,
                     (unsigned long long)m_context.rbp, (unsigned long long)m_context.rsp);
        std::fprintf(stderr, "  r8 =0x%llx r9 =0x%llx r10=0x%llx r11=0x%llx\n",
                     (unsigned long long)m_context.r8, (unsigned long long)m_context.r9,
                     (unsigned long long)m_context.r10, (unsigned long long)m_context.r11);
        std::fprintf(stderr, "  r12=0x%llx r13=0x%llx r14=0x%llx r15=0x%llx\n",
                     (unsigned long long)m_context.r12, (unsigned long long)m_context.r13,
                     (unsigned long long)m_context.r14, (unsigned long long)m_context.r15);
        std::fprintf(stderr, "[CPU FAULT] stack walk from [RSP] (return addresses):\n");
        {
            GuestAddr sp = m_context.rsp;
            for (int i = 0; i < 24 && m_memory.is_valid_address(sp, 8); ++i, sp += 8) {
                GuestAddr word = 0;
                if (!m_memory.read_value(sp, word) || word == 0) break;
                GuestAddr o2 = 0;
                const LoadedModule* m2 = nullptr;
                if (DynamicLinker* lk = active_linker()) m2 = lk->module_for_addr(word, &o2);
                if (m2 != nullptr) {
                    std::fprintf(stderr, "  [rsp+0x%02x] 0x%016llx  module handle=0x%x module+0x%llx\n",
                                 (unsigned)(i * 8), (unsigned long long)word, m2->handle,
                                 (unsigned long long)o2);
                } else {
                    std::fprintf(stderr, "  [rsp+0x%02x] 0x%016llx\n",
                                 (unsigned)(i * 8), (unsigned long long)word);
                }
            }
        }
        std::fprintf(stderr, "Last RIPs:\n");
        for (int i = 1; i <= 10; ++i) {
            int idx = (s_rip_hist_idx - i + 128) % 128;
            std::fprintf(stderr, "  #%d: 0x%llx\n", i, (unsigned long long)s_rip_history[idx]);
        }
        dump_control_xfers(1);
        // Filter on the range the operator armed, not a leftover address. A
        // zero here means "watch not armed" and shows the whole ring.
        dump_write_watch_ring(write_watch_addr());
        return StepResult::MemoryFault;
    }

    const u8* raw_code = m_memory.to_host_ptr<u8>(m_context.rip);
    if (!raw_code) {
        return StepResult::MemoryFault;
    }

    const GuestAddr start_rip = m_context.rip;

    struct ExecHistoryEntry {
        GuestAddr rip;
        usize len;
        u8 bytes[8];
    };
    static thread_local ExecHistoryEntry s_ring_hist[32] = {};
    static thread_local usize s_ring_hist_idx = 0;

    if ((raw_code[0] == 0xDB && raw_code[1] == 0xC5) || raw_code[0] == 0xCC || raw_code[0] == 0xF4) {
        if (start_rip >= 0xf6d600 && start_rip <= 0xf6e000) {
            std::fprintf(stderr, "\n[FAULT HIT] Opcode %02x %02x at RIP=0x%llx (TID %u)\n",
                         raw_code[0], raw_code[1], (unsigned long long)start_rip, (unsigned)ps4::t_guest_tid);
            std::fprintf(stderr, "[FAULT HIT] Last 16 instructions executed:\n");
            for (int i = 16; i >= 1; --i) {
                usize idx = (s_ring_hist_idx + 32 - i) % 32;
                const auto& ent = s_ring_hist[idx];
                std::fprintf(stderr, "  [%02d] RIP=0x%llx (len=%zu): %02x %02x %02x %02x %02x %02x %02x %02x\n",
                             17 - i, (unsigned long long)ent.rip, ent.len,
                             ent.bytes[0], ent.bytes[1], ent.bytes[2], ent.bytes[3],
                             ent.bytes[4], ent.bytes[5], ent.bytes[6], ent.bytes[7]);
            }
            std::fprintf(stderr, "[FAULT HIT] Memory dump [RIP-48 .. RIP+32]:\n");
            for (GuestAddr addr = start_rip - 48; addr < start_rip + 32; addr += 16) {
                u8 b[16] = {0};
                m_memory.read_bytes(addr, b, 16);
                std::fprintf(stderr, "  0x%llx: %02x %02x %02x %02x %02x %02x %02x %02x  %02x %02x %02x %02x %02x %02x %02x %02x\n",
                             (unsigned long long)addr,
                             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                             b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
            }
            std::fflush(stderr);
        }
    }

    struct RingRecorder {
        GuestAddr rip;
        u8 bytes[8];
        const GuestAddr& end_rip;
        RingRecorder(GuestAddr r, const u8* c, const GuestAddr& er) : rip(r), end_rip(er) {
            std::memcpy(bytes, c, 8);
        }
        ~RingRecorder() {
            usize idx = s_ring_hist_idx++ % 32;
            s_ring_hist[idx].rip = rip;
            s_ring_hist[idx].len = (end_rip >= rip) ? (end_rip - rip) : 0;
            std::memcpy(s_ring_hist[idx].bytes, bytes, 8);
        }
    } ring_recorder(start_rip, raw_code, m_context.rip);
    // ---- one-shot tracer: PS4_TRACE_AT=<guest_addr> ------------------------
    // Prints registers and the chain walk at the first execution of <addr>.
    // PS4_TRACE_ITER=N also re-dumps after N further hits.
    {
        static const GuestAddr s_at = [&] {
            const char* v = std::getenv("PS4_TRACE_AT");
            return v ? static_cast<GuestAddr>(std::strtoull(v, nullptr, 16)) : 0;
        }();
        static const u64 s_after = [] {
            const char* v = std::getenv("PS4_TRACE_ITER");
            return v ? std::strtoull(v, nullptr, 10) : 0;
        }();
        static u64 s_hits = 0;
        if (s_at != 0 && start_rip == s_at && (s_hits <= s_after)) {
            const CpuContext& c = m_context;
            std::fprintf(stderr, "[TRACE] hit #%llu at rip=0x%llx\n",
                         (unsigned long long)s_hits, (unsigned long long)start_rip);
            std::fprintf(stderr,
                "[TRACE] rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n"
                "[TRACE] rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n"
                "[TRACE] r8 =%016llx r9 =%016llx r10=%016llx r11=%016llx\n"
                "[TRACE] r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n"
                "[TRACE] rip=%016llx rflags=%016llx\n",
                (unsigned long long)c.rax,(unsigned long long)c.rbx,(unsigned long long)c.rcx,
                (unsigned long long)c.rdx,(unsigned long long)c.rsi,(unsigned long long)c.rdi,
                (unsigned long long)c.rbp,(unsigned long long)c.rsp,(unsigned long long)c.r8,
                (unsigned long long)c.r9,(unsigned long long)c.r10,(unsigned long long)c.r11,
                (unsigned long long)c.r12,(unsigned long long)c.r13,(unsigned long long)c.r14,
                (unsigned long long)c.r15,(unsigned long long)c.rip,(unsigned long long)c.rflags);
            auto q = [&](GuestAddr a) { u64 v = 0; m_memory.read_bytes(a, &v, 8); return v; };
            std::fprintf(stderr, "[TRACE] sentinel[0x160246468]=0x%016llx  "
                         "bucket[0x160246470]=0x%016llx  bucket[0x160246468+rax]=0x%016llx\n",
                         (unsigned long long)q(0x160246468ULL),
                         (unsigned long long)q(0x160246470ULL),
                         (unsigned long long)q(0x160246468ULL + c.rax));
            auto dump = [&](const char* tag, GuestAddr a) {
                u8 b[64];
                if (!m_memory.read_bytes(a, b, sizeof(b))) {
                    std::fprintf(stderr, "[TRACE]   %-6s [0x%llx] unreadable\n", tag,
                                 (unsigned long long)a);
                    return;
                }
                std::fprintf(stderr, "[TRACE]   %-6s [0x%llx]", tag, (unsigned long long)a);
                for (int i = 0; i < 64; i += 8) {
                    u64 v = 0; std::memcpy(&v, b + i, 8);
                    std::fprintf(stderr, " %016llx", (unsigned long long)v);
                }
                std::fprintf(stderr, "\n");
            };
            dump("rdi", c.rdi); dump("rcx", c.rcx); dump("rax", c.rax);
            if (s_hits == 0) {
                // walk the chain: node -> [node+0x2018], comparing [node+0x2010] to r12
                u64 node = c.rcx, sent = c.rax, key = c.r12, steps = 0;
                std::fprintf(stderr, "[TRACE] chain: start rcx=0x%llx sentinel rax=0x%llx key r12=0x%llx\n",
                             (unsigned long long)node,(unsigned long long)sent,(unsigned long long)key);
                GuestAddr first = node;
                for (; steps < 100000; ++steps) {
                    u64 k = 0, nx = 0;
                    if (!m_memory.read_bytes(node + 0x2010, &k, 8) ||
                        !m_memory.read_bytes(node + 0x2018, &nx, 8)) {
                        std::fprintf(stderr, "[TRACE]   unreadable at node 0x%llx\n",
                                     (unsigned long long)node);
                        break;
                    }
                    if (steps < 10)
                        std::fprintf(stderr, "[TRACE]   node[%2llu] @0x%llx key[+0x2010]=0x%llx next[+0x2018]=0x%llx%s\n",
                                     (unsigned long long)steps,(unsigned long long)node,
                                     (unsigned long long)k,(unsigned long long)nx,
                                     (k == key ? "  <== KEY MATCH" : ""));
                    else if (nx == sent)
                        std::fprintf(stderr, "[TRACE]   reached sentinel after %llu steps\n",
                                     (unsigned long long)(steps + 1));
                    if (nx == sent) break;
                    if (nx == node) {
                        std::fprintf(stderr, "[TRACE]   SELF-LOOP at node 0x%llx after %llu steps\n",
                                     (unsigned long long)node,(unsigned long long)steps);
                        break;
                    }
                    if (steps >= 1 && nx == (node == first ? 0 : 0)) { /* noop */ }
                    node = nx;
                }
                std::fprintf(stderr, "[TRACE] chain walk ended at step %llu\n",
                             (unsigned long long)steps);
                // call stack via the frame pointer chain
                std::fprintf(stderr, "[TRACE] stack (rbp chain):\n");
                GuestAddr fp = c.rbp;
                for (int d = 0; d < 24 && fp != 0; ++d) {
                    u64 saved = 0, ret = 0;
                    if (!m_memory.read_bytes(fp, &saved, 8) || !m_memory.read_bytes(fp + 8, &ret, 8)) break;
                    std::fprintf(stderr, "[TRACE]   #%02d fp=0x%016llx ret=0x%016llx\n",
                                 d, (unsigned long long)fp, (unsigned long long)ret);
                    if (saved <= fp) break;
                    fp = saved;
                }
            }
            ++s_hits;
        }
    }
    // Off by default: 60 M hash lookups is not free. PS4_RIP_HIST=1 turns it on.
    if (s_rip_hist_on) ++s_rip_hist[start_rip];
    s_rip_history[s_rip_hist_idx++ % 128] = start_rip;
    if (jit_stats_enabled()) {
        jit_stats_record_interp(m_memory, start_rip);
    }
    u8 rex = 0;
    bool has_rex = false;
    bool operand_size_16 = false;
    [[maybe_unused]] bool lock_prefix = false;
    u8 rep_prefix = 0;
    u8 seg_override = 0;

    usize cursor = 0;
    while (cursor < 15 && m_memory.is_valid_address(start_rip + cursor, 1)) {
        const u8 b = raw_code[cursor];
        if (b == 0x66) { operand_size_16 = true; cursor++; }
        else if (b == 0xF0) { lock_prefix = true; cursor++; }
        else if (b == 0xF2 || b == 0xF3) { rep_prefix = b; cursor++; }
        else if (b == 0x64 || b == 0x65) { seg_override = b; cursor++; }
        else if (b == 0x2E || b == 0x3E || b == 0x26 || b == 0x36 || b == 0x67) { cursor++; }
        else if ((b & 0xF0) == 0x40) { rex = b; has_rex = true; cursor++; break; }
        else break;
    }

    const u8 rex_w = (rex >> 3) & 1;
    const u8 rex_r = (rex >> 2) & 1;
    const u8 rex_x = (rex >> 1) & 1;
    const u8 rex_b = rex & 1;
    const usize op_bits = rex_w ? 64 : (operand_size_16 ? 16 : 32);

    if (!m_memory.is_valid_address(start_rip + cursor, 1)) {
        return StepResult::MemoryFault;
    }

    const u8 op0 = raw_code[cursor++];
    const GuestAddr next_ip_base = start_rip + cursor;

    // Helper lambda to read R/M value
    auto read_rm = [&](const ModRmResult& m, usize bits) -> u64 {
        if (m.is_reg) {
            if (bits == 64) return get_gpr64(m_context, m.rm_id);
            if (bits == 32) return static_cast<u32>(get_gpr64(m_context, m.rm_id));
            if (bits == 16) return static_cast<u16>(get_gpr64(m_context, m.rm_id));
            return get_gpr8(m_context, m.rm_id, has_rex);
        }
        if (bits == 64) {
            if ((m.mem_addr & 7) == 0) {
                const auto* p = m_memory.to_host_ptr<u64>(m.mem_addr);
                if (p) return __atomic_load_n(p, __ATOMIC_ACQUIRE);
            }
            u64 v = 0; m_memory.read_value(m.mem_addr, v); return v;
        }
        if (bits == 32) {
            if ((m.mem_addr & 3) == 0) {
                const auto* p = m_memory.to_host_ptr<u32>(m.mem_addr);
                if (p) return __atomic_load_n(p, __ATOMIC_ACQUIRE);
            }
            u32 v = 0; m_memory.read_value(m.mem_addr, v); return v;
        }
        if (bits == 16) {
            if ((m.mem_addr & 1) == 0) {
                const auto* p = m_memory.to_host_ptr<u16>(m.mem_addr);
                if (p) return __atomic_load_n(p, __ATOMIC_ACQUIRE);
            }
            u16 v = 0; m_memory.read_value(m.mem_addr, v); return v;
        }
        const auto* p = m_memory.to_host_ptr<u8>(m.mem_addr);
        if (p) return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        u8 v = 0; m_memory.read_value(m.mem_addr, v); return v;
    };

    // Helper lambda to write R/M value
    auto write_rm = [&](const ModRmResult& m, u64 val, usize bits) {
        if (m.is_reg) {
            if (bits == 64) set_gpr64(m_context, m.rm_id, val);
            else if (bits == 32) set_gpr32(m_context, m.rm_id, static_cast<u32>(val));
            else if (bits == 16) set_gpr16(m_context, m.rm_id, static_cast<u16>(val));
            else set_gpr8(m_context, m.rm_id, has_rex, static_cast<u8>(val));
            return;
        }
        if (m.mem_addr <= 0x10) {
            std::fprintf(stderr, "[WRITE_ZERO] RIP=0x%llx addr=0x%llx val=0x%llx bits=%zu\n",
                         (unsigned long long)start_rip, (unsigned long long)m.mem_addr, (unsigned long long)val, bits);
        }
        // The aligned fast paths below store through a host pointer and would
        // bypass an armed write-watch; take the recorded path when it overlaps.
        if (const GuestAddr wl = write_watch_addr();
            wl != 0 && m.mem_addr < write_watch_end() && wl < m.mem_addr + bits / 8) {
            if (bits == 64) m_memory.write_value(m.mem_addr, val);
            else if (bits == 32) m_memory.write_value(m.mem_addr, static_cast<u32>(val));
            else if (bits == 16) m_memory.write_value(m.mem_addr, static_cast<u16>(val));
            else m_memory.write_value(m.mem_addr, static_cast<u8>(val));
            return;
        }
        if (bits == 64) {
            if ((m.mem_addr & 7) == 0) {
                auto* p = m_memory.to_host_ptr<u64>(m.mem_addr);
                if (p) { __atomic_store_n(p, val, __ATOMIC_RELEASE); return; }
            }
            m_memory.write_value(m.mem_addr, val);
            return;
        }
        if (bits == 32) {
            if ((m.mem_addr & 3) == 0) {
                auto* p = m_memory.to_host_ptr<u32>(m.mem_addr);
                if (p) { __atomic_store_n(p, static_cast<u32>(val), __ATOMIC_RELEASE); return; }
            }
            m_memory.write_value(m.mem_addr, static_cast<u32>(val));
            return;
        }
        if (bits == 16) {
            if ((m.mem_addr & 1) == 0) {
                auto* p = m_memory.to_host_ptr<u16>(m.mem_addr);
                if (p) { __atomic_store_n(p, static_cast<u16>(val), __ATOMIC_RELEASE); return; }
            }
            m_memory.write_value(m.mem_addr, static_cast<u16>(val));
            return;
        }
        auto* p = m_memory.to_host_ptr<u8>(m.mem_addr);
        if (p) { __atomic_store_n(p, static_cast<u8>(val), __ATOMIC_RELEASE); return; }
        m_memory.write_value(m.mem_addr, static_cast<u8>(val));
    };

    // 0x90: NOP
    if (op0 == 0x90) {
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0xF4: HLT
    if (op0 == 0xF4) {
        return StepResult::Halted;
    }

    // 0x0F 0x05: SYSCALL
    if (op0 == 0x0F && m_memory.is_valid_address(start_rip + cursor, 1) && raw_code[cursor] == 0x05) {
        SyscallContext sctx(m_memory);
        sctx.cpu     = &m_context;
        sctx.args[0] = m_context.rdi;
        sctx.args[1] = m_context.rsi;
        sctx.args[2] = m_context.rdx;
        sctx.args[3] = m_context.r10;
        sctx.args[4] = m_context.r8;
        sctx.args[5] = m_context.r9;
        sctx.rcx     = m_context.rcx;
        sctx.rax     = m_context.rax;
        sctx.rsp     = m_context.rsp;
        sctx.rbp     = m_context.rbp;

        record_xfer(start_rip, 0, m_context.rsp, m_context.rbp, 3);
        m_syscalls.dispatch(static_cast<u32>(m_context.rax), sctx);
        if (!sctx.cpu_rip_modified) {
            m_context.rax = sctx.rax;
            m_context.rip = start_rip + cursor + 1;
        }

        if (m_syscalls.has_exited()) {
            return StepResult::Exited;
        }
        return StepResult::Syscall;
    }

    // 0x50 - 0x57: PUSH reg64
    if (op0 >= 0x50 && op0 <= 0x57) {
        const u8 reg = (rex_b ? 8 : 0) | (op0 - 0x50);
        m_context.rsp -= 8;
        if (!m_memory.write_value(m_context.rsp, get_gpr64(m_context, reg))) {
            return StepResult::MemoryFault;
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0x58 - 0x5F: POP reg64
    if (op0 >= 0x58 && op0 <= 0x5F) {
        const u8 reg = (rex_b ? 8 : 0) | (op0 - 0x58);
        u64 val = 0;
        if (!m_memory.read_value(m_context.rsp, val)) {
            return StepResult::MemoryFault;
        }
        m_context.rsp += 8;
        set_gpr64(m_context, reg, val);
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0x68: PUSH imm32 (sign-extended)
    if (op0 == 0x68 && m_memory.is_valid_address(start_rip + cursor, 4)) {
        i32 imm32 = 0;
        std::memcpy(&imm32, raw_code + cursor, 4);
        m_context.rsp -= 8;
        m_memory.write_value(m_context.rsp, static_cast<u64>(static_cast<i64>(imm32)));
        m_context.rip = start_rip + cursor + 4;
        return StepResult::Ok;
    }

    // 0x6A: PUSH imm8 (sign-extended)
    if (op0 == 0x6A && m_memory.is_valid_address(start_rip + cursor, 1)) {
        i8 imm8 = static_cast<i8>(raw_code[cursor]);
        m_context.rsp -= 8;
        m_memory.write_value(m_context.rsp, static_cast<u64>(static_cast<i64>(imm8)));
        m_context.rip = start_rip + cursor + 1;
        return StepResult::Ok;
    }

    // 0xC9: LEAVE (mov rsp, rbp; pop rbp)
    if (op0 == 0xC9) {
        m_context.rsp = m_context.rbp;
        u64 val = 0;
        if (!m_memory.read_value(m_context.rsp, val)) return StepResult::MemoryFault;
        m_context.rsp += 8;
        m_context.rbp = val;
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0x89: MOV r/m, reg
    if (op0 == 0x89) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        const u64 src_val = (op_bits == 64) ? get_gpr64(m_context, m.reg_id) :
                            (op_bits == 32) ? static_cast<u32>(get_gpr64(m_context, m.reg_id)) :
                            static_cast<u16>(get_gpr64(m_context, m.reg_id));
        write_rm(m, src_val, op_bits);
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0x8B: MOV reg, r/m
    if (op0 == 0x8B) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        const u64 val = read_rm(m, op_bits);
        if (op_bits == 64) set_gpr64(m_context, m.reg_id, val);
        else if (op_bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(val));
        else set_gpr16(m_context, m.reg_id, static_cast<u16>(val));
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0x88: MOV r/m8, reg8
    if (op0 == 0x88) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        write_rm(m, get_gpr8(m_context, m.reg_id, has_rex), 8);
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0x8A: MOV reg8, r/m8
    if (op0 == 0x8A) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        set_gpr8(m_context, m.reg_id, has_rex, static_cast<u8>(read_rm(m, 8)));
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0x8D: LEA reg, [m]
    if (op0 == 0x8D) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid || m.is_reg) return StepResult::IllegalInstruction;
        if (op_bits == 64) set_gpr64(m_context, m.reg_id, m.mem_addr);
        else if (op_bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(m.mem_addr));
        else set_gpr16(m_context, m.reg_id, static_cast<u16>(m.mem_addr));
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0xC6: MOV r/m8, imm8
    if (op0 == 0xC6) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context, 1);
        if (!m.valid || (m.reg_id & 7) != 0) return StepResult::IllegalInstruction;
        const u8 imm8 = raw_code[cursor + m.bytes_consumed];
        write_rm(m, imm8, 8);
        m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
        return StepResult::Ok;
    }

    // 0xC7: MOV r/m, imm16 / imm32 (sign-extended if 64-bit)
    if (op0 == 0xC7) {
        const usize imm_bytes = (op_bits == 16) ? 2 : 4;
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context, imm_bytes);
        if (!m.valid || (m.reg_id & 7) != 0) return StepResult::IllegalInstruction;
        u64 val = 0;
        if (op_bits == 16) {
            u16 imm16 = 0;
            std::memcpy(&imm16, raw_code + cursor + m.bytes_consumed, 2);
            val = imm16;
        } else {
            i32 imm32 = 0;
            std::memcpy(&imm32, raw_code + cursor + m.bytes_consumed, 4);
            val = (op_bits == 64) ? static_cast<u64>(static_cast<i64>(imm32)) : static_cast<u32>(imm32);
        }
        write_rm(m, val, op_bits);
        m_context.rip = start_rip + cursor + m.bytes_consumed + imm_bytes;
        return StepResult::Ok;
    }

    // 0xB8 - 0xBF: MOV reg, imm
    if (op0 >= 0xB8 && op0 <= 0xBF) {
        const u8 reg = (rex_b ? 8 : 0) | (op0 - 0xB8);
        if (rex_w) {
            u64 imm64 = 0;
            std::memcpy(&imm64, raw_code + cursor, 8);
            set_gpr64(m_context, reg, imm64);
            m_context.rip = start_rip + cursor + 8;
        } else if (operand_size_16) {
            u16 imm16 = 0;
            std::memcpy(&imm16, raw_code + cursor, 2);
            set_gpr16(m_context, reg, imm16);
            m_context.rip = start_rip + cursor + 2;
        } else {
            u32 imm32 = 0;
            std::memcpy(&imm32, raw_code + cursor, 4);
            set_gpr32(m_context, reg, imm32);
            m_context.rip = start_rip + cursor + 4;
        }
        return StepResult::Ok;
    }

    // 0xB0 - 0xB7: MOV reg8, imm8
    if (op0 >= 0xB0 && op0 <= 0xB7) {
        const u8 reg = (rex_b ? 8 : 0) | (op0 - 0xB0);
        const u8 imm8 = raw_code[cursor];
        set_gpr8(m_context, reg, has_rex, imm8);
        m_context.rip = start_rip + cursor + 1;
        return StepResult::Ok;
    }

    // 0xC5 / 0xC4: VEX prefix (AVX instruction set)
    if (op0 == 0xC5 || op0 == 0xC4) {
        u8 vex_r = 0, vex_x = 0, vex_b = 0, vex_w = 0;
        u8 vex_vvvv = 0, vex_pp = 0;
        u8 vex_map = 1;
        bool vex_l = false;


        if (op0 == 0xC5) {
            if (cursor + 1 >= 15) return StepResult::IllegalInstruction;
            const u8 b1 = raw_code[cursor++];
            vex_r = (b1 & 0x80) ? 0 : 1;
            vex_x = 0;
            vex_b = 0;
            vex_vvvv = (~(b1 >> 3)) & 0x0F;
            vex_l = (b1 & 4) != 0;
            vex_pp = b1 & 3;
            vex_map = 1;
        } else {
            if (cursor + 2 >= 15) return StepResult::IllegalInstruction;
            const u8 b1 = raw_code[cursor++];
            const u8 b2 = raw_code[cursor++];
            vex_r = (b1 & 0x80) ? 0 : 1;
            vex_x = (b1 & 0x40) ? 0 : 1;
            vex_b = (b1 & 0x20) ? 0 : 1;
            vex_map = b1 & 0x1F;
            vex_w = (b2 >> 7) & 1;
            vex_vvvv = (~(b2 >> 3)) & 0x0F;
            vex_l = (b2 & 4) != 0;
            vex_pp = b2 & 3;
        }

        if (cursor >= 15) return StepResult::IllegalInstruction;
        g_vex_L = vex_l;
        g_vex_map = vex_map;
        g_vex_r = vex_r; g_vex_x = vex_x; g_vex_b = vex_b; g_vex_vvvv = vex_vvvv;
        const u8 vex_op = raw_code[cursor++];
        g_vex_op = vex_op;
        const GuestAddr next_ip_base_vex = start_rip + cursor;

        // VMOVLPS store (VEX.128.0F 13 /r, pp=0): copy the low 64 bits of the
        // ModR/M.reg XMM register to memory.
        // VMOVLPD store (VEX.128.0F 13 /r, pp=1): copy the low 64 bits of the
        // ModR/M.reg XMM register to memory (double-precision variant).
        // VEX.L=1 is invalid for both. vvvv is ignored (reserved).
        if (vex_map == 1 && vex_op == 0x13) {
            if (vex_l) return StepResult::IllegalInstruction;
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex,
                                  vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid || m.is_reg) return StepResult::IllegalInstruction;
            m_memory.write_value(m.mem_addr, m_context.ymm[m.reg_id & 15].q[0]);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VMOVUPS / VMOVUPD / VMOVSS / VMOVSD (0x10 load, 0x11 store)
        if (vex_op == 0x10) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            if (vex_pp == 2) { // VMOVSS (32-bit float)
                if (m.is_reg) {
                    m_context.ymm[dest].d[0] = m_context.ymm[m.rm_id & 15].d[0];
                } else {
                    u32 val = 0;
                    m_memory.read_value(m.mem_addr, val);
                    m_context.ymm[dest] = CpuContext::YmmReg{};
                    m_context.ymm[dest].d[0] = val;
                }
            } else if (vex_pp == 3) { // VMOVSD (64-bit float)
                if (m.is_reg) {
                    m_context.ymm[dest].q[0] = m_context.ymm[m.rm_id & 15].q[0];
                } else {
                    u64 val = 0;
                    m_memory.read_value(m.mem_addr, val);
                    m_context.ymm[dest] = CpuContext::YmmReg{};
                    m_context.ymm[dest].q[0] = val;
                }
            } else { // VMOVUPS / VMOVUPD (128-bit or 256-bit unaligned)
                const usize bytes = vex_l ? 32 : 16;
                if (m.is_reg) {
                    std::memcpy(&m_context.ymm[dest], &m_context.ymm[m.rm_id & 15], bytes);
                } else {
                    m_memory.read_bytes(m.mem_addr, &m_context.ymm[dest], bytes);
                }
                if (!vex_l) {
                    m_context.ymm[dest].q[2] = 0;
                    m_context.ymm[dest].q[3] = 0;
                }
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        if (vex_op == 0x11) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 src = m.reg_id & 15;
            if (vex_pp == 2) { // VMOVSS
                if (m.is_reg) {
                    m_context.ymm[m.rm_id & 15].d[0] = m_context.ymm[src].d[0];
                } else {
                    m_memory.write_value(m.mem_addr, m_context.ymm[src].d[0]);
                }
            } else if (vex_pp == 3) { // VMOVSD
                if (m.is_reg) {
                    m_context.ymm[m.rm_id & 15].q[0] = m_context.ymm[src].q[0];
                } else {
                    m_memory.write_value(m.mem_addr, m_context.ymm[src].q[0]);
                }
            } else { // VMOVUPS / VMOVUPD
                const usize bytes = vex_l ? 32 : 16;
                if (m.is_reg) {
                    std::memcpy(&m_context.ymm[m.rm_id & 15], &m_context.ymm[src], bytes);
                } else {
                    m_memory.write_bytes(m.mem_addr, &m_context.ymm[src], bytes);
                }
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VXORPS (0x57) / VXORPD
        if (vex_op == 0x57) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2_val{};
            if (m.is_reg) {
                src2_val = m_context.ymm[m.rm_id & 15];
            } else {
                m_memory.read_bytes(m.mem_addr, &src2_val, vex_l ? 32 : 16);
            }
            const usize num_qwords = vex_l ? 4 : 2;
            for (usize q = 0; q < num_qwords; ++q) {
                m_context.ymm[dest].q[q] = m_context.ymm[src1].q[q] ^ src2_val.q[q];
            }
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VMOVDQA / VMOVDQU (0x6F load, 0x7F store)
        if (vex_op == 0x6F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const usize bytes = vex_l ? 32 : 16;
            if (m.is_reg) {
                std::memcpy(&m_context.ymm[dest], &m_context.ymm[m.rm_id & 15], bytes);
            } else {
                m_memory.read_bytes(m.mem_addr, &m_context.ymm[dest], bytes);
            }
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        if (vex_op == 0x7F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 src = m.reg_id & 15;
            const usize bytes = vex_l ? 32 : 16;
            if (m.is_reg) {
                std::memcpy(&m_context.ymm[m.rm_id & 15], &m_context.ymm[src], bytes);
            } else {
                m_memory.write_bytes(m.mem_addr, &m_context.ymm[src], bytes);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPSHUFD (0x70)
        if (vex_op == 0x70) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            CpuContext::YmmReg src{};
            if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);

            auto shuffle_dwords = [&](u32* dst_d, const u32* src_d) {
                dst_d[0] = src_d[(imm8 >> 0) & 3];
                dst_d[1] = src_d[(imm8 >> 2) & 3];
                dst_d[2] = src_d[(imm8 >> 4) & 3];
                dst_d[3] = src_d[(imm8 >> 6) & 3];
            };
            shuffle_dwords(m_context.ymm[dest].d, src.d);
            if (vex_l) {
                shuffle_dwords(m_context.ymm[dest].d + 4, src.d + 4);
            } else {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VLDMXCSR / VSTMXCSR m32 (map 1, op 0xAE /2 /3)
        if (vex_map == 1 && vex_op == 0xAE) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (m.valid && !m.is_reg && ((m.reg_id & 7) == 2 || (m.reg_id & 7) == 3)) {
                const bool ok = (m.reg_id & 7) == 2 ? m_memory.read_value(m.mem_addr, m_context.mxcsr)
                                                    : m_memory.write_value(m.mem_addr, m_context.mxcsr);
                if (!ok) return StepResult::MemoryFault;
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }
        }

        // VEXTRACTPS r/m32, xmm, imm8 (map 3, 66, op 0x17)
        if (vex_map == 3 && vex_op == 0x17 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm = raw_code[cursor + m.bytes_consumed];
            const u32 val = m_context.ymm[m.reg_id & 15].d[imm & 3];
            if (m.is_reg) {
                set_gpr64(m_context, m.rm_id, val); // 32-bit GPR write zero-extends
            } else if (!m_memory.write_value(m.mem_addr, val)) {
                return StepResult::MemoryFault;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPHADDW / VPHADDD (map 2, 66, op 0x01 / 0x02): pairwise horizontal
        // add within each 128-bit lane; low half from src1, high half from src2.
        if (vex_map == 2 && (vex_op == 0x01 || vex_op == 0x02) && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const CpuContext::YmmReg a = m_context.ymm[vex_vvvv & 15];
            CpuContext::YmmReg b{};
            if (m.is_reg) b = m_context.ymm[m.rm_id & 15];
            else if (!m_memory.read_bytes(m.mem_addr, &b, vex_l ? 32 : 16)) return StepResult::MemoryFault;
            CpuContext::YmmReg res{};
            for (usize lane = 0; lane < (vex_l ? 2u : 1u); ++lane) {
                if (vex_op == 0x02) {
                    const usize o = lane * 4;
                    for (usize i = 0; i < 2; ++i) {
                        res.d[o + i] = a.d[o + 2 * i] + a.d[o + 2 * i + 1];
                        res.d[o + 2 + i] = b.d[o + 2 * i] + b.d[o + 2 * i + 1];
                    }
                } else {
                    const usize o = lane * 8;
                    for (usize i = 0; i < 4; ++i) {
                        res.w[o + i] = static_cast<u16>(a.w[o + 2 * i] + a.w[o + 2 * i + 1]);
                        res.w[o + 4 + i] = static_cast<u16>(b.w[o + 2 * i] + b.w[o + 2 * i + 1]);
                    }
                }
            }
            m_context.ymm[dest] = res; // VEX.128 zeroes bits 255:128 (res is zero there)
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPSHUFB (map 2, op 0x00)
        if (vex_map == 2 && vex_op == 0x00) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg mask_reg{};
            const usize bytes = vex_l ? 32 : 16;
            if (m.is_reg) mask_reg = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &mask_reg, bytes);

            CpuContext::YmmReg orig = m_context.ymm[src1];
            CpuContext::YmmReg res{};
            for (usize lane = 0; lane < (vex_l ? 2 : 1); ++lane) {
                const usize base = lane * 16;
                for (usize i = 0; i < 16; ++i) {
                    const u8 m_byte = mask_reg.b[base + i];
                    if (m_byte & 0x80) {
                        res.b[base + i] = 0;
                    } else {
                        res.b[base + i] = orig.b[base + (m_byte & 0x0F)];
                    }
                }
            }
            m_context.ymm[dest] = res;
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // BEXTR / SHLX / SARX / SHRX (map 2, op 0xF7)
        if (vex_map == 2 && vex_op == 0xF7) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const usize bits = vex_w ? 64 : 32;
            const u64 rm_val = read_rm(m, bits);
            const u8 vvvv = vex_vvvv & 15;
            if (vex_pp == 0) { // BEXTR dst, rm, control(vvvv)
                const u64 ctrl = vex_w ? get_gpr64(m_context, vvvv) : static_cast<u32>(get_gpr64(m_context, vvvv));
                const u64 start = ctrl & 0xFF;
                const u64 len = (ctrl >> 8) & 0xFF;
                u64 res = 0;
                if (start < bits) {
                    const u64 temp = rm_val >> start;
                    const u64 mask = (len >= bits) ? ~0ULL : ((1ULL << len) - 1);
                    res = temp & mask;
                }
                if (bits == 64) set_gpr64(m_context, m.reg_id, res);
                else set_gpr32(m_context, m.reg_id, static_cast<u32>(res));
                m_context.set_flag(CpuContext::FLAG_ZF, res == 0);
                m_context.set_flag(CpuContext::FLAG_CF, false);
                m_context.set_flag(CpuContext::FLAG_OF, false);
            } else if (vex_pp == 1) { // SHLX dst, rm, src(vvvv)
                const u64 shift = (get_gpr64(m_context, vvvv) & (bits - 1));
                const u64 res = (bits == 64) ? (rm_val << shift) : (static_cast<u32>(rm_val) << shift);
                if (bits == 64) set_gpr64(m_context, m.reg_id, res);
                else set_gpr32(m_context, m.reg_id, static_cast<u32>(res));
            } else if (vex_pp == 2) { // SARX dst, rm, src(vvvv)
                const u64 shift = (get_gpr64(m_context, vvvv) & (bits - 1));
                const u64 res = (bits == 64) ? static_cast<u64>(static_cast<i64>(rm_val) >> shift) :
                                               static_cast<u32>(static_cast<i32>(rm_val) >> shift);
                if (bits == 64) set_gpr64(m_context, m.reg_id, res);
                else set_gpr32(m_context, m.reg_id, static_cast<u32>(res));
            } else { // SHRX dst, rm, src(vvvv)
                const u64 shift = (get_gpr64(m_context, vvvv) & (bits - 1));
                const u64 res = (bits == 64) ? (rm_val >> shift) : (static_cast<u32>(rm_val) >> shift);
                if (bits == 64) set_gpr64(m_context, m.reg_id, res);
                else set_gpr32(m_context, m.reg_id, static_cast<u32>(res));
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // ANDN (map 2, op 0xF2)
        if (vex_map == 2 && vex_op == 0xF2) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const usize bits = vex_w ? 64 : 32;
            const u64 rm_val = read_rm(m, bits);
            const u8 vvvv = vex_vvvv & 15;
            const u64 v_val = vex_w ? get_gpr64(m_context, vvvv) : static_cast<u32>(get_gpr64(m_context, vvvv));
            const u64 res = (~v_val) & rm_val;
            if (bits == 64) {
                set_gpr64(m_context, m.reg_id, res);
                m_context.set_flag(CpuContext::FLAG_SF, static_cast<i64>(res) < 0);
            } else {
                set_gpr32(m_context, m.reg_id, static_cast<u32>(res));
                m_context.set_flag(CpuContext::FLAG_SF, static_cast<i32>(res) < 0);
            }
            m_context.set_flag(CpuContext::FLAG_ZF, res == 0);
            m_context.set_flag(CpuContext::FLAG_CF, false);
            m_context.set_flag(CpuContext::FLAG_OF, false);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // BLSR / BLSMSK / BLSI (map 2, op 0xF3)
        if (vex_map == 2 && vex_op == 0xF3) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const usize bits = vex_w ? 64 : 32;
            const u64 src = read_rm(m, bits);
            const u8 sub_op = m.reg_id & 7;
            u64 res = 0;
            if (sub_op == 1) res = (src - 1) & src;      // BLSR
            else if (sub_op == 2) res = (src - 1) ^ src; // BLSMSK
            else if (sub_op == 3) res = (-src) & src;     // BLSI
            else return StepResult::IllegalInstruction;

            const u8 dst = vex_vvvv & 15;
            if (bits == 64) {
                set_gpr64(m_context, dst, res);
                m_context.set_flag(CpuContext::FLAG_SF, static_cast<i64>(res) < 0);
            } else {
                set_gpr32(m_context, dst, static_cast<u32>(res));
                m_context.set_flag(CpuContext::FLAG_SF, static_cast<i32>(res) < 0);
            }
            m_context.set_flag(CpuContext::FLAG_CF, src == 0);
            m_context.set_flag(CpuContext::FLAG_ZF, res == 0);
            m_context.set_flag(CpuContext::FLAG_OF, false);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VBROADCASTSS (map 2, op 0x18)
        if (vex_map == 2 && vex_op == 0x18) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            u32 val = 0;
            if (m.is_reg) val = m_context.ymm[m.rm_id & 15].d[0];
            else m_memory.read_value(m.mem_addr, val);
            const usize count = vex_l ? 8 : 4;
            for (usize i = 0; i < count; ++i) m_context.ymm[dest].d[i] = val;
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VBROADCASTSD / VBROADCASTF128 / VPBROADCASTD / VPBROADCASTQ.
        //
        // These belong HERE, not under the legacy `op1 == 0x38` escape: a VEX
        // instruction carries no 0F escape byte, so `op1` is the opcode itself
        // and the 0F38 block is never entered. vex_op already consumed the
        // opcode byte, so `cursor` points straight at the ModR/M -- reading
        // another raw_code[cursor++] here would decode the first disp32 byte as
        // ModR/M.
        //
        // mono-ps4+0x120d9d is `vbroadcastsd ymm0, QWORD PTR [r14+0xfce8]`,
        // used to initialise a 1024-entry bucket table via four 32-byte
        // vmovups. With it missing, ymm0 kept stale lanes and the allocator's
        // empty-list guard (bucket head == sentinel) never matched.
        if (vex_map == 2 && (vex_op == 0x19 || vex_op == 0x59 ||
                             vex_op == 0x1A || vex_op == 0x58)) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex,
                                  vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid || m.is_reg) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const usize nbytes = (vex_op == 0x58) ? 4 : (vex_op == 0x1A) ? 16 : 8;
            CpuContext::YmmReg src{};
            if (!m_memory.read_bytes(m.mem_addr, src.b, nbytes)) return StepResult::MemoryFault;
            CpuContext::YmmReg res{};
            if (vex_op == 0x58) {
                u32 v = 0; std::memcpy(&v, src.b, 4);
                const int n = vex_l ? 8 : 4;
                for (int i = 0; i < n; ++i) res.d[i] = v;
            } else if (vex_op == 0x1A) {
                std::memcpy(&res.b[0], src.b, 16);
                if (vex_l) std::memcpy(&res.b[16], src.b, 16);
            } else { // 0x19 / 0x59 VBROADCASTSD, VPBROADCASTQ
                u64 v = 0; std::memcpy(&v, src.b, 8);
                const int n = vex_l ? 4 : 2;
                for (int i = 0; i < n; ++i) std::memcpy(&res.b[i * 8], &v, 8);
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPMULLD (map 2, op 0x40): multiply packed signed 32-bit integers
        // VEX.128/256.66.0F38.WIG 40 /r -- NO imm8
        if (vex_map == 2 && vex_op == 0x40) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            const usize count = vex_l ? 8 : 4;
            for (usize i = 0; i < count; ++i) {
                const i32 a = static_cast<i32>(m_context.ymm[src1].d[i]);
                const i32 b = static_cast<i32>(src2.d[i]);
                const i64 prod = static_cast<i64>(a) * static_cast<i64>(b);
                m_context.ymm[dest].d[i] = static_cast<u32>(prod);
            }
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPMOVZX (map 2, op 0x20..0x25) / VPMOVSX (map 2, op 0x30..0x35)
        // 0x20/0x30: BW (byte to word)
        // 0x21/0x31: BD (byte to dword)
        // 0x22/0x32: BQ (byte to qword)
        // 0x23/0x33: WD (word to dword)
        // 0x24/0x34: WQ (word to qword)
        // 0x25/0x35: DQ (dword to qword)
        if (vex_map == 2 && ((vex_op >= 0x20 && vex_op <= 0x25) || (vex_op >= 0x30 && vex_op <= 0x35)) && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const bool is_signed = (vex_op >= 0x30);
            const u8 op_type = vex_op & 0x0F;

            usize src_bytes = 0;
            switch (op_type) {
                case 0: src_bytes = vex_l ? 16 : 8; break;
                case 1: src_bytes = vex_l ? 8 : 4; break;
                case 2: src_bytes = vex_l ? 4 : 2; break;
                case 3: src_bytes = vex_l ? 16 : 8; break;
                case 4: src_bytes = vex_l ? 8 : 4; break;
                case 5: src_bytes = vex_l ? 16 : 8; break;
            }

            CpuContext::YmmReg src{};
            if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src, src_bytes);

            CpuContext::YmmReg res{};
            switch (op_type) {
                case 0: { // BW: byte to word
                    const usize count = vex_l ? 16 : 8;
                    for (usize i = 0; i < count; ++i) {
                        res.w[i] = is_signed ? static_cast<u16>(static_cast<i16>(static_cast<i8>(src.b[i])))
                                             : static_cast<u16>(src.b[i]);
                    }
                    break;
                }
                case 1: { // BD: byte to dword
                    const usize count = vex_l ? 8 : 4;
                    for (usize i = 0; i < count; ++i) {
                        res.d[i] = is_signed ? static_cast<u32>(static_cast<i32>(static_cast<i8>(src.b[i])))
                                             : static_cast<u32>(src.b[i]);
                    }
                    break;
                }
                case 2: { // BQ: byte to qword
                    const usize count = vex_l ? 4 : 2;
                    for (usize i = 0; i < count; ++i) {
                        res.q[i] = is_signed ? static_cast<u64>(static_cast<i64>(static_cast<i8>(src.b[i])))
                                             : static_cast<u64>(src.b[i]);
                    }
                    break;
                }
                case 3: { // WD: word to dword
                    const usize count = vex_l ? 8 : 4;
                    for (usize i = 0; i < count; ++i) {
                        res.d[i] = is_signed ? static_cast<u32>(static_cast<i32>(static_cast<i16>(src.w[i])))
                                             : static_cast<u32>(src.w[i]);
                    }
                    break;
                }
                case 4: { // WQ: word to qword
                    const usize count = vex_l ? 4 : 2;
                    for (usize i = 0; i < count; ++i) {
                        res.q[i] = is_signed ? static_cast<u64>(static_cast<i64>(static_cast<i16>(src.w[i])))
                                             : static_cast<u64>(src.w[i]);
                    }
                    break;
                }
                case 5: { // DQ: dword to qword
                    const usize count = vex_l ? 4 : 2;
                    for (usize i = 0; i < count; ++i) {
                        res.q[i] = is_signed ? static_cast<u64>(static_cast<i64>(static_cast<i32>(src.d[i])))
                                             : static_cast<u64>(src.d[i]);
                    }
                    break;
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPTEST (map 2, op 0x17): packed bit test
        // VEX.128/256.66.0F38.W0 17 /r -- NO imm8
        // Sets ZF = ((src1 & src2) == 0), CF = ((~src1 & src2) == 0)
        if (vex_map == 2 && vex_op == 0x17 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            bool zf = true, cf = true;
            const usize qwords = vex_l ? 4 : 2;
            for (usize i = 0; i < qwords; ++i) {
                u64 a = m_context.ymm[src1].q[i];
                u64 b = src2.q[i];
                if ((a & b) != 0) zf = false;
                if ((a & ~b) != 0) cf = false;
            }
            m_context.rflags &= ~(CpuContext::FLAG_ZF | CpuContext::FLAG_CF);
            if (zf) m_context.rflags |= CpuContext::FLAG_ZF;
            if (cf) m_context.rflags |= CpuContext::FLAG_CF;

            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VDPPS (map 3, op 0x40): dot product of packed single-precision floats
        // VEX.128/256.66.0F3A 40 /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x40 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const usize lanes = vex_l ? 2 : 1;
            for (usize lane = 0; lane < lanes; ++lane) {
                const usize base = lane * 4;
                float sum = 0.0f;
                for (usize i = 0; i < 4; ++i) {
                    if (imm8 & (0x10u << i)) {
                        sum += m_context.ymm[src1].f[base + i] * src2.f[base + i];
                    }
                }
                for (usize i = 0; i < 4; ++i) {
                    res.f[base + i] = (imm8 & (1u << i)) ? sum : 0.0f;
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VDPPD (map 3, op 0x41): dot product of packed double-precision floats
        // VEX.128/256.66.0F3A 41 /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x41 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const usize lanes = vex_l ? 2 : 1;
            for (usize lane = 0; lane < lanes; ++lane) {
                const usize base = lane * 2;
                double sum = 0.0;
                for (usize i = 0; i < 2; ++i) {
                    if (imm8 & (0x10u << i)) {
                        sum += m_context.ymm[src1].df[base + i] * src2.df[base + i];
                    }
                }
                for (usize i = 0; i < 2; ++i) {
                    res.df[base + i] = (imm8 & (1u << i)) ? sum : 0.0;
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VMOVMSKPS / VMOVMSKPD (map 1, op 0x50): extract sign mask from packed FP
        // pp=0: VMOVMSKPS (single), pp=1: VMOVMSKPD (double)
        if (vex_op == 0x50) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid || !m.is_reg) return StepResult::IllegalInstruction;
            const u8 dest_reg = m.reg_id & 15;
            const u8 src = m.rm_id & 15;
            u32 mask = 0;
            if (vex_pp == 1) { // VMOVMSKPD: double (2 or 4 elements)
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) {
                    u64 val = 0;
                    std::memcpy(&val, &m_context.ymm[src].q[i], 8);
                    if (static_cast<i64>(val) < 0) mask |= (1u << i);
                }
            } else { // VMOVMSKPS: single (4 or 8 elements)
                const usize count = vex_l ? 8 : 4;
                for (usize i = 0; i < count; ++i) {
                    u32 val = 0;
                    std::memcpy(&val, &m_context.ymm[src].d[i], 4);
                    if (static_cast<i32>(val) < 0) mask |= (1u << i);
                }
            }
            set_gpr32(m_context, dest_reg, mask);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VMOVD / VMOVQ (0x6E GPR -> Vector, 0x7E Vector -> GPR)
        if (vex_op == 0x6E) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            m_context.ymm[dest] = CpuContext::YmmReg{};
            if (vex_w) {
                m_context.ymm[dest].q[0] = read_rm(m, 64);
            } else {
                m_context.ymm[dest].d[0] = static_cast<u32>(read_rm(m, 32));
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        if (vex_op == 0x7E) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            if (vex_pp == 2) { // VMOVQ xmm1, xmm2/m64 (F3): load, zero bits 255:64
                u64 val = 0;
                if (m.is_reg) val = m_context.ymm[m.rm_id & 15].q[0];
                else if (!m_memory.read_value(m.mem_addr, val)) return StepResult::MemoryFault;
                m_context.ymm[m.reg_id & 15] = CpuContext::YmmReg{};
                m_context.ymm[m.reg_id & 15].q[0] = val;
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }
            // 66: VMOVD/VMOVQ r/m32/64, xmm
            const u8 src = m.reg_id & 15;
            if (vex_w) {
                write_rm(m, m_context.ymm[src].q[0], 64);
            } else {
                write_rm(m, m_context.ymm[src].d[0], 32);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VMOVQ xmm1/m64, xmm2 (VEX.128.66.0F D6 /r): a STORE -- ModR/M.reg is
        // the source. This used to be decoded as a load into ModR/M.reg, so
        // every `vmovq [mem], xmm` was dropped and clobbered the source
        // register instead; mono-ps4's GC size-map initialiser then left its
        // table unwritten and the allocator scanned it forever.
        if (vex_map == 1 && vex_op == 0xD6 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u64 val = m_context.ymm[m.reg_id & 15].q[0];
            if (m.is_reg) {
                auto& dst = m_context.ymm[m.rm_id & 15];
                dst.q[0] = val;
                dst.q[1] = dst.q[2] = dst.q[3] = 0;
            } else if (!m_memory.write_value(m.mem_addr, val)) {
                return StepResult::MemoryFault;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VMOVHPS store (VEX.128.0F 17 /r): copy the high 64 bits of the
        // ModR/M.reg XMM register to memory (vex_pp=0 for PS, vex_pp=1 for PD).
        // VEX.L=1 is invalid.
        if (vex_map == 1 && vex_op == 0x17) {
            if (vex_l) return StepResult::IllegalInstruction;
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex,
                                  vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid || m.is_reg) return StepResult::IllegalInstruction;
            m_memory.write_value(m.mem_addr, m_context.ymm[m.reg_id & 15].q[1]);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VMOVLPS load (VEX.128.0F 12 /r, pp=0 or 1): load 64 bits from memory to low 64 bits of dest
        // vex_pp=0 && mod==3: VMOVHLPS xmm1, xmm2, xmm3 (dst.q[0]=src_rm.q[1], dst.q[1]=src_vvvv.q[1])
        // vex_pp=2 (F3): VMOVSLDUP (duplicate even 32-bit floats)
        // vex_pp=3 (F2): VMOVDDUP (duplicate low 64-bit qword in each lane)
        if (vex_map == 1 && vex_op == 0x12) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex,
                                  vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;

            if (vex_pp == 2) { // VMOVSLDUP
                CpuContext::YmmReg src{};
                if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) {
                    m_context.ymm[dest].f[i * 2] = src.f[i * 2];
                    m_context.ymm[dest].f[i * 2 + 1] = src.f[i * 2];
                }
                if (!vex_l) {
                    m_context.ymm[dest].q[2] = 0;
                    m_context.ymm[dest].q[3] = 0;
                }
            } else if (vex_pp == 3) { // VMOVDDUP
                CpuContext::YmmReg src{};
                if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 8);
                m_context.ymm[dest].q[0] = src.q[0];
                m_context.ymm[dest].q[1] = src.q[0];
                if (vex_l) {
                    m_context.ymm[dest].q[2] = src.q[2];
                    m_context.ymm[dest].q[3] = src.q[2];
                } else {
                    m_context.ymm[dest].q[2] = 0;
                    m_context.ymm[dest].q[3] = 0;
                }
            } else { // vex_pp == 0 (PS) or 1 (PD)
                if (vex_l) return StepResult::IllegalInstruction;
                if (m.is_reg) {
                    if (vex_pp != 0) return StepResult::IllegalInstruction;
                    // VMOVHLPS: dst.q[0] = src_rm.q[1], dst.q[1] = src_vvvv.q[1]
                    m_context.ymm[dest].q[0] = m_context.ymm[m.rm_id & 15].q[1];
                    m_context.ymm[dest].q[1] = m_context.ymm[src1].q[1];
                } else {
                    u64 val = 0;
                    m_memory.read_value(m.mem_addr, val);
                    m_context.ymm[dest].q[0] = val;
                    m_context.ymm[dest].q[1] = m_context.ymm[src1].q[1];
                }
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VMOVHPS load (VEX.128.0F 16 /r, pp=0 or 1): load 64 bits from memory to high 64 bits
        // vex_pp=0 && mod==3: VMOVLHPS xmm1, xmm2, xmm3 (dst.q[0]=src_vvvv.q[0], dst.q[1]=src_rm.q[0])
        // vex_pp=2 (F3): VMOVSHDUP (duplicate odd 32-bit floats)
        if (vex_map == 1 && vex_op == 0x16) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex,
                                  vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;

            if (vex_pp == 2) { // VMOVSHDUP
                CpuContext::YmmReg src{};
                if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) {
                    m_context.ymm[dest].f[i * 2] = src.f[i * 2 + 1];
                    m_context.ymm[dest].f[i * 2 + 1] = src.f[i * 2 + 1];
                }
                if (!vex_l) {
                    m_context.ymm[dest].q[2] = 0;
                    m_context.ymm[dest].q[3] = 0;
                }
            } else { // vex_pp == 0 or 1
                if (vex_l) return StepResult::IllegalInstruction;
                if (m.is_reg) {
                    if (vex_pp != 0) return StepResult::IllegalInstruction;
                    // VMOVLHPS: dst.q[0] = src_vvvv.q[0], dst.q[1] = src_rm.q[0]
                    m_context.ymm[dest].q[0] = m_context.ymm[src1].q[0];
                    m_context.ymm[dest].q[1] = m_context.ymm[m.rm_id & 15].q[0];
                } else {
                    u64 val = 0;
                    m_memory.read_value(m.mem_addr, val);
                    m_context.ymm[dest].q[0] = m_context.ymm[src1].q[0];
                    m_context.ymm[dest].q[1] = val;
                }
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VUNPCKLPS / VUNPCKLPD (map 1, op 0x14): unpack low packed single/double
        // pp=0: VUNPCKLPS (single), pp=1: VUNPCKLPD (double)
        if (vex_map == 1 && (vex_op == 0x14 || vex_op == 0x15)) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const bool is_pd = (vex_pp == 1);
            const usize elements = is_pd ? (vex_l ? 4 : 2) : (vex_l ? 8 : 4);
            const bool is_low = (vex_op == 0x14);

            for (usize lane = 0; lane < (vex_l ? 2 : 1); ++lane) {
                const usize base = lane * elements;
                const usize src1_base = lane * elements;
                const usize src2_base = lane * elements;
                for (usize i = 0; i < elements; ++i) {
                    const usize res_idx = base + i;
                    if (is_low) {
                        // UNPCKL: interleave low elements from src1 and src2
                        if (i < elements / 2) {
                            if (is_pd) {
                                res.q[res_idx] = m_context.ymm[src1].q[src1_base + i];
                            } else {
                                res.d[res_idx] = m_context.ymm[src1].d[src1_base + i];
                            }
                        } else {
                            const usize src2_idx = i - elements / 2;
                            if (is_pd) {
                                res.q[res_idx] = src2.q[src2_base + src2_idx];
                            } else {
                                res.d[res_idx] = src2.d[src2_base + src2_idx];
                            }
                        }
                    } else {
                        // UNPCKH: interleave high elements from src1 and src2
                        if (i < elements / 2) {
                            const usize src1_idx = elements / 2 + i;
                            if (is_pd) {
                                res.q[res_idx] = m_context.ymm[src1].q[src1_base + src1_idx];
                            } else {
                                res.d[res_idx] = m_context.ymm[src1].d[src1_base + src1_idx];
                            }
                        } else {
                            const usize src2_idx = elements / 2 + i - elements / 2;
                            if (is_pd) {
                                res.q[res_idx] = src2.q[src2_base + elements / 2 + src2_idx];
                            } else {
                                res.d[res_idx] = src2.d[src2_base + elements / 2 + src2_idx];
                            }
                        }
                    }
                }
            }
            m_context.ymm[dest] = res;
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VHADDPS (VEX.128/256.0F 7C): horizontal packed single-precision
        // addition, independently within each 128-bit lane.
        if (vex_map == 1 && vex_op == 0x7C) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex,
                                  vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) {
                src2 = m_context.ymm[m.rm_id & 15];
            } else if (!m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16)) {
                return StepResult::MemoryFault;
            }
            CpuContext::YmmReg result{};
            const usize lanes = vex_l ? 2 : 1;
            for (usize lane = 0; lane < lanes; ++lane) {
                const usize base = lane * 4;
                const auto& a = m_context.ymm[src1].f;
                const auto& b = src2.f;
                result.f[base + 0] = a[base + 0] + a[base + 1];
                result.f[base + 1] = a[base + 2] + a[base + 3];
                result.f[base + 2] = b[base + 0] + b[base + 1];
                result.f[base + 3] = b[base + 2] + b[base + 3];
            }
            m_context.ymm[dest] = result;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPACKSSDW: pack signed dwords from the two source operands into
        // signed saturated words, independently for each 128-bit lane.
        if (vex_map == 1 && vex_op == 0x6B) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2_val{};
            if (m.is_reg) {
                src2_val = m_context.ymm[m.rm_id & 15];
            } else if (!m_memory.read_bytes(m.mem_addr, &src2_val, vex_l ? 32 : 16)) {
                return StepResult::MemoryFault;
            }

            const auto& src1_val = m_context.ymm[src1];
            CpuContext::YmmReg res{};
            const usize lanes = vex_l ? 2 : 1;
            for (usize lane = 0; lane < lanes; ++lane) {
                const usize d = lane * 4;
                const usize w = lane * 8;
                for (usize i = 0; i < 4; ++i) {
                    const i64 a = static_cast<i32>(src1_val.d[d + i]);
                    const i64 b = static_cast<i32>(src2_val.d[d + i]);
                    const auto saturate = [](i64 value) -> u16 {
                        if (value > 32767) return 0x7FFF;
                        if (value < -32768) return 0x8000;
                        return static_cast<u16>(static_cast<i16>(value));
                    };
                    res.w[w + i] = saturate(a);
                    res.w[w + 4 + i] = saturate(b);
                }
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPUNPCKLBW (0x60), VPUNPCKLWD (0x61), VPUNPCKLDQ (0x62),
        // VPUNPCKHBW (0x68), VPUNPCKHWD (0x69), VPUNPCKHDQ (0x6A),
        // VPUNPCKLQDQ (0x6C), VPUNPCKHQDQ (0x6D)
        if (vex_map == 1 && (vex_op == 0x60 || vex_op == 0x61 || vex_op == 0x62 ||
                             vex_op == 0x68 || vex_op == 0x69 || vex_op == 0x6A ||
                             vex_op == 0x6C || vex_op == 0x6D)) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2_val{};
            if (m.is_reg) {
                src2_val = m_context.ymm[m.rm_id & 15];
            } else {
                m_memory.read_bytes(m.mem_addr, &src2_val, vex_l ? 32 : 16);
            }
            const auto s1 = m_context.ymm[src1];
            CpuContext::YmmReg res{};

            const usize lanes = vex_l ? 2 : 1;
            for (usize l = 0; l < lanes; ++l) {
                const usize q_base = l * 2;
                const usize d_base = l * 4;
                const usize w_base = l * 8;
                const usize b_base = l * 16;

                if (vex_op == 0x6C) { // VPUNPCKLQDQ: low qwords
                    res.q[q_base + 0] = s1.q[q_base + 0];
                    res.q[q_base + 1] = src2_val.q[q_base + 0];
                } else if (vex_op == 0x6D) { // VPUNPCKHQDQ: high qwords
                    res.q[q_base + 0] = s1.q[q_base + 1];
                    res.q[q_base + 1] = src2_val.q[q_base + 1];
                } else if (vex_op == 0x62) { // VPUNPCKLDQ: low dwords
                    res.d[d_base + 0] = s1.d[d_base + 0];
                    res.d[d_base + 1] = src2_val.d[d_base + 0];
                    res.d[d_base + 2] = s1.d[d_base + 1];
                    res.d[d_base + 3] = src2_val.d[d_base + 1];
                } else if (vex_op == 0x6A) { // VPUNPCKHDQ: high dwords
                    res.d[d_base + 0] = s1.d[d_base + 2];
                    res.d[d_base + 1] = src2_val.d[d_base + 2];
                    res.d[d_base + 2] = s1.d[d_base + 3];
                    res.d[d_base + 3] = src2_val.d[d_base + 3];
                } else if (vex_op == 0x61) { // VPUNPCKLWD: low words
                    for (usize i = 0; i < 4; ++i) {
                        res.w[w_base + 2 * i] = s1.w[w_base + i];
                        res.w[w_base + 2 * i + 1] = src2_val.w[w_base + i];
                    }
                } else if (vex_op == 0x69) { // VPUNPCKHWD: high words
                    for (usize i = 0; i < 4; ++i) {
                        res.w[w_base + 2 * i] = s1.w[w_base + 4 + i];
                        res.w[w_base + 2 * i + 1] = src2_val.w[w_base + 4 + i];
                    }
                } else if (vex_op == 0x60) { // VPUNPCKLBW: low bytes
                    for (usize i = 0; i < 8; ++i) {
                        res.b[b_base + 2 * i] = s1.b[b_base + i];
                        res.b[b_base + 2 * i + 1] = src2_val.b[b_base + i];
                    }
                } else if (vex_op == 0x68) { // VPUNPCKHBW: high bytes
                    for (usize i = 0; i < 8; ++i) {
                        res.b[b_base + 2 * i] = s1.b[b_base + 8 + i];
                        res.b[b_base + 2 * i + 1] = src2_val.b[b_base + 8 + i];
                    }
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPXOR (0xEF) / VPOR (0xEB) / VORPS (0x56) / VPAND (0xDB) / VANDPS (0x54) / VPANDN (0xDF) / VANDNPS (0x55)
        if (vex_op == 0xEF || vex_op == 0xEB || vex_op == 0xDB || vex_op == 0xDF ||
            vex_op == 0x56 || vex_op == 0x54 || vex_op == 0x55) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2_val{};
            if (m.is_reg) {
                src2_val = m_context.ymm[m.rm_id & 15];
            } else {
                m_memory.read_bytes(m.mem_addr, &src2_val, vex_l ? 32 : 16);
            }
            const usize num_qwords = vex_l ? 4 : 2;
            for (usize q = 0; q < num_qwords; ++q) {
                if (vex_op == 0xEF) m_context.ymm[dest].q[q] = m_context.ymm[src1].q[q] ^ src2_val.q[q];
                else if (vex_op == 0xEB || vex_op == 0x56) m_context.ymm[dest].q[q] = m_context.ymm[src1].q[q] | src2_val.q[q];
                else if (vex_op == 0xDB || vex_op == 0x54) m_context.ymm[dest].q[q] = m_context.ymm[src1].q[q] & src2_val.q[q];
                else if (vex_op == 0xDF || vex_op == 0x55) m_context.ymm[dest].q[q] = (~m_context.ymm[src1].q[q]) & src2_val.q[q];
            }
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VMOVAPS / VMOVAPD (0x28 load, 0x29 store)
        if (vex_op == 0x28) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const usize bytes = vex_l ? 32 : 16;
            if (m.is_reg) std::memcpy(&m_context.ymm[dest], &m_context.ymm[m.rm_id & 15], bytes);
            else m_memory.read_bytes(m.mem_addr, &m_context.ymm[dest], bytes);
            if (!vex_l) { m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0; }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        if (vex_op == 0x29) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 src = m.reg_id & 15;
            const usize bytes = vex_l ? 32 : 16;
            if (m.is_reg) std::memcpy(&m_context.ymm[m.rm_id & 15], &m_context.ymm[src], bytes);
            else m_memory.write_bytes(m.mem_addr, &m_context.ymm[src], bytes);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x71, 0x72, 0x73: Vector shifts by immediate
        if (vex_op == 0x71 || vex_op == 0x72 || vex_op == 0x73) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 sub_op = m.reg_id & 7;
            const u8 dest = vex_vvvv & 15;
            CpuContext::YmmReg src{};
            if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const usize lanes = vex_l ? 2 : 1;
            for (usize l = 0; l < lanes; ++l) {
                u8* dst_lane = reinterpret_cast<u8*>(&res) + l * 16;
                const u8* src_lane = reinterpret_cast<const u8*>(&src) + l * 16;
                if (vex_op == 0x73) {
                    if (sub_op == 7) { // VPSLLDQ (shift left bytes)
                        for (usize i = 0; i < 16; ++i) {
                            dst_lane[i] = (i >= imm8) ? src_lane[i - imm8] : 0;
                        }
                    } else if (sub_op == 3) { // VPSRLDQ (shift right bytes)
                        for (usize i = 0; i < 16; ++i) {
                            dst_lane[i] = (i + imm8 < 16) ? src_lane[i + imm8] : 0;
                        }
                    } else if (sub_op == 6) { // VPSLLQ (shift left qwords by bits)
                        for (usize q = 0; q < 2; ++q) {
                            const u64 v = src.q[l * 2 + q];
                            res.q[l * 2 + q] = (imm8 < 64) ? (v << imm8) : 0;
                        }
                    } else if (sub_op == 2) { // VPSRLQ (shift right qwords by bits)
                        for (usize q = 0; q < 2; ++q) {
                            const u64 v = src.q[l * 2 + q];
                            res.q[l * 2 + q] = (imm8 < 64) ? (v >> imm8) : 0;
                        }
                    }
                } else if (vex_op == 0x72) {
                    // VPSRLD (2), VPSRAD (4), VPSLLD (6)
                    for (usize d = 0; d < 4; ++d) {
                        const u32 v = src.d[l * 4 + d];
                        if (sub_op == 6) res.d[l * 4 + d] = (imm8 < 32) ? (v << imm8) : 0;
                        else if (sub_op == 2) res.d[l * 4 + d] = (imm8 < 32) ? (v >> imm8) : 0;
                        else if (sub_op == 4) res.d[l * 4 + d] = (imm8 < 32) ? (static_cast<u32>(static_cast<i32>(v) >> imm8)) : (static_cast<i32>(v) < 0 ? 0xFFFFFFFF : 0);
                    }
                } else if (vex_op == 0x71) {
                    // VPSRLW (2), VPSRAW (4), VPSLLW (6)
                    for (usize w = 0; w < 8; ++w) {
                        const u16 v = src.w[l * 8 + w];
                        if (sub_op == 6) res.w[l * 8 + w] = (imm8 < 16) ? (v << imm8) : 0;
                        else if (sub_op == 2) res.w[l * 8 + w] = (imm8 < 16) ? (v >> imm8) : 0;
                        else if (sub_op == 4) res.w[l * 8 + w] = (imm8 < 16) ? (static_cast<u16>(static_cast<i16>(v) >> imm8)) : (static_cast<i16>(v) < 0 ? 0xFFFF : 0);
                    }
                }
            }
            m_context.ymm[dest] = res;
            if (!vex_l) { m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0; }
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // 0x51 (VSQRT), 0x52 (VRSQRT), 0x53 (VRCP), 0x58 (VADD), 0x59 (VMUL), 0x5C (VSUB), 0x5E (VDIV)
        if (vex_op == 0x51 || vex_op == 0x52 || vex_op == 0x53 ||
            vex_op == 0x58 || vex_op == 0x59 || vex_op == 0x5C || vex_op == 0x5D ||
            vex_op == 0x5E || vex_op == 0x5F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : (vex_pp >= 2 ? 8 : 16));

            if (vex_pp == 2) { // VADDSS / VMULSS / VSUBSS / VDIVSS (scalar float)
                float s1 = 0.0f, s2 = 0.0f;
                std::memcpy(&s1, &m_context.ymm[src1].d[0], 4);
                std::memcpy(&s2, &src2.d[0], 4);
                float r = (vex_op == 0x51) ? std::sqrt(s2) :
                          (vex_op == 0x52) ? (1.0f / std::sqrt(s2)) :
                          (vex_op == 0x53) ? (1.0f / s2) :
                          (vex_op == 0x58) ? (s1 + s2) :
                          (vex_op == 0x59) ? (s1 * s2) :
                          (vex_op == 0x5C) ? (s1 - s2) :
                          (vex_op == 0x5D) ? ((s1 < s2) ? s1 : s2) : // VMINSS
                          (vex_op == 0x5E) ? (s1 / s2) :
                                             ((s1 > s2) ? s1 : s2); // VMAXSS (0x5F)
                CpuContext::YmmReg res = m_context.ymm[src1];
                std::memcpy(&res.d[0], &r, 4);
                res.q[2] = 0; res.q[3] = 0;
                m_context.ymm[dest] = res;
            } else if (vex_pp == 3) { // VADDSD / VMULSD / VSUBSD / VDIVSD (scalar double)
                double s1 = 0.0, s2 = 0.0;
                std::memcpy(&s1, &m_context.ymm[src1].q[0], 8);
                std::memcpy(&s2, &src2.q[0], 8);
                double r = (vex_op == 0x51) ? std::sqrt(s2) :
                           (vex_op == 0x58) ? (s1 + s2) :
                           (vex_op == 0x59) ? (s1 * s2) :
                           (vex_op == 0x5C) ? (s1 - s2) :
                           (vex_op == 0x5D) ? ((s1 < s2) ? s1 : s2) : // VMINSD
                           (vex_op == 0x5E) ? (s1 / s2) :
                                             ((s1 > s2) ? s1 : s2); // VMAXSD (0x5F)
                CpuContext::YmmReg res = m_context.ymm[src1];
                std::memcpy(&res.q[0], &r, 8);
                res.q[2] = 0; res.q[3] = 0;
                m_context.ymm[dest] = res;
            } else if (vex_pp == 0) { // VADDPS / VMULPS / VSUBPS / VDIVPS (packed float)
                const usize count = vex_l ? 8 : 4;
                for (usize i = 0; i < count; ++i) {
                    float s1 = 0.0f, s2 = 0.0f;
                    std::memcpy(&s1, &m_context.ymm[src1].d[i], 4);
                    std::memcpy(&s2, &src2.d[i], 4);
                    float r = (vex_op == 0x51) ? std::sqrt(s2) :
                              (vex_op == 0x52) ? (1.0f / std::sqrt(s2)) :
                              (vex_op == 0x53) ? (1.0f / s2) :
                              (vex_op == 0x58) ? (s1 + s2) :
                              (vex_op == 0x59) ? (s1 * s2) :
                              (vex_op == 0x5C) ? (s1 - s2) :
                              (vex_op == 0x5D) ? ((s1 < s2) ? s1 : s2) : // VMINPS
                              (vex_op == 0x5E) ? (s1 / s2) :
                                                 ((s1 > s2) ? s1 : s2); // VMAXPS (0x5F)
                    std::memcpy(&m_context.ymm[dest].d[i], &r, 4);
                }
                if (!vex_l) { m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0; }
            } else { // VADDPD / VMULPD / VSUBPD / VDIVPD (packed double)
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) {
                    double s1 = 0.0, s2 = 0.0;
                    std::memcpy(&s1, &m_context.ymm[src1].q[i], 8);
                    std::memcpy(&s2, &src2.q[i], 8);
                    double r = (vex_op == 0x51) ? std::sqrt(s2) :
                               (vex_op == 0x58) ? (s1 + s2) :
                              (vex_op == 0x59) ? (s1 * s2) :
                              (vex_op == 0x5C) ? (s1 - s2) :
                              (vex_op == 0x5D) ? ((s1 < s2) ? s1 : s2) : // VMINPD
                              (vex_op == 0x5E) ? (s1 / s2) :
                                                 ((s1 > s2) ? s1 : s2); // VMAXPD (0x5F)
                    std::memcpy(&m_context.ymm[dest].q[i], &r, 8);
                }
                if (!vex_l) { m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0; }
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPCMPGTQ (map 2, op 0x37, pp=1): compare packed qwords for greater than (signed)
        if (vex_map == 2 && vex_op == 0x37 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const usize count = vex_l ? 4 : 2;
            for (usize i = 0; i < count; ++i) {
                const i64 a = static_cast<i64>(m_context.ymm[src1].q[i]);
                const i64 b = static_cast<i64>(src2.q[i]);
                res.q[i] = (a > b) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
            }
            m_context.ymm[dest] = res;
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPMINSB/VPMINSD/VPMINUW/VPMINUD / VPMAXSB/VPMAXSD/VPMAXUW/VPMAXUD (map 2, op 0x38-0x3F)
        if (vex_map == 2 && vex_op >= 0x38 && vex_op <= 0x3F && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            if (vex_op == 0x38 || vex_op == 0x3C) { // PMINSB / PMAXSB (bytes)
                const usize count = vex_l ? 32 : 16;
                for (usize i = 0; i < count; ++i) {
                    const i8 a = static_cast<i8>(m_context.ymm[src1].b[i]);
                    const i8 b = static_cast<i8>(src2.b[i]);
                    res.b[i] = (vex_op == 0x38) ? ((a < b) ? a : b) : ((a > b) ? a : b);
                }
            } else if (vex_op == 0x39 || vex_op == 0x3D) { // PMINSD / PMAXSD (dwords signed)
                const usize count = vex_l ? 8 : 4;
                for (usize i = 0; i < count; ++i) {
                    const i32 a = static_cast<i32>(m_context.ymm[src1].d[i]);
                    const i32 b = static_cast<i32>(src2.d[i]);
                    res.d[i] = static_cast<u32>((vex_op == 0x39) ? ((a < b) ? a : b) : ((a > b) ? a : b));
                }
            } else if (vex_op == 0x3A || vex_op == 0x3E) { // PMINUW / PMAXUW (words unsigned)
                const usize count = vex_l ? 16 : 8;
                for (usize i = 0; i < count; ++i) {
                    const u16 a = m_context.ymm[src1].w[i];
                    const u16 b = src2.w[i];
                    res.w[i] = (vex_op == 0x3A) ? ((a < b) ? a : b) : ((a > b) ? a : b);
                }
            } else { // 0x3B / 0x3F: PMINUD / PMAXUD (dwords unsigned)
                const usize count = vex_l ? 8 : 4;
                for (usize i = 0; i < count; ++i) {
                    const u32 a = m_context.ymm[src1].d[i];
                    const u32 b = src2.d[i];
                    res.d[i] = (vex_op == 0x3B) ? ((a < b) ? a : b) : ((a > b) ? a : b);
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (vex_op == 0x2E || vex_op == 0x2F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 src1 = m.reg_id & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, (vex_pp == 1) ? 8 : 4);

            m_context.rflags &= ~(CpuContext::FLAG_ZF | CpuContext::FLAG_PF | CpuContext::FLAG_CF | CpuContext::FLAG_OF | CpuContext::FLAG_SF | CpuContext::FLAG_AF);
            if (vex_pp == 1) { // double
                double d1 = 0.0, d2 = 0.0;
                std::memcpy(&d1, &m_context.ymm[src1].q[0], 8);
                std::memcpy(&d2, &src2.q[0], 8);
                if (std::isnan(d1) || std::isnan(d2)) {
                    m_context.rflags |= (CpuContext::FLAG_ZF | CpuContext::FLAG_PF | CpuContext::FLAG_CF);
                } else if (d1 < d2) {
                    m_context.rflags |= CpuContext::FLAG_CF;
                } else if (d1 == d2) {
                    m_context.rflags |= CpuContext::FLAG_ZF;
                }
            } else { // float
                float f1 = 0.0f, f2 = 0.0f;
                std::memcpy(&f1, &m_context.ymm[src1].d[0], 4);
                std::memcpy(&f2, &src2.d[0], 4);
                if (std::isnan(f1) || std::isnan(f2)) {
                    m_context.rflags |= (CpuContext::FLAG_ZF | CpuContext::FLAG_PF | CpuContext::FLAG_CF);
                } else if (f1 < f2) {
                    m_context.rflags |= CpuContext::FLAG_CF;
                } else if (f1 == f2) {
                    m_context.rflags |= CpuContext::FLAG_ZF;
                }
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x2A: VCVTSI2SS (pp=2) / VCVTSI2SD (pp=3)
        if (vex_op == 0x2A) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg res = m_context.ymm[src1];

            if (vex_pp == 3) { // VCVTSI2SD: integer to double
                double dval = 0.0;
                if (vex_w) {
                    const i64 ival = static_cast<i64>(read_rm(m, 64));
                    dval = static_cast<double>(ival);
                } else {
                    const i32 ival = static_cast<i32>(read_rm(m, 32));
                    dval = static_cast<double>(ival);
                }
                std::memcpy(&res.q[0], &dval, 8);
            } else { // VCVTSI2SS: integer to float
                float fval = 0.0f;
                if (vex_w) {
                    const i64 ival = static_cast<i64>(read_rm(m, 64));
                    fval = static_cast<float>(ival);
                } else {
                    const i32 ival = static_cast<i32>(read_rm(m, 32));
                    fval = static_cast<float>(ival);
                }
                std::memcpy(&res.d[0], &fval, 4);
            }
            res.q[2] = 0; res.q[3] = 0;
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPERMILPS (map 3, op 0x04, pp=1): permute in-lane single-precision
        // VEX.128/256.66.0F3A 04 /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x04 && (vex_pp == 1 || vex_pp == 0)) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg src{};
            if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const usize lanes = vex_l ? 2 : 1;
            for (usize lane = 0; lane < lanes; ++lane) {
                const usize base = lane * 4;
                for (usize i = 0; i < 4; ++i) {
                    const u8 sel = (imm8 >> (i * 2)) & 3;
                    res.d[base + i] = src.d[base + sel];
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPERMILPD (map 3, op 0x05, pp=1): permute in-lane double-precision
        // VEX.128/256.66.0F3A 05 /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x05 && (vex_pp == 1 || vex_pp == 0)) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg src{};
            if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const usize lanes = vex_l ? 2 : 1;
            for (usize lane = 0; lane < lanes; ++lane) {
                const usize base = lane * 2;
                for (usize i = 0; i < 2; ++i) {
                    const u8 sel = (imm8 >> i) & 1;
                    res.q[base + i] = src.q[base + sel];
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VROUNDPS (map 3, op 0x08, pp=1): round packed single-precision floats
        // VEX.128/256.66.0F3A 08 /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x08 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg src{};
            if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);

            const u8 mode = (imm8 & 8) ? 0 : static_cast<u8>(imm8 & 3);
            CpuContext::YmmReg res{};
            const usize count = vex_l ? 8 : 4;
            for (usize i = 0; i < count; ++i) {
                switch (mode) {
                    case 0: res.f[i] = std::nearbyint(src.f[i]); break;
                    case 1: res.f[i] = std::floor(src.f[i]); break;
                    case 2: res.f[i] = std::ceil(src.f[i]); break;
                    case 3: default: res.f[i] = std::trunc(src.f[i]); break;
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VROUNDPD (map 3, op 0x09, pp=1): round packed double-precision floats
        // VEX.128/256.66.0F3A 09 /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x09 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg src{};
            if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);

            const u8 mode = (imm8 & 8) ? 0 : static_cast<u8>(imm8 & 3);
            CpuContext::YmmReg res{};
            const usize count = vex_l ? 4 : 2;
            for (usize i = 0; i < count; ++i) {
                switch (mode) {
                    case 0: res.df[i] = std::nearbyint(src.df[i]); break;
                    case 1: res.df[i] = std::floor(src.df[i]); break;
                    case 2: res.df[i] = std::ceil(src.df[i]); break;
                    case 3: default: res.df[i] = std::trunc(src.df[i]); break;
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VROUNDSS (map 3, op 0x0A, pp=1): round scalar single-precision float
        // VEX.128.66.0F3A 0A /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x0A && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            float in_val = 0.0f;
            if (m.is_reg) in_val = m_context.ymm[m.rm_id & 15].f[0];
            else m_memory.read_bytes(m.mem_addr, &in_val, 4);

            const u8 mode = (imm8 & 8) ? 0 : static_cast<u8>(imm8 & 3);
            CpuContext::YmmReg res = m_context.ymm[src1];
            switch (mode) {
                case 0: res.f[0] = std::nearbyint(in_val); break;
                case 1: res.f[0] = std::floor(in_val); break;
                case 2: res.f[0] = std::ceil(in_val); break;
                case 3: default: res.f[0] = std::trunc(in_val); break;
            }
            res.q[2] = 0;
            res.q[3] = 0;
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VROUNDSD (map 3, op 0x0B, pp=1): round scalar double-precision float
        // VEX.128.66.0F3A 0B /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x0B && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            double in_val = 0.0;
            if (m.is_reg) in_val = m_context.ymm[m.rm_id & 15].df[0];
            else m_memory.read_bytes(m.mem_addr, &in_val, 8);

            const u8 mode = (imm8 & 8) ? 0 : static_cast<u8>(imm8 & 3);
            CpuContext::YmmReg res = m_context.ymm[src1];
            switch (mode) {
                case 0: res.df[0] = std::nearbyint(in_val); break;
                case 1: res.df[0] = std::floor(in_val); break;
                case 2: res.df[0] = std::ceil(in_val); break;
                case 3: default: res.df[0] = std::trunc(in_val); break;
            }
            res.q[2] = 0;
            res.q[3] = 0;
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPBLENDD (map 3, op 0x02, pp=1) & VBLENDPS (map 3, op 0x0C, pp=1)
        // VEX.128/256.66.0F3A 02 / 0C /r ib -- HAS imm8
        if (vex_map == 3 && (vex_op == 0x02 || vex_op == 0x0C) && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res = m_context.ymm[src1];
            const usize count = vex_l ? 8 : 4;
            for (usize i = 0; i < count; ++i) {
                if (imm8 & (1u << i)) {
                    res.d[i] = src2.d[i];
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VBLENDPD (map 3, op 0x0D, pp=1): blend packed 64-bit double floats using imm8 mask
        // VEX.128/256.66.0F3A 0D /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x0D && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res = m_context.ymm[src1];
            const usize count = vex_l ? 4 : 2;
            for (usize i = 0; i < count; ++i) {
                if (imm8 & (1u << i)) {
                    res.q[i] = src2.q[i];
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPBLENDW (map 3, op 0x0E): blend packed 16-bit words using imm8 mask
        // VEX.128/256.66.0F3A 0E /r ib -- HAS imm8
        if (vex_map == 3 && vex_op == 0x0E && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            const usize count = vex_l ? 16 : 8;
            for (usize i = 0; i < count; ++i) {
                if (imm8 & (1u << i)) {
                    m_context.ymm[dest].w[i] = src2.w[i];
                } else {
                    m_context.ymm[dest].w[i] = m_context.ymm[src1].w[i];
                }
            }
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // 0x2A: VCVTSI2SS (pp=2) / VCVTSI2SD (pp=3)
        if (vex_op == 0x2C) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            if (vex_pp == 3) { // VCVTTSD2SI: double to integer with truncation
                double dval = 0.0;
                if (m.is_reg) std::memcpy(&dval, &m_context.ymm[m.rm_id & 15].q[0], 8);
                else m_memory.read_bytes(m.mem_addr, &dval, 8);
                if (vex_w) set_gpr64(m_context, m.reg_id, x86_cvt_i64(dval));
                else set_gpr32(m_context, m.reg_id, x86_cvt_i32(dval));
            } else { // VCVTTSS2SI: float to integer with truncation
                float fval = 0.0f;
                if (m.is_reg) std::memcpy(&fval, &m_context.ymm[m.rm_id & 15].d[0], 4);
                else m_memory.read_bytes(m.mem_addr, &fval, 4);
                if (vex_w) set_gpr64(m_context, m.reg_id, x86_cvt_i64(fval));
                else set_gpr32(m_context, m.reg_id, x86_cvt_i32(fval));
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x5A: VCVTPS2PD (pp=0) / VCVTPD2PS (pp=1) / VCVTSS2SD (pp=2) / VCVTSD2SS (pp=3)
        if (vex_map == 1 && vex_op == 0x5A) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg res = m_context.ymm[src1];
            if (vex_pp == 3) { // VCVTSD2SS: double to float (scalar)
                double dval = 0.0;
                if (m.is_reg) std::memcpy(&dval, &m_context.ymm[m.rm_id & 15].q[0], 8);
                else m_memory.read_bytes(m.mem_addr, &dval, 8);
                const float fval = static_cast<float>(dval);
                std::memcpy(&res.d[0], &fval, 4);
                res.q[2] = 0; res.q[3] = 0;
            } else if (vex_pp == 2) { // VCVTSS2SD: float to double (scalar)
                float fval = 0.0f;
                if (m.is_reg) std::memcpy(&fval, &m_context.ymm[m.rm_id & 15].d[0], 4);
                else m_memory.read_bytes(m.mem_addr, &fval, 4);
                const double dval = static_cast<double>(fval);
                std::memcpy(&res.q[0], &dval, 8);
                res.q[2] = 0; res.q[3] = 0;
            } else if (vex_pp == 0) { // VCVTPS2PD: packed float to double
                CpuContext::YmmReg src{};
                if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 16 : 8);
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) res.df[i] = static_cast<double>(src.f[i]);
                if (!vex_l) { res.q[2] = 0; res.q[3] = 0; }
            } else if (vex_pp == 1) { // VCVTPD2PS: packed double to float
                CpuContext::YmmReg src{};
                if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) res.f[i] = static_cast<float>(src.df[i]);
                for (usize i = count; i < 8; ++i) res.d[i] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x5B: VCVTDQ2PS (pp=0) / VCVTPS2DQ (pp=1) / VCVTTPS2DQ (pp=2)
        if (vex_map == 1 && vex_op == 0x5B) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg src{};
            if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const usize count = vex_l ? 8 : 4;
            if (vex_pp == 0) { // VCVTDQ2PS: dword to float
                for (usize i = 0; i < count; ++i) {
                    res.f[i] = static_cast<float>(static_cast<i32>(src.d[i]));
                }
            } else if (vex_pp == 1) { // VCVTPS2DQ: float to dword (round to nearest)
                for (usize i = 0; i < count; ++i) {
                    res.d[i] = x86_cvt_i32(std::nearbyint(src.f[i]));
                }
            } else if (vex_pp == 2) { // VCVTTPS2DQ: float to dword (truncate)
                for (usize i = 0; i < count; ++i) {
                    res.d[i] = static_cast<u32>(static_cast<i32>(std::trunc(src.f[i])));
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0xE6: VCVTTPD2DQ (pp=1) / VCVTDQ2PD (pp=2) / VCVTPD2DQ (pp=3)
        if (vex_map == 1 && vex_op == 0xE6) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg res{};
            if (vex_pp == 2) { // VCVTDQ2PD: packed dword to double
                CpuContext::YmmReg src{};
                if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 16 : 8);
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) res.df[i] = static_cast<double>(static_cast<i32>(src.d[i]));
                if (!vex_l) { res.q[2] = 0; res.q[3] = 0; }
            } else { // VCVTTPD2DQ (pp=1, truncate) / VCVTPD2DQ (pp=3, round)
                CpuContext::YmmReg src{};
                if (m.is_reg) src = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src, vex_l ? 32 : 16);
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) {
                    res.d[i] = (vex_pp == 1) ? static_cast<u32>(static_cast<i32>(std::trunc(src.df[i])))
                                             : x86_cvt_i32(std::nearbyint(src.df[i]));
                }
                for (usize i = count; i < 8; ++i) res.d[i] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VEXTRACTF128 / VEXTRACTI128 (map 3, opcode 0x19)
        if (vex_map == 3 && vex_op == 0x19) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 src_reg = m.reg_id & 15;
            const usize offset_q = (imm8 & 1) ? 2 : 0;
            if (m.is_reg) {
                const u8 dst_reg = m.rm_id & 15;
                m_context.ymm[dst_reg].q[0] = m_context.ymm[src_reg].q[offset_q];
                m_context.ymm[dst_reg].q[1] = m_context.ymm[src_reg].q[offset_q + 1];
                m_context.ymm[dst_reg].q[2] = 0;
                m_context.ymm[dst_reg].q[3] = 0;
            } else {
                m_memory.write_bytes(m.mem_addr, &m_context.ymm[src_reg].q[offset_q], 16);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VINSERTF128 / VINSERTI128 (map 3, opcode 0x18)
        if (vex_map == 3 && vex_op == 0x18) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg res = m_context.ymm[src1];
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, 16);
            const usize offset_q = (imm8 & 1) ? 2 : 0;
            res.q[offset_q] = src2.q[0];
            res.q[offset_q + 1] = src2.q[1];
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VCMPPS / VCMPPD / VCMPSS / VCMPSD (map 1, op 0xC2): compare packed/scalar FP
        // pp=0: VCMPPS (packed single), pp=1: VCMPPD (packed double)
        // pp=2: VCMPSS (scalar single), pp=3: VCMPSD (scalar double)
        if (vex_map == 1 && vex_op == 0xC2) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, (vex_pp >= 2) ? 8 : (vex_l ? 32 : 16));

            CpuContext::YmmReg res{};
            const bool is_pd = (vex_pp == 1);
            const bool is_scalar = (vex_pp >= 2);
            const usize count = is_scalar ? 1 : (is_pd ? (vex_l ? 4 : 2) : (vex_l ? 8 : 4));

            for (usize i = 0; i < count; ++i) {
                bool cmp_result = false;
                if (is_pd) {
                    double a = 0.0, b = 0.0;
                    std::memcpy(&a, &m_context.ymm[src1].q[i], 8);
                    std::memcpy(&b, &src2.q[i], 8);
                    switch (imm8 & 0x1F) {
                        case 0: cmp_result = (a == b); break;      // EQ
                        case 1: cmp_result = (a < b); break;       // LT
                        case 2: cmp_result = (a <= b); break;      // LE
                        case 3: cmp_result = std::isnan(a) || std::isnan(b); break; // UNORD
                        case 4: cmp_result = (a != b); break;      // NEQ
                        case 5: cmp_result = (a >= b); break;      // NLT
                        case 6: cmp_result = (a > b); break;       // NLE
                        case 7: cmp_result = !(std::isnan(a) || std::isnan(b)); break; // ORD
                        default: cmp_result = false;
                    }
                    res.q[i] = cmp_result ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
                } else {
                    float a = 0.0f, b = 0.0f;
                    std::memcpy(&a, &m_context.ymm[src1].d[i], 4);
                    std::memcpy(&b, &src2.d[i], 4);
                    switch (imm8 & 0x1F) {
                        case 0: cmp_result = (a == b); break;      // EQ
                        case 1: cmp_result = (a < b); break;       // LT
                        case 2: cmp_result = (a <= b); break;      // LE
                        case 3: cmp_result = std::isnan(a) || std::isnan(b); break; // UNORD
                        case 4: cmp_result = (a != b); break;      // NEQ
                        case 5: cmp_result = (a >= b); break;      // NLT
                        case 6: cmp_result = (a > b); break;       // NLE
                        case 7: cmp_result = !(std::isnan(a) || std::isnan(b)); break; // ORD
                        default: cmp_result = false;
                    }
                    res.d[i] = cmp_result ? 0xFFFFFFFFU : 0U;
                }
            }
            if (is_scalar) {
                // For scalar, preserve upper bits from src1
                if (is_pd) {
                    res.q[1] = m_context.ymm[src1].q[1];
                    if (vex_l) { res.q[2] = m_context.ymm[src1].q[2]; res.q[3] = m_context.ymm[src1].q[3]; }
                } else {
                    res.d[1] = m_context.ymm[src1].d[1];
                    res.d[2] = m_context.ymm[src1].d[2];
                    res.d[3] = m_context.ymm[src1].d[3];
                    if (vex_l) { res.q[2] = m_context.ymm[src1].q[2]; res.q[3] = m_context.ymm[src1].q[3]; }
                }
            } else if (!vex_l) {
                res.q[2] = 0; res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VSHUFPS (map 1, op 0xC6, pp=0): shuffle packed single-precision floats
        // VEX.128/256.0F C6 /r ib -- HAS imm8
        if (vex_map == 1 && vex_op == 0xC6 && vex_pp == 0) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            const usize lanes = vex_l ? 2 : 1;
            for (usize lane = 0; lane < lanes; ++lane) {
                const usize base = lane * 4;
                res.d[base + 0] = m_context.ymm[src1].d[base + ((imm8 >> 0) & 3)];
                res.d[base + 1] = m_context.ymm[src1].d[base + ((imm8 >> 2) & 3)];
                res.d[base + 2] = src2.d[base + ((imm8 >> 4) & 3)];
                res.d[base + 3] = src2.d[base + ((imm8 >> 6) & 3)];
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VSHUFPD (map 1, op 0xC6, pp=1): shuffle packed double-precision floats
        // VEX.128/256.66.0F C6 /r ib -- HAS imm8
        if (vex_map == 1 && vex_op == 0xC6 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            if (!vex_l) {
                res.q[0] = m_context.ymm[src1].q[(imm8 >> 0) & 1];
                res.q[1] = src2.q[(imm8 >> 1) & 1];
                res.q[2] = 0;
                res.q[3] = 0;
            } else {
                res.q[0] = m_context.ymm[src1].q[(imm8 >> 0) & 1];
                res.q[1] = src2.q[(imm8 >> 1) & 1];
                res.q[2] = m_context.ymm[src1].q[2 + ((imm8 >> 2) & 1)];
                res.q[3] = src2.q[2 + ((imm8 >> 3) & 1)];
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPEXTRW (map 1, op 0xC5, pp=1): extract word into GPR
        // VEX.128.66.0F C5 /r ib -- HAS imm8
        if (vex_map == 1 && vex_op == 0xC5 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            u16 val = 0;
            if (m.is_reg) {
                val = m_context.ymm[m.rm_id & 15].w[imm8 & 7];
            } else {
                CpuContext::YmmReg src{};
                m_memory.read_bytes(m.mem_addr, &src, 16);
                val = src.w[imm8 & 7];
            }
            set_gpr64(m_context, dest, static_cast<u64>(val));
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VBLENDVPS (map 3, op 0x4A, pp=1): blend packed singles using sign bits from mask reg
        // VEX.128/256.66.0F3A 4A /r /is4 ib -- HAS imm8
        // imm8[7:4] = mask_reg (YMM index)
        if (vex_map == 3 && vex_op == 0x4A && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            const u8 mask_reg = (imm8 >> 4) & 0xF;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res = m_context.ymm[src1];
            const usize count = vex_l ? 8 : 4;
            for (usize i = 0; i < count; ++i) {
                if (m_context.ymm[mask_reg].d[i] & 0x80000000U) { // High bit of each float
                    res.f[i] = src2.f[i];
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VBLENDVPD (map 3, op 0x4B, pp=1): blend packed doubles using sign bits from mask reg
        // VEX.128/256.66.0F3A 4B /r /is4 ib -- HAS imm8
        // imm8[7:4] = mask_reg (YMM index)
        if (vex_map == 3 && vex_op == 0x4B && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            const u8 mask_reg = (imm8 >> 4) & 0xF;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res = m_context.ymm[src1];
            const usize count = vex_l ? 4 : 2;
            for (usize i = 0; i < count; ++i) {
                if (m_context.ymm[mask_reg].q[i] & 0x8000000000000000ULL) { // High bit of each double
                    res.f[i * 2] = src2.f[i * 2];
                    res.f[i * 2 + 1] = src2.f[i * 2 + 1];
                }
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPINSRB (map 3, op 0x20, pp=1): insert byte from r/m8 into xmm at imm8 position
        if (vex_map == 3 && vex_op == 0x20 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src = vex_vvvv & 15;
            CpuContext::YmmReg res = m_context.ymm[src];
            u8 val = 0;
            if (m.is_reg) {
                val = static_cast<u8>(get_gpr64(m_context, m.rm_id & 15));
            } else {
                m_memory.read_value(m.mem_addr, val);
            }
            res.b[imm8 & 15] = val;
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPINSRW (map 1, op 0xC4, pp=1): insert word from r/m16 into xmm at imm8 position
        if (vex_map == 1 && vex_op == 0xC4 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src = vex_vvvv & 15;
            CpuContext::YmmReg res = m_context.ymm[src];
            u16 val = 0;
            if (m.is_reg) {
                val = static_cast<u16>(get_gpr64(m_context, m.rm_id & 15));
            } else {
                m_memory.read_value(m.mem_addr, val);
            }
            res.w[imm8 & 7] = val;
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VINSERTPS (map 3, op 0x21, pp=1): insert single float from src2 into dest, zeroing specified elements
        if (vex_map == 3 && vex_op == 0x21 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg res = m_context.ymm[src1];
            u32 src_val = 0;
            if (m.is_reg) {
                const u8 count_s = (imm8 >> 6) & 3;
                src_val = m_context.ymm[m.rm_id & 15].d[count_s];
            } else {
                m_memory.read_value(m.mem_addr, src_val);
            }
            const u8 count_d = (imm8 >> 4) & 3;
            res.d[count_d] = src_val;
            const u8 zmask = imm8 & 0x0F;
            for (usize i = 0; i < 4; ++i) {
                if (zmask & (1u << i)) {
                    res.d[i] = 0;
                }
            }
            res.q[2] = 0;
            res.q[3] = 0;
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPEXTRB (0x14), VPEXTRW (0x15), VPEXTRD / VPEXTRQ (0x16)
        if (vex_map == 3 && (vex_op == 0x14 || vex_op == 0x15 || vex_op == 0x16)) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 src = m.reg_id & 15;
            if (vex_op == 0x16) {
                if (vex_w) {
                    const u64 val = m_context.ymm[src].q[imm8 & 1];
                    write_rm(m, val, 64);
                } else {
                    const u32 val = m_context.ymm[src].d[imm8 & 3];
                    write_rm(m, val, 32);
                }
            } else if (vex_op == 0x15) {
                const u16 val = m_context.ymm[src].w[imm8 & 7];
                write_rm(m, val, 16);
            } else {
                const u8 val = m_context.ymm[src].b[imm8 & 15];
                write_rm(m, val, 8);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPINSRD / VPINSRQ (map 3, op 0x22, pp=1): insert dword/qword from r/m into xmm at imm8 position
        if (vex_map == 3 && vex_op == 0x22 && vex_pp == 1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg res = m_context.ymm[src1];
            if (vex_w) {
                u64 val = 0;
                if (m.is_reg) {
                    val = get_gpr64(m_context, m.rm_id & 15);
                } else {
                    m_memory.read_value(m.mem_addr, val);
                }
                res.q[imm8 & 1] = val;
            } else {
                u32 val = 0;
                if (m.is_reg) {
                    val = static_cast<u32>(get_gpr64(m_context, m.rm_id & 15));
                } else {
                    m_memory.read_value(m.mem_addr, val);
                }
                res.d[imm8 & 3] = val;
            }
            if (!vex_l) {
                res.q[2] = 0;
                res.q[3] = 0;
            }
            m_context.ymm[dest] = res;
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // VPADDQ (0xD4), VPADDD (0xFE), VPADDW (0xFD), VPADDB (0xFC)
        if (vex_map == 1 && (vex_op == 0xD4 || vex_op == 0xFE || vex_op == 0xFD || vex_op == 0xFC)) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);
            if (vex_op == 0xD4) {
                const usize count = vex_l ? 4 : 2;
                for (usize i = 0; i < count; ++i) m_context.ymm[dest].q[i] = m_context.ymm[src1].q[i] + src2.q[i];
            } else if (vex_op == 0xFE) {
                const usize count = vex_l ? 8 : 4;
                for (usize i = 0; i < count; ++i) m_context.ymm[dest].d[i] = m_context.ymm[src1].d[i] + src2.d[i];
            } else if (vex_op == 0xFD) {
                const usize count = vex_l ? 16 : 8;
                for (usize i = 0; i < count; ++i) m_context.ymm[dest].w[i] = m_context.ymm[src1].w[i] + src2.w[i];
            } else {
                const usize count = vex_l ? 32 : 16;
                for (usize i = 0; i < count; ++i) m_context.ymm[dest].b[i] = m_context.ymm[src1].b[i] + src2.b[i];
            }
            if (!vex_l) { m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0; }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // VPCMPEQB (0x74), VPCMPEQW (0x75), VPCMPEQD (0x76)
        // VPCMPGTB (0x64), VPCMPGTW (0x65), VPCMPGTD (0x66)
        if (vex_map == 1 && (vex_op == 0x74 || vex_op == 0x75 || vex_op == 0x76 ||
                             vex_op == 0x64 || vex_op == 0x65 || vex_op == 0x66)) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            const u8 src1 = vex_vvvv & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, vex_l ? 32 : 16);

            CpuContext::YmmReg res{};
            if (vex_op == 0x76) { // VPCMPEQD
                const usize count = vex_l ? 8 : 4;
                for (usize i = 0; i < count; ++i) {
                    res.d[i] = (m_context.ymm[src1].d[i] == src2.d[i]) ? 0xFFFFFFFFU : 0U;
                }
            } else if (vex_op == 0x75) { // VPCMPEQW
                const usize count = vex_l ? 16 : 8;
                for (usize i = 0; i < count; ++i) {
                    res.w[i] = (m_context.ymm[src1].w[i] == src2.w[i]) ? 0xFFFFU : 0U;
                }
            } else if (vex_op == 0x74) { // VPCMPEQB
                const usize count = vex_l ? 32 : 16;
                for (usize i = 0; i < count; ++i) {
                    res.b[i] = (m_context.ymm[src1].b[i] == src2.b[i]) ? 0xFFU : 0U;
                }
            } else if (vex_op == 0x66) { // VPCMPGTD
                const usize count = vex_l ? 8 : 4;
                for (usize i = 0; i < count; ++i) {
                    res.d[i] = (static_cast<i32>(m_context.ymm[src1].d[i]) > static_cast<i32>(src2.d[i])) ? 0xFFFFFFFFU : 0U;
                }
            } else if (vex_op == 0x65) { // VPCMPGTW
                const usize count = vex_l ? 16 : 8;
                for (usize i = 0; i < count; ++i) {
                    res.w[i] = (static_cast<i16>(m_context.ymm[src1].w[i]) > static_cast<i16>(src2.w[i])) ? 0xFFFFU : 0U;
                }
            } else if (vex_op == 0x64) { // VPCMPGTB
                const usize count = vex_l ? 32 : 16;
                for (usize i = 0; i < count; ++i) {
                    res.b[i] = (static_cast<i8>(m_context.ymm[src1].b[i]) > static_cast<i8>(src2.b[i])) ? 0xFFU : 0U;
                }
            }
            m_context.ymm[dest] = res;
            if (!vex_l) {
                m_context.ymm[dest].q[2] = 0;
                m_context.ymm[dest].q[3] = 0;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // Do not treat an unsupported VEX instruction as a successful no-op.
        // Advancing RIP here leaves destination registers stale and can corrupt
        // guest control flow long after the original instruction.
        {
            usize extra_imm = 0;
            if (vex_map == 3) {
                extra_imm = 1;
            } else if (vex_map == 1) {
                if (vex_op == 0x70 || vex_op == 0x71 || vex_op == 0x72 || vex_op == 0x73 ||
                    vex_op == 0xC2 || vex_op == 0xC4 || vex_op == 0xC5 || vex_op == 0xC6) {
                    extra_imm = 1;
                }
            }
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_vex, vex_r, vex_x, vex_b, seg_override, m_context, extra_imm);
            std::fprintf(stderr, "[CPU UNIMPL VEX] map=%u op=0x%02x at RIP=0x%llx vex_l=%u vex_pp=%u vex_vvvv=%u tid=%u\n",
                         static_cast<unsigned>(vex_map), static_cast<unsigned>(vex_op),
                         static_cast<unsigned long long>(start_rip),
                         static_cast<unsigned>(vex_l), static_cast<unsigned>(vex_pp), static_cast<unsigned>(vex_vvvv),
                         static_cast<unsigned>(ps4::t_guest_tid));
            // A mis-sized earlier instruction lands RIP mid-instruction and
            // shows up here as a bogus VEX prefix; the history shows which.
            for (usize i = 0; i < 16; ++i) {
                const GuestAddr h_rip = s_rip_history[(s_rip_hist_idx + 128 - 16 + i) % 128];
                if (h_rip != 0)
                    std::fprintf(stderr, "  [%02zu] RIP=0x%llx\n", i, (unsigned long long)h_rip);
            }
            std::fflush(stderr);
            (void)m;
            return StepResult::IllegalInstruction;
        }
    }

    // 0x0F: Two-byte opcodes
    if (op0 == 0x0F && m_memory.is_valid_address(start_rip + cursor, 1)) {
        const u8 op1 = raw_code[cursor++];
        const GuestAddr next_ip_base_0f = start_rip + cursor;

        // 0x0F 0x38: 3-byte escape prefix
        if (op1 == 0x38 && m_memory.is_valid_address(start_rip + cursor, 1)) {
            const u8 op2 = raw_code[cursor++];
            const GuestAddr next_ip_base_0f38 = start_rip + cursor;

            // 0x0F 0x38 0xF0: MOVBE reg, m
            if (op2 == 0xF0) {
                auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f38, rex_r, rex_x, rex_b, seg_override, m_context);
                if (!m.valid || m.is_reg) return StepResult::IllegalInstruction;
                if (rex_w) {
                    u64 val = 0;
                    if (!m_memory.read_value(m.mem_addr, val)) return StepResult::MemoryFault;
                    set_gpr64(m_context, m.reg_id, __builtin_bswap64(val));
                } else if (operand_size_16) {
                    u16 val = 0;
                    if (!m_memory.read_value(m.mem_addr, val)) return StepResult::MemoryFault;
                    set_gpr16(m_context, m.reg_id, __builtin_bswap16(val));
                } else {
                    u32 val = 0;
                    if (!m_memory.read_value(m.mem_addr, val)) return StepResult::MemoryFault;
                    set_gpr32(m_context, m.reg_id, __builtin_bswap32(val));
                }
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }

            // 0x0F 0x38 0xF1: MOVBE m, reg
            if (op2 == 0xF1) {
                auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f38, rex_r, rex_x, rex_b, seg_override, m_context);
                if (!m.valid || m.is_reg) return StepResult::IllegalInstruction;
                if (rex_w) {
                    const u64 val = __builtin_bswap64(get_gpr64(m_context, m.reg_id));
                    if (!m_memory.write_value(m.mem_addr, val)) return StepResult::MemoryFault;
                } else if (operand_size_16) {
                    const u16 val = __builtin_bswap16(static_cast<u16>(get_gpr64(m_context, m.reg_id)));
                    if (!m_memory.write_value(m.mem_addr, val)) return StepResult::MemoryFault;
                } else {
                    const u32 val = __builtin_bswap32(static_cast<u32>(get_gpr64(m_context, m.reg_id)));
                    if (!m_memory.write_value(m.mem_addr, val)) return StepResult::MemoryFault;
                }
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }

            // 0x0F 0x38 0x00: PSHUFB xmm1, xmm2/m128
            if (op2 == 0x00) {
                auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f38, rex_r, rex_x, rex_b, seg_override, m_context);
                if (!m.valid) return StepResult::IllegalInstruction;
                const u8 dest = m.reg_id & 15;
                CpuContext::YmmReg mask_reg{};
                if (m.is_reg) mask_reg = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &mask_reg, 16);

                CpuContext::YmmReg orig = m_context.ymm[dest];
                CpuContext::YmmReg res{};
                for (int i = 0; i < 16; ++i) {
                    const u8 m_byte = mask_reg.b[i];
                    if (m_byte & 0x80) {
                        res.b[i] = 0;
                    } else {
                        res.b[i] = orig.b[m_byte & 0x0F];
                    }
                }
                m_context.ymm[dest] = res;
                m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0;
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }

            // 0x0F 0x38 0x04: PMADDUBSW xmm1, xmm2/m128
            if (op2 == 0x04) {
                auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f38, rex_r, rex_x, rex_b, seg_override, m_context);
                if (!m.valid) return StepResult::IllegalInstruction;
                const u8 dest = m.reg_id & 15;
                CpuContext::YmmReg src2{};
                if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src2, 16);

                CpuContext::YmmReg orig = m_context.ymm[dest];
                CpuContext::YmmReg res{};
                for (int i = 0; i < 8; ++i) {
                    const u8 u0 = orig.b[2 * i];
                    const u8 u1 = orig.b[2 * i + 1];
                    const i8 s0 = static_cast<i8>(src2.b[2 * i]);
                    const i8 s1 = static_cast<i8>(src2.b[2 * i + 1]);
                    i32 prod = (static_cast<i32>(u0) * static_cast<i32>(s0)) +
                               (static_cast<i32>(u1) * static_cast<i32>(s1));
                    if (prod > 32767) prod = 32767;
                    else if (prod < -32768) prod = -32768;
                    res.w[i] = static_cast<i16>(prod);
                }
                m_context.ymm[dest] = res;
                m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0;
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }

            std::fprintf(stderr, "[CPU UNIMPL 0F38] opcode=0x%02x at RIP=0x%llx\n",
                         static_cast<unsigned>(op2),
                         static_cast<unsigned long long>(start_rip));
            return StepResult::IllegalInstruction;
        }

        // 0x0F 0x3A: 3-byte escape prefix with imm8
        if (op1 == 0x3A && m_memory.is_valid_address(start_rip + cursor, 1)) {
            const u8 op2 = raw_code[cursor++];
            const GuestAddr next_ip_base_0f3a = start_rip + cursor;

            // 0x0F 0x3A 0x0F: PALIGNR xmm1, xmm2/m128, imm8
            if (op2 == 0x0F) {
                auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f3a, rex_r, rex_x, rex_b, seg_override, m_context, 1);
                if (!m.valid) return StepResult::IllegalInstruction;
                const u8 imm8 = raw_code[cursor + m.bytes_consumed];
                const u8 dest = m.reg_id & 15;
                CpuContext::YmmReg src2{};
                if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
                else m_memory.read_bytes(m.mem_addr, &src2, 16);

                u8 concat[32];
                std::memcpy(concat, &src2.b[0], 16);
                std::memcpy(concat + 16, &m_context.ymm[dest].b[0], 16);

                CpuContext::YmmReg res{};
                if (imm8 < 32) {
                    for (int i = 0; i < 16; ++i) {
                        res.b[i] = (imm8 + i < 32) ? concat[imm8 + i] : 0;
                    }
                }
                m_context.ymm[dest] = res;
                m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0;
                m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
                return StepResult::Ok;
            }

            std::fprintf(stderr, "[CPU UNIMPL 0F3A] opcode=0x%02x at RIP=0x%llx\n",
                         static_cast<unsigned>(op2),
                         static_cast<unsigned long long>(start_rip));
            return StepResult::IllegalInstruction;
        }

        // 0x0F 0x0B: UD2 (Undefined instruction / abort / trap)
        if (op1 == 0x0B) {
            std::fprintf(stderr, "[CPU] UD2 instruction executed at RIP=0x%llx\n", (unsigned long long)start_rip);
            std::fprintf(stderr, "[CPU] Recent RIP history before UD2 (last 64 instructions):\n");
            for (usize i = 0; i < 64; ++i) {
                const GuestAddr h_rip = s_rip_history[(s_rip_hist_idx + 64 + i) % 128];
                if (h_rip != 0) {
                    std::fprintf(stderr, "  [%02zu] RIP=0x%llx\n", i, (unsigned long long)h_rip);
                }
            }
            return StepResult::IllegalInstruction;
        }

        // 0x0F 0xB6: MOVZX reg, r/m8
        if (op1 == 0xB6) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 byte_val = static_cast<u8>(read_rm(m, 8));
            if (rex_w) set_gpr64(m_context, m.reg_id, byte_val);
            else if (operand_size_16) set_gpr16(m_context, m.reg_id, byte_val);
            else set_gpr32(m_context, m.reg_id, byte_val);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xB7: MOVZX reg, r/m16
        if (op1 == 0xB7) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u16 word_val = static_cast<u16>(read_rm(m, 16));
            if (rex_w) set_gpr64(m_context, m.reg_id, word_val);
            else set_gpr32(m_context, m.reg_id, word_val);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xBE: MOVSX reg, r/m8
        if (op1 == 0xBE) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const i8 byte_val = static_cast<i8>(read_rm(m, 8));
            if (rex_w) set_gpr64(m_context, m.reg_id, static_cast<u64>(static_cast<i64>(byte_val)));
            else if (operand_size_16) set_gpr16(m_context, m.reg_id, static_cast<u16>(static_cast<i16>(byte_val)));
            else set_gpr32(m_context, m.reg_id, static_cast<u32>(static_cast<i32>(byte_val)));
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xBF: MOVSX reg, r/m16
        if (op1 == 0xBF) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const i16 word_val = static_cast<i16>(read_rm(m, 16));
            if (rex_w) set_gpr64(m_context, m.reg_id, static_cast<u64>(static_cast<i64>(word_val)));
            else set_gpr32(m_context, m.reg_id, static_cast<u32>(static_cast<i32>(word_val)));
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x80 - 0x8F: Jcc rel32
        if (op1 >= 0x80 && op1 <= 0x8F) {
            i32 rel32 = 0;
            std::memcpy(&rel32, raw_code + cursor, 4);
            const bool cond = check_condition(m_context, op1 & 0x0F);
            m_context.rip = (start_rip + cursor + 4) + (cond ? rel32 : 0);
            return StepResult::Ok;
        }

        // 0x0F 0x90 - 0x9F: SETcc r/m8
        if (op1 >= 0x90 && op1 <= 0x9F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const bool cond = check_condition(m_context, op1 & 0x0F);
            write_rm(m, cond ? 1 : 0, 8);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x40 - 0x4F: CMOVcc reg, r/m
        if (op1 >= 0x40 && op1 <= 0x4F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            if (check_condition(m_context, op1 & 0x0F)) {
                const u64 val = read_rm(m, op_bits);
                if (op_bits == 64) set_gpr64(m_context, m.reg_id, val);
                else if (op_bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(val));
                else set_gpr16(m_context, m.reg_id, static_cast<u16>(val));
            } else if (op_bits == 32) {
                // In 64-bit mode a 32-bit CMOV clears bits 63:32 even when not taken.
                set_gpr32(m_context, m.reg_id, static_cast<u32>(get_gpr64(m_context, m.reg_id)));
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xAE: RDFSBASE (reg=0), RDGSBASE (reg=1), WRFSBASE (reg=2), WRGSBASE (reg=3)
        if (op1 == 0xAE) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            if (m.is_reg) {
                const u8 sub_op = m.reg_id & 7;
                if (sub_op == 0) set_gpr64(m_context, m.rm_id, m_context.fs_base);
                else if (sub_op == 1) set_gpr64(m_context, m.rm_id, m_context.gs_base);
                else if (sub_op == 2) m_context.fs_base = get_gpr64(m_context, m.rm_id);
                else if (sub_op == 3) m_context.gs_base = get_gpr64(m_context, m.rm_id);
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }
            // LDMXCSR m32 / STMXCSR m32
            if ((m.reg_id & 7) == 2 || (m.reg_id & 7) == 3) {
                const bool ok = (m.reg_id & 7) == 2 ? m_memory.read_value(m.mem_addr, m_context.mxcsr)
                                                    : m_memory.write_value(m.mem_addr, m_context.mxcsr);
                if (!ok) return StepResult::MemoryFault;
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }
        }

        // 0x0F 0xAF: IMUL reg, r/m
        if (op1 == 0xAF) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u64 src = read_rm(m, op_bits);
            const u64 dst = (op_bits == 64) ? get_gpr64(m_context, m.reg_id) :
                            (op_bits == 32) ? static_cast<u32>(get_gpr64(m_context, m.reg_id)) :
                            static_cast<u16>(get_gpr64(m_context, m.reg_id));
            if (op_bits == 64) {
                const __int128 full = static_cast<__int128>(static_cast<i64>(dst)) * static_cast<__int128>(static_cast<i64>(src));
                const i64 res = static_cast<i64>(full);
                set_gpr64(m_context, m.reg_id, static_cast<u64>(res));
                const bool overflow = (full != res);
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            } else if (op_bits == 32) {
                const i64 full = static_cast<i64>(static_cast<i32>(dst)) * static_cast<i64>(static_cast<i32>(src));
                const i32 res = static_cast<i32>(full);
                set_gpr32(m_context, m.reg_id, static_cast<u32>(res));
                const bool overflow = (full != res);
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            } else {
                const i32 full = static_cast<i32>(static_cast<i16>(dst)) * static_cast<i32>(static_cast<i16>(src));
                const i16 res = static_cast<i16>(full);
                set_gpr16(m_context, m.reg_id, static_cast<u16>(res));
                const bool overflow = (full != res);
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x1F: NOP r/m (multi-byte NOP)
        if (op1 == 0x1F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x31: RDTSC — wall-clock nanoseconds so guest TSC-delta timeout
        // loops expire in real time. (Was: +100 per execution, i.e. frozen.)
        if (op1 == 0x31) {
            const u64 tsc = static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count());
            m_context.rax = static_cast<u32>(tsc);
            m_context.rdx = static_cast<u32>(tsc >> 32);
            m_context.rip = start_rip + cursor;
            return StepResult::Ok;
        }

        // 0x0F 0xA2: CPUID
        if (op1 == 0xA2) {
            const u32 leaf = static_cast<u32>(m_context.rax);
            if (leaf == 0) {
                m_context.rax = 13;
                m_context.rbx = 0x756e6547; // "Genu"
                m_context.rdx = 0x49656e69; // "ineI"
                m_context.rcx = 0x6c65746e; // "ntel"
            } else if (leaf == 1) {
                m_context.rax = 0x000306c3; // Haswell/Jaguar-like family/model
                m_context.rbx = 0x00080800; // 8 cores
                m_context.rcx = 0xfdbae3bf; // SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AES
                m_context.rdx = 0xbfebfbff; // FPU, VME, DE, PSE, TSC, MSR, PAE, MCE, CX8, APIC, SEP, SSE, SSE2
            } else if (leaf == 7) {
                m_context.rax = 0;
                m_context.rbx = 0x00000029; // FSGSBASE, BMI1, AVX2, BMI2
                m_context.rcx = 0;
                m_context.rdx = 0;
            } else {
                m_context.rax = 0;
                m_context.rbx = 0;
                m_context.rcx = 0;
                m_context.rdx = 0;
            }
            m_context.rip = start_rip + cursor;
            return StepResult::Ok;
        }

        // 0x0F 0xB0 / 0xB1: CMPXCHG r/m, reg
        if (op1 == 0xB0 || op1 == 0xB1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const usize bits = (op1 == 0xB0) ? 8 : op_bits;
            const u64 acc = (bits == 8) ? get_gpr8(m_context, 0, has_rex) : (bits == 64) ? m_context.rax
                          : (bits == 16) ? static_cast<u16>(m_context.rax) : static_cast<u32>(m_context.rax);
            // Accumulator write on failure: 16-bit merges, 32-bit zero-extends.
            auto set_acc = [&](u64 v) {
                if (bits == 8) set_gpr8(m_context, 0, has_rex, static_cast<u8>(v));
                else if (bits == 16) set_gpr16(m_context, 0, static_cast<u16>(v));
                else if (bits == 32) set_gpr32(m_context, 0, static_cast<u32>(v));
                else m_context.rax = v;
            };
            const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
            if (!m.is_reg) {
                // Atomic single-step compare-exchange; see guest_atomic_cas.
                const u64 reg_val = (bits == 8) ? get_gpr8(m_context, m.reg_id, has_rex)
                                   : (bits == 64) ? get_gpr64(m_context, m.reg_id)
                                   : static_cast<u32>(get_gpr64(m_context, m.reg_id));
                int handled = -1;
                u64 actual = 0;
                if (bits == 64) handled = guest_atomic_cas<u64>(m_memory, m.mem_addr, static_cast<u64>(acc), static_cast<u64>(reg_val), reinterpret_cast<u64*>(&actual));
                else if (bits == 32) handled = guest_atomic_cas<u32>(m_memory, m.mem_addr, static_cast<u32>(acc), static_cast<u32>(reg_val), reinterpret_cast<u32*>(&actual));
                else if (bits == 16) handled = guest_atomic_cas<u16>(m_memory, m.mem_addr, static_cast<u16>(acc), static_cast<u16>(reg_val), reinterpret_cast<u16*>(&actual));
                else handled = guest_atomic_cas<u8>(m_memory, m.mem_addr, static_cast<u8>(acc), static_cast<u8>(reg_val), reinterpret_cast<u8*>(&actual));
                if (handled >= 0) {
                    const u64 rm_val = actual & mask;
                    update_flags_sub(m_context, acc, rm_val, acc - rm_val, bits);
                    m_context.set_flag(CpuContext::FLAG_ZF, handled == 1);
                    if (handled == 0) set_acc(rm_val);
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u64 rm_val = read_rm(m, bits);
            update_flags_sub(m_context, acc, rm_val, acc - rm_val, bits);
            if ((acc & mask) == (rm_val & mask)) {
                m_context.set_flag(CpuContext::FLAG_ZF, true);
                const u64 reg_val = (bits == 8) ? get_gpr8(m_context, m.reg_id, has_rex) : (bits == 64) ? get_gpr64(m_context, m.reg_id) : static_cast<u32>(get_gpr64(m_context, m.reg_id));
                write_rm(m, reg_val, bits);
            } else {
                m_context.set_flag(CpuContext::FLAG_ZF, false);
                // The destination is written even on failure (with its own
                // value), so a 32-bit register destination is zero-extended.
                if (m.is_reg) write_rm(m, rm_val, bits);
                set_acc(rm_val);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xC0 / 0xC1: XADD r/m, reg
        if (op1 == 0xC0 || op1 == 0xC1) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const usize bits = (op1 == 0xC0) ? 8 : op_bits;
            const u64 reg_val = (bits == 8) ? get_gpr8(m_context, m.reg_id, has_rex) : (bits == 64) ? get_gpr64(m_context, m.reg_id) : static_cast<u32>(get_gpr64(m_context, m.reg_id));
            if (!m.is_reg) {
                // Atomic fetch-add; see guest_atomic_xadd.
                int handled = -1;
                u64 old_raw = 0;
                if (bits == 64) handled = guest_atomic_xadd<u64>(m_memory, m.mem_addr, static_cast<u64>(reg_val), reinterpret_cast<u64*>(&old_raw));
                else if (bits == 32) handled = guest_atomic_xadd<u32>(m_memory, m.mem_addr, static_cast<u32>(reg_val), reinterpret_cast<u32*>(&old_raw));
                else if (bits == 16) handled = guest_atomic_xadd<u16>(m_memory, m.mem_addr, static_cast<u16>(reg_val), reinterpret_cast<u16*>(&old_raw));
                else handled = guest_atomic_xadd<u8>(m_memory, m.mem_addr, static_cast<u8>(reg_val), reinterpret_cast<u8*>(&old_raw));
                if (handled >= 0) {
                    const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
                    const u64 rm_val = old_raw & mask;
                    const u64 sum = (rm_val + reg_val) & mask;
                    update_flags_add(m_context, rm_val, reg_val, sum, bits);
                    if (bits == 8) set_gpr8(m_context, m.reg_id, has_rex, static_cast<u8>(rm_val));
                    else if (bits == 64) set_gpr64(m_context, m.reg_id, rm_val);
                    else if (bits == 16) set_gpr16(m_context, m.reg_id, static_cast<u16>(rm_val));
                    else set_gpr32(m_context, m.reg_id, static_cast<u32>(rm_val));
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u64 rm_val = read_rm(m, bits);
            const u64 sum = (rm_val + reg_val);
            update_flags_add(m_context, rm_val, reg_val, sum, bits);
            write_rm(m, sum, bits);
            if (bits == 8) set_gpr8(m_context, m.reg_id, has_rex, static_cast<u8>(rm_val));
            else if (bits == 64) set_gpr64(m_context, m.reg_id, rm_val);
            else if (bits == 16) set_gpr16(m_context, m.reg_id, static_cast<u16>(rm_val));
            else set_gpr32(m_context, m.reg_id, static_cast<u32>(rm_val));
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xBC / 0xBD: BSF (0xBC), BSR (0xBD) / TZCNT / LZCNT
        // 0F A4/A5: SHLD r/m, r, imm8/CL;  0F AC/AD: SHRD r/m, r, imm8/CL.
        // Unimplemented before (skipped by the ModR/M fallback).
        if (op1 == 0xA4 || op1 == 0xA5 || op1 == 0xAC || op1 == 0xAD) {
            const bool imm = (op1 == 0xA4 || op1 == 0xAC);
            const bool left = (op1 == 0xA4 || op1 == 0xA5);
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context, imm ? 1 : 0);
            if (!m.valid) return StepResult::IllegalInstruction;
            const usize bits = op_bits;
            u8 count = imm ? raw_code[cursor + m.bytes_consumed] : static_cast<u8>(m_context.rcx);
            count &= (bits == 64) ? 0x3F : 0x1F;
            if (count != 0) {
                const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
                const u64 dst = read_rm(m, bits) & mask;
                const u64 src = ((bits == 64) ? get_gpr64(m_context, m.reg_id) : get_gpr64(m_context, m.reg_id) & mask);
                // 16-bit with count > 16 is undefined; use the 32-bit concatenation.
                const unsigned __int128 wide = left ? ((static_cast<unsigned __int128>(dst) << bits) | src)
                                                    : ((static_cast<unsigned __int128>(src) << bits) | dst);
                u64 res;
                bool cf;
                if (left) {
                    res = static_cast<u64>(wide >> (bits - count)) & mask;
                    cf = ((dst >> (bits - count)) & 1) != 0;
                } else {
                    res = static_cast<u64>(wide >> count) & mask;
                    cf = ((dst >> (count - 1)) & 1) != 0;
                }
                const u64 msb = 1ULL << (bits - 1);
                update_flags_logic(m_context, res, bits);
                m_context.set_flag(CpuContext::FLAG_CF, cf);
                m_context.set_flag(CpuContext::FLAG_OF, ((res ^ dst) & msb) != 0);
                write_rm(m, res, bits);
            } else if (bits == 32 && m.is_reg) {
                set_gpr32(m_context, m.rm_id, static_cast<u32>(get_gpr64(m_context, m.rm_id)));
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed + (imm ? 1 : 0);
            return StepResult::Ok;
        }

        // F3 0F B8: POPCNT r, r/m. ZF = (src == 0); CF/OF/SF/AF/PF cleared.
        // Unimplemented before: the ModR/M fallback skipped it as a NOP.
        if (op1 == 0xB8 && rep_prefix == 0xF3) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u64 mask = (op_bits == 64) ? ~0ULL : ((1ULL << op_bits) - 1);
            const u64 src = read_rm(m, op_bits) & mask;
            const u64 cnt = static_cast<u64>(__builtin_popcountll(src));
            if (op_bits == 64) set_gpr64(m_context, m.reg_id, cnt);
            else if (op_bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(cnt));
            else set_gpr16(m_context, m.reg_id, static_cast<u16>(cnt));
            m_context.rflags &= ~(CpuContext::FLAG_CF | CpuContext::FLAG_PF | CpuContext::FLAG_AF |
                                  CpuContext::FLAG_ZF | CpuContext::FLAG_SF | CpuContext::FLAG_OF);
            if (src == 0) m_context.rflags |= CpuContext::FLAG_ZF;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        if (op1 == 0xBC || op1 == 0xBD) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u64 val = read_rm(m, op_bits);
            const u64 mask = (op_bits == 64) ? ~0ULL : ((1ULL << op_bits) - 1);
            const u64 masked_val = val & mask;

            if (rep_prefix == 0xF3) {
                // TZCNT (0xBC) / LZCNT (0xBD)
                if (op1 == 0xBD) {
                    // LZCNT: Count leading zeros
                    u64 cnt = 0;
                    if (masked_val == 0) {
                        cnt = op_bits;
                        m_context.set_flag(CpuContext::FLAG_CF, true);
                        m_context.set_flag(CpuContext::FLAG_ZF, false);
                    } else {
                        m_context.set_flag(CpuContext::FLAG_CF, false);
                        if (op_bits == 64) {
                            cnt = static_cast<u64>(__builtin_clzll(masked_val));
                        } else if (op_bits == 32) {
                            cnt = static_cast<u64>(__builtin_clz(static_cast<u32>(masked_val)));
                        } else {
                            cnt = static_cast<u64>(__builtin_clz(static_cast<u32>(masked_val) << 16));
                        }
                        m_context.set_flag(CpuContext::FLAG_ZF, (cnt == 0));
                    }
                    m_context.set_flag(CpuContext::FLAG_OF, false);
                    m_context.set_flag(CpuContext::FLAG_SF, false);
                    if (op_bits == 64) set_gpr64(m_context, m.reg_id, cnt);
                    else if (op_bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(cnt));
                    else set_gpr16(m_context, m.reg_id, static_cast<u16>(cnt));
                } else {
                    // TZCNT: Count trailing zeros
                    u64 cnt = 0;
                    if (masked_val == 0) {
                        cnt = op_bits;
                        m_context.set_flag(CpuContext::FLAG_CF, true);
                        m_context.set_flag(CpuContext::FLAG_ZF, false);
                    } else {
                        m_context.set_flag(CpuContext::FLAG_CF, false);
                        if (op_bits == 64) {
                            cnt = static_cast<u64>(__builtin_ctzll(masked_val));
                        } else if (op_bits == 32) {
                            cnt = static_cast<u64>(__builtin_ctz(static_cast<u32>(masked_val)));
                        } else {
                            cnt = static_cast<u64>(__builtin_ctz(static_cast<u32>(masked_val) | 0x10000));
                        }
                        m_context.set_flag(CpuContext::FLAG_ZF, (cnt == 0));
                    }
                    m_context.set_flag(CpuContext::FLAG_OF, false);
                    m_context.set_flag(CpuContext::FLAG_SF, false);
                    if (op_bits == 64) set_gpr64(m_context, m.reg_id, cnt);
                    else if (op_bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(cnt));
                    else set_gpr16(m_context, m.reg_id, static_cast<u16>(cnt));
                }
            } else {
                // BSF (0xBC) / BSR (0xBD)
                if (masked_val == 0) {
                    m_context.set_flag(CpuContext::FLAG_ZF, true);
                } else {
                    m_context.set_flag(CpuContext::FLAG_ZF, false);
                    u64 idx = 0;
                    if (op1 == 0xBC) {
                        if (op_bits == 64) idx = static_cast<u64>(__builtin_ctzll(masked_val));
                        else idx = static_cast<u64>(__builtin_ctz(static_cast<u32>(masked_val)));
                    } else {
                        if (op_bits == 64) idx = 63 - static_cast<u64>(__builtin_clzll(masked_val));
                        else if (op_bits == 32) idx = 31 - static_cast<u64>(__builtin_clz(static_cast<u32>(masked_val)));
                        else idx = 15 - static_cast<u64>(__builtin_clz(static_cast<u32>(masked_val) << 16));
                    }
                    if (op_bits == 64) set_gpr64(m_context, m.reg_id, idx);
                    else if (op_bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(idx));
                    else set_gpr16(m_context, m.reg_id, static_cast<u16>(idx));
                }
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xA3 / 0xAB / 0xB3 / 0xBB: BT / BTS / BTR / BTC r/m, reg
        if (op1 == 0xA3 || op1 == 0xAB || op1 == 0xB3 || op1 == 0xBB) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u64 bit_offset = get_gpr64(m_context, m.reg_id) % op_bits;
            if (!m.is_reg && (lock_prefix || op1 != 0xA3)) {
                int handled = -1;
                u64 old_raw = 0;
                const u64 mask = (1ULL << bit_offset);
                if (op1 == 0xAB) { // BTS
                    if (op_bits == 64) handled = guest_atomic_or<u64>(m_memory, m.mem_addr, mask, &old_raw);
                    else if (op_bits == 32) { u32 o = 0; handled = guest_atomic_or<u32>(m_memory, m.mem_addr, static_cast<u32>(mask), &o); old_raw = o; }
                    else { u16 o = 0; handled = guest_atomic_or<u16>(m_memory, m.mem_addr, static_cast<u16>(mask), &o); old_raw = o; }
                } else if (op1 == 0xB3) { // BTR
                    if (op_bits == 64) handled = guest_atomic_and<u64>(m_memory, m.mem_addr, ~mask, &old_raw);
                    else if (op_bits == 32) { u32 o = 0; handled = guest_atomic_and<u32>(m_memory, m.mem_addr, static_cast<u32>(~mask), &o); old_raw = o; }
                    else { u16 o = 0; handled = guest_atomic_and<u16>(m_memory, m.mem_addr, static_cast<u16>(~mask), &o); old_raw = o; }
                } else if (op1 == 0xBB) { // BTC
                    if (op_bits == 64) handled = guest_atomic_xor<u64>(m_memory, m.mem_addr, mask, &old_raw);
                    else if (op_bits == 32) { u32 o = 0; handled = guest_atomic_xor<u32>(m_memory, m.mem_addr, static_cast<u32>(mask), &o); old_raw = o; }
                    else { u16 o = 0; handled = guest_atomic_xor<u16>(m_memory, m.mem_addr, static_cast<u16>(mask), &o); old_raw = o; }
                }
                if (handled >= 0) {
                    m_context.set_flag(CpuContext::FLAG_CF, ((old_raw >> bit_offset) & 1) != 0);
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            u64 val = read_rm(m, op_bits);
            m_context.set_flag(CpuContext::FLAG_CF, ((val >> bit_offset) & 1) != 0);
            if (op1 == 0xAB) val |= (1ULL << bit_offset);
            else if (op1 == 0xB3) val &= ~(1ULL << bit_offset);
            else if (op1 == 0xBB) val ^= (1ULL << bit_offset);
            if (op1 != 0xA3) write_rm(m, val, op_bits);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xBA: Group 8 (BT/BTS/BTR/BTC r/m, imm8)
        if (op1 == 0xBA) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context, 1);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 imm8 = raw_code[cursor + m.bytes_consumed];
            const u64 bit_offset = imm8 % op_bits;
            const u8 sub_op = m.reg_id & 7;
            if (!m.is_reg && (lock_prefix || sub_op != 4)) {
                int handled = -1;
                u64 old_raw = 0;
                const u64 mask = (1ULL << bit_offset);
                if (sub_op == 5) { // BTS
                    if (op_bits == 64) handled = guest_atomic_or<u64>(m_memory, m.mem_addr, mask, &old_raw);
                    else if (op_bits == 32) { u32 o = 0; handled = guest_atomic_or<u32>(m_memory, m.mem_addr, static_cast<u32>(mask), &o); old_raw = o; }
                    else { u16 o = 0; handled = guest_atomic_or<u16>(m_memory, m.mem_addr, static_cast<u16>(mask), &o); old_raw = o; }
                } else if (sub_op == 6) { // BTR
                    if (op_bits == 64) handled = guest_atomic_and<u64>(m_memory, m.mem_addr, ~mask, &old_raw);
                    else if (op_bits == 32) { u32 o = 0; handled = guest_atomic_and<u32>(m_memory, m.mem_addr, static_cast<u32>(~mask), &o); old_raw = o; }
                    else { u16 o = 0; handled = guest_atomic_and<u16>(m_memory, m.mem_addr, static_cast<u16>(~mask), &o); old_raw = o; }
                } else if (sub_op == 7) { // BTC
                    if (op_bits == 64) handled = guest_atomic_xor<u64>(m_memory, m.mem_addr, mask, &old_raw);
                    else if (op_bits == 32) { u32 o = 0; handled = guest_atomic_xor<u32>(m_memory, m.mem_addr, static_cast<u32>(mask), &o); old_raw = o; }
                    else { u16 o = 0; handled = guest_atomic_xor<u16>(m_memory, m.mem_addr, static_cast<u16>(mask), &o); old_raw = o; }
                }
                if (handled >= 0) {
                    m_context.set_flag(CpuContext::FLAG_CF, ((old_raw >> bit_offset) & 1) != 0);
                    m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
                    return StepResult::Ok;
                }
            }
            u64 val = read_rm(m, op_bits);
            m_context.set_flag(CpuContext::FLAG_CF, ((val >> bit_offset) & 1) != 0);
            if (sub_op == 5) val |= (1ULL << bit_offset);
            else if (sub_op == 6) val &= ~(1ULL << bit_offset);
            else if (sub_op == 7) val ^= (1ULL << bit_offset);
            if (sub_op != 4) write_rm(m, val, op_bits);
            m_context.rip = start_rip + cursor + m.bytes_consumed + 1;
            return StepResult::Ok;
        }

        // Legacy SSE moves (0x10/0x11 MOVUPS/MOVUPD/MOVSS/MOVSD, 0x28/0x29
        // MOVAPS/MOVAPD). F3 (MOVSS) moves 4 bytes and F2 (MOVSD) 8: a scalar
        // load from memory zeroes the rest of the low 128 bits, a scalar
        // register move or store touches only the low element. Legacy SSE
        // leaves bits 255:128 of the YMM register alone.
        //
        // These used to move all 16 bytes regardless of prefix: `movsd [m], xmm`
        // wrote 8 bytes past its slot. Mono's AOT generic trampoline saves
        // xmm0-7 with movsd right below the saved GPRs, so the store of xmm7
        // overwrote the saved RAX -- the vtable of every lazily-resolved
        // virtual call -- and mono_magic_trampoline asserted `vt`.
        if (op1 == 0x10 || op1 == 0x11 || op1 == 0x28 || op1 == 0x29) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const bool scalar = (op1 == 0x10 || op1 == 0x11) && (rep_prefix == 0xF3 || rep_prefix == 0xF2);
            const usize n = !scalar ? 16 : (rep_prefix == 0xF3 ? 4 : 8);
            const bool load = (op1 == 0x10 || op1 == 0x28);
            const u8 xreg = m.reg_id & 15;
            if (load) {
                auto& dst = m_context.ymm[xreg];
                if (m.is_reg) {
                    std::memcpy(&dst, &m_context.ymm[m.rm_id & 15], n);
                } else {
                    if (scalar) dst.q[0] = dst.q[1] = 0;
                    if (!m_memory.read_bytes(m.mem_addr, &dst, n)) return StepResult::MemoryFault;
                }
            } else {
                if (m.is_reg) std::memcpy(&m_context.ymm[m.rm_id & 15], &m_context.ymm[xreg], n);
                else if (!m_memory.write_bytes(m.mem_addr, &m_context.ymm[xreg], n)) return StepResult::MemoryFault;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        if (op1 == 0x57) { // XORPS
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, 16);
            for (usize q = 0; q < 2; ++q) m_context.ymm[dest].q[q] ^= src2.q[q];
            m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
                // 0x0F 0x6F (MOVDQA/MOVDQU load), 0x0F 0x7F (MOVDQA/MOVDQU store)
        if (op1 == 0x6F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            if (m.is_reg) std::memcpy(&m_context.ymm[dest], &m_context.ymm[m.rm_id & 15], 16);
            else m_memory.read_bytes(m.mem_addr, &m_context.ymm[dest], 16);
            m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (op1 == 0x7F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 src = m.reg_id & 15;
            if (m.is_reg) std::memcpy(&m_context.ymm[m.rm_id & 15], &m_context.ymm[src], 16);
            else m_memory.write_bytes(m.mem_addr, &m_context.ymm[src], 16);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x74 (PCMPEQB), 0x0F 0x75 (PCMPEQW), 0x0F 0x76 (PCMPEQD)
        // 0x0F 0x64 (PCMPGTB), 0x0F 0x65 (PCMPGTW), 0x0F 0x66 (PCMPGTD)
        if (op1 == 0x74 || op1 == 0x75 || op1 == 0x76 ||
            op1 == 0x64 || op1 == 0x65 || op1 == 0x66) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, 16);

            CpuContext::YmmReg res{};
            if (op1 == 0x76) { // PCMPEQD
                for (usize i = 0; i < 4; ++i) {
                    res.d[i] = (m_context.ymm[dest].d[i] == src2.d[i]) ? 0xFFFFFFFFU : 0U;
                }
            } else if (op1 == 0x75) { // PCMPEQW
                for (usize i = 0; i < 8; ++i) {
                    res.w[i] = (m_context.ymm[dest].w[i] == src2.w[i]) ? 0xFFFFU : 0U;
                }
            } else if (op1 == 0x74) { // PCMPEQB
                for (usize i = 0; i < 16; ++i) {
                    res.b[i] = (m_context.ymm[dest].b[i] == src2.b[i]) ? 0xFFU : 0U;
                }
            } else if (op1 == 0x66) { // PCMPGTD
                for (usize i = 0; i < 4; ++i) {
                    res.d[i] = (static_cast<i32>(m_context.ymm[dest].d[i]) > static_cast<i32>(src2.d[i])) ? 0xFFFFFFFFU : 0U;
                }
            } else if (op1 == 0x65) { // PCMPGTW
                for (usize i = 0; i < 8; ++i) {
                    res.w[i] = (static_cast<i16>(m_context.ymm[dest].w[i]) > static_cast<i16>(src2.w[i])) ? 0xFFFFU : 0U;
                }
            } else if (op1 == 0x64) { // PCMPGTB
                for (usize i = 0; i < 16; ++i) {
                    res.b[i] = (static_cast<i8>(m_context.ymm[dest].b[i]) > static_cast<i8>(src2.b[i])) ? 0xFFU : 0U;
                }
            }
            m_context.ymm[dest] = res;
            m_context.ymm[dest].q[2] = 0;
            m_context.ymm[dest].q[3] = 0;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x6E (MOVD/MOVQ GPR -> XMM), 0x0F 0x7E (MOVD/MOVQ XMM -> GPR / XMM load), 0x0F 0xD6 (MOVQ store)
        if (op1 == 0x6E) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            m_context.ymm[dest] = CpuContext::YmmReg{};
            if (rex_w) m_context.ymm[dest].q[0] = read_rm(m, 64);
            else m_context.ymm[dest].d[0] = static_cast<u32>(read_rm(m, 32));
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (op1 == 0x7E) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            if (rep_prefix == 0xF3) { // MOVQ xmm1, xmm2/m64
                const u8 dest = m.reg_id & 15;
                u64 val = 0;
                if (m.is_reg) val = m_context.ymm[m.rm_id & 15].q[0];
                else m_memory.read_value(m.mem_addr, val);
                m_context.ymm[dest] = CpuContext::YmmReg{};
                m_context.ymm[dest].q[0] = val;
            } else { // MOVD/MOVQ r/m32/64, xmm
                const u8 src = m.reg_id & 15;
                if (rex_w) write_rm(m, m_context.ymm[src].q[0], 64);
                else write_rm(m, m_context.ymm[src].d[0], 32);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (op1 == 0xD6) { // MOVQ xmm2/m64, xmm1
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 src = m.reg_id & 15;
            if (m.is_reg) {
                m_context.ymm[m.rm_id & 15] = CpuContext::YmmReg{};
                m_context.ymm[m.rm_id & 15].q[0] = m_context.ymm[src].q[0];
            } else {
                m_memory.write_value(m.mem_addr, m_context.ymm[src].q[0]);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (op1 == 0xEF || op1 == 0xEB || op1 == 0xDB || op1 == 0xDF) { // PXOR / POR / PAND / PANDN
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            CpuContext::YmmReg src2{};
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, 16);
            for (usize q = 0; q < 2; ++q) {
                if (op1 == 0xEF) m_context.ymm[dest].q[q] ^= src2.q[q];
                else if (op1 == 0xEB) m_context.ymm[dest].q[q] |= src2.q[q];
                else if (op1 == 0xDB) m_context.ymm[dest].q[q] &= src2.q[q];
                else m_context.ymm[dest].q[q] = (~m_context.ymm[dest].q[q]) & src2.q[q];
            }
            m_context.ymm[dest].q[2] = 0; m_context.ymm[dest].q[3] = 0;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x2A: CVTSI2SS (rep=0xF3) / CVTSI2SD (rep=0xF2)
        if (op1 == 0x2A) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            if (rep_prefix == 0xF2) { // CVTSI2SD: integer to double
                double dval = 0.0;
                if (rex_w) {
                    const i64 ival = static_cast<i64>(read_rm(m, 64));
                    dval = static_cast<double>(ival);
                } else {
                    const i32 ival = static_cast<i32>(read_rm(m, 32));
                    dval = static_cast<double>(ival);
                }
                std::memcpy(&m_context.ymm[dest].q[0], &dval, 8);
            } else { // CVTSI2SS: integer to float
                float fval = 0.0f;
                if (rex_w) {
                    const i64 ival = static_cast<i64>(read_rm(m, 64));
                    fval = static_cast<float>(ival);
                } else {
                    const i32 ival = static_cast<i32>(read_rm(m, 32));
                    fval = static_cast<float>(ival);
                }
                std::memcpy(&m_context.ymm[dest].d[0], &fval, 4);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x2C: CVTTSS2SI (rep=0xF3) / CVTTSD2SI (rep=0xF2)
        if (op1 == 0x2C) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            if (rep_prefix == 0xF2) { // CVTTSD2SI
                double dval = 0.0;
                if (m.is_reg) std::memcpy(&dval, &m_context.ymm[m.rm_id & 15].q[0], 8);
                else m_memory.read_bytes(m.mem_addr, &dval, 8);
                if (rex_w) set_gpr64(m_context, m.reg_id, x86_cvt_i64(dval));
                else set_gpr32(m_context, m.reg_id, x86_cvt_i32(dval));
            } else { // CVTTSS2SI
                float fval = 0.0f;
                if (m.is_reg) std::memcpy(&fval, &m_context.ymm[m.rm_id & 15].d[0], 4);
                else m_memory.read_bytes(m.mem_addr, &fval, 4);
                if (rex_w) set_gpr64(m_context, m.reg_id, x86_cvt_i64(fval));
                else set_gpr32(m_context, m.reg_id, x86_cvt_i32(fval));
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x2E (UCOMISS), 0x0F 0x2F (COMISS / UCOMISD / COMISD)
        if (op1 == 0x2E || op1 == 0x2F) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 src1 = m.reg_id & 15;
            CpuContext::YmmReg src2{};
            const bool is_double = (operand_size_16 || rep_prefix == 0x66);
            if (m.is_reg) src2 = m_context.ymm[m.rm_id & 15];
            else m_memory.read_bytes(m.mem_addr, &src2, is_double ? 8 : 4);

            m_context.rflags &= ~(CpuContext::FLAG_ZF | CpuContext::FLAG_PF | CpuContext::FLAG_CF | CpuContext::FLAG_OF | CpuContext::FLAG_SF | CpuContext::FLAG_AF);
            if (is_double) {
                double d1 = 0.0, d2 = 0.0;
                std::memcpy(&d1, &m_context.ymm[src1].q[0], 8);
                std::memcpy(&d2, &src2.q[0], 8);
                if (std::isnan(d1) || std::isnan(d2)) {
                    m_context.rflags |= (CpuContext::FLAG_ZF | CpuContext::FLAG_PF | CpuContext::FLAG_CF);
                } else if (d1 < d2) {
                    m_context.rflags |= CpuContext::FLAG_CF;
                } else if (d1 == d2) {
                    m_context.rflags |= CpuContext::FLAG_ZF;
                }
            } else {
                float f1 = 0.0f, f2 = 0.0f;
                std::memcpy(&f1, &m_context.ymm[src1].d[0], 4);
                std::memcpy(&f2, &src2.d[0], 4);
                if (std::isnan(f1) || std::isnan(f2)) {
                    m_context.rflags |= (CpuContext::FLAG_ZF | CpuContext::FLAG_PF | CpuContext::FLAG_CF);
                } else if (f1 < f2) {
                    m_context.rflags |= CpuContext::FLAG_CF;
                } else if (f1 == f2) {
                    m_context.rflags |= CpuContext::FLAG_ZF;
                }
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0x5A: CVTSS2SD / CVTSD2SS
        if (op1 == 0x5A) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 dest = m.reg_id & 15;
            if (rep_prefix == 0xF2) { // CVTSD2SS
                double dval = 0.0;
                if (m.is_reg) std::memcpy(&dval, &m_context.ymm[m.rm_id & 15].q[0], 8);
                else m_memory.read_bytes(m.mem_addr, &dval, 8);
                const float fval = static_cast<float>(dval);
                std::memcpy(&m_context.ymm[dest].d[0], &fval, 4);
            } else { // CVTSS2SD
                float fval = 0.0f;
                if (m.is_reg) std::memcpy(&fval, &m_context.ymm[m.rm_id & 15].d[0], 4);
                else m_memory.read_bytes(m.mem_addr, &fval, 4);
                const double dval = static_cast<double>(fval);
                std::memcpy(&m_context.ymm[dest].q[0], &dval, 8);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }

        // 0x0F 0xC7: Group 9 (CMPXCHG8B / CMPXCHG16B / RDRAND)
        if (op1 == 0xC7) {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (!m.valid) return StepResult::IllegalInstruction;
            const u8 sub_op = m.reg_id & 7;
            if (sub_op == 1) { // CMPXCHG8B / CMPXCHG16B
                if (m.is_reg) return StepResult::IllegalInstruction;
                std::lock_guard<std::mutex> lock(get_atomic_mutex(m.mem_addr));
                if (rex_w) {
                    // CMPXCHG16B m128: compares RDX:RAX with m128
                    u64 mem_low = 0, mem_high = 0;
                    if (!m_memory.read_value(m.mem_addr, mem_low) ||
                        !m_memory.read_value(m.mem_addr + 8, mem_high)) {
                        return StepResult::MemoryFault;
                    }
                    if (m_context.rax == mem_low && m_context.rdx == mem_high) {
                        m_context.set_flag(CpuContext::FLAG_ZF, true);
                        if (!m_memory.write_value(m.mem_addr, m_context.rbx) ||
                            !m_memory.write_value(m.mem_addr + 8, m_context.rcx)) {
                            return StepResult::MemoryFault;
                        }
                    } else {
                        m_context.set_flag(CpuContext::FLAG_ZF, false);
                        m_context.rax = mem_low;
                        m_context.rdx = mem_high;
                    }
                } else {
                    // CMPXCHG8B m64: compares EDX:EAX with m64
                    u64 mem_val = 0;
                    if (!m_memory.read_value(m.mem_addr, mem_val)) {
                        return StepResult::MemoryFault;
                    }
                    const u32 mem_low = static_cast<u32>(mem_val);
                    const u32 mem_high = static_cast<u32>(mem_val >> 32);
                    if (static_cast<u32>(m_context.rax) == mem_low && static_cast<u32>(m_context.rdx) == mem_high) {
                        m_context.set_flag(CpuContext::FLAG_ZF, true);
                        const u64 write_val = static_cast<u32>(m_context.rbx) | (static_cast<u64>(static_cast<u32>(m_context.rcx)) << 32);
                        if (!m_memory.write_value(m.mem_addr, write_val)) {
                            return StepResult::MemoryFault;
                        }
                    } else {
                        m_context.set_flag(CpuContext::FLAG_ZF, false);
                        m_context.rax = mem_low;
                        m_context.rdx = mem_high;
                    }
                }
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            } else if (sub_op == 6) { // RDRAND
                static u64 rand_seed = 0x123456789abcdef0ULL;
                rand_seed = rand_seed * 6364136223846793005ULL + 1442695040888963407ULL;
                if (op_bits == 64) set_gpr64(m_context, m.rm_id, rand_seed);
                else if (op_bits == 32) set_gpr32(m_context, m.rm_id, static_cast<u32>(rand_seed));
                else set_gpr16(m_context, m.rm_id, static_cast<u16>(rand_seed));
                m_context.set_flag(CpuContext::FLAG_CF, true);
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }
        }

        // 0x0F 0xC8+r: BSWAP r32/r64 (no ModR/M). It used to fall through to
        // the ModR/M-skipping fallback below, which "executed" it as a NOP and
        // swallowed the next instruction bytes -- in mono-ps4's SHA-1 that
        // skipped a MOVBE store and then landed RIP mid-instruction.
        if (op1 >= 0xC8 && op1 <= 0xCF) {
            const u8 reg = static_cast<u8>((op1 - 0xC8) | (rex_b ? 8 : 0));
            if (rex_w) {
                set_gpr64(m_context, reg, __builtin_bswap64(get_gpr64(m_context, reg)));
            } else if (operand_size_16) {
                set_gpr16(m_context, reg, 0); // undefined; zero as on Intel hardware
            } else {
                set_gpr32(m_context, reg, __builtin_bswap32(static_cast<u32>(get_gpr64(m_context, reg))));
            }
            m_context.rip = start_rip + cursor;
            return StepResult::Ok;
        }

        // Generic fallback for any other 0x0F opcode with ModR/M: skipped as a
        // NOP (correct for hints such as 0F 18-1F / 0F 0D). Anything else
        // skipped here is unimplemented and silently wrong, so name each
        // opcode the first time it is skipped.
        {
            auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base_0f, rex_r, rex_x, rex_b, seg_override, m_context);
            if (m.valid) {
                const bool hint = (op1 >= 0x18 && op1 <= 0x1F) || op1 == 0x0D;
                static std::atomic<u64> s_reported[4]{};
                const u64 bit = 1ULL << (op1 & 63);
                if (!hint && !(s_reported[op1 >> 6].fetch_or(bit) & bit)) {
                    std::fprintf(stderr, "[CPU] 0F %02X at RIP=0x%llx is unimplemented; skipped as a NOP\n",
                                 op1, (unsigned long long)start_rip);
                }
                m_context.rip = start_rip + cursor + m.bytes_consumed;
                return StepResult::Ok;
            }
        }
        std::fprintf(stderr, "[CPU] Unhandled 0x0F opcode: 0x%02x at RIP=0x%llx\n", op1, (unsigned long long)start_rip);
        return StepResult::IllegalInstruction;
    }

    // 0x63: MOVSXD reg64, r/m32
    if (op0 == 0x63) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        const i32 val32 = static_cast<i32>(read_rm(m, 32));
        set_gpr64(m_context, m.reg_id, static_cast<u64>(static_cast<i64>(val32)));
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0x69 / 0x6B: IMUL reg, r/m, imm32 / imm8
    if (op0 == 0x69 || op0 == 0x6B) {
        const usize imm_sz = (op0 == 0x6B) ? 1 : (op_bits == 16 ? 2 : 4);
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context, imm_sz);
        if (!m.valid) return StepResult::IllegalInstruction;
        if (cursor + m.bytes_consumed + imm_sz > 15) return StepResult::IllegalInstruction;

        i64 imm = 0;
        if (op0 == 0x6B) {
            imm = static_cast<i64>(static_cast<i8>(raw_code[cursor + m.bytes_consumed]));
        } else if (op_bits == 16) {
            i16 imm16 = 0;
            std::memcpy(&imm16, raw_code + cursor + m.bytes_consumed, 2);
            imm = static_cast<i64>(imm16);
        } else {
            i32 imm32 = 0;
            std::memcpy(&imm32, raw_code + cursor + m.bytes_consumed, 4);
            imm = static_cast<i64>(imm32);
        }

        if (op_bits == 64) {
            const i64 src = static_cast<i64>(read_rm(m, 64));
            const __int128 full = static_cast<__int128>(src) * static_cast<__int128>(imm);
            const i64 res = static_cast<i64>(full);
            set_gpr64(m_context, m.reg_id, static_cast<u64>(res));
            const bool overflow = (full != res);
            m_context.set_flag(CpuContext::FLAG_CF, overflow);
            m_context.set_flag(CpuContext::FLAG_OF, overflow);
        } else if (op_bits == 32) {
            const i32 src = static_cast<i32>(read_rm(m, 32));
            const i64 full = static_cast<i64>(src) * static_cast<i64>(imm);
            const i32 res = static_cast<i32>(full);
            set_gpr32(m_context, m.reg_id, static_cast<u32>(res));
            const bool overflow = (full != res);
            m_context.set_flag(CpuContext::FLAG_CF, overflow);
            m_context.set_flag(CpuContext::FLAG_OF, overflow);
        } else {
            const i16 src = static_cast<i16>(read_rm(m, 16));
            const i32 full = static_cast<i32>(src) * static_cast<i32>(imm);
            const i16 res = static_cast<i16>(full);
            set_gpr16(m_context, m.reg_id, static_cast<u16>(res));
            const bool overflow = (full != res);
            m_context.set_flag(CpuContext::FLAG_CF, overflow);
            m_context.set_flag(CpuContext::FLAG_OF, overflow);
        }

        m_context.rip = start_rip + cursor + m.bytes_consumed + imm_sz;
        return StepResult::Ok;
    }

    // ALU Instructions with standard ModR/M
    // 0x00 - 0x03: ADD
    // 0x08 - 0x0B: OR
    // 0x10 - 0x13: ADC
    // 0x18 - 0x1B: SBB
    // 0x20 - 0x23: AND
    // 0x28 - 0x2B: SUB
    // 0x30 - 0x33: XOR
    // 0x38 - 0x3B: CMP
    if (op0 <= 0x3B && (op0 & 4) == 0) {
        const u8 alu_major = (op0 >> 3) & 7;
        const u8 alu_dir   = op0 & 3;
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        const usize bits = (alu_dir & 1) ? op_bits : 8;
        if ((alu_dir & 2) == 0) {
            // dir 0/1: OP r/m, reg
            const u64 src = (bits == 8) ? get_gpr8(m_context, m.reg_id, has_rex) : (bits == 64) ? get_gpr64(m_context, m.reg_id) : static_cast<u32>(get_gpr64(m_context, m.reg_id));
            if (!m.is_reg && lock_prefix && alu_major != 7 /* not CMP */) {
                int handled = -1;
                u64 old_raw = 0;
                if (alu_major == 0) { // ADD
                    if (bits == 64) handled = guest_atomic_xadd<u64>(m_memory, m.mem_addr, src, &old_raw);
                    else if (bits == 32) { u32 o = 0; handled = guest_atomic_xadd<u32>(m_memory, m.mem_addr, static_cast<u32>(src), &o); old_raw = o; }
                    else if (bits == 16) { u16 o = 0; handled = guest_atomic_xadd<u16>(m_memory, m.mem_addr, static_cast<u16>(src), &o); old_raw = o; }
                    else { u8 o = 0; handled = guest_atomic_xadd<u8>(m_memory, m.mem_addr, static_cast<u8>(src), &o); old_raw = o; }
                } else if (alu_major == 1) { // OR
                    if (bits == 64) handled = guest_atomic_or<u64>(m_memory, m.mem_addr, src, &old_raw);
                    else if (bits == 32) { u32 o = 0; handled = guest_atomic_or<u32>(m_memory, m.mem_addr, static_cast<u32>(src), &o); old_raw = o; }
                    else if (bits == 16) { u16 o = 0; handled = guest_atomic_or<u16>(m_memory, m.mem_addr, static_cast<u16>(src), &o); old_raw = o; }
                    else { u8 o = 0; handled = guest_atomic_or<u8>(m_memory, m.mem_addr, static_cast<u8>(src), &o); old_raw = o; }
                } else if (alu_major == 4) { // AND
                    if (bits == 64) handled = guest_atomic_and<u64>(m_memory, m.mem_addr, src, &old_raw);
                    else if (bits == 32) { u32 o = 0; handled = guest_atomic_and<u32>(m_memory, m.mem_addr, static_cast<u32>(src), &o); old_raw = o; }
                    else if (bits == 16) { u16 o = 0; handled = guest_atomic_and<u16>(m_memory, m.mem_addr, static_cast<u16>(src), &o); old_raw = o; }
                    else { u8 o = 0; handled = guest_atomic_and<u8>(m_memory, m.mem_addr, static_cast<u8>(src), &o); old_raw = o; }
                } else if (alu_major == 5) { // SUB
                    if (bits == 64) handled = guest_atomic_xadd<u64>(m_memory, m.mem_addr, static_cast<u64>(0 - src), &old_raw);
                    else if (bits == 32) { u32 o = 0; handled = guest_atomic_xadd<u32>(m_memory, m.mem_addr, static_cast<u32>(0 - src), &o); old_raw = o; }
                    else if (bits == 16) { u16 o = 0; handled = guest_atomic_xadd<u16>(m_memory, m.mem_addr, static_cast<u16>(0 - src), &o); old_raw = o; }
                    else { u8 o = 0; handled = guest_atomic_xadd<u8>(m_memory, m.mem_addr, static_cast<u8>(0 - src), &o); old_raw = o; }
                } else if (alu_major == 6) { // XOR
                    if (bits == 64) handled = guest_atomic_xor<u64>(m_memory, m.mem_addr, src, &old_raw);
                    else if (bits == 32) { u32 o = 0; handled = guest_atomic_xor<u32>(m_memory, m.mem_addr, static_cast<u32>(src), &o); old_raw = o; }
                    else if (bits == 16) { u16 o = 0; handled = guest_atomic_xor<u16>(m_memory, m.mem_addr, static_cast<u16>(src), &o); old_raw = o; }
                    else { u8 o = 0; handled = guest_atomic_xor<u8>(m_memory, m.mem_addr, static_cast<u8>(src), &o); old_raw = o; }
                }
                if (handled >= 0) {
                    const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
                    const u64 dst = old_raw & mask;
                    execute_alu_op(m_context, alu_major, dst, src, bits);
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u64 dst = read_rm(m, bits);
            const u64 res = execute_alu_op(m_context, alu_major, dst, src, bits);
            if (alu_major != 7) write_rm(m, res, bits); // CMP doesn't write back
        } else {
            // dir 2/3: OP reg, r/m
            const u64 src = read_rm(m, bits);
            const u64 dst = (bits == 8) ? get_gpr8(m_context, m.reg_id, has_rex) : (bits == 64) ? get_gpr64(m_context, m.reg_id) : static_cast<u32>(get_gpr64(m_context, m.reg_id));
            const u64 res = execute_alu_op(m_context, alu_major, dst, src, bits);
            if (alu_major != 7) {
                if (bits == 64) set_gpr64(m_context, m.reg_id, res);
                else if (bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(res));
                else if (bits == 16) set_gpr16(m_context, m.reg_id, static_cast<u16>(res));
                else set_gpr8(m_context, m.reg_id, has_rex, static_cast<u8>(res));
            }
        }
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0x04, 0x05, 0x0C, 0x0D, 0x14, 0x15, 0x1C, 0x1D, 0x24, 0x25, 0x2C, 0x2D, 0x34, 0x35, 0x3C, 0x3D:
    // ALU AL/AX/EAX/RAX, imm
    if (op0 <= 0x3D && ((op0 & 7) == 4 || (op0 & 7) == 5)) {
        const u8 alu_op = (op0 >> 3) & 7;
        const bool is_8bit = ((op0 & 1) == 0);
        if (is_8bit) {
            const u8 imm8 = raw_code[cursor++];
            const u64 dst = m_context.rax & 0xFF;
            const u64 res = execute_alu_op(m_context, alu_op, dst, imm8, 8);
            if (alu_op != 7) m_context.rax = (m_context.rax & ~0xFFULL) | (res & 0xFF);
        } else if (op_bits == 16) {
            u16 imm16 = 0;
            std::memcpy(&imm16, raw_code + cursor, 2);
            cursor += 2;
            const u64 dst = m_context.rax & 0xFFFF;
            const u64 res = execute_alu_op(m_context, alu_op, dst, imm16, 16);
            if (alu_op != 7) m_context.rax = (m_context.rax & ~0xFFFFULL) | (res & 0xFFFF);
        } else {
            i32 imm32 = 0;
            std::memcpy(&imm32, raw_code + cursor, 4);
            cursor += 4;
            const u64 imm = (op_bits == 64) ? static_cast<u64>(static_cast<i64>(imm32)) : static_cast<u32>(imm32);
            const u64 dst = (op_bits == 64) ? m_context.rax : static_cast<u32>(m_context.rax);
            const u64 res = execute_alu_op(m_context, alu_op, dst, imm, op_bits);
            if (alu_op != 7) {
                if (op_bits == 64) m_context.rax = res;
                else m_context.rax = static_cast<u32>(res);
            }
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0xA8 / 0xA9: TEST AL/AX/EAX/RAX, imm
    if (op0 == 0xA8 || op0 == 0xA9) {
        if (op0 == 0xA8) {
            const u8 imm8 = raw_code[cursor++];
            update_flags_logic(m_context, (m_context.rax & 0xFF) & imm8, 8);
        } else if (op_bits == 16) {
            u16 imm16 = 0;
            std::memcpy(&imm16, raw_code + cursor, 2);
            cursor += 2;
            update_flags_logic(m_context, (m_context.rax & 0xFFFF) & imm16, 16);
        } else {
            u32 imm32 = 0;
            std::memcpy(&imm32, raw_code + cursor, 4);
            cursor += 4;
            const u64 imm = (op_bits == 64) ? static_cast<u64>(static_cast<i64>(static_cast<i32>(imm32))) : imm32;
            const u64 dst = (op_bits == 64) ? m_context.rax : static_cast<u32>(m_context.rax);
            update_flags_logic(m_context, dst & imm, op_bits);
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0xA4: MOVSB, 0xA5: MOVSW/MOVSD/MOVSQ
    if (op0 == 0xA4 || op0 == 0xA5) {
        const usize size = (op0 == 0xA4) ? 1 : rex_w ? 8 : operand_size_16 ? 2 : 4;
        const bool df = m_context.flag_df();
        const i64 step = df ? -static_cast<i64>(size) : static_cast<i64>(size);

        if (rep_prefix == 0xF3) {
            u64 count = m_context.rcx;
            while (count > 0) {
                u64 val = 0;
                if (!m_memory.read_bytes(m_context.rsi, &val, size)) return StepResult::MemoryFault;
                if (!m_memory.write_bytes(m_context.rdi, &val, size)) return StepResult::MemoryFault;
                m_context.rsi += step;
                m_context.rdi += step;
                count--;
            }
            m_context.rcx = 0;
        } else {
            u64 val = 0;
            if (!m_memory.read_bytes(m_context.rsi, &val, size)) return StepResult::MemoryFault;
            if (!m_memory.write_bytes(m_context.rdi, &val, size)) return StepResult::MemoryFault;
            m_context.rsi += step;
            m_context.rdi += step;
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0xAA: STOSB, 0xAB: STOSW/STOSD/STOSQ
    if (op0 == 0xAA || op0 == 0xAB) {
        const usize size = (op0 == 0xAA) ? 1 : rex_w ? 8 : operand_size_16 ? 2 : 4;
        const bool df = m_context.flag_df();
        const i64 step = df ? -static_cast<i64>(size) : static_cast<i64>(size);
        const u64 val = m_context.rax;

        if (rep_prefix == 0xF3) {
            u64 count = m_context.rcx;
            while (count > 0) {
                if (!m_memory.write_bytes(m_context.rdi, &val, size)) return StepResult::MemoryFault;
                m_context.rdi += step;
                count--;
            }
            m_context.rcx = 0;
        } else {
            if (!m_memory.write_bytes(m_context.rdi, &val, size)) return StepResult::MemoryFault;
            m_context.rdi += step;
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0xAC: LODSB, 0xAD: LODSW/LODSD/LODSQ
    if (op0 == 0xAC || op0 == 0xAD) {
        const usize size = (op0 == 0xAC) ? 1 : rex_w ? 8 : operand_size_16 ? 2 : 4;
        const bool df = m_context.flag_df();
        const i64 step = df ? -static_cast<i64>(size) : static_cast<i64>(size);

        u64 val = 0;
        if (!m_memory.read_bytes(m_context.rsi, &val, size)) return StepResult::MemoryFault;
        if (size == 1) set_gpr8(m_context, 0, has_rex, static_cast<u8>(val));
        else if (size == 2) set_gpr16(m_context, 0, static_cast<u16>(val));
        else if (size == 4) set_gpr32(m_context, 0, static_cast<u32>(val));
        else set_gpr64(m_context, 0, val);
        m_context.rsi += step;
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0xAE: SCASB, 0xAF: SCASW/SCASD/SCASQ
    if (op0 == 0xAE || op0 == 0xAF) {
        const usize size = (op0 == 0xAE) ? 1 : rex_w ? 8 : operand_size_16 ? 2 : 4;
        const bool df = m_context.flag_df();
        const i64 step = df ? -static_cast<i64>(size) : static_cast<i64>(size);
        const u64 acc = m_context.rax;

        if (rep_prefix == 0xF2 || rep_prefix == 0xF3) { // REPNZ / REPZ
            u64 count = m_context.rcx;
            while (count > 0) {
                u64 mem_val = 0;
                if (!m_memory.read_bytes(m_context.rdi, &mem_val, size)) return StepResult::MemoryFault;
                update_flags_sub(m_context, acc, mem_val, acc - mem_val, size * 8);
                m_context.rdi += step;
                count--;
                const bool zf = m_context.flag_zf();
                if (rep_prefix == 0xF3 && !zf) break;
                if (rep_prefix == 0xF2 && zf) break;
            }
            m_context.rcx = count;
        } else {
            u64 mem_val = 0;
            if (!m_memory.read_bytes(m_context.rdi, &mem_val, size)) return StepResult::MemoryFault;
            update_flags_sub(m_context, acc, mem_val, acc - mem_val, size * 8);
            m_context.rdi += step;
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0x86 / 0x87: XCHG r/m, reg
    if (op0 == 0x86 || op0 == 0x87) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        const usize bits = (op0 == 0x86) ? 8 : op_bits;
        const u64 reg_val = (bits == 8) ? get_gpr8(m_context, m.reg_id, has_rex) :
                            (bits == 64) ? get_gpr64(m_context, m.reg_id) :
                            static_cast<u32>(get_gpr64(m_context, m.reg_id));
        u64 rm_val = 0;
        int handled = -1;
        if (!m.is_reg) {
            // Atomic exchange; see guest_atomic_xchg. XCHG is implicitly locked
            // on x86 even without a 0xF0 prefix.
            u64 old_raw = 0;
            if (bits == 64) handled = guest_atomic_xchg<u64>(m_memory, m.mem_addr, static_cast<u64>(reg_val), reinterpret_cast<u64*>(&old_raw));
            else if (bits == 32) handled = guest_atomic_xchg<u32>(m_memory, m.mem_addr, static_cast<u32>(reg_val), reinterpret_cast<u32*>(&old_raw));
            else if (bits == 16) handled = guest_atomic_xchg<u16>(m_memory, m.mem_addr, static_cast<u16>(reg_val), reinterpret_cast<u16*>(&old_raw));
            else handled = guest_atomic_xchg<u8>(m_memory, m.mem_addr, static_cast<u8>(reg_val), reinterpret_cast<u8*>(&old_raw));
            if (handled >= 0) rm_val = old_raw;
        }
        if (handled < 0) {
            rm_val = read_rm(m, bits);
            write_rm(m, reg_val, bits);
        }
        if (bits == 8) set_gpr8(m_context, m.reg_id, has_rex, static_cast<u8>(rm_val));
        else if (bits == 64) set_gpr64(m_context, m.reg_id, rm_val);
        else if (bits == 32) set_gpr32(m_context, m.reg_id, static_cast<u32>(rm_val));
        else set_gpr16(m_context, m.reg_id, static_cast<u16>(rm_val));
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0x91 - 0x97: XCHG EAX/RAX, reg
    if (op0 >= 0x91 && op0 <= 0x97) {
        const u8 reg = (rex_b ? 8 : 0) | (op0 - 0x90);
        if (op_bits == 64) {
            std::swap(m_context.rax, get_gpr64(m_context, reg));
        } else if (op_bits == 32) {
            const u32 tmp = static_cast<u32>(m_context.rax);
            m_context.rax = static_cast<u32>(get_gpr64(m_context, reg));
            set_gpr32(m_context, reg, tmp);
        } else {
            const u16 tmp = static_cast<u16>(m_context.rax);
            set_gpr16(m_context, 0, static_cast<u16>(get_gpr64(m_context, reg)));
            set_gpr16(m_context, reg, tmp);
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0x84 / 0x85: TEST r/m, reg
    if (op0 == 0x84 || op0 == 0x85) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        const usize bits = (op0 == 0x84) ? 8 : op_bits;
        const u64 src = (bits == 8) ? get_gpr8(m_context, m.reg_id, has_rex) : (bits == 64) ? get_gpr64(m_context, m.reg_id) : static_cast<u32>(get_gpr64(m_context, m.reg_id));
        const u64 dst = read_rm(m, bits);
        const u64 res = (dst & src);
        update_flags_logic(m_context, res, bits);
        m_context.rip = start_rip + cursor + m.bytes_consumed;
        return StepResult::Ok;
    }

    // 0x80 / 0x81 / 0x83: ALU r/m, imm
    if (op0 == 0x80 || op0 == 0x81 || op0 == 0x83) {
        const usize imm_sz = (op0 == 0x80 || op0 == 0x83) ? 1 : (op_bits == 16 ? 2 : 4);
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context, imm_sz);
        if (!m.valid) return StepResult::IllegalInstruction;
        const u8 alu_op = m.reg_id & 7;
        const usize bits = (op0 == 0x80) ? 8 : op_bits;

        u64 imm = 0;
        if (op0 == 0x83) {
            const i8 imm8 = static_cast<i8>(raw_code[cursor + m.bytes_consumed]);
            imm = (bits == 64) ? static_cast<u64>(static_cast<i64>(imm8)) : static_cast<u32>(static_cast<i32>(imm8));
        } else if (op0 == 0x80) {
            imm = raw_code[cursor + m.bytes_consumed];
        } else if (op_bits == 16) {
            u16 imm16 = 0;
            std::memcpy(&imm16, raw_code + cursor + m.bytes_consumed, 2);
            imm = imm16;
        } else {
            i32 imm32 = 0;
            std::memcpy(&imm32, raw_code + cursor + m.bytes_consumed, 4);
            imm = (bits == 64) ? static_cast<u64>(static_cast<i64>(imm32)) : static_cast<u32>(imm32);
        }

        if (!m.is_reg && lock_prefix && alu_op != 7 /* not CMP */) {
            int handled = -1;
            u64 old_raw = 0;
            if (alu_op == 0) { // ADD
                if (bits == 64) handled = guest_atomic_xadd<u64>(m_memory, m.mem_addr, imm, &old_raw);
                else if (bits == 32) { u32 o = 0; handled = guest_atomic_xadd<u32>(m_memory, m.mem_addr, static_cast<u32>(imm), &o); old_raw = o; }
                else if (bits == 16) { u16 o = 0; handled = guest_atomic_xadd<u16>(m_memory, m.mem_addr, static_cast<u16>(imm), &o); old_raw = o; }
                else { u8 o = 0; handled = guest_atomic_xadd<u8>(m_memory, m.mem_addr, static_cast<u8>(imm), &o); old_raw = o; }
            } else if (alu_op == 1) { // OR
                if (bits == 64) handled = guest_atomic_or<u64>(m_memory, m.mem_addr, imm, &old_raw);
                else if (bits == 32) { u32 o = 0; handled = guest_atomic_or<u32>(m_memory, m.mem_addr, static_cast<u32>(imm), &o); old_raw = o; }
                else if (bits == 16) { u16 o = 0; handled = guest_atomic_or<u16>(m_memory, m.mem_addr, static_cast<u16>(imm), &o); old_raw = o; }
                else { u8 o = 0; handled = guest_atomic_or<u8>(m_memory, m.mem_addr, static_cast<u8>(imm), &o); old_raw = o; }
            } else if (alu_op == 4) { // AND
                if (bits == 64) handled = guest_atomic_and<u64>(m_memory, m.mem_addr, imm, &old_raw);
                else if (bits == 32) { u32 o = 0; handled = guest_atomic_and<u32>(m_memory, m.mem_addr, static_cast<u32>(imm), &o); old_raw = o; }
                else if (bits == 16) { u16 o = 0; handled = guest_atomic_and<u16>(m_memory, m.mem_addr, static_cast<u16>(imm), &o); old_raw = o; }
                else { u8 o = 0; handled = guest_atomic_and<u8>(m_memory, m.mem_addr, static_cast<u8>(imm), &o); old_raw = o; }
            } else if (alu_op == 5) { // SUB
                if (bits == 64) handled = guest_atomic_xadd<u64>(m_memory, m.mem_addr, static_cast<u64>(0 - imm), &old_raw);
                else if (bits == 32) { u32 o = 0; handled = guest_atomic_xadd<u32>(m_memory, m.mem_addr, static_cast<u32>(0 - imm), &o); old_raw = o; }
                else if (bits == 16) { u16 o = 0; handled = guest_atomic_xadd<u16>(m_memory, m.mem_addr, static_cast<u16>(0 - imm), &o); old_raw = o; }
                else { u8 o = 0; handled = guest_atomic_xadd<u8>(m_memory, m.mem_addr, static_cast<u8>(0 - imm), &o); old_raw = o; }
            } else if (alu_op == 6) { // XOR
                if (bits == 64) handled = guest_atomic_xor<u64>(m_memory, m.mem_addr, imm, &old_raw);
                else if (bits == 32) { u32 o = 0; handled = guest_atomic_xor<u32>(m_memory, m.mem_addr, static_cast<u32>(imm), &o); old_raw = o; }
                else if (bits == 16) { u16 o = 0; handled = guest_atomic_xor<u16>(m_memory, m.mem_addr, static_cast<u16>(imm), &o); old_raw = o; }
                else { u8 o = 0; handled = guest_atomic_xor<u8>(m_memory, m.mem_addr, static_cast<u8>(imm), &o); old_raw = o; }
            }
            if (handled >= 0) {
                const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
                const u64 dst = old_raw & mask;
                execute_alu_op(m_context, alu_op, dst, imm, bits);
                m_context.rip = start_rip + cursor + m.bytes_consumed + imm_sz;
                return StepResult::Ok;
            }
        }

        const u64 dst = read_rm(m, bits);
        const u64 res = execute_alu_op(m_context, alu_op, dst, imm, bits);
        if (alu_op != 7) write_rm(m, res, bits); // CMP doesn't write back
        m_context.rip = start_rip + cursor + m.bytes_consumed + imm_sz;
        return StepResult::Ok;
    }

    // 0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3: Group 2 Shift and Rotate instructions
    if (op0 == 0xC0 || op0 == 0xC1 || op0 == 0xD0 || op0 == 0xD1 || op0 == 0xD2 || op0 == 0xD3) {
        const bool is_imm8 = (op0 == 0xC0 || op0 == 0xC1);
        const bool is_one  = (op0 == 0xD0 || op0 == 0xD1);
        const usize bits   = (op0 == 0xC0 || op0 == 0xD0 || op0 == 0xD2) ? 8 : op_bits;
        const usize imm_sz = is_imm8 ? 1 : 0;

        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context, imm_sz);
        if (!m.valid) return StepResult::IllegalInstruction;
        const u8 sub_op = m.reg_id & 7;

        u8 count = 0;
        if (is_imm8) {
            count = raw_code[cursor + m.bytes_consumed];
        } else if (is_one) {
            count = 1;
        } else {
            count = static_cast<u8>(m_context.rcx & 0xFF);
        }
        count &= (bits == 64) ? 0x3F : 0x1F;

        const u64 val = read_rm(m, bits);
        const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
        u64 res = val;

        if (count > 0) {
            const u64 msb = 1ULL << (bits - 1);
            const bool cf_in = m_context.flag_cf();
            // OF is defined only for 1-bit shifts/rotates. For larger counts it
            // is undefined: shifts clear it and rotates leave it, matching the
            // JIT's translation of shifts.
            const bool one = (count == 1);
            if (sub_op == 0) { // ROL
                const u8 cnt = count % bits;
                res = cnt ? (((val << cnt) | (val >> (bits - cnt))) & mask) : val;
                const bool cf = (res & 1) != 0;
                m_context.set_flag(CpuContext::FLAG_CF, cf);
                if (one) m_context.set_flag(CpuContext::FLAG_OF, ((res & msb) != 0) != cf);
            } else if (sub_op == 1) { // ROR
                const u8 cnt = count % bits;
                res = cnt ? (((val >> cnt) | (val << (bits - cnt))) & mask) : val;
                m_context.set_flag(CpuContext::FLAG_CF, (res & msb) != 0);
                if (one) m_context.set_flag(CpuContext::FLAG_OF, ((res & msb) != 0) != ((res & (msb >> 1)) != 0));
            } else if (sub_op == 2 || sub_op == 3) { // RCL / RCR: rotate through CF
                // Effective count: masked count mod (bits + 1) for 8/16-bit.
                const u8 cnt = (bits == 8) ? count % 9 : (bits == 16) ? count % 17 : count;
                bool cf = cf_in;
                res = val & mask;
                for (u8 k = 0; k < cnt; ++k) {
                    if (sub_op == 2) {
                        const bool out = (res & msb) != 0;
                        res = ((res << 1) | (cf ? 1 : 0)) & mask;
                        cf = out;
                    } else {
                        const bool out = (res & 1) != 0;
                        res = (res >> 1) | (cf ? msb : 0);
                        cf = out;
                    }
                }
                m_context.set_flag(CpuContext::FLAG_CF, cf);
                if (one && sub_op == 2) m_context.set_flag(CpuContext::FLAG_OF, ((res & msb) != 0) != cf);
                if (one && sub_op == 3) m_context.set_flag(CpuContext::FLAG_OF, ((res & msb) != 0) != ((res & (msb >> 1)) != 0));
            } else if (sub_op == 4 || sub_op == 6) { // SHL / SAL
                res = (count < bits) ? ((val << count) & mask) : 0;
                const bool cf = (count <= bits) ? (((val >> (bits - count)) & 1) != 0) : false;
                update_flags_logic(m_context, res, bits);
                m_context.set_flag(CpuContext::FLAG_CF, cf);
                m_context.set_flag(CpuContext::FLAG_OF, one && (((res & msb) != 0) != cf));
            } else if (sub_op == 5) { // SHR
                res = (count < bits) ? ((val >> count) & mask) : 0;
                const bool cf = (count <= bits) ? (((val >> (count - 1)) & 1) != 0) : false;
                update_flags_logic(m_context, res, bits);
                m_context.set_flag(CpuContext::FLAG_CF, cf);
                m_context.set_flag(CpuContext::FLAG_OF, one && (val & msb) != 0);
            } else if (sub_op == 7) { // SAR
                i64 sval = 0;
                if (bits == 64) sval = static_cast<i64>(val);
                else if (bits == 32) sval = static_cast<i64>(static_cast<i32>(val));
                else if (bits == 16) sval = static_cast<i64>(static_cast<i16>(val));
                else sval = static_cast<i64>(static_cast<i8>(val));
                res = static_cast<u64>(sval >> ((count < bits) ? count : (bits - 1))) & mask;
                const bool cf = (count < bits) ? (((val >> (count - 1)) & 1) != 0) : (sval < 0);
                update_flags_logic(m_context, res, bits);
                m_context.set_flag(CpuContext::FLAG_CF, cf);
                m_context.set_flag(CpuContext::FLAG_OF, false);
            }
            write_rm(m, res, bits);
        } else if (bits == 32 && m.is_reg) {
            // A zero count changes nothing, but a 32-bit register destination
            // is still written, which clears bits 63:32.
            set_gpr32(m_context, m.rm_id, static_cast<u32>(val));
        }

        m_context.rip = start_rip + cursor + m.bytes_consumed + imm_sz;
        return StepResult::Ok;
    }

    // 0xF6 / 0xF7: Unary ALU / Multiply / Divide / Test
    if (op0 == 0xF6 || op0 == 0xF7) {
        const usize bits = (op0 == 0xF6) ? 8 : op_bits;
        const u8 modrm_peek = raw_code[cursor];
        const u8 sub_op = (modrm_peek >> 3) & 7;
        const usize extra_imm = (sub_op == 0) ? (bits == 8 ? 1 : (op_bits == 16 ? 2 : 4)) : 0;

        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context, extra_imm);
        if (!m.valid) return StepResult::IllegalInstruction;
        if (sub_op == 0) { // TEST r/m, imm
            u64 imm = 0;
            if (bits == 8) imm = raw_code[cursor + m.bytes_consumed];
            else if (bits == 16) { u16 imm16 = 0; std::memcpy(&imm16, raw_code + cursor + m.bytes_consumed, 2); imm = imm16; }
            else { u32 imm32 = 0; std::memcpy(&imm32, raw_code + cursor + m.bytes_consumed, 4); imm = imm32; }
            const u64 val = read_rm(m, bits);
            update_flags_logic(m_context, val & imm, bits);
            m_context.rip = start_rip + cursor + m.bytes_consumed + extra_imm;
            return StepResult::Ok;
        }
        if (sub_op == 2) { // NOT r/m
            if (!m.is_reg && lock_prefix) {
                int handled = -1;
                u64 old_raw = 0;
                if (bits == 64) handled = guest_atomic_xor<u64>(m_memory, m.mem_addr, ~0ULL, &old_raw);
                else if (bits == 32) { u32 o = 0; handled = guest_atomic_xor<u32>(m_memory, m.mem_addr, 0xFFFFFFFFU, &o); old_raw = o; }
                else if (bits == 16) { u16 o = 0; handled = guest_atomic_xor<u16>(m_memory, m.mem_addr, 0xFFFFU, &o); old_raw = o; }
                else { u8 o = 0; handled = guest_atomic_xor<u8>(m_memory, m.mem_addr, 0xFFU, &o); old_raw = o; }
                if (handled >= 0) {
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u64 val = read_rm(m, bits);
            write_rm(m, ~val, bits);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (sub_op == 3) { // NEG r/m
            if (!m.is_reg && lock_prefix) {
                int handled = -1;
                u64 old_raw = 0;
                if (bits == 64) handled = guest_atomic_neg<u64>(m_memory, m.mem_addr, &old_raw);
                else if (bits == 32) { u32 o = 0; handled = guest_atomic_neg<u32>(m_memory, m.mem_addr, &o); old_raw = o; }
                else if (bits == 16) { u16 o = 0; handled = guest_atomic_neg<u16>(m_memory, m.mem_addr, &o); old_raw = o; }
                else { u8 o = 0; handled = guest_atomic_neg<u8>(m_memory, m.mem_addr, &o); old_raw = o; }
                if (handled >= 0) {
                    const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
                    const u64 val = old_raw & mask;
                    execute_alu_op(m_context, 5 /* SUB */, 0, val, bits);
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u64 val = read_rm(m, bits);
            const u64 res = execute_alu_op(m_context, 5 /* SUB */, 0, val, bits);
            write_rm(m, res, bits);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (sub_op == 4) { // MUL r/m
            const u64 val = read_rm(m, bits);
            if (bits == 8) {
                const u16 res = static_cast<u16>(m_context.rax & 0xFF) * static_cast<u16>(val & 0xFF);
                m_context.rax = (m_context.rax & ~0xFFFFULL) | res;
                const bool overflow = (res >> 8) != 0;
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            } else if (bits == 16) {
                const u32 res = static_cast<u32>(m_context.rax & 0xFFFF) * static_cast<u32>(val & 0xFFFF);
                m_context.rax = (m_context.rax & ~0xFFFFULL) | (res & 0xFFFF);
                m_context.rdx = (m_context.rdx & ~0xFFFFULL) | ((res >> 16) & 0xFFFF);
                const bool overflow = (res >> 16) != 0;
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            } else if (bits == 32) {
                const u64 res = static_cast<u64>(static_cast<u32>(m_context.rax)) * static_cast<u64>(static_cast<u32>(val));
                m_context.rax = static_cast<u32>(res);
                m_context.rdx = static_cast<u32>(res >> 32);
                const bool overflow = (res >> 32) != 0;
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            } else { // 64
                const unsigned __int128 res = static_cast<unsigned __int128>(m_context.rax) * static_cast<unsigned __int128>(val);
                m_context.rax = static_cast<u64>(res);
                m_context.rdx = static_cast<u64>(res >> 64);
                const bool overflow = (m_context.rdx != 0);
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (sub_op == 5) { // IMUL r/m
            const u64 val = read_rm(m, bits);
            if (bits == 8) {
                const i16 res = static_cast<i16>(static_cast<i8>(m_context.rax)) * static_cast<i16>(static_cast<i8>(val));
                m_context.rax = (m_context.rax & ~0xFFFFULL) | static_cast<u16>(res);
                const bool overflow = (res != static_cast<i16>(static_cast<i8>(res)));
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            } else if (bits == 16) {
                const i32 res = static_cast<i32>(static_cast<i16>(m_context.rax)) * static_cast<i32>(static_cast<i16>(val));
                m_context.rax = (m_context.rax & ~0xFFFFULL) | (static_cast<u32>(res) & 0xFFFF);
                m_context.rdx = (m_context.rdx & ~0xFFFFULL) | ((static_cast<u32>(res) >> 16) & 0xFFFF);
                const bool overflow = (res != static_cast<i32>(static_cast<i16>(res)));
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            } else if (bits == 32) {
                const i64 res = static_cast<i64>(static_cast<i32>(m_context.rax)) * static_cast<i64>(static_cast<i32>(val));
                m_context.rax = static_cast<u32>(res);
                m_context.rdx = static_cast<u32>(static_cast<u64>(res) >> 32);
                const bool overflow = (res != static_cast<i64>(static_cast<i32>(res)));
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            } else { // 64
                const __int128 res = static_cast<__int128>(static_cast<i64>(m_context.rax)) * static_cast<__int128>(static_cast<i64>(val));
                m_context.rax = static_cast<u64>(res);
                m_context.rdx = static_cast<u64>(res >> 64);
                const bool overflow = (res != static_cast<__int128>(static_cast<i64>(res)));
                m_context.set_flag(CpuContext::FLAG_CF, overflow);
                m_context.set_flag(CpuContext::FLAG_OF, overflow);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (sub_op == 6) { // DIV r/m
            const u64 val = read_rm(m, bits);
            if (val == 0) return StepResult::IllegalInstruction;
            if (bits == 8) {
                const u16 dividend = static_cast<u16>(m_context.rax & 0xFFFF);
                const u16 quotient = dividend / static_cast<u8>(val);
                const u8 remainder = dividend % static_cast<u8>(val);
                if (quotient > 0xFF) return StepResult::IllegalInstruction;
                m_context.rax = (m_context.rax & ~0xFFFFULL) | ((static_cast<u16>(remainder) << 8) | (quotient & 0xFF));
            } else if (bits == 16) {
                const u32 dividend = ((m_context.rdx & 0xFFFF) << 16) | (m_context.rax & 0xFFFF);
                const u32 quotient = dividend / static_cast<u16>(val);
                const u16 remainder = dividend % static_cast<u16>(val);
                if (quotient > 0xFFFF) return StepResult::IllegalInstruction;
                m_context.rax = (m_context.rax & ~0xFFFFULL) | (quotient & 0xFFFF);
                m_context.rdx = (m_context.rdx & ~0xFFFFULL) | remainder;
            } else if (bits == 32) {
                const u64 dividend = (static_cast<u64>(static_cast<u32>(m_context.rdx)) << 32) | static_cast<u32>(m_context.rax);
                const u64 quotient = dividend / static_cast<u32>(val);
                const u32 remainder = dividend % static_cast<u32>(val);
                if (quotient > 0xFFFFFFFFULL) return StepResult::IllegalInstruction;
                m_context.rax = static_cast<u32>(quotient);
                m_context.rdx = remainder;
            } else { // 64
                const unsigned __int128 dividend = (static_cast<unsigned __int128>(m_context.rdx) << 64) | m_context.rax;
                const unsigned __int128 quotient = dividend / val;
                const u64 remainder = static_cast<u64>(dividend % val);
                if (quotient > ~0ULL) return StepResult::IllegalInstruction;
                m_context.rax = static_cast<u64>(quotient);
                m_context.rdx = remainder;
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (sub_op == 7) { // IDIV r/m
            const u64 val = read_rm(m, bits);
            if (val == 0) return StepResult::IllegalInstruction;
            if (bits == 8) {
                const i16 dividend = static_cast<i16>(m_context.rax & 0xFFFF);
                const i8 divisor = static_cast<i8>(val);
                if (divisor == -1 && dividend == -128) return StepResult::IllegalInstruction;
                const i16 quotient = dividend / divisor;
                const i8 remainder = dividend % divisor;
                if (quotient < -128 || quotient > 127) return StepResult::IllegalInstruction;
                m_context.rax = (m_context.rax & ~0xFFFFULL) | ((static_cast<u16>(static_cast<u8>(remainder)) << 8) | static_cast<u8>(quotient));
            } else if (bits == 16) {
                const i32 dividend = static_cast<i32>(((m_context.rdx & 0xFFFF) << 16) | (m_context.rax & 0xFFFF));
                const i16 divisor = static_cast<i16>(val);
                if (divisor == -1 && dividend == -32768) return StepResult::IllegalInstruction;
                const i32 quotient = dividend / divisor;
                const i16 remainder = dividend % divisor;
                if (quotient < -32768 || quotient > 32767) return StepResult::IllegalInstruction;
                m_context.rax = (m_context.rax & ~0xFFFFULL) | static_cast<u16>(quotient);
                m_context.rdx = (m_context.rdx & ~0xFFFFULL) | static_cast<u16>(remainder);
            } else if (bits == 32) {
                const i64 dividend = static_cast<i64>((static_cast<u64>(static_cast<u32>(m_context.rdx)) << 32) | static_cast<u32>(m_context.rax));
                const i32 divisor = static_cast<i32>(val);
                if (divisor == -1 && dividend == static_cast<i64>(INT32_MIN)) return StepResult::IllegalInstruction;
                const i64 quotient = dividend / divisor;
                const i32 remainder = dividend % divisor;
                if (quotient < INT32_MIN || quotient > INT32_MAX) return StepResult::IllegalInstruction;
                m_context.rax = static_cast<u32>(quotient);
                m_context.rdx = static_cast<u32>(remainder);
            } else { // 64
                const __int128 dividend = (static_cast<__int128>(m_context.rdx) << 64) | static_cast<unsigned __int128>(m_context.rax);
                const i64 divisor = static_cast<i64>(val);
                if (divisor == -1 && dividend == static_cast<__int128>(INT64_MIN)) return StepResult::IllegalInstruction;
                const __int128 quotient = dividend / divisor;
                const i64 remainder = static_cast<i64>(dividend % divisor);
                if (quotient < static_cast<__int128>(INT64_MIN) || quotient > static_cast<__int128>(INT64_MAX)) return StepResult::IllegalInstruction;
                m_context.rax = static_cast<u64>(quotient);
                m_context.rdx = static_cast<u64>(remainder);
            }
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
    }

    // 0x70 - 0x7F: Jcc rel8
    if (op0 >= 0x70 && op0 <= 0x7F) {
        const i8 rel8 = static_cast<i8>(raw_code[cursor]);
        const bool cond = check_condition(m_context, op0 & 0x0F);
        m_context.rip = (start_rip + cursor + 1) + (cond ? rel8 : 0);
        return StepResult::Ok;
    }

    // 0xEB: JMP rel8
    if (op0 == 0xEB) {
        const i8 rel8 = static_cast<i8>(raw_code[cursor]);
        m_context.rip = start_rip + cursor + 1 + rel8;
        return StepResult::Ok;
    }

    // 0xE0: LOOPNE rel8 / LOOPNZ rel8
    if (op0 == 0xE0) {
        const i8 rel8 = static_cast<i8>(raw_code[cursor]);
        const u64 cnt_reg = (m_context.rcx & 0xFFFFFFFFFFFFFFFFULL);
        m_context.rcx = cnt_reg - 1;
        const bool cond = (m_context.rflags & CpuContext::FLAG_ZF) == 0 && m_context.rcx != 0;
        m_context.rip = (start_rip + cursor + 1) + (cond ? rel8 : 0);
        return StepResult::Ok;
    }

    // 0xE1: LOOPE rel8 / LOOPZ rel8
    if (op0 == 0xE1) {
        const i8 rel8 = static_cast<i8>(raw_code[cursor]);
        const u64 cnt_reg = (m_context.rcx & 0xFFFFFFFFFFFFFFFFULL);
        m_context.rcx = cnt_reg - 1;
        const bool cond = (m_context.rflags & CpuContext::FLAG_ZF) != 0 && m_context.rcx != 0;
        m_context.rip = (start_rip + cursor + 1) + (cond ? rel8 : 0);
        return StepResult::Ok;
    }

    // 0xE2: LOOP rel8
    if (op0 == 0xE2) {
        const i8 rel8 = static_cast<i8>(raw_code[cursor]);
        const u64 cnt_reg = (m_context.rcx & 0xFFFFFFFFFFFFFFFFULL);
        m_context.rcx = cnt_reg - 1;
        const bool cond = m_context.rcx != 0;
        m_context.rip = (start_rip + cursor + 1) + (cond ? rel8 : 0);
        return StepResult::Ok;
    }

    // 0xEB: JMP rel8

    // 0xE9: JMP rel32
    if (op0 == 0xE9) {
        i32 rel32 = 0;
        std::memcpy(&rel32, raw_code + cursor, 4);
        m_context.rip = start_rip + cursor + 4 + rel32;
        return StepResult::Ok;
    }

    // 0xE8: CALL rel32
    if (op0 == 0xE8) {
        i32 rel32 = 0;
        std::memcpy(&rel32, raw_code + cursor, 4);
        m_context.rsp -= 8;
        const u64 ret_addr = start_rip + cursor + 4;
        if (!m_memory.write_value(m_context.rsp, ret_addr)) {
            return StepResult::MemoryFault;
        }
        record_xfer(start_rip, ret_addr + rel32, m_context.rsp, m_context.rbp, 0);
        m_context.rip = ret_addr + rel32;
        return StepResult::Ok;
    }

    // 0xC3: RET
    if (op0 == 0xC3) {
        u64 ret_addr = 0;
        if (!m_memory.read_value(m_context.rsp, ret_addr)) {
            return StepResult::MemoryFault;
        }
        record_xfer(start_rip, ret_addr, m_context.rsp + 8, m_context.rbp, 1);
        m_context.rsp += 8;
        m_context.rip = ret_addr;
        return StepResult::Ok;
    }

    // 0xC2: RET imm16
    if (op0 == 0xC2) {
        u16 imm16 = 0;
        std::memcpy(&imm16, raw_code + cursor, 2);
        u64 ret_addr = 0;
        if (!m_memory.read_value(m_context.rsp, ret_addr)) {
            return StepResult::MemoryFault;
        }
        m_context.rsp += 8 + imm16;
        m_context.rip = ret_addr;
        return StepResult::Ok;
    }

    // 0xFE: Group 4 (INC r/m8, DEC r/m8)
    if (op0 == 0xFE) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        const u8 sub_op = m.reg_id & 7;
        const bool old_cf = m_context.flag_cf();
        if (sub_op == 0) { // INC r/m8
            if (!m.is_reg && lock_prefix) {
                u8 old_raw = 0;
                if (guest_atomic_xadd<u8>(m_memory, m.mem_addr, 1, &old_raw) >= 0) {
                    const u8 res = static_cast<u8>(old_raw + 1);
                    update_flags_add(m_context, old_raw, 1, res, 8);
                    m_context.set_flag(CpuContext::FLAG_CF, old_cf);
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u8 val = static_cast<u8>(read_rm(m, 8));
            const u8 res = static_cast<u8>(val + 1);
            update_flags_add(m_context, val, 1, res, 8);
            m_context.set_flag(CpuContext::FLAG_CF, old_cf);
            write_rm(m, res, 8);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (sub_op == 1) { // DEC r/m8
            if (!m.is_reg && lock_prefix) {
                u8 old_raw = 0;
                if (guest_atomic_xadd<u8>(m_memory, m.mem_addr, static_cast<u8>(-1), &old_raw) >= 0) {
                    const u8 res = static_cast<u8>(old_raw - 1);
                    update_flags_sub(m_context, old_raw, 1, res, 8);
                    m_context.set_flag(CpuContext::FLAG_CF, old_cf);
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u8 val = static_cast<u8>(read_rm(m, 8));
            const u8 res = static_cast<u8>(val - 1);
            update_flags_sub(m_context, val, 1, res, 8);
            m_context.set_flag(CpuContext::FLAG_CF, old_cf);
            write_rm(m, res, 8);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
    }

    // 0xFF: Group 5 (INC, DEC, CALL, JMP, PUSH)
    if (op0 == 0xFF) {
        auto m = decode_modrm(raw_code + cursor, 15 - cursor, next_ip_base, rex_r, rex_x, rex_b, seg_override, m_context);
        if (!m.valid) return StepResult::IllegalInstruction;
        const u8 sub_op = m.reg_id & 7;

        if (sub_op == 0) { // INC r/m
            const bool old_cf = m_context.flag_cf();
            if (!m.is_reg && lock_prefix) {
                int handled = -1;
                u64 old_raw = 0;
                if (op_bits == 64) handled = guest_atomic_xadd<u64>(m_memory, m.mem_addr, 1ULL, &old_raw);
                else if (op_bits == 32) { u32 o = 0; handled = guest_atomic_xadd<u32>(m_memory, m.mem_addr, 1U, &o); old_raw = o; }
                else { u16 o = 0; handled = guest_atomic_xadd<u16>(m_memory, m.mem_addr, 1U, &o); old_raw = o; }
                if (handled >= 0) {
                    const u64 mask = (op_bits == 64) ? ~0ULL : ((1ULL << op_bits) - 1);
                    const u64 val = old_raw & mask;
                    const u64 res = (val + 1) & mask;
                    update_flags_add(m_context, val, 1, res, op_bits);
                    m_context.set_flag(CpuContext::FLAG_CF, old_cf);
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u64 val = read_rm(m, op_bits);
            const u64 res = (val + 1);
            update_flags_add(m_context, val, 1, res, op_bits);
            m_context.set_flag(CpuContext::FLAG_CF, old_cf);
            write_rm(m, res, op_bits);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (sub_op == 1) { // DEC r/m
            const bool old_cf = m_context.flag_cf();
            if (!m.is_reg && lock_prefix) {
                int handled = -1;
                u64 old_raw = 0;
                if (op_bits == 64) handled = guest_atomic_xadd<u64>(m_memory, m.mem_addr, static_cast<u64>(-1LL), &old_raw);
                else if (op_bits == 32) { u32 o = 0; handled = guest_atomic_xadd<u32>(m_memory, m.mem_addr, static_cast<u32>(-1), &o); old_raw = o; }
                else { u16 o = 0; handled = guest_atomic_xadd<u16>(m_memory, m.mem_addr, static_cast<u16>(-1), &o); old_raw = o; }
                if (handled >= 0) {
                    const u64 mask = (op_bits == 64) ? ~0ULL : ((1ULL << op_bits) - 1);
                    const u64 val = old_raw & mask;
                    const u64 res = (val - 1) & mask;
                    update_flags_sub(m_context, val, 1, res, op_bits);
                    m_context.set_flag(CpuContext::FLAG_CF, old_cf);
                    m_context.rip = start_rip + cursor + m.bytes_consumed;
                    return StepResult::Ok;
                }
            }
            const u64 val = read_rm(m, op_bits);
            const u64 res = (val - 1);
            update_flags_sub(m_context, val, 1, res, op_bits);
            m_context.set_flag(CpuContext::FLAG_CF, old_cf);
            write_rm(m, res, op_bits);
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
        if (sub_op == 2) { // CALL r/m64
            const u64 target = read_rm(m, 64);
            m_context.rsp -= 8;
            const u64 ret_addr = start_rip + cursor + m.bytes_consumed;
            if (!m_memory.write_value(m_context.rsp, ret_addr)) return StepResult::MemoryFault;
            record_xfer(start_rip, target, m_context.rsp, m_context.rbp, 0);
            m_context.rip = target;
            return StepResult::Ok;
        }
        if (sub_op == 4) { // JMP r/m64 (PLT stubs)
            const u64 target = read_rm(m, 64);
            record_xfer(start_rip, target, m_context.rsp, m_context.rbp, 2);
            m_context.rip = target;
            return StepResult::Ok;
        }
        if (sub_op == 6) { // PUSH r/m64
            const u64 val = read_rm(m, 64);
            m_context.rsp -= 8;
            if (!m_memory.write_value(m_context.rsp, val)) return StepResult::MemoryFault;
            m_context.rip = start_rip + cursor + m.bytes_consumed;
            return StepResult::Ok;
        }
    }

    // 0x98: CBW (16-bit), CWDE (32-bit), CDQE (64-bit)
    if (op0 == 0x98) {
        if (rex_w) {
            m_context.rax = static_cast<u64>(static_cast<i64>(static_cast<i32>(m_context.rax)));
        } else if (operand_size_16) {
            set_gpr16(m_context, 0, static_cast<u16>(static_cast<i16>(static_cast<i8>(m_context.rax))));
        } else {
            m_context.rax = static_cast<u32>(static_cast<i32>(static_cast<i16>(m_context.rax)));
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // 0x99: CWD (16-bit), CDQ (32-bit), CQO (64-bit)
    if (op0 == 0x99) {
        if (rex_w) {
            const i64 val = static_cast<i64>(m_context.rax);
            m_context.rdx = (val < 0) ? ~0ULL : 0ULL;
        } else if (operand_size_16) {
            const i16 val = static_cast<i16>(m_context.rax);
            set_gpr16(m_context, 2, (val < 0) ? 0xFFFF : 0);
        } else {
            const i32 val = static_cast<i32>(m_context.rax);
            m_context.rdx = (val < 0) ? 0xFFFFFFFFULL : 0ULL;
        }
        m_context.rip = start_rip + cursor;
        return StepResult::Ok;
    }

    // ---------------------------------------------------------------- x87 ---
    // Encodings implemented, all verified against the Intel SDM opcode map:
    //   D9 /0  FLD m32fp     D9 /2 FST m32fp   D9 /3 FSTP m32fp
    //   D9 /5  FLDCW m16     D9 /7 FNSTCW m16
    //   D9 C0+i FLD ST(i)    D9 C8+i FXCH ST(i) D9 E0 FCHS   D9 E1 FABS
    //   DD /0  FLD m64fp     DD /2 FST m64fp   DD /3 FSTP m64fp
    //   DB /0  FILD m16int   DB /3 FISTP m16int
    //   DF /0  FILD m32int   DF /5 FILD m64int DF /7 FISTP m32int
    //   D8/DC/DE/D0 arithmetic and compare (D8=m32fp, DC=m64fp, DE=m16int)
    // NOT implemented: FISTP m64int does not exist in x87 (FIST/FISTP are 16
    // and 32-bit integer only), FBLD/FBSTP (m80 BCD), FCOMPP, FUCOMPP.
    if (op0 == 0xD8 || op0 == 0xD9 || op0 == 0xDA || op0 == 0xDB ||
        op0 == 0xDC || op0 == 0xDD || op0 == 0xDE || op0 == 0xDF) {
        // Debug: track x87 instructions in the Mono runtime loop
        if (start_rip >= 0xf6d600 && start_rip <= 0xf6d700) {
            std::fprintf(stderr, "[CPU DEBUG x87] op0=0x%02x at RIP=0x%llx modrm=0x%02x mod=%u reg=%u rm=%u\n",
                         static_cast<unsigned>(op0), static_cast<unsigned long long>(start_rip),
                         static_cast<unsigned>(raw_code[cursor]), static_cast<unsigned>((raw_code[cursor]>>6)&3),
                         static_cast<unsigned>((raw_code[cursor]>>3)&7), static_cast<unsigned>(raw_code[cursor]&7));
        }
        // Decode the ModRM (and any SIB/disp) to get the operand address AND the
        // true instruction length. len must be advanced as the bytes are
        // consumed -- conflating "length" with "cursor" here silently truncated
        // every short x87 instruction.
        // `cursor` already counts the opcode byte (see the 0x99 handler, which
        // advances with `start_rip + cursor` for a 1-byte opcode). Starting at
        // cursor+1 skipped a byte and decoded the wrong ModRM entirely.
        usize len = cursor;
        u8 modrm = 0;
        if (len >= 8) return StepResult::MemoryFault;
        modrm = raw_code[len++];
        const u8 mod = (modrm >> 6) & 3;
        const u8 regf = (modrm >> 3) & 7;
        const u8 rmf = modrm & 7;

        usize rmf_is_mem = (mod != 3);
        if (rmf_is_mem && rmf == 4) ++len;   // SIB follows
        usize disp_size = 0;
        if (mod == 1) disp_size = 1;
        else if (mod == 2) disp_size = 4;

        GuestAddr ea = 0;
        bool have_ea = false;
        i32 disp = 0;
        if (len + disp_size > 8) return StepResult::MemoryFault;
        if (disp_size == 1) {
            disp = static_cast<i8>(raw_code[len]);
        } else if (disp_size == 4) {
            std::memcpy(&disp, raw_code + len, 4);
        }
        len += disp_size;

        if (rmf_is_mem) {
            if (mod == 0 && rmf == 5) {
                // RIP-relative: displacement is relative to the END of the
                // instruction, which is only known now that len is final.
                ea = start_rip + len + disp;
                have_ea = true;
            } else {
                const GuestAddr kReg[8] = {
                    m_context.rax, m_context.rcx, m_context.rdx, m_context.rbx,
                    m_context.rsp, m_context.rbp, m_context.rsi, m_context.rdi};
                GuestAddr base = kReg[rmf];
                if (rmf == 4) {
                    u8 sib = 0;
                    if (len >= 1) sib = raw_code[len - disp_size - 1];
                    const u8 scale = sib >> 6;
                    const u8 index = (sib >> 3) & 7;
                    const u8 bsel = sib & 7;
                    const GuestAddr kIdx[8] = {
                        m_context.rax, m_context.rcx, m_context.rdx, m_context.rbx,
                        m_context.rsp, m_context.rbp, m_context.rsi, m_context.rdi};
                    base = kReg[bsel] + (kIdx[index] << scale);
                }
                ea = base + disp;
                have_ea = true;
            }
        }

        X87State& f = m_context.fpu;
        auto load_mem = [&](usize n, i64* out) -> bool {
            if (!have_ea) return false;
            u64 raw = 0;
            if (!m_memory.read_bytes(ea, &raw, n)) return false;
            *out = static_cast<i64>(raw);
            return true;
        };
        auto push_double = [&](double d) {
            u64 bits = 0;
            std::memcpy(&bits, &d, sizeof(bits));
            f.push(bits, 0);
        };
        // x87 memory operands are IEEE-754 ENCODINGS, not integers. Reinterpret
        // the bits; converting the value to an integer (the previous behaviour)
        // stored the whole-number part as a bit pattern, which is not a float.
        auto store_double = [&](usize n) -> bool {
            if (!have_ea) return false;
            double d;
            std::memcpy(&d, &f.st[f.top], sizeof(d));
            if (n == 4) { const float v = static_cast<float>(d); return m_memory.write_bytes(ea, &v, 4); }
            return m_memory.write_bytes(ea, &d, 8);
        };
        // Reads an IEEE-754 operand of n bytes and pushes it as a double.
        // n == 4 -> binary32, n == 8 -> binary64, n == 10 -> 80-bit extended.
        auto load_float = [&](usize n) -> bool {
            if (!have_ea) return false;
            if (n == 4) {
                float v = 0;
                if (!m_memory.read_bytes(ea, &v, 4)) return false;
                push_double(static_cast<double>(v));
                return true;
            }
            if (n == 8) {
                double v = 0;
                if (!m_memory.read_bytes(ea, &v, 8)) return false;
                push_double(v);
                return true;
            }
            if (n == 10) {
                u8 b[10];
                if (!m_memory.read_bytes(ea, b, 10)) return false;
                push_double(f80_to_double(b));
                return true;
            }
            return false;
        };
        // Rounding is explicit: the host FPCR is never touched, so a guest that
        // reads the host rounding mode cannot observe us changing it.
        //
        // Control word layout (Intel SDM): bits 8-9 are precision control,
        // bits 10-11 are rounding control. They are distinct fields and the
        // guest manipulates them separately -- `and ecx,0xFCFF` clears PC while
        // preserving RC, then `or ecx,0x0200` sets PC=10 (53-bit) and leaves
        // RC=00. So the mode must come from bits 10-11 and nothing else.
        //
        // PRECISION CONTROL IS IGNORED. Each stack slot is a host double, so
        // arithmetic is always 53-bit and PC=24/53/64 cannot be honoured. That
        // happens to match the PC=10 the guest selects, so no visible difference
        // arises here, but a guest asking for 64-bit extended would silently get
        // double precision. An m80 load/store boundary conversion is subject to
        // the same 53-bit limit.
        auto nearest_even = [](double d) -> double {
            const double fl = std::floor(d);
            const double diff = d - fl;
            if (diff > 0.5) return fl + 1.0;
            if (diff < 0.5) return fl;
            return (std::fmod(fl, 2.0) == 0.0) ? fl : (fl + 1.0);
        };
        // trunc=true is FISTTP: the instruction rounds toward zero and ignores
        // the control word's rounding control, per the SDM.
        auto store_int = [&](usize n, bool trunc = false) -> bool {
            if (!have_ea) return false;
            double d; std::memcpy(&d, &f.st[f.top], sizeof(d));

            // RC = bits 10-11: 00 nearest-even, 01 down(-inf), 10 up(+inf),
            // 11 truncate(toward zero).
            const int rc = trunc ? 3 : ((f.cw >> 10) & 3);
            double adj;
            switch (rc) {
                case 0:  adj = nearest_even(d); break;
                case 1:  adj = std::floor(d);   break;
                case 2:  adj = std::ceil(d);    break;
                default: adj = std::trunc(d);   break;
            }

            // Overflow or NaN yields the "integer indefinite" value and sets IE.
            const bool bad = std::isnan(d) || std::isinf(d);
            bool over = bad;
            if (!over) {
                if (n == 2)      over = (d < -32768.0    || d > 32767.0);
                else if (n == 4) over = (d < -2147483648.0 || d > 2147483647.0);
                else             over = (d < -9223372036854775808.0 || d >= 9223372036854775808.0);
            }
            if (over) {
                f.sw |= 0x0001;   // IE
                f.sw |= 0x0100;   // also set the C2-ish "invalid" style flag we track
                if (n == 2) { const u16 v = 0x8000; return m_memory.write_bytes(ea, &v, 2); }
                if (n == 4) { const u32 v = 0x80000000u; return m_memory.write_bytes(ea, &v, 4); }
                const u64 v = 0x8000000000000000ULL;
                return m_memory.write_bytes(ea, &v, 8);
            }

            // Inexact -> set IE. All exceptions are masked (cw 0x037F default),
            // so this records rather than traps.
            if (adj != d) f.sw |= 0x0001;
            f.sw |= 0x0100;
            if (n == 2) { const i16 v = static_cast<i16>(adj); return m_memory.write_bytes(ea, &v, 2); }
            if (n == 4) { const i32 v = static_cast<i32>(adj); return m_memory.write_bytes(ea, &v, 4); }
            const i64 v = static_cast<i64>(adj); return m_memory.write_bytes(ea, &v, 8);
        };

        static const bool s_trace_x87 = std::getenv("PS4_TRACE_X87") != nullptr;
        if (s_trace_x87) {
            std::fprintf(stderr, "[x87] op=0x%02x modrm=0x%02x mod=%u ext=%u rm=%u len=%zu ea=0x%llx\n",
                         op0, modrm, (unsigned)mod, (unsigned)regf, (unsigned)rmf, len,
                         (unsigned long long)ea);
        }
        bool handled = true;
        switch (op0) {
            case 0xD9: {
                if (mod == 3) {
                    if ((modrm & 0xC0) == 0xC0) {           // D9 C0+i / C8+i / D0+i / E0+i
                        const u8 i = rmf;
                        // SDM, D9 register form (mod=11): the reg field selects the
                        // operation and rm selects the operand/constant.
                        //   000 FLD ST(i)   001 FXCH ST(i)   010 FST ST(i)   011 FSTP ST(i)
                        //   100 FCHS(rm=0) FABS(rm=1) FTST(rm=4) FXAM(rm=5)
                        //   101 constants: FLD1(rm=0) ... FLDZ(rm=6)
                        // Previously reg=101 executed FABS, so every FLD1/FLDZ in the
                        // guest silently took the absolute value instead of pushing a
                        // constant.
                        auto peek = [&](u8 slot) { double v; std::memcpy(&v, &f.st[slot], sizeof(v)); return v; };
                        auto poke = [&](u8 slot, double v) { std::memcpy(&f.st[slot], &v, sizeof(v)); };
                        auto pop1 = [&] { f.top = static_cast<u8>((f.top + 1) & 7); };
                        // rmf is a LOGICAL stack index: ST(i) lives in physical
                        // slot (TOP + i) & 7. Indexing st[rmf] directly addressed
                        // the wrong register whenever TOP + i wrapped.
                        const u8 phys = static_cast<u8>((f.top + i) & 7);
                        switch (regf) {
                            case 0: push_double(peek(phys)); break;                  // FLD ST(i)
                            case 1: { const double t = peek(f.top);
                                      poke(f.top, peek(phys)); poke(phys, t); } break; // FXCH ST(i)
                            case 2: poke(phys, peek(f.top)); break;                  // FST ST(i)
                            case 3: poke(phys, peek(f.top)); pop1(); break;         // FSTP ST(i)
                            case 4:
                                if (i == 0) push_double(-peek(f.top));             // FCHS
                                else if (i == 1) push_double(std::abs(peek(f.top))); // FABS
                                else handled = false;                              // FTST/FXAM unimplemented
                                break;
                            case 5:
                                if (i == 0) push_double(1.0);                      // D9 E8 FLD1
                                else if (i == 6) push_double(0.0);                 // D9 EE FLDZ
                                else handled = false;                              // other constants
                                break;
                            default: handled = false;
                        }
                    } else { handled = false; }
                } else {
                    switch (regf) {
                        case 0: if (!load_float(4)) handled = false; break;        // FLD m32fp
                        case 2: if (!store_double(4)) handled = false; break;       // FST m32fp
                        case 3: if (!store_double(4)) handled = false;
                                f.top = static_cast<u8>((f.top + 1) & 7); break;     // FSTP m32fp
                        case 5: { u16 cw = 0;
                                  if (m_memory.read_bytes(ea, &cw, 2)) f.cw = cw;
                                  else { std::fprintf(stderr, "[x87] FLDCW read FAILED ea=0x%llx\n",
                                                      (unsigned long long)ea); handled = false; } } break;  // FLDCW
                        case 7: { if (!m_memory.write_bytes(ea, &f.cw, 2)) {
                                    std::fprintf(stderr, "[x87] FNSTCW write FAILED ea=0x%llx\n",
                                                 (unsigned long long)ea);
                                    handled = false; } }
                                break;                                            // FNSTCW
                        default: handled = false;
                    }
                }
                break;
            }
            case 0xDD: {
                switch (regf) {
                    case 0: if (!load_float(8)) handled = false; break;             // DD /0 FLD m64fp
                    case 1: if (store_int(8, true)) f.top = static_cast<u8>((f.top + 1) & 7);
                            else handled = false; break;                           // DD /1 FISTTP m64int
                    case 2: if (!store_double(8)) handled = false; break;            // DD /2 FST m64fp
                    case 3: if (!store_double(8)) handled = false;
                            f.top = static_cast<u8>((f.top + 1) & 7); break;         // FSTP m64fp
                    default: handled = false;
                }
                break;
            }
            case 0xDB: {
                i64 raw = 0;
                switch (regf) {
                    case 0: if (load_mem(4, &raw)) push_double(static_cast<double>(raw));
                            else handled = false; break;                            // DB /0 FILD m32int
                    case 1: if (store_int(4, true)) f.top = static_cast<u8>((f.top + 1) & 7);
                            else handled = false; break;                           // DB /1 FISTTP m32int
                    case 2: store_int(4); break;                                    // DB /2 FIST m32int (no pop)
                    case 3: if (store_int(4)) f.top = static_cast<u8>((f.top + 1) & 7);
                            else handled = false; break;                           // DB /3 FISTP m32int
                    case 5: if (!load_float(10)) handled = false; break;             // FLD m80
                    case 7: {                                                       // FSTP m80
                        if (!have_ea) { handled = false; break; }
                        double d; std::memcpy(&d, &f.st[f.top], sizeof(d));
                        u8 buf[10]; double_to_f80(d, buf);
                        if (!m_memory.write_bytes(ea, buf, 10)) handled = false;
                        f.top = static_cast<u8>((f.top + 1) & 7);
                        break;
                    }
                    default: handled = false;
                }
                break;
            }
            case 0xDE: {
                // DE C0+i = FADDP ST(i), ST(0), then pop. No memory form for DE.
                if (mod == 3 && regf == 0) {
                    // ST(i) is logical; its physical slot is (TOP + i) & 7.
                    const u8 phys = static_cast<u8>((f.top + rmf) & 7);
                    double a; std::memcpy(&a, &f.st[f.top], sizeof(a));   // ST(0)
                    double b; std::memcpy(&b, &f.st[phys], sizeof(b));    // ST(i)
                    const double r = b + a;
                    std::memcpy(&f.st[phys], &r, sizeof(r));
                    f.top = static_cast<u8>((f.top + 1) & 7);             // pop
                } else {
                    handled = false;
                }
                break;
            }
            case 0xDF: {
                i64 raw = 0;
                if (mod == 3) {
                    // DF F0+i = FCOMIP ST(i): set ZF/PF/CF then pop.
                    if (regf == 6) {
                        const u8 phys = static_cast<u8>((f.top + rmf) & 7);  // ST(i)
                        double a; std::memcpy(&a, &f.st[f.top], sizeof(a));    // ST(0)
                        double b; std::memcpy(&b, &f.st[phys], sizeof(b));
                        u64 fl = m_context.rflags & ~static_cast<u64>(0x8D); // ZF|PF|CF
                        if (std::isnan(a) || std::isnan(b)) {
                            fl |= 0x8D;                     // ZF=PF=CF=1, unordered
                            f.sw |= 0x0100;
                        } else if (a > b) {
                            /* ZF=PF=CF=0 */
                        } else if (a < b) {
                            fl |= 0x1;                      // CF=1
                        } else {
                            fl |= 0x40;                     // ZF=1
                        }
                        m_context.rflags = fl;
                        f.top = static_cast<u8>((f.top + 1) & 7);   // pop
                    } else {
                        handled = false;
                    }
                    break;
                }
                switch (regf) {
                    case 0: if (load_mem(2, &raw)) push_double(static_cast<double>(raw));
                            else handled = false; break;                            // DF /0 FILD m16int
                    case 1: if (store_int(2, true)) f.top = static_cast<u8>((f.top + 1) & 7);
                            else handled = false; break;                           // DF /1 FISTTP m16int
                    case 2: store_int(2); break;                                    // DF /2 FIST m16int (no pop)
                    case 3: if (store_int(2)) f.top = static_cast<u8>((f.top + 1) & 7);
                            else handled = false; break;                           // DF /3 FISTP m16int
                    case 5: if (load_mem(8, &raw)) push_double(static_cast<double>(raw));
                            else handled = false; break;                            // DF /5 FILD m64int
                    // DF /6 is FBSTP m80 (packed-BCD store), not FISTTP m32int.
                    // It is not implemented; leaving it out makes it fault loudly
                    // instead of storing a binary integer in place of BCD.
                    case 6: handled = false; break;                                // DF /6 FBSTP m80 (unimplemented)
                    case 7: if (store_int(8)) f.top = static_cast<u8>((f.top + 1) & 7);
                            else handled = false; break;                           // DF /7 FISTP m64int
                    default: handled = false;
                }
                break;
            }
            default:
                handled = false;
        }
        if (handled) {
            // One-shot trace so a bad operand address is visible without
            // spamming a hot path.
            m_context.rip = start_rip + len;
            return StepResult::Ok;
        }
        // fall through to the unhandled report
    }

    std::fprintf(stderr, "[CPU] Unhandled opcode at RIP=0x%llx: %02x %02x %02x %02x %02x %02x %02x %02x (cursor=%zu, rex=0x%02x, op0=0x%02x)\n",                 (unsigned long long)start_rip,
                 raw_code[0], raw_code[1], raw_code[2], raw_code[3], raw_code[4], raw_code[5], raw_code[6], raw_code[7],
                 cursor, rex, op0);
    std::fprintf(stderr, "[CPU] Recent RIP history (last 32 instructions):\n");
    for (usize i = 0; i < 32; ++i) {
        const GuestAddr h_rip = s_rip_history[(s_rip_hist_idx + 128 - 32 + i) % 128];
        if (h_rip != 0) {
            std::fprintf(stderr, "  [%02zu] RIP=0x%llx\n", i, (unsigned long long)h_rip);
        }
    }
    return StepResult::IllegalInstruction;
}

RunResult CpuExecutor::run(usize max_instructions) {
    RunResult result{};

    while (true) {
        if (max_instructions > 0 && result.instructions_executed >= max_instructions) {
            result.status = StepResult::Ok;
            break;
        }

        // Try JIT block execution if compiler attached
        if (m_jit && !m_syscalls.has_exited()) {
            usize retired = 0;
            if (m_jit->execute_block(m_context, m_memory, &retired)) {
                // A block that side-exits retires fewer instructions than it
                // compiled; count what actually ran.
                result.instructions_executed += retired;
                if (jit_stats_enabled()) {
                    jit_stats_record_jit_block(retired);
                }

                if (m_syscalls.has_exited()) {
                    result.status = StepResult::Exited;
                    result.exit_code = m_syscalls.exit_code();
                    break;
                }
                continue;
            }
        }

        StepResult step_res = step();
        result.instructions_executed++;

        if (step_res == StepResult::Exited) {
            result.status = StepResult::Exited;
            result.exit_code = m_syscalls.exit_code();
            break;
        }

        if (step_res == StepResult::Halted) {
            result.status = StepResult::Halted;
            break;
        }

        if (step_res != StepResult::Ok && step_res != StepResult::Syscall) {
            result.status = step_res;
            break;
        }
    }

    return result;
}

} // namespace ps4
