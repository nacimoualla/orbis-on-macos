#pragma once

#include "ps4/common.hpp"
#include <string_view>

// Linux's <sys/syscall.h> defines SYS_* as macros with Linux numbers, which
// would silently replace the Orbis constants below. Include it first and drop
// those macros (only the Linux/QEMU CPU test harness hits this).
#if defined(__linux__)
#include <sys/syscall.h>
#undef SYS_exit
#undef SYS_fork
#undef SYS_read
#undef SYS_write
#undef SYS_open
#undef SYS_close
#undef SYS_wait4
#undef SYS_getpid
#undef SYS_munmap
#undef SYS_mprotect
#undef SYS_gettimeofday
#undef SYS_fstat
#undef SYS_clock_gettime
#undef SYS_nanosleep
#undef SYS_mmap
#undef SYS_lseek
#undef SYS_dynlib_dlsym
#undef SYS_dynlib_get_list
#undef SYS_dynlib_load_prx
#undef SYS_dynlib_unload_prx
#endif

namespace ps4 {

// Orbis OS / FreeBSD 9 Standard System Call Numbers
inline constexpr u32 SYS_exit             = 1;
inline constexpr u32 SYS_fork             = 2;
inline constexpr u32 SYS_read             = 3;
inline constexpr u32 SYS_write            = 4;
inline constexpr u32 SYS_open             = 5;
inline constexpr u32 SYS_close            = 6;
inline constexpr u32 SYS_wait4            = 7;
inline constexpr u32 SYS_getpid           = 20;
inline constexpr u32 SYS_munmap           = 73;
inline constexpr u32 SYS_mprotect         = 74;
inline constexpr u32 SYS_gettimeofday     = 116;
inline constexpr u32 SYS_fstat            = 189;
inline constexpr u32 SYS_clock_gettime    = 232;
inline constexpr u32 SYS_nanosleep        = 240;
inline constexpr u32 SYS_mmap             = 477;
inline constexpr u32 SYS_lseek            = 478;

// Orbis OS custom kernel syscalls
inline constexpr u32 SYS_dynlib_dlsym     = 591;
inline constexpr u32 SYS_dynlib_get_list  = 592;
inline constexpr u32 SYS_dynlib_load_prx  = 594;
inline constexpr u32 SYS_dynlib_unload_prx = 595;

inline std::string_view syscall_name(u32 num) {
    switch (num) {
        case SYS_exit:            return "sys_exit";
        case SYS_fork:            return "sys_fork";
        case SYS_read:            return "sys_read";
        case SYS_write:           return "sys_write";
        case SYS_open:            return "sys_open";
        case SYS_close:           return "sys_close";
        case SYS_wait4:           return "sys_wait4";
        case SYS_getpid:          return "sys_getpid";
        case SYS_munmap:          return "sys_munmap";
        case SYS_mprotect:        return "sys_mprotect";
        case SYS_gettimeofday:    return "sys_gettimeofday";
        case SYS_fstat:           return "sys_fstat";
        case SYS_clock_gettime:   return "sys_clock_gettime";
        case SYS_nanosleep:       return "sys_nanosleep";
        case SYS_mmap:            return "sys_mmap";
        case SYS_lseek:           return "sys_lseek";
        case SYS_dynlib_dlsym:    return "sys_dynlib_dlsym";
        case SYS_dynlib_get_list: return "sys_dynlib_get_list";
        case SYS_dynlib_load_prx: return "sys_dynlib_load_prx";
        case SYS_dynlib_unload_prx: return "sys_dynlib_unload_prx";
        default:                  return "unknown_syscall";
    }
}

} // namespace ps4
