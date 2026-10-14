#include "ps4/syscall_table.hpp"
#include "ps4/vfs.hpp"
#include <iostream>

namespace ps4 {

SyscallTable::SyscallTable(MemoryManager& memory)
    : m_memory(memory), m_out_stream(&std::cout) {}

void SyscallTable::register_handler(u32 syscall_num, SyscallHandler handler) {
    m_handlers[syscall_num] = std::move(handler);
}

bool SyscallTable::is_registered(u32 syscall_num) const {
    return m_handlers.find(syscall_num) != m_handlers.end();
}

OrbisResult SyscallTable::dispatch(u32 syscall_num, SyscallContext& ctx) {
    if (!ctx.memory) {
        ctx.memory = &m_memory;
    }
    if (!ctx.vfs) {
        ctx.vfs = m_vfs;
    }

    auto it = m_handlers.find(syscall_num);
    if (it != m_handlers.end()) {
        return it->second(ctx);
    }

    std::cerr << "[HLE] Unimplemented syscall " << syscall_num
              << " (" << syscall_name(syscall_num) << ")\n";

    // -38 is -ENOSYS (Function not implemented)
    ctx.rax = static_cast<u64>(-38);
    return -38;
}

void SyscallTable::register_default_handlers() {
    // -------------------------------------------------------------------------
    // SYS_exit (1)
    // -------------------------------------------------------------------------
    register_handler(SYS_exit, [this](SyscallContext& ctx) -> OrbisResult {
        const i32 status = static_cast<i32>(ctx.args[0]);
        m_has_exited = true;
        m_exit_code = status;
        ctx.rax = 0;

        if (m_terminate_on_exit) {
            std::exit(status);
        }
        return 0;
    });

    // -------------------------------------------------------------------------
    // SYS_read (3)
    // -------------------------------------------------------------------------
    register_handler(SYS_read, [](SyscallContext& ctx) -> OrbisResult {
        const i32 fd = static_cast<i32>(ctx.args[0]);
        const GuestAddr buf_addr = ctx.args[1];
        const usize nbytes = static_cast<usize>(ctx.args[2]);

        if (!ctx.memory || !ctx.memory->is_valid_address(buf_addr, nbytes)) {
            ctx.rax = static_cast<u64>(-14); // -EFAULT
            return -14;
        }

        if (ctx.vfs) {
            char* host_buffer = ctx.memory->to_host_ptr<char>(buf_addr);
            const i64 res = ctx.vfs->read(fd, host_buffer, nbytes);
            ctx.rax = static_cast<u64>(res);
            return res >= 0 ? 0 : static_cast<OrbisResult>(res);
        }

        ctx.rax = static_cast<u64>(-9); // -EBADF
        return -9;
    });

    // -------------------------------------------------------------------------
    // SYS_write (4)
    // -------------------------------------------------------------------------
    register_handler(SYS_write, [this](SyscallContext& ctx) -> OrbisResult {
        const i32 fd = static_cast<i32>(ctx.args[0]);
        const GuestAddr buf_addr = ctx.args[1];
        const usize nbytes = static_cast<usize>(ctx.args[2]);

        if (!ctx.memory || !ctx.memory->is_valid_address(buf_addr, nbytes)) {
            ctx.rax = static_cast<u64>(-14); // -EFAULT
            return -14;
        }

        // fd 1 = stdout, fd 2 = stderr
        if (fd == 1 || fd == 2) {
            const char* host_buffer = ctx.memory->to_host_ptr<char>(buf_addr);
            if (host_buffer) {
                auto& stream = (m_out_stream != nullptr) ? *m_out_stream : std::cout;
                stream.write(host_buffer, nbytes);
                stream.flush();
                ctx.rax = nbytes;
                return 0;
            }
        }

        // Delegate to VFS for other file descriptors
        if (ctx.vfs && fd >= 3) {
            const char* host_buffer = ctx.memory->to_host_ptr<char>(buf_addr);
            const i64 res = ctx.vfs->write(fd, host_buffer, nbytes);
            ctx.rax = static_cast<u64>(res);
            return res >= 0 ? 0 : static_cast<OrbisResult>(res);
        }

        // EBADF (-9): Bad file descriptor
        ctx.rax = static_cast<u64>(-9);
        return -9;
    });

    // -------------------------------------------------------------------------
    // SYS_open (5)
    // -------------------------------------------------------------------------
    register_handler(SYS_open, [](SyscallContext& ctx) -> OrbisResult {
        const GuestAddr path_addr = ctx.args[0];
        const i32 flags = static_cast<i32>(ctx.args[1]);
        const i32 mode = static_cast<i32>(ctx.args[2]);

        if (!ctx.memory || !ctx.memory->is_valid_address(path_addr)) {
            ctx.rax = static_cast<u64>(-14); // -EFAULT
            return -14;
        }

        const char* path_str = ctx.memory->to_host_ptr<char>(path_addr);
        if (!path_str || !ctx.vfs) {
            ctx.rax = static_cast<u64>(-14);
            return -14;
        }

        const i32 fd = ctx.vfs->open(path_str, flags, mode);
        ctx.rax = static_cast<u64>(fd);
        return fd >= 0 ? 0 : fd;
    });

    // -------------------------------------------------------------------------
    // SYS_close (6)
    // -------------------------------------------------------------------------
    register_handler(SYS_close, [](SyscallContext& ctx) -> OrbisResult {
        const i32 fd = static_cast<i32>(ctx.args[0]);
        if (ctx.vfs) {
            const i32 res = ctx.vfs->close(fd);
            ctx.rax = static_cast<u64>(res);
            return res;
        }
        ctx.rax = static_cast<u64>(-9);
        return -9;
    });

    // -------------------------------------------------------------------------
    // SYS_lseek (478)
    // -------------------------------------------------------------------------
    register_handler(SYS_lseek, [](SyscallContext& ctx) -> OrbisResult {
        const i32 fd = static_cast<i32>(ctx.args[0]);
        const i64 offset = static_cast<i64>(ctx.args[1]);
        const i32 whence = static_cast<i32>(ctx.args[2]);

        if (ctx.vfs) {
            const i64 res = ctx.vfs->lseek(fd, offset, whence);
            ctx.rax = static_cast<u64>(res);
            return res >= 0 ? 0 : static_cast<OrbisResult>(res);
        }
        ctx.rax = static_cast<u64>(-9);
        return -9;
    });

    // -------------------------------------------------------------------------
    // SYS_fstat (189)
    // -------------------------------------------------------------------------
    register_handler(SYS_fstat, [](SyscallContext& ctx) -> OrbisResult {
        const i32 fd = static_cast<i32>(ctx.args[0]);
        const GuestAddr stat_addr = ctx.args[1];

        if (!ctx.memory || !ctx.memory->is_valid_address(stat_addr, sizeof(OrbisStat))) {
            ctx.rax = static_cast<u64>(-14); // -EFAULT
            return -14;
        }

        if (ctx.vfs) {
            OrbisStat st{};
            const i32 res = ctx.vfs->fstat(fd, st);
            if (res == 0) {
                ctx.memory->write_bytes(stat_addr, &st, sizeof(st));
            }
            ctx.rax = static_cast<u64>(res);
            return res;
        }
        ctx.rax = static_cast<u64>(-9);
        return -9;
    });

    // -------------------------------------------------------------------------
    // SYS_getpid (20)
    // -------------------------------------------------------------------------
    register_handler(SYS_getpid, [](SyscallContext& ctx) -> OrbisResult {
        ctx.rax = 1001; // Mock simulated PID
        return 0;
    });

    // -------------------------------------------------------------------------
    // SYS_mprotect (74)
    // -------------------------------------------------------------------------
    register_handler(SYS_mprotect, [](SyscallContext& ctx) -> OrbisResult {
        const GuestAddr addr = ctx.args[0];
        const GuestSize len = ctx.args[1];
        const u32 bsd_prot = static_cast<u32>(ctx.args[2]);

        // Translate BSD PROT_* (R=1, W=2, X=4) to ELF PF_* (X=1, W=2, R=4)
        u32 elf_flags = 0;
        if (bsd_prot & 0x1) elf_flags |= PF_R; // PROT_READ  -> PF_R
        if (bsd_prot & 0x2) elf_flags |= PF_W; // PROT_WRITE -> PF_W
        if (bsd_prot & 0x4) elf_flags |= PF_X; // PROT_EXEC  -> PF_X

        if (!ctx.memory || !ctx.memory->protect(addr, len, elf_flags)) {
            ctx.rax = static_cast<u64>(-12); // -ENOMEM
            return -12;
        }

        ctx.rax = 0;
        return 0;
    });

}

} // namespace ps4
