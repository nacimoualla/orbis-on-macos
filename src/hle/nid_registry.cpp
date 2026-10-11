#include "ps4/audio_manager.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/thread.hpp"
#include "ps4/vfs.hpp"
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

namespace ps4 {

namespace {

// Guest heap pointer for simple HLE malloc allocations
GuestAddr g_guest_heap_current = 0x80000000ULL; // 2GB virtual mark

// Direct physical memory offset counter
GuestAddr g_direct_phys_current = 0x100000000ULL; // 4GB mark

// Global ThreadManager instance
std::unique_ptr<ThreadManager> g_thread_manager;

inline ThreadManager& get_thread_manager(MemoryManager& mem) {
    if (!g_thread_manager) {
        g_thread_manager = std::make_unique<ThreadManager>(mem);
    }
    return *g_thread_manager;
}

// Global AudioOutManager instance
std::unique_ptr<AudioOutManager> g_audio_manager;

inline AudioOutManager& get_audio_manager() {
    if (!g_audio_manager) {
        g_audio_manager = std::make_unique<AudioOutManager>();
    }
    return *g_audio_manager;
}

// Global VirtualFileSystem instance
std::unique_ptr<VirtualFileSystem> g_vfs;
VirtualFileSystem* g_external_vfs = nullptr;

inline VirtualFileSystem& get_vfs() {
    if (g_external_vfs) {
        return *g_external_vfs;
    }
    if (!g_vfs) {
        g_vfs = std::make_unique<VirtualFileSystem>();
    }
    return *g_vfs;
}

// Fios2 file handle tracking
struct FiosFile {
    std::unique_ptr<std::ifstream> stream;
    u64 size{0};
    u64 offset{0};
    std::string path;
};
static std::unordered_map<i32, FiosFile> s_fios_files;
static i32 s_next_fios_fd = 200;

} // namespace

NidRegistry::NidRegistry() = default;

void NidRegistry::set_vfs(VirtualFileSystem* vfs) {
    g_external_vfs = vfs;
}

void NidRegistry::register_symbol(Nid nid, std::string_view name, std::string_view module_name, HleFunction handler) {
    HleSymbol sym;
    sym.nid = nid;
    sym.name = std::string(name);
    sym.module_name = std::string(module_name);
    sym.handler = std::move(handler);

    m_symbols_by_nid[nid] = sym;
    m_nids_by_name[sym.name] = nid;
}

const HleSymbol* NidRegistry::find_by_nid(Nid nid) const {
    auto it = m_symbols_by_nid.find(nid);
    if (it != m_symbols_by_nid.end()) {
        return &it->second;
    }
    return nullptr;
}

const HleSymbol* NidRegistry::find_by_name(std::string_view name) const {
    auto it = m_nids_by_name.find(std::string(name));
    if (it != m_nids_by_name.end()) {
        return find_by_nid(it->second);
    }
    return nullptr;
}

void NidRegistry::register_default_libraries() {
    // =========================================================================
    // libkernel: Core Orbis OS runtime services
    // =========================================================================

    // sceKernelExitProcess(int status)
    register_symbol(0x6424D00705E3BEBEULL, "sceKernelExitProcess", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const i32 status = static_cast<i32>(ctx.rdi);
            std::cout << "[HLE] sceKernelExitProcess(" << status << ")\n";
            ctx.rax = 0;
        });

    // sceKernelLoadStartModule
    // NID: 0xC33BEA4F852A297FULL
    register_symbol(0xC33BEA4F852A297FULL, "sceKernelLoadStartModule", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            static u32 s_module_id = 0x2000;
            const GuestAddr path_addr = ctx.rdi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : "unknown";
            const u32 handle = ++s_module_id;
            std::fprintf(stderr, "[HLE] sceKernelLoadStartModule('%s') -> handle 0x%x\n", path, handle);
            ctx.rax = handle;
        });

    // sceKernelAllocateDirectMemory(off_t search_start, off_t search_end, size_t length, size_t align, int type, off_t* out_addr)
    register_symbol(0x51EAECE91C9A12B0ULL, "sceKernelAllocateDirectMemory", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize length = ctx.rdx;
            const GuestSize align  = ctx.rcx > 0 ? ctx.rcx : GUEST_PAGE_SIZE;
            const GuestAddr out_ptr = ctx.r9;

            // Align physical address
            g_direct_phys_current = (g_direct_phys_current + align - 1) & ~(align - 1);
            const GuestAddr allocated_phys = g_direct_phys_current;
            g_direct_phys_current += length;

            if (mem.is_valid_address(out_ptr, sizeof(GuestAddr))) {
                mem.write_value(out_ptr, allocated_phys);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14); // -EFAULT
            }
        });

    // sceKernelMapDirectMemory(void** out_addr, size_t length, int protection, int flags, off_t phys_addr, size_t align)
    register_symbol(0x6002A3859BC62DFBULL, "sceKernelMapDirectMemory", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr out_ptr  = ctx.rdi;
            const GuestAddr phys_addr = ctx.r8;

            // In our unified memory architecture, virtual address maps 1:1 to direct memory
            if (mem.is_valid_address(out_ptr, sizeof(GuestAddr))) {
                mem.write_value(out_ptr, phys_addr);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // sceKernelUsleep(unsigned int useconds)
    register_symbol(0x3230BAEFBC36C18BULL, "sceKernelUsleep", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            const useconds_t usec = static_cast<useconds_t>(ctx.rdi);
            ::usleep(usec);
            ctx.rax = SCE_OK;
        });

    // sceKernelVirtualQuery(const void* addr, int flags, void* info, size_t size)
    // NID: 0xAD58D1BC72745FA7ULL ("rVjRvHJ0X6c")
    register_symbol(0xAD58D1BC72745FA7ULL, "sceKernelVirtualQuery", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr addr = ctx.rdi;
            const GuestAddr info_ptr = ctx.rdx;
            const GuestSize size = ctx.rcx;

            if (addr >= 0x2000000ULL || !mem.is_valid_address(info_ptr, size)) {
                // Report SCE_KERNEL_ERROR_EACCES to signify end of mapped range
                ctx.rax = 0x8002000dULL;
                return;
            }

            mem.zero_memory(info_ptr, size);

            const u64 start = addr;
            const u64 end = 0x2000000ULL; // 32MB mapped range
            const u32 prot = 7;           // RWX
            const u8 flags = 1;           // Committed

            mem.write_value(info_ptr + 0x00, start);
            mem.write_value(info_ptr + 0x08, end);
            mem.write_value(info_ptr + 0x18, prot);
            mem.write_value(info_ptr + 0x20, flags);

            ctx.rax = SCE_OK;
        });

    // sceKernelDirectMemoryQuery(off_t phys_addr, int flags, void* info, size_t size)
    // NID: 0x047A2E2D0CE1D17DULL ("BHouLQzh0X0")
    register_symbol(0x047A2E2D0CE1D17DULL, "sceKernelDirectMemoryQuery", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr addr = ctx.rdi;
            const GuestAddr info_ptr = ctx.rdx;
            const GuestSize size = ctx.rcx;

            if (addr >= 0x2000000ULL || !mem.is_valid_address(info_ptr, size)) {
                // Report SCE_KERNEL_ERROR_EACCES to terminate the direct memory scan
                ctx.rax = 0x8002000dULL;
                return;
            }

            mem.zero_memory(info_ptr, size);

            const u64 start = addr;
            const u64 end = 0x2000000ULL;
            const u64 flags = 1;

            mem.write_value(info_ptr + 0x00, start);
            mem.write_value(info_ptr + 0x08, end);
            mem.write_value(info_ptr + 0x10, flags);

            ctx.rax = SCE_OK;
        });

    // scePthreadCreate(ScePthread* thread, const ScePthreadAttr* attr, void* (*entry)(void*), void* arg, const char* name)
    register_symbol(0x7FD4FA3050DCDE51ULL, "scePthreadCreate", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr out_thread_ptr = ctx.rdi;
            const GuestAddr attr_ptr = ctx.rsi;
            const GuestAddr entry = ctx.rdx;
            const GuestAddr arg = ctx.rcx;
            auto& tm = get_thread_manager(mem);

            ScePthreadAttr attr{};
            if (attr_ptr != 0 && mem.is_valid_address(attr_ptr, sizeof(attr))) {
                mem.read_bytes(attr_ptr, &attr, sizeof(attr));
            }
            const ScePthreadId tid = tm.create_thread(entry, arg, attr_ptr ? &attr : nullptr);
            if (out_thread_ptr != 0 && mem.is_valid_address(out_thread_ptr, sizeof(u64))) {
                mem.write_value(out_thread_ptr, static_cast<u64>(tid));
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadCreate NID alias: 0xE9482DC15FB4CDBEULL ("6UgtwV+0zb4")
    register_symbol(0xE9482DC15FB4CDBEULL, "scePthreadCreate", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr out_thread_ptr = ctx.rdi;
            const GuestAddr attr_ptr = ctx.rsi;
            const GuestAddr entry = ctx.rdx;
            const GuestAddr arg = ctx.rcx;
            auto& tm = get_thread_manager(mem);

            ScePthreadAttr attr{};
            if (attr_ptr != 0 && mem.is_valid_address(attr_ptr, sizeof(attr))) {
                mem.read_bytes(attr_ptr, &attr, sizeof(attr));
            }
            const ScePthreadId tid = tm.create_thread(entry, arg, attr_ptr ? &attr : nullptr);
            if (out_thread_ptr != 0 && mem.is_valid_address(out_thread_ptr, sizeof(u64))) {
                mem.write_value(out_thread_ptr, static_cast<u64>(tid));
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrInit NID: 0x9EC628351CB0C0D8ULL ("nsYoNRywwNg")
    register_symbol(0x9EC628351CB0C0D8ULL, "scePthreadAttrInit", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetstacksize NID: 0x5135F325B5A18531ULL ("UTXzJbWhhTE")
    register_symbol(0x5135F325B5A18531ULL, "scePthreadAttrSetstacksize", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetinheritsched NID: 0x7976D44A911A4EC0ULL ("eXbUSpEaTsA")
    register_symbol(0x7976D44A911A4EC0ULL, "scePthreadAttrSetinheritsched", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetschedpolicy NID: 0xE3E87D133C0A1782ULL ("4+h9EzwKF4I")
    register_symbol(0xE3E87D133C0A1782ULL, "scePthreadAttrSetschedpolicy", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrSetaffinity NID: 0x1573D61CD93C39FDULL ("FXPWHNk8Of0")
    register_symbol(0x1573D61CD93C39FDULL, "scePthreadAttrSetaffinity", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadAttrDestroy NID: 0xEB6282C04326CDC3ULL ("62KCwEMmzcM")
    register_symbol(0xEB6282C04326CDC3ULL, "scePthreadAttrDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadSetaffinity NID: 0x6EDDC24C12A61B22ULL ("bt3CTBKmGyI")
    register_symbol(0x6EDDC24C12A61B22ULL, "scePthreadSetaffinity", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadJoin(ScePthread thread, void** value_ptr)
    register_symbol(0xC4F0E23991209B0FULL, "scePthreadJoin", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const ScePthreadId tid = static_cast<ScePthreadId>(ctx.rdi);
            const GuestAddr val_ptr = ctx.rsi;
            auto& tm = get_thread_manager(mem);
            u64 ret_val = 0;
            if (tm.join_thread(tid, &ret_val)) {
                if (val_ptr != 0 && mem.is_valid_address(val_ptr, sizeof(u64))) {
                    mem.write_value(val_ptr, ret_val);
                }
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-3); // ESRCH
            }
        });

    // scePthreadSelf()
    register_symbol(0xC052F1176214A36BULL, "scePthreadSelf", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            auto& tm = get_thread_manager(mem);
            ctx.rax = static_cast<u64>(tm.current_thread_id());
        });

    // scePthreadMutexInit(ScePthreadMutex* mutex, const ScePthreadMutexattr* attr, const char* name)
    register_symbol(0x429E57A7F66472D5ULL, "scePthreadMutexInit", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            const u32 mid = tm.mutex_create();
            if (mtx_ptr != 0 && mem.is_valid_address(mtx_ptr, sizeof(u32))) {
                mem.write_value(mtx_ptr, mid);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14); // EFAULT
            }
        });

    // scePthreadMutexLock(ScePthreadMutex* mutex)
    register_symbol(0xE86895318FDC64A0ULL, "scePthreadMutexLock", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 mid = 0;
            if (mem.read_value(mtx_ptr, mid)) {
                if (mid == 0) { // Auto-init on zeroed memory
                    mid = tm.mutex_create();
                    mem.write_value(mtx_ptr, mid);
                }
                ctx.rax = tm.mutex_lock(mid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadMutexTrylock(ScePthreadMutex* mutex)
    register_symbol(0x98150D89A0831626ULL, "scePthreadMutexTrylock", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 mid = 0;
            if (mem.read_value(mtx_ptr, mid)) {
                if (mid == 0) {
                    mid = tm.mutex_create();
                    mem.write_value(mtx_ptr, mid);
                }
                ctx.rax = tm.mutex_trylock(mid) ? SCE_OK : static_cast<u64>(-16); // EBUSY
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadMutexUnlock(ScePthreadMutex* mutex)
    register_symbol(0xEFB2023594895C85ULL, "scePthreadMutexUnlock", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 mid = 0;
            if (mem.read_value(mtx_ptr, mid)) {
                ctx.rax = tm.mutex_unlock(mid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadMutexLock (alternate NID: 0xF542B5BCB6507EDEULL / "9UK1vLZQft4")
    register_symbol(0xF542B5BCB6507EDEULL, "scePthreadMutexLock", "libkernel",
        [this](CpuContext& ctx, MemoryManager& mem) {
            const auto* s = find_by_nid(0xE86895318FDC64A0ULL);
            if (s && s->handler) s->handler(ctx, mem);
        });

    // scePthreadMutexUnlock (alternate NID: 0xB67DD5943D211BADULL / "tn3VlD0hG60")
    register_symbol(0xB67DD5943D211BADULL, "scePthreadMutexUnlock", "libkernel",
        [this](CpuContext& ctx, MemoryManager& mem) {
            const auto* s = find_by_nid(0xEFB2023594895C85ULL);
            if (s && s->handler) s->handler(ctx, mem);
        });

    // scePthreadYield()
    // NID: 0x4FBDA1CFA7DFAB4FULL ("T72hz6ffq08")
    register_symbol(0x4FBDA1CFA7DFAB4FULL, "scePthreadYield", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexDestroy(ScePthreadMutex* mutex)
    register_symbol(0x6F1BCDF3BDC5D06AULL, "scePthreadMutexDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 mid = 0;
            if (mem.read_value(mtx_ptr, mid)) {
                ctx.rax = tm.mutex_destroy(mid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondInit(ScePthreadCond* cond, const ScePthreadCondattr* attr, const char* name)
    register_symbol(0x8CD7808E58BAF551ULL, "scePthreadCondInit", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            const u32 cid = tm.cond_create();
            if (cond_ptr != 0 && mem.is_valid_address(cond_ptr, sizeof(u32))) {
                mem.write_value(cond_ptr, cid);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondWait(ScePthreadCond* cond, ScePthreadMutex* mutex)
    register_symbol(0x296D8E4AE9BFFDF0ULL, "scePthreadCondWait", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            const GuestAddr mtx_ptr = ctx.rsi;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0, mid = 0;
            if (mem.read_value(cond_ptr, cid) && mem.read_value(mtx_ptr, mid)) {
                ctx.rax = tm.cond_wait(cid, mid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondSignal(ScePthreadCond* cond)
    register_symbol(0x3231F2A7EF228E10ULL, "scePthreadCondSignal", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0;
            if (mem.read_value(cond_ptr, cid)) {
                ctx.rax = tm.cond_signal(cid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondBroadcast(ScePthreadCond* cond)
    register_symbol(0x1B8682F5CF60505FULL, "scePthreadCondBroadcast", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0;
            if (mem.read_value(cond_ptr, cid)) {
                ctx.rax = tm.cond_broadcast(cid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // scePthreadCondDestroy(ScePthreadCond* cond)
    register_symbol(0x6C46F0C0F39695EFULL, "scePthreadCondDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr cond_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            u32 cid = 0;
            if (mem.read_value(cond_ptr, cid)) {
                ctx.rax = tm.cond_destroy(cid) ? SCE_OK : static_cast<u64>(-22);
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // sceKernelGettimeofday(struct timeval* tp, struct timezone* tzp)
    register_symbol(0x78D1E6F34C96C7A1ULL, "sceKernelGettimeofday", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr tp_ptr = ctx.rdi;
            if (tp_ptr != 0 && mem.is_valid_address(tp_ptr, 16)) {
                struct timeval tv{};
                ::gettimeofday(&tv, nullptr);
                mem.write_value(tp_ptr, static_cast<i64>(tv.tv_sec));
                mem.write_value(tp_ptr + 8, static_cast<i64>(tv.tv_usec));
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // =========================================================================
    // libc: Standard C library services
    // =========================================================================

    // puts(const char* s)
    register_symbol(0x5C0F663677A000A8ULL, "puts", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr str_addr = ctx.rdi;
            if (mem.is_valid_address(str_addr)) {
                const char* str = mem.to_host_ptr<char>(str_addr);
                if (str) {
                    std::cout << str << "\n";
                    ctx.rax = 0;
                    return;
                }
            }
            ctx.rax = static_cast<u64>(-1);
        });

    // malloc(size_t size)
    register_symbol(0x9E990E49EC698A4DULL, "malloc", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize size = ctx.rdi;
            g_guest_heap_current = (g_guest_heap_current + 15) & ~15ULL;
            const GuestAddr allocated = g_guest_heap_current;
            g_guest_heap_current += size;
            if (size > 0 && mem.is_valid_address(allocated, size)) {
                mem.zero_memory(allocated, size);
            }
            ctx.rax = allocated;
        });

    // free(void* ptr)
    register_symbol(0x7E16B7B8361BD5BBULL, "free", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });
    register_symbol(0xB4886CAA3D2AB051ULL, "free", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0;
        });

    // sceCommonDialogInitialize
    register_symbol(0xBA85292C6364CA09ULL, "sceCommonDialogInitialize", "libSceCommonDialog",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // memset(void* dest, int ch, size_t n)
    // NID: 0xF334C5BC120020DFULL ("8zTFvBIAIN8")
    register_symbol(0xF334C5BC120020DFULL, "memset", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const u8 ch          = static_cast<u8>(ctx.rsi & 0xFF);
            const GuestSize n    = ctx.rdx;
            if (n > 0 && mem.is_valid_address(dest, n)) {
                void* d_ptr = mem.to_host_ptr<void>(dest);
                if (d_ptr) {
                    std::memset(d_ptr, ch, n);
                }
            }
            ctx.rax = dest;
        });

    // snprintf(char* str, size_t size, const char* format, ...)
    // NID: 0x78B743C3A974FDB5ULL ("eLdDw6l0-bU")
    register_symbol(0x78B743C3A974FDB5ULL, "snprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr buf = ctx.rdi;
            const GuestSize sz  = ctx.rsi;
            const GuestAddr fmt_addr = ctx.rdx;
            const char* fmt = (fmt_addr > 0x1000 && mem.is_valid_address(fmt_addr, 1))
                                ? mem.to_host_ptr<char>(fmt_addr) : nullptr;
            if (sz > 0 && mem.is_valid_address(buf, 1)) {
                char* dst = mem.to_host_ptr<char>(buf);
                if (fmt && dst) {
                    // Try formatting
                    // For string / int args
                    const char* s_arg = (ctx.rcx > 0x1000 && mem.is_valid_address(ctx.rcx, 1))
                                            ? mem.to_host_ptr<char>(ctx.rcx) : nullptr;
                    if (s_arg && std::strcmp(fmt, "%s") == 0) {
                        ctx.rax = std::snprintf(dst, sz, "%s", s_arg);
                    } else if (s_arg) {
                        ctx.rax = std::snprintf(dst, sz, fmt, s_arg, ctx.r8, ctx.r9);
                    } else {
                        ctx.rax = std::snprintf(dst, sz, fmt, ctx.rcx, ctx.r8, ctx.r9);
                    }
                    return;
                }
                mem.write_value(buf, static_cast<u8>(0));
            }
            ctx.rax = 0;
        });

    // vsnprintf(char* str, size_t size, const char* format, va_list ap)
    // NID: 0x43657E8AABE3802DULL ("Q2V+iqvjgC0")
    register_symbol(0x43657E8AABE3802DULL, "vsnprintf", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestSize maxlen = ctx.rsi;
            const GuestAddr fmt_addr = ctx.rdx;
            if (dest == 0 || maxlen == 0 || fmt_addr == 0 || !mem.is_valid_address(dest, 1)) {
                ctx.rax = 0;
                return;
            }
            const char* fmt = (fmt_addr > 0x1000 && mem.is_valid_address(fmt_addr, 1))
                                ? mem.to_host_ptr<char>(fmt_addr) : nullptr;
            char* out = mem.to_host_ptr<char>(dest);
            if (!fmt || !out) {
                ctx.rax = 0;
                return;
            }
            const int written = std::snprintf(out, maxlen, "%s", fmt);
            ctx.rax = static_cast<u64>(written >= 0 ? written : 0);
        });

    // strncpy(char* dest, const char* src, size_t n)
    // NID: 0xEAC256896491BAA9ULL ("6sJWiWSRuqk")
    register_symbol(0xEAC256896491BAA9ULL, "strncpy", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestAddr src  = ctx.rsi;
            const GuestSize len  = ctx.rdx;
            if (len > 0 && mem.is_valid_address(dest, len) && mem.is_valid_address(src, 1)) {
                bool null_seen = false;
                for (GuestSize i = 0; i < len; ++i) {
                    u8 b = 0;
                    if (!null_seen && mem.read_value(src + i, b)) {
                        if (b == 0) null_seen = true;
                    } else {
                        b = 0;
                    }
                    mem.write_value(dest + i, b);
                }
            }
            ctx.rax = dest;
        });

    // strncmp(const char* s1, const char* s2, size_t n)
    // NID: 0x69EB328EB1D55B2EULL ("aesyjrHVWy4")
    register_symbol(0x69EB328EB1D55B2EULL, "strncmp", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr s1_addr = ctx.rdi;
            const GuestAddr s2_addr = ctx.rsi;
            const GuestSize n = ctx.rdx;
            if (n == 0) {
                ctx.rax = 0;
                return;
            }
            const char* s1 = (s1_addr > 0x1000 && mem.is_valid_address(s1_addr, 1))
                                ? mem.to_host_ptr<char>(s1_addr) : nullptr;
            const char* s2 = (s2_addr > 0x1000 && mem.is_valid_address(s2_addr, 1))
                                ? mem.to_host_ptr<char>(s2_addr) : nullptr;
            if (!s1 || !s2) {
                ctx.rax = (s1 == s2) ? 0 : (s1 ? 1 : -1);
                return;
            }
            ctx.rax = static_cast<u64>(static_cast<i64>(std::strncmp(s1, s2, n)));
        });

    // strlen(const char* s)
    // NID: 0x8F856258D1C4830CULL ("j4ViWNHEgww")
    register_symbol(0x8F856258D1C4830CULL, "strlen", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr str = ctx.rdi;
            u64 len = 0;
            u8 ch = 0;
            while (mem.read_value(str + len, ch) && ch != 0 && len < 4096) {
                len++;
            }
            ctx.rax = len;
        });

    // strstr(const char* haystack, const char* needle)
    // NID: 0xBE28B014C68D6A60ULL ("viiwFMaNamA")
    register_symbol(0xBE28B014C68D6A60ULL, "strstr", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr hay = ctx.rdi;
            const GuestAddr needle = ctx.rsi;
            u8 n0 = 0;
            if (!mem.read_value(needle, n0) || n0 == 0) {
                ctx.rax = hay;
                return;
            }
            u64 hlen = 0;
            u8 ch = 0;
            while (mem.read_value(hay + hlen, ch) && ch != 0 && hlen < 4096) {
                if (ch == n0) {
                    bool match = true;
                    u64 nlen = 1;
                    while (true) {
                        u8 nch = 0;
                        if (!mem.read_value(needle + nlen, nch) || nch == 0) break;
                        u8 hch = 0;
                        if (!mem.read_value(hay + hlen + nlen, hch) || hch != nch) {
                            match = false;
                            break;
                        }
                        nlen++;
                    }
                    if (match) {
                        ctx.rax = hay + hlen;
                        return;
                    }
                }
                hlen++;
            }
            ctx.rax = 0;
        });

    // sceKernelCreateSema(SceKernelSema* sema, const char* name, uint32_t attr, int initCount, int maxCount, void* opt)
    // NID: 0xD7CF31E7B258A748ULL ("188x57JYp0g")
    register_symbol(0xD7CF31E7B258A748ULL, "sceKernelCreateSema", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr sema_ptr = ctx.rdi;
            if (sema_ptr != 0 && mem.is_valid_address(sema_ptr, sizeof(u32))) {
                static u32 sema_id = 1;
                mem.write_value(sema_ptr, sema_id++);
            }
            ctx.rax = SCE_OK;
        });

    // sceKernelDeleteSema(SceKernelSema sema)
    // NID: 0x47526F9FC6D2096FULL ("R1Jvn8bSCW8")
    register_symbol(0x47526F9FC6D2096FULL, "sceKernelDeleteSema", "libkernel",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = SCE_OK;
        });

    // sceKernelIsNeoMode()
    // NID: 0x5AC95C2B51507062ULL ("WslcK1FQcGI")
    register_symbol(0x5AC95C2B51507062ULL, "sceKernelIsNeoMode", "libkernel",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = 0; // Standard PS4 Base mode
        });

    // scePthreadMutexInit NID: 0x726A3544862F6BDAULL ("cmo1RIYva9o")
    register_symbol(0x726A3544862F6BDAULL, "scePthreadMutexInit", "libkernel",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mtx_ptr = ctx.rdi;
            auto& tm = get_thread_manager(mem);
            const u32 mid = tm.mutex_create();
            if (mtx_ptr != 0 && mem.is_valid_address(mtx_ptr, sizeof(u32))) {
                mem.write_value(mtx_ptr, mid);
            }
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexattrInit NID: 0x17C6D41F0006DBCEULL ("F8bUHwAG284")
    register_symbol(0x17C6D41F0006DBCEULL, "scePthreadMutexattrInit", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexattrSettype NID: 0x88CA7C42913E5CEEULL ("iMp8QpE+XO4")
    register_symbol(0x88CA7C42913E5CEEULL, "scePthreadMutexattrSettype", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexattrSetprotocol NID: 0xD451AF5348BDB1A4ULL ("1FGvU0i9saQ")
    register_symbol(0xD451AF5348BDB1A4ULL, "scePthreadMutexattrSetprotocol", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // scePthreadMutexattrDestroy NID: 0xB2658492D8B2C86DULL ("smWEktiyyG0")
    register_symbol(0xB2658492D8B2C86DULL, "scePthreadMutexattrDestroy", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // libkernel synchronization helper NID: 0x688F8E782CFCC6B4ULL ("aI+OeCz8xrQ")
    register_symbol(0x688F8E782CFCC6B4ULL, "sceKernelSyncStub", "libkernel",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = SCE_OK;
        });

    // memalign(size_t alignment, size_t size)
    // NID: 0x5237F72B332F4662ULL ("Ujf3KzMvRmI")
    register_symbol(0x5237F72B332F4662ULL, "memalign", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestSize alignment = ctx.rdi > 0 ? ctx.rdi : 16;
            const GuestSize size = ctx.rsi;
            g_guest_heap_current = (g_guest_heap_current + alignment - 1) & ~(alignment - 1);
            const GuestAddr allocated = g_guest_heap_current;
            g_guest_heap_current += size;
            if (size > 0 && mem.is_valid_address(allocated, size)) {
                mem.zero_memory(allocated, size);
            }
            ctx.rax = allocated;
        });

    // memcpy(void* dest, const void* src, size_t n)
    // NID: 0x437541C425E1507BULL ("Q3VBxCXhUHs")
    register_symbol(0x437541C425E1507BULL, "memcpy", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr dest = ctx.rdi;
            const GuestAddr src  = ctx.rsi;
            const GuestSize n    = ctx.rdx;
            if (n > 0 && mem.is_valid_address(dest, n) && mem.is_valid_address(src, n)) {
                const void* s_ptr = mem.to_host_ptr<void>(src);
                void* d_ptr = mem.to_host_ptr<void>(dest);
                if (s_ptr && d_ptr) {
                    std::memcpy(d_ptr, s_ptr, n);
                }
            }
            ctx.rax = dest;
        });

    // __cxa_atexit(void (*func)(void*), void* arg, void* dso_handle)
    // NID: 0xB6CBC49A77A7CF8FULL ("tsvEmnenz48")
    register_symbol(0xB6CBC49A77A7CF8FULL, "__cxa_atexit", "libc",
        [](CpuContext& ctx, MemoryManager&) {
            ctx.rax = 0; // Success
        });

    // strcmp(const char* s1, const char* s2)
    // NID: 0x3AF6F675224E02E1ULL ("Ovb2dSJOAuE")
    register_symbol(0x3AF6F675224E02E1ULL, "strcmp", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr s1 = ctx.rdi;
            const GuestAddr s2 = ctx.rsi;
            if (s1 == s2) {
                ctx.rax = 0;
                return;
            }
            if (!mem.is_valid_address(s1, 1) || !mem.is_valid_address(s2, 1)) {
                ctx.rax = (s1 < s2) ? static_cast<u64>(-1) : 1;
                return;
            }
            u64 idx = 0;
            i32 diff = 0;
            while (idx < 4096) {
                u8 c1 = 0, c2 = 0;
                if (!mem.read_value(s1 + idx, c1) || !mem.read_value(s2 + idx, c2)) {
                    diff = (s1 < s2) ? -1 : 1;
                    break;
                }
                if (c1 != c2) {
                    diff = static_cast<i32>(c1) - static_cast<i32>(c2);
                    break;
                }
                if (c1 == 0) break;
                idx++;
            }
            ctx.rax = static_cast<u64>(static_cast<i64>(diff));
        });

    // __cxa_guard_acquire(int64_t* guard)
    // NID: 0xDC63E98D0740313CULL ("3GPpjQdAMTw")
    register_symbol(0xDC63E98D0740313CULL, "__cxa_guard_acquire", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr guard_ptr = ctx.rdi;
            u8 val = 0;
            if (guard_ptr != 0 && mem.read_value(guard_ptr, val)) {
                if (val == 0) {
                    mem.write_value(guard_ptr, static_cast<u8>(1));
                    ctx.rax = 1; // Needs initialization
                    return;
                }
            }
            ctx.rax = 0; // Already initialized
        });

    // __cxa_guard_release(int64_t* guard)
    // NID: 0xF6B01E00D4F6B721ULL ("9rAeANT2tyE")
    register_symbol(0xF6B01E00D4F6B721ULL, "__cxa_guard_release", "libc",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr guard_ptr = ctx.rdi;
            if (guard_ptr != 0 && mem.is_valid_address(guard_ptr, 1)) {
                mem.write_value(guard_ptr, static_cast<u8>(1));
            }
            ctx.rax = 0;
        });

    // =========================================================================
    // libaudioout: PlayStation 4 Audio Output Services
    // =========================================================================

    // sceAudioOutOpen(SceUserServiceUserId userId, int32_t type, int32_t index, uint32_t paramLen, const void* param)
    register_symbol(0xE1D5176E5161A4FEULL, "sceAudioOutOpen", "libaudioout",
        [](CpuContext& ctx, MemoryManager&) {
            const i32 type = static_cast<i32>(ctx.rsi);
            auto& am = get_audio_manager();
            const i32 handle = am.open_port(type);
            ctx.rax = static_cast<u64>(handle);
        });

    // sceAudioOutClose(int32_t handle)
    register_symbol(0xEAA1CDA949437E55ULL, "sceAudioOutClose", "libaudioout",
        [](CpuContext& ctx, MemoryManager&) {
            const i32 handle = static_cast<i32>(ctx.rdi);
            auto& am = get_audio_manager();
            ctx.rax = am.close_port(handle) ? SCE_OK : static_cast<u64>(-1);
        });

    // sceAudioOutOutput(int32_t handle, const void* ptr)
    register_symbol(0x69661413865EE60CULL, "sceAudioOutOutput", "libaudioout",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 handle = static_cast<i32>(ctx.rdi);
            const GuestAddr ptr = ctx.rsi;
            auto& am = get_audio_manager();

            constexpr usize SAMPLE_COUNT = 512 * 2; // 512 stereo frames
            constexpr usize BYTE_SIZE = SAMPLE_COUNT * sizeof(i16);

            if (ptr != 0 && mem.is_valid_address(ptr, BYTE_SIZE)) {
                const auto* pcm = mem.to_host_ptr<i16>(ptr);
                if (pcm) {
                    std::span<const i16> samples(pcm, SAMPLE_COUNT);
                    am.output_samples(handle, samples);
                    ctx.rax = SCE_OK;
                    return;
                }
            }
            ctx.rax = static_cast<u64>(-14);
        });

    // sceAudioOutSetVolume(int32_t handle, int32_t flag, int32_t* vol)
    register_symbol(0x8F596D3C1F4FA286ULL, "sceAudioOutSetVolume", "libaudioout",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 handle = static_cast<i32>(ctx.rdi);
            const GuestAddr vol_ptr = ctx.rdx;
            auto& am = get_audio_manager();

            if (vol_ptr != 0 && mem.is_valid_address(vol_ptr, 8)) {
                i32 vol_l = 0, vol_r = 0;
                mem.read_value(vol_ptr, vol_l);
                mem.read_value(vol_ptr + 4, vol_r);
                const f32 left = static_cast<f32>(vol_l) / static_cast<f32>(SCE_AUDIO_OUT_VOLUME_MAX);
                const f32 right = static_cast<f32>(vol_r) / static_cast<f32>(SCE_AUDIO_OUT_VOLUME_MAX);
                am.set_volume(handle, left, right);
                ctx.rax = SCE_OK;
            } else {
                ctx.rax = static_cast<u64>(-14);
            }
        });

    // sceAudioOutGetPortInfo(int32_t handle, SceAudioOutPortInfo* info)
    register_symbol(0xD9BDDE84CE9A1E02ULL, "sceAudioOutGetPortInfo", "libaudioout",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 handle = static_cast<i32>(ctx.rdi);
            const GuestAddr info_ptr = ctx.rsi;
            auto& am = get_audio_manager();

            if (info_ptr != 0 && mem.is_valid_address(info_ptr, sizeof(SceAudioOutPortInfo))) {
                SceAudioOutPortInfo info{};
                if (am.get_port_info(handle, info)) {
                    mem.write_bytes(info_ptr, &info, sizeof(info));
                    ctx.rax = SCE_OK;
                    return;
                }
            }
            ctx.rax = static_cast<u64>(-1);
        });

    // =========================================================================
    // libsave_data: PlayStation 4 Save Data Services
    // =========================================================================

    // sceSaveDataMount(const SceSaveDataMount* mount, SceSaveDataMountResult* result)
    register_symbol(0x948296238BC4F301ULL, "sceSaveDataMount", "libsave_data",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr mount_ptr = ctx.rdi;
            const GuestAddr result_ptr = ctx.rsi;
            auto& vfs = get_vfs();

            if (mount_ptr != 0 && mem.is_valid_address(mount_ptr, sizeof(SceSaveDataMount)) &&
                result_ptr != 0 && mem.is_valid_address(result_ptr, sizeof(SceSaveDataMountResult))) {
                SceSaveDataMount mount_req{};
                mem.read_bytes(mount_ptr, &mount_req, sizeof(mount_req));

                SceSaveDataMountResult mount_res{};
                const i32 ret = vfs.mount_savedata(mount_req.user_id, mount_req.dir_name.data, mount_req.mount_mode, mount_res);
                if (ret == SCE_OK) {
                    mem.write_bytes(result_ptr, &mount_res, sizeof(mount_res));
                    ctx.rax = SCE_OK;
                    return;
                }
            }
            ctx.rax = static_cast<u64>(-14);
        });

    // sceSaveDataUmount(const SceSaveDataMountResult* result)
    register_symbol(0x32791B317075AB6EULL, "sceSaveDataUmount", "libsave_data",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr result_ptr = ctx.rdi;
            auto& vfs = get_vfs();

            if (result_ptr != 0 && mem.is_valid_address(result_ptr, sizeof(SceSaveDataMountResult))) {
                SceSaveDataMountResult mount_res{};
                mem.read_bytes(result_ptr, &mount_res, sizeof(mount_res));
                const i32 ret = vfs.unmount_savedata(mount_res.mount_path);
                ctx.rax = (ret == SCE_OK) ? SCE_OK : static_cast<u64>(-1);
                return;
            }
            ctx.rax = static_cast<u64>(-14);
        });

    // ==========================================
    // libSceFios2 - Fast I/O System 2
    // ==========================================

    // sceFiosFileExists(int profile, const char* path, ...)
    // NID: 0x3703873113363E9CULL ("NwOHMRM2Ppw")
    register_symbol(0x3703873113363E9CULL, "sceFiosFileExists", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = 0;
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            bool exists = false;
            if (resolved) {
                std::error_code ec;
                exists = std::filesystem::is_regular_file(*resolved, ec);
            }
            std::fprintf(stderr, "[FIOS2] sceFiosFileExists('%s') -> %d\n", path, exists ? 1 : 0);
            ctx.rax = exists ? 1 : 0;
        });

    // sceFiosDirectoryExists(int profile, const char* path, ...)
    // NID: 0x38EBAF1CA4EEE0E7ULL ("OOuvHKTu4Oc")
    register_symbol(0x38EBAF1CA4EEE0E7ULL, "sceFiosDirectoryExists", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = 0;
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            bool exists = false;
            if (resolved) {
                std::error_code ec;
                exists = std::filesystem::is_directory(*resolved, ec);
            }
            std::fprintf(stderr, "[FIOS2] sceFiosDirectoryExists('%s') -> %d\n", path, exists ? 1 : 0);
            ctx.rax = exists ? 1 : 0;
        });

    // sceFiosStat / sceFiosFileGetStatSync(int profile, const char* path, SceFiosStat* stat)
    // NID: 0x8DACAF634EC2E5D9ULL ("jayvY07C5dk")
    register_symbol(0x8DACAF634EC2E5D9ULL, "sceFiosStat", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const GuestAddr stat_ptr  = ctx.rdx;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path || stat_ptr == 0 || !mem.is_valid_address(stat_ptr, 0x40)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                std::fprintf(stderr, "[FIOS2] sceFiosStat('%s') -> not found\n", path);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::error_code ec;
            const auto st = std::filesystem::status(*resolved, ec);
            if (!std::filesystem::exists(st)) {
                std::fprintf(stderr, "[FIOS2] sceFiosStat('%s' -> '%s') -> not found\n", path, resolved->c_str());
                ctx.rax = static_cast<u64>(-1);
                return;
            }

            mem.zero_memory(stat_ptr, 0x40);
            const bool is_dir = std::filesystem::is_directory(st);
            const u64 file_sz = is_dir ? 0 : std::filesystem::file_size(*resolved, ec);
            const u32 stat_flags = is_dir ? 0x1 : 0x0;

            mem.write_value(stat_ptr + 0x00, file_sz);
            mem.write_value(stat_ptr + 0x20, stat_flags);

            std::fprintf(stderr, "[FIOS2] sceFiosStat('%s') -> size=%llu, flags=0x%x\n",
                         path, (unsigned long long)file_sz, stat_flags);
            ctx.rax = SCE_OK;
        });

    // sceFiosFHOpen / sceFiosFHOpenSync(int profile, const char* path, ...)
    // NID: 0xCC5F3F091BD15E73ULL ("zF8-CRvRXnM")
    register_symbol(0xCC5F3F091BD15E73ULL, "sceFiosFHOpen", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHOpen('%s') -> not found, returning -1\n", path);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::error_code ec;
            if (!std::filesystem::is_regular_file(*resolved, ec)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHOpen('%s' -> '%s') -> not regular file\n", path, resolved->c_str());
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto stream = std::make_unique<std::ifstream>(*resolved, std::ios::binary);
            if (!stream->is_open()) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHOpen('%s' -> '%s') -> open failed\n", path, resolved->c_str());
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            const u64 sz = std::filesystem::file_size(*resolved, ec);
            const i32 fd = s_next_fios_fd++;
            s_fios_files[fd] = FiosFile{
                .stream = std::move(stream),
                .size = sz,
                .offset = 0,
                .path = path
            };
            std::fprintf(stderr, "[FIOS2] sceFiosFHOpen('%s' -> '%s') -> handle %d (size=%llu)\n",
                         path, resolved->c_str(), fd, (unsigned long long)sz);
            ctx.rax = static_cast<u64>(fd);
        });

    // sceFiosFHRead / sceFiosFHReadSync(int profile, int fh, void* buf, size_t size)
    // NID: 0x2A5FD36EB0D4F583ULL ("Kl-TbrDU9YM")
    register_symbol(0x2A5FD36EB0D4F583ULL, "sceFiosFHRead", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 fd = static_cast<i32>(ctx.rsi);
            const GuestSize size = ctx.rdx;
            const GuestAddr buf_addr = ctx.rcx;
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end() || !it->second.stream || !it->second.stream->is_open()) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHRead: invalid handle %d\n", fd);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (buf_addr == 0 || !mem.is_valid_address(buf_addr, size)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHRead: invalid buffer 0x%llx (size=%llu)\n",
                             (unsigned long long)buf_addr, (unsigned long long)size);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            u8* dest = mem.to_host_ptr<u8>(buf_addr);
            it->second.stream->seekg(it->second.offset, std::ios::beg);
            it->second.stream->read(reinterpret_cast<char*>(dest), size);
            const std::streamsize bytes_read = it->second.stream->gcount();
            it->second.offset += static_cast<u64>(bytes_read);
            std::fprintf(stderr, "[FIOS2] sceFiosFHRead(fd=%d, size=%llu) -> read %lld bytes\n",
                         fd, (unsigned long long)size, (long long)bytes_read);
            ctx.rax = static_cast<u64>(bytes_read);
        });

    // sceFiosFHPread / sceFiosFHPreadSync(int profile, int fh, size_t offset, size_t size, void* buf)
    // NID: 0x80C71F3AD1D6EB39ULL ("gMcfOtHW6zk")
    register_symbol(0x80C71F3AD1D6EB39ULL, "sceFiosFHPread", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const i32 fd = static_cast<i32>(ctx.rsi);
            const u64 offset = ctx.rdx;
            const GuestSize size = ctx.rcx;
            const GuestAddr buf_addr = ctx.r8;
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end() || !it->second.stream || !it->second.stream->is_open()) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHPread: invalid handle %d\n", fd);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (buf_addr == 0 || !mem.is_valid_address(buf_addr, size)) {
                std::fprintf(stderr, "[FIOS2] sceFiosFHPread: invalid buffer 0x%llx (size=%llu)\n",
                             (unsigned long long)buf_addr, (unsigned long long)size);
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            u8* dest = mem.to_host_ptr<u8>(buf_addr);
            it->second.stream->seekg(offset, std::ios::beg);
            it->second.stream->read(reinterpret_cast<char*>(dest), size);
            const std::streamsize bytes_read = it->second.stream->gcount();
            std::fprintf(stderr, "[FIOS2] sceFiosFHPread(fd=%d, offset=%llu, size=%llu) -> read %lld bytes\n",
                         fd, (unsigned long long)offset, (unsigned long long)size, (long long)bytes_read);
            ctx.rax = static_cast<u64>(bytes_read);
        });

    // sceFiosFHGetSize / sceFiosFHStat(int fh)
    // NID: 0x32B445ADD829B31FULL ("MrRFrdgpsx8")
    register_symbol(0x32B445ADD829B31FULL, "sceFiosFHGetSize", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            const i32 fd = static_cast<i32>(ctx.rdi);
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end()) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::fprintf(stderr, "[FIOS2] sceFiosFHGetSize(fd=%d) -> %llu\n", fd, (unsigned long long)it->second.size);
            ctx.rax = it->second.size;
        });

    // sceFiosFHSeek(int fh, int64_t offset, int whence)
    // NID: 0xC5179279BC0A0290ULL ("xReSebwKApA")
    register_symbol(0xC5179279BC0A0290ULL, "sceFiosFHSeek", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            const i32 fd = static_cast<i32>(ctx.rdi);
            const i64 offset = static_cast<i64>(ctx.rsi);
            const i32 whence = static_cast<i32>(ctx.rdx);
            auto it = s_fios_files.find(fd);
            if (it == s_fios_files.end()) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            if (whence == 0) { // SEEK_SET
                it->second.offset = static_cast<u64>(offset >= 0 ? offset : 0);
            } else if (whence == 1) { // SEEK_CUR
                it->second.offset = static_cast<u64>(std::max<i64>(0, static_cast<i64>(it->second.offset) + offset));
            } else if (whence == 2) { // SEEK_END
                it->second.offset = static_cast<u64>(std::max<i64>(0, static_cast<i64>(it->second.size) + offset));
            }
            std::fprintf(stderr, "[FIOS2] sceFiosFHSeek(fd=%d, offset=%lld, whence=%d) -> pos %llu\n",
                         fd, (long long)offset, whence, (unsigned long long)it->second.offset);
            ctx.rax = it->second.offset;
        });

    // sceFiosFHClose(int profile, int fh)
    // NID: 0x00EBA3486A94FA6BULL ("AOujSGqU+ms")
    register_symbol(0x00EBA3486A94FA6BULL, "sceFiosFHClose", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            const i32 fd = static_cast<i32>(ctx.rsi);
            s_fios_files.erase(fd);
            std::fprintf(stderr, "[FIOS2] sceFiosFHClose(fd=%d) -> closed\n", fd);
            ctx.rax = SCE_OK;
        });

    // sceFiosExists(int profile, const char* path)
    // NID: 0x9D6BB36B465D7EA0ULL ("nWuza0ZdfqA")
    register_symbol(0x9D6BB36B465D7EA0ULL, "sceFiosExists", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = 0;
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            bool exists = false;
            if (resolved) {
                std::error_code ec;
                exists = std::filesystem::exists(*resolved, ec);
            }
            std::fprintf(stderr, "[FIOS2] sceFiosExists('%s') -> %d\n", path, exists ? 1 : 0);
            ctx.rax = exists ? 1 : 0;
        });

    // sceFiosFileDelete(int profile, const char* path)
    // NID: 0x6C3BA91206D0E859ULL ("bDupEgbQ6Fk")
    register_symbol(0x6C3BA91206D0E859ULL, "sceFiosFileDelete", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = SCE_OK;
        });

    // sceFiosFileGetSize(int profile, const char* path)
    // NID: 0x514AE2697CBB1BDDULL ("UUriaXy7G90")
    register_symbol(0x514AE2697CBB1BDDULL, "sceFiosFileGetSize", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::error_code ec;
            const u64 sz = std::filesystem::file_size(*resolved, ec);
            ctx.rax = ec ? static_cast<u64>(-1) : sz;
        });

    // sceFiosFileRead(int profile, const char* path, void* buf, size_t size, ...)
    // NID: 0xC6EB4B6D076AC9BEULL ("xutLbQdqyb4")
    register_symbol(0xC6EB4B6D076AC9BEULL, "sceFiosFileRead", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& mem) {
            const GuestAddr path_addr = ctx.rsi;
            const GuestAddr buf_addr  = ctx.rdx;
            const GuestSize size      = ctx.rcx;
            const char* path = (path_addr > 0x1000 && mem.is_valid_address(path_addr, 1))
                                   ? mem.to_host_ptr<char>(path_addr) : nullptr;
            if (!path || buf_addr == 0 || !mem.is_valid_address(buf_addr, size)) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            auto& vfs = get_vfs();
            auto resolved = vfs.resolve_path(path);
            if (!resolved) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            std::ifstream file(*resolved, std::ios::binary);
            if (!file.is_open()) {
                ctx.rax = static_cast<u64>(-1);
                return;
            }
            file.read(reinterpret_cast<char*>(mem.to_host_ptr<u8>(buf_addr)), size);
            ctx.rax = static_cast<u64>(file.gcount());
        });

    // sceFiosInitialize(const void* params)
    // NID: 0x960213B81B11A36AULL ("lgITuBsRo2o")
    register_symbol(0x960213B81B11A36AULL, "sceFiosInitialize", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            std::fprintf(stderr, "[FIOS2] sceFiosInitialize() -> SCE_OK\n");
            ctx.rax = SCE_OK;
        });

    // Archive Mount stubs (return negative so engine uses loose files on /app0)
    register_symbol(0xF8546F2A49D48F52ULL, "sceFiosArchiveMountWithOrder", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = static_cast<u64>(-1);
        });
    register_symbol(0x186AAE70BF45F982ULL, "sceFiosArchiveMountWithOrderSync", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = static_cast<u64>(-1);
        });
    register_symbol(0xB8292024EAD850BEULL, "sceFiosArchiveGetMountBufferSize", "libSceFios2",
        [](CpuContext& ctx, MemoryManager& /*mem*/) {
            ctx.rax = 0x10000;
        });
}

} // namespace ps4
