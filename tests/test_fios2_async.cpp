// Unit test for the libSceFios2 async open path.
//
// sceFiosFHOpen (async) must return an SceFiosOp id that is never 0 (0 is the
// unregistered-stub default and does not name an operation), and
// sceFiosOpIsDone / sceFiosOpGetError must report that op as complete with the
// right error. sceFiosFHOpenSync keeps returning SCE_OK.

#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/vfs.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

// assert() is compiled out under NDEBUG (Release), and several checks here
// have side effects (initialize, read/write). CHECK always evaluates.
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond "\n"; \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

constexpr ps4::Nid kNidFHOpenAsync = 0x7ABE93910154BE9DULL;     // sceFiosFHOpen
constexpr ps4::Nid kNidFHOpenSync  = 0x6F8E1A9D5D83ECADULL;     // sceFiosFHOpenSync
constexpr ps4::Nid kNidOpIsDone    = 0x6DF828D8EB66AB3DULL;     // sceFiosOpIsDone
constexpr ps4::Nid kNidOpGetError  = 0x5FEEEB21F63DECFBULL;     // sceFiosOpGetError
constexpr ps4::Nid kNidOpDelete    = 0xE5CC8472294EFC9DULL;     // sceFiosOpDelete

constexpr ps4::GuestAddr kPathAddr = 0x400100;
constexpr ps4::GuestAddr kFhAddr   = 0x400200;

void write_guest_string(ps4::MemoryManager& mem, ps4::GuestAddr addr, const char* s) {
    const auto len = std::strlen(s) + 1;
    CHECK(mem.is_valid_address(addr, len));
    std::memcpy(mem.to_host_ptr<char>(addr), s, len);
}

}  // namespace

int main() {
    std::cout << "[FIOS2] Async sceFiosFHOpen op-handle tests...\n";

    ps4::MemoryManager memory;
    CHECK(memory.initialize());

    ps4::SyscallTable syscalls(memory);
    syscalls.register_default_handlers();
    syscalls.set_terminate_on_exit(false);

    ps4::NidRegistry registry;
    registry.register_default_libraries();

    const auto temp_dir = std::filesystem::temp_directory_path() / "ps4_test_fios2_async";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir / "app0");
    {
        std::ofstream f(temp_dir / "app0" / "probe.bin", std::ios::binary);
        const char payload[16] = {0};
        f.write(payload, sizeof(payload));
    }
    ps4::VirtualFileSystem vfs(temp_dir);
    registry.set_vfs(&vfs);

    const auto* open_async = registry.find_by_nid(kNidFHOpenAsync);
    const auto* open_sync  = registry.find_by_nid(kNidFHOpenSync);
    const auto* op_is_done = registry.find_by_nid(kNidOpIsDone);
    const auto* op_error   = registry.find_by_nid(kNidOpGetError);
    const auto* op_delete  = registry.find_by_nid(kNidOpDelete);
    CHECK(open_async && open_sync && op_is_done && op_error && op_delete);
    CHECK(open_async->name == "sceFiosFHOpen");
    CHECK(op_is_done->name == "sceFiosOpIsDone");

    // 1. Successful async open: nonzero op, handle written, op done with SCE_OK.
    {
        write_guest_string(memory, kPathAddr, "/app0/probe.bin");
        CHECK(memory.write_value<ps4::i32>(kFhAddr, 0));

        ps4::CpuContext ctx{};
        ctx.rsi = kFhAddr;
        ctx.rdx = kPathAddr;
        open_async->handler(ctx, memory);
        const auto op = ctx.rax;
        CHECK(op != 0);

        ps4::i32 fh = 0;
        CHECK(memory.read_value(kFhAddr, fh));
        CHECK(fh >= 200);

        ps4::CpuContext done{};
        done.rdi = op;
        op_is_done->handler(done, memory);
        CHECK(done.rax == 1);

        ps4::CpuContext err{};
        err.rdi = op;
        op_error->handler(err, memory);
        CHECK(static_cast<ps4::i64>(err.rax) == 0);

        ps4::CpuContext del{};
        del.rdi = op;
        op_delete->handler(del, memory);
        CHECK(del.rax == ps4::SCE_OK);
        std::cout << "  ok: async open returns op 0x" << std::hex << op << std::dec
                  << ", handle " << fh << ", op reports done\n";
    }

    // 2. Failed async open: still a nonzero op, done, with a negative error.
    {
        write_guest_string(memory, kPathAddr, "/app0/does_not_exist.bin");

        ps4::CpuContext ctx{};
        ctx.rsi = kFhAddr;
        ctx.rdx = kPathAddr;
        open_async->handler(ctx, memory);
        const auto op = ctx.rax;
        CHECK(op != 0);

        ps4::CpuContext done{};
        done.rdi = op;
        op_is_done->handler(done, memory);
        CHECK(done.rax == 1);

        ps4::CpuContext err{};
        err.rdi = op;
        op_error->handler(err, memory);
        CHECK(static_cast<ps4::i64>(err.rax) == -1);
        std::cout << "  ok: failed async open reports done with error -1\n";
    }

    // 3. Sync variant is unchanged: SCE_OK on success.
    {
        write_guest_string(memory, kPathAddr, "/app0/probe.bin");

        ps4::CpuContext ctx{};
        ctx.rsi = kFhAddr;
        ctx.rdx = kPathAddr;
        open_sync->handler(ctx, memory);
        CHECK(ctx.rax == ps4::SCE_OK);
        std::cout << "  ok: sceFiosFHOpenSync still returns SCE_OK\n";
    }

    registry.set_vfs(nullptr);
    std::filesystem::remove_all(temp_dir);
    std::cout << "=== All FIOS2 async open tests passed ===\n";
    return 0;
}
