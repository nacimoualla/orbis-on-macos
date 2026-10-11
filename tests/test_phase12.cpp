#include "ps4/dynamic_linker.hpp"
#include "ps4/memory.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/syscall_table.hpp"
#include "ps4/vfs.hpp"
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace ps4;

void test_vfs_path_resolution() {
    std::cout << "[Test 12.1] Testing VFS mount and path resolution...\n";

    const auto temp_dir = std::filesystem::temp_directory_path() / "ps4_test_vfs_1";
    std::filesystem::remove_all(temp_dir);

    VirtualFileSystem vfs(temp_dir);

    bool is_ro = false;
    auto res_app0 = vfs.resolve_path("/app0/title/param.sfo", &is_ro);
    assert(res_app0.has_value());
    assert(res_app0->string().find("app0") != std::string::npos);
    assert(!is_ro);

    auto res_data = vfs.resolve_path("/data/cache.bin");
    assert(res_data.has_value());
    assert(res_data->string().find("data") != std::string::npos);

    // Custom mount
    const auto custom_host = temp_dir / "custom_dir";
    std::filesystem::create_directories(custom_host);
    assert(vfs.mount("/custom", custom_host, true));

    auto res_custom = vfs.resolve_path("/custom/asset.png", &is_ro);
    assert(res_custom.has_value());
    assert(is_ro == true);

    // Unmount
    assert(vfs.unmount("/custom"));
    auto res_unmounted = vfs.resolve_path("/custom/asset.png");
    assert(!res_unmounted.has_value());

    std::filesystem::remove_all(temp_dir);
    std::cout << "  -> VFS path resolution passed!\n";
}

void test_vfs_file_io() {
    std::cout << "[Test 12.2] Testing VFS file descriptor I/O (open, write, seek, read, fstat, close)...\n";

    const auto temp_dir = std::filesystem::temp_directory_path() / "ps4_test_vfs_2";
    std::filesystem::remove_all(temp_dir);

    VirtualFileSystem vfs(temp_dir);

    // Open new file for write
    const i32 fd = vfs.open("/data/test_file.txt", ORBIS_O_CREAT | ORBIS_O_RDWR | ORBIS_O_TRUNC, 0644);
    assert(fd >= 3);
    assert(vfs.active_fd_count() == 1);

    const std::string text = "PlayStation 4 Virtual File System on Apple Silicon!";
    const i64 written = vfs.write(fd, text.data(), text.size());
    assert(written == static_cast<i64>(text.size()));

    // Seek to beginning
    const i64 pos = vfs.lseek(fd, 0, ORBIS_SEEK_SET);
    assert(pos == 0);

    // Stat file
    OrbisStat st{};
    assert(vfs.fstat(fd, st) == 0);
    assert(st.st_size == static_cast<i64>(text.size()));

    // Read back content
    char read_buf[128]{0};
    const i64 bytes_read = vfs.read(fd, read_buf, sizeof(read_buf));
    assert(bytes_read == static_cast<i64>(text.size()));
    assert(std::string(read_buf) == text);

    // Close
    assert(vfs.close(fd) == 0);
    assert(vfs.active_fd_count() == 0);

    // Check file exists on host
    assert(vfs.exists("/data/test_file.txt"));

    // Remove
    assert(vfs.unlink("/data/test_file.txt") == 0);
    assert(!vfs.exists("/data/test_file.txt"));

    std::filesystem::remove_all(temp_dir);
    std::cout << "  -> VFS file descriptor I/O passed!\n";
}

void test_vfs_readonly_protection() {
    std::cout << "[Test 12.3] Testing VFS read-only mount enforcement...\n";

    const auto temp_dir = std::filesystem::temp_directory_path() / "ps4_test_vfs_ro";
    std::filesystem::remove_all(temp_dir);

    VirtualFileSystem vfs(temp_dir);
    const auto ro_host = temp_dir / "ro_assets";
    std::filesystem::create_directories(ro_host);

    // Create a file in host directly
    {
        std::ofstream f(ro_host / "config.ini");
        f << "key=value\n";
    }

    assert(vfs.mount("/ro_mount", ro_host, /*read_only=*/true));

    // Open for reading should succeed
    const i32 fd_ro = vfs.open("/ro_mount/config.ini", ORBIS_O_RDONLY);
    assert(fd_ro >= 3);
    char buf[16]{0};
    assert(vfs.read(fd_ro, buf, sizeof(buf)) > 0);
    assert(vfs.close(fd_ro) == 0);

    // Open for writing should fail with -EACCES (-13)
    const i32 fd_wr = vfs.open("/ro_mount/config.ini", ORBIS_O_WRONLY);
    assert(fd_wr == -13);

    const i32 fd_creat = vfs.open("/ro_mount/new_file.dat", ORBIS_O_CREAT | ORBIS_O_WRONLY);
    assert(fd_creat == -13);

    std::filesystem::remove_all(temp_dir);
    std::cout << "  -> Read-only enforcement passed!\n";
}

void test_vfs_syscall_integration() {
    std::cout << "[Test 12.4] Testing FreeBSD syscall table integration with VFS...\n";

    MemoryManager memory;
    assert(memory.initialize());

    const auto temp_dir = std::filesystem::temp_directory_path() / "ps4_test_vfs_sys";
    std::filesystem::remove_all(temp_dir);

    VirtualFileSystem vfs(temp_dir);
    SyscallTable syscalls(memory);
    syscalls.set_vfs(&vfs);
    syscalls.register_default_handlers();

    // Place path string into guest memory
    const GuestAddr path_addr = 0x20000;
    const std::string path_str = "/data/save_state.bin";
    memory.write_bytes(path_addr, path_str.c_str(), path_str.size() + 1);

    // 1. SYS_open (5)
    SyscallContext ctx_open(memory, &vfs);
    ctx_open.args[0] = path_addr;
    ctx_open.args[1] = static_cast<u64>(ORBIS_O_CREAT | ORBIS_O_RDWR | ORBIS_O_TRUNC);
    ctx_open.args[2] = 0644;
    assert(syscalls.dispatch(SYS_open, ctx_open) == 0);
    const i32 fd = static_cast<i32>(ctx_open.rax);
    assert(fd >= 3);

    // 2. SYS_write (4)
    const GuestAddr data_addr = 0x21000;
    const u32 sample_data[4] = {0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x87654321};
    memory.write_bytes(data_addr, sample_data, sizeof(sample_data));

    SyscallContext ctx_write(memory, &vfs);
    ctx_write.args[0] = static_cast<u64>(fd);
    ctx_write.args[1] = data_addr;
    ctx_write.args[2] = sizeof(sample_data);
    assert(syscalls.dispatch(SYS_write, ctx_write) == 0);
    assert(ctx_write.rax == sizeof(sample_data));

    // 3. SYS_lseek (478)
    SyscallContext ctx_lseek(memory, &vfs);
    ctx_lseek.args[0] = static_cast<u64>(fd);
    ctx_lseek.args[1] = 0;
    ctx_lseek.args[2] = ORBIS_SEEK_SET;
    assert(syscalls.dispatch(SYS_lseek, ctx_lseek) == 0);
    assert(ctx_lseek.rax == 0);

    // 4. SYS_fstat (189)
    const GuestAddr stat_addr = 0x22000;
    SyscallContext ctx_fstat(memory, &vfs);
    ctx_fstat.args[0] = static_cast<u64>(fd);
    ctx_fstat.args[1] = stat_addr;
    assert(syscalls.dispatch(SYS_fstat, ctx_fstat) == 0);
    OrbisStat st_read{};
    memory.read_bytes(stat_addr, &st_read, sizeof(st_read));
    assert(st_read.st_size == sizeof(sample_data));

    // 5. SYS_read (3)
    const GuestAddr read_dst = 0x23000;
    SyscallContext ctx_read(memory, &vfs);
    ctx_read.args[0] = static_cast<u64>(fd);
    ctx_read.args[1] = read_dst;
    ctx_read.args[2] = sizeof(sample_data);
    assert(syscalls.dispatch(SYS_read, ctx_read) == 0);
    assert(ctx_read.rax == sizeof(sample_data));

    u32 read_data[4]{0};
    memory.read_bytes(read_dst, read_data, sizeof(read_data));
    assert(std::memcmp(sample_data, read_data, sizeof(sample_data)) == 0);

    // 6. SYS_close (6)
    SyscallContext ctx_close(memory, &vfs);
    ctx_close.args[0] = static_cast<u64>(fd);
    assert(syscalls.dispatch(SYS_close, ctx_close) == 0);

    std::filesystem::remove_all(temp_dir);
    std::cout << "  -> FreeBSD syscall VFS dispatch passed!\n";
}

void test_savedata_hle() {
    std::cout << "[Test 12.5] Testing sceSaveDataMount & sceSaveDataUmount HLE...\n";

    MemoryManager memory;
    assert(memory.initialize());

    NidRegistry nid_reg;
    nid_reg.register_default_libraries();

    const auto* mount_sym = nid_reg.find_by_name("sceSaveDataMount");
    assert(mount_sym != nullptr);

    const auto* umount_sym = nid_reg.find_by_name("sceSaveDataUmount");
    assert(umount_sym != nullptr);

    // Prepare SceSaveDataMount structure in guest memory
    const GuestAddr mount_req_addr = 0x30000;
    const GuestAddr mount_res_addr = 0x31000;

    SceSaveDataMount req{};
    req.user_id = 0x10000001;
    std::strncpy(req.dir_name.data, "PROFILE00", sizeof(req.dir_name.data) - 1);
    req.mount_mode = SCE_SAVE_DATA_MOUNT_MODE_CREATE | SCE_SAVE_DATA_MOUNT_MODE_RDWR;
    memory.write_bytes(mount_req_addr, &req, sizeof(req));

    CpuContext ctx{};
    ctx.rdi = mount_req_addr;
    ctx.rsi = mount_res_addr;

    // Call sceSaveDataMount
    mount_sym->handler(ctx, memory);
    assert(ctx.rax == SCE_OK);

    SceSaveDataMountResult res{};
    memory.read_bytes(mount_res_addr, &res, sizeof(res));
    assert(std::string(res.mount_path).find("/savedata") != std::string::npos);
    assert(res.required_blocks > 0);

    // Call sceSaveDataUmount
    ctx.rdi = mount_res_addr;
    umount_sym->handler(ctx, memory);
    assert(ctx.rax == SCE_OK);

    std::cout << "  -> Save data HLE mounting and unmounting passed!\n";
}

int main() {
    std::cout << "=== Phase 12: Virtual File System & Save Data Subsystem Tests ===\n";

    test_vfs_path_resolution();
    test_vfs_file_io();
    test_vfs_readonly_protection();
    test_vfs_syscall_integration();
    test_savedata_hle();

    std::cout << "=== All Phase 12 tests passed successfully! ===\n";
    return 0;
}
