#pragma once

#include "ps4/common.hpp"
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ps4 {

// Orbis / FreeBSD standard open flags
inline constexpr i32 ORBIS_O_RDONLY   = 0x0000;
inline constexpr i32 ORBIS_O_WRONLY   = 0x0001;
inline constexpr i32 ORBIS_O_RDWR     = 0x0002;
inline constexpr i32 ORBIS_O_NONBLOCK = 0x0004;
inline constexpr i32 ORBIS_O_APPEND   = 0x0008;
inline constexpr i32 ORBIS_O_CREAT    = 0x0200;
inline constexpr i32 ORBIS_O_TRUNC    = 0x0400;
inline constexpr i32 ORBIS_O_EXCL     = 0x0800;

// Orbis / FreeBSD seek whence
inline constexpr i32 ORBIS_SEEK_SET = 0;
inline constexpr i32 ORBIS_SEEK_CUR = 1;
inline constexpr i32 ORBIS_SEEK_END = 2;

// Orbis / FreeBSD stat structure layout (packed/aligned for guest memory)
#pragma pack(push, 4)
struct OrbisStat {
    u32 st_dev;           // Device ID
    u32 st_ino;           // Inode number
    u16 st_mode;          // Mode/permissions
    u16 st_nlink;         // Hard links
    u32 st_uid;           // User ID
    u32 st_gid;           // Group ID
    u32 st_rdev;          // Device type for special files
    u32 _pad0;
    i64 st_atime_sec;     // Access time (seconds)
    i64 st_atimensec;     // Access time (nanoseconds)
    i64 st_mtime_sec;     // Modification time (seconds)
    i64 st_mtimensec;     // Modification time (nanoseconds)
    i64 st_ctime_sec;     // Status change time (seconds)
    i64 st_ctimensec;     // Status change time (nanoseconds)
    i64 st_size;          // File size in bytes
    i64 st_blocks;        // Number of 512B blocks allocated
    u32 st_blksize;       // Optimal block size for I/O
    u32 st_flags;         // User defined flags
    u32 st_gen;           // File generation number
    u32 _pad1;
};
#pragma pack(pop)

// Save data structures
inline constexpr i32 SCE_SAVE_DATA_MOUNT_MODE_RDONLY = 1;
inline constexpr i32 SCE_SAVE_DATA_MOUNT_MODE_RDWR   = 2;
inline constexpr i32 SCE_SAVE_DATA_MOUNT_MODE_CREATE = 4;

#pragma pack(push, 1)
struct SceSaveDataDirName {
    char data[32];
};

struct SceSaveDataMount {
    u32 user_id;
    SceSaveDataDirName dir_name;
    GuestAddr fingerprint;
    u32 blocks;
    u32 mount_mode;
    u8 reserved[32];
};

struct SceSaveDataMountResult {
    char mount_path[16];
    u32 required_blocks;
    u8 reserved[32];
};
#pragma pack(pop)

struct VfsMountPoint {
    std::string guest_prefix;
    std::filesystem::path host_path;
    bool read_only{false};
};

struct VfsFileDescriptor {
    int host_fd{-1};
    std::string guest_path;
    int flags{0};
    bool is_open{false};
};

class VirtualFileSystem {
public:
    VirtualFileSystem();
    explicit VirtualFileSystem(const std::filesystem::path& sandbox_root);
    ~VirtualFileSystem();

    // Prevent copy, allow move
    VirtualFileSystem(const VirtualFileSystem&) = delete;
    VirtualFileSystem& operator=(const VirtualFileSystem&) = delete;
    VirtualFileSystem(VirtualFileSystem&&) noexcept;
    VirtualFileSystem& operator=(VirtualFileSystem&&) noexcept;

    /// Mount a guest directory prefix to a host filesystem path
    bool mount(std::string_view guest_prefix, const std::filesystem::path& host_path, bool read_only = false);

    /// Unmount a guest directory prefix
    bool unmount(std::string_view guest_prefix);

    /// Resolve a guest path (e.g. "/app0/data/config.bin") to host path
    std::optional<std::filesystem::path> resolve_path(std::string_view guest_path, bool* out_read_only = nullptr) const;

    /// Open a virtual file descriptor
    i32 open(std::string_view guest_path, i32 flags, i32 mode = 0644);

    /// Close a virtual file descriptor
    i32 close(i32 fd);

    /// Read bytes from virtual file descriptor
    i64 read(i32 fd, void* buffer, usize count);

    /// Write bytes to virtual file descriptor
    i64 write(i32 fd, const void* buffer, usize count);

    /// Reposition read/write file offset
    i64 lseek(i32 fd, i64 offset, i32 whence);

    /// Get file status by descriptor
    i32 fstat(i32 fd, OrbisStat& stat_out);

    /// Create directory
    i32 mkdir(std::string_view guest_path, i32 mode = 0755);

    /// Remove file
    i32 unlink(std::string_view guest_path);

    /// Check if guest path exists
    bool exists(std::string_view guest_path) const;

    /// Mount a save data directory
    i32 mount_savedata(u32 user_id, const std::string& dir_name, u32 mount_mode, SceSaveDataMountResult& out_result);

    /// Unmount a save data directory
    i32 unmount_savedata(const std::string& mount_path);

    /// Get current sandbox root
    [[nodiscard]] const std::filesystem::path& sandbox_root() const noexcept { return m_sandbox_root; }

    /// Reset and close all open file descriptors
    void reset();

    /// Get active descriptor count
    [[nodiscard]] usize active_fd_count() const;

private:
    std::filesystem::path m_sandbox_root;
    std::vector<VfsMountPoint> m_mounts;
    std::unordered_map<i32, VfsFileDescriptor> m_fd_table;
    i32 m_next_fd{3}; // 0, 1, 2 reserved for stdio
    i32 m_next_savedata_mount_idx{0};
    mutable std::mutex m_mutex;

    void setup_default_mounts();
};

} // namespace ps4
