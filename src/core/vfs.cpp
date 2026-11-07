#include "ps4/vfs.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// struct stat timestamps are st_*timespec on Darwin and st_*tim on Linux (the
// Linux build only backs the QEMU CPU test harness).
#if defined(__APPLE__)
#define VFS_ST_ATIM st_atimespec
#define VFS_ST_MTIM st_mtimespec
#define VFS_ST_CTIM st_ctimespec
#else
#define VFS_ST_ATIM st_atim
#define VFS_ST_MTIM st_mtim
#define VFS_ST_CTIM st_ctim
#endif

namespace ps4 {

namespace {

std::string normalize_guest_path(std::string_view path) {
    std::string norm;
    norm.reserve(path.size());

    // Replace backslashes and collapse consecutive slashes
    bool last_was_slash = false;
    for (char c : path) {
        if (c == '\\' || c == '/') {
            if (!last_was_slash) {
                norm.push_back('/');
                last_was_slash = true;
            }
        } else {
            norm.push_back(c);
            last_was_slash = false;
        }
    }

    if (norm.empty()) {
        norm = "/";
    }
    return norm;
}

int map_open_flags(i32 flags) {
    int host_flags = 0;
    const i32 access_mode = flags & 0x3;
    if (access_mode == ORBIS_O_RDONLY) {
        host_flags |= O_RDONLY;
    } else if (access_mode == ORBIS_O_WRONLY) {
        host_flags |= O_WRONLY;
    } else if (access_mode == ORBIS_O_RDWR) {
        host_flags |= O_RDWR;
    }

    if (flags & ORBIS_O_NONBLOCK) host_flags |= O_NONBLOCK;
    if (flags & ORBIS_O_APPEND)   host_flags |= O_APPEND;
    if (flags & ORBIS_O_CREAT)    host_flags |= O_CREAT;
    if (flags & ORBIS_O_TRUNC)    host_flags |= O_TRUNC;
    if (flags & ORBIS_O_EXCL)     host_flags |= O_EXCL;

    return host_flags;
}

int map_seek_whence(i32 whence) {
    switch (whence) {
        case ORBIS_SEEK_SET: return SEEK_SET;
        case ORBIS_SEEK_CUR: return SEEK_CUR;
        case ORBIS_SEEK_END: return SEEK_END;
        default:             return SEEK_SET;
    }
}

} // namespace

VirtualFileSystem::VirtualFileSystem()
    : VirtualFileSystem(std::filesystem::current_path() / "vfs_root") {}

VirtualFileSystem::VirtualFileSystem(const std::filesystem::path& sandbox_root)
    : m_sandbox_root(std::filesystem::absolute(sandbox_root)) {
    setup_default_mounts();
}

VirtualFileSystem::~VirtualFileSystem() {
    reset();
}

VirtualFileSystem::VirtualFileSystem(VirtualFileSystem&& other) noexcept {
    std::lock_guard lock(other.m_mutex);
    m_sandbox_root = std::move(other.m_sandbox_root);
    m_mounts = std::move(other.m_mounts);
    m_fd_table = std::move(other.m_fd_table);
    m_next_fd = other.m_next_fd;
}

VirtualFileSystem& VirtualFileSystem::operator=(VirtualFileSystem&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_mutex, other.m_mutex);
        reset();
        m_sandbox_root = std::move(other.m_sandbox_root);
        m_mounts = std::move(other.m_mounts);
        m_fd_table = std::move(other.m_fd_table);
        m_next_fd = other.m_next_fd;
    }
    return *this;
}

void VirtualFileSystem::setup_default_mounts() {
    std::error_code ec;
    std::filesystem::create_directories(m_sandbox_root / "app0", ec);
    std::filesystem::create_directories(m_sandbox_root / "data", ec);
    std::filesystem::create_directories(m_sandbox_root / "savedata", ec);

    mount("/app0", m_sandbox_root / "app0", false);
    mount("/data", m_sandbox_root / "data", false);
    mount("/savedata", m_sandbox_root / "savedata", false);
}

bool VirtualFileSystem::mount(std::string_view guest_prefix, const std::filesystem::path& host_path, bool read_only) {
    std::lock_guard lock(m_mutex);

    std::string prefix = normalize_guest_path(guest_prefix);
    if (prefix.back() == '/' && prefix.size() > 1) {
        prefix.pop_back();
    }

    // Check if already mounted, replace if so
    for (auto& m : m_mounts) {
        if (m.guest_prefix == prefix) {
            m.host_path = std::filesystem::absolute(host_path);
            m.read_only = read_only;
            return true;
        }
    }

    m_mounts.push_back(VfsMountPoint{
        .guest_prefix = prefix,
        .host_path = std::filesystem::absolute(host_path),
        .read_only = read_only,
    });

    // Sort mounts by length descending so longer prefixes match first
    std::sort(m_mounts.begin(), m_mounts.end(), [](const auto& a, const auto& b) {
        return a.guest_prefix.length() > b.guest_prefix.length();
    });

    return true;
}

bool VirtualFileSystem::unmount(std::string_view guest_prefix) {
    std::lock_guard lock(m_mutex);

    std::string prefix = normalize_guest_path(guest_prefix);
    if (prefix.back() == '/' && prefix.size() > 1) {
        prefix.pop_back();
    }

    auto it = std::remove_if(m_mounts.begin(), m_mounts.end(), [&](const auto& m) {
        return m.guest_prefix == prefix;
    });

    if (it != m_mounts.end()) {
        m_mounts.erase(it, m_mounts.end());
        return true;
    }
    return false;
}

std::optional<std::filesystem::path> VirtualFileSystem::resolve_path(std::string_view guest_path, bool* out_read_only) const {
    std::lock_guard lock(m_mutex);

    const std::string norm = normalize_guest_path(guest_path);

    for (const auto& m : m_mounts) {
        const std::string& prefix = m.guest_prefix;

        if (norm == prefix || norm.rfind(prefix + "/", 0) == 0) {
            std::string subpath;
            if (norm.length() > prefix.length()) {
                subpath = norm.substr(prefix.length() + 1); // skip trailing slash
            }

            std::filesystem::path resolved = m.host_path;
            if (!subpath.empty()) {
                resolved /= subpath;
            }

            if (out_read_only) {
                *out_read_only = m.read_only;
            }
            return resolved;
        }
    }

    return std::nullopt;
}

i32 VirtualFileSystem::open(std::string_view guest_path, i32 flags, i32 mode) {
    bool is_ro = false;
    auto resolved = resolve_path(guest_path, &is_ro);
    if (!resolved) {
        return -ENOENT; // -2
    }

    const i32 access_mode = flags & 0x3;
    if (is_ro && (access_mode != ORBIS_O_RDONLY || (flags & (ORBIS_O_CREAT | ORBIS_O_TRUNC)))) {
        return -EACCES; // -13
    }

    const int host_flags = map_open_flags(flags);

    // If writing or creating, ensure parent directory exists
    if ((flags & (ORBIS_O_CREAT | ORBIS_O_WRONLY | ORBIS_O_RDWR)) != 0) {
        std::error_code ec;
        std::filesystem::create_directories(resolved->parent_path(), ec);
    }

    const int host_fd = ::open(resolved->c_str(), host_flags, static_cast<mode_t>(mode));
    if (host_fd < 0) {
        return -errno;
    }

    std::lock_guard lock(m_mutex);
    const i32 virt_fd = m_next_fd++;
    m_fd_table[virt_fd] = VfsFileDescriptor{
        .host_fd = host_fd,
        .guest_path = std::string(guest_path),
        .flags = flags,
        .is_open = true,
    };

    return virt_fd;
}

i32 VirtualFileSystem::close(i32 fd) {
    std::lock_guard lock(m_mutex);

    auto it = m_fd_table.find(fd);
    if (it == m_fd_table.end() || !it->second.is_open) {
        return -EBADF; // -9
    }

    const int host_fd = it->second.host_fd;
    m_fd_table.erase(it);

    if (host_fd >= 0) {
        if (::close(host_fd) != 0) {
            return -errno;
        }
    }

    return 0;
}

i64 VirtualFileSystem::read(i32 fd, void* buffer, usize count) {
    if (!buffer) return -EFAULT;

    int host_fd = -1;
    {
        std::lock_guard lock(m_mutex);
        auto it = m_fd_table.find(fd);
        if (it == m_fd_table.end() || !it->second.is_open) {
            return -EBADF;
        }
        host_fd = it->second.host_fd;
    }

    const ssize_t bytes = ::read(host_fd, buffer, count);
    if (bytes < 0) {
        return -errno;
    }
    return bytes;
}

i64 VirtualFileSystem::write(i32 fd, const void* buffer, usize count) {
    if (!buffer) return -EFAULT;

    int host_fd = -1;
    {
        std::lock_guard lock(m_mutex);
        auto it = m_fd_table.find(fd);
        if (it == m_fd_table.end() || !it->second.is_open) {
            return -EBADF;
        }
        host_fd = it->second.host_fd;
    }

    const ssize_t bytes = ::write(host_fd, buffer, count);
    if (bytes < 0) {
        return -errno;
    }
    return bytes;
}

i64 VirtualFileSystem::lseek(i32 fd, i64 offset, i32 whence) {
    int host_fd = -1;
    {
        std::lock_guard lock(m_mutex);
        auto it = m_fd_table.find(fd);
        if (it == m_fd_table.end() || !it->second.is_open) {
            return -EBADF;
        }
        host_fd = it->second.host_fd;
    }

    const int host_whence = map_seek_whence(whence);
    const off_t pos = ::lseek(host_fd, static_cast<off_t>(offset), host_whence);
    if (pos < 0) {
        return -errno;
    }
    return static_cast<i64>(pos);
}

i32 VirtualFileSystem::fstat(i32 fd, OrbisStat& stat_out) {
    int host_fd = -1;
    {
        std::lock_guard lock(m_mutex);
        auto it = m_fd_table.find(fd);
        if (it == m_fd_table.end() || !it->second.is_open) {
            return -EBADF;
        }
        host_fd = it->second.host_fd;
    }

    struct stat st{};
    if (::fstat(host_fd, &st) != 0) {
        return -errno;
    }

    std::memset(&stat_out, 0, sizeof(OrbisStat));
    stat_out.st_dev        = static_cast<u32>(st.st_dev);
    stat_out.st_ino        = static_cast<u32>(st.st_ino);
    stat_out.st_mode       = static_cast<u16>(st.st_mode);
    stat_out.st_nlink      = static_cast<u16>(st.st_nlink);
    stat_out.st_uid        = static_cast<u32>(st.st_uid);
    stat_out.st_gid        = static_cast<u32>(st.st_gid);
    stat_out.st_rdev       = static_cast<u32>(st.st_rdev);
    stat_out.st_size       = static_cast<i64>(st.st_size);
    stat_out.st_blocks     = static_cast<i64>(st.st_blocks);
    stat_out.st_blksize    = static_cast<u32>(st.st_blksize);
    stat_out.st_atime_sec  = static_cast<i64>(st.VFS_ST_ATIM.tv_sec);
    stat_out.st_atimensec  = static_cast<i64>(st.VFS_ST_ATIM.tv_nsec);
    stat_out.st_mtime_sec  = static_cast<i64>(st.VFS_ST_MTIM.tv_sec);
    stat_out.st_mtimensec  = static_cast<i64>(st.VFS_ST_MTIM.tv_nsec);
    stat_out.st_ctime_sec  = static_cast<i64>(st.VFS_ST_CTIM.tv_sec);
    stat_out.st_ctimensec  = static_cast<i64>(st.VFS_ST_CTIM.tv_nsec);

    return 0;
}

i32 VirtualFileSystem::mkdir(std::string_view guest_path, i32 mode) {
    (void)mode;
    bool is_ro = false;
    auto resolved = resolve_path(guest_path, &is_ro);
    if (!resolved) {
        return -ENOENT;
    }
    if (is_ro) {
        return -EACCES;
    }

    std::error_code ec;
    if (std::filesystem::create_directories(*resolved, ec) || !ec) {
        return 0;
    }
    return -EIO;
}

i32 VirtualFileSystem::unlink(std::string_view guest_path) {
    bool is_ro = false;
    auto resolved = resolve_path(guest_path, &is_ro);
    if (!resolved) {
        return -ENOENT;
    }
    if (is_ro) {
        return -EACCES;
    }

    std::error_code ec;
    if (std::filesystem::remove(*resolved, ec)) {
        return 0;
    }
    return ec ? -ENOENT : 0;
}

bool VirtualFileSystem::exists(std::string_view guest_path) const {
    auto resolved = resolve_path(guest_path);
    if (!resolved) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::exists(*resolved, ec);
}

void VirtualFileSystem::reset() {
    std::lock_guard lock(m_mutex);
    for (auto& [fd, desc] : m_fd_table) {
        if (desc.is_open && desc.host_fd >= 0) {
            ::close(desc.host_fd);
            desc.is_open = false;
        }
    }
    m_fd_table.clear();
    m_next_fd = 3;
}

usize VirtualFileSystem::active_fd_count() const {
    std::lock_guard lock(m_mutex);
    return m_fd_table.size();
}

} // namespace ps4
