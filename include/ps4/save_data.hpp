#pragma once

// libSceSaveData guest structures, laid out as the PS4 SDK defines them
// (cross-checked against shadPS4's savedata.cpp). Every pointer member is a
// guest address; the static_asserts pin the offsets the guest code uses.

#include "ps4/common.hpp"
#include <cstddef>

namespace ps4 {

// Error codes (SCE_SAVE_DATA_ERROR_*)
inline constexpr u32 SCE_SAVE_DATA_ERROR_PARAMETER         = 0x809F0000;
inline constexpr u32 SCE_SAVE_DATA_ERROR_NOT_INITIALIZED   = 0x809F0001;
inline constexpr u32 SCE_SAVE_DATA_ERROR_BUSY              = 0x809F0003;
inline constexpr u32 SCE_SAVE_DATA_ERROR_NOT_MOUNTED       = 0x809F0004;
inline constexpr u32 SCE_SAVE_DATA_ERROR_EXISTS            = 0x809F0007;
inline constexpr u32 SCE_SAVE_DATA_ERROR_NOT_FOUND         = 0x809F0008;
inline constexpr u32 SCE_SAVE_DATA_ERROR_INTERNAL          = 0x809F000B;
inline constexpr u32 SCE_SAVE_DATA_ERROR_MOUNT_FULL        = 0x809F000C;
inline constexpr u32 SCE_SAVE_DATA_ERROR_BAD_MOUNTED       = 0x809F000D;
inline constexpr u32 SCE_SAVE_DATA_ERROR_INVALID_LOGIN_USER = 0x809F0011;
inline constexpr u32 SCE_SAVE_DATA_ERROR_MEMORY_NOT_READY  = 0x809F0012;

// SceSaveDataMountMode bits
inline constexpr u32 SCE_SAVE_DATA_MOUNT_MODE_RDONLY       = 1u << 0;
inline constexpr u32 SCE_SAVE_DATA_MOUNT_MODE_RDWR         = 1u << 1;
inline constexpr u32 SCE_SAVE_DATA_MOUNT_MODE_CREATE       = 1u << 2; // fails if it exists
inline constexpr u32 SCE_SAVE_DATA_MOUNT_MODE_DESTRUCT_OFF = 1u << 3;
inline constexpr u32 SCE_SAVE_DATA_MOUNT_MODE_COPY_ICON    = 1u << 4;
inline constexpr u32 SCE_SAVE_DATA_MOUNT_MODE_CREATE2      = 1u << 5; // creates only if missing

inline constexpr u32 SCE_SAVE_DATA_MOUNT_STATUS_NOTHING = 0;
inline constexpr u32 SCE_SAVE_DATA_MOUNT_STATUS_CREATED = 1;

// SceSaveDataParamType
inline constexpr u32 SCE_SAVE_DATA_PARAM_TYPE_ALL        = 0;
inline constexpr u32 SCE_SAVE_DATA_PARAM_TYPE_TITLE      = 1;
inline constexpr u32 SCE_SAVE_DATA_PARAM_TYPE_SUB_TITLE  = 2;
inline constexpr u32 SCE_SAVE_DATA_PARAM_TYPE_DETAIL     = 3;
inline constexpr u32 SCE_SAVE_DATA_PARAM_TYPE_USER_PARAM = 4;
inline constexpr u32 SCE_SAVE_DATA_PARAM_TYPE_MTIME      = 5;

// SceSaveDataSortKey / SceSaveDataSortOrder
inline constexpr u32 SCE_SAVE_DATA_SORT_KEY_DIRNAME     = 0;
inline constexpr u32 SCE_SAVE_DATA_SORT_KEY_USER_PARAM  = 1;
inline constexpr u32 SCE_SAVE_DATA_SORT_KEY_BLOCKS      = 2;
inline constexpr u32 SCE_SAVE_DATA_SORT_KEY_MTIME       = 3;
inline constexpr u32 SCE_SAVE_DATA_SORT_KEY_FREE_BLOCKS = 5;
inline constexpr u32 SCE_SAVE_DATA_SORT_ORDER_ASCENT    = 0;
inline constexpr u32 SCE_SAVE_DATA_SORT_ORDER_DESCENT   = 1;

inline constexpr u64 SCE_SAVE_DATA_BLOCK_SIZE = 32768; // 32 KiB
inline constexpr u64 SCE_SAVE_DATA_BLOCKS_MAX = 32768; // 1 GiB

struct SceSaveDataTitleId {
    char data[10];
    char pad[6];
};
static_assert(sizeof(SceSaveDataTitleId) == 16);

struct SceSaveDataDirName {
    char data[32];
};
static_assert(sizeof(SceSaveDataDirName) == 32);

struct SceSaveDataMountPoint {
    char data[16]; // "/savedataN"
};
static_assert(sizeof(SceSaveDataMountPoint) == 16);

struct SceSaveDataParam {
    char title[128];
    char sub_title[128];
    char detail[1024];
    u32 user_param;
    u32 pad;
    i64 mtime; // time_t
    u8 reserved[32];
};
static_assert(offsetof(SceSaveDataParam, user_param) == 0x500);
static_assert(offsetof(SceSaveDataParam, mtime) == 0x508);
static_assert(sizeof(SceSaveDataParam) == 0x530);

struct SceSaveDataMount {
    i32 user_id;
    u32 pad;
    GuestAddr title_id;    // const SceSaveDataTitleId*, null = this title
    GuestAddr dir_name;    // const SceSaveDataDirName*
    GuestAddr fingerprint; // const SceSaveDataFingerprint*
    u64 blocks;
    u32 mount_mode;
    u8 reserved[32];
};
static_assert(offsetof(SceSaveDataMount, blocks) == 0x20);
static_assert(offsetof(SceSaveDataMount, mount_mode) == 0x28);
static_assert(sizeof(SceSaveDataMount) == 0x50);

struct SceSaveDataMount2 {
    i32 user_id;
    u32 pad;
    GuestAddr dir_name; // const SceSaveDataDirName*
    u64 blocks;
    u32 mount_mode;
    u8 reserved[32];
    u32 pad2;
};
static_assert(offsetof(SceSaveDataMount2, mount_mode) == 0x18);
static_assert(sizeof(SceSaveDataMount2) == 0x40);

struct SceSaveDataMountResult {
    SceSaveDataMountPoint mount_point;
    u64 required_blocks;
    u32 unused;
    u32 mount_status; // SCE_SAVE_DATA_MOUNT_STATUS_*
    u8 reserved[28];
    u32 pad;
};
static_assert(offsetof(SceSaveDataMountResult, required_blocks) == 0x10);
static_assert(offsetof(SceSaveDataMountResult, mount_status) == 0x1C);
static_assert(sizeof(SceSaveDataMountResult) == 0x40);

struct SceSaveDataMountInfo {
    u64 blocks;
    u64 free_blocks;
    u8 reserved[32];
};
static_assert(sizeof(SceSaveDataMountInfo) == 0x30);

struct SceSaveDataDelete {
    i32 user_id;
    u32 pad;
    GuestAddr title_id; // const SceSaveDataTitleId*, null = this title
    GuestAddr dir_name; // const SceSaveDataDirName*
    u32 unused;
    u8 reserved[32];
    u32 pad2;
};
static_assert(sizeof(SceSaveDataDelete) == 0x40);

struct SceSaveDataDirNameSearchCond {
    i32 user_id;
    u32 pad;
    GuestAddr title_id; // const SceSaveDataTitleId*, null = this title
    GuestAddr dir_name; // const SceSaveDataDirName*: pattern, '%' = any run, '_' = one char
    u32 key;            // SCE_SAVE_DATA_SORT_KEY_*
    u32 order;          // SCE_SAVE_DATA_SORT_ORDER_*
    u8 reserved[32];
};
static_assert(offsetof(SceSaveDataDirNameSearchCond, key) == 0x18);
static_assert(sizeof(SceSaveDataDirNameSearchCond) == 0x40);

struct SceSaveDataSearchInfo {
    u64 blocks;
    u64 free_blocks;
    u8 reserved[32];
};
static_assert(sizeof(SceSaveDataSearchInfo) == 0x30);

struct SceSaveDataDirNameSearchResult {
    u32 hit_num;           // matches on disk
    u32 pad;
    GuestAddr dir_names;   // SceSaveDataDirName[dir_names_num]
    u32 dir_names_num;     // capacity of dir_names (input)
    u32 set_num;           // entries written
    GuestAddr params;      // SceSaveDataParam[dir_names_num], may be null
    GuestAddr infos;       // SceSaveDataSearchInfo[dir_names_num], may be null
    u8 reserved[12];
    u32 pad2;
};
static_assert(offsetof(SceSaveDataDirNameSearchResult, dir_names_num) == 0x10);
static_assert(offsetof(SceSaveDataDirNameSearchResult, params) == 0x18);
static_assert(offsetof(SceSaveDataDirNameSearchResult, infos) == 0x20);
static_assert(sizeof(SceSaveDataDirNameSearchResult) == 0x38);

struct SceSaveDataIcon {
    GuestAddr buf;
    u64 buf_size;
    u64 data_size;
    u8 reserved[32];
};
static_assert(sizeof(SceSaveDataIcon) == 0x38);

struct SceSaveDataEvent {
    u32 type;
    i32 error_code;
    i32 user_id;
    u32 pad;
    SceSaveDataTitleId title_id;
    SceSaveDataDirName dir_name;
    u8 reserved[40];
};
static_assert(sizeof(SceSaveDataEvent) == 0x68);

struct SceSaveDataMemoryData {
    GuestAddr buf;
    u64 buf_size;
    i64 offset;
    u8 reserved[40];
};
static_assert(sizeof(SceSaveDataMemoryData) == 0x40);

struct SceSaveDataMemoryGet2 {
    i32 user_id;
    u32 pad;
    GuestAddr data;  // SceSaveDataMemoryData*
    GuestAddr param; // SceSaveDataParam*
    GuestAddr icon;  // SceSaveDataIcon*
    u32 slot_id;
    u8 reserved[28];
};
static_assert(offsetof(SceSaveDataMemoryGet2, slot_id) == 0x20);
static_assert(sizeof(SceSaveDataMemoryGet2) == 0x40);

struct SceSaveDataMemorySet2 {
    i32 user_id;
    u32 pad;
    GuestAddr data;  // const SceSaveDataMemoryData[data_num]
    GuestAddr param; // const SceSaveDataParam*
    GuestAddr icon;  // const SceSaveDataIcon*
    u32 data_num;
    u32 slot_id;
    u8 reserved[32];
};
static_assert(offsetof(SceSaveDataMemorySet2, data_num) == 0x20);
static_assert(sizeof(SceSaveDataMemorySet2) == 0x48);

struct SceSaveDataMemorySetup2 {
    u32 option;
    i32 user_id;
    u64 memory_size;
    u64 icon_memory_size;
    GuestAddr init_param; // const SceSaveDataParam*
    GuestAddr init_icon;  // const SceSaveDataIcon*
    u32 slot_id;
    u8 reserved[20];
};
static_assert(offsetof(SceSaveDataMemorySetup2, memory_size) == 0x08);
static_assert(offsetof(SceSaveDataMemorySetup2, slot_id) == 0x28);
static_assert(sizeof(SceSaveDataMemorySetup2) == 0x40);

struct SceSaveDataMemorySetupResult {
    u64 existed_memory_size; // 0 when the save memory was just created
    u8 reserved[16];
};
static_assert(sizeof(SceSaveDataMemorySetupResult) == 0x18);

} // namespace ps4
