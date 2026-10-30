#pragma once

// libSceNpTrophy guest structures, laid out as the PS4 SDK defines them
// (cross-checked against shadPS4's np_trophy.h).

#include "ps4/common.hpp"
#include <cstddef>

namespace ps4 {

inline constexpr u32 SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT         = 0x80551604;
inline constexpr u32 SCE_NP_TROPHY_ERROR_INVALID_HANDLE           = 0x80551608;
inline constexpr u32 SCE_NP_TROPHY_ERROR_INVALID_CONTEXT          = 0x80551609;
inline constexpr u32 SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID        = 0x8055160A;
inline constexpr u32 SCE_NP_TROPHY_ERROR_INVALID_GROUP_ID         = 0x8055160B;
inline constexpr u32 SCE_NP_TROPHY_ERROR_TROPHY_ALREADY_UNLOCKED  = 0x8055160C;
inline constexpr u32 SCE_NP_TROPHY_ERROR_NOT_REGISTERED           = 0x8055160F;
inline constexpr u32 SCE_NP_TROPHY_ERROR_CONTEXT_ALREADY_EXISTS   = 0x80551613;
inline constexpr u32 SCE_NP_TROPHY_ERROR_CONTEXT_EXCEEDS_MAX      = 0x80551623;
inline constexpr u32 SCE_NP_TROPHY_ERROR_HANDLE_EXCEEDS_MAX       = 0x80551624;

inline constexpr i32 SCE_NP_TROPHY_INVALID_HANDLE    = -1;
inline constexpr i32 SCE_NP_TROPHY_INVALID_CONTEXT   = -1;
inline constexpr i32 SCE_NP_TROPHY_INVALID_TROPHY_ID = -1;
inline constexpr i32 SCE_NP_TROPHY_BASE_GAME_GROUP_ID = -1;
inline constexpr i32 SCE_NP_TROPHY_NUM_MAX = 128;

struct SceNpTrophyFlagArray {
    u32 flag_bits[SCE_NP_TROPHY_NUM_MAX / 32]; // bit (id & 31) of word (id >> 5)
};
static_assert(sizeof(SceNpTrophyFlagArray) == 16);

struct SceNpTrophyData {
    u64 size;
    i32 trophy_id;
    u8 unlocked;
    u8 reserved[3];
    u64 timestamp; // SceRtcTick: microseconds since 0001-01-01
};
static_assert(offsetof(SceNpTrophyData, timestamp) == 0x10);
static_assert(sizeof(SceNpTrophyData) == 0x18);

struct SceNpTrophyDetails {
    u64 size;
    i32 trophy_id;
    i32 trophy_grade;
    i32 group_id;
    u8 hidden;
    u8 reserved[3];
    char name[128];
    char description[1024];
};
static_assert(offsetof(SceNpTrophyDetails, name) == 0x18);
static_assert(sizeof(SceNpTrophyDetails) == 0x498);

struct SceNpTrophyGameData {
    u64 size;
    u32 unlocked_trophies;
    u32 unlocked_platinum;
    u32 unlocked_gold;
    u32 unlocked_silver;
    u32 unlocked_bronze;
    u32 progress_percentage;
};
static_assert(sizeof(SceNpTrophyGameData) == 0x20);

struct SceNpTrophyGameDetails {
    u64 size;
    u32 num_groups;
    u32 num_trophies;
    u32 num_platinum;
    u32 num_gold;
    u32 num_silver;
    u32 num_bronze;
    char title[128];
    char description[1024];
};
static_assert(offsetof(SceNpTrophyGameDetails, title) == 0x20);
static_assert(sizeof(SceNpTrophyGameDetails) == 0x4A0);

struct SceNpTrophyGroupData {
    u64 size;
    i32 group_id;
    u32 unlocked_trophies;
    u32 unlocked_platinum;
    u32 unlocked_gold;
    u32 unlocked_silver;
    u32 unlocked_bronze;
    u32 progress_percentage;
    u8 reserved[4];
};
static_assert(sizeof(SceNpTrophyGroupData) == 0x28);

struct SceNpTrophyGroupDetails {
    u64 size;
    i32 group_id;
    u32 num_trophies;
    u32 num_platinum;
    u32 num_gold;
    u32 num_silver;
    u32 num_bronze;
    char title[128];
    char description[1024];
};
static_assert(offsetof(SceNpTrophyGroupDetails, title) == 0x20);
static_assert(sizeof(SceNpTrophyGroupDetails) == 0x4A0);

} // namespace ps4
