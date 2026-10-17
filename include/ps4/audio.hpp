#pragma once

#include "ps4/common.hpp"

namespace ps4 {

// PS4 Audio Out Port Types
inline constexpr i32 SCE_AUDIO_OUT_PORT_TYPE_MAIN   = 0;
inline constexpr i32 SCE_AUDIO_OUT_PORT_TYPE_BGM    = 1;
inline constexpr i32 SCE_AUDIO_OUT_PORT_TYPE_VOICE  = 2;
inline constexpr i32 SCE_AUDIO_OUT_PORT_TYPE_PADSPK = 3;

// PS4 Audio Out Formats
inline constexpr i32 SCE_AUDIO_OUT_PARAM_FORMAT_S16_STEREO   = 0;
inline constexpr i32 SCE_AUDIO_OUT_PARAM_FORMAT_S16_8CH      = 1;
inline constexpr i32 SCE_AUDIO_OUT_PARAM_FORMAT_FLOAT_STEREO = 2;
inline constexpr i32 SCE_AUDIO_OUT_PARAM_FORMAT_FLOAT_8CH    = 3;

// Max Volume Level (0x8000 = unity gain / 1.0f)
inline constexpr i32 SCE_AUDIO_OUT_VOLUME_FLAG_L_CH = 0x01;
inline constexpr i32 SCE_AUDIO_OUT_VOLUME_FLAG_R_CH = 0x02;
inline constexpr i32 SCE_AUDIO_OUT_VOLUME_MAX       = 0x8000;

struct AudioPortConfig {
    u32 sample_rate{48000};
    u32 channels{2};
    i32 format{SCE_AUDIO_OUT_PARAM_FORMAT_S16_STEREO};
    f32 volume_left{1.0f};
    f32 volume_right{1.0f};
    u32 grain{256};
};

struct SceAudioOutPortInfo {
    i32 status{0};
    i32 volume_left{SCE_AUDIO_OUT_VOLUME_MAX};
    i32 volume_right{SCE_AUDIO_OUT_VOLUME_MAX};
    u32 queued_samples{0};
};

} // namespace ps4
