#pragma once

#include "ps4/audio.hpp"
#include "ps4/common.hpp"
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace ps4 {

struct AudioPort {
    i32 handle{0};
    i32 type{SCE_AUDIO_OUT_PORT_TYPE_MAIN};
    AudioPortConfig config{};
    std::mutex mutex;
    std::deque<i16> sample_queue; // 16-bit interleaved PCM samples
    u64 total_samples_submitted{0};
    u64 total_samples_read{0};
};

class AudioOutManager {
public:
    AudioOutManager();
    ~AudioOutManager();

    // Non-copyable
    AudioOutManager(const AudioOutManager&) = delete;
    AudioOutManager& operator=(const AudioOutManager&) = delete;

    /// Open a new audio out port
    i32 open_port(i32 type, const AudioPortConfig& config = {});

    /// Close an open audio port
    bool close_port(i32 handle);

    /// Submit 16-bit interleaved PCM audio samples
    bool output_samples(i32 handle, std::span<const i16> samples);

    /// Submit 32-bit float interleaved PCM audio samples (converted to 16-bit)
    bool output_samples_float(i32 handle, std::span<const f32> samples);

    /// Set volume for left and right channels (0.0f - 1.0f)
    bool set_volume(i32 handle, f32 left, f32 right);

    /// Read samples from port queue (used by CoreAudio render callback)
    usize read_samples(i32 handle, std::span<i16> out_buffer);

    /// Get current queued sample count for a port
    usize get_queued_sample_count(i32 handle);

    /// Get port information
    bool get_port_info(i32 handle, SceAudioOutPortInfo& out_info);

    /// Check if port handle is valid and open
    bool is_port_open(i32 handle);

    /// Active port count
    usize active_port_count();

private:
    std::mutex m_manager_mutex;
    i32 m_next_handle{1};
    std::unordered_map<i32, std::unique_ptr<AudioPort>> m_ports;
};

} // namespace ps4
