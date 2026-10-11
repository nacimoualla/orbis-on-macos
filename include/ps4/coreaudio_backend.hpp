#pragma once

#include "ps4/audio_manager.hpp"
#include "ps4/common.hpp"
#include <memory>
#include <span>

namespace ps4 {

class CoreAudioBackend {
public:
    explicit CoreAudioBackend(AudioOutManager& manager, i32 port_handle = 1);
    ~CoreAudioBackend();

    // Non-copyable
    CoreAudioBackend(const CoreAudioBackend&) = delete;
    CoreAudioBackend& operator=(const CoreAudioBackend&) = delete;

    /// Initialize CoreAudio AudioQueue with 48kHz Stereo 16-bit PCM format
    bool initialize(bool headless = false);

    /// Start audio playback
    bool start();

    /// Stop audio playback
    bool stop();

    /// Check if audio queue is running
    [[nodiscard]] bool is_running() const noexcept;

    /// Read next buffer of audio samples from manager (used by AudioQueue or mock)
    usize process_audio_buffer(std::span<i16> out_buffer);

    /// Check if in headless mode
    [[nodiscard]] bool is_headless() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ps4
