#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudio/CoreAudio.h>
#include "ps4/coreaudio_backend.hpp"
#include <iostream>

namespace ps4 {

namespace {

static void AudioQueueCallback(void* userData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer) {
    auto* backend = static_cast<CoreAudioBackend*>(userData);
    if (!backend || !backend->is_running()) return;

    const usize sampleCount = inBuffer->mAudioDataBytesCapacity / sizeof(i16);
    std::span<i16> out_span(static_cast<i16*>(inBuffer->mAudioData), sampleCount);
    backend->process_audio_buffer(out_span);
    inBuffer->mAudioDataByteSize = inBuffer->mAudioDataBytesCapacity;

    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, nullptr);
}

} // namespace

struct CoreAudioBackend::Impl {
    AudioOutManager& manager;
    i32 port_handle{1};
    bool is_running{false};
    bool headless{false};

    AudioQueueRef audio_queue{nullptr};
    AudioQueueBufferRef audio_buffers[3]{nullptr, nullptr, nullptr};
    AudioStreamBasicDescription asbd{};

    explicit Impl(AudioOutManager& mgr, i32 handle)
        : manager(mgr), port_handle(handle) {}
};

CoreAudioBackend::CoreAudioBackend(AudioOutManager& manager, i32 port_handle)
    : m_impl(std::make_unique<Impl>(manager, port_handle)) {}

CoreAudioBackend::~CoreAudioBackend() {
    stop();
    if (m_impl && m_impl->audio_queue) {
        AudioQueueDispose(m_impl->audio_queue, true);
        m_impl->audio_queue = nullptr;
    }
}

bool CoreAudioBackend::initialize(bool headless) {
    if (!m_impl) return false;
    m_impl->headless = headless;

    if (headless) {
        return true;
    }

    // Configure 48kHz, Stereo, 16-bit Signed Integer Linear PCM
    m_impl->asbd.mSampleRate = 48000.0;
    m_impl->asbd.mFormatID = kAudioFormatLinearPCM;
    m_impl->asbd.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    m_impl->asbd.mChannelsPerFrame = 2;
    m_impl->asbd.mBitsPerChannel = 16;
    m_impl->asbd.mBytesPerFrame = 4;
    m_impl->asbd.mFramesPerPacket = 1;
    m_impl->asbd.mBytesPerPacket = 4;

    OSStatus status = AudioQueueNewOutput(&m_impl->asbd,
                                          AudioQueueCallback,
                                          this,
                                          nullptr,
                                          nullptr,
                                          0,
                                          &m_impl->audio_queue);
    if (status != noErr) {
        std::cerr << "[CoreAudio] Failed to initialize AudioQueue (status: " << status
                  << "). Falling back to headless audio mode.\n";
        m_impl->headless = true;
        return true;
    }

    // Allocate 3 buffers of 512 stereo samples (2048 bytes each)
    constexpr UInt32 bufferSize = 2048;
    for (int i = 0; i < 3; ++i) {
        status = AudioQueueAllocateBuffer(m_impl->audio_queue, bufferSize, &m_impl->audio_buffers[i]);
        if (status != noErr) {
            AudioQueueDispose(m_impl->audio_queue, true);
            m_impl->audio_queue = nullptr;
            m_impl->headless = true;
            return true;
        }

        std::memset(m_impl->audio_buffers[i]->mAudioData, 0, bufferSize);
        m_impl->audio_buffers[i]->mAudioDataByteSize = bufferSize;
        AudioQueueEnqueueBuffer(m_impl->audio_queue, m_impl->audio_buffers[i], 0, nullptr);
    }

    return true;
}

bool CoreAudioBackend::start() {
    if (!m_impl) return false;
    if (m_impl->is_running) return true;

    if (m_impl->headless || !m_impl->audio_queue) {
        m_impl->is_running = true;
        return true;
    }

    OSStatus status = AudioQueueStart(m_impl->audio_queue, nullptr);
    if (status == noErr) {
        m_impl->is_running = true;
        return true;
    }
    return false;
}

bool CoreAudioBackend::stop() {
    if (!m_impl) return false;
    if (!m_impl->is_running) return true;

    if (m_impl->audio_queue) {
        AudioQueueStop(m_impl->audio_queue, true);
    }
    m_impl->is_running = false;
    return true;
}

bool CoreAudioBackend::is_running() const noexcept {
    return m_impl && m_impl->is_running;
}

usize CoreAudioBackend::process_audio_buffer(std::span<i16> out_buffer) {
    if (!m_impl) return 0;
    return m_impl->manager.read_samples(m_impl->port_handle, out_buffer);
}

bool CoreAudioBackend::is_headless() const noexcept {
    return m_impl && m_impl->headless;
}

} // namespace ps4
